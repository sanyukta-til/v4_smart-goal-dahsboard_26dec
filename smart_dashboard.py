import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path
from functools import lru_cache
from itertools import product, combinations
from config import get_output_file, DASHBOARD_TITLE, DASHBOARD_PORT, CACHE_TTL, SMART_COLUMNS, HIGH_QUALITY_THRESHOLD, MEDIUM_QUALITY_THRESHOLD, LOW_QUALITY_THRESHOLD
from smart_rules import (
    EMPTY_METRIC_VALUES,
    MEASURABLE_QUANTIFIERS,
    SPECIFIC_DELIVERABLE_TERMS,
    TIME_KEYWORDS,
    TIME_REGEX_PATTERNS,
    has_countable_number,
    create_combined_text,
    explain_smart_score,
    get_smart_vocabulary,
    is_empty_value,
)
import re
from zipfile import BadZipFile

# Lightweight NLP utilities
try:
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    import nltk
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False

try:
    from rapidfuzz import fuzz
    _RAPIDFUZZ_AVAILABLE = True
except Exception:
    _RAPIDFUZZ_AVAILABLE = False

st.set_page_config(page_title=DASHBOARD_TITLE, layout="wide")

BASE_DIR = Path(__file__).resolve().parent
NLTK_LOCAL_DATA = BASE_DIR / "nltk_data"
NLTK_LOCAL_DATA.mkdir(exist_ok=True)


@st.cache_resource(show_spinner=False)
def _init_nlp_resources():
    resources = {
        "lemmatizer": None,
        "wordnet_ready": False,
        "fuzzy_ready": _RAPIDFUZZ_AVAILABLE,
    }
    if _NLTK_AVAILABLE:
        try:
            nltk_local_path = str(NLTK_LOCAL_DATA)
            if nltk_local_path not in nltk.data.path:
                nltk.data.path.insert(0, nltk_local_path)

            # Ensure wordnet is available
            nltk.data.find('corpora/wordnet')
            nltk.data.find('corpora/omw-1.4')
            resources["wordnet_ready"] = True
        except (LookupError, BadZipFile):
            try:
                # Attempt a fresh download in case the local corpus is missing or corrupted
                nltk.download('wordnet', quiet=True, download_dir=nltk_local_path)
                nltk.download('omw-1.4', quiet=True, download_dir=nltk_local_path)
                if nltk_local_path not in nltk.data.path:
                    nltk.data.path.insert(0, nltk_local_path)
                nltk.data.find('corpora/wordnet')
                nltk.data.find('corpora/omw-1.4')
                resources["wordnet_ready"] = True
            except Exception:
                resources["wordnet_ready"] = False
                st.warning("⚠️ WordNet corpus unavailable or corrupted. Falling back to keyword-only SMART analysis.")
        except Exception:
            resources["wordnet_ready"] = False
            st.warning("⚠️ Unexpected NLP initialization issue. Falling back to keyword-only SMART analysis.")
        try:
            resources["lemmatizer"] = WordNetLemmatizer()
        except Exception:
            resources["lemmatizer"] = None
    return resources


_NLP_RES = _init_nlp_resources()


def _tokenize(text: str) -> set:
    if not isinstance(text, str):
        return set()
    # basic normalization and tokenization
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9% ]+", " ", lowered)
    parts = [p for p in lowered.split() if p]
    if _NLP_RES.get("lemmatizer"):
        try:
            parts = [_NLP_RES["lemmatizer"].lemmatize(p) for p in parts]
        except Exception:
            pass
    return set(parts)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _expand_with_synonyms(base_terms: list) -> set:
    expanded = set()
    for term in base_terms:
        expanded.add(term.lower())
        expanded.update(_tokenize(term))
        if _NLP_RES.get("wordnet_ready"):
            try:
                for syn in wn.synsets(term):
                    for l in syn.lemmas():
                        expanded.add(l.name().replace('_', ' ').lower())
            except Exception:
                # fallback silently
                pass
    # also include lemmatized single tokens
    flat_tokens = set()
    for t in list(expanded):
        flat_tokens.update(_tokenize(t))
    expanded.update(flat_tokens)
    return expanded

# Theme Toggle - Top Right Corner (Compact)
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.markdown(f"# {DASHBOARD_TITLE}")
with col3:
    # Compact theme toggle
    st.markdown("""
    <style>
    .compact-theme {
        font-size: 0.8em;
        margin: 0;
        padding: 0;
    }
    .theme-status {
        font-size: 0.7em;
        margin-top: -10px;
        padding: 2px 5px;
        border-radius: 3px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="compact-theme">🎨 Theme</div>', unsafe_allow_html=True)
    theme_mode = st.selectbox(
        "Choose Theme:",
        ["Light", "Dark", "Auto"],
        index=0,
        help="Select your preferred theme mode",
        label_visibility="collapsed"
    )
    
    # Show current theme status in a very compact way
    if theme_mode == "Dark":
        st.markdown('<div class="theme-status" style="background-color: #2d2d2d; color: #ffffff;">🌙 Dark</div>', unsafe_allow_html=True)
    elif theme_mode == "Light":
        st.markdown('<div class="theme-status" style="background-color: #f0f2f6; color: #262730;">☀️ Light</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="theme-status" style="background-color: #e8f4fd; color: #1f77b4;">🔄 Auto</div>', unsafe_allow_html=True)

# View Toggle - Employee Based vs Goal Based
st.markdown("---")
st.markdown("### 📊 View Mode")
view_mode = st.radio(
    "Select View:",
    ["Goal Based", "Employee Based"],
    index=1,  # Default to Employee Based
    horizontal=True,
    help="Choose whether to view metrics based on goal count or unique employee count"
)
st.markdown("---")

# Apply comprehensive theme styling
if theme_mode == "Dark":
    st.markdown("""
    <style>
    /* Main App Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Main Container */
    .main .block-container {
        background-color: #0e1117;
        color: #ffffff;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers and Text */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Metrics Cards */
    .stMetric {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .stMetric > div {
        color: #ffffff;
    }
    
    .stMetric [data-testid="metric-value"] {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .stMetric [data-testid="metric-label"] {
        color: #d1d5db !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
    }
    
    /* Expanders */
    .stExpander {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
    }
    
    .stExpander > div {
        background-color: #1f2937;
    }
    
    .stExpander > div > div {
        color: #ffffff;
    }
    
    /* Selectboxes and Multiselect */
    .stSelectbox > div > div {
        background-color: #1f2937;
        color: #ffffff;
        border: 1px solid #374151;
    }
    
    .stMultiSelect > div > div {
        background-color: #1f2937;
        color: #ffffff;
        border: 1px solid #374151;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1f2937;
    }
    
    .css-1d391kg .stSelectbox > div > div {
        background-color: #374151;
        color: #ffffff;
    }
    
    /* Tables */
    .stTable {
        background-color: #1f2937;
        color: #ffffff;
    }
    
    .stTable table {
        background-color: #1f2937;
        color: #ffffff;
    }
    
    .stTable th {
        background-color: #374151;
        color: #ffffff;
        border: 1px solid #4b5563;
    }
    
    .stTable td {
        background-color: #1f2937;
        color: #ffffff;
        border: 1px solid #4b5563;
    }
    
    /* Info, Success, Warning, Error boxes */
    .stAlert {
        background-color: #1f2937;
        border: 1px solid #374151;
        color: #ffffff;
    }
    
    .stInfo {
        background-color: #1e3a8a;
        border: 1px solid #3b82f6;
        color: #ffffff;
    }
    
    .stSuccess {
        background-color: #065f46;
        border: 1px solid #10b981;
        color: #ffffff;
    }
    
    .stWarning {
        background-color: #92400e;
        border: 1px solid #f59e0b;
        color: #ffffff;
    }
    
    .stError {
        background-color: #991b1b;
        border: 1px solid #ef4444;
        color: #ffffff;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #ffffff;
    }
    
    .stMarkdown p {
        color: #d1d5db;
    }
    
    /* Plotly charts background */
    .js-plotly-plot {
        background-color: #1f2937;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1f2937;
        border: 1px solid #374151;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1f2937;
        color: #d1d5db;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #374151;
        color: #ffffff;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #374151;
        color: #ffffff;
        border: 1px solid #4b5563;
    }
    
    .stButton > button:hover {
        background-color: #4b5563;
        color: #ffffff;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: #374151;
    }
    
    /* Code blocks */
    .stCode {
        background-color: #1f2937;
        color: #ffffff;
        border: 1px solid #374151;
    }
    
    /* Custom theme status styling */
    .theme-status {
        background-color: #374151 !important;
        color: #ffffff !important;
        border: 1px solid #4b5563;
    }
    
    /* Vibrant accent colors for dark theme */
    .accent-coral { color: #ff6b6b !important; }
    .accent-teal { color: #4ecdc4 !important; }
    .accent-blue { color: #45b7d1 !important; }
    .accent-green { color: #96ceb4 !important; }
    .accent-yellow { color: #feca57 !important; }
    .accent-pink { color: #ff9ff3 !important; }
    .accent-purple { color: #5f27cd !important; }
    .accent-orange { color: #ff9f43 !important; }
    
    /* Enhanced metric styling with vibrant accents */
    .stMetric [data-testid="metric-value"] {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    
    /* Vibrant borders for interactive elements */
    .stSelectbox > div > div:focus,
    .stMultiSelect > div > div:focus {
        border-color: #ff6b6b !important;
        box-shadow: 0 0 0 2px rgba(255, 107, 107, 0.2) !important;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
        border: none;
        color: #ffffff;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff5252, #26a69a);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)
elif theme_mode == "Light":
    st.markdown("""
    <style>
    /* Main App Background */
    .stApp {
        background-color: #ffffff;
        color: #262730;
    }
    
    /* Main Container */
    .main .block-container {
        background-color: #ffffff;
        color: #262730;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers and Text */
    h1, h2, h3, h4, h5, h6 {
        color: #262730 !important;
    }
    
    /* Metrics Cards */
    .stMetric {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    }
    
    .stMetric > div {
        color: #262730;
    }
    
    .stMetric [data-testid="metric-value"] {
        color: #262730 !important;
        font-weight: 600;
    }
    
    .stMetric [data-testid="metric-label"] {
        color: #6c757d !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 8px;
    }
    
    /* Expanders */
    .stExpander {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
    }
    
    .stExpander > div {
        background-color: #f8f9fa;
    }
    
    .stExpander > div > div {
        color: #262730;
    }
    
    /* Selectboxes and Multiselect */
    .stSelectbox > div > div {
        background-color: #ffffff;
        color: #262730;
        border: 1px solid #ced4da;
    }
    
    .stMultiSelect > div > div {
        background-color: #ffffff;
        color: #262730;
        border: 1px solid #ced4da;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    .css-1d391kg .stSelectbox > div > div {
        background-color: #ffffff;
        color: #262730;
    }
    
    /* Tables */
    .stTable {
        background-color: #ffffff;
        color: #262730;
    }
    
    .stTable table {
        background-color: #ffffff;
        color: #262730;
    }
    
    .stTable th {
        background-color: #f8f9fa;
        color: #262730;
        border: 1px solid #dee2e6;
    }
    
    .stTable td {
        background-color: #ffffff;
        color: #262730;
        border: 1px solid #dee2e6;
    }
    
    /* Info, Success, Warning, Error boxes */
    .stAlert {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        color: #262730;
    }
    
    .stInfo {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    
    .stSuccess {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    .stWarning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    
    .stError {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #262730;
    }
    
    .stMarkdown p {
        color: #495057;
    }
    
    /* Plotly charts background */
    .js-plotly-plot {
        background-color: #ffffff;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        color: #6c757d;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #262730;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #007bff;
        color: #ffffff;
        border: 1px solid #007bff;
    }
    
    .stButton > button:hover {
        background-color: #0056b3;
        color: #ffffff;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: #007bff;
    }
    
    /* Code blocks */
    .stCode {
        background-color: #f8f9fa;
        color: #262730;
        border: 1px solid #e9ecef;
    }
    
    /* Custom theme status styling */
    .theme-status {
        background-color: #e9ecef !important;
        color: #262730 !important;
        border: 1px solid #ced4da;
    }
    </style>
    """, unsafe_allow_html=True)

# Use the configured file from config.py (admin-controlled)
selected_file = get_output_file()

# --- SMART Framework Info ---
SMART_INFO = '''
### SMART Framework Definitions & Audit Keywords

**S - Specific**: Clear, unambiguous, focused, action verbs (Implement, Develop, etc.)
**M - Measurable**: Quantifiable, metrics, KPIs, targets, progress tracking
**A - Achievable**: Realistic, feasible, resources/skills available, not impossible
**R - Relevant**: Aligned to strategy, meaningful, impactful
**T - Time-bound**: Clear deadline, timeline, milestones, urgency
'''

@st.cache_data(ttl=CACHE_TTL)
def load_data(file_path=None):
    try:
        if file_path is None:
            file_path = get_output_file()
        
        # Check if file exists
        import os
        if not os.path.exists(file_path):
            st.error(f"❌ Output file not found: {file_path}")
            st.info("🔄 Please run the goal audit script first to generate the processed data.")
            st.code("python goal_audit.py goals_input.xlsx --output goals_output.xlsx", language="bash")
            return pd.DataFrame()
        
        df = pd.read_excel(file_path)
        
        # Check if data is empty
        if df.empty:
            st.error("❌ The data file is empty")
            return pd.DataFrame()
        
        # Optimize data types for better performance and prevent pyarrow errors
        for col in df.columns:
            if df[col].dtype == 'object':
                # Convert to string and limit length for better performance
                df[col] = df[col].astype(str).str[:300]  # Limit string length further
                # Replace problematic characters that cause pyarrow issues
                df[col] = df[col].str.replace('%', ' percent', regex=False)
                df[col] = df[col].str.replace('(', ' [', regex=False)
                df[col] = df[col].str.replace(')', ']', regex=False)
                # Handle percentage values in any column
                df[col] = df[col].str.replace(r'(\d+\.?\d*)%', r'\1 percent', regex=True)
                # Clean up any problematic characters
                df[col] = df[col].str.replace(r'[^\w\s\.\-\[\]\(\)]', ' ', regex=True)
                # Replace NaN strings with empty string
                df[col] = df[col].replace(['nan', 'NaN', 'None'], '')
            
            # Handle NaN values in numeric columns that might cause issues
            elif df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(0)
            
            # Convert boolean columns to proper boolean type
            elif df[col].dtype == 'bool':
                df[col] = df[col].fillna(False)
        
        st.success(f"✅ Loaded {len(df)} rows of data successfully")
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("💡 Try running: `python goal_audit.py goals_input.xlsx --output goals_output.xlsx`")
        return pd.DataFrame()

df = load_data(selected_file)

# Check if data was loaded successfully and has required columns
if df.empty:
    st.stop()

@st.cache_resource(show_spinner=False)
def _build_smart_lexicon():
    """Build expanded lexicon using centralized vocabulary"""
    base = get_smart_vocabulary()
    
    # Expand with synonyms where appropriate (skip negatives expansion)
    expanded = {
        "specific": _expand_with_synonyms(base["specific"]),
        "measurable": _expand_with_synonyms(base["measurable"]),
        "achievable_pos": _expand_with_synonyms(base["achievable_pos"]),
        "achievable_neg": set([t.lower() for t in base["achievable_neg"]]),
        "relevant": _expand_with_synonyms(base["relevant"]),
        "timebound": _expand_with_synonyms(base["timebound"]),
    }
    return expanded


SMART_LEXICON = _build_smart_lexicon()
# Get base vocabulary for fast mode
SMART_VOCAB_BASE = get_smart_vocabulary()

def _contains_phrase_fuzzy(text: str, phrases: set, threshold: int = 90) -> bool:
    if not isinstance(text, str) or not text:
        return False
    if not _NLP_RES.get("fuzzy_ready"):
        # fallback to simple any substring
        lowered = text.lower()
        return any(p in lowered for p in phrases)
    lowered = text.lower()
    for p in phrases:
        if _fuzzy_match(lowered, p, threshold):
            return True
    return False


@lru_cache(maxsize=4096)
def _fuzzy_match(lowered_text: str, phrase: str, threshold: int) -> bool:
    try:
        return fuzz.partial_ratio(phrase, lowered_text) >= threshold
    except Exception:
        return False


def analyze_smart_goal(goal_desc, target, metric, business_unit=None, domain=None):
    """
    Enhanced SMART analysis that layers synonym/fuzzy NLP on top of the fast baseline.
    Guarantees it never downgrades a pillar that already passed in fast mode.
    """
    if pd.isna(goal_desc) or str(goal_desc).strip() == '' or str(goal_desc).strip().lower() == 'nan':
        return {'Specific': False, 'Measurable': False, 'Achievable': False, 'Relevant': False, 'TimeBound': False}
    
    # Start with the fast analysis as a conservative baseline
    fast_result = analyze_smart_goal_fast(goal_desc, target, metric, business_unit, domain)
    result = fast_result.copy()

    # Short-circuit if everything is already satisfied
    if all(result.values()):
        return result

    goal_text = str(goal_desc)
    metric_text = '' if pd.isna(metric) else str(metric)
    target_text = '' if pd.isna(target) else str(target)
    
    goal_lower = goal_text.lower()
    metric_lower = metric_text.lower()
    target_lower = target_text.lower()
    combined_text = f"{goal_text} {target_text} {metric_text}".lower()
    has_countable = (
        has_countable_number(goal_lower) or
        has_countable_number(metric_lower) or
        has_countable_number(target_lower) or
        has_countable_number(combined_text)
    )

    tokens_goal = _tokenize(goal_text)
    tokens_metric = _tokenize(metric_text)
    tokens_target = _tokenize(target_text)

    # Read current UI tuning
    thr = st.session_state.get("smart_fuzzy_threshold", 95)
    require_ctx = st.session_state.get("smart_require_context_syn", True)

    vocab = SMART_VOCAB_BASE

    # --- Specific ---
    if not result['Specific']:
        strong_action_verbs = set(vocab["specific"][:30])
        action_nouns = {
            'addition', 'creation', 'development', 'implementation', 'establishment',
            'launch', 'design', 'integration', 'migration', 'upgrade', 'enhancement',
            'expansion', 'improvement', 'reduction', 'delivery', 'execution'
        }
        has_action_context = any(verb in goal_lower for verb in strong_action_verbs) or \
                             any(noun in goal_lower for noun in action_nouns)
        has_deliverable = any(term in combined_text for term in SPECIFIC_DELIVERABLE_TERMS)

        specific_has_keywords = len(tokens_goal & SMART_LEXICON["specific"]) >= 1
        fuzzy_specific = _contains_phrase_fuzzy(goal_text, SMART_LEXICON["specific"], threshold=thr)

        is_metric_description_only = ('percentage of' in goal_lower or 'percent of' in goal_lower) and not has_action_context
        has_sufficient_detail = len(goal_lower.split()) >= 5
        keyword_specific = (specific_has_keywords and has_sufficient_detail and has_deliverable)

        if require_ctx and fuzzy_specific:
            fuzzy_specific = fuzzy_specific and has_action_context and has_deliverable

        if (
            (has_action_context and has_deliverable and not is_metric_description_only)
            or (keyword_specific and not is_metric_description_only)
            or fuzzy_specific
        ):
            result['Specific'] = True

    # --- Measurable ---
    if not result['Measurable']:
        metric_empty = is_empty_value(metric_text)
        target_empty = is_empty_value(target_text)

        has_explicit_numbers = bool(re.search(r'\d+', goal_text)) and (
            '%' in goal_lower or
            any(word in goal_lower for word in ['target', 'goal', 'reach', 'achieve', 'exceed', 'attain', 'against', 'versus']) or
            has_countable
        )

        def is_valid_target_value(text):
            if not text or pd.isna(text):
                return False
            text_str = str(text).strip().lower()
            if not text_str or text_str == 'nan':
                return False
            if re.match(r'^\s*(20\d{2}|q[1-4]|h[12]|fy\d{2,4})\s*$', text_str):
                return False
            if re.search(r'\d+[.,]?\d*\s*%|\d+\s*percent|₹|\$|€|£|\d+\s*/\s*\d+|\d+\s*:\s*\d+', text_str):
                return True
            if re.search(r'(>=|<=|>|<|at least|no more than|increase to|reduce to|improve to)', text_str):
                return True
            target_context = [
                'target', 'achieve', 'reach', 'exceed', 'attain', 'goal',
                'increase to', 'reduce to', 'improve to', 'decrease to',
                'at least', 'no more than', 'minimum', 'maximum', 'baseline'
            ]
            return any(ctx in text_str for ctx in target_context) and re.search(r'\d+', text_str)

        has_target_value = not target_empty and is_valid_target_value(target_text)
        has_metric_value = not metric_empty and (
            any(ch.isdigit() for ch in metric_text) or
            any(unit in metric_lower for unit in ['%', 'percent', 'ratio', 'count', 'number', 'kpi', 'metric'])
        )

        has_percent_with_target = (
            ('%' in goal_text or 'percent' in goal_lower or 'percentage' in goal_lower) and
            any(word in goal_lower for word in ['target', 'achieve', 'reach', 'exceed', 'attain', 'goal of', 'increase to', 'decrease to', 'reduce to', 'improve to', 'at least', 'no more than'])
        )

        is_descriptive_only = ('percentage of' in goal_lower or 'percent of' in goal_lower) and \
                              not any(word in goal_lower for word in ['achieve', 'target', 'reach', 'exceed', 'attain', 'against', 'versus'])

        measurable_action_words = {'reach', 'achieve', 'exceed', 'attain', 'target', 'goal', 'measure', 'track', 'monitor', 'quantify', 'report'}
        has_measurable_intent = any(word in goal_lower for word in measurable_action_words)
        has_measurable_keywords = len(tokens_goal & SMART_LEXICON["measurable"]) >= 1
        fuzzy_measurable = _contains_phrase_fuzzy(goal_text, SMART_LEXICON["measurable"], threshold=thr)
        has_defined_metric = not metric_empty and len(metric_text) > 5

        # Check for AOP (Annual Operating Plan) targets - these are measurable
        aop_patterns = [
            r'\baop\s+target',  # "aop target"
            r'\baop\s+targets',  # "aop targets"
            r'\bagainst\s+(?:the\s+)?(?:set\s+)?(?:management\s+)?aop\s+target',  # "against the set Management AOP Target"
            r'\bagainst\s+(?:the\s+)?(?:management\s+)?aop\s+targets',  # "against management AOP targets"
            r'\bagainst\s+aop',  # "against aop"
            r'\bas\s+per\s+aop',  # "as per aop"
            r'\baop\s+for',  # "aop for"
            r'\baop\s+sheet',  # "aop sheet"
            r'\bmanagement\s+aop\s+target',  # "management aop target"
            r'\bmanagement\s+aop\s+targets',  # "management aop targets"
            r'\bmanagement\s+aop',  # "management aop"
            r'\bannual\s+operating\s+plan\s+target',  # "annual operating plan target"
            r'\baop\s+plan',  # "aop plan"
            r'\baop\s+budget',  # "aop budget"
            r'\baop\s+achievement',  # "aop achievement"
            r'\baop\s+goal',  # "aop goal"
            r'\baop\s+goals',  # "aop goals"
            r'\baop\s+group\s+target',  # "aop group target"
            r'\baop\s+register',  # "aop register"
        ]
        has_aop_target = any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in aop_patterns)
        
        has_numeric = bool(re.search(r'\d', combined_text))
        measurable = (
            has_aop_target or  # AOP targets are always measurable
            has_explicit_numbers or
            (has_percent_with_target and not is_descriptive_only) or
            has_target_value or
            has_metric_value or
            has_countable or
            (has_measurable_intent and (has_measurable_keywords or fuzzy_measurable)) or
            has_defined_metric
        )
        if has_aop_target or (has_numeric or has_countable) and measurable:
            result['Measurable'] = True

    # --- Achievable ---
    if not result['Achievable']:
        neg_hit = any(n in goal_lower for n in SMART_LEXICON["achievable_neg"]) or \
                  _contains_phrase_fuzzy(goal_text, SMART_LEXICON["achievable_neg"], threshold=max(92, thr))

        unrealistic_patterns = ['zero bugs', 'zero downtime', '100 percent', 'guaranteed', 'no exceptions',
                                'infinite', 'instant', 'no limits', 'zero error', 'zero failure',
                                'perfect score', 'flawless', 'error-free', 'bug-free']
        has_unrealistic_patterns = any(pattern in goal_lower for pattern in unrealistic_patterns)

        pos_hit = len(tokens_goal & SMART_LEXICON["achievable_pos"]) > 0 or \
                  _contains_phrase_fuzzy(goal_text, SMART_LEXICON["achievable_pos"], threshold=max(90, thr))
        has_feasibility_indicators = pos_hit or any(
            word in goal_lower for word in [
                'mvp', 'phased', 'staged', 'incremental', 'realistic', 'feasible',
                'manageable', 'attainable', 'resources', 'budget', 'capacity'
            ]
        )
        has_unrealistic_language = any(word in goal_lower for word in [
            'all', 'every', 'always', 'never', 'impossible', 'perfect', '100%', 'zero-defect', 'unlimited', 'instantaneous'
        ]) or has_unrealistic_patterns
        is_well_structured = len(goal_lower.split()) >= 7 and not has_unrealistic_language

        if (not neg_hit) and (has_feasibility_indicators or is_well_structured):
            result['Achievable'] = True

    # --- Relevant ---
    if not result['Relevant']:
        # Use enhanced has_relevant function with Business Unit and Domain context
        from smart_rules import has_relevant
        if has_relevant(goal_desc, metric_text, business_unit, domain):
            result['Relevant'] = True
        else:
            # Additional checks for business outcomes (fallback)
            business_outcomes = {
                'revenue', 'growth', 'cost', 'efficiency', 'productivity', 'satisfaction',
                'customer', 'business', 'team', 'process', 'system', 'quality', 'performance',
                'p&l', 'unit economics', 'pipeline contribution', 'c-sat', 'sla', 'tat'
            }
            has_business_outcome = any(term in goal_lower for term in business_outcomes)
            has_alignment_context = any(
                word in goal_lower for word in [
                    'for', 'to support', 'to drive', 'contribute', 'align', 'enable',
                    'to improve', 'to enhance', 'to reduce', 'to increase'
                ]
            )
            if has_business_outcome and has_alignment_context and len(goal_lower.split()) >= 8:
                result['Relevant'] = True

    # --- TimeBound ---
    if not result['TimeBound']:
        has_time_pattern = any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in TIME_REGEX_PATTERNS)
        if has_time_pattern:
            result['TimeBound'] = True
        else:
            has_keyword = any(keyword in combined_text for keyword in TIME_KEYWORDS)
            has_deadline_context = any(
                word in combined_text for word in ['by', 'within', 'deadline', 'due', 'target date', 'end of']
            )
            result['TimeBound'] = has_keyword and has_deadline_context

    return {pillar: bool(val) for pillar, val in result.items()}

# Apply SMART analysis to all goals
st.markdown("---")
col_mode_a, col_mode_b, col_mode_c = st.columns([1,2,2])
with col_mode_a:
    enhanced_nlp = st.checkbox(
        "Use enhanced NLP (synonyms + fuzzy)",
        value=False,
        help="More accurate but slower on large datasets"
    )
with col_mode_b:
    if enhanced_nlp:
        fuzzy_threshold = st.slider("NLP fuzzy threshold", min_value=85, max_value=100, value=95, step=1, help="Higher = stricter matches")
    else:
        fuzzy_threshold = 95
with col_mode_c:
    if enhanced_nlp:
        require_ctx_syn = st.checkbox("Require context with synonyms", value=True, help="Synonym/fuzzy hits must also have action/alignment context")
    else:
        require_ctx_syn = True

# Share settings with analysis functions via session state
st.session_state["smart_fuzzy_threshold"] = fuzzy_threshold
st.session_state["smart_require_context_syn"] = require_ctx_syn

# Show analysis status with progress indicator
if enhanced_nlp:
    st.info("🔍 Analyzing goals for SMART criteria (enhanced NLP mode - this may take longer for large datasets)...")
    progress_bar = st.progress(0)
    status_text = st.empty()
else:
    st.info("🔍 Analyzing goals for SMART criteria (fast mode)...")
    progress_bar = None
    status_text = None

def analyze_smart_goal_fast(goal_desc, target, metric, business_unit=None, domain=None):
    """Fast SMART analysis without synonyms/fuzzy matching - uses centralized vocabulary."""
    if pd.isna(goal_desc) or str(goal_desc).strip() == '' or str(goal_desc).strip().lower() == 'nan':
        return {'Specific': False, 'Measurable': False, 'Achievable': False, 'Relevant': False, 'TimeBound': False}

    goal_text = str(goal_desc)
    metric_text = '' if pd.isna(metric) else str(metric)
    target_text = '' if pd.isna(target) else str(target)
    
    text = goal_text.lower()
    metric_lower = metric_text.lower()
    target_lower = target_text.lower()
    combined_text = f"{goal_text} {target_text} {metric_text}".lower()
    has_countable = (
        has_countable_number(text) or
        has_countable_number(metric_lower) or
        has_countable_number(target_lower) or
        has_countable_number(combined_text)
    )

    # Use centralized vocabulary
    vocab = SMART_VOCAB_BASE
    specific_terms = set(vocab["specific"])
    measurable_terms = set(vocab["measurable"])
    achievable_neg = set(vocab["achievable_neg"])
    achievable_pos = set(vocab["achievable_pos"])
    relevant_terms = set(vocab["relevant"])
    time_terms = set(vocab["timebound"])

    # Specific - require action context + deliverable
    specific_has_action = any(term in text for term in specific_terms) or any(term in metric_lower for term in specific_terms)
    has_deliverable = any(term in combined_text for term in SPECIFIC_DELIVERABLE_TERMS)
    specific = specific_has_action and has_deliverable

    # Measurable: Check for empty values first
    metric_empty = is_empty_value(metric_text)
    target_empty = is_empty_value(target_text)
    
    # Stricter target detection helper
    def is_valid_target_fast(txt):
        if not txt or pd.isna(txt):
            return False
        txt_str = str(txt).strip().lower()
        if not txt_str or txt_str == 'nan':
            return False
        # Exclude standalone years/quarters
        if re.match(r'^\s*(20\d{2}|q[1-4]|h[12]|fy\d{2,4})\s*$', txt_str):
            return False
        # Require numbers with units, comparators, or target context
        return bool(re.search(r'\d+[.,]?\d*\s*%|\d+\s*percent|₹|\$|€|£|\d+\s*/\s*\d+|>=|<=|>|<|at least|target|achieve|reach', txt_str))
    
    # Check for AOP (Annual Operating Plan) targets - these are measurable
    aop_patterns = [
        r'\baop\s+target',
        r'\baop\s+targets',
        r'\bagainst\s+aop',
        r'\bas\s+per\s+aop',
        r'\baop\s+for',
        r'\baop\s+sheet',
        r'\bmanagement\s+aop',
        r'\bannual\s+operating\s+plan\s+target',
        r'\baop\s+plan',
        r'\baop\s+budget',
        r'\baop\s+achievement',
        r'\baop\s+goal',
        r'\baop\s+goals',
    ]
    has_aop_target = any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in aop_patterns)
    
    has_numbers = any(ch.isdigit() for ch in goal_text) and (
        '%' in text or
        any(word in text for word in ['target', 'achieve', 'reach', 'goal', 'attain']) or
        has_countable
    )
    has_percent = ('%' in goal_text) or ('percent' in text) or ('percentage' in text)
    has_target_value = not target_empty and is_valid_target_fast(target_text)
    has_metric_value = not metric_empty and (any(ch.isdigit() for ch in metric_text) or 
                                            any(unit in metric_lower for unit in ['%', 'percent', 'ratio', 'count', 'number', 'kpi', 'metric']))
    has_measurable_keywords = any(t in text for t in measurable_terms) or any(t in metric_lower for t in measurable_terms)
    
    measurable = has_aop_target or has_numbers or has_percent or has_target_value or has_metric_value or has_measurable_keywords or has_countable
    has_numeric = bool(re.search(r'\d', combined_text))
    # AOP targets are always measurable, even without explicit numbers
    if not has_aop_target and not has_numeric and not has_countable:
        measurable = False

    # Achievable: Stricter logic
    neg = any(n in text for n in achievable_neg)
    unrealistic_patterns = ['zero bugs', 'zero downtime', '100 percent', 'guaranteed', 'no exceptions', 
                          'infinite', 'instant', 'no limits', 'zero error', 'zero failure', 
                          'perfect score', 'flawless', 'error-free', 'bug-free']
    has_unrealistic = any(pattern in text for pattern in unrealistic_patterns)
    pos = any(t in text for t in achievable_pos)
    has_feasibility = pos or any(word in text for word in ['mvp', 'phased', 'staged', 'incremental', 'realistic', 'feasible', 'manageable', 'attainable'])
    is_well_structured = len(text.split()) >= 7 and not has_unrealistic
    
    achievable = (not neg) and (has_feasibility or is_well_structured)

    # Relevant: Use enhanced has_relevant function with Business Unit and Domain context
    from smart_rules import has_relevant
    relevant = has_relevant(goal_desc, metric_text, business_unit, domain)

    # TimeBound: Include Indian fiscal calendar patterns
    has_time_keywords = any(t in text for t in time_terms) or any(t in metric_lower for t in time_terms) or any(t in target_lower for t in time_terms)
    
    # Indian fiscal calendar regex patterns
    indian_fy_patterns = [
        r'\bfy\s*[\'\-]?\s*\d{2,4}\b',
        r'\bh[12]\s+fy\s*\d{2,4}\b',
        r'\bq[1-4][\'\-]\d{2}\b',
        r'\bq[1-4]\s+fy\s*\d{2,4}\b',
        r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\'\-]\d{2}\b',
    ]
    has_indian_fy = any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in indian_fy_patterns)
    
    # Date formats
    date_patterns = [
        r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',
        r'\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b',
    ]
    has_date_format = any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in date_patterns)
    
    # Sprint patterns
    sprint_patterns = [
        r'\bsprint\s+\d+\b',
        r'\bpi[-]\d+\b',
        r'\biteration\s+\d+\b',
        r'\bnext\s+sprint\b'
    ]
    has_sprint_ref = any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in sprint_patterns)
    
    timebound = has_time_keywords or has_indian_fy or has_date_format or has_sprint_ref

    return {'Specific': specific, 'Measurable': measurable, 'Achievable': achievable, 'Relevant': relevant, 'TimeBound': timebound}

analysis_fn = analyze_smart_goal if enhanced_nlp else analyze_smart_goal_fast

# Process with progress indicator for NLP mode
if enhanced_nlp and progress_bar is not None:
    total_rows = len(df)
    results = []
    for idx, row in df.iterrows():
        result = analysis_fn(
            row.get('Goal Description', ''),
            row.get('Target', ''),
            row.get('Goal Metric / Measurement Criteria', ''),
            row.get('Business Unit', None),
            row.get('Domain', None)
        )
        results.append(result)
        if (idx + 1) % max(1, total_rows // 20) == 0 or (idx + 1) == total_rows:
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
            if status_text:
                status_text.text(f"Processing {idx + 1} of {total_rows} goals...")
    smart_analysis = pd.DataFrame(results)
    progress_bar.progress(1.0)
    if status_text:
        status_text.text("✅ Analysis complete!")
else:
    smart_analysis = df.apply(lambda row: analysis_fn(
        row.get('Goal Description', ''),
        row.get('Target', ''),
        row.get('Goal Metric / Measurement Criteria', ''),
        row.get('Business Unit', None),
        row.get('Domain', None)
    ), axis=1, result_type='expand')

# Add SMART columns to dataframe
smart_bool = smart_analysis[SMART_COLUMNS].fillna(False).astype(bool)
for col in SMART_COLUMNS:
    df[col] = smart_bool[col]
df['All_SMART_Pillars'] = smart_bool.all(axis=1)
df['Pillar_Present'] = smart_bool.any(axis=1)

# Recalculate SMART score strictly from current pillar booleans
df['SMART_score'] = smart_bool.astype(int).sum(axis=1)

# Ensure combined analysis text is available for explanations and exports
if 'Combined_Text' not in df.columns:
    df['Combined_Text'] = df.apply(create_combined_text, axis=1)

# Create quality categories (aligned with definitions: High=4-5, Medium=2-3, Low=0-1)
df['high_quality'] = df['SMART_score'] >= HIGH_QUALITY_THRESHOLD  # 4-5
df['medium_quality'] = (df['SMART_score'] >= 2) & (df['SMART_score'] <= 3)  # 2-3
df['low_quality'] = df['SMART_score'] <= 1  # 0-1

# Check for other SMART-related columns
if 'Target Defined' in df.columns:
    df['Has_Target'] = df['Target Defined']
else:
    # Stricter target detection: exclude years/quarters without context
    def is_valid_target(text):
        if pd.isna(text) or str(text).strip().lower() in ('', 'nan'):
            return False
        text_str = str(text).strip().lower()
        # Exclude standalone years/quarters
        if re.match(r'^\s*(20\d{2}|q[1-4]|h[12]|fy\d{2,4})\s*$', text_str):
            return False
        # Require numbers with units, comparators, or target context
        return bool(re.search(r'\d+[.,]?\d*\s*%|\d+\s*percent|₹|\$|€|£|\d+\s*/\s*\d+|>=|<=|>|<|at least|target|achieve|reach', text_str))
    
    df['Has_Target'] = df['Target'].apply(is_valid_target)

# Create Has_Metric column
if 'Goal Metric / Measurement Criteria' in df.columns:
    df['Has_Metric'] = df['Goal Metric / Measurement Criteria'].notna() & (df['Goal Metric / Measurement Criteria'].astype(str).str.strip() != '') & (df['Goal Metric / Measurement Criteria'].astype(str).str.strip().str.lower() != 'nan')
else:
    df['Has_Metric'] = False

st.success(f"✅ SMART analysis complete! Average SMART score: {df['SMART_score'].mean():.2f}")

# Clean manager names (remove IDs in brackets)
manager_col = None
if 'Manager Name (Manager ID)' in df.columns:
    manager_col = 'Manager Name (Manager ID)'
    df['Manager_Name_Clean'] = df[manager_col].str.replace(r'\s*\([^)]*\)', '', regex=True).str.strip()
    # Count unique managers
    unique_managers = df['Manager_Name_Clean'].nunique()
    st.info(f"👥 **Total Unique Managers:** {unique_managers}")
elif 'Manager Name' in df.columns:
    manager_col = 'Manager Name'
    df['Manager_Name_Clean'] = df[manager_col].str.replace(r'\s*\([^)]*\)', '', regex=True).str.strip()
    unique_managers = df['Manager_Name_Clean'].nunique()
    st.info(f"👥 **Total Unique Managers:** {unique_managers}")

# Handle blank goals - mark them as not filled
if 'Goal Description' in df.columns:
    df['Goal_Not_Filled'] = df['Goal Description'].isna() | (df['Goal Description'].str.strip() == '') | (df['Goal Description'].str.strip() == 'nan')
    blank_goals_count = df['Goal_Not_Filled'].sum()
    if blank_goals_count > 0:
        st.info(f"📝 **Note:** {blank_goals_count} goals are marked as 'Not Filled' due to blank descriptions")

# --- Helper Functions ---
def get_quality(score):
    """Classify quality: High=4-5, Medium=2-3, Low=0-1"""
    if score >= HIGH_QUALITY_THRESHOLD: return "High"  # 4-5
    elif score >= 2 and score <= 3: return "Medium"  # 2-3
    else: return "Low"  # 0-1

def get_missing_pillars(missing_col):
    if pd.isna(missing_col) or not missing_col: return []
    return [x.strip() for x in missing_col.split(",") if x.strip()]

def goal_load_bucket(n):
    if n < 2: return "<2 Goals"
    elif 2 <= n <= 3: return "2-3 Goals"
    elif 4 <= n <= 5: return "4-5 Goals"
    else: return ">5 Goals"

def generate_smart_feedback(row):
    """
    Generate clean, readable SMART feedback with rewritten goals based on Goal Description
    """
    feedback_parts = []
    goal_desc = str(row.get('Goal Description', '')).strip()
    
    # Generate AI-powered rewritten goal first
    rewritten_goal = generate_ai_rewritten_goal(row)
    
    if rewritten_goal:
        feedback_parts.append(f"AI Rewritten SMART Goal:\n{rewritten_goal}")
        
        # Add SMART pillar analysis for the rewritten goal
        feedback_parts.append("\nSMART Pillar Analysis:")
        
        # Check each SMART pillar for the rewritten goal
        smart_analysis = []
        if 'implement' in rewritten_goal.lower() or 'develop' in rewritten_goal.lower() or 'launch' in rewritten_goal.lower():
            smart_analysis.append("✓ Specific: Clear action verb identified")
        else:
            smart_analysis.append("✗ Specific: Action verb could be clearer")
        
        if any(word in rewritten_goal.lower() for word in ['%', 'percent', '25%', '30%', '40%', '90%', '95%', '98%']):
            smart_analysis.append("✓ Measurable: Specific metrics included")
        else:
            smart_analysis.append("✗ Measurable: Metrics could be more specific")
        
        if 'within' in rewritten_goal.lower() or 'by end' in rewritten_goal.lower() or 'months' in rewritten_goal.lower():
            smart_analysis.append("✓ TimeBound: Clear timeframe specified")
        else:
            smart_analysis.append("✗ TimeBound: Timeframe could be clearer")
        
        feedback_parts.append("\n".join(smart_analysis))
        
        # Add improvement suggestions based on original goal analysis
        feedback_parts.append("\nKey Improvements Made:")
        improvements = []
        
        # Analyze original goal for improvements
        original_goal_lower = goal_desc.lower()
        
        # Check for action verbs
        action_verbs = ['implement', 'develop', 'launch', 'create', 'establish', 'achieve', 'complete', 'deliver', 
                       'optimize', 'enhance', 'streamline', 'automate', 'standardize', 'expand', 'consolidate']
        has_action_verb = any(verb in original_goal_lower for verb in action_verbs)
        if not has_action_verb:
            improvements.append("• Added clear action verb")
        
        # Check for measurable elements
        has_metrics = any(word in original_goal_lower for word in ['%', 'percent', '25%', '30%', '40%', '90%', '95%', '98%', 'increase', 'decrease', 'reduce', 'achieve'])
        if not has_metrics:
            improvements.append("• Included specific metrics and percentages")
        
        # Check for time elements
        has_timeframe = any(word in original_goal_lower for word in ['within', 'by end', 'months', 'quarter', '2025', '2024', 'deadline'])
        if not has_timeframe:
            improvements.append("• Added concrete timeframe")
        
        # Check for relevance
        business_keywords = ['customer', 'team', 'process', 'system', 'quality', 'cost', 'efficiency', 'productivity', 'satisfaction']
        has_relevance = any(word in original_goal_lower for word in business_keywords)
        if not has_relevance:
            improvements.append("• Connected to business impact")
        
        if improvements:
            feedback_parts.append("\n".join(improvements))
        else:
            feedback_parts.append("• Goal already meets most SMART criteria")
    
    # If no rewritten goal could be generated, provide basic feedback
    else:
        feedback_parts.append("Unable to rewrite: Goal description is too vague or incomplete")
        feedback_parts.append("Suggestion: Provide more context about what you want to achieve")
    
    return "\n\n".join(feedback_parts) if feedback_parts else "Goal meets SMART criteria"

def generate_ai_rewritten_goal(row):
    """
    Intelligently transform goals to be fully SMART-compliant while preserving ALL original technical words,
    KPI metrics, and specific meaning - using AI logic to ensure every goal meets all SMART criteria
    """
    goal_desc = str(row.get('Goal Description', '')).strip()
    if not goal_desc or goal_desc.lower() in ['nan', 'none', '']:
        return None
    
    # Preserve the original goal exactly as written
    original_goal = goal_desc
    original_lower = original_goal.lower()
    
    # AI Analysis: Extract key components from the original goal
    goal_components = {
        'subject': None,           # What is being acted upon
        'action': None,            # What action is being taken
        'context': None,           # Business context/domain
        'existing_metrics': None,  # Any existing numbers/percentages
        'time_elements': None,     # Any existing time references
        'technical_terms': []      # All technical keywords to preserve
    }
    
    # Extract technical terms and context (preserve ALL original meaning)
    technical_keywords = []
    if 'customer' in original_lower or 'client' in original_lower:
        technical_keywords.append('customer')
        goal_components['context'] = 'customer'
    if 'team' in original_lower or 'employee' in original_lower or 'staff' in original_lower:
        technical_keywords.append('team')
        goal_components['context'] = 'team'
    if 'process' in original_lower or 'workflow' in original_lower:
        technical_keywords.append('process')
        goal_components['context'] = 'process'
    if 'system' in original_lower or 'platform' in original_lower or 'tool' in original_lower:
        technical_keywords.append('system')
        goal_components['context'] = 'system'
    if 'quality' in original_lower or 'standard' in original_lower:
        technical_keywords.append('quality')
        goal_components['context'] = 'quality'
    if 'cost' in original_lower or 'expense' in original_lower or 'budget' in original_lower:
        technical_keywords.append('cost')
        goal_components['context'] = 'cost'
    if 'efficiency' in original_lower or 'productivity' in original_lower:
        technical_keywords.append('efficiency')
        goal_components['context'] = 'efficiency'
    if 'sales' in original_lower or 'revenue' in original_lower:
        technical_keywords.append('sales')
        goal_components['context'] = 'sales'
    if 'training' in original_lower or 'development' in original_lower:
        technical_keywords.append('training')
    if 'automation' in original_lower or 'digital' in original_lower:
        technical_keywords.append('automation')
    if 'uptime' in original_lower or 'reliability' in original_lower:
        technical_keywords.append('reliability')
    if 'performance' in original_lower:
        technical_keywords.append('performance')
    if 'satisfaction' in original_lower:
        technical_keywords.append('satisfaction')
    if 'support' in original_lower or 'service' in original_lower:
        technical_keywords.append('support')
    if 'compliance' in original_lower:
        technical_keywords.append('compliance')
    if 'defects' in original_lower or 'errors' in original_lower:
        technical_keywords.append('defects')
    if 'target' in original_lower or 'quota' in original_lower:
        technical_keywords.append('target')
    if 'project' in original_lower or 'delivery' in original_lower:
        technical_keywords.append('project')
    
    goal_components['technical_terms'] = technical_keywords
    
    # AI Analysis: Determine the core action from the original goal
    action_verbs = {
        'improve': 'improve', 'better': 'improve', 'enhance': 'enhance', 'optimize': 'optimize',
        'upgrade': 'upgrade', 'develop': 'develop', 'create': 'create', 'build': 'build',
        'design': 'design', 'establish': 'establish', 'launch': 'launch', 'start': 'start',
        'begin': 'begin', 'initiate': 'initiate', 'reduce': 'reduce', 'decrease': 'decrease',
        'minimize': 'minimize', 'cut': 'cut', 'lower': 'lower', 'increase': 'increase',
        'grow': 'grow', 'expand': 'expand', 'boost': 'boost', 'raise': 'raise',
        'maintain': 'maintain', 'sustain': 'sustain', 'keep': 'maintain', 'preserve': 'preserve',
        'achieve': 'achieve', 'complete': 'complete', 'deliver': 'deliver', 'accomplish': 'accomplish',
        'finish': 'complete', 'implement': 'implement', 'streamline': 'streamline', 'standardize': 'standardize'
    }
    
    # Find the most relevant action verb
    detected_action = None
    for verb, action in action_verbs.items():
        if verb in original_lower:
            detected_action = action
            break
    
    goal_components['action'] = detected_action or 'implement'
    
    # AI Analysis: Check for existing measurable elements
    import re
    numbers = re.findall(r'\d+\.?\d*', original_goal)
    percentages = re.findall(r'\d+\.?\d*%', original_goal)
    goal_components['existing_metrics'] = numbers + percentages
    
    # AI Analysis: Check for existing time elements
    time_patterns = ['within', 'by end', 'months', 'quarter', '2025', '2024', 'deadline', 'year', 'week', 'days']
    existing_time = [word for word in time_patterns if word in original_lower]
    goal_components['time_elements'] = existing_time
    
    # AI Logic: Build SMART goal intelligently
    smart_goal_parts = []
    
    # 1. SPECIFIC: Ensure clear action verb
    if not detected_action:
        # AI selects most appropriate action based on context
        if goal_components['context'] == 'customer':
            smart_goal_parts.append("Enhance")
        elif goal_components['context'] == 'team':
            smart_goal_parts.append("Develop")
        elif goal_components['context'] == 'process':
            smart_goal_parts.append("Optimize")
        elif goal_components['context'] == 'system':
            smart_goal_parts.append("Improve")
        elif goal_components['context'] == 'quality':
            smart_goal_parts.append("Enhance")
        elif goal_components['context'] == 'cost':
            smart_goal_parts.append("Reduce")
        elif goal_components['context'] == 'efficiency':
            smart_goal_parts.append("Optimize")
        elif goal_components['context'] == 'sales':
            smart_goal_parts.append("Increase")
        else:
            smart_goal_parts.append("Implement")
    else:
        smart_goal_parts.append(detected_action.title())
    
    # 2. Add the original goal content (preserving ALL technical terms)
    smart_goal_parts.append(original_goal)
    
    # 3. MEASURABLE: Add intelligent metrics if missing
    if not goal_components['existing_metrics']:
        # AI generates context-appropriate measurable metrics
        if goal_components['context'] == 'customer':
            if 'satisfaction' in technical_keywords:
                smart_goal_parts.append("to achieve 90% satisfaction rating")
            elif 'support' in technical_keywords or 'service' in technical_keywords:
                smart_goal_parts.append("to achieve 95% first-call resolution rate")
            else:
                smart_goal_parts.append("to increase engagement scores by 25%")
        elif goal_components['context'] == 'team':
            if 'training' in technical_keywords or 'development' in technical_keywords:
                smart_goal_parts.append("to complete 100% of required training modules")
            elif 'productivity' in technical_keywords or 'performance' in technical_keywords:
                smart_goal_parts.append("to increase team output by 25%")
            else:
                smart_goal_parts.append("to achieve 95% project on-time delivery rate")
        elif goal_components['context'] == 'process':
            if 'automation' in technical_keywords:
                smart_goal_parts.append("to automate 60% of manual processes")
            else:
                smart_goal_parts.append("to reduce processing time by 30%")
        elif goal_components['context'] == 'system':
            if 'reliability' in technical_keywords or 'uptime' in technical_keywords:
                smart_goal_parts.append("to achieve 99.5% system uptime")
            else:
                smart_goal_parts.append("to reduce response times by 40%")
        elif goal_components['context'] == 'quality':
            if 'compliance' in technical_keywords:
                smart_goal_parts.append("to achieve 98% compliance rate")
            elif 'defects' in technical_keywords:
                smart_goal_parts.append("to reduce defects by 50%")
            else:
                smart_goal_parts.append("to achieve 95% quality score")
        elif goal_components['context'] == 'cost':
            smart_goal_parts.append("to reduce operational costs by 15%")
        elif goal_components['context'] == 'efficiency':
            smart_goal_parts.append("to improve overall efficiency by 25%")
        elif goal_components['context'] == 'sales':
            if 'target' in technical_keywords or 'quota' in technical_keywords:
                smart_goal_parts.append("to exceed sales targets by 20%")
            else:
                smart_goal_parts.append("to improve conversion rates by 30%")
        else:
            smart_goal_parts.append("to achieve measurable improvements")
    
    # 4. TIME-BOUND: Add intelligent timeframe if missing
    if not goal_components['time_elements']:
        # AI determines appropriate timeframe based on action type and complexity
        if detected_action in ['implement', 'launch', 'start', 'begin', 'initiate']:
            smart_goal_parts.append("within 3 months")
        elif detected_action in ['develop', 'create', 'build', 'design', 'establish']:
            smart_goal_parts.append("within 4 months")
        elif detected_action in ['optimize', 'enhance', 'improve', 'upgrade', 'streamline']:
            smart_goal_parts.append("within 6 months")
        elif detected_action in ['achieve', 'complete', 'deliver', 'accomplish', 'finish']:
            smart_goal_parts.append("by end of Q3 2025")
        elif detected_action in ['maintain', 'sustain', 'keep', 'preserve']:
            smart_goal_parts.append("throughout the year")
        else:
            smart_goal_parts.append("within 6 months")
    
    # 5. ACHIEVABLE & RELEVANT: Ensure the goal makes business sense
    # The AI has already ensured relevance by preserving original context and technical terms
    
    # Build the final SMART goal
    smart_goal = " ".join(smart_goal_parts)
    
    # AI Quality Check: Ensure grammatical correctness and readability
    smart_goal = smart_goal.strip()
    smart_goal = ' '.join(smart_goal.split())  # Remove double spaces
    
    # Ensure proper sentence structure
    if not smart_goal.endswith('.') and not smart_goal.endswith('!') and not smart_goal.endswith('?'):
        smart_goal += "."
    
    # Fix capitalization
    if smart_goal and smart_goal[0].islower():
        smart_goal = smart_goal[0].upper() + smart_goal[1:]
    
    # Clean up spacing and punctuation
    smart_goal = smart_goal.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
    smart_goal = smart_goal.replace(' to ', ' to ').replace(' within ', ' within ').replace(' by end of ', ' by end of ')
    
    return smart_goal.strip()

def generate_ai_goal_template(domain, role):
    """
    Generate AI-powered goal templates based on domain and role
    """
    templates = {
        'Technology': {
            'Developer': [
                "Implement automated testing framework to achieve 95% code coverage within 3 months",
                "Develop and deploy 3 new microservices to improve system scalability by 40% by Q3",
                "Optimize database queries to reduce response time by 50% within 2 months"
            ],
            'Manager': [
                "Lead team of 8 developers to deliver 5 major features with 90% on-time delivery by Q4",
                "Implement agile best practices to increase team velocity by 25% within 6 months",
                "Reduce technical debt by 30% through systematic refactoring by end of year"
            ]
        },
        'Sales': {
            'Representative': [
                "Achieve 120% of quarterly sales target by closing 15 new enterprise deals by Q3",
                "Increase average deal size by 25% through upselling strategies within 4 months",
                "Maintain 95% customer retention rate while acquiring 20 new customers by Q4"
            ],
            'Manager': [
                "Lead sales team to exceed annual target by 15% through strategic initiatives by year-end",
                "Implement sales training program to improve conversion rates by 30% within 6 months",
                "Expand market presence by entering 3 new territories by Q3"
            ]
        },
        'Marketing': {
            'Specialist': [
                "Increase website traffic by 40% through SEO optimization within 4 months",
                "Launch 3 new marketing campaigns to generate 500 qualified leads by Q3",
                "Improve email open rates from 25% to 35% through A/B testing by Q2"
            ],
            'Manager': [
                "Develop comprehensive marketing strategy to increase brand awareness by 50% by year-end",
                "Implement marketing automation to improve lead nurturing efficiency by 40% within 6 months",
                "Establish partnerships with 5 industry influencers to expand reach by Q4"
            ]
        }
    }
    
    return templates.get(domain, {}).get(role, [
        "Implement strategic initiative to achieve measurable business impact within defined timeframe",
        "Optimize key processes to improve efficiency and effectiveness by end of quarter",
        "Develop and execute plan to meet organizational objectives with clear success metrics"
    ])

# --- Preprocessing ---
smart_cols = SMART_COLUMNS
df["Quality"] = df["SMART_score"].apply(get_quality)
# Enhanced duplicate detection - check for actual duplicates considering Sub Goal ID
def detect_duplicates(df):
    """Detect duplicates considering Goal ID and Sub Goal ID - uses Employee_Duplicate from audit if available"""
    df_copy = df.copy()
    
    # Handle NaN values by filling them with empty string for comparison (needed for both paths)
    df_clean = df_copy.copy()
    df_clean['Goal ID'] = df_clean['Goal ID'].fillna('')
    df_clean['Sub Goal ID'] = df_clean['Sub Goal ID'].fillna('')
    
    # If Employee_Duplicate exists from audit file, use it (single source of truth)
    if "Employee_Duplicate" in df_copy.columns:
        df_copy["Is Duplicate"] = df_copy["Employee_Duplicate"].astype(bool)
        st.info("✅ Using Employee_Duplicate flag from audit file (single source of truth)")
    else:
        # Initialize duplicate flags
        df_copy["Is Duplicate"] = False
        df_copy["Is Sub Goal Duplicate"] = False

    df_copy["Is Duplicate"] = df_clean.duplicated(subset=["Employee ID", "Goal ID", "Sub Goal ID"], keep=False)
    # Also create Employee_Duplicate for consistency
    df_copy["Employee_Duplicate"] = df_copy["Is Duplicate"]
    
    # Additional check for Sub Goal Name duplicates within the same employee and goal
    if "Sub Goal Name" in df_copy.columns:
        for emp_id, emp_group in df_clean.groupby("Employee ID"):
            if len(emp_group) <= 1:
                continue
                
            # Check for Sub Goal Name duplicates within the same employee and goal
            sub_goal_groups = emp_group.groupby(["Goal ID", "Sub Goal Name"])
            for (goal_id, sub_goal), sub_group in sub_goal_groups:
                if len(sub_group) > 1 and pd.notna(sub_goal) and str(sub_goal).strip() != '':
                    df_copy.loc[sub_group.index, "Is Sub Goal Duplicate"] = True
    
    return df_copy

# Apply enhanced duplicate detection
df = detect_duplicates(df)

# Add intelligent duplicate detection for detailed analysis (separate function)
def get_intelligent_duplicates(df):
    """
    Get intelligent duplicate detection results for detailed analysis
    without affecting the main pie chart breakdown
    """
    df_copy = df.copy()
    
    # Normalize goal descriptions for better comparison
    def normalize_text(text):
        if pd.isna(text):
            return ""
        text = str(text).strip().lower()
        text = ' '.join(text.split())
        text = text.replace('.', '').replace(',', '').replace(';', '').replace(':', '')
        text = text.replace('"', '').replace("'", '')
        return text
    
    # Create normalized version for comparison
    df_copy['Goal_Description_Normalized'] = df_copy['Goal Description'].apply(normalize_text)
    
    # Check for exact duplicates in normalized text
    df_copy['Is_Intelligent_Duplicate'] = df_copy.duplicated(subset=['Goal_Description_Normalized'], keep=False)
    
    # Additional validation: check if goals are actually different despite similar text
    def validate_duplicates(group):
        if len(group) <= 1:
            return pd.Series([False] * len(group), index=group.index)
        
        # Get the normalized descriptions for this group
        normalized_descs = group['Goal_Description_Normalized'].tolist()
        original_descs = group['Goal Description'].tolist()
        employee_ids = group['Employee ID'].tolist()
        
        # If all descriptions are exactly the same after normalization, they're true duplicates
        if len(set(normalized_descs)) == 1:
            return pd.Series([True] * len(group), index=group.index)
        
        # Check for high similarity but with meaningful differences
        duplicates = []
        for i, (norm1, orig1, emp1) in enumerate(zip(normalized_descs, original_descs, employee_ids)):
            is_dup = False
            for j, (norm2, orig2, emp2) in enumerate(zip(normalized_descs, original_descs, employee_ids)):
                if i != j:
                    # Check if they're the same employee (shouldn't be marked as duplicate)
                    if emp1 == emp2:
                        continue
                    
                    # Check if normalized texts are identical
                    if norm1 == norm2:
                        # Additional check: are the original texts actually the same?
                        if orig1.strip().lower() == orig2.strip().lower():
                            is_dup = True
                            break
            
            duplicates.append(is_dup)
        
        return pd.Series(duplicates, index=group.index)
    
    # Apply validation to groups of potential duplicates
    duplicate_groups = df_copy[df_copy['Is_Intelligent_Duplicate'] == True].groupby('Goal_Description_Normalized')
    for _, group in duplicate_groups:
        validated_duplicates = validate_duplicates(group)
        df_copy.loc[group.index, 'Is_Intelligent_Duplicate'] = validated_duplicates
    
    # Additional safety check: ensure same employee goals are never marked as duplicates
    df_copy['Is_Intelligent_Duplicate'] = df_copy['Is_Intelligent_Duplicate'] & (df_copy.groupby('Employee ID')['Employee ID'].transform('count') > 1)
    
    return df_copy['Is_Intelligent_Duplicate']
# Calculate missing pillar count with error handling
try:
    df["Missing Pillar Count"] = df[SMART_COLUMNS].sum(axis=1).apply(lambda x: 5 - x)
    df["Has Missing Pillar"] = df["Missing Pillar Count"] > 0
except KeyError as e:
    st.warning(f"⚠️ Missing SMART columns: {e}. Creating default missing pillar count...")
    df["Missing Pillar Count"] = 0  # Default to no missing pillars
    df["Has Missing Pillar"] = False
# Handle missing columns with error handling
try:
    df["Missing Target"] = df["Has_Target"] == False
except KeyError:
    st.warning("⚠️ Has_Target column not found. Creating default values...")
    df["Missing Target"] = False

if "Has_Metric" in df.columns:
    df["Missing Metric"] = df["Has_Metric"] == False
else:
    if "Goal Metric / Measurement Criteria" in df.columns:
        df["Missing Metric"] = df["Goal Metric / Measurement Criteria"].isna() | (df["Goal Metric / Measurement Criteria"].astype(str).str.strip() == "")
    else:
        st.warning("⚠️ Goal Metric / Measurement Criteria column not found. Creating default values...")
        df["Missing Metric"] = False
# Handle missing Employee ID column
if "Employee ID" not in df.columns:
    st.warning("⚠️ Employee ID column not found. Creating default values...")
    df["Employee ID"] = "EMP_" + df.index.astype(str)

# Optimize dataframe operations to avoid fragmentation warnings
df["Employee Goal Count"] = df.groupby("Employee ID")["Employee ID"].transform("count")
df["Goal Load Bucket"] = df["Employee Goal Count"].apply(goal_load_bucket)

# Handle missing Goal Description column
if "Goal Description" not in df.columns:
    st.warning("⚠️ Goal Description column not found. Using Goal KRA as fallback...")
    if "Goal KRA" in df.columns:
        df["Goal Description"] = df["Goal KRA"]
    else:
        df["Goal Description"] = "Goal " + df.index.astype(str)

# Create new columns in batch to avoid fragmentation
new_columns = pd.DataFrame({
    "SMART Feedback": df.apply(generate_smart_feedback, axis=1),
    "AI Rewritten Goal": df.apply(generate_ai_rewritten_goal, axis=1)
})

# Concatenate new columns at once
df = pd.concat([df, new_columns], axis=1)

# --- Sidebar Filters & Info ---
st.sidebar.header("Filters")

# Business Unit filter
bu_options = sorted(df["Business Unit"].dropna().unique())
bu = st.sidebar.multiselect("Business Unit", options=bu_options, default=None)
st.sidebar.markdown(f"*📊 {len(bu_options)} Business Units available*")

# Department filter - cascading based on Business Unit selection
if "Department" in df.columns:
    # If Business Unit is selected, filter departments for those business units
    if bu:
        # Filter data by selected business units
        filtered_for_dept = df[df["Business Unit"].isin(bu)]
        dept_options = sorted(filtered_for_dept["Department"].dropna().unique())
        st.sidebar.markdown(f"*📊 {len(dept_options)} Departments available in selected Business Unit(s)*")
    else:
        # Show all departments if no business unit is selected
        dept_options = sorted(df["Department"].dropna().unique())
        st.sidebar.markdown(f"*📊 {len(dept_options)} Departments available*")
    
    dept = st.sidebar.multiselect("Department", options=dept_options, default=None)
else:
    dept_options = []
    dept = None
    st.sidebar.warning("⚠️ Department column not found in data")

# Domain filter - ensure it's reading the correct column
if "Domain" in df.columns:
    domain_options = sorted(df["Domain"].dropna().unique())
    domain = st.sidebar.multiselect("Domain", options=domain_options, default=None)
    st.sidebar.markdown(f"*📊 {len(domain_options)} Domains available*")
else:
    domain_options = []
    domain = None
    st.sidebar.warning("⚠️ Domain column not found in data")

# Add Manager/Lead filtering
manager_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['manager', 'lead', 'supervisor', 'reporting'])]
if manager_cols:
    # Automatically select the first manager column found
    manager_col = manager_cols[0]
    
    # Extract manager names from "Manager Name (Manager ID)" format
    def extract_manager_name(manager_text):
        if pd.isna(manager_text) or not manager_text:
            return "Unknown"
        manager_str = str(manager_text).strip()
        # Check if format is "Name (ID)"
        if '(' in manager_str and ')' in manager_str:
            # Extract just the name part before the parentheses
            manager_name = manager_str.split('(')[0].strip()
            return manager_name if manager_name else "Unknown"
        else:
            return manager_str
    
    # Create a clean manager name column for analysis
    df['Manager_Name_Clean'] = df[manager_col].apply(extract_manager_name)
    
    # Get available managers based on current business/department filters
    def get_available_managers():
        temp_filtered = df.copy()
        if bu: temp_filtered = temp_filtered[temp_filtered["Business Unit"].isin(bu)]
        if dept: temp_filtered = temp_filtered[temp_filtered["Department"].isin(dept)]
        if domain and "Domain" in df: temp_filtered = temp_filtered[temp_filtered["Domain"].isin(domain)]
        return sorted(temp_filtered['Manager_Name_Clean'].dropna().unique())
    
    # Get managers based on current filters
    available_managers = get_available_managers()
    selected_managers = st.sidebar.multiselect(f"Manager/Lead ({manager_col})", options=available_managers, default=None)
    
    # Show manager count based on current filters
    if bu or dept or (domain and "Domain" in df):
        st.sidebar.markdown(f"**📊 Available Managers:** {len(available_managers)} (filtered by current selection)")
    else:
        st.sidebar.markdown(f"**📊 Total Managers:** {len(available_managers)}")
else:
    # If no manager columns found, try to create one from common patterns
    manager_col = None
    selected_managers = None
    df['Manager_Name_Clean'] = None

# Remove Final Status filter as it's not useful
# final_status = st.sidebar.multiselect("Final Status", options=sorted(df["Final Status"].dropna().unique()) if "Final Status" in df else [], default=None)
all_smart = st.sidebar.checkbox("Show only goals with all SMART pillars", value=False)

# SMART Score filter - applicable to all tables in Tables tab
st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Table-Specific Filters**")
smart_score_min = st.sidebar.slider(
    "SMART Score (Min)", 
    min_value=0.0, 
    max_value=5.0, 
    value=0.0, 
    step=0.1,
    help="Filter employees by minimum average SMART score in Tables tab. Shows only employees whose average SMART score meets or exceeds this value."
)
st.sidebar.markdown(f"*🔍 Shows employees with average SMART score ≥ {smart_score_min:.1f}*")

st.sidebar.markdown(SMART_INFO)

# Show available manager/lead columns info
if manager_cols:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Manager/Lead Column Detected:**")
    st.sidebar.markdown(f"• {manager_col}")
    st.sidebar.markdown("**ℹ️ Format:** Manager Name (Manager ID)")
    st.sidebar.markdown("**🔄 Cascading Filter:** Manager list updates based on Business Unit/Department selection")
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**ℹ️ No manager/lead columns detected**")
    st.sidebar.markdown("Common column names: Manager, Lead, Supervisor, Reporting Manager")

filtered = df.copy()
if bu: filtered = filtered[filtered["Business Unit"].isin(bu)]
if dept: filtered = filtered[filtered["Department"].isin(dept)]
if domain and "Domain" in df: filtered = filtered[filtered["Domain"].isin(domain)]
if selected_managers and manager_col: filtered = filtered[filtered['Manager_Name_Clean'].isin(selected_managers)]
# if final_status and "Final Status" in df: filtered = filtered[filtered["Final Status"].isin(final_status)] # Removed as per edit
if all_smart and "All_SMART_Pillars" in df: filtered = filtered[filtered["All_SMART_Pillars"] == True]

smart_cols = SMART_COLUMNS
def smart_combo_label(row):
    # Handle NaN values in SMART columns
    combo_parts = []
    for col in smart_cols:
        if col in row.index:
            val = row[col]
            # Treat NaN as False, but also handle string values
            if pd.notna(val) and val and str(val).lower() not in ['false', '0', 'nan']:
                combo_parts.append(col[0])  # First letter of column name
    return ''.join(combo_parts) or "None"

filtered["SMART Combo"] = filtered.apply(smart_combo_label, axis=1)
combo_counts = filtered["SMART Combo"].value_counts().reset_index()
combo_counts.columns = ["SMART Pillar Combo", "# Goals"]

total_goals = len(filtered)
def pct_and_n(mask):
    n = mask.sum()
    pct = (n / total_goals * 100) if total_goals else 0
    return f"{pct:.1f} percent ({n})"

top_table = pd.DataFrame({
    "Metric": [
        "Total Goals",
        "High Quality Goals (SMART_score ≥ 4)",
        "Medium Quality Goals (SMART_score = 3)",
        "Low Quality Goals (SMART_score ≤ 2)",
        "Fully SMART (all 5 pillars)",
        "Specific (S)",
        "Measurable (M)",
        "Achievable (A)",
        "Relevant (R)",
        "TimeBound (T)",
    ],
    "Value": [
        str(total_goals),
        pct_and_n(filtered["Quality"] == "High"),
        pct_and_n(filtered["Quality"] == "Medium"),
        pct_and_n(filtered["Quality"] == "Low"),
        pct_and_n(filtered[smart_cols].sum(axis=1) == 5),
        pct_and_n(filtered["Specific"] == True),
        pct_and_n(filtered["Measurable"] == True),
        pct_and_n(filtered["Achievable"] == True),
        pct_and_n(filtered["Relevant"] == True),
        pct_and_n(filtered["TimeBound"] == True),
    ]
})

# --- Helper Functions for Employee-Based Metrics ---
def get_employee_based_quality_metrics(filtered_df, smart_cols):
    """Calculate employee-based quality metrics based on average SMART score per employee"""
    # Calculate average SMART score per employee
    emp_avg_scores = filtered_df.groupby("Employee ID")["SMART_score"].mean()
    
    # Categorize employees based on their average SMART score
    # High quality: average SMART score 4-5
    high_quality_employees = int((emp_avg_scores >= 4).sum())
    
    # Medium quality: average SMART score 2-3
    medium_quality_employees = int(((emp_avg_scores >= 2) & (emp_avg_scores < 4)).sum())
    
    # Low quality: average SMART score 0-1
    low_quality_employees = int((emp_avg_scores < 2).sum())
    
    # Count employees with at least one fully SMART goal (all 5 pillars)
    fully_smart_employees = filtered_df[filtered_df[smart_cols].sum(axis=1) == 5]["Employee ID"].nunique()
    
    return {
        "high_quality": high_quality_employees,
        "medium_quality": medium_quality_employees,
        "low_quality": low_quality_employees,
        "fully_smart": fully_smart_employees
    }

def get_employee_based_bu_metrics(df_for_bu):
    """Calculate Business Unit metrics based on employee count"""
    if 'Business Unit' not in df_for_bu.columns or df_for_bu['Business Unit'].isna().all():
        return pd.DataFrame(columns=['Business Unit', 'Average SMART Score', 'Number of Employees'])
    
    bu_metrics = df_for_bu.groupby("Business Unit").agg({
        'SMART_score': 'mean',
        'Employee ID': 'nunique'
    }).reset_index()
    bu_metrics.columns = ['Business Unit', 'Average SMART Score', 'Number of Employees']
    bu_metrics = bu_metrics.sort_values('Average SMART Score', ascending=False)
    return bu_metrics

def get_employee_based_domain_metrics(df_for_domain):
    """Calculate Domain metrics based on employee count"""
    if 'Domain' not in df_for_domain.columns:
        return pd.DataFrame(columns=['Domain', 'Average SMART Score', 'Number of Employees'])
    
    df_for_domain['SMART_score'] = df_for_domain['SMART_score'].fillna(0)
    df_for_domain['Domain'] = df_for_domain['Domain'].fillna('Unknown')
    
    domain_metrics = df_for_domain.groupby('Domain').agg({
        'SMART_score': 'mean',
        'Employee ID': 'nunique'
    }).reset_index()
    domain_metrics.columns = ['Domain', 'Average SMART Score', 'Number of Employees']
    domain_metrics = domain_metrics.sort_values('Average SMART Score', ascending=False)
    return domain_metrics

# --- Tabs for Dashboard and Tables ---
tabs = st.tabs(["Dashboard", "Tables"])

with tabs[0]:
    st.title("SMART Goal Quality Dashboard - Infographics")
    st.markdown("""
    **All visuals below update based on your filter selections (Business Unit, Department, Domain).**
    """)
    
    # Calculate all metrics
    unique_employees = filtered["Employee ID"].nunique()
    # Calculate employees with filled goals (employees who have actual goal descriptions)
    employees_with_goals = filtered[filtered["Goal Description"].notna() & (filtered["Goal Description"].str.strip() != "")]["Employee ID"].nunique()
    # Calculate employees without filled goals
    employees_without_goals = unique_employees - employees_with_goals
    # Calculate goals with filled descriptions
    goals_filled = filtered[filtered["Goal Description"].notna() & (filtered["Goal Description"].str.strip() != "")].shape[0]
    # Calculate goals without filled descriptions
    goals_not_filled = len(filtered) - goals_filled
    # Calculate employees with SMART score 3 and above
    emp_avg_scores = filtered.groupby("Employee ID")["SMART_score"].mean()
    employees_smart_3_plus = int((emp_avg_scores >= 3).sum())

    avg_goals_per_employee = len(filtered) / unique_employees if unique_employees > 0 else 0
    avg_smart_score = filtered["SMART_score"].mean() if len(filtered) > 0 else 0
    total_goals = len(filtered)
    
    # Calculate metrics based on view mode
    if view_mode == "Employee Based":
        # Employee-based metrics
        emp_quality_metrics = get_employee_based_quality_metrics(filtered, smart_cols)
        high_quality_count = emp_quality_metrics["high_quality"]
        medium_quality_count = emp_quality_metrics["medium_quality"]
        low_quality_count = emp_quality_metrics["low_quality"]
        fully_smart_count = emp_quality_metrics["fully_smart"]
        total_count = unique_employees  # Use employee count as denominator
    else:
        # Goal-based metrics (current behavior)
        high_quality_count = (filtered["Quality"] == "High").sum()
        medium_quality_count = (filtered["Quality"] == "Medium").sum()
        low_quality_count = (filtered["Quality"] == "Low").sum()
        fully_smart_count = (filtered[smart_cols].sum(axis=1) == 5).sum()
        total_count = total_goals  # Use goal count as denominator
    
    # Keep duplicate goals as goal-based (not employee-based)
    duplicate_goals = filtered["Is Duplicate"].astype(bool).sum()
    non_duplicate_goals = len(filtered) - duplicate_goals
    
    # Show manager/lead filter impact
    if manager_col and selected_managers:
        st.info(f"📊 **Filtered by Manager/Lead:** {', '.join(selected_managers)}")
        st.markdown(f"**Showing goals for {len(selected_managers)} selected manager(s)/lead(s)**")
        
    # Combined KPI Dashboard - All metrics in one clean section
    st.markdown("### 📊 Key Performance Indicators")
    
    # First row - Core metrics with vibrant styling (always show 6 columns: Total Employees, Goals Filled, Goals Not Filled, Total Goals, Avg Goals/Employee, Avg SMART Score)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        if theme_mode == "Dark":
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937, #374151); border: 1px solid #4b5563; border-radius: 12px; padding: 20px; margin: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
                <div style="color: #ff6b6b; font-size: 0.9em; font-weight: 600; margin-bottom: 5px;">👥 Total Employees</div>
                <div style="color: #ffffff; font-size: 2em; font-weight: 700;">{unique_employees:,}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric(
                "👥 Total Employees", 
                f"{unique_employees:,}",
                help="Number of unique employees with goals in the system"
            )
    
    # Goals Filled metric - shows employee count in Employee Based view, goal count in Goal Based view
    with col2:
        if view_mode == "Employee Based":
            filled_count = employees_with_goals
            filled_label = "📝 Goals Filled"
            filled_help = "Number of employees who have actually written goal descriptions (not empty/null)"
        else:
            filled_count = goals_filled
            filled_label = "📝 Goals Filled"
            filled_help = "Number of goals with filled descriptions (not empty/null)"
        
        if theme_mode == "Dark":
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937, #374151); border: 1px solid #4b5563; border-radius: 12px; padding: 20px; margin: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
                <div style="color: #ffa726; font-size: 0.9em; font-weight: 600; margin-bottom: 5px;">{filled_label}</div>
                <div style="color: #ffffff; font-size: 2em; font-weight: 700;">{filled_count:,}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric(
                filled_label, 
                f"{filled_count:,}",
                help=filled_help
            )
    
    # Goals Not Filled metric - shows employee count in Employee Based view, goal count in Goal Based view
    with col3:
        if view_mode == "Employee Based":
            not_filled_count = employees_without_goals
            not_filled_label = "⚠️ Goals Not Filled"
            not_filled_help = "Number of employees who have actually not written goal descriptions (empty/null)"
        else:
            not_filled_count = goals_not_filled
            not_filled_label = "⚠️ Goals Not Filled"
            not_filled_help = "Number of goals without filled descriptions (empty/null)"
        
        if theme_mode == "Dark":
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937, #374151); border: 1px solid #4b5563; border-radius: 12px; padding: 20px; margin: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
                <div style="color: #ff6b6b; font-size: 0.9em; font-weight: 600; margin-bottom: 5px;">{not_filled_label}</div>
                <div style="color: #ffffff; font-size: 2em; font-weight: 700;">{not_filled_count:,}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric(
                not_filled_label, 
                f"{not_filled_count:,}",
                help=not_filled_help
            )
    
    # Adjust column assignments
    col_goals, col_avg, col_smart = col4, col5, col6
    
    with col_goals:
        if theme_mode == "Dark":
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937, #374151); border: 1px solid #4b5563; border-radius: 12px; padding: 20px; margin: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
                <div style="color: #4ecdc4; font-size: 0.9em; font-weight: 600; margin-bottom: 5px;">🎯 Total Goals</div>
                <div style="color: #ffffff; font-size: 2em; font-weight: 700;">{total_goals:,}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric(
                "🎯 Total Goals", 
                f"{total_goals:,}",
                help="Total number of goals across all employees and departments"
            )
    with col_avg:
        if theme_mode == "Dark":
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937, #374151); border: 1px solid #4b5563; border-radius: 12px; padding: 20px; margin: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
                <div style="color: #45b7d1; font-size: 0.9em; font-weight: 600; margin-bottom: 5px;">📊 Average Goals per Employee</div>
                <div style="color: #ffffff; font-size: 2em; font-weight: 700;">{avg_goals_per_employee:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric(
                "📊 Average Goals per Employee", 
                f"{avg_goals_per_employee:.1f}",
                help="Average number of goals assigned per employee"
            )
    with col_smart:
        if theme_mode == "Dark":
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937, #374151); border: 1px solid #4b5563; border-radius: 12px; padding: 20px; margin: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
                <div style="color: #96ceb4; font-size: 0.9em; font-weight: 600; margin-bottom: 5px;">⭐ Average SMART Score</div>
                <div style="color: #ffffff; font-size: 2em; font-weight: 700;">{avg_smart_score:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric(
                "⭐ Average SMART Score", 
                f"{avg_smart_score:.2f}",
                help="Average SMART framework score (0-5) across all goals"
            )
    
    # Second row - Quality metrics with vibrant styling (showing percentages)
    high_quality_pct = (high_quality_count / total_count * 100) if total_count > 0 else 0
    medium_quality_pct = (medium_quality_count / total_count * 100) if total_count > 0 else 0
    low_quality_pct = (low_quality_count / total_count * 100) if total_count > 0 else 0
    fully_smart_pct = (fully_smart_count / total_count * 100) if total_count > 0 else 0
    
    # Set labels based on view mode
    if view_mode == "Employee Based":
        high_quality_label = "🏆 Employees with High Quality Goals"
        medium_quality_label = "⚖️ Employees with Medium Quality Goals"
        low_quality_label = "⚠️ Employees with Low Quality Goals"
        fully_smart_label = "✅ Employees with Fully SMART Goals"
        high_quality_help = f"Employees with average SMART score 4-5 ({high_quality_count:,} employees)"
        medium_quality_help = f"Employees with average SMART score 2-3 ({medium_quality_count:,} employees)"
        low_quality_help = f"Employees with average SMART score 0-1 ({low_quality_count:,} employees)"
        fully_smart_help = f"Employees with at least one fully SMART goal ({fully_smart_count:,} employees)"
    else:
        high_quality_label = "🏆 High Quality Goals"
        medium_quality_label = "⚖️ Medium Quality Goals"
        low_quality_label = "⚠️ Low Quality Goals"
        fully_smart_label = "✅ Fully SMART Goals"
        high_quality_help = f"Goals with SMART score 4-5 ({high_quality_count:,} goals)"
        medium_quality_help = f"Goals with SMART score 2-3 ({medium_quality_count:,} goals)"
        low_quality_help = f"Goals with SMART score 0-1 ({low_quality_count:,} goals)"
        fully_smart_help = f"Goals meeting all 5 SMART criteria ({fully_smart_count:,} goals)"
    
    col5, col6, col7, col8, col9 = st.columns(5)
    with col5:
        if theme_mode == "Dark":
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937, #374151); border: 1px solid #4b5563; border-radius: 12px; padding: 20px; margin: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
                <div style="color: #feca57; font-size: 0.9em; font-weight: 600; margin-bottom: 5px;">{high_quality_label}</div>
                <div style="color: #ffffff; font-size: 2em; font-weight: 700;">{high_quality_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric(
                high_quality_label, 
                f"{high_quality_pct:.1f}%",
                help=high_quality_help
            )
    with col6:
        if theme_mode == "Dark":
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937, #374151); border: 1px solid #4b5563; border-radius: 12px; padding: 20px; margin: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
                <div style="color: #ff9f43; font-size: 0.9em; font-weight: 600; margin-bottom: 5px;">{medium_quality_label}</div>
                <div style="color: #ffffff; font-size: 2em; font-weight: 700;">{medium_quality_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric(
                medium_quality_label, 
                f"{medium_quality_pct:.1f}%",
                help=medium_quality_help
            )
    with col7:
        if theme_mode == "Dark":
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937, #374151); border: 1px solid #4b5563; border-radius: 12px; padding: 20px; margin: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
                <div style="color: #ff6b6b; font-size: 0.9em; font-weight: 600; margin-bottom: 5px;">{low_quality_label}</div>
                <div style="color: #ffffff; font-size: 2em; font-weight: 700;">{low_quality_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric(
                low_quality_label, 
                f"{low_quality_pct:.1f}%",
                help=low_quality_help
            )
    with col8:
        if theme_mode == "Dark":
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937, #374151); border: 1px solid #4b5563; border-radius: 12px; padding: 20px; margin: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
                <div style="color: #5f27cd; font-size: 0.9em; font-weight: 600; margin-bottom: 5px;">{fully_smart_label}</div>
                <div style="color: #ffffff; font-size: 2em; font-weight: 700;">{fully_smart_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric(
                fully_smart_label, 
                f"{fully_smart_pct:.1f}%",
                help=fully_smart_help
            )
    
    with col9:
        if theme_mode == "Dark":
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937, #374151); border: 1px solid #4b5563; border-radius: 12px; padding: 20px; margin: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
                <div style="color: #00b894; font-size: 0.9em; font-weight: 600; margin-bottom: 5px;">🎯 Smart Employees</div>
                <div style="color: #ffffff; font-size: 2em; font-weight: 700;">{employees_smart_3_plus:,}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric(
                "🎯 Smart Employees", 
                f"{employees_smart_3_plus:,}",
                help="Number of employees with average SMART score of 3.0 or higher (good quality goals)"
            )
    
    # Third row - Duplicate analysis with vibrant styling
    col10, col11 = st.columns(2)
    with col10:
        if theme_mode == "Dark":
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937, #374151); border: 1px solid #4b5563; border-radius: 12px; padding: 20px; margin: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
                <div style="color: #ff9ff3; font-size: 0.9em; font-weight: 600; margin-bottom: 5px;">🔄 Duplicate Goals</div>
                <div style="color: #ffffff; font-size: 2em; font-weight: 700;">{duplicate_goals:,}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric(
                "🔄 Duplicate Goals", 
                f"{duplicate_goals:,}",
                help="Goals with identical Employee ID, Goal ID, and Sub Goal ID"
            )
    with col11:
        if theme_mode == "Dark":
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937, #374151); border: 1px solid #4b5563; border-radius: 12px; padding: 20px; margin: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
                <div style="color: #00d2d3; font-size: 0.9em; font-weight: 600; margin-bottom: 5px;">✅ Non-Duplicate Goals</div>
                <div style="color: #ffffff; font-size: 2em; font-weight: 700;">{non_duplicate_goals:,}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric(
                "✅ Non-Duplicate Goals", 
                f"{non_duplicate_goals:,}",
                help="Unique goals with different Employee ID, Goal ID, or Sub Goal ID"
            )
    
    st.markdown("---")  # Add separator
    

    
    # Business Unit wise SMART goal score graph (STAGNANT - Not affected by filters)
    st.markdown("#### Business Unit SMART Goal Performance (Overall View)")
    if view_mode == "Employee Based":
        st.info("📊 **This graph shows OVERALL performance across ALL business units based on unique employee count, regardless of current filters**")
    else:
        st.info("📊 **This graph shows OVERALL performance across ALL business units, regardless of current filters**")
    
    # Use the original unfiltered data for business unit performance
    # Handle NaN SMART scores by filling with 0 or using a default value
    df_for_bu = df.copy()
    if 'SMART_score' in df_for_bu.columns:
        df_for_bu['SMART_score'] = df_for_bu['SMART_score'].fillna(0)
    else:
        df_for_bu['SMART_score'] = 0
    
    # Calculate metrics based on view mode
    if view_mode == "Employee Based":
        bu_smart_scores = get_employee_based_bu_metrics(df_for_bu)
        count_column = 'Number of Employees'
        count_label = 'Employee Count'
    else:
        # Ensure Business Unit column exists and has data
        if 'Business Unit' in df_for_bu.columns and not df_for_bu['Business Unit'].isna().all():
            bu_smart_scores = df_for_bu.groupby("Business Unit")["SMART_score"].agg(['mean', 'count']).reset_index()
            bu_smart_scores.columns = ['Business Unit', 'Average SMART Score', 'Number of Goals']
            bu_smart_scores = bu_smart_scores.sort_values('Average SMART Score', ascending=False)
        else:
            st.warning("⚠️ Business Unit column not found or has no data")
            bu_smart_scores = pd.DataFrame(columns=['Business Unit', 'Average SMART Score', 'Number of Goals'])
        count_column = 'Number of Goals'
        count_label = 'Goal Count'
    
    # Display the sorted business unit performance
    if len(bu_smart_scores) > 0:
        st.markdown(f"**Top Performing Business Unit:** {bu_smart_scores.iloc[0]['Business Unit']} (Score: {bu_smart_scores.iloc[0]['Average SMART Score']:.2f})")
        
        # Add data availability note
        total_bus = len(bu_smart_scores)
        st.markdown(f"*📊 Data available for {total_bus} business units*")
    else:
        st.warning("⚠️ No business unit data available")
    
    # Choose color scheme based on theme - Vibrant modern colors
    if theme_mode == "Dark":
        color_scheme = {True: '#ff6b6b', False: '#4ecdc4'}  # Coral & Teal
        template = "plotly_dark"
    else:
        color_scheme = {True: '#e74c3c', False: '#3498db'}  # Red & Blue
        template = "plotly_white"
    
    # Highlight filtered business units if any are selected
    if bu:
        st.markdown(f"**🔍 Currently Filtered Business Units:** {', '.join(bu)}")
        # Add visual indicator for filtered vs unfiltered
        bu_smart_scores['Is_Filtered'] = bu_smart_scores['Business Unit'].isin(bu)
        
        fig_bu_scores = px.bar(
            bu_smart_scores, 
            x='Business Unit', 
            y='Average SMART Score',
            color='Is_Filtered',
            title="Average SMART Score by Business Unit (Overall View)",
            color_discrete_map=color_scheme,
            labels={'Average SMART Score': 'Avg SMART Score', 'Is_Filtered': 'Currently Filtered'},
            template=template,
            hover_data=[count_column]
        )
    else:
        fig_bu_scores = px.bar(
            bu_smart_scores, 
            x='Business Unit', 
            y='Average SMART Score',
            color=count_column,
            title="Average SMART Score by Business Unit (Overall View)",
            color_continuous_scale='Viridis' if theme_mode != "Dark" else 'Turbo' if theme_mode != "Dark" else 'Turbo',
            labels={'Average SMART Score': 'Avg SMART Score', count_column: count_label},
            template=template
        )
    
    fig_bu_scores.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_bu_scores, use_container_width=True)
    
    # Domain wise SMART goal score graph
    st.markdown("#### Domain SMART Goal Performance")
    if len(bu) > 0:
        if view_mode == "Employee Based":
            st.info("📊 **This graph shows Domain performance for the selected Business Unit(s) based on unique employee count**")
        else:
            st.info("📊 **This graph shows Domain performance for the selected Business Unit(s)**")
    else:
        if view_mode == "Employee Based":
            st.info("📊 **This graph shows OVERALL Domain performance across the organization based on unique employee count**")
        else:
            st.info("📊 **This graph shows OVERALL Domain performance across the organization**")
    
    # Use filtered data for domain performance (changes based on business unit filter)
    df_for_domain = filtered.copy()
    
    # Calculate domain metrics based on view mode
    if view_mode == "Employee Based":
        domain_smart_scores = get_employee_based_domain_metrics(df_for_domain)
        count_column = 'Number of Employees'
        count_label = 'Employee Count'
    else:
        # Handle NaN SMART scores and Domain values
        df_for_domain['SMART_score'] = df_for_domain['SMART_score'].fillna(0)
        df_for_domain['Domain'] = df_for_domain['Domain'].fillna('Unknown')
        
        # Calculate domain averages
        domain_smart_scores = df_for_domain.groupby('Domain').agg({
            'SMART_score': ['mean', 'count'],
            'Employee ID': 'nunique'
        }).round(2)
        
        domain_smart_scores.columns = ['Average SMART Score', 'Number of Goals', 'Number of Employees']
        domain_smart_scores = domain_smart_scores.reset_index()
        count_column = 'Number of Goals'
        count_label = 'Goal Count'
    
    # Sort by average SMART score
    domain_smart_scores = domain_smart_scores.sort_values('Average SMART Score', ascending=False)
    
    # Create domain chart
    if len(domain_smart_scores) > 0:
        # Show domain summary metrics ABOVE the chart
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Domains", len(domain_smart_scores))
        with col2:
            best_domain = domain_smart_scores.iloc[0]['Domain']
            best_score = domain_smart_scores.iloc[0]['Average SMART Score']
            st.metric("Highest Performing Domain", f"{best_domain} ({best_score:.2f})")
        
        # Now show the chart
        # Set hover data: when goal-based, show employee count; when employee-based, show nothing extra (count is already the color)
        hover_data_list = []
        if view_mode == "Goal Based" and 'Number of Employees' in domain_smart_scores.columns:
            hover_data_list = ['Number of Employees']
        
        fig_domain_scores = px.bar(
            domain_smart_scores, 
            x='Domain', 
            y='Average SMART Score',
            color=count_column,
            title="Average SMART Score by Domain",
            color_continuous_scale='Viridis' if theme_mode != "Dark" else 'Turbo',
            labels={'Average SMART Score': 'Avg SMART Score', count_column: count_label},
            template=template,
            hover_data=hover_data_list if hover_data_list else None
        )
        
        fig_domain_scores.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_domain_scores, use_container_width=True)
    else:
        st.warning("No domain data available for the selected filters")
    
    st.markdown("#### SMART Pillar Coverage")
    
    # Handle NaN values in SMART columns and calculate percentages
    smart_pillar_data = {}
    smart_pillar_percentages = {}
    
    # Calculate based on view mode
    if view_mode == "Employee Based":
        # Count employees who have at least one goal meeting each pillar
        total_count = filtered["Employee ID"].nunique()
        for col in smart_cols:
            if col in filtered.columns:
                # Count unique employees who have at least one goal with this pillar
                employees_with_pillar = filtered[filtered[col].fillna(False) == True]["Employee ID"].nunique()
                percentage = (employees_with_pillar / total_count * 100) if total_count > 0 else 0
                smart_pillar_data[col] = employees_with_pillar
                smart_pillar_percentages[col] = percentage
            else:
                smart_pillar_data[col] = 0
                smart_pillar_percentages[col] = 0
    else:
        # Goal-based: count goals meeting each pillar
        total_goals = len(filtered)
        for col in smart_cols:
            if col in filtered.columns:
                # Count True values, treating NaN as False
                count = filtered[col].fillna(False).sum()
                percentage = (count / total_goals * 100) if total_goals > 0 else 0
                smart_pillar_data[col] = count
                smart_pillar_percentages[col] = percentage
            else:
                smart_pillar_data[col] = 0
                smart_pillar_percentages[col] = 0
    
    smart_pillar_counts = pd.Series(smart_pillar_data)
    smart_pillar_pct = pd.Series(smart_pillar_percentages)
    
    # Always show the graph, even if data is limited
    if len(smart_pillar_counts) > 0:
        # Create a DataFrame with both counts and percentages for hover data
        pillar_df = pd.DataFrame({
            'Pillar': smart_pillar_counts.index,
            'Percentage': smart_pillar_pct.values,
            'Count': smart_pillar_counts.values
        })
        
        # Set labels and title based on view mode
        if view_mode == "Employee Based":
            y_label = "Percentage of Employees (%)"
            title_text = "Percentage of Employees Meeting Each SMART Pillar"
        else:
            y_label = "Percentage of Goals (%)"
            title_text = "Percentage of Goals Meeting Each SMART Pillar"
        
        fig_smart_pillars = px.bar(
            pillar_df, 
            x='Pillar', 
            y='Percentage', 
            labels={"x": "SMART Pillar", "y": y_label}, 
            title=title_text,
            color='Percentage',
            color_continuous_scale='Viridis' if theme_mode != "Dark" else 'Turbo',
            template=template,
            hover_data=['Count'],
            text='Percentage'  # Add percentage labels on bars
        )
        fig_smart_pillars.update_traces(
            texttemplate='%{text:.1f}%',  # Format percentage labels
            textposition='outside'  # Position labels outside bars
        )
        fig_smart_pillars.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(range=[0, 100])  # Set y-axis to 0-100%
        )
        st.plotly_chart(fig_smart_pillars, use_container_width=True)
        
        # Show data summary with percentages
        avg_coverage = smart_pillar_pct.mean()
        if view_mode == "Employee Based":
            total_count = filtered["Employee ID"].nunique()
            st.info(f"📊 **Data Summary:** Average SMART pillar coverage: {avg_coverage:.1f}% across {total_count} employees")
        else:
            total_goals = len(filtered)
            st.info(f"📊 **Data Summary:** Average SMART pillar coverage: {avg_coverage:.1f}% across {total_goals} goals")
    else:
        st.warning("⚠️ No SMART pillar data available for visualization")
    # Problem breakdown and Goal load side by side
    st.markdown("---")  # Add visual separator
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚨 Breakdown of Goal Problems")
        st.markdown("*Analysis of common issues affecting goal quality*")
        
        # Check if Employee_Duplicate column exists and compare with Is Duplicate
        has_employee_duplicate = "Employee_Duplicate" in filtered.columns
        
        if has_employee_duplicate:
            # Check if the two duplicate fields are the same (convert to boolean first)
            duplicate_same = filtered["Is Duplicate"].astype(bool).equals(filtered["Employee_Duplicate"].astype(bool))
            
            if duplicate_same:
                st.info("✅ **Duplicate Detection:** Both 'Duplicate' and 'Employee Duplicate' fields are identical")
                problem_counts = {
                    "Missing Metrics": filtered["Missing Metric"].mean(),
                    "Missing Targets": filtered["Missing Target"].mean(),
                    "Missing SMART Pillar": filtered["Has Missing Pillar"].mean(),
                    "Duplicate Goals": filtered["Is Duplicate"].astype(bool).mean(),  # Merged into one category
                }
            else:
                # Calculate the difference (convert to boolean first)
                duplicate_diff = (filtered["Is Duplicate"].astype(bool) != filtered["Employee_Duplicate"].astype(bool)).mean()
                
                problem_counts = {
                    "Missing Metrics": filtered["Missing Metric"].mean(),
                    "Missing Targets": filtered["Missing Target"].mean(),
                    "Missing SMART Pillar": filtered["Has Missing Pillar"].mean(),
                    "Exact Goal Duplicates": filtered["Is Duplicate"].astype(bool).mean(),
                    "Employee-Level Duplicates": filtered["Employee_Duplicate"].astype(bool).mean(),
                    "Different Duplicate Types": duplicate_diff,
                }
        else:
            st.info("ℹ️ **Duplicate Detection:** Only 'Duplicate' field available")
            problem_counts = {
                "Missing Metrics": filtered["Missing Metric"].mean(),
                "Missing Targets": filtered["Missing Target"].mean(),
                "Missing SMART Pillar": filtered["Has Missing Pillar"].mean(),
                "Duplicate Goals": filtered["Is Duplicate"].astype(bool).mean(),
            }
        
        # Problem types explanation moved below the pie chart for better layout
        
        # Choose color scheme for pie chart based on theme - Vibrant modern palette
        if theme_mode == "Dark":
            pie_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43', '#ee5a24', '#c44569']
        else:
            pie_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#f1c40f', '#8e44ad', '#16a085']
        
        fig_problems = px.pie(
            names=list(problem_counts.keys()),
            values=[v*100 for v in problem_counts.values()],
            title="Breakdown of Goal Problems (%)",
            color_discrete_sequence=pie_colors
        )
        fig_problems.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_problems, use_container_width=True)
        
        # Add detailed explanation below the pie chart in an expandable section
        st.markdown("---")  # Add separator
        
        with st.expander("📋 **Problem Types Explanation** (Click to expand)", expanded=False):
            if has_employee_duplicate and not duplicate_same:
                st.markdown("""
                **Exact Goal Duplicates:** Identical goal descriptions across different employees
                
                **Employee-Level Duplicates:** Multiple goals for the same employee (may be similar but not identical)
                
                **Different Duplicate Types:** Cases where the two duplicate fields disagree
                """)
            else:
                st.markdown("""
                **Duplicate Goals:** Goals with identical descriptions (exact duplicates)
                """)
            
            # Add duplicate detection confidence information
            st.markdown("**🔍 Detection Method:**")
            st.markdown("""
            **Smart Detection:** Text normalization + Employee validation + Meaningful difference checking
            """)
            
            # Add concise analysis
            st.markdown("**📊 Quick Analysis:**")
            total_problems = sum(problem_counts.values())
            if total_problems > 0:
                # Show key metrics in compact format
                col_x, col_y = st.columns(2)
                with col_x:
                    st.metric("📊 Problem Rate", f"{total_problems*100:.1f}%")
                with col_y:
                    top_problem = max(problem_counts.items(), key=lambda x: x[1])
                    st.metric("⚠️ Top Issue", f"{top_problem[1]*100:.1f}%")
            else:
                st.success("🎉 **Excellent!** No major problems detected")
        
        # Show detailed duplicate analysis in expandable section
        if has_employee_duplicate:
            with st.expander("🔍 **Detailed Duplicate Analysis** (Click to expand)", expanded=False):
                # Create a comparison table
                duplicate_analysis = pd.DataFrame({
                    'Metric': [
                        'Total Goals',
                        'Exact Duplicates (Is Duplicate)',
                        'Employee Duplicates (Employee_Duplicate)',
                        'Both Types of Duplicates',
                        'Only Exact Duplicates',
                        'Only Employee Duplicates',
                        'No Duplicates'
                    ],
                    'Count': [
                        len(filtered),
                        filtered["Is Duplicate"].astype(bool).sum(),
                        filtered["Employee_Duplicate"].astype(bool).sum(),
                        (filtered["Is Duplicate"].astype(bool) & filtered["Employee_Duplicate"].astype(bool)).sum(),
                        (filtered["Is Duplicate"].astype(bool) & ~filtered["Employee_Duplicate"].astype(bool)).sum(),
                        (~filtered["Is Duplicate"].astype(bool) & filtered["Employee_Duplicate"].astype(bool)).sum(),
                        (~filtered["Is Duplicate"].astype(bool) & ~filtered["Employee_Duplicate"].astype(bool)).sum()
                    ],
                    'Percentage': [
                        100.0,
                        (filtered["Is Duplicate"].astype(bool).sum() / len(filtered) * 100) if len(filtered) > 0 else 0,
                        (filtered["Employee_Duplicate"].astype(bool).sum() / len(filtered) * 100) if len(filtered) > 0 else 0,
                        ((filtered["Is Duplicate"].astype(bool) & filtered["Employee_Duplicate"].astype(bool)).sum() / len(filtered) * 100) if len(filtered) > 0 else 0,
                        ((filtered["Is Duplicate"].astype(bool) & ~filtered["Employee_Duplicate"].astype(bool)).sum() / len(filtered) * 100) if len(filtered) > 0 else 0,
                        ((~filtered["Is Duplicate"].astype(bool) & filtered["Employee_Duplicate"].astype(bool)).sum() / len(filtered) * 100) if len(filtered) > 0 else 0,
                        ((~filtered["Is Duplicate"].astype(bool) & ~filtered["Employee_Duplicate"].astype(bool)).sum() / len(filtered) * 100) if len(filtered) > 0 else 0
                    ]
                })
                
                # Format percentages
                duplicate_analysis['Percentage'] = duplicate_analysis['Percentage'].round(1)
                duplicate_analysis['Percentage'] = duplicate_analysis['Percentage'].astype(str) + '%'
                
                st.dataframe(duplicate_analysis, use_container_width=True, hide_index=True)
            
                # Show sample of each type
                if duplicate_same:
                    st.success("✅ Both duplicate fields are identical - no confusion in data")
                else:
                    st.warning("⚠️ Different duplicate fields detected - showing sample data for clarity")
                
                    # Show sample of exact duplicates with intelligent analysis
                    exact_duplicates = filtered[filtered["Is Duplicate"].astype(bool) == True]
                    if len(exact_duplicates) > 0:
                        st.markdown("**📋 Sample of Exact Duplicates:**")
                        
                        # Use intelligent duplicate detection for better analysis
                        intelligent_duplicates = get_intelligent_duplicates(exact_duplicates)
                        true_duplicates = exact_duplicates[intelligent_duplicates]
                        
                        if len(true_duplicates) > 0:
                            # Group duplicates by their normalized description to show related goals
                            def normalize_for_grouping(text):
                                if pd.isna(text):
                                    return ""
                                text = str(text).strip().lower()
                                text = ' '.join(text.split())
                                text = text.replace('.', '').replace(',', '').replace(';', '').replace(':', '')
                                text = text.replace('"', '').replace("'", '')
                                return text
                            
                            true_duplicates_copy = true_duplicates.copy()
                            true_duplicates_copy['Normalized_Desc'] = true_duplicates_copy['Goal Description'].apply(normalize_for_grouping)
                            
                            # Show grouped duplicates
                            duplicate_groups = true_duplicates_copy.groupby('Normalized_Desc')
                            for i, (normalized_desc, group) in enumerate(duplicate_groups):
                                if i >= 3:  # Limit to 3 groups
                                    break
                                
                                st.markdown(f"**Group {i+1} - True Duplicates:**")
                                group_display = group[["Employee Name", "Goal Description"]].reset_index(drop=True)
                                st.dataframe(group_display, use_container_width=True, hide_index=True)
                                
                                # Show the normalized version for verification
                                st.markdown(f"*Normalized text: `{normalized_desc}`*")
                                st.markdown("---")
                            
                            # Add verification note
                            st.info("""
                            **🔍 Intelligent Duplicate Detection:** 
                            These goals have been verified as true duplicates using advanced text normalization and validation.
                            """)
                        else:
                            st.success("✅ **No true duplicates found** - All goals marked as duplicates are actually different when analyzed intelligently")
                        
                        # Add concise verification note
                        st.markdown("**🔍 Verification:** Intelligent detection checks for same text content across different employees")
                    
                    # Show sample of employee duplicates
                    emp_duplicates = filtered[filtered["Employee_Duplicate"].astype(bool) == True]
                    if len(emp_duplicates) > 0:
                        st.markdown("**👥 Sample of Employee Duplicates:**")
                        sample_emp = emp_duplicates[["Employee Name", "Goal Description"]].head(3)
                        st.dataframe(sample_emp, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 📊 Goal Load: Employees by Goal Count Bucket")
        st.markdown("*Distribution of goals across employees*")
        
        # Calculate goal load properly
        goal_load = filtered.drop_duplicates(["Employee ID"])[["Employee ID", "Goal Load Bucket"]]
        
        # Ensure we have the correct bucket labels and values
        load_counts = goal_load["Goal Load Bucket"].value_counts()
        
        # Reindex to ensure all buckets are present with proper order
        expected_buckets = ["<2 Goals", "2-3 Goals", "4-5 Goals", ">5 Goals"]
        load_counts = load_counts.reindex(expected_buckets, fill_value=0)
        
        # Create pie chart with proper data
        fig_load = px.pie(
            names=load_counts.index,
            values=load_counts.values,
            title="Goal Load: Employees by Goal Count Bucket",
            color_discrete_sequence=pie_colors
        )
        fig_load.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_load, use_container_width=True)
        
        # Add concise goal load analysis below the pie chart
        st.markdown("---")  # Add separator
        st.markdown("**📊 Quick Analysis:**")
        
        # Calculate key metrics
        total_employees = len(goal_load)
        avg_goals_per_employee = len(filtered) / total_employees if total_employees > 0 else 0
        most_common_bucket = load_counts.idxmax() if len(load_counts) > 0 else "N/A"
        
        # Display concise information in compact format
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("👥 Total Employees", total_employees)
        with col_b:
            st.metric("🎯 Avg Goals/Employee", f"{avg_goals_per_employee:.1f}")
        
        # Show most common load with color coding
        if most_common_bucket == ">5 Goals":
            st.warning(f"⚠️ **Most Common:** {most_common_bucket} - Consider workload balance")
        elif most_common_bucket == "<2 Goals":
            st.info(f"ℹ️ **Most Common:** {most_common_bucket} - May need goal setting guidance")
        else:
            st.success(f"✅ **Most Common:** {most_common_bucket} - Good goal distribution")
    
    # Add final separator for clean layout
    st.markdown("---")
    
    # Manager/Lead Quality Analysis
    if manager_col and selected_managers:
        st.markdown("#### Goal Quality by Manager/Lead")
        manager_quality = filtered.groupby('Manager_Name_Clean').agg({
            'Goal Description': 'count',
            'SMART_score': 'mean',
            'Specific': 'mean',
            'Measurable': 'mean',
            'Achievable': 'mean',
            'TimeBound': 'mean',
            'Relevant': 'mean'
        }).round(3)
        
        # Add quality columns if they exist, otherwise calculate them
        if 'high_quality' in filtered.columns:
            manager_quality['high_quality'] = filtered.groupby('Manager_Name_Clean')['high_quality'].mean()
        else:
            manager_quality['high_quality'] = 0
            
        if 'low_quality' in filtered.columns:
            manager_quality['low_quality'] = filtered.groupby('Manager_Name_Clean')['low_quality'].mean()
        else:
            manager_quality['low_quality'] = 0
        manager_quality.columns = ['Total_Goals', 'Avg_SMART_Score', 'High_Quality_Rate', 'Low_Quality_Rate',
                                 'Specific_Rate', 'Measurable_Rate', 'Achievable_Rate', 'TimeBound_Rate', 'Relevant_Rate']
        manager_quality = manager_quality.reset_index()
        
        # Add original manager text for reference
        if manager_col in filtered.columns:
            manager_mapping = filtered.groupby('Manager_Name_Clean')[manager_col].first()
            manager_quality['Original_Manager_Text'] = manager_quality['Manager_Name_Clean'].map(manager_mapping)
        
        # Display manager quality table
        st.dataframe(manager_quality, use_container_width=True, hide_index=True)
        
        # Manager quality comparison chart
        fig_manager = px.bar(
            manager_quality, 
            x='Manager_Name_Clean', 
            y='Avg_SMART_Score',
            title=f'Average SMART Score by Manager/Lead',
            color='Total_Goals',
            color_continuous_scale='Viridis' if theme_mode != "Dark" else 'Turbo'
        )
        st.plotly_chart(fig_manager, use_container_width=True)
        
        # SMART pillar comparison across managers
        smart_cols_manager = SMART_COLUMNS
        manager_smart_data = []
        for _, row in manager_quality.iterrows():
            manager_name = row['Manager_Name_Clean']
            for col in smart_cols_manager:
                if col in row:
                    manager_smart_data.append({
                        'Manager': manager_name,
                        'SMART_Pillar': col,
                        'Rate': row[f'{col}_Rate']
                    })
        
        if manager_smart_data:
            manager_smart_df = pd.DataFrame(manager_smart_data)
            fig_manager_smart = px.bar(
                manager_smart_df,
                x='Manager',
                y='Rate',
                color='SMART_Pillar',
                title=f'SMART Pillar Rates by Manager/Lead',
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_manager_smart, use_container_width=True)
    
    # General Manager/Lead Overview (even when none selected)
    if manager_col:
        # Determine if filters are applied
        filters_applied = bu or dept or (domain and "Domain" in df) or (selected_managers and manager_col)
        
        if filters_applied:
            st.markdown("#### Manager/Lead Performance Overview (Filtered View)")
            st.info(f"📊 **Showing manager performance for the current filter selection**")
            
            # Use filtered data for manager analysis
            filtered_manager_quality = filtered.groupby('Manager_Name_Clean').agg({
                'Goal Description': 'count',
                'SMART_score': 'mean'
            }).round(3)
            
            # Add quality columns if they exist, otherwise calculate them
            if 'high_quality' in filtered.columns:
                filtered_manager_quality['high_quality'] = filtered.groupby('Manager_Name_Clean')['high_quality'].mean()
            else:
                filtered_manager_quality['high_quality'] = 0
                
            if 'low_quality' in filtered.columns:
                filtered_manager_quality['low_quality'] = filtered.groupby('Manager_Name_Clean')['low_quality'].mean()
            else:
                filtered_manager_quality['low_quality'] = 0
            filtered_manager_quality.columns = ['Total_Goals', 'Avg_SMART_Score', 'High_Quality_Rate', 'Low_Quality_Rate']
            filtered_manager_quality = filtered_manager_quality.sort_values('Avg_SMART_Score', ascending=False).reset_index()
            
            # Add original manager text for reference
            if manager_col in filtered.columns:
                manager_mapping = filtered.groupby('Manager_Name_Clean')[manager_col].first()
                filtered_manager_quality['Original_Manager_Text'] = filtered_manager_quality['Manager_Name_Clean'].map(manager_mapping)
            
            # Show filter context
            filter_context = []
            if bu: filter_context.append(f"Business Unit: {', '.join(bu)}")
            if dept: filter_context.append(f"Department: {', '.join(dept)}")
            if domain and "Domain" in df: filter_context.append(f"Domain: {', '.join(domain)}")
            if selected_managers: filter_context.append(f"Selected Managers: {', '.join(selected_managers)}")
            
            st.markdown(f"**🔍 Current Filters:** {', '.join(filter_context)}")
            st.markdown(f"**📊 Managers in Filter:** {len(filtered_manager_quality)}")
            
        else:
            st.markdown("#### Overall Manager/Lead Performance Overview")
            st.info("📊 **Showing overall manager performance across all business units**")
            
            # Use unfiltered data for overall analysis
            filtered_manager_quality = df.groupby('Manager_Name_Clean').agg({
                'Goal Description': 'count',
                'SMART_score': 'mean'
            }).round(3)
            
            # Add quality columns if they exist, otherwise calculate them
            if 'high_quality' in df.columns:
                filtered_manager_quality['high_quality'] = df.groupby('Manager_Name_Clean')['high_quality'].mean()
            else:
                filtered_manager_quality['high_quality'] = 0
                
            if 'low_quality' in df.columns:
                filtered_manager_quality['low_quality'] = df.groupby('Manager_Name_Clean')['low_quality'].mean()
            else:
                filtered_manager_quality['low_quality'] = 0
            filtered_manager_quality.columns = ['Total_Goals', 'Avg_SMART_Score', 'High_Quality_Rate', 'Low_Quality_Rate']
            filtered_manager_quality = filtered_manager_quality.sort_values('Avg_SMART_Score', ascending=False).reset_index()
            
            # Add original manager text for reference
            if manager_col in df.columns:
                manager_mapping = df.groupby('Manager_Name_Clean')[manager_col].first()
                filtered_manager_quality['Original_Manager_Text'] = filtered_manager_quality['Manager_Name_Clean'].map(manager_mapping)
        
        # Top and bottom performers
        col1, col2 = st.columns(2)
        with col1:
            if filters_applied:
                st.markdown("**Top 3 Managers by SMART Score (Filtered):**")
            else:
                st.markdown("**Top 3 Managers by SMART Score (Overall):**")
            top_managers = filtered_manager_quality.head(3)
            st.dataframe(top_managers, use_container_width=True, hide_index=True)
        
        with col2:
            if filters_applied:
                st.markdown("**Bottom 3 Managers by SMART Score (Filtered):**")
            else:
                st.markdown("**Bottom 3 Managers by SMART Score (Overall):**")
            bottom_managers = filtered_manager_quality.tail(3)
            st.dataframe(bottom_managers, use_container_width=True, hide_index=True)
        
        # Add Top 5 and Bottom 5 Employees when manager filter is applied
        if manager_col and selected_managers:
            st.markdown("---")
            st.markdown("#### 📊 Employee Performance in Selected Manager's Team")
            
            # Get employees under the selected managers
            manager_employees = filtered[filtered['Manager_Name_Clean'].isin(selected_managers)].copy()
            
            if len(manager_employees) > 0:
                # Calculate employee-level SMART scores
                employee_smart_scores = manager_employees.groupby(['Employee Name', 'Employee ID']).agg({
                    'SMART_score': 'mean',
                    'Goal Description': 'count',
                    'Business Unit': 'first',
                    'Department': 'first',
                    'Domain': 'first'
                }).round(2)
                employee_smart_scores.columns = ['Avg_SMART_Score', 'Total_Goals', 'Business_Unit', 'Department', 'Domain']
                employee_smart_scores = employee_smart_scores.sort_values('Avg_SMART_Score', ascending=False).reset_index()
                
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**Top 5 Employees by SMART Score (Overall):**")
                    top_employees = employee_smart_scores.head(5)
                    st.dataframe(top_employees, use_container_width=True, hide_index=True)
                
                with col4:
                    st.markdown("**Bottom 5 Employees by SMART Score (Overall):**")
                    bottom_employees = employee_smart_scores.tail(5)
                    st.dataframe(bottom_employees, use_container_width=True, hide_index=True)
            else:
                st.info("No employees found under the selected manager(s)")
        
        # Manager summary statistics
        if filters_applied:
            st.markdown("#### Manager/Lead Summary Statistics (Filtered View)")
        else:
            st.markdown("#### Manager/Lead Summary Statistics (Overall View)")
            
        manager_stats = pd.DataFrame({
            'Metric': [
                'Total Unique Managers/Leads',
                'Managers with High Quality Goals (≥4 SMART Score)',
                'Managers with Medium Quality Goals (3 SMART Score)',
                'Managers with Low Quality Goals (≤2 SMART Score)',
                'Average Goals per Manager',
                'Highest Goal Count per Manager',
                'Lowest Goal Count per Manager'
            ],
            'Value': [
                len(filtered_manager_quality),
                len(filtered_manager_quality[filtered_manager_quality['Avg_SMART_Score'] >= 4]),
                len(filtered_manager_quality[filtered_manager_quality['Avg_SMART_Score'] == 3]),
                len(filtered_manager_quality[filtered_manager_quality['Avg_SMART_Score'] <= 2]),
                round(filtered_manager_quality['Total_Goals'].mean(), 1),
                filtered_manager_quality['Total_Goals'].max(),
                filtered_manager_quality['Total_Goals'].min()
            ]
        })
        st.dataframe(manager_stats, use_container_width=True, hide_index=True)

    # SMART Pillar Combination Distribution - Moved to end of Dashboard tab
    st.markdown("---")  # Add visual separator
    st.markdown("#### SMART Pillar Combination Distribution")
    fig_combo = px.bar(
        combo_counts, 
        x="SMART Pillar Combo", 
        y="# Goals", 
        title="Distribution of SMART Pillar Combinations",
        color="# Goals",
        color_continuous_scale='Viridis' if theme_mode != "Dark" else 'Turbo' if theme_mode != "Dark" else 'Turbo',
        template=template
    )
    fig_combo.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_combo, use_container_width=True)

with tabs[1]:
    st.title("SMART Goal Quality Dashboard - Tables")
    
    # Apply SMART Score filter to all tables in Tables tab
    filtered_for_tables = filtered.copy()
    if smart_score_min > 0:
        # Calculate average SMART score per employee
        emp_avg_scores = filtered_for_tables.groupby('Employee ID')['SMART_score'].mean()
        # Get employee IDs that meet the minimum score
        keep_ids = emp_avg_scores[emp_avg_scores >= smart_score_min].index
        # Filter the data to only include employees meeting the criteria
        filtered_for_tables = filtered_for_tables[filtered_for_tables['Employee ID'].isin(keep_ids)]
        st.info(f"🎯 **SMART Score Filter Applied:** Showing only employees with average SMART score ≥ {smart_score_min:.1f} ({len(keep_ids)} employees)")
        st.markdown("---")
    
    # Recalculate summary tables with filtered data
    filtered_for_tables["SMART Combo"] = filtered_for_tables.apply(smart_combo_label, axis=1)
    combo_counts_filtered = filtered_for_tables["SMART Combo"].value_counts().reset_index()
    combo_counts_filtered.columns = ["SMART Pillar Combo", "# Goals"]
    
    total_goals_filtered = len(filtered_for_tables)
    def pct_and_n_filtered(mask):
        n = mask.sum()
        pct = (n / total_goals_filtered * 100) if total_goals_filtered else 0
        return f"{pct:.1f} percent ({n})"
    
    top_table_filtered = pd.DataFrame({
        "Metric": [
            "Total Goals",
            "High Quality Goals (SMART_score ≥ 4)",
            "Medium Quality Goals (SMART_score = 3)",
            "Low Quality Goals (SMART_score ≤ 2)",
            "Fully SMART (all 5 pillars)",
            "Specific (S)",
            "Measurable (M)",
            "Achievable (A)",
            "Relevant (R)",
            "TimeBound (T)",
        ],
        "Value": [
            str(total_goals_filtered),
            pct_and_n_filtered(filtered_for_tables["Quality"] == "High"),
            pct_and_n_filtered(filtered_for_tables["Quality"] == "Medium"),
            pct_and_n_filtered(filtered_for_tables["Quality"] == "Low"),
            pct_and_n_filtered(filtered_for_tables[smart_cols].sum(axis=1) == 5),
            pct_and_n_filtered(filtered_for_tables["Specific"] == True),
            pct_and_n_filtered(filtered_for_tables["Measurable"] == True),
            pct_and_n_filtered(filtered_for_tables["Achievable"] == True),
            pct_and_n_filtered(filtered_for_tables["Relevant"] == True),
            pct_and_n_filtered(filtered_for_tables["TimeBound"] == True),
        ]
    })
    
    # Expandable section for SMART Assessment Criteria
    with st.expander("ℹ️ Click here for SMART Goal Assessment Criteria & Quality Definitions", expanded=False):
        st.markdown("""
        ### 📋 SMART Goal Assessment Criteria
        
        This dashboard evaluates goals based on the **SMART framework** - a widely recognized methodology for setting effective goals.
        Each goal is scored on **5 pillars**, with a total possible score of **5 points**.
        
        ---
        
        #### 🎯 The 5 SMART Pillars
        
        **1. Specific (S)** - Is the goal clear and well-defined?
        - ✅ Contains action verbs like: implement, develop, establish, launch, design, automate, create, enhance, integrate, migrate, deploy, improve, optimize, reduce, increase, etc.
        - ✅ Uses clarity keywords: clear, precise, detailed, focused, unambiguous
        - 📊 **Evaluated from:** Goal Description, Sub Goal Description
        
        **2. Measurable (M)** - Can progress be tracked with metrics?
        - ✅ Contains numbers, percentages, or quantifiable metrics
        - ✅ Includes measurement verbs: increase, reduce, achieve, monitor, track, quantify, measure, analyze, benchmark, validate
        - ✅ Has measurement keywords: number, percent, ratio, amount, metric, KPI, target, progress
        - 📊 **Evaluated from:** Goal Description, Target, Prefix Target, Goal Metric / Measurement Criteria
        
        **3. Achievable (A)** - Is the goal realistic and attainable?
        - ✅ Does NOT contain unrealistic words: all, every, always, never, completely, fully, impossible, perfect
        - ✅ Does NOT use vague urgency: "as soon as possible" or "ASAP"
        - ✅ Includes achievability keywords: realistic, feasible, resources, skills, capability, manageable, attainable, within capacity
        - 📊 **Evaluated from:** Goal Description, Sub Goal Description, Combined goal text
        
        **4. Relevant (R)** - Is the goal aligned with business objectives?
        - ✅ Contains alignment verbs: align with, support, contribute to, drive, advance, enhance, strengthen
        - ✅ Includes business context: customer, team, process, system, quality, cost, efficiency, productivity, business objective, strategic priority
        - ✅ Uses relevance keywords: connected, important, strategic, meaningful, impactful, worthwhile
        - 📊 **Evaluated from:** Goal Description, Sub Goal Description, Goal Metric / Measurement Criteria
        
        **5. Time-Bound (T)** - Does the goal have a clear deadline or timeframe?
        - ✅ Contains time-related verbs: complete by, launch by, deliver within, finalize by, meet by
        - ✅ Includes time references: Q1, Q2, Q3, Q4, FY, quarter, month, week, deadline, timeline, milestone, end of quarter, annual, monthly, quarterly, yearly
        - ✅ Has temporal indicators: within, by, before, after, until, per month, per year, 2024, 2025, 2026
        - 📊 **Evaluated from:** Goal Description, Sub Goal Description, Combined goal text
        
        ---
        
        #### 🏆 Quality Classification
        
        Goals are automatically classified into **3 quality tiers** based on their SMART score:
        
        | Quality Level | SMART Score | Description | What it means |
        |--------------|-------------|-------------|---------------|
        | **🟢 High Quality (Better)** | 4-5 points | Meets 4 or all 5 SMART pillars | Goal is well-defined, measurable, achievable, relevant, and time-bound. Ready for execution! |
        | **🟡 Medium Quality (Good)** | 2-3 points | Meets 2-3 SMART pillars | Goal has potential but needs refinement in specific areas. Review missing pillars. |
        | **🔴 Low Quality (Needs Improvement)** | 0-2 points | Meets 0-2 SMART pillars | Goal lacks clarity and structure. Requires significant improvement. |
        
        ---
        
        #### 📊 Data Sources - Columns from Raw Sheet
        
        The assessment uses the following columns from your uploaded data file:
        
        **Primary Goal Information:**
        - `Goal Description` - Main goal text (primary source for all SMART pillars)
        - `Sub Goal Description` - Additional context for sub-goals
        - `Goal ID` - Unique identifier for each goal
        - `Sub Goal ID` - Identifier for sub-goals
        
        **Measurement & Metrics:**
        - `Target` - Numeric or text target value
        - `Prefix Target` - Target prefix (e.g., "increase by", "reduce to")
        - `Goal Metric / Measurement Criteria` - How success will be measured
        - `Goal Weightage Percentage` - Importance weight of the goal
        
        **Employee & Organizational Data:**
        - `Employee ID` - Unique employee identifier
        - `Employee Name` - Employee's full name
        - `Business Unit` - Business unit/division
        - `Department` - Department within business unit
        - `Domain` / `Function` / `Business` - Additional organizational groupings
        - `Designation` - Job title/role
        - `Manager Name (Manager ID)` - Direct manager/supervisor
        
        **Analysis Outputs:**
        - `SMART_score` - Total score (0-5) calculated by the system
        - `Specific`, `Measurable`, `Achievable`, `Relevant`, `TimeBound` - Individual pillar flags (True/False)
        - `Missing SMART Components` - List of pillars that are not met
        - `Final Status` - Quality classification (Better/Good/Needs Improvement)
        - `AI Rewritten Goal` - AI-generated SMART-compliant version of the goal
        - `SMART Feedback` - Detailed feedback on improvements needed
        
        ---
        
        #### 🔍 How Assessment Works
        
        1. **Combined Text Analysis:** The system creates a comprehensive view by combining Goal Description, Sub Goal Description, Target, and Metrics
        2. **Keyword Matching:** Uses 100+ industry-standard keywords and phrases to detect SMART elements
        3. **Pattern Recognition:** Identifies numeric values, percentages, dates, and time references
        4. **Scoring:** Each pillar gets 1 point if criteria are met, totaling to a 0-5 scale
        5. **Quality Classification:** Final classification based on total SMART score
        
        **Note:** The dashboard performs real-time analysis using the same criteria as the goal audit script for consistency.
        """)
    
    st.markdown("""
    #### All tables and charts below update based on your filter selections (Business Unit, Department, Domain).
    - **% = percent of filtered goals**
    - **(n) = number of goals**
    - **Fully SMART = all 5 pillars met**
    - **SMART Combo = which pillars are met (e.g., SM = Specific+Measurable, SMART = all 5)**
    """)
    
    # --- Employee-level Table (Moved to TOP) ---
    st.markdown("### 👥 Employee-Level Breakdown")
    st.markdown("**Detailed view of all goals with SMART analysis and AI feedback**")
    
    emp_cols = [
        "Employee Name", "Employee ID", "Business Unit", "Department", "Domain", "Designation", "Goal Description", "AI Rewritten Goal", "SMART_score", "Quality",
        "All_SMART_Pillars", "Employee_Duplicate",
        "Specific", "Measurable", "Achievable", "Relevant", "TimeBound"
    ]
    
    # Add manager column if available
    if 'Manager_Name_Clean' in filtered.columns:
        emp_cols.insert(6, 'Manager_Name_Clean')  # Insert after Designation
    
    # Add goal not filled column if available
    if 'Goal_Not_Filled' in filtered.columns:
        emp_cols.append('Goal_Not_Filled')
    
    # Add manager columns if available
    if manager_col:
        emp_cols.insert(6, manager_col)  # Insert original manager column after Designation
    
    emp_cols = [c for c in emp_cols if c in filtered.columns]
    
    # Add explanation for the enhanced columns
    st.markdown("""
    **AI Rewritten Goal:** Complete SMART-compliant goals rewritten by AI analysis
    
    **SMART Feedback & Analysis:** Detailed breakdown of improvements and SMART pillar compliance
    """)
    
    # Optional filtering by SMART pillars
    st.markdown("**Filter by SMART Pillar (Optional):**")
    smart_pillars = ["Specific", "Measurable", "Achievable", "Relevant", "TimeBound"]
    filter_pillar = st.selectbox(
        "Show only goals missing this SMART pillar:", 
        ["Show All Goals"] + smart_pillars,
        key="smart_pillar_filter_1"
    )
    
    # Data loading controls
    col_la, col_lb, _ = st.columns([1, 1, 2])
    with col_la:
        load_all_rows = st.checkbox("Load all rows", value=False, help="Disable safety caps and load the full dataset")
    with col_lb:
        max_rows_cap = st.number_input("Max rows (when not loading all)", min_value=100, max_value=2000000, value=1000, step=100)
    
    # Apply filtering only if a specific pillar is selected
    if filter_pillar == "Show All Goals":
        if load_all_rows:
            filtered_emp = filtered_for_tables.copy()
        else:
            if len(filtered_for_tables) > max_rows_cap:
                filtered_emp = filtered_for_tables.head(int(max_rows_cap))
                st.warning(f"⚠️ Performance optimization: Showing first {int(max_rows_cap)} goals out of {len(filtered_for_tables)} total goals. Enable 'Load all rows' to view everything.")
            else:
                filtered_emp = filtered_for_tables.copy()
        st.info(f"📊 Showing {len(filtered_emp)} goals with SMART feedback")
    else:
        filtered_emp = filtered_for_tables[filtered_for_tables[filter_pillar] == False]
        # Limit filtered results for performance unless Load all
        if not load_all_rows:
            if len(filtered_emp) > max_rows_cap:
                filtered_emp = filtered_emp.head(int(max_rows_cap))
                st.warning(f"⚠️ Performance optimization: Showing first {int(max_rows_cap)} filtered goals out of {len(filtered_for_tables[filtered_for_tables[filter_pillar] == False])} total filtered goals. Enable 'Load all rows' to view everything.")
        st.info(f"🔍 Showing {len(filtered_emp)} goals missing the '{filter_pillar}' SMART pillar")

    
    # Create a more readable version of the table with better formatting
    if emp_cols:
        with st.spinner("🔄 Processing data for display..."):
            # Create a copy for display with better formatting - use filtered_emp after filtering
            display_df = filtered_emp[emp_cols].copy()

        # Prepare combined text once for explanation alignment
        working_emp = filtered_emp.copy()
        if 'Combined_Text' not in working_emp.columns:
            working_emp['Combined_Text'] = working_emp.apply(create_combined_text, axis=1)

        # Guarantee SMART_score stays in lockstep with pillar booleans
        if set(smart_pillars).issubset(working_emp.columns):
            working_emp['SMART_score'] = working_emp[smart_pillars].fillna(False).astype(bool).astype(int).sum(axis=1)
        if set(smart_pillars).issubset(display_df.columns):
            display_df['SMART_score'] = display_df[smart_pillars].fillna(False).astype(bool).astype(int).sum(axis=1)

        # Use filtered_emp to get all columns needed for explanations
        display_df['Combined Analysis Text'] = working_emp.loc[display_df.index, 'Combined_Text']
        
        # Add combined SMART explanation column
        explanations_series = working_emp.loc[display_df.index].apply(explain_smart_score, axis=1)
        display_df['SMART Explanation'] = explanations_series.apply(
            lambda e: "\n".join([f"{pillar}: {expl}" for pillar, expl in e.items()])
        )
        # Ensure SMART_score matches the explanation/flags exactly
        score_from_explanation = explanations_series.apply(
            lambda e: sum(1 for expl in e.values() if isinstance(expl, str) and expl.strip().startswith("✓"))
        )
        display_df['SMART_score'] = score_from_explanation
        working_emp.loc[display_df.index, 'SMART_score'] = score_from_explanation.values
        
        # Move Combined Analysis Text to after Goal Description
        cols_order = list(display_df.columns)
        if 'Combined Analysis Text' in cols_order:
            cols_order.remove('Combined Analysis Text')
            goal_idx = cols_order.index('Goal Description') if 'Goal Description' in cols_order else 0
            cols_order.insert(goal_idx + 1, 'Combined Analysis Text')
            display_df = display_df[cols_order]
        
        # Move SMART Explanation after SMART_score
        if 'SMART Explanation' in display_df.columns and 'SMART_score' in display_df.columns:
            cols = list(display_df.columns)
            cols.remove('SMART Explanation')
            idx = cols.index('SMART_score')
            cols.insert(idx + 1, 'SMART Explanation')
            display_df = display_df[cols]
        
        # Format SMART Feedback column for better readability (limit heavy formatting unless Load all)
        if 'SMART Feedback' in display_df.columns:
            if load_all_rows or len(display_df) <= 100:
                    # Split long feedback into multiple lines
                    def format_feedback(feedback):
                        if pd.isna(feedback):
                            return ""
                        feedback_str = str(feedback)
                        # Replace separators with line breaks for better readability
                        formatted = feedback_str.replace(" | ", "\n\n")
                        return formatted
                    display_df['SMART Feedback'] = display_df['SMART Feedback'].apply(format_feedback)
            else:
                st.info("📊 Performance note: SMART Feedback formatting applied to first 100 rows for faster loading. Enable 'Load all rows' to format everything.")
                idx = display_df.head(100).index
                display_df.loc[idx, 'SMART Feedback'] = display_df.loc[idx, 'SMART Feedback'].apply(
                    lambda x: (str(x).replace(" | ", "\n\n") if pd.notna(x) else "")
                )
        
    # Color coding for SMART columns
    def highlight_smart(val):
        if val is True:
            return 'background-color: #d4edda'  # green
        elif val is False:
            return 'background-color: #f8d7da'  # red
        return ''
    
    # Color coding for blank goal descriptions
    def highlight_blank_goals(val):
        if pd.isna(val) or str(val).strip() == '' or str(val).strip().lower() == 'nan':
            return 'background-color: #fff3cd; color: #856404; font-weight: bold;'  # warning yellow
        return ''
    
    # Style the SMART Feedback column to make it more readable
    def style_smart_feedback(val):
        if pd.isna(val):
            return ''
        val_str = str(val)
        if '🔴' in val_str:
            return 'background-color: #fff3cd; color: #856404; font-weight: bold; max-width: 400px; white-space: pre-wrap;'  # warning yellow
        elif '✅' in val_str:
            return 'background-color: #d4edda; color: #155724; font-weight: bold; max-width: 400px; white-space: pre-wrap;'  # success green
        elif '💡' in val_str:
            return 'background-color: #e2e3e5; color: #383d41; font-style: italic; max-width: 400px; white-space: pre-wrap;'  # info gray
        elif '🤖' in val_str:
            return 'background-color: #d1ecf1; color: #0c5460; font-weight: bold; max-width: 400px; white-space: pre-wrap;'  # AI blue
        return 'max-width: 400px; white-space: pre-wrap;'
    
    # Apply styling
    styled = display_df.style.map(highlight_smart, subset=smart_pillars)
    
    # Apply styling to Goal Description column for blank goals
    if 'Goal Description' in display_df.columns:
        styled = styled.map(highlight_blank_goals, subset=['Goal Description'])
    
    # Apply styling to SMART Feedback column
    if 'SMART Feedback' in display_df.columns:
        styled = styled.map(style_smart_feedback, subset=['SMART Feedback'])
    
    # Add tooltip information for SMART columns
    st.markdown("""
    **📋 Combined Analysis Text:** Shows all fields (Goal Description, Sub Goal, Target, Metric) used for SMART scoring, separated by " | "
    
    **🧐 SMART Explanation:** Detailed breakdown of why each SMART pillar scored the way it did.
    """)
    
    # Display the styled table with explanations columns
    try:
        st.dataframe(styled, use_container_width=True, hide_index=True, height=400)
    except Exception as e:
        st.error(f"Error displaying styled table: {e}")
        st.info("Showing raw data instead...")
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
    
    # CSV Download Button - Optimized for large datasets
    st.markdown("---")
    col_dl1, col_dl2, col_dl3 = st.columns([2, 2, 2])
    
    with col_dl1:
        st.markdown(f"**📥 Download Options**")
        st.info(f"📊 **{len(filtered_emp):,} goals** ready for download")
    
    with col_dl2:
        # Prepare CSV data efficiently - use a function that processes on-demand
        def prepare_csv_data(df_to_export, source_df=None):
            """Prepare CSV data without heavy formatting for faster export"""
            export_df = df_to_export.copy()
            # Use original SMART Feedback if available (before formatting)
            if 'SMART Feedback' in export_df.columns and source_df is not None and 'SMART Feedback' in source_df.columns:
                try:
                    # Match by index to get original unformatted feedback
                    export_df['SMART Feedback'] = source_df.loc[export_df.index, 'SMART Feedback'].values
                except:
                    pass  # Keep formatted version if matching fails
            # Convert to CSV efficiently
            return export_df.to_csv(index=False).encode('utf-8')
        
        # Download button for displayed data
        try:
            csv_data = prepare_csv_data(display_df, filtered_emp)
            st.download_button(
                label=f"📥 Download Displayed ({len(display_df):,} rows)",
                data=csv_data,
                file_name=f"employee_goals_displayed_{len(display_df)}_rows_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help=f"Download the {len(display_df):,} goals currently displayed in the table above",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error preparing download: {e}")
    
    with col_dl3:
        # Always show download for full filtered dataset
        try:
            # Prepare full dataset export (use original filtered_emp, not display_df)
            full_export_df = filtered_emp[emp_cols].copy()
            full_csv_data = full_export_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label=f"📥 Download ALL ({len(filtered_emp):,} rows)",
                data=full_csv_data,
                file_name=f"employee_goals_complete_{len(filtered_emp)}_rows_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help=f"Download all {len(filtered_emp):,} goals matching current filters (complete dataset)",
                use_container_width=True
            )
            if not load_all_rows and len(filtered_emp) > max_rows_cap:
                st.caption(f"💡 Full dataset has {len(filtered_emp):,} rows (more than displayed)")
        except Exception as e:
            st.error(f"Error preparing full download: {e}")
            st.info("Try enabling 'Load all rows' checkbox above")
    
    # Add data availability note
    st.markdown("---")
    st.markdown("**📊 Data Availability Note**")
    
    # Calculate data availability statistics
    total_employees = filtered['Employee ID'].nunique()
    employees_with_complete_data = filtered.groupby('Employee ID').apply(
        lambda x: all(col in x.columns and not x[col].isna().all() for col in ['Specific', 'Measurable', 'Achievable', 'Relevant', 'TimeBound']),
        include_groups=False
    ).sum()
    
    # Calculate blank goals count
    blank_goals_count = filtered['Goal_Not_Filled'].sum() if 'Goal_Not_Filled' in filtered.columns else 0
    total_goals_count = len(filtered)
    
    st.markdown(f"""
    *📈 **Data Coverage:** {employees_with_complete_data}/{total_employees} employees have complete SMART pillar data*  
    *📝 **Blank Goals:** {blank_goals_count}/{total_goals_count} goals marked as 'Not Filled' ({blank_goals_count/total_goals_count*100:.1f}%)*  
    *🔄 **Data Processing:** All SMART analysis based on available goal descriptions and metrics*
    """)
    
    # Highlight blank goals if any exist
    if blank_goals_count > 0:
        st.warning(f"⚠️ **{blank_goals_count} goals have blank descriptions** - These are highlighted in yellow in the table above")
        
        # Show breakdown of blank goals by department/business unit
        if len(filtered) > 0:
            blank_goals_df = filtered[filtered['Goal_Not_Filled'] == True]
            if len(blank_goals_df) > 0:
                st.markdown("**📊 Blank Goals Breakdown:**")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if 'Department' in blank_goals_df.columns:
                        dept_blank = blank_goals_df['Department'].value_counts().head(5)
                        st.markdown("**By Department:**")
                        st.dataframe(dept_blank, use_container_width=True)
                
                with col_b:
                    if 'Business Unit' in blank_goals_df.columns:
                        bu_blank = blank_goals_df['Business Unit'].value_counts().head(5)
                        st.markdown("**By Business Unit:**")
                        st.dataframe(bu_blank, use_container_width=True)
    
    # Add simple AI Goal Writing Assistant
    st.markdown("---")
    st.markdown("**🎯 AI Goal Writing Assistant**")
    st.markdown("Get personalized SMART goal templates and suggestions based on your role and domain!")
    st.info("⚡ **Performance Tip**: Use filters to reduce data size for faster loading. The dashboard automatically limits large datasets for optimal performance.")
    
    # Create simple AI assistant
    ai_col1, ai_col2 = st.columns(2)
    
    with ai_col1:
        selected_domain = st.selectbox(
            "Select your domain:",
            ["Technology", "Sales", "Marketing", "Operations", "Finance", "HR", "Other"],
            key="ai_domain_select_1"
        )
        
        selected_role = st.selectbox(
            "Select your role level:",
            ["Individual Contributor", "Team Lead", "Manager", "Director", "Executive"],
            key="ai_role_select_1"
        )
    
    with ai_col2:
        # Use session state to store generated templates
        if 'ai_templates' not in st.session_state:
            st.session_state.ai_templates = []
        
        if st.button("🤖 Generate AI Goal Templates", type="primary", key="generate_templates_btn_1"):
            # Generate AI templates
            ai_templates = generate_ai_goal_template(selected_domain, selected_role)
            st.session_state.ai_templates = ai_templates
            st.session_state.show_templates = True
            st.rerun()
        
        # Display templates if they exist
        if st.session_state.get('show_templates', False) and st.session_state.ai_templates:
            st.success("✨ AI-generated SMART goal templates for you!")
            
            for i, template in enumerate(st.session_state.ai_templates, 1):
                st.markdown(f"**Template {i}:** {template}")
            
            # Add a button to clear templates
            if st.button("🗑️ Clear Templates", key="clear_templates_btn_1"):
                st.session_state.ai_templates = []
                st.session_state.show_templates = False
                st.rerun()
    
    # Add simple writing tips
    st.markdown("**🧠 Writing Tips:**")
    st.markdown("""
    - **Start with Action Verbs**: Implement, Develop, Optimize, Launch
    - **Make it Measurable**: Include numbers, percentages, or specific metrics
    - **Set Timeframes**: Use quarters, months, or specific deadlines
    - **Ensure Relevance**: Connect to business objectives
    """)
    
    # --- Summary Tables (Moved below Employee breakdown) ---
    st.markdown("---")
    st.markdown("### 📊 Summary Tables")
    
    st.markdown("#### Overall SMART Goal Statistics")
    st.dataframe(top_table_filtered, use_container_width=True, hide_index=True)
    
    st.markdown("#### SMART Pillar Combination Table (PnC)")
    st.dataframe(combo_counts_filtered, use_container_width=True, hide_index=True)


# Add enhanced CSS for better visual appearance
st.markdown("""
<style>
/* Enhanced dashboard styling with modern colors */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

/* Modern page background */
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* Enhanced text colors */
p, div, span {
    color: #2c3e50;
}

/* Modern info and warning boxes */
.stAlert[data-baseweb="notification"] {
    border-radius: 15px;
    border: 2px solid rgba(102, 126, 234, 0.2);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15);
}

/* Enhanced selectbox styling */
.stSelectbox > div > div {
    border-radius: 10px;
    border: 2px solid rgba(102, 126, 234, 0.2);
    transition: all 0.3s ease;
}

.stSelectbox > div > div:hover {
    border-color: rgba(102, 126, 234, 0.5);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
}

/* Improved KPI cards with modern gradient */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    border: 2px solid rgba(255,255,255,0.3);
    transition: all 0.3s ease;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
}

[data-testid="metric-container"] label {
    color: white !important;
    font-weight: 700;
    font-size: 1.1rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

[data-testid="metric-container"] [data-testid="metric-value"] {
    color: white !important;
    font-size: 2.5rem !important;
    font-weight: 800;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

/* Enhanced table styling with modern colors */
.stDataFrame {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
    border: 2px solid rgba(102, 126, 234, 0.1);
}

.stDataFrame table {
    border-collapse: separate;
    border-spacing: 0;
}

.stDataFrame th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 700;
    padding: 16px 20px;
    text-align: left;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    font-size: 1.1rem;
}

.stDataFrame td {
    padding: 16px 20px;
    border-bottom: 2px solid #f0f2f6;
    transition: all 0.2s ease;
}

.stDataFrame tr:hover {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    transform: scale(1.01);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
}

/* Improve readability of SMART Feedback column */
[data-testid="stDataFrame"] td {
    max-width: 400px;
    white-space: pre-wrap;
    word-wrap: break-word;
    line-height: 1.4;
}

/* Enhanced chart styling with modern colors */
[data-testid="stPlotlyChart"] {
    border-radius: 20px;
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    padding: 1.5rem;
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border: 2px solid rgba(102, 126, 234, 0.1);
    transition: all 0.3s ease;
}

[data-testid="stPlotlyChart"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.25);
}

/* Better section headers with modern colors */
h1, h2, h3, h4 {
    color: #2c3e50;
    font-weight: 700;
    margin-bottom: 1.5rem;
    text-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Improved sidebar with modern colors */
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem;
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    border: 2px solid rgba(102, 126, 234, 0.1);
}

/* Enhanced expander styling with modern colors */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    font-weight: 700;
    padding: 1.2rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}

.streamlit-expanderHeader:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
}

/* Better info boxes with modern colors */
.stAlert {
    border-radius: 15px;
    border: none;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15);
    border: 2px solid rgba(102, 126, 234, 0.1);
}

/* Improved button styling with modern colors */
.stButton > button {
    border-radius: 30px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    color: white;
    font-weight: 700;
    padding: 0.8rem 2.5rem;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    transition: all 0.3s ease;
    font-size: 1.1rem;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
}

/* Enhanced tab styling with modern colors */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 15px;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border: 2px solid rgba(102, 126, 234, 0.1);
    color: #6c757d;
    font-weight: 600;
    padding: 0.8rem 1.5rem;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
    transform: translateY(-2px);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-color: #667eea;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
}
</style>
""", unsafe_allow_html=True)

st.caption("Dashboard auto-updates if goals_output.xlsx is updated. Add new columns to the Excel file as needed; the dashboard will not break.") 