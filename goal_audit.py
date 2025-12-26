import sys
import argparse
import re
import pandas as pd
from config import get_input_file, get_output_file
from smart_rules import (
    EMPTY_METRIC_VALUES,
    ACHIEVABLE_NEG,
    SMART_KEYWORDS,
    SMART_VERBS,
    TIME_REGEX_PATTERNS,
    create_combined_text,
    explain_smart_score,
    has_achievable,
    has_countable_number,
    has_measurable,
    has_relevant,
    has_specific,
    has_timebound,
    is_empty_value,
)

CURRENCY = r"(₹|\$|€|£)"
PCT = r"(\d{1,3}(?:[.,]\d{1,2})?\s*%|\b\d{1,3}\s*percent\b|\bpercentage points\b|\bpp\b|\bbps\b)"
NUM = r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|\d+)"
RANGE = rf"({NUM}\s*(?:-|to|–|—|~)\s*{NUM})"
RATIO = r"(\d+\s*/\s*\d+|\b\d+\s*:\s*\d+)"
FREQ = r"\b(daily|weekly|biweekly|fortnightly|monthly|quarterly|yearly|per\s+(day|week|month|quarter|year))\b"
COMPARATORS = r"\b(>=|<=|>|<|at least|no more than|not less than|reduce to|increase to|improve to|cap at|floor at)\b"


MEASURABLE_PATTERNS = re.compile(
    rf"{PCT}|{CURRENCY}\s*{NUM}|{NUM}\s*{CURRENCY}|{RANGE}|{RATIO}|{COMPARATORS}|{FREQ}",
    re.IGNORECASE
)

def has_measurable(text: str) -> bool:
    txt = str(text)
    if not txt or txt.strip().lower() == 'nan':
        return False
    txt_lower = txt.lower()
    has_countable = has_countable_number(txt_lower)
    
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
    has_aop_target = any(re.search(pattern, txt_lower, re.IGNORECASE) for pattern in aop_patterns)
    if has_aop_target:
        return True  # AOP targets are always measurable
    
    if MEASURABLE_PATTERNS.search(txt):
        if not re.search(r'\d', txt_lower) and not any(
            w in txt_lower for w in ['%', 'percent', 'ratio', 'growth of', 'increase to', 'reduce to']
        ) and not has_countable:
            return False
        return True
    found_verb = any(v in txt_lower for v in SMART_VERBS['Measurable'])
    found_kw = any(k in txt_lower for k in SMART_KEYWORDS['Measurable'])
    if not re.search(r'\d', txt_lower) and not any(
        w in txt_lower for w in ['%', 'percent', 'ratio', 'growth of', 'increase to', 'reduce to']
    ) and not has_countable:
        return False
    if found_verb or found_kw:
        return True
    return has_countable


def has_achievable(text: str) -> bool:
    txt = str(text).lower()
    if not txt or txt == 'nan':
        return False
    if any(w in txt for w in ACHIEVABLE_NEG):
        return False
    if "as soon as possible" in txt or "asap" in txt:
        return False
    found_verb = any(v in txt for v in SMART_VERBS['Achievable'])
    found_kw = any(k in txt for k in SMART_KEYWORDS['Achievable'])
    phased = any(x in txt for x in ['phased', 'pilot', 'mvp', 'stage', 'incremental'])
    return found_verb or found_kw or phased


# has_relevant is now imported from smart_rules.py with enhanced Business Unit and Domain support

# TIME_PATTERNS is now using TIME_REGEX_PATTERNS from smart_rules
TIME_PATTERNS = re.compile('|'.join(TIME_REGEX_PATTERNS), re.IGNORECASE | re.VERBOSE)

def has_target(text: str) -> bool:
    """Stricter target detection: requires numbers with units/comparators/context, not just any number"""
    if is_empty_value(text):
        return False
    s = str(text).strip()
    
    # Exclude common non-target numbers (years, quarters, IDs)
    s_lower = s.lower()
    if re.search(r'\b(20\d{2}|q[1-4]|h[12]|fy\d{2,4})\b', s_lower):
        # Only count if it's part of a target phrase, not standalone
        if not any(phrase in s_lower for phrase in ['target', 'achieve', 'reach', 'by', 'increase', 'reduce', 'improve']):
            # Check if it's a standalone year/quarter - exclude it
            if re.match(r'^\s*(20\d{2}|q[1-4]|h[12]|fy\d{2,4})\s*$', s_lower):
                return False
    
    # Require numbers with units, comparators, or target context
    # Percentage, currency, ratios, ranges
    if re.search(rf"{PCT}|{CURRENCY}\s*{NUM}|{NUM}\s*{CURRENCY}|{RANGE}|{RATIO}", s, re.IGNORECASE):
        return True
    
    # Comparators with numbers
    if re.search(rf"{COMPARATORS}.*{NUM}|{NUM}.*{COMPARATORS}", s, re.IGNORECASE):
        return True
    
    # Numbers with target/measurement context words
    target_context = ['target', 'achieve', 'reach', 'exceed', 'attain', 'goal', 'increase to', 'reduce to', 
                     'improve to', 'decrease to', 'at least', 'no more than', 'minimum', 'maximum', 'baseline']
    if any(ctx in s_lower for ctx in target_context) and re.search(r'\d+', s):
        return True
    
    # Frequency patterns (per month, per year, etc.)
    if re.search(rf"{FREQ}", s, re.IGNORECASE):
        return True
    
    return False

def has_metric(text: str) -> bool:
    return not is_empty_value(text)


def detect_exact_dup(df: pd.DataFrame) -> pd.Series:
    # Check for duplicates based on Employee ID, Goal ID, and Sub Goal ID
    # If Goal ID is same but Sub Goal ID is different, it's NOT a duplicate
    # Handle NaN values by filling them with empty string for comparison
    df_clean = df.copy()
    df_clean['Goal ID'] = df_clean['Goal ID'].fillna('')
    df_clean['Sub Goal ID'] = df_clean['Sub Goal ID'].fillna('')
    return df_clean.duplicated(subset=["Employee ID", "Goal ID", "Sub Goal ID"], keep=False)

def compute_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Create combined text column for evaluation
    df['Combined_Text'] = df.apply(create_combined_text, axis=1)
    df['Combined Analysis Text'] = df['Combined_Text']
    
    # Apply SMART checks
    df['Specific'] = df['Combined_Text'].apply(has_specific)
    df['Measurable'] = df['Combined_Text'].apply(has_measurable)
    df['Achievable'] = df['Combined_Text'].apply(has_achievable)
    df['TimeBound'] = df['Combined_Text'].apply(has_timebound)
    df['Relevant'] = df.apply(
        lambda r: has_relevant(
            r.get('Goal Description', ''),
            r.get('Goal Metric / Measurement Criteria', ''),
            r.get('Business Unit', None),
            r.get('Domain', None)
        ),
        axis=1
    )
    df['Has_Target'] = df['Target'].apply(has_target)
    df['Has_Metric'] = df['Goal Metric / Measurement Criteria'].apply(has_metric)
    df['Exact_Dup'] = detect_exact_dup(df)
    df['SMART_score'] = df[['Specific','Measurable','Achievable','TimeBound','Relevant']].sum(axis=1)
    df['low_quality'] = df['SMART_score'] <= 2
    df['high_quality'] = df['SMART_score'].isin([4,5])
    # Add Missing SMART Components column
    smart_cols = ['Specific', 'Measurable', 'Achievable', 'TimeBound', 'Relevant']
    df['Missing SMART Components'] = df.apply(
        lambda row: ', '.join([col for col in smart_cols if not row[col]]), axis=1
    )
    # Pillar_Present: True if ANY SMART pillar is present
    df['Pillar_Present'] = df[smart_cols].any(axis=1)
    # Add Final Status
    def final_status(score):
        if score >= 4:
            return 'Better'
        elif 2 <= score <= 3:
            return 'Good'
        else:
            return 'Needs Improvement'
    df['Final Status'] = df['SMART_score'].apply(final_status)
    explanations = df.apply(explain_smart_score, axis=1)
    df['SMART Explanation'] = explanations.apply(
        lambda expl: "\n".join([f"{pillar}: {reason}" for pillar, reason in expl.items()])
    )
    # All SMART pillars flag
    df['All_SMART_Pillars'] = df[smart_cols].sum(axis=1) == 5
    # Employee-specific duplicate flag - same logic as exact duplicate
    # Handle NaN values by filling them with empty string for comparison
    df_clean_emp = df.copy()
    df_clean_emp['Goal ID'] = df_clean_emp['Goal ID'].fillna('')
    df_clean_emp['Sub Goal ID'] = df_clean_emp['Sub Goal ID'].fillna('')
    df['Employee_Duplicate'] = df_clean_emp.duplicated(subset=['Employee ID', 'Goal ID', 'Sub Goal ID'], keep=False)
    return df

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Enhanced grouping with business and function/domains
    group_cols = ['Business Unit', 'Department']
    if 'Domain' in df.columns:
        group_cols.append('Domain')
    if 'Function' in df.columns:
        group_cols.append('Function')
    if 'Business' in df.columns:
        group_cols.append('Business')
    
    for col in ['Pillar_Present', 'Has_Target', 'Has_Metric', 'Exact_Dup']:
        if col not in df.columns:
            df[col] = False
    agg_dict = {
        'total_goals': ('Goal Description','count'),
        'unique_employees': ('Employee ID','nunique'),
        'avg_smart_score': ('SMART_score','mean'),
        'pct_low_quality': ('low_quality', lambda x: x.mean()*100),
        'pct_high_quality': ('high_quality', lambda x: x.mean()*100),
        'pct_missing_specific': ('Specific', lambda x: (~x).mean()*100),
        'pct_missing_measurable': ('Measurable', lambda x: (~x).mean()*100),
        'pct_missing_achievable': ('Achievable', lambda x: (~x).mean()*100),
        'pct_missing_timebound': ('TimeBound', lambda x: (~x).mean()*100),
        'pct_missing_relevant': ('Relevant', lambda x: (~x).mean()*100),
        'pct_pillar_missing': ('Pillar_Present', lambda x: (~x).mean()*100),
        'pct_target_missing': ('Has_Target', lambda x: (~x).mean()*100),
        'pct_metric_missing': ('Has_Metric', lambda x: (~x).mean()*100),
        'pct_exact_duplicates': ('Exact_Dup', lambda x: x.mean()*100),
        'avg_goals_per_employee': ('goal_count','mean'),
    }
    if 'over_6' in df.columns:
        agg_dict['pct_employees_over_6'] = ('over_6', lambda x: x.mean()*100)
    agg = df.groupby(group_cols).agg(**agg_dict).reset_index()
    return agg

def build_analytics(df: pd.DataFrame, df_emp: pd.DataFrame) -> dict:
    dims = ['Specific','Measurable','Achievable','TimeBound','Relevant']
    missed = pd.DataFrame({
        'SMART_Dimension': dims,
        'Missed_Count': [ (~df[d]).sum() for d in dims ],
        'Missed_Pct':   [ round((~df[d]).mean()*100,1) for d in dims]
    }).sort_values('Missed_Count', ascending=False)
    over6 = df_emp['over_6'].sum()
    under6= len(df_emp)-over6
    load = pd.DataFrame({'Category':['>6 Goals','≤6 Goals'],'Count':[over6,under6]})
    def safe_count(colname):
        if colname in df.columns:
            return df[colname].value_counts().rename_axis(colname).reset_index(name='Count')
        return pd.DataFrame({colname: [], 'Count': []})
    return {
        'missed': missed,
        'load': load,
        'pillar': safe_count('Pillar_Present'),
        'target': safe_count('Has_Target'),
        'metric': safe_count('Has_Metric'),
        'dup': safe_count('Exact_Dup'),
    }

def build_business_function_summaries(df: pd.DataFrame) -> dict:
    """Create additional summary views for business and function/domain comparisons"""
    summaries = {}
    
    # Business Unit summary
    if 'Business Unit' in df.columns:
        business_summary = df.groupby('Business Unit').agg({
            'Goal Description': 'count',
            'Employee ID': 'nunique',
            'SMART_score': 'mean',
            'Specific': 'mean',
            'Measurable': 'mean',
            'Achievable': 'mean',
            'TimeBound': 'mean',
            'Relevant': 'mean',
            'low_quality': 'mean',
            'high_quality': 'mean'
        }).round(3)
        business_summary.columns = ['Total_Goals', 'Unique_Employees', 'Avg_SMART_Score', 
                                  'Specific_Rate', 'Measurable_Rate', 'Achievable_Rate', 
                                  'TimeBound_Rate', 'Relevant_Rate', 'Low_Quality_Rate', 'High_Quality_Rate']
        summaries['Business_Unit_Summary'] = business_summary.reset_index()
    
    # Function summary
    if 'Function' in df.columns:
        function_summary = df.groupby('Function').agg({
            'Goal Description': 'count',
            'Employee ID': 'nunique',
            'SMART_score': 'mean',
            'Specific': 'mean',
            'Measurable': 'mean',
            'Achievable': 'mean',
            'TimeBound': 'mean',
            'Relevant': 'mean',
            'low_quality': 'mean',
            'high_quality': 'mean'
        }).round(3)
        function_summary.columns = ['Total_Goals', 'Unique_Employees', 'Avg_SMART_Score', 
                                  'Specific_Rate', 'Measurable_Rate', 'Achievable_Rate', 
                                  'TimeBound_Rate', 'Relevant_Rate', 'Low_Quality_Rate', 'High_Quality_Rate']
        summaries['Function_Summary'] = function_summary.reset_index()
    
    # Domain summary
    if 'Domain' in df.columns:
        domain_summary = df.groupby('Domain').agg({
            'Goal Description': 'count',
            'Employee ID': 'nunique',
            'SMART_score': 'mean',
            'Specific': 'mean',
            'Measurable': 'mean',
            'Achievable': 'mean',
            'TimeBound': 'mean',
            'Relevant': 'mean',
            'low_quality': 'mean',
            'high_quality': 'mean'
        }).round(3)
        domain_summary.columns = ['Total_Goals', 'Unique_Employees', 'Avg_SMART_Score', 
                                'Specific_Rate', 'Measurable_Rate', 'Achievable_Rate', 
                                'TimeBound_Rate', 'Relevant_Rate', 'Low_Quality_Rate', 'High_Quality_Rate']
        summaries['Domain_Summary'] = domain_summary.reset_index()
    
    # Business summary
    if 'Business' in df.columns:
        business_summary = df.groupby('Business').agg({
            'Goal Description': 'count',
            'Employee ID': 'nunique',
            'SMART_score': 'mean',
            'Specific': 'mean',
            'Measurable': 'mean',
            'Achievable': 'mean',
            'TimeBound': 'mean',
            'Relevant': 'mean',
            'low_quality': 'mean',
            'high_quality': 'mean'
        }).round(3)
        business_summary.columns = ['Total_Goals', 'Unique_Employees', 'Avg_SMART_Score', 
                                  'Specific_Rate', 'Measurable_Rate', 'Achievable_Rate', 
                                  'TimeBound_Rate', 'Relevant_Rate', 'Low_Quality_Rate', 'High_Quality_Rate']
        summaries['Business_Summary'] = business_summary.reset_index()
    
    # Cross-dimensional summary (Business Unit + Function/Domain combinations)
    if 'Business Unit' in df.columns and ('Function' in df.columns or 'Domain' in df.columns):
        cross_cols = ['Business Unit']
        if 'Function' in df.columns:
            cross_cols.append('Function')
        if 'Domain' in df.columns:
            cross_cols.append('Domain')
        
        cross_summary = df.groupby(cross_cols).agg({
            'Goal Description': 'count',
            'Employee ID': 'nunique',
            'SMART_score': 'mean',
            'low_quality': 'mean',
            'high_quality': 'mean'
        }).round(3)
        cross_summary.columns = ['Total_Goals', 'Unique_Employees', 'Avg_SMART_Score', 
                               'Low_Quality_Rate', 'High_Quality_Rate']
        summaries['Cross_Dimensional_Summary'] = cross_summary.reset_index()
    
    return summaries

def create_comparison_matrices(df: pd.DataFrame) -> dict:
    """Create comparison matrices for easy visual comparison across dimensions"""
    matrices = {}
    
    # SMART Score comparison matrix by Business Unit and Function
    if 'Business Unit' in df.columns and 'Function' in df.columns:
        smart_matrix = df.pivot_table(
            values='SMART_score',
            index='Business Unit',
            columns='Function',
            aggfunc='mean',
            fill_value=0
        ).round(2)
        matrices['SMART_Score_Matrix'] = smart_matrix
    
    # SMART Score comparison matrix by Business Unit and Domain
    if 'Business Unit' in df.columns and 'Domain' in df.columns:
        smart_domain_matrix = df.pivot_table(
            values='SMART_score',
            index='Business Unit',
            columns='Domain',
            aggfunc='mean',
            fill_value=0
        ).round(2)
        matrices['SMART_Score_Domain_Matrix'] = smart_domain_matrix
    
    # Goal Quality comparison matrix
    if 'Business Unit' in df.columns and 'Function' in df.columns:
        quality_matrix = df.pivot_table(
            values='high_quality',
            index='Business Unit',
            columns='Function',
            aggfunc='mean',
            fill_value=0
        ).round(3)
        matrices['Goal_Quality_Matrix'] = quality_matrix
    
    # Employee distribution matrix
    if 'Business Unit' in df.columns and 'Function' in df.columns:
        emp_matrix = df.pivot_table(
            values='Employee ID',
            index='Business Unit',
            columns='Function',
            aggfunc='nunique',
            fill_value=0
        )
        matrices['Employee_Distribution_Matrix'] = emp_matrix
    
    return matrices

def main():
    parser = argparse.ArgumentParser(description='Goal Audit Script')
    parser.add_argument('input_file', nargs='?', default=get_input_file(), help='Path to employee goals Excel file')
    parser.add_argument('--output', default=get_output_file(), help='Output Excel file name')
    args = parser.parse_args()
    
    print(f"📊 Processing input file: {args.input_file}")
    print(f"📊 Output file: {args.output}")
    
    xls = pd.ExcelFile(args.input_file)
    # Try 'All_Goals' sheet first (created by auto_sync), then fall back to original sheet name
    sheet_name = None
    if 'All_Goals' in xls.sheet_names:
        sheet_name = 'All_Goals'
        print("📋 Using 'All_Goals' sheet (from auto_sync)")
    elif 'GoalPlanDetailedReportDownload ' in xls.sheet_names:
        sheet_name = 'GoalPlanDetailedReportDownload '
        print("📋 Using 'GoalPlanDetailedReportDownload ' sheet")
    else:
        # Use first sheet as fallback
        sheet_name = xls.sheet_names[0]
        print(f"📋 Using first available sheet: '{sheet_name}'")
    
    df_raw = pd.read_excel(xls, sheet_name)
    print(f"✅ Loaded data: {len(df_raw)} rows, {len(df_raw.columns)} columns")
    
    df_raw['goal_count'] = df_raw.groupby('Employee ID')['Employee ID'].transform('count')
    print("✅ Added goal count column")
    
    df = compute_flags(df_raw)
    print("✅ Computed SMART flags")
    
    # Enhanced grouping with business and function/domains
    group_cols = ['Employee ID', 'Business Unit', 'Department']
    if 'Domain' in df.columns:
        group_cols.append('Domain')
    if 'Function' in df.columns:
        group_cols.append('Function')
    if 'Business' in df.columns:
        group_cols.append('Business')
    
    df_emp = df.groupby(group_cols).agg(
        goal_count=('goal_count','first')
    ).reset_index()
    df_emp['over_6'] = df_emp['goal_count']>6
    summary = build_summary(df)
    analytics = build_analytics(df, df_emp)
    business_function_summaries = build_business_function_summaries(df)
    comparison_matrices = create_comparison_matrices(df)
    
    print("✅ Building summary and analytics...")
    
    with pd.ExcelWriter(args.output, engine='xlsxwriter') as writer:
        print("📝 Writing All_Goals sheet...")
        df.to_excel(writer, sheet_name='All_Goals', index=False)
        print("📝 Writing Summary sheet...")
        summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Add business and function summary sheets
        print("📝 Writing business and function summary sheets...")
        for sheet_name, data in business_function_summaries.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Add comparison matrix sheets
        print("📝 Writing comparison matrix sheets...")
        for sheet_name, matrix in comparison_matrices.items():
            matrix.to_excel(writer, sheet_name=sheet_name)
        
        print("📝 Writing Analytics sheet...")
        workbook  = writer.book
        worksheet = workbook.add_worksheet('Analytics')
        writer.sheets['Analytics'] = worksheet
        row = 0
        for title, table in analytics.items():
            worksheet.write(row, 0, title.upper())
            table.to_excel(writer, sheet_name='Analytics', startrow=row+1, index=False)
            row += len(table) + 3
    
    print(f"✅ Dashboard workbook written to {args.output}")
    print(f"📊 Added business and function summary sheets for better comparison and visualization")
    print(f"🔍 Added comparison matrices for cross-dimensional analysis")

if __name__ == '__main__':
    main()
