import re
from typing import Dict, List, Set, Optional

import pandas as pd


EMPTY_METRIC_VALUES = {
    '', 'nan', 'na', 'n/a', 'tbd', 'to be decided', '–', '-', 't.b.d.', 'pending',
    'not defined', 'not specified'
}

SPECIFIC_DELIVERABLE_TERMS: Set[str] = {
    'plan', 'strategy', 'process', 'system', 'platform', 'module', 'feature', 'product',
    'campaign', 'programme', 'program', 'initiative', 'dashboard', 'report', 'analysis',
    'pipeline', 'workflow', 'service', 'experience', 'journey', 'content', 'goal', 'revenue',
    'growth', 'retention', 'acquisition', 'accounts', 'customers', 'users', 'audience',
    'traffic', 'metrics', 'baseline', 'target', 'sla', 'tat', 'quality', 'performance',
    'compliance', 'framework', 'roadmap', 'prototype', 'tool', 'automation', 'integration',
    'processes', 'operations', 'portfolio', 'playbook', 'charter', 'business case', 'rollout plan',
    'brief', 'newsletter', 'content calendar', 'runbook', 'knowledge base', 'training deck', 'scorecard'
}

MEASURABLE_QUANTIFIERS: List[str] = [
    '%', 'percent', 'ratio', 'growth of', 'increase to', 'reduce to', 'variance', 'benchmark', 'index', 'score',
    'rate', 'baseline', 'conversion', 'uplift', 'throughput'
]

TIME_REGEX_PATTERNS: List[str] = [
    r'\bby\s+(?:end of\s+)?(?:q[1-4]|quarter|month|week|year|fy\s*\d{2,4}|h[12]\s*fy?\d{2,4}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|\d{4})\b',
    r'\bwithin\s+\d+\s+(?:day|days|week|weeks|month|months|quarter|quarters|year|years)\b',
    r'\b(?:deadline|milestone|eod|eow|eom|eo[qy]|cob|tat)\b',
    r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b',
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    r'\bq[1-4](?:\s*(?:fy)?\d{2,4}|\'\d{2}|-\d{2})?\b',
    r'\bfy\s*\d{2,4}\b',
    r'\bh[12]\s*fy?\d{2,4}\b',
    r'\bnext\s+(?:month|quarter|year|sprint)\b'
]

TIME_KEYWORDS: Set[str] = {'timeline', 'schedule', 'timeframe', 'whenever', 'ongoing', 'yearly', 'monthly', 'weekly', 'daily', 'quarterly', 'mid-year', 'year-end'}

TIME_STRICT_PATTERNS: List[str] = TIME_REGEX_PATTERNS

COUNTABLE_NOUNS: Set[str] = {
    'account', 'accounts', 'customer', 'customers', 'client', 'clients',
    'deal', 'deals', 'meeting', 'meetings', 'contact', 'contacts',
    'goal', 'goals', 'project', 'projects', 'initiative', 'initiatives',
    'campaign', 'campaigns', 'metric', 'metrics', 'issue', 'issues',
    'task', 'tasks', 'ticket', 'tickets', 'feature', 'features',
    'story', 'stories', 'milestone', 'milestones', 'release', 'releases',
    'pipeline', 'pipelines', 'lead', 'leads', 'opportunity', 'opportunities',
    'objective', 'objectives', 'portfolio', 'portfolios', 'visit', 'visits',
    'session', 'sessions', 'article', 'articles', 'video', 'videos',
    'subscriber', 'subscribers', 'view', 'views', 'impression', 'impressions'
}

_COUNTABLE_FORWARD = re.compile(
    r'\b(?:top\s+|new\s+|first\s+|last\s+)?\d+\+?\s+(?:[a-z&/-]+\s+){0,2}(?:' + '|'.join(COUNTABLE_NOUNS) + r')\b'
)
_COUNTABLE_BACKWARD = re.compile(
    r'\b(?:' + '|'.join(COUNTABLE_NOUNS) + r')\s+(?:[a-z&/-]+\s+){0,2}(?:top\s+|new\s+|first\s+|last\s+)?\d+\+?\b'
)

def has_countable_number(text: str) -> bool:
    if not text:
        return False
    txt = str(text).lower()
    if not re.search(r'\d', txt):
        return False
    if _COUNTABLE_FORWARD.search(txt) or _COUNTABLE_BACKWARD.search(txt):
        return True
    return False


def get_smart_vocabulary() -> Dict[str, List[str]]:
    """
    Centralized SMART vocabulary used by both fast and enhanced analysis.
    This ensures consistency across all analysis modes.
    """
    return {
        "specific": [
            'implement', 'develop', 'establish', 'launch', 'design', 'automate', 'create', 'define', 'standardize',
            'expand', 'streamline', 'introduce', 'revamp', 'upgrade', 'enhance', 'integrate', 'migrate', 'deploy',
            'formulate', 'customise', 'increase', 'reduce', 'improve', 'optimize', 'deliver', 'build', 'train',
            'achieve', 'complete', 'execute', 'acquire', 'utilize', 'collaborate', 'monitor', 'track', 'quantify',
            'report', 'finalize', 'meet', 'clear', 'precise', 'detailed', 'focused', 'unambiguous',
            'architect', 'refactor', 'modularize', 'containerize', 'externalize', 'templatize', 'codify',
            'orchestrate', 'parameterize', 'operationalize', 'harden', 'remediate', 'deprecate', 'sunset',
            'triage', 'standardise', 'productionize', 'pilot', 'rollout', 'scale', 'localize', 'backport',
            'uplift', 'tune', 'profile', 'benchmark', 'stabilize', 'annotate', 'label', 'curate', 'featurize',
            'retrain', 'fine-tune', 'evaluate', 'ab test', 'sandbox', 'guardrail', 'red-team', 'instrument',
            'log', 'telemetry', 'observe', 'observability', 'data contract', 'schema', 'sli', 'slo', 'rag',
            'evals', 'onboard', 'federate', 'unify', 'consolidate', 'migrate-off', 'decouple', 'replatform',
            'rehost', 'baseline', 'gate', 'validate', 'certify', 'sign-off', 'publish', 'curate', 'syndicate',
            'schedule', 'catalog', 'tag', 'taxonomy', 'seo', 'opengraph', 'sitemap', 'schema.org', 'runbook',
            'playbook', 'sop', 'on-call', 'incident review', 'postmortem', 'rca', 'debt register', 'backlog',
            'prototype', 'draft', 'document', 'publish roadmap', 'codify process', 'coordinate', 'facilitate',
            'spearhead', 'lead', 'prioritize', 'drive execution'
        ],
        "measurable": [
            'increase', 'reduce', 'achieve', 'monitor', 'track', 'quantify', 'report', 'improve', 'decrease',
            'exceed', 'attain', 'deliver', 'reach', 'measure', 'analyze', 'benchmark', 'calculate', 'validate',
            'number', 'percent', 'ratio', 'amount', 'metric', 'kpi', 'target', 'progress', 'verifiable', 'yoy',
            'qoq', 'year on year', 'quarter on quarter', 'half yearly', 'conversion', 'adoption', 'activation',
            'retention', 'churn', 'throughput', 'latency', 'availability', 'error rate', 'incident rate', 'mttr',
            'mttd', 'lead time', 'cycle time', 'velocity', 'defect density', 'test coverage', 'ci pass rate',
            'unit cost', 'cost per user', 'cloud spend', 'finops', 'utilization', 'efficiency', 'waste', 'budget variance',
            'pipeline', 'win rate', 'acv', 'arr', 'nrr', 'expansion', 'upsell', 'cross-sell', 'nps', 'csat', 'ces',
            'response time', 'ttr', 'sla attainment', 'vulnerabilities', 'cve count', 'patch coverage', 'audit findings',
            'policy violations', 'false positive rate', 'rto', 'rpo', 'slo attainment', 'error budget', 'capacity buffer',
            'mau', 'dau', 'pageviews', 'sessions', 'ad yield', 'monetization', 'hit rate', 'conversion ratio',
            'customer retention rate', 'case resolution time', 'lead to win ratio', 'service level adherence',
            'ctr', 'click-through rate', 'open rate', 'bounce rate', 'watch time', 'view time', 'completion rate',
            'customer satisfaction', 'net promoter score', 'gross margin', 'inventory turn', 'utilization rate',
            'aop', 'aop target', 'aop targets', 'annual operating plan', 'annual operating plan target', 'against aop',
            'as per aop', 'aop sheet', 'management aop', 'aop plan', 'aop budget', 'aop achievement', 'aop goal', 'aop goals'
        ],
        "achievable_neg": [
            'all', 'every', 'always', 'never', 'completely', 'fully', 'impossible', 'perfect', 'asap', 'as soon as possible',
            'guarantee', 'zero-defect', '100%', 'unlimited', 'instantaneous', 'no-risk', 'immediately', 'overnight', 'solve all',
            'any and all', 'for every scenario', 'zero bugs', 'zero downtime', '100 percent', 'guaranteed', 'no exceptions',
            'infinite', 'instant', 'no limits', 'zero error', 'zero failure', 'perfect score', 'flawless', 'error-free', 'bug-free'
        ],
        "achievable_pos": [
            'realistic', 'feasible', 'resources', 'skills', 'capability', 'stretch', 'manageable', 'attainable', 'budget',
            'support', 'dependency', 'within capacity', 'resourced', 'pragmatic', 'staged', 'phased', 'piloted', 'staffed',
            'de-risked', 'dependency-tracked', 'capacity-aligned', 'right-sized', 'mvp', 'iterative', 'incremental',
            'resourced plan', 'phased rollout', 'pilot phase', 'proof of concept', 'validated approach', 'prioritized backlog',
            'capacity plan', 'risk mitigation', 'contingency plan', 'dependency map', 'training plan'
        ],
        "relevant": [
            'align with', 'support', 'contribute to', 'drive', 'advance', 'enhance', 'strengthen', 'customer', 'team',
            'process', 'system', 'quality', 'cost', 'efficiency', 'productivity', 'satisfaction', 'revenue', 'growth',
            'performance', 'business objective', 'strategic priority', 'org goal', 'organisational goal', 'impact', 'outcome',
            'customer value', 'key initiative', 'business need', 'organisational need', 'organisational challenge',
            'organisational opportunity', 'organisational objective', 'connected', 'important', 'strategic', 'meaningful',
            'impactful', 'worthwhile', 'okr', 'kr', 'key result', 'north star', 'roadmap', 'initiative', 'bet', 'business case',
            'roi', 'payback', 'launch readiness', 'enablement', 'content', 'messaging', 'positioning', 'pipeline contribution',
            'aop', 'annual plan', 'portfolio initiative', 'bu objective', 'p&l', 'unit economics', 'c-sat driver', 'sla',
            'tat commitments', 'editorial calendar', 'monetization', 'ad yield', 'pageviews', 'sessions', 'mau', 'dau', 'retention',
            'churn', 'safety', 'privacy', 'bias', 'robustness', 'transparency', 'model risk', 'model governance', 'model card',
            'evals', 'golden path', 'secure defaults', 'paved road', 'idp', 'developer portal', 'template', 'scaffolding', 'lineage',
            'catalog', 'pii', 'data quality', 'freshness', 'completeness', 'accuracy', 'threat modeling', 'sbom', 'sast', 'dast',
            'secret scanning', 'posture', 'edr', 'iam hygiene', 'least privilege', 'rightsizing', 'reservations', 'savings plans',
            'idle resource cleanup', 'tagging compliance', 'playbook', 'enablement assets', 'talk track', 'qbr', 'ebr',
            'health score', 'escalation reduction', 'onboarding', 'ramp-up', 'competency', 'skill matrix', 'learning path',
            'certification', 'time-to-value', 'self-serve', 'standardization', 'reusability', 'automation', 'productivity',
            'market share', 'customer retention', 'business outcome', 'operational efficiency', 'compliance objective',
            'customer experience', 'brand awareness', 'audience growth', 'commercial impact', 'strategic pillar'
        ],
        "timebound": [
            'complete by', 'launch by', 'deliver within', 'finalize by', 'meet by', 'quarter', 'week', 'month', 'schedule',
            'timeframe', 'due date', 'fy', 'q1', 'q2', 'q3', 'q4', 'end of q1', 'end of q2', 'end of q3', 'end of q4',
            'end of fy', 'end of year', 'end of quarter', 'end of month', 'end of week', 'end of day', 'annual', 'monthly',
            'quarterly', 'yearly', 'deadline', 'timeline', 'milestone', 'urgency', 'by', 'date', 'within', 'before', 'after',
            'until', 'per month', 'per year', '2024', '2025', '2026', 'annually', 'yoy', 'qoq', 'half yearly', 'biannually',
            'eow', 'eom', 'eoq', 'eoy', 'next sprint', 'sprint', 'pi-1', 'pi planning', 'iteration',
            'target date', 'gate', 'cutover date', 'freeze', 'change window', 'remediation window',
            'pre-ga', 'ga', 'public launch', 'fy24', 'fy25', 'fy26', 'fy-24', 'fy-25', 'fy-26', 'fy 2024', 'fy 2025',
            'fy 2026', 'h1 fy24', 'h1 fy25', 'h1 fy26', 'h2 fy24', 'h2 fy25', 'h2 fy26', 'h1', 'h2',
            "q1'24", "q1'25", "q1'26", "q2'24", "q2'25", "q2'26", "q3'24", "q3'25", "q3'26", "q4'24", "q4'25", "q4'26",
            'q1-24', 'q1-25', 'q2-24', 'q2-25', 'q3-24', 'q3-25', 'q4-24', 'q4-25', 'mar-24', 'mar-25', 'apr-24', 'apr-25',
            'may-24', 'may-25', 'jun-24', 'jun-25', "mar'24", "mar'25", "apr'24", "apr'25", "dec'24", "dec'25", 'sprint 1',
            'sprint 2', 'sprint 3', 'sprint 4', 'sprint 5', 'sprint 6', 'sprint 7', 'sprint 8', 'sprint 9', 'sprint 10',
            'pi-1', 'pi-2', 'pi-3', 'pi-4', 'iteration 1', 'iteration 2', 'iteration 3', 'next sprint', 'diwali', 'eoss',
            'festival season', 'festival campaign', 'fortnight', 'fortnightly', 'quarter-end', 'mid-year', 'year-end', 'month-end'
        ]
    }


SMART_VOCAB = get_smart_vocabulary()

SMART_VERBS = {
    'Specific': SMART_VOCAB['specific'],
    'Measurable': SMART_VOCAB['measurable'],
    'Achievable': SMART_VOCAB['achievable_pos'],
    'Relevant': SMART_VOCAB['relevant'],
    'TimeBound': SMART_VOCAB['timebound'],
}

SMART_KEYWORDS = {
    'Specific': SMART_VOCAB['specific'],
    'Measurable': SMART_VOCAB['measurable'],
    'Achievable': SMART_VOCAB['achievable_pos'],
    'Relevant': SMART_VOCAB['relevant'],
    'TimeBound': SMART_VOCAB['timebound'],
}

ACHIEVABLE_NEG = set(term.lower() for term in SMART_VOCAB['achievable_neg'])


def is_empty_value(text) -> bool:
    if pd.isna(text):
        return True
    return str(text).strip().lower() in EMPTY_METRIC_VALUES


def has_specific(text: str) -> bool:
    txt = str(text).lower().strip()
    if not txt or txt == 'nan':
        return False
    found_verb = any(v in txt for v in SMART_VERBS['Specific'])
    found_deliverable = any(term in txt for term in SPECIFIC_DELIVERABLE_TERMS)
    return found_verb and found_deliverable


def has_measurable(text: str) -> bool:
    txt = str(text)
    if not txt or txt.strip().lower() == 'nan':
        return False
    txt_lower = txt.lower()
    has_digits = bool(re.search(r'\d', txt_lower))
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
        return has_digits or has_countable
    if not has_digits and not has_countable:
        return False
    found_verb = any(v in txt_lower for v in SMART_VERBS['Measurable'])
    found_kw = any(k in txt_lower for k in SMART_KEYWORDS['Measurable'])
    return found_verb or found_kw or has_countable


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
    phased = any(x in txt for x in ['phased', 'pilot', 'mvp', 'stage', 'incremental', 'iterative'])
    return found_verb or found_kw or phased


def get_business_unit_keywords(business_unit: Optional[str] = None) -> Set[str]:
    """
    Returns relevant keywords for a given Business Unit.
    If business_unit is None, returns empty set.
    """
    if not business_unit:
        return set()
    
    bu_lower = str(business_unit).lower().strip()
    
    # Map Business Units to their relevant keywords
    bu_keyword_map = {
        'adsales': {'ad', 'ads', 'advertising', 'advertiser', 'advertisers', 'sales', 'revenue', 'monetization', 
                   'campaign', 'campaigns', 'impression', 'impressions', 'click', 'clicks', 'ctr', 'cpm', 'cpc',
                   'ad yield', 'ad inventory', 'ad placement', 'ad format', 'programmatic', 'display', 'video ads',
                   'native ads', 'sponsored', 'brand', 'branding', 'marketing', 'media', 'publisher', 'publishers'},
        'engineering': {'code', 'coding', 'develop', 'development', 'engineer', 'engineering', 'software', 'application',
                       'app', 'system', 'platform', 'api', 'sdk', 'infrastructure', 'architecture', 'design', 'build',
                       'deploy', 'deployment', 'release', 'version', 'feature', 'features', 'bug', 'bugs', 'fix',
                       'testing', 'test', 'qa', 'quality', 'performance', 'scalability', 'reliability', 'security',
                       'database', 'backend', 'frontend', 'fullstack', 'devops', 'ci/cd', 'automation', 'tooling'},
        'product': {'product', 'products', 'feature', 'features', 'roadmap', 'strategy', 'launch', 'launches',
                   'user experience', 'ux', 'ui', 'design', 'research', 'analytics', 'metrics', 'kpi', 'user',
                   'users', 'customer', 'customers', 'feedback', 'insights', 'validation', 'mvp', 'prototype',
                   'iteration', 'improvement', 'enhancement', 'optimization'},
        'sales': {'sales', 'revenue', 'revenue growth', 'deal', 'deals', 'opportunity', 'opportunities', 'pipeline',
                 'forecast', 'quota', 'target', 'account', 'accounts', 'client', 'clients', 'customer', 'customers',
                 'prospect', 'prospects', 'lead', 'leads', 'conversion', 'win rate', 'acv', 'arr', 'nrr', 'upsell',
                 'cross-sell', 'retention', 'churn', 'relationship', 'engagement'},
        'marketing': {'marketing', 'campaign', 'campaigns', 'brand', 'branding', 'awareness', 'acquisition', 'lead',
                     'leads', 'generation', 'content', 'content marketing', 'seo', 'sem', 'social media', 'email',
                     'email marketing', 'event', 'events', 'webinar', 'webinars', 'conference', 'advertising',
                     'promotion', 'promotions', 'messaging', 'positioning', 'go-to-market', 'gtm', 'demand gen'},
        'customer support': {'customer', 'customers', 'support', 'service', 'help', 'assistance', 'ticket', 'tickets',
                           'issue', 'issues', 'resolution', 'resolve', 'satisfaction', 'csat', 'nps', 'response time',
                           'ttr', 'time to resolve', 'sla', 'quality', 'experience', 'onboarding', 'training',
                           'documentation', 'knowledge base', 'faq', 'chat', 'email support', 'phone support'},
        'operations': {'operation', 'operations', 'process', 'processes', 'workflow', 'workflows', 'efficiency',
                      'productivity', 'automation', 'optimization', 'streamline', 'standardize', 'procedure',
                      'procedure', 'sop', 'runbook', 'playbook', 'compliance', 'governance', 'risk', 'security',
                      'infrastructure', 'system', 'systems', 'tool', 'tools', 'platform', 'platforms'},
        'finance': {'finance', 'financial', 'budget', 'budgets', 'cost', 'costs', 'expense', 'expenses', 'revenue',
                  'profit', 'profitability', 'margin', 'margins', 'forecast', 'forecasting', 'planning', 'analysis',
                  'reporting', 'reports', 'audit', 'compliance', 'accounting', 'tax', 'treasury', 'investment'},
        'hr': {'hr', 'human resources', 'talent', 'recruitment', 'recruiting', 'hiring', 'onboarding', 'training',
              'development', 'performance', 'review', 'reviews', 'employee', 'employees', 'engagement', 'retention',
              'culture', 'diversity', 'inclusion', 'compensation', 'benefits', 'policy', 'policies', 'compliance'},
    }
    
    # Try exact match first
    if bu_lower in bu_keyword_map:
        return bu_keyword_map[bu_lower]
    
    # Try partial match (e.g., "AdSales" contains "adsales")
    for key, keywords in bu_keyword_map.items():
        if key in bu_lower or bu_lower in key:
            return keywords
    
    # Default: extract meaningful words from business unit name
    words = re.findall(r'\b\w+\b', bu_lower)
    return set(words) if words else set()


def get_domain_keywords(domain: Optional[str] = None) -> Set[str]:
    """
    Returns relevant keywords for a given Domain.
    If domain is None, returns empty set.
    """
    if not domain:
        return set()
    
    domain_lower = str(domain).lower().strip()
    
    # Map Domains to their relevant keywords
    domain_keyword_map = {
        'customer support': {'customer', 'customers', 'support', 'service', 'help', 'assistance', 'ticket', 'tickets',
                           'issue', 'issues', 'resolution', 'resolve', 'satisfaction', 'csat', 'nps', 'response time',
                           'ttr', 'time to resolve', 'sla', 'quality', 'experience', 'onboarding', 'training',
                           'documentation', 'knowledge base', 'faq', 'chat', 'email support', 'phone support',
                           'customer success', 'retention', 'churn', 'engagement'},
        'sales': {'sales', 'revenue', 'revenue growth', 'deal', 'deals', 'opportunity', 'opportunities', 'pipeline',
                 'forecast', 'quota', 'target', 'account', 'accounts', 'client', 'clients', 'customer', 'customers',
                 'prospect', 'prospects', 'lead', 'leads', 'conversion', 'win rate', 'acv', 'arr', 'nrr', 'upsell',
                 'cross-sell', 'retention', 'churn', 'relationship', 'engagement', 'qbr', 'ebr'},
        'engineering': {'code', 'coding', 'develop', 'development', 'engineer', 'engineering', 'software', 'application',
                       'app', 'system', 'platform', 'api', 'sdk', 'infrastructure', 'architecture', 'design', 'build',
                       'deploy', 'deployment', 'release', 'version', 'feature', 'features', 'bug', 'bugs', 'fix',
                       'testing', 'test', 'qa', 'quality', 'performance', 'scalability', 'reliability', 'security',
                       'database', 'backend', 'frontend', 'fullstack', 'devops', 'ci/cd', 'automation', 'tooling'},
        'product': {'product', 'products', 'feature', 'features', 'roadmap', 'strategy', 'launch', 'launches',
                   'user experience', 'ux', 'ui', 'design', 'research', 'analytics', 'metrics', 'kpi', 'user',
                   'users', 'customer', 'customers', 'feedback', 'insights', 'validation', 'mvp', 'prototype',
                   'iteration', 'improvement', 'enhancement', 'optimization'},
        'marketing': {'marketing', 'campaign', 'campaigns', 'brand', 'branding', 'awareness', 'acquisition', 'lead',
                     'leads', 'generation', 'content', 'content marketing', 'seo', 'sem', 'social media', 'email',
                     'email marketing', 'event', 'events', 'webinar', 'webinars', 'conference', 'advertising',
                     'promotion', 'promotions', 'messaging', 'positioning', 'go-to-market', 'gtm', 'demand gen'},
        'operations': {'operation', 'operations', 'process', 'processes', 'workflow', 'workflows', 'efficiency',
                      'productivity', 'automation', 'optimization', 'streamline', 'standardize', 'procedure',
                      'procedure', 'sop', 'runbook', 'playbook', 'compliance', 'governance', 'risk', 'security',
                      'infrastructure', 'system', 'systems', 'tool', 'tools', 'platform', 'platforms'},
        'data': {'data', 'analytics', 'analysis', 'insights', 'reporting', 'reports', 'dashboard', 'dashboards',
                'metrics', 'kpi', 'kpis', 'measurement', 'tracking', 'monitoring', 'database', 'warehouse',
                'warehousing', 'etl', 'pipeline', 'pipelines', 'model', 'models', 'ml', 'machine learning',
                'ai', 'artificial intelligence', 'prediction', 'forecast', 'forecasting'},
        'security': {'security', 'secure', 'safety', 'protection', 'vulnerability', 'vulnerabilities', 'threat',
                    'threats', 'risk', 'risks', 'compliance', 'audit', 'audits', 'policy', 'policies', 'access',
                    'authentication', 'authorization', 'encryption', 'privacy', 'gdpr', 'pii', 'data protection'},
    }
    
    # Try exact match first
    if domain_lower in domain_keyword_map:
        return domain_keyword_map[domain_lower]
    
    # Try partial match
    for key, keywords in domain_keyword_map.items():
        if key in domain_lower or domain_lower in key:
            return keywords
    
    # Default: extract meaningful words from domain name
    words = re.findall(r'\b\w+\b', domain_lower)
    return set(words) if words else set()


def has_relevant(desc: str, metric: str, business_unit: Optional[str] = None, domain: Optional[str] = None) -> bool:
    """
    Enhanced relevancy check that considers Business Unit and Domain context.
    
    Args:
        desc: Goal description
        metric: Goal metric/measurement criteria
        business_unit: Business Unit name (optional)
        domain: Domain name (optional)
    
    Returns:
        True if goal is relevant, False otherwise
    """
    desc = str(desc).lower() if desc is not None else ''
    metric = str(metric).lower() if metric is not None else ''
    combined = f"{desc} {metric}"
    
    # Get context-specific keywords
    bu_keywords = get_business_unit_keywords(business_unit)
    domain_keywords = get_domain_keywords(domain)
    
    # Check if goal contains keywords relevant to Business Unit
    has_bu_relevance = False
    if bu_keywords:
        # Check if any business unit keyword appears in the combined text
        for keyword in bu_keywords:
            # Use word boundary to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, combined, re.IGNORECASE):
                has_bu_relevance = True
                break
    
    # Check if goal contains keywords relevant to Domain
    has_domain_relevance = False
    if domain_keywords:
        # Check if any domain keyword appears in the combined text
        for keyword in domain_keywords:
            # Use word boundary to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, combined, re.IGNORECASE):
                has_domain_relevance = True
                break
    
    # If both Business Unit and Domain are provided, goal is relevant if it matches either
    # If only one is provided, goal is relevant if it matches that one
    if business_unit and domain:
        if has_bu_relevance or has_domain_relevance:
            return True
    elif business_unit and has_bu_relevance:
        return True
    elif domain and has_domain_relevance:
        return True
    
    # Original relevancy checks (alignment cues and outcome ties)
    alignment_cues = [
        'align with', 'support', 'contribute to', 'drive', 'advance', 'enable', 'unlock',
        'to support', 'to drive', 'to enable', 'to advance', 'to contribute',
        'business objective', 'strategic priority', 'okr', 'key result', 'north star',
        'company goal', 'org goal', 'organisational goal', 'key initiative',
        'business need', 'organisational need', 'organisational challenge',
        'organisational opportunity', 'organisational objective'
    ]
    has_alignment = any(cue in combined for cue in alignment_cues)
    outcome_patterns = [
        r'to\s+(reduce|decrease|lower|cut|minimize)\s+(cost|expense|spend|budget|waste)',
        r'to\s+(increase|improve|boost|enhance|raise|grow)\s+(revenue|sales|income|profit|retention|satisfaction|customer|growth)',
        r'to\s+(reduce|decrease|minimize)\s+(churn|attrition|errors|defects|incidents)',
        r'to\s+(improve|increase|enhance)\s+(efficiency|productivity|performance|quality|experience)'
    ]
    has_outcome_tie = any(re.search(pattern, combined, re.IGNORECASE) for pattern in outcome_patterns)
    if has_alignment or has_outcome_tie:
        return True
    
    # Fallback: require business keywords WITH context
    business_keywords = ['revenue', 'growth', 'cost', 'efficiency', 'productivity', 'satisfaction',
                         'customer', 'business', 'team', 'process', 'system', 'quality', 'performance']
    context_words = ['for', 'to', 'support', 'drive', 'improve', 'enhance', 'reduce', 'increase']
    has_business_with_context = any(
        kw in combined and any(ctx in combined for ctx in context_words)
        for kw in business_keywords
    )
    word_count = len(combined.split())
    return has_business_with_context and word_count >= 8


def has_timebound(text: str) -> bool:
    txt = str(text).lower()
    if not txt or txt == 'nan':
        return False
    if any(re.search(pattern, txt, re.IGNORECASE) for pattern in TIME_STRICT_PATTERNS):
        return True
    has_keyword = any(keyword in txt for keyword in TIME_KEYWORDS)
    has_context = any(word in txt for word in ['by', 'within', 'deadline', 'due', 'target date', 'end of'])
    return has_keyword and has_context


def create_combined_text(row: pd.Series) -> str:
    parts = []
    if pd.notna(row.get('Goal Description', '')):
        parts.append(f"📝 Goal: {str(row['Goal Description']).strip()}")
    if pd.notna(row.get('Sub Goal Description', '')):
        parts.append(f"📋 Sub Goal: {str(row['Sub Goal Description']).strip()}")
    prefix = str(row.get('Prefix Target', '')).strip() if pd.notna(row.get('Prefix Target', '')) else ''
    target = str(row.get('Target', '')).strip() if pd.notna(row.get('Target', '')) else ''
    if prefix and target:
        parts.append(f"🎯 Prefix: {prefix}")
        parts.append(f"🎯 Target: {target}")
    elif target:
        parts.append(f"🎯 Target: {target}")
    elif prefix:
        parts.append(f"🎯 Prefix: {prefix}")
    metric = row.get('Goal Metric / Measurement Criteria', '')
    if pd.notna(metric) and str(metric).strip():
        parts.append(f"📊 Metric: {str(metric).strip()}")
    return ' | '.join(parts)


def explain_smart_score(row: pd.Series) -> Dict[str, str]:
    explanations = {}
    combined_text = str(row.get('Combined_Text', '') or '').lower()
    goal_desc = str(row.get('Goal Description', '')).strip()
    goal_lower = goal_desc.lower()
    target = str(row.get('Target', '')).strip() if pd.notna(row.get('Target', '')) else ''
    metric = str(row.get('Goal Metric / Measurement Criteria', '')).strip() if pd.notna(row.get('Goal Metric / Measurement Criteria', '')) else ''

    if row.get('Specific', False):
        found_verbs = [v for v in SMART_VERBS['Specific'] if v in goal_lower]
        deliverable_terms = [term for term in SPECIFIC_DELIVERABLE_TERMS if term in combined_text]
        if found_verbs and deliverable_terms:
            explanations['Specific'] = f"✓ Action verb(s): {', '.join(found_verbs[:2])} → Deliverable: {', '.join(deliverable_terms[:2])}"
        elif found_verbs:
            explanations['Specific'] = f"✓ Contains action verb(s): {', '.join(found_verbs[:3])}"
        else:
            explanations['Specific'] = "✓ Specific action keywords with deliverable context"
    else:
        explanations['Specific'] = "✗ Missing clear action verbs or specific details"

    if row.get('Measurable', False):
        reasons = []
        has_numeric_value = bool(re.search(r'\d', combined_text))
        countable_match = has_countable_number(combined_text) or has_countable_number(goal_lower)
        if '%' in goal_desc or 'percent' in goal_lower:
            if 'ytd' in goal_lower or 'year to date' in goal_lower or 'year-to-date' in goal_lower:
                reasons.append("percentage with YTD context")
            elif any(word in goal_lower for word in ['against', 'versus', 'vs']):
                reasons.append("percentage with comparison")
            else:
                reasons.append("percentage/metric")
        if target and not is_empty_value(target):
            target_lower = target.lower()
            if any(word in target_lower for word in ['increase', 'grow', 'improve', 'raise', 'boost']):
                reasons.append("target (higher is better)")
            elif any(word in target_lower for word in ['reduce', 'decrease', 'lower', 'minimize', 'cut']):
                reasons.append("target (lower is better)")
            else:
                reasons.append("target value")
        if metric and not is_empty_value(metric):
            reasons.append("measurement criteria")
        if has_numeric_value:
            reasons.append("numeric value")
        if countable_match:
            reasons.append("countable target")
        if not has_numeric_value and not countable_match:
            explanations['Measurable'] = "✗ Missing numbers or explicit quantifiers"
        else:
            explanations['Measurable'] = f"✓ Has measurable elements: {', '.join(reasons[:2])}" if reasons else "✓ Contains measurement keywords"
    else:
        explanations['Measurable'] = "✗ Missing numbers, percentages, or measurement criteria"

    if row.get('Achievable', False):
        if any(word in goal_lower for word in ['mvp', 'phased', 'staged', 'incremental', 'realistic', 'feasible']):
            explanations['Achievable'] = "✓ Contains feasibility indicators (phased, realistic, etc.)"
        elif len(goal_lower.split()) >= 7:
            explanations['Achievable'] = "✓ Well-structured goal without unrealistic language"
        else:
            explanations['Achievable'] = "✓ No unrealistic indicators found"
    else:
        unrealistic = [w for w in ['all', 'every', 'always', 'never', 'impossible', 'perfect', '100%', 'zero', 'guaranteed'] if w in goal_lower]
        if unrealistic:
            explanations['Achievable'] = f"✗ Contains unrealistic language: {', '.join(unrealistic[:2])}"
        else:
            explanations['Achievable'] = "✗ Lacks feasibility indicators or structure"

    if row.get('Relevant', False):
        alignment_cues = ['align', 'support', 'contribute', 'drive', 'enable', 'okr', 'key result', 'business objective', 'strategic']
        found_cues = [c for c in alignment_cues if c in combined_text]
        if found_cues:
            explanations['Relevant'] = f"✓ Contains alignment cues: {', '.join(found_cues[:2])}"
        elif re.search(r'to\s+(reduce|increase|improve)\s+(cost|revenue|efficiency|productivity)', combined_text):
            explanations['Relevant'] = "✓ Contains business outcome tie (to reduce cost, increase revenue, etc.)"
        else:
            explanations['Relevant'] = "✓ Contains business keywords with context"
    else:
        explanations['Relevant'] = "✗ Missing alignment cues or business outcome ties"

    if row.get('TimeBound', False):
        matches = [pat for pat in TIME_REGEX_PATTERNS if re.search(pat, combined_text, re.IGNORECASE)]
        if matches:
            explanations['TimeBound'] = "✓ Contains time reference"
        else:
            explanations['TimeBound'] = "✓ Contains time keywords"
    else:
        explanations['TimeBound'] = "✗ Missing time references (by, within, Q1, FY, deadline, etc.)"

    return explanations

