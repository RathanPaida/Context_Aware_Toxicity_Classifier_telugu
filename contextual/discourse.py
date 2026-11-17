"""Detect reported speech, quotations of abuse, and administrative complaints."""
import re
from typing import List, Optional

# Quotation regex supporting standard double, single, and curly quotes
QUOTE_PATTERN = re.compile(r'["“"\'‘](.*?)["”"\'’]')

# Reporting verbs in English, Romanized Telugu, and Telugu script
REPORTING_VERBS = [
    # English
    r'\b(?:they|he|she|someone|user|player)\s+(?:said|wrote|posted|messaged|called\s+me|called\s+us|told\s+me)\b',
    r'\b(?:accused\s+me\s+of|branded\s+me|labeled\s+me)\b',
    # Romanized Telugu
    r'\b(?:vaadu|aame|vaallu|atanu|user)\s+(?:nannu|maani)?\s*(?:annadu|annaru|andi|thittadu|thittaru)\b',
    r'\b(?:nannu|ninnu)\s+\w+\s+(?:ani\s+)?(?:annadu|annaru|andi|thittadu)\b',
    # Telugu Script
    r'(?:అతను|ఆమె|వాళ్ళు|సభ్యుడు)\s+(?:నన్ను|మమ్మల్ని)?\s*(?:అన్నాడు|అంది|అన్నారు|దూషించాడు|దూషించింది)',
    r'నన్ను\s+.*?\s+(?:అని|అంటూ)\s+(?:అన్నాడు|అంది|దూషించాడు)',
]

# Administrative reporting actions
ADMIN_REPORT_ACTIONS = [
    # English
    r'\b(?:i\s+reported|reported\s+(?:the|this|him|her|them|message|comment|user)|flagged\s+(?:the|this|comment|message)|filed\s+a\s+(?:report|complaint)|alerted\s+the\s+mods|escalated\s+this)\b',
    r'\b(?:because\s+personal\s+attacks|violates\s+civility|policy\s+violation|unacceptable\s+behavior)\b',
    # Romanized Telugu
    r'\b(?:anduke\s+report\s+chesa|complaint\s+chesa|block\s+cheyamantunnav|moderator\s+ki\s+cheppa|check\s+cheyandi)\b',
    # Telugu Script
    r'(?:అందుకే\s+రిపోర్ట్\s+చేశాను|ఫిర్యాదు\s+చేశాను|అధికారుల\s+దృష్టికి\s+తెచ్చాను|హెచ్చరించాను)',
]

# Counterspeech & Admonishment patterns
COUNTERSPEECH_PATTERNS = [
    # English
    r'\b(?:please\s+stop\s+calling|do\s+not\s+call\s+people|refrain\s+from\s+calling|we\s+do\s+not\s+tolerate\s+calling|keep\s+your\s+comments\s+civil)\b',
    # Romanized Telugu
    r'\b(?:dayachesi\s+evarni\s+\w+\s+ani\s+thittakandi|alanti\s+boothulu\s+matladoddhu|maryadaga\s+matladandi)\b',
    # Telugu Script
    r'(?:దయచేసి\s+ఎవరినీ.*?అని\s+దూషించకండి|మర్యాదగా\s+మాట్లాడండి|అసభ్య\s+పదజాలాన్ని\s+ఉపయోగించకండి)',
]

# Parent context prompts asking for report
PARENT_REPORT_PROMPTS = [
    r'\b(?:what\s+did\s+(?:they|he|she|the\s+other\s+player)\s+say|what\s+happened|why\s+did\s+you\s+report|why\s+was\s+it\s+flagged)\b',
    r'(?:ఎందుకు\s+ఫిర్యాదు\s+చేశారు|ఏం\s+జరిగింది|వివాదం\s+ఎందుకు)',
    r'\b(?:enduku\s+report\s+chesav|em\s+jarigindi|complaint\s+chesav)\b'
]


def is_reporting_discourse(target_text: str, history: Optional[List[str]] = None) -> bool:
    """Return True if target_text exhibits meta-linguistic reporting, quoting of abuse, or counterspeech."""
    if not target_text or not isinstance(target_text, str):
        return False
    
    t_lower = target_text.lower()
    
    # Check for direct quotation of text
    has_quotes = bool(QUOTE_PATTERN.search(target_text))
    
    # Check reporting verbs
    has_reporting_verb = any(re.search(p, t_lower) for p in REPORTING_VERBS)
    
    # Check admin reporting action
    has_admin_action = any(re.search(p, t_lower) for p in ADMIN_REPORT_ACTIONS)
    
    # Check counterspeech / admonishment
    has_counterspeech = any(re.search(p, t_lower) for p in COUNTERSPEECH_PATTERNS)
    
    if has_counterspeech:
        return True
        
    # If it has admin action AND (reporting verb or quotes) -> definite report
    if has_admin_action and (has_reporting_verb or has_quotes):
        return True
        
    # Explicit reporting verb + quotation
    if has_reporting_verb and has_quotes:
        return True
        
    # Parent context reinforcement: if parent asked "What did they say to you?" and target has reporting verb or admin action
    if history:
        parent_text = history[-1].lower() if len(history) > 0 else ""
        parent_asked_report = any(re.search(p, parent_text) for p in PARENT_REPORT_PROMPTS)
        if parent_asked_report and (has_reporting_verb or has_admin_action or has_quotes):
            return True
            
    return False
