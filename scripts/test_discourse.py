import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contextual.discourse import is_reporting_discourse

tests = [
    # Positive reporting cases (should be True -> routes to adapted specialist)
    ('What exactly did the other player say to you?', 'They said, "You are a useless idiot." I reported the message because personal attacks are unacceptable.', True),
    ('What happened?', 'He called me a useless idiot, so I reported him.', True),
    ('Why the report?', 'vaadu nannu "waste daddamma" annadu, anduke report chesa.', True),
    ('ఎందుకు రిపోర్ట్ చేశారు?', 'అతను నన్ను చెత్త మనిషి అని దూషించాడు, అందుకే రిపోర్ట్ చేశాను.', True),
    ('What happened?', 'Please stop calling people useless idiots on talk pages.', True),
    
    # Standard conversation cases (should be False -> routed to old deployment)
    ('Why are you editing this article?', 'Shut up you stupid idiot, stop vandalizing or I will destroy you.', False),
    ('Why are you editing?', 'You are a useless idiot.', False),
    ('"You are a useless idiot."', '"You are a useless idiot."', False), # Direct attack in fake quotes without reporting verb
    ('ఈ విషయం గురించి నీ అభిప్రాయం ఏమిటి?', 'నువ్వు ఒక చెత్త మనిషివి, నీలాంటి వాళ్ళు బతకడం అనవసరం.', False),
    ('e roju match chusava?', 'waste kukkala unnav, nee moham chusuko ra lucha.', False),
    ('Thanks for clarifying.', 'Thank you for explaining.', False),
]

all_passed = True
for parent, target, expected in tests:
    res = is_reporting_discourse(target, [parent])
    status = 'PASS' if res == expected else 'FAIL'
    if res != expected:
        all_passed = False
    print(f"[{status}] expected={expected}, got={res} | target=\"{target[:55]}...\"")

print("\nDiscourse Classifier Check All Passed:", all_passed)
