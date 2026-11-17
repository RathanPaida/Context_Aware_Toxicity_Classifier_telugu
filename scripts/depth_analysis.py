import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contextual.config import load_config
from contextual.inference import create_predictor

cfg = load_config('configs/demo_adaptive_v1.json')
pred = create_predictor(cfg)

history = [
    "మన ప్రాజెక్ట్ ప్రెజెంటేషన్ రేపు ఎన్ని గంటలకు ఉంది?",
    "ఉదయం పది గంటలకు ఉంది. మనం తొమ్మిదిన్నరకు అక్కడికి వెళ్దాం.",
    "నేను ఫలితాల పట్టికను స్లైడ్లలో చేర్చాను.",
    "బాగుంది. కానీ అన్ని భాషల్లో మన మోడల్ మెరుగ్గా పనిచేసిందని రాశావు.",
    "మొత్తం స్కోరు కొంచెం పెరిగింది కాబట్టి అలా రాశాను.",
    "మొత్తం స్కోరు పెరిగినా, ప్రతి భాషలో ఫలితం మెరుగుపడలేదు కదా.",
    "సరే. అయితే ఆ వాక్యాన్ని ఎలా మార్చాలి?"
]
target = "నీ అభిప్రాయంతో నేను ఏకీభవించడం లేదు. ప్రతి భాష ఫలితాన్ని విడిగా చూపించి, మన మోడల్ పరిమితులను కూడా వివరించాలి."

print("=" * 65)
print("SCORE VS CONTEXT DEPTH ON TELUGU DISAGREEMENT THREAD")
print("=" * 65)
print(f"{'Depth':<8} | {'Parents':<10} | {'Score':<10} | {'Decision':<10} | {'Delta':<10}")
print("-" * 65)

for d in range(len(history) + 1):
    sub_hist = history[-d:] if d > 0 else []
    res = pred.predict(target, sub_hist)
    verdict = 'TOXIC' if res['is_toxic'] else 'SAFE'
    score = res['toxicity_score']
    delta = res.get('context_score_delta', 0.0)
    print(f"{d:<8} | {len(sub_hist):<10} | {score:<10.4f} | {verdict:<10} | {delta:+10.4f}")
