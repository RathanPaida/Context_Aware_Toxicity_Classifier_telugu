import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import joblib
from contextual.config import load_config, location
from contextual.inference import create_predictor

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

print("=" * 80)
print("DIAGNOSING LONG TELUGU DISAGREEMENT THREAD")
print(f"Target: {target}")
print(f"Number of parents: {len(history)}")
print("=" * 80)

# 1. Run through demo adaptive model
cfg = load_config('configs/demo_adaptive_v1.json')
pred = create_predictor(cfg)
res = pred.predict(target, history)

print("\n--- Adaptive Predictor Result ---")
print(f"Final Score: {res['toxicity_score']:.4f} (Threshold: 0.5000)")
print(f"Verdict: {'TOXIC' if res['is_toxic'] else 'SAFE'}")
print(f"Isolated Score: {res['isolated_score']:.4f}")
print(f"Context Score Delta: {res['context_score_delta']:+.4f}")
print(f"Script Bucket: {res.get('script_bucket')}")
print(f"Expert Weights: {res.get('expert_weights')}")
print(f"Component Scores: {res.get('component_scores')}")

# 2. Break down components of the mixed ensemble
mix_cfg = load_config('configs/demo_clean_ensemble.json')
mix_art = joblib.load(location(mix_cfg, 'ensemble'))
pred_mix = create_predictor(mix_cfg)
res_mix = pred_mix.predict(target, history)

print("\n--- Mixed Ensemble Component Breakdown ---")
print(f"Total Mixed Score: {res_mix['toxicity_score']:.4f} (Threshold: {mix_art.get('threshold'):.4f})")
if hasattr(pred_mix, 'lexical') and pred_mix.lexical:
    for k, m in pred_mix.lexical.items():
        prob = m.predict_proba([target])[0, 1]
        print(f"  Lexical [{k}] on target alone: {prob:.4f}")
        for i, h in enumerate(history):
            h_prob = m.predict_proba([h])[0, 1]
            print(f"    Parent {i+1} [{k}]: {h_prob:.4f} | \"{h[:30]}...\"")

if hasattr(pred_mix, 'neural') and pred_mix.neural:
    res_neural = pred_mix.neural.predict(target, history)
    print(f"\n  Neural Model in context: {res_neural['toxicity_score']:.4f}")
    print(f"  Neural Model in isolation: {res_neural['isolated_score']:.4f}")
    print(f"  Neural Context Delta: {res_neural['context_score_delta']:+.4f}")
    if 'turn_attention' in res_neural:
        print(f"  Neural Turn Attention: {res_neural['turn_attention']}")
