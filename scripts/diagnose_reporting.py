"""Diagnose lexical vs neural component scores on reporting sentences and contrasting cases."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import joblib
import torch
from contextual.config import load_config, location
from contextual.inference import create_predictor

def main():
    cfg_adaptive = load_config('configs/demo_adaptive_v1.json')
    pred_adaptive = create_predictor(cfg_adaptive)

    parent = 'What happened in the discussion?'
    cases = [
        ('Reporting insult (intended: Non-toxic)', 'He called me a useless idiot, so I reported him.', False),
        ('Direct insult (intended: Toxic)', 'You are a useless idiot.', True),
        ('Counterspeech (intended: Non-toxic)', 'Please stop calling people idiots.', False),
        ('Endorsed insult (intended: Toxic)', 'He called you an idiot, and he was right.', True),
        ('Reporting with quotes (intended: Non-toxic)', 'He called me "a useless idiot" so I reported him.', False),
    ]

    print("=" * 80)
    print("STEP 1 & 2: INPUT CHECK & COMPONENT SCORE INSPECTION")
    print("Parent Context:", parent)
    print("=" * 80)

    for name, tgt, intended_toxic in cases:
        res = pred_adaptive.predict(tgt, [parent])
        score = res.get('toxicity_score', 0.0)
        is_toxic = res.get('is_toxic', False)
        status = "CORRECT" if is_toxic == intended_toxic else "FALSE POSITIVE" if is_toxic and not intended_toxic else "FALSE NEGATIVE"
        
        print(f"\nCase: {name}")
        print(f"Target: \"{tgt}\"")
        print(f"Intended: {'TOXIC' if intended_toxic else 'NON-TOXIC'} | Predicted: {'TOXIC' if is_toxic else 'NON-TOXIC'} [{status}]")
        print(f"Final Adaptive Score: {score:.4f} (Threshold: 0.5000)")
        print(f"Isolated Score: {res.get('isolated_score', 0.0):.4f} | Context Delta: {res.get('context_score_delta', 0.0):+.4f}")
        print(f"Script Bucket: {res.get('script_bucket')}")
        if 'expert_weights' in res:
            print(f"Expert Weights: {res['expert_weights']}")

    # Now let's inspect the underlying English ensemble and Mixed ensemble components directly
    print("\n" + "=" * 80)
    print("INSPECTING UNDERLYING EXPERTS (LEXICAL VS NEURAL)")
    print("=" * 80)
    
    # 1. English ensemble:
    eng_cfg = load_config('configs/english_cga_v1_ensemble.json')
    eng_artifact = joblib.load(location(eng_cfg, 'ensemble'))
    print("\nEnglish Ensemble Artifact Components:", list(eng_artifact.keys()))
    print("English Ensemble Weights:", eng_artifact.get('weights'))
    print("English Ensemble Threshold:", eng_artifact.get('threshold'))

    # 2. Mixed ensemble:
    mix_cfg = load_config('configs/demo_clean_ensemble.json')
    mix_artifact = joblib.load(location(mix_cfg, 'ensemble'))
    print("\nMixed Ensemble Artifact Components:", list(mix_artifact.keys()))
    print("Mixed Ensemble Weights:", mix_artifact.get('weights'))
    print("Mixed Ensemble Threshold:", mix_artifact.get('threshold'))

    # Let's inspect individual component scores for the reporting sentence
    test_target = 'He called me a useless idiot, so I reported him.'
    print(f"\n--- Detailed Component Breakdown for: \"{test_target}\" ---")
    
    # Run through english predictor
    pred_eng = create_predictor(eng_cfg)
    res_eng = pred_eng.predict(test_target, [parent])
    print(f"English Ensemble Total Score: {res_eng['toxicity_score']:.4f}")
    if hasattr(pred_eng, 'lexical') and pred_eng.lexical:
        for k, m in pred_eng.lexical.items():
            prob = m.predict_proba([test_target])[0, 1]
            print(f"  English Lexical [{k}]: {prob:.4f}")
    if hasattr(pred_eng, 'neural') and pred_eng.neural:
        prob = pred_eng.neural.predict(test_target, [parent])['toxicity_score']
        print(f"  English Neural Contextual: {prob:.4f}")

    # Run through mixed predictor
    pred_mix = create_predictor(mix_cfg)
    res_mix = pred_mix.predict(test_target, [parent])
    print(f"Mixed Ensemble Total Score: {res_mix['toxicity_score']:.4f}")
    if hasattr(pred_mix, 'lexical') and pred_mix.lexical:
        for k, m in pred_mix.lexical.items():
            prob = m.predict_proba([test_target])[0, 1]
            print(f"  Mixed Lexical [{k}]: {prob:.4f}")
    if hasattr(pred_mix, 'neural') and pred_mix.neural:
        prob = pred_mix.neural.predict(test_target, [parent])['toxicity_score']
        print(f"  Mixed Neural Contextual: {prob:.4f}")

if __name__ == '__main__':
    main()
