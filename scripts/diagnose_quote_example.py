import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import joblib
from contextual.config import load_config, location
from contextual.inference import create_predictor
from contextual.adapter_finetune import get_adapted_encoder, forward_examples
from contextual.model import ConversationClassifier

parent = "What exactly did the other player say to you?"
target = 'They said, "You are a useless idiot." I reported the message because personal attacks are unacceptable.'

print("=" * 80)
print("DIAGNOSIS ON USER'S SPECIFIC QUOTED ABUSE EXAMPLE")
print(f"Parent: {parent}")
print(f"Target: {target}")
print("=" * 80)

# 1. Active demo predictor (what runs in the web interface)
cfg_demo = load_config('configs/demo_adaptive_v1.json')
pred_demo = create_predictor(cfg_demo)
res_demo = pred_demo.predict(target, [parent])
print("\n1. Active Demo Server (demo_adaptive_v1.json):")
print(f"   Final Decision Score: {res_demo['toxicity_score']:.4f} (Threshold: 0.5000)")
print(f"   Verdict: {'TOXIC' if res_demo['is_toxic'] else 'SAFE'}")
print(f"   Script Bucket: {res_demo['script_bucket']}")
print(f"   Expert Weights: {res_demo.get('expert_weights')}")
print(f"   Component Scores: {res_demo.get('component_scores')}")

# 2. English ensemble components
eng_cfg = load_config('configs/english_cga_v1_ensemble.json')
pred_eng = create_predictor(eng_cfg)
res_eng = pred_eng.predict(target, [parent])
print("\n2. English Ensemble Breakdown:")
print(f"   Total English Score: {res_eng['toxicity_score']:.4f}")
if hasattr(pred_eng, 'lexical') and pred_eng.lexical:
    for k, m in pred_eng.lexical.items():
        print(f"     Lexical [{k}]: {m.predict_proba([target])[0, 1]:.4f}")
if hasattr(pred_eng, 'neural') and pred_eng.neural:
    print(f"     Neural: {pred_eng.neural.predict(target, [parent])['toxicity_score']:.4f}")

# 3. Mixed ensemble components
mix_cfg = load_config('configs/demo_clean_ensemble.json')
pred_mix = create_predictor(mix_cfg)
res_mix = pred_mix.predict(target, [parent])
print("\n3. Mixed Ensemble Breakdown:")
print(f"   Total Mixed Score: {res_mix['toxicity_score']:.4f}")
if hasattr(pred_mix, 'lexical') and pred_mix.lexical:
    for k, m in pred_mix.lexical.items():
        print(f"     Lexical [{k}]: {m.predict_proba([target])[0, 1]:.4f}")
if hasattr(pred_mix, 'neural') and pred_mix.neural:
    print(f"     Neural: {pred_mix.neural.predict(target, [parent])['toxicity_score']:.4f}")

# 4. Newly adapted contrast neural model
print("\n4. Newly Adapted Contrast Neural Model (contextual_contrast_lora):")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder, tokenizer = get_adapted_encoder(cfg_demo, device)
head = ConversationClassifier(encoder.config.hidden_size, cfg_demo).to(device)
contrast_ckpt = torch.load('models/checkpoints/contextual_contrast_lora/contrast_adapted_state.pt', map_location='cpu', weights_only=False)
from contextual.adapter_finetune import restore_trainable
restore_trainable(encoder, contrast_ckpt['adapter'])
head.load_state_dict(contrast_ckpt['head'])
encoder.eval(); head.eval()
with torch.no_grad():
    row = [{'texts': [parent, target], 'label': 0}]
    score_contrast = forward_examples(encoder, tokenizer, head, row, cfg_demo, device).sigmoid().item()
print(f"   Adapted Neural Score: {score_contrast:.4f} (Threshold: 0.5000)")
print(f"   Adapted Verdict: {'TOXIC' if score_contrast >= 0.5 else 'SAFE'}")
