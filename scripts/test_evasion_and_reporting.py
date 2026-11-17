import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from contextual.config import load_config
from contextual.adapter_finetune import get_adapted_encoder, forward_examples
from contextual.model import ConversationClassifier

cfg = load_config('configs/demo_adaptive_v1.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder, tokenizer = get_adapted_encoder(cfg, device)
head = ConversationClassifier(encoder.config.hidden_size, cfg).to(device)

contrast_ckpt = torch.load('models/checkpoints/contextual_contrast_lora/contrast_adapted_state.pt', map_location='cpu', weights_only=False)
from contextual.adapter_finetune import restore_trainable
restore_trainable(encoder, contrast_ckpt['adapter'])
head.load_state_dict(contrast_ckpt['head'])
encoder.eval()
head.eval()

parent = 'What exactly did the other player say to you?'
tests = [
    ('Reporting with quotes', 'They said, "You are a useless idiot." I reported the message because personal attacks are unacceptable.'),
    ('Reporting without quotes', 'He called me a useless idiot, so I reported him.'),
    ('Direct attack with quotes (evasion attempt)', '"You are a useless idiot."'),
    ('Direct attack without quotes', 'You are a useless idiot.'),
    ('Endorsement with quotes', 'They said "You are a useless idiot", and they were right.'),
    ('Telugu reporting with quotes', 'అతను "నువ్వు ఒక చెత్త మనిషివి" అని దూషించాడు, అందుకే రిపోర్ట్ చేశాను.'),
    ('Telugu direct attack', 'నువ్వు ఒక చెత్త మనిషివి, నీలాంటి వాళ్ళు బతకడం అనవసరం.'),
    ('Romanized Telugu reporting', 'vaadu nannu "waste daddamma" annadu, anduke report chesa.'),
    ('Romanized Telugu direct attack', 'waste kukkala unnav, nee moham chusuko ra lucha.')
]

print("=" * 80)
print("TESTING ADAPTED MODEL ON REPORTING VS DIRECT ATTACKS VS QUOTE EVASION")
print("=" * 80)

for name, tgt in tests:
    with torch.no_grad():
        score = forward_examples(encoder, tokenizer, head, [{'texts': [parent, tgt], 'label': 0}], cfg, device).sigmoid().item()
    verdict = 'TOXIC' if score >= 0.5 else 'SAFE'
    print(f"{name:<45} | Score: {score:.4f} | Verdict: {verdict}")
