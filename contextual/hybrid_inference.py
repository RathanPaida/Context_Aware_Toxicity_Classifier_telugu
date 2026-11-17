"""Hybrid Adaptive Predictor: routes reported speech to the contrast specialist,

while preserving the exact frozen adaptive deployment for all standard conversations.
"""
import copy
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .adaptive_inference import AdaptivePredictor
from .config import load_config, project_path
from .discourse import is_reporting_discourse
from .preprocessing import normalize_text


class ContrastReportingSpecialist:
    """Neural LoRA specialist trained to disentangle reported abuse and counterspeech."""
    def __init__(self, checkpoint_dir: str = 'models/checkpoints/contextual_contrast_lora', device: str = 'cpu'):
        from .adapter_finetune import get_adapted_encoder, restore_trainable
        from .model import ConversationClassifier
        
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        cfg_path = project_path('configs/demo_adaptive_v1.json')
        self.cfg = copy.deepcopy(json.loads(cfg_path.read_text(encoding='utf-8')))
        self.cfg['training']['device'] = str(self.device)
        
        self.encoder, self.tokenizer = get_adapted_encoder(self.cfg, self.device)
        self.head = ConversationClassifier(self.encoder.config.hidden_size, self.cfg).to(self.device)
        
        state_path = project_path(checkpoint_dir) / 'contrast_adapted_state.pt'
        if not state_path.exists():
            raise FileNotFoundError(f"Contrast adapted state not found at {state_path}")
            
        ckpt = torch.load(state_path, map_location='cpu', weights_only=False)
        restore_trainable(self.encoder, ckpt['adapter'])
        self.head.load_state_dict(ckpt['head'])
        self.encoder.eval()
        self.head.eval()
        self.threshold = 0.50

    @torch.inference_mode()
    def predict(self, text: str, history: List[str]):
        from .adapter_finetune import forward_examples
        
        row_full = [{'texts': history + [text], 'label': 0}]
        row_iso = [{'texts': [text], 'label': 0}]
        
        score_full = forward_examples(self.encoder, self.tokenizer, self.head, row_full, self.cfg, self.device).sigmoid().item()
        score_iso = forward_examples(self.encoder, self.tokenizer, self.head, row_iso, self.cfg, self.device).sigmoid().item()
        
        return {
            'toxicity_score': float(score_full),
            'is_toxic': bool(score_full >= self.threshold),
            'threshold': self.threshold,
            'isolated_score': float(score_iso),
            'context_score_delta': float(score_full - score_iso),
            'engine_used': 'contrast_reporting_specialist',
            'discourse_frame': 'reported_speech_or_moderation',
            'script_bucket': 'reporting_discourse_frame',
            'expert_weights': {'contrast_neural_specialist': 1.0, 'baseline_ensemble': 0.0},
            'score_semantics': 'Trained probability with reporting disentanglement; threshold 0.50'
        }


class HybridAdaptivePredictor:
    """Seamless hybrid router:

    - Standard conversations -> 100% frozen old deployment (AdaptivePredictor).
    - Quoted abuse / reporting -> ContrastReportingSpecialist.
    """
    def __init__(self, adaptive_policy_path: str = 'reports/adaptive_v1/policy.json', device: str = 'cpu'):
        self.runtime_device = device
        self.threshold = 0.50
        
        # 1. Old baseline deployment (100% unchanged)
        self.baseline_predictor = AdaptivePredictor(adaptive_policy_path, device=device)
        
        # 2. Contrast specialist
        self.contrast_specialist = ContrastReportingSpecialist(device=device)

    def predict(self, text: str, history: Optional[List[str]] = None, include_embeddings: bool = False):
        history = [] if history is None else history
        text_norm = normalize_text(text)
        history_norm = [normalize_text(t) for t in history]
        
        # Check if text contains reporting discourse / quoted abuse
        if is_reporting_discourse(text_norm, history_norm):
            result = self.contrast_specialist.predict(text_norm, history_norm)
            result['turns_used'] = len(history_norm) + 1
            result['ancestors_omitted'] = 0
            result['policy'] = 'hybrid_reporting_routed_v1'
            result['embeddings_available'] = False
            result['has_context_training'] = True
            result['attention_scope'] = 'Specialist neural attention conditioned on dialogue history'
            return result
        else:
            # Fall through 100% to the exact old deployment
            result = self.baseline_predictor.predict(text, history, include_embeddings)
            result['engine_used'] = 'baseline_adaptive_policy'
            result['discourse_frame'] = 'standard_dialogue'
            return result

    def predict_thread(self, messages: list, target_id: str, include_embeddings: bool = False):
        from .inference import Predictor
        return Predictor.predict_thread(self, messages, target_id, include_embeddings)
