"""Select one frozen ensemble before scoring, preserving its original threshold."""
import copy
import json
from threading import RLock

from .config import project_path
from .language_routing import LanguageRouter
from .preprocessing import normalize_text


class RoutedPredictor:
    def __init__(self, artifact, device='cpu'):
        from .adaptive_experiment import validate_identity
        self.bundle = json.loads(project_path(artifact).read_text(encoding='utf-8'))
        if self.bundle['version'] != 'language_router_v1':
            raise ValueError('Unsupported routing artifact')
        for identity in self.bundle['identities'].values():
            validate_identity(identity)
        self.cfg = copy.deepcopy(self.bundle['config'])
        self.runtime_device = device
        self.threshold = None  # Each selected expert supplies its own saved boundary.
        self.router = LanguageRouter(**self.bundle['detector'])
        self.experts = {}
        self.lock = RLock()

    def _expert(self, name):
        from .ensemble import EnsemblePredictor
        from .adaptive_experiment import validate_identity
        if name not in self.experts:
            identity = self.bundle['identities'][name]
            validate_identity(identity)
            self.experts[name] = EnsemblePredictor(identity['ensemble'], device=self.runtime_device)
        return self.experts[name]

    def predict(self, text, history=None, include_embeddings=False):
        history = [] if history is None else history
        if not isinstance(text, str) or not isinstance(history, list) or any(not isinstance(t, str) for t in history):
            raise ValueError('Provide text and a list of ancestor strings')
        text = normalize_text(text)
        history = [normalize_text(t) for t in history]
        if not text or any(not t for t in history):
            raise ValueError('Empty message after normalization')
        original_count = len(history)
        history = (history + [text])[-self.cfg['model']['max_turns']:-1]
        with self.lock:
            routing = self.router.route(text, history)
            name = routing['selected_expert']
            result = dict(self._expert(name).predict(text, history, include_embeddings))
        result.update(routing)
        result.update(selected_model=name + '_ensemble', routing_policy='language_router_v1',
                      selected_artifact=self.bundle['identities'][name]['ensemble'],
                      detected_text_type=routing['target_language']['type'],
                      ancestors_omitted=original_count-len(history),
                      score_semantics='Selected ensemble score; not calibrated probability. Compare with its returned threshold.',
                      routing_evaluation_status='New demo policy; no benchmark improvement established.')
        return result

    def predict_thread(self, messages, target_id, include_embeddings=False):
        from .inference import Predictor
        return Predictor.predict_thread(self, messages, target_id, include_embeddings)
