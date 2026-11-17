"""Serve the frozen validation policy; unknown scripts fall back to the mixed model."""
import copy
import json

import numpy as np

from .adaptive_policy import EXPERTS, POLICY_VERSION, margin_score, script_bucket, validate_policy
from .config import project_path
from .preprocessing import normalize_text


class AdaptivePredictor:
    def __init__(self, artifact, device='cpu'):
        from .adaptive_experiment import validate_identity
        self.bundle=json.loads(project_path(artifact).read_text(encoding='utf-8'))
        if self.bundle['version'] != POLICY_VERSION or self.bundle['experts'] != list(EXPERTS):
            raise ValueError('Unsupported adaptive policy')
        validate_policy(self.bundle['policy'])
        for identity in self.bundle['identities'].values(): validate_identity(identity)
        self.cfg=copy.deepcopy(self.bundle['config'])
        self.cfg['training']['device']=device
        self.runtime_device=device
        self.threshold=.5
        self.experts={}

    def _expert(self,name):
        from .ensemble import EnsemblePredictor
        from .adaptive_experiment import validate_identity
        if name not in self.experts:
            validate_identity(self.bundle['identities'][name])
            self.experts[name]=EnsemblePredictor(self.bundle['identities'][name]['ensemble'],device=self.runtime_device)
        return self.experts[name]

    def predict(self,text,history=None,include_embeddings=False):
        history=[] if history is None else history
        if not isinstance(text,str) or not isinstance(history,list) or any(not isinstance(t,str) for t in history):
            raise ValueError('Provide text and a list of ancestor strings')
        text=normalize_text(text); history=[normalize_text(t) for t in history]
        if not text or any(not t for t in history): raise ValueError('Empty message after normalization')
        original_length=len(history)
        history=(history+[text])[-self.cfg['model']['max_turns']:-1]
        bucket=script_bucket(text,history)
        policy=self.bundle['policy']
        branch=policy['branches'].get(bucket,policy['fallback'])
        weights=np.asarray(branch['weights'])
        full=np.zeros(3); alone=np.zeros(3)
        components={}
        if weights[0] > 0:
            result=self._expert('english').predict(text,history,False)
            full[0],alone[0]=result['toxicity_score'],result['isolated_score']
        if weights[1] > 0 or weights[2] > 0:
            mixed=self._expert('mixed')
            result=mixed.predict(text,history,False)
            full[1],alone[1]=result['toxicity_score'],result['isolated_score']
            if weights[2] > 0:
                w=mixed.bundle['weights'].get('contextual_transformer',0)
                if w <= 0: raise ValueError('Mixed neural expert unavailable')
                full[2]=result['component_scores']['contextual_transformer']
                alone[2]=np.clip(full[2]+(alone[1]-full[1])/w,0,1)
        score=float(margin_score(full@weights,branch['threshold']))
        isolated=float(margin_score(alone@weights,branch['threshold']))
        for i,name in enumerate(EXPERTS):
            if weights[i] > 0: components[name]=float(full[i])
        return {'toxicity_score':score,'is_toxic':score>=.5,'threshold':.5,
                'isolated_score':isolated,'context_score_delta':score-isolated,
                'turns_used':len(history)+1,'ancestors_omitted':original_length-len(history),
                'turn_attention':[], 'embeddings_available':False,'has_context_training':True,
                'policy':policy['name'],'script_bucket':bucket,'language_identified':False,
                'expert_weights':dict(zip(EXPERTS,weights.tolist())), 'component_scores':components,
                'score_semantics':'Threshold-normalized decision score; not calibrated probability',
                'attention_scope':'No combined attention or embedding is defined across experts'}

    def predict_thread(self,messages,target_id,include_embeddings=False):
        from .inference import Predictor
        return Predictor.predict_thread(self,messages,target_id,include_embeddings)
