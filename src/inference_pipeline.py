"""
Phase 6: Stateful Two-Stage Moderation Pipeline.

  1. Lexicon gate (<1 ms): severe slurs in a message with no thread context and no
     praise cue are flagged immediately. Mild / ambiguous terms are only reported.
  2. Context-aware cross-encoder: everything else is scored by the trained model with
     the preceding thread turns as context. Decision thresholds come from the
     checkpoint's metadata (tuned on validation), not from constants in this file.

Removed from the old version: the "safe fast exit" (substring match on phrases like
"good night" let "good night, I will kill you" through), hard-coded category scores
for lexicon hits, fake attention values for fast exits, and silent use of an
untrained model when no checkpoint exists.
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import LABEL_DISPLAY_NAMES, LEXICON_PATH, SCENARIOS_PATH, TOXICITY_LABELS, InferenceConfig, get_logger
from src.model import (
    ContextAwareToxicityClassifier,
    clean_token,
    encode_example,
    find_best_checkpoint,
    load_classifier_from_checkpoint,
    load_tokenizer,
    to_model_inputs,
)
from src.preprocessing import IndicPreprocessor

logger = get_logger("inference_pipeline")

SEVERE_SUBLABELS = ("severe_toxic", "threat", "identity_attack")


class ModelNotReadyError(RuntimeError):
    pass


# ═══════════════════════════════════════════════════════════════════════
# Thread Cache: Redis (opt-in) with in-memory fallback
# ═══════════════════════════════════════════════════════════════════════

class ThreadContextCache:
    def __init__(self, cfg: InferenceConfig):
        self.ttl = cfg.cache_ttl_seconds
        self.max_history = cfg.max_history_stored
        self.redis_client = None
        self._store: Dict[str, Dict[str, Any]] = {}
        if cfg.use_redis:
            try:
                import redis
                client = redis.Redis(host=cfg.redis_host, port=cfg.redis_port, socket_timeout=1.0)
                client.ping()
                self.redis_client = client
                logger.info(f"  ✓ Redis thread cache at {cfg.redis_host}:{cfg.redis_port}")
            except Exception as e:
                logger.warning(f"  Redis unavailable ({e}); using in-memory cache")

    def get_history(self, thread_id: str, limit: int) -> List[str]:
        if self.redis_client:
            try:
                items = self.redis_client.lrange(f"thread:{thread_id}:history", 0, limit - 1)
                return [i.decode("utf-8") for i in reversed(items)]
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        self._expire()
        entry = self._store.get(thread_id)
        return entry["messages"][-limit:] if entry else []

    def push_message(self, thread_id: str, message: str):
        if self.redis_client:
            try:
                key = f"thread:{thread_id}:history"
                pipe = self.redis_client.pipeline()
                pipe.lpush(key, message)
                pipe.ltrim(key, 0, self.max_history - 1)
                pipe.expire(key, self.ttl)
                pipe.execute()
                return
            except Exception as e:
                logger.warning(f"Redis push failed: {e}")
        now = time.time()
        entry = self._store.setdefault(thread_id, {"messages": [], "expires_at": now + self.ttl})
        entry["messages"] = (entry["messages"] + [message])[-self.max_history:]
        entry["expires_at"] = now + self.ttl

    def clear_history(self, thread_id: str):
        if self.redis_client:
            try:
                self.redis_client.delete(f"thread:{thread_id}:history")
            except Exception:
                pass
        self._store.pop(thread_id, None)

    def _expire(self):
        now = time.time()
        for k in [k for k, v in self._store.items() if v["expires_at"] < now]:
            del self._store[k]


# ═══════════════════════════════════════════════════════════════════════
# Lexicon gate
# ═══════════════════════════════════════════════════════════════════════

_PUNCT = re.compile(r"[^\w\sఀ-౿]", re.UNICODE)  # keep Telugu block incl. vowel signs


class FastTargetFilter:
    """Lexicon lookup driven by src/resources/lexicons.json."""

    def __init__(self, lexicon_path: Path = LEXICON_PATH):
        lex = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
        self.severe = {w.lower() for words in lex["severe"].values() for w in words}
        self.mild = {w.lower() for words in lex["mild"].values() for w in words}
        self.severe_phrases = [p.lower() for p in lex.get("severe_phrases", [])]
        self.mild_phrases = [p.lower() for p in lex.get("mild_phrases", [])]
        self.praise_cues = [p.lower() for p in lex.get("praise_cues", [])]
        self.masked = [re.compile(p) for p in lex.get("masked_patterns", [])]

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(_PUNCT.sub(" ", text.lower()).split())

    def analyze(self, text: str) -> Dict[str, Any]:
        raw = text.lower()
        norm = self._normalize(text)
        padded = f" {norm} "
        tokens = set(norm.split())
        severe = sorted(tokens & self.severe) + [p for p in self.severe_phrases if f" {p} " in padded]
        severe += [m.group(0) for rx in self.masked for m in [rx.search(raw)] if m]
        mild = sorted(tokens & self.mild) + [p for p in self.mild_phrases if f" {p} " in padded]
        praise = [c for c in self.praise_cues if c in norm]
        return {"severe": sorted(set(severe)), "mild": sorted(set(mild)), "praise_cues": praise}


# ═══════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════

class TwoStageModerationPipeline:

    def __init__(self, checkpoint_dir: Optional[str] = None, device: Optional[str] = None,
                 config: Optional[InferenceConfig] = None):
        self.cfg = config or InferenceConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.preprocessor = IndicPreprocessor(enable_transliteration=False, enable_normalization=True)
        self.fast_filter = FastTargetFilter()
        self.cache = ThreadContextCache(self.cfg)

        self.model: Optional[ContextAwareToxicityClassifier] = None
        self.tokenizer = None
        self.metadata: Dict[str, Any] = {}
        self.checkpoint: Optional[Path] = Path(checkpoint_dir) if checkpoint_dir else find_best_checkpoint()
        self.trained = False

        if self.checkpoint and (self.checkpoint / "model.pt").exists():
            self.model, self.tokenizer, self.metadata = load_classifier_from_checkpoint(self.checkpoint, self.device)
            self.trained = True
            src_stage = self.metadata.get("release", {}).get("source", self.checkpoint.name)
            self.status = f"loaded {self.checkpoint.name}" + (f" ({src_stage})" if src_stage != self.checkpoint.name else "")
        elif self.cfg.require_checkpoint:
            self.status = ("no trained checkpoint found in models/checkpoints/ — train first "
                           "(python run_pipeline.py --all) or set ALLOW_UNTRAINED=1 for UI testing")
            logger.error(self.status)
        else:
            logger.warning("ALLOW_UNTRAINED=1: serving an UNTRAINED model — predictions are meaningless")
            self.model = ContextAwareToxicityClassifier().to(self.device).eval()
            self.tokenizer = load_tokenizer(self.model.model_name)
            self.status = "untrained (ALLOW_UNTRAINED=1)"

        self.thresholds = {l: float(self.metadata.get("thresholds", {}).get(l, self.cfg.toxic_threshold))
                           for l in TOXICITY_LABELS}
        self.max_len = int(self.metadata.get("max_seq_length", self.cfg.max_seq_length))
        logger.info(f"Pipeline ready on {self.device}: {self.status} | thresholds={self.thresholds}")

    @property
    def ready(self) -> bool:
        return self.model is not None

    def info(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "trained": self.trained,
            "status": self.status,
            "release_source": self.metadata.get("release", {}).get("source"),
            "label_display_names": LABEL_DISPLAY_NAMES,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "model_name": getattr(self.model, "model_name", None),
            "num_heads": getattr(self.model, "num_heads", None),
            "device": str(self.device),
            "thresholds": self.thresholds,
            "trained_at": self.metadata.get("timestamp"),
            "stage": self.metadata.get("stage"),
        }

    # ── main entry point ──────────────────────────────────────────────
    def moderate(self, text: str, thread_id: Optional[str] = None, topic: Optional[str] = None,
                 update_history: bool = False, history_override: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Classify `text` given the thread's preceding turns.
        `history_override` supplies explicit context (oldest → newest) without touching the cache.
        """
        t0 = time.perf_counter()
        cleaned = self.preprocessor.clean_text(text)

        standalone = thread_id in (None, "", "default_thread")
        if history_override is not None:
            history = [self.preprocessor.clean_text(h) for h in history_override if h and h.strip()]
        elif standalone:
            history = []
        else:
            history = self.cache.get_history(thread_id, self.cfg.history_turns)
        history = history[-self.cfg.history_turns:]
        topic = topic.strip() if topic and topic.strip().lower() not in {"general", "none", "default"} else None

        lex = self.fast_filter.analyze(cleaned)
        context_used = " ⏎ ".join(history) if history else "(none — first turn)"

        def finish(result: Dict[str, Any]) -> Dict[str, Any]:
            if update_history and not standalone and history_override is None:
                if not history or history[-1] != cleaned:
                    self.cache.push_message(thread_id, cleaned)
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            result["lexicon"] = lex
            result["context_used"] = context_used
            result["thresholds"] = self.thresholds
            result["untrained"] = not self.trained
            result.setdefault("needs_review", False)
            return result

        # ── Stage 1: lexicon fast exit ──
        if self.cfg.lexicon_fast_exit and lex["severe"] and not history and not lex["praise_cues"]:
            score = self.cfg.lexicon_flag_score
            return finish({
                "action": "FLAG",
                "is_toxic": True,
                "toxicity_score": score,
                "stage_routed": "Stage 1 Lexicon Gate",
                "decided_by": "word list",
                "categories": {l: (score if l == "toxic" else None) for l in TOXICITY_LABELS},
                "reason": f"Contains a severe abusive word ({', '.join(lex['severe'])}) and there is no earlier message that changes its meaning.",
                "explanation": f"Severe term(s) {lex['severe']} with no mitigating context; model not consulted.",
                "attention_analysis": None,
            })

        if not self.ready:
            raise ModelNotReadyError(self.status)

        # ── Stage 2: model ──
        enc = encode_example(self.tokenizer, cleaned, history=history, topic=topic, max_length=self.max_len)
        with torch.no_grad():
            out = self.model(**to_model_inputs(enc, self.device), return_attention=True)
        probs = out["probs"][0].float().cpu().numpy()
        cats = {l: round(float(probs[i]), 4) for i, l in enumerate(TOXICITY_LABELS)}

        reasons = []
        if cats["toxic"] >= self.thresholds["toxic"]:
            reasons.append(f"toxic {cats['toxic']:.2f} ≥ {self.thresholds['toxic']:.2f}")
        for l in SEVERE_SUBLABELS:
            th = max(self.thresholds[l], self.cfg.subtype_flag_threshold)
            if cats[l] >= th:
                reasons.append(f"{l} {cats[l]:.2f} ≥ {th:.2f}")
        is_toxic = bool(reasons)
        score = max([cats["toxic"]] + ([cats[l] for l in SEVERE_SUBLABELS] if is_toxic else []))

        if is_toxic:
            explanation = "Flagged by model: " + "; ".join(reasons) + "."
        else:
            explanation = f"Model score toxic={cats['toxic']:.2f} below threshold {self.thresholds['toxic']:.2f}."
        if lex["severe"] and not is_toxic:
            explanation += f" Lexicon terms {lex['severe']} were present but the model judged them benign in context."
        elif lex["mild"]:
            explanation += f" Mild/ambiguous terms noted: {lex['mild']}."

        needs_review = abs(cats["toxic"] - self.thresholds["toxic"]) <= self.cfg.review_margin
        top = [LABEL_DISPLAY_NAMES[l].lower() for l in TOXICITY_LABELS[1:] if cats[l] >= self.thresholds[l]]
        if is_toxic:
            reason = "The model rates this message as harmful" + (f" ({', '.join(top)})" if top else "")
        else:
            reason = "The model does not consider this message harmful"
        reason += " given the earlier messages." if history else "."
        if needs_review:
            reason += " The score is close to the decision line, so a person should double-check."
        if lex["severe"] and not is_toxic:
            reason += f" It contains '{', '.join(lex['severe'])}', but the context suggests it is not meant as abuse."

        return finish({
            "action": "FLAG" if is_toxic else "ALLOW",
            "is_toxic": is_toxic,
            "toxicity_score": round(score, 4),
            "stage_routed": f"Stage 2 Cross-Encoder ({self.model.num_heads}-head attention)",
            "decided_by": "model",
            "categories": cats,
            # score at which each label alone flags the message (None = cannot flag on its own)
            "flag_lines": {l: (self.thresholds[l] if l == "toxic" else
                               max(self.thresholds[l], self.cfg.subtype_flag_threshold) if l in SEVERE_SUBLABELS else None)
                           for l in TOXICITY_LABELS},
            "needs_review": needs_review,
            "reason": reason,
            "explanation": explanation,
            "attention_analysis": self._attention(enc, out),
        })

    def _attention(self, enc: Dict[str, List[int]], out: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        weights = out.get("token_attention")
        if weights is None:
            return None
        w = weights[0].float().cpu().numpy()
        special = set(self.tokenizer.all_special_ids)
        toks = self.tokenizer.convert_ids_to_tokens(enc["input_ids"])
        items, ctx_sum, tgt_sum = [], 0.0, 0.0
        for tid, tok, weight, is_t, seg in zip(enc["input_ids"], toks, w, enc["target_mask"], enc["segments"]):
            if tid in special or seg in ("special", "prefix"):
                continue
            display = clean_token(tok)
            if not display:
                continue
            items.append({"token": display, "attention": round(float(weight), 4), "is_target": bool(is_t),
                          # WordPiece "##x" continues the previous word; SentencePiece marks word *starts* with "▁"
                          "continuation": tok.startswith("##"), "segment": seg})
            if is_t:
                tgt_sum += float(weight)
            else:
                ctx_sum += float(weight)
        total = ctx_sum + tgt_sum
        top = sorted(items, key=lambda x: x["attention"], reverse=True)[:5]
        return {
            "num_heads": self.model.num_heads,
            # Share of the whole-text pooling attention that falls on context tokens.
            # A rough saliency proxy, not a causal measure of context influence.
            "context_influence_pct": round(100 * ctx_sum / total, 1) if total > 0 else 0.0,
            "top_attended_words": [t["token"] for t in top],
            "tokens": items,
        }

    def clear_thread(self, thread_id: str):
        self.cache.clear_history(thread_id)


# ═══════════════════════════════════════════════════════════════════════
# Demo: run the scenarios from src/resources/scenarios.json
# ═══════════════════════════════════════════════════════════════════════

def run_inference_demo():
    pipeline = TwoStageModerationPipeline()
    if not pipeline.ready:
        print(f"Cannot run demo: {pipeline.status}")
        return
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    passed = 0
    for sc in scenarios:
        pipeline.clear_thread(sc["thread_id"])
        print(f"\n--- {sc['title']} (topic: {sc['topic']}) ---")
        res = None
        for turn in sc["turns"]:
            res = pipeline.moderate(turn["text"], thread_id=sc["thread_id"], topic=sc["topic"], update_history=True)
            print(f"  {turn['speaker']}: {turn['text']}")
            print(f"    → {res['action']} score={res['toxicity_score']} [{res['stage_routed']}] {res['latency_ms']}ms")
        got = "FLAGGED" if res["is_toxic"] else "ALLOWED"
        ok = got == sc["expected"]
        passed += ok
        print(f"  expected {sc['expected']}, got {got} → {'OK' if ok else 'MISMATCH'}")
    print(f"\n{passed}/{len(scenarios)} demo scenarios match expectations")


if __name__ == "__main__":
    run_inference_demo()
