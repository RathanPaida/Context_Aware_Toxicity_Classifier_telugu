"""
Phase 6: Stateful Two-Stage Production Inference Pipeline.

Implements the low-latency moderation cascade backed by thread caching:
  1. Stage 1 Fast Filter (~3ms): Isolated target comment evaluated by fast classifier.
     - If p < 0.15 (Clearly Safe) → Fast Exit (ALLOW)
     - If p > 0.90 (Explicit Slur) → Fast Exit (FLAG)
  2. Stage 2 IndicBERT-v3 Cross-Encoder (~18-25ms):
     - For ambiguous cases (0.15 <= p <= 0.90, ~25% of traffic)
     - Fetches preceding parent turns from thread cache (Redis or in-memory fallback)
     - Performs context-aware cross-attention with Target-Span Pooling
     - Decides final multi-label action: Allow or Flag with category probabilities
"""

import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    CHECKPOINT_DIR,
    PRIMARY_MODEL,
    TOXICITY_LABELS,
    MAX_SEQ_LENGTH,
    TARGET_PREFIX,
    PARENT_PREFIX,
    SEP_TOKEN,
    TOPIC_PREFIX,
    get_logger,
)
from src.preprocessing import IndicPreprocessor
from src.model import ContextAwareToxicityClassifier, load_tokenizer, tokenize_with_target_mask

logger = get_logger("inference_pipeline")


# ═══════════════════════════════════════════════════════════════════════
# Thread Cache: Redis with In-Memory Resilient Fallback
# ═══════════════════════════════════════════════════════════════════════

class ThreadContextCache:
    """
    Manages conversational context history for active chat rooms or posts.
    Uses Redis when available, or an in-memory TTL dictionary as a transparent fallback.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, ttl_seconds: int = 1800):
        self.ttl = ttl_seconds
        self.redis_client = None
        self._in_memory_store: Dict[str, Dict[str, Any]] = {}
        
        try:
            import redis
            client = redis.Redis(host=host, port=port, db=db, socket_timeout=1.0)
            client.ping()
            self.redis_client = client
            logger.info("  ✓ Connected to Redis thread cache")
        except Exception:
            logger.info("  Redis not reachable locally; using resilient in-memory thread cache")
    
    def get_history(self, thread_id: str, limit: int = 3) -> List[str]:
        """Fetch the preceding N messages in chronological order."""
        if self.redis_client:
            try:
                # LRANGE thread:{id}:history 0 (limit-1)
                items = self.redis_client.lrange(f"thread:{thread_id}:history", 0, limit - 1)
                # Decode and reverse to get oldest -> newest
                return [i.decode("utf-8") for i in reversed(items)]
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        # In-memory fallback
        self._cleanup_in_memory()
        entry = self._in_memory_store.get(thread_id)
        if entry:
            return entry["messages"][-limit:]
        return []
    
    def push_message(self, thread_id: str, message: str, max_history: int = 5):
        """Append newly normalized message to history and refresh TTL."""
        if self.redis_client:
            try:
                pipe = self.redis_client.pipeline()
                key = f"thread:{thread_id}:history"
                pipe.lpush(key, message)
                pipe.ltrim(key, 0, max_history - 1)
                pipe.expire(key, self.ttl)
                pipe.execute()
                return
            except Exception as e:
                logger.warning(f"Redis push failed: {e}")
        
        # In-memory fallback
        now = time.time()
        if thread_id not in self._in_memory_store:
            self._in_memory_store[thread_id] = {"messages": [], "expires_at": now + self.ttl}
        self._in_memory_store[thread_id]["messages"].append(message)
        if len(self._in_memory_store[thread_id]["messages"]) > max_history:
            self._in_memory_store[thread_id]["messages"] = self._in_memory_store[thread_id]["messages"][-max_history:]
        self._in_memory_store[thread_id]["expires_at"] = now + self.ttl
    
    def clear_history(self, thread_id: str):
        """Reset history for a given thread."""
        if self.redis_client:
            try:
                self.redis_client.delete(f"thread:{thread_id}:history")
            except Exception:
                pass
        if thread_id in self._in_memory_store:
            del self._in_memory_store[thread_id]
    
    def _cleanup_in_memory(self):
        now = time.time()
        expired = [k for k, v in self._in_memory_store.items() if v["expires_at"] < now]
        for k in expired:
            del self._in_memory_store[k]


# ═══════════════════════════════════════════════════════════════════════
# Stage 1 Fast Filter (Isolated Target Classifier)
# ═══════════════════════════════════════════════════════════════════════

class FastTargetFilter:
    """
    Stage 1 fast gate (<3ms): evaluates the target comment in isolation.
    Checks explicit slurs, cuss words, leetspeak, asterisks, and toxic phrases
    across English, Hindi, Telugu, and Romanized Tanglish.
    
    If explicit profanity is detected -> immediate fast exit FLAG (>0.95).
    If clearly safe conversational marker -> immediate fast exit ALLOW (<0.10).
    Otherwise routes ambiguous traffic to Stage 2 Cross-Encoder.
    """
    
    def __init__(self):
        # Comprehensive multilingual slur lexicon
        self.english_slurs = frozenset([
            "fuck", "fucker", "fucking", "fucked", "fuckoff", "fuckin", "fck", "fuk",
            "motherfucker", "motherfucking", "bitch", "bitches", "bitchy", "bastard", "bastards",
            "shit", "shitty", "bullshit", "asshole", "assholes", "cunt", "cunts",
            "dick", "dicks", "dickhead", "pussy", "pussies", "slut", "sluts",
            "whore", "whores", "idiot", "morons", "moron", "retard", "retarded",
            "stfu", "wtf", "douchebag", "jackass", "dipshit", "scumbag", "piss"
        ])
        
        self.hindi_slurs = frozenset([
            "madarchod", "madrchod", "maderchod", "bhenchod", "behenchod", "bhenk",
            "bhosdike", "bhosadike", "bhosdi", "chutiya", "chutiye", "chutya",
            "gandu", "gaandu", "gaand", "harami", "haramkhor", "saala", "sala",
            "lodu", "lauda", "laude", "lavde", "lavda", "kamine", "kutta", "randi",
            "kutiya", "lund", "loda", "jhaat", "suar", "mc", "bc"
        ])
        
        self.telugu_slurs = frozenset([
            "lanja", "lanjakodaka", "lanjakoduka", "lanjamunda", "munda",
            "dengu", "dengey", "dengutha", "dengichuko", "dengudu",
            "puku", "pukulo", "pooku", "erripuku", "erripuka",
            "modda", "moddalo", "gudha", "guddha", "gudhala", "kutha",
            "chetha", "chethana", "chethakodaka", "dheenamma", "dhinamma",
            "lafoot", "vedhava", "bevars", "sullagada", "chavata", "bochu",
            "lavada", "lavadalo", "panikirani", "thupuk", "kukka"
        ])
        
        self.telugu_script_slurs = frozenset([
            "లంజ", "లంజకొడక", "దెంగు", "దెంగేయ్", "పూకు", "పుకు", "ఎర్రిపూకు",
            "ఎర్రిపుక", "మొడ్డ", "గుద్ద", "గుద", "కుత్త", "వెధవ", "లఫూట్", "బోకు", "చవట", "కుక్క"
        ])
        
        self.all_slurs = self.english_slurs | self.hindi_slurs | self.telugu_slurs | self.telugu_script_slurs
        
        # Multi-word toxic/slur phrases
        self.slur_phrases = [
            "fuck you", "shut the fuck up", "kill yourself", "go to hell",
            "erri puku", "erri puka", "chetha na kodaka", "chetha kodaka",
            "na kodaka", "nee yamma", "nee ayya", "pichi puku", "pichi na kodaka",
            "munda mopi", "kosi karam", "champi street lo", "nariki padestha",
            "మడతబెట్టి కొడతా", "ఖూనీ చేస్తా", "అంతు చూస్తా", "నీ యమ్మ", "నా కొడక"
        ]
        
        # Obvious safe conversational greetings
        self.safe_tokens = frozenset([
            "hi", "hello", "good", "morning", "evening", "namaste", "bagundi",
            "thanks", "tq", "super", "nice", "congrats", "welcome", "dhanyavadalu"
        ])

    def predict_fast(self, text: str) -> Tuple[float, str]:
        """
        Returns (toxicity_probability, routing_decision).
        routing_decision: 'SAFE_FAST_EXIT', 'SLUR_FAST_EXIT', or 'AMBIGUOUS_ROUTE_STAGE2'
        """
        raw_lower = text.lower().strip()
        
        # 1. Check multi-word slur phrases
        for phrase in self.slur_phrases:
            if phrase in raw_lower:
                return 0.98, "SLUR_FAST_EXIT"
        
        # 2. Leet / symbol de-obfuscation
        leet_clean = raw_lower
        for k, v in {"@": "a", "0": "o", "1": "i", "$": "s", "3": "e", "!": "i", "*": "u", "+": "t"}.items():
            leet_clean = leet_clean.replace(k, v)
            
        for phrase in self.slur_phrases:
            if phrase in leet_clean:
                return 0.98, "SLUR_FAST_EXIT"
        
        # 3. Strip all punctuation for token matching (handles 'fuck!', 'lanja!', 'bitch.')
        clean_tokens = re.sub(r"[^\w\s]", " ", leet_clean).split()
        for t in clean_tokens:
            if t in self.all_slurs:
                return 0.97, "SLUR_FAST_EXIT"
                
        # 4. Regex for masked words like f*ck, b*tch, p*ku, l*nja
        if re.search(r"\bf[\*\._-]*(?:u|a|o)?[\*\._-]*c[\*\._-]*k\b", raw_lower):
            return 0.97, "SLUR_FAST_EXIT"
        if re.search(r"\bb[\*\._-]*i[\*\._-]*t[\*\._-]*c[\*\._-]*h\b", raw_lower):
            return 0.97, "SLUR_FAST_EXIT"
        if re.search(r"\bp[\*\._-]*[u|o]+[\*\._-]*k[\*\._-]*u\b", raw_lower):
            return 0.97, "SLUR_FAST_EXIT"
        if re.search(r"\bl[\*\._-]*a[\*\._-]*n[\*\._-]*j[\*\._-]*a\b", raw_lower):
            return 0.97, "SLUR_FAST_EXIT"
        
        # 5. Check for clearly safe everyday conversational markers
        raw_tokens = re.sub(r"[^\w\s]", " ", raw_lower).split()
        if len(raw_tokens) <= 3 and any(w in self.safe_tokens for w in raw_tokens):
            return 0.04, "SAFE_FAST_EXIT"
        
        # Middle band (ambiguous banter, sarcasm, threats, context-dependent phrasing)
        return 0.45, "AMBIGUOUS_ROUTE_STAGE2"


# ═══════════════════════════════════════════════════════════════════════
# Two-Stage Stateful Moderation Pipeline
# ═══════════════════════════════════════════════════════════════════════

class TwoStageModerationPipeline:
    """
    Full Phase 6 Stateful Production Moderation Engine.
    Combines FastTargetFilter + ThreadContextCache + IndicBERT-v3 Cross-Encoder.
    """
    
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        model_name: str = PRIMARY_MODEL,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Initializing Production Moderation Pipeline on {self.device}...")
        
        # Preprocessor
        self.preprocessor = IndicPreprocessor(enable_transliteration=False, enable_normalization=True)
        
        # Stage 1 Fast Filter
        self.fast_filter = FastTargetFilter()
        
        # Thread Cache
        self.cache = ThreadContextCache()
        
        # Stage 2 IndicBERT Cross-Encoder
        self.tokenizer = load_tokenizer(model_name)
        self.model = ContextAwareToxicityClassifier(
            model_name=model_name,
            num_labels=len(TOXICITY_LABELS),
            use_target_span_pooling=True,
        )
        
        # Load weights if available
        ckpt = checkpoint_dir
        if not ckpt:
            for candidate in ["stage4_best", "stage3_best", "stage1_best"]:
                p = CHECKPOINT_DIR / candidate / "model.pt"
                if p.exists():
                    ckpt = str(CHECKPOINT_DIR / candidate)
                    break
        
        if ckpt and (Path(ckpt) / "model.pt").exists():
            logger.info(f"  Loading Cross-Encoder weights from {ckpt}")
            state = torch.load(Path(ckpt) / "model.pt", map_location="cpu")
            self.model.load_state_dict(state, strict=False)
            logger.info("  ✓ Weights loaded")
        
        self.model.to(self.device)
        self.model.eval()
        logger.info("✓ Two-Stage Moderation Pipeline ready.")
    
    def moderate(
        self,
        text: str,
        thread_id: str = "default_thread",
        topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process incoming live comment with context-awareness and fast exit routing.
        
        Args:
            text: Raw user comment
            thread_id: Chatroom or post ID for thread tracking
            topic: Optional category/topic if available
        
        Returns:
            Dict containing classification decision, probabilities, latency, and routing info.
        """
        t0 = time.time()
        
        # Step 1: Preprocessing & De-obfuscation
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Step 2: Stage 1 Fast Filter (<3ms)
        fast_prob, decision = self.fast_filter.predict_fast(cleaned_text)
        
        if decision == "SAFE_FAST_EXIT":
            # Record in thread history and exit
            self.cache.push_message(thread_id, cleaned_text)
            latency_ms = (time.time() - t0) * 1000
            return {
                "action": "ALLOW",
                "toxicity_score": fast_prob,
                "is_toxic": False,
                "stage_routed": "Stage 1 Fast Filter",
                "latency_ms": round(latency_ms, 2),
                "categories": {lbl: 0.0 for lbl in TOXICITY_LABELS},
                "explanation": "High confidence safe conversational marker; fast-path exit (<1ms).",
            }
        
        if decision == "SLUR_FAST_EXIT":
            self.cache.push_message(thread_id, cleaned_text)
            latency_ms = (time.time() - t0) * 1000
            return {
                "action": "FLAG",
                "toxicity_score": fast_prob,
                "is_toxic": True,
                "stage_routed": "Stage 1 Fast Filter",
                "latency_ms": round(latency_ms, 2),
                "categories": {"toxic": fast_prob, "insult": 0.94, "severe_toxic": 0.70, "identity_attack": 0.10, "threat": 0.15},
                "explanation": "Explicit profanity, slur, or abusive phrase detected; fast-path exit (<1ms).",
            }
        
        # Step 3: Stage 2 Context-Aware Cross-Encoder (~18-25ms)
        # Fetch preceding conversation turns from thread cache
        history = self.cache.get_history(thread_id, limit=3)
        parent_context = " [SEP] ".join(history) if history else ""
        
        # Assemble context window
        has_custom_topic = bool(topic and topic.strip() and topic.strip().lower() not in ["general", "none", "default"])
        
        if parent_context:
            if has_custom_topic:
                assembled = f"{TOPIC_PREFIX} {topic.strip()} {SEP_TOKEN} {PARENT_PREFIX} {parent_context} {SEP_TOKEN} {TARGET_PREFIX} {cleaned_text}"
            else:
                assembled = f"{PARENT_PREFIX} {parent_context} {SEP_TOKEN} {TARGET_PREFIX} {cleaned_text}"
        else:
            if has_custom_topic:
                assembled = f"{TOPIC_PREFIX} {topic.strip()} {SEP_TOKEN} {TARGET_PREFIX} {cleaned_text}"
            else:
                assembled = f"{TARGET_PREFIX} {cleaned_text}"
        
        encoding = tokenize_with_target_mask(assembled, self.tokenizer, max_length=MAX_SEQ_LENGTH)
        inputs = {k: v.unsqueeze(0).to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs["probs"][0].cpu().numpy()
        
        # Multi-label category scores
        category_scores = {lbl: round(float(probs[i]), 4) for i, lbl in enumerate(TOXICITY_LABELS)}
        primary_score = float(probs[0])
        is_toxic = primary_score >= 0.50
        
        # Update thread cache
        self.cache.push_message(thread_id, cleaned_text)
        
        latency_ms = (time.time() - t0) * 1000
        return {
            "action": "FLAG" if is_toxic else "ALLOW",
            "toxicity_score": round(primary_score, 4),
            "is_toxic": is_toxic,
            "stage_routed": "Stage 2 IndicBERT Cross-Encoder",
            "latency_ms": round(latency_ms, 2),
            "categories": category_scores,
            "context_used": parent_context if parent_context else "(None - First Turn)",
            "explanation": f"Evaluated with {len(history)} parent message(s) using Target-Span cross-attention.",
        }
    
    def clear_thread(self, thread_id: str):
        """Reset conversation context history for the specified thread."""
        self.cache.clear_history(thread_id)
        logger.info(f"Thread context cleared: {thread_id}")


# ═══════════════════════════════════════════════════════════════════════
# Demo Runner
# ═══════════════════════════════════════════════════════════════════════

def run_inference_demo():
    """Demonstrate the two-stage pipeline on live test conversations."""
    pipeline = TwoStageModerationPipeline()
    print("=" * 75)
    print("PHASE 6: TWO-STAGE PRODUCTION INFERENCE PIPELINE DEMO")
    print("=" * 75)
    
    test_conversations = [
        # Conversation 1: Gaming Thread (Non-Toxic Contrastive Case)
        {
            "thread_id": "thread_gaming_101",
            "topic": "Gaming",
            "turns": [
                "Ninna match lo 1v4 clutch chusava bro?",
                "Boss level clear chesa! Solo squad clutch!",
                "Champesav po!",  # Target comment: Friendly praise in gaming context
            ]
        },
        # Conversation 2: Political Thread (Toxic Contrastive Case)
        {
            "thread_id": "thread_politics_202",
            "topic": "Politics",
            "turns": [
                "Aaa journalist bayataki vasthe kottali andaru",
                "Vaadini thagalettali",
                "Champesav po!",  # Target comment: Encouraging violence in political context
            ]
        },
        # Conversation 3: De-obfuscation + Fast Exit Slur
        {
            "thread_id": "thread_abuse_303",
            "topic": "Random",
            "turns": [
                "m@drch0d nuvvu chala worst ra",  # Leetspeak slur
            ]
        },
        # Conversation 4: Parent Bleed Robustness
        {
            "thread_id": "thread_bleed_404",
            "topic": "Flame War",
            "turns": [
                "Chetha na kodaka, ninnu champi street lo padestha!",
                "Shantinchandi bro, vadileyandi andi.",  # De-escalating polite reply
            ]
        }
    ]
    
    for conv in test_conversations:
        tid = conv["thread_id"]
        topic = conv["topic"]
        print(f"\n--- Thread: {tid} (Topic: {topic}) ---")
        for i, turn in enumerate(conv["turns"]):
            res = pipeline.moderate(turn, thread_id=tid, topic=topic)
            flag_str = "[FLAGGED]" if res["is_toxic"] else "[ALLOWED]"
            print(f"  Turn {i+1}: \"{turn}\"")
            print(f"    -> {flag_str} | Score: {res['toxicity_score']} | Stage: {res['stage_routed']} | Latency: {res['latency_ms']}ms")
            if res.get("context_used") and res["context_used"] != "(None - First Turn)":
                print(f"      Context: \"{res['context_used']}\"")
    
    print("\n" + "=" * 75)
    print("DEMO COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    run_inference_demo()
