"""
Phase 2: Indic-Specific Preprocessing & Script Unification Pipeline.

Implements the deterministic 4-stage preprocessing pipeline:
  1. De-Obfuscation (Regex L33t normalization, URL/handle stripping)
  2. Transliteration (ai4bharat/indic-xlit: Romanized Telugu → Native Script)
  3. Script Normalization (indic-nlp-library: Unicode canonicalization)
  (Context window assembly is done on token ids in src/model.py:encode_example)
"""

import re
import unicodedata
from typing import List

from src.config import LEET_MAP, get_logger

logger = get_logger("preprocessing")


# ═══════════════════════════════════════════════════════════════════════
# Stage 1: De-Obfuscation & Adversarial Normalization
# ═══════════════════════════════════════════════════════════════════════

# Pre-compiled patterns for performance
_URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|bit\.ly/\S+|t\.co/\S+", re.IGNORECASE
)
_HANDLE_PATTERN = re.compile(r"(?:^|(?<=\s))@[a-zA-Z0-9_]+\b")
_HASHTAG_CLEAN = re.compile(r"#(\w+)")  # Keep word, drop #

# Leetspeak: only substitute when digit/symbol is embedded in alphabetic context
# e.g., "m@drch0d" → "madrchod" but "user@email" stays
_LEET_EMBEDDED_PATTERN = re.compile(
    r"(?<=[a-zA-Z])([0134579@$!+])(?=[a-zA-Z])"
)

# Repeated characters: >2 consecutive identical chars → 2
_CHAR_REPEAT_PATTERN = re.compile(r"(.)\1{2,}")

# Whitespace normalization
_MULTI_SPACE = re.compile(r"\s+")


def _leet_replace(match: re.Match) -> str:
    """Replace leetspeak character with its alphabetic equivalent."""
    char = match.group(1)
    return LEET_MAP.get(char, char)


def deobfuscate(text: str) -> str:
    """
    Stage 1: Remove adversarial obfuscation from text.
    
    Operations:
      - Strip URLs and @handles
      - Clean hashtags (keep word)
      - Replace embedded leetspeak digits/symbols with letters
      - Collapse >2 repeating characters to 2
      - Normalize whitespace
      - Normalize Unicode to NFC form
    
    Args:
        text: Raw input text (potentially obfuscated)
    
    Returns:
        Cleaned, de-obfuscated text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Strip URLs
    text = _URL_PATTERN.sub(" ", text)
    
    # Strip @handles (only when preceded by start of string or whitespace)
    text = _HANDLE_PATTERN.sub(" ", text)
    
    # Clean hashtags: #CoolMovie → CoolMovie
    text = _HASHTAG_CLEAN.sub(r"\1", text)
    
    # Replace leading $ before letters (e.g. $hit → shit)
    text = re.sub(r"(?:^|(?<=\s))\$([a-zA-Z]+)", r"s\1", text)
    
    # Leetspeak normalization (embedded in alphabetic strings)
    # Apply iteratively to handle adjacent leet chars: "m@drch0d" → "madrchod"
    prev = ""
    while prev != text:
        prev = text
        text = _LEET_EMBEDDED_PATTERN.sub(_leet_replace, text)
    
    # Collapse runs of >2 identical chars to 2 ("chalaaaaa" → "chalaa", "gooood" → "good").
    # Collapsing to 1 corrupts real words ("gooood" → "god").
    text = _CHAR_REPEAT_PATTERN.sub(r"\1\1", text)
    
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    
    # Normalize whitespace
    text = _MULTI_SPACE.sub(" ", text).strip()
    
    return text


# ═══════════════════════════════════════════════════════════════════════
# Stage 2: Selective Phonetic Transliteration (ai4bharat/indic-xlit)
# ═══════════════════════════════════════════════════════════════════════

# --- Language Detection Helpers ---

# Common English words that should NOT be transliterated (top ~500 by frequency)
# This is a fast heuristic; we expand it with a dictionary check at runtime
_ENGLISH_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "me", "my",
    "we", "our", "you", "your", "he", "she", "it", "they", "them",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "and", "but", "or", "nor", "not", "no", "yes", "if", "then", "else",
    "so", "too", "very", "just", "also", "more", "most", "much", "many",
    "some", "any", "all", "each", "every", "few", "both", "such",
    "than", "other", "same", "own", "only", "about", "after", "again",
    "before", "between", "by", "for", "from", "in", "into", "of", "on",
    "out", "over", "to", "up", "with", "at", "as", "like", "through",
    "during", "until", "while", "because", "since", "how", "when", "where",
    "why", "here", "there", "now", "then", "well", "still", "already",
    "always", "never", "sometimes", "often", "usually", "really", "quite",
    "even", "back", "go", "going", "gone", "went", "come", "came",
    "get", "got", "make", "made", "take", "took", "give", "gave",
    "know", "knew", "think", "thought", "say", "said", "see", "saw",
    "want", "need", "try", "keep", "let", "help", "show", "tell",
    "call", "ask", "work", "play", "run", "move", "live", "long",
    "good", "great", "bad", "new", "old", "big", "small", "first",
    "last", "next", "right", "left", "best", "better", "high", "low",
    "man", "woman", "people", "time", "day", "year", "way", "world",
    "life", "hand", "part", "place", "case", "point", "group", "number",
    "game", "movie", "video", "song", "music", "love", "hate", "bro",
    "dude", "man", "lol", "lmao", "omg", "wtf", "ok", "okay",
    "please", "thank", "thanks", "sorry", "hello", "hi", "hey", "bye",
    "sir", "madam", "boss", "super", "nice", "cool", "hot", "fire",
    "best", "worst", "top", "noob", "pro", "edit", "post", "comment",
    "like", "share", "follow", "subscribe", "channel", "watch",
})

# Telugu Unicode block range
_TELUGU_RANGE = range(0x0C00, 0x0C80)


def _is_telugu_script(word: str) -> bool:
    """Check if a word is already in Telugu (Devanagari-adjacent) script."""
    return any(ord(ch) in _TELUGU_RANGE for ch in word)


def _is_likely_english(word: str) -> bool:
    """
    Heuristic check if a Latin-script word is likely English.
    Uses stopword set + basic patterns.
    """
    w = word.lower().strip()
    if len(w) <= 1:
        return True  # Single letters → keep as-is
    if w in _ENGLISH_STOPWORDS:
        return True
    # Words with common English suffixes
    if w.endswith(("ing", "tion", "ment", "ness", "able", "ible", "ous", "ive", "ful", "less", "ly")):
        return True
    return False


class Transliterator:
    """
    Selective phonetic transliterator for Romanized Telugu → Native Telugu script.
    
    Uses ai4bharat/indic-xlit when available, falls back to a no-op pass-through
    (logging a warning) if the package is not installed. This ensures the pipeline
    remains functional even without the transliteration model.
    """
    
    def __init__(self, src_lang: str = "en", tgt_lang: str = "te"):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._engine = None
        self._available = False
        self._init_engine()
    
    def _init_engine(self):
        """Attempt to load the XlitEngine; gracefully degrade if unavailable."""
        try:
            from ai4bharat.transliteration import XlitEngine
            self._engine = XlitEngine(
                src_script_type="roman",
                beam_width=4,
                rescore=True,
            )
            self._available = True
            logger.info("ai4bharat/indic-xlit engine loaded successfully.")
        except ImportError:
            logger.warning(
                "ai4bharat-transliteration not installed. "
                "Transliteration will be skipped (pass-through mode). "
                "Install with: pip install ai4bharat-transliteration"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize XlitEngine: {e}. Using pass-through mode.")
    
    def transliterate_word(self, word: str) -> str:
        """
        Transliterate a single Romanized Telugu word to native Telugu script.
        
        Returns the word unchanged if:
          - It's already in Telugu script
          - It's likely an English word
          - The transliteration engine is unavailable
        """
        if not word or not word.strip():
            return word
        
        # Already in Telugu script
        if _is_telugu_script(word):
            return word
        
        # Likely English → keep in Latin
        if _is_likely_english(word):
            return word
        
        # Non-alphabetic → keep as-is
        if not any(c.isalpha() for c in word):
            return word
        
        if not self._available:
            return word  # Pass-through
        
        try:
            result = self._engine.translit_word(word, lang_code=self.tgt_lang, topk=1)
            if result and self.tgt_lang in result and result[self.tgt_lang]:
                return result[self.tgt_lang][0]
        except Exception as e:
            logger.debug(f"Transliteration failed for '{word}': {e}")
        
        return word  # Fallback: return original
    
    def transliterate_sentence(self, text: str) -> str:
        """
        Apply word-level selective transliteration to a full sentence.
        
        English words are preserved in Latin script; Romanized Telugu words
        are converted to native Telugu Unicode.
        """
        if not text:
            return ""
        
        tokens = text.split()
        result_tokens = []
        
        for token in tokens:
            # Separate punctuation from word
            prefix = ""
            suffix = ""
            word = token
            
            # Strip leading punctuation
            while word and not word[0].isalnum() and ord(word[0]) < 0x0C00:
                prefix += word[0]
                word = word[1:]
            
            # Strip trailing punctuation
            while word and not word[-1].isalnum() and ord(word[-1]) < 0x0C00:
                suffix = word[-1] + suffix
                word = word[:-1]
            
            if word:
                word = self.transliterate_word(word)
            
            result_tokens.append(prefix + word + suffix)
        
        return " ".join(result_tokens)
    
    @property
    def is_available(self) -> bool:
        return self._available


# ═══════════════════════════════════════════════════════════════════════
# Stage 3: Unicode Canonicalization (indic-nlp-library)
# ═══════════════════════════════════════════════════════════════════════

class IndicNormalizer:
    """
    Telugu Unicode normalizer using indic-nlp-library.
    
    Operations:
      - Standardizes nasalization markers (anusvara vs. chandrabindu)
      - Strips stray Zero-Width Non-Joiners (ZWNJ) 
      - Unifies composite vowel signs
      - Applies NFC normalization
    """
    
    def __init__(self, lang: str = "te"):
        self.lang = lang
        self._normalizer = None
        self._available = False
        self._init_normalizer()
    
    def _init_normalizer(self):
        """Load the IndicNormalizerFactory; gracefully degrade if unavailable."""
        try:
            from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
            factory = IndicNormalizerFactory()
            self._normalizer = factory.get_normalizer(self.lang)
            self._available = True
            logger.info(f"indic-nlp normalizer loaded for language: {self.lang}")
        except ImportError:
            logger.warning(
                "indic-nlp-library not installed. "
                "Script normalization will use basic Unicode NFC only. "
                "Install with: pip install indic-nlp-library"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize IndicNormalizer: {e}. Using basic mode.")
    
    def normalize(self, text: str) -> str:
        """
        Apply Telugu-specific Unicode normalization.
        
        Falls back to basic Unicode NFC if indic-nlp-library is unavailable.
        """
        if not text:
            return ""
        
        # Remove Zero-Width Non-Joiners and Zero-Width Joiners
        text = text.replace("\u200C", "")  # ZWNJ
        text = text.replace("\u200D", "")  # ZWJ
        text = text.replace("\u200B", "")  # Zero-width space
        text = text.replace("\uFEFF", "")  # BOM
        
        if self._available:
            try:
                text = self._normalizer.normalize(text)
            except Exception as e:
                logger.debug(f"IndicNormalizer.normalize() failed: {e}")
        
        # Always apply NFC normalization
        text = unicodedata.normalize("NFC", text)
        
        return text
    
    @property
    def is_available(self) -> bool:
        return self._available


# ═══════════════════════════════════════════════════════════════════════
# Full Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════════

class IndicPreprocessor:
    """
    End-to-end preprocessing pipeline for Indic & Telugu text.
    
    Chains the text-level stages:
      1. De-obfuscation
      2. Selective transliteration (Roman Telugu → Native Telugu)
      3. Unicode normalization (indic-nlp-library)

    Context-window assembly (topic / history / target) lives in
    src/model.py:encode_example, which works at the token-id level.
    
    Usage:
        preprocessor = IndicPreprocessor()
        
        clean = preprocessor.clean_text("m@drch0d nuvvu chalaaaaa")
    """
    
    def __init__(
        self,
        src_lang: str = "en",
        tgt_lang: str = "te",
        enable_transliteration: bool = True,
        enable_normalization: bool = True,
    ):
        self.transliterator = Transliterator(src_lang, tgt_lang) if enable_transliteration else None
        self.normalizer = IndicNormalizer(tgt_lang) if enable_normalization else None
        
        logger.info(
            f"IndicPreprocessor initialized: "
            f"transliteration={'ON' if self.transliterator and self.transliterator.is_available else 'OFF'}, "
            f"normalization={'ON' if self.normalizer and self.normalizer.is_available else 'OFF'}"
        )
    
    def clean_text(self, text: str) -> str:
        """
        Apply stages 1-3 to a single text string.
        
        Pipeline: raw → deobfuscate → transliterate → normalize
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Stage 1: De-obfuscation
        text = deobfuscate(text)
        
        # Stage 2: Selective transliteration
        if self.transliterator:
            text = self.transliterator.transliterate_sentence(text)
        
        # Stage 3: Unicode normalization
        if self.normalizer:
            text = self.normalizer.normalize(text)
        
        return text.strip()
    
    def process_single(self, text: str) -> str:
        """
        Process a single (non-threaded) text through stages 1-3.
        Convenience wrapper for datasets without thread structure.
        """
        return self.clean_text(text)
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts through stages 1-3."""
        return [self.clean_text(t) for t in texts]


# ═══════════════════════════════════════════════════════════════════════
# Utility: Quick Smoke Test
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    """Run a quick smoke test of the preprocessing pipeline."""
    print("=" * 70)
    print("PREPROCESSING PIPELINE SMOKE TEST")
    print("=" * 70)
    
    # Stage 1 tests
    test_cases = [
        ("m@drch0d nuvvu chalaaaaa", "Stage 1: Leet + repeat collapse"),
        ("https://evil.com @troll_bot check this l@nja", "Stage 1: URL/handle + leet"),
        ("#CoolMovie eppudu release??", "Stage 1: Hashtag clean"),
        ("vaadu ch3kkesadu $hit", "Stage 1: Mixed obfuscation"),
    ]
    
    print("\n--- Stage 1: De-Obfuscation ---")
    for text, desc in test_cases:
        result = deobfuscate(text)
        print(f"  [{desc}]")
        print(f"    IN:  {text}")
        print(f"    OUT: {result}")
    
    # Full pipeline test (stages 1-3)
    print("\n--- Stages 1-3: Full Text Cleaning ---")
    preprocessor = IndicPreprocessor()
    
    full_tests = [
        "m@drch0d nuvvu chalaaaaa baaga",
        "emra idi chala worst movie",
        "Boss level clear chesa! Champesav po!",
    ]
    
    for text in full_tests:
        result = preprocessor.clean_text(text)
        print(f"    IN:  {text}")
        print(f"    OUT: {result}")
        print()
    
    print("=" * 70)
    print("SMOKE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    smoke_test()
