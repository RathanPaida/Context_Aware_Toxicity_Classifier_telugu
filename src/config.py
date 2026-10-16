"""
Central Configuration for Context-Aware Indic Toxicity Classifier.

All paths, hyperparameters, model identifiers, dataset references,
and preprocessing constants live here.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────── Paths ───────────────────────────
PROJECT_ROOT = Path(r"d:\NLP_projects\Context_Aware_Toxicity_classifier")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

# Create all directories
for d in [RAW_DIR, PROCESSED_DIR, CACHE_DIR, MODEL_DIR, CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────── Model Identifiers ───────────────────────
PRIMARY_MODEL = "ai4bharat/IndicBERTv2-MLM-Sam-TLM"  # IndicBERT-v3 family (270M equivalent)
FALLBACK_MODEL = "google/muril-base-cased"
TRANSLITERATION_MODEL = "ai4bharat/indic-xlit"  # en→te engine
FASTTEXT_LID_MODEL = str(DATA_DIR / "lid.176.bin")  # FastText language-ID


# ─────────────────────── Dataset Identifiers ─────────────────────
@dataclass
class DatasetConfig:
    """Configuration for a single dataset source."""
    name: str
    hf_id: Optional[str]  # Hugging Face dataset ID (None if local/Kaggle)
    subset: Optional[str] = None
    split: Optional[str] = None
    local_path: Optional[Path] = None  # Path if already downloaded
    pipeline_role: str = ""
    language: str = "en"


DATASETS = {
    # Stage 1: Context Mechanics (English)
    "toxic_chat": DatasetConfig(
        name="toxic_chat",
        hf_id="lmsys/toxic-chat",
        subset="toxicchat0124",
        pipeline_role="Stage 1: Context Mechanics",
        language="en",
    ),
    # Stage 1: Identity Debiasing (English)
    "jigsaw_bias": DatasetConfig(
        name="jigsaw_bias",
        hf_id=None,  # Already downloaded from Kaggle
        local_path=PROJECT_ROOT / "jigsaw-unintended-bias-in-toxicity-classification",
        pipeline_role="Stage 1: Identity Debiasing",
        language="en",
    ),
    # Stage 2: Telugu MLM Domain Adaptation
    "indic_corp_v2": DatasetConfig(
        name="indic_corp_v2",
        hf_id="ai4bharat/IndicCorp-Monolingual",
        subset="te",
        pipeline_role="Stage 2: Domain MLM",
        language="te",
    ),
    # Stage 3A: Code-Mix Slang
    "dravidian_codemix": DatasetConfig(
        name="dravidian_codemix",
        hf_id="IsaacRodgz/DravidianCodeMix-Dataset",
        pipeline_role="Stage 3A: Code-Mix Slang",
        language="dravidian-mix",
    ),
    # Stage 3B: Telugu Safety
    "cultural_safety_telugu": DatasetConfig(
        name="cultural_safety_telugu",
        hf_id="salmankhanpm/Cultural-Safety-Telugu",
        pipeline_role="Stage 3B: Telugu Safety",
        language="te",
    ),
}


# ─────────────────── Preprocessing Constants ──────────────────────
# Leetspeak / obfuscation mapping
LEET_MAP = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
    "7": "t", "@": "a", "$": "s", "!": "i", "+": "t",
}

# Maximum repetition before collapse (>2 repeating chars → single)
MAX_CHAR_REPEAT = 2

# Context window settings
MAX_SEQ_LENGTH = 512
CONTEXT_BUDGET_TOKENS = MAX_SEQ_LENGTH  # Total token budget
TARGET_MIN_TOKENS = 64  # Minimum tokens reserved for target text
PARENT_MIN_TOKENS = 64  # Never truncate parent below this

# Special tokens for context assembly
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
TOPIC_PREFIX = "Topic:"
CONTEXT_PREFIX = "Context_"
PARENT_PREFIX = "Parent:"
TARGET_PREFIX = "Target:"


# ────────────────── Stage 1 Hyperparameters ──────────────────────
@dataclass
class Stage1Config:
    """Stage 1: Context Mechanics Training (English)."""
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 32
    max_seq_length: int = 512
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    label_columns: List[str] = field(default_factory=lambda: [
        "toxic", "severe_toxic", "identity_attack", "insult", "threat"
    ])
    num_labels: int = 5
    # Focal Loss parameters
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    max_samples: Optional[int] = None


# ────────────────── Stage 2 Hyperparameters ──────────────────────
@dataclass
class Stage2Config:
    """Stage 2: Telugu MLM Adaptation."""
    learning_rate: float = 5e-5
    max_steps: int = 10_000
    batch_size: int = 16
    max_seq_length: int = 512
    mlm_probability: float = 0.15
    warmup_steps: int = 500
    weight_decay: float = 0.01
    fp16: bool = True
    # Data sampling
    telugu_sample_size: int = 250_000  # Sample 250K Telugu sentences
    logging_steps: int = 100
    save_steps: int = 2000
    eval_steps: int = 1000


# ────────────────── Stage 3 Hyperparameters ──────────────────────
@dataclass
class Stage3Config:
    """Stage 3: Code-Mix & Telugu Slang."""
    learning_rate: float = 1e-5
    num_epochs: int = 4
    batch_size: int = 16
    max_seq_length: int = 512
    freeze_ratio: float = 0.5  # Freeze bottom 50% layers
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 250


# ────────────────── Stage 4 Hyperparameters ──────────────────────
@dataclass
class Stage4Config:
    """Stage 4: Synthetic Thread Fusion."""
    learning_rate: float = 5e-6
    num_epochs: int = 3
    batch_size: int = 8
    max_seq_length: int = 512
    freeze_ratio: float = 0.75  # Freeze bottom 75% layers + embedding
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    logging_steps: int = 20
    save_steps: int = 100
    eval_steps: int = 50


# ──────────────────── Label Mappings ─────────────────────────────
# Unified multi-label toxicity schema
TOXICITY_LABELS = ["toxic", "severe_toxic", "identity_attack", "insult", "threat"]
LABEL2ID = {label: idx for idx, label in enumerate(TOXICITY_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(TOXICITY_LABELS)}

# DravidianCodeMix label mapping → unified schema
DRAVIDIAN_LABEL_MAP = {
    0: [0, 0, 0, 0, 0],  # Not offensive
    1: [1, 0, 0, 0, 0],  # Offensive (maps to toxic)
}

# Cultural-Safety-Telugu category mapping → unified schema
# Mapped from actual dataset categories to [toxic, severe_toxic, identity_attack, insult, threat]
CULTURAL_SAFETY_MAP = {
    # Primary categories (exact match to dataset)
    "Harassment & Profanity": [1, 0, 0, 1, 0],            # toxic + insult
    "Hate Speech & Identity Attacks": [1, 1, 1, 0, 0],    # toxic + severe + identity_attack
    "Violence & Physical Harm": [1, 1, 0, 0, 1],          # toxic + severe + threat
    "Criminal Activity Planning": [1, 1, 0, 0, 1],        # toxic + severe + threat
    "Sexually Explicit Content": [1, 1, 0, 1, 0],         # toxic + severe + insult
    "Weapons & Dangerous Substances": [1, 1, 0, 0, 1],    # toxic + severe + threat
    "Self-Harm & Suicide": [1, 1, 0, 0, 1],               # toxic + severe + threat
    "Child Safety & Exploitation": [1, 1, 0, 0, 1],       # toxic + severe + threat
    "Privacy & PII Violations": [1, 0, 0, 0, 0],          # toxic
    "Deception, Fraud & Manipulation": [1, 0, 0, 0, 0],   # toxic
    "Misinformation & Political Influence": [1, 0, 0, 0, 0],  # toxic
    "Cyber Threats & Malware": [1, 0, 0, 0, 1],           # toxic + threat
    "Sensitive / Needs Caution": [1, 0, 0, 0, 0],         # toxic
    "Immoral / Unethical Behavior": [1, 0, 0, 1, 0],      # toxic + insult
    "Unauthorized Professional Advice": [1, 0, 0, 0, 0],  # toxic (mild)
    "Intellectual Property Violations": [1, 0, 0, 0, 0],  # toxic (mild)
    "High-Risk Governance Decisions": [1, 0, 0, 0, 0],    # toxic (mild)
    # Legacy/blueprint names (backward compatibility)
    "Harassment": [1, 0, 0, 1, 0],
    "Profanity": [1, 0, 0, 1, 0],
    "Hate Speech": [1, 1, 1, 0, 0],
    "Identity Attacks": [1, 0, 1, 0, 0],
    "Violence": [1, 1, 0, 0, 1],
    "Safe": [0, 0, 0, 0, 0],
    "SAFE": [0, 0, 0, 0, 0],
}


# ──────────────────── Logging ────────────────────────────────────
import logging

def get_logger(name: str) -> logging.Logger:
    """Create a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        # File handler
        fh = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
