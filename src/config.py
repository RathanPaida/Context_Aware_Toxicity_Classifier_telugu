"""
Central Configuration for Context-Aware Indic Toxicity Classifier.

All paths, hyperparameters, model identifiers, dataset references,
decision thresholds and preprocessing constants live here.

Paths are derived from the repository location (no machine-specific
absolute paths). Any of them can be overridden with environment variables:

    INDIC_SHIELD_ROOT        project root          (default: repo root)
    INDIC_SHIELD_DATA_DIR    data directory        (default: <root>/data)
    INDIC_SHIELD_MODEL_DIR   model directory       (default: <root>/models)
    INDIC_SHIELD_LOG_DIR     log directory         (default: <root>/logs)
    INDIC_SHIELD_MODEL       backbone HF model id  (default: PRIMARY_MODEL below)
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


# ─────────────────────────── Paths ───────────────────────────
PROJECT_ROOT = _env_path("INDIC_SHIELD_ROOT", Path(__file__).resolve().parent.parent)
DATA_DIR = _env_path("INDIC_SHIELD_DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"            # put manually-downloaded files here (Jigsaw, local CSVs)
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = _env_path("INDIC_SHIELD_MODEL_DIR", PROJECT_ROOT / "models")
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
LOG_DIR = _env_path("INDIC_SHIELD_LOG_DIR", PROJECT_ROOT / "logs")
REPORTS_DIR = PROJECT_ROOT / "reports"   # evaluation results consumed by the web app
RESOURCES_DIR = Path(__file__).resolve().parent / "resources"  # lexicons, demo scenarios

EVAL_RESULTS_PATH = REPORTS_DIR / "eval_results.json"
LEXICON_PATH = RESOURCES_DIR / "lexicons.json"
SCENARIOS_PATH = RESOURCES_DIR / "scenarios.json"


def indic_supervised_sources() -> list:
    """Processed Stage 3 datasets: built-ins + anything loaded from data/raw/custom/."""
    built_in = [PROCESSED_DIR / "dravidian_codemix.parquet", PROCESSED_DIR / "cultural_safety_telugu.parquet"]
    return [p for p in built_in if p.exists()] + sorted(PROCESSED_DIR.glob("custom_*.parquet"))


def ensure_dirs() -> None:
    """Create the writable project directories (safe to call repeatedly)."""
    for d in [RAW_DIR, PROCESSED_DIR, CACHE_DIR, MODEL_DIR, CHECKPOINT_DIR, LOG_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()


# ─────────────────────── Model Identifiers ───────────────────────
PRIMARY_MODEL = os.environ.get("INDIC_SHIELD_MODEL", "ai4bharat/IndicBERTv2-MLM-Sam-TLM")
FALLBACK_MODEL = "google/muril-base-cased"
TRANSLITERATION_MODEL = "ai4bharat/indic-xlit"  # en→te engine

# Checkpoint names, in the order the curriculum produces them.
# Inference / evaluation load the most advanced one that exists.
CHECKPOINT_PRIORITY = ["stage4_best", "stage3_best", "stage1_best"]


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
    # Stage 1: Toxicity mechanics (English)
    "toxic_chat": DatasetConfig(
        name="toxic_chat",
        hf_id="lmsys/toxic-chat",
        subset="toxicchat0124",
        pipeline_role="Stage 1: Toxicity Mechanics",
        language="en",
    ),
    # Stage 1: Identity Debiasing (English) — manual Kaggle download into data/raw/
    "jigsaw_bias": DatasetConfig(
        name="jigsaw_bias",
        hf_id=None,
        local_path=RAW_DIR / "jigsaw-unintended-bias-in-toxicity-classification",
        pipeline_role="Stage 1: Identity Debiasing",
        language="en",
    ),
    # Stage 2: Telugu MLM Domain Adaptation
    "indic_corp_v2": DatasetConfig(
        name="indic_corp_v2",
        hf_id="ai4bharat/IndicCorpV2",
        subset="indiccorp_v2",
        split="tel_Telu",
        pipeline_role="Stage 2: Domain MLM",
        language="te",
    ),
    # Stage 3A: Code-mix offensive language.
    # NOTE: this HF dataset is the *Tamil* (not Telugu) DravidianLangTech offensive
    # language corpus. It is still useful for Dravidian code-mix transfer, but a
    # Telugu code-mixed dataset should be added (see docs/PROJECT_PLAN.md).
    "dravidian_codemix": DatasetConfig(
        name="dravidian_codemix",
        hf_id="IsaacRodgz/DravidianCodeMix-Dataset",
        pipeline_role="Stage 3A: Code-Mix Offensive (Tamil)",
        language="ta-en",
    ),
    # Stage 3B: Telugu Safety (LLM-prompt safety data, translated)
    "cultural_safety_telugu": DatasetConfig(
        name="cultural_safety_telugu",
        hf_id="salmankhanpm/Cultural-Safety-Telugu",
        pipeline_role="Stage 3B: Telugu Safety",
        language="te",
    ),
}


# ─────────────────── Preprocessing Constants ──────────────────────
# Leetspeak / obfuscation mapping (only applied to symbols embedded between letters)
LEET_MAP = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
    "7": "t", "@": "a", "$": "s", "!": "i", "+": "t",
}

# Sequence settings
MAX_SEQ_LENGTH = 256          # 512 is rarely needed for chat comments and 4x slower
TARGET_MAX_TOKENS = 128       # target is never truncated below this; longer targets are cut
HISTORY_TURNS = 3             # number of preceding thread turns fed as context

# Segment prefixes (kept for readability of the context segment)
TOPIC_PREFIX = "Topic:"
CONTEXT_PREFIX = "Context_"
PARENT_PREFIX = "Parent:"
TARGET_PREFIX = "Target:"


# ──────────────────── Label Mappings ─────────────────────────────
# Unified multi-label toxicity schema
TOXICITY_LABELS = ["toxic", "severe_toxic", "identity_attack", "insult", "threat"]
LABEL2ID = {label: idx for idx, label in enumerate(TOXICITY_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(TOXICITY_LABELS)}

# Value used in label columns when a dataset does not annotate that label.
# Such entries are masked out of the loss and the metrics.
UNKNOWN_LABEL = -1

# DravidianCodeMix (IsaacRodgz) integer labels. Verified against the HF rows:
#   0 = Not offensive, 1 = not-Tamil (other language) -> dropped,
#   2..5 = offensive sub-types (untargeted / targeted individual / group / other).
# Only the `toxic` bit is reliable; sub-types are left UNKNOWN.
DRAVIDIAN_SAFE_LABELS = {0}
DRAVIDIAN_DROP_LABELS = {1}
DRAVIDIAN_OFFENSIVE_LABELS = {2, 3, 4, 5}

# Cultural-Safety-Telugu: only categories that correspond to *toxic language*
# are used. Rows whose only categories are things like fraud, IP or medical
# advice are dropped — they are unsafe LLM requests, not toxic comments.
# Vector order: [toxic, severe_toxic, identity_attack, insult, threat]
CULTURAL_SAFETY_MAP = {
    "Harassment & Profanity": [1, 0, 0, 1, 0],
    "Hate Speech & Identity Attacks": [1, 1, 1, 0, 0],
    "Violence & Physical Harm": [1, 1, 0, 0, 1],
    "Sexually Explicit Content": [1, 0, 0, 0, 0],
    # Legacy/blueprint names (backward compatibility)
    "Harassment": [1, 0, 0, 1, 0],
    "Profanity": [1, 0, 0, 1, 0],
    "Hate Speech": [1, 1, 1, 0, 0],
    "Identity Attacks": [1, 0, 1, 0, 0],
    "Violence": [1, 1, 0, 0, 1],
}


# ──────────────────── Shared training defaults ───────────────────
@dataclass
class TrainConfig:
    """Hyperparameters shared by the supervised stages (1, 3, 4)."""
    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-4   # randomly-initialised heads learn faster than the encoder
    num_epochs: int = 3
    batch_size: int = 16
    max_seq_length: int = MAX_SEQ_LENGTH
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    fp16: bool = True                  # only used when CUDA is available
    freeze_ratio: float = 0.0          # fraction of encoder layers to freeze
    freeze_embeddings: bool = True     # needed to fit IndicBERTv2 (250k vocab) on a 6 GB GPU
    gradient_checkpointing: bool = False  # set True if you still run out of VRAM (≈30% slower)
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    eval_steps: int = 0                # 0 → evaluate once per epoch
    logging_steps: int = 50
    max_samples: Optional[int] = None
    early_stopping_patience: int = 2
    seed: int = 42


@dataclass
class Stage1Config(TrainConfig):
    """Stage 1: English toxicity mechanics (toxic-chat + Jigsaw)."""
    learning_rate: float = 2e-5
    num_epochs: int = 2
    batch_size: int = 16


@dataclass
class Stage2Config:
    """Stage 2: Telugu MLM Adaptation."""
    learning_rate: float = 5e-5
    max_steps: int = 10_000
    batch_size: int = 16
    max_seq_length: int = 128
    mlm_probability: float = 0.15
    warmup_steps: int = 500
    weight_decay: float = 0.01
    fp16: bool = True
    telugu_sample_size: int = 250_000
    logging_steps: int = 100
    save_steps: int = 2000
    eval_steps: int = 1000
    num_workers: int = 0               # >0 is unreliable on Windows


@dataclass
class Stage3Config(TrainConfig):
    """Stage 3: Code-Mix & Telugu Slang."""
    learning_rate: float = 1e-5
    num_epochs: int = 4
    batch_size: int = 16
    freeze_ratio: float = 0.5


@dataclass
class Stage4Config(TrainConfig):
    """Stage 4: Synthetic Thread Fusion."""
    learning_rate: float = 1e-5
    num_epochs: int = 5
    batch_size: int = 8
    freeze_ratio: float = 0.75
    topic_dropout: float = 0.5         # drop the topic string so the model can't shortcut on it


# Settings used by `run_pipeline.py --quick-run` (smoke test, not real training)
QUICK_RUN = {
    "stage1_max_samples": 2000,
    "stage1_epochs": 1,
    "stage2_max_steps": 100,
    "stage2_sample_size": 2000,
    "stage3_max_samples": 1500,
    "stage3_epochs": 1,
    "stage4_epochs": 2,
}


# ──────────────────── Inference / decision settings ──────────────
@dataclass
class InferenceConfig:
    """
    Decision settings for the moderation pipeline.

    `toxic_threshold` is only a fallback: training writes per-label thresholds
    tuned on the validation set into the checkpoint's metadata.json, and the
    pipeline uses those when present.
    """
    toxic_threshold: float = 0.5
    # Severe sub-labels (threat / severe_toxic / identity_attack) can flag on their own
    # when they are confidently high, even if the generic `toxic` head is lower.
    subtype_flag_threshold: float = 0.7
    history_turns: int = HISTORY_TURNS
    max_history_stored: int = 10
    max_seq_length: int = MAX_SEQ_LENGTH
    # Lexicon gate
    lexicon_fast_exit: bool = True     # skip the model for unambiguous severe slurs w/o context
    lexicon_flag_score: float = 0.97   # reported score for a lexicon fast-exit
    # Thread cache
    cache_ttl_seconds: int = 1800
    redis_host: str = os.environ.get("REDIS_HOST", "localhost")
    redis_port: int = int(os.environ.get("REDIS_PORT", "6379"))
    use_redis: bool = os.environ.get("USE_REDIS", "0") == "1"
    # Refuse to serve predictions from an untrained (random-head) model
    require_checkpoint: bool = os.environ.get("ALLOW_UNTRAINED", "0") != "1"


# ──────────────────── Evaluation release gates ───────────────────
RELEASE_GATES = {
    "contrastive_flip": {"op": ">=", "value": 82.0, "unit": "%"},
    "identity_bias":    {"op": "<=", "value": 3.5,  "unit": "%"},
    "parent_bleed":     {"op": ">=", "value": 92.0, "unit": "%"},
    "script_parity":    {"op": "<=", "value": 2.0,  "unit": "%"},
    "latency_p50":      {"op": "<=", "value": 25.0, "unit": "ms"},
}


# ──────────────────── Logging ────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Create a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.propagate = False
    return logger
