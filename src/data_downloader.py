"""
Phase 1: Open Dataset Ingestion & Formatting.

Downloads (or reads from data/raw/) all datasets and converts them to one schema:

    text | context | topic | conv_id | split | source | toxic | severe_toxic | identity_attack | insult | threat

Label value -1 (UNKNOWN_LABEL) means "this dataset does not annotate this label";
such entries are masked out of the loss and of the metrics.

Datasets handled:
  1. lmsys/toxic-chat          → Stage 1 (English)
  2. Jigsaw Unintended Bias    → Stage 1 (English, manual Kaggle download)
  3. ai4bharat/IndicCorp       → Stage 2 (Telugu MLM)
  4. DravidianCodeMix          → Stage 3A (Tamil code-mix offensive)
  5. Cultural-Safety-Telugu    → Stage 3B (Telugu)
"""

import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    CULTURAL_SAFETY_MAP,
    DATASETS,
    DRAVIDIAN_DROP_LABELS,
    DRAVIDIAN_OFFENSIVE_LABELS,
    DRAVIDIAN_SAFE_LABELS,
    PROCESSED_DIR,
    RAW_DIR,
    TOXICITY_LABELS,
    UNKNOWN_LABEL,
    get_logger,
)

logger = get_logger("data_downloader")

SCHEMA_COLUMNS = ["text", "context", "topic", "conv_id", "split", "source"] + TOXICITY_LABELS


def _finalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Fill schema columns, drop empty / duplicate texts."""
    for col in ["context", "topic", "conv_id", "split"]:
        if col not in df.columns:
            df[col] = ""
    for col in TOXICITY_LABELS:
        if col not in df.columns:
            df[col] = UNKNOWN_LABEL
        df[col] = df[col].astype(int)
    df["source"] = source
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset=["text", "context"]).reset_index(drop=True)
    if before != len(df):
        logger.info(f"  Dropped {before - len(df):,} duplicate rows")
    return df[SCHEMA_COLUMNS]


def _hash_split(texts: pd.Series, val: float = 0.1, test: float = 0.1) -> pd.Series:
    """Deterministic train/val/test assignment by text hash (stable across reruns)."""
    def bucket(t: str) -> str:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        return "test" if h < test else ("val" if h < test + val else "train")
    return texts.map(bucket)


def _find_local(*names: str) -> List[Path]:
    return [RAW_DIR / n for n in names if (RAW_DIR / n).exists()]


# ═══════════════════════════════════════════════════════════════════════
# 1. lmsys/toxic-chat
# ═══════════════════════════════════════════════════════════════════════

def download_toxic_chat() -> pd.DataFrame:
    """
    toxic-chat annotates the *user prompt* only. `model_output` is the assistant's
    reply, which comes *after* the prompt, so it is NOT used as conversational
    context (the old code fed it in as the "parent", which is backwards).
    Only `toxic` is annotated; other labels are UNKNOWN.
    """
    logger.info("Processing lmsys/toxic-chat...")
    local = _find_local("toxic-chat_annotation_train.csv", "toxic-chat_annotation_test.csv")
    if local:
        frames = []
        for f in local:
            d = pd.read_csv(f, encoding="utf-8")
            d["split"] = "test" if "test" in f.name else "train"
            frames.append(d)
    else:
        from datasets import load_dataset
        cfg = DATASETS["toxic_chat"]
        ds = load_dataset(cfg.hf_id, cfg.subset)
        frames = []
        for split in ds:
            d = ds[split].to_pandas()
            d["split"] = split
            frames.append(d)
    raw = pd.concat(frames, ignore_index=True)

    df = pd.DataFrame({
        "text": raw["user_input"],
        "conv_id": raw["conv_id"].fillna("").astype(str),
        "split": raw["split"],
        "toxic": (raw["toxicity"].fillna(0) >= 1).astype(int),
    })
    df = _finalize(df, "toxic_chat")
    logger.info(f"  toxic-chat: {len(df):,} rows | toxic={df['toxic'].mean():.1%}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. Jigsaw Unintended Bias
# ═══════════════════════════════════════════════════════════════════════

def download_jigsaw_bias() -> pd.DataFrame:
    """Jigsaw (manual Kaggle download to data/raw/jigsaw-unintended-bias-in-toxicity-classification/)."""
    logger.info("Processing Jigsaw Unintended Bias...")
    train_path = DATASETS["jigsaw_bias"].local_path / "train.csv"
    cols = ["comment_text", "target", "severe_toxicity", "identity_attack", "insult", "threat"]
    if train_path.exists():
        raw = pd.read_csv(train_path, usecols=cols)
    else:
        # Jigsaw Unintended Bias is built on Civil Comments; the HF copy has the same rows and
        # score columns and needs no Kaggle account.
        logger.info(f"  {train_path} not found — using the Hugging Face copy (google/civil_comments)")
        from datasets import load_dataset
        ds = load_dataset("google/civil_comments", split="train")
        ds = ds.select_columns(["text", "toxicity", "severe_toxicity", "identity_attack", "insult", "threat"])
        raw = ds.to_pandas().rename(columns={"text": "comment_text", "toxicity": "target"})
    df = pd.DataFrame({
        "text": raw["comment_text"],
        "toxic": (raw["target"].fillna(0) >= 0.5).astype(int),
        # Jigsaw sub-scores are fractions of raters; >=0.5 is very rare for severe_toxicity,
        # so use a lower bar for the sub-labels (still requires the comment to be toxic overall).
        "severe_toxic": (raw["severe_toxicity"].fillna(0) >= 0.2).astype(int),
        "identity_attack": (raw["identity_attack"].fillna(0) >= 0.5).astype(int),
        "insult": (raw["insult"].fillna(0) >= 0.5).astype(int),
        "threat": (raw["threat"].fillna(0) >= 0.5).astype(int),
    })
    df = _finalize(df, "jigsaw_bias")
    for label in TOXICITY_LABELS:
        logger.info(f"    {label}: {df[label].mean():.2%} positive")
    return df


def sample_jigsaw(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Down-sample keeping all rare positives (threat / identity_attack) and ≥25% toxic."""
    if len(df) <= n:
        return df
    rare = df[(df["threat"] == 1) | (df["identity_attack"] == 1) | (df["severe_toxic"] == 1)]
    rest = df.drop(rare.index)
    # Keep rare positives (threat etc. are <1%), but cap them so the sample isn't mostly toxic.
    rare = rare.sample(min(len(rare), n // 4), random_state=seed)
    toxic, safe = rest[rest["toxic"] == 1], rest[rest["toxic"] == 0]
    remaining = max(0, n - len(rare))
    n_toxic = min(len(toxic), max(int(remaining * toxic.shape[0] / max(1, len(rest))), remaining // 4))
    n_safe = min(len(safe), remaining - n_toxic)
    out = pd.concat([rare, toxic.sample(n_toxic, random_state=seed), safe.sample(n_safe, random_state=seed)])
    logger.info(f"  Sampled Jigsaw to {len(out):,} rows (kept {len(rare):,} rare-label positives)")
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
# 3. IndicCorp (Telugu) — Stage 2 MLM
# ═══════════════════════════════════════════════════════════════════════

def download_indic_corp_telugu(sample_size: int = 250_000) -> pd.DataFrame:
    logger.info(f"Processing IndicCorp Telugu (sampling {sample_size:,} sentences)...")
    output_path = PROCESSED_DIR / "indic_corp_telugu_mlm.parquet"
    if output_path.exists():
        df = pd.read_parquet(output_path)
        if len(df) >= sample_size:
            logger.info(f"  Already processed ({len(df):,} rows): {output_path}")
            return df

    from datasets import load_dataset
    texts: List[str] = []
    # (hf_id, config, split, text fields). "ai4bharat/IndicCorp-Monolingual" (old code) does not exist on the Hub.
    cfg = DATASETS["indic_corp_v2"]
    sources = [(cfg.hf_id, cfg.subset, cfg.split, ("text",)),
               ("ai4bharat/samanantar", "te", "train", ("tgt",))]
    for hf_id, subset, split, fields in sources:
        try:
            ds = load_dataset(hf_id, subset, split=split, streaming=True)
            for example in ds:
                t = next((example[f] for f in fields if example.get(f)), "")
                if t and len(t.strip()) > 10:
                    texts.append(t.strip())
                if len(texts) >= sample_size:
                    break
            if texts:
                logger.info(f"  Collected {len(texts):,} sentences from {hf_id}")
                break
        except Exception as e:
            logger.warning(f"  {hf_id} failed: {e}")

    df = pd.DataFrame({"text": texts, "source": "telugu_corpus"})
    if len(df):
        df.to_parquet(output_path, index=False)
    else:
        logger.error("  No Telugu corpus could be downloaded; Stage 2 will be skipped.")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 4. DravidianCodeMix (Tamil offensive) — Stage 3A
# ═══════════════════════════════════════════════════════════════════════

def download_dravidian_codemix() -> pd.DataFrame:
    """
    IsaacRodgz/DravidianCodeMix-Dataset: Tamil code-mixed YouTube comments.
    Labels (verified from the rows): 0 = not offensive, 1 = not-Tamil (other
    language — the old code labelled these TOXIC), 2-5 = offensive sub-types.
    Official train/dev/test splits are preserved (the old code merged and reshuffled them).
    """
    logger.info("Processing DravidianCodeMix (Tamil)...")
    local = {s: RAW_DIR / f"{s}-00000-of-00001.parquet" for s in ["train", "dev", "test"]}
    frames = []
    if all(p.exists() for p in local.values()):
        for split, p in local.items():
            d = pd.read_parquet(p)
            d["split"] = split
            frames.append(d)
    else:
        from datasets import load_dataset
        ds = load_dataset(DATASETS["dravidian_codemix"].hf_id)
        for split in ds:
            d = ds[split].to_pandas()
            d["split"] = split
            frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    logger.info(f"  Raw label distribution: {raw['label'].value_counts().sort_index().to_dict()}")

    raw = raw[~raw["label"].isin(DRAVIDIAN_DROP_LABELS)]
    known = raw["label"].isin(DRAVIDIAN_SAFE_LABELS | DRAVIDIAN_OFFENSIVE_LABELS)
    raw = raw[known]
    df = pd.DataFrame({
        "text": raw["text"],
        "split": raw["split"].replace({"dev": "val"}),
        "toxic": raw["label"].isin(DRAVIDIAN_OFFENSIVE_LABELS).astype(int),
    })
    df = _finalize(df, "dravidian_codemix")
    logger.info(f"  DravidianCodeMix: {len(df):,} rows | toxic={df['toxic'].mean():.1%}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 5. Cultural-Safety-Telugu — Stage 3B
# ═══════════════════════════════════════════════════════════════════════

def _parse_categories(value) -> List[str]:
    """Categories arrive as a list, a numpy array, JSON, or a Python-repr string."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    s = str(value).strip()
    if not s or s.lower() in {"none", "nan", "[]"}:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(s)
            return [str(v) for v in parsed] if isinstance(parsed, (list, tuple)) else [str(parsed)]
        except Exception:
            continue
    return [s]


def download_cultural_safety_telugu() -> pd.DataFrame:
    """
    salmankhanpm/Cultural-Safety-Telugu is an LLM-*prompt* safety set translated to Telugu
    (e.g. "how do I get poison"), not a toxic-comment corpus. We keep:
      * SAFE rows → all labels 0
      * UNSAFE rows with at least one toxicity-related category (harassment, hate,
        violence, sexual) → mapped labels; unmapped sub-labels stay 0 because the
        category list is exhaustive for that row
    and drop UNSAFE rows that are only fraud / IP / advice / etc.
    """
    logger.info("Processing Cultural-Safety-Telugu...")
    local = RAW_DIR / "train_0.jsonl"
    if local.exists():
        with open(local, encoding="utf-8") as f:
            raw = pd.DataFrame([json.loads(l) for l in f if l.strip()])
    else:
        from datasets import load_dataset
        raw = load_dataset(DATASETS["cultural_safety_telugu"].hf_id, split="train").to_pandas()

    text_col = "prompt_telugu" if "prompt_telugu" in raw.columns else "prompt"
    label_col = "safety_label" if "safety_label" in raw.columns else "prompt_label"

    rows, dropped = [], 0
    for text, label, cats in zip(raw[text_col], raw[label_col], raw.get("safety_categories", [None] * len(raw))):
        label = str(label).strip().upper()
        if label == "SAFE":
            rows.append({"text": text, **{l: 0 for l in TOXICITY_LABELS}})
            continue
        if label != "UNSAFE":
            dropped += 1
            continue
        vec = np.zeros(len(TOXICITY_LABELS), dtype=int)
        mapped = False
        for c in _parse_categories(cats):
            if c.strip() in CULTURAL_SAFETY_MAP:
                vec |= np.array(CULTURAL_SAFETY_MAP[c.strip()])
                mapped = True
        if mapped:
            rows.append({"text": text, **dict(zip(TOXICITY_LABELS, vec.tolist()))})
        else:
            dropped += 1

    df = pd.DataFrame(rows)
    df["split"] = _hash_split(df["text"].astype(str))
    df = _finalize(df, "cultural_safety_telugu")
    logger.info(f"  Cultural-Safety-Telugu: kept {len(df):,}, dropped {dropped:,} non-toxicity UNSAFE rows")
    for label in TOXICITY_LABELS:
        logger.info(f"    {label}: {df[label].mean():.2%} positive")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 6. Your own data — data/raw/custom/*.csv | *.jsonl | *.parquet
# ═══════════════════════════════════════════════════════════════════════

CUSTOM_DIR = RAW_DIR / "custom"
_SAFE_LABEL_WORDS = ("not", "non", "normal", "none", "safe", "neutral", "clean", "no_hate")


def _label_to_int(value) -> int:
    """0/1, True/False, or a class name. Names containing not/non/normal/safe/... count as non-toxic."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return UNKNOWN_LABEL
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    try:
        return int(float(value) >= 0.5)
    except (TypeError, ValueError):
        s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        return 0 if any(w in s for w in _SAFE_LABEL_WORDS) else 1


def load_custom_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load every file in data/raw/custom/ into the unified schema. Expected columns:
        text                         (required)
        label  OR any of toxic/severe_toxic/identity_attack/insult/threat   (0/1; missing = unknown)
        context                      (optional; earlier turns, oldest first, separated by newlines)
        topic, split                 (optional; split = train/val/test, otherwise assigned by hash)
    Rows with a context are thread data and also teach the model to use context.
    """
    out: Dict[str, pd.DataFrame] = {}
    if not CUSTOM_DIR.exists():
        return out
    readers = {".csv": pd.read_csv, ".parquet": pd.read_parquet,
               ".jsonl": lambda p: pd.read_json(p, lines=True), ".json": pd.read_json}
    for path in sorted(CUSTOM_DIR.iterdir()):
        reader = readers.get(path.suffix.lower())
        if reader is None:
            continue
        raw = reader(path)
        if "text" not in raw.columns:
            logger.error(f"  {path.name}: no 'text' column (columns: {list(raw.columns)}) — skipped")
            continue
        df = pd.DataFrame({"text": raw["text"]})
        for col in ["context", "topic", "split"]:
            if col in raw.columns:
                df[col] = raw[col].fillna("").astype(str)
        label_cols = [c for c in TOXICITY_LABELS if c in raw.columns]
        if label_cols:
            for c in label_cols:
                df[c] = raw[c].map(_label_to_int)
        elif "label" in raw.columns:
            mapping = {v: _label_to_int(v) for v in raw["label"].dropna().unique()}
            logger.info(f"  {path.name}: label mapping (check this!) {mapping}")
            df["toxic"] = raw["label"].map(_label_to_int)
        else:
            logger.error(f"  {path.name}: needs a 'label' column or one of {TOXICITY_LABELS} — skipped")
            continue
        df = df[df["toxic"] != UNKNOWN_LABEL] if "toxic" in df.columns else df
        if "split" not in df.columns or not df["split"].isin(["train", "val", "test"]).all():
            df["split"] = _hash_split(df["text"].astype(str))
        name = f"custom_{path.stem}"
        df = _finalize(df, name)
        out[name] = df
        logger.info(f"  {name}: {len(df):,} rows | toxic={(df['toxic'] == 1).mean():.1%} | "
                    f"with context={(df['context'].str.len() > 0).mean():.0%} | splits={df['split'].value_counts().to_dict()}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# Master pipeline
# ═══════════════════════════════════════════════════════════════════════

def download_and_format_all(skip_indic_corp: bool = False, jigsaw_sample_size: int = 200_000,
                            telugu_sample_size: int = 250_000) -> Dict[str, pd.DataFrame]:
    logger.info("=" * 70)
    logger.info("DATASET DOWNLOAD & FORMATTING")
    logger.info("=" * 70)
    results: Dict[str, pd.DataFrame] = {}

    steps = [
        ("toxic_chat", download_toxic_chat),
        ("jigsaw_bias", lambda: sample_jigsaw(download_jigsaw_bias(), jigsaw_sample_size)),
        ("dravidian_codemix", download_dravidian_codemix),
        ("cultural_safety_telugu", download_cultural_safety_telugu),
    ]
    for name, fn in steps:
        try:
            df = fn()
            df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
            results[name] = df
            logger.info(f"✓ {name}: {len(df):,} rows")
        except Exception as e:
            logger.error(f"✗ {name} failed: {e}")

    for name, df in load_custom_datasets().items():
        df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
        results[name] = df

    if not skip_indic_corp:
        try:
            results["indic_corp_telugu"] = download_indic_corp_telugu(telugu_sample_size)
        except Exception as e:
            logger.error(f"✗ indic_corp_telugu failed: {e}")

    logger.info("SUMMARY")
    for name, df in results.items():
        tox = f"{(df['toxic'] == 1).mean():.1%}" if "toxic" in df.columns else "n/a"
        logger.info(f"  {name:28s} {len(df):>10,} rows | toxic {tox}")
    return results


def build_stage1_dataset(toxic_chat: pd.DataFrame, jigsaw: pd.DataFrame, val_ratio: float = 0.1,
                         seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combine toxic-chat + Jigsaw; split by conversation id where available, stratified otherwise."""
    from sklearn.model_selection import train_test_split

    combined = pd.concat([toxic_chat, jigsaw], ignore_index=True)
    has_conv = combined["conv_id"].astype(str).str.len() > 0
    conv, noconv = combined[has_conv], combined[~has_conv]

    parts_train, parts_val = [], []
    if len(conv):
        conv_ids = np.asarray(conv["conv_id"].astype(str).unique(), dtype=object)  # pandas 3 returns an Arrow array
        tr_ids, va_ids = train_test_split(conv_ids, test_size=val_ratio, random_state=seed)
        parts_train.append(conv[conv["conv_id"].isin(tr_ids)])
        parts_val.append(conv[conv["conv_id"].isin(va_ids)])
    if len(noconv):
        tr, va = train_test_split(noconv, test_size=val_ratio, random_state=seed, stratify=noconv["toxic"])
        parts_train.append(tr)
        parts_val.append(va)

    train_df = pd.concat(parts_train).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(parts_val).sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df.to_parquet(PROCESSED_DIR / "stage1_train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "stage1_val.parquet", index=False)
    logger.info(f"  Stage 1 train={len(train_df):,} val={len(val_df):,} | toxic train={train_df['toxic'].mean():.1%}")
    return train_df, val_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and format all datasets")
    parser.add_argument("--skip-indic-corp", action="store_true")
    parser.add_argument("--jigsaw-sample", type=int, default=200_000)
    args = parser.parse_args()
    res = download_and_format_all(skip_indic_corp=args.skip_indic_corp, jigsaw_sample_size=args.jigsaw_sample)
    if "toxic_chat" in res:
        build_stage1_dataset(res["toxic_chat"], res.get("jigsaw_bias", res["toxic_chat"].iloc[:0]))
