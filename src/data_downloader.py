"""
Phase 1: Zero-Gated Open Dataset Strategy - Download & Format.

Downloads all required datasets from Hugging Face (ungated, publicly accessible)
and converts them to a unified format for the training pipeline.

Datasets handled:
  1. lmsys/toxic-chat          → Stage 1: Context Mechanics
  2. Jigsaw Unintended Bias    → Stage 1: Identity Debiasing  
  3. ai4bharat/IndicCorp       → Stage 2: Telugu MLM
  4. DravidianCodeMix           → Stage 3A: Code-Mix Slang
  5. Cultural-Safety-Telugu     → Stage 3B: Telugu Safety
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    PROJECT_ROOT,
    RAW_DIR,
    PROCESSED_DIR,
    DATASETS,
    TOXICITY_LABELS,
    LABEL2ID,
    DRAVIDIAN_LABEL_MAP,
    CULTURAL_SAFETY_MAP,
    get_logger,
)

logger = get_logger("data_downloader")


# ═══════════════════════════════════════════════════════════════════════
# 1. lmsys/toxic-chat — Stage 1: Context Mechanics
# ═══════════════════════════════════════════════════════════════════════

def download_toxic_chat() -> pd.DataFrame:
    """
    Download and format lmsys/toxic-chat for Stage 1 training.
    
    This dataset contains multi-turn user-assistant dialogs teaching
    self-attention how preceding turns alter target utterance toxicity.
    
    Unified Schema:
        text: str           - The user input text
        context: str        - Model output / conversational context  
        toxic: int          - Binary toxicity label
        severe_toxic: int   - 0 (not available in this dataset)
        identity_attack: int - 0
        insult: int         - 0
        threat: int         - 0
        source: str         - "toxic_chat"
    """
    logger.info("Processing lmsys/toxic-chat...")
    
    # Check for local files first (already downloaded)
    local_train = PROJECT_ROOT / "toxic-chat_annotation_train.csv"
    local_test = PROJECT_ROOT / "toxic-chat_annotation_test.csv"
    local_train_alt = PROJECT_ROOT / "toxic-chat_annotation_train (1).csv"
    local_test_alt = PROJECT_ROOT / "toxic-chat_annotation_test (1).csv"
    
    dfs = []
    
    # Try local files
    for f in [local_train, local_train_alt]:
        if f.exists():
            logger.info(f"  Loading local file: {f.name}")
            df = pd.read_csv(f, encoding="utf-8")
            dfs.append(df)
            break
    
    for f in [local_test, local_test_alt]:
        if f.exists():
            logger.info(f"  Loading local file: {f.name}")
            df = pd.read_csv(f, encoding="utf-8")
            dfs.append(df)
            break
    
    if not dfs:
        # Download from Hugging Face
        logger.info("  Downloading from Hugging Face...")
        from datasets import load_dataset
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")
        for split in ["train", "test"]:
            if split in ds:
                dfs.append(ds[split].to_pandas())
    
    if not dfs:
        raise RuntimeError("Failed to load toxic-chat dataset")
    
    raw_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"  Raw shape: {raw_df.shape}")
    logger.info(f"  Columns: {list(raw_df.columns)}")
    
    # Map to unified schema
    unified = pd.DataFrame({
        "text": raw_df["user_input"].fillna(""),
        "context": raw_df["model_output"].fillna(""),
        "conv_id": raw_df["conv_id"].fillna(""),
        "toxic": (raw_df["toxicity"].fillna(0) >= 1).astype(int),
        "severe_toxic": 0,
        "identity_attack": 0,
        "insult": 0,
        "threat": 0,
        "source": "toxic_chat",
    })
    
    logger.info(f"  Toxic distribution: {unified['toxic'].value_counts().to_dict()}")
    return unified


# ═══════════════════════════════════════════════════════════════════════
# 2. Jigsaw Unintended Bias — Stage 1: Identity Debiasing
# ═══════════════════════════════════════════════════════════════════════

def download_jigsaw_bias() -> pd.DataFrame:
    """
    Load and format Jigsaw Unintended Bias dataset for Stage 1.
    
    This dataset prevents false-positive flags when demographic, religious,
    or regional identity terms appear in neutral discussions.
    
    The 'target' column is a continuous toxicity score [0, 1].
    We binarize at threshold 0.5 for the primary label, and extract
    sub-labels from dedicated columns.
    """
    logger.info("Processing Jigsaw Unintended Bias...")
    
    jigsaw_dir = PROJECT_ROOT / "jigsaw-unintended-bias-in-toxicity-classification"
    train_path = jigsaw_dir / "train.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(
            f"Jigsaw dataset not found at {train_path}. "
            "Download from Kaggle: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification"
        )
    
    logger.info(f"  Loading {train_path} (this may take a minute)...")
    
    # Read only required columns to save memory
    cols_needed = [
        "comment_text", "target", "severe_toxicity", "obscene",
        "identity_attack", "insult", "threat", "sexual_explicit",
    ]
    df = pd.read_csv(train_path, usecols=cols_needed)
    logger.info(f"  Raw shape: {df.shape}")
    
    # Binarize continuous scores at 0.5 threshold
    unified = pd.DataFrame({
        "text": df["comment_text"].fillna(""),
        "context": "",  # No conversational context in Jigsaw
        "conv_id": "",
        "toxic": (df["target"].fillna(0) >= 0.5).astype(int),
        "severe_toxic": (df["severe_toxicity"].fillna(0) >= 0.5).astype(int),
        "identity_attack": (df["identity_attack"].fillna(0) >= 0.5).astype(int),
        "insult": (df["insult"].fillna(0) >= 0.5).astype(int),
        "threat": (df["threat"].fillna(0) >= 0.5).astype(int),
        "source": "jigsaw_bias",
    })
    
    logger.info(f"  Label distribution:")
    for label in TOXICITY_LABELS:
        logger.info(f"    {label}: {unified[label].value_counts().to_dict()}")
    
    return unified


# ═══════════════════════════════════════════════════════════════════════
# 3. IndicCorp (Telugu subset) — Stage 2: Telugu MLM Adaptation
# ═══════════════════════════════════════════════════════════════════════

def download_indic_corp_telugu(sample_size: int = 250_000) -> pd.DataFrame:
    """
    Download Telugu text from IndicCorp for Masked Language Modeling.
    
    This provides billions of native Telugu tokens to sharpen subword
    embeddings in the encoder backbone.
    
    We sample `sample_size` sentences for tractable MLM fine-tuning.
    """
    logger.info(f"Processing IndicCorp Telugu (sampling {sample_size:,} sentences)...")
    
    output_path = PROCESSED_DIR / "indic_corp_telugu_mlm.parquet"
    if output_path.exists():
        logger.info(f"  Already processed. Loading from {output_path}")
        return pd.read_parquet(output_path)
    
    from datasets import load_dataset
    
    logger.info("  Downloading Telugu subset from ai4bharat/IndicCorp...")
    
    try:
        # Try streaming to avoid downloading the entire corpus
        ds = load_dataset(
            "ai4bharat/IndicCorp-Monolingual",
            "te",
            split="train",
            streaming=True,
        )
        
        # Sample sentences
        texts = []
        for i, example in enumerate(ds):
            if i >= sample_size:
                break
            text_field = example.get("text", example.get("sentence", ""))
            if text_field and len(text_field.strip()) > 10:
                texts.append(text_field.strip())
            
            if (i + 1) % 50000 == 0:
                logger.info(f"    Collected {len(texts):,} / {sample_size:,} sentences...")
        
        df = pd.DataFrame({"text": texts, "source": "indic_corp_telugu"})
        logger.info(f"  Collected {len(df):,} Telugu sentences")
        
    except Exception as e:
        logger.warning(f"  Failed to download IndicCorp: {e}")
        logger.info("  Attempting alternative: ai4bharat/samanantar (Telugu subset)...")
        
        try:
            ds = load_dataset(
                "ai4bharat/samanantar",
                "te",
                split="train",
                streaming=True,
            )
            
            texts = []
            for i, example in enumerate(ds):
                if i >= sample_size:
                    break
                text = example.get("tgt", example.get("src", ""))
                if text and len(text.strip()) > 10:
                    texts.append(text.strip())
            
            df = pd.DataFrame({"text": texts, "source": "samanantar_telugu"})
            logger.info(f"  Collected {len(df):,} Telugu sentences from Samanantar")
            
        except Exception as e2:
            logger.error(f"  All Telugu corpus downloads failed: {e2}")
            logger.info("  Creating placeholder — you'll need to provide Telugu text manually.")
            df = pd.DataFrame({"text": [], "source": []})
    
    if len(df) > 0:
        df.to_parquet(output_path, index=False)
        logger.info(f"  Saved to {output_path}")
    
    return df


# ═══════════════════════════════════════════════════════════════════════
# 4. DravidianCodeMix — Stage 3A: Code-Mix Slang
# ═══════════════════════════════════════════════════════════════════════

def download_dravidian_codemix() -> pd.DataFrame:
    """
    Load DravidianCodeMix dataset for Stage 3A training.
    
    Captures phonetic spelling variations, Latin-script transliteration,
    and Dravidian syntactic structures.
    
    Local parquet files are checked first (already present in workspace).
    """
    logger.info("Processing DravidianCodeMix...")
    
    # Check for local parquet files
    train_parquet = PROJECT_ROOT / "train-00000-of-00001.parquet"
    dev_parquet = PROJECT_ROOT / "dev-00000-of-00001.parquet"
    test_parquet = PROJECT_ROOT / "test-00000-of-00001.parquet"
    
    dfs = []
    
    for split_name, path in [("train", train_parquet), ("dev", dev_parquet), ("test", test_parquet)]:
        if path.exists():
            logger.info(f"  Loading local {split_name}: {path.name}")
            df = pd.read_parquet(path)
            df["split"] = split_name
            dfs.append(df)
    
    if not dfs:
        # Download from Hugging Face
        logger.info("  Downloading from Hugging Face...")
        from datasets import load_dataset
        ds = load_dataset("IsaacRodgz/DravidianCodeMix-Dataset")
        for split in ds:
            df = ds[split].to_pandas()
            df["split"] = split
            dfs.append(df)
    
    raw_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"  Raw shape: {raw_df.shape}")
    logger.info(f"  Columns: {list(raw_df.columns)}")
    logger.info(f"  Label distribution: {raw_df['label'].value_counts().to_dict()}")
    
    # Map binary label to multi-label schema
    unified = pd.DataFrame({
        "text": raw_df["text"].fillna(""),
        "context": "",
        "conv_id": "",
        "split": raw_df["split"],
        "source": "dravidian_codemix",
    })
    
    # Map labels: 0 = not offensive, 1 = offensive
    for label_name in TOXICITY_LABELS:
        unified[label_name] = 0
    
    for idx, row in raw_df.iterrows():
        label_val = int(row["label"])
        if label_val in DRAVIDIAN_LABEL_MAP:
            for i, label_name in enumerate(TOXICITY_LABELS):
                unified.at[idx, label_name] = DRAVIDIAN_LABEL_MAP[label_val][i]
        elif label_val >= 1:
            # Any non-zero label → toxic
            unified.at[idx, "toxic"] = 1
    
    logger.info(f"  Unified toxic distribution: {unified['toxic'].value_counts().to_dict()}")
    return unified


# ═══════════════════════════════════════════════════════════════════════
# 5. Cultural-Safety-Telugu — Stage 3B: Telugu Safety
# ═══════════════════════════════════════════════════════════════════════

def download_cultural_safety_telugu() -> pd.DataFrame:
    """
    Load Cultural-Safety-Telugu dataset for Stage 3B training.
    
    Categorizes native Telugu prompts across:
    Harassment, Profanity, Hate Speech, Identity Attacks, and Violence.
    """
    logger.info("Processing Cultural-Safety-Telugu...")
    
    # Check for local JSONL file
    local_jsonl = PROJECT_ROOT / "train_0.jsonl"
    
    if local_jsonl.exists():
        logger.info(f"  Loading local file: {local_jsonl.name}")
        records = []
        with open(local_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        raw_df = pd.DataFrame(records)
    else:
        # Download from Hugging Face
        logger.info("  Downloading from Hugging Face...")
        from datasets import load_dataset
        ds = load_dataset("salmankhanpm/Cultural-Safety-Telugu", split="train")
        raw_df = ds.to_pandas()
    
    logger.info(f"  Raw shape: {raw_df.shape}")
    logger.info(f"  Columns: {list(raw_df.columns)}")
    
    # Extract text and labels
    # Use prompt_telugu if available, otherwise prompt
    text_col = "prompt_telugu" if "prompt_telugu" in raw_df.columns else "prompt"
    label_col = "safety_label" if "safety_label" in raw_df.columns else "prompt_label"
    
    logger.info(f"  Using text column: {text_col}")
    logger.info(f"  Using label column: {label_col}")
    
    if label_col in raw_df.columns:
        logger.info(f"  Label distribution: {raw_df[label_col].value_counts().head(10).to_dict()}")
    
    # Build unified DataFrame
    unified = pd.DataFrame({
        "text": raw_df[text_col].fillna(""),
        "context": "",
        "conv_id": "",
        "source": "cultural_safety_telugu",
    })
    
    # Initialize labels
    for label_name in TOXICITY_LABELS:
        unified[label_name] = 0
    
    # Primary label: safety_label (SAFE/UNSAFE)
    if label_col in raw_df.columns:
        for idx, row in raw_df.iterrows():
            label = str(row.get(label_col, "")).strip().upper()
            if label == "UNSAFE":
                unified.at[idx, "toxic"] = 1
    
    # Overlay specific categories for fine-grained labels
    if "safety_categories" in raw_df.columns:
        for idx, row in raw_df.iterrows():
            cats = row.get("safety_categories")
            if cats is None:
                continue
            if isinstance(cats, str):
                try:
                    cats = json.loads(cats)
                except:
                    cats = [cats]
            if not isinstance(cats, list):
                cats = [str(cats)]
            
            for cat in cats:
                cat_str = str(cat).strip()
                if cat_str in CULTURAL_SAFETY_MAP:
                    labels = CULTURAL_SAFETY_MAP[cat_str]
                    for i, label_name in enumerate(TOXICITY_LABELS):
                        if labels[i]:
                            unified.at[idx, label_name] = 1
    
    logger.info(f"  Unified label distribution:")
    for label in TOXICITY_LABELS:
        logger.info(f"    {label}: {unified[label].value_counts().to_dict()}")
    
    return unified


# ═══════════════════════════════════════════════════════════════════════
# Master Download & Format Pipeline
# ═══════════════════════════════════════════════════════════════════════

def download_and_format_all(
    skip_indic_corp: bool = False,
    jigsaw_sample_size: int = 200_000,
) -> Dict[str, pd.DataFrame]:
    """
    Download and format all datasets for the training pipeline.
    
    Args:
        skip_indic_corp: Skip IndicCorp download (large, for Stage 2 only)
        jigsaw_sample_size: Number of Jigsaw samples to use (full dataset is ~1.8M)
    
    Returns:
        Dict mapping dataset name to formatted DataFrame
    """
    logger.info("=" * 70)
    logger.info("STARTING DATASET DOWNLOAD & FORMATTING PIPELINE")
    logger.info("=" * 70)
    
    results = {}
    
    # 1. Toxic Chat
    try:
        toxic_chat = download_toxic_chat()
        toxic_chat.to_parquet(PROCESSED_DIR / "toxic_chat.parquet", index=False)
        results["toxic_chat"] = toxic_chat
        logger.info(f"✓ toxic_chat: {len(toxic_chat):,} samples saved")
    except Exception as e:
        logger.error(f"✗ toxic_chat failed: {e}")
    
    # 2. Jigsaw Bias
    try:
        jigsaw = download_jigsaw_bias()
        
        # Sample to reduce training time
        if len(jigsaw) > jigsaw_sample_size:
            # Stratified sampling to preserve class balance
            toxic_mask = jigsaw["toxic"] == 1
            n_toxic = toxic_mask.sum()
            n_safe = len(jigsaw) - n_toxic
            
            toxic_ratio = n_toxic / len(jigsaw)
            n_toxic_sample = int(jigsaw_sample_size * toxic_ratio)
            n_safe_sample = jigsaw_sample_size - n_toxic_sample
            
            # Ensure minimum toxic samples
            n_toxic_sample = max(n_toxic_sample, min(n_toxic, jigsaw_sample_size // 4))
            n_safe_sample = jigsaw_sample_size - n_toxic_sample
            
            toxic_samples = jigsaw[toxic_mask].sample(
                n=min(n_toxic_sample, n_toxic), random_state=42
            )
            safe_samples = jigsaw[~toxic_mask].sample(
                n=min(n_safe_sample, n_safe), random_state=42
            )
            jigsaw = pd.concat([toxic_samples, safe_samples]).sample(
                frac=1, random_state=42
            ).reset_index(drop=True)
            
            logger.info(f"  Sampled {len(jigsaw):,} from Jigsaw (stratified)")
        
        jigsaw.to_parquet(PROCESSED_DIR / "jigsaw_bias.parquet", index=False)
        results["jigsaw_bias"] = jigsaw
        logger.info(f"✓ jigsaw_bias: {len(jigsaw):,} samples saved")
    except Exception as e:
        logger.error(f"✗ jigsaw_bias failed: {e}")
    
    # 3. IndicCorp Telugu (for Stage 2 MLM)
    if not skip_indic_corp:
        try:
            indic_corp = download_indic_corp_telugu()
            results["indic_corp_telugu"] = indic_corp
            logger.info(f"✓ indic_corp_telugu: {len(indic_corp):,} sentences saved")
        except Exception as e:
            logger.error(f"✗ indic_corp_telugu failed: {e}")
    else:
        logger.info("⊘ indic_corp_telugu: Skipped (skip_indic_corp=True)")
    
    # 4. DravidianCodeMix
    try:
        dravidian = download_dravidian_codemix()
        dravidian.to_parquet(PROCESSED_DIR / "dravidian_codemix.parquet", index=False)
        results["dravidian_codemix"] = dravidian
        logger.info(f"✓ dravidian_codemix: {len(dravidian):,} samples saved")
    except Exception as e:
        logger.error(f"✗ dravidian_codemix failed: {e}")
    
    # 5. Cultural Safety Telugu
    try:
        cultural = download_cultural_safety_telugu()
        cultural.to_parquet(PROCESSED_DIR / "cultural_safety_telugu.parquet", index=False)
        results["cultural_safety_telugu"] = cultural
        logger.info(f"✓ cultural_safety_telugu: {len(cultural):,} samples saved")
    except Exception as e:
        logger.error(f"✗ cultural_safety_telugu failed: {e}")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("DATASET DOWNLOAD SUMMARY")
    logger.info("=" * 70)
    for name, df in results.items():
        toxic_count = df["toxic"].sum() if "toxic" in df.columns else "N/A"
        logger.info(f"  {name:30s}: {len(df):>10,} rows  |  Toxic: {toxic_count}")
    logger.info("=" * 70)
    
    return results


# ═══════════════════════════════════════════════════════════════════════
# Stage 1 Combined Dataset Builder
# ═══════════════════════════════════════════════════════════════════════

def build_stage1_dataset(
    toxic_chat: pd.DataFrame,
    jigsaw: pd.DataFrame,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine Toxic Chat + Jigsaw into a unified Stage 1 training dataset.
    
    Creates train/val splits while ensuring:
      - Toxic Chat conversation IDs are not split across train/val
      - Stratified sampling on the toxic label
    
    Returns:
        (train_df, val_df) tuple
    """
    logger.info("Building Stage 1 combined dataset...")
    
    # Combine
    combined = pd.concat([toxic_chat, jigsaw], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split: keep conversation groups together
    from sklearn.model_selection import train_test_split
    
    # For entries with conv_id, group by conversation
    has_conv = combined["conv_id"].notna() & (combined["conv_id"] != "")
    conv_entries = combined[has_conv]
    no_conv_entries = combined[~has_conv]
    
    if len(conv_entries) > 0:
        unique_convs = conv_entries["conv_id"].unique()
        train_convs, val_convs = train_test_split(
            unique_convs, test_size=val_ratio, random_state=42
        )
        conv_train = conv_entries[conv_entries["conv_id"].isin(train_convs)]
        conv_val = conv_entries[conv_entries["conv_id"].isin(val_convs)]
    else:
        conv_train = pd.DataFrame()
        conv_val = pd.DataFrame()
    
    if len(no_conv_entries) > 0:
        noconv_train, noconv_val = train_test_split(
            no_conv_entries, test_size=val_ratio, random_state=42,
            stratify=no_conv_entries["toxic"],
        )
    else:
        noconv_train = pd.DataFrame()
        noconv_val = pd.DataFrame()
    
    train_df = pd.concat([conv_train, noconv_train], ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    
    val_df = pd.concat([conv_val, noconv_val], ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    
    logger.info(f"  Stage 1 Train: {len(train_df):,} samples")
    logger.info(f"  Stage 1 Val:   {len(val_df):,} samples")
    logger.info(f"  Train toxic %: {train_df['toxic'].mean():.2%}")
    logger.info(f"  Val toxic %:   {val_df['toxic'].mean():.2%}")
    
    # Save
    train_df.to_parquet(PROCESSED_DIR / "stage1_train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "stage1_val.parquet", index=False)
    
    return train_df, val_df


# ═══════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and format all datasets")
    parser.add_argument("--skip-indic-corp", action="store_true",
                        help="Skip IndicCorp Telugu download")
    parser.add_argument("--jigsaw-sample", type=int, default=200_000,
                        help="Number of Jigsaw samples to use")
    args = parser.parse_args()
    
    results = download_and_format_all(
        skip_indic_corp=args.skip_indic_corp,
        jigsaw_sample_size=args.jigsaw_sample,
    )
    
    # Build Stage 1 combined dataset if both sources are available
    if "toxic_chat" in results and "jigsaw_bias" in results:
        build_stage1_dataset(results["toxic_chat"], results["jigsaw_bias"])
