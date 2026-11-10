"""
Stage 3: Code-Mix & Telugu Slang Fine-Tuning.

Data: DravidianCodeMix (Tamil code-mixed offensive) + Cultural-Safety-Telugu
      + any file you drop in data/raw/custom/ (see src/data_downloader.py:load_custom_datasets).
Initialisation (curriculum order):
  * classifier heads + encoder from Stage 1 (stage1_best/model.pt), then
  * encoder replaced by the Telugu-adapted Stage 2 encoder (stage2_best/encoder) if it exists.
The old code picked stage1 *or* stage2 (stage1 first), so Stage 2 was never used.

Splits: each source's own train / val / test split is respected; the test split
is never seen in training and is used by src/evaluation.py.
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CHECKPOINT_DIR, PRIMARY_MODEL, Stage3Config, get_logger, indic_supervised_sources
from src.model import ContextAwareToxicityClassifier, load_model_weights, load_tokenizer
from src.preprocessing import IndicPreprocessor
from src.training_utils import ToxicityDataset, train_classifier

logger = get_logger("stage3_training")

def load_stage3_frames(max_samples: Optional[int] = None, seed: int = 42):
    frames = []
    for p in indic_supervised_sources():
        df = pd.read_parquet(p)
        logger.info(f"  {p.name}: {len(df):,} rows, splits={df['split'].value_counts().to_dict()}")
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No Stage 3 data in data/processed/ — run `python run_pipeline.py --download-only`")
    df = pd.concat(frames, ignore_index=True)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    if max_samples:
        train = train.sample(min(len(train), max_samples), random_state=seed)
        val = val.sample(min(len(val), max(200, max_samples // 5)), random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_stage3_init_model(model_name: str, config: Stage3Config) -> ContextAwareToxicityClassifier:
    model = ContextAwareToxicityClassifier(model_name, focal_alpha=config.focal_alpha, focal_gamma=config.focal_gamma)
    stage1 = CHECKPOINT_DIR / "stage1_best"
    if (stage1 / "model.pt").exists():
        load_model_weights(model, stage1)
    else:
        logger.warning("No Stage 1 checkpoint — heads start from random init")

    for name in ["stage2_best", "stage2_final"]:
        enc_dir = CHECKPOINT_DIR / name / "encoder"
        if enc_dir.exists():
            logger.info(f"  Using Telugu-adapted encoder from {enc_dir}")
            model.encoder = AutoModel.from_pretrained(str(enc_dir))
            break
    else:
        logger.warning("No Stage 2 encoder found — continuing with the Stage 1 / pretrained encoder")
    return model


def train_stage3(config: Optional[Stage3Config] = None, model_name: str = PRIMARY_MODEL,
                 max_samples: Optional[int] = None, resume: bool = False):
    config = config or Stage3Config()
    train_df, val_df, test_df = load_stage3_frames(max_samples or config.max_samples, config.seed)
    logger.info(f"  train={len(train_df):,} val={len(val_df):,} test(held out)={len(test_df):,}")

    tokenizer = load_tokenizer(model_name)
    model = build_stage3_init_model(model_name, config)
    model.configure_training(config.freeze_ratio, config.freeze_embeddings, config.gradient_checkpointing)

    pre = IndicPreprocessor(enable_transliteration=False, enable_normalization=True)
    metrics = train_classifier(
        model, tokenizer,
        ToxicityDataset(train_df, tokenizer, config.max_seq_length, pre, train=True),
        ToxicityDataset(val_df, tokenizer, config.max_seq_length, pre),
        config,
        save_dir=CHECKPOINT_DIR / "stage3_best", stage="stage3",
        extra_metadata={"data": "DravidianCodeMix (Tamil) + Cultural-Safety-Telugu", "train_rows": len(train_df)},
        resume=resume,
    )
    logger.info(f"Stage 3 done. Best val macro-F1 (tuned): {metrics.get('macro_f1@threshold', float('nan')):.4f}")
    return metrics


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stage 3: Code-Mix & Telugu Slang Fine-Tuning")
    p.add_argument("--model", default=PRIMARY_MODEL)
    p.add_argument("--epochs", type=int, default=Stage3Config.num_epochs)
    p.add_argument("--lr", type=float, default=Stage3Config.learning_rate)
    p.add_argument("--batch-size", type=int, default=Stage3Config.batch_size)
    p.add_argument("--max-samples", type=int, default=None)
    a = p.parse_args()
    train_stage3(Stage3Config(learning_rate=a.lr, num_epochs=a.epochs, batch_size=a.batch_size), a.model, a.max_samples)
