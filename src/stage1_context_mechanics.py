"""
Stage 1: English toxicity mechanics (lmsys/toxic-chat + Jigsaw Unintended Bias).

Trains the full encoder + attention heads on English multi-label toxicity.
Note: neither dataset contains real conversational threads, so this stage teaches
*what toxicity looks like* and identity-term debiasing; context use is learned in
Stage 4 (and should be learned from real threaded data — see docs/PROJECT_PLAN.md).
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CHECKPOINT_DIR, PRIMARY_MODEL, PROCESSED_DIR, Stage1Config, get_logger
from src.model import ContextAwareToxicityClassifier, load_tokenizer
from src.training_utils import ToxicityDataset, train_classifier

logger = get_logger("stage1_training")


def train_stage1(config: Optional[Stage1Config] = None, model_name: str = PRIMARY_MODEL):
    config = config or Stage1Config()
    train_path, val_path = PROCESSED_DIR / "stage1_train.parquet", PROCESSED_DIR / "stage1_val.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found — run `python run_pipeline.py --download-only` first.")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path) if val_path.exists() else None
    if config.max_samples:
        train_df = train_df.sample(min(len(train_df), config.max_samples), random_state=config.seed)
        if val_df is not None:
            val_df = val_df.sample(min(len(val_df), max(200, config.max_samples // 5)), random_state=config.seed)

    tokenizer = load_tokenizer(model_name)
    model = ContextAwareToxicityClassifier(model_name, focal_alpha=config.focal_alpha, focal_gamma=config.focal_gamma)
    model.configure_training(config.freeze_ratio, config.freeze_embeddings, config.gradient_checkpointing)

    # English data: preprocessing (Telugu normalisation / transliteration) is not applied.
    train_ds = ToxicityDataset(train_df, tokenizer, config.max_seq_length, train=True)
    val_ds = ToxicityDataset(val_df, tokenizer, config.max_seq_length) if val_df is not None else None

    metrics = train_classifier(
        model, tokenizer, train_ds, val_ds, config,
        save_dir=CHECKPOINT_DIR / "stage1_best", stage="stage1",
        extra_metadata={"data": "toxic-chat + jigsaw", "train_rows": len(train_df)},
    )
    logger.info(f"Stage 1 done. Best val macro-F1 (tuned): {metrics.get('macro_f1@threshold', float('nan')):.4f}")
    return metrics


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stage 1: English toxicity mechanics")
    p.add_argument("--model", default=PRIMARY_MODEL)
    p.add_argument("--lr", type=float, default=Stage1Config.learning_rate)
    p.add_argument("--epochs", type=int, default=Stage1Config.num_epochs)
    p.add_argument("--batch-size", type=int, default=Stage1Config.batch_size)
    p.add_argument("--max-samples", type=int, default=None)
    a = p.parse_args()
    train_stage1(Stage1Config(learning_rate=a.lr, num_epochs=a.epochs, batch_size=a.batch_size,
                              max_samples=a.max_samples), a.model)
