"""
Stage 3: Code-Mix & Telugu Slang Fine-Tuning.

Fine-tunes the model on Dravidian code-mixed data (IsaacRodgz/DravidianCodeMix)
and native Telugu safety data (salmankhanpm/Cultural-Safety-Telugu).

Key Implementation Details:
  - Freezes bottom 50% of encoder layers to preserve Stage 1 relational context reasoning
  - Updates top 50% layers + Target-Span Pooling classification head
  - Uses Class-Weighted Focal Loss (alpha=0.75, gamma=2.0)
  - Maps Dravidian profanity, regional slurs, and Telugu script insults into toxic subspace
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    Stage3Config,
    PROCESSED_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    PRIMARY_MODEL,
    TOXICITY_LABELS,
    MAX_SEQ_LENGTH,
    TARGET_PREFIX,
    PARENT_PREFIX,
    SEP_TOKEN,
    get_logger,
)
from src.model import (
    ContextAwareToxicityClassifier,
    load_tokenizer,
    tokenize_with_target_mask,
)
from src.preprocessing import IndicPreprocessor

logger = get_logger("stage3_training")


# ═══════════════════════════════════════════════════════════════════════
# Stage 3 Dataset
# ═══════════════════════════════════════════════════════════════════════

class Stage3Dataset(Dataset):
    """
    Dataset combining Dravidian Code-Mix and Cultural-Safety-Telugu data.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = MAX_SEQ_LENGTH,
        preprocessor: Optional[IndicPreprocessor] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor
        self.label_columns = [l for l in TOXICITY_LABELS if l in df.columns]
        
        logger.info(f"Stage3Dataset initialized with {len(self.df):,} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        text = str(row.get("text", ""))
        context = str(row.get("context", ""))
        
        # Apply preprocessing
        if self.preprocessor:
            text = self.preprocessor.clean_text(text)
            if context and context.strip():
                context = self.preprocessor.clean_text(context)
        
        # Format text with prefix
        if context and context.strip():
            formatted = f"{PARENT_PREFIX} {context} {SEP_TOKEN} {TARGET_PREFIX} {text}"
        else:
            formatted = f"{TARGET_PREFIX} {text}"
        
        encoding = tokenize_with_target_mask(
            formatted, self.tokenizer, self.max_length
        )
        
        # Labels
        labels = torch.zeros(len(self.label_columns), dtype=torch.float)
        for i, col in enumerate(self.label_columns):
            labels[i] = float(row.get(col, 0))
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "target_mask": encoding["target_mask"],
            "labels": labels,
        }


# ═══════════════════════════════════════════════════════════════════════
# Evaluation Helper
# ═══════════════════════════════════════════════════════════════════════

def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_mask = batch["target_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_mask=target_mask,
                labels=labels,
            )
            
            total_loss += outputs["loss"].item()
            probs = outputs["probs"].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            all_preds.append(preds)
            all_targets.append(labels.cpu().numpy().astype(int))
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    metrics = {
        "loss": total_loss / max(1, len(val_loader)),
        "macro_f1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "micro_f1": f1_score(all_targets, all_preds, average="micro", zero_division=0),
    }
    
    for i, label_name in enumerate(TOXICITY_LABELS):
        if i < all_targets.shape[1]:
            metrics[f"{label_name}_f1"] = f1_score(
                all_targets[:, i], all_preds[:, i], zero_division=0
            )
    
    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Stage 3 Training Function
# ═══════════════════════════════════════════════════════════════════════

def train_stage3(
    config: Optional[Stage3Config] = None,
    model_name: str = PRIMARY_MODEL,
    base_checkpoint: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """
    Execute Stage 3 training: Dravidian Code-Mix & Telugu Safety.
    
    Args:
        config: Stage3Config hyperparameters
        model_name: HuggingFace model identifier
        base_checkpoint: Path to checkpoint from previous stage (Stage 1 or 2)
        max_samples: Optional limit on training samples for quick runs
    """
    if config is None:
        config = Stage3Config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # ─── Load Data ───
    dravidian_path = PROCESSED_DIR / "dravidian_codemix.parquet"
    telugu_safety_path = PROCESSED_DIR / "cultural_safety_telugu.parquet"
    
    dfs = []
    if dravidian_path.exists():
        df_drav = pd.read_parquet(dravidian_path)
        logger.info(f"  Loaded Dravidian Code-Mix: {len(df_drav):,} samples")
        dfs.append(df_drav)
    if telugu_safety_path.exists():
        df_tel = pd.read_parquet(telugu_safety_path)
        logger.info(f"  Loaded Cultural-Safety-Telugu: {len(df_tel):,} samples")
        dfs.append(df_tel)
    
    if not dfs:
        logger.error("No Stage 3 data found in processed directory!")
        return
    
    combined_df = pd.concat(dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    if max_samples and len(combined_df) > max_samples:
        combined_df = combined_df.iloc[:max_samples]
        logger.info(f"  Sampled dataset to {len(combined_df):,} samples")
    
    # Split 90% train / 10% val
    val_size = int(len(combined_df) * 0.1)
    train_df = combined_df.iloc[:-val_size].reset_index(drop=True)
    val_df = combined_df.iloc[-val_size:].reset_index(drop=True)
    logger.info(f"  Train set: {len(train_df):,} | Val set: {len(val_df):,}")
    
    # ─── Tokenizer & Preprocessor ───
    tokenizer = load_tokenizer(model_name)
    preprocessor = IndicPreprocessor(
        enable_transliteration=False,
        enable_normalization=True,
    )
    
    train_dataset = Stage3Dataset(train_df, tokenizer, config.max_seq_length, preprocessor)
    val_dataset = Stage3Dataset(val_df, tokenizer, config.max_seq_length, preprocessor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # ─── Model Initialization ───
    model = ContextAwareToxicityClassifier(
        model_name=model_name,
        num_labels=5,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
        use_target_span_pooling=True,
    )
    
    # Load weights if checkpoint provided
    if base_checkpoint:
        ckpt_path = Path(base_checkpoint)
        if (ckpt_path / "model.pt").exists():
            logger.info(f"Loading weights from {ckpt_path / 'model.pt'}")
            state = torch.load(ckpt_path / "model.pt", map_location="cpu")
            model.load_state_dict(state, strict=False)
            logger.info("  ✓ Loaded weights successfully")
        elif (ckpt_path / "model").exists():
            logger.info(f"Loading encoder weights from {ckpt_path / 'model'}")
            from transformers import AutoModel
            model.encoder = AutoModel.from_pretrained(str(ckpt_path / "model"))
            logger.info("  ✓ Loaded encoder from MLM checkpoint")
    
    # Freeze bottom 50% layers (preserving Stage 1 relational context reasoning)
    model.freeze_layers(freeze_ratio=config.freeze_ratio)
    model.to(device)
    
    # ─── Optimizer & Scheduler ───
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = GradScaler() if config.fp16 else None
    
    logger.info("=" * 70)
    logger.info("STARTING STAGE 3 TRAINING: CODE-MIX & TELUGU SLANG")
    logger.info(f"  Epochs: {config.num_epochs} | Batch: {config.batch_size} | LR: {config.learning_rate}")
    logger.info("=" * 70)
    
    best_val_f1 = 0.0
    global_step = 0
    save_dir = CHECKPOINT_DIR / "stage3_best"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Stage 3 Epoch {epoch+1}/{config.num_epochs}", dynamic_ncols=True)
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_mask = batch["target_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            if config.fp16 and scaler is not None:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        target_mask=target_mask,
                        labels=labels,
                    )
                    loss = outputs["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_mask=target_mask,
                    labels=labels,
                )
                loss = outputs["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            f"Epoch {epoch+1} Val Macro-F1: {val_metrics['macro_f1']:.4f} | Loss: {val_metrics['loss']:.4f}"
        )
        if val_metrics["macro_f1"] > best_val_f1 or epoch == 0:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), save_dir / "model.pt")
            tokenizer.save_pretrained(save_dir / "tokenizer")
            with open(save_dir / "metadata.json", "w") as f:
                json.dump({
                    "stage": 3,
                    "epoch": epoch + 1,
                    "macro_f1": best_val_f1,
                    "metrics": val_metrics,
                }, f, indent=2)
            logger.info(f"  ✓ Saved Stage 3 best checkpoint to {save_dir}")
    
    logger.info(f"Stage 3 complete. Best Val Macro-F1: {best_val_f1:.4f}")
    return str(save_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3: Code-Mix & Telugu Slang Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-checkpoint", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    
    cfg = Stage3Config(learning_rate=args.lr, num_epochs=args.epochs, batch_size=args.batch_size)
    train_stage3(config=cfg, base_checkpoint=args.base_checkpoint, max_samples=args.max_samples)
