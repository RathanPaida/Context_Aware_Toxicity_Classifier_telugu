"""
Stage 1: Context Mechanics Training (English).

Fine-tunes the full IndicBERT-v3 encoder on English conversational toxicity data
(lmsys/toxic-chat + Jigsaw Unintended Bias) to teach self-attention how preceding
turns alter target utterance toxicity.

This is the foundation stage — the model learns:
  - How parent text flips the toxicity polarity of target text
  - Multi-label toxicity classification (toxic, severe, identity_attack, insult, threat)
  - Bias-aware behavior around demographic/religious/regional identity terms
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
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    Stage1Config,
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
    FocalLoss,
    load_tokenizer,
    tokenize_with_target_mask,
)
from src.preprocessing import IndicPreprocessor

logger = get_logger("stage1_training")


# ═══════════════════════════════════════════════════════════════════════
# Dataset Class
# ═══════════════════════════════════════════════════════════════════════

class Stage1Dataset(Dataset):
    """
    PyTorch Dataset for Stage 1 training data.
    
    Handles both:
      - Toxic Chat (has context from model_output)
      - Jigsaw Bias (no conversational context, single text)
    
    For Toxic Chat entries with context, formats as:
        [SEP] Parent: {context} [SEP] Target: {text} [SEP]
    
    For Jigsaw entries, formats as:
        Target: {text}
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
        
        logger.info(f"Stage1Dataset created: {len(self.df):,} samples")
        logger.info(f"  Label columns: {self.label_columns}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        text = str(row.get("text", ""))
        context = str(row.get("context", ""))
        
        # Apply preprocessing (stages 1-3) if available
        if self.preprocessor:
            text = self.preprocessor.clean_text(text)
            if context:
                context = self.preprocessor.clean_text(context)
        
        # Format with context structure
        if context and context.strip():
            formatted = f"{PARENT_PREFIX} {context} {SEP_TOKEN} {TARGET_PREFIX} {text}"
        else:
            formatted = f"{TARGET_PREFIX} {text}"
        
        # Tokenize with target mask
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
# Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train_stage1(
    config: Optional[Stage1Config] = None,
    model_name: str = PRIMARY_MODEL,
    resume_from: Optional[str] = None,
):
    """
    Execute Stage 1 training: Context Mechanics on English data.
    
    Args:
        config: Stage1Config hyperparameters (uses defaults if None)
        model_name: HuggingFace model identifier
        resume_from: Path to checkpoint to resume from
    """
    if config is None:
        config = Stage1Config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # ─── Load Data ───
    logger.info("Loading Stage 1 datasets...")
    
    train_path = PROCESSED_DIR / "stage1_train.parquet"
    val_path = PROCESSED_DIR / "stage1_val.parquet"
    
    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}")
        logger.info("Run data_downloader.py first to prepare datasets.")
        return
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path) if val_path.exists() else None
    
    if hasattr(config, "max_samples") and config.max_samples:
        if len(train_df) > config.max_samples:
            train_df = train_df.iloc[:config.max_samples].reset_index(drop=True)
        if val_df is not None and len(val_df) > config.max_samples // 5:
            val_df = val_df.iloc[:config.max_samples // 5].reset_index(drop=True)
    
    logger.info(f"  Train: {len(train_df):,} samples")
    if val_df is not None:
        logger.info(f"  Val:   {len(val_df):,} samples")
    
    # ─── Initialize Components ───
    logger.info("Initializing model and tokenizer...")
    
    tokenizer = load_tokenizer(model_name)
    
    # Initialize preprocessor (Stage 1 is English-only, minimal preprocessing)
    preprocessor = IndicPreprocessor(
        enable_transliteration=False,  # English data, no transliteration needed
        enable_normalization=False,
    )
    
    # Create datasets
    train_dataset = Stage1Dataset(train_df, tokenizer, config.max_seq_length, preprocessor)
    
    val_dataset = None
    if val_df is not None:
        val_dataset = Stage1Dataset(val_df, tokenizer, config.max_seq_length, preprocessor)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    
    # ─── Model ───
    model = ContextAwareToxicityClassifier(
        model_name=model_name,
        num_labels=config.num_labels,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
        use_target_span_pooling=True,
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params:     {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,}")
    
    # ─── Optimizer & Scheduler ───
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    
    # Mixed precision
    scaler = GradScaler() if config.fp16 else None
    
    logger.info(f"  Total steps: {total_steps:,}")
    logger.info(f"  Warmup steps: {warmup_steps:,}")
    logger.info(f"  LR: {config.learning_rate}")
    logger.info(f"  Focal Loss: α={config.focal_alpha}, γ={config.focal_gamma}")
    
    # ─── Training Loop ───
    logger.info("\n" + "=" * 70)
    logger.info("STARTING STAGE 1 TRAINING: CONTEXT MECHANICS")
    logger.info("=" * 70)
    
    best_val_f1 = 0.0
    global_step = 0
    training_log = []
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config.num_epochs}",
            dynamic_ncols=True,
        )
        
        for batch in pbar:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_mask = batch["target_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
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
            epoch_steps += 1
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{epoch_loss / epoch_steps:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })
            
            # Logging
            if global_step % config.logging_steps == 0:
                log_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": loss.item(),
                    "avg_loss": epoch_loss / epoch_steps,
                    "lr": scheduler.get_last_lr()[0],
                }
                training_log.append(log_entry)
            
            # Evaluation
            if val_loader and global_step % config.eval_steps == 0:
                val_metrics = evaluate(model, val_loader, device, config)
                logger.info(
                    f"  [Step {global_step}] Val F1={val_metrics['macro_f1']:.4f} | "
                    f"Val Loss={val_metrics['loss']:.4f}"
                )
                
                # Save best model
                if val_metrics["macro_f1"] > best_val_f1:
                    best_val_f1 = val_metrics["macro_f1"]
                    save_checkpoint(
                        model, tokenizer, optimizer, scheduler,
                        epoch, global_step, val_metrics,
                        CHECKPOINT_DIR / "stage1_best",
                    )
                    logger.info(f"  ✓ New best model saved (F1={best_val_f1:.4f})")
                
                model.train()
            
            # Periodic checkpoint
            if global_step % config.save_steps == 0:
                save_checkpoint(
                    model, tokenizer, optimizer, scheduler,
                    epoch, global_step, {},
                    CHECKPOINT_DIR / f"stage1_step_{global_step}",
                )
        
        # End of epoch evaluation
        avg_epoch_loss = epoch_loss / epoch_steps
        logger.info(f"\nEpoch {epoch+1} complete — Avg Loss: {avg_epoch_loss:.4f}")
        
        if val_loader:
            val_metrics = evaluate(model, val_loader, device, config)
            logger.info(f"  Val Macro-F1: {val_metrics['macro_f1']:.4f}")
            logger.info(f"  Val Loss:     {val_metrics['loss']:.4f}")
            
            for label_name in TOXICITY_LABELS:
                if label_name in val_metrics:
                    logger.info(f"    {label_name}: F1={val_metrics[label_name]:.4f}")
            
            if val_metrics["macro_f1"] >= best_val_f1 or epoch == 0:
                best_val_f1 = val_metrics["macro_f1"]
                save_checkpoint(
                    model, tokenizer, optimizer, scheduler,
                    epoch, global_step, val_metrics,
                    CHECKPOINT_DIR / "stage1_best",
                )
                logger.info(f"  ✓ New best model saved (F1={best_val_f1:.4f})")
    
    # Save final checkpoint
    save_checkpoint(
        model, tokenizer, optimizer, scheduler,
        config.num_epochs - 1, global_step, {},
        CHECKPOINT_DIR / "stage1_final",
    )
    
    # Save training log
    log_path = LOG_DIR / "stage1_training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)
    
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 1 TRAINING COMPLETE")
    logger.info(f"  Best Val Macro-F1: {best_val_f1:.4f}")
    logger.info(f"  Checkpoint: {CHECKPOINT_DIR / 'stage1_best'}")
    logger.info(f"  Training log: {log_path}")
    logger.info("=" * 70)
    
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model: ContextAwareToxicityClassifier,
    val_loader: DataLoader,
    device: torch.device,
    config: Stage1Config,
) -> Dict[str, float]:
    """
    Evaluate the model on the validation set.
    
    Returns:
        Dict with loss, macro_f1, per-label F1 scores
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0
    
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_mask = batch["target_mask"].to(device)
        labels = batch["labels"].to(device)
        
        if config.fp16:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_mask=target_mask,
                    labels=labels,
                )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_mask=target_mask,
                labels=labels,
            )
        
        total_loss += outputs["loss"].item()
        n_batches += 1
        
        # Threshold probabilities at 0.5
        preds = (outputs["probs"] > 0.5).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Compute metrics
    metrics = {
        "loss": total_loss / max(n_batches, 1),
    }
    
    # Per-label F1
    label_f1s = []
    for i, label_name in enumerate(TOXICITY_LABELS[:all_preds.shape[1]]):
        f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        metrics[label_name] = f1
        label_f1s.append(f1)
    
    metrics["macro_f1"] = np.mean(label_f1s) if label_f1s else 0.0
    
    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint Management
# ═══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model, tokenizer, optimizer, scheduler,
    epoch, global_step, metrics,
    save_dir: Path,
):
    """Save a training checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), save_dir / "model.pt")
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir / "tokenizer")
    
    # Save optimizer and scheduler states
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, save_dir / "training_state.pt")
    
    # Save metadata
    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
        "model_name": model.model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"  Checkpoint saved to {save_dir}")


# ═══════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 1: Context Mechanics Training")
    parser.add_argument("--model", type=str, default=PRIMARY_MODEL,
                        help="HuggingFace model name")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (reduce if OOM)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    config = Stage1Config(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
    
    train_stage1(config=config, model_name=args.model, resume_from=args.resume)
