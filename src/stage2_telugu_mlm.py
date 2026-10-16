"""
Stage 2: Telugu MLM Domain Adaptation.

Performs unsupervised Masked Language Modeling (15% token masking) on the 
encoder body using native Telugu text from IndicCorp. This re-aligns 
subword embeddings to colloquial Telugu syntax and transliterated output.

Key Points:
  - Classification head is NOT trained here (unsupervised)
  - All encoder layers are fine-tuned with MLM objective
  - Teaches the model Telugu-specific vocabulary and syntax
  - 250K Telugu sentences sampled from IndicCorp
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    Stage2Config,
    PROCESSED_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    PRIMARY_MODEL,
    FALLBACK_MODEL,
    MAX_SEQ_LENGTH,
    get_logger,
)
from src.preprocessing import IndicPreprocessor

logger = get_logger("stage2_mlm")


# ═══════════════════════════════════════════════════════════════════════
# Telugu MLM Dataset
# ═══════════════════════════════════════════════════════════════════════

class TeluguMLMDataset(Dataset):
    """
    Dataset for Telugu Masked Language Modeling.
    
    Tokenizes Telugu sentences and prepares them for MLM training.
    The DataCollatorForLanguageModeling handles random masking during
    batch collation (not at dataset creation time).
    """
    
    def __init__(
        self,
        texts: list,
        tokenizer,
        max_length: int = MAX_SEQ_LENGTH,
        preprocessor: Optional[IndicPreprocessor] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor
        
        # Pre-tokenize and filter
        logger.info(f"Pre-tokenizing {len(texts):,} Telugu sentences...")
        self.encodings = []
        
        for text in tqdm(texts, desc="Tokenizing", dynamic_ncols=True):
            if not text or not isinstance(text, str):
                continue
            
            # Apply preprocessing (Unicode normalization primarily)
            if self.preprocessor:
                text = self.preprocessor.clean_text(text)
            
            if len(text.strip()) < 5:
                continue
            
            encoding = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding=False,  # Padding handled by DataCollator
                return_special_tokens_mask=True,
            )
            
            # Only keep if we have meaningful tokens
            if len(encoding["input_ids"]) > 5:
                self.encodings.append(encoding)
        
        logger.info(f"  Kept {len(self.encodings):,} valid tokenized sentences")
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.encodings[idx].items()}


# ═══════════════════════════════════════════════════════════════════════
# Stage 2 Training
# ═══════════════════════════════════════════════════════════════════════

def train_stage2(
    config: Optional[Stage2Config] = None,
    model_name: str = PRIMARY_MODEL,
    stage1_checkpoint: Optional[str] = None,
):
    """
    Execute Stage 2 training: Telugu MLM Domain Adaptation.
    
    This stage applies Masked Language Modeling on Telugu text to:
    1. Re-align subword embeddings to Telugu morphology
    2. Improve tokenization fertility for Dravidian text
    3. Prepare the encoder for Stage 3 Telugu-specific classification
    
    Args:
        config: Stage2Config hyperparameters
        model_name: HuggingFace model identifier
        stage1_checkpoint: Path to Stage 1 checkpoint (for encoder weights)
    """
    if config is None:
        config = Stage2Config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"  VRAM: {mem_gb:.1f} GB")
    
    # ─── Load Telugu Text Data ───
    logger.info("Loading Telugu MLM data...")
    
    mlm_data_path = PROCESSED_DIR / "indic_corp_telugu_mlm.parquet"
    
    if mlm_data_path.exists():
        df = pd.read_parquet(mlm_data_path)
        texts = df["text"].tolist()
    else:
        logger.warning(f"Telugu MLM data not found at {mlm_data_path}")
        logger.info("Attempting to download Telugu data...")
        
        from src.data_downloader import download_indic_corp_telugu
        df = download_indic_corp_telugu(sample_size=config.telugu_sample_size)
        texts = df["text"].tolist()
    
    if not texts:
        logger.error("No Telugu text data available for MLM training!")
        return
    
    logger.info(f"  Loaded {len(texts):,} Telugu sentences")
    
    # Sample if needed
    if len(texts) > config.telugu_sample_size:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(texts), config.telugu_sample_size, replace=False)
        texts = [texts[i] for i in indices]
        logger.info(f"  Sampled to {len(texts):,} sentences")
    
    # ─── Initialize Components ───
    logger.info(f"Loading MLM model: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)
        logger.info(f"  ✓ Loaded {model_name} for MLM")
    except Exception as e:
        logger.warning(f"  Failed to load {model_name}: {e}")
        logger.info(f"  Falling back to {FALLBACK_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        mlm_model = AutoModelForMaskedLM.from_pretrained(FALLBACK_MODEL)
        model_name = FALLBACK_MODEL
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    
    # Load Stage 1 encoder weights if available
    if stage1_checkpoint:
        stage1_path = Path(stage1_checkpoint) / "model.pt"
        if stage1_path.exists():
            logger.info(f"  Loading Stage 1 encoder weights from {stage1_path}")
            state_dict = torch.load(stage1_path, map_location="cpu")
            
            # Extract only encoder weights (skip classification head)
            encoder_state = {
                k.replace("encoder.", ""): v 
                for k, v in state_dict.items() 
                if k.startswith("encoder.")
            }
            
            mlm_model.base_model.load_state_dict(encoder_state, strict=False)
            logger.info("  ✓ Stage 1 encoder weights loaded")
    
    mlm_model.to(device)
    
    # Initialize preprocessor for Telugu normalization
    preprocessor = IndicPreprocessor(
        enable_transliteration=False,  # Telugu text already in native script
        enable_normalization=True,     # Unicode canonicalization important
    )
    
    # ─── Dataset & DataLoader ───
    # Split: 95% train, 5% eval
    split_idx = int(len(texts) * 0.95)
    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]
    
    train_dataset = TeluguMLMDataset(
        train_texts, tokenizer, config.max_seq_length, preprocessor
    )
    eval_dataset = TeluguMLMDataset(
        eval_texts, tokenizer, config.max_seq_length, preprocessor
    )
    
    # MLM Data Collator (handles random masking)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.mlm_probability,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True,
    )
    
    # ─── Optimizer & Scheduler ───
    optimizer = torch.optim.AdamW(
        mlm_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )
    
    scaler = GradScaler() if config.fp16 else None
    
    logger.info(f"  Train samples: {len(train_dataset):,}")
    logger.info(f"  Eval samples:  {len(eval_dataset):,}")
    logger.info(f"  Max steps: {config.max_steps:,}")
    logger.info(f"  MLM probability: {config.mlm_probability}")
    logger.info(f"  LR: {config.learning_rate}")
    
    # ─── Training Loop ───
    logger.info("\n" + "=" * 70)
    logger.info("STARTING STAGE 2 TRAINING: TELUGU MLM ADAPTATION")
    logger.info("=" * 70)
    
    mlm_model.train()
    global_step = 0
    best_eval_loss = float("inf")
    training_log = []
    running_loss = 0.0
    log_steps = 0
    
    # Create infinite data iterator
    def infinite_loader():
        while True:
            for batch in train_loader:
                yield batch
    
    data_iter = iter(infinite_loader())
    pbar = tqdm(total=config.max_steps, desc="MLM Training", dynamic_ncols=True)
    
    while global_step < config.max_steps:
        batch = next(data_iter)
        
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        if config.fp16 and scaler is not None:
            with autocast():
                outputs = mlm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(mlm_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = mlm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlm_model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        global_step += 1
        running_loss += loss.item()
        log_steps += 1
        
        pbar.update(1)
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg": f"{running_loss / log_steps:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })
        
        # Logging
        if global_step % config.logging_steps == 0:
            avg_loss = running_loss / log_steps
            log_entry = {
                "step": global_step,
                "loss": loss.item(),
                "avg_loss": avg_loss,
                "lr": scheduler.get_last_lr()[0],
                "perplexity": np.exp(min(avg_loss, 100)),
            }
            training_log.append(log_entry)
            logger.info(
                f"  Step {global_step}/{config.max_steps} | "
                f"Loss={avg_loss:.4f} | PPL={log_entry['perplexity']:.1f} | "
                f"LR={scheduler.get_last_lr()[0]:.2e}"
            )
            running_loss = 0.0
            log_steps = 0
        
        # Evaluation
        if global_step % config.eval_steps == 0:
            eval_loss = evaluate_mlm(mlm_model, eval_loader, device)
            eval_ppl = np.exp(min(eval_loss, 100))
            
            logger.info(
                f"  [Eval @ Step {global_step}] "
                f"Loss={eval_loss:.4f} | PPL={eval_ppl:.1f}"
            )
            
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_mlm_checkpoint(
                    mlm_model, tokenizer, optimizer, scheduler,
                    global_step, eval_loss,
                    CHECKPOINT_DIR / "stage2_best",
                )
                logger.info(f"  ✓ New best MLM model (Loss={eval_loss:.4f})")
            
            mlm_model.train()
        
        # Periodic save
        if global_step % config.save_steps == 0:
            save_mlm_checkpoint(
                mlm_model, tokenizer, optimizer, scheduler,
                global_step, loss.item(),
                CHECKPOINT_DIR / f"stage2_step_{global_step}",
            )
    
    pbar.close()
    
    # Final save
    save_mlm_checkpoint(
        mlm_model, tokenizer, optimizer, scheduler,
        global_step, running_loss / max(log_steps, 1),
        CHECKPOINT_DIR / "stage2_final",
    )
    
    # Save training log
    log_path = LOG_DIR / "stage2_training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)
    
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2 TRAINING COMPLETE: TELUGU MLM ADAPTATION")
    logger.info(f"  Best Eval Loss: {best_eval_loss:.4f}")
    logger.info(f"  Best Eval PPL:  {np.exp(min(best_eval_loss, 100)):.1f}")
    logger.info(f"  Checkpoint: {CHECKPOINT_DIR / 'stage2_best'}")
    logger.info(f"  Training log: {log_path}")
    logger.info("=" * 70)
    
    return mlm_model, tokenizer


# ═══════════════════════════════════════════════════════════════════════
# MLM Evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_mlm(model, eval_loader, device) -> float:
    """Evaluate MLM model and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        total_loss += outputs.loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint Management
# ═══════════════════════════════════════════════════════════════════════

def save_mlm_checkpoint(model, tokenizer, optimizer, scheduler, step, loss, save_dir):
    """Save MLM training checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full model (HuggingFace format for easy loading)
    model.save_pretrained(save_dir / "model")
    tokenizer.save_pretrained(save_dir / "tokenizer")
    
    # Save training state
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }, save_dir / "training_state.pt")
    
    # Save metadata
    metadata = {
        "step": step,
        "loss": loss,
        "perplexity": float(np.exp(min(loss, 100))),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"  MLM checkpoint saved to {save_dir}")


# ═══════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 2: Telugu MLM Adaptation")
    parser.add_argument("--model", type=str, default=PRIMARY_MODEL,
                        help="HuggingFace model name")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--sample-size", type=int, default=250000,
                        help="Number of Telugu sentences to sample")
    parser.add_argument("--stage1-checkpoint", type=str, default=None,
                        help="Path to Stage 1 checkpoint for encoder initialization")
    args = parser.parse_args()
    
    config = Stage2Config(
        learning_rate=args.lr,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        telugu_sample_size=args.sample_size,
    )
    
    train_stage2(
        config=config,
        model_name=args.model,
        stage1_checkpoint=args.stage1_checkpoint,
    )
