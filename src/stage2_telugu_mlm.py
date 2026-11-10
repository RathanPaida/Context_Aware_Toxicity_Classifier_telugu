"""
Stage 2: Telugu MLM Domain Adaptation.

Masked-language-model training (15% masking) of the encoder on native Telugu text
(IndicCorpV2), starting from the Stage 1 encoder. No classification head is trained.

Output: `stage2_best/encoder` (HF format) + `stage2_best/tokenizer`, picked up by Stage 3.
Robustness:
  * memory options (frozen embeddings, gradient checkpointing, accumulation) from the hardware profile
  * early stopping on held-out MLM loss
  * resume point (`stage2_last/`) every eval; `resume=True` continues from it
  * only one checkpoint on disk (the old code wrote ~2.2 GB every 2,000 steps)
"""

import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CHECKPOINT_DIR, LOG_DIR, PRIMARY_MODEL, PROCESSED_DIR, Stage2Config, get_logger
from src.model import atomic_torch_save, ensure_disk_space, write_json_atomic
from src.preprocessing import IndicPreprocessor
from src.training_utils import clear_done, last_dir_for, mark_done, set_seed

logger = get_logger("stage2_mlm")

SAVE_DIR = CHECKPOINT_DIR / "stage2_best"


class TeluguMLMDataset(Dataset):
    """Pre-tokenised Telugu sentences (token ids only); masking happens in the collator."""

    def __init__(self, texts: List[str], tokenizer, max_length: int, preprocessor: Optional[IndicPreprocessor] = None):
        self.items = []
        for text in tqdm(texts, desc="Tokenizing", dynamic_ncols=True, mininterval=5):
            if not isinstance(text, str):
                continue
            if preprocessor:
                text = preprocessor.clean_text(text)
            if len(text.strip()) < 5:
                continue
            enc = tokenizer(text, max_length=max_length, truncation=True, return_special_tokens_mask=True)
            if len(enc["input_ids"]) > 5:
                self.items.append((enc["input_ids"], enc["special_tokens_mask"]))
        logger.info(f"  kept {len(self.items):,} sentences")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ids, special = self.items[idx]
        return {"input_ids": torch.tensor(ids), "special_tokens_mask": torch.tensor(special)}


def load_texts(config: Stage2Config) -> List[str]:
    path = PROCESSED_DIR / "indic_corp_telugu_mlm.parquet"
    if path.exists():
        texts = pd.read_parquet(path)["text"].tolist()
    else:
        from src.data_downloader import download_indic_corp_telugu
        texts = download_indic_corp_telugu(sample_size=config.telugu_sample_size)["text"].tolist()
    if len(texts) > config.telugu_sample_size:
        idx = np.random.RandomState(42).choice(len(texts), config.telugu_sample_size, replace=False)
        texts = [texts[i] for i in idx]
    return texts


@torch.no_grad()
def evaluate_mlm(model, loader, device, use_amp: bool) -> float:
    """Average MLM loss with a fixed masking seed, so evals are comparable."""
    model.eval()
    cpu_state = torch.get_rng_state()
    torch.manual_seed(1234)
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            total += model(**batch).loss.item()
        n += 1
    torch.set_rng_state(cpu_state)
    model.train()
    return total / max(1, n)


def save_encoder(mlm_model, tokenizer, step: int, eval_loss: float, config: Stage2Config) -> None:
    """Write encoder + tokenizer into a temp dir, then swap it in (never leaves a half-written checkpoint)."""
    tmp = SAVE_DIR.parent / (SAVE_DIR.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    n_bytes = sum(p.numel() * 2 for p in mlm_model.base_model.parameters())
    ensure_disk_space(SAVE_DIR.parent, n_bytes, "Stage 2 encoder")
    mlm_model.base_model.save_pretrained(tmp / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(tmp / "tokenizer")
    write_json_atomic(tmp / "metadata.json", {
        "stage": "stage2", "step": step, "loss": eval_loss, "perplexity": float(math.exp(min(eval_loss, 50))),
        "data": "IndicCorpV2 Telugu", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {k: v for k, v in vars(config).items()},
    })
    shutil.rmtree(SAVE_DIR, ignore_errors=True)
    tmp.rename(SAVE_DIR)
    logger.info(f"  saved Telugu-adapted encoder → {SAVE_DIR} (eval loss {eval_loss:.4f})")


def load_stage1_encoder(mlm_model, stage1_checkpoint: Optional[str]) -> None:
    if not stage1_checkpoint or not (Path(stage1_checkpoint) / "model.pt").exists():
        logger.warning("  no Stage 1 checkpoint — MLM starts from the pretrained encoder")
        return
    state = torch.load(Path(stage1_checkpoint) / "model.pt", map_location="cpu", weights_only=True)
    prefix = "encoder."  # strip only the leading prefix ("encoder.encoder.layer.N" → "encoder.layer.N")
    enc = {k[len(prefix):]: v.float() for k, v in state.items() if k.startswith(prefix)}
    result = mlm_model.base_model.load_state_dict(enc, strict=False)
    missing = [k for k in result.missing_keys if "pooler" not in k]
    if missing:
        raise RuntimeError(f"Stage 1 → MLM weight transfer failed; missing {len(missing)} keys e.g. {missing[:3]}")
    logger.info(f"  loaded Stage 1 encoder ({len(enc)} tensors)")


def train_stage2(config: Optional[Stage2Config] = None, model_name: str = PRIMARY_MODEL,
                 stage1_checkpoint: Optional[str] = None, resume: bool = False):
    config = config or Stage2Config()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.fp16 and device.type == "cuda"
    last_dir = last_dir_for(SAVE_DIR)
    clear_done(SAVE_DIR)

    texts = load_texts(config)
    if not texts:
        logger.error("No Telugu text available — Stage 2 skipped (Stage 3 will use the Stage 1 encoder).")
        return None
    logger.info(f"  {len(texts):,} Telugu sentences | device {device} | batch {config.batch_size}x"
                f"{config.gradient_accumulation_steps} | max steps {config.max_steps:,}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)
    load_stage1_encoder(mlm_model, stage1_checkpoint)
    if config.freeze_embeddings:
        for p in mlm_model.get_input_embeddings().parameters():
            p.requires_grad = False
    if config.gradient_checkpointing:
        mlm_model.gradient_checkpointing_enable()
    mlm_model.to(device)

    pre = IndicPreprocessor(enable_transliteration=False, enable_normalization=True)
    split = int(len(texts) * 0.97)
    train_ds = TeluguMLMDataset(texts[:split], tokenizer, config.max_seq_length, pre)
    eval_ds = TeluguMLMDataset(texts[split:][:2000], tokenizer, config.max_seq_length, pre)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=config.mlm_probability)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collator,
                              num_workers=config.num_workers, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=config.batch_size * 2, collate_fn=collator)

    params = [p for p in mlm_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, config.warmup_steps, config.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step, best_loss, bad_evals, history = 0, float("inf"), 0, []
    if resume and (last_dir / "resume.pt").exists():
        st = torch.load(last_dir / "resume.pt", map_location="cpu", weights_only=False)
        mlm_model.load_state_dict(st["trainable"], strict=False)
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        step, best_loss, bad_evals, history = st["step"], st["best_loss"], st["bad_evals"], st["history"]
        logger.info(f"  resumed Stage 2 at step {step}")
    elif last_dir.exists():
        shutil.rmtree(last_dir, ignore_errors=True)

    def save_resume():
        last_dir.mkdir(parents=True, exist_ok=True)
        trainable = {n: p.detach().cpu() for n, p in mlm_model.named_parameters() if p.requires_grad}
        ensure_disk_space(last_dir, sum(t.numel() * 4 for t in trainable.values()) * 3, "Stage 2 resume state")
        atomic_torch_save({"trainable": trainable, "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "step": step,
                           "best_loss": best_loss, "bad_evals": bad_evals, "history": history,
                           "python_rng": random.getstate()}, last_dir / "resume.pt")

    mlm_model.train()
    pbar = tqdm(total=config.max_steps, initial=step, desc="Stage 2 MLM", dynamic_ncols=True, mininterval=5)
    running, micro = 0.0, 0
    stopped = False
    while step < config.max_steps and not stopped:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = mlm_model(**batch).loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            running += loss.item()
            micro += 1
            if micro % config.gradient_accumulation_steps:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            pbar.update(1)
            if step % config.logging_steps == 0:
                pbar.set_postfix(loss=f"{running / config.logging_steps:.3f}")
                running = 0.0
            if step % config.eval_steps == 0 or step >= config.max_steps:
                ev = evaluate_mlm(mlm_model, eval_loader, device, use_amp)
                history.append({"step": step, "eval_loss": ev})
                logger.info(f"  [step {step}] eval loss {ev:.4f} | ppl {math.exp(min(ev, 50)):.1f}")
                if ev < best_loss - 1e-3:
                    best_loss, bad_evals = ev, 0
                    save_encoder(mlm_model, tokenizer, step, ev, config)
                else:
                    bad_evals += 1
                    if config.early_stopping_patience and bad_evals >= config.early_stopping_patience:
                        logger.info(f"  early stopping: no improvement in {bad_evals} evals")
                        stopped = True
                save_resume()
            if step >= config.max_steps or stopped:
                break
    pbar.close()

    (LOG_DIR / "stage2_training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    mark_done(SAVE_DIR, {"stage": "stage2", "best_eval_loss": best_loss, "steps": step})
    shutil.rmtree(last_dir, ignore_errors=True)
    logger.info(f"Stage 2 done: best eval loss {best_loss:.4f} (ppl {math.exp(min(best_loss, 50)):.1f})")
    return best_loss


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stage 2: Telugu MLM Adaptation")
    p.add_argument("--model", default=PRIMARY_MODEL)
    p.add_argument("--max-steps", type=int, default=Stage2Config.max_steps)
    p.add_argument("--batch-size", type=int, default=Stage2Config.batch_size)
    p.add_argument("--stage1-checkpoint", default=str(CHECKPOINT_DIR / "stage1_best"))
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    train_stage2(Stage2Config(max_steps=a.max_steps, batch_size=a.batch_size), a.model, a.stage1_checkpoint, a.resume)
