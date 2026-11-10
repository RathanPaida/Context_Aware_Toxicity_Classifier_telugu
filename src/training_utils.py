"""
Shared training utilities for the supervised stages (1, 3, 4).

Replaces three copy-pasted training loops. Fixes over the old loops:
  * NaN / None context no longer becomes the literal string "nan"
  * AMP only on CUDA (torch.amp), so CPU runs don't crash / warn
  * separate learning rates for the pretrained encoder and the new heads
  * labels a dataset does not annotate (-1) are masked out of loss and metrics
  * per-label decision thresholds are tuned on validation and stored in the checkpoint
  * early stopping, seeding, gradient accumulation
"""

import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import DONE_MARKER, TOXICITY_LABELS, UNKNOWN_LABEL, TrainConfig, get_logger
from src.model import (
    ContextAwareToxicityClassifier,
    _special_ids,
    collate_encodings,
    atomic_torch_save,
    encode_example,
    ensure_disk_space,
    save_model_checkpoint,
    write_json_atomic,
)
from src.preprocessing import IndicPreprocessor

logger = get_logger("training")

HISTORY_SEPARATOR = "\n"  # multi-turn context stored in one string column is split on this


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_str(value: Any) -> str:
    """None / NaN / 'nan' / 'None' → ''."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class ToxicityDataset(Dataset):
    """
    Generic dataset over a DataFrame with columns:
        text (required), context (optional, turns separated by '\\n'),
        topic (optional), and any of TOXICITY_LABELS (missing/NaN → -1 = unknown).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
        preprocessor: Optional[IndicPreprocessor] = None,
        topic_dropout: float = 0.0,
        train: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor
        self.topic_dropout = topic_dropout
        self.train = train

        labels = np.full((len(self.df), len(TOXICITY_LABELS)), UNKNOWN_LABEL, dtype=np.float32)
        for i, col in enumerate(TOXICITY_LABELS):
            if col in self.df.columns:
                vals = pd.to_numeric(self.df[col], errors="coerce").fillna(UNKNOWN_LABEL).to_numpy()
                labels[:, i] = vals
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def _clean(self, s: str) -> str:
        return self.preprocessor.clean_text(s) if (self.preprocessor and s) else s

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        text = self._clean(clean_str(row.get("text")))
        context = clean_str(row.get("context"))
        history = [self._clean(t) for t in context.split(HISTORY_SEPARATOR) if t.strip()] if context else []
        topic = clean_str(row.get("topic"))
        if self.train and topic and self.topic_dropout > 0 and random.random() < self.topic_dropout:
            topic = ""

        enc = encode_example(self.tokenizer, text, history=history, topic=topic, max_length=self.max_length)
        enc["labels"] = self.labels[idx]
        enc["row_index"] = idx
        return enc


def make_loader(dataset: ToxicityDataset, batch_size: int, shuffle: bool) -> DataLoader:
    _, _, pad_id = _special_ids(dataset.tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda b: collate_encodings(b, pad_id),
        pin_memory=torch.cuda.is_available(),
    )


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


@torch.no_grad()
def predict(model, loader: DataLoader, device, fp16: bool = False) -> Dict[str, np.ndarray]:
    model.eval()
    probs, labels, losses = [], [], []
    use_amp = fp16 and device.type == "cuda"
    for batch in loader:
        inputs = {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "target_mask")}
        y = batch["labels"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(**inputs, labels=y)
        losses.append(out["loss"].item())
        probs.append(out["probs"].float().cpu().numpy())
        labels.append(y.cpu().numpy())
    return {
        "probs": np.vstack(probs) if probs else np.zeros((0, len(TOXICITY_LABELS))),
        "labels": np.vstack(labels) if labels else np.zeros((0, len(TOXICITY_LABELS))),
        "loss": float(np.mean(losses)) if losses else float("nan"),
    }


def compute_metrics(probs: np.ndarray, labels: np.ndarray, thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Per-label F1 @0.5, F1 @tuned threshold and ROC-AUC, ignoring unknown (-1) labels.
    Labels with no positive examples in the split are reported but excluded from macro averages.
    """
    metrics: Dict[str, Any] = {"per_label": {}}
    tuned = {}
    f1_default, f1_tuned = [], []
    for i, name in enumerate(TOXICITY_LABELS):
        known = labels[:, i] >= 0
        y, p = labels[known, i].astype(int), probs[known, i]
        entry: Dict[str, Any] = {"n": int(known.sum()), "positives": int(y.sum()) if len(y) else 0}
        if len(y) == 0 or y.sum() == 0:
            entry["note"] = "no positives in split"
            tuned[name] = (thresholds or {}).get(name, 0.5)
            metrics["per_label"][name] = entry
            continue

        entry["f1@0.5"] = float(f1_score(y, p >= 0.5, zero_division=0))
        if thresholds and name in thresholds:
            best_t = thresholds[name]
        else:
            scores = [f1_score(y, p >= t, zero_division=0) for t in THRESHOLD_GRID]
            best_t = float(THRESHOLD_GRID[int(np.argmax(scores))])
        entry["threshold"] = best_t
        entry["f1@threshold"] = float(f1_score(y, p >= best_t, zero_division=0))
        if len(np.unique(y)) == 2:
            entry["roc_auc"] = float(roc_auc_score(y, p))
        tuned[name] = best_t
        f1_default.append(entry["f1@0.5"])
        f1_tuned.append(entry["f1@threshold"])
        metrics["per_label"][name] = entry

    metrics["macro_f1@0.5"] = float(np.mean(f1_default)) if f1_default else 0.0
    metrics["macro_f1@threshold"] = float(np.mean(f1_tuned)) if f1_tuned else 0.0
    metrics["thresholds"] = tuned
    return metrics


def log_metrics(prefix: str, m: Dict[str, Any]) -> None:
    logger.info(f"{prefix} loss={m.get('loss', float('nan')):.4f} | macro-F1@0.5={m['macro_f1@0.5']:.4f} "
                f"| macro-F1@tuned={m['macro_f1@threshold']:.4f}")
    for name, e in m["per_label"].items():
        if "f1@0.5" in e:
            logger.info(f"    {name:16s} n={e['n']:6d} pos={e['positives']:5d} F1@0.5={e['f1@0.5']:.3f} "
                        f"F1@{e['threshold']:.2f}={e['f1@threshold']:.3f} AUC={e.get('roc_auc', float('nan')):.3f}")
        else:
            logger.info(f"    {name:16s} n={e['n']:6d} ({e.get('note', '')})")


# ═══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════

# ── resume / completion markers ─────────────────────────────────────

def last_dir_for(save_dir: Path) -> Path:
    """stage3_best → stage3_last (where the end-of-epoch resume state lives)."""
    save_dir = Path(save_dir)
    return save_dir.parent / (save_dir.name.replace("_best", "") + "_last")


def is_done(save_dir: Path) -> bool:
    return (Path(save_dir) / DONE_MARKER).exists()


def mark_done(save_dir: Path, info: Dict[str, Any]) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    write_json_atomic(Path(save_dir) / DONE_MARKER, {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), **info})


def clear_done(save_dir: Path) -> None:
    (Path(save_dir) / DONE_MARKER).unlink(missing_ok=True)


def save_resume_state(model, optimizer, scheduler, scaler, last_dir: Path, info: Dict[str, Any]) -> None:
    """Trainable weights (fp32) + optimizer/scheduler/scaler/RNG state. Frozen weights are rebuilt on resume."""
    last_dir = Path(last_dir)
    last_dir.mkdir(parents=True, exist_ok=True)
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n: b.detach().cpu() for n, b in model.named_buffers()}
    approx = sum(t.numel() * t.element_size() for t in trainable.values()) * 3  # weights + 2 Adam moments
    ensure_disk_space(last_dir, approx, "resume state")
    atomic_torch_save({
        "trainable": trainable, "buffers": buffers,
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        "info": info,
    }, last_dir / "resume.pt")
    logger.info(f"  resume point saved → {last_dir} (epoch {info.get('completed_epochs')})")


def load_resume_state(model, optimizer, scheduler, scaler, last_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(last_dir) / "resume.pt"
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    result = model.load_state_dict({**state["trainable"], **state["buffers"]}, strict=False)
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    missing = expected & set(result.missing_keys)
    if missing:
        raise RuntimeError(f"Resume state in {last_dir} does not match the model (e.g. {sorted(missing)[:3]}); "
                           "delete that folder to restart the stage.")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    rng = state.get("rng", {})
    try:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception as e:  # different GPU count etc. — not fatal
        logger.warning(f"  could not restore RNG state: {e}")
    logger.info(f"  resumed from {last_dir} after epoch {state['info'].get('completed_epochs')}")
    return state["info"]


def train_classifier(
    model: ContextAwareToxicityClassifier,
    tokenizer,
    train_dataset: ToxicityDataset,
    val_dataset: Optional[ToxicityDataset],
    config: TrainConfig,
    save_dir: Path,
    stage: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
    selection_fn: Optional[Callable[[ContextAwareToxicityClassifier, Dict[str, Any]], float]] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Train `model`, evaluate every epoch (or every `config.eval_steps`), save the best
    checkpoint to `save_dir` together with tuned thresholds and metrics.

    resume=True continues from `<stage>_last/` (written at the end of every epoch) if it exists.
    When the stage finishes, a DONE marker is written into `save_dir` and the resume state is removed.

    `selection_fn(model, val_metrics) -> float` overrides the model-selection score
    (default: tuned macro-F1 on the validation set).
    """
    set_seed(config.seed)
    device = get_device()
    model.to(device)
    use_amp = config.fp16 and device.type == "cuda"
    save_dir, last_dir = Path(save_dir), last_dir_for(save_dir)
    clear_done(save_dir)
    logger.info(f"[{stage}] device={device} amp={use_amp} train={len(train_dataset):,} "
                f"val={len(val_dataset) if val_dataset else 0:,} | batch {config.batch_size}x"
                f"{config.gradient_accumulation_steps} | epochs {config.num_epochs}")

    train_loader = make_loader(train_dataset, config.batch_size, shuffle=True)
    val_loader = make_loader(val_dataset, config.batch_size * 2, shuffle=False) if val_dataset else None

    enc_params = [p for p in model.encoder_parameters() if p.requires_grad]
    head_params = [p for p in model.head_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": config.learning_rate},
            {"params": head_params, "lr": config.head_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / config.gradient_accumulation_steps))
    total_steps = steps_per_epoch * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score, best_metrics, bad_evals, global_step, start_epoch = -float("inf"), {}, 0, 0, 0
    history: List[Dict[str, Any]] = []

    if resume:
        info = load_resume_state(model, optimizer, scheduler, scaler, last_dir)
        if info:
            start_epoch = info["completed_epochs"]
            global_step, bad_evals = info["global_step"], info["bad_evals"]
            best_score, best_metrics, history = info["best_score"], info["best_metrics"], info["history"]
            if info.get("stopped"):
                start_epoch = config.num_epochs  # it had already early-stopped
    elif last_dir.exists():
        shutil.rmtree(last_dir, ignore_errors=True)  # stale state from an older run

    def run_eval() -> bool:
        """Evaluate, maybe save. Returns True if training should stop early."""
        nonlocal best_score, best_metrics, bad_evals
        if val_loader is None:
            return False
        pred = predict(model, val_loader, device, config.fp16)
        m = compute_metrics(pred["probs"], pred["labels"])
        m["loss"] = pred["loss"]
        log_metrics(f"[{stage}] step {global_step} val", m)
        score = selection_fn(model, m) if selection_fn else m["macro_f1@threshold"]
        if not np.isfinite(score):
            logger.warning(f"[{stage}] selection score is {score}; treating as very low")
            score = -1e9
        m["selection_score"] = score
        history.append({"step": global_step, "score": score, "loss": m["loss"]})
        model.train()
        if score > best_score:
            best_score, best_metrics, bad_evals = score, m, 0
            save_model_checkpoint(model, tokenizer, save_dir, {
                "stage": stage,
                "step": global_step,
                "thresholds": m["thresholds"],
                "val_metrics": m,
                "max_seq_length": config.max_seq_length,
                **(extra_metadata or {}),
            })
            logger.info(f"  new best ({score:.4f}) -> {save_dir}")
            return False
        bad_evals += 1
        return config.early_stopping_patience > 0 and bad_evals >= config.early_stopping_patience

    stop = False
    t_start = time.time()
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"{stage} epoch {epoch + 1}/{config.num_epochs}", dynamic_ncols=True,
                    mininterval=5)
        optimizer.zero_grad()
        for i, batch in enumerate(pbar):
            inputs = {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "target_mask")}
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**inputs, labels=labels)["loss"] / config.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % config.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.set_postfix(loss=f"{loss.item() * config.gradient_accumulation_steps:.4f}")
                if config.eval_steps and global_step % config.eval_steps == 0:
                    if run_eval():
                        stop = True
                        break
        if not stop and not config.eval_steps and run_eval():
            stop = True
        elapsed = time.time() - t_start
        done_epochs = epoch + 1 - start_epoch
        remaining = (config.num_epochs - epoch - 1) * elapsed / max(1, done_epochs)
        logger.info(f"[{stage}] epoch {epoch + 1}/{config.num_epochs} done ({elapsed / 60:.1f} min so far)"
                    + ("" if stop else f" | at most ~{remaining / 60:.0f} min left in this stage"))
        save_resume_state(model, optimizer, scheduler, scaler, last_dir, {
            "completed_epochs": epoch + 1, "global_step": global_step, "bad_evals": bad_evals,
            "best_score": best_score, "best_metrics": best_metrics, "history": history, "stopped": stop,
        })
        if stop:
            logger.info(f"[{stage}] early stopping after epoch {epoch + 1}")
            break

    if val_loader is None:  # nothing to select on: save the final weights
        save_model_checkpoint(model, tokenizer, save_dir, {"stage": stage, "step": global_step, **(extra_metadata or {})})

    best_metrics["history"] = history
    mark_done(save_dir, {"stage": stage, "best_score": best_score, "global_step": global_step})
    shutil.rmtree(last_dir, ignore_errors=True)
    return best_metrics
