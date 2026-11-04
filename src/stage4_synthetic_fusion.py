"""
Stage 4: Synthetic Thread Fusion (contrastive twin pairs).

Same target utterance, opposite label depending on the parent context.

Data comes from src/resources/stage4_synthetic.json. Fixes over the old version:
  * no duplicated templates with "(Round N)" / "(Thread N)" suffixes — those suffixes
    were a label leak (safe parents always said "Round", toxic ones "Thread") and the
    duplication made the 80/20 split put copies of training rows in validation;
  * validation is on held-out *templates*, so the flip metric measures generalisation;
  * anything in the evaluation slices (src/resources/eval_slices.json) is excluded;
  * topic dropout so the model cannot decide from the topic string alone;
  * replay of Stage 3 data to limit catastrophic forgetting on real comments.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    CHECKPOINT_DIR, PRIMARY_MODEL, RESOURCES_DIR, TOXICITY_LABELS, Stage4Config, get_logger, indic_supervised_sources,
)
from src.model import ContextAwareToxicityClassifier, find_best_checkpoint, load_model_weights, load_tokenizer
from src.preprocessing import IndicPreprocessor
from src.training_utils import ToxicityDataset, get_device, make_loader, predict, train_classifier

logger = get_logger("stage4_fusion")

STAGE4_DATA = RESOURCES_DIR / "stage4_synthetic.json"
EVAL_SLICES = RESOURCES_DIR / "eval_slices.json"


def _eval_texts() -> set:
    if not EVAL_SLICES.exists():
        return set()
    ev = json.loads(EVAL_SLICES.read_text(encoding="utf-8"))
    texts = {p["target"] for p in ev.get("contrastive_pairs", [])}
    texts |= set(ev.get("parent_bleed", {}).get("polite_replies", []))
    for it in ev.get("script_parity", []):
        texts |= {it["roman"], it["native"]}
    return texts


def build_synthetic_dataset() -> pd.DataFrame:
    """One row per (template, side) + de-escalation rows. Columns follow the unified schema."""
    data = json.loads(STAGE4_DATA.read_text(encoding="utf-8"))
    banned = _eval_texts()
    rows: List[Dict] = []

    for t in data["templates"]:
        if t["target"] in banned:
            logger.warning(f"  skipping template {t['id']} — target is in the evaluation slices")
            continue
        for side in ("safe", "toxic"):
            label = t[side]["label"]
            rows.append({
                "group": t["id"], "twin": side, "topic": t[side]["topic"], "context": t[side]["parent"],
                "text": t["target"], **dict(zip(TOXICITY_LABELS, label)),
            })

    de = data.get("deescalation", {})
    for i, parent in enumerate(de.get("toxic_parents", [])):
        for j, reply in enumerate(de.get("replies", [])):
            if reply in banned:
                continue
            rows.append({
                "group": f"deesc_{j}", "twin": "deescalation", "topic": "", "context": parent,
                "text": reply, **{l: 0 for l in TOXICITY_LABELS},
            })
    df = pd.DataFrame(rows)
    logger.info(f"  Synthetic rows: {len(df):,} ({df['group'].nunique()} groups)")
    return df


def split_by_group(df: pd.DataFrame, val_frac: float, seed: int):
    """Hold out whole groups, stratified so validation has both twin templates and de-escalations."""
    rng = np.random.RandomState(seed)
    val_groups = set()
    is_twin = df["twin"].isin(["safe", "toxic"])
    for subset in (df[is_twin], df[~is_twin]):
        groups = np.array(sorted(subset["group"].unique()))
        if len(groups) < 3:
            continue
        rng.shuffle(groups)
        val_groups |= set(groups[: max(1, int(round(len(groups) * val_frac)))])
    in_val = df["group"].isin(val_groups)
    return df[~in_val].reset_index(drop=True), df[in_val].reset_index(drop=True)


def flip_accuracy(model, tokenizer, df: pd.DataFrame, thresholds: Dict[str, float], max_len: int) -> Dict[str, float]:
    """Fraction of templates where the safe twin is ALLOW and the toxic twin is FLAG."""
    twins = df[df["twin"].isin(["safe", "toxic"])].reset_index(drop=True)
    if twins.empty:
        return {"flip_accuracy": float("nan"), "pairs": 0}
    ds = ToxicityDataset(twins, tokenizer, max_len)
    pred = predict(model, make_loader(ds, 16, False), get_device())
    flagged = pred["probs"][:, 0] >= thresholds.get("toxic", 0.5)
    twins = twins.assign(correct=flagged == (twins["toxic"].to_numpy() == 1))
    ok = twins.groupby("group")["correct"].all()
    return {"flip_accuracy": float(ok.mean()), "pairs": int(len(ok))}


def train_stage4(config: Optional[Stage4Config] = None, model_name: str = PRIMARY_MODEL,
                 base_checkpoint: Optional[str] = None, replay_rows: int = 2000):
    config = config or Stage4Config()
    synth = build_synthetic_dataset()
    train_df, val_df = split_by_group(synth, 0.2, config.seed)
    logger.info(f"  Synthetic train={len(train_df)} val={len(val_df)} (held-out groups: {sorted(val_df['group'].unique())})")

    # Replay real Stage 3 data so the model does not forget non-synthetic comments.
    replay = []
    for p in indic_supervised_sources():
        d = pd.read_parquet(p)
        replay.append(d[d["split"] == "train"])
    if replay and replay_rows:
        r = pd.concat(replay).sample(frac=1, random_state=config.seed).head(replay_rows)
        train_df = pd.concat([train_df, r], ignore_index=True)
        logger.info(f"  + {len(r):,} replayed Stage 3 rows")

    tokenizer = load_tokenizer(model_name)
    model = ContextAwareToxicityClassifier(model_name, focal_alpha=config.focal_alpha, focal_gamma=config.focal_gamma)
    base = Path(base_checkpoint) if base_checkpoint else (CHECKPOINT_DIR / "stage3_best")
    if not (base / "model.pt").exists():
        base = find_best_checkpoint()
    if base is not None and (base / "model.pt").exists():
        load_model_weights(model, base)
    else:
        logger.warning("No base checkpoint found — Stage 4 starts from the pretrained encoder")
    model.configure_training(config.freeze_ratio, config.freeze_embeddings, config.gradient_checkpointing)

    pre = IndicPreprocessor(enable_transliteration=False, enable_normalization=True)
    val_ds = ToxicityDataset(val_df, tokenizer, config.max_seq_length, pre) if len(val_df) else None

    def select(m, metrics):
        if val_ds is None:
            return metrics["macro_f1@threshold"]
        flip = flip_accuracy(m, tokenizer, val_df, metrics["thresholds"], config.max_seq_length)
        metrics.update(flip)
        logger.info(f"  held-out flip accuracy: {flip['flip_accuracy']:.2%} over {flip['pairs']} templates")
        return flip["flip_accuracy"] + 0.1 * metrics["macro_f1@threshold"]

    metrics = train_classifier(
        model, tokenizer,
        ToxicityDataset(train_df, tokenizer, config.max_seq_length, pre, topic_dropout=config.topic_dropout, train=True),
        val_ds, config,
        save_dir=CHECKPOINT_DIR / "stage4_best", stage="stage4",
        extra_metadata={"data": "synthetic contrastive twins + stage3 replay", "base_checkpoint": str(base)},
        selection_fn=select,
    )
    return metrics


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stage 4: Synthetic Thread Fusion")
    p.add_argument("--model", default=PRIMARY_MODEL)
    p.add_argument("--epochs", type=int, default=Stage4Config.num_epochs)
    p.add_argument("--lr", type=float, default=Stage4Config.learning_rate)
    p.add_argument("--batch-size", type=int, default=Stage4Config.batch_size)
    p.add_argument("--base-checkpoint", default=None)
    p.add_argument("--replay-rows", type=int, default=2000)
    a = p.parse_args()
    train_stage4(Stage4Config(learning_rate=a.lr, num_epochs=a.epochs, batch_size=a.batch_size),
                 a.model, a.base_checkpoint, a.replay_rows)
