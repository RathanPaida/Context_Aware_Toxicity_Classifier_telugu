"""
Baselines and context ablation for the report.

    python -m src.baselines            # or: python run_pipeline.py --baselines

1. Lexicon only            — FastTargetFilter: flag if any severe or mild term is present
2. TF-IDF + LogisticReg    — char 2-5-gram TF-IDF, trained on the Stage 3 train split
3. Trained model           — the pipeline's model on the same held-out test split
4. Context ablation        — trained model on the contextual slices with vs. without the parent
                             (if removing context changes nothing, the model is not using it)

All numbers are for the `toxic` label on the held-out test split of the Indic datasets
(DravidianCodeMix + Cultural-Safety-Telugu + data/raw/custom/*). Results → reports/baselines.json
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import REPORTS_DIR, RESOURCES_DIR, get_logger, indic_supervised_sources
from src.inference_pipeline import FastTargetFilter, TwoStageModerationPipeline
from src.preprocessing import IndicPreprocessor

logger = get_logger("baselines")


def _scores(y, pred, prob=None) -> Dict[str, float]:
    out = {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }
    if prob is not None and len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, prob))
    return out


def load_splits():
    frames = [pd.read_parquet(p) for p in indic_supervised_sources()]
    if not frames:
        raise FileNotFoundError("No Indic datasets in data/processed — run `python run_pipeline.py --download-only`")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["toxic"] >= 0]
    pre = IndicPreprocessor(enable_transliteration=False, enable_normalization=True)
    df = df.assign(clean=df["text"].astype(str).map(pre.clean_text))
    return df[df["split"] == "train"], df[df["split"] == "val"], df[df["split"] == "test"]


def run_baselines(with_model: bool = True) -> Dict[str, Any]:
    train, val, test = load_splits()
    y_test = test["toxic"].to_numpy().astype(int)
    logger.info(f"train={len(train):,} val={len(val):,} test={len(test):,} | test toxic={y_test.mean():.1%}")
    results: Dict[str, Any] = {"n_test": int(len(test)), "test_toxic_rate": float(y_test.mean()), "methods": {}}

    # 1. Lexicon only
    lex = FastTargetFilter()
    hits = np.array([bool(a["severe"] or a["mild"]) for a in map(lex.analyze, test["clean"])], dtype=int)
    results["methods"]["lexicon_only"] = _scores(y_test, hits)

    # 2. TF-IDF + logistic regression (threshold tuned on val)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=200_000, sublinear_tf=True)
    X_train = vec.fit_transform(train["clean"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
    clf.fit(X_train, train["toxic"].astype(int))
    p_val = clf.predict_proba(vec.transform(val["clean"]))[:, 1]
    grid = np.arange(0.05, 0.96, 0.05)
    th = float(grid[np.argmax([f1_score(val["toxic"].astype(int), p_val >= t, zero_division=0) for t in grid])])
    p_test = clf.predict_proba(vec.transform(test["clean"]))[:, 1]
    results["methods"]["tfidf_logreg"] = {**_scores(y_test, (p_test >= th).astype(int), p_test), "threshold": th}

    # 3 + 4. Trained model
    if with_model:
        pipe = TwoStageModerationPipeline()
        if not pipe.ready:
            logger.warning(f"Skipping model rows: {pipe.status}")
        else:
            from src.training_utils import ToxicityDataset, make_loader, predict
            ds = ToxicityDataset(test.drop(columns=["clean"]), pipe.tokenizer, pipe.max_len,
                                 IndicPreprocessor(enable_transliteration=False, enable_normalization=True))
            prob = predict(pipe.model, make_loader(ds, 32, False), pipe.device)["probs"][:, 0]
            results["methods"]["trained_model"] = {
                **_scores(y_test, (prob >= pipe.thresholds["toxic"]).astype(int), prob),
                "threshold": pipe.thresholds["toxic"], "checkpoint": pipe.info()["checkpoint"],
            }
            results["context_ablation"] = context_ablation(pipe)

    _print(results)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "baselines.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Written {out}")
    return results


def context_ablation(pipe: TwoStageModerationPipeline) -> Dict[str, Any]:
    """Same targets scored with and without the parent turn."""
    slices = json.loads((RESOURCES_DIR / "eval_slices.json").read_text(encoding="utf-8"))
    score = lambda t, h: pipe.moderate(t, history_override=h)  # noqa: E731

    rows = []
    for p in slices["contrastive_pairs"]:
        no_ctx = score(p["target"], [])
        safe = score(p["target"], [p["safe"]["parent"]])
        tox = score(p["target"], [p["toxic"]["parent"]])
        rows.append({
            "target": p["target"],
            "p_no_context": no_ctx["categories"]["toxic"], "p_safe_parent": safe["categories"]["toxic"],
            "p_toxic_parent": tox["categories"]["toxic"],
            # Without context both twins are the identical input, so a flip is impossible (0%).
            "flip_correct": (not safe["is_toxic"]) and tox["is_toxic"],
        })
    shift = [r["p_toxic_parent"] - r["p_safe_parent"] for r in rows]
    bleed = []
    for parent in slices["parent_bleed"]["toxic_parents"]:
        for reply in slices["parent_bleed"]["polite_replies"]:
            bleed.append(score(reply, [parent])["categories"]["toxic"] - score(reply, [])["categories"]["toxic"])
    return {
        "contrastive_flip_with_context": float(np.mean([r["flip_correct"] for r in rows])),
        "mean_prob_shift_toxic_minus_safe_parent": float(np.mean(shift)),
        "pairs_where_toxic_parent_raises_score": int(sum(s > 0 for s in shift)),
        "n_pairs": len(rows),
        "parent_bleed_mean_prob_increase": float(np.mean(bleed)),
        "rows": rows,
    }


def _print(r: Dict[str, Any]) -> None:
    logger.info("=" * 70)
    logger.info(f"Held-out test, label=toxic, n={r['n_test']:,} (toxic rate {r['test_toxic_rate']:.1%})")
    logger.info(f"  {'method':16s} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6}")
    for name, m in r["methods"].items():
        logger.info(f"  {name:16s} {m['f1']:6.3f} {m['precision']:6.3f} {m['recall']:6.3f} {m.get('roc_auc', float('nan')):6.3f}")
    ab = r.get("context_ablation")
    if ab:
        logger.info(f"Context ablation over {ab['n_pairs']} contrastive pairs:")
        logger.info(f"  flip accuracy with context: {ab['contrastive_flip_with_context']:.0%}  (without context: max 0%)")
        logger.info(f"  mean P(toxic) shift toxic-parent − safe-parent: {ab['mean_prob_shift_toxic_minus_safe_parent']:+.3f} "
                    f"({ab['pairs_where_toxic_parent_raises_score']}/{ab['n_pairs']} pairs in the right direction)")
        logger.info(f"  parent bleed: toxic parent raises polite-reply P(toxic) by {ab['parent_bleed_mean_prob_increase']:+.3f} on average")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_baselines()
