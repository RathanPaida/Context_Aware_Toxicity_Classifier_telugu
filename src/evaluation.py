"""
Phase 5: Contextual Evaluation Protocol.

Runs the *full moderation pipeline* (lexicon gate + model, with the checkpoint's
tuned thresholds) on:
  1. Contrastive context flip — same target, safe vs toxic parent
  2. Caste & regional identity mentions — subgroup false-positive rate
  3. Parent-bleed robustness — polite reply after a toxic parent
  4. Script parity — Romanized vs native Telugu
  5. Latency — measured on this machine
and, most importantly, standard metrics on the held-out *test* split of the real
Stage 3 data (never used in training or model selection).

Slices live in src/resources/eval_slices.json. The old version multiplied 6 pairs
into "300" with "(variation i)" suffixes and the web app showed hard-coded
numbers; results are now measured and written to reports/eval_results.json.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import EVAL_RESULTS_PATH, RELEASE_GATES, RESOURCES_DIR, get_logger, indic_supervised_sources
from src.inference_pipeline import TwoStageModerationPipeline
from src.preprocessing import IndicPreprocessor
from src.training_utils import ToxicityDataset, compute_metrics, make_loader, predict

logger = get_logger("evaluation_suite")


def _gate(name: str, value: float, n: int) -> Dict[str, Any]:
    g = RELEASE_GATES[name]
    passed = value >= g["value"] if g["op"] == ">=" else value <= g["value"]
    return {
        "value": round(value, 2), "unit": g["unit"], "n": n,
        "gate": f"{'≥' if g['op'] == '>=' else '≤'} {g['value']}{g['unit']}",
        "passed": bool(passed),
    }


def run_evaluation_suite(pipeline: Optional[TwoStageModerationPipeline] = None,
                         output_path: Path = EVAL_RESULTS_PATH) -> Dict[str, Any]:
    pipeline = pipeline or TwoStageModerationPipeline()
    if not pipeline.ready:
        raise RuntimeError(f"Cannot evaluate: {pipeline.status}")
    slices = json.loads((RESOURCES_DIR / "eval_slices.json").read_text(encoding="utf-8"))
    flag = lambda text, hist=None, topic=None: pipeline.moderate(text, history_override=hist or [], topic=topic)  # noqa: E731
    results: Dict[str, Any] = {"slices": {}, "details": {}}

    # 1. Contrastive context flip
    rows = []
    for p in slices["contrastive_pairs"]:
        s = flag(p["target"], [p["safe"]["parent"]], p["safe"]["topic"])
        t = flag(p["target"], [p["toxic"]["parent"]], p["toxic"]["topic"])
        rows.append({"target": p["target"], "safe_score": s["toxicity_score"], "toxic_score": t["toxicity_score"],
                     "correct": (not s["is_toxic"]) and t["is_toxic"]})
    results["slices"]["contrastive_flip"] = _gate("contrastive_flip", 100 * np.mean([r["correct"] for r in rows]), len(rows))
    results["details"]["contrastive_flip"] = rows

    # 2. Identity bias (all items are non-toxic)
    rows = []
    for term in slices["identity_terms"]:
        for tpl in slices["identity_templates"]:
            text = tpl.format(term=term)
            r = flag(text)
            rows.append({"term": term, "text": text, "score": r["toxicity_score"], "flagged": r["is_toxic"]})
    results["slices"]["identity_bias"] = _gate("identity_bias", 100 * np.mean([r["flagged"] for r in rows]), len(rows))
    per_term = pd.DataFrame(rows).groupby("term")["flagged"].mean().mul(100).round(1).to_dict()
    results["details"]["identity_bias"] = {"fpr_by_term": per_term, "flagged": [r for r in rows if r["flagged"]]}

    # 3. Parent bleed (replies are non-toxic)
    rows = []
    for parent in slices["parent_bleed"]["toxic_parents"]:
        for reply in slices["parent_bleed"]["polite_replies"]:
            r = flag(reply, [parent])
            rows.append({"parent": parent, "reply": reply, "score": r["toxicity_score"], "allowed": not r["is_toxic"]})
    results["slices"]["parent_bleed"] = _gate("parent_bleed", 100 * np.mean([r["allowed"] for r in rows]), len(rows))
    results["details"]["parent_bleed"] = [r for r in rows if not r["allowed"]]

    # 4. Script parity
    y, pr, pn = [], [], []
    for it in slices["script_parity"]:
        y.append(int(it["is_toxic"]))
        pr.append(int(flag(it["roman"])["is_toxic"]))
        pn.append(int(flag(it["native"])["is_toxic"]))
    f1r, f1n = f1_score(y, pr, zero_division=0), f1_score(y, pn, zero_division=0)
    results["slices"]["script_parity"] = _gate("script_parity", 100 * abs(f1r - f1n), len(y))
    results["details"]["script_parity"] = {"f1_roman": f1r, "f1_native": f1n}

    # 5. Latency of the model path (lexicon fast exits excluded)
    probe = "Repu mee batch lo okkadu kooda hospital lekunda undadu, chusukondi"
    for _ in range(3):
        flag(probe, ["Maa meedha complaint chesina gang repu ground loki vasthundhi"])
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        flag(probe, ["Maa meedha complaint chesina gang repu ground loki vasthundhi"])
        times.append((time.perf_counter() - t0) * 1000)
    results["slices"]["latency_p50"] = _gate("latency_p50", float(np.median(times)), len(times))
    results["details"]["latency_ms"] = {"p50": float(np.median(times)), "p95": float(np.percentile(times, 95)),
                                        "device": str(pipeline.device)}

    # 6. Held-out real test split (the number that actually matters)
    test_frames = []
    for p in indic_supervised_sources():
        d = pd.read_parquet(p)
        test_frames.append(d[d["split"] == "test"])
    if test_frames:
        test_df = pd.concat(test_frames, ignore_index=True)
        ds = ToxicityDataset(test_df, pipeline.tokenizer, pipeline.max_len,
                             IndicPreprocessor(enable_transliteration=False, enable_normalization=True))
        pred = predict(pipeline.model, make_loader(ds, 32, False), pipeline.device)
        m = compute_metrics(pred["probs"], pred["labels"], thresholds=pipeline.thresholds)
        m["n"] = len(test_df)
        by_source = {}
        for src in test_df["source"].unique():
            idx = (test_df["source"] == src).to_numpy()
            by_source[src] = compute_metrics(pred["probs"][idx], pred["labels"][idx], thresholds=pipeline.thresholds)["per_label"]["toxic"]
        m["toxic_by_source"] = by_source
        results["heldout_test"] = m
        logger.info(f"Held-out test (n={len(test_df):,}): macro-F1={m['macro_f1@threshold']:.4f} "
                    f"toxic F1={m['per_label']['toxic'].get('f1@threshold', float('nan')):.4f} "
                    f"AUC={m['per_label']['toxic'].get('roc_auc', float('nan')):.4f}")
    else:
        logger.warning("No processed Stage 3 data — held-out test metrics skipped")

    results["checkpoint"] = pipeline.info()
    results["evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    logger.info("=" * 70)
    for k, v in results["slices"].items():
        logger.info(f"  {k:18s} {v['value']:>8}{v['unit']:<3} (n={v['n']:<4}) gate {v['gate']:<10} "
                    f"{'PASS' if v['passed'] else 'FAIL'}")
    logger.info("  Note: slices 1-4 are tiny hand-written sanity checks; report the held-out test metrics.")
    logger.info("=" * 70)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    logger.info(f"Results written to {output_path}")
    return results


if __name__ == "__main__":
    run_evaluation_suite()
