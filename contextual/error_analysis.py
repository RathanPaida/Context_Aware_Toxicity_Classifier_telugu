"""Inspect validation mistakes, parent confusion, and label consistency.

Compares full-context predictions against isolated target predictions on the
validation split to identify where context hurts (false alarms induced by parents)
versus where context helps.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

from .config import load_config, location, project_path, write_json
from .data import fingerprint, read_jsonl
from .embeddings import TextEncoder, cached_embeddings, device_for
from .inference import Predictor
from .model import ConversationClassifier, collate_examples
from .training import fit_threshold, metrics, predict_examples


def inspect_validation_errors(cfg, checkpoint_path=None, report_path=None):
    ckpt_dir = project_path(checkpoint_path or location(cfg, "checkpoint"))
    predictor = Predictor(ckpt_dir)
    threshold = predictor.threshold
    data_path = location(cfg, "prepared")
    examples = read_jsonl(data_path)
    val_examples = [e for e in examples if e["split"] == "val"]
    print(f"Loaded {len(val_examples)} validation examples from {data_path}", flush=True)

    vectors, lookup = cached_embeddings(cfg, examples, predictor.encoder)
    print("Computing contextual predictions on validation...", flush=True)
    prob_context = predict_examples(predictor.model, val_examples, vectors, lookup, cfg)

    print("Computing isolated (target-only) predictions on validation...", flush=True)
    val_isolated = [{**e, "texts": [e["texts"][-1]]} for e in val_examples]
    prob_isolated = predict_examples(predictor.model, val_isolated, vectors, lookup, cfg)

    # Compute parent isolated scores for contextual examples
    parent_scores = {}
    contextual_indices = [i for i, e in enumerate(val_examples) if len(e["texts"]) > 1]
    parent_examples = []
    for i in contextual_indices:
        ex = val_examples[i]
        # Most immediate parent
        parent_examples.append({"texts": [ex["texts"][-2]], "label": 0})
    if parent_examples:
        prob_parents = predict_examples(predictor.model, parent_examples, vectors, lookup, cfg)
        for idx, p_score in zip(contextual_indices, prob_parents):
            parent_scores[idx] = float(p_score)

    labels = np.asarray([e["label"] for e in val_examples])
    pred_context = (prob_context >= threshold).astype(int)
    pred_isolated = (prob_isolated >= threshold).astype(int)

    # Error categorization
    false_positives = np.where((labels == 0) & (pred_context == 1))[0]
    false_negatives = np.where((labels == 1) & (pred_context == 0))[0]
    true_positives = np.where((labels == 1) & (pred_context == 1))[0]
    true_negatives = np.where((labels == 0) & (pred_context == 0))[0]

    # Context harmed: isolated was correct, but context made it wrong
    context_harmed_fp = np.where((labels == 0) & (pred_isolated == 0) & (pred_context == 1))[0]
    context_harmed_fn = np.where((labels == 1) & (pred_isolated == 1) & (pred_context == 0))[0]
    context_harmed = np.sort(np.concatenate([context_harmed_fp, context_harmed_fn]))

    # Context helped: isolated was wrong, but context made it right
    context_helped_tp = np.where((labels == 1) & (pred_isolated == 0) & (pred_context == 1))[0]
    context_helped_tn = np.where((labels == 0) & (pred_isolated == 1) & (pred_context == 0))[0]
    context_helped = np.sort(np.concatenate([context_helped_tp, context_helped_tn]))

    # Parent-confusion analysis for context_harmed_fp:
    # Check if high parent toxicity is associated with the false alarm on benign reply
    parent_confusion_cases = []
    for idx in context_harmed_fp:
        ex = val_examples[idx]
        p_score = parent_scores.get(idx)
        case = {
            "id": ex["id"],
            "source": ex["source"],
            "target_text": ex["texts"][-1],
            "parent_texts": ex["texts"][:-1],
            "label": int(labels[idx]),
            "context_prob": float(prob_context[idx]),
            "isolated_prob": float(prob_isolated[idx]),
            "parent_prob": p_score,
            "turns": len(ex["texts"]),
            "likely_parent_bleed": bool(p_score is not None and p_score >= threshold),
        }
        parent_confusion_cases.append(case)

    # Label inconsistency audit across the entire dataset:
    # Group examples by target text fingerprint and find contradictions
    text_to_labels = defaultdict(list)
    for ex in examples:
        target = ex["texts"][-1]
        fp = fingerprint(target)
        text_to_labels[fp].append((ex["label"], ex["id"], ex["source"], target))

    inconsistent_groups = []
    for fp, items in text_to_labels.items():
        unique_lbls = {item[0] for item in items}
        if len(unique_lbls) > 1:
            inconsistent_groups.append({
                "fingerprint": fp,
                "text": items[0][3],
                "occurrences": len(items),
                "labels": [item[0] for item in items],
                "sources": [item[2] for item in items],
                "ids": [item[1] for item in items],
            })

    # Summary
    summary = {
        "checkpoint": str(ckpt_dir),
        "threshold": float(threshold),
        "val_total": len(val_examples),
        "val_contextual": len(contextual_indices),
        "metrics_context": metrics(labels, prob_context, threshold),
        "metrics_isolated": metrics(labels, prob_isolated, threshold),
        "errors": {
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "true_positives": len(true_positives),
            "true_negatives": len(true_negatives),
        },
        "context_impact": {
            "context_harmed_total": len(context_harmed),
            "context_harmed_false_alarms": len(context_harmed_fp),
            "context_harmed_missed_toxic": len(context_harmed_fn),
            "context_helped_total": len(context_helped),
            "context_helped_caught_toxic": len(context_helped_tp),
            "context_helped_suppressed_false_alarm": len(context_helped_tn),
            "parent_bleed_suspects": sum(1 for c in parent_confusion_cases if c["likely_parent_bleed"]),
        },
        "label_inconsistency": {
            "inconsistent_target_texts": len(inconsistent_groups),
            "total_conflicting_rows": sum(g["occurrences"] for g in inconsistent_groups),
        },
        "sample_parent_bleed_false_alarms": parent_confusion_cases[:15],
        "sample_context_helped": [
            {
                "id": val_examples[i]["id"],
                "source": val_examples[i]["source"],
                "target_text": val_examples[i]["texts"][-1],
                "parent_texts": val_examples[i]["texts"][:-1],
                "label": int(labels[i]),
                "context_prob": float(prob_context[i]),
                "isolated_prob": float(prob_isolated[i]),
            }
            for i in context_helped[:10]
        ],
        "sample_inconsistent_labels": inconsistent_groups[:10],
    }

    out_path = project_path(report_path or "reports/validation_error_analysis.json")
    write_json(out_path, summary)
    print(f"Validation error analysis written to {out_path}", flush=True)

    # Print summary highlights
    print("\n" + "=" * 60)
    print("VALIDATION ERROR INSPECTION SUMMARY")
    print("=" * 60)
    print(f"Total Validation Examples: {len(val_examples)} (Contextual: {len(contextual_indices)})")
    print(f"Threshold: {threshold:.4f}")
    print(f"Contextual F1: {summary['metrics_context']['f1']:.4f} | Isolated F1: {summary['metrics_isolated']['f1']:.4f}")
    print(f"Contextual AP: {summary['metrics_context']['average_precision']:.4f} | Isolated AP: {summary['metrics_isolated']['average_precision']:.4f}")
    print(f"\nContext-Harmed Examples: {len(context_harmed)} (where adding parents caused a mistake)")
    print(f"  - False alarms created by parent context: {len(context_harmed_fp)}")
    print(f"  - Missed toxic targets caused by parent context: {len(context_harmed_fn)}")
    print(f"  - Parent-bleed suspects (abusive parent + benign reply flagged): {summary['context_impact']['parent_bleed_suspects']}")
    print(f"\nContext-Helped Examples: {len(context_helped)} (where adding parents fixed a mistake)")
    print(f"  - Caught toxic targets with context: {len(context_helped_tp)}")
    print(f"  - Suppressed false alarms with context: {len(context_helped_tn)}")
    print(f"\nLabel Inconsistencies across full dataset:")
    print(f"  - Unique texts with contradictory labels: {len(inconsistent_groups)}")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/contextual_candidate.json", help="Config file")
    parser.add_argument("--checkpoint", help="Override checkpoint directory")
    parser.add_argument("--report", default="reports/validation_error_analysis.json", help="Output report path")
    args = parser.parse_args()
    cfg = load_config(args.config)
    inspect_validation_errors(cfg, args.checkpoint, args.report)


if __name__ == "__main__":
    main()
