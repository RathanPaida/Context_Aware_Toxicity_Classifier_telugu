"""Target-only neural baseline for controlled architecture comparison.

Uses identical IndicBERT frozen embeddings, identical data splits, identical
seed, and identical training hyperparameters as the contextual candidate, but
passes ONLY the target message vector through an MLP classification head without
any conversation Transformer or context history.
"""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from .config import load_config, location, project_path, write_json
from .data import read_jsonl
from .embeddings import TextEncoder, cached_embeddings, device_for
from .training import fit_threshold, metrics


class TargetOnlyClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=192, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def collate_target_only(examples, vectors, lookup, device):
    indices = [lookup[ex["texts"][-1]] for ex in examples]
    x = vectors[indices].to(device)
    labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.float32, device=device)
    return x, labels


@torch.inference_mode()
def predict_target_only(model, examples, vectors, lookup, batch_size, device):
    model.eval()
    predictions = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        x, _ = collate_target_only(batch, vectors, lookup, device)
        predictions.extend(model(x).sigmoid().cpu().tolist())
    return np.asarray(predictions)


def train_target_only(cfg):
    t = cfg["training"]
    m_cfg = cfg["model"]
    random.seed(t["seed"])
    np.random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    torch.set_num_threads(t["cpu_threads"])
    device = device_for(cfg)

    examples = read_jsonl(location(cfg, "prepared"))
    from .evaluation import validate_splits, file_hash
    if cfg['data'].get('expected_prepared_sha256') and file_hash(location(cfg, 'prepared')) != cfg['data']['expected_prepared_sha256']:
        raise ValueError('Prepared data does not match the frozen experiment SHA-256')
    split = validate_splits(examples)

    encoder = TextEncoder(cfg)
    vectors, lookup = cached_embeddings(cfg, examples, encoder)

    hidden_size = m_cfg.get("hidden_size", 192)
    dropout = m_cfg.get("dropout", 0.1)
    model = TargetOnlyClassifier(vectors.shape[-1], hidden_size=hidden_size, dropout=dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    positives = sum(e["label"] for e in split["train"])
    weight = torch.tensor((len(split["train"]) - positives) / positives, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

    best_score, best_state, stale, history = -float("inf"), None, 0, []
    rng = random.Random(t["seed"])

    print(f"Training Target-Only Neural Baseline (hidden_size={hidden_size}, lr={t['learning_rate']}, epochs={t['epochs']})...", flush=True)

    for epoch in range(t["epochs"]):
        model.train()
        rows = list(split["train"])
        rng.shuffle(rows)
        losses = []
        for start in range(0, len(rows), t["batch_size"]):
            batch = rows[start:start + t["batch_size"]]
            x, y = collate_target_only(batch, vectors, lookup, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))

        val_probs = predict_target_only(model, split["val"], vectors, lookup, t["batch_size"], device)
        val_labels = [e["label"] for e in split["val"]]
        score = float(average_precision_score(val_labels, val_probs))
        entry = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_ap": score}
        history.append(entry)
        print(f"Epoch {epoch+1:2d} | Loss: {entry['loss']:.4f} | Val AP: {score:.4f}", flush=True)

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= t["patience"]:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state)
    val_probs = predict_target_only(model, split["val"], vectors, lookup, t["batch_size"], device)
    val_labels = [e["label"] for e in split["val"]]
    threshold = fit_threshold(val_labels, val_probs)

    checkpoint = location(cfg, "checkpoint")
    checkpoint.mkdir(parents=True, exist_ok=True)
    encoder.save(checkpoint / "encoder")
    torch.save(best_state, checkpoint / "head.pt")

    metadata = {
        "config": cfg,
        "input_size": vectors.shape[-1],
        "threshold": threshold,
        "encoder_revision": encoder.revision,
        "epochs": history,
        "validation_average_precision": best_score,
        "prepared_sha256": hashlib.sha256(location(cfg, "prepared").read_bytes()).hexdigest(),
        "model_type": "target_only_neural_mlp",
        "has_context_training": False,
        "labels": ["toxic"],
    }
    write_json(checkpoint / "metadata.json", metadata)

    # Evaluate on test
    test = split["test"]
    test_probs = predict_target_only(model, test, vectors, lookup, t["batch_size"], device)
    test_labels = [e["label"] for e in test]

    report = {
        "model_type": "target_only_neural_baseline",
        "threshold": threshold,
        "validation_average_precision": best_score,
        "test": metrics(test_labels, test_probs, threshold),
        "by_source": {},
        "notes": "Target-only neural baseline trained on identical splits and frozen embeddings as the contextual model, without conversation history.",
    }

    for source in sorted({e["source"] for e in test}):
        ids = [i for i, e in enumerate(test) if e["source"] == source]
        report["by_source"][source] = metrics(np.asarray(test_labels)[ids], test_probs[ids], threshold)

    context_ids = [i for i, e in enumerate(test) if len(e["texts"]) > 1]
    if context_ids:
        report["contextual_test_subset"] = metrics(
            np.asarray(test_labels)[context_ids], test_probs[context_ids], threshold
        )

    report_path = location(cfg, "report")
    write_json(report_path, report)
    print(f"Target-only neural baseline report written to {report_path}", flush=True)

    print("\n" + "=" * 60)
    print("TARGET-ONLY NEURAL BASELINE RESULTS")
    print("=" * 60)
    print(f"Validation AP: {best_score:.4f} | Optimal Threshold: {threshold:.4f}")
    print(f"Test F1: {report['test']['f1']:.4f} | Test AP: {report['test']['average_precision']:.4f} | Test ROC-AUC: {report['test']['roc_auc']:.4f}")
    if context_ids:
        print(f"On Contextual Test Subset: F1: {report['contextual_test_subset']['f1']:.4f} | AP: {report['contextual_test_subset']['average_precision']:.4f}")
    for src, m in report["by_source"].items():
        print(f"  {src}: F1={m['f1']:.4f}, AP={m['average_precision']:.4f}")
    print("=" * 60)

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/target_only_neural.json", help="Config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_target_only(cfg)


if __name__ == "__main__":
    main()
