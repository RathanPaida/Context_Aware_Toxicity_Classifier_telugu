"""Train on train only; select epoch and threshold on validation; report test once."""
import copy
import hashlib
import random

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score

from .config import location, write_json
from .data import read_jsonl
from .embeddings import TextEncoder, cached_embeddings, device_for
from .model import ConversationClassifier, collate_examples


def fit_threshold(labels, probabilities):
    if set(labels) != {0, 1}:
        raise ValueError("Threshold tuning requires both classes in validation")
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    scores = np.divide(2*precision*recall, precision+recall,
                       out=np.zeros_like(precision), where=(precision+recall) != 0)
    return float(thresholds[np.argmax(scores[:-1])])


def metrics(labels, probabilities, threshold):
    pred = np.asarray(probabilities) >= threshold
    both = len(set(labels)) > 1
    from sklearn.metrics import precision_score, recall_score
    return {"precision": float(precision_score(labels, pred, zero_division=0)),
            "recall": float(recall_score(labels, pred, zero_division=0)), "n": len(labels), "positive_count": int(np.sum(labels)),
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probabilities)) if both else None,
            "average_precision": float(average_precision_score(labels, probabilities)) if both else None,
            "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist()}


@torch.inference_mode()
def predict_examples(model, examples, vectors, lookup, cfg):
    model.eval()
    predictions = []
    batch_size = cfg["training"]["batch_size"]
    device = next(model.parameters()).device
    for start in range(0, len(examples), batch_size):
        x, valid, _ = collate_examples(examples[start:start+batch_size], vectors, lookup, device)
        predictions.extend(model(x, valid)["logits"].sigmoid().cpu().tolist())
    return np.asarray(predictions)


def train(cfg):
    t = cfg["training"]
    random.seed(t["seed"])
    np.random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    torch.set_num_threads(t["cpu_threads"])
    context_dropout = t.get("context_dropout", 0.0)
    if not 0 <= context_dropout <= 1:
        raise ValueError("context_dropout must be between zero and one")
    examples = read_jsonl(location(cfg, "prepared"))
    from .evaluation import validate_splits, file_hash
    if cfg['data'].get('expected_prepared_sha256') and file_hash(location(cfg, 'prepared')) != cfg['data']['expected_prepared_sha256']:
        raise ValueError('Prepared data does not match the frozen experiment SHA-256')
    split = validate_splits(examples)
    for name, rows in split.items():
        if {e["label"] for e in rows} != {0, 1}:
            raise ValueError(f"{name} requires both binary classes")
    if cfg["data"]["require_real_context"] and not any(len(e["texts"]) > 1 for e in split["train"]):
        raise ValueError("Training split contains no contextual examples")
    encoder = TextEncoder(cfg)
    print(f"Embedding {len(examples)} examples with {encoder.cfg['encoder']['name']}", flush=True)
    vectors, lookup = cached_embeddings(cfg, examples, encoder)
    model = ConversationClassifier(vectors.shape[-1], cfg).to(device_for(cfg))
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    positives = sum(e["label"] for e in split["train"])
    weight = torch.tensor((len(split["train"])-positives)/positives, device=device_for(cfg))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    best_score, best_state, stale, history = -float("inf"), None, 0, []
    rng = random.Random(t["seed"])
    for epoch in range(t["epochs"]):
        model.train()
        rows = list(split["train"])
        rng.shuffle(rows)
        losses = []
        for start in range(0, len(rows), t["batch_size"]):
            batch = rows[start:start+t["batch_size"]]
            # Drop whole histories, retaining the true target and its original label.
            batch = [{**ex, "texts": [ex["texts"][-1]]} if rng.random() < context_dropout else ex
                     for ex in batch]
            x, valid, y = collate_examples(batch, vectors, lookup, device_for(cfg))
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, valid)["logits"], y)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
        val = predict_examples(model, split["val"], vectors, lookup, cfg)
        score = float(average_precision_score([e["label"] for e in split["val"]], val))
        entry = {"epoch": epoch+1, "loss": float(np.mean(losses)), "validation_ap": score}
        history.append(entry)
        print(entry, flush=True)
        if score > best_score:
            best_score, best_state, stale = score, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}), 0
        else:
            stale += 1
            if stale >= t["patience"]:
                break
    model.load_state_dict(best_state)
    validation = predict_examples(model, split["val"], vectors, lookup, cfg)
    threshold = fit_threshold([e["label"] for e in split["val"]], validation)
    checkpoint = location(cfg, "checkpoint")
    checkpoint.mkdir(parents=True, exist_ok=True)
    # Bundle encoder + tokenizer for reproducible offline inference, no cwd/cache dependency.
    encoder.save(checkpoint / "encoder")
    torch.save(best_state, checkpoint / "head.pt")
    metadata = {"config": cfg, "input_size": vectors.shape[-1], "threshold": threshold,
                "encoder_revision": encoder.revision, "epochs": history, "validation_average_precision": best_score,
                "prepared_sha256": hashlib.sha256(location(cfg, "prepared").read_bytes()).hexdigest(),
                "has_context_training": any(len(e["texts"]) > 1 for e in split["train"]),
                "labels": ["toxic"], "task": "target toxicity; conversation embedding is not a separate conversation label"}
    write_json(checkpoint / "metadata.json", metadata)
    return evaluate(cfg)


def evaluate(cfg):
    from .inference import Predictor
    predictor = Predictor(location(cfg, "checkpoint"))
    saved_cfg = predictor.cfg
    data_path = location(cfg, "prepared")
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != predictor.metadata["prepared_sha256"]:
        raise ValueError("Prepared data changed since training; refusing an incomparable evaluation")
    examples = read_jsonl(data_path)
    test = [e for e in examples if e["split"] == "test"]
    vectors, lookup = cached_embeddings(cfg, examples, predictor.encoder)
    probabilities = predict_examples(predictor.model, test, vectors, lookup, saved_cfg)
    isolated = [{**e, "texts": [e["texts"][-1]]} for e in test]
    ablated = predict_examples(predictor.model, isolated, vectors, lookup, saved_cfg)
    labels = [e["label"] for e in test]
    report = {"threshold": predictor.threshold, "test": metrics(labels, probabilities, predictor.threshold),
              "context_removed_ablation": metrics(labels, ablated, predictor.threshold),
              "mean_absolute_probability_change": float(np.abs(probabilities-ablated).mean()),
              "has_context_training": predictor.metadata["has_context_training"], "by_source": {}}
    for source in sorted({e["source"] for e in test}):
        ids = [i for i, e in enumerate(test) if e["source"] == source]
        report["by_source"][source] = metrics(np.asarray(labels)[ids], probabilities[ids], predictor.threshold)
    context_ids = [i for i, e in enumerate(test) if len(e["texts"]) > 1]
    report["contextual_test"] = metrics(np.asarray(labels)[context_ids], probabilities[context_ids], predictor.threshold) if context_ids else None
    report["contextual_test_without_parents"] = metrics(np.asarray(labels)[context_ids], ablated[context_ids], predictor.threshold) if context_ids else None
    report["validation_average_precision"] = predictor.metadata.get("validation_average_precision")
    report['prepared_sha256'] = predictor.metadata['prepared_sha256']
    report['config'] = cfg
    report["notes"] = "Ablation removes context from the same model; it is not an independently trained baseline. Scores are not calibrated probabilities. Attention is not a causal explanation."
    write_json(location(cfg, "report"), report)
    from .evaluation import prediction_rows
    write_json(location(cfg, 'report').with_suffix('.predictions.json'), prediction_rows(test, probabilities, ablated))
    return report
