"""Fine-tune LoRA model and evaluate ensembles on reporting vs insult contrasting dataset.

Preserves split integrity, tracks validation AP and contrastive accuracy,
and tests on the untouched test split.
"""
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline

from contextual.adapter_finetune import (
    forward_examples,
    get_adapted_encoder,
    predict_online,
    restore_trainable,
    trainable_state
)
from contextual.config import load_config, location, project_path, write_json
from contextual.data import read_jsonl
from contextual.embeddings import device_for
from contextual.evaluation import validate_splits
from contextual.model import ConversationClassifier
from contextual.preprocessing import normalize_text
from contextual.training import fit_threshold, metrics


def main():
    print("=" * 80)
    print("STEP 4 & 5: CONTRASTIVE FINE-TUNING & ENSEMBLE RE-EVALUATION")
    print("=" * 80)

    # 1. Load configs & datasets
    cfg_base = load_config('configs/contextual_clean_lora.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    contrast_rows = read_jsonl('data/processed/reporting_contrast_v1.jsonl')
    contrast_splits = validate_splits(contrast_rows)
    print(f"Contrast dataset counts: train={len(contrast_splits['train'])}, val={len(contrast_splits['val'])}, test={len(contrast_splits['test'])}")

    base_rows = read_jsonl('data/processed/contextual_clean_v1.jsonl')
    base_splits = validate_splits(base_rows)
    print(f"Base dataset counts: train={len(base_splits['train'])}, val={len(base_splits['val'])}, test={len(base_splits['test'])}")

    # 2. Build training pool
    rng = random.Random(42)
    # Sample 2000 balanced background examples from base train
    pos_base = [r for r in base_splits['train'] if r['label'] == 1]
    neg_base = [r for r in base_splits['train'] if r['label'] == 0]
    bg_train = rng.sample(pos_base, 1000) + rng.sample(neg_base, 1000)
    
    # Oversample contrastive train examples 10x so model gets strong gradient signal on reporting
    contrast_train = contrast_splits['train'] * 10
    train_pool = bg_train + contrast_train
    rng.shuffle(train_pool)
    print(f"Mixed Training Pool: {len(train_pool)} examples ({len(bg_train)} background + {len(contrast_train)} contrastive)")

    # 3. Build validation & test pools
    # Validation pool includes all 24 contrastive val examples + 500 background val examples
    bg_val = rng.sample(base_splits['val'], 500)
    val_pool = contrast_splits['val'] + bg_val
    
    test_contrast = contrast_splits['test']

    # 4. Load base LoRA encoder & head from existing checkpoint
    print("\nLoading base checkpoint weights...")
    encoder, tokenizer = get_adapted_encoder(cfg_base, device)
    head = ConversationClassifier(encoder.config.hidden_size, cfg_base).to(device)

    ckpt = torch.load('models/checkpoints/contextual_clean_lora/resume.pt', map_location='cpu', weights_only=False)
    restore_trainable(encoder, ckpt['adapter'])
    head.load_state_dict(ckpt['head'])
    print("Restored trainable adapter and classification head!")

    # 5. Measure pre-finetuning performance on untouched test contrast cases
    print("\n--- Baseline Performance on Untouched Test Contrast Cases (Before Fine-Tuning) ---")
    pre_test_preds = predict_online(encoder, tokenizer, head, test_contrast, cfg_base, device)
    
    def evaluate_contrast_cases(examples, scores, threshold=0.5):
        direct = [s for e, s in zip(examples, scores) if e['category'] == 'direct_attack']
        report = [s for e, s in zip(examples, scores) if e['category'] == 'reporting_insult']
        counter = [s for e, s in zip(examples, scores) if e['category'] == 'counterspeech']
        endorse = [s for e, s in zip(examples, scores) if e['category'] == 'endorsed_insult']
        
        # Direct attack: toxic (intended 1) -> recall
        rec_direct = sum(s >= threshold for s in direct) / len(direct) if direct else 0.0
        # Reporting: non-toxic (intended 0) -> accuracy (1 - false positive rate)
        acc_report = sum(s < threshold for s in report) / len(report) if report else 0.0
        # Counterspeech: non-toxic (intended 0) -> accuracy
        acc_counter = sum(s < threshold for s in counter) / len(counter) if counter else 0.0
        # Endorsed: toxic (intended 1) -> recall
        rec_endorse = sum(s >= threshold for s in endorse) / len(endorse) if endorse else 0.0

        return {
            'direct_attack_recall': rec_direct,
            'reporting_accuracy': acc_report,
            'counterspeech_accuracy': acc_counter,
            'endorsed_attack_recall': rec_endorse,
            'mean_direct_score': float(np.mean(direct)) if direct else 0.0,
            'mean_report_score': float(np.mean(report)) if report else 0.0,
            'mean_counter_score': float(np.mean(counter)) if counter else 0.0,
            'mean_endorse_score': float(np.mean(endorse)) if endorse else 0.0,
        }

    pre_eval = evaluate_contrast_cases(test_contrast, pre_test_preds)
    for k, v in pre_eval.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 6. Fine-tuning loop
    print("\nStarting targeted contrastive adaptation...")
    positives = sum(e['label'] for e in train_pool)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train_pool)-positives)/positives, device=device))
    
    # Targeted low learning rate for gentle boundary adjustment
    optimizer = torch.optim.AdamW([
        {'params': [p for p in encoder.parameters() if p.requires_grad], 'lr': 1.5e-5},
        {'params': head.parameters(), 'lr': 1e-4}
    ], weight_decay=0.01)
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    batch_size = 8
    accum = 4
    epochs = 3
    steps = (len(train_pool) + batch_size - 1) // batch_size

    best_val_score = -float('inf')
    best_state = None

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        encoder.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        rows = list(train_pool)
        rng.shuffle(rows)

        for step in range(0, len(rows), batch_size):
            batch = rows[step:step+batch_size]
            with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                logits = forward_examples(encoder, tokenizer, head, batch, cfg_base, device)
                labels = torch.tensor([e['label'] for e in batch], dtype=torch.float32, device=device)
                loss = criterion(logits, labels) / accum
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * accum)

            if (step // batch_size + 1) % accum == 0 or (step + batch_size) >= len(rows):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        # Validation
        encoder.eval()
        head.eval()
        val_preds = predict_online(encoder, tokenizer, head, val_pool, cfg_base, device)
        val_labels = [e['label'] for e in val_pool]
        val_ap = float(average_precision_score(val_labels, val_preds))
        
        # Contrastive val performance
        val_contrast = [e for e in val_pool if e['source'] == 'reporting_contrast_v1']
        val_contrast_preds = val_preds[:len(val_contrast)]
        c_eval = evaluate_contrast_cases(val_contrast, val_contrast_preds)

        # Objective balances general validation AP and reporting discrimination
        composite_score = val_ap + 0.5 * (c_eval['reporting_accuracy'] + c_eval['direct_attack_recall'])

        elapsed = time.perf_counter() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) - Loss: {np.mean(losses):.4f} | Val AP: {val_ap:.4f} | "
              f"Val Rep Acc: {c_eval['reporting_accuracy']:.2f} | Val Direct Rec: {c_eval['direct_attack_recall']:.2f} | "
              f"Rep Mean Score: {c_eval['mean_report_score']:.4f}")

        if composite_score > best_val_score:
            best_val_score = composite_score
            best_state = {
                'adapter': trainable_state(encoder),
                'head': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                'epoch': epoch + 1,
                'val_ap': val_ap,
                'val_eval': c_eval
            }

    # 7. Restore best adapted weights
    print("\nRestoring best adapted model...")
    restore_trainable(encoder, best_state['adapter'])
    head.load_state_dict(best_state['head'])
    encoder.eval()
    head.eval()

    # Save adapted checkpoint
    out_dir = Path('models/checkpoints/contextual_contrast_lora')
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state['head'], out_dir / 'head.pt')
    torch.save(best_state, out_dir / 'contrast_adapted_state.pt')
    print(f"Saved adapted checkpoint to {out_dir}")

    # 8. Re-evaluate neural model on untouched test contrast cases
    print("\n" + "=" * 80)
    print("POST-ADAPTATION TEST EVALUATION (UNTOUCHED TEST CASES)")
    print("=" * 80)
    post_test_preds = predict_online(encoder, tokenizer, head, test_contrast, cfg_base, device)
    post_eval = evaluate_contrast_cases(test_contrast, post_test_preds)

    print("\n--- Contrastive Performance Comparison on Untouched Test Set ---")
    print(f"{'Metric':<32} | {'Before Fine-Tuning':<20} | {'After Fine-Tuning':<20} | {'Change':<10}")
    print("-" * 90)
    for k in ['reporting_accuracy', 'counterspeech_accuracy', 'direct_attack_recall', 'endorsed_attack_recall',
              'mean_report_score', 'mean_counter_score', 'mean_direct_score', 'mean_endorse_score']:
        b = pre_eval[k]
        a = post_eval[k]
        diff = a - b
        print(f"{k:<32} | {b:<20.4f} | {a:<20.4f} | {diff:+10.4f}")

    # Inspect per-example results on test cases
    print("\n--- Individual Test Case Predictions ---")
    for e, score in zip(test_contrast, post_test_preds):
        cat = e['category']
        lang = e['language']
        tgt = e['texts'][-1]
        parent = e['texts'][0]
        lbl = e['label']
        is_tox = bool(score >= 0.5)
        status = "CORRECT" if is_tox == (lbl == 1) else "WRONG"
        print(f"[{lang.upper()} - {cat}] Status: {status} (Score: {score:.4f}, Label: {lbl})")
        print(f"  Target: \"{tgt}\"")

    # 9. Test lexical models (TF-IDF) & Ensemble behavior
    print("\n" + "=" * 80)
    print("LEXICAL BASELINE & ENSEMBLE EVALUATION ON CONTRASTIVE DATA")
    print("=" * 80)

    # Fit word and char TF-IDF on the augmented training set
    train_texts = [e['texts'][-1] for e in train_pool]
    train_labels = [e['label'] for e in train_pool]

    word_pipe = make_pipeline(
        TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2), max_features=50000, min_df=2),
        LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    )
    word_pipe.fit(train_texts, train_labels)

    char_pipe = make_pipeline(
        TfidfVectorizer(preprocessor=normalize_text, analyzer='char_wb', ngram_range=(2, 5), max_features=50000, min_df=2),
        LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    )
    char_pipe.fit(train_texts, train_labels)

    test_targets = [e['texts'][-1] for e in test_contrast]
    word_test_scores = word_pipe.predict_proba(test_targets)[:, 1]
    char_test_scores = char_pipe.predict_proba(test_targets)[:, 1]

    # Evaluate ensemble mixtures: soft voting between lexical and adapted neural
    # Test weights: (char, word, neural)
    weight_candidates = [
        (0.0, 0.0, 1.0),      # Pure Neural
        (0.2, 0.0, 0.8),      # High Neural blend
        (0.33, 0.33, 0.34),   # Equal blend
        (0.5, 0.0, 0.5),      # Char + Neural
    ]

    print("\n--- Testing Ensemble Combinations on Test Contrast Cases ---")
    for wc, ww, wn in weight_candidates:
        blend_scores = wc * char_test_scores + ww * word_test_scores + wn * post_test_preds
        b_eval = evaluate_contrast_cases(test_contrast, blend_scores)
        print(f"Weights (char={wc:.2f}, word={ww:.2f}, neural={wn:.2f}):")
        print(f"  Reporting Acc: {b_eval['reporting_accuracy']:.2f} | Direct Attack Rec: {b_eval['direct_attack_recall']:.2f} | "
              f"Mean Rep Score: {b_eval['mean_report_score']:.4f} | Mean Direct Score: {b_eval['mean_direct_score']:.4f}")

    # 10. Save results artifact
    report_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pre_finetuning_test_eval': pre_eval,
        'post_finetuning_test_eval': post_eval,
        'test_predictions': [
            {
                'id': e['id'],
                'group': e['group'],
                'category': e['category'],
                'language': e['language'],
                'parent': e['texts'][0],
                'target': e['texts'][-1],
                'label': e['label'],
                'neural_score_before': float(pre_s),
                'neural_score_after': float(post_s),
                'char_tfidf_score': float(c_s),
                'word_tfidf_score': float(w_s),
            }
            for e, pre_s, post_s, c_s, w_s in zip(test_contrast, pre_test_preds, post_test_preds, char_test_scores, word_test_scores)
        ]
    }

    report_path = Path('reports/reporting_contrast_experiment_results.json')
    write_json(report_path, report_data)
    print(f"\nSaved full contrastive experiment results to {report_path}")

if __name__ == '__main__':
    main()
