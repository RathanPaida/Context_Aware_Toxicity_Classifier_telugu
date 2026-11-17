"""Resumable LoRA training with full validation and shared lossless text encoding."""
import argparse
import copy
import gc
import hashlib
import json
import random
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import average_precision_score
from transformers import AutoModel, AutoTokenizer

from .config import load_config, location, write_json
from .data import read_jsonl
from .embeddings import encode_texts, device_for
from .evaluation import file_hash, validate_splits, summarize, prediction_rows
from .model import ConversationClassifier
from .training import fit_threshold, metrics


def atomic_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def trainable_state(model):
    return {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}


def restore_trainable(model, state):
    params = dict(model.named_parameters())
    if set(state) != {k for k, p in params.items() if p.requires_grad}:
        raise ValueError('Adapter parameter names do not match checkpoint')
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)


def get_adapted_encoder(cfg, device):
    ec, ac = cfg['encoder'], cfg['adapter']
    kwargs = {'revision': ec['revision'], 'local_files_only': ec['local_files_only']}
    tokenizer = AutoTokenizer.from_pretrained(ec['name'], **kwargs)
    base = AutoModel.from_pretrained(ec['name'], **kwargs).to(device)
    if not ac['enabled']:
        raise ValueError('This entry point requires LoRA; use target_only.py for frozen baselines')
    model = get_peft_model(base, LoraConfig(r=ac['r'], lora_alpha=ac['lora_alpha'],
        lora_dropout=ac['lora_dropout'], target_modules=ac['target_modules'], bias='none'))
    if cfg['training'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    print('Trainable/total parameters:', model.get_nb_trainable_parameters(), flush=True)
    return model, tokenizer


def encode_batch_online(model, tokenizer, texts, device, max_tokens, batch_size=8):
    return encode_texts(model, tokenizer, texts, device, max_tokens, batch_size)


def forward_examples(encoder, tokenizer, head, rows, cfg, device):
    texts = list(dict.fromkeys(t for e in rows for t in e['texts'][-cfg['model']['max_turns']:]))
    lookup = {t: i for i, t in enumerate(texts)}
    encoding_texts = texts
    if cfg['data'].get('text_normalization') == 'nfc_controls_whitespace_v1':
        from .preprocessing import normalize_text
        encoding_texts = [normalize_text(t) for t in texts]
    vectors = encode_batch_online(encoder, tokenizer, encoding_texts, device,
        cfg['encoder']['max_tokens'], cfg['encoder']['batch_size'])
    indices = [[lookup[t] for t in e['texts'][-cfg['model']['max_turns']:]] for e in rows]
    width = max(map(len, indices))
    x = torch.zeros(len(rows), width, vectors.shape[-1], device=device)
    valid = torch.zeros(len(rows), width, dtype=torch.bool, device=device)
    for i, ids in enumerate(indices):
        x[i, :len(ids)] = vectors[ids]
        valid[i, :len(ids)] = True
    return head(x, valid)['logits']


@torch.inference_mode()
def predict_online(encoder, tokenizer, head, rows, cfg, device):
    encoder.eval()
    head.eval()
    batch_size = cfg['training']['batch_size']
    fp16 = cfg['training'].get('fp16', False) and device.type == 'cuda'
    result = []
    for start in range(0, len(rows), batch_size):
        with torch.amp.autocast(device.type, enabled=fp16):
            result.extend(forward_examples(encoder, tokenizer, head, rows[start:start+batch_size], cfg, device).sigmoid().cpu().tolist())
        if (start // batch_size + 1) % cfg['training'].get('progress_steps', 100) == 0:
            print(f'Evaluation progress: {min(start+batch_size, len(rows))}/{len(rows)}', flush=True)
    return np.array(result)


def run_adapter_experiment(cfg):
    t = cfg['training']
    random.seed(t['seed']); np.random.seed(t['seed']); torch.manual_seed(t['seed'])
    torch.set_num_threads(t['cpu_threads'])
    device = device_for(cfg)
    if t.get('require_cuda', False) and device.type != 'cuda':
        raise RuntimeError('CUDA unavailable. Recover GPU and retry; see SUBMISSION_HANDOFF.md. CPU smoke tests use a separate configuration.')
    data_hash = file_hash(location(cfg, 'prepared'))
    expected = cfg['data'].get('expected_prepared_sha256')
    if expected and expected != data_hash:
        raise ValueError('Prepared dataset differs from the frozen experiment. Restore it or create a new data version/config.')
    examples = read_jsonl(location(cfg, 'prepared'))
    splits = validate_splits(examples)
    if cfg['data']['require_real_context'] and not any(len(e['texts']) > 1 for e in splits['train']):
        raise ValueError('No context in training split')
    checkpoint = location(cfg, 'checkpoint')
    signature = hashlib.sha256(json.dumps({'cfg': cfg, 'data_sha256': data_hash, 'implementation': 'lora_v2'}, sort_keys=True).encode()).hexdigest()
    resume_path = checkpoint / 'resume.pt'
    # Never overwrite an unrelated completed experiment.
    if (checkpoint / 'metadata.json').exists() and not resume_path.exists():
        raise ValueError('Existing checkpoint has no matching resume state. Choose a new checkpoint directory.')
    resume = torch.load(resume_path, map_location='cpu', weights_only=True) if resume_path.exists() else None
    if resume and resume['signature'] != signature:
        raise ValueError('Resume configuration/data mismatch. Choose a new run directory.')
    encoder, tokenizer = get_adapted_encoder(cfg, device)
    head = ConversationClassifier(encoder.config.hidden_size, cfg).to(device)
    # Same full training pool as the frozen comparison; no source-dependent subsampling.
    pool = splits['train']
    positives = sum(e['label'] for e in pool)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(pool)-positives)/positives, device=device))
    optimizer = torch.optim.AdamW([
        {'params': [p for p in encoder.parameters() if p.requires_grad], 'lr': t['encoder_lr']},
        {'params': head.parameters(), 'lr': t['learning_rate']}], weight_decay=t['weight_decay'])
    fp16 = t.get('fp16', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=fp16)
    rng = random.Random(t['seed'])
    history, best, best_ap, stale, start_epoch = [], None, -float('inf'), 0, 0
    if resume:
        restore_trainable(encoder, resume['adapter'])
        head.load_state_dict(resume['head'])
        optimizer.load_state_dict(resume['optimizer']); scaler.load_state_dict(resume['scaler'])
        history, best, best_ap, stale = resume['history'], resume['best'], resume['best_ap'], resume['stale']
        start_epoch = resume['epoch']
        rng.setstate(resume['python_rng']); torch.set_rng_state(resume['torch_rng'])
        if device.type == 'cuda' and resume['cuda_rng']:
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
        print(f'Resuming after epoch {start_epoch}', flush=True)
    batch_size, accum = t['batch_size'], t['gradient_accumulation_steps']
    if accum < 1 or not 0 <= t.get('context_dropout', 0) <= 1:
        raise ValueError('Invalid accumulation/context dropout')
    steps = (len(pool) + batch_size - 1) // batch_size
    for epoch in range(start_epoch, t['epochs']):
        if stale >= t['patience']:
            break
        rows = list(pool); rng.shuffle(rows)
        epoch_started = time.perf_counter()
        encoder.train(); head.train(); optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, start in enumerate(range(0, len(rows), batch_size)):
            batch = rows[start:start+batch_size]
            batch = [{**e, 'texts': [e['texts'][-1]]} if rng.random() < t.get('context_dropout', 0) else e for e in batch]
            group_first_step = (step // accum) * accum
            group_start = group_first_step * batch_size
            group_size = min(accum * batch_size, len(rows) - group_start)
            with torch.amp.autocast(device.type, enabled=fp16):
                logits = forward_examples(encoder, tokenizer, head, batch, cfg, device)
                labels = torch.tensor([e['label'] for e in batch], dtype=torch.float32, device=device)
                raw_loss = criterion(logits, labels)
                loss = raw_loss * len(batch) / group_size
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; last completed epoch remains resumable')
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0 or step + 1 == steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), t['gradient_clip'])
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(raw_loss.detach()))
            if (step + 1) % t.get('progress_steps', 100) == 0:
                elapsed = time.perf_counter() - epoch_started
                remaining = elapsed / (step+1) * (steps-step-1)
                print(f'Epoch {epoch+1} step {step+1}/{steps}; loss {np.mean(losses):.4f}; training ETA {remaining/60:.1f} min (validation follows)', flush=True)
                write_json(location(cfg, 'report').with_suffix('.progress.json'),
                           {'status': 'training', 'epoch': epoch+1, 'step': step+1, 'steps': steps,
                            'elapsed_seconds': elapsed, 'training_eta_seconds': remaining})
        val = predict_online(encoder, tokenizer, head, splits['val'], cfg, device)
        ap = float(average_precision_score([e['label'] for e in splits['val']], val))
        history.append({'epoch': epoch+1, 'loss': float(np.mean(losses)), 'validation_ap': ap,
                        'seconds': time.perf_counter() - epoch_started})
        print(history[-1], flush=True)
        if ap > best_ap:
            best_ap, stale = ap, 0
            best = {'adapter': trainable_state(encoder), 'head': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
        else:
            stale += 1
        atomic_save({'signature': signature, 'epoch': epoch+1, 'history': history, 'best': best,
                     'best_ap': best_ap, 'stale': stale, 'adapter': trainable_state(encoder),
                     'head': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                     'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                     'python_rng': rng.getstate(), 'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}, resume_path)
    if best is None:
        raise RuntimeError('No valid training epoch completed')
    restore_trainable(encoder, best['adapter']); head.load_state_dict(best['head'])
    encoder.eval(); head.eval()
    val = predict_online(encoder, tokenizer, head, splits['val'], cfg, device)
    threshold = fit_threshold([e['label'] for e in splits['val']], val)
    # Compare the reloaded, merged encoder with the unmerged adapter on a real batch.
    probe = splits['val'][:batch_size]
    reference = predict_online(encoder, tokenizer, head, probe, cfg, device)
    merged = encoder.merge_and_unload()
    merged.save_pretrained(checkpoint / 'encoder'); tokenizer.save_pretrained(checkpoint / 'encoder')
    atomic_save(best['head'], checkpoint / 'head.pt')
    metadata = {'config': cfg, 'input_size': merged.config.hidden_size, 'threshold': threshold,
                'prepared_sha256': data_hash, 'has_context_training': True, 'labels': ['toxic'],
                'model_type': 'contextual_lora_v2', 'epochs': history,
                'selection_metric': 'full_validation_average_precision',
                'validation_average_precision': float(average_precision_score([e['label'] for e in splits['val']], val)),
                'training_examples': len(pool), 'positive_weight': float((len(pool)-positives)/positives)}
    write_json(checkpoint / 'metadata.json', metadata)
    del encoder, merged, optimizer, resume
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()
    from .embeddings import TextEncoder
    saved = TextEncoder(cfg, saved_path=checkpoint / 'encoder')
    reloaded = predict_online(saved.model, saved.tokenizer, head, probe, cfg, device)
    np.testing.assert_allclose(reference, reloaded, atol=1e-3 if fp16 else 1e-5, rtol=1e-2 if fp16 else 1e-4)
    test = splits['test']
    pred = predict_online(saved.model, saved.tokenizer, head, test, cfg, device)
    alone = predict_online(saved.model, saved.tokenizer, head, [{**e, 'texts': [e['texts'][-1]]} for e in test], cfg, device)
    report = {**metadata, **summarize(test, pred, threshold, alone),
              'validation': metrics([e['label'] for e in splits['val']], val, threshold),
              'checkpoint_reload_verified': True}
    write_json(location(cfg, 'report'), report)
    write_json(location(cfg, 'report').with_suffix('.predictions.json'), prediction_rows(test, pred, alone))
    write_json(location(cfg, 'report').with_suffix('.validation_predictions.json'), prediction_rows(splits['val'], val))
    write_json(location(cfg, 'report').with_suffix('.progress.json'), {'status': 'complete', 'epochs': len(history)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/contextual_lora_v2.json')
    args = parser.parse_args()
    cfg = load_config(args.config)
    try:
        run_adapter_experiment(cfg)
    except (KeyboardInterrupt, Exception) as exc:
        write_json(location(cfg, 'report').with_suffix('.progress.json'),
                   {'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                    'error_type': type(exc).__name__, 'detail': str(exc),
                    'recovery': 'Rerun the same configuration to resume the last completed epoch; see RUN_AND_REPORT.md.'})
        raise


if __name__ == '__main__': main()
