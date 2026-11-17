"""Evaluate fixed checkpoints at explicit context budgets; never train or tune on test."""
import argparse
import copy
import json
import time

import numpy as np
import torch

from .config import load_config, location, project_path, write_json
from .data import examples_from_nodes, read_jsonl
from .english_experiment import depth_bin
from .evaluation import file_hash, prediction_rows, validate_splits
from .preprocessing import normalize_text
from .training import metrics, predict_examples


def grouped_metrics(rows, scores, threshold):
    scores = np.asarray(scores)
    if scores.shape != (len(rows),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Expected one finite probability in [0,1] per example')
    labels = np.asarray([e['label'] for e in rows])
    result = {'overall': metrics(labels, scores, threshold), 'by_available_ancestors': {}, 'by_source': {}}
    bins = [depth_bin(e.get('available_ancestors')) for e in rows]
    for name in ['0', '1', '2', '3', '4-7', '8-11', '12+', 'unknown']:
        mask = np.asarray([b == name for b in bins])
        result['by_available_ancestors'][name] = metrics(labels[mask], scores[mask], threshold) if mask.any() else None
    for source in sorted({e['source'] for e in rows}):
        mask = np.asarray([e['source'] == source for e in rows])
        result['by_source'][source] = metrics(labels[mask], scores[mask], threshold)
    context = np.asarray([len(e['texts']) > 1 for e in rows])
    result['contextual'] = metrics(labels[context], scores[context], threshold) if context.any() else None
    return result


def budget_rows(rows, budget):
    if not isinstance(budget, int) or budget < 0:
        raise ValueError('Parent budget must be a nonnegative integer')
    return [{**e, 'texts': e['texts'][-(budget+1):]} for e in rows]


def score_budgets(predictor, rows, budgets):
    """One fresh encoding pass; reuse message vectors across budgets, without disk caches."""
    from .ensemble import EnsemblePredictor
    ensemble = isinstance(predictor, EnsemblePredictor)
    neural = predictor.neural if ensemble else predictor
    limit = predictor.cfg['model']['max_turns']-1
    if not budgets or any(b < 0 or b > limit for b in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError(f'Use distinct parent budgets from 0 to {limit}; changing a saved positional embedding is not supported')
    if neural:
        texts = list(dict.fromkeys(t for e in rows for t in e['texts'][-(max(budgets)+1):]))
        print(f'Freshly encoding {len(texts)} unique messages; device={neural.runtime_device}; FP32 inference', flush=True)
        vectors = neural.encoder.encode(texts)
        lookup = {text: i for i, text in enumerate(texts)}
    lexical = np.zeros(len(rows))
    if ensemble:
        targets = [normalize_text(e['texts'][-1]) for e in rows]
        for name, model in predictor.bundle['lexical'].items():
            weight = predictor.bundle['weights'].get(name, 0)
            if weight:
                lexical += weight*model.predict_proba(targets)[:, 1]
    result = {}
    for budget in budgets:
        print(f'Scoring {len(rows)} examples with at most {budget} parents', flush=True)
        neural_scores = predict_examples(neural.model, budget_rows(rows, budget), vectors, lookup, neural.cfg) if neural else np.zeros(len(rows))
        result[budget] = lexical + predictor.bundle['weights'].get('contextual_transformer', 0)*neural_scores if ensemble else neural_scores
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--budgets', type=int, nargs='+', default=[0, 1, 2, 3, 5, 11])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--challenge', help='Optional separate labelled thread JSONL; no fitting')
    parser.add_argument('--reference-report', help='Saved trainer/ensemble report for FP32 score and decision parity at maximum budget')
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg['training']['device'] = args.device
    torch.set_num_threads(cfg['training']['cpu_threads'])
    data_path = location(cfg, 'prepared')
    digest = file_hash(data_path)
    if digest != cfg['data'].get('expected_prepared_sha256'):
        raise ValueError('Prepared data hash differs or is not frozen in this config')
    if args.challenge:
        challenge_path = project_path(args.challenge)
        c = copy.deepcopy(cfg)
        c['data']['capture_ancestry_counts'] = True
        rows = examples_from_nodes(read_jsonl(challenge_path), c, 'custom_threads')
        rows = [{**e, 'texts': [normalize_text(t) for t in e['texts']]} for e in rows]
        evaluated_hash = file_hash(challenge_path)
    else:
        rows = validate_splits(read_jsonl(data_path))[args.split]
        evaluated_hash = digest
    if not rows:
        raise ValueError('No evaluation examples')
    from .inference import create_predictor
    predictor = create_predictor(cfg)
    trained_hash = predictor.bundle['prepared_sha256'] if hasattr(predictor, 'bundle') else predictor.metadata['prepared_sha256']
    if trained_hash != digest:
        raise ValueError('Checkpoint and prepared dataset identities differ')
    start = time.monotonic()
    scores = score_budgets(predictor, rows, args.budgets)
    output = project_path(args.output)
    result = {'config_path': args.config, 'prepared_sha256': digest, 'evaluated_sha256': evaluated_hash,
              'population': 'authored_development_challenge' if args.challenge else args.split,
              'threshold': predictor.threshold, 'device': predictor.runtime_device,
              'precision': 'FP32 serving-style, not the trainer CUDA FP16 evaluation',
              'budgets': {str(b): grouped_metrics(rows, p, predictor.threshold) for b, p in scores.items()},
              'seconds': time.monotonic()-start,
              'notes': 'Same fixed checkpoint/threshold. Missing old depth metadata stays unknown. Original-label context-removal stress test; no relabelling. Empty bins are null. Tiny/one-class bins cannot establish generalization.'}
    if 0 in scores:
        from .statistics import connected_groups, paired_f1_interval
        ids = [i for i, e in enumerate(rows) if len(e['texts']) > 1]
        groups = connected_groups(rows)
        result['contextual_budget_minus_zero_intervals'] = {
            str(b): paired_f1_interval([rows[i]['label'] for i in ids], p[ids], scores[0][ids],
                                      [groups[i] for i in ids], predictor.threshold, predictor.threshold)
            for b, p in scores.items() if b != 0 and ids}
    if args.reference_report:
        if args.challenge or max(args.budgets) != cfg['model']['max_turns']-1:
            raise ValueError('Reference comparison requires the prepared split and full configured window')
        reference_path = project_path(args.reference_report)
        reference = json.loads(reference_path.read_text(encoding='utf-8'))
        if reference['prepared_sha256'] != digest or reference['threshold'] != predictor.threshold:
            raise ValueError('Reference dataset or threshold differs')
        suffix = '.validation_predictions.json' if args.split == 'val' else '.predictions.json'
        records = json.loads(reference_path.with_suffix(suffix).read_text(encoding='utf-8'))
        from .benchmark_compare import align_predictions
        original_scores = align_predictions(records, rows)
        current = scores[max(args.budgets)]
        result['serving_parity'] = {'reference_report': str(reference_path),
            'maximum_absolute_score_difference': float(np.abs(current-original_scores).max()),
            'mean_absolute_score_difference': float(np.abs(current-original_scores).mean()),
            'decision_flips_at_fixed_threshold': int(np.sum((current >= predictor.threshold) != (original_scores >= predictor.threshold))),
            'note': 'Descriptive parity check; do not retune on test to hide differences.'}
    write_json(output, result)
    predictions = prediction_rows(rows, scores[max(args.budgets)])
    for i, row in enumerate(predictions):
        row['scores_by_parent_budget'] = {str(b): float(p[i]) for b, p in scores.items()}
    write_json(output.with_suffix('.predictions.json'), predictions)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
