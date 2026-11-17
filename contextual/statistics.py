"""Paired group-bootstrap F1 differences; descriptive only, never model selection."""
import argparse
import json

import numpy as np

from .config import load_config, location, project_path, write_json
from .data import fingerprint, read_jsonl
from .evaluation import file_hash


def connected_groups(rows):
    parents, owners = {}, {}
    def find(g):
        parents.setdefault(g, g)
        while parents[g] != g:
            parents[g] = parents[parents[g]]
            g = parents[g]
        return g
    for e in rows:
        for t in e['texts']:
            key = fingerprint(t)
            if key in owners: parents[find(e['group'])] = find(owners[key])
            owners[key] = e['group']
    return [find(e['group']) for e in rows]


def paired_f1_interval(labels, scores, reference_scores, groups, threshold, reference_threshold, resamples=1000, seed=2026):
    if resamples < 1: raise ValueError('resamples must be positive')
    y = np.asarray(labels, dtype=bool)
    a, b = np.asarray(scores) >= threshold, np.asarray(reference_scores) >= reference_threshold
    if not (len(y) == len(a) == len(b) == len(groups)) or len(y) == 0:
        raise ValueError('Paired populations must be nonempty and aligned')
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.zeros((inverse.max()+1, 6), dtype=np.int64)
    contributions = np.column_stack([a & y, a & ~y, ~a & y, b & y, b & ~y, ~b & y])
    np.add.at(counts, inverse, contributions.astype(np.int64))
    def difference(c):
        denom_a, denom_b = 2*c[0]+c[1]+c[2], 2*c[3]+c[4]+c[5]
        return (2*c[0]/denom_a if denom_a else 0) - (2*c[3]/denom_b if denom_b else 0)
    rng = np.random.default_rng(seed)
    deltas = [difference(counts[rng.integers(0,len(counts),len(counts))].sum(axis=0)) for _ in range(resamples)]
    return {'f1_difference': float(difference(counts.sum(axis=0))),
            'percentile_95_interval': np.quantile(deltas, [.025,.975]).tolist(),
            'groups': len(counts), 'examples': len(y), 'resamples': resamples, 'seed': seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/contextual_clean_lora.json')
    parser.add_argument('--report', default='reports/clean_ensemble.json')
    parser.add_argument('--output', default='reports/clean_ensemble_uncertainty.json')
    args = parser.parse_args()
    cfg, path = load_config(args.config), project_path(args.report)
    report = json.loads(path.read_text())
    if report['prepared_sha256'] != file_hash(location(cfg,'prepared')): raise ValueError('Data hash mismatch')
    rows = [e for e in read_jsonl(location(cfg,'prepared')) if e['split']=='test']
    records = json.loads(path.with_suffix('.predictions.json').read_text())
    lookup = {r['id']:r for r in records}
    if len(lookup)!=len(records) or set(lookup)!={e['id'] for e in rows}: raise ValueError('Prediction IDs mismatch')
    if any(lookup[e['id']]['label'] != e['label'] for e in rows): raise ValueError('Prediction label mismatch')
    baseline = next(r for r in report['single_models'] if r['name']==report['best_single_by_validation'])
    groups, y = connected_groups(rows), [e['label'] for e in rows]
    p = [lookup[e['id']]['score'] for e in rows]
    q = [lookup[e['id']]['component_scores'][baseline['name']] for e in rows]
    result = {'comparison': 'ensemble minus '+baseline['name'],
              'overall': paired_f1_interval(y,p,q,groups,report['threshold'],baseline['threshold']),
              'notes': 'Percentile intervals resample connected conversation/shared-text groups. Exploratory test reuse and validation-selection uncertainty are not removed by bootstrapping. An interval crossing zero does not establish improvement.'}
    indices = [i for i,e in enumerate(rows) if len(e['texts'])>1]
    if indices:
        result['contextual'] = paired_f1_interval(*[[v[i] for i in indices] for v in (y,p,q,groups)],
                                                  report['threshold'],baseline['threshold'])
    write_json(project_path(args.output),result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
