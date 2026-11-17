"""Compare saved predictions on matched English test IDs; describe Telugu separately."""
import argparse
import itertools
import json

import numpy as np

from .config import load_config, location, project_path, write_json
from .context_sweep import grouped_metrics
from .data import read_jsonl
from .evaluation import file_hash, validate_splits
from .statistics import connected_groups, paired_f1_interval


def align_predictions(records, rows, allow_superset=False):
    lookup = {r['id']: r for r in records}
    expected = {e['id'] for e in rows}
    if len(lookup) != len(records) or not expected <= set(lookup) or (not allow_superset and expected != set(lookup)):
        raise ValueError('Prediction IDs missing, duplicated, or from a different evaluation population')
    if any(lookup[e['id']]['label'] != e['label'] or lookup[e['id']]['source'] != e['source'] or
           lookup[e['id']]['group'] != e['group'] for e in rows):
        raise ValueError('Prediction labels, sources, or groups differ')
    scores = np.asarray([lookup[e['id']]['score'] for e in rows])
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Invalid prediction scores')
    return scores


def verify_anchor_subset(rows, anchor):
    lookup = {e['id']: e for e in anchor}
    for e in rows:
        old = lookup.get(e['id'])
        if old is None or any(e[key] != old[key] for key in ('label', 'texts', 'source', 'group', 'split')):
            raise ValueError('English rows do not exactly match the old held-out benchmark; paired comparison refused')


def compare(cfg, report_paths, historical_paths, output, resamples=1000):
    digest = file_hash(location(cfg, 'prepared'))
    if digest != cfg['data'].get('expected_prepared_sha256'):
        raise ValueError('Prepared English dataset hash mismatch')
    rows = read_jsonl(location(cfg, 'prepared'))
    test = validate_splits(rows)['test']
    old_cfg = load_config('configs/contextual_clean_lora.json')
    old_digest = file_hash(location(old_cfg, 'prepared')) if historical_paths else None
    anchor = read_jsonl(location(old_cfg, 'prepared')) if historical_paths else []
    if historical_paths:
        if old_digest != old_cfg['data']['expected_prepared_sha256']:
            raise ValueError('Historical clean dataset hash mismatch')
        verify_anchor_subset(rows, anchor)
    results, predictions, telugu = {}, {}, {}
    for historical, paths in [(False, report_paths), (True, historical_paths)]:
        for name in paths:
            path = project_path(name)
            report = json.loads(path.read_text(encoding='utf-8'))
            if report.get('prepared_sha256') != (old_digest if historical else digest):
                raise ValueError(f'Report dataset mismatch: {path}')
            records = json.loads(path.with_suffix('.predictions.json').read_text(encoding='utf-8'))
            if historical:
                align_predictions(records, [e for e in anchor if e['split'] == 'test'])
            scores = align_predictions(records, test, allow_superset=historical)
            key = ('mixed/' if historical else 'english/')+path.stem
            threshold = report['threshold']
            if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError('Invalid fixed threshold')
            results[key] = {'threshold': threshold, 'report_sha256': file_hash(path),
                            'predictions_sha256': file_hash(path.with_suffix('.predictions.json')),
                            **grouped_metrics(test, scores, threshold)}
            predictions[key] = scores
            if historical:
                te = [e for e in anchor if e['split']=='test' and e['source']=='cultural_safety_telugu']
                # This is a different task/population; do not calculate paired language differences.
                telugu[key] = grouped_metrics(te, align_predictions(records, te, True), threshold)
    groups = connected_groups(test)
    y = np.asarray([e['label'] for e in test])
    contextual = np.asarray([len(e['texts']) > 1 for e in test])
    paired = {}
    for a, b in itertools.combinations(results, 2):
        entry = {}
        for subset, mask in [('overall', np.ones(len(test), dtype=bool)), ('contextual', contextual)]:
            if mask.any():
                entry[subset] = paired_f1_interval(y[mask], predictions[a][mask], predictions[b][mask],
                    np.asarray(groups)[mask], results[a]['threshold'], results[b]['threshold'], resamples=resamples)
        paired[a+' minus '+b] = entry
    result = {'prepared_sha256': digest, 'english_same_test_population': results,
              'paired_group_bootstrap': paired, 'historical_telugu_prompt_safety_isolated_only': telugu,
              'interpretation': 'English personal-attack and Telugu prompt-safety tasks are different. Do not interpret their raw score gap as a language effect. Mixed models used Telugu and Tamil training; English models used only English. Historical thresholds remain fixed at their own original validation selections. Intervals do not correct test reuse or multiple comparisons.'}
    write_json(project_path(output), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--reports', nargs='+', required=True)
    parser.add_argument('--historical-reports', nargs='*', default=[])
    parser.add_argument('--output', required=True)
    parser.add_argument('--resamples', type=int, default=1000)
    args = parser.parse_args()
    if args.resamples < 100:
        parser.error('Use at least 100 resamples (1000 recommended)')
    result = compare(load_config(args.config), args.reports, args.historical_reports, args.output, args.resamples)
    for name, row in result['english_same_test_population'].items():
        print(name, row['overall'])


if __name__ == '__main__':
    main()
