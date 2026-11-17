"""Audit returned saved English scores without loading checkpoints or fitting models."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contextual.config import write_json
from contextual.data import read_jsonl
from contextual.evaluation import file_hash, validate_splits
from contextual.context_sweep import grouped_metrics
from contextual.benchmark_compare import align_predictions, verify_anchor_subset
from contextual.training import metrics, fit_threshold
from contextual.statistics import connected_groups, paired_f1_interval


def main():
    archive_path = ROOT/'reports/english_cga_v1_results_for_review.zip'
    with zipfile.ZipFile(archive_path) as archive:
        def read(name):
            return json.loads(archive.read(name))
        manifest = read('reports/english_cga_v1_return_manifest.json')
        checks = []
        def check(name, passed):
            checks.append({'check': name, 'passed': bool(passed)})
        check('archive CRC', archive.testzip() is None)
        check('all declared expected files', not manifest['missing_expected'])
        for name, digest in manifest['sha256'].items():
            name = name.replace('\\', '/')
            check('archive SHA '+name, name in archive.namelist() and hashlib.sha256(archive.read(name)).hexdigest() == digest)
        cfg = read('configs/english_cga_v1_lora.json')
        prepared = ROOT/cfg['paths']['prepared']
        check('local English data matches archive identity', file_hash(prepared) == cfg['data']['expected_prepared_sha256'])
        rows = read_jsonl(prepared)
        split = validate_splits(rows)
        anchor = read_jsonl(ROOT/'data/processed/contextual_clean_v1.jsonl')
        verify_anchor_subset(rows, anchor)
        check('English population/splits match historical clean data', True)
        def close(a, b):
            if isinstance(a, dict):
                return isinstance(b, dict) and all(k in b and close(v, b[k]) for k, v in a.items())
            if isinstance(a, list):
                return isinstance(b, list) and len(a) == len(b) and all(close(x,y) for x,y in zip(a,b))
            if a is None: return b is None
            if isinstance(a, (float,int)): return isinstance(b,(float,int)) and np.isclose(a,b,atol=1e-10,rtol=1e-9)
            return a == b
        model_results, scores = {}, {}
        comparison = read('reports/english_cga_v1_comparison.json')
        for stem in ['tfidf','target_only','frozen','lora','ensemble']:
            name = 'reports/english_cga_v1_'+stem
            report = read(name+'.json')
            records = read(name+'.predictions.json')
            p = align_predictions(records, split['test'])
            check(stem+' data identity', report['prepared_sha256'] == file_hash(prepared))
            summary = grouped_metrics(split['test'], p, report['threshold'])
            check(stem+' full test metrics', close(summary['overall'], report['test']))
            check(stem+' contextual test metrics', close(summary['contextual'], report.get('contextual_test',report.get('contextual_test_subset'))))
            stored = comparison['english_same_test_population']['english/english_cga_v1_'+stem]
            check(stem+' comparison metrics', close(summary, stored))
            if name+'.validation_predictions.json' in archive.namelist():
                val = align_predictions(read(name+'.validation_predictions.json'), split['val'])
                check(stem+' validation F1 threshold', np.isclose(fit_threshold([e['label'] for e in split['val']], val), report['threshold']))
            model_results[stem] = {'threshold':report['threshold'], **summary}
            scores[stem] = p
        for stem, pop in [('budget_val','val'),('budget_test','test'),('lora_budget_test','test')]:
            name = 'reports/english_cga_v1_'+stem
            report = read(name+'.json')
            lookup = {r['id']:r for r in read(name+'.predictions.json')}
            check(stem+' IDs', set(lookup) == {e['id'] for e in split[pop]})
            for budget, result in report['budgets'].items():
                p = [lookup[e['id']]['scores_by_parent_budget'][budget] for e in split[pop]]
                check(stem+' budget '+budget+' metrics', close(grouped_metrics(split[pop], p, report['threshold']),result))
        challenges = {}
        for stem in ['english_shared_telugu_ensemble_challenge','english_shared_telugu_lora_challenge','english_cga_v1_telugu_transfer_challenge']:
            name = 'reports/'+stem
            report, records = read(name+'.json'), read(name+'.predictions.json')
            check(stem+' challenge hash', file_hash(ROOT/'data/raw/custom_threads.jsonl') == report['evaluated_sha256'])
            for budget, result in report['budgets'].items():
                p = [r['scores_by_parent_budget'][budget] for r in records]
                check(stem+' budget '+budget+' metrics', close(metrics([r['label'] for r in records],p,report['threshold']),result['overall']))
            challenges[stem] = {'threshold':report['threshold'], 'full_context':report['budgets']['11']['overall'],
                                'no_context':report['budgets']['0']['overall'],
                                'score_range': [min(r['score'] for r in records),max(r['score'] for r in records)]}
        groups = connected_groups(split['test'])
        y = [e['label'] for e in split['test']]
        paired = {}
        for baseline in ['tfidf','lora']:
            paired['ensemble_minus_'+baseline] = paired_f1_interval(y,scores['ensemble'],scores[baseline],groups,
                model_results['ensemble']['threshold'],model_results[baseline]['threshold'])
        mixed_report = read('reports/clean_ensemble.json')
        old_p = align_predictions(read('reports/clean_ensemble.predictions.json'), split['test'], True)
        paired['ensemble_minus_mixed_ensemble'] = paired_f1_interval(y,scores['ensemble'],old_p,groups,
            model_results['ensemble']['threshold'],mixed_report['threshold'])
        details = {'reviewed_archive':str(archive_path), 'archive_sha256':file_hash(archive_path),
                   'archive_size_bytes':archive_path.stat().st_size, 'archive_entries':len(archive.namelist()),
                   'attachment_identity_confirmed':False,
                   'attachment_note':'Windows transfer helper could not apply file attributes. Reviewed existing project ZIP of the same expected name/size; uploaded bytes not independently compared.',
                   'checks':checks, 'all_checks_passed':all(x['passed'] for x in checks),
                   'split_positive_counts':{k:sum(e['label'] for e in v) for k,v in split.items()},
                   'model_results':model_results, 'paired_intervals':paired, 'challenges':challenges,
                   'included_stage_logs':[n for n in archive.namelist() if n.endswith('.log')],
                   'production_training_or_inference_started':False}
        write_json(ROOT/'reports/english_post_run_verification.json', details)
        print(json.dumps({'checks':len(checks),'failed':[c for c in checks if not c['passed']],
                          'positives':details['split_positive_counts'],'paired':paired,
                          'challenge_ranges':{k:v['score_range'] for k,v in challenges.items()}},indent=2))


if __name__ == '__main__': main()
