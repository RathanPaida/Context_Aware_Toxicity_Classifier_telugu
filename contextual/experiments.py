"""Comparable target-only baseline and transparent experiment summaries."""
import hashlib
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .config import location, project_path, write_json
from .data import read_jsonl
from .training import fit_threshold, metrics
from .evaluation import validate_splits, summarize, prediction_rows
import joblib


def train_baseline(cfg):
    from .evaluation import file_hash
    if cfg['data'].get('expected_prepared_sha256') and file_hash(location(cfg, 'prepared')) != cfg['data']['expected_prepared_sha256']:
        raise ValueError('Prepared data does not match the frozen experiment SHA-256')
    settings = cfg['baseline']
    rows = read_jsonl(location(cfg, 'prepared'))
    splits = validate_splits(rows)
    model = make_pipeline(
        TfidfVectorizer(analyzer='char_wb', ngram_range=tuple(settings['ngram_range']),
                        max_features=settings['max_features'], sublinear_tf=True),
        LogisticRegression(C=settings['C'], max_iter=settings['max_iter'],
                           random_state=cfg['training']['seed']))
    print('Fitting TF-IDF baseline on training targets only', flush=True)
    model.fit([r['texts'][-1] for r in splits['train']], [r['label'] for r in splits['train']])
    val = model.predict_proba([r['texts'][-1] for r in splits['val']])[:, 1]
    threshold = fit_threshold([r['label'] for r in splits['val']], val)
    report = {'threshold': threshold, 'prepared_sha256': hashlib.sha256(location(cfg, 'prepared').read_bytes()).hexdigest()}
    report['validation'] = metrics([r['label'] for r in splits['val']], val, threshold)
    test = splits['test']
    pred = model.predict_proba([r['texts'][-1] for r in test])[:, 1]
    report.update(summarize(test, pred, threshold))
    report_path = project_path(settings['report'])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = report_path.with_suffix('.joblib')
    temporary = model_path.with_suffix('.tmp')
    joblib.dump({'pipeline': model, 'threshold': threshold, 'prepared_sha256': report['prepared_sha256']}, temporary)
    temporary.replace(model_path)
    write_json(report_path.with_suffix('.predictions.json'), prediction_rows(test, pred))
    write_json(project_path(settings['report']), report)
    return report


def compare_reports(cfg):
    import pandas as pd
    rows = []
    for name, path in [('TF-IDF', project_path(cfg['baseline']['report'])),
                       ('Candidate', location(cfg, 'report'))]:
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding='utf-8'))
        rows.append({'model': name, 'validation_AP': r.get('validation_average_precision',
                     r.get('validation', {}).get('average_precision')),
                     'test_F1': r['test']['f1'], 'test_AP': r['test']['average_precision']})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import argparse
    from .config import load_config
    parser = argparse.ArgumentParser(description='Train a target-only TF-IDF baseline on the configured splits')
    parser.add_argument('--config', default='configs/contextual_submission.json')
    args = parser.parse_args()
    print(json.dumps(train_baseline(load_config(args.config)), indent=2))
