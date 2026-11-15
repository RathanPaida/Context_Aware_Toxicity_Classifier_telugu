"""Comparable target-only baseline and transparent experiment summaries."""
import hashlib
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .config import location, project_path, write_json
from .data import read_jsonl
from .training import fit_threshold, metrics


def train_baseline(cfg):
    settings = cfg['baseline']
    rows = read_jsonl(location(cfg, 'prepared'))
    splits = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val', 'test')}
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
    report['test'] = metrics([r['label'] for r in test], pred, threshold)
    report['by_source'] = {}
    for source in sorted({r['source'] for r in test}):
        ids = [i for i, r in enumerate(test) if r['source'] == source]
        report['by_source'][source] = metrics([test[i]['label'] for i in ids], pred[ids], threshold)
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
