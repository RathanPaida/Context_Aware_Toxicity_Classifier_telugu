"""Validation-selected soft voting; test labels never choose weights or thresholds."""
import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from .config import load_config, location, project_path, write_json
from .data import read_jsonl
from .evaluation import file_hash, validate_splits, summarize, prediction_rows
from .preprocessing import normalize_text
from .training import fit_threshold, metrics


def select_weights(labels, scores, contextual, resolution=4):
    """Predeclared objective: equal weight on overall and contextual validation AP.

    Endpoints include each single model, so adding an ensemble is never compulsory.
    Exact ties prefer fewer components. Threshold is subsequently fitted on validation.
    """
    y, matrix, mask = np.asarray(labels), np.asarray(scores), np.asarray(contextual, dtype=bool)
    if (matrix.ndim != 2 or matrix.shape[0] != len(y) or not np.isfinite(matrix).all()
            or matrix.shape[1] == 0 or len(mask) != len(y) or set(y) != {0, 1}
            or np.any(matrix < 0) or np.any(matrix > 1) or resolution < 1):
        raise ValueError('Invalid validation score matrix')
    has_context = mask.any() and len(set(y[mask])) == 2
    candidates = []
    for numerator in itertools.product(range(resolution+1), repeat=matrix.shape[1]):
        if sum(numerator) != resolution: continue
        weight = np.asarray(numerator, dtype=float)/resolution
        p = matrix @ weight
        overall = float(average_precision_score(y, p))
        context = float(average_precision_score(y[mask], p[mask])) if has_context else overall
        threshold = fit_threshold(y, p)
        candidates.append({'weights': weight.tolist(), 'objective': (overall+context)/2,
                           'overall_ap': overall, 'contextual_ap': context if has_context else None,
                           'threshold': threshold, 'validation': metrics(y, p, threshold)})
    candidates.sort(key=lambda r: (r['objective'], -sum(w > 0 for w in r['weights'])), reverse=True)
    return candidates[0], candidates


def aligned_predictions(path, rows, field='score'):
    records = json.loads(Path(path).read_text(encoding='utf-8'))
    by_id = {e['id']: e for e in records}
    if len(by_id) != len(records) or set(by_id) != {e['id'] for e in rows}:
        raise ValueError(f'Prediction IDs do not match population: {path}')
    if any(by_id[e['id']]['label'] != e['label'] for e in rows):
        raise ValueError('Prediction labels disagree with frozen data')
    values = np.array([by_id[e['id']][field] for e in rows])
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError('Invalid probability values')
    return values


def run_ensemble(cfg, output, neural_report=None):
    data_hash = file_hash(location(cfg, 'prepared'))
    expected = cfg['data'].get('expected_prepared_sha256')
    if expected and expected != data_hash: raise ValueError('Prepared-data hash mismatch')
    splits = validate_splits(read_jsonl(location(cfg, 'prepared')))
    val, test = splits['val'], splits['test']
    baseline_path = project_path(cfg['baseline']['report']).with_suffix('.joblib')
    if not baseline_path.exists():
        from .experiments import train_baseline
        train_baseline(cfg)
    char = joblib.load(baseline_path)
    if char['prepared_sha256'] != data_hash: raise ValueError('TF-IDF model trained on different data')
    word = make_pipeline(TfidfVectorizer(preprocessor=normalize_text, tokenizer=str.split, token_pattern=None,
        ngram_range=(1, 2), max_features=100000, min_df=2, sublinear_tf=True),
        LogisticRegression(C=1.0, max_iter=1000, random_state=cfg['training']['seed']))
    print('Fitting word TF-IDF on training targets only', flush=True)
    with threadpool_limits(limits=2):
        word.fit([e['texts'][-1] for e in splits['train']], [e['label'] for e in splits['train']])
    lexical = {'char_tfidf': char['pipeline'], 'word_tfidf': word}
    names = list(lexical)
    vs = [model.predict_proba([e['texts'][-1] for e in val])[:, 1] for model in lexical.values()]
    ts = [model.predict_proba([e['texts'][-1] for e in test])[:, 1] for model in lexical.values()]
    ablated = [p.copy() for p in ts]
    neural_checkpoint, neural_identity = None, None
    if neural_report:
        path = project_path(neural_report)
        report = json.loads(path.read_text())
        if report['prepared_sha256'] != data_hash: raise ValueError('Neural training population differs')
        vs.append(aligned_predictions(path.with_suffix('.validation_predictions.json'), val))
        ts.append(aligned_predictions(path.with_suffix('.predictions.json'), test))
        ablated.append(aligned_predictions(path.with_suffix('.predictions.json'), test, 'isolated_score'))
        names.append('contextual_transformer')
        neural_checkpoint = report['config']['paths']['checkpoint']
        checkpoint = project_path(neural_checkpoint)
        neural_identity = {'metadata_sha256': file_hash(checkpoint/'metadata.json'),
                           'head_sha256': file_hash(checkpoint/'head.pt'),
                           'encoder_files': {str(p.relative_to(checkpoint)): file_hash(p)
                                             for p in (checkpoint/'encoder').iterdir() if p.is_file()}}
    y = [e['label'] for e in val]
    selected, candidates = select_weights(y, np.column_stack(vs), [len(e['texts']) > 1 for e in val])
    weights, threshold = np.asarray(selected['weights']), selected['threshold']
    print('Selected by validation:', dict(zip(names, weights)), flush=True)
    probabilities = np.column_stack(ts) @ weights
    without = np.column_stack(ablated) @ weights
    singles = []
    for i, name in enumerate(names):
        t = fit_threshold(y, vs[i])
        candidate = next(c for c in candidates if c['weights'][i] == 1)
        singles.append({'name': name, 'validation_objective': candidate['objective'],
                        'threshold': t, 'validation': metrics(y, vs[i], t),
                        **summarize(test, ts[i], t, ablated[i])})
    best_single = max(singles, key=lambda r: r['validation_objective'])
    result = {'model_type': 'validation_selected_soft_vote', 'prepared_sha256': data_hash,
              'weights': dict(zip(names, weights.tolist())), 'threshold': threshold,
              'selection': 'Maximum mean of overall and contextual validation AP; grid step 0.25; threshold maximizes overall validation F1. Test labels are not used for selection.',
              'validation': metrics(y, np.column_stack(vs)@weights, threshold),
              'validation_objective': selected['objective'], 'validation_candidates': candidates,
              'single_models': singles, 'best_single_by_validation': best_single['name'],
              'test_f1_difference_from_best_validation_single': float(summarize(test, probabilities, threshold)['test']['f1']-best_single['test']['f1']),
              **summarize(test, probabilities, threshold, without),
              'notes': 'Exploratory: the historical test set has been inspected. Voting scores are not calibrated probabilities. No source-specific threshold or test-selected weights.'}
    output = project_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {'config': cfg, 'prepared_sha256': data_hash, 'lexical': lexical,
                'weights': result['weights'], 'threshold': threshold,
                'neural_checkpoint': neural_checkpoint, 'neural_identity': neural_identity}
    temporary = output.with_suffix('.tmp')
    joblib.dump(artifact, temporary)
    temporary.replace(output.with_suffix('.joblib'))
    write_json(output, result)
    saved_predictions = prediction_rows(test, probabilities, without)
    for i, row in enumerate(saved_predictions):
        row['component_scores'] = {name: float(ts[j][i]) for j, name in enumerate(names)}
    write_json(output.with_suffix('.predictions.json'), saved_predictions)
    write_json(output.with_suffix('.validation_predictions.json'), prediction_rows(val, np.column_stack(vs)@weights))
    return result


class EnsemblePredictor:
    def __init__(self, artifact, device='cpu'):
        self.bundle = joblib.load(project_path(artifact))
        self.cfg = self.bundle['config']
        self.threshold = self.bundle['threshold']
        self.neural = None
        self.runtime_device = 'cpu'
        if self.bundle['weights'].get('contextual_transformer', 0) > 0:
            from .inference import Predictor
            checkpoint = project_path(self.bundle['neural_checkpoint'])
            identity = self.bundle['neural_identity']
            if file_hash(checkpoint/'metadata.json') != identity['metadata_sha256'] or file_hash(checkpoint/'head.pt') != identity['head_sha256']:
                raise ValueError('Ensemble neural checkpoint identity changed')
            if any(file_hash(checkpoint/p) != digest for p, digest in identity['encoder_files'].items()):
                raise ValueError('Ensemble encoder identity changed')
            self.neural = Predictor(checkpoint, device=device)
            self.cfg = self.neural.cfg
            self.runtime_device = self.neural.runtime_device

    def predict(self, text, history=None, include_embeddings=False):
        history = [] if history is None else history
        if not isinstance(text, str) or not text.strip(): raise ValueError('text must be nonempty')
        if not isinstance(history, list) or any(not isinstance(t, str) or not t.strip() for t in history):
            raise ValueError('history must contain nonempty ancestor strings')
        if self.cfg['data'].get('text_normalization') == 'nfc_controls_whitespace_v1':
            text = normalize_text(text)
            history = [normalize_text(t) for t in history]
            if not text or any(not t for t in history): raise ValueError('Empty text after normalization')
        if self.neural:
            result = self.neural.predict(text, history, include_embeddings)
            neural_score, isolated = result['toxicity_score'], result['isolated_score']
        else:
            result = {'turns_used': 1, 'ancestors_omitted': len(history), 'turn_attention': [],
                      'has_context_training': False, 'embeddings_available': False}
            neural_score = isolated = 0.0
        component = {name: float(model.predict_proba([text])[0, 1]) for name, model in self.bundle['lexical'].items()}
        component['contextual_transformer'] = neural_score
        score = sum(w*component[name] for name, w in self.bundle['weights'].items())
        alone = score + self.bundle['weights'].get('contextual_transformer', 0)*(isolated-neural_score)
        result.update(toxicity_score=score, is_toxic=score >= self.threshold, threshold=self.threshold,
                      isolated_score=alone, context_score_delta=score-alone,
                      component_scores=component, ensemble_weights=self.bundle['weights'],
                      attention_scope='Neural component only; not a causal explanation of the ensemble')
        return result

    def predict_thread(self, messages, target_id, include_embeddings=False):
        from .inference import Predictor
        return Predictor.predict_thread(self, messages, target_id, include_embeddings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/contextual_submission.json')
    parser.add_argument('--output', default='reports/lexical_ensemble.json')
    parser.add_argument('--neural-report')
    parser.add_argument('--predict')
    parser.add_argument('--parent', action='append', default=[])
    parser.add_argument('--artifact')
    args = parser.parse_args()
    if args.predict is not None:
        if not args.artifact: parser.error('--predict requires --artifact')
        print(json.dumps(EnsemblePredictor(args.artifact).predict(args.predict, args.parent), ensure_ascii=False, indent=2))
    else:
        report = run_ensemble(load_config(args.config), args.output, args.neural_report)
        print(json.dumps({k: report[k] for k in ('weights', 'validation', 'test', 'contextual_test', 'test_f1_difference_from_best_validation_single')}, indent=2))


if __name__ == '__main__': main()
