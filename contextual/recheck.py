"""Reevaluate saved contextual models without retraining or trusting old caches."""
import argparse
import json
import torch
from .config import load_config, location, project_path, write_json
from .data import read_jsonl, examples_from_nodes
from .evaluation import file_hash, summarize, prediction_rows
from .inference import Predictor
from .training import predict_examples


def recheck(cfg, output, contextual=False):
    torch.set_num_threads(cfg['training']['cpu_threads'])
    predictor = Predictor(location(cfg, 'checkpoint'))
    if contextual:
        if file_hash(location(cfg, 'prepared')) != predictor.metadata['prepared_sha256']:
            raise ValueError('Prepared dataset changed since this checkpoint was trained')
        rows = [e for e in read_jsonl(location(cfg, 'prepared')) if e['split'] == 'test' and len(e['texts']) > 1]
        population = 'historical_contextual_test_only'
    else:
        rows = examples_from_nodes(read_jsonl(project_path(cfg['data']['custom_threads'])), cfg, 'custom_challenge')
        population = 'curated_development_challenge_not_independent_test'
    texts = list(dict.fromkeys(t for e in rows for t in e['texts']))
    vectors = predictor.encoder.encode(texts)
    lookup = {t: i for i, t in enumerate(texts)}
    scores = predict_examples(predictor.model, rows, vectors, lookup, predictor.cfg)
    alone = predict_examples(predictor.model, [{**e, 'texts': [e['texts'][-1]]} for e in rows], vectors, lookup, predictor.cfg)
    report = {'population': population, 'checkpoint': str(location(cfg, 'checkpoint')),
              'threshold': predictor.threshold, 'fresh_embeddings': True,
              **summarize(rows, scores, predictor.threshold, alone)}
    if not contextual:
        prepared = read_jsonl(location(cfg, 'prepared'))
        from .data import fingerprint
        train_texts = {fingerprint(t) for e in prepared if e['split'] == 'train' for t in e['texts']}
        report['examples_sharing_any_training_text'] = sum(any(fingerprint(t) in train_texts for t in e['texts']) for e in rows)
    write_json(output, report)
    write_json(output.with_suffix('.predictions.json'), prediction_rows(rows, scores, alone))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--contextual', action='store_true')
    args = parser.parse_args()
    print(json.dumps(recheck(load_config(args.config), project_path(args.output), args.contextual), indent=2))


if __name__ == '__main__': main()
