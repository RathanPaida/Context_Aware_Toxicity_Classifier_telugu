"""Common evaluation populations, data identity, and per-example predictions."""
import hashlib
from collections import Counter

import numpy as np

from .data import fingerprint


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_splits(examples):
    """Reject accidental regeneration, invalid labels, and shared-text leakage."""
    splits = {s: [] for s in ('train', 'val', 'test')}
    owners, groups, ids = {}, {}, set()
    for ex in examples:
        if ex['id'] in ids:
            raise ValueError('Duplicate example ID')
        ids.add(ex['id'])
        split = ex['split']
        if split not in splits or ex['label'] not in (0, 1) or not ex['texts']:
            raise ValueError('Invalid prepared example')
        if groups.setdefault(ex['group'], split) != split:
            raise ValueError('Conversation group crosses splits')
        for text in ex['texts']:
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Empty prepared message')
            if owners.setdefault(fingerprint(text), split) != split:
                raise ValueError('Message text crosses splits')
        splits[split].append(ex)
    for name, rows in splits.items():
        if {r['label'] for r in rows} != {0, 1}:
            raise ValueError(f'{name} needs both classes')
    return splits


def summarize(test, probabilities, threshold, without_parents=None):
    from .training import metrics
    y, p = np.array([e['label'] for e in test]), np.asarray(probabilities)
    if len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError('Invalid prediction count or nonfinite predictions')
    context = np.array([len(e['texts']) > 1 for e in test])
    report = {'test': metrics(y, p, threshold), 'by_source': {},
              'test_ids_sha256': hashlib.sha256('\n'.join(e['id'] for e in test).encode()).hexdigest(),
              'contextual_test': metrics(y[context], p[context], threshold) if context.any() else None}
    for source in sorted({e['source'] for e in test}):
        mask = np.array([e['source'] == source for e in test])
        report['by_source'][source] = metrics(y[mask], p[mask], threshold)
    if without_parents is not None:
        alone = np.asarray(without_parents)
        if alone.shape != p.shape or not np.isfinite(alone).all():
            raise ValueError('Invalid ablation predictions')
        report['context_removed_ablation'] = metrics(y, alone, threshold)
        report['contextual_test_without_parents'] = metrics(y[context], alone[context], threshold) if context.any() else None
        report['mean_absolute_score_change'] = float(np.abs(p - alone).mean())
        report['contextual_mean_absolute_score_change'] = float(np.abs(p[context] - alone[context]).mean()) if context.any() else None
    return report


def prediction_rows(examples, probabilities, without_parents=None):
    return [{ 'id': e['id'], 'group': e['group'], 'source': e['source'],
              'label': e['label'], 'has_context': len(e['texts']) > 1,
              'score': float(probabilities[i]),
              **({'isolated_score': float(without_parents[i])} if without_parents is not None else {})}
            for i, e in enumerate(examples)]
