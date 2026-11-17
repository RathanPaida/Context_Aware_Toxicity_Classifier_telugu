"""Conservative multilingual text normalization and reproducible dataset audit.

Preserve negation, punctuation, emoji, casing, script and joiners. Do not strip
stopwords, stem, translate, or change context-dependent labels automatically.
"""
import argparse
from collections import Counter, defaultdict
import json
import unicodedata

from .config import project_path, write_json
from .data import fingerprint, read_jsonl
from .evaluation import file_hash, validate_splits


def normalize_text(text):
    if not isinstance(text, str):
        raise ValueError('Expected a text string')
    text = unicodedata.normalize('NFC', text)
    text = ''.join(c if c in '\n\r\t' or unicodedata.category(c) != 'Cc' else ' ' for c in text)
    return ' '.join(text.split())


def audit(rows):
    splits = validate_splits(rows)
    counts, isolated_labels, chains, normalized_splits = Counter(), defaultdict(set), Counter(), defaultdict(set)
    texts = list(dict.fromkeys(t for e in rows for t in e['texts']))
    for text in texts:
        normalized = normalize_text(text)
        counts['texts_changed_by_conservative_normalization'] += normalized != text
        counts['texts_with_control_characters'] += any(unicodedata.category(c) == 'Cc' and c not in '\n\r\t' for c in text)
        counts['texts_not_NFC'] += unicodedata.normalize('NFC', text) != text
        counts['texts_over_2000_characters'] += len(text) > 2000
        counts['empty_after_normalization'] += not normalized
    for e in rows:
        signature = tuple(fingerprint(normalize_text(t)) for t in e['texts'])
        chains[(e['source'], e['split'], signature, e['label'])] += 1
        if len(e['texts']) == 1:
            isolated_labels[(e['source'], signature[0])].add(e['label'])
        for text in e['texts']:
            normalized_splits[fingerprint(normalize_text(text))].add(e['split'])
    return {'examples': len(rows), 'unique_messages': len(texts),
            'split_counts': {s: len(v) for s, v in splits.items()}, 'text_audit': dict(counts),
            'duplicate_same_source_chain_label_rows': sum(n-1 for n in chains.values()),
            'conflicting_isolated_text_groups_within_source': sum(len(labels) > 1 for labels in isolated_labels.values()),
            'normalized_shared_text_groups_crossing_splits': sum(len(s) > 1 for s in normalized_splits.values()),
            'by_source_split_label': dict(Counter(f'{e["source"]}/{e["split"]}/{e["label"]}' for e in rows)),
            'policy': 'Audit only; original benchmark and labels unchanged. Exact same-chain duplicates may overweight training. Differing labels for a target in different contexts are not automatically errors. Near-duplicate semantics require manual adjudication.'}


def clean_snapshot(rows):
    """Create a new version: normalize, quarantine ambiguous isolated labels, deduplicate.

    Existing split membership is retained. If normalization introduces cross-split
    overlap, reject the version and require regrouping rather than hiding leakage.
    """
    normalized = [{**e, 'texts': [normalize_text(t) for t in e['texts']]} for e in rows]
    labels = defaultdict(set)
    for e in normalized:
        if len(e['texts']) == 1:
            labels[(e['source'], fingerprint(e['texts'][0]))].add(e['label'])
    seen, kept, excluded = set(), [], []
    for e in normalized:
        reason = None
        signature = (e['source'], e['split'], tuple(fingerprint(t) for t in e['texts']), e['label'])
        if any(not t for t in e['texts']):
            reason = 'empty_after_normalization'
        elif len(e['texts']) == 1 and len(labels[(e['source'], fingerprint(e['texts'][0]))]) > 1:
            reason = 'contradictory_isolated_labels'
        elif signature in seen:
            reason = 'duplicate_same_chain_label'
        if reason:
            excluded.append({'id': e['id'], 'reason': reason})
        else:
            seen.add(signature)
            kept.append(e)
    validate_splits(kept)
    return kept, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/processed/contextual.jsonl')
    parser.add_argument('--report', default='reports/preprocessing_audit.json')
    parser.add_argument('--clean-output', help='Optional distinct prepared-data version; never the input path')
    args = parser.parse_args()
    source, report_path = project_path(args.input), project_path(args.report)
    rows = read_jsonl(source)
    report = {'input': str(source), 'prepared_sha256': file_hash(source), **audit(rows)}
    if args.clean_output:
        target = project_path(args.clean_output)
        if target == source: raise ValueError('Refusing to overwrite the frozen input dataset')
        kept, excluded = clean_snapshot(rows)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            for e in kept: stream.write(json.dumps(e, ensure_ascii=False)+'\n')
        temporary.replace(target)
        write_json(report_path.with_suffix('.excluded.json'), excluded)
        report['cleaned'] = {'path': str(target), 'examples': len(kept), 'sha256': file_hash(target),
                             'excluded_by_reason': dict(Counter(e['reason'] for e in excluded)),
                             'audit': audit(kept), 'comparison': 'New population: retrain all compared models on this same version.'}
    write_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
