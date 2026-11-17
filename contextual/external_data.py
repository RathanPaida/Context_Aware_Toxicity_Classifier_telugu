"""Pinned public-data downloads and explicit adapters, isolated from the frozen benchmark."""
import argparse
import csv
import json
from collections import Counter, defaultdict

import requests

from .config import load_config, project_path, write_json
from .data import fingerprint, grouped_split, read_jsonl
from .evaluation import file_hash, validate_splits


def fetch_sources(settings):
    results = []
    for source in settings['sources']:
        for item in source['files']:
            path = project_path(item['path'])
            path.parent.mkdir(parents=True, exist_ok=True)
            receipt = path.with_suffix(path.suffix + '.receipt.json')
            previous = json.loads(receipt.read_text()) if receipt.exists() else None
            if not (path.exists() and previous and previous['url'] == item['url'] and previous['sha256'] == file_hash(path)):
                temporary = path.with_suffix(path.suffix + '.partial')
                with requests.get(item['url'], stream=True, timeout=settings['timeout_seconds']) as response:
                    response.raise_for_status()
                    size = 0
                    with temporary.open('wb') as stream:
                        for block in response.iter_content(settings['chunk_bytes']):
                            size += len(block)
                            if size > settings['max_file_bytes']:
                                raise ValueError('Source exceeds configured size limit')
                            stream.write(block)
                # Parse before accepting a download as complete.
                if source['format'] == 'toxichat':
                    for row in read_jsonl(temporary):
                        if 'reddit_thread' not in row: raise ValueError('Invalid ToxiChat row')
                else:
                    with temporary.open(encoding='utf-8-sig', newline='') as stream:
                        reader = csv.DictReader(stream)
                        if not {'label', 'text'} <= set(reader.fieldnames or []): raise ValueError('Invalid MACD CSV')
                temporary.replace(path)
                write_json(receipt, {'url': item['url'], 'sha256': file_hash(path), 'source': source['name']})
            results.append({'source': source['name'], 'path': str(path), 'sha256': file_hash(path)})
            print('Verified:', path.name, flush=True)
    write_json(project_path(settings['download_manifest']), results)
    return results


def convert_source(source):
    examples = []
    for item in source['files']:
        path = project_path(item['path'])
        original_split = item['split']
        if source['format'] == 'toxichat':
            for i, row in enumerate(read_jsonl(path)):
                thread = row['reddit_thread']
                if not thread or any(not isinstance(t.get('text'), str) or not t['text'].strip() for t in thread):
                    raise ValueError(f'Empty ToxiChat thread at {path}:{i+1}')
                ids = [str(t['turn_id']) for t in thread]
                if len(set(ids)) != len(ids): raise ValueError('Duplicate ToxiChat turn IDs')
                group = f'{source["name"]}:{original_split}:{i}'
                for j, turn in enumerate(thread):
                    label = source['label_map'][turn['offense_label']]
                    examples.append({'id': f'{group}:{ids[j]}', 'group': group, 'source': source['name'],
                                     'texts': [t['text'] for t in thread[:j+1]], 'label': label,
                                     'message_ids': [f'{group}:{x}' for x in ids[:j+1]],
                                     'original_split': original_split})
                # Generated model continuations are deliberately not treated as natural user replies.
        else:
            with path.open(encoding='utf-8-sig', newline='') as stream:
                for i, row in enumerate(csv.DictReader(stream)):
                    if not row['text'].strip(): continue
                    label = source['label_map'][row['label']]
                    identity = f'{source["name"]}:{original_split}:{i}'
                    examples.append({'id': identity, 'group': f'{source["name"]}:{fingerprint(row["text"])}',
                                     'source': source['name'], 'texts': [row['text']], 'message_ids': [identity],
                                     'label': label, 'original_split': original_split})
    return examples


def prepare_external(settings):
    rows = [e for source in settings['sources'] for e in convert_source(source)]
    # Quarantine contradictory isolated labels, never arbitrarily decide the true label.
    labels = defaultdict(set)
    for e in rows:
        if len(e['texts']) == 1: labels[(e['source'], fingerprint(e['texts'][0]))].add(e['label'])
    conflicts = {key for key, value in labels.items() if len(value) > 1}
    kept = [e for e in rows if not (len(e['texts']) == 1 and (e['source'], fingerprint(e['texts'][0])) in conflicts)]
    cfg = load_config(settings['training_config'])
    prepared = grouped_split(kept, cfg)
    validate_splits(prepared)
    destination = project_path(settings['prepared'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for e in prepared: stream.write(json.dumps(e, ensure_ascii=False) + '\n')
    temporary.replace(destination)
    manifest = {'examples': len(prepared), 'contextual_examples': sum(len(e['texts']) > 1 for e in prepared),
                'sources': dict(Counter(e['source'] for e in prepared)),
                'splits': dict(Counter(e['split'] for e in prepared)),
                'quarantined_conflicting_isolated_rows': len(rows)-len(kept),
                'prepared_sha256': file_hash(destination),
                'policy': 'New shared-text/conversation grouped splits; not official benchmark splits. Original split is retained as provenance. External experiment only; frozen main benchmark unchanged.'}
    write_json(project_path(settings['manifest']), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/data_sources.json')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    settings = json.loads(project_path(args.config).read_text())
    if args.download: fetch_sources(settings)
    if args.prepare: print(json.dumps(prepare_external(settings), indent=2))
    if not args.download and not args.prepare: parser.print_help()


if __name__ == '__main__': main()
