"""Download separately; prepare isolated English experiments without launching training."""
import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict

from .config import load_config, location, project_path, write_json
from .data import download, fingerprint, grouped_split, load_cga, read_jsonl
from .evaluation import file_hash, validate_splits
from .external_data import convert_source, fetch_sources
from .preprocessing import normalize_text


def source_settings():
    settings = json.loads(project_path('configs/data_sources.json').read_text(encoding='utf-8'))
    settings['sources'] = [s for s in settings['sources'] if s['format'] == 'toxichat']
    settings['download_manifest'] = 'reports/english_toxichat_downloads.json'
    return settings


def depth_bin(depth):
    if depth is None:
        return 'unknown'
    if depth < 0:
        raise ValueError('Negative ancestor count')
    if depth <= 3:
        return str(depth)
    if depth <= 7:
        return '4-7'
    if depth <= 11:
        return '8-11'
    return '12+'


def inventory(rows):
    counts = Counter()
    positives = Counter()
    for row in rows:
        key = f"{row.get('split', 'challenge')}/{depth_bin(row.get('available_ancestors'))}"
        counts[key] += 1
        positives[key] += row['label']
    return {key: {'n': n, 'positive_count': positives[key], 'negative_count': n-positives[key]}
            for key, n in sorted(counts.items())}


def align_anchor(raw, anchor, source, max_turns):
    """Keep the already-reviewed population exactly; raw data only recovers depth."""
    lookup = {e['id']: e for e in raw}
    selected = [e for e in anchor if e['source'] == source]
    if not selected:
        raise ValueError(f'Anchor contains no {source} examples')
    result = []
    for row in selected:
        full = lookup.get(row['id'])
        if full is None or full['label'] != row['label']:
            raise ValueError(f"Raw source does not match anchor: {row['id']}")
        window = [normalize_text(t) for t in full['texts'][-max_turns:]]
        if window != row['texts'] or full['message_ids'][-max_turns:] != row['message_ids']:
            raise ValueError('Anchor text/window/IDs differ. Restore matching raw data; do not silently change splits.')
        result.append({**row, 'available_ancestors': full['available_ancestors'],
                       'retained_ancestors': len(row['texts'])-1,
                       'depth_provenance': full['depth_provenance']})
    validate_splits(result)
    return result


def fresh_split(raw, cfg, max_turns):
    """Normalize and group FULL released paths before applying the model window."""
    normalized = [{**r, 'texts': [normalize_text(t) for t in r['texts']]} for r in raw]
    isolated = defaultdict(set)
    for row in normalized:
        if len(row['texts']) == 1:
            isolated[fingerprint(row['texts'][0])].add(row['label'])
    kept, seen, excluded = [], set(), Counter()
    for row in normalized:
        signature = (tuple(fingerprint(t) for t in row['texts']), row['label'])
        if any(not t for t in row['texts']):
            excluded['empty_after_normalization'] += 1
        elif len(row['texts']) == 1 and len(isolated[fingerprint(row['texts'][0])]) > 1:
            excluded['contradictory_isolated_label'] += 1
        elif signature in seen:
            excluded['duplicate_full_chain_label'] += 1
        else:
            seen.add(signature)
            kept.append(row)
    grouped_split(kept, cfg)
    validate_splits(kept)
    for row in kept:
        row['texts'] = row['texts'][-max_turns:]
        row['message_ids'] = row['message_ids'][-max_turns:]
        row['retained_ancestors'] = len(row['texts'])-1
    validate_splits(kept)
    return kept, dict(excluded)


def make_configs(tag, digest, max_turns):
    configs = {}
    for kind, template in [('lora', 'contextual_clean_lora'), ('frozen', 'contextual_clean_frozen'),
                           ('target_only', 'contextual_clean_target_only')]:
        cfg = load_config(f'configs/{template}.json')
        cfg['paths'].update(prepared=f'data/processed/{tag}.jsonl',
            embeddings=f'data/cache/{tag}_{"lora" if kind == "lora" else "frozen"}.pt',
            checkpoint=f'models/checkpoints/{tag}_{kind}', report=f'reports/{tag}_{kind}.json',
            manifest=f'reports/{tag}_manifest.json')
        cfg['model']['max_turns'] = max_turns
        cfg['encoder']['batch_size'] = 8
        cfg['data'].update(expected_prepared_sha256=digest, prepared_only=True,
                           local_sources=[], include_custom_threads=False, capture_ancestry_counts=True)
        cfg['baseline']['report'] = f'reports/{tag}_tfidf.json'
        configs[f'configs/{tag}_{kind}.json'] = cfg
    demo = copy.deepcopy(configs[f'configs/{tag}_lora.json'])
    demo['paths']['ensemble'] = f'reports/{tag}_ensemble.joblib'
    demo['paths']['report'] = f'reports/{tag}_ensemble.json'
    demo['training']['device'] = 'cpu'
    configs[f'configs/{tag}_ensemble.json'] = demo
    return configs


def prepare_english(source='cga', tag=None, anchor=None, new_split=False, max_turns=12):
    tag = tag or f'english_{source}_v1'
    if not re.fullmatch(r'english_[a-z0-9_]+', tag) or max_turns < 1:
        raise ValueError('Use a tag beginning english_ with lowercase letters, digits, underscores; max-turns >= 1')
    cfg = load_config('configs/contextual_clean_lora.json')
    cfg['data']['capture_ancestry_counts'] = True
    # Complete paths needed for depth and, for new splits, shared-ancestor grouping.
    cfg['model']['max_turns'] = 1_000_000
    stats = {}
    if source == 'cga':
        raw = load_cga(cfg, stats=stats)
        source_name = 'cga_wiki'
        anchor = anchor or 'data/processed/contextual_clean_v1.jsonl'
        provenance = [location(cfg, 'raw')/cfg['data']['cga']['directory']/name
                      for name in ('utterances.jsonl', 'conversations.json', 'provenance.json')]
    else:
        settings = source_settings()
        raw = convert_source(settings['sources'][0])
        source_name = 'toxichat_human'
        anchor = anchor or 'data/processed/external_clean_v1.jsonl'
        provenance = [project_path(item['path']) for item in settings['sources'][0]['files']]
        for row in raw:
            row.update(available_ancestors=len(row['texts'])-1,
                       depth_provenance='released_human_thread_prefix_not_full_reddit_tree')
    anchor_path = project_path(anchor)
    anchor_hash, excluded = None, {}
    if new_split:
        rows, excluded = fresh_split(raw, cfg, max_turns)
        policy = 'NEW grouped splits: historical mixed checkpoints may have seen these test texts; do not compare them here.'
    else:
        if not anchor_path.exists():
            raise FileNotFoundError(f'{anchor_path}: restore your cleaned benchmark, or explicitly use --new-split with a NEW tag.')
        anchor_hash = file_hash(anchor_path)
        expected_cfg = load_config('configs/contextual_clean_lora.json' if source == 'cga' else 'configs/external_clean_lora.json')
        if anchor_hash != expected_cfg['data']['expected_prepared_sha256']:
            raise ValueError('Anchor differs from the reviewed clean benchmark hash; do not mix benchmark versions.')
        rows = align_anchor(raw, read_jsonl(anchor_path), source_name, max_turns)
        policy = 'Exact cleaned English subset, existing IDs/labels/texts/groups/splits preserved; raw source adds depth only.'
    payload = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows)
    digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    configs = make_configs(tag, digest, max_turns)
    target = project_path(f'data/processed/{tag}.jsonl')
    # Refuse conflicting versions before any mutation; reruns of identical prep are safe.
    if target.exists() and file_hash(target) != digest:
        raise ValueError('Prepared version already exists with different content. Use a new --tag.')
    for path, value in configs.items():
        p = project_path(path)
        if p.exists() and json.loads(p.read_text(encoding='utf-8')) != value:
            raise ValueError(f'Config already edited: {path}. Preserve it; use a new --tag for new preparation.')
    manifest = {'tag': tag, 'source': source_name, 'prepared_sha256': digest,
                'anchor_path': str(anchor_path) if not new_split else None, 'anchor_sha256': anchor_hash,
                'split_policy': policy, 'max_turns_including_target': max_turns,
                'examples': len(rows), 'splits': dict(Counter(r['split'] for r in rows)),
                'depth_by_split': inventory(rows), 'max_available_ancestors': max(r['available_ancestors'] for r in rows),
                'window_truncated_examples': sum(r['available_ancestors'] > r['retained_ancestors'] for r in rows),
                'raw_sha256': {str(p): file_hash(p) for p in provenance if p.is_file()},
                'raw_skips': stats, 'new_split_exclusions': excluded,
                'training_started': False, 'configs': list(configs)}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    temporary.write_bytes(payload.encode('utf-8'))
    temporary.replace(target)
    for path, value in configs.items():
        write_json(project_path(path), value)
    write_json(project_path(f'reports/{tag}_manifest.json'), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['download', 'prepare'])
    parser.add_argument('--source', choices=['cga', 'toxichat'], default='cga')
    parser.add_argument('--tag')
    parser.add_argument('--anchor')
    parser.add_argument('--new-split', action='store_true')
    parser.add_argument('--max-turns', type=int, default=12)
    args = parser.parse_args()
    if args.action == 'download':
        if args.source == 'cga':
            print(download(load_config('configs/contextual_clean_lora.json')))
        else:
            fetch_sources(source_settings())
    else:
        print(json.dumps(prepare_english(args.source, args.tag, args.anchor, args.new_split, args.max_turns), indent=2))


if __name__ == '__main__':
    main()
