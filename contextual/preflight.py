"""Fast environment/data checks; never trains or downloads models."""
import argparse
import importlib.metadata
import json
import shutil
import sys

import torch
from huggingface_hub import snapshot_download

from .config import ROOT, load_config, location, write_json
from .evaluation import file_hash, validate_splits
from .data import read_jsonl


def check(cfg, require_gpu=False):
    digest = file_hash(location(cfg, 'prepared'))
    expected = cfg['data'].get('expected_prepared_sha256')
    if expected and digest != expected: raise ValueError('Prepared dataset hash mismatch; rerun the documented preprocessing step.')
    splits = validate_splits(read_jsonl(location(cfg, 'prepared')))
    cuda = torch.cuda.is_available()
    if require_gpu and not cuda: raise RuntimeError('CUDA unavailable; do not start full LoRA training.')
    if require_gpu:
        x = torch.ones(8, 8, device='cuda')
        assert torch.isfinite(x @ x).all()
        torch.cuda.synchronize()
    cache = None
    if cfg['encoder']['local_files_only']:
        name = cfg['encoder']['name']
        from pathlib import Path
        cache = str(Path(name).resolve()) if Path(name).is_dir() else snapshot_download(
            name, revision=cfg['encoder']['revision'], local_files_only=True)
    free = shutil.disk_usage(ROOT).free
    if require_gpu and free < 4 * 1024**3:
        raise RuntimeError('Less than 4 GiB free; make room for encoder and resume artifacts before training.')
    return {'python': sys.version, 'interpreter': sys.executable,
            'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
            'cuda_available': cuda, 'gpu': torch.cuda.get_device_name(0) if cuda else None,
            'prepared_sha256': digest, 'splits': {k: len(v) for k, v in splits.items()},
            'cached_encoder': cache, 'free_disk_gib': round(free/1024**3, 2),
            'checkpoint': str(location(cfg, 'checkpoint')), 'full_training_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/contextual_clean_lora.json')
    parser.add_argument('--require-gpu', action='store_true')
    args = parser.parse_args()
    result = check(load_config(args.config), args.require_gpu)
    write_json(ROOT/'reports/preflight.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
