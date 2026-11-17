"""Package English experiment evidence without datasets, checkpoints, or credentials."""
import argparse
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contextual.config import write_json
from contextual.evaluation import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='english_cga_v1')
    args = parser.parse_args()
    tag = args.tag
    if not re.fullmatch(r'english_[a-z0-9_]+', tag):
        parser.error('Invalid experiment tag')
    expected = [f'reports/{tag}_{name}.json' for name in
                ['manifest', 'preflight', 'tfidf', 'target_only', 'frozen', 'lora', 'ensemble', 'comparison', 'budget_val', 'budget_test']]
    expected += [f'reports/{tag}_environment.txt', f'reports/{tag}_code_tests.xml']
    files = set((ROOT/'reports').glob(f'{tag}*.json'))
    files.update((ROOT/'configs').glob(f'{tag}*.json'))
    files.update((ROOT/'reports').glob('english_shared_*.json'))
    files.update((ROOT/'reports').glob(f'{tag}*.txt'))
    files.update((ROOT/'reports').glob(f'{tag}*.xml'))
    files.update((ROOT/'reports/runs').glob(f'{tag}*.log'))
    files.update((ROOT/'contextual').glob('*.py'))
    files.update((ROOT/'tests').glob('test_english*.py'))
    for name in ['ENGLISH_EXPERIMENT_PLAN.md', 'POST_RUN_REVIEW.md', 'requirements-tested.txt',
                 'scripts/collect_english_results.py', 'tests/test_contextual.py']:
        if (ROOT/name).is_file(): files.add(ROOT/name)
    for stem in ['clean_tfidf', 'contextual_clean_lora_evaluation', 'clean_ensemble']:
        for suffix in ['.json', '.predictions.json']:
            path = ROOT/'reports'/(stem+suffix)
            if path.is_file(): files.add(path)
    manifest_path = ROOT/'reports'/f'{tag}_return_manifest.json'
    files.discard(manifest_path)
    receipt = {'tag': tag, 'missing_expected': [p for p in expected if not (ROOT/p).is_file()],
               'sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)},
               'note': 'No raw corpus or weights included. Source/code included for reproducibility. Review logs before sharing; missing files are not successful runs.'}
    write_json(manifest_path, receipt)
    files.add(manifest_path)
    output = ROOT/'reports'/f'{tag}_results_for_review.zip'
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT))
    print(output)
    print('Missing expected:', json.dumps(receipt['missing_expected']))


if __name__ == '__main__':
    main()
