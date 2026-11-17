"""Package adaptive validation/evaluation evidence, not model weights or raw text."""
from pathlib import Path
import json
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from contextual.config import write_json
from contextual.evaluation import file_hash


def main():
    expected=['val_scores.json','policy.json','selection.json','test_scores.json','evaluation.json',
              'challenge_scores.json','challenge_evaluation.json']
    directory=ROOT/'reports/adaptive_v1'
    directory.mkdir(parents=True,exist_ok=True)
    files=set(directory.glob('*.json')) | set(directory.glob('*.xml')) | set(directory.glob('*.log'))
    files.update((ROOT/'contextual').glob('*.py'))
    files.update((ROOT/'tests').glob('test_adaptive*.py'))
    files.update((ROOT/'configs').glob('*adaptive*.json'))
    for name in ['ADAPTIVE_MODEL_PLAN.md','VERIFIED_RESULTS.md','ENGLISH_POST_RUN_REVIEW.md',
                 'scripts/collect_adaptive_results.py','scripts/generate_verified_summary.py',
                 'scripts/verify_http.py','tests/test_contextual.py','app/server.py',
                 'reports/adaptive_development_tests.xml']:
        if (ROOT/name).is_file(): files.add(ROOT/name)
    manifest=directory/'return_manifest.json'; files.discard(manifest)
    write_json(manifest,{'missing_expected':[f for f in expected if not (directory/f).exists()],
        'sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in sorted(files)},
        'note':'Raw text and model weights excluded; inspect logs before sharing. Missing files remain missing.'})
    files.add(manifest)
    output=ROOT/'reports/adaptive_v1_results_for_review.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files): archive.write(path,path.relative_to(ROOT))
    print(output)


if __name__=='__main__': main()
