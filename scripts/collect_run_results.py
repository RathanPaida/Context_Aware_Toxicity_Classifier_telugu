"""Collect small review artifacts, not raw comments or large model weights."""
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contextual.config import write_json


def main():
    expected = ['reports/preflight.json', 'reports/preprocessing_audit.json',
                'reports/clean_tfidf.json', 'reports/contextual_clean_lora_evaluation.json',
                'reports/clean_ensemble.json', 'reports/clean_lora_challenge.json',
                'reports/submission_tests.xml', 'reports/submission_provenance.json',
                'SUBMISSION_REPORT.md', 'RUN_AND_REPORT.md', 'requirements-tested.txt']
    optional = ['reports/contextual_clean_lora_evaluation.progress.json',
                'reports/contextual_clean_frozen_evaluation.json',
                'reports/contextual_clean_target_only_evaluation.json',
                'reports/submission_verification.json', 'reports/submission_http_verification.json']
    optional += ['reports/clean_ensemble_uncertainty.json', 'reports/clean_ensemble.predictions.json',
                 'reports/clean_tfidf.predictions.json', 'reports/contextual_clean_lora_evaluation.predictions.json']
    files = [ROOT/p for p in expected+optional if (ROOT/p).is_file()]
    files += list((ROOT/'configs').glob('*.json'))
    files += sorted((ROOT/'reports/runs').glob('*.log'), key=lambda p:p.stat().st_mtime)[-8:]
    receipt = {'included': [str(p.relative_to(ROOT)) for p in files],
               'missing_expected': [p for p in expected if not (ROOT/p).is_file()],
               'note': 'Missing results remain missing. Review transcript paths before sharing. Raw text, credentials and model weights are not deliberately included; inspect logs before upload.'}
    write_json(ROOT/'reports/run_results_manifest.json', receipt)
    files.append(ROOT/'reports/run_results_manifest.json')
    destination = ROOT/'reports/run_results_for_review.zip'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in files: archive.write(p, arcname=str(p.relative_to(ROOT)))
    print(destination)
    print('Missing results:', receipt['missing_expected'])


if __name__ == '__main__': main()
