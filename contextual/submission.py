"""Generate an evidence-based submission report; never synthesizes missing scores."""
import json
import platform
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

from .config import ROOT, load_config, location, project_path, write_json
from .data import read_jsonl, examples_from_nodes
from .evaluation import file_hash, validate_splits


def build_report():
    cfg = load_config('configs/contextual_submission.json')
    rows = read_jsonl(location(cfg, 'prepared'))
    splits = validate_splits(rows)
    digest = file_hash(location(cfg, 'prepared'))
    custom = examples_from_nodes(read_jsonl(project_path(cfg['data']['custom_threads'])), cfg, 'challenge')
    packages = {}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'numpy', 'pandas', 'scikit-learn',
                 'fastapi', 'uvicorn', 'pytest', 'nbformat', 'nbclient', 'tokenizers', 'joblib'):
        try: packages[name] = version(name)
        except PackageNotFoundError: packages[name] = 'not installed'
    provenance = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                  'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages,
                  'prepared_sha256': digest, 'split_counts': {k: len(v) for k, v in splits.items()},
                  'challenge_targets': len(custom), 'challenge_positives': sum(e['label'] for e in custom),
                  'source_sha256': {str(p.relative_to(ROOT)): file_hash(p) for folder in ('contextual', 'configs', 'app', 'scripts', 'tests')
                                    for p in (ROOT / folder).glob('*') if p.suffix in ('.py', '.json')}}
    provenance['evidence_sha256'] = {p.name: file_hash(p) for p in (ROOT / 'reports').glob('*.json')
                                    if p.name != 'submission_provenance.json'}
    try:
        provenance['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError): pass
    write_json(ROOT / 'reports' / 'submission_provenance.json', provenance)
    lines = ['# Context-aware toxicity: submission report', '',
             'Current run instructions: [RUN_AND_REPORT.md](RUN_AND_REPORT.md). User-run results appear only when their report files exist; pending rows below are not measured results.', '',
             'This report is generated from JSON evidence by `python -m contextual.submission`.',
             'It distinguishes historical experiments from corrected, newly verified work.', '',
             '## Task and method', '',
             'The task is binary toxicity of a target reply given its actual ancestors. Individual message embeddings feed a hierarchical Transformer and a target-specific classifier. No domain selector or keyword verdict override is used. Long paragraphs use content-token-weighted chunk pooling.',
             '', '## Data and evaluation', '',
             f'Prepared data SHA-256: `{digest}`.',
             f'Splits: {len(splits["train"]):,} train, {len(splits["val"]):,} validation, {len(splits["test"]):,} test.',
             f'{sum(len(e["texts"]) > 1 for e in rows):,}/{len(rows):,} examples have ancestors.',
             'Conversations/articles and identical normalized message texts are kept together. Near-duplicates still need manual review.',
             'CGA labels personal attacks; Tamil labels offensiveness; Telugu labels prompt safety. These are related but different constructs.',
             'The existing test split has been inspected during development; treat comparisons as exploratory, not a fresh final generalization estimate.', '',
             '## Identical-population comparison', '',
             '| Model | Overall n | Overall F1 | Overall AP | Overall ROC-AUC | Context n | Context F1 | Context AP | Context FP |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    specs = [('TF-IDF (rerun)', 'submission_tfidf.json'),
             ('Target-only neural (historical)', 'target_only_neural_evaluation.json'),
             ('Context dropout 0.2 (historical)', 'contextual_candidate_evaluation.json'),
             ('Context dropout 0 (historical)', 'contextual_no_cdropout_evaluation.json'),
             ('LoRA v1 (historical)', 'contextual_lora_evaluation.json'),
             ('LoRA v2 (corrected)', 'contextual_lora_v2_evaluation.json')]
    for name, filename in specs:
        path = ROOT / 'reports' / filename
        if not path.exists():
            lines.append(f'| {name} | Pending | — | — | — | — | — | — | — |')
            continue
        r = json.loads(path.read_text(encoding='utf-8'))
        context = r.get('contextual_test', r.get('contextual_test_subset'))
        test = r['test']
        f = lambda value: f'{value:.4f}' if value is not None else '—'
        lines.append(f'| {name} | {test["n"]} | {f(test["f1"])} | {f(test["average_precision"])} | {f(test["roc_auc"])} | {context["n"] if context else "—"} | {f(context["f1"]) if context else "—"} | {f(context["average_precision"]) if context else "—"} | {context["confusion_matrix"][0][1] if context else "—"} |')
    lines += ['', 'TF-IDF remains the strongest measured overall baseline. Historical LoRA had the highest observed contextual-subset F1, but used different sampling and biased checkpoint selection. This does not prove superior reasoning or isolate a causal benefit of LoRA.',
              '', '## Development challenge', '',
              f'The curated Telugu/Romanized Telugu file contains {len(custom)} labelled targets; parent messages may be unlabelled. It is excluded from the frozen training dataset. Human annotation provenance is not established by the file schema; do not claim independent human validation without records.',
              'These challenge cases were designed around known weaknesses. They are a development diagnostic, not a representative independent test set.']
    for title, filename in [('LoRA v1 challenge', 'submission_lora_challenge.json'),
                            ('LoRA v1 contextual recheck with fresh embeddings', 'submission_lora_contextual_recheck.json')]:
        p = ROOT / 'reports' / filename
        if p.exists():
            r = json.loads(p.read_text(encoding='utf-8'))
            lines += ['', f'### {title}', '',
                      f'Population: `{r["population"]}`. n={r["test"]["n"]}. F1 with context={r["test"]["f1"]:.4f}; without context={r["context_removed_ablation"]["f1"]:.4f}.',
                      f'With-context confusion matrix [TN, FP; FN, TP]: `{r["test"]["confusion_matrix"]}`.']
    external = ROOT / 'reports/external_data_manifest.json'
    if external.exists():
        e = json.loads(external.read_text())
        lines += ['', '## Separate additional-data experiment', '',
                  f'{e["examples"]:,} validated examples, including {e["contextual_examples"]:,} with parents. Sources: `{e["sources"]}`. Splits: `{e["splits"]}`.',
                  f'{e["quarantined_conflicting_isolated_rows"]} isolated rows with conflicting labels were quarantined. Original benchmark membership was not changed.',
                  'Source URLs and pinned revisions are in `configs/data_sources.json`; file hashes are in `reports/external_downloads.json`.',
                  'These are new conversation/shared-text grouped splits, not the sources\' official splits. English ToxiChat human turns supply context; MACD Telugu supplies isolated comments.']
        baseline = ROOT / 'reports/external_tfidf.json'
        if baseline.exists():
            r = json.loads(baseline.read_text())
            lines += [f'Executed TF-IDF baseline: test n={r["test"]["n"]:,}, F1={r["test"]["f1"]:.4f}, AP={r["test"]["average_precision"]:.4f}; contextual subset F1={r["contextual_test"]["f1"]:.4f}.',
                      f'Telugu isolated-comment F1={r["by_source"]["macd_telugu"]["f1"]:.4f}. This is not Telugu contextual accuracy and is not directly comparable to the original benchmark.',
                      'External LoRA training is pending GPU recovery; no neural improvement is claimed from the additional data.']
    preprocessing = ROOT / 'reports/preprocessing_audit.json'
    if preprocessing.exists():
        clean = json.loads(preprocessing.read_text()).get('cleaned')
        if clean:
            lines += ['', '## Cleaned main dataset and new experiment', '',
                      f'{clean["examples"]:,} examples; splits: `{clean["audit"]["split_counts"]}`. Exclusions: `{clean["excluded_by_reason"]}`.',
                      f'SHA-256: `{clean["sha256"]}`.',
                      'NFC/control-character/whitespace normalization is shared with inference. Negation, script, emoji and joiners are preserved. No exact normalized-message leakage was detected. This is a different evaluation population from the historical table above.',
                      '', '| Cleaned-data model | Status | Test n | Overall F1 | Context F1 |',
                      '|---|---|---:|---:|---:|']
            for name, filename in [('Character TF-IDF', 'clean_tfidf.json'),
                                   ('Frozen IndicBERT + context attention', 'contextual_clean_frozen_evaluation.json'),
                                   ('Target-only neural', 'contextual_clean_target_only_evaluation.json'),
                                   ('IndicBERT + LoRA + context attention', 'contextual_clean_lora_evaluation.json'),
                                   ('Validation-selected voting', 'clean_ensemble.json')]:
                p = ROOT/'reports'/filename
                if not p.exists():
                    lines.append(f'| {name} | Not run | - | - | - |')
                else:
                    r = json.loads(p.read_text())
                    c = r.get('contextual_test', r.get('contextual_test_subset'))
                    context_f1 = f'{c["f1"]:.4f}' if c else '-'
                    lines.append(f'| {name} | Recorded | {r["test"]["n"]} | {r["test"]["f1"]:.4f} | {context_f1} |')
            ensemble_path = ROOT/'reports/clean_ensemble.json'
            if ensemble_path.exists():
                r = json.loads(ensemble_path.read_text())
                lines += ['', f'Voting weights: `{r["weights"]}`. Selection: {r["selection"]}',
                          f'Test F1 difference from the best validation-selected single component: {r["test_f1_difference_from_best_validation_single"]:+.4f}. This is a measured difference, not automatically evidence of generalization.']
            lines += ['', 'See `reports/clean_ensemble_uncertainty.json` after running the group-bootstrap analysis. Missing files mean the experiment or analysis has not completed.']
    verification = ROOT / 'reports/submission_verification.json'
    if verification.exists():
        verified = json.loads(verification.read_text())
        lines += ['', '## Execution verification', '',
                  f'Earlier notebook verification used **{verified["notebook_execution"]}**. These receipts describe the earlier default-mode run; they are not proof that the new cleaned experiment has run.',
                  'The historical CPU demo passed ASGI and real localhost HTTP checks; see `reports/submission_http_verification.json`. Verify the new artifact after your training run.',
                  'The GPU subsequently recovered. Full training was intentionally stopped on user request, and the user-run guide is RUN_AND_REPORT.md.']
        tests = ROOT/'reports/submission_tests.xml'
        if tests.exists():
            cases = ET.parse(tests).findall('.//testcase')
            skipped = sum(c.find('skipped') is not None for c in cases)
            failed = sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
            lines += [f'Latest regression receipt: {len(cases)-skipped-failed} passed, {skipped} skipped, {failed} failed/error. Tiny training fixtures do not establish full-data model quality.']
    lines += ['', '## Corrections and remaining limits', '',
              '- One shared differentiable chunk encoder is used for adapter training, validation and inference.',
              '- LoRA v2 uses full validation and the full training pool; class weighting matches that pool.',
              '- Adapter/head/optimizer/RNG state is saved atomically after each epoch and resumed only for matching data/configuration.',
              '- Cache identities include actual model parameters and tokenizer, preventing silent reuse across adapted encoders.',
              '- The corrected trainer checks merged-checkpoint reload predictions and reports context-removal metrics on exactly the same examples.',
              '- Thresholds maximize validation F1; this is threshold selection, not probability calibration or a guarantee of a chosen false-alarm rate.',
              '- A full corrected GPU run, repeated-seed comparison, independent annotation and fresh final test set remain required for stronger claims.',
              '- Local model weights/data are not committed to Git; the source distribution includes explicit setup and recovery instructions.',
              '', '## Submission instructions', '',
              'See [RUN_AND_REPORT.md](RUN_AND_REPORT.md) for the current complete run order and exact files to send back. [SUBMISSION_HANDOFF.md](SUBMISSION_HANDOFF.md) retains earlier dataset/access background. Historical PDF reports may be stale; this regenerated Markdown and JSON evidence distinguish measured work from pending experiments.']
    (ROOT / 'SUBMISSION_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return provenance


if __name__ == '__main__':
    print(json.dumps(build_report(), indent=2))
