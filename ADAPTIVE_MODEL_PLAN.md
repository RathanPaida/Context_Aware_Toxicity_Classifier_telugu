# Final project run, results to return, and paper-submission checklist
Project: Context_Aware_Toxicity_classifier
Prepared: 17 November 2026
Workflow: verify software → score validation → freeze policy → score/evaluate test → evaluate Telugu challenge → verify demo → return evidence → write and submit the paper.

## Read this first
This is the final execution and reporting handoff, not a declaration that the new adaptive model has already improved. The main implementation was saved in your Windows project during the earlier work. Connection interruptions prevented confirmation of the final test suite. No production adaptive run or new full training run was completed by the assistant.

The current workspace used to create this document is separate from your Windows project. This document does not install missing Python files, models or data. Step 2 checks your local files; Step 3 verifies the implementation before expensive scoring.

Save this file in your project root as ADAPTIVE_MODEL_PLAN.md. Keep all old models, reports, prepared data and original run notes. Run the commands in PowerShell, one block at a time, in the Python environment that successfully completed your English experiment. Stop when a command fails; do not continue with missing or invalid outputs.

No additional dataset download or full retraining is needed for this comparison, provided your existing artifacts remain available. More data is a separate requirement if you want stronger claims about Telugu context, unseen domains or generalization to all parent depths.

## What this experiment is intended to resolve
| Issue | Required evidence |
|---|---|
| Several narrative numbers were incorrect | Regenerate tables from saved predictions; use the verified values in the paper |
| English ensemble failed on the Telugu development challenge | Compare all experts on the same challenge; inspect recall, false negatives, false positives and component scores |
| Different experts may suit different inputs | Compare the existing mixed ensemble, mixed-expert blending and conservative script-based routing/blending using validation only |
| Extra benefit from parent context remains uncertain | Compare real context with no context, report paired confidence intervals, and inspect reversed/unrelated-parent sensitivity |
| Results must match the submitted demo | Check the explicit adaptive artifact and the report returned by the running API |

The three experts are the saved English ensemble, saved mixed-language ensemble, and the neural component of the mixed-language ensemble. They are related models, not three independent training runs. The neural encoders already use Transformers and LoRA adaptation; another architecture is not required just to run this comparison.

## 1. Open the project and record the environment
Open PowerShell. Activate the environment you used for the successful English run. Do not perform a blanket package upgrade.

```powershell
Set-Location 'D:\NLP_projects\Context_Aware_Toxicity_classifier'
New-Item -ItemType Directory -Force reports/adaptive_v1 | Out-Null

python --version
python -c "import sys, torch; print('Python executable:', sys.executable); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>&1 | Tee-Object reports/adaptive_v1/environment.log
python -m pip freeze | Out-File -Encoding utf8 reports/adaptive_v1/packages.log
```

If CUDA is available and working:
```powershell
$runDevice = 'cuda'
```
Otherwise:
```powershell
$runDevice = 'cpu'
```

Keep the same terminal open so `$runDevice` remains defined. Close other training jobs you started yourself before scoring. If an earlier test process is still running, let it finish before starting another.

## 2. Confirm the saved implementation and inputs exist
```powershell
@(
  'contextual/adaptive_policy.py',
  'contextual/adaptive_experiment.py',
  'contextual/adaptive_inference.py',
  'configs/adaptive_experiment.json',
  'configs/demo_clean_ensemble.json',
  'configs/english_cga_v1_ensemble.json',
  'scripts/generate_verified_summary.py',
  'scripts/collect_adaptive_results.py',
  'scripts/verify_http.py',
  'tests/test_adaptive.py',
  'tests/test_contextual.py',
  'data/processed/contextual_clean_v1.jsonl',
  'data/processed/english_cga_v1.jsonl',
  'data/raw/custom_threads.jsonl',
  'reports/clean_ensemble.joblib',
  'reports/english_cga_v1_ensemble.joblib'
) | ForEach-Object { '{0}: {1}' -f (Test-Path -LiteralPath $_), $_ }
```

Every entry must show True. Both trained checkpoint directories, including their encoder files, must also remain where their configurations/artifacts expect them. Check the configured paths:

```powershell
Get-Content configs/adaptive_experiment.json
(Get-Content configs/demo_clean_ensemble.json -Raw | ConvertFrom-Json).paths
(Get-Content configs/english_cga_v1_ensemble.json -Raw | ConvertFrom-Json).paths
```

Do not create `configs/demo_adaptive_v1.json` by hand: the fitting step generates it. Do not edit an expected hash to bypass a mismatch. Missing code or checkpoints must be restored before continuing; downloading more text will not replace them.

Record the source state for reproducibility:
```powershell
git rev-parse HEAD | Out-File -Encoding utf8 reports/adaptive_v1/git_commit.log
git status --short | Out-File -Encoding utf8 reports/adaptive_v1/git_status.log
```

The Git commit alone does not identify uncommitted code changes. The results collector also includes relevant source files. Do not commit raw datasets, private text or large model weights just to satisfy this checklist.

## 3. Pass the software checks before scoring
Start with the focused tests:
```powershell
python -m pytest tests/test_adaptive.py tests/test_contextual.py::test_adaptive_saved_experts_roundtrip -q 2>&1 | Tee-Object reports/adaptive_v1/focused_tests.log
if ($LASTEXITCODE -ne 0) { throw 'Focused tests failed. Stop and return focused_tests.log.' }
```

Then check the full test suite:
```powershell
python -m pytest -q --junitxml=reports/adaptive_v1/tests.xml 2>&1 | Tee-Object reports/adaptive_v1/tests.log
if ($LASTEXITCODE -ne 0) { throw 'Tests failed. Stop and return tests.log and tests.xml.' }
```

These tests include a tiny synthetic model round trip. They do not retrain your production models. Preserve any skip reason; a skip is not a pass. The final all-pass result has not been confirmed by the assistant, so do not omit this step.

If a test fails, do not remove the assertion merely to continue. Send the failure and current files for review. If the process is interrupted, distinguish interruption from a completed failed test.

## 4. Regenerate the corrected existing-result tables
```powershell
python scripts/generate_verified_summary.py 2>&1 | Tee-Object reports/adaptive_v1/verified_summary.log
if ($LASTEXITCODE -ne 0) { throw 'Saved-result summary failed. Stop and return the log.' }
```

Open `VERIFIED_RESULTS.md` and `ENGLISH_POST_RUN_REVIEW.md` in the project root. The generated tables use saved predictions and thresholds, not manually copied narrative numbers. If an older prediction report is missing, restore it rather than inventing its numbers.

Correct your paper/report using these principles:
- Compare English models on the identical English test examples.
- Label overall F1 and contextual-only F1 separately.
- Name positive-class F1 explicitly; do not silently call it macro-F1.
- Average precision, ROC-AUC and ordinary accuracy are different metrics.
- Report precision and recall together. Better F1 may involve more false positives.
- The Tamil code-mixed source is not Telugu. Keep source names and label meanings accurate.
- English personal-attack labels, Telugu prompt-safety labels and Tamil offensiveness labels are not identical tasks. A shared binary encoding does not make them semantically equivalent.
- Preserve the original inaccurate run notes as history; use corrected tables for claims.

## 5. Export validation scores
```powershell
python -m contextual.adaptive_experiment score --split val --device $runDevice 2>&1 | Tee-Object reports/adaptive_v1/score_val.log
if ($LASTEXITCODE -ne 0) { throw 'Validation scoring failed. Stop here.' }
```
Expected file: `reports/adaptive_v1/val_scores.json`.

The code scores the saved English and mixed ensembles sequentially and extracts the mixed neural expert. It reuses encoded message vectors within an expert pass, records original/no-context/control scores and lexical component scores, and checks dataset/model identities. It does not change trained weights.

This is an inference job, not another five-epoch training job. It may still take substantial time on CPU. No fixed minute estimate has been established. Keep the progress log; elapsed scoring time is also recorded in the output. FP32 scoring can differ slightly from historical FP16 trainer scores.

## 6. Fit and freeze the policy using validation only
```powershell
python -m contextual.adaptive_experiment fit 2>&1 | Tee-Object reports/adaptive_v1/fit.log
if ($LASTEXITCODE -ne 0) { throw 'Policy fitting failed. Stop here.' }
```
Expected outputs:
- `reports/adaptive_v1/policy.json`
- `reports/adaptive_v1/selection.json`
- `configs/demo_adaptive_v1.json`

Inspect the chosen method:
```powershell
$selection = Get-Content reports/adaptive_v1/selection.json -Raw | ConvertFrom-Json
$selection.policy | ConvertTo-Json -Depth 10
$selection.selection_rule
$selection.limitations
```

### What the selector actually does
It partitions the existing validation population by connected conversation/shared-text groups into approximately 60% for policy fitting and 40% for policy selection, with seed 2026. Every source needs both classes in both partitions. Script-specific fitting requires at least 100 examples, 15 examples of each class and 15 connected groups in that fitting subset. The global mixed-expert candidate uses the pooled fitting population.

Candidate families include the unchanged mixed ensemble, global mixed-expert mixtures, script-specific hard routing and script-specific soft blending. The selection objective averages source-level positive-class F1 and contextual-subset F1. A candidate is ineligible if a source/context F1 falls by more than 0.01 relative to the original mixed ensemble on the selection partition. The search requires more than 0.005 objective gain before replacing its current winner. A threshold-only change or retention of the original mixed ensemble is a valid outcome.

The router detects scripts, not true language. Latin-only text can be English, Romanized Telugu or another language. The English expert is capped at 25% weight in the ambiguous-Latin soft blend and is not used as a pure hard route. Native Telugu/Tamil and mixed-script routes use mixed experts; unknown-script inputs retain the mixed fallback. These restrictions reduce one obvious routing risk but do not guarantee performance on unsupported languages.

Freeze the policy now. Do not change weights, thresholds, parent limits or preprocessing after seeing test/challenge results.

The base experts already used the original validation set for their own selection. The new partition does not undo that reuse. Earlier test results have also informed this project. Treat this follow-up comparison as exploratory; a fresh untouched benchmark is needed for stronger confirmatory generalization claims.

## 7. Score the held-out test population
```powershell
python -m contextual.adaptive_experiment score --split test --device $runDevice 2>&1 | Tee-Object reports/adaptive_v1/score_test.log
if ($LASTEXITCODE -ne 0) { throw 'Test scoring failed. Stop here.' }
```
Expected file: `reports/adaptive_v1/test_scores.json`.

The command requires a frozen policy and checks that the data and expert identities match it. Do not bypass these checks. Use the same device as validation where possible, and document any change.

## 8. Evaluate the selected method and parent-context evidence
```powershell
python -m contextual.adaptive_experiment evaluate 2>&1 | Tee-Object reports/adaptive_v1/evaluate_test.log
if ($LASTEXITCODE -ne 0) { throw 'Test evaluation failed. Stop here.' }
```
Expected files: `reports/adaptive_v1/evaluation.json` and `evaluation.predictions.json`. This step uses saved scores and does not need another encoder pass.

| Report field | Read it to answer |
|---|---|
| `test` | What are the selected policy's overall metrics? |
| `details.by_source` | Which source improved or regressed? |
| `details.contextual` | How does it perform on targets with retained parents? |
| `details.by_available_ancestors` | What is measured at each observed ancestry-depth group? |
| `expert_baselines` | How do the existing experts perform on the same examples? |
| `paired_vs_experts` / `paired_by_source` | What are the paired F1 differences and intervals? |
| `routing_counts` | How many targets fall into each script group, and with what metrics? |
| `without_context_fixed_route` | How does performance change when parents are removed? |
| `context_f1_interval` / `context_ap_interval` | Does estimated context improvement exclude zero? |
| `context_controls` | What happens with reversed or unrelated parents? |
| `component_diagnostics` | How are lexical scores distributed for positive/negative targets? |

The prediction file labels cases where context helped, hurt or did not change correctness. Use the IDs to inspect examples in your local data. Record representative errors without changing the frozen policy.

Routing is held fixed to the original input during the no-context/control comparison, so a route change is not confused with a context effect. Unrelated parents are drawn from training conversations of the same source; the target is unchanged. Reversal is evaluated only when at least two ancestors exist. Empty subsets should remain empty or not applicable, not be reported as 0% or 100% accuracy.

The controls reuse original labels. Since different parents can change a target's meaning, these are sensitivity checks, not independently annotated counterfactual accuracy. If an improvement interval includes zero, the evidence does not establish a reliable advantage. Bootstrap intervals do not eliminate prior test reuse, selection uncertainty, domain mismatch or small-subgroup limitations.

## 9. Evaluate the Telugu development challenge separately
```powershell
python -m contextual.adaptive_experiment score --challenge data/raw/custom_threads.jsonl --device $runDevice 2>&1 | Tee-Object reports/adaptive_v1/score_challenge.log
if ($LASTEXITCODE -ne 0) { throw 'Challenge scoring failed. Stop here.' }

python -m contextual.adaptive_experiment evaluate --challenge data/raw/custom_threads.jsonl 2>&1 | Tee-Object reports/adaptive_v1/evaluate_challenge.log
if ($LASTEXITCODE -ne 0) { throw 'Challenge evaluation failed. Stop here.' }
```
Expected files under `reports/adaptive_v1`:
- `challenge_scores.json`
- `challenge_evaluation.json`
- `challenge_evaluation.predictions.json`

Compare recall, false negatives, false positives, precision, F1 and AP against all three experts. Check native-script and ambiguous-Latin groups separately. Fewer missed toxic targets may come with more false alarms; include both outcomes.

The existing 51 authored targets are a development challenge, not an independent Telugu-wide benchmark. They must not be used to fit this policy. A zero-sized unrelated-parent control is expected when no same-source training donors exist. Component score distributions help investigate the English failure but do not alone establish a single causal explanation.

## 10. Verify the actual demo and benchmark display
After Step 8 has generated the benchmark report:
```powershell
python scripts/verify_http.py --config configs/demo_adaptive_v1.json --output reports/adaptive_v1/http_verification.json --startup-timeout 180 2>&1 | Tee-Object reports/adaptive_v1/http.log
if ($LASTEXITCODE -ne 0) { throw 'HTTP check failed. Preserve the logs and report the failure.' }
```

This starts a temporary local server, checks its selected artifact/report, requests a prediction and stops the server. The existing script has a separate prediction request timeout; `--startup-timeout` only extends startup. A first prediction may be slow because experts are loaded lazily. If it times out, return the logs rather than treating it as proof of an accuracy failure. The proposed separate `--prediction-timeout` option was not confirmed saved, so it is deliberately not used here.

For a manual demo:
```powershell
$env:TOXICITY_CONFIG = 'configs/demo_adaptive_v1.json'
python app/server.py
```
Open http://127.0.0.1:7860. Try an isolated target and the same target with genuine oldest-to-newest ancestors. Include English, Telugu script, Romanized Telugu and mixed-script text. Record surprising cases without tuning to them. Press Ctrl+C to stop the server before packaging results.

The adaptive output uses a threshold-normalized decision score with boundary 0.5. It is not a calibrated probability or confidence percentage. There is no single combined attention or embedding explanation across these experts; do not invent one in the paper.

## 11. Write a short run note and collect the evidence
Create `reports/adaptive_v1/run_notes.log` in a text editor and complete this template:
```
Run date:
Python version and environment:
PyTorch version:
Device used for validation/test/challenge:
GPU model and VRAM, or CPU and RAM:
Focused tests outcome:
Full tests outcome and skip reasons:
Selected policy name:
HTTP verification outcome:
Approximate time for validation scoring:
Approximate time for test scoring:
Approximate time for challenge scoring:
Errors, interruptions, retries or device changes:
Any code/configuration/data/checkpoint changes after starting:
Manual examples worth reviewing (language, example ID, expected behavior):
```

Do not manually retype long metric tables. Keep the exact generated JSON evidence.

```powershell
if (Test-Path scratch/submission_http.log) {
    Copy-Item -LiteralPath scratch/submission_http.log -Destination reports/adaptive_v1/server.log
}

python scripts/collect_adaptive_results.py
if ($LASTEXITCODE -ne 0) { throw 'Collection failed. Preserve the report folder.' }

Get-Content reports/adaptive_v1/return_manifest.json -Raw
```

Send `reports/adaptive_v1_results_for_review.zip`. The collector packages available score exports, selection/evaluation reports, logs, tests and relevant source, and excludes raw datasets and model weights. It includes this guide if you saved it in the project root under the stated filename.

Check `missing_expected` in the manifest. An empty list confirms expected main files were found, not that all checks passed. Confirm tests and HTTP verification separately. If a stage failed, send the partial package and the failing command; do not create placeholder results. Logs may contain local paths or text excerpts, so inspect them before public sharing.

## 12. Exactly what to return before paper writing
### Required
1. `reports/adaptive_v1_results_for_review.zip`.
2. The selected policy name and whether tests/HTTP verification passed.
3. `run_notes.log` with environment, timings and deviations.
4. The latest `VERIFIED_RESULTS.md` if it is not included in the ZIP.
5. Any traceback from a failed or interrupted stage.

### If you want a final paper-focused review in the same reply
Also provide the proposed paper title, target journal/conference or college template, page limit, submission deadline, and the claims you intend to make. Supply the original dataset/paper citations and license information you already collected; do not guess missing licenses. A proposed abstract or outline is helpful, but it is not necessary to write the entire paper before the results are checked.

### What the review will determine
- Whether the comparison supports using the adaptive method, a threshold change, or the existing mixed ensemble.
- Whether English gains come at a Telugu/source-specific cost.
- Whether parent context improves F1/AP reliably or remains inconclusive.
- Which numbers and tables are safe to use, and which claims must be limited.
- Whether any remaining issue blocks submission or can be disclosed as a limitation.

## 13. Recovery rules
| Problem | What to do |
|---|---|
| Missing code/checkpoint/data | Stop; restore the exact saved artifact or return the missing-file list |
| Import error | Activate the successful earlier environment; return the exception if it persists |
| CUDA unavailable | Use CPU; record the device change |
| GPU memory error | Stop other GPU jobs; if no completed score output exists, retry the failed stage on CPU |
| Completed score file already exists | Preserve it and continue if it belongs to this exact frozen experiment |
| Frozen policy already exists | Do not refit after examining test results; continue with the frozen policy |
| Hash/population mismatch | Stop; do not edit expected hashes to bypass the check |
| Interrupted scoring | Preserve the failed log; there is no per-batch resume, so rerun only the incomplete stage |
| Partial/invalid JSON after interruption | Preserve the partial file separately, confirm the stage failed, then rerun that stage |
| HTTP timeout | Return both HTTP and server logs; startup timeout does not extend prediction timeout |
| Baseline retained or context gain uncertain | Report it honestly; it is a valid experimental outcome |

For a deliberately new experiment, all output paths in a copied settings file must use a fresh namespace, and every command must receive that file via `--settings`. The provided collector currently targets `adaptive_v1`. Do not start another search after inspecting this test without documenting the exploratory iteration and arranging a fresh final evaluation.

## 14. Do you need more data before submission?
For a paper limited to an honest exploratory comparison of the existing datasets and challenge, no additional download is automatically required. Whether that scope meets your venue's requirements is a separate question; this checklist cannot guarantee acceptance.

For claims about reliable Telugu parent-context benefit, language identification, broad robustness or an untouched final benchmark, additional independent evidence is needed. More isolated toxic comments do not supply missing conversational supervision.

If expanding the evidence, follow this order:
1. Define a consistent target label and distinguish personal attacks, offensiveness and unsafe requests.
2. Obtain appropriately licensed or consented real conversation threads with stable IDs and genuine reply links. Use the existing `ENGLISH_EXPERIMENT_PLAN.md` for the project's earlier download resources; verify licenses before reuse.
3. Include native Telugu, Romanized Telugu, English and code-mixed inputs, both classes, and actual 0/1/2/3/deeper ancestor chains. Do not manufacture parents for isolated comments.
4. Use at least two suitable language annotators, blind to model predictions; document instructions, disagreement and adjudication. A context-dependence study needs separately assessed target-only and contextual views with a protocol limiting annotator carryover.
5. Split by connected threads/shared text, and keep near duplicates together. Separate fresh development data from a final untouched test set before choosing models or thresholds.
6. Use a pilot to assess class balance, agreement and uncertainty. A small pilot is not automatically a sufficient benchmark; determine sample size from the intended claim and observed variability.
7. Freeze the method on development data, evaluate once on the final set, and report sparse subgroups rather than claiming uniform performance.

You do not need to delay an honestly scoped paper indefinitely to add every model or dataset. You do need to narrow unsupported claims. A future language detector must be validated on Romanized/code-mixed text; recognizing script is insufficient.

## 15. Paper structure and tables to prepare after review
| Section | Include |
|---|---|
| Abstract | Task, data scope, implemented method, measured result, key limitation; write final numbers last |
| Introduction | Why target meaning may depend on genuine ancestors; concrete research questions |
| Related work | Verified original sources for contextual toxicity, datasets, encoder and LoRA; no invented references |
| Data | Source licenses, labels, language/script coverage, counts, exclusions, split method and leakage controls |
| Method | Normalization, encoder adaptation, turn aggregation, parent limit, lexical features, ensemble weights and policy selection |
| Experimental setup | Frozen hashes, hardware, package versions, seeds, thresholds, model-selection process and evaluation precision |
| Results | Same-population comparisons, per-source outcomes, confusion matrices, confidence intervals and context ablations |
| Error analysis | False positives/negatives, Romanized text, script ambiguity, context-helped/context-hurt cases |
| Limitations/ethics | Domain-label mismatch, validation/test reuse, small authored challenge, bias, data rights and misuse risks |
| Conclusion | Findings actually supported; separate future work from completed experiments |

Prepare these tables/figures from generated files:
1. Dataset/source table: train/validation/test counts, positives, labels, languages/scripts, context availability and split unit.
2. Existing English comparison on identical test examples: F1, AP, precision, recall, threshold and contextual-only F1.
3. Adaptive versus three expert baselines by source: counts, metrics, false positives/negatives and paired F1 differences/intervals.
4. Parent-context table: full versus removed context, observed depth counts, contextual F1/AP intervals and control sensitivity.
5. Separate Telugu authored-challenge table, prominently labelled development diagnostics.
6. Reproducibility table: devices, versions, seeds, runtime, hashes and exact configuration paths.

Do not treat depth groups as interchangeable randomized experiments. Depth is associated with different examples and may have tiny sample sizes. Preserve the earlier fixed-budget ablation results separately from the new grouped-depth metrics.

## 16. Final submission gates
- [ ] The focused and full software tests completed, and failures are resolved.
- [ ] Validation policy, test results and challenge results were produced with matching frozen identities.
- [ ] No test/challenge labels were used to fit the new policy; prior test-informed development is disclosed.
- [ ] Tables were generated from saved evidence and use correct populations and metric names.
- [ ] False positives and false negatives are both reported.
- [ ] Context benefit is described as uncertain if its interval includes zero.
- [ ] The 51-target challenge is not presented as a representative independent Telugu benchmark.
- [ ] Tamil results are not relabelled Telugu; different source label meanings are acknowledged.
- [ ] Software acceptance of inputs is not confused with demonstrated accuracy on them.
- [ ] The demo serves the stated model and displays its matching benchmark, or a demo limitation is explicitly reported.
- [ ] Dataset licenses, citations, preprocessing, exclusions and split methodology are documented.
- [ ] Reproducibility information includes uncommitted implementation changes where relevant.
- [ ] Any single-seed limitation is disclosed; no run-to-run stability claim is invented.
- [ ] No unsupported state-of-the-art, universal accuracy or causal attention claim appears.
- [ ] Venue formatting, anonymization, authorship, AI-use disclosure and artifact rules are checked against the actual submission instructions.
- [ ] The result package has been reviewed and remaining limitations incorporated into the manuscript.

The current trained model retains at most the latest 11 ancestors plus the target. A reply normally has one immediate parent and may have many ancestors; siblings are not extra parents. Accepting a longer history does not mean the model uses unlimited context. Do not claim “best results irrespective of number of parents” or “works accurately on any text.”

### Honest wording if results remain inconclusive
> We evaluated existing English and mixed-language experts and a validation-selected combination on fixed source-specific populations. Parent-context ablations showed the reported point differences, but intervals that include zero do not establish a reliable additional benefit. The authored Telugu challenge provides development diagnostics rather than a language-wide benchmark.
