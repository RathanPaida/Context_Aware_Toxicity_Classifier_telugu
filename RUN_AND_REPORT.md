# Complete run guide and what to send back

## 1. Current status — read this first

You asked me to finish the changes and let **you** run the project. The long training process was stopped. No complete corrected LoRA run or cleaned-data neural ensemble result is being claimed.

Completed before your run:

- Conservative multilingual preprocessing, duplicate removal, quarantine of ambiguous isolated labels and split-leakage checks.
- Consistent normalization during training and inference; negation, punctuation, emoji, Telugu script and joiners are preserved.
- IndicBERT transfer learning with LoRA adapters, a Transformer over conversation turns, attention pooling and a target-specific classifier.
- All-chunk encoding of long messages, full-validation checkpoint selection and epoch-level resume.
- Character TF-IDF and word TF-IDF baselines, optional neural soft-voting ensemble, validation-only weight/threshold selection, checkpoint identity checks and inference support.
- Group-bootstrap confidence intervals for ensemble F1 differences, generated only after model selection.
- Separate configurations for cleaned, historical and external datasets; historical results are preserved.
- A stage-by-stage PowerShell runner, environment preflight, progress JSON and result-collection ZIP.
- Small regression tests; no full experiment is required to test the code.

The GPU is now available. The earlier lost-GPU warning in older documents describes a previous session. The brief original-data v2 run was intentionally stopped before completing an epoch; it produced no completed v2 checkpoint. Its old progress file is marked interrupted.

An earlier character/word-only ensemble selected 100% character TF-IDF on validation, so it did **not** improve on that baseline. The cleaned-data ensemble with the newly trained transformer still needs your run. The code allows a single model to win; it never forces an ensemble or invents an improvement.

## 2. Use this dataset and configuration for the main run

Use **`configs/contextual_clean_lora.json`** throughout the main experiment below.

| Item | Expected value |
|---|---|
| Prepared data | `data/processed/contextual_clean_v1.jsonl` |
| Examples | 78,232 |
| Training | 54,578 |
| Validation | 11,776 |
| Test | 11,878 |
| Prepared SHA-256 | `368817cc6203e762d0a5b5a9f777761759fbf8769d0f3ec0b2d5dadb039b6bd6` |
| New checkpoint | `models/checkpoints/contextual_clean_lora/` |
| Neural report | `reports/contextual_clean_lora_evaluation.json` |
| Baseline report | `reports/clean_tfidf.json` |
| Ensemble report/artifact | `reports/clean_ensemble.json` / `.joblib` |

The preprocessing step removed 566 duplicate rows and quarantined 41 rows with conflicting isolated labels. The cleaned snapshot has no detected exact normalized-message overlap across train/validation/test. Near-duplicate meaning and annotation correctness still require human review.

The original benchmark has **78,839** examples and different test membership/counts. Do not directly compare a new cleaned-data F1 to an old original-data F1 and call the difference an architectural improvement. Compare the newly trained models on the **same cleaned split**.

The existing curated 51-target Telugu challenge remains outside training. It is a diagnostic, not an independently annotated final test set.

## 3. Environment setup

Open PowerShell or your Anaconda environment terminal:

```powershell
Set-Location D:\NLP_projects\Context_Aware_Toxicity_classifier
python -c "import sys; print(sys.executable); print(sys.version)"
```

Use the environment where your previous experiments worked. `requirements-tested.txt` records the versions used for testing; it is not a universal Windows/GPU lockfile. Do not replace a working CUDA installation unnecessarily.

If dependencies are missing, install the required packages in that environment:

```powershell
python -m pip install -r requirements-contextual.txt
```

The requirements now include `httpx[socks]`. This addresses the observed missing `socksio` dependency when a SOCKS proxy is configured. The main training config uses `local_files_only: true`, so an already cached encoder does not need a network call.

Check the GPU:

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU UNAVAILABLE')"
```

Proceed with full LoRA training only when CUDA is available. If the GPU is lost again, save your work, restart Windows and repeat these checks. No reboot or driver change was performed automatically.

If the pinned encoder is not in your cache, download it once from an environment with permitted network access:

```powershell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('ai4bharat/IndicBERTv2-MLM-Sam-TLM', revision='bb783337859e3d7957de5ba82766ddea51e8fc3e'))"
```

If your network, proxy, institution or provider blocks access, resolve that access locally and retry. Do not paste tokens/passwords into a notebook or send them to me. Download availability is not guaranteed by the code.

## 4. Small checks before the expensive run

```powershell
python -m pytest -q --junitxml=reports/submission_tests.xml
```

These tests use tiny synthetic fixtures for training checks. They do not train the real 78,232-example model. One optional legacy transliteration test may be skipped if `indic_transliteration` is not installed; this does not disable the active contextual pipeline.

If a test fails, stop and send the full failing traceback and `reports/submission_tests.xml`. Do not start a multi-hour run with an unresolved active-pipeline test failure.

## 5. Recommended execution: run each stage separately

Use **either** the PowerShell runner **or** the equivalent Python commands. Do not launch both copies, and do not run a training notebook simultaneously with CLI training.

### Stage A — reproduce and validate preprocessing

```powershell
.\scripts\run_submission.ps1 -Stage Prepare
.\scripts\run_submission.ps1 -Stage Check
```

Equivalent commands:

```powershell
python -m contextual.preprocessing --clean-output data/processed/contextual_clean_v1.jsonl
python -m contextual.preflight --config configs/contextual_clean_lora.json
```

Inspect `reports/preprocessing_audit.json`. Check `cleaned.examples`, `cleaned.audit.split_counts`, the SHA-256 above and zero `normalized_shared_text_groups_crossing_splits`. Inspect `reports/preprocessing_audit.excluded.json` for excluded IDs and reasons.

This step preserves the original dataset and does not merge new data into it. Do not run the generic `--prepare` against a cleaned configuration: that could reconstruct a different corpus, so the code rejects it.

If the original `data/processed/contextual.jsonl` is missing on a fresh clone, you need the exact permitted dataset snapshot or its original source inputs. Downloading CGA alone does not recreate the old mixture of Telugu/Tamil parquet data. Use the separate external-data experiment in section 10 if those inputs are unavailable, and describe it as a different dataset.

### Stage B — train the character TF-IDF baseline

```powershell
.\scripts\run_submission.ps1 -Stage Baseline
```

Equivalent:

```powershell
python -m contextual.experiments --config configs/contextual_clean_lora.json
```

Expected files:

- `reports/clean_tfidf.json`
- `reports/clean_tfidf.joblib`
- `reports/clean_tfidf.predictions.json`

This learns from training targets only, chooses its threshold on validation and evaluates the same cleaned test population used by the neural model.

### Stage C — train the contextual transformer with LoRA

```powershell
.\scripts\run_submission.ps1 -Stage Train
```

Equivalent:

```powershell
python -m contextual.preflight --config configs/contextual_clean_lora.json --require-gpu
python -u -m contextual.adapter_finetune --config configs/contextual_clean_lora.json
```

What the configuration does:

- Starts from the pinned pretrained IndicBERT encoder.
- Uses LoRA on query/value projections, rather than training every encoder parameter.
- Uses 32 training examples per batch, 32 encoder chunks per batch, mixed precision and gradient checkpointing.
- Trains up to five epochs, with validation-AP selection and early-stopping patience three.
- Uses all training examples and the complete validation set.
- Retains up to 12 actual ancestor/target turns. Every token chunk of each retained message contributes.
- Saves resume state after each completed epoch, then restores the best epoch and validates the merged saved encoder.

Watch progress in the terminal and `reports/contextual_clean_lora_evaluation.progress.json`. Training ETA excludes validation, final evaluation and checkpoint writing. The earlier brief run showed that batching substantially improves throughput, but a complete duration has not been measured. Use the observed first-epoch timing instead of treating an estimate as a deadline.

Success requires all these artifacts:

- `models/checkpoints/contextual_clean_lora/encoder/`
- `models/checkpoints/contextual_clean_lora/head.pt`
- `models/checkpoints/contextual_clean_lora/metadata.json`
- `models/checkpoints/contextual_clean_lora/resume.pt`
- `reports/contextual_clean_lora_evaluation.json`
- `reports/contextual_clean_lora_evaluation.predictions.json`
- `reports/contextual_clean_lora_evaluation.validation_predictions.json`

In the report verify `checkpoint_reload_verified: true`, `training_examples: 54578`, `validation.n: 11776`, `test.n: 11878` and the expected prepared hash. Contextual and context-removed scores must use identical example counts. A progress file alone is **not** a completed result.

### Stage D — select and evaluate the ensemble

Run this only after stages B and C succeed:

```powershell
.\scripts\run_submission.ps1 -Stage Ensemble
```

Equivalent:

```powershell
python -m contextual.ensemble --config configs/contextual_clean_lora.json --neural-report reports/contextual_clean_lora_evaluation.json --output reports/clean_ensemble.json
```

This fits a word TF-IDF classifier on training targets, combines it with character TF-IDF and the contextual transformer, and considers weights in increments of 0.25. The **predeclared selection objective** is the mean of overall validation AP and contextual-subset validation AP. This gives the project's context objective equal importance alongside overall performance. A single-model endpoint is eligible. The final threshold maximizes overall validation F1.

No test label chooses weights or thresholds. Test predictions are aligned by example ID, not assumed to have matching order. The saved ensemble verifies the exact neural checkpoint files during inference.

Inspect `reports/clean_ensemble.json`:

- `weights`: which components were actually selected?
- `validation_objective` and `validation_candidates`: how was selection made?
- `single_models`: same-test comparisons with each component.
- `best_single_by_validation`: the reference chosen without test selection.
- Overall/contextual F1, AP, precision, recall and confusion matrices.
- Context removal and `test_f1_difference_from_best_validation_single`.

If the contextual transformer's weight is zero, the selected artifact does not use context. It may be useful as a lexical model, but must not be presented as evidence of contextual reasoning. If an ensemble loses on the test set, report that outcome honestly; do not retune its weights using test results.

### Stage E — evaluate the authored Telugu challenge

```powershell
.\scripts\run_submission.ps1 -Stage Challenge
```

Equivalent:

```powershell
python -m contextual.recheck --config configs/contextual_clean_lora.json --output reports/clean_lora_challenge.json
```

This reevaluates the trained transformer on the existing 51 target examples, with and without parents. Keep the diagnostic separate from representative test accuracy. Do not train on these examples and continue to describe them as unseen evaluation.

### Stage F — uncertainty, final report and review ZIP

```powershell
.\scripts\run_submission.ps1 -Stage Report
```

Equivalent:

```powershell
python -m contextual.statistics
python -m contextual.submission
python scripts/collect_run_results.py
```

The bootstrap resamples connected conversation/shared-text groups and compares the selected ensemble with the best single component chosen on validation. It reports F1-difference intervals overall and on contextual examples. An interval crossing zero does not establish an improvement. These intervals do not repair the bias from repeatedly inspecting the historical test set.

Outputs include `reports/clean_ensemble_uncertainty.json`, `SUBMISSION_REPORT.md`, `reports/submission_provenance.json` and **`reports/run_results_for_review.zip`**.

### Optional: one command after the small checks pass

```powershell
.\scripts\run_submission.ps1 -Stage All
```

This intentionally runs the full sequence above and stops at the first failed stage. `-Stage Check` is the safe default. If PowerShell's execution policy blocks scripts, use the equivalent Python commands instead of changing that policy. Runner transcripts are saved under `reports/runs/`.

## 6. Notebook alternative

The CLI sequence is recommended for clear logs and resume behavior. If you prefer Jupyter:

1. Run preprocessing and the baseline first using stages A and B.
2. In the terminal that will start Jupyter:

```powershell
$env:TOXICITY_CONFIG = "configs/contextual_clean_lora.json"
$env:TOXICITY_RUN_LORA = "1"
python -m jupyterlab
```

3. Open `LoRA_Fine_Tuning.ipynb`, select the same environment and confirm the printed checkpoint ends in `contextual_clean_lora`.
4. Run its cells in order. Do not launch CLI training at the same time.
5. After training, run stages D-F from the terminal.
6. Turn training back off before using review notebooks:

```powershell
$env:TOXICITY_RUN_LORA = "0"
$env:TOXICITY_RUN_TRAINING = "0"
Remove-Item Env:TOXICITY_CONFIG -ErrorAction SilentlyContinue
```

`Context_Aware_Toxicity.ipynb` reviews evidence and inference. `Data_Downloads.ipynb` separates source downloads. Neither is a substitute for completing the cleaned-data training sequence above.

## 7. Check the saved model and demo after your run

For the new contextual transformer:

```powershell
$env:TOXICITY_CONFIG = "configs/demo_clean_lora.json"
python app/server.py
```

For the selected ensemble, stop the first server with Ctrl+C, then:

```powershell
$env:TOXICITY_CONFIG = "configs/demo_clean_ensemble.json"
python app/server.py
```

Open `http://127.0.0.1:7860` and `http://127.0.0.1:7860/api/health`. Verify `ready`, the checkpoint/artifact, predictor type, device and threshold. The demo configs use CPU for stable inference. The old `configs/demo.json` remains a historical fallback; choosing it does not demonstrate the new model.

Check a normal reply, direct abuse, benign reply to an abusive parent, quotation and a Telugu/Romanized Telugu thread. Report failures instead of adding rules to force those examples to pass. Attention values belong to the neural component and are not causal explanations of an ensemble verdict. Lexical-only artifacts have no neural embeddings or turn attention.

After stopping the server, clear the override if returning to ordinary commands:

```powershell
Remove-Item Env:TOXICITY_CONFIG -ErrorAction SilentlyContinue
```

## 8. If something stops in the middle

| Problem | Exact next action |
|---|---|
| Interrupted training | Rerun stage C with the identical config. It resumes the last completed epoch; an incomplete epoch repeats. Before the first completed epoch it starts again. |
| GPU out of memory | Stop other GPU programs. If necessary, copy the config and set training batch 8, encoder batch 8 and accumulation 4. Give the copy new checkpoint/report/cache paths. Use that same new config for its baseline, ensemble and later steps. Do not overwrite or mix resume directories. |
| Resume mismatch | Restore the config/data that created that resume file, or use a new run directory. Do not bypass the signature check. |
| CUDA unavailable or GPU lost | Save work and recover the GPU/driver before full training. A CPU demo can still run. |
| Missing cached encoder | Complete the one-time pinned download in section 3, then repeat preflight. |
| `socksio` error | Install `httpx[socks]` in the same environment, or use the already cached model with offline loading. |
| Prepared hash mismatch | Check the input snapshot and preprocessing version. Send both expected/actual hashes; do not edit the expected hash just to silence the error. |
| `UNEXPECTED cls.*` on base encoder loading | Those are the unused masked-language-model prediction head when loading `BertModel`. This observed case is expected. Missing encoder weights, tensor shape mismatches or different errors still need investigation. |
| Missing neural validation predictions | Stage C did not finish with the current trainer. Do not construct an ensemble from an unrelated report. |
| API 503 | Read `/api/health` and check encoder/head/metadata or the ensemble artifact exists. |
| Download 403/timeout | Retry the individual source, inspect its access requirements and retain verified completed files. Do not treat an HTML error page as data. |

No program can guarantee that hardware/network failures never interrupt a run. The project now makes failures explicit and protects completed work. Do not suppress an exception and continue with stale notebook state.

## 9. Exactly what to send me after the run

Send **`reports/run_results_for_review.zip`** and this short completed template:

```text
Run completed through stage: [A/B/C/D/E/F]
Config used: [exact filename]
Did you edit the config? [no / list every change]
Dataset version and SHA-256:
GPU and available VRAM:
Python executable and torch version:
Total runtime and seconds per epoch:
Training finished normally or resumed?:
Best validation epoch/AP:
Baseline overall F1/AP:
Neural overall F1/AP:
Neural contextual F1/AP:
Neural contextual F1 without parents:
Ensemble weights:
Ensemble overall and contextual F1/AP:
Ensemble F1-difference confidence intervals:
Telugu challenge F1 and confusion matrix:
Demo problems or surprising predictions:
Errors/warnings, if any:
```

The ZIP collects small reports, configs, prediction scores/IDs and recent runner logs. It does not deliberately include raw conversations or model weights. Inspect logs for local paths or sensitive text before sharing. Model weights, cache tensors, raw corpora and `resume.pt` are not needed for the first review.

If a run fails, **do not wait for the whole project**. Run:

```powershell
python scripts/collect_run_results.py
```

Then send the ZIP, the exact command, full traceback, last successful epoch/step and GPU status. `reports/run_results_manifest.json` lists missing expected results; missing results are not fabricated. If you used individual Python commands, copy the terminal output or save a log yourself because only the PowerShell runner creates transcripts automatically.

I will use those files to check data identity, leakage, validation selection, actual improvement over same-population baselines, false-positive/false-negative trade-offs, context benefit, uncertainty, and the claims suitable for submission.

## 10. Optional experiments and remaining external/human work

These are separate from the main run. Complete the main comparison before spending time on all of them.

### Frozen encoder and target-only neural controls

```powershell
python run_pipeline.py --config configs/contextual_clean_frozen.json --train
python -m contextual.target_only --config configs/contextual_clean_target_only.json
```

These use the same cleaned data. They help distinguish pretrained feature transfer, conversation attention and adapter tuning. Their training budgets differ from the LoRA budget, so report that limitation. A claim that a completely different BERT/Transformer architecture is superior requires an actual controlled experiment; no unsupported alternative-architecture gain is claimed here.

### Additional English context and Telugu comment data

```powershell
python -m contextual.external_data --download --prepare
python -m contextual.preprocessing --input data/processed/external_contextual.jsonl --report reports/external_preprocessing_audit.json --clean-output data/processed/external_clean_v1.jsonl
python -m contextual.experiments --config configs/external_clean_lora.json
python -m contextual.adapter_finetune --config configs/external_clean_lora.json
python -m contextual.ensemble --config configs/external_clean_lora.json --neural-report reports/external_clean_lora_evaluation.json --output reports/external_clean_ensemble.json
python -m contextual.statistics --config configs/external_clean_lora.json --report reports/external_clean_ensemble.json --output reports/external_clean_ensemble_uncertainty.json
```

The cleaned external snapshot contains 34,237 examples (24,036 train / 5,080 validation / 5,121 test). ToxiChat provides English human conversation turns; MACD provides Telugu isolated comments. MACD source labels are inverted into this project's convention: source 0 abusive becomes toxic=1. Repository-pinned URLs are in `configs/data_sources.json`; downloaded-file hashes and separate manifests are in `reports/`. These are new grouped splits, not official source benchmarks.

Resources: [ToxiChat](https://github.com/abaheti95/ToxiChat), [MACD](https://github.com/ShareChatAI/MACD), [CGA-WIKI](https://convokit.cornell.edu/documentation/awry.html). Preserve source attribution and verify redistribution terms before publishing raw data. Do not directly compare external-dataset scores against original/clean-main scores.

### CivilComments — optional access remains unresolved

The earlier storage download returned HTTP 403. Follow the detailed provider/download steps in section 5 of `SUBMISSION_HANDOFF.md` and the [official TFDS catalog](https://www.tensorflow.org/datasets/catalog/civil_comments). The contextual configuration is `civil_comments/CivilCommentsInContext`, with `contextual_toxicity` and parent text. Its ordinary supervised label key is not automatically the contextual label; inspect the fields explicitly. No unverified CivilComments adapter is connected to the main training run.

### Independent Telugu conversation evaluation — human work remains

Obtain permitted real Telugu/Romanized Telugu conversations; preserve ancestry and remove identifiers. Have two Telugu-capable annotators independently label the target while viewing its context, record disagreements and adjudication, and keep a new final test set untouched until model selection is finished. The existing authored challenge cannot establish human agreement or representative accuracy. No human labels, consent or agreement statistics were fabricated.

### Repeated seeds and final submission

For uncertainty across training runs, copy a selected config for seeds 42/43/44 with distinct checkpoint/report/cache paths while keeping the dataset hash fixed. Do not mix seed states in one resume folder. Report mean/spread and a fresh final-test evaluation before making strong generalization claims.

Finish a manual browser demonstration, review citations and license notices, prepare your institutional report/slides and submit through your own account. No paid compute, driver changes, external publication or institutional submission was performed. Package source, configs, reports and instructions; share only the selected checkpoint through a permitted large-file channel if offline inference is required.
