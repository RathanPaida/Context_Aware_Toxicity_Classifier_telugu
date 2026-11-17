# English experiments, context depth, and comparison with Telugu

Prepared 17 November 2026. Run this guide from `D:\NLP_projects\Context_Aware_Toxicity_classifier`.

## 1. What to do, and what this experiment can establish

**Recommended next experiment:** train the existing model families on English CGA-Wiki only, keeping exactly the English train/validation/test assignments from your reviewed clean benchmark. Compare those English-trained models with your completed mixed-language models on the **same English test examples**. Separately show the existing Telugu results, with their different task and context availability clearly labelled.

Then measure performance for different actual ancestor counts and evaluate the same checkpoint with 0, 1, 2, 3, 5 and 11 retained parents. This will answer whether context helps and whether shorter histories are sufficient. It will not prove that all context lengths have equal accuracy.

Three corrections to the original goal:

1. **One direct parent versus many ancestors:** a reply normally has one direct parent. Its parent can have a parent, and so on. This project supports a root-to-target chain. Two separate direct parents form a different graph problem; the current `parent_id` API does not implement that. Do not turn sibling replies into parents.
2. **Arbitrary input history versus unlimited model context:** inference can receive a longer history, but the saved model retains the latest 11 ancestors and the target (`max_turns=12`). It reports omitted ancestors. It does not use every ancestor from an arbitrarily long conversation. Increasing that number in an old checkpoint's config will not extend its learned positional embeddings safely.
3. **Best results for every depth cannot be guaranteed.** Use held-out depth-specific metrics, sample counts, error analysis and uncertainty. A successful prediction on a long synthetic chain checks software behaviour, not long-context accuracy.

Do not spend your next run downloading millions of unrelated isolated comments. They cannot provide the missing long-context or Telugu conversation supervision. Your project already uses a BERT-family encoder, conversation-level Transformer attention, LoRA transfer learning and a validation-selected ensemble. Adding another architecture before a controlled comparison is lower priority.

## 2. What has been completed before your run

No production model training, full model evaluation, or new corpus download was started for this update. Preparation used the already-downloaded files; code tests use small synthetic fixtures.

The following changes are ready:

| Change | Purpose |
|---|---|
| `contextual/english_experiment.py` | Separate English downloads; exact clean English subset preparation; frozen data hashes; separate model configurations; raw ancestry audit. |
| `contextual/context_sweep.py` | Fixed-checkpoint parent-budget evaluation, per-depth/source metrics, context-benefit intervals, and optional serving-versus-saved-score comparison. One encoding pass is reused across budgets. Supports the existing ensemble and authored Telugu challenge. |
| `contextual/benchmark_compare.py` | Align saved predictions by IDs/labels/groups, reject mismatched populations, compare English models with historical mixed models, and produce paired group-bootstrap intervals. |
| `contextual/data.py` | Optional full released-path depth metadata, without changing the default historical data format. Full ancestry is still validated, even when only recent parents are retained. |
| `contextual/training.py`, `contextual/target_only.py` | Save dataset identity and per-example baseline scores required for comparisons. Target-only reporting also handles undefined one-class subset AP without crashing. |
| `scripts/collect_english_results.py` | Collect reports, configs, logs and relevant code into a review ZIP; exclude raw datasets and model weights. |

English data and four configs have already been prepared locally. Rerunning the preparation step with identical inputs is safe. It refuses conflicting data or edited configurations instead of overwriting them.

Verification: the project test suite passed with 85 tests and one optional legacy dependency test skipped. Small-fixture checks exercise the new exports, budget scoring, saved-score alignment and precision-parity reporting. The preparation rerun reproduced the same SHA-256. Historical English metrics were also recomputed from already-saved predictions, without loading a production model, in `reports/english_cga_v1_historical_reference.json`.

### Actual English dataset audit

The cleaned English subset has **22,832 examples: 15,775 train, 3,480 validation, 3,577 test**. It uses the existing CGA comment-level personal-attack labels. No Telugu or Tamil examples are in its training pool.

| Available ancestors in released path | Train examples | Validation examples | Test examples | Toxic test targets |
|---|---:|---:|---:|---:|
| 0 | 5,723 | 1,251 | 1,295 | 78 |
| 1 | 4,501 | 987 | 990 | 47 |
| 2 | 2,510 | 531 | 577 | 37 |
| 3 | 1,440 | 331 | 355 | 37 |
| 4–7 | 1,556 | 374 | 356 | 73 |
| 8–11 | 45 | 6 | 4 | 2 |
| 12+ | 0 | 0 | 0 | 0 |

The largest valid ancestry count is **10**, and no prepared English example is truncated by the 12-turn window. The 8–11 bin has only four test examples: do not make a robust performance claim for that bin. There is no empirical accuracy evidence for 11 or more ancestors in this dataset.

The raw loader found 22,959 eligible examples before matching the cleaned subset. It excluded 2,509 targets with missing ancestry, 543 with empty ancestry, and 177 with invalid ancestry. These are excluded target paths, not independent counts of broken conversations. Skipping an incomplete chain is deliberate; making the child a fake root would misrepresent the evidence. Root depth is relative to the released corpus after section headers are removed, not proof that the original website had no earlier context.

Prepared file: `data/processed/english_cga_v1.jsonl`.

SHA-256: `df39ad6827c55019bced4f34d37c6dda054b871c32c5f445b8d7cf8a45d259b3`.

Manifest: `reports/english_cga_v1_manifest.json`.

## 3. English dataset choices and download resources

Sources below were checked on 17 November 2026. A documented URL is not a guarantee that the host will be reachable from your network when you run it.

| Dataset | Context and labels | Decision |
|---|---|---|
| **CGA-Wiki** | Reply links and comment-level personal-attack labels. The official corpus includes 4,188 conversations and 30,021 utterances, including material our strict loader excludes. | Main controlled experiment. Already downloaded locally; archive is 45,200,295 bytes, about 43.1 MiB. |
| **ToxiChat, human turns only** | Released human Reddit conversation prefixes and Safe/Offensive labels. Separate generated chatbot continuations are present in the release. | Optional second English domain. Our local raw conversion has 2,000 roots, 2,000 one-ancestor targets and 623 two-ancestor targets, before cleaning. Do not use it to claim accuracy on 3+ ancestors. |
| **CAD: Contextual Abuse Dataset** | Expert annotations made with conversational context, Reddit reply relations, several abuse categories and non-abusive categories. | Most relevant next data candidate to inspect for more context variety. Depth coverage must be measured; do not assume it supplies many long chains. Separate acquisition/audit instructions are in Section 15. |
| **CivilCommentsInContext** | Context-aware annotations with supplied `parent_text`. 9,969 examples in a train-only release, with `contextual_toxicity`. | Useful optional one-parent study, not a solution to arbitrary-depth training. Requires a separate adapter and grouped splits; do not use `as_supervised=True`, which selects the ordinary toxicity field. |
| **CGA-CMV** | Reply trees, but the documented outcome is whether the final comment was removed for a rule violation. | Do not copy the conversation outcome onto every comment. It is not a drop-in replacement for comment-level toxicity labels. |

Primary resources:

- [CGA-Wiki documentation](https://convokit.cornell.edu/documentation/awry.html)
- [CGA-Wiki archive used by the project](https://zissou.infosci.cornell.edu/convokit/datasets/conversations-gone-awry-corpus/conversations-gone-awry-corpus.zip)
- [ToxiChat authors' repository, schema and CC BY-SA 4.0 licence](https://github.com/abaheti95/ToxiChat)
- ToxiChat pinned source revision already in `configs/data_sources.json`: `c1ffebd76596481b26aab35074816641b076f992`. The downloader uses the three files under `data/ToxiChat/` at that commit.
- [CAD original paper](https://aclanthology.org/2021.naacl-main.182/)
- [CAD original repository and errata](https://github.com/dongpng/cad_naacl2021)
- [CAD original release](https://doi.org/10.5281/zenodo.4881008)
- [CAD ConvoKit schema and download instructions](https://www.convokit.cornell.edu/documentation/cad.html)
- [CivilComments official catalog](https://www.tensorflow.org/datasets/catalog/civil_comments)
- [CGA-CMV label documentation](https://convokit.cornell.edu/documentation/awry_cmv.html)

Keep dataset attribution, version, download receipt and applicable terms with the submission. Do not redistribute raw corpora merely because a URL is public. The review collector excludes them.

## 4. Before executing anything

1. Keep the current `contextual_clean_*` checkpoints, `clean_ensemble.joblib`, cleaned data and old reports. They are the reference run.
2. Use the same Python environment that completed your previous LoRA run. Do not blindly upgrade Torch, Transformers or CUDA immediately before this experiment.
3. Open Anaconda PowerShell or your usual working terminal. This guide uses **PowerShell**, not notebook Python syntax.
4. Run:

```powershell
Set-Location 'D:\NLP_projects\Context_Aware_Toxicity_classifier'
python -c "import sys,torch,transformers; print(sys.executable); print(torch.__version__); print(transformers.__version__); print('CUDA',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m pip freeze > reports/english_cga_v1_environment.txt
git diff --stat
```

5. Close other GPU jobs. Keep the laptop plugged in and disable sleep for the training interval.
6. Allow enough disk space for **separate exported encoders, caches and resume checkpoints**. The four-GiB preflight floor is only a minimal failure check; several models can require substantially more. Check free space yourself before training multiple variants.
7. If using Jupyter, verify that the kernel's `sys.executable` is this interpreter. For long stages, the terminal is simpler to recover than a browser notebook kernel.

### Set up one-stage-at-a-time logging

Paste this function once into the current PowerShell session. It does not start a run. It logs stdout and stderr, reports duration and raises on a nonzero exit. Each call below runs only its named stage.

```powershell
function Run-PythonStep {
    param([string]$Name, [string[]]$PythonArgs)
    New-Item -ItemType Directory -Force -Path 'reports/runs' | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $log = "reports/runs/english_cga_v1_${Name}_${stamp}.log"
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & python -u @PythonArgs 2>&1 | Tee-Object -FilePath $log
        $runExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
        $timer.Stop()
    }
    Write-Host "Elapsed: $($timer.Elapsed); log: $log"
    if ($runExitCode -ne 0) { throw "Stage failed (exit $runExitCode). Read $log before continuing." }
}
```

A validation failure should stop dependent stages. No code can guarantee that downloads, GPU memory, drivers or power will never interrupt a run. The goal is clear errors, preserved earlier results, and recoverable stages.

## 5. Download English data separately

### Step 5A — CGA-Wiki only

```powershell
Run-PythonStep -Name 'download_cga' -PythonArgs @('-m','contextual.english_experiment','download','--source','cga')
```

Expected behaviour:

1. Verify and reuse `data/raw/contextual/cga-wiki.zip` if it is already valid.
2. Otherwise download to a temporary file, with bounded retries and size limits.
3. Validate the ZIP before accepting it, then extract the allowed data files.
4. Save source URL and archive SHA-256 in `data/raw/contextual/cga-wiki/provenance.json`.

Expected extracted files include `utterances.jsonl` and `conversations.json`. Do not call training from this step. A missing `conversations.json` or changed raw source can prevent exact benchmark alignment later; return the failure instead of replacing the historical benchmark.

If it fails: keep the log, check the source URL in a browser, and rerun this stage after the connection is restored. Do not manually remove parent references to make preparation pass.

### Step 5B — optional ToxiChat, independent of CGA

```powershell
Run-PythonStep -Name 'download_toxichat' -PythonArgs @('-m','contextual.english_experiment','download','--source','toxichat')
```

This command selects only ToxiChat from the external-source settings; it does not also download MACD Telugu. It verifies cached file receipts and parses downloaded JSONL before accepting a file. A failed file can be retried by rerunning the command. Files remain under `data/raw/external/toxichat_human/`; the separate receipt is `reports/english_toxichat_downloads.json`.

You can defer Step 5B until the main CGA experiment is complete.

## 6. Prepare and inspect English-only data

Run this even if the new data file is already present; it confirms reproducibility against the local inputs:

```powershell
Run-PythonStep -Name 'prepare' -PythonArgs @('-m','contextual.english_experiment','prepare','--source','cga')
Get-Content reports/english_cga_v1_manifest.json
```

Checks performed:

1. Read real CGA reply paths, ignoring section headers.
2. Validate all available ancestors, not just the model window.
3. Match raw targets to the existing clean English examples by ID, label, message IDs and normalized text.
4. Preserve all reviewed split memberships and groups. No new test examples are added in this mode.
5. Verify both classes in all three splits, no group crossing and no exact normalized message crossing between splits.
6. Record full available depth separately from the retained window.
7. Write a separate prepared JSONL, manifest and hash-locked configs.

Confirm the counts and SHA in Section 2. The four generated configurations are:

- `configs/english_cga_v1_lora.json`
- `configs/english_cga_v1_frozen.json`
- `configs/english_cga_v1_target_only.json`
- `configs/english_cga_v1_ensemble.json` — inference routing to the future English ensemble artifact.

The data hash is embedded in every config. Prepared data, embedding cache, checkpoint and report paths are separate from the completed mixed-language experiment. Frozen and target-only models share a compatible frozen embedding cache; LoRA does not share that cache.

Do not use `run_pipeline.py --all` or its generic `--prepare` for these configs. Use the specific preparation command above. The configs deliberately reject generic preparation.

### If the reviewed cleaned benchmark is missing

Preferred: restore `data/processed/contextual_clean_v1.jsonl` from your working project and confirm its expected SHA. It is required for a matched historical comparison.

If that is impossible, create a clearly separate experiment:

```powershell
python -m contextual.english_experiment prepare --source cga --new-split --tag english_cga_fresh_v1
```

This mode normalizes and deduplicates source paths, quarantines contradictory isolated labels, groups full paths before windowing, and generates new grouped splits. Use the `english_cga_fresh_v1` filenames throughout. **Omit all `--historical-reports` comparisons.** Old models may already have seen the newly assigned test examples. The new split is not an official benchmark split and is not directly comparable with published leaderboard numbers.

## 7. Code and environment checks

```powershell
Run-PythonStep -Name 'tests' -PythonArgs @('-m','pytest','-q','--junitxml=reports/english_cga_v1_code_tests.xml')
Run-PythonStep -Name 'preflight' -PythonArgs @('-m','contextual.preflight','--config','configs/english_cga_v1_lora.json','--require-gpu')
Copy-Item reports/preflight.json reports/english_cga_v1_preflight.json
```

The tests use tiny fixtures; they are not the English training run. Preflight checks prepared-data identity, splits, CUDA operation, disk space and the cached pinned encoder.

Do not start full LoRA training unless preflight passes. If CUDA is unavailable, return the report/log. Frozen heads and TF-IDF can run on CPU, but full encoder work can be much slower.

### If the pinned encoder is not cached

The generated configs reuse your existing IndicBERT revision and prefer local files. Download the exact revision in a **separate step**, then rerun preflight:

```powershell
Run-PythonStep -Name 'download_encoder' -PythonArgs @('-c',"from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='ai4bharat/IndicBERTv2-MLM-Sam-TLM', revision='bb783337859e3d7957de5ba82766ddea51e8fc3e'))")
```

Use your existing environment's Hugging Face configuration. Do not paste access tokens into logs or the review ZIP. This public model should not require you to hard-code a token.

## 8. Train the comparison models, one at a time

Keep the same prepared file, seed and encoder revision for the first pass. Fit on training only; select epochs/thresholds using validation. Existing trainers also produce test reports at completion, so this is an exploratory protocol once those reports are inspected. Avoid repeated test-driven tuning.

### Step 8A — target-only character TF-IDF + logistic regression

```powershell
Run-PythonStep -Name 'tfidf' -PythonArgs @('-m','contextual.experiments','--config','configs/english_cga_v1_lora.json')
```

Expected: `reports/english_cga_v1_tfidf.json`, `.predictions.json` and `.joblib`. It uses only target text. It is the inexpensive baseline against which additional computation must justify itself.

### Step 8B — target-only frozen IndicBERT + MLP

```powershell
Run-PythonStep -Name 'target_only' -PythonArgs @('-m','contextual.target_only','--config','configs/english_cga_v1_target_only.json')
```

Expected: `reports/english_cga_v1_target_only.json`, test and validation predictions, and `models/checkpoints/english_cga_v1_target_only/`.

This is an independently trained target-only baseline. Removing parents from the contextual model is a different experiment and cannot replace this baseline. The MLP checkpoint is not compatible with the conversation-model `Predictor`; use its saved predictions for this comparison, not the thread demo API.

### Step 8C — frozen IndicBERT + conversation Transformer

```powershell
Run-PythonStep -Name 'frozen' -PythonArgs @('run_pipeline.py','--config','configs/english_cga_v1_frozen.json','--train')
```

Expected: `reports/english_cga_v1_frozen.json`, `.predictions.json`, and its own checkpoint directory. This isolates the contribution of context modelling without updating the pretrained encoder. Its compatible embedding cache should be reusable after Step 8B.

### Step 8D — IndicBERT LoRA + conversation Transformer

```powershell
Run-PythonStep -Name 'lora' -PythonArgs @('-m','contextual.adapter_finetune','--config','configs/english_cga_v1_lora.json')
```

Initial settings retain the successful main-run recipe: five maximum epochs, batch size 8, encoder batch size 8, accumulation 4, FP16 on CUDA, gradient checkpointing and validation-based early stopping. Effective example batch size is approximately 32; the final partial accumulation can differ.

Expected outputs:

- `reports/english_cga_v1_lora.json`
- `reports/english_cga_v1_lora.predictions.json`
- `reports/english_cga_v1_lora.validation_predictions.json`
- `reports/english_cga_v1_lora.progress.json`, ending with `status: complete`
- `models/checkpoints/english_cga_v1_lora/metadata.json`, `head.pt`, `encoder/`, and resume state.

Check finite loss, validation AP, completion status and reload verification. Lower training loss alone does not prove improvement.

**If interrupted:** rerun exactly the same command/config to resume after the last saved epoch. Work in an incomplete epoch is repeated. Do not edit the config and expect an existing resume signature to match.

**If out of memory:** stop other GPU programs first. If a smaller batch is necessary, make a new variant config with a new checkpoint/cache/report path. Reduce example and/or encoder batch size (for example 8 to 4), increase accumulation if preserving effective batch size, and record that change. Do not overwrite the failed run or reuse its mismatched resume state. Return the actual error; an exit code alone does not establish OOM.

### Step 8E — validation-selected ensemble

```powershell
Run-PythonStep -Name 'ensemble' -PythonArgs @('-m','contextual.ensemble','--config','configs/english_cga_v1_lora.json','--neural-report','reports/english_cga_v1_lora.json','--output','reports/english_cga_v1_ensemble.json')
```

This uses character TF-IDF, a newly fitted word TF-IDF component and the LoRA model. Weights come from validation only, including single-model endpoints; a component can legitimately receive zero weight. The selector uses the mean of overall and contextual validation AP, then sets a validation F1 threshold.

Expected: ensemble report, `.joblib`, test and validation predictions. It does not automatically include the target-only MLP or frozen contextual model. Those are controlled baselines; adding every available model to the ensemble is not automatically beneficial.

Keep an ensemble only if its measured benefit justifies inference cost and errors. Do not force nonzero weights or choose weights from test results.

## 9. Compare models and existing Telugu results

Run after all five English reports exist:

```powershell
Run-PythonStep -Name 'compare' -PythonArgs @(
    '-m','contextual.benchmark_compare',
    '--config','configs/english_cga_v1_lora.json',
    '--reports',
    'reports/english_cga_v1_tfidf.json',
    'reports/english_cga_v1_target_only.json',
    'reports/english_cga_v1_frozen.json',
    'reports/english_cga_v1_lora.json',
    'reports/english_cga_v1_ensemble.json',
    '--historical-reports',
    'reports/clean_tfidf.json',
    'reports/contextual_clean_lora_evaluation.json',
    'reports/clean_ensemble.json',
    '--output','reports/english_cga_v1_comparison.json'
)
```

If intentionally skipping an English model, remove just its report argument and explain the omission. Do not copy another report into its filename.

The comparison checks data hashes and matches every test ID, label, group and source. Historical English examples must also have the same texts and splits. It then recomputes English metrics on the identical 3,577-example population and gives paired group-bootstrap F1 differences overall and on the 2,282 contextual examples. Shared text and conversation groups are resampled together.

### What the English-versus-Telugu table means

Your existing completed **mixed-training** results are:

| Model | English CGA test F1 (3,577 targets) | English contextual subset F1 (2,282 targets) | Telugu prompt-safety test F1 (1,787 targets; isolated) |
|---|---:|---:|---:|
| Character TF-IDF | 0.6035 | 0.6061 | 0.8374 |
| LoRA contextual model | 0.7394 | 0.7612 | 0.8336 |
| Ensemble | 0.7459 | 0.7713 | 0.8419 |

These are saved results, not new runs. English has 274 positive targets out of 3,577; Telugu prompt-safety has 569 out of 1,787. Labels, domains, prevalence and context availability differ. **The higher Telugu F1 does not prove that Telugu language modelling is better.** The Tamil code-mixed source is a third language dataset and must not be labelled Telugu.

The new comparison answers a different, valid question: on the same English holdout, does English-only training outperform the completed mixed-language recipe? Mixed models retain their original mixed-validation thresholds, while English models use English validation. Describe this as a comparison of complete training/selection recipes, not a pure causal effect of language mixture. AP helps assess ranking without the threshold difference, but still does not remove all training differences.

For each model report precision, recall, F1, average precision, ROC-AUC where defined, confusion matrix, sample count, positive count, training languages and context policy. Accuracy alone is misleading with rare attacks. Do not declare improvement from a small F1 increase if the paired interval crosses zero. The intervals do not correct repeated test inspection or all-pairs multiple comparisons.

## 10. Evaluate different context lengths

### Step 10A — validation sweep

```powershell
Run-PythonStep -Name 'budget_val' -PythonArgs @('-m','contextual.context_sweep','--config','configs/english_cga_v1_ensemble.json','--split','val','--budgets','0','1','2','3','5','11','--device','cuda','--reference-report','reports/english_cga_v1_ensemble.json','--output','reports/english_cga_v1_budget_val.json')
```

Use `--device cpu` if necessary, expecting slower encoding. The sweep freshly encodes unique messages once and reuses their vectors across context budgets. It does not retrain the encoder or fit thresholds.

Read the result as follows:

1. `overall` evaluates the same population at each retained-parent budget.
2. `contextual` excludes original roots, so roots do not dilute the measured context effect.
3. `by_available_ancestors` groups targets by their original released-path depth, even when the inference budget is zero.
4. `0` parents keeps only the target; it does not use an empty input.
5. A short natural chain is not padded with invented parents to reach a budget.
6. Empty bins are `null`; one-class AP/AUC are undefined; they are not failed model runs.
7. `contextual_budget_minus_zero_intervals` compares each budget with zero parents on paired examples.
8. Budget removal keeps the original target label. This is an information-removal robustness study; it does not mean annotators would assign the same label if shown a different context.

Keep the 11-parent cap for the initial comparison. If you later choose a cheaper cap using validation, record that choice before inspecting its test sweep. For example, prefer a shorter budget only if validation performance and recall are acceptable under a predeclared tolerance. Do not pick a separate best test budget for each depth.

### Step 10B — fixed test sweep

```powershell
Run-PythonStep -Name 'budget_test' -PythonArgs @('-m','contextual.context_sweep','--config','configs/english_cga_v1_ensemble.json','--split','test','--budgets','0','1','2','3','5','11','--device','cuda','--reference-report','reports/english_cga_v1_ensemble.json','--output','reports/english_cga_v1_budget_test.json')
```

Save the report and its `.predictions.json`. If explaining the neural component specifically, repeat with `configs/english_cga_v1_lora.json`, reference `reports/english_cga_v1_lora.json`, and output `reports/english_cga_v1_lora_budget_test.json`. The lexical components use no parents and can dilute an ensemble's sensitivity to context.

### Precision and deployment parity

The trainer's CUDA evaluation may use FP16. The sweep uses the saved predictor in FP32, matching its arithmetic policy, although device and batch size can still cause small numerical differences. The `serving_parity` section reports maximum/mean score differences and decision flips against the saved full-window predictions at the unchanged threshold.

If flips or metric differences are material, return both files. Do not tune a new threshold on test to hide the discrepancy. Before deployment, choose the runtime precision/device and validate its threshold on validation. This sweep does not by itself verify the HTTP server or CPU/GPU numerical identity.

## 11. Evaluate the actual ensemble on your Telugu challenge

The earlier 51-example Telugu/Romanized-Telugu result evaluated the neural model, not the ensemble. The new sweep supports both:

```powershell
Run-PythonStep -Name 'telugu_ensemble_challenge' -PythonArgs @('-m','contextual.context_sweep','--config','configs/demo_clean_ensemble.json','--challenge','data/raw/custom_threads.jsonl','--budgets','0','1','2','3','5','11','--device','cuda','--output','reports/english_shared_telugu_ensemble_challenge.json')
Run-PythonStep -Name 'telugu_lora_challenge' -PythonArgs @('-m','contextual.context_sweep','--config','configs/contextual_clean_lora.json','--challenge','data/raw/custom_threads.jsonl','--budgets','0','1','2','3','5','11','--device','cuda','--output','reports/english_shared_telugu_lora_challenge.json')
```

These use your completed mixed-language checkpoints. They do not train on the challenge. Keep the challenge marked **authored development examples**, not an independently annotated final benchmark. With only 51 targets, do not infer broad Telugu performance or choose production settings from it. Its value is finding failures such as a benign reply inheriting a toxic parent's label.

Optional diagnostic: run the English-trained ensemble on the same challenge with its own config and a distinct output name. Call it **English-training to Telugu challenge transfer**, not a fair language benchmark. Keep its threshold frozen; do not use the challenge for model selection.

## 12. What is needed for a defensible Telugu contextual comparison

More isolated Telugu data does not fill the contextual gap. MACD can improve or test isolated abuse detection, but cannot manufacture reply ancestry. Its project adapter maps source `0` (abusive) to project `1` (toxic), and source `1` (safe) to project `0` (safe). Source: [MACD authors' repository](https://github.com/ShareChatAI/MACD). Confirm source terms before redistributing it.

For the contextual claim, use the following collection plan:

1. Define one shared target-level annotation rubric for English and Telugu: directed personal attack/abuse versus non-abusive target. Decide explicitly how quotation, reclaimed language, profanity, counterspeech and indirect insults are treated.
2. Obtain real Telugu and Romanized-Telugu conversation chains from a source you are permitted to use. Preserve source/message IDs, conversation ID, direct `parent_id`, timestamps and provenance. Remove personal identifying details consistently where required.
3. Retain all known ancestors needed for interpretation. Never join unrelated comments, use future replies as context, or turn a missing parent into a fake root.
4. Record native Telugu, Romanized Telugu and code-mixed text separately. Do not silently translate English and call the result independently collected Telugu evidence.
5. Aim first for an annotation pilot of 200–300 targets across independent conversations, including 0, 1, 2, 3 and 4+ ancestors. This is a planning target, not a statistically guaranteed sufficient sample.
6. Have at least two fluent annotators label each target while seeing its actual prior chain; adjudicate disagreements. Record agreement and unresolved uncertainty. The target's label is not the parent's label.
7. Use the pilot's class balance, disagreement and group structure to plan the final sample. For a larger study, aim for roughly 1,000 or more targets and adequate positive/negative examples per reported depth bin. These are practical collection goals, not a power calculation. Combine sparse bins rather than inventing confidence.
8. Freeze an independent final test set by conversation/source and check exact plus near-duplicate overlap with all previous training and challenge data. Keep shared ancestors in one split. Do not use the final set to tune thresholds, budgets or ensemble weights.
9. Store messages in the existing node schema: `id`, `parent_id` (or null), `conversation_id`, `text`, and `label` for annotated targets; unlabelled ancestors may have `label: null`. Keep the annotations/provenance in an accompanying file. Use explicit language metadata as well.
10. Return schema, counts, class distribution, depth distribution, agreement/adjudication details, split policy and licence/access constraints before claiming cross-language generalization. A shared-schema adapter and final language-matched analysis can then be checked against actual files.

The proposed collection has not been performed. Access to private conversations, human annotation and decisions about consent are actions for you and your annotators; no such messages or data were requested from third parties here.

## 13. Optional second English domain: ToxiChat

After downloading it in Step 5B, keep it separate from CGA. Offensiveness and personal attack are related but different annotation targets.

If `data/processed/external_clean_v1.jsonl` from your previous work exists and matches its config's hash:

```powershell
python -m contextual.english_experiment prepare --source toxichat
```

This preserves the cleaned external benchmark's English subset. If that anchor does not exist, explicitly choose new grouped splits:

```powershell
python -m contextual.english_experiment prepare --source toxichat --new-split --tag english_toxichat_fresh_v1
```

Do not use both commands as successive versions under the same tag. Inspect the resulting manifest. The raw release has at most two preceding human turns; extracted title/body within the first turn is not a list of additional reply parents. Generated GPT/DialoGPT continuations are excluded.

Using the chosen tag, run:

```powershell
# Example for preserved external-clean splits; change the tag if you used --new-split.
python -m contextual.experiments --config configs/english_toxichat_v1_lora.json
python -m contextual.target_only --config configs/english_toxichat_v1_target_only.json
python run_pipeline.py --config configs/english_toxichat_v1_frozen.json --train
python -m contextual.adapter_finetune --config configs/english_toxichat_v1_lora.json
python -m contextual.ensemble --config configs/english_toxichat_v1_lora.json --neural-report reports/english_toxichat_v1_lora.json --output reports/english_toxichat_v1_ensemble.json
python -m contextual.context_sweep --config configs/english_toxichat_v1_ensemble.json --split test --budgets 0 1 2 --device cuda --output reports/english_toxichat_v1_budget_test.json
```

Run each line separately and check its exit status. Use the logging helper with a matching log prefix if collecting this experiment. Compare its model reports with `contextual.benchmark_compare` and the corresponding config, **without** `--historical-reports`: that option is specifically for the main cleaned CGA/mixed benchmark. Do not compare CGA test F1 and ToxiChat test F1 as a paired model difference.

For an eventual cross-domain transfer evaluation, train/select on one domain and score the other without fitting to its test labels. Check shared-text overlap across domains first. This protocol requires an explicit independent evaluation population and should be recorded separately from within-domain retraining.

## 14. Timing and interruption expectations

These are planning ranges, not measured runtimes for the new English experiment. Message length, token chunking, GPU, CPU, storage and caching matter more than cell count.

| Stage | Planning expectation | What to watch |
|---|---|---|
| Cached CGA validation/extraction | Seconds to a few minutes | ZIP validation, provenance receipt. |
| Fresh CGA download | Network-dependent; about 43 MiB locally | Bytes downloaded and retries. It is not a multi-gigabyte corpus. |
| English preparation | Usually seconds to a few minutes | Counts, exclusions, exact hash, no training. |
| Tests and preflight | Usually under a few minutes | Test failures, interpreter, CUDA and cache. |
| TF-IDF | Often minutes; CPU-dependent | Fit completion, validation threshold, report files. |
| Frozen/target-only | First encoding can take minutes or much longer on CPU; later shared-cache reuse saves time | Encoding progress, epoch validation AP, early stopping. |
| LoRA | Reserve a substantial GPU session; measure the first complete epoch and validation | Previous mixed run took about 109 minutes for five logged epochs; this is context, not an English runtime promise. |
| Ensemble | Usually CPU minutes after neural reports exist | Word model fit and validation weight selection; no new neural training. |
| Context sweep | One fresh encoding pass plus cheap head evaluations | Fresh encoding progress, device, budget messages. CPU can be slow. |
| Saved-score comparison and ZIP | Usually seconds to minutes | Matching IDs/hashes and missing-artifact list. |

Estimate remaining LoRA time from measured completed epochs plus validation, export and evaluation. Do not scale the old duration solely by example count: English conversation examples may require more encoding work per target.

If a stage fails, save the full traceback and command. Do not continue to dependent cells and then interpret missing artifacts as model failures. Separate download, preparation and training makes the cause much clearer than one long cell.

## 15. Additional data: inspect CAD before adding more architectures

CAD is the next candidate I would inspect. The primary paper describes context-aware expert annotations; the authors provide a corrected v1.1 release. The ConvoKit conversion documents reply relationships, original parent metadata and excluded split categories. Its documentation also lists CC BY 4.0. These are reasons to inspect it, not evidence that its longest chains are complete or plentiful.

### Download and audit steps

1. Open the authors' [release](https://doi.org/10.5281/zenodo.4881008) and [README](https://github.com/dongpng/cad_naacl2021). Prefer the corrected `cad_v1_1.tsv` plus codebook and errata for source-of-truth inspection. Download the data/codebook files, not all historical trained models. Do not run the release's old training environment inside your working environment.
2. Keep the original files in `data/raw/cad_original/`, with the landing-page URL, release version, access date, file hashes and accompanying licence. If the landing page presents an archive, inspect its contents before extraction and extract only needed data/documentation into that directory.
3. Optional convenience format: in a separate environment, install ConvoKit and use its documented download. This avoids altering the proven training environment:

```powershell
python -m venv .venv-cad-download
.\.venv-cad-download\Scripts\python.exe -m pip install convokit
.\.venv-cad-download\Scripts\python.exe -c "from convokit import download; print(download('contextual-abuse'))"
.\.venv-cad-download\Scripts\python.exe -m pip freeze > reports/english_shared_cad_download_environment.txt
```

4. Record the directory printed by the command. Preserve the downloaded `utterances.jsonl`, `conversations.json`, and metadata; do not overwrite CGA files. If this distribution/version is unavailable, use the authors' release instead and return the exact error.
5. Inspect actual label values and whether multi-label fields are strings, lists or delimited text. Preserve original categories. A possible **project-defined** binary mapping is the three directed-abuse categories to toxic, with neutral/counterspeech/non-hateful-slur-only cases non-abusive. Confirm this against the codebook and multi-label cases before implementation; do not map every non-neutral label to toxic.
6. Inspect `info_id.parent` alongside `reply_to`. The conversion can set an absent/invalid parent to null; treating that as a genuine root would overstate complete context. Count missing-parent chains separately. Preserve original parent IDs and flag unresolved ancestry.
7. Distinguish post titles, post bodies and comments. Title/body components are not automatically separate chronological replies. Decide whether they are root context or excluded targets and document the choice.
8. Respect `exclude_empty`, `exclude_bot`, `exclude_lang` and `exclude_image` source categories; do not blindly use these as labelled training targets. Check whether they are needed as unlabelled ancestry and whether that ancestry is usable.
9. Audit valid path depths 0,1,2,3,4–7,8–11,12+, class counts at each depth, same-text overlap across source splits, and malformed/cross-thread links. Keep complete conversations and shared messages together. If official splits leak contextual text, create clearly labelled new grouped splits instead of calling them official scores.
10. Return a text-only schema/summary report: filenames, hashes, column/metadata names, label-value frequencies, original split counts, missing-parent counts, depth counts and exclusion counts. Avoid sending private or restricted raw text. We can then validate a CAD adapter against the actual release and freeze a separate experiment.

**CAD training is not wired into the new English commands yet.** Its original/converted multi-label mapping and ancestry completeness need inspection before a reliable target-label adapter can be claimed. The implemented CGA/ToxiChat runs do not depend on this optional acquisition. Do not delay the controlled CGA experiment waiting for CAD.

### CivilComments alternative

Only if a one-parent contextual-toxicity study is useful, use the official TFDS `civil_comments/CivilCommentsInContext` configuration in a separate environment. Read dictionary records (`as_supervised=False`), use `contextual_toxicity`, preserve `article_id` and parent identity, predeclare a binary threshold such as 0.5 as your project choice, and group linked/shared texts before splitting. Report sensitivity to this binarization if needed. The catalog lists a roughly 427 MiB download even though the context subset is much smaller. Downloads previously failed from this project environment; return the exact current failure if it recurs. A current successful download was not tested here. No adapter or training result for this dataset is claimed.

## 16. Further model improvements, in priority order

1. **First finish the controlled experiment.** Check whether LoRA improves contextual AP/recall and whether the ensemble adds a reliable benefit. A simpler model can be the better submission if its performance is comparable and deployment is easier.
2. **Error analysis before architecture changes:** review validation false positives/negatives by depth, quotation, counterspeech, sarcasm, negation, benign replies to abusive parents, and long messages. Record whether the label or ancestry is uncertain. Do not repair test labels opportunistically to favour the model.
3. **Repeat a small, preregistered seed set**, such as 13, 42, 2026, only if time permits. Keep the prepared split/hash fixed. Give every seed separate checkpoint/cache/report paths; ensemble each seed using its validation data. Report mean/spread, not only the best seed. Repeated seeds quantify training variability, not new independent test evidence.
4. **Check calibration if probabilities will be presented to users.** Current scores are not proven calibrated probabilities. Fit any calibration on validation or a separate calibration set, then evaluate once; preserve ranking and report reliability/Brier-style results. The current new scripts do not fit a calibrator automatically.
5. **Consider XLM-R only after the baseline comparison.** See the [official model card](https://huggingface.co/FacebookAI/xlm-roberta-base). It is a multilingual Transformer candidate, not a guaranteed upgrade. Start with a frozen encoder comparison on the same split, pin its exact Hub commit, use separate caches/checkpoints, verify tokenizer chunking and dimension compatibility on a tiny fixture, then consider LoRA. Do not reuse IndicBERT embeddings or weights. English-only RoBERTa would be a different-language-capability comparison and would not solve Telugu contextual annotation.
6. **Longer history is a separate research variant.** First obtain adequate labelled long paths. Then train a new model with a larger positional window, or design hierarchical summarization/retrieval/streaming aggregation. Do not just edit `max_turns` in an existing saved model. Verify that any summary/retrieval uses ancestors only and does not omit the decisive abuse context. Larger attention windows increase memory cost.
7. **Context-length augmentation is optional, not automatically correct.** Removing a decisive ancestor can make an original context-dependent label impossible to infer. If trying random history truncation/dropout, make it a separate validation-controlled experiment with unchanged target provenance and explicit limits; do not relabel solely from model predictions.

For this submission, avoid a broad BERT/RoBERTa/DeBERTa/ensemble search with repeated test inspection. A well-supported comparison and an honest limitation section are more valuable than many unvalidated architectures.

## 17. What to include in the project report

1. Define the task as target-message classification given available prior reply ancestry; describe the specific label construct for each dataset.
2. Explain the pretrained encoder, per-message long-text chunk pooling, conversation Transformer, target bypass, LoRA and validation-selected ensemble. Attention weights are not a causal explanation.
3. Show the dataset table with source, language, labels, number of independent groups, examples, class balance, context-depth distribution, split policy and exclusions.
4. Show target-only lexical, target-only neural, frozen contextual, LoRA contextual and ensemble results on identical English test IDs.
5. Add historical mixed-versus-English-only results on the same English holdout; distinguish them from the separate Telugu prompt-safety comparison.
6. Include parent-budget and natural-depth tables with counts. Mark the four-example long-depth bin as insufficient and 12+ as untested.
7. Report precision/recall trade-offs, AP, confusion matrices, paired uncertainty and runtime. Include representative permitted validation errors with sensitive details removed.
8. State that the historical test has already been inspected and the new comparisons are exploratory. A truly final claim needs an independent, untouched test set.
9. Record model/data hashes, exact configs, package versions, seeds and evaluation device/precision.
10. Include limitations: cross-language label mismatch, little genuine contextual Telugu evaluation, missing ancestry exclusions, finite window, domain shift, rare long chains, uncalibrated scores and subgroup error uncertainty.

Do not write “works equally well for any number of parents,” “perfect multilingual toxicity detection,” or “ensemble definitely improves every language.” Use the measured scope instead.

## 18. What to return after running

First run the main CGA stages through comparison, validation/test sweeps, and the two Telugu challenge checks if possible. Then:

```powershell
python scripts/collect_english_results.py --tag english_cga_v1
```

Return **`reports/english_cga_v1_results_for_review.zip`**. Before upload, inspect the manifest's missing-file list and review logs for machine-specific or sensitive information. The package deliberately excludes raw text datasets and model weights. It includes code/config snapshots, report hashes and saved scores needed to reproduce the comparison.

Expected main evidence:

- English preparation manifest, preflight, environment snapshot and test XML.
- TF-IDF, target-only, frozen, LoRA and ensemble JSON reports.
- Their per-example prediction files; LoRA and ensemble validation predictions.
- LoRA progress/completion file and complete stage logs, including failed attempts.
- English matched-population comparison and paired intervals.
- Validation/test budget reports, predictions, and precision-parity results.
- Existing mixed-model source results included by the collector.
- Telugu ensemble/neural challenge reports if run.
- A short note giving GPU/VRAM, interpreter, total time per stage, any changes, skipped steps and what you observed.

You can write that note to `reports/english_cga_v1_run_notes.txt`; the collector includes it. For optional ToxiChat, collect a second ZIP with its actual tag. If returning CAD acquisition details, name the JSON summaries `reports/english_shared_cad_*.json` so they are picked up; attach its separate environment text if needed.

If a stage fails, **you do not need to wait for every other stage**. Return the exact command, full traceback/log, generated config, available manifest/preflight and the collector ZIP. Do not rename failed runs as completed runs or supply only a screenshot saying “unexpected status.”

With those files I can check whether the English-only recipe improved on the mixed model, whether parent context helped at each supported depth, whether the ensemble gain is reliable, whether serving precision changes decisions, and what is still needed before submission. No new performance improvement is claimed until those results exist.
