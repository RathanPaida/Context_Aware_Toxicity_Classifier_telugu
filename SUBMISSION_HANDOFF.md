# Earlier submission handoff and dataset background

**Superseded run order:** use [RUN_AND_REPORT.md](RUN_AND_REPORT.md). The GPU has recovered, training was intentionally stopped on user request, and the current main experiment uses the cleaned dataset. GPU-blocked statements and old run commands below document the earlier session; sections on optional access and annotation remain useful.

## 1. Current status

The code, data adapters, review notebooks, regression tests, baseline experiments and demo are implemented. The corrected full-size LoRA experiments are **not trained yet**. This machine's NVIDIA driver reported a lost GPU and requested a reboot; PyTorch reported `cuda.is_available() == False`. CPU inference and small training tests work. No reboot, paid compute purchase, account login or publication was performed.

Use `SUBMISSION_REPORT.md` for measured conclusions. Do not submit the older audit/PDF as the final results without correcting its claims. Historical models, reports and the completed-run notebook were preserved. Existing user edits were not reset.

Completed work:

- Fixed shared long-text encoding for training, validation and inference; every content-token chunk contributes.
- Changed LoRA checkpoint selection to the full validation set and training to the full training pool, with matching class weights.
- Added atomic epoch checkpoints, optimizer/RNG resume, merged-model reload verification and fixed-data checks.
- Made embedding caches depend on actual encoder weights and tokenizer, including adapted weights.
- Corrected comparison populations: the 2,291 examples with parents are distinct from all 3,612 CGA test examples.
- Reran TF-IDF and independently rechecked the existing LoRA checkpoint with fresh embeddings.
- Evaluated the 51 authored Telugu challenge targets; retained them outside the original training benchmark.
- Downloaded, hashed and prepared ToxiChat human turns and MACD Telugu in a separate experiment, and trained its TF-IDF baseline.
- Added separate download, review and LoRA notebooks. Default review mode does not start training.
- Configured an explicit existing CPU demo checkpoint and exposed checkpoint/device/threshold in API health.

Measured evidence is in `reports/`. `submission_tests.xml` records 72 passed and one skipped test. The skip is the **optional legacy Telugu transliteration dependency** (`indic_transliteration`), not a skipped test of the active contextual pipeline. `submission_verification.json` records notebook execution method, cell timings and real-checkpoint API checks. Consult these generated files if rerunning changes the counts.

## 2. Run the current project now

Use the existing Python environment that ran your experiments. Keep its working installation before attempting package upgrades.

```powershell
Set-Location D:\NLP_projects\Context_Aware_Toxicity_classifier
python -c "import sys; print(sys.executable)"
python -m pytest -q
python scripts/verify_submission.py --live-kernel
python scripts/verify_http.py
python -m contextual.submission
```

All three notebooks passed **live Jupyter-kernel execution** in their default modes. `verify_submission.py --live-kernel` repeats that check and writes outputs/timings. Without the flag it uses sequential in-process Python, useful when kernel sockets are unavailable, and labels that method explicitly. It downloads only when a source is not already verified and forces training off. It also tests the real saved checkpoint through FastAPI's in-process ASGI client. `verify_http.py` separately starts a bounded localhost server, checks actual HTTP responses and stops its own server; this also passed. Neither script claims a visual browser test.

For interactive notebooks:

```powershell
$env:TOXICITY_RUN_TRAINING = "0"
$env:TOXICITY_RUN_LORA = "0"
Remove-Item Env:TOXICITY_CONFIG -ErrorAction SilentlyContinue
python -m jupyterlab
```

1. Open `Context_Aware_Toxicity.ipynb` and select this same Python interpreter/kernel.
2. Restart kernel and Run All. Review the actual inference checkpoint printed in the output.
3. Open `Data_Downloads.ipynb` for separate CGA, ToxiChat and MACD download cells.
4. Keep `LoRA_Fine_Tuning.ipynb` in review mode until GPU recovery.

For the web demo:

```powershell
$env:TOXICITY_CONFIG = "configs/demo.json"
python app/server.py
```

Open `http://127.0.0.1:7860`, `http://127.0.0.1:7860/docs` and `http://127.0.0.1:7860/api/health`. The current demo explicitly uses the historical no-context-dropout checkpoint on CPU. This is not the untrained LoRA v2 model. Use a second terminal for other commands. Stop the server with Ctrl+C when finished. If port 7860 is already used, stop your earlier server or change `server.port` in the demo config.

Confirm health shows `ready: true`; enter a parent and target; verify thread and isolated scores are returned. The frontend does not require a conversation-topic selector. A missing model returns 503, not a made-up score. ASGI and actual HTTP checks have been automated; this browser interaction remains a short manual demonstration check.

For another machine, install `requirements-contextual.txt` in a fresh environment and use `requirements-tested.txt` as the recorded version reference. It is not a cross-platform lockfile; the recorded CUDA-specific PyTorch wheel may need a platform-compatible replacement. The legacy `src/` tests can require additional dependencies from the older environment. Do not mix legacy cascade checkpoints with the active `contextual/` pipeline.

## 3. Required hardware recovery and corrected full LoRA run

This is a hardware/runtime blocker, not a missing dataset permission. I cannot complete a safe reboot while preserving your active applications. Perform these steps yourself:

1. Save all notebooks and other work. Stop running kernels and GPU applications.
2. Restart Windows normally.
3. Reopen the same environment and project directory.
4. Check the GPU before starting training:

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU UNAVAILABLE')"
```

5. Continue only if `nvidia-smi` succeeds and PyTorch prints `True`. If the lost-GPU message persists, resolve the NVIDIA driver/device issue or use an institution-provided GPU. Account access, driver changes, paid compute and file transfer to an external machine must be handled in your environment; no accounts or paid services were used here.
6. Run the corrected full experiment:

```powershell
python -m contextual.adapter_finetune --config configs/contextual_lora_v2.json
```

Equivalent: `python run_pipeline.py --config configs/contextual_lora_v2.json --train`. For this frozen benchmark, do not rebuild the dataset during the run. The config verifies its SHA-256.

7. Progress reports training steps and estimated remaining training time, followed by full validation progress. The estimate is observed throughput, not a promised duration. Full training may take many hours; the first epoch determines a useful estimate.
8. If interrupted, run exactly the same command. `resume.pt` restores the last **completed epoch**; work in the interrupted epoch is repeated. Recovery is not per-batch. If no epoch completed, training restarts. A mismatched configuration/data hash is rejected.
9. If memory is insufficient, make a **new config and new checkpoint/report paths** with a smaller batch size; increase gradient accumulation if maintaining effective batch size. Changing config in an existing resume directory is deliberately rejected. Do not delete historical models to fix a resume mismatch.
10. Success produces `models/checkpoints/contextual_lora_v2/{encoder,head.pt,metadata.json,resume.pt}`, `reports/contextual_lora_v2_evaluation.json` and its predictions file. Check `checkpoint_reload_verified`, full validation count **11,893**, training count **54,960**, test count **11,986** and contextual test count **2,291**.
11. Regenerate `python -m contextual.submission`. The report should replace the v2 Pending row with measured scores.

Until these steps finish, describe the result as a tested implementation with historical model evaluations. Do not claim that corrected v2 training improved accuracy.

## 4. Data already available and downloaded

The original benchmark remains 78,839 examples (54,960 train / 11,893 validation / 11,986 test), with 14,598 examples containing ancestors. It combines English personal-attack, Tamil offensiveness and Telugu prompt-safety labels. These are different constructs. High prompt-safety performance does not prove Telugu conversational moderation.

The separate external experiment now contains:

| Source | Prepared examples | What it provides |
|---|---:|---|
| ToxiChat human turns | 4,560 | English conversation turns and offensive-language labels |
| MACD Telugu | 29,996 | Telugu isolated comments, abusive/non-abusive labels |
| Total | 34,556 | 2,623 targets have preceding turns |

External split counts: 24,275 train / 5,113 validation / 5,168 test. Sixty-seven isolated rows with contradictory labels were quarantined. Conversation/shared-text groups define **new splits**; these are not official benchmark scores. Source split membership remains in each example as provenance. Generated ToxiChat chatbot continuations are excluded.

Download/reproduce these sources with:

```powershell
python -m contextual.external_data --download --prepare
python -m contextual.experiments --config configs/contextual_external.json
```

Completed files are checked against receipts and reused. Partial downloads are not accepted as complete. On a network failure, rerun the affected notebook source cell or the command; other verified files remain available. Download status is not a prediction score.

The exact commit-pinned URLs and label maps are in `configs/data_sources.json`. Download hashes are in `reports/external_downloads.json`. Preparation counts/hash are in `reports/external_data_manifest.json`.

After GPU recovery, the optional separate neural experiment is:

```powershell
python -m contextual.adapter_finetune --config configs/contextual_external.json
```

Compare it only against `reports/external_tfidf.json` on these exact external splits. Do not compare its raw F1 against the original benchmark and call the difference an improvement. It starts from the configured pretrained encoder, not from the historical LoRA checkpoint. The frozen original benchmark has not been merged with these new data.

Dataset resources and attribution:

- [CGA-WIKI documentation](https://convokit.cornell.edu/documentation/awry.html): authentic reply structure and per-comment personal attacks. The loader counts and excludes missing/invalid ancestor chains instead of fabricating missing text.
- [ToxiChat authors' repository](https://github.com/abaheti95/ToxiChat): data and citation for *Just Say No: Analyzing the Stance of Neural Dialogue Generation in Offensive Contexts*. Repository license is CC BY-SA 4.0. Preserve its attribution/license when redistributing applicable material.
- [MACD authors' repository](https://github.com/ShareChatAI/MACD): the README defines **0=abusive, 1=non-abusive**; this project's adapter reverses them to toxic=1. An explicit redistribution license was not identified in the viewed repository root. Include download instructions rather than republishing its corpus, and obtain clarification from the maintainers if redistribution is required.
- [IndicBERTv2 model](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-Sam-TLM): encoder and model-card attribution. The configured revision is pinned; retain upstream notices when sharing weights.

The existing optional Telugu/Tamil parquet sources are reused locally. A fresh clone without those files will not recreate the same 78,839-example benchmark automatically; the expected hash should fail. Transfer the exact permitted dataset snapshot, or reconstruct it from its documented original sources and save a **new version/config**, not an allegedly identical reproduction. The standalone external experiment has a reproducible public download path and does not require those parquet files.

## 5. Optional large-data download: CivilComments

CivilComments is optional, not required for the working demo or baseline. Its storage URL returned **HTTP 403** during this session. More English data also will not fill the Telugu-context gap by itself.

1. Read [the official TensorFlow Datasets entry](https://www.tensorflow.org/datasets/catalog/civil_comments) and select the context configuration if studying contextual toxicity.
2. Use a separate data-preparation environment so TensorFlow dependencies do not disturb the working PyTorch environment. Reserve several GB of disk space for downloads and extraction.
3. Install `tensorflow-datasets` and a compatible TensorFlow package in that environment.
4. Run the following Python code there:

```python
import tensorflow_datasets as tfds
builder = tfds.builder("civil_comments/CivilCommentsInContext")
print(builder.info)
builder.download_and_prepare()
dataset = builder.as_dataset(split="train", as_supervised=False)
example = next(iter(tfds.as_numpy(dataset)))
print(example.keys())
```

5. If access still returns 403, use the source/manual-download instructions linked from the official builder and any account/terms workflow required by the provider. Do not save the HTML error page as a ZIP, disable validation or copy credentials into the notebook. These access steps could not be completed here.
6. Preserve comment IDs, parent IDs/text, article identifiers, annotation fractions and source version. Use the documented contextual target label for a contextual-label experiment; decide a binary threshold on the label fraction in the protocol before inspecting outcomes. Do not silently substitute the ordinary non-contextual annotation.
7. Write a new adapter and unit tests against the actual downloaded schema, checking parent existence, label coverage, duplicates and parent/target relationships. It is intentionally not wired into the current training config because the actual data download could not be inspected here.
8. Group connected replies/articles and normalized shared texts before splitting. The official documentation warns that ordinary supplied splits were not constructed around parent context. Keep a new data manifest and train comparable baselines on that same dataset.
9. Run this as a separate experiment first; preserve all original result files. Include CC0/source attribution and configuration information in the report.

## 6. The most useful remaining data work: independent Telugu conversation labels

The 51 existing authored challenge targets (32 safe / 19 toxic) are useful diagnostics. They are not an independently collected, representative final test set. The model's F1 on this challenge was about 0.596, with 14 false positives and 5 false negatives. Do not tune repeatedly on these examples and then call them an untouched test.

1. Define the project label precisely: toxicity of the **target** given its preceding conversation, with examples of allowed quotation, disagreement, banter and genuine abuse.
2. Obtain permission to use real conversations or collect consented examples. Remove personal identifiers; preserve thread structure. Do not invent missing parent turns.
3. Gather a range of Telugu script and Romanized Telugu conversations. Include benign replies to abusive parents, quotations, negation, sarcasm, direct abuse and context-dependent phrases. A few hundred independent threads are a practical pilot, not a statistical sufficiency guarantee.
4. Keep target message ID, conversation ID, parent ID, text, source/consent record, language/script, annotator IDs, independent labels, disagreement and adjudicated label. Store sensitive provenance outside a public submission.
5. Have two Telugu-capable annotators label targets independently while viewing the same parents. Report agreement, class distribution and adjudication procedure. This requires real annotators; no human agreement or consent was fabricated here.
6. Reserve a fresh final test set before model changes. Prevent thread/article/duplicate leakage across train, validation and test. Use additional challenge cases for error analysis separately.
7. Convert approved training/development annotations to the documented custom-thread schema. For a new data version only, set `include_custom_threads: true` and use new prepared/cache/checkpoint/report paths. Keep final-test conversations outside that training input.
8. Select models and thresholds using train/validation only. Evaluate the untouched test once after finalizing choices. Report per-language/script F1, precision/recall, AP, confusion matrices and context-removal results on the same targets.

## 7. Strengthen the comparison before claiming a model improvement

The current original-benchmark TF-IDF overall F1 is 0.717, above historical LoRA's 0.672. On the contextual subset, historical LoRA F1 is 0.696 versus TF-IDF's 0.629. Fresh same-model ablation gives 0.665 without parents. This demonstrates sensitivity to context on this population, not general sarcasm understanding or a clean causal comparison between architectures.

1. Finish corrected LoRA v2 on the unchanged original splits.
2. Compare full validation AP, overall and contextual F1/AP, precision/recall, and false positives. Preserve the validation-selected threshold.
3. For repeated seeds, copy the v2 config to distinct files, set `training.seed` (for example 42, 43, 44), and change **checkpoint, report and embeddings paths** for each run. Keep prepared path/hash fixed. Run each config separately. Never resume a different seed into the same directory.
4. If stronger architectural claims are required, rerun the frozen and target-only neural baselines with matched training settings and the same seeds/splits. The existing historical runs are exploratory evidence.
5. Report mean and spread across seeds. Use thread/group-level bootstrap intervals for paired evaluation differences, since several targets share conversations; avoid treating correlated replies as independent.
6. Do not select a winner based on the already repeatedly inspected test split. Use the fresh independent test from section 6 for the final generalization claim.
7. Deploy a new checkpoint only after checking its reload, score trade-offs and demo behavior. Copy `configs/demo.json` to a new demo config pointing to that checkpoint/report, then set `TOXICITY_CONFIG` explicitly. Keep the existing demo config for rollback.

## 8. Runtime, errors and recovery

Cell runtimes depend on GPU availability, sequence lengths, cache state and network. Actual review-mode cell timings are in `reports/submission_verification.json` and notebook metadata. They are not training-time estimates. Full lossless chunk encoding is more expensive than truncating paragraphs. Legacy embedding caches rebuild once because their old identity omitted actual encoder weights.

No pipeline can guarantee that hardware/network failures never interrupt a cell. The corrected workflow uses separate downloads, validation, visible progress, explicit failure messages and recoverable completed epochs. Do not suppress an exception and continue using stale notebook variables.

| Symptom | Action |
|---|---|
| Lost GPU / CUDA unavailable | Follow section 3 before full training; use the CPU demo meanwhile. |
| Missing parent in external corpus | Review counted excluded chains; fix a custom annotation's structure instead of making up a parent. |
| Download failure / invalid ZIP | Rerun that source cell; retain validated files and inspect the actual response/source. |
| Unexpected `cls.*` keys when loading an MLM as a base encoder | These may be an unused pretraining prediction head. Check the complete loading report; do not dismiss missing encoder weights or unrelated shape errors. |
| Prepared hash mismatch | Restore the exact data snapshot or create a new version/config; do not remove the check to force comparability. |
| Resume configuration mismatch | Use the original config or create a new run directory. |
| API 503 | Read `/api/health`, confirm the selected checkpoint has encoder, head and metadata. |
| Long embedding/training cell | Check progress and resource use; initial encoding is expensive, while a verified cache is reusable. |

## 9. Submission package and final checks

1. Regenerate `python -m contextual.submission` after final experiments. Review `SUBMISSION_REPORT.md`, JSON reports and `submission_provenance.json` together.
2. Include the three current notebooks, `contextual/`, `app/`, `configs/`, `scripts/`, `run_pipeline.py`, tests, dependency files, README, this handoff, generated report and relevant JSON/XML evidence. Keep older modules/notebooks clearly labelled historical if included.
3. Explain the architecture, actual data sizes, label definitions, leakage controls, validation selection, baselines, ablations, known errors and limitations in your presentation. Do not replace missing experiment scores with predictions of improvement.
4. On your own machine, demonstrate from a restarted kernel and through the browser. Live-kernel and HTTP checks passed automatically; visual interaction is the remaining UI/presentation verification.
5. If the evaluator needs offline inference, deliver the selected checkpoint's `encoder/`, `head.pt` and `metadata.json` through an approved large-file channel and retain model-license notices. A Git clone alone excludes local weights and raw data. Do not attach unrelated checkpoints or enormous caches unnecessarily. `resume.pt` is needed to continue training, not for inference.
6. Include data download instructions and hashes. Do not redistribute a dataset unless its terms permit that distribution. Review the custom-thread examples before a public submission.
7. Use `git status --short` and `git diff` to review the changed files, then commit the intended source/evidence files using your normal workflow. No remote push, public deployment or submission to an institution was performed here.
8. If the deadline arrives before GPU recovery or new annotation, submit this honestly as a working research prototype with historical results and clearly identified pending experiments. A completed implementation is not evidence that the corrected model has been trained or that Telugu contextual performance is solved.
