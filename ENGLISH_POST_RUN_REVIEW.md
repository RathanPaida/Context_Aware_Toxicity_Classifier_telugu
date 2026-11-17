# English experiment review and final submission steps

Reviewed 17 November 2026. No production training or inference was started during this review. Metrics below were checked against saved per-example predictions.

## 1. Decision

**The English experiment completed successfully, and the English ensemble is the strongest measured English recipe. Keep it as the English candidate. Do not use it as a replacement for the mixed-language model on Telugu.**

You do not need to rerun the full training pipeline to address this review. Before submission, correct the narrative notes, verify the intended demo artifact, and state the limits of the contextual and multilingual claims.

The evidence supports a useful ensemble improvement. It does not support the stronger claims that context always helps, that more parents always improve results, or that English-only training improves every architecture and language.

## 2. Exactly what was reviewed

Reviewed file: `reports/english_cga_v1_results_for_review.zip`, 3,876,065 bytes, 72 archive entries.

SHA-256: `408d307f8b32b8087e23d85c90769f4aeb4ab99cc87a5a50687663b4629765f8`.

The uploaded file import failed because its transfer helper could not apply file attributes on Windows. The existing project ZIP has the expected filename and the same reported size as the latest upload, but uploaded bytes were not independently compared. All conclusions here refer explicitly to the project ZIP identified above.

Evidence checks:

- All expected artifact filenames in the returned manifest are present.
- Archive CRC and every declared member SHA-256 passed.
- The English prepared-data hash matches the experiment and all five model reports.
- English examples, labels, texts, groups and splits match the corresponding historical cleaned population.
- Recomputed test metrics match the reports and comparison table.
- Saved validation thresholds checked for models with returned validation scores.
- All returned English budget tables and Telugu challenge overall metrics match their saved scores.
- Selected paired bootstrap intervals were independently recomputed from saved predictions.
- **140 saved-evidence checks passed.** See `reports/english_post_run_verification.json`; reproduce with `python scripts/review_english_results.py`.

The ZIP includes a test XML showing 85 passed and one optional legacy dependency test skipped. It includes **no stage `.log` files**, despite the file-completeness check passing: stage logs were not mandatory in that manifest. Completion is supported by reports, predictions and the LoRA completion/reload fields; complete console histories and all stage durations cannot be independently reconstructed.

## 3. Corrected English results

All rows use the same 3,577 English test targets, with 274 toxic targets. The contextual subset contains 2,282 targets, with 196 toxic targets. F1 is **positive/toxic-class F1**, not macro-F1 or accuracy. AP means average precision.

| Training recipe/model | Test F1 | Test AP | Contextual F1 | Test precision | Test recall |
|---|---:|---:|---:|---:|---:|
| English character TF-IDF | 0.7259 | 0.8049 | 0.7446 | 0.7705 | 0.6861 |
| English frozen target-only MLP | 0.6299 | 0.6951 | 0.6267 | 0.5673 | 0.7080 |
| English frozen contextual Transformer | 0.6546 | 0.7322 | 0.6600 | 0.6487 | 0.6606 |
| English LoRA contextual Transformer | 0.7232 | 0.8194 | 0.7433 | 0.7313 | 0.7153 |
| **English ensemble** | **0.7869** | **0.8629** | **0.7981** | **0.7435** | **0.8358** |
| Historical mixed TF-IDF, English subset | 0.6035 | 0.7007 | 0.6061 | 0.7611 | 0.5000 |
| Historical mixed LoRA, English subset | 0.7394 | 0.8059 | 0.7612 | 0.7491 | 0.7299 |
| Historical mixed ensemble, English subset | 0.7459 | 0.8404 | 0.7713 | 0.8505 | 0.6642 |

The English ensemble selected **75% character TF-IDF, 0% word TF-IDF and 25% contextual LoRA**, using validation. Its fixed threshold is **0.306118171793546**. Word TF-IDF alone had test F1 0.5759 and received zero weight; forcing it into the ensemble is not justified.

LoRA validation AP improved over all five epochs: 0.6849, 0.7050, 0.7380, 0.7798, 0.8157. The saved report records checkpoint reload verification, and progress ends at `complete`. Logged epoch durations sum to **56.92 minutes**, excluding any unreported setup, export or failed-attempt overhead. This is not a complete end-to-end runtime.

The target-only/frozen baselines are useful negative results. Do not hide them because they underperform TF-IDF. LoRA's ranking AP is stronger than the English TF-IDF baseline, but its standalone F1 is slightly lower at the selected thresholds. The improvement comes from combining complementary predictions, not from every neural model beating the lexical baseline.

## 4. Is improvement supported?

Paired percentile intervals resample connected conversation/shared-text groups: 581 groups overall and 540 in the contextual subset. Values below are **F1 percentage-point differences**.

| Comparison | Overall difference | 95% paired interval | Interpretation |
|---|---:|---:|---|
| English ensemble minus English TF-IDF | +6.11 | +1.91 to +10.26 | Positive evidence for the ensemble on this holdout. |
| English ensemble minus English LoRA | +6.37 | +3.00 to +9.53 | Positive evidence for combining them. |
| English ensemble minus historical mixed ensemble | +4.10 | +0.18 to +8.57 | Positive but relatively marginal lower bound; keep exploratory qualifications. |
| English LoRA minus historical mixed LoRA | −1.61 | −4.15 to +1.19 | No established English-only LoRA improvement. |

On the contextual subset, English ensemble minus historical mixed ensemble is **+2.68 points**, interval **−2.02 to +7.43**: that particular improvement is not established by this interval.

These are not untouched confirmatory results. The historical test has already been inspected, multiple models/comparisons are reported, and the intervals do not adjust for all selection decisions or multiple comparisons. Also, English and mixed models used their own validation populations and thresholds. Describe the comparison as one between complete recipes, not a causal experiment isolating language mixture.

### The precision/recall trade-off matters

Compared with the old mixed ensemble on the identical English test set:

- True positives rise from **182 to 229**; missed attacks fall from **92 to 45**.
- False positives rise from **32 to 79**.
- Precision falls from **0.8505 to 0.7435**, while recall rises from **0.6642 to 0.8358**.
- Both have **124 total errors**, so their ordinary accuracy is identical despite the better toxic-class F1.

For a moderation-assistance demonstration, show this trade-off. Do not claim that the new model reduced every type of error. A stricter operating threshold would require validation-based selection according to a stated precision/recall goal, not test tuning.

## 5. What the context-budget experiment says

This table uses FP32 saved-checkpoint inference and the same ensemble threshold. Each budget is a maximum number of preceding ancestors; naturally shorter paths are not padded.

| Retained-parent budget | Validation contextual F1 | Test contextual F1 | Test overall F1 |
|---:|---:|---:|---:|
| 0 | 0.8216 | 0.7844 | 0.7763 |
| 1 | 0.8235 | 0.7876 | 0.7786 |
| 2 | 0.8287 | 0.8000 | 0.7878 |
| 3 | 0.8303 | 0.8088 | 0.7943 |
| 5 | 0.8367 | 0.7953 | 0.7849 |
| 11 | 0.8419 | 0.7981 | 0.7869 |

Keep the existing **11-parent capacity / 12-turn configuration** for this result: validation preferred the full budget. Selecting three parents now because its test F1 is highest would be test-driven selection.

The ensemble's full-context test benefit over zero parents is **+1.37 F1 points**, interval **−2.02 to +4.72**. The LoRA model's FP32 benefit is **+3.23 points**, interval **−0.87 to +7.50**. Both intervals include zero. Say that context changes predictions and improves observed recall/F1 in this experiment, but its incremental aggregate benefit remains uncertain.

The no-parent ensemble already reaches overall F1 **0.7763**. Much of the ensemble gain over TF-IDF therefore remains without parent context. Do not attribute the whole gain to conversational attention; the neural target representation can also provide complementary information.

### Natural depth is a different breakdown

Full-budget ensemble performance grouped by actual available ancestry:

| Available ancestors | Test examples | Positive targets | F1 |
|---|---:|---:|---:|
| 0 | 1,295 | 78 | 0.7564 |
| 1 | 990 | 47 | 0.7059 |
| 2 | 577 | 37 | 0.8193 |
| 3 | 355 | 37 | 0.8000 |
| 4–7 | 356 | 73 | 0.8395 |
| 8–11 | 4 | 2 | 1.0000 |
| 12+ | 0 | 0 | Not evaluated |

The perfect long-bin score is based on **four examples** and is not reliable evidence of long-context generalization. The valid released paths contain at most ten ancestors. Neither table establishes performance for arbitrary parent counts or multiple direct parents.

## 6. Telugu: do not replace the mixed model with the English ensemble

All three rows use the same 51 authored challenge targets, including 19 toxic targets. This is a development diagnostic, not a large independently annotated benchmark.

| Model | Full-context F1 | AP | Toxic targets detected | False positives |
|---|---:|---:|---:|---:|
| Mixed-language ensemble | 0.6522 | 0.7540 | 15/19 | 12 |
| Mixed-language LoRA | 0.6154 | 0.6063 | 16/19 | 17 |
| English-trained ensemble | **0.0000** | 0.4797 | **0/19** | 0 |

All English-ensemble challenge scores are below its English threshold: full-context scores range from approximately **0.0157 to 0.2883**, while the threshold is 0.3061. This failure is confirmed. The notes' explanation attributing it solely to character TF-IDF is not experimentally isolated by the returned outputs; English lexical weighting, domain shift, encoder adaptation and threshold mismatch are plausible contributors. Do not present one of these as a proven sole cause.

The mixed ensemble's challenge F1 is **0.6667 without parents**, versus 0.6522 with full context. Its context-benefit interval includes zero. Thus this challenge does not establish a Telugu contextual benefit either.

Keep the English and mixed checkpoints separately named. Use the English candidate for the English demonstration and the existing mixed model for an explicitly labelled multilingual experiment. Do not silently route Romanized Telugu through the English model or invent a language router based on script alone. Independent Telugu contextual annotation remains the highest-value data gap if the project promises robust contextual Telugu moderation.

## 7. Correct the narrative run notes before submission

The archived `english_cga_v1_run_notes.txt` contains values that conflict with the JSON reports and saved scores. Use this review and the machine-readable verification as the checked source. Preserve the original notes as an audit trail; do not submit their incorrect values as final results.

| Item | Verified value |
|---|---|
| Python recorded by preflight | 3.10.20, not 3.12.9 |
| Train / validation / test positives | 1,221 / 285 / 274 |
| English TF-IDF threshold | 0.223936404444859 |
| Target-only MLP threshold | 0.7648286819458008 |
| Frozen contextual threshold | 0.9457253813743591 |
| Target-only test AP | 0.6951266576792441 |
| Frozen contextual test AP | 0.7321879342015708 |
| English ensemble test ROC-AUC | 0.9814866378938855 |
| Historical mixed TF-IDF English AP | 0.7006739871634022 |
| Historical mixed LoRA English AP | 0.8059415019068421 |
| Historical mixed ensemble English AP | 0.8403759291767717 |

Several AP values and the ensemble budget sweep in those notes also differ from their source files. Use the complete tables above. Avoid the broad statement that English-only training is statistically superior across models: the standalone English LoRA result does not support that.

## 8. Serving evidence and the small code correction

The returned English ensemble has **zero full-window decision flips** between saved scores and CUDA FP32 inference on both validation and test. Maximum absolute score differences are approximately 0.00444 on validation and 0.000538 on test. The LoRA full-window test parity check also has zero decision flips. These checks do not verify CPU or the HTTP application, and the LoRA zero-parent FP32 predictions differ slightly from the trainer's zero-parent results.

During review, the English ensemble inference config was found to reference the LoRA report in `paths.report`. That would make `/api/benchmarks` show LoRA results while `/api/moderate` served the ensemble. Fixed:

- `configs/english_cga_v1_ensemble.json` now points to `reports/english_cga_v1_ensemble.json`.
- The config generator makes the same correction for future English experiments.
- A regression assertion covers the report path.
- `scripts/verify_http.py` now accepts an explicit config/output, verifies the chosen checkpoint/artifact, and checks that the displayed report and served threshold agree.

This changes report routing and verification, not trained weights, ensemble weights, thresholds or the scores reviewed above. The original returned ZIP is unchanged. No retraining is required.

## 9. Exactly what to do next

### Step 1 — keep the completed run

Keep all English and mixed checkpoints, reports, configs, prepared-data hashes and the original results ZIP. Do not rerun training merely to repair presentation or report text. Keep the successful English threshold and weights fixed for this submission result.

### Step 2 — run the short selected-model HTTP checks

From the project root in the working Python environment, run each command separately:

```powershell
python scripts/verify_http.py --config configs/english_cga_v1_ensemble.json --output reports/english_cga_v1_http_verification.json --startup-timeout 120
python scripts/verify_http.py --config configs/demo_clean_ensemble.json --output reports/english_shared_mixed_http_verification.json --startup-timeout 120
```

These load the existing saved models, send a tiny HTTP smoke request and stop the temporary server. They do not train or run the full benchmark. The English config uses CPU, so the first load may take time. If a command fails, keep the traceback and `scratch/submission_http.log` immediately before another check overwrites that log.

Expect `ready: true`, `predictor_type: EnsemblePredictor`, the intended ensemble artifact, and the corresponding threshold. English should show approximately **0.3061181718**. The check also verifies the benchmark endpoint's report and threshold. It is an operational smoke test, not a new accuracy evaluation or proof of all CPU/GPU numerical equivalence.

### Step 3 — choose the demo deliberately

For the English demonstration:

```powershell
$env:TOXICITY_CONFIG = 'configs/english_cga_v1_ensemble.json'
python app/server.py
```

For a separate mixed-language demonstration, stop that server and launch:

```powershell
$env:TOXICITY_CONFIG = 'configs/demo_clean_ensemble.json'
python app/server.py
```

Check `/api/health` before showing predictions. State the selected training recipe in your presentation. Do not imply that the English ensemble has proven Telugu accuracy. Do not change the server's default model silently just to make one demo example look better.

### Step 4 — update the submission document and slides

Use the model table, paired intervals, recall/false-positive trade-off, natural-depth counts and limitations from this review. Explain that the ensemble is validation-selected. Include all controlled baselines. Describe attention as model weighting, not a causal explanation. Report the Telugu transfer failure as a limitation rather than hiding it.

A defensible central claim is:

> On the existing grouped English CGA test population, a validation-selected combination of character TF-IDF and a LoRA-adapted contextual Transformer improved toxic-class F1 and average precision over the evaluated single models. Performance varied by context availability, and contextual Telugu generalization remains insufficiently validated.

Also state that this is an exploratory reused test set, and that the new ensemble's contextual advantage over the historical mixed ensemble has an interval crossing zero.

### Step 5 — decide whether more experiments are actually needed

For an academic submission centred on this English benchmark, the completed controlled experiment is sufficient to report honestly after the operational/presentation checks. Whether it meets the marking rubric is a separate decision. More training epochs, a new encoder, CAD or ToxiChat are **optional follow-up work**, not repairs required by these results.

If the project promises a reliable multilingual or arbitrary-depth deployment, the current evidence is insufficient. Prioritize independent labelled Telugu conversations and more long-chain examples, using the collection/audit steps in `ENGLISH_EXPERIMENT_PLAN.md`. More isolated English data will not establish those claims. Additional seeds or an untouched holdout would strengthen the English conclusion; do not choose the best seed using test F1.

### Step 6 — return only the remaining evidence

Send the two HTTP verification JSON files, any failure traceback/log, and the updated report/slides if you want a final submission review. No full rerun ZIP is necessary if nothing else changed. If you run additional experiments, give them separate paths and include configs, validation choices, predictions and complete logs.

The scripts and reports added during this review are in the project. The original training artifacts and submitted archive were preserved.
