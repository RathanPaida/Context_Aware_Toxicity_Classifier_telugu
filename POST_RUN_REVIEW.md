# Post-run review: completed cleaned-data experiment

## Verdict

The submitted ZIP contains a completed five-epoch neural experiment, validation-selected ensemble, challenge evaluation and uncertainty analysis. All expected report files are present. The ensemble is a reasonable final candidate for this research project, but broad Telugu conversational reliability is not established.

I reviewed the ZIP found at `reports/run_results_for_review.zip`. I recomputed metrics and additional confidence intervals from its saved predictions. I did not retrain models or run full model inference. The original ZIP and measured reports were preserved.

## Verification

- Cleaned data: 78,232 examples; 54,578 train, 11,776 validation and 11,878 test.
- Prepared-data hash matches the configured cleaned version.
- Test IDs and labels align across the three principal models.
- Recomputed overall F1, AP, ROC-AUC, precision/recall and confusion matrices match the reported values.
- Ensemble scores exactly match the recorded component scores and weights.
- Checked ZIP evidence hashes match its provenance; current available source hashes also match the recorded source snapshot at review time.
- Regression receipt: 77 passed, one optional test skipped, zero failures.
- The neural report records `checkpoint_reload_verified: true` and five completed epochs.
- Several earlier training attempts ended with exit code 1; the final attempt and later stages completed. The transcripts do not include enough native Python error detail to establish the cause of each earlier failure.

The actual successful configuration used training batch size **8**, encoder batch size **8** and gradient accumulation **4**, giving an effective batch of 32. This differs from the initial 32/32/1 proposal; document the configuration actually used.

The five recorded epochs took about **108.83 minutes combined**, including their validation timing, excluding unsuccessful attempts and later evaluation/report work. Validation AP rose from 0.6848 to 0.7655 while training loss fell from 0.7425 to 0.5824. Epoch 5 was best among the five observed epochs. This does not prove convergence, nor require another training run before submission.

## Same-population results

All rows below use the same 11,878 cleaned test examples; the contextual subset contains 2,282 examples and 196 positives.

| Model | Overall F1 | Overall AP | Context F1 | Context AP |
|---|---:|---:|---:|---:|
| Character TF-IDF | 0.7192 | 0.7865 | 0.6061 | 0.7265 |
| IndicBERT + LoRA + conversation attention | 0.6948 | 0.7584 | 0.7612 | 0.8213 |
| 50/50 ensemble | 0.7287 | 0.8041 | 0.7713 | 0.8603 |

Selected weights: character TF-IDF **0.5**, word TF-IDF **0.0**, contextual transformer **0.5**. Ensemble threshold: **0.482697802797977**. These were selected using validation, not by choosing the best test result.

The neural model alone is weaker than character TF-IDF overall, but much stronger on examples with parents. The ensemble balances their strengths and has the highest observed overall and contextual F1 in this comparison.

Per-source ensemble F1 is 0.7459 for CGA, 0.8419 for Telugu prompt safety and 0.6886 for Tamil offensiveness. The latter is slightly below character TF-IDF's 0.6932. The ensemble is not better on every source.

## What the uncertainty actually supports

All intervals below are paired group-bootstrap percentile intervals, with 1,000 resamples of connected conversation/shared-text groups. Differences are in F1 percentage points, not relative percentage improvements.

| Comparison | Observed difference | 95% interval | Interpretation |
|---|---:|---:|---|
| Ensemble vs neural, overall | +3.39 points | +2.54 to +4.26 | Positive difference on this evaluation population |
| Ensemble vs character TF-IDF, overall | +0.95 points | -0.09 to +1.83 | Small observed gain; interval includes zero |
| Ensemble vs neural, contextual subset | +1.02 points | -2.51 to +4.36 | Contextual superiority over neural alone is not established |
| Ensemble vs character TF-IDF, contextual subset | +16.53 points | +10.66 to +22.24 | Substantial contextual gain on this population |
| Ensemble with parents vs without parents, contextual subset | +5.62 points | +2.18 to +9.19 | Evidence that the supplied context helps this model here |

The ZIP's original uncertainty report compares the ensemble against the **validation-selected contextual transformer**. It does not establish a statistically clear overall win over TF-IDF. I computed that additional comparison separately; its interval crosses zero.

These intervals do not account for all selection uncertainty, dataset bias or repeated development-time test inspection. They are not substitutes for a fresh independent test set.

## Context behavior and its trade-off

Neural contextual F1 is 0.7612 with parents versus 0.6939 without parents. Supplying parents catches 34 more toxic replies, with 25 additional false positives.

Ensemble contextual F1 is 0.7713 with parents versus 0.7152 without parents. Supplying parents catches 22 more toxic replies, with 11 additional false positives.

Compared with the neural model on contextual examples, the ensemble reduces false positives from 53 to 27 but also detects fewer toxic replies: 140 rather than 153. It has higher precision and lower recall. Present this trade-off instead of saying every aspect improved.

## Telugu challenge: unresolved limitation

The neural model's 51-example authored challenge has:

- F1: **0.6154**; precision: **0.4848**; recall: **0.8421**.
- Confusion matrix [TN, FP; FN, TP]: **[[15, 17], [3, 16]]**.
- It detects 16 of 19 toxic examples but falsely flags **17 of 32 safe examples** (53.1%).
- Without parents, F1 is **0.6275**, slightly higher. Context benefit on the English benchmark does not generalize to this small Telugu diagnostic.
- Recorded exact training-text overlap is zero, but this is still an authored development challenge, not an independently collected representative test.

The challenge file evaluates **the neural model**, not the selected ensemble. Do not attribute its 0.6154 F1 to the ensemble. Evaluate the final ensemble separately before making a final-model Telugu claim. The current `contextual.recheck` entry point loads the neural checkpoint directly; simply passing an ensemble demo config to it would not evaluate the ensemble.

## Required final checks before submission

### 1. Freeze and preserve this candidate

Keep the submitted ZIP, configs, prepared-data hash, neural checkpoint and `reports/clean_ensemble.joblib`. Do not change ensemble weights or thresholds after looking at test errors. Do not overwrite this run while exploring another seed or architecture.

There is no reason to rerun the entire project solely because the first attempts failed: the final reports and score files are complete and internally consistent.

### 2. Verify the actual final demo

Use the ensemble configuration explicitly:

```powershell
Set-Location D:\NLP_projects\Context_Aware_Toxicity_classifier
$env:TOXICITY_CONFIG = "configs/demo_clean_ensemble.json"
python app/server.py
```

Open `http://127.0.0.1:7860/api/health` and the main webpage. Confirm `ready: true`, predictor type `EnsemblePredictor`, ensemble artifact `reports/clean_ensemble.joblib`, and the ensemble threshold. Test normal replies, abusive replies and real parent/target pairs.

The ZIP's existing HTTP/ASGI verification receipts refer to the **older historical demo checkpoint**. They are not proof that this new ensemble demo has been verified.

### 3. Check evaluation-to-demo numerical consistency

The submitted trainer uses CUDA FP16 autocast in `predict_online`. The current demo Predictor uses full-precision inference, and the demo configuration selects CPU. The reload check also uses a small probe and relaxed FP16 tolerance. These facts do not invalidate the saved-score arithmetic, but leave a serving-parity check outstanding.

Compare a fixed sample of contextual and isolated examples, including scores close to the decision threshold, between the saved ensemble predictions and the deployed `EnsemblePredictor`. Record maximum score difference and changed decisions. If differences are material, align the evaluation and serving precision, then regenerate validation scores, validation-selected weights/threshold and test reports with the chosen inference path. This is an evaluation/export correction; it does not automatically require retraining.

### 4. Evaluate the final ensemble on the same 51 challenge targets

Use `EnsemblePredictor` / `create_predictor` with `configs/demo_clean_ensemble.json`, not the neural-only `contextual.recheck` command. For each challenge target, save its ID/label, the ensemble score with parents and its returned `isolated_score`. Summarize both using the saved ensemble threshold. Keep this diagnostic separate from the main benchmark and do not tune on its labels.

### 5. Update the submission text

Use the result table above, the actual batch/accumulation settings, data-cleaning counts, validation-selection procedure and uncertainty caveats. Retain dataset citations and license notices from RUN_AND_REPORT.md.

A defensible conclusion is:

> On the fixed cleaned evaluation split, a validation-selected 50/50 character TF-IDF and contextual IndicBERT-LoRA ensemble achieved overall F1 0.7287 and contextual-subset F1 0.7713. Removing parent context reduced the ensemble's contextual F1 to 0.7152. The overall gain over TF-IDF was small and its group-bootstrap interval included zero. Telugu conversation reliability remains limited and requires independent evaluation.

Do not claim human-level reasoning, reliable sarcasm detection, uniformly improved multilingual performance or production readiness. F1 is not accuracy.

## Further improvements: prioritize evidence, not another architecture

1. Obtain independently annotated Telugu/Romanized Telugu conversations, particularly benign replies to toxic parents, quotations and context-dependent expressions. Retain an untouched final test set and report annotator agreement.
2. If time permits, repeat the selected training configuration with new seeds and separate output paths; report mean and spread. One seed does not quantify training variance.
3. Complete the optional same-cleaned-data target-only neural and frozen-encoder controls for stronger architectural attribution. Context removal from one model is not the same experiment as training a target-only model.
4. Only then consider longer training, other transformer backbones or more data. Validation AP was still improving at epoch five, so more epochs are an experiment worth testing on validation, not an assured improvement.
5. More isolated comments or a larger English corpus will not directly solve the demonstrated Telugu-context gap. Do not delay a truthful prototype submission just to accumulate a larger row count.

## Send back next

Send the final ensemble challenge report/predictions, serving-parity summary and a health response or demo screenshot showing the actual ensemble artifact. Include exact commands and any traceback if a final check fails. You do not need to rerun all training to provide these items.

Independent verification results from this review are saved in `reports/post_run_review_verification.json`. The original ZIP was not modified.
