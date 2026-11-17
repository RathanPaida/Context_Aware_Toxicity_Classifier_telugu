# Context-aware toxicity: submission report

Current run instructions: [RUN_AND_REPORT.md](RUN_AND_REPORT.md). User-run results appear only when their report files exist; pending rows below are not measured results.

This report is generated from JSON evidence by `python -m contextual.submission`.
It distinguishes historical experiments from corrected, newly verified work.

## Task and method

The task is binary toxicity of a target reply given its actual ancestors. Individual message embeddings feed a hierarchical Transformer and a target-specific classifier. No domain selector or keyword verdict override is used. Long paragraphs use content-token-weighted chunk pooling.

## Data and evaluation

Prepared data SHA-256: `de69b37150ecae47c1d74ef7017c289f6d6d22018b86041cc141792f9a58007f`.
Splits: 54,960 train, 11,893 validation, 11,986 test.
14,598/78,839 examples have ancestors.
Conversations/articles and identical normalized message texts are kept together. Near-duplicates still need manual review.
CGA labels personal attacks; Tamil labels offensiveness; Telugu labels prompt safety. These are related but different constructs.
The existing test split has been inspected during development; treat comparisons as exploratory, not a fresh final generalization estimate.

## Identical-population comparison

| Model | Overall n | Overall F1 | Overall AP | Overall ROC-AUC | Context n | Context F1 | Context AP | Context FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF (rerun) | 11986 | 0.7170 | 0.7874 | 0.9249 | 2291 | 0.6287 | 0.7379 | 33 |
| Target-only neural (historical) | 11986 | 0.6713 | 0.7387 | 0.9041 | 2291 | 0.6165 | 0.6950 | 89 |
| Context dropout 0.2 (historical) | 11986 | 0.6664 | 0.7218 | 0.8987 | 2291 | 0.6087 | 0.6983 | 92 |
| Context dropout 0 (historical) | 11986 | 0.6675 | 0.7275 | 0.9010 | 2291 | 0.6445 | 0.7273 | 90 |
| LoRA v1 (historical) | 11986 | 0.6721 | 0.7302 | 0.9038 | 2291 | 0.6959 | 0.7846 | 27 |
| LoRA v2 (corrected) | Pending | — | — | — | — | — | — | — |

TF-IDF remains the strongest measured overall baseline. Historical LoRA had the highest observed contextual-subset F1, but used different sampling and biased checkpoint selection. This does not prove superior reasoning or isolate a causal benefit of LoRA.

## Development challenge

The curated Telugu/Romanized Telugu file contains 51 labelled targets; parent messages may be unlabelled. It is excluded from the frozen training dataset. Human annotation provenance is not established by the file schema; do not claim independent human validation without records.
These challenge cases were designed around known weaknesses. They are a development diagnostic, not a representative independent test set.

### LoRA v1 challenge

Population: `curated_development_challenge_not_independent_test`. n=51. F1 with context=0.5957; without context=0.6087.
With-context confusion matrix [TN, FP; FN, TP]: `[[18, 14], [5, 14]]`.

### LoRA v1 contextual recheck with fresh embeddings

Population: `historical_contextual_test_only`. n=2291. F1 with context=0.6959; without context=0.6647.
With-context confusion matrix [TN, FP; FN, TP]: `[[2068, 27], [77, 119]]`.

## Separate additional-data experiment

34,556 validated examples, including 2,623 with parents. Sources: `{'toxichat_human': 4560, 'macd_telugu': 29996}`. Splits: `{'test': 5168, 'train': 24275, 'val': 5113}`.
67 isolated rows with conflicting labels were quarantined. Original benchmark membership was not changed.
Source URLs and pinned revisions are in `configs/data_sources.json`; file hashes are in `reports/external_downloads.json`.
These are new conversation/shared-text grouped splits, not the sources' official splits. English ToxiChat human turns supply context; MACD Telugu supplies isolated comments.
Executed TF-IDF baseline: test n=5,168, F1=0.8772, AP=0.9589; contextual subset F1=0.6113.
Telugu isolated-comment F1=0.8972. This is not Telugu contextual accuracy and is not directly comparable to the original benchmark.
External LoRA training is pending GPU recovery; no neural improvement is claimed from the additional data.

## Cleaned main dataset and new experiment

78,232 examples; splits: `{'train': 54578, 'val': 11776, 'test': 11878}`. Exclusions: `{'duplicate_same_chain_label': 566, 'contradictory_isolated_labels': 41}`.
SHA-256: `368817cc6203e762d0a5b5a9f777761759fbf8769d0f3ec0b2d5dadb039b6bd6`.
NFC/control-character/whitespace normalization is shared with inference. Negation, script, emoji and joiners are preserved. No exact normalized-message leakage was detected. This is a different evaluation population from the historical table above.

| Cleaned-data model | Status | Test n | Overall F1 | Context F1 |
|---|---|---:|---:|---:|
| Character TF-IDF | Recorded | 11878 | 0.7192 | 0.6061 |
| Frozen IndicBERT + context attention | Not run | - | - | - |
| Target-only neural | Not run | - | - | - |
| IndicBERT + LoRA + context attention | Recorded | 11878 | 0.6948 | 0.7612 |
| Validation-selected voting | Recorded | 11878 | 0.7287 | 0.7713 |

Voting weights: `{'char_tfidf': 0.5, 'word_tfidf': 0.0, 'contextual_transformer': 0.5}`. Selection: Maximum mean of overall and contextual validation AP; grid step 0.25; threshold maximizes overall validation F1. Test labels are not used for selection.
Test F1 difference from the best validation-selected single component: +0.0339. This is a measured difference, not automatically evidence of generalization.

See `reports/clean_ensemble_uncertainty.json` after running the group-bootstrap analysis. Missing files mean the experiment or analysis has not completed.

## Execution verification

Earlier notebook verification used **sequential in-process**. These receipts describe the earlier default-mode run; they are not proof that the new cleaned experiment has run.
The historical CPU demo passed ASGI and real localhost HTTP checks; see `reports/submission_http_verification.json`. Verify the new artifact after your training run.
The GPU subsequently recovered. Full training was intentionally stopped on user request, and the user-run guide is RUN_AND_REPORT.md.
Latest regression receipt: 77 passed, 1 skipped, 0 failed/error. Tiny training fixtures do not establish full-data model quality.

## Corrections and remaining limits

- One shared differentiable chunk encoder is used for adapter training, validation and inference.
- LoRA v2 uses full validation and the full training pool; class weighting matches that pool.
- Adapter/head/optimizer/RNG state is saved atomically after each epoch and resumed only for matching data/configuration.
- Cache identities include actual model parameters and tokenizer, preventing silent reuse across adapted encoders.
- The corrected trainer checks merged-checkpoint reload predictions and reports context-removal metrics on exactly the same examples.
- Thresholds maximize validation F1; this is threshold selection, not probability calibration or a guarantee of a chosen false-alarm rate.
- A full corrected GPU run, repeated-seed comparison, independent annotation and fresh final test set remain required for stronger claims.
- Local model weights/data are not committed to Git; the source distribution includes explicit setup and recovery instructions.

## Submission instructions

See [RUN_AND_REPORT.md](RUN_AND_REPORT.md) for the current complete run order and exact files to send back. [SUBMISSION_HANDOFF.md](SUBMISSION_HANDOFF.md) retains earlier dataset/access background. Historical PDF reports may be stale; this regenerated Markdown and JSON evidence distinguish measured work from pending experiments.
