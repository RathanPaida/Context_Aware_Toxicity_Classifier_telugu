# Project Completion Plan — Context-Aware Telugu Toxicity Classifier

Status after the refactor (2026-11-03). The code now runs end to end. The full
curriculum (stages 1–4, evaluation, demo, API) was smoke-tested with a tiny test
model on synthetic data. **No real model has been trained yet**, so no metric
should be quoted until Phase 3 below is done.

---

## What was wrong (and is now fixed in code)

| # | Problem | Effect | Fix |
|---|---------|--------|-----|
| 1 | `PROJECT_ROOT = d:\NLP_projects\...` hard-coded | Nothing could even be imported on this machine, so the web UI showed fake fallback results | Paths derived from repo location + env overrides (`src/config.py`) |
| 2 | Next.js page fabricated verdicts (keyword `if`s, fixed `latency_ms: 16.8`) when the API failed | Demos looked correct while the model never ran | Removed; errors are shown as errors |
| 3 | `/api/benchmarks`, README, and Next.js table showed hard-coded "100% / 0.0% / 16.8 ms" | Metrics were not measured | Read from `reports/eval_results.json`, written by `--evaluate` |
| 4 | Input was truncated from the right | Long context deleted the **target** text; the target mask then fell back to the whole sequence | `encode_example` truncates oldest context first; the target is never dropped |
| 5 | `load_state_dict(strict=False)` everywhere | Key mismatches silently left a random model | Strict loader: missing encoder/head weights raise |
| 6 | Stage 2 `k.replace("encoder.", "")` | Stripped the inner `encoder.` too, so **all Stage 1 transformer layers were discarded** | Prefix-only strip + verification |
| 7 | Stage 3 loaded Stage 1 *or* Stage 2 (Stage 1 first) | Stage 2 (Telugu MLM) was never used | Stage 1 heads + Stage 2 encoder |
| 8 | DravidianCodeMix label `1` (= *not Tamil*) mapped to **toxic** | Model learned "foreign language = toxic". Also the dataset is **Tamil**, not Telugu | Label 1 dropped; documented as Tamil |
| 9 | Cultural-Safety categories stored as a Python-repr string; `json.loads` failed | Sub-labels (threat, identity_attack, …) were never set | Robust parsing; non-toxicity categories (fraud, IP…) dropped |
| 10 | toxic-chat: the assistant reply was used as the "parent" | Context came *after* the target | Context removed; unknown sub-labels masked (-1) |
| 11 | Unannotated sub-labels were treated as 0 | Model taught "not a threat" from data that never annotated threats | Focal loss + metrics mask -1 labels |
| 12 | Stage 4: 20 templates × 75 copies with `(Round N)` / `(Thread N)` suffixes; random split | Label leak (suffix reveals class), val = copies of train, "100% flip" meaningless | One row per template; held-out templates; topic dropout; Stage 3 replay |
| 13 | Evaluation "300 pairs" = 6 pairs × 50; all eval targets were Stage 4 training targets | Test-set contamination | Eval slices in JSON; Stage 4 auto-excludes them; honest *n* |
| 14 | Lexicon "safe fast exit" by substring (`"good night"`, `"bro"` …) | "kill you bro" / "good night, I will kill you" → ALLOWED | Removed; only severe slurs can fast-flag |
| 15 | Lexicon flagged `bc` (because), `kukka`, `na kodaka`, `kosi karam` … instantly | False positives | Severe vs mild tiers in `src/resources/lexicons.json` |
| 16 | Fixed 0.5 threshold, fabricated category scores (`identity_attack: 0.50`) | Uncalibrated / fake outputs | Per-label thresholds tuned on validation, stored in checkpoint |
| 17 | `torch.cuda.amp` forced; stage 3/4 ignored CLI args; `"nan"` context strings; `"goood"`→`"god"` | Crashes on CPU, wrong configs, noisy inputs | Shared `src/training_utils.py`, `run_pipeline.py` passes args |

---

## Phase 0 — Environment (½ day)

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv && .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   `indic-nlp-library` is not installed in your current Python; without it only NFC normalisation is applied.
2. Log in to Hugging Face (`huggingface-cli login`) to avoid rate limits.
3. Optional on Windows: enable Developer Mode (HF cache symlink warning).
4. Smoke test the code (≈2 min, tiny subsets, not a real model):
   ```bash
   python run_pipeline.py --download-only --skip-indic-corp
   python run_pipeline.py --stage1 --stage3 --stage4 --quick-run
   ```
   Then **delete `models/checkpoints/`** before real training. Quick-run checkpoints would otherwise be loaded by the app.

## Phase 1 — Data (2–4 days; most important for "correct results")

**Done (2026-11-04, no manual downloads needed):** `python run_pipeline.py --download-only` produced
toxic-chat 9.7k, Jigsaw via `google/civil_comments` 60k sample (all 5 labels), DravidianCodeMix 41.7k
(24.6% toxic, label 1 dropped), Cultural-Safety-Telugu 4.4k (threat 11%, identity 10.6%), IndicCorpV2 Telugu
5k sentences (raise with `--telugu-sample 50000` for real Stage 2). The old Stage 2 source
`ai4bharat/IndicCorp-Monolingual` does not exist; it is now `ai4bharat/IndicCorpV2` / `tel_Telu`.

1. (Optional) the original Kaggle Jigsaw in `data/raw/jigsaw-unintended-bias-in-toxicity-classification/` is used instead if present.
2. Baseline to beat (held-out test, `toxic`): char TF-IDF + LogReg **F1 0.688 / AUC 0.895**.
3. **Add real Telugu code-mixed toxicity data** — drop it in `data/raw/custom/` (format in README) and rerun `--download-only`; no code changes needed. The current "Telugu" supervision is Tamil comments plus translated LLM prompts, which is the main reason the model cannot be good at Telugu. Search HF / Kaggle / DravidianLangTech shared tasks for Telugu code-mixed hate/offensive datasets (e.g. the DravidianLangTech 2024 Telugu hate-speech task). Verify the licence and label meanings by reading rows, as was done for DravidianCodeMix. Add a loader in `src/data_downloader.py` producing the same schema, with official splits.
4. **Build a real contrastive / thread evaluation set.** The current slices have 6–45 items. Target 200–400 contrastive pairs and 100+ de-escalation pairs:
   - written or collected by ≥2 native Telugu speakers, each item labelled independently;
   - report inter-annotator agreement (Cohen's κ);
   - split 50/50 into `stage4_synthetic.json` (train) and `eval_slices.json` (test), **split by target phrase**, so no target appears in both.
5. Have a native speaker review `src/resources/lexicons.json` (severe vs mild tiers) and the 6 new de-escalation replies in `stage4_synthetic.json`.

## Phase 2 — Training on an RTX 3050 6 GB (2–4 days of compute)

IndicBERTv2 is about 278M parameters; full fine-tuning with AdamW is close to the 6 GB limit.

0. Close other GPU users first (browser hardware acceleration, Jupyter kernels holding CUDA, old training runs) — a
   first test run ran out of memory because an orphaned training process still held ~4 GB.
   Embeddings are frozen by default (`TrainConfig.freeze_embeddings`): IndicBERTv2's 250k-token embedding table is
   ~190M of its 290M parameters, and training it does not fit in 6 GB. Trainable params: 97.7M.
   If it still does not fit: `gradient_checkpointing=True` in `TrainConfig`, then `--batch-size 4`.
1. Start with a 1k-row timing run to measure speed and memory:
   `python run_pipeline.py --stage1 --max-samples 1000 --epochs 1 --batch-size 8`
2. If it runs out of memory: `--batch-size 4`, set `gradient_accumulation_steps=4` in `Stage1Config`, or `freeze_ratio=0.25`. `MAX_SEQ_LENGTH=256` is already set (was 512).
3. Stage 1: use `--jigsaw-sample 60000` during download if 200k is too slow. 1–2 epochs.
4. Stage 2 (MLM): `--max-steps 3000 --telugu-sample 50000` is a reasonable budget. Check that perplexity decreases in the log.
5. Stage 3: default 4 epochs with early stopping. Watch the per-label log lines; labels with "no positives in split" are not learned.
6. Stage 4: only after Phase 1.4 provides enough pairs; with the current 10 templates it mainly checks that the code works.
7. Keep every run's `metadata.json` (it stores the data, thresholds and metrics) for the report.

## Phase 3 — Evaluation & error analysis (2 days)

1. `python run_pipeline.py --evaluate --demo`, then read `reports/eval_results.json`.
2. Report **held-out test macro-F1 / AUC first**; contextual slices with their *n*.
3. Baselines for the report — `python run_pipeline.py --baselines` (lexicon, TF-IDF, model, context ablation). Also worth adding by hand:
   - lexicon only (`FastTargetFilter`);
   - TF-IDF + logistic regression;
   - same model **without context** (pass `history=[]`), which is the ablation that shows context helps;
   - model without the extra attention blocks (`use_multihead_attention=False`).
4. Error analysis: 50 false positives and 50 false negatives grouped by cause (slang, script, sarcasm, context).
5. If thresholds look off on the test set, re-tune on validation only, never on test.

## Phase 4 — Application (1 day)

1. `python app/server.py`. Health should show `ready: true` and the checkpoint stage.
2. Next.js: `cd frontend && npm install && npm run dev` (set `BACKEND_URL` if the API is not on 127.0.0.1:7860). Only a syntax check was run after the edit, not `npm run build`.
3. Update `src/resources/scenarios.json` so every demo scenario is one the trained model actually gets right, or show mismatches openly.

## Phase 5 — Notebook, report, cleanup (1–2 days)

1. ~~Notebook~~ — done: it now imports from `src/` and reads `reports/*.json`; the duplicate `notebooks/` copy was removed.
2. Update `docs/Context_Aware_Toxicity_Classifier_Project_Report.pdf` with measured numbers, dataset caveats (Tamil vs Telugu, prompt-safety data) and the ablations.
3. ~~Add tests~~ — done: `python -m pytest tests -q` (36 tests). Add an API test (503 without checkpoint) if you extend the server.
4. Commit in logical steps (data fixes, model fixes, app fixes) so the history is reviewable.

## Suggested schedule

| Week | Work |
|------|------|
| 1 | Phase 0, Phase 1 (data acquisition and annotation start) |
| 2 | Phase 1 annotation finish, Phase 2 training |
| 3 | Phase 3 evaluation, baselines, error analysis |
| 4 | Phase 4 app, Phase 5 notebook / report |
