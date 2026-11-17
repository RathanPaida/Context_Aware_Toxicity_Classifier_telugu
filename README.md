# Context-aware toxicity classifier

This project predicts toxicity of a target reply using its actual parent chain. A pretrained IndicBERT encoder produces individual message vectors; a hierarchical Transformer learns contextual and conversation representations. The active pipeline is in `contextual/`. The older `src/` cascade is retained as historical work.

Start with **[RUN_AND_REPORT.md](RUN_AND_REPORT.md)** for the complete current run order and what to send back after execution. Full training is left to the user.

Read [SUBMISSION_REPORT.md](SUBMISSION_REPORT.md) for measured results and [SUBMISSION_HANDOFF.md](SUBMISSION_HANDOFF.md) for detailed setup, blocked steps, data downloads and submission instructions.

## Start here

Use the existing working Python environment, or install `requirements-contextual.txt` in a fresh environment. `requirements-tested.txt` records the tested versions; CUDA packages are platform dependent. A fresh Git clone does not include local model weights or raw datasets.

```powershell
python -m pytest -q
python scripts/verify_submission.py --live-kernel
python scripts/verify_http.py
python -m contextual.submission
$env:TOXICITY_CONFIG = "configs/demo.json"
python app/server.py
```

Open http://127.0.0.1:7860 for the demo and `/docs` for the API. `/api/health` identifies the actual checkpoint and device. The default demo uses the existing historical no-context-dropout model on CPU; corrected LoRA v2 is not silently substituted or claimed trained. API scores are model outputs, not calibrated confidence.

## Notebooks

| Notebook | Default behavior |
|---|---|
| [Context_Aware_Toxicity.ipynb](Context_Aware_Toxicity.ipynb) | Review saved results, validate frozen data and run real checkpoint inference |
| [Data_Downloads.ipynb](Data_Downloads.ipynb) | Separate verified CGA, ToxiChat and MACD download/preparation cells |
| [LoRA_Fine_Tuning.ipynb](LoRA_Fine_Tuning.ipynb) | Review; training requires explicit opt-in and a working GPU |
| `notebooks/Context_Aware_Toxicity_Completed_Run.ipynb` | Preserved historical run |

Earlier versions of all three notebooks passed live Jupyter-kernel execution. The main and LoRA notebooks have since been updated for the user-run workflow; their old outputs were cleared. Cell timings and actual-checkpoint ASGI API checks are in `reports/submission_verification.json`. A real localhost HTTP server also passed health, webpage, API documentation and prediction checks (`reports/submission_http_verification.json`). A visual browser demonstration remains a manual presentation check.

## Training and additional data

The GPU recovered. The brief run was stopped on user request. To start the current cleaned-data experiment yourself:

```powershell
python -m contextual.adapter_finetune --config configs/contextual_clean_lora.json
```

The same command resumes the last completed epoch with matching configuration/data. The corrected implementation shares lossless chunk pooling across training/evaluation/inference, uses full validation and the full training pool, and verifies merged-model reload. It rejects unrelated existing checkpoints. No full v2 accuracy result exists until this run completes.

Additional public data is isolated from the frozen benchmark:

```powershell
python -m contextual.external_data --download --prepare
python -m contextual.experiments --config configs/contextual_external.json
python -m contextual.adapter_finetune --config configs/contextual_external.json
```

The first two steps were executed; external neural training remains pending. There are 34,556 prepared examples, including 2,623 with parents. MACD Telugu examples are isolated comments. Source URLs, pinned revisions, label mappings and hashes are recorded. New grouped splits are not official benchmark splits. See the handoff for sources and attribution.

## Inference

```python
from contextual.config import load_config, location
from contextual.inference import Predictor
cfg = load_config("configs/demo.json")
model = Predictor(location(cfg, "checkpoint"), device="cpu")
result = model.predict("Target reply", history=["Earlier parent"], include_embeddings=True)
```

`history` is ordered oldest to newest. `predict_thread(messages, target_id)` follows only true ancestors, never sibling replies or future children. HTTP routes are `POST /api/moderate` and `POST /api/thread`. A topic field is rejected. Missing models return HTTP 503.

Custom-thread JSONL uses `id`, `parent_id`, `conversation_id`, `text` and optional target `label` (0 safe, 1 toxic). Referenced ancestors must exist; duplicate IDs/cycles are rejected. Current configs exclude the authored challenge from training via `include_custom_threads: false`. Add new annotated training data only in a new version with new output paths.

## Evidence and limits

- Original benchmark: 78,839 examples; train/validation/test membership is checked against a fixed data hash.
- Rerun TF-IDF remains strongest overall among measured original-benchmark models (F1 0.717).
- Historical LoRA contextual F1 is 0.696 versus 0.665 with parents removed from the same model; historical selection/sampling limitations prevent strong architectural claims.
- Authored Telugu challenge F1 is 0.596; independent Telugu conversation annotation and an untouched test set remain needed.
- The report distinguishes English personal attacks, Tamil offensiveness and Telugu prompt safety. Scores from different data populations are not directly comparable.
- Thresholds are selected on validation F1; attention is not causal explanation. No guarantee of sarcasm or banter understanding is made.

All project-relative paths resolve against the repository. Choose configs with `--config` or `TOXICITY_CONFIG`. Use `TOXICITY_PROJECT_ROOT` when launching notebooks elsewhere. Download failures, hardware failures and mismatched data/configurations stop explicitly rather than producing invented outputs.
