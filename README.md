# Context-aware toxicity classifier

The default pipeline embeds **each individual message**, then learns a representation of the **whole parent-to-child reply chain** with a hierarchical Transformer. It scores the target message using its own embedding, its contextual embedding, and the pooled conversation embedding. No conversation-type selector, lexicon bypass, topic-specific rule, or fabricated score is used in the active pipeline.

## Run

Use Python 3.10 or later in the same environment as your notebook kernel:

```powershell
python -m pip install -r requirements-contextual.txt
python run_pipeline.py --all
```

Or open **[Context_Aware_Toxicity.ipynb](Context_Aware_Toxicity.ipynb)** in Jupyter and Run All. It imports the individual Python modules rather than duplicating them. Initial corpus/model download needs network access. A missing download stops with instructions; there is no silently substituted dataset. Full training can take substantial time depending on hardware and data size.

```powershell
python run_pipeline.py --download
python run_pipeline.py --prepare
python run_pipeline.py --train
python run_pipeline.py --evaluate
python run_pipeline.py --predict "Your reply" --parent "Earlier parent message" --embeddings
python app/server.py
python -m pytest tests/test_contextual.py -q
```

Repeat `--parent` oldest to newest. The backend serves a web interface at the host/port configured in JSON (default http://127.0.0.1:7860); API documentation is at `/docs`. The optional Next frontend uses the same API (`cd frontend`, `npm install`, `npm run dev`). Neither UI asks whether the conversation is gaming, politics, or another category.

## Configuration and paths

**[configs/contextual.json](configs/contextual.json)** owns model ID/revision, token budget, turn window, batch sizes, learning rate, split fractions, seed, download URL, size/time limits, server settings, and all data/output paths. Select another file with `--config` or `TOXICITY_CONFIG`. Relative paths resolve against the repository location, **not the current working directory**. A notebook launched outside the repository can use `TOXICITY_PROJECT_ROOT`.

There are no hard-coded predictions, category rules, machine-specific paths, or decision cutoffs in the new workflow. A runnable implementation still necessarily has configurable defaults and mathematical/structural constants (binary labels, tensor axes, arithmetic). The threshold is fitted from validation scores and stored in the checkpoint; hidden input width is inferred from the pretrained encoder.

| Artifact | Default path |
| --- | --- |
| Downloaded CGA archive and extracted corpus | `data/raw/contextual/` |
| Prepared examples and fixed train/val/test membership | `data/processed/contextual.jsonl` |
| Frozen individual-message embedding cache | `data/cache/contextual_embeddings.pt` |
| Encoder, tokenizer, trained head, settings, threshold | `models/checkpoints/contextual/` |
| Data counts and source manifest | `reports/contextual_data_manifest.json` |
| Measured results and context-removal ablation | `reports/contextual_evaluation.json` |

The checkpoint bundles its encoder/tokenizer. Inference after training is local and does not depend on a remote model revision or the original working directory.

## Data

The downloader obtains [Cornell CGA-WIKI](https://convokit.cornell.edu/documentation/awry.html), which contains authentic reply links and per-comment personal-attack labels. It excludes section headers and never uses a future conversation outcome as the current message label. Downloads are atomic and recorded with SHA-256 provenance. Preserve source attribution and review source terms for your intended use.

Existing `data/processed/cultural_safety_telugu.parquet` and `dravidian_codemix.parquet` are reused if present, with their existing binary `toxic` labels. The latter is **Tamil**, not Telugu. These isolated comments supplement supervision; they cannot establish contextual Telugu performance. Their old split assignments are replaced by the new group-safe split, so scores are not comparable to old reports. Missing optional sources are recorded in the manifest. They are not silently downloaded, translated, or synthesized.

For your own conversations, put JSONL at `data/raw/custom_threads.jsonl` (configurable):

```json
{"id":"m1","parent_id":null,"conversation_id":"t1","text":"First message","label":0}
{"id":"m2","parent_id":"m1","conversation_id":"t1","text":"Reply to the first message","label":1}
```

Labels above illustrate the schema only; annotate actual text correctly. `label` can be omitted for an unlabelled ancestor. IDs must be unique, every referenced parent must exist in the same conversation, and cycles are rejected. Provide enough independent conversations and both classes for train/validation/test. Only actual ancestors are used, never sibling branches or future children. Paragraphs remain intact as individual messages; long text uses all token chunks.

## Python API

```python
from contextual.config import load_config, location
from contextual.inference import Predictor

cfg = load_config()
model = Predictor(location(cfg, "checkpoint"))
result = model.predict("Target reply", history=["Earlier parent"], include_embeddings=True)
# result: individual_embeddings, contextual_embeddings, conversation_embedding,
# toxicity_score, threshold, isolated_score, context_score_delta, turn_attention
```

For branching threads use `model.predict_thread(messages, target_id)` with the same node schema. The HTTP equivalents are `POST /api/moderate` with `text`/`history`, and `POST /api/thread` with `messages`/`target_id`. A `topic` field is rejected. An absent model returns HTTP 503, never an untrained verdict.

## Architecture and evaluation limits

- **Token encoder:** configurable pretrained IndicBERT, frozen for cached embeddings. Mean pooling across every content-token chunk gives one vector per message. This is an MLM backbone, not a dedicated sentence-similarity encoder.
- **Conversation encoder:** learned relative-to-window positions, multi-head Transformer layers, attention pooling over the root-to-target chain, and a target-specific classification head. The first retained message can be a truncated ancestor; omitted turn count is reported.
- **Learning:** train-only class weighting, validation average precision for early stopping, validation F1 for threshold selection, then held-out test evaluation by source. Entire conversations, articles and groups sharing any message text stay in one split.
- **Scope:** one binary target-toxicity output. CGA personal attacks, local offensiveness, and toxicity are related but different annotations. There are no unsupported threat/identity subtype scores or separate conversation-toxicity labels.
- **Limitations:** semantic context is learned implicitly, not assigned a named domain. This does not guarantee sarcasm/banter understanding. Multilingual pretraining does not prove Telugu/code-mixed accuracy. Scores are not calibrated confidence estimates. Attention is not causal explanation. Full-context versus target-only ablation tests context sensitivity, not necessarily correctness.

See [docs/CONTEXTUAL_REDESIGN.md](docs/CONTEXTUAL_REDESIGN.md) for the audit and implementation decisions.

## Existing work

Your pre-existing modified notebook, evaluation module and reports were preserved. `src/`, the older Indic Shield notebook, and [docs/LEGACY_README.md](docs/LEGACY_README.md) describe the previous experimental cascade; they are **not imported by the new default pipeline**. Do not mix its checkpoints or metrics with the new model. The canonical notebook is `Context_Aware_Toxicity.ipynb` and canonical modules are in `contextual/`.
