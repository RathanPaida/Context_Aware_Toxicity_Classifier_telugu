# Indic Shield AI: Context-Aware Toxicity Classifier for Indian & Telugu Code-Mixed Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

> **Two-stage moderation cascade: a lexicon gate (<1ms) plus a context-aware IndicBERT cross-encoder with extra multi-head attention, for Telugu, Tanglish and other Indic code-mixed text.**
>
> ⚠️ Status: the code pipeline is fixed and runs end to end; the model must be (re)trained and evaluated before any metric is quoted. See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md).

---

## 🌟 Overview

Evaluating isolated social media comments fails when toxicity depends on conversational flow, sarcasm, or regional idioms. In Indian languages—specifically **Telugu** and **Romanized Telugu (Tanglish/Tenglish)**—this problem is compounded by script variations, code-mixing, and context inversion.

### Target Contextual Challenges:
1. **Benign Banter vs Threat (Polarity Flip):**
   - *Gaming banter:* `"Ninna match lo 1v4 clutch chusava bro?"` → `"Champesav po!"` is **benign praise** ("You killed it / amazing!").
   - *Violent context:* `"Aaa journalist bayataki vasthe kottali"` → `"Champesav po!"` is an **explicit threat** ("Kill them off!").
2. **Cuss Words / Slang Used as Positive Exclamations:**
   - `"Bro that was fucking insane clutch! You are god tier!"` → Allowed (Multi-Head Attention recognizes high-energy gaming appreciation).
3. **Zero-Cuss Violent Threats:**
   - `"Repu mee batch lo okkadu kooda hospital lekunda undadu, chusukondi"` → Flagged (Contextual threat identified from sentence semantics despite zero profanity).
4. **Parent-Bleed Robustness (De-escalation):**
   - After extreme threats or slurs, polite de-escalations like `"Shantinchandi bro, vadileyandi andi"` are protected from parent toxicity bleed.
5. **Caste & Regional Identity Safety:**
   - Mentions of communities or regions (e.g. *Reddy, Kamma, Mala, Madiga, Andhra, Telangana*) should not be flagged (measured as subgroup false-positive rate).

---

## 🏗️ Architecture & Pipeline

```
                                 [ Incoming Utterance ]
                                           │
                             [ IndicPreprocessor Clean ]
                             (Regex De-obf, IndicNLP)
                                           │
                        ┌──────────────────┴──────────────────┐
                        │                                     │
           [ Stage 1: Fast Target Filter ]                    │
           - Multilingual Profanity Lexicons                  │
           - Compound Slurs & Leetspeak Exits                 │
                        │                                     │
             ┌──────────┴──────────┐                          │
      p > 0.90 (Slur)       p < 0.15 (Safe)                   │
             │                     │                          │
       🚨 FLAGGED             ✅ ALLOWED                      │
          (<1ms)                 (<1ms)                       │
                                                              │
                                       0.15 <= p <= 0.90 (Ambiguous / Contextual)
                                                              │
                                                [ Thread History Retrieval ]
                                                (Automatic Conversational Context)
                                                              │
                                                [ Stage 2: 8-Head Transformer Attention ]
                                                - Canonical Format:
                                                  [CLS] Parent [SEP] Target [SEP]
                                                - 8-Head Contextual Self-Attention
                                                - 8-Head Target-to-Context Cross-Attention
                                                - Attentive Whole-Text & Target Saliency
                                                - Multi-Label Risk Heads (5 Categories)
                                                - Asymmetric Focal Loss
                                                              │
                                                    ┌─────────┴─────────┐
                                              p >= Threshold      p < Threshold
                                                    │                   │
                                               🚨 FLAGGED          ✅ ALLOWED
                                                 (~17ms)             (~17ms)
```

### 1. Stage 1: Fast Target Filter (<1ms)
- Instant lookup with comprehensive multi-lingual lexicons (English, Hindi, Telugu, and Romanized Tanglish).
- Normalizes punctuation (`lanja!`, `fuck!`) and obfuscations (`f*ck`, `m@drch0d`).
- Short-circuits explicit abuse in <1ms without GPU overhead.

### 2. Stage 2: 8-Head Multi-Head Attention Cross-Encoder (~17ms)
- **Backbone:** `ai4bharat/IndicBERTv2-MLM-Sam-TLM` (768 hidden dimensions).
- **8-Head Contextual Self-Attention:** Captures relational dynamics across the entire conversation window.
- **8-Head Target-to-Context Cross-Attention:** Uses target tokens as Queries ($Q$) attending to parent context ($K, V$), isolating contextual intent.
- **Attentive Whole-Text & Target Span Pooling:** Computes token-level saliency weights for interpretability and token heatmaps.
- **Multi-Perspective Classification Head:** 3-layer deep MLP fusing [CLS], whole-text attentive pooling, and cross-attended target representations.
- **Asymmetric Class-Weighted Focal Loss:** Mitigates severe class imbalance on rare violent threats and insults.

---

## 🔬 4-Stage Transfer Learning Process

1. **Stage 1: Context Mechanics (Open English Threads)**
   - English multi-label toxicity on `toxic-chat` (user prompts) and `jigsaw-unintended-bias`.
   - Neither dataset has real conversation threads, so this stage learns toxicity cues and identity debiasing, not context use.
2. **Stage 2: Telugu Domain Adaptation (MLM)**
   - Continued Masked Language Modeling on `IndicCorpV2 Telugu` (250K sentences).
   - Adapts model vocabulary and representations to modern native Telugu script.
3. **Stage 3: Code-Mix & Slang Fine-Tuning**
   - Fine-tuned on `IsaacRodgz/DravidianCodeMix-Dataset` (**Tamil** code-mixed offensive language) + `Cultural-Safety-Telugu` (toxicity-related categories only).
   - Initialised from the Stage 1 classifier with the Stage 2 Telugu-adapted encoder.
4. **Stage 4: Synthetic Thread Fusion**
   - Contrastive twin pairs (same target, safe vs toxic parent) from `src/resources/stage4_synthetic.json`, validated on held-out templates, with Stage 3 replay.

---

## 🚀 Quick Start

### Train (on any NVIDIA GPU machine — see [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md))

```bash
pip install -r requirements.txt
python run_pipeline.py --check-env      # GPU, disk, internet, other GPU processes
python run_pipeline.py --all            # download → stages 1-4 → release → evaluation → baselines
python run_pipeline.py --all --resume   # continue after any interruption
```

Batch size and memory savers are picked automatically for the GPU (`src/hardware.py`). Every stage saves a resume point
each epoch and a `DONE` marker when finished. After Stage 4, the better of Stage 3 / Stage 4 *on real validation data*
is copied to `models/checkpoints/release/`, which is what the app serves (~0.6 GB, fp16).
`--preset quick` runs the whole pipeline on tiny subsets in a few minutes (its model is not usable).

### Run the app

```bash
python app/server.py                    # http://localhost:7860  (no Node needed)
# or the Next.js version, same screens:
cd frontend && npm install && npm run dev   # http://localhost:3000  (needs the Python server running)
```

Both interfaces have the same three screens:

- **Check a message** — add the earlier messages (optional) and the message to check. You get *Allowed / Flagged /
  Needs review*, a one-sentence reason, the harm score against the decision line, scores per type of harm, and
  (optionally) the same message checked *without* the earlier messages side by side.
- **Conversation** — a chat where every new message is checked with the ones before it; examples play out turn by turn
  and show expected vs. actual.
- **Results** — trained stages, evaluation and the comparison with simple baselines, read from `reports/`.

Without a trained model the app says so (only the word list works). `ALLOW_UNTRAINED=1` serves a random model for UI
work and is shown with a red warning banner.

### Tests

```bash
python -m pytest tests -q               # no GPU / network needed
```

### Adding your own (Telugu) dataset

Put a `.csv` / `.jsonl` / `.parquet` file in `data/raw/custom/` with a `text` column and either a `label` column
(0/1 or class names such as `Hate` / `Non-Hate`) or any of `toxic, severe_toxic, identity_attack, insult, threat`.
Optional: `context` (earlier turns, oldest first, one per line), `topic`, `split` (`train`/`val`/`test`). Re-run
`--download-only`; the file is then used by Stage 3, Stage 4 replay, the held-out evaluation and the baselines
automatically. Check the logged label mapping.

Native-script Telugu data automatically gets a Romanized copy (`చంపేసావ్ పో!` → `champesav po!`) so the model also
learns Telugu typed in English letters; turn this off with `--no-romanize`.

Paths, thresholds and hyper-parameters are in `src/config.py`; word lists, demo scenarios and evaluation slices are
JSON files in `src/resources/`.

### Notebook

[`Indic_Shield_AI_Complete_Pipeline.ipynb`](Indic_Shield_AI_Complete_Pipeline.ipynb) imports from `src/` and shows the
data, the model inputs, the measured results and an attention view for any message.

---

## 📊 Evaluation

Metrics are **measured, not hard-coded**. After training run:

```bash
python run_pipeline.py --evaluate --baselines
```

This writes `reports/eval_results.json` (shown in the web app's Benchmarks tab) with:

| Slice | What it measures | Release gate |
| :--- | :--- | :---: |
| Held-out test split | Macro-F1 / ROC-AUC on the real Stage 3 test data (never trained on) | — |
| Contrastive context flip | Same target, safe vs toxic parent: both correct | ≥ 82% |
| Identity & caste mentions | Subgroup false-positive rate on neutral sentences | ≤ 3.5% |
| Parent-bleed robustness | Polite reply after a toxic parent is allowed | ≥ 92% |
| Script parity | \|ΔF1\| Romanized vs native Telugu | ≤ 2% |
| Latency | p50 of the model path on the evaluation machine | ≤ 25 ms |

`--baselines` writes `reports/baselines.json`: lexicon-only and char-TF-IDF + logistic regression on the same held-out
test split, plus a context ablation (same targets with vs. without the parent). Measured so far (no transformer trained yet):

| Method (held-out test, `toxic`, n=4,610) | F1 | ROC-AUC |
| :--- | :---: | :---: |
| Lexicon only | 0.023 | – |
| Char TF-IDF + LogReg | 0.688 | 0.895 |

The contextual slices (`src/resources/eval_slices.json`) are small hand-written sanity checks (6–45 items each); report them with their *n*. Stage 4 training data is automatically de-duplicated against them.

> The numbers previously listed here (100% flip match, 0.0% FPR, 16.8 ms on an RTX 4060) were not reproducible: the evaluation items were copies of the training templates and the web app returned hard-coded values.

---

## 📁 Repository Structure

```
├── app/
│   ├── server.py                 # FastAPI: /api/moderate, /api/health, /api/benchmarks, /api/baselines, /api/scenarios
│   └── static/                   # web page served at :7860 (index.html, app.js, style.css — no build step)
├── frontend/                     # Next.js version of the same page (src/components/*, src/lib/api.js)
├── src/
│   ├── config.py                 # paths, presets, hyper-parameters, thresholds
│   ├── hardware.py               # GPU detection → training settings; --check-env
│   ├── data_downloader.py        # public datasets + data/raw/custom/ → data/processed/
│   ├── preprocessing.py          # de-obfuscation, normalisation, Romanized-Telugu conversion
│   ├── model.py                  # cross-encoder, input assembly, fp16/atomic checkpoints
│   ├── training_utils.py         # shared training loop, metrics, threshold tuning, resume
│   ├── stage1_context_mechanics.py … stage4_synthetic_fusion.py
│   ├── evaluation.py             # release selection + evaluation → reports/eval_results.json
│   ├── baselines.py              # word list / TF-IDF / model + context ablation → reports/baselines.json
│   ├── inference_pipeline.py     # word-list gate + model, used by the server
│   └── resources/                # lexicons, demo scenarios, evaluation slices, Stage 4 pairs (JSON)
├── tests/                        # pytest, no GPU / network
├── docs/TRAINING_GUIDE.md        # how to train on another computer
├── run_pipeline.py               # one command for everything
└── requirements.txt
```

---

## 📜 License
Apache 2.0 / MIT
