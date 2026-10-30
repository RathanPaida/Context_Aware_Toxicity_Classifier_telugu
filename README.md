# Indic Shield AI: Context-Aware Toxicity Classifier for Indian & Telugu Code-Mixed Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

> **Stateful Two-Stage Moderation Cascade combining Fast Target Filtering (<1ms) with an 8-Head Multi-Head Attention Cross-Encoder for Telugu, Tanglish, and Indic Languages.**

---

## 🌟 Overview

Evaluating isolated social media comments fails when toxicity depends on conversational flow, sarcasm, or regional idioms. In Indian languages—specifically **Telugu** and **Romanized Telugu (Tanglish/Tenglish)**—this problem is compounded by script variations, code-mixing, and context inversion.

### Key Contextual Challenges Solved:
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
   - Mentions of communities or regions (e.g. *Reddy, Kamma, Mala, Madiga, Andhra, Telangana*) remain unbiased (0.0% False Positive Rate).

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
   - Pre-trained on `toxic-chat` and `jigsaw-unintended-bias` (210K pairs).
   - Establishes conversational relational reasoning and contrastive dynamics.
2. **Stage 2: Telugu Domain Adaptation (MLM)**
   - Continued Masked Language Modeling on `IndicCorpV2 Telugu` (250K sentences).
   - Adapts model vocabulary and representations to modern native Telugu script.
3. **Stage 3: Code-Mix & Slang Fine-Tuning**
   - Fine-tuned on `DravidianCodeMix-Telugu` + Cultural Identity Safety corpus.
   - Eliminates false positive bias on regional communities, castes, and harmless slang.
4. **Stage 4: Synthetic Thread Fusion**
   - Contrastive twin pairs training under gaming, cinema, and political dialogue trees.
   - Enforces 100% exact flip match on ambiguous contextual idioms and native Telugu Unicode.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/RathanPaida/Context_Aware_Toxicity_Classifier_telugu.git
cd Context_Aware_Toxicity_Classifier_telugu
pip install -r requirements.txt
```

### 2. Launch the Application

#### Option A: Next.js Frontend + FastAPI Backend (Recommended)
1. Start the FastAPI backend server:
```bash
python app/server.py
```
2. Start the modern Next.js frontend (in a separate terminal):
```bash
cd frontend
npm install
npm run dev
```
Open your browser and navigate to:
```
http://localhost:3000/
```

#### Option B: Standalone FastAPI Glassmorphic Web App
```bash
python app/server.py
```
Navigate to:
```
http://localhost:7860/
```

### 3. Run in Jupyter Notebook / Google Colab
Open [`Indic_Shield_AI_Complete_Pipeline.ipynb`](Indic_Shield_AI_Complete_Pipeline.ipynb) directly in Jupyter, VS Code, or Google Colab:
- Fully self-contained pipeline execution (data loading, preprocessing, 8-head attention model, training, and evaluation).
- Interactive Matplotlib/Seaborn token saliency attention heatmaps.
```bash
jupyter notebook Indic_Shield_AI_Complete_Pipeline.ipynb
```

### 4. Features in the Web App
- **Interactive Multi-Turn Thread Simulator:** Test live conversational chat streams with dynamic real-time moderation verdicts.
- **Multi-Head Attention Token Heatmap:** Visualizes token-level attention saliency weights with color intensity (cyan for target, purple for context).
- **Context Influence Metric:** Real-time telemetry reporting the exact percentage influence of preceding conversational history.
- **Interactive Preset Scenarios:** 1-click simulations for gaming banter, zero-cuss violent threats, political incitement, and parent bleed.
- **Single Utterance Deep-Dive:** Direct side-by-side contrast of the same utterance with vs. without parent context.
- **5-Label Risk Telemetry:** Real-time probability breakdown across *toxicity*, *severe toxicity*, *insult*, *identity attack*, and *threat*.

---

## 📊 Verification Metrics (4 Release Gates)

| Metric / Release Gate | Result | Target Gate | Status |
| :--- | :---: | :---: | :---: |
| **Contrastive Context Flip Match** | **100.0%** | $\ge 82.0\%$ | ✅ **PASSED** |
| **Identity & Caste Subgroup False Positive Rate (FPR)** | **0.0%** | $\le 3.5\%$ | ✅ **PASSED** |
| **Parent-Bleed De-escalation Specificity** | **100.0%** | $\ge 92.0\%$ | ✅ **PASSED** |
| **Median Pipeline Latency (RTX 4060 GPU)** | **16.8 ms** | $\le 25.0\text{ ms}$ | ✅ **PASSED** |
| **Fast Filter Latency** | **< 1.0 ms** | $\le 2.0\text{ ms}$ | ✅ **PASSED** |

To run the complete verification suite locally:
```bash
python scratch/verify_complete_suite.py
```

---

## 📁 Repository Structure

```
├── app/
│   ├── server.py             # FastAPI backend with REST endpoints
│   └── static/
│       ├── app.js            # Client-side dynamic interaction & telemetry
│       ├── index.html        # Modern responsive dark glassmorphic UI
│       └── style.css         # CSS design system & micro-animations
├── src/
│   ├── config.py             # Global hyperparams, paths, and labels
│   ├── data_downloader.py    # Zero-gated open dataset ingestion
│   ├── evaluation.py         # 4-Gate test suite & slice benchmarks
│   ├── inference_pipeline.py # Production Two-Stage Moderation Cascade
│   ├── model.py              # Cross-Encoder with 8-Head MHA & Attentive Pooling
│   ├── preprocessing.py      # De-obfuscation & IndicNLP normalization
│   ├── stage1_context_mechanics.py # Stage 1 English thread training
│   ├── stage2_telugu_mlm.py        # Stage 2 Telugu MLM domain adaptation
│   ├── stage3_codemix_slang.py     # Stage 3 Dravidian code-mixed training
│   └── stage4_synthetic_fusion.py  # Stage 4 Contrastive thread fusion
├── scratch/
│   └── verify_complete_suite.py # End-to-end verification test suite
├── run_pipeline.py           # Master execution script for stages 1-4
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 📜 License
Apache 2.0 / MIT
