# Training guide — running the full training on another computer

This trains the real model, unattended, on any Windows/Linux machine with an NVIDIA GPU.
You then copy **one folder** back to the laptop that runs the app.

## What the other computer needs

| | Minimum | Better |
|---|---|---|
| GPU | NVIDIA, 4 GB VRAM (with CUDA drivers) | 8 GB+ |
| Free disk | 15 GB | 25 GB |
| RAM | 8 GB | 16 GB |
| Internet | Needed for the first run (model ~1.1 GB + datasets ~1.5 GB) | |
| Python | 3.10 – 3.13 | |

The settings (batch size, memory savers) are chosen automatically from the GPU size, so you don't need to tune anything.

## Steps

```bash
# 1. Get the code (USB copy or git clone) and open a terminal in the project folder
cd NLP_project

# 2. Create an environment and install
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build of PyTorch
pip install -r requirements.txt

# 3. Check the machine (GPU, disk, internet, other programs using the GPU)
python run_pipeline.py --check-env

# 4. Train everything (can run overnight; safe to leave alone)
python run_pipeline.py --all
```

Before step 4, **close browsers, games and Jupyter notebooks**. They hold GPU memory, and `--check-env` lists any
processes it finds on the GPU.

When it finishes, the last lines of the log say exactly what to copy, for example:

```
DONE — copy these back to the machine that runs the app
  1. ...\models\checkpoints\release   (~0.58 GB)  → same path in your project
  2. ...\reports   (evaluation + baselines, used by the Results page)
```

## If it stops (power cut, crash, laptop closed)

Run the same command with `--resume`:

```bash
python run_pipeline.py --all --resume
```

- Finished stages are skipped (each writes a `DONE` file in its checkpoint folder).
- An interrupted stage continues from the end of its last completed epoch (Stage 2: from its last evaluation).
- Downloaded data is reused.

To redo one stage from scratch, delete its folder, e.g. `models/checkpoints/stage3_best`, and resume.

## What happens, and roughly how long

Measured on an RTX 3050 6 GB laptop GPU: about 20 training rows per second. Other rows are estimates.

| Step | What it does | 4–6 GB GPU | 8–12 GB GPU | 16 GB+ |
|---|---|---|---|---|
| Download | toxic-chat, Civil Comments (Jigsaw), DravidianCodeMix, Cultural-Safety-Telugu (+ Romanized copies), IndicCorpV2 Telugu | 10–30 min | same | same |
| Stage 1 | English toxicity, ~63k rows × up to 3 epochs | ~3 h | ~1.5 h | ~45 min |
| Stage 2 | Telugu masked-language-model, up to 20k steps (estimate) | ~3 h | ~1.5 h | ~1 h |
| Stage 3 | Dravidian code-mix + Telugu, ~41k rows × up to 6 epochs | ~3.5 h | ~1.8 h | ~1 h |
| Stage 4 | contrastive thread pairs + replay | ~20 min | ~10 min | ~5 min |
| Release + evaluation + baselines | picks Stage 3 or 4 on real data, measures everything | ~10 min | ~5 min | ~5 min |
| **Total** | | **~10 h** | **~5 h** | **~3 h** |

Early stopping usually ends stages sooner. The log prints how many minutes remain after every epoch.

## Back on your laptop

1. Copy `models/checkpoints/release/` into the same place in your project (create the folders if needed).
2. Copy `reports/` over yours.
3. Run `python app/server.py` and open http://localhost:7860. The header should say **Model ready · stage…**.

## Troubleshooting

| Message | Fix |
|---|---|
| `CUDA out of memory` | Close other GPU programs (see `--check-env`). Still failing: `python run_pipeline.py --all --resume --batch-size 4` |
| `Not enough disk space to save …` | Free space, then `--resume`. Each checkpoint is ~0.6 GB; resume state ~1.2 GB per running stage. |
| `no CUDA GPU detected` | Install the CUDA build of PyTorch (step 2). Check `python -c "import torch; print(torch.cuda.is_available())"`. |
| `cannot reach huggingface.co` / HTTP 429 | Check the internet connection; for rate limits run `huggingface-cli login` and retry. |
| Training very slow, GPU memory full | Another process is using the GPU; Windows then spills into normal RAM. Close it and `--resume`. |
| Want a quick test first | `python run_pipeline.py --all --preset quick` (a few minutes; its model is **not** usable). Delete `models/checkpoints/` before the real run. |

## Useful options

```
--preset quick            tiny smoke test of the whole pipeline
--batch-size N            override the automatic batch size (stages 1/3/4)
--skip-indic-corp         skip the Telugu corpus and Stage 2
--no-romanize             don't add Romanized-Telugu copies of native-script data
--allow-cpu               train without a GPU (days — not recommended)
```

Environment variables (optional): `INDIC_SHIELD_DATA_DIR`, `INDIC_SHIELD_MODEL_DIR`, `INDIC_SHIELD_REPORTS_DIR`,
`INDIC_SHIELD_LOG_DIR` to put data/checkpoints on another drive, and `INDIC_SHIELD_MIN_DISK_GB` to change the 15 GB check.
