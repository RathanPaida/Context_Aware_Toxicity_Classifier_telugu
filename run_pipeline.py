"""
Master pipeline: Context-Aware Indic & Telugu Toxicity Classifier.

  Phase 1  download / format datasets            --download-only
  Phase 2  preprocessing smoke test              --preprocess-test
  Phase 4  Stage 1 English toxicity mechanics    --stage1
           Stage 2 Telugu MLM adaptation         --stage2
           Stage 3 Dravidian code-mix fine-tune  --stage3
           Stage 4 synthetic thread fusion       --stage4
  Phase 5  evaluation → reports/eval_results.json  --evaluate
           baselines + context ablation → reports/baselines.json  --baselines
  Phase 6  scenario demo                         --demo

Checkpoint chain: stage1_best → stage2_best/encoder → stage3_best (stage1 heads + stage2 encoder) → stage4_best.
`--quick-run` is a smoke test on tiny subsets (settings in src/config.py: QUICK_RUN); its checkpoints are not usable models.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import (
    CHECKPOINT_DIR, PRIMARY_MODEL, QUICK_RUN, Stage1Config, Stage2Config, Stage3Config, Stage4Config, get_logger,
)

logger = get_logger("pipeline")


def banner(title: str):
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)


def pick(cli_value, default):
    return default if cli_value is None else cli_value


def run_download(args):
    banner("PHASE 1: DATASET INGESTION & FORMATTING")
    from src.data_downloader import build_stage1_dataset, download_and_format_all
    res = download_and_format_all(
        skip_indic_corp=args.skip_indic_corp,
        jigsaw_sample_size=args.jigsaw_sample,
        telugu_sample_size=QUICK_RUN["stage2_sample_size"] if args.quick_run else pick(args.telugu_sample, Stage2Config.telugu_sample_size),
    )
    if "toxic_chat" in res:
        jig = res.get("jigsaw_bias")
        if jig is None:
            logger.warning("Jigsaw missing — Stage 1 will use toxic-chat only")
            jig = res["toxic_chat"].iloc[:0]
        build_stage1_dataset(res["toxic_chat"], jig)


def run_preprocessing_test(_args):
    banner("PHASE 2: PREPROCESSING SMOKE TEST")
    from src.preprocessing import smoke_test
    smoke_test()


def run_stage1(args):
    banner("STAGE 1: ENGLISH TOXICITY MECHANICS")
    from src.stage1_context_mechanics import train_stage1
    q = args.quick_run
    cfg = Stage1Config(
        learning_rate=pick(args.lr, Stage1Config.learning_rate),
        num_epochs=QUICK_RUN["stage1_epochs"] if q else pick(args.epochs, Stage1Config.num_epochs),
        batch_size=pick(args.batch_size, Stage1Config.batch_size),
        max_samples=QUICK_RUN["stage1_max_samples"] if q else args.max_samples,
    )
    train_stage1(cfg, args.model)


def run_stage2(args):
    banner("STAGE 2: TELUGU MLM DOMAIN ADAPTATION")
    from src.stage2_telugu_mlm import train_stage2
    q = args.quick_run
    cfg = Stage2Config(
        learning_rate=pick(args.lr_mlm, Stage2Config.learning_rate),
        max_steps=QUICK_RUN["stage2_max_steps"] if q else pick(args.max_steps, Stage2Config.max_steps),
        batch_size=pick(args.batch_size_mlm, Stage2Config.batch_size),
        telugu_sample_size=QUICK_RUN["stage2_sample_size"] if q else pick(args.telugu_sample, Stage2Config.telugu_sample_size),
    )
    if q:
        cfg.warmup_steps = 10
        cfg.eval_steps = cfg.save_steps = cfg.max_steps
    stage1 = args.stage1_checkpoint or (str(CHECKPOINT_DIR / "stage1_best") if (CHECKPOINT_DIR / "stage1_best" / "model.pt").exists() else None)
    train_stage2(cfg, args.model, stage1)


def run_stage3(args):
    banner("STAGE 3: DRAVIDIAN CODE-MIX FINE-TUNING")
    from src.stage3_codemix_slang import train_stage3
    q = args.quick_run
    cfg = Stage3Config(
        learning_rate=pick(args.lr3, Stage3Config.learning_rate),
        num_epochs=QUICK_RUN["stage3_epochs"] if q else pick(args.epochs3, Stage3Config.num_epochs),
        batch_size=pick(args.batch_size, Stage3Config.batch_size),
    )
    train_stage3(cfg, args.model, QUICK_RUN["stage3_max_samples"] if q else args.max_samples)


def run_stage4(args):
    banner("STAGE 4: SYNTHETIC THREAD FUSION")
    from src.stage4_synthetic_fusion import train_stage4
    q = args.quick_run
    cfg = Stage4Config(
        learning_rate=pick(args.lr4, Stage4Config.learning_rate),
        num_epochs=QUICK_RUN["stage4_epochs"] if q else pick(args.epochs4, Stage4Config.num_epochs),
        batch_size=min(pick(args.batch_size, Stage4Config.batch_size), Stage4Config.batch_size),
    )
    train_stage4(cfg, args.model, replay_rows=200 if q else 2000)


def run_evaluation(_args):
    banner("PHASE 5: EVALUATION")
    from src.evaluation import run_evaluation_suite
    run_evaluation_suite()


def run_baselines(_args):
    banner("BASELINES & CONTEXT ABLATION")
    from src.baselines import run_baselines as rb
    rb()


def run_demo(_args):
    banner("PHASE 6: SCENARIO DEMO")
    from src.inference_pipeline import run_inference_demo
    run_inference_demo()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag, h in [("--all", "run every phase"), ("--download-only", "phase 1"), ("--preprocess-test", "phase 2"),
                    ("--stage1", "stage 1"), ("--stage2", "stage 2"), ("--stage3", "stage 3"), ("--stage4", "stage 4"),
                    ("--evaluate", "phase 5"), ("--baselines", "lexicon / TF-IDF baselines + context ablation"),
                    ("--demo", "phase 6"), ("--quick-run", "tiny smoke-test settings")]:
        p.add_argument(flag, action="store_true", help=h)
    p.add_argument("--model", default=PRIMARY_MODEL)
    # Leave defaults as None so each stage's own config default applies.
    p.add_argument("--lr", type=float, help="Stage 1 learning rate")
    p.add_argument("--epochs", type=int, help="Stage 1 epochs")
    p.add_argument("--lr3", type=float, help="Stage 3 learning rate")
    p.add_argument("--epochs3", type=int, help="Stage 3 epochs")
    p.add_argument("--lr4", type=float, help="Stage 4 learning rate")
    p.add_argument("--epochs4", type=int, help="Stage 4 epochs")
    p.add_argument("--batch-size", type=int, help="batch size for stages 1/3/4")
    p.add_argument("--max-samples", type=int, help="cap on training rows (stages 1/3)")
    p.add_argument("--lr-mlm", type=float)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size-mlm", type=int)
    p.add_argument("--telugu-sample", type=int)
    p.add_argument("--skip-indic-corp", action="store_true")
    p.add_argument("--jigsaw-sample", type=int, default=200_000)
    p.add_argument("--stage1-checkpoint", default=None)
    args = p.parse_args()

    phases = [args.download_only, args.preprocess_test, args.stage1, args.stage2, args.stage3, args.stage4, args.evaluate, args.baselines, args.demo]
    if not (args.all or any(phases)):
        p.print_help()
        return

    import torch
    logger.info(f"Backbone: {args.model} | device: {'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} "
                f"| quick-run: {args.quick_run}")
    t0 = time.time()
    steps = [
        (args.download_only, run_download), (args.preprocess_test, run_preprocessing_test),
        (args.stage1, run_stage1), (args.stage2, run_stage2), (args.stage3, run_stage3), (args.stage4, run_stage4),
        (args.evaluate, run_evaluation), (args.baselines, run_baselines), (args.demo, run_demo),
    ]
    try:
        for selected, fn in steps:
            if args.all or selected:
                fn(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    logger.info(f"Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
