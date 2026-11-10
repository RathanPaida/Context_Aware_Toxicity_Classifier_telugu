"""
Master pipeline: Context-Aware Indic & Telugu Toxicity Classifier.

    python run_pipeline.py --check-env          # is this machine ready?
    python run_pipeline.py --all                # everything, unattended (preset "full")
    python run_pipeline.py --all --resume       # continue after a crash / reboot

Phases (each can also be run alone):
  --download-only   datasets → data/processed/
  --stage1          English toxicity (toxic-chat + Jigsaw/Civil Comments)
  --stage2          Telugu MLM adaptation (IndicCorpV2)
  --stage3          Dravidian code-mix + Telugu (+ data/raw/custom/)
  --stage4          synthetic contrastive threads
  --select-release  pick stage3 vs stage4 on real validation data → checkpoints/release/
  --evaluate        → reports/eval_results.json
  --baselines       → reports/baselines.json
  --demo            scenario demo

Settings come from three layers (later wins): preset (src/config.py PRESETS) → hardware
profile (src/hardware.py, picks batch size etc. for the GPU) → explicit CLI flags.
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
    CHECKPOINT_DIR, PRESETS, PRIMARY_MODEL, RELEASE_CHECKPOINT, REPORTS_DIR,
    Stage1Config, Stage2Config, Stage3Config, Stage4Config, get_logger,
)

logger = get_logger("pipeline")


def banner(title: str):
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)


def pick(cli_value, default):
    return default if cli_value is None else cli_value


def apply_hardware(cfg, args, mlm: bool = False):
    """Hardware profile → config, unless the user passed explicit values."""
    prof = args.profile
    if mlm:
        cfg.batch_size = pick(args.batch_size_mlm, prof.mlm_batch_size)
        cfg.gradient_accumulation_steps = 1 if args.batch_size_mlm else prof.mlm_gradient_accumulation_steps
        cfg.freeze_embeddings = prof.mlm_freeze_embeddings
        cfg.gradient_checkpointing = prof.gradient_checkpointing
        return cfg
    cfg.batch_size = pick(args.batch_size, prof.batch_size)
    cfg.gradient_accumulation_steps = 1 if args.batch_size else prof.gradient_accumulation_steps
    cfg.gradient_checkpointing = prof.gradient_checkpointing
    cfg.freeze_embeddings = prof.freeze_embeddings
    cfg.max_seq_length = prof.max_seq_length
    return cfg


def stage_done(name: str) -> bool:
    from src.training_utils import is_done
    return is_done(CHECKPOINT_DIR / name)


def skip_if_done(args, name: str) -> bool:
    if args.resume and stage_done(name):
        logger.info(f"  {name} already finished (DONE marker) — skipping. Delete it to retrain.")
        return True
    return False


# ── phases ─────────────────────────────────────────────────────────────

def run_check_env(args) -> bool:
    from src.hardware import check_environment
    return check_environment(allow_cpu=args.allow_cpu)


def run_download(args):
    banner("PHASE 1: DATASETS")
    from src.data_downloader import build_stage1_dataset, download_and_format_all
    from src.config import PROCESSED_DIR
    if args.resume and (PROCESSED_DIR / "stage1_train.parquet").exists() and (PROCESSED_DIR / "dravidian_codemix.parquet").exists():
        logger.info("  processed data already present — skipping download (delete data/processed to redo)")
        return
    pr = PRESETS[args.preset]
    res = download_and_format_all(
        skip_indic_corp=args.skip_indic_corp,
        jigsaw_sample_size=pick(args.jigsaw_sample, pr["jigsaw_sample"]),
        telugu_sample_size=pick(args.telugu_sample, pr["telugu_sample"]),
        romanize=not args.no_romanize,
    )
    if "toxic_chat" in res:
        jig = res.get("jigsaw_bias")
        if jig is None:
            logger.warning("Jigsaw/Civil Comments missing — Stage 1 will use toxic-chat only")
            jig = res["toxic_chat"].iloc[:0]
        build_stage1_dataset(res["toxic_chat"], jig)


def run_preprocessing_test(_args):
    banner("PREPROCESSING SMOKE TEST")
    from src.preprocessing import smoke_test
    smoke_test()


def run_stage1(args):
    banner("STAGE 1: ENGLISH TOXICITY")
    if skip_if_done(args, "stage1_best"):
        return
    from src.stage1_context_mechanics import train_stage1
    pr = PRESETS[args.preset]
    cfg = apply_hardware(Stage1Config(
        learning_rate=pick(args.lr, Stage1Config.learning_rate),
        num_epochs=pick(args.epochs, pr["stage1_epochs"]),
        max_samples=pick(args.max_samples, pr["stage1_max_samples"]),
    ), args)
    train_stage1(cfg, args.model, resume=args.resume)


def run_stage2(args):
    banner("STAGE 2: TELUGU MLM ADAPTATION")
    if skip_if_done(args, "stage2_best"):
        return
    from src.stage2_telugu_mlm import train_stage2
    pr = PRESETS[args.preset]
    cfg = apply_hardware(Stage2Config(
        learning_rate=pick(args.lr_mlm, Stage2Config.learning_rate),
        max_steps=pick(args.max_steps, pr["stage2_max_steps"]),
        telugu_sample_size=pick(args.telugu_sample, pr["telugu_sample"]),
        eval_steps=pr["stage2_eval_steps"],
    ), args, mlm=True)
    cfg.warmup_steps = min(cfg.warmup_steps, max(1, cfg.max_steps // 10))
    stage1 = args.stage1_checkpoint or (str(CHECKPOINT_DIR / "stage1_best")
                                        if (CHECKPOINT_DIR / "stage1_best" / "model.pt").exists() else None)
    train_stage2(cfg, args.model, stage1, resume=args.resume)


def run_stage3(args):
    banner("STAGE 3: DRAVIDIAN CODE-MIX + TELUGU")
    if skip_if_done(args, "stage3_best"):
        return
    from src.stage3_codemix_slang import train_stage3
    pr = PRESETS[args.preset]
    cfg = apply_hardware(Stage3Config(
        learning_rate=pick(args.lr3, Stage3Config.learning_rate),
        num_epochs=pick(args.epochs3, pr["stage3_epochs"]),
    ), args)
    train_stage3(cfg, args.model, pick(args.max_samples, pr["stage3_max_samples"]), resume=args.resume)


def run_stage4(args):
    banner("STAGE 4: SYNTHETIC THREAD FUSION")
    if skip_if_done(args, "stage4_best"):
        return
    from src.stage4_synthetic_fusion import train_stage4
    pr = PRESETS[args.preset]
    cfg = apply_hardware(Stage4Config(
        learning_rate=pick(args.lr4, Stage4Config.learning_rate),
        num_epochs=pick(args.epochs4, pr["stage4_epochs"]),
    ), args)
    cfg.batch_size = min(cfg.batch_size, Stage4Config.batch_size)  # tiny dataset: small batches
    cfg.gradient_accumulation_steps = 1
    train_stage4(cfg, args.model, replay_rows=pr["stage4_replay"], resume=args.resume)


def run_select_release(_args):
    banner("RELEASE SELECTION")
    from src.evaluation import select_release_checkpoint
    select_release_checkpoint()


def run_evaluation(_args):
    banner("EVALUATION")
    from src.evaluation import run_evaluation_suite
    run_evaluation_suite()


def run_baselines(_args):
    banner("BASELINES & CONTEXT ABLATION")
    from src.baselines import run_baselines as rb
    rb()


def run_demo(_args):
    banner("SCENARIO DEMO")
    from src.inference_pipeline import run_inference_demo
    run_inference_demo()


def print_handoff():
    rel = CHECKPOINT_DIR / RELEASE_CHECKPOINT
    banner("DONE — copy these back to the machine that runs the app")
    if (rel / "model.pt").exists():
        size = sum(f.stat().st_size for f in rel.rglob("*") if f.is_file()) / 1e9
        logger.info(f"  1. {rel}   (~{size:.2f} GB)  → same path in your project")
    else:
        logger.info(f"  1. {rel} was not created — check the log above")
    logger.info(f"  2. {REPORTS_DIR}   (evaluation + baselines, used by the Results page)")
    logger.info("  Then on your laptop: python app/server.py")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    flags = [("--all", "run every phase"), ("--check-env", "environment check only"),
             ("--download-only", "datasets"), ("--preprocess-test", "preprocessing smoke test"),
             ("--stage1", "stage 1"), ("--stage2", "stage 2"), ("--stage3", "stage 3"), ("--stage4", "stage 4"),
             ("--select-release", "choose stage3/stage4 for the app"), ("--evaluate", "evaluation"),
             ("--baselines", "baselines + context ablation"), ("--demo", "scenario demo"),
             ("--resume", "continue interrupted stages, skip finished ones"),
             ("--allow-cpu", "allow training without a CUDA GPU (very slow)"),
             ("--skip-indic-corp", "no Telugu corpus (Stage 2 is skipped)"),
             ("--no-romanize", "do not add Romanized-Telugu copies of native-script data")]
    for flag, h in flags:
        p.add_argument(flag, action="store_true", help=h)
    p.add_argument("--quick-run", action="store_true", help="alias for --preset quick")
    p.add_argument("--preset", choices=sorted(PRESETS), default="full")
    p.add_argument("--model", default=PRIMARY_MODEL)
    for name, typ, h in [("--lr", float, "stage 1 lr"), ("--epochs", int, "stage 1 epochs"),
                         ("--lr3", float, "stage 3 lr"), ("--epochs3", int, "stage 3 epochs"),
                         ("--lr4", float, "stage 4 lr"), ("--epochs4", int, "stage 4 epochs"),
                         ("--batch-size", int, "override auto batch size (stages 1/3/4)"),
                         ("--max-samples", int, "cap training rows (stages 1/3)"),
                         ("--lr-mlm", float, "stage 2 lr"), ("--max-steps", int, "stage 2 steps"),
                         ("--batch-size-mlm", int, "override stage 2 batch size"),
                         ("--telugu-sample", int, "Telugu sentences for stage 2"),
                         ("--jigsaw-sample", int, "Jigsaw/Civil Comments rows")]:
        p.add_argument(name, type=typ, help=h)
    p.add_argument("--stage1-checkpoint", default=None)
    args = p.parse_args()
    if args.quick_run:
        args.preset = "quick"

    phases = [args.check_env, args.download_only, args.preprocess_test, args.stage1, args.stage2, args.stage3,
              args.stage4, args.select_release, args.evaluate, args.baselines, args.demo]
    if not (args.all or any(phases)):
        p.print_help()
        return

    from src.hardware import detect_profile
    args.profile = detect_profile()
    logger.info(f"Backbone: {args.model} | preset: {args.preset} | resume: {args.resume}")
    logger.info(f"Hardware: {args.profile.describe()}")

    training = args.all or any([args.stage1, args.stage2, args.stage3, args.stage4])
    if args.check_env or args.all:
        ok = run_check_env(args)
        if not ok and (args.all or training):
            logger.error("Environment check failed — fix the problems above (or see docs/TRAINING_GUIDE.md).")
            sys.exit(1)
    elif training and args.profile.device == "cpu" and not args.allow_cpu:
        logger.error("No CUDA GPU — training would take days. Pass --allow-cpu to force it.")
        sys.exit(1)

    t0 = time.time()
    steps = [
        (args.download_only, run_download), (args.preprocess_test, run_preprocessing_test),
        (args.stage1, run_stage1),
        (args.stage2 and not args.skip_indic_corp, run_stage2),
        (args.stage3, run_stage3), (args.stage4, run_stage4),
        (args.select_release, run_select_release),
        (args.evaluate, run_evaluation), (args.baselines, run_baselines), (args.demo, run_demo),
    ]
    try:
        for selected, fn in steps:
            if fn is run_preprocessing_test and args.all:
                continue
            if fn is run_stage2 and args.all and args.skip_indic_corp:
                continue
            if args.all or selected:
                fn(args)
                if args.all:
                    logger.info(f"  [elapsed {(time.time() - t0) / 3600:.1f} h]")
    except KeyboardInterrupt:
        logger.warning("Interrupted — rerun the same command with --resume to continue.")
        sys.exit(130)
    except Exception:
        logger.exception("Pipeline failed — fix the error above, then rerun with --resume to continue.")
        sys.exit(1)
    logger.info(f"Total time: {(time.time() - t0) / 60:.1f} min")
    if args.all:
        print_handoff()


if __name__ == "__main__":
    main()
