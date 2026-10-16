"""
Master Pipeline Orchestrator: Context-Aware Indic & Telugu Toxicity Classifier.

Executes the full 6-phase engineering blueprint:
  Phase 1: Dataset Ingestion & Formatting (Zero-Gated Open Strategy)
  Phase 2: Indic-Specific Preprocessing & Normalization Smoke Tests
  Phase 3: Model Architecture Setup (IndicBERT-v3 + Target-Span Pooling + Focal Loss)
  Phase 4: 4-Stage Cross-Lingual Transfer Curriculum:
           - Stage 1: Context Mechanics (toxic-chat + Jigsaw Bias)
           - Stage 2: Telugu MLM Domain Adaptation (IndicCorp 250K)
           - Stage 3: Dravidian Code-Mix & Telugu Slang Fine-Tuning
           - Stage 4: Synthetic Thread Fusion (Contrastive Twin Pairs)
  Phase 5: Indian-Specific Contextual Evaluation Protocol (4 Release Gates)
  Phase 6: Stateful Two-Stage Production Inference Cascade Demo
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.config import (
    Stage1Config,
    Stage2Config,
    Stage3Config,
    Stage4Config,
    PRIMARY_MODEL,
    PROCESSED_DIR,
    CHECKPOINT_DIR,
    get_logger,
)

logger = get_logger("pipeline")


def run_download(args):
    """Phase 1: Download and format all datasets."""
    logger.info("=" * 70)
    logger.info("PHASE 1: DATASET INGESTION & FORMATTING (ZERO-GATED)")
    logger.info("=" * 70)
    
    from src.data_downloader import download_and_format_all, build_stage1_dataset
    
    results = download_and_format_all(
        skip_indic_corp=args.skip_indic_corp,
        jigsaw_sample_size=args.jigsaw_sample,
    )
    
    if "toxic_chat" in results and "jigsaw_bias" in results:
        build_stage1_dataset(results["toxic_chat"], results["jigsaw_bias"])
    
    return results


def run_preprocessing_test(args):
    """Phase 2: Run preprocessing pipeline smoke test."""
    logger.info("=" * 70)
    logger.info("PHASE 2: INDIC-SPECIFIC PREPROCESSING & NORMALIZATION TEST")
    logger.info("=" * 70)
    
    from src.preprocessing import smoke_test
    smoke_test()


def run_stage1(args):
    """Stage 1: Context Mechanics Training."""
    logger.info("=" * 70)
    logger.info("PHASE 4 - STAGE 1: CONTEXT MECHANICS TRAINING (ENGLISH THREADS)")
    logger.info("=" * 70)
    
    from src.stage1_context_mechanics import train_stage1
    
    batch_size = 16 if args.quick_run else (args.batch_size if args.batch_size else 16)
    epochs = 1 if args.quick_run else (args.epochs if args.epochs else 3)
    max_samples = 2000 if args.quick_run else None
    
    config = Stage1Config(
        learning_rate=args.lr if args.lr else 2e-5,
        num_epochs=epochs,
        batch_size=batch_size,
        max_samples=max_samples,
    )
    model_name = args.model if args.model else PRIMARY_MODEL
    train_stage1(config=config, model_name=model_name)


def run_stage2(args):
    """Stage 2: Telugu MLM Domain Adaptation."""
    logger.info("=" * 70)
    logger.info("PHASE 4 - STAGE 2: TELUGU MLM DOMAIN ADAPTATION (INDICCORP)")
    logger.info("=" * 70)
    
    from src.stage2_telugu_mlm import train_stage2
    
    max_steps = 100 if args.quick_run else (args.max_steps if args.max_steps else 10000)
    sample_size = 2000 if args.quick_run else (args.telugu_sample if args.telugu_sample else 250000)
    
    config = Stage2Config(
        learning_rate=args.lr_mlm if args.lr_mlm else 5e-5,
        max_steps=max_steps,
        batch_size=args.batch_size_mlm if args.batch_size_mlm else 16,
        telugu_sample_size=sample_size,
    )
    
    model_name = args.model if args.model else PRIMARY_MODEL
    stage1_ckpt = args.stage1_checkpoint
    if stage1_ckpt is None:
        best_ckpt = CHECKPOINT_DIR / "stage1_best"
        if best_ckpt.exists():
            stage1_ckpt = str(best_ckpt)
            logger.info(f"  Auto-detected Stage 1 checkpoint: {stage1_ckpt}")
    
    train_stage2(config=config, model_name=model_name, stage1_checkpoint=stage1_ckpt)


def run_stage3(args):
    """Stage 3: Dravidian Code-Mix & Telugu Slang Fine-Tuning."""
    logger.info("=" * 70)
    logger.info("PHASE 4 - STAGE 3: DRAVIDIAN CODE-MIX & TELUGU SLANG FINE-TUNING")
    logger.info("=" * 70)
    
    from src.stage3_codemix_slang import train_stage3
    
    epochs = 1 if args.quick_run else 4
    max_samples = 1500 if args.quick_run else None
    
    config = Stage3Config(
        learning_rate=1e-5,
        num_epochs=epochs,
        batch_size=16,
        freeze_ratio=0.5,
    )
    
    # Auto-detect base checkpoint from previous stages
    base_ckpt = None
    for ckpt in ["stage1_best", "stage2_final"]:
        p = CHECKPOINT_DIR / ckpt
        if (p / "model.pt").exists() or (p / "model").exists():
            base_ckpt = str(p)
            break
            
    train_stage3(config=config, model_name=args.model, base_checkpoint=base_ckpt, max_samples=max_samples)


def run_stage4(args):
    """Stage 4: Synthetic Thread Fusion."""
    logger.info("=" * 70)
    logger.info("PHASE 4 - STAGE 4: SYNTHETIC THREAD FUSION (CONTRASTIVE TWINS)")
    logger.info("=" * 70)
    
    from src.stage4_synthetic_fusion import train_stage4
    
    epochs = 3 if args.quick_run else 3
    n_pairs = 300 if args.quick_run else 1500
    
    config = Stage4Config(
        learning_rate=3e-5 if args.quick_run else 5e-6,
        num_epochs=epochs,
        batch_size=8,
        freeze_ratio=0.75,
    )
    
    stage3_ckpt = str(CHECKPOINT_DIR / "stage3_best") if (CHECKPOINT_DIR / "stage3_best").exists() else None
    train_stage4(config=config, model_name=args.model, stage3_checkpoint=stage3_ckpt, n_pairs=n_pairs)


def run_evaluation(args):
    """Phase 5: Indian-Specific Contextual Evaluation Protocol."""
    logger.info("=" * 70)
    logger.info("PHASE 5: INDIAN-SPECIFIC CONTEXTUAL EVALUATION PROTOCOL")
    logger.info("=" * 70)
    
    from src.evaluation import run_evaluation_suite
    from src.model import ContextAwareToxicityClassifier, load_tokenizer
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(args.model)
    model = ContextAwareToxicityClassifier(model_name=args.model)
    
    # Check for highest stage checkpoint
    for ckpt in ["stage4_best", "stage3_best", "stage1_best"]:
        p = CHECKPOINT_DIR / ckpt / "model.pt"
        if p.exists():
            logger.info(f"Loading checkpoint: {p}")
            model.load_state_dict(torch.load(p, map_location="cpu"), strict=False)
            break
            
    model.to(device)
    run_evaluation_suite(model, tok, device)


def run_inference_demo(args):
    """Phase 6: Stateful Two-Stage Production Inference Cascade Demo."""
    logger.info("=" * 70)
    logger.info("PHASE 6: TWO-STAGE PRODUCTION INFERENCE CASCADE DEMO")
    logger.info("=" * 70)
    
    from src.inference_pipeline import run_inference_demo as exec_demo
    exec_demo()


def main():
    parser = argparse.ArgumentParser(
        description="Context-Aware Indic & Telugu Toxicity Classifier Master Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Pipeline stages
    parser.add_argument("--all", action="store_true", help="Execute full pipeline (all 6 phases)")
    parser.add_argument("--download-only", action="store_true", help="Phase 1: Ingest and format datasets")
    parser.add_argument("--preprocess-test", action="store_true", help="Phase 2: Preprocessing smoke test")
    parser.add_argument("--stage1", action="store_true", help="Stage 1: Context Mechanics training")
    parser.add_argument("--stage2", action="store_true", help="Stage 2: Telugu MLM Domain Adaptation")
    parser.add_argument("--stage3", action="store_true", help="Stage 3: Dravidian Code-Mix & Telugu Slang")
    parser.add_argument("--stage4", action="store_true", help="Stage 4: Synthetic Thread Fusion")
    parser.add_argument("--evaluate", action="store_true", help="Phase 5: Contextual Evaluation Protocol")
    parser.add_argument("--demo", action="store_true", help="Phase 6: Production Inference Cascade Demo")
    
    # Execution mode
    parser.add_argument("--quick-run", action="store_true", help="Fast execution mode with calibrated subsets")
    
    # Hyperparameters
    parser.add_argument("--model", type=str, default=PRIMARY_MODEL, help=f"Model identifier (default: {PRIMARY_MODEL})")
    parser.add_argument("--lr", type=float, default=2e-5, help="Stage 1 LR")
    parser.add_argument("--epochs", type=int, default=3, help="Stage 1 Epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr-mlm", type=float, default=5e-5, help="Stage 2 LR")
    parser.add_argument("--max-steps", type=int, default=10000, help="Stage 2 Max steps")
    parser.add_argument("--batch-size-mlm", type=int, default=16, help="Stage 2 Batch size")
    parser.add_argument("--telugu-sample", type=int, default=250000, help="Telugu sample size")
    parser.add_argument("--skip-indic-corp", action="store_true", help="Skip IndicCorp download")
    parser.add_argument("--jigsaw-sample", type=int, default=200000, help="Jigsaw sample size")
    parser.add_argument("--stage1-checkpoint", type=str, default=None, help="Stage 1 checkpoint path")
    
    args = parser.parse_args()
    
    # If no stage specified, default to quick-run full pipeline
    if not any([args.all, args.download_only, args.preprocess_test, args.stage1, args.stage2, args.stage3, args.stage4, args.evaluate, args.demo]):
        args.all = True
        args.quick_run = True
        logger.info("No specific stage selected; executing full pipeline in calibrated --quick-run mode.")
    
    start_time = time.time()
    
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  CONTEXT-AWARE INDIC TOXICITY CLASSIFIER PIPELINE                ║")
    logger.info("║  Cross-Lingual Transfer Learning Architecture                     ║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info(f"  Backbone: {args.model}")
    logger.info(f"  Hardware: {'CUDA GPU' if __import__('torch').cuda.is_available() else 'CPU'}")
    logger.info(f"  Quick Run Mode: {args.quick_run}")
    logger.info("")
    
    try:
        if args.all or args.download_only:
            run_download(args)
        
        if args.all or args.preprocess_test:
            run_preprocessing_test(args)
        
        if args.all or args.stage1:
            run_stage1(args)
        
        if args.all or args.stage2:
            run_stage2(args)
            
        if args.all or args.stage3:
            run_stage3(args)
            
        if args.all or args.stage4:
            run_stage4(args)
            
        if args.all or args.evaluate:
            run_evaluation(args)
            
        if args.all or args.demo:
            run_inference_demo(args)
        
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user.")
    except Exception as e:
        logger.error(f"\nPipeline execution failed: {e}", exc_info=True)
        raise
    
    elapsed = time.time() - start_time
    logger.info(f"\nTotal pipeline elapsed time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
