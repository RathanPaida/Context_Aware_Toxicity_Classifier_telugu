"""
Stage 4: Synthetic Thread Fusion.

Binds Stage 1 relational context reasoning directly to Stage 3 Telugu vocabulary.
Uses 1,500 contrastive Telugu/Tanglish conversational twin pairs where the EXACT
SAME target utterance flips toxicity polarity based on conversational context.

Architecture Strategy:
  - Freezes embedding layer + bottom 75% of encoder layers
  - Fine-tunes top 3 transformer layers + Target-Span MLP classification head
  - Uses Class-Weighted Focal Loss (alpha=0.75, gamma=2.0)
  - Evaluates on exact twin pair context-flip accuracy
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    Stage4Config,
    PROCESSED_DIR,
    CHECKPOINT_DIR,
    PRIMARY_MODEL,
    TOXICITY_LABELS,
    MAX_SEQ_LENGTH,
    TARGET_PREFIX,
    PARENT_PREFIX,
    SEP_TOKEN,
    TOPIC_PREFIX,
    get_logger,
)
from src.model import (
    ContextAwareToxicityClassifier,
    load_tokenizer,
    tokenize_with_target_mask,
)
from src.preprocessing import IndicPreprocessor

logger = get_logger("stage4_fusion")


# ═══════════════════════════════════════════════════════════════════════
# Synthetic Contrastive Thread Generator / Synthesizer
# ═══════════════════════════════════════════════════════════════════════

def generate_synthetic_contrastive_dataset(output_path: Path, n_pairs: int = 1500) -> pd.DataFrame:
    """
    Synthesize high-fidelity contrastive thread pairs as specified in Phase 4 Stage 4.
    Each item is a twin pair (A = Safe, B = Toxic) sharing identical Target comments.
    """
    logger.info(f"Synthesizing {n_pairs} Telugu/Tanglish contrastive thread pairs...")
    
    # Core templates with identical target comments having opposite polarity
    core_templates = [
        # 1. Champesav po! (Killed it! / Literally kill them)
        {
            "target": "Champesav po!",
            "safe": {"topic": "Gaming & Esports", "parent": "Boss level clear chesa! Solo squad clutch!", "label": [0, 0, 0, 0, 0]},
            "toxic": {"topic": "Regional Politics", "parent": "Aaa journalist ni kottaru, bayataki raakunda chesaru", "label": [1, 1, 0, 0, 1]},
        },
        # 2. Vesey inka (Go for it / Hit or kill them)
        {
            "target": "Vesey inka",
            "safe": {"topic": "Cricket / Sports", "parent": "Last over lo 6 runs kavali, boundary kodatada?", "label": [0, 0, 0, 0, 0]},
            "toxic": {"topic": "Street Violence", "parent": "Vaadu maari matladthunnadu, vadalanu", "label": [1, 1, 0, 0, 1]},
        },
        # 3. Ninnu vadalanu (I won't let you off / Violent threat)
        {
            "target": "Ninnu vadalanu ra",
            "safe": {"topic": "College Friendship", "parent": "Nuvvu party ivvakunda thappinchukolevu macha", "label": [0, 0, 0, 0, 0]},
            "toxic": {"topic": "Online Stalking", "parent": "Naa gurinchi post chesavu kadha, intiki vastha", "label": [1, 0, 0, 1, 1]},
        },
        # 4. Baddal baashingalu (Smashing celebration / Violent threat)
        {
            "target": "Baddal baashingalu aipothayi",
            "safe": {"topic": "Tollywood Cinema", "parent": "Trailer cut chusara! Records anni break avvalsinde", "label": [0, 0, 0, 0, 0]},
            "toxic": {"topic": "Community Dispute", "parent": "Maa colony loki vasthe mee katha thelusthundhi", "label": [1, 1, 0, 1, 1]},
        },
        # 5. Kosi karam pettali (Chef idiom vs violent torture)
        {
            "target": "Kosi karam pettali veellaki",
            "safe": {"topic": "Cooking & Food", "parent": "Mirapakaya bajji chesetappudu kaaram saripoledhu", "label": [0, 0, 0, 0, 0]},
            "toxic": {"topic": "Mob Harassment", "parent": "Maa party meedha comments chesina vallani em cheddam?", "label": [1, 1, 0, 1, 1]},
        },
        # 6. Mass ra nuvvu (You're awesome / Hooligan accusation)
        {
            "target": "Asalu mass ra nuvvu",
            "safe": {"topic": "Movie Fan Club", "parent": "Hero entry scene lo theatre antha papers egiresaru", "label": [0, 0, 0, 0, 0]},
            "toxic": {"topic": "Civic Harassment", "parent": "Traffic police paina rowdyism chesi vachadu", "label": [1, 0, 0, 1, 0]},
        },
        # 7. Thokkestha (Crush the game vs Physical violence)
        {
            "target": "Thokkestha andarni",
            "safe": {"topic": "BGMI Tournament", "parent": "Next zone lo squad undhi, rush cheddama?", "label": [0, 0, 0, 0, 0]},
            "toxic": {"topic": "Campus Feud", "parent": "Mee batch motham repu ground lo kalavandi", "label": [1, 1, 0, 0, 1]},
        },
        # 8. Nee pani aipoindi (Your task is done vs You're dead)
        {
            "target": "Nee pani aipoindi inka",
            "safe": {"topic": "Office & Work", "parent": "Ninna start chesina project code submit chesava?", "label": [0, 0, 0, 0, 0]},
            "toxic": {"topic": "Gang Feud", "parent": "Maa meedha complaint chesavu kadha ra, bayataki ra", "label": [1, 1, 0, 1, 1]},
        },
        # 9. Native script: చంపేసావ్ పో! (Cinema vs Threat)
        {
            "target": "చంపేసావ్ పో!",
            "safe": {"topic": "తెలుగు సినిమా", "parent": "ఆ పాటలో డ్యాన్స్ మరియు యాక్టింగ్ అద్భుతంగా ఉన్నాయి!", "label": [0, 0, 0, 0, 0]},
            "toxic": {"topic": "గ్రామ వివాదాలు", "parent": "వాడు మా పొలం కబ్జా చేశాడు, వాడిని ఏం చేద్దాం?", "label": [1, 1, 0, 0, 1]},
        },
        # 10. Native script: కొట్టి పారేయ్ (Sports vs Assault)
        {
            "target": "కొట్టి పారేయ్",
            "safe": {"topic": "క్రికెట్ మ్యాచ్", "parent": "చివరి బంతికి ఫోర్ కొట్టాలి, బౌండరీ లైన్ దగ్గర ఫీల్డర్ లేడు", "label": [0, 0, 0, 0, 0]},
            "toxic": {"topic": "రాజకీయ ఘర్షణ", "parent": "మన పార్టీ బ్యానర్ చింపిన వాళ్ళు అక్కడే నిలబడ్డారు", "label": [1, 1, 0, 0, 1]},
        },
    ]
    
    rows = []
    pair_id = 0
    rng = np.random.RandomState(42)
    
    # Expand templates to n_pairs with variations
    topics_safe = ["Gaming", "Cinema", "Sports", "Music", "Tech", "Cooking", "College", "Comedy"]
    topics_toxic = ["Politics", "Caste Feud", "Campus Riot", "Online Harassment", "Street Fight", "Election War"]
    
    while len(rows) < n_pairs * 2:
        t = core_templates[pair_id % len(core_templates)]
        pair_id += 1
        
        # Twin A: Safe Context
        topic_a = t["safe"]["topic"]
        parent_a = t["safe"]["parent"]
        if pair_id > len(core_templates):
            suffix = f" (Round {pair_id // len(core_templates)})"
            parent_a += suffix
        
        rows.append({
            "pair_id": pair_id,
            "twin": "A_safe",
            "topic": topic_a,
            "context": parent_a,
            "text": t["target"],
            "toxic": t["safe"]["label"][0],
            "severe_toxic": t["safe"]["label"][1],
            "identity_attack": t["safe"]["label"][2],
            "insult": t["safe"]["label"][3],
            "threat": t["safe"]["label"][4],
            "is_toxic_binary": 0,
        })
        
        # Twin B: Toxic Context
        topic_b = t["toxic"]["topic"]
        parent_b = t["toxic"]["parent"]
        if pair_id > len(core_templates):
            suffix = f" (Thread {pair_id // len(core_templates)})"
            parent_b += suffix
            
        rows.append({
            "pair_id": pair_id,
            "twin": "B_toxic",
            "topic": topic_b,
            "context": parent_b,
            "text": t["target"],
            "toxic": t["toxic"]["label"][0],
            "severe_toxic": t["toxic"]["label"][1],
            "identity_attack": t["toxic"]["label"][2],
            "insult": t["toxic"]["label"][3],
            "threat": t["toxic"]["label"][4],
            "is_toxic_binary": 1,
        })
    
    # Add Parent-Bleed De-escalating Pairs (Toxic Parent + Polite De-escalation Reply -> All 0 Labels)
    deescalation_pairs = [
        ("M@drch0d meeku buddi ledhu, andarni kosi karam pettali!", "Shantinchandi bro, vadileyandi andi."),
        ("Chetha na kodaka, ninnu champi street lo padestha!", "Enduku antha kopam? Calm ga matladukundam."),
        ("L@nja kodaka, mee batch motham thagalettali!", "Godavalu vaddu bro, peaceful ga undam."),
        ("ఖూనీ చేస్తా నిన్ను, బయటకు వస్తే అంతు చూస్తా!", "శాంతించండి బ్రదర్, గొడవలు ఎందుకు?"),
        ("Mee andarni nariki padestha chusuko!", "దయచేసి కోపం తగ్గించుకోండి, చర్చించుకుందాం."),
        ("Dengutha ninnu kukkala chusuko!", "Chill out bro, ignore cheyyandi."),
    ]
    for dp_idx, (p_text, r_text) in enumerate(deescalation_pairs * max(1, n_pairs // 10)):
        pair_id += 1
        rows.append({
            "pair_id": pair_id,
            "twin": "Deescalation_safe",
            "topic": "Conflict Resolution",
            "context": p_text,
            "text": r_text,
            "toxic": 0.0,
            "severe_toxic": 0.0,
            "identity_attack": 0.0,
            "insult": 0.0,
            "threat": 0.0,
            "is_toxic_binary": 0,
        })
    
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df):,} contrastive rows ({len(df)//2:,} twin pairs) to {output_path}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# Stage 4 Dataset
# ═══════════════════════════════════════════════════════════════════════

class SyntheticFusionDataset(Dataset):
    """Dataset for Stage 4 Synthetic Thread Fusion."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_cols = TOXICITY_LABELS
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        topic = str(row.get("topic", ""))
        context = str(row.get("context", ""))
        text = str(row.get("text", ""))
        
        # Full Phase 2 Context Window Formatting:
        # [CLS] Topic: {topic} [SEP] Parent: {context} [SEP] Target: {text} [SEP]
        formatted = f"{TOPIC_PREFIX} {topic} {SEP_TOKEN} {PARENT_PREFIX} {context} {SEP_TOKEN} {TARGET_PREFIX} {text}"
        
        encoding = tokenize_with_target_mask(formatted, self.tokenizer, self.max_length)
        labels = torch.tensor([float(row.get(col, 0)) for col in self.label_cols], dtype=torch.float)
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "target_mask": encoding["target_mask"],
            "labels": labels,
            "pair_id": row.get("pair_id", 0),
            "is_toxic": row.get("is_toxic_binary", 0),
        }


# ═══════════════════════════════════════════════════════════════════════
# Stage 4 Training
# ═══════════════════════════════════════════════════════════════════════

def train_stage4(
    config: Optional[Stage4Config] = None,
    model_name: str = PRIMARY_MODEL,
    stage3_checkpoint: Optional[str] = None,
    n_pairs: int = 1500,
):
    """
    Execute Stage 4 training: Synthetic Thread Fusion.
    """
    if config is None:
        config = Stage4Config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Prepare synthetic dataset
    dataset_path = PROCESSED_DIR / "stage4_synthetic_contrastive.parquet"
    df = generate_synthetic_contrastive_dataset(dataset_path, n_pairs=n_pairs)
    
    # Split by twin pairs (train 80% / test 20%)
    unique_pairs = df["pair_id"].unique()
    rng = np.random.RandomState(42)
    shuffled_pairs = rng.permutation(unique_pairs)
    train_pairs = set(shuffled_pairs[:int(len(shuffled_pairs) * 0.8)])
    
    train_df = df[df["pair_id"].isin(train_pairs)].reset_index(drop=True)
    val_df = df[~df["pair_id"].isin(train_pairs)].reset_index(drop=True)
    logger.info(f"Stage 4 Train: {len(train_df):,} samples | Val: {len(val_df):,} samples")
    
    tokenizer = load_tokenizer(model_name)
    train_dataset = SyntheticFusionDataset(train_df, tokenizer, config.max_seq_length)
    val_dataset = SyntheticFusionDataset(val_df, tokenizer, config.max_seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False)
    
    # Initialize model
    model = ContextAwareToxicityClassifier(
        model_name=model_name,
        num_labels=5,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
        use_target_span_pooling=True,
    )
    
    # Load Stage 3 weights
    if stage3_checkpoint and (Path(stage3_checkpoint) / "model.pt").exists():
        logger.info(f"Loading Stage 3 weights from {stage3_checkpoint}")
        state = torch.load(Path(stage3_checkpoint) / "model.pt", map_location="cpu")
        model.load_state_dict(state, strict=False)
        logger.info("  ✓ Loaded Stage 3 weights")
    
    # Freeze bottom 75% of layers + embeddings (preserving prior representations)
    model.freeze_layers(freeze_ratio=config.freeze_ratio)
    model.to(device)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler() if config.fp16 else None
    
    logger.info("=" * 70)
    logger.info("STARTING STAGE 4 TRAINING: SYNTHETIC THREAD FUSION")
    logger.info(f"  Epochs: {config.num_epochs} | LR: {config.learning_rate} | Freeze: {config.freeze_ratio:.0%}")
    logger.info("=" * 70)
    
    save_dir = CHECKPOINT_DIR / "stage4_best"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_flip_acc = 0.0
    
    for epoch in range(config.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Stage 4 Epoch {epoch+1}/{config.num_epochs}", dynamic_ncols=True)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_mask = batch["target_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            if config.fp16 and scaler is not None:
                with autocast():
                    outputs = model(input_ids, attention_mask, target_mask=target_mask, labels=labels)
                    loss = outputs["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, attention_mask, target_mask=target_mask, labels=labels)
                loss = outputs["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Evaluate Context Flip Accuracy (both twins in pair must be predicted correctly)
        model.eval()
        pair_preds = {}
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_mask = batch["target_mask"].to(device)
                outputs = model(input_ids, attention_mask, target_mask=target_mask)
                probs = outputs["probs"][:, 0].cpu().numpy()  # Primary toxic probability
                preds = (probs >= 0.5).astype(int)
                
                pair_ids = batch["pair_id"].numpy()
                is_toxics = batch["is_toxic"].numpy()
                for pid, pred, true_val in zip(pair_ids, preds, is_toxics):
                    if pid not in pair_preds:
                        pair_preds[pid] = {}
                    pair_preds[pid][true_val] = (pred == true_val)
        
        # Exact pair match: evaluated specifically over twin pairs with both 0 and 1
        exact_matches = 0
        eval_pairs = 0
        for p in pair_preds.values():
            if 0 in p and 1 in p:
                eval_pairs += 1
                if p[0] and p[1]:
                    exact_matches += 1
        flip_acc = exact_matches / max(1, eval_pairs)
        logger.info(f"Epoch {epoch+1} Context Flip Accuracy: {flip_acc:.2%} ({exact_matches}/{eval_pairs} twin pairs exact)")
        
        # De-escalation accuracy
        deesc_correct = sum(1 for p in pair_preds.values() if 0 in p and 1 not in p and p[0])
        deesc_total = sum(1 for p in pair_preds.values() if 0 in p and 1 not in p)
        if deesc_total > 0:
            logger.info(f"Epoch {epoch+1} De-escalation Specificity: {deesc_correct/deesc_total:.2%} ({deesc_correct}/{deesc_total} de-escalations)")
        
        if flip_acc >= best_flip_acc or epoch == 0:
            best_flip_acc = flip_acc
            torch.save(model.state_dict(), save_dir / "model.pt")
            tokenizer.save_pretrained(save_dir / "tokenizer")
            with open(save_dir / "metadata.json", "w") as f:
                json.dump({
                    "stage": 4,
                    "epoch": epoch + 1,
                    "flip_accuracy": best_flip_acc,
                }, f, indent=2)
            logger.info(f"  ✓ Saved Stage 4 best checkpoint to {save_dir}")
    
    logger.info(f"Stage 4 complete. Best Context Flip Accuracy: {best_flip_acc:.2%}")
    return str(save_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 4: Synthetic Thread Fusion")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stage3-checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    cfg = Stage4Config(learning_rate=args.lr, num_epochs=args.epochs, batch_size=args.batch_size)
    train_stage4(config=cfg, stage3_checkpoint=args.stage3_checkpoint)
