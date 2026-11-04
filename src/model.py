"""
Phase 3: Model Architecture — IndicBERT Cross-Encoder with Target-Span Pooling.

Implements:
  - Multi-label classification head fusing [CLS], whole-text attentive pooling,
    target-to-context cross-attention and target-span pooling
  - Class-weighted Focal Loss with support for partially-labelled data
  - Token-level input assembly that never truncates the target utterance
  - Strict checkpoint save / load helpers
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.config import (
    CHECKPOINT_DIR,
    CHECKPOINT_PRIORITY,
    CONTEXT_PREFIX,
    FALLBACK_MODEL,
    MAX_SEQ_LENGTH,
    PARENT_PREFIX,
    PRIMARY_MODEL,
    TARGET_MAX_TOKENS,
    TOPIC_PREFIX,
    TOXICITY_LABELS,
    get_logger,
)

logger = get_logger("model")


# ═══════════════════════════════════════════════════════════════════════
# Focal Loss Implementation
# ═══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Class-weighted focal loss for multi-label classification.

        FL(pt) = -α_t · (1 - pt)^γ · log(pt)

    Targets equal to -1 (label not annotated by the source dataset) are
    masked out, so e.g. toxic-chat rows — which only annotate `toxic` —
    do not teach the model that they are "not a threat".
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        mask = (targets >= 0).float()
        safe_targets = targets.clamp(min=0)

        bce = F.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(safe_targets == 1, probs, 1 - probs)
        alpha_t = torch.where(
            safe_targets == 1,
            torch.full_like(probs, self.alpha),
            torch.full_like(probs, 1 - self.alpha),
        )
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce * mask

        if self.reduction == "mean":
            return loss.sum() / mask.sum().clamp(min=1.0)
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ═══════════════════════════════════════════════════════════════════════
# Target-Span Pooling Head (Legacy Fallback)
# ═══════════════════════════════════════════════════════════════════════

class TargetSpanPoolingHead(nn.Module):
    """[CLS ; mean(target span)] → Linear → GELU → Dropout → Linear."""

    def __init__(self, hidden_size: int, num_labels: int, dropout_p: float = 0.2):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size * 2, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.dense2 = nn.Linear(hidden_size, num_labels)

    def forward(self, cls_output: torch.Tensor, target_span_output: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cls_output, target_span_output], dim=-1)
        return self.dense2(self.dropout(self.activation(self.dense1(x))))


# ═══════════════════════════════════════════════════════════════════════
# Multi-Head Contextual Self-Attention Layer
# ═══════════════════════════════════════════════════════════════════════

class MultiHeadContextSelfAttention(nn.Module):
    """Extra multi-head self-attention block over the whole (context + target) sequence."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout_p: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_p, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        key_padding_mask = attention_mask == 0
        attn_out, attn_weights = self.mha(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        return self.norm(hidden_states + self.dropout(attn_out)), attn_weights


# ═══════════════════════════════════════════════════════════════════════
# Multi-Head Target-to-Context Cross-Attention Layer
# ═══════════════════════════════════════════════════════════════════════

class MultiHeadContextCrossAttention(nn.Module):
    """
    Cross-attention in which every token queries only the *context* tokens
    (keys/values = valid, non-target tokens). Rows without any context get a
    zero delta, so single-turn inputs pass through unchanged (plus LayerNorm).
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout_p: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_p, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        context_mask = (attention_mask == 1) & (target_mask == 0)
        # The [CLS]/[SEP] tokens are always "context"; require real context beyond those.
        has_context = context_mask.sum(dim=-1) > 2

        if not has_context.any():
            return self.norm(hidden_states), None

        # Rows without context would get an all-masked key set (→ NaN); give them the full mask
        # and zero their output below.
        safe_context_mask = torch.where(has_context.unsqueeze(-1), context_mask, attention_mask.bool())
        attn_out, cross_weights = self.mha(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~safe_context_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        row_gate = has_context.view(-1, 1, 1)
        attn_out = torch.where(row_gate, attn_out, torch.zeros_like(attn_out))
        if cross_weights is not None:
            cross_weights = torch.where(row_gate, cross_weights, torch.zeros_like(cross_weights))
        return self.norm(hidden_states + self.dropout(attn_out)), cross_weights


# ═══════════════════════════════════════════════════════════════════════
# Attentive Pooling
# ═══════════════════════════════════════════════════════════════════════

class AttentivePooling(nn.Module):
    """Learnable attention pooling: a_i = softmax(w^T tanh(W h_i)) over masked positions."""

    def __init__(self, hidden_size: int, bottleneck_size: int = 128):
        super().__init__()
        self.proj = nn.Linear(hidden_size, bottleneck_size)
        self.scoring = nn.Linear(bottleneck_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.scoring(torch.tanh(self.proj(hidden_states))).squeeze(-1)
        fill_val = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
        scores = scores.masked_fill(mask == 0, fill_val)
        attn_weights = F.softmax(scores, dim=-1)

        valid = mask.sum(dim=-1, keepdim=True)
        fallback = mask.float() / valid.clamp(min=1)
        attn_weights = torch.where(valid > 0, attn_weights, fallback.to(attn_weights.dtype))

        pooled = torch.sum(attn_weights.unsqueeze(-1) * hidden_states, dim=1)
        return pooled, attn_weights


# ═══════════════════════════════════════════════════════════════════════
# Multi-Perspective Attention Classification Head
# ═══════════════════════════════════════════════════════════════════════

class MultiHeadAttentionClassificationHead(nn.Module):
    """3-layer MLP over [CLS ; whole-text pool ; cross-attended target pool ; target pool]."""

    def __init__(self, hidden_size: int, num_labels: int, dropout_p: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_p)
        self.fc3 = nn.Linear(hidden_size, num_labels)

    def forward(self, cls_vec, whole_text_pool, cross_target_pool, target_attn_pool) -> torch.Tensor:
        fused = torch.cat([cls_vec, whole_text_pool, cross_target_pool, target_attn_pool], dim=-1)
        x = self.dropout1(self.act1(self.norm1(self.fc1(fused))))
        x = self.dropout2(self.act2(self.norm2(self.fc2(x))))
        return self.fc3(x)


# ═══════════════════════════════════════════════════════════════════════
# Main Model
# ═══════════════════════════════════════════════════════════════════════

class ContextAwareToxicityClassifier(nn.Module):
    """
    Context-aware toxicity classifier:
      1. IndicBERT / MuRIL transformer encoder
      2. Extra multi-head self-attention over the whole dialogue
      3. Target-to-context multi-head cross-attention
      4. Attentive whole-text and target-span pooling
      5. Multi-perspective MLP head (multi-label, sigmoid)
    """

    def __init__(
        self,
        model_name: str = PRIMARY_MODEL,
        num_labels: int = len(TOXICITY_LABELS),
        dropout_p: float = 0.2,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        use_multihead_attention: bool = True,
        num_heads: int = 8,
        load_pretrained_encoder: bool = True,
        allow_fallback: bool = False,
        **_legacy_kwargs,  # e.g. use_target_span_pooling from older call sites
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_multihead_attention = use_multihead_attention
        self.num_heads = num_heads

        self.encoder = self._load_encoder(model_name, load_pretrained_encoder, allow_fallback)
        hidden_size = self.encoder.config.hidden_size
        self.hidden_size = hidden_size

        self.mha_self = MultiHeadContextSelfAttention(hidden_size, num_heads, dropout_p)
        self.mha_cross = MultiHeadContextCrossAttention(hidden_size, num_heads, dropout_p)
        self.whole_text_pooler = AttentivePooling(hidden_size)
        self.target_pooler = AttentivePooling(hidden_size)
        self.mha_head = MultiHeadAttentionClassificationHead(hidden_size, num_labels, dropout_p)
        # Legacy head, only used when use_multihead_attention=False
        self.classifier = TargetSpanPoolingHead(hidden_size, num_labels, dropout_p)

        self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        logger.info(f"  Encoder: {self.model_name} | hidden={hidden_size} | labels={num_labels} | heads={num_heads}")

    def _load_encoder(self, model_name: str, pretrained: bool, allow_fallback: bool):
        try:
            if pretrained:
                return AutoModel.from_pretrained(model_name)
            # Architecture only — weights will come from a checkpoint.
            return AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        except Exception as e:
            if not allow_fallback or model_name == FALLBACK_MODEL:
                raise RuntimeError(f"Could not load encoder '{model_name}': {e}") from e
            logger.error(f"Failed to load {model_name} ({e}); falling back to {FALLBACK_MODEL}")
            self.model_name = FALLBACK_MODEL
            return AutoModel.from_pretrained(FALLBACK_MODEL)

    def encoder_parameters(self):
        return self.encoder.parameters()

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        **_unused,
    ) -> Dict[str, Any]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_hidden = outputs.last_hidden_state
        cls_output = seq_hidden[:, 0, :]

        if target_mask is None:
            target_mask = attention_mask.clone()

        result: Dict[str, Any] = {}
        if self.use_multihead_attention:
            h_self, self_attn = self.mha_self(seq_hidden, attention_mask)
            whole_text_pool, token_attention = self.whole_text_pooler(h_self, attention_mask)
            h_cross, cross_attn = self.mha_cross(h_self, target_mask, attention_mask)
            target_pool, target_attention = self.target_pooler(h_self, target_mask)
            cross_target_pool, _ = self.target_pooler(h_cross, target_mask)
            logits = self.mha_head(cls_output, whole_text_pool, cross_target_pool, target_pool)
            if return_attention:
                result.update({
                    "token_attention": token_attention,
                    "target_attention": target_attention,
                    "self_attention": self_attn,
                    "cross_attention": cross_attn,
                })
        else:
            m = target_mask.unsqueeze(-1).float()
            target_span = (seq_hidden * m).sum(1) / m.sum(1).clamp(min=1e-8)
            logits = self.classifier(cls_output, target_span)

        result["logits"] = logits
        result["probs"] = torch.sigmoid(logits.float())
        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels)
        return result

    def configure_training(self, freeze_ratio: float = 0.0, freeze_embeddings: bool = True,
                           gradient_checkpointing: bool = False):
        """
        Memory/regularisation setup used by every stage.

        freeze_embeddings: IndicBERTv2's ~250k-token embedding table is ~190M of its ~278M
        parameters; training it with AdamW needs ~2 GB extra and does not fit a 6 GB GPU.
        """
        if freeze_embeddings and hasattr(self.encoder, "embeddings"):
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        self.freeze_layers(freeze_ratio)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Trainable params: {trainable:,}/{total:,} ({trainable / total:.1%}) | "
                    f"embeddings {'frozen' if freeze_embeddings else 'trained'} | grad-ckpt {gradient_checkpointing}")

    def freeze_layers(self, freeze_ratio: float = 0.5):
        """Freeze embeddings + the bottom `freeze_ratio` of encoder layers."""
        if freeze_ratio <= 0:
            return
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, "layer"):
            layers = self.encoder.layer
        else:
            logger.warning("Could not find encoder layers for freezing")
            return

        n_freeze = int(len(layers) * freeze_ratio)
        if hasattr(self.encoder, "embeddings"):
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
        for layer in list(layers)[:n_freeze]:
            for p in layer.parameters():
                p.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Froze {n_freeze}/{len(layers)} encoder layers. Trainable: {trainable:,}/{total:,} ({trainable/total:.1%})")


# ═══════════════════════════════════════════════════════════════════════
# Input assembly
# ═══════════════════════════════════════════════════════════════════════

def _special_ids(tokenizer) -> Tuple[int, int, int]:
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer has no CLS/BOS or SEP/EOS token")
    return cls_id, sep_id, pad_id


def encode_example(
    tokenizer,
    target: str,
    history: Optional[Sequence[str]] = None,
    topic: Optional[str] = None,
    max_length: int = MAX_SEQ_LENGTH,
    target_max_tokens: int = TARGET_MAX_TOKENS,
) -> Dict[str, List[int]]:
    """
    Build model inputs:

        [CLS] Topic: t [SEP] Context_1: … [SEP] Parent: … [SEP] <target> [SEP]
        └──────────────── context segment ────────────────┘    └─ target ─┘

    Guarantees:
      * the target is never dropped (only cut if it alone exceeds `target_max_tokens`);
      * when context must be truncated, the *oldest* turns go first, and a partially
        kept turn keeps its most recent tokens;
      * `target_mask` marks exactly the target tokens (no string searching).

    `history` is chronological (oldest → newest); the last element is the parent.
    Returns unpadded python lists; use `collate_encodings` to batch.
    """
    cls_id, sep_id, _ = _special_ids(tokenizer)
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)  # noqa: E731

    target_ids = enc(target or "")[:target_max_tokens]
    budget = max_length - len(target_ids) - 3  # [CLS] … [SEP] target [SEP]

    topic_ids: List[int] = []
    if topic and topic.strip() and budget > 8:
        topic_ids = enc(f"{TOPIC_PREFIX} {topic.strip()}")[: min(24, budget - 1)]
        budget -= len(topic_ids) + 1

    turns = [t for t in (history or []) if t and str(t).strip()]
    kept_turns: List[List[int]] = []
    for i, turn in enumerate(reversed(turns)):  # newest first
        position = len(turns) - i
        prefix = PARENT_PREFIX if i == 0 else f"{CONTEXT_PREFIX}{position}:"
        ids = enc(f"{prefix} {turn}")
        cost = len(ids) + (1 if kept_turns else 0)
        if cost <= budget:
            kept_turns.insert(0, ids)
            budget -= cost
        else:
            room = budget - (1 if kept_turns else 0)
            if room >= 8:
                kept_turns.insert(0, ids[-room:])  # keep the most recent tokens of this turn
            break

    context_ids: List[int] = []
    for part in ([topic_ids] if topic_ids else []) + kept_turns:
        if context_ids:
            context_ids.append(sep_id)
        context_ids.extend(part)

    if context_ids:
        input_ids = [cls_id] + context_ids + [sep_id] + target_ids + [sep_id]
        target_start = len(context_ids) + 2
    else:
        input_ids = [cls_id] + target_ids + [sep_id]
        target_start = 1

    target_mask = [0] * len(input_ids)
    for j in range(target_start, target_start + len(target_ids)):
        target_mask[j] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "target_mask": target_mask,
    }


def collate_encodings(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    """Dynamic padding collate fn. Non-sequence fields (labels, ids) are stacked/collected."""
    max_len = max(len(b["input_ids"]) for b in batch)
    out: Dict[str, Any] = {}
    for key, pad in (("input_ids", pad_id), ("attention_mask", 0), ("target_mask", 0)):
        out[key] = torch.tensor([b[key] + [pad] * (max_len - len(b[key])) for b in batch], dtype=torch.long)
    if "labels" in batch[0]:
        out["labels"] = torch.stack([torch.as_tensor(b["labels"], dtype=torch.float) for b in batch])
    for key in batch[0]:
        if key not in out:
            out[key] = [b[key] for b in batch]
    return out


def to_model_inputs(encoding: Dict[str, List[int]], device) -> Dict[str, torch.Tensor]:
    """Single (unbatched) encoding → batch of 1 on `device`."""
    return {k: torch.tensor([encoding[k]], dtype=torch.long, device=device)
            for k in ("input_ids", "attention_mask", "target_mask")}


def clean_token(tok: str) -> str:
    """Strip WordPiece/SentencePiece markers for display."""
    return tok.replace("##", "").replace("▁", "").replace("Ġ", "").strip()


def load_tokenizer(model_name_or_path: str = PRIMARY_MODEL):
    """Load the tokenizer. Raises instead of silently switching to another vocabulary."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    return tokenizer


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═══════════════════════════════════════════════════════════════════════

def save_model_checkpoint(model: ContextAwareToxicityClassifier, tokenizer, save_dir: Path, metadata: Dict[str, Any]):
    """Write model.pt + tokenizer/ + metadata.json (model name, thresholds, metrics…)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")
    tokenizer.save_pretrained(save_dir / "tokenizer")
    meta = {
        "model_name": model.model_name,
        "labels": TOXICITY_LABELS,
        "num_heads": model.num_heads,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **metadata,
    }
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=float)
    logger.info(f"  Checkpoint saved to {save_dir}")


def read_checkpoint_metadata(ckpt_dir: Path) -> Dict[str, Any]:
    p = Path(ckpt_dir) / "metadata.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_model_weights(model: nn.Module, ckpt_dir: Path, allow_missing_heads: bool = False) -> Dict[str, Any]:
    """
    Load model.pt into `model`, failing loudly on mismatches.

    The old code used `strict=False` everywhere, which silently leaves the model
    randomly initialised when keys do not match. Here missing *encoder* keys are
    always an error; missing head keys are only tolerated when explicitly allowed
    (e.g. initialising from a checkpoint with a different head).
    """
    ckpt_dir = Path(ckpt_dir)
    state = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys]
    unexpected = [k for k in result.unexpected_keys]
    missing_encoder = [k for k in missing if k.startswith("encoder.")]
    missing_heads = [k for k in missing if not k.startswith("encoder.")]

    if missing_encoder:
        raise RuntimeError(f"{ckpt_dir}: {len(missing_encoder)} encoder weights missing, e.g. {missing_encoder[:3]}")
    if missing_heads and not allow_missing_heads:
        raise RuntimeError(f"{ckpt_dir}: {len(missing_heads)} head weights missing, e.g. {missing_heads[:3]}")
    if unexpected:
        logger.warning(f"{ckpt_dir}: ignoring {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    logger.info(f"  Loaded weights from {ckpt_dir / 'model.pt'}")
    return read_checkpoint_metadata(ckpt_dir)


def find_best_checkpoint(checkpoint_dir: Path = CHECKPOINT_DIR) -> Optional[Path]:
    for name in CHECKPOINT_PRIORITY:
        p = Path(checkpoint_dir) / name
        if (p / "model.pt").exists():
            return p
    return None


def load_classifier_from_checkpoint(ckpt_dir: Path, device=None) -> Tuple[ContextAwareToxicityClassifier, Any, Dict[str, Any]]:
    """Rebuild model + tokenizer from a checkpoint directory (no pretrained download of the encoder weights)."""
    ckpt_dir = Path(ckpt_dir)
    meta = read_checkpoint_metadata(ckpt_dir)
    model_name = meta.get("model_name", PRIMARY_MODEL)
    model = ContextAwareToxicityClassifier(
        model_name=model_name,
        num_labels=len(meta.get("labels", TOXICITY_LABELS)),
        num_heads=meta.get("num_heads", 8),
        load_pretrained_encoder=False,
    )
    load_model_weights(model, ckpt_dir)
    tok_dir = ckpt_dir / "tokenizer"
    tokenizer = load_tokenizer(tok_dir if tok_dir.exists() else model_name)
    if device is not None:
        model.to(device)
    model.eval()
    return model, tokenizer, meta
