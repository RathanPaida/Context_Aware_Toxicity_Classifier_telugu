"""
Phase 3: Model Architecture — IndicBERT-v3 Cross-Encoder with Target-Span Pooling.

Implements:
  - Multi-label classification head with [CLS] + Target-Span pooling
  - Class-Weighted Focal Loss for severe class imbalance
  - Custom tokenization with target span tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    PRIMARY_MODEL,
    FALLBACK_MODEL,
    TOXICITY_LABELS,
    LABEL2ID,
    ID2LABEL,
    MAX_SEQ_LENGTH,
    TARGET_PREFIX,
    SEP_TOKEN,
    get_logger,
)

logger = get_logger("model")


# ═══════════════════════════════════════════════════════════════════════
# Focal Loss Implementation
# ═══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Class-Weighted Focal Loss for severe class imbalance.
    
    FL(pt) = -α_t · (1 - pt)^γ · log(pt)
    
    When a comment is clearly non-toxic (pt → 1), the modulating factor 
    (1 - pt)^γ approaches zero, zeroing out its gradient. The optimizer 
    concentrates 90%+ of parameter updates on ambiguous, context-dependent 
    edge cases.
    
    Args:
        alpha: Class weight for positive class (default: 0.75)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss for multi-label classification.
        
        Args:
            logits: Raw model outputs, shape (batch, num_labels)
            targets: Binary targets, shape (batch, num_labels)
        
        Returns:
            Scalar focal loss
        """
        probs = torch.sigmoid(logits)
        
        # Binary cross-entropy component
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        
        # pt = p if y=1 else (1-p)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Alpha weighting: α for positive, (1-α) for negative
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Focal modulation: (1 - pt)^γ
        focal_weight = (1 - pt).pow(self.gamma)
        
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ═══════════════════════════════════════════════════════════════════════
# Target-Span Pooling Head
# ═══════════════════════════════════════════════════════════════════════

class TargetSpanPoolingHead(nn.Module):
    """
    Two-layer MLP classification head with [CLS] + Target-Span pooling.
    
    When contexts are long (3-5 parent messages), standard [CLS] pooling 
    can accidentally flag a benign reply simply because the parent comment 
    was toxic. Concatenating the global [CLS] vector with the mean-pooled 
    vector of only the Target tokens prevents parent-toxicity bleed.
    
    Architecture:
        [CLS_vec ; TargetSpan_mean] → Linear(2*hidden, hidden) → GELU → 
        Dropout(0.2) → Linear(hidden, num_labels)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size * 2, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.dense2 = nn.Linear(hidden_size, num_labels)
    
    def forward(
        self,
        cls_output: torch.Tensor,
        target_span_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cls_output: [CLS] token representation, shape (batch, hidden)
            target_span_output: Mean-pooled target span, shape (batch, hidden)
        
        Returns:
            Logits, shape (batch, num_labels)
        """
        # Concatenate [CLS] and target span representations
        combined = torch.cat([cls_output, target_span_output], dim=-1)
        
        x = self.dense1(combined)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.dense2(x)
        
        return logits


# ═══════════════════════════════════════════════════════════════════════
# Main Model: Context-Aware Toxicity Classifier
# ═══════════════════════════════════════════════════════════════════════

class ContextAwareToxicityClassifier(nn.Module):
    """
    Full Cross-Encoder architecture for context-aware toxicity classification.
    
    Uses IndicBERT-v3 (or MuRIL) as the backbone encoder. By passing the 
    concatenated [CLS] Context [SEP] Target [SEP] sequence into the model 
    simultaneously, every token in the Telugu target utterance attends 
    directly to every token in the parent comment across all transformer layers.
    
    Components:
        1. Encoder: IndicBERT-v3 / MuRIL backbone
        2. Head: TargetSpanPoolingHead (CLS + Target mean-pool → MLP)
        3. Loss: Class-Weighted Focal Loss
    """
    
    def __init__(
        self,
        model_name: str = PRIMARY_MODEL,
        num_labels: int = 5,
        dropout_p: float = 0.2,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        use_target_span_pooling: bool = True,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_target_span_pooling = use_target_span_pooling
        
        # Load encoder backbone
        logger.info(f"Loading encoder backbone: {model_name}")
        try:
            self.encoder = AutoModel.from_pretrained(model_name)
            logger.info(f"  ✓ Loaded {model_name}")
        except Exception as e:
            logger.warning(f"  Failed to load {model_name}: {e}")
            logger.info(f"  Falling back to {FALLBACK_MODEL}")
            self.encoder = AutoModel.from_pretrained(FALLBACK_MODEL)
            self.model_name = FALLBACK_MODEL
        
        hidden_size = self.encoder.config.hidden_size
        
        # Classification head
        if use_target_span_pooling:
            self.classifier = TargetSpanPoolingHead(
                hidden_size=hidden_size,
                num_labels=num_labels,
                dropout_p=dropout_p,
            )
        else:
            # Simple [CLS] pooling fallback
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden_size, num_labels),
            )
        
        # Loss function
        self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        logger.info(f"  Hidden size: {hidden_size}")
        logger.info(f"  Num labels: {num_labels}")
        logger.info(f"  Target-Span Pooling: {use_target_span_pooling}")
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional target-span pooling.
        
        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            attention_mask: Attention mask, shape (batch, seq_len)
            token_type_ids: Segment IDs, shape (batch, seq_len) [optional]
            target_mask: Binary mask for Target span tokens, shape (batch, seq_len)
            labels: Multi-label binary targets, shape (batch, num_labels)
        
        Returns:
            Dict with 'logits', 'loss' (if labels provided), 'probs'
        """
        # Encode
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        
        outputs = self.encoder(**encoder_kwargs)
        
        # [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        
        if self.use_target_span_pooling and target_mask is not None:
            # Mean-pool only the Target span tokens
            # target_mask: (batch, seq_len) → expand to (batch, seq_len, hidden)
            mask_expanded = target_mask.unsqueeze(-1).expand_as(
                outputs.last_hidden_state
            ).float()
            
            sum_embeddings = (outputs.last_hidden_state * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-8)
            target_span_output = sum_embeddings / sum_mask  # (batch, hidden)
            
            logits = self.classifier(cls_output, target_span_output)
        else:
            if self.use_target_span_pooling:
                # Fallback: use CLS for both (no target mask available)
                logits = self.classifier(cls_output, cls_output)
            else:
                logits = self.classifier(cls_output)
        
        result = {
            "logits": logits,
            "probs": torch.sigmoid(logits),
        }
        
        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels)
        
        return result
    
    def freeze_layers(self, freeze_ratio: float = 0.5):
        """
        Freeze bottom N% of encoder layers.
        
        Used in Stage 3+ to preserve Stage 1 context mechanics 
        while adapting top layers for Telugu-specific features.
        
        Args:
            freeze_ratio: Fraction of layers to freeze (0.5 = bottom 50%)
        """
        # Get encoder layers
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, "layer"):
            layers = self.encoder.layer
        else:
            logger.warning("Could not find encoder layers for freezing")
            return
        
        n_layers = len(layers)
        n_freeze = int(n_layers * freeze_ratio)
        
        # Freeze embedding layer
        if hasattr(self.encoder, "embeddings"):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
        
        # Freeze bottom layers
        for i, layer in enumerate(layers):
            if i < n_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        logger.info(
            f"Froze {n_freeze}/{n_layers} encoder layers ({freeze_ratio:.0%}). "
            f"Trainable: {trainable:,} / {total:,} params ({trainable/total:.1%})"
        )


# ═══════════════════════════════════════════════════════════════════════
# Tokenization Utilities with Target Span Tracking
# ═══════════════════════════════════════════════════════════════════════

def tokenize_with_target_mask(
    text: str,
    tokenizer,
    max_length: int = MAX_SEQ_LENGTH,
    target_prefix: str = TARGET_PREFIX,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a pre-formatted context string and create a target span mask.
    
    The target span is identified by finding the last occurrence of 
    TARGET_PREFIX in the text and masking all tokens after it (until the 
    final [SEP]).
    
    Returns:
        Dict with 'input_ids', 'attention_mask', 'target_mask'
    """
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = encoding["input_ids"].squeeze(0)
    attention_mask = encoding["attention_mask"].squeeze(0)
    
    # Build target mask
    target_mask = torch.zeros_like(input_ids)
    
    # Find the target prefix tokens
    target_prefix_ids = tokenizer.encode(target_prefix, add_special_tokens=False)
    
    # Scan for the target prefix in the input
    seq_len = input_ids.size(0)
    prefix_len = len(target_prefix_ids)
    
    target_start = -1
    for i in range(seq_len - prefix_len, -1, -1):  # Search from end
        if input_ids[i:i + prefix_len].tolist() == target_prefix_ids:
            target_start = i + prefix_len
            break
    
    if target_start > 0:
        # Find the end (last SEP or end of non-padding)
        sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id else tokenizer.eos_token_id
        target_end = seq_len
        
        for i in range(target_start, seq_len):
            if input_ids[i] == sep_id or input_ids[i] == tokenizer.pad_token_id:
                target_end = i
                break
        
        target_mask[target_start:target_end] = 1
    else:
        # Fallback: use all non-special, non-padding tokens
        special_ids = set(tokenizer.all_special_ids)
        pad_id = tokenizer.pad_token_id
        for i in range(seq_len):
            if input_ids[i].item() not in special_ids and input_ids[i].item() != pad_id:
                target_mask[i] = 1
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_mask": target_mask,
    }


def load_tokenizer(model_name: str = PRIMARY_MODEL):
    """Load and configure the tokenizer for the model."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        logger.warning(f"Failed to load tokenizer for {model_name}, using {FALLBACK_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    
    return tokenizer
