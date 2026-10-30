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
from typing import Dict, List, Optional, Tuple, Any
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
# Target-Span Pooling Head (Legacy Fallback)
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
        combined = torch.cat([cls_output, target_span_output], dim=-1)
        x = self.dense1(combined)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.dense2(x)
        return logits


# ═══════════════════════════════════════════════════════════════════════
# Multi-Head Contextual Self-Attention Layer
# ═══════════════════════════════════════════════════════════════════════

class MultiHeadContextSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention over the entire contextual sequence:
    [CLS] Topic: ... [SEP] Parent: ... [SEP] Target: ... [SEP]
    
    Enables every token across the whole text (both context history and 
    target utterance) to attend to all other tokens across 8 diverse 
    representation heads (sentiment, intent, sarcasm, de-escalation, etc.).
    """
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout_p: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len), 1 for real tokens, 0 for padding
        Returns:
            contextualized_states: (batch, seq_len, hidden_size)
            attn_weights: (batch, seq_len, seq_len)
        """
        # PyTorch MultiheadAttention key_padding_mask: True for tokens to ignore (padding)
        key_padding_mask = (attention_mask == 0)
        
        attn_out, attn_weights = self.mha(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        out = self.norm(hidden_states + self.dropout(attn_out))
        return out, attn_weights


# ═══════════════════════════════════════════════════════════════════════
# Multi-Head Target-to-Context Cross-Attention Layer
# ═══════════════════════════════════════════════════════════════════════

class MultiHeadContextCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention where Target Utterance tokens act as Queries (Q)
    attending to Parent Context tokens as Keys (K) and Values (V).
    
    Directly extracts contextual influence: "Given what was said in the parent 
    conversation, what is the contextual intent of each target word?"
    Crucial for distinguishing playful gaming banter from violent threats.
    """
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout_p: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            target_mask: (batch, seq_len), 1 for target tokens
            attention_mask: (batch, seq_len), 1 for valid tokens
        """
        # Context tokens: valid tokens that are NOT part of target
        context_mask = (attention_mask == 1) & (target_mask == 0)
        has_context = (context_mask.sum(dim=-1) > 0)
        
        if not has_context.any():
            # No parent context (first turn) -> pass through self-attended states
            return hidden_states, None
        
        # Mask out anything that is NOT parent context
        context_key_padding = (context_mask == 0)
        
        attn_out, cross_weights = self.mha(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=context_key_padding,
            need_weights=True,
            average_attn_weights=True,
        )
        out = self.norm(hidden_states + self.dropout(attn_out))
        return out, cross_weights


# ═══════════════════════════════════════════════════════════════════════
# Attentive Whole-Text & Span Pooling Head
# ═══════════════════════════════════════════════════════════════════════

class AttentivePooling(nn.Module):
    """
    Learnable Attention-based sequence pooling.
    Computes token importance scores a_i = softmax(w^T tanh(W h_i)) across 
    the whole text, allowing the model to dynamically weight the most salient 
    semantic words rather than taking a naive unweighted average.
    """
    def __init__(self, hidden_size: int, bottleneck_size: int = 128):
        super().__init__()
        self.proj = nn.Linear(hidden_size, bottleneck_size)
        self.scoring = nn.Linear(bottleneck_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            mask: (batch, seq_len), 1 for tokens to pool over, 0 to mask
        Returns:
            pooled: (batch, hidden_size)
            attn_weights: (batch, seq_len) normalized attention weights
        """
        scores = self.scoring(torch.tanh(self.proj(hidden_states))).squeeze(-1)  # (batch, seq_len)
        fill_val = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
        scores = scores.masked_fill(mask == 0, fill_val)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Handle case where all tokens in mask are 0
        valid_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        fallback_weights = (mask.float() / valid_sum)
        attn_weights = torch.where(mask.sum(dim=-1, keepdim=True) > 0, attn_weights, fallback_weights)
        
        pooled = torch.sum(attn_weights.unsqueeze(-1) * hidden_states, dim=1)  # (batch, hidden_size)
        return pooled, attn_weights


# ═══════════════════════════════════════════════════════════════════════
# Multi-Perspective Attention Classification Head
# ═══════════════════════════════════════════════════════════════════════

class MultiHeadAttentionClassificationHead(nn.Module):
    """
    Deep 3-layer MLP classifier fusing 4 complementary attention representations:
      1. Global sequence token [CLS]
      2. Whole-text multi-head attentive pool (full dialogue context)
      3. Cross-attended target pool (target interpreted through parent)
      4. Attentive target-span pool (target utterance focus)
    """
    def __init__(self, hidden_size: int, num_labels: int, dropout_p: float = 0.2):
        super().__init__()
        in_dim = hidden_size * 4
        self.fc1 = nn.Linear(in_dim, hidden_size * 2)
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_p)
        
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_p)
        
        self.fc3 = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        cls_vec: torch.Tensor,
        whole_text_pool: torch.Tensor,
        cross_target_pool: torch.Tensor,
        target_attn_pool: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([cls_vec, whole_text_pool, cross_target_pool, target_attn_pool], dim=-1)
        x = self.dropout1(self.act1(self.norm1(self.fc1(fused))))
        x = self.dropout2(self.act2(self.norm2(self.fc2(x))))
        logits = self.fc3(x)
        return logits


# ═══════════════════════════════════════════════════════════════════════
# Main Model: Context-Aware Toxicity Classifier with Multi-Head Attention
# ═══════════════════════════════════════════════════════════════════════

class ContextAwareToxicityClassifier(nn.Module):
    """
    Full Context-Aware Toxicity Classifier with Multi-Head Attention Transformers.
    
    Captures the context across the whole text using:
      1. IndicBERT-v3 / MuRIL Transformer backbone encoder
      2. 8-Head Multi-Head Contextual Self-Attention across whole dialogue
      3. 8-Head Multi-Head Cross-Attention between Target and Parent Context
      4. Attentive Whole-Text Pooling (token-level saliency extraction)
      5. Multi-Perspective Classification Head
      6. Class-Weighted Focal Loss for severe class imbalance
    """
    
    def __init__(
        self,
        model_name: str = PRIMARY_MODEL,
        num_labels: int = 5,
        dropout_p: float = 0.2,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        use_target_span_pooling: bool = True,
        use_multihead_attention: bool = True,
        num_heads: int = 8,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_target_span_pooling = use_target_span_pooling
        self.use_multihead_attention = use_multihead_attention
        self.num_heads = num_heads
        
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
        self.hidden_size = hidden_size
        
        # 1. Multi-Head Contextual Self-Attention Block (8 Heads)
        self.mha_self = MultiHeadContextSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_p=dropout_p,
        )
        
        # 2. Multi-Head Target-to-Context Cross-Attention Block (8 Heads)
        self.mha_cross = MultiHeadContextCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_p=dropout_p,
        )
        
        # 3. Attentive Whole-Text and Target Poolers
        self.whole_text_pooler = AttentivePooling(hidden_size=hidden_size)
        self.target_pooler = AttentivePooling(hidden_size=hidden_size)
        
        # 4. Multi-Head Attention Classification Head
        self.mha_head = MultiHeadAttentionClassificationHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout_p=dropout_p,
        )
        
        # Legacy Fallback Head (for backward compatibility if needed)
        self.classifier = TargetSpanPoolingHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout_p=dropout_p,
        )
        
        # Loss function: Class-Weighted Focal Loss
        self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        logger.info(f"  Hidden size: {hidden_size}")
        logger.info(f"  Num labels: {num_labels}")
        logger.info(f"  Multi-Head Attention: {use_multihead_attention} ({num_heads} heads)")
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass through Transformer Backbone and Multi-Head Attention layers.
        
        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            attention_mask: Attention mask, shape (batch, seq_len)
            token_type_ids: Segment IDs, shape (batch, seq_len) [optional]
            target_mask: Binary mask for Target span tokens, shape (batch, seq_len)
            labels: Multi-label binary targets, shape (batch, num_labels)
            return_attention: If True, returns token attention weights and cross-attention
        
        Returns:
            Dict with 'logits', 'probs', 'loss' (if labels provided), and attention telemetry.
        """
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        
        outputs = self.encoder(**encoder_kwargs)
        seq_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        cls_output = seq_hidden[:, 0, :]        # (batch, hidden)
        
        # Ensure target_mask exists
        if target_mask is None:
            target_mask = attention_mask.clone()
        
        if self.use_multihead_attention:
            # 1. Whole-Text Multi-Head Contextual Self-Attention (8 Heads)
            h_self, self_attn_matrix = self.mha_self(seq_hidden, attention_mask)
            
            # 2. Attentive Whole-Text Pooling (Token-Level Saliency)
            whole_text_pool, token_attention_scores = self.whole_text_pooler(h_self, attention_mask)
            
            # 3. Target-to-Context Multi-Head Cross-Attention (8 Heads)
            h_cross, cross_attn_matrix = self.mha_cross(h_self, target_mask, attention_mask)
            
            # 4. Attentive Target-Span Pooling
            target_pool, target_attn_scores = self.target_pooler(h_self, target_mask)
            
            # 5. Cross-Attended Target Pooling
            cross_target_pool, _ = self.target_pooler(h_cross, target_mask)
            
            # 6. Deep MLP Multi-Head Attention Head
            logits = self.mha_head(cls_output, whole_text_pool, cross_target_pool, target_pool)
            
            result = {
                "logits": logits,
                "probs": torch.sigmoid(logits),
            }
            
            if return_attention or not self.training:
                result["token_attention"] = token_attention_scores
                result["target_attention"] = target_attn_scores
                result["self_attention"] = self_attn_matrix
                result["cross_attention"] = cross_attn_matrix
        else:
            # Legacy fallback: simple CLS + target mean-pool
            mask_expanded = target_mask.unsqueeze(-1).expand_as(seq_hidden).float()
            sum_embeddings = (seq_hidden * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-8)
            target_span_output = sum_embeddings / sum_mask
            logits = self.classifier(cls_output, target_span_output)
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
        Used to preserve base multilingual representations while adapting 
        top layers and Multi-Head Attention mechanisms for Telugu/Indic context.
        """
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
