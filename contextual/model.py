"""Hierarchical Transformer over explicit root-to-target reply paths."""
import torch
from torch import nn


class ConversationClassifier(nn.Module):
    def __init__(self, input_size, cfg):
        super().__init__()
        m = cfg["model"]
        width = m["hidden_size"]
        self.max_turns = m["max_turns"]
        self.projection = nn.Linear(input_size, width)
        self.position = nn.Embedding(self.max_turns, width)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=m["heads"],
            dim_feedforward=m["feedforward_size"], dropout=m["dropout"],
            activation="gelu", batch_first=True)
        self.conversation = nn.TransformerEncoder(layer, num_layers=m["layers"], enable_nested_tensor=False)
        self.pool = nn.Linear(width, 1)
        # Target skips contextual pooling, so a toxic parent need not taint a benign reply.
        self.classifier = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(),
                                        nn.Dropout(m["dropout"]), nn.Linear(width, 1))

    def forward(self, individual, valid):
        if individual.ndim != 3 or valid.shape != individual.shape[:2]:
            raise ValueError("Expected embeddings [batch, turns, features] and mask [batch, turns]")
        if not valid.any(dim=1).all() or individual.shape[1] > self.max_turns:
            raise ValueError("Empty or overlong conversation window")
        # Collation uses contiguous right padding; reject holes rather than select wrong target.
        lengths = valid.sum(dim=1)
        positions = torch.arange(individual.shape[1], device=individual.device)
        if not torch.equal(valid, positions.unsqueeze(0) < lengths.unsqueeze(1)):
            raise ValueError("Conversation mask must use right padding")
        projected = self.projection(individual)
        contextual = self.conversation(projected + self.position(positions), src_key_padding_mask=~valid)
        weights = self.pool(contextual).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(dim=-1)
        whole = (contextual * weights.unsqueeze(-1)).sum(dim=1)
        rows = torch.arange(individual.shape[0], device=individual.device)
        target = projected[rows, lengths-1]
        contextual_target = contextual[rows, lengths-1]
        logits = self.classifier(torch.cat((target, contextual_target, whole), dim=-1)).squeeze(-1)
        return {"logits": logits, "individual_embeddings": individual,
                "contextual_embeddings": contextual, "conversation_embedding": whole,
                "turn_attention": weights}


def collate_examples(examples, vectors, lookup, device):
    indices = [[lookup[t] for t in ex["texts"]] for ex in examples]
    width = max(map(len, indices))
    x = torch.zeros(len(examples), width, vectors.shape[-1])
    valid = torch.zeros(len(examples), width, dtype=torch.bool)
    for i, ids in enumerate(indices):
        x[i, :len(ids)] = vectors[ids]
        valid[i, :len(ids)] = True
    labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.float32, device=device)
    return x.to(device), valid.to(device), labels
