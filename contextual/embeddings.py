"""Frozen pretrained token encoder; all chunks contribute to each text vector."""
import hashlib
import json

import torch
from transformers import AutoModel, AutoTokenizer

from .config import location


def device_for(cfg):
    requested = cfg["training"]["device"]
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)


class TextEncoder:
    def __init__(self, cfg, saved_path=None):
        self.cfg = cfg
        self.device = device_for(cfg)
        ec = cfg["encoder"]
        name = str(saved_path or ec["name"])
        kwargs = {"local_files_only": True} if saved_path else {"revision": ec["revision"], "local_files_only": ec["local_files_only"]}
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        if not self.tokenizer.is_fast:
            raise ValueError("This encoder requires a fast tokenizer for lossless overflow chunking")
        self.backend = self.tokenizer.backend_tokenizer
        self.backend.no_truncation()
        self.backend.no_padding()
        self.model = AutoModel.from_pretrained(name, **kwargs).to(self.device).eval()
        self.model.requires_grad_(False)
        self.dimension = self.model.config.hidden_size
        self.revision = getattr(self.model.config, "_commit_hash", None) or ec["revision"]
        self.max_tokens = min(ec["max_tokens"], self.model.config.max_position_embeddings, self.tokenizer.model_max_length)
        self.content_tokens = self.max_tokens - self.tokenizer.num_special_tokens_to_add(pair=False)
        if self.content_tokens <= 0:
            raise ValueError("encoder.max_tokens must leave room for text after special tokens")

    @torch.inference_mode()
    def encode(self, texts):
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Provide nonempty strings to encode")
        # Never truncate a paragraph: weighted chunk pooling includes every token.
        sums = torch.zeros(len(texts), self.dimension, device=self.device)
        counts = torch.zeros(len(texts), device=self.device)
        batch, owners = [], []
        completed_batches = 0

        def flush():
            nonlocal completed_batches
            encoded = self.tokenizer.pad(batch, padding=True, return_tensors="pt")
            special = encoded.pop("special_tokens_mask").to(self.device)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            vectors = self.model(**encoded).last_hidden_state.float()
            mask = encoded["attention_mask"].bool() & ~special.bool()
            indices = torch.tensor(owners, device=self.device)
            sums.index_add_(0, indices, (vectors * mask.unsqueeze(-1)).sum(dim=1))
            counts.index_add_(0, indices, mask.sum(dim=1).float())
            completed_batches += 1
            if completed_batches % self.cfg["encoder"].get("progress_batches", 100) == 0:
                print(f"Embedding progress: {owners[-1] + 1:,}/{len(texts):,} texts", flush=True)
            batch.clear()
            owners.clear()

        for owner, text in enumerate(texts):
            # Use the public tokenizers backend so wrapper versions cannot silently
            # pre-truncate before exposing overflow. Check token conservation.
            encoded_text = self.backend.encode(text, add_special_tokens=False)
            original_count = len(encoded_text.ids)
            encoded_text.truncate(self.content_tokens)
            chunks = [encoded_text, *encoded_text.overflowing]
            if sum(len(chunk.ids) for chunk in chunks) != original_count:
                raise RuntimeError("Tokenizer chunking dropped content tokens")
            for chunk in chunks:
                encoded_chunk = self.backend.post_process(chunk, add_special_tokens=True)
                features = {"input_ids": encoded_chunk.ids, "attention_mask": encoded_chunk.attention_mask,
                            "special_tokens_mask": encoded_chunk.special_tokens_mask}
                if "token_type_ids" in self.tokenizer.model_input_names:
                    features["token_type_ids"] = encoded_chunk.type_ids
                batch.append(features)
                owners.append(owner)
                if len(batch) == self.cfg["encoder"]["batch_size"]:
                    flush()
        if batch:
            flush()
        if (counts == 0).any():
            raise ValueError("Text produced no content tokens")
        return (sums / counts.unsqueeze(-1)).cpu()

    def save(self, directory):
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)


def cached_embeddings(cfg, examples, encoder):
    texts = list(dict.fromkeys(text for ex in examples for text in ex["texts"]))
    signature = hashlib.sha256(json.dumps({"encoder": cfg["encoder"], "resolved_revision": encoder.revision,
        "texts": texts, "pooling": "all_content_tokens_chunk_mean_v2"}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    path = location(cfg, "embeddings")
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if cached["signature"] == signature:
            print("Reusing verified embedding cache", flush=True)
            return cached["vectors"], {text: i for i, text in enumerate(texts)}
    vectors = encoder.encode(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"signature": signature, "vectors": vectors}, temporary)
    temporary.replace(path)
    return vectors, {text: i for i, text in enumerate(texts)}
