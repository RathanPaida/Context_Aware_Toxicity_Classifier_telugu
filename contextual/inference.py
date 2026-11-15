"""Predict from text and actual ancestors; there is deliberately no topic argument."""
import json
from pathlib import Path

import torch

from .data import ancestor_chain
from .embeddings import TextEncoder, device_for
from .model import ConversationClassifier


class Predictor:
    def __init__(self, checkpoint):
        checkpoint = Path(checkpoint)
        metadata_path = checkpoint / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No trained checkpoint at {checkpoint}. Run python run_pipeline.py --all first.")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.cfg = self.metadata["config"]
        self.threshold = self.metadata["threshold"]
        if not 0 <= self.threshold <= 1:
            raise ValueError("Invalid checkpoint threshold")
        self.encoder = TextEncoder(self.cfg, saved_path=checkpoint / "encoder")
        self.model = ConversationClassifier(self.metadata["input_size"], self.cfg).to(device_for(self.cfg))
        self.model.load_state_dict(torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=True), strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text, history=None, include_embeddings=False):
        """History must be the actual ancestor chain, oldest first."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        history = [] if history is None else history
        if not isinstance(history, list) or any(not isinstance(t, str) or not t.strip() for t in history):
            raise ValueError("history must be a list of nonempty ancestor messages")
        original = history + [text]
        texts = original[-self.cfg["model"]["max_turns"]:]
        vectors = self.encoder.encode(texts).to(device_for(self.cfg)).unsqueeze(0)
        valid = torch.ones(vectors.shape[:2], dtype=torch.bool, device=vectors.device)
        output = self.model(vectors, valid)
        isolated = self.model(vectors[:, -1:], valid[:, -1:])
        score = float(output["logits"].sigmoid().item())
        alone = float(isolated["logits"].sigmoid().item())
        result = {"is_toxic": score >= self.threshold, "toxicity_score": score,
                  "threshold": self.threshold, "isolated_score": alone,
                  "context_score_delta": score-alone, "turn_attention": output["turn_attention"][0].cpu().tolist(),
                  "turns_used": len(texts), "ancestors_omitted": len(original)-len(texts),
                  "has_context_training": self.metadata["has_context_training"]}
        if include_embeddings:
            result.update(individual_embeddings=vectors[0].cpu().tolist(),
                          contextual_embeddings=output["contextual_embeddings"][0].cpu().tolist(),
                          conversation_embedding=output["conversation_embedding"][0].cpu().tolist())
        return result

    def predict_thread(self, messages, target_id, include_embeddings=False):
        nodes = {str(m["id"]): {**m, "id": str(m["id"]),
                 "parent_id": str(m["parent_id"]) if m.get("parent_id") is not None else None} for m in messages}
        if len(nodes) != len(messages):
            raise ValueError("Duplicate message IDs")
        chain = ancestor_chain(nodes, str(target_id), len(nodes))
        result = self.predict(chain[-1]["text"], [n["text"] for n in chain[:-1]], include_embeddings)
        result["message_ids"] = [n["id"] for n in chain[-result["turns_used"]:]]
        return result
