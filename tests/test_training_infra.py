"""
Tests for the training infrastructure (no network, no GPU):
checkpoint saving/loading, resume state, hardware profiles, release selection,
Romanized-Telugu augmentation and display segments.

    python -m pytest tests -q
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.model as model_mod
from src.evaluation import choose_release
from src.hardware import EFFECTIVE_BATCH, profile_for
from src.model import (
    CheckpointSaveError, atomic_torch_save, encode_example, ensure_disk_space, load_model_weights, to_fp16_state,
)
from src.preprocessing import romanize_telugu
from src.training_utils import last_dir_for, load_resume_state, save_resume_state


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.head = nn.Linear(4, 1)


# ── checkpoints ────────────────────────────────────────────────────────

def test_fp16_checkpoint_roundtrip(tmp_path):
    src = TinyModel()
    atomic_torch_save(to_fp16_state(src.state_dict()), tmp_path / "model.pt")
    saved = torch.load(tmp_path / "model.pt", weights_only=True)
    assert all(v.dtype == torch.float16 for v in saved.values())
    dst = TinyModel()
    load_model_weights(dst, tmp_path)
    for (k, a), b in zip(src.state_dict().items(), dst.state_dict().values()):
        assert b.dtype == torch.float32, k
        assert torch.allclose(a, b, atol=1e-3), k


def test_atomic_save_failure_leaves_old_file_intact(tmp_path, monkeypatch):
    target = tmp_path / "model.pt"
    atomic_torch_save({"x": torch.ones(1)}, target)

    def boom(*a, **k):
        raise OSError("No space left on device")

    monkeypatch.setattr(model_mod.torch, "save", boom)
    with pytest.raises(CheckpointSaveError, match="No space left"):
        atomic_torch_save({"x": torch.zeros(1)}, target)
    monkeypatch.undo()
    assert torch.load(target, weights_only=True)["x"].item() == 1.0
    assert not (tmp_path / "model.pt.tmp").exists()


def test_disk_space_check_raises_clear_error(tmp_path):
    with pytest.raises(CheckpointSaveError, match="Not enough disk space"):
        ensure_disk_space(tmp_path, needed_bytes=10 ** 18)


# ── resume ─────────────────────────────────────────────────────────────

def _train_objects(model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = get_linear_schedule_with_warmup(opt, 1, 100)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return opt, sched, scaler


def test_resume_state_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = TinyModel()
    opt, sched, scaler = _train_objects(m)
    for _ in range(3):
        loss = m.head(m.encoder(torch.randn(2, 4))).sum()
        loss.backward()
        opt.step(); sched.step(); opt.zero_grad()
    info = {"completed_epochs": 2, "global_step": 3, "bad_evals": 0, "best_score": 0.5,
            "best_metrics": {}, "history": [], "stopped": False}
    save_resume_state(m, opt, sched, scaler, tmp_path / "s_last", info)

    m2 = TinyModel()
    opt2, sched2, scaler2 = _train_objects(m2)
    got = load_resume_state(m2, opt2, sched2, scaler2, tmp_path / "s_last")
    assert got["completed_epochs"] == 2
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)
    assert sched2.last_epoch == sched.last_epoch
    assert opt2.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]


def test_resume_missing_returns_none(tmp_path):
    m = TinyModel()
    assert load_resume_state(m, *_train_objects(m), tmp_path / "nothing") is None


def test_last_dir_naming():
    assert last_dir_for(Path("ck/stage3_best")).name == "stage3_last"


# ── hardware profiles ─────────────────────────────────────────────────

@pytest.mark.parametrize("vram,tier,ckpt,frozen", [
    (4.0, "small", True, True), (6.4, "small", True, True),
    (8.0, "medium", False, True), (12.0, "medium", False, True),
    (24.0, "large", False, False), (None, "cpu", False, True),
])
def test_hardware_profiles(vram, tier, ckpt, frozen):
    p = profile_for(vram)
    assert p.tier == tier
    assert p.gradient_checkpointing == ckpt
    assert p.freeze_embeddings == frozen
    assert p.batch_size * p.gradient_accumulation_steps == EFFECTIVE_BATCH


# ── release selection ─────────────────────────────────────────────────

def test_release_prefers_stage4_when_not_worse():
    scores = {"stage4_best": {"macro_f1": 0.70}, "stage3_best": {"macro_f1": 0.705}}
    assert choose_release(["stage4_best", "stage3_best"], scores, 0.01) == "stage4_best"


def test_release_falls_back_to_stage3_when_stage4_hurts_real_data():
    scores = {"stage4_best": {"macro_f1": 0.60}, "stage3_best": {"macro_f1": 0.70}}
    assert choose_release(["stage4_best", "stage3_best"], scores, 0.01) == "stage3_best"


def test_release_without_scores_uses_most_advanced():
    assert choose_release(["stage3_best", "stage1_best"], {}, 0.01) == "stage3_best"


# ── Romanized Telugu ──────────────────────────────────────────────────

def test_romanize_telugu():
    pytest.importorskip("indic_transliteration")
    assert romanize_telugu("చంపేసావ్ పో!") == "champesav po!"
    assert romanize_telugu("నువ్వు చాలా") == "nuvvu chala"
    out = romanize_telugu("విషం కావాలి")
    assert out == "visham kavali" and out.isascii()


def test_romanize_leaves_latin_text_alone():
    assert romanize_telugu("already roman text") == "already roman text"


# ── display segments ──────────────────────────────────────────────────

class WhitespaceTokenizer:
    cls_token_id, sep_token_id, pad_token_id, bos_token_id, eos_token_id = 1, 2, 0, None, None

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(w, len(self.vocab) + 10) for w in text.split()]


def test_segments_mark_prefixes_turns_and_target():
    tok = WhitespaceTokenizer()
    enc = encode_example(tok, "champesav po", history=["old msg", "parent msg"], topic="Gaming")
    seg = enc["segments"]
    assert len(seg) == len(enc["input_ids"])
    assert seg.count("prefix") == 3                     # "Topic:", "Context_1:", "Parent:"
    assert seg.count("topic") == 1 and seg.count("turn-1") == 2 and seg.count("turn-2") == 2
    assert [i for i, s in enumerate(seg) if s == "target"] == [i for i, m in enumerate(enc["target_mask"]) if m]
