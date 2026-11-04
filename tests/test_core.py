"""
Core regression tests (no network, no GPU, no model download).

    python -m pytest tests -q
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_downloader import _label_to_int, _parse_categories
from src.inference_pipeline import FastTargetFilter
from src.model import FocalLoss, collate_encodings, encode_example, load_model_weights
from src.preprocessing import deobfuscate
from src.training_utils import clean_str


class WhitespaceTokenizer:
    """Minimal tokenizer stand-in: one id per whitespace token."""
    cls_token_id, sep_token_id, pad_token_id, bos_token_id, eos_token_id = 1, 2, 0, None, None

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(w, len(self.vocab) + 10) for w in text.split()]


# ── input assembly ─────────────────────────────────────────────────────

def test_target_mask_marks_exactly_the_target():
    tok = WhitespaceTokenizer()
    enc = encode_example(tok, "champesav po", history=["boss level clear"], topic="Gaming")
    target_ids = [i for i, m in zip(enc["input_ids"], enc["target_mask"]) if m]
    assert target_ids == tok.encode("champesav po")
    assert enc["input_ids"][0] == tok.cls_token_id and enc["input_ids"][-1] == tok.sep_token_id


def test_long_context_never_drops_target():
    tok = WhitespaceTokenizer()
    history = ["old " * 500, "middle " * 300, "the actual parent"]
    enc = encode_example(tok, "champesav po", history=history, topic="Politics", max_length=64)
    assert len(enc["input_ids"]) <= 64
    assert sum(enc["target_mask"]) == 2
    assert all(i in enc["input_ids"] for i in tok.encode("the actual parent"))  # newest turn kept


def test_no_context_input():
    tok = WhitespaceTokenizer()
    enc = encode_example(tok, "hello there")
    assert enc["input_ids"] == [1] + tok.encode("hello there") + [2]
    assert enc["target_mask"] == [0, 1, 1, 0]


def test_collate_pads_dynamically():
    tok = WhitespaceTokenizer()
    a = {**encode_example(tok, "a b c"), "labels": [1, -1, -1, -1, -1]}
    b = {**encode_example(tok, "d"), "labels": [0, 0, 0, 0, 0]}
    batch = collate_encodings([a, b], pad_id=0)
    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"][1].tolist() == [1, 1, 1, 0, 0]


# ── loss ───────────────────────────────────────────────────────────────

def test_focal_loss_ignores_unknown_labels():
    fl = FocalLoss()
    logits = torch.tensor([[2.0, -3.0, 5.0]])
    masked = fl(logits, torch.tensor([[1.0, -1.0, -1.0]]))
    only_first = fl(logits[:, :1], torch.tensor([[1.0]]))
    assert torch.isclose(masked, only_first)


# ── checkpoint loading ─────────────────────────────────────────────────

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.head = nn.Linear(2, 1)


def test_strict_loader_rejects_missing_weights(tmp_path):
    torch.save({"head.weight": torch.zeros(1, 2), "head.bias": torch.zeros(1)}, tmp_path / "model.pt")
    with pytest.raises(RuntimeError, match="encoder weights missing"):
        load_model_weights(TinyModel(), tmp_path)


def test_strict_loader_accepts_full_state(tmp_path):
    torch.save(TinyModel().state_dict(), tmp_path / "model.pt")
    load_model_weights(TinyModel(), tmp_path)


# ── lexicon gate ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def lexicon():
    return FastTargetFilter()


@pytest.mark.parametrize("text", ["fuck!", "lanja!", "madarchod!", "f*ck you", "ఎర్రిపూకు నువ్వు"])
def test_severe_terms_detected(lexicon, text):
    assert lexicon.analyze(text)["severe"]


@pytest.mark.parametrize("text", ["kill you bro", "good night, I will kill you", "bc I was late",
                                  "Kosi karam pettali veellaki", "Bro that was fucking insane clutch!"])
def test_ambiguous_text_is_not_fast_flagged(lexicon, text):
    # These must reach the model: either no hit, or only mild terms.
    assert not lexicon.analyze(text)["severe"]


# ── preprocessing & data helpers ───────────────────────────────────────

def test_deobfuscate():
    assert deobfuscate("m@drch0d") == "madrchod"
    assert deobfuscate("gooood movie") == "good movie"
    assert deobfuscate("https://x.com @bot l@nja") == "lanja"


@pytest.mark.parametrize("value,expected", [(None, ""), (float("nan"), ""), ("nan", ""), (" hi ", "hi")])
def test_clean_str(value, expected):
    assert clean_str(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("['Violence & Physical Harm', 'Criminal Activity Planning']", ["Violence & Physical Harm", "Criminal Activity Planning"]),
    ('["Harassment & Profanity"]', ["Harassment & Profanity"]),
    (["Hate Speech"], ["Hate Speech"]),
    ("[]", []),
    (None, []),
])
def test_parse_categories(value, expected):
    assert _parse_categories(value) == expected


@pytest.mark.parametrize("value,expected", [
    (1, 1), (0, 0), ("1", 1), (True, 1), ("Hate", 1), ("Non-Hate", 0),
    ("Not_offensive", 0), ("Offensive_Targeted_Insult_Individual", 1), ("normal", 0),
])
def test_label_to_int(value, expected):
    assert _label_to_int(value) == expected
