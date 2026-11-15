import copy
import json
import zipfile

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

from contextual.config import load_config, location, project_path
from contextual.data import ancestor_chain, examples_from_nodes, grouped_split, prepare, download
from contextual.model import ConversationClassifier
from contextual.training import fit_threshold


def test_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    original = location(load_config(), "prepared")
    monkeypatch.chdir(tmp_path)
    assert location(load_config(), "prepared") == original
    assert original.is_absolute()


def test_ancestry_excludes_sibling_and_future():
    nodes = {i: {"id": i, "text": i, "parent_id": parent, "conversation_id": "thread"}
             for i, parent in (("root", None), ("child", "root"), ("sibling", "root"), ("future", "child"))}
    assert [n["id"] for n in ancestor_chain(nodes, "child", 10)] == ["root", "child"]
    assert [n["id"] for n in ancestor_chain(nodes, "child", 1)] == ["child"]
    nodes["root"]["parent_id"] = "child"
    with pytest.raises(ValueError, match="cycle"):
        ancestor_chain(nodes, "child", 1)


def test_broken_parent_rejected():
    nodes = {"a": {"text": "a", "parent_id": "missing", "conversation_id": "one"}}
    with pytest.raises(ValueError, match="Missing parent"):
        ancestor_chain(nodes, "a", 10)


def test_incomplete_corpus_chains_are_skipped_without_inventing_context():
    cfg = load_config()
    records = [
        {"id": "root", "parent_id": None, "conversation_id": "one", "text": "hello", "label": 0},
        {"id": "child", "parent_id": "root", "conversation_id": "one", "text": "reply", "label": 1},
        {"id": "orphan", "parent_id": "absent", "conversation_id": "two", "text": "orphan", "label": 0},
        {"id": "descendant", "parent_id": "orphan", "conversation_id": "two", "text": "reply", "label": 1},
        {"id": "blank", "parent_id": None, "conversation_id": "three", "text": " ", "label": 0},
        {"id": "after_blank", "parent_id": "blank", "conversation_id": "three", "text": "reply", "label": 1},
        {"id": "cycle", "parent_id": "cycle", "conversation_id": "four", "text": "reply", "label": 0},
    ]
    with pytest.raises(ValueError, match="Missing parent"):
        examples_from_nodes(records, cfg, "custom_threads")
    stats = {}
    examples = examples_from_nodes(records, cfg, "cga_wiki", skip_missing_parents=True, stats=stats)
    assert [e["message_ids"] for e in examples] == [["root"], ["root", "child"]]
    assert stats == {"skipped_missing_ancestry": 2, "skipped_empty_ancestry": 2,
                     "skipped_invalid_ancestry": 1}


def test_shared_messages_never_cross_splits():
    examples = [{"group": str(i), "texts": [f"unique parent {i}", f"label {label} target {i}"], "label": label}
                for i in range(40) for label in (0, 1)]
    examples.extend([{"group": "duplicate", "texts": examples[0]["texts"], "label": 0}])
    result = grouped_split(examples, load_config())
    seen = {}
    for ex in result:
        for text in ex["texts"]:
            assert seen.setdefault(text, ex["split"]) == ex["split"]


def test_padding_and_parent_effect_and_gradients():
    cfg = load_config()
    cfg["model"].update(hidden_size=12, heads=3, layers=1, feedforward_size=24, dropout=0)
    torch.manual_seed(1)
    model = ConversationClassifier(8, cfg).eval()
    values = torch.randn(1, 2, 8, requires_grad=True)
    result = model(values, torch.ones(1, 2, dtype=torch.bool))
    padded = torch.cat([values, torch.randn(1, 1, 8)], dim=1)
    again = model(padded, torch.tensor([[True, True, False]]))
    torch.testing.assert_close(result["logits"], again["logits"])
    assert again["turn_attention"][0, -1] == 0
    result["logits"].sum().backward()
    assert values.grad[0, 0].abs().sum() > 0
    assert values.grad[0, 1].abs().sum() > 0
    changed = values.detach().clone()
    changed[:, 0] *= -1
    assert not torch.allclose(model(changed, torch.ones(1, 2, dtype=torch.bool))["logits"], result["logits"])


def test_threshold_fits_observed_scores():
    threshold = fit_threshold([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4])
    assert threshold == pytest.approx(0.3)
    with pytest.raises(ValueError):
        fit_threshold([0, 0], [0.1, 0.2])


def test_refuse_isolated_context_training(tmp_path):
    cfg = load_config()
    cfg["paths"]["raw"] = str(tmp_path)
    cfg["data"]["custom_threads"] = str(tmp_path / "absent.jsonl")
    with pytest.raises(ValueError, match="No real parent-child"):
        prepare(cfg)


@pytest.fixture
def tiny_config(tmp_path):
    """Synthetic fixture for execution tests only, never shipped as trained model."""
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    cfg = copy.deepcopy(load_config())
    encoder = tmp_path / "tiny_encoder"
    encoder.mkdir()
    vocab = {word: i for i, word in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "stop", "please", "attack"])}
    backend = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
        pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    tokenizer.save_pretrained(encoder)
    BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=12, num_hidden_layers=1,
                        num_attention_heads=3, intermediate_size=24, max_position_embeddings=32)).save_pretrained(encoder)
    cfg["encoder"].update(name=str(encoder), local_files_only=True, max_tokens=16, batch_size=8)
    cfg["model"].update(hidden_size=12, heads=3, layers=1, feedforward_size=24, dropout=0, max_turns=4)
    cfg["training"].update(epochs=2, batch_size=16, device="cpu", cpu_threads=1)
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = str(tmp_path / value)
    cfg["baseline"] = {"max_features": 100, "ngram_range": [2, 3], "C": 1.0, "max_iter": 100, "report": str(tmp_path / "baseline.json")}
    cfg["training"]["context_dropout"] = 0.2
    cfg["data"]["local_sources"] = []
    cfg["data"]["custom_threads"] = str(tmp_path / "absent.jsonl")
    raw = location(cfg, "raw")
    raw.mkdir(parents=True)
    nodes = []
    for i in range(40):
        nodes.extend([{"id": f"p{i}", "conversation_id": str(i), "reply-to": None, "text": f"hello {i}",
                       "meta": {"comment_has_personal_attack": 0}},
                      {"id": f"c{i}", "conversation_id": str(i), "reply-to": f"p{i}", "text": f"attack {i}",
                       "meta": {"comment_has_personal_attack": 1}}])
    with zipfile.ZipFile(raw / cfg["data"]["cga"]["archive"], "w") as archive:
        archive.writestr("corpus/utterances.jsonl", "\n".join(json.dumps(n) for n in nodes))
    return cfg


def test_complete_offline_pipeline(tiny_config):
    from contextual.training import train, evaluate
    from contextual.inference import Predictor
    cfg = tiny_config
    download(cfg)
    manifest = prepare(cfg)
    assert manifest["real_context_examples"] == 40
    report = train(cfg)
    assert report["test"]["n"] > 0
    predictor = Predictor(location(cfg, "checkpoint"))
    result = predictor.predict("hello", ["attack"], include_embeddings=True)
    assert np.asarray(result["individual_embeddings"]).shape == (2, 12)
    assert len(result["conversation_embedding"]) == 12
    assert len(result["turn_attention"]) == 2
    assert 0 <= result["toxicity_score"] <= 1
    assert predictor.predict("hello")["context_score_delta"] == pytest.approx(0)
    assert evaluate(cfg) == report
    with pytest.raises(TypeError):
        predictor.predict("hello", topic="gaming")


def test_long_text_chunks_preserve_content(tiny_config):
    from contextual.embeddings import TextEncoder
    encoder = TextEncoder(tiny_config)
    short = encoder.encode(["hello " * 20])
    extended = encoder.encode(["hello " * 20 + "attack " * 20])
    assert short.shape == extended.shape
    assert not torch.allclose(short, extended)


def test_api_rejects_topic_and_untrained_predictions(monkeypatch):
    from fastapi import HTTPException
    from pydantic import ValidationError
    from app import server
    monkeypatch.setattr(server, "predictor", None)
    # Exercise schema + handler without localhost sockets (restricted CI compatible).
    with pytest.raises(ValidationError):
        server.ModerateRequest(text="hello", topic="gaming")
    with pytest.raises(HTTPException) as exc:
        server.moderate(server.ModerateRequest(text="hello"))
    assert exc.value.status_code == 503
    assert server.index().status_code == 200
    assert not server.health()["ready"]


def test_notebook_run_all(tiny_config, tmp_path, monkeypatch):
    import nbformat
    from contextual.config import ROOT
    config_path = tmp_path / "notebook_config.json"
    config_path.write_text(json.dumps(tiny_config), encoding="utf-8")
    monkeypatch.setenv("TOXICITY_CONFIG", str(config_path))
    notebook = nbformat.read(ROOT / "Context_Aware_Toxicity.ipynb", as_version=4)
    nbformat.validate(notebook)
    monkeypatch.chdir(ROOT / "notebooks")
    namespace = {"__name__": "__main__"}
    # Run the unmodified cells in order in this Python environment. A live
    # Jupyter kernel additionally needs localhost sockets, unavailable in CI.
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"notebook_cell_{i}", "exec"), namespace)
    assert namespace["thread_result"]["turns_used"] > 0
