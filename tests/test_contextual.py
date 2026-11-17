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


def test_english_prediction_exports_and_budget_sweep(tiny_config, tmp_path, monkeypatch):
    from contextual.training import train
    from contextual.target_only import train_target_only
    from contextual.inference import Predictor
    from contextual.context_sweep import score_budgets
    from contextual.data import read_jsonl
    cfg = tiny_config
    download(cfg)
    prepare(cfg)
    report = train(cfg)
    assert 'prepared_sha256' in report
    assert location(cfg, 'report').with_suffix('.predictions.json').exists()
    rows = [e for e in read_jsonl(location(cfg, 'prepared')) if e['split'] == 'test']
    predictor = Predictor(location(cfg, 'checkpoint'), device='cpu')
    scores = score_budgets(predictor, rows, [0, 1])
    direct = [predictor.predict(e['texts'][-1], e['texts'][-2:-1])['toxicity_score'] for e in rows]
    np.testing.assert_allclose(scores[1], direct, atol=1e-6)
    from contextual.context_sweep import main as sweep_main
    from contextual.evaluation import file_hash
    from contextual.config import write_json
    cfg['data']['expected_prepared_sha256'] = file_hash(location(cfg, 'prepared'))
    config_path = tmp_path/'sweep_config.json'
    output_path = tmp_path/'sweep.json'
    write_json(config_path, cfg)
    monkeypatch.setattr('sys.argv', ['context_sweep', '--config', str(config_path), '--output', str(output_path),
                                   '--device', 'cpu', '--budgets', '0', '3',
                                   '--reference-report', str(location(cfg, 'report'))])
    sweep_main()
    saved = json.loads(output_path.read_text())
    assert saved['serving_parity']['decision_flips_at_fixed_threshold'] == 0
    assert '3' in saved['contextual_budget_minus_zero_intervals']
    cfg['paths']['checkpoint'] = str(location(cfg, 'checkpoint').parent/'target_only')
    cfg['paths']['report'] = str(location(cfg, 'report').with_name('target_only.json'))
    target_report = train_target_only(cfg)
    assert 'prepared_sha256' in target_report
    for suffix in ('.predictions.json', '.validation_predictions.json'):
        assert location(cfg, 'report').with_suffix(suffix).exists()


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
    monkeypatch.setenv("TOXICITY_RUN_TRAINING", "1")
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


def test_custom_threads_schema_and_contrastive_pairs():
    from contextual.data import read_jsonl, examples_from_nodes
    cfg = load_config()
    custom_path = project_path(cfg["data"]["custom_threads"])
    assert custom_path.exists(), f"Missing custom threads at {custom_path}"
    records = read_jsonl(custom_path)
    assert len(records) >= 50
    examples = examples_from_nodes(records, cfg, "custom_threads", skip_missing_parents=False)
    assert len(examples) >= 40
    # Verify presence of multi-turn examples
    multiturn = [e for e in examples if len(e["texts"]) > 1]
    assert len(multiturn) >= 30
    # Verify contrastive twins: identical target text with differing labels in different contexts
    target_labels = {}
    twins_found = False
    for e in examples:
        target = e["texts"][-1].strip().lower()
        if target in target_labels and target_labels[target] != e["label"]:
            twins_found = True
            break
        target_labels[target] = e["label"]
    assert twins_found, "Custom threads must include contrastive twin pairs with context-dependent labels"


def test_online_encoder_matches_saved_pooling_and_has_tail_gradient(tiny_config):
    from contextual.embeddings import TextEncoder, encode_texts
    encoder = TextEncoder(tiny_config)
    encoder.model.requires_grad_(True)
    text = 'hello ' * 24 + 'attack ' * 24
    online = encode_texts(encoder.model, encoder.tokenizer, [text], encoder.device,
                          encoder.max_tokens, tiny_config['encoder']['batch_size'])
    torch.testing.assert_close(online.detach().cpu(), encoder.encode([text]))
    online.square().sum().backward()
    token = encoder.tokenizer.convert_tokens_to_ids('attack')
    assert encoder.model.get_input_embeddings().weight.grad[token].abs().sum() > 0


def test_cache_rejects_changed_encoder_weights(tiny_config):
    from contextual.embeddings import TextEncoder, cached_embeddings
    encoder = TextEncoder(tiny_config)
    rows = [{'texts': ['hello']}]
    original, _ = cached_embeddings(tiny_config, rows, encoder)
    with torch.no_grad():
        encoder.model.get_input_embeddings().weight[encoder.tokenizer.convert_tokens_to_ids('hello'), 0] += 2
    changed, _ = cached_embeddings(tiny_config, rows, encoder)
    assert not torch.allclose(original, changed)


def test_lora_full_validation_save_reload_and_resume(tiny_config):
    from contextual.adapter_finetune import run_adapter_experiment
    cfg = tiny_config
    cfg['adapter'] = {'enabled': True, 'r': 2, 'lora_alpha': 4, 'lora_dropout': 0,
                      'target_modules': ['query', 'value']}
    cfg['training'].update(encoder_lr=0.001, gradient_accumulation_steps=3,
                           gradient_checkpointing=False, require_cuda=False, fp16=False)
    download(cfg)
    manifest = prepare(cfg)
    report = run_adapter_experiment(cfg)
    assert report['validation']['n'] == manifest['splits']['val']
    assert report['training_examples'] == manifest['splits']['train']
    assert report['checkpoint_reload_verified']
    assert report['contextual_test']['n'] == report['contextual_test_without_parents']['n']
    assert run_adapter_experiment(cfg)['test'] == report['test']


def test_summary_uses_actual_context_subset():
    from contextual.evaluation import summarize
    rows = [{'id': 'a', 'source': 'cga', 'texts': ['root'], 'label': 0},
            {'id': 'b', 'source': 'cga', 'texts': ['root', 'reply'], 'label': 1}]
    report = summarize(rows, [0.9, 0.9], 0.5)
    assert report['by_source']['cga']['n'] == 2
    assert report['contextual_test']['n'] == 1
    assert report['contextual_test']['confusion_matrix'] == [[0, 0], [0, 1]]


def test_external_adapters_preserve_labels_and_human_ancestry(tmp_path):
    from contextual.external_data import convert_source
    toxichat = tmp_path / 'train.jsonl'
    toxichat.write_text(json.dumps({'reddit_thread': [
        {'turn_id': 1, 'text': 'a real parent', 'offense_label': 'Safe'},
        {'turn_id': 2, 'text': 'a real reply', 'offense_label': 'Offensive'}],
        'generated_reply': 'Must not enter the training data'})+'\n')
    source = {'name': 'toxichat_human', 'format': 'toxichat',
              'label_map': {'Safe': 0, 'Offensive': 1},
              'files': [{'path': str(toxichat), 'split': 'train'}]}
    examples = convert_source(source)
    assert len(examples) == 2
    assert examples[1]['texts'] == ['a real parent', 'a real reply']
    assert examples[1]['label'] == 1
    csv = tmp_path / 'telugu.csv'
    csv.write_text('text,label\nabusive text,0\nsafe text,1\n')
    rows = convert_source({'name': 'macd_telugu', 'format': 'macd',
        'label_map': {'0': 1, '1': 0}, 'files': [{'path': str(csv), 'split': 'test'}]})
    assert [r['label'] for r in rows] == [1, 0]
    assert all(len(r['texts']) == 1 and r['original_split'] == 'test' for r in rows)


def test_cli_routes_adapter_and_rejects_wrong_data_preparation(monkeypatch):
    import run_pipeline
    import contextual.adapter_finetune as adapter
    calls = []
    monkeypatch.setattr(adapter, 'run_adapter_experiment', lambda cfg: calls.append(cfg) or {})
    run_pipeline.main(['--config', 'configs/contextual_lora_v2.json', '--train'])
    assert len(calls) == 1 and calls[0]['adapter']['enabled']
    with pytest.raises(ValueError, match='external_data'):
        run_pipeline.main(['--config', 'configs/contextual_external.json', '--prepare'])


def test_conservative_preprocessing_and_clean_version():
    from contextual.preprocessing import normalize_text, clean_snapshot, audit
    assert normalize_text('  NOT\x00 toxic\t🙂 e\u0301 ') == 'NOT toxic 🙂 é'
    assert normalize_text('తెలుగు 👩\u200d💻') == 'తెలుగు 👩\u200d💻'
    rows = [{'id': f'{s}{y}', 'group': f'{s}{y}', 'source': 'fixture', 'split': s,
             'texts': [f'{s} unique message {y}'], 'label': y}
            for s in ('train', 'val', 'test') for y in (0, 1)]
    rows += [{**rows[0], 'id': 'duplicate'},
             {'id': 'amb0', 'group': 'amb', 'source': 'fixture', 'split': 'train', 'texts': ['uncertain'], 'label': 0},
             {'id': 'amb1', 'group': 'amb', 'source': 'fixture', 'split': 'train', 'texts': ['uncertain'], 'label': 1}]
    cleaned, excluded = clean_snapshot(rows)
    assert len(cleaned) == 6 and len(excluded) == 3
    assert audit(cleaned)['normalized_shared_text_groups_crossing_splits'] == 0
    assert rows[0]['texts'] == ['train unique message 0']


def test_preprocessing_rejects_normalization_leakage():
    from contextual.preprocessing import clean_snapshot
    rows = [{'id': f'{s}{y}', 'group': f'{s}{y}', 'source': 'fixture', 'split': s,
             'texts': [f'{s} text {y}'], 'label': y}
            for s in ('train', 'val', 'test') for y in (0, 1)]
    rows[0]['texts'] = ['é']
    rows[2]['texts'] = ['e\u0301']
    with pytest.raises(ValueError, match='crosses splits'):
        clean_snapshot(rows)


def test_ensemble_selection_and_prediction_alignment(tmp_path):
    from contextual.ensemble import select_weights, aligned_predictions
    selected, candidates = select_weights([0, 1, 0, 1], [[.1,.8],[.9,.2],[.2,.7],[.8,.3]], [1,1,1,1])
    assert selected['weights'] == [1., 0.]
    assert any(c['weights'] == [0., 1.] for c in candidates)
    path = tmp_path/'pred.json'
    path.write_text(json.dumps([{'id':'b','label':1,'score':.8},{'id':'a','label':0,'score':.2}]))
    rows = [{'id':'a','label':0},{'id':'b','label':1}]
    np.testing.assert_allclose(aligned_predictions(path, rows), [.2,.8])
    with pytest.raises(ValueError, match='IDs'):
        aligned_predictions(path, rows[:1])


def test_ensemble_roundtrip_and_neural_component(tiny_config, tmp_path):
    import joblib
    from contextual.adapter_finetune import run_adapter_experiment
    from contextual.ensemble import run_ensemble, EnsemblePredictor
    from contextual.data import read_jsonl
    from contextual.inference import Predictor
    cfg = tiny_config
    cfg['adapter'] = {'enabled': True, 'r': 2, 'lora_alpha': 4, 'lora_dropout': 0,
                      'target_modules': ['query', 'value']}
    cfg['training'].update(encoder_lr=.001, gradient_accumulation_steps=2,
                          gradient_checkpointing=False, require_cuda=False, fp16=False, epochs=1)
    cfg['data']['text_normalization'] = 'nfc_controls_whitespace_v1'
    download(cfg); prepare(cfg)
    neural = run_adapter_experiment(cfg)
    output = tmp_path/'ensemble.json'
    report = run_ensemble(cfg, output, location(cfg, 'report'))
    example = next(e for e in read_jsonl(location(cfg, 'prepared')) if e['split']=='test')
    predicted = EnsemblePredictor(output.with_suffix('.joblib')).predict(example['texts'][-1], example['texts'][:-1])
    from contextual.context_sweep import score_budgets
    selected_predictor = EnsemblePredictor(output.with_suffix('.joblib'))
    sweep = score_budgets(selected_predictor, [example], [0, 1])
    assert sweep[1][0] == pytest.approx(predicted['toxicity_score'], abs=1e-6)
    expected = json.loads(output.with_suffix('.predictions.json').read_text())[0]['score']
    assert predicted['toxicity_score'] == pytest.approx(expected, abs=1e-6)
    # Force a pure neural endpoint in this synthetic fixture to exercise the saved-weight checks and thread API.
    bundle = joblib.load(output.with_suffix('.joblib'))
    bundle['weights'] = {'char_tfidf':0., 'word_tfidf':0., 'contextual_transformer':1.}
    bundle['threshold'] = neural['threshold']
    joblib.dump(bundle, tmp_path/'neural_endpoint.joblib')
    ensemble = EnsemblePredictor(tmp_path/'neural_endpoint.joblib')
    assert score_budgets(ensemble, [example], [0, 1])[1][0] == pytest.approx(
        ensemble.predict(example['texts'][-1], example['texts'][:-1])['toxicity_score'], abs=1e-6)
    actual = ensemble.predict('hello\x00 please', ['attack'])
    reference = Predictor(location(cfg, 'checkpoint')).predict('hello please', ['attack'])
    assert actual['toxicity_score'] == pytest.approx(reference['toxicity_score'], abs=1e-6)
    threaded = ensemble.predict_thread([
        {'id':'p','parent_id':None,'conversation_id':'x','text':'attack'},
        {'id':'t','parent_id':'p','conversation_id':'x','text':'hello'}], 't')
    assert threaded['message_ids'] == ['p','t']


def test_group_bootstrap_and_connected_threads():
    from contextual.statistics import paired_f1_interval, connected_groups
    rows = [{'group':'a','texts':['shared','reply a']},
            {'group':'b','texts':['shared','reply b']},
            {'group':'c','texts':['independent']}]
    groups = connected_groups(rows)
    assert groups[0] == groups[1] and groups[0] != groups[2]
    interval = paired_f1_interval([0,1,0,1], [.1,.8,.2,.9], [.1,.8,.2,.9], ['a','a','b','b'], .5,.5, resamples=30)
    assert interval['f1_difference'] == 0
    assert interval['percentile_95_interval'] == [0.,0.]


def test_adaptive_saved_experts_roundtrip(tiny_config,tmp_path):
    from contextual.adapter_finetune import run_adapter_experiment
    from contextual.ensemble import run_ensemble
    from contextual.config import write_json
    from contextual.evaluation import file_hash
    from contextual.data import read_jsonl
    from contextual.adaptive_experiment import export_scores,fit,evaluate
    from contextual.adaptive_inference import AdaptivePredictor
    cfg=tiny_config
    cfg['adapter']={'enabled':True,'r':2,'lora_alpha':4,'lora_dropout':0,'target_modules':['query','value']}
    cfg['training'].update(encoder_lr=.001,gradient_accumulation_steps=2,gradient_checkpointing=False,
                           require_cuda=False,fp16=False,epochs=1)
    download(cfg); prepare(cfg)
    cfg['data']['expected_prepared_sha256']=file_hash(location(cfg,'prepared'))
    run_adapter_experiment(cfg)
    ensemble_path=tmp_path/'adaptive_expert.json'
    run_ensemble(cfg,ensemble_path,location(cfg,'report'))
    # Force a nonzero neural component so the diagnostic expert is available.
    import joblib
    bundle=joblib.load(ensemble_path.with_suffix('.joblib'))
    bundle['weights']={'char_tfidf':.5,'word_tfidf':0.,'contextual_transformer':.5}
    joblib.dump(bundle,ensemble_path.with_suffix('.joblib'))
    cfg['paths']['ensemble']=str(ensemble_path.with_suffix('.joblib'))
    config_path=tmp_path/'adaptive_expert_config.json'; write_json(config_path,cfg)
    settings={'mixed_config':str(config_path),'english_config':str(config_path),
              'scores_directory':str(tmp_path/'adaptive_scores'),'policy':str(tmp_path/'adaptive_policy.json'),
              'report':str(tmp_path/'adaptive_evaluation.json'),'demo_config':str(tmp_path/'adaptive_demo.json')}
    with pytest.raises(ValueError,match='Freeze'):
        export_scores(settings,split='test')
    export_scores(settings,split='val')
    policy=fit(settings)
    assert policy['policy']['name'] in {entry['policy']['name'] for entry in policy['leaderboard']}
    export_scores(settings,split='test')
    evaluated=evaluate(settings)
    records=json.loads((tmp_path/'adaptive_evaluation.predictions.json').read_text())
    example=next(e for e in read_jsonl(location(cfg,'prepared')) if e['id']==records[0]['id'])
    result=AdaptivePredictor(settings['policy']).predict(example['texts'][-1],example['texts'][:-1])
    assert result['toxicity_score']==pytest.approx(records[0]['score'],abs=1e-6)
    assert evaluated['model_type']=='conservative_adaptive_experts'
    assert result['language_identified'] is False
    # Check serving parity for all three experts even if this fixture selects the baseline.
    from contextual.adaptive_policy import margin_score
    score_bundle=json.loads((tmp_path/'adaptive_scores/test_scores.json').read_text())
    stored=next(r for r in score_bundle['records'] if r['id']==example['id'])
    adaptive=AdaptivePredictor(settings['policy'])
    weights=[.25,.25,.5]
    adaptive.bundle['policy']['branches']['latin_ambiguous']={'weights':weights,'threshold':.4}
    blended=adaptive.predict(example['texts'][-1],example['texts'][:-1])
    for key,scores in [('toxicity_score','scores'),('isolated_score','isolated_scores')]:
        expected=margin_score(sum(w*p for w,p in zip(weights,stored[scores])),.4)
        assert blended[key]==pytest.approx(float(expected),abs=1e-6)
    with pytest.raises(ValueError,match='already exists'): fit(settings)
