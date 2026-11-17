import copy

import numpy as np
import pytest

from contextual.config import load_config
from contextual.data import ancestor_chain, examples_from_nodes
from contextual.english_experiment import align_anchor, depth_bin, make_configs, source_settings
from contextual.context_sweep import budget_rows, grouped_metrics
from contextual.benchmark_compare import align_predictions, verify_anchor_subset


def sample_rows():
    return [{'id': str(i), 'group': str(i), 'source': 'cga_wiki', 'label': label,
             'texts': [f'unique {i} text'], 'message_ids': [str(i)], 'split': split,
             'available_ancestors': 0, 'depth_provenance': 'fixture'}
            for i, (split, label) in enumerate((s, y) for s in ('train', 'val', 'test') for y in (0, 1))]


def test_full_depth_survives_window_and_cannot_hide_missing_ancestor():
    cfg = load_config()
    cfg['data']['capture_ancestry_counts'] = True
    cfg['model']['max_turns'] = 3
    records = [{'id': str(i), 'parent_id': str(i-1) if i else None, 'conversation_id': 't',
                'text': f'turn {i}', 'label': i % 2} for i in range(20)]
    rows = examples_from_nodes(records, cfg, 'fixture')
    assert rows[-1]['available_ancestors'] == 19
    assert rows[-1]['retained_ancestors'] == 2
    assert rows[-1]['message_ids'] == ['17', '18', '19']
    records[0]['parent_id'] = 'absent'
    with pytest.raises(ValueError, match='Missing parent'):
        examples_from_nodes(records, cfg, 'fixture')


def test_zero_window_rejected():
    with pytest.raises(ValueError, match='max_turns'):
        ancestor_chain({}, 'missing', 0)


def test_anchor_alignment_preserves_population_and_rejects_source_drift():
    anchor = sample_rows()
    raw = copy.deepcopy(anchor)
    assert align_anchor(raw, anchor, 'cga_wiki', 12)[0]['split'] == 'train'
    raw[0]['texts'] = ['changed label evidence']
    with pytest.raises(ValueError, match='Anchor text'):
        align_anchor(raw, anchor, 'cga_wiki', 12)


def test_budget_zero_keeps_target_and_does_not_mutate_depth():
    row = {'texts': ['a', 'b', 'c', 'd'], 'available_ancestors': 9}
    assert budget_rows([row], 0)[0]['texts'] == ['d']
    assert budget_rows([row], 2)[0]['texts'] == ['b', 'c', 'd']
    assert budget_rows([row], 2)[0]['available_ancestors'] == 9
    assert len(row['texts']) == 4
    with pytest.raises(ValueError): budget_rows([row], -1)


def test_empty_and_unknown_bins_are_not_invented():
    rows = sample_rows()[:2]
    del rows[1]['available_ancestors']
    result = grouped_metrics(rows, [.1, .9], .5)
    assert result['by_available_ancestors']['12+'] is None
    assert result['by_available_ancestors']['unknown']['n'] == 1
    assert result['by_available_ancestors']['0']['average_precision'] is None
    assert depth_bin(12) == '12+'
    with pytest.raises(ValueError): grouped_metrics(rows, [np.nan, .9], .5)


def test_report_pairing_rejects_wrong_population_and_labels():
    rows = sample_rows()
    records = [{**r, 'score': .5} for r in rows]
    assert len(align_predictions(records, rows)) == 6
    with pytest.raises(ValueError): align_predictions(records[:-1], rows)
    records[-1]['label'] = 0
    with pytest.raises(ValueError, match='labels'): align_predictions(records, rows)
    other = copy.deepcopy(rows)
    other[0]['split'] = 'test'
    with pytest.raises(ValueError, match='held-out'): verify_anchor_subset(other, rows)


def test_configs_are_separate_and_downloads_are_english_only():
    configs = make_configs('english_fixture', 'hash', 12)
    assert len(configs) == 4
    for cfg in configs.values():
        assert 'english_fixture' in cfg['paths']['checkpoint']
        assert cfg['data']['expected_prepared_sha256'] == 'hash'
        assert cfg['data']['local_sources'] == []
    assert all(s['format'] == 'toxichat' for s in source_settings()['sources'])
