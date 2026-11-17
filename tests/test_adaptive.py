import copy

import numpy as np
import pytest

from contextual.adaptive_policy import (FALLBACK, apply_policy, fit_policies, margin_score,
    partition_validation, script_bucket, weight_grid, validate_policy)
from contextual.adaptive_experiment import context_controls, align_bundle
from contextual.statistics import paired_ap_interval


def validation_rows():
    rows=[]
    for source,text in [('en','English phrase'),('te','తెలుగు వాక్యం'),('ta','தமிழ் உரை')]:
        for i in range(160):
            rows.append({'id':f'{source}{i}','group':f'{source}{i}','source':source,'split':'val',
                         'label':i%2,'texts':[f'{text} {i}']})
    return rows


def test_script_is_not_english_detector_and_considers_history():
    assert script_bucket('meeru ela unnaru')=='latin_ambiguous'
    assert script_bucket('తెలుగు')=='telugu_script'
    assert script_bucket('தமிழ்')=='tamil_script'
    assert script_bucket('Hello',['తెలుగు'])=='mixed_script'
    assert script_bucket('123 😀')=='other_or_empty'


def test_expert_caps_and_boundary_preservation():
    assert all(w[0]==0 for w in weight_grid())
    assert all(w[0]<=.25 for w in weight_grid(.25))
    p=np.array([0,.2,.5,.8,1])
    for threshold in [0,.2,.5,1]:
        assert np.array_equal(margin_score(p,threshold)>=.5,p>=threshold)
    with pytest.raises(ValueError): margin_score([np.nan],.5)


def test_group_partition_and_selection_exclude_test_and_challenge():
    rows=validation_rows()
    fit,select=partition_validation(rows)
    assert not ({rows[i]['group'] for i in fit}&{rows[i]['group'] for i in select})
    for split in ('test','challenge','train'):
        changed=copy.deepcopy(rows); changed[0]['split']=split
        with pytest.raises(ValueError,match='validation only'): partition_validation(changed)


def test_mixed_baseline_is_retained_when_alternatives_are_worse():
    rows=validation_rows(); y=np.array([r['label'] for r in rows])
    good=np.where(y,.9,.1); bad=1-good
    result=fit_policies(rows,np.column_stack([bad,good,bad]),.5)
    assert result['policy']['name']=='mixed_existing'
    assert not set(result['fit_ids'])&set(result['selection_ids'])


def test_unseen_script_falls_back_and_fixed_route_ablation():
    rows=[{'texts':['123'],'label':1}]
    policy={'fallback':{'weights':FALLBACK,'threshold':.5},
            'branches':{'latin_ambiguous':{'weights':[.25,.75,0],'threshold':.5}}}
    values=np.array([[.99,.1,.9]])
    assert apply_policy(policy,rows,values)[0]==pytest.approx(.1)
    assert apply_policy(policy,rows,values,fixed_buckets=['latin_ambiguous'])[0]==pytest.approx(.3225)


def test_context_controls_do_not_replace_target_or_use_test_donors():
    rows=[{'id':'a','group':'a','source':'en','texts':['old','parent','target']}]
    donors=[{'group':'b','source':'en','split':'train','texts':['d1','d2','donor target']}]
    controls,eligible=context_controls(rows,donors)
    assert controls['reversed'][0]['texts']==['parent','old','target']
    assert controls['unrelated'][0]['texts']==['d1','d2','target']
    assert eligible==[True]
    donors[0]['split']='test'
    with pytest.raises(ValueError,match='training'): context_controls(rows,donors)


def test_ap_interval_identical_predictions():
    result=paired_ap_interval([0,1,0,1],[.1,.8,.2,.9],[.1,.8,.2,.9],['a','a','b','b'],resamples=20)
    assert result['ap_difference']==0
    assert result['percentile_95_interval']==[0,0]


def test_corrupted_scores_and_population_rejected():
    rows=[{'id':'x','group':'g','source':'en','label':1,'texts':['hello']}]
    bundle={'split':'val','experts':['english_ensemble','mixed_ensemble','mixed_neural'],
            'records':[{'id':'x','group':'g','source':'en','label':1,'bucket':'latin_ambiguous',
                        'scores':[.1,.2,.3],'isolated_scores':[.1,.2,.3]}]}
    assert align_bundle(bundle,rows,'val')[0].shape==(1,3)
    bundle['records'][0]['scores'][0]=float('nan')
    with pytest.raises(ValueError,match='score matrix'): align_bundle(bundle,rows,'val')


def test_unsafe_language_routing_artifact_rejected():
    policy={'fallback':{'weights':FALLBACK,'threshold':.5},
            'branches':{'latin_ambiguous':{'weights':[1,0,0],'threshold':.5}}}
    with pytest.raises(ValueError,match='routing limits'): validate_policy(policy)
    policy['branches']['latin_ambiguous']['weights']=[.25,.75,0]
    validate_policy(policy)
    policy['branches']['latin_ambiguous']['threshold']=float('nan')
    with pytest.raises(ValueError,match='threshold'): validate_policy(policy)
