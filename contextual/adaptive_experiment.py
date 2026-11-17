"""Score existing experts, select a conservative policy on validation, evaluate separately."""
import argparse
import copy
import gc
import json
import time

import numpy as np
import torch

from .adaptive_policy import EXPERTS, apply_policy, fit_policies, row_bucket
from .benchmark_compare import verify_anchor_subset
from .config import load_config, location, project_path, write_json
from .data import examples_from_nodes, read_jsonl, fingerprint
from .ensemble import EnsemblePredictor
from .evaluation import file_hash, validate_splits
from .preprocessing import normalize_text
from .statistics import connected_groups, paired_f1_interval, paired_ap_interval
from .training import predict_examples, metrics


def read_settings(path='configs/adaptive_experiment.json'):
    return json.loads(project_path(path).read_text(encoding='utf-8'))


def populations(settings):
    mixed = load_config(settings['mixed_config'])
    english = load_config(settings['english_config'])
    if mixed['model']['max_turns'] != english['model']['max_turns']:
        raise ValueError('Experts must use the same parent budget for this experiment')
    for cfg in (mixed,english):
        if file_hash(location(cfg,'prepared')) != cfg['data']['expected_prepared_sha256']:
            raise ValueError('Prepared expert dataset changed')
    rows = read_jsonl(location(mixed,'prepared'))
    erows = read_jsonl(location(english,'prepared'))
    validate_splits(erows)
    verify_anchor_subset(erows,rows)
    depth = {e['id']:e.get('available_ancestors') for e in erows}
    for row in rows:
        if row['id'] in depth and depth[row['id']] is not None:
            row['available_ancestors'] = depth[row['id']]
        elif len(row['texts']) == 1:
            row['available_ancestors'] = 0
    return mixed, english, validate_splits(rows)


def artifact_identity(cfg):
    artifact = project_path(cfg['paths']['ensemble'])
    checkpoint = location(cfg,'checkpoint')
    paths = [checkpoint/'metadata.json', checkpoint/'head.pt'] + sorted((checkpoint/'encoder').glob('*'))
    return {'ensemble':str(artifact), 'ensemble_sha256':file_hash(artifact),
            'checkpoint_files':{str(p):file_hash(p) for p in paths if p.is_file()}}


def validate_identity(identity):
    if file_hash(identity['ensemble']) != identity['ensemble_sha256']:
        raise ValueError('Expert ensemble changed after score export')
    if any(file_hash(path) != digest for path,digest in identity['checkpoint_files'].items()):
        raise ValueError('Expert weights changed after score export')


def context_controls(rows, training_rows):
    """Stress controls retain target; unrelated parents come only from training."""
    rng=np.random.default_rng(2026)
    pools={}
    for row in training_rows:
        if row.get('split') != 'train': raise ValueError('Control donors must be training rows')
        if len(row['texts'])>1:
            pools.setdefault(row['source'],[]).append(row)
    reversed_rows, unrelated_rows, eligible=[],[],[]
    for row in rows:
        history=row['texts'][:-1]; target=row['texts'][-1]
        reversed_rows.append({**row,'texts':list(reversed(history))+[target]})
        candidates=pools.get(row['source'],[])
        donor=None
        if history and candidates:
            for index in rng.choice(len(candidates),size=min(len(candidates),40),replace=False):
                candidate=candidates[index]
                if len(candidate['texts'])-1 >= len(history) and candidate['group']!=row['group']:
                    parents=candidate['texts'][-(len(history)+1):-1]
                    if fingerprint(target) not in {fingerprint(t) for t in parents} and parents!=history:
                        donor=parents; break
        unrelated_rows.append({**row,'texts':(donor if donor is not None else history)+[target]})
        eligible.append(donor is not None)
    return {'reversed':reversed_rows,'unrelated':unrelated_rows}, eligible


def expert_scores(predictor, rows, controls=None):
    """Reuse one encoding pass for full and removed context and the neural component."""
    neural = predictor.neural
    if neural is None:
        raise ValueError('This experiment requires the existing nonzero neural expert')
    controls=controls or {}
    all_rows=rows+[r for values in controls.values() for r in values]
    texts = list(dict.fromkeys(t for r in all_rows for t in r['texts']))
    print(f'Encoding {len(texts)} messages for {len(rows)} targets',flush=True)
    vectors = neural.encoder.encode(texts)
    lookup = {t:i for i,t in enumerate(texts)}
    full = predict_examples(neural.model, rows, vectors, lookup, neural.cfg)
    alone = predict_examples(neural.model,[{**r,'texts':[r['texts'][-1]]} for r in rows],vectors,lookup,neural.cfg)
    target = [normalize_text(r['texts'][-1]) for r in rows]
    lexical = np.zeros(len(rows))
    components = {}
    for name,model in predictor.bundle['lexical'].items():
        components[name] = model.predict_proba(target)[:,1]
        lexical += predictor.bundle['weights'].get(name,0)*components[name]
    weight = predictor.bundle['weights'].get('contextual_transformer',0)
    result={'full':lexical+weight*full, 'alone':lexical+weight*alone,
            'neural_full':full, 'neural_alone':alone,
            'char_score':components['char_tfidf'], 'word_score':components['word_tfidf']}
    for name,values in controls.items():
        score=predict_examples(neural.model,values,vectors,lookup,neural.cfg)
        result[name]=lexical+weight*score
        result['neural_'+name]=score
    return result


def export_scores(settings, split='val', device='cpu', challenge=None):
    frozen = None
    if split == 'test' or challenge:
        if not project_path(settings['policy']).exists():
            raise ValueError('Freeze the validation policy before scoring test/challenge')
        frozen = json.loads(project_path(settings['policy']).read_text(encoding='utf-8'))
    mixed, english, splits = populations(settings)
    identities = {name:artifact_identity(cfg) for name,cfg in [('english',english),('mixed',mixed)]}
    if frozen and (identities != frozen['identities'] or
                   mixed['data']['expected_prepared_sha256'] != frozen['prepared_sha256']):
        raise ValueError('Data/experts differ from frozen validation policy')
    if challenge:
        c = copy.deepcopy(mixed)
        c['data']['capture_ancestry_counts'] = True
        rows = examples_from_nodes(read_jsonl(project_path(challenge)),c,'custom_threads')
        rows = [{**r,'texts':[normalize_text(t) for t in r['texts']]} for r in rows]
        split = 'challenge'
        population_hash = file_hash(project_path(challenge))
    else:
        rows = splits[split]
        population_hash = file_hash(location(mixed,'prepared'))
    if not rows: raise ValueError('Empty population')
    controls, eligible=context_controls(rows,splits['train'])
    output = project_path(settings['scores_directory'])/(split+'_scores.json')
    if output.exists():
        raise ValueError(f'{output} exists. Preserve the run; choose a new scores directory for another run.')
    results, thresholds = {}, {}
    torch.set_num_threads(mixed['training']['cpu_threads'])
    start = time.monotonic()
    # Sequential loading keeps both large encoders from occupying GPU memory together.
    for name,cfg in [('english',english),('mixed',mixed)]:
        print('Scoring expert:',name,split,flush=True)
        predictor = EnsemblePredictor(cfg['paths']['ensemble'],device=device)
        if predictor.bundle['prepared_sha256'] != cfg['data']['expected_prepared_sha256']:
            raise ValueError('Expert artifact dataset mismatch')
        results[name] = expert_scores(predictor,rows,controls)
        thresholds[name+'_ensemble'] = predictor.threshold
        if name == 'mixed': thresholds['mixed_neural'] = predictor.neural.threshold
        del predictor
        gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()
    records = []
    for i,row in enumerate(rows):
        records.append({'id':row['id'],'group':row['group'],'source':row['source'],'label':row['label'],
            'split':split,'bucket':row_bucket(row),'has_context':len(row['texts'])>1,
            'available_ancestors':row.get('available_ancestors'),
            'scores':[float(results['english']['full'][i]),float(results['mixed']['full'][i]),float(results['mixed']['neural_full'][i])],
            'isolated_scores':[float(results['english']['alone'][i]),float(results['mixed']['alone'][i]),float(results['mixed']['neural_alone'][i])],
            'reversed_scores':[float(results['english']['reversed'][i]),float(results['mixed']['reversed'][i]),float(results['mixed']['neural_reversed'][i])],
            'unrelated_scores':[float(results['english']['unrelated'][i]),float(results['mixed']['unrelated'][i]),float(results['mixed']['neural_unrelated'][i])],
            'unrelated_eligible':eligible[i],
            'lexical_scores':{n:{k:float(results[n][k][i]) for k in ('char_score','word_score')} for n in results}})
    result = {'split':split,'population_sha256':population_hash,'experts':list(EXPERTS),
              'identities':identities,'thresholds':thresholds,'records':records,
              'device':device,'precision':'FP32','seconds':time.monotonic()-start}
    write_json(output,result)
    print(output)
    return result


def align_bundle(bundle, rows, split):
    records = bundle['records']
    lookup = {r['id']:r for r in records}
    if bundle['split'] != split or bundle['experts'] != list(EXPERTS) or len(lookup) != len(records) or set(lookup) != {r['id'] for r in rows}:
        raise ValueError('Score population/expert order mismatch')
    for row in rows:
        r = lookup[row['id']]
        if any(r[k] != row[k] for k in ('label','source','group')) or r['bucket'] != row_bucket(row):
            raise ValueError('Stored score metadata differs from frozen population')
    matrix = np.asarray([lookup[r['id']]['scores'] for r in rows])
    alone = np.asarray([lookup[r['id']]['isolated_scores'] for r in rows])
    for values in (matrix,alone):
        if values.shape != (len(rows),3) or not np.isfinite(values).all() or ((values < 0)|(values>1)).any():
            raise ValueError('Invalid expert score matrix')
    return matrix, alone


def fit(settings):
    policy_path = project_path(settings['policy'])
    if policy_path.exists(): raise ValueError('Frozen policy already exists; use a new experiment directory')
    if project_path(settings['demo_config']).exists():
        raise ValueError('Demo config already exists; preserve it and use a new experiment name')
    mixed, english, splits = populations(settings)
    path = project_path(settings['scores_directory'])/'val_scores.json'
    bundle = json.loads(path.read_text(encoding='utf-8'))
    if bundle['population_sha256'] != mixed['data']['expected_prepared_sha256']:
        raise ValueError('Validation identity mismatch')
    for identity in bundle['identities'].values(): validate_identity(identity)
    matrix,_ = align_bundle(bundle,splits['val'],'val')
    result = fit_policies(splits['val'],matrix,bundle['thresholds']['mixed_ensemble'])
    result.update(identities=bundle['identities'], thresholds=bundle['thresholds'],
                  prepared_sha256=bundle['population_sha256'], validation_scores_sha256=file_hash(path),
                  config=mixed, experts=list(EXPERTS))
    write_json(policy_path,result)
    # This report is validation only until explicit evaluate; it is never a test result.
    write_json(policy_path.with_name('selection.json'),result)
    cfg = copy.deepcopy(mixed)
    cfg['paths'].pop('ensemble',None)
    cfg['paths']['adaptive'] = settings['policy']
    cfg['paths']['report'] = settings['report']
    cfg['training']['device'] = 'cpu'
    write_json(project_path(settings['demo_config']),cfg)
    print('Frozen selected policy:',result['policy']['name'])
    return result


def evaluate(settings, challenge=None):
    mixed,_,splits = populations(settings)
    policy = json.loads(project_path(settings['policy']).read_text(encoding='utf-8'))
    split = 'challenge' if challenge else 'test'
    bundle = json.loads((project_path(settings['scores_directory'])/(split+'_scores.json')).read_text(encoding='utf-8'))
    if bundle['identities'] != policy['identities']:
        raise ValueError('Evaluation experts differ from frozen selection experts')
    for identity in policy['identities'].values(): validate_identity(identity)
    if challenge:
        c=copy.deepcopy(mixed); c['data']['capture_ancestry_counts']=True
        rows=examples_from_nodes(read_jsonl(project_path(challenge)),c,'custom_threads')
        rows=[{**r,'texts':[normalize_text(t) for t in r['texts']]} for r in rows]
        expected=file_hash(project_path(challenge))
    else:
        rows=splits['test']; expected=mixed['data']['expected_prepared_sha256']
    if bundle['population_sha256'] != expected: raise ValueError('Evaluation data changed')
    matrix,alone=align_bundle(bundle,rows,split)
    from .context_sweep import grouped_metrics
    p=apply_policy(policy['policy'],rows,matrix)
    q=apply_policy(policy['policy'],rows,alone)  # Keep routing fixed to isolate the context effect.
    y=[r['label'] for r in rows]; groups=connected_groups(rows)
    result={'model_type':'conservative_adaptive_experts','threshold':.5,'policy':policy['policy'],
            'prepared_sha256':policy['prepared_sha256'],'evaluated_sha256':expected,'population':split,
            'test':metrics(y,p,.5),'details':grouped_metrics(rows,p,.5),
            'without_context_fixed_route':grouped_metrics(rows,q,.5),
            'expert_baselines':{},'paired_vs_experts':{}, 'routing_counts':{},
            'paired_by_source':{}, 'component_diagnostics':{},
            'selection_limitations':policy['limitations'],
            'score_semantics':'Threshold-normalized decision score, NOT calibrated probability.',
            'seconds_to_score_experts':bundle['seconds']}
    for i,name in enumerate(EXPERTS):
        threshold=policy['thresholds'][name]
        result['expert_baselines'][name]=grouped_metrics(rows,matrix[:,i],threshold)
        result['paired_vs_experts'][name]=paired_f1_interval(y,p,matrix[:,i],groups,.5,threshold)
    for source in sorted({r['source'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['source']==source]
        result['paired_by_source'][source]={name:paired_f1_interval(
            [y[i] for i in ids],p[ids],matrix[ids,j],[groups[i] for i in ids],.5,policy['thresholds'][name])
            for j,name in enumerate(EXPERTS)}
    for bucket in sorted({row_bucket(r) for r in rows}):
        ids=[i for i,r in enumerate(rows) if row_bucket(r)==bucket]
        result['routing_counts'][bucket]={'n':len(ids),'metrics':metrics([y[i] for i in ids],p[ids],.5)}
    ids=[i for i,r in enumerate(rows) if len(r['texts'])>1]
    if ids:
        result['context_f1_interval']=paired_f1_interval([y[i] for i in ids],p[ids],q[ids],
                                                       [groups[i] for i in ids],.5,.5)
        if len({y[i] for i in ids})==2:
            result['context_ap_interval']=paired_ap_interval([y[i] for i in ids],p[ids],q[ids],[groups[i] for i in ids])
    lookup={r['id']:r for r in bundle['records']}
    # Score ranges help distinguish weak ranking from an unsuitable operating threshold.
    # Lexical scores have no independently selected thresholds in this experiment.
    for name in ('english','mixed'):
        result['component_diagnostics'][name]={}
        for component in ('char_score','word_score'):
            result['component_diagnostics'][name][component]={}
            for label in (0,1):
                values=[lookup[r['id']]['lexical_scores'][name][component] for r in rows if r['label']==label]
                result['component_diagnostics'][name][component][str(label)]={
                    'n':len(values),'mean':float(np.mean(values)) if values else None,
                    'min':float(min(values)) if values else None,'max':float(max(values)) if values else None}
    result['context_controls']={}
    for name in ('reversed','unrelated'):
        control_matrix=np.asarray([lookup[r['id']][name+'_scores'] for r in rows])
        cp=apply_policy(policy['policy'],rows,control_matrix)
        ids=[i for i,r in enumerate(rows) if (len(r['texts'])>2 if name=='reversed' else lookup[r['id']]['unrelated_eligible'])]
        result['context_controls'][name]={'n':len(ids),
            'original_label_metrics':metrics([y[i] for i in ids],cp[ids],.5) if ids else None,
            'true_context_metrics_same_subset':metrics([y[i] for i in ids],p[ids],.5) if ids else None,
            'note':'Counterfactual stress control; original labels reused, NOT newly annotated accuracy. Router held fixed.'}
    output=project_path(settings['report']) if not challenge else project_path(settings['scores_directory'])/'challenge_evaluation.json'
    write_json(output,result)
    write_json(output.with_suffix('.predictions.json'),[
        {'id':r['id'],'group':r['group'],'source':r['source'],'label':r['label'],
         'score':float(p[i]),'isolated_score':float(q[i]),'bucket':row_bucket(r),
         'context_decision_effect': 'helped' if (p[i]>=.5)==bool(r['label']) and (q[i]>=.5)!=bool(r['label'])
             else 'hurt' if (p[i]>=.5)!=bool(r['label']) and (q[i]>=.5)==bool(r['label']) else 'same_correctness'}
        for i,r in enumerate(rows)])
    print(json.dumps({'test':result['test'],'policy':policy['policy']['name'],'output':str(output)},indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['score','fit','evaluate'])
    parser.add_argument('--settings',default='configs/adaptive_experiment.json')
    parser.add_argument('--split',choices=['val','test'],default='val')
    parser.add_argument('--challenge')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args(); settings=read_settings(args.settings)
    if args.action=='score': export_scores(settings,args.split,args.device,args.challenge)
    elif args.action=='fit':
        if args.challenge or args.split != 'val': parser.error('Fit accepts validation only')
        fit(settings)
    else: evaluate(settings,args.challenge)


if __name__=='__main__': main()
