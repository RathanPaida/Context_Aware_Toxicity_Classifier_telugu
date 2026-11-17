"""Conservative script-aware expert selection, fitted without test/challenge labels."""
import unicodedata

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .statistics import connected_groups
from .training import fit_threshold, metrics

EXPERTS = ('english_ensemble', 'mixed_ensemble', 'mixed_neural')
FALLBACK = [0., 1., 0.]
POLICY_VERSION = 'script_soft_vote_v1'


def script_bucket(text, history=None):
    """Script evidence across target+ancestors; Latin is NOT an English label."""
    scripts = set()
    for c in ' '.join([text]+list(history or [])):
        if not c.isalpha():
            continue
        name = unicodedata.name(c, '')
        scripts.add('telugu' if 'TELUGU' in name else 'tamil' if 'TAMIL' in name else
                    'latin' if 'LATIN' in name else 'other')
    if len(scripts) > 1:
        return 'mixed_script'
    return {'latin':'latin_ambiguous', 'telugu':'telugu_script', 'tamil':'tamil_script'}.get(
        next(iter(scripts), ''), 'other_or_empty')


def row_bucket(row):
    return script_bucket(row['texts'][-1], row['texts'][:-1])


def margin_score(probabilities, threshold):
    """Map a fixed decision boundary to 0.5; this is NOT probability calibration."""
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not 0 <= threshold <= 1:
        raise ValueError('Invalid scores/threshold')
    return np.where(p < threshold, .5*p/max(threshold, 1e-12),
                    .5+.5*(p-threshold)/max(1-threshold, 1e-12))


def population_objective(rows, p, threshold=.5):
    """Equal source F1 plus contextual F1; prevalence-heavy sources cannot dominate."""
    terms = []
    for source in sorted({r['source'] for r in rows}):
        ids = [i for i,r in enumerate(rows) if r['source'] == source]
        if len({rows[i]['label'] for i in ids}) == 2:
            terms.append(metrics([rows[i]['label'] for i in ids], np.asarray(p)[ids], threshold)['f1'])
    ids = [i for i,r in enumerate(rows) if len(r['texts']) > 1]
    if ids and len({rows[i]['label'] for i in ids}) == 2:
        terms.append(metrics([rows[i]['label'] for i in ids], np.asarray(p)[ids], threshold)['f1'])
    if not terms:
        raise ValueError('Selection requires at least one two-class population')
    return float(np.mean(terms))


def weight_grid(english_cap=0., hard=False):
    candidates = []
    for e in range(5):
        for m in range(5-e):
            w = [e/4, m/4, (4-e-m)/4]
            if w[0] <= english_cap and (not hard or sum(x > 0 for x in w) == 1):
                candidates.append(w)
    # Ties prefer the established mixed ensemble, then fewer experts.
    return sorted(candidates, key=lambda w:(w != FALLBACK, sum(x > 0 for x in w), w[0]))


def fitted_branch(rows, matrix, weights):
    p = np.asarray(matrix)@np.asarray(weights)
    threshold = fit_threshold([r['label'] for r in rows], p)
    return {'weights':weights, 'threshold':threshold}


def supported(rows, min_examples=100, min_class=15, min_groups=15):
    return (len(rows) >= min_examples and
            min(sum(r['label']==v for r in rows) for v in (0,1)) >= min_class and
            len(set(connected_groups(rows))) >= min_groups)


def apply_policy(policy, rows, matrix, fixed_buckets=None):
    validate_policy(policy)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (len(rows), len(EXPERTS)) or not np.isfinite(matrix).all() or ((matrix<0)|(matrix>1)).any():
        raise ValueError('Expected finite aligned expert probabilities')
    buckets = fixed_buckets or [row_bucket(r) for r in rows]
    if len(buckets) != len(rows):
        raise ValueError('Routing bucket count mismatch')
    values = np.zeros(len(rows))
    for i,bucket in enumerate(buckets):
        branch = policy['branches'].get(bucket, policy['fallback'])
        weights = np.asarray(branch['weights'])
        if weights.shape != (3,) or np.any(weights < 0) or not np.isclose(weights.sum(),1):
            raise ValueError('Invalid expert weights')
        values[i] = margin_score(matrix[i]@weights, branch['threshold'])
    return values


def validate_policy(policy):
    """Reject malformed artifacts and policies outside the declared routing limits."""
    allowed = {'latin_ambiguous', 'telugu_script', 'tamil_script', 'mixed_script'}
    if not set(policy['branches']).issubset(allowed):
        raise ValueError('Unsupported routing bucket')
    for bucket, branch in [('fallback', policy['fallback']), *policy['branches'].items()]:
        weights = np.asarray(branch['weights'], dtype=float)
        if (weights.shape != (3,) or not np.isfinite(weights).all() or
                (weights < 0).any() or not np.isclose(weights.sum(), 1)):
            raise ValueError('Invalid expert weights')
        cap = .25 if bucket == 'latin_ambiguous' else 0.
        if weights[0] > cap or (bucket == 'fallback' and not np.array_equal(weights, FALLBACK)):
            raise ValueError('Policy violates conservative routing limits')
        margin_score([0., 1.], branch['threshold'])


def partition_validation(rows, seed=2026):
    if not rows or any(r.get('split') != 'val' for r in rows):
        raise ValueError('Fit/select accepts validation only; never test or challenge')
    groups = connected_groups(rows)
    fit, select = next(GroupShuffleSplit(n_splits=1, test_size=.4, random_state=seed).split(rows, groups=groups))
    for ids in (fit, select):
        for source in sorted({r['source'] for r in rows}):
            if {rows[i]['label'] for i in ids if rows[i]['source']==source} != {0,1}:
                raise ValueError('Group split lacks both classes for a source; collect more validation groups')
    return fit, select


def fit_policies(rows, matrix, mixed_threshold):
    matrix = np.asarray(matrix, dtype=float)
    fit_ids, select_ids = partition_validation(rows)
    fit = [rows[i] for i in fit_ids]
    select = [rows[i] for i in select_ids]
    fm, sm = matrix[fit_ids], matrix[select_ids]
    fallback = {'weights':FALLBACK, 'threshold':mixed_threshold}
    candidates = [{'name':'mixed_existing', 'fallback':fallback, 'branches':{}}]
    buckets = [row_bucket(r) for r in fit]
    # A uniform mixed-expert blend never uses the English specialist.
    for w in weight_grid():
        branch = fitted_branch(fit, fm, w)
        candidates.append({'name':'mixed_global_'+str(w), 'fallback':fallback,
                           'branches':{b:branch for b in sorted(set(buckets)) if b != 'other_or_empty'}})
    for hard in (True,False):
        branches = {}
        for bucket in sorted(set(buckets)):
            ids = [i for i,b in enumerate(buckets) if b == bucket]
            subset = [fit[i] for i in ids]
            if bucket == 'other_or_empty' or not supported(subset):
                continue
            # Unknown Latin can be Romanized Telugu/Tamil. Never route it wholly to English.
            cap = .25 if bucket == 'latin_ambiguous' and not hard else 0.
            best, value = None, -1.
            for w in weight_grid(cap, hard):
                branch = fitted_branch(subset, fm[ids], w)
                p = margin_score(fm[ids]@w, branch['threshold'])
                score = population_objective(subset,p)
                if score > value+1e-12:
                    best, value = branch, score
            branches[bucket] = best
        candidates.append({'name':'script_hard_router' if hard else 'script_soft_blend',
                           'fallback':fallback, 'branches':branches})
    reference = apply_policy(candidates[0], select, sm)
    def subset_f1(ids,p):
        return metrics([select[i]['label'] for i in ids],p[ids],.5)['f1']
    guards = {source:[i for i,r in enumerate(select) if r['source']==source] for source in sorted({r['source'] for r in select})}
    guards['contextual'] = [i for i,r in enumerate(select) if len(r['texts']) > 1]
    leaderboard = []
    for policy in candidates:
        p = apply_policy(policy,select,sm)
        deltas = {k:subset_f1(ids,p)-subset_f1(ids,reference) for k,ids in guards.items()
                  if ids and len({select[i]['label'] for i in ids}) == 2}
        leaderboard.append({'policy':policy,'selection_objective':population_objective(select,p),
                            'group_f1_differences':deltas, 'eligible':all(v >= -.01 for v in deltas.values())})
    winner = leaderboard[0]
    # Require a small predeclared gain and no >1-point source/context F1 regression.
    for candidate in leaderboard[1:]:
        if candidate['eligible'] and candidate['selection_objective'] > winner['selection_objective']+.005:
            winner = candidate
    return {'version':POLICY_VERSION, 'policy':winner['policy'], 'leaderboard':leaderboard,
            'fit_ids':[r['id'] for r in fit], 'selection_ids':[r['id'] for r in select],
            'selection_rule':'Equal-source and contextual F1; require >0.005 improvement and no >0.01 subgroup F1 loss; retain mixed baseline otherwise.',
            'limitations':'Existing experts already used original validation for selection. This partition separates new policy fitting/checking only; it does not create an independent validation benchmark. Script is not language; Romanized Telugu remains ambiguous. No test/challenge labels used.'}
