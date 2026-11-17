"""Audit saved scores and build paper assets. Never trains or loads neural models."""
from pathlib import Path
import collections
import hashlib
import json
import math
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper_submission'
OUT.mkdir(exist_ok=True)
R = ROOT / 'reports/adaptive_v1'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

policy = read(R/'policy.json')
test = read(R/'evaluation.json')
challenge = read(R/'challenge_evaluation.json')
checks=[]
def check(name, condition):
    checks.append({'check':name,'passed':bool(condition)})
    if not condition: raise ValueError(name)

archive=ROOT/'reports/adaptive_v1_results_for_review.zip'
with zipfile.ZipFile(archive) as z:
    check('ZIP CRC', z.testzip() is None)
    manifest=read(R/'return_manifest.json')
    for path,sha in manifest['sha256'].items():
        check('Manifest '+path, digest(ROOT/path)==sha)
        check('Archive bytes '+path, hashlib.sha256(z.read(path.replace('\\','/'))).hexdigest()==sha)
check('Complete expected evidence',not manifest['missing_expected'])
check('Frozen validation score identity',digest(R/'val_scores.json')==policy['validation_scores_sha256'])
check('Prepared mixed identity',digest(ROOT/'data/processed/contextual_clean_v1.jsonl')==policy['prepared_sha256'])
check('Fit/check IDs disjoint',not set(policy['fit_ids'])&set(policy['selection_ids']))

def scored(bundle,field):
    vals=[]
    for row in bundle['records']:
        branch=policy['policy']['branches'].get(row['bucket'],policy['policy']['fallback'])
        p=sum(w*s for w,s in zip(branch['weights'],row[field])); t=branch['threshold']
        vals.append(.5*p/max(t,1e-12) if p<t else .5+.5*(p-t)/max(1-t,1e-12))
    return np.array(vals)

def metric_check(name,rows,p,reported,t=.5):
    y=np.array([r['label'] for r in rows]); pred=p>=t
    for key,value in [('f1',f1_score(y,pred,zero_division=0)),
                      ('precision',precision_score(y,pred,zero_division=0)),
                      ('recall',recall_score(y,pred,zero_division=0)),
                      ('average_precision',average_precision_score(y,p))]:
        check(name+' '+key, math.isclose(value,reported[key],abs_tol=1e-10))
    check(name+' confusion',confusion_matrix(y,pred,labels=[0,1]).tolist()==reported['confusion_matrix'])

timings={}
for split,report in [('test',test),('challenge',challenge)]:
    bundle=read(R/(split+'_scores.json')); rows=bundle['records']
    check(split+' expert identities',bundle['identities']==policy['identities'])
    p=scored(bundle,'scores'); q=scored(bundle,'isolated_scores')
    metric_check(split,rows,p,report['test'])
    output=read(R/('evaluation.predictions.json' if split=='test' else 'challenge_evaluation.predictions.json'))
    lookup={r['id']:r for r in output}
    check(split+' prediction IDs',len(lookup)==len(rows) and set(lookup)=={r['id'] for r in rows})
    check(split+' prediction parity',all(abs(lookup[r['id']]['score']-p[i])<1e-10 and abs(lookup[r['id']]['isolated_score']-q[i])<1e-10 for i,r in enumerate(rows)))
    for source in report['details']['by_source']:
        ids=[i for i,r in enumerate(rows) if r['source']==source]
        metric_check(split+'/'+source,[rows[i] for i in ids],p[ids],report['details']['by_source'][source])
    for j,name in enumerate(bundle['experts']):
        metric_check(split+'/'+name,rows,np.array([r['scores'][j] for r in rows]),report['expert_baselines'][name]['overall'],policy['thresholds'][name])
    ids=[i for i,r in enumerate(rows) if r['has_context']]
    metric_check(split+'/context',[rows[i] for i in ids],p[ids],report['details']['contextual'])
    metric_check(split+'/no-context',[rows[i] for i in ids],q[ids],report['without_context_fixed_route']['contextual'])
    timings[split]=bundle['seconds']
timings['val']=read(R/'val_scores.json')['seconds']
tree=ET.parse(R/'tests.xml'); suites=list(tree.getroot().iter('testsuite'))
tests={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
check('Returned test suite',tests=={'tests':96,'failures':0,'errors':0,'skipped':1})
http=read(R/'http_verification.json')
check('Returned HTTP check',http['prediction_status']==200 and http['benchmark_status']==200 and http['health']['predictor_type']=='AdaptivePredictor')

data=[json.loads(l) for l in (ROOT/'data/processed/contextual_clean_v1.jsonl').read_text(encoding='utf-8').splitlines()]
counts=collections.defaultdict(lambda:[0,0])
for row in data:
    counts[(row['source'],row['split'])][0]+=1; counts[(row['source'],row['split'])][1]+=row['label']
audit={'basis':'Current local archive and matching local source artifacts; no model inference or training.',
       'archive':str(archive),'archive_sha256':digest(archive),'checks':checks,
       'tests':tests,'recorded_scoring_seconds':timings,
       'timing_conflict':'run_notes.log says test ~362 seconds and challenge ~8 seconds; score JSON records test 14481.516 and challenge 13.968 seconds. Do not claim ~6 minute test scoring without reconciliation.',
       'note':'Bootstrap intervals are retained from the verified, hashed reports; point metrics and policy/prediction parity were independently recomputed. Uploaded Library ZIP identity was not independently byte-compared.'}
(OUT/'evidence_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')

def fmt(v): return f'{v:.4f}'
def tex_table(headers,rows,caption,label,columns=None):
    cols=columns or ('l'+'r'*(len(headers)-1))
    return '\\begin{table}[htbp]\n\\centering\\small\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'+'\n'.join(' & '.join(map(str,r))+' \\\\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n\\caption{'+caption+'}\\label{'+label+'}\n\\end{table}\n'

tables=[]
names={'cga_wiki':'English CGA','cultural_safety_telugu':'Telugu safety','dravidian_codemix_tamil':'Tamil code-mixed'}
rows=[]
for source in names:
    vals=[counts[(source,split)] for split in ['train','val','test']]
    rows.append([names[source]]+[f'{n:,} ({p:,})' for n,p in vals])
tables.append(tex_table(['Source','Train','Validation','Test'],rows,'Cleaned examples, with positive counts in parentheses. These are project-specific grouped splits, not official benchmark splits.','tab:data'))
rows=[]
for source,name in [('overall','Combined'),*names.items()]:
    a=test['test'] if source=='overall' else test['details']['by_source'][source]
    get=lambda e: test['expert_baselines'][e]['overall'] if source=='overall' else test['expert_baselines'][e]['by_source'][source]
    rows.append([name,fmt(get('english_ensemble')['f1']),fmt(get('mixed_neural')['f1']),fmt(get('mixed_ensemble')['f1']),fmt(a['f1']),fmt(a['average_precision'])])
tables.append(tex_table(['Population','English ens.','Mixed neural','Mixed ens.','Adaptive','Adaptive AP'],rows,'Positive-class F1 for all model columns, plus adaptive AP. All entries are from the same FP32 adaptive score export. Combined metrics mix source label definitions.','tab:main'))
rows=[]
for key,name in [('mixed_neural','Mixed neural'),('mixed_ensemble','Mixed ensemble'),('english_ensemble','English ensemble'),('adaptive','Adaptive')]:
    m=challenge['test'] if key=='adaptive' else challenge['expert_baselines'][key]['overall']
    cm=m['confusion_matrix'];rows.append([name,fmt(m['f1']),fmt(m['average_precision']),fmt(m['precision']),fmt(m['recall']),cm[0][1],cm[1][0]])
tables.append(tex_table(['Model','F1','AP','Precision','Recall','FP','FN'],rows,'Authored Telugu/Romanized development challenge: 51 targets, 19 positive. This is not an independent representative benchmark.','tab:challenge'))
rows=[]
for scope,name in [('overall','Combined'),*names.items()]:
    v=test['paired_vs_experts']['mixed_ensemble'] if scope=='overall' else test['paired_by_source'][scope]['mixed_ensemble']
    low,high=v['percentile_95_interval'];rows.append([name,f'{100*v["f1_difference"]:+.3f}',f'[{100*low:+.3f}, {100*high:+.3f}]'])
tables.append(tex_table(['Population',r'$\Delta$F1 (pp)','95\% interval (pp)'],rows,'Adaptive minus mixed ensemble; 1,000 paired connected-group bootstrap resamples, seed 2026. All intervals include zero. These are exploratory unadjusted intervals.','tab:delta'))
rows=[]
for bucket,branch in policy['policy']['branches'].items():
    rows.append([bucket.replace('_',r'\_'),*[fmt(w) for w in branch['weights']],f'{branch["threshold"]:.6f}'])
f=policy['policy']['fallback'];rows.append(['Fallback',*[fmt(w) for w in f['weights']],f'{f["threshold"]:.6f}'])
tables.append(tex_table(['Script group','English ens.','Mixed ens.','Mixed neural','Raw threshold'],rows,'Frozen validation-selected policy. The script group is not a language label.','tab:policy'))
(OUT/'results_tables.tex').write_text('\n'.join(tables),encoding='utf-8')

# Human-readable independent project overview. It is not a compiled LaTeX manuscript.
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='CoverTitle',fontSize=26,leading=31,textColor=colors.HexColor('#17324d'),spaceAfter=18))
styles['BodyText'].fontSize=10;styles['BodyText'].leading=15;styles['BodyText'].spaceAfter=9
styles['Heading1'].textColor=colors.HexColor('#17324d');styles['Heading1'].spaceAfter=14
story=[]
def p(text,style='BodyText'):story.append(Paragraph(text,styles[style]))
def table(headers,rows,widths):
    cells=[[Paragraph(str(x),styles['BodyText']) for x in row] for row in [headers]+rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3ebf4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#61788f')),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7fa')])]))
    story.append(t);story.append(Spacer(1,12))
def page(title):story.append(PageBreak());p(title,'Heading1')
def footer(c,doc):
    c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#61788f'));c.drawString(42,26,'Context-aware toxicity | Evidence-based project overview | 17 November 2026');c.drawRightString(A4[0]-42,26,str(doc.page))

p('Context-aware toxicity classification','CoverTitle')
p('Architecture, empirical contribution, verified results and submission readiness','Heading2')
p('The project classifies a target message using its genuine ancestor chain when available. It combines lexical models with an IndicBERT-v2 encoder and a hierarchical conversation Transformer, then evaluates a conservative script-based combination of saved experts.')
p('<b>Main finding.</b> The adaptive policy is a tradeoff, not a clear overall winner. Combined F1 is 0.7299 versus 0.7284 for the mixed ensemble; the paired interval includes zero. English-only specialization performs best on English, while the mixed ensemble remains stronger on the small Telugu development challenge.')
table(['Evidence','Observed status'],[['Software','95 passed; 1 optional legacy transliteration test skipped'],['Serving','Returned localhost HTTP health, benchmark and prediction checks passed; CPU demo'],['Validation choice','script_soft_blend; fitting 6,978 rows, selection 4,798 rows'],['Remaining work','Author/venue information, data attribution and rights, compilation, timing discrepancy and final claim review']],[135,375])
p('This overview uses the current local results archive and matching local files. All 64 manifest entries matched; saved-score policy parity and key metrics were recomputed. No new model training or inference was run to prepare it. The companion LaTeX is an editable manuscript draft, not a claim of journal acceptance.')
page('1. Task, data and preparation')
p('The binary target differs by source: English personal attacks, mapped Telugu prompt-safety categories and Tamil code-mixed offensiveness. These are related but not identical labels. All naturally contextual main-test examples come from English CGA; the Indic training sources are isolated inputs.')
table(['Source','Train','Validation','Test'],[[names[s],*[f'{counts[(s,k)][0]:,}' for k in ['train','val','test']]] for s in names],[170,110,115,115])
p('The cleaned pool contains 78,232 targets: 54,578 training, 11,776 validation and 11,878 test. Cleaning removed 566 same-chain/label duplicates and 41 contradictory isolated-label examples from the prior pool. Unicode NFC, control-character handling and whitespace normalization preserve script and lexical cues.')
p('Reply links are followed root-to-target. Header nodes are excluded; malformed CGA ancestry is excluded with audit counts rather than invented. Conversation/article and shared-message grouping reduce leakage. These project splits are not the datasets\' official benchmark splits; no near-duplicate semantic leakage guarantee is established.')
p('The authored challenge has 51 Telugu/Romanized targets, 19 positive, in 45 connected groups. It is development evidence and is not used to fit the adaptive policy. It cannot establish language-wide robustness.')
page('2. Architecture and inference')
d=Drawing(510,220)
boxes=[(0,162,510,42,'Target + actual ancestors; retain latest 11 parents'),(0,102,245,42,'IndicBERT-v2 + LoRA; token pooling'),(265,102,245,42,'Target-only TF-IDF + logistic models'),(0,42,245,42,'Conversation Transformer + pooling'),(265,42,245,42,'Saved English / mixed expert scores'),(0,0,510,28,'Script policy -> weighted score -> fixed decision threshold')]
for x,y,w,h,label in boxes:
    d.add(Rect(x,y,w,h,rx=5,fillColor=colors.HexColor('#e7eef5'),strokeColor=colors.HexColor('#61788f')));d.add(String(x+w/2,y+h/2-3,label,fontName='Helvetica',fontSize=9,textAnchor='middle'))
for x,y1,y2 in [(122,162,144),(387,162,144),(122,102,84),(387,102,84),(122,42,28),(387,42,28)]:d.add(Line(x,y1,x,y2,strokeColor=colors.HexColor('#61788f')))
story.append(d);story.append(Spacer(1,15))
p('All content tokens contribute through chunked mean pooling, with a maximum 128 tokens per chunk including special tokens. Chunk pooling preserves token coverage but not cross-chunk token interactions. Each message vector projects to 192 dimensions. Two conversation Transformer layers use four heads, feed-forward width 384 and dropout 0.1.')
p('The classification head concatenates the projected target, contextualized target and attention-pooled conversation vector. The target skip path helps distinguish a benign reply from a toxic parent, but does not guarantee this behavior. Padding is masked. The model uses at most 12 turns, including the target.')
p('LoRA adapts query/value matrices with rank 8, alpha 16 and dropout 0.1. The English ensemble uses 0.75 character TF-IDF + 0.25 neural score; the mixed ensemble uses 0.50 character TF-IDF + 0.50 neural score. Word TF-IDF was a candidate but has zero selected weight in these saved ensembles.')
page('3. What is contributed, and what is not')
p('<b>Defensible contribution:</b> a reproducible integration and empirical comparison of ancestry-aware encoding, lexical/neural experts, conservative script-dependent blending and grouped context diagnostics across heterogeneous sources.')
p('BERT-family encoders, LoRA, Transformer attention, lexical classifiers and soft voting are established techniques. Neither contextual toxicity detection nor deciding when context is useful is a new research problem. The paper must not claim a first-ever architecture, new attention mechanism or state-of-the-art result.')
p('The router is restricted: ambiguous Latin input may include Romanized Telugu, so English-specialist weight is capped at 0.25. Other script routes use mixed experts; unknown scripts fall back to the original mixed ensemble. Script identification does not identify language or guarantee that the best expert is selected for an individual input.')
p('Candidate policy fitting uses a grouped portion of the existing validation population. The remaining portion selects among candidates using source/context F1 and regression guards. Existing experts had already used the original validation set, so this procedure is not fresh independent validation. Earlier test inspection also makes the final analysis exploratory.')
table(['Scope','Appropriate description'],[['Method','An implemented integration of established components'],['Novelty','Empirical contribution and controlled comparison; priority not established'],['Generalization','Source-specific measured performance, not equal accuracy across languages'],['Parent context','Some ranking evidence, inconclusive F1 advantage and weak control separation']],[125,385])
page('4. Results and context interpretation')
rows=[]
for s,n in names.items():rows.append([n,*[fmt(test['expert_baselines'][e]['by_source'][s]['f1']) for e in ['english_ensemble','mixed_ensemble']],fmt(test['details']['by_source'][s]['f1'])])
rows.append(['Combined',fmt(test['expert_baselines']['english_ensemble']['overall']['f1']),fmt(test['expert_baselines']['mixed_ensemble']['overall']['f1']),fmt(test['test']['f1'])])
table(['Population','English ensemble','Mixed ensemble','Adaptive F1'],rows,[160,120,115,115])
p('Adaptive minus mixed overall F1 is +0.144 percentage points (95% paired interval -0.424 to +0.744). Overall AP decreases from 0.8041 to 0.8007. There are 19 more true positives and 22 more false positives. This does not establish an overall improvement.')
p('On 2,282 contextual English targets, full versus removed context gives F1 0.7813 versus 0.7658: +1.549 points, interval -1.678 to +4.979. AP rises from 0.8560 to 0.8754: +1.945 points, interval +0.616 to +3.132. This supports an exploratory ranking advantage, not an established F1 gain.')
p('Reversed parents slightly outperform original order on their matched subset. Unrelated parents produce nearly identical AP and slightly higher F1 on eligible targets. These original-label controls do not show clear use of correct ancestry semantics. They are sensitivity tests, not newly annotated accuracy.')
p('Depth 8-11 contains only four main-test examples, and no test example has 12+ ancestors. Perfect scores on four examples are not robust depth evidence. No claim of uniform performance at arbitrary history length is supported.')
page('5. Challenge, reproducibility and next actions')
table(['Challenge model','F1','Recall','False negatives'],[[n,fmt(m['f1']),fmt(m['recall']),m['confusion_matrix'][1][0]] for n,m in [('English ensemble',challenge['expert_baselines']['english_ensemble']['overall']),('Mixed ensemble',challenge['expert_baselines']['mixed_ensemble']['overall']),('Adaptive',challenge['test'])]],[180,100,100,130])
p('Adaptive recovers from the English specialist\'s zero-recall failure, but the mixed ensemble is better on this challenge: F1 0.6522 versus 0.6222. Keep the frozen adaptive outcome in the paper. Do not silently switch to a challenge-selected policy and call it validation-selected.')
p('Returned logs record Python 3.10.20, PyTorch 2.12.0+cu130 and an RTX 4060 Laptop GPU with approximately 8 GB VRAM. Evaluation exports report CUDA; the HTTP demo ran on CPU. Tests completed with 95 passes and one skip.')
p('<b>Timing discrepancy:</b> the narrative says about six minutes for test scoring, but the score artifact records 14,481.516 seconds (about 241.36 minutes). Challenge timing also differs: 8 seconds in notes versus 13.968 in JSON. Resolve these before reporting speed; this overview makes no throughput claim.')
p('<b>Before submission:</b> complete author and affiliation information; choose the venue template; verify data citations, licensing and label conversion; compile the LaTeX twice and review all pages; apply the final checklist. New training is not required for an honestly scoped exploratory paper. Stronger novelty, robust Telugu-context or generalization claims require additional independent evidence.')
p('Core references: Zhang et al., ACL 2018 (Conversations Gone Awry); Doddapaneni et al., ACL 2023 (IndicBERT v2); Hu et al., arXiv:2106.09685 (LoRA); Pavlopoulos et al., ACL 2020 (contextual toxicity); Xenos et al., WOAH 2021 (context sensitivity). Full references and repository URLs are supplied in the LaTeX package.')
SimpleDocTemplate(str(OUT/'PROJECT_EXPLAINED.pdf'),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=45,title='Context-aware toxicity: architecture and verified results',author='Project documentation').build(story,onFirstPage=footer,onLaterPages=footer)
print(json.dumps({'checks':len(checks),'failed':sum(not c['passed'] for c in checks),'tests':tests,'recorded_seconds':timings,'output':str(OUT)},indent=2))
