'use client';

import { useEffect, useState } from 'react';
import { api, pct } from '../lib/api';

function Table({ headers, rows }) {
  return (
    <div className="table-wrap">
      <table>
        <thead><tr>{headers.map((h) => <th key={h}>{h}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

const METHOD_NAMES = { lexicon_only: 'Word list only', tfidf_logreg: 'TF-IDF + logistic regression', trained_model: 'This model' };
const num = (x) => <span className="num">{x === undefined || x === null ? '–' : x.toFixed(3)}</span>;

export default function ResultsView() {
  const [bench, setBench] = useState(null);
  const [base, setBase] = useState(null);

  useEffect(() => {
    api('/api/benchmarks').then(setBench);
    api('/api/baselines').then(setBase);
  }, []);

  if (!bench) return <div className="spinner">Loading…</div>;
  if (!bench.ok) return <div className="empty">{bench.error}</div>;
  const b = bench.data;

  return (
    <div className="stack">
      <section className="panel">
        <h2>Trained models</h2>
        <Table headers={['Stage', 'Checkpoint', 'Data', 'Result']}
               rows={b.training_stages.map((s) => [String(s.stage), s.checkpoint, s.data,
                 <span key="m" className={s.metric.startsWith('not') ? 'dim' : ''}>{s.metric}</span>])} />
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Evaluation</h2>
          {b.evaluated && <span className="hint">measured {b.evaluated_at}</span>}
        </div>
        {!b.evaluated && <p className="hint">{b.message}</p>}
        {b.evaluated && (
          <>
            <p className="hint">The held-out test set is real data the model never trained on. The other rows are small hand-written checks — read them with their n.</p>
            <Table headers={['Check', 'What it measures', 'Result', 'Target', '']}
                   rows={b.slices.map((s) => [s.name, s.description, <span key="v" className="num">{s.value}</span>, s.gate,
                     <span key="s" className={s.status === 'FAILED' ? 'fail' : s.status === 'PASSED' ? 'pass' : 'dim'}>{s.status.toLowerCase()}</span>])} />
          </>
        )}
      </section>

      <section className="panel">
        <h2>Compared with simple methods</h2>
        {!base && <div className="spinner">Loading…</div>}
        {base && (!base.ok || !base.data.available) && <p className="hint">{base.ok ? base.data.message : base.error}</p>}
        {base && base.ok && base.data.available && (
          <>
            <p className="hint">Held-out test set, n = {base.data.n_test.toLocaleString()}, {pct(base.data.test_toxic_rate)} toxic. Higher is better.</p>
            <Table headers={['Method', 'F1', 'Precision', 'Recall', 'ROC-AUC']}
                   rows={Object.entries(base.data.methods).map(([k, m]) => [METHOD_NAMES[k] || k, num(m.f1), num(m.precision), num(m.recall), num(m.roc_auc)])} />
            {base.data.context_ablation ? (
              <p>
                Context check: on {base.data.context_ablation.n_pairs} message pairs where only the earlier message differs, the model got both
                right in {pct(base.data.context_ablation.contrastive_flip_with_context)} of pairs (impossible without context). A threatening
                earlier message raised the score in {base.data.context_ablation.pairs_where_toxic_parent_raises_score} of {base.data.context_ablation.n_pairs} pairs.
              </p>
            ) : (
              <p className="hint">Train a model and rerun python run_pipeline.py --baselines to add it to this table.</p>
            )}
          </>
        )}
      </section>
    </div>
  );
}
