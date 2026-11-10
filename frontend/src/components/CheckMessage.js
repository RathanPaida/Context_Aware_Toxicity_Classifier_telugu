'use client';

import { useState } from 'react';
import ResultCard from './ResultCard';
import { compareSentence, moderate } from '../lib/api';

export default function CheckMessage({ scenarios, labels }) {
  const [context, setContext] = useState(['']);
  const [target, setTarget] = useState('');
  const [topic, setTopic] = useState('');
  const [compare, setCompare] = useState(true);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null); // {withCtx, without?}

  async function run(ctx = context, tgt = target, tpc = topic) {
    const text = tgt.trim();
    if (!text) return;
    const history = ctx.map((c) => c.trim()).filter(Boolean);
    setBusy(true);
    setResult(null);
    const withCtx = await moderate(text, history, tpc.trim());
    const without = history.length && compare ? await moderate(text, [], tpc.trim()) : null;
    setResult({ withCtx, without });
    setBusy(false);
  }

  function loadExample(id) {
    const sc = scenarios.find((s) => s.id === id);
    if (!sc) return;
    const ctx = sc.turns.slice(0, -1).map((t) => t.text);
    const tgt = sc.turns[sc.turns.length - 1].text;
    setContext(ctx.length ? ctx : ['']);
    setTarget(tgt);
    setTopic(sc.topic || '');
    run(ctx, tgt, sc.topic || '');
  }

  return (
    <div className="grid-2">
      <form className="panel" noValidate onSubmit={(e) => { e.preventDefault(); run(); }}>
        <div>
          <h2>Check a message</h2>
          <p className="hint">The same words can be friendly or threatening depending on what was said before, so add the earlier messages when you have them.</p>
        </div>

        <fieldset>
          <legend>Earlier messages <span className="optional">optional · oldest first</span></legend>
          <ol className="context-list">
            {context.map((c, i) => (
              <li key={i} className="context-item">
                <span className="context-num">{i + 1}.</span>
                <textarea rows={2} value={c} placeholder="What someone said earlier…" aria-label="Earlier message"
                          onChange={(e) => setContext(context.map((x, j) => (j === i ? e.target.value : x)))} />
                <button type="button" className="btn btn-icon" aria-label="Remove this message"
                        onClick={() => setContext(context.filter((_, j) => j !== i))}>✕</button>
              </li>
            ))}
          </ol>
          <div><button type="button" className="btn btn-small" onClick={() => setContext([...context, ''])}>+ Add earlier message</button></div>
        </fieldset>

        <div className="field">
          <label htmlFor="target">Message to check</label>
          <textarea id="target" rows={3} value={target} placeholder="e.g. Champesav po!"
                    onChange={(e) => setTarget(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) run(); }} />
        </div>

        <div className="field">
          <label htmlFor="topic">Topic <span className="optional">optional</span></label>
          <input id="topic" type="text" value={topic} placeholder="e.g. Cricket, Politics, Movies" onChange={(e) => setTopic(e.target.value)} />
        </div>

        <div className="actions">
          <button type="submit" className="btn btn-primary" disabled={busy}>Check message</button>
          <label className="check">
            <input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} /> Also check without the earlier messages
          </label>
        </div>

        <div className="examples">
          <label htmlFor="example-select">Or try an example:</label>
          <select id="example-select" value="" onChange={(e) => loadExample(e.target.value)}>
            <option value="">Choose an example…</option>
            {scenarios.map((s) => <option key={s.id} value={s.id}>{s.title}</option>)}
          </select>
        </div>
      </form>

      <div aria-live="polite">
        {busy && <div className="spinner">Checking…</div>}
        {!busy && !result && <div className="empty">The result will appear here.</div>}
        {!busy && result && !result.without && <ResultCard result={result.withCtx} labels={labels} />}
        {!busy && result && result.without && (
          <div className="stack">
            <div className="compare">
              <ResultCard result={result.withCtx} caption="With earlier messages" labels={labels} />
              <ResultCard result={result.without} caption="Without earlier messages" labels={labels} />
            </div>
            {!result.withCtx.error && !result.without.error && (
              <p className="compare-summary">{compareSentence(result.withCtx, result.without)}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
