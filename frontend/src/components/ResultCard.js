import { DEFAULT_LABELS, VERDICT_TEXT, pct, toxicScore, verdictOf } from '../lib/api';

function segmentLabel(seg, nTurns) {
  if (seg === 'target') return 'Checked';
  if (seg === 'topic') return 'Topic';
  const n = Number(seg.split('-')[1]);
  return n === nTurns ? 'Just before' : `Earlier ${n}`;
}

/** One row per message (topic / earlier messages / checked message), shaded by attention. */
function Highlight({ tokens }) {
  const max = Math.max(...tokens.map((t) => t.attention), 1e-6);
  const nTurns = Math.max(0, ...tokens.filter((t) => (t.segment || '').startsWith('turn-')).map((t) => Number(t.segment.split('-')[1])));
  const rows = [];
  tokens.forEach((t, i) => {
    const seg = t.segment || (t.is_target ? 'target' : 'turn-1');
    let row = rows[rows.length - 1];
    if (!row || row.seg !== seg) {
      row = { seg, parts: [] };
      rows.push(row);
    } else if (!t.continuation) {
      row.parts.push(' ');
    }
    const rgb = t.is_target ? 'var(--hl-target)' : 'var(--hl-context)';
    const alpha = (0.06 + 0.55 * (t.attention / max)).toFixed(2);
    row.parts.push(
      <span key={i} className="tok" title={`${(t.attention * 100).toFixed(1)}% attention`}
            style={{ background: `rgba(${rgb}, ${alpha})` }}>{t.token}</span>,
    );
  });
  return (
    <div className="highlight">
      {rows.map((r, i) => (
        <div key={i} className="hl-row">
          <span className="hl-label">{segmentLabel(r.seg, nTurns)}</span>
          <span className="hl-text">{r.parts}</span>
        </div>
      ))}
    </div>
  );
}

/** Red bar only when this label alone would flag the message. */
function isOver(r, k, val) {
  const line = r.flag_lines ? r.flag_lines[k] : r.thresholds && r.thresholds[k];
  return line !== null && line !== undefined && val >= line;
}

export default function ResultCard({ result: r, caption, labels = DEFAULT_LABELS }) {
  const v = verdictOf(r);
  if (v === 'error') {
    return (
      <article className="result" data-verdict="error">
        {caption && <div className="result-caption">{caption}</div>}
        <div className="verdict"><span className="verdict-label">{VERDICT_TEXT.error}</span></div>
        <p className="reason">{r.error}</p>
      </article>
    );
  }
  const threshold = r.thresholds ? r.thresholds.toxic : 0.5;
  const score = toxicScore(r);
  const lean = v === 'review' ? ` · leaning ${r.is_toxic ? 'flagged' : 'allowed'}` : '';
  const a = r.attention_analysis;
  const lex = r.lexicon || {};

  return (
    <article className="result" data-verdict={v}>
      {caption && <div className="result-caption">{caption}</div>}
      <div className="verdict">
        <span className="verdict-label">{VERDICT_TEXT[v]}</span>
        <span className="verdict-score">harm score {pct(score)}{lean}</span>
      </div>
      <p className="reason">{r.reason || r.explanation}</p>
      <div>
        <div className="meter" role="img" aria-label={`Harm score ${pct(score)}, decision line ${pct(threshold)}`}>
          <div className="meter-fill" style={{ width: `${Math.max(2, score * 100)}%` }} />
          <div className="meter-threshold" style={{ left: `${threshold * 100}%` }} />
        </div>
        <div className="meter-legend"><span>0%</span><span>decision line {pct(threshold)}</span><span>100%</span></div>
      </div>

      {r.decided_by === 'model' && r.categories && (
        <ul className="categories" aria-label="Scores by type of harm">
          {Object.entries(r.categories).map(([k, val]) => (
            <li key={k} className="category" data-over={String(isOver(r, k, val))}>
              <span className="category-name">{labels[k] || k}</span>
              <span className="category-bar"><span className="category-fill" style={{ width: `${(val || 0) * 100}%`, display: 'block' }} /></span>
              <span className="category-value">{pct(val)}</span>
            </li>
          ))}
        </ul>
      )}
      {r.decided_by === 'word list' && (
        <p className="note">Decided by the word list (matched: {(lex.severe || []).join(', ')}). The model was not needed.</p>
      )}
      {r.untrained && <p className="note">⚠ Untrained model — this result is random.</p>}

      <details className="more">
        <summary>Details</summary>
        {a && a.tokens && a.tokens.length > 0 && (
          <>
            <p className="hint">Where the model looked (darker = more attention):</p>
            <Highlight tokens={a.tokens} />
          </>
        )}
        <dl className="kv">
          <dt>Decided by</dt><dd>{r.decided_by === 'model' ? `model${a ? ` (${a.num_heads}-head attention)` : ''}` : 'word list'}</dd>
          <dt>Earlier messages used</dt><dd>{r.history && r.history.length ? r.history.length : 'none'}</dd>
          <dt>Text after clean-up</dt><dd>{r.cleaned_text || '–'}</dd>
          <dt>Words noted</dt><dd>{[...(lex.severe || []), ...(lex.mild || [])].join(', ') || 'none'}</dd>
          <dt>Time</dt><dd>{r.latency_ms} ms</dd>
        </dl>
      </details>
    </article>
  );
}
