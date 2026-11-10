// Header with model status + the warning banner (offline / no model / untrained).

export function statusFromHealth(health) {
  if (!health) return { state: 'warn', text: 'Connecting…', banner: null };
  if (!health.ok) {
    return { state: 'error', text: 'Server offline', banner: { kind: 'error', text: 'Cannot reach the server. Start it with: python app/server.py' } };
  }
  const h = health.data;
  if (h.ready && h.trained) {
    const when = h.trained_at ? ` · trained ${h.trained_at.slice(0, 10)}` : '';
    return { state: 'ready', text: `Model ready · ${h.release_source || h.stage || 'model'}${when}`, banner: null };
  }
  if (h.ready) {
    return { state: 'error', text: 'Untrained model', banner: { kind: 'error', text: 'Demo mode: the model is untrained (ALLOW_UNTRAINED=1), so its predictions are random. Do not use these results.' } };
  }
  return { state: 'warn', text: 'No trained model', banner: { kind: 'warn', text: 'No trained model found. Only the word list works until you train one: python run_pipeline.py --all (see docs/TRAINING_GUIDE.md), then restart the server.' } };
}

export default function StatusBar({ health }) {
  const s = statusFromHealth(health);
  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="brand-name">Indic Shield</span>
          <span className="brand-tag">Toxicity checker for Telugu &amp; code-mixed chat — it reads the earlier messages too</span>
        </div>
        <span className="status" data-state={s.state}>{s.text}</span>
      </header>
      {s.banner && <div className="banner" data-kind={s.banner.kind}>{s.banner.text}</div>}
    </>
  );
}
