'use client';

import { useState, useEffect, useRef } from 'react';

export default function IndicShieldApp() {
  const [activeTab, setActiveTab] = useState('simulator'); // 'simulator' | 'deepdive' | 'benchmarks'
  const [backendStatus, setBackendStatus] = useState({ online: false, mode: 'Connecting...', heads: null });
  const [presets, setPresets] = useState([]);
  const [benchmarks, setBenchmarks] = useState(null);

  // Chat simulator state
  const [threadId] = useState('live_session_thread');
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [inputTopic, setInputTopic] = useState('Gaming');
  const [isLoading, setIsLoading] = useState(false);
  const [currentResult, setCurrentResult] = useState(null);
  const chatScrollRef = useRef(null);

  // Deep dive state
  const [deepDiveTarget, setDeepDiveTarget] = useState('Champesav po!');
  const [deepDiveParent, setDeepDiveParent] = useState('Ninna match lo 1v4 clutch chusava bro?');
  const [deepDiveTopic, setDeepDiveTopic] = useState('Gaming');
  const [deepDiveResult, setDeepDiveResult] = useState(null);
  const [deepDiveLoading, setDeepDiveLoading] = useState(false);

  // Check backend health on mount
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch('/api/health');
        if (res.ok) {
          const data = await res.json();
          if (data.ready) {
            const name = (data.model_name || 'model').split('/').pop();
            setBackendStatus({ online: true, mode: `${name} · ${data.stage || 'checkpoint'} (${(data.device || '').toUpperCase()})`, heads: data.num_heads });
          } else {
            setBackendStatus({ online: false, mode: 'Backend up — model not trained', heads: null });
          }
        } else {
          setBackendStatus({ online: false, mode: 'Backend offline', heads: null });
        }
      } catch {
        setBackendStatus({ online: false, mode: 'Backend offline', heads: null });
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [messages]);

  // Load scenarios + measured benchmarks from the backend
  useEffect(() => {
    fetch('/api/scenarios')
      .then((r) => (r.ok ? r.json() : []))
      .then((list) => setPresets(list.map((sc) => ({ ...sc, desc: sc.explanation }))))
      .catch(() => setPresets([]));
    fetch('/api/benchmarks')
      .then((r) => (r.ok ? r.json() : null))
      .then(setBenchmarks)
      .catch(() => setBenchmarks(null));
  }, []);

  // API call. On failure an explicit ERROR result is returned — the old version
  // fabricated verdicts with keyword rules and displayed them as model output.
  async function callModerationAPI(text, history = [], topic = '', customThreadId = null) {
    try {
      const res = await fetch('/api/moderate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          thread_id: customThreadId || threadId,
          topic: topic || undefined,
          history,
          update_history: false,
        }),
      });
      if (res.ok) return await res.json();
      const body = await res.json().catch(() => ({}));
      return errorResult(body.detail || `HTTP ${res.status}`);
    } catch (err) {
      return errorResult(`Backend unreachable (${err.message}). Start it with: python app/server.py`);
    }
  }

  function errorResult(message) {
    return {
      action: 'ERROR',
      is_toxic: false,
      toxicity_score: 0,
      stage_routed: 'unavailable',
      latency_ms: 0,
      categories: {},
      explanation: message,
      attention_analysis: null,
    };
  }

  // Load a preset scenario into the thread
  async function loadPreset(preset) {
    setIsLoading(true);
    setInputTopic(preset.topic);
    const newMessages = [];
    let lastResult = null;

    for (let i = 0; i < preset.turns.length; i++) {
      const turn = preset.turns[i];
      const isTargetTurn = i === preset.turns.length - 1;
      const history = newMessages.map((m) => m.text);
      const res = await callModerationAPI(turn.text, history, preset.topic);

      newMessages.push({
        id: `${preset.id}_${i}`,
        speaker: turn.speaker,
        text: turn.text,
        isTarget: isTargetTurn,
        result: res,
      });
      lastResult = res;
    }

    setMessages(newMessages);
    setCurrentResult(lastResult);
    setIsLoading(false);
  }

  // Send a custom message in the chat
  async function handleSendMessage(e) {
    e.preventDefault();
    if (!inputText.trim() || isLoading) return;

    const text = inputText.trim();
    setInputText('');
    setIsLoading(true);

    const history = messages.map((m) => m.text);
    const res = await callModerationAPI(text, history, inputTopic);

    const newMsg = {
      id: `msg_${Date.now()}`,
      speaker: `User ${messages.length + 1}`,
      text: text,
      isTarget: true,
      result: res,
    };

    setMessages((prev) => [...prev, newMsg]);
    setCurrentResult(res);
    setIsLoading(false);
  }

  // Run deep-dive analysis
  async function handleRunDeepDive() {
    if (!deepDiveTarget.trim() || deepDiveLoading) return;
    setDeepDiveLoading(true);

    const [withCtx, withoutCtx] = await Promise.all([
      callModerationAPI(deepDiveTarget, deepDiveParent ? [deepDiveParent] : [], deepDiveTopic, 'deepdive_with_ctx'),
      callModerationAPI(deepDiveTarget, [], deepDiveTopic, 'deepdive_without_ctx'),
    ]);

    setDeepDiveResult({ withCtx, withoutCtx });
    setDeepDiveLoading(false);
  }

  return (
    <div>
      {/* Header */}
      <header className="app-header">
        <div className="logo-group">
          <div className="logo-badge">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="2.2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
          </div>
          <div className="logo-text">
            <h1>Indic Shield AI</h1>
            <span>Context-Aware Toxicity Classifier</span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="nav-tabs">
          <button
            className={`nav-tab-btn ${activeTab === 'simulator' ? 'active' : ''}`}
            onClick={() => setActiveTab('simulator')}
          >
            <span>💬 Thread Simulator</span>
          </button>
          <button
            className={`nav-tab-btn ${activeTab === 'deepdive' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('deepdive');
              if (!deepDiveResult) handleRunDeepDive();
            }}
          >
            <span>🔬 Context Deep-Dive</span>
          </button>
          <button
            className={`nav-tab-btn ${activeTab === 'benchmarks' ? 'active' : ''}`}
            onClick={() => setActiveTab('benchmarks')}
          >
            <span>📊 4-Gate Benchmarks</span>
          </button>
        </div>

        {/* Status Indicators */}
        <div className="header-status-group">
          <div className="status-pill">
            <span className="status-dot"></span>
            <span>{backendStatus.mode}</span>
          </div>
          <div className="arch-pill">
            <span>{backendStatus.heads ? `${backendStatus.heads}-Head Transformer Attention` : 'Transformer Attention'}</span>
          </div>
        </div>
      </header>

      {/* Main Content Layout */}
      <main className="main-layout">
        {/* TAB 1: THREAD SIMULATOR */}
        {activeTab === 'simulator' && (
          <div className="moderation-grid">
            {/* Left Column: Conversational Stream */}
            <div className="glass-card">
              <div className="card-header">
                <span className="card-title">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                  </svg>
                  Conversational Thread Stream
                </span>
                <button
                  className="ghost-btn"
                  onClick={() => {
                    setMessages([]);
                    setCurrentResult(null);
                  }}
                >
                  Clear Thread
                </button>
              </div>

              {/* Preset Scenarios Selector */}
              <div className="preset-bar">
                {presets.length === 0 && <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>No scenarios (backend offline?)</span>}
                {presets.map((preset) => (
                  <button
                    key={preset.id}
                    className="preset-chip"
                    onClick={() => loadPreset(preset)}
                    disabled={isLoading}
                  >
                    {preset.title}
                  </button>
                ))}
              </div>

              {/* Chat Stream Window */}
              <div className="chat-stream-container" ref={chatScrollRef}>
                {messages.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
                    <p style={{ fontSize: '14px', marginBottom: '8px' }}>Thread is empty.</p>
                    <p style={{ fontSize: '12px' }}>Choose a preset above or type a comment below to test real-time moderation.</p>
                  </div>
                ) : (
                  messages.map((m) => {
                    const isFlagged = m.result?.is_toxic;
                    return (
                      <div
                        key={m.id}
                        className={`chat-bubble ${m.isTarget ? 'target-turn' : 'context-turn'}`}
                      >
                        <div className="chat-speaker">
                          <span className={m.isTarget ? 'speaker-target' : 'speaker-context'}>
                            {m.speaker} {m.isTarget ? '(Target Evaluated)' : '(Preceding Context)'}
                          </span>
                          {m.result && (
                            <span className={`verdict-tag ${isFlagged ? 'tag-flagged' : 'tag-allowed'}`}>
                              {m.result.action} ({(m.result.toxicity_score * 100).toFixed(0)}%)
                            </span>
                          )}
                        </div>
                        <div>{m.text}</div>
                      </div>
                    );
                  })
                )}
              </div>

              {/* Input Form */}
              <form onSubmit={handleSendMessage} className="chat-input-bar">
                <input
                  type="text"
                  className="chat-input"
                  placeholder="Type a Telugu / Tanglish comment (e.g. 'Champesav po!')..."
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  disabled={isLoading}
                />
                <button type="submit" className="send-btn" disabled={isLoading || !inputText.trim()}>
                  {isLoading ? 'Evaluating...' : 'Moderate Turn'}
                </button>
              </form>
            </div>

            {/* Right Column: Live Diagnostic Inspector */}
            <div className="glass-card">
              <div className="card-header">
                <span className="card-title">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="16" x2="12" y2="12" />
                    <line x1="12" y1="8" x2="12.01" y2="8" />
                  </svg>
                  8-Head Attention Diagnostic Inspector
                </span>
                <span style={{ fontSize: '11px', color: 'var(--accent-cyan)', fontWeight: 600 }}>
                  8 Heads Active
                </span>
              </div>

              {currentResult ? (
                <>
                  {/* Verdict Hero Banner */}
                  <div
                    className={`verdict-hero ${
                      currentResult.is_toxic ? 'verdict-flagged' : 'verdict-allowed'
                    }`}
                  >
                    <div className="verdict-hero-badge">
                      <span>{currentResult.is_toxic ? '🚨 FLAGGED' : '✅ ALLOWED'}</span>
                    </div>
                    <div className="verdict-score-box">
                      <div className="verdict-score-num">
                        {(currentResult.toxicity_score * 100).toFixed(1)}%
                      </div>
                      <div className="verdict-score-label">Toxicity Risk Score</div>
                    </div>
                  </div>

                  {/* Telemetry Stats Grid */}
                  <div className="stats-grid">
                    <div className="stat-card">
                      <div className="stat-label">Context Influence</div>
                      <div className="stat-val" style={{ color: '#38bdf8' }}>
                        {currentResult.attention_analysis?.context_influence_pct || 0}%
                      </div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-label">Stage Routed</div>
                      <div className="stat-val" style={{ fontSize: '13px', color: '#a5b4fc' }}>
                        {currentResult.stage_routed || '-'}
                      </div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-label">Pipeline Latency</div>
                      <div className="stat-val" style={{ color: '#34d399' }}>
                        {currentResult.latency_ms} ms
                      </div>
                    </div>
                  </div>

                  {/* Multi-Head Attention Token Heatmap */}
                  <div className="heatmap-section">
                    <div className="section-subtitle">
                      <span>Multi-Head Attention Token Heatmap</span>
                      <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
                        <span style={{ color: '#38bdf8' }}>■ Target</span> |{' '}
                        <span style={{ color: '#c084fc' }}>■ Context</span>
                      </span>
                    </div>
                    <div className="token-heatmap-wrap">
                      {currentResult.attention_analysis?.tokens?.length > 0 ? (
                        currentResult.attention_analysis.tokens.map((tok, idx) => {
                          const isTarget = tok.is_target;
                          const intensity = Math.min(0.85, 0.15 + (tok.attention || 0.1) * 2);
                          const bg = isTarget
                            ? `rgba(56, 189, 248, ${intensity})`
                            : `rgba(192, 132, 252, ${intensity})`;
                          return (
                            <span
                              key={idx}
                              className={`attn-token ${isTarget ? 'target-tok' : 'context-tok'}`}
                              style={{ background: bg }}
                              title={`${isTarget ? 'Target' : 'Context'} Token: "${tok.token}" (Saliency: ${(
                                tok.attention * 100
                              ).toFixed(1)}%)`}
                            >
                              <span>{tok.token}</span>
                              <span className="attn-val">{(tok.attention * 100).toFixed(0)}%</span>
                            </span>
                          );
                        })
                      ) : (
                        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                          No token attention data available.
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Multi-Label Risk Bars */}
                  <div className="category-bars">
                    <div className="section-subtitle">
                      <span>Multi-Label Risk Profile</span>
                    </div>
                    {Object.entries(currentResult.categories || {}).map(([cat, val]) => (
                      <div key={cat} className="cat-row">
                        <div className="cat-info">
                          <span>{cat.replace('_', ' ').toUpperCase()}</span>
                          <span>{val === null || val === undefined ? 'n/a' : `${(val * 100).toFixed(1)}%`}</span>
                        </div>
                        <div className="cat-bar-bg">
                          <div
                            className="cat-bar-fill"
                            style={{
                              width: `${Math.min(100, Math.max(2, (val || 0) * 100))}%`,
                              background:
                                val >= 0.5
                                  ? 'linear-gradient(90deg, #f43f5e, #ef4444)'
                                  : 'linear-gradient(90deg, #6366f1, #38bdf8)',
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Contextual Intent Explanation */}
                  <div className="explanation-box">
                    <strong>Intent Diagnostics:</strong> {currentResult.explanation}
                  </div>
                </>
              ) : (
                <div style={{ textAlign: 'center', padding: '80px 20px', color: 'var(--text-muted)' }}>
                  <p>Select a turn in the chat or load a preset scenario to view live attention telemetry.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 2: CONTEXTUAL CONTRAST DEEP-DIVE */}
        {activeTab === 'deepdive' && (
          <div className="glass-card" style={{ maxWidth: '1000px', margin: '0 auto' }}>
            <div className="card-header">
              <span className="card-title">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="12 2 2 7 12 12 22 7 12 2" />
                  <polyline points="2 17 12 22 22 17" />
                  <polyline points="2 12 12 17 22 12" />
                </svg>
                Contextual Polarity Contrast Deep-Dive
              </span>
            </div>

            <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', marginBottom: '20px' }}>
              Test how the <strong>exact same target utterance</strong> flips its decision depending on preceding
              conversational context.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
              <div>
                <label style={{ fontSize: '12px', fontWeight: 700, color: 'var(--accent-purple)', display: 'block', marginBottom: '6px' }}>
                  Parent Context (Conversational History)
                </label>
                <textarea
                  className="chat-input"
                  style={{ width: '100%', height: '85px', resize: 'none' }}
                  value={deepDiveParent}
                  onChange={(e) => setDeepDiveParent(e.target.value)}
                  placeholder="e.g. Ninna match lo 1v4 clutch chusava bro?"
                />
              </div>
              <div>
                <label style={{ fontSize: '12px', fontWeight: 700, color: 'var(--accent-cyan)', display: 'block', marginBottom: '6px' }}>
                  Target Utterance to Moderate
                </label>
                <textarea
                  className="chat-input"
                  style={{ width: '100%', height: '85px', resize: 'none' }}
                  value={deepDiveTarget}
                  onChange={(e) => setDeepDiveTarget(e.target.value)}
                  placeholder="e.g. Champesav po!"
                />
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '24px' }}>
              <button className="send-btn" onClick={handleRunDeepDive} disabled={deepDiveLoading}>
                {deepDiveLoading ? 'Computing Dual Evaluation...' : 'Run Dual Comparison'}
              </button>
            </div>

            {/* Comparison Cards */}
            {deepDiveResult && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                {/* With Context */}
                <div
                  style={{
                    background: 'rgba(15, 23, 42, 0.75)',
                    padding: '20px',
                    borderRadius: 'var(--radius-md)',
                    border: `1px solid ${deepDiveResult.withCtx.is_toxic ? 'rgba(239, 68, 68, 0.4)' : 'rgba(16, 185, 129, 0.4)'}`,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <span style={{ fontSize: '13px', fontWeight: 700, color: '#ffffff' }}>WITH Parent Context</span>
                    <span className={`verdict-tag ${deepDiveResult.withCtx.is_toxic ? 'tag-flagged' : 'tag-allowed'}`}>
                      {deepDiveResult.withCtx.action} ({(deepDiveResult.withCtx.toxicity_score * 100).toFixed(0)}%)
                    </span>
                  </div>
                  <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginBottom: '14px' }}>
                    {deepDiveResult.withCtx.explanation}
                  </p>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                    Context Influence: <strong style={{ color: '#38bdf8' }}>{deepDiveResult.withCtx.attention_analysis?.context_influence_pct || 0}%</strong>
                  </div>
                </div>

                {/* Without Context */}
                <div
                  style={{
                    background: 'rgba(15, 23, 42, 0.75)',
                    padding: '20px',
                    borderRadius: 'var(--radius-md)',
                    border: `1px solid ${deepDiveResult.withoutCtx.is_toxic ? 'rgba(239, 68, 68, 0.4)' : 'rgba(16, 185, 129, 0.4)'}`,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <span style={{ fontSize: '13px', fontWeight: 700, color: '#ffffff' }}>WITHOUT Context (Isolated)</span>
                    <span className={`verdict-tag ${deepDiveResult.withoutCtx.is_toxic ? 'tag-flagged' : 'tag-allowed'}`}>
                      {deepDiveResult.withoutCtx.action} ({(deepDiveResult.withoutCtx.toxicity_score * 100).toFixed(0)}%)
                    </span>
                  </div>
                  <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginBottom: '14px' }}>
                    {deepDiveResult.withoutCtx.explanation}
                  </p>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                    Context Influence: <strong style={{ color: '#a5b4fc' }}>n/a (no context given)</strong>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: BENCHMARKS & ARCHITECTURE */}
        {activeTab === 'benchmarks' && (
          <div className="glass-card" style={{ maxWidth: '1000px', margin: '0 auto' }}>
            <div className="card-header">
              <span className="card-title">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 20V10" />
                  <path d="M12 20V4" />
                  <path d="M6 20v-6" />
                </svg>
                Indian-Specific 4-Gate Release Benchmarks
              </span>
              {benchmarks?.evaluated && (
                <span className="badge-passed">
                  {benchmarks.slices.filter((x) => x.passed).length}/{benchmarks.slices.length} passed · {benchmarks.evaluated_at}
                </span>
              )}
            </div>

            {!benchmarks && <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)' }}>Backend offline — no benchmark data.</p>}
            {benchmarks && !benchmarks.evaluated && (
              <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)' }}>{benchmarks.message}</p>
            )}

            {benchmarks?.evaluated && (
              <table className="benchmark-table">
                <thead>
                  <tr>
                    <th>Benchmark Slice</th>
                    <th>Target Gate</th>
                    <th>Observed Result</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {benchmarks.slices.map((b) => (
                    <tr key={b.id}>
                      <td><strong>{b.name}</strong><br /><span style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>{b.description}</span></td>
                      <td>{b.gate}</td>
                      <td style={{ color: b.passed ? '#34d399' : '#f87171', fontWeight: 700 }}>{b.value}</td>
                      <td><span className={b.passed ? 'badge-passed' : 'verdict-tag tag-flagged'}>{b.status}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {benchmarks?.training_stages && (
              <ul style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '16px', paddingLeft: '20px' }}>
                {benchmarks.training_stages.map((st) => (
                  <li key={st.stage}><strong>Stage {st.stage}</strong> ({st.checkpoint}): {st.metric}</li>
                ))}
              </ul>
            )}

            {/* Architecture Summary Box */}
            <div style={{ marginTop: '30px', padding: '20px', background: 'rgba(10, 15, 29, 0.7)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
              <h3 style={{ fontSize: '15px', color: '#ffffff', marginBottom: '10px' }}>4-Stage Cross-Lingual Transfer Curriculum</h3>
              <ul style={{ fontSize: '13px', color: 'var(--text-secondary)', paddingLeft: '20px', lineHeight: '1.7' }}>
                <li><strong>Stage 1 (Toxicity Mechanics):</strong> English toxic-chat + Jigsaw Unintended Bias.</li>
                <li><strong>Stage 2 (Telugu Domain Adaptation):</strong> Masked language modelling on Telugu IndicCorp text.</li>
                <li><strong>Stage 3 (Code-Mix Fine-Tuning):</strong> DravidianCodeMix (Tamil) + Cultural-Safety-Telugu.</li>
                <li><strong>Stage 4 (Synthetic Thread Fusion):</strong> Contrastive twin pairs (same target, safe vs toxic parent).</li>
              </ul>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
