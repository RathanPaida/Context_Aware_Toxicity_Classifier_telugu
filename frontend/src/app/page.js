"use client";

import { useEffect, useState } from "react";

const PRESETS = [
  {
    title: "Case 1: Counterspeech Defense (English)",
    history: ["Shut up you stupid idiot, stop vandalizing or I will destroy you."],
    target: "I am following Wikipedia guidelines, please keep your comments civil."
  },
  {
    title: "Case 2: Direct Attack (English)",
    history: ["Why are you editing this article?"],
    target: "Shut up you stupid idiot, stop vandalizing or I will destroy you."
  },
  {
    title: "Case 3: Native Telugu Harassment (Telugu Script)",
    history: ["ఈ విషయం గురించి నీ అభిప్రాయం ఏమిటి?"],
    target: "నువ్వు ఒక చెత్త మనిషివి, నీలాంటి వాళ్ళు బతకడం అనవసరం."
  },
  {
    title: "Case 4: Calm Reply to Attack (Telugu Script)",
    history: ["నువ్వు ఒక చెత్త మనిషివి, నీలాంటి వాళ్ళు బతకడం అనవసరం."],
    target: "మనం గౌరవప్రదంగా మాట్లాడదాం, ఇలాంటి పదాలు వాడకండి."
  },
  {
    title: "Case 5: Romanized Telugu Abuse (Latin Script)",
    history: ["e roju match chusava?"],
    target: "waste kukkala unnav, nee moham chusuko ra lucha."
  },
  {
    title: "Case 6: Quoted Abuse Report (\"They said...\")",
    history: ["What exactly did the other player say to you?"],
    target: "They said, “You are a useless idiot.” I reported the message because personal attacks are unacceptable."
  }
];

export default function HomePage() {
  const [history, setHistory] = useState([
    "Shut up you stupid idiot, stop vandalizing or I will destroy you."
  ]);
  const [target, setTarget] = useState(
    "I am following Wikipedia guidelines, please keep your comments civil."
  );
  const [health, setHealth] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [execStatus, setExecStatus] = useState("Awaiting analysis");

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ ready: false, detail: "Backend unreachable" }));
  }, []);

  const addTurn = () => {
    if (history.length >= 11) {
      alert("Maximum window is 11 ancestors plus 1 target reply.");
      return;
    }
    setHistory([...history, ""]);
  };

  const removeTurn = (idx) => {
    setHistory(history.filter((_, i) => i !== idx));
  };

  const updateTurn = (idx, val) => {
    const updated = [...history];
    updated[idx] = val;
    setHistory(updated);
  };

  const loadPreset = (idx) => {
    const p = PRESETS[idx];
    setHistory([...p.history]);
    setTarget(p.target);
    executeAnalysis(p.target, p.history);
  };

  const clearAll = () => {
    setHistory([]);
    setTarget("");
    setResult(null);
    setExecStatus("Inputs cleared");
  };

  const executeAnalysis = async (targetText, historyArray) => {
    const textToAnalyze = (targetText ?? target).trim();
    if (!textToAnalyze) {
      alert("Please provide a target message to analyze.");
      return;
    }

    const turns = (historyArray ?? history).map((t) => t.trim()).filter(Boolean);

    setLoading(true);
    setExecStatus("Processing with neural encoder...");

    try {
      const res = await fetch("/api/moderate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textToAnalyze, history: turns })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Analysis failed");

      setResult(data);
      setExecStatus("Evaluation complete");
    } catch (err) {
      alert("Analysis error: " + err.message);
      setExecStatus("Error occurred");
    } finally {
      setLoading(false);
    }
  };

  const isToxic = result?.is_toxic;
  const score = result?.toxicity_score ?? null;
  const isoScore = result?.isolated_score ?? null;
  const delta = result?.context_score_delta ?? null;
  const threshold = result?.threshold ?? health?.threshold ?? null;
  const pct = score !== null ? Math.min(Math.max(score * 100, 0), 100) : 0;

  return (
    <div className="container">
      {/* Top Bar */}
      <header className="header">
        <div className="header-title">
          <h1>Context-Aware Toxicity Engine</h1>
          <p>
            Hierarchical Multilingual Architecture · Target-Bypass Classification Head ·
            Conservative Expert Routing
          </p>
        </div>
        <div className="badge-bar">
          <span
            className={`badge ${
              health ? (health.ready ? "online" : "offline") : ""
            }`}
          >
            {health
              ? health.ready
                ? `Online (${(health.device || "CPU").toUpperCase()}) — ${health.predictor_type || health.model}`
                : health.detail || "Model offline"
              : "Connecting..."}
          </span>
          <span className="badge active">Policy: script_soft_blend</span>
          <span className="badge">
            Threshold: {threshold !== null ? threshold.toFixed(4) : "Per selected model"}
          </span>
          <a
            href="http://127.0.0.1:7860/docs"
            target="_blank"
            rel="noreferrer"
            className="badge"
            style={{ color: "#38bdf8" }}
          >
            API Docs →
          </a>
        </div>
      </header>

      {/* Preset Scenarios */}
      <div className="presets-container">
        <div className="presets-header">
          <span>1-Click Test Scenarios (Demonstrating Context-Aware Decisions)</span>
        </div>
        <div className="preset-chips">
          {PRESETS.map((p, idx) => (
            <button
              key={idx}
              type="button"
              className="chip-btn"
              onClick={() => loadPreset(idx)}
            >
              <span
                className={`dot ${
                  idx === 0 || idx === 3 || idx === 5 ? "safe" : "toxic"
                }`}
              />
              <strong>Case {idx + 1}:</strong> {p.title.split(": ")[1]}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid">
        {/* Input Card */}
        <div className="card">
          <div className="card-title">
            <span>Conversation Thread Input</span>
            <span
              style={{
                fontSize: "11px",
                color: "var(--text-dim)",
                fontWeight: 400
              }}
            >
              Oldest ancestors first
            </span>
          </div>

          <div className="field-label">
            <span>Preceding Conversation Turns (Ancestors)</span>
            <span
              style={{
                fontFamily: "var(--font-mono)",
                color: "var(--text-dim)"
              }}
            >
              {history.length} ancestor{history.length === 1 ? "" : "s"}
            </span>
          </div>

          <div className="timeline">
            {history.map((turn, i) => (
              <div key={i} className="turn-row">
                <div className="turn-tag">Turn {i + 1}</div>
                <textarea
                  rows={2}
                  value={turn}
                  onChange={(e) => updateTurn(i, e.target.value)}
                  placeholder={`Ancestor comment #${i + 1}...`}
                />
                <button
                  type="button"
                  className="btn-remove"
                  onClick={() => removeTurn(i)}
                  title="Remove turn"
                >
                  ×
                </button>
              </div>
            ))}
          </div>

          <button type="button" className="btn-add" onClick={addTurn}>
            + Add Preceding Turn
          </button>

          <div className="field-label">
            <span style={{ color: "#ffffff" }}>Target Reply to Evaluate</span>
            <span style={{ color: "#38bdf8", fontSize: "11px" }}>
              Classified in context
            </span>
          </div>

          <div className="target-box">
            <textarea
              rows={3}
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              placeholder="Type or paste the reply message here..."
            />
          </div>

          <div className="actions">
            <button
              type="button"
              className="btn-primary"
              disabled={loading}
              onClick={() => executeAnalysis()}
            >
              {loading ? "Analyzing..." : "Analyze Target Reply"}
            </button>
            <button type="button" className="btn-secondary" onClick={clearAll}>
              Clear All
            </button>
          </div>
        </div>

        {/* Output Card */}
        <div className="card">
          <div className="card-title">
            <span>Moderation &amp; Architectural Diagnostics</span>
            <span
              style={{ fontSize: "11px", color: "var(--text-dim)" }}
            >
              {execStatus}
            </span>
          </div>

          {/* Verdict Banner */}
          <div
            className={`verdict-banner ${
              result ? (isToxic ? "toxic" : "safe") : ""
            }`}
          >
            <div className="verdict-main">
              <h2>
                {result
                  ? isToxic
                    ? "TOXIC / POLICY VIOLATION"
                    : "NON-TOXIC (SAFE)"
                  : "Ready for Input"}
              </h2>
              <p>
                {result
                  ? isToxic
                    ? "Target comment is classified as abusive/toxic in this conversation context."
                    : "Target comment is classified as respectful and non-violating."
                  : "Enter a thread or choose a scenario above to evaluate."}
              </p>
            </div>
            <div className="score-large">
              {score !== null ? score.toFixed(4) : "—"}
            </div>
          </div>

          {/* Gauge */}
          <div className="gauge-wrapper">
            <div className="gauge-header">
              <span>Decision Score (not calibrated confidence)</span>
              <span>
                Threshold:{" "}
                {threshold !== null
                  ? threshold.toFixed(4)
                  : "Per selected model"}
              </span>
            </div>
            <div className="gauge-bar">
              <div
                className={`gauge-fill ${
                  result ? (isToxic ? "toxic" : "safe") : ""
                }`}
                style={{ width: `${pct}%` }}
              />
              <div
                className="gauge-threshold-line"
                title="Selected model decision boundary"
                style={{ left: `${(threshold ?? 0.5) * 100}%`, display: threshold === null ? "none" : "block" }}
              />
            </div>
          </div>

          {/* Context Influence Delta */}
          <div className="delta-card">
            <div className="delta-metric">
              <div className="label">In Context</div>
              <div className="num">
                {score !== null ? score.toFixed(4) : "—"}
              </div>
            </div>
            <div className="delta-metric">
              <div className="label">In Isolation</div>
              <div className="num">
                {isoScore !== null ? isoScore.toFixed(4) : "—"}
              </div>
            </div>
            <div className="delta-metric">
              <div className="label">Context Impact (Δ)</div>
              <div
                className="num"
                style={{
                  color:
                    delta === null || Math.abs(delta) < 0.0005
                      ? "var(--text-muted)"
                      : delta > 0
                      ? "#f87171"
                      : "#34d399"
                }}
              >
                {delta !== null
                  ? `${delta > 0 ? "+" : ""}${delta.toFixed(4)}`
                  : "—"}
              </div>
            </div>
          </div>

          {/* Diagnostic Grid */}
          <div className="meta-grid">
            <div className="meta-item">
              <div className="label">Detected Discourse / Script</div>
              <div className="val">
                {result?.engine_used === "contrast_reporting_specialist"
                  ? "Quoted/Reported Discourse"
                  : result?.script_bucket || "—"}
              </div>
            </div>
            <div className="meta-item">
              <div className="label">Active Engine / Mixture</div>
              <div className="val">
                {result?.engine_used === "contrast_reporting_specialist"
                  ? "Specialist (Adapted LoRA)"
                  : result?.selected_model || (result?.expert_weights
                  ? `En: ${(result.expert_weights.english_ensemble * 100).toFixed(
                      0
                    )}% | Mix: ${(
                      result.expert_weights.mixed_ensemble * 100
                    ).toFixed(0)}%`
                  : "Standard Adaptive")}
              </div>
            </div>
            <div className="meta-item">
              <div className="label">Total Turns Evaluated</div>
              <div className="val">
                {result
                  ? `${
                      result.turns_used ||
                      (result.ancestors_omitted ? "Truncated" : 1)
                    } turns`
                  : "—"}
              </div>
            </div>
            <div className="meta-item">
              <div className="label">Routing Reason</div>
              <div className="val" style={{ color: "#34d399" }}>
                {result?.routing_reason || "Awaiting prediction"}
              </div>
            </div>
          </div>

          {/* Raw JSON Explorer */}
          <details>
            <summary>Inspect Complete Model Output Payload</summary>
            <pre>
              {result
                ? JSON.stringify(result, null, 2)
                : "// Raw response from /api/moderate will appear here"}
            </pre>
          </details>
        </div>
      </div>
    </div>
  );
}
