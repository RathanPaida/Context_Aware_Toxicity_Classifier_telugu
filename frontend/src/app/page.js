"use client";

import { useEffect, useState } from "react";

export default function ContextAwareApp() {
  const [text, setText] = useState("");
  const [history, setHistory] = useState("");
  const [health, setHealth] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch("/api/health").then(r => r.json()).then(setHealth).catch(e => setError(e.message));
  }, []);

  async function analyze(event) {
    event.preventDefault();
    setBusy(true); setError(""); setResult(null);
    try {
      const response = await fetch("/api/moderate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, history: history.split("\n").filter(t => t.trim()) })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
      setResult(data);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  return (
    <main style={{ maxWidth: "850px", margin: "4rem auto", padding: "2rem" }}>
      <h1>Context-aware toxicity</h1>
      <p>Individual message embeddings. Learned parent?child context. No conversation category to select.</p>
      <p role="status">{health ? (health.ready ? "Trained model ready" : health.detail || "Model unavailable") : "Checking model?"}</p>
      <form onSubmit={analyze} style={{ display: "grid", gap: "1rem" }}>
        <label htmlFor="history">Parent messages, oldest first (one per line)</label>
        <textarea id="history" rows={5} value={history} onChange={e => setHistory(e.target.value)} />
        <label htmlFor="target">Target message or paragraph</label>
        <textarea id="target" rows={4} required value={text} onChange={e => setText(e.target.value)} />
        <button disabled={busy || !health?.ready}>{busy ? "Analyzing?" : "Analyze"}</button>
      </form>
      {error && <p role="alert">{error}</p>}
      {result && <section aria-live="polite">
        <h2>{result.is_toxic ? "Flagged target" : "Target below threshold"}</h2>
        <pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{JSON.stringify(result, null, 2)}</pre>
      </section>}
      <p>The result refers to the target message. Scores are not calibrated confidence estimates, and attention is not a causal explanation.</p>
    </main>
  );
}
