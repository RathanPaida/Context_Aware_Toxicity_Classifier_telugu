/**
 * Indic Shield AI - Web Application Client Logic
 * Handles interactive multi-turn thread simulation, diagnostic visualization,
 * scenario presets, and benchmarks.
 */

// State
let currentThreadId = "thread_live_" + Math.floor(1000 + Math.random() * 9000);
let presetScenarios = [];
let activeScenarioId = null;

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("thread-id-display").textContent = currentThreadId;
  initApp();
});

async function initApp() {
  try {
    // 1. Health check
    const healthRes = await fetch("/api/health");
    if (healthRes.ok) {
      const data = await healthRes.json();
      document.getElementById("system-status-text").textContent = `IndicBERT-v3 (${data.device.toUpperCase()} Online)`;
    }
  } catch (err) {
    console.warn("Health check error:", err);
    document.getElementById("system-status-text").textContent = "Connecting to pipeline...";
  }

  // 2. Load Preset Scenarios
  loadScenarios();

  // 3. Load Benchmarks
  loadBenchmarks();
}

// ─────────────────────────────────────────────────────────────────────────────
// Preset Scenarios
// ─────────────────────────────────────────────────────────────────────────────

async function loadScenarios() {
  try {
    const res = await fetch("/api/scenarios");
    if (!res.ok) return;
    presetScenarios = await res.json();
    renderScenariosCarousel();
  } catch (err) {
    console.error("Failed to load scenarios:", err);
  }
}

function renderScenariosCarousel() {
  const container = document.getElementById("scenarios-container");
  container.innerHTML = "";

  presetScenarios.forEach((s) => {
    const chip = document.createElement("button");
    chip.className = `scenario-chip ${activeScenarioId === s.id ? "active" : ""}`;
    chip.id = `chip-${s.id}`;
    chip.textContent = s.title;
    chip.onclick = () => runScenario(s);
    container.appendChild(chip);
  });
}

async function runScenario(scenario) {
  activeScenarioId = scenario.id;
  renderScenariosCarousel();

  // Reset thread
  currentThreadId = scenario.thread_id;
  document.getElementById("thread-id-display").textContent = currentThreadId;

  // Clear thread on backend
  await fetch("/api/clear-thread", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: currentThreadId }),
  });

  // Clear thread UI
  const stream = document.getElementById("thread-stream");
  stream.innerHTML = "";

  // Play through the turns with subtle delay
  for (let i = 0; i < scenario.turns.length; i++) {
    const turn = scenario.turns[i];
    await processMessage(turn.text, turn.speaker, i === scenario.turns.length - 1);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat Interaction
// ─────────────────────────────────────────────────────────────────────────────

function handleInputKeydown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendComment();
  }
}

function onTopicChange() {
  // Topic selection deprecated - conversation context is auto-inferred from thread history
}

async function resetThread() {
  currentThreadId = "thread_live_" + Math.floor(1000 + Math.random() * 9000);
  document.getElementById("thread-id-display").textContent = currentThreadId;
  activeScenarioId = null;
  renderScenariosCarousel();

  await fetch("/api/clear-thread", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: currentThreadId }),
  });

  const stream = document.getElementById("thread-stream");
  stream.innerHTML = `
    <div class="thread-empty-state" id="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
      </svg>
      <p>Thread context cleared.<br>Type a comment below to start a new stateful session.</p>
    </div>
  `;

  resetDiagnosticView();
}

function resetDiagnosticView() {
  const card = document.getElementById("verdict-card");
  card.className = "verdict-card";
  document.getElementById("verdict-headline").textContent = "STANDBY";
  document.getElementById("verdict-headline").style.color = "var(--safe-green)";
  document.getElementById("verdict-explanation").textContent = "Awaiting comment input...";
  document.getElementById("toxicity-score-val").textContent = "--%";
  document.getElementById("telemetry-route").textContent = "--";
  document.getElementById("telemetry-latency").textContent = "-- ms";

  const categories = ["toxic", "severe_toxic", "identity_attack", "insult", "threat"];
  categories.forEach((cat) => {
    document.getElementById(`score-${cat}`).textContent = "0.0%";
    const bar = document.getElementById(`bar-${cat}`);
    bar.style.width = "0%";
    bar.className = "progress-fill";
  });

  document.getElementById("seq-parent").textContent = "(No preceding turn)";
  document.getElementById("seq-target").textContent = "(Awaiting text)";
  document.getElementById("prep-raw").textContent = "--";
  document.getElementById("prep-clean").textContent = "--";

  const ctxPctEl = document.getElementById("attn-ctx-pct");
  if (ctxPctEl) ctxPctEl.textContent = "0.0%";
  const topWordsEl = document.getElementById("attn-top-words");
  if (topWordsEl) topWordsEl.textContent = "--";
  const heatmapContainer = document.getElementById("token-heatmap-container");
  if (heatmapContainer) heatmapContainer.innerHTML = '<span style="font-size: 12px; color: var(--text-muted);">(Multi-Head Attention token weights will display here after evaluation)</span>';
}

async function sendComment() {
  const input = document.getElementById("chat-input-text");
  const text = input.value.trim();
  if (!text) return;

  input.value = "";
  const sendBtn = document.getElementById("btn-send-comment");
  sendBtn.disabled = true;

  try {
    await processMessage(text, "You", true);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

async function processMessage(text, author = "User", isLast = true) {
  // Hide empty state if present
  const empty = document.getElementById("empty-state");
  if (empty) empty.remove();

  // Call Moderation API
  try {
    const res = await fetch("/api/moderate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        thread_id: currentThreadId,
        update_history: true,
      }),
    });

    if (!res.ok) throw new Error("API error: " + res.statusText);
    const data = await res.json();

    // Render message in stream
    appendMessageBubble(author, text, data);

    // Update diagnostic panel
    updateDiagnosticInspector(data);
  } catch (err) {
    console.error("Moderation request failed:", err);
    appendMessageBubble(author, text, {
      is_toxic: false,
      toxicity_score: 0.0,
      action: "ERROR",
      stage_routed: "Network Error",
      latency_ms: 0,
      categories: {},
    });
  }
}

function appendMessageBubble(author, text, result) {
  const stream = document.getElementById("thread-stream");
  const msgEl = document.createElement("div");
  msgEl.className = "chat-msg";

  const isToxic = result.is_toxic;
  const verdictClass = isToxic ? "verdict-flag" : "verdict-allow";
  const verdictText = isToxic
    ? `🚩 FLAGGED (${(result.toxicity_score * 100).toFixed(1)}%)`
    : `✓ ALLOWED (${(result.toxicity_score * 100).toFixed(1)}%)`;

  const initials = author.slice(0, 2).toUpperCase();

  msgEl.innerHTML = `
    <div class="msg-avatar">${initials}</div>
    <div class="msg-content">
      <div class="msg-header">
        <span class="msg-author">${escapeHtml(author)}</span>
        <span class="msg-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
      </div>
      <div class="msg-body">
        ${escapeHtml(text)}
        <div>
          <span class="msg-verdict ${verdictClass}">${verdictText}</span>
        </div>
      </div>
    </div>
  `;

  stream.appendChild(msgEl);
  stream.scrollTop = stream.scrollHeight;
}

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostic Inspector Visualizer
// ─────────────────────────────────────────────────────────────────────────────

function updateDiagnosticInspector(data) {
  const isToxic = data.is_toxic;
  const card = document.getElementById("verdict-card");
  card.className = `verdict-card ${isToxic ? "flagged" : ""}`;

  const headline = document.getElementById("verdict-headline");
  headline.textContent = isToxic ? "FLAGGED (TOXIC)" : "ALLOWED (BENIGN)";
  headline.style.color = isToxic ? "var(--toxic-red)" : "var(--safe-green)";

  document.getElementById("verdict-explanation").textContent =
    data.explanation || (isToxic ? "Utterance flagged under conversational context." : "Utterance verified safe.");

  const pct = Math.round(data.toxicity_score * 100);
  document.getElementById("toxicity-score-val").textContent = `${pct}%`;

  // Telemetry
  document.getElementById("telemetry-route").textContent = data.stage_routed;
  document.getElementById("telemetry-latency").textContent = `${data.latency_ms} ms`;
  document.getElementById("telemetry-pooling").textContent = "MHA 8-Head Fusion";

  // Category Risk Bars
  const cats = data.categories || {};
  for (const [key, val] of Object.entries(cats)) {
    const scoreEl = document.getElementById(`score-${key}`);
    const barEl = document.getElementById(`bar-${key}`);
    if (scoreEl && barEl) {
      const p = Math.round(val * 100);
      scoreEl.textContent = `${p}%`;
      barEl.style.width = `${p}%`;
      if (p >= 50) {
        barEl.className = "progress-fill high";
      } else {
        barEl.className = "progress-fill";
      }
    }
  }

  // Multi-Head Attention Heatmap & Analysis
  const attn = data.attention_analysis;
  if (attn) {
    const ctxPctEl = document.getElementById("attn-ctx-pct");
    if (ctxPctEl) ctxPctEl.textContent = `${attn.context_influence_pct || 0}%`;
    
    const topWordsEl = document.getElementById("attn-top-words");
    if (topWordsEl) topWordsEl.textContent = (attn.top_attended_words && attn.top_attended_words.length > 0) ? attn.top_attended_words.join(", ") : "--";

    renderTokenHeatmap("token-heatmap-container", attn.tokens);
  }

  // Context Sequence Breakdown
  const parentEl = document.getElementById("seq-parent");
  if (parentEl) parentEl.textContent = data.context_used || "(No preceding history)";
  const targetEl = document.getElementById("seq-target");
  if (targetEl) targetEl.textContent = data.cleaned_text || data.raw_text;

  // Preprocessor State
  document.getElementById("prep-raw").textContent = data.raw_text || "--";
  document.getElementById("prep-clean").textContent = data.cleaned_text || "--";
}

function renderTokenHeatmap(containerId, tokens) {
  const heatmapContainer = document.getElementById(containerId);
  if (!heatmapContainer) return;
  if (!tokens || tokens.length === 0) {
    heatmapContainer.innerHTML = '<span style="font-size: 12px; color: var(--text-muted);">(No tokens found)</span>';
    return;
  }
  heatmapContainer.innerHTML = "";
  const maxAttn = Math.max(...tokens.map((t) => t.attention), 0.0001);

  tokens.forEach((tok) => {
    const span = document.createElement("span");
    const isTarget = tok.is_target;
    span.className = `attn-token ${isTarget ? "target-tok" : "context-tok"}`;

    const intensity = Math.min(0.85, 0.15 + (tok.attention / maxAttn) * 0.7);
    const bgRgba = isTarget
      ? `rgba(56, 189, 248, ${intensity})`
      : `rgba(192, 132, 252, ${intensity})`;
    span.style.background = bgRgba;

    span.title = `${isTarget ? "Target Utterance" : "Parent Context"} Token: "${tok.token}" (Attention: ${(tok.attention * 100).toFixed(2)}%)`;
    span.innerHTML = `<span>${escapeHtml(tok.token)}</span><span class="attn-val">${(tok.attention * 100).toFixed(1)}%</span>`;
    heatmapContainer.appendChild(span);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab 2: Single Comment Deep Dive
// ─────────────────────────────────────────────────────────────────────────────

async function runDeepDive() {
  const parent = document.getElementById("deepdive-parent").value.trim();
  const target = document.getElementById("deepdive-target").value.trim();

  if (!target) {
    alert("Please enter a target utterance to evaluate.");
    return;
  }

  // Update preview code pill
  const ddParent = document.getElementById("dd-parent-txt");
  if (ddParent) ddParent.textContent = parent || "(None - Standalone)";
  const ddTarget = document.getElementById("dd-target-txt");
  if (ddTarget) ddTarget.textContent = target;

  try {
    const res = await fetch("/api/moderate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: target,
        thread_id: "deepdive_temp_thread",
        parent_context: parent || null,
      }),
    });

    if (!res.ok) throw new Error("Deep-dive API error: " + res.statusText);
    const data = await res.json();

    const card = document.getElementById("deepdive-verdict-card");
    const isToxic = data.is_toxic;
    card.className = `verdict-card ${isToxic ? "flagged" : ""}`;

    const headline = document.getElementById("deepdive-verdict-headline");
    headline.textContent = isToxic ? "FLAGGED (TOXIC)" : "ALLOWED (BENIGN)";
    headline.style.color = isToxic ? "var(--toxic-red)" : "var(--safe-green)";

    document.getElementById("deepdive-verdict-expl").textContent =
      data.explanation || `Evaluated via ${data.stage_routed}`;

    const pct = Math.round(data.toxicity_score * 100);
    document.getElementById("deepdive-score-val").textContent = `${pct}%`;
    document.getElementById("deepdive-latency-val").textContent = `${data.latency_ms} ms (${data.stage_routed})`;

    // Multi-Head Attention Heatmap for Deep Dive
    if (data.attention_analysis) {
      const attn = data.attention_analysis;
      const ddCtxPct = document.getElementById("deepdive-attn-ctx-pct");
      if (ddCtxPct) ddCtxPct.textContent = `${attn.context_influence_pct || 0}%`;
      const ddTopWords = document.getElementById("deepdive-attn-top-words");
      if (ddTopWords) ddTopWords.textContent = (attn.top_attended_words && attn.top_attended_words.length > 0) ? attn.top_attended_words.join(", ") : "--";
      renderTokenHeatmap("deepdive-token-heatmap-container", attn.tokens);
    }
  } catch (err) {
    console.error("Deep dive failed:", err);
    alert("Deep dive request failed: " + err.message);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab 3: Benchmarks Dashboard
// ─────────────────────────────────────────────────────────────────────────────

async function loadBenchmarks() {
  try {
    const res = await fetch("/api/benchmarks");
    if (!res.ok) return;
    const data = await res.json();

    // Render 4 Gates
    const cardsContainer = document.getElementById("benchmark-cards-container");
    cardsContainer.innerHTML = "";

    data.slices.forEach((s) => {
      const card = document.createElement("div");
      card.className = "benchmark-card";
      card.innerHTML = `
        <div class="benchmark-card-header">
          <span class="benchmark-title">${s.name}</span>
          <span class="badge-tag" style="background: rgba(16, 185, 129, 0.2); color: var(--safe-green); border-color: rgba(16, 185, 129, 0.4);">
            ${s.status}
          </span>
        </div>
        <p class="benchmark-desc">${s.description}</p>
        <div class="benchmark-metric-box">
          <div>
            <div style="font-size: 11px; color: var(--text-dim); text-transform: uppercase;">${s.metric}</div>
            <div class="benchmark-val">${s.value}</div>
          </div>
          <div class="benchmark-gate">Release Gate: <strong>${s.gate}</strong></div>
        </div>
      `;
      cardsContainer.appendChild(card);
    });

    // Render 4 Training Stages
    const stagesContainer = document.getElementById("training-stages-container");
    stagesContainer.innerHTML = "";

    data.training_stages.forEach((st) => {
      const box = document.createElement("div");
      box.className = "telemetry-box";
      box.innerHTML = `
        <div style="font-size: 11px; color: var(--cyan); font-weight: 700;">STAGE ${st.stage}</div>
        <div style="font-size: 13px; font-weight: 600; color: #fff; margin: 2px 0;">${st.name}</div>
        <div style="font-size: 11px; color: var(--text-muted);">${st.data}</div>
        <div style="font-size: 12px; font-family: var(--font-mono); color: var(--safe-green); margin-top: 4px;">${st.metric}</div>
      `;
      stagesContainer.appendChild(box);
    });
  } catch (err) {
    console.error("Failed to load benchmarks:", err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Navigation & Modals
// ─────────────────────────────────────────────────────────────────────────────

function switchTab(tabId) {
  const tabs = ["studio", "deepdive", "benchmarks"];
  tabs.forEach((t) => {
    document.getElementById(`tab-${t}`).classList.remove("active");
    document.getElementById(`view-${t}`).style.display = "none";
  });

  document.getElementById(`tab-${tabId}`).classList.add("active");
  document.getElementById(`view-${tabId}`).style.display = tabId === "studio" ? "grid" : "flex";
}

function openArchModal() {
  document.getElementById("arch-modal").classList.add("active");
}

function closeArchModal(event) {
  if (event) event.stopPropagation();
  document.getElementById("arch-modal").classList.remove("active");
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
