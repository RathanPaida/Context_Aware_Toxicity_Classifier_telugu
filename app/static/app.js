/*
 * Indic Shield — FastAPI page (no build step).
 * Same views and behaviour as the Next.js app in frontend/.
 */

const LABELS = { toxic: "Toxic", severe_toxic: "Severe", identity_attack: "Identity attack", insult: "Insult", threat: "Threat" };
const VERDICT_TEXT = { allow: "Allowed", flag: "Flagged", review: "Needs review", error: "Could not check" };

const state = {
  health: null,
  scenarios: [],
  chat: [],            // {speaker: "a"|"b", name, text, result}
  selected: null,
  resultsLoaded: false,
};

// ── helpers ─────────────────────────────────────────────────────────

/** Tiny element builder. Strings become text nodes (never HTML). */
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") Object.assign(el.style, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
const $ = (id) => document.getElementById(id);
const pct = (x) => (x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`);

async function api(path, body) {
  try {
    const res = await fetch(path, body ? {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    } : undefined);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) return { ok: false, error: data.detail || `Server error (${res.status})` };
    return { ok: true, data };
  } catch {
    return { ok: false, error: "Cannot reach the server. Start it with: python app/server.py" };
  }
}

function verdictOf(r) {
  if (!r || r.error) return "error";
  if (r.needs_review) return "review";
  return r.is_toxic ? "flag" : "allow";
}

async function moderate(text, history, topic) {
  const res = await api("/api/moderate", { text, history, topic: topic || null });
  return res.ok ? res.data : { error: res.error };
}

// ── status ──────────────────────────────────────────────────────────

function showBanner(kind, text) {
  const b = $("banner");
  b.dataset.kind = kind;
  b.textContent = text;
  b.hidden = false;
}

async function refreshStatus() {
  const res = await api("/api/health");
  const status = $("status");
  $("banner").hidden = true;
  if (!res.ok) {
    status.dataset.state = "error";
    status.textContent = "Server offline";
    showBanner("error", "Cannot reach the server. Start it with: python app/server.py");
    return;
  }
  const hl = res.data;
  state.health = hl;
  Object.assign(LABELS, hl.label_display_names || {});
  if (hl.ready && hl.trained) {
    status.dataset.state = "ready";
    const src = hl.release_source || hl.stage || "model";
    const when = hl.trained_at ? ` · trained ${hl.trained_at.slice(0, 10)}` : "";
    status.textContent = `Model ready · ${src}${when}`;
  } else if (hl.ready) {
    status.dataset.state = "error";
    status.textContent = "Untrained model";
    showBanner("error", "Demo mode: the model is untrained (ALLOW_UNTRAINED=1), so its predictions are random. Do not use these results.");
  } else {
    status.dataset.state = "warn";
    status.textContent = "No trained model";
    showBanner("warn", "No trained model found. Only the word list works until you train one: python run_pipeline.py --all (see docs/TRAINING_GUIDE.md), then restart the server.");
  }
}

// ── result card ─────────────────────────────────────────────────────

function segmentLabel(seg, nTurns) {
  if (seg === "target") return "Checked";
  if (seg === "topic") return "Topic";
  const n = Number(seg.split("-")[1]);
  return n === nTurns ? "Just before" : `Earlier ${n}`;
}

/** One row per message (topic / earlier messages / checked message), shaded by attention. */
function renderHighlight(tokens) {
  const max = Math.max(...tokens.map((t) => t.attention), 1e-6);
  const nTurns = Math.max(0, ...tokens.filter((t) => (t.segment || "").startsWith("turn-")).map((t) => Number(t.segment.split("-")[1])));
  const rows = [];
  let row = null;
  tokens.forEach((t) => {
    const seg = t.segment || (t.is_target ? "target" : "turn-1");
    if (!row || row.seg !== seg) {
      row = { seg, text: h("span", { class: "hl-text" }) };
      rows.push(row);
    } else if (!t.continuation) {
      row.text.append(" ");
    }
    const rgb = t.is_target ? "var(--hl-target)" : "var(--hl-context)";
    const alpha = (0.06 + 0.55 * (t.attention / max)).toFixed(2);
    row.text.append(h("span", { class: "tok", title: `${(t.attention * 100).toFixed(1)}% attention`,
      style: { background: `rgba(${rgb}, ${alpha})` } }, t.token));
  });
  return h("div", { class: "highlight" }, rows.map((r) =>
    h("div", { class: "hl-row" }, h("span", { class: "hl-label" }, segmentLabel(r.seg, nTurns)), r.text)));
}

function renderResultCard(r, caption) {
  const v = verdictOf(r);
  if (v === "error") {
    return h("article", { class: "result", "data-verdict": "error" },
      caption && h("div", { class: "result-caption" }, caption),
      h("div", { class: "verdict" }, h("span", { class: "verdict-label" }, VERDICT_TEXT.error)),
      h("p", { class: "reason" }, r.error));
  }
  const threshold = r.thresholds ? r.thresholds.toxic : 0.5;
  const score = r.categories && r.categories.toxic !== null && r.categories.toxic !== undefined ? r.categories.toxic : r.toxicity_score;
  const lean = v === "review" ? ` · leaning ${r.is_toxic ? "flagged" : "allowed"}` : "";

  const card = h("article", { class: "result", "data-verdict": v },
    caption && h("div", { class: "result-caption" }, caption),
    h("div", { class: "verdict" },
      h("span", { class: "verdict-label" }, VERDICT_TEXT[v]),
      h("span", { class: "verdict-score" }, `harm score ${pct(score)}${lean}`)),
    h("p", { class: "reason" }, r.reason || r.explanation || ""),
    h("div", {},
      h("div", { class: "meter", role: "img", "aria-label": `Harm score ${pct(score)}, decision line ${pct(threshold)}` },
        h("div", { class: "meter-fill", style: { width: `${Math.max(2, score * 100)}%` } }),
        h("div", { class: "meter-threshold", style: { left: `${threshold * 100}%` } })),
      h("div", { class: "meter-legend" }, h("span", {}, "0%"), h("span", {}, `decision line ${pct(threshold)}`), h("span", {}, "100%"))),
  );

  if (r.decided_by === "model" && r.categories) {
    card.append(h("ul", { class: "categories", "aria-label": "Scores by type of harm" },
      Object.entries(r.categories).map(([k, val]) => {
        const line = r.flag_lines ? r.flag_lines[k] : (r.thresholds && r.thresholds[k]);
        const over = line !== null && line !== undefined && val >= line;
        return h("li", { class: "category", "data-over": String(!!over) },
          h("span", { class: "category-name" }, LABELS[k] || k),
          h("span", { class: "category-bar" }, h("span", { class: "category-fill", style: { width: `${(val || 0) * 100}%`, display: "block" } })),
          h("span", { class: "category-value" }, pct(val)));
      })));
  } else if (r.decided_by === "word list") {
    card.append(h("p", { class: "note" }, `Decided by the word list (matched: ${((r.lexicon && r.lexicon.severe) || []).join(", ")}). The model was not needed.`));
  }
  if (r.untrained) card.append(h("p", { class: "note" }, "⚠ Untrained model — this result is random."));

  const details = h("details", { class: "more" }, h("summary", {}, "Details"));
  const a = r.attention_analysis;
  if (a && a.tokens && a.tokens.length) {
    details.append(
      h("p", { class: "hint" }, "Where the model looked (darker = more attention):"),
      renderHighlight(a.tokens),
    );
  }
  const lex = r.lexicon || {};
  details.append(h("dl", { class: "kv" },
    h("dt", {}, "Decided by"), h("dd", {}, r.decided_by === "model" ? `model${a ? ` (${a.num_heads}-head attention)` : ""}` : "word list"),
    h("dt", {}, "Earlier messages used"), h("dd", {}, r.history && r.history.length ? String(r.history.length) : "none"),
    h("dt", {}, "Text after clean-up"), h("dd", {}, r.cleaned_text || "–"),
    h("dt", {}, "Words noted"), h("dd", {}, [...(lex.severe || []), ...(lex.mild || [])].join(", ") || "none"),
    h("dt", {}, "Time"), h("dd", {}, `${r.latency_ms} ms`),
  ));
  card.append(details);
  return card;
}

// ── Check a message ────────────────────────────────────────────────

function renumber() {
  [...$("context-list").children].forEach((li, i) => { li.firstChild.textContent = `${i + 1}.`; });
}

function addContextRow(value = "") {
  const ta = h("textarea", { rows: "2", placeholder: "What someone said earlier…", "aria-label": "Earlier message" });
  ta.value = value;
  const li = h("li", { class: "context-item" },
    h("span", { class: "context-num" }, ""),
    ta,
    h("button", { type: "button", class: "btn btn-icon", "aria-label": "Remove this message", onclick: () => { li.remove(); renumber(); } }, "✕"));
  $("context-list").append(li);
  renumber();
  return ta;
}

function compareSentence(withCtx, without) {
  const a = withCtx.categories && withCtx.categories.toxic != null ? withCtx.categories.toxic : withCtx.toxicity_score;
  const b = without.categories && without.categories.toxic != null ? without.categories.toxic : without.toxicity_score;
  const diff = Math.round((a - b) * 100);
  const change = verdictOf(withCtx) !== verdictOf(without)
    ? `The earlier messages changed the decision (${VERDICT_TEXT[verdictOf(without)]} → ${VERDICT_TEXT[verdictOf(withCtx)]}).`
    : "The decision is the same either way.";
  return `${change} Harm score ${diff >= 0 ? "+" : ""}${diff} points with context (${pct(b)} → ${pct(a)}).`;
}

async function runCheck(e) {
  if (e) e.preventDefault();
  const target = $("target").value.trim();
  const out = $("check-result");
  if (!target) { $("target").focus(); return; }
  const history = [...$("context-list").querySelectorAll("textarea")].map((t) => t.value.trim()).filter(Boolean);
  const topic = $("topic").value.trim();
  const btn = $("check-btn");
  btn.disabled = true;
  out.replaceChildren(h("div", { class: "spinner" }, "Checking…"));

  const withCtx = await moderate(target, history, topic);
  if (history.length && $("compare").checked) {
    const without = await moderate(target, [], topic);
    const both = !withCtx.error && !without.error;
    out.replaceChildren(h("div", { class: "stack" },
      h("div", { class: "compare" },
        renderResultCard(withCtx, "With earlier messages"),
        renderResultCard(without, "Without earlier messages")),
      both && h("p", { class: "compare-summary" }, compareSentence(withCtx, without))));
  } else {
    out.replaceChildren(renderResultCard(withCtx));
  }
  btn.disabled = false;
}

function loadExampleIntoForm(id) {
  const sc = state.scenarios.find((s) => s.id === id);
  if (!sc) return;
  $("context-list").replaceChildren();
  sc.turns.slice(0, -1).forEach((t) => addContextRow(t.text));
  if (sc.turns.length === 1) addContextRow();
  $("target").value = sc.turns[sc.turns.length - 1].text;
  $("topic").value = sc.topic || "";
  runCheck();
}

// ── Conversation ────────────────────────────────────────────────────

function renderChat() {
  const chat = $("chat");
  if (!state.chat.length) {
    chat.replaceChildren(h("p", { class: "hint" }, "No messages yet. Type one below or load an example."));
    return;
  }
  chat.replaceChildren(...state.chat.map((m, i) => {
    const v = m.result ? verdictOf(m.result) : null;
    return h("button", {
      type: "button", class: "bubble", "data-side": m.speaker, "aria-pressed": String(state.selected === i),
      onclick: () => selectMessage(i),
    },
      h("span", { class: "bubble-meta" },
        h("span", {}, m.name || (m.speaker === "a" ? "Person A" : "Person B")),
        h("span", { class: "badge", "data-verdict": v || "" }, v ? VERDICT_TEXT[v] : "checking…")),
      h("span", {}, m.text));
  }));
  chat.scrollTop = chat.scrollHeight;
}

function selectMessage(i) {
  state.selected = i;
  renderChat();
  const m = state.chat[i];
  $("chat-result").replaceChildren(m.result ? renderResultCard(m.result, `Message ${i + 1}`) : h("div", { class: "spinner" }, "Checking…"));
}

async function sendChat(text, speaker, name) {
  const history = state.chat.map((m) => m.text);
  const msg = { speaker, name, text, result: null };
  state.chat.push(msg);
  const i = state.chat.length - 1;
  selectMessage(i);
  msg.result = await moderate(text, history, null);
  selectMessage(i);
  return msg.result;
}

async function playScenario(id) {
  const sc = state.scenarios.find((s) => s.id === id);
  if (!sc) return;
  state.chat = [];
  $("expectation").hidden = true;
  const speakers = [...new Set(sc.turns.map((t) => t.speaker))];
  let last = null;
  for (const t of sc.turns) {
    last = await sendChat(t.text, speakers.indexOf(t.speaker) % 2 ? "b" : "a", t.speaker);
  }
  if (last && !last.error) {
    const got = last.is_toxic ? "FLAGGED" : "ALLOWED";
    const ex = $("expectation");
    ex.dataset.match = String(got === sc.expected);
    ex.textContent = `${sc.title}: expected ${sc.expected.toLowerCase()}, the model says ${got.toLowerCase()} ${got === sc.expected ? "✓" : "✗"}`;
    ex.hidden = false;
  }
}

// ── Results ─────────────────────────────────────────────────────────

function table(headers, rows) {
  return h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, headers.map((x) => h("th", {}, x)))),
    h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c) => h("td", {}, c)))))));
}

async function loadResults() {
  if (state.resultsLoaded) return;
  const out = $("results");
  const [bench, base] = await Promise.all([api("/api/benchmarks"), api("/api/baselines")]);
  if (!bench.ok) { out.replaceChildren(h("div", { class: "empty" }, bench.error)); return; }
  state.resultsLoaded = true;
  const b = bench.data;
  const blocks = [];

  blocks.push(h("section", { class: "panel" },
    h("h2", {}, "Trained models"),
    table(["Stage", "Checkpoint", "Data", "Result"], b.training_stages.map((s) => [
      String(s.stage), s.checkpoint, s.data, h("span", { class: s.metric.startsWith("not") ? "dim" : "" }, s.metric)]))));

  if (b.evaluated) {
    blocks.push(h("section", { class: "panel" },
      h("div", { class: "panel-head" }, h("h2", {}, "Evaluation"), h("span", { class: "hint" }, `measured ${b.evaluated_at}`)),
      h("p", { class: "hint" }, "The held-out test set is real data the model never trained on. The other rows are small hand-written checks — read them with their n."),
      table(["Check", "What it measures", "Result", "Target", ""], b.slices.map((s) => [
        s.name, s.description, h("span", { class: "num" }, s.value), s.gate,
        h("span", { class: s.status === "FAILED" ? "fail" : s.status === "PASSED" ? "pass" : "dim" }, s.status.toLowerCase())]))));
  } else {
    blocks.push(h("section", { class: "panel" }, h("h2", {}, "Evaluation"), h("p", { class: "hint" }, b.message)));
  }

  if (base.ok && base.data.available) {
    const d = base.data;
    const names = { lexicon_only: "Word list only", tfidf_logreg: "TF-IDF + logistic regression", trained_model: "This model" };
    const num = (x) => h("span", { class: "num" }, x === undefined || x === null ? "–" : x.toFixed(3));
    const panel = h("section", { class: "panel" },
      h("h2", {}, "Compared with simple methods"),
      h("p", { class: "hint" }, `Held-out test set, n = ${d.n_test.toLocaleString()}, ${pct(d.test_toxic_rate)} toxic. Higher is better.`),
      table(["Method", "F1", "Precision", "Recall", "ROC-AUC"], Object.entries(d.methods).map(([k, m]) => [
        names[k] || k, num(m.f1), num(m.precision), num(m.recall), num(m.roc_auc)])));
    const ab = d.context_ablation;
    if (ab) {
      panel.append(h("p", {}, `Context check: on ${ab.n_pairs} message pairs where only the earlier message differs, the model got both right in ${pct(ab.contrastive_flip_with_context)} of pairs (impossible without context). A threatening earlier message raised the score in ${ab.pairs_where_toxic_parent_raises_score} of ${ab.n_pairs} pairs.`));
    } else {
      panel.append(h("p", { class: "hint" }, "Train a model and rerun python run_pipeline.py --baselines to add it to this table."));
    }
    blocks.push(panel);
  } else {
    blocks.push(h("section", { class: "panel" }, h("h2", {}, "Compared with simple methods"),
      h("p", { class: "hint" }, base.ok ? base.data.message : base.error)));
  }
  out.replaceChildren(...blocks);
}

// ── navigation & init ───────────────────────────────────────────────

function showView(name) {
  document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.view === name)));
  document.querySelectorAll(".view").forEach((v) => { v.hidden = v.id !== `view-${name}`; });
  if (name === "results") loadResults();
}

async function loadScenarios() {
  const res = await api("/api/scenarios");
  if (!res.ok) return;
  state.scenarios = res.data;
  for (const sel of [$("example-select"), $("scenario-select")]) {
    for (const s of state.scenarios) sel.append(h("option", { value: s.id }, s.title));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showView(t.dataset.view)));
  $("add-context").addEventListener("click", () => addContextRow().focus());
  $("check-form").addEventListener("submit", runCheck);
  $("target").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runCheck(e); });
  $("example-select").addEventListener("change", (e) => { if (e.target.value) loadExampleIntoForm(e.target.value); e.target.value = ""; });
  $("scenario-select").addEventListener("change", (e) => { if (e.target.value) playScenario(e.target.value); e.target.value = ""; });
  $("clear-chat").addEventListener("click", () => {
    state.chat = []; state.selected = null; $("expectation").hidden = true; renderChat();
    $("chat-result").replaceChildren(h("div", { class: "empty" }, "Select a message to see its result."));
  });
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = $("chat-input");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    $("expectation").hidden = true;
    const sp = $("speaker");
    sendChat(text, sp.value);
    sp.value = sp.value === "a" ? "b" : "a";  // alternate speakers by default
  });
  addContextRow();
  renderChat();
  refreshStatus();
  loadScenarios();
  setInterval(refreshStatus, 30000);
});
