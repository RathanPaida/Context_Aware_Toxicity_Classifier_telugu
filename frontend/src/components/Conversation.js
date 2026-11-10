'use client';

import { useEffect, useRef, useState } from 'react';
import ResultCard from './ResultCard';
import { VERDICT_TEXT, moderate, verdictOf } from '../lib/api';

export default function Conversation({ scenarios, labels }) {
  const [messages, setMessages] = useState([]); // {speaker, name, text, result}
  const [selected, setSelected] = useState(null);
  const [speaker, setSpeaker] = useState('a');
  const [input, setInput] = useState('');
  const [expectation, setExpectation] = useState(null);
  const chatRef = useRef(null);
  const messagesRef = useRef([]);

  useEffect(() => {
    messagesRef.current = messages;
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages]);

  async function send(text, who, name) {
    const history = messagesRef.current.map((m) => m.text);
    const index = messagesRef.current.length;
    const pending = [...messagesRef.current, { speaker: who, name, text, result: null }];
    messagesRef.current = pending;
    setMessages(pending);
    setSelected(index);
    const result = await moderate(text, history);
    const done = messagesRef.current.map((m, i) => (i === index ? { ...m, result } : m));
    messagesRef.current = done;
    setMessages(done);
    return result;
  }

  async function playScenario(id) {
    const sc = scenarios.find((s) => s.id === id);
    if (!sc) return;
    messagesRef.current = [];
    setMessages([]);
    setExpectation(null);
    const speakers = [...new Set(sc.turns.map((t) => t.speaker))];
    let last = null;
    for (const t of sc.turns) {
      last = await send(t.text, speakers.indexOf(t.speaker) % 2 ? 'b' : 'a', t.speaker);
    }
    if (last && !last.error) {
      const got = last.is_toxic ? 'FLAGGED' : 'ALLOWED';
      setExpectation({ match: got === sc.expected, text: `${sc.title}: expected ${sc.expected.toLowerCase()}, the model says ${got.toLowerCase()} ${got === sc.expected ? '✓' : '✗'}` });
    }
  }

  function onSubmit(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text) return;
    setInput('');
    setExpectation(null);
    send(text, speaker);
    setSpeaker(speaker === 'a' ? 'b' : 'a');
  }

  function clear() {
    messagesRef.current = [];
    setMessages([]);
    setSelected(null);
    setExpectation(null);
  }

  const current = selected !== null ? messages[selected] : null;

  return (
    <div className="grid-2">
      <div className="panel">
        <div className="panel-head">
          <h2>Conversation</h2>
          <div className="actions">
            <select aria-label="Load an example conversation" value="" onChange={(e) => playScenario(e.target.value)}>
              <option value="">Load an example…</option>
              {scenarios.map((s) => <option key={s.id} value={s.id}>{s.title}</option>)}
            </select>
            <button type="button" className="btn btn-small" onClick={clear}>Clear</button>
          </div>
        </div>
        <p className="hint">Every new message is checked together with the messages before it. Click a message to see why.</p>
        <div className="chat" ref={chatRef} aria-live="polite">
          {messages.length === 0 && <p className="hint">No messages yet. Type one below or load an example.</p>}
          {messages.map((m, i) => {
            const v = m.result ? verdictOf(m.result) : '';
            return (
              <button key={i} type="button" className="bubble" data-side={m.speaker}
                      aria-pressed={String(selected === i)} onClick={() => setSelected(i)}>
                <span className="bubble-meta">
                  <span>{m.name || (m.speaker === 'a' ? 'Person A' : 'Person B')}</span>
                  <span className="badge" data-verdict={v}>{v ? VERDICT_TEXT[v] : 'checking…'}</span>
                </span>
                <span>{m.text}</span>
              </button>
            );
          })}
        </div>
        {expectation && <div className="expectation" data-match={String(expectation.match)}>{expectation.text}</div>}
        <form className="composer" onSubmit={onSubmit}>
          <select aria-label="Who is speaking" value={speaker} onChange={(e) => setSpeaker(e.target.value)}>
            <option value="a">Person A</option>
            <option value="b">Person B</option>
          </select>
          <input type="text" value={input} placeholder="Type a message…" autoComplete="off" aria-label="Message"
                 onChange={(e) => setInput(e.target.value)} />
          <button type="submit" className="btn btn-primary">Send</button>
        </form>
      </div>
      <div aria-live="polite">
        {!current && <div className="empty">Select a message to see its result.</div>}
        {current && !current.result && <div className="spinner">Checking…</div>}
        {current && current.result && <ResultCard result={current.result} caption={`Message ${selected + 1}`} labels={labels} />}
      </div>
    </div>
  );
}
