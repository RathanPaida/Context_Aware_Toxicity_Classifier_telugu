'use client';

import { useEffect, useState } from 'react';
import StatusBar from '../components/StatusBar';
import CheckMessage from '../components/CheckMessage';
import Conversation from '../components/Conversation';
import ResultsView from '../components/ResultsView';
import About from '../components/About';
import { DEFAULT_LABELS, api } from '../lib/api';

const TABS = [
  ['check', 'Check a message'],
  ['conversation', 'Conversation'],
  ['results', 'Results'],
  ['about', 'About'],
];

export default function Home() {
  const [tab, setTab] = useState('check');
  const [health, setHealth] = useState(null);
  const [scenarios, setScenarios] = useState([]);

  useEffect(() => {
    const refresh = () => api('/api/health').then(setHealth);
    refresh();
    api('/api/scenarios').then((r) => r.ok && setScenarios(r.data));
    const id = setInterval(refresh, 30000);
    return () => clearInterval(id);
  }, []);

  const labels = { ...DEFAULT_LABELS, ...(health && health.ok ? health.data.label_display_names : {}) };

  return (
    <div className="shell">
      <StatusBar health={health} />
      <nav className="tabs" role="tablist" aria-label="Sections">
        {TABS.map(([id, label]) => (
          <button key={id} className="tab" role="tab" aria-selected={tab === id} onClick={() => setTab(id)}>{label}</button>
        ))}
      </nav>
      <main>
        {/* Keep the first two views mounted so typed input and the chat survive tab switches. */}
        <section hidden={tab !== 'check'} role="tabpanel"><CheckMessage scenarios={scenarios} labels={labels} /></section>
        <section hidden={tab !== 'conversation'} role="tabpanel"><Conversation scenarios={scenarios} labels={labels} /></section>
        {tab === 'results' && <section role="tabpanel"><ResultsView /></section>}
        {tab === 'about' && <section role="tabpanel"><About /></section>}
      </main>
    </div>
  );
}
