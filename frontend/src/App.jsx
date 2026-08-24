import { useState, useEffect, useRef } from 'react';
import './App.css';

const API = 'http://localhost:5000';

const STATUS_META = {
  on_track:     { label: 'On track',     cls: 'ok' },
  off_track:    { label: 'Off track',    cls: 'warn' },
  stuck:        { label: 'Stuck',        cls: 'bad' },
  goal_reached: { label: 'Goal reached', cls: 'done' },
};

// Icon + accent class per activity-timeline event kind.
const EVENT_META = {
  iter:      { icon: '',   cls: 'ev-iter' },
  objective: { icon: '🎯', cls: 'ev-objective' },
  nav:       { icon: '⚠',  cls: 'ev-nav' },
  supervise: { icon: '🧭', cls: 'ev-supervise' },
  retry:     { icon: '⏳', cls: 'ev-retry' },
  intervene: { icon: '🛑', cls: 'ev-intervene' },
  todo:      { icon: '📋', cls: 'ev-todo' },
  action:    { icon: '⚙',  cls: 'ev-action' },
  thought:   { icon: '💭', cls: 'ev-thought' },
  result:    { icon: '↳',  cls: 'ev-result' },
  change:    { icon: '👁', cls: 'ev-change' },
  narration: { icon: '👁', cls: 'ev-narration' },
};

// Glyph per todo-item status for the supervisor plan checklist.
const TODO_MARKS = { pending: '○', in_progress: '▶', done: '✓', failed: '✗', skipped: '−' };

// Strip the "HH:MM:SS | LEVEL   | " prefix the backend log formatter prepends.
function stripPrefix(line) {
  return line.replace(/^\d{2}:\d{2}:\d{2}\s*\|\s*\w+\s*\|\s*/, '');
}

// Cursor speed: how fast the agent's pointer travels to a click target.
// 1% = a deliberate crawl, 100% = near-instant. Live — the agent re-reads it
// mid-run, so it stays enabled while a task is running.
function CursorSpeedSlider({ value, min, onChange }) {
  const readout = `${Math.round(value)}%`;
  return (
    <div className="speed-picker" title="How fast the agent's cursor travels to a click target">
      <span className="model-picker-label">Cursor Speed</span>
      <input
        type="range"
        className="speed-range"
        min={min} max="100" step="1"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      <span className="speed-readout">{readout}</span>
    </div>
  );
}

// A styled model picker. `value` is a profile id (or 'SAME' when includeSame).
function ModelDropdown({ label, models, value, onSelect, disabled, includeSame, onOpen }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  const selected = models.find((m) => m.profile === value) || null;
  const isSame = includeSame && (value === 'SAME' || value === '');
  const triggerName = isSame ? 'Same as actioner'
    : (selected ? selected.model : (models.length ? 'Select model' : 'Loading…'));
  const pick = (p) => { onSelect(p); setOpen(false); };

  return (
    <div className="model-picker">
      <span className="model-picker-label">{label}</span>
      <div className={`model-dropdown ${open ? 'open' : ''}`} ref={ref}>
        <button
          type="button"
          className={`model-trigger${!isSame && selected && !selected.available ? ' offline' : ''}`}
          disabled={disabled}
          onClick={() => { if (!disabled) { const n = !open; if (n && onOpen) onOpen(); setOpen(n); } }}
          title={disabled ? 'Stop the current task to switch models' : `Choose the ${label.toLowerCase()} model for the next task`}
        >
          <span className={`model-dot ${isSame ? '' : (selected ? (selected.available ? 'on' : 'off') : '')}`} />
          <span className="model-trigger-name">{triggerName}</span>
          <span className="model-caret" aria-hidden="true">▾</span>
        </button>
        {open && (
          <div className="model-menu" role="listbox">
            {includeSame && (
              <button
                type="button"
                className={`model-option${isSame ? ' active' : ''}`}
                onClick={() => pick('SAME')}
              >
                <span className="model-dot" />
                <span className="model-option-name">Same as actioner</span>
              </button>
            )}
            {models.length === 0 && <div className="model-empty">No models found in .env</div>}
            {models.map((m) => (
              <button
                type="button"
                key={m.profile}
                role="option"
                aria-selected={m.profile === value}
                className={`model-option${m.profile === value ? ' active' : ''}${m.available ? '' : ' offline'}`}
                disabled={!m.available}
                onClick={() => { if (m.available) pick(m.profile); }}
              >
                <span className={`model-dot ${m.available ? 'on' : 'off'}`} />
                <span className="model-option-name">{m.model}</span>
                {m.vision === false && <span className="model-status">text</span>}
                {!m.available && <span className="model-status">offline</span>}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [debugLogs, setDebugLogs] = useState([]);
  const [debugOpen, setDebugOpen] = useState(false);
  const [screenshotTs, setScreenshotTs] = useState(null);
  const [screenshotOk, setScreenshotOk] = useState(false);
  const [models, setModels] = useState([]);
  const [activeProfile, setActiveProfile] = useState('');
  const [navActiveProfile, setNavActiveProfile] = useState('SAME');
  const [cursorSpeed, setCursorSpeed] = useState(30);
  const [cursorSpeedMin, setCursorSpeedMin] = useState(1);

  // Supervisor (header) state, parsed from [NAV] markers.
  const [goal, setGoal] = useState('');
  const [nav, setNav] = useState({ status: '', reasoning: '', objective: '' });
  const [objectives, setObjectives] = useState([]);
  const [issue, setIssue] = useState('');
  // Supervisor plan (todo list), parsed from [TODOS] JSON markers.
  const [todos, setTodos] = useState([]);
  // Unified activity timeline: chronological, typed events.
  const [timeline, setTimeline] = useState([]);

  const messagesEndRef = useRef(null);
  const logsEndRef = useRef(null);
  const timelineEndRef = useRef(null);
  const todosEndRef = useRef(null);
  const eventSourceRef = useRef(null);
  const screenshotTimerRef = useRef(null);
  const iterRef = useRef(0);
  const evIdRef = useRef(0);

  const loadModels = () => {
    fetch(`${API}/api/models`).then(r => r.json()).then(data => {
      setModels(data.models || []);
      setActiveProfile(prev => prev || data.active || '');
      setNavActiveProfile(prev => (prev && prev !== 'SAME' ? prev : (data.nav_active || 'SAME')));
    }).catch(() => {});
  };
  useEffect(() => {
    loadModels();
    const id = setInterval(loadModels, 15000);  // refresh availability
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    fetch(`${API}/api/cursor_speed`).then(r => r.json()).then(d => {
      if (typeof d.speed === 'number') setCursorSpeed(d.speed);
      if (typeof d.min === 'number') setCursorSpeedMin(d.min);
    }).catch(() => {});
  }, []);

  const handleCursorSpeedChange = (speed) => {
    setCursorSpeed(speed);  // move the thumb immediately; the POST is fire-and-forget
    fetch(`${API}/api/cursor_speed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speed }),
    }).catch(() => {});
  };

  const handleSelectModel = (profile, role = 'actioner') => {
    if (role === 'navigator') setNavActiveProfile(profile);
    else setActiveProfile(profile);
    fetch(`${API}/api/select_model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ profile, role }),
    }).catch(() => {});
  };

  useEffect(() => {
    if (isRunning) {
      screenshotTimerRef.current = setInterval(() => setScreenshotTs(Date.now()), 2000);
    } else {
      clearInterval(screenshotTimerRef.current);
    }
    return () => clearInterval(screenshotTimerRef.current);
  }, [isRunning]);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);
  useEffect(() => { if (debugOpen) logsEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [debugLogs, debugOpen]);
  useEffect(() => { timelineEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [timeline]);
  // Keep the newest todos in view — the active item is always near the bottom
  // of a growing plan. block:'nearest' scrolls only the panel body, not the page.
  useEffect(() => { todosEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }, [todos]);

  const closeEventSource = () => {
    if (eventSourceRef.current) { eventSourceRef.current.close(); eventSourceRef.current = null; }
  };

  const pushEvent = (kind, text, extra = {}) => {
    const id = evIdRef.current++;
    setTimeline(prev => [...prev.slice(-250), { id, kind, text, iter: iterRef.current, ...extra }]);
  };

  const resetRunState = () => {
    iterRef.current = 0;
    evIdRef.current = 0;
    setNav({ status: '', reasoning: '', objective: '' });
    setObjectives([]);
    setIssue('');
    setTodos([]);
    setTimeline([]);
    setScreenshotOk(false);
  };

  const handleStop = async () => {
    closeEventSource();
    setIsRunning(false);
    try { await fetch(`${API}/api/stop`, { method: 'POST' }); } catch (_) {}
    setMessages(prev => [...prev, { type: 'error', content: 'Task stopped by user.' }]);
  };

  const handleNewConversation = async () => {
    closeEventSource();
    setIsRunning(false);
    try { await fetch(`${API}/api/reset`, { method: 'POST' }); } catch (_) {}
    setMessages([]);
    setDebugLogs([]);
    setGoal('');
    resetRunState();
    setScreenshotTs(null);
  };

  // Route a log line into the supervisor header + activity timeline.
  const parseLine = (data) => {
    const msg = stripPrefix(data);
    if (msg.startsWith('[ITER]')) {
      const n = parseInt(msg.replace('[ITER]', '').trim(), 10);
      if (!Number.isNaN(n)) iterRef.current = n;
      pushEvent('iter', `Step ${iterRef.current}`);
    } else if (msg.startsWith('[TODOS]')) {
      // Full plan state as one-line JSON — drives the supervisor checklist.
      try { setTodos(JSON.parse(msg.slice('[TODOS]'.length).trim())); } catch (_) {}
    } else if (msg.startsWith('[TODO]')) {
      pushEvent('todo', msg.replace('[TODO]', '').trim());
    } else if (msg.startsWith('[SUPERVISE]')) {
      pushEvent('supervise', `Supervisor: ${msg.replace('[SUPERVISE]', '').trim()}`);
    } else if (msg.startsWith('[RETRY]')) {
      pushEvent('retry', msg.replace('[RETRY]', '').trim());
    } else if (msg.startsWith('[NAV]')) {
      if (msg.includes('objective:')) {
        const obj = msg.split('objective:').pop().trim();
        setNav(prev => ({ ...prev, objective: obj }));
        setObjectives(prev => (prev[prev.length - 1] === obj ? prev : [...prev, obj]));
        pushEvent('objective', obj);
      } else {
        const parts = msg.replace('[NAV]', '').trim().split('|');
        const status = parts[0].trim();
        const reasoning = parts.slice(1).join('|').trim();
        setNav(prev => ({ ...prev, status, reasoning }));
        // Only surface concerning verdicts inline; on_track just updates the header.
        if (status && status !== 'on_track' && reasoning) pushEvent('nav', `${status}: ${reasoning}`);
      }
    } else if (msg.startsWith('[ISSUE]')) {
      setIssue(msg.slice('[ISSUE]'.length).trim());
    } else if (msg.startsWith('[INTERVENE]')) {
      pushEvent('intervene', msg.replace('[INTERVENE]', '').trim());
    } else if (msg.startsWith('[TOOL]')) {
      pushEvent('action', msg.replace('[TOOL]', '').replace('→', '').trim());
    } else if (msg.startsWith('[THOUGHT]')) {
      pushEvent('thought', msg.replace('[THOUGHT]', '').trim());
    } else if (msg.startsWith('[RESULT]')) {
      const t = msg.replace('[RESULT]', '').trim();
      pushEvent('result', t, { error: t.toLowerCase().startsWith('error') });
    } else if (msg.startsWith('[PROGRESS]')) {
      const changed = msg.includes('changed') && !msg.includes('nochange');
      pushEvent('result', changed ? 'screen changed' : 'no visible change', { progress: true, changed });
    } else if (msg.startsWith('[CHANGE]')) {
      pushEvent('change', msg.replace('[CHANGE]', '').trim());
    } else if (msg.startsWith('[NARRATOR]')) {
      pushEvent('narration', msg.replace('[NARRATOR]', '').trim());
    } else if (msg.includes('[SCREENSHOT_READY]')) {
      setScreenshotTs(Date.now());
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isRunning) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { type: 'user', content: userMessage }]);
    setDebugLogs([]);
    setGoal(userMessage);
    resetRunState();
    setScreenshotTs(Date.now());
    setIsRunning(true);

    try {
      const response = await fetch(`${API}/api/task`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task: userMessage })
      });
      if (!response.ok) throw new Error('Failed to start task');

      let finalAnswer = null;
      let question = null;
      let aborted = false;
      const eventSource = new EventSource(`${API}/api/logs`);
      eventSourceRef.current = eventSource;

      eventSource.onmessage = (event) => {
        const data = event.data;

        if (data === '[DONE]') {
          eventSource.close();
          eventSourceRef.current = null;
          setIsRunning(false);
          setMessages(prev => [...prev, {
            type: aborted ? 'error' : 'agent',
            content: aborted
              ? 'Task aborted: the screen stopped changing. Check the debug logs.'
              : (question || finalAnswer || 'Run ended without a verified result — check the activity log for what happened.')
          }]);
          return;
        }
        if (data.startsWith('[ABORTED]')) { aborted = true; return; }
        if (data.startsWith('[QUESTION]')) { question = data.slice('[QUESTION]'.length).trim(); return; }
        if (data.startsWith('[ANSWER]')) { finalAnswer = data.slice('[ANSWER]'.length).trim(); return; }

        parseLine(data);
        setDebugLogs(prev => [...prev, data]);
      };

      eventSource.onerror = () => {
        eventSource.close();
        eventSourceRef.current = null;
        setIsRunning(false);
        setMessages(prev => [...prev, { type: 'error', content: 'Connection error. Check if backend is running.' }]);
      };
    } catch (error) {
      setIsRunning(false);
      setMessages(prev => [...prev, { type: 'error', content: `Error: ${error.message}` }]);
    }
  };

  const statusMeta = STATUS_META[nav.status] || null;

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div className="brand-text">
            <span className="brand-title">Computer Use Agent</span>
            <span className="brand-sub">Supervisor · Operator · Perception</span>
          </div>
        </div>
        <div className="header-actions">
          <CursorSpeedSlider value={cursorSpeed} min={cursorSpeedMin} onChange={handleCursorSpeedChange} />
          {/* Actioner (and narrator) consume screenshots → vision profiles only.
              Supervisor is text-only reasoning → every profile qualifies. */}
          <ModelDropdown
            label="Actioner"
            models={models.filter((m) => m.vision !== false)}
            value={activeProfile}
            disabled={isRunning}
            onOpen={loadModels}
            onSelect={(p) => handleSelectModel(p, 'actioner')}
          />
          <ModelDropdown
            label="Supervisor"
            models={models}
            value={navActiveProfile}
            disabled={isRunning}
            includeSame
            onOpen={loadModels}
            onSelect={(p) => handleSelectModel(p, 'navigator')}
          />
        </div>
      </header>

      <div className="workspace">
        {/* LEFT — supervisor status (compact) + activity timeline (tall) */}
        <div className="col col-left">
          <section className="panel nav-panel supervisor">
            <div className="panel-head">
              <span>🧭 Supervisor</span>
              {statusMeta && <span className={`status-pill ${statusMeta.cls}`}>{statusMeta.label}</span>}
            </div>
            <div className="panel-body supervisor-body">
              <div className="field-label">Goal</div>
              <div className="goal-text">{goal || '—'}</div>
              <div className="field-label">Plan</div>
              {todos.length === 0 ? (
                <div className="muted">No plan yet.</div>
              ) : (
                <ul className="todo-list">
                  {todos.map((t) => (
                    <li
                      key={t.id}
                      className={`todo-item ${t.status}${t.corrective ? ' corrective' : ''}`}
                      title={t.corrective ? 'Corrective item (fixes a mistake)' : undefined}
                    >
                      <span className="todo-mark">{TODO_MARKS[t.status] || '○'}</span>
                      <span className="todo-text">{t.corrective ? '⚠ ' : ''}{t.text}</span>
                    </li>
                  ))}
                </ul>
              )}
              {nav.reasoning && <div className="reasoning">“{nav.reasoning}”</div>}
              {issue && (
                <>
                  <div className="field-label">Outstanding issue</div>
                  <div className="issue-text">⚠ {issue}</div>
                </>
              )}
              <div ref={todosEndRef} />
            </div>
          </section>

          <section className="panel timeline-panel">
            <div className="panel-head">
              <span>📜 Activity</span>
              {isRunning && <span className="live-dot" title="running" />}
            </div>
            <div className="panel-body timeline">
              {timeline.length === 0 && <div className="muted">Actions, thoughts, perception, and supervisor steering appear here.</div>}
              {timeline.map((ev) => {
                const meta = EVENT_META[ev.kind] || EVENT_META.result;
                if (ev.kind === 'iter') {
                  return <div key={ev.id} className="tl-divider"><span>{ev.text}</span></div>;
                }
                const cls = ev.kind === 'result' && ev.error ? 'ev-result err'
                  : ev.kind === 'result' && ev.progress ? (ev.changed ? 'ev-result ok' : 'ev-result flat')
                  : meta.cls;
                return (
                  <div key={ev.id} className={`tl-event ${cls}`}>
                    <span className="tl-icon">{meta.icon}</span>
                    <span className="tl-text">{ev.text}</span>
                  </div>
                );
              })}
              <div ref={timelineEndRef} />
            </div>
          </section>
        </div>

        {/* CENTER — the agent's view */}
        <div className="col col-center">
          <div className="screenshot-wrap">
            {screenshotTs ? (
              <>
                <img
                  key={screenshotTs}
                  src={`${API}/api/screenshot?t=${screenshotTs}`}
                  alt="Agent screenshot"
                  className="screenshot"
                  style={{ display: screenshotOk ? 'block' : 'none' }}
                  onLoad={() => setScreenshotOk(true)}
                  onError={() => setScreenshotOk(false)}
                />
                {!screenshotOk && <div className="placeholder">Waiting for screenshot…</div>}
              </>
            ) : (
              <div className="placeholder">Start a task to see the agent's view.</div>
            )}
          </div>
          <div className="legend">
            <span className="legend-item"><span className="swatch orange" /> OCR text</span>
            <span className="legend-item"><span className="swatch blue" /> Interactive control (incl. filled fields)</span>
            <span className="legend-item"><span className="swatch cyan" /> Last click</span>
            <span className="legend-item"><span className="swatch magenta" /> Changed since last action</span>
          </div>
        </div>

        {/* RIGHT — conversation (in focus) + input */}
        <div className="col col-right">
          <section className="panel chat-panel">
            <div className="panel-head">
              <span>💬 Conversation</span>
              <button
                type="button"
                className="new-convo-button compact"
                onClick={handleNewConversation}
                title="Wipe context and screenshots, start a fresh conversation"
              >
                🗑 New
              </button>
            </div>
            <div className="panel-body chat-body">
              {messages.length === 0 && <div className="muted">Enter a task below to begin.</div>}
              {messages.map((m, i) => (
                <div key={i} className={`chat-msg ${m.type}`}>
                  <div className="chat-msg-label">
                    {m.type === 'user' ? '👤 You' : m.type === 'error' ? '❌ Error' : '🤖 Agent'}
                  </div>
                  <div>{m.content}</div>
                </div>
              ))}
              {isRunning && (
                <div className="chat-msg agent">
                  <div className="chat-msg-label">🤖 Agent</div>
                  <div><span className="loading"><span></span><span></span><span></span></span> Working…</div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
            <form onSubmit={handleSubmit} className="input-form">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Enter a task…"
                disabled={isRunning}
                className="input-field"
              />
              {isRunning
                ? <button type="button" onClick={handleStop} className="stop-button">⏹</button>
                : <button type="submit" disabled={!input.trim()} className="send-button">▶</button>}
            </form>
          </section>
        </div>
      </div>

      {/* Raw logs — collapsible drawer for power users */}
      <div className={`debug-drawer ${debugOpen ? 'open' : ''}`}>
        <div className="debug-drawer-head" onClick={() => setDebugOpen(o => !o)}>
          <span>🔍 Raw debug logs {debugLogs.length > 0 && `(${debugLogs.length})`}</span>
          <span className="toggle">{debugOpen ? '▼' : '▲'}</span>
        </div>
        {debugOpen && (
          <div className="debug-drawer-body">
            {debugLogs.length === 0
              ? <div className="muted">No logs yet.</div>
              : (
                <pre className="debug-logs">
                  {debugLogs.map((log, idx) => (
                    <div
                      key={idx}
                      className={`log-line${log.includes('[THOUGHT]') ? ' thought' : ''}${log.includes('[TOOL]') ? ' tool-call' : ''}${log.includes('[NAV]') ? ' nav' : ''}${log.includes('[CHANGE]') ? ' change' : ''}${log.includes('[NARRATOR]') ? ' narrator' : ''}`}
                    >{log}</div>
                  ))}
                  <div ref={logsEndRef} />
                </pre>
              )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
