import { useState, useEffect, useRef } from 'react';
import './App.css';

const API = 'http://localhost:5000';

const STATUS_META = {
  on_track:     { label: 'On track',     cls: 'ok' },
  off_track:    { label: 'Off track',    cls: 'warn' },
  stuck:        { label: 'Stuck',        cls: 'bad' },
  goal_reached: { label: 'Goal reached', cls: 'done' },
};

// Strip the "HH:MM:SS | LEVEL   | " prefix the backend log formatter prepends.
function stripPrefix(line) {
  return line.replace(/^\d{2}:\d{2}:\d{2}\s*\|\s*\w+\s*\|\s*/, '');
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [debugLogs, setDebugLogs] = useState([]);
  const [debugOpen, setDebugOpen] = useState(false);
  const [screenshotTs, setScreenshotTs] = useState(null);
  const [screenshotOk, setScreenshotOk] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);

  // Structured agent state, parsed from the log markers.
  const [goal, setGoal] = useState('');
  const [nav, setNav] = useState({ status: '', reasoning: '', objective: '' });
  const [objectives, setObjectives] = useState([]);
  const [action, setAction] = useState('');
  const [thoughts, setThoughts] = useState([]);
  const [perception, setPerception] = useState({ change: '', narration: '' });

  const messagesEndRef = useRef(null);
  const logsEndRef = useRef(null);
  const thoughtsEndRef = useRef(null);
  const eventSourceRef = useRef(null);
  const screenshotTimerRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/api/info`).then(r => r.json()).then(setModelInfo).catch(() => {});
  }, []);

  // Poll the screenshot every 2s while running (fallback; [SCREENSHOT_READY] also refreshes it).
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
  useEffect(() => { thoughtsEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [thoughts]);

  const closeEventSource = () => {
    if (eventSourceRef.current) { eventSourceRef.current.close(); eventSourceRef.current = null; }
  };

  const handleStop = async () => {
    closeEventSource();
    setIsRunning(false);
    try { await fetch(`${API}/api/stop`, { method: 'POST' }); } catch (_) {}
    setMessages(prev => [...prev, { type: 'error', content: 'Task stopped by user.' }]);
  };

  // Start a fresh conversation during usage: wipe backend context + all UI state.
  const handleNewConversation = async () => {
    closeEventSource();
    setIsRunning(false);
    try { await fetch(`${API}/api/reset`, { method: 'POST' }); } catch (_) {}
    setMessages([]);
    setDebugLogs([]);
    setGoal('');
    setNav({ status: '', reasoning: '', objective: '' });
    setObjectives([]);
    setAction('');
    setThoughts([]);
    setPerception({ change: '', narration: '' });
    setScreenshotOk(false);
    setScreenshotTs(null);
  };

  // Route a log line into the structured agent panels.
  const parseLine = (data) => {
    const msg = stripPrefix(data);
    if (msg.startsWith('[NAV]')) {
      if (msg.includes('objective:')) {
        const obj = msg.split('objective:').pop().trim();
        setNav(prev => ({ ...prev, objective: obj }));
        setObjectives(prev => (prev[prev.length - 1] === obj ? prev : [...prev, obj]));
      } else {
        const parts = msg.replace('[NAV]', '').trim().split('|');
        setNav(prev => ({ ...prev, status: parts[0].trim(), reasoning: parts.slice(1).join('|').trim() }));
      }
    } else if (msg.startsWith('[TOOL]')) {
      setAction(msg.replace('[TOOL]', '').replace('→', '').trim());
    } else if (msg.startsWith('[THOUGHT]')) {
      const t = msg.replace('[THOUGHT]', '').trim();
      setThoughts(prev => [...prev.slice(-40), t]);
    } else if (msg.startsWith('[CHANGE]')) {
      setPerception(prev => ({ ...prev, change: msg.replace('[CHANGE]', '').trim() }));
    } else if (msg.startsWith('[NARRATOR]')) {
      setPerception(prev => ({ ...prev, narration: msg.replace('[NARRATOR]', '').trim() }));
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
    setNav({ status: '', reasoning: '', objective: '' });
    setObjectives([]);
    setAction('');
    setThoughts([]);
    setPerception({ change: '', narration: '' });
    setScreenshotOk(false);
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
              : (question || finalAnswer || 'Task completed! Check debug logs for details.')
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
      <div className="header">
        <div className="header-title">
          <h1>🤖 Computer Use Agent</h1>
          <p>Multi-agent · navigator — actioner — narrator</p>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="new-convo-button"
            onClick={handleNewConversation}
            title="Wipe context and screenshots, start a fresh conversation"
          >
            🗑 New conversation
          </button>
          {modelInfo?.model && modelInfo.model !== 'unknown' && (
            <div className="model-badge">
              <span className="model-label">Model</span>
              <span className="model-name">{modelInfo.model}</span>
            </div>
          )}
        </div>
      </div>

      <div className="workspace">
        {/* LEFT — navigator + perception */}
        <div className="col col-left">
          <section className="panel nav-panel">
            <div className="panel-head">
              <span>🧭 Navigator</span>
              {statusMeta && <span className={`status-pill ${statusMeta.cls}`}>{statusMeta.label}</span>}
            </div>
            <div className="panel-body">
              <div className="field-label">Ultimate goal</div>
              <div className="goal-text">{goal || '—'}</div>
              <div className="field-label">Current objective</div>
              <div className="objective-text">{nav.objective || '—'}</div>
              {nav.reasoning && <div className="reasoning">“{nav.reasoning}”</div>}
              {objectives.length > 0 && (
                <>
                  <div className="field-label">Objective trail</div>
                  <ol className="objective-trail">
                    {objectives.map((o, i) => (
                      <li key={i} className={i === objectives.length - 1 ? 'current' : ''}>{o}</li>
                    ))}
                  </ol>
                </>
              )}
            </div>
          </section>

          <section className="panel perception-panel">
            <div className="panel-head"><span>👁 Perception</span></div>
            <div className="panel-body">
              <div className="field-label">Since last action</div>
              <div className="mono-text">{perception.change || '—'}</div>
              <div className="field-label">Visual observation (narrator)</div>
              <div className="narration-text">{perception.narration || '—'}</div>
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
            <span className="legend-item"><span className="swatch blue" /> Interactive region</span>
            <span className="legend-item"><span className="swatch cyan" /> Last click</span>
            <span className="legend-item"><span className="swatch magenta" /> Changed since last action</span>
          </div>
        </div>

        {/* RIGHT — actioner + conversation */}
        <div className="col col-right">
          <section className="panel actioner-panel">
            <div className="panel-head">
              <span>⚙️ Actioner</span>
              {isRunning && <span className="live-dot" title="running" />}
            </div>
            <div className="panel-body">
              <div className="field-label">Current action</div>
              <div className="action-text">{action || '—'}</div>
              <div className="field-label">Thought chain</div>
              <div className="thoughts">
                {thoughts.length === 0
                  ? <div className="muted">—</div>
                  : thoughts.map((t, i) => <div key={i} className="thought-item">{t}</div>)}
                <div ref={thoughtsEndRef} />
              </div>
            </div>
          </section>

          <section className="panel chat-panel">
            <div className="panel-head"><span>💬 Conversation</span></div>
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
          </section>
        </div>
      </div>

      {/* Input bar */}
      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter a task for the agent…"
          disabled={isRunning}
          className="input-field"
        />
        {isRunning
          ? <button type="button" onClick={handleStop} className="stop-button">⏹ Stop</button>
          : <button type="submit" disabled={!input.trim()} className="send-button">▶</button>}
      </form>

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
