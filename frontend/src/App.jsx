import { useState, useEffect, useRef } from 'react';
import './App.css';

const API = 'http://localhost:5000';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [debugLogs, setDebugLogs] = useState([]);
  const [debugExpanded, setDebugExpanded] = useState(false);
  const [panelHeight, setPanelHeight] = useState(300);
  const dragStateRef = useRef(null); // { startY, startHeight }
  const [screenshotTs, setScreenshotTs] = useState(null);
  const [screenshotOk, setScreenshotOk] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const messagesEndRef = useRef(null);
  const logsEndRef = useRef(null);
  const eventSourceRef = useRef(null);
  const screenshotTimerRef = useRef(null);

  const handleResizeMouseDown = (e) => {
    e.preventDefault();
    dragStateRef.current = { startY: e.clientY, startHeight: panelHeight };
    const onMove = (ev) => {
      const delta = dragStateRef.current.startY - ev.clientY;
      setPanelHeight(Math.max(80, dragStateRef.current.startHeight + delta));
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  };

  // Fetch model info once on mount
  useEffect(() => {
    fetch(`${API}/api/info`)
      .then(r => r.json())
      .then(setModelInfo)
      .catch(() => {});
  }, []);

  // Poll screenshot every 2 s while the agent is running
  useEffect(() => {
    if (isRunning) {
      screenshotTimerRef.current = setInterval(() => {
        setScreenshotTs(Date.now());
      }, 2000);
    } else {
      clearInterval(screenshotTimerRef.current);
    }
    return () => clearInterval(screenshotTimerRef.current);
  }, [isRunning]);

  const scrollToBottom = (ref) => {
    ref.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom(messagesEndRef);
  }, [messages]);

  useEffect(() => {
    if (debugExpanded) {
      scrollToBottom(logsEndRef);
    }
  }, [debugLogs, debugExpanded]);

  const closeEventSource = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  };

  const handleStop = async () => {
    closeEventSource();
    setIsRunning(false);
    try {
      await fetch(`${API}/api/stop`, { method: 'POST' });
    } catch (_) {}
    setMessages(prev => [...prev, { type: 'error', content: 'Task stopped by user.' }]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isRunning) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { type: 'user', content: userMessage }]);
    setDebugLogs([]);
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
              : (finalAnswer || 'Task completed! Check debug logs for details.')
          }]);
          return;
        }

        if (data.startsWith('[ABORTED]')) {
          aborted = true;
          return;
        }

        // Capture the agent's final answer — don't add it to debug logs
        if (data.startsWith('[ANSWER]')) {
          finalAnswer = data.slice('[ANSWER]'.length).trim();
          return;
        }

        setDebugLogs(prev => [...prev, data]);
      };

      eventSource.onerror = () => {
        eventSource.close();
        eventSourceRef.current = null;
        setIsRunning(false);
        setMessages(prev => [...prev, {
          type: 'error',
          content: 'Connection error. Check if backend is running.'
        }]);
      };

    } catch (error) {
      setIsRunning(false);
      setMessages(prev => [...prev, {
        type: 'error',
        content: `Error: ${error.message}`
      }]);
    }
  };

  return (
    <div className="app">
      <div className="header">
        <div className="header-title">
          <h1>🤖 Computer Use Agent</h1>
          <p>
            {modelInfo?.model && modelInfo.model !== 'unknown'
              ? modelInfo.model
              : 'AI-powered automation assistant'}
          </p>
        </div>
      </div>

      <div className="main-container">
        {/* Chat Area */}
        <div className="chat-container">
          <div className="messages">
            {messages.length === 0 && (
              <div className="welcome">
                <h2>Welcome!</h2>
                <p>Enter a task below to get started.</p>
                <div className="examples">
                  <p>Example tasks:</p>
                  <ul>
                    <li>Create a new patient record</li>
                    <li>Fill out the registration form</li>
                    <li>Navigate to the dashboard</li>
                  </ul>
                </div>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.type}`}>
                <div className="message-label">
                  {msg.type === 'user' ? '👤 You' :
                   msg.type === 'error' ? '❌ Error' : '🤖 Agent'}
                </div>
                <div className="message-content">{msg.content}</div>
              </div>
            ))}

            {isRunning && (
              <div className="message agent">
                <div className="message-label">🤖 Agent</div>
                <div className="message-content">
                  <div className="loading">
                    <span></span><span></span><span></span>
                  </div>
                  Working on it...
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSubmit} className="input-form">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Enter a task for the agent..."
              disabled={isRunning}
              className="input-field"
            />
            {isRunning ? (
              <button type="button" onClick={handleStop} className="stop-button">
                ⏹ Stop
              </button>
            ) : (
              <button type="submit" disabled={!input.trim()} className="send-button">
                ▶
              </button>
            )}
          </form>
        </div>

        {/* Debug Panels */}
        <div className={`debug-row ${debugExpanded ? 'expanded' : ''}`}>
          <div className="debug-row-header" onClick={() => setDebugExpanded(!debugExpanded)}>
            <span>🔍 Debug Logs {debugLogs.length > 0 && `(${debugLogs.length})`}</span>
            <span className="debug-row-divider" />
            <span>📷 Latest Screenshot</span>
            <span className="toggle">{debugExpanded ? '▼' : '▲'}</span>
          </div>

          {debugExpanded && (
            <>
            <div className="debug-resize-handle" onMouseDown={handleResizeMouseDown} />
            <div className="debug-row-content" style={{ height: panelHeight }}>
              {/* Logs */}
              <div className="debug-content">
                {debugLogs.length === 0 ? (
                  <div className="debug-empty">No logs yet. Start a task to see debug output.</div>
                ) : (
                  <pre className="debug-logs">
                    {debugLogs.map((log, idx) => (
                      <div
                        key={idx}
                        className={`log-line${log.includes('[THOUGHT]') ? ' thought' : ''}${log.includes('[TOOL]') ? ' tool-call' : ''}`}
                      >{log}</div>
                    ))}
                    <div ref={logsEndRef} />
                  </pre>
                )}
              </div>

              {/* Screenshot */}
              <div className="debug-content screenshot-content">
                {screenshotTs ? (
                  <>
                    <img
                      key={screenshotTs}
                      src={`${API}/api/screenshot?t=${screenshotTs}`}
                      alt="Latest agent screenshot"
                      className="debug-screenshot"
                      style={{ display: screenshotOk ? 'block' : 'none' }}
                      onLoad={() => setScreenshotOk(true)}
                      onError={() => setScreenshotOk(false)}
                    />
                    {!screenshotOk && (
                      <div className="debug-empty">Waiting for screenshot...</div>
                    )}
                  </>
                ) : (
                  <div className="debug-empty">Start a task to see screenshots.</div>
                )}
              </div>
            </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
