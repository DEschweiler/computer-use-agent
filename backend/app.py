#!/usr/bin/env python3
"""
Flask backend for Computer Use Agent
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import sys
import os
import threading
import queue
import time
import multiprocessing
import signal
from email.utils import formatdate

# Add parent directory to path to import agent
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

app = Flask(__name__)
CORS(app)

# Global queue for streaming logs


log_queue = queue.Queue()
# Global process handle for agent
agent_process = None
# Lock to prevent race conditions on agent_process
_agent_lock = threading.Lock()




def run_agent_task_proc(task, log_queue):
    """Run agent task in a subprocess, forwarding logs via a queue."""
    import sys
    import logging
    import traceback
    from agent import ComputerAgent, _load_config

    # The new agent uses logging (not print), so redirect_stdout/stderr won't
    # capture it — StreamHandler stores the stream reference at creation time.
    # Attach a custom handler directly to the agent logger instead.
    class QueueLogHandler(logging.Handler):
        def __init__(self, q):
            super().__init__()
            self.q = q
        def emit(self, record):
            try:
                self.q.put(self.format(record) + "\n")
            except Exception:
                pass

    queue_handler = QueueLogHandler(log_queue)
    queue_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    ))
    agent_logger = logging.getLogger("agent")
    agent_logger.addHandler(queue_handler)
    agent_logger.propagate = False  # avoid double-logging to real stderr

    try:
        log_queue.put("[SYSTEM] Initializing agent...\n")
        agent = ComputerAgent(_load_config())
        log_queue.put(f"[SYSTEM] Starting task: {task}\n")
        result = agent.run(task)
        actions = result["actions"]
        log_queue.put(f"[SYSTEM] Task completed with {len(actions)} actions\n")
        log_queue.put("[DONE]")
    except Exception as e:
        log_queue.put(f"[ERROR] {str(e)}\n")
        log_queue.put(f"[ERROR] {traceback.format_exc()}\n")
        log_queue.put("[DONE]")


@app.route('/api/task', methods=['POST'])
def start_task():
    """Start a new agent task"""
    data = request.json
    task = data.get('task', '')

    print(f"[API] Received task: {task}")

    if not task:
        return jsonify({'error': 'No task provided'}), 400

    with _agent_lock:
        # Clear the queue
        while not log_queue.empty():
            log_queue.get()

        global agent_process
        if agent_process is not None and agent_process.is_alive():
            return jsonify({'error': 'Agent already running'}), 409

        # Use multiprocessing.Queue for inter-process log streaming
        mp_log_queue = multiprocessing.Queue()
        agent_process = multiprocessing.Process(target=run_agent_task_proc, args=(task, mp_log_queue), daemon=True)
        agent_process.start()

    def forward_logs():
        while True:
            try:
                msg = mp_log_queue.get(timeout=30)
                log_queue.put(msg)
                if msg == '[DONE]':
                    break
            except Exception:
                break

    threading.Thread(target=forward_logs, daemon=True).start()

    print(f"[API] Task started in subprocess")
    return jsonify({'status': 'started', 'task': task})

@app.route('/api/stop', methods=['POST'])
def stop_task():
    """Immediately terminate the agent subprocess."""
    global agent_process
    if agent_process is not None and agent_process.is_alive():
        agent_process.terminate()
        agent_process.join(timeout=2)
        log_queue.put("[SYSTEM] Agent process terminated by user.\n")
        log_queue.put("[DONE]")
        agent_process = None
        return jsonify({'status': 'terminated'})
    else:
        log_queue.put("[SYSTEM] No agent process running.\n")
        log_queue.put("[DONE]")
        return jsonify({'status': 'not_running'})


@app.route('/api/screenshot')
def latest_screenshot():
    """Serve the latest annotated debug screenshot with Last-Modified / 304 support."""
    import os
    from flask import send_file
    path = os.path.join(os.environ.get('TEMP', '/tmp'), 'agent_screenshot_debug.png')
    if not os.path.exists(path):
        return jsonify({'error': 'No screenshot available yet'}), 404

    mtime = os.path.getmtime(path)
    last_modified = formatdate(mtime, usegmt=True)

    # Return 304 if the client already has the latest version.
    ims = request.headers.get('If-Modified-Since')
    if ims == last_modified:
        return '', 304

    resp = send_file(path, mimetype='image/png')
    resp.headers['Last-Modified'] = last_modified
    resp.headers['Cache-Control'] = 'no-cache'
    return resp


@app.route('/api/logs')
def stream_logs():
    """Stream logs using Server-Sent Events"""
    def generate():
        while True:
            try:
                # Wait for log message with timeout
                message = log_queue.get(timeout=30)

                if message == "[DONE]":
                    yield f"data: {message}\n\n"
                    break

                yield f"data: {message}\n\n"
            except queue.Empty:
                # Send keepalive
                yield ": keepalive\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/')
def index():
    """Serve the frontend HTML"""
    frontend_path = os.path.join(parent_dir, 'frontend.html')
    try:
        with open(frontend_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f'''
        <html>
            <head><title>Frontend Not Found</title></head>
            <body style="font-family: sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 2rem;">
                <h1>❌ Frontend not found</h1>
                <p>Could not find frontend.html at: {frontend_path}</p>
            </body>
        </html>
        ''', 404

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})


@app.route('/api/info')
def info():
    """Return backend model info"""
    profile = os.getenv('ACTIVE_PROFILE', 'gemma').upper()
    return jsonify({
        'model': os.getenv(f'{profile}_MODEL', 'unknown'),
        'endpoint': os.getenv(f'{profile}_ENDPOINT', 'unknown')
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤖 Computer Use Agent - Backend Started")
    print("="*60)
    print("\n✨ Open the web interface:")
    print("   👉 http://localhost:3000")
    print("\n" + "="*60 + "\n")
    app.run(debug=False, port=5000, threaded=True)
