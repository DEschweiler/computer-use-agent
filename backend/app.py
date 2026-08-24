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
from concurrent.futures import ThreadPoolExecutor
import requests
import urllib3
from dotenv import load_dotenv, dotenv_values

# Add parent directory to path to import agent
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

_ENV_PATH = os.path.join(parent_dir, '.env')
load_dotenv(_ENV_PATH)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)
CORS(app)

# Global queue for streaming logs


log_queue = queue.Queue()
# Global process handle for agent
agent_process = None
# Lock to prevent race conditions on agent_process
_agent_lock = threading.Lock()
# Model profiles selected in the UI (override the .env defaults for the next task).
# _selected_nav_profile == "SAME" means the supervisor reuses the actioner's model.
_selected_profile = None
_selected_nav_profile = None


# --- Model profiles (from .env) ------------------------------------------- #

def _env_config():
    """Merged view of .env + process env (process env wins)."""
    cfg = dict(dotenv_values(_ENV_PATH))
    cfg.update(os.environ)
    return cfg


def _list_profiles(cfg=None):
    """Every profile with both a *_MODEL and a *_ENDPOINT in the config.

    Each profile carries a 'vision' capability flag (from <PROFILE>_VISION,
    default true). Text-only profiles (VISION=0) may serve as the SUPERVISOR
    but never as the actioner — the actioner and narrator consume screenshots."""
    cfg = cfg or _env_config()
    profiles = []
    for key, val in cfg.items():
        if key.endswith('_MODEL') and val:
            prof = key[:-len('_MODEL')]
            endpoint = cfg.get(f'{prof}_ENDPOINT')
            if prof and endpoint:
                vision_flag = str(cfg.get(f'{prof}_VISION', '1')).strip().lower()
                profiles.append({
                    'profile': prof,
                    'model': val,
                    'endpoint': endpoint,
                    'vision': vision_flag not in ('0', 'false', 'no', 'off'),
                })
    profiles.sort(key=lambda p: p['profile'])
    return profiles


def _active_profile(cfg=None):
    cfg = cfg or _env_config()
    return (_selected_profile or cfg.get('ACTIVE_PROFILE') or '').upper()


def _active_nav_profile(cfg=None):
    """Supervisor profile: UI selection, else NAVIGATOR_PROFILE from .env, else map
    a direct NAVIGATOR_MODEL to a profile, else 'SAME' (reuse the actioner model)."""
    cfg = cfg or _env_config()
    if _selected_nav_profile is not None:
        return _selected_nav_profile.upper()
    val = (cfg.get('NAVIGATOR_PROFILE') or '').upper()
    if val and val not in ('SAME', 'ACTIONER', 'NONE'):
        return val
    nm = cfg.get('NAVIGATOR_MODEL')
    if nm:
        for p in _list_profiles(cfg):
            if p['model'] == nm:
                return p['profile'].upper()
    return 'SAME'


def _endpoint_base(endpoint):
    """Strip a trailing /chat/completions so we can query the sibling /models."""
    return (endpoint or '').rsplit('/chat/completions', 1)[0].rstrip('/')


def _live_models_for_base(base, api_key):
    """Set of model ids the server at *base* is actually serving (via GET /models),
    or an empty set if the endpoint is unreachable."""
    if not base:
        return set()
    try:
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        resp = requests.get(base + '/models', headers=headers, timeout=6, verify=False)
        resp.raise_for_status()
        return {m.get('id') for m in resp.json().get('data', [])}
    except Exception:
        return set()


def _availability(profiles, cfg):
    """Map each profile -> bool: is its model id actually being served? Groups by
    endpoint base so each proxy's /models is fetched once (concurrently)."""
    bases = {}  # base -> api_key (first profile that uses it)
    for p in profiles:
        base = _endpoint_base(p['endpoint'])
        if base and base not in bases:
            bases[base] = cfg.get(f"{p['profile']}_API_KEY", '')
    live_by_base = {}
    if bases:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for base, live in ex.map(lambda b: (b, _live_models_for_base(b, bases[b])), list(bases)):
                live_by_base[base] = live
    return {
        p['profile']: p['model'] in live_by_base.get(_endpoint_base(p['endpoint']), set())
        for p in profiles
    }


def _temp_dir():
    return os.environ.get('TEMP', '/tmp')


def _clear_conversation(delete_screenshot=True):
    """Delete the persisted conversation context (and optionally the last
    screenshot). Used on startup (always fresh) and by /api/reset."""
    paths = [os.path.join(_temp_dir(), 'agent_session.json')]
    if delete_screenshot:
        paths.append(os.path.join(_temp_dir(), 'agent_screenshot_debug.png'))
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"[API] Could not remove {p}: {exc}")




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
        # Check if the run ended with an abort
        aborted = actions and actions[-1].get("error", "").startswith("aborted")
        # A clarifying question ends the run and awaits the user's reply; it
        # takes precedence over any answer text.
        question = result.get("question")
        # Extract the final answer from the last verified task_complete action
        final_answer = None
        for action in reversed(actions):
            if action.get("action") == "task_complete" and action.get("verified"):
                final_answer = action.get("claim", "").strip()
                break
        log_queue.put(f"[SYSTEM] Task completed with {len(actions)} actions\n")
        if aborted:
            log_queue.put("[ABORTED]\n")
        elif question:
            log_queue.put(f"[QUESTION] {question}\n")
        elif final_answer:
            log_queue.put(f"[ANSWER] {final_answer}\n")
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

        # Delete the previous run's debug screenshot immediately, so the UI shows
        # "waiting" instead of a stale frame until the agent writes a fresh one.
        stale_shot = os.path.join(os.environ.get('TEMP', '/tmp'), 'agent_screenshot_debug.png')
        try:
            os.remove(stale_shot)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"[API] Could not remove stale screenshot: {exc}")

        global agent_process
        if agent_process is not None and agent_process.is_alive():
            return jsonify({'error': 'Agent already running'}), 409

        # Apply the UI-selected model profiles so the spawned agent picks them up
        # (the child inherits os.environ; load_dotenv won't override existing keys).
        if _selected_profile:
            os.environ['ACTIVE_PROFILE'] = _selected_profile
        if _selected_nav_profile is not None:
            # 'SAME' → supervisor reuses the actioner model (agent maps it to no override).
            os.environ['NAVIGATOR_PROFILE'] = _selected_nav_profile

        # Use multiprocessing.Queue for inter-process log streaming
        mp_log_queue = multiprocessing.Queue()
        agent_process = multiprocessing.Process(target=run_agent_task_proc, args=(task, mp_log_queue), daemon=True)
        agent_process.start()

    def forward_logs():
        # NOTE: a quiet gap is NOT an error — a reasoning-model LLM call can
        # legitimately produce no log line for minutes. The forwarder must only
        # exit on [DONE] or when the agent process is actually gone; breaking
        # on queue.Empty (as this once did) silently froze the activity feed
        # for the rest of the run while the agent kept working.
        while True:
            try:
                msg = mp_log_queue.get(timeout=30)
            except queue.Empty:
                proc = agent_process
                if proc is None or not proc.is_alive():
                    break
                continue
            except Exception:
                break
            log_queue.put(msg)
            if msg == '[DONE]':
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


@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    """Start a fresh conversation during usage: stop any running task and wipe
    the persisted context + last screenshot. (Startup is always fresh already.)"""
    global agent_process
    with _agent_lock:
        if agent_process is not None and agent_process.is_alive():
            agent_process.terminate()
            agent_process.join(timeout=2)
            agent_process = None
        _clear_conversation(delete_screenshot=True)
        while not log_queue.empty():
            log_queue.get()
    print("[API] Conversation reset — context and screenshot cleared.")
    return jsonify({'status': 'reset'})


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
    """Return backend model info for the active profile."""
    cfg = _env_config()
    profile = _active_profile(cfg) or 'GEMMA'
    return jsonify({
        'model': cfg.get(f'{profile}_MODEL', 'unknown'),
        'endpoint': cfg.get(f'{profile}_ENDPOINT', 'unknown'),
        'profile': profile,
    })


@app.route('/api/models')
def list_models():
    """List all model profiles from .env, mark the active one, and probe which
    endpoints are currently reachable so the UI can grey out offline ones."""
    cfg = _env_config()
    profiles = _list_profiles(cfg)
    avail = _availability(profiles, cfg)  # model id must be served, not just host reachable
    models = [
        {'profile': p['profile'], 'model': p['model'],
         'available': avail.get(p['profile'], False), 'vision': p['vision']}
        for p in profiles
    ]
    return jsonify({
        'models': models,
        'active': _active_profile(cfg),          # actioner
        'nav_active': _active_nav_profile(cfg),  # supervisor ('SAME' = reuse actioner)
    })


@app.route('/api/select_model', methods=['POST'])
def select_model():
    """Select the profile used for the NEXT task. role='actioner' (default) sets the
    actioner model; role='navigator' sets the supervisor model ('SAME' = reuse actioner)."""
    global _selected_profile, _selected_nav_profile
    data = request.json or {}
    role = (data.get('role') or 'actioner').strip().lower()
    prof = (data.get('profile') or '').strip().upper()
    profiles = _list_profiles()
    known = {p['profile'].upper() for p in profiles}
    vision_by_prof = {p['profile'].upper(): p['vision'] for p in profiles}
    if role == 'navigator':
        # Supervisor is text-only reasoning — any profile qualifies.
        if prof != 'SAME' and prof not in known:
            return jsonify({'error': f'unknown profile: {prof}'}), 400
        _selected_nav_profile = prof or 'SAME'
        print(f"[API] Supervisor profile selected: {_selected_nav_profile}")
        return jsonify({'status': 'ok', 'nav_active': _selected_nav_profile})
    if prof not in known:
        return jsonify({'error': f'unknown profile: {prof}'}), 400
    if not vision_by_prof.get(prof, True):
        # The actioner (and the narrator, which reuses its model) consume
        # screenshots — a text-only profile cannot drive them.
        return jsonify({'error': f'profile {prof} is text-only (VISION=0) — '
                                 'it can only be used as the supervisor'}), 400
    _selected_profile = prof
    print(f"[API] Actioner profile selected: {prof}")
    return jsonify({'status': 'ok', 'active': prof})


# --- Cursor speed ---------------------------------------------------------- #
# How fast the agent's cursor travels to a click target: 1 = a deliberate crawl,
# 100 = near-instant. Floored at 1 rather than 0 because 0% would read as "the
# cursor doesn't move", which is never what it means. Kept in a file the agent
# re-reads on every glide rather than in env only, so dragging the slider takes
# effect during a running task and not just on the next one.

_SPEED_DEFAULT = 30.0
_SPEED_MIN = 1.0


def _cursor_speed_path():
    return os.path.join(_temp_dir(), 'agent_cursor_speed.txt')


def _cursor_speed_default():
    """Fall back to .env / process env, matching agent.py's own default."""
    try:
        return max(_SPEED_MIN, min(100.0, float(_env_config().get('MOUSE_GLIDE_SPEED_PCT', _SPEED_DEFAULT))))
    except (TypeError, ValueError):
        return _SPEED_DEFAULT


@app.route('/api/cursor_speed')
def get_cursor_speed():
    """Current cursor speed, 1 (slowest) - 100 (near-instant)."""
    try:
        with open(_cursor_speed_path(), encoding='utf-8') as fh:
            value = max(_SPEED_MIN, min(100.0, float(fh.read().strip())))
    except (OSError, ValueError):
        value = _cursor_speed_default()
    return jsonify({'speed': value, 'min': _SPEED_MIN})


@app.route('/api/cursor_speed', methods=['POST'])
def set_cursor_speed():
    data = request.json or {}
    try:
        value = float(data.get('speed'))
    except (TypeError, ValueError):
        return jsonify({'error': 'speed must be a number'}), 400
    value = max(_SPEED_MIN, min(100.0, value))
    try:
        with open(_cursor_speed_path(), 'w', encoding='utf-8') as fh:
            fh.write(f'{value:.1f}')
    except OSError as exc:
        return jsonify({'error': f'could not persist cursor speed: {exc}'}), 500
    # Also seed the env, so a task spawned before the file is read still agrees.
    os.environ['MOUSE_GLIDE_SPEED_PCT'] = f'{value:.1f}'
    print(f"[API] Cursor speed set to {value:.0f}%")
    return jsonify({'status': 'ok', 'speed': value})


if __name__ == '__main__':
    # Startup is always a fresh conversation.
    _clear_conversation(delete_screenshot=True)
    print("\n" + "="*60)
    print("🤖 Computer Use Agent - Backend Started")
    print("="*60)
    print("\n✨ Open the web interface:")
    print("   👉 http://localhost:3000")
    print("\n" + "="*60 + "\n")
    app.run(debug=False, port=5000, threaded=True)
