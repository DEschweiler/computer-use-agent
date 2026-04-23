# Computer Use Agent

AI-powered agent that controls the foreground window on your desktop. A React + Flask web interface lets you submit tasks and watch the agent work in real time.

## How it works

The agent captures the active foreground window and runs two perception signals in parallel:

- **RapidOCR** (PPOCRv5, Latin) — detects text elements with confidence ≥ 0.9. Shown as orange boxes in the annotated screenshot.
- **OmniParser YOLOv8** (via ONNX Runtime) — detects interactive regions without text (buttons, icons, empty fields). Shown as blue boxes. YOLO boxes that overlap an OCR box by ≥ 5 % of the OCR box's area are suppressed to avoid redundancy.

Both element lists are sent together with the annotated screenshot to a vision LLM. The model returns tool calls (click, type, key press, …) which are executed via PyAutoGUI. This loop repeats until the task is complete.

## Requirements

- Python 3.10+
- Node.js 18+ (for the web frontend)
- A vision-capable LLM accessible via a chat completions API

## Setup

### 1. Download models (one-time)

```bash
conda activate visualagent
python download_models.py
```

This downloads and verifies:
- **RapidOCR** models → `~/.rapidocr/models/`
- **OmniParser YOLO** model → `models/icon_detect.onnx` (~100 MB from HuggingFace)

### 2. Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment variables

Create a `.env` file in the project root. The agent uses a **profile system**: set `ACTIVE_PROFILE` to the name of the profile you want to use, then define the three variables for that profile using the pattern `<PROFILE>_ENDPOINT`, `<PROFILE>_API_KEY`, and `<PROFILE>_MODEL`. Multiple profiles can coexist in the same file.

```env
# Select the active profile (case-insensitive)
ACTIVE_PROFILE=myprofile

# --- Profile: myprofile ---
MYPROFILE_ENDPOINT=https://your-llm-endpoint/v1/chat/completions
MYPROFILE_API_KEY=your_api_key
MYPROFILE_MODEL=your-model-name

# --- Profile: another ---
ANOTHER_ENDPOINT=https://other-endpoint/v1/chat/completions
ANOTHER_API_KEY=other_key
ANOTHER_MODEL=other-model-name
```

Switching models is done by changing `ACTIVE_PROFILE` — no code changes required.

Optional tuning via environment variables:

| Variable | Default | Description |
|---|---|---|
| `YOLO_IMGSZ` | `1920` | Input resolution fed to YOLO (multiple of 32). Higher = better small-icon recall, slower. |
| `YOLO_CONF_THRESH` | `0.05` | YOLO confidence threshold. |
| `YOLO_IOU_THRESH` | `0.3` | IoU threshold for YOLO-internal NMS. |
| `YOLO_OCR_OVERLAP_THRESH` | `0.05` | Fraction of an OCR box that a YOLO box must cover to be suppressed. |

### 4. Frontend dependencies

```bash
cd frontend
npm install
```

## Running

The quickest way on Windows is the provided batch files (run each in its own terminal):

```
1_run_backend.bat   # starts Flask on http://localhost:5000
2_run_frontend.bat  # starts Vite dev server on http://localhost:3000
3_start_agent.bat   # opens the localhost on port 3000 in your browser
```

Or start them manually:

```bash
# Terminal 1 — backend
cd backend
python app.py

# Terminal 2 — frontend
cd frontend
npm run dev

# Then open **http://localhost:3000** in your browser.
```

## Web interface

- Type a task in the chat input and press **▶** to start.
- The agent's actions and log output stream in real time in the debug panel.
- Click **⏹ Stop** to immediately terminate the agent and re-enable the input.

## Architecture

```
┌─────────────────┐
│  React Frontend │  (Port 3000)
│   Dark Mode UI  │
└────────┬────────┘
         │ HTTP + SSE
         ▼
┌─────────────────┐
│  Flask Backend  │  (Port 5000)
│   Agent Runner  │
└────────┬────────┘
         │ subprocess
         ▼
┌──────────────────────────────┐
│        Computer Agent        │
│                              │
│  ┌───────────┐ ┌──────────┐  │
│  │ RapidOCR  │ │   YOLO   │  │  (parallel)
│  │ (text)    │ │ (icons)  │  │
│  └─────┬─────┘ └────┬─────┘  │
│        └──────┬──────┘        │
│           merge + dedup       │
│               │               │
│          PyAutoGUI            │
└──────────────────────────────┘
```

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/task` | Start a new agent task (`{"task": "..."}`) |
| `POST` | `/api/stop` | Terminate the running agent immediately |
| `GET`  | `/api/logs` | Stream agent logs via Server-Sent Events |
| `GET`  | `/api/screenshot` | Latest annotated debug screenshot (PNG) |
| `GET`  | `/api/health` | Health check |
| `GET`  | `/api/info` | Agent and model info |


