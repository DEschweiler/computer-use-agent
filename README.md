# Computer Use Agent

AI-powered agent that controls the foreground window on your desktop using **Tesseract OCR**. A React + Flask web interface lets you submit tasks and watch the agent work in real time.

## How it works

The agent captures the active foreground window, runs multi-pass Tesseract OCR to detect text elements, and sends both the OCR output and a screenshot to a vision model. The model returns tool calls (click, type, key press, …) which are executed via PyAutoGUI. This loop repeats until the task is complete.

## Requirements

- Python 3.10+
- Node.js 18+ (for the web frontend)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and on `PATH` (or at the default `%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe` on Windows)
- A vision-capable LLM accessible via a chat completions API

## Setup

### 1. Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment variables

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

# Optional
OCR_LANG=eng    # Tesseract language code(s), e.g. "deu+eng"
```

Switching models is done by changing `ACTIVE_PROFILE` — no code changes required.

### 3. Tesseract OCR

| Platform | Install |
|----------|---------|
| Windows  | [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) |
| macOS    | `brew install tesseract` |
| Ubuntu   | `sudo apt-get install tesseract-ocr` |

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
```

Or start them manually:

```bash
# Terminal 1 — backend
cd backend
python app.py

# Terminal 2 — frontend
cd frontend
npm run dev
```

Then open **http://localhost:3000** in your browser.

You can also run the agent directly from the command line (no web UI):

```bash
python agent.py
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
┌─────────────────┐
│ Computer Agent  │
│ OCR + PyAutoGUI │
└─────────────────┘
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


