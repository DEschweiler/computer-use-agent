#!/usr/bin/env python3
"""
OCR-only computer-use agent (refactored).

Operates on foreground-window screenshots + RapidOCR, designed for
generalist navigation of native Windows apps and Citrix-hosted remote apps
where no structural UI tree is available.

Design notes (differences from the previous version):
  * Every action tool carries a 'thought' parameter — the model externalizes
    its reasoning ("what changed, what I'm doing, what I expect") before each
    action. This replaces machine-generated outcome strings.
  * The screenshot sent to the LLM is annotated: OCR boxes + stable IDs,
    plus a crosshair at the last click location. Same image is written to
    the debug path so the frontend sees what the model sees.
  * Composite click_and_type action for the common click-then-type pattern.
  * Progress trail built from the model's own thoughts, pinned at the top
    of the latest observation. Replaces the old action journal.
  * Tight context diet: 2 recent full observations; older tool_call/result
    pairs collapsed to one-line trail entries and removed from the stream.
  * Thrashing detector over the last 6 trail entries.
  * Wait-aware stuck counter (model-initiated waits don't tick it).
  * IoU-based OCR dedup (RapidOCR rarely overlaps, kept as safety net).
  * Verifier uses explicit task_complete / continue_working; no auto-execute.
  * DwmGetWindowAttribute for true window bounds (fixes edge-click misses).

Public surface preserved for backend/frontend compatibility:
  * ComputerAgent, AgentConfig, _load_config, interactive, main
  * agent.run(instruction) -> dict with 'started_at', 'duration_seconds',
    'actions', 'tokens'
  * actions_log entries retain 'iteration', 'action', 'args', 'result' keys;
    task_complete entries retain 'claim', 'verified', 'reason'
  * Debug screenshot written to $TEMP/agent_screenshot_debug.png each parse
  * logging.getLogger("agent") is the channel the backend hooks
  * Log markers [SCREENSHOT_READY], [TASK_RESULT] preserved
"""

from __future__ import annotations

import base64
import concurrent.futures
import datetime
import difflib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import urllib3
from collections import deque
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pyautogui
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

# --- Environment setup ----------------------------------------------------- #

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# RapidOCR engine — created on first use so import-time cost is zero.
_rapidocr_engine: Optional[RapidOCR] = None


def _get_rapidocr_engine() -> RapidOCR:
    global _rapidocr_engine
    if _rapidocr_engine is None:
        _rapidocr_engine = RapidOCR(params={
            "Rec.lang_type": LangRec.LATIN,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.MOBILE,
            "Global.log_level": "error",
        })
    return _rapidocr_engine


# YOLO icon-detector engine (OmniParser v2) — lazy-loaded on first use. -----
# First call: checks models/icon_detect.onnx (put there by download_models.py);
# if absent, falls back to downloading + exporting from HuggingFace Hub.

_YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1920"))
_YOLO_CONF_THRESH = float(os.getenv("YOLO_CONF_THRESH", "0.05"))
_YOLO_IOU_THRESH = float(os.getenv("YOLO_IOU_THRESH", "0.3"))
# Fraction of a YOLO box's area that must be covered by an OCR box to suppress it.
# Intentionally low (5 %) so even a small text label inside a large icon box
# causes the YOLO box to be dropped — avoiding redundant blue overlays.
_YOLO_OCR_OVERLAP_THRESH = float(os.getenv("YOLO_OCR_OVERLAP_THRESH", "0.05"))

# Local model directory — download_models.py writes here.
_MODELS_DIR = Path(__file__).parent / "models"
_LOCAL_ONNX = _MODELS_DIR / "icon_detect.onnx"

_yolo_session: Any = None       # onnxruntime.InferenceSession | None | _YOLO_FAILED
_yolo_input_name: str = "images"
_YOLO_FAILED = object()         # sentinel: init attempted and failed — do not retry

# Pillow 9.1+ uses Image.Resampling; Pillow <9.1 used Image.BILINEAR directly.
_BILINEAR = getattr(getattr(Image, "Resampling", None), "BILINEAR", None) or Image.BILINEAR


def _get_yolo_session():
    """Return a cached ORT InferenceSession for the YOLO icon detector.

    Resolution order:
    1. models/icon_detect.onnx (put there by download_models.py) — preferred.
    2. ONNX next to the HF-cached .pt (exported on first run if missing).
    Returns None if setup failed (will not retry).
    """
    global _yolo_session, _yolo_input_name
    if _yolo_session is _YOLO_FAILED:
        return None
    if _yolo_session is not None:
        return _yolo_session
    _log = logging.getLogger("agent")
    try:
        import onnxruntime as ort

        def _load_session(onnx_path: Path) -> "ort.InferenceSession":
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return ort.InferenceSession(
                str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
            )

        def _export_from_hub() -> Path:
            from huggingface_hub import hf_hub_download
            pt_path = hf_hub_download(
                repo_id="microsoft/OmniParser-v2.0",
                filename="icon_detect/model.pt",
            )
            onnx_path = Path(pt_path).with_suffix(".onnx")
            if not onnx_path.exists():
                _log.info("Exporting YOLO icon detector to ONNX (one-time) ...")
                from ultralytics import YOLO as _YOLO
                exported = _YOLO(pt_path).export(
                    format="onnx", imgsz=_YOLO_IMGSZ, simplify=True, dynamic=True
                )
                onnx_path = Path(exported)
                _log.info("ONNX export complete: %s", onnx_path)
            return onnx_path

        # 1. Prefer local models/ directory.
        if _LOCAL_ONNX.exists():
            onnx_path = _LOCAL_ONNX
            _log.info("Loading local ONNX: %s", onnx_path)
        else:
            _log.info(
                "models/icon_detect.onnx not found — falling back to HF Hub download."
                " Run download_models.py once to avoid this."
            )
            onnx_path = _export_from_hub()

        try:
            sess = _load_session(onnx_path)
        except Exception as load_err:
            # Possibly a corrupted ONNX — delete and re-export from Hub.
            _log.warning("ORT failed to load %s (%s); re-exporting ...", onnx_path.name, load_err)
            onnx_path.unlink(missing_ok=True)
            onnx_path = _export_from_hub()
            sess = _load_session(onnx_path)

        _yolo_input_name = sess.get_inputs()[0].name
        _yolo_session = sess
        _log.info(
            "YOLO icon detector ready (imgsz=%d, input=%r): %s",
            _YOLO_IMGSZ, _yolo_input_name, onnx_path.name,
        )
    except Exception as exc:
        _log.error(
            "YOLO icon detector init failed (%s: %s) — icon detection disabled.",
            type(exc).__name__, exc, exc_info=True,
        )
        _yolo_session = _YOLO_FAILED
    return _yolo_session if _yolo_session is not _YOLO_FAILED else None


def _letterbox(
    image: "Image.Image", size: int
) -> "Tuple[np.ndarray, float, int, int]":
    """Resize *image* to a (size x size) square with grey letterbox padding.

    Returns (inp, scale, pad_top, pad_left) where *inp* is a float32 ndarray
    of shape (1, 3, size, size) ready for ORT inference.
    """
    w, h = image.size
    scale = size / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = image.resize((nw, nh), _BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_left = (size - nw) // 2
    pad_top = (size - nh) // 2
    canvas.paste(resized, (pad_left, pad_top))
    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]   # (1, 3, H, W)
    return arr, scale, pad_top, pad_left


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """Greedy IoU-based NMS. Returns indices of boxes to keep (highest score first)."""
    order = scores.argsort()[::-1]
    kept: List[int] = []
    while order.size > 0:
        i = int(order[0])
        kept.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[order[1:], 0])
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[order[1:], 1])
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[order[1:], 2])
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[order[1:], 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes_xyxy[i, 2] - boxes_xyxy[i, 0]) * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1])
        area_j = ((boxes_xyxy[order[1:], 2] - boxes_xyxy[order[1:], 0])
                  * (boxes_xyxy[order[1:], 3] - boxes_xyxy[order[1:], 1]))
        iou = inter / (area_i + area_j - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return kept


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")

# --- Tool schema (OpenAI-compatible function calling) ---------------------- #
#
# Every tool takes a 'thought' parameter. It is REQUIRED except where noted —
# the schema marks it required so small models include it reliably, but the
# executor never rejects an action for missing thought (we just log "(no
# thought)"). This gives us structured reasoning without brittle enforcement.

_THOUGHT_PARAM = {
    "type": "string",
    "description": (
        "One short sentence: (a) what changed since your last action and whether "
        "it matches what you expected, (b) what you are doing now, (c) what you "
        "expect to happen. Keep it under 40 words."
    ),
}

COMPUTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": (
                "Click a UI element. Prefer element_id (from the element list) — it resolves "
                "to exact screen coordinates. Use x/y only when no suitable element_id exists "
                "(e.g. clicking an empty edit field whose only nearby element is a label)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "element_id": {"type": "string", "description": "Short ID like 'e47' from the current element list."},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"},
                    "thought": _THOUGHT_PARAM,
                },
                "required": ["thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "double_click",
            "description": (
                "Double-click an element or coordinate. In text fields this selects the "
                "word under the cursor — you can then type to replace it or press Delete "
                "to remove it. Repeat on remaining words to clear a field word-by-word "
                "when Ctrl+A does not work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "element_id": {"type": "string"},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "thought": _THOUGHT_PARAM,
                },
                "required": ["thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": (
                "Type text into the currently focused field. Click the field first, or use "
                "click_and_type to combine both steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "thought": _THOUGHT_PARAM,
                },
                "required": ["text", "thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_and_type",
            "description": (
                "Click a target and then type text. Use this for the common 'focus a field "
                "and fill it' pattern — it avoids ordering mistakes between click and type. "
                "Specify either element_id or x/y; 'text' is what to type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "element_id": {"type": "string"},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "text": {"type": "string"},
                    "thought": _THOUGHT_PARAM,
                },
                "required": ["text", "thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keypress",
            "description": "Press one or more keys simultaneously, e.g. ['ctrl','a'] or ['enter'].",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {"type": "array", "items": {"type": "string"}},
                    "thought": _THOUGHT_PARAM,
                },
                "required": ["keys", "thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the mouse wheel at a position. Use x/y to place the cursor before scrolling (defaults to screen center). scroll_clicks: positive = scroll UP (toward top of page), negative = scroll DOWN (toward bottom). Use 200–1000 for a normal scroll, 100–3000 to jump a large section.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "scroll_clicks": {"type": "integer", "description": "Wheel clicks to scroll. Positive = up, negative = down. Typical range: 200-1000; use 100-3000 for a large jump."},
                    "thought": _THOUGHT_PARAM,
                },
                "required": ["scroll_clicks", "thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": (
                "Wait a short period (e.g. for a dialog to appear, a slow save to complete). "
                "Default 1.5s. Use this when you expect the screen to change on its own — "
                "the stuck-screen detector knows not to count waits against you."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "number"},
                    "thought": _THOUGHT_PARAM,
                },
                "required": ["thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_task",
            "description": (
                "Declare the task finished. You MUST cite evidence: specific text "
                "currently visible on screen that proves the task succeeded. "
                "CRITICAL: do NOT cite field labels, form titles, menu items, or "
                "other UI chrome that was already there before you acted — those "
                "prove nothing. Cite the OUTCOME of your work: a value you typed "
                "that now shows in a field, a confirmation/success message, a row "
                "that now appears in a list, a status label that changed. If you "
                "cannot find any such outcome text on screen, the task is most "
                "likely NOT done — do not call finish_task yet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Short summary for the user of what was done."
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "2-5 strings, each being a SHORT, VERBATIM snippet copied "
                            "directly from the OCR text on screen — ideally just the "
                            "raw value itself (e.g. '444444', 'Gespeichert', "
                            "'Max Mustermann'). Do NOT add any surrounding words, "
                            "explanations, or context ('field shows', 'updated to', "
                            "etc.) — only the literal text as OCR reads it."
                        ),
                    },
                    "thought": _THOUGHT_PARAM,
                },
                "required": ["message", "evidence", "thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_windows",
            "description": (
                "List all visible windows with their titles. Use this to discover which "
                "applications are open before calling focus_window."
            ),
            "parameters": {
                "type": "object",
                "properties": {"thought": _THOUGHT_PARAM},
                "required": ["thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "focus_window",
            "description": (
                "Bring a window to the foreground by title (case-insensitive substring "
                "match). Call list_windows first to see available titles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Substring of the window title to match."},
                    "thought": _THOUGHT_PARAM,
                },
                "required": ["title", "thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": (
                "Launch an application by name via the Windows Start menu (opens Start, types "
                "the name, presses Enter), then brings it to the foreground. Use this when the "
                "application you need is NOT listed by list_windows. After launching, confirm "
                "the correct window is focused before acting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Application name to search for and launch, e.g. 'HospitalRun', 'Notepad'."},
                    "thought": _THOUGHT_PARAM,
                },
                "required": ["name", "thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "Ask the user a clarifying question and stop. Use ONLY when you genuinely cannot "
                "determine which application or target the request refers to and no reasonable "
                "default exists. Ends the current run; the user's reply continues this same "
                "conversation, so you will not lose progress or repeat completed work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "A specific, concrete question for the user."},
                    "thought": _THOUGHT_PARAM,
                },
                "required": ["question", "thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Integer addition/subtraction for deriving pixel coordinates (e.g. the "
                "midpoint between two OCR elements, or an offset from a known coordinate)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "op": {"type": "string", "enum": ["+", "-"]},
                    "b": {"type": "integer"},
                    "thought": _THOUGHT_PARAM,
                },
                "required": ["a", "op", "b", "thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_chars",
            "description": (
                "Press End (move cursor to end of field) then Backspace a specific number "
                "of times. Use as a last resort when both Ctrl+A and double_click failed. "
                "Count should match the number of characters currently in the field."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of Backspace keypresses to send after moving to End (1-500).",
                    },
                    "thought": _THOUGHT_PARAM,
                },
                "required": ["count", "thought"],
            },
        },
    },
]

# Verification-only tools: offered ONLY during completion verification.
_VERIFICATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "finish_task",
            "description": (
                "Confirm the task is truly done based on the current screen. "
                "Cite OUTCOME evidence only — text that shows the work succeeded. "
                "Do NOT cite field labels, form titles, or menu items that were "
                "already on screen before the actor acted; those prove nothing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "2-5 strings, each being a SHORT, VERBATIM snippet copied "
                            "directly from the OCR text on screen — ideally just the "
                            "raw value itself (e.g. '444444', 'Gespeichert'). "
                            "Do NOT add surrounding words or context."
                        ),
                    },
                    "thought": _THOUGHT_PARAM,
                },
                "required": ["message", "evidence", "thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "continue_working",
            "description": (
                "Reject the completion claim — the task is not yet done. Give a short reason. "
                "The main loop will resume with your next observation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "thought": _THOUGHT_PARAM,
                },
                "required": ["reason", "thought"],
            },
        },
    },
]


# --- Navigator (goal-holding critic) --------------------------------------- #
#
# The navigator is a SEPARATE, cheap, text-only reasoning role. It never touches
# the mouse or keyboard. Each step it looks at the ultimate goal, the change-set
# ("what the last action caused"), and the progress trail — from a neutral
# outside viewpoint — and answers: are we closer, further, or stuck, and what is
# the single next objective the actioner should pursue. On 'stuck' it may force a
# replan by issuing a different next_intent; corrective moves (dismiss a dialog,
# scroll back) are expressed as intents that the actioner executes through its
# own guarded tools — preserving a single action pathway.

NAVIGATOR_SYSTEM_PROMPT = """\
You are the NAVIGATOR for a computer-use agent operating a hospital information \
system (HIS). You do NOT control the mouse or keyboard — a separate ACTIONER \
does that. Your job is to keep the actioner oriented toward the ultimate goal.

You cannot assume a fixed click-path: the HIS layout is not known in advance and \
each action's consequences are only learned by observing the screen afterward. \
So you work in a closed loop: hold the ultimate goal, and after each action judge \
whether it moved the agent CLOSER to that goal, then set the next concrete \
objective.

Every step you receive:
  - ULTIMATE GOAL — the user's task; this never changes.
  - CURRENT OBJECTIVE — what the actioner was just trying to do.
  - SINCE THE LAST ACTION — a ground-truth diff of what changed on screen \
(elements that appeared, disappeared, moved, or whose text changed; and whether \
a scroll, popup, or full screen-replacement occurred). The environment is \
quiescent between actions, so these changes were caused by the last action.
  - PROGRESS SO FAR — the history of actions and their outcomes.
  - CURRENT SCREEN ELEMENTS — the text/elements currently visible.

Call `assess` exactly once with:
  - status: on_track | off_track | stuck | goal_reached
  - reasoning: ONE sentence — did the last action move closer to the goal?
  - next_intent: the SINGLE next objective for the actioner, phrased as a concrete \
instruction achievable in a few actions (e.g. "open the patient search and enter \
ID 444444", not "complete the task"). Keep the objective small and verifiable.
  - guidance (optional): a short tactical correction, especially when off_track or \
stuck — e.g. "a confirmation dialog is open; dismiss it before anything else", \
"the field scrolled out of view, scroll up ~200px to bring it back", or "this \
approach has not changed the screen twice; try keyboard Tab instead of clicking".

Rules:
  - Judge only against the ULTIMATE GOAL, not against whether the last action \
"worked" in isolation. An action can succeed yet move away from the goal.
  - If SINCE THE LAST ACTION shows a popup/dialog or a screen replacement, the \
next_intent must deal with that first.
  - Only report goal_reached when the diff/screen shows concrete OUTCOME evidence \
of success (a typed value now present, a success message, a new row) — not merely \
that the right form or menu is visible.
  - Be decisive and brief. You are the map, not the driver.
"""

NAVIGATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "assess",
            "description": (
                "Report where the task stands relative to the ULTIMATE GOAL and set "
                "the next concrete objective for the actioner."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["on_track", "off_track", "stuck", "goal_reached"],
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "One sentence: did the last action move closer to the ultimate goal?",
                    },
                    "next_intent": {
                        "type": "string",
                        "description": (
                            "The single next objective for the actioner, as a concrete "
                            "instruction achievable in a few actions."
                        ),
                    },
                    "guidance": {
                        "type": "string",
                        "description": "Optional short tactical correction or hint for the actioner.",
                    },
                },
                "required": ["status", "reasoning", "next_intent"],
            },
        },
    },
]


# --- Narrator (on-demand visual observer) ---------------------------------- #
#
# The deterministic change-set is blind to visual-only state (a button greying
# out, a spinner, a red validation border, a selected row) because those move no
# element and change no text. The narrator is a VLM call — EXPENSIVE, so invoked
# only when the cheap signals are insufficient: an ambiguous transition (popup /
# screen-replaced) or a stuck run. It is GROUNDED in the deterministic diff to
# curb hallucination, and it stays goal-agnostic (describe, don't advise) — the
# navigator does the goal reasoning.

NARRATOR_SYSTEM_PROMPT = """\
You are the OBSERVER for a computer-use agent. You are shown the current \
screenshot and a deterministic list of element changes since the last action \
(ground truth). Describe, in 1-3 concrete sentences, WHAT CHANGED on screen — \
paying special attention to VISUAL or STATE changes the element list cannot \
capture: a button becoming enabled/disabled or greyed out, a spinner or progress \
indicator, a row becoming selected/highlighted, a red/coloured validation border, \
a checkbox or toggle flipping, a colour change, a dialog overlaying the page.

Rules:
  - Describe only what is actually visible. If nothing beyond the listed element \
changes is apparent, say so briefly.
  - Do NOT give instructions or judge progress — only report observations.
  - Be specific about location ("the Save button, bottom-right") so the actioner \
can act on it.
"""

NARRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "describe",
            "description": "Report what visibly changed on screen since the last action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "1-3 sentences describing what changed on screen.",
                    },
                    "visual_state_changes": {
                        "type": "string",
                        "description": (
                            "Visual/state changes NOT captured by the element diff "
                            "(disabled/greyed button, spinner, highlight, validation "
                            "colour, etc.), or empty if none."
                        ),
                    },
                },
                "required": ["summary"],
            },
        },
    },
]


# --- DPI awareness --------------------------------------------------------- #

def _enable_dpi_awareness() -> None:
    """Make the process DPI-aware so coordinates are physical pixels."""
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor v2
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    except Exception as exc:
        log.debug("DPI awareness setup failed: %s", exc)


# --- Element model + registry --------------------------------------------- #

@dataclass
class Element:
    """A single UI element detected by OCR, with a stable ID across parses."""
    stable_id: str
    text: str
    control_type: str
    source: str
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    confidence: int = 0
    automation_id: str = ""
    parent_text: str = ""
    last_seen_frame: int = 0

    def as_prompt_line(self) -> str:
        if self.source == "yolo":
            return (
                f"[{self.stable_id}] <interactive> @({self.center_x},{self.center_y})"
                f" {self.width}x{self.height}px"
            )
        text = self.text.replace("\n", " ").strip()
        if len(text) > 60:
            text = text[:57] + "..."
        return f"[{self.stable_id}] '{text}' @({self.center_x},{self.center_y})"


@dataclass
class ElementChange:
    """A single per-element delta between two consecutive frames."""
    kind: str            # 'appeared' | 'disappeared' | 'moved' | 'text_changed'
    element: Element     # the CURRENT element (for disappeared, the last-seen one)
    old_text: str = ""   # text_changed only
    dx: int = 0          # moved only (new - old center)
    dy: int = 0          # moved only

    def as_line(self) -> str:
        el = self.element
        if el.source == "yolo":
            label = f"[{el.stable_id}] <interactive> @({el.center_x},{el.center_y})"
        else:
            t = (el.text or "").replace("\n", " ").strip()
            if len(t) > 40:
                t = t[:37] + "..."
            label = f"[{el.stable_id}] '{t}'"
        if self.kind == "text_changed":
            return f"{label}  ('{self.old_text}' → '{el.text}')"
        if self.kind == "moved":
            return f"{label}  (moved {self.dx:+d},{self.dy:+d})"
        return label


@dataclass
class ChangeSet:
    """
    Ground-truth summary of what changed between the previous frame and this one,
    derived from the ElementRegistry's stable-ID reconciliation. This is the
    observation the agents reason over — "what my last action caused" — rather
    than a bare snapshot of the current screen.

    `transition` names a whole-screen event when one is detected, so a scroll or
    a dialog does not flood the per-element diff:
      '' (local edits) | 'scroll' | 'popup' | 'replaced'
    """
    appeared: List[ElementChange] = field(default_factory=list)
    disappeared: List[ElementChange] = field(default_factory=list)
    moved: List[ElementChange] = field(default_factory=list)
    text_changed: List[ElementChange] = field(default_factory=list)
    transition: str = ""
    scroll_dy: int = 0        # scroll only: median vertical shift (+ = content moved down)

    @property
    def is_empty(self) -> bool:
        return not (self.appeared or self.disappeared or self.moved or self.text_changed)

    def render(self) -> str:
        """Human/LLM-readable 'SINCE YOUR LAST ACTION' block. Empty string if nothing changed."""
        if self.transition == "replaced":
            return (
                "SINCE YOUR LAST ACTION: the screen was REPLACED "
                f"(≈{len(self.disappeared)} elements gone, "
                f"{len(self.appeared)} new). You are likely on a different view/window."
            )
        if self.transition == "scroll":
            direction = "down" if self.scroll_dy < 0 else "up"
            gone = ", ".join(c.as_line() for c in self.disappeared[:8]) or "(none)"
            new = ", ".join(c.as_line() for c in self.appeared[:8]) or "(none)"
            return (
                f"SINCE YOUR LAST ACTION: SCROLLED {direction} ≈{abs(self.scroll_dy)}px. "
                f"Left the viewport: {gone}. Newly visible: {new}. "
                "To return to what you had before, scroll the opposite direction."
            )

        lines: List[str] = []
        if self.transition == "popup":
            lines.append(
                "A DIALOG/POPUP appeared on top of the previous screen "
                "(existing elements stayed). Deal with the dialog before anything else."
            )
        if self.text_changed:
            lines.append("changed: " + "; ".join(c.as_line() for c in self.text_changed[:12]))
        if self.appeared:
            lines.append("appeared: " + ", ".join(c.as_line() for c in self.appeared[:12]))
        if self.disappeared:
            lines.append("disappeared: " + ", ".join(c.as_line() for c in self.disappeared[:12]))
        if self.moved:
            lines.append("moved: " + ", ".join(c.as_line() for c in self.moved[:8]))
        if not lines:
            return "SINCE YOUR LAST ACTION: no visible change on screen."
        return "SINCE YOUR LAST ACTION:\n  " + "\n  ".join(lines)

    def summary(self) -> str:
        """Compact single-line summary — safe for the (newline-delimited) log/SSE stream."""
        if self.transition == "replaced":
            return f"screen replaced ({len(self.disappeared)} gone, {len(self.appeared)} new)"
        if self.transition == "scroll":
            d = "down" if self.scroll_dy < 0 else "up"
            return f"scrolled {d} ~{abs(self.scroll_dy)}px ({len(self.appeared)} newly visible)"
        if self.is_empty:
            return "no visible change"
        parts = []
        if self.text_changed:
            parts.append(f"{len(self.text_changed)} changed")
        if self.appeared:
            parts.append(f"{len(self.appeared)} appeared")
        if self.disappeared:
            parts.append(f"{len(self.disappeared)} disappeared")
        if self.moved:
            parts.append(f"{len(self.moved)} moved")
        prefix = "popup + " if self.transition == "popup" else ""
        return prefix + ", ".join(parts)


class ElementRegistry:
    """Assigns short stable IDs to OCR elements across parses."""

    # A matched element must shift at least this many px (either axis) to count
    # as "moved" — filters OCR jitter and sub-pixel box wobble.
    _MOVE_EPS = 8

    def __init__(self, match_threshold: float = 0.62, evict_after_frames: int = 20):
        self._by_id: Dict[str, Element] = {}
        self._next_id = 1
        self._frame = 0
        self._match_threshold = match_threshold
        self._evict_after = evict_after_frames

    def _mint_id(self) -> str:
        sid = f"e{self._next_id}"
        self._next_id += 1
        return sid

    @staticmethod
    def _similarity(new: Element, old: Element, screen_w: int, screen_h: int) -> float:
        if new.text and old.text:
            text_sim = difflib.SequenceMatcher(None, new.text.lower(), old.text.lower()).ratio()
        else:
            text_sim = 0.4 if (not new.text and not old.text) else 0.0
        type_sim = 1.0 if new.control_type == old.control_type else 0.3
        source_sim = 1.0 if new.source == old.source else 0.5
        dx = abs(new.center_x - old.center_x) / max(1, screen_w)
        dy = abs(new.center_y - old.center_y) / max(1, screen_h)
        pos_sim = max(0.0, 1.0 - 4.0 * (dx + dy))
        dw = abs(new.width - old.width) / max(1, screen_w)
        dh = abs(new.height - old.height) / max(1, screen_h)
        size_sim = max(0.0, 1.0 - 4.0 * (dw + dh))
        return (
            0.45 * text_sim + 0.15 * type_sim + 0.10 * source_sim
            + 0.20 * pos_sim + 0.10 * size_sim
        )

    def reconcile(
        self, new_elements: List[Element], screen_w: int, screen_h: int
    ) -> Tuple[List[Element], ChangeSet]:
        """Assign stable IDs and, as a side product, compute the frame-to-frame
        ChangeSet (appeared / disappeared / moved / text-changed + transition)."""
        prev_frame = self._frame
        self._frame += 1
        # Shallow copy so old Element objects survive the in-place id reassignments
        # below (we replace dict values, we don't mutate the old objects).
        prev_snapshot = dict(self._by_id)
        prev_visible_ids = {
            rid for rid, el in prev_snapshot.items() if el.last_seen_frame == prev_frame
        }

        # Position gate: a real element never teleports horizontally, and moves
        # vertically only within roughly a screen (scroll). Reject candidate
        # matches beyond these bounds even when text is similar — this stops a
        # button ('Speichern') from being relabeled as a far-away status
        # ('Gespeichert'), which both pollutes the change-set and corrupts the
        # stable IDs the executor clicks by.
        max_dx = 0.30 * screen_w
        max_dy = 0.85 * screen_h

        unmatched_ids = set(self._by_id.keys())
        result: List[Element] = []
        change = ChangeSet()
        matched_ids: set = set()
        for elem in new_elements:
            best_id, best_score = None, self._match_threshold
            for rid in unmatched_ids:
                old = self._by_id[rid]
                if abs(elem.center_x - old.center_x) > max_dx:
                    continue
                if abs(elem.center_y - old.center_y) > max_dy:
                    continue
                score = self._similarity(elem, old, screen_w, screen_h)
                if score > best_score:
                    best_score = score
                    best_id = rid
            if best_id is not None:
                old = prev_snapshot[best_id]
                elem.stable_id = best_id
                elem.last_seen_frame = self._frame
                self._by_id[best_id] = elem
                unmatched_ids.remove(best_id)
                matched_ids.add(best_id)
                # Only diff against elements that were actually visible last frame.
                if best_id in prev_visible_ids:
                    if elem.text.strip() != old.text.strip():
                        change.text_changed.append(
                            ElementChange("text_changed", elem, old_text=old.text.strip())
                        )
                    dx = elem.center_x - old.center_x
                    dy = elem.center_y - old.center_y
                    if abs(dx) >= self._MOVE_EPS or abs(dy) >= self._MOVE_EPS:
                        change.moved.append(ElementChange("moved", elem, dx=dx, dy=dy))
            else:
                elem.stable_id = self._mint_id()
                elem.last_seen_frame = self._frame
                self._by_id[elem.stable_id] = elem
                # Genuinely new only if there was a prior frame to be new against.
                if prev_frame > 0:
                    change.appeared.append(ElementChange("appeared", elem))
            result.append(elem)

        # Disappeared = visible last frame, not matched this frame.
        for rid in prev_visible_ids - matched_ids:
            change.disappeared.append(ElementChange("disappeared", prev_snapshot[rid]))

        # The similarity matcher is text-weighted, so an in-place value replacement
        # (a field going 'Speichern' → '444444', or an empty YOLO field → typed text)
        # falls below threshold and lands as a disappeared+appeared pair at the same
        # box. Re-pair those into text_changed — it's the #1 form-filling signal.
        self._pair_inplace_text_changes(change)

        self._classify_transition(change, len(prev_visible_ids))

        stale = [rid for rid, el in self._by_id.items()
                 if self._frame - el.last_seen_frame > self._evict_after]
        for rid in stale:
            del self._by_id[rid]
        return result, change

    @staticmethod
    def _box_iou(a: Element, b: Element) -> float:
        ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
        ix2 = min(a.x + a.width, b.x + b.width)
        iy2 = min(a.y + a.height, b.y + b.height)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = a.width * a.height + b.width * b.height - inter
        return inter / union if union > 0 else 0.0

    def _pair_inplace_text_changes(self, change: ChangeSet, iou_thresh: float = 0.3) -> None:
        """Collapse co-located disappeared+appeared pairs into text_changed."""
        if not change.appeared or not change.disappeared:
            return
        used: set = set()
        kept_appeared: List[ElementChange] = []
        for app in change.appeared:
            best_i, best_iou = None, iou_thresh
            for i, dis in enumerate(change.disappeared):
                if i in used:
                    continue
                iou = self._box_iou(app.element, dis.element)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i is None:
                kept_appeared.append(app)
            else:
                used.add(best_i)
                old_text = change.disappeared[best_i].element.text.strip()
                change.text_changed.append(
                    ElementChange("text_changed", app.element, old_text=old_text)
                )
        change.appeared = kept_appeared
        change.disappeared = [d for i, d in enumerate(change.disappeared) if i not in used]

    @staticmethod
    def _classify_transition(change: ChangeSet, prev_n: int) -> None:
        """Detect a whole-screen event (scroll / popup / replaced) from the raw
        deltas, so downstream renderers can collapse the noise into one fact."""
        if prev_n == 0:
            return  # first frame — everything is 'appeared', not a transition
        gone, new = len(change.disappeared), len(change.appeared)

        # Screen replaced: most of the previous screen gone, lots of new content.
        if prev_n >= 5 and gone >= 0.7 * prev_n and new >= max(3, 0.4 * prev_n):
            change.transition = "replaced"
            return

        # Scroll: a coherent block of elements shifted by a common vertical delta.
        if len(change.moved) >= 3:
            dys = sorted(c.dy for c in change.moved)
            median = dys[len(dys) // 2]
            if abs(median) >= 20:
                coherent = sum(1 for c in change.moved if abs(c.dy - median) <= 15)
                if coherent >= 3 and coherent >= 0.6 * len(change.moved):
                    change.transition = "scroll"
                    change.scroll_dy = median
                    return

        # Popup/dialog: prior screen mostly intact, a cluster of new elements arrived.
        if new >= 3 and gone <= 2 and prev_n >= 3 and (prev_n - gone) >= 0.6 * prev_n:
            change.transition = "popup"

    def get(self, stable_id: str) -> Optional[Element]:
        return self._by_id.get(stable_id)


# --- OCR ------------------------------------------------------------------- #

def _rapidocr_to_boxes(result, min_score: float = 0.8) -> List[Dict[str, Any]]:
    """Convert a RapidOCR result (polygon-based) to axis-aligned box dicts.

    Only detections with a recognition confidence *score* > *min_score* are
    returned.
    """
    if result.boxes is None:
        return []
    boxes: List[Dict[str, Any]] = []
    for poly, text, score in zip(result.boxes, result.txts, result.scores):
        if score <= min_score:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x, y = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        boxes.append({
            "text": text,
            "conf": int(score * 100),
            "x": x, "y": y,
            "w": x2 - x, "h": y2 - y,
            "cx": (x + x2) // 2,
            "cy": (y + y2) // 2,
        })
    return boxes


def _iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Intersection-over-union for two axis-aligned boxes dict('x','y','w','h')."""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _coverage(yolo: Dict[str, Any], ocr: Dict[str, Any]) -> float:
    """Fraction of the *ocr* box's area that is covered by the *yolo* box.

    Using the OCR box as reference catches large YOLO boxes (e.g. an edit
    field) that contain a small text label inside them: the OCR box is almost
    entirely overlapped even though the YOLO box itself is much bigger.
    """
    ix1 = max(yolo["x"], ocr["x"])
    iy1 = max(yolo["y"], ocr["y"])
    ix2 = min(yolo["x"] + yolo["w"], ocr["x"] + ocr["w"])
    iy2 = min(yolo["y"] + yolo["h"], ocr["y"] + ocr["h"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ocr_area = ocr["w"] * ocr["h"]
    return inter / ocr_area if ocr_area > 0 else 0.0


def extract_ocr_elements(image: Image.Image) -> List[Element]:
    """Run RapidOCR on *image* and return a de-duplicated Element list."""
    engine = _get_rapidocr_engine()
    try:
        result = engine(np.array(image.convert("RGB")))
    except Exception:
        log.exception("RapidOCR failed")
        return []

    all_boxes = _rapidocr_to_boxes(result)

    # IoU-based dedup (safety net — RapidOCR rarely produces overlapping boxes).
    all_boxes.sort(key=lambda b: -b["conf"])
    kept: List[Dict[str, Any]] = []
    for box in all_boxes:
        is_dup = False
        for k in kept:
            if _iou(box, k) < 0.5:
                continue
            if difflib.SequenceMatcher(None, box["text"].lower(), k["text"].lower()).ratio() >= 0.6:
                is_dup = True
                break
        if not is_dup:
            kept.append(box)

    return [
        Element(
            stable_id="",
            text=b["text"], control_type="Text", source="ocr",
            x=b["x"], y=b["y"], width=b["w"], height=b["h"],
            center_x=b["cx"], center_y=b["cy"],
            confidence=b["conf"],
        )
        for b in kept
    ]


def ocr_signature(elements: List[Element]) -> frozenset:
    return frozenset(
        el.text.strip().lower()
        for el in elements
        if len(el.text.strip()) >= 2
    )


def extract_icon_elements(image: "Image.Image") -> List[Element]:
    """Run the OmniParser YOLO icon detector on *image*.

    Returns a list of Elements with source="yolo" and empty text, representing
    visually detected interactive regions (buttons, icons, empty fields).
    Only boxes that survive NMS and meet the confidence threshold are returned;
    overlapping with OCR boxes is handled by the caller.
    """
    session = _get_yolo_session()
    if session is None:
        return []
    try:
        inp, scale, pad_top, pad_left = _letterbox(image.convert("RGB"), _YOLO_IMGSZ)
        raw = session.run(None, {_yolo_input_name: inp})[0]  # (1, 5, N) or (1, N, 5)
        preds = raw[0]  # (5, N) or (N, 5)
        if preds.shape[0] < preds.shape[1]:  # (5, N) → transpose to (N, 5)
            preds = preds.T
        # columns: cx, cy, w, h, conf  (single-class model)
        scores = preds[:, 4]
        mask = scores > _YOLO_CONF_THRESH
        preds, scores = preds[mask], scores[mask]
        if len(preds) == 0:
            return []
        # Convert from IMGSZ space to original image pixel coords
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = (cx - bw / 2 - pad_left) / scale
        y1 = (cy - bh / 2 - pad_top) / scale
        x2 = (cx + bw / 2 - pad_left) / scale
        y2 = (cy + bh / 2 - pad_top) / scale
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        keep = _nms(boxes_xyxy, scores, _YOLO_IOU_THRESH)
        W, H = image.size
        elements: List[Element] = []
        for idx in keep:
            xi1 = max(0, int(boxes_xyxy[idx, 0]))
            yi1 = max(0, int(boxes_xyxy[idx, 1]))
            xi2 = min(W, int(boxes_xyxy[idx, 2]))
            yi2 = min(H, int(boxes_xyxy[idx, 3]))
            if xi2 <= xi1 or yi2 <= yi1:
                continue
            elements.append(Element(
                stable_id="",
                text="", control_type="Icon", source="yolo",
                x=xi1, y=yi1, width=xi2 - xi1, height=yi2 - yi1,
                center_x=(xi1 + xi2) // 2, center_y=(yi1 + yi2) // 2,
                confidence=int(scores[idx] * 100),
            ))
        return elements
    except Exception as exc:
        log.error("YOLO icon detection failed: %s: %s", type(exc).__name__, exc)
        return []


# --- Screen capture (foreground window only) ------------------------------ #

def _true_window_rect_windows(hwnd) -> Optional[Tuple[int, int, int, int]]:
    """Return (x, y, w, h) using DWM extended frame bounds (excludes drop shadow)."""
    try:
        import ctypes
        import ctypes.wintypes
        dwmapi = ctypes.windll.dwmapi
        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        rect = ctypes.wintypes.RECT()
        hr = dwmapi.DwmGetWindowAttribute(
            ctypes.wintypes.HWND(hwnd),
            ctypes.wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )
        if hr != 0:
            return None
        x, y = rect.left, rect.top
        w, h = rect.right - rect.left, rect.bottom - rect.top
        if w > 0 and h > 0:
            return (x, y, w, h)
    except Exception as exc:
        log.debug("DwmGetWindowAttribute failed: %s", exc)
    return None


def capture_foreground() -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Screenshot only the foreground window. Returns (image, (x, y, w, h)) where
    x, y are screen-absolute. OCR coords from the image must be offset by
    (x, y) to become screen-absolute before clicking.
    """
    if platform.system() == "Windows":
        try:
            import ctypes
            import ctypes.wintypes
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()

            # Prefer DWM extended frame bounds (excludes drop shadow).
            rect = _true_window_rect_windows(hwnd)
            if rect is None:
                r = ctypes.wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(r))
                rect = (r.left, r.top, r.right - r.left, r.bottom - r.top)

            x, y, w, h = rect
            if w > 0 and h > 0:
                img = pyautogui.screenshot(region=(x, y, w, h))
                return img, (x, y, w, h)
        except Exception as exc:
            log.warning("Foreground window capture failed (%s); falling back to full screen.", exc)
        import ctypes
        user32 = ctypes.windll.user32
        sw, sh = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        return pyautogui.screenshot(region=(0, 0, sw, sh)), (0, 0, sw, sh)

    if platform.system() == "Darwin":
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            path = tmp.name
        try:
            subprocess.run(["screencapture", "-x", path], check=True)
            img = Image.open(path).copy()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        w, h = img.size
        return img, (0, 0, w, h)

    img = pyautogui.screenshot()
    w, h = img.size
    return img, (0, 0, w, h)


# --- Image annotation ------------------------------------------------------ #

def _get_font(size: int = 11) -> Optional[ImageFont.ImageFont]:
    """Small truetype font if available, else default bitmap font."""
    candidates = [
        "arial.ttf", "Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _diff_regions(
    prev: "Image.Image",
    cur: "Image.Image",
    cell: int = 24,
    pixel_thresh: int = 28,
    min_cell_frac: float = 0.06,
    max_regions: int = 12,
) -> List[Tuple[int, int, int, int]]:
    """Coarse pixel-diff between two same-size frames → changed-region rectangles.

    Grayscale abs-diff is thresholded, pooled into a cell grid, and adjacent
    changed cells are merged (4-neighbour connected components). Returns up to
    `max_regions` (x, y, w, h) boxes in image coords, largest first. Full-frame
    regions (>60% area — i.e. the whole screen changed) are dropped; the caller
    already skips this entirely on scroll/replaced transitions.
    """
    if prev.size != cur.size:
        return []
    a = np.asarray(prev.convert("L"), dtype=np.int16)
    b = np.asarray(cur.convert("L"), dtype=np.int16)
    diff = np.abs(a - b) > pixel_thresh          # bool HxW
    if not diff.any():
        return []
    H, W = diff.shape
    gh, gw = (H + cell - 1) // cell, (W + cell - 1) // cell
    padded = np.zeros((gh * cell, gw * cell), dtype=np.int32)
    padded[:H, :W] = diff
    counts = padded.reshape(gh, cell, gw, cell).sum(axis=(1, 3))
    grid = counts >= int(cell * cell * min_cell_frac)

    visited = np.zeros_like(grid)
    regions: List[Tuple[int, int, int, int, int]] = []
    for cy in range(gh):
        for cx in range(gw):
            if not grid[cy, cx] or visited[cy, cx]:
                continue
            stack = [(cy, cx)]
            visited[cy, cx] = True
            minx = maxx = cx
            miny = maxy = cy
            while stack:
                yy, xx = stack.pop()
                minx, maxx = min(minx, xx), max(maxx, xx)
                miny, maxy = min(miny, yy), max(maxy, yy)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = yy + dy, xx + dx
                    if 0 <= ny < gh and 0 <= nx < gw and grid[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            rx, ry = minx * cell, miny * cell
            rw = min((maxx + 1) * cell, W) - rx
            rh = min((maxy + 1) * cell, H) - ry
            area = rw * rh
            if area > 0.6 * W * H:      # whole-screen change — not a useful highlight
                continue
            regions.append((rx, ry, rw, rh, area))
    regions.sort(key=lambda r: -r[4])
    return [(x, y, w, h) for (x, y, w, h, _) in regions[:max_regions]]


def annotate_screenshot(
    screenshot: Image.Image,
    elements: List[Element],
    win_x: int,
    win_y: int,
    click_marker: Optional[Tuple[int, int]] = None,
    max_label_elements: int = 120,
    highlight_regions: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Image.Image:
    """
    Draw OCR bounding boxes + stable IDs onto the screenshot, and optionally
    a crosshair at click_marker (screen-absolute coords).

    highlight_regions: image-relative (x, y, w, h) rectangles marking pixels that
    changed since the previous frame (from the pixel-diff). Drawn UNDER the
    element boxes as a translucent magenta wash so the model's eye is pulled to
    "what just changed" — catching visual-state changes OCR/YOLO cannot see
    (a button greying out, a spinner, a red validation border).

    Returns a new RGB image. Input is not modified.
    """
    out = screenshot.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    font = _get_font(11)

    # Change-highlight wash first, so element boxes/labels stay legible on top.
    for (rx, ry, rw, rh) in (highlight_regions or []):
        draw.rectangle([rx, ry, rx + rw, ry + rh], fill=(255, 0, 200, 40),
                       outline=(255, 0, 200, 230), width=2)
        if font is not None:
            draw.text((rx + 2, max(0, ry - 12)), "Δ changed", fill=(255, 120, 220), font=font)

    # Sort elements by confidence; label only the top N to avoid visual clutter
    # on very dense screens. Low-confidence elements still get a faint box.
    elems_sorted = sorted(elements, key=lambda el: -el.confidence)

    for i, el in enumerate(elems_sorted):
        ix = el.x - win_x
        iy = el.y - win_y
        iw = el.width
        ih = el.height

        # Faint box for every element; brighter for high-confidence / labeled ones.
        # OCR elements: orange. YOLO icon elements: blue.
        is_yolo = el.source == "yolo"
        color_full = (50, 150, 255, 220) if is_yolo else (255, 140, 0, 220)
        color_faint = (50, 150, 255, 90) if is_yolo else (255, 140, 0, 90)
        label_color = (130, 210, 255) if is_yolo else (255, 200, 80)
        # Always label YOLO elements (they're few and high-value); OCR elements
        # need confidence >= 40 to earn a label (avoids cluttering low-conf noise).
        show_label = i < max_label_elements and (is_yolo or el.confidence >= 40)
        if show_label:
            draw.rectangle([ix, iy, ix + iw, iy + ih], outline=color_full, width=1)
            # ID label above the box (or below if near top edge).
            label = el.stable_id
            label_y = iy - 12 if iy >= 14 else iy + ih + 1
            # Text with a thin dark backdrop for legibility on any background.
            if font is not None:
                try:
                    tw = draw.textlength(label, font=font)
                except Exception:
                    tw = 8 * len(label)
                th = 11
                draw.rectangle(
                    [ix, label_y, ix + tw + 3, label_y + th + 2],
                    fill=(0, 0, 0, 180),
                )
                draw.text((ix + 2, label_y), label, fill=label_color, font=font)
            else:
                draw.text((ix + 2, max(0, label_y)), label, fill=label_color)
        else:
            draw.rectangle([ix, iy, ix + iw, iy + ih], outline=color_faint, width=1)

    # Crosshair at last-click location.
    if click_marker is not None:
        cx, cy = click_marker[0] - win_x, click_marker[1] - win_y
        r = 10
        # Cyan circle + crosshair, with dark halo for contrast.
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 0, 0, 220), width=3)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 255, 255, 255), width=2)
        draw.line([cx - r - 4, cy, cx + r + 4, cy], fill=(0, 255, 255, 255), width=2)
        draw.line([cx, cy - r - 4, cx, cy + r + 4], fill=(0, 255, 255, 255), width=2)
        if font is not None:
            draw.text((cx + r + 6, cy - 6), "last click", fill=(0, 255, 255), font=font)

    return out


# --- Action executor ------------------------------------------------------- #

_KEY_MAP = {
    "strg": "ctrl", "control": "ctrl",
    "meta": "win", "super": "win", "command": "win", "cmd": "win",
    "option": "alt",
    "return": "enter",
    "esc": "esc", "escape": "esc",
    "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right",
    "delete": "del",
}

# Actions that manipulate the focused window and must NOT run until an
# application has been deliberately brought to the foreground this run.
# focus_window / open_application / list_windows / calculate / wait are exempt
# (they are how you *reach* a focused state, or are side-effect free).
_FOCUS_REQUIRED_ACTIONS = frozenset({
    "click", "double_click", "click_and_type", "type", "type_text",
    "keypress", "scroll", "delete_chars",
})


class ActionExecutor:
    """Pure input dispatcher. No knowledge of screens or LLMs."""

    def __init__(self, width: int, height: int, registry: ElementRegistry):
        self.width = width
        self.height = height
        self.registry = registry
        self.last_click_point: Optional[Tuple[int, int]] = None
        self.last_click_age: int = 0  # iterations since last click, for marker fade
        # Window-focus gating + persistence. `focused_once` guards input actions
        # until a focus has succeeded this run; `target_window` is the substring
        # of the last successfully-focused window, persisted across follow-up
        # requests in the same conversation (see save_session/load_session).
        self.focused_once: bool = False
        self.target_window: Optional[str] = None
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0

    def _resolve_point(self, args: Dict[str, Any]) -> Tuple[Optional[Tuple[int, int]], str]:
        eid = args.get("element_id")
        if eid:
            el = self.registry.get(eid)
            if el is None:
                return None, (
                    f"error: element_id '{eid}' not found (may have been evicted). "
                    "Use an id from the LATEST element list or fall back to x/y."
                )
            return (el.center_x, el.center_y), ""
        x, y = args.get("x"), args.get("y")
        if x is None or y is None:
            return None, "error: no element_id or coordinates provided"
        return (int(x), int(y)), ""

    def _clamp(self, x: int, y: int) -> Tuple[int, int]:
        x = max(2, min(self.width - 2, int(x)))
        y = max(2, min(self.height - 2, int(y)))
        return x, y

    def execute(self, action_type: str, args: Dict[str, Any]) -> str:
        try:
            # Gate: never touch the screen until a target application has been
            # deliberately focused this run. Prevents acting on whatever window
            # happened to be frontmost (e.g. the agent's own UI).
            if action_type in _FOCUS_REQUIRED_ACTIONS and not self.focused_once:
                return (
                    "error: no application focused yet. Before any click/type/scroll you MUST "
                    "bring the target application to the foreground — call focus_window (use "
                    "list_windows first to see exact titles), or open_application if it is not "
                    "running. Derive which application from the user's request."
                )

            if action_type == "open_application":
                name = args.get("name", "")
                if not name:
                    return "error: no application name provided"
                return self._open_application(name)

            if action_type in ("click", "double_click"):
                point, err = self._resolve_point(args)
                if point is None:
                    return err
                x, y = self._clamp(*point)
                button = args.get("button", "left")
                if action_type == "double_click":
                    pyautogui.doubleClick(x, y)
                else:
                    pyautogui.click(x, y, button=button)
                self.last_click_point = (x, y)
                self.last_click_age = 0
                time.sleep(0.4)
                return f"clicked ({x},{y})"

            if action_type == "click_and_type":
                point, err = self._resolve_point(args)
                if point is None:
                    return err
                x, y = self._clamp(*point)
                text = args.get("text", "")
                pyautogui.click(x, y)
                self.last_click_point = (x, y)
                self.last_click_age = 0
                time.sleep(0.3)
                pyautogui.write(text, interval=0.03)
                time.sleep(0.2)
                return f"clicked ({x},{y}) and typed {len(text)} chars"

            if action_type in ("type", "type_text"):
                text = args.get("text", "")
                pyautogui.write(text, interval=0.03)
                time.sleep(0.2)
                return f"typed {len(text)} chars"

            if action_type == "keypress":
                keys = [_KEY_MAP.get(k.lower(), k.lower()) for k in args.get("keys", [])]
                if not keys:
                    return "error: no keys"
                if len(keys) == 1:
                    time.sleep(0.1)
                    pyautogui.press(keys[0])
                    time.sleep(0.1)
                else:
                    time.sleep(0.2)
                    pyautogui.hotkey(*keys)
                    time.sleep(0.2)
                return f"pressed {'+'.join(keys)}"

            if action_type == "scroll":
                x = args.get("x", self.width // 2)
                y = args.get("y", self.height // 2)
                scroll_clicks = int(args.get("scroll_clicks", args.get("scroll_y", 0)))
                pyautogui.moveTo(*self._clamp(x, y))
                pyautogui.scroll(scroll_clicks)  # pyautogui: positive=up, negative=down
                return f"scrolled {scroll_clicks} clicks ({'up' if scroll_clicks > 0 else 'down'})"

            if action_type == "wait":
                secs = float(args.get("seconds", 1.5))
                time.sleep(min(secs, 10.0))
                return f"waited {secs}s"

            if action_type == "list_windows":
                return self._list_windows()

            if action_type == "focus_window":
                title = args.get("title", "")
                if not title:
                    return "error: no title provided"
                return self._focus_window(title)

            if action_type == "calculate":
                a = args.get("a")
                op = args.get("op")
                b = args.get("b")
                if a is None or b is None or op not in ("+", "-"):
                    return "error: calculate requires a, b, op in ('+', '-')"
                result = int(a) + int(b) if op == "+" else int(a) - int(b)
                return f"result: {result}"

            if action_type == "delete_chars":
                count = max(1, min(int(args.get("count", 1)), 500))
                pyautogui.press("end")
                time.sleep(0.1)
                for _ in range(count):
                    pyautogui.press("backspace")
                    time.sleep(0.02)
                return f"pressed End then backspace {count} times"

            return f"error: unknown action {action_type}"

        except pyautogui.FailSafeException:
            return "error: failsafe triggered"
        except Exception as exc:
            log.exception("Action error")
            return f"error: {exc}"

    def tick_click_age(self) -> None:
        """Called once per iteration. After 2 iterations, forget the click marker."""
        self.last_click_age += 1
        if self.last_click_age >= 2:
            self.last_click_point = None

    # --- Window management helpers (Windows) ------------------------------ #

    @staticmethod
    def _list_windows() -> str:
        if platform.system() != "Windows":
            return "error: list_windows is only supported on Windows"
        import ctypes
        import ctypes.wintypes
        user32 = ctypes.windll.user32
        titles: List[str] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        def enum_cb(hwnd, _lparam):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    titles.append(buf.value)
            return True

        user32.EnumWindows(enum_cb, 0)
        if not titles:
            return "(no visible windows found)"
        return "Visible windows:\n" + "\n".join(f"  - {t}" for t in titles)

    def _focus_window(self, title_substr: str) -> str:
        if platform.system() != "Windows":
            return "error: focus_window is only supported on Windows"
        import ctypes
        import ctypes.wintypes
        user32 = ctypes.windll.user32
        title_lower = title_substr.lower()
        target_hwnd = None

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        def enum_cb(hwnd, _lparam):
            nonlocal target_hwnd
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    if title_lower in buf.value.lower():
                        target_hwnd = hwnd
                        return False
            return True

        user32.EnumWindows(enum_cb, 0)
        if target_hwnd is None:
            return f"error: no window matching '{title_substr}' found"
        SW_RESTORE = 9
        if user32.IsIconic(target_hwnd):
            user32.ShowWindow(target_hwnd, SW_RESTORE)
        user32.SetForegroundWindow(target_hwnd)
        time.sleep(0.5)
        length = user32.GetWindowTextLengthW(target_hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(target_hwnd, buf, length + 1)
        # Record the focus so input actions are unblocked and the target
        # persists across follow-up requests in this conversation.
        self.focused_once = True
        self.target_window = title_substr
        return f"focused '{buf.value}'"

    def _open_application(self, name: str) -> str:
        """Launch an app by name via the Start menu (Win → type → Enter), then
        bring its window to the foreground. Returns a human-readable result."""
        if platform.system() != "Windows":
            return "error: open_application is only supported on Windows"
        pyautogui.press("win")
        time.sleep(0.6)
        pyautogui.write(name, interval=0.03)
        time.sleep(0.8)
        pyautogui.press("enter")
        # The app can take a few seconds to spawn its window; poll and focus.
        deadline = time.time() + 8.0
        while time.time() < deadline:
            time.sleep(1.0)
            res = self._focus_window(name)  # sets focused_once/target_window on success
            if not res.startswith("error"):
                return f"launched '{name}' and {res}"
        return (
            f"launched '{name}' via the Start menu, but no window matching it appeared yet. "
            "Call wait() then focus_window, or list_windows to check the exact title."
        )


# --- Progress trail (model-thought-driven) -------------------------------- #

@dataclass
class TrailEntry:
    iteration: int
    action: str
    target_key: str     # element_id or "x,y" or text snippet — used for thrashing detection
    thought: str
    result: str
    screen_changed: bool
    target_label: str = ""  # human-readable label of the element interacted with


class ProgressTrail:
    """
    Full history of all actions enriched with the model's own thoughts.

    Purpose: give the model complete memory of WHAT it has tried and
    WHY, without dragging full observations through the prompt. Rendered
    as a pinned block at the top of every new observation.
    """

    def __init__(self, max_entries: Optional[int] = None):
        self._entries: Deque[TrailEntry] = deque(maxlen=max_entries)

    def record(self, entry: TrailEntry) -> None:
        self._entries.append(entry)

    def entries(self) -> List[TrailEntry]:
        return list(self._entries)

    def export(self) -> List[Dict[str, Any]]:
        """Serialize entries for session persistence."""
        return [asdict(e) for e in self._entries]

    def load(self, entries: List[Dict[str, Any]]) -> None:
        """Restore entries from a persisted session (appends to current)."""
        for d in entries:
            try:
                self._entries.append(TrailEntry(**d))
            except TypeError:
                continue  # tolerate schema drift across versions

    def render(self) -> str:
        if not self._entries:
            return ""
        lines = ["PROGRESS SO FAR (all actions, thoughts, and outcomes — do NOT repeat failed approaches):"]
        for e in self._entries:
            status = "✓ screen changed" if e.screen_changed else "— no visible change"
            thought = e.thought.strip() or "(no thought)"
            target = f"{e.target_key}('{e.target_label}')" if e.target_label else e.target_key
            lines.append(
                f"  #{e.iteration} {e.action}({target}) — {thought} → {e.result}; {status}"
            )
        return "\n".join(lines)

    def thrashing_warning(self) -> Optional[str]:
        """
        Detect thrashing: look at the trailing run of consecutive no-change
        entries (ignoring any earlier screen changes). If the last 3+ entries
        all had no screen change and hit ≤3 distinct targets, warn.
        """
        all_entries = list(self._entries)
        # Walk backwards to find the longest trailing run with no screen change.
        tail = []
        for e in reversed(all_entries):
            if e.screen_changed:
                break
            tail.append(e)
        tail = list(reversed(tail))  # chronological order

        if len(tail) < 3:
            return None
        targets = {e.target_key for e in tail if e.target_key}
        if 0 < len(targets) <= 3:
            tlist = ", ".join(sorted(targets))
            return (
                f"⚠ THRASHING DETECTED: the last {len(tail)} actions all had NO visible "
                f"screen change and targeted only {{{tlist}}}. Stop repeating. "
                "Consider: (a) wait(2) in case the app is slow,"
                "(b) read your own thoughts and re-assess what you did and can do "
                "(c) try keyboard Tab to reach the target, "
                "(c) try a keyboard shortcut to submit instead of clicking, "
                "(d) focus a different window, "
                "(e) re-read the screen — the target may not be where you think it is."
            )
        return None


def target_key_from_args(action: str, args: Dict[str, Any]) -> str:
    """A short, stable string identifying the target of an action — for thrashing detection."""
    if action == "scroll":
        clicks = args.get("scroll_clicks", 0)
        direction = "up" if clicks > 0 else "down"
        return f"scroll({direction},{abs(clicks)})"
    if action in ("click", "double_click", "click_and_type"):
        eid = args.get("element_id")
        if eid:
            return eid
        x, y = args.get("x"), args.get("y")
        if x is not None and y is not None:
            # Bucket to 20px cells so near-duplicate clicks are seen as "the same"
            return f"{int(x) // 20 * 20},{int(y) // 20 * 20}"
        return ""
    if action in ("type", "type_text"):
        t = (args.get("text") or "")[:20]
        return f'text:"{t}"'
    if action == "keypress":
        return "+".join(args.get("keys", []))
    if action == "focus_window":
        return (args.get("title") or "")[:20]
    if action == "wait":
        return "wait"
    return ""


# --- Evidence check (generic task completion verification) --------------- #

def _extract_meaningful_words(text: str) -> List[str]:
    """Extract lowercased words of length >=4 OR any token containing digits.
    These are the tokens we require to be findable in OCR."""
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", text)
    out = []
    for t in tokens:
        if any(ch.isdigit() for ch in t) or len(t) >= 4:
            out.append(t.lower())
    return out


def _best_fuzzy_match(word: str, ocr_blob: str) -> Tuple[float, str]:
    """Find the best fuzzy match for `word` in the OCR blob. Returns (ratio, matched_token)."""
    best_ratio = 0.0
    best_tok = ""
    # Scan words of similar length (±2) for efficiency.
    for tok in re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", ocr_blob):
        if abs(len(tok) - len(word)) > 2:
            continue
        r = difflib.SequenceMatcher(None, word, tok.lower()).ratio()
        if r > best_ratio:
            best_ratio = r
            best_tok = tok
    return best_ratio, best_tok


def check_evidence_in_ocr(
    evidence: List[str],
    elements: List[Element],
    min_hit_ratio: float = 0.6,
    fuzzy_threshold: float = 0.82,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    For each evidence string, check whether a sufficient fraction of its
    meaningful words are present in the current OCR (exact or fuzzy match).

    Returns (all_passed, details) where `details` is a list of per-evidence
    dicts with keys 'evidence', 'passed', 'missing', 'near_misses'.
    Near-misses are surfaced so the model can see "you said 'Mustermann' but
    the screen reads 'Mustermnn' — probably a typing or OCR issue".
    """
    ocr_blob = " ".join(el.text for el in elements)
    ocr_blob_lower = ocr_blob.lower()

    details: List[Dict[str, Any]] = []
    all_passed = True

    for claim in evidence:
        claim = (claim or "").strip()
        if not claim:
            details.append({"evidence": claim, "passed": False,
                            "missing": ["(empty evidence string)"], "near_misses": []})
            all_passed = False
            continue

        words = _extract_meaningful_words(claim)
        if not words:
            # No substantive words to check — be lenient and accept, but note it.
            details.append({"evidence": claim, "passed": True,
                            "missing": [], "near_misses": [],
                            "note": "no checkable tokens"})
            continue

        missing: List[str] = []
        near_misses: List[Tuple[str, str, float]] = []  # (word, matched, ratio)
        for w in words:
            if w in ocr_blob_lower:
                continue
            ratio, tok = _best_fuzzy_match(w, ocr_blob)
            if ratio >= fuzzy_threshold:
                near_misses.append((w, tok, ratio))
                # Near-miss counts as a hit for the ratio, but we record it.
                continue
            missing.append(w)

        hit = len(words) - len(missing)
        passed = (hit / len(words)) >= min_hit_ratio
        if not passed:
            all_passed = False

        details.append({
            "evidence": claim,
            "passed": passed,
            "missing": missing,
            "near_misses": [f"'{w}' ≈ '{t}'" for w, t, _ in near_misses],
        })

    return all_passed, details


def log_evidence_summary(evidence: List[str], details: List[Dict[str, Any]]) -> None:
    """Short INFO log describing the evidence check outcome."""
    passed = sum(1 for d in details if d["passed"])
    log.info("evidence check: %d/%d passed (%d items total)",
             passed, len(details), len(evidence))
    for d in details:
        mark = "✓" if d["passed"] else "✗"
        extra = ""
        if d.get("missing"):
            extra = f"  missing: {', '.join(d['missing'])}"
        if d.get("near_misses"):
            extra += f"  near: {'; '.join(d['near_misses'])}"
        log.info("  %s %r%s", mark, d["evidence"], extra)


def format_evidence_rejection(details: List[Dict[str, Any]]) -> str:
    """Build a message telling the model exactly which evidence items failed."""
    lines = ["Evidence check FAILED — your cited evidence was not findable in OCR:"]
    for d in details:
        if d["passed"]:
            continue
        lines.append(f"  • '{d['evidence']}'")
        if d["missing"]:
            lines.append(f"      missing words: {', '.join(d['missing'])}")
        if d["near_misses"]:
            lines.append(f"      near-misses (typo / OCR garble?): {'; '.join(d['near_misses'])}")
    lines.append(
        "Look carefully at the current screen. Either (a) the action did not "
        "produce what you claimed — keep working — or (b) OCR garbled the text, "
        "in which case re-cite evidence using what OCR actually shows."
    )
    return "\n".join(lines)



class ConversationHistory:
    """
    Keeps:
      * System prompt (pinned)
      * Initial task (pinned, with first screenshot)
      * The last N observation exchanges WITH their images + full OCR
      * Older observation messages collapsed to one short line (no image)
      * Tool_call / tool_result pairs older than the last `keep_tool_turns`
        turns are removed from the outgoing stream entirely — their content
        lives on in the ProgressTrail, which is always part of the latest
        observation.
    """

    def __init__(self, system_prompt: str, keep_recent: int = 2, keep_tool_turns: int = 2):
        self._messages: List[dict] = [{"role": "system", "content": system_prompt}]
        self._keep_recent = keep_recent
        self._keep_tool_turns = keep_tool_turns
        self._task: str = ""

    def add_initial_user(self, task: str, element_text: str, screenshot_b64: str) -> None:
        """Send the first observation: screenshot → elements → task (no progress trail yet)."""
        self._task = task
        self._messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                {"type": "text", "text": f"{element_text}\n\nTASK: {task}"},
            ],
        })

    def add_assistant(self, message: dict) -> None:
        self._messages.append(message)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def ensure_tool_results_for(self, tool_calls: List[dict], placeholder: str = "(no action taken)") -> int:
        """
        Enforce the OpenAI message-shape invariant: every assistant tool_call
        must be followed (before the next assistant message) by a 'tool' role
        message with a matching tool_call_id. Call this after a tool_calls
        batch is processed — it appends placeholder responses for any ids
        that weren't already answered. Returns the number of placeholders added.

        Without this, a break mid-loop or an unrecognized tool name leaves
        dangling tool_calls which cause 400 Bad Request on the next turn.
        """
        existing = {
            m.get("tool_call_id")
            for m in self._messages
            if m.get("role") == "tool"
        }
        added = 0
        for tc in tool_calls:
            tc_id = tc.get("id")
            if not tc_id:
                continue
            if tc_id not in existing:
                self.add_tool_result(tc_id, placeholder)
                added += 1
        return added

    def add_observation(self, element_text: str, screenshot_b64: str, note: str = "") -> None:
        """Append an observation message ordered: screenshot → elements → task → progress."""
        text = element_text
        if self._task:
            text += f"\n\nTASK: {self._task}"
        if note:
            text += f"\n\n{note}"
        self._messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                {"type": "text", "text": text},
            ],
        })

    def add_nudge(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def messages_for_api(self) -> List[dict]:
        """
        Return a slimmed copy:
          - Old observation messages: dropped entirely (history lives in the ProgressTrail
            which is prepended to every new observation, so old OCR dumps and screenshots
            add no value)
          - Old assistant tool-call / tool-result pairs: dropped entirely
        We identify 'turns' by assistant tool_calls messages; the last
        `keep_tool_turns` of those (plus their tool results) are kept,
        earlier ones are removed.
        """
        msgs = [dict(m) for m in self._messages]
        to_drop: set = set()

        # Step 1: drop old observation (vision) messages entirely.
        vision_indices = [
            i for i, m in enumerate(msgs)
            if m.get("role") == "user" and isinstance(m.get("content"), list)
            and any(c.get("type") == "image_url" for c in m["content"])
        ]
        keep_vision = set(vision_indices[-self._keep_recent:])
        for i in vision_indices:
            if i not in keep_vision:
                to_drop.add(i)

        # Step 2: drop old tool-call / tool-result pairs.
        # Find assistant messages with tool_calls; keep the last N, drop earlier
        # (plus their matching 'tool' role messages by id).
        assistant_tc_indices = [
            i for i, m in enumerate(msgs)
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        keep_tc = set() if self._keep_tool_turns == 0 else set(assistant_tc_indices[-self._keep_tool_turns:])
        kept_tc_ids: set = set()
        for i in assistant_tc_indices:
            if i not in keep_tc:
                to_drop.add(i)
            else:
                for tc in (msgs[i].get("tool_calls") or []):
                    kept_tc_ids.add(tc.get("id"))

        # Drop 'tool' messages whose tool_call_id we've dropped.
        for i, m in enumerate(msgs):
            if m.get("role") == "tool" and m.get("tool_call_id") not in kept_tc_ids:
                to_drop.add(i)

        return [m for i, m in enumerate(msgs) if i not in to_drop]


# --- Agent ----------------------------------------------------------------- #

SYSTEM_PROMPT = """\
You are an expert AI agent controlling a computer via OCR + screenshot observation. \
You receive, every step: (1) an annotated screenshot with orange boxes for OCR \
elements and blue boxes for interactive regions, both labeled with stable IDs, plus \
a cyan crosshair marking your last click; (2) two element lists — \
"OCR TEXT ELEMENTS" in the form [id] 'text' @(cx,cy), and \
"INTERACTIVE REGIONS" in the form [id] <interactive> @(cx,cy) WxH (visually detected \
buttons, icons, and empty input fields that OCR cannot see); all coordinates are \
screen-absolute; (3) a TASK reminder; (4) a PROGRESS SO FAR block summarizing \
what you have done, what you thought, and what changed.

BEFORE EVERY ACTION — read the PROGRESS SO FAR block carefully. For each \
listed step, ask: Did that action succeed? Did the result build toward the \
goal? If a prior step produced useful information (e.g. a window list, a \
coordinate, a confirmed click), use it directly — do NOT repeat that step. \
Only pick an action that meaningfully advances on what is already known. \

REASONING PATTERN — for every action, include a 'thought' that covers:
  1. What changed since my last action, and does it match what I expected?
  2. What I am doing now.
  3. What I expect to happen next.
Keep it under 50 words. This is how you remember your plan and notice mistakes.

RULES:
- FIRST ACTION, ALWAYS: bring the correct application to the foreground before \
  any click/type/keypress/scroll. Derive WHICH application from the user's \
  request. Call focus_window (call list_windows first if you need the exact \
  titles). Input actions are BLOCKED by the executor until a window has been \
  focused this run — a focus is not optional. \
  - If the target application is not open (not in list_windows), call \
    open_application(name) to launch it, then continue. \
  - If it is genuinely unclear which application the request refers to and no \
    reasonable default exists, call ask_user(question) with a specific question \
    and stop; the user's reply continues this same conversation.
- Prefer element_id over raw x/y. Coordinates are screen-absolute. \
  If you must use raw x/y, always derive them from the @(cx,cy) values in the \
  current element list — it should always be in pixel coordinates, not in ratios. Use the \
  calculate tool to adjust (e.g. cx + 40) when the click target is beside a label.
- Form fields are often NOT detected by OCR when empty — only the label beside \
  them is. An empty field may appear as an INTERACTIVE REGION (blue box, <interactive>); \
  click it by element_id if present. Otherwise click just to the right of the \
  label, or use click_and_type with x/y to focus-and-fill in one step. Verify by \
  checking whether your typed text appears in the next OCR TEXT ELEMENTS list.
- INTERACTIVE REGIONS (<interactive>) have no text — use their element_id to click them. \
  They represent buttons, icons, and empty fields the OCR cannot read.
- If OCR shows text that should have been cleared is still there, your clear \
  attempt did NOT succeed — do not retype or you will append instead of replace. \
  Use this escalation ladder until one works: \
  (1) keypress(['ctrl','a']) then keypress(['backspace']) — backspace after select-all \
  reliably deletes the selection on all platforms including Citrix; do NOT use delete here \
  as it is an extended key that Citrix may not handle correctly; \
  (2) double_click the field to select the word under the cursor, then press \
  Delete to remove it — repeat for each remaining word to clear the field \
  word-by-word; this works on virtually every platform even when Ctrl+A does not; \
  (3) delete_chars(N) where N is the number of characters in the field — \
  this presses End first so cursor position does not matter.
- If the cyan crosshair in the screenshot is visibly off from your intended \
  target (e.g. an edit field and you clicked on a label), do NOT repeat the same \
  click blindly. Instead, use calculate to derive corrected coordinates \
  (e.g. element_cx + 40) from the nearest element's @(cx,cy) listed below, then \
  click with raw x/y.
- If the screen does not change after an action, do NOT repeat the same action. \
  Reassess. If thrashing is warned about, change approach entirely.
- If you expect a delay (save, dialog appearing, data loading), use wait() \
  explicitly — waits are not counted against the stuck-screen detector.
- To declare the task done, call finish_task. You MUST cite 2-5 short pieces \
  of evidence: specific text currently visible on screen that proves the task \
  succeeded. Quote the text literally as it appears. \
  CRITICAL: field labels, form titles, menu items, and other UI chrome that \
  was already on screen before you acted do NOT count as evidence — they prove \
  nothing. Cite the OUTCOME of your work: a value you typed that now shows in \
  a field, a confirmation/success message, a new row in a list, a changed \
  status. If you cannot find such outcome text on the screen, the task is \
  most likely not done — keep working instead of calling finish_task. \
  Your evidence is checked against OCR; bogus evidence is rejected and you \
  must keep working.
- You may issue MULTIPLE tool calls in a single response whenever you are \
  confident the actions are safe to run in sequence without seeing the \
  intermediate result first (e.g. list_windows → focus_window, or filling \
  several known fields with click_and_type, or a keypress followed by \
  type_text). Batch such steps into one response to save round-trips. \
  Do NOT batch actions whose target or arguments depend on what the screen \
  shows after a previous action in the same batch.
- Work in the UI's language (German or English).
"""

@dataclass
class AgentConfig:
    endpoint: str
    api_key: str
    model: str
    max_iterations: int = 40
    keep_recent_exchanges: int = 1
    keep_tool_turns: int = 0
    temperature: float = 0.1
    max_tokens: int = 1024
    request_timeout: int = 120
    request_retries: int = 2
    verify_tls: bool = False
    ocr_min_conf: int = 30
    save_debug_screenshots: bool = True
    # Navigator (goal-holding critic). When enabled, a cheap text-only role
    # re-assesses progress toward the goal each step and sets the actioner's
    # next objective. Set NAVIGATOR=0 in the environment to A/B against the
    # plain ReAct loop.
    use_navigator: bool = True
    navigator_max_tokens: int = 400
    # Narrator (on-demand VLM observer). Invoked only on ambiguous transitions
    # or when stuck, to catch visual-state changes the element diff misses.
    # Set NARRATOR=0 to disable. Requires the navigator.
    use_narrator: bool = True
    # Session continuity: when True, each run() loads the prior task's trail +
    # task list from the session file and continues the conversation, saving
    # again at the end. Startup and the "New conversation" control clear it.
    continue_session: bool = True


@dataclass
class NavigatorState:
    """The navigator's running belief: fixed goal + evolving objective/assessment."""
    ultimate_goal: str = ""
    current_intent: str = ""          # the objective the actioner is pursuing now
    last_status: str = ""             # on_track | off_track | stuck | goal_reached
    last_reasoning: str = ""
    guidance: str = ""                # tactical hint injected into the actioner's note
    stuck_rounds: int = 0             # consecutive navigator 'stuck'/'off_track' verdicts
    last_narration: str = ""          # most recent narrator observation, fed to the next assessment
    prior_tasks: List[str] = field(default_factory=list)  # earlier requests in this conversation

    def render_for_actioner(self) -> str:
        """The block prepended to the actioner's observation each step."""
        lines = [f"CURRENT OBJECTIVE (set by navigator): {self.current_intent}"]
        if self.guidance:
            lines.append(f"NAVIGATOR GUIDANCE: {self.guidance}")
        return "\n".join(lines)


# --- Session persistence (conversation continuity) ------------------------- #
#
# A conversation is a sequence of user requests that share context. Because the
# backend runs each task in a fresh subprocess, continuity is carried on disk:
# the effective context the model sees each step is the ProgressTrail (always
# re-injected into the latest observation), so persisting the trail + the task
# list is enough to "continue the whole conversation". Startup and the in-app
# "New conversation" control call clear_session().

def _temp_dir() -> Path:
    return Path(os.environ.get("TEMP", tempfile.gettempdir()))


def _session_path() -> Path:
    return _temp_dir() / "agent_session.json"


def _debug_screenshot_path() -> Path:
    return _temp_dir() / "agent_screenshot_debug.png"


def load_session() -> Dict[str, Any]:
    """Return the persisted session dict, or {} if none / unreadable."""
    try:
        return json.loads(_session_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_session(data: Dict[str, Any]) -> None:
    try:
        _session_path().write_text(json.dumps(data), encoding="utf-8")
    except Exception as exc:
        log.debug("session save failed: %s", exc)


def clear_session() -> None:
    """Wipe conversation context and the last annotated screenshot.

    Called on startup (always a fresh conversation) and by the in-app
    'New conversation' control during usage.
    """
    for p in (_session_path(), _debug_screenshot_path()):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.debug("clear_session: could not remove %s (%s)", p, exc)


class ComputerAgent:
    def __init__(self, config: AgentConfig):
        self.cfg = config
        _enable_dpi_awareness()

        if platform.system() == "Windows":
            import ctypes
            user32 = ctypes.windll.user32
            self.width, self.height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        else:
            self.width, self.height = pyautogui.size()
        log.info("Primary monitor: %dx%d", self.width, self.height)

        # Pre-warm both engines at startup (single-threaded) so the first
        # _parse_screen call pays zero init cost and parallel inference is safe.
        log.info("Initializing OCR engine ...")
        _get_rapidocr_engine()
        log.info("Initializing YOLO icon detector ...")
        _get_yolo_session()

        self.registry = ElementRegistry()
        self.executor = ActionExecutor(self.width, self.height, self.registry)
        self._last_ocr_signature: Optional[frozenset] = None
        self._no_change_streak = 0
        self._last_action_was_wait = False
        self._prev_screenshot: Optional[Image.Image] = None  # raw frame, for pixel-diff highlight

    # --- screen parsing --------------------------------------------------- #

    def _parse_screen(self) -> Tuple[str, List[Element], ChangeSet]:
        """
        Capture foreground window, run OCR, reconcile IDs, ANNOTATE the image
        (boxes + IDs + crosshair at last-click), return (base64 of annotated
        image, elements, change-set-vs-previous-frame).
        """
        t0 = time.monotonic()
        screenshot, (win_x, win_y, win_w, win_h) = capture_foreground()
        log.info("screenshot: %.2fs (window %dx%d at %d,%d)",
                 time.monotonic() - t0, win_w, win_h, win_x, win_y)

        t1 = time.monotonic()
        log.info("Starting OCR + icon detection (parallel) ...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            ocr_future = pool.submit(extract_ocr_elements, screenshot)
            icon_future = pool.submit(extract_icon_elements, screenshot)
            ocr_elements = ocr_future.result()
            icon_elements = icon_future.result()
        log.info(
            "OCR+YOLO: %.2fs \u2192 %d text / %d icon elements",
            time.monotonic() - t1, len(ocr_elements), len(icon_elements),
        )

        # Offset coords from window-relative to screen-absolute.
        for el in ocr_elements:
            el.x += win_x; el.y += win_y
            el.center_x += win_x; el.center_y += win_y
        for el in icon_elements:
            el.x += win_x; el.y += win_y
            el.center_x += win_x; el.center_y += win_y

        # Merge: keep all OCR elements; add YOLO icon elements that don't
        # overlap any OCR box by more than YOLO_OCR_OVERLAP_THRESH of the
        # YOLO box's own area. Using coverage (intersection/YOLO-area) rather
        # than IoU catches the common case where a small text label sits inside
        # a larger icon box — IoU would be tiny, but the boxes visually overlap.
        ocr_boxes = [
            {"x": el.x, "y": el.y, "w": el.width, "h": el.height}
            for el in ocr_elements
        ]
        merged: List[Element] = list(ocr_elements)
        for icon_el in icon_elements:
            icon_box = {"x": icon_el.x, "y": icon_el.y,
                        "w": icon_el.width, "h": icon_el.height}
            if any(_coverage(icon_box, ob) >= _YOLO_OCR_OVERLAP_THRESH for ob in ocr_boxes):
                continue  # overlaps an OCR box — skip
            merged.append(icon_el)

        reconciled, change = self.registry.reconcile(merged, self.width, self.height)

        # Pixel-diff highlight: mark what changed since the previous frame — but
        # skip on scroll/replaced, where "everything" changed and a wash is noise.
        highlight_regions: List[Tuple[int, int, int, int]] = []
        if (self._prev_screenshot is not None
                and change.transition not in ("scroll", "replaced")):
            try:
                highlight_regions = _diff_regions(self._prev_screenshot, screenshot)
            except Exception as exc:
                log.debug("pixel-diff highlight failed: %s", exc)
        self._prev_screenshot = screenshot  # raw frame for the next diff

        # Annotate — same image for LLM and for the debug screenshot file.
        annotated = annotate_screenshot(
            screenshot, reconciled, win_x, win_y,
            click_marker=self.executor.last_click_point,
            highlight_regions=highlight_regions,
        )

        if self.cfg.save_debug_screenshots:
            self._save_debug(annotated)

        buf = BytesIO()
        annotated.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        return b64, reconciled, change

    def _save_debug(self, annotated: Image.Image) -> None:
        """Write the SAME annotated image to the shared debug path."""
        try:
            out = Path(os.environ.get("TEMP", tempfile.gettempdir())) / "agent_screenshot_debug.png"
            annotated.save(out, format="PNG")
            log.info("[SCREENSHOT_READY]")
        except Exception as exc:
            log.debug("debug screenshot save failed: %s", exc)

    @staticmethod
    def _format_elements(elements: List[Element], limit: int = 200) -> str:
        """Format elements into two labeled sections: OCR text and YOLO icons."""
        ocr_els = sorted(
            [el for el in elements if el.source != "yolo"],
            key=lambda el: (el.center_y // 16, el.center_x),
        )
        yolo_els = sorted(
            [el for el in elements if el.source == "yolo"],
            key=lambda el: (el.center_y // 16, el.center_x),
        )
        parts: List[str] = []

        ocr_lines = [el.as_prompt_line() for el in ocr_els[:limit]]
        if len(ocr_els) > limit:
            ocr_lines.append(f"... (+{len(ocr_els) - limit} more, not shown)")
        parts.append(
            "OCR TEXT ELEMENTS:\n" + ("\n".join(ocr_lines) if ocr_lines else "(none)")
        )

        if yolo_els:
            yolo_limit = 100
            yolo_lines = [el.as_prompt_line() for el in yolo_els[:yolo_limit]]
            if len(yolo_els) > yolo_limit:
                yolo_lines.append(f"... (+{len(yolo_els) - yolo_limit} more, not shown)")
            parts.append(
                "INTERACTIVE REGIONS (icon detector):\n" + "\n".join(yolo_lines)
            )

        return "\n\n".join(parts) if parts else "(no elements detected)"

    # --- API call --------------------------------------------------------- #

    _CONTEXT_DUMP_PATH = Path(tempfile.gettempdir()) / "agent_context_dump.txt"

    def _dump_context(self, messages: List[dict]) -> None:
        """Write the current LLM message list to a human-readable file, overwriting each call.
        Images are replaced with a short placeholder to keep the file readable."""
        try:
            lines: List[str] = []
            for i, msg in enumerate(messages):
                role = msg.get("role", "?").upper()
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls")
                tool_call_id = msg.get("tool_call_id")

                lines.append(f"{'='*70}")
                header = f"[{i}] {role}"
                if tool_call_id:
                    header += f" (tool_call_id={tool_call_id})"
                lines.append(header)
                lines.append(f"{'='*70}")

                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            lines.append(part["text"])
                        elif part.get("type") == "image_url":
                            lines.append("[IMAGE]")
                elif isinstance(content, str) and content:
                    lines.append(content)

                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        lines.append(f"  TOOL_CALL: {fn.get('name')}  args={fn.get('arguments')}")

                lines.append("")

            self._CONTEXT_DUMP_PATH.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass  # never let debug output break the agent

    def _call_llm(
        self, messages: List[dict], tools: Optional[List[dict]] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        self._dump_context(messages)
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "tools": tools if tools is not None else COMPUTER_TOOLS,
            "tool_choice": "auto",
            "max_tokens": max_tokens if max_tokens is not None else self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "skip_special_tokens": False,
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Optional[Exception] = None
        for attempt in range(1 + self.cfg.request_retries):
            try:
                resp = requests.post(
                    self.cfg.endpoint, headers=headers, json=payload,
                    verify=self.cfg.verify_tls, timeout=self.cfg.request_timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                log.warning("API call failed (%s). Retry in %ds.", exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.cfg.request_retries + 1} attempts: {last_exc}")

    # --- navigator (goal-holding critic) ---------------------------------- #

    def _navigate(
        self,
        nav: "NavigatorState",
        change: ChangeSet,
        trail: ProgressTrail,
        elements_text: str,
        tokens: Dict[str, int],
    ) -> None:
        """Re-assess progress toward the ultimate goal from a neutral, text-only
        outside viewpoint, and update `nav` in place (objective + guidance). Runs
        in its own short context (not the actioner's history) — cheap, and the
        separation is what gives the 'neutral outside view'. Never executes input.
        """
        narration_block = (
            f"VISUAL OBSERVATION (from the narrator): {nav.last_narration}\n\n"
            if nav.last_narration else ""
        )
        prior_block = (
            f"EARLIER REQUESTS IN THIS CONVERSATION (already handled): "
            f"{'; '.join(nav.prior_tasks)}\n"
            if nav.prior_tasks else ""
        )
        user = (
            f"{prior_block}"
            f"ULTIMATE GOAL (current request): {nav.ultimate_goal}\n"
            f"CURRENT OBJECTIVE: {nav.current_intent or '(none yet — set the first objective)'}\n\n"
            f"{change.render()}\n\n"
            f"{narration_block}"
            f"{trail.render() or 'PROGRESS SO FAR: (nothing done yet)'}\n\n"
            f"CURRENT SCREEN ELEMENTS:\n{elements_text}\n\n"
            "Assess progress toward the ULTIMATE GOAL and set the next objective."
        )
        nav.last_narration = ""  # consumed — don't carry a stale narration forward
        messages = [
            {"role": "system", "content": NAVIGATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        try:
            resp = self._call_llm(
                messages, tools=NAVIGATOR_TOOLS, max_tokens=self.cfg.navigator_max_tokens
            )
        except Exception as exc:
            log.warning("Navigator call failed (%s) — keeping previous objective.", exc)
            return

        tokens["calls"] += 1
        if "usage" in resp:
            u = resp["usage"]
            tokens["input"] += u.get("prompt_tokens", 0)
            tokens["output"] += u.get("completion_tokens", 0)
            tokens["total"] += u.get("total_tokens", 0)

        tcs = (resp["choices"][0]["message"].get("tool_calls") or [])
        if not tcs:
            log.info("[NAV] no assessment returned — keeping objective %r", nav.current_intent)
            return
        try:
            args = json.loads(tcs[0]["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}

        nav.last_status = (args.get("status") or "").strip()
        nav.last_reasoning = (args.get("reasoning") or "").strip()
        next_intent = (args.get("next_intent") or "").strip()
        if next_intent:
            nav.current_intent = next_intent
        nav.guidance = (args.get("guidance") or "").strip()
        if nav.last_status == "goal_reached" and not nav.guidance:
            nav.guidance = (
                "Navigator believes the goal is reached — if you can cite OUTCOME "
                "evidence visible on screen, call finish_task; otherwise keep working."
            )
        if nav.last_status in ("off_track", "stuck"):
            nav.stuck_rounds += 1
        else:
            nav.stuck_rounds = 0

        log.info("[NAV] %s | %s", nav.last_status or "?", nav.last_reasoning)
        log.info("[NAV] → objective: %s", nav.current_intent)

    def _narrate(self, screenshot_b64: str, change: ChangeSet, tokens: Dict[str, int]) -> str:
        """On-demand VLM observer: describe visual/state changes the deterministic
        diff misses (greyed button, spinner, validation colour). Grounded in the
        change-set to curb hallucination. Returns a short string, or "" on failure.
        """
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
            {"type": "text", "text": (
                "Deterministic element diff since the last action (ground truth):\n"
                f"{change.render()}\n\n"
                "Describe what changed on the screen, especially visual/state changes "
                "not captured above. Call `describe` once."
            )},
        ]
        messages = [
            {"role": "system", "content": NARRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            resp = self._call_llm(
                messages, tools=NARRATOR_TOOLS, max_tokens=self.cfg.navigator_max_tokens
            )
        except Exception as exc:
            log.warning("Narrator call failed (%s).", exc)
            return ""

        tokens["calls"] += 1
        if "usage" in resp:
            u = resp["usage"]
            tokens["input"] += u.get("prompt_tokens", 0)
            tokens["output"] += u.get("completion_tokens", 0)
            tokens["total"] += u.get("total_tokens", 0)

        msg = resp["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        if not tcs:
            return (msg.get("content") or "").strip()
        try:
            args = json.loads(tcs[0]["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            return ""
        summary = (args.get("summary") or "").strip()
        visual = (args.get("visual_state_changes") or "").strip()
        if visual and visual.lower() not in ("none", "n/a", ""):
            return (summary + f" [visual: {visual}]").strip()
        return summary

    # --- main loop -------------------------------------------------------- #

    def run(self, instruction: str) -> Dict[str, Any]:
        start = time.time()
        start_dt = datetime.datetime.now()
        log.info("=" * 70)
        log.info("TASK: %s", instruction)
        log.info("=" * 70)

        # Reset per-run state (registry/executor are per-process anyway).
        self._no_change_streak = 0
        self._last_ocr_signature = None
        self._last_action_was_wait = False
        self._prev_screenshot = None
        self.registry = ElementRegistry()
        self.executor = ActionExecutor(self.width, self.height, self.registry)

        # Load prior conversation context for continuity (empty on a fresh
        # conversation — startup and "New conversation" clear the session file).
        session = load_session() if self.cfg.continue_session else {}
        prior_tasks: List[str] = session.get("tasks", [])
        continuing = bool(prior_tasks)
        iter_base = int(session.get("last_iteration", 0))
        if continuing:
            log.info("Continuing conversation: %d earlier request(s), %d trail entries carried.",
                     len(prior_tasks), len(session.get("trail", [])))

        # Deterministic focus BEFORE the first frame: on a continued conversation,
        # re-focus the window this conversation was already driving so the first
        # screenshot — and the navigator's first objective — are grounded in the
        # real application, not whatever happened to be frontmost (the browser UI).
        # On a fresh conversation there is no target yet; the executor's focus gate
        # then forces the model to focus/open the right app as its first action.
        target_window = session.get("target_window")
        if target_window:
            res = self.executor._focus_window(target_window)
            log.info("[FOCUS] startup re-focus of %r → %s", target_window, res)

        screenshot_b64, elements, _ = self._parse_screen()  # first frame: no change-set
        elements_text = self._format_elements(elements)
        self._last_ocr_signature = ocr_signature(elements)

        history = ConversationHistory(
            SYSTEM_PROMPT,
            keep_recent=self.cfg.keep_recent_exchanges,
            keep_tool_turns=self.cfg.keep_tool_turns,
        )

        actions_log: List[Dict[str, Any]] = []
        tokens = {"input": 0, "output": 0, "total": 0, "calls": 0}
        trail = ProgressTrail()
        trail.load(session.get("trail", []))  # continuity: prior actions/thoughts/outcomes
        nudge_count = 0
        pending_question: Optional[str] = None  # set if the model calls ask_user

        # Navigator: set the first objective from the goal + the current screen,
        # then prepend it to the actioner's first observation.
        nav: Optional[NavigatorState] = None
        if self.cfg.use_navigator:
            nav = NavigatorState(ultimate_goal=instruction, prior_tasks=prior_tasks)
            if self.executor.focused_once:
                # Grounded in the real app (continued conversation, or a startup
                # re-focus succeeded): let the navigator set the first objective
                # from the actual screen.
                self._navigate(nav, ChangeSet(), trail, elements_text, tokens)
            else:
                # Fresh conversation: the first frame is pre-focus (often the
                # agent's own UI), so the navigator would fabricate a bogus
                # objective like "click [e1]". Seed a fixed focus-first objective
                # instead; the navigator re-assesses in-loop once the correct
                # application is focused.
                nav.current_intent = (
                    "Bring the correct application for this request to the foreground FIRST "
                    "(focus_window, or open_application if it is not running). If it is unclear "
                    "which application the request refers to, ask_user. Do not click or type "
                    "until the right application is focused."
                )
            initial_text = nav.render_for_actioner() + "\n\n" + elements_text
        else:
            initial_text = elements_text

        # On a continued conversation, front-load the memory so the actioner
        # knows it is mid-conversation and sees everything it already did.
        if continuing:
            prior = "; ".join(prior_tasks)
            initial_text = (
                f"(CONTINUING CONVERSATION. Earlier requests you already handled: {prior}. "
                f"The screen is where that work left off; your full action history is below.)\n\n"
                f"{trail.render()}\n\n" + initial_text
            )
        history.add_initial_user(instruction, initial_text, screenshot_b64)

        for iteration in range(self.cfg.max_iterations):
            log.info("--- iteration %d ---", iteration + 1)

            try:
                response = self._call_llm(history.messages_for_api())
            except Exception as exc:
                log.error("LLM call terminally failed: %s", exc)
                actions_log.append({"iteration": iteration + 1, "error": str(exc)})
                break

            tokens["calls"] += 1
            if "usage" in response:
                u = response["usage"]
                tokens["input"] += u.get("prompt_tokens", 0)
                tokens["output"] += u.get("completion_tokens", 0)
                tokens["total"] += u.get("total_tokens", 0)

            choice = response["choices"][0]
            msg = choice["message"]
            history.add_assistant(msg)
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                text = (msg.get("content") or "").strip()
                log.info("Model text (no tool call): %s", text[:200])
                if any(p in text.lower() for p in ("task is complete", "task complete", "done")):
                    log.info("Model indicates completion in text — stopping.")
                    break
                nudge_count += 1
                if nudge_count >= 2:
                    log.info("Re-parsing screen after repeated text-only responses.")
                    screenshot_b64, elements, _ = self._parse_screen()
                    note = trail.render()
                    note = (note + "\n\n" if note else "") + \
                           "Please issue a tool call to make progress. Include a 'thought'."
                    history.add_observation(self._format_elements(elements), screenshot_b64, note=note)
                    nudge_count = 0
                else:
                    history.add_nudge("Please issue a tool call with a 'thought' to make progress.")
                continue

            nudge_count = 0
            task_done = False
            iteration_actions: List[Tuple[str, Dict[str, Any], str]] = []  # (fn_name, args, result)
            pre_action_elements = {el.stable_id: el for el in elements}  # capture before actions change the screen

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                thought = (args.get("thought") or "").strip()
                log.info("[TOOL] → %s(%s)", fn_name, {k: v for k, v in args.items() if k != "thought"})
                if thought:
                    log.info("[THOUGHT] %s", thought)

                if fn_name in ("finish_task", "task_complete"):
                    claim = args.get("message", "")
                    evidence = args.get("evidence") or []
                    if not isinstance(evidence, list):
                        evidence = [str(evidence)]

                    # Cheap check first: does the model's own evidence appear in the current OCR?
                    evidence_ok, ev_details = check_evidence_in_ocr(evidence, elements)
                    log_evidence_summary(evidence, ev_details)

                    if not evidence_ok:
                        rejection = format_evidence_rejection(ev_details)
                        history.add_tool_result(tc["id"], rejection)
                        actions_log.append({
                            "iteration": iteration + 1,
                            "action": "task_complete",
                            "claim": claim,
                            "evidence": evidence,
                            "evidence_check": ev_details,
                            "verified": False,
                            "reason": "evidence not found in current OCR",
                            "thought": thought,
                        })
                        # Do NOT break — the model should keep working. The rejection
                        # is now in its context and the trail will record this attempt.
                        iteration_actions.append((fn_name, args, "evidence check failed"))
                        continue

                    # Evidence cleared the cheap check. Escalate to LLM verification.
                    verified, reason = self._verify_completion(
                        instruction, claim, evidence, history, tokens, trail
                    )
                    history.add_tool_result(tc["id"], f"verification: {reason}")
                    actions_log.append({
                        "iteration": iteration + 1,
                        "action": "task_complete",
                        "claim": claim,
                        "evidence": evidence,
                        "evidence_check": ev_details,
                        "verified": verified,
                        "reason": reason,
                        "thought": thought,
                    })
                    if verified:
                        if claim.strip():
                            log.info("[TASK_RESULT] %s", claim.strip())
                        task_done = True
                    break

                if fn_name == "ask_user":
                    question = (args.get("question") or "").strip()
                    log.info("[QUESTION] %s", question)
                    history.add_tool_result(tc["id"], "asked the user; ending run to await their reply.")
                    actions_log.append({
                        "iteration": iteration + 1,
                        "action": "ask_user",
                        "question": question,
                        "thought": thought,
                    })
                    pending_question = question
                    task_done = True  # end the run; the user's reply continues this conversation
                    break

                action_type = "type" if fn_name == "type_text" else fn_name
                # Strip 'thought' before handing args to the executor.
                exec_args = {k: v for k, v in args.items() if k != "thought"}
                result = self.executor.execute(action_type, exec_args)
                log.info("   %s", result)
                history.add_tool_result(tc["id"], result)
                actions_log.append({
                    "iteration": iteration + 1,
                    "action": fn_name,
                    "args": args,
                    "result": result,
                    "thought": thought,
                })
                iteration_actions.append((fn_name, args, result))

            # Enforce the assistant.tool_calls <-> tool.tool_call_id invariant:
            # any tool_calls we broke out of (e.g. after finish_task) or didn't
            # recognize must still have a placeholder tool response or the next
            # API call will 400.
            dangling = history.ensure_tool_results_for(tool_calls)
            if dangling:
                log.debug("Added %d placeholder tool results for unprocessed calls", dangling)

            if task_done:
                break

            # Observe new screen state.
            time.sleep(0.2)
            self.executor.tick_click_age()
            screenshot_b64, elements, change = self._parse_screen()
            log.info("[CHANGE] %s", change.summary())
            new_sig = ocr_signature(elements)
            screen_changed = new_sig != self._last_ocr_signature
            self._last_ocr_signature = new_sig

            # Wait-aware stuck counter: don't count iterations where the model's
            # ONLY action was wait(). Any non-wait action ticks the counter.
            had_non_wait_action = any(a != "wait" for a, _, _ in iteration_actions)
            if screen_changed:
                self._no_change_streak = 0
            elif had_non_wait_action:
                self._no_change_streak += 1
            # (else: only waits this turn, don't tick)

            # Record one trail entry per action in this iteration.
            for fn_name, args, result in iteration_actions:
                thought = (args.get("thought") or "").strip()
                eid = args.get("element_id", "")
                el = pre_action_elements.get(eid)
                target_label = el.text.strip() if el else ""
                # Condense list_windows output: keep titles but drop the header line.
                trail_result = result
                if fn_name == "list_windows":
                    found = [line.strip().lstrip("- ") for line in result.splitlines()
                             if line.strip().startswith("-")]
                    trail_result = f"{len(found)} windows: {'; '.join(found)}"
                trail.record(TrailEntry(
                    iteration=iter_base + iteration + 1,
                    action=fn_name,
                    target_key=target_key_from_args(fn_name, args),
                    thought=thought,
                    result=trail_result,
                    screen_changed=screen_changed,
                    target_label=target_label,
                ))

            elements_text = self._format_elements(elements)

            # Narrator (expensive VLM) only when the cheap signals are insufficient:
            # an ambiguous transition (popup/replaced) or an ongoing stuck run. Runs
            # BEFORE the navigator so the navigator gets 'eyes' on the hard screen.
            narration = ""
            if (nav is not None and self.cfg.use_narrator
                    and (change.transition in ("popup", "replaced") or nav.stuck_rounds >= 2)):
                narration = self._narrate(screenshot_b64, change, tokens)
                if narration:
                    log.info("[NARRATOR] %s", narration)
                    nav.last_narration = narration  # consumed by the navigator below

            # Navigator re-assesses progress toward the goal (text-only, cheap) and
            # updates the objective/guidance the actioner sees next.
            if nav is not None:
                self._navigate(nav, change, trail, elements_text, tokens)

            # Build the next observation note: navigator objective first (the
            # steering signal), then the change-set (what the last action caused),
            # then any narrator observation, then the full trail and warnings.
            note_parts: List[str] = []
            if nav is not None:
                note_parts.append(nav.render_for_actioner())
            note_parts.append(change.render())
            if narration:
                note_parts.append("VISUAL OBSERVATION (narrator): " + narration)
            note_parts.append(trail.render())
            thrash = trail.thrashing_warning()
            if thrash:
                log.warning(thrash)
                note_parts.append(thrash)
            if self._no_change_streak >= 3:
                log.warning("Screen stuck for %d iterations", self._no_change_streak)
                note_parts.append(
                    "⚠ Screen unchanged for multiple iterations — try an entirely "
                    "different approach (different target, keyboard navigation, or wait)."
                )
            if self._no_change_streak >= 5:
                log.error("Screen unchanged for %d consecutive iterations — aborting.",
                          self._no_change_streak)
                actions_log.append({"iteration": iteration + 1, "error": "aborted: screen stuck"})
                break

            note = "\n\n".join(p for p in note_parts if p)
            history.add_observation(elements_text, screenshot_b64, note=note)

        duration = time.time() - start
        log.info("=" * 70)
        log.info("DONE in %dm%.1fs | actions=%d | api_calls=%d | tokens in/out/total=%d/%d/%d",
                 int(duration // 60), duration % 60,
                 len(actions_log), tokens["calls"],
                 tokens["input"], tokens["output"], tokens["total"])
        log.info("=" * 70)

        # Persist the conversation so the next request continues from here.
        if self.cfg.continue_session:
            entries = trail.entries()
            save_session({
                "tasks": prior_tasks + [instruction],
                "trail": trail.export(),
                "last_iteration": entries[-1].iteration if entries else iter_base,
                # Persist the focused app so follow-up requests re-focus it
                # deterministically. Falls back to the prior value if this run
                # never (re-)focused. Wiped by clear_session on 'New conversation'.
                "target_window": self.executor.target_window or session.get("target_window"),
            })

        return {
            "started_at": start_dt.isoformat(),
            "duration_seconds": duration,
            "actions": actions_log,
            "tokens": tokens,
            "question": pending_question,  # non-None if the run ended on ask_user
        }

    # --- completion verification ----------------------------------------- #

    def _verify_completion(
        self,
        task: str,
        claim: str,
        evidence: List[str],
        history: ConversationHistory,
        tokens: Dict[str, int],
        trail: ProgressTrail,
    ) -> Tuple[bool, str]:
        """
        Re-screenshot, re-check evidence against the FRESH OCR (things may
        have changed in the ~0.6s since the actor's claim), and if it still
        holds, ask the model with a restricted tool set (finish_task |
        continue_working). The verifier must itself cite evidence, which is
        also checked against OCR before acceptance. No auto-execution of
        corrective actions.
        """
        log.info("Verifying completion: %s", claim)
        time.sleep(0.1)
        self.executor.tick_click_age()
        screenshot_b64, elements, change = self._parse_screen()

        # Re-check actor's evidence against the fresh OCR.
        evidence_ok, ev_details = check_evidence_in_ocr(evidence, elements)
        log.info("verifier re-check of actor's evidence:")
        log_evidence_summary(evidence, ev_details)
        if not evidence_ok:
            rejection = format_evidence_rejection(ev_details)
            # Push into history so the actor sees the rejection on its next turn.
            history.add_observation(
                self._format_elements(elements), screenshot_b64,
                note=trail.render() + "\n\n" + rejection if trail.entries() else rejection,
            )
            return False, "evidence disappeared on re-check of fresh screen"

        elements_text = self._format_elements(elements)
        note_parts = [change.render()] if not change.is_empty else []
        if trail.entries():
            note_parts.append(trail.render())
        note_parts.append(
            f"VERIFICATION REQUEST: the actor claimed the task is done ('{claim}') "
            f"and cited evidence:\n  - " + "\n  - ".join(f"'{e}'" for e in evidence) +
            f"\nTask was: '{task}'. Look carefully at the current screen. "
            "If truly done, call finish_task with YOUR OWN evidence (text you see "
            "on the current screen). Otherwise call continue_working with a short "
            "reason."
        )
        history.add_observation(elements_text, screenshot_b64, note="\n\n".join(note_parts))

        try:
            response = self._call_llm(history.messages_for_api(), tools=_VERIFICATION_TOOLS)
        except Exception as exc:
            log.warning("Verification call failed: %s. NOT accepting claim.", exc)
            return False, f"api failure during verification — not accepting: {exc}"

        tokens["calls"] += 1
        if "usage" in response:
            u = response["usage"]
            tokens["input"] += u.get("prompt_tokens", 0)
            tokens["output"] += u.get("completion_tokens", 0)
            tokens["total"] += u.get("total_tokens", 0)

        vmsg = response["choices"][0]["message"]
        history.add_assistant(vmsg)
        vtcs = vmsg.get("tool_calls") or []
        decision: Optional[Tuple[bool, str]] = None
        if vtcs:
            tc = vtcs[0]
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            if fn in ("finish_task", "task_complete"):
                # Verifier must also ground its answer in visible evidence.
                v_evidence = args.get("evidence") or []
                if not isinstance(v_evidence, list):
                    v_evidence = [str(v_evidence)]
                v_ok, v_details = check_evidence_in_ocr(v_evidence, elements)
                log.info("verifier's own evidence check:")
                log_evidence_summary(v_evidence, v_details)
                if not v_ok:
                    history.add_tool_result(tc["id"], format_evidence_rejection(v_details))
                    decision = (False, "verifier cited evidence not findable in OCR")
                else:
                    history.add_tool_result(tc["id"], "verified complete")
                    decision = (True, "model confirmed after seeing current screen")
            elif fn == "continue_working":
                reason = args.get("reason", "no reason given")
                history.add_tool_result(tc["id"], f"noted: {reason}")
                decision = (False, f"model rejected claim: {reason}")
            else:
                # Unknown tool — close the id with a placeholder.
                history.add_tool_result(tc["id"], f"ignored: unknown tool '{fn}' during verification")
                decision = (False, f"verifier called unknown tool: {fn}")

        if decision is None:
            # Model emitted text or no tool calls at all.
            text = (vmsg.get("content") or "").strip()[:200]
            decision = (False, f"verifier did not call a recognized tool (said: {text!r})")

        # Safety net: ensure every verifier tool_call has a matching tool response.
        history.ensure_tool_results_for(vtcs, placeholder="(verifier response ignored)")
        return decision


# --- Entry point ----------------------------------------------------------- #

def _load_config() -> AgentConfig:
    profile = os.getenv("ACTIVE_PROFILE").upper()
    endpoint = os.getenv(f"{profile}_ENDPOINT")
    api_key = os.getenv(f"{profile}_API_KEY")
    model = os.getenv(f"{profile}_MODEL")
    if not endpoint or not model:
        raise SystemExit(
            f"Missing configuration. Set {profile}_ENDPOINT and {profile}_MODEL "
            "(or LLM_ENDPOINT / LLM_MODEL) in your .env file."
        )
    use_navigator = os.getenv("NAVIGATOR", "1").strip().lower() not in ("0", "false", "no", "off")
    use_narrator = os.getenv("NARRATOR", "1").strip().lower() not in ("0", "false", "no", "off")
    continue_session = os.getenv("CONTINUE_SESSION", "1").strip().lower() not in ("0", "false", "no", "off")
    return AgentConfig(
        endpoint=endpoint, api_key=api_key, model=model,
        use_navigator=use_navigator, use_narrator=use_narrator,
        continue_session=continue_session,
    )


def interactive(agent: ComputerAgent) -> None:
    print("\n" + "=" * 72)
    print("COMPUTER AGENT")
    print("=" * 72)
    print("Type an instruction, 'new' to start a fresh conversation, or 'quit' to exit.\n")
    while True:
        try:
            instr = input("task > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not instr:
            continue
        if instr.lower() in {"quit", "exit", "q"}:
            break
        if instr.lower() in {"new", "reset"}:
            clear_session()
            print("Started a fresh conversation (context and last screenshot cleared).")
            continue
        try:
            agent.run(instr)
        except Exception:
            log.exception("Task failed")


def main() -> None:
    cfg = _load_config()
    log.info("Profile endpoint: %s | model: %s", cfg.endpoint, cfg.model)
    clear_session()  # startup is always a fresh conversation
    agent = ComputerAgent(cfg)
    interactive(agent)


if __name__ == "__main__":
    main()