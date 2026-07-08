#!/usr/bin/env python3
"""
OCR-only computer-use agent (refactored).

Operates on foreground-window screenshots + RapidOCR, designed for
generalist navigation of native Windows apps and Citrix-hosted remote apps
where no structural UI tree is available.

Design notes (differences from the previous version):
  * HIERARCHICAL supervisor/actioner split. The SUPERVISOR owns a dynamically
    growing todo list (Plan/TodoItem) and is the outer control loop: it is
    called at sub-task BOUNDARIES (item done/blocked, budget exhausted) and on
    ALARMS (wrong-field mutation, stuck screen) — not after every action. The
    ACTIONER is a bounded subroutine that only ever sees the SINGLE current
    item (never the user's full request): information asymmetry is what makes
    the hierarchy real.
  * Machine-verified progress: item ticks and final completion require OCR
    evidence; where an item declares expected_value/expected_label, evidence
    must ALSO sit next to the right field label (label-anchored geometry).
    A deterministic wrong-field guard runs after every action and raises a
    supervisor alarm the moment a value lands next to the wrong label.
  * Completion is supervisor-owned: the actioner reports subtask_done /
    subtask_blocked; only the supervisor may declare task_complete, and only
    with machine-checked final evidence and no open items.
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
  * DwmGetWindowAttribute for true window bounds (fixes edge-click misses).

Public surface preserved for backend/frontend compatibility:
  * ComputerAgent, AgentConfig, _load_config, interactive, main
  * agent.run(instruction) -> dict with 'started_at', 'duration_seconds',
    'actions', 'tokens', 'question'
  * actions_log entries retain 'iteration', 'action', 'args', 'result' keys;
    the final supervisor completion is a 'task_complete' entry with 'claim',
    'verified', 'reason' (the backend extracts the answer from it)
  * Debug screenshot written to $TEMP/agent_screenshot_debug.png each parse
  * logging.getLogger("agent") is the channel the backend hooks
  * Log markers [SCREENSHOT_READY], [TASK_RESULT] preserved; new markers
    [TODOS] (full plan as one-line JSON) and [TODO] (single tick/add events)
    drive the frontend checklist
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
# Fraction of an OCR box that must be covered by a YOLO box for the OCR element
# to inherit the 'interactive' attribute (text INSIDE a control, e.g. a filled
# edit field or a labeled button). Deliberately much higher than the suppression
# threshold: a label merely grazing a field box suppresses the redundant blue
# overlay but must NOT be tagged interactive itself.
_YOLO_TEXT_FUSE_THRESH = float(os.getenv("YOLO_TEXT_FUSE_THRESH", "0.5"))

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
            "name": "subtask_done",
            "description": (
                "Declare the CURRENT TASK (the single task assigned by your "
                "supervisor) finished. You MUST cite evidence: specific text "
                "currently visible on screen that proves THIS task's outcome. "
                "CRITICAL: do NOT cite field labels, form titles, menu items, or "
                "other UI chrome that was already there before you acted — those "
                "prove nothing. Cite the OUTCOME of your work: a value you typed "
                "that now shows in a field, a confirmation/success message, a row "
                "that now appears in a list, a status label that changed. The claim "
                "is verified against a fresh screenshot; a false claim is rejected "
                "and you must keep working."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Short summary of what was done for this task."
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "1-4 strings, each being a SHORT, VERBATIM snippet copied "
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
            "name": "subtask_blocked",
            "description": (
                "Report that you cannot complete the CURRENT TASK: required "
                "information is missing, the expected control/screen does not "
                "exist, or repeated attempts keep failing. The supervisor will "
                "replan (it may rephrase the task, take another route, or ask the "
                "user). Do NOT use this to skip normal work — try at least a "
                "couple of different approaches first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Concrete, specific reason — what you tried, what the "
                            "screen shows instead, what is missing."
                        ),
                    },
                    "thought": _THOUGHT_PARAM,
                },
                "required": ["reason", "thought"],
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

# --- Supervisor (plan-owning controller) ------------------------------------ #
#
# The supervisor is the hierarchical superior of the actioner. It never touches
# the mouse or keyboard. It owns the PLAN — a dynamically growing todo list —
# and the actioner is only ever shown the SINGLE item currently assigned to it
# (never the user's full request). The supervisor is invoked at sub-task
# BOUNDARIES (item completed/blocked, action budget exhausted) and on ALARMS
# (wrong-field mutation, stuck screen), not after every action: per-click
# micromanagement is where dual-model systems thrash. Item completion and final
# task completion are machine-verified against OCR (presence + label adjacency)
# before the supervisor may tick them.

SUPERVISOR_SYSTEM_PROMPT = """\
You are the SUPERVISOR of a computer-use agent operating a hospital information \
system (HIS). You never touch the mouse or keyboard — a separate ACTIONER does. \
You are the actioner's boss: you own the PLAN, a growing todo list, and the \
actioner is only ever shown the SINGLE item you assign it. It never sees the \
user's full request — what you write in an item (and its context field) is ALL \
the actioner knows about the work.

The HIS layout is not known in advance, so you cannot write the whole plan \
upfront. Work with a rolling frontier: keep only 1-3 concrete pending items \
ahead, then EXTEND or REVISE the plan every time you are called, based on what \
the screen now shows.

WRITING GOOD ITEMS
- Each item must be achievable in a handful of actions (open a menu, fill one \
or two fields, save a form). Split anything bigger.
- Each item must be self-contained: put every fact the actioner needs (names, \
values to type, which record/patient) into the item text or its context field — \
the actioner cannot see the user request or the other items.
- ONLY for items that ENTER or CHANGE data: set expected_value (the exact text \
that should appear) and expected_label (the field label EXACTLY as written on \
screen, e.g. 'Last Name *' — never a description like 'button' or an invented \
label). These are machine-verified against OCR, and typing into a WRONG field \
is auto-detected from them — they are your early-warning system. Leave BOTH \
empty for navigation/click items (open a form, press save); presence evidence \
covers those.

WHEN YOU ARE CALLED you receive: the ultimate goal, the todo list, WHY you are \
called (item completed / blocked / budget exhausted / wrong-field alarm / stuck \
/ new request), what changed on screen, recent actions, and the current screen \
elements. Call `update_plan` exactly once:
- current_item_verdict — judge the ▶ item the actioner just worked on. 'done' \
ONLY when its outcome is confirmed on screen (a machine evidence check has \
already run; its result is shown to you — trust it over the actioner's claim). \
'not_done' keeps the item active for another round; 'failed' abandons it (then \
add a replacement item that takes a different route).
- add_items — ONLY genuinely NEW work. The todo list shown to you is the \
COMPLETE plan: pending (○) items are already queued — do NOT re-add them; \
duplicates of open items are dropped automatically. Items that FIX A MISTAKE \
(value typed into the wrong field, wrong record opened, stray dialog, corrupted \
data) get corrective=true and jump to the FRONT of the queue: mistakes are \
fixed before any new work.
- obsolete_item_ids — close OPEN items that should not be worked: duplicates, \
superseded plans, or work already covered by completed items. Working a stale \
item re-executes actions against already-saved data — close it instead.
- control:
    'continue'      — keep working (the normal case).
    'ask_user'      — something only the user can resolve (missing information, \
ambiguous requirement); put the specific question in control_detail.
    'stop'          — the goal is unreachable or the same failure keeps \
repeating despite replanning; put the reason in control_detail.
    'task_complete' — EVERYTHING is done. Allowed only when no items are open \
(close leftover duplicates/superseded items via obsolete_item_ids IN THE SAME \
call). Provide final_evidence: 2-4 OUTCOME values currently visible on screen, \
each with the field label it sits next to when applicable. They are \
machine-checked against OCR — field labels alone, form titles, and menu names \
prove nothing. Put a short user-facing summary of what was accomplished in \
control_detail.

RULES
- Never mark an item done because the actioner says so — only on verified \
screen evidence.
- A popup/dialog or screen replacement must be dealt with before anything \
else: insert a corrective item for it.
- If the same item keeps failing (blocked or budget-exhausted twice), do NOT \
reissue it unchanged — rephrase it, split it, take a different route through \
the UI, or stop.
- Use the exact field labels and values as they appear in the UI's language \
(German or English).
- Be decisive and brief. You are the boss, not a commentator.
"""

SUPERVISOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": (
                "Assess the event you were called for, give a verdict on the "
                "current todo item, extend/revise the plan, and decide how to "
                "proceed. Call exactly once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "assessment": {
                        "type": "string",
                        "description": "ONE sentence: where the task stands after this event.",
                    },
                    "current_item_verdict": {
                        "type": "string",
                        "enum": ["done", "not_done", "failed", "no_current_item"],
                        "description": (
                            "Verdict on the ▶ item the actioner just worked on. 'done' only "
                            "with confirmed on-screen outcome; 'not_done' keeps it active; "
                            "'failed' abandons it (add a replacement)."
                        ),
                    },
                    "verdict_reason": {
                        "type": "string",
                        "description": "One short line justifying the verdict.",
                    },
                    "add_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": (
                                        "Concrete, self-contained instruction for the actioner, "
                                        "achievable in a handful of actions."
                                    ),
                                },
                                "context": {
                                    "type": "string",
                                    "description": (
                                        "Facts the actioner needs for this item (values, names, "
                                        "record IDs). The actioner sees nothing else."
                                    ),
                                },
                                "expected_value": {
                                    "type": "string",
                                    "description": (
                                        "DATA-ENTRY items only: exact text that must appear on "
                                        "screen when this item succeeds. Leave empty for "
                                        "navigation/click items."
                                    ),
                                },
                                "expected_label": {
                                    "type": "string",
                                    "description": (
                                        "DATA-ENTRY items only: the field label EXACTLY as visible "
                                        "on screen (e.g. 'Last Name *'). Never a description like "
                                        "'button'. Leave empty for navigation/click items."
                                    ),
                                },
                                "corrective": {
                                    "type": "boolean",
                                    "description": "true = fixes a mistake/side-effect; jumps to the FRONT of the queue.",
                                },
                            },
                            "required": ["text"],
                        },
                        "description": "New todo items. Omit or leave empty when the plan already covers the next steps.",
                    },
                    "obsolete_item_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Ids of OPEN items that should NOT be worked: duplicates, "
                            "superseded plans, or work already satisfied by completed "
                            "items. They are closed as 'skipped' without execution. "
                            "Clear such leftovers with (or before) task_complete — "
                            "completion is rejected while they stay open, and working "
                            "them would redo actions against already-saved data."
                        ),
                    },
                    "guidance": {
                        "type": "string",
                        "description": "Optional ONE-line tactical hint shown to the actioner with its next item.",
                    },
                    "control": {
                        "type": "string",
                        "enum": ["continue", "ask_user", "stop", "task_complete"],
                    },
                    "control_detail": {
                        "type": "string",
                        "description": (
                            "ask_user: the question. stop: the reason. task_complete: a short "
                            "user-facing summary of what was accomplished."
                        ),
                    },
                    "final_evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string", "description": "Verbatim OCR outcome text visible on screen."},
                                "label": {"type": "string", "description": "Field label the value sits next to (empty if not applicable)."},
                            },
                            "required": ["value"],
                        },
                        "description": "task_complete only: 2-4 outcome values currently visible on screen.",
                    },
                },
                "required": ["assessment", "current_item_verdict", "control"],
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
    # OCR text that sits INSIDE a YOLO-detected control (filled edit field,
    # labeled button): the redundant YOLO box is dropped at merge time, but the
    # "this is clickable" information is fused onto the text element.
    interactive: bool = False

    def as_prompt_line(self) -> str:
        if self.source == "yolo":
            return (
                f"[{self.stable_id}] <interactive> @({self.center_x},{self.center_y})"
                f" {self.width}x{self.height}px"
            )
        text = self.text.replace("\n", " ").strip()
        if len(text) > 60:
            text = text[:57] + "..."
        suffix = " <interactive>" if self.interactive else ""
        return f"[{self.stable_id}] '{text}' @({self.center_x},{self.center_y}){suffix}"


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
    # Screen-absolute position of a blinking text caret, detected as a byproduct
    # of decaret (the caret is REMOVED from the screenshot, so this is the ONLY
    # signal that a click landed in an edit field — clicking into a field often
    # causes no other visible change).
    caret_xy: Optional[Tuple[int, int]] = None

    @property
    def is_empty(self) -> bool:
        return not (self.appeared or self.disappeared or self.moved or self.text_changed)

    def render(self) -> str:
        """Human/LLM-readable 'SINCE YOUR LAST ACTION' block. Empty string if nothing changed."""
        base = self._render_base()
        if self.caret_xy:
            base += (
                f"\nFOCUS INDICATOR: a text caret is blinking at "
                f"({self.caret_xy[0]},{self.caret_xy[1]}) — the edit field there IS "
                "focused. (The caret is removed from the screenshot; a click into a "
                "field often causes no other visible change. Do NOT re-click — type "
                "or select text now.)"
            )
        return base

    def _render_base(self) -> str:
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
        s = self._summary_base()
        if self.caret_xy:
            s += f"; caret blinking at ({self.caret_xy[0]},{self.caret_xy[1]}) — field focused"
        return s

    def _summary_base(self) -> str:
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


def _foreground_rect_win() -> Optional[Tuple[int, int, int, int]]:
    """Screen-absolute (x, y, w, h) of the foreground window on Windows, or None."""
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
    return (x, y, w, h) if (w > 0 and h > 0) else None


def capture_foreground() -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Screenshot only the foreground window. Returns (image, (x, y, w, h)) where
    x, y are screen-absolute. OCR coords from the image must be offset by
    (x, y) to become screen-absolute before clicking.
    """
    if platform.system() == "Windows":
        try:
            rect = _foreground_rect_win()
            if rect is not None:
                x, y, w, h = rect
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


# --- Caret removal (temporal de-flicker before OCR) ------------------------ #
#
# A focused edit field draws a blinking text caret; RapidOCR reads its thin
# vertical stroke as 'l'/'I'/'|'/'1', and when it abuts typed text it merges
# into the word ("Doe" -> "Doel"). Post-OCR filtering cannot undo the merge, so
# we remove the caret *before* OCR: capture a few frames across the blink cycle,
# find pixels that change between them (the caret, plus any incidental
# animation), and inpaint those regions from their surroundings. Polarity-
# agnostic (fills from neighbouring background), so it works for a dark caret on
# a light field or vice versa. Requires the caret to blink (>=1 'off' frame).

# 4 frames x 0.18s spans 0.54s — just over the ~0.53s ON phase of the standard
# Windows caret blink, so at least one caret-OFF frame is guaranteed. (3 frames
# spanned only 0.36s and could land entirely inside the ON phase, letting the
# caret survive into OCR as a phantom 'l'/'I'.)
_DECARET_FRAMES = int(os.getenv("DECARET_FRAMES", "4"))
_DECARET_INTERVAL_S = float(os.getenv("DECARET_INTERVAL_S", "0.18"))
_DECARET_DIFF_THRESH = int(os.getenv("DECARET_DIFF_THRESH", "26"))  # 0-255 grayscale delta
_DECARET_RING_PX = int(os.getenv("DECARET_RING_PX", "3"))          # width of the surrounding ring sampled per region

# Actions after which a focused edit field (and thus a blinking caret) is likely,
# so the next observation should be captured with caret removal enabled.
_CARET_INDUCING_ACTIONS = frozenset({
    "click", "double_click", "click_and_type", "type", "type_text", "keypress",
})


def capture_foreground_burst(
    n: int = _DECARET_FRAMES, interval_s: float = _DECARET_INTERVAL_S
) -> Tuple[List[Image.Image], Tuple[int, int, int, int]]:
    """Capture *n* screenshots of the foreground window *interval_s* apart, all
    aligned to the same rect, so a blinking caret is caught in multiple phases.
    Returns (frames, (x, y, w, h)). Falls back to a single frame on any error."""
    if platform.system() == "Windows":
        try:
            rect = _foreground_rect_win()
            if rect is not None:
                x, y, w, h = rect
                frames: List[Image.Image] = []
                for i in range(max(1, n)):
                    if i:
                        time.sleep(interval_s)
                    frames.append(pyautogui.screenshot(region=(x, y, w, h)))
                return frames, rect
        except Exception as exc:
            log.warning("Foreground burst capture failed (%s); single frame.", exc)
    img, rect = capture_foreground()
    return [img], rect


def _decaret(frames: List[Image.Image]) -> Tuple[Image.Image, List[Tuple[int, int]]]:
    """Return (healed_image, caret_centers): the first frame with the blinking
    caret (and any other pixels that changed between frames) replaced by the
    REAL pixels from the frame where the caret is off — never a synthesised/
    inpainted colour — plus the centers (image-relative x, y) of healed regions
    whose shape looks like a text caret (thin vertical bar). The caret is the
    ONLY visible evidence that a click focused an edit field, and healing it
    erases that evidence from the screenshot — so it is reported instead.

    Method: diff the frames to a change mask, group it into regions (each a
    blinking element), and for each region sample the field colour in a thin ring
    that FOLLOWS THE REGION'S SHAPE (dilate(region) - region), excluding every
    other changed pixel so a neighbouring blinker cannot pollute it. Then pick the
    frame whose region best matches that surrounding colour (the caret-off frame)
    and copy its real pixels. This stays correct when the caret abuts text
    ('Doe|' -> 'Doe') and when several elements blink independently. Needs the
    caret off in >=1 frame; with <2 frames or on any error, returns frame 0."""
    if not frames:
        raise ValueError("_decaret: no frames")
    ref = frames[0]
    if len(frames) < 2:
        return ref, []
    try:
        import cv2
        arrs = [np.asarray(f.convert("RGB")).astype(np.uint8) for f in frames]
        h = min(a.shape[0] for a in arrs)
        w = min(a.shape[1] for a in arrs)
        arrs = [a[:h, :w] for a in arrs]                 # guard off-by-one size drift
        stack = np.stack(arrs, axis=0).astype(np.int16)  # (N, h, w, 3)
        lum = stack.mean(axis=3)                          # (N, h, w)
        spread = lum.max(axis=0) - lum.min(axis=0)        # per-pixel temporal range
        mask = (spread > _DECARET_DIFF_THRESH).astype(np.uint8)
        changed = int(mask.sum())
        if changed == 0:
            return ref, []
        ring_k = np.ones((2 * _DECARET_RING_PX + 1, 2 * _DECARET_RING_PX + 1), np.uint8)
        n_labels, labels = cv2.connectedComponents(mask, connectivity=8)
        out = arrs[0].copy()
        carets: List[Tuple[int, int]] = []
        for lbl in range(1, n_labels):
            region = labels == lbl
            # Caret-shaped blinker? Thin vertical bar: much taller than wide,
            # text-line sized. Recorded BEFORE healing erases the evidence.
            ys, xs = np.nonzero(region)
            rw = int(xs.max() - xs.min() + 1)
            rh = int(ys.max() - ys.min() + 1)
            if rw <= 8 and 8 <= rh <= 64 and rh >= 2 * rw:
                carets.append((int(xs.mean()), int(ys.mean())))
            # Shape-following ring just outside the region; never sample another
            # changed pixel (mask == 0), so a nearby blinker can't bias the colour.
            ring = (cv2.dilate(region.astype(np.uint8), ring_k) > 0) & (mask == 0)
            if not ring.any():
                continue
            ref_color = np.median(stack[0][ring], axis=0)   # surrounding field colour
            best = min(
                range(len(arrs)),
                key=lambda f, _r=region, _c=ref_color: float(
                    np.abs(np.median(stack[f][_r], axis=0) - _c).sum()
                ),
            )
            out[region] = arrs[best][region]                # copy the caret-off frame's real pixels
        log.info("decaret: healed %d changed px in %d region(s) across %d frames%s",
                 changed, n_labels - 1, len(frames),
                 f"; caret detected at {carets[0]}" if carets else "")
        return Image.fromarray(out), carets
    except Exception as exc:
        log.warning("_decaret failed (%s: %s) — using raw frame.", type(exc).__name__, exc)
        return ref, []


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
        # OCR elements: orange. YOLO icon elements — and OCR text fused with a
        # YOLO control (text inside a clickable field/button): blue.
        is_yolo = el.source == "yolo"
        is_clickable = is_yolo or el.interactive
        color_full = (50, 150, 255, 220) if is_clickable else (255, 140, 0, 220)
        color_faint = (50, 150, 255, 90) if is_clickable else (255, 140, 0, 90)
        label_color = (130, 210, 255) if is_clickable else (255, 200, 80)
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
            # Models copy the id straight from the prompt line '[e75] ...' —
            # tolerate the brackets instead of burning a round on an error.
            eid = str(eid).strip().strip("[]")
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

    def render(self, limit: Optional[int] = None) -> str:
        if not self._entries:
            return ""
        entries = list(self._entries)
        if limit is not None and len(entries) > limit:
            header = (f"PROGRESS SO FAR (last {limit} of {len(entries)} actions, "
                      "with thoughts and outcomes):")
            entries = entries[-limit:]
        else:
            header = ("PROGRESS SO FAR (all actions, thoughts, and outcomes — "
                      "do NOT repeat failed approaches):")
        lines = [header]
        for e in entries:
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


# --- Spatial evidence (label-anchored checks + wrong-field guard) ---------- #
#
# Presence-only evidence has a hole: a value typed into the WRONG field still
# passes, because the text is on screen — just in the wrong place. These helpers
# anchor a value to the field label it must sit next to, using pure geometry on
# the OCR elements (no LLM cost):
#   * check_value_near_label — used to tick a todo item and for final evidence.
#   * guard_wrong_field      — runs after every action on the deterministic
#     change-set; fires an alarm the moment a value lands next to a label other
#     than the expected one (the #1 form-filling mistake).

def _fuzzy_text_match(needle: str, hay: str, thresh: float = 0.8) -> bool:
    """True if `needle` matches `hay` (containment or whole-string fuzzy)."""
    n = (needle or "").strip().lower()
    h = (hay or "").strip().lower()
    if not n or not h:
        return False
    if n in h or h in n:
        return True
    return difflib.SequenceMatcher(None, n, h).ratio() >= thresh


def _value_in_text(value: str, text: str, fuzzy_threshold: float = 0.85) -> bool:
    """Does element `text` contain the (possibly OCR-garbled) `value`?

    Mirrors check_evidence_in_ocr's word logic, scoped to one element: at least
    60% of the value's meaningful words must be present (exact or fuzzy)."""
    if not text:
        return False
    words = _extract_meaningful_words(value)
    if not words:
        return _fuzzy_text_match(value, text, fuzzy_threshold)
    text_lower = text.lower()
    hits = 0
    for w in words:
        if w in text_lower:
            hits += 1
            continue
        ratio, _tok = _best_fuzzy_match(w, text)
        if ratio >= fuzzy_threshold:
            hits += 1
    return (hits / len(words)) >= 0.6


def _label_in_text(label: str, text: str) -> bool:
    """Does `text` begin with (or contain as words) the field label?

    Catches the very common OCR merge of label and value into ONE element
    ('Id: P00004' → 'IdP00004', 'Nachname: Doe' → 'Nachname Doe'), where
    box-to-box adjacency can never match."""
    def norm(s: str) -> str:
        return " ".join(re.sub(r"[:*]", " ", (s or "").lower()).split())
    ln, tn = norm(label), norm(text)
    if not ln or not tn:
        return False
    return tn.startswith(ln) or f" {ln} " in f" {tn} "


def _find_label_elements(label: str, elements: List[Element]) -> List[Element]:
    """OCR elements whose text matches `label` (fuzzy, tolerant of ':' etc.)."""
    out = []
    for el in elements:
        if el.source == "yolo" or not (el.text or "").strip():
            continue
        el_text = el.text.strip().rstrip(":").strip()
        if _fuzzy_text_match(label.strip().rstrip(":"), el_text, 0.75):
            out.append(el)
    return out


def _is_value_adjacent_to_label(value_el: Element, label_el: Element) -> bool:
    """Geometric adjacency for form layouts: value on the same row to the right
    of the label, or directly below it."""
    if value_el is label_el:
        return False
    # Same row, value to the right of the label.
    row_tol = max(14, int(0.8 * max(value_el.height, label_el.height)))
    if abs(value_el.center_y - label_el.center_y) <= row_tol:
        gap = value_el.x - (label_el.x + label_el.width)
        if -10 <= gap <= 400:
            return True
    # Value directly below the label (label-above-field layout).
    vgap = value_el.y - (label_el.y + label_el.height)
    if 0 <= vgap <= 70:
        overlap = (min(value_el.x + value_el.width, label_el.x + label_el.width)
                   - max(value_el.x, label_el.x))
        if overlap > 0 or abs(value_el.x - label_el.x) <= 40:
            return True
    return False


def _nearest_label(el: Element, elements: List[Element]) -> Optional[Element]:
    """The label element `el` most plausibly belongs to (left on the same row,
    or directly above), or None."""
    best, best_d = None, float("inf")
    for cand in elements:
        if cand is el or cand.source == "yolo":
            continue
        text = (cand.text or "").strip()
        if not text or not any(ch.isalpha() for ch in text):
            continue
        if _is_value_adjacent_to_label(el, cand):
            d = abs(el.center_y - cand.center_y) + max(0, el.x - (cand.x + cand.width))
            if d < best_d:
                best, best_d = cand, d
    return best


def check_value_near_label(
    value: str, label: str, elements: List[Element]
) -> Tuple[str, str]:
    """Label-anchored evidence check. Returns (verdict, detail):
      'pass'     — the value is on screen adjacent to (or merged with) the label.
      'fail'     — the value is missing, or present only next to OTHER labels.
      'no_label' — the expected label is not on screen (fall back to presence)."""
    value_els = [el for el in elements
                 if el.source != "yolo" and _value_in_text(value, el.text)]
    # Same-element case first: OCR often merges 'Label: value' into one box,
    # and box-to-box adjacency can never match an element against itself.
    for v_el in value_els:
        if _label_in_text(label, v_el.text):
            return "pass", f"'{v_el.text.strip()}' contains both '{label}' and the value"
    label_els = _find_label_elements(label, elements)
    if not label_els:
        return "no_label", f"label '{label}' not found on screen (OCR may have missed it)"
    if not value_els:
        return "fail", f"value '{value}' is not visible anywhere on screen"
    for v_el in value_els:
        for l_el in label_els:
            if _is_value_adjacent_to_label(v_el, l_el):
                return "pass", (
                    f"'{v_el.text.strip()}' found next to '{l_el.text.strip()}'"
                )
    near = _nearest_label(value_els[0], elements)
    where = f" (it appears next to '{near.text.strip()}')" if near else ""
    return "fail", f"value '{value}' is on screen but NOT next to '{label}'{where}"


def guard_wrong_field(
    change: "ChangeSet", todo: "TodoItem", elements: List[Element]
) -> Optional[str]:
    """Deterministic wrong-field detector, run after every action.

    If the current todo expects `value` next to `label` and the change-set shows
    that value materializing ONLY next to DIFFERENT labels, return an alarm
    string; else None. Pure geometry — no LLM call.

    Echo suppression: applications commonly mirror an edited value elsewhere on
    the same screen (page title, breadcrumb, list row — e.g. HospitalRun shows
    'John Doe-Doe' in the header the moment the Last Name field is edited).
    So the alarm fires only when the value landed EXCLUSIVELY in wrong places:
    one hit next to the expected label proves the edit itself was correct, and
    the other occurrences are echoes, not mistakes."""
    if todo is None or not todo.expected_value or not todo.expected_label:
        return None
    # Scan text_changed AND appeared: typing into an EMPTY field registers as
    # 'appeared' (there was no old element to pair with), not 'text_changed' —
    # without this, the correct-field hit is invisible and an echo elsewhere
    # (page title) raises a false alarm.
    mutations = change.text_changed + change.appeared
    if not mutations:
        return None
    expected_labels = _find_label_elements(todo.expected_label, elements)
    if not expected_labels:
        # The expected label is not OCR-visible — geometry cannot distinguish
        # right from wrong field, so do not alarm (avoid false positives).
        return None
    wrong_hits: List[Element] = []
    for ch in mutations:
        el = ch.element
        if not _value_in_text(todo.expected_value, el.text):
            continue
        if (_label_in_text(todo.expected_label, el.text)
                or any(_is_value_adjacent_to_label(el, l_el) for l_el in expected_labels)):
            return None  # landed in the right field — other occurrences are echoes
        wrong_hits.append(el)
    if not wrong_hits:
        return None
    el = wrong_hits[0]
    near = _nearest_label(el, elements)
    near_txt = f"'{near.text.strip()}'" if near else "an unidentified field"
    return (
        f"the value '{todo.expected_value}' just appeared next to {near_txt}, "
        f"but it was expected next to '{todo.expected_label}'. The actioner "
        f"probably typed into the wrong field."
    )


def _extract_tool_args_from_content(
    content: str, known_keys: Optional[frozenset] = None
) -> Dict[str, Any]:
    """Best-effort recovery of tool-call arguments from plain text content.

    Some models/servers leak the function call into `content` — raw JSON, a
    fenced ```json block, JSON wrapped in prose, or a
    {"name": ..., "arguments": {...}} envelope — instead of populating
    tool_calls. Scans every '{' with a tolerant raw_decode, preferring the
    first object that contains one of `known_keys`. Returns {} on failure."""
    if not content:
        return {}

    def unwrap(obj: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("arguments", "parameters"):
            inner = obj.get(key)
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except json.JSONDecodeError:
                    inner = None
            if isinstance(inner, dict):
                return inner
        return obj

    decoder = json.JSONDecoder()
    fallback: Dict[str, Any] = {}
    for m in re.finditer(r"\{", content):
        try:
            obj, _end = decoder.raw_decode(content[m.start():])
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        obj = unwrap(obj)
        if not isinstance(obj, dict):
            continue
        if known_keys and known_keys & set(obj.keys()):
            return obj
        if not fallback:
            fallback = obj
        if not known_keys:
            return obj
    return fallback


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

    def add_initial_user(self, task_block: str, element_text: str, screenshot_b64: str) -> None:
        """Send the first observation: screenshot → elements → current task block.

        `task_block` is the supervisor-rendered CURRENT TASK block (see
        render_current_task) — the actioner's ONLY view of the work. It is
        re-appended to every observation and swapped via set_task() whenever
        the supervisor assigns the next todo item."""
        self._task = task_block
        self._messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                {"type": "text", "text": f"{element_text}\n\n{task_block}"},
            ],
        })

    def set_task(self, task_block: str) -> None:
        """Swap the CURRENT TASK block (supervisor assigned the next item)."""
        self._task = task_block

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
            text += f"\n\n{self._task}"
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
You are the ACTIONER of a two-level computer-use agent, controlling a computer \
via OCR + screenshot observation. A SUPERVISOR owns the overall plan and assigns \
you ONE task at a time in a "CURRENT TASK" block — that block is your entire \
job. Do NOT pursue anything beyond the current task: no extra fields, no \
navigation "while you are at it", no guessing at the larger goal. \
You receive, every step: (1) an annotated screenshot with orange boxes for OCR \
elements and blue boxes for interactive regions, both labeled with stable IDs, plus \
a cyan crosshair marking your last click; (2) two element lists — \
"OCR TEXT ELEMENTS" in the form [id] 'text' @(cx,cy) — entries suffixed \
<interactive> are text sitting INSIDE a clickable control (a FILLED edit field, \
a labeled button): that element IS the control, click it to interact — and \
"INTERACTIVE REGIONS" in the form [id] <interactive> @(cx,cy) WxH (visually detected \
buttons, icons, and EMPTY input fields that OCR cannot see); all coordinates are \
screen-absolute; (3) the CURRENT TASK block; (4) a PROGRESS SO FAR block summarizing \
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
  any click/type/keypress/scroll. Derive WHICH application from the CURRENT \
  TASK and its CONTEXT. Call focus_window (call list_windows first if you need \
  the exact titles). Input actions are BLOCKED by the executor until a window \
  has been focused this run — a focus is not optional. \
  - If the target application is not open (not in list_windows), call \
    open_application(name) to launch it, then continue. \
  - If it is genuinely unclear which application the task refers to and no \
    reasonable default exists, call subtask_blocked with a specific reason; \
    the supervisor will resolve it (possibly by asking the user).
- Prefer element_id over raw x/y. Coordinates are screen-absolute. \
  If you must use raw x/y, always derive them from the @(cx,cy) values in the \
  current element list — it should always be in pixel coordinates, not in ratios. Use the \
  calculate tool to adjust (e.g. cx + 40) when the click target is beside a label.
- To EDIT a field that ALREADY CONTAINS text (e.g. change an existing last \
  name): the existing text element IS the field — click that text element \
  (ideally one tagged <interactive>), then clear and retype. Do NOT hunt for a \
  separate empty INTERACTIVE REGION for it; a filled field is not listed there.
- EMPTY form fields are often NOT detected by OCR — only the label beside \
  them is. An empty field may appear as an INTERACTIVE REGION (blue box, <interactive>); \
  click it by element_id if present. Otherwise click just to the right of the \
  label, or use click_and_type with x/y to focus-and-fill in one step. Verify by \
  checking whether your typed text appears in the next OCR TEXT ELEMENTS list.
- INTERACTIVE REGIONS (<interactive> section) have no text — use their element_id to \
  click them. They represent buttons, icons, and EMPTY fields the OCR cannot read.
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
- Clicking INTO an edit field usually causes NO visible change — the blinking \
  caret is removed from screenshots. Watch for the FOCUS INDICATOR line ("a text \
  caret is blinking at (x,y)") in the observation: it means your click DID focus \
  the field at that position. Proceed with Ctrl+A / typing — do NOT click again \
  or switch to another element. Judge the edit by the text change afterwards.
- If a warning/error dialog appears, READ its message in the OCR elements \
  BEFORE dismissing it — it usually names the cause. If that cause is outside \
  your CURRENT TASK (e.g. "required fields missing" while your task is only to \
  press save), dismiss the dialog and call subtask_blocked QUOTING the message. \
  Never repeat the action that raised the dialog hoping for a different result.
- If the screen does not change after an action, do NOT repeat the same action. \
  Reassess. If thrashing is warned about, change approach entirely.
- If you expect a delay (save, dialog appearing, data loading), use wait() \
  explicitly — waits are not counted against the stuck-screen detector.
- To declare the CURRENT TASK done, call subtask_done. You MUST cite 1-4 short \
  pieces of evidence: specific text currently visible on screen that proves \
  THIS task's outcome. Quote the text literally as it appears. \
  CRITICAL: field labels, form titles, menu items, and other UI chrome that \
  was already on screen before you acted do NOT count as evidence — they prove \
  nothing. Cite the OUTCOME of your work: a value you typed that now shows in \
  a field, a confirmation/success message, a new row in a list, a changed \
  status. If you cannot find such outcome text on the screen, the task is \
  most likely not done — keep working instead of calling subtask_done. \
  Your evidence is checked against OCR (including WHERE it appears, next to \
  which label); bogus evidence is rejected and you must keep working. \
- If you cannot complete the CURRENT TASK — the control does not exist, \
  information is missing, or several different approaches failed — call \
  subtask_blocked with a concrete reason. That is the correct move; do NOT \
  drift into other work or silently pick a different goal.
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
    # Supervisor (plan-owning controller). It is integral to the architecture —
    # it owns the todo list, assigns one item at a time to the actioner, and is
    # the only role that can complete the overall task. NAVIGATOR=0 is accepted
    # for backwards compatibility but only logs a warning.
    use_navigator: bool = True
    # The supervisor writes plans (multiple todo items + verdict), so it needs
    # more room than the old per-step navigator did.
    navigator_max_tokens: int = 700
    # Actioner iterations allowed per todo item before the supervisor is forced
    # to review (revise/split/fail the item). Sub-task boundaries — not every
    # action — are where supervision happens; this bounds a runaway sub-task.
    subtask_max_iterations: int = 8
    # Optional stronger model for the supervisor (text-only reasoning).
    # None → use the actioner's model. Same endpoint/key (shared proxy); only the
    # model id differs. Set via NAVIGATOR_MODEL or NAVIGATOR_PROFILE in .env.
    navigator_model: Optional[str] = None
    # Narrator (on-demand VLM observer). Invoked only on ambiguous transitions
    # or when stuck, to catch visual-state changes the element diff misses.
    # Set NARRATOR=0 to disable. Requires the navigator.
    use_narrator: bool = True
    # Session continuity: when True, each run() loads the prior task's trail +
    # task list from the session file and continues the conversation, saving
    # again at the end. Startup and the "New conversation" control clear it.
    continue_session: bool = True


# --- Plan (the supervisor's dynamically growing todo list) ------------------ #

_STATUS_GLYPHS = {"pending": "○", "in_progress": "▶", "done": "✓", "failed": "✗",
                  "skipped": "−"}


@dataclass
class TodoItem:
    """One work item in the supervisor's plan. The actioner only ever sees the
    single item assigned to it (text + context + expected outcome)."""
    id: int
    text: str
    status: str = "pending"        # pending | in_progress | done | failed
    context: str = ""              # facts the actioner needs (values, names, IDs)
    expected_value: str = ""       # exact text that must appear on success
    expected_label: str = ""       # field label the value must appear next to
    corrective: bool = False       # fixes a mistake — scheduled before normal items
    kind: str = "normal"           # normal | focus (auto-completes on window focus)
    attempts: int = 0              # boundaries hit while this item was active
    notes: str = ""                # supervisor notes / failure reason


class Plan:
    """Ordered todo list with stable ids. Selection order: corrective pending
    items first (insertion order), then normal pending items."""

    def __init__(self) -> None:
        self._items: List[TodoItem] = []
        self._next_id = 1

    def items(self) -> List[TodoItem]:
        return list(self._items)

    def add(self, text: str, *, context: str = "", expected_value: str = "",
            expected_label: str = "", corrective: bool = False,
            kind: str = "normal") -> TodoItem:
        item = TodoItem(
            id=self._next_id,
            text=" ".join((text or "").split()),   # keep [TODOS] marker single-line
            context=(context or "").strip(),
            expected_value=(expected_value or "").strip(),
            expected_label=(expected_label or "").strip(),
            corrective=bool(corrective),
            kind=kind,
        )
        self._next_id += 1
        self._items.append(item)
        return item

    def current(self) -> Optional[TodoItem]:
        for it in self._items:
            if it.status == "in_progress":
                return it
        return None

    def activate_next(self) -> Optional[TodoItem]:
        """Promote the next pending item (corrective first) to in_progress."""
        if self.current() is not None:
            return self.current()
        pending = [it for it in self._items if it.status == "pending"]
        if not pending:
            return None
        nxt = next((it for it in pending if it.corrective), pending[0])
        nxt.status = "in_progress"
        return nxt

    def mark(self, item: TodoItem, status: str, note: str = "") -> None:
        item.status = status
        if note:
            item.notes = note

    def has_open_items(self) -> bool:
        return any(it.status in ("pending", "in_progress") for it in self._items)

    def get_by_id(self, item_id: int) -> Optional[TodoItem]:
        return next((it for it in self._items if it.id == item_id), None)

    def find_open_duplicate(self, text: str, expected_value: str = "",
                            expected_label: str = "") -> Optional[TodoItem]:
        """An OPEN (pending/in_progress) item duplicating the given one.

        PRECISE matches only: identical normalized text, or an identical
        non-empty expected value+label pair. Deliberately NO fuzzy text
        matching — form items are templated ("Fill in the X field with 'Y'"),
        so near-identical wording with a different field/value is NORMAL, and
        a fuzzy threshold once classified "Fill in the Last Name field with
        'Doe'" as a duplicate of the First Name item, silently deleting a
        required step from the plan. The failure asymmetry rules: a duplicate
        slipping through costs one redundant verify cycle (and can be swept
        via obsolete_item_ids); a false drop invisibly removes required work."""
        norm = " ".join((text or "").lower().split())
        ev = (expected_value or "").strip().lower()
        el = (expected_label or "").strip().lower()
        for it in self._items:
            if it.status not in ("pending", "in_progress"):
                continue
            if norm and norm == " ".join(it.text.lower().split()):
                return it
            if (ev and el
                    and ev == it.expected_value.strip().lower()
                    and el == it.expected_label.strip().lower()):
                return it
        return None

    def pending_corrective(self) -> Optional[TodoItem]:
        return next((it for it in self._items
                     if it.corrective and it.status in ("pending", "in_progress")), None)

    def render(self, max_done: int = 6) -> str:
        """Multi-line view for the supervisor prompt (old completed items elided)."""
        if not self._items:
            return "  (empty — nothing planned yet)"
        closed = [it for it in self._items if it.status in ("done", "failed", "skipped")]
        shown_closed_ids = {id(it) for it in closed[-max_done:]}
        hidden = len(closed) - len(shown_closed_ids)
        lines: List[str] = []
        if hidden > 0:
            lines.append(f"  … {hidden} earlier completed item(s) not shown")
        for it in self._items:
            if it.status in ("done", "failed", "skipped") and id(it) not in shown_closed_ids:
                continue
            extra: List[str] = []
            if it.corrective:
                extra.append("CORRECTIVE")
            if it.expected_value and it.expected_label:
                extra.append(f"expect '{it.expected_value}' near '{it.expected_label}'")
            if it.attempts:
                extra.append(f"attempts={it.attempts}")
            if it.notes:
                extra.append(it.notes)
            suffix = f"  ({'; '.join(extra)})" if extra else ""
            lines.append(f"  [{it.id}] {_STATUS_GLYPHS.get(it.status, '?')} {it.text}{suffix}")
        return "\n".join(lines)

    def to_marker(self) -> str:
        """Compact single-line JSON for the [TODOS] log marker (drives the UI)."""
        return json.dumps(
            [{"id": it.id, "text": it.text, "status": it.status,
              "corrective": it.corrective} for it in self._items],
            ensure_ascii=False, separators=(",", ":"),
        )

    def export(self) -> List[Dict[str, Any]]:
        return [asdict(it) for it in self._items]

    def load(self, entries: List[Dict[str, Any]]) -> None:
        for d in entries:
            try:
                item = TodoItem(**d)
            except TypeError:
                continue  # tolerate schema drift across versions
            if item.status == "in_progress":   # process died mid-item — retry it
                item.status = "pending"
            self._items.append(item)
            self._next_id = max(self._next_id, item.id + 1)


@dataclass
class SupervisorState:
    """The supervisor's running state: fixed goal + the plan it owns."""
    ultimate_goal: str = ""
    prior_tasks: List[str] = field(default_factory=list)
    plan: Plan = field(default_factory=Plan)
    guidance: str = ""            # one-line hint handed to the actioner with its next item
    last_assessment: str = ""
    control: str = "continue"     # continue | ask_user | stop | task_complete
    control_detail: str = ""
    final_evidence: List[Dict[str, str]] = field(default_factory=list)
    last_narration: str = ""      # narrator observation consumed by the next call
    fail_note: str = ""           # e.g. a rejected task_complete — shown at the next call


def render_current_task(item: TodoItem, guidance: str = "") -> str:
    """The CURRENT TASK block — the ONLY view of the work the actioner gets."""
    lines = [f"CURRENT TASK (assigned by your supervisor — do ONLY this): {item.text}"]
    if item.context:
        lines.append(f"CONTEXT: {item.context}")
    if item.expected_value and item.expected_label:
        lines.append(
            f"EXPECTED OUTCOME: '{item.expected_value}' visible next to "
            f"'{item.expected_label}'."
        )
    elif item.expected_value:
        lines.append(f"EXPECTED OUTCOME: '{item.expected_value}' visible on screen.")
    if guidance:
        lines.append(f"SUPERVISOR GUIDANCE: {guidance}")
    lines.append(
        "When the outcome is visible on screen, call subtask_done with verbatim "
        "OCR evidence. If you cannot complete this task, call subtask_blocked "
        "with a concrete reason."
    )
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
        # Unchanged-screen fast path: cached result of the last full parse.
        self._last_elements: Optional[List[Element]] = None
        self._last_rect: Optional[Tuple[int, int, int, int]] = None

    # --- screen parsing --------------------------------------------------- #

    def _parse_screen(self, decaret: bool = False) -> Tuple[str, List[Element], ChangeSet]:
        """
        Capture foreground window, run OCR, reconcile IDs, ANNOTATE the image
        (boxes + IDs + crosshair at last-click), return (base64 of annotated
        image, elements, change-set-vs-previous-frame).

        When *decaret* is True, capture a short burst of frames and remove the
        blinking text caret before OCR (see _decaret) — used after actions that
        focus/edit a field, where the caret corrupts field text OCR.
        """
        t0 = time.monotonic()
        caret_centers: List[Tuple[int, int]] = []
        if decaret:
            frames, (win_x, win_y, win_w, win_h) = capture_foreground_burst()
            screenshot, caret_centers = _decaret(frames)
        else:
            screenshot, (win_x, win_y, win_w, win_h) = capture_foreground()
        log.info("screenshot: %.2fs (window %dx%d at %d,%d)%s",
                 time.monotonic() - t0, win_w, win_h, win_x, win_y,
                 " [decaret]" if decaret else "")

        # Fast path: BIT-IDENTICAL (RGB-exact) to the previous frame in the same
        # window rect → OCR/YOLO are deterministic, so their output would be
        # identical too; reuse the cached elements and skip 4-8s of inference.
        # Exact equality, deliberately: any tolerance can swallow a real hint
        # (a single small glyph is <24px; a red→green indicator of equal
        # luminance is invisible to a grayscale diff), and the cost asymmetry
        # favors strictness — a false "changed" wastes one parse, a false
        # "unchanged" hides a real change from the agent. Static-window capture
        # is deterministic (composited framebuffer), so exactness still fires
        # in practice. Same-rect required: cached coords are screen-absolute.
        rect = (win_x, win_y, win_w, win_h)
        if (self._prev_screenshot is not None and self._last_elements is not None
                and rect == self._last_rect):
            a = np.asarray(self._prev_screenshot.convert("RGB"), dtype=np.uint8)
            b = np.asarray(screenshot.convert("RGB"), dtype=np.uint8)
            identical = a.shape == b.shape and np.array_equal(a, b)
            if a.shape == b.shape and not identical:
                # Diagnostics: a tiny diff defeating the fast path is either a
                # real small hint (correctly triggering a full parse) or an
                # animation/healing artifact eating the speedup — make it
                # visible so the trade-off can be judged from real runs.
                n_diff = int((a != b).any(axis=2).sum())
                if n_diff <= 200:
                    log.info("fast path missed: %d px differ (small hint, animation, "
                             "or healing artifact) — full parse", n_diff)
            if identical:
                log.info("screen unchanged — OCR/YOLO skipped, cached elements reused")
                self._prev_screenshot = screenshot
                change = ChangeSet()
                if caret_centers:
                    change.caret_xy = (caret_centers[0][0] + win_x,
                                       caret_centers[0][1] + win_y)
                annotated = annotate_screenshot(
                    screenshot, self._last_elements, win_x, win_y,
                    click_marker=self.executor.last_click_point,
                )
                if self.cfg.save_debug_screenshots:
                    self._save_debug(annotated)
                buf = BytesIO()
                annotated.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode(), self._last_elements, change

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
            coverages = [_coverage(icon_box, ob) for ob in ocr_boxes]
            if any(c >= _YOLO_OCR_OVERLAP_THRESH for c in coverages):
                # Redundant blue overlay — drop the YOLO box, but FUSE its
                # meaning onto OCR text that sits mostly inside it: that text
                # IS a clickable control (filled edit field, labeled button).
                # Without this the "interactive" information would be lost and
                # a filled field would look like inert text.
                for el, cov in zip(ocr_elements, coverages):
                    if cov >= _YOLO_TEXT_FUSE_THRESH:
                        el.interactive = True
                continue
            merged.append(icon_el)

        reconciled, change = self.registry.reconcile(merged, self.width, self.height)
        self._last_elements = reconciled     # cache for the unchanged-screen fast path
        self._last_rect = rect
        if caret_centers:
            # Surface the (healed-away) caret as a focus indicator, screen-absolute.
            change.caret_xy = (caret_centers[0][0] + win_x, caret_centers[0][1] + win_y)

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
    def _format_elements(elements: List[Element], limit: int = 200,
                         include_yolo: bool = True) -> str:
        """Format elements into two labeled sections: OCR text and YOLO icons.

        include_yolo=False omits the INTERACTIVE REGIONS section — YOLO lines
        carry no text, so they are pure prefill cost for roles that never click
        (the supervisor plans from labels/values, not from anonymous boxes)."""
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

        if yolo_els and include_yolo:
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
        max_tokens: Optional[int] = None, model: Optional[str] = None,
    ) -> dict:
        self._dump_context(messages)
        payload = {
            "model": model or self.cfg.model,
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

    # --- supervisor (plan-owning controller) ------------------------------- #

    def _supervise(
        self,
        sup: "SupervisorState",
        boundary: str,
        change: ChangeSet,
        trail: ProgressTrail,
        elements: List[Element],
        elements_text: str,
        tokens: Dict[str, int],
    ) -> bool:
        """One supervisor round at a sub-task boundary: present the event, the
        plan, and the screen; apply the resulting update_plan (verdict, new
        items, control) to `sup` in place. Runs in its own short context —
        the information asymmetry (the actioner never sees the goal or the
        plan) is what makes the hierarchy real. Returns False if the LLM call
        failed (plan left unchanged)."""
        prior_block = (
            f"EARLIER REQUESTS IN THIS CONVERSATION (already handled): "
            f"{'; '.join(sup.prior_tasks)}\n\n"
            if sup.prior_tasks else ""
        )
        parts: List[str] = [
            f"{prior_block}ULTIMATE GOAL (the user's request): {sup.ultimate_goal}",
            f"TODO LIST (your plan — the actioner sees ONLY the ▶ item):\n{sup.plan.render()}",
            f"WHY YOU ARE CALLED NOW: {boundary}",
        ]
        if sup.fail_note:
            parts.append(f"⚠ {sup.fail_note}")
            sup.fail_note = ""
        change_block = change.render()
        if change_block:
            parts.append(change_block)
        if sup.last_narration:
            parts.append(f"VISUAL OBSERVATION (from the narrator): {sup.last_narration}")
            sup.last_narration = ""
        parts.append(trail.render(limit=20) or "PROGRESS SO FAR: (nothing done yet)")
        # Full element view (OCR + interactive regions): after OCR/YOLO fusion
        # the surviving <interactive> lines are mostly EMPTY fields and icon
        # buttons — exactly the "what could be filled/clicked next" signal a
        # planner needs, and there are few of them, so the token cost is small.
        parts.append(f"CURRENT SCREEN ELEMENTS:\n{elements_text}")
        parts.append(
            "Call update_plan exactly once: verdict on the ▶ item, extend/revise "
            "the plan if needed, and set control."
        )
        messages = [
            {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        try:
            resp = self._call_llm(
                messages, tools=SUPERVISOR_TOOLS, max_tokens=self.cfg.navigator_max_tokens,
                model=self.cfg.navigator_model,
            )
        except Exception as exc:
            log.warning("Supervisor call failed (%s) — plan unchanged.", exc)
            return False

        tokens["calls"] += 1
        if "usage" in resp:
            u = resp["usage"]
            tokens["input"] += u.get("prompt_tokens", 0)
            tokens["output"] += u.get("completion_tokens", 0)
            tokens["total"] += u.get("total_tokens", 0)

        sup_choice = resp["choices"][0]
        sup_msg = sup_choice["message"]
        tcs = sup_msg.get("tool_calls") or []
        if tcs:
            try:
                args = json.loads(tcs[0]["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
        else:
            # Fallback: smaller models sometimes emit the update_plan JSON as
            # plain text instead of a proper tool call. Recover it rather than
            # stalling the run on a silent no-op.
            known = frozenset({"assessment", "current_item_verdict", "add_items", "control"})
            args = _extract_tool_args_from_content(sup_msg.get("content") or "", known)
            if not (isinstance(args, dict) and known & set(args.keys())):
                snippet = " ".join((sup_msg.get("content") or "").split())[:180]
                finish = sup_choice.get("finish_reason") or "?"
                # finish_reason distinguishes truncation ('length' → raise
                # navigator_max_tokens) from a genuinely empty/prose reply.
                log.info("[INTERVENE] supervisor produced no usable update_plan — plan unchanged "
                         "(finish_reason=%s%s)", finish,
                         f", said: {snippet!r}" if snippet else ", empty content")
                return False
            log.info("supervisor update_plan recovered from text content")

        sup.last_assessment = (args.get("assessment") or "").strip()
        verdict = (args.get("current_item_verdict") or "no_current_item").strip().lower()
        verdict_reason = (args.get("verdict_reason") or "").strip()

        # Verdict on the current item — with a label-anchored override: a 'done'
        # tick on an item with a machine-checkable expected outcome must pass
        # geometry, no matter what the supervisor believes.
        cur = sup.plan.current()
        if cur is not None and verdict in ("done", "failed", "not_done"):
            if verdict == "done" and cur.expected_value and cur.expected_label:
                v, detail = check_value_near_label(
                    cur.expected_value, cur.expected_label, elements
                )
                if v == "fail":
                    log.info("[INTERVENE] done-verdict overridden by label-anchored check: %s", detail)
                    verdict = "not_done"
                    verdict_reason = f"machine check failed: {detail}"
            if verdict == "done":
                sup.plan.mark(cur, "done", verdict_reason)
                log.info("[TODO] ✓ %s", cur.text)
            elif verdict == "failed":
                sup.plan.mark(cur, "failed", verdict_reason or "abandoned by supervisor")
                log.info("[TODO] ✗ %s", cur.text)
            else:
                cur.attempts += 1
                if verdict_reason:
                    cur.notes = verdict_reason

        # New items (corrective ones are scheduled before all normal work).
        # DEDUP: some models re-emit their whole remaining frontier every round;
        # an item duplicating an OPEN one is dropped, never queued twice.
        corrective_added = False
        for entry in (args.get("add_items") or []):
            if not isinstance(entry, dict):
                continue
            text = (entry.get("text") or "").strip()
            if not text:
                continue
            dup = sup.plan.find_open_duplicate(
                text, entry.get("expected_value") or "", entry.get("expected_label") or ""
            )
            if dup is not None:
                log.info("[TODO] ≈ dropped duplicate of open item [%d]: %s", dup.id, text)
                continue
            item = sup.plan.add(
                text,
                context=entry.get("context") or "",
                expected_value=entry.get("expected_value") or "",
                expected_label=entry.get("expected_label") or "",
                corrective=bool(entry.get("corrective")),
            )
            corrective_added = corrective_added or item.corrective
            log.info("[TODO] + %s%s", "⚠ " if item.corrective else "", item.text)

        # Close items the supervisor declares obsolete (duplicate, superseded,
        # or already satisfied by completed work). This is the ONLY way to clear
        # leftovers other than working them — without it, one stale item forces
        # a full (and possibly DANGEROUS) redundant work cycle, because the
        # completion gate rightly refuses to finish while items are open.
        for oid in (args.get("obsolete_item_ids") or []):
            try:
                oid = int(oid)
            except (TypeError, ValueError):
                continue
            item = sup.plan.get_by_id(oid)
            if item is not None and item.status in ("pending", "in_progress"):
                sup.plan.mark(item, "skipped", "obsolete — closed by supervisor")
                log.info("[TODO] − %s (obsolete)", item.text)

        # Corrective work PREEMPTS: suspend the active normal item so
        # activate_next() picks the corrective item first; the suspended item
        # is re-activated (pending) once corrections are done.
        if corrective_added:
            active = sup.plan.current()
            if active is not None and not active.corrective:
                active.status = "pending"
                log.info("[TODO] ⏸ %s (preempted by corrective item)", active.text)

        sup.guidance = (args.get("guidance") or "").strip()
        control = (args.get("control") or "continue").strip().lower()
        if control not in ("continue", "ask_user", "stop", "task_complete"):
            control = "continue"
        sup.control = control
        sup.control_detail = (args.get("control_detail") or "").strip()
        sup.final_evidence = [
            e for e in (args.get("final_evidence") or []) if isinstance(e, dict)
        ]

        # UI status pill + assessment line.
        if control == "task_complete":
            ui_status = "goal_reached"
        elif verdict == "failed" or "ALARM" in boundary:
            ui_status = "off_track"
        elif "STUCK" in boundary or "BUDGET" in boundary:
            ui_status = "stuck"
        else:
            ui_status = "on_track"
        log.info("[NAV] %s | %s", ui_status, sup.last_assessment or verdict_reason or boundary)
        log.info("[TODOS] %s", sup.plan.to_marker())
        corrective = sup.plan.pending_corrective()
        log.info("[ISSUE] %s", corrective.text if corrective else "")  # empty clears the UI
        return True

    def _boundary_round(
        self,
        sup: "SupervisorState",
        boundary: str,
        change: ChangeSet,
        trail: ProgressTrail,
        elements: List[Element],
        elements_text: str,
        tokens: Dict[str, int],
        actions_log: List[Dict[str, Any]],
        iteration_no: int,
    ) -> str:
        """Run a supervisor round and act on its control decision.
        Returns 'continue' | 'ask_user' | 'stop' | 'complete'."""
        # Surface the round in the activity timeline BEFORE the (potentially
        # slow) LLM call, so a long supervisor think never looks like a hang.
        log.info("[SUPERVISE] %s", " ".join(boundary.split())[:220])
        if not self._supervise(sup, boundary, change, trail, elements, elements_text, tokens):
            log.info("[INTERVENE] supervisor round failed — continuing with the current plan")
            return "continue"  # keep working the current plan

        if sup.control == "ask_user":
            question = sup.control_detail or "The supervisor needs more information to proceed."
            sup.control_detail = question
            log.info("[INTERVENE] ask_user: %s", question)
            log.info("[QUESTION] %s", question)
            actions_log.append({
                "iteration": iteration_no,
                "action": "supervisor_ask_user",
                "question": question,
                "reason": sup.last_assessment,
            })
            return "ask_user"

        if sup.control == "stop":
            reason = sup.control_detail or sup.last_assessment or "supervisor stopped the run"
            log.error("Supervisor stop: %s", reason)
            log.info("[INTERVENE] stop: %s", reason)
            actions_log.append({
                "iteration": iteration_no,
                "error": f"aborted: supervisor stop — {reason}",
            })
            return "stop"

        if sup.control == "task_complete":
            if sup.plan.has_open_items():
                open_ids = [it.id for it in sup.plan.items()
                            if it.status in ("pending", "in_progress")]
                sup.fail_note = (
                    f"Your previous task_complete was REJECTED: items {open_ids} are "
                    "still open. Work them — or, if they are duplicates/superseded/"
                    "already satisfied by completed work, close them by listing their "
                    "ids in obsolete_item_ids TOGETHER WITH task_complete."
                )
                log.info("[INTERVENE] task_complete rejected — open items remain: %s", open_ids)
                return "continue"
            ok, detail = self._check_final_evidence(sup.final_evidence, elements)
            if not ok:
                sup.fail_note = (
                    f"Your previous task_complete was REJECTED — {detail}. Either the "
                    "work is not actually done (add items to finish it) or you must "
                    "cite different evidence that is really on screen."
                )
                log.info("[INTERVENE] task_complete rejected — %s", detail)
                return "continue"
            claim = sup.control_detail or sup.last_assessment or "Task completed."
            actions_log.append({
                "iteration": iteration_no,
                "action": "task_complete",
                "claim": claim,
                "evidence": [(e.get("value") or "") for e in sup.final_evidence],
                "verified": True,
                "reason": f"supervisor completion — {detail}",
            })
            log.info("[TASK_RESULT] %s", claim)
            return "complete"

        return "continue"

    @staticmethod
    def _check_final_evidence(
        evidence: List[Dict[str, str]], elements: List[Element]
    ) -> Tuple[bool, str]:
        """Machine-check the supervisor's final evidence against the current OCR.

        PRESENCE is the hard requirement — a value not on screen means the
        completion is hallucinated and is rejected. Label adjacency is checked
        when a label is given, but a location mismatch only downgrades to a
        note: the spatial guarantee against wrong-field edits was already
        enforced at item-tick time (and by the wrong-field guard); values also
        legitimately appear away from 'their' label (page titles, merged OCR
        boxes, list rows), so failing completion on location here produces
        false rejections, not safety."""
        entries = [e for e in evidence if (e.get("value") or "").strip()]
        if not entries:
            return False, "no final_evidence values provided"
        details: List[str] = []
        for e in entries:
            value = e["value"].strip()
            label = (e.get("label") or "").strip()
            present, _d = check_evidence_in_ocr([value], elements)
            if not present:
                return False, f"value '{value}' not found in current OCR"
            if label:
                verdict, detail = check_value_near_label(value, label, elements)
                if verdict == "pass":
                    details.append(detail)
                else:
                    details.append(f"'{value}' present (location vs '{label}' unverified: {detail})")
            else:
                details.append(f"'{value}' present")
        return True, "; ".join(details)

    def _verify_subtask_claim(
        self,
        item: Optional[TodoItem],
        evidence: List[str],
        elements: List[Element],
    ) -> Tuple[bool, str]:
        """Machine-verify an actioner subtask_done claim against the FRESH OCR:
        presence of its cited evidence, plus the item's label-anchored expected
        outcome when defined. Pure checks — no LLM call."""
        evidence = [e for e in evidence if (e or "").strip()]
        if not evidence:
            return False, (
                "Claim REJECTED: no evidence cited. Call subtask_done again with "
                "1-4 verbatim OCR snippets that prove this task's outcome."
            )
        ok, details = check_evidence_in_ocr(evidence, elements)
        log_evidence_summary(evidence, details)
        if not ok:
            return False, format_evidence_rejection(details)
        if item is not None and item.expected_value and item.expected_label:
            verdict, detail = check_value_near_label(
                item.expected_value, item.expected_label, elements
            )
            log.info("label-anchored check: %s — %s", verdict, detail)
            if verdict == "fail":
                return False, (
                    f"Evidence check FAILED (wrong location): {detail}. The expected "
                    f"outcome is '{item.expected_value}' next to '{item.expected_label}'. "
                    "Fix this before declaring the task done."
                )
            if verdict == "no_label":
                return True, f"evidence found on screen ({detail})"
            return True, detail
        return True, "evidence found on screen"

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
            # 1-3 sentences of observation — 300 tokens is plenty; don't inherit
            # the supervisor's larger planning budget.
            resp = self._call_llm(
                messages, tools=NARRATOR_TOOLS,
                max_tokens=min(300, self.cfg.navigator_max_tokens),
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
        self._last_elements = None
        self._last_rect = None
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

        # First frame (no change-set). Decaret is ON: on a continued conversation
        # a field may still be focused from the previous request, and on a fresh
        # one the user just typed into the chat input — either way a blinking
        # caret can corrupt the very OCR the supervisor plans from (a caret
        # after 'Doe-Doe' once read as 'Doe-Doel' and spawned a phantom
        # corrective item).
        screenshot_b64, elements, change = self._parse_screen(decaret=True)
        elements_text = self._format_elements(elements)
        self._last_ocr_signature = ocr_signature(elements)

        actions_log: List[Dict[str, Any]] = []
        tokens = {"input": 0, "output": 0, "total": 0, "calls": 0}
        trail = ProgressTrail()
        trail.load(session.get("trail", []))  # continuity: prior actions/thoughts/outcomes
        nudge_count = 0
        pending_question: Optional[str] = None  # set if the supervisor asks the user
        text_only_streak = 0                    # consecutive turns with no tool call (honest-abort guard)

        if not self.cfg.use_navigator:
            log.warning("NAVIGATOR=0 requested, but the supervisor is integral to "
                        "this architecture — it stays enabled.")

        # Supervisor state: owns the plan (todo list). The plan is persisted per
        # conversation, so a follow-up request continues an existing plan.
        sup = SupervisorState(ultimate_goal=instruction, prior_tasks=prior_tasks)
        sup.plan.load(session.get("todos", []))

        run_over = False       # a terminal control decision was taken pre-loop
        task_done = False
        empty_plan_rounds = 0

        # --- Bootstrap the plan -------------------------------------------- #
        if not self.executor.focused_once:
            # Fresh conversation: the first frame is pre-focus (often the agent's
            # own UI), so a supervisor plan drawn from it would be bogus. Seed a
            # deterministic focus item — it auto-completes the moment a window
            # focus succeeds, and THAT boundary hands the supervisor a real
            # screen to plan the actual work from.
            if not any(it.kind == "focus" and it.status == "pending"
                       for it in sup.plan.items()):
                sup.plan.add(
                    "Bring the correct application for the user's request to the "
                    "foreground: use list_windows / focus_window, or "
                    "open_application if it is not running. Do not click or type "
                    "before that.",
                    context=(f"The user's request is: '{instruction}'. Derive the right "
                             "application from it. If genuinely ambiguous, call "
                             "subtask_blocked."),
                    kind="focus",
                    corrective=True,
                )
            log.info("[TODOS] %s", sup.plan.to_marker())
        else:
            # Grounded in the real app (continued conversation, or the startup
            # re-focus succeeded): let the supervisor plan from the actual screen.
            outcome = self._boundary_round(
                sup, "NEW USER REQUEST — assess the current screen and plan the work. "
                     "No pending item covers this request yet: you MUST add_items for it "
                     "IN THIS RESPONSE (or complete/stop). An update without add_items "
                     "is a wasted round. If you spot a problem on screen (wrong value, "
                     "open dialog), the fix must itself be an added item.",
                change, trail, elements, elements_text, tokens, actions_log, 0,
            )
            if outcome == "ask_user":
                pending_question = sup.control_detail
                run_over = True
            elif outcome == "stop":
                run_over = True
            elif outcome == "complete":
                task_done = True
                run_over = True

        # Activate the first item (re-asking the supervisor if the plan is empty).
        current_item: Optional[TodoItem] = None
        while not run_over:
            current_item = sup.plan.activate_next()
            if current_item is not None:
                break
            empty_plan_rounds += 1
            if empty_plan_rounds > 3:
                log.error("Supervisor produced no actionable items — aborting.")
                actions_log.append({"iteration": 0,
                                    "error": "aborted: supervisor produced no plan"})
                run_over = True
                break
            if empty_plan_rounds > 1:
                # Feedback beats standing instructions: tell the model its last
                # response was discarded, not just what the rules are.
                sup.fail_note = (
                    "REJECTED: your previous update added no usable items while "
                    "control was 'continue'. The actioner is IDLE and nothing will "
                    "happen until you act. Respond NOW with add_items (at least one "
                    "concrete item), or set control to task_complete/stop."
                )
            outcome = self._boundary_round(
                sup, "NO PENDING ITEMS — the actioner has nothing to do. You MUST respond "
                        "with add_items (at least one concrete item), or set control to "
                        "task_complete/stop. Assessing without adding items is a wasted "
                        "round and will simply be retried.",
                change, trail, elements, elements_text, tokens, actions_log, 0,
            )
            if outcome == "ask_user":
                pending_question = sup.control_detail
                run_over = True
            elif outcome == "stop":
                run_over = True
            elif outcome == "complete":
                task_done = True
                run_over = True

        history = ConversationHistory(
            SYSTEM_PROMPT,
            keep_recent=self.cfg.keep_recent_exchanges,
            keep_tool_turns=self.cfg.keep_tool_turns,
        )
        if current_item is not None:
            log.info("[NAV] → objective: %s", current_item.text)
            log.info("[TODOS] %s", sup.plan.to_marker())
            task_block = render_current_task(current_item, sup.guidance)
            sup.guidance = ""
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
            history.add_initial_user(task_block, initial_text, screenshot_b64)

        subtask_iters = 0   # actioner iterations spent on the current item
        for iteration in range(self.cfg.max_iterations):
            if run_over:
                break
            log.info("--- iteration %d ---", iteration + 1)
            log.info("[ITER] %d", iter_base + iteration + 1)  # timeline group boundary

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
                # NO free-text completion: declaring "done"/"complete" in prose used
                # to break here, bypassing the evidence check — that caused false
                # completions (and misfires when the server's tool-call parser leaks
                # a call into content). Completion MUST go through subtask_done, which
                # is evidence-checked. If the model never issues a tool call, abort
                # honestly rather than claim success.
                text_only_streak += 1
                if text_only_streak >= 4:
                    log.error("Model returned no tool call %d turns in a row — aborting "
                              "(check the server's tool-call parser).", text_only_streak)
                    actions_log.append({
                        "iteration": iteration + 1,
                        "error": "aborted: model issued no tool calls (server tool-call parser?)",
                    })
                    break
                nudge = ("You did not issue a tool call — every step MUST be a tool call. "
                         "If you believe the CURRENT TASK is finished, call subtask_done and cite OUTCOME "
                         "evidence currently visible on screen; otherwise take the next concrete action.")
                nudge_count += 1
                if nudge_count >= 2:
                    log.info("Re-parsing screen after repeated text-only responses.")
                    screenshot_b64, elements, _ = self._parse_screen()
                    note = trail.render()
                    note = (note + "\n\n" if note else "") + nudge
                    history.add_observation(self._format_elements(elements), screenshot_b64, note=note)
                    nudge_count = 0
                else:
                    history.add_nudge(nudge)
                continue
            text_only_streak = 0  # got a tool call — reset the no-tool-call counter

            nudge_count = 0
            boundary: Optional[str] = None       # supervisor boundary reason (this iteration)
            pending_claim: Optional[Dict[str, Any]] = None  # subtask_done awaiting verification
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

                # Legacy aliases (older prompts/persisted habits) map onto the
                # new sub-task tools so the run degrades gracefully.
                if fn_name in ("finish_task", "task_complete"):
                    fn_name = "subtask_done"
                if fn_name == "ask_user":
                    fn_name = "subtask_blocked"
                    args.setdefault("reason", args.get("question", ""))

                if fn_name == "subtask_done":
                    claim = args.get("message", "")
                    evidence = args.get("evidence") or []
                    if not isinstance(evidence, list):
                        evidence = [str(evidence)]
                    pending_claim = {"claim": claim, "evidence": evidence, "thought": thought}
                    history.add_tool_result(
                        tc["id"], "claim received — verifying against a fresh screenshot."
                    )
                    iteration_actions.append((fn_name, args, "verifying claim"))
                    break  # nothing may run after a completion claim

                if fn_name == "subtask_blocked":
                    reason = (args.get("reason") or "").strip() or "(no reason given)"
                    history.add_tool_result(
                        tc["id"], "reported blocked — the supervisor will replan."
                    )
                    actions_log.append({
                        "iteration": iteration + 1,
                        "action": "subtask_blocked",
                        "reason": reason,
                        "thought": thought,
                    })
                    iteration_actions.append((fn_name, args, f"blocked: {reason}"))
                    item_txt = current_item.text if current_item else "(no active item)"
                    boundary = f"ACTIONER BLOCKED on item '{item_txt}': {reason}"
                    break

                action_type = "type" if fn_name == "type_text" else fn_name
                # Strip 'thought' before handing args to the executor.
                exec_args = {k: v for k, v in args.items() if k != "thought"}
                result = self.executor.execute(action_type, exec_args)
                log.info("   %s", result)
                log.info("[RESULT] %s", result)  # timeline: outcome of the action above
                actions_log.append({
                    "iteration": iteration + 1,
                    "action": fn_name,
                    "args": args,
                    "result": result,
                    "thought": thought,
                })
                history.add_tool_result(tc["id"], result)
                iteration_actions.append((fn_name, args, result))

            # Enforce the assistant.tool_calls <-> tool.tool_call_id invariant:
            # any tool_calls we broke out of (e.g. after subtask_done) or didn't
            # recognize must still have a placeholder tool response or the next
            # API call will 400.
            dangling = history.ensure_tool_results_for(tool_calls)
            if dangling:
                log.debug("Added %d placeholder tool results for unprocessed calls", dangling)

            # Observe new screen state. If an action this turn focused or edited
            # a field, a text caret is likely — remove it before OCR (decaret).
            # A pending completion claim also forces decaret: its evidence is
            # typically a value in a still-focused field.
            time.sleep(0.2)
            self.executor.tick_click_age()
            did_edit = any(a in _CARET_INDUCING_ACTIONS for a, _, _ in iteration_actions)
            # Snapshot of what the actioner was looking at when it acted/claimed —
            # transient confirmations (toasts) can fade during the claim
            # round-trip, so a pending claim may need to be checked against this.
            claim_time_elements = elements
            screenshot_b64, elements, change = self._parse_screen(
                decaret=did_edit or pending_claim is not None
            )
            log.info("[CHANGE] %s", change.summary())
            new_sig = ocr_signature(elements)
            screen_changed = new_sig != self._last_ocr_signature
            self._last_ocr_signature = new_sig
            log.info("[PROGRESS] %s", "changed" if screen_changed else "nochange")  # timeline badge

            # Wait-aware stuck counter: don't count iterations where the model's
            # ONLY action was wait(). Any non-wait action ticks the counter.
            had_non_wait_action = any(a != "wait" for a, _, _ in iteration_actions)
            if screen_changed:
                self._no_change_streak = 0
            elif had_non_wait_action:
                self._no_change_streak += 1
            # (else: only waits this turn, don't tick)

            # --- Verify a pending completion claim (machine check, no LLM) --- #
            # Runs BEFORE trail recording so the trail shows the claim's OUTCOME,
            # not a dangling 'verifying claim' — the actioner reads the trail and
            # otherwise invents a failure narrative ("my claim was premature")
            # for claims that actually passed and advanced the plan.
            claim_note = ""
            if pending_claim is not None:
                verified, detail = self._verify_subtask_claim(
                    current_item, pending_claim["evidence"], elements
                )
                if not verified:
                    # Transient-evidence fallback: confirmation toasts routinely
                    # fade during the ~15-20s claim round-trip (LLM + OCR).
                    # Evidence that WAS visible on the screen the actioner
                    # claimed from is legitimate — nothing the agent did in
                    # between could have invalidated it (a claim ends the turn).
                    # Without this, a rejected save-toast makes the actioner
                    # click Save again (a redundant write — dangerous in a HIS).
                    log.info("re-checking claim against the claim-time screen (transient evidence?)")
                    stale_ok, stale_detail = self._verify_subtask_claim(
                        current_item, pending_claim["evidence"], claim_time_elements
                    )
                    if stale_ok:
                        verified = True
                        detail = (stale_detail + " — visible at claim time; since "
                                  "disappeared (transient confirmation), accepted")
                actions_log.append({
                    "iteration": iteration + 1,
                    "action": "subtask_done",
                    "claim": pending_claim["claim"],
                    "evidence": pending_claim["evidence"],
                    "verified": verified,
                    "reason": detail,
                    "thought": pending_claim["thought"],
                })
                if verified:
                    item_txt = current_item.text if current_item else "(no active item)"
                    boundary = (
                        f"ITEM COMPLETED (verified): the actioner reports "
                        f"'{pending_claim['claim']}' on item '{item_txt}'. "
                        f"Machine evidence check PASSED: {detail}. Give verdict "
                        "'done' unless you can name a concrete on-screen reason it "
                        "is not — leaving a machine-verified item open wastes a "
                        "full round re-proving it."
                    )
                    log.info("[RESULT] subtask_done verified — %s", detail)
                else:
                    claim_note = detail
                    log.info("[RESULT] subtask_done REJECTED — evidence check failed")
                claim_outcome = (
                    "VERIFIED ✓ — task completed; the supervisor advances the plan"
                    if verified else "REJECTED — evidence not confirmed on screen"
                )
                iteration_actions = [
                    (fn, a, claim_outcome if fn == "subtask_done" else r)
                    for fn, a, r in iteration_actions
                ]

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
            subtask_iters += 1

            # --- Focus auto-tick (deterministic — no supervisor needed) ------ #
            if (boundary is None and current_item is not None
                    and current_item.kind == "focus" and self.executor.focused_once):
                sup.plan.mark(current_item, "done", "window focused")
                log.info("[TODO] ✓ %s", current_item.text)
                log.info("[TODOS] %s", sup.plan.to_marker())
                boundary = ("FOCUS ACHIEVED: the target application is now in the "
                            "foreground. Plan the actual work item(s) from this screen.")

            # --- Wrong-field guard (deterministic geometry, every action) ---- #
            if boundary is None and current_item is not None:
                alarm = guard_wrong_field(change, current_item, elements)
                if alarm:
                    log.warning("wrong-field guard: %s", alarm)
                    log.info("[INTERVENE] wrong-field alarm")
                    log.info("[ISSUE] %s", alarm)
                    boundary = (f"WRONG-FIELD ALARM (detected by geometry, not by the "
                                f"actioner): {alarm} Insert a corrective item to revert "
                                "the wrong field before anything else.")

            # --- Budget / stuck boundaries ----------------------------------- #
            if boundary is None and subtask_iters >= self.cfg.subtask_max_iterations:
                item_txt = current_item.text if current_item else "(no active item)"
                boundary = (f"BUDGET EXHAUSTED: {subtask_iters} action rounds spent on "
                            f"item '{item_txt}' without completion. Revise it, split it, "
                            "take another route, or fail it.")
            if (boundary is None and self._no_change_streak >= 4
                    and self._no_change_streak % 2 == 0):
                # Every 2nd stuck round, not every round — the supervisor needs a
                # chance to see whether its last redirect worked before re-firing.
                boundary = (f"STUCK: the screen has not changed for "
                            f"{self._no_change_streak} action rounds. Redirect the "
                            "actioner (different route/keyboard) or change the plan.")
            if self._no_change_streak >= 7:
                log.error("Screen unchanged for %d consecutive iterations — aborting.",
                          self._no_change_streak)
                log.info("[INTERVENE] stop: screen unchanged for %d consecutive steps",
                         self._no_change_streak)
                actions_log.append({"iteration": iteration + 1, "error": "aborted: screen stuck"})
                break

            # Narrator (expensive VLM) only when a supervisor round is imminent
            # AND the cheap signals are insufficient (ambiguous transition, stuck
            # screen, alarm). Its narration is consumed by the supervisor — a
            # narrator call with no boundary pending is ~20s of latency for a
            # ride-along note the actioner rarely needs.
            narration = ""
            if (boundary is not None and self.cfg.use_narrator
                    and (change.transition in ("popup", "replaced")
                         or "STUCK" in boundary or "ALARM" in boundary
                         or "BUDGET" in boundary)):
                narration = self._narrate(screenshot_b64, change, tokens)
                if narration:
                    log.info("[NARRATOR] %s", narration)
                    sup.last_narration = narration  # consumed by the supervisor below

            # --- Supervisor boundary: verdict, replan, control --------------- #
            switched_item = False
            if boundary is not None:
                subtask_iters = 0
                outcome = self._boundary_round(
                    sup, boundary, change, trail, elements, elements_text,
                    tokens, actions_log, iteration + 1,
                )
                if outcome == "ask_user":
                    pending_question = sup.control_detail or "The supervisor needs input to proceed."
                    break
                if outcome == "stop":
                    break
                if outcome == "complete":
                    task_done = True
                    break
                # continue → make sure an item is active (replanning if the
                # supervisor ticked the last one without adding new work).
                nxt = sup.plan.current() or sup.plan.activate_next()
                while nxt is None and not run_over:
                    empty_plan_rounds += 1
                    if empty_plan_rounds > 3:
                        log.error("Supervisor produced no actionable items — aborting.")
                        actions_log.append({"iteration": iteration + 1,
                                            "error": "aborted: supervisor produced no plan"})
                        run_over = True
                        break
                    if empty_plan_rounds > 1:
                        sup.fail_note = (
                            "REJECTED: your previous update added no usable items while "
                            "control was 'continue'. The actioner is IDLE and nothing will "
                            "happen until you act. Respond NOW with add_items (at least one "
                            "concrete item), or set control to task_complete/stop."
                        )
                    outcome = self._boundary_round(
                        sup, "NO PENDING ITEMS — the actioner has nothing to do. You MUST respond "
                        "with add_items (at least one concrete item), or set control to "
                        "task_complete/stop. Assessing without adding items is a wasted "
                        "round and will simply be retried.",
                        change, trail, elements, elements_text, tokens, actions_log,
                        iteration + 1,
                    )
                    if outcome == "ask_user":
                        pending_question = sup.control_detail or "The supervisor needs input to proceed."
                        run_over = True
                        break
                    if outcome == "stop":
                        run_over = True
                        break
                    if outcome == "complete":
                        task_done = True
                        run_over = True
                        break
                    nxt = sup.plan.activate_next()
                if run_over:
                    break
                empty_plan_rounds = 0
                switched_item = nxt is not current_item
                if switched_item:
                    # A different item is a different approach — give it a fresh
                    # runway on the stuck counter (max_iterations still caps the run).
                    self._no_change_streak = 0
                current_item = nxt
                history.set_task(render_current_task(current_item, sup.guidance))
                sup.guidance = ""
                log.info("[NAV] → objective: %s", current_item.text)
                log.info("[TODOS] %s", sup.plan.to_marker())

            # Build the next observation note: claim/plan feedback first, then the
            # change-set (what the last action caused), then any narrator
            # observation, then the full trail and warnings.
            note_parts: List[str] = []
            if switched_item:
                note_parts.append(
                    "The supervisor reviewed progress and updated the plan — your "
                    "CURRENT TASK block above is up to date; work on THAT."
                )
            if claim_note:
                note_parts.append(claim_note)
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
                # Persist the supervisor's plan so a follow-up request (including
                # a reply to an ask_user question) continues the same todo list.
                "todos": sup.plan.export(),
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
    # The actioner (and the narrator, which reuses its model) consume
    # screenshots every step — a text-only profile cannot drive them.
    if (os.getenv(f"{profile}_VISION") or "1").strip().lower() in ("0", "false", "no", "off"):
        log.warning(
            "ACTIVE_PROFILE %s is marked text-only (%s_VISION=0), but the actioner "
            "and narrator need a VISION model — screenshots will be sent anyway and "
            "the run will likely fail. Pick a vision profile as actioner.",
            profile, profile,
        )
    use_navigator = os.getenv("NAVIGATOR", "1").strip().lower() not in ("0", "false", "no", "off")
    use_narrator = os.getenv("NARRATOR", "1").strip().lower() not in ("0", "false", "no", "off")
    continue_session = os.getenv("CONTINUE_SESSION", "1").strip().lower() not in ("0", "false", "no", "off")
    # Optional stronger supervisor model. NAVIGATOR_PROFILE is primary (a profile
    # name whose *_MODEL is used; 'SAME'/empty → reuse the actioner model). Falls
    # back to NAVIGATOR_MODEL (direct id) only when no profile is set. Shared proxy.
    nav_profile = (os.getenv("NAVIGATOR_PROFILE") or "").strip()
    navigator_model = None
    if nav_profile and nav_profile.upper() not in ("SAME", "ACTIONER", "NONE"):
        navigator_model = os.getenv(f"{nav_profile.upper()}_MODEL")
    elif not nav_profile:
        navigator_model = os.getenv("NAVIGATOR_MODEL")
    if navigator_model:
        log.info("Supervisor uses a separate model: %s", navigator_model)
    try:
        subtask_budget = max(2, int(os.getenv("SUBTASK_MAX_ITER", "8")))
    except ValueError:
        subtask_budget = 8
    # Per-request timeout (seconds). Raise via .env when the supervisor runs on
    # a large, possibly overloaded model whose responses can take minutes.
    try:
        request_timeout = max(10, int(os.getenv("REQUEST_TIMEOUT", "120")))
    except ValueError:
        request_timeout = 120
    return AgentConfig(
        endpoint=endpoint, api_key=api_key, model=model,
        use_navigator=use_navigator, use_narrator=use_narrator,
        continue_session=continue_session, navigator_model=navigator_model,
        subtask_max_iterations=subtask_budget,
        request_timeout=request_timeout,
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