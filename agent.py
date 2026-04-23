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
from dataclasses import dataclass, field
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
            "description": "Double-click an element or coordinate.",
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
                            "2-5 short strings, each quoting OUTCOME text literally "
                            "as it appears on screen. Not labels. Not field names. "
                            "Things like typed values, confirmation messages, new "
                            "list rows, changed statuses."
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
                "Press Backspace a specific number of times to delete characters in the "
                "focused field. Use this when Ctrl+A / Delete failed to clear a field, "
                "or when you need to erase a known number of characters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of Backspace keypresses to send (1-200).",
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
                            "2-5 short strings quoting OUTCOME text: typed values "
                            "now visible, confirmation messages, new rows, changed "
                            "statuses. Not labels or field names."
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
        text = self.text.replace("\n", " ").strip()
        if len(text) > 60:
            text = text[:57] + "..."
        return f"[{self.stable_id}] '{text}' @({self.center_x},{self.center_y})"


class ElementRegistry:
    """Assigns short stable IDs to OCR elements across parses."""

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

    def reconcile(self, new_elements: List[Element], screen_w: int, screen_h: int) -> List[Element]:
        self._frame += 1
        unmatched_ids = set(self._by_id.keys())
        result: List[Element] = []
        for elem in new_elements:
            best_id, best_score = None, self._match_threshold
            for rid in unmatched_ids:
                old = self._by_id[rid]
                score = self._similarity(elem, old, screen_w, screen_h)
                if score > best_score:
                    best_score = score
                    best_id = rid
            if best_id is not None:
                elem.stable_id = best_id
                elem.last_seen_frame = self._frame
                self._by_id[best_id] = elem
                unmatched_ids.remove(best_id)
            else:
                elem.stable_id = self._mint_id()
                elem.last_seen_frame = self._frame
                self._by_id[elem.stable_id] = elem
            result.append(elem)
        stale = [rid for rid, el in self._by_id.items()
                 if self._frame - el.last_seen_frame > self._evict_after]
        for rid in stale:
            del self._by_id[rid]
        return result

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


def annotate_screenshot(
    screenshot: Image.Image,
    elements: List[Element],
    win_x: int,
    win_y: int,
    click_marker: Optional[Tuple[int, int]] = None,
    max_label_elements: int = 120,
) -> Image.Image:
    """
    Draw OCR bounding boxes + stable IDs onto the screenshot, and optionally
    a crosshair at click_marker (screen-absolute coords).

    Returns a new RGB image. Input is not modified.
    """
    out = screenshot.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    font = _get_font(11)

    # Sort elements by confidence; label only the top N to avoid visual clutter
    # on very dense screens. Low-confidence elements still get a faint box.
    elems_sorted = sorted(elements, key=lambda el: -el.confidence)

    for i, el in enumerate(elems_sorted):
        ix = el.x - win_x
        iy = el.y - win_y
        iw = el.width
        ih = el.height

        # Faint box for every element; brighter for high-confidence / labeled ones.
        if i < max_label_elements and el.confidence >= 40:
            draw.rectangle([ix, iy, ix + iw, iy + ih], outline=(255, 140, 0, 220), width=1)
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
                draw.text((ix + 2, label_y), label, fill=(255, 200, 80), font=font)
            else:
                draw.text((ix + 2, max(0, label_y)), label, fill=(255, 200, 80))
        else:
            draw.rectangle([ix, iy, ix + iw, iy + ih], outline=(255, 140, 0, 90), width=1)

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
}


class ActionExecutor:
    """Pure input dispatcher. No knowledge of screens or LLMs."""

    def __init__(self, width: int, height: int, registry: ElementRegistry):
        self.width = width
        self.height = height
        self.registry = registry
        self.last_click_point: Optional[Tuple[int, int]] = None
        self.last_click_age: int = 0  # iterations since last click, for marker fade
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
                    pyautogui.press(keys[0])
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
                count = max(1, min(int(args.get("count", 1)), 200))
                for _ in range(count):
                    pyautogui.press("backspace")
                    time.sleep(0.03)
                return f"pressed backspace {count} times"

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

    @staticmethod
    def _focus_window(title_substr: str) -> str:
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
        return f"focused '{buf.value}'"


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
    min_hit_ratio: float = 0.7,
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
        self._pinned_user_idx: Optional[int] = None

    def add_initial_user(self, task: str, element_text: str, screenshot_b64: str) -> None:
        self._messages.append({
            "role": "user",
            "content": f"TASK: {task}",
        })
        self._pinned_user_idx = len(self._messages) - 1

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
        text = (note + "\n\n" if note else "") + f"Updated screen:\n{element_text}"
        self._messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
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
            and i != self._pinned_user_idx
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

    @staticmethod
    def _summarize_old_observation(text: str) -> str:
        lines = [l for l in text.splitlines() if l.strip()]
        element_count = sum(1 for l in lines if l.startswith("[e"))
        header = lines[0][:120] if lines else "(prior observation)"
        return f"[earlier observation — {element_count} elements on screen] {header}"


# --- Agent ----------------------------------------------------------------- #

SYSTEM_PROMPT = """\
You are an expert AI agent controlling a computer via OCR + screenshot observation. \
You receive, every step: (1) a PROGRESS SO FAR block summarizing what you have \
done, what you thought, and what changed; (2) a text list of OCR elements \
in the form [id] 'text' @(cx,cy) with screen-absolute coordinates; (3) an \
annotated screenshot with orange OCR boxes showing the listed elements and stable \
IDs drawn on interactive text, plus a cyan crosshair marking your last click.

BEFORE EVERY ACTION — read the PROGRESS SO FAR block carefully. For each \
listed step, ask: Did that action succeed? Did the result build toward the \
goal? If a prior step produced useful information (e.g. a window list, a \
coordinate, a confirmed click), use it directly — do NOT repeat that step. \
Only pick an action that meaningfully advances on what is already known. \

REASONING PATTERN — for every action, include a 'thought' that covers:
  1. What changed since my last action, and does it match what I expected?
  2. What I am doing now.
  3. What I expect to happen next.
Keep it under 40 words. This is how you remember your plan and notice mistakes.

RULES:
- Prefer element_id over raw x/y. Coordinates are screen-absolute.
- Form fields are often NOT detected by OCR when empty — only the label beside \
  them is. In that case, click just to the right of the label, or use \
  click_and_type with x/y to focus-and-fill in one step. Verify by checking \
  whether your typed text appears in the next element list.
- If OCR shows text that should have been cleared is still there, your clear \
  attempt (Ctrl+A, Backspace, Delete) did NOT succeed. Do not just retype — \
  try a different clear approach or you will append instead of replace. \
  If Ctrl+A did not select the text, click the field to focus it, then call \
  delete_chars with the number of characters you need to erase (e.g. \
  delete_chars(10) to erase 10 chars).
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
    ocr_upscale: float = 3.0
    ocr_min_conf: int = 30
    save_debug_screenshots: bool = True


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

        self.registry = ElementRegistry()
        self.executor = ActionExecutor(self.width, self.height, self.registry)
        self._last_ocr_signature: Optional[frozenset] = None
        self._no_change_streak = 0
        self._last_action_was_wait = False

    # --- screen parsing --------------------------------------------------- #

    def _parse_screen(self) -> Tuple[str, List[Element]]:
        """
        Capture foreground window, run OCR, reconcile IDs, ANNOTATE the image
        (boxes + IDs + crosshair at last-click), return (base64 of annotated image, elements).
        """
        t0 = time.monotonic()
        screenshot, (win_x, win_y, win_w, win_h) = capture_foreground()
        log.info("screenshot: %.2fs (window %dx%d at %d,%d)",
                 time.monotonic() - t0, win_w, win_h, win_x, win_y)

        t1 = time.monotonic()
        log.info("Starting OCR (RapidOCR) ...")
        ocr_elements = extract_ocr_elements(screenshot)
        log.info("OCR: %.2fs → %d lines", time.monotonic() - t1, len(ocr_elements))

        # Offset OCR coords from window-relative to screen-absolute.
        for el in ocr_elements:
            el.x += win_x
            el.y += win_y
            el.center_x += win_x
            el.center_y += win_y

        reconciled = self.registry.reconcile(ocr_elements, self.width, self.height)

        # Annotate — same image for LLM and for the debug screenshot file.
        annotated = annotate_screenshot(
            screenshot, reconciled, win_x, win_y,
            click_marker=self.executor.last_click_point,
        )

        if self.cfg.save_debug_screenshots:
            self._save_debug(annotated)

        buf = BytesIO()
        annotated.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        return b64, reconciled

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
        if not elements:
            return "(no elements detected)"
        elements_sorted = sorted(elements, key=lambda el: (el.center_y // 16, el.center_x))
        lines = [el.as_prompt_line() for el in elements_sorted[:limit]]
        suffix = f"\n... (+{len(elements) - limit} more, not shown)" if len(elements) > limit else ""
        return "\n".join(lines) + suffix

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

    def _call_llm(self, messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
        self._dump_context(messages)
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "tools": tools if tools is not None else COMPUTER_TOOLS,
            "tool_choice": "auto",
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
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

    # --- main loop -------------------------------------------------------- #

    def run(self, instruction: str) -> Dict[str, Any]:
        start = time.time()
        start_dt = datetime.datetime.now()
        log.info("=" * 70)
        log.info("TASK: %s", instruction)
        log.info("=" * 70)

        # Reset per-task state.
        self._no_change_streak = 0
        self._last_ocr_signature = None
        self._last_action_was_wait = False
        self.registry = ElementRegistry()
        self.executor = ActionExecutor(self.width, self.height, self.registry)

        screenshot_b64, elements = self._parse_screen()
        elements_text = self._format_elements(elements)
        self._last_ocr_signature = ocr_signature(elements)

        history = ConversationHistory(
            SYSTEM_PROMPT,
            keep_recent=self.cfg.keep_recent_exchanges,
            keep_tool_turns=self.cfg.keep_tool_turns,
        )
        history.add_initial_user(instruction, elements_text, screenshot_b64)

        actions_log: List[Dict[str, Any]] = []
        tokens = {"input": 0, "output": 0, "total": 0, "calls": 0}
        trail = ProgressTrail()
        nudge_count = 0

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
                    screenshot_b64, elements = self._parse_screen()
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
            time.sleep(0.6)
            self.executor.tick_click_age()
            screenshot_b64, elements = self._parse_screen()
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
                    iteration=iteration + 1,
                    action=fn_name,
                    target_key=target_key_from_args(fn_name, args),
                    thought=thought,
                    result=trail_result,
                    screen_changed=screen_changed,
                    target_label=target_label,
                ))

            # Build the next observation note: trail + thrashing/stuck warnings.
            note_parts: List[str] = [trail.render()]
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
            history.add_observation(self._format_elements(elements), screenshot_b64, note=note)

        duration = time.time() - start
        log.info("=" * 70)
        log.info("DONE in %dm%.1fs | actions=%d | api_calls=%d | tokens in/out/total=%d/%d/%d",
                 int(duration // 60), duration % 60,
                 len(actions_log), tokens["calls"],
                 tokens["input"], tokens["output"], tokens["total"])
        log.info("=" * 70)

        return {
            "started_at": start_dt.isoformat(),
            "duration_seconds": duration,
            "actions": actions_log,
            "tokens": tokens,
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
        time.sleep(0.6)
        self.executor.tick_click_age()
        screenshot_b64, elements = self._parse_screen()

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
        note_parts = [trail.render()] if trail.entries() else []
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
    return AgentConfig(endpoint=endpoint, api_key=api_key, model=model)


def interactive(agent: ComputerAgent) -> None:
    print("\n" + "=" * 72)
    print("COMPUTER AGENT")
    print("=" * 72)
    print("Type an instruction, or 'quit' to exit.\n")
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
        try:
            agent.run(instr)
        except Exception:
            log.exception("Task failed")


def main() -> None:
    cfg = _load_config()
    log.info("Profile endpoint: %s | model: %s", cfg.endpoint, cfg.model)
    agent = ComputerAgent(cfg)
    interactive(agent)


if __name__ == "__main__":
    main()