#!/usr/bin/env python3
"""
OCR-only computer-use agent.

Captures only the foreground window and runs Tesseract OCR to detect text
elements. A persistent element registry assigns stable short IDs (e.g. 'e47')
that survive across parses via fuzzy feature-based matching.

Features:
  * Foreground-window-only screenshot — background apps are never analysed.
  * Multi-pass Tesseract OCR with adaptive + grayscale pre-processing.
  * Persistent element registry with stable IDs across frames.
  * Sliding-window conversation history: old screenshots are evicted, old
    OCR blocks are summarized. Massively reduces token usage over long tasks.
  * OCR-signature-based screen-change detection (robust against cursor blinks
    and 1-px redraws).
  * Programmatic completion verification before escalating to an LLM call.
  * System-role prompt for persistent rules.
  * DPI-aware capture on Windows.
  * Retry logic for transient API failures.

All heavy lifting (the LLM itself) happens on the server; everything local
stays light.
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
from collections import defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyautogui
import pytesseract
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw

# --- Environment setup ----------------------------------------------------- #

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# Tesseract path: env var wins, otherwise use a sensible default on Windows.
_TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if _TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
elif platform.system() == "Windows":
    _default = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe"
    if _default.exists():
        pytesseract.pytesseract.tesseract_cmd = str(_default)

# OCR language: env var
OCR_LANG = os.getenv("OCR_LANG", "eng")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")

# --- Tool schema (OpenAI-compatible function calling) ---------------------- #

COMPUTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": (
                "Click a UI element. Prefer element_id (from the element list) — it resolves "
                "to exact screen coordinates. Use x/y only when no suitable element_id exists "
                "(e.g. clicking empty canvas areas or icons not detected by OCR)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "element_id": {"type": "string", "description": "Short ID like 'e47' from the current element list."},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"},
                },
                "required": [],
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
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text into the currently focused field. Click the field first.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
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
                "properties": {"keys": {"type": "array", "items": {"type": "string"}}},
                "required": ["keys"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll at a position. scroll_y > 0 scrolls down; < 0 scrolls up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "scroll_y": {"type": "integer"},
                },
                "required": ["scroll_y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait a short period (e.g. for a dialog to appear). Default 1.5s.",
            "parameters": {
                "type": "object",
                "properties": {"seconds": {"type": "number"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Signal task completion. Only call when ALL steps are verified done.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
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
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "focus_window",
            "description": (
                "Bring a window to the foreground by title. Uses substring matching "
                "(case-insensitive). Call list_windows first to see available titles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Substring of the window title to match."},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Perform integer arithmetic to derive precise pixel coordinates. "
                "Use this when you need a position that is not directly available as an "
                "element_id — e.g. the midpoint between two OCR elements, or an offset "
                "from a known coordinate. Supports addition and subtraction only. "
                "Returns the integer result so you can feed it into x/y of click/scroll."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First operand (pixel value)."},
                    "op": {"type": "string", "enum": ["+", "-"], "description": "Operator: '+' or '-'."},
                    "b": {"type": "integer", "description": "Second operand (pixel value)."},
                },
                "required": ["a", "op", "b"],
            },
        },
    },
]

# --- DPI awareness --------------------------------------------------------- #

def _enable_dpi_awareness() -> None:
    """Make the process DPI-aware so GetSystemMetrics returns physical pixels."""
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        try:
            # Per-monitor DPI aware v2 (Windows 10 1703+)
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
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
    stable_id: str                      # e.g. "e47"
    text: str
    control_type: str                   # always "Text" for OCR elements
    source: str                         # always "ocr"
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    confidence: int = 0                 # Tesseract confidence 0-100
    automation_id: str = ""
    parent_text: str = ""
    last_seen_frame: int = 0

    def as_prompt_line(self) -> str:
        """One-line representation for the LLM prompt. Short on purpose."""
        text = self.text.replace("\n", " ").strip()
        if len(text) > 60:
            text = text[:57] + "..."
        return f"[{self.stable_id}] '{text}' @({self.center_x},{self.center_y})"


class ElementRegistry:
    """Assigns short stable IDs to elements and re-identifies them across parses."""

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
        # Text — strongest signal. Use ratio; empty strings get a neutral score.
        if new.text and old.text:
            text_sim = difflib.SequenceMatcher(None, new.text.lower(), old.text.lower()).ratio()
        else:
            text_sim = 0.4 if (not new.text and not old.text) else 0.0

        # Control type must match to get credit.
        type_sim = 1.0 if new.control_type == old.control_type else 0.3

        # Source match (all OCR, so always 1.0, kept for stability).
        source_sim = 1.0 if new.source == old.source else 0.5

        # Relative position similarity (robust to window moves of same app).
        dx = abs(new.center_x - old.center_x) / max(1, screen_w)
        dy = abs(new.center_y - old.center_y) / max(1, screen_h)
        pos_sim = max(0.0, 1.0 - 4.0 * (dx + dy))

        # Size similarity.
        dw = abs(new.width - old.width) / max(1, screen_w)
        dh = abs(new.height - old.height) / max(1, screen_h)
        size_sim = max(0.0, 1.0 - 4.0 * (dw + dh))

        return (
            0.45 * text_sim
            + 0.15 * type_sim
            + 0.10 * source_sim
            + 0.20 * pos_sim
            + 0.10 * size_sim
        )

    def reconcile(self, new_elements: List[Element], screen_w: int, screen_h: int) -> List[Element]:
        """Assign stable IDs to new elements by matching against the registry."""
        self._frame += 1
        unmatched_ids = set(self._by_id.keys())
        result: List[Element] = []

        # Greedy matching: for each new element, find its best candidate.
        # For large screens this is O(n * m). Screens typically have <300 elements, so fine.
        for elem in new_elements:
            best_id, best_score = None, self._match_threshold
            for rid in unmatched_ids:
                old = self._by_id[rid]
                score = self._similarity(elem, old, screen_w, screen_h)
                if score > best_score:
                    best_score = score
                    best_id = rid

            if best_id is not None:
                # Reuse: update positional/size info but keep the ID.
                elem.stable_id = best_id
                elem.last_seen_frame = self._frame
                self._by_id[best_id] = elem
                unmatched_ids.remove(best_id)
            else:
                elem.stable_id = self._mint_id()
                elem.last_seen_frame = self._frame
                self._by_id[elem.stable_id] = elem

            result.append(elem)

        # Evict long-unseen entries so the registry doesn't grow unbounded.
        stale = [rid for rid, el in self._by_id.items()
                 if self._frame - el.last_seen_frame > self._evict_after]
        for rid in stale:
            del self._by_id[rid]

        return result

    def get(self, stable_id: str) -> Optional[Element]:
        return self._by_id.get(stable_id)

    def resolve_fuzzy(self, text: str) -> Optional[Element]:
        """Last-resort lookup by text. Used when the model guesses a text name."""
        text = text.lower().strip()
        best, best_score = None, 0.6
        for el in self._by_id.values():
            if not el.text:
                continue
            score = difflib.SequenceMatcher(None, text, el.text.lower()).ratio()
            if score > best_score:
                best_score = score
                best = el
        return best


# --- OCR ------------------------------------------------------------------- #

def _ocr_passes(image: Image.Image, upscale: float) -> List[Tuple[str, np.ndarray, str]]:
    """
    Return (name, processed_image, tesseract_config) tuples for multi-pass OCR.

    Each pass catches a different class of text:
      * adaptive:  text on locally-varying backgrounds (PSM 11, fine block size)
      * grayscale: antialiased low-contrast text that binarization destroys (PSM 11)
      * block:     structured text in dialogs/forms/toolbars (PSM 6)

    Upscaling happens once and is shared across all passes.
    All passes run concurrently in the thread pool, so the extra pass costs no
    additional wall-clock time as long as a CPU thread is available.
    """
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    # Block size 17 (must be odd) isolates individual glyphs
    # 3x-upscaled UI screenshots where character spacing is tight.
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 10)

    psm11 = f"--oem 1 --psm 11 -l {OCR_LANG}"
    psm6  = f"--oem 1 --psm 6  -l {OCR_LANG}"
    return [
        ("adaptive",  adaptive, psm11),
        ("grayscale", gray,     psm11),
        ("block",     gray,     psm6),
    ]


def _run_ocr_pass(name: str, processed: np.ndarray, config: str,
                  upscale: float, min_conf: int) -> List[Dict[str, Any]]:
    """Run a single Tesseract pass and return grouped lines. Thread-safe."""
    try:
        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT, config=config)
    except Exception:
        log.exception("OCR pass %s failed", name)
        return []

    lines: Dict[tuple, list] = defaultdict(list)
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        try:
            conf = int(float(data["conf"][i]))
        except (ValueError, TypeError):
            continue
        if conf < min_conf or not text:
            continue
        lines[(data["block_num"][i], data["line_num"][i])].append({
            "text": text, "conf": conf,
            "x": int(data["left"][i] / upscale),
            "y": int(data["top"][i] / upscale),
            "w": int(data["width"][i] / upscale),
            "h": int(data["height"][i] / upscale),
        })

    result: List[Dict[str, Any]] = []
    for words in lines.values():
        if not words:
            continue
        words.sort(key=lambda w: w["x"])
        text = " ".join(w["text"] for w in words)
        x = min(w["x"] for w in words)
        y = min(w["y"] for w in words)
        x2 = max(w["x"] + w["w"] for w in words)
        y2 = max(w["y"] + w["h"] for w in words)
        avg_conf = sum(w["conf"] for w in words) // len(words)
        result.append({
            "text": text, "conf": avg_conf,
            "x": x, "y": y, "w": x2 - x, "h": y2 - y,
            "cx": (x + x2) // 2, "cy": (y + y2) // 2,
        })
    return result


def extract_ocr_elements(image: Image.Image, upscale: float = 3.0, min_conf: int = 20) -> List[Element]:
    """Run multiple Tesseract passes in parallel, merge results, return deduped line-level Elements."""
    passes = _ocr_passes(image, upscale)

    # Run OCR passes concurrently — Tesseract releases the GIL.
    all_lines: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(passes)) as pool:
        futures = {
            pool.submit(_run_ocr_pass, name, processed, config, upscale, min_conf): name
            for name, processed, config in passes
        }
        for future in concurrent.futures.as_completed(futures):
            all_lines.extend(future.result())

    # Dedupe across passes: two lines match if their centers are close and text is similar.
    # Keep the highest-confidence version on overlap.
    all_lines.sort(key=lambda l: -l["conf"])
    kept: List[Dict[str, Any]] = []
    for line in all_lines:
        is_dup = False
        for k in kept:
            if abs(line["cx"] - k["cx"]) > 10 or abs(line["cy"] - k["cy"]) > 10:
                continue
            if difflib.SequenceMatcher(None, line["text"].lower(), k["text"].lower()).ratio() >= 0.6:
                is_dup = True
                break
        if not is_dup:
            kept.append(line)

    return [
        Element(
            stable_id="",
            text=l["text"], control_type="Text", source="ocr",
            x=l["x"], y=l["y"], width=l["w"], height=l["h"],
            center_x=l["cx"], center_y=l["cy"],
            confidence=l["conf"],
        )
        for l in kept
    ]


def ocr_signature(elements: List[Element]) -> frozenset:
    """Stable signature of the visible text for change detection."""
    return frozenset(
        el.text.strip().lower()
        for el in elements
        if len(el.text.strip()) >= 2
    )


# --- Screen capture (foreground window only) -------------------------------- #

def capture_foreground() -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Screenshot only the foreground/active window.

    Returns (image, (win_x, win_y, win_w, win_h)) where win_x/win_y are the
    window's top-left position in screen coordinates. OCR coordinates derived
    from the returned image must be offset by (win_x, win_y) to become
    screen-absolute before clicking.

    Falls back to full-screen capture if the window rect cannot be determined.
    """
    if platform.system() == "Windows":
        try:
            import ctypes
            import ctypes.wintypes
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            rect = ctypes.wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            x, y = rect.left, rect.top
            w, h = rect.right - rect.left, rect.bottom - rect.top
            if w > 0 and h > 0:
                img = pyautogui.screenshot(region=(x, y, w, h))
                return img, (x, y, w, h)
        except Exception as exc:
            log.warning("Foreground window capture failed (%s); falling back to full screen.", exc)
        # Full-screen fallback
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
        # FAILSAFE off; we do our own bounds checking, and letting it raise
        # mid-task is worse than a clamped click.
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0

    def _resolve_point(self, args: Dict[str, Any]) -> Tuple[Optional[Tuple[int, int]], str]:
        """Resolve a click target. Returns ((x, y), "") on success or (None, error_msg)."""
        eid = args.get("element_id")
        if eid:
            el = self.registry.get(eid)
            if el is None:
                log.warning("Unknown element_id: %s — may be stale/evicted", eid)
                return None, (
                    f"error: element_id '{eid}' not found (it may have been evicted after "
                    "a screen change). Use an element_id from the LATEST element list, "
                    "or fall back to x/y coordinates."
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
                time.sleep(0.4)
                return f"success: clicked ({x},{y})"

            if action_type == "type" or action_type == "type_text":
                text = args.get("text", "")
                pyautogui.write(text, interval=0.03)
                time.sleep(0.2)
                return f"success: typed {len(text)} chars"

            if action_type == "keypress":
                keys = [_KEY_MAP.get(k.lower(), k.lower()) for k in args.get("keys", [])]
                if not keys:
                    return "error: no keys"
                if len(keys) == 1:
                    pyautogui.press(keys[0])
                else:
                    time.sleep(0.3)
                    pyautogui.hotkey(*keys)
                    time.sleep(0.3)
                return f"success: pressed {'+'.join(keys)}"

            if action_type == "scroll":
                x = args.get("x", self.width // 2)
                y = args.get("y", self.height // 2)
                scroll_y = int(args.get("scroll_y", 0))
                pyautogui.moveTo(*self._clamp(x, y))
                pyautogui.scroll(-scroll_y)
                return f"success: scrolled {scroll_y}"

            if action_type == "wait":
                secs = float(args.get("seconds", 1.5))
                time.sleep(min(secs, 10.0))
                return f"success: waited {secs}s"

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
                    return "error: calculate requires integer args 'a', 'b' and op '+' or '-'"
                result = int(a) + int(b) if op == "+" else int(a) - int(b)
                return f"result: {result}"

            return f"error: unknown action {action_type}"

        except pyautogui.FailSafeException:
            return "error: failsafe triggered (mouse at screen corner)"
        except Exception as exc:
            log.exception("Action error")
            return f"error: {exc}"

    # --- Window management helpers (Windows-only, stubs on other OS) ------- #

    @staticmethod
    def _list_windows() -> str:
        """Return a newline-separated list of visible window titles."""
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
        """Bring the first window whose title contains *title_substr* to the foreground."""
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
                        return False  # stop enumeration
            return True

        user32.EnumWindows(enum_cb, 0)

        if target_hwnd is None:
            return f"error: no window matching '{title_substr}' found"

        # Restore if minimized, then bring to foreground.
        SW_RESTORE = 9
        if user32.IsIconic(target_hwnd):
            user32.ShowWindow(target_hwnd, SW_RESTORE)
        user32.SetForegroundWindow(target_hwnd)
        time.sleep(0.5)

        # Read actual title for confirmation.
        length = user32.GetWindowTextLengthW(target_hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(target_hwnd, buf, length + 1)
        return f"success: focused '{buf.value}'"


# --- Conversation history with sliding window ----------------------------- #

class ConversationHistory:
    """
    Maintains the message list for the LLM with a sliding window:
      * The system prompt is pinned.
      * The initial user task message is pinned.
      * Recent N exchanges keep their full content (screenshots, OCR blocks).
      * Older exchanges get their images stripped and OCR replaced by short summaries.
    """

    def __init__(self, system_prompt: str, keep_recent: int = 3):
        self._messages: List[dict] = [{"role": "system", "content": system_prompt}]
        self._keep_recent = keep_recent
        self._pinned_user_idx: Optional[int] = None

    def add_initial_user(self, task: str, element_text: str, screenshot_b64: str) -> None:
        self._messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"TASK: {task}\n\nCurrent screen state:\n{element_text}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
            ],
        })
        self._pinned_user_idx = len(self._messages) - 1

    def add_assistant(self, message: dict) -> None:
        self._messages.append(message)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

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
        """Append a plain text user message (no screenshot). Used to nudge the model."""
        self._messages.append({"role": "user", "content": text})

    def messages_for_api(self) -> List[dict]:
        """Return a slimmed copy suitable for the API: old screenshots stripped."""
        msgs = [dict(m) for m in self._messages]

        # Identify which user observation messages are "recent" (keep full) vs old.
        # We count from the end: leave the last `keep_recent` vision user messages intact.
        vision_indices = [
            i for i, m in enumerate(msgs)
            if m.get("role") == "user" and isinstance(m.get("content"), list)
            and any(c.get("type") == "image_url" for c in m["content"])
            and i != self._pinned_user_idx  # the initial task message is pinned
        ]
        keep_set = set(vision_indices[-self._keep_recent:])

        for i in vision_indices:
            if i in keep_set:
                continue
            # Strip image; shorten OCR/element dump to a one-line summary.
            content = msgs[i]["content"]
            text_parts = [c["text"] for c in content if c.get("type") == "text"]
            combined = "\n".join(text_parts)
            summary = self._summarize_old_observation(combined)
            msgs[i] = {"role": "user", "content": summary}

        return msgs

    @staticmethod
    def _summarize_old_observation(text: str) -> str:
        # Pull the first line (task / status note) and count elements.
        lines = [l for l in text.splitlines() if l.strip()]
        element_count = sum(1 for l in lines if l.startswith("[e"))
        header = lines[0][:120] if lines else "(prior observation)"
        return f"[earlier observation — {element_count} elements on screen] {header}"


# --- Completion verification ---------------------------------------------- #

_SUCCESS_PATTERNS = [
    r"\bsuccess(fully)?\b", r"\bsaved?\b", r"\bgespeichert\b", r"\bcreated?\b",
    r"\berstellt\b", r"\bcomplete[d]?\b", r"\babgeschlossen\b", r"\bok\b",
]
_ERROR_PATTERNS = [
    r"\berror\b", r"\bfehler\b", r"\binvalid\b", r"\brequired\b",
    r"\bpflichtfeld\b", r"\bungültig\b", r"\bfailed\b", r"\bfehlgeschlagen\b",
]


def quick_completion_check(elements: List[Element], task: str) -> Tuple[bool, str]:
    """
    Cheap heuristic completion check. Returns (clear_pass, note).
      * clear_pass=True → very confident the task looks done; skip LLM verification.
      * clear_pass=False → either ambiguous or obviously not done; escalate.
    The returned note is always shown to the LLM verifier for context.
    """
    all_text = " ".join(el.text.lower() for el in elements)
    has_error = any(re.search(p, all_text) for p in _ERROR_PATTERNS)
    has_success = any(re.search(p, all_text) for p in _SUCCESS_PATTERNS)

    # Look for task-specific keywords (any word >=4 chars from the task that appears on screen).
    task_words = [w.lower() for w in re.findall(r"[A-Za-zÄÖÜäöüß]{4,}", task)]
    task_words_present = [w for w in task_words if w in all_text]

    if has_error:
        return False, f"⚠ Error indicators visible on screen: look for error/Fehler."
    if has_success and task_words_present:
        return True, f"✓ Success indicator present and task keywords found: {task_words_present[:5]}"
    return False, f"No clear success/error signal. Task keywords present on screen: {task_words_present[:5]}"


# --- Agent ----------------------------------------------------------------- #

SYSTEM_PROMPT = """\
You are an expert AI agent controlling a computer to complete tasks through visual \
observation and automated actions. You receive a screenshot AND a structured list of \
OCR-detected text elements on every step.

RULES (persistent, always apply):
- Prefer element_id (like 'e47') over raw x/y coordinates. IDs are stable across steps \
  while the UI is unchanged, and resolve to exact screen-absolute centers.
- Element list format: [id] 'text' @(cx,cy) — all coordinates are screen-absolute.
- For form fields: the label and the input field are separate visual elements. Click in \
  the input area (near but not on the label) to focus it, then type.
- After typing, verify the value appears in the next element list before proceeding.
- If the screen does not change after an action, do NOT repeat the same action. Reassess.
- Only call task_complete when ALL requested steps are verified done AND a success \
  indicator is visible AND no error is on screen.
- To switch between applications, call list_windows to see what is open, then \
  focus_window with a title substring to bring it to the foreground. The next screenshot \
  will automatically capture the newly focused window.
- Never fabricate element IDs. If you are unsure, use coordinates.
- Work in the language of the UI (German or English as appropriate).
"""


@dataclass
class AgentConfig:
    endpoint: str
    api_key: str
    model: str
    max_iterations: int = 40
    keep_recent_exchanges: int = 3
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

    # --- screen parsing --------------------------------------------------- #

    def _parse_screen(self) -> Tuple[str, List[Element]]:
        """Capture foreground window, run OCR, reconcile IDs. Returns (b64 png, elements)."""
        t0 = time.monotonic()
        screenshot, (win_x, win_y, win_w, win_h) = capture_foreground()
        log.info("screenshot: %.2fs (window %dx%d at %d,%d)",
                 time.monotonic() - t0, win_w, win_h, win_x, win_y)

        t1 = time.monotonic()
        log.info("Starting OCR (upscale=%.1fx, 2 passes parallel) ...", self.cfg.ocr_upscale)
        ocr_elements = extract_ocr_elements(screenshot, self.cfg.ocr_upscale, self.cfg.ocr_min_conf)
        log.info("OCR: %.2fs → %d lines", time.monotonic() - t1, len(ocr_elements))

        # Offset OCR coordinates from window-relative to screen-absolute so that
        # click/scroll actions (which use screen coordinates) work correctly.
        for el in ocr_elements:
            el.x += win_x
            el.y += win_y
            el.center_x += win_x
            el.center_y += win_y

        log.info("OCR: %d lines | parse total: %.2fs", len(ocr_elements), time.monotonic() - t0)

        reconciled = self.registry.reconcile(ocr_elements, self.width, self.height)

        if self.cfg.save_debug_screenshots:
            self._save_debug(screenshot, reconciled, win_x, win_y)

        # Base64-encode the screenshot.
        buf = BytesIO()
        screenshot.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        return b64, reconciled

    def _save_debug(
        self, screenshot: Image.Image, elements: List[Element], win_x: int = 0, win_y: int = 0
    ) -> None:
        try:
            annotated = screenshot.copy()
            draw = ImageDraw.Draw(annotated)
            for el in elements:
                # Convert screen-absolute coordinates back to image-relative for drawing.
                ix, iy = el.x - win_x, el.y - win_y
                draw.rectangle([ix, iy, ix + el.width, iy + el.height], outline=(255, 140, 0), width=1)
                draw.text((ix, max(0, iy - 10)), el.stable_id, fill=(255, 140, 0))
            out = Path(os.environ.get("TEMP", tempfile.gettempdir())) / "agent_screenshot_debug.png"
            annotated.save(out, format="PNG")
            log.info("[SCREENSHOT_READY]")
        except Exception as exc:
            log.debug("debug screenshot failed: %s", exc)

    @staticmethod
    def _format_elements(elements: List[Element], limit: int = 200) -> str:
        if not elements:
            return "(no elements detected)"
        # Sort top-to-bottom, left-to-right for readability.
        elements_sorted = sorted(elements, key=lambda el: (el.center_y // 16, el.center_x))
        lines = [el.as_prompt_line() for el in elements_sorted[:limit]]
        suffix = f"\n... (+{len(elements) - limit} more, not shown)" if len(elements) > limit else ""
        return "\n".join(lines) + suffix

    # --- API call --------------------------------------------------------- #

    def _call_llm(self, messages: List[dict]) -> dict:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "tools": COMPUTER_TOOLS,
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
                    self.cfg.endpoint,
                    headers=headers, json=payload,
                    verify=self.cfg.verify_tls,
                    timeout=self.cfg.request_timeout,
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

        # Reset per-task state so sequential tasks don't leak stale data.
        self._no_change_streak = 0
        self._last_ocr_signature = None
        self.registry = ElementRegistry()
        self.executor.registry = self.registry

        screenshot_b64, elements = self._parse_screen()
        elements_text = self._format_elements(elements)
        self._last_ocr_signature = ocr_signature(elements)

        history = ConversationHistory(SYSTEM_PROMPT, keep_recent=self.cfg.keep_recent_exchanges)
        history.add_initial_user(instruction, elements_text, screenshot_b64)

        actions_log: List[Dict[str, Any]] = []
        tokens = {"input": 0, "output": 0, "total": 0, "calls": 0}
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
            finish = choice.get("finish_reason", "")
            history.add_assistant(msg)

            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                # Model emitted text. Nudge it toward actions.
                text = msg.get("content", "") or ""
                log.info("Model text: %s", text[:200])
                if any(p in text.lower() for p in ("task is complete", "task complete", "done")):
                    log.info("✓ Model indicates completion in text.")
                    break
                if finish != "stop":
                    log.warning("Unexpected finish_reason: %s", finish)
                    break
                nudge_count += 1
                if nudge_count >= 2:
                    # After repeated nudges, re-parse the screen to give fresh context.
                    log.info("Re-parsing screen after %d consecutive text-only responses.", nudge_count)
                    screenshot_b64, elements = self._parse_screen()
                    history.add_observation(
                        self._format_elements(elements), screenshot_b64,
                        note="Screen re-captured after repeated text-only responses. Please issue a tool call.",
                    )
                    nudge_count = 0
                else:
                    log.info("No tool call returned; sending text nudge.")
                    history.add_nudge("Please issue a tool call to make progress. "
                                      "The screen has not changed.")
                continue

            nudge_count = 0  # reset nudge counter when we get tool calls
            task_done = False
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                log.info("→ %s(%s)", fn_name, args)

                if fn_name == "task_complete":
                    claim = args.get("message", "")
                    verified, reason, corrective = self._verify_completion(
                        instruction, claim, history, tokens)
                    history.add_tool_result(tc["id"], f"verification: {reason}")
                    actions_log.append({
                        "iteration": iteration + 1, "action": "task_complete",
                        "claim": claim, "verified": verified, "reason": reason,
                    })
                    actions_log.extend(corrective)
                    if verified:
                        if claim.strip():
                            log.info("[TASK_RESULT] %s", claim.strip())
                        task_done = True
                    break

                action_type = "type" if fn_name == "type_text" else fn_name
                result = self.executor.execute(action_type, args)
                log.info("   %s", result)
                history.add_tool_result(tc["id"], result)
                actions_log.append({
                    "iteration": iteration + 1, "action": fn_name,
                    "args": args, "result": result,
                })

            if task_done:
                break

            # Observe the new screen state.
            time.sleep(0.8)
            screenshot_b64, elements = self._parse_screen()
            new_sig = ocr_signature(elements)
            if new_sig == self._last_ocr_signature:
                self._no_change_streak += 1
            else:
                self._no_change_streak = 0
            self._last_ocr_signature = new_sig

            note = ""
            if self._no_change_streak >= 1:
                note = (
                    f"⚠ Screen text unchanged for {self._no_change_streak} action(s). "
                    "Your last action may have had no effect — reassess before repeating."
                )
            if self._no_change_streak >= 3:
                note += "\nConsider a completely different approach."
                log.warning("Screen stuck for %d iterations", self._no_change_streak)
            if self._no_change_streak >= 5:
                log.error("Screen unchanged for %d consecutive iterations — aborting.",
                          self._no_change_streak)
                actions_log.append({"iteration": iteration + 1,
                                    "error": "aborted: screen stuck"})
                break

            history.add_observation(self._format_elements(elements), screenshot_b64, note=note)

        # --- summary ---
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
        self, task: str, claim: str,
        history: ConversationHistory, tokens: Dict[str, int],
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        log.info("Verifying completion: %s", claim)
        time.sleep(0.8)
        screenshot_b64, elements = self._parse_screen()

        quick_ok, note = quick_completion_check(elements, task)
        if quick_ok:
            log.info("✓ Quick check passed: %s", note)
            return True, f"accepted by quick check ({note})", []

        # Escalate to the model — but make it clear we want a yes/no judgment.
        log.info("Escalating to LLM verification: %s", note)
        elements_text = self._format_elements(elements)
        history.add_observation(
            elements_text, screenshot_b64,
            note=(
                f"VERIFICATION REQUEST: You claimed the task is complete ('{claim}'). "
                f"Programmatic check says: {note}. "
                "Critically assess the screen. If truly done, call task_complete again with a "
                "precise confirmation. Otherwise, continue with more actions."
            ),
        )
        try:
            response = self._call_llm(history.messages_for_api())
        except Exception as exc:
            log.warning("Verification call failed: %s. NOT accepting claim.", exc)
            return False, f"api failure during verification — not accepting: {exc}", []

        tokens["calls"] += 1
        if "usage" in response:
            u = response["usage"]
            tokens["input"] += u.get("prompt_tokens", 0)
            tokens["output"] += u.get("completion_tokens", 0)
            tokens["total"] += u.get("total_tokens", 0)

        vmsg = response["choices"][0]["message"]
        history.add_assistant(vmsg)
        vtcs = vmsg.get("tool_calls") or []
        if vtcs and vtcs[0]["function"]["name"] == "task_complete":
            history.add_tool_result(vtcs[0]["id"], "verified complete")
            return True, "model confirmed after seeing current screen", []

        # Model wants to continue: execute its first action so we don't lose progress.
        corrective_actions: List[Dict[str, Any]] = []
        for tc in vtcs:
            fn_name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            action_type = "type" if fn_name == "type_text" else fn_name
            result = self.executor.execute(action_type, args)
            history.add_tool_result(tc["id"], result)
            corrective_actions.append({
                "action": fn_name, "args": args, "result": result,
                "source": "verification_corrective",
            })
        return False, "model chose to continue", corrective_actions


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