"""
bot/logger.py — Stdout logging + in-memory ring buffer.

All log records are:
  1. Printed to stdout  (Render captures this in its native log dashboard)
  2. Stored in a thread-safe deque (max 2000 entries, oldest auto-dropped)
  3. Optionally broadcast live to WebSocket clients via a pluggable callback

Endpoints wired up in app.py:
  GET  /api/logs              → last N log lines as JSON
  GET  /api/logs?limit=500    → up to 500 lines
  GET  /api/logs?level=ERROR  → filter by minimum level
  WS   /ws                    → receives {"event":"log", "data":{...}} in real time
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, List, Optional

# ── In-memory log store ───────────────────────────────────────────────────────

_LOG_BUFFER: Deque[Dict] = deque(maxlen=2000)
_BUFFER_LOCK = threading.Lock()

# Pluggable async callback — set by app.py once the WS broadcast is ready
_ws_broadcast_callback: Optional[Callable] = None


def set_ws_broadcast(callback: Callable) -> None:
    """Called by app.py to wire up live WS streaming of log entries."""
    global _ws_broadcast_callback
    _ws_broadcast_callback = callback


def get_logs(limit: int = 200, min_level: str = "DEBUG") -> List[Dict]:
    """Return the most recent `limit` log entries at or above `min_level`."""
    numeric = getattr(logging, min_level.upper(), logging.DEBUG)
    with _BUFFER_LOCK:
        entries = list(_LOG_BUFFER)
    filtered = [e for e in entries if e["level_no"] >= numeric]
    return filtered[-limit:]


def clear_logs() -> None:
    with _BUFFER_LOCK:
        _LOG_BUFFER.clear()


# ── Custom handler ────────────────────────────────────────────────────────────

class _MemoryHandler(logging.Handler):
    """
    Appends every log record to the in-memory buffer and optionally
    fires the WebSocket broadcast callback (non-blocking fire-and-forget).
    """

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "level_no":  record.levelno,
            "logger":    record.name,
            "message":   self.format(record),
            "module":    record.module,
            "funcName":  record.funcName,
            "lineno":    record.lineno,
        }
        with _BUFFER_LOCK:
            _LOG_BUFFER.append(entry)

        # Fire-and-forget WebSocket push (if callback is registered)
        cb = _ws_broadcast_callback
        if cb is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.call_soon_threadsafe(
                        lambda e=entry: asyncio.ensure_future(
                            cb({"event": "log", "data": e})
                        )
                    )
            except Exception:
                pass  # Never let logging crash the bot


# ── Public setup function ─────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: str = "") -> None:
    """
    Configures root logger with:
      - StreamHandler  stdout   (Render captures this)
      - _MemoryHandler          in-memory buffer + live WS push

    log_file is accepted for call-site compatibility but intentionally ignored.
    No files are ever written to disk.
    """
    fmt     = "%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    plain_formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    try:
        import colorlog
        colour_fmt = (
            "%(log_color)s%(asctime)s  %(levelname)-8s%(reset)s  "
            "%(cyan)s%(name)-28s%(reset)s  %(message)s"
        )
        stdout_handler = colorlog.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(colorlog.ColoredFormatter(
            colour_fmt, datefmt=datefmt,
            log_colors={
                "DEBUG":    "white",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            },
        ))
    except ImportError:
        stdout_handler.setFormatter(plain_formatter)

    # Memory handler (plain text, no ANSI codes stored)
    memory_handler = _MemoryHandler()
    memory_handler.setFormatter(plain_formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(stdout_handler)
    root.addHandler(memory_handler)

    # Silence noisy third-party libraries
    for lib in ("urllib3", "httpx", "httpcore", "alpaca",
                "werkzeug", "engineio", "socketio", "aiohttp.access"):
        logging.getLogger(lib).setLevel(logging.WARNING)