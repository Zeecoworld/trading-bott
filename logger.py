"""
bot/logger.py — Structured, colourised logging to stdout + file.
"""
import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str = "logs/bot.log") -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]

    try:
        import colorlog
        colour_fmt = (
            "%(log_color)s%(asctime)s  %(levelname)-8s%(reset)s  "
            "%(cyan)s%(name)-28s%(reset)s  %(message)s"
        )
        handlers[0] = colorlog.StreamHandler(sys.stdout)
        handlers[0].setFormatter(colorlog.ColoredFormatter(
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
        handlers[0].setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    handlers[1].setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)

    # Silence noisy libraries
    for lib in ("urllib3", "httpx", "httpcore", "alpaca", "werkzeug", "engineio", "socketio"):
        logging.getLogger(lib).setLevel(logging.WARNING)