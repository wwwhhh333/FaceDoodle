"""Centralized logging configuration for FaceDoodle multi-process application.

Each process (Producer / Consumer / UI) calls ``setup_logging()`` once at
startup.  Configures the **root logger** so ``logging.getLogger(__name__)``
in every module inherits the handlers, formatter, and file output.
"""

import logging
import logging.handlers
import os
import sys

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")

_format = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def setup_logging(verbose=False):
    """Configure the root logger for the current process. Call once per process."""
    root = logging.getLogger()
    if root.handlers:
        return

    _project_root = os.path.dirname(LOG_DIR)
    if not os.path.isdir(os.path.join(_project_root, "app")):
        raise RuntimeError(f"LOG_DIR misconfigured: {LOG_DIR}")

    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Console handler: INFO+ (DEBUG in verbose)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(_format)
    root.addHandler(console)

    # File handler: WARNING+ to disk (rotating, 5 MB x 3 backups)
    error_file = os.path.join(LOG_DIR, "error.log")
    file_handler = logging.handlers.RotatingFileHandler(
        error_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(_format)
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for lib in ("matplotlib", "PIL", "urllib3", "asyncio"):
        logging.getLogger(lib).setLevel(logging.WARNING)
