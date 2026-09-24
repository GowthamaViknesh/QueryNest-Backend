"""Logging: console + a rotating file (logs/querynest.log, 5 files x 5 MB)."""

import logging
from logging.handlers import RotatingFileHandler

from querynest.config import settings

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(console_level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if getattr(root, "_querynest", False):
        return
    root.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(FORMAT))
    root.addHandler(console)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    file = RotatingFileHandler(settings.log_dir / "querynest.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    file.setFormatter(logging.Formatter(FORMAT))
    root.addHandler(file)
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "anthropic", "psycopg.pool"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root._querynest = True
