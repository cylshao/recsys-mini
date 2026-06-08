"""Small IO and logging utilities."""
from __future__ import annotations

import logging
import pickle
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def get_logger(name: str = "recsys") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def log_metrics(logger: logging.Logger, metrics: dict[str, Any], indent: str = "  ") -> None:
    """Log a metrics dict: floats with 4 decimals, everything else as-is."""
    for k, v in metrics.items():
        logger.info(f"{indent}{k}: {v:.4f}" if isinstance(v, float) else f"{indent}{k}: {v}")


@contextmanager
def timer(name: str, logger: logging.Logger | None = None):
    log = logger or get_logger()
    t0 = time.time()
    log.info(f"▶ {name} start")
    try:
        yield
    finally:
        log.info(f"✓ {name} done in {time.time() - t0:.2f}s")


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)
