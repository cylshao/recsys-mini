"""Config loading and global path management."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "config.yaml"


@dataclass
class Config:
    raw: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, key: str) -> Path:
        rel = self.raw["paths"][key]
        p = (PROJECT_ROOT / rel).resolve()
        return p

    def ensure_dirs(self) -> None:
        for k in ("raw_dir", "processed_dir"):
            self.path(k).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with p.open("r", encoding="utf-8") as f:
        return Config(raw=yaml.safe_load(f))
