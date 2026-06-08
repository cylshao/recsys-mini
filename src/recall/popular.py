"""Popularity fallback recall: global top-N most interacted items."""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .base import BaseRecaller


class PopularRecaller(BaseRecaller):
    name = "popular"

    def __init__(self, topk: int = 200):
        # `topk` is the configured cap; we still keep the full ranked list so
        # recall() can always backfill k items after removing a user's history.
        self.topk = topk
        self._top_items: List[Tuple[int, float]] = []
        self._user_history: Dict[int, set[int]] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        pop = train_df.groupby("item_id").size().sort_values(ascending=False)
        if len(pop) == 0:
            self._top_items = []
            self._user_history = {}
            return
        # Normalize scores to [0, 1] for fair multi-route fusion.
        max_p = pop.iloc[0]
        self._top_items = [(int(i), float(c) / max_p) for i, c in pop.items()]
        self._user_history = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

    def recall(self, user_id: int, k: int) -> List[Tuple[int, float]]:
        seen = self._user_history.get(user_id, set())
        out: List[Tuple[int, float]] = []
        for item_id, score in self._top_items:
            if item_id in seen:
                continue
            out.append((item_id, score))
            if len(out) >= k:
                break
        return out
