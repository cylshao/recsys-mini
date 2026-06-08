"""Unified recaller interface.

Every recall route implements:
    fit(train_df) -> None
    recall(user_id, k) -> List[(item_id, score)]
    recall_batch(user_ids, k) -> dict[user_id, List[(item_id, score)]]
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

import pandas as pd


class BaseRecaller(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Build the model from training interactions."""

    @abstractmethod
    def recall(self, user_id: int, k: int) -> List[Tuple[int, float]]:
        """Return up to k (item_id, score) candidates for one user."""

    def recall_batch(
        self, user_ids: Iterable[int], k: int
    ) -> dict[int, List[Tuple[int, float]]]:
        """Default batch implementation: loop over recall()."""
        return {u: self.recall(u, k) for u in user_ids}
