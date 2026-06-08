"""Negative sampling for the ranking model.

Strategy:
  1. Items the user already interacted with cannot be negatives.
  2. Sample by item_popularity ** 0.75 (the word2vec trick) so the negative
     set is not dominated by cold items.
  3. Draw `neg_ratio` distinct negatives per positive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_COLUMNS = ["user_id", "item_id", "ts", "label"]


def build_user_history(df: pd.DataFrame) -> dict[int, set[int]]:
    """Map each user_id to the set of item_ids they interacted with."""
    return df.groupby("user_id")["item_id"].apply(set).to_dict()


def sample_negatives(
    pos_df: pd.DataFrame,
    user_history: dict[int, set[int]],
    all_items: np.ndarray,
    item_pop: np.ndarray,
    neg_ratio: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample `neg_ratio` negatives for each positive (user, item, ts).

    Args:
        pos_df:       positive rows, columns: user_id, item_id, ts
        user_history: user_id -> set of seen item_ids (train history + positives)
        all_items:    1-D array of all candidate item_ids
        item_pop:     popularity per item, aligned with `all_items` (sampling weight)
        neg_ratio:    number of negatives per positive
        seed:         RNG seed

    Returns:
        DataFrame with columns: user_id, item_id, ts, label
    """
    if len(all_items) != len(item_pop):
        raise ValueError(
            f"all_items ({len(all_items)}) and item_pop ({len(item_pop)}) length mismatch"
        )
    if len(pos_df) == 0:
        return pd.DataFrame(columns=_COLUMNS)

    rng = np.random.default_rng(seed)
    weights = np.power(item_pop, 0.75)
    weights = weights / weights.sum()

    # Oversample per draw to absorb collisions with seen/already-picked items.
    draw_size = max(neg_ratio * 3, neg_ratio + 10)
    max_attempts = 10

    rows: list[tuple] = []
    for u, i, ts in zip(pos_df["user_id"].values, pos_df["item_id"].values, pos_df["ts"].values):
        rows.append((u, i, ts, 1))
        seen = user_history.get(u, set())
        chosen: set[int] = set()
        attempts = 0
        while len(chosen) < neg_ratio and attempts < max_attempts:
            cand = rng.choice(all_items, size=draw_size, p=weights, replace=True)
            for c in cand:
                c = int(c)
                if c in seen or c in chosen:
                    continue
                chosen.add(c)
                rows.append((u, c, ts, 0))
                if len(chosen) >= neg_ratio:
                    break
            attempts += 1

    return pd.DataFrame(rows, columns=_COLUMNS)
