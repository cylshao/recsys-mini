"""Ranking metrics: AUC and GAUC (group AUC, grouped by user)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def gauc(df: pd.DataFrame, label_col: str = "label", score_col: str = "score",
         group_col: str = "user_id", weight_by: str = "impressions") -> float:
    """GAUC: compute AUC per user, then take a weighted average.

    weight_by:
        impressions: weight by each user's sample count (common)
        equal:       equal weight per user
    """
    group_aucs: list[tuple[float, float]] = []
    for u, g in df.groupby(group_col):
        if g[label_col].nunique() < 2:
            continue
        a = auc(g[label_col].values, g[score_col].values)
        if np.isnan(a):
            continue
        w = len(g) if weight_by == "impressions" else 1.0
        group_aucs.append((a, w))

    if not group_aucs:
        return float("nan")
    aucs = np.array([a for a, _ in group_aucs])
    ws = np.array([w for _, w in group_aucs])
    return float((aucs * ws).sum() / ws.sum())


def evaluate_ranker(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "AUC": auc(df["label"].values, df["score"].values),
        "GAUC": gauc(df),
        "n_samples": len(df),
        "n_users": df["user_id"].nunique(),
        "pos_ratio": float(df["label"].mean()),
    }
