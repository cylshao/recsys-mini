"""Feature construction.

WARNING: all statistical features must be computed on train data only,
otherwise we leak future information. For valid/test samples we reuse the
feature values computed on the train set.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------
# 1) Static profiles (from users.dat / movies.dat)
# -----------------------------------------------------------

def build_user_profile(users: pd.DataFrame) -> pd.DataFrame:
    """User static features: gender, age, occupation."""
    u = users.copy()
    u["gender"] = (u["gender"] == "M").astype(int)
    return u[["user_id", "gender", "age", "occupation"]]


def build_item_profile(movies: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Item static features: genres as multi-hot columns."""
    m = movies.copy()
    genres_split = m["genres"].str.split("|")
    all_genres = sorted({g for gs in genres_split for g in gs})
    for g in all_genres:
        m[f"g_{g}"] = genres_split.apply(lambda gs: int(g in gs))
    feat_cols = ["item_id"] + [f"g_{g}" for g in all_genres]
    return m[feat_cols], [f"g_{g}" for g in all_genres]


# -----------------------------------------------------------
# 2) Train-set statistics (train only, to avoid leakage)
# -----------------------------------------------------------

def build_user_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """User behavior stats: interaction count, average rating."""
    g = train_df.groupby("user_id")
    s = pd.DataFrame({
        "u_inter_cnt": g.size(),
        "u_avg_rating": g["rating"].mean(),
    }).reset_index()
    return s


def build_item_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """Item stats: interaction count (popularity), average rating."""
    g = train_df.groupby("item_id")
    s = pd.DataFrame({
        "i_inter_cnt": g.size(),
        "i_avg_rating": g["rating"].mean(),
    }).reset_index()
    s["i_log_pop"] = np.log1p(s["i_inter_cnt"])
    return s


def build_user_genre_pref(
    train_df: pd.DataFrame, item_genre: pd.DataFrame
) -> pd.DataFrame:
    """User preference per genre (share of history falling in each genre)."""
    df = train_df.merge(item_genre, on="item_id", how="left")
    genre_cols = [c for c in item_genre.columns if c.startswith("g_")]
    agg = df.groupby("user_id")[genre_cols].mean().reset_index()
    agg.columns = ["user_id"] + [f"u{c}" for c in genre_cols]
    return agg


# -----------------------------------------------------------
# 3) Assemble the final feature table
# -----------------------------------------------------------

def assemble_samples(
    samples: pd.DataFrame,
    user_profile: pd.DataFrame,
    item_profile: pd.DataFrame,
    user_stats: pd.DataFrame,
    item_stats: pd.DataFrame,
    user_genre_pref: pd.DataFrame,
    genre_cols: List[str],
    fillna: float = 0.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """Expand (user_id, item_id) pairs into a feature-rich table."""
    df = samples.merge(user_profile, on="user_id", how="left")
    df = df.merge(item_profile, on="item_id", how="left")
    df = df.merge(user_stats, on="user_id", how="left")
    df = df.merge(item_stats, on="item_id", how="left")
    df = df.merge(user_genre_pref, on="user_id", how="left")

    # Cross feature: sum of the user's genre preferences over this item's genres
    # (an approximate user-item match score).
    matches = []
    for g in genre_cols:
        u_col = f"u{g}"
        if u_col in df.columns and g in df.columns:
            df[f"x_{g}"] = df[u_col] * df[g]
            matches.append(f"x_{g}")
    df["x_match_sum"] = df[matches].sum(axis=1) if matches else 0.0

    df = df.fillna(fillna)

    feature_cols = (
        ["gender", "age", "occupation"]
        + genre_cols
        + ["u_inter_cnt", "u_avg_rating", "i_inter_cnt", "i_avg_rating", "i_log_pop"]
        + [f"u{c}" for c in genre_cols]
        + matches
        + ["x_match_sum"]
    )
    return df, feature_cols
