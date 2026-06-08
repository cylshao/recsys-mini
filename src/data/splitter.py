"""Train / valid / test splitting.

Two strategies:
  1. time_split:      global split by time, strictly leak-free. Good for
                      temporally dense data (e.g. KuaiRand).
  2. leave_one_out:   per user, last positive -> test, second-to-last -> valid.
                      Good for temporally sparse / long-tail data (e.g. MovieLens);
                      the standard academic protocol.

Notes:
  - LOO does not guarantee global train_time < valid/test_time, so a weak
    cross-user leak is possible, but all feature statistics still come from the
    train slice only, which is acceptable in practice.
  - Any split should be paired with filter_eval_users to drop users/items that
    never appear in train.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd


def time_split(
    df: pd.DataFrame,
    valid_days: int,
    test_days: int,
    ts_col: str = "ts",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Global time split: last `test_days` -> test, the `valid_days` before -> valid."""
    df = df.sort_values(ts_col).reset_index(drop=True)
    t_max = df[ts_col].max()
    test_start = t_max - pd.Timedelta(days=test_days)
    valid_start = test_start - pd.Timedelta(days=valid_days)

    train = df[df[ts_col] < valid_start].copy()
    valid = df[(df[ts_col] >= valid_start) & (df[ts_col] < test_start)].copy()
    test = df[df[ts_col] >= test_start].copy()
    return train, valid, test


def leave_one_out_split(
    df: pd.DataFrame,
    min_inter: int = 3,
    ts_col: str = "ts",
    user_col: str = "user_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leave-one-out split.

    Per user (sorted by ts):
      - n_inter <  min_inter: all rows go to train, user is not evaluated
      - n_inter >= min_inter: last row -> test, second-to-last -> valid, rest -> train

    This guarantees:
      - every evaluated user keeps at least (min_inter - 2) rows in train (for recall)
      - #test users ~= #valid users ~= #users meeting min_inter

    `min_inter` must be >= 3 so an evaluated user still has train history.
    """
    if min_inter < 3:
        raise ValueError(f"min_inter must be >= 3 for LOO, got {min_inter}")

    df = df.sort_values([user_col, ts_col]).reset_index(drop=True)
    rank_desc = df.groupby(user_col).cumcount(ascending=False)
    n_inter = df.groupby(user_col)[user_col].transform("size")

    eligible = n_inter >= min_inter
    test_mask = eligible & (rank_desc == 0)
    valid_mask = eligible & (rank_desc == 1)
    train_mask = ~(test_mask | valid_mask)

    train = df[train_mask].reset_index(drop=True)
    valid = df[valid_mask].reset_index(drop=True)
    test = df[test_mask].reset_index(drop=True)
    return train, valid, test


def filter_eval_users(train: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only users and items seen in train; the model cannot score the rest."""
    train_users = set(train["user_id"].unique())
    train_items = set(train["item_id"].unique())
    return eval_df[
        eval_df["user_id"].isin(train_users) & eval_df["item_id"].isin(train_items)
    ].copy()
