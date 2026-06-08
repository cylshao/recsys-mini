"""MovieLens-1M data loading.

File format ("::"-separated, ISO-8859-1 encoded):
  ratings.dat: UserID::MovieID::Rating::Timestamp
  movies.dat : MovieID::Title::Genres
  users.dat  : UserID::Gender::Age::Occupation::Zip-code
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

_ENCODING = "ISO-8859-1"

# Narrow numeric dtypes to cut memory; text columns stay as the default object dtype.
_RATINGS_DTYPE: Dict[str, str] = {"user_id": "int32", "item_id": "int32", "rating": "int8"}
_MOVIES_DTYPE: Dict[str, str] = {"item_id": "int32"}
_USERS_DTYPE: Dict[str, str] = {"user_id": "int32", "age": "int8", "occupation": "int8"}


def _read_dat(path: Path, names: List[str], dtype: Dict[str, str]) -> pd.DataFrame:
    """Read one '::'-separated MovieLens .dat file.

    The multi-character separator forces pandas' python parser engine.
    """
    if not path.exists():
        raise FileNotFoundError(f"MovieLens-1M file not found: {path}")
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=names,
        dtype=dtype,
        encoding=_ENCODING,
    )


def load_movielens_1m(raw_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the ratings / movies / users tables of MovieLens-1M."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"MovieLens-1M raw dir not found: {raw_dir}")

    ratings = _read_dat(
        raw_dir / "ratings.dat", ["user_id", "item_id", "rating", "ts"], _RATINGS_DTYPE
    )
    movies = _read_dat(
        raw_dir / "movies.dat", ["item_id", "title", "genres"], _MOVIES_DTYPE
    )
    users = _read_dat(
        raw_dir / "users.dat", ["user_id", "gender", "age", "occupation", "zip"], _USERS_DTYPE
    )

    # Timestamps are unix epoch seconds.
    ratings["ts"] = pd.to_datetime(ratings["ts"], unit="s")
    return ratings, movies, users


def filter_kcore(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Iterative k-core filtering.

    Repeatedly drops users/items with too few interactions until the frame
    stabilizes (removing one side can push the other below threshold).
    """
    prev = -1
    while len(df) != prev:
        prev = len(df)
        user_cnt = df.groupby("user_id").size()
        item_cnt = df.groupby("item_id").size()
        keep_users = user_cnt[user_cnt >= min_user].index
        keep_items = item_cnt[item_cnt >= min_item].index
        df = df[df["user_id"].isin(keep_users) & df["item_id"].isin(keep_items)]
    return df.reset_index(drop=True)


def to_implicit(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Explicit ratings -> implicit positives: rating >= threshold is a positive."""
    pos = df[df["rating"] >= threshold].copy()
    pos["label"] = 1
    return pos.reset_index(drop=True)
