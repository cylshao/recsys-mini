"""MovieLens-1M data loading.

File format (`::`-separated, ISO-8859-1):
  ratings.dat: UserID::MovieID::Rating::Timestamp
  movies.dat : MovieID::Title::Genres
  users.dat  : UserID::Gender::Age::Occupation::Zip-code
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_movielens_1m(raw_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    ratings = pd.read_csv(
        raw_dir / "ratings.dat",
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "item_id", "rating", "ts"],
        encoding="ISO-8859-1",
    )
    movies = pd.read_csv(
        raw_dir / "movies.dat",
        sep="::",
        engine="python",
        header=None,
        names=["item_id", "title", "genres"],
        encoding="ISO-8859-1",
    )
    users = pd.read_csv(
        raw_dir / "users.dat",
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="ISO-8859-1",
    )
    ratings["ts"] = pd.to_datetime(ratings["ts"], unit="s")
    return ratings, movies, users
