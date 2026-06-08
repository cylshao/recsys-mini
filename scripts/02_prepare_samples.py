"""Sample preparation: load -> k-core -> explicit->implicit -> split -> write parquet.

This is the second step of the recsys-mini pipeline. It turns raw MovieLens-1M
ratings into model-ready train / valid / test parquet files.

Pipeline:
  1. Load raw ratings / movies / users from ``data/raw/ml-1m/``
  2. K-core filtering: iteratively drop users/items with too few interactions
  3. Explicit -> implicit: keep only ``rating >= positive_threshold`` as positives
  4. Split (LOO or time-based, controlled by ``dataset.split_strategy``)
  5. Drop users/items from valid/test that are unseen in train
  6. Write 5 parquet files to ``data/processed/``

Outputs (under ``data/processed/``):
  - train.parquet  / valid.parquet / test.parquet  (user_id, item_id, rating, ts, label)
  - movies.parquet / users.parquet                  (side info, unchanged)

Run:
  python scripts/02_prepare_samples.py

See also:
  - docs/01-data-source-analysis.md §13 — why these specific thresholds and
    split strategy were chosen, and how to tune them for other datasets.
  - conf/config.yaml — all tunable parameters (k-core, positive threshold,
    split strategy, LOO / time-split parameters).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running directly: `python scripts/02_prepare_samples.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.loader import filter_kcore, load_movielens_1m, to_implicit
from src.data.splitter import filter_eval_users, leave_one_out_split, time_split
from src.utils.io import get_logger, timer


def _log_stats(
    log: logging.Logger,
    name: str,
    df: pd.DataFrame,
    with_time: bool = True,
    suffix: str = "",
) -> None:
    """Log row/user/item counts (plus density + time range when available)."""
    tail = f"  ({suffix})" if suffix else ""
    if len(df) == 0:
        log.info(f"  {name:<6} EMPTY{tail}")
        return

    rows = len(df)
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    line = f"  {name:<6} rows={rows:>9,}  users={n_users:>6,}  items={n_items:>5,}"
    if with_time and "ts" in df.columns:
        density = rows / (n_users * n_items) * 100  # percent
        ts_lo = df["ts"].min().strftime("%Y-%m-%d")
        ts_hi = df["ts"].max().strftime("%Y-%m-%d")
        line += f"  density={density:5.2f}%  ts=[{ts_lo} -> {ts_hi}]"
    log.info(line + tail)


def _drop_rate(before: int, after: int) -> str:
    """Format the drop rate between two row counts."""
    if before == 0:
        return "n/a"
    return f"{(1 - after / before) * 100:.2f}% dropped"


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    log = get_logger("prepare")

    raw_dir = cfg.path("raw_dir")
    out_dir = cfg.path("processed_dir")
    ds = cfg["dataset"]

    strategy = ds.get("split_strategy", "loo")
    if strategy not in ("loo", "time"):
        log.error(f"unknown split strategy: {strategy!r} (expected 'loo' or 'time')")
        sys.exit(1)

    if not (raw_dir / "ratings.dat").exists():
        log.error(
            f"raw data not found at {raw_dir}. "
            f"Run `bash scripts/01_download_data.sh` first."
        )
        sys.exit(1)

    # ===== Stage 1/5: load raw =====
    with timer("[1/5] load MovieLens-1M", log):
        ratings, movies, users = load_movielens_1m(raw_dir)
        n_raw = len(ratings)
        _log_stats(log, "raw", ratings, with_time=False)

    # ===== Stage 2/5: k-core filtering =====
    with timer(
        f"[2/5] k-core filter "
        f"(min_user={ds['min_user_inter']}, min_item={ds['min_item_inter']})",
        log,
    ):
        ratings = filter_kcore(ratings, ds["min_user_inter"], ds["min_item_inter"])
        n_kcore = len(ratings)
        _log_stats(log, "kept", ratings, with_time=False, suffix=_drop_rate(n_raw, n_kcore))

    # ===== Stage 3/5: explicit -> implicit =====
    thr = ds["positive_threshold"]
    with timer(f"[3/5] explicit -> implicit (rating >= {thr})", log):
        pos = to_implicit(ratings, thr)
        n_pos = len(pos)
        log.info(
            f"  pos    rows={n_pos:>9,}  "
            f"({_drop_rate(n_kcore, n_pos)} as weak / negative ratings)"
        )

    # ===== Stage 4/5: split =====
    with timer(f"[4/5] split (strategy={strategy})", log):
        if strategy == "loo":
            train, valid, test = leave_one_out_split(pos, min_inter=ds["loo_min_inter"])
        else:  # "time" (already validated above)
            train, valid, test = time_split(pos, ds["valid_days"], ds["test_days"])

        log.info("  before filtering eval users:")
        for split_name, split_df in (("train", train), ("valid", valid), ("test", test)):
            _log_stats(log, split_name, split_df)

        # Eval rows whose user/item was never in train are unpredictable
        # -> drop them so they don't artificially deflate Recall@K.
        valid = filter_eval_users(train, valid)
        test = filter_eval_users(train, test)

        log.info("  after dropping unseen-in-train users/items:")
        for split_name, split_df in (("train", train), ("valid", valid), ("test", test)):
            _log_stats(log, split_name, split_df)

    for split_name, split_df in (("train", train), ("valid", valid), ("test", test)):
        if len(split_df) == 0:
            log.error(
                f"{split_name} split is empty after filtering; "
                f"check k-core / threshold / split params in conf/config.yaml"
            )
            sys.exit(1)

    # ===== Stage 5/5: write parquet =====
    with timer("[5/5] write parquet", log):
        outputs = {
            "train.parquet":  train,
            "valid.parquet":  valid,
            "test.parquet":   test,
            "movies.parquet": movies,
            "users.parquet":  users,
        }
        for fname, df in outputs.items():
            path = out_dir / fname
            df.to_parquet(path, index=False)
            log.info(
                f"  OK {fname:<16} {_size_mb(path):>6.2f} MB  {len(df):>9,} rows"
            )

    # ===== Final summary =====
    log.info("")
    log.info("=" * 68)
    log.info("Summary")
    log.info("=" * 68)
    log.info(f"  raw                  {n_raw:>9,} rows")
    log.info(f"  after k-core         {n_kcore:>9,} rows  ({_drop_rate(n_raw, n_kcore)})")
    log.info(f"  after impl. (>={thr})    {n_pos:>9,} rows  ({_drop_rate(n_kcore, n_pos)})")
    log.info(f"  -> train              {len(train):>9,} rows")
    log.info(f"  -> valid              {len(valid):>9,} rows")
    log.info(f"  -> test               {len(test):>9,} rows")
    log.info(f"  output dir: {out_dir}")


if __name__ == "__main__":
    main()
