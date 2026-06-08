"""Train the ranking model (LightGBM).

Sample construction:
  positives = real positive feedback in train + valid
  negatives = popularity-based sampling (avoiding each user's history)
  features  = user/item profiles + train-only statistics (leak-free)
  split     = train -> fit, valid -> early stopping, test -> reserved for 05 (e2e eval)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import load_config
from src.data.sampler import build_user_history, sample_negatives
from src.eval.rank_metrics import evaluate_ranker
from src.features.builder import (
    assemble_samples,
    build_item_profile,
    build_item_stats,
    build_user_genre_pref,
    build_user_profile,
    build_user_stats,
)
from src.rank.lgb_ranker import LGBRanker
from src.utils.io import get_logger, log_metrics, save_pickle, timer


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    log = get_logger("rank")
    seed = cfg["seed"]

    proc = cfg.path("processed_dir")
    art = cfg.path("artifacts_dir")
    if not (proc / "train.parquet").exists():
        log.error("processed data not found; run scripts/02_prepare_samples.py first")
        sys.exit(1)

    train = pd.read_parquet(proc / "train.parquet")
    valid = pd.read_parquet(proc / "valid.parquet")
    movies = pd.read_parquet(proc / "movies.parquet")
    users = pd.read_parquet(proc / "users.parquet")

    # ----- features (train-only stats to avoid leakage) -----
    with timer("build features", log):
        user_profile = build_user_profile(users)
        item_profile, genre_cols = build_item_profile(movies)
        user_stats = build_user_stats(train)
        item_stats = build_item_stats(train)
        user_genre_pref = build_user_genre_pref(train, item_profile)

    def assemble(samples: pd.DataFrame):
        return assemble_samples(
            samples, user_profile, item_profile,
            user_stats, item_stats, user_genre_pref, genre_cols,
        )

    # ----- negative sampling -----
    all_items = train["item_id"].unique()
    item_pop = train.groupby("item_id").size().reindex(all_items).fillna(1).to_numpy()
    user_history_train = build_user_history(train)
    # User history at valid time = train interactions + valid itself
    # (the latter prevents sampling a valid positive as its own negative).
    user_history_full = build_user_history(pd.concat([train, valid]))
    neg_ratio = cfg["ranker"]["neg_ratio"]

    with timer("negative sampling (train)", log):
        train_samples = sample_negatives(
            train[["user_id", "item_id", "ts"]],
            user_history_train, all_items, item_pop,
            neg_ratio=neg_ratio, seed=seed,
        )
    with timer("negative sampling (valid)", log):
        valid_samples = sample_negatives(
            valid[["user_id", "item_id", "ts"]],
            user_history_full, all_items, item_pop,
            neg_ratio=neg_ratio, seed=seed + 1,
        )
    log.info(f"train samples: {len(train_samples):,}  pos ratio: {train_samples['label'].mean():.3f}")
    log.info(f"valid samples: {len(valid_samples):,}  pos ratio: {valid_samples['label'].mean():.3f}")

    # ----- assemble feature tables -----
    with timer("assemble train features", log):
        train_df, feature_cols = assemble(train_samples)
    with timer("assemble valid features", log):
        valid_df, _ = assemble(valid_samples)
    log.info(f"n_features: {len(feature_cols)}")

    # ----- train -----
    with timer("LightGBM training", log):
        lgb_cfg = cfg["ranker"]["lgb"]
        ranker = LGBRanker(
            params=lgb_cfg,
            num_boost_round=lgb_cfg["num_boost_round"],
            early_stopping_rounds=lgb_cfg["early_stopping_rounds"],
        )
        ranker.fit(train_df, valid_df, feature_cols)

    # ----- evaluate on valid -----
    valid_df["score"] = ranker.predict(valid_df)
    log.info("-" * 60)
    log.info("ranker metrics on valid:")
    log_metrics(log, evaluate_ranker(valid_df))

    log.info("-" * 60)
    log.info("top-15 important features (by gain):")
    for _, row in ranker.feature_importance().head(15).iterrows():
        log.info(f"  {row['feature']:<25} gain={row['gain']:>12.1f}  split={int(row['split'])}")

    # ----- persist -----
    save_pickle(ranker, art / "ranker_lgb.pkl")
    save_pickle({
        "user_profile": user_profile,
        "item_profile": item_profile,
        "user_stats": user_stats,
        "item_stats": item_stats,
        "user_genre_pref": user_genre_pref,
        "genre_cols": genre_cols,
        "feature_cols": feature_cols,
    }, art / "feature_store.pkl")
    log.info(f"ranker + feature store saved to: {art}")


if __name__ == "__main__":
    main()
