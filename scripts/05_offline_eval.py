"""End-to-end offline evaluation: recall -> ranking -> Top-N.

This reflects the system's true effectiveness; recall HitRate or ranking AUC
alone are both misleading. Produces three comparable metric blocks:
  1. [recall]            stage metrics on the candidate pool
  2. [recall + ranking]  full-chain Top-N (the real number)
  3. [baseline]          recall straight-through Top-N (skip ranking)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from src.config import load_config
from src.eval.recall_metrics import evaluate_recall
from src.features.builder import assemble_samples
from src.recall.multi_recall import MultiRecaller
from src.utils.io import get_logger, load_pickle, log_metrics, timer

_REQUIRED_ARTIFACTS = [
    "recall_itemcf.pkl",
    "recall_popular.pkl",
    "ranker_lgb.pkl",
    "feature_store.pkl",
]


def main() -> None:
    cfg = load_config()
    log = get_logger("e2e")

    proc = cfg.path("processed_dir")
    art = cfg.path("artifacts_dir")

    if not (proc / "test.parquet").exists():
        log.error("processed data not found; run scripts/02_prepare_samples.py first")
        sys.exit(1)
    missing = [f for f in _REQUIRED_ARTIFACTS if not (art / f).exists()]
    if missing:
        log.error(f"missing artifacts {missing}; run 03_train_recall.py and 04_train_rank.py first")
        sys.exit(1)

    train = pd.read_parquet(proc / "train.parquet")
    test = pd.read_parquet(proc / "test.parquet")

    itemcf = load_pickle(art / "recall_itemcf.pkl")
    pop = load_pickle(art / "recall_popular.pkl")
    ranker = load_pickle(art / "ranker_lgb.pkl")
    fs = load_pickle(art / "feature_store.pkl")

    multi = MultiRecaller(
        recallers={"itemcf": itemcf, "popular": pop},
        weights=cfg["recall"]["weights"],
    )

    gt = test.groupby("user_id")["item_id"].apply(set).to_dict()
    eval_users = list(gt.keys())
    recall_k = cfg["eval"]["recall_for_rank"]
    final_n = cfg["eval"]["final_topn"]
    total_items = train["item_id"].nunique()
    log.info(f"test users: {len(eval_users):,}")

    def report(label: str, recs: dict, k_list: list[int]) -> None:
        log.info("-" * 60)
        log.info(label)
        log_metrics(log, evaluate_recall(recs, gt, k_list=k_list, total_items=total_items))

    # ---------- stage 1: recall only ----------
    with timer("recall stage", log):
        recall_only: dict[int, list[tuple[int, float]]] = {}
        for u in tqdm(eval_users, desc="recall"):
            items = multi.recall(u, k_each=recall_k, k_final=recall_k)
            recall_only[u] = [(i, s) for i, s, _ in items]
    report(f"[recall] stage metrics (top {recall_k} candidates):",
           recall_only, cfg["eval"]["topk_list"])

    # ---------- stage 2: recall + ranking ----------
    with timer("ranking stage (single batch scoring)", log):
        cand_df = pd.DataFrame(
            [(u, item_id) for u, items in recall_only.items() for item_id, _ in items],
            columns=["user_id", "item_id"],
        )
        cand_df["ts"] = pd.Timestamp("1970-01-01")  # placeholder: no time feature yet
        cand_df["label"] = 0                         # placeholder: unused at inference

        scored, _ = assemble_samples(
            cand_df, fs["user_profile"], fs["item_profile"],
            fs["user_stats"], fs["item_stats"], fs["user_genre_pref"], fs["genre_cols"],
        )
        scored["score"] = ranker.predict(scored)

    # Re-rank each user's candidates by model score, keep final_n.
    reranked: dict[int, list[tuple[int, float]]] = {}
    for u, g in scored.groupby("user_id"):
        g = g.sort_values("score", ascending=False).head(final_n)
        reranked[u] = list(zip(g["item_id"].astype(int).tolist(),
                               g["score"].astype(float).tolist()))
    report(f"[recall + ranking] e2e metrics (Top-{final_n}):", reranked, [final_n])

    # ---------- baseline: recall straight-through Top-N (no ranking) ----------
    recall_topn = {u: items[:final_n] for u, items in recall_only.items()}
    report(f"[baseline] recall-only Top-{final_n} (skip ranking):", recall_topn, [final_n])


if __name__ == "__main__":
    main()
