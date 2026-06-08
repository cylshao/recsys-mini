"""Train recall models (ItemCF + Popular) and evaluate single-route vs fused recall."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import load_config
from src.eval.recall_metrics import evaluate_recall
from src.recall.itemcf import ItemCFRecaller
from src.recall.multi_recall import MultiRecaller
from src.recall.popular import PopularRecaller
from src.utils.io import get_logger, log_metrics, save_pickle, timer


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    log = get_logger("recall")

    proc = cfg.path("processed_dir")
    art = cfg.path("artifacts_dir")
    if not (proc / "train.parquet").exists():
        log.error("processed data not found; run scripts/02_prepare_samples.py first")
        sys.exit(1)

    train = pd.read_parquet(proc / "train.parquet")
    valid = pd.read_parquet(proc / "valid.parquet")

    # ----- train recall routes -----
    with timer("train ItemCF", log):
        cf_cfg = cfg["recall"]["itemcf"]
        itemcf = ItemCFRecaller(
            sim_topk=cf_cfg["sim_topk"],
            user_history_len=cf_cfg["user_history_len"],
            use_iuf=cf_cfg["use_iuf"],
        )
        itemcf.fit(train)

    with timer("train Popular", log):
        pop = PopularRecaller(topk=cfg["recall"]["popular"]["topk"])
        pop.fit(train)

    save_pickle(itemcf, art / "recall_itemcf.pkl")
    save_pickle(pop, art / "recall_popular.pkl")

    multi = MultiRecaller(
        recallers={"itemcf": itemcf, "popular": pop},
        weights=cfg["recall"]["weights"],
    )

    # ----- evaluate against valid as ground truth -----
    gt = valid.groupby("user_id")["item_id"].apply(set).to_dict()
    eval_users = list(gt.keys())
    topk = max(cfg["eval"]["topk_list"])
    topk_list = cfg["eval"]["topk_list"]
    total_items = train["item_id"].nunique()
    log.info(f"eval users: {len(eval_users):,}")

    def report(name: str, recommendations: dict) -> None:
        log.info("-" * 60)
        log.info(f"{name} recall evaluation")
        metrics = evaluate_recall(recommendations, gt, topk_list, total_items=total_items)
        log_metrics(log, metrics)

    report("ItemCF (single route)", {u: itemcf.recall(u, topk) for u in eval_users})
    report("Popular (single route)", {u: pop.recall(u, topk) for u in eval_users})

    fused = {
        u: [(i, s) for i, s, _ in multi.recall(u, k_each=topk, k_final=topk)]
        for u in eval_users
    }
    report("Multi-route fusion (RRF)", fused)

    log.info(f"recall models saved to: {art}")


if __name__ == "__main__":
    main()
