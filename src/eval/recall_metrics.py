"""Recall-stage metrics: HitRate@K, Recall@K, NDCG@K, Coverage@K."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple


def evaluate_recall(
    recommendations: Dict[int, List[Tuple[int, float]]],
    ground_truth: Dict[int, set[int]],
    k_list: Iterable[int],
    total_items: int | None = None,
) -> Dict[str, float]:
    """Evaluate recall results.

    Args:
        recommendations: user -> [(item, score), ...], already sorted by score
        ground_truth:    user -> set of truly interacted items (from valid/test)
        k_list:          the K values to evaluate
        total_items:     catalog size, used to compute coverage
    """
    metrics: Dict[str, float] = {}
    eval_users = [u for u in ground_truth if u in recommendations and ground_truth[u]]
    n_users = len(eval_users)
    if n_users == 0:
        return {"n_eval_users": 0}

    for k in k_list:
        hits = 0
        recalls: List[float] = []
        ndcgs: List[float] = []
        all_recommended: set[int] = set()

        for u in eval_users:
            truth = ground_truth[u]
            topk = [item for item, _ in recommendations[u][:k]]
            all_recommended.update(topk)

            hit_set = set(topk) & truth
            if hit_set:
                hits += 1
            recalls.append(len(hit_set) / len(truth))

            # NDCG
            dcg = 0.0
            for rank, item in enumerate(topk, start=1):
                if item in truth:
                    dcg += 1.0 / math.log2(rank + 1)
            ideal_n = min(len(truth), k)
            idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_n + 1))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        metrics[f"HitRate@{k}"] = hits / n_users
        metrics[f"Recall@{k}"] = sum(recalls) / n_users
        metrics[f"NDCG@{k}"] = sum(ndcgs) / n_users
        if total_items:
            metrics[f"Coverage@{k}"] = len(all_recommended) / total_items

    metrics["n_eval_users"] = n_users
    return metrics
