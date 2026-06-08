"""Multi-route recall fusion.

Strategy:
  - each route recalls its own top-K independently
  - fuse with RRF (Reciprocal Rank Fusion); a weighted sum is also possible
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from .base import BaseRecaller


class MultiRecaller:
    def __init__(self, recallers: Dict[str, BaseRecaller], weights: Dict[str, float]):
        self.recallers = recallers
        self.weights = weights

    def recall(
        self, user_id: int, k_each: int, k_final: int, rrf_k: int = 60
    ) -> List[Tuple[int, float, Dict[str, float]]]:
        """Return [(item_id, fused_score, source_scores), ...]."""
        per_source: Dict[str, List[Tuple[int, float]]] = {}
        for name, r in self.recallers.items():
            per_source[name] = r.recall(user_id, k_each)

        # RRF fusion: score(i) = sum_s weight_s * 1/(rrf_k + rank_s(i))
        fused: Dict[int, float] = defaultdict(float)
        sources: Dict[int, Dict[str, float]] = defaultdict(dict)
        for name, items in per_source.items():
            w = self.weights.get(name, 1.0)
            for rank, (item_id, score) in enumerate(items, start=1):
                fused[item_id] += w / (rrf_k + rank)
                sources[item_id][name] = score

        out = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k_final]
        return [(int(i), float(s), sources[i]) for i, s in out]

    def recall_batch(
        self, user_ids, k_each: int, k_final: int
    ) -> Dict[int, List[Tuple[int, float, Dict[str, float]]]]:
        return {u: self.recall(u, k_each, k_final) for u in user_ids}
