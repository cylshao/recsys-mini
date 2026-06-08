"""ItemCF: item-item co-occurrence collaborative filtering recall.

Similarity (with IUF, an inverse-user-frequency penalty for power users):
    sim(i, j) = sum_{u in U_i & U_j}  1 / log(1 + |I_u|)
                ---------------------------------------------
                            sqrt(|U_i| * |U_j|)

Scoring at inference:
    score(u, j) = sum_{i in history(u)}  sim(i, j) * w_i
where w_i is a recency weight or simply uniform.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from .base import BaseRecaller


class ItemCFRecaller(BaseRecaller):
    name = "itemcf"

    def __init__(
        self,
        sim_topk: int = 100,
        user_history_len: int = 50,
        use_iuf: bool = True,
    ):
        self.sim_topk = sim_topk
        self.user_history_len = user_history_len
        self.use_iuf = use_iuf
        # item -> List[(neighbor_item, sim)], sorted desc, truncated to sim_topk
        self._item_sim: Dict[int, List[Tuple[int, float]]] = {}
        # user -> last N (item_id, ts)
        self._user_history: Dict[int, List[Tuple[int, pd.Timestamp]]] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        df = train_df.sort_values("ts")

        # 1) user -> [items], item -> interaction count
        user_items = df.groupby("user_id")["item_id"].apply(list).to_dict()
        item_count: Dict[int, int] = df.groupby("item_id").size().to_dict()

        # 2) co-occurrence counts (IUF-weighted).
        #    The matrix is symmetric, so we only iterate unordered pairs (x < y)
        #    and update both directions -> half the work of a full double loop.
        co: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for items in tqdm(user_items.values(), desc="ItemCF co-occurrence"):
            uniq = list(dict.fromkeys(items))  # dedup, keep order
            iuf = 1.0 / math.log(1 + len(items)) if self.use_iuf else 1.0
            for x in range(len(uniq)):
                a = uniq[x]
                co_a = co[a]
                for y in range(x + 1, len(uniq)):
                    b = uniq[y]
                    co_a[b] += iuf
                    co[b][a] += iuf

        # 3) normalize to cosine similarity, keep top-sim_topk neighbors
        item_sim: Dict[int, List[Tuple[int, float]]] = {}
        for a, neighbors in tqdm(co.items(), desc="ItemCF normalize"):
            norm_a = math.sqrt(item_count[a])
            scored = [(b, c / (norm_a * math.sqrt(item_count[b])))
                      for b, c in neighbors.items()]
            scored.sort(key=lambda x: x[1], reverse=True)
            item_sim[a] = scored[: self.sim_topk]
        self._item_sim = item_sim

        # 4) keep each user's most recent history (used at inference)
        history = (
            df.groupby("user_id")[["item_id", "ts"]]
            .apply(lambda x: list(zip(x["item_id"].tolist(), x["ts"].tolist())))
            .to_dict()
        )
        self._user_history = {
            u: items[-self.user_history_len:] for u, items in history.items()
        }

    def recall(self, user_id: int, k: int) -> List[Tuple[int, float]]:
        hist = self._user_history.get(user_id)
        if not hist:
            return []
        seen = {i for i, _ in hist}

        # Uniform weighting over history items; recency decay could be added here.
        scores: Dict[int, float] = defaultdict(float)
        for item_id, _ts in hist:
            for neighbor, sim in self._item_sim.get(item_id, ()):
                if neighbor not in seen:
                    scores[neighbor] += sim
        if not scores:
            return []

        out = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        # Normalize scores to [0, 1] (max is the top item) for fair fusion.
        max_s = out[0][1] or 1.0
        return [(int(i), float(s) / max_s) for i, s in out]
