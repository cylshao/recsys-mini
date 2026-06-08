"""LightGBM CTR ranker (binary classification)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass
class LGBRanker:
    params: Dict[str, Any] = field(default_factory=dict)
    num_boost_round: int = 500
    early_stopping_rounds: int = 30
    feature_cols: List[str] = field(default_factory=list)
    model: Optional[lgb.Booster] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "label",
    ) -> None:
        self.feature_cols = feature_cols
        dtrain = lgb.Dataset(train_df[feature_cols], label=train_df[label_col])
        dvalid = lgb.Dataset(
            valid_df[feature_cols], label=valid_df[label_col], reference=dtrain
        )
        self.model = lgb.train(
            params=self.params,
            train_set=dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=True),
                lgb.log_evaluation(period=50),
            ],
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Ranker not fitted")
        return self.model.predict(df[self.feature_cols])

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Ranker not fitted")
        return pd.DataFrame({
            "feature": self.feature_cols,
            "gain": self.model.feature_importance(importance_type="gain"),
            "split": self.model.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False).reset_index(drop=True)
