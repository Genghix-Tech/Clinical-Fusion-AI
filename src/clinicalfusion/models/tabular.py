"""Gradient-boosted tabular model (XGBoost or CatBoost). Both handle NaN natively."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..data.schema import TAB_FEATURES


class TabularGBM:
    def __init__(self, backend: str = "xgboost", seed: int = 0):
        self.backend = backend
        self.seed = seed
        self.model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, Xv: pd.DataFrame, yv: np.ndarray) -> "TabularGBM":
        X, Xv = X[TAB_FEATURES].to_numpy(float), Xv[TAB_FEATURES].to_numpy(float)
        if self.backend == "xgboost":
            from xgboost import XGBClassifier

            self.model = XGBClassifier(
                n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=3, reg_lambda=2.0, eval_metric="logloss", early_stopping_rounds=30,
                random_state=self.seed, n_jobs=2, verbosity=0,
            )
            self.model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        elif self.backend == "catboost":
            try:
                from catboost import CatBoostClassifier
            except ImportError as e:  # pragma: no cover
                raise ImportError("catboost backend requested: pip install 'clinicalfusion[catboost]'") from e
            self.model = CatBoostClassifier(
                iterations=600, depth=4, learning_rate=0.05, random_seed=self.seed, verbose=False,
                early_stopping_rounds=30, thread_count=2,
            )
            self.model.fit(X, y, eval_set=(Xv, yv))
        else:
            raise ValueError(f"unknown tab_backend {self.backend!r} (use 'xgboost' or 'catboost')")
        return self

    def logit(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.model.predict_proba(X[TAB_FEATURES].to_numpy(float))[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Per-patient additive SHAP-style contributions to the log-odds, shape (N, F). Not causal."""
        A = X[TAB_FEATURES].to_numpy(float)
        if self.backend == "xgboost":
            import xgboost as xgb

            c = self.model.get_booster().predict(xgb.DMatrix(A), pred_contribs=True)
        else:  # catboost
            from catboost import Pool

            c = self.model.get_feature_importance(Pool(A), type="ShapValues")
        return np.asarray(c)[:, : len(TAB_FEATURES)]

    # persistence ---------------------------------------------------------------------------
    def to_bytes(self) -> bytes:
        if self.backend == "xgboost":
            return bytes(self.model.get_booster().save_raw("ubj"))
        import tempfile, os  # catboost has no raw-bytes API
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.cbm")
            self.model.save_model(p)
            return open(p, "rb").read()

    @classmethod
    def from_bytes(cls, backend: str, raw: bytes) -> "TabularGBM":
        obj = cls(backend)
        if backend == "xgboost":
            from xgboost import XGBClassifier

            m = XGBClassifier()
            m.load_model(bytearray(raw))
            obj.model = m
        else:
            import tempfile, os
            from catboost import CatBoostClassifier

            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "m.cbm")
                open(p, "wb").write(raw)
                m = CatBoostClassifier()
                m.load_model(p)
            obj.model = m
        return obj
