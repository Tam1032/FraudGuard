"""Load trained artifacts and score single transactions."""

import json
import threading

import duckdb
import joblib
import lightgbm as lgb
import numpy as np

from fraudguard import config, features
from fraudguard.explain import Explainer


class Scorer:
    def __init__(self, model_dir=config.MODEL_DIR, db_path=config.HISTORY_DB, record_history=True):
        meta = json.loads((model_dir / "meta.json").read_text())
        self.features = meta["features"]
        self.categories = meta["categories"]
        self.iforest_cols = meta["iforest_cols"]
        self.threshold = meta["threshold"]
        self.booster = lgb.Booster(model_file=str(model_dir / "lgbm.txt"))
        self.iforest = joblib.load(model_dir / "iforest.joblib")
        self.explainer = Explainer(self.booster)
        self.con = duckdb.connect(str(db_path))
        self.record_history = record_history
        self._lock = threading.Lock()  # one DuckDB connection, used by one request at a time

    def score(self, txn: dict) -> dict:
        with self._lock:
            row = features.compute_one(self.con, txn)
            if self.record_history:
                features.append_to_history(self.con, txn)
        X = features.to_model_frame(row, self.categories)
        X["iforest_score"] = -self.iforest.score_samples(X[self.iforest_cols].fillna(-999))
        X = X[self.features]
        prob = float(self.booster.predict(X)[0])
        return {
            "transaction_id": int(txn["TransactionID"]),
            "fraud_probability": round(prob, 6),
            # 0-1000 like a credit score; easier for analysts to read than a probability.
            "risk_score": int(np.round(prob * 1000)),
            "is_alert": prob > self.threshold,
            "threshold": round(self.threshold, 6),
            "reasons": self.explainer.reasons(X)[0],
        }
