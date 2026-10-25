"""SHAP reason codes: why a transaction got its score."""

import warnings

import numpy as np
import pandas as pd
import shap


class Explainer:
    def __init__(self, booster):
        self._explainer = shap.TreeExplainer(booster)

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            values = self._explainer.shap_values(X)
        # Older shap versions return [class0, class1] for binary models.
        if isinstance(values, list):
            values = values[1]
        return np.asarray(values)

    def reasons(self, X: pd.DataFrame, top_k: int = 3) -> list[list[dict]]:
        """Top features pushing each row towards fraud, with their values."""
        contrib = self.contributions(X)
        out = []
        for i in range(len(X)):
            order = np.argsort(-contrib[i])[:top_k]
            out.append([
                {
                    "feature": X.columns[j],
                    "value": _jsonable(X.iloc[i, j]),
                    "contribution": round(float(contrib[i, j]), 4),
                }
                for j in order
                if contrib[i, j] > 0
            ])
        return out


def _jsonable(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)
