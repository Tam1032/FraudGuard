"""Metrics for imbalanced fraud data."""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score


def threshold_for_alert_rate(scores: np.ndarray, alert_rate: float) -> float:
    """Scores strictly above this value are flagged: at most `alert_rate` of transactions."""
    return float(np.quantile(scores, 1 - alert_rate))


def evaluate(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores > threshold
    prec, rec, _ = precision_recall_curve(y, scores)
    best_f1 = np.max(2 * prec * rec / np.clip(prec + rec, 1e-12, None))
    return {
        "pr_auc": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores),
        "precision_at_threshold": precision_score(y, pred, zero_division=0),
        "recall_at_threshold": recall_score(y, pred, zero_division=0),
        "f1_at_threshold": f1_score(y, pred, zero_division=0),
        "best_f1": float(best_f1),
        "alert_rate": float(pred.mean()),
        "fraud_rate": float(np.mean(y)),
    }


def recall_at_precision(y: np.ndarray, scores: np.ndarray, min_precision: float) -> float:
    """Highest recall reachable while keeping precision >= min_precision."""
    prec, rec, _ = precision_recall_curve(y, scores)
    ok = rec[prec >= min_precision]
    return float(ok.max()) if len(ok) else 0.0
