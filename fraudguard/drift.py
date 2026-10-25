"""Model monitoring: data drift and performance decay over the "production" period.

    python -m fraudguard.drift

Reference = validation period, current = test period (later in time, never seen in
training). Writes an Evidently HTML report, PSI per feature and weekly PR-AUC.
"""

import json

import mlflow
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import average_precision_score

from fraudguard import config

PSI_ALERT = 0.2  # common rule of thumb: > 0.2 means a significant shift


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two numeric samples."""
    ref, cur = ref.dropna(), cur.dropna()
    if ref.nunique() < 2 or cur.empty:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def weekly_pr_auc(df: pd.DataFrame) -> pd.DataFrame:
    week = (df["TransactionDT"] // (7 * 86400)).astype(int)
    rows = []
    for w, g in df.groupby(week):
        if g["isFraud"].nunique() == 2:
            rows.append({"week": w, "n": len(g), "fraud_rate": g["isFraud"].mean(),
                         "pr_auc": average_precision_score(g["isFraud"], g["score"])})
    return pd.DataFrame(rows)


def main():
    meta = json.loads((config.MODEL_DIR / "meta.json").read_text())
    ref = pd.read_parquet(config.REPORT_DIR / "scored_valid.parquet")
    cur = pd.read_parquet(config.REPORT_DIR / "scored_test.parquet")

    numeric = [c for c in meta["features"] if c not in meta["categorical"] and c in ref.columns] + ["score"]
    psi_table = pd.DataFrame(
        [{"feature": c, "psi": psi(ref[c], cur[c])} for c in numeric]
    ).sort_values("psi", ascending=False)
    psi_table["drifted"] = psi_table["psi"] > PSI_ALERT
    psi_table.to_csv(config.REPORT_DIR / "psi.csv", index=False)

    weekly = pd.concat([weekly_pr_auc(ref).assign(period="reference"), weekly_pr_auc(cur).assign(period="production")])
    weekly.to_csv(config.REPORT_DIR / "weekly_pr_auc.csv", index=False)

    cols = [c for c in meta["features"] if c in ref.columns] + ["score"]
    snapshot = Report([DataDriftPreset()]).run(reference_data=ref[cols], current_data=cur[cols])
    html = config.REPORT_DIR / "drift_report.html"
    snapshot.save_html(str(html))

    summary = {
        "score_psi": float(psi_table.loc[psi_table.feature == "score", "psi"].iloc[0]),
        "n_drifted_features": int(psi_table["drifted"].sum()),
        "pr_auc_reference": average_precision_score(ref["isFraud"], ref["score"]),
        "pr_auc_production": average_precision_score(cur["isFraud"], cur["score"]),
    }
    (config.REPORT_DIR / "drift_summary.json").write_text(json.dumps(summary, indent=2))

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.EXPERIMENT)
    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_metrics(summary)
        for f in ("psi.csv", "weekly_pr_auc.csv", "drift_report.html"):
            mlflow.log_artifact(str(config.REPORT_DIR / f))

    print(json.dumps(summary, indent=2))
    print("\nMost shifted features (PSI):")
    print(psi_table.head(10).round(3).to_string(index=False))
    print(f"\nFull report: {html}")


if __name__ == "__main__":
    main()
