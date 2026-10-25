"""Paths and settings shared by the training pipeline and the API."""

import os
from pathlib import Path

ROOT = Path(os.getenv("FRAUDGUARD_HOME", Path(__file__).resolve().parents[1]))

RAW_CSV = ROOT / "data" / "raw" / "train_transaction.csv"
HISTORY_DB = ROOT / "data" / "history.duckdb"
FEATURE_SQL = ROOT / "sql" / "features.sql"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
EXPERIMENT = "fraudguard"

# Time-based split on TransactionDT (fractions of the full time range).
# Train on the first 65%, tune on the next 15%, treat the last 20% as "production".
TRAIN_END = 0.65
VALID_END = 0.80

# Share of transactions analysts can review. The alert threshold is set so that
# this share of validation traffic is flagged.
ALERT_RATE = float(os.getenv("ALERT_RATE", "0.02"))

# Raw columns read from the CSV. V1..V339 are left out on purpose: they add little
# on top of the behavioural features and make the pipeline much slower.
RAW_COLUMNS = (
    ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"]
    + [f"card{i}" for i in range(1, 7)]
    + ["addr1", "addr2", "dist1", "P_emaildomain", "R_emaildomain"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in (1, 2, 3, 4, 10, 15)]
    + [f"M{i}" for i in (4, 5, 6)]
)

CATEGORICAL = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "M4", "M5", "M6"]

# Columns produced by sql/features.sql that are not model inputs.
NON_FEATURES = ["TransactionID", "isFraud", "TransactionDT", "uid"]
