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

# A reduced set of V columns: Vesta's 339 engineered features come in groups of
# near-duplicates (same missing pattern, correlation > 0.75); one column per group
# keeps most of the signal at a third of the memory.
V_COLUMNS = [f"V{i}" for i in (
    1, 3, 4, 6, 8, 11, 13, 14, 17, 20, 23, 26, 27, 30, 36, 37, 40, 41, 44, 47, 48, 54, 56, 59,
    62, 65, 67, 68, 70, 76, 78, 80, 82, 86, 88, 89, 91, 107, 108, 111, 115, 117, 120, 121, 123,
    124, 127, 129, 130, 136, 138, 139, 142, 147, 156, 160, 162, 165, 166, 169, 171, 173, 175,
    176, 178, 180, 182, 185, 187, 188, 198, 203, 205, 207, 209, 210, 215, 218, 220, 221, 223,
    224, 226, 228, 229, 234, 235, 238, 240, 250, 252, 253, 257, 258, 260, 261, 264, 266, 267,
    271, 274, 277, 281, 283, 284, 285, 286, 289, 291, 294, 296, 297, 301, 303, 305, 307, 309,
    310, 314, 320,
)]

# Raw columns read from the CSV.
RAW_COLUMNS = (
    ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"]
    + [f"card{i}" for i in range(1, 7)]
    + ["addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"M{i}" for i in range(1, 10)]
    + V_COLUMNS
)

CATEGORICAL = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"] + [f"M{i}" for i in range(1, 10)]

# Frequency encoding: how common each value is in the training period. Fraud often
# comes from rare cards, addresses and email domains.
# Not uid: most uids in later months are new, so training rows (always seen) and
# test rows (mostly unseen, frequency 0) would follow different distributions.
FREQ_COLUMNS = ["card1", "card2", "card3", "card5", "addr1", "P_emaildomain", "R_emaildomain"]

# Columns produced by sql/features.sql that are not model inputs.
NON_FEATURES = ["TransactionID", "isFraud", "TransactionDT", "uid"]
