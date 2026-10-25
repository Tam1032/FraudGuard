"""Load raw transactions into DuckDB and compute features with sql/features.sql."""

import duckdb
import pandas as pd

from fraudguard import config

# The API only needs this much history to fill the longest window (30 days).
HISTORY_WINDOW_SECS = 30 * 86400


def _feature_sql() -> str:
    return config.FEATURE_SQL.read_text()


def load_history(csv_path=config.RAW_CSV, db_path=config.HISTORY_DB) -> duckdb.DuckDBPyConnection:
    """Copy the needed raw columns from the Kaggle CSV into a DuckDB file."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download train_transaction.csv from "
            "https://www.kaggle.com/c/ieee-fraud-detection/data into data/raw/."
        )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    cols = ", ".join(config.RAW_COLUMNS)
    con.execute(
        f"CREATE OR REPLACE TABLE transactions AS "
        f"SELECT {cols} FROM read_csv_auto('{csv_path}', header=true)"
    )
    return con


def compute_all(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Features for every transaction in the history table (training time)."""
    con.execute("CREATE OR REPLACE TEMP VIEW tx AS SELECT * FROM transactions")
    return con.execute(_feature_sql()).df()


def _stage_new(con: duckdb.DuckDBPyConnection, txn: dict) -> dict:
    """Put one incoming transaction in temp table new_tx, cast to the history schema."""
    row = {c: txn.get(c) for c in config.RAW_COLUMNS}
    row["isFraud"] = None
    con.execute("CREATE OR REPLACE TEMP TABLE new_tx AS SELECT * FROM transactions LIMIT 0")
    incoming = pd.DataFrame([row])
    con.register("incoming", incoming)
    con.execute("INSERT INTO new_tx BY NAME SELECT * FROM incoming")
    con.unregister("incoming")
    return row


def compute_one(con: duckdb.DuckDBPyConnection, txn: dict) -> pd.DataFrame:
    """Features for one new transaction, using the card's recent history (serving time)."""
    row = _stage_new(con, txn)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE tx AS
        SELECT * FROM transactions
        WHERE card1 IS NOT DISTINCT FROM ?
          AND TransactionDT BETWEEN ? AND ?
          AND TransactionID <> ?
        UNION ALL
        SELECT * FROM new_tx
        """,
        [row["card1"], row["TransactionDT"] - HISTORY_WINDOW_SECS, row["TransactionDT"], row["TransactionID"]],
    )
    feats = con.execute(_feature_sql()).df()
    return feats[feats["TransactionID"] == row["TransactionID"]].tail(1)


def append_to_history(con: duckdb.DuckDBPyConnection, txn: dict) -> None:
    """Store a scored transaction so later velocity features can see it."""
    _stage_new(con, txn)
    con.execute("INSERT INTO transactions SELECT * FROM new_tx")


def frequency_maps(df: pd.DataFrame) -> dict:
    """Share of rows holding each value, learned from the training period only."""
    return {c: df[c].astype(str).value_counts(normalize=True).to_dict() for c in config.FREQ_COLUMNS}


def add_frequency(df: pd.DataFrame, maps: dict) -> pd.DataFrame:
    """Add <col>_freq columns; values never seen in training get 0."""
    df = df.copy()
    for c, m in maps.items():
        df[f"{c}_freq"] = df[c].astype(str).map(m).fillna(0.0).astype("float64")
    return df


def to_model_frame(df: pd.DataFrame, categories: dict | None = None) -> pd.DataFrame:
    """Drop id/label columns and set categorical dtypes the way LightGBM expects."""
    X = df.drop(columns=[c for c in config.NON_FEATURES if c in df.columns]).copy()
    for col in config.CATEGORICAL:
        if categories is not None:
            X[col] = pd.Categorical(X[col], categories=categories[col])
        else:
            X[col] = X[col].astype("category")
    numeric = [c for c in X.columns if c not in config.CATEGORICAL]
    X[numeric] = X[numeric].apply(pd.to_numeric, errors="coerce").astype("float32")
    return X.copy()  # one consolidated block; avoids pandas fragmentation warnings
