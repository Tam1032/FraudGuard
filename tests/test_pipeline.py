"""Smoke tests. Run after training (`make smoke` does both on synthetic data)."""

import duckdb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudguard import config, features

pytestmark = pytest.mark.skipif(
    not (config.MODEL_DIR / "meta.json").exists(), reason="train a model first (make smoke)"
)


def _sample_txns(n=30):
    con = duckdb.connect(str(config.HISTORY_DB), read_only=True)
    rows = con.execute(
        "SELECT * FROM transactions WHERE TransactionID IN "
        "(SELECT TransactionID FROM transactions USING SAMPLE 1000 ROWS) ORDER BY TransactionDT"
    ).df()
    con.close()
    rows = rows.astype(object).where(rows.notna(), None)
    return rows.head(n).to_dict("records")


def test_serving_features_match_training_features():
    """The API's one-row feature query must give the same values as the batch query."""
    con = duckdb.connect(str(config.HISTORY_DB), read_only=True)
    batch = features.compute_all(con).set_index("TransactionID")
    con.close()
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{config.HISTORY_DB}' AS h (READ_ONLY)")
    con.execute("CREATE TABLE transactions AS SELECT * FROM h.transactions")
    for txn in _sample_txns():
        online = features.compute_one(con, txn).set_index("TransactionID").drop(columns="isFraud")
        expected = batch.loc[online.index, online.columns]
        # Rows sharing the exact same timestamp are peers in a RANGE window; skip those rare cases.
        if (batch["TransactionDT"] == txn["TransactionDT"]).sum() > 1:
            continue
        pd.testing.assert_frame_equal(online, expected, check_dtype=False, check_exact=False)


def test_score_endpoint():
    from fraudguard.api import app

    txn = _sample_txns(1)[0]
    txn["TransactionID"] = 99_999_999  # new id so the history store is not confused
    payload = {k: v for k, v in txn.items() if k != "isFraud"}
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"]
        resp = client.post("/score", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert 0 <= body["risk_score"] <= 1000
    assert isinstance(body["is_alert"], bool)
    assert len(body["reasons"]) <= 3


def test_score_rejects_bad_amount():
    from fraudguard.api import app

    with TestClient(app) as client:
        resp = client.post("/score", json={"TransactionID": 1, "TransactionDT": 1, "TransactionAmt": -5, "card1": 1})
    assert resp.status_code == 422
