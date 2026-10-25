# FraudGuard: transaction fraud detection and risk scoring

An end-to-end fraud detection service on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
payment data: SQL behavioural features, an imbalanced-data model with an anomaly score,
a REST API that returns a risk score with reasons, and drift monitoring over later months.

```mermaid
flowchart LR
    A[train_transaction.csv] --> B[(DuckDB history)]
    B --> C[sql/features.sql<br/>velocity, amount deviation, timing]
    C --> D[Isolation Forest<br/>anomaly score]
    D --> E[LightGBM<br/>cost-sensitive]
    E --> F[Alert threshold<br/>from analyst budget]
    E --> G[MLflow<br/>params, metrics, model]
    F --> H[FastAPI /score<br/>risk score + SHAP reasons]
    B --> H
    E --> I[Evidently + PSI<br/>drift report]
```

## Results

Test set = the last 20% of the timeline, never seen during training or threshold tuning.
Alert threshold set so that 2% of transactions are sent to analysts.

| Model | PR-AUC | ROC-AUC | Precision @ alert | Recall @ alert | Recall @ precision ≥ 0.5 |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression (baseline) | _fill in_ | | | | |
| LightGBM + SQL features + anomaly score | _fill in_ | | | | |

Numbers come from `reports/metrics.json` after `make train`. Drift results are in
`reports/drift_summary.json` and `reports/drift_report.html`.

## Quick start

```bash
pip install -r requirements.txt

# 1. Check the setup end to end on fake data (about 1 minute)
make smoke

# 2. Real data: download train_transaction.csv from
#    https://www.kaggle.com/c/ieee-fraud-detection/data into data/raw/
make train      # features, models, threshold, SHAP review, MLflow logging
make drift      # PSI per feature, weekly PR-AUC, Evidently HTML report
make serve      # API on http://localhost:8000/docs
make mlflow     # experiment UI on http://localhost:5000
```

With Docker:

```bash
docker compose run --rm train      # train + drift inside the container
docker compose up api mlflow
```

Score a transaction:

```bash
curl -X POST localhost:8000/score -H 'Content-Type: application/json' -d '{
  "TransactionID": 4000001, "TransactionDT": 15811200, "TransactionAmt": 249.0,
  "ProductCD": "W", "card1": 13926, "card4": "visa", "card6": "debit",
  "addr1": 315, "P_emaildomain": "gmail.com", "D1": 14
}'
```

```json
{
  "transaction_id": 4000001,
  "fraud_probability": 0.031,
  "risk_score": 31,
  "is_alert": false,
  "threshold": 0.27,
  "reasons": [
    {"feature": "uid_tx_1h", "value": 3, "contribution": 0.41},
    {"feature": "amt_to_uid_mean_30d", "value": 4.2, "contribution": 0.22}
  ]
}
```

(Example response shape; values depend on the trained model.)

## Design decisions

- **Time-based split, not random.** Fraud patterns change over time; a random split leaks
  future behaviour into training and overstates performance. Train on the first 65% of the
  timeline, tune on the next 15%, test on the last 20%.
- **One SQL file for training and serving.** `sql/features.sql` runs over the full history at
  training time and over one card's last 30 days plus the new transaction in the API.
  `tests/test_pipeline.py` checks that both paths give the same feature values, which rules out
  training/serving skew. A cumulative "transactions so far" feature was dropped because the API
  cannot compute it from a 30-day window and it drifts by construction.
- **Client id (`uid`).** Card id alone mixes many people. `card1 + addr1 + (day - D1)` approximates
  one account, because D1 counts days since the account's first transaction.
- **Imbalance.** About 3.5% of transactions are fraud. The model uses `scale_pos_weight`
  instead of SMOTE: oversampling synthetic points in a 60-feature space with many missing
  values adds noise and makes the pipeline slower without a clear gain.
- **Metrics.** Accuracy is meaningless here (96.5% by predicting "legit"). PR-AUC is the main
  metric; precision and recall are reported at the operating threshold.
- **Threshold from an alert budget.** Analysts can only review so many cases. The threshold flags
  `ALERT_RATE` (default 2%) of validation traffic, then is applied unchanged to the test period.
- **Anomaly score as a feature.** An Isolation Forest fitted on legitimate traffic only gives an
  unsupervised "how unusual is this" signal that also helps on fraud types missing from the labels.
- **Reason codes.** SHAP values give the top three features behind each alert, so an analyst sees
  *why* a transaction was flagged. `reports/false_positives.csv` lists the 50 highest-scoring
  legitimate transactions with their reasons, as a starting point for error analysis.
- **Drift monitoring.** PSI per feature and on the score (> 0.2 = significant shift), weekly PR-AUC
  through the production period, and an Evidently data drift report.

## Project layout

```
sql/features.sql          behavioural features (DuckDB)
fraudguard/features.py    load CSV, run the SQL for batch and single transactions
fraudguard/train.py       split, Isolation Forest, baselines, LightGBM, threshold, MLflow
fraudguard/metrics.py     PR-AUC, metrics at threshold, recall at fixed precision
fraudguard/explain.py     SHAP reason codes
fraudguard/scorer.py      loads artifacts, scores one transaction
fraudguard/api.py         FastAPI app
fraudguard/drift.py       PSI, weekly PR-AUC, Evidently report
scripts/make_synthetic.py fake data with the same schema, for smoke tests only
tests/                    train/serve feature parity, API tests
```

## Next steps

- Retrain on a rolling window when score PSI crosses the threshold, and compare with the frozen model.
- Streamlit dashboard for the alert queue.
- Graph features (shared devices, emails, addresses between cards).
- LLM-written alert summaries from the SHAP reasons.
