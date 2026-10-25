"""Train the fraud model end to end.

    python -m fraudguard.train

Steps: load CSV into DuckDB -> SQL features -> time-based split -> Isolation Forest
anomaly score -> Logistic Regression baseline and LightGBM -> alert threshold from the
validation set -> metrics, SHAP false-positive review, MLflow logging, saved artifacts.
"""

import json
import os
import time

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraudguard import config, features
from fraudguard.explain import Explainer
from fraudguard.metrics import evaluate, recall_at_precision, threshold_for_alert_rate

N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "3000"))
SEED = 42


def time_split(df: pd.DataFrame):
    t0, t1 = df["TransactionDT"].min(), df["TransactionDT"].max()
    train_end = t0 + config.TRAIN_END * (t1 - t0)
    valid_end = t0 + config.VALID_END * (t1 - t0)
    train = df[df["TransactionDT"] < train_end]
    valid = df[(df["TransactionDT"] >= train_end) & (df["TransactionDT"] < valid_end)]
    test = df[df["TransactionDT"] >= valid_end]
    return train, valid, test


def fit_iforest(X: pd.DataFrame, y: np.ndarray, numeric_cols: list[str]) -> IsolationForest:
    """Unsupervised anomaly detector fitted on legitimate traffic only."""
    legit = X.loc[y == 0, numeric_cols].fillna(-999)
    sample = legit.sample(min(len(legit), 200_000), random_state=SEED)
    return IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(sample)


def anomaly_score(iforest: IsolationForest, X: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
    # score_samples is higher for normal points; flip it so higher = more anomalous.
    return -iforest.score_samples(X[numeric_cols].fillna(-999))


def main():
    started = time.time()
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.REPORT_DIR.mkdir(exist_ok=True)

    print("Loading CSV into DuckDB and computing SQL features...")
    con = features.load_history()
    df = features.compute_all(con)
    con.close()
    df = df[df["isFraud"].notna()]  # skip transactions the API added to history (no label)
    train, valid, test = time_split(df)
    print(f"rows: train={len(train):,} valid={len(valid):,} test={len(test):,}")

    freq_maps = features.frequency_maps(train)
    train, valid, test = (features.add_frequency(d, freq_maps) for d in (train, valid, test))

    X_train = features.to_model_frame(train)
    categories = {c: list(X_train[c].cat.categories) for c in config.CATEGORICAL}
    X_valid = features.to_model_frame(valid, categories)
    X_test = features.to_model_frame(test, categories)
    y_train, y_valid, y_test = (d["isFraud"].to_numpy().astype(int) for d in (train, valid, test))

    numeric_cols = [c for c in X_train.columns if c not in config.CATEGORICAL]
    print("Fitting Isolation Forest...")
    iforest = fit_iforest(X_train, y_train, numeric_cols)
    for X in (X_train, X_valid, X_test):
        X["iforest_score"] = anomaly_score(iforest, X, numeric_cols)
    numeric_cols = numeric_cols + ["iforest_score"]

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.EXPERIMENT)
    results = {}

    print("Training Logistic Regression baseline...")
    with mlflow.start_run(run_name="logreg_baseline"):
        logreg = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        ).fit(X_train[numeric_cols], y_train)
        s_valid = logreg.predict_proba(X_valid[numeric_cols])[:, 1]
        s_test = logreg.predict_proba(X_test[numeric_cols])[:, 1]
        thr = threshold_for_alert_rate(s_valid, config.ALERT_RATE)
        results["logreg"] = _log_eval(y_valid, s_valid, y_test, s_test, thr)
        mlflow.log_params({"model": "logreg", "class_weight": "balanced", "alert_rate": config.ALERT_RATE})

    print("Training LightGBM...")
    with mlflow.start_run(run_name="lightgbm"):
        # POS_WEIGHT=auto weights fraud by the class ratio (about 27x); 1 = no weighting.
        # PR-AUC and the alert threshold only use the ranking, so weighting is optional.
        pw = os.getenv("POS_WEIGHT", "1")
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1) if pw == "auto" else float(pw)
        params = dict(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.03,
            num_leaves=127,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            # Only track PR-AUC. With the default extra log-loss metric, early stopping
            # halts as soon as log-loss stops improving, which with class weighting is
            # after a handful of trees.
            metric="average_precision",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
        model = lgb.LGBMClassifier(**params).fit(
            X_train, y_train,
            eval_X=(X_valid,), eval_y=(y_valid,),
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(200)],
        )
        s_valid = model.predict_proba(X_valid)[:, 1]
        s_test = model.predict_proba(X_test)[:, 1]
        thr = threshold_for_alert_rate(s_valid, config.ALERT_RATE)
        results["lightgbm"] = _log_eval(y_valid, s_valid, y_test, s_test, thr)
        mlflow.log_params({**{k: v for k, v in params.items() if k != "n_jobs"},
                           "best_iteration": model.best_iteration_, "alert_rate": config.ALERT_RATE})

        print(f"LightGBM best iteration: {model.best_iteration_}")
        booster = model.booster_
        booster.save_model(str(config.MODEL_DIR / "lgbm.txt"), num_iteration=model.best_iteration_)
        joblib.dump(iforest, config.MODEL_DIR / "iforest.joblib")
        joblib.dump(freq_maps, config.MODEL_DIR / "freq_maps.joblib")
        meta = {
            "features": list(X_train.columns),
            "categorical": config.CATEGORICAL,
            "categories": categories,
            "iforest_cols": numeric_cols[:-1],
            "threshold": thr,
            "alert_rate": config.ALERT_RATE,
            "metrics": results,
        }
        (config.MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

        # Keep scored validation/test sets for drift monitoring.
        valid.assign(score=s_valid).to_parquet(config.REPORT_DIR / "scored_valid.parquet")
        test.assign(score=s_test).to_parquet(config.REPORT_DIR / "scored_test.parquet")

        print("Reviewing top false positives with SHAP...")
        fp = review_false_positives(booster, X_test, test, s_test, y_test)
        fp.to_csv(config.REPORT_DIR / "false_positives.csv", index=False)

        importance = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
        importance.sort_values(ascending=False).to_csv(config.REPORT_DIR / "feature_importance.csv")

        mlflow.log_artifacts(str(config.MODEL_DIR), artifact_path="model")
        mlflow.log_artifact(str(config.REPORT_DIR / "false_positives.csv"))
        mlflow.log_artifact(str(config.REPORT_DIR / "feature_importance.csv"))

    (config.REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print_table(results)
    print(f"Done in {time.time() - started:.0f}s. Artifacts in {config.MODEL_DIR}")


def _log_eval(y_valid, s_valid, y_test, s_test, thr) -> dict:
    out = {"threshold": thr}
    for name, y, s in (("valid", y_valid, s_valid), ("test", y_test, s_test)):
        m = evaluate(y, s, thr)
        m["recall_at_precision_50"] = recall_at_precision(y, s, 0.5)
        out[name] = m
        mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items()})
    return out


def review_false_positives(booster, X_test, test, scores, y, n=50) -> pd.DataFrame:
    """Highest-scoring legitimate transactions and the features that flagged them."""
    idx = np.where(y == 0)[0]
    top = idx[np.argsort(-scores[idx])[:n]]
    reasons = Explainer(booster).reasons(X_test.iloc[top][booster.feature_name()])
    return pd.DataFrame({
        "TransactionID": test["TransactionID"].to_numpy()[top],
        "score": scores[top],
        "reasons": ["; ".join(f"{r['feature']}={r['value']}" for r in rs) for rs in reasons],
    })


def print_table(results: dict):
    rows = []
    for model, r in results.items():
        t = r["test"]
        rows.append([model, t["pr_auc"], t["roc_auc"], t["precision_at_threshold"],
                     t["recall_at_threshold"], t["f1_at_threshold"], t["recall_at_precision_50"]])
    table = pd.DataFrame(rows, columns=["model", "PR-AUC", "ROC-AUC", "precision@alert", "recall@alert",
                                        "F1@alert", "recall@P>=0.5"])
    print(f"\nTest set (last {1 - config.VALID_END:.0%} of time), alert rate {config.ALERT_RATE:.1%}:")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
