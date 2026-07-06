"""
Trains the same 8-model ensemble as your original NIFTY50 project,
but generically for any ticker's engineered features. Since you chose
"retrain per ticker on demand", this runs fresh each time (subject to
the cache layer's TTL so a repeated request for the same ticker within
the cache window doesn't retrain from scratch every time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from backend.config import get_settings

settings = get_settings()

MODEL_FACTORY = {
    "AdaBoost": lambda: AdaBoostClassifier(n_estimators=200, random_state=42),
    "GradientBoost": lambda: GradientBoostingClassifier(n_estimators=200, random_state=42),
    "LogisticRegression": lambda: LogisticRegression(max_iter=2000),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "LightGBM": lambda: LGBMClassifier(n_estimators=300, random_state=42, verbose=-1),
    "SVM": lambda: SVC(probability=True, kernel="rbf", random_state=42),
    "XGBoost": lambda: XGBClassifier(
        n_estimators=300, random_state=42, eval_metric="logloss"
    ),
}


def train_all_models(features_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Chronological (not random) train/test split — this is time series data,
    shuffling would leak the future into training and inflate accuracy.
    """
    X = features_df[feature_cols]
    y = features_df["target"]

    split_idx = int(len(X) * (1 - settings.TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    trained_models = {}
    metrics = {}

    for name, factory in MODEL_FACTORY.items():
        model = factory()
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)

        trained_models[name] = model
        metrics[name] = round(float(acc), 4)

    return {
        "models": trained_models,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


def predict_latest(bundle: dict, features_df: pd.DataFrame) -> dict:
    """Run every trained model on the most recent feature row."""
    latest = features_df[bundle["feature_cols"]].iloc[[-1]]
    latest_scaled = bundle["scaler"].transform(latest)

    per_model = []
    up_votes = down_votes = 0
    up_probs = []

    for name, model in bundle["models"].items():
        pred = int(model.predict(latest_scaled)[0])
        proba = model.predict_proba(latest_scaled)[0]
        up_prob = float(proba[1])
        up_probs.append(up_prob)

        if pred == 1:
            up_votes += 1
        else:
            down_votes += 1

        per_model.append({
            "model": name,
            "prediction": "UP" if pred == 1 else "DOWN",
            "confidence": round(float(max(proba)) * 100, 2),
            "up_probability": round(up_prob * 100, 2),
            "backtest_accuracy": bundle["metrics"][name] * 100,
        })

    majority = "UP" if up_votes >= down_votes else "DOWN"
    avg_up_prob = float(np.mean(up_probs)) * 100

    return {
        "per_model": per_model,
        "up_votes": up_votes,
        "down_votes": down_votes,
        "majority_direction": majority,
        "majority_confidence": round(max(up_votes, down_votes) / len(bundle["models"]) * 100, 2),
        "avg_up_probability": round(avg_up_prob, 2),
    }
