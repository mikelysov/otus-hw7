"""
ДЗ 7: Финальный ансамбль - Создание ML ансамбля
"""

import os
import json
from config import (
    OUTPUT_DIR,
    ENSEMBLE_DIR,
    LABEL_THRESHOLD,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    GB_N_ESTIMATORS,
    GB_MAX_DEPTH,
    ENSEMBLE_WEIGHTS,
    FEATURES,
)
from utils import logger, load_data

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def create_labels(df, threshold=LABEL_THRESHOLD):
    """Create labels for classification"""
    df = df.copy()
    df = df.sort_values(["tic", "date"]).reset_index(drop=True)
    df["future_return"] = df.groupby("tic")["close"].pct_change(periods=5).shift(-5)
    df["label"] = (df["future_return"] > threshold).astype(int)
    return df.dropna(subset=["future_return", "label"])


def main():
    logger.info("=" * 60)
    logger.info("ДЗ 7: Финальный ансамбль - Создание ML ансамбля")

    train_df, trade_df = load_data(OUTPUT_DIR)
    feature_cols = [f for f in FEATURES["ml"] if f in train_df.columns]

    for f in feature_cols:
        train_df[f] = train_df[f].fillna(0)
        trade_df[f] = trade_df[f].fillna(0)

    logger.info("Создание меток")
    train_labeled = create_labels(train_df)
    trade_labeled = create_labels(trade_df)
    logger.info(f"Train labels: {train_labeled['label'].value_counts().to_dict()}")

    X_train = train_labeled[feature_cols].values
    y_train = train_labeled["label"].values
    X_trade = trade_labeled[feature_cols].values
    y_trade = trade_labeled["label"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_trade_scaled = scaler.transform(X_trade)

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    logger.info("Обучение Random Forest")
    rf_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_split, y_train_split)
    rf_acc = accuracy_score(y_trade, rf_model.predict(X_trade_scaled))
    joblib.dump(rf_model, os.path.join(ENSEMBLE_DIR, "rf_model.pkl"))

    logger.info("Обучение Gradient Boosting")
    gb_model = GradientBoostingClassifier(
        n_estimators=GB_N_ESTIMATORS,
        max_depth=GB_MAX_DEPTH,
        learning_rate=0.1,
        random_state=42,
    )
    gb_model.fit(X_train_split, y_train_split)
    gb_acc = accuracy_score(y_trade, gb_model.predict(X_trade_scaled))
    joblib.dump(gb_model, os.path.join(ENSEMBLE_DIR, "gb_model.pkl"))
    joblib.dump(scaler, os.path.join(ENSEMBLE_DIR, "scaler.pkl"))

    rf_prob = rf_model.predict_proba(X_trade_scaled)[:, 1]
    gb_prob = gb_model.predict_proba(X_trade_scaled)[:, 1]
    ensemble_prob = (rf_prob + gb_prob) / 2
    ensemble_acc = accuracy_score(y_trade, (ensemble_prob >= 0.5).astype(int))

    w_ensemble_prob = (
        ENSEMBLE_WEIGHTS["rf"] * rf_prob + ENSEMBLE_WEIGHTS["gb"] * gb_prob
    )
    w_ensemble_acc = accuracy_score(y_trade, (w_ensemble_prob >= 0.5).astype(int))

    logger.info(f"RF Accuracy: {rf_acc:.4f}")
    logger.info(f"GB Accuracy: {gb_acc:.4f}")
    logger.info(f"Ensemble Accuracy: {ensemble_acc:.4f}")
    logger.info(f"Weighted Accuracy: {w_ensemble_acc:.4f}")

    ensemble_results = {
        "rf_accuracy": rf_acc,
        "gb_accuracy": gb_acc,
        "ensemble_accuracy": ensemble_acc,
        "weighted_accuracy": w_ensemble_acc,
    }
    with open(os.path.join(ENSEMBLE_DIR, "ensemble_results.json"), "w") as f:
        json.dump(ensemble_results, f, indent=2)

    logger.info("Ансамбль создан!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
