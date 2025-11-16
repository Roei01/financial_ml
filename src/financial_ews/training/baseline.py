from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from financial_ews.data.yahoo_downloader import download_price_history
from financial_ews.data.dataset_builder import build_supervised_dataset


def train_test_split_time_series(
    X, y, train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a time-series dataset into train and test sets, preserving order.
    """
    n = len(X)
    split_idx = int(n * train_ratio)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def main() -> None:
    # 1) הורדת נתונים
    df = download_price_history("SPY", start="2015-01-01")

    # 2) בניית dataset
    X, y = build_supervised_dataset(df, price_col="Adj Close", threshold=0.02)

    # 3) חלוקה ל-train/test לפי ציר הזמן
    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y, train_ratio=0.8)

    # 4) סקיילינג לפיצ'רים (חשוב למודלים ליניאריים)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5) מודל בסיסי
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # 6) הערכה
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {auc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    main()

