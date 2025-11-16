from __future__ import annotations

from typing import Tuple

import pandas as pd


def _extract_price_series(price_df: pd.DataFrame, price_col: str) -> pd.Series:
    """
    Extract a single price Series from a DataFrame that may have:
    - flat columns: ['Open', 'High', 'Low', 'Close', 'Adj Close', ...]
    - or MultiIndex columns from yfinance (e.g. ('Adj Close', 'SPY')).
    """
    # מקרה רגיל: עמודה רגילה בשם 'Adj Close'
    if price_col in price_df.columns and not isinstance(price_df.columns, pd.MultiIndex):
        return price_df[price_col]

    # אם יש MultiIndex
    if isinstance(price_df.columns, pd.MultiIndex):
        # בדיקה אם price_col מופיע ברמת העמודות הראשונה
        if price_col in price_df.columns.get_level_values(0):
            # xs = cross section: נחתוך לפי הרמה הראשונה
            tmp = price_df.xs(price_col, axis=1, level=0)
            # אם קיבלנו DataFrame (למשל כמה טיקרים) – ניקח את העמודה הראשונה
            if isinstance(tmp, pd.DataFrame):
                return tmp.iloc[:, 0]
            return tmp

    raise ValueError(
        f"Could not find a suitable price series for '{price_col}' in DataFrame columns: {price_df.columns}"
    )


def build_supervised_dataset(
    price_df: pd.DataFrame,
    price_col: str = "Adj Close",
    threshold: float = 0.02,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a simple supervised dataset for high-volatility prediction.

    Parameters
    ----------
    price_df : pd.DataFrame
        DataFrame with a DatetimeIndex and at least one price column.
    price_col : str
        Which column to use as the base price (e.g. "Adj Close").
    threshold : float
        Absolute next-day return above this threshold will be labeled as 1.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns: return_1d, vol_5d, vol_21d.
    y : pd.Series
        Binary target: 1 = high volatility event on the next day, 0 otherwise.
    """
    # נחלץ סדרת מחיר אחת נקייה
    price_series = _extract_price_series(price_df, price_col=price_col)

    # נוודא אינדקס ממוין
    price_series = price_series.sort_index()

    df = pd.DataFrame({price_col: price_series})

    # תשואה יומית
    df["return_1d"] = df[price_col].pct_change()

    # סטיות תקן מתגלגלות
    df["vol_5d"] = df["return_1d"].rolling(window=5).std()
    df["vol_21d"] = df["return_1d"].rolling(window=21).std()

    # התשואה של היום הבא
    df["next_return_1d"] = df["return_1d"].shift(-1)
    df["target_high_vol"] = (df["next_return_1d"].abs() >= threshold).astype(int)

    # מסירים שורות עם NaN שנוצרו מרולינג / שיפט
    df = df.dropna(subset=["return_1d", "vol_5d", "vol_21d", "next_return_1d"])

    feature_cols = ["return_1d", "vol_5d", "vol_21d"]
    X = df[feature_cols]
    y = df["target_high_vol"]

    return X, y
