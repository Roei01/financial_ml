from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from financial_ews.data.yahoo_downloader import download_price_history
from financial_ews.data.dataset_builder import _extract_price_series


@dataclass
class NextDayForecast:
    """
    אובייקט קטן שמרכז את התחזית ליום הבא.
    """
    ticker: str
    as_of_date: pd.Timestamp
    high_vol_prob: float  # הסתברות לתנודתיות חזקה
    up_prob: float        # הסתברות לעלייה
    down_prob: float      # הסתברות לירידה/לא־עלייה

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of_date": self.as_of_date.isoformat(),
            "high_vol_prob": self.high_vol_prob,
            "up_prob": self.up_prob,
            "down_prob": self.down_prob,
        }


def _build_feature_frame(
    raw_df: pd.DataFrame,
    price_col: str = "Adj Close",
    threshold: float = 0.02,
) -> pd.DataFrame:
    """
    בונה DataFrame עם פיצ'רים + מטרות ליום הבא:
    - return_1d
    - vol_5d
    - vol_21d
    - next_return_1d
    - target_high_vol  (אירוע תנודתיות חזקה כן/לא)
    - direction_up     (האם מחר עלייה כן/לא)
    """
    price_series = _extract_price_series(raw_df, price_col=price_col)
    price_series = price_series.sort_index()

    df = pd.DataFrame({price_col: price_series})

    # תשואה יומית
    df["return_1d"] = df[price_col].pct_change()

    # סטיית תקן מתגלגלת (מדד תנודתיות)
    df["vol_5d"] = df["return_1d"].rolling(window=5).std()
    df["vol_21d"] = df["return_1d"].rolling(window=21).std()

    # תשואה של היום הבא
    df["next_return_1d"] = df["return_1d"].shift(-1)

    # מטרה 1: האם מחר תהיה תנועה חזקה?
    df["target_high_vol"] = (df["next_return_1d"].abs() >= threshold).astype(int)

    # מטרה 2: כיוון (האם מחר עלייה?)
    df["direction_up"] = (df["next_return_1d"] > 0).astype(int)

    # מנקים שורות עם NaN שנוצר מרולינג/שיפט
    df = df.dropna(
        subset=["return_1d", "vol_5d", "vol_21d", "next_return_1d"]
    )

    return df


def forecast_next_day_for_ticker(
    ticker: str,
    start: str = "2015-01-01",
    threshold: float = 0.02,
) -> Tuple[NextDayForecast, pd.DataFrame]:
    """
    מאמן מודלים על כל ההיסטוריה של המניה, ומחזיר תחזית ליום הבא
    + את טבלת הפיצ'רים (לשימוש לגרף).
    """
    raw_df = download_price_history(ticker, start=start)

    feat_df = _build_feature_frame(raw_df, price_col="Adj Close", threshold=threshold)

    feature_cols = ["return_1d", "vol_5d", "vol_21d"]
    X = feat_df[feature_cols]
    y_vol = feat_df["target_high_vol"]
    y_dir = feat_df["direction_up"]

    # --- מודל לתנודתיות גבוהה ---
    vol_scaler = StandardScaler()
    X_vol_scaled = vol_scaler.fit_transform(X)

    vol_model = LogisticRegression(
        class_weight="balanced",
        max_iter=500,
        solver="lbfgs",
    )
    vol_model.fit(X_vol_scaled, y_vol)

    # --- מודל לכיוון (UP / DOWN) ---
    dir_scaler = StandardScaler()
    X_dir_scaled = dir_scaler.fit_transform(X)

    dir_model = LogisticRegression(
        class_weight="balanced",
        max_iter=500,
        solver="lbfgs",
    )
    dir_model.fit(X_dir_scaled, y_dir)

    # --- תחזית עבור השורה האחרונה (היום האחרון) ---
    x_last = X.iloc[[-1]]

    x_last_vol = vol_scaler.transform(x_last)
    x_last_dir = dir_scaler.transform(x_last)

    p_high_vol = float(vol_model.predict_proba(x_last_vol)[0, 1])
    p_up = float(dir_model.predict_proba(x_last_dir)[0, 1])
    p_down = 1.0 - p_up

    as_of_date = feat_df.index[-1]

    forecast = NextDayForecast(
        ticker=ticker,
        as_of_date=as_of_date,
        high_vol_prob=p_high_vol,
        up_prob=p_up,
        down_prob=p_down,
    )

    return forecast, feat_df


def main() -> None:
    """
    CLI קטן:
    python -m financial_ews.prediction.next_day -t AAPL
    """
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Next-day volatility & direction forecast for a given ticker."
    )
    parser.add_argument(
        "-t", "--ticker", type=str, default="SPY", help="Ticker symbol, e.g. AAPL, TSLA, SPY"
    )
    parser.add_argument(
        "--start", type=str, default="2015-01-01", help="Start date for history (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Absolute next-day return threshold for 'high volatility' (e.g. 0.02 = 2%)",
    )

    args = parser.parse_args()

    print(f"Downloading data and training models for {args.ticker}...")
    forecast, feat_df = forecast_next_day_for_ticker(
        ticker=args.ticker,
        start=args.start,
        threshold=args.threshold,
    )

    print("\n=== Next-day forecast ===")
    print(f"As of date: {forecast.as_of_date.date()}  (last trading day in data)")
    print(f"Ticker    : {forecast.ticker}")
    print(f"High-volatility probability : {forecast.high_vol_prob * 100:.1f}%")
    print(f"Up move probability         : {forecast.up_prob * 100:.1f}%")
    print(f"Down/flat probability       : {forecast.down_prob * 100:.1f}%")

    # --- גרף יפה של המחיר ב־90 הימים האחרונים ---
    last_n = 90
    recent = feat_df.tail(last_n)

    if "Adj Close" in recent.columns:
        price_series = recent["Adj Close"]
    else:
        # אם מסיבה כלשהי עמודה חסרה – נוותר בשקט
        price_series = None

    if price_series is not None:
        plt.figure(figsize=(10, 4))
        price_series.plot(
            title=f"{forecast.ticker} - last {last_n} days (as of {forecast.as_of_date.date()})"
        )
        plt.xlabel("Date")
        plt.ylabel("Adj Close Price")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

