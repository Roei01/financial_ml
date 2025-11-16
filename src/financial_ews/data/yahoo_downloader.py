"""
Utilities for downloading financial time-series data from Yahoo Finance.

This module will be used as the first step in our end-to-end ML pipeline:
- Fetch raw historical prices
- Save them locally for reproducible experiments
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf


def download_price_history(
    ticker: str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    cache_dir: str = "data/raw",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download historical OHLCV data for a single ticker from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        The ticker symbol (e.g. "AAPL", "^VIX", "SPY").
    start : str
        Start date in "YYYY-MM-DD" format.
    end : str | None
        End date in "YYYY-MM-DD" format. If None, uses today's date.
    interval : str
        Sampling interval, e.g. "1d", "1h", "1wk".
    cache_dir : str
        Directory where the downloaded CSV will be stored.
    force_refresh : bool
        If False and a cached file exists, load it from disk instead of downloading again.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by DatetimeIndex with OHLCV columns.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    safe_ticker = ticker.replace("^", "")
    filename = f"{safe_ticker}_{interval}_{start}_{end or 'latest'}.csv"
    file_path = cache_path / filename

    if file_path.exists() and not force_refresh:
        try:
            return pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        except ValueError:
            # If the cached file is corrupted or missing the 'Date' column,
            # re-download the data and overwrite the cache.
            file_path.unlink(missing_ok=True)

    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    data.index.name = "Date"
    data.to_csv(file_path)

    return data


def download_multiple_tickers(
    tickers: List[str],
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    cache_dir: str = "data/raw",
    force_refresh: bool = False,
) -> dict:
    """
    Download historical data for multiple tickers.

    Returns a dict: {ticker: DataFrame}
    """
    results: dict = {}

    for t in tickers:
        df = download_price_history(
            ticker=t,
            start=start,
            end=end,
            interval=interval,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )
        results[t] = df

    return results

