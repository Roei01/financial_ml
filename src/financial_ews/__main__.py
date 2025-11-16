"""
Entry point for simple manual tests.

Example:
    python -m financial_ews
"""

from financial_ews.data.yahoo_downloader import download_price_history


def main() -> None:
    ticker = "SPY"  # S&P 500 ETF as a simple default
    print(f"Downloading historical data for {ticker}...")
    df = download_price_history(ticker=ticker, start="2020-01-01")
    print(df.head())
    print(f"\nDownloaded {len(df)} rows.")


if __name__ == "__main__":
    main()

