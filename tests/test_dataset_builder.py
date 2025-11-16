import pandas as pd

from financial_ews.data.dataset_builder import build_supervised_dataset


def test_build_supervised_dataset_basic():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    prices = pd.Series(range(100, 140), index=dates, name="Adj Close")
    df = pd.DataFrame({"Adj Close": prices})

    X, y = build_supervised_dataset(df, price_col="Adj Close", threshold=0.01)

    assert len(X) == len(y)
    assert len(X) > 0
    assert set(X.columns) == {"return_1d", "vol_5d", "vol_21d"}
    assert y.isin([0, 1]).all()


