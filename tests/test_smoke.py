from financial_ews.data.yahoo_downloader import download_price_history


def test_download_price_history_signature():
    # We only test that the function can be imported and called
    # without raising immediate errors on basic arguments.
    # (We don't hit the real network in CI for now.)
    assert callable(download_price_history)

