"""
Optional Online Smoke Test for Risk Ghost.

Validates that real network requests to Yahoo Finance work for a single liquid ticker.
Skipped automatically during standard offline CI/pytest unless explicitly enabled
via `--online-smoke` flag or environment variable `RUN_ONLINE_SMOKE=1`.
"""

import os
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.data_provider import get_ticker_metadata, fetch_historical_prices_and_volumes, search_company
except ImportError:
    from data_provider import get_ticker_metadata, fetch_historical_prices_and_volumes, search_company


@pytest.mark.skipif(
    os.getenv("RUN_ONLINE_SMOKE") != "1",
    reason="Online smoke tests require live internet access. Set RUN_ONLINE_SMOKE=1 to run.",
)
def test_online_ticker_metadata():
    meta = get_ticker_metadata("AAPL")
    assert meta.symbol == "AAPL"
    assert meta.currency == "USD"
    assert meta.quote_type in ("EQUITY", "ETF")


@pytest.mark.skipif(
    os.getenv("RUN_ONLINE_SMOKE") != "1",
    reason="Online smoke tests require live internet access. Set RUN_ONLINE_SMOKE=1 to run.",
)
def test_online_search_company():
    results = search_company("Apple", max_results=3)
    assert len(results) > 0
    symbols = [r["symbol"] for r in results]
    assert "AAPL" in symbols


@pytest.mark.skipif(
    os.getenv("RUN_ONLINE_SMOKE") != "1",
    reason="Online smoke tests require live internet access. Set RUN_ONLINE_SMOKE=1 to run.",
)
def test_online_fetch_history():
    close_df, volume_df, meta_map, ts = fetch_historical_prices_and_volumes(["AAPL"], period="15d")
    assert not close_df.empty
    assert "AAPL" in close_df.columns
    assert len(close_df) >= 5

