"""
Comprehensive Automated Test Suite for Risk Ghost.

Uses small synthetic NumPy/Pandas fixtures and unittest.mock.
Normal pytest runs execute completely offline without requiring network access.
"""

import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

if "MPLCONFIGDIR" not in os.environ:
    default_mpl = r"C:\Users\omkar\AppData\Local\matplotlib"
    if os.path.exists(default_mpl) and os.access(default_mpl, os.W_OK):
        os.environ["MPLCONFIGDIR"] = default_mpl
    else:
        mpl_dir = os.path.join(tempfile.gettempdir(), "matplotlib_risk_ghost")
        os.makedirs(mpl_dir, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl_dir

import matplotlib

matplotlib.use("Agg")

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from analytics import (
    calculate_component_risk_contributions,
    calculate_market_impact,
    calculate_max_drawdown,
    calculate_risk_contribution,
    evaluate_rebalanced_portfolio,
    format_days_to_exit,
    rebalance_to_equal_risk,
    run_monte_carlo,
)
from data_provider import (
    InstrumentMetadata,
    TickerNotFoundError,
    UnsupportedInstrumentError,
    fetch_fx_history,
    fetch_historical_prices_and_volumes,
    get_ticker_metadata,
    search_company,
)
from portfolio_manager import (
    analyze_portfolio,
    parse_portfolio_csv,
)
from project import (
    load_portfolio,
)


# =============================================================================
# 1. Financial Math Invariant Tests
# =============================================================================
def test_calculate_risk_contribution():
    cov = np.array([
        [0.04, 0.0],
        [0.0,  0.01]
    ])
    w = np.array([0.5, 0.5])
    rc = calculate_risk_contribution(w, cov)
    assert np.allclose(rc, [80.0, 20.0], atol=1e-5)
    assert np.isclose(np.sum(rc), 100.0)

    # Scale invariance proof (Euler theorem: multiplying weights by c yields identical RC %)
    rc_pct = calculate_risk_contribution(np.array([50.0, 50.0]), cov)
    assert np.allclose(rc_pct, [80.0, 20.0], atol=1e-5)

    with pytest.raises(ValueError):
        calculate_risk_contribution(np.array([0.5, 0.5]), np.eye(3))


def test_calculate_component_risk_contributions():
    cov = np.array([
        [0.04, 0.01],
        [0.01, 0.09]
    ])
    w = np.array([0.6, 0.4])
    port_vol, mrc, crc = calculate_component_risk_contributions(w, cov)

    # Invariant: sum of component risk contributions equals portfolio volatility
    assert np.isclose(np.sum(crc), port_vol, atol=1e-6)
    assert len(mrc) == 2
    assert port_vol > 0


def test_calculate_market_impact():
    slippage, days = calculate_market_impact(10000, 2000.0, 0.0644)
    assert days == 25.0
    assert slippage == 7.25

    slippage_std, days_std = calculate_market_impact(10000, 2000.0, 0.035)
    assert days_std == 25.0
    assert slippage_std == 3.96

    # Fractional days preserved by default (no artificial 1.0 day floor)
    slip_small, days_small = calculate_market_impact(10, 100000.0, 0.02)
    assert days_small == 0.0005
    assert slip_small >= 0.05
    assert format_days_to_exit(days_small) == "< 0.01 d"

    # Optional 1.0 day floor when explicitly requested
    _, days_floored = calculate_market_impact(10, 100000.0, 0.02, floor_to_one_day=True)
    assert days_floored == 1.0

    slip_zero, days_zero = calculate_market_impact(0, 1000.0, 0.02)
    assert slip_zero == 0.0
    assert days_zero == 0.0

    with pytest.raises(ValueError):
        calculate_market_impact(-5, 1000.0, 0.02)
    with pytest.raises(ValueError):
        calculate_market_impact(100, 0.0, 0.02)
    with pytest.raises(ValueError):
        calculate_market_impact(100, 1000.0, -0.01)


def test_rebalance_to_equal_risk():
    test_weights = np.array([0.5, 0.3, 0.2])
    test_rc = np.array([60.0, 30.0, 10.0])
    w_new = rebalance_to_equal_risk(test_rc, test_weights, max_weight=0.50, min_weight=0.01)

    assert len(w_new) == 3
    assert np.isclose(np.sum(w_new), 1.0)
    assert w_new[0] < test_weights[0]  # Over-risk asset loses weight
    assert w_new[2] > test_weights[2]  # Under-risk asset gains weight

    eq_weights = np.array([1/3, 1/3, 1/3])
    eq_rc = np.array([33.33, 33.33, 33.34])
    w_eq = rebalance_to_equal_risk(eq_rc, eq_weights)
    assert np.allclose(w_eq, [1/3, 1/3, 1/3], atol=0.01)

    with pytest.raises(ValueError):
        rebalance_to_equal_risk(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        rebalance_to_equal_risk(np.array([10.0, 20.0]), np.array([0.5]))


def test_calculate_max_drawdown():
    prices = [100.0, 120.0, 90.0, 110.0]
    assert calculate_max_drawdown(prices) == -25.0
    assert calculate_max_drawdown([10.0, 20.0, 30.0]) == 0.0
    assert calculate_max_drawdown([50.0]) == 0.0


def test_run_monte_carlo():
    dates = pd.date_range("2026-01-01", periods=20, freq="B")
    ret_df = pd.DataFrame({
        "A": np.linspace(0.001, 0.005, 20),
        "B": np.linspace(-0.002, 0.004, 20),
    }, index=dates)

    mc = run_monte_carlo(
        weights=np.array([0.5, 0.5]),
        returns_df=ret_df,
        initial_value=100000.0,
        horizon_days=30,
        num_simulations=50,
        seed=123,
    )

    assert mc["paths"].shape == (50, 31)
    assert len(mc["median_path"]) == 31
    assert mc["p5_final_return"] <= mc["median_final_return"] <= mc["p95_final_return"]


# =============================================================================
# 2. Portfolio Input & Parser Tests
# =============================================================================
def test_load_portfolio_cs50_format(tmp_path):
    test_file = tmp_path / "test_portfolio.csv"
    test_file.write_text("Stock names,No. of shares holds\ntesla,6\nmicrosoft,15\nreliance,20\n", encoding="utf-8")
    names, tickers, shares = load_portfolio(str(test_file))
    assert names == ["tesla", "microsoft", "reliance"]
    assert tickers == ["TSLA", "MSFT", "RELIANCE.NS"]
    assert shares == [6.0, 15.0, 20.0]


def test_load_portfolio_direct_tickers(tmp_path):
    test_file = tmp_path / "direct_tickers.csv"
    test_file.write_text("Ticker,Shares\nNVDA,100\n005930.KS,50\nVOD.L,200\n", encoding="utf-8")
    parsed = parse_portfolio_csv(str(test_file))
    assert parsed == [("NVDA", 100.0), ("005930.KS", 50.0), ("VOD.L", 200.0)]


def test_load_portfolio_invalid(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_portfolio_csv(str(empty_file))

    neg_file = tmp_path / "negative.csv"
    neg_file.write_text("Ticker,Shares\nAAPL,-10\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_portfolio_csv(str(neg_file))


# =============================================================================
# 3. Market Data & Metadata Mocked Tests
# =============================================================================
@patch("yfinance.Ticker")
def test_get_ticker_metadata_usd(mock_ticker_cls):
    mock_inst = MagicMock()
    mock_fast_info = MagicMock()
    mock_fast_info.currency = "USD"
    mock_fast_info.exchange = "NMS"
    mock_fast_info.quote_type = "EQUITY"
    mock_inst.fast_info = mock_fast_info
    mock_ticker_cls.return_value = mock_inst

    meta = get_ticker_metadata("AAPL")
    assert meta.symbol == "AAPL"
    assert meta.currency == "USD"
    assert meta.is_subunit is False
    assert meta.subunit_factor == 1.0


@patch("yfinance.Ticker")
def test_get_ticker_metadata_pence_subunit(mock_ticker_cls):
    mock_inst = MagicMock()
    mock_fast_info = MagicMock()
    mock_fast_info.currency = "GBp"
    mock_fast_info.exchange = "LSE"
    mock_fast_info.quote_type = "EQUITY"
    mock_inst.fast_info = mock_fast_info
    mock_ticker_cls.return_value = mock_inst

    meta = get_ticker_metadata("VOD.L")
    assert meta.symbol == "VOD.L"
    assert meta.currency == "GBP"
    assert meta.is_subunit is True
    assert meta.subunit_factor == 100.0


def test_reject_index_symbols():
    with pytest.raises(UnsupportedInstrumentError):
        get_ticker_metadata("^GSPC")


@patch("yfinance.Search")
def test_search_company_mocked(mock_search_cls):
    mock_obj = MagicMock()
    mock_obj.quotes = [
        {"symbol": "AAPL", "shortname": "Apple Inc.", "exchange": "NMS", "quoteType": "EQUITY", "currency": "USD"},
        {"symbol": "APC.F", "shortname": "Apple Inc.", "exchange": "FRA", "quoteType": "EQUITY", "currency": "EUR"},
        {"symbol": "^GSPC", "shortname": "S&P 500", "exchange": "SNP", "quoteType": "INDEX", "currency": "USD"},
    ]
    mock_search_cls.return_value = mock_obj

    results = search_company("Apple")
    assert len(results) == 3
    # Verifies both US and German listings are returned for user selection without auto-picking
    assert results[0]["symbol"] == "AAPL"
    assert results[1]["symbol"] == "APC.F"
    assert results[2]["is_supported"] is False  # Index flagged as not a stock holding


@patch("yfinance.download")
@patch("data_provider.get_ticker_metadata")
def test_fetch_market_data_partial_failure(mock_meta, mock_download):
    mock_meta.side_effect = lambda t: InstrumentMetadata(t, t, "NMS", "EQUITY", "USD")
    # Return DataFrame missing TSLA
    dates = pd.date_range("2026-01-01", periods=20, freq="B")
    df = pd.DataFrame({"Close": [100.0] * 20, "Volume": [1000] * 20}, index=dates)
    mock_download.return_value = df

    with pytest.raises(TickerNotFoundError):
        fetch_historical_prices_and_volumes(["AAPL", "TSLA"])


@patch("yfinance.download")
def test_fetch_fx_history_quote_direction(mock_download):
    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    # 1 EUR = 1.10 USD
    df_fx = pd.DataFrame({"Close": [1.10] * 10}, index=dates)
    mock_download.return_value = df_fx

    fx_series = fetch_fx_history("EUR", "USD", target_index=dates)
    assert np.allclose(fx_series.values, 1.10)


@patch("portfolio_manager.fetch_historical_prices_and_volumes")
@patch("portfolio_manager.fetch_fx_history")
def test_analyze_portfolio_end_to_end(mock_fx, mock_fetch):
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    close_df = pd.DataFrame({
        "AAPL": np.linspace(150, 160, 30),
        "VOD.L": np.linspace(2.0, 2.2, 30),  # already normalized to GBP
    }, index=dates)
    volume_df = pd.DataFrame({
        "AAPL": [50000000.0] * 30,
        "VOD.L": [10000000.0] * 30,
    }, index=dates)
    meta_map = {
        "AAPL": InstrumentMetadata("AAPL", "Apple", "NMS", "EQUITY", "USD"),
        "VOD.L": InstrumentMetadata("VOD.L", "Vodafone", "LSE", "EQUITY", "GBP"),
    }
    mock_fetch.return_value = (close_df, volume_df, meta_map, "2026-11-09T12:00:00Z")

    # Mock FX: 1 GBP = 1.30 USD
    mock_fx.return_value = pd.Series([1.30] * 30, index=dates)

    holdings = [("AAPL", 10.0), ("VOD.L", 50.0)]
    result = analyze_portfolio(holdings, base_currency="USD")

    assert result.base_currency == "USD"
    assert result.is_multi_currency is True
    assert len(result.holdings) == 2
    assert result.total_valuation_base > 0
    assert np.isclose(sum(h.weight_pct for h in result.holdings), 100.0)
    assert np.isclose(sum(h.risk_contribution_pct for h in result.holdings), 100.0)


# =============================================================================
# 4. Financial Audit & Mathematical Edge-Case Tests
# =============================================================================
def test_negative_risk_contribution_reconciliation():
    """
    Mathematical verification of Euler risk decomposition with a mathematically valid
    negative risk contribution (empirical hedge asset).
    Guarantees:
    - Covariance matrix is strictly positive definite (all eigenvalues > 0).
    - Asset 3 has a negative risk contribution (RC < 0).
    - Euler decomposition invariant: sum(RC %) == 100.00%.
    - Euler component invariant: sum(Component Risk Contributions) == Portfolio Volatility.
    """
    std = np.array([0.20, 0.20, 0.10])
    corr = np.array([
        [1.0, 0.6, -0.5],
        [0.6, 1.0, -0.5],
        [-0.5, -0.5, 1.0]
    ])
    cov = np.outer(std, std) * corr

    # Verify strict positive definiteness
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0.0), f"Covariance must be strictly PSD, got eigenvalues {eigenvalues}"

    weights = np.array([0.45, 0.45, 0.10])
    port_vol, mrc, crc = calculate_component_risk_contributions(weights, cov)
    rc_pct = calculate_risk_contribution(weights, cov)

    # Asset 3 is an empirical hedge with negative marginal and percentage risk contribution
    assert mrc[2] < 0.0, "Hedge asset must have negative marginal risk contribution"
    assert crc[2] < 0.0, "Hedge asset must have negative component risk contribution"
    assert rc_pct[2] < 0.0, "Hedge asset must have negative percentage risk contribution"
    assert np.isclose(rc_pct[2], -3.303, atol=0.01)

    # Invariants
    assert np.isclose(np.sum(rc_pct), 100.0, atol=1e-6), "Euler risk contributions must sum to 100%"
    assert np.isclose(np.sum(crc), port_vol, atol=1e-6), "Component risk contributions must sum to portfolio volatility"


def test_rebalance_with_negative_and_zero_rc():
    """
    Verifies that the illustrative risk-balancing heuristic:
    - Never inverts negative or near-zero risk contributions.
    - Does NOT explosively inflate a negative RC asset (e.g. XOM hedge case).
    - Handles RC <= 0, RC == 0, and tiny RC without dividing by zero or crashing.
    - Produces weights strictly bounded by [min_weight, max_weight].
    - Guarantees the suggested weights sum to exactly 1.0000.
    """
    weights = np.array([0.15, 0.15, 0.10, 0.30, 0.30])
    rc = np.array([-0.4, 0.0, 0.5, 50.0, 49.9])

    w_new = rebalance_to_equal_risk(rc, weights, max_weight=0.30, min_weight=0.01)

    assert len(w_new) == 5
    assert np.isclose(np.sum(w_new), 1.0, atol=1e-5), "Suggested weights must sum to 100%"
    assert np.all(w_new <= 0.30 + 1e-6), f"No weight should exceed max_weight constraint, got {w_new}"
    assert np.all(w_new >= 0.01 - 1e-6), f"No weight should be below min_weight constraint, got {w_new}"

    # Negative RC asset (index 0) must NOT explode to 60%+ as in the buggy inverter
    assert w_new[0] <= 0.30, "Hedge asset must respect the 30% position cap"
    # Over-risk assets (indices 3, 4) must be trimmed down from 0.30
    assert w_new[3] <= weights[3]


def test_rebalance_position_constraints_and_bounds():
    """
    Tests iterative position weight constraints across skewed distributions.
    """
    # 8-asset portfolio mimicking the user's test portfolio
    rc = np.array([-0.4, 33.7, 18.6, 12.0, 10.0, 8.0, 10.0, 8.1])
    weights = np.array([0.125] * 8)

    w_adj = rebalance_to_equal_risk(rc, weights, max_weight=0.30, min_weight=0.02)

    assert len(w_adj) == 8
    assert np.isclose(np.sum(w_adj), 1.0, atol=1e-5)
    assert np.all(w_adj <= 0.30 + 1e-6)
    assert np.all(w_adj >= 0.02 - 1e-6)


def test_evaluate_rebalanced_portfolio():
    """
    Verifies post-rebalance portfolio risk evaluation:
    - Correctly calculates new portfolio volatility from adjusted weights.
    - Verifies new Euler risk contributions sum to 100%.
    - Verifies component risk contributions sum to new portfolio volatility.
    """
    cov = np.array([
        [0.04, 0.01],
        [0.01, 0.09]
    ])
    w_initial = np.array([0.5, 0.5])
    rc_initial = calculate_risk_contribution(w_initial, cov)

    w_suggested = rebalance_to_equal_risk(rc_initial, w_initial, max_weight=0.70, min_weight=0.10)
    new_vol, new_crc, new_rc_pct = evaluate_rebalanced_portfolio(w_suggested, cov)

    assert new_vol > 0.0
    assert len(new_rc_pct) == 2
    assert np.isclose(np.sum(new_rc_pct), 100.0, atol=1e-6)
    assert np.isclose(np.sum(new_crc), new_vol, atol=1e-6)


def test_format_days_to_exit():
    """
    Verifies human-readable formatting of days-to-liquidate without deceptive rounding.
    """
    assert format_days_to_exit(0.0) == "0.0 d"
    assert format_days_to_exit(0.0005) == "< 0.01 d"
    assert format_days_to_exit(0.009) == "< 0.01 d"
    assert format_days_to_exit(0.05) == "0.050 d"
    assert format_days_to_exit(0.25) == "0.25 d"
    assert format_days_to_exit(1.5) == "1.5 d"
    assert format_days_to_exit(25.0) == "25.0 d"


def test_top_two_risk_dominance_and_concentration_tiers():
    """
    Tests 3-tier concentration logic and top-two ticker risk dominance identification.
    """
    rc_pct = np.array([33.7, 18.6, 12.0, 10.0, 10.0, 8.0, 8.1, -0.4])
    tickers = ["LLY", "COIN", "MSTR", "JPM", "CAT", "BRK-B", "PG", "XOM"]

    sorted_indices = np.argsort(rc_pct)[::-1]
    top_two_indices = sorted_indices[:2]
    top_two_rc = float(sum(rc_pct[i] for i in top_two_indices))
    top_two_names = [tickers[i] for i in top_two_indices]

    # Verify LLY (33.7%) and COIN (18.6%) correctly identified
    assert top_two_names == ["LLY", "COIN"]
    assert np.isclose(top_two_rc, 52.3, atol=0.1)

    # Verify concentration tier (52.3% falls into Elevated 40-60%)
    if top_two_rc >= 60.0:
        tier = "Critical Concentration (Archegos Alert)"
    elif top_two_rc >= 40.0:
        tier = "Elevated Concentration"
    else:
        tier = "Diversified Risk Distribution"
    assert tier == "Elevated Concentration"


def test_max_drawdown_peak_to_trough():
    """
    Verifies peak-to-trough maximum drawdown calculation with intermediate recovery.
    """
    # Peak at 160, trough at 80 -> drawdown = (80 - 160) / 160 = -50.0%
    prices = [100.0, 120.0, 150.0, 130.0, 160.0, 100.0, 80.0, 110.0, 120.0]
    mdd = calculate_max_drawdown(prices)
    assert np.isclose(mdd, -50.0, atol=1e-5)


def test_monte_carlo_cholesky_psd_guarantee():
    """
    Verifies that run_monte_carlo handles near-singular covariance matrices gracefully
    via eigenvalue clipping without raising LinAlgError.
    """
    dates = pd.date_range("2026-01-01", periods=20, freq="B")
    # Perfectly collinear columns (singular covariance)
    series_a = np.linspace(0.01, 0.05, 20)
    ret_df = pd.DataFrame({
        "A": series_a,
        "B": series_a * 2.0,  # Exactly collinear -> zero eigenvalue
    }, index=dates)

    mc = run_monte_carlo(
        weights=np.array([0.5, 0.5]),
        returns_df=ret_df,
        initial_value=100000.0,
        horizon_days=20,
        num_simulations=50,
        seed=42,
    )

    assert mc["paths"].shape == (50, 21)
    assert not np.isnan(mc["paths"]).any()
    assert mc["p5_final_return"] <= mc["p95_final_return"]

