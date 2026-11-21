"""
Portfolio Manager for Risk Ghost.

Orchestrates:
- Portfolio CSV parsing (flexible columns, direct tickers, no hardcoded allowlist).
- Online market data retrieval & ticker validation.
- Base currency conversion & date alignment across international listings.
- End-to-end portfolio diagnostics assembly.
"""

import csv
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from src.data_provider import (
        InstrumentMetadata,
        fetch_historical_prices_and_volumes,
        fetch_fx_history,
        MarketDataError,
        TickerNotFoundError,
        UnsupportedInstrumentError,
    )
    from src.analytics import (
        calculate_risk_contribution,
        calculate_component_risk_contributions,
        calculate_market_impact,
        rebalance_to_equal_risk,
        evaluate_rebalanced_portfolio,
        calculate_max_drawdown,
        calculate_portfolio_timeseries,
        run_monte_carlo,
    )
except ImportError:
    from data_provider import (
        InstrumentMetadata,
        fetch_historical_prices_and_volumes,
        fetch_fx_history,
        MarketDataError,
        TickerNotFoundError,
        UnsupportedInstrumentError,
    )
    from analytics import (
        calculate_risk_contribution,
        calculate_component_risk_contributions,
        calculate_market_impact,
        rebalance_to_equal_risk,
        evaluate_rebalanced_portfolio,
        calculate_max_drawdown,
        calculate_portfolio_timeseries,
        run_monte_carlo,
    )


@dataclass
class HoldingItem:
    identifier: str
    shares: float
    ticker: str = ""
    name: str = ""
    currency: str = "USD"
    latest_local_price: float = 0.0
    latest_base_price: float = 0.0
    position_value_base: float = 0.0
    weight_pct: float = 0.0
    risk_contribution_pct: float = 0.0
    marginal_risk_contribution: float = 0.0
    component_risk_contribution: float = 0.0
    daily_volatility_pct: float = 0.0
    adv_shares: float = 0.0
    days_to_exit: float = 0.0
    slippage_pct: float = 0.0
    suggested_weight_pct: float = 0.0
    projected_risk_contribution_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    risk_flag: str = "Normal"


@dataclass
class PortfolioAnalysisResult:
    base_currency: str
    total_valuation_base: float
    daily_portfolio_volatility_pct: float
    annualized_portfolio_volatility_pct: float
    holdings: List[HoldingItem]
    top_two_risk_concentration_pct: float
    top_two_tickers: List[str]
    concentration_tier: str
    archegos_alert: bool
    knight_capital_alert: bool
    rebalanced_annualized_volatility_pct: float
    rebalance_volatility_delta_pct: float
    fetch_timestamp: str
    base_prices_df: pd.DataFrame
    returns_df: pd.DataFrame
    cov_matrix: pd.DataFrame
    correlation_matrix: pd.DataFrame
    portfolio_value_history: pd.Series
    is_multi_currency: bool
    fx_pairs_used: List[str] = field(default_factory=list)


def parse_portfolio_csv(filepath_or_buffer) -> List[Tuple[str, float]]:
    """
    Parses portfolio CSV files with flexible headers.
    Accepts:
      - 'Stock names,No. of shares holds' (CS50P format)
      - 'Ticker,Shares' / 'Symbol,Quantity'
      - Raw 2-column CSV without headers
    Returns list of (identifier, shares).
    """
    holdings_raw: List[Tuple[str, float]] = []

    if isinstance(filepath_or_buffer, str):
        with open(filepath_or_buffer, mode="r", encoding="utf-8-sig") as f:
            reader = list(csv.reader(f))
    else:
        # Buffer (e.g. from Streamlit UploadedFile)
        content = filepath_or_buffer.read()
        if isinstance(content, bytes):
            text = content.decode("utf-8-sig", errors="replace")
        else:
            text = str(content)
        reader = list(csv.reader(text.splitlines()))

    if not reader:
        raise ValueError("Portfolio CSV file is empty.")

    start_row = 0
    first_row = [c.strip().lower() for c in reader[0]]

    # Detect header presence
    has_header = any(keyword in col for col in first_row for keyword in ("stock", "ticker", "symbol", "name", "share", "qty", "quantity"))
    if has_header:
        start_row = 1

    for row_idx, row in enumerate(reader[start_row:], start=start_row + 1):
        if not row or len(row) < 2:
            continue
        raw_id = row[0].strip()
        raw_qty = row[1].strip().replace(",", "")
        if not raw_id:
            continue
        try:
            qty = float(raw_qty)
            if qty < 0:
                raise ValueError(f"Line {row_idx}: Share quantity cannot be negative ({qty}).")
            holdings_raw.append((raw_id, qty))
        except ValueError as e:
            raise ValueError(f"Line {row_idx}: Invalid quantity '{raw_qty}' for '{raw_id}'.") from e

    if not holdings_raw:
        raise ValueError("No valid holdings found in portfolio file.")

    return holdings_raw


def analyze_portfolio(
    raw_holdings: List[Tuple[str, float]],
    base_currency: str = "USD",
    period: str = "60d",
) -> PortfolioAnalysisResult:
    """
    Full portfolio diagnostic pipeline:
    1. Validates tickers online.
    2. Downloads historical price and volume data.
    3. Handles multi-currency FX conversions to base currency.
    4. Computes Euler risk contributions, liquidity metrics, drawdowns, and correlation.
    """
    base_curr = base_currency.strip().upper()
    tickers = [item[0].strip().upper() for item in raw_holdings]
    shares = [item[1] for item in raw_holdings]

    if len(tickers) == 0:
        raise ValueError("Portfolio must contain at least one holding.")

    # 1. Fetch market data online
    close_df, volume_df, metadata_map, fetch_time = fetch_historical_prices_and_volumes(
        tickers=tickers,
        period=period,
    )

    # 2. Currency check & FX conversion
    currencies = {t: metadata_map[t].standard_currency for t in tickers}
    unique_currencies = set(currencies.values())
    is_multi_currency = (len(unique_currencies) > 1) or any(c != base_curr for c in unique_currencies)

    base_prices_df = close_df.copy()
    fx_pairs_used = []

    for t in tickers:
        native_curr = currencies[t]
        if native_curr != base_curr:
            fx_rate_series = fetch_fx_history(
                from_currency=native_curr,
                to_currency=base_curr,
                period=period,
                target_index=close_df.index,
            )
            # Convert close price series into base currency
            base_prices_df[t] = close_df[t] * fx_rate_series
            fx_pairs_used.append(f"{native_curr}->{base_curr}")

    # 3. Portfolio calculations in base currency
    latest_local_prices = [float(close_df[t].iloc[-1]) for t in tickers]
    latest_base_prices = [float(base_prices_df[t].iloc[-1]) for t in tickers]
    position_values_base = [s * p for s, p in zip(shares, latest_base_prices)]
    total_val = sum(position_values_base)

    if total_val <= 0.0:
        weights = np.array([1.0 / len(tickers)] * len(tickers))
    else:
        weights = np.array([v / total_val for v in position_values_base])

    # Time series, daily returns, and covariance in base currency
    port_val_series, returns_df, cov_matrix = calculate_portfolio_timeseries(base_prices_df, shares)
    cov_arr = cov_matrix.values
    corr_matrix = returns_df.corr()

    # Asset daily volatilities (in base currency)
    daily_vols = returns_df.std().values

    # Euler risk decomposition
    port_vol, mrc, crc = calculate_component_risk_contributions(weights, cov_arr)
    rc_pct = calculate_risk_contribution(weights, cov_arr)

    # Bounded illustrative risk-balancing heuristic tilt
    suggested_weights = rebalance_to_equal_risk(rc_pct, weights, max_weight=0.30, min_weight=0.01)

    # Recalculate portfolio volatility and post-rebalance risk contributions
    new_port_vol, new_crc, new_rc_pct = evaluate_rebalanced_portfolio(suggested_weights, cov_arr)
    annualized_vol = port_vol * np.sqrt(252) * 100.0
    new_annualized_vol = new_port_vol * np.sqrt(252) * 100.0
    vol_delta = new_annualized_vol - annualized_vol

    # Average daily volume in native share units
    adv_values = [float(volume_df[t].mean()) if t in volume_df else 0.0 for t in tickers]

    # Liquidity impact and drawdowns
    holding_items: List[HoldingItem] = []
    knight_capital_alert = False

    for i, t in enumerate(tickers):
        slip, days = calculate_market_impact(
            shares_owned=shares[i],
            adv=adv_values[i],
            volatility=daily_vols[i],
        )
        if days >= 5.0 or slip >= 3.0:
            knight_capital_alert = True

        mdd = calculate_max_drawdown(base_prices_df[t].values)
        w_p = weights[i] * 100.0
        r_p = rc_pct[i]
        diff = r_p - w_p

        if r_p < 0:
            flag = "Hedge / Risk Reducer"
        elif diff >= 10.0:
            flag = "High Risk Contributor"
        elif diff <= -5.0:
            flag = "Low Risk Contributor"
        else:
            flag = "Balanced"

        holding_items.append(
            HoldingItem(
                identifier=raw_holdings[i][0],
                shares=shares[i],
                ticker=t,
                name=metadata_map[t].name,
                currency=currencies[t],
                latest_local_price=latest_local_prices[i],
                latest_base_price=latest_base_prices[i],
                position_value_base=position_values_base[i],
                weight_pct=w_p,
                risk_contribution_pct=r_p,
                marginal_risk_contribution=mrc[i],
                component_risk_contribution=crc[i],
                daily_volatility_pct=daily_vols[i] * 100.0,
                adv_shares=adv_values[i],
                days_to_exit=days,
                slippage_pct=slip,
                suggested_weight_pct=suggested_weights[i] * 100.0,
                projected_risk_contribution_pct=float(new_rc_pct[i]),
                max_drawdown_pct=mdd,
                risk_flag=flag,
            )
        )

    # Top two risk concentration analysis
    sorted_indices = np.argsort(rc_pct)[::-1]
    top_two_indices = sorted_indices[:2] if len(rc_pct) >= 2 else sorted_indices[:1]
    top_two_rc = float(sum(rc_pct[i] for i in top_two_indices))
    top_two_tickers = [tickers[i] for i in top_two_indices]

    archegos_alert = top_two_rc >= 60.0
    if archegos_alert:
        conc_tier = "Critical Concentration (Archegos Alert)"
    elif top_two_rc >= 40.0:
        conc_tier = "Elevated Concentration"
    else:
        conc_tier = "Diversified Risk Distribution"

    daily_port_vol_pct = port_vol * 100.0

    return PortfolioAnalysisResult(
        base_currency=base_curr,
        total_valuation_base=total_val,
        daily_portfolio_volatility_pct=daily_port_vol_pct,
        annualized_portfolio_volatility_pct=annualized_vol,
        holdings=holding_items,
        top_two_risk_concentration_pct=top_two_rc,
        top_two_tickers=top_two_tickers,
        concentration_tier=conc_tier,
        archegos_alert=archegos_alert,
        knight_capital_alert=knight_capital_alert,
        rebalanced_annualized_volatility_pct=new_annualized_vol,
        rebalance_volatility_delta_pct=vol_delta,
        fetch_timestamp=fetch_time,
        base_prices_df=base_prices_df,
        returns_df=returns_df,
        cov_matrix=cov_matrix,
        correlation_matrix=corr_matrix,
        portfolio_value_history=port_val_series,
        is_multi_currency=is_multi_currency,
        fx_pairs_used=list(set(fx_pairs_used)),
    )

