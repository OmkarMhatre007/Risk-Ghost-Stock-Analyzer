"""
Risk Ghost: Portfolio Risk & Liquidity Diagnostic (CLI Entry Point)

Quantifies latent concentration risk and execution liquidity using Euler's
homogeneous risk decomposition and the Almgren-Chriss square-root impact heuristic.
Uses online market data via yfinance without hardcoded universe restrictions or offline CSV fallbacks.
"""

import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from data_provider import (
    fetch_historical_prices_and_volumes,
)
from portfolio_manager import (
    analyze_portfolio,
    parse_portfolio_csv,
)

# Friendly convenience aliases for common names (not an allowlist or restriction)
CONVENIENCE_ALIASES = {
    "apple": "AAPL",
    "nvidia": "NVDA",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "meta": "META",
    "facebook": "META",
    "amazon": "AMZN",
    "samsung": "005930.KS",
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
}

# Retained as empty dict or alias view for backward compatibility with external imports
STOCK_MAP = CONVENIENCE_ALIASES


def load_portfolio(filepath: str = "portfolio.csv") -> tuple[list[str], list[str], list[float]]:
    """
    Loads portfolio holdings from CSV.
    Accepts direct tickers or common company names.
    Does NOT restrict instruments to a hardcoded universe.
    Returns:
        stock_names: List of input identifiers
        tickers: List of resolved Yahoo Finance ticker symbols
        shares: List of share quantities
    """
    raw_holdings = parse_portfolio_csv(filepath)
    stock_names: list[str] = []
    tickers: list[str] = []
    shares: list[float] = []

    for name_or_ticker, qty in raw_holdings:
        clean_name = name_or_ticker.strip()
        lower_name = clean_name.lower()
        # Check convenience aliases first, otherwise use ticker directly
        ticker_sym = CONVENIENCE_ALIASES.get(lower_name, clean_name.upper())

        stock_names.append(clean_name)
        tickers.append(ticker_sym)
        shares.append(qty)

    return stock_names, tickers, shares


def load_data(tickers: list[str], period: str = "60d") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Online-only market data loader.
    Fetches historical closing prices and share volumes from Yahoo Finance into memory.
    Does not dump data to local CSVs and does not fall back to stale offline CSVs.
    """
    print(f"Fetching online market data for: {', '.join(tickers)}...")
    close_df, volume_df, _, _ = fetch_historical_prices_and_volumes(
        tickers=tickers,
        period=period,
    )
    print("Successfully retrieved online market data.")
    return close_df, volume_df


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("\nRisk Ghost Stock Analyzer: Portfolio Risk & Liquidity Diagnostic (CLI)")
        print("\nUsage:")
        print("  python project.py [PORTFOLIO_CSV] [BASE_CURRENCY]")
        print("\nArguments:")
        print("  PORTFOLIO_CSV   Path to portfolio CSV file (default: portfolio.csv)")
        print("                  Accepts columns 'Ticker,Shares' or 'Stock names,No. of shares holds'")
        print("  BASE_CURRENCY   Base reporting currency (default: USD, e.g. USD, EUR, GBP, INR)")
        print("\nExamples:")
        print("  python project.py")
        print("  python project.py portfolio.csv USD")
        print("  python project.py my_portfolio.csv EUR")
        print("\nSupported Instruments:")
        print("  Any valid Yahoo Finance equity or ETF ticker (e.g. AAPL, NVDA, RELIANCE.NS, VOD.L, 005930.KS)")
        print("  (Note: Example tickers are illustrative and do not form a restricted allowlist)\n")
        sys.exit(0)

    filepath = sys.argv[1] if len(sys.argv) > 1 else "portfolio.csv"
    base_currency = sys.argv[2] if len(sys.argv) > 2 else "USD"

    print("\n" + "=" * 72)
    print("           Risk Ghost Stock Analyzer - Quantitative Finance")
    print("=" * 72)
    print(f"Portfolio File: {filepath}")
    print(f"Base Currency:  {base_currency}")

    try:
        raw_holdings = parse_portfolio_csv(filepath)
    except Exception as e:
        print(f"\n[Error] Unable to load portfolio file '{filepath}': {e}")
        sys.exit(1)

    # Normalize identifiers with convenience aliases if matched, else raw ticker
    normalized_holdings = []
    for ident, qty in raw_holdings:
        sym = CONVENIENCE_ALIASES.get(ident.strip().lower(), ident.strip().upper())
        normalized_holdings.append((sym, qty))

    print(f"\nAnalyzing {len(normalized_holdings)} holdings online...")
    try:
        result = analyze_portfolio(normalized_holdings, base_currency=base_currency)
    except Exception as e:
        print(f"\n[Error] Analysis failed: {e}")
        sys.exit(1)

    print(f"\nMarket Data Timestamp (UTC): {result.fetch_timestamp}")
    if result.is_multi_currency:
        print(f"International FX conversions applied: {', '.join(result.fx_pairs_used)}")
    print(f"Total Valuation: {result.total_valuation_base:,.2f} {result.base_currency}")
    print(
        f"Portfolio Volatility: {result.daily_portfolio_volatility_pct:.2f}% daily "
        f"({result.annualized_portfolio_volatility_pct:.1f}% annualized)"
    )

    # Section 1: Capital Weight vs Risk Contribution
    print("\n1. Capital Weight % vs. Risk Contribution % (Euler Decomposition):")
    print(f"{'Ticker':<12} | {'Currency':<8} | {'Weight %':<9} | {'Risk %':<9} | {'Diagnostic Note'}")
    print("-" * 72)
    for h in result.holdings:
        print(
            f"{h.ticker:<12} | {h.currency:<8} | {h.weight_pct:>8.1f}% | {h.risk_contribution_pct:>8.1f}% | {h.risk_flag}"
        )

    if result.archegos_alert:
        print(f"\n[!] Concentration Alert: Top 2 assets drive {result.top_two_risk_concentration_pct:.1f}% of portfolio risk.")
        print("    High covariance/volatility creates latent fragility despite nominal diversification.")

    # Section 2: Liquidity Diagnostics
    print("\n2. Heuristic Liquidity & Market Impact (Almgren-Chriss Model):")
    print(f"{'Ticker':<12} | {'Shares':<10} | {'ADV (Shares)':<14} | {'Days to Exit':<12} | {'Slippage %'}")
    print("-" * 72)
    for h in result.holdings:
        print(
            f"{h.ticker:<12} | {int(h.shares):>10,d} | {h.adv_shares:>14,.0f} | {h.days_to_exit:>10.1f} d | {h.slippage_pct:>9.2f}%"
        )
    print("   Note: Slippage is an illustrative model estimate (0.05% spread floor + square-root impact), not a broker quote.")

    # Section 3: Risk-Balancing Adjustment
    print("\n3. Illustrative Equal-Risk Adjustment Heuristic:")
    print(f"{'Ticker':<12} | {'Current Wt %':<13} | {'Adjusted Wt %':<15} | {'Daily Vol %'}")
    print("-" * 72)
    for h in result.holdings:
        print(
            f"{h.ticker:<12} | {h.weight_pct:>11.1f}% | {h.suggested_weight_pct:>13.1f}% | {h.daily_volatility_pct:>10.2f}%"
        )
    print("   Note: One-step heuristic adjustment toward equal risk contribution, not an exact non-linear optimizer.")

    # Section 4: Drawdowns
    print("\n4. Historical 60-Day Max Drawdowns (Base Currency):")
    for h in result.holdings:
        print(f"   {h.ticker:<12}: {h.max_drawdown_pct:>6.2f}%")

    print("\n" + "=" * 72 + "\n")


if __name__ == "__main__":
    main()
