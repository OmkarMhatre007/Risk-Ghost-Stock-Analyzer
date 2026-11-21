"""
Risk Ghost Stock Analyzer: Core Quantitative Engine Package.
"""

from src.analytics import (
    calculate_component_risk_contributions,
    calculate_market_impact,
    calculate_max_drawdown,
    calculate_portfolio_timeseries,
    calculate_risk_contribution,
    evaluate_rebalanced_portfolio,
    format_days_to_exit,
    rebalance_to_equal_risk,
    run_monte_carlo,
)
from src.charts import (
    plot_capital_vs_risk,
    plot_correlation_heatmap,
    plot_drawdown_history,
    plot_liquidity_matrix,
    plot_monte_carlo,
)
from src.data_provider import (
    FXConversionError,
    InsufficientDataError,
    InstrumentMetadata,
    MarketDataError,
    RateLimitError,
    TickerNotFoundError,
    UnsupportedInstrumentError,
    fetch_fx_history,
    fetch_historical_prices_and_volumes,
    get_ticker_metadata,
    search_company,
)
from src.portfolio_manager import (
    HoldingItem,
    PortfolioAnalysisResult,
    analyze_portfolio,
    calculate_portfolio_timeseries,
    parse_portfolio_csv,
)

__all__ = [
    "calculate_component_risk_contributions",
    "calculate_market_impact",
    "calculate_max_drawdown",
    "calculate_portfolio_timeseries",
    "calculate_risk_contribution",
    "evaluate_rebalanced_portfolio",
    "format_days_to_exit",
    "rebalance_to_equal_risk",
    "run_monte_carlo",
    "plot_capital_vs_risk",
    "plot_correlation_heatmap",
    "plot_drawdown_history",
    "plot_liquidity_matrix",
    "plot_monte_carlo",
    "FXConversionError",
    "InsufficientDataError",
    "InstrumentMetadata",
    "MarketDataError",
    "RateLimitError",
    "TickerNotFoundError",
    "UnsupportedInstrumentError",
    "fetch_fx_history",
    "fetch_historical_prices_and_volumes",
    "get_ticker_metadata",
    "search_company",
    "HoldingItem",
    "PortfolioAnalysisResult",
    "analyze_portfolio",
    "parse_portfolio_csv",
]
