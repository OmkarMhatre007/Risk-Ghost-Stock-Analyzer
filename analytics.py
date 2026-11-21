"""
Quantitative Portfolio Analytics & Risk Engine for Risk Ghost.

Implements:
- Euler homogeneous risk decomposition (Marginal, Component, and Percentage Risk Contribution).
- Almgren-Chriss square-root heuristic liquidity analysis (ADV and days to exit).
- Illustrative risk-balancing adjustment heuristic.
- Maximum drawdown calculation.
- Covariance and correlation estimation.
- Covariance-aware multivariate Monte Carlo scenario analysis.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def calculate_risk_contribution(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """
    Computes Percentage Risk Contribution (RC %) using Euler's decomposition.
    
    Mathematical formula:
        sigma_p = sqrt(w.T @ Sigma @ w)
        MRC = (Sigma @ w) / sigma_p
        CRC_i = w_i * MRC_i = w_i * (Sigma @ w)_i / sigma_p
        RC_%_i = (CRC_i / sigma_p) * 100 = (w_i * (Sigma @ w)_i / sigma_p^2) * 100
        
    Invariant:
        sum(RC_%_i) == 100.0% (for positive portfolio variance)
    """
    w = np.array(weights, dtype=float)
    cov = np.array(cov_matrix, dtype=float)

    if w.ndim != 1:
        raise ValueError("Weights must be a 1D array.")
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be a 2D square matrix.")
    if len(w) != cov.shape[0]:
        raise ValueError(f"Dimension mismatch: {len(w)} weights vs {cov.shape[0]}x{cov.shape[1]} covariance matrix.")

    port_var = float(w @ cov @ w)
    if port_var <= 0.0:
        return np.zeros(len(w))

    port_vol = math.sqrt(port_var)
    mrc = (cov @ w) / port_vol
    crc = w * mrc
    rc_pct = (crc / port_vol) * 100.0
    return rc_pct


def calculate_component_risk_contributions(weights: np.ndarray, cov_matrix: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Returns:
        portfolio_vol: Total portfolio volatility sigma_p
        mrc: Marginal Risk Contributions vector
        crc: Component Risk Contributions vector (reconciles sum(crc) == sigma_p)
    """
    w = np.array(weights, dtype=float)
    cov = np.array(cov_matrix, dtype=float)
    port_var = float(w @ cov @ w)
    if port_var <= 0.0:
        return 0.0, np.zeros(len(w)), np.zeros(len(w))

    port_vol = math.sqrt(port_var)
    mrc = (cov @ w) / port_vol
    crc = w * mrc
    return port_vol, mrc, crc


def calculate_market_impact(
    shares_owned: float,
    adv: float,
    volatility: float,
    floor_to_one_day: bool = False,
) -> Tuple[float, float]:
    """
    Computes heuristic liquidity metrics:
    1. Days to Exit: Estimated trading days assuming an institutional 20% ADV volume
       participation cap: Days = Shares / (0.20 * ADV).
       Preserves actual fractional trading days unless floor_to_one_day=True.
    2. Expected Slippage %: Almgren-Chriss square-root market impact heuristic with
       a 0.05% (5 bps) baseline retail bid-ask spread assumption:
       Slippage % = 0.05 + 0.5 * vol_pct * sqrt(Shares / ADV)

    Note: These are educational heuristic estimates, not guaranteed execution broker quotes.
    """
    if shares_owned < 0:
        raise ValueError("Shares owned cannot be negative")
    if shares_owned == 0:
        return 0.0, 0.0
    if adv <= 0:
        raise ValueError("ADV must be positive")
    if volatility < 0:
        raise ValueError("Volatility cannot be negative")

    raw_days = shares_owned / (0.20 * adv)
    days_to_exit = max(1.0, raw_days) if floor_to_one_day else raw_days

    vol_pct = volatility * 100.0 if volatility <= 1.0 else volatility
    slippage_pct = 0.05 + 0.5 * vol_pct * math.sqrt(shares_owned / adv)

    return round(float(slippage_pct), 2), float(days_to_exit)


def format_days_to_exit(days: float) -> str:
    """Formats estimated days to exit for clear UI display without deceptive rounding."""
    if days == 0.0:
        return "0.0 d"
    if days < 0.01:
        return "< 0.01 d"
    if days < 0.1:
        return f"{days:.3f} d"
    if days < 1.0:
        return f"{days:.2f} d"
    return f"{days:.1f} d"


def rebalance_to_equal_risk(
    risk_contributions: np.ndarray,
    weights: np.ndarray,
    max_weight: float = 0.30,
    min_weight: float = 0.01,
) -> np.ndarray:
    """
    Illustrative heuristic risk-balancing adjustment toward equal risk contribution.

    Mathematical Design & Constraints:
    - Never directly inverts negative or near-zero risk contributions.
    - Assets acting as hedges (RC <= 0) or negligible risk contributors (RC < 1%)
      do not receive explosive scaling; their multipliers remain neutral.
    - Assets with excessive risk contribution (RC > Target) receive dampening multipliers.
    - Assets with modest positive risk contribution (0 < RC < Target) receive bounded scaling.
    - Uses a damped square-root ratio: multiplier = clip(sqrt(target_rc / rc_i), 0.5, 2.0).
    - Enforces position bounds: w_min <= w_i <= w_max (with w_max >= 1.05 / N).
    - Iteratively normalizes so sum(w_i) == 1.0000.

    Important: This is an illustrative heuristic adjustment, NOT an exact non-linear equal-risk optimizer.
    """
    rc = np.array(risk_contributions, dtype=float)
    w = np.array(weights, dtype=float)

    if len(rc) == 0 or len(w) == 0:
        raise ValueError("Inputs cannot be empty")
    if len(rc) != len(w):
        raise ValueError("Risk contributions and weights must have matching length.")

    n = len(rc)
    if n == 1:
        return np.array([1.0], dtype=float)

    target_rc = 100.0 / n
    # Ensure position constraints are mathematically feasible for N assets
    effective_max = max(max_weight, 1.05 / n)
    effective_min = min(min_weight, 0.95 / n)

    multipliers = []
    for r_val in rc:
        if r_val <= 0.0:
            # Asset is an empirical variance reducer / hedge.
            # Do NOT invert negative numbers; keep multiplier neutral (1.0).
            multipliers.append(1.0)
        elif r_val < 1.0:
            # Near-zero positive risk: apply modest capped scaling (max 1.5x)
            multipliers.append(1.5)
        else:
            # Damped square-root scaling ratio
            ratio = math.sqrt(target_rc / r_val)
            multipliers.append(float(np.clip(ratio, 0.5, 2.0)))

    # Compute raw target weights
    w_raw = w * np.array(multipliers)
    if np.sum(w_raw) <= 0:
        w_raw = np.ones(n) / n
    else:
        w_raw = w_raw / np.sum(w_raw)

    # Iterative projection onto [effective_min, effective_max] and sum == 1
    w_adj = w_raw.copy()
    for _ in range(25):
        w_clipped = np.clip(w_adj, effective_min, effective_max)
        excess = 1.0 - np.sum(w_clipped)
        if abs(excess) < 1e-7:
            w_adj = w_clipped
            break
        unconstrained = (w_clipped > effective_min + 1e-6) & (w_clipped < effective_max - 1e-6)
        if not np.any(unconstrained):
            w_adj = w_clipped + excess / n
            break
        w_clipped[unconstrained] += excess * (w_clipped[unconstrained] / np.sum(w_clipped[unconstrained]))
        w_adj = w_clipped

    # Guaranteed normalization
    w_final = w_adj / np.sum(w_adj)
    return np.round(w_final, 6)


def evaluate_rebalanced_portfolio(
    suggested_weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluates the resulting portfolio standard deviation and new Euler risk contributions
    under the suggested adjusted weights.
    Returns:
        new_portfolio_volatility (float, daily std)
        new_crc (np.ndarray, component risk contributions)
        new_rc_pct (np.ndarray, percentage risk contributions, sums to 100%)
    """
    w = np.array(suggested_weights, dtype=float)
    cov = np.array(cov_matrix, dtype=float)
    port_var = float(w.T @ cov @ w)
    if port_var <= 0:
        return 0.0, np.zeros_like(w), np.zeros_like(w)

    port_vol = math.sqrt(port_var)
    mrc = (cov @ w) / port_vol
    crc = w * mrc
    rc_pct = (w * (cov @ w)) / port_var * 100.0
    return port_vol, crc, rc_pct


def calculate_max_drawdown(prices: List[float] | np.ndarray | pd.Series) -> float:
    """
    Computes maximum peak-to-trough historical drawdown percentage:
        MDD = min_t (P_t - peak_t) / peak_t * 100%
    """
    p_arr = np.array(prices, dtype=float)
    if len(p_arr) <= 1:
        return 0.0

    peak = p_arr[0]
    max_dd = 0.0
    for p in p_arr:
        if p > peak:
            peak = p
        if peak > 0:
            dd = (p - peak) / peak
            if dd < max_dd:
                max_dd = dd

    return round(float(max_dd) * 100.0, 2)


def calculate_portfolio_timeseries(
    base_prices_df: pd.DataFrame,
    shares: List[float],
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Computes portfolio historical valuation series, daily returns, and covariance matrix.
    base_prices_df: DataFrame of closing prices already converted into the base reporting currency.
    shares: Quantity of shares per instrument.
    Returns:
        portfolio_val_series: Historical portfolio value series in base currency
        returns_df: Asset daily returns DataFrame
        cov_matrix: Covariance matrix of daily returns
    """
    tickers = list(base_prices_df.columns)
    shares_arr = np.array(shares, dtype=float)

    # Asset daily simple returns
    prev = base_prices_df.shift(1)
    returns_df = ((base_prices_df - prev) / prev).dropna()
    cov_matrix = returns_df.cov()

    # Historical portfolio value each day
    val_df = base_prices_df.mul(shares_arr, axis=1)
    port_val_series = val_df.sum(axis=1)

    return port_val_series, returns_df, cov_matrix


def run_monte_carlo(
    weights: np.ndarray,
    returns_df: pd.DataFrame,
    initial_value: float = 100000.0,
    horizon_days: int = 60,
    num_simulations: int = 1000,
    seed: int = 42,
) -> Dict[str, np.ndarray | float]:
    """
    Covariance-aware Monte Carlo scenario analysis using multivariate normal daily returns.
    
    Assumptions:
    - Daily returns follow a stationary multivariate normal distribution N(mu, Sigma).
    - Preserves empirical cross-asset correlations via Cholesky decomposition of Sigma.
    - Constant weights over the projection horizon (no dynamic intraday rebalancing).
    - Statistical percentiles (e.g., 5th and 95th) represent confidence bounds, not hard worst-case limits.
    """
    np.random.seed(seed)
    w = np.array(weights, dtype=float)
    mean_vec = returns_df.mean().values
    cov_mat = returns_df.cov().values

    n_assets = len(w)

    # Cholesky factorization for correlated random shocks
    # Use eigenvalue clipping if covariance is non-positive-definite due to empirical noise
    try:
        L = np.linalg.cholesky(cov_mat)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov_mat)
        vals_clipped = np.maximum(vals, 1e-8)
        cov_psd = vecs @ np.diag(vals_clipped) @ vecs.T
        L = np.linalg.cholesky(cov_psd)

    # Simulated paths: (num_simulations, horizon_days + 1)
    paths = np.zeros((num_simulations, horizon_days + 1))
    paths[:, 0] = initial_value

    for t in range(1, horizon_days + 1):
        # Uncorrelated standard normal draws (num_simulations, n_assets)
        Z = np.random.standard_normal((num_simulations, n_assets))
        # Correlated returns: R = mu + Z @ L.T
        sim_returns = mean_vec + Z @ L.T
        # Portfolio return for each simulation: r_p = sim_returns @ w
        port_r = sim_returns @ w
        paths[:, t] = paths[:, t - 1] * (1.0 + port_r)

    median_path = np.median(paths, axis=0)
    p5_path = np.percentile(paths, 5, axis=0)
    p95_path = np.percentile(paths, 95, axis=0)
    p25_path = np.percentile(paths, 25, axis=0)
    p75_path = np.percentile(paths, 75, axis=0)

    final_values = paths[:, -1]
    final_returns = (final_values - initial_value) / initial_value * 100.0

    return {
        "paths": paths,
        "median_path": median_path,
        "p5_path": p5_path,
        "p95_path": p95_path,
        "p25_path": p25_path,
        "p75_path": p75_path,
        "p5_final_return": float(np.percentile(final_returns, 5)),
        "median_final_return": float(np.percentile(final_returns, 50)),
        "p95_final_return": float(np.percentile(final_returns, 95)),
    }

