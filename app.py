"""
Risk Ghost: Portfolio Risk & Liquidity Analytics Platform.

Designed with the iconic Apple Careers visual identity:
- Pure black canvas (#000000) and deep slate cards (#161617).
- High-contrast crisp white typography (#ffffff / #f5f5f7) and editorial light-gray (#a1a1a6).
- Signature Apple cyan-to-purple gradient text headline.
- Glassmorphic navigation header and 2x2 rounded showcase cards.
- High-contrast Matplotlib visualizations and accessible financial metric cards.
"""

from __future__ import annotations

import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

# Ensure writable cache directory for Matplotlib on Windows
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
import matplotlib.pyplot as plt

from pathlib import Path
import sys

# Ensure repository root is in sys.path for Streamlit Cloud deployment
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Also add src/ to path so bare module names resolve as fallback
_src_dir = str(ROOT_DIR / "src")
if _src_dir not in sys.path:
    sys.path.append(_src_dir)

import numpy as np
import pandas as pd
import streamlit as st

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
    MarketDataError,
    TickerNotFoundError,
    UnsupportedInstrumentError,
    fetch_fx_history,
    fetch_historical_prices_and_volumes,
    search_company,
)
from src.portfolio_manager import (
    HoldingItem,
    PortfolioAnalysisResult,
    parse_portfolio_csv,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Risk Ghost Stock Analyzer | Quantitative Finance",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Apple Careers Iconic Dark Aesthetic (Custom CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Base Deep Space Canvas */
    html, body, [class*="css"], [data-testid="stAppViewContainer"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif;
        color: #f5f5f7 !important;
        background-color: #000000 !important;
        -webkit-font-smoothing: antialiased;
    }
    
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 5rem;
        max-width: 1200px;
    }

    /* All General Text High Contrast */
    p, span, label, div {
        color: #f5f5f7;
    }

    /* Apple Careers Multi-Color Signature Gradients */
    /* 1. Iconic Warm Sunset (Matching User's Screenshot: Orange to Coral/Pink) */
    .gradient-sunset {
        background: linear-gradient(90deg, #ff9f0a 0%, #ff5e3a 45%, #ff2d55 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
        font-weight: 800;
    }

    /* 2. Iconic Aurora (Cyan to Electric Blue to Purple) */
    .gradient-aurora {
        background: linear-gradient(90deg, #30b0c7 0%, #2997ff 45%, #af52de 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
        font-weight: 800;
    }

    /* 3. Fresh Electric Mint to Cyan (Growth & Diversification) */
    .gradient-mint {
        background: linear-gradient(90deg, #30d158 0%, #30b0c7 50%, #64d2ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
        font-weight: 800;
    }

    /* 4. Radiant Violet to Hot Pink (Monte Carlo & Rigor) */
    .gradient-violet {
        background: linear-gradient(90deg, #bf5af2 0%, #da3ba3 50%, #ff375f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
        font-weight: 800;
    }

    /* 5. Golden Amber to Coral (Archegos Concentration Alert) */
    .gradient-amber {
        background: linear-gradient(90deg, #ffd60a 0%, #ff9f0a 50%, #ff453a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
        font-weight: 800;
    }

    /* Legacy compatibility class */
    .hero-gradient {
        background: linear-gradient(90deg, #ff9f0a 0%, #ff5e3a 45%, #ff2d55 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
        font-weight: 800;
    }
    .apple-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 28px;
        background: rgba(22, 22, 23, 0.85);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 980px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 36px;
    }

    .apple-nav-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff !important;
        text-decoration: none;
    }

    .apple-nav-brand-symbol {
        color: #2997ff;
        font-size: 1.35rem;
    }

    /* Apple Careers Editorial Hero Section */
    .hero-eyebrow {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff !important;
        letter-spacing: -0.01em;
        margin-bottom: 12px;
    }

    .hero-headline {
        font-size: 4.2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.04;
        color: #ffffff !important;
        margin-bottom: 32px;
        max-width: 980px;
    }

    /* Iconic Apple Cyan-Blue-Purple Gradient */
    .hero-gradient {
        background: linear-gradient(90deg, #30b0c7 0%, #2997ff 45%, #af52de 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
    }

    .hero-columns {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 36px;
        margin-bottom: 36px;
    }

    .hero-body-text {
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.65;
        color: #a1a1a6 !important;
    }

    /* Apple Careers Feature Cards (2x2 Grid) */
    .showcase-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
        margin-bottom: 48px;
    }

    .showcase-card {
        background: #161617;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 28px;
        padding: 40px 36px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 280px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease;
    }

    .showcase-card:hover {
        transform: translateY(-3px);
        border-color: rgba(255, 255, 255, 0.25);
    }

    .showcase-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #ffffff !important;
        margin-bottom: 12px;
    }

    .showcase-desc {
        font-size: 1.05rem;
        line-height: 1.5;
        color: #a1a1a6 !important;
        margin-bottom: 28px;
    }

    /* Apple Pill Buttons */
    .apple-white-pill {
        display: inline-block;
        background-color: #ffffff !important;
        color: #000000 !important;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 10px 24px;
        border-radius: 980px;
        text-decoration: none;
        box-shadow: 0 2px 10px rgba(255, 255, 255, 0.2);
        transition: transform 0.2s ease, opacity 0.2s ease;
        border: none;
        cursor: pointer;
    }

    .apple-white-pill:hover {
        opacity: 0.92;
        transform: scale(1.02);
    }

    /* Financial Metric Cards */
    .apple-card {
        background-color: #161617 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 22px !important;
        padding: 26px 28px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
        margin-bottom: 20px;
        transition: transform 0.2s ease;
    }

    .apple-card:hover {
        transform: translateY(-2px);
    }

    .metric-eyebrow {
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #86868b !important;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #ffffff !important;
        line-height: 1.1;
    }

    .metric-unit {
        font-size: 1.15rem;
        font-weight: 500;
        color: #86868b !important;
        margin-left: 6px;
    }

    .metric-caption {
        font-size: 0.88rem;
        color: #a1a1a6 !important;
        margin-top: 8px;
    }

    /* Badges with Vibrant Apple Accents */
    .badge-status {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 980px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }

    .badge-safe {
        background-color: #0c2b18 !important;
        color: #30d158 !important;
        border: 1px solid rgba(48, 209, 88, 0.4) !important;
    }

    .badge-warning {
        background-color: #332007 !important;
        color: #ff9f0a !important;
        border: 1px solid rgba(255, 159, 10, 0.4) !important;
    }

    .badge-danger {
        background-color: #361115 !important;
        color: #ff453a !important;
        border: 1px solid rgba(255, 69, 58, 0.45) !important;
    }

    .badge-aurora {
        background: rgba(48, 176, 199, 0.15) !important;
        color: #64d2ff !important;
        border: 1px solid rgba(100, 210, 255, 0.4) !important;
    }

    .badge-sunset {
        background: rgba(255, 159, 10, 0.15) !important;
        color: #ff9f0a !important;
        border: 1px solid rgba(255, 159, 10, 0.4) !important;
    }

    .badge-violet {
        background: rgba(191, 90, 242, 0.15) !important;
        color: #bf5af2 !important;
        border: 1px solid rgba(191, 90, 242, 0.4) !important;
    }

    .badge-mint {
        background: rgba(48, 209, 88, 0.15) !important;
        color: #30d158 !important;
        border: 1px solid rgba(48, 209, 88, 0.4) !important;
    }

    /* Section Headings */
    .section-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #ffffff !important;
        margin-top: 28px;
        margin-bottom: 6px;
    }

    .section-subtitle {
        font-size: 1rem;
        color: #a1a1a6 !important;
        margin-bottom: 22px;
    }

    /* Streamlit Expander, Inputs, and Tables Dark Overrides */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 20px !important;
        background-color: #161617 !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
        margin-bottom: 24px !important;
    }

    div[data-testid="stExpander"] details summary {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* =========================================================================
       BULLETPROOF APPLE WHITE PILL BUTTONS (CRISP BLACK TEXT ON ALL ELEMENTS)
       ========================================================================= */
    div.stButton > button,
    div.stDownloadButton > button,
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"],
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 980px !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        font-size: 0.94rem !important;
        box-shadow: 0 4px 14px rgba(255, 255, 255, 0.18) !important;
        transition: all 0.2s ease !important;
    }

    /* Override any nested text color so text is 100% visible pure black */
    div.stButton button *,
    div.stButton button p,
    div.stButton button span,
    div.stButton button div,
    div.stDownloadButton button *,
    div.stDownloadButton button p,
    div.stDownloadButton button span,
    div.stDownloadButton button div,
    button[data-testid="baseButton-secondary"] *,
    button[data-testid="baseButton-secondary"] p,
    button[data-testid="baseButton-secondary"] span,
    button[data-testid="baseButton-primary"] *,
    button[data-testid="baseButton-primary"] p,
    button[data-testid="baseButton-primary"] span,
    div[data-testid="stFormSubmitButton"] button *,
    div[data-testid="stFormSubmitButton"] button p,
    div[data-testid="stFormSubmitButton"] button span,
    .apple-white-pill,
    .apple-white-pill * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 700 !important;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background-color: #e5e5ea !important;
        color: #000000 !important;
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(255, 255, 255, 0.28) !important;
    }

    div.stButton button:hover *,
    div.stDownloadButton button:hover *,
    button[data-testid="baseButton-secondary"]:hover *,
    button[data-testid="baseButton-primary"]:hover * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }

    /* Streamlit Input fields */
    input[type="text"], input[type="number"], div[data-baseweb="select"] > div {
        background-color: #1c1c1e !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
    }

    /* Radio Navigation Segmented Styling */
    div[role="radiogroup"] {
        background-color: #1c1c1e !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 980px !important;
        padding: 4px !important;
        gap: 4px !important;
    }

    div[role="radiogroup"] label {
        border-radius: 980px !important;
        padding: 6px 20px !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #a1a1a6 !important;
        margin: 0 !important;
        transition: all 0.2s ease !important;
    }

    div[role="radiogroup"] label[data-checked="true"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-weight: 700 !important;
    }

    div[role="radiogroup"] label[data-checked="true"] *,
    div[role="radiogroup"] label[data-checked="true"] p,
    div[role="radiogroup"] label[data-checked="true"] span,
    div[role="radiogroup"] label[data-checked="true"] div {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 700 !important;
    }

    div[role="radiogroup"] label[data-checked="false"] *,
    div[role="radiogroup"] label[data-checked="false"] p,
    div[role="radiogroup"] label[data-checked="false"] span {
        color: #a1a1a6 !important;
        -webkit-text-fill-color: #a1a1a6 !important;
    }

    /* Table / Dataframe Dark Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 18px !important;
        overflow: hidden !important;
    }

    /* Info & Alert Banners */
    .apple-info-banner {
        background: rgba(41, 151, 255, 0.1);
        border: 1px solid rgba(41, 151, 255, 0.3);
        border-radius: 16px;
        padding: 16px 22px;
        color: #2997ff !important;
        font-size: 0.95rem;
        margin-bottom: 24px;
    }

    .apple-alert-banner {
        background: rgba(255, 69, 58, 0.12);
        border: 1px solid rgba(255, 69, 58, 0.35);
        border-radius: 16px;
        padding: 16px 22px;
        color: #ff453a !important;
        font-size: 0.95rem;
        margin-bottom: 24px;
    }

    /* Footer */
    .apple-footer {
        text-align: center;
        padding-top: 54px;
        font-size: 0.85rem;
        color: #86868b !important;
        border-top: 1px solid #2c2c2e;
        margin-top: 72px;
        letter-spacing: 0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Streamlit Caching for Public Market Data
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, max_entries=50, show_spinner=False)
def fetch_cached_public_market_data(
    tickers_tuple: tuple[str, ...],
    period: str = "60d",
    interval: str = "1d",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict], str]:
    """
    Public market data cache.
    Caches only public ticker data (Closes, Volumes, Metadata), NOT private user quantities.
    Keyed on symbols, period, and interval with a 1-hour TTL.
    """
    close_df, volume_df, meta_map, fetch_time = fetch_historical_prices_and_volumes(
        tickers=list(tickers_tuple),
        period=period,
        interval=interval,
    )
    meta_serialized = {
        k: {
            "symbol": v.symbol,
            "name": v.name,
            "exchange": v.exchange,
            "quote_type": v.quote_type,
            "currency": v.currency,
            "is_subunit": v.is_subunit,
            "subunit_factor": v.subunit_factor,
            "fetch_timestamp": v.fetch_timestamp,
        }
        for k, v in meta_map.items()
    }
    return close_df, volume_df, meta_serialized, fetch_time


@st.cache_data(ttl=3600, max_entries=50, show_spinner=False)
def fetch_cached_fx_history(
    from_curr: str,
    to_curr: str,
    period: str = "60d",
) -> pd.Series:
    """Cached public foreign exchange rates."""
    return fetch_fx_history(from_currency=from_curr, to_currency=to_curr, period=period)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "holdings_list" not in st.session_state:
    st.session_state.holdings_list = [
        {"ticker": "NVDA", "shares": 80.0},
        {"ticker": "TSLA", "shares": 100.0},
        {"ticker": "MSFT", "shares": 150.0},
        {"ticker": "AAPL", "shares": 120.0},
        {"ticker": "GOOGL", "shares": 60.0},
    ]

if "editor_version" not in st.session_state:
    st.session_state.editor_version = 0

if "last_uploaded_csv_sig" not in st.session_state:
    st.session_state.last_uploaded_csv_sig = None

if "imported_success_msg" not in st.session_state:
    st.session_state.imported_success_msg = None

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "search_query" not in st.session_state:
    st.session_state.search_query = ""


# -----------------------------------------------------------------------------
# Apple Careers Navigation Bar
# -----------------------------------------------------------------------------
col_nav_1, col_nav_2 = st.columns([1, 2])
with col_nav_1:
    st.markdown(
        """
        <div style="display:flex; flex-direction:column; gap:2px; padding-top:4px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:1.5rem; color:#2997ff; font-weight:800;">◈</span>
                <span style="font-size:1.35rem; font-weight:800; letter-spacing:-0.03em; color:#ffffff;">Risk Ghost Stock Analyzer</span>
            </div>
            <span style="font-size:0.75rem; color:#86868b; font-weight:400; padding-left:36px; letter-spacing:0.02em;">Created by Omkar M. Mhatre</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_nav_2:
    selected_view = st.radio(
        "Navigation",
        ["Portfolio Diagnostic", "Company Discovery", "Methodology & Disclosures"],
        horizontal=True,
        label_visibility="collapsed",
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)


# =============================================================================
# TAB 1: PORTFOLIO DIAGNOSTIC
# =============================================================================
if selected_view == "Portfolio Diagnostic":

    # Apple Careers Editorial Hero Section (Multi-Color Gradient Typography)
    st.markdown(
        """
        <div style="padding-top: 12px; margin-bottom: 32px;">
            <div class="hero-eyebrow">
                <span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:#2997ff; margin-right:8px; vertical-align:middle;"></span>
                QUANTITATIVE FINANCE
            </div>
            <div class="hero-headline">
                Your portfolio's <span class="gradient-aurora">true risk</span> may be <span class="gradient-sunset">invisible.</span>
            </div>
            <div class="hero-columns">
                <div class="hero-body-text">
                    Traditional retail brokerage dashboards display only nominal capital weight percentages 
                    and screen execution prices &mdash; obscuring how correlated, high-volatility equities 
                    quietly account for 60% to 90% of your total portfolio variance.
                </div>
                <div class="hero-body-text">
                    Risk Ghost uses <span style="color:#64d2ff; font-weight:600;">Euler homogeneous risk decomposition</span> and <span style="color:#ff9f0a; font-weight:600;">Almgren-Chriss market impact</span> 
                    models to expose latent concentration risk, calculate actual execution liquidity, 
                    and project covariance-aware Monte Carlo scenarios.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Apple Careers 2x2 Showcase Cards (Multi-Color Accents)
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="showcase-grid">
            <div class="showcase-card">
                <div>
                    <div class="showcase-title"><span class="gradient-aurora">Euler Attribution</span></div>
                    <div class="showcase-desc">
                        Decompose total portfolio volatility into exact component risk contributions using Euler's theorem for linearly homogeneous functions.
                    </div>
                </div>
                <div>
                    <span class="badge-status badge-aurora">Variance Decomposition</span>
                </div>
            </div>
            <div class="showcase-card">
                <div>
                    <div class="showcase-title"><span class="gradient-sunset">Liquidity Traps</span></div>
                    <div class="showcase-desc">
                        Identify execution friction by calculating Days to Liquidate under institutional 20% ADV rules and Almgren-Chriss square-root slippage.
                    </div>
                </div>
                <div>
                    <span class="badge-status badge-sunset">Market Microstructure</span>
                </div>
            </div>
            <div class="showcase-card">
                <div>
                    <div class="showcase-title"><span class="gradient-amber">Concentration Alerts</span></div>
                    <div class="showcase-desc">
                        Detect Archegos-style vulnerabilities when top-2 assets monopolize 60% or more of total risk despite nominal diversification.
                    </div>
                </div>
                <div>
                    <span class="badge-status badge-danger">Archegos Guard</span>
                </div>
            </div>
            <div class="showcase-card">
                <div>
                    <div class="showcase-title"><span class="gradient-violet">Monte Carlo Engine</span></div>
                    <div class="showcase-desc">
                        Project 1,000 correlated multi-asset return paths using empirical covariance Cholesky factorization over 60 trading days.
                    </div>
                </div>
                <div>
                    <span class="badge-status badge-violet">Covariance-Aware</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Portfolio Input & Controls
    # -------------------------------------------------------------------------
    with st.expander("Configure Portfolio & Base Currency", expanded=True):
        col_p1, col_p2 = st.columns([2, 1])

        with col_p1:
            st.markdown("##### Portfolio Holdings Table")
            st.caption("Enter any Yahoo Finance ticker directly (e.g. AAPL, NVDA, RELIANCE.NS, VOD.L, 005930.KS).")

            holdings_df = pd.DataFrame(st.session_state.holdings_list)
            editor_key = f"holdings_editor_{st.session_state.get('editor_version', 0)}"
            edited_df = st.data_editor(
                holdings_df,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker Symbol", help="Yahoo Finance symbol"),
                    "shares": st.column_config.NumberColumn("Shares / Units", min_value=0.01, step=1.0),
                },
                key=editor_key,
            )

        with col_p2:
            st.markdown("##### Base Reporting Currency")
            base_currency = st.selectbox(
                "Base Currency",
                ["USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD", "CHF"],
                index=0,
                help="Historical prices and position values will be converted into this base currency.",
            )

            st.markdown("##### Quick Portfolio Presets")
            col_preset_1, col_preset_2 = st.columns(2)
            with col_preset_1:
                if st.button("Tech Giants (US)", key="preset_tech", use_container_width=True):
                    st.session_state.holdings_list = [
                        {"ticker": "NVDA", "shares": 80.0},
                        {"ticker": "TSLA", "shares": 100.0},
                        {"ticker": "MSFT", "shares": 150.0},
                        {"ticker": "AAPL", "shares": 120.0},
                        {"ticker": "GOOGL", "shares": 60.0},
                    ]
                    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
                    st.session_state["imported_success_msg"] = None
                    st.rerun()
            with col_preset_2:
                if st.button("Global Mixed FX", key="preset_global", use_container_width=True):
                    st.session_state.holdings_list = [
                        {"ticker": "AAPL", "shares": 100.0},
                        {"ticker": "VOD.L", "shares": 800.0},
                        {"ticker": "RELIANCE.NS", "shares": 40.0},
                    ]
                    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
                    st.session_state["imported_success_msg"] = None
                    st.rerun()

            st.markdown("##### CSV Import")
            uploaded_file = st.file_uploader(
                "Upload Portfolio CSV",
                type=["csv"],
                help="Accepts 'Ticker, Shares' or 'Stock names, No. of shares holds'",
                key="portfolio_csv_uploader",
            )
            if uploaded_file is not None:
                file_sig = f"{uploaded_file.name}_{uploaded_file.size}"
                if st.session_state.get("last_uploaded_csv_sig") != file_sig:
                    try:
                        parsed = parse_portfolio_csv(uploaded_file)
                        st.session_state.holdings_list = [{"ticker": item[0], "shares": item[1]} for item in parsed]
                        st.session_state["last_uploaded_csv_sig"] = file_sig
                        st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
                        st.session_state["imported_success_msg"] = f"Imported {len(parsed)} holdings successfully from {uploaded_file.name}."
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to parse CSV: {e}")
            else:
                st.session_state["last_uploaded_csv_sig"] = None
                st.session_state["imported_success_msg"] = None

            if st.session_state.get("imported_success_msg"):
                st.success(st.session_state["imported_success_msg"])

        # Ingest clean holdings
        clean_holdings = []
        for _, row in edited_df.iterrows():
            t_str = str(row.get("ticker", "")).strip().upper()
            q_val = float(row.get("shares", 0.0) or 0.0)
            if t_str and q_val > 0:
                clean_holdings.append((t_str, q_val))

    # -------------------------------------------------------------------------
    # Analysis Pipeline
    # -------------------------------------------------------------------------
    if not clean_holdings:
        st.info("Add at least one holding with a positive share quantity above to run analysis.")
    else:
        tickers = [h[0] for h in clean_holdings]
        shares = [h[1] for h in clean_holdings]

        analysis_error = None
        analysis_result = None

        with st.spinner("Analyzing portfolio risk and liquidity online..."):
            try:
                # 1. Fetch public market data using cached helper
                close_df, volume_df, meta_map_raw, fetch_time = fetch_cached_public_market_data(
                    tickers_tuple=tuple(tickers),
                    period="60d",
                )

                # 2. Multi-currency detection and historical FX alignment
                base_prices_df = close_df.copy()
                is_multi_curr = False
                fx_pairs = []

                for t in tickers:
                    meta = meta_map_raw[t]
                    t_curr = meta["currency"]

                    if t_curr.upper() != base_currency.upper():
                        is_multi_curr = True
                        fx_series = fetch_cached_fx_history(
                            from_curr=t_curr,
                            to_curr=base_currency,
                            period="60d",
                        )
                        norm_target = close_df.index.tz_localize(None) if hasattr(close_df.index, "tz") and close_df.index.tz else close_df.index
                        fx_aligned = fx_series.reindex(norm_target).ffill(limit=5).bfill(limit=5)
                        base_prices_df[t] = close_df[t] * fx_aligned
                        fx_pairs.append(f"{t_curr}->{base_currency}")

                # 3. Capital weights in base reporting currency
                latest_base_prices = [float(base_prices_df[t].iloc[-1]) for t in tickers]
                latest_local_prices = [float(close_df[t].iloc[-1]) for t in tickers]
                pos_values = [s * p for s, p in zip(shares, latest_base_prices)]
                total_val = sum(pos_values)
                weights = np.array([v / total_val if total_val > 0 else 1.0 / len(tickers) for v in pos_values])

                # 4. Returns & Covariance matrix in base currency
                port_val_series, returns_df, cov_matrix = calculate_portfolio_timeseries(base_prices_df, shares)
                cov_arr = cov_matrix.values
                daily_vols = returns_df.std().values

                # 5. Euler Risk Decomposition
                port_vol, mrc, crc = calculate_component_risk_contributions(weights, cov_arr)
                rc_pct = calculate_risk_contribution(weights, cov_arr)

                # Bounded illustrative heuristic risk-balancing tilt
                suggested_w = rebalance_to_equal_risk(rc_pct, weights, max_weight=0.30, min_weight=0.01)

                # Recalculate portfolio volatility and post-rebalance risk contributions
                new_port_vol, new_crc, new_rc_pct = evaluate_rebalanced_portfolio(suggested_w, cov_arr)
                annualized_vol = port_vol * np.sqrt(252) * 100.0
                new_annualized_vol = new_port_vol * np.sqrt(252) * 100.0
                vol_delta = new_annualized_vol - annualized_vol

                # 6. Liquidity Diagnostics
                adv_values = [float(volume_df[t].mean()) if t in volume_df else 0.0 for t in tickers]
                holding_items: list[HoldingItem] = []
                knight_alert = False

                for i, t in enumerate(tickers):
                    slip, days = calculate_market_impact(shares[i], adv_values[i], daily_vols[i])
                    if days >= 5.0 or slip >= 3.0:
                        knight_alert = True
                    mdd = calculate_max_drawdown(base_prices_df[t].values)
                    w_pct = weights[i] * 100.0
                    r_pct = rc_pct[i]
                    diff = r_pct - w_pct
                    if r_pct < 0:
                        flag = "Hedge / Risk Reducer"
                    elif diff >= 10.0:
                        flag = "High Risk Contributor"
                    elif diff <= -5.0:
                        flag = "Low Risk Contributor"
                    else:
                        flag = "Balanced"

                    holding_items.append(
                        HoldingItem(
                            identifier=t,
                            shares=shares[i],
                            ticker=t,
                            name=meta_map_raw[t]["name"],
                            currency=meta_map_raw[t]["currency"],
                            latest_local_price=latest_local_prices[i],
                            latest_base_price=latest_base_prices[i],
                            position_value_base=pos_values[i],
                            weight_pct=w_pct,
                            risk_contribution_pct=r_pct,
                            marginal_risk_contribution=mrc[i],
                            component_risk_contribution=crc[i],
                            daily_volatility_pct=daily_vols[i] * 100.0,
                            adv_shares=adv_values[i],
                            days_to_exit=days,
                            slippage_pct=slip,
                            suggested_weight_pct=suggested_w[i] * 100.0,
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

                analysis_result = PortfolioAnalysisResult(
                    base_currency=base_currency,
                    total_valuation_base=total_val,
                    daily_portfolio_volatility_pct=port_vol * 100.0,
                    annualized_portfolio_volatility_pct=annualized_vol,
                    holdings=holding_items,
                    top_two_risk_concentration_pct=top_two_rc,
                    top_two_tickers=top_two_tickers,
                    concentration_tier=conc_tier,
                    archegos_alert=archegos_alert,
                    knight_capital_alert=knight_alert,
                    rebalanced_annualized_volatility_pct=new_annualized_vol,
                    rebalance_volatility_delta_pct=vol_delta,
                    fetch_timestamp=fetch_time,
                    base_prices_df=base_prices_df,
                    returns_df=returns_df,
                    cov_matrix=cov_matrix,
                    correlation_matrix=returns_df.corr(),
                    portfolio_value_history=port_val_series,
                    is_multi_currency=is_multi_curr,
                    fx_pairs_used=list(set(fx_pairs)),
                )
            except (MarketDataError, TickerNotFoundError, UnsupportedInstrumentError, InsufficientDataError, FXConversionError) as err:
                analysis_error = str(err)
            except Exception as err:
                analysis_error = f"Unexpected analysis failure: {err}"

        # ---------------------------------------------------------------------
        # Output Presentation
        # ---------------------------------------------------------------------
        if analysis_error:
            st.markdown(
                f"""
                <div class="apple-alert-banner">
                    <strong>Market Data Request Failed:</strong> {analysis_error}<br>
                    <small>Verify ticker symbols exist on Yahoo Finance and have sufficient trading history.</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Retry Market Data Request"):
                st.cache_data.clear()
                st.rerun()

        elif analysis_result:
            # Metadata Disclosure
            st.caption(
                f"Data Timestamp (UTC): {analysis_result.fetch_timestamp} • "
                f"Historical daily data from Yahoo Finance • Non-executable diagnostic quotes"
            )

            if analysis_result.is_multi_currency:
                st.markdown(
                    f"""
                    <div class="apple-info-banner">
                        <strong>Multi-Currency Portfolio:</strong> Foreign prices converted into 
                        <code>{analysis_result.base_currency}</code> using daily FX rates ({', '.join(analysis_result.fx_pairs_used)}).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Apple-Style Multi-Color Financial Metric Cards
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)

            with col_m1:
                st.markdown(
                    f"""
                    <div class="apple-card">
                        <div class="metric-eyebrow" style="color:#64d2ff !important;">TOTAL VALUATION</div>
                        <div class="metric-value">{analysis_result.total_valuation_base:,.0f}<span class="metric-unit" style="color:#64d2ff !important;">{analysis_result.base_currency}</span></div>
                        <div class="metric-caption">{len(analysis_result.holdings)} evaluated holdings</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_m2:
                st.markdown(
                    f"""
                    <div class="apple-card">
                        <div class="metric-eyebrow" style="color:#ffd60a !important;">PORTFOLIO VOLATILITY</div>
                        <div class="metric-value" style="color:#ffd60a !important;">{analysis_result.annualized_portfolio_volatility_pct:.1f}<span class="metric-unit" style="color:#ffd60a !important;">%</span></div>
                        <div class="metric-caption">{analysis_result.daily_portfolio_volatility_pct:.2f}% daily standard deviation</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_m3:
                if analysis_result.archegos_alert:
                    conc_color = "#ff453a"
                    status_badge = '<span class="badge-status badge-danger">Critical Concentration (≥60%)</span>'
                elif analysis_result.top_two_risk_concentration_pct >= 40.0:
                    conc_color = "#ff9f0a"
                    status_badge = '<span class="badge-status badge-warning">Elevated Concentration (40-60%)</span>'
                else:
                    conc_color = "#30d158"
                    status_badge = '<span class="badge-status badge-mint">Diversified Risk (<40%)</span>'

                top_names = " + ".join(analysis_result.top_two_tickers) if analysis_result.top_two_tickers else "N/A"
                st.markdown(
                    f"""
                    <div class="apple-card">
                        <div class="metric-eyebrow" style="color:#ff375f !important;">TOP 2 RISK CONCENTRATION</div>
                        <div class="metric-value" style="color: {conc_color} !important;">{analysis_result.top_two_risk_concentration_pct:.1f}<span class="metric-unit">%</span></div>
                        <div class="metric-caption">{status_badge}<br><span style="font-size:0.78rem; color:#86868b;">Top: {top_names}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_m4:
                liq_badge = (
                    '<span class="badge-status badge-warning">Illiquidity Alert</span>'
                    if analysis_result.knight_capital_alert
                    else '<span class="badge-status badge-mint">High Market Depth</span>'
                )
                liq_color = "#ff9f0a" if analysis_result.knight_capital_alert else "#30d158"
                st.markdown(
                    f"""
                    <div class="apple-card">
                        <div class="metric-eyebrow" style="color:#30d158 !important;">LIQUIDITY PROFILE</div>
                        <div class="metric-value" style="font-size:1.6rem; margin-top:6px; color:{liq_color} !important;">{'Attention' if analysis_result.knight_capital_alert else 'Liquid'}</div>
                        <div class="metric-caption">{liq_badge}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if analysis_result.archegos_alert:
                st.markdown(
                    f"""
                    <div class="apple-alert-banner">
                        <strong>Concentration Alert:</strong> Your top two holdings contribute 
                        <strong>{analysis_result.top_two_risk_concentration_pct:.1f}%</strong> of total portfolio variance. 
                        Nominal diversification provides little downside protection under market stress.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # -----------------------------------------------------------------
            # Visual Analytics Grid
            # -----------------------------------------------------------------
            st.markdown("<div class='section-title'><span class='gradient-aurora'>Risk Attribution</span> & Pairwise Correlation</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-subtitle'>Euler homogeneous risk decomposition vs pairwise correlation.</div>", unsafe_allow_html=True)

            col_viz_1, col_viz_2 = st.columns([3, 2])

            with col_viz_1:
                st.markdown("##### Capital Weight % vs. True Risk Contribution %")
                fig_rc = plot_capital_vs_risk(analysis_result.holdings)
                st.pyplot(fig_rc)
                plt.close(fig_rc)
                st.caption("Coral bars highlight positions where true risk contribution exceeds capital allocation by ≥ 10%. Green bars indicate net variance reducers / hedges (RC < 0).")

            with col_viz_2:
                st.markdown("##### Return Correlation Matrix")
                fig_corr = plot_correlation_heatmap(analysis_result.correlation_matrix)
                st.pyplot(fig_corr)
                plt.close(fig_corr)
                st.caption("Pairwise Pearson correlation of daily simple returns.")

            # Historical Drawdowns
            st.markdown("<div class='section-title'><span class='gradient-amber'>Historical Drawdown</span> Trajectories</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-subtitle'>60 trading-day historical peak-to-trough performance across holdings.</div>", unsafe_allow_html=True)
            fig_dd = plot_drawdown_history(analysis_result.base_prices_df)
            st.pyplot(fig_dd)
            plt.close(fig_dd)

            # -----------------------------------------------------------------
            # Diagnostic Holdings Table
            # -----------------------------------------------------------------
            st.markdown("<div class='section-title'><span class='gradient-aurora'>Diagnostic</span> Holdings Table</div>", unsafe_allow_html=True)

            summary_rows = []
            for h in analysis_result.holdings:
                summary_rows.append({
                    "Ticker": h.ticker,
                    "Name": h.name,
                    "Currency": h.currency,
                    "Shares": f"{h.shares:,.0f}",
                    f"Position ({analysis_result.base_currency})": f"{h.position_value_base:,.2f}",
                    "Capital Weight": f"{h.weight_pct:.1f}%",
                    "Risk Contribution": f"{h.risk_contribution_pct:.1f}%",
                    "Daily Volatility": f"{h.daily_volatility_pct:.2f}%",
                    "Days to Liquidate": format_days_to_exit(h.days_to_exit),
                    "Est. Slippage": f"{h.slippage_pct:.2f}%",
                    "Max Drawdown": f"{h.max_drawdown_pct:.2f}%",
                    "Diagnostic Status": h.risk_flag,
                })

            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            # CSV Download
            csv_data = pd.DataFrame(summary_rows).to_csv(index=False)
            st.download_button(
                label="Download Diagnostic Report (CSV)",
                data=csv_data,
                file_name=f"risk_ghost_diagnostic_{base_currency}.csv",
                mime="text/csv",
            )

            # -----------------------------------------------------------------
            # Liquidity & Risk-Balancing Adjustment
            # -----------------------------------------------------------------
            col_lq, col_rb = st.columns([1, 1])

            with col_lq:
                st.markdown("<div class='section-title'><span class='gradient-sunset'>Liquidity</span> & Market Impact</div>", unsafe_allow_html=True)
                fig_liq = plot_liquidity_matrix(analysis_result.holdings)
                st.pyplot(fig_liq)
                plt.close(fig_liq)
                st.caption("Based on institutional 20% ADV execution heuristic. Slippage includes a 0.05% (5 bps) baseline retail spread assumption for liquid equities, not zero market impact.")

            with col_rb:
                st.markdown("<div class='section-title'><span class='gradient-mint'>Illustrative</span> Risk-Balancing</div>", unsafe_allow_html=True)
                st.caption(
                    f"Illustrative heuristic tilt (max 30% position cap). "
                    f"Tilts projected portfolio volatility from {analysis_result.annualized_portfolio_volatility_pct:.1f}% to "
                    f"{analysis_result.rebalanced_annualized_volatility_pct:.1f}% ({analysis_result.rebalance_volatility_delta_pct:+.1f}%). "
                    f"Evaluated against empirical covariance; not an exact non-linear equal-risk optimizer."
                )

                rebal_display = []
                for h in analysis_result.holdings:
                    delta = h.suggested_weight_pct - h.weight_pct
                    rebal_display.append({
                        "Ticker": h.ticker,
                        "Current Wt": f"{h.weight_pct:.1f}%",
                        "Current RC": f"{h.risk_contribution_pct:.1f}%",
                        "Suggested Wt (30% Cap)": f"{h.suggested_weight_pct:.1f}%",
                        "Target Adjustment": f"{delta:+.1f}%",
                        "Projected RC": f"{h.projected_risk_contribution_pct:.1f}%",
                    })
                st.dataframe(pd.DataFrame(rebal_display), use_container_width=True)

            # -----------------------------------------------------------------
            # Covariance-Aware Monte Carlo Scenarios
            # -----------------------------------------------------------------
            st.markdown("<div class='section-title'><span class='gradient-violet'>Covariance-Aware</span> Monte Carlo Scenarios</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-subtitle'>1,000 correlated multi-asset return paths projected over a 60-day horizon using Cholesky factorization.</div>",
                unsafe_allow_html=True,
            )

            mc_results = run_monte_carlo(
                weights=np.array([h.weight_pct / 100.0 for h in analysis_result.holdings]),
                returns_df=analysis_result.returns_df,
                initial_value=analysis_result.total_valuation_base,
                horizon_days=60,
                num_simulations=1000,
            )

            col_mc_chart, col_mc_stats = st.columns([3, 1])

            with col_mc_chart:
                fig_mc = plot_monte_carlo(mc_results, currency=analysis_result.base_currency)
                st.pyplot(fig_mc)
                plt.close(fig_mc)

            with col_mc_stats:
                st.markdown(
                    f"""
                    <div class="apple-card" style="margin-top:20px;">
                        <div class="metric-eyebrow">60-Day Statistical Bounds</div>
                        <div style="margin-top:14px;">
                            <span style="color:#30d158; font-weight:600; font-size:0.85rem;">95th Percentile:</span><br>
                            <span style="font-size:1.5rem; font-weight:700; color:#ffffff;">{mc_results['p95_final_return']:+.1f}%</span>
                        </div>
                        <div style="margin-top:14px;">
                            <span style="color:#2997ff; font-weight:600; font-size:0.85rem;">Median Path:</span><br>
                            <span style="font-size:1.5rem; font-weight:700; color:#ffffff;">{mc_results['median_final_return']:+.1f}%</span>
                        </div>
                        <div style="margin-top:14px;">
                            <span style="color:#ff453a; font-weight:600; font-size:0.85rem;">5th Percentile:</span><br>
                            <span style="font-size:1.5rem; font-weight:700; color:#ffffff;">{mc_results['p5_final_return']:+.1f}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =============================================================================
# TAB 2: COMPANY DISCOVERY
# =============================================================================
elif selected_view == "Company Discovery":
    st.markdown(
        """
        <div style="padding-top: 12px; margin-bottom: 32px;">
            <div class="hero-eyebrow">
                <span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:#ff9f0a; margin-right:8px; vertical-align:middle;"></span>
                GLOBAL INSTRUMENT DISCOVERY
            </div>
            <div class="hero-headline">
                Discover Global Equities <span class="gradient-sunset">& ETFs.</span>
            </div>
            <div class="hero-columns">
                <div class="hero-body-text">
                    Search international listings by company name using the verified Yahoo Finance search interface.
                </div>
                <div class="hero-body-text">
                    Review exchange, instrument type, and trading currency before selecting candidate listings.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        query = st.text_input("Enter Company Name or Ticker (e.g. Apple, Tata, Reliance, Microsoft, LVMH)", value="")
    with col_q2:
        st.write("")
        st.write("")
        search_clicked = st.button("Search Yahoo Finance", use_container_width=True)

    if search_clicked and query.strip():
        with st.spinner(f"Searching for '{query.strip()}'..."):
            try:
                results = search_company(query.strip(), max_results=10)
                st.session_state.search_results = results
            except Exception as e:
                st.error(f"Search failed: {e}")

    if st.session_state.search_results:
        st.markdown(f"##### Candidate Listings ({len(st.session_state.search_results)} found)")
        st.caption("Multiple listings and share classes may exist for the same firm. Select the specific ticker intended.")

        for r in st.session_state.search_results:
            col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns([2, 3, 2, 2, 2])
            with col_c1:
                st.markdown(f"<span style='font-size:1.15rem; font-weight:800; color:#64d2ff;'>{r['symbol']}</span>", unsafe_allow_html=True)
            with col_c2:
                st.write(r["name"])
            with col_c3:
                st.caption(f"Exchange: {r['exchange']}")
            with col_c4:
                st.caption(f"Type: {r['quote_type']}")
            with col_c5:
                if st.button(f"Add {r['symbol']}", key=f"add_{r['symbol']}"):
                    st.session_state.holdings_list.append({"ticker": r["symbol"], "shares": 10.0})
                    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
                    st.success(f"Added {r['symbol']} to portfolio.")
            st.markdown("<hr style='margin: 8px 0; border: none; border-bottom: 1px solid #2c2c2e;'>", unsafe_allow_html=True)


# =============================================================================
# TAB 3: METHODOLOGY & DISCLOSURES
# =============================================================================
elif selected_view == "Methodology & Disclosures":
    st.markdown(
        """
        <div style="padding-top: 12px; margin-bottom: 32px;">
            <div class="hero-eyebrow">
                <span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:#bf5af2; margin-right:8px; vertical-align:middle;"></span>
                QUANTITATIVE RIGOR & REGULATION
            </div>
            <div class="hero-headline">
                Methodology & <span class="gradient-violet">Regulatory Disclosures.</span>
            </div>
            <div class="hero-columns">
                <div class="hero-body-text">
                    Theoretical foundations of Euler homogeneous risk attribution and Almgren-Chriss market microstructure.
                </div>
                <div class="hero-body-text">
                    Compliance notice regarding unofficial market data retrieval and open-source usage terms.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 1. Euler Homogeneous Risk Decomposition
        Portfolio volatility $\\sigma_p = \\sqrt{W^T \\Sigma W}$ is linearly homogeneous of degree 1.
        By Euler's theorem:
        $$\\sigma_p(W) = \\sum_{i=1}^n w_i \\frac{\\partial \\sigma_p}{\\partial w_i}$$
        
        The marginal risk contribution (MRC) is:
        $$\\text{MRC}_i = \\frac{(\\Sigma W)_i}{\\sigma_p}$$
        
        Component risk contribution ($CRC_i = w_i \\cdot \\text{MRC}_i$) reconciles exactly to total portfolio volatility:
        $$\\sum_{i=1}^n CRC_i = \\sigma_p$$
        
        Percentage risk contribution is defined as:
        $$RC_{\\%, i} = \\frac{w_i (\\Sigma W)_i}{\\sigma_p^2} \\times 100\\%$$

        ### 2. Market Impact & Liquidity Model
        - **Days to Liquidate:** Based on the institutional prudence benchmark restricting daily selling volume to 20% of Average Daily Volume (ADV):
          $$\\text{Days to Exit} = \\max\\left(1.0, \\frac{\\text{Shares}}{0.20 \\times \\text{ADV}}\\right)$$
        - **Expected Slippage:** Modeled using the Almgren-Chriss square-root impact formulation with a 5 basis point baseline retail spread:
          $$\\text{Slippage \\%} = 0.05 + 0.5 \\times \\sigma_{\\text{daily}} \\times \\sqrt{\\frac{\\text{Shares}}{\\text{ADV}}}$$
          *Disclosure:* This slippage metric is an educational heuristic model, not a guaranteed broker execution quote.

        ### 3. International FX Conversion & Subunits
        - Multi-currency portfolios convert historical prices into the user-selected reporting currency using daily exchange rate series from Yahoo Finance.
        - London Stock Exchange securities quoted in pence (`GBp` / `GBX`) are normalized to pounds (`GBP`) by dividing by 100.
        - Trading calendars are synchronized via bounded forward-filling (up to 2 trading days) to accommodate regional market holidays.

        ### 4. Data Provider & Terms Disclosure
        - **Provider:** Market data is retrieved via the open-source `yfinance` library.
        - **Unofficial Status:** `yfinance` is an unofficial community library and is neither affiliated with nor endorsed by Yahoo Inc.
        - **Usage Terms:** Open-source licensing of the software does not grant unrestricted redistribution rights to underlying financial data. Historical quotes are intended strictly for personal research, diagnostic analysis, and computer science educational purposes.
        """
    )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="apple-footer">
        Risk Ghost &bull; Quantitative Portfolio Risk Diagnostic &bull; Inspired by Apple's Iconic Careers Design &bull; Built with Python & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
