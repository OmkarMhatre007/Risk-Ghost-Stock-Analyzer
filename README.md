# Risk Ghost Stock Analyzer: Exposing Latent Concentration Risk and Market Impact

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit%20Cloud-FF4B4B?logo=streamlit)](https://risk-ghost-stock-analyzer-a2gmf7buzk5ms6icvqludo.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-24%20passed-brightgreen.svg)](tests/)

#### Video Demo: Upcoming...
#### Research Paper: [Read the full academic methodology report (docs/REPORT.md)](docs/REPORT.md)
#### Live App: [Visit App](https://risk-ghost-stock-analyzer-a2gmf7buzk5ms6icvqludo.streamlit.app/)
---

## 1. Project Overview

Retail portfolio dashboards typically show only **Weight %** (capital allocation) and **Current Price**. However, this simplicity conceals two critical portfolio risks:

1. **Concentration Risk:** An investor holding 3–5 stocks might assume equal nominal weights guarantee diversification. In reality, a volatile and correlated stock can contribute **50% to 90% of total portfolio risk** (Euler homogeneous risk decomposition).
2. **Liquidity Illusion:** Selling quantities relative to **Average Daily Volume (ADV)** moves market price against the investor, causing liquidation delays and execution slippage (Almgren-Chriss square-root impact heuristic).

**Risk Ghost Stock Analyzer** evaluates multi-asset portfolios against empirical price and volume data retrieved **online via Yahoo Finance**. It provides both an interactive web platform (featuring an original, Apple Careers-inspired minimalist aesthetic) and a command-line interface (CLI) to compute:
* Capital Weight % vs. True Risk Contribution % (Euler Decomposition, including empirical hedge detection)
* Heuristic Liquidity Metrics (Days to Exit & Expected Slippage %)
* Illustrative Bounded Risk-Balancing Adjustment (with position-weight caps & post-tilt risk evaluation)
* 60-Day Peak-to-Trough Historical Drawdown
* Asset Return Correlation Heatmap
* Covariance-Aware Multivariate Monte Carlo Scenarios (guaranteed PSD via spectral decomposition)

---

## 2. Key Architecture & Features

### 2.1 Universal Ticker Support & Search (No Allowlist)
- Users can enter any Yahoo Finance ticker directly (`AAPL`, `NVDA`, `005930.KS`, `RELIANCE.NS`, `VOD.L`).
- Search companies by name via the verified `yfinance.Search` API, displaying candidate listings with exchange, instrument type, and trading currency without auto-selecting the first match.
- Core scope focuses on listed equities and ETFs; indices (e.g., `^GSPC`) are explicitly rejected as purchasable holdings.

### 2.2 Online-Only Market Data & Caching
- **Zero Offline Fallbacks:** Operates entirely on live, in-memory data retrieved from Yahoo Finance.
- **Failures Handled Honestly:** Network timeouts, rate limits, missing tickers, and holiday calendar gaps are surfaced with clear diagnostic messages and retry controls rather than silently loading stale CSVs.
- **Streamlit Caching:** Public ticker history is cached with a 1-hour TTL (`@st.cache_data(ttl=3600)`) based strictly on ticker symbols, date window, and intervals. Private user quantities and portfolios are never globally cached. Fetch timestamps are prominently displayed.

### 2.3 International Listings & Base-Currency FX Support
- Automatic quote currency detection.
- Normalizes London Stock Exchange pence quote conventions (`GBp` / `GBX` converted to `GBP`).
- Multi-currency portfolios convert historical prices into the chosen base reporting currency (e.g., `USD`, `EUR`, `GBP`, `INR`) using daily online FX rates before computing covariance, volatility, and risk contributions.
- Share volume is preserved in its native trading units when computing Average Daily Volume (ADV).

### 2.4 Apple Careers-Inspired Visual Design
- Streamlit web interface crafted with an original, minimalist design philosophy: generous whitespace, refined typography, subtle borders, crisp metric cards, and zero dashboard clutter.
- High-contrast Matplotlib visualizations:
  - Capital Allocation vs. Risk Contribution bar charts.
  - Pairwise Pearson correlation heatmap.
  - Historical drawdown trajectories.
  - Days-to-Exit vs. Slippage liquidity matrix.
  - Covariance-aware Monte Carlo fan charts (5th, 25th, 50th, 75th, 95th percentiles).

### 2.5 Clean Modular Architecture
```
Risk-Ghost-Stock-Analyzer/
├── app.py                      # Main Streamlit web application entry point
├── project.py                  # CLI diagnostic entry point
├── requirements.txt            # Python dependencies for local & cloud runtime
├── pytest.ini                  # Pytest configuration
├── LICENSE                     # Full MIT License text
├── README.md                   # Project documentation & usage guide
├── .gitignore                  # Git tracking exclusion rules
├── .env.example                # Sample environment configuration
├── .streamlit/
│   └── config.toml             # Streamlit dark theme configuration
├── data/
│   └── portfolio.csv           # Default portfolio asset holdings
├── docs/
│   └── REPORT.md               # Detailed academic report & mathematical research paper
├── src/
│   ├── __init__.py             # Package initializer & public API
│   ├── analytics.py            # Euler decomposition, liquidity, Monte Carlo engine
│   ├── charts.py               # Apple-inspired Matplotlib dark visualizations
│   ├── data_provider.py        # Online Yahoo Finance provider & FX converter
│   └── portfolio_manager.py    # Multi-currency normalization & portfolio orchestrator
└── tests/
    ├── __init__.py             # Test package initializer
    ├── test_project.py         # Comprehensive unit tests (24 tests, offline fixtures)
    └── test_smoke.py           # Optional live network smoke tests
```

---

## 3. Mathematical Methodology

### 3.1 Euler Homogeneous Risk Decomposition

Portfolio volatility $\sigma_p = \sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}$ is mathematically homogeneous of degree 1.

Euler's Theorem guarantees that the sum of component risk contributions equals total portfolio volatility:

$$
\text{MRC}_i = \frac{(\mathbf{\Sigma} \mathbf{w})_i}{\sigma_p}
$$

$$
\text{CRC}_i = w_i \cdot \text{MRC}_i = \frac{w_i (\mathbf{\Sigma} \mathbf{w})_i}{\sigma_p}, \quad \sum_{i=1}^n \text{CRC}_i = \sigma_p
$$

Percentage risk contribution $\text{RC}_i$ (scaled to sum to 100):

$$
\text{RC}_i = \frac{\text{CRC}_i}{\sigma_p} \times 100 = \frac{w_i (\mathbf{\Sigma} \mathbf{w})_i}{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}} \times 100, \quad \sum_{i=1}^n \text{RC}_i = 100
$$

### 3.2 Almgren-Chriss Square-Root Liquidity Model

$$
\text{Days to Exit} = \frac{\text{Shares}}{0.20 \times \text{ADV}}
$$

Expected execution slippage percentage:

$$
\text{Expected Slippage} = 0.05 + 0.5 \times \sigma_{\text{daily}} \times \sqrt{\frac{\text{Shares}}{\text{ADV}}}
$$

*Note: Slippage is an illustrative percentage estimate (incorporating a 0.05% retail spread baseline plus volatility-scaled square-root impact), not a guaranteed broker execution quote. Preserves actual fractional trading days (e.g. `< 0.01 d`).*

### 3.3 Illustrative Bounded Risk-Balancing Adjustment

* Assets acting as hedges ($\text{RC}_i \le 0$) maintain neutral scaling ($1.0\times$).
* Modest risk contributors ($0 < \text{RC}_i < 1$) are capped at $1.5\times$.
* High risk contributors are dampened using a square-root scaling ratio:

$$
\text{multiplier}_i = \text{clip}\left(\sqrt{\frac{\text{Target RC}}{\text{RC}_i}}, 0.5, 2.0\right)
$$

* Position constraints are iteratively projected onto $[w_{\min}=0.01, w_{\max}=0.30]$ ensuring $\sum_{i=1}^n w_i = 1.0$.
* Recalculates and displays resulting post-tilt portfolio volatility and risk contributions.

*Note: This is an illustrative heuristic adjustment toward risk parity, not an exact non-linear optimizer.*

### 3.4 Maximum Drawdown

$$
\text{MDD} = \min_{t} \left( \frac{P_t - \max_{0 \le \tau \le t} P_\tau}{\max_{0 \le \tau \le t} P_\tau} \right) \times 100
$$

### 3.5 Covariance-Aware Monte Carlo Simulation

$$
R_{\text{sim}} = \mu + Z L^T, \quad \mathbf{\Sigma}_{\text{PSD}} = V \Lambda_{\text{PSD}} V^T = L L^T, \quad Z \sim \mathcal{N}(0, I)
$$

To guarantee a positive semi-definite covariance matrix, spectral clipping is applied to eigenvalues:

$$
\Lambda_{\text{PSD}} = \text{diag}\left(\max(\lambda_i, 10^{-8})\right)
$$

This ensures robust Cholesky factorization $L = \text{chol}(\mathbf{\Sigma}_{\text{PSD}})$ across all asset combinations over 60 trading days without singular matrix errors.

---

## 4. How to Run Locally

### 4.1 Prerequisites & Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 4.2 Interactive Streamlit Web Application
Launch the Apple-inspired web platform:
```bash
streamlit run app.py
```

### 4.3 Command-Line Interface (CLI)
Analyze the default portfolio (`data/portfolio.csv`):
```bash
python project.py
```
Or specify a custom CSV file and base reporting currency:
```bash
python project.py data/portfolio.csv EUR
```

### 4.4 Automated Testing
Run the offline automated test suite (24 tests, synthetic NumPy/Pandas fixtures, 0 network dependencies):
```bash
pytest -v
```

To run the optional live internet smoke test against Yahoo Finance:
```bash
# PowerShell
$env:RUN_ONLINE_SMOKE="1"; pytest tests/test_smoke.py -v

# Bash
RUN_ONLINE_SMOKE=1 pytest tests/test_smoke.py -v
```

### 4.5 How to Use the Application

> **🚀 The app is live at:** [https://risk-ghost-stock-analyzer-a2gmf7buzk5ms6icvqludo.streamlit.app/](https://risk-ghost-stock-analyzer-a2gmf7buzk5ms6icvqludo.streamlit.app/)

#### Step 1 — Build Your Portfolio
- Open the app and go to the **Portfolio Diagnostic** tab.
- In the sidebar, enter each stock ticker (e.g. `AAPL`, `RELIANCE.NS`, `TSLA`), share quantity, and purchase currency.
- Click **Add Holding** to add it to your portfolio, or use the **preset buttons** (Tech Giants, Global Mixed FX) to load a sample portfolio instantly.

#### Step 2 — Run the Analysis
- Once your holdings are added, click **Analyse Portfolio**.
- The app fetches live price and volume data from Yahoo Finance and computes all risk metrics in real time.

#### Step 3 — Interpret the Risk Dashboard
- **Capital vs Risk Bar Chart** — Compare your capital weight (%) against your true risk contribution (%). A stock dominating risk far beyond its capital weight is a hidden concentration risk.
- **Risk Contribution Breakdown** — See each holding's Marginal Risk Contribution (MRC) and Component Risk Contribution (CRC) derived from Euler's theorem.
- **Liquidity Matrix** — View estimated days-to-exit and expected slippage for each position relative to Average Daily Volume (ADV).
- **Correlation Heatmap** — Identify which assets move together and amplify portfolio-level risk.
- **Drawdown History** — Visualise peak-to-trough drawdowns across your portfolio over the last 60 trading days.

#### Step 4 — Monte Carlo Simulation
- Navigate to the **Monte Carlo** section to simulate 1 000 portfolio value paths over a 60-day horizon.
- Inspect the fan-chart for best/worst/median outcomes and read the VaR (Value-at-Risk) summary below the chart.

#### Step 5 — Rebalance Suggestions
- View the **Equal-Risk Rebalancing** table to see suggested target weights that equalise each holding's risk contribution.
- Compare current vs target allocation and estimated trade sizes.

#### Step 6 — Discover Companies
- Switch to the **Company Discovery** tab and search any company name or ticker to pull key fundamentals (sector, market cap, P/E, 52-week range) directly from Yahoo Finance.

#### Step 7 — Methodology & Disclosures
- Visit the **Methodology & Disclosures** tab for a full explanation of the mathematical models, data sources, regulatory disclaimers, and academic references used in the app.

---

## 5. Regulatory & Provider Disclosures
- **Data Source:** Market data is fetched online via `yfinance`.
- **Unofficial Status:** `yfinance` is an unofficial community library and is neither affiliated with nor endorsed by Yahoo Inc.
- **Redistribution Terms:** Open-source licensing of this software does not grant unrestricted redistribution rights to underlying financial data. Quotes are intended solely for personal research, educational diagnostic analysis, and computer science study.

---

## 6. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
