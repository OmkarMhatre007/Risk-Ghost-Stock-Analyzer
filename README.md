# Risk Ghost Stock Analyzer: Exposing Latent Concentration Risk and Market Impact

#### Video Demo: <YOUR_YOUTUBE_OR_LOOM_VIDEO_URL_HERE>

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

---

## 3. Mathematical Methodology

### 3.1 Euler Homogeneous Risk Decomposition
Portfolio volatility $\sigma_p = \sqrt{W^T \Sigma W}$ is homogeneous of degree 1.
$$\text{MRC} = \frac{\Sigma W}{\sigma_p}$$
$$\text{CRC}_i = w_i \cdot \text{MRC}_i = \frac{w_i (\Sigma W)_i}{\sigma_p}, \quad \sum_{i=1}^n \text{CRC}_i = \sigma_p$$
$$RC_{\%, i} = \frac{\text{CRC}_i}{\sigma_p} \times 100\% = \frac{w_i (\Sigma W)_i}{W^T \Sigma W} \times 100\%, \quad \sum_{i=1}^n RC_{\%, i} = 100\%$$

### 3.2 Almgren-Chriss Square-Root Liquidity Model
$$\text{Days to Exit} = \frac{\text{Shares}}{0.20 \times \text{ADV}}$$
$$\text{Expected Slippage \%} = 0.05\% + 0.5 \times \sigma_{\text{daily}} \times \sqrt{\frac{\text{Shares}}{\text{ADV}}}$$
*Note: Preserves actual fractional trading days (e.g. `< 0.01 d`). Slippage is an illustrative heuristic estimate (incorporating a 0.05% retail spread baseline), not a guaranteed broker execution quote.*

### 3.3 Illustrative Bounded Risk-Balancing Adjustment
- Assets acting as hedges ($RC_i \le 0$) maintain neutral scaling ($1.0\times$).
- Modest risk contributors ($0 < RC_i < 1\%$) are capped at $1.5\times$.
- High risk contributors are dampened using a square-root scaling ratio: $\text{multiplier}_i = \text{clip}\left(\sqrt{\frac{\text{Target } RC}{RC_i}}, 0.5, 2.0\right)$.
- Position constraints are iteratively projected onto $[w_{\min}=1\%, w_{\max}=30\%]$ ensuring $\sum_i w_i = 100\%$.
- Recalculates and displays resulting post-tilt portfolio volatility and risk contributions.
*Note: This is an illustrative heuristic adjustment toward risk parity, not an exact non-linear optimizer.*

### 3.4 Maximum Drawdown
$$\text{MDD} = \min_{t} \left( \frac{P_t - \max_{0 \le \tau \le t} P_\tau}{\max_{0 \le \tau \le t} P_\tau} \right) \times 100\%$$

### 3.5 Covariance-Aware Monte Carlo Simulation
$$R_{\text{sim}} = \mu + Z L^T, \quad \Sigma_{\text{PSD}} = V \Lambda_{\text{PSD}} V^T = L L^T, \quad Z \sim \mathcal{N}(0, I)$$
Uses eigenvalue spectral clipping ($\max(\lambda_i, 10^{-8})$) to guarantee positive semi-definite covariance, ensuring robust Cholesky decomposition across all asset combinations over 60 trading days.

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
Analyze the default `portfolio.csv`:
```bash
python project.py
```
Or specify a custom CSV file and base reporting currency:
```bash
python project.py my_portfolio.csv EUR
```

### 4.4 Automated Testing
Run the offline automated test suite (24 tests, synthetic NumPy/Pandas fixtures, 0 network dependencies):
```bash
pytest test_project.py -v
```

To run the optional live internet smoke test against Yahoo Finance:
```bash
# PowerShell
$env:RUN_ONLINE_SMOKE="1"; pytest test_smoke.py -v

# Bash
RUN_ONLINE_SMOKE=1 pytest test_smoke.py -v
```

---

## 5. Regulatory & Provider Disclosures
- **Data Source:** Market data is fetched online via `yfinance`.
- **Unofficial Status:** `yfinance` is an unofficial community library and is neither affiliated with nor endorsed by Yahoo Inc.
- **Redistribution Terms:** Open-source licensing of this software does not grant unrestricted redistribution rights to underlying financial data. Quotes are intended solely for personal research, educational diagnostic analysis, and computer science study.
