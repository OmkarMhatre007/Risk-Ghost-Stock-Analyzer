"""
Matplotlib Charting Engine for Risk Ghost.

Styled with the Apple Careers iconic deep-dark aesthetic:
- Deep black and dark slate card backgrounds (#000000 / #161617).
- High-contrast crisp white typography (#ffffff / #f5f5f7).
- Vibrant Apple neon accents (electric blue, cyan, purple, coral, mint).
- Clean borders, no visual clutter, generous whitespace.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from typing import Any

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
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

try:
    from src.portfolio_manager import HoldingItem
except ImportError:
    from portfolio_manager import HoldingItem

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Apple Careers Dark Palette
APPLE_BG = "#000000"
APPLE_CARD_BG = "#161617"
APPLE_TEXT = "#f5f5f7"
APPLE_MUTED = "#86868b"
APPLE_BORDER = "#2c2c2e"
APPLE_BLUE = "#2997ff"
APPLE_CYAN = "#30b0c7"
APPLE_PURPLE = "#af52de"
APPLE_CORAL = "#ff453a"
APPLE_GREEN = "#30d158"
APPLE_ORANGE = "#ff9f0a"
APPLE_GRAY_BAR = "#3a3a3c"

# Custom Apple diverging colormap: coral (-1) -> deep card background (0) -> electric blue (+1)
APPLE_CORR_CMAP = LinearSegmentedColormap.from_list(
    "apple_diverging",
    [APPLE_CORAL, "#2c2c2e", APPLE_BLUE]
)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
plt.rcParams["text.color"] = APPLE_TEXT
plt.rcParams["axes.labelcolor"] = APPLE_MUTED
plt.rcParams["xtick.color"] = APPLE_MUTED
plt.rcParams["ytick.color"] = APPLE_MUTED
plt.rcParams["axes.edgecolor"] = APPLE_BORDER
plt.rcParams["axes.linewidth"] = 0.8


def create_figure(figsize: tuple[float, float] = (9, 4.5)) -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, facecolor=APPLE_CARD_BG)
    ax.set_facecolor(APPLE_CARD_BG)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(APPLE_BORDER)
    ax.spines["bottom"].set_color(APPLE_BORDER)
    ax.tick_params(colors=APPLE_MUTED, labelsize=9)
    return fig, ax


def plot_capital_vs_risk(holdings: list[HoldingItem]) -> Figure:
    """
    Apple Careers Dark styled paired horizontal bar chart:
    Capital Weight % vs True Risk Contribution %.
    """
    fig, ax = create_figure(figsize=(9, max(3.6, len(holdings) * 0.75)))

    tickers = [h.ticker for h in holdings][::-1]
    weights = [h.weight_pct for h in holdings][::-1]
    risks = [h.risk_contribution_pct for h in holdings][::-1]

    y = np.arange(len(tickers))
    bar_height = 0.32

    # Capital Weight bars
    bars_w = ax.barh(
        y + bar_height / 2,
        weights,
        height=bar_height,
        color=APPLE_GRAY_BAR,
        label="Capital Weight %",
        edgecolor="none",
    )

    # Risk Contribution bars: vibrant coral if risk dominates, mint green if hedge/negative, else electric blue
    risk_colors = [
        APPLE_CORAL if (r - w) >= 10.0 else (APPLE_GREEN if r < 0 else APPLE_BLUE)
        for r, w in zip(risks, weights)
    ]
    bars_r = ax.barh(
        y - bar_height / 2,
        risks,
        height=bar_height,
        color=risk_colors,
        label="Risk Contribution %",
        edgecolor="none",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(tickers, fontsize=10, fontweight="bold", color=APPLE_TEXT)
    ax.set_xlabel("Percentage (%)", fontsize=10, color=APPLE_MUTED, labelpad=8)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="x", linestyle="--", alpha=0.2, color=APPLE_BORDER)

    # Value callouts on top of bars
    for bar in bars_w:
        width = bar.get_width()
        if width > 1:
            ax.text(
                width + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center",
                ha="left",
                fontsize=8.5,
                color=APPLE_MUTED,
            )

    for bar in bars_r:
        width = bar.get_width()
        if width > 1:
            ax.text(
                width + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                color=APPLE_TEXT,
            )
        elif width < 0:
            ax.text(
                width - 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center",
                ha="right",
                fontsize=8.5,
                fontweight="bold",
                color=APPLE_GREEN,
            )

    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.15),
        ncol=2,
        fontsize=9,
        labelcolor=APPLE_TEXT,
    )
    fig.tight_layout()
    return fig


def plot_drawdown_history(base_prices_df: pd.DataFrame) -> Figure:
    """
    Historical peak-to-trough drawdown time series on deep-dark card background.
    """
    fig, ax = create_figure(figsize=(9, 4.2))

    palette = [APPLE_BLUE, APPLE_CYAN, APPLE_PURPLE, APPLE_CORAL, APPLE_ORANGE, APPLE_GREEN]

    for idx, col in enumerate(base_prices_df.columns):
        series = base_prices_df[col]
        rolling_max = series.cummax()
        dd = (series - rolling_max) / rolling_max * 100.0
        color = palette[idx % len(palette)]
        ax.plot(
            dd.index,
            dd.values,
            label=col,
            color=color,
            linewidth=2.0,
            alpha=0.9,
        )

    ax.axhline(0, color=APPLE_BORDER, linewidth=1.0)
    ax.set_ylabel("Drawdown (%)", fontsize=10, color=APPLE_MUTED, labelpad=8)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(True, linestyle="--", alpha=0.2, color=APPLE_BORDER)
    ax.legend(
        frameon=False,
        loc="lower left",
        ncol=min(5, len(base_prices_df.columns)),
        fontsize=8.5,
        labelcolor=APPLE_TEXT,
    )
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> Figure:
    """
    High-contrast correlation matrix heatmap styled for dark mode.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor=APPLE_CARD_BG)
    ax.set_facecolor(APPLE_CARD_BG)

    matrix = corr_df.values
    n = len(corr_df.columns)

    # High-contrast dark diverging colormap
    ax.imshow(matrix, cmap=APPLE_CORR_CMAP, vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr_df.columns, fontsize=9.5, fontweight="bold", color=APPLE_TEXT)
    ax.set_yticklabels(corr_df.columns, fontsize=9.5, fontweight="bold", color=APPLE_TEXT)

    ax.tick_params(top=False, bottom=False, left=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            text_color = "#ffffff" if abs(val) > 0.45 else APPLE_MUTED
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9.5,
                fontweight="normal",
            )

    fig.tight_layout()
    return fig


def plot_monte_carlo(mc_results: dict[str, Any], currency: str = "USD") -> Figure:
    """
    Apple Careers dark fan chart visualization of covariance-aware Monte Carlo projections.
    """
    fig, ax = create_figure(figsize=(9, 4.4))

    paths = mc_results["paths"]
    median_path = mc_results["median_path"]
    p5_path = mc_results["p5_path"]
    p95_path = mc_results["p95_path"]
    p25_path = mc_results["p25_path"]
    p75_path = mc_results["p75_path"]

    horizon = paths.shape[1] - 1
    days = np.arange(horizon + 1)

    # 5th to 95th percentile band (subtle cyan/blue glow)
    ax.fill_between(
        days,
        p5_path,
        p95_path,
        color=APPLE_CYAN,
        alpha=0.15,
        label="5th – 95th Percentile Confidence",
    )

    # 25th to 75th percentile band
    ax.fill_between(
        days,
        p25_path,
        p75_path,
        color=APPLE_BLUE,
        alpha=0.28,
        label="25th – 75th Interquartile Range",
    )

    # Median projection line
    ax.plot(
        days,
        median_path,
        color=APPLE_BLUE,
        linewidth=2.4,
        label="Median Projection",
    )

    ax.set_xlabel("Projection Horizon (Trading Days)", fontsize=10, color=APPLE_MUTED, labelpad=8)
    ax.set_ylabel(f"Simulated Valuation ({currency})", fontsize=10, color=APPLE_MUTED, labelpad=8)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, linestyle="--", alpha=0.2, color=APPLE_BORDER)

    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=8.5,
        labelcolor=APPLE_TEXT,
    )
    fig.tight_layout()
    return fig


def plot_liquidity_matrix(holdings: list[HoldingItem]) -> Figure:
    """
    Visualizes days to exit against estimated slippage % on deep-dark card background.
    Includes smart staggered annotation offsets to prevent label collision on overlapping liquid positions.
    """
    fig, ax = create_figure(figsize=(9, 4.2))

    days = [h.days_to_exit for h in holdings]
    slippage = [h.slippage_pct for h in holdings]
    tickers = [h.ticker for h in holdings]

    colors = [APPLE_CORAL if (d >= 5.0 or s >= 3.0) else APPLE_CYAN for d, s in zip(days, slippage)]

    ax.scatter(days, slippage, color=colors, s=140, alpha=0.9, edgecolors="#ffffff", linewidth=0.6)

    # 5-day warning threshold line
    ax.axvline(5.0, color=APPLE_ORANGE, linestyle=":", alpha=0.5, label="5-Day Participation Limit")

    # Group overlapping points to stagger text annotations
    cluster_counts: dict[tuple[int, int], int] = {}
    stagger_offsets = [
        (6, 2),
        (6, -14),
        (6, 18),
        (6, -28),
        (24, 4),
        (24, -12),
        (24, 20),
        (24, -26),
    ]

    for i, txt in enumerate(tickers):
        # Quantize coordinates to identify overlapping clusters
        coord_key = (int(round(days[i] * 10)), int(round(slippage[i] * 20)))
        idx_in_cluster = cluster_counts.get(coord_key, 0)
        cluster_counts[coord_key] = idx_in_cluster + 1

        ox, oy = stagger_offsets[idx_in_cluster % len(stagger_offsets)]

        ax.annotate(
            f" {txt}",
            (days[i], slippage[i]),
            fontsize=9.0,
            fontweight="bold",
            color=APPLE_TEXT,
            xytext=(ox, oy),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color=APPLE_MUTED, lw=0.6, alpha=0.5) if idx_in_cluster > 0 else None,
        )

    ax.set_xlabel("Days to Liquidate (20% ADV Limit)", fontsize=10, color=APPLE_MUTED, labelpad=8)
    ax.set_ylabel("Expected Slippage (%)", fontsize=10, color=APPLE_MUTED, labelpad=8)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=1))
    ax.set_xlim(left=max(0.0, min(days) * 0.8 - 0.1))
    ax.grid(True, linestyle="--", alpha=0.2, color=APPLE_BORDER)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5, labelcolor=APPLE_TEXT)
    fig.tight_layout()
    return fig
