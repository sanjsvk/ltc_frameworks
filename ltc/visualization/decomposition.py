"""
ltc.visualization.decomposition — contribution waterfall and area charts.

Shows how observed sales decompose into baseline, STC, and LTC components,
with optional ground-truth overlay.

Public API
----------
plot_contribution_area(df, decomp, truth, channels, title, ax)
plot_contribution_waterfall(decomp, truth, channels, title, ax)
plot_ltc_vs_truth(decomp, truth, channels, title, axes)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
from pathlib import Path


_CHANNEL_COLORS = {
    "tv": "#1f77b4",
    "search": "#ff7f0e",
    "social": "#2ca02c",
    "display": "#d62728",
    "video": "#9467bd",
}
_BASELINE_COLOR = "#aec7e8"
_TRUTH_COLOR = "black"


def plot_contribution_area(
    df: pd.DataFrame,
    decomp: pd.DataFrame,
    truth: pd.DataFrame | None = None,
    channels: list[str] | None = None,
    component: str = "ltc",
    title: str = "Sales Decomposition",
    ax: Axes | None = None,
    figsize: tuple = (14, 6),
) -> Axes:
    """
    Stacked area chart of sales decomposition (baseline + STC/LTC per channel).

    Parameters
    ----------
    df : pd.DataFrame
        Observed scenario DataFrame (for date index and observed sales).
    decomp : pd.DataFrame
        Output of model.decompose().
    truth : pd.DataFrame or None
        Ground-truth DataFrame; if provided, overlays true LTC as a line.
    channels : list[str] or None
        Channels to include.  Default: all 5.
    component : str
        "ltc" or "stc" — which contribution type to stack.
    title : str
        Chart title.
    ax : matplotlib.axes.Axes or None
        Existing axes to plot on.  If None, creates a new figure.
    figsize : tuple

    Returns
    -------
    Axes
    """
    if channels is None:
        channels = ["tv", "search", "social", "display", "video"]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Build x-axis: use dates if available, else integer index
    x = df["date"].to_numpy() if "date" in df.columns else np.arange(len(decomp))

    # Stack layers bottom-up: baseline → STC channels → LTC channels
    bottom = np.zeros(len(decomp))
    ax.fill_between(x, bottom, bottom + decomp["baseline"].to_numpy(),
                    alpha=0.6, color=_BASELINE_COLOR, label="Baseline")
    bottom += decomp["baseline"].to_numpy()

    # Add STC if plotting ltc (show full decomposition)
    if component == "ltc":
        for ch in channels:
            stc_col = f"stc_{ch}"
            if stc_col in decomp.columns:
                stc = decomp[stc_col].to_numpy()
                ax.fill_between(x, bottom, bottom + stc, alpha=0.5,
                                color=_CHANNEL_COLORS.get(ch, "gray"), label=f"STC {ch}")
                bottom += stc

    for ch in channels:
        col = f"{component}_{ch}"
        if col in decomp.columns:
            vals = decomp[col].to_numpy()
            ax.fill_between(x, bottom, bottom + vals, alpha=0.8,
                            color=_CHANNEL_COLORS.get(ch, "gray"), label=f"LTC {ch}")
            bottom += vals

    # Observed sales line
    if "net_sales_observed" in df.columns:
        ax.plot(x, df["net_sales_observed"].to_numpy(), color="black",
                linewidth=1.5, label="Observed Sales", zorder=5)

    # Ground-truth total LTC line
    if truth is not None:
        ltc_true_cols = [f"ltc_{ch}_true" for ch in channels if f"ltc_{ch}_true" in truth.columns]
        if ltc_true_cols:
            true_total_ltc = truth[ltc_true_cols].sum(axis=1).to_numpy()
            ax.plot(x, true_total_ltc, color="red", linewidth=1.5,
                    linestyle="--", label="True Total LTC", zorder=6)

    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Net Sales ($M)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1f M"))
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    return ax


def plot_ltc_vs_truth(
    decomp: pd.DataFrame,
    truth: pd.DataFrame,
    channels: list[str] | None = None,
    title: str = "Estimated vs. True LTC by Channel",
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Grid of per-channel plots comparing estimated LTC vs. ground-truth LTC.

    Parameters
    ----------
    decomp : pd.DataFrame
        Output of model.decompose().
    truth : pd.DataFrame
        Ground-truth DataFrame.
    channels : list[str] or None
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if channels is None:
        channels = ["tv", "search", "social", "display", "video"]

    n = len(channels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=14)

    x = np.arange(len(decomp))

    for i, ch in enumerate(channels):
        ax = axes[i // ncols][i % ncols]
        est_col = f"ltc_{ch}"
        true_col = f"ltc_{ch}_true"

        if est_col in decomp.columns:
            ax.plot(x, decomp[est_col].to_numpy(), label="Estimated", color=_CHANNEL_COLORS.get(ch, "blue"))
        if true_col in truth.columns:
            ax.plot(x, truth[true_col].to_numpy(), label="True", color=_TRUTH_COLOR,
                    linestyle="--", linewidth=1.2)

        ax.set_title(ch.upper(), fontsize=11)
        ax.set_ylabel("LTC ($M)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, output_path: str | Path, dpi: int = 150) -> None:
    """Save a matplotlib figure to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"[viz] Saved figure to {output_path}")
