"""
ltc.visualization.brand_stock_plot — latent brand stock evolution plots.

Visualises how the latent brand equity stock builds and decays over time,
comparing model estimates against the known ground-truth stock series.
These plots are central to the paper's S2 (spend pause) analysis.

Public API
----------
plot_stock_evolution(df, decomp, truth, channels, title, figsize)
plot_spend_pause_zoom(df, truth, channels, pause_start, pause_end, figsize)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from ltc.transforms.brand_stock import brand_stock_dynamics, DEFAULT_PARAMS

_CHANNEL_COLORS = {
    "tv": "#1f77b4",
    "search": "#ff7f0e",
    "social": "#2ca02c",
    "display": "#d62728",
    "video": "#9467bd",
}


def plot_stock_evolution(
    df: pd.DataFrame,
    truth: pd.DataFrame,
    estimated_ltc: pd.DataFrame | None = None,
    channels: list[str] | None = None,
    title: str = "Brand Stock Evolution",
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Plot latent brand stock over time for each channel.

    Shows:
      - Ground-truth stock series (brand_stock_{ch}_true)
      - Implied stock from model's LTC estimate (ltc_estimated / ltc_coef)
      - Spend series (shaded) for context

    Parameters
    ----------
    df : pd.DataFrame
        Observed DataFrame (spend and date columns).
    truth : pd.DataFrame
        Ground-truth DataFrame (brand_stock_{ch}_true).
    estimated_ltc : pd.DataFrame or None
        model.decompose() output; if provided, shows inferred stock.
    channels : list[str] or None
        Channels with tracked stock (default: tv, video, social — the three
        channels with brand_stock_{ch}_true columns in the data).
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if channels is None:
        channels = ["tv", "video", "social"]

    n = len(channels)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14)
    if n == 1:
        axes = [axes]

    x = df["date"].to_numpy() if "date" in df.columns else np.arange(len(df))

    for i, ch in enumerate(channels):
        ax = axes[i]
        true_col = f"brand_stock_{ch}_true"

        # Shade spend as background
        spend_col = f"spend_{ch}"
        if spend_col in df.columns:
            ax_twin = ax.twinx()
            ax_twin.fill_between(x, 0, df[spend_col].to_numpy(),
                                 alpha=0.15, color=_CHANNEL_COLORS.get(ch, "gray"),
                                 label=f"Spend {ch}")
            ax_twin.set_ylabel("Spend ($M)", fontsize=8, color="gray")
            ax_twin.tick_params(axis="y", labelcolor="gray", labelsize=8)

        # True stock
        if true_col in truth.columns:
            ax.plot(x, truth[true_col].to_numpy(), color=_CHANNEL_COLORS.get(ch, "blue"),
                    linewidth=2.0, label="True Stock", zorder=3)

        # Inferred stock from model LTC (LTC / ltc_coef)
        if estimated_ltc is not None:
            est_col = f"ltc_{ch}"
            ltc_coef = DEFAULT_PARAMS.get(ch, {}).get("ltc_coef", 1.0)
            if est_col in estimated_ltc.columns and ltc_coef > 0:
                inferred_stock = estimated_ltc[est_col].to_numpy() / ltc_coef
                ax.plot(x, inferred_stock, color="red", linewidth=1.5,
                        linestyle="--", label="Inferred Stock (model)", zorder=4)

        ax.set_ylabel(f"{ch.upper()} Stock", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Date", fontsize=10)
    plt.tight_layout()
    return fig


def plot_spend_pause_zoom(
    df: pd.DataFrame,
    truth: pd.DataFrame,
    estimated_ltc_dict: dict[str, pd.DataFrame] | None = None,
    channels: list[str] | None = None,
    pause_start: int = 100,
    pause_end: int = 120,
    title: str = "S2: LTC During Spend Pause",
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Zoom into the spend pause period (S2 scenario) showing:
      - True LTC continuing after spend drops to zero
      - Framework estimates during pause (can reveal systematic under-estimation)

    Parameters
    ----------
    df : pd.DataFrame
        Observed scenario DataFrame.
    truth : pd.DataFrame
        Ground-truth DataFrame.
    estimated_ltc_dict : dict[str, pd.DataFrame] or None
        {model_name: decompose_output} mapping for overlay comparison.
    channels : list[str] or None
        Channels to show (default: tv, video).
    pause_start : int
        Week index where spend pause begins.
    pause_end : int
        Week index where spend pause ends.
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if channels is None:
        channels = ["tv", "video"]

    zoom = slice(max(0, pause_start - 10), min(len(df), pause_end + 10))
    x = df["date"].iloc[zoom].to_numpy() if "date" in df.columns else np.arange(len(df))[zoom]

    n = len(channels)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    fig.suptitle(title, fontsize=14)
    if n == 1:
        axes = [axes]

    model_colors = ["red", "orange", "purple", "green", "brown"]

    for i, ch in enumerate(channels):
        ax = axes[i]

        # Shade pause window
        pause_x_start = df["date"].iloc[pause_start] if "date" in df.columns else pause_start
        pause_x_end = df["date"].iloc[min(pause_end, len(df) - 1)] if "date" in df.columns else pause_end
        ax.axvspan(pause_x_start, pause_x_end, alpha=0.12, color="gray", label="Spend Pause")

        # True LTC
        true_col = f"ltc_{ch}_true"
        if true_col in truth.columns:
            ax.plot(x, truth[true_col].iloc[zoom].to_numpy(),
                    color="black", linewidth=2.0, label="True LTC", zorder=4)

        # Model estimates
        if estimated_ltc_dict:
            for j, (model_name, decomp) in enumerate(estimated_ltc_dict.items()):
                est_col = f"ltc_{ch}"
                if est_col in decomp.columns:
                    ax.plot(x, decomp[est_col].iloc[zoom].to_numpy(),
                            color=model_colors[j % len(model_colors)],
                            linewidth=1.5, linestyle="--", label=model_name)

        ax.set_title(f"{ch.upper()} LTC", fontsize=12)
        ax.set_ylabel("LTC ($M)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig
