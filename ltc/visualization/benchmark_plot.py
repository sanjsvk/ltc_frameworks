"""
ltc.visualization.benchmark_plot — cross-framework comparison charts.

Creates the paper-ready figures comparing all 10 models across 5 scenarios:
  - Heatmap: recovery_accuracy (models × scenarios)
  - Radar / spider chart: per-scenario framework comparison
  - Bar chart: channel-level LTC recovery ratio per scenario
  - Waterfall: bias decomposition (over vs. under attribution)

Public API
----------
plot_recovery_heatmap(benchmark_df, title, figsize, ax)
plot_scenario_radar(benchmark_df, scenarios, title, figsize)
plot_channel_recovery_bars(channel_table, scenario, title, figsize)
plot_bias_waterfall(results, scenario, channel, figsize)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from pathlib import Path


# Framework groupings for colour coding in charts
_FRAMEWORK_COLORS = {
    "geo_adstock":     "#1f77b4",  # F1 — blue family
    "weibull_adstock": "#aec7e8",
    "almon_pdl":       "#6baed6",
    "dual_adstock":    "#2171b5",
    "koyck":           "#ff7f0e",  # F2 — orange family
    "ardl":            "#fdae6b",
    "finite_dl":       "#e6550d",
    "kalman_dlm":      "#2ca02c",  # F3 — green family
    "mcmc_stock":      "#31a354",
    "bsts":            "#74c476",
}

_SCENARIO_LABELS = {
    "S1": "S1\nClean\nBenchmark",
    "S2": "S2\nSpend\nPause",
    "S3": "S3\nCollinearity",
    "S4": "S4\nStruct.\nBreak",
    "S5": "S5\nWeak LTC",
}


def plot_recovery_heatmap(
    benchmark_df: pd.DataFrame,
    title: str = "LTC Recovery Accuracy (%) — Models × Scenarios",
    figsize: tuple = (10, 7),
    ax: Axes | None = None,
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 100.0,
) -> Axes:
    """
    Heatmap of recovery accuracy across models (rows) × scenarios (columns).

    Parameters
    ----------
    benchmark_df : pd.DataFrame
        Output of build_benchmark_table(metric="recovery_accuracy").
        Index = model names, columns = scenario IDs.
    title : str
    figsize : tuple
    ax : Axes or None
    cmap : str
        Matplotlib colormap (default: "RdYlGn" — red=bad, green=good).
    vmin, vmax : float
        Colormap range.  Default: 0–100 for recovery_accuracy.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    data = benchmark_df.copy()
    # Rename scenario columns to multi-line labels
    data.columns = [_SCENARIO_LABELS.get(c, c) for c in data.columns]

    im = ax.imshow(data.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Annotate cells with values
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.iloc[i, j]
            text_color = "black" if 20 < val < 80 else "white"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    # Axis labels
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, fontsize=10)
    ax.set_yticks(range(len(data.index)))

    # Colour model names by framework
    yticklabels = ax.set_yticklabels(data.index, fontsize=10)
    for label in yticklabels:
        model = label.get_text()
        color = _FRAMEWORK_COLORS.get(model, "black")
        label.set_color(color)

    plt.colorbar(im, ax=ax, label="Recovery Accuracy (%)")
    ax.set_title(title, fontsize=13, pad=12)

    # Framework group separators
    f1_end = 3  # after dual_adstock
    f2_end = 6  # after finite_dl
    for y_sep in [f1_end - 0.5, f2_end - 0.5]:
        ax.axhline(y_sep, color="white", linewidth=2)

    return ax


def plot_scenario_radar(
    benchmark_df: pd.DataFrame,
    title: str = "Framework Comparison Across Scenarios",
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """
    Radar (spider) chart showing per-framework average recovery across scenarios.

    Parameters
    ----------
    benchmark_df : pd.DataFrame
        Models × scenarios recovery_accuracy table.
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    scenarios = [c for c in benchmark_df.columns]
    n_scenarios = len(scenarios)
    angles = np.linspace(0, 2 * np.pi, n_scenarios, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_SCENARIO_LABELS.get(s, s) for s in scenarios], fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)

    for model in benchmark_df.index:
        values = benchmark_df.loc[model, scenarios].tolist()
        values += values[:1]  # close polygon
        color = _FRAMEWORK_COLORS.get(model, "gray")
        ax.plot(angles, values, color=color, linewidth=2, label=model)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title(title, fontsize=13, pad=20)
    return fig


def plot_channel_recovery_bars(
    channel_table: pd.DataFrame,
    scenario: str = "S1",
    metric_label: str = "Total Recovery Ratio",
    title: str | None = None,
    figsize: tuple = (12, 5),
    ax: Axes | None = None,
) -> Axes:
    """
    Grouped bar chart of per-channel LTC recovery ratio for one scenario.

    Parameters
    ----------
    channel_table : pd.DataFrame
        Output of build_channel_table().  Index = models, columns = channels.
    scenario : str
        Used in the title.
    metric_label : str
    title : str or None
    figsize : tuple
    ax : Axes or None

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if title is None:
        title = f"{scenario}: {metric_label} by Channel"

    channels = [c for c in channel_table.columns if c != "total"]
    n_models = len(channel_table)
    x = np.arange(len(channels))
    width = 0.8 / n_models

    for i, model in enumerate(channel_table.index):
        vals = channel_table.loc[model, channels].to_numpy(dtype=float)
        offset = (i - n_models / 2 + 0.5) * width
        color = _FRAMEWORK_COLORS.get(model, "gray")
        ax.bar(x + offset, vals, width=width, label=model, color=color, alpha=0.85)

    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="Perfect recovery")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in channels], fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_bias_waterfall(
    results: list[dict],
    scenario: str = "S2",
    channel: str = "tv",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Horizontal bar chart of bias (over/under attribution) per model.

    Parameters
    ----------
    results : list[dict]
        List of score_model() outputs.
    scenario : str
    channel : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    scenario_results = [r for r in results if r.get("scenario") == scenario]
    models, biases = [], []
    for r in scenario_results:
        model = r.get("model", "?")
        b = r.get("ltc", {}).get(channel, {}).get("bias", float("nan"))
        models.append(model)
        biases.append(b)

    colors = ["#d62728" if b > 0 else "#1f77b4" for b in biases]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(models, biases, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Bias ($M/week) — positive = over-attribution", fontsize=11)
    ax.set_title(f"{scenario}: LTC Bias per Model — {channel.upper()}", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig
