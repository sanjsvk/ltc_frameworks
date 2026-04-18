"""ltc.visualization — plotting utilities for decomposition and benchmarking."""
from .decomposition import plot_contribution_area, plot_ltc_vs_truth, save_figure
from .brand_stock_plot import plot_stock_evolution, plot_spend_pause_zoom
from .benchmark_plot import (
    plot_recovery_heatmap,
    plot_scenario_radar,
    plot_channel_recovery_bars,
    plot_bias_waterfall,
)

__all__ = [
    "plot_contribution_area",
    "plot_ltc_vs_truth",
    "save_figure",
    "plot_stock_evolution",
    "plot_spend_pause_zoom",
    "plot_recovery_heatmap",
    "plot_scenario_radar",
    "plot_channel_recovery_bars",
    "plot_bias_waterfall",
]
