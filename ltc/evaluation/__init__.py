"""ltc.evaluation — metrics, scoring, and benchmark aggregation."""
from .metrics import (
    mape, recovery_accuracy, mae, rmse, correlation,
    ci_coverage, bias, total_recovery_ratio, compute_all_metrics,
)
from .scorer import score_model, score_ltc_only, aggregate_channel_scores
from .benchmark import load_results, build_benchmark_table, build_channel_table, rank_models, save_benchmark

__all__ = [
    "mape", "recovery_accuracy", "mae", "rmse", "correlation",
    "ci_coverage", "bias", "total_recovery_ratio", "compute_all_metrics",
    "score_model", "score_ltc_only", "aggregate_channel_scores",
    "load_results", "build_benchmark_table", "build_channel_table", "rank_models", "save_benchmark",
]
