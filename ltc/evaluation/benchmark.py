"""
ltc.evaluation.benchmark — aggregate results across models × scenarios.

Reads individual result JSON files from outputs/results/ and assembles
a tidy benchmark DataFrame for use in notebooks and the paper.

Public API
----------
load_results(results_dir)                     → list[dict]
build_benchmark_table(results, metric)        → pd.DataFrame  (models × scenarios)
build_channel_table(results, metric, channel) → pd.DataFrame  (models × scenarios for one channel)
rank_models(benchmark_df)                     → pd.DataFrame  (model ranking summary)
save_benchmark(benchmark_df, output_path)     → None
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


_SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]
_CHANNELS = ["tv", "search", "social", "display", "video", "total"]


def load_results(results_dir: str | Path) -> list[dict]:
    """
    Load all experiment result JSON files from results_dir.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing {model}_{scenario}.json files.

    Returns
    -------
    list[dict] — one dict per result file, in arbitrary order.
    """
    results_dir = Path(results_dir)
    results: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            results.append(json.load(f))
    if not results:
        raise FileNotFoundError(f"No result JSON files found in {results_dir}")
    return results


def build_benchmark_table(
    results: list[dict],
    metric: str = "recovery_accuracy",
    component: str = "ltc",
    channel: str = "total",
) -> pd.DataFrame:
    """
    Build a (models × scenarios) pivot table for a given metric.

    Parameters
    ----------
    results : list[dict]
        Output of load_results() or a list of score_model() dicts.
    metric : str
        Metric name: "mape", "recovery_accuracy", "mae", "rmse",
        "correlation", "bias", "total_recovery_ratio".  Default: "recovery_accuracy".
    component : str
        "ltc" or "stc".  Default: "ltc".
    channel : str
        Channel name or "total".  Default: "total".

    Returns
    -------
    pd.DataFrame — index = model names, columns = scenario IDs.
    """
    records: list[dict] = []
    for r in results:
        model = r.get("model", "unknown")
        scenario = r.get("scenario", "unknown")
        value = (
            r.get(component, {})
             .get(channel, {})
             .get(metric, float("nan"))
        )
        records.append({"model": model, "scenario": scenario, "value": value})

    df = pd.DataFrame(records)
    if df.empty:
        return df

    pivot = df.pivot_table(index="model", columns="scenario", values="value", aggfunc="mean")
    # Reorder columns to S1..S5 order
    ordered_cols = [c for c in _SCENARIOS if c in pivot.columns]
    return pivot[ordered_cols]


def build_channel_table(
    results: list[dict],
    metric: str = "total_recovery_ratio",
    component: str = "ltc",
    scenario: str = "S1",
) -> pd.DataFrame:
    """
    Build a (models × channels) table for a given scenario.

    Parameters
    ----------
    results : list[dict]
    metric : str
    component : str  "ltc" or "stc"
    scenario : str   Scenario to filter on.

    Returns
    -------
    pd.DataFrame — index = model, columns = channel names.
    """
    scenario_results = [r for r in results if r.get("scenario") == scenario]
    records: list[dict] = []
    for r in scenario_results:
        model = r.get("model", "unknown")
        row: dict = {"model": model}
        for ch in _CHANNELS:
            row[ch] = r.get(component, {}).get(ch, {}).get(metric, float("nan"))
        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("model")
    return df[[c for c in _CHANNELS if c in df.columns]]


def rank_models(
    benchmark_df: pd.DataFrame,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """
    Rank models by their average metric across all scenarios.

    Parameters
    ----------
    benchmark_df : pd.DataFrame
        Output of build_benchmark_table().
    higher_is_better : bool
        True for recovery_accuracy/correlation; False for mape/mae/rmse.

    Returns
    -------
    pd.DataFrame with columns: model, mean, std, worst_case, rank.
    """
    summary = pd.DataFrame({
        "mean": benchmark_df.mean(axis=1),
        "std": benchmark_df.std(axis=1),
        "worst_case": benchmark_df.min(axis=1) if higher_is_better else benchmark_df.max(axis=1),
    })
    summary["rank"] = summary["mean"].rank(ascending=not higher_is_better).astype(int)
    return summary.sort_values("rank")


def save_benchmark(
    benchmark_df: pd.DataFrame,
    output_path: str | Path,
    fmt: str = "csv",
) -> None:
    """
    Save the benchmark table to a file.

    Parameters
    ----------
    benchmark_df : pd.DataFrame
    output_path : str or Path
    fmt : str
        "csv" (default) or "latex".
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        benchmark_df.to_csv(output_path)
    elif fmt == "latex":
        with open(output_path, "w") as f:
            f.write(benchmark_df.to_latex(float_format="%.1f"))
    else:
        raise ValueError(f"fmt must be 'csv' or 'latex', got '{fmt}'")

    print(f"[benchmark] Saved to {output_path}")
