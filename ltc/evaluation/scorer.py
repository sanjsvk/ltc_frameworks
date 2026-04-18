"""
ltc.evaluation.scorer — score a fitted model's decomposition against ground truth.

Public API
----------
score_model(decomposition, truth_df, channels)  → dict
score_ltc_only(decomposition, truth_df, channels) → dict
score_stc_only(decomposition, truth_df, channels) → dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ltc.evaluation.metrics import compute_all_metrics, total_recovery_ratio

_CHANNELS = ["tv", "search", "social", "display", "video"]


def score_model(
    decomposition: pd.DataFrame,
    truth_df: pd.DataFrame,
    channels: list[str] | None = None,
    model_name: str = "unknown",
    scenario: str = "unknown",
) -> dict:
    """
    Score a model's decompose() output against ground-truth columns.

    Parameters
    ----------
    decomposition : pd.DataFrame
        Output of model.decompose() — must contain ltc_{ch} and stc_{ch} columns.
    truth_df : pd.DataFrame
        Ground-truth DataFrame from loader.split_observed_truth()[1].
        Must contain ltc_{ch}_true, stc_{ch}_true columns.
    channels : list[str] or None
        Channels to evaluate.  Default: all 5.
    model_name : str
        Model identifier for the results dict.
    scenario : str
        Scenario identifier for the results dict.

    Returns
    -------
    dict with structure:
        {
          "model": str,
          "scenario": str,
          "ltc": {
              "total": {mape, recovery_accuracy, mae, ...},
              "tv":    {mape, ...},
              "search": ...,
              ...
          },
          "stc": {
              "total": {...},
              "tv":    {...}, ...
          }
        }
    """
    if channels is None:
        channels = _CHANNELS

    results: dict = {"model": model_name, "scenario": scenario, "ltc": {}, "stc": {}}

    # --- LTC scoring ---
    ltc_estimated_total = np.zeros(len(decomposition))
    ltc_true_total = np.zeros(len(truth_df))

    for ch in channels:
        est_col = f"ltc_{ch}"
        true_col = f"ltc_{ch}_true"

        if est_col not in decomposition.columns or true_col not in truth_df.columns:
            continue

        est = decomposition[est_col].to_numpy(float)
        true = truth_df[true_col].to_numpy(float)

        # Align lengths (trim to shorter if needed)
        n = min(len(est), len(true))
        est, true = est[:n], true[:n]

        results["ltc"][ch] = compute_all_metrics(est, true)

        ltc_estimated_total[:n] += est
        ltc_true_total[:n] += true

    results["ltc"]["total"] = compute_all_metrics(
        ltc_estimated_total[:n], ltc_true_total[:n]
    )

    # --- STC scoring ---
    stc_estimated_total = np.zeros(len(decomposition))
    stc_true_total = np.zeros(len(truth_df))

    for ch in channels:
        est_col = f"stc_{ch}"
        true_col = f"stc_{ch}_true"

        if est_col not in decomposition.columns or true_col not in truth_df.columns:
            continue

        est = decomposition[est_col].to_numpy(float)
        true = truth_df[true_col].to_numpy(float)
        n = min(len(est), len(true))
        est, true = est[:n], true[:n]

        results["stc"][ch] = compute_all_metrics(est, true)
        stc_estimated_total[:n] += est
        stc_true_total[:n] += true

    results["stc"]["total"] = compute_all_metrics(
        stc_estimated_total[:n], stc_true_total[:n]
    )

    # --- Scenario-specific diagnostics ---
    results["diagnostics"] = _compute_scenario_diagnostics(
        decomposition, truth_df, channels, scenario
    )

    return results


def _compute_scenario_diagnostics(
    decomposition: pd.DataFrame,
    truth_df: pd.DataFrame,
    channels: list[str],
    scenario: str,
) -> dict:
    """
    Compute scenario-specific diagnostic metrics.

    S2: Check LTC recovery during spend pause (weeks 104-112).
    S4: Check LTC recovery pre-break (weeks 0-103) vs post-break (weeks 104-261).
    """
    diag: dict = {}

    if scenario == "S2":
        # Evaluate LTC recovery during spend pause weeks 104-112
        pause_slice = slice(104, 113)
        for ch in ["tv", "video"]:
            est_col = f"ltc_{ch}"
            true_col = f"ltc_{ch}_true"
            if est_col in decomposition.columns and true_col in truth_df.columns:
                est_pause = decomposition[est_col].iloc[pause_slice].to_numpy(float)
                true_pause = truth_df[true_col].iloc[pause_slice].to_numpy(float)
                diag[f"ltc_{ch}_pause_recovery_ratio"] = total_recovery_ratio(est_pause, true_pause)

    elif scenario == "S4":
        # Evaluate LTC in pre-break (weeks 0-103) and post-break (weeks 104+)
        for ch in ["tv", "video"]:
            est_col = f"ltc_{ch}"
            true_col = f"ltc_{ch}_true"
            if est_col in decomposition.columns and true_col in truth_df.columns:
                for label, slc in [("pre_break", slice(0, 104)), ("post_break", slice(104, None))]:
                    est_s = decomposition[est_col].iloc[slc].to_numpy(float)
                    true_s = truth_df[true_col].iloc[slc].to_numpy(float)
                    diag[f"ltc_{ch}_{label}_recovery_ratio"] = total_recovery_ratio(est_s, true_s)

    return diag


def score_ltc_only(
    decomposition: pd.DataFrame,
    truth_df: pd.DataFrame,
    channels: list[str] | None = None,
) -> dict[str, dict]:
    """
    Convenience function: return only the LTC metrics per channel.

    Returns
    -------
    dict[channel → metrics_dict]  plus "total" key.
    """
    full = score_model(decomposition, truth_df, channels)
    return full["ltc"]


def aggregate_channel_scores(
    scores: dict, metric: str = "mape"
) -> dict[str, float]:
    """
    Extract a single metric across all channels from a score_model() result.

    Parameters
    ----------
    scores : dict
        Output of score_model().
    metric : str
        Metric name (e.g., "mape", "recovery_accuracy", "total_recovery_ratio").

    Returns
    -------
    dict[channel → float]
    """
    return {
        ch: scores["ltc"].get(ch, {}).get(metric, float("nan"))
        for ch in _CHANNELS + ["total"]
        if ch in scores["ltc"]
    }
