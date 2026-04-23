"""
F1 Decay Search — Framework 1 parameter estimation.

Purpose
-------
Justify and validate the decay search grids used by F1 models (geo_adstock,
weibull_adstock, almon_pdl, dual_adstock) by running each model on S1 (clean
benchmark) and recording the optimal decay parameters the grid search selects.

This serves two functions for the paper:
  1. Shows that the grid bounds are wide enough to contain the truth.
  2. Establishes the "learned" decay values as fixed starting points for
     cross-experiment comparability analysis.

Estimation approach
-------------------
For each F1 model, fit on S1 (observed columns only) and extract the
per-channel decay parameters that minimise in-sample RSS.  These are
model-estimated parameters — the grid search IS the estimation procedure.

Output
------
Writes decay estimates to estimated_params.json under the key "f1".
Also appends rows to parameter_log.csv for each estimated decay.

Usage
-----
    python experiments/parameter_estimation/f1_decay_search.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ltc.data.loader import load_scenario, split_observed_truth
from ltc.models.framework1 import GeometricAdstockOLS, WeibullAdstockNLS, AlmonPDL, DualAdstockOLS

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("experiments/parameter_estimation")
CHANNELS = ["tv", "search", "social", "display", "video"]

# Ground-truth decay values (for reference — not used in fitting)
GT_DECAYS = {
    "tv": 0.90, "search": 0.30, "social": 0.82, "display": 0.65, "video": 0.88
}

# Current YAML configs (defaults)
GEO_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
DUAL_STC_GRID = [0.10, 0.20, 0.30, 0.40, 0.50]
DUAL_LTC_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def run_f1_decay_search(data_dir: Path = DATA_DIR) -> dict:
    """Fit all F1 models on S1 and extract estimated decay parameters."""
    print("=" * 60)
    print("F1 Decay Search — Scenario S1 (Clean Benchmark)")
    print("=" * 60)

    df_full = load_scenario(data_dir, "S1")
    df_obs, df_truth = split_observed_truth(df_full)

    results = {}

    # ── GeometricAdstockOLS ───────────────────────────────────────────────
    print("\n[1/4] GeometricAdstockOLS")
    config_geo = {
        "decay_grid": GEO_GRID,
        "feature": "impressions",
        "channels": CHANNELS,
        "fit_intercept": True,
    }
    model_geo = GeometricAdstockOLS()
    model_geo.fit(df_obs, config_geo)
    params_geo = model_geo.get_params()
    decays_geo = params_geo.get("channel_decays", {})
    results["geo_adstock"] = {"channel_decays": decays_geo}
    print(f"  Optimal decays: {decays_geo}")
    print(f"  True LTC decays: {GT_DECAYS}")
    for ch in CHANNELS:
        est = decays_geo.get(ch, float("nan"))
        truth = GT_DECAYS[ch]
        print(f"    {ch:10s}: estimated={est:.2f}  truth={truth:.2f}  gap={est - truth:+.2f}")

    # ── DualAdstockOLS ────────────────────────────────────────────────────
    print("\n[2/4] DualAdstockOLS")
    config_dual = {
        "stc_decay_grid": DUAL_STC_GRID,
        "ltc_decay_grid": DUAL_LTC_GRID,
        "feature": "impressions",
        "channels": CHANNELS,
    }
    model_dual = DualAdstockOLS()
    model_dual.fit(df_obs, config_dual)
    params_dual = model_dual.get_params()
    decays_dual = params_dual.get("channel_decays", {})
    results["dual_adstock"] = {"channel_decays": decays_dual}
    print(f"  Optimal STC/LTC decays: {decays_dual}")

    # ── WeibullAdstockNLS ─────────────────────────────────────────────────
    print("\n[3/4] WeibullAdstockNLS")
    config_weibull = {
        "max_lag": 20,
        "feature": "impressions",
        "channels": CHANNELS,
        "shape_bounds": [0.5, 5.0],
        "scale_bounds": [1.0, 15.0],
    }
    model_weibull = WeibullAdstockNLS()
    model_weibull.fit(df_obs, config_weibull)
    params_weibull = model_weibull.get_params()
    results["weibull_adstock"] = {"channel_params": params_weibull.get("channel_params", {})}
    print(f"  Weibull shape/scale: {params_weibull.get('channel_params', {})}")

    # ── AlmonPDL ──────────────────────────────────────────────────────────
    print("\n[4/4] AlmonPDL")
    config_almon = {
        "max_lag": 13,
        "degree": 3,
        "stc_cutoff": 4,
        "feature": "impressions",
        "channels": CHANNELS,
    }
    model_almon = AlmonPDL()
    model_almon.fit(df_obs, config_almon)
    params_almon = model_almon.get_params()
    results["almon_pdl"] = {"channel_weights": params_almon.get("channel_weights", {})}

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GRID VALIDATION SUMMARY")
    print("=" * 60)
    print("geo_adstock optimal decays vs. grid coverage:")
    for ch in CHANNELS:
        est = decays_geo.get(ch, float("nan"))
        in_grid = min(GEO_GRID) <= est <= max(GEO_GRID)
        print(f"  {ch:10s}: {est:.2f}  grid=[{min(GEO_GRID):.2f}, {max(GEO_GRID):.2f}]  covered={in_grid}")
    print("\nNote: geo_adstock fits a combined STC+LTC decay — it cannot")
    print("recover the true LTC decay (0.82-0.90) independently.")
    print("This is the core F1 limitation the paper demonstrates.")

    return results


def write_parameter_log_rows(results: dict, log_path: Path) -> None:
    """Append F1 parameter rows to parameter_log.csv."""
    rows = []
    for model_name, params in results.items():
        if "channel_decays" in params:
            for ch, decay_info in params["channel_decays"].items():
                if isinstance(decay_info, dict):
                    for k, v in decay_info.items():
                        rows.append({
                            "exp_id": "PARAM_EST",
                            "model": model_name,
                            "parameter_name": f"{ch}_{k}",
                            "value": v,
                            "assumed_or_estimated": "estimated",
                            "rationale": "Grid search RSS minimisation on S1 (observed columns only)",
                        })
                else:
                    rows.append({
                        "exp_id": "PARAM_EST",
                        "model": model_name,
                        "parameter_name": f"decay_{ch}",
                        "value": decay_info,
                        "assumed_or_estimated": "estimated",
                        "rationale": "Grid search RSS minimisation on S1 (observed columns only)",
                    })

    if rows:
        df_rows = pd.DataFrame(rows)
        header = not log_path.exists()
        df_rows.to_csv(log_path, mode="a", header=header, index=False)
        print(f"\nAppended {len(rows)} rows to {log_path}")


if __name__ == "__main__":
    results = run_f1_decay_search()
    out_file = OUTPUT_DIR / "estimated_params.json"
    if out_file.exists():
        with open(out_file) as f:
            all_params = json.load(f)
    else:
        all_params = {}
    all_params["f1"] = results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_params, f, indent=2, default=str)
    print(f"\nWrote F1 results to {out_file}")

    log_path = Path("outputs/parameter_log.csv")
    write_parameter_log_rows(results, log_path)
