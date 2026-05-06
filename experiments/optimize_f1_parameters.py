"""
Step 4 — Framework 1 Parameter Optimization Script

Performs grid search on F1 model parameters (decay, polynomial degree, max_lag)
per scenario to maximize LTC recovery_accuracy.

Frozen parameters serve as baseline; optimization results show calibration sensitivity.
"""

import json
import itertools
from pathlib import Path
import pandas as pd
import numpy as np
from ltc.data.loader import load_scenario
from ltc.data.features import build_features
from ltc.models.framework1.geometric_regression import GeometricAdstockOLS
from ltc.models.framework1.almon_regression import AlmonPDL
from ltc.evaluation.scorer import score_model

# Configuration
SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]
DATA_PATH = Path("data/raw")
RESULTS_DIR = Path("outputs/results")
RESULTS_DIR.mkdir(exist_ok=True)

# F1 Model configurations
F1_MODELS = {
    "geo_adstock": {
        "frozen_decays": {"tv": 0.90, "search": 0.20, "social": 0.82, "display": 0.65, "video": 0.88},
        "grid_range": np.arange(0.10, 1.00, 0.05).round(2),
    },
    "almon_pdl": {
        "frozen_config": {"stc_degree": 2, "ltc_degree": 3},
        "degree_grid": [2, 3, 4],
        "lag_grid": [13, 26, 39, 52],
    },
}


def optimize_geo_adstock(df: pd.DataFrame, scenario: str) -> dict:
    """
    Grid search on decay parameters per channel.
    Returns optimized decay values and recovered recovery_accuracy.
    """
    frozen_decays = F1_MODELS["geo_adstock"]["frozen_decays"]
    grid_range = F1_MODELS["geo_adstock"]["grid_range"]
    channels = ["tv", "search", "social", "display", "video"]

    best_recovery = 0.0
    best_decays = frozen_decays.copy()
    best_mape = float('inf')

    print(f"\n[geo_adstock × {scenario}] Starting grid search...")

    # Grid search per channel independently (approximation; full search would be exponential)
    for ch in channels:
        for decay_val in grid_range:
            config = {
                "decay_grid": {ch: [decay_val]},
                "feature": "impressions",
            }

            # Quick test fit
            model = GeometricAdstockOLS()
            try:
                model.fit(df, config)
                decomp = model.decompose(df)
                metrics = score_model(model, df)

                recovery = metrics.get("ltc", {}).get("total", {}).get("recovery_accuracy", 0.0)
                mape = metrics.get("ltc", {}).get("total", {}).get("mape", float('inf'))

                if recovery > best_recovery:
                    best_recovery = recovery
                    best_decays[ch] = decay_val
                    best_mape = mape
                    print(f"  {ch}: decay={decay_val:.2f} → recovery={recovery:.1f}%")
            except Exception as e:
                continue

    return {
        "model": "geo_adstock",
        "scenario": scenario,
        "optimized_decays": best_decays,
        "optimized_recovery": best_recovery,
        "optimized_mape": best_mape,
    }


def optimize_almon_pdl(df: pd.DataFrame, scenario: str) -> dict:
    """
    Grid search on polynomial degree and max_lag.
    """
    frozen_config = F1_MODELS["almon_pdl"]["frozen_config"]
    degree_grid = F1_MODELS["almon_pdl"]["degree_grid"]
    lag_grid = F1_MODELS["almon_pdl"]["lag_grid"]

    best_recovery = 0.0
    best_config = frozen_config.copy()
    best_mape = float('inf')

    print(f"\n[almon_pdl × {scenario}] Starting grid search...")

    for stc_deg, ltc_deg, max_lag in itertools.product(degree_grid, degree_grid, lag_grid):
        config = {
            "stc_max_lag": 6,
            "stc_degree": stc_deg,
            "ltc_degree": ltc_deg,
            "ltc_max_lag_override": {ch: max_lag for ch in ["tv", "search", "social", "display", "video"]},
            "feature": "impressions",
        }

        try:
            model = AlmonPDL()
            model.fit(df, config)
            metrics = score_model(model, df)

            recovery = metrics.get("ltc", {}).get("total", {}).get("recovery_accuracy", 0.0)
            mape = metrics.get("ltc", {}).get("total", {}).get("mape", float('inf'))

            if recovery > best_recovery:
                best_recovery = recovery
                best_config = {"stc_degree": stc_deg, "ltc_degree": ltc_deg, "max_lag": max_lag}
                best_mape = mape
                print(f"  stc_deg={stc_deg}, ltc_deg={ltc_deg}, max_lag={max_lag} → recovery={recovery:.1f}%")
        except Exception as e:
            continue

    return {
        "model": "almon_pdl",
        "scenario": scenario,
        "optimized_config": best_config,
        "optimized_recovery": best_recovery,
        "optimized_mape": best_mape,
    }


def main():
    """Run F1 parameter optimization across all scenarios."""

    print("=" * 100)
    print("STEP 4 — FRAMEWORK 1 PARAMETER OPTIMIZATION")
    print("=" * 100)

    optimization_results = []

    for scenario in SCENARIOS[:4]:  # S1-S4 (S5 universally fails)
        print(f"\n{'='*100}")
        print(f"SCENARIO: {scenario}")
        print(f"{'='*100}")

        # Load data
        try:
            df = load_scenario(DATA_PATH, scenario)
            print(f"✓ Loaded {len(df)} weeks of {scenario} data")
        except FileNotFoundError:
            print(f"✗ Data not found for {scenario}; skipping")
            continue

        # Optimize geo_adstock
        geo_result = optimize_geo_adstock(df, scenario)
        optimization_results.append(geo_result)

        # Optimize almon_pdl
        almon_result = optimize_almon_pdl(df, scenario)
        optimization_results.append(almon_result)

    # Save results
    output_file = RESULTS_DIR / "f1_optimization_results.json"
    with open(output_file, 'w') as f:
        json.dump(optimization_results, f, indent=2)
    print(f"\n✓ Saved F1 optimization results to {output_file}")

    # Print summary
    print("\n" + "="*100)
    print("F1 OPTIMIZATION SUMMARY")
    print("="*100)
    for result in optimization_results:
        print(f"\n{result['model']} × {result['scenario']}:")
        print(f"  Optimized recovery: {result['optimized_recovery']:.1f}%")
        print(f"  Optimized MAPE: {result['optimized_mape']:.1f}%")


if __name__ == "__main__":
    main()
