"""
Impact Assessment: Quantify Effect of 10 Structural Issues on Paper Claims

This script:
1. Loads all 5 scenarios (S1-S5) from data/raw/
2. Runs all 10 models on each scenario
3. Extracts recovery %, MAPE %, and channel-level breakdown
4. Compares against published paper values (Section 4-7)
5. Identifies which issues affect results and paper claims
6. Quantifies impact magnitude
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add ltc to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ltc.data.loader import load_scenario
from ltc.data.features import build_features
from ltc.evaluation.scorer import score_model

# Model imports
from ltc.models.framework1.geometric_regression import GeometricAdstockOLS
from ltc.models.framework1.weibull_regression import WeibullAdstockNLS
from ltc.models.framework1.almon_regression import AlmonPDL
from ltc.models.framework1.dual_adstock import DualAdstockOLS

from ltc.models.framework2.koyck_model import KoyckModel
from ltc.models.framework2.ardl_model import ARDLModel
from ltc.models.framework2.finite_dl_model import FiniteDLModel

from ltc.models.framework3.kalman_dlm import KalmanDLM
from ltc.models.framework3.mcmc_latent_stock import MCMCLatentStock
from ltc.models.framework3.bayesian_sts import BayesianStructuralTS

# Model registry
MODEL_REGISTRY = {
    # Framework 1
    "geo_adstock": GeometricAdstockOLS,
    "weibull_adstock": WeibullAdstockNLS,
    "almon_pdl": AlmonPDL,
    "dual_adstock": DualAdstockOLS,
    # Framework 2
    "koyck": KoyckModel,
    "ardl": ARDLModel,
    "finite_dl": FiniteDLModel,
    # Framework 3
    "kalman_dlm": KalmanDLM,
    "mcmc_stock": MCMCLatentStock,
    "bsts": BayesianStructuralTS,
}

# Published baseline from Section 4, Table 3 (S1 baseline)
PUBLISHED_S1_BASELINE = {
    "bsts": {"recovery": 82.4, "mape": 17.6},
    "kalman_dlm": {"recovery": 82.0, "mape": 18.0},
    "geo_adstock": {"recovery": 69.9, "mape": 30.1},
    "mcmc_stock": {"recovery": 72.6, "mape": 27.4},
    "finite_dl": {"recovery": 50.3, "mape": 49.7},
    "koyck": {"recovery": 46.4, "mape": 53.6},
    "almon_pdl": {"recovery": 42.6, "mape": 57.4},
    "weibull_adstock": {"recovery": 10.5, "mape": 89.5},
    "ardl": {"recovery": 0.0, "mape": 316.8},
    "dual_adstock": {"recovery": 0.0, "mape": 789.9},
}

# Load configurations
BASE_CONFIG = {
    "feature": "impressions",
    "channels": ["tv", "search", "social", "display", "video"],
}

def get_config_for_model(model_name: str) -> dict:
    """Get optimized config per model (frozen S1 parameters)."""
    config = BASE_CONFIG.copy()

    if model_name == "geo_adstock":
        config.update({
            "decay_grid": [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90],
            "max_lag_override": {"tv": 52, "search": 8, "social": 26, "display": 16, "video": 52},
        })
    elif model_name == "weibull_adstock":
        config.update({
            "max_lag": 52,
            "shape_bounds": (0.8, 3.0),
            "scale_bounds": (1.0, 15.0),
            "max_lag_override": {"tv": 52, "search": 8, "social": 26, "display": 16, "video": 52},
        })
    elif model_name == "dual_adstock":
        config.update({
            "stc_decay_grid": [0.1, 0.2, 0.3, 0.4, 0.5],
            "ltc_decay_grid": [0.5, 0.6, 0.7, 0.8, 0.9],
        })
    elif model_name == "almon_pdl":
        config.update({
            "ltc_degree": 2,
            "max_lag": 52,
            "max_lag_override": {"tv": 52, "search": 8, "social": 26, "display": 16, "video": 52},
        })
    elif model_name == "koyck":
        config.update({
            "max_lag": 52,
            "ltc_degree": 2,
        })
    elif model_name == "ardl":
        config.update({
            "ltc_degree": 3,
            "stc_lags": 1,
            "exog_lags": 0,
        })
    elif model_name == "finite_dl":
        config.update({
            "max_lag": 52,
            "ltc_degree": 2,
            "max_lag_override": {"tv": 52, "search": 8, "social": 26, "display": 16, "video": 52},
        })
    elif model_name == "kalman_dlm":
        config.update({
            "level": True,
            "trend": True,
            "seasonal": None,  # Note: missing seasonal for non-S1 scenarios
        })
    elif model_name == "mcmc_stock":
        config.update({
            "delta_prior": ("logit_normal", -2.0, 0.8),
            "build_rate_prior": ("lognormal", 0.0, 0.5),
            "ltc_coef_prior": ("normal", 0.5, 1.0),
            "chains": 4,
            "tune": 1500,
            "draws": 1000,
            "target_accept": 0.99,
        })
    elif model_name == "bsts":
        config.update({
            "level": True,
            "trend": True,
            "seasonal_periods": 52,
        })

    return config

def run_model(model_class, df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Run a model and extract metrics.

    Returns:
        (decomposition_df, metrics_dict)
    """
    try:
        model = model_class()
        model.fit(df, config)
        decomp = model.decompose(df)

        # Score
        try:
            metrics = score_model(model, df)
        except Exception as e:
            print(f"  [!] Scoring failed: {e}")
            metrics = {"recovery_accuracy": 0.0, "mape": 999.0, "error": str(e)}

        return decomp, metrics
    except Exception as e:
        print(f"  [!] Model fit failed: {e}")
        return None, {"recovery_accuracy": 0.0, "mape": 999.0, "error": str(e)}

def extract_channel_recovery(df_truth: pd.DataFrame, decomp_df: pd.DataFrame,
                             channel: str) -> Dict[str, float]:
    """Extract per-channel recovery metrics."""
    ltc_true_col = f"ltc_{channel}_true"
    ltc_est_col = f"ltc_{channel}"

    if ltc_true_col not in df_truth.columns or ltc_est_col not in decomp_df.columns:
        return {"recovery": 0.0, "mape": 0.0, "bias": 0.0, "correlation": 0.0}

    y_true = df_truth[ltc_true_col].values
    y_est = decomp_df[ltc_est_col].values

    # Recovery accuracy
    mae_true = np.mean(np.abs(y_true))
    mae_error = np.mean(np.abs(y_true - y_est))
    recovery = 100.0 * (1.0 - mae_error / mae_true) if mae_true > 0 else 0.0

    # MAPE
    mape = np.mean(np.abs((y_true - y_est) / (np.abs(y_true) + 1e-6))) * 100.0

    # Bias
    bias = np.mean(y_est - y_true)

    # Correlation
    if np.std(y_true) > 1e-6 and np.std(y_est) > 1e-6:
        corr = np.corrcoef(y_true, y_est)[0, 1]
    else:
        corr = 0.0

    return {
        "recovery": recovery,
        "mape": mape,
        "bias": bias,
        "correlation": corr,
    }

def main():
    output_dir = Path("/c/github/ltc/impact_assessment")
    output_dir.mkdir(exist_ok=True)

    # Load all scenarios
    data_dir = Path("/c/github/ltc/data/raw")
    scenarios = {}
    for scenario_name in ["S1", "S2", "S3", "S4", "S5"]:
        csv_path = data_dir / f"{scenario_name}.csv"
        print(f"Loading {scenario_name}...")
        scenarios[scenario_name] = pd.read_csv(csv_path)

    # Results accumulator
    results = {
        "s1_baseline": {},
        "per_scenario": {},
        "impact_summary": [],
        "issue_analysis": {},
    }

    # Test S1 baseline against published
    print("\n" + "="*80)
    print("PHASE 1: S1 BASELINE COMPARISON")
    print("="*80)

    df_s1 = scenarios["S1"]
    s1_results = {}

    for model_name in sorted(MODEL_REGISTRY.keys()):
        print(f"\nTesting {model_name}...")
        model_class = MODEL_REGISTRY[model_name]
        config = get_config_for_model(model_name)

        decomp_df, metrics = run_model(model_class, df_s1, config)

        recovery = metrics.get("recovery_accuracy", 0.0)
        mape = metrics.get("mape", 999.0)

        published = PUBLISHED_S1_BASELINE.get(model_name, {})
        pub_recovery = published.get("recovery", 0.0)
        pub_mape = published.get("mape", 0.0)

        delta_recovery = recovery - pub_recovery
        delta_mape = mape - pub_mape

        print(f"  Recovery: {recovery:.1f}% (published: {pub_recovery:.1f}%, Δ: {delta_recovery:+.1f}pp)")
        print(f"  MAPE:     {mape:.1f}% (published: {pub_mape:.1f}%, Δ: {delta_mape:+.1f}pp)")

        if abs(delta_recovery) > 5 or abs(delta_mape) > 10:
            print(f"  [!] SIGNIFICANT DIFFERENCE - Potential bug impact")

        s1_results[model_name] = {
            "recovery": recovery,
            "mape": mape,
            "published_recovery": pub_recovery,
            "published_mape": pub_mape,
            "delta_recovery": delta_recovery,
            "delta_mape": delta_mape,
        }

        # Extract channel breakdown
        if decomp_df is not None:
            s1_results[model_name]["channel_breakdown"] = {}
            for ch in ["tv", "search", "social", "display", "video"]:
                ch_metrics = extract_channel_recovery(df_s1, decomp_df, ch)
                s1_results[model_name]["channel_breakdown"][ch] = ch_metrics

    results["s1_baseline"] = s1_results

    # Save interim results
    with open(output_dir / "s1_baseline_results.json", "w") as f:
        json.dump(s1_results, f, indent=2, default=str)

    print("\n" + "="*80)
    print("PHASE 2: SCENARIO SENSITIVITY ANALYSIS (S2-S5)")
    print("="*80)

    for scenario_name in ["S2", "S3", "S4", "S5"]:
        print(f"\n--- {scenario_name} ---")
        df_scenario = scenarios[scenario_name]
        scenario_results = {}

        for model_name in sorted(MODEL_REGISTRY.keys()):
            model_class = MODEL_REGISTRY[model_name]
            config = get_config_for_model(model_name)

            decomp_df, metrics = run_model(model_class, df_scenario, config)

            recovery = metrics.get("recovery_accuracy", 0.0)
            mape = metrics.get("mape", 999.0)

            # Compare to S1
            s1_recovery = s1_results[model_name]["recovery"]
            delta_recovery_vs_s1 = recovery - s1_recovery

            print(f"  {model_name}: recovery {recovery:.1f}% (Δ vs S1: {delta_recovery_vs_s1:+.1f}pp)")

            scenario_results[model_name] = {
                "recovery": recovery,
                "mape": mape,
                "delta_vs_s1": delta_recovery_vs_s1,
            }

            # Channel breakdown
            if decomp_df is not None:
                scenario_results[model_name]["channel_breakdown"] = {}
                for ch in ["tv", "search", "social", "display", "video"]:
                    ch_metrics = extract_channel_recovery(df_scenario, decomp_df, ch)
                    scenario_results[model_name]["channel_breakdown"][ch] = ch_metrics

        results["per_scenario"][scenario_name] = scenario_results

    # Save results
    with open(output_dir / "all_scenario_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "="*80)
    print("PHASE 3: ISSUE-IMPACT MAPPING")
    print("="*80)

    # Known issues and affected models
    issues = {
        "dual_adstock_coef_indexing": {
            "description": "DualAdstockOLS line 150: coef_idx incremented for missing channels without adding to X",
            "affected_models": ["dual_adstock"],
            "framework": "F1",
            "severity": "high",
            "expected_impact": "Coefficient indices misaligned → wrong channel decomposition or crash",
        },
        "weibull_coef_indexing": {
            "description": "WeibullAdstockNLS line 158: uses enumerate(channels) index instead of feature_names index",
            "affected_models": ["weibull_adstock"],
            "framework": "F1",
            "severity": "high",
            "expected_impact": "Wrong coefficients assigned to channels → decomposition mismatch",
        },
        "geometric_coef_alignment": {
            "description": "GeometricAdstockOLS: similar indexing issue if channels missing",
            "affected_models": ["geo_adstock"],
            "framework": "F1",
            "severity": "medium",
            "expected_impact": "Reduced if no channels are missing (all 5 present in S1-S5)",
        },
        "almon_degree_naming": {
            "description": "AlmonPDL: ltc_degree vs stc_degree parameter confusion",
            "affected_models": ["almon_pdl"],
            "framework": "F1",
            "severity": "medium",
            "expected_impact": "Polynomial lag order may not match intent",
        },
        "ardl_reconstruction": {
            "description": "ARDLModel: AR reconstruction complexity in decompose()",
            "affected_models": ["ardl"],
            "framework": "F2",
            "severity": "high",
            "expected_impact": "Decomposition inaccuracy; already showing 0% recovery in S1",
        },
        "finite_dl_weibull_dims": {
            "description": "FiniteDLModel: Weibull weight dimension mismatch (max_lag vs actual lag count)",
            "affected_models": ["finite_dl"],
            "framework": "F2",
            "severity": "medium",
            "expected_impact": "Weight broadcasting or shape mismatch",
        },
        "koyck_index_fragility": {
            "description": "KoyckModel: Index arithmetic in lines 110, 155",
            "affected_models": ["koyck"],
            "framework": "F2",
            "severity": "medium",
            "expected_impact": "Edge-case crashes if AR order changes or data is sparse",
        },
        "f3_missing_exog_coefs": {
            "description": "BayesianStructuralTS & KalmanDLM: get_params() missing exog_coefs",
            "affected_models": ["bsts", "kalman_dlm"],
            "framework": "F3",
            "severity": "medium",
            "expected_impact": "Non-reproducible results; parameters incomplete for paper supplementary",
        },
        "kalman_unused_ltc_decay": {
            "description": "KalmanDLM line 96: ltc_decay_per_channel config ignored",
            "affected_models": ["kalman_dlm"],
            "framework": "F3",
            "severity": "low",
            "expected_impact": "Config not enforced, but model still fits without explicit per-channel decay",
        },
        "mcmc_missing_channel_handling": {
            "description": "MCMCLatentStock: Missing channel handling in get_params()",
            "affected_models": ["mcmc_stock"],
            "framework": "F3",
            "severity": "medium",
            "expected_impact": "Non-reproducible parameters; missing channel states in serialization",
        },
    }

    # Correlate issues with observed results
    impact_analysis = {}

    for issue_key, issue_desc in issues.items():
        affected = issue_desc["affected_models"]
        impact_analysis[issue_key] = {
            "description": issue_desc["description"],
            "affected_models": affected,
            "framework": issue_desc["framework"],
            "severity": issue_desc["severity"],
            "expected_impact": issue_desc["expected_impact"],
            "observed_impact": {},
        }

        for model in affected:
            s1_delta = s1_results[model]["delta_recovery"]
            s1_mape_delta = s1_results[model]["delta_mape"]

            impact_analysis[issue_key]["observed_impact"][model] = {
                "delta_recovery_vs_published": s1_delta,
                "delta_mape_vs_published": s1_mape_delta,
                "assessment": "CRITICAL" if abs(s1_delta) > 10 else "SIGNIFICANT" if abs(s1_delta) > 5 else "MINOR",
            }

    results["issue_analysis"] = impact_analysis

    # Save full analysis
    with open(output_dir / "issue_impact_analysis.json", "w") as f:
        json.dump(impact_analysis, f, indent=2)

    return results

if __name__ == "__main__":
    results = main()
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - Results saved to /c/github/ltc/impact_assessment/")
    print("="*80)
