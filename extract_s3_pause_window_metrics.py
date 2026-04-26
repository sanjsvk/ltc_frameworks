"""
Extract pause window MAPE (weeks 100-120) for all models on S3 scenario.

Requires: Fitted models from S3, or ability to refit.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from ltc.data.loader import load_scenario, split_observed_truth
from ltc.data.features import build_feature_set
from experiments.registry import MODEL_REGISTRY, CONFIG_MAP, FRAMEWORK_GROUPS
import yaml

DATA_DIR = Path("data/raw")
CONFIG_DIR = Path("experiments/configs")
RESULTS_DIR = Path("outputs/results")

def load_config(model_name: str) -> dict:
    """Load hyperparameters for a model from its framework YAML config."""
    config_key = CONFIG_MAP.get(model_name, "framework1")
    config_path = CONFIG_DIR / f"{config_key}.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        all_configs = yaml.safe_load(f)
    return all_configs.get(model_name, {})

def compute_pause_window_metrics(model_name: str, scenario: str = "S3"):
    """Fit model on S3 and compute pause-window MAPE."""

    # Load data
    df_full = load_scenario(DATA_DIR, scenario)
    df_obs, df_truth = split_observed_truth(df_full)

    # Define pause window (weeks 100-120)
    pause_mask = (df_full['week_id'] >= 100) & (df_full['week_id'] <= 120)
    df_pause_obs = df_obs[pause_mask]
    df_pause_truth = df_truth[pause_mask]

    # Fit model
    model_cls = MODEL_REGISTRY[model_name]
    config = load_config(model_name)
    model = model_cls()

    try:
        model.fit(df_obs, config)
    except Exception as e:
        print(f"  {model_name}: fit() failed — {e}")
        return None

    # Get full decomposition
    try:
        decomp_full = model.decompose(df_obs)
    except Exception as e:
        print(f"  {model_name}: decompose() failed — {e}")
        return None

    # Extract pause window decomposition
    decomp_pause = decomp_full[pause_mask]

    # Compute metrics for full series and pause window
    # LTC total = sum of all ltc_* columns
    ltc_cols = [col for col in decomp_full.columns if col.startswith('ltc_')]

    ltc_full = decomp_full[ltc_cols].sum(axis=1)
    ltc_pause = decomp_pause[ltc_cols].sum(axis=1)

    truth_ltc_cols = [col for col in df_truth.columns if col.startswith('ltc_') and col.endswith('_true')]
    truth_ltc_full = df_truth[truth_ltc_cols].sum(axis=1)
    truth_ltc_pause = df_pause_truth[truth_ltc_cols].sum(axis=1)

    # Compute MAPE
    def mape(y_true, y_pred):
        denom = np.abs(y_true) + 1e-9
        return 100 * np.mean(np.abs(y_pred - y_true) / denom)

    full_mape = mape(truth_ltc_full.values, ltc_full.values)
    pause_mape = mape(truth_ltc_pause.values, ltc_pause.values)
    ratio = pause_mape / full_mape if full_mape > 0 else np.nan

    return {
        'full_mape': full_mape,
        'pause_mape': pause_mape,
        'ratio': ratio
    }

# Run extraction
models = [
    "bsts", "kalman_dlm", "mcmc_stock",
    "geo_adstock", "weibull_adstock", "almon_pdl", "dual_adstock",
    "finite_dl", "koyck", "ardl"
]

frameworks = {
    "geo_adstock": "F1", "weibull_adstock": "F1", "almon_pdl": "F1", "dual_adstock": "F1",
    "koyck": "F2", "ardl": "F2", "finite_dl": "F2",
    "kalman_dlm": "F3", "mcmc_stock": "F3", "bsts": "F3"
}

print("=" * 120)
print("PAUSE WINDOW MAPE EXTRACTION (S3, Weeks 100-120)")
print("=" * 120)
print()

results = {}
for model in models:
    print(f"Computing {model}... ", end="", flush=True)
    metrics = compute_pause_window_metrics(model, "S3")
    if metrics:
        results[model] = metrics
        print(f"full={metrics['full_mape']:.1f}%, pause={metrics['pause_mape']:.1f}%, ratio={metrics['ratio']:.2f}x")
    else:
        print("FAILED")

print()
print("=" * 120)
print(f"{'Model':<20} {'F':<3} {'Full-Series MAPE':<20} {'Pause-Window MAPE':<20} {'Ratio':<10}")
print("=" * 120)

for model in models:
    if model in results:
        m = results[model]
        f = frameworks.get(model, "?")
        print(f"{model:<20} {f:<3} {m['full_mape']:>6.1f}% {' ':<13} {m['pause_mape']:>6.1f}% {' ':<13} {m['ratio']:>5.2f}x")

print("=" * 120)
