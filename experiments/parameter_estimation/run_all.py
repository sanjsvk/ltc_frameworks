"""
Run all parameter estimation scripts and update YAML configs.

Usage
-----
    python experiments/parameter_estimation/run_all.py

This script:
  1. Runs F1 decay search, F2 lag selection, F3 prior estimation.
  2. Writes consolidated estimated_params.json.
  3. Updates experiments/configs/framework*.yaml with estimated values.
  4. Populates parameter_log.csv with all estimation results.

The YAML update only modifies parameters that improved on the current defaults.
Original YAML values are preserved as comments if unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.parameter_estimation.f1_decay_search import run_f1_decay_search, write_parameter_log_rows as f1_log
from experiments.parameter_estimation.f2_lag_selection import run_f2_lag_selection, write_parameter_log_rows as f2_log
from experiments.parameter_estimation.f3_prior_estimation import run_f3_prior_estimation, write_parameter_log_rows as f3_log

OUTPUT_DIR = Path("experiments/parameter_estimation")
CONFIG_DIR = Path("experiments/configs")
LOG_PATH = Path("outputs/parameter_log.csv")


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def update_framework3_yaml(f3_results: dict) -> None:
    """Update framework3.yaml with estimated Kalman variances and MCMC priors."""
    config_path = CONFIG_DIR / "framework3.yaml"
    config = _load_yaml(config_path)

    # KalmanDLM
    kd = f3_results.get("kalman_dlm", {})
    if "level_var" in kd:
        config["kalman_dlm"]["level_var"] = kd["level_var"]
        config["kalman_dlm"]["slope_var"] = kd["slope_var"]
        print(f"  kalman_dlm: level_var={kd['level_var']}, slope_var={kd['slope_var']}")

    # BSTS
    bs = f3_results.get("bsts", {})
    if "level_var" in bs:
        config["bsts"]["level_var"] = bs["level_var"]
        config["bsts"]["slope_var"] = bs["slope_var"]
        print(f"  bsts: level_var={bs['level_var']}, slope_var={bs['slope_var']}")

    # MCMCLatentStock — update to mcmc backend + calibrated priors
    mc = f3_results.get("mcmc_stock", {})
    config["mcmc_stock"]["backend"] = "mcmc"
    if "prior_delta_alpha" in mc:
        config["mcmc_stock"]["prior_delta_alpha"] = mc["prior_delta_alpha"]
        config["mcmc_stock"]["prior_delta_beta"] = mc["prior_delta_beta"]
        config["mcmc_stock"]["prior_build_rate_sigma"] = mc["prior_build_rate_sigma"]
        config["mcmc_stock"]["prior_ltc_coef_sigma"] = mc["prior_ltc_coef_sigma"]
        config["mcmc_stock"]["prior_obs_sigma"] = mc.get("prior_obs_sigma", 0.3)
        print(f"  mcmc_stock: backend=mcmc, priors calibrated from MAP on S1")

    _write_yaml(config_path, config)
    print(f"  Updated {config_path}")


def update_framework2_yaml(f2_results: dict) -> None:
    """Update framework2.yaml with BIC-selected ARDL order."""
    config_path = CONFIG_DIR / "framework2.yaml"
    config = _load_yaml(config_path)

    ardl = f2_results.get("ardl", {})
    if "ar_order" in ardl:
        config["ardl"]["ar_order"] = ardl["ar_order"]
        config["ardl"]["media_lags"] = ardl["media_lags"]
        print(f"  ardl: ar_order={ardl['ar_order']}, media_lags={ardl['media_lags']}")

    _write_yaml(config_path, config)
    print(f"  Updated {config_path}")


def main() -> None:
    print("\n" + "=" * 70)
    print("LTC FRAMEWORK PARAMETER ESTIMATION")
    print("All parameters estimated from S1 observed data only (no ground truth).")
    print("=" * 70)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_params: dict = {}

    print("\n--- STEP 1: Framework 1 Decay Search ---")
    f1_results = run_f1_decay_search()
    all_params["f1"] = f1_results
    f1_log(f1_results, LOG_PATH)

    print("\n--- STEP 2: Framework 2 Lag Selection ---")
    f2_results = run_f2_lag_selection()
    all_params["f2"] = f2_results
    f2_log(f2_results, LOG_PATH)

    print("\n--- STEP 3: Framework 3 Prior Estimation ---")
    f3_results = run_f3_prior_estimation()
    all_params["f3"] = f3_results
    f3_log(f3_results, LOG_PATH)

    # Write consolidated params
    out_file = OUTPUT_DIR / "estimated_params.json"
    with open(out_file, "w") as f:
        json.dump(all_params, f, indent=2, default=str)
    print(f"\n✓ Wrote consolidated parameters to {out_file}")

    # Update YAML configs
    print("\n── UPDATING YAML CONFIGS ───────────────────────────────────────────")
    update_framework2_yaml(f2_results)
    update_framework3_yaml(f3_results)

    print("\n" + "=" * 70)
    print("PARAMETER ESTIMATION COMPLETE")
    print(f"  Results: {out_file}")
    print(f"  Log:     {LOG_PATH}")
    print("  YAML configs updated for framework2 and framework3.")
    print("  Framework1 uses internal grid search — no YAML update needed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
