"""
experiments.run_experiment — CLI entry point for running LTC framework experiments.

Usage examples:
  # Single model, single scenario
  python experiments/run_experiment.py --model geo_adstock --scenario S1

  # Single model, all scenarios
  python experiments/run_experiment.py --model mcmc_stock --all-scenarios

  # Full benchmark: all models × all scenarios
  python experiments/run_experiment.py --all-models --all-scenarios

  # Framework group
  python experiments/run_experiment.py --framework F3_state_space --scenario S2

Outputs written to:
  outputs/results/{model}_{scenario}.json   — metrics JSON
  outputs/figures/{model}_{scenario}_decomp.png  — decomposition chart
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import yaml

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltc.data.loader import load_scenario, split_observed_truth, SCENARIOS
from ltc.data.features import build_feature_set
from ltc.evaluation.scorer import score_model as eval_score_model
from ltc.visualization.decomposition import plot_ltc_vs_truth, save_figure
from experiments.registry import MODEL_REGISTRY, CONFIG_MAP, FRAMEWORK_GROUPS

# Paths (relative to repo root)
DATA_DIR = Path("data/raw")
CONFIG_DIR = Path("experiments/configs")
RESULTS_DIR = Path("outputs/results")
FIGURES_DIR = Path("outputs/figures")


def load_config(model_name: str) -> dict:
    """Load hyperparameters for a model from its framework YAML config."""
    config_key = CONFIG_MAP.get(model_name, "framework1")
    config_path = CONFIG_DIR / f"{config_key}.yaml"
    if not config_path.exists():
        click.echo(f"[warn] Config file not found: {config_path}. Using empty config.")
        return {}
    with open(config_path) as f:
        all_configs = yaml.safe_load(f)
    return all_configs.get(model_name, {})


def run_one(model_name: str, scenario: str, save_fig: bool = True) -> dict:
    """
    Run a single model × scenario experiment.

    Returns
    -------
    dict — score_model() output (metrics).
    """
    click.echo(f"[run] {model_name} × {scenario} ...")

    # --- Data loading ---
    try:
        df_full = load_scenario(DATA_DIR, scenario)
    except FileNotFoundError as e:
        click.echo(f"[error] {e}")
        return {}

    df_obs, df_truth = split_observed_truth(df_full)

    # --- Model instantiation and fitting ---
    if model_name not in MODEL_REGISTRY:
        click.echo(f"[error] Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")
        return {}

    model_cls = MODEL_REGISTRY[model_name]
    config = load_config(model_name)
    model = model_cls()

    try:
        model.fit(df_obs, config)
    except Exception as e:
        click.echo(f"[error] fit() failed for {model_name} × {scenario}: {e}")
        return {}

    # --- Decomposition ---
    try:
        decomp = model.decompose(df_obs)
    except Exception as e:
        click.echo(f"[error] decompose() failed: {e}")
        return {}

    # --- Evaluation ---
    scores = eval_score_model(
        decomp, df_truth, model_name=model_name, scenario=scenario
    )
    params = model.get_params()
    scores["fitted_params"] = params

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{model_name}_{scenario}.json"
    with open(result_path, "w") as f:
        json.dump(scores, f, indent=2, default=str)
    click.echo(f"[save] Results → {result_path}")

    # --- Optional figure ---
    if save_fig:
        try:
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            fig = plot_ltc_vs_truth(
                decomp, df_truth,
                title=f"{model_name} — {scenario}: Estimated vs. True LTC"
            )
            fig_path = FIGURES_DIR / f"{model_name}_{scenario}_decomp.png"
            save_figure(fig, fig_path)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            click.echo(f"[warn] Figure generation failed: {e}")

    # Print summary
    ltc_total = scores.get("ltc", {}).get("total", {})
    click.echo(
        f"  → LTC recovery_accuracy={ltc_total.get('recovery_accuracy', 'n/a'):.1f}%  "
        f"mape={ltc_total.get('mape', 'n/a'):.1f}%  "
        f"total_ratio={ltc_total.get('total_recovery_ratio', 'n/a'):.3f}"
    )
    return scores


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

@click.command()
@click.option("--model", default=None, help="Model name (e.g., geo_adstock, mcmc_stock)")
@click.option("--scenario", default=None, help="Scenario ID: S1, S2, S3, S4, S5")
@click.option("--all-scenarios", "all_scenarios", is_flag=True, default=False,
              help="Run across all 5 scenarios")
@click.option("--all-models", "all_models", is_flag=True, default=False,
              help="Run all registered models")
@click.option("--framework", default=None,
              help="Run all models in a framework group: F1_static_adstock, F2_dynamic_ts, F3_state_space")
@click.option("--no-fig", "no_fig", is_flag=True, default=False,
              help="Skip figure generation (faster)")
@click.option("--data-dir", "data_dir", default=str(DATA_DIR),
              help="Path to directory containing scenario CSVs")
def main(
    model: str | None,
    scenario: str | None,
    all_scenarios: bool,
    all_models: bool,
    framework: str | None,
    no_fig: bool,
    data_dir: str,
) -> None:
    """Run LTC framework experiments against synthetic MMM scenario data."""
    global DATA_DIR
    DATA_DIR = Path(data_dir)

    # Resolve model list
    if all_models:
        models_to_run = list(MODEL_REGISTRY.keys())
    elif framework:
        models_to_run = FRAMEWORK_GROUPS.get(framework, [])
        if not models_to_run:
            click.echo(f"[error] Unknown framework '{framework}'. Options: {list(FRAMEWORK_GROUPS)}")
            sys.exit(1)
    elif model:
        models_to_run = [model]
    else:
        click.echo("[error] Specify --model, --framework, or --all-models")
        sys.exit(1)

    # Resolve scenario list
    if all_scenarios:
        scenarios_to_run = SCENARIOS
    elif scenario:
        scenarios_to_run = [scenario]
    else:
        click.echo("[error] Specify --scenario or --all-scenarios")
        sys.exit(1)

    click.echo(f"Running {len(models_to_run)} model(s) × {len(scenarios_to_run)} scenario(s) ...")
    click.echo(f"Models: {models_to_run}")
    click.echo(f"Scenarios: {scenarios_to_run}")
    click.echo("")

    total = len(models_to_run) * len(scenarios_to_run)
    completed = 0
    failed = 0

    for m in models_to_run:
        for s in scenarios_to_run:
            result = run_one(m, s, save_fig=not no_fig)
            if result:
                completed += 1
            else:
                failed += 1

    click.echo(f"\nDone: {completed}/{total} succeeded, {failed} failed.")
    click.echo(f"Results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
