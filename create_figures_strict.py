"""
STRICT FIGURE GENERATION WITH VALIDATION
Create figures ONE BY ONE with data validation at each step
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("outputs/results")
FIGURES_DIR = Path("outputs/figures")
DPI = 300
FONT_SIZE = 11
LABEL_SIZE = 9
TITLE_WEIGHT = "bold"
GRID_ALPHA = 0.3

COLORS = {
    "F1": "#d62728",  # Red
    "F2": "#1f77b4",  # Blue
    "F3": "#2ca02c",  # Green
}

MODEL_FRAMEWORK = {
    "geo_adstock": "F1", "weibull_adstock": "F1", "almon_pdl": "F1", "dual_adstock": "F1",
    "koyck": "F2", "ardl": "F2", "finite_dl": "F2",
    "kalman_dlm": "F3", "mcmc_stock": "F3", "bsts": "F3",
}

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_all_data():
    """Load ALL JSON data into single dataframe."""
    results = defaultdict(dict)

    for json_file in sorted(RESULTS_DIR.glob("*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
            parts = json_file.stem.split('_')
            scenario = parts[-1]
            model = '_'.join(parts[:-1])
            results[model][scenario] = data

    # Convert to dataframe
    records = []
    for model in sorted(results.keys()):
        for scenario in sorted(results[model].keys()):
            json_data = results[model][scenario]
            ltc_recovery = json_data.get("ltc", {}).get("total", {}).get("recovery_accuracy", np.nan)

            records.append({
                "Model": model,
                "Framework": MODEL_FRAMEWORK.get(model, "Unknown"),
                "Scenario": scenario,
                "Recovery": ltc_recovery,
                "Raw_JSON": json_data
            })

    df = pd.DataFrame(records)
    return df

def validate_figure_data(df, figure_name, expected_checks):
    """Validate that figure data matches expected values."""
    print(f"\n{'='*80}")
    print(f"VALIDATING FIGURE: {figure_name}")
    print(f"{'='*80}")

    for check_name, check_data in expected_checks.items():
        model = check_data["model"]
        scenario = check_data["scenario"]
        expected_value = check_data["expected"]

        actual = df[(df["Model"] == model) & (df["Scenario"] == scenario)]["Recovery"].values
        actual_value = actual[0] if len(actual) > 0 else None

        is_match = actual_value is not None and abs(actual_value - expected_value) < 1
        status = "PASS" if is_match else "FAIL"

        print(f"{status}: {model:20s} {scenario:3s} - Expected {expected_value:6.1f}%, Got {actual_value:6.2f}%")

        if not is_match:
            print(f"       ^^^ MISMATCH DETECTED ^^^")

def create_figure_2_heatmap(df):
    """Figure 2: Cross-Scenario Heatmap."""
    print("\n" + "="*80)
    print("CREATING FIGURE 2: Cross-Scenario Heatmap")
    print("="*80)

    # Validate critical values
    validate_figure_data(df, "Figure 2", {
        "bsts_s1": {"model": "bsts", "scenario": "S1", "expected": 82.4},
        "ardl_s2": {"model": "ardl", "scenario": "S2", "expected": 68.8},
        "mcmc_s3": {"model": "mcmc_stock", "scenario": "S3", "expected": 99.0},
        "geo_s1": {"model": "geo_adstock", "scenario": "S1", "expected": 69.9},
    })

    # Create pivot table
    pivot = df.pivot_table(values="Recovery", index="Model", columns="Scenario", aggfunc='first')

    print("\nPivot Table (what will be displayed):")
    print(pivot)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=LABEL_SIZE)
    ax.set_yticklabels(pivot.index, fontsize=LABEL_SIZE)

    ax.set_xlabel("Scenario", fontsize=FONT_SIZE)
    ax.set_ylabel("Model", fontsize=FONT_SIZE)
    ax.set_title("Figure 2: Cross-Scenario Heatmap (Recovery % 0-100)", fontsize=FONT_SIZE, weight=TITLE_WEIGHT)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Recovery %")

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7, color="black")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_02_Cross_Scenario_Heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    print("\nFigure 2 created successfully!")
    print(f"Saved to: {FIGURES_DIR / 'Figure_02_Cross_Scenario_Heatmap.png'}")

def create_figure_5_hierarchy(df):
    """Figure 5: Framework Hierarchy."""
    print("\n" + "="*80)
    print("CREATING FIGURE 5: Framework Hierarchy")
    print("="*80)

    # Get S1 baseline for all models
    s1_data = df[df["Scenario"] == "S1"].copy()

    print("\nS1 Recovery Values (all models):")
    print(s1_data[["Model", "Framework", "Recovery"]].to_string(index=False))

    # Group by framework
    f1_data = s1_data[s1_data["Framework"] == "F1"]["Recovery"].values
    f2_data = s1_data[s1_data["Framework"] == "F2"]["Recovery"].values
    f3_data = s1_data[s1_data["Framework"] == "F3"]["Recovery"].values

    print(f"\nFramework Averages (S1):")
    print(f"F1: {f1_data.mean():.1f}% (models: {', '.join(s1_data[s1_data['Framework']=='F1']['Model'].values)})")
    print(f"F2: {f2_data.mean():.1f}% (models: {', '.join(s1_data[s1_data['Framework']=='F2']['Model'].values)})")
    print(f"F3: {f3_data.mean():.1f}% (models: {', '.join(s1_data[s1_data['Framework']=='F3']['Model'].values)})")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = [f1_data, f2_data, f3_data]
    bp = ax.boxplot(data_to_plot, tick_labels=["F1\n(Static Adstock)", "F2\n(Dynamic AR)", "F3\n(State-Space)"],
                    patch_artist=True, widths=0.6)

    colors_list = ["#d62728", "#1f77b4", "#2ca02c"]
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Recovery Accuracy (%)", fontsize=FONT_SIZE)
    ax.set_title("Figure 5: Framework Hierarchy (S1 Baseline)", fontsize=FONT_SIZE, weight=TITLE_WEIGHT)
    ax.grid(axis='y', alpha=GRID_ALPHA)
    ax.set_ylim(-10, 100)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_05_Framework_Hierarchy.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    print("\nFigure 5 created successfully!")

if __name__ == "__main__":
    print("Loading all data...")
    df = load_all_data()
    print(f"Loaded {len(df)} records")
    print(f"Models: {df['Model'].nunique()}")
    print(f"Scenarios: {df['Scenario'].nunique()}")

    print("\n" + "="*80)
    print("FULL DATA MATRIX")
    print("="*80)
    pivot = df.pivot_table(values="Recovery", index="Model", columns="Scenario")
    print(pivot)

    # Create figures one by one with validation
    create_figure_2_heatmap(df)
    create_figure_5_hierarchy(df)

    print("\n" + "="*80)
    print("FIGURE CREATION COMPLETE")
    print("="*80)
    print("\nNext: Verify figures visually, then create remaining figures")
