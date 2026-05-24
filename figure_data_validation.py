"""
STRICT DATA VALIDATION FOR FIGURE GENERATION
Load all data, validate against section tables, then create figures
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("outputs/results")

def load_all_json_data():
    """Load ALL JSON files and inspect structure."""
    results = defaultdict(dict)

    for json_file in sorted(RESULTS_DIR.glob("*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
            parts = json_file.stem.split('_')
            scenario = parts[-1]
            model = '_'.join(parts[:-1])
            results[model][scenario] = data

    return results

def extract_ltc_recovery(results):
    """Extract LTC recovery accuracy from JSON files."""
    data = []

    for model in sorted(results.keys()):
        for scenario in sorted(results[model].keys()):
            json_data = results[model][scenario]

            # Get LTC total recovery
            ltc_recovery = None
            if "ltc" in json_data and "total" in json_data["ltc"]:
                ltc_recovery = json_data["ltc"]["total"].get("recovery_accuracy")

            data.append({
                "Model": model,
                "Scenario": scenario,
                "LTC_Recovery": ltc_recovery,
                "Raw": json_data
            })

    return pd.DataFrame(data)

def print_data_summary(df):
    """Print summary of extracted data."""
    print("\n" + "="*80)
    print("DATA EXTRACTION SUMMARY")
    print("="*80)
    print(f"\nTotal records: {len(df)}")
    print(f"Models: {df['Model'].nunique()}")
    print(f"Scenarios: {df['Scenario'].nunique()}")

    print("\n--- LTC RECOVERY BY MODEL AND SCENARIO ---")
    pivot = df.pivot_table(values='LTC_Recovery', index='Model', columns='Scenario', aggfunc='first')
    print(pivot.to_string())

    print("\n--- MODELS BY FRAMEWORK ---")
    framework_map = {
        "geo_adstock": "F1", "weibull_adstock": "F1", "almon_pdl": "F1", "dual_adstock": "F1",
        "koyck": "F2", "ardl": "F2", "finite_dl": "F2",
        "kalman_dlm": "F3", "mcmc_stock": "F3", "bsts": "F3"
    }
    for framework in ["F1", "F2", "F3"]:
        models = [m for m, f in framework_map.items() if f == framework]
        print(f"{framework}: {', '.join(models)}")

    return pivot

def validate_against_section_tables():
    """
    VALIDATION CHECKLIST against Section 4, Table 3
    Table 3 S1 values (from RESULTS_SECTION_4_DRAFT.md lines 83-98):

    bsts: S1=82.4%, S2=81.0%, S3=76.8%, S4=81.6%, S5=0.0%
    kalman_dlm: S1=82.0%, S2=83.1%, S3=64.9%, S4=75.4%, S5=0.0%
    mcmc_stock: S1=72.4%, S2=59.9%, S3=99.0%, S4=90.9%, S5=0.0%
    geo_adstock: S1=69.9%, S2=83.1%, S3=43.2%, S4=63.4%, S5=0.0%
    finite_dl: S1=50.3%, S2=54.6%, S3=58.0%, S4=40.5%, S5=0.0%
    koyck: S1=46.4%, S2=43.0%, S3=53.7%, S4=52.3%, S5=0.0%
    almon_pdl: S1=42.6%, S2=18.7%, S3=40.6%, S4=68.6%, S5=0.0%
    weibull_adstock: S1=10.5%, S2=30.5%, S3=0.0%, S4=-23.2%, S5=0.0%
    ardl: S1=0.0%, S2=68.8%, S3=63.3%, S4=-19.8%, S5=0.0%
    dual_adstock: S1=0.0%, S2=0.0%, S3=0.0%, S4=-578%, S5=0.0%
    """

    expected_values = {
        "bsts": {"S1": 82.4, "S2": 81.0, "S3": 76.8, "S4": 81.6, "S5": 0.0},
        "kalman_dlm": {"S1": 82.0, "S2": 83.1, "S3": 64.9, "S4": 75.4, "S5": 0.0},
        "mcmc_stock": {"S1": 72.4, "S2": 59.9, "S3": 99.0, "S4": 90.9, "S5": 0.0},
        "geo_adstock": {"S1": 69.9, "S2": 83.1, "S3": 43.2, "S4": 63.4, "S5": 0.0},
        "finite_dl": {"S1": 50.3, "S2": 54.6, "S3": 58.0, "S4": 40.5, "S5": 0.0},
        "koyck": {"S1": 46.4, "S2": 43.0, "S3": 53.7, "S4": 52.3, "S5": 0.0},
        "almon_pdl": {"S1": 42.6, "S2": 18.7, "S3": 40.6, "S4": 68.6, "S5": 0.0},
        "weibull_adstock": {"S1": 10.5, "S2": 30.5, "S3": 0.0, "S4": -23.2, "S5": 0.0},
        "ardl": {"S1": 0.0, "S2": 68.8, "S3": 63.3, "S4": -19.8, "S5": 0.0},
        "dual_adstock": {"S1": 0.0, "S2": 0.0, "S3": 0.0, "S4": -578.0, "S5": 0.0},
    }

    return expected_values

if __name__ == "__main__":
    print("Loading all JSON data...")
    results = load_all_json_data()
    print(f"Loaded {sum(len(v) for v in results.values())} model-scenario combinations")

    print("\nExtracting LTC recovery values...")
    df = extract_ltc_recovery(results)

    print("\nPrinting data summary...")
    pivot = print_data_summary(df)

    print("\n" + "="*80)
    print("VALIDATION: Expected values from Section 4, Table 3")
    print("="*80)
    expected = validate_against_section_tables()
    for model in sorted(expected.keys()):
        print(f"\n{model}:")
        actual = df[df["Model"] == model][["Scenario", "LTC_Recovery"]].set_index("Scenario")
        for scenario in ["S1", "S2", "S3", "S4", "S5"]:
            exp = expected[model][scenario]
            act = actual.loc[scenario, "LTC_Recovery"] if scenario in actual.index else None
            match = "OK" if (act is not None and abs(act - exp) < 1) else "MISMATCH"
            print(f"  {scenario}: Expected {exp:7.1f}%, Actual {act}, {match}")

    print("\n" + "="*80)
    print("NEXT: Use this validated data to create figures ONE BY ONE")
    print("="*80)
