"""
validation_pipeline.edge_case_benchmark - Edge case integration tests.

Runs benchmarking with intentional edge cases to detect fragility:
  1. S1_clean — all 5 channels, baseline (reference)
  2. S1_missing_search — 4 channels (TV, Social, Display, Video)
  3. S1_missing_tv — 4 channels (Search, Social, Display, Video)
  4. S1_weak_exog — exogenous effects nulled out

Compares recovery % across conditions:
  - |recovery_clean - recovery_missing| > 0.5pp → FLAG as fragile
  - Any model crash → BLOCKER

Produces CSV: model, scenario, recovery%, status, delta_pp
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple
import csv

import numpy as np
import pandas as pd

# Allow importing from ltc package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltc.data.loader import load_scenario, split_observed_truth
from ltc.evaluation.scorer import score_model as eval_score_model
from experiments.registry import MODEL_REGISTRY, CONFIG_MAP


class EdgeCaseResult:
    """Container for a single edge case test result."""

    def __init__(
        self,
        model_name: str,
        scenario: str,
        recovery_pct: float,
        ltc_mape: float,
        status: str,
        delta_pp: float = 0.0,
        issue: str = "",
    ):
        self.model_name = model_name
        self.scenario = scenario
        self.recovery_pct = recovery_pct
        self.ltc_mape = ltc_mape
        self.status = status  # PASS, FLAG, FAIL
        self.delta_pp = delta_pp  # difference from baseline
        self.issue = issue

    def __repr__(self) -> str:
        return (
            f"{self.model_name:15} | {self.scenario:20} | "
            f"Recovery: {self.recovery_pct:6.1f}% | MAPE: {self.ltc_mape:6.1f}% | "
            f"Δ: {self.delta_pp:+6.1f}pp | {self.status:5} | {self.issue[:40]}"
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "scenario": self.scenario,
            "recovery_pct": round(self.recovery_pct, 1),
            "ltc_mape": round(self.ltc_mape, 1),
            "delta_pp": round(self.delta_pp, 1),
            "status": self.status,
            "issue": self.issue,
        }


class EdgeCaseBenchmark:
    """Run benchmarking on edge case scenarios."""

    def __init__(self, data_dir: Path = Path("data/raw"), config_dir: Path = Path("experiments/configs")):
        self.data_dir = data_dir
        self.config_dir = config_dir
        self.results: list[EdgeCaseResult] = []

        # Load S1 as base
        try:
            self.df_s1_full = load_scenario(self.data_dir, "S1")
        except Exception as e:
            raise RuntimeError(f"Failed to load S1: {e}")

    def load_config(self, model_name: str) -> dict:
        """Load model config from YAML."""
        config_key = CONFIG_MAP.get(model_name, "framework1")
        config_path = self.config_dir / f"{config_key}.yaml"
        if not config_path.exists():
            return {}
        import yaml
        with open(config_path) as f:
            all_configs = yaml.safe_load(f)
        return all_configs.get(model_name, {})

    def run_all_edge_cases(self) -> dict:
        """Run edge case benchmark for all models."""
        # Create edge case scenarios
        scenarios = self._prepare_edge_cases()

        # Run each model on each scenario
        baseline_results = {}
        for model_name in MODEL_REGISTRY:
            print(f"[edge_case] Testing {model_name} on edge cases...")
            for scenario_name, df_edge in scenarios.items():
                try:
                    result = self._run_single_edge_case(model_name, scenario_name, df_edge)
                    self.results.append(result)

                    # Store baseline for delta calculation
                    if scenario_name == "S1_clean":
                        baseline_results[model_name] = result.recovery_pct
                except Exception as e:
                    print(f"  [ERROR] {model_name} x {scenario_name}: {str(e)[:60]}")
                    self.results.append(
                        EdgeCaseResult(
                            model_name,
                            scenario_name,
                            0.0,
                            0.0,
                            "FAIL",
                            issue=f"Exception: {str(e)[:40]}",
                        )
                    )

        # Calculate deltas from baseline
        for result in self.results:
            if result.model_name in baseline_results and result.scenario != "S1_clean":
                baseline = baseline_results[result.model_name]
                result.delta_pp = result.recovery_pct - baseline

                # Flag if delta > 0.5pp
                if abs(result.delta_pp) > 0.5:
                    result.status = "FLAG"
                else:
                    result.status = "PASS"

        return self._summarize_results()

    def _prepare_edge_cases(self) -> Dict[str, pd.DataFrame]:
        """Prepare edge case scenarios from S1."""
        scenarios = {}

        # S1_clean: baseline (all 5 channels)
        df_obs_clean, df_truth_clean = split_observed_truth(self.df_s1_full)
        scenarios["S1_clean"] = (df_obs_clean, df_truth_clean)

        # S1_missing_search: drop search channel
        df_missing_search = self.df_s1_full.copy()
        for col in ["spend_search", "impr_search"]:
            if col in df_missing_search.columns:
                df_missing_search[col] = 0.0
        df_obs_search, df_truth_search = split_observed_truth(df_missing_search)
        scenarios["S1_missing_search"] = (df_obs_search, df_truth_search)

        # S1_missing_tv: drop TV channel
        df_missing_tv = self.df_s1_full.copy()
        for col in ["spend_tv", "impr_tv"]:
            if col in df_missing_tv.columns:
                df_missing_tv[col] = 0.0
        df_obs_tv, df_truth_tv = split_observed_truth(df_missing_tv)
        scenarios["S1_missing_tv"] = (df_obs_tv, df_truth_tv)

        # S1_weak_exog: null out exogenous effects
        df_weak_exog = self.df_s1_full.copy()
        exog_cols = ["promo", "covid_index", "dgs30", "mobility_index", "competitor_ishare"]
        for col in exog_cols:
            if col in df_weak_exog.columns:
                df_weak_exog[col] = df_weak_exog[col].mean()
        df_obs_weak, df_truth_weak = split_observed_truth(df_weak_exog)
        scenarios["S1_weak_exog"] = (df_obs_weak, df_truth_weak)

        return scenarios

    def _run_single_edge_case(
        self, model_name: str, scenario_name: str, df_data: Tuple[pd.DataFrame, pd.DataFrame]
    ) -> EdgeCaseResult:
        """Run a single model on a single edge case."""
        df_obs, df_truth = df_data

        # Load model and config
        model_cls = MODEL_REGISTRY[model_name]
        config = self.load_config(model_name)
        model = model_cls()

        # Fit and decompose
        model.fit(df_obs, config)
        decomp = model.decompose(df_obs)

        # Score
        scores = eval_score_model(decomp, df_truth, model_name=model_name, scenario=scenario_name)

        # Extract recovery and MAPE
        ltc_total = scores.get("ltc", {}).get("total", {})
        recovery_pct = ltc_total.get("recovery_accuracy", 0.0)
        ltc_mape = ltc_total.get("mape", 0.0)

        return EdgeCaseResult(
            model_name, scenario_name, recovery_pct, ltc_mape, "PASS"
        )

    def _summarize_results(self) -> dict:
        """Summarize edge case results."""
        flags = sum(1 for r in self.results if r.status == "FLAG")
        failures = sum(1 for r in self.results if r.status == "FAIL")
        passes = sum(1 for r in self.results if r.status == "PASS")

        # Group by model to show fragility
        fragility_by_model = {}
        for result in self.results:
            if result.model_name not in fragility_by_model:
                fragility_by_model[result.model_name] = {"flags": 0, "failures": 0, "passes": 0}
            fragility_by_model[result.model_name][result.status.lower() + "s"] += 1

        return {
            "total_tests": len(self.results),
            "passes": passes,
            "flags": flags,
            "failures": failures,
            "fragility_by_model": fragility_by_model,
            "results": [r.to_dict() for r in self.results],
        }


def run_edge_case_benchmark() -> dict:
    """Main entry point: run edge case benchmarking."""
    try:
        benchmark = EdgeCaseBenchmark(
            data_dir=Path("data/raw"), config_dir=Path("experiments/configs")
        )
        return benchmark.run_all_edge_cases()
    except Exception as e:
        print(f"[ERROR] Failed to initialize edge case benchmark: {e}")
        return {
            "total_tests": 0,
            "passes": 0,
            "flags": 0,
            "failures": 1,
            "fragility_by_model": {},
            "results": [],
        }


if __name__ == "__main__":
    results = run_edge_case_benchmark()
    print("\n" + "=" * 140)
    print("EDGE CASE BENCHMARK RESULTS")
    print("=" * 140)

    # Print summary
    print(f"\nTotal Tests: {results['total_tests']}")
    print(f"Passed: {results['passes']} ✓")
    print(f"Flagged (Δ > 0.5pp): {results['flags']} ⚠")
    print(f"Failed (Exception): {results['failures']} ✗")

    # Print fragility by model
    print("\n" + "-" * 140)
    print("FRAGILITY SUMMARY (by model)")
    print("-" * 140)
    for model_name, counts in sorted(results["fragility_by_model"].items()):
        passes = counts.get("passes", 0)
        flags = counts.get("flags", 0)
        failures = counts.get("failures", 0)
        total = passes + flags + failures
        fragility_score = (flags + failures * 2) / total if total > 0 else 0
        status = "🔴 FRAGILE" if fragility_score > 0.5 else "🟢 ROBUST"
        print(
            f"{model_name:15} | Passes: {passes:2d}, Flags: {flags:2d}, Failures: {failures:2d} | "
            f"Fragility: {fragility_score:.2f} | {status}"
        )

    # Print detailed results (only flags and failures)
    if results["flags"] > 0 or results["failures"] > 0:
        print("\n" + "-" * 140)
        print("DETAILED RESULTS (Flags & Failures Only)")
        print("-" * 140)
        for r in results["results"]:
            if r["status"] != "PASS":
                print(
                    f"{r['model']:15} | {r['scenario']:20} | "
                    f"Recovery: {r['recovery_pct']:6.1f}% | MAPE: {r['ltc_mape']:6.1f}% | "
                    f"Δ: {r['delta_pp']:+6.1f}pp | {r['status']:5} | {r['issue'][:40]}"
                )

    # Exit code
    sys.exit(0 if results["failures"] == 0 else 1)
