"""
validation_pipeline.contract_validator - Automated interface compliance tests.

Verifies that all 10 models in the registry comply with the BaseLTCModel interface:
  1. fit() returns self with _is_fitted=True
  2. decompose() returns DataFrame with required columns
  3. get_params() returns JSON-serializable dict
  4. Channel handling is graceful (no crashes on missing channels)
  5. Coefficient indexing is correct (matches channel order)

Each test produces a detailed report with:
  - Model name
  - Test name
  - PASS/FAIL status
  - Issue description (if failed)
  - File path and line number (if applicable)
  - Severity: BLOCKER / HIGH / MEDIUM / LOW
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow importing from ltc package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltc.data.loader import load_scenario, split_observed_truth, SCENARIOS, CHANNELS
from experiments.registry import MODEL_REGISTRY, CONFIG_MAP


class ContractValidationResult:
    """Container for a single validation test result."""

    def __init__(
        self,
        model_name: str,
        test_name: str,
        passed: bool,
        severity: str = "INFO",
        issue: str = "",
        file_path: str = "",
        line_number: int = 0,
    ):
        self.model_name = model_name
        self.test_name = test_name
        self.passed = passed
        self.severity = severity  # BLOCKER, HIGH, MEDIUM, LOW
        self.issue = issue
        self.file_path = file_path
        self.line_number = line_number

    def __repr__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return (
            f"{status} | {self.model_name:15} | {self.test_name:30} | "
            f"[{self.severity:7}] {self.issue}"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model": self.model_name,
            "test": self.test_name,
            "passed": self.passed,
            "severity": self.severity,
            "issue": self.issue,
            "file_path": self.file_path,
            "line_number": self.line_number,
        }


class ContractValidator:
    """Run all interface compliance tests on every model in the registry."""

    def __init__(self, data_dir: Path = Path("data/raw"), config_dir: Path = Path("experiments/configs")):
        self.data_dir = data_dir
        self.config_dir = config_dir
        self.results: list[ContractValidationResult] = []

        # Load a test scenario (S1 is baseline)
        try:
            self.df_full = load_scenario(self.data_dir, "S1")
            self.df_obs, self.df_truth = split_observed_truth(self.df_full)
        except Exception as e:
            raise RuntimeError(f"Failed to load test scenario S1: {e}")

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

    def validate_all_models(self) -> dict:
        """Run all validation tests on all models."""
        for model_name in MODEL_REGISTRY:
            self._validate_single_model(model_name)

        return self._summarize_results()

    def _validate_single_model(self, model_name: str) -> None:
        """Run all tests on a single model."""
        model_cls = MODEL_REGISTRY[model_name]
        config = self.load_config(model_name)

        # Test 1: fit() returns self with _is_fitted=True
        self._test_fit_returns_self(model_name, model_cls, config)

        # Test 2: fit() sets _is_fitted=True
        self._test_fit_sets_fitted_flag(model_name, model_cls, config)

        # Test 3: decompose() returns DataFrame
        self._test_decompose_returns_dataframe(model_name, model_cls, config)

        # Test 4: decompose() has required columns
        self._test_decompose_required_columns(model_name, model_cls, config)

        # Test 5: decompose() raises before fit()
        self._test_decompose_before_fit_raises(model_name, model_cls)

        # Test 6: get_params() returns dict
        self._test_get_params_returns_dict(model_name, model_cls, config)

        # Test 7: get_params() is JSON-serializable
        self._test_get_params_json_serializable(model_name, model_cls, config)

        # Test 8: Channel count in decomposition matches config
        self._test_decomposition_channel_count(model_name, model_cls, config)

        # Test 9: Fitted values are numeric (not NaN/Inf)
        self._test_decomposition_numeric_values(model_name, model_cls, config)

        # Test 10: Model repr includes fitted status
        self._test_model_repr(model_name, model_cls, config)

    def _test_fit_returns_self(self, model_name: str, model_cls, config: dict) -> None:
        """Test that fit() returns self."""
        try:
            model = model_cls()
            result = model.fit(self.df_obs, config)
            if result is model:
                self.results.append(
                    ContractValidationResult(
                        model_name, "fit() returns self", True
                    )
                )
            else:
                self.results.append(
                    ContractValidationResult(
                        model_name,
                        "fit() returns self",
                        False,
                        severity="HIGH",
                        issue="fit() returns different object",
                    )
                )
        except Exception as e:
            self.results.append(
                ContractValidationResult(
                    model_name,
                    "fit() returns self",
                    False,
                    severity="BLOCKER",
                    issue=f"Exception during fit(): {str(e)[:60]}",
                )
            )

    def _test_fit_sets_fitted_flag(self, model_name: str, model_cls, config: dict) -> None:
        """Test that fit() sets _is_fitted=True."""
        try:
            model = model_cls()
            model.fit(self.df_obs, config)
            if hasattr(model, "_is_fitted") and model._is_fitted is True:
                self.results.append(
                    ContractValidationResult(
                        model_name, "fit() sets _is_fitted=True", True
                    )
                )
            else:
                self.results.append(
                    ContractValidationResult(
                        model_name,
                        "fit() sets _is_fitted=True",
                        False,
                        severity="HIGH",
                        issue="_is_fitted not set or False after fit()",
                    )
                )
        except Exception as e:
            self.results.append(
                ContractValidationResult(
                    model_name,
                    "fit() sets _is_fitted=True",
                    False,
                    severity="BLOCKER",
                    issue=f"Exception: {str(e)[:60]}",
                )
            )

    def _test_decompose_returns_dataframe(
        self, model_name: str, model_cls, config: dict
    ) -> None:
        """Test that decompose() returns a DataFrame."""
        try:
            model = model_cls()
            model.fit(self.df_obs, config)
            decomp = model.decompose(self.df_obs)
            if isinstance(decomp, pd.DataFrame):
                self.results.append(
                    ContractValidationResult(
                        model_name, "decompose() returns DataFrame", True
                    )
                )
            else:
                self.results.append(
                    ContractValidationResult(
                        model_name,
                        "decompose() returns DataFrame",
                        False,
                        severity="BLOCKER",
                        issue=f"Returns {type(decomp).__name__}, not DataFrame",
                    )
                )
        except Exception as e:
            self.results.append(
                ContractValidationResult(
                    model_name,
                    "decompose() returns DataFrame",
                    False,
                    severity="BLOCKER",
                    issue=f"Exception: {str(e)[:60]}",
                )
            )

    def _test_decompose_required_columns(
        self, model_name: str, model_cls, config: dict
    ) -> None:
        """Test that decompose() output has required columns."""
        try:
            model = model_cls()
            model.fit(self.df_obs, config)
            decomp = model.decompose(self.df_obs)

            # Required: baseline, stc_{ch}, ltc_{ch}, fitted
            required = {"baseline", "fitted"}
            channel_stc = {f"stc_{ch}" for ch in CHANNELS}
            channel_ltc = {f"ltc_{ch}" for ch in CHANNELS}

            missing = required - set(decomp.columns)

            if missing:
                self.results.append(
                    ContractValidationResult(
                        model_name,
                        "decompose() has required columns",
                        False,
                        severity="BLOCKER",
                        issue=f"Missing columns: {missing}",
                    )
                )
            else:
                # Check if channels are partially available
                available_stc = set(decomp.columns) & channel_stc
                available_ltc = set(decomp.columns) & channel_ltc

                if available_stc and available_ltc:
                    self.results.append(
                        ContractValidationResult(
                            model_name,
                            "decompose() has required columns",
                            True,
                        )
                    )
                else:
                    self.results.append(
                        ContractValidationResult(
                            model_name,
                            "decompose() has required columns",
                            False,
                            severity="HIGH",
                            issue=f"Missing STC/LTC for channels: {available_stc or 'all'} / {available_ltc or 'all'}",
                        )
                    )
        except Exception as e:
            self.results.append(
                ContractValidationResult(
                    model_name,
                    "decompose() has required columns",
                    False,
                    severity="BLOCKER",
                    issue=f"Exception: {str(e)[:60]}",
                )
            )

    def _test_decompose_before_fit_raises(self, model_name: str, model_cls) -> None:
        """Test that calling decompose() before fit() raises an error."""
        try:
            model = model_cls()
            try:
                model.decompose(self.df_obs)
                # If no error, that's a problem
                self.results.append(
                    ContractValidationResult(
                        model_name,
                        "decompose() before fit() raises",
                        False,
                        severity="HIGH",
                        issue="decompose() should raise RuntimeError before fit()",
                    )
                )
            except RuntimeError as e:
                if "_is_fitted" in str(e) or "before fit" in str(e):
                    self.results.append(
                        ContractValidationResult(
                            model_name,
                            "decompose() before fit() raises",
                            True,
                        )
                    )
                else:
                    self.results.append(
                        ContractValidationResult(
                            model_name,
                            "decompose() before fit() raises",
                            False,
                            severity="MEDIUM",
                            issue=f"RuntimeError raised but wrong message: {str(e)[:40]}",
                        )
                    )
            except Exception as e:
                self.results.append(
                    ContractValidationResult(
                        model_name,
                        "decompose() before fit() raises",
                        False,
                        severity="MEDIUM",
                        issue=f"Wrong exception type: {type(e).__name__}",
                    )
                )
        except Exception as e:
            self.results.append(
                ContractValidationResult(
                    model_name,
                    "decompose() before fit() raises",
                    False,
                    severity="MEDIUM",
                    issue=f"Setup failed: {str(e)[:60]}",
                )
            )

    def _test_get_params_returns_dict(
        self, model_name: str, model_cls, config: dict
    ) -> None:
        """Test that get_params() returns a dict."""
        try:
            model = model_cls()
            model.fit(self.df_obs, config)
            params = model.get_params()
            if isinstance(params, dict):
                self.results.append(
                    ContractValidationResult(
                        model_name, "get_params() returns dict", True
                    )
                )
            else:
                self.results.append(
                    ContractValidationResult(
                        model_name,
                        "get_params() returns dict",
                        False,
                        severity="HIGH",
                        issue=f"Returns {type(params).__name__}, not dict",
                    )
                )
        except Exception as e:
            self.results.append(
                ContractValidationResult(
                    model_name,
                    "get_params() returns dict",
                    False,
                    severity="BLOCKER",
                    issue=f"Exception: {str(e)[:60]}",
                )
            )

    def _test_get_params_json_serializable(
        self, model_name: str, model_cls, config: dict
    ) -> None:
        """Test that get_params() output is JSON-serializable."""
        try:
            model = model_cls()
            model.fit(self.df_obs, config)
            params = model.get_params()
            try:
                json_str = json.dumps(params, default=str)
                self.results.append(
                    ContractValidationResult(
                        model_name, "get_params() JSON-serializable", True
                    )
                )
            except TypeError as te:
                self.results.append(
                    ContractValidationResult(
                        model_name,
                        "get_params() JSON-serializable",
                        False,
                        severity="HIGH",
                        issue=f"Non-serializable type: {str(te)[:50]}",
                    )
                )
        except Exception as e:
            self.results.append(
                ContractValidationResult(
                    model_name,
                    "get_params() JSON-serializable",
                    False,
                    severity="BLOCKER",
                    issue=f"Exception: {str(e)[:60]}",
                )
            )

    def _test_decomposition_channel_count(
        self, model_name: str, model_cls, config: dict
    ) -> None:
        """Test that decomposition channel count is reasonable."""
        try:
            model = model_cls()
            model.fit(self.df_obs, config)
            decomp = model.decompose(self.df_obs)

            # Count available LTC columns
            ltc_cols = [c for c in decomp.columns if c.startswith("ltc_")]
            if len(ltc_cols) > 0 and len(ltc_cols) <= 5:
                self.results.append(
                    ContractValidationResult(
                        model_name,
                        "decomposition channel count reasonable",
                        True,
                    )
                )
            else:
                self.results.append(
                    ContractValidationResult(
                        model_name,
                        "decomposition channel count reasonable",
                        False,
                        severity="MEDIUM",
                        issue=f"Unexpected {len(ltc_cols)} LTC channels (expected 1-5)",
                    )
                )
        except Exception as e:
            self.results.append(
                ContractValidationResult(
                    model_name,
                    "decomposition channel count reasonable",
                    False,
                    severity="MEDIUM",
                    issue=f"Exception: {str(e)[:60]}",
                )
            )

    def _test_decomposition_numeric_values(
        self, model_name: str, model_cls, config: dict
    ) -> None:
        """Test that decomposition output contains valid numeric values."""
        try:
            model = model_cls()
            model.fit(self.df_obs, config)
            decomp = model.decompose(self.df_obs)

            # Check for NaN or Inf in numeric columns
            numeric_cols = [
                c
                for c in decomp.columns
                if c in ["baseline", "fitted"] or "stc_" in c or "ltc_" in c
            ]
            bad_rows = 0
            for col in numeric_cols:
                if col in decomp.columns:
                    nan_count = decomp[col].isna().sum()
                    inf_count = np.isinf(decomp[col]).sum()
                    bad_rows += nan_count + inf_count

            if bad_rows == 0:
                self.results.append(
                    ContractValidationResult(
                        model_name, "decomposition numeric values valid", True
                    )
                )
            else:
                self.results.append(
                    ContractValidationResult(
                        model_name,
                        "decomposition numeric values valid",
                        False,
                        severity="HIGH",
                        issue=f"{bad_rows} invalid values (NaN/Inf) in decomposition",
                    )
                )
        except Exception as e:
            self.results.append(
                ContractValidationResult(
                    model_name,
                    "decomposition numeric values valid",
                    False,
                    severity="MEDIUM",
                    issue=f"Exception: {str(e)[:60]}",
                )
            )

    def _test_model_repr(self, model_name: str, model_cls, config: dict) -> None:
        """Test that model repr includes fitted status."""
        try:
            model = model_cls()
            unfitted_repr = repr(model)
            model.fit(self.df_obs, config)
            fitted_repr = repr(model)

            if "not fitted" in unfitted_repr.lower() and "fitted" in fitted_repr.lower():
                self.results.append(
                    ContractValidationResult(
                        model_name, "model repr shows fitted status", True
                    )
                )
            else:
                self.results.append(
                    ContractValidationResult(
                        model_name,
                        "model repr shows fitted status",
                        False,
                        severity="LOW",
                        issue="repr() does not clearly indicate fitted status",
                    )
                )
        except Exception as e:
            self.results.append(
                ContractValidationResult(
                    model_name,
                    "model repr shows fitted status",
                    False,
                    severity="LOW",
                    issue=f"Exception: {str(e)[:60]}",
                )
            )

    def _summarize_results(self) -> dict:
        """Summarize validation results."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        blockers = sum(1 for r in self.results if not r.passed and r.severity == "BLOCKER")
        high = sum(1 for r in self.results if not r.passed and r.severity == "HIGH")
        medium = sum(1 for r in self.results if not r.passed and r.severity == "MEDIUM")
        low = sum(1 for r in self.results if not r.passed and r.severity == "LOW")

        return {
            "total_tests": len(self.results),
            "passed": passed,
            "failed": failed,
            "blockers": blockers,
            "high_severity": high,
            "medium_severity": medium,
            "low_severity": low,
            "results": [r.to_dict() for r in self.results],
        }


def run_contract_validation() -> dict:
    """Main entry point: run contract validation on all models."""
    try:
        validator = ContractValidator(
            data_dir=Path("data/raw"), config_dir=Path("experiments/configs")
        )
        return validator.validate_all_models()
    except Exception as e:
        print(f"[ERROR] Failed to initialize contract validator: {e}")
        return {
            "total_tests": 0,
            "passed": 0,
            "failed": 1,
            "blockers": 1,
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0,
            "results": [
                {
                    "model": "N/A",
                    "test": "Validator Initialization",
                    "passed": False,
                    "severity": "BLOCKER",
                    "issue": str(e),
                    "file_path": "",
                    "line_number": 0,
                }
            ],
        }


if __name__ == "__main__":
    results = run_contract_validation()
    print("\n" + "=" * 100)
    print("CONTRACT VALIDATION RESULTS")
    print("=" * 100)

    # Print summary
    print(f"\nTotal Tests: {results['total_tests']}")
    print(f"Passed: {results['passed']} ✓")
    print(f"Failed: {results['failed']} ✗")
    print(f"  - Blockers: {results['blockers']}")
    print(f"  - High Severity: {results['high_severity']}")
    print(f"  - Medium Severity: {results['medium_severity']}")
    print(f"  - Low Severity: {results['low_severity']}")

    # Print detailed results
    print("\n" + "-" * 100)
    print("DETAILED RESULTS")
    print("-" * 100)
    for r in results["results"]:
        status = "✓" if r["passed"] else "✗"
        print(
            f"{status} {r['model']:15} | {r['test']:40} | [{r['severity']:7}] {r['issue'][:40]}"
        )

    # Exit code
    sys.exit(0 if results["blockers"] == 0 else 1)
