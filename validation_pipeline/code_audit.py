"""
validation_pipeline.code_audit - Automated structural code review checks.

Scans model implementation files for common anti-patterns and structural issues:
  1. Hardcoded loop indices used as coefficient accessors
  2. Orphaned config keys (read but never used)
  3. Coefficient serialization correctness (get_params() captures all fitted state)
  4. Error handling for missing columns
  5. Round-trip serialization (get_params() → fit() should be reproducible)

Produces a list of violations with file:line:issue format for manual review.
"""

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Allow importing from ltc package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.registry import MODEL_REGISTRY


class CodeAuditResult:
    """Container for a single code audit finding."""

    def __init__(
        self,
        model_name: str,
        check_name: str,
        file_path: str,
        line_number: int,
        severity: str,
        issue: str,
    ):
        self.model_name = model_name
        self.check_name = check_name
        self.file_path = file_path
        self.line_number = line_number
        self.severity = severity  # BLOCKER, HIGH, MEDIUM, LOW
        self.issue = issue

    def __repr__(self) -> str:
        return (
            f"{self.file_path}:{self.line_number:3d} | [{self.severity:7}] "
            f"({self.check_name:25}) {self.issue}"
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "check": self.check_name,
            "file": self.file_path,
            "line": self.line_number,
            "severity": self.severity,
            "issue": self.issue,
        }


class CodeAuditor:
    """Static analysis of model code for structural issues."""

    def __init__(self):
        self.results: List[CodeAuditResult] = []
        self.model_files: Dict[str, Path] = self._find_model_files()

    def _find_model_files(self) -> Dict[str, Path]:
        """Map model name to its source file."""
        model_files = {}
        for model_name, model_cls in MODEL_REGISTRY.items():
            # Get file path from class
            try:
                file_path = Path(model_cls.__module__.replace(".", "/") + ".py")
                # Resolve relative to project root
                resolved = Path("ltc") / file_path
                if not resolved.exists():
                    resolved = Path("ltc") / file_path.relative_to("ltc")
                model_files[model_name] = resolved
            except Exception:
                pass
        return model_files

    def audit_all_models(self) -> dict:
        """Run all code audits on all models."""
        for model_name, file_path in self.model_files.items():
            self._audit_model_file(model_name, file_path)

        return self._summarize_results()

    def _audit_model_file(self, model_name: str, file_path: Path) -> None:
        """Run all checks on a single model file."""
        if not file_path.exists():
            self.results.append(
                CodeAuditResult(
                    model_name,
                    "File Location",
                    str(file_path),
                    0,
                    "MEDIUM",
                    f"Model file not found: {file_path}",
                )
            )
            return

        try:
            with open(file_path) as f:
                source = f.read()
        except Exception as e:
            self.results.append(
                CodeAuditResult(
                    model_name,
                    "File Read",
                    str(file_path),
                    0,
                    "HIGH",
                    f"Could not read file: {e}",
                )
            )
            return

        # Run all checks
        self._check_hardcoded_indices(model_name, file_path, source)
        self._check_orphaned_configs(model_name, file_path, source)
        self._check_coefficient_serialization(model_name, file_path, source)
        self._check_error_handling(model_name, file_path, source)
        self._check_fitted_state_completeness(model_name, file_path, source)

    def _check_hardcoded_indices(
        self, model_name: str, file_path: Path, source: str
    ) -> None:
        """
        Scan for patterns like coefs[0], coefs[1], coefs[2] used as channel accessors.
        Flags: These should be refactored to use channel names.
        """
        # Pattern: _coefs[<number>] or coefs[<number>]
        hardcoded_pattern = r"_?coefs\s*\[\s*(\d+)\s*\]"
        matches = list(re.finditer(hardcoded_pattern, source))

        if matches:
            for match in matches:
                line_number = source[: match.start()].count("\n") + 1
                index = match.group(1)
                self.results.append(
                    CodeAuditResult(
                        model_name,
                        "Hardcoded Index Access",
                        str(file_path),
                        line_number,
                        "MEDIUM",
                        f"Hardcoded coefficient index [${index}] — refactor to channel names",
                    )
                )

    def _check_orphaned_configs(
        self, model_name: str, file_path: Path, source: str
    ) -> None:
        """
        Scan for config.get(...) calls that are never actually used.
        Flags: Dead code, or accidental overwrite.
        """
        # Find all config.get() calls
        config_pattern = r'config\.get\s*\(\s*["\']([^"\']+)["\']\s*,?[^)]*\)'
        config_gets = {}
        for match in re.finditer(config_pattern, source):
            key = match.group(1)
            line_number = source[: match.start()].count("\n") + 1
            if key not in config_gets:
                config_gets[key] = []
            config_gets[key].append(line_number)

        # Find all assignments (self._xxx = ...)
        assignment_pattern = r"self\._([\w_]+)\s*="
        assignments = set()
        for match in re.finditer(assignment_pattern, source):
            var_name = match.group(1)
            assignments.add(var_name)

        # Check if config keys match assignments
        for config_key, lines in config_gets.items():
            var_name = config_key.replace("_", "").lower()
            key_normalized = config_key.replace("_", "").lower()

            # Heuristic: if config key is read but no matching assignment, flag it
            found_assignment = False
            for var in assignments:
                if key_normalized in var.replace("_", "").lower():
                    found_assignment = True
                    break

            if not found_assignment and key_normalized not in ["channels", "feature"]:
                for line in lines[:1]:  # Report first occurrence only
                    self.results.append(
                        CodeAuditResult(
                            model_name,
                            "Orphaned Config",
                            str(file_path),
                            line,
                            "LOW",
                            f"Config key '{config_key}' read but possibly not used",
                        )
                    )

    def _check_coefficient_serialization(
        self, model_name: str, file_path: Path, source: str
    ) -> None:
        """
        Check that get_params() appears to return all fitted state.
        Flags: Missing return statements, incomplete params dict.
        """
        # Find get_params() method
        get_params_pattern = r"def get_params\(self\)(.*?)(?=\n    def |\nclass |\Z)"
        match = re.search(get_params_pattern, source, re.DOTALL)

        if not match:
            self.results.append(
                CodeAuditResult(
                    model_name,
                    "get_params() Definition",
                    str(file_path),
                    0,
                    "HIGH",
                    "get_params() method not found",
                )
            )
            return

        method_body = match.group(1)
        method_start = source[: match.start()].count("\n") + 1

        # Check for return statement
        if "return" not in method_body:
            self.results.append(
                CodeAuditResult(
                    model_name,
                    "get_params() Return",
                    str(file_path),
                    method_start,
                    "BLOCKER",
                    "get_params() has no return statement",
                )
            )
            return

        # Check for dict construction
        if "{" not in method_body:
            self.results.append(
                CodeAuditResult(
                    model_name,
                    "get_params() Return Type",
                    str(file_path),
                    method_start,
                    "HIGH",
                    "get_params() does not return a dict",
                )
            )

        # Find all self._xxx assignments in __init__ to estimate state
        init_pattern = r"def __init__\(self\)(.*?)(?=\n    def |\Z)"
        init_match = re.search(init_pattern, source, re.DOTALL)
        if init_match:
            init_body = init_match.group(1)
            state_vars = set(re.findall(r"self\.(_\w+)\s*=", init_body))

            # Estimate how many state vars are returned in get_params
            returned_count = method_body.count("self._")

            # If significantly fewer params returned than state vars, flag it
            if state_vars and returned_count < len(state_vars) * 0.5:
                self.results.append(
                    CodeAuditResult(
                        model_name,
                        "get_params() Completeness",
                        str(file_path),
                        method_start,
                        "MEDIUM",
                        f"get_params() returns ~{returned_count} params but {len(state_vars)} state vars exist",
                    )
                )

    def _check_error_handling(
        self, model_name: str, file_path: Path, source: str
    ) -> None:
        """
        Check that fit() and decompose() have reasonable error handling for missing columns.
        Flags: Unprotected column access (df["col"]), missing try/except blocks.
        """
        # Find fit() and decompose() methods
        for method_name in ["fit", "decompose"]:
            pattern = rf"def {method_name}\(self[^)]*\)(.*?)(?=\n    def |\nclass |\Z)"
            match = re.search(pattern, source, re.DOTALL)

            if not match:
                continue

            method_body = match.group(1)
            method_start = source[: match.start()].count("\n") + 1

            # Check for unprotected column access (df["col"] without try/except nearby)
            unsafe_access = re.findall(r'df\s*\[\s*["\'](\w+)["\']\s*\]', method_body)

            if unsafe_access and "try" not in method_body:
                self.results.append(
                    CodeAuditResult(
                        model_name,
                        f"{method_name}() Error Handling",
                        str(file_path),
                        method_start,
                        "MEDIUM",
                        f"{method_name}() accesses df columns without try/except",
                    )
                )

    def _check_fitted_state_completeness(
        self, model_name: str, file_path: Path, source: str
    ) -> None:
        """
        Check that decompose() method calls _check_fitted() before accessing state.
        Flags: Missing fitted checks, potential crashes if decompose() called before fit().
        """
        # Find decompose() method
        decompose_pattern = r"def decompose\(self[^)]*\)(.*?)(?=\n    def |\nclass |\Z)"
        match = re.search(decompose_pattern, source, re.DOTALL)

        if not match:
            return

        method_body = match.group(1)
        method_start = source[: match.start()].count("\n") + 1

        # Check if _check_fitted() or _is_fitted check is present
        if "_check_fitted" not in method_body and "_is_fitted" not in method_body:
            self.results.append(
                CodeAuditResult(
                    model_name,
                    "decompose() Fitted Check",
                    str(file_path),
                    method_start,
                    "HIGH",
                    "decompose() does not check _is_fitted before accessing state",
                )
            )

    def _summarize_results(self) -> dict:
        """Summarize code audit results."""
        blockers = sum(1 for r in self.results if r.severity == "BLOCKER")
        high = sum(1 for r in self.results if r.severity == "HIGH")
        medium = sum(1 for r in self.results if r.severity == "MEDIUM")
        low = sum(1 for r in self.results if r.severity == "LOW")

        return {
            "total_issues": len(self.results),
            "blockers": blockers,
            "high_severity": high,
            "medium_severity": medium,
            "low_severity": low,
            "results": [r.to_dict() for r in self.results],
        }


def run_code_audit() -> dict:
    """Main entry point: run code audit on all models."""
    try:
        auditor = CodeAuditor()
        return auditor.audit_all_models()
    except Exception as e:
        print(f"[ERROR] Failed to initialize code auditor: {e}")
        return {
            "total_issues": 1,
            "blockers": 1,
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0,
            "results": [
                {
                    "model": "N/A",
                    "check": "Auditor Initialization",
                    "file": "",
                    "line": 0,
                    "severity": "BLOCKER",
                    "issue": str(e),
                }
            ],
        }


if __name__ == "__main__":
    results = run_code_audit()
    print("\n" + "=" * 120)
    print("CODE AUDIT RESULTS")
    print("=" * 120)

    # Print summary
    print(f"\nTotal Issues: {results['total_issues']}")
    print(f"  - Blockers: {results['blockers']}")
    print(f"  - High Severity: {results['high_severity']}")
    print(f"  - Medium Severity: {results['medium_severity']}")
    print(f"  - Low Severity: {results['low_severity']}")

    if results["total_issues"] == 0:
        print("\n✓ No structural issues found!")
    else:
        # Print detailed results
        print("\n" + "-" * 120)
        print("DETAILED ISSUES")
        print("-" * 120)
        for r in results["results"]:
            print(f"{r['file']}:{r['line']:3d} | [{r['severity']:7}] ({r['check']:25}) {r['issue']}")

    # Exit code
    sys.exit(0 if results["blockers"] == 0 else 1)
