"""
validation_pipeline.validate_before_benchmark - Main validation orchestrator.

Runs all 3 validators (Part 1, 2, 3) before benchmarking:
  1. contract_validator.py — Interface compliance tests
  2. code_audit.py — Structural code review
  3. edge_case_benchmark.py — Edge case integration tests

Generates a comprehensive report:
  - validation_pipeline/VALIDATION_REPORT_{timestamp}.md
  - Exit code 0 (pass) or 1 (fail)

Returns summary dict with keys:
  - total_issues: sum of all findings
  - blockers: count of BLOCKER-severity issues
  - high_severity: count of HIGH-severity issues
  - passed: number of tests that passed
  - report_path: path to detailed markdown report
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Allow importing from ltc package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation_pipeline.contract_validator import run_contract_validation
from validation_pipeline.code_audit import run_code_audit
from validation_pipeline.edge_case_benchmark import run_edge_case_benchmark


def run_validation() -> Dict[str, Any]:
    """Run all validation checks."""
    print("[validation] Starting pre-benchmark validation pipeline...")

    # Phase 1: Contract validation
    print("[validation] Phase 1: Contract validation (interface compliance)...")
    contract_results = run_contract_validation()
    print(f"  → {contract_results['passed']}/{contract_results['total_tests']} tests passed")

    # Phase 2: Code audit
    print("[validation] Phase 2: Code audit (structural checks)...")
    audit_results = run_code_audit()
    print(f"  → {audit_results['total_issues']} issues found")

    # Phase 3: Edge case benchmark
    print("[validation] Phase 3: Edge case benchmark...")
    edge_results = run_edge_case_benchmark()
    print(f"  → {edge_results['passes']}/{edge_results['total_tests']} edge cases passed")

    # Aggregate results
    blockers = (
        contract_results.get("blockers", 0)
        + audit_results.get("blockers", 0)
        + edge_results.get("failures", 0)
    )
    high_severity = (
        contract_results.get("high_severity", 0)
        + audit_results.get("high_severity", 0)
        + edge_results.get("flags", 0)
    )

    # Generate report
    report_path = _generate_report(contract_results, audit_results, edge_results)

    summary = {
        "passed": contract_results.get("passed", 0),
        "total_issues": (
            contract_results.get("failed", 0)
            + audit_results.get("total_issues", 0)
            + edge_results.get("failures", 0)
        ),
        "blockers": blockers,
        "high_severity": high_severity,
        "report_path": str(report_path),
        "contract_results": contract_results,
        "audit_results": audit_results,
        "edge_results": edge_results,
    }

    return summary


def _generate_report(
    contract_results: Dict,
    audit_results: Dict,
    edge_results: Dict,
) -> Path:
    """Generate a comprehensive markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = Path("validation_pipeline") / f"VALIDATION_REPORT_{timestamp}.md"

    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate numbers
    contract_blockers = contract_results.get("blockers", 0)
    contract_high = contract_results.get("high_severity", 0)
    contract_passed = contract_results.get("passed", 0)
    contract_total = contract_results.get("total_tests", 0)

    audit_blockers = audit_results.get("blockers", 0)
    audit_high = audit_results.get("high_severity", 0)
    audit_issues = audit_results.get("total_issues", 0)

    edge_failures = edge_results.get("failures", 0)
    edge_flags = edge_results.get("flags", 0)
    edge_total = edge_results.get("total_tests", 0)

    total_blockers = contract_blockers + audit_blockers + edge_failures
    total_high = contract_high + audit_high + edge_flags

    # Build report
    with open(report_path, "w") as f:
        f.write("# Pre-Benchmark Validation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(
            f"| Metric | Value | Status |\n"
            f"|--------|-------|--------|\n"
            f"| Total Blockers | {total_blockers} | {'✓ PASS' if total_blockers == 0 else '✗ FAIL'} |\n"
            f"| High Severity Issues | {total_high} | {'✓ PASS' if total_high == 0 else '⚠ REVIEW'} |\n"
            f"| Contract Tests Passed | {contract_passed}/{contract_total} | {'✓ PASS' if contract_blockers == 0 else '✗ FAIL'} |\n"
            f"| Code Audit Issues | {audit_issues} | {'✓ PASS' if audit_blockers == 0 else '✗ FAIL'} |\n"
            f"| Edge Cases Passed | {edge_results.get('passes', 0)}/{edge_total} | {'✓ PASS' if edge_failures == 0 else '✗ FAIL'} |\n"
        )

        # Recommendation
        f.write("\n## Recommendation\n\n")
        if total_blockers > 0:
            f.write(
                f"❌ **DO NOT BENCHMARK**: {total_blockers} blocking issues must be fixed before proceeding.\n\n"
            )
        elif total_high > 0:
            f.write(
                f"⚠️ **PROCEED WITH CAUTION**: {total_high} high-severity issues detected. Review before publication.\n\n"
            )
        else:
            f.write("✓ **SAFE TO PROCEED**: All validation checks passed.\n\n")

        # Phase 1: Contract Validation
        f.write("---\n\n")
        f.write("## Phase 1: Contract Validation\n\n")
        f.write(
            f"**Status**: {contract_passed}/{contract_total} tests passed | "
            f"Blockers: {contract_blockers} | High: {contract_high}\n\n"
        )

        f.write("### Detailed Results\n\n")
        f.write(
            "| Model | Test | Status | Severity | Issue |\n"
            "|-------|------|--------|----------|-------|\n"
        )
        for result in contract_results.get("results", []):
            status = "✓" if result["passed"] else "✗"
            f.write(
                f"| {result['model']} | {result['test'][:25]} | {status} | "
                f"{result['severity']} | {result['issue'][:50]} |\n"
            )

        # Phase 2: Code Audit
        f.write("\n---\n\n")
        f.write("## Phase 2: Code Audit\n\n")
        f.write(
            f"**Status**: {audit_issues} issues found | "
            f"Blockers: {audit_blockers} | High: {audit_high}\n\n"
        )

        if audit_issues == 0:
            f.write("✓ No structural issues found.\n\n")
        else:
            f.write("### Issues by Severity\n\n")
            for severity in ["BLOCKER", "HIGH", "MEDIUM", "LOW"]:
                severity_issues = [
                    r for r in audit_results.get("results", []) if r["severity"] == severity
                ]
                if severity_issues:
                    f.write(f"#### {severity}\n\n")
                    f.write(
                        "| File | Line | Check | Issue |\n"
                        "|------|------|-------|-------|\n"
                    )
                    for result in severity_issues:
                        f.write(
                            f"| {Path(result['file']).name} | {result['line']} | "
                            f"{result['check'][:25]} | {result['issue'][:50]} |\n"
                        )
                    f.write("\n")

        # Phase 3: Edge Case Benchmark
        f.write("---\n\n")
        f.write("## Phase 3: Edge Case Benchmark\n\n")
        f.write(
            f"**Status**: {edge_results.get('passes', 0)}/{edge_total} edge cases passed | "
            f"Flags: {edge_flags} | Failures: {edge_failures}\n\n"
        )

        f.write("### Fragility by Model\n\n")
        fragility = edge_results.get("fragility_by_model", {})
        if fragility:
            f.write(
                "| Model | Passes | Flags | Failures | Fragility |\n"
                "|-------|--------|-------|----------|----------|\n"
            )
            for model_name in sorted(fragility.keys()):
                counts = fragility[model_name]
                passes = counts.get("passes", 0)
                flags = counts.get("flags", 0)
                failures = counts.get("failures", 0)
                total = passes + flags + failures
                fragility_score = (flags + failures * 2) / total if total > 0 else 0
                f.write(
                    f"| {model_name} | {passes} | {flags} | {failures} | {fragility_score:.2f} |\n"
                )
            f.write("\n")

        f.write("### Detailed Results\n\n")
        f.write("Only flagged and failed edge cases are shown below.\n\n")
        edge_issues = [r for r in edge_results.get("results", []) if r["status"] != "PASS"]
        if edge_issues:
            f.write(
                "| Model | Scenario | Recovery | MAPE | Delta | Status | Issue |\n"
                "|-------|----------|----------|------|-------|--------|-------|\n"
            )
            for result in edge_issues:
                f.write(
                    f"| {result['model']} | {result['scenario']} | {result['recovery_pct']:.1f}% | "
                    f"{result['ltc_mape']:.1f}% | {result['delta_pp']:+.1f}pp | {result['status']} | "
                    f"{result['issue'][:40]} |\n"
                )
        else:
            f.write("✓ All edge cases passed.\n\n")

        # Next Steps
        f.write("---\n\n")
        f.write("## Next Steps\n\n")
        if total_blockers > 0:
            f.write(f"1. **Fix all {total_blockers} blocking issues** listed above.\n")
            f.write("2. Re-run validation: `python validation_pipeline/validate_before_benchmark.py`\n")
            f.write("3. Only proceed to benchmarking once all blockers are resolved.\n\n")
        else:
            f.write("✓ Validation complete. You may now proceed to benchmarking.\n")
            f.write("  Run: `python experiments/run_experiment.py --all-models --all-scenarios`\n\n")

    return report_path


def main() -> int:
    """Main entry point."""
    print("\n" + "=" * 100)
    print("PRE-BENCHMARK VALIDATION PIPELINE")
    print("=" * 100 + "\n")

    summary = run_validation()

    print("\n" + "=" * 100)
    print("VALIDATION SUMMARY")
    print("=" * 100)
    print(f"\nPassed Tests: {summary['passed']}")
    print(f"Total Issues: {summary['total_issues']}")
    print(f"Blocking Issues: {summary['blockers']}")
    print(f"High Severity: {summary['high_severity']}")

    print(f"\nDetailed Report: {summary['report_path']}")

    # Print recommendation
    print("\n" + "-" * 100)
    if summary["blockers"] > 0:
        print(f"❌ DO NOT BENCHMARK: {summary['blockers']} blocking issues must be fixed.")
        return 1
    elif summary["high_severity"] > 0:
        print(f"⚠️ PROCEED WITH CAUTION: {summary['high_severity']} high-severity issues found.")
        return 0  # Still allow benchmarking but with warning
    else:
        print("✓ SAFE TO PROCEED: All validation checks passed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
