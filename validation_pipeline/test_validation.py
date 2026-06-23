"""
Quick test of validation pipeline components.

This script tests each validator individually to ensure they work correctly.
"""

import sys
from pathlib import Path

# Allow importing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("=" * 100)
print("VALIDATION PIPELINE TEST")
print("=" * 100)

# Test 1: Code Audit
print("\n[1] Testing Code Audit...")
try:
    from validation_pipeline.code_audit import run_code_audit
    result = run_code_audit()
    print(f"  ✓ Code audit completed: {result['total_issues']} issues found")
except Exception as e:
    print(f"  ✗ Code audit failed: {e}")
    sys.exit(1)

# Test 2: Contract Validator
print("\n[2] Testing Contract Validator...")
try:
    from validation_pipeline.contract_validator import run_contract_validation
    result = run_contract_validation()
    print(f"  ✓ Contract validation completed: {result['passed']}/{result['total_tests']} tests passed")
except Exception as e:
    print(f"  ✗ Contract validation failed: {e}")
    sys.exit(1)

# Test 3: Edge Case Benchmark
print("\n[3] Testing Edge Case Benchmark...")
print("  (This will take ~90 seconds...)")
try:
    from validation_pipeline.edge_case_benchmark import run_edge_case_benchmark
    result = run_edge_case_benchmark()
    print(f"  ✓ Edge case benchmark completed: {result['passes']}/{result['total_tests']} passed")
except Exception as e:
    print(f"  ✗ Edge case benchmark failed: {e}")
    sys.exit(1)

# Test 4: Full Validation
print("\n[4] Testing Full Validation Orchestrator...")
try:
    from validation_pipeline.validate_before_benchmark import run_validation
    result = run_validation()
    print(f"  ✓ Full validation completed")
    print(f"    - Blockers: {result['blockers']}")
    print(f"    - High Severity: {result['high_severity']}")
    print(f"    - Report: {result['report_path']}")
except Exception as e:
    print(f"  ✗ Full validation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 100)
print("✓ ALL TESTS PASSED")
print("=" * 100)
