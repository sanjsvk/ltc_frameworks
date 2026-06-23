"""
validation_pipeline — Automated structural prevention system for LTC framework benchmarking.

This module provides a 3-part validation system that runs before every benchmark:
  1. contract_validator.py — Interface compliance tests
  2. code_audit.py — Structural code review checks
  3. edge_case_benchmark.py — Edge case integration tests

Main entry point: validate_before_benchmark.run_validation()

Usage:
    from validation_pipeline import validate_before_benchmark
    result = validate_before_benchmark.run_validation()
    if result['blockers'] > 0:
        print(f"DO NOT BENCHMARK: {result['blockers']} blocking issues")
        sys.exit(1)
"""

from validation_pipeline.validate_before_benchmark import run_validation

__all__ = ["run_validation"]
