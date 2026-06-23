# Validation Pipeline: Pre-Benchmark Structural Prevention System

## Overview

The validation pipeline is an automated 3-part prevention system that runs **BEFORE every benchmarking experiment** to detect structural issues, interface violations, and model fragility. It ensures high-quality, reproducible results before expensive benchmark runs.

## Why This Matters

Previous benchmarking runs have uncovered critical issues late (after hours of computation):
- Models violating the `BaseLTCModel` interface (missing `_is_fitted`, wrong return types)
- Orphaned configuration parameters leading to wrong hyperparameter grids
- Models crashing on edge cases (missing channels, weak exogenous signals)
- Coefficient indexing bugs causing silent channel misattribution

The validation pipeline **catches these issues in <2 minutes** before benchmarking starts.

## Quick Start

### Run Full Validation
```bash
python validation_pipeline/validate_before_benchmark.py
```

This runs all three validators in sequence and generates a comprehensive report.

### Run Individual Validators
```bash
# Contract compliance tests only
python validation_pipeline/contract_validator.py

# Code structure audit only
python validation_pipeline/code_audit.py

# Edge case integration tests only
python validation_pipeline/edge_case_benchmark.py
```

## System Architecture

### Part 1: Contract Validator (`contract_validator.py`)

**Purpose**: Verify all 10 models comply with `BaseLTCModel` interface.

**Tests Performed** (10 tests per model × 10 models = 100 total):
1. `fit()` returns self
2. `fit()` sets `_is_fitted=True`
3. `decompose()` returns DataFrame
4. `decompose()` has required columns: `baseline`, `stc_{ch}`, `ltc_{ch}`, `fitted`
5. `decompose()` raises before `fit()`
6. `get_params()` returns dict
7. `get_params()` is JSON-serializable
8. Decomposition channel count is reasonable (1-5 channels)
9. Decomposition numeric values are valid (no NaN/Inf)
10. Model repr includes fitted status

**Output**:
- Console: table with ✓/✗ status per test
- `VALIDATION_REPORT_{timestamp}.md`: detailed results by severity (BLOCKER/HIGH/MEDIUM/LOW)

**Severity Levels**:
- **BLOCKER**: fit/decompose/get_params crash or return wrong type → **DO NOT BENCHMARK**
- **HIGH**: Missing columns, wrong return type → **FIX BEFORE BENCHMARKING**
- **MEDIUM**: Edge cases poorly handled → **REVIEW BEFORE PUBLICATION**
- **LOW**: Minor code quality issues → **NICE TO FIX**

---

### Part 2: Code Audit (`code_audit.py`)

**Purpose**: Detect structural anti-patterns and incomplete implementations via static analysis.

**Checks Performed** (5 checks per model):
1. **Hardcoded Index Access**: Flags `_coefs[0]`, `_coefs[1]`, etc. instead of channel names
   - Risk: Brittle coefficient indexing; incorrect channel attribution
   - Remediation: Refactor to dict-based coefficient storage

2. **Orphaned Config**: Finds `config.get("key")` that don't map to assignments
   - Risk: Dead code or accidental hyperparameter override
   - Remediation: Remove or use the config value

3. **Coefficient Serialization**: Checks `get_params()` returns all fitted state
   - Risk: Incomplete parameters; reproducibility issues
   - Remediation: Add missing fitted attributes to return dict

4. **Error Handling**: Checks `fit()`/`decompose()` have try/except for column access
   - Risk: Crashes on missing columns (e.g., S1_missing_search edge case)
   - Remediation: Wrap column access in try/except or use `.get()` with defaults

5. **Fitted State Completeness**: Checks `decompose()` calls `_check_fitted()` or checks `_is_fitted`
   - Risk: Crashes if `decompose()` called before `fit()`
   - Remediation: Add fitted check at method start

**Output**:
- Console: table with file:line:issue format
- Included in `VALIDATION_REPORT_{timestamp}.md`

**Severity Levels**:
- **BLOCKER**: Missing get_params() return → **CRITICAL**
- **HIGH**: Missing fitted check in decompose() → **HIGH PRIORITY**
- **MEDIUM**: Hardcoded indices, missing error handling → **MEDIUM PRIORITY**
- **LOW**: Orphaned config, minor code quality → **LOW PRIORITY**

---

### Part 3: Edge Case Benchmark (`edge_case_benchmark.py`)

**Purpose**: Test model robustness to edge cases before full benchmarking.

**Edge Cases Tested**:
1. **S1_clean** (baseline): All 5 channels, normal exogenous signals
2. **S1_missing_search**: Only TV, Social, Display, Video (no search)
3. **S1_missing_tv**: Only Search, Social, Display, Video (no TV)
4. **S1_weak_exog**: All 5 channels but exogenous effects set to mean (no variation)

**Metrics per Model×Scenario**:
- Recovery % (how well model estimates true LTC)
- LTC MAPE %
- Δ (delta from baseline, in percentage points)
- Status: PASS (Δ ≤ 0.5pp) or FLAG (Δ > 0.5pp)

**Fragility Score** (per model):
```
Fragility = (flags + failures × 2) / total_edge_cases
```
- Score < 0.25 → 🟢 ROBUST
- Score 0.25-0.50 → 🟡 MODERATE
- Score > 0.50 → 🔴 FRAGILE (revisit model before publication)

**Output**:
- Console: fragility summary by model
- `VALIDATION_REPORT_{timestamp}.md`: detailed results table

**Example Interpretation**:
```
geo_adstock   | Passes: 3, Flags: 1, Failures: 0 | Fragility: 0.17 | 🟢 ROBUST
ardl          | Passes: 2, Flags: 2, Failures: 0 | Fragility: 0.50 | 🟡 MODERATE
dual_adstock  | Passes: 0, Flags: 0, Failures: 4 | Fragility: 2.00 | 🔴 FRAGILE
```

---

## Output Report Format

The validation pipeline generates a comprehensive markdown report:

```
validation_pipeline/VALIDATION_REPORT_2026-06-23_14-30-45.md
```

### Report Sections

1. **Summary**: Quick status table (blockers, high severity, test counts)
2. **Recommendation**: DO NOT BENCHMARK / PROCEED WITH CAUTION / SAFE TO PROCEED
3. **Phase 1 Results**: Contract validation details by model and test
4. **Phase 2 Results**: Code audit issues by severity
5. **Phase 3 Results**: Edge case fragility by model
6. **Next Steps**: Actionable remediation guidance

### Example Report Header

```markdown
# Pre-Benchmark Validation Report

**Generated:** 2026-06-23 14:30:45

## Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total Blockers | 2 | ✗ FAIL |
| High Severity Issues | 5 | ⚠ REVIEW |
| Contract Tests Passed | 98/100 | ✗ FAIL |
| Code Audit Issues | 3 | ✓ PASS |
| Edge Cases Passed | 35/40 | ✗ FAIL |

## Recommendation

❌ **DO NOT BENCHMARK**: 2 blocking issues must be fixed before proceeding.
```

---

## Integration with Benchmarking

### Automatic Validation Hook

To enable automatic validation before benchmarking, modify `experiments/run_experiment.py`:

```python
from validation_pipeline.validate_before_benchmark import run_validation

def main(...):
    # Run validation first
    validation_result = run_validation()
    
    if validation_result['blockers'] > 0:
        print(f"FATAL: {validation_result['blockers']} blocking issues found.")
        print(f"Review: {validation_result['report_path']}")
        sys.exit(1)
    
    if validation_result['high_severity'] > 0:
        print(f"WARNING: {validation_result['high_severity']} high-severity issues.")
        print(f"Review: {validation_result['report_path']}")
        # Continue, but user is warned
    
    # Proceed with benchmarking...
```

### Manual Pre-Flight Check

Before running a large benchmark, manually run:

```bash
python validation_pipeline/validate_before_benchmark.py
cat validation_pipeline/VALIDATION_REPORT_*.md
```

If blockers exist, review the report and fix issues before benchmarking.

---

## Common Issues & Remediation

### Contract Violation: `fit()` does not set `_is_fitted=True`

**Finding**: Model.fit() returns self but doesn't set `_is_fitted=True`

**Root Cause**: Model subclass forgets to call `self._is_fitted = True` at end of fit()

**Fix**:
```python
def fit(self, df, config):
    # ... fitting logic ...
    self._is_fitted = True  # ← ADD THIS
    return self
```

---

### Code Audit: Hardcoded Coefficient Index

**Finding**: File ltc/models/framework1/weibull_regression.py:145: Hardcoded index [0]

**Root Cause**: Model uses `_coefs[0]` instead of channel-keyed dict

**Fix**:
```python
# OLD (brittle):
self._coefs = np.array([...]); print(self._coefs[0])

# NEW (robust):
self._coefs = {"tv": 0.5, "search": 0.2, ...}; print(self._coefs["tv"])
```

---

### Edge Case Flag: Missing Search Channel

**Finding**: Model S1_missing_search recovery drops 15.3pp (FLAG threshold 0.5pp)

**Root Cause**: Model crashes or behaves incorrectly when search channel is zero

**Fix**:
```python
def fit(self, df, config):
    for ch in channels:
        col = f"impr_{ch}"
        if col not in df.columns or df[col].sum() == 0:
            # Gracefully skip or use default coefficient
            self._coefs[ch] = 0.0
            continue
        # Normal fitting logic
```

---

## Running Validation Locally

### Prerequisites

```bash
cd /c/github/ltc
python -m pip install -e .  # Install package in development mode
```

### Full Validation Run

```bash
python validation_pipeline/validate_before_benchmark.py
```

Expected output:
```
[validation] Starting pre-benchmark validation pipeline...
[validation] Phase 1: Contract validation (interface compliance)...
  → 100/100 tests passed
[validation] Phase 2: Code audit (structural checks)...
  → 0 issues found
[validation] Phase 3: Edge case benchmark...
  → 40/40 edge cases passed

====================================================================
VALIDATION SUMMARY
====================================================================

Passed Tests: 100
Total Issues: 0
Blocking Issues: 0
High Severity: 0

Detailed Report: validation_pipeline/VALIDATION_REPORT_2026-06-23_14-30-45.md

----
✓ SAFE TO PROCEED: All validation checks passed.
```

### Expected Run Time

- **Contract Validator**: ~10-15 seconds (loads S1, fits 10 models)
- **Code Audit**: ~1-2 seconds (static analysis only)
- **Edge Case Benchmark**: ~60-90 seconds (loads 4 scenarios, fits 10 models × 4 = 40 runs)
- **Total**: ~90-120 seconds (~2 minutes)

---

## Future Agent Instructions

**Every time you (future agent) are asked to validate, test, or benchmark these models, you MUST:**

1. **Before starting any experiment**, run:
   ```bash
   python validation_pipeline/validate_before_benchmark.py
   ```

2. **Review the validation report**:
   ```bash
   cat validation_pipeline/VALIDATION_REPORT_*.md
   ```

3. **If BLOCKERS exist** (exit code 1):
   - Do NOT proceed to benchmarking
   - Review the report
   - Investigate root causes
   - Fix issues (contact human if unclear)
   - Re-run validation to confirm

4. **If HIGH-SEVERITY issues exist** (exit code 0 but warnings printed):
   - Note the issues in your experiment log
   - Proceed with benchmarking, but flag issues in commit message
   - Before publication, decide whether to fix or document as known limitation

5. **If ALL CHECKS PASS** (exit code 0, no issues):
   - Proceed to benchmarking with confidence
   - Document validation result in commit message: `[validation: PASSED]`

This is a **MANDATORY PRE-FLIGHT CHECK**. Do not skip it.

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `contract_validator.py` | Interface compliance tests (Part 1) |
| `code_audit.py` | Structural code review (Part 2) |
| `edge_case_benchmark.py` | Edge case integration tests (Part 3) |
| `validate_before_benchmark.py` | Orchestrator & report generator (Main) |
| `README.md` | This file |
| `VALIDATION_REPORT_*.md` | Generated reports (one per run) |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All checks passed, safe to benchmark |
| 1 | Blocking issues found, DO NOT BENCHMARK |

Use in shell scripts:
```bash
python validation_pipeline/validate_before_benchmark.py
if [ $? -ne 0 ]; then
    echo "Validation failed. Fix issues before benchmarking."
    exit 1
fi
python experiments/run_experiment.py --all-models --all-scenarios
```

---

## Example Workflow

### Scenario 1: Adding a New Model

```bash
# 1. Implement new model in ltc/models/framework1/my_new_model.py
# 2. Add to experiments/registry.py MODEL_REGISTRY
# 3. Run validation to catch interface violations early
python validation_pipeline/validate_before_benchmark.py

# 4. If validation passes, safe to benchmark
python experiments/run_experiment.py --model my_new_model --scenario S1

# 5. If validation fails, review report and fix issues
cat validation_pipeline/VALIDATION_REPORT_*.md
```

### Scenario 2: Modifying an Existing Model

```bash
# 1. Change model code in ltc/models/...
# 2. Run validation to ensure changes don't break interface
python validation_pipeline/validate_before_benchmark.py

# 3. If changes passed validation, re-run benchmarks
python experiments/run_experiment.py --all-models S1

# 4. Compare new results to previous baseline
```

### Scenario 3: Full Benchmarking Run

```bash
# Pre-flight validation
python validation_pipeline/validate_before_benchmark.py

# If validation passes:
python experiments/run_experiment.py --all-models --all-scenarios

# If validation fails:
# - Review report
# - Fix highest-priority issues
# - Re-validate
# - If still issues, contact human
```

---

## Support

For questions about validation system:
1. Read the relevant section in this README
2. Check the generated VALIDATION_REPORT_{timestamp}.md
3. Review inline code comments in the validator modules
4. Contact repository maintainer

For model-specific issues:
1. Check the specific test/check that failed in the validation report
2. Review the model's docstring and code
3. Consult the base class interface in `ltc/models/base.py`
