# Validation Pipeline System Overview

## Executive Summary

The **Validation Pipeline** is a 3-part automated structural prevention system designed to catch interface violations, code quality issues, and model fragility **before** expensive benchmarking runs. It runs in ~2-3 minutes and provides comprehensive reporting via markdown.

**Key Benefits:**
- ✓ Catches interface violations before benchmarking (saves hours)
- ✓ Detects structural anti-patterns via static code analysis
- ✓ Tests robustness to edge cases (missing channels, weak signals)
- ✓ Generates detailed markdown reports with remediation guidance
- ✓ Integrates seamlessly into benchmarking workflow

**Exit Codes:**
- `0` = Safe to benchmark (pass or high-severity warnings only)
- `1` = Blocking issues found (MUST fix before benchmarking)

---

## System Architecture

```
validation_pipeline/
├── __init__.py                      # Package entry point
├── contract_validator.py            # PART 1: Interface compliance tests
├── code_audit.py                    # PART 2: Structural code review
├── edge_case_benchmark.py           # PART 3: Edge case integration tests
├── validate_before_benchmark.py     # ORCHESTRATOR: Runs all 3 + generates report
├── test_validation.py               # Quick test of all components
├── README.md                        # Detailed architecture & troubleshooting
├── USAGE_GUIDE.md                   # Quick start & common scenarios
├── SYSTEM_OVERVIEW.md               # This file
└── VALIDATION_REPORT_*.md           # Generated reports (one per run)
```

---

## Part 1: Contract Validator

**File**: `contract_validator.py`
**Runtime**: ~10-15 seconds
**Tests per Model**: 10

### What It Does

Verifies that all 10 models comply with `BaseLTCModel` interface:

```python
class BaseLTCModel(ABC):
    def fit(self, df: pd.DataFrame, config: dict) -> "BaseLTCModel":
        """Fit model on observed data; return self"""
    
    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with columns: baseline, stc_{ch}, ltc_{ch}, fitted"""
    
    def get_params(self) -> dict:
        """Return fitted parameters as JSON-serializable dict"""
```

### Tests Performed

| # | Test | Purpose | Severity if Failed |
|---|------|---------|-------------------|
| 1 | `fit()` returns self | Chainable interface | BLOCKER |
| 2 | `fit()` sets `_is_fitted=True` | Prevent premature decompose() | HIGH |
| 3 | `decompose()` returns DataFrame | Correct return type | BLOCKER |
| 4 | Required columns present | baseline, stc_{ch}, ltc_{ch}, fitted | BLOCKER |
| 5 | `decompose()` before `fit()` raises | Fitted check works | HIGH |
| 6 | `get_params()` returns dict | Correct return type | HIGH |
| 7 | `get_params()` JSON-serializable | No numpy arrays, custom objects | HIGH |
| 8 | Channel count reasonable (1-5) | Sanity check | MEDIUM |
| 9 | No NaN/Inf in decomposition | Valid numeric values | HIGH |
| 10 | Model repr shows fitted status | Good debugging UX | LOW |

### Example Output

```
✓ geo_adstock        | fit() returns self                  | ✓ PASS
✓ geo_adstock        | fit() sets _is_fitted=True          | ✓ PASS
✓ geo_adstock        | decompose() returns DataFrame       | ✓ PASS
✓ geo_adstock        | decompose() required columns        | ✓ PASS
...
✗ weibull_adstock    | get_params() JSON-serializable      | ✗ FAIL [HIGH]
```

### Run Independently

```bash
python validation_pipeline/contract_validator.py
```

---

## Part 2: Code Audit

**File**: `code_audit.py`
**Runtime**: ~1-2 seconds
**Checks per Model**: 5 (static analysis)

### What It Does

Scans model source code for common anti-patterns and structural issues:

### Checks Performed

| Check | Pattern | Risk | Severity |
|-------|---------|------|----------|
| **Hardcoded Indices** | `_coefs[0]`, `_coefs[1]` | Brittle indexing; channel misattribution | MEDIUM |
| **Orphaned Config** | `config.get("key")` never used | Dead code; parameter ignored | LOW |
| **Coefficient Serialization** | `get_params()` incomplete | Reproducibility lost | MEDIUM |
| **Error Handling** | `df["col"]` without try/except | Crashes on missing columns | MEDIUM |
| **Fitted State Check** | `decompose()` no `_check_fitted()` | Crashes if called before fit() | HIGH |

### Example Output

```
ltc/models/framework1/weibull_regression.py:145 | [MEDIUM] Hardcoded index [0]
ltc/models/framework2/ardl_model.py:89          | [HIGH]   Missing fitted check
ltc/models/framework3/mcmc_latent_stock.py:203  | [MEDIUM] Orphaned config 'jump_scale'
```

### Run Independently

```bash
python validation_pipeline/code_audit.py
```

---

## Part 3: Edge Case Benchmark

**File**: `edge_case_benchmark.py`
**Runtime**: ~60-90 seconds
**Scenarios**: 4 per model; 10 models × 4 = 40 total

### What It Does

Tests model robustness to realistic edge cases before full benchmarking:

### Edge Cases Tested

| Scenario | Description | Tests | Purpose |
|----------|-------------|-------|---------|
| **S1_clean** | Baseline: all 5 channels | Baseline recovery | Reference |
| **S1_missing_search** | No search impressions | Handles missing channels | Is model robust? |
| **S1_missing_tv** | No TV impressions | Handles missing channels | Is model robust? |
| **S1_weak_exog** | Exog effects = mean (no variation) | Handles weak signals | Depends on exog? |

### Fragility Score

```
Fragility = (flags + failures × 2) / total_edge_cases

Score < 0.25 → 🟢 ROBUST   (minimal sensitivity)
Score 0.25–0.50 → 🟡 MODERATE  (some sensitivity)
Score > 0.50 → 🔴 FRAGILE   (high sensitivity, review before pub)
```

### Example Output

```
geo_adstock   | Passes: 3, Flags: 1, Failures: 0 | Fragility: 0.17 | 🟢 ROBUST
ardl          | Passes: 2, Flags: 2, Failures: 0 | Fragility: 0.50 | 🟡 MODERATE
dual_adstock  | Passes: 0, Flags: 0, Failures: 4 | Fragility: 2.00 | 🔴 FRAGILE
```

### Run Independently

```bash
python validation_pipeline/edge_case_benchmark.py
```

---

## Part 4: Orchestrator & Reporting

**File**: `validate_before_benchmark.py`
**Runtime**: ~120 seconds total (includes all 3 parts)

### What It Does

1. Runs all three validators in sequence
2. Aggregates results
3. Generates comprehensive markdown report
4. Returns summary dict for programmatic use
5. Sets exit code: 0 (pass) or 1 (blockers found)

### Main Entry Point

```python
from validation_pipeline.validate_before_benchmark import run_validation

result = run_validation()

# result contains:
{
    "passed": 100,              # Number of tests that passed
    "total_issues": 0,          # Total issues across all validators
    "blockers": 0,              # Number of BLOCKER issues
    "high_severity": 0,         # Number of HIGH issues
    "report_path": "...",       # Path to markdown report
    "contract_results": {...},  # Detailed results from Part 1
    "audit_results": {...},     # Detailed results from Part 2
    "edge_results": {...},      # Detailed results from Part 3
}
```

### Run Full Validation

```bash
python validation_pipeline/validate_before_benchmark.py
```

### Report Format

Generated at: `validation_pipeline/VALIDATION_REPORT_2026-06-23_14-30-45.md`

**Sections:**
1. Summary table (blockers, high severity, test counts)
2. Recommendation (DO NOT BENCHMARK / PROCEED WITH CAUTION / SAFE)
3. Phase 1 Results (contract validation by model)
4. Phase 2 Results (code audit by severity)
5. Phase 3 Results (edge case fragility by model)
6. Next Steps (remediation guidance)

---

## Integration with Benchmarking

### Automatic Validation Hook

Modified `experiments/run_experiment.py` to automatically run validation:

```python
from validation_pipeline.validate_before_benchmark import run_validation

def main(...):
    # Run validation first
    validation_result = run_validation()
    
    # Check for blockers
    if validation_result['blockers'] > 0:
        click.echo(f"❌ FATAL: {validation_result['blockers']} blocking issues")
        click.echo(f"Review: {validation_result['report_path']}")
        sys.exit(1)
    
    # Warn about high-severity issues
    if validation_result['high_severity'] > 0:
        click.echo(f"⚠️ WARNING: {validation_result['high_severity']} high-severity issues")
        click.echo(f"Review: {validation_result['report_path']}")
    
    # Proceed with benchmarking
    # ... run experiments ...
```

### Usage

Benchmarking now automatically validates:

```bash
python experiments/run_experiment.py --all-models --all-scenarios
```

This command:
1. Runs validation (~2 min)
2. If blockers found, aborts with clear error
3. If no blockers, proceeds to benchmarking (~60 min for full run)
4. Prints validation summary at start

### Exit Codes

- `0` = Experiments completed (regardless of results)
- `1` = Blocking validation issues found (aborted before benchmarking)

---

## Severity Level Definitions

### BLOCKER (Exit Code 1)

**When to use**: Critical issues that prevent proper testing

**Examples**:
- Model crashes during fit()
- decompose() returns wrong type
- get_params() missing return statement
- Interface contract violated

**Action**: **MUST FIX before benchmarking**. Abort pipeline with error code 1.

### HIGH

**When to use**: Significant structural issues that affect reliability

**Examples**:
- fit() doesn't set _is_fitted=True
- Missing columns in decomposition
- No error handling for missing inputs
- Fragility > 1.0pp on edge cases

**Action**: Fix if possible; proceed with warning and note in documentation.

### MEDIUM

**When to use**: Code quality or robustness issues

**Examples**:
- Hardcoded array indices instead of channel dicts
- Incomplete coefficient serialization
- Orphaned configuration parameters
- Fragility 0.5–1.0pp on edge cases

**Action**: Review and consider fixing before publication.

### LOW

**When to use**: Minor suggestions; doesn't affect functionality

**Examples**:
- Inconsistent naming conventions
- Dead code comments
- Missing docstrings
- Fragility < 0.5pp on edge cases

**Action**: Nice to fix; not blocking.

---

## Example Validation Run

### Scenario: All Tests Pass

```bash
$ python validation_pipeline/validate_before_benchmark.py

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

$ echo $?
0
```

### Scenario: Blocking Issues Found

```bash
$ python validation_pipeline/validate_before_benchmark.py

[validation] Phase 1: Contract validation...
  ✗ weibull_adstock: get_params() returns non-dict
[validation] Phase 2: Code audit...
  ✗ ardl_model.py: Missing fitted check
...

====================================================================
❌ FATAL: 2 blocking issue(s) found.
DO NOT PROCEED WITH BENCHMARKING.

Review: validation_pipeline/VALIDATION_REPORT_2026-06-23_14-30-45.md

$ echo $?
1
```

---

## File Organization

```
ltc/
├── models/
│   ├── base.py                      # ← Interface all validators test against
│   ├── framework1/
│   │   ├── geometric_regression.py
│   │   ├── weibull_regression.py
│   │   ├── almon_regression.py
│   │   └── dual_adstock.py
│   ├── framework2/
│   │   ├── koyck_model.py
│   │   ├── ardl_model.py
│   │   └── finite_dl_model.py
│   └── framework3/
│       ├── kalman_dlm.py
│       ├── mcmc_latent_stock.py
│       └── bayesian_sts.py
└── data/
    └── loader.py                    # ← Used by edge case benchmark
```

```
validation_pipeline/
├── contract_validator.py            # Tests fit/decompose/get_params on all models
├── code_audit.py                    # Static analysis of model code
├── edge_case_benchmark.py           # Loads scenarios, runs models, compares recovery
├── validate_before_benchmark.py     # Orchestrates all 3, generates report
├── test_validation.py               # Quick test of all components
├── README.md                        # Architecture & troubleshooting (detailed)
├── USAGE_GUIDE.md                   # Quick start & scenarios
├── SYSTEM_OVERVIEW.md               # This file
└── VALIDATION_REPORT_*.md           # Generated reports
```

```
experiments/
├── run_experiment.py                # ← MODIFIED to include validation hook
├── registry.py                      # Model registry used by all validators
└── configs/
    ├── framework1.yaml              # Used by edge case benchmark
    ├── framework2.yaml
    └── framework3.yaml
```

---

## Dependencies

```python
# Core dependencies (already in environment)
numpy
pandas
scipy
click
yaml

# Specific to validation pipeline
# (all imports are standard library or above)
```

No new dependencies required. Validation pipeline uses only standard library + existing project dependencies.

---

## Performance Characteristics

| Component | Time | Scenarios | Models | Total Runs |
|-----------|------|-----------|--------|-----------|
| Contract Validator | ~10-15s | 1 | 10 | 100 tests |
| Code Audit | ~1-2s | — | 10 | Static analysis |
| Edge Case Benchmark | ~60-90s | 4 | 10 | 40 model runs |
| Orchestrator + Reporting | ~5s | — | — | Report generation |
| **TOTAL** | **~90-120s** | | | |

To skip edge cases (faster validation for quick checks):

```bash
python validation_pipeline/contract_validator.py
python validation_pipeline/code_audit.py
# Total: ~15-20 seconds
```

---

## Success Criteria

A model passes validation if:

1. **Contract Validation**
   - ✓ fit() returns self
   - ✓ fit() sets _is_fitted=True
   - ✓ decompose() returns DataFrame with required columns
   - ✓ decompose() raises before fit()
   - ✓ get_params() returns JSON-serializable dict
   - ✓ No NaN/Inf in output

2. **Code Audit**
   - ✓ No BLOCKER or HIGH severity issues
   - ✓ Error handling for missing columns
   - ✓ Fitted check in decompose()

3. **Edge Case Benchmark**
   - ✓ Runs without crashing on all 4 scenarios
   - ✓ Fragility score < 1.0 (prefers < 0.25)
   - ✓ Recovery doesn't degrade >1pp on edge cases

---

## Future Enhancements (Out of Scope)

Potential additions for future iterations:

- [ ] Automated fix suggestions (code generation for common issues)
- [ ] Performance benchmarking (fit time, memory usage per model)
- [ ] Reproducibility checks (round-trip serialization tests)
- [ ] Statistical tests (significance of recovery improvements)
- [ ] CI/CD integration (GitHub Actions pipeline)
- [ ] Dashboard (web UI for historical validation results)

---

## Questions?

See:
1. **Quick Start**: `USAGE_GUIDE.md`
2. **Detailed Architecture**: `README.md`
3. **This Overview**: `SYSTEM_OVERVIEW.md`
4. **Test Components**: `python validation_pipeline/test_validation.py`

