# Validation Pipeline — Quick Usage Guide

## TL;DR

Before benchmarking, always run:

```bash
python validation_pipeline/validate_before_benchmark.py
```

If exit code is 0 (no blockers), safe to benchmark.
If exit code is 1 (blockers found), review report and fix issues.

---

## Step-by-Step Guide

### Step 1: Run Validation (2-3 minutes)

```bash
cd /c/github/ltc
python validation_pipeline/validate_before_benchmark.py
```

**Expected Output:**
```
====================================================================
PRE-BENCHMARK VALIDATION PIPELINE
====================================================================

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

### Step 2: Check Exit Code

```bash
echo $?
```

- **0** = All tests passed → Safe to benchmark
- **1** = Blocking issues found → DO NOT BENCHMARK

### Step 3: Review Report (if issues found)

```bash
cat validation_pipeline/VALIDATION_REPORT_2026-06-23_14-30-45.md
```

The report contains:
- Summary table of all issues by severity
- Recommendation (DO NOT BENCHMARK / PROCEED WITH CAUTION / SAFE)
- Detailed results for each phase
- Next steps for fixing issues

### Step 4: Proceed or Fix Issues

**If no blockers (exit code 0):**
```bash
python experiments/run_experiment.py --all-models --scenario S1
```

**If blockers found (exit code 1):**
1. Read the validation report
2. Identify the failing test/check
3. Fix the underlying issue in the model code
4. Re-run validation
5. Repeat until all blockers resolved

---

## Individual Validator Commands

### Contract Validation Only

Test interface compliance (fit/decompose/get_params):

```bash
python validation_pipeline/contract_validator.py
```

### Code Audit Only

Check for structural anti-patterns:

```bash
python validation_pipeline/code_audit.py
```

### Edge Case Benchmark Only

Test robustness to missing channels and weak signals:

```bash
python validation_pipeline/edge_case_benchmark.py
```

---

## Understanding Severity Levels

### BLOCKER

**What**: Model violates interface contract or crashes on basic operations

**Examples**:
- `fit()` does not set `_is_fitted=True`
- `decompose()` returns wrong type (list instead of DataFrame)
- `get_params()` has no return statement

**Action**: **DO NOT BENCHMARK** — must fix before proceeding

### HIGH

**What**: Model works but has significant structural issues

**Examples**:
- Missing columns in decomposition output
- No fitted check in decompose()
- Error handling missing for column access

**Action**: Fix before benchmarking if possible; if not, proceed with caution

### MEDIUM

**What**: Code quality or robustness issues that may affect edge cases

**Examples**:
- Hardcoded loop indices instead of channel names
- Model fragile to missing channels (recovery Δ > 0.5pp)
- Incomplete coefficient serialization

**Action**: Review and consider fixing before publication

### LOW

**What**: Minor code quality suggestions

**Examples**:
- Orphaned config parameters
- Inconsistent naming conventions

**Action**: Nice to fix, but not blocking

---

## Common Scenarios

### Scenario 1: Everything Passes

```bash
$ python validation_pipeline/validate_before_benchmark.py
...
✓ SAFE TO PROCEED: All validation checks passed.

$ python experiments/run_experiment.py --all-models --all-scenarios
# Benchmarking proceeds
```

### Scenario 2: High-Severity Issues but No Blockers

```bash
$ python validation_pipeline/validate_before_benchmark.py
...
⚠️ PROCEED WITH CAUTION: 3 high-severity issues found.

$ python experiments/run_experiment.py --all-models --scenario S1
# Benchmarking proceeds, but note issues in commit message
```

### Scenario 3: Blocking Issues Found

```bash
$ python validation_pipeline/validate_before_benchmark.py
...
❌ DO NOT BENCHMARK: 2 blocking issues must be fixed.

$ cat validation_pipeline/VALIDATION_REPORT_*.md
# Read report, identify issues, fix code

$ python validation_pipeline/validate_before_benchmark.py
# Re-validate until all blockers resolved
```

---

## Integration with Benchmarking

The validation pipeline is automatically integrated into `experiments/run_experiment.py`:

```bash
python experiments/run_experiment.py --all-models --scenario S1
```

This command now:
1. Runs validation automatically (2-3 minutes)
2. If blockers found, aborts with error message
3. If no blockers, proceeds to benchmarking
4. Prints validation result summary at start

To **skip validation** (not recommended), edit `run_experiment.py` and comment out the validation import.

---

## Interpreting the Validation Report

### Part 1: Contract Validation

Shows which tests passed/failed for each model:

```
✓ geo_adstock   | fit() returns self          | ✓ PASS
✓ geo_adstock   | fit() sets _is_fitted=True  | ✓ PASS
✓ geo_adstock   | decompose() returns DataFrame | ✓ PASS
✗ weibull_adstock | get_params() returns dict | ✗ FAIL | [HIGH]
```

If a test fails, review the model file and the base class interface in `ltc/models/base.py`.

### Part 2: Code Audit

Shows structural issues found via static analysis:

```
ltc/models/framework1/weibull_regression.py:145 | [MEDIUM] Hardcoded index [0]
ltc/models/framework2/ardl_model.py:89          | [HIGH]   Missing fitted check
```

Click the file name to jump to the issue in your editor.

### Part 3: Edge Case Benchmark

Shows model fragility to edge cases:

```
geo_adstock   | S1_missing_search | Recovery: 82.9% | MAPE: 17.1% | Δ: -0.3pp | PASS
ardl          | S1_missing_tv     | Recovery: 45.2% | MAPE: 54.8% | Δ: +45.2pp | FLAG
```

Large Δ (> 0.5pp) indicates the model is sensitive to missing channels or weak signals.

---

## Troubleshooting

### Validation Won't Run: "ModuleNotFoundError: No module named 'ltc'"

**Problem**: Package not installed in development mode

**Solution**:
```bash
cd /c/github/ltc
pip install -e .
```

### Validation Slow: Takes >3 minutes

**Problem**: Edge case benchmark is slow (40 model runs)

**Solution**: Skip edge cases to test only contract validation:
```bash
python validation_pipeline/contract_validator.py
python validation_pipeline/code_audit.py
```

### Model Fails Contract Validation

**Problem**: Model violates interface, or fit() crashes

**Solution**:
1. Review error message in validation report
2. Open the model file listed in the report
3. Check `fit()`, `decompose()`, `get_params()` methods
4. Compare to `ltc/models/base.py` for correct interface
5. Fix and re-validate

---

## Best Practices

### Before Benchmarking

Always run validation first:
```bash
python validation_pipeline/validate_before_benchmark.py
cat validation_pipeline/VALIDATION_REPORT_*.md  # review if issues
python experiments/run_experiment.py --all-models --all-scenarios
```

### After Modifying a Model

Run validation to catch regressions:
```bash
# Edit model code...
python validation_pipeline/validate_before_benchmark.py
# If all pass, safe to benchmark
```

### Before Publishing Results

Review the validation report to document any known limitations:
```
# In paper/commit message:
"Validation report shows:
  - All models pass contract validation
  - Edge case fragility: ARDL +15pp on S1_missing_tv
  - Recommendation: Use BSTS or kalman_dlm for production
"
```

---

## Next Steps

- Read `README.md` in this directory for detailed architecture
- Run `python validation_pipeline/test_validation.py` to test all components
- Use `validate_before_benchmark.py` as a pre-flight check for every experiment

