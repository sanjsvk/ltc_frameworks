# Validation Pipeline — Quick Reference Card

## One-Liner Commands

```bash
# Before benchmarking (ALWAYS RUN THIS FIRST)
python validation_pipeline/validate_before_benchmark.py

# Individual validators
python validation_pipeline/contract_validator.py     # Interface tests
python validation_pipeline/code_audit.py             # Code quality checks
python validation_pipeline/edge_case_benchmark.py    # Robustness tests

# Run after implementing changes
python validation_pipeline/test_validation.py        # Quick system test
```

---

## Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | All checks passed OR warnings only | ✓ Safe to benchmark |
| 1 | Blocking issues found | ✗ STOP — Fix before benchmarking |

---

## Severity Levels

| Level | Meaning | Example | Action |
|-------|---------|---------|--------|
| **BLOCKER** | Critical interface violation | fit() crashes | **MUST FIX** |
| **HIGH** | Significant structural issue | Missing columns | Fix if possible |
| **MEDIUM** | Code quality / robustness issue | Hardcoded indices | Consider fixing |
| **LOW** | Minor suggestion | Naming convention | Nice to fix |

---

## What Each Validator Tests

### Part 1: Contract Validator (10 tests × 10 models = 100)
```
✓ fit() returns self
✓ fit() sets _is_fitted=True
✓ decompose() returns DataFrame
✓ decompose() has required columns
✓ decompose() raises before fit()
✓ get_params() returns dict
✓ get_params() JSON-serializable
✓ Channel count reasonable
✓ No NaN/Inf in output
✓ Model repr shows fitted status
```

### Part 2: Code Audit (5 checks × 10 models = 50)
```
✓ No hardcoded loop indices
✓ No orphaned config keys
✓ Coefficient serialization complete
✓ Error handling for missing columns
✓ Fitted check in decompose()
```

### Part 3: Edge Case Benchmark (4 scenarios × 10 models = 40)
```
S1_clean       → Recovery baseline
S1_missing_search → -Search channel
S1_missing_tv  → -TV channel
S1_weak_exog   → Weak exogenous signals
```

---

## Interpreting Output

### Example 1: All Pass
```
✓ SAFE TO PROCEED: All validation checks passed.
```
→ Run benchmarking: `python experiments/run_experiment.py --all-models --all-scenarios`

### Example 2: High-Severity Warnings (No Blockers)
```
⚠️ PROCEED WITH CAUTION: 3 high-severity issues found.
Review: validation_pipeline/VALIDATION_REPORT_2026-06-23_14-30-45.md
```
→ Safe to benchmark, but review issues before publication

### Example 3: Blocking Issues (STOP)
```
❌ DO NOT BENCHMARK: 2 blocking issues must be fixed.
Review: validation_pipeline/VALIDATION_REPORT_2026-06-23_14-30-45.md
```
→ Fix issues, re-validate, then benchmark

---

## Report Sections (VALIDATION_REPORT_*.md)

1. **Summary** — Quick counts (blockers, high severity, passed tests)
2. **Recommendation** — DO NOT / PROCEED WITH CAUTION / SAFE
3. **Phase 1 Results** — Contract validation details
4. **Phase 2 Results** — Code audit issues by severity
5. **Phase 3 Results** — Edge case fragility by model
6. **Next Steps** — Remediation guidance

---

## Performance

| Component | Time | Notes |
|-----------|------|-------|
| Contract Validator | 10-15 sec | Loads 1 scenario, fits 10 models |
| Code Audit | 1-2 sec | Static analysis only |
| Edge Case Benchmark | 60-90 sec | 4 scenarios × 10 models |
| **TOTAL** | **~2-3 min** | Acceptable pre-flight check |

---

## Common Issues & Quick Fixes

### "ModuleNotFoundError: No module named 'ltc'"
```bash
cd /c/github/ltc
pip install -e .
```

### Validation Slow (>3 minutes)
Skip edge cases for quick check:
```bash
python validation_pipeline/contract_validator.py  # 15 sec
python validation_pipeline/code_audit.py          # 2 sec
```

### Model Fails Contract Test
1. Open validation report
2. Find failing test (e.g., "get_params() returns dict")
3. Check model file
4. Compare to `ltc/models/base.py`
5. Fix and re-validate

---

## Integration Points

### Automatic Hook in Benchmarking
```python
# experiments/run_experiment.py now includes:
from validation_pipeline.validate_before_benchmark import run_validation

result = run_validation()
if result['blockers'] > 0:
    sys.exit(1)  # Abort benchmarking
```

### Pre-Benchmark Workflow
```
1. python validation_pipeline/validate_before_benchmark.py
   ↓ (if exit code 1, STOP and fix)
2. cat validation_pipeline/VALIDATION_REPORT_*.md  (review if warnings)
   ↓ (if exit code 0)
3. python experiments/run_experiment.py --all-models --all-scenarios
```

---

## Fragility Score Interpretation

```
Score 0.0-0.25  → 🟢 ROBUST   (model handles edge cases well)
Score 0.25-0.50 → 🟡 MODERATE (some sensitivity, monitor)
Score > 0.50    → 🔴 FRAGILE  (high sensitivity, review results)
```

Example:
```
geo_adstock   Fragility: 0.17 🟢
kalman_dlm    Fragility: 0.33 🟡
dual_adstock  Fragility: 1.50 🔴
```

---

## For Future Agents (MANDATORY)

Per `next_steps.txt`, every experiment run MUST include:

```python
# Step 1: Run validation
python validation_pipeline/validate_before_benchmark.py

# Step 2: Review report if issues found
cat validation_pipeline/VALIDATION_REPORT_*.md

# Step 3: If no blockers, proceed
python experiments/run_experiment.py --all-models --all-scenarios

# Step 4: Document validation in commit
# [validation: PASSED] or [validation: 3x HIGH issues, see report]
```

**This is a MANDATORY pre-flight check. Do not skip it.**

---

## Documentation Files

| File | Purpose | When to Read |
|------|---------|--------------|
| **QUICK_REFERENCE.md** | This file — commands and quick answers | Getting started (you are here) |
| **USAGE_GUIDE.md** | Step-by-step guide, common scenarios | First time using validation |
| **README.md** | Detailed architecture, troubleshooting | Deep dive into system |
| **SYSTEM_OVERVIEW.md** | Technical design, performance specs | Understanding how it works |

---

## Support

**Quick Question?** → Check this QUICK_REFERENCE.md
**How-to?** → Read USAGE_GUIDE.md
**How does it work?** → Read README.md
**Technical details?** → Read SYSTEM_OVERVIEW.md
**Something broken?** → Check README.md troubleshooting section

---

## Key Takeaways

✓ Always run validation before benchmarking
✓ Exit code 1 means STOP (fix blocking issues)
✓ Exit code 0 means safe to proceed
✓ Validation takes ~2-3 minutes
✓ Reports are in markdown (human-readable)
✓ This is a MANDATORY pre-flight check

**TL;DR**: `python validation_pipeline/validate_before_benchmark.py` then check exit code.
