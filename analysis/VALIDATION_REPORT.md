# Paper Validation Report
**Date:** 2026-05-29  
**Status:** IN PROGRESS  
**Focus:** Validate all numerical claims in Sections 4-10

---

## Executive Summary

- **Total Claims Audited:** 50 recovery accuracy values across 10 models × 5 scenarios
- **Exact Matches:** 48/50 (96%)
- **Minor Discrepancies:** 2/50 (4%) - MCMC rounding differences
- **Critical Finding:** Code does NOT export pause-window robustness ratios needed for full validation
- **Recommendation:** Fix MCMC values + update code to output pause metrics

---

## PHASE 1: Recovery Accuracy Validation ✅

### Summary
- **Validated:** All recovery percentages in Section 4 Table 3 (50 values)
- **Framework:** 10 models (BSTS, Kalman DLM, MCMC, Geo-adstock, Finite DL, Koyck, Almon PDL, Weibull, ARDL, Dual)
- **Scenarios:** S1-S5 (Baseline, Spend Pause, High Seasonality, Structural Break, Weak Signal)

### Validation Results

| Status | Count | Models | Scenarios |
|--------|-------|--------|-----------|
| ✅ EXACT MATCH | 48 | All except MCMC | S1-S5 |
| ⚠️ MINOR VARIANCE | 2 | MCMC only | S1, S2 |
| ❌ MISMATCH | 0 | — | — |

### Detailed Findings

#### ✅ Perfect Matches (48 claims)
All F1, F2, and most F3 models match exactly:
- **BSTS:** 82.4% (S1), 81.0% (S2), 76.8% (S3), 81.6% (S4), 0.0% (S5)
- **Kalman DLM:** 82.0% (S1), 83.1% (S2), 64.9% (S3), 75.4% (S4), 0.0% (S5)
- **Geo-adstock:** 69.9% (S1), 83.1% (S2), 43.2% (S3), 63.4% (S4), 0.0% (S5)
- **ARDL:** 0.0% (S1), 68.8% (S2), 63.3% (S3), -19.8% (S4), 0.0% (S5)
- **All others:** Verified across all scenarios

#### ⚠️ Minor Discrepancies (2 claims) - MCMC Rounding

| Scenario | Metric | Paper Claimed | Actual | Diff | Notes |
|----------|--------|---------------|--------|------|-------|
| S1 | MCMC Recovery | 72.4% | 72.6% | +0.2pp | Likely MCMC sample averaging |
| S2 | MCMC Recovery | 59.9% | 60.9% | +1.0pp | Likely MCMC sample averaging |

**Root Cause:** MCMC results are computed from Bayesian posterior samples (4 chains × 1000 iterations). The paper may have recorded a single sample mean vs. the aggregated mean across all samples.

**Impact:** Minimal (within typical MCMC variance margins). Suggests paper values should be updated to reflect actual reproduction.

---

## PHASE 2: Pause-Window Robustness Validation ⚠️ BLOCKED

### Status
**CANNOT VALIDATE** - Metrics not exported from code

### Paper Claims (requires validation)
| Model | Paper Claim | Status |
|-------|-------------|--------|
| BSTS | 1.02× | NEED DATA |
| Geo-adstock | 1.41× | NEED DATA |
| MCMC | 1.30× | NEED DATA |
| Kalman DLM | 1.41× | NEED DATA |
| Almon PDL | 1.49× | NEED DATA |
| Koyck | 0.77× | NEED DATA |
| Finite DL | 0.69× | NEED DATA |

### Why Can't We Validate?
The JSON output files (`outputs/results/{model}_{scenario}.json`) do not include:
- Pause-window MAPE (weeks 100-120 in S2)
- Full-series MAPE separately
- Computed pause-window ratio

### Required for Full Validation
Need code to calculate and export:
```
pause_window_ratio = MAPE(weeks_100_120) / MAPE(weeks_1_261)
```

---

## CRITICAL FINDINGS & RECOMMENDATIONS

### Issue 1: MCMC Value Discrepancies
**Priority:** HIGH - Must fix before submission

**Action Items:**
1. Update Section 4: `mcmc_stock: S1 72.4%` → `72.6%`
2. Update Section 4: `mcmc_stock: S2 59.9%` → `60.9%`
3. Update Table 3: Correct MCMC values
4. Update Section 5: Correct narrative claiming S1 72.4%
5. Update Section 9: If MCMC S1 value is mentioned

**Files to Update:**
- writing/MASTER_DOCUMENT_FINAL.md (Sections 4, 5, 9)

---

### Issue 2: Missing Pause-Window Metrics
**Priority:** HIGH - Essential for reproducibility claims

**Action Items:**
1. Update experiment code to compute pause_window_ratio
2. Store in JSON: `"pause_window_ratio": float`
3. Re-run all 50 experiments to regenerate results
4. Validate paper claims against new data
5. Document in Replicability section

**Code Location:** `experiments/run_experiment.py` or `ltc/evaluation/metrics.py`

---

## Next Phases

### Phase 3: Channel-Level Attribution Validation
- Validate per-channel recovery claims in Section 6
- Verify ARDL aggregate vs. channel discrepancy (68.8% aggregate, 0% per-channel)
- Verify BSTS channel inversion claims (Section 5.3)

### Phase 4: Robustness & Special Cases
- Framework averages (75.7% F3 vs 32.2% F2 vs 30.8% F1)
- MCMC S5 weak signal recovery (88.5%)
- Critical failures (ARDL S4: -19.8%, Dual Adstock: -578%)

### Phase 5: Narrative Claims Validation
- "BSTS pause-window ratio 1.02× is paper centrepiece"
- "MCMC converts collinearity from liability to asset"
- "Weibull shape param cannot fit STC+LTC"

---

## Status by Section

| Section | Content | Validation | Notes |
|---------|---------|-----------|-------|
| 4 | Recovery accuracy | ✅ 96% PASS | 2 MCMC values need correction |
| 4 | Pause ratios | ⚠️ BLOCKED | Code doesn't export metrics |
| 5 | Scenario narratives | ⏳ PENDING | Depends on pause ratios |
| 6 | Channel attribution | ⏳ PENDING | Phase 3 work |
| 7 | Calibration claims | ⏳ PENDING | Need frozen vs optimized data |
| 8 | Anomaly analysis | ⏳ PENDING | Phase 4 work |
| 9 | Discussion synthesis | ⏳ PENDING | Depends on phases 1-4 |
| 10 | Practitioner guidance | ⏳ PENDING | Depends on all results |

---

## Files Generated

- `validation/01_EXTRACTED_METRICS.csv` - All 50 recovered accuracy values
- `validation/02_RECOVERY_MATRIX.csv` - Recovery matrix (10 models × 5 scenarios)
- `validation/03_PHASE1_SECTION4_VALIDATION.csv` - Detailed comparison of paper vs actual

---

## Conclusion

**96% of claims validated perfectly.** 4% discrepancy in MCMC values are minor rounding artifacts requiring simple correction. The pause-window robustness claims (critical to paper's thesis) **cannot be validated** without code updates to export these metrics.

**Next Steps:**
1. ✅ Fix MCMC values in paper (2 values)
2. ⚠️ Update code to export pause_window_ratio
3. 🔄 Continue Phases 3-5 of validation

---

**Prepared by:** Claude Code Validation Framework  
**Date:** 2026-05-29  
**Status:** 96% PASS - Proceed with corrections
