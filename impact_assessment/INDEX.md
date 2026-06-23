# Impact Assessment Index

**Analysis Date:** 2026-06-23  
**Scope:** 10 structural issues across 10 models (F1, F2, F3 frameworks)  
**Status:** COMPLETE

---

## Files in This Directory

### 1. **IMPACT_REPORT.md** (START HERE)
   - Executive summary table
   - Detailed analysis of all 10 issues
   - Severity classification (CRITICAL, HIGH, MEDIUM, LOW)
   - Impact on S1-S5 empirical results
   - Paper claim risk assessment
   - Publication recommendation per issue

   **Key Finding:** Framework hierarchy (F3 > F1 > F2) unaffected. 5 issues must be fixed before publication (1-2 hours of work).

### 2. **detailed_findings.txt**
   - Line-by-line code analysis for each issue
   - Root cause explanation
   - Code snippets showing the bug
   - Impact analysis (all channels present vs missing)
   - Specific fix approach with code examples
   - Diagnostic steps to verify fix

   **Use Case:** For developers implementing fixes.

### 3. **comparison_before_after.csv**
   - Quantitative impact table
   - S1 recovery before/after fix (estimated)
   - Paper claims affected
   - Recommendation per issue
   - Framework-level summary

   **Use Case:** Quick reference for impact magnitude.

### 4. **RECOMMENDATION.md**
   - Publication decision for each issue
   - DO NOT FIX vs FIX IMMEDIATELY vs INVESTIGATE
   - Fix priority and timeline
   - Verification steps for each issue
   - Summary of what needs fixing before publication
   - Action plan with estimated time (1-2 hours)

   **Key Recommendation:** Fix Issues #4, #5, #8-10 (1-2 hours), then publish.

### 5. **CRITICAL_QUESTIONS_ANSWERED.md**
   - Answers to 10 critical questions about issue impact
   - Does the bug change published recovery %? (mostly NO)
   - Does it break reproducibility? (Issues #8-10 YES)
   - Does it change framework ranking? (NO)
   - Timeline and complexity to fix (1-2 hours, LOW complexity)
   - Risk assessment

   **Key Finding:** Only Issues #8-10 (missing exog_coefs) actually break reproducibility. All other issues are inert for S1-S5 data.

---

## Quick Reference: Issues Summary

### Issues That MUST Be Fixed Before Publication (1-2 hours)

| # | Model | Problem | Fix Time | Reason |
|---|-------|---------|----------|--------|
| 4 | almon_pdl | Unclear semantics (stc_degree vs ltc_degree) | 10 min | Semantic clarity required |
| 5 | ardl | S1 failure (0% recovery) | 30 min | Must understand root cause |
| 8 | bsts | Missing exog_coefs in get_params() | 5 min | Breaks reproducibility claim |
| 9 | kalman_dlm | Missing exog_coefs in get_params() | 5 min | Breaks reproducibility claim |
| 10 | mcmc_stock | Incomplete channel handling in get_params() | 10 min | Breaks reproducibility claim |

### Issues That Should NOT Be Fixed Before Publication (Technical Debt)

| # | Model | Problem | Why Skip | Fix Later |
|---|-------|---------|----------|-----------|
| 1 | dual_adstock | Coef index misalignment (missing channels) | S1-S5 complete | Refactor coefficient lookup |
| 2 | weibull_adstock | Coef index using enumerate | S1-S5 complete | Same refactor as #1 |
| 3 | geo_adstock | Coef index using enumerate | S1-S5 complete | Same refactor as #1-2 |

### Issues That Should Be Investigated (Optional Before Publication)

| # | Model | Problem | Why Optional | Timeline |
|---|-------|---------|--------------|----------|
| 6 | finite_dl | Weibull weight dimension mismatch | Recovery still 50% (mid-range) | 15 min if time permits |
| 7 | koyck | Index arithmetic fragility | No crash reported | 15 min (add assertions) |

---

## Impact on Paper Claims

### ✅ Unaffected Claims (No Fix Needed)

1. **"Framework 3 dominates F1 and F2"**
   - bsts 82.4% > geo_adstock 69.9% > finite_dl 50.3%
   - All empirical results are correct
   - No issues change this ranking

2. **"F1 weak on baseline scenario"**
   - geo_adstock 69.9%, almon_pdl 42.6% are correct
   - Coefficient indexing bugs inert for complete data

3. **"Channel attribution fails for F2 models"** (S2 analysis)
   - ARDL 68.8% aggregate but per-channel misattributed
   - No issues affect channel decomposition

4. **"MCMC converges reliably"**
   - R-hat < 1.05 for all 19 parameters in S1
   - Missing exog_coefs doesn't affect convergence

### ⚠️ Affected Claims (Requires Fix)

1. **"Results are reproducible via get_params()"**
   - Currently FALSE (Issues #8-10 omit exog_coefs)
   - After fix: TRUE
   - **Action:** Fix get_params() for bsts, kalman_dlm, mcmc_stock

2. **"ARDL failure is due to [architectural/config] limitation"**
   - Currently unclear (Issue #5 root cause unknown)
   - After investigation: Will clarify reason
   - **Action:** Investigate ARDL S1 config vs S2 config

---

## Framework Impact Summary

### F1 (Static Adstock): 4 Models

| Model | S1 Recovery | Issues | Impact |
|-------|------------|--------|--------|
| geo_adstock | 69.9% | #3 (coef index) | NONE (bug inert) |
| weibull_adstock | 10.5% | #2 (coef index) | NONE (bug inert) |
| almon_pdl | 42.6% | #4 (degree semantics) | MUST VERIFY |
| dual_adstock | 0.0% | #1 (coef index) | NONE (bug inert) |

**F1 Status:** Publication-ready after Issue #4 verification

### F2 (Dynamic Time-Series): 3 Models

| Model | S1 Recovery | Issues | Impact |
|-------|------------|--------|--------|
| koyck | 46.4% | #7 (index fragility) | LOW (add assertions) |
| ardl | 0.0% | #5 (S1 failure) | MUST INVESTIGATE |
| finite_dl | 50.3% | #6 (weibull dims) | OPTIONAL (investigate) |

**F2 Status:** Requires Issue #5 investigation (30 min). Issues #6-7 optional.

### F3 (State-Space): 3 Models

| Model | S1 Recovery | Issues | Impact |
|-------|------------|--------|--------|
| kalman_dlm | 82.0% | #9 (missing exog_coefs) | MUST FIX (5 min) |
| mcmc_stock | 72.6% | #10 (missing channels) | MUST FIX (10 min) |
| bsts | 82.4% | #8 (missing exog_coefs) | MUST FIX (5 min) |

**F3 Status:** Requires fix of get_params() (20 min total). Empirical results unaffected.

---

## Timeline to Publication-Ready

```
Start: 0 min
├─ Issue #5 investigation (ARDL S1): 30 min
├─ Issue #4 verification (AlmonPDL): 10 min
├─ Issues #8-10 fixes (get_params): 15 min
├─ Issue #7 assertions (optional): 15 min
├─ Issue #6 investigation (optional): 15 min
├─ Full S1-S5 regression test: 30 min
└─ End: 90 min (65 min blocking, 30 min optional)

PUBLICATION READY: After blocking issues (65 min minimum)
```

---

## Critical Path Summary

### Must Do (Blocking)
1. ✅ Investigate Issue #5 (ARDL S1) — 30 min
2. ✅ Verify Issue #4 (AlmonPDL) — 10 min
3. ✅ Fix Issues #8-10 (get_params) — 15 min
4. ✅ Regression test (S1-S5) — 30 min
**Total: 85 min**

### Nice to Have (Optional)
- Issue #6: Weibull weight dimension check — 15 min
- Issue #7: Koyck bounds assertions — 15 min

### Technical Debt (Post-Publication)
- Issues #1-3: Refactor coefficient indexing — 1-2 hours

---

## For the Paper

### Reproducibility Statement (After Fixes)
> "All fitted parameters are available via the `get_params()` method of each model, enabling full reproducibility of the baseline, STC, and LTC decompositions. See Supplementary Table X for parameter values from the S1 baseline scenario."

### ARDL Limitation Statement (After Issue #5 Investigation)
> "The ARDL model requires careful configuration of the AR lag order (ltc_degree parameter). On the S1 baseline scenario, standard configurations result in unstable AR polynomials [/alternatively: create high collinearity]. The model recovers on the S2 spend-pause scenario where the structural break provides identification. This limitation is documented in Appendix B."

### Technical Limitations Acknowledged
> "The following structural assumptions are recognized:
> - F1 models assume geometric or Weibull adstock can capture both STC and LTC
> - F2 models require careful lag specification to avoid collinearity
> - F3 models require sufficient data (261 weeks) to estimate latent brand stock dynamics
> All models are evaluated on synthetic ground-truth data and results may not generalize to empirical MMM applications with non-linear feedback loops or atypical spend patterns."

---

## Verification Checklist (Before Final Submission)

- [ ] Issue #4: AlmonPDL degree verified (ltc_degree used correctly)
- [ ] Issue #5: ARDL S1 root cause identified and documented
- [ ] Issue #8: bsts get_params() includes exog_coefs
- [ ] Issue #9: kalman_dlm get_params() includes exog_coefs
- [ ] Issue #10: mcmc_stock get_params() includes all 5 channels
- [ ] All S1-S5 benchmarks re-run after fixes
- [ ] No regression in empirical recovery percentages
- [ ] Supplementary Table X has complete get_params() output
- [ ] ARDL limitation documented in paper/appendix
- [ ] Reproducibility statement reviewed and accurate

---

## Next Steps

1. **Read IMPACT_REPORT.md** (5 min) for executive summary
2. **Read RECOMMENDATION.md** (10 min) for publication decision
3. **Read detailed_findings.txt** (20 min) for fix instructions
4. **Implement fixes** (1-2 hours)
5. **Run regression test** (30 min)
6. **Update paper** with ARDL limitation and reproducibility statement
7. **Submit** with confidence in empirical results and reproducibility

---

## Contact & Questions

**Analysis performed:** Static code inspection + correlation with published baseline  
**Methodology:** Line-by-line code review, root cause analysis, impact simulation  
**Confidence level:** HIGH (90%+) for Issues #1-3, #8-10; MEDIUM (70%) for Issues #4-7  

**If questions arise:**
- See detailed_findings.txt for code-level explanations
- See CRITICAL_QUESTIONS_ANSWERED.md for impact scenarios
- Check RECOMMENDATION.md for priority and timeline

