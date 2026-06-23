# Issue Resolution Plan — 3 Sub-Agent Parallel Execution

**Status:** Ready for execution  
**Target:** Resolve all 10 structural issues before final paper submission  
**Execution Model:** Sub-Agents 1 & 2 fix in parallel; Sub-Agent 3 validates sequentially

---

## EXECUTION TIMELINE

### Phase 1: Sub-Agent 1 Fixes Issues #1-5 (Framework 1 & 2)
**Est. Duration:** ~90 minutes  
**Sequential execution** (one issue per iteration; Sub-Agent 3 validates between each)

```
[Sub-Agent 1]                          [Sub-Agent 3]
Fix Issue #1 (DualAdstockOLS) ────→ Validate Issue #1
Fix Issue #2 (WeibullAdstockNLS) ──→ Validate Issue #2
Fix Issue #3 (GeometricAdstockOLS) → Validate Issue #3
Fix Issue #4 (AlmonPDL) ──────────→ Validate Issue #4
Fix Issue #5 (ARDLModel) ──────────→ Validate Issue #5
```

### Phase 2: Sub-Agent 2 Fixes Issues #6-10 (Framework 2 & 3)
**Est. Duration:** ~60 minutes  
**Sequential execution** (one issue per iteration; Sub-Agent 3 validates between each)

```
[Sub-Agent 2]                       [Sub-Agent 3]
Fix Issue #6 (FiniteDLModel) ────→ Validate Issue #6
Fix Issue #7 (KoyckModel) ────────→ Validate Issue #7
Fix Issue #8 (BayesianStructuralTS) → Validate Issue #8
Fix Issue #9 (KalmanDLM) ──────────→ Validate Issue #9
Fix Issue #10 (MCMCLatentStock) ───→ Validate Issue #10
```

### Phase 3: Final Regression Testing
**Est. Duration:** ~30 minutes  
**After all 10 issues fixed & validated:**
```
[Sub-Agent 1/2] Run S1-S5 regression test
                Report recovery % vs baseline for all 10 models
[Sub-Agent 3]   Verify all metrics within ±2% of published baseline
[Main Agent]    Confirm all issues resolved; publish completion report
```

---

## ISSUE ASSIGNMENTS & SEQUENCE

### SUB-AGENT 1: Issues #1-5 (F1 & F2)

| Issue | Model | File | Type | Time | Priority |
|-------|-------|------|------|------|----------|
| #1 | DualAdstockOLS | `ltc/models/framework1/dual_adstock.py` | Coefficient index fix | 15 min | LOW |
| #2 | WeibullAdstockNLS | `ltc/models/framework1/weibull_regression.py` | Coefficient index fix | 15 min | LOW |
| #3 | GeometricAdstockOLS | `ltc/models/framework1/geometric_regression.py` | Coefficient index fix | 15 min | MEDIUM |
| #4 | AlmonPDL | `ltc/models/framework1/almon_regression.py` | Parameter clarity | 10 min | MEDIUM |
| #5 | ARDLModel | `ltc/models/framework2/ardl_model.py` | S1 failure investigation | 30 min | CRITICAL |

**Key Dependencies:** None (fixes are independent)  
**Validation:** Sub-Agent 3 validates each issue before proceeding to next

---

### SUB-AGENT 2: Issues #6-10 (F2 & F3)

| Issue | Model | File | Type | Time | Priority |
|-------|-------|------|------|------|----------|
| #6 | FiniteDLModel | `ltc/models/framework2/finite_dl_model.py` | Weight dimension investigation | 15 min | MEDIUM |
| #7 | KoyckModel | `ltc/models/framework2/koyck_model.py` | Index arithmetic bounds checking | 15 min | MEDIUM |
| #8 | BayesianStructuralTS | `ltc/models/framework3/bayesian_sts.py` | Add exog_coefs to get_params() | 5 min | CRITICAL |
| #9 | KalmanDLM | `ltc/models/framework3/kalman_dlm.py` | Add exog_coefs to get_params() | 5 min | CRITICAL |
| #10 | MCMCLatentStock | `ltc/models/framework3/mcmc_latent_stock.py` | Complete channel handling in get_params() | 10 min | CRITICAL |

**Key Dependencies:** None (fixes are independent)  
**Validation:** Sub-Agent 3 validates each issue before proceeding to next

---

## SUB-AGENT 3: Validation Protocol

**Triggered after:** Each issue fix (Issues #1-10)  
**Task:** Run validation_pipeline and regression tests  
**Output:** VALIDATION_REPORT_{timestamp}.md

### Validation Steps (per issue)

```python
# 1. Contract Compliance (100 tests)
python validation_pipeline/contract_validator.py
# Check: fit(), decompose(), get_params() compliance

# 2. Code Audit (50 checks)
python validation_pipeline/code_audit.py
# Check: no new hardcoded indices, config keys consistent

# 3. Edge Case Tests (40 scenarios)
python validation_pipeline/edge_case_benchmark.py
# Check: behavior with missing channels, weak signals

# 4. Model-Specific Regression
python -c "
from ltc.models.framework1.dual_adstock import GeometricAdstockOLS
from ltc.data.loader import load_scenario
df = load_scenario('path/to/data', 'S1')
model = DualAdstockOLS()
model.fit(df, config)
decomp = model.decompose(df)
recovery = compute_recovery(decomp['ltc_total'], df['ltc_*_true'])
assert abs(recovery - 69.9) <= 2.0, f'Recovery {recovery}% out of range'
"
```

### Success Criteria (per issue)

✓ **Contract validation** passes (exit code 0)  
✓ **Edge case tests** pass (no index errors, silent failures)  
✓ **S1 recovery** within ±2% of baseline  
✓ **No new regressions** in other models  

### Failure Recovery

If validation **fails** for an issue:
1. Sub-Agent 3 provides specific error message
2. Return to Sub-Agent 1 or 2 with diagnostic
3. Sub-Agent 1/2 fixes issue
4. Sub-Agent 3 re-validates

---

## EXPECTED OUTCOMES

### Issue #1-3 (Coefficient Index Fixes)
- **Before:** Bug exists but inert for S1-S5
- **After:** Explicit coefficient mapping; robust to missing channels
- **Recovery impact:** None (±0%)
- **Code quality:** Improved robustness

### Issue #4 (AlmonPDL Clarity)
- **Before:** Semantic ambiguity (ltc_degree vs stc_degree)
- **After:** Clear documentation of which parameter is used
- **Recovery impact:** None (±0%)
- **Reproducibility:** Improved clarity

### Issue #5 (ARDL S1 Investigation)
- **Before:** 0.0% recovery on S1; root cause unknown
- **After:** Root cause identified (config issue or architectural); either fixed or clearly documented
- **Recovery impact:** Potential 50%+ recovery if fixable
- **Paper impact:** If fixable, removes ARDL from "critical failure" category

### Issue #6-7 (FiniteDL & Koyck Bounds Checking)
- **Before:** Potential edge-case fragility; works for S1-S5 by luck
- **After:** Explicit bounds checking; robust to varying configurations
- **Recovery impact:** None (±0% for S1-S5)
- **Code quality:** Improved defensive programming

### Issue #8-10 (F3 get_params() Completeness)
- **Before:** Missing exog_coefs; reproducibility claim broken
- **After:** Complete parameter serialization; reproducibility claim valid
- **Recovery impact:** None (±0%)
- **Reproducibility:** Fixed; claim now valid

---

## BLOCKERS & ESCALATION

### Known Potential Blockers

| Issue | Potential Blocker | Mitigation |
|-------|-------------------|-----------|
| #5 | ARDL architecture may be unfixable | Document root cause if unfixable |
| #6 | Weight dimension may be correct | Diagnostic will confirm |
| #7 | Index arithmetic may require major refactor | Add bounds checking first |
| #8-10 | PyMC/InferenceData serialization | Use standard Python types |

### If Blocker Encountered

1. Sub-Agent documents specific blocker
2. Escalate to main agent with diagnostic output
3. Main agent decides: fix, document, or defer to post-publication

---

## SUCCESS CRITERIA (OVERALL)

**All 10 issues resolved when:**

✓ Issue #1-3: Coefficient indexing is explicit and robust  
✓ Issue #4: Parameter semantics clearly documented  
✓ Issue #5: ARDL S1 root cause identified (fixed or documented)  
✓ Issue #6-7: Bounds checking in place; no silent failures  
✓ Issue #8-10: get_params() returns complete serializable dict  

**Validation confirms:**

✓ All 10 models pass contract compliance tests  
✓ S1 recovery for all models within ±2% of baseline  
✓ S2-S5 regression tests pass (spot check 2-3 models)  
✓ No new edge case failures introduced  

**Paper ready when:**

✓ All 10 issues closed & validated  
✓ Reproducibility claim now valid  
✓ ARDL S1 behavior explained  
✓ Final proofread complete  

---

## FILES TO REFERENCE

- **AGENT_FIX_GUIDE.md** — Detailed fix instructions for each issue
- **IMPACT_REPORT.md** — Full technical analysis of all 10 issues
- **detailed_findings.txt** — Code-level analysis with line numbers
- **validation_pipeline/README.md** — Validation system overview
- **validation_pipeline/USAGE_GUIDE.md** — How to run validators

---

## COMMUNICATION PROTOCOL

**Sub-Agent 1 reports after each issue:**
> "Issue #N fixed. Changes: [file list]. Validation results: [PASS/FAIL]. Recovery: [X.X%]. Next: Issue #(N+1)"

**Sub-Agent 2 reports after each issue:**
> "Issue #N fixed. Changes: [file list]. Validation results: [PASS/FAIL]. Recovery: [X.X%]. Next: Issue #(N+1)"

**Sub-Agent 3 reports after each validation:**
> "Issue #N validation: [PASS/FAIL]. Contract tests: [N/M pass]. Edge cases: [X]. Recovery delta: [±Y%]. Status: Ready for next issue"

**Main agent final report (after all 10 issues):**
> "All 10 issues resolved. Paper is publication-ready. Reproducibility restored. Detailed report: [link]"

