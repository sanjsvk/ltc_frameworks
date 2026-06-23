# Issue Resolution Guide for Sub-Agents

**Date:** 2026-06-23  
**Total Issues:** 10 (distributed across 2 sub-agents)  
**Validation:** Each fix validated by validation_pipeline before proceeding to next issue  
**Overall Goal:** Restore reproducibility and resolve blocking issues before publication

---

## ASSIGNMENT MATRIX

### Sub-Agent 1: Framework 1 & Framework 2 Issues (Issues #1-5)
**Responsible for:** GeometricAdstockOLS, WeibullAdstockNLS, DualAdstockOLS, AlmonPDL, ARDLModel  
**Issues to fix:** #1, #2, #3, #4, #5 (in sequence)  
**Total time estimate:** ~90 minutes  
**Dependencies:** None (issues are independent)

### Sub-Agent 2: Framework 3 Issues (Issues #6-10)
**Responsible for:** BayesianStructuralTS, KalmanDLM, MCMCLatentStock, + evaluation code  
**Issues to fix:** #6, #7, #8, #9, #10 (in sequence)  
**Total time estimate:** ~60 minutes  
**Dependencies:** None (issues are independent)

### Sub-Agent 3: Validation & Regression Testing
**Triggered after:** Each issue is fixed (issues #1-10)  
**Task:** Run validation_pipeline and edge case tests for the fixed issue  
**Success criteria:** exit code 0 on validation, recovery_accuracy within ±2% of baseline

---

## ISSUE #1: DualAdstockOLS Coefficient Index Misalignment

**File:** `ltc/models/framework1/dual_adstock.py`  
**Lines:** 144-158 (decompose method)  
**Priority:** LOW (inert for S1-S5; technical debt only)  
**Time:** 15 minutes  
**Assigned to:** Sub-Agent 1

### Problem Statement
In `decompose()`, when a channel is missing from the data, the code increments `coef_idx` without adding corresponding coefficients to the regression matrix. This causes coefficient indices to misalign when reconstructing contributions.

### Root Cause
```
Current behavior:
  Line 150: coef_idx += 2  # Reserve space for missing channel
  Line 151: continue       # But don't add to X_parts
  
Result: self._coefs array has fewer values than coef_idx expects
```

### Step-by-Step Fix

1. **Read the current decompose() method** (lines 144-158)
   - Understand the coef_idx iteration logic
   - Identify where missing channel handling occurs

2. **Implement coefficient mapping**
   ```python
   # In decompose(), replace simple coef_idx with explicit mapping:
   coef_map = {}  # Maps channel → actual coefficient index
   coef_idx = 0
   for ch in self._channels:
       col_stc = f"adstock_stc_{ch}"
       col_ltc = f"adstock_ltc_{ch}"
       if col_stc in df.columns and col_ltc in df.columns:
           coef_map[ch] = coef_idx
           coef_idx += 2
       else:
           coef_map[ch] = None  # Channel missing
   ```

3. **Update coefficient lookup** (lines 156+)
   ```python
   # OLD: coef = self._coefs[coef_idx]
   # NEW:
   if coef_map[ch] is not None:
       idx = coef_map[ch]
       stc_dict[ch] = pd.Series(self._coefs[idx] * stc_ad, index=index)
       ltc_dict[ch] = pd.Series(self._coefs[idx+1] * ltc_ad, index=index)
   else:
       stc_dict[ch] = pd.Series(0.0, index=index)
       ltc_dict[ch] = pd.Series(0.0, index=index)
   ```

4. **Add defensive assertion** in fit() method
   ```python
   # After fitting, verify all channels present:
   for ch in self._channels:
       assert f"adstock_stc_{ch}" in feature_names, f"Channel {ch} missing in fit"
   ```

5. **Test locally** with missing channel scenario
   ```python
   # Create test data with one channel missing (e.g., remove "search")
   # Verify decompose() returns correct indices and no index out of bounds
   ```

### Success Criteria
- [ ] Code compiles without errors
- [ ] `decompose()` uses explicit `coef_map` instead of simple `coef_idx`
- [ ] All 5 channels in S1 return correct coefficients
- [ ] Edge case test (missing channel) returns 0.0 instead of wrong channel's coefficient
- [ ] Validation pipeline returns exit code 0
- [ ] Recovery accuracy on S1 remains within ±2% of baseline (69.9%)

### Blockers
- None; issue is isolated to this model

---

## ISSUE #2: WeibullAdstockNLS Coefficient Index Misalignment

**File:** `ltc/models/framework1/weibull_regression.py`  
**Lines:** 148-167 (decompose method)  
**Priority:** LOW (inert for S1-S5; consolidate with Issue #3)  
**Time:** 15 minutes  
**Assigned to:** Sub-Agent 1

### Problem Statement
`decompose()` uses `enumerate(self._channels)` to index into `self._coefs`, but coefficients are stored in `feature_names` order which skips missing channels. If a channel is missing, indices misalign.

### Root Cause
```
Mismatch between enumeration order and feature storage order:
  enumerate([tv, search, social, display, video]) → 0,1,2,3,4
  feature_names[...] when search missing → [tv(0), social(1), display(2), video(3), ...]
  
Result: self._coefs[1] gets social coefficient instead of search
```

### Step-by-Step Fix

1. **Read decompose() method** (lines 148-167)
   - Understand how enumerate index is used
   - Identify self._feature_names storage

2. **Replace enumerate index with feature_names lookup**
   ```python
   # OLD (line 158):
   # coef = self._coefs[i]
   
   # NEW:
   feature_name = f"weibull_adstock_{ch}"
   if feature_name in self._feature_names:
       idx = self._feature_names.index(feature_name)
       coef = self._coefs[idx]
   else:
       coef = 0.0  # Channel not in features
   ```

3. **Add comment explaining the fix**
   ```python
   # Use feature_names index instead of enumerate to handle missing channels
   ```

4. **Test locally** with all 5 channels present
   ```python
   # Verify recovery on S1 matches baseline
   ```

### Success Criteria
- [ ] Code compiles without errors
- [ ] `decompose()` uses `feature_names.index()` instead of `enumerate()` index
- [ ] Recovery accuracy on S1 remains 10.5% ±2%
- [ ] Edge case test (missing channel) returns 0.0 instead of wrong channel's coef
- [ ] Validation pipeline returns exit code 0

### Blockers
- None; issue is isolated

---

## ISSUE #3: GeometricAdstockOLS Coefficient Index Alignment

**File:** `ltc/models/framework1/geometric_regression.py`  
**Lines:** 156-167 (decompose method)  
**Priority:** MEDIUM (same pattern as Issue #2; consolidate)  
**Time:** 15 minutes  
**Assigned to:** Sub-Agent 1

### Problem Statement
Same as Issue #2: uses `enumerate(channels)` instead of `feature_names` index for coefficient lookup.

### Root Cause
Duplicate pattern from Issue #2.

### Step-by-Step Fix

1. **Apply identical fix to Issue #2**
   ```python
   # Replace enumerate index with feature_names lookup
   feature_name = f"adstock_{ch}"
   if feature_name in self._feature_names:
       idx = self._feature_names.index(feature_name)
       coef = self._coefs[idx]
   else:
       coef = 0.0
   ```

2. **Verify fix matches Issue #2 pattern**
   - Same approach, different feature_name prefix

3. **Test on S1**

### Success Criteria
- [ ] Same pattern as Issue #2 applied correctly
- [ ] Recovery on S1 remains 69.9% ±2%
- [ ] Validation pipeline returns exit code 0

### Blockers
- None

---

## ISSUE #4: AlmonPDL Degree Naming Confusion

**File:** `ltc/models/framework1/almon_regression.py`  
**Lines:** [TBD - read file to identify]  
**Priority:** MEDIUM (semantic clarity for reproducibility)  
**Time:** 10 minutes  
**Assigned to:** Sub-Agent 1

### Problem Statement
Configuration has both `ltc_degree` and `stc_degree` parameters. Unclear which one Almon PDL model uses, or if there's confusion between them.

### Root Cause
Parameter naming ambiguity in config and code.

### Step-by-Step Fix

1. **Read almon_regression.py completely**
   - Find where `ltc_degree` and `stc_degree` appear
   - Identify which one is actually used in `fit()`

2. **Verify correct parameter usage**
   ```python
   # In fit(), confirm which degree is used:
   # Option A: degree = config.get("ltc_degree")
   # Option B: degree = config.get("stc_degree")
   # Option C: Both are used for different lags
   ```

3. **Add clarifying comments**
   ```python
   # Almon PDL uses ltc_degree (NOT stc_degree) because:
   # [reason based on your investigation]
   ```

4. **Verify in experiments/configs/framework1.yaml**
   - Check that config provides the correct degree parameter
   - Update config comment if ambiguous

5. **Test on S1**

### Success Criteria
- [ ] Code clearly documents which degree parameter is used
- [ ] `fit()` uses consistent parameter throughout
- [ ] experiments/configs/framework1.yaml provides correct parameter
- [ ] Recovery on S1 remains 42.6% ±2%
- [ ] Validation pipeline returns exit code 0

### Blockers
- None; independent investigation required

---

## ISSUE #5: ARDLModel S1 Failure (Root Cause Investigation)

**File:** `ltc/models/framework2/ardl_model.py`  
**Lines:** [TBD - read file to identify fit() and decompose()]  
**Priority:** CRITICAL (blocks publication; understanding required)  
**Time:** 30 minutes (investigation + potential fix)  
**Assigned to:** Sub-Agent 1

### Problem Statement
ARDLModel achieves 68.8% recovery on S2 (spend pause scenario) but 0.0% on S1 (baseline). Root cause unclear: config issue, code bug, or architectural limitation?

### Known Facts
- S2 (spend pause) works well: 68.8% recovery
- S1 (baseline) fails: 0.0% recovery
- Paper claims this is F2 architectural limitation, but needs validation
- S1 failure appears to be prior misspecification (Bayesian-style issue)

### Diagnostic Approach

1. **Read ardl_model.py completely**
   - Understand AR polynomial construction
   - Identify how `ltc_degree` is used
   - Check stability constraints

2. **Run S1 diagnostic**
   ```python
   # Load S1 data
   # Run ARDL with current frozen config
   # Print:
   #   - Fitted AR coefficients
   #   - Auto-regressive polynomial roots (stability check)
   #   - Residual diagnostics (ACF/PACF)
   #   - Predicted vs actual LTC over time
   ```

3. **Test hypothesis: AR polynomial instability**
   ```python
   # Check if AR roots are outside unit circle
   # If unstable, tighten constraints in fit()
   # Re-run and compare recovery
   ```

4. **Test hypothesis: ltc_degree parameter**
   ```python
   # Try ltc_degree = 1, 2, 3 on S1
   # Report recovery for each
   # Document which value works best
   ```

5. **Test hypothesis: Prior misspecification (Bayesian issue)**
   - This is less likely given ARDL is OLS-based
   - But check if regularization is hiding the issue

### Possible Fixes (depending on investigation)

**If instability found:**
```python
# In fit(), add stability constraint:
from numpy.polynomial import Polynomial
roots = Polynomial(ar_poly).roots()
if any(abs(r) <= 1.0 for r in roots):
    # Tighten constraints or reduce degree
    ltc_degree = max(1, ltc_degree - 1)
```

**If ltc_degree wrong:**
```python
# In config, change ltc_degree value based on investigation
# Update experiments/configs/framework2.yaml
```

**If architectural limitation confirmed:**
```python
# Document clearly why S1 fails on ARDL:
# - Add comment explaining the root cause
# - Note that S2 works due to spend pause providing cleaner signal
# - Reference this in paper's Discussion section
```

### Success Criteria
- [ ] Root cause of S1 failure is identified and documented
- [ ] If fixable: S1 recovery improved to >50%
- [ ] If architectural: Clear documentation of why and when it occurs
- [ ] S2 recovery remains stable at ~68.8%
- [ ] Validation pipeline returns exit code 0
- [ ] Paper Discussion section updated if root cause changes interpretation

### Blockers
- Requires investigation; may not have a simple fix

---

## ISSUE #6: FiniteDLModel Weibull Weight Dimension Mismatch

**File:** `ltc/models/framework2/finite_dl_model.py`  
**Lines:** 122-126 (distributed lag weight computation)  
**Priority:** MEDIUM (investigate before publication)  
**Time:** 15 minutes (investigation + fix)  
**Assigned to:** Sub-Agent 2

### Problem Statement
Weibull weight dimension may be computed using `max_lag` but actual distributed lag uses `ltc_degree` effective lags, causing potential shape mismatch between weight array and lag accumulation.

### Root Cause
```
Weight dimensions computed as: len(w) = max_lag
But lag loop uses: ltc_degree or AR order
If ltc_degree != max_lag, broadcasting fails or accumulates wrong shapes
```

### Diagnostic Approach

1. **Read finite_dl_model.py completely**
   - Identify how Weibull weights are computed
   - Find where they're applied in the lag accumulation
   - Check if max_lag is consistent with ltc_degree

2. **Run S1 diagnostic**
   ```python
   # Check weight dimensions:
   # Print len(weights), max_lag, ltc_degree
   # Print weight array shape at each iteration
   # Verify no broadcasting warnings/errors
   ```

3. **Test with different ltc_degree values**
   ```python
   # Try ltc_degree = 1, 2, 3, 4 on S1
   # Report recovery for each
   # Document which produces 50.3%
   ```

### Possible Fixes

**If dimension mismatch found:**
```python
# In fit(), ensure consistency:
max_lag = self.ltc_degree  # Make explicit
weights = weibull(..., shape=..., scale=..., max_lag=max_lag)

# Verify all loops use same lag dimension:
for i in range(max_lag):  # NOT enumerate
    adstock[t] += weights[i] * X[t-i]
```

**If silent truncation:**
```python
# Add assertion:
assert len(weights) == self.ltc_degree, \
    f"Weight dimension {len(weights)} != ltc_degree {self.ltc_degree}"
```

### Success Criteria
- [ ] Code compiles without errors
- [ ] Weight dimensions match lag dimension (explicit assertion added)
- [ ] Recovery on S1 remains 50.3% ±2%
- [ ] No broadcasting warnings in verbose output
- [ ] Validation pipeline returns exit code 0
- [ ] Edge case test (different ltc_degree) works correctly

### Blockers
- None; investigation may reveal no issue exists

---

## ISSUE #7: KoyckModel Index Arithmetic Fragility

**File:** `ltc/models/framework2/koyck_model.py`  
**Lines:** 110, 155 (AR lag index arithmetic)  
**Priority:** MEDIUM (add bounds checking before publication)  
**Time:** 15 minutes  
**Assigned to:** Sub-Agent 2

### Problem Statement
The Koyck transformation involves recursive lag algebra with index arithmetic that could fail silently or crash if AR lag order is misspecified or data has gaps. Current implementation works for S1-S5 but is fragile to edge cases.

### Root Cause
```
Recursive lag construction depends on prior iteration values:
  for i in ar_lags:
      ... use indices that depend on i-1

If ar_lags is mis-ordered or has gaps, arithmetic breaks silently
```

### Diagnostic Approach

1. **Read koyck_model.py completely**
   - Identify the recursive AR lag construction (lines ~110, ~155)
   - Understand what ar_lags contains
   - Check if there are bounds or order assumptions

2. **Review AR lag validation**
   ```python
   # Check if ar_lags are validated:
   # - Are they sequential (0, 1, 2, ...)?
   # - Are they bounded by data length?
   # - Are there error checks for gaps?
   ```

3. **Run S1 diagnostic**
   ```python
   # Print ar_lags values
   # Print index arithmetic at each iteration
   # Verify no index out of bounds warnings
   ```

### Possible Fixes

**Add explicit bounds checking:**
```python
# In fit(), before using ar_lags:
for lag in ar_lags:
    assert 0 <= lag < len(data), f"Lag {lag} out of bounds [0, {len(data)})"

# Or replace with explicit range:
max_ar_lags = min(self.ar_order, len(data) - 1)
ar_lags = range(1, max_ar_lags + 1)  # Explicit sequence
```

**Add defensive comments:**
```python
# Koyck recursion requires sequential lags in increasing order
# If ar_lags has gaps, index arithmetic fails silently
```

### Success Criteria
- [ ] Code compiles without errors
- [ ] All index arithmetic has explicit bounds checks
- [ ] Comments explain why bounds checking is critical
- [ ] Recovery on S1 remains 46.4% ±2%
- [ ] Edge case test (different AR order) works without silent failures
- [ ] Validation pipeline returns exit code 0

### Blockers
- None; straightforward bounds checking

---

## ISSUE #8: BayesianStructuralTS Missing exog_coefs

**File:** `ltc/models/framework3/bayesian_sts.py`  
**Lines:** [TBD - locate get_params() method]  
**Priority:** CRITICAL (blocks reproducibility)  
**Time:** 5 minutes  
**Assigned to:** Sub-Agent 2

### Problem Statement
`get_params()` returns model parameters but omits exogenous coefficient dictionary. This breaks the claim that "results are reproducible via get_params()".

### Root Cause
Incomplete parameter serialization in `get_params()` method.

### Step-by-Step Fix

1. **Read get_params() method**
   - Identify what's currently being returned
   - Check fit() to see where exog_coefs are stored

2. **Add exogenous coefficients to return dict**
   ```python
   def get_params(self) -> dict:
       params = {
           # ... existing params ...
           "exog_coefs": dict(self._exog_coefs) if self._exog_coefs else {},
       }
       return params
   ```

3. **Verify structure**
   - `exog_coefs` should be a dict with exogenous variable names as keys
   - Values should be fitted coefficients (scalars or arrays)

4. **Test locally**
   ```python
   # Fit model on S1
   params = model.get_params()
   assert "exog_coefs" in params, "exog_coefs missing from get_params()"
   assert isinstance(params["exog_coefs"], dict), "exog_coefs should be dict"
   ```

5. **Update docstring** if needed

### Success Criteria
- [ ] Code compiles without errors
- [ ] `get_params()` includes "exog_coefs" key
- [ ] `exog_coefs` is a dict (not None, not empty)
- [ ] Recovery on S1 remains 82.4% ±2%
- [ ] Validation pipeline returns exit code 0
- [ ] `get_params()` output is JSON serializable

### Blockers
- None; straightforward addition

---

## ISSUE #9: KalmanDLM Missing exog_coefs

**File:** `ltc/models/framework3/kalman_dlm.py`  
**Lines:** [TBD - locate get_params() method]  
**Priority:** CRITICAL (blocks reproducibility)  
**Time:** 5 minutes  
**Assigned to:** Sub-Agent 2

### Problem Statement
Same as Issue #8: `get_params()` omits exogenous coefficient dictionary.

### Root Cause
Incomplete parameter serialization.

### Step-by-Step Fix

1. **Apply identical fix to Issue #8**
   ```python
   # Add to get_params():
   "exog_coefs": dict(self._exog_coefs) if self._exog_coefs else {},
   ```

2. **Verify Kalman-specific structure**
   - Check if exog_coefs are stored differently than BayesianStructuralTS
   - Ensure serialization works for Kalman matrix format

3. **Test on S1**

### Success Criteria
- [ ] Same pattern as Issue #8
- [ ] Recovery on S1 remains 82.0% ±2%
- [ ] Validation pipeline returns exit code 0
- [ ] `get_params()` is JSON serializable

### Blockers
- None

---

## ISSUE #10: MCMCLatentStock Missing Channel Handling

**File:** `ltc/models/framework3/mcmc_latent_stock.py`  
**Lines:** [TBD - locate get_params() method]  
**Priority:** CRITICAL (blocks reproducibility)  
**Time:** 10 minutes  
**Assigned to:** Sub-Agent 2

### Problem Statement
`get_params()` may not include all 5 channels or MCMC posterior diagnostics (R-hats, effective sample size). Incomplete parameter serialization.

### Root Cause
Incomplete posterior summary in `get_params()`.

### Step-by-Step Fix

1. **Read get_params() method**
   - Identify what's currently returned
   - Check what idata (InferenceData) stores

2. **Ensure all 5 channels included**
   ```python
   def get_params(self) -> dict:
       params = {
           # ... existing params ...
           "channel_posteriors": {},
           "convergence_diagnostics": {},
       }
       
       # For each channel, store posterior mean + credible interval
       for ch in self._channels:
           posterior_draw = self.idata.posterior[f"ltc_coef_{ch}"]
           params["channel_posteriors"][ch] = {
               "mean": float(posterior_draw.mean()),
               "std": float(posterior_draw.std()),
               "q2.5": float(posterior_draw.quantile(0.025)),
               "q97.5": float(posterior_draw.quantile(0.975)),
           }
       
       # Add convergence diagnostics
       params["convergence_diagnostics"]["n_chains"] = self.idata.posterior.sizes.get("chain", 4)
       params["convergence_diagnostics"]["n_draws"] = self.idata.posterior.sizes.get("draw", 1000)
       
       return params
   ```

3. **Verify R-hat values included** (if available in idata)
   ```python
   # Optional: Include R-hat for each parameter
   if hasattr(self.idata, "posterior"):
       r_hats = az.rhat(self.idata)
       params["r_hats"] = r_hats.to_dict()
   ```

4. **Test locally**
   ```python
   params = model.get_params()
   assert "channel_posteriors" in params
   for ch in ["tv", "search", "social", "display", "video"]:
       assert ch in params["channel_posteriors"], f"{ch} missing"
   ```

### Success Criteria
- [ ] Code compiles without errors
- [ ] `get_params()` includes all 5 channels in posteriors
- [ ] All posterior statistics (mean, std, CI) included
- [ ] Convergence diagnostics present (n_chains, n_draws, R-hat if available)
- [ ] Recovery on S1 remains 72.6% ±2%
- [ ] Validation pipeline returns exit code 0
- [ ] Output is JSON serializable (may need custom encoder for numpy types)

### Blockers
- None; may require numpy-to-float conversion for JSON serialization

---

## VALIDATION WORKFLOW

### After Each Issue Fix

**Sub-Agent 1 or 2 completes an issue → Sub-Agent 3 runs validation:**

```
Sub-Agent 1/2: Completes Issue #N
                ↓
Sub-Agent 3:   Run validation_pipeline/validate_before_benchmark.py
                - Checks contract compliance
                - Runs edge case tests
                - Generates VALIDATION_REPORT_{timestamp}.md
                ↓
                If exit code 0:
                  ✓ Issue #N VALIDATED
                  Move to next issue
                ↓
                If exit code 1:
                  ✗ Issue #N FAILED validation
                  Return to Sub-Agent 1/2 with specific error
```

### Regression Testing (After All 10 Issues)

**After all fixes validated:**
```
Run: python experiments/run_experiment.py --scenario S1 --all-models

Expected results (within ±2% of baseline):
  - geo_adstock: 69.9% recovery
  - weibull_adstock: 10.5% recovery
  - almon_pdl: 42.6% recovery
  - dual_adstock: 0.0% recovery (architectural)
  - ardl: >50% recovery (after Issue #5 fix)
  - finite_dl: 50.3% recovery
  - koyck: 46.4% recovery
  - kalman_dlm: 82.0% recovery
  - bsts: 82.4% recovery
  - mcmc_stock: 72.6% recovery
```

---

## EXECUTION PLAN

### Phase 1: Sub-Agent 1 (Issues #1-5)
1. Fix Issue #1 (DualAdstockOLS) — 15 min
2. Sub-Agent 3 validates Issue #1
3. Fix Issue #2 (WeibullAdstockNLS) — 15 min
4. Sub-Agent 3 validates Issue #2
5. Fix Issue #3 (GeometricAdstockOLS) — 15 min
6. Sub-Agent 3 validates Issue #3
7. Fix Issue #4 (AlmonPDL) — 10 min
8. Sub-Agent 3 validates Issue #4
9. Fix Issue #5 (ARDLModel) — 30 min (investigation + fix)
10. Sub-Agent 3 validates Issue #5

**Phase 1 Total:** ~90 minutes

### Phase 2: Sub-Agent 2 (Issues #6-10)
1. Fix Issue #6 — 10 min
2. Sub-Agent 3 validates Issue #6
3. Fix Issue #7 — 10 min
4. Sub-Agent 3 validates Issue #7
5. Fix Issue #8 (BayesianStructuralTS) — 5 min
6. Sub-Agent 3 validates Issue #8
7. Fix Issue #9 (KalmanDLM) — 5 min
8. Sub-Agent 3 validates Issue #9
9. Fix Issue #10 (MCMCLatentStock) — 10 min
10. Sub-Agent 3 validates Issue #10

**Phase 2 Total:** ~50 minutes

### Phase 3: Final Regression Testing
- Run all 10 models on S1-S5
- Verify recovery within ±2% of baseline
- **Phase 3 Total:** ~30 minutes

---

## KEY FILES

- **IMPACT_REPORT.md** — Full technical analysis of all 10 issues
- **detailed_findings.txt** — Code-level analysis with line numbers
- **comparison_before_after.csv** — Quantitative impact table

---

## QUESTIONS FOR SUB-AGENTS

**If clarification needed during fix:**
1. Read the relevant section in IMPACT_REPORT.md
2. Check detailed_findings.txt for code context
3. Reference the exact file:line numbers provided in this guide
4. If still unclear, document the blocker for the user

