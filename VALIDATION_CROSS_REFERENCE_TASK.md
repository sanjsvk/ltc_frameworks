# Cross-Reference Parentheticals Task

**For: Validation Agent (af30341259a41d8cf)**

**Execute After:** Phases 3, 4, 5, and 6 validation complete

---

## Objective
Add navigation parentheticals to the paper that cross-reference explained claims in Sections 4-5 to their detailed mechanistic explanations in Sections 8-9.

## Why This Matters
Readers encountering surprising results (e.g., ARDL 0% → 68.8% resurrection) in early sections need signposting to detailed explanations. These parentheticals provide non-intrusive navigation without disrupting flow.

---

## Task: Add 8 Cross-Reference Parentheticals

### 1. Almon PDL Discontinuity Failure
- **Location:** Section 4.1, line ~373
- **Trigger phrase:** "fail on discontinuous spend patterns"
- **Add after phrase:** 
  ```
  (see Section 8.3 for mechanistic explanation)
  ```

### 2. Weibull Architectural Ceiling
- **Location:** Section 4.1, line ~373
- **Trigger phrase:** "sacrifice one for the other"
- **Add after phrase:**
  ```
  (see Section 8.3 for detailed analysis)
  ```

### 3. Dual Adstock Sign-Flip
- **Location:** Section 4.1, line ~377
- **Trigger phrase:** "sign-flipped predictions and negative LTC estimates"
- **Add after phrase:**
  ```
  (see Section 8.3 for root cause analysis)
  ```

### 4. ARDL 0% → 68.8% Resurrection
- **Location:** Section 4.2, first major mention of resurrection
- **Trigger phrase:** "0% to 68.8%"
- **Add after phrase:**
  ```
  (see Section 8.3 for prior misspecification explanation)
  ```

### 5. Geo_adstock Identification Paradox
- **Location:** Section 4.2, line ~399
- **Trigger phrase:** "paradoxically improves"
- **Add after phrase:**
  ```
  (see Section 8.4 for mechanistic explanation)
  ```

### 6. Kalman DLM Seasonal State Issue
- **Location:** Section 4.3, mention of S3 degradation
- **Trigger phrase:** "due to missing explicit seasonal state"
- **Add after phrase:**
  ```
  (see Section 8.3 for detailed analysis)
  ```

### 7. MCMC S3 Peak Performance
- **Location:** Section 4.3, mention of 99% recovery
- **Trigger phrase:** "MCMC peaks at 99.0% recovery"
- **Add after phrase:**
  ```
  (see Section 8.4 for explanation of Bayesian flexibility)
  ```

### 8. BSTS Channel Inversions
- **Location:** Section 4.3, mention of channel inversion
- **Trigger phrase:** "BSTS inverts ranking"
- **Add after phrase:**
  ```
  (see Section 8.3 for channel-level caveat)
  ```

---

## Format Guidelines

- **Placement:** End of sentence containing trigger phrase
- **Style:** Brief, non-intrusive parentheticals
- **Scope:** Only at FIRST major mention in Sections 4-5
- **Syntax:** Standard markdown `(see Section X.Y for ...)`

---

## Validation

After adding all 8 references:
1. Verify each cross-reference target section exists and contains the referenced content
2. Confirm parentheticals don't disrupt sentence flow
3. Check no duplicate cross-references to same section
4. Validate all links point to actual section numbers (e.g., 8.3, 8.4)

---

## Report After Completion

Create `CROSS_REFERENCE_COMPLETION_REPORT.md` documenting:
- All 8 references added with line numbers
- Verification checklist (4 items above)
- Any adjustments made to phrasings
- Timestamp of completion

---

**Status:** Pending Validation Agent execution after Phases 3-6
**Created:** 2026-05-29
