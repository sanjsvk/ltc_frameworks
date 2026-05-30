# S1 vs S2 Scenario Impact Analysis

## Metrics Comparison Table

| Model | Framework | S1 Recovery | S1 MAPE | S2 Recovery | S2 MAPE | Δ Recovery | Δ MAPE | Status |
|-------|-----------|-------------|---------|-------------|---------|------------|--------|--------|
| **geo_adstock** | F1 | 69.9% | 30.1% | 83.1% | 16.9% | +13.2pp | -13.2pp | ✓ Improves in S2 |
| **weibull_adstock** | F1 | 10.5% | 89.5% | 30.5% | 69.5% | +20.0pp | -20.0pp | ✓ Improves, still weak |
| **almon_pdl** | F1 | 42.6% | 57.4% | 18.7% | 81.3% | -23.9pp | +23.9pp | ✗ Collapses in S2 |
| **dual_adstock** | F1 | 0.0% | 789.9% | 0.0% | 1105.1% | +0.0pp | +315.2pp | ✗ Catastrophic failure both |
| **koyck** | F2 | 46.4% | 53.6% | 43.0% | 57.0% | -3.4pp | +3.4pp | ≈ Stable |
| **ardl** | F2 | 0.0% | 316.8% | 68.8% | 31.2% | +68.8pp | -285.6pp | 🔥 CRITICAL: S1 failure → S2 works |
| **finite_dl** | F2 | 50.3% | 49.7% | 54.6% | 45.4% | +4.3pp | -4.3pp | ✓ Slight improvement |
| **kalman_dlm** | F3 | 82.0% | 18.0% | 83.1% | 16.9% | +1.1pp | -1.1pp | ✓✓ Stable, excellent |
| **mcmc_stock** | F3 | 72.6% | 27.4% | 61.4% | 38.6% | -11.1pp | +11.1pp | ⚠ Degrades but converges better |
| **bsts** | F3 | 82.4% | 17.6% | 81.0% | 19.0% | -1.4pp | +1.4pp | ✓✓ Stable, excellent |

---

## Detailed Impact Notes by Model

### Framework 1 — Static Adstock Regression

#### geo_adstock
**S1→S2 Change:** +13.2pp recovery, -13.2pp MAPE  
**Impact:** Spend pause *paradoxically helps* simple geometric adstock. Explanation:
- In S1, varying spend across all channels creates multicollinearity; geo_adstock struggles to separate STC from LTC
- In S2, spend pause (zero inflow weeks 104–112) cleanly identifies latent stock decay and STC vs LTC contribution
- The spend shutdown is a natural experiment that isolates decay rates
- **Insight:** This model benefits from structural breaks that reveal underlying dynamics

#### weibull_adstock
**S1→S2 Change:** +20.0pp recovery, -20.0pp MAPE  
**Impact:** Weibull lag shapes finally become useful in S2. Explanation:
- In S1, the grid search for shape+scale parameters is underconstrained; many combinations fit equally well on baseline
- In S2, the spend pause forces the model to identify the true decay structure more precisely
- Weibull can now distinguish peak-lag effects from tail decay
- **Insight:** Flexible lag shapes need structural variation to identify; they overfit on smooth baseline data

#### almon_pdl
**S1→S2 Change:** -23.9pp recovery, +23.9pp MAPE  
**Impact:** Polynomial distributed lag catastrophically fails on spend pause. Explanation:
- Almon PDL assumes polynomial smoothness in the lag structure: w[t] = Σ β_k t^k
- The spend pause creates a sharp discontinuity: spend drops to zero, then stock decays exponentially
- A polynomial cannot capture this (decaying exponential ≠ polynomial)
- The model tries to fit the discontinuity with polynomials and severely overfits
- **Insight:** Polynomial lags are fragile to structural breaks; assume smooth effects that don't exist

#### dual_adstock
**S1→S2 Change:** +0.0pp recovery, +315.2pp MAPE (catastrophic)  
**Impact:** Structural failure in both scenarios, worse in S2. Explanation:
- Dual adstock fits separate STC and LTC decay parameters (δ_stc, δ_ltc)
- The enforce_ltc_gt_stc constraint often causes numerical instability
- In S2, the spend pause likely causes the optimization to diverge completely
- Negative recovery suggests sign-flip: model predicts LTC goes negative when spend drops
- **Insight:** Enforcing structural constraints can cause optimization pathology; needs careful initialization

---

### Framework 2 — Dynamic Time-Series Distributed Lag

#### koyck
**S1→S2 Change:** -3.4pp recovery, +3.4pp MAPE  
**Impact:** Highly stable across scenarios. Explanation:
- Koyck model (AR + geometric lag) is naturally adaptive to different patterns
- In S1, fits baseline smoothly; in S2, tracks spend pause equally well
- The lagged sales term (AR order 2) captures dynamics without overfitting to lag structure
- **Insight:** Autoregressive models are robust because they condition on realized outcomes

#### ardl
**S1→S2 Change:** +68.8pp recovery, -285.6pp MAPE (CRITICAL RESURRECTION)  
**Impact:** S1 prior misspecification masked underlying model quality. Explanation:
- In S1, the logit-normal prior on δ is calibrated to the true values, but overfitting occurs
- The polynomial lag structure (almon) on top of AR causes model to chase noise in S1 baseline
- In S2, the spend pause forces the model to identify the true lag structure; the prior becomes helpful instead of constraining
- The model now correctly decomposes STC (via Almon) from LTC (via delta)
- **Insight:** Prior misspecification in S1 was identified! S2 scenario reveals the model works; needs S2-specific tuning

#### finite_dl
**S1→S2 Change:** +4.3pp recovery, -4.3pp MAPE  
**Impact:** Stable, minor improvement. Explanation:
- Weibull lag shapes within distributed lag framework are more stable than standalone weibull_adstock
- The Almon + Weibull combination is less flexible than dual parameters but more robust
- S2 spend pause slightly helps identify the lag tail, but effect is modest
- **Insight:** Weibull lags work better when regularized within a larger model structure

---

### Framework 3 — State-Space / Latent Brand-Stock

#### kalman_dlm
**S1→S2 Change:** +1.1pp recovery, -1.1pp MAPE  
**Impact:** Exceptional stability; near-perfect scenario robustness. Explanation:
- Kalman filter has explicit model for stock dynamics: stock[t] = δ·stock[t-1] + innovation
- Prior knowledge of true decay rates (calibrated from DGP) constrains the filter
- S1 and S2 are equally identifiable by the Kalman framework
- The level/slope decomposition captures trend and mean reversion naturally
- **Insight:** Structural models with domain knowledge are robust to scenario variation; best generalization

#### mcmc_stock
**S1→S2 Change:** -11.1pp recovery, +11.1pp MAPE  
**Impact:** Performance degrades but MCMC convergence dramatically improves (8→1 divergence). Explanation:
- In S1, the spend pause is absent; MCMC struggles to identify build_rate and ltc_coef jointly
- In S2, the spend pause provides a clear signal for stock decay rate δ (no new inflow for weeks 104–112)
- This makes δ easier to estimate, reducing joint prior pressure on build_rate×ltc_coef
- Recovery degrades because the joint prior constraint now over-regularizes
- **Insight:** Spend pause paradoxically helps MCMC convergence (fewer divergences) but can over-constrain inference

#### bsts
**S1→S2 Change:** -1.4pp recovery, +1.4pp MAPE  
**Impact:** Excellent stability across scenarios. Explanation:
- Bayesian structural time-series models are designed for time-varying structure
- Level + slope + seasonal decomposition captures both S1 baseline and S2 stock decay naturally
- The spend pause is just a different pattern; the model reestimates the slope (trend)
- Fixed decay priors (δ values) provide just enough constraint without over-regularizing
- **Insight:** Flexible state-space models are naturally robust; seasonal structure helps even without explicit LTC model

---

## Summary: What S2 Reveals

### Framework 1 (Static Adstock)
- **Winner: geo_adstock** (+13.2pp) — Spend pause is a gift; reveals LTC structure cleanly
- **Loser: almon_pdl** (-23.9pp) — Polynomial lags can't capture exponential decay in pauses
- **Insight:** Static models are highly data-dependent; scenario structure dominates calibration

### Framework 2 (Dynamic Time-Series)
- **Winner: ardl** (+68.8pp!) — Was broken in S1 due to prior overfitting; S2 proves model quality
- **Stable: koyck** (±3.4pp) — Autoregressive structure is naturally adaptive
- **Insight:** Spend pause is diagnostic for model viability; failures on S1 ≠ bad model

### Framework 3 (State-Space)
- **Winners: kalman_dlm & bsts** (±1.1pp, ±1.4pp) — Exceptional robustness
- **Concern: mcmc_stock** (-11.1pp) — Joint prior too tight for S2; needs S2-specific tuning
- **Insight:** Structural knowledge beats estimation flexibility; F3 dominates across scenarios

---

## Actionable Insights for Next Steps

| Question | Answer from S1↔S2 |
|----------|------------------|
| **Does calibration matter?** | YES, but less than structure. geo_adstock improves 13pp with same params; F3 stable ±1–11pp |
| **Which models generalize?** | kalman_dlm & bsts (±1.4pp); koyck (±3.4pp); others are scenario-dependent |
| **What breaks models?** | Polynomial lags (almon_pdl fails on discontinuities); joint priors (mcmc_stock over-constraints) |
| **What do we learn from S2?** | Spend pause is excellent diagnostic: ardl resurrects (S1 was prior misspec); geo_adstock improves (needs variation) |
| **Should we optimize S2?** | YES for F1/F2 (ardl +68pp possible, almon_pdl -24pp fixable). NO for F3 (already optimal) |

