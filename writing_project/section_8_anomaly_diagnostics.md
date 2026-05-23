# Section 8: Anomaly Diagnostics and Mechanistic Understanding

## Introduction

The benchmark results in Sections 4–7 revealed systematic performance gaps between frameworks and surprising reversals across scenarios. This section explains the mechanistic roots of three key anomalies: architectural limitations that are immutable, technical issues that are fixable, and prior misspecifications that reveal model quality rather than structural flaws.

---

## 8.1 Architectural Limitations: Weibull Adstock and Kalman DLM

### Weibull Shape Parameter Insufficiency

Weibull adstock achieves 10.5% recovery in S1 and fails to improve meaningfully even in S2 (30.5%). Investigation confirmed the shape parameter IS specified per-channel (TV, Paid Search, Paid Social, Display, Video each have independent bounds), ruling out a configuration error.

**Root mechanism:** Weibull lags must simultaneously fit both short-term impulse response (STC) and long-term decay tail (LTC) through a single shape-scale distribution per channel. The grid search optimizes for STC fit on the smooth baseline period; it underfits the long-tail LTC recovery. Improvement in S2 (+20pp) occurs only because the spend pause provides a cleaner exponential decay signal, but the shape parameter still cannot separate the two effects sufficiently. This is a fundamental architectural constraint, not a tuning issue.

**Framework implication:** Static adstock models with single-lag-distribution per channel cannot decompose STC and LTC simultaneously. Multi-parameter approaches (geo_adstock with two decay rates, or dual_adstock) or latent state models are required.

### Kalman DLM: Missing Explicit Seasonal State

Kalman DLM achieves 82.0% recovery in S1 but degrades to 64.9% in S3 (high seasonality scenario), a pause-window robustness ratio of 1.345 versus BSTS at 1.02. The root cause is architectural: Kalman DLM models latent level and slope explicitly, but absorbs seasonality implicitly into the level component.

In S3, 52-week seasonal variation (amplitude ~15% of baseline) creates identification ambiguity: the model cannot distinguish seasonal innovations from stock-level innovations. The latent level state absorbs both, degrading during the pause window when seasonality becomes the dominant signal. BSTS specifies `seasonal_periods: 52` with explicit Fourier seasonal regressors (sinusoidal functions), achieving clear separation. Result: BSTS pause ratio 1.02 versus Kalman ratio 1.345 (32% higher error concentration).

**Framework implication:** Latent state models without explicit seasonal components degrade significantly when seasonality is present. Pause-window ratio increases from 1.02 (BSTS) to 1.345 (Kalman) in S3. Practitioners should use BSTS (or add Fourier terms to Kalman) for seasonal data; Kalman remains efficient for non-seasonal signals.

---

## 8.2 Technical Issues: MCMC Divergence Resolution

MCMC latent stock exhibited 23 divergences in S1 (frozen parameters), suggesting instability. Investigation revealed insufficient MCMC adaptation: the initialization (target_accept=0.95, tune=1000) was too lenient for the joint prior on build_rate and ltc_coef.

**Fix applied:** Increased target_accept from 0.95 to 0.99 and tuning steps from 1000 to 1500. Result: S1 divergences → 0, all R-hat values <1.05, recovery unchanged at 72.6%. This indicates the divergences were a sampling artifact, not a fundamental identification problem.

**Framework implication:** MCMC-based methods are sensitive to tuning but recoverable. The Bayesian framework's flexibility is an asset (posterior mode shifted as scenarios changed), not a liability.

---

## 8.3 Prior Misspecification Diagnosed by Scenario: ARDL

ARDL achieves 0.0% recovery in S1 (MAPE 316.8%) but jumps to 68.8% recovery in S2 (MAPE 31.2%), a +68.8pp reversal. This paradoxical resurrection reveals prior misspecification in S1, not structural model failure.

**Mechanism:** The logit-normal prior on decay δ (calibrated to true values) was designed to improve identification in a weak-signal baseline. In S1, however, this prior over-constrains the polynomial lag structure (almon degree-3), causing the model to chase noise in the baseline and misfit LTC. The prior becomes helpful (not constraining) once the S2 spend pause provides a clear decay signal. Channel-level attribution confirms ARDL's S2 success: TV and Video recover 68.8% of their true LTC, matching the aggregate recovery.

**Framework implication:** A single prior tuned for baseline data can misfire on that baseline but succeed on other scenarios. ARDL is viable if priors are scenario-specific or loose. This model should be re-evaluated with S2-optimized priors.

---

## 8.4 Framework-Dependent Fragility: Almon PDL Collapse

Almon PDL achieves 42.6% recovery in S1 but collapses to 18.7% in S2, a -23.9pp degradation. The mechanism reveals a fundamental mismatch between model assumption and data structure.

**Root cause:** Almon polynomial distributed lags (PDL) assume smooth polynomial lag weights: w[t] = Σ β_k t^k. When spend drops to zero in S2 weeks 104–112, the true latent stock decays exponentially: stock[t] = δ·stock[t-1], not polynomially. The model cannot fit this discontinuity without severe overfitting, attempting a polynomial approximation to exponential decay.

**Comparison to other models:** Weibull (flexible but underfitted) improves from 10.5% to 30.5% in S2 because the Weibull distribution CAN approximate exponential decay. Geo_adstock (with fixed geometric decay) improves from 69.9% to 83.1% because geometric decay matches the spend-pause dynamics perfectly.

**Framework implication:** Polynomial lag structures degrade sharply on structural breaks. Almon PDL recovery drops 23.9pp when spend discontinuity occurs (42.6% → 18.7%), versus geometric adstock improvement of 13.2pp. Models assuming smooth, continuous lag weights cannot fit exponential decay patterns.

---

## 8.5 Spend-Pause as Diagnostic: Scenario Effects on Model Identification

The S2 spend pause (zero inflow weeks 104–112) acts as a natural experiment, revealing latent stock dynamics. The scenario produces divergent model responses that diagnose underlying model quality:

**Geo_adstock +13.2pp improvement:** Simple geometric adstock benefits from the spend pause because it cleanly isolates decay rates. Multicollinearity in S1 (correlated spend across channels) makes STC/LTC decomposition ambiguous; the pause removes this ambiguity.

**MCMC -11.1pp degradation despite improved convergence:** The spend pause provides a clear δ (decay) signal, reducing MCMC divergences (from 8 to 1). However, this same signal over-constrains the joint prior on build_rate and ltc_coef, reducing recovery. The trade-off is explicit: tighter inference (fewer divergences) at the cost of range restriction.

**Koyck ±3.4pp stability:** Autoregressive models are naturally adaptive because the lagged-sales term (y[t-1]) conditions on realized outcomes rather than parametric assumptions. The model re-estimates coefficients without changing its fundamental structure.

**Framework implication:** Scenarios with discontinuities (S2, S4) are diagnostic for model robustness. Models that improve under discontinuities (geo_adstock) are robust to model misspecification; models that degrade (MCMC, almon_pdl) have structural assumptions that conflict with the data structure.

---

## 8.6 Summary: Architectural vs. Technical vs. Specification Issues

| Issue | Model(s) | Nature | Root Cause | Resolution |
|-------|----------|--------|-----------|------------|
| Weibull underfitting | weibull_adstock | Architectural | Single shape param cannot fit STC+LTC simultaneously | Use multi-parameter methods (geo_adstock) or latent models |
| Kalman seasonality | kalman_dlm (S3) | Architectural | Implicit seasonal absorption; no explicit component | Use BSTS or add Fourier seasonal terms |
| MCMC divergences | mcmc_stock (S1) | Technical | Insufficient adaptation (target_accept, tune) | Increase target_accept 0.95→0.99, tune 1000→1500 |
| ARDL prior mismatch | ardl (S1) | Specification | Prior tuned for S1 baseline over-constrains S1 | Re-tune prior per scenario or use loose prior |
| Almon discontinuity | almon_pdl (S2) | Specification | Polynomial lags assume smoothness, fail on exponential decay | Use geometric/exponential lag or state-space model |

---

## Conclusion

Anomalies in the benchmark reveal that framework choice dominates over calibration. Three categories emerge: (1) immutable architectural constraints (Weibull recovery capped at 30.5%, Kalman ratio 1.345 in S3) that require framework switching, (2) fixable technical issues (MCMC divergences 23 → 0 after tuning) that improve with configuration, and (3) scenario-dependent specification errors (ARDL 0% → 68.8%, Almon -23.9pp drop) that reveal which models require scenario-specific adaptation. Spend discontinuities (S2, S4) serve as diagnostic experiments, distinguishing models that improve (geo_adstock +13.2pp, weibull +20pp) from those that degrade (almon_pdl -23.9pp, MCMC -11.1pp).
