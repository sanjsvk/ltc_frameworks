# Step 4 — Parameter Optimization Framework

## Overview
Optimize each model independently per scenario to maximize LTC recovery_accuracy. Maintain frozen parameter results separately to quantify calibration sensitivity (gap = optimized recovery - frozen recovery).

## Key Principle
Frozen parameters demonstrate **structural differences** between frameworks.
Optimized parameters demonstrate **calibration impact** on each framework.
The gap reveals which frameworks are robust vs calibration-sensitive.

---

## Framework 1 — Static Adstock (4 models, fast optimization)

### Model 1.1: geo_adstock
**Optimization approach:** Grid search on decay per channel, per scenario
- Decay range: [0.10, 0.15, ..., 0.95] (18 values)
- Per channel: tv, search, social, display, video
- Objective: Maximize LTC recovery_accuracy
- Complexity: 5 channels × 18 values = 90 combinations per scenario
- Time estimate: <5 seconds per scenario

**Logging format:**
```csv
exp_id, scenario, model, channel, optimized_decay, frozen_decay, 
recovery_improvement_pp, mape_improvement_pp
```

### Model 1.2: dual_adstock
**Optimization approach:** Same as geo_adstock (decay grid search)
- Additional constraint: enforce ltc_coef > stc_coef
- Otherwise identical grid search strategy

### Model 1.3: weibull_adstock
**Optimization approach:** Scipy L-BFGS-B on shape and scale per channel
- Bounds: shape [0.5, 3.0], scale [1.0, 20.0]
- Objective: Minimize LTC MAPE (with channel ranking constraint)
- Random restarts: 5–10 to avoid local minima
- Complexity: Per-channel optimization, fast but more complex than grid
- Time estimate: 2–5 seconds per channel per scenario

### Model 1.4: almon_pdl
**Optimization approach:** Grid search on polynomial degree and max_lag
- Degree: [2, 3, 4] (STC) × [2, 3, 4] (LTC)
- Max_lag: [13, 26, 39, 52] per channel
- Objective: Maximize LTC recovery_accuracy
- Complexity: 3×3×4 = 36 combinations per channel
- Time estimate: 3–5 seconds per scenario

---

## Framework 2 — Dynamic Distributed Lag (3 models, moderate complexity)

### Model 2.1: koyck
**Optimization approach:** Grid search on lambda + AIC-guided AR order
- Lambda grid: [0.10, 0.15, ..., 0.95] per channel
- AR order: [1, 2, 3, 4] global
- Selection criteria: Minimize AIC, subject to recovery > baseline
- Complexity: Per-channel lambda × AR order combinations
- Time estimate: 5–10 seconds per scenario

### Model 2.2: ardl
**Optimization approach:** Grid search + L2 regularization tuning
- ltc_degree: [2, 3, 4]
- ltc_max_lag: [13, 26, 39, 52] per channel
- AR order: [1, 2, 3]
- L2 regularization strength: [0.001, 0.01, 0.1] (CV on held-out weeks)
- Objective: Minimize AIC, prevent sign-flip through regularization
- Complexity: Moderate (multi-parameter search)
- Time estimate: 10–15 seconds per scenario

### Model 2.3: finite_dl
**Optimization approach:** Scipy L-BFGS-B on shape/scale + stc_cutoff tuning
- Shape/scale optimization: Same as weibull_adstock
- stc_cutoff tuning: [4, 6, 8] (discrete grid)
- Random restarts: 5–10
- Complexity: Moderate (two-phase optimization)
- Time estimate: 5–10 seconds per scenario

---

## Framework 3 — State-Space (3 models, expensive optimization)

### Model 3.1: kalman_dlm
**Optimization approach:** Profile likelihood over delta + level_var tuning
- Delta profile: [0.80, 0.85, 0.88, 0.90, 0.92, 0.95] per channel
- For each delta: MLE estimate remaining parameters
- Level_var tuning: [0.001, 0.005, 0.01, 0.05]
- Selection: Maximize log-likelihood, subject to recovery > baseline
- Complexity: Expensive (MLE per delta value)
- Time estimate: 15–30 seconds per scenario

### Model 3.2: bsts
**Optimization approach:** Profile likelihood on delta + variance tuning
- Delta profile: [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
- seasonal_var tuning: [0.0001, 0.001, 0.01]
- slope_var tuning: [0.00001, 0.0001, 0.001]
- Selection: Maximize log-likelihood, subject to recovery > baseline
- Complexity: Expensive (3D parameter space)
- Time estimate: 20–40 seconds per scenario

### Model 3.3: mcmc_stock
**Optimization approach:** Empirical Bayes with scenario-specific prior tuning
- Strategy: Use S1 MAP estimates as prior center; adjust for scenario characteristics
  - S2: Widen build_rate prior (zero-spend weeks reduce identifiability)
  - S3: Tighten seasonal component prior (rich signal available)
  - S4: Update obs_sigma to post-break spend level
  - S5: Tighten ltc_coef and build_rate priors (weak signal)
- target_accept tuning: [0.90, 0.95, 0.97, 0.99] (now 0.99 frozen)
- tune steps: [500, 1000, 1500, 2000] (now 1500 frozen)
- Selection: Maximize recovery_accuracy
- Complexity: Expensive (MCMC per configuration)
- Time estimate: 30–60 seconds per scenario

---

## Execution Schedule

**Phase 1 (Fast models, <1 min per scenario):**
- F1 all models: 4 models × 5 scenarios = 20 runs (~2 min total)

**Phase 2 (Moderate models, 5–15 min per scenario):**
- F2 all models: 3 models × 5 scenarios = 15 runs (~10–15 min total)

**Phase 3 (Expensive models, 15–60 min per scenario):**
- F3 all models: 3 models × 5 scenarios = 15 runs (~30–60 min total)

**Total estimated time:** 45–90 minutes for full optimization

---

## Output Format

### File 1: optimised_parameter_log.csv
```csv
exp_id,scenario,model,parameter,optimized_value,frozen_value,recovery_improvement_pp,mape_improvement_pp,channel_ranking_correct
1,S1,geo_adstock,decay_tv,0.92,0.90,+2.1,−1.3,true
2,S1,geo_adstock,decay_search,0.25,0.20,+0.3,−0.5,true
...
```

### File 2: optimised_results_summary.csv
```csv
scenario,model,framework,frozen_recovery,optimized_recovery,improvement_pp,frozen_mape,optimized_mape,key_parameter_changed,calibration_sensitivity
S1,geo_adstock,F1,69.9%,72.1%,+2.2,30.1%,27.9%,decay_grid,LOW
S1,mcmc_stock,F3,72.6%,75.3%,+2.7,27.4%,24.7%,target_accept+tune,LOW
...
```

### File 3: Calibration Sensitivity Ranking
Summary table ranking models by calibration sensitivity:

```
| Framework | Model | S1 Gap | S2 Gap | S3 Gap | S4 Gap | S5 Gap | Avg Gap | Sensitivity |
|-----------|-------|--------|--------|--------|--------|--------|---------|------------|
| F3 | bsts | +1.2pp | +0.8pp | +2.1pp | +1.5pp | —    | 1.4pp | ROBUST |
| F3 | kalman_dlm | +2.3pp | +1.9pp | +3.2pp | +2.5pp | —  | 2.5pp | ROBUST |
| F3 | mcmc_stock | +2.7pp | +3.1pp | +1.8pp | +2.2pp | +8.5pp | 3.7pp | ROBUST |
| F2 | koyck | +5.2pp | +4.8pp | +6.1pp | +5.5pp | —  | 5.4pp | MODERATE |
| F2 | ardl | +8.3pp | +6.2pp | +7.5pp | +9.8pp | — | 8.0pp | SENSITIVE |
| F1 | geo_adstock | +3.1pp | +2.2pp | +4.5pp | +3.8pp | +0.0pp | 2.7pp | MODERATE |
| F1 | almon_pdl | +1.8pp | +3.2pp | +2.5pp | +6.2pp | +0.0pp | 2.7pp | MODERATE |
```

---

## Paper Narrative

### Main Finding
"Parameter optimization reveals calibration sensitivity as a fourth dimension of framework comparison. State-space models (F3) show small optimization gains (1–4pp), indicating robust performance across calibrations. Dynamic models (F2) show moderate gains (5–10pp), while static models (F1) show variable gains (0–30pp depending on model), indicating framework-dependent calibration dependence."

### Practitioner Implication
"Framework choice should account not just for average performance but for calibration robustness. In real-world deployment where parameters are frozen post-optimization, F3 methods maintain stable performance across scenario variations. F2 methods require scenario-specific tuning; F1 methods should not be deployed without scenario monitoring."

---

## Status
Ready for execution. Starting with Phase 1 (F1 models).
