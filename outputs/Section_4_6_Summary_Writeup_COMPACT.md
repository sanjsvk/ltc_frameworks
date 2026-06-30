# Section 4.6 Summary: Full Recovery Matrix & Framework Hierarchy

## Overview

Table 3 consolidates the recovery accuracy across all 10 models and 5 scenarios, revealing a clear hierarchy of framework robustness. State-space methods (Framework 3) maintain consistent performance across diverse environments (61–65% average recovery), while static and dynamic methods exhibit high scenario dependence (0–84% recovery). Crucially, S5 (weak-signal identification boundary) establishes the limit of frozen-parameter approaches: all 10 models collapse to 0.0% recovery, confirming that signal strength—not architectural design—determines identifiability below an empirical threshold. Supplementary analysis shows MCMC recovers 88.5% with scenario-specific priors, proving the 0% collapse is calibration-dependent.

## Key Results

**Framework 3** dominates (BSTS/Kalman DLM 82%, MCMC 99% on S3), though all three hit 0% on S5 frozen parameters. **Framework 2** ranges widely (finite_dl stable at 40–58%, ARDL bimodal 0%/69%, koyck 39–54%). **Framework 1** spans both robustness (geo_adstock 70–83%) and failure (dual_adstock 0%, weibull_adstock 0–12%). The S2 spend-pause reveals structural insights: BSTS achieves 1.02× pause-window ratio (error-invariant to discontinuity), while geo_adstock's 1.40× reflects its dependence on spend variation.

## Technical Notes

Recovery accuracy is defined as `max(0, 100 - MAPE)`, capping at 0%. Uncapped values reveal severity: S4 failures show ARDL −119.8%, dual_adstock −1478%, weibull_adstock −21.5%—predictions worse than zero-LTC baseline. S5 universal collapse (all models 0%) does not indicate architecture failure but empirical non-identifiability under weak signal ($0.176M LTC vs $1.229M S1).

---

## Table 3: Full Recovery Matrix – All Models, All Scenarios

[TABLE FOLLOWS]
