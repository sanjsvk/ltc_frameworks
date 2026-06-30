# Table 3: Full Recovery Matrix – All Models, All Scenarios

| Model            | Framework | S1     | S2     | S3     | S4     | S5 (frozen) | Avg Recovery | Notes                                                                                 |
|------------------|-----------|-------:|-------:|-------:|-------:|------------:|-------------:|---------------------------------------------------------------------------------------|
| geo_adstock      | F1        | 69.9%  | 83.1%  | 43.2%  | 63.4%  |  0.0%       |  51.9%       | Best F1; S2 spend pause reveals latent LTC (+13pp); collapses on weak signal          |
| weibull_adstock  | F1        | 11.9%  | 31.6%  |  0.0%  |  0.0%  |  0.0%       |   8.7%       | Architectural limitation; Weibull lag shape cannot isolate LTC in 3 of 5 scenarios    |
| almon_pdl        | F1        | 42.6%  | 18.7%  | 40.6%  | 68.6%  |  0.0%       |  34.1%       | Scenario-variable; polynomial lags hurt S2 discontinuity, unexpectedly strong on S4   |
| dual_adstock     | F1        |  0.0%  |  0.0%  |  0.0%  |  0.0%  |  0.0%       |   0.0%       | Catastrophic sign-flip failure across all scenarios; OLS cannot separate dual streams  |
| koyck            | F2        | 46.4%  | 43.0%  | 53.7%  | 52.3%  |  0.0%       |  39.1%       | Stable mid-range F2; single lambda limits LTC depth; collapses on weak signal          |
| ardl             | F2        |  0.0%  | 68.8%  | 63.3%  |  0.0%  |  0.0%       |  26.4%       | Bimodal: S1 prior misspecification; strong S2–S3; display collinearity collapses S4    |
| finite_dl        | F2        | 50.3%  | 54.6%  | 58.0%  | 40.5%  |  0.0%       |  40.7%       | Most stable F2 model; consistent 40–58% on S1–S4; falls to 0% on weak signal          |
| kalman_dlm       | F3        | 82.0%  | 83.1%  | 64.9%  | 75.4%  |  0.0%       |  61.1%       | State-space excellence S1–S4; no seasonal state limits S3; fails S5 identification     |
| mcmc_stock       | F3        | 72.6%  | 60.9%  | 98.9%  | 91.8%  |  0.0% †     |  64.8%       | Best on S3/S4; S5 achieves 88.5% with scenario-specific priors (not frozen)            |
| bsts             | F3        | 82.4%  | 81.0%  | 76.8%  | 81.6%  |  0.0%       |  64.4%       | Most consistent F3; tightest cross-scenario variance across S1–S4                      |
| **Avg per Scenario** |       | **45.8%** | **52.5%** | **49.9%** | **47.4%** | **0.0%** | **39.1%** | |

† mcmc_stock S5 frozen = 0.0%; with scenario-calibrated priors: **88.5%** (supplementary run)

## Notes for Paper

- Source: `outputs/results/{model}_{scenario}.json`, field `ltc.total.recovery_accuracy`
- S5 frozen results: 0.0% across all 10 models (weak-signal identification boundary)
- mcmc_stock S5 exception: 88.5% recovery with scenario-specific priors (separate supplementary run, not frozen-parameter JSON)
- Average Recovery includes S5=0% in denominator across all 5 scenarios (depresses F3 averages; should be noted explicitly in paper)

## Framework Hierarchy (from Table 3)

**Tier 1 – State-Space Excellence (avg 63–65%)**
- bsts: 64.4%
- mcmc_stock: 64.8%
- kalman_dlm: 61.1%

**Tier 2 – Dynamic/Distributed-Lag Mid-Range (avg 39–40%)**
- finite_dl: 40.7%
- koyck: 39.1%

**Tier 3 – Static Adstock Variable (avg 34–52%)**
- geo_adstock: 51.9% (best F1)
- almon_pdl: 34.1%

**Tier 4 – Architectural Failures (avg 0–26%)**
- ardl: 26.4% (bimodal; strong S2-S3, fails S1/S4/S5)
- weibull_adstock: 8.7%
- dual_adstock: 0.0% (universal failure)
