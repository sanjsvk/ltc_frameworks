# LTC Paper - Experiment Notes

Findings appended automatically after each model-run.
Format: ## EXP-[id] - [date]

---


## EXP-01a — S1 × F1 × geo_adstock — 2026-04-22
**Finding:** geo_adstock over-estimates total LTC by 26.7% on S1. Aggregate MAPE=30.1%. TV budget share over-recovered by 15.8pp.
**Paper point:** Section 6 opening — performance ceiling (ideal conditions).
**Key metrics:** ltc_mape_total=30.13%  ltc_total_error=+0.267  baseline_mape=15.85%
**Budget errors (pp):** tv=+0.158  search=+0.411  social=+0.274  display=-0.290  video=-0.554
**Anomaly:** None

## EXP-01b — S1 × F1 × weibull_adstock — 2026-04-22
**Finding:** weibull_adstock over-estimates total LTC by 89.1% on S1. Aggregate MAPE=90.3%. TV budget share over-recovered by 27.2pp.
**Paper point:** Section 6 opening — performance ceiling (ideal conditions).
**Key metrics:** ltc_mape_total=90.30%  ltc_total_error=+0.891  baseline_mape=22.45%
**Budget errors (pp):** tv=+0.272  search=+0.439  social=+0.151  display=-0.049  video=-0.814
**Anomaly:** None

## EXP-01c — S1 × F1 × almon_pdl — 2026-04-22
**Finding:** almon_pdl under-estimates total LTC by 56.7% on S1. Aggregate MAPE=57.4%. TV budget share over-recovered by 328.3pp.
**Paper point:** Section 6 opening — performance ceiling (ideal conditions).
**Key metrics:** ltc_mape_total=57.42%  ltc_total_error=-0.567  baseline_mape=9.24%
**Budget errors (pp):** tv=+3.283  search=+0.060  social=+0.477  display=-0.822  video=-2.998
**Anomaly:** None

## EXP-01d — S1 × F1 × dual_adstock — 2026-04-22
**Finding:** dual_adstock under-estimates total LTC by 789.9% on S1. Aggregate MAPE=789.9%. TV budget share over-recovered by 83.6pp.
**Paper point:** Section 6 opening — performance ceiling (ideal conditions).
**Key metrics:** ltc_mape_total=789.93%  ltc_total_error=-7.899  baseline_mape=12.89%
**Budget errors (pp):** tv=+0.836  search=+0.001  social=+0.461  display=-0.134  video=-1.163
**Anomaly:** None

## EXP-02a — S1 × F2 × koyck — 2026-04-22
**Finding:** koyck under-estimates total LTC by 53.3% on S1. Aggregate MAPE=53.6%. TV budget share under-recovered by 29.4pp.
**Paper point:** Section 6 opening — performance ceiling (ideal conditions).
**Key metrics:** ltc_mape_total=53.62%  ltc_total_error=-0.533  baseline_mape=23.73%
**Budget errors (pp):** tv=-0.294  search=+0.171  social=+0.311  display=+0.028  video=-0.215
**Anomaly:** None

## EXP-02b — S1 × F2 × ardl — 2026-04-22
**Finding:** ardl under-estimates total LTC by 208.0% on S1. Aggregate MAPE=210.2%. TV budget share under-recovered by 34.4pp.
**Paper point:** Section 6 opening — performance ceiling (ideal conditions).
**Key metrics:** ltc_mape_total=210.17%  ltc_total_error=-2.080  baseline_mape=36.83%
**Budget errors (pp):** tv=-0.344  search=-0.150  social=-0.178  display=-0.067  video=-0.261
**Anomaly:** None

## EXP-02c — S1 × F2 × finite_dl — 2026-04-22
**Finding:** finite_dl under-estimates total LTC by 50.3% on S1. Aggregate MAPE=49.7%. TV budget share over-recovered by 52.1pp.
**Paper point:** Section 6 opening — performance ceiling (ideal conditions).
**Key metrics:** ltc_mape_total=49.67%  ltc_total_error=-0.503  baseline_mape=11.07%
**Budget errors (pp):** tv=+0.521  search=-0.027  social=+0.547  display=-0.047  video=-0.995
**Anomaly:** None

## EXP-03a — S1 × F3 × kalman_dlm — 2026-04-22
**Finding:** kalman_dlm under-estimates total LTC by 1091.4% on S1. Aggregate MAPE=1091.6%. TV budget share over-recovered by 89.7pp.
**Paper point:** Section 6 opening — performance ceiling (ideal conditions).
**Key metrics:** ltc_mape_total=1091.62%  ltc_total_error=-10.914  baseline_mape=12.14%
**Budget errors (pp):** tv=+0.897  search=-0.061  social=+0.505  display=-0.083  video=-1.259
**Anomaly:** None

## EXP-03b — S1 × F3 × mcmc_stock — 2026-04-22
**Finding:** mcmc_stock under-estimates total LTC by 27.9% on S1. Aggregate MAPE=27.8%. TV budget share under-recovered by 16.4pp.
**Paper point:** Section 6 opening — performance ceiling (ideal conditions).
**Key metrics:** ltc_mape_total=27.83%  ltc_total_error=-0.279  baseline_mape=25.30%
**Budget errors (pp):** tv=-0.164  search=+0.234  social=+0.204  display=-0.047  video=-0.227
**Anomaly:** None

## EXP-03c — S1 × F3 × bsts — 2026-04-22
**Finding:** bsts under-estimates total LTC by 4.6% on S1. Aggregate MAPE=17.6%. TV budget share under-recovered by 0.7pp.
**Paper point:** Section 6 opening — performance ceiling (ideal conditions).
**Key metrics:** ltc_mape_total=17.57%  ltc_total_error=-0.046  baseline_mape=15.10%
**Budget errors (pp):** tv=-0.007  search=+0.393  social=+0.266  display=-0.143  video=-0.509
**Anomaly:** None

## EXP-03a — S1 × F3 × kalman_dlm — 2026-04-22
**Finding:** kalman_dlm under-estimates total LTC by 99.3% on S1. Aggregate MAPE=99.3%. TV budget share under-recovered by 17.9pp.
**Paper point:** Section 6 opening — performance ceiling (ideal conditions).
**Key metrics:** ltc_mape_total=99.31%  ltc_total_error=-0.993  baseline_mape=16.11%
**Budget errors (pp):** tv=-0.179  search=+0.353  social=+0.283  display=-0.146  video=-0.311
**Anomaly:** None

## EXP-03b — S1 × F3 × mcmc_stock — 2026-04-22
**Finding:** mcmc_stock under-estimates total LTC by 27.9% on S1. Aggregate MAPE=27.8%. TV budget share under-recovered by 16.4pp.
**Paper point:** Section 6 opening — performance ceiling (ideal conditions).
**Key metrics:** ltc_mape_total=27.83%  ltc_total_error=-0.279  baseline_mape=25.30%
**Budget errors (pp):** tv=-0.164  search=+0.234  social=+0.204  display=-0.047  video=-0.227
**Anomaly:** None

## EXP-03c — S1 × F3 × bsts — 2026-04-22
**Finding:** bsts under-estimates total LTC by 4.6% on S1. Aggregate MAPE=17.6%. TV budget share under-recovered by 0.7pp.
**Paper point:** Section 6 opening — performance ceiling (ideal conditions).
**Key metrics:** ltc_mape_total=17.57%  ltc_total_error=-0.046  baseline_mape=15.10%
**Budget errors (pp):** tv=-0.007  search=+0.393  social=+0.266  display=-0.143  video=-0.509
**Anomaly:** None

## EXP-03a — S1 × F3 × kalman_dlm — 2026-04-22
**Finding:** kalman_dlm over-estimates total LTC by 5.4% on S1. Aggregate MAPE=18.0%. TV budget share under-recovered by 17.9pp.
**Paper point:** Section 6 opening — performance ceiling (ideal conditions).
**Key metrics:** ltc_mape_total=18.00%  ltc_total_error=+0.054  baseline_mape=16.11%
**Budget errors (pp):** tv=-0.179  search=+0.353  social=+0.283  display=-0.146  video=-0.311
**Anomaly:** None

## EXP-03b — S1 × F3 × mcmc_stock — 2026-04-22
**Finding:** mcmc_stock under-estimates total LTC by 27.9% on S1. Aggregate MAPE=27.8%. TV budget share under-recovered by 16.4pp.
**Paper point:** Section 6 opening — performance ceiling (ideal conditions).
**Key metrics:** ltc_mape_total=27.83%  ltc_total_error=-0.279  baseline_mape=25.30%
**Budget errors (pp):** tv=-0.164  search=+0.234  social=+0.204  display=-0.047  video=-0.227
**Anomaly:** None

## EXP-03c — S1 × F3 × bsts — 2026-04-22
**Finding:** bsts under-estimates total LTC by 4.6% on S1. Aggregate MAPE=17.6%. TV budget share under-recovered by 0.7pp.
**Paper point:** Section 6 opening — performance ceiling (ideal conditions).
**Key metrics:** ltc_mape_total=17.57%  ltc_total_error=-0.046  baseline_mape=15.10%
**Budget errors (pp):** tv=-0.007  search=+0.393  social=+0.266  display=-0.143  video=-0.509
**Anomaly:** None
