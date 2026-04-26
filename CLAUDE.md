# LTC Frameworks — Project Context

## Project Overview

**Title:** Long-Term Media Contribution (LTC) Estimation — Framework Benchmarking Study
**Purpose:** Research paper written for an EB1-A visa portfolio as a Data Scientist, evaluating multiple analytical frameworks for estimating long-term media ROI in Media Mix Modelling (MMM).
**Angle:** Data science + marketing, comparing methodological trade-offs across realistic synthetic scenarios.

---

## Research Goal

Evaluate and compare three classes of frameworks for estimating Long-Term Contributions (LTC) of media investments, using synthetic ground-truth data to objectively benchmark each approach:

1. **Static Adstock Regression** — geometric / Weibull adstock transformations applied to impressions; coefficient interpreted as combined STC+LTC
2. **Dynamic Time-Series Models** — e.g., Dynamic Linear Models (DLM), Kalman Filter-based; decompose evolving media effects over time
3. **State-Space / Latent Brand-Stock Models** — explicit latent brand equity accumulation; closest to ground-truth data-generating process

---

## Ground Truth Data-Generating Process

```
Net Sales = Baseline + STC + LTC + Exogenous Effects + Noise
```

**Short-Term Contribution (STC):** Geometric adstock on impressions with channel-specific decay.

**Long-Term Contribution (LTC):** Latent brand stock model:
```
stock[t] = δ × stock[t-1] + build_rate × √spend[t]
LTC[t]   = ltc_coef × stock[t]
```

**Baseline:** Piecewise linear trend + annual seasonality + holiday uplifts (~$10M–$12M/week).

---

## Dataset Specifications

| Attribute | Value |
|-----------|-------|
| Time span | 2020-01-06 to 2025-12-29 |
| Granularity | Weekly (261 weeks) |
| Scenarios | S1–S5 (5 synthetic scenarios) |
| Channels | TV, Paid Search, Paid Social, Display, Video |
| Columns | 39 per CSV |
| KPI | Net sales ($M weekly) |

### Channel Summary

| Channel | Avg Weekly Spend | Always-On % | LTC Potential | δ (retention) |
|---------|-----------------|-------------|---------------|---------------|
| TV | $1.0M | 40% | High | 0.90 |
| Paid Search | $0.2M | 80% | Negligible | 0.30 |
| Paid Social | $0.28M | 50% | Medium | 0.82 |
| Display | $0.10M | 85% | Low | 0.65 |
| Video | $0.50M | 30% | High | 0.88 |

### Contribution Benchmarks (Weekly Average)

- STC: ~$1.58M (~15% of observed sales) — TV & Video dominant
- LTC: ~$1.23M (~12% of observed sales) — TV & Video = 77% of total LTC
- Exogenous effects: -$1.5M to +$2.0M range

---

## Exogenous Variables

- `promo` — Promotional calendar intensity (0–0.18)
- `covid_index` — Pandemic trajectory index (2020–2022)
- `dgs30` — 30-year US Treasury yield
- `mobility_index` — Consumer mobility
- `competitor_ishare` — Competitor impression share

---

## Column Schema (39 columns)

- **Date:** `week_id`, `date`, `year`, `quarter`, `week_of_year`, `scenario`
- **Observed (model inputs):** `net_sales_observed`, `spend_*`, `impr_*` (5 channels each)
- **Exogenous inputs:** `promo`, `covid_index`, `dgs30`, `mobility_index`, `competitor_ishare`
- **Ground truth:** `baseline_true`, `exog_effect_true`, `noise_true`, `stc_*_true`, `ltc_*_true`, `brand_stock_*_true`

---

## Reference Document

Full technical write-up, assumptions, and scenario design: `docs/MMM_Synthetic_Data_WriteUp.docx` (gitignored — local only).

---

## Repo Architecture

### Directory Structure

```
ltc_frameworks/
├── pyproject.toml                   # package metadata + pinned dependencies
├── ltc/                             # main installable Python package
│   ├── transforms/                  # stateless lag/adstock primitives (pure functions)
│   │   ├── geometric.py             # geometric_adstock(x, decay)
│   │   ├── weibull.py               # weibull_adstock(x, shape, scale, max_lag)
│   │   ├── almon.py                 # almon_pdl_weights(degree, max_lag)
│   │   ├── koyck.py                 # koyck_transform(y, x, lambda_)
│   │   └── brand_stock.py           # brand_stock_dynamics(spend, delta, build_rate)
│   ├── data/
│   │   ├── loader.py                # load_scenario(path, scenario) → pd.DataFrame
│   │   └── features.py              # build_features(df) → X_obs, X_truth
│   ├── models/
│   │   ├── base.py                  # abstract BaseLTCModel: fit / decompose / get_params
│   │   ├── framework1/              # static adstock regression (F1)
│   │   │   ├── geometric_regression.py   # GeometricAdstockOLS
│   │   │   ├── weibull_regression.py     # WeibullAdstockNLS
│   │   │   ├── almon_regression.py       # AlmonPDL
│   │   │   └── dual_adstock.py           # DualAdstockOLS
│   │   ├── framework2/              # dynamic time-series distributed lag (F2)
│   │   │   ├── koyck_model.py            # KoyckModel
│   │   │   ├── ardl_model.py             # ARDLModel
│   │   │   └── finite_dl_model.py        # FiniteDLModel
│   │   └── framework3/              # state-space / latent brand-stock (F3)
│   │       ├── kalman_dlm.py             # KalmanDLM
│   │       ├── mcmc_latent_stock.py      # MCMCLatentStock (PyMC)
│   │       └── bayesian_sts.py           # BayesianStructuralTS
│   ├── evaluation/
│   │   ├── metrics.py               # recovery_accuracy, mape, ci_coverage
│   │   ├── scorer.py                # score_model(model, df) → dict
│   │   └── benchmark.py             # run_benchmark(models, scenarios) → DataFrame
│   └── visualization/
│       ├── decomposition.py         # plot_contribution_area(df, estimated, truth)
│       ├── brand_stock_plot.py      # plot_stock_evolution(df, estimated, truth)
│       └── benchmark_plot.py        # plot_heatmap(benchmark_df), plot_radar(benchmark_df)
├── experiments/
│   ├── registry.py                  # MODEL_REGISTRY = {"geo_adstock": GeometricAdstockOLS, ...}
│   ├── configs/
│   │   ├── framework1.yaml          # hyperparameter grids for F1 models
│   │   ├── framework2.yaml          # hyperparameter grids for F2 models
│   │   └── framework3.yaml          # hyperparameter grids / priors for F3 models
│   └── run_experiment.py            # CLI: --model, --scenario, --all-scenarios
├── notebooks/                       # per-framework exploration + final benchmark
├── outputs/
│   ├── results/                     # JSON per model×scenario run (gitignored)
│   ├── figures/                     # saved PNG/SVG (gitignored)
│   └── reports/                     # paper-ready summary tables (tracked)
└── docs/                            # gitignored — local write-up only
```

### Model Registry

```python
MODEL_REGISTRY = {
    # Framework 1 — Static Adstock Regression
    "geo_adstock":     GeometricAdstockOLS,
    "weibull_adstock": WeibullAdstockNLS,
    "almon_pdl":       AlmonPDL,
    "dual_adstock":    DualAdstockOLS,
    # Framework 2 — Dynamic Time-Series Distributed Lag
    "koyck":           KoyckModel,
    "ardl":            ARDLModel,
    "finite_dl":       FiniteDLModel,
    # Framework 3 — State-Space / Latent Brand-Stock
    "kalman_dlm":      KalmanDLM,
    "mcmc_stock":      MCMCLatentStock,
    "bsts":            BayesianStructuralTS,
}
```

### Interface Contract (all models)

```python
class BaseLTCModel(ABC):
    def fit(self, df: pd.DataFrame, config: dict) -> "BaseLTCModel":
        """Fit on observed columns only — no ground truth."""

    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with estimated: baseline, stc_{ch}, ltc_{ch} columns."""

    def get_params(self) -> dict:
        """Return fitted parameters as a serialisable dict."""
```

### Experiment Flow

```
run_experiment.py --model geo_adstock --scenario S1
  → loader.py:   load_scenario(data/raw/, "S1")
  → features.py: build_features(df)  [observed cols only]
  → registry:    MODEL_REGISTRY["geo_adstock"]
  → model.fit(df, config)
  → model.decompose(df)
  → scorer.py:   score_model(decomposition, ground_truth)
  → outputs/results/geo_adstock_S1.json
  → outputs/figures/geo_adstock_S1_decomp.png
```

---

## Experimental Results (2026-04-26)

### S1 Baseline Results — 10 Models Evaluated

| Model | Framework | MAPE | Recovery | Status |
|-------|-----------|------|----------|--------|
| **bsts** | F3 | 17.6% | 82.4% | ✓ Best overall |
| **kalman_dlm** | F3 | 18.0% | 82.0% | ✓ State-space dominant |
| **geo_adstock** | F1 | 30.1% | 69.9% | ✓ Acceptable static baseline |
| **mcmc_stock** | F3 | 27.4% | 72.6% | ✓ R-hat<1.05 convergence |
| **finite_dl** | F2 | 49.7% | 50.3% | ✓ Mid-range |
| **koyck** | F2 | 53.6% | 46.4% | ✓ Mid-range |
| **almon_pdl** | F1 | 57.4% | 42.6% | ✓ Weak |
| **weibull_adstock** | F1 | 89.5% | 10.5% | ⚠ Architectural limitation |
| **ardl** | F2 | 316.8% | 0.0% | ✗ Critical failure (S1) |
| **dual_adstock** | F1 | 789.9% | 0.0% | ✗ Sign-flip failure |

**Key Finding:** Framework 3 (state-space) models significantly outperform F1 & F2 on baseline scenario.

### S2 Spend-Pause Results — Frozen S1 Parameters

**Scenario 2:** TV + Video spend = 0 for weeks 104–112; tests latent stock persistence.

| Model | S1 Recovery | S2 Recovery | Δ Recovery | Impact |
|-------|-------------|-------------|------------|--------|
| **geo_adstock** | 69.9% | 83.1% | +13.2pp | ✓ Improves; spend pause reveals LTC |
| **weibull_adstock** | 10.5% | 30.5% | +20.0pp | ✓ Lag shapes finally useful |
| **ardl** | 0.0% | 68.8% | +68.8pp | 🔥 S1 failure → S2 works (prior misspec) |
| **finite_dl** | 50.3% | 54.6% | +4.3pp | ✓ Slight improvement |
| **koyck** | 46.4% | 43.0% | -3.4pp | ≈ Stable |
| **kalman_dlm** | 82.0% | 83.1% | +1.1pp | ✓✓ Excellent stability |
| **bsts** | 82.4% | 81.0% | -1.4pp | ✓✓ Excellent stability |
| **almon_pdl** | 42.6% | 18.7% | -23.9pp | ✗ Polynomial lags fail on discontinuities |
| **mcmc_stock** | 72.6% | 61.4% | -11.1pp | ⚠ Degrades; divergences 8→1 |
| **dual_adstock** | 0.0% | 0.0% | +0.0pp | ✗ Catastrophic both scenarios |

**Critical Insights:**
1. **Scenario as diagnostic:** Spend pause reveals whether models can identify latent structure; ardl S1 failure was prior misspecification, not structural flaw
2. **Framework robustness:** F3 ±1–11pp range; F1/F2 highly scenario-dependent (±4–69pp range)
3. **Structural breaks:** Simple models (geo_adstock) benefit from clean signals; complex models (almon_pdl) fail on discontinuities
4. **MCMC convergence:** Spend pause improves MCMC (fewer divergences) but can over-constrain inference (recovery degradation)

---

## Debugging & Code Improvements (2026-04-26)

### Issues Fixed

1. **Weibull max_lag_override** — per-channel lag windows now enforced (was being ignored)
2. **MCMC chains** — increased 2→4 for convergence diagnostics
3. **MCMC R-hat logging** — all 19 parameters show R-hat<1.05 (excellent convergence)
4. **Stock initialization** — confirmed steady-state formula applied correctly
5. **Joint prior constraint** — added PyMC Potential for build_rate×ltc_coef plausibility

### Configuration Frozen (S1 Optimized)

See `S1 Frozen Parameter Set` in `RUN_LOG.md` for full hyperparameter grids:
- **Framework 1:** Tightened decay grids; max_lag_override (TV/Video 52w, Search 8w, Display 16w, Social 26w)
- **Framework 2:** ardl ltc_degree 2→3; lag shape constraints per model
- **Framework 3:** Logit-normal δ priors; 4 MCMC chains; steady-state stock initialization

---

## Next Steps

1. **Optimize S2-specific parameters** — quantify calibration impact (expect F1/F2 to gain 5–15pp)
2. **Run S3–S5 scenarios** with frozen S1 params — profile full sensitivity landscape
3. **Analyze calibration vs. structure trade-off** — compare S1-frozen vs. S2-optimized results
4. **Paper write-up** — framework comparison, scenario sensitivity, methodological insights

---

## References

- **Run Log:** `RUN_LOG.md` (detailed experiment tracking, config changes, fixes)
- **Analysis:** `S1_vs_S2_ANALYSIS.md` (model-by-model impact breakdown with insights)
- **Data:** `outputs/results/{model}_{scenario}.json` (metrics per model×scenario)
- **Figures:** `outputs/figures/{model}_{scenario}_decomp.png` (LTC decomposition visualizations)
