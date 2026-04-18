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
