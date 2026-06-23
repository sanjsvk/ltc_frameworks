# Section 3.1: Synthetic Data Framework - All Formulas

## Equation 1: Net Sales Model (Five-Component Decomposition)

$$\text{Net Sales}[t] = \text{Baseline}[t] + \sum_{c} \text{STC}_c[t] + \sum_{c} \text{LTC}_c[t] + \text{Exog}[t] + \epsilon[t]$$

**Where:**
- $\text{Net Sales}[t]$ = Observed weekly net sales (in $M) at time $t$
- $\text{Baseline}[t]$ = Piecewise linear trend + annual seasonality + holiday uplifts (~$10M–$12M per week)
- $\text{STC}_c[t]$ = Short-term contribution of channel $c$ at time $t$ (geometric adstock on impressions)
- $\text{LTC}_c[t]$ = Long-term contribution of channel $c$ at time $t$ (latent brand stock accumulation)
- $\text{Exog}[t]$ = Exogenous effects (promotional intensity, COVID-19, Treasury yield, mobility, competitor share)
- $\epsilon[t]$ = Gaussian noise with constant variance (signal-to-noise ratio matching real MMM data)
- $c$ = Channel index (TV, Paid Search, Paid Social, Display, Video)
- $t$ = Time index (weekly, 261 weeks total)

**Interpretation:** Net sales decompose into five independent additive components: structural baseline, two media effects (STC immediate, LTC persistent), external factors, and measurement error.

---

## Equation 2: Short-Term Contribution via Geometric Adstock

$$\text{Adstocked}_c[t] = \text{Impr}_c[t] + \lambda_c \times \text{Adstocked}_c[t-1]$$

**Where:**
- $\text{Adstocked}_c[t]$ = Adstocked (transformed) impressions for channel $c$ at time $t$
- $\text{Impr}_c[t]$ = Raw impressions for channel $c$ at time $t$
- $\lambda_c$ = Channel-specific decay rate (geometric adstock parameter, 0.30–0.90 by channel)
- $\text{Adstocked}_c[t-1]$ = Lagged adstocked impressions (recursion for carryover)

**Channel-Specific Decay Rates ($\lambda_c$ values):**
- TV: $\lambda_{TV} = 0.90$ (high persistence, brand-building channel)
- Video: $\lambda_{Video} = 0.88$ (high persistence, brand-building channel)
- Paid Social: $\lambda_{Social} = 0.82$ (medium persistence)
- Display: $\lambda_{Display} = 0.65$ (medium-low persistence)
- Paid Search: $\lambda_{Search} = 0.30$ (low persistence, direct response channel)

**STC Coefficient:** Channel-level elasticity $\alpha_c$ multiplied by adstocked impressions:
$$\text{STC}_c[t] = \alpha_c \times \text{Adstocked}_c[t]$$

Total STC across all channels ≈ $1.58M per week (~15% of observed sales).

**Interpretation:** Short-term effects persist via recursive adstocking; higher $\lambda_c$ values mean longer memory. This equation captures immediate brand awareness boost from exposure.

---

## Equation 3: Long-Term Contribution via Latent Brand Stock Accumulation

$$\text{Stock}_{c}[t] = \delta_c \times \text{Stock}_{c}[t-1] + \beta_c \times \sqrt{\text{Spend}_{c}[t]}$$

**Where:**
- $\text{Stock}_{c}[t]$ = Latent brand stock for channel $c$ at time $t$ (unobserved state variable)
- $\delta_c$ = Stock retention rate (decay coefficient, 0.30–0.90 by channel)
- $\text{Stock}_{c}[t-1]$ = Lagged brand stock (carryover from prior periods)
- $\beta_c$ = Stock build rate (how quickly spending accumulates brand equity, channel-specific)
- $\text{Spend}_{c}[t]$ = Media spend for channel $c$ at time $t$ (in $M)
- $\sqrt{\text{Spend}_{c}[t]}$ = Square-root transformation (diminishing returns to additional spending)

**Stock Retention Rates ($\delta_c$ values):**
- TV: $\delta_{TV} = 0.90$ (90% of prior period stock retained, longest memory)
- Video: $\delta_{Video} = 0.88$ (88% retention)
- Paid Social: $\delta_{Social} = 0.82$ (82% retention)
- Display: $\delta_{Display} = 0.65$ (65% retention)
- Paid Search: $\delta_{Search} = 0.30$ (30% retention, quickest decay)

**Stock Initialization:** Steady-state formula applied at $t=0$:
$$\text{Stock}_{c}[0] = \frac{\beta_c \times \sqrt{\text{Spend}_{c,\text{avg}}}}{1 - \delta_c}$$

where $\text{Spend}_{c,\text{avg}}$ is the average historical spend for channel $c$.

**Interpretation:** Brand stock acts as latent equity built from cumulative spending, decaying over time independent of current spend. This equation captures brand-building effects that persist after advertising stops.

---

## Equation 4: Long-Term Contribution (LTC) as Linear Function of Brand Stock

$$\text{LTC}_{c}[t] = \gamma_c \times \text{Stock}_{c}[t]$$

**Where:**
- $\text{LTC}_{c}[t]$ = Long-term contribution of channel $c$ at time $t$ (sales impact, in $M)
- $\gamma_c$ = LTC coefficient (converts latent brand stock to sales contribution, channel-specific)
- $\text{Stock}_{c}[t]$ = Latent brand stock from Equation 3 (at time $t$)

**LTC Magnitude (Ground Truth):**
- Total LTC across all channels ≈ $1.23M per week (~12% of observed sales)
- TV LTC ≈ 35% of total LTC ($0.43M per week)
- Video LTC ≈ 42% of total LTC ($0.52M per week)
- Paid Social LTC ≈ 13% of total LTC ($0.16M per week)
- Display LTC ≈ 5% of total LTC ($0.06M per week)
- Paid Search LTC ≈ 5% of total LTC ($0.06M per week)

**Interpretation:** LTC is the direct sales impact of accumulated brand stock. Unlike STC (which depends on current impressions), LTC depends only on accumulated stock and persists even when current spending stops.

---

## Equation 5: Mean Absolute Percentage Error (MAPE) – Full Time Series

$$\text{MAPE}_{\text{LTC}} = \text{mean}\left(\frac{|\text{LTC}_{\text{recovered}}[t] - \text{LTC}_{\text{true}}[t]|}{\text{LTC}_{\text{true}}[t]}\right) \times 100$$

**Where:**
- $\text{MAPE}_{\text{LTC}}$ = Mean absolute percentage error on LTC estimates (0–100%)
- $\text{LTC}_{\text{recovered}}[t]$ = Estimated LTC by the fitted model at time $t$
- $\text{LTC}_{\text{true}}[t]$ = Ground-truth LTC from data-generating process at time $t$
- $\text{mean}(\cdot)$ = Average across all $t$ (261 weeks)

**Bounds:**
- MAPE = 0%: Perfect recovery (estimated LTC matches true LTC exactly)
- MAPE = 100%: Recovered estimates are on average equal in magnitude but opposite in sign to true values
- MAPE > 100%: Recovered estimates are systematically much worse than zero-baseline

**Interpretation:** MAPE measures average absolute error in percentage terms. It is scale-independent, making it comparable across scenarios with different LTC magnitudes.

---

## Equation 6: Recovery Accuracy (Complement of MAPE)

$$\text{Recovery} = \left(1 - \frac{\text{MAPE}_{\text{LTC}}}{100}\right) \times 100$$

**Where:**
- $\text{Recovery}$ = Recovery accuracy (0–100%)
- $\text{MAPE}_{\text{LTC}}$ = Mean absolute percentage error from Equation 5

**Interpretation Examples:**
- Recovery = 80% ⟺ MAPE = 20% (method recovers 80% of true LTC on average)
- Recovery = 0% ⟺ MAPE ≥ 100% (estimates are worthless or worse than zero-baseline)
- Recovery = 100% ⟺ MAPE = 0% (perfect recovery, estimated = true LTC exactly)

**Relationship:** Recovery is used throughout the paper as the primary metric for framework comparison. Higher recovery indicates better identification of true long-term effects.

---

## Equation 7: Pause-Window Robustness Ratio

$$\text{Robustness Ratio} = \frac{\text{MAPE}_{\text{pause}}}{\text{MAPE}_{\text{full}}}$$

**Where:**
- $\text{Robustness Ratio}$ = Dimensionless ratio (typically 0.7–1.5×)
- $\text{MAPE}_{\text{pause}}$ = MAPE computed on pause-window subset (weeks 100–120, 21-week window)
- $\text{MAPE}_{\text{full}}$ = MAPE computed on full time series (261 weeks)

**Interpretation:**
- Ratio ≈ 1.0: Error is constant across time; method is structurally robust
- Ratio < 1.0: Error decreases during pause window (method improves on discontinuity, often due to overfitting correction)
- Ratio > 1.0: Error increases during pause window (method is fragile to structural breaks)
- Ratio > 1.35: Severe fragility; architecture cannot adapt to spend discontinuity

**Robustness Tiers (from Equation 7):**
- **Tier 1** (Architecturally Robust): Ratio < 1.10× (BSTS, Kalman DLM in baseline)
- **Tier 2** (Identification-Sensitive): Ratio 1.10–1.35× (MCMC, almon_pdl, ARDL, dual_adstock)
- **Tier 3** (Data-Dependent): Ratio > 1.35× (geo_adstock, weibull_adstock)

**Diagnostic Scenario:** Applies to S2 (spend pause) and S4 (structural break). Reveals whether methods can identify latent stock persistence independent of current spend.

---

## Equation 8: Channel-Level Budget Allocation Error

$$\text{Budget Error}_{c} = \frac{\text{Contribution}_{c,\text{recovered}}}{\sum \text{Recovered}} - \frac{\text{Contribution}_{c,\text{true}}}{\sum \text{True}}$$

**Where:**
- $\text{Budget Error}_{c}$ = Budget allocation error for channel $c$ (−1.0 to +1.0, dimensionless)
- $\text{Contribution}_{c,\text{recovered}}$ = Estimated total LTC for channel $c$ (summed across all weeks)
- $\sum \text{Recovered}$ = Total estimated LTC across all channels and weeks
- $\text{Contribution}_{c,\text{true}}$ = Ground-truth total LTC for channel $c$
- $\sum \text{True}$ = Total ground-truth LTC across all channels and weeks

**Interpretation:**
- Budget Error = 0: Channel allocation is perfect (estimated %-share matches true %-share)
- Budget Error > 0: Channel is overestimated (model allocates too much LTC to this channel)
- Budget Error < 0: Channel is underestimated (model allocates too little LTC to this channel)
- |Budget Error| > 0.05: Significant allocation error (>5 percentage-point error in budget share)

**Critical Finding:** Aggregate recovery can mask channel-level misattribution. ARDL achieves 68.8% aggregate recovery in S2 but returns 0% for Video (Budget Error = −0.30 for Video LTC channel). Practitioners validating only aggregate metrics will accept models that misallocate budget across channels.

**Validation Rule:** Before deploying any framework, validate that all channels satisfy |Budget Error| < 0.10 across at least two scenarios (e.g., baseline + spend pause).

---

## Summary

**Total Equations in Section 3.1:** 8

**Equation Classes:**
1. **Data-Generating Process** (Eq 1-4): Define ground truth LTC via adstock STC and latent brand stock
2. **Evaluation Metrics** (Eq 5-8): Define how model performance is measured

**All Variables Defined:** Yes
- Time index: $t$ (weeks)
- Channel index: $c$ (TV, Paid Search, Paid Social, Display, Video)
- All coefficients and parameters include channel-specific values or ranges
- All functional forms (adstock, stock dynamics, metrics) explicitly specified

**Ready for Republishing:** Yes
- All formulas converted to proper LaTeX markdown ($$...$$ format)
- All variables and parameters defined with numerical values where ground truth is known
- All equations numbered and cross-referenced
- Format compatible with markdown viewers, Word equation editors, and LaTeX renderers

**Equations 1-4 Relationship:**
- Eq 1 (Net Sales) = Baseline + STC + LTC + Exog + Noise
- Eq 2 (Adstocked Impressions) → feeds into STC calculation
- Eq 3 (Brand Stock Dynamics) → latent state equation
- Eq 4 (LTC Conversion) → observation equation, links stock to sales

**Equations 5-8 Relationship:**
- Eq 5-6 (MAPE and Recovery) → aggregate model evaluation
- Eq 7 (Robustness Ratio) → scenario sensitivity evaluation
- Eq 8 (Budget Error) → channel-level validation

**Ground Truth Anchor:** All five diagnostic scenarios (S1–S5) use the same data-generating process (Eqs 1-4) with known parameter values. Model recovery accuracy (Eq 6) measures how well each framework identifies the true LTC dynamics encoded in Eqs 3-4.
