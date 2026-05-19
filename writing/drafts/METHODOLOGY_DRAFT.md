# SECTION 3: METHODOLOGY

**Target Length:** 1500–2000 words  
**Structure:** Three subsections (3.1 Synthetic Data Framework, 3.2 Estimation Frameworks, 3.3 Evaluation Metrics)  
**Status:** DRAFT — Ready for evaluation

---

## 3.1 Synthetic Data Framework

Ground-truth LTC is unavailable from observational marketing data. Real-world MMM estimates rely on assumptions about causal relationships that cannot be directly validated; a model can recover aggregate sales but misattribute effects across channels, or correctly identify one channel while failing on another. To evaluate whether frameworks recover true LTC, we constructed a synthetic data-generating process with known ground-truth LTC contributions.

The fundamental equation is:

**Equation 1:**
$$\text{net\_sales}[t] = \text{baseline}[t] + \sum_{\text{ch}} \text{STC}_{\text{ch}}[t] + \sum_{\text{ch}} \text{LTC}_{\text{ch}}[t] + \text{exog}[t] + \varepsilon[t]$$

Sales in week $t$ decompose into five components: a baseline (trend + seasonality), short-term contributions from current-week media, long-term contributions from accumulated brand equity, exogenous effects, and observation noise.

### Baseline Generation

Baseline trends capture sustained economic and seasonal demand patterns:

$$\text{baseline}[t] = \text{piecewise\_linear}[t] + \text{seasonality}[t] + \text{holiday\_lift}[t]$$

The piecewise linear component changes slope at week 104 (2022-02-07) to reflect business environment shifts. Annual seasonality uses a 52-week cycle with 20% amplitude variation. Holiday effects apply uplifts during Thanksgiving, Christmas, and New Year periods. The resulting baseline ranges $10M–$12M per week, matching realistic retail and e-commerce spending patterns.

### Short-Term Contribution (STC)

STC captures immediate sales response to current-week media impressions via geometric adstock decay:

**Equation 2:**
$$\text{adstocked}_{\text{ch}}[t] = \text{impressions}_{\text{ch}}[t] + \lambda_{\text{ch}} \times \text{adstocked}_{\text{ch}}[t-1]$$

$$\text{STC}_{\text{ch}}[t] = \beta_{\text{ch}} \times \text{adstocked}_{\text{ch}}[t]$$

The decay parameter $\lambda_{\text{ch}}$ varies by channel: TV $\lambda = 0.45$, Search $\lambda = 0.10$, Social $\lambda = 0.35$, Display $\lambda = 0.25$, Video $\lambda = 0.40$. These reflect realistic patterns where always-on search channels have minimal carry-over, while brand-building channels (TV, Video) show stronger persistence.

### Long-Term Contribution (LTC)

LTC arises from a latent brand equity stock that accumulates from sustained spending and depreciates over time:

**Equation 3:**
$$\text{stock}_{\text{ch}}[t] = \delta_{\text{ch}} \times \text{stock}_{\text{ch}}[t-1] + \text{build\_rate}_{\text{ch}} \times \sqrt{\text{spend}_{\text{ch}}[t]}$$

**Equation 4:**
$$\text{LTC}_{\text{ch}}[t] = \text{ltc\_coef}_{\text{ch}} \times \text{stock}_{\text{ch}}[t]$$

The stock equation captures how brand equity persists (retention rate $\delta_{\text{ch}}$) and grows with spending (at a decreasing rate via square root). The square root functional form reflects diminishing returns to cumulative spending. LTC coefficient $\text{ltc\_coef}_{\text{ch}}$ translates brand stock into sales contribution. Channel-level retention rates are: TV $\delta = 0.90$, Video $\delta = 0.88$, Social $\delta = 0.82$, Display $\delta = 0.65$, Search $\delta = 0.30$. TV and Video, positioned as brand-building channels, retain 88–90% of brand equity weekly; always-on Search retains only 30%. Build rates are channel-specific: TV 0.60, Search 0.15, Social 0.35, Display 0.10, Video 0.55. LTC coefficients range 0.005–0.015 across channels, scaled to produce realistic aggregate LTC ($1.23M weekly, ~12% of observed sales).

### Exogenous Effects

Exogenous variables capture business environment variation:

$$\text{exog}[t] = \beta_{\text{promo}} \times \text{promo}[t] + \beta_{\text{covid}} \times \text{covid\_index}[t] + \beta_{\text{dgs}} \times \text{dgs30}[t] + \beta_{\text{mobility}} \times \text{mobility\_index}[t] + \beta_{\text{comp}} \times \text{competitor\_ishare}[t]$$

Promotional calendar intensity (0–0.18) captures point-in-time campaigns; COVID-19 index (2020–2022) tracks pandemic phases; 30-year Treasury yield reflects consumer credit conditions; consumer mobility tracks foot traffic; competitor impression share proxies competitive pressure. These effects combine to produce swings of ±$1.5M–±$2.0M in weekly sales, reflecting realistic market volatility.

### Noise

Observation noise $\varepsilon[t] \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = 0.30$ (thousand dollars), reflecting measurement error, unmodeled small effects, and natural sales volatility. This produces a signal-to-noise ratio of approximately 2.5 for LTC signal (true LTC ~$1.23M vs noise std ~$0.30M).

### Five Diagnostic Scenarios

**Scenario 1 — Baseline (S1, Low Collinearity).** All parameters fixed to true DGP values. Spend variation across channels and time is high (TV quarterly bursts, Social consistent, Search daily peaks). This scenario tests framework performance when identification is optimal.

**Scenario 2 — Spend Pause (S2, Structural Break).** TV and Video spend drop to zero weeks 104–112 (eight weeks), then resume at previous levels. This tests whether models correctly identify LTC by observing its persistence when spending pauses. Frameworks that tie identification to spend variation struggle; structural models benefit from the clean separation between accumulation and decay.

**Scenario 3 — High Seasonality (S3, Collinearity).** Annual seasonality amplitude increases from 20% to 40%, creating collinearity between seasonal baseline patterns and spend patterns. Search, always-on but lower amplitude, has higher collinearity with baseline. Static adstock models cannot separate seasonal baseline from LTC structure.

**Scenario 4 — Permanent Spend Reduction (S4, Level Shift).** Weeks 104 onward, all media spend drops to 20% of pre-break levels and remains constant. This tests adaptation to permanent structural shifts (as opposed to temporary pauses). Dynamic models calibrated on high-spend regimes may mispredict under persistent low-spend conditions.

**Scenario 5 — Weak LTC Signal (S5, Low SNR).** All LTC contributions reduced by 50% ($1.23M → $0.62M). This tests identification at the signal-to-noise boundary. With noise fixed at $0.30M, the LTC signal-to-noise ratio drops from 2.5 to 1.2, approaching the identification threshold for many methods.

### Parameter Specification and Fixed-Parameter Design

All 10 models are evaluated with decay parameters (STC $\lambda$, LTC $\delta$) fixed to their true data-generating values. This design choice is intentional: it isolates **structural framework differences** from **calibration effects**. When decay parameters are misspecified, all methods suffer; when parameters are correct, only structural differences remain.

**Rationale:** To answer "Which framework architecture is most robust to scenario variation?", we must rule out parameter misspecification as a confound. If geo_adstock fails in S3 because the static decay cannot adapt to seasonality, that is a structural limitation. If geo_adstock fails because we mis-estimated $\lambda$, that is a calibration artifact. By fixing parameters to truth, we measure structure cleanly.

**Validation:** In Section 7 (Calibration Sensitivity), we re-optimize parameters per scenario and quantify the gap (optimized recovery – frozen recovery). This reveals which frameworks are robust to parameter uncertainty and which are sensitive.

---

## 3.2 Estimation Frameworks

We evaluate three classes of frameworks, each representing a distinct architectural approach to LTC estimation.

### Framework 1: Static Adstock Regression (4 models)

**Core assumption:** LTC and STC both arise from a single adstock transformation of impressions; coefficients are jointly estimated and interpreted as combined effects.

**Models:** Geometric Adstock OLS, Weibull Adstock NLS, Almon Polynomial Distributed Lag (PDL), Dual Adstock (separate STC + LTC decay per channel).

For geometric adstock, we fit:
$$\text{sales}[t] = \alpha + \beta_{\text{STC}} \times \text{adstocked}_{\text{STC}}[t] + \beta_{\text{LTC}} \times \text{adstocked}_{\text{LTC}}[t] + \varepsilon[t]$$

where adstocked impressions use decay rates $\lambda_{\text{STC}}$ and $\lambda_{\text{LTC}}$ (estimated or fixed).

**Assumed vs estimated:** Decay rates $\lambda$ and LTC retention rates $\delta$ fixed to true DGP values. Coefficients $\beta$ estimated via OLS or NLS. This framework cannot jointly optimize decay while constraining it to true values; it assumes practitioners have prior knowledge of decay.

**Structural advantages:** Simple, interpretable, computationally efficient.

**Structural limitations:** Single decay per channel cannot resolve STC (fast) from LTC (slow) under collinearity. When baseline and seasonality confound spend patterns (S3), static adstock fails. Cannot adapt to structural breaks (S4) that shift spend regimes.

### Framework 2: Dynamic Time-Series Distributed Lag (3 models)

**Core assumption:** LTC is captured through dynamic lag structures that decompose effects into STC (short lags) and LTC (long lags); autoregressive terms capture evolving dynamics.

**Models:** Koyck Model (AR + geometric lag), ARDL (Autoregressive Distributed Lag), Finite Distributed Lag with Weibull shape.

For ARDL(p,q), we fit:
$$\text{sales}[t] = \alpha + \sum_{i=1}^{p} \rho_i \text{sales}[t-i] + \sum_{j=0}^{q} \beta_j \text{impressions}_{\text{ch}}[t-j] + \varepsilon[t]$$

**Assumed vs estimated:** STC decay $\lambda$ and LTC retention $\delta$ can be estimated implicitly through lag weights. Autoregressive coefficients $\rho_i$ estimated. This framework adapts decay to data more flexibly than F1.

**Structural advantages:** Lag flexibility allows models to detect when spend patterns shift (S2, S4). AR terms naturally capture momentum and carryover. More adaptive than static models.

**Structural limitations:** AR structure assumes sales have intrinsic momentum independent of media. Under high collinearity (S3), AR terms overfit and invade the LTC lag space. Long distributed lags (52+ weeks) lead to parameter proliferation and multicollinearity. Coefficient signs can flip under structural breaks (S4), inverting channel rankings.

### Framework 3: State-Space / Latent Brand-Stock (3 models)

**Core assumption:** Latent brand stock explicitly models the accumulation and decay mechanism; level, trend, and seasonal components are separately modeled and jointly estimated via Kalman filter or Bayesian posterior.

**Models:** Kalman Dynamic Linear Model with fixed decay, Bayesian Structural Time-Series (BSTS with seasonal component), MCMC-based Latent Stock Model.

For latent stock models:
$$\text{stock}_{\text{ch}}[t] = \delta_{\text{ch}} \times \text{stock}_{\text{ch}}[t-1] + \text{build\_rate}_{\text{ch}} \times \sqrt{\text{spend}_{\text{ch}}[t]}$$
$$\text{sales}[t] = \text{level}[t] + \text{trend}[t] + \text{seasonal}[t] + \sum_{\text{ch}} \text{ltc\_coef}_{\text{ch}} \times \text{stock}_{\text{ch}}[t] + \text{noise}[t]$$

where latent level, trend, and seasonal are evolved with unknown variances (estimated or fixed).

**Assumed vs estimated:** Decay rates $\delta$ fixed to true DGP (Kalman, BSTS) or estimated via posterior (MCMC). Level/trend/seasonal variance parameters estimated. This framework directly encodes the true DGP structure, making it closest to ground truth.

**Structural advantages:** Explicit stock dynamics robustly identify LTC even under collinearity (S3) because baseline and stock are separate. Naturally adapts to level shifts (S4) via trend component. MCMC provides posterior uncertainty, enabling Bayesian flexibility.

**Structural limitations:** Kalman filter and BSTS require specification of state variances; misspecification degrades performance. Fixed decay parameters inflexible to regime changes (though less problematic than static models because structure is correct). MCMC computationally expensive and requires prior specification.

---

## 3.3 Evaluation Metrics

### LTC Recovery Accuracy

**Equation 5:**
$$\text{LTC\_MAPE} = \text{mean}\left(\frac{|\text{recovered\_ltc} - \text{ltc\_true}|}{\text{ltc\_true}}\right) \times 100$$

MAPE measures percentage error in LTC recovery per week. For each week $t$, we compare recovered LTC (from model decomposition) to true LTC (from data generation). Taking the mean of weekly ratios gives the overall accuracy.

**Equation 6:**
$$\text{Recovery} = (1 - \text{LTC\_MAPE}/100) \times 100$$

Recovery is the inverse: 100% recovery means zero error, 0% recovery means complete failure. An LTC_MAPE of 30% corresponds to 70% recovery.

### Pause-Window Robustness Ratio

**Equation 7:**
$$\text{Robustness\_ratio} = \frac{\text{pause\_window\_MAPE}}{\text{full\_series\_MAPE}}$$

In scenarios with structural breaks (S2, S4), we compute MAPE separately for weeks 100–120 (pause or break window) and for the full 261-week series. The ratio indicates error concentration. A ratio near 1.0 means error is evenly distributed (robust). A ratio >1.3 means error concentrates in the break region (fragile).

### Channel-Level Attribution and Budget Error

**Equation 8:**
$$\text{budget\_error}_{\text{ch}} = \text{recovered\_share}_{\text{ch}} - \text{true\_share}_{\text{ch}}$$

where 
$$\text{share}_{\text{ch}} = \frac{\text{LTC}_{\text{ch}} + \text{STC}_{\text{ch}}}{\sum_{\text{all ch}} (\text{LTC} + \text{STC})}$$

Practitioners allocate budgets based on channel ROI estimates. If a model recovers incorrect channel shares, budget recommendations will be wrong even if aggregate recovery is high. We report per-channel recovery accuracy and aggregate budget error to validate that channel decomposition is correct.

### Why Channel-Level Validation Is Critical

Standard MMM benchmarking reports aggregate LTC recovery (e.g., "the model achieves 82% accuracy"). However, this aggregate masks channel-level failures. In this study's S2 scenario, the ARDL model recovers 68.8% aggregate LTC but 0% recovery for every individual channel. The model captured the total magnitude but misattributed effects completely: TV should dominate (δ=0.90), yet ARDL assigned zero LTC to TV. Practitioners using this model to allocate budgets would systematically over-invest in low-LTC channels and under-invest in high-LTC channels, compounding the misallocation at every budget cycle.

Channel-level validation is therefore mandatory. We report recovery per channel and flag models that achieve good aggregate through offsetting errors.

---

## Replicability

All results are reproducible using the seed value 42 in the mmm_synthetic_generator.py script, which resides in the root of this repository. The script generates five scenario datasets (S1–S5) as CSV files in the `data/` directory with 261 weeks of observations, 39 columns per CSV, and full ground-truth labels (baseline_true, stc_*_true, ltc_*_true, brand_stock_*_true). All model code, experimental configurations, and results JSON files will be made available upon acceptance for publication, satisfying the transparency requirements of Journal of Marketing Research, Marketing Science, and International Journal of Research in Marketing.

---

## Word Count Check
Target: 1500–2000 words
[To be counted upon final draft completion]
