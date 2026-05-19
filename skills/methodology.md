# sections/methodology.md
## Methodology

Target length: 1500–2000 words. Three subsections.
This is the most technically demanding section — write it first.

---

## Three Subsections

### 3.1 Synthetic Data Framework

**What to cover:**
- Why synthetic data: ground truth is unavailable in real MMM data. State this clearly.
- The fundamental equation — display it with equation number
- The five components: baseline, STC, LTC, exogenous effects, noise
- Baseline generation: piecewise trend, seasonality, holidays
- LTC generation mechanism: latent stock model (equations with numbers)
- The five scenarios: one paragraph each
- Why parameters were fixed to true DGP values (isolates structural from calibration differences)

**Required equations (numbered):**
```
Eq 1: net_sales[t] = baseline[t] + Σ STC_ch[t] + Σ LTC_ch[t] + exog[t] + ε[t]
Eq 2: adstocked[t] = impressions[t] + λ × adstocked[t-1]
Eq 3: stock[t] = δ × stock[t-1] + build_rate × √spend[t]
Eq 4: LTC[t] = ltc_coef × stock[t]
```

**Scenario table** (required — see @visuals.md for format):
| Scenario | Key Property | Structural Challenge | Framework Tested |
|---|---|---|---|
| S1 | Low collinearity | Performance ceiling | All methods |
| S2 | Spend pause (wks 104–112) | LTC persistence after spend stops | F1/F3 separation |
| S3 | High seasonal collinearity | Spend-baseline confound | F1 collapse |
| S4 | Permanent spend reduction | Structural break adaptation | F3 advantage |
| S5 | Weak LTC signal (×0.35) | False discovery control | Prior sensitivity |

**Parameter table** (required):
List all channel-level parameters (true STC decay, true LTC delta, build rate, LTC coef)

### 3.2 Estimation Frameworks

**What to cover:**
One subsection per framework. For each:
- Core structural assumption (one sentence)
- Methods included
- What parameters are assumed vs estimated
- Why this framework would succeed or fail structurally

**Assumed vs estimated table** (required):
State which parameters were fixed to true DGP values and which were estimated.
Justify the fixed-parameter design explicitly.

**Key sentence to include:**
"To isolate structural framework differences from calibration effects, all decay
parameters were fixed to their true data-generating values. Under this design,
differences in recovery accuracy across scenarios reflect structural framework
properties, not parameter mis-specification."

### 3.3 Evaluation Metrics

**What to cover:**
Define every metric with an equation. Do not assume the reader knows MAPE.

**Required metric definitions:**
```
Eq 5: LTC_MAPE = mean(|recovered_ltc - ltc_true| / ltc_true) × 100

Eq 6: Recovery = (1 - LTC_MAPE/100) × 100

Eq 7: Robustness_ratio = pause_window_MAPE / full_series_MAPE

Eq 8: budget_error[ch] = recovered_share[ch] - true_share[ch]
      where share[ch] = (LTC[ch] + STC[ch]) / Σ(LTC + STC)
```

**Justify channel-level validation:**
Explain why aggregate metrics alone are insufficient.
Reference the ARDL S2 finding (68.8% aggregate, 0% channel) as motivation
for requiring channel-level attribution as a validation criterion.

---

## Replicability Note (Required)

End the section with one paragraph on replicability.
State the random seed (42), the generator script name, and where data/code
will be made available. This satisfies JMR, Marketing Science, and IJRM requirements.

---

## Do Not Include
- Results — those go in Section 4
- Literature citations for methods beyond the founding papers
- Detailed derivations — move to appendix if needed
