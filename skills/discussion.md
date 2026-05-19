# sections/discussion.md
## Discussion

Target length: 800–1200 words. Interpret findings — do not repeat them.

---

## Three Things Discussion Must Do

1. Connect findings to mechanisms in the literature
2. Pre-empt the three reviewer objections the paper will receive
3. State the practical implications clearly

---

## Structure

### 5.1 The Robustness Spectrum Taxonomy
Introduce the three-tier taxonomy formally:
- Tier 1 Structural (ratio 0.95–1.10): BSTS, MCMC — scenario-invariant
- Tier 2 Identification-Dependent (ratio 1.10–1.35): Kalman, finite_dl — sensitive to spend variation
- Tier 3 Fragile (ratio > 1.35): geo, weibull, almon — error concentrates at structural breaks

Connect to literature: why does spend-dependence create fragility?
Cite Hanssens et al. on identification in time-series models.

### 5.2 The Channel Attribution Problem
State the principle: aggregate accuracy is necessary but insufficient.
Connect to the ARDL and Koyck findings.
Generalise: this is not unique to MMM — any multivariate decomposition model
can achieve aggregate fit through offsetting channel-level errors.
Propose channel-level validation as a mandatory criterion.

### 5.3 MCMC as Production Standard
Summarise the evidence: highest average recovery, correct channel ranking,
Video LTC identification, weak signal recovery with priors.
State the limitation honestly: computational cost, prior sensitivity.
Propose when to use MCMC vs BSTS vs simpler methods.

### 5.4 Pre-empting Reviewer Objections

**Objection 1: "Results depend on synthetic data assumptions"**
Response: Scenarios are calibrated to real-world parameter ranges.
Multiple scenarios test structural properties, not a single DGP.
Known ground truth is the only way to measure exact recovery — not a weakness.

**Objection 2: "MCMC is too slow for production use"**
Response: Attribution error has a cost. Mis-attributing $X of LTC to wrong channels
over N planning cycles compounds. Compute cost of MCMC (~60s per run) is small
relative to budget allocation errors shown in Table 6.

**Objection 3: "Results may not generalise to real data"**
Response: Parameter ranges (δ 0.65–0.90, noise $0.15M–$0.30M) are calibrated to
published estimates. Structural failures (collinearity, spend pauses, mix shifts)
are real business conditions every practitioner faces. Future work should validate
on real data using the framework as a prior.

---

## What Not to Do
- Do not present new results here
- Do not repeat tables from Results section
- Do not hedge findings that are directly supported by ground truth comparison
- Do not end without connecting back to the paper's central claim
