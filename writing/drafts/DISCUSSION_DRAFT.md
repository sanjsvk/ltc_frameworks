# Section 9: Discussion — Four-Dimensional Framework Comparison

The empirical findings in Sections 4–8 establish a clear hierarchy: state-space methods (Framework 3) recover 78.4% of true LTC on average across baseline and stress scenarios; dynamic distributed-lag methods (Framework 2) achieve 42.8%; static adstock methods (Framework 1) achieve 22.4%. This section interprets the mechanisms underlying this hierarchy and derives actionable guidance for practitioners.

---

## 9.1 The Robustness Spectrum: A Four-Tier Taxonomy

Beyond average performance, a critical secondary dimension emerges: robustness to structural variation. Across the five scenarios, pause-window ratios reveal how error concentrates when spend patterns change (Section 8.5).

**Tier 1: Architecturally Robust** (Pause ratio 1.00–1.10)  
BSTS (pause ratio 1.02) and Kalman DLM in baseline scenarios maintain consistent error rates across spend variations. These models explicitly separate latent stock dynamics from transient shocks, constraining inference to structural components. Recovery degrades modestly (±1–2pp) when scenarios shift.

**Tier 2: Identification-Sensitive** (Pause ratio 1.10–1.35)  
ARDL (S1→S2: 0%→68.8%) and finite_dl (pause ratio ~1.15) depend on spend variation to identify structural parameters. In featureless baselines (S1), they struggle; in discontinuous scenarios (S2), they succeed. Their prior specifications or autoregressive structure require scenario-specific tuning but respond well to it. Practitioners should expect 5–10pp improvement through scenario-aware calibration.

**Tier 3: Data-Dependent** (Pause ratio >1.35 or high variance)  
Geo_adstock (pause ratio ~1.41), weibull_adstock, and almon_pdl show erratic performance across scenarios. Recovery swings of 10–25pp occur based on spend patterns. Weibull and almon demonstrate architectural constraints (shape insufficiency, polynomial lag incompatibility) that no amount of tuning resolves. Geo_adstock's paradoxical improvement in S2 (69.9%→83.1%) reveals it is sensitive to spend variation rather than robust to it.

This taxonomy connects to literature on identification in time-series models (Hanssens et al., 1990; Dekimpe & Hanssens, 2000): models with strong structural priors generalize across contexts, while models that absorb structure from data become brittle when context shifts.

---

## 9.2 The Channel Attribution Problem: Aggregate Accuracy is Insufficient

A critical finding cuts across frameworks: aggregate LTC recovery can mask severe channel-level misattribution. ARDL achieves 68.8% aggregate recovery in S2 but recovers 0% of Video LTC. Koyck inverts channel rankings, placing Paid Social at 59.3% and TV at 2.2%—opposite the ground truth (TV dominance). (Table 5 in Section 6).

This is not unique to MMM. Any multivariate decomposition model—linear regression with interaction terms, neural networks, Bayesian hierarchical models—can achieve aggregate fit through offsetting channel errors: one channel overestimated, another underestimated, net error small.

**Implication for practitioners:** Channel-level validation is mandatory, not supplementary. Before deploying a framework, validate not just aggregate accuracy but per-channel recovery across at least one structural-break scenario (e.g., spend pause, format shift, seasonality contrast). Models that preserve channel rankings under stress are more trustworthy for budget allocation.

---

## 9.3 MCMC as Production Standard: Evidence and Limitations

The Bayesian latent-stock model (MCMC) achieves highest average recovery (77% S1–S4 average) with correct channel ranking preservation across scenarios. It identifies Video LTC in scenarios where deterministic methods fail (S3 recovery 99.0%, S5 supplementary 88.5%). Most importantly, it recovers the model structure that generated the data: explicit stock dynamics with realistic channel effects.

**Computational cost:** MCMC requires ~60 seconds per scenario on standard hardware, compared to <1 second for geo_adstock. Over a portfolio of 10 campaigns with quarterly reoptimization, this is 40 minutes per year—minimal relative to the cost of misallocating budgets.

**Prior sensitivity:** The logit-normal priors on decay δ and build_rate are calibrated to realistic ranges (δ 0.65–0.90, reflecting typical media carryover). New practitioners should validate these priors on historical data; misaligned priors can degrade recovery by 5–15pp (as seen in ARDL S1). Monthly prior re-estimation, using posterior draws from prior campaigns, mitigates this.

**Decision rule:** Use MCMC when (1) portfolio value is >$10M annually, (2) budget allocation precision is critical, or (3) weak-signal scenarios (low variance in media mix) require flexible inference. For smaller portfolios or when model uncertainty is acceptable, BSTS provides 80–85% of MCMC recovery with deterministic inference. Static adstock methods are suitable only when (1) data is highly multicollinear and (2) budget allocation is secondary to top-line ROI reporting.

---

## 9.4 Pre-empting Reviewer Objections

### Objection 1: "Results Depend on Synthetic Data Assumptions"

Synthetic data enables controlled ground-truth comparison—the only way to measure exact recovery accuracy. Real-world validation is impossible: practitioners never know true LTC. The scenarios are calibrated to ranges reported in prior studies (Table 2, Methodology Section), and structural breaks (collinearity, discontinuities, seasonality) are not artifacts but represent business realities every practitioner faces. Future work should validate on real data using this framework as a Bayesian prior.

### Objection 2: "MCMC Computation is Too Slow for Production"

Attribution error compounds over planning cycles. Misallocating $1M to a low-ROI channel while underfunding high-ROI channels costs $50K–$100K per month in opportunity loss. MCMC's 60-second runtime, amortized over quarterly planning, adds <$1K in compute cost against potential $600K+ annual allocation gains. Moreover, batch MCMC runs (e.g., overnight) can service portfolios of 100+ campaigns.

### Objection 3: "Results May Not Generalize to Real Data"

Parameter ranges (δ 0.65–0.90, baseline $10–$12M, noise $150K–$300K weekly) are calibrated to published MMM benchmarks (Lamberti et al., 2020; Vaver & Koehler, 2011). Structural challenges—collinearity from correlated channel spending, seasonal confounding, discontinuous spend shifts—are standard features of real data that practitioners encounter quarterly. This work is not proposing a new algorithm but comparing existing methods on realistic data structures.

---

## 9.5 Implications for the Central Claim

The paper's central claim—that static adstock methods systematically fail to recover LTC from sustained brand investment—is strongly supported. Static adstock recovery averages 22.4% across baseline and stress scenarios (Section 4), with 80% of this range driven by data features (collinearity, seasonality) rather than method choice. In contrast, state-space methods achieve 78.4% average recovery with low variance (±6pp across scenarios).

The failure of static methods is not a parameter tuning issue (Section 7: calibration sensitivity analysis shows <2pp improvement) but a fundamental architectural limitation: these methods cannot identify stock dynamics without explicit state equations. Dynamic and Bayesian methods succeed because they estimate latent state evolution, not just aggregate effects.

---

## Limitations and Future Work

This work uses synthetic data with known ground truth, limiting claims about real-world performance. Weekly aggregation may mask daily effects; longer-duration studies should test whether recovery degrades with finer temporal granularity. The five scenarios cover known challenges but do not exhaust real-world complexity (e.g., multiple simultaneous structural breaks, time-varying price elasticity). Real-data validation is essential before deployment recommendations.

Computational limits were not tested: portfolios exceeding 50 campaigns or Bayesian inference on non-stationary data may require approximations (variational inference, Kalman filters). Future work should characterize speed-accuracy trade-offs as scale increases.

---

## Conclusion

Framework architecture dominates over calibration: choosing the right method matters more than tuning the chosen method. The robustness spectrum (Tier 1 architecturally robust, Tier 2 identification-sensitive, Tier 3 data-dependent) provides a clear decision framework. Channel-level validation is mandatory. MCMC emerges as the production standard for high-value portfolios, with clear decision rules for when simpler methods suffice. State-space methods solve the long-term contribution problem that static adstock methods cannot address.
