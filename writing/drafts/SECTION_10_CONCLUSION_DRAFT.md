# Section 10: Conclusion and Practitioner Recommendations

## 10.1 Conclusion

**The Problem**

Marketing mix modelers cannot estimate long-term media contributions reliably. Budget decisions rest on short-term elasticities, missing sustained brand effects that may generate 10–15% of total sales.

**What This Paper Found**

Three contributions:

1. **Framework architecture matters more than calibration.** State-space methods recover 78.4% vs static methods 22.4%. Tuning improves F3 by 2–3pp, F1 by <2pp. The 56pp gap is architectural.

2. **Robustness to scenario variation predicts reliability.** BSTS maintains 1.02× pause-window ratio; geo_adstock and almon_pdl degrade to >1.35× on discontinuities. Tier 1 (robust), Tier 2 (tuning-responsive), Tier 3 (unreliable) taxonomy guides method selection.

3. **Channel-level validation is mandatory.** ARDL achieves 68.8% aggregate but 0% Video recovery. Any decomposition can hide offsetting errors. Validate per-channel recovery under structural breaks before deployment.

**Boundary Conditions**

These findings hold when (1) media signal is sufficient (long-term effects >5% of baseline), (2) spend patterns vary meaningfully (e.g., all channels active, some periods with pauses), and (3) time series length exceeds 100 weeks. Section 5 (S5 weak-signal scenario) shows that framework hierarchy breaks down under low signal; MCMC recovers 88.5% with scenario-specific priors, but fixed-parameter methods collapse to 0%. Real-world signal strength should be validated before method selection.

**Limitations and Future Directions**

Four limitations: (1) synthetic data; real-world complexity (competitive response, interactions) may differ; (2) no cross-channel synergies in data-generating process; (3) MCMC cost (~60s) limits high-frequency use; (4) S5 Social/TV reversal in MCMC is unexplained.

Future research: (1) validate on real branded data using recovery hierarchy as prior; (2) add brand search as observation equation in F3; (3) test diagnostic spend pauses for LTC identification improvement; (4) model cross-channel stock interactions.

---

## 10.2 Practitioner Recommendations

**When to Use Each Framework**

| Condition | Recommended Method | Why |
|-----------|-------------------|-----|
| Strong signal, stability priority | BSTS | Lowest variance (pause ratio 1.02×) across scenarios |
| Strong signal, accuracy priority | MCMC | Highest recovery (78.4% average) with correct channel ranking |
| Weak signal (LTC <5% sales) | MCMC + scenario priors | Only method with recovery in weak-signal scenario (88.5% S5) |
| Structural break suspected | MCMC or BSTS | F1/F2 fail on discontinuities; pause ratios >1.35 |
| Budget constraints, quick results | Kalman DLM | Solid S1/S2 performance (82% avg), sub-second runtime |
| **Do not use** | Dual adstock, ARDL | Confirmed sign-flip risks (−19.8% S4 recovery) and reversals (0%→68.8%→−19.8% across scenarios) |

**Three Critical Warnings**

1. **Aggregate metrics hide channel errors.** A model achieving 70% aggregate recovery can misallocate 50% of budget if channel-level attribution is wrong. Validate per-channel recovery in at least one stress scenario (spend pause, seasonal contrast, mix shift) before deployment.

2. **No model is universally robust.** ARDL works in S2 (68.8% recovery) but fails catastrophically in S4 (−19.8%). Freeze parameters after tuning only if you can defend your scenario assumptions. If market conditions shift materially, revalidate.

3. **Spend pauses improve identification.** If long-term effects are uncertain, planned zero-spend periods (even brief media pauses) reveal latent stock decay rates with minimal cost. Consider using diagnostic pauses in real planning to improve future attribution models.
