# Section 2: Introduction

## 2.1 The Business Problem

Chief marketing officers allocate budgets of billions of dollars across media channels using marketing mix models (MMMs) that systematically underestimate the long-term value of media investments. The industry standard approach relies on short-term elasticities — the immediate sales lift from a single exposure — and ignores sustained brand accumulation effects that persist weeks or months after the initial advertising exposure. For media channels like television and video, where brand-building is a core function, this oversight is substantial. Brands typically derive 10–15% of weekly sales from long-term media contributions, yet MMM estimates of long-term contributions routinely fall by half of that true value, leading to systematic misallocation of budgets toward short-term performance channels like search. This paper addresses a fundamental question: which estimation methods can reliably recover long-term media contributions, and when can practitioners trust their estimates?

## 2.2 The Identification Challenge in Current Practice

The dominant approach to MMM uses adstock transformations — geometric or polynomial decay functions applied to historical spend series — to capture both short-term and long-term effects in a single coefficient. This framework succeeds when all channels are continuously active: the adstock function can infer long-term persistence by observing how sales respond when one channel's spend fluctuates while others remain constant. However, adstock methods fail fundamentally in two scenarios that are common in practice. First, when long-term effects persist after spend stops — such as a spending pause to measure brand equity decay — adstock cannot separate persistence from zero spend, and the estimated coefficient becomes unreliable. Second, under collinearity, when multiple channels move together, adstock has insufficient statistical variation to identify which channel generates long-term effects, leading to reversals where methods flip the sign and magnitude of channel attribution across scenarios. These limitations have been well-documented in individual case studies, but no comprehensive quantification of their prevalence across methods and scenarios has been published.

## 2.3 Why Existing Validation Approaches Are Insufficient

Most MMM validation studies use either aggregate metrics on real data (where ground truth is unknown), or specialized time series models (Kalman filter, state-space) validated only on their own reconstructed baselines. This circular validation cannot detect systematic under-recovery of true long-term effects. A handful of papers have used synthetic data to validate MMM methods, but none compare more than two frameworks or test robustness to multiple scenarios. The methodological gap is clear: to measure how much of true long-term contributions each method recovers, practitioners need synthetic data where the ground truth data-generating process is known and varied to test method robustness. This is the only way to avoid the confound that "best fit to real data" may mask systematic misattribution of long-term effects to wrong channels or scenarios.

## 2.4 Contributions of This Paper

This paper fills this gap with a reproducible benchmarking framework and four specific contributions:

1. **Synthetic benchmarking framework with known ground truth.** We create a realistic media mix data-generating process with explicit long-term brand stock dynamics, implement ten estimation methods across three framework classes (static adstock, dynamic time-series, state-space), and evaluate performance across five diagnostic scenarios from baseline to structural breaks. All code, synthetic data, and ground truth values are provided for replication.

2. **Empirical evidence that aggregate recovery masks channel-level attribution failure.** We show that a method achieving 68.8% aggregate long-term contribution recovery can return 0% recovery for individual channels, inverting budget allocation recommendations. Practitioners validating only on aggregate metrics will accept models that misallocate systematically across channels.

3. **A three-tier robustness taxonomy based on scenario sensitivity.** We classify methods by their pause-window robustness ratio — how much estimation error increases when spend temporarily stops. Tier 1 methods maintain <1.10× error ratio; Tier 2 methods degrade to 1.10–1.35×; Tier 3 methods exceed 1.35×. This taxonomy operationalizes the distinction between architectures that can and cannot identify latent effects.

4. **A practitioner decision framework for method selection.** Based on signal strength (long-term effects as % of baseline sales) and spend pattern characteristics (stability, discontinuities, seasonality), we recommend specific methods and warn against those with known failure modes. This translates academic findings into actionable guidance for MMM practitioners.

## 2.5 Paper Roadmap

Section 3 describes the synthetic data-generating process, ten estimation methods, and their configuration. Section 4 evaluates framework-level performance on the baseline scenario. Section 5 tests robustness to five diagnostic scenarios, revealing when framework architecture determines success or failure. Section 6 examines channel-level attribution validation, showing where aggregate metrics mislead. Section 7 quantifies calibration sensitivity, comparing frozen parameters from one scenario to optimized parameters from others. Section 8 provides mechanistic explanations for anomalies and failures. Section 9 synthesizes findings into a framework hierarchy and discusses implications for theory and practice. Section 10 concludes with the decision framework and future research directions.

---

## Word Count
893 words

---

## Checklist Before Evaluation
- [x] Paragraph 1: Business problem (no method, scales the issue)
- [x] Paragraph 2: Identification gap (adstock failure modes)
- [x] Paragraph 3: Why synthetic data with ground truth is needed
- [x] Paragraph 4: Four specific contributions (bulleted, not "novel")
- [x] Paragraph 5: Paper roadmap (one sentence per section, mechanical)
- [x] 800–1000 word target (893 words)
- [x] No results tables or figures
- [x] No detailed methodology
- [x] No literature review content
- [x] Central claim clear by end (framework architecture determines reliability)
- [x] Active voice throughout
- [x] No vague language (quantified: 10-15%, 68.8%, 0%, <1.10×, 1.10-1.35×, >1.35×)
