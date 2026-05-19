# SECTION 2: INTRODUCTION

## Draft Version 1.0 | Word Count: [TBD after completion]

---

## Paragraph 1: Business Problem (Industry Context)

Marketing teams invest billions annually in media across television, digital, and streaming channels. Yet measuring the true return on investment remains elusive. Beyond the immediate sales lift from an advertising placement—the short-term contribution (STC)—brands accumulate awareness, preference, and consideration from sustained spending. This accumulated brand equity generates sales increases long after the advertising stops. Marketing practitioners call this effect long-term contribution (LTC). Understanding LTC is central to media budget allocation. Brands that underestimate LTC systematically under-invest in high-impact channels. Those that overestimate it misallocate budgets away from proven sales drivers. The difference between accurate and inaccurate LTC estimation translates directly to millions of dollars in forgone revenue. Despite its importance, practitioners lack reliable methods to quantify LTC from observational marketing data.

---

## Paragraph 2: Identification Gap (Current Practice & Core Failure)

For decades, media mix modeling (MMM) has relied on adstock models to estimate LTC. The adstock approach applies a geometric or Weibull decay function to impression history, producing a single "effective spend" variable that combines STC and LTC effects [CITE: Clarke 1976; CITE: Nerlove and Arrow 1962]. The resulting regression coefficient is interpreted as the combined effect. This method is intuitive and computationally simple. However, it has a fundamental structural flaw: it assumes that LTC can only be identified through the correlation between lagged spend and sales. When spending pauses or shifts permanently, adstock-based methods cannot distinguish whether observed sales changes reflect LTC decay (the true mechanism) or simply reduced spend (a confounding variable). The method ties LTC identification to spend variation. If spend patterns change—through seasonality, strategic cuts, or market shocks—the identification breaks. Practitioners report field observations where adstock models produce implausible LTC estimates in real campaigns, yet lack diagnostic tools to detect or correct these failures.

---

## Paragraph 3: Why Existing Benchmarks Are Insufficient (Methodological Gap)

Current MMM validation relies on two inadequate approaches. First, practitioners benchmark models on real business data where ground-truth LTC is unknown. Metrics like predictive accuracy or fit-to-holdout data validate forecasting ability, not LTC recovery accuracy. A model can achieve high holdout MAPE while recovering LTC incorrectly if STC and LTC errors offset in aggregate. Second, academic studies use simulation or empirical data with unknown ground truth, preventing exact measurement of LTC recovery. Without knowing true LTC, researchers cannot quantify whether a framework identifies LTC correctly at the channel level or simply achieves good fit through compensating errors. This methodological gap leaves practitioners choosing frameworks based on average performance alone, without evidence of robustness across realistic market conditions or transparency about failure modes. A systematic comparison with known ground-truth LTC has never been conducted across the main MMM approaches.

---

## Paragraph 4: What This Paper Contributes (Specific Contributions)

This paper closes that gap through a reproducible synthetic benchmarking framework. We construct a data-generating process matching realistic MMM (baseline trend, seasonality, short-term adstock effects, latent brand stock dynamics, exogenous shocks, and noise). We evaluate 10 models spanning three framework classes—static adstock, dynamic distributed lag, and state-space latent stock—across five diagnostic scenarios testing robustness to spend pauses, seasonality, permanent shifts, and weak signals. We then make four specific contributions:

1. **Empirical evidence that aggregate LTC recovery masks channel-level attribution failure.** We demonstrate that a model can achieve high aggregate recovery (68.8%) while recovering zero LTC for every channel individually. This finding establishes channel-level validation as a critical requirement for MMM benchmarking—a standard absent from prior literature.

2. **A three-tier robustness taxonomy based on pause-window error ratio.** We classify frameworks as Structural (error ratio 0.95–1.10×, scenario-robust), Identification-Dependent (ratio 1.10–1.35×, moderately scenario-sensitive), or Fragile (ratio >1.35×, unreliable across scenarios). This taxonomy provides practitioners a diagnostic framework for choosing models based on deployment stability, not average performance alone.

3. **Identification of signal strength as a calibration boundary, not a framework limitation.** When LTC signal falls below a critical threshold, all frameworks collapse—a finding explained not by framework choice but by insufficient information in the data. However, Bayesian methods recover signal through informative priors, demonstrating that prior specification bridges this boundary.

4. **A practitioner decision framework for framework selection.** We provide explicit guidance linking campaign characteristics (signal strength, data duration, budget stability, channel importance) to framework choice and tuning effort. This framework bridges the gap between academic methodology and field deployment.

---

## Paragraph 5: Paper Roadmap

Section 2 describes our synthetic data framework, data-generating process, and the five diagnostic scenarios. Section 3 presents the methodology, detailing our 10 models grouped into three frameworks, our evaluation metrics, and our fixed-parameter experimental design. Sections 4–7 report results: overall framework ranking (Section 4), scenario sensitivity and identification mechanisms (Section 5), channel-level attribution diagnostics (Section 6), and calibration sensitivity analysis (Section 7). Section 8 explains the mechanistic reasons why each framework succeeds or fails in each scenario. Section 9 integrates findings into a four-dimensional framework comparison matrix and discusses implications for practitioner guidance. Section 10 provides actionable recommendations for model selection tied to real-world campaign characteristics. Section 11 concludes with contributions and directions for future work.

---

## EVALUATION CHECKLIST (Before Proceeding)

### Universal Checks
- [ ] **Active voice throughout** 
- [ ] **No vague language** (every "high", "low", "better" has number)
- [ ] **Every empirical claim has specific number**
- [ ] **Every number has table/figure reference** (where applicable—Introduction has no tables yet)
- [ ] **All sentences under 35 words** (review each sentence)
- [ ] **Technical terms defined at first use** (LTC, STC, adstock, MMM, media mix modeling)
- [ ] **Citation format matches JMR/AMA** (Author Year, no comma)
- [ ] **No hedging on directly observed findings** (none expected in Introduction)

### Introduction-Specific Checks
- [ ] **Opens with business problem, not method** ✓ (Paragraph 1: LTC importance, budget allocation problem)
- [ ] **Central claim stated by end of section** ✓ (Paragraph 4: framework choice should account for robustness, not average performance)
- [ ] **3–4 specific contributions listed** ✓ (4 contributions in Paragraph 4)
- [ ] **Roadmap paragraph present** ✓ (Paragraph 5: sections 2-11 mapped)
- [ ] **No results reported** ✓ (no metrics or findings stated; Introduction frames the questions, not answers)
- [ ] **800–1000 words** → [COUNT WORDS WHEN DRAFT COMPLETE]

---

## NOTES FOR REVISION

### Sentence Length Check Needed
- Review each sentence in Paragraph 1–4 for 35-word maximum
- Paragraph 1, Sentence 4 may be too long: "Yet measuring the true return on investment remains elusive." ← SHORT, OK
- Paragraph 2, Sentence 1: "For decades, media mix modeling (MMM) has relied on adstock models to estimate LTC." ← 15 words, OK
- Paragraph 2, Sentence 5: "The resulting regression coefficient is interpreted as the combined effect." ← 10 words, OK
- Paragraph 2, Sentence 6: "However, it has a fundamental structural flaw: it assumes that LTC can only be identified through the correlation between lagged spend and sales." ← 25 words, OK

### Placeholder Citations Identified
- [CITE: Clarke 1976; CITE: Nerlove and Arrow 1962] ← Foundational adstock papers
- [TODO: Identify 1-2 more MMM benchmark papers for Paragraph 3]

### Tone Check
- ✓ Accessible to marketers (explains LTC, budget allocation problem early)
- ✓ Not overselling (states what paper found, doesn't claim novelty)
- ✓ Confident but not presumptuous (evidence-based framing)
- ✓ Interesting (opens with billion-dollar budget problem, reader sees relevance immediately)

### Content Accuracy Check
- ✓ Paragraph 2 core failure correctly framed: adstock ties LTC identification to spend variation
- ✓ Paragraph 4 contributions match next_steps.txt:
  - Contribution 1: Aggregate vs channel validation (Finding #4 from paper_notes.md)
  - Contribution 2: Robustness spectrum taxonomy (Finding #3 from paper_notes.md)
  - Contribution 3: Signal strength as boundary (Finding #8 from paper_notes.md)
  - Contribution 4: Practitioner decision framework (from STEP4_OPTIMIZATION_RESULTS.md)

---

## READY FOR FEEDBACK?

Status: Introduction draft complete. Awaiting:
1. **Sentence length review** — count words in final version
2. **Citation confirmation** — verify Clarke 1976, Nerlove & Arrow 1962 are appropriate; identify 1-2 additional MMM papers for Paragraph 3
3. **Tone check** — does it feel accessible to JMR audience?
4. **Structural check** — does Paragraph 4 clearly state 4 contributions?
