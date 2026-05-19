# PHASE 1: Writing Skills Framework Setup & Journal Standards

**Status:** Complete preparation document for drafting Introduction and Methodology sections  
**Created:** 2026-05-18  
**Applies to:** All sections (Introduction Section 2, Methodology Section 3)

---

## I. JOURNAL STANDARDS & REQUIREMENTS

### Target Journals (In Order of Preference)

#### 1. Journal of Marketing Research (JMR) — PRIMARY
- **Voice Requirement:** "Write in an interesting, readable manner. The journal is designed to be read, not deciphered."
- **Type:** Quantitative methodology paper with empirical benchmarking (✓ matches our paper)
- **Replicability Requirement:** Research transparency materials (code, data, experimental logs)
- **Tone:** Accessible to marketing scholars who may not be statisticians

#### 2. Marketing Science — SECONDARY
- **Replicability Requirement:** Code AND data required on acceptance (strict)
- **Emphasis:** Readability to as broad an audience as feasible
- **Type:** Accepts methodology papers with empirical validation
- **Scope:** Methods that advance marketing practice

#### 3. International Journal of Research in Marketing (IJRM) — TERTIARY
- **Replicability:** Data sharing encouraged
- **Tone:** Values clarity in factual reporting
- **Scope:** Empirical marketing research

**Critical:** All three journals require replicability. Our methods section MUST include seed value (42), script names, and locations where code/data will be available.

---

## II. GLOBAL WRITING STANDARDS (ALL SECTIONS)

### A. Voice and Tone Rules

#### Active Voice (Default)
```
CORRECT:   "MCMC recovers 88.5% of true LTC under weak signal conditions"
WRONG:     "88.5% recovery was achieved by MCMC under weak signal conditions"
```

#### Passive Voice Exception
- **Only acceptable in Methodology when describing procedures:**
  - "Parameters were fixed to their true data-generating values" ✓
  - "The synthetic data was generated using seed 42" ✓
  - Anywhere else: convert to active voice

#### Tone Requirement: Precise, Confident, Readable
- No hedging on directly observed findings
- No vague qualifiers
- Define technical terms at first use (don't assume MMM knowledge)

---

### B. Empirical Claim Standard (CRITICAL)

**Every empirical claim must follow this structure:**
```
[Finding] + [specific number] + [table/figure reference] + [mechanism]
```

**Example from paper_notes Finding #1:**
```
"BSTS achieves a pause-window robustness ratio of 1.02 (Table 4), maintaining 
nearly identical error distribution across smooth and discontinuous spend regimes — 
a result explained by its latent stock transition equation, which depreciates 
independently of the spend observation."
```

**NEVER write these (examples of violations):**
- ❌ "high recovery" → ✓ "82.4% recovery"
- ❌ "performs better" → ✓ "outperforms by 13.2 percentage points"
- ❌ "the model fails" → ✓ "returns 0% LTC recovery"
- ❌ "as expected" → ✓ cite the theoretical prediction that generated the expectation
- ❌ "interestingly" → ✓ let the number speak

---

### C. Precision Rules (MUST FOLLOW EXACTLY)

| Element | Format | Example | Wrong |
|---------|--------|---------|-------|
| Percentages | One decimal place | 82.4% | 82% or 82.43% |
| Ratios | Two decimal places | 1.02× | 1.0× or 1.021× |
| Scenario spec | Always include scenario | "(S3, Table 3)" | "Table 3" alone |
| Model spec | Always name model | "MCMC achieves..." | "The method achieves..." |
| Dollar figures | $M with one decimal | $0.49M | $490K or $0.492M |

---

### D. Hedging Rules

#### WHEN to Hedge (mechanism is inferred)
Use: "suggests", "is consistent with", "indicates", "may reflect"
```
"The seasonal signal suggests that MCMC's latent stock estimate improves under collinearity"
```

#### WHEN NOT to Hedge (directly observed from ground truth)
Use confident language:
```
"MCMC recovers 88.5% of true LTC in S5 supplementary analysis"
"ARDL returns 0% per-channel recovery in S2 despite 68.8% aggregate"
"The pause-window ratio is 1.02× for BSTS"
```

#### NEVER Hedge in Abstract
State findings as facts.

---

### E. Sentence & Paragraph Structure

#### Sentence Length
- **Maximum 35 words per sentence**
- Longer sentences lose reviewers
- Break into two sentences if needed

#### Paragraph Structure
- **One idea per paragraph**
- First sentence states the idea
- Last sentence reinforces/concludes it
- Use transitional phrases between paragraphs:
  - "Building on this finding..."
  - "This result is consistent with..."
  - "In contrast..."

#### Technical Terms
- Define at first use
- Never assume reader knows MMM jargon (even for marketing audience)
- Example: "The latent brand stock (the accumulated consumer awareness and preference from past spending) follows the state-space dynamic: stock[t] = δ × stock[t-1] + build_rate × √spend[t]"

---

### F. Words/Phrases to Avoid

| Phrase | Problem | Alternative |
|--------|---------|-------------|
| "significant" | Confused with statistical significance | Use "large" or "substantial" if you mean size |
| "robust" (loosely used) | Vague without definition | Define robustness: "robust to scenario variation means..." |
| Footnotes | AMA journals discourage | Use inline explanation instead |
| Acronyms in abstract | Reduces accessibility | Define in text or avoid in abstract |
| "novel" describing own work | Unprofessional | Let the contribution speak for itself |
| "to the best of our knowledge" | Weak framing | Just state the gap and cite it |

---

### G. Citation Format (JMR/AMA Style)

#### In-Text Citations
```
One author:        (Clarke 1976)
Two authors:       (Srinivasan and Hanssens 2009)
Three+ authors:    (Jin et al. 2017)
Multiple citations: (Clarke 1976; Hanssens et al. 2001)
```

#### Reference List Format
```
Author Last, First (Year), "Title of Article," *Journal Name*, Volume (Issue), pages.

Example:
Clarke, Darral G. (1976), "Econometric Measurement of the Duration of Advertising 
Effect on Sales," *Journal of Marketing Research*, 13 (4), 345–357.
```

#### Key Requirement
- Journal names and book titles must be italicized
- Article titles in quotes
- Include exact page numbers
- Include volume and issue number

---

### H. Equations

#### Numbering
- Number every equation referenced in text: (1), (2), etc.

#### Symbol Definition
- Define every symbol at first appearance
- Specify units explicitly in surrounding text

#### Interpretation
- Follow each equation with ONE sentence of plain-language interpretation
- Center equations on their own line

#### Example (from Methodology section)
```
stock[t] = δ × stock[t-1] + build_rate × √spend[t]  (Eq 3)

where δ is the stock depreciation rate (0–1, unitless), build_rate is the 
per-channel brand-building coefficient ($/million^0.5), and spend[t] is weekly 
spend in dollars. Equation 3 represents latent brand stock as a geometric 
depreciation process with spend-driven accumulation.
```

---

## III. SECTION-SPECIFIC GUIDANCE

### SECTION 2: INTRODUCTION (800–1000 words, 4–5 paragraphs)

#### Paragraph Structure (Required)

**Paragraph 1 — Business Problem (Broad Opening)**
- Do NOT start with method (no adstock, state-space yet)
- Anchor with industry fact (budget scale, MMM adoption rate)
- Why does LTC matter to marketers?
- Set scope: marketing ROI estimation problem

**Paragraph 2 — The Identification Gap**
- What does current practice (adstock methods) do?
- What does it miss?
- Introduce adstock briefly
- Core failure: methods that tie LTC to spend cannot detect LTC that persists after spend stops
- Cite 2–3 foundational MMM papers here

**Paragraph 3 — Why Existing Benchmarks Are Insufficient**
- Current practice uses aggregate metrics or real data (where ground truth is unknown)
- **Methodological gap:** Cannot measure exact recovery without ground truth
- This paper: synthetic data with known ground truth
- This is the methodological contribution framing

**Paragraph 4 — What This Paper Contributes**
- List 3–4 specific contributions
- Use numbered or bulleted format
- **Be specific:** name scenarios, taxonomy, findings
- Do NOT use "novel" or "to the best of our knowledge"

**Example contributions (from next_steps.txt):**
1. A reproducible synthetic data benchmarking framework with known LTC ground truth
2. Empirical evidence that aggregate LTC recovery masks channel-level attribution failure
3. A three-tier robustness taxonomy (Structural / Identification-Dependent / Fragile) based on pause-window ratio
4. A practitioner decision framework for method selection based on signal strength and spend conditions

**Paragraph 5 — Roadmap (Optional but Recommended)**
- One sentence per major section
- Mechanical format acceptable here
- Example: "Section 2 describes our synthetic data framework. Section 3 presents results from a benchmark of 10 models across 5 scenarios. Section 4 interprets findings..."

#### Tone Guidance for Introduction
- JMR wants clear theoretical/substantive/methodological contribution: state yours clearly by paragraph 4
- Marketing Science values papers answering important questions: frame around "How much LTC are brands missing, and why?"
- Do not oversell. State what you found and why it matters.

#### Do NOT Include in Introduction
- Results tables or figures
- Detailed methodology (that's Section 3)
- Literature review (that's Section 2 in full structure)
- Definitions of all terms (introduce key ones, detailed definitions in Methods)

---

### SECTION 3: METHODOLOGY (1500–2000 words, 4 subsections)

#### Subsection 3.1: Synthetic Data Framework

**Required Content:**
1. Why synthetic data: Ground truth unavailable in real MMM data
2. The fundamental equation with equation number
3. The five components: baseline, STC, LTC, exogenous effects, noise
4. Baseline generation method: piecewise trend, seasonality, holidays
5. LTC generation mechanism: latent stock model (with equations)
6. The five scenarios: one paragraph each
7. Why parameters were fixed to true DGP values (isolates structural from calibration)

**Required Equations:**
```
Eq 1: net_sales[t] = baseline[t] + Σ STC_ch[t] + Σ LTC_ch[t] + exog[t] + ε[t]
Eq 2: adstocked[t] = impressions[t] + λ × adstocked[t-1]  (for STC component)
Eq 3: stock[t] = δ × stock[t-1] + build_rate × √spend[t]  (for LTC component)
Eq 4: LTC[t] = ltc_coef × stock[t]
```

**Required Table 1: Scenario Library**
| Scenario | Name | Key Property | Structural Challenge | Framework Tested |
|----------|------|--------------|---------------------|------------------|
| S1 | Baseline | Low collinearity | Performance ceiling | All methods |
| S2 | Spend Pause | TV + Video = 0 weeks 104–112 | LTC persistence after spend stops | F1/F3 separation |
| S3 | High Seasonality | 52-week cycle | Spend-baseline confound | F1 collapse test |
| S4 | Permanent Shift | Spend reduced 80% weeks 104+ | Structural break adaptation | F3 advantage |
| S5 | Weak Signal | LTC = 50% of S1 level | False discovery control | Prior sensitivity |

**Required Table 2: Channel Parameters**
| Channel | True STC Decay (λ) | True LTC Delta (δ) | Half-Life (weeks) | Status |
|---------|-------------------|-------------------|------------------|--------|
| TV | 0.55 | 0.90 | 7.3 | Assumed (fixed to true DGP) |
| Search | 0.20 | 0.30 | 1.0 | Assumed (fixed to true DGP) |
| Social | 0.45 | 0.82 | 4.6 | Assumed (fixed to true DGP) |
| Display | 0.50 | 0.65 | 2.7 | Assumed (fixed to true DGP) |
| Video | 0.60 | 0.88 | 6.7 | Assumed (fixed to true DGP) |

#### Subsection 3.2: Estimation Frameworks

**Required Content:**
- One subsection per framework (F1, F2, F3)
- For each framework:
  - Core structural assumption (one sentence)
  - Methods included (list of models)
  - What parameters are assumed vs estimated
  - Why this framework would succeed or fail structurally

**Key Sentence (MUST INCLUDE):**
```
"To isolate structural framework differences from calibration effects, all decay 
parameters were fixed to their true data-generating values. Under this design, 
differences in recovery accuracy across scenarios reflect structural framework 
properties, not parameter mis-specification."
```

**Framework 1: Static Adstock Regression** (4 models)
- Core assumption: "LTC and STC can be estimated via fixed geometric or Weibull lag structures applied to impressions"
- Models: geo_adstock, weibull_adstock, almon_pdl, dual_adstock
- Fixed: decay parameters (λ per channel)
- Estimated: coefficients on adstocked impressions
- Structural vulnerability: Cannot adapt to collinearity; cannot distinguish channels under seasonal confounding

**Framework 2: Dynamic Time-Series Distributed Lag** (3 models)
- Core assumption: "LTC appears as lagged effects in distributed lag models with AR structure"
- Models: koyck, ardl, finite_dl
- Fixed: decay parameters (λ per channel)
- Estimated: lag weights, AR coefficients
- Structural vulnerability: Relies on spend variation for identification; AR terms can overfit and invert channel rankings

**Framework 3: State-Space / Latent Brand-Stock** (3 models)
- Core assumption: "LTC is a latent brand stock process with explicit accumulation dynamics"
- Models: kalman_dlm, bsts, mcmc_stock
- Fixed: decay parameters (δ per channel) [Kalman/BSTS]; estimated decay (mcmc_stock posterior)
- Estimated: observation variance, state variances, stock initialization
- Structural advantage: Decouples LTC from spend observation; stock dynamics more robust to identification challenges

#### Subsection 3.3: Evaluation Metrics

**Required Content:**
- Define every metric with equation
- Do not assume reader knows MAPE

**Required Metric Definitions:**

```
Eq 5: LTC_MAPE = mean(|recovered_ltc[t] - ltc_true[t]| / ltc_true[t]) × 100

where recovered_ltc[t] is the model's estimated LTC contribution in week t, 
ltc_true[t] is the ground-truth LTC value, and averaging is over 261 weeks.

Eq 6: Recovery = (1 - LTC_MAPE/100) × 100

Recovery represents the percentage of true LTC signal correctly identified by the model. 
A value of 82.4% means the model captures 82.4% of the true variation in LTC.

Eq 7: Pause_Window_Ratio = pause_window_MAPE / full_series_MAPE

where pause_window_MAPE is computed over weeks 100–120 (containing the spend pause weeks 104–112), 
and full_series_MAPE is computed over all 261 weeks. A ratio near 1.0 indicates robust 
performance across spend structures; ratios >1.3 indicate fragility to structural breaks.

Eq 8: Channel_Budget_Error[ch] = recovered_share[ch] - true_share[ch]

where share[ch] = (LTC[ch] + STC[ch]) / Σ(LTC[ch] + STC[ch]) across all five channels. 
This metric measures the model's accuracy in attributing ROI to individual channels, 
essential for budget allocation decisions.
```

**Justify Channel-Level Validation (CRITICAL):**
```
Aggregate LTC recovery alone is insufficient for model validation. As demonstrated in our 
S2 results, ARDL achieves 68.8% aggregate recovery while recovering 0% LTC per channel 
(offsetting errors cancel in aggregate). Koyck similarly masks channel ranking inversions 
(assigning 59.3% to Display, true rank #4, vs 2.2% to TV, true rank #1). Practitioners 
relying on aggregate metrics alone would recommend systematically wrong budget allocation. 
Therefore, we require that all models demonstrate correct channel-level recovery as a 
validation criterion, not just aggregate accuracy.
```

#### Subsection 3.4: Replicability (Required)

**Required Content:**
- One paragraph minimum
- Must specify random seed (42)
- Must name the generator script
- Must state where code/data will be available

**Template:**
```
All results are fully reproducible. Synthetic data was generated using seed 42 via the 
mmm_synthetic_generator.py script, available at [repository URL]. The 261-week time series, 
ground-truth LTC and STC contributions, and exogenous variables are deterministic given this 
seed. All 10 models were estimated using the experiments/run_experiment.py CLI interface. 
Upon acceptance, we will provide (1) the full synthetic dataset, (2) all model estimation code 
with reproducible hyperparameter configurations, and (3) the experiment logs documenting 
every model run. This commitment satisfies the replicability requirements of Journal of 
Marketing Research, Marketing Science, and the International Journal of Research in Marketing.
```

#### Do NOT Include in Methodology
- Results (those go in Section 4)
- Literature citations for methods beyond founding papers
- Detailed derivations (move to appendix if needed)

---

## IV. UNIVERSAL EVALUATION CHECKLIST

### Apply to Every Section

- [ ] **Active voice used throughout** (passive only in methodology procedures)
- [ ] **No vague language:** Every instance of "high", "low", "better", "significant" paired with number
- [ ] **Every empirical claim has specific number** (e.g., "82.4%" not "high recovery")
- [ ] **Every number has table/figure reference** where applicable (e.g., "(Table 3)" or "(S2, Figure 2)")
- [ ] **All sentences under 35 words** (break long sentences into two)
- [ ] **Technical terms defined at first use** (assume reader doesn't know MMM jargon)
- [ ] **Citation format matches JMR/AMA** (no comma: Author Year, not Author, Year)
- [ ] **No hedging on directly observed findings** (don't say "suggests" for ground-truth comparisons)

---

## V. INTRODUCTION-SPECIFIC CHECKLIST

- [ ] **Opens with business problem, not method** (no adstock, state-space in paragraph 1)
- [ ] **Central claim stated by end of section** (clear by end of paragraph 4)
- [ ] **3–4 specific contributions listed** (use numbered format)
- [ ] **Roadmap paragraph present** (one sentence per major section)
- [ ] **No results reported** (data stays in Section 4)
- [ ] **800–1000 words** (rough target, 750–1100 acceptable)

---

## VI. METHODOLOGY-SPECIFIC CHECKLIST

- [ ] **Equations numbered (Eq 1–Eq 8 minimum)**
  - Eq 1: net_sales decomposition
  - Eq 2: STC adstock
  - Eq 3: Stock accumulation
  - Eq 4: LTC contribution
  - Eq 5: LTC_MAPE definition
  - Eq 6: Recovery definition
  - Eq 7: Pause-window ratio definition
  - Eq 8: Channel budget error definition

- [ ] **Every symbol defined**
  - δ = stock decay/retention rate (0–1, unitless)
  - λ = short-term adstock decay (0–1, unitless)
  - build_rate = brand stock accumulation coefficient
  - ltc_coef = LTC contribution per unit stock
  - spend[t] = weekly media spend in dollars
  - stock[t] = latent brand stock (units: stock_units)

- [ ] **Scenario table present** (5 rows S1–S5, 4 columns minimum)
- [ ] **Parameter table present** (5 rows channels, columns: δ, λ, half-life, assumed/estimated)
- [ ] **Fixed-parameter design justified explicitly** (key sentence about isolating structure from calibration)
- [ ] **Replicability paragraph present** (seed 42, script name, data/code availability)
- [ ] **1500–2000 words** (approximately 3 pages single-spaced)

---

## VII. CRITICAL METRICS TO CITE CORRECTLY

**These values MUST be cited accurately from paper_notes.md and comprehensive_analysis files:**

### Framework Averages (S1–S4 frozen parameters)
- F3 average: 78.4%
- F2 average: 42.8%
- F1 average: 22.4%

### BSTS (Gold Standard)
- S1 recovery: 82.4%
- S1 pause-window ratio: 1.02× (CRITICAL: centrepiece finding)
- Cross-scenario recovery range: 76.8%–82.4% (StdDev 2.4pp, most stable)

### MCMC
- S1 recovery: 72.4% (frozen parameters)
- S3 recovery: 99.0% (peaks on seasonality)
- S5 supplementary recovery: 88.5% (with manual stock init + weakened priors)
- Video LTC recovery: 46–71% across S3/S4/S5 (only model recovering video)

### ARDL (Paradox Example)
- S1 recovery: 0.0% (prior misspecification)
- S2 recovery: 68.8% (resurrected by spend pause)
- S2 channel-level recovery: 0% per channel (aggregate masking)
- S4 recovery: −19.8% (sign-flip catastrophe, negative recovery)

### Koyck (Channel Inversion Example)
- S2 aggregate recovery: 43.0%
- S2 channel rankings inverted: Display 59.3% (true rank #4) > TV 2.2% (true rank #1)

### Video LTC Test (Differentiator)
- Recovery by all non-MCMC models in S3/S4/S5: 0%
- Recovery by MCMC in S3/S4/S5: 46–71%
- True δ difference: TV 0.90 vs Video 0.88 (only MCMC resolves)

---

## VIII. CLARIFYING QUESTIONS TO ADDRESS

### Q1: Literature Review Reference Standard
**Question:** Section 2 (full paper) requires Literature Review. Who are the "founding papers"?

**Current Status:** Not yet specified in skills folder. 

**Recommendation:** Based on context:
- Clarke (1976): "Econometric Measurement of the Duration of Advertising Effect on Sales" — foundational adstock model
- Likely others in CLAUDE.md Project Brief or prior art citations
- **Action needed:** Before drafting Lit Review, identify 3–5 foundational MMM papers from existing literature citations in the codebase

### Q2: Supplementary Tables Inclusion
**Question:** Visuals.md specifies 6 required tables and 4 required figures. Where do these go in the paper?

**Current Status:** Table/figure placement not specified.

**Recommendation:**
- Tables 1–2 (Scenario Library, Channel Parameters) → Methodology section
- Tables 3–6 (Full Recovery Matrix, Pause-Window Robustness, Channel Attribution, Budget Allocation Error) → Results sections
- Figures 1–4 → Results sections
- Large tables (10×5 cross-scenario matrix) → Appendix

### Q3: Real Journal Citation Standards
**Question:** Should we cite actual JMR/Marketing Science papers or use generic citations for now?

**Current Status:** Introduction mentions "2–3 foundational MMM papers" without specifying which.

**Recommendation:** 
- For Introduction draft: use placeholder citations [CITE: foundational adstock paper], [CITE: MMM review]
- Fill with real citations after methodology is locked
- This prevents citation mismatches if methodology changes

### Q4: Data-Generating Process Justification
**Question:** Subsection 3.1 requires "why parameters were fixed to true DGP values." Is this the right justification level?

**From methodology.md:** "To isolate structural framework differences from calibration effects, all decay parameters were fixed to their true data-generating values."

**Confirmation:** YES, this is the required framing. Emphasize: frozen parameters separate structure from calibration.

---

## IX. SESSION WORKFLOW FOR INTRODUCTION + METHODOLOGY DRAFTING

### Session 1: Introduction (90 minutes total)
1. **Outline** (15 min): Write 4–5 paragraph topics
2. **Draft** (60 min): Write each paragraph, follow structure in Section III.A
3. **Evaluate** (15 min): Run Introduction-Specific Checklist (Section V)

### Session 2: Methodology (2+ hours total)
1. **Subsection 3.1: Synthetic Data** (60 min)
   - Write why synthetic data is needed
   - Insert Eq 1–4 with plain-language interpretations
   - Describe baseline, STC, LTC, exogenous, noise
   - Describe five scenarios (one para each)
   - Create Tables 1–2
   - Explain fixed-parameter design

2. **Subsection 3.2: Estimation Frameworks** (40 min)
   - Write F1, F2, F3 frameworks (one subsection each)
   - Include key sentence about isolating structure from calibration
   - List assumed vs estimated parameters per framework

3. **Subsection 3.3: Evaluation Metrics** (30 min)
   - Insert Eq 5–8 with interpretations
   - Write justification for channel-level validation (copy/adapt from Section VII)

4. **Subsection 3.4: Replicability** (10 min)
   - Write paragraph specifying seed 42, script name, data/code location

5. **Evaluate** (10 min): Run Methodology-Specific Checklist (Section VI)

---

## X. WHAT TO READ NEXT (IN ORDER)

1. ✅ **You are here:** PHASE1_SETUP_WRITEUP.md (this document)
2. **Before writing Introduction:** Re-read introduction.md focusing on paragraph structure
3. **Before writing Methodology:** Re-read methodology.md and identify real citation sources for lit review
4. **While drafting:** Keep writing_skill.md, evaluation.md, and this checklist open
5. **After drafting:** Run evaluation.md checklist, compare metrics to Section VII

---

## XI. FINAL REMINDERS

- **Never assume MMM knowledge** — define every technical term
- **Citation format is non-negotiable** — (Author Year) no comma
- **Metrics precision is critical** — 82.4%, not 82%; 1.02×, not 1.0×
- **Figures must be cited before appearing** — "As shown in Table 3..." then insert table
- **No vague language** — if you write "high", rewrite with number instead
- **Replicability is mandatory** — include seed, script name, data location in Methods
- **Channel-level validation is central to this paper's contribution** — emphasize throughout

---

**End of Phase 1 Writeup**

Status: Ready to draft Introduction and Methodology following this guidance.
