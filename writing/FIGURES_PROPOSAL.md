# Proposed Figure Suite for LTC Frameworks Paper
## Making This a Visual Learning Paper

---

## REQUIRED FIGURES (4)
*From next_steps.txt — essential for paper coherence*

### Figure 1: Robustness Spectrum
**Type:** Horizontal bar chart  
**Data source:** Pause-window ratios (S2) for all 10 models  
**Axis:** Models (y) vs Pause-Window Ratio (x)  
**Key insight:** Visual taxonomy of Tier 1/2/3 robustness  
**Metric:** BSTS 1.02×, Kalman 1.345×, geo_adstock 1.41×, etc.  
**Color coding:** Green (robust <1.10×), Yellow (sensitive 1.10–1.35×), Red (fragile >1.35×)

### Figure 2: Cross-Scenario Recovery Heatmap
**Type:** 10×5 heatmap (models × scenarios)  
**Data source:** Recovery accuracy % for all model-scenario pairs  
**Color scale:** Red (0–20%) → Yellow (40–60%) → Green (80%+)  
**Key insight:** Framework hierarchy (F3 green block, F1 red block); scenario effects (S2 column shifts)  
**Annotations:** Color-code cells by framework (F1, F2, F3 borders)  
**Secondary:** Add framework averages as row sums

### Figure 3: S2 Pause Window Detail
**Type:** Time series (weeks 95–125) with dual y-axis  
**Left axis:** Ground truth LTC (true signal)  
**Right axis:** Model estimates (top 3 models: BSTS, Kalman, geo_adstock)  
**Annotation:** Shade pause period (weeks 104–112)  
**Key insight:** How models respond when spend drops to zero; divergence during pause  
**Data source:** Week-by-week recovery from results JSON

### Figure 4: Channel Attribution Comparison
**Type:** Grouped bar chart or radar plot (6 dimensions: 5 channels + aggregate)  
**Models:** ARDL, Koyck, BSTS, MCMC (top 4 models)  
**Scenario:** S2 (where channel inversion is most dramatic)  
**Key insight:** ARDL 68.8% aggregate but 0% Video; Koyck inverts Social>TV  
**Bar chart option:** X=Channel, Y=Recovery %, Grouped by model  
**Radar option:** Each model is a polygon; overlays show misalignment

---

## RECOMMENDED ADDITIONAL FIGURES (8–10)
*For visual learning and deeper understanding*

### Figure 5: Framework Hierarchy Visualization
**Type:** Dot plot with error bars or violin plot  
**X-axis:** Three frameworks (F1, F2, F3)  
**Y-axis:** Recovery accuracy % (0–100%)  
**Data:** All 5 scenarios pooled per framework  
**Show:** Mean + std dev or box plot showing variance  
**Key insight:** F3: 78.4% ±x.x, F2: 42.8% ±x.x, F1: 22.4% ±x.x  
**Annotation:** Individual points color-coded by scenario (S1–S5)

### Figure 6: Calibration Sensitivity (Frozen vs Optimized)
**Type:** Butterfly/back-to-back bar chart  
**Structure:** Each model has two bars: Frozen (left) | Optimized (right)  
**Data source:** STEP4_OPTIMIZATION_RESULTS.md  
**Color:** Frozen = gray, Optimized = green (improvement)  
**Key insight:** F3 gains 2–3pp, F2 gains 5–10pp, F1 gains <2pp  
**Annotation:** Label improvement in pp on connecting line

### Figure 7: Scenario Difficulty Ranking
**Type:** Heatmap (models × scenarios, showing relative difficulty)  
**Data:** Recovery % per model-scenario, normalized by framework average  
**Color:** Green (beats framework avg), Red (below avg)  
**Key insight:** Which scenarios stress each framework  
**Helps:** Practitioners understand which scenario applies to them

### Figure 8: Pause-Window Ratio Timeline (S2 Detail)
**Type:** Multi-line plot (weeks 95–125)  
**Y-axis:** Cumulative pause-window error (MAPE) over time  
**Lines:** Top 4 models (BSTS, Kalman, geo, MCMC)  
**Annotation:** Shade pause period (104–112)  
**Key insight:** When does error accumulate during pause? How fast does each model recover?  
**Before/After:** Show error rate pre-pause vs during vs post-pause

### Figure 9: Channel-Level Attribution Heatmap (Per-Model, Per-Scenario)
**Type:** Small multiples heatmap (5×5 grid for 5 channels × 5 scenarios, one per model)  
**Top 4 models:** BSTS, MCMC, ARDL, Koyck  
**Color:** Recovery % per channel  
**Key insight:** ARDL Video=0% in S2 despite 68.8% aggregate  
**Annotations:** Star (*) channel that recovers <10%

### Figure 10: Video LTC Signal Loss Pattern
**Type:** Line plot (5 scenarios, 10 models)  
**Y-axis:** Video LTC recovery % (0–100%)  
**X-axis:** Scenarios (S1–S5)  
**Lines:** One per model; color by framework (F1=red, F2=blue, F3=green)  
**Key insight:** F3 (green) maintains Video signal; F1/F2 (red/blue) drop to 0% in S3–S5  
**Diagnostic value:** Video LTC as test of model robustness

### Figure 11: MCMC Convergence Diagnostics (R-hat Values)
**Type:** Dot plot or heatmap  
**Y-axis:** 19 parameters (decay per channel, build_rate, ltc_coef, etc.)  
**X-axis:** Scenarios (S1–S5)  
**Color/size:** R-hat value (green <1.05, yellow 1.05–1.10, red >1.10)  
**Key insight:** MCMC convergence is excellent across scenarios  
**Annotation:** Reference line at 1.05 (gold standard)

### Figure 12: Budget Allocation Error by Model
**Type:** Stacked horizontal bar chart  
**X-axis:** Budget allocation error % (−30% to +30%)  
**Y-axis:** Models (sorted by error magnitude)  
**Stack:** Error per channel (TV, Search, Social, Display, Video in different colors)  
**Scenario:** S1 (baseline) or show top 3 worst-case scenarios  
**Key insight:** Which models misallocate most? Which channels are error hotspots?

### Figure 13: Robustness Taxonomy Visualization
**Type:** Scatter plot (Tier classification) or clustered bar chart  
**X-axis:** Framework (F1, F2, F3)  
**Y-axis:** Pause-window ratio (0.95–1.50)  
**Points:** Each model, sized by average recovery %  
**Color:** Framework (F1=red, F2=blue, F3=green)  
**Annotation:** Draw tier boundaries (1.10×, 1.35×)  
**Key insight:** Visual representation of Section 9.1 taxonomy

---

## OPTIONAL SUPPLEMENTARY FIGURES (3–5)

### Figure A: Model Ranking Reversals (S1 vs S2 vs S3, etc.)
**Type:** Alluvial/Sankey diagram showing rank changes across scenarios  
**Key insight:** Which models change position most?

### Figure B: Scenario Characteristics (DGP Parameters)
**Type:** Radar plot showing 5 scenario attributes  
**Attributes:** Signal strength, spend variation, seasonality, collinearity, discontinuity  
**Key insight:** Context for why certain scenarios stress certain models

### Figure C: Framework Comparison Matrix
**Type:** Summary table converted to visual (3×3 grid with icons/colors)  
**Dimensions:** Performance vs Stability vs Interpretability  
**Shows:** Why practitioners should choose each framework

### Figure D: MCMC Posterior Distributions (S1 Baseline)
**Type:** Violin plots for key parameters (build_rate, ltc_coef, δ per channel)  
**Key insight:** Parameter uncertainty; relative importance  
**Audience:** Methods researchers who want to understand MCMC behavior

### Figure E: Time-Series Decomposition Example
**Type:** Multi-panel time series (1 week × all 261 weeks or S2 detail)  
**Panels:** Baseline | STC | LTC | Noise | Observed sales  
**For one scenario (S1):** Show estimated vs ground truth  
**Key insight:** How do the components add up? What does LTC look like?

---

## RECOMMENDATION FOR A VISUAL LEARNING PAPER

**Tier 1 (Essential — include all 4 + Figures 5, 11, 13):** 7 figures total
- Robustness Spectrum (Figure 1)
- Cross-Scenario Heatmap (Figure 2)  
- S2 Pause Window Detail (Figure 3)
- Channel Attribution (Figure 4)
- Framework Hierarchy (Figure 5) — establishes F3 >> F1 visually
- MCMC Convergence (Figure 11) — validates robustness
- Robustness Taxonomy (Figure 13) — operationalizes Section 9.1

**Tier 2 (Highly Recommended — add 2–3):** Figures 6, 10, 12
- Calibration Sensitivity (Figure 6) — shows structure beats tuning
- Video LTC Signal Loss (Figure 10) — diagnostic visualization
- Budget Allocation Error (Figure 12) — practical implications

**Tier 3 (Optional — enhance if space allows):** Figures 7, 8, 9, A–E
- Scenario Difficulty (Figure 7)
- Pause Window Timeline (Figure 8)
- Channel-Level Detail (Figure 9)
- Supplementary (A–E) for methods researchers

---

## Data Source Summary

All data available in:
- **Recovery metrics:** `outputs/results/{model}_{scenario}.json`
- **Channel attribution:** `comprehensive_analysis/S3_S4_S5_CHANNEL_ATTRIBUTION.txt`
- **Calibration:** `comprehensive_analysis/STEP4_OPTIMIZATION_RESULTS.md`
- **Pause-window detail:** Individual JSON files have week-by-week data
- **MCMC diagnostics:** `outputs/results/mcmc_stock_S{N}.json` (R-hat in metadata)

---

## Implementation Strategy

1. **Script:** Create `figures_generation.py` to load JSON data and generate all plots
2. **Style:** Consistent color scheme (F1=red, F2=blue, F3=green)
3. **Fonts:** Publication-quality (Calibri 11pt labels, 8pt captions)
4. **Format:** PNG @ 300dpi for initial draft; PDF for final submission
5. **Layout:** Full-width figures where possible (journal typically 3.5"–7" width)

---

## Estimated Time to Generate

- **Tier 1 (7 figures):** ~2–3 hours (matplotlib + seaborn)
- **Tier 1+2 (10 figures):** ~3–4 hours
- **All Tiers (14+ figures):** ~5–6 hours (includes quality review)

**Total word-count impact:** Adding Figures 5–13 = minimal text overhead (captions ~500 words total)

---

## Recommendation

**For a visual learning paper focused on practitioner decision-making:**

Suggest **Tier 1 + Tier 2 (10 figures total)** as the optimal balance:
- Core message clear (Figures 1, 2, 5, 13)
- Robustness validated (Figure 11)
- Practical implications shown (Figures 6, 10, 12)
- Full detail preserved (Figures 3, 4)
- Space efficient for journal submission

This creates a paper that works for **three audiences:**
1. **Methods researchers:** MCMC convergence, robustness taxonomy, optimization
2. **Practitioners:** Channel attribution, decision table, budget allocation
3. **Executives:** Framework hierarchy, robustness spectrum, recommendations

