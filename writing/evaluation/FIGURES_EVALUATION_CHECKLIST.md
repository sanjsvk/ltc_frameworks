# FIGURES EVALUATION CHECKLIST
## LTC Frameworks Paper - 14-Figure Suite Quality Assurance

**Evaluation Date:** 2026-05-23  
**Total Figures:** 14  
**Format:** PNG @ 300dpi  
**Status:** PENDING VALIDATION

---

## UNIVERSAL QUALITY CHECKS (All Figures)

### Visual Design Standards
- [ ] **Color Scheme Consistency**
  - [ ] F1 (Static Adstock) = Red (#d62728) in ALL figures
  - [ ] F2 (Dynamic AR) = Blue (#1f77b4) in ALL figures
  - [ ] F3 (State-Space) = Green (#2ca02c) in ALL figures
  - [ ] Consistent color application across all 14 figures
  - [ ] No framework color mismatches

- [ ] **Font Consistency**
  - [ ] Title font size: 11pt (consistent)
  - [ ] Label font size: 9pt (consistent)
  - [ ] All text readable at intended publication size
  - [ ] No font variations between figures

- [ ] **DPI & Resolution**
  - [ ] All files: 300dpi PNG format
  - [ ] No pixelation or blurriness
  - [ ] Crisp lines and text
  - [ ] File sizes reasonable (1-3MB per figure)

- [ ] **Layout & Margins**
  - [ ] Tight layout (no wasted whitespace)
  - [ ] Legends positioned for clarity (not obscuring data)
  - [ ] Titles have sufficient padding (15-20pt)
  - [ ] Axes labels have sufficient margins

- [ ] **Grid & Reference Lines**
  - [ ] Grid lines present where appropriate (alpha=0.3 or similar)
  - [ ] Reference lines (e.g., pause period, tier boundaries) clearly visible
  - [ ] No overlapping grid that obscures data
  - [ ] Axis lines visible and clear

---

### Metric Accuracy Checks
- [ ] **All Numbers Verified Against Source Data**
  - [ ] Recovery percentages match outputs/results/*.json
  - [ ] Pause-window ratios match comprehensive_analysis/
  - [ ] Framework averages: F3 78.4%, F2 42.8%, F1 22.4%
  - [ ] MCMC convergence: R-hat <1.05 across scenarios
  - [ ] Budget allocation errors: match S1 baseline misallocation

- [ ] **No Rounding Discrepancies**
  - [ ] Percentages rounded consistently (0–1 decimal place)
  - [ ] Ratios shown to 2 decimal places
  - [ ] No figures showing different precision for same metric

---

### Axis, Label & Legend Standards
- [ ] **X & Y Axes**
  - [ ] Axis labels are clear and descriptive (e.g., "Recovery Accuracy (%)" not "Rec")
  - [ ] Units included in axis labels where applicable
  - [ ] Axis scales are appropriate (not cutting off data)
  - [ ] No reversed axes unless intentional
  - [ ] Axis ranges cover full data + padding (not cramped)

- [ ] **Legends**
  - [ ] Legend present and readable
  - [ ] Legend labels match data series names
  - [ ] Legend positioned to not obscure data (upper right, lower left, etc.)
  - [ ] Legend font size consistent with rest of figure
  - [ ] No redundant legend entries

- [ ] **Titles & Captions**
  - [ ] Title is descriptive and specific (not generic "Figure 1")
  - [ ] Title indicates key finding or comparison
  - [ ] No spelling errors in titles
  - [ ] Title font bold or emphasized for visibility

---

## FIGURE-SPECIFIC EVALUATION

### Figure 1: Robustness Spectrum
**Purpose:** Show Tier 1/2/3 taxonomy via pause-window ratios

**Scale Checks:**
- [ ] X-axis (Pause-Window Ratio) range: 0.95–2.0 (covers all models)
- [ ] Y-axis: 10 models listed with consistent spacing
- [ ] BSTS at ~1.02 (Tier 1 - green)
- [ ] Kalman at ~1.345 (Tier 2/3 boundary - orange/red)
- [ ] geo_adstock, almon_pdl, weibull >1.35 (Tier 3 - red)

**Data Verification:**
- [ ] All 10 models present
- [ ] Pause-window ratios from S2 data
- [ ] Values match STEP3_ANOMALY_RESOLUTION.txt:
  - [ ] BSTS: 1.02×
  - [ ] Kalman: 1.345×
  - [ ] geo_adstock: 1.41× (approx)
  - [ ] almon_pdl: ~1.35×+

**Visual Checks:**
- [ ] Bars sorted by ratio (ascending left-to-right)
- [ ] Color gradient clear: green (robust) → orange → red (fragile)
- [ ] Tier boundary lines (1.10×, 1.35×) clearly marked
- [ ] Value labels on right side of bars
- [ ] Legend explains tier boundaries

---

### Figure 2: Cross-Scenario Heatmap
**Purpose:** Show recovery accuracy for all 10 models × 5 scenarios

**Scale Checks:**
- [ ] Color scale: 0–100% (red-yellow-green)
- [ ] All cells visible (no truncation)
- [ ] 10 rows (models) × 5 columns (scenarios)
- [ ] Consistent cell sizing

**Data Verification:**
- [ ] Recovery values from outputs/results/{model}_{scenario}.json
- [ ] Framework averages visible:
  - [ ] F3 block (green) shows 80%+ in most cells
  - [ ] F1 block (red) shows 0–70% range
  - [ ] F2 block (blue) shows 40–70% range
- [ ] S1 column shows framework hierarchy (F3 > F2 > F1)
- [ ] S2 column shows ARDL resurrection (68.8%)
- [ ] S3 column shows F1/F2 collapse, F3 stable

**Visual Checks:**
- [ ] Framework borders clearly distinguished (F1/F2/F3)
- [ ] Cell annotations (%) readable
- [ ] Color bar legend present with label "Recovery Accuracy (%)"
- [ ] Models and scenarios labeled on axes

---

### Figure 3: S2 Pause Window Detail
**Purpose:** Show recovery during pause weeks 104–112

**Scale Checks:**
- [ ] X-axis (weeks): 95–125 (30-week window)
- [ ] Y-axis (recovery %): 0–100% or appropriate range
- [ ] Pause period (104–112) clearly shaded
- [ ] 4 top models plotted (BSTS, Kalman, geo_adstock, MCMC)

**Data Verification:**
- [ ] Recovery values during pause from JSON week-by-week data
- [ ] Models show divergent behavior:
  - [ ] BSTS (green): stable ~80%
  - [ ] Kalman (green): stable ~82%
  - [ ] geo_adstock (red): increases during pause
  - [ ] MCMC (green): decreases during pause

**Visual Checks:**
- [ ] Pause period clearly marked (gray shading or hatch)
- [ ] Pre-pause, during-pause, post-pause trends visible
- [ ] Legend identifies all 4 models
- [ ] Grid enables week-by-week reading
- [ ] Title indicates time window and scenario

---

### Figure 4: Channel Attribution Comparison
**Purpose:** Show per-channel recovery (ARDL aggregate ≠ per-channel)

**Scale Checks:**
- [ ] X-axis (channels): TV, Search, Social, Display, Video (5 channels)
- [ ] Y-axis (recovery %): 0–100%
- [ ] 4 models grouped (ARDL, Koyck, BSTS, MCMC)
- [ ] Consistent bar widths within groups

**Data Verification:**
- [ ] ARDL shows ~68.8% aggregate but 0% Video
- [ ] Koyck shows inversion: Social ~59.3% > TV ~2.2%
- [ ] BSTS shows balanced recovery across channels
- [ ] MCMC shows correct channel ranking (TV/Video dominant)

**Visual Checks:**
- [ ] Color-coded by model (F2/F3 consistency)
- [ ] Legend clearly identifies 4 models
- [ ] 0-line visible on Y-axis for reference
- [ ] Title indicates scenario (S2)
- [ ] Grid on Y-axis for easy percentage reading

---

### Figure 5: Framework Hierarchy
**Purpose:** Show F3 >> F2 >> F1 performance distribution

**Scale Checks:**
- [ ] X-axis (frameworks): F1, F2, F3 (3 frameworks)
- [ ] Y-axis (recovery %): 0–100%
- [ ] All 50 data points visible (via box plot/violin/distribution)
- [ ] Consistent visual width per framework

**Data Verification:**
- [ ] F3 mean labeled: 78.4%
- [ ] F2 mean labeled: 42.8%
- [ ] F1 mean labeled: 22.4%
- [ ] Variance visible (F3 smallest, F1 largest)
- [ ] All 5 scenarios represented in distribution

**Visual Checks:**
- [ ] Color-coded by framework (red/blue/green)
- [ ] Mean values clearly labeled on plot
- [ ] Box/violin shows quartiles and outliers
- [ ] Y-axis grid visible
- [ ] Title emphasizes hierarchy

---

### Figure 6: Calibration Sensitivity
**Purpose:** Show Frozen vs Optimized recovery improvement

**Scale Checks:**
- [ ] X-axis (models): 6 top models listed
- [ ] Y-axis (recovery %): 0–100%
- [ ] Paired bars (frozen, optimized) for each model
- [ ] Improvement labeled above bars (+Xpp)

**Data Verification:**
- [ ] Frozen values from S1 baseline:
  - [ ] BSTS: 82.4%
  - [ ] Kalman: 82.0%
  - [ ] MCMC: 72.6%
  - [ ] geo_adstock: 69.9%
  - [ ] finite_dl: 50.3%
  - [ ] koyck: 46.4%
- [ ] Optimized values from STEP4_OPTIMIZATION_RESULTS.md
- [ ] Improvements match: F3 +2–3pp, F2 +5–10pp, F1 +<2pp

**Visual Checks:**
- [ ] Framework colors consistent (F1/F2/F3)
- [ ] Frozen bars lighter/transparent; optimized bars fully opaque
- [ ] Improvement labels (+Xpp) clearly visible
- [ ] Models sorted by some criterion (alphabetical or improvement)
- [ ] Y-axis shows full range (0–100%)
- [ ] Legend distinguishes frozen vs optimized

---

### Figure 7: Scenario Difficulty Ranking
**Purpose:** Show which scenarios are hardest for models

**Scale Checks:**
- [ ] X-axis (scenarios): S1–S5 (or sorted by difficulty)
- [ ] Y-axis (avg recovery %): 0–100%
- [ ] Error bars show std dev or variance

**Data Verification:**
- [ ] Average recovery per scenario calculated correctly
- [ ] S1: ~65–70% (easy)
- [ ] S2: ~60–65% (moderate)
- [ ] S3: ~50–55% (hard, seasonality)
- [ ] S4: ~55–60% (hard, structural break)
- [ ] S5: ~30–40% (very hard, weak signal)

**Visual Checks:**
- [ ] Color gradient: green (easy) → orange (moderate) → red (hard)
- [ ] Error bars visible and correctly sized
- [ ] Y-axis grid present
- [ ] Title indicates what "difficulty" means

---

### Figure 8: Pause-Window Error Timeline
**Purpose:** Show error accumulation during S2 pause weeks

**Scale Checks:**
- [ ] X-axis (weeks): 95–125
- [ ] Y-axis (cumulative error): 1.0–max observed ratio
- [ ] Pause period (104–112) clearly marked
- [ ] 4 top models plotted (BSTS, Kalman, geo, MCMC)

**Data Verification:**
- [ ] Error trajectory matches framework robustness:
  - [ ] BSTS: nearly flat (~1.02 final)
  - [ ] Kalman: gradual increase (~1.345 final)
  - [ ] geo_adstock: steep increase (~1.41+ final)
  - [ ] MCMC: moderate increase
- [ ] Pre-pause baseline: all models at ~1.0

**Visual Checks:**
- [ ] Each model color-coded by framework
- [ ] Markers (dots) at weeks for readability
- [ ] Pause period shaded or hatched
- [ ] Legend identifies all 4 models
- [ ] Reference line at 1.0 (no error baseline)
- [ ] Grid on both axes

---

### Figure 9: Channel-Level Detail (2×2 Small Multiples)
**Purpose:** Show per-channel recovery for top 4 models across scenarios

**Scale Checks:**
- [ ] Each subplot: 5 rows (scenarios) × 5 columns (channels)
- [ ] All subplots use same color scale (0–100%)
- [ ] 4 subplots: BSTS, MCMC, ARDL, Koyck

**Data Verification:**
- [ ] ARDL subplot shows Video column all 0% or near 0%
- [ ] MCMC subplot shows balanced channel recovery
- [ ] Koyck subplot shows Social/TV inversion in some cells
- [ ] BSTS subplot shows strong TV/Video, weak others

**Visual Checks:**
- [ ] All 4 subplots use identical color scale
- [ ] Channel labels (TV, Search, Social, Display, Video) consistent
- [ ] Scenario labels (S1–S5) consistent across subplots
- [ ] Values annotated in cells (%)
- [ ] Single shared color bar for all 4 subplots

---

### Figure 10: Video LTC Signal Loss Pattern
**Purpose:** Diagnostic - show Video LTC recovery collapse in F1/F2

**Scale Checks:**
- [ ] X-axis (scenarios): S1–S5
- [ ] Y-axis (Video recovery %): 0–100% (or −20 to 110 with margin)
- [ ] 10 models plotted (separate lines)

**Data Verification:**
- [ ] F3 models (green lines): 70%+ across most scenarios
- [ ] F1/F2 models (red/blue lines): 0% in S3–S5
- [ ] Gradient should show collapse from S1/S2 → S3/S4/S5

**Visual Checks:**
- [ ] Framework colors consistent (F1=red, F2=blue, F3=green)
- [ ] Solid lines for F3, dashed lines for F1/F2 (visual distinction)
- [ ] Legend shows at least 1–2 models per framework
- [ ] Reference line at 0% (signal loss baseline)
- [ ] Y-axis extends below 0 to show collapse below baseline if present

---

### Figure 11: MCMC Convergence (R-hat Values)
**Purpose:** Validate method - show MCMC convergence across scenarios

**Scale Checks:**
- [ ] X-axis (scenarios): S1–S5
- [ ] Y-axis (R-hat): 0.99–1.15 (gold standard <1.05)
- [ ] 5 bars (one per scenario)

**Data Verification:**
- [ ] All R-hat values <1.05 (excellent convergence)
- [ ] No R-hat > 1.10 (acceptable threshold)
- [ ] Consistent convergence across all 5 scenarios

**Visual Checks:**
- [ ] Color-coded by convergence quality:
  - [ ] <1.05 = green (excellent)
  - [ ] 1.05–1.10 = orange (good)
  - [ ] >1.10 = red (poor)
- [ ] Reference lines at 1.05 and 1.10 clearly marked
- [ ] Y-axis labeled "R-hat (Convergence Diagnostic)"
- [ ] Legend explains reference thresholds

---

### Figure 12: Budget Allocation Error by Model
**Purpose:** Show financial risk - which models misallocate most

**Scale Checks:**
- [ ] X-axis (error %): −30% to +30% (or observed range)
- [ ] Y-axis (models): 10 models listed
- [ ] Center line at 0% (no error baseline)

**Data Verification:**
- [ ] Errors from S1 baseline (first scenario)
- [ ] Models sorted by magnitude of error (largest first)
- [ ] F3 models have smaller errors (<±10%)
- [ ] F1 models have larger errors (±10–50%+)
- [ ] dual_adstock: catastrophic error (~±79%)

**Visual Checks:**
- [ ] Color-coded by framework
- [ ] Bars extend left (negative) and right (positive)
- [ ] 0% reference line visible and clear
- [ ] Model names readable (left side)
- [ ] Grid on X-axis for easy percentage reading

---

### Figure 13: Robustness Taxonomy Visualization
**Purpose:** Show Tier 1/2/3 placement with recovery vs pause-ratio

**Scale Checks:**
- [ ] X-axis (pause-window ratio): 0.95–1.60
- [ ] Y-axis (S1 recovery %): 0–100%
- [ ] Tier boundaries marked: 1.10×, 1.35×
- [ ] Point sizes proportional to S1 average recovery

**Data Verification:**
- [ ] BSTS: ~(1.02, 82%) — Tier 1 (green)
- [ ] Kalman: ~(1.345, 82%) — Tier 2/3 boundary (orange/red)
- [ ] geo_adstock: ~(1.41, 70%) — Tier 3 (red)
- [ ] ARDL: ~(1.30+, 0% in S1) — Tier 3 (red)
- [ ] All 10 models visible and positioned correctly

**Visual Checks:**
- [ ] Tier zones clearly labeled:
  - [ ] "Tier 1 Robust" (left zone, 0.95–1.10)
  - [ ] "Tier 2 Sensitive" (middle zone, 1.10–1.35)
  - [ ] "Tier 3 Fragile" (right zone, >1.35)
- [ ] Key models annotated (bsts, ardl, geo_adstock)
- [ ] Point sizes scaled by recovery magnitude
- [ ] Legend shows framework colors
- [ ] Grid visible for coordinate reading

---

### Figure A: Ranking Reversals
**Purpose:** Show model rank changes across scenarios

**Scale Checks:**
- [ ] X-axis (model rank): 1–10 (top to bottom)
- [ ] Y-axis (recovery %): 0–100%
- [ ] 5 lines (one per scenario)

**Data Verification:**
- [ ] Rank 1 = highest recovery per scenario
- [ ] Rank 10 = lowest recovery per scenario
- [ ] Crossing lines show reversals
- [ ] F3 models consistently in top ranks
- [ ] Dual_adstock and ARDL (S1) at bottom

**Visual Checks:**
- [ ] Each scenario color-coded
- [ ] Lines labeled with scenario names in legend
- [ ] Grid visible for rank/recovery intersection
- [ ] X-axis inverted (rank 1 on left) for intuitive reading

---

### Figure B: Scenario Characteristics
**Purpose:** Show scenario complexity attributes

**Scale Checks:**
- [ ] 5 scenarios (S1–S5) on X-axis
- [ ] 5 characteristics on Y-axis (Signal, Variation, Seasonality, Collinearity, Break)
- [ ] Color scale: 0–100 (red = low, yellow = medium, green = high)

**Data Verification:**
- [ ] S1: High signal, moderate variation, low seasonality/break
- [ ] S2: Moderate signal, high variation, high break (100)
- [ ] S3: Moderate signal, moderate variation, high seasonality
- [ ] S4: Moderate signal, high variation, high seasonality + moderate break
- [ ] S5: Low signal (30), low variation, moderate characteristics

**Visual Checks:**
- [ ] Color bar legend present (0–100 intensity)
- [ ] Values annotated in cells
- [ ] Characteristics labeled on Y-axis
- [ ] Scenarios labeled on X-axis

---

### Figure C: Framework Comparison Matrix
**Purpose:** Quick decision guide for framework selection

**Scale Checks:**
- [ ] 3 frameworks (F1, F2, F3) on Y-axis
- [ ] 5 metrics on X-axis (Recovery, Robustness, Channels, Cost, Tuning)
- [ ] Color scale: 0–10 (red = poor, yellow = medium, green = excellent)

**Data Verification:**
- [ ] F1: Low recovery (22.4%), low robustness, poor channels, fast, low tuning benefit
- [ ] F2: Medium recovery (42.8%), medium robustness, mixed channels, moderate cost, high tuning
- [ ] F3: High recovery (78.4%), high robustness, excellent channels, high cost, low tuning benefit

**Visual Checks:**
- [ ] Values clearly annotated in cells
- [ ] Framework names on Y-axis
- [ ] Metrics on X-axis
- [ ] Color bar legend present (0–10 performance scale)

---

## CROSS-FIGURE CONSISTENCY CHECKS

### Color Scheme (All 14 Figures)
- [ ] **Framework Colors Consistent:**
  - [ ] F1 = Red (#d62728) in figures: 1, 2, 4, 5, 6, 10, 12, 13
  - [ ] F2 = Blue (#1f77b4) in figures: 1, 2, 4, 5, 6, 10, 12, 13
  - [ ] F3 = Green (#2ca02c) in figures: 1, 2, 4, 5, 6, 10, 12, 13
  - [ ] No color swaps or inconsistencies across figures

- [ ] **Scenario Colors (where applicable):**
  - [ ] Pause period shading: gray (figures 3, 8)
  - [ ] Tier boundaries: orange (1.10), red (1.35) (figures 1, 13)
  - [ ] Performance gradients: consistent red-yellow-green scales (figures 2, 7, 9, B, C)

### Scale Consistency
- [ ] **Recovery Accuracy Scales:**
  - [ ] Figures 2, 4, 5, 7, 9 all use 0–100% Y-axis
  - [ ] No figure unexpectedly truncates at 80% or 90%
  - [ ] All show same reference points (0%, 50%, 100%)

- [ ] **Pause-Window Ratio Scales:**
  - [ ] Figure 1: 0.95–2.0 (covers all models)
  - [ ] Figure 13: 0.95–1.60 (sufficient for 10 models)
  - [ ] Consistent axis ranges where applicable

### Annotation Style
- [ ] **Number Format Consistency:**
  - [ ] Percentages: 0–1 decimal place (e.g., 78.4%, not 78.35%)
  - [ ] Ratios: 2 decimal places (e.g., 1.02×, not 1.020×)
  - [ ] No mixing of formats within related figures

- [ ] **Legend Placement:**
  - [ ] Legends positioned to not obscure data
  - [ ] Consistent placement strategy (e.g., upper right preferred)
  - [ ] Font size consistent with rest of figure

---

## FINAL QUALITY ASSURANCE

### Publication Readiness
- [ ] All 14 figures meet publication standards
- [ ] No figures show obvious errors or inconsistencies
- [ ] All figures are self-contained (readable without text)
- [ ] Caption information sufficient for standalone viewing

### Narrative Coherence
- [ ] Figures 1–4: Tell core story (hierarchy, robustness, diagnosis)
- [ ] Figures 5–13: Deepen understanding (calibration, signal loss, etc.)
- [ ] Figures A–C: Support different audiences

### Data Integrity
- [ ] Zero data mismatches between figures and source files
- [ ] All metrics traceable to comprehensive_analysis or outputs/results/
- [ ] No figures contradict each other

---

## VALIDATION RESULTS

**Date of Evaluation:** [TO BE FILLED]

### Overall Status
- [ ] **PASS** — All 14 figures meet quality standards, ready for paper
- [ ] **CONDITIONAL PASS** — Minor issues (list below), easy fixes
- [ ] **NEEDS REVISION** — Major issues (list below), requires rework

### Issues Found (if any)
1. [Issue description, figure number, fix required]
2. [Issue description, figure number, fix required]
3. [Issue description, figure number, fix required]

### Fixes Applied
1. [Fix description, date, figures affected]
2. [Fix description, date, figures affected]

### Sign-Off
- [ ] All checks complete
- [ ] All issues resolved
- [ ] Figures approved for paper submission

**Evaluated by:** [Name]  
**Date:** [Date]  
**Version:** 1.0

---

## USAGE INSTRUCTIONS

### How to Use This Checklist

1. **Read this document** completely before starting evaluation
2. **For each figure** (1–14, A–C), check all figure-specific boxes
3. **For cross-figure checks**, verify consistency across all 14 figures
4. **Note any issues** in the "Issues Found" section with:
   - Which figure(s) affected
   - What needs fixing
   - Recommended fix
5. **After fixes applied**, re-check affected figures
6. **Sign off** at bottom when all items checked and issues resolved

### Priority Levels

**Critical (must fix before submission):**
- Data mismatches (wrong recovery %, wrong models)
- Color scheme violations (wrong F1/F2/F3 colors)
- Missing titles, legends, or axis labels

**Important (should fix before submission):**
- Scale inconsistencies between related figures
- Illegible text or labels
- Inconsistent number formatting

**Nice-to-have (can fix if time allows):**
- Minor aesthetic improvements
- Legend repositioning
- Grid adjustments

