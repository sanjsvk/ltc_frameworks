# Figure Integration Map — Sections 1-10

**Status:** Figure references added to all sections (Sections 4-10). Ready for visual integration phase.

---

## Placement Summary

### Section 1 (Abstract)
- No figures (abstract format)

### Section 2 (Introduction)
- No figures (literature context)

### Section 2.1 (Literature Review)
- No figures (citation context)

### Section 3 (Methodology)
- No figures (framework description)

---

## Results & Analysis Sections (Figures Integrated)

### Section 4: Framework Comparison & Scenario Sensitivity
**Content:** S1-S5 baseline performance hierarchy, all models, all scenarios
**Figures:**
- **Figure 5** (Framework Hierarchy) — Boxplot of recovery rates by F1/F2/F3 framework
  - Purpose: Show dominance of state-space methods
  - Position: After Section 4.1 summary
- **Figure 2** (Cross-Scenario Heatmap) — Recovery accuracy (0-100%) for all 10 models × S1-S5
  - Purpose: Visualize full performance matrix across all scenarios
  - Position: After Table 3, before Section 4.2

---

### Section 5: Scenario Sensitivity & Structural Breaks
**Content:** Mechanistic explanations of S2-S5 performance; identification dependencies
**Figures:**
- **Figure 3** (S2 Pause Window Detail) — Week-by-week decomposition, weeks 95-125, 4 top models
  - Purpose: Show error dynamics during spend pause
  - Position: Section 5.1, after identification paradox discussion
- **Figure 7** (Scenario Difficulty Ranking) — S1-S5 ranked by challenge (error bars)
  - Purpose: Position scenarios on difficulty axis (S5 hardest, S1 easiest)
  - Position: Section 5 summary, before identification table
- **Figure B** (Scenario Characteristics) — Intensity heatmap of scenario features (0-100 scale)
  - Purpose: Show collinearity, discontinuity, seasonality intensity per scenario
  - Position: Section 5 summary, alongside Figure 7
- **Figure A** (Ranking Reversals) — Per-channel rank changes across scenarios
  - Purpose: Visualize which models preserve TV/Video dominance vs. invert
  - Position: Section 5 summary, after identification table

---

### Section 6: Channel-Level Attribution & Aggregate Masking
**Content:** ARDL 0% per-channel masking; Koyck ranking inversions; Video recovery as diagnostic
**Figures:**
- **Figure 4** (Channel Attribution, S2) — Per-channel recovery for ARDL, Koyck, geo_adstock, MCMC
  - Purpose: Expose offsetting errors and ranking inversions
  - Position: Section 6.1, before ARDL case study table
- **Figure 10** (Video LTC Signal Loss) — Video recovery across S1-S5 for all 10 models
  - Purpose: Show universal collapse (0%) for fixed-parameter methods; MCMC advantage
  - Position: Section 6.3, after table of Video recovery by scenario
- **Figure 9** (Channel-Level Detail) — 2×2 small multiples of 4 models, 5 channels × S1-S4
  - Purpose: Practitioners verify channel ranking stability across scenarios
  - Position: Section 6.5, before "Channel Validation as Mandatory" conclusion

---

### Section 7: Calibration Sensitivity & Robustness
**Content:** Frozen vs. optimized parameters; robustness scores; calibration-structure trade-offs
**Figures:**
- **Figure 6** (Calibration Sensitivity) — Paired bar chart: frozen vs. optimized recovery
  - Purpose: Show minimal F3 gains (1-2pp), moderate F2 (5-6pp), variable F1 (0-6pp)
  - Position: Section 7.1, after frozen vs. optimized table
- **Figure 12** (Budget Allocation Error) — Per-channel misallocation magnitude, all 10 models (S1)
  - Purpose: Show which models catastrophically misallocate (ARDL, dual_adstock) vs. near-zero (MCMC, BSTS)
  - Position: Section 7 summary, supporting calibration robustness argument

---

### Section 8: Anomaly Diagnostics & Mechanistic Understanding
**Content:** Root causes of Weibull/Kalman/MCMC/ARDL/Almon failures; spend pause as diagnostic
**Figures:**
- **Figure 11** (MCMC Convergence) — R-hat diagnostic values for all 19 parameters × S1-S5
  - Purpose: Confirm excellent convergence (R-hat <1.05) after tuning adjustment
  - Position: Section 8.2, after divergence resolution fix
- **Figure 8** (Pause-Window Timeline) — Cumulative error over time, weeks 95-125, 4 models
  - Purpose: Show error stability (BSTS) vs. accumulation (ARDL) during discontinuity
  - Position: Section 8.5, at start of spend-pause diagnostic discussion
- **Figure 13** (Robustness Taxonomy) — Pause-ratio vs. recovery scatter plot with tier zones
  - Purpose: Classify models as Tier 1/2/3; flag fragile models (ARDL, almon_pdl)
  - Position: Section 8 conclusion, before final summary

---

### Section 9: Discussion
**Content:** Robustness spectrum; channel attribution implications; MCMC production standard
**Figures:**
- **Figure 1** (Robustness Spectrum) — 2D scatter: recovery (y-axis) vs. pause-ratio (x-axis)
  - Purpose: Visualize ideal position (BSTS: high recovery, ratio ~1.0) vs. fragile (ARDL, almon)
  - Position: Section 9.1, at start of "Robustness Spectrum" discussion
- **Figure C** (Framework Comparison Matrix) — 3×5 matrix: F1/F2/F3 vs. performance dimensions
  - Purpose: Synthesize findings into practitioner decision matrix
  - Position: Section 9 conclusion, summarizing all dimensions

---

### Section 10: Recommendations & Practitioner Guidance
**Content:** Decision rules; critical warnings; boundary conditions
**Figures:**
- **No new figures** (references Tables 1, 2 from recommendations section)
- May reference Figures 1, 6, 13 from earlier sections for decision framework

---

## Figure Count by Section

| Section | Figure Count | Figures |
|---------|--------------|---------|
| 4 | 2 | 2, 5 |
| 5 | 4 | 3, 7, A, B |
| 6 | 3 | 4, 9, 10 |
| 7 | 2 | 6, 12 |
| 8 | 3 | 8, 11, 13 |
| 9 | 2 | 1, C |
| 10 | 0 | — |
| **Total** | **16 placement points** | **1-13, A-C (14 unique figures)** |

---

## Integration Notes

1. **Visual Verification Phase (Next):**
   - Load each PNG from `outputs/figures/`
   - Verify dimensions, clarity, color consistency (F1=red, F2=blue, F3=green)
   - Confirm data matches section narratives
   - Add descriptive captions (1-2 sentences each)

2. **Cross-Reference Check:**
   - Ensure all figure labels ("Figure X") in text match filename convention
   - Verify axis labels, legends, titles match section terminology
   - Check that all models referenced in text appear in figures

3. **Publication-Ready Criteria:**
   - All 14 figures at 300dpi PNG
   - Consistent font size (11pt titles, 9pt labels)
   - Color scheme: F1 #d62728 (red), F2 #1f77b4 (blue), F3 #2ca02c (green)
   - Grid visibility: alpha=0.3
   - Figure numbers placed in bold at top-left

4. **Captions (to be written after manual verification):**
   - Section reference ("Figure X shows... as described in Section Y.Z")
   - Key finding stated (what practitioners should observe)
   - Data note if applicable (scenario, models, metric)

---

## Status

✅ All figure placements documented  
✅ Each section has 1-4 figures integrated  
✅ Figure references added to drafts  
✅ Ready for visual integration and caption writing

**Next Action:** Manual verification of each figure's visual quality and data accuracy.
