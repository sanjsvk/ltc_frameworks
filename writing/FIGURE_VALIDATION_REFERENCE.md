# Figure Validation Reference — Quick Lookup

**Purpose:** For each figure in `outputs/figures/`, know exactly where to find validation data.

**How to use:**
1. Open figure file from `outputs/figures/`
2. Look up filename below
3. Go to specified section file and line numbers
4. Spot-check 3-4 values
5. Update caption if needed

---

## Figure Files & Validation Data Locations

### **Figure 1: robustness_spectrum.png**
**Location to validate:** Section 9 (SECTION_9_DISCUSSION_DRAFT.md)
- Look for: "Figure 1 (Robustness Spectrum)" reference
- Validate against: Table in Section 7 (line 37-44, Robustness Score Table)
  - BSTS: 80.5% recovery, 2.4pp StdDev, score 78.6
  - ARDL: 28.1% recovery, 38.9pp StdDev, score 20.2
  - Kalman: 76.4% recovery, 8.4pp StdDev, score 70.5
- **Check:** X-axis (pause-ratio) ranges 1.0-1.5×, Y-axis (recovery) 0-100%

---

### **Figure 2: cross_scenario_heatmap.png**
**Location to validate:** Section 4 (RESULTS_SECTION_4_DRAFT.md, lines 83-98)
- Table 3: Full Recovery Matrix
- All 10 models × 5 scenarios (S1-S5)
- **Spot-check values:**
  - BSTS: S1=82.4%, S2=81.0%, S3=76.8%, S4=81.6%, S5=0.0%
  - ARDL: S1=0.0%, S2=68.8%, S3=63.3%, S4=-19.8%, S5=0.0%
  - geo_adstock: S1=69.9%, S2=83.1%, S3=43.2%, S4=63.4%, S5=0.0%
- **Check:** Color gradient 0% (red/dark) to 100% (green/light), all 10 models labeled

---

### **Figure 3: s2_pause_window_detail.png**
**Location to validate:** Section 5 (RESULTS_SECTION_5_DRAFT.md, lines 9-24)
- S2 Spend Pause table (lines 13-18)
- Time window: weeks 95-125, pause at 104-112
- **Spot-check:**
  - geo_adstock: S1→S2 improvement +13.2pp
  - ARDL: S1→S2 improvement +68.8pp
  - BSTS: S1→S2 minimal change -1.4pp
- **Check:** Week-by-week trajectory shown, pause window (104-112) clearly marked

---

### **Figure 4: channel_attribution_s2.png**
**Location to validate:** Section 6 (RESULTS_SECTION_6_DRAFT.md)
- **ARDL S2 table (lines 15-22):**
  - TV: 0%, Video: 0%, Social: 0%, Display: 0%, Search: 0%
  - Aggregate: 68.8%
- **Koyck S2 table (lines 32-38):**
  - TV: 2.2% (rank 5), Video: 14.9% (rank 3), Social: 50.4% (rank 1), Display: 59.3% (rank 2), Search: 0%
- **MCMC S2 (implied from context):**
  - TV dominant (highest %), Video second, Social/Display/Search lower
- **Check:** 4 models shown, 5 channels per model, bars or grouped display, S2 scenario label

---

### **Figure 5: framework_hierarchy.png**
**Location to validate:** Section 4 (RESULTS_SECTION_4_DRAFT.md)
- Table 3 (lines 83-98): Full Recovery Matrix, S1 baseline
- **F1 models (red):** geo_adstock 69.9%, almon_pdl 42.6%, weibull_adstock 10.5%, dual_adstock 0.0%
- **F2 models (blue):** finite_dl 50.3%, koyck 46.4%, ardl 0.0%
- **F3 models (green):** bsts 82.4%, kalman_dlm 82.0%, mcmc_stock 72.4%
- **Check:** Boxplot shows F3 median ~82%, F2 ~50%, F1 ~40%, clear separation

---

### **Figure 6: calibration_sensitivity.png**
**Location to validate:** Section 7 (RESULTS_SECTION_7_DRAFT.md, lines 13-24)
- Frozen vs Optimized Comparison table
- **Spot-check:**
  - BSTS: frozen 82.4% → optimized 84.1% (+1.7pp)
  - ARDL: frozen 0.0% → optimized 6.2% (+6.2pp)
  - geo_adstock: frozen 69.9% → optimized 72.0% (+2.1pp)
  - weibull_adstock: frozen 10.5% → optimized 10.7% (+0.2pp)
- **Check:** Paired bars (frozen & optimized side-by-side), improvement labels shown

---

### **Figure 7: scenario_difficulty_ranking.png**
**Location to validate:** Section 5 (RESULTS_SECTION_5_DRAFT.md)
- Scenario Sensitivity discussion (lines 37-94)
- **Difficulty order (hardest to easiest):**
  - S5 (Weak Signal) — all models 0% with frozen params
  - S3 (High Seasonality) — F1 avg 20.9%
  - S4 (Structural Break) — ARDL -19.8%
  - S2 (Spend Pause) — mixed (geo +13.2pp, ARDL +68.8pp)
  - S1 (Baseline) — F3 avg 75.7%
- **Check:** Scenarios ranked on x-axis, error bars or difficulty score on y-axis

---

### **Figure 8: pause_window_timeline.png**
**Location to validate:** Section 8 (RESULTS_SECTION_8_DRAFT.md, lines 63-73)
- S2 Spend Pause as Diagnostic (lines 63-73)
- Time window: weeks 95-125, pause 104-112
- **Model behaviors:**
  - BSTS: error stable (~1.02× ratio)
  - ARDL: error accumulates during pause
  - geo_adstock: error improves during pause
  - MCMC: error slightly increases
- **Check:** Cumulative error plot, weeks 95-125 on x-axis, pause window shaded/marked

---

### **Figure 9: channel_level_detail.png**
**Location to validate:** Section 6 (RESULTS_SECTION_6_DRAFT.md, lines 65-70)
- MCMC Channel Stability table
- **S1–S4 channel ranking (MCMC):**
  - S1: TV(92%) > Video(77%) > Social(61%) > Display(14%) > Search(0%)
  - S2: TV(79%) > Video(68%) > Social(45%) > Display(9%) > Search(0%)
  - S3: TV(92%) > Social(70%) > Video(56%) > Display(9%) > Search(0%)
  - S4: TV(87%) > Social(51%) > Video(46%) > Display(5%) > Search(0%)
- **Check:** 2×2 grid (4 models), each shows 5 channels × 4 scenarios (S1-S4)

---

### **Figure 10: video_ltc_signal_loss.png**
**Location to validate:** Section 6 (RESULTS_SECTION_6_DRAFT.md, lines 48-57)
- Video LTC Recovery table (lines 48-55)
- **By scenario (Video recovery only):**
  - S3: MCMC 56%, Kalman 0%, BSTS 0%, Koyck 5%, ARDL 0%, geo_adstock 0%
  - S4: MCMC 46%, Kalman 0%, BSTS 0%, Koyck 0%, ARDL 0%, geo_adstock 5%
  - S5: MCMC 71%, Kalman 0%, BSTS 0%, Koyck 0%, ARDL 0%, geo_adstock 0%
- **Check:** 10 models shown, grouped by framework (F1=red, F2=blue, F3=green), MCMC clearly above others

---

### **Figure 11: mcmc_convergence.png**
**Location to validate:** Section 8 (RESULTS_SECTION_8_DRAFT.md, lines 31-37)
- MCMC Divergence Resolution (lines 31-37)
- **Key values:**
  - S1 divergences: 23 → 0 (after tuning adjustment)
  - All R-hat <1.05 after fix
  - Recovery: 72.6% (unchanged)
- **Check:** R-hat values all <1.05 across all 5 scenarios (S1-S5), no horizontal red line at 1.05 threshold crossed

---

### **Figure 12: budget_allocation_error.png**
**Location to validate:** Section 7 (RESULTS_SECTION_7_DRAFT.md, lines 76-78)
- Budget Allocation Error reference (see Section 3 Methodology Eq 8)
- **Models by error magnitude (worst to best):**
  - Worst: dual_adstock (catastrophic), ARDL (high)
  - Mid: geo_adstock, almon_pdl (moderate)
  - Best: MCMC, BSTS (near-zero)
- **Check:** 10 models sorted by magnitude, color-coded by framework

---

### **Figure 13: robustness_taxonomy.png**
**Location to validate:** Section 8 (RESULTS_SECTION_8_DRAFT.md, lines 79-85)
- Anomaly Summary Table (lines 79-85)
- Plus Table 3 from Section 4 (pause-window ratios)
- **Tier zones:**
  - Tier 1 (<1.10×): BSTS ~1.02, Kalman ~1.02
  - Tier 2 (1.10–1.35×): ARDL, finite_dl, koyck
  - Tier 3 (>1.35×): geo_adstock ~1.41, almon_pdl >1.35
- **Check:** Scatter plot, pause-ratio (x-axis) 1.0–1.5×, recovery (y-axis) 0–100%, tier zones shaded

---

### **Figure A: ranking_reversals.png**
**Location to validate:** Section 5 (RESULTS_SECTION_5_DRAFT.md, lines 49-51)
- Ranking Reversals reference
- Section 6 (Koyck ranking inversion, lines 32-38)
- **Ground truth ranking:** TV > Video > Social > Display > Search
- **Model ranking changes:**
  - MCMC: preserves correct ranking (or minor swap)
  - Koyck: inverts (Social/Display > TV/Video)
  - geo_adstock: varies by scenario
  - ARDL: channel-level failures mask
- **Check:** Rank changes visible across S1-S5, color-coded by model

---

### **Figure B: scenario_characteristics.png**
**Location to validate:** Section 5 (RESULTS_SECTION_5_DRAFT.md, lines 37-94)
- Scenario Identification table (lines 88-94)
- Section 3 Methodology (S1-S5 descriptions, lines 29-40)
- **Scenario features (0-100 intensity):**
  - S1 (Baseline): low collinearity, low discontinuity, low seasonality
  - S2 (Spend Pause): no collinearity change, high discontinuity, no seasonality change
  - S3 (High Seasonality): high collinearity, no discontinuity, high seasonality
  - S4 (Structural Break): medium collinearity, high discontinuity, low seasonality
  - S5 (Weak Signal): low collinearity, no discontinuity, low seasonality
- **Check:** Heatmap 0-100 scale, 3 features × 5 scenarios visible

---

### **Figure C: framework_comparison_matrix.png**
**Location to validate:** Section 9 (SECTION_9_DISCUSSION_DRAFT.md)
- Framework Comparison Matrix reference
- Section 7 & 8 summary tables for context
- **Matrix dimensions (F1/F2/F3 vs. 5 performance dimensions):**
  - F3 high on: average recovery, pause-window robustness, channel validation, production readiness
  - F2 moderate on: average recovery, calibration sensitivity
  - F1 low/variable on: most dimensions
- **Check:** 3×5 matrix clearly labeled, color-coded by framework

---

## Quick Validation Workflow

```
For each figure:

1. Open: outputs/figures/[filename].png
2. Look up: this document (find filename)
3. Go to: Section file + line numbers listed
4. Find: Table or narrative section
5. Spot-check: 3-4 values from table match figure
6. Status: ✅ Pass (matches) or ⚠️ Review (needs update)
7. Note: Any discrepancies for caption updates
```

---

## Status Tracking

As you verify each figure, mark here:

- [ ] Figure 1: robustness_spectrum.png
- [ ] Figure 2: cross_scenario_heatmap.png
- [ ] Figure 3: s2_pause_window_detail.png
- [ ] Figure 4: channel_attribution_s2.png
- [ ] Figure 5: framework_hierarchy.png
- [ ] Figure 6: calibration_sensitivity.png
- [ ] Figure 7: scenario_difficulty_ranking.png
- [ ] Figure 8: pause_window_timeline.png
- [ ] Figure 9: channel_level_detail.png
- [ ] Figure 10: video_ltc_signal_loss.png
- [ ] Figure 11: mcmc_convergence.png
- [ ] Figure 12: budget_allocation_error.png
- [ ] Figure 13: robustness_taxonomy.png
- [ ] Figure A: ranking_reversals.png
- [ ] Figure B: scenario_characteristics.png
- [ ] Figure C: framework_comparison_matrix.png
