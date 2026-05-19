# visuals.md
## Tables and Figures

Read this whenever a section requires a table or figure.

---

## Table Standards (IJRM and AMA)

- Submit tables as editable text, not images
- Title above the table, note below
- Number consecutively in order of appearance
- No vertical rules. No cell shading.
- Every table must be cited in the text before it appears
- Do not duplicate in a table data already described fully in prose

**Table title format:**
TABLE [N]
[Descriptive title in title case]

**Note format (below table):**
*Note.* [Explanation of abbreviations, data source, symbols.]

**Significant figures:**
- Recovery %: one decimal place (82.4%)
- MAPE %: one decimal place (17.6%)
- Ratios: two decimal places (1.02×)
- Dollar figures: two decimal places ($0.49M)

**Bold:** best performer per scenario/column
**Italics:** use sparingly for emphasis within a cell

---

## Required Tables

**Table 1 — Scenario Library**
Columns: Scenario ID | Name | Key structural property | Diagnostic purpose | Avg media % | Noise std

**Table 2 — Parameter Specifications**
Columns: Channel | True STC decay (λ) | True LTC delta (δ) | Half-life (wks) | Assumed/Estimated

**Table 3 — Full Recovery Matrix**
Rows: 10 models | Columns: S1–S5 + Average S1–S4
Bold best per column. Flag critical failures (0%, negatives) with dagger symbol †.

**Table 4 — Pause-Window Robustness Ratios**
Columns: Model | Framework | Full MAPE | Pause MAPE | Ratio | Tier
Sort ascending by ratio. Include tier classification column.

**Table 5 — Channel Attribution Summary**
Show for S2, S3, S4. Models: MCMC, BSTS, Kalman, Koyck, ARDL.
Columns per model: TV recovery | Video recovery | Social recovery | Rank correct?

**Table 6 — Budget Allocation Error**
Columns: Model | TV error | Video error | Social error | Display error | Search error
Positive = over-allocated, negative = under-allocated.

---

## Required Figures

**Figure 1 — Robustness Spectrum**
Bar chart. All 10 models sorted ascending by pause-window ratio.
Reference line at 1.0 = "perfect scenario invariance".
Colour by framework: F1 one colour, F2 another, F3 another.
Caption: "FIGURE 1. Pause-window robustness ratio for all estimation methods.
Ratio = pause-window MAPE / full-series MAPE. Values near 1.0 indicate
error invariance to spend discontinuity."

**Figure 2 — Cross-Scenario Heatmap**
10 rows (models) × 5 columns (scenarios). Colour = recovery %.
Green = high recovery, red = low/negative.
Caption explains that S5 universal collapse reflects frozen parameter design.

**Figure 3 — S2 Pause Window Detail**
Line chart, weeks 95–125.
Lines: true LTC, BSTS recovered, Kalman recovered, geo_adstock recovered.
Shaded region: spend pause window (weeks 104–112).
Caption: "FIGURE 3. True and recovered TV LTC during and after the spend
pause (shaded region, weeks 104–112). State-space methods maintain LTC
estimation through the pause; static adstock collapses."

**Figure 4 — Channel Attribution Comparison**
Bar chart per channel (5 channels on x-axis).
Grouped bars: true LTC share, MCMC recovered, Koyck recovered, ARDL recovered.
Show for S3 (most striking channel failure scenario).

---

## Figure Caption Format
FIGURE [N]. [One sentence describing what is shown.] [One sentence on what to observe.]

---

## Accessibility
- Use colourblind-safe palette (avoid red-green only distinction)
- All figures legible in greyscale (for print)
- Minimum 300 DPI for submission
