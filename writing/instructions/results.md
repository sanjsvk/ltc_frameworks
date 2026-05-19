# sections/results.md
## Results

Target length: 2000–2500 words. Structured around scenarios, not models.

Source all numbers from: experiment_log.csv, paper_notes.md, channel attribution files.

---

## Structure

### 4.1 Performance Ceiling — S1 Clean Benchmark
Establishes the best-case performance of each framework.
Key numbers: F3 avg 78%+ recovery, F1 avg 22%, F2 avg 43%
Report all 10 models in Table 3 (full results matrix).
Note ARDL and dual adstock critical failures even in S1 — explain mechanism briefly.

### 4.2 LTC Persistence Under Spend Discontinuity — S2
The paper's headline experiment.
Lead with the pause-window robustness ratio table (Table 4).
Centrepiece number: BSTS 1.02×, geo_adstock 1.41×, weibull 1.49×
Call out the F2 paradox (ratio < 1.0) and explain it is overfitting correction not robustness.
Reference channel attribution finding: ARDL 68.8% aggregate, 0% channel recovery.

### 4.3 Collinearity and Seasonal Confounding — S3
F1 collapse: avg 20.9% recovery
MCMC anomaly: 99.0% recovery — explain mechanism (seasonal regularity aids identification)
BSTS channel inversion: Display ranked #1 when true rank is #4
F2 partial success: AR structure absorbs seasonal variation

### 4.4 Structural Break Adaptation — S4
F3 dominance: avg 82.7% recovery
F1 collapse: avg 3.5% (weibull and dual adstock go negative)
ARDL catastrophic reversal: 68.8% in S2 → -19.8% in S4 — this is the paper's
strongest cautionary finding. State the mechanism: AR structure calibrated to
pre-break data actively mispredicts post-break dynamics.
almon_pdl anomaly: 68.6% — investigate and state mechanism.

### 4.5 Weak Signal and False Discovery — S5
Universal collapse under frozen parameters: all 10 models at 0%
MCMC supplementary with scenario-specific priors: 88.5% recovery
Kalman and BSTS remain at 0% — fixed decay is the binding constraint,
not observation noise (tested explicitly)
Paper point: prior specification is the essential differentiator under weak signal

### 4.6 Channel Attribution Validation
This subsection is essential. Do not treat it as supplementary.
State the aggregate vs channel finding as a general principle.
Present the ARDL and Koyck S2 channel tables.
Present Video LTC signal loss pattern across all non-MCMC models in S3/S4/S5.
State: "Video LTC recovery serves as a diagnostic test for model robustness."

---

## How to Lead Each Subsection

Lead with the finding, not a description of the table.

WRONG: "Table 3 presents recovery rates for all ten models across five scenarios."
RIGHT: "F3 state-space methods recover 78.4% of true LTC on average across S1–S4,
        compared to 42.8% for F2 distributed lag models and 22.4% for F1 static
        adstock methods (Table 3)."

---

## Required Tables in This Section
- Table 3: Full recovery matrix (10 models × 5 scenarios)
- Table 4: Pause-window robustness ratios
- Table 5: Channel attribution summary (S2/S3/S4 for MCMC, ARDL, Koyck, BSTS)
- Table 6: Budget allocation error by model and scenario

See @visuals.md for formatting standards.

---

## Required Figures in This Section
- Figure 1: Robustness spectrum bar chart (sorted by pause-window ratio)
- Figure 2: Cross-scenario recovery heatmap
- Figure 3: S2 pause window LTC comparison (weeks 95–125)
- Figure 4: Channel attribution comparison (radar or bar)

See @visuals.md for formatting standards.

---

## What Not to Do
- Do not interpret findings in this section — that is Discussion (Section 5)
- Do not cite literature here — results speak for themselves
- Do not repeat numbers already shown in tables — reference the table
- Do not present anomalies without a brief mechanistic note
