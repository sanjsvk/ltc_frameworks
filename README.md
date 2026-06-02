# LTC Frameworks: Long-Term Media Contribution Estimation

Benchmarking study evaluating **10 analytical methods across 3 frameworks** for estimating long-term media contributions (LTC) in Marketing Mix Modeling (MMM). Uses **synthetic data with known ground truth** to measure recovery accuracy, robustness to structural breaks, and channel-level attribution precision.

**Key Finding:** State-space methods (BSTS, Bayesian MCMC) recover 79.3% of true LTC on average, compared to 44.2% for dynamic models and 29.6% for static adstock. However, aggregate recovery metrics mask channel-level attribution failures: two methods achieve 68.8% aggregate recovery while returning 0% for individual channels.

---

## Quick Start (5 minutes)

### 1. Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/sanjsvk/ltc_frameworks.git
cd ltc_frameworks

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package with all dependencies
pip install -e ".[dev]"
```

**Dependencies are fully pinned in `pyproject.toml`:**
- Core: numpy, pandas, scipy
- Econometrics: statsmodels
- Bayesian: pymc, numpyro
- Time-series: pykalman
- ML: scikit-learn
- Viz: matplotlib, seaborn
- Config: pyyaml, click
- Development: pytest, ruff, jupyterlab

### 2. Run Your First Experiment

```bash
# Single model × scenario (takes ~2–5 min depending on framework)
python experiments/run_experiment.py --model bsts --scenario S1

# Check outputs:
# - outputs/results/bsts_S1.json              ← Metrics (recovery%, MAPE, etc.)
# - outputs/figures/bsts_S1_decomp.png        ← Decomposition chart
```

### 3. Run Multiple Models

```bash
# All 10 models × 1 scenario (~10 min)
python experiments/run_experiment.py --all-models --scenario S1

# Framework group × all scenarios (~15 min for F3)
python experiments/run_experiment.py --framework F3_state_space --all-scenarios

# Full benchmark: all 10 models × all 5 scenarios (~2–3 hours)
python experiments/run_experiment.py --all-models --all-scenarios
```

Results are saved as JSON to `outputs/results/{model}_{scenario}.json`.

---

## Workflow: From Data to Results

```
Step 1: Prepare Data
├─ Synthetic data pre-loaded in data/raw/
└─ data/raw/mmm_synthetic_generator.py (to regenerate)

Step 2: Train Models
├─ python experiments/run_experiment.py --model [MODEL] --scenario [S1-S5]
└─ Hyperparameters loaded from experiments/configs/framework*.yaml

Step 3: Generate Outputs
├─ outputs/results/{model}_{scenario}.json    ← Metrics JSON
└─ outputs/figures/{model}_{scenario}_decomp.png  ← Decomposition chart

Step 4: Evaluate Results
├─ recovery_accuracy % (primary metric)
├─ MAPE % (error metric)
├─ per-channel breakdown
└─ fitted parameters for reproducibility
```

---

## What This Repository Contains

### The Problem

CMOs allocate budgets using MMMs designed for **short-term elasticities**, missing 10–15% of true ROI from **long-term effects**. Current methods fail under:
- **Spending pauses** (can't separate persistence from zero spend)
- **Collinearity** (channels move together → attribution reversals)
- **Weak signals** (insufficient variance → identification failure)

### The Solution

This repo provides:
1. **Reproducible benchmarking framework** with ground-truth data
2. **10 implementations** (4 static + 3 dynamic + 3 state-space)
3. **5 diagnostic scenarios** (baseline, spend pause, seasonality, structural break, weak signal)
4. **Metrics** for evaluating recovery accuracy, robustness, and channel precision
5. **Decision framework** for practitioners on method selection

### 10 Models Evaluated

| Framework | Models | Strengths | Weaknesses |
|-----------|--------|-----------|-----------|
| **F1: Static Adstock** | geometric, weibull, almon_pdl, dual_adstock | Simple, interpretable | Fails on pauses & collinearity |
| **F2: Dynamic Time-Series** | koyck, ardl, finite_dl | Flexible lag shapes | Channel attribution instability |
| **F3: State-Space** | kalman_dlm, mcmc_stock, bsts | Robust to breaks | Computationally expensive |

---

## Repository Structure

```
ltc_frameworks/
├── ltc/                          # Core replicable package
│   ├── data/                     # Data loading (CSV → DataFrame)
│   ├── models/
│   │   ├── framework1/           # 4 static adstock models
│   │   ├── framework2/           # 3 dynamic time-series models
│   │   └── framework3/           # 3 state-space/latent stock models
│   ├── evaluation/
│   │   ├── metrics.py            # recovery_accuracy, MAPE, CI coverage
│   │   ├── scorer.py             # Unified scoring interface
│   │   └── benchmark.py          # Multi-model comparison
│   └── visualization/            # Plotting utilities
│
├── experiments/
│   ├── run_experiment.py         # Main CLI interface
│   ├── registry.py               # Model registry
│   └── configs/                  # YAML hyperparameter grids
│
├── data/
│   ├── raw/                      # 5 scenario CSVs (261 weeks × 39 columns each)
│   └── processed/                # Feature-engineered data (generated)
│
├── outputs/
│   ├── results/                  # JSON per model×scenario (gitignored)
│   ├── figures/                  # 120+ PNG figures tracked in git
│   └── reports/                  # Summary tables
│
├── notebooks/                    # Jupyter exploration notebooks
├── writing/                      # Paper manuscript (MASTER_DOCUMENT_FINAL.md)
├── analysis/                     # Validation reports & intermediate analysis
├── scripts/                      # Utility scripts (archived)
│
├── pyproject.toml                # Package metadata & dependencies
├── CLAUDE.md                     # Project instructions
├── .gitignore                    # Git configuration (secrets, outputs)
└── README.md                     # This file
```

### Key Entry Points

- **Run experiments:** `experiments/run_experiment.py`
- **Register models:** `experiments/registry.py`
- **Score results:** `ltc/evaluation/scorer.py`
- **Paper manuscript:** `writing/MASTER_DOCUMENT_FINAL.md`

---

## Understanding Results

### Result JSON Structure

Each `outputs/results/{model}_{scenario}.json` contains:

```json
{
  "model": "geo_adstock",
  "scenario": "S1",
  "ltc": {
    "total": {
      "mape": 30.1,
      "recovery_accuracy": 69.9,
      "mae": 0.42,
      "rmse": 0.51,
      "correlation": 0.89,
      "bias": -0.15,
      "total_recovery_ratio": 1.01
    },
    "tv": { "mape": 18.2, "recovery_accuracy": 81.8, ... },
    "search": { ... },
    "social": { ... },
    "display": { ... },
    "video": { ... }
  },
  "stc": { "total": {...}, "tv": {...}, ... },
  "fitted_params": {
    "decay_tv": 0.65,
    "decay_search": 0.42,
    ...
  }
}
```

### Key Metrics Explained

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| **recovery_accuracy** | 0–100% | How much true LTC was recovered; 100% = perfect |
| **MAPE** | 0–∞% | Mean Absolute Percentage Error; lower is better |
| **total_recovery_ratio** | 0–∞ | Sum(estimated) / Sum(true); 1.0 = perfect |
| **correlation** | -1 to 1 | Shape similarity (independent of scale) |
| **bias** | -∞ to ∞ | Signed error; positive = over-estimate |

### Decomposition Figures

Each PNG shows 3 subplots:
1. **Observed vs Fitted Sales** — Does model fit the data?
2. **LTC Recovery** — How accurate are LTC estimates?
3. **Stacked Components** — STC + LTC breakdown over time

---

## Common Workflows

### Test a Single Model

```bash
# Train geo_adstock on baseline scenario
python experiments/run_experiment.py --model geo_adstock --scenario S1

# Load results
python -c "
import json
with open('outputs/results/geo_adstock_S1.json') as f:
    result = json.load(f)
print(f\"Recovery: {result['ltc']['total']['recovery_accuracy']:.1f}%\")
print(f\"MAPE: {result['ltc']['total']['mape']:.1f}%\")
"
```

### Compare Two Models Across Scenarios

```bash
# Run both
python experiments/run_experiment.py --model geo_adstock --all-scenarios
python experiments/run_experiment.py --model mcmc_stock --all-scenarios

# Compare in Python
python -c "
import json, pandas as pd

models = ['geo_adstock', 'mcmc_stock']
results = {}

for m in models:
    results[m] = {}
    for s in ['S1', 'S2', 'S3', 'S4', 'S5']:
        with open(f'outputs/results/{m}_{s}.json') as f:
            j = json.load(f)
        results[m][s] = j['ltc']['total']['recovery_accuracy']

df = pd.DataFrame(results)
print(df)
print(f'\nGeo Adstock avg: {df.loc[\"geo_adstock\"].mean():.1f}%')
print(f'MCMC avg: {df.loc[\"mcmc_stock\"].mean():.1f}%')
"
```

### Debug a Model's Fitted Parameters

```python
from ltc.data.loader import load_scenario, split_observed_truth
from ltc.models.framework1.geometric_regression import GeometricAdstockOLS
import json

# Load scenario data
df_full = load_scenario("data/raw", "S1")
df_obs, df_truth = split_observed_truth(df_full)

# Train model
model = GeometricAdstockOLS()
model.fit(df_obs, {"decay_grid": [0.3, 0.5, 0.7]})

# Print fitted parameters
params = model.get_params()
print(json.dumps(params, indent=2))

# Also print ground truth for comparison
print("\nGround truth LTC (TV channel):")
print(df_truth['ltc_tv_true'].describe())
```

### Regenerate Synthetic Data

```bash
# If you want to create new scenarios with different parameters:
python data/raw/mmm_synthetic_generator.py

# This overwrites data/raw/S1.csv, S2.csv, ..., S5.csv
# Modify the generator script to change:
# - Time span (weeks)
# - Channel parameters (delta, build_rate, ltc_coef)
# - Scenario intensity (seasonality, discontinuity, noise)
```

### Skip Figure Generation (Faster)

If you only care about metrics and not visualizations:

```bash
python experiments/run_experiment.py --all-models --all-scenarios --no-fig
```

---

## Environment & Security

### Credentials

This repository has **no private credentials**. All configuration is:
- Hardcoded defaults in code
- Set via CLI arguments
- Captured in checked-in YAML configs

If you need API keys or tokens:
1. Create `.env` (automatically .gitignored)
2. Load with `python-dotenv` (not included — add if needed)
3. Never commit `.env` files

### Data Privacy

Synthetic data is fully generated; no real customer/campaign data included.

---

## Experiment Details

### Synthetic Data Generation

**Process:** Hardcoded in `ltc/data/loader.py`
- **Time span:** Jan 2020 – Dec 2025 (261 weeks)
- **Channels:** 5 (TV, Search, Social, Display, Video)
- **True STC:** ~$1.58M/week (geometric adstock)
- **True LTC:** ~$1.23M/week (latent brand stock)
- **Exogenous:** Promo intensity, COVID index, Treasury yield, mobility, competitor share

**Ground truth:** Available in `data/{scenario}.csv` under columns:
- `*_true` — true contribution values
- `baseline_true`, `stc_*_true`, `ltc_*_true` — component breakdown

### 5 Diagnostic Scenarios

| Scenario | Objective | Weeks 104–112 | Interpretation |
|----------|-----------|---------------|-----------------|
| **S1: Baseline** | Establish floor | Normal | Clean identification |
| **S2: Spend Pause** | Test latent decay | TV+Video = $0 | Can model identify persistence? |
| **S3: Seasonality** | Test collinearity | 85% intensity | Does framework separate signal? |
| **S4: Struct. Break** | Test regime stability | 20% of baseline | Adapt to permanent shift? |
| **S5: Weak Signal** | Test identifiability | Normal (low variance) | Resolve weak effects? |

### Metrics

- **Recovery Accuracy (%):** `(estimated_LTC / true_LTC) × 100` — primary metric
- **MAPE (%):** Mean Absolute Percentage Error on LTC estimates
- **Pause Ratio:** `MAPE(pause_window) / MAPE(full_series)` — robustness to structural breaks
- **Channel Recovery (%):** Per-channel recovery accuracy (diagnostic for F2)

---

## 10 Models: Quick Reference

### Framework 1: Static Adstock Regression (Fast, Simple)

| Model | Method | Pros | Cons |
|-------|--------|------|------|
| **geo_adstock** | Geometric decay | Interpretable, fast | Fixed lag, no adaptation |
| **weibull_adstock** | Weibull-shaped lags | Flexible decay shape | Slower, complex grid |
| **almon_pdl** | Polynomial lags | Smooth lag weights | Fails on discontinuities |
| **dual_adstock** | Geometric + Weibull | Hybrid flexibility | Slow, high params |

### Framework 2: Dynamic Time-Series Distributed Lag (Medium Speed)

| Model | Method | Pros | Cons |
|-------|--------|------|------|
| **koyck** | Shared lambda decay | Fast, captures autocorr | Shared lambda all channels |
| **ardl** | AR + DL regression | Flexible, interpretable | MA(1) error, channel unstable |
| **finite_dl** | Finite lags + Almon | Smooth, constrained | Less flexible than ARDL |

### Framework 3: State-Space / Bayesian (Slow, Flexible)

| Model | Method | Pros | Cons |
|-------|--------|------|------|
| **kalman_dlm** | Kalman filter + trend | Handles seasonality | Computationally slow |
| **mcmc_stock** | Latent brand stock + MCMC | High flexibility, uncertainty | Very slow, prior-sensitive |
| **bsts** | Bayesian structural TS | State-of-art | Slow, requires tuning |

**Framework Ranking by Recovery Accuracy:**
1. **F3 (State-Space):** 79.3% average (BSTS, Kalman, MCMC)
2. **F2 (Dynamic TS):** 44.2% average (Koyck, ARDL, Finite DL)
3. **F1 (Static Adstock):** 29.6% average (Geo, Weibull, Almon, Dual)

**When to Use Which:**
- **Real-time, low data?** → F1 (geo_adstock) — fast, no tuning
- **Flexible lags needed?** → F2 (ardl) — faster than F3, still flexible
- **Accuracy critical?** → F3 (bsts or mcmc) — best recovery, worth the computation

---

## Troubleshooting

### Common Issues

**Q: "FileNotFoundError: Scenario CSV not found"**
- **Cause:** Data files missing or wrong path
- **Fix:** Check `data/raw/` contains S1.csv–S5.csv, or run:
  ```bash
  python data/raw/mmm_synthetic_generator.py
  ```

**Q: "ModuleNotFoundError: No module named 'ltc'"**
- **Cause:** Package not installed
- **Fix:** Run from repo root and install:
  ```bash
  pip install -e .
  ```

**Q: MCMC takes forever (Framework 3 is slow)**
- **Cause:** Sampling 1000+ draws with 4 chains
- **Fix:** Reduce draws in `experiments/configs/framework3.yaml` or skip figures:
  ```bash
  python experiments/run_experiment.py --model mcmc_stock --scenario S1 --no-fig
  ```

**Q: "RuntimeError: decompose() called before fit()"**
- **Cause:** Calling decompose() without first fitting
- **Fix:** Ensure fit() is called before decompose():
  ```python
  model.fit(df_obs, config)  # First
  decomp = model.decompose(df_obs)  # Then
  ```

**Q: Results JSON is empty or missing fields**
- **Cause:** Model failed during fitting or scoring
- **Fix:** Check console output for errors, or re-run with print statements:
  ```bash
  python experiments/run_experiment.py --model geo_adstock --scenario S1 2>&1 | grep -i error
  ```

---

## Citation

```bibtex
@article{Vijayakumar2026LTC,
  author = {Vijayakumar, Sanjan},
  title = {Long-Term Media Contribution Estimation: 
           Framework Benchmarking Study},
  journal = {Journal of Marketing Research},
  year = {2026},
  note = {Available at https://github.com/sanjsvk/ltc_frameworks}
}
```

---

## Contributing

This is a research artifact. Modifications should focus on:
- Extending to real data
- Adding new frameworks
- Improving documentation

Pull requests welcome; please test against all scenarios before submitting.

---

## License

See CLAUDE.md for research context and usage guidelines.

---

## Contact

**Author:** Sanjan Vijayakumar  
**Email:** sanjan.svk@gmail.com  
**Repository:** https://github.com/sanjsvk/ltc_frameworks
