# LTC Frameworks: Long-Term Media Contribution Estimation

Benchmarking study evaluating **10 analytical methods across 3 frameworks** for estimating long-term media contributions (LTC) in Marketing Mix Modeling (MMM). Uses **synthetic data with known ground truth** to measure recovery accuracy, robustness to structural breaks, and channel-level attribution precision.

**Key Finding:** State-space methods (BSTS, Bayesian MCMC) recover 79.3% of true LTC on average, compared to 44.2% for dynamic models and 29.6% for static adstock. However, aggregate recovery metrics mask channel-level attribution failures: two methods achieve 68.8% aggregate recovery while returning 0% for individual channels.

---

## Quick Start

### Installation

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

### Run Experiments

```bash
# Single model × scenario (takes ~2–5 min depending on framework)
python experiments/run_experiment.py --model bsts --scenario S1

# All 10 models × 1 scenario
python experiments/run_experiment.py --scenario S2 --all-models

# All 10 models × all 5 scenarios (full benchmark: ~45 min)
python experiments/run_experiment.py --all-models --all-scenarios
```

Results are saved as JSON to `outputs/results/{model}_{scenario}.json`.

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
