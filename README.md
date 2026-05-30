# LTC Frameworks: Long-Term Media Contribution Estimation

A benchmarking study comparing analytical frameworks for estimating long-term media contributions (LTC) in Media Mix Modelling (MMM), using synthetic ground-truth data.

## Research Question

Media investment drives both **short-term sales spikes** (STC) and **long-term brand equity accumulation** (LTC). The LTC component — representing effects that persist weeks or months after exposure — is notoriously difficult to isolate. This project benchmarks three methodological families against known ground truth to evaluate when each approach succeeds or fails.

## Frameworks Under Evaluation

| Framework | Description |
|-----------|-------------|
| **Static Adstock Regression** | Geometric/Weibull adstock applied to impressions; treats media effect as a single decaying coefficient |
| **Dynamic Time-Series Models** | DLM/Kalman Filter-based; allows media effectiveness to evolve over time |
| **State-Space / Latent Brand-Stock Models** | Explicit latent brand equity stock that builds and decays; closest to the true data-generating process |

## Synthetic Dataset

Ground truth data generated across **5 scenarios**, **5 media channels**, and **261 weeks (2020–2025)**:

- **Channels:** TV, Paid Search, Paid Social, Display, Video
- **True STC:** ~$1.58M/week (~15% of sales)
- **True LTC:** ~$1.23M/week (~12% of sales)
- **LTC mechanism:** Latent brand stock — `stock[t] = δ × stock[t-1] + build_rate × √spend[t]`

Each scenario varies channel mix, spend patterns, and LTC signal strength to stress-test each framework under different conditions.

## Replication Instructions

All code, data, and results are provided for reproducibility. To replicate:

```bash
# Install dependencies
pip install -r pyproject.toml

# Run experiments for a single model and scenario
python experiments/run_experiment.py --model bsts --scenario S1

# Run all 10 models across all 5 scenarios
python experiments/run_experiment.py --all-models --all-scenarios
```

Results are stored as JSON in `outputs/results/{model}_{scenario}.json`.

## Repository Structure

```
ltc_frameworks/
├── ltc/                    # Core Python package (reusable models + utilities)
│   ├── data/              # Data loading & feature engineering
│   ├── models/            # 10 estimation methods across 3 frameworks
│   │   ├── framework1/    # Static Adstock (4 models)
│   │   ├── framework2/    # Dynamic Time-Series (3 models)
│   │   └── framework3/    # State-Space / Latent Stock (3 models)
│   ├── evaluation/        # Metrics & scoring
│   └── visualization/     # Plotting utilities
├── experiments/           # Main experiment runner
│   ├── run_experiment.py  # CLI interface
│   ├── registry.py        # Model registry
│   └── configs/           # Hyperparameter grids per framework
├── data/                  # Synthetic CSV datasets (5 scenarios × 261 weeks)
├── outputs/
│   ├── results/           # JSON outputs (per model × scenario) - gitignored
│   ├── figures/           # 120+ publication-quality PNG figures
│   └── reports/           # Summary tables
├── notebooks/             # Jupyter analysis & exploration
├── writing/               # Paper manuscript & figures
├── analysis/              # Validation reports & intermediate analysis
├── scripts/               # Utility scripts (conversion, extraction, validation)
├── pyproject.toml         # Dependencies & package metadata
├── CLAUDE.md              # Project instructions
└── README.md              # This file
```

## Key Files for Replication

- **Data generation:** Hardcoded in `ltc/data/loader.py` (imports from `data/` CSVs)
- **Model registry:** `experiments/registry.py` (maps model names to classes)
- **Hyperparameter configs:** `experiments/configs/` (per-framework parameter grids)
- **Evaluation metrics:** `ltc/evaluation/metrics.py` (recovery_accuracy, MAPE, etc.)
- **Paper manuscript:** `writing/MASTER_DOCUMENT_FINAL.md` (markdown source)

## Citation

If you use this framework, please cite:

```bibtex
@article{Vijayakumar2026LTC,
  author = {Vijayakumar, Sanjan},
  title = {Long-Term Media Contribution Estimation: Framework Benchmarking Study},
  journal = {Journal of Marketing Research},
  year = {2026},
  note = {Available at https://github.com/sanjsvk/ltc_frameworks}
}
```

## Context

This research is part of an EB1-A visa portfolio demonstrating original analytical contributions at the intersection of econometrics, marketing science, and applied machine learning.
