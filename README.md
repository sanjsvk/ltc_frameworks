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

## Structure

```
ltc_frameworks/
├── CLAUDE.md          # Project context for Claude Code
├── README.md          # This file
├── data/              # Synthetic CSVs (generated)
├── notebooks/         # Analysis notebooks per framework
└── outputs/           # Results, charts, paper artifacts
```

## Context

This research is part of an EB1-A visa portfolio authored by a Data Scientist, demonstrating original analytical contributions at the intersection of econometrics, marketing science, and applied machine learning.
