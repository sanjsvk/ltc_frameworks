"""
Latent brand stock dynamics — the true data-generating process for LTC.

This module implements the exact brand stock model used to generate the
synthetic dataset.  It is both:

  1. A simulation utility (used to verify data generation logic)
  2. A feature-engineering primitive for Framework 3 models that directly
     parameterise δ and build_rate

Brand equity accumulation equation:
    stock[t] = δ · stock[t-1] + build_rate · √spend[t]

Long-term contribution:
    LTC[t] = ltc_coef · stock[t]

Key properties:
  - δ (retention rate): fraction of last week's stock preserved.
    δ = 0.90 → half-life ≈ 6.6 weeks (TV)
    δ = 0.30 → half-life < 1 week (Paid Search — near-zero LTC)
  - build_rate: spend-to-stock elasticity; square root ensures diminishing
    returns to spend (doubling spend < doubles stock build).
  - Steady-state stock: stock_ss = build_rate · √avg_spend / (1 - δ)
  - Initialisation: stock[0] = steady-state value to avoid burn-in artifact.

References:
    Internal synthetic data specification (docs/MMM_Synthetic_Data_WriteUp.docx)
"""

import numpy as np


# Ground-truth parameters from the synthetic data-generating process
# (per docs/MMM_Synthetic_Data_WriteUp.docx, Table: LTC parameters)
DEFAULT_PARAMS: dict[str, dict] = {
    "tv":     {"delta": 0.90, "build_rate": 0.600, "ltc_coef": 0.0809},
    "search": {"delta": 0.30, "build_rate": 0.100, "ltc_coef": 0.3949},
    "social": {"delta": 0.82, "build_rate": 0.350, "ltc_coef": 0.1936},
    "display": {"delta": 0.65, "build_rate": 0.200, "ltc_coef": 0.3481},
    "video":  {"delta": 0.88, "build_rate": 0.550, "ltc_coef": 0.1495},
}


def brand_stock_dynamics(
    spend: np.ndarray,
    delta: float,
    build_rate: float,
    init_stock: float | None = None,
) -> np.ndarray:
    """
    Simulate latent brand stock evolution given a spend series.

    Parameters
    ----------
    spend : np.ndarray, shape (T,)
        Weekly media spend in $M.
    delta : float
        Stock retention rate in [0, 1).  Higher = longer memory.
    build_rate : float
        Spend-to-stock build elasticity (> 0).
    init_stock : float or None
        Initial stock level at t=0.  If None, initialised to steady-state:
            stock_ss = build_rate * sqrt(mean(spend)) / (1 - delta)

    Returns
    -------
    np.ndarray, shape (T,)
        Latent brand stock series.
    """
    if not 0.0 <= delta < 1.0:
        raise ValueError(f"delta must be in [0, 1), got {delta}")
    if build_rate <= 0:
        raise ValueError(f"build_rate must be positive, got {build_rate}")

    T = len(spend)
    stock = np.empty(T, dtype=float)

    # Steady-state initialisation avoids artificial burn-in ramp
    if init_stock is None:
        avg_spend = np.mean(spend[spend > 0]) if np.any(spend > 0) else 1.0
        init_stock = build_rate * np.sqrt(avg_spend) / (1.0 - delta)

    stock[0] = delta * init_stock + build_rate * np.sqrt(max(spend[0], 0.0))
    for t in range(1, T):
        stock[t] = delta * stock[t - 1] + build_rate * np.sqrt(max(spend[t], 0.0))

    return stock


def brand_stock_ltc(
    spend: np.ndarray,
    delta: float,
    build_rate: float,
    ltc_coef: float,
    init_stock: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute both the latent stock series and the resulting LTC series.

    Parameters
    ----------
    spend : np.ndarray, shape (T,)
        Weekly media spend in $M.
    delta : float
        Stock retention rate.
    build_rate : float
        Spend-to-stock elasticity.
    ltc_coef : float
        Coefficient mapping stock units to $M sales.
    init_stock : float or None
        Initial stock (None → steady-state).

    Returns
    -------
    stock : np.ndarray, shape (T,)
        Latent brand stock series.
    ltc : np.ndarray, shape (T,)
        Long-term contribution in $M.
    """
    stock = brand_stock_dynamics(spend, delta, build_rate, init_stock)
    ltc = ltc_coef * stock
    return stock, ltc


def stock_half_life(delta: float) -> float:
    """
    Return the stock half-life in periods implied by retention rate delta.

    Half-life = -log(2) / log(delta)
    """
    if delta <= 0:
        return 0.0
    if delta >= 1.0:
        return float("inf")
    return -np.log(2) / np.log(delta)


def stock_steady_state(
    avg_spend: float, delta: float, build_rate: float
) -> float:
    """
    Return the equilibrium stock level for a constant spend level.

    stock_ss = build_rate * sqrt(avg_spend) / (1 - delta)
    """
    if delta >= 1.0:
        raise ValueError("No finite steady state when delta >= 1")
    return build_rate * np.sqrt(max(avg_spend, 0.0)) / (1.0 - delta)
