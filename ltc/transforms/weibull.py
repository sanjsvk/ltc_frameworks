"""
Weibull CDF adstock transformation.

Unlike geometric adstock (which forces a monotonically decreasing lag weight),
the Weibull CDF adstock allows a flexible lag shape:

    w[l] = 1 - exp(-(l / scale)^shape)          (CDF of Weibull distribution)

Normalised weights are applied as a finite-window convolution over `max_lag`
periods.  This enables:
  - Delayed peak effects (shape > 1 → increasing then decreasing weights)
  - Pure decay (shape ≈ 1 → near-geometric)
  - Short sharp effects (large shape, small scale)

Parameters
----------
shape (k) : controls whether lag weights rise then fall (k > 1) or
             decay monotonically (k ≤ 1).
scale (λ) : stretches or compresses the distribution along the lag axis;
             larger values push peak effect further into the future.

References:
    Jin et al. (2017) "Bayesian Methods for Media Mix Modeling with Carryover
    and Shape Effects" — Google / Robyn convention.
"""

import numpy as np
from scipy.special import gamma as gamma_fn


def weibull_cdf_weights(shape: float, scale: float, max_lag: int) -> np.ndarray:
    """
    Compute normalised Weibull CDF lag weights for lags 1 … max_lag.

    Parameters
    ----------
    shape : float
        Weibull shape parameter k > 0.
    scale : float
        Weibull scale parameter λ > 0.
    max_lag : int
        Number of lags to compute weights for.

    Returns
    -------
    np.ndarray, shape (max_lag,)
        Non-negative weights summing to 1.0.
    """
    if shape <= 0 or scale <= 0:
        raise ValueError("shape and scale must be positive")
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")

    lags = np.arange(1, max_lag + 1, dtype=float)
    # CDF value at each lag — difference gives the PDF mass in each bucket
    cdf = 1.0 - np.exp(-((lags / scale) ** shape))
    weights = np.diff(np.concatenate([[0.0], cdf]))
    total = weights.sum()
    if total == 0:
        raise ValueError("All Weibull weights are zero — increase max_lag or adjust params")
    return weights / total


def weibull_adstock(
    x: np.ndarray,
    shape: float,
    scale: float,
    max_lag: int = 20,
    normalise: bool = True,
) -> np.ndarray:
    """
    Apply Weibull CDF adstock to a 1-D media array.

    The transformation is a weighted sum of current and past values:

        adstocked[t] = Σ_{l=0}^{max_lag-1} w[l] * x[t - l]

    where w[l] are the (normalised) Weibull CDF lag weights.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Raw media input over T time periods.
    shape : float
        Weibull shape parameter k > 0.
    scale : float
        Weibull scale parameter λ > 0 (in units of lags/weeks).
    max_lag : int
        Truncation window — lags beyond this are ignored.
    normalise : bool
        If True (default), weights sum to 1 so units are preserved.

    Returns
    -------
    np.ndarray, shape (T,)
        Adstocked media values.
    """
    weights = weibull_cdf_weights(shape, scale, max_lag)
    T = len(x)
    out = np.zeros(T, dtype=float)
    for t in range(T):
        for l, w in enumerate(weights):
            if t - l >= 0:
                out[t] += w * x[t - l]
    return out


def weibull_peak_lag(shape: float, scale: float) -> float:
    """
    Return the continuous lag at which Weibull PDF is maximised.

    Peak occurs at:  l* = scale * ((k-1)/k)^(1/k)   for k > 1
    For k ≤ 1 the mode is at l=0 (instantaneous peak).
    """
    if shape <= 1:
        return 0.0
    return scale * (((shape - 1) / shape) ** (1.0 / shape))
