"""
Geometric (exponential) adstock transformation.

The geometric adstock models carryover as a simple exponential decay:

    adstocked[t] = x[t] + decay * adstocked[t-1]

This is the most widely-used adstock form in MMM due to its simplicity and
interpretability. The single `decay` parameter (θ) controls the fraction of
last week's adstocked impressions carried forward.

Half-life relationship:  half_life = -log(2) / log(decay)

References:
    Broadbent (1979); Koletsi & Fulgoni (2010)
"""

import numpy as np


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Apply geometric adstock to a 1-D array of media values.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Raw media input (impressions or spend) over T time periods.
    decay : float
        Carryover decay rate in (0, 1).  Higher values = longer memory.
        decay=0 collapses to no carryover (instantaneous effect only).

    Returns
    -------
    np.ndarray, shape (T,)
        Adstocked media values.  Same units as `x`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 0.0, 0.0, 0.0])
    >>> geometric_adstock(x, decay=0.5)
    array([1.    , 0.5   , 0.25  , 0.125 ])
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must be in [0, 1), got {decay}")

    T = len(x)
    out = np.empty(T, dtype=float)
    out[0] = x[0]
    for t in range(1, T):
        out[t] = x[t] + decay * out[t - 1]
    return out


def geometric_adstock_matrix(X: np.ndarray, decays: np.ndarray) -> np.ndarray:
    """
    Apply geometric adstock column-wise to a 2-D media matrix.

    Parameters
    ----------
    X : np.ndarray, shape (T, C)
        Media matrix with T time periods and C channels.
    decays : np.ndarray, shape (C,)
        Per-channel decay rates.

    Returns
    -------
    np.ndarray, shape (T, C)
        Adstocked media matrix.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2-D (T x C)")
    if len(decays) != X.shape[1]:
        raise ValueError("len(decays) must equal number of columns in X")

    return np.column_stack(
        [geometric_adstock(X[:, c], decays[c]) for c in range(X.shape[1])]
    )


def geometric_half_life(decay: float) -> float:
    """Return the half-life (in periods) implied by a decay rate."""
    if decay <= 0:
        return 0.0
    return -np.log(2) / np.log(decay)
