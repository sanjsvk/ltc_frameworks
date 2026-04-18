"""
Koyck (geometric distributed lag) model transformation.

The Koyck transformation converts an infinite geometric lag model into an
estimable OLS/IV equation by algebraic manipulation.

Original model:
    y[t] = α + β₀·x[t] + β₁·x[t-1] + β₂·x[t-2] + ... + ε[t]

Assuming geometric lag weights  βₗ = β₀·λˡ  (0 < λ < 1):

    y[t] = α·(1-λ) + β₀·x[t] + λ·y[t-1] + (ε[t] - λ·ε[t-1])

This collapses infinite lags into a single lagged dependent variable.
The parameter λ (the "Koyck lambda") has the same role as the decay
parameter in geometric adstock — it controls how fast past impacts fade.

Key property: the long-run multiplier (total LTC + STC) is  β₀ / (1 - λ).

Estimation notes:
  - The composite error term (ε[t] - λ·ε[t-1]) is MA(1) — OLS is consistent
    but not efficient; use HAC standard errors or IV estimation.
  - The lagged dependent variable y[t-1] may correlate with the error
    in finite samples — Instrumental Variables can correct this.

References:
    Koyck (1954); Greene (2003) ch. 19; Hanssens et al. (2001)
"""

import numpy as np


def koyck_regressors(
    y: np.ndarray,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the Koyck-transformed regressor matrix [X[t], y[t-1]] and
    the aligned response vector y[t], discarding the first observation.

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        Dependent variable (observed sales).
    X : np.ndarray, shape (T, C)  or  (T,)
        Current-period media inputs (C channels, or 1 channel as 1-D).

    Returns
    -------
    y_t : np.ndarray, shape (T-1,)
        Response vector aligned to t = 1 … T-1.
    Z : np.ndarray, shape (T-1, C+1)
        Regressor matrix: columns = [X[t], …, y[t-1]].
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y_t = y[1:]          # response: t = 1 … T-1
    X_t = X[1:]          # current media: t = 1 … T-1
    y_lag = y[:-1]       # lagged response: y[t-1]

    Z = np.column_stack([X_t, y_lag])
    return y_t, Z


def koyck_long_run_multiplier(beta0: float, lambda_: float) -> float:
    """
    Compute the long-run multiplier (total accumulated effect) for a single channel.

    Long-run effect = β₀ / (1 - λ)

    Parameters
    ----------
    beta0 : float
        Contemporaneous coefficient on the media variable.
    lambda_ : float
        Koyck lambda (lag decay) in (0, 1).

    Returns
    -------
    float
        Total accumulated response per unit of media.
    """
    if not 0.0 < lambda_ < 1.0:
        raise ValueError(f"lambda_ must be in (0, 1), got {lambda_}")
    return beta0 / (1.0 - lambda_)


def koyck_decompose(
    y: np.ndarray,
    X: np.ndarray,
    beta0: np.ndarray,
    lambda_: float,
    alpha: float,
) -> dict[str, np.ndarray]:
    """
    Decompose fitted Koyck model into STC and LTC contributions per period.

    The Koyck structure means LTC at time t is the portion of the lagged
    sales term  λ·y[t-1]  attributable to past media accumulation.  Here we
    use the simpler approximation: split the long-run multiplier into
    STC (contemporaneous β₀) and LTC (residual long-run effect).

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        Observed sales.
    X : np.ndarray, shape (T, C)
        Media inputs (C channels).
    beta0 : np.ndarray, shape (C,)
        Fitted contemporaneous media coefficients.
    lambda_ : float
        Fitted Koyck lambda.
    alpha : float
        Fitted intercept (corresponds to α·(1-λ) in the transformed equation).

    Returns
    -------
    dict with keys:
        "baseline"  : np.ndarray (T,) — constant + Koyck momentum from non-media
        "stc_{ch}"  : np.ndarray (T,) per channel — β₀·x[t]
        "ltc_{ch}"  : np.ndarray (T,) per channel — residual long-run accumulation
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    C = X.shape[1]

    result: dict[str, np.ndarray] = {}

    # Baseline: intercept term (converted back from transformed α·(1-λ))
    baseline_const = alpha / (1.0 - lambda_) if lambda_ < 1.0 else alpha
    result["baseline"] = np.full(len(y), baseline_const)

    for c in range(C):
        stc = beta0[c] * X[:, c]
        long_run = koyck_long_run_multiplier(beta0[c], lambda_)
        # LTC is the residual long-run effect beyond the contemporaneous impact
        ltc = (long_run - beta0[c]) * X[:, c]
        result[f"stc_{c}"] = stc
        result[f"ltc_{c}"] = ltc

    return result
