"""
Almon Polynomial Distributed Lag (PDL) transformation.

The Almon PDL constrains lag weights to lie on a polynomial of degree `d`:

    w[l] = a_0 + a_1*l + a_2*l^2 + ... + a_d*l^d,  l = 0, 1, ..., L

where the coefficients {a_k} are free parameters estimated from data, but the
polynomial constraint drastically reduces the number of free parameters from
(L+1) to (d+1), improving estimation stability for long lag structures.

Optionally the weights at l=0 and/or l=L can be constrained to zero
(endpoint constraints), which is standard in distributed lag modelling.

The PDL is typically applied by replacing the raw media regressor matrix
X (T × L+1 lags) with a compressed Z (T × d+1) using the Almon mapping
matrix A, then estimating β in:

    y = Z·β + ε,  where  Z = X·A

The original lag weights are recovered as  w = A·β.

References:
    Almon (1965) "The Distributed Lag Between Capital Appropriations and
    Expenditures"; Greene (2003) Econometric Analysis, ch. 19.
"""

import numpy as np


def almon_basis_matrix(max_lag: int, degree: int) -> np.ndarray:
    """
    Construct the Almon polynomial basis matrix A of shape (max_lag+1, degree+1).

    Row l of A is [1, l, l^2, ..., l^degree].  Multiplying a lag matrix
    X (T × L+1) by A gives the compressed regressor matrix Z (T × d+1).

    Parameters
    ----------
    max_lag : int
        Maximum lag L (inclusive).  Weights defined for lags 0..L.
    degree : int
        Polynomial degree d.  Requires d <= max_lag.

    Returns
    -------
    np.ndarray, shape (max_lag+1, degree+1)
    """
    if degree > max_lag:
        raise ValueError(f"degree ({degree}) cannot exceed max_lag ({max_lag})")
    lags = np.arange(max_lag + 1, dtype=float)
    return np.column_stack([lags**k for k in range(degree + 1)])


def almon_pdl_weights(
    coeffs: np.ndarray,
    max_lag: int,
    degree: int,
    endpoint_constraints: tuple[bool, bool] = (False, False),
) -> np.ndarray:
    """
    Recover lag weights from Almon polynomial coefficients.

    Parameters
    ----------
    coeffs : np.ndarray, shape (degree+1,)
        Polynomial coefficients [a_0, a_1, ..., a_d].
    max_lag : int
        Maximum lag L.
    degree : int
        Polynomial degree d.
    endpoint_constraints : (bool, bool)
        If (True, _) forces w[0]=0 (near-zero constraint at lag 0).
        If (_, True) forces w[L]=0 (far-zero constraint at lag L).

    Returns
    -------
    np.ndarray, shape (max_lag+1,)
        Lag weights for lags 0..L.
    """
    A = almon_basis_matrix(max_lag, degree)
    weights = A @ coeffs

    # Apply endpoint constraints by zeroing the boundary weights
    if endpoint_constraints[0]:
        weights[0] = 0.0
    if endpoint_constraints[1]:
        weights[-1] = 0.0

    return weights


def build_lag_matrix(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Build a lagged regressor matrix from a 1-D media array.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Raw media input.
    max_lag : int
        Maximum lag L.  Columns = [x[t], x[t-1], ..., x[t-L]].

    Returns
    -------
    np.ndarray, shape (T, max_lag+1)
        Lag matrix; values before t=0 are zero-padded.
    """
    T = len(x)
    X_lag = np.zeros((T, max_lag + 1), dtype=float)
    for l in range(max_lag + 1):
        if l == 0:
            X_lag[:, l] = x
        else:
            X_lag[l:, l] = x[:-l]
    return X_lag


def almon_compressed_regressors(
    x: np.ndarray, max_lag: int, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the compressed Almon regressor matrix Z = X_lag @ A.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
    max_lag : int
    degree : int

    Returns
    -------
    Z : np.ndarray, shape (T, degree+1)
        Compressed regressor matrix to use as model input.
    A : np.ndarray, shape (max_lag+1, degree+1)
        Almon basis matrix (kept for weight recovery after fitting).
    """
    X_lag = build_lag_matrix(x, max_lag)
    A = almon_basis_matrix(max_lag, degree)
    Z = X_lag @ A
    return Z, A
