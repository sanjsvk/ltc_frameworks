"""
Framework 1 — AlmonPDL (Polynomial Distributed Lag regression)

Rather than a single decay parameter, the Almon PDL constrains lag weights
to follow a polynomial of degree `d` over `max_lag` lags.  This gives a
flexible lag distribution (can rise then fall) while drastically reducing
the parameter count from (L+1) to (d+1) per channel.

The compressed regressor matrix Z = X_lag @ A (Almon basis) is computed
once, then OLS is run on Z directly.  The actual lag weights are recovered
post-estimation as w = A @ β.

Interpretive split:
  - STC = contribution attributable to lags 0–stc_cutoff
  - LTC = contribution attributable to lags stc_cutoff+1 … max_lag

`stc_cutoff` is a config parameter (default: 4 weeks, matching typical
media planning convention that effects beyond 4 weeks are "long-term").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ltc.models.base import BaseLTCModel
from ltc.transforms.almon import almon_compressed_regressors, build_lag_matrix

_EXOG = ["promo", "covid_index", "dgs30", "mobility_index", "competitor_ishare"]


class AlmonPDL(BaseLTCModel):
    """
    OLS with Almon Polynomial Distributed Lag transformation.

    Hyperparameters (via config dict)
    ---------------------------------
    max_lag : int
        Maximum lag L (weeks).  Default: 13 (one quarter).
    degree : int
        Polynomial degree d.  Default: 3.
    stc_cutoff : int
        Lags 0..stc_cutoff attributed to STC; lags above to LTC.  Default: 4.
    feature : str
        "impressions" or "spend".  Default: "impressions".
    channels : list[str]
        Default: all 5 channels.
    """

    name = "almon_pdl"
    framework = "F1_static_adstock"

    def __init__(self) -> None:
        super().__init__()
        self._max_lag: int = 13
        self._degree: int = 3
        self._stc_cutoff: int = 4
        self._feature: str = "impressions"
        self._channels: list[str] = []
        self._channel_weights: dict[str, np.ndarray] = {}  # recovered lag weights per ch
        self._channel_coefs: dict[str, np.ndarray] = {}    # polynomial coefs per ch
        self._exog_coefs: np.ndarray | None = None
        self._intercept: float = 0.0
        self._exog_names: list[str] = []

    def fit(self, df: pd.DataFrame, config: dict) -> "AlmonPDL":
        self._max_lag = config.get("max_lag", 13)
        self._degree = config.get("degree", 3)
        self._stc_cutoff = config.get("stc_cutoff", 4)
        self._feature = config.get("feature", "impressions")
        self._channels = config.get("channels", ["tv", "search", "social", "display", "video"])
        exog_cols = [c for c in _EXOG if c in df.columns]
        self._exog_names = exog_cols

        y = df["net_sales_observed"].to_numpy(dtype=float)
        prefix = "impr" if self._feature == "impressions" else "spend"

        X_parts: list[np.ndarray] = []
        channel_A: dict[str, np.ndarray] = {}

        for ch in self._channels:
            col = f"{prefix}_{ch}"
            if col not in df.columns:
                continue
            x_raw = df[col].to_numpy(dtype=float)
            Z, A = almon_compressed_regressors(x_raw, self._max_lag, self._degree)
            X_parts.append(Z)
            channel_A[ch] = A

        # Exogenous controls
        exog_matrix = np.column_stack([df[c].to_numpy(float) for c in exog_cols]) if exog_cols else np.empty((len(y), 0))
        if exog_matrix.shape[1] > 0:
            X_parts.append(exog_matrix)
        X_parts.append(np.ones((len(y), 1)))

        X = np.hstack(X_parts)
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # Recover per-channel lag weights and polynomial coefs
        idx = 0
        d1 = self._degree + 1
        for ch in self._channels:
            if ch in channel_A:
                beta_ch = coefs[idx: idx + d1]
                self._channel_coefs[ch] = beta_ch
                self._channel_weights[ch] = channel_A[ch] @ beta_ch
                idx += d1

        n_exog = exog_matrix.shape[1]
        self._exog_coefs = coefs[idx: idx + n_exog]
        self._intercept = coefs[idx + n_exog]
        self._is_fitted = True
        return self

    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        prefix = "impr" if self._feature == "impressions" else "spend"
        T = len(df)
        index = df.index
        stc_dict: dict[str, pd.Series] = {}
        ltc_dict: dict[str, pd.Series] = {}

        for ch in self._channels:
            col = f"{prefix}_{ch}"
            if col not in df.columns or ch not in self._channel_weights:
                stc_dict[ch] = pd.Series(0.0, index=index)
                ltc_dict[ch] = pd.Series(0.0, index=index)
                continue

            x_raw = df[col].to_numpy(dtype=float)
            w = self._channel_weights[ch]  # shape (max_lag+1,)
            X_lag = build_lag_matrix(x_raw, self._max_lag)  # (T, max_lag+1)
            contrib_per_lag = X_lag * w[np.newaxis, :]       # (T, max_lag+1)

            stc = contrib_per_lag[:, : self._stc_cutoff + 1].sum(axis=1)
            ltc = contrib_per_lag[:, self._stc_cutoff + 1 :].sum(axis=1)
            stc_dict[ch] = pd.Series(stc, index=index)
            ltc_dict[ch] = pd.Series(ltc, index=index)

        baseline_val = np.full(T, self._intercept)
        for j, ecol in enumerate(self._exog_names):
            if ecol in df.columns:
                baseline_val += self._exog_coefs[j] * df[ecol].to_numpy(float)

        baseline = pd.Series(baseline_val, index=index)
        return self._make_decomposition_frame(index, self._channels, baseline, stc_dict, ltc_dict)

    def get_params(self) -> dict:
        self._check_fitted()
        return {
            "model": self.name,
            "max_lag": self._max_lag,
            "degree": self._degree,
            "stc_cutoff": self._stc_cutoff,
            "channel_weights": {ch: w.tolist() for ch, w in self._channel_weights.items()},
            "intercept": float(self._intercept),
        }
