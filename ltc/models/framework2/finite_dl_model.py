"""
Framework 2 — FiniteDLModel (Finite Distributed Lag with shape constraints)

A finite distributed lag model applies a weighted sum of current and lagged
media variables, where the lag weights follow a parametric shape (Weibull or
Almon polynomial).  Unlike ARDL, there is no lagged dependent variable; unlike
Framework 1 static adstock, the full lag profile is estimated jointly rather
than via a single decay parameter.

Model:
    y[t] = α + Σ_c Σ_{l=0}^{L} w_c[l] · x_c[t-l] + γ·z[t] + ε[t]

where w_c[l] are the lag weights for channel c, constrained to follow
either a Weibull CDF or Almon polynomial shape.

STC/LTC split:
  - STC = weighted media contribution from lags 0..stc_cutoff
  - LTC = weighted media contribution from lags stc_cutoff+1..L

This model explicitly attributes the lag distribution shape to the data,
making it a semi-parametric middle ground between pure geometric adstock
and latent stock models.

Fits better than ARDL when no AR dynamics are needed but lag shape flexibility
is still required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ltc.models.base import BaseLTCModel
from ltc.transforms.weibull import weibull_cdf_weights
from ltc.transforms.almon import almon_compressed_regressors, build_lag_matrix

_EXOG = ["promo", "covid_index", "dgs30", "mobility_index", "competitor_ishare"]


class FiniteDLModel(BaseLTCModel):
    """
    Finite Distributed Lag model with Weibull or Almon lag shape.

    Hyperparameters (via config dict)
    ---------------------------------
    lag_shape : str
        "weibull" (default) or "almon".
    max_lag : int
        Maximum lag L.  Default: 13.
    stc_cutoff : int
        Lag split between STC and LTC.  Default: 4.
    feature : str
        "impressions" or "spend".  Default: "impressions".
    channels : list[str]
        Default: all 5 channels.
    # Weibull-specific
    shape_init : float
        Initial shape parameter.  Default: 1.5.
    scale_init : float
        Initial scale parameter.  Default: 4.0.
    # Almon-specific
    degree : int
        Polynomial degree.  Default: 3.
    """

    name = "finite_dl"
    framework = "F2_dynamic_ts"

    def __init__(self) -> None:
        super().__init__()
        self._lag_shape: str = "weibull"
        self._max_lag: int = 13
        self._stc_cutoff: int = 4
        self._feature: str = "impressions"
        self._channels: list[str] = []
        self._channel_weights: dict[str, np.ndarray] = {}
        self._channel_coefs: dict[str, float] = {}     # overall scale per channel
        self._exog_coefs: np.ndarray | None = None
        self._intercept: float = 0.0
        self._exog_names: list[str] = []

    def fit(self, df: pd.DataFrame, config: dict) -> "FiniteDLModel":
        self._lag_shape = config.get("lag_shape", "weibull")
        self._max_lag = config.get("max_lag", 13)
        self._stc_cutoff = config.get("stc_cutoff", 4)
        self._feature = config.get("feature", "impressions")
        self._channels = config.get("channels", ["tv", "search", "social", "display", "video"])
        exog_cols = [c for c in _EXOG if c in df.columns]
        self._exog_names = exog_cols
        prefix = "impr" if self._feature == "impressions" else "spend"

        y = df["net_sales_observed"].to_numpy(float)
        T = len(y)

        # Step 1: optimise lag shape parameters per channel
        for ch in self._channels:
            col = f"{prefix}_{ch}"
            if col not in df.columns:
                self._channel_weights[ch] = np.zeros(self._max_lag + 1)
                continue
            x_raw = df[col].to_numpy(float)

            if self._lag_shape == "weibull":
                shape_init = config.get("shape_init", 1.5)
                scale_init = config.get("scale_init", 4.0)

                def obj(params: list) -> float:
                    s, sc = params
                    if s <= 0 or sc <= 0:
                        return 1e9
                    try:
                        w = weibull_cdf_weights(s, sc, self._max_lag)
                    except Exception:
                        return 1e9
                    w_full = np.concatenate([[0.0], w])  # lag 0 = first weight
                    if len(w_full) < self._max_lag + 1:
                        w_full = np.pad(w_full, (0, self._max_lag + 1 - len(w_full)))
                    X_lag = build_lag_matrix(x_raw, self._max_lag)
                    adstocked = X_lag @ w_full[:self._max_lag + 1]
                    corr = np.corrcoef(adstocked, y)[0, 1]
                    return -(corr ** 2) if not np.isnan(corr) else 1e9

                res = minimize(obj, [shape_init, scale_init], method="Nelder-Mead")
                s_opt, sc_opt = res.x
                try:
                    w = weibull_cdf_weights(s_opt, sc_opt, self._max_lag)
                    w_full = np.concatenate([[w[0]], w])[:self._max_lag + 1]
                except Exception:
                    w_full = np.ones(self._max_lag + 1) / (self._max_lag + 1)
                self._channel_weights[ch] = w_full

            else:  # almon
                degree = config.get("degree", 3)
                Z, A = almon_compressed_regressors(x_raw, self._max_lag, degree)
                # OLS on compressed Z to get polynomial coefs, recover weights
                coefs_ch, _, _, _ = np.linalg.lstsq(
                    np.column_stack([Z, np.ones(T)]), y, rcond=None
                )
                beta_ch = coefs_ch[:degree + 1]
                self._channel_weights[ch] = A @ beta_ch

        # Step 2: build full regressor matrix and fit joint OLS
        X_parts: list[np.ndarray] = []
        for ch in self._channels:
            col = f"{prefix}_{ch}"
            if col not in df.columns:
                continue
            x_raw = df[col].to_numpy(float)
            X_lag = build_lag_matrix(x_raw, self._max_lag)
            w = self._channel_weights[ch]
            adstocked = X_lag @ w
            X_parts.append(adstocked.reshape(-1, 1))

        if exog_cols:
            X_parts.append(df[exog_cols].to_numpy(float))
        X_parts.append(np.ones((T, 1)))
        X = np.hstack(X_parts)

        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        n_ch = sum(1 for ch in self._channels if f"{prefix}_{ch}" in df.columns)
        for i, ch in enumerate([c for c in self._channels if f"{prefix}_{c}" in df.columns]):
            self._channel_coefs[ch] = float(coefs[i])

        self._exog_coefs = coefs[n_ch: n_ch + len(exog_cols)]
        self._intercept = float(coefs[-1])
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
            x_raw = df[col].to_numpy(float)
            w = self._channel_weights[ch]
            scale = self._channel_coefs.get(ch, 1.0)
            X_lag = build_lag_matrix(x_raw, self._max_lag)
            contrib_per_lag = scale * X_lag * w[np.newaxis, :]
            stc = contrib_per_lag[:, : self._stc_cutoff + 1].sum(axis=1)
            ltc = contrib_per_lag[:, self._stc_cutoff + 1:].sum(axis=1)
            stc_dict[ch] = pd.Series(stc, index=index)
            ltc_dict[ch] = pd.Series(ltc, index=index)

        baseline_val = np.full(T, self._intercept)
        for j, ecol in enumerate(self._exog_names):
            if ecol in df.columns and j < len(self._exog_coefs):
                baseline_val += self._exog_coefs[j] * df[ecol].to_numpy(float)

        baseline = pd.Series(baseline_val, index=index)
        return self._make_decomposition_frame(index, self._channels, baseline, stc_dict, ltc_dict)

    def get_params(self) -> dict:
        self._check_fitted()
        return {
            "model": self.name,
            "lag_shape": self._lag_shape,
            "max_lag": self._max_lag,
            "stc_cutoff": self._stc_cutoff,
            "channel_weights": {ch: w.tolist() for ch, w in self._channel_weights.items()},
            "channel_coefs": self._channel_coefs,
            "intercept": self._intercept,
        }
