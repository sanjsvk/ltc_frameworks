"""
Framework 3 — BayesianStructuralTS (Bayesian Structural Time Series)

A BSTS model decomposes sales into explicit structural components:
  - Local linear trend (level + slope, both random walks)
  - Seasonal component (weekly or annual — modelled via trigonometric terms)
  - Regression component (media + exogenous regressors)

Unlike KalmanDLM (which uses static media coefficients + time-varying level),
BSTS allows all structural components to evolve simultaneously, providing a
richer decomposition of organic vs. media-driven sales.

Model:
    y[t]     = μ[t] + τ[t] + β·X[t] + ε[t]     ε ~ N(0, σ²_ε)
    μ[t]     = μ[t-1] + δ[t-1] + η[t]           η ~ N(0, σ²_μ)   (level)
    δ[t]     = δ[t-1] + ζ[t]                     ζ ~ N(0, σ²_δ)   (slope)
    τ[t]     = -Σ_{j=1}^{s-1} τ[t-j] + ω[t]    ω ~ N(0, σ²_τ)   (seasonal)

Implementation:
  This module implements a simplified BSTS via the Kalman filter with a
  trigonometric seasonal component.  For a full Bayesian treatment with
  spike-and-slab variable selection, see the R `bsts` package or
  install `orbit-ml` (Python).

  If `orbit` is installed, the model can delegate to it directly.
  Otherwise, falls back to the custom Kalman implementation.

BSTS advantages:
  + Full structural decomposition (trend, seasonal, media) with uncertainty
  + Spike-and-slab priors for media variable selection (full model)
  + Natural handling of S4 (structural break) via time-varying level
  + Posterior predictive checks for model validation

References:
    Scott & Varian (2014) "Predicting the Present with Bayesian Structural
    Time Series"; Harvey (1990) Forecasting, Structural Time Series Models.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from ltc.models.base import BaseLTCModel
from ltc.transforms.geometric import geometric_adstock

_EXOG = ["promo", "covid_index", "dgs30", "mobility_index", "competitor_ishare"]
_CHANNELS = ["tv", "search", "social", "display", "video"]


class BayesianStructuralTS(BaseLTCModel):
    """
    Bayesian Structural Time Series model with local linear trend + seasonal.

    Hyperparameters (via config dict)
    ---------------------------------
    level_var : float
        Level innovation variance σ²_μ.  Default: 0.05.
    slope_var : float
        Slope innovation variance σ²_δ.  Default: 0.005.
    seasonal_periods : int
        Number of seasonal harmonics.  Default: 2 (captures annual cycle).
    obs_var_init : float
        Initial observation noise variance.  Default: estimated from data.
    stc_cutoff : int
        Lag split for STC/LTC attribution.  Default: 4.
    decay_per_channel : dict[str, float]
        Adstock decays for media.
    feature : str
        "impressions" or "spend".  Default: "impressions".
    channels : list[str]
    backend : str
        "custom" (Kalman, default) or "orbit" (orbit-ml, if installed).
    """

    name = "bsts"
    framework = "F3_state_space"

    _DEFAULT_DECAYS = {"tv": 0.55, "search": 0.20, "social": 0.45, "display": 0.50, "video": 0.60}

    def __init__(self) -> None:
        super().__init__()
        self._media_coefs: dict[str, float] = {}
        self._exog_coefs: np.ndarray | None = None
        self._exog_names: list[str] = []
        self._trend_series: np.ndarray | None = None   # smoothed level
        self._seasonal_series: np.ndarray | None = None
        self._channels: list[str] = []
        self._feature: str = "impressions"
        self._decays: dict[str, float] = {}
        self._stc_cutoff: int = 4

    def fit(self, df: pd.DataFrame, config: dict) -> "BayesianStructuralTS":
        level_var = config.get("level_var", 0.05)
        slope_var = config.get("slope_var", 0.005)
        n_harmonics = config.get("seasonal_periods", 2)
        self._channels = config.get("channels", _CHANNELS)
        self._feature = config.get("feature", "impressions")
        self._stc_cutoff = config.get("stc_cutoff", 6)
        # Support new stc_decay_per_channel key; fall back to legacy decay_per_channel
        self._decays = config.get("stc_decay_per_channel", config.get("decay_per_channel", self._DEFAULT_DECAYS))
        exog_cols = [c for c in _EXOG if c in df.columns]
        self._exog_names = exog_cols

        backend = config.get("backend", "custom")
        if backend == "orbit":
            self._fit_orbit(df, config)
        else:
            self._fit_custom_kalman(df, config, level_var, slope_var, n_harmonics)

        self._is_fitted = True
        return self

    def _fit_custom_kalman(
        self,
        df: pd.DataFrame,
        config: dict,
        level_var: float,
        slope_var: float,
        n_harmonics: int,
    ) -> None:
        """
        Custom Kalman filter BSTS with trigonometric seasonal component.

        State vector: [level, slope, cos_1, sin_1, cos_2, sin_2, ...]
        """
        prefix = "impr" if self._feature == "impressions" else "spend"
        exog_cols = self._exog_names
        y = df["net_sales_observed"].to_numpy(float)
        T = len(y)

        # Pre-estimate media and exog coefficients via OLS
        X_parts: list[np.ndarray] = []
        ch_included: list[str] = []
        for ch in self._channels:
            col = f"{prefix}_{ch}"
            if col in df.columns:
                d = self._decays.get(ch, 0.5)
                ad = geometric_adstock(df[col].to_numpy(float), d)
                X_parts.append(ad.reshape(-1, 1))
                ch_included.append(ch)

        # Fourier seasonal regressors (52-week annual cycle)
        t_vec = np.arange(T, dtype=float)
        for h in range(1, n_harmonics + 1):
            X_parts.append(np.cos(2 * np.pi * h * t_vec / 52).reshape(-1, 1))
            X_parts.append(np.sin(2 * np.pi * h * t_vec / 52).reshape(-1, 1))

        if exog_cols:
            X_parts.append(df[exog_cols].to_numpy(float))
        X_parts.append(np.ones((T, 1)))

        X_ols = np.hstack(X_parts)
        coefs, _, _, _ = np.linalg.lstsq(X_ols, y, rcond=None)

        for i, ch in enumerate(ch_included):
            self._media_coefs[ch] = float(coefs[i])

        n_ch = len(ch_included)
        n_seas = 2 * n_harmonics
        seas_coefs = coefs[n_ch: n_ch + n_seas]
        n_exog = len(exog_cols)
        self._exog_coefs = coefs[n_ch + n_seas: n_ch + n_seas + n_exog]

        # Residual after removing media and exog
        media_contrib = sum(
            self._media_coefs[ch] * geometric_adstock(df[f"{prefix}_{ch}"].to_numpy(float), self._decays.get(ch, 0.5))
            for ch in ch_included if f"{prefix}_{ch}" in df.columns
        )
        seas_contrib = np.zeros(T)
        for h in range(n_harmonics):
            seas_contrib += seas_coefs[2 * h] * np.cos(2 * np.pi * (h + 1) * t_vec / 52)
            seas_contrib += seas_coefs[2 * h + 1] * np.sin(2 * np.pi * (h + 1) * t_vec / 52)
        exog_contrib = sum(
            self._exog_coefs[j] * df[ecol].to_numpy(float)
            for j, ecol in enumerate(exog_cols)
            if ecol in df.columns
        ) if exog_cols else 0.0

        y_resid = y - media_contrib - seas_contrib - exog_contrib

        # Kalman filter on residual for local linear trend
        G = np.array([[1.0, 1.0], [0.0, 1.0]])
        F = np.array([[1.0, 0.0]])
        W = np.diag([level_var, slope_var])
        obs_var = config.get("obs_var_init") or np.var(y_resid) * 0.1
        V = np.array([[obs_var]])

        m = np.array([y_resid[0], 0.0])
        C = np.eye(2) * 10.0
        m_filt = np.zeros((T, 2))
        C_filt = np.zeros((T, 2, 2))

        for t in range(T):
            m_pred = G @ m
            C_pred = G @ C @ G.T + W
            S = F @ C_pred @ F.T + V
            K = C_pred @ F.T / S[0, 0]
            m = m_pred + K.flatten() * (y_resid[t] - (F @ m_pred)[0])
            C = (np.eye(2) - np.outer(K.flatten(), F.flatten())) @ C_pred
            m_filt[t] = m
            C_filt[t] = C

        # RTS smoother
        m_smooth = np.zeros((T, 2))
        m_smooth[-1] = m_filt[-1]
        for t in range(T - 2, -1, -1):
            m_pred = G @ m_filt[t]
            C_pred = G @ C_filt[t] @ G.T + W
            L = C_filt[t] @ G.T @ np.linalg.inv(C_pred)
            m_smooth[t] = m_filt[t] + L @ (m_smooth[t + 1] - m_pred)

        self._trend_series = m_smooth[:, 0]
        self._seasonal_series = seas_contrib

    def _fit_orbit(self, df: pd.DataFrame, config: dict) -> None:
        """Delegate to orbit-ml if installed."""
        try:
            from orbit.models import DLT
        except ImportError:
            warnings.warn("orbit-ml not installed — falling back to custom Kalman.", stacklevel=2)
            level_var = config.get("level_var", 0.05)
            slope_var = config.get("slope_var", 0.005)
            n_harmonics = config.get("seasonal_periods", 2)
            self._fit_custom_kalman(df, config, level_var, slope_var, n_harmonics)
            return
        # orbit integration placeholder — implement when orbit API is confirmed
        raise NotImplementedError("orbit-ml integration is not yet implemented")

    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        prefix = "impr" if self._feature == "impressions" else "spend"
        T = len(df)
        index = df.index
        stc_dict: dict[str, pd.Series] = {}
        ltc_dict: dict[str, pd.Series] = {}

        for ch in self._channels:
            col = f"{prefix}_{ch}"
            if col not in df.columns or ch not in self._media_coefs:
                stc_dict[ch] = pd.Series(0.0, index=index)
                ltc_dict[ch] = pd.Series(0.0, index=index)
                continue
            x_raw = df[col].to_numpy(float)
            d = self._decays.get(ch, 0.5)
            ad = geometric_adstock(x_raw, d)
            total = self._media_coefs[ch] * ad
            stc = self._media_coefs[ch] * x_raw
            ltc = total - stc
            stc_dict[ch] = pd.Series(stc, index=index)
            ltc_dict[ch] = pd.Series(ltc, index=index)

        baseline_val = (self._trend_series.copy() if self._trend_series is not None else np.zeros(T))
        if self._seasonal_series is not None:
            baseline_val = baseline_val + self._seasonal_series
        for j, ecol in enumerate(self._exog_names):
            if ecol in df.columns and self._exog_coefs is not None and j < len(self._exog_coefs):
                baseline_val = baseline_val + self._exog_coefs[j] * df[ecol].to_numpy(float)

        baseline = pd.Series(baseline_val, index=index)
        return self._make_decomposition_frame(index, self._channels, baseline, stc_dict, ltc_dict)

    def get_params(self) -> dict:
        self._check_fitted()
        return {
            "model": self.name,
            "media_coefs": self._media_coefs,
            "decays": self._decays,
        }
