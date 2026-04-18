"""
Framework 3 — KalmanDLM (Dynamic Linear Model via Kalman Filter)

A Dynamic Linear Model treats the underlying sales process as a latent
state-space system with a local linear trend component and time-varying
regression coefficients:

    Observation:  y[t] = F[t]' · θ[t] + v[t],    v[t] ~ N(0, V)
    State:        θ[t] = G · θ[t-1] + w[t],       w[t] ~ N(0, W)

Where θ[t] contains the local level, slope, and media coefficients.
The Kalman filter recursively estimates θ[t] forward in time; the smoother
(RTS) recovers the full posterior over the state series.

Benefits over F1/F2:
  + Time-varying coefficients can adapt to structural breaks (S4)
  + The latent level absorbs trend and seasonality without explicit coding
  + Uncertainty propagates naturally — credible intervals on all components
  - Does not explicitly model the brand stock accumulation (so S2 partial)
  - Requires tuning of W (state evolution variance) — under-specified W
    collapses to static OLS; over-specified W overfits week-to-week

Implementation: scipy-based Kalman filter and smoother.
Media coefficients are assumed fixed (state evolution variance = 0) to
keep the model estimable on 261 weeks; only the level/slope evolve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import solve

from ltc.models.base import BaseLTCModel
from ltc.transforms.geometric import geometric_adstock

_EXOG = ["promo", "covid_index", "dgs30", "mobility_index", "competitor_ishare"]
_CHANNELS = ["tv", "search", "social", "display", "video"]


class KalmanDLM(BaseLTCModel):
    """
    Kalman Filter Dynamic Linear Model with local linear trend + media regressors.

    Media coefficients are fixed (estimated via OLS pre-fit); the local level
    and slope evolve as a random walk with drift.  Media contributions are then
    decomposed by attributing y to each media channel via its coefficient × input.

    Hyperparameters (via config dict)
    ---------------------------------
    level_var : float
        State evolution variance for the local level.  Default: 0.01.
    slope_var : float
        State evolution variance for the slope.  Default: 0.001.
    obs_var : float
        Observation noise variance.  If None, estimated via EM-style iteration.
    decay_per_channel : dict[str, float]
        Geometric adstock decay per channel applied before regression.
        Default: {"tv":0.55, "search":0.20, "social":0.45, "display":0.50, "video":0.60}
    stc_cutoff_weeks : int
        Weeks of carryover attributed to STC.  Default: 4.
    feature : str
        "impressions" or "spend".  Default: "impressions".
    channels : list[str]
    """

    name = "kalman_dlm"
    framework = "F3_state_space"

    # Default adstock decays (STC-range) from synthetic data spec
    _DEFAULT_DECAYS = {"tv": 0.55, "search": 0.20, "social": 0.45, "display": 0.50, "video": 0.60}

    def __init__(self) -> None:
        super().__init__()
        self._media_coefs: dict[str, float] = {}
        self._exog_coefs: np.ndarray | None = None
        self._exog_names: list[str] = []
        self._level_series: np.ndarray | None = None   # smoothed local level
        self._channels: list[str] = []
        self._feature: str = "impressions"
        self._decays: dict[str, float] = {}
        self._stc_cutoff: int = 4
        self._obs_var: float = 1.0

    def fit(self, df: pd.DataFrame, config: dict) -> "KalmanDLM":
        level_var = config.get("level_var", 0.01)
        slope_var = config.get("slope_var", 0.001)
        self._obs_var = config.get("obs_var", None)
        self._channels = config.get("channels", _CHANNELS)
        self._feature = config.get("feature", "impressions")
        self._stc_cutoff = config.get("stc_cutoff_weeks", 4)
        self._decays = config.get("decay_per_channel", self._DEFAULT_DECAYS)
        exog_cols = [c for c in _EXOG if c in df.columns]
        self._exog_names = exog_cols
        prefix = "impr" if self._feature == "impressions" else "spend"

        y = df["net_sales_observed"].to_numpy(float)
        T = len(y)

        # Step 1: Pre-compute adstocked media features for OLS initialisation
        adstocked: dict[str, np.ndarray] = {}
        for ch in self._channels:
            col = f"{prefix}_{ch}"
            if col in df.columns:
                d = self._decays.get(ch, 0.5)
                adstocked[ch] = geometric_adstock(df[col].to_numpy(float), d)

        # Step 2: OLS to get initial media and exog coefficients
        X_parts: list[np.ndarray] = []
        for ch in self._channels:
            if ch in adstocked:
                X_parts.append(adstocked[ch].reshape(-1, 1))
        if exog_cols:
            X_parts.append(df[exog_cols].to_numpy(float))
        X_parts.append(np.ones((T, 1)))
        X_ols = np.hstack(X_parts)
        coefs_ols, _, _, _ = np.linalg.lstsq(X_ols, y, rcond=None)

        ch_list = [ch for ch in self._channels if ch in adstocked]
        for i, ch in enumerate(ch_list):
            self._media_coefs[ch] = float(coefs_ols[i])
        n_ch = len(ch_list)
        self._exog_coefs = coefs_ols[n_ch: n_ch + len(exog_cols)]

        # Step 3: Subtract media and exog contributions → residual for DLM
        media_contrib = np.zeros(T)
        for ch in ch_list:
            media_contrib += self._media_coefs[ch] * adstocked[ch]
        exog_contrib = np.zeros(T)
        for j, ecol in enumerate(exog_cols):
            exog_contrib += self._exog_coefs[j] * df[ecol].to_numpy(float)
        y_resid = y - media_contrib - exog_contrib

        # Step 4: Kalman filter on residual (local linear trend)
        # State = [level, slope]
        # Transition: G = [[1,1],[0,1]]
        # Observation: F = [1, 0]
        G = np.array([[1.0, 1.0], [0.0, 1.0]])
        F = np.array([[1.0, 0.0]])
        W = np.diag([level_var, slope_var])

        obs_var_init = self._obs_var if self._obs_var is not None else np.var(y_resid) * 0.1
        V = np.array([[obs_var_init]])

        # Initialise
        m = np.array([y_resid[0], 0.0])  # [level, slope]
        C = np.eye(2) * 10.0

        # Forward pass (filter)
        m_filt = np.zeros((T, 2))
        C_filt = np.zeros((T, 2, 2))

        for t in range(T):
            # Predict
            m_pred = G @ m
            C_pred = G @ C @ G.T + W
            # Update
            S = F @ C_pred @ F.T + V
            K = C_pred @ F.T @ np.linalg.inv(S)
            innovation = y_resid[t] - (F @ m_pred)[0]
            m = m_pred + (K * innovation).flatten()
            C = (np.eye(2) - K @ F) @ C_pred
            m_filt[t] = m
            C_filt[t] = C

        # Backward pass (smoother — RTS smoother)
        m_smooth = np.zeros((T, 2))
        m_smooth[-1] = m_filt[-1]
        for t in range(T - 2, -1, -1):
            m_pred = G @ m_filt[t]
            C_pred = G @ C_filt[t] @ G.T + W
            L = C_filt[t] @ G.T @ np.linalg.inv(C_pred)
            m_smooth[t] = m_filt[t] + L @ (m_smooth[t + 1] - m_pred)

        self._level_series = m_smooth[:, 0]  # smoothed local level
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
            if col not in df.columns or ch not in self._media_coefs:
                stc_dict[ch] = pd.Series(0.0, index=index)
                ltc_dict[ch] = pd.Series(0.0, index=index)
                continue
            x_raw = df[col].to_numpy(float)
            d = self._decays.get(ch, 0.5)
            adstocked = geometric_adstock(x_raw, d)
            total = self._media_coefs[ch] * adstocked
            # Approximate STC/LTC split via decay
            stc = self._media_coefs[ch] * x_raw
            ltc = total - stc
            stc_dict[ch] = pd.Series(stc, index=index)
            ltc_dict[ch] = pd.Series(ltc, index=index)

        # Baseline = smoothed local level + exogenous
        baseline_val = self._level_series.copy()
        for j, ecol in enumerate(self._exog_names):
            if ecol in df.columns and j < len(self._exog_coefs):
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
