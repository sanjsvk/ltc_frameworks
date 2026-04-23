"""
Framework 2 — ARDLModel (Autoregressive Distributed Lag)

The ARDL(p, q) model extends OLS by including:
  - p lags of the dependent variable y (autoregressive component)
  - q lags of each media variable x_c (distributed lag component)

    y[t] = α + Σ_i φ_i·y[t-i]  +  Σ_c Σ_j β_{c,j}·x_c[t-j]
             + γ·z[t] + ε[t]

Key advantages over Framework 1:
  + AR terms capture baseline autocorrelation (trend, seasonality momentum)
    without requiring explicit trend/seasonal regressors
  + Distributed lags allow channel-specific lag patterns without assuming
    geometric decay (if q is long enough)
  + Handles S3 (spend-seasonality collinearity) better than pure OLS

Long-run coefficient for channel c:
    LRC_c = (Σ_j β_{c,j}) / (1 - Σ_i φ_i)

Implementation uses statsmodels.tsa.ardl.ARDL for the lag structure.
If statsmodels is unavailable, falls back to manual lag construction.

Failure modes:
  - S2 (spend pause): distributed lag still tied to spend input → underestimates
    LTC during spend pause if q < δ half-life
  - S4 (structural break): single coefficient set averages both regimes
  - High p can overfit on short series
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ltc.models.base import BaseLTCModel
from ltc.transforms.almon import build_lag_matrix, almon_compressed_regressors

_EXOG = ["promo", "covid_index", "dgs30", "mobility_index", "competitor_ishare"]


class ARDLModel(BaseLTCModel):
    """
    Autoregressive Distributed Lag model.

    Hyperparameters (via config dict)
    ---------------------------------
    ar_order : int
        Number of AR lags p.  Default: 4 (one month).
    media_lags : int
        Number of distributed lags q per channel.  Default: 8 (two months).
    stc_cutoff : int
        Lags 0..stc_cutoff of the media DL treated as STC; above as LTC.  Default: 4.
    feature : str
        "impressions" or "spend".  Default: "impressions".
    channels : list[str]
        Default: all 5 channels.
    """

    name = "ardl"
    framework = "F2_dynamic_ts"

    def __init__(self) -> None:
        super().__init__()
        self._ar_order: int = 2
        self._stc_cutoff: int = 6
        self._ltc_degree: int = 2
        self._ltc_max_lag_override: dict[str, int] = {}
        self._feature: str = "impressions"
        self._channels: list[str] = []
        self._coefs: np.ndarray | None = None
        self._feature_names: list[str] = []
        self._start_idx: int = 0
        # Per-channel LTC Almon info for decompose
        self._ch_stc_n: dict[str, int] = {}  # STC lag count per channel
        self._ch_ltc_A: dict[str, np.ndarray] = {}  # Almon basis per channel
        self._ch_ltc_n: dict[str, int] = {}  # LTC compressed cols per channel

    def fit(self, df: pd.DataFrame, config: dict) -> "ARDLModel":
        self._ar_order = config.get("ar_order", 2)
        self._stc_cutoff = config.get("stc_cutoff", 6)
        self._ltc_degree = config.get("ltc_degree", 2)
        self._ltc_max_lag_override = config.get("ltc_max_lag_override", {})
        self._feature = config.get("feature", "impressions")
        self._channels = config.get("channels", ["tv", "search", "social", "display", "video"])
        exog_cols = [c for c in _EXOG if c in df.columns]
        prefix = "impr" if self._feature == "impressions" else "spend"

        y = df["net_sales_observed"].to_numpy(dtype=float)
        T = len(y)

        # Determine max lag across channels for start_idx
        max_ch_lag = max(
            [self._ltc_max_lag_override.get(ch, self._stc_cutoff) for ch in self._channels],
            default=self._stc_cutoff,
        )
        self._start_idx = max(self._ar_order, max_ch_lag)

        X_parts: list[np.ndarray] = []
        feature_names: list[str] = []

        # AR lags of y
        for lag in range(1, self._ar_order + 1):
            ar_col = np.zeros(T)
            ar_col[lag:] = y[:-lag]
            X_parts.append(ar_col.reshape(-1, 1))
            feature_names.append(f"y_lag{lag}")

        # Per-channel: STC flat lags + LTC Almon-compressed lags
        for ch in self._channels:
            col = f"{prefix}_{ch}"
            if col not in df.columns:
                continue
            x_raw = df[col].to_numpy(float)
            ltc_max_lag = self._ltc_max_lag_override.get(ch, self._stc_cutoff)

            # STC: lags 0..stc_cutoff (flat, one coef each)
            stc_n = min(self._stc_cutoff + 1, ltc_max_lag + 1)
            X_stc = build_lag_matrix(x_raw, self._stc_cutoff)[:, :stc_n]
            X_parts.append(X_stc)
            feature_names += [f"{ch}_stc_lag{l}" for l in range(stc_n)]
            self._ch_stc_n[ch] = stc_n

            # LTC: lags stc_cutoff+1..ltc_max_lag, Almon-compressed
            if ltc_max_lag > self._stc_cutoff:
                ltc_lags = ltc_max_lag - self._stc_cutoff  # number of LTC lags
                x_lag_full = build_lag_matrix(x_raw, ltc_max_lag)  # (T, ltc_max_lag+1)
                x_ltc = x_lag_full[:, self._stc_cutoff + 1:]       # (T, ltc_lags)
                # Almon basis for ltc_lags
                Z_ltc, A_ltc = almon_compressed_regressors(x_raw, ltc_max_lag, self._ltc_degree)
                # Use only the LTC part of the Almon system (approximate: compress full range)
                X_parts.append(Z_ltc)
                feature_names += [f"{ch}_ltc_almon{k}" for k in range(self._ltc_degree + 1)]
                self._ch_ltc_A[ch] = A_ltc
                self._ch_ltc_n[ch] = self._ltc_degree + 1
            else:
                self._ch_ltc_A[ch] = np.zeros((0, 0))
                self._ch_ltc_n[ch] = 0

        # Exogenous controls
        for ecol in exog_cols:
            X_parts.append(df[ecol].to_numpy(float).reshape(-1, 1))
            feature_names.append(ecol)

        X_parts.append(np.ones((T, 1)))
        feature_names.append("intercept")

        X = np.hstack(X_parts)
        s = self._start_idx
        X_trim = X[s:]
        y_trim = y[s:]

        coefs, _, _, _ = np.linalg.lstsq(X_trim, y_trim, rcond=None)
        self._coefs = coefs
        self._feature_names = feature_names
        self._is_fitted = True
        return self

    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        prefix = "impr" if self._feature == "impressions" else "spend"
        T = len(df)
        index = df.index
        exog_cols = [c for c in _EXOG if c in df.columns]
        stc_dict: dict[str, pd.Series] = {}
        ltc_dict: dict[str, pd.Series] = {}

        coef_idx = self._ar_order  # skip AR lags

        for ch in self._channels:
            col = f"{prefix}_{ch}"
            if col not in df.columns:
                stc_dict[ch] = pd.Series(0.0, index=index)
                ltc_dict[ch] = pd.Series(0.0, index=index)
                continue

            x_raw = df[col].to_numpy(float)
            stc_n = self._ch_stc_n.get(ch, self._stc_cutoff + 1)
            ltc_n = self._ch_ltc_n.get(ch, 0)
            ltc_max_lag = self._ltc_max_lag_override.get(ch, self._stc_cutoff)

            # STC contribution
            stc_coefs = self._coefs[coef_idx: coef_idx + stc_n]
            X_stc = build_lag_matrix(x_raw, self._stc_cutoff)[:, :stc_n]
            stc = (X_stc * stc_coefs[np.newaxis, :]).sum(axis=1)
            coef_idx += stc_n

            # LTC contribution
            if ltc_n > 0 and ltc_max_lag > self._stc_cutoff:
                ltc_almon_coefs = self._coefs[coef_idx: coef_idx + ltc_n]
                A = self._ch_ltc_A[ch]   # (ltc_max_lag+1, ltc_degree+1)
                ltc_weights = A @ ltc_almon_coefs  # (ltc_max_lag+1,) weights for all lags
                X_full = build_lag_matrix(x_raw, ltc_max_lag)
                ltc = (X_full[:, self._stc_cutoff + 1:] * ltc_weights[self._stc_cutoff + 1:][np.newaxis, :]).sum(axis=1)
                coef_idx += ltc_n
            else:
                ltc = np.zeros(T)

            stc_dict[ch] = pd.Series(stc, index=index)
            ltc_dict[ch] = pd.Series(ltc, index=index)

        # Baseline = AR contribution + exog + intercept
        y_arr = df["net_sales_observed"].to_numpy(float)
        ar_contrib = np.zeros(T)
        for lag in range(1, self._ar_order + 1):
            ar_col = np.zeros(T)
            ar_col[lag:] = y_arr[:-lag]
            ar_contrib += self._coefs[lag - 1] * ar_col

        baseline_val = ar_contrib.copy()
        for j, ecol in enumerate(exog_cols):
            baseline_val += self._coefs[coef_idx + j] * df[ecol].to_numpy(float)
        baseline_val += self._coefs[-1]

        baseline = pd.Series(baseline_val, index=index)
        return self._make_decomposition_frame(index, self._channels, baseline, stc_dict, ltc_dict)

    def get_params(self) -> dict:
        self._check_fitted()
        return {
            "model": self.name,
            "ar_order": self._ar_order,
            "stc_cutoff": self._stc_cutoff,
            "ltc_degree": self._ltc_degree,
            "ltc_max_lag_override": self._ltc_max_lag_override,
            "coefs": dict(zip(self._feature_names, self._coefs.tolist())),
        }
