"""
Framework 1 — WeibullAdstockNLS

Replaces the geometric adstock (fixed exponential decay) with a Weibull CDF
adstock, which allows a more flexible lag shape: the peak media response can
be delayed by 1-3 weeks (shape > 1) rather than forced to week 0.

This is the adstock form used by Robyn (Meta's open-source MMM) and by
Google's LightweightMMM.  It adds two parameters per channel (shape, scale)
over the geometric baseline, at the cost of nonlinear optimisation.

Estimation: scipy.optimize.minimize (L-BFGS-B) minimises RSS over (shape, scale)
per channel with OLS closed-form for the linear coefficients at each iterate
(a "nested" or "profile likelihood" optimisation).

Failure modes (same as geometric, plus):
  - Larger parameter space → more local optima and overfitting on short series.
  - Still cannot separate STC from LTC without explicit structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ltc.models.base import BaseLTCModel
from ltc.transforms.weibull import weibull_adstock

_EXOG = ["promo", "covid_index", "dgs30", "mobility_index", "competitor_ishare"]


class WeibullAdstockNLS(BaseLTCModel):
    """
    OLS regression with per-channel Weibull CDF adstock (shape + scale optimised).

    Hyperparameters (via config dict)
    ---------------------------------
    max_lag : int
        Truncation window for Weibull weights.  Default: 20 weeks.
    feature : str
        "impressions" or "spend".  Default: "impressions".
    channels : list[str]
        Default: all 5 channels.
    shape_bounds : (float, float)
        Search bounds for Weibull shape parameter.  Default: (0.5, 5.0).
    scale_bounds : (float, float)
        Search bounds for Weibull scale parameter (weeks).  Default: (1.0, 15.0).
    """

    name = "weibull_adstock"
    framework = "F1_static_adstock"

    def __init__(self) -> None:
        super().__init__()
        self._channel_params: dict[str, dict] = {}  # {ch: {"shape": ..., "scale": ...}}
        self._coefs: np.ndarray | None = None
        self._feature_names: list[str] = []
        self._channels: list[str] = []
        self._feature: str = "impressions"
        self._max_lag: int = 20

    def fit(self, df: pd.DataFrame, config: dict) -> "WeibullAdstockNLS":
        self._feature = config.get("feature", "impressions")
        self._channels = config.get("channels", ["tv", "search", "social", "display", "video"])
        self._max_lag = config.get("max_lag", 52)
        shape_bounds_cfg = config.get("shape_bounds", (0.8, 3.0))
        scale_bounds_cfg = config.get("scale_bounds", (1.0, 15.0))
        exog_cols = [c for c in _EXOG if c in df.columns]

        y = df["net_sales_observed"].to_numpy(dtype=float)
        prefix = "impr" if self._feature == "impressions" else "spend"

        # Optimise shape + scale per channel via profile likelihood
        for ch in self._channels:
            col = f"{prefix}_{ch}"
            if col not in df.columns:
                self._channel_params[ch] = {"shape": 1.0, "scale": 4.0}
                continue

            x_raw = df[col].to_numpy(dtype=float)
            ch_shape_bounds = shape_bounds_cfg.get(ch, (0.8, 3.0)) if isinstance(shape_bounds_cfg, dict) else shape_bounds_cfg
            ch_scale_bounds = scale_bounds_cfg.get(ch, (1.0, 15.0)) if isinstance(scale_bounds_cfg, dict) else scale_bounds_cfg
            ch_shape_init = float(sum(ch_shape_bounds)) / 2
            ch_scale_init = float(sum(ch_scale_bounds)) / 2

            def neg_r2(params: list[float]) -> float:
                shape, scale = params
                try:
                    adstocked = weibull_adstock(x_raw, shape, scale, self._max_lag)
                except Exception:
                    return 1e6
                corr = np.corrcoef(adstocked, y)[0, 1]
                return -(corr ** 2)

            result = minimize(
                neg_r2,
                x0=[ch_shape_init, ch_scale_init],
                bounds=[ch_shape_bounds, ch_scale_bounds],
                method="L-BFGS-B",
            )
            self._channel_params[ch] = {"shape": result.x[0], "scale": result.x[1]}

        # Build full regressor matrix and fit OLS
        X_parts: list[np.ndarray] = []
        feature_names: list[str] = []

        for ch in self._channels:
            col = f"{prefix}_{ch}"
            if col in df.columns:
                p = self._channel_params[ch]
                adstocked = weibull_adstock(
                    df[col].to_numpy(dtype=float), p["shape"], p["scale"], self._max_lag
                )
                X_parts.append(adstocked.reshape(-1, 1))
                feature_names.append(f"weibull_adstock_{ch}")

        for ecol in exog_cols:
            X_parts.append(df[ecol].to_numpy(dtype=float).reshape(-1, 1))
            feature_names.append(ecol)

        X_parts.append(np.ones((len(y), 1)))
        feature_names.append("intercept")

        X = np.hstack(X_parts)
        self._feature_names = feature_names
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self._coefs = coefs
        self._is_fitted = True
        return self

    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        prefix = "impr" if self._feature == "impressions" else "spend"
        T = len(df)
        index = df.index
        stc_dict: dict[str, pd.Series] = {}
        ltc_dict: dict[str, pd.Series] = {}

        for i, ch in enumerate(self._channels):
            col = f"{prefix}_{ch}"
            if col not in df.columns:
                stc_dict[ch] = pd.Series(0.0, index=index)
                ltc_dict[ch] = pd.Series(0.0, index=index)
                continue
            p = self._channel_params[ch]
            x_raw = df[col].to_numpy(dtype=float)
            adstocked = weibull_adstock(x_raw, p["shape"], p["scale"], self._max_lag)
            coef = self._coefs[i]
            total = coef * adstocked
            # STC = peak-week response (weight[0] × x[t] × coef)
            # Approximate: weight[0] is the first Weibull CDF weight
            from ltc.transforms.weibull import weibull_cdf_weights
            w = weibull_cdf_weights(p["shape"], p["scale"], self._max_lag)
            stc = coef * w[0] * x_raw
            ltc = total - stc
            stc_dict[ch] = pd.Series(stc, index=index)
            ltc_dict[ch] = pd.Series(ltc, index=index)

        baseline_val = np.zeros(T)
        for j, name in enumerate(self._feature_names):
            if name == "intercept":
                baseline_val += self._coefs[j]
            elif name in _EXOG and name in df.columns:
                baseline_val += self._coefs[j] * df[name].to_numpy(dtype=float)

        baseline = pd.Series(baseline_val, index=index)
        return self._make_decomposition_frame(index, self._channels, baseline, stc_dict, ltc_dict)

    def get_params(self) -> dict:
        self._check_fitted()
        return {
            "model": self.name,
            "feature": self._feature,
            "max_lag": self._max_lag,
            "channel_params": self._channel_params,
            "coefs": dict(zip(self._feature_names, self._coefs.tolist())),
        }
