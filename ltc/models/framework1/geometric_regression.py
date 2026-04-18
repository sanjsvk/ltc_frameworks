"""
Framework 1 — GeometricAdstockOLS

The simplest and most widely deployed MMM model.  A single geometric adstock
decay is applied to each channel's impressions (or spend), then the adstocked
media variables plus exogenous controls are regressed on net sales via OLS.

Key assumption: all media response (STC + LTC) is captured by a single
decaying impression coefficient.  The decay parameter θ determines how
much of last week's carryover persists.  Because a single adstock encodes
both the contemporaneous and accumulated effect, there is no explicit
STC/LTC split — the split is post-hoc via the decompose() method, which
attributes the immediate-period effect to STC and the carryover portion to LTC.

Failure modes:
  - S2 (spend pause): LTC continues post-pause from brand stock, but
    geometric adstock = 0 when spend = 0, so LTC is underestimated.
  - S3 (collinearity): OLS cannot separate seasonally-correlated media
    from organic baseline; media coefficient inflated.
  - S4 (structural break): single θ averages both regimes — wrong in both.

Parameter optimisation:
  - `decay` is grid-searched over config["decay_grid"] using in-sample R².
  - Or `scipy.optimize.minimize_scalar` on negative R² for continuous search.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ltc.models.base import BaseLTCModel
from ltc.transforms.geometric import geometric_adstock

_EXOG = ["promo", "covid_index", "dgs30", "mobility_index", "competitor_ishare"]


class GeometricAdstockOLS(BaseLTCModel):
    """
    OLS regression with a single geometric adstock per channel.

    Hyperparameters (via config dict)
    ---------------------------------
    decay_grid : list[float], optional
        Discrete decay values to grid-search.  Default: 0.1 to 0.9 in 0.1 steps.
    feature : str
        "impressions" (default) or "spend" — the media input to adstock.
    channels : list[str]
        Channel names.  Default: all 5 channels.
    fit_intercept : bool
        Whether to include an intercept (baseline level).  Default: True.
    """

    name = "geo_adstock"
    framework = "F1_static_adstock"

    def __init__(self) -> None:
        super().__init__()
        # Fitted attributes (set by fit())
        self._decay: float | None = None            # per-model single shared decay
        self._channel_decays: dict[str, float] = {} # per-channel optimal decay
        self._coefs: np.ndarray | None = None        # OLS coefficients
        self._feature_names: list[str] = []
        self._channels: list[str] = []
        self._feature: str = "impressions"
        self._fit_intercept: bool = True

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, config: dict) -> "GeometricAdstockOLS":
        """
        Grid-search optimal decay per channel, then fit joint OLS.

        The search minimises in-sample residual sum of squares over the
        `decay_grid` values, fitting a separate optimal decay for each channel.
        """
        decay_grid = config.get("decay_grid", [round(x * 0.1, 1) for x in range(1, 10)])
        self._feature = config.get("feature", "impressions")
        self._channels = config.get("channels", ["tv", "search", "social", "display", "video"])
        self._fit_intercept = config.get("fit_intercept", True)
        exog_cols = [c for c in _EXOG if c in df.columns]

        y = df["net_sales_observed"].to_numpy(dtype=float)
        prefix = "impr" if self._feature == "impressions" else "spend"

        # Step 1: find optimal decay per channel via grid search
        for ch in self._channels:
            col = f"{prefix}_{ch}"
            if col not in df.columns:
                self._channel_decays[ch] = 0.5
                continue
            x_raw = df[col].to_numpy(dtype=float)

            best_decay, best_r2 = 0.5, -np.inf
            for d in decay_grid:
                adstocked = geometric_adstock(x_raw, d)
                # Quick univariate R² (controls not included at this stage)
                corr = np.corrcoef(adstocked, y)[0, 1]
                r2 = corr ** 2
                if r2 > best_r2:
                    best_r2, best_decay = r2, d
            self._channel_decays[ch] = best_decay

        # Step 2: build full regressor matrix with optimal adstocks + exog
        X_parts: list[np.ndarray] = []
        feature_names: list[str] = []

        for ch in self._channels:
            col = f"{prefix}_{ch}"
            if col in df.columns:
                x_raw = df[col].to_numpy(dtype=float)
                adstocked = geometric_adstock(x_raw, self._channel_decays[ch])
                X_parts.append(adstocked.reshape(-1, 1))
                feature_names.append(f"adstock_{ch}")

        for ecol in exog_cols:
            X_parts.append(df[ecol].to_numpy(dtype=float).reshape(-1, 1))
            feature_names.append(ecol)

        if self._fit_intercept:
            X_parts.append(np.ones((len(y), 1)))
            feature_names.append("intercept")

        X = np.hstack(X_parts)
        self._feature_names = feature_names

        # Step 3: OLS via least-squares
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self._coefs = coefs
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # decompose
    # ------------------------------------------------------------------
    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split fitted sales into baseline, per-channel STC and LTC.

        STC[ch][t]  = coef[ch] * adstocked[t] - coef[ch] * decay * adstocked[t-1]
                    ≈ coef[ch] * raw_impr[t]   (contemporaneous effect only)
        LTC[ch][t]  = coef[ch] * decay * adstocked[t-1]
                    = total adstock contribution minus STC
        """
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

            x_raw = df[col].to_numpy(dtype=float)
            d = self._channel_decays[ch]
            adstocked = geometric_adstock(x_raw, d)
            coef = self._coefs[i]

            total_contrib = coef * adstocked
            # STC = contemporaneous response = coef * raw media
            stc = coef * x_raw
            # LTC = carryover portion = total - contemporaneous
            ltc = total_contrib - stc

            stc_dict[ch] = pd.Series(stc, index=index)
            ltc_dict[ch] = pd.Series(ltc, index=index)

        # Baseline = intercept + exogenous effects
        baseline_val = np.zeros(T)
        n_media = len(self._channels)
        for j, name in enumerate(self._feature_names):
            if name == "intercept":
                baseline_val += self._coefs[j]
            elif name in _EXOG and name in df.columns:
                baseline_val += self._coefs[j] * df[name].to_numpy(dtype=float)

        baseline = pd.Series(baseline_val, index=index)
        return self._make_decomposition_frame(index, self._channels, baseline, stc_dict, ltc_dict)

    # ------------------------------------------------------------------
    # get_params
    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        self._check_fitted()
        return {
            "model": self.name,
            "feature": self._feature,
            "channel_decays": self._channel_decays,
            "coefs": dict(zip(self._feature_names, self._coefs.tolist())),
        }
