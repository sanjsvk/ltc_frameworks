"""
Framework 1 — DualAdstockOLS

Explicitly models STC and LTC as two separate geometric adstocks per channel,
each with its own decay rate:

    STC_adstock[t] = x[t] + θ_stc · STC_adstock[t-1]   (fast decay)
    LTC_adstock[t] = x[t] + θ_ltc · LTC_adstock[t-1]   (slow decay)

The regression then fits two separate coefficients (β_stc, β_ltc) per channel.
This is structurally closer to how the synthetic data was generated, but still
misspecifies the LTC mechanism (geometric instead of latent brand stock) and
constrains both adstocks to the same spend input.

Compared to single-adstock:
  + Explicit STC/LTC separation in the model structure
  + More interpretable per-component coefficients
  - Requires searching a 2D decay grid per channel (computationally heavier)
  - Collinearity between the two adstocks can cause sign flips in coefficients
  - Still fails on S2 (spend pause) because LTC adstock = 0 when spend = 0

Parameter search: joint grid over (θ_stc, θ_ltc) pairs where θ_stc < θ_ltc,
minimising in-sample RSS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product

from ltc.models.base import BaseLTCModel
from ltc.transforms.geometric import geometric_adstock

_EXOG = ["promo", "covid_index", "dgs30", "mobility_index", "competitor_ishare"]


class DualAdstockOLS(BaseLTCModel):
    """
    OLS with separate fast (STC) and slow (LTC) geometric adstocks per channel.

    Hyperparameters (via config dict)
    ---------------------------------
    stc_decay_grid : list[float]
        Candidate short-term decay values.  Default: [0.1, 0.2, 0.3, 0.4, 0.5].
    ltc_decay_grid : list[float]
        Candidate long-term decay values.  Default: [0.5, 0.6, 0.7, 0.8, 0.9].
    feature : str
        "impressions" or "spend".  Default: "impressions".
    channels : list[str]
        Default: all 5 channels.
    """

    name = "dual_adstock"
    framework = "F1_static_adstock"

    def __init__(self) -> None:
        super().__init__()
        self._channel_decays: dict[str, dict] = {}  # {ch: {"stc": θ, "ltc": θ}}
        self._coefs: np.ndarray | None = None
        self._feature_names: list[str] = []
        self._channels: list[str] = []
        self._feature: str = "impressions"

    def fit(self, df: pd.DataFrame, config: dict) -> "DualAdstockOLS":
        stc_grid_cfg = config.get("stc_decay_grid", [0.1, 0.2, 0.3, 0.4, 0.5])
        ltc_grid_cfg = config.get("ltc_decay_grid", [0.5, 0.6, 0.7, 0.8, 0.9])
        enforce_ltc_gt_stc = config.get("enforce_ltc_gt_stc", True)
        self._feature = config.get("feature", "impressions")
        self._channels = config.get("channels", ["tv", "search", "social", "display", "video"])
        exog_cols = [c for c in _EXOG if c in df.columns]

        y = df["net_sales_observed"].to_numpy(dtype=float)
        prefix = "impr" if self._feature == "impressions" else "spend"

        # Find optimal (θ_stc, θ_ltc) per channel — enforce θ_stc < θ_ltc
        for ch in self._channels:
            col = f"{prefix}_{ch}"
            if col not in df.columns:
                self._channel_decays[ch] = {"stc": 0.3, "ltc": 0.7}
                continue

            x_raw = df[col].to_numpy(dtype=float)
            stc_grid = stc_grid_cfg.get(ch, [0.3]) if isinstance(stc_grid_cfg, dict) else stc_grid_cfg
            ltc_grid = ltc_grid_cfg.get(ch, [0.7]) if isinstance(ltc_grid_cfg, dict) else ltc_grid_cfg
            best_pair, best_r2 = (stc_grid[0], ltc_grid[-1]), -np.inf

            for d_stc, d_ltc in product(stc_grid, ltc_grid):
                if enforce_ltc_gt_stc and d_stc >= d_ltc:
                    continue
                a_stc = geometric_adstock(x_raw, d_stc)
                a_ltc = geometric_adstock(x_raw, d_ltc)
                X_ch = np.column_stack([a_stc, a_ltc])
                # Quick OLS R² for this channel alone
                coefs_ch, _, _, _ = np.linalg.lstsq(
                    np.column_stack([X_ch, np.ones(len(y))]), y, rcond=None
                )
                y_hat = np.column_stack([X_ch, np.ones(len(y))]) @ coefs_ch
                ss_res = np.sum((y - y_hat) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                if r2 > best_r2:
                    best_r2 = r2
                    best_pair = (d_stc, d_ltc)

            self._channel_decays[ch] = {"stc": best_pair[0], "ltc": best_pair[1]}

        # Build full regressor matrix
        X_parts: list[np.ndarray] = []
        feature_names: list[str] = []

        for ch in self._channels:
            col = f"{prefix}_{ch}"
            if col not in df.columns:
                continue
            x_raw = df[col].to_numpy(dtype=float)
            d = self._channel_decays[ch]
            X_parts.append(geometric_adstock(x_raw, d["stc"]).reshape(-1, 1))
            X_parts.append(geometric_adstock(x_raw, d["ltc"]).reshape(-1, 1))
            feature_names += [f"stc_adstock_{ch}", f"ltc_adstock_{ch}"]

        for ecol in exog_cols:
            X_parts.append(df[ecol].to_numpy(float).reshape(-1, 1))
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

        coef_idx = 0
        for ch in self._channels:
            col = f"{prefix}_{ch}"
            if col not in df.columns:
                stc_dict[ch] = pd.Series(0.0, index=index)
                ltc_dict[ch] = pd.Series(0.0, index=index)
                coef_idx += 2
                continue
            x_raw = df[col].to_numpy(float)
            d = self._channel_decays[ch]
            stc_ad = geometric_adstock(x_raw, d["stc"])
            ltc_ad = geometric_adstock(x_raw, d["ltc"])
            stc_dict[ch] = pd.Series(self._coefs[coef_idx] * stc_ad, index=index)
            ltc_dict[ch] = pd.Series(self._coefs[coef_idx + 1] * ltc_ad, index=index)
            coef_idx += 2

        baseline_val = np.zeros(T)
        for j in range(coef_idx, len(self._feature_names)):
            name = self._feature_names[j]
            if name == "intercept":
                baseline_val += self._coefs[j]
            elif name in df.columns:
                baseline_val += self._coefs[j] * df[name].to_numpy(float)

        baseline = pd.Series(baseline_val, index=index)
        return self._make_decomposition_frame(index, self._channels, baseline, stc_dict, ltc_dict)

    def get_params(self) -> dict:
        self._check_fitted()
        return {
            "model": self.name,
            "feature": self._feature,
            "channel_decays": self._channel_decays,
            "coefs": dict(zip(self._feature_names, self._coefs.tolist())),
        }
