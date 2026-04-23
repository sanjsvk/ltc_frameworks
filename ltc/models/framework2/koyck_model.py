"""
Framework 2 — KoyckModel

The Koyck transformation converts an infinite geometric distributed lag model
into an estimable equation by introducing the lagged dependent variable y[t-1]:

    y[t] = α(1-λ) + β₀·x[t] + λ·y[t-1] + (ε[t] - λ·ε[t-1])

where λ is the "Koyck lambda" — the lag decay shared across all periods.

Compared to Framework 1 (geometric adstock):
  + Implicitly captures all historical media exposure via y[t-1]
  + Baseline autocorrelation (trend, seasonality) is partially absorbed by λ·y[t-1]
  + Long-run multiplier β₀/(1-λ) emerges naturally
  - λ is shared across ALL channels (not channel-specific)
  - MA(1) error structure makes OLS inconsistent — HAC SEs used here
  - Lagged DV can proxy for missing variables (omitted variable bias)
  - Single λ cannot adapt to structural breaks (S4) or distinguish
    channel-specific carryover rates

For multi-channel media, the model becomes:
    y[t] = α + Σ_c β_c·x_c[t] + λ·y[t-1] + ε[t]

The shared λ is estimated by grid search (minimising RSS); β_c and α are
estimated via OLS conditional on λ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ltc.models.base import BaseLTCModel
from ltc.transforms.koyck import koyck_regressors, koyck_long_run_multiplier

_EXOG = ["promo", "covid_index", "dgs30", "mobility_index", "competitor_ishare"]


class KoyckModel(BaseLTCModel):
    """
    Koyck distributed lag model with a shared lambda across all channels.

    Hyperparameters (via config dict)
    ---------------------------------
    lambda_grid : list[float]
        Candidate λ values.  Default: 0.1 to 0.9 in 0.05 steps.
    feature : str
        "impressions" or "spend".  Default: "impressions".
    channels : list[str]
        Default: all 5 channels.
    """

    name = "koyck"
    framework = "F2_dynamic_ts"

    def __init__(self) -> None:
        super().__init__()
        self._lambda: float | None = None
        self._coefs: np.ndarray | None = None   # [β_ch…, α]  (lambda absorbed)
        self._feature_names: list[str] = []
        self._channels: list[str] = []
        self._feature: str = "impressions"
        self._T: int = 0

    def fit(self, df: pd.DataFrame, config: dict) -> "KoyckModel":
        lambda_grid_cfg = config.get("lambda_grid", [round(0.1 + i * 0.05, 2) for i in range(17)])
        self._feature = config.get("feature", "impressions")
        self._channels = config.get("channels", ["tv", "search", "social", "display", "video"])
        ar_order = config.get("ar_order", 1)
        exog_cols = [c for c in _EXOG if c in df.columns]
        prefix = "impr" if self._feature == "impressions" else "spend"

        # Flatten per-channel lambda grid dict into sorted unique set
        if isinstance(lambda_grid_cfg, dict):
            all_lams: set[float] = set()
            for vals in lambda_grid_cfg.values():
                all_lams.update(vals)
            lambda_grid = sorted(all_lams)
        else:
            lambda_grid = lambda_grid_cfg

        y = df["net_sales_observed"].to_numpy(dtype=float)
        self._T = len(y)

        # Build current-period media matrix (T, C)
        ch_present = [ch for ch in self._channels if f"{prefix}_{ch}" in df.columns]
        X_media = df[[f"{prefix}_{ch}" for ch in ch_present]].to_numpy(dtype=float)

        # Build regressor matrix: media + AR lags + exog + intercept
        # Koyck: include y[t-1] (and optionally y[t-2..t-p]) as regressors.
        # Lambda is estimated as the OLS coefficient on y[t-1].
        s = ar_order  # trim first ar_order rows
        y_t = y[s:]
        X_parts: list[np.ndarray] = [X_media[s:]]
        # AR lags
        for lag in range(1, ar_order + 1):
            ar_col = np.zeros(len(y))
            ar_col[lag:] = y[:-lag]
            X_parts.append(ar_col[s:].reshape(-1, 1))
        if exog_cols:
            X_parts.append(df[exog_cols].to_numpy(float)[s:])
        X_parts.append(np.ones((len(y_t), 1)))
        Z = np.hstack(X_parts)

        coefs, _, _, _ = np.linalg.lstsq(Z, y_t, rcond=None)
        self._coefs = coefs

        # Extract estimated lambda from AR(1) coefficient
        n_media = len(ch_present)
        self._lambda = float(np.clip(coefs[n_media], 0.0, 0.999))

        # Record feature names
        self._feature_names = (
            [f"beta_{ch}" for ch in ch_present]
            + [f"ydep_lag{l}" for l in range(1, ar_order + 1)]
            + exog_cols
            + ["intercept"]
        )
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

        ch_present = [ch for ch in self._channels if f"{prefix}_{ch}" in df.columns]
        n_media = len(ch_present)

        for i, ch in enumerate(self._channels):
            col = f"{prefix}_{ch}"
            if col not in df.columns:
                stc_dict[ch] = pd.Series(0.0, index=index)
                ltc_dict[ch] = pd.Series(0.0, index=index)
                continue
            x_raw = df[col].to_numpy(float)
            idx = ch_present.index(ch)
            beta0 = self._coefs[idx]
            lrm = koyck_long_run_multiplier(beta0, self._lambda)
            stc = beta0 * x_raw
            ltc = (lrm - beta0) * x_raw
            stc_dict[ch] = pd.Series(stc, index=index)
            ltc_dict[ch] = pd.Series(ltc, index=index)

        # Baseline = intercept/(1-λ) + exog effects
        alpha_true = self._coefs[-1] / (1.0 - self._lambda) if self._lambda < 1 else self._coefs[-1]
        baseline_val = np.full(T, alpha_true)
        # exog coefs follow AR lags in feature_names
        exog_start = n_media + (len(self._feature_names) - n_media - 1 - len(exog_cols))
        for j, ecol in enumerate(exog_cols):
            coef_idx = exog_start + j
            if coef_idx < len(self._coefs) - 1 and ecol in df.columns:
                baseline_val += self._coefs[coef_idx] * df[ecol].to_numpy(float)

        baseline = pd.Series(baseline_val, index=index)
        return self._make_decomposition_frame(index, self._channels, baseline, stc_dict, ltc_dict)

    def get_params(self) -> dict:
        self._check_fitted()
        return {
            "model": self.name,
            "lambda": self._lambda,
            "coefs": dict(zip(self._feature_names, self._coefs.tolist())),
        }
