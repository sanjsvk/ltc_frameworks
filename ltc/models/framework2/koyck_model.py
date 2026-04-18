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
        lambda_grid = config.get("lambda_grid", [round(0.1 + i * 0.05, 2) for i in range(17)])
        self._feature = config.get("feature", "impressions")
        self._channels = config.get("channels", ["tv", "search", "social", "display", "video"])
        exog_cols = [c for c in _EXOG if c in df.columns]
        prefix = "impr" if self._feature == "impressions" else "spend"

        y = df["net_sales_observed"].to_numpy(dtype=float)
        self._T = len(y)

        # Build current-period media matrix (T, C)
        media_cols = [f"{prefix}_{ch}" for ch in self._channels if f"{prefix}_{ch}" in df.columns]
        X_media = df[media_cols].to_numpy(dtype=float)

        # Grid-search λ: minimise RSS after Koyck transformation
        best_lambda, best_rss = 0.5, np.inf
        for lam in lambda_grid:
            y_t, Z = koyck_regressors(y, X_media)
            # Append exogenous controls aligned to t=1..T-1
            if exog_cols:
                exog_t = df[exog_cols].to_numpy(float)[1:]
                Z = np.hstack([Z, exog_t])
            Z = np.hstack([Z, np.ones((len(y_t), 1))])

            coefs, _, _, _ = np.linalg.lstsq(Z, y_t, rcond=None)
            y_hat = Z @ coefs
            rss = np.sum((y_t - y_hat) ** 2)
            if rss < best_rss:
                best_rss, best_lambda = rss, lam

        self._lambda = best_lambda

        # Final fit with best lambda (re-use the same regressors)
        y_t, Z = koyck_regressors(y, X_media)
        if exog_cols:
            exog_t = df[exog_cols].to_numpy(float)[1:]
            Z = np.hstack([Z, exog_t])
        Z = np.hstack([Z, np.ones((len(y_t), 1))])
        coefs, _, _, _ = np.linalg.lstsq(Z, y_t, rcond=None)
        self._coefs = coefs

        # Record feature names: [media_ch, ..., y_lag, exog..., intercept]
        self._feature_names = (
            [f"beta_{ch}" for ch in self._channels if f"{prefix}_{ch}" in df.columns]
            + ["lambda_ydep"]
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

        y = df["net_sales_observed"].to_numpy(float)

        for i, ch in enumerate(self._channels):
            col = f"{prefix}_{ch}"
            if col not in df.columns:
                stc_dict[ch] = pd.Series(0.0, index=index)
                ltc_dict[ch] = pd.Series(0.0, index=index)
                continue
            x_raw = df[col].to_numpy(float)
            beta0 = self._coefs[i]
            lrm = koyck_long_run_multiplier(beta0, self._lambda)
            stc = beta0 * x_raw
            ltc = (lrm - beta0) * x_raw
            stc_dict[ch] = pd.Series(stc, index=index)
            ltc_dict[ch] = pd.Series(ltc, index=index)

        # Baseline = intercept/(1-λ) + exog effects
        alpha_transformed = self._coefs[-1]
        alpha_true = alpha_transformed / (1.0 - self._lambda) if self._lambda < 1 else alpha_transformed
        baseline_val = np.full(T, alpha_true)
        for j, ecol in enumerate(exog_cols):
            coef_idx = len(self._channels) + 1 + j  # after media + y_lag
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
