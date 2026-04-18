"""
ltc.data.features — build model-ready feature matrices from observed data.

This module handles all pre-processing steps between the raw CSV and the
numpy arrays passed to model fitting:
  - Adstock-transformed impression matrices (for Framework 1/2)
  - Raw spend and impression matrices (for Framework 3)
  - Exogenous variable matrices
  - Normalisation (optional, for Bayesian models)

No ground-truth columns are ever included in outputs — that boundary is
enforced by always operating on the observed-only split from loader.py.

Public API
----------
build_media_matrix(df, feature, channels)     → np.ndarray (T, C)
build_exog_matrix(df)                         → np.ndarray (T, E)
build_feature_set(df, config)                 → FeatureSet (namedtuple)
normalise(X, method)                          → (X_norm, scaler_params)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ltc.data.loader import CHANNELS, IMPR_COLS, SPEND_COLS

# Exogenous variable columns present in the observed split
EXOG_COLS: list[str] = [
    "promo", "covid_index", "dgs30", "mobility_index", "competitor_ishare"
]

# Time feature columns useful for baseline modelling
TIME_COLS: list[str] = ["year", "quarter", "week_of_year"]


@dataclass
class FeatureSet:
    """
    Container for all model-ready arrays derived from one scenario DataFrame.

    Attributes
    ----------
    y : np.ndarray (T,)
        Observed net sales (target variable).
    spend : np.ndarray (T, C)
        Raw weekly spend per channel — used by Framework 3.
    impressions : np.ndarray (T, C)
        Raw weekly impressions per channel — used as base for adstock transforms.
    exog : np.ndarray (T, E)
        Exogenous control variables (promo, covid, rates, mobility, competitor).
    time_index : np.ndarray (T,)
        Integer week index 0…T-1.
    dates : pd.Series
        Date series aligned to the weekly index.
    channels : list[str]
        Channel names in the column order of spend / impressions matrices.
    exog_names : list[str]
        Exogenous variable names in the column order of exog matrix.
    scenario : str
        Scenario identifier loaded from the DataFrame.
    """
    y: np.ndarray
    spend: np.ndarray
    impressions: np.ndarray
    exog: np.ndarray
    time_index: np.ndarray
    dates: pd.Series
    channels: list[str]
    exog_names: list[str]
    scenario: str


def build_media_matrix(
    df: pd.DataFrame,
    feature: str = "impressions",
    channels: list[str] | None = None,
) -> np.ndarray:
    """
    Extract a (T, C) media matrix from an observed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Observed scenario DataFrame (from loader.split_observed_truth).
    feature : str
        "impressions" (default) or "spend".
    channels : list[str] or None
        Channel subset; defaults to all 5 channels.

    Returns
    -------
    np.ndarray, shape (T, C)
        Media values with channels in the order of `channels`.
    """
    if channels is None:
        channels = CHANNELS

    if feature == "impressions":
        cols = [f"impr_{ch}" for ch in channels]
    elif feature == "spend":
        cols = [f"spend_{ch}" for ch in channels]
    else:
        raise ValueError(f"feature must be 'impressions' or 'spend', got '{feature}'")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    return df[cols].to_numpy(dtype=float)


def build_exog_matrix(
    df: pd.DataFrame,
    cols: list[str] | None = None,
) -> np.ndarray:
    """
    Extract the exogenous control variable matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Observed scenario DataFrame.
    cols : list[str] or None
        Exogenous columns to include; defaults to EXOG_COLS.

    Returns
    -------
    np.ndarray, shape (T, E)
    """
    if cols is None:
        cols = [c for c in EXOG_COLS if c in df.columns]
    return df[cols].to_numpy(dtype=float)


def build_feature_set(
    df: pd.DataFrame,
    channels: list[str] | None = None,
    exog_cols: list[str] | None = None,
) -> FeatureSet:
    """
    Build a complete FeatureSet from an observed scenario DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Observed scenario DataFrame.
    channels : list[str] or None
        Channel subset (default: all 5).
    exog_cols : list[str] or None
        Exogenous columns to include (default: EXOG_COLS).

    Returns
    -------
    FeatureSet
    """
    if channels is None:
        channels = CHANNELS
    if exog_cols is None:
        exog_cols = [c for c in EXOG_COLS if c in df.columns]

    y = df["net_sales_observed"].to_numpy(dtype=float)
    spend = build_media_matrix(df, feature="spend", channels=channels)
    impressions = build_media_matrix(df, feature="impressions", channels=channels)
    exog = build_exog_matrix(df, cols=exog_cols)
    time_index = np.arange(len(df), dtype=int)
    dates = df["date"].reset_index(drop=True) if "date" in df.columns else pd.Series(time_index)
    scenario = df["scenario"].iloc[0] if "scenario" in df.columns else "unknown"

    return FeatureSet(
        y=y,
        spend=spend,
        impressions=impressions,
        exog=exog,
        time_index=time_index,
        dates=dates,
        channels=channels,
        exog_names=exog_cols,
        scenario=scenario,
    )


def normalise(
    X: np.ndarray,
    method: str = "zscore",
    params: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Normalise a feature matrix and return scaler parameters for inversion.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix to normalise.
    method : str
        "zscore" (subtract mean, divide by std) or "minmax" (scale to [0,1]).
    params : dict or None
        Pre-fitted scaler params (for applying to test data).  If None,
        fitted from X.

    Returns
    -------
    X_norm : np.ndarray
        Normalised matrix.
    params : dict
        Scaler parameters ({"mean", "std"} or {"min", "range"}).
    """
    if method == "zscore":
        if params is None:
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            std = np.where(std == 0, 1.0, std)  # avoid division by zero
            params = {"mean": mean, "std": std}
        X_norm = (X - params["mean"]) / params["std"]

    elif method == "minmax":
        if params is None:
            xmin = X.min(axis=0)
            xrange = X.max(axis=0) - xmin
            xrange = np.where(xrange == 0, 1.0, xrange)
            params = {"min": xmin, "range": xrange}
        X_norm = (X - params["min"]) / params["range"]

    else:
        raise ValueError(f"method must be 'zscore' or 'minmax', got '{method}'")

    return X_norm, params
