"""
ltc.models.base — abstract interface that every LTC estimator must implement.

Design rationale
----------------
All models — across Framework 1 (static adstock), Framework 2 (dynamic TS),
and Framework 3 (state-space) — must expose an identical three-method
interface.  This allows the experiment runner and evaluation layer to treat
every model identically, enabling fair comparison.

The three methods are:
  fit(df, config)     — learn parameters from observed data (no ground truth)
  decompose(df)       — return per-period contribution estimates
  get_params()        — return fitted parameters as a JSON-serialisable dict

Decompose output contract
-------------------------
decompose() must return a pd.DataFrame with at least these columns:
  "baseline"         — non-media baseline contribution ($M)
  "stc_{ch}"         — short-term contribution for channel ch ($M)
  "ltc_{ch}"         — long-term contribution for channel ch ($M)
  "fitted"           — fitted / predicted net sales ($M)

Optional columns:
  "ltc_{ch}_lower"   — lower bound of LTC credible/confidence interval
  "ltc_{ch}_upper"   — upper bound
  "stc_{ch}_lower", "stc_{ch}_upper"  — same for STC
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseLTCModel(ABC):
    """
    Abstract base class for all LTC estimation models.

    Subclass this and implement fit(), decompose(), and get_params().
    The experiment runner and evaluation layer only interact via this interface.
    """

    # Human-readable name used in result files, logs, and benchmark tables.
    # Override in each subclass.
    name: str = "base"

    # Framework family label — set in each subclass for grouping in benchmarks.
    # One of: "F1_static_adstock", "F2_dynamic_ts", "F3_state_space"
    framework: str = "unknown"

    def __init__(self) -> None:
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, df: pd.DataFrame, config: dict) -> "BaseLTCModel":
        """
        Fit the model on observed data.

        Parameters
        ----------
        df : pd.DataFrame
            Observed scenario DataFrame (no ground-truth columns).
            Obtain via loader.split_observed_truth()[0].
        config : dict
            Model-specific hyperparameters (e.g., decay rates, lag depth,
            prior distributions).  Loaded from experiments/configs/*.yaml.

        Returns
        -------
        self : BaseLTCModel
            Returns self to allow chaining: model.fit(df, config).decompose(df)
        """

    @abstractmethod
    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decompose observed sales into estimated contribution components.

        Must be called after fit().  May be called on the same or held-out data.

        Parameters
        ----------
        df : pd.DataFrame
            Observed scenario DataFrame (same schema as passed to fit).

        Returns
        -------
        pd.DataFrame, shape (T, ≥ 1 + 2*C)
            At minimum contains columns:
              "baseline", "stc_{ch}", "ltc_{ch}" for each channel ch,
              and "fitted" (sum of all components).
        """

    @abstractmethod
    def get_params(self) -> dict:
        """
        Return the fitted parameters as a JSON-serialisable dictionary.

        Useful for:
          - Logging and reproducibility
          - Comparing estimated vs. ground-truth parameters
          - Serialising results to outputs/results/*.json

        Returns
        -------
        dict
            Flat or nested dict of parameter names → values (Python scalars
            or lists, not numpy arrays).
        """

    # ------------------------------------------------------------------
    # Shared utilities available to all subclasses
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if decompose() is called before fit()."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__}.decompose() called before fit(). "
                "Call fit(df, config) first."
            )

    def _make_decomposition_frame(
        self,
        index: pd.Index,
        channels: list[str],
        baseline: pd.Series,
        stc: dict[str, pd.Series],
        ltc: dict[str, pd.Series],
        **extra_cols: pd.Series,
    ) -> pd.DataFrame:
        """
        Assemble a standard decomposition DataFrame from component series.

        Parameters
        ----------
        index : pd.Index
            Row index (typically integer 0…T-1).
        channels : list[str]
            Channel names (e.g., ["tv", "search", ...]).
        baseline : pd.Series (T,)
            Estimated organic baseline.
        stc : dict str → pd.Series
            Short-term contributions keyed by channel name.
        ltc : dict str → pd.Series
            Long-term contributions keyed by channel name.
        **extra_cols
            Any additional columns (e.g., "ltc_tv_lower", "ltc_tv_upper").

        Returns
        -------
        pd.DataFrame with columns: baseline, stc_{ch}…, ltc_{ch}…, fitted, + extras.
        """
        data: dict[str, pd.Series] = {"baseline": baseline}

        for ch in channels:
            data[f"stc_{ch}"] = stc.get(ch, pd.Series(0.0, index=index))
            data[f"ltc_{ch}"] = ltc.get(ch, pd.Series(0.0, index=index))

        data.update(extra_cols)

        result = pd.DataFrame(data, index=index)

        # "fitted" = sum of all estimated components
        component_cols = ["baseline"] + [f"stc_{ch}" for ch in channels] + [f"ltc_{ch}" for ch in channels]
        result["fitted"] = result[component_cols].sum(axis=1)

        return result

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}({fitted_str})"
