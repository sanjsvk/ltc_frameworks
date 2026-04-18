"""ltc.data — data loading and feature engineering for MMM scenario CSVs."""

from .loader import (
    load_scenario,
    load_all_scenarios,
    split_observed_truth,
    SCENARIOS,
    CHANNELS,
    OBSERVED_COLS,
    TRUTH_COLS,
    SPEND_COLS,
    IMPR_COLS,
)
from .features import build_feature_set, build_media_matrix, build_exog_matrix, normalise, FeatureSet

__all__ = [
    "load_scenario",
    "load_all_scenarios",
    "split_observed_truth",
    "build_feature_set",
    "build_media_matrix",
    "build_exog_matrix",
    "normalise",
    "FeatureSet",
    "SCENARIOS",
    "CHANNELS",
    "OBSERVED_COLS",
    "TRUTH_COLS",
    "SPEND_COLS",
    "IMPR_COLS",
]
