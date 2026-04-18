"""
experiments.registry — maps model name strings to model classes.

Add new models here to make them available to run_experiment.py.
"""

from ltc.models.framework1 import GeometricAdstockOLS, WeibullAdstockNLS, AlmonPDL, DualAdstockOLS
from ltc.models.framework2 import KoyckModel, ARDLModel, FiniteDLModel
from ltc.models.framework3 import KalmanDLM, MCMCLatentStock, BayesianStructuralTS

# Registry maps CLI name → model class
MODEL_REGISTRY: dict = {
    # Framework 1 — Static Adstock Regression
    "geo_adstock":     GeometricAdstockOLS,
    "weibull_adstock": WeibullAdstockNLS,
    "almon_pdl":       AlmonPDL,
    "dual_adstock":    DualAdstockOLS,
    # Framework 2 — Dynamic Time-Series Distributed Lag
    "koyck":           KoyckModel,
    "ardl":            ARDLModel,
    "finite_dl":       FiniteDLModel,
    # Framework 3 — State-Space / Latent Brand-Stock
    "kalman_dlm":      KalmanDLM,
    "mcmc_stock":      MCMCLatentStock,
    "bsts":            BayesianStructuralTS,
}

FRAMEWORK_GROUPS: dict[str, list[str]] = {
    "F1_static_adstock": ["geo_adstock", "weibull_adstock", "almon_pdl", "dual_adstock"],
    "F2_dynamic_ts":     ["koyck", "ardl", "finite_dl"],
    "F3_state_space":    ["kalman_dlm", "mcmc_stock", "bsts"],
}

# Config file that holds hyperparams for each model
CONFIG_MAP: dict[str, str] = {
    **{m: "framework1" for m in FRAMEWORK_GROUPS["F1_static_adstock"]},
    **{m: "framework2" for m in FRAMEWORK_GROUPS["F2_dynamic_ts"]},
    **{m: "framework3" for m in FRAMEWORK_GROUPS["F3_state_space"]},
}
