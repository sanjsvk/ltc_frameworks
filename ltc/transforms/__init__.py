"""
ltc.transforms — stateless lag and adstock transformation primitives.

All functions are pure (no side effects, no class state) and operate on
numpy arrays.  Models import individual functions as needed.

Available transforms
--------------------
geometric_adstock            geometric.py   — exponential decay adstock
weibull_adstock              weibull.py     — flexible Weibull CDF adstock
almon_compressed_regressors  almon.py       — Almon PDL regressor compression
koyck_regressors             koyck.py       — Koyck lagged-DV transformation
brand_stock_dynamics         brand_stock.py — latent brand stock simulation (DGP)
"""

from .geometric import geometric_adstock, geometric_adstock_matrix, geometric_half_life
from .weibull import weibull_adstock, weibull_cdf_weights, weibull_peak_lag
from .almon import almon_compressed_regressors, almon_pdl_weights, build_lag_matrix
from .koyck import koyck_regressors, koyck_long_run_multiplier, koyck_decompose
from .brand_stock import (
    brand_stock_dynamics,
    brand_stock_ltc,
    stock_half_life,
    stock_steady_state,
    DEFAULT_PARAMS,
)

__all__ = [
    "geometric_adstock",
    "geometric_adstock_matrix",
    "geometric_half_life",
    "weibull_adstock",
    "weibull_cdf_weights",
    "weibull_peak_lag",
    "almon_compressed_regressors",
    "almon_pdl_weights",
    "build_lag_matrix",
    "koyck_regressors",
    "koyck_long_run_multiplier",
    "koyck_decompose",
    "brand_stock_dynamics",
    "brand_stock_ltc",
    "stock_half_life",
    "stock_steady_state",
    "DEFAULT_PARAMS",
]
