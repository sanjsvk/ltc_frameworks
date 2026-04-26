"""
Framework 3 — MCMCLatentStock

This model directly parameterises the ground-truth data-generating process:

    stock[t] = δ · stock[t-1] + build_rate · √spend[t]
    LTC[t]   = ltc_coef · stock[t]
    STC[t]   = stc_coef · adstocked_impr[t]   (pre-estimated via OLS, fixed in MCMC)
    y[t]     = intercept + Σ LTC[t] + Σ STC[t] + γ·z[t] + ε[t]

Strategy: STC coefficients and exogenous effects are pre-estimated via OLS and
held fixed during MCMC to reduce the parameter space. MCMC samples the latent
stock parameters (δ, build_rate, ltc_coef) per channel plus the intercept and
observation noise, which are the parameters of interest for the paper.

This is a well-established semi-Bayes approach: nuisance parameters (STC, exog)
are estimated via OLS with n=261 data points providing strong identification;
the latent stock parameters, which are structurally harder to identify, get
full posterior uncertainty via NUTS.

PyMC model structure (per channel):
  - δ_ch       ~ Beta(α, β)           strong prior toward persistence (configurable)
  - build_rate ~ HalfNormal(σ_br)     positive, scale from empirical calibration
  - ltc_coef   ~ HalfNormal(σ_lc)     positive, scale from empirical calibration
  - intercept  ~ Normal(ȳ_net, 2.0)   weakly informative around residual mean
  - σ          ~ HalfNormal(σ_obs)    observation noise scale

Sampler: numpyro NUTS (JAX-based) to avoid C compiler dependency.

References:
    Brand stock DGP: CLAUDE.md § Ground Truth Data-Generating Process
    Semi-Bayes strategy: Gelman et al. (2013) BDA3, Ch. 4
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ltc.models.base import BaseLTCModel
from ltc.transforms.brand_stock import brand_stock_ltc, DEFAULT_PARAMS
from ltc.transforms.geometric import geometric_adstock

_EXOG = ["promo", "covid_index", "dgs30", "mobility_index", "competitor_ishare"]
_CHANNELS = ["tv", "search", "social", "display", "video"]

# Fixed STC adstock decays (short-term, from synthetic data spec)
_STC_DECAYS = {"tv": 0.55, "search": 0.20, "social": 0.45, "display": 0.50, "video": 0.60}


class MCMCLatentStock(BaseLTCModel):
    """
    MCMC estimation of the latent brand stock model (exact DGP form).

    The full MCMC path uses PyMC + NuMPyRo NUTS (JAX backend).
    Falls back to MAP (scipy optimisation) if PyMC is unavailable.

    Hyperparameters (via config dict)
    ---------------------------------
    backend : str
        "mcmc" (PyMC NUTS sampler) or "map" (scipy MAP estimate).  Default: "map".
    draws : int
        MCMC draws per chain.  Default: 1000.
    tune : int
        MCMC tuning steps.  Default: 500.
    chains : int
        Number of MCMC chains.  Default: 2.
    target_accept : float
        NUTS target acceptance rate.  Default: 0.9.
    prior_delta_alpha : float
        Beta distribution α for δ prior.  Default: 8.
    prior_delta_beta : float
        Beta distribution β for δ prior.  Default: 2.
    prior_build_rate_sigma : float
        HalfNormal σ for build_rate prior.  Default: 0.5.
    prior_ltc_coef_sigma : float
        HalfNormal σ for ltc_coef prior.  Default: 0.2.
    prior_obs_sigma : float
        HalfNormal σ for observation noise prior.  Default: 0.3.
    channels : list[str]
    feature : str
        "spend" (default for this model).
    """

    name = "mcmc_stock"
    framework = "F3_state_space"

    def __init__(self) -> None:
        super().__init__()
        self._channel_params: dict[str, dict] = {}
        self._exog_coefs: np.ndarray | None = None
        self._exog_names: list[str] = []
        self._intercept: float = 0.0
        self._channels: list[str] = []
        self._feature: str = "spend"
        self._backend: str = "map"
        self._posterior: dict | None = None
        self._stc_adstocked: dict[str, np.ndarray] = {}

    def fit(self, df: pd.DataFrame, config: dict) -> "MCMCLatentStock":
        self._backend = config.get("backend", "map")
        self._channels = config.get("channels", _CHANNELS)
        self._feature = config.get("feature", "spend")
        exog_cols = [c for c in _EXOG if c in df.columns]
        self._exog_names = exog_cols

        if self._backend == "mcmc":
            self._fit_mcmc(df, config)
        else:
            self._fit_map(df, config)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # MAP estimation (scipy — no PyMC dependency)
    # ------------------------------------------------------------------
    def _fit_map(self, df: pd.DataFrame, config: dict) -> None:
        """
        Maximum A Posteriori estimation of latent stock parameters.

        For each channel, optimises (δ, build_rate, ltc_coef) to minimise
        RSS between total model fit and observed sales.  Then fits a joint
        OLS for the STC, exog, and intercept components.
        """
        y = df["net_sales_observed"].to_numpy(float)
        T = len(y)
        exog_cols = self._exog_names

        # Step 1: fit per-channel latent stock parameters using grid init + NM
        for ch in self._channels:
            col = f"spend_{ch}"
            if col not in df.columns:
                self._channel_params[ch] = DEFAULT_PARAMS.get(
                    ch, {"delta": 0.5, "build_rate": 0.3, "ltc_coef": 0.1, "stc_coef": 5.0}
                )
                continue

            spend = df[col].to_numpy(float)
            p0 = DEFAULT_PARAMS.get(ch, {"delta": 0.5, "build_rate": 0.3, "ltc_coef": 0.1})

            def neg_corr(params: list) -> float:
                delta, br, lc = params
                if not (0.0 < delta < 1.0 and br > 0 and lc > 0):
                    return 1e9
                try:
                    _, ltc = brand_stock_ltc(spend, delta, br, lc)
                except Exception:
                    return 1e9
                corr = np.corrcoef(ltc, y)[0, 1]
                return -(corr ** 2) if not np.isnan(corr) else 1e9

            res = minimize(
                neg_corr,
                x0=[p0["delta"], p0["build_rate"], p0["ltc_coef"]],
                bounds=[(0.01, 0.995), (0.01, 5.0), (0.001, 2.0)],
                method="L-BFGS-B",
            )
            d_opt, br_opt, lc_opt = res.x
            self._channel_params[ch] = {
                "delta": float(d_opt),
                "build_rate": float(br_opt),
                "ltc_coef": float(lc_opt),
            }

        # Step 2: compute LTC contributions with fitted stock params
        ltc_matrix = np.zeros((T, len(self._channels)))
        for i, ch in enumerate(self._channels):
            if f"spend_{ch}" in df.columns:
                p = self._channel_params[ch]
                _, ltc = brand_stock_ltc(
                    df[f"spend_{ch}"].to_numpy(float), p["delta"], p["build_rate"], p["ltc_coef"]
                )
                ltc_matrix[:, i] = ltc

        # Step 3: OLS for STC + exog + intercept on y - LTC
        y_net = y - ltc_matrix.sum(axis=1)
        X_parts: list[np.ndarray] = []
        stc_names: list[str] = []
        stc_adstocked: dict[str, np.ndarray] = {}

        for ch in self._channels:
            col = f"impr_{ch}"
            if col in df.columns:
                d_stc = _STC_DECAYS.get(ch, 0.5)
                ad = geometric_adstock(df[col].to_numpy(float), d_stc)
                stc_adstocked[ch] = ad
                X_parts.append(ad.reshape(-1, 1))
                stc_names.append(ch)

        if exog_cols:
            X_parts.append(df[exog_cols].to_numpy(float))
        X_parts.append(np.ones((T, 1)))
        X = np.hstack(X_parts)
        coefs_stc, _, _, _ = np.linalg.lstsq(X, y_net, rcond=None)

        for i, ch in enumerate(stc_names):
            self._channel_params[ch]["stc_coef"] = float(coefs_stc[i])

        self._exog_coefs = coefs_stc[len(stc_names): len(stc_names) + len(exog_cols)]
        self._intercept = float(coefs_stc[-1])
        self._stc_adstocked = stc_adstocked

    # ------------------------------------------------------------------
    # MCMC estimation (PyMC + NuMPyRo NUTS)
    # ------------------------------------------------------------------
    def _fit_mcmc(self, df: pd.DataFrame, config: dict) -> None:
        """
        Full MCMC posterior estimation using PyMC + NuMPyRo (JAX) NUTS.

        Strategy
        --------
        1. Pre-estimate STC (geometric adstock on impressions) and exogenous
           effects via OLS.  These coefficients are held fixed during MCMC
           to focus sampling on the latent stock parameters.
        2. Construct y_net = y - stc_fixed - exog_fixed.
        3. MCMC samples (δ_ch, build_rate_ch, ltc_coef_ch) per channel,
           plus intercept and observation noise σ.
        4. Latent stock recurrence uses pytensor.scan for JAX compatibility.

        Priors (configurable via framework3.yaml)
        -----------------------------------------
        δ          ~ Beta(α, β)          mean = α/(α+β), calibrated from data
        build_rate ~ HalfNormal(σ_br)    σ estimated via empirical calibration
        ltc_coef   ~ HalfNormal(σ_lc)    σ estimated via empirical calibration
        intercept  ~ Normal(ȳ_net, 2.0)
        σ          ~ HalfNormal(σ_obs)
        """
        try:
            import pymc as pm
            import pytensor
            import pytensor.tensor as pt
        except ImportError:
            warnings.warn(
                "PyMC not available — falling back to MAP estimation.",
                stacklevel=2,
            )
            self._fit_map(df, config)
            return

        draws = config.get("draws", 1000)
        tune = config.get("tune", 1000)
        chains = config.get("chains", 4)  # Increased from 2 → 4 for better convergence diagnostics
        target_accept = config.get("target_accept", 0.95)

        # Prior hyperparameters (set in framework3.yaml via parameter estimation)
        delta_prior_type = config.get("delta_prior_type", "beta")
        delta_alpha = config.get("prior_delta_alpha", 8)
        delta_beta = config.get("prior_delta_beta", 2)
        delta_mu_logit: dict = config.get("delta_prior_mean_logit", {})
        delta_std_logit: float = config.get("delta_prior_std_logit", 0.30)
        build_rate_sigma = config.get("prior_build_rate_sigma", 0.5)
        ltc_coef_sigma = config.get("prior_ltc_coef_sigma", 0.2)
        obs_sigma_prior = config.get("prior_obs_sigma", 0.3)
        # prior_obs_sigma may be a dict (per-scenario) resolved by orchestrator to float
        if isinstance(obs_sigma_prior, dict):
            obs_sigma_prior = 0.3

        y = df["net_sales_observed"].to_numpy(float)
        T = len(y)
        exog_cols = self._exog_names

        # ── Step 1: pre-estimate STC + exog via OLS ──────────────────────────
        stc_adstocked: dict[str, np.ndarray] = {}
        for ch in self._channels:
            if f"impr_{ch}" in df.columns:
                stc_adstocked[ch] = geometric_adstock(
                    df[f"impr_{ch}"].to_numpy(float), _STC_DECAYS.get(ch, 0.5)
                )

        stc_names = list(stc_adstocked.keys())
        X_parts: list[np.ndarray] = [stc_adstocked[ch].reshape(-1, 1) for ch in stc_names]
        if exog_cols:
            X_parts.append(df[exog_cols].to_numpy(float))
        X_parts.append(np.ones((T, 1)))
        X_ols = np.hstack(X_parts)
        coefs_ols, _, _, _ = np.linalg.lstsq(X_ols, y, rcond=None)

        stc_fixed = np.zeros(T)
        for i, ch in enumerate(stc_names):
            stc_coef = float(max(coefs_ols[i], 0.0))
            stc_fixed += stc_coef * stc_adstocked[ch]
            self._channel_params.setdefault(ch, {})["stc_coef"] = stc_coef

        n_ch = len(stc_names)
        exog_fixed = np.zeros(T)
        if exog_cols:
            self._exog_coefs = coefs_ols[n_ch: n_ch + len(exog_cols)]
            exog_fixed = df[exog_cols].to_numpy(float) @ self._exog_coefs
        else:
            self._exog_coefs = np.array([])

        y_net = (y - stc_fixed - exog_fixed).astype("float64")

        channels_with_spend = [ch for ch in self._channels if f"spend_{ch}" in df.columns]

        # ── Step 2: build PyMC model ──────────────────────────────────────────
        with pm.Model() as _model:
            intercept = pm.Normal("intercept", mu=float(y_net.mean()), sigma=2.0)
            sigma = pm.HalfNormal("sigma", sigma=obs_sigma_prior)

            ltc_total_pt = pt.zeros(T)

            for ch in channels_with_spend:
                spend = df[f"spend_{ch}"].to_numpy(dtype="float64")
                spend_pt = pt.as_tensor_variable(spend)

                if delta_prior_type == "logit_normal" and ch in delta_mu_logit:
                    logit_delta = pm.Normal(
                        f"logit_delta_{ch}", mu=delta_mu_logit[ch], sigma=delta_std_logit
                    )
                    delta = pm.Deterministic(f"delta_{ch}", pm.math.sigmoid(logit_delta))
                else:
                    delta = pm.Beta(f"delta_{ch}", alpha=delta_alpha, beta=delta_beta)
                build_rate = pm.HalfNormal(f"build_rate_{ch}", sigma=build_rate_sigma)
                ltc_coef = pm.HalfNormal(f"ltc_coef_{ch}", sigma=ltc_coef_sigma)

                # Joint plausibility constraint: implied LTC should be < 50% of observed sales
                # Penalty if build_rate × ltc_coef product is too large (overfitting risk)
                avg_spend = float(np.mean(spend[spend > 0]) if np.any(spend > 0) else 1.0)
                avg_stock_approx = build_rate * pt.sqrt(avg_spend) / (1.0 - delta + 1e-6)
                implied_ltc_contribution = build_rate * ltc_coef * avg_stock_approx / 10.0  # scale to $M
                max_plausible_ltc = 0.5 * y_net.mean()  # max 50% of average observed signal
                pm.Potential(
                    f"plausibility_{ch}",
                    pt.switch(
                        implied_ltc_contribution > max_plausible_ltc,
                        -100.0 * (implied_ltc_contribution - max_plausible_ltc) ** 2,
                        0.0
                    )
                )

                # Latent stock recurrence via pytensor.scan (JAX-compatible)
                # Steady-state initialization: stock_ss = build_rate · √spend[0] / (1 - δ)
                stock_init = (
                    build_rate * pt.sqrt(pt.maximum(spend_pt[0], 0.0))
                    / (1.0 - delta + 1e-6)
                )
                print(f"[mcmc_stock] {ch} steady-state init: stock_init = build_rate * sqrt(spend[0]) / (1 - delta)")

                def _stock_step(sp_t, s_prev, d, br):
                    return d * s_prev + br * pt.sqrt(pt.maximum(sp_t, 0.0))

                stocks, _ = pytensor.scan(
                    fn=_stock_step,
                    sequences=[spend_pt[1:]],
                    outputs_info=[stock_init],
                    non_sequences=[delta, build_rate],
                )
                stock_series = pt.concatenate([[stock_init], stocks])
                pm.Deterministic(f"stock_{ch}", stock_series)
                ltc_series = ltc_coef * stock_series
                pm.Deterministic(f"ltc_{ch}", ltc_series)
                ltc_total_pt = ltc_total_pt + ltc_series

            mu = intercept + ltc_total_pt
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_net)

            # ── Step 3: sample ─────────────────────────────────────────────
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                progressbar=True,
                random_seed=42,
                nuts_sampler="numpyro",
            )

        # ── Step 4: extract posterior means → channel_params ─────────────────
        self._posterior = trace
        self._intercept = float(trace.posterior["intercept"].mean().item())

        # Log R-hat diagnostics to check convergence
        print("[mcmc_stock] R-hat convergence diagnostics (< 1.05 is good):")
        try:
            import arviz as az
            rhat = az.rhat(trace)
            for var_name in rhat.data_vars:
                rhat_val = float(rhat[var_name].values)
                status = "✓" if rhat_val < 1.05 else "⚠"
                print(f"  {status} {var_name}: {rhat_val:.4f}")
        except Exception as e:
            print(f"  [warn] Could not compute R-hat: {e}")

        for ch in channels_with_spend:
            self._channel_params.setdefault(ch, {}).update(
                {
                    "delta": float(trace.posterior[f"delta_{ch}"].mean().item()),
                    "build_rate": float(trace.posterior[f"build_rate_{ch}"].mean().item()),
                    "ltc_coef": float(trace.posterior[f"ltc_coef_{ch}"].mean().item()),
                }
            )

        self._stc_adstocked = stc_adstocked

    # ------------------------------------------------------------------
    # decompose / get_params
    # ------------------------------------------------------------------
    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        T = len(df)
        index = df.index
        stc_dict: dict[str, pd.Series] = {}
        ltc_dict: dict[str, pd.Series] = {}

        for ch in self._channels:
            p = self._channel_params.get(ch)
            if p is None or f"spend_{ch}" not in df.columns:
                stc_dict[ch] = pd.Series(0.0, index=index)
                ltc_dict[ch] = pd.Series(0.0, index=index)
                continue

            spend = df[f"spend_{ch}"].to_numpy(float)
            _, ltc = brand_stock_ltc(spend, p["delta"], p["build_rate"], p["ltc_coef"])
            ltc_dict[ch] = pd.Series(ltc, index=index)

            stc_coef = p.get("stc_coef", 0.0)
            if f"impr_{ch}" in df.columns:
                d_stc = _STC_DECAYS.get(ch, 0.5)
                ad = geometric_adstock(df[f"impr_{ch}"].to_numpy(float), d_stc)
                stc_dict[ch] = pd.Series(stc_coef * ad, index=index)
            else:
                stc_dict[ch] = pd.Series(0.0, index=index)

        baseline_val = np.full(T, self._intercept)
        for j, ecol in enumerate(self._exog_names):
            if ecol in df.columns and self._exog_coefs is not None and j < len(self._exog_coefs):
                baseline_val = baseline_val + self._exog_coefs[j] * df[ecol].to_numpy(float)

        baseline = pd.Series(baseline_val, index=index)
        return self._make_decomposition_frame(index, self._channels, baseline, stc_dict, ltc_dict)

    def get_params(self) -> dict:
        self._check_fitted()
        return {
            "model": self.name,
            "backend": self._backend,
            "channel_params": self._channel_params,
            "intercept": self._intercept,
            "exog_coefs": self._exog_coefs.tolist() if self._exog_coefs is not None else [],
            "exog_names": self._exog_names,
        }
