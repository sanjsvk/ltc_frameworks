"""
experiments.run_brief — Experiment execution orchestrator for the LTC paper.

Runs all 15 EXPs (50 model-runs total) in order as specified in the
experiment execution brief.  Generates all per-experiment plots, logs
metrics to experiment_log.csv, appends findings to paper_notes.md, and
records parameters in parameter_log.csv.

EXP map
-------
EXP-01  S1 × F1   EXP-06  S2 × F3   EXP-11  S4 × F2
EXP-02  S1 × F2   EXP-07  S3 × F1   EXP-12  S4 × F3
EXP-03  S1 × F3   EXP-08  S3 × F2   EXP-13  S5 × F1
EXP-04  S2 × F1   EXP-09  S3 × F3   EXP-14  S5 × F2
EXP-05  S2 × F2   EXP-10  S4 × F1   EXP-15  S5 × F3

Each EXP runs all models within the framework, producing one log row per model.

Usage
-----
    # Run all 15 EXPs (50 model-runs)
    python experiments/run_brief.py

    # Run specific EXPs only
    python experiments/run_brief.py --exp 1 2 3

    # Run single experiment (for testing)
    python experiments/run_brief.py --exp 1

Outputs
-------
    outputs/experiment_log.csv      — one row per method per scenario
    outputs/paper_notes.md          — findings appended after each EXP
    outputs/parameter_log.csv       — fitted parameters per model-run
    outputs/results/{model}_{scenario}.json
    outputs/figures/exp{id}_{model}_{scenario}_*.png  (5 plots per run)
"""

from __future__ import annotations

import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltc.data.loader import load_scenario, split_observed_truth
from ltc.evaluation.metrics import mape, correlation, total_recovery_ratio
from experiments.registry import MODEL_REGISTRY, FRAMEWORK_GROUPS, CONFIG_MAP

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw"
CONFIG_DIR = REPO_ROOT / "experiments" / "configs"
RESULTS_DIR = REPO_ROOT / "outputs" / "results"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
LOG_DIR = REPO_ROOT / "outputs"

EXP_LOG = LOG_DIR / "experiment_log.csv"
PAPER_NOTES = LOG_DIR / "paper_notes.md"
PARAM_LOG = LOG_DIR / "parameter_log.csv"

CHANNELS = ["tv", "search", "social", "display", "video"]

# ── EXP map ──────────────────────────────────────────────────────────────────
# (exp_num, scenario, framework_key, [model_names], sub_label)
_F1_MODELS = FRAMEWORK_GROUPS["F1_static_adstock"]   # 4 models
_F2_MODELS = FRAMEWORK_GROUPS["F2_dynamic_ts"]        # 3 models
_F3_MODELS = FRAMEWORK_GROUPS["F3_state_space"]       # 3 models

EXP_DEFINITIONS: list[tuple[int, str, str, list[str]]] = [
    # (exp_num, scenario, framework_label, models)
    (1,  "S1", "F1", _F1_MODELS),
    (2,  "S1", "F2", _F2_MODELS),
    (3,  "S1", "F3", _F3_MODELS),
    (4,  "S2", "F1", _F1_MODELS),
    (5,  "S2", "F2", _F2_MODELS),
    (6,  "S2", "F3", _F3_MODELS),
    (7,  "S3", "F1", _F1_MODELS),
    (8,  "S3", "F2", _F2_MODELS),
    (9,  "S3", "F3", _F3_MODELS),
    (10, "S4", "F1", _F1_MODELS),
    (11, "S4", "F2", _F2_MODELS),
    (12, "S4", "F3", _F3_MODELS),
    (13, "S5", "F1", _F1_MODELS),
    (14, "S5", "F2", _F2_MODELS),
    (15, "S5", "F3", _F3_MODELS),
]

_SUB_LABELS = "abcdefghij"


# ── Config loading ─────────────────────────────────────────────────────────
def _load_config(model_name: str) -> dict:
    config_key = CONFIG_MAP.get(model_name, "framework1")
    config_path = CONFIG_DIR / f"{config_key}.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        all_cfg = yaml.safe_load(f)
    return all_cfg.get(model_name, {})


# ── Brief-specific metrics ─────────────────────────────────────────────────
def compute_brief_metrics(
    decomp: pd.DataFrame,
    truth_df: pd.DataFrame,
    channels: list[str] = CHANNELS,
) -> dict:
    """
    Compute the full set of brief-specified metrics.

    Returns
    -------
    dict with keys:
        ltc_mape_{ch}, ltc_total_error_{ch}, ltc_correlation_{ch}  per channel
        stc_mape_{ch}, stc_total_error_{ch}                        per channel
        baseline_mape, media_pct_error
        budget_error_{ch}                                           per channel
        ltc_mape_total, ltc_total_error_total
    """
    metrics: dict = {}

    # Per-channel LTC and STC metrics
    ltc_rec_total = np.zeros(len(decomp))
    ltc_true_total = np.zeros(len(truth_df))
    stc_rec_total = np.zeros(len(decomp))
    stc_true_total = np.zeros(len(truth_df))

    budget_true: dict[str, float] = {}
    budget_rec: dict[str, float] = {}

    for ch in channels:
        # LTC
        ltc_rec_col = f"ltc_{ch}"
        ltc_true_col = f"ltc_{ch}_true"
        if ltc_rec_col in decomp.columns and ltc_true_col in truth_df.columns:
            rec = decomp[ltc_rec_col].to_numpy(float)
            true = truth_df[ltc_true_col].to_numpy(float)
            n = min(len(rec), len(true))
            rec, true = rec[:n], true[:n]
            metrics[f"ltc_mape_{ch}"] = round(mape(rec, true), 4)
            metrics[f"ltc_correlation_{ch}"] = round(correlation(rec, true), 4)
            metrics[f"ltc_total_error_{ch}"] = round(total_recovery_ratio(rec, true) - 1.0, 4)
            ltc_rec_total[:n] += rec
            ltc_true_total[:n] += true
        else:
            metrics[f"ltc_mape_{ch}"] = float("nan")
            metrics[f"ltc_correlation_{ch}"] = float("nan")
            metrics[f"ltc_total_error_{ch}"] = float("nan")

        # STC
        stc_rec_col = f"stc_{ch}"
        stc_true_col = f"stc_{ch}_true"
        if stc_rec_col in decomp.columns and stc_true_col in truth_df.columns:
            s_rec = decomp[stc_rec_col].to_numpy(float)
            s_true = truth_df[stc_true_col].to_numpy(float)
            ns = min(len(s_rec), len(s_true))
            s_rec, s_true = s_rec[:ns], s_true[:ns]
            metrics[f"stc_mape_{ch}"] = round(mape(s_rec, s_true), 4)
            metrics[f"stc_total_error_{ch}"] = round(total_recovery_ratio(s_rec, s_true) - 1.0, 4)
            stc_rec_total[:ns] += s_rec
            stc_true_total[:ns] += s_true
            budget_true[ch] = float(s_true.sum()) + (true.sum() if f"ltc_{ch}_true" in truth_df.columns else 0.0)
            budget_rec[ch] = float(s_rec.sum()) + (rec.sum() if f"ltc_{ch}" in decomp.columns else 0.0)

    # Aggregate LTC
    n = min(np.count_nonzero(ltc_rec_total != 0), len(ltc_true_total))
    metrics["ltc_mape_total"] = round(mape(ltc_rec_total, ltc_true_total), 4)
    metrics["ltc_total_error_total"] = round(total_recovery_ratio(ltc_rec_total, ltc_true_total) - 1.0, 4)
    metrics["ltc_correlation_total"] = round(correlation(ltc_rec_total, ltc_true_total), 4)

    # Baseline MAPE
    if "baseline" in decomp.columns and "baseline_true" in truth_df.columns:
        b_rec = decomp["baseline"].to_numpy(float)
        b_true = truth_df["baseline_true"].to_numpy(float)
        n = min(len(b_rec), len(b_true))
        metrics["baseline_mape"] = round(mape(b_rec[:n], b_true[:n]), 4)
    else:
        metrics["baseline_mape"] = float("nan")

    # Media % error
    if "media_contribution_pct_true" in truth_df.columns and "fitted" in decomp.columns:
        fitted = decomp["fitted"].to_numpy(float)
        total_media_rec = (ltc_rec_total + stc_rec_total)
        rec_pct = np.where(np.abs(fitted) > 1e-6, total_media_rec / fitted, 0.0)
        true_pct = truth_df["media_contribution_pct_true"].to_numpy(float)
        n = min(len(rec_pct), len(true_pct))
        metrics["media_pct_error"] = round(float(np.mean(rec_pct[:n]) - np.mean(true_pct[:n])), 4)
    else:
        metrics["media_pct_error"] = float("nan")

    # Budget allocation error
    total_true_budget = sum(budget_true.values())
    total_rec_budget = sum(budget_rec.values())
    for ch in channels:
        true_share = budget_true.get(ch, 0.0) / total_true_budget if total_true_budget > 0 else 0.0
        rec_share = budget_rec.get(ch, 0.0) / total_rec_budget if total_rec_budget > 0 else 0.0
        metrics[f"budget_error_{ch}"] = round(rec_share - true_share, 4)

    return metrics


# ── Per-experiment plots ───────────────────────────────────────────────────
CH_COLORS = {
    "tv": "#1f77b4", "search": "#ff7f0e", "social": "#2ca02c",
    "display": "#d62728", "video": "#9467bd"
}


def _plot_ltc_time_series(
    decomp: pd.DataFrame, truth_df: pd.DataFrame, model_name: str, scenario: str
) -> plt.Figure:
    """Plot 1: Recovered vs True LTC by channel — full 5yr weekly time series."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    fig.suptitle(f"{model_name} — {scenario}: Recovered vs True LTC", fontsize=13)
    dates = pd.to_datetime(truth_df["date"]) if "date" in truth_df.columns else range(len(truth_df))
    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        col_r = f"ltc_{ch}"
        col_t = f"ltc_{ch}_true"
        color = CH_COLORS[ch]
        if col_r in decomp.columns:
            ax.plot(dates, decomp[col_r].to_numpy(float), color=color, lw=1.5, label="Recovered")
        if col_t in truth_df.columns:
            ax.plot(dates, truth_df[col_t].to_numpy(float), "k--", lw=1.2, alpha=0.8, label="True")
        ax.set_ylabel(f"{ch.upper()} ($M)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    return fig


def _plot_stc_time_series(
    decomp: pd.DataFrame, truth_df: pd.DataFrame, model_name: str, scenario: str
) -> plt.Figure:
    """Plot 2: Recovered vs True STC by channel — full 5yr weekly time series."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    fig.suptitle(f"{model_name} — {scenario}: Recovered vs True STC", fontsize=13)
    dates = pd.to_datetime(truth_df["date"]) if "date" in truth_df.columns else range(len(truth_df))
    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        color = CH_COLORS[ch]
        if f"stc_{ch}" in decomp.columns:
            ax.plot(dates, decomp[f"stc_{ch}"].to_numpy(float), color=color, lw=1.5, label="Recovered")
        if f"stc_{ch}_true" in truth_df.columns:
            ax.plot(dates, truth_df[f"stc_{ch}_true"].to_numpy(float), "k--", lw=1.2, alpha=0.8, label="True")
        ax.set_ylabel(f"{ch.upper()} ($M)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    return fig


def _plot_baseline_time_series(
    decomp: pd.DataFrame, truth_df: pd.DataFrame, model_name: str, scenario: str
) -> plt.Figure:
    """Plot 3: Recovered vs True Baseline — weekly."""
    fig, ax = plt.subplots(figsize=(14, 4))
    dates = pd.to_datetime(truth_df["date"]) if "date" in truth_df.columns else range(len(truth_df))
    if "baseline" in decomp.columns:
        ax.plot(dates, decomp["baseline"].to_numpy(float), color="#1f77b4", lw=1.5, label="Recovered baseline")
    if "baseline_true" in truth_df.columns:
        ax.plot(dates, truth_df["baseline_true"].to_numpy(float), "k--", lw=1.2, alpha=0.8, label="True baseline")
    ax.set_title(f"{model_name} — {scenario}: Recovered vs True Baseline")
    ax.set_ylabel("Baseline ($M)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _plot_ltc_error_over_time(
    decomp: pd.DataFrame, truth_df: pd.DataFrame, model_name: str, scenario: str
) -> plt.Figure:
    """Plot 4: LTC recovery error over time — (recovered - true) per channel."""
    fig, ax = plt.subplots(figsize=(14, 5))
    dates = pd.to_datetime(truth_df["date"]) if "date" in truth_df.columns else range(len(truth_df))
    has_data = False
    for ch in CHANNELS:
        col_r = f"ltc_{ch}"
        col_t = f"ltc_{ch}_true"
        if col_r in decomp.columns and col_t in truth_df.columns:
            err = decomp[col_r].to_numpy(float) - truth_df[col_t].to_numpy(float)
            ax.plot(dates, err, color=CH_COLORS[ch], lw=1.2, alpha=0.8, label=ch.upper())
            has_data = True
    ax.axhline(0, color="black", lw=1.0, linestyle="--")
    ax.set_title(f"{model_name} — {scenario}: LTC Recovery Error (Recovered − True)")
    ax.set_ylabel("Error ($M)")
    ax.set_xlabel("Date")
    if has_data:
        ax.legend(ncol=5, fontsize=8)
    ax.grid(True, alpha=0.3)
    # For S2: mark spend pause window (weeks 104-112)
    if scenario == "S2" and "date" in truth_df.columns:
        d = pd.to_datetime(truth_df["date"])
        if len(d) > 112:
            ax.axvspan(d.iloc[104], d.iloc[112], alpha=0.15, color="red", label="Spend pause")
    # For S4: mark structural break (week 104)
    if scenario == "S4" and "date" in truth_df.columns:
        d = pd.to_datetime(truth_df["date"])
        if len(d) > 104:
            ax.axvline(d.iloc[104], color="red", lw=1.5, linestyle=":", label="Mix shift")
    plt.tight_layout()
    return fig


def _plot_ltc_scatter(
    decomp: pd.DataFrame, truth_df: pd.DataFrame, model_name: str, scenario: str
) -> plt.Figure:
    """Plot 5: Scatter recovered vs true LTC per channel."""
    n_ch = len(CHANNELS)
    fig, axes = plt.subplots(1, n_ch, figsize=(14, 3.5))
    fig.suptitle(f"{model_name} — {scenario}: LTC Scatter (Recovered vs True)", fontsize=11)
    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        col_r = f"ltc_{ch}"
        col_t = f"ltc_{ch}_true"
        if col_r in decomp.columns and col_t in truth_df.columns:
            x = truth_df[col_t].to_numpy(float)
            y = decomp[col_r].to_numpy(float)
            n = min(len(x), len(y))
            ax.scatter(x[:n], y[:n], s=8, alpha=0.5, color=CH_COLORS[ch])
            lo, hi = min(x[:n].min(), y[:n].min()), max(x[:n].max(), y[:n].max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="Perfect")
        ax.set_title(ch.upper(), fontsize=9)
        ax.set_xlabel("True ($M)", fontsize=8)
        ax.set_ylabel("Recovered ($M)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def generate_five_plots(
    decomp: pd.DataFrame,
    truth_df: pd.DataFrame,
    exp_id: str,
    model_name: str,
    scenario: str,
    figures_dir: Path,
) -> list[Path]:
    """Generate and save all 5 per-experiment plots.  Returns list of saved paths."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    prefix = figures_dir / f"{exp_id}_{model_name}_{scenario}"
    saved = []

    plot_fns = [
        ("ltc_ts",      _plot_ltc_time_series),
        ("stc_ts",      _plot_stc_time_series),
        ("baseline_ts", _plot_baseline_time_series),
        ("ltc_error",   _plot_ltc_error_over_time),
        ("ltc_scatter", _plot_ltc_scatter),
    ]
    for suffix, fn in plot_fns:
        try:
            fig = fn(decomp, truth_df, model_name, scenario)
            path = Path(f"{prefix}_{suffix}.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
        except Exception as e:
            print(f"    [warn] Plot {suffix} failed: {e}")

    return saved


# ── Logging helpers ────────────────────────────────────────────────────────
_EXP_LOG_FIELDS = [
    "exp_id", "scenario", "framework", "method", "timestamp",
    "ltc_mape_tv", "ltc_mape_search", "ltc_mape_social", "ltc_mape_display", "ltc_mape_video",
    "ltc_total_error_tv", "ltc_total_error_search", "ltc_total_error_social",
    "ltc_total_error_display", "ltc_total_error_video",
    "ltc_correlation_tv", "ltc_correlation_search", "ltc_correlation_social",
    "ltc_correlation_display", "ltc_correlation_video",
    "stc_mape_tv", "stc_mape_search", "stc_mape_social", "stc_mape_display", "stc_mape_video",
    "stc_total_error_tv", "stc_total_error_search", "stc_total_error_social",
    "stc_total_error_display", "stc_total_error_video",
    "ltc_mape_total", "ltc_total_error_total", "ltc_correlation_total",
    "baseline_mape", "media_pct_error",
    "budget_error_tv", "budget_error_search", "budget_error_social",
    "budget_error_display", "budget_error_video",
    "notes",
]

_PARAM_LOG_FIELDS = [
    "exp_id", "model", "parameter_name", "value",
    "assumed_or_estimated", "rationale",
]


def _ensure_log_headers() -> None:
    """Create log files with headers if they don't exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not EXP_LOG.exists():
        with open(EXP_LOG, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_EXP_LOG_FIELDS).writeheader()
        print(f"Created {EXP_LOG}")

    if not PARAM_LOG.exists():
        with open(PARAM_LOG, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_PARAM_LOG_FIELDS).writeheader()
        print(f"Created {PARAM_LOG}")

    if not PAPER_NOTES.exists():
        PAPER_NOTES.write_text(
            "# LTC Paper — Experiment Notes\n\n"
            "Findings appended automatically after each model-run.\n"
            "Format: ## EXP-[id] — [date]\n\n"
            "---\n\n"
        )
        print(f"Created {PAPER_NOTES}")


def _append_exp_log(
    exp_id: str, scenario: str, framework: str, method: str,
    metrics: dict, notes: str = ""
) -> None:
    row = {f: metrics.get(f, float("nan")) for f in _EXP_LOG_FIELDS}
    row.update({
        "exp_id": exp_id,
        "scenario": scenario,
        "framework": framework,
        "method": method,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "notes": notes,
    })
    with open(EXP_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_EXP_LOG_FIELDS)
        writer.writerow(row)


def _append_param_log(exp_id: str, model_name: str, fitted_params: dict) -> None:
    rows = []
    # Decay-related keys to flag as key experiment parameters
    decay_keys = {"channel_decays", "decays", "channel_params"}

    for k, v in fitted_params.items():
        assumed = "assumed" if k in decay_keys else "estimated"
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                rows.append({
                    "exp_id": exp_id, "model": model_name,
                    "parameter_name": f"{k}.{sub_k}", "value": str(sub_v),
                    "assumed_or_estimated": assumed,
                    "rationale": "Selected decay parameter" if k in decay_keys else "Model fit on observed data",
                })
        elif isinstance(v, list):
            rows.append({
                "exp_id": exp_id, "model": model_name,
                "parameter_name": k, "value": str(v),
                "assumed_or_estimated": "estimated",
                "rationale": "Model fit on observed data",
            })
        else:
            rows.append({
                "exp_id": exp_id, "model": model_name,
                "parameter_name": k, "value": str(v),
                "assumed_or_estimated": "estimated",
                "rationale": "Model fit on observed data",
            })
    with open(PARAM_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PARAM_LOG_FIELDS)
        writer.writerows(rows)


def _append_paper_note(
    exp_id: str, scenario: str, framework: str, method: str,
    metrics: dict, anomalies: str = ""
) -> None:
    """Generate and append a paper note entry with metric interpretation."""
    ltc_mape_total = metrics.get("ltc_mape_total", float("nan"))
    ltc_err_total = metrics.get("ltc_total_error_total", float("nan"))
    budget_errs = {ch: metrics.get(f"budget_error_{ch}", float("nan")) for ch in CHANNELS}

    # Interpret over/under estimation
    if not np.isnan(ltc_err_total):
        direction = "over-estimates" if ltc_err_total > 0 else "under-estimates"
        pct = abs(ltc_err_total) * 100
        finding = (
            f"{method} {direction} total LTC by {pct:.1f}% on {scenario}. "
            f"Aggregate MAPE={ltc_mape_total:.1f}%."
        )
    else:
        finding = f"{method} — metrics unavailable (model may have failed)."

    # Budget errors for TV (key upper-funnel channel)
    tv_budget = budget_errs.get("tv", float("nan"))
    if not np.isnan(tv_budget):
        tv_direction = "over" if tv_budget > 0 else "under"
        finding += f" TV budget share {tv_direction}-recovered by {abs(tv_budget)*100:.1f}pp."

    # Paper narrative mapping
    paper_map = {
        "S1": "Section 6 opening — performance ceiling (ideal conditions).",
        "S2": "Section 6 headline — LTC persistence after spend pause.",
        "S3": "Section 6 — collinearity failure and practical MMM implication.",
        "S4": "Section 6 — structural break and time-varying attribution.",
        "S5": "Section 6 — false discovery and uncertainty quantification.",
    }
    paper_point = paper_map.get(scenario, "Section 6.")

    note = (
        f"\n## {exp_id} — {scenario} × {framework} × {method} — {datetime.now():%Y-%m-%d}\n"
        f"**Finding:** {finding}\n"
        f"**Paper point:** {paper_point}\n"
        f"**Key metrics:** ltc_mape_total={ltc_mape_total:.2f}%  "
        f"ltc_total_error={ltc_err_total:+.3f}  baseline_mape={metrics.get('baseline_mape', float('nan')):.2f}%\n"
        f"**Budget errors (pp):** "
        + "  ".join(f"{ch}={metrics.get(f'budget_error_{ch}', float('nan')):+.3f}" for ch in CHANNELS)
        + "\n"
        f"**Anomaly:** {anomalies if anomalies else 'None'}\n"
    )
    with open(PAPER_NOTES, "a", encoding="utf-8") as f:
        f.write(note)


# ── Core run function ──────────────────────────────────────────────────────
def run_one_model(
    exp_id: str,
    scenario: str,
    framework: str,
    model_name: str,
) -> dict:
    """
    Fit one model on one scenario, compute brief metrics, generate 5 plots, log everything.

    Returns the metrics dict (empty dict on failure).
    """
    print(f"\n  Running {exp_id} — {scenario} × {framework} × {model_name}")

    # ── Load data ──────────────────────────────────────────────────────────
    try:
        df_full = load_scenario(DATA_DIR, scenario)
    except FileNotFoundError as e:
        print(f"    [error] {e}")
        return {}

    df_obs, df_truth = split_observed_truth(df_full)

    # ── Fit model ──────────────────────────────────────────────────────────
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        print(f"    [error] Unknown model '{model_name}'")
        return {}

    config = _load_config(model_name)

    # Inject scenario-specific prior_obs_sigma for mcmc_stock
    if model_name == "mcmc_stock":
        obs_sigma_cfg = config.get("prior_obs_sigma", 0.3)
        if isinstance(obs_sigma_cfg, dict):
            config = dict(config)
            config["prior_obs_sigma"] = obs_sigma_cfg.get(scenario, 0.3)

    model = model_cls()
    anomalies = ""

    try:
        model.fit(df_obs, config)
    except Exception as e:
        msg = f"fit() failed: {e}"
        print(f"    [error] {msg}")
        traceback.print_exc()
        anomalies = msg
        return {}

    # ── Decompose ──────────────────────────────────────────────────────────
    try:
        decomp = model.decompose(df_obs)
    except Exception as e:
        msg = f"decompose() failed: {e}"
        print(f"    [error] {msg}")
        anomalies = msg
        return {}

    # ── Metrics ───────────────────────────────────────────────────────────
    try:
        metrics = compute_brief_metrics(decomp, df_truth)
    except Exception as e:
        print(f"    [warn] Metrics failed: {e}")
        metrics = {}
        anomalies += f" | metrics: {e}"

    # Print key metrics to console
    print(
        f"    ltc_mape_total={metrics.get('ltc_mape_total', float('nan')):.1f}%  "
        f"ltc_total_error={metrics.get('ltc_total_error_total', float('nan')):+.3f}  "
        f"baseline_mape={metrics.get('baseline_mape', float('nan')):.1f}%"
    )

    # ── Save JSON results ──────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{model_name}_{scenario}.json"
    try:
        fitted_params = model.get_params()
        full_results = {"model": model_name, "scenario": scenario, "metrics": metrics,
                        "fitted_params": fitted_params}
        with open(result_path, "w") as f:
            json.dump(full_results, f, indent=2, default=str)
    except Exception as e:
        print(f"    [warn] JSON save failed: {e}")
        fitted_params = {}

    # ── Generate 5 plots ───────────────────────────────────────────────────
    try:
        saved_plots = generate_five_plots(decomp, df_truth, exp_id, model_name, scenario, FIGURES_DIR)
        print(f"    Plots saved: {FIGURES_DIR}/")
    except Exception as e:
        print(f"    [warn] Plots failed: {e}")
        saved_plots = []

    # ── Log to files ───────────────────────────────────────────────────────
    _append_exp_log(exp_id, scenario, framework, model_name, metrics, anomalies)
    _append_param_log(exp_id, model_name, fitted_params)
    _append_paper_note(exp_id, scenario, framework, model_name, metrics, anomalies)

    return metrics


# ── EXP orchestrator ──────────────────────────────────────────────────────
def run_exp(exp_num: int) -> bool:
    """Run one EXP (all models within the framework × scenario).  Returns True if all pass."""
    matches = [e for e in EXP_DEFINITIONS if e[0] == exp_num]
    if not matches:
        print(f"[error] EXP-{exp_num:02d} not defined.")
        return False

    exp_num, scenario, framework, models = matches[0]
    exp_id_base = f"EXP-{exp_num:02d}"

    print(f"\n{'=' * 70}")
    print(f"Running {exp_id_base} — {scenario} × {framework} ({len(models)} model(s))")
    print(f"{'=' * 70}")
    print(f"Metrics will be saved to:  {EXP_LOG}")
    print(f"Plots will be saved to:    {FIGURES_DIR}/")
    print(f"Results will be saved to:  {RESULTS_DIR}/")
    print()

    all_passed = True
    for i, model_name in enumerate(models):
        exp_id = f"{exp_id_base}{_SUB_LABELS[i]}"
        metrics = run_one_model(exp_id, scenario, framework, model_name)
        if not metrics:
            all_passed = False

    print(f"\n  {exp_id_base} complete — results in {EXP_LOG}")
    return all_passed


# ── CLI ───────────────────────────────────────────────────────────────────
@click.command()
@click.option(
    "--exp", "-e",
    multiple=True, type=int,
    help="EXP number(s) to run (1-15). Omit to run all 15.",
)
@click.option(
    "--skip-param-est", "skip_param_est", is_flag=True, default=False,
    help="Skip parameter estimation step (use existing configs).",
)
def main(exp: tuple[int, ...], skip_param_est: bool) -> None:
    """Run LTC paper experiments per the execution brief."""
    _ensure_log_headers()

    exp_nums = sorted(exp) if exp else list(range(1, 16))

    if not skip_param_est and not (REPO_ROOT / "experiments" / "parameter_estimation" / "estimated_params.json").exists():
        print("\n[info] Running parameter estimation first (use --skip-param-est to bypass)...")
        try:
            from experiments.parameter_estimation.run_all import main as run_param_est
            run_param_est()
        except Exception as e:
            print(f"[warn] Parameter estimation failed: {e}. Continuing with defaults.")

    print(f"\nRunning EXPs: {exp_nums}")
    print(f"Total model-runs: {sum(len(e[3]) for e in EXP_DEFINITIONS if e[0] in exp_nums)}")

    for en in exp_nums:
        success = run_exp(en)
        if not success and en <= 3:
            print(f"\n[STOP] EXP-{en:02d} had failures. Fix implementation before continuing.")
            print("Re-run with --exp to resume from this point.")
            sys.exit(1)

    print(f"\n{'=' * 70}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"  Experiment log:  {EXP_LOG}")
    print(f"  Paper notes:     {PAPER_NOTES}")
    print(f"  Parameter log:   {PARAM_LOG}")
    print(f"  Results JSON:    {RESULTS_DIR}/")
    print(f"  Figures:         {FIGURES_DIR}/")
    print("Run cross-experiment summary plots next:")
    print("  python experiments/cross_exp_plots.py")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
