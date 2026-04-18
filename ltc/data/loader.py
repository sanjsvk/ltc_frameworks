"""
ltc.data.loader — load synthetic MMM scenario CSVs.

Expected data layout
--------------------
data/raw/
    S1.csv   — Scenario 1: Clean Benchmark
    S2.csv   — Scenario 2: Latent Stock Persistence (spend pause weeks 104-112)
    S3.csv   — Scenario 3: High Spend-Seasonality Collinearity
    S4.csv   — Scenario 4: Structural Break / Media Mix Shift (week 104)
    S5.csv   — Scenario 5: Weak and Noisy LTC

Each CSV has 261 rows (weekly, 2020-01-06 to 2025-12-29) and 39 columns.
See CLAUDE.md or docs/ for the full column schema.

Public API
----------
load_scenario(data_dir, scenario)   → pd.DataFrame (full 39-col frame)
load_all_scenarios(data_dir)        → dict[str, pd.DataFrame]
SCENARIOS                           → list of valid scenario IDs
OBSERVED_COLS                       → columns safe to pass to models
TRUTH_COLS                          → ground-truth columns (validation only)
"""

from pathlib import Path
import pandas as pd

# Valid scenario identifiers
SCENARIOS: list[str] = ["S1", "S2", "S3", "S4", "S5"]

# Columns available to models during fitting (no ground truth leakage)
OBSERVED_COLS: list[str] = [
    "date", "year", "quarter", "week_of_year",
    "net_sales_observed",
    "spend_tv", "spend_search", "spend_social", "spend_display", "spend_video",
    "impr_tv", "impr_search", "impr_social", "impr_display", "impr_video",
    "promo", "covid_index", "dgs30", "mobility_index", "competitor_ishare",
]

# Media spend and impression columns (for transform primitives)
SPEND_COLS: list[str] = [
    "spend_tv", "spend_search", "spend_social", "spend_display", "spend_video"
]
IMPR_COLS: list[str] = [
    "impr_tv", "impr_search", "impr_social", "impr_display", "impr_video"
]
CHANNELS: list[str] = ["tv", "search", "social", "display", "video"]

# Ground-truth columns — used only in evaluation, never passed to model.fit()
TRUTH_COLS: list[str] = [
    "baseline_true", "exog_effect_true", "noise_true", "media_contribution_pct_true",
    "stc_tv_true", "stc_search_true", "stc_social_true", "stc_display_true", "stc_video_true",
    "ltc_tv_true", "ltc_search_true", "ltc_social_true", "ltc_display_true", "ltc_video_true",
    "brand_stock_tv_true", "brand_stock_video_true", "brand_stock_social_true",
]


def load_scenario(
    data_dir: str | Path,
    scenario: str,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Load a single scenario CSV from data/raw/.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing scenario CSVs (e.g., "data/raw").
    scenario : str
        Scenario identifier: one of "S1", "S2", "S3", "S4", "S5".
    parse_dates : bool
        If True (default), cast the `date` column to datetime.

    Returns
    -------
    pd.DataFrame, shape (261, 39)
        Full scenario frame with all observed and ground-truth columns.

    Raises
    ------
    ValueError
        If `scenario` is not a valid identifier.
    FileNotFoundError
        If the CSV file does not exist at the expected path.
    """
    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Valid options: {SCENARIOS}"
        )

    # Support both naming conventions: S1.csv and mmm_synthetic_S1.csv
    data_dir = Path(data_dir)
    path = data_dir / f"{scenario}.csv"
    if not path.exists():
        path = data_dir / f"mmm_synthetic_{scenario}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Scenario CSV not found in {data_dir} (tried {scenario}.csv and mmm_synthetic_{scenario}.csv).\n"
            f"Run the data generation script first: data/raw/mmm_synthetic_generator.py"
        )

    df = pd.read_csv(path)
    if parse_dates and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Ensure scenario column matches the requested ID
    if "scenario" not in df.columns:
        df["scenario"] = scenario

    return df


def load_all_scenarios(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load all available scenario CSVs from data_dir.

    Skips scenarios whose CSV files do not yet exist (warns instead of raising).

    Parameters
    ----------
    data_dir : str or Path
        Directory containing scenario CSVs.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of scenario ID → DataFrame for all successfully loaded scenarios.
    """
    data_dir = Path(data_dir)
    loaded: dict[str, pd.DataFrame] = {}

    for scenario in SCENARIOS:
        path = data_dir / f"{scenario}.csv"
        if not path.exists():
            path = data_dir / f"mmm_synthetic_{scenario}.csv"
        if path.exists():
            loaded[scenario] = load_scenario(data_dir, scenario)
        else:
            print(f"[loader] Warning: {scenario} not found — skipping")

    if not loaded:
        raise FileNotFoundError(
            f"No scenario CSVs found in {data_dir}. "
            "Generate data first (see notebooks/00_data_exploration.ipynb)."
        )

    return loaded


def split_observed_truth(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a scenario DataFrame into observed (model-safe) and ground-truth halves.

    Parameters
    ----------
    df : pd.DataFrame
        Full scenario DataFrame (all 39 columns).

    Returns
    -------
    obs : pd.DataFrame
        Observed columns only — safe to pass to model.fit().
    truth : pd.DataFrame
        Ground-truth columns — for evaluation only.
    """
    obs_cols = [c for c in OBSERVED_COLS if c in df.columns]
    truth_cols = [c for c in TRUTH_COLS if c in df.columns]

    # Always carry week_id and scenario through both splits for alignment
    for meta_col in ["week_id", "scenario"]:
        if meta_col in df.columns:
            if meta_col not in obs_cols:
                obs_cols = [meta_col] + obs_cols
            if meta_col not in truth_cols:
                truth_cols = [meta_col] + truth_cols

    return df[obs_cols].copy(), df[truth_cols].copy()
