"""
MMM Synthetic Data Generator
=============================
Generates 5 scenario CSVs for long-term media contribution benchmarking.
Each CSV covers 2020-01-06 to 2025-12-29 (261 weekly observations).

Observed sales = baseline_true + SUM(stc_ch) + SUM(ltc_ch) + noise_true

Ground truth columns are included for validation.
Media models should only use observed columns + exogenous inputs.

Scenarios:
  S1 - Clean benchmark (geometric LTC, low collinearity, low noise)
  S2 - Latent stock persistence (spend pause mid-series, stock continues)
  S3 - High spend-seasonality collinearity (corr ~0.80)
  S4 - Structural break / media mix shift (TV drops yr3, Video+Social compensate)
  S5 - Weak + noisy LTC (low delta, high noise, small true LTC)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# 0. GLOBAL CONSTANTS
# ─────────────────────────────────────────────

SEED = 42
N_WEEKS = 261
START_DATE = pd.Timestamp("2020-01-06")  # Monday
CHANNELS = ["tv", "search", "social", "display", "video"]

# Base CPM per channel ($)
BASE_CPM = {
    "tv":      25.0,
    "search":   3.0,
    "social":  14.0,
    "display":  5.0,
    "video":   18.0,
}

# Base weekly spend ($M) — campaign-period average
BASE_SPEND = {
    "tv":      1.00,
    "search":  0.20,
    "social":  0.28,
    "display": 0.10,
    "video":   0.50,
}

# Always-on floor as fraction of base spend
ALWAYS_ON_FLOOR = {
    "tv":      0.40,
    "search":  0.80,
    "social":  0.50,
    "display": 0.85,
    "video":   0.30,
}

# Holiday spend multipliers
HOLIDAY_SPEND_MULT = {
    "tv":      2.8,
    "search":  1.6,
    "social":  2.2,
    "display": 1.4,
    "video":   2.6,
}

# Short-term adstock decay on impressions (geometric, per week)
STC_DECAY = {
    "tv":      0.55,
    "search":  0.20,
    "social":  0.45,
    "display": 0.50,
    "video":   0.60,
}

# STC coefficient: sales ($M) per 1M impressions after adstock
# Calibrated so total STC avg ~$1.58M/week (~15% of ~$10.5M sales)
STC_COEF = {
    "tv":      4.96,
    "search":  4.58,
    "social":  9.66,
    "display": 3.16,
    "video":   3.98,
}

# LTC latent stock retention rate (delta) — default
LTC_DELTA_DEFAULT = {
    "tv":      0.90,
    "search":  0.30,
    "social":  0.82,
    "display": 0.65,
    "video":   0.88,
}

# LTC coefficient: sales ($M) per unit of brand stock
# Calibrated so total LTC avg ~$1.26M/week (~12% of ~$10.5M sales)
LTC_COEF = {
    "tv":      0.0809,
    "search":  0.3949,
    "social":  0.1936,
    "display": 0.3481,
    "video":   0.1495,
}

# Stock build rate: fraction of spend ($M) that builds latent stock
STOCK_BUILD_RATE = {
    "tv":      0.60,
    "search":  0.10,
    "social":  0.35,
    "display": 0.20,
    "video":   0.55,
}


# ─────────────────────────────────────────────
# 1. DATE + CALENDAR HELPERS
# ─────────────────────────────────────────────

def make_date_index(n=N_WEEKS, start=START_DATE):
    dates = [start + pd.Timedelta(weeks=i) for i in range(n)]
    return pd.DatetimeIndex(dates)


def day_of_year(dt):
    return dt.timetuple().tm_yday


def week_in_year(w):
    """Return week index within its year (0-based)."""
    return w % 52


def build_holiday_mask(dates):
    """
    Returns array of holiday spike weights (0 = no holiday).
    Uses US calendar: New Year, Easter, Memorial Day, July 4,
    Labor Day, Thanksgiving, Black Friday, Cyber Monday, Christmas.
    Weights reflect relative demand uplift.
    """
    weights = np.zeros(len(dates))
    for i, dt in enumerate(dates):
        m, d = dt.month, dt.day
        # Christmas week
        if m == 12 and 20 <= d <= 26:
            weights[i] = 1.0
        # Black Friday / Thanksgiving week
        elif m == 11 and 21 <= d <= 29:
            weights[i] = 0.85
        # Cyber Monday week (one after BF)
        elif m == 11 and 28 <= d <= 30:
            weights[i] = max(weights[i], 0.70)
        elif m == 12 and 1 <= d <= 2:
            weights[i] = max(weights[i], 0.70)
        # New Year week
        elif m == 1 and d <= 7:
            weights[i] = 0.35
        # Memorial Day week
        elif m == 5 and 23 <= d <= 31:
            weights[i] = 0.28
        # July 4th week
        elif m == 7 and 1 <= d <= 7:
            weights[i] = 0.30
        # Labor Day week
        elif m == 9 and 1 <= d <= 7:
            weights[i] = 0.22
        # Easter (approximate: late March / April)
        elif m in (3, 4) and 20 <= d <= 30:
            weights[i] = 0.20
        # Back to school
        elif m == 8 and 15 <= d <= 31:
            weights[i] = 0.15
    return weights


def build_event_spikes(dates):
    """
    Minor spend spikes for TV/Video for events like Super Bowl,
    March Madness, summer tentpole, etc.
    Returns dict {channel: array of spike multipliers}.
    """
    tv_spikes   = np.ones(len(dates))
    vid_spikes  = np.ones(len(dates))
    for i, dt in enumerate(dates):
        m, d = dt.month, dt.day
        # Super Bowl (first two weeks of February)
        if m == 2 and d <= 14:
            tv_spikes[i]  = 3.8
            vid_spikes[i] = 3.2
        # March Madness
        elif m == 3 and d >= 15:
            tv_spikes[i]  = 1.6
            vid_spikes[i] = 1.4
        elif m == 4 and d <= 7:
            tv_spikes[i]  = 1.5
            vid_spikes[i] = 1.3
        # Summer tentpole (July 4th + Olympics years approx)
        elif m == 7 and d >= 20:
            tv_spikes[i]  = 1.3
            vid_spikes[i] = 1.2
    return {"tv": tv_spikes, "video": vid_spikes}


# ─────────────────────────────────────────────
# 2. BASELINE GENERATOR
# ─────────────────────────────────────────────

def generate_baseline(dates, rng):
    """
    Baseline = piecewise trend + annual seasonality + weekly pattern
               + holiday uplift + post-holiday dip + noise.
    No media, no promos, no exogenous shocks.
    Units: $M net sales.
    """
    n = len(dates)
    weeks = np.arange(n)

    # --- Piecewise trend ($M) ---
    trend = np.zeros(n)
    for i, w in enumerate(weeks):
        if w < 52:
            trend[i] = 10.0 + (w / 52) * 0.40
        elif w < 104:
            trend[i] = 10.40 + ((w - 52) / 52) * 0.10
        elif w < 156:
            trend[i] = 10.50 + ((w - 104) / 52) * 0.75
        elif w < 208:
            trend[i] = 11.25 + ((w - 156) / 52) * 0.35
        else:
            trend[i] = 11.60 + ((w - 208) / 53) * 0.45

    # --- Annual seasonality ---
    doys = np.array([day_of_year(dt) for dt in dates])
    annual_seas = (
        0.80 * np.sin(2 * np.pi * (doys - 30) / 365)
        + np.where(doys > 270, 0.50 * np.sin(np.pi * (doys - 270) / 95), 0)
        + np.where((doys > 150) & (doys < 220),
                   0.12 * np.sin(np.pi * (doys - 150) / 70), 0)
    )

    # --- Holiday uplift ---
    holiday_w = build_holiday_mask(dates)
    holiday_uplift = holiday_w * 2.0  # up to $2M on Christmas week

    # --- Post-holiday dip ---
    post_dip = np.zeros(n)
    for i, dt in enumerate(dates):
        m, d = dt.month, dt.day
        if (m == 1 and 7 < d <= 21) or (m == 12 and 27 <= d <= 31):
            post_dip[i] = -0.25
        if m == 2 and d <= 14:
            post_dip[i] = -0.12

    # --- Summer trough (mild) ---
    wiy = week_in_year(weeks)
    summer_dip = np.where((wiy >= 28) & (wiy <= 34), -0.15, 0.0)

    # --- Low-period softness ---
    months = np.array([dt.month for dt in dates])
    low_period = np.where((months == 2) | (months == 9), -0.12, 0.0)

    # --- Noise ---
    noise = rng.normal(0, 0.18, n)

    baseline = trend + annual_seas + holiday_uplift + post_dip + summer_dip + low_period + noise
    return np.maximum(baseline, 7.0)


# ─────────────────────────────────────────────
# 3. SPEND SERIES GENERATOR
# ─────────────────────────────────────────────

def generate_spend(dates, baseline, holiday_mask, event_spikes, rng,
                   seasonal_corr=0.40, spend_scale=None):
    """
    Generate weekly spend per channel ($M).
    seasonal_corr: how tightly spend tracks seasonal index (0=independent, 1=perfect).
    spend_scale: dict of per-channel multipliers (for scenario overrides).
    """
    n = len(dates)
    weeks = np.arange(n)
    spend = {}

    # Normalised baseline as a seasonal index (0–1)
    b_min, b_max = baseline.min(), baseline.max()
    seas_index = (baseline - b_min) / (b_max - b_min)

    event_tv  = event_spikes.get("tv",    np.ones(n))
    event_vid = event_spikes.get("video", np.ones(n))

    for ch in CHANNELS:
        base_s   = BASE_SPEND[ch]
        floor_f  = ALWAYS_ON_FLOOR[ch]
        hol_mult = HOLIDAY_SPEND_MULT[ch]
        floor    = base_s * floor_f

        # Campaign burst pattern (quarterly for TV/Video, monthly for others)
        burst = np.zeros(n)
        wiy = week_in_year(weeks)
        if ch in ("tv", "video"):
            # Quarterly flights: weeks 0-8, 13-21, 26-34, 39-47 in each year
            flight_windows = [(0, 8), (13, 21), (26, 34), (39, 47)]
            for s, e in flight_windows:
                mask = (wiy >= s) & (wiy <= e)
                burst[mask] = base_s * (1 - floor_f) * 0.8
        else:
            # More continuous with monthly peaks
            burst = base_s * (1 - floor_f) * (
                0.5 + 0.5 * np.sin(2 * np.pi * wiy / 52)
            )

        # Holiday scaling
        hol_component = holiday_mask * base_s * (hol_mult - 1)

        # Seasonal correlation component
        seas_component = seasonal_corr * base_s * 0.5 * seas_index

        # Event spikes
        event_mult = np.ones(n)
        if ch == "tv":
            event_mult = event_tv
        elif ch == "video":
            event_mult = event_vid

        raw = (floor + burst + hol_component + seas_component) * event_mult

        # Idiosyncratic noise (±8%)
        noise = rng.uniform(0.92, 1.08, n)
        raw = raw * noise

        # Apply scenario scale override
        if spend_scale and ch in spend_scale:
            raw = raw * spend_scale[ch]

        spend[ch] = np.maximum(raw, floor * 0.5)

    return spend


# ─────────────────────────────────────────────
# 4. IMPRESSIONS GENERATOR
# ─────────────────────────────────────────────

def generate_impressions(spend, baseline, dates, rng, demand_search_weight=0.08):
    """
    Impressions = (spend / CPM) * 1000  + CPM efficiency noise.
    Search impressions also get a small demand-correlated component.
    Units: millions of impressions.
    """
    n = len(dates)
    b_min, b_max = baseline.min(), baseline.max()
    demand_index = (baseline - b_min) / (b_max - b_min)

    impressions = {}
    for ch in CHANNELS:
        base_cpm = BASE_CPM[ch]
        # Time-varying CPM efficiency (±12% drift + noise)
        weeks = np.arange(n)
        cpm_drift = 1.0 + 0.06 * np.sin(2 * np.pi * weeks / 104)  # slow 2yr cycle
        cpm_noise = rng.uniform(0.90, 1.10, n)
        effective_cpm = base_cpm * cpm_drift * cpm_noise

        impr = (spend[ch] * 1e6) / effective_cpm / 1e6  # in millions

        # Search: add demand-correlated component
        if ch == "search":
            demand_component = demand_search_weight * demand_index * impr.mean()
            impr = impr + demand_component

        impressions[ch] = np.maximum(impr, 0.001)

    return impressions


# ─────────────────────────────────────────────
# 5. STC GENERATOR (adstock on impressions)
# ─────────────────────────────────────────────

def generate_stc(impressions, rng, decay_override=None):
    """
    Apply geometric adstock to impressions, then multiply by STC coefficient.
    Returns dict of STC series per channel ($M).
    """
    stc = {}
    for ch in CHANNELS:
        impr = impressions[ch]
        decay = (decay_override or {}).get(ch, STC_DECAY[ch])
        coef  = STC_COEF[ch]
        n = len(impr)

        adstocked = np.zeros(n)
        adstocked[0] = impr[0]
        for t in range(1, n):
            adstocked[t] = impr[t] + decay * adstocked[t - 1]

        stc[ch] = adstocked * coef

    return stc


# ─────────────────────────────────────────────
# 6. LTC GENERATOR (latent brand stock)
# ─────────────────────────────────────────────

def generate_ltc(spend, rng, delta_override=None, spend_pause=None,
                 coef_scale=1.0):
    """
    Latent stock model:
        stock[t] = delta * stock[t-1] + build_rate * f(spend[t])
        ltc[t]   = ltc_coef * stock[t]

    spend_pause: dict {ch: (start_week, end_week)} — zero spend during pause
                 but stock continues to depreciate (key test for S2).
    coef_scale: global LTC coefficient scalar (for S5 weak LTC).
    """
    ltc    = {}
    stocks = {}

    for ch in CHANNELS:
        s      = spend[ch].copy()
        delta  = (delta_override or {}).get(ch, LTC_DELTA_DEFAULT[ch])
        coef   = LTC_COEF[ch] * coef_scale
        build  = STOCK_BUILD_RATE[ch]
        n      = len(s)

        # Apply spend pause (spend goes to zero, stock still depreciates)
        if spend_pause and ch in spend_pause:
            ps, pe = spend_pause[ch]
            s[ps:pe] = 0.0

        # Concave build function: sqrt dampens diminishing returns
        build_input = build * np.sqrt(np.maximum(s, 0))

        stock = np.zeros(n)
        stock[0] = build_input[0] / (1 - delta + 1e-9)  # warm start
        for t in range(1, n):
            stock[t] = delta * stock[t - 1] + build_input[t]

        stocks[ch] = stock
        ltc[ch]    = stock * coef

    return ltc, stocks


# ─────────────────────────────────────────────
# 7. EXOGENOUS VARIABLES GENERATOR
# ─────────────────────────────────────────────

def generate_exogenous(dates, rng):
    """
    Generate realistic exogenous variable series.
    These influence observed net sales as non-paid media effects.

    Variables:
      promo          - binary + magnitude (0 = no promo, >0 = promo lift fraction)
      covid_index    - 0 to 1 shock index (1 = full lockdown, 0 = normal)
      dgs30          - 30-year treasury yield (%)
      mobility_index - normalised mobility (1 = normal, <1 = restricted)
      competitor_ishare - competitor impression share (0–1)
    """
    n = len(dates)
    weeks = np.arange(n)

    # --- Promo calendar ---
    # Major holiday promos + seasonal promos
    promo = np.zeros(n)
    for i, dt in enumerate(dates):
        m, d = dt.month, dt.day
        if m == 11 and 21 <= d <= 30:   # Black Friday / Thanksgiving
            promo[i] = 0.18
        elif m == 12 and 20 <= d <= 31: # Christmas
            promo[i] = 0.15
        elif m == 7 and 1 <= d <= 7:    # July 4th
            promo[i] = 0.10
        elif m == 5 and 23 <= d <= 31:  # Memorial Day
            promo[i] = 0.08
        elif m == 9 and 1 <= d <= 7:    # Labor Day
            promo[i] = 0.07
        elif m == 8 and 15 <= d <= 31:  # Back to school
            promo[i] = 0.06
        elif m == 2 and 10 <= d <= 16:  # Valentine's
            promo[i] = 0.05
        elif m == 3 and 15 <= d <= 31:  # Spring sale
            promo[i] = 0.04
        # Occasional off-peak promos (~6 random weeks per year)
    off_peak_weeks = rng.choice(np.where(promo == 0)[0],
                                size=min(30, (promo == 0).sum()), replace=False)
    promo[off_peak_weeks] = rng.uniform(0.02, 0.06, len(off_peak_weeks))

    # --- COVID index ---
    # 2020 lockdown, gradual recovery, variant waves
    covid = np.zeros(n)
    for i, dt in enumerate(dates):
        yr, m = dt.year, dt.month
        if yr == 2020:
            if m in (3, 4):
                covid[i] = 0.85   # initial lockdown
            elif m == 5:
                covid[i] = 0.65
            elif m in (6, 7):
                covid[i] = 0.40
            elif m in (8, 9):
                covid[i] = 0.30
            elif m in (10, 11, 12):
                covid[i] = 0.45   # second wave
        elif yr == 2021:
            if m in (1, 2):
                covid[i] = 0.50   # winter surge
            elif m in (3, 4, 5):
                covid[i] = 0.30
            elif m in (6, 7, 8):
                covid[i] = 0.10   # vaccine rollout
            elif m in (9, 10, 11, 12):
                covid[i] = 0.20   # delta wave
        elif yr == 2022:
            if m in (1, 2):
                covid[i] = 0.25   # omicron
            else:
                covid[i] = max(0, 0.10 - (m - 2) * 0.01)
        # 2023+ essentially zero
    covid = np.clip(covid + rng.normal(0, 0.02, n), 0, 1)

    # --- DGS30 (30-yr treasury yield %) ---
    # Historically low 2020-2021, rising 2022-2023, stabilising 2024-2025
    dgs30 = np.zeros(n)
    for i, dt in enumerate(dates):
        yr, m = dt.year, dt.month
        if yr == 2020:
            dgs30[i] = 1.20 + m * 0.02
        elif yr == 2021:
            dgs30[i] = 1.50 + m * 0.04
        elif yr == 2022:
            dgs30[i] = 2.00 + m * 0.22   # rapid rise
        elif yr == 2023:
            dgs30[i] = 4.60 + rng.normal(0, 0.15)
        elif yr == 2024:
            dgs30[i] = 4.40 + rng.normal(0, 0.12)
        else:
            dgs30[i] = 4.20 + rng.normal(0, 0.10)
    dgs30 = np.clip(dgs30 + rng.normal(0, 0.05, n), 0.5, 6.0)

    # --- Mobility index ---
    # Mirrors covid inversely with some lag
    mobility = 1.0 - (covid * 0.75)
    mobility = np.clip(mobility + rng.normal(0, 0.02, n), 0.10, 1.05)

    # --- Competitor impression share ---
    # Slow-moving, slightly inverse to own media investment periods
    comp_base = 0.35 + 0.05 * np.sin(2 * np.pi * weeks / 104)
    comp_noise = rng.normal(0, 0.02, n)
    competitor_ishare = np.clip(comp_base + comp_noise, 0.15, 0.60)

    return {
        "promo":              promo,
        "covid_index":        covid,
        "dgs30":              dgs30,
        "mobility_index":     mobility,
        "competitor_ishare":  competitor_ishare,
    }


# ─────────────────────────────────────────────
# 8. EXOGENOUS EFFECT ON SALES
# ─────────────────────────────────────────────

def exogenous_sales_effect(exog, baseline):
    """
    Map exogenous variables to a net sales effect ($M).
    These are included in observed sales but separated from media STC/LTC.

    Effects:
      promo:             +lift on sales (additive, fraction of baseline)
      covid_index:       negative shock (reduces sales)
      dgs30:             mild negative effect (higher rates = lower consumer spend)
      mobility_index:    positive when high (people going out and shopping)
      competitor_ishare: negative (competitor steals share)
    """
    effect = (
          baseline * exog["promo"] * 0.80           # promo lifts ~80% of stated rate
        - baseline * exog["covid_index"] * 0.25     # covid suppresses up to 25%
        - baseline * np.clip(exog["dgs30"] - 2.0, 0, 4) * 0.01  # rate drag above 2%
        + baseline * (exog["mobility_index"] - 1.0) * 0.08       # mobility bonus/penalty
        - baseline * exog["competitor_ishare"] * 0.06            # competitor drag
    )
    return effect


# ─────────────────────────────────────────────
# 9. ASSEMBLE DATAFRAME
# ─────────────────────────────────────────────

def assemble_df(dates, baseline, spend, impressions, stc, ltc, stocks,
                exog, exog_effect, noise_std=0.18, rng=None, scenario_tag="S1"):
    n = len(dates)
    rng = rng or np.random.default_rng(SEED)

    noise = rng.normal(0, noise_std, n)

    # Observed net sales
    stc_total  = sum(stc[ch] for ch in CHANNELS)
    ltc_total  = sum(ltc[ch] for ch in CHANNELS)
    net_sales  = baseline + stc_total + ltc_total + exog_effect + noise
    net_sales  = np.maximum(net_sales, baseline * 0.5)

    media_contribution_pct = (stc_total + ltc_total) / net_sales

    rows = {
        # Date
        "week_id":        np.arange(1, n + 1),
        "date":           [dt.strftime("%Y-%m-%d") for dt in dates],
        "year":           [dt.year for dt in dates],
        "quarter":        [dt.quarter for dt in dates],
        "week_of_year":   [dt.isocalendar()[1] for dt in dates],
        "scenario":       scenario_tag,

        # Observed
        "net_sales_observed": np.round(net_sales, 4),
    }

    # Spend + impressions
    for ch in CHANNELS:
        rows[f"spend_{ch}"]  = np.round(spend[ch], 6)
        rows[f"impr_{ch}"]   = np.round(impressions[ch], 4)

    # Exogenous inputs
    for k, v in exog.items():
        rows[k] = np.round(v, 4)

    # Ground truth
    rows["baseline_true"]             = np.round(baseline, 4)
    rows["exog_effect_true"]          = np.round(exog_effect, 4)
    rows["noise_true"]                = np.round(noise, 4)
    rows["media_contribution_pct_true"] = np.round(media_contribution_pct, 4)

    for ch in CHANNELS:
        rows[f"stc_{ch}_true"] = np.round(stc[ch], 4)
        rows[f"ltc_{ch}_true"] = np.round(ltc[ch], 4)

    for ch in ("tv", "video", "social"):
        rows[f"brand_stock_{ch}_true"] = np.round(stocks.get(ch, np.zeros(n)), 4)

    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────
# 10. SCENARIO GENERATORS
# ─────────────────────────────────────────────

def generate_scenario(scenario_id, dates, rng):
    """Dispatch to scenario-specific generation logic."""
    baseline     = generate_baseline(dates, rng)
    holiday_mask = build_holiday_mask(dates)
    event_spikes = build_event_spikes(dates)
    exog         = generate_exogenous(dates, rng)
    exog_effect  = exogenous_sales_effect(exog, baseline)

    if scenario_id == "S1":
        return _s1(dates, baseline, holiday_mask, event_spikes, exog, exog_effect, rng)
    elif scenario_id == "S2":
        return _s2(dates, baseline, holiday_mask, event_spikes, exog, exog_effect, rng)
    elif scenario_id == "S3":
        return _s3(dates, baseline, holiday_mask, event_spikes, exog, exog_effect, rng)
    elif scenario_id == "S4":
        return _s4(dates, baseline, holiday_mask, event_spikes, exog, exog_effect, rng)
    elif scenario_id == "S5":
        return _s5(dates, baseline, holiday_mask, event_spikes, exog, exog_effect, rng)
    else:
        raise ValueError(f"Unknown scenario: {scenario_id}")


def _s1(dates, baseline, holiday_mask, event_spikes, exog, exog_effect, rng):
    """S1: Clean benchmark — geometric LTC, low collinearity, low noise."""
    spend      = generate_spend(dates, baseline, holiday_mask, event_spikes,
                                rng, seasonal_corr=0.20)
    impressions = generate_impressions(spend, baseline, dates, rng)
    stc        = generate_stc(impressions, rng)
    ltc, stocks = generate_ltc(spend, rng)
    return assemble_df(dates, baseline, spend, impressions, stc, ltc, stocks,
                       exog, exog_effect, noise_std=0.15, rng=rng, scenario_tag="S1")


def _s2(dates, baseline, holiday_mask, event_spikes, exog, exog_effect, rng):
    """
    S2: Latent stock persistence.
    TV + Video spend paused weeks 104-112 (yr 3 start).
    Stock continues depreciating during pause — LTC doesn't collapse.
    """
    spend       = generate_spend(dates, baseline, holiday_mask, event_spikes,
                                 rng, seasonal_corr=0.30)
    # Apply spend pause to TV and Video
    spend["tv"][104:112]    = spend["tv"][104:112] * 0.05  # near-zero spend
    spend["video"][104:112] = spend["video"][104:112] * 0.05

    impressions = generate_impressions(spend, baseline, dates, rng)
    stc         = generate_stc(impressions, rng)
    # High delta so stock persists through pause
    delta_override = {**LTC_DELTA_DEFAULT, "tv": 0.93, "video": 0.91}
    ltc, stocks = generate_ltc(spend, rng, delta_override=delta_override,
                                spend_pause={"tv": (104, 112), "video": (104, 112)})
    return assemble_df(dates, baseline, spend, impressions, stc, ltc, stocks,
                       exog, exog_effect, noise_std=0.18, rng=rng, scenario_tag="S2")


def _s3(dates, baseline, holiday_mask, event_spikes, exog, exog_effect, rng):
    """
    S3: High spend-seasonality collinearity (~0.80).
    All channels tightly track seasonal index — makes attribution hard.
    """
    spend       = generate_spend(dates, baseline, holiday_mask, event_spikes,
                                 rng, seasonal_corr=0.80)
    impressions = generate_impressions(spend, baseline, dates, rng)
    stc         = generate_stc(impressions, rng)
    ltc, stocks = generate_ltc(spend, rng)
    return assemble_df(dates, baseline, spend, impressions, stc, ltc, stocks,
                       exog, exog_effect, noise_std=0.20, rng=rng, scenario_tag="S3")


def _s4(dates, baseline, holiday_mask, event_spikes, exog, exog_effect, rng):
    """
    S4: Structural break — TV drops 60% from week 104, Video+Social compensate.
    Total spend stays roughly constant. LTC re-routes across channels.
    """
    spend = generate_spend(dates, baseline, holiday_mask, event_spikes,
                           rng, seasonal_corr=0.35)

    # Structural break from week 104 onward
    spend["tv"][104:]     = spend["tv"][104:]     * 0.40
    spend["video"][104:]  = spend["video"][104:]  * 1.60
    spend["social"][104:] = spend["social"][104:] * 1.35

    impressions = generate_impressions(spend, baseline, dates, rng)
    stc         = generate_stc(impressions, rng)
    ltc, stocks = generate_ltc(spend, rng)
    return assemble_df(dates, baseline, spend, impressions, stc, ltc, stocks,
                       exog, exog_effect, noise_std=0.18, rng=rng, scenario_tag="S4")


def _s5(dates, baseline, holiday_mask, event_spikes, exog, exog_effect, rng):
    """
    S5: Weak + noisy LTC.
    Low delta (fast stock depreciation), reduced LTC coefficients,
    higher baseline noise. True LTC ~5-8% of sales.
    """
    spend       = generate_spend(dates, baseline, holiday_mask, event_spikes,
                                 rng, seasonal_corr=0.35,
                                 spend_scale={ch: 0.75 for ch in CHANNELS})
    impressions = generate_impressions(spend, baseline, dates, rng)
    stc         = generate_stc(impressions, rng)
    # Low delta = fast stock decay
    delta_override = {"tv": 0.72, "search": 0.20, "social": 0.68,
                      "display": 0.55, "video": 0.75}
    ltc, stocks = generate_ltc(spend, rng, delta_override=delta_override,
                                coef_scale=0.35)  # weak LTC coefficients
    return assemble_df(dates, baseline, spend, impressions, stc, ltc, stocks,
                       exog, exog_effect, noise_std=0.30, rng=rng, scenario_tag="S5")


# ─────────────────────────────────────────────
# 11. PLOTTING
# ─────────────────────────────────────────────

def plot_scenario(df, scenario_id, output_dir):
    """
    Generate a full diagnostic plot suite for one scenario.
    Saves a single multi-panel PNG per scenario.

    Panels:
      1. Observed net sales vs baseline (weekly)
      2. Sales decomposition: stacked area (baseline / STC / LTC / exog / noise)
      3. STC per channel (weekly)
      4. LTC per channel (weekly)
      5. Media contribution % of observed sales
      6. Spend per channel (weekly $M)
      7. Impressions per channel (weekly M)
      8. Brand stock — TV, Video, Social (weekly)
      9. Exogenous variables (promo, covid, mobility, competitor ishare)
     10. DGS30 yield over time
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.dates as mdates

    dates = pd.to_datetime(df["date"])

    CH_COLORS = {
        "tv":      "#3266ad",
        "search":  "#e07b39",
        "social":  "#2a9d60",
        "display": "#9b59b6",
        "video":   "#c0392b",
    }
    EXOG_COLORS = {
        "promo":             "#e07b39",
        "covid_index":       "#c0392b",
        "mobility_index":    "#2a9d60",
        "competitor_ishare": "#9b59b6",
    }

    fig = plt.figure(figsize=(20, 36))
    fig.suptitle(
        f"MMM Synthetic Data — {scenario_id}",
        fontsize=16, fontweight="bold", y=0.995
    )

    gs = fig.add_gridspec(10, 2, hspace=0.55, wspace=0.30)

    def fmt_date_axis(ax):
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        ax.tick_params(axis="x", labelsize=8)

    def y_mil(ax):
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.1f}M"))

    # ── Panel 1: Observed vs Baseline ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, df["net_sales_observed"], color="#3266ad", lw=1.2,
             label="net sales observed")
    ax1.plot(dates, df["baseline_true"], color="#888", lw=1.0, ls="--",
             label="baseline (true)")
    ax1.set_title("observed net sales vs baseline", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    y_mil(ax1)
    fmt_date_axis(ax1)

    # ── Panel 2: Stacked decomposition ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    stc_total  = sum(df[f"stc_{ch}_true"] for ch in CHANNELS)
    ltc_total  = sum(df[f"ltc_{ch}_true"] for ch in CHANNELS)
    exog_pos   = df["exog_effect_true"].clip(lower=0)
    exog_neg   = df["exog_effect_true"].clip(upper=0)

    ax2.stackplot(
        dates,
        df["baseline_true"],
        stc_total,
        ltc_total,
        exog_pos,
        labels=["baseline", "STC (all channels)", "LTC (all channels)", "exog (positive)"],
        colors=["#d0d8e8", "#3266ad", "#e07b39", "#2a9d60"],
        alpha=0.85,
    )
    ax2.fill_between(dates, exog_neg, 0, color="#c0392b", alpha=0.4,
                     label="exog (negative)")
    ax2.plot(dates, df["net_sales_observed"], color="#222", lw=0.8,
             ls=":", label="observed")
    ax2.set_title("weekly sales decomposition (stacked)", fontsize=10)
    ax2.legend(fontsize=7, loc="upper left", ncol=3)
    y_mil(ax2)
    fmt_date_axis(ax2)

    # ── Panel 3: STC per channel ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    for ch in CHANNELS:
        ax3.plot(dates, df[f"stc_{ch}_true"], color=CH_COLORS[ch],
                 lw=0.9, label=ch)
    ax3.set_title("STC by channel (true)", fontsize=10)
    ax3.legend(fontsize=7)
    y_mil(ax3)
    fmt_date_axis(ax3)

    # ── Panel 4: LTC per channel ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    for ch in CHANNELS:
        ax4.plot(dates, df[f"ltc_{ch}_true"], color=CH_COLORS[ch],
                 lw=0.9, label=ch)
    ax4.set_title("LTC by channel (true)", fontsize=10)
    ax4.legend(fontsize=7)
    y_mil(ax4)
    fmt_date_axis(ax4)

    # ── Panel 5: Media contribution % ───────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    ax5.fill_between(dates, df["media_contribution_pct_true"] * 100,
                     color="#3266ad", alpha=0.35)
    ax5.plot(dates, df["media_contribution_pct_true"] * 100,
             color="#3266ad", lw=1.0)
    ax5.axhline(df["media_contribution_pct_true"].mean() * 100,
                color="#c0392b", lw=0.8, ls="--",
                label=f"avg {df['media_contribution_pct_true'].mean()*100:.1f}%")
    ax5.set_title("media contribution % of observed net sales", fontsize=10)
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax5.legend(fontsize=8)
    fmt_date_axis(ax5)

    # ── Panel 6: Spend per channel ───────────────────────────────────────
    ax6 = fig.add_subplot(gs[4, 0])
    for ch in CHANNELS:
        ax6.plot(dates, df[f"spend_{ch}"], color=CH_COLORS[ch],
                 lw=0.85, label=ch)
    ax6.set_title("weekly spend by channel ($M)", fontsize=10)
    ax6.legend(fontsize=7)
    y_mil(ax6)
    fmt_date_axis(ax6)

    # ── Panel 7: Impressions per channel ────────────────────────────────
    ax7 = fig.add_subplot(gs[4, 1])
    for ch in CHANNELS:
        ax7.plot(dates, df[f"impr_{ch}"], color=CH_COLORS[ch],
                 lw=0.85, label=ch)
    ax7.set_title("weekly impressions by channel (M)", fontsize=10)
    ax7.legend(fontsize=7)
    ax7.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.2f}M")
    )
    fmt_date_axis(ax7)

    # ── Panel 8: Brand stocks ────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[5, :])
    stock_cols = {
        "tv":     ("brand_stock_tv_true",     CH_COLORS["tv"]),
        "video":  ("brand_stock_video_true",  CH_COLORS["video"]),
        "social": ("brand_stock_social_true", CH_COLORS["social"]),
    }
    for ch, (col, color) in stock_cols.items():
        if col in df.columns:
            ax8.plot(dates, df[col], color=color, lw=1.0, label=f"{ch} brand stock")
    ax8.set_title("latent brand stock — TV, Video, Social (true)", fontsize=10)
    ax8.legend(fontsize=8)
    ax8.set_ylabel("stock units", fontsize=8)
    fmt_date_axis(ax8)

    # ── Panel 9: Exogenous variables ─────────────────────────────────────
    ax9 = fig.add_subplot(gs[6, :])
    for col, color in EXOG_COLORS.items():
        ax9.plot(dates, df[col], color=color, lw=0.9, label=col)
    ax9.set_title("exogenous variables", fontsize=10)
    ax9.legend(fontsize=7, ncol=2)
    fmt_date_axis(ax9)

    # ── Panel 10: DGS30 ──────────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[7, :])
    ax10.plot(dates, df["dgs30"], color="#555", lw=1.0)
    ax10.fill_between(dates, df["dgs30"], alpha=0.15, color="#555")
    ax10.set_title("DGS30 — 30-year treasury yield (%)", fontsize=10)
    ax10.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.1f}%")
    )
    fmt_date_axis(ax10)

    # ── Panel 11: STC+LTC stacked per channel ───────────────────────────
    ax11 = fig.add_subplot(gs[8, :])
    stc_vals = [df[f"stc_{ch}_true"].values for ch in CHANNELS]
    ltc_vals = [df[f"ltc_{ch}_true"].values for ch in CHANNELS]
    combined = [s + l for s, l in zip(stc_vals, ltc_vals)]
    ax11.stackplot(
        dates, combined,
        labels=CHANNELS,
        colors=[CH_COLORS[ch] for ch in CHANNELS],
        alpha=0.80,
    )
    ax11.set_title("total media contribution (STC + LTC) stacked by channel", fontsize=10)
    ax11.legend(fontsize=7, loc="upper left", ncol=5)
    y_mil(ax11)
    fmt_date_axis(ax11)

    # ── Panel 12: Annual aggregation bar ────────────────────────────────
    ax12 = fig.add_subplot(gs[9, :])
    df["year"] = pd.to_datetime(df["date"]).dt.year
    annual = df.groupby("year").agg(
        net_sales=("net_sales_observed", "sum"),
        baseline=("baseline_true", "sum"),
        stc=("media_contribution_pct_true", lambda x: (
            df.loc[x.index, [f"stc_{ch}_true" for ch in CHANNELS]].sum(axis=1).sum()
        )),
        ltc=("media_contribution_pct_true", lambda x: (
            df.loc[x.index, [f"ltc_{ch}_true" for ch in CHANNELS]].sum(axis=1).sum()
        )),
    ).reset_index()

    x = np.arange(len(annual))
    w = 0.22
    ax12.bar(x - w,   annual["baseline"], w, label="baseline", color="#d0d8e8", edgecolor="#aaa", lw=0.5)
    ax12.bar(x,       annual["stc"],      w, label="STC",      color="#3266ad", edgecolor="#aaa", lw=0.5)
    ax12.bar(x + w,   annual["ltc"],      w, label="LTC",      color="#e07b39", edgecolor="#aaa", lw=0.5)
    ax12.set_xticks(x)
    ax12.set_xticklabels(annual["year"].astype(str), fontsize=9)
    ax12.set_title("annual aggregation — baseline vs STC vs LTC ($M total)", fontsize=10)
    ax12.legend(fontsize=8)
    ax12.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}M"))

    # Save
    plot_path = output_dir / f"plots_{scenario_id}.png"
    fig.savefig(plot_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  plots saved → {plot_path.name}")


# ─────────────────────────────────────────────
# 12. MAIN
# ─────────────────────────────────────────────

def main():
    output_dir = Path("mmm_synthetic_data")
    output_dir.mkdir(exist_ok=True)

    dates = make_date_index()
    scenarios = ["S1", "S2", "S3", "S4", "S5"]

    summary_rows = []

    for sid in scenarios:
        rng = np.random.default_rng(SEED)   # same seed per scenario for reproducibility
        print(f"Generating {sid}...", end=" ", flush=True)

        df = generate_scenario(sid, dates, rng)

        # Save CSV
        out_path = output_dir / f"mmm_synthetic_{sid}.csv"
        df.to_csv(out_path, index=False)

        # Summary stats
        avg_sales      = df["net_sales_observed"].mean()
        avg_baseline   = df["baseline_true"].mean()
        avg_media_pct  = df["media_contribution_pct_true"].mean()
        avg_stc        = sum(df[f"stc_{ch}_true"].mean() for ch in CHANNELS)
        avg_ltc        = sum(df[f"ltc_{ch}_true"].mean() for ch in CHANNELS)
        total_spend    = sum(df[f"spend_{ch}"].sum() for ch in CHANNELS)

        summary_rows.append({
            "scenario":              sid,
            "avg_weekly_sales_M":    round(avg_sales, 3),
            "avg_baseline_M":        round(avg_baseline, 3),
            "avg_media_pct":         round(avg_media_pct * 100, 1),
            "avg_stc_M":             round(avg_stc, 3),
            "avg_ltc_M":             round(avg_ltc, 3),
            "total_5yr_spend_M":     round(total_spend, 2),
            "rows":                  len(df),
            "columns":               len(df.columns),
        })

        print(f"done — {len(df)} rows, {len(df.columns)} cols, "
              f"avg sales ${avg_sales:.2f}M, media {avg_media_pct*100:.1f}%")

        # Generate plots
        plot_scenario(df, sid, output_dir)

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "scenario_summary.csv", index=False)

    print("\n── Scenario Summary ──────────────────────────────────────────")
    print(summary_df.to_string(index=False))
    print(f"\nAll files saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
