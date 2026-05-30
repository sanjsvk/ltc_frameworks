"""
Extract channel-level LTC recovery and MAPE metrics for all 10 models x 5 scenarios.
Used for Phase 3 validation of Section 6 (Channel-Level Attribution).
"""

import json
import os
import csv
from pathlib import Path

RESULTS_DIR = Path("C:/github/ltc/outputs/results")
OUTPUT_CSV = Path("C:/github/ltc/validation/04_CHANNEL_LEVEL_METRICS.csv")

MODELS = [
    "bsts", "kalman_dlm", "mcmc_stock",
    "geo_adstock", "weibull_adstock", "almon_pdl", "dual_adstock",
    "koyck", "ardl", "finite_dl",
]
SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]
CHANNELS = ["tv", "search", "social", "display", "video", "total"]

FRAMEWORK = {
    "bsts": "F3", "kalman_dlm": "F3", "mcmc_stock": "F3",
    "geo_adstock": "F1", "weibull_adstock": "F1", "almon_pdl": "F1", "dual_adstock": "F1",
    "koyck": "F2", "ardl": "F2", "finite_dl": "F2",
}


def main():
    rows = []
    missing = []
    for model in MODELS:
        for scen in SCENARIOS:
            fp = RESULTS_DIR / f"{model}_{scen}.json"
            if not fp.exists():
                missing.append(str(fp))
                continue
            try:
                with open(fp, "r") as f:
                    raw = f.read()
                # JSON cannot contain NaN; replace with null for parsing
                raw = raw.replace("NaN", "null").replace("Infinity", "null").replace("-null", "null")
                data = json.loads(raw)
            except Exception as e:
                missing.append(f"{fp}: parse error: {e}")
                continue
            ltc = data.get("ltc", {})
            for ch in CHANNELS:
                ch_data = ltc.get(ch, {})
                rows.append({
                    "model": model,
                    "framework": FRAMEWORK[model],
                    "scenario": scen,
                    "channel": ch,
                    "recovery_accuracy": round(ch_data.get("recovery_accuracy", float("nan")), 2) if ch_data else None,
                    "mape": round(ch_data.get("mape", float("nan")), 2) if ch_data else None,
                    "correlation": round(ch_data.get("correlation") or 0.0, 3) if ch_data else None,
                    "bias": round(ch_data.get("bias", float("nan")), 4) if ch_data else None,
                    "total_recovery_ratio": round(ch_data.get("total_recovery_ratio", float("nan")), 3) if ch_data else None,
                })

    # write csv
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "framework", "scenario", "channel",
            "recovery_accuracy", "mape", "correlation", "bias", "total_recovery_ratio",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
    if missing:
        print("MISSING/ERROR FILES:")
        for m in missing:
            print("  ", m)


if __name__ == "__main__":
    main()
