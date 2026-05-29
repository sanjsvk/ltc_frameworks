"""Compute framework-level aggregates and Robustness Score from actual JSON data."""
import csv
import math

CSV_PATH = "C:/github/ltc/validation/04_CHANNEL_LEVEL_METRICS.csv"


def main():
    rows = list(csv.DictReader(open(CSV_PATH)))
    # Aggregate (total) recovery per model x scenario
    agg = {}
    for r in rows:
        if r["channel"] != "total":
            continue
        agg.setdefault(r["model"], {})[r["scenario"]] = float(r["recovery_accuracy"])

    # Compute S1-S4 averages, std, robustness score = avg / (1 + std_pp/100)
    # Actually paper formulation: Robustness Score = Mean / (1 + StdDev)
    print(f"{'Model':<16} {'S1':>6} {'S2':>6} {'S3':>6} {'S4':>6} {'S5':>6} {'Avg(1-4)':>9} {'Std(1-4)':>9} {'RobScore':>9}")
    print("-" * 95)
    framework = {"bsts": "F3", "kalman_dlm": "F3", "mcmc_stock": "F3",
                 "geo_adstock": "F1", "weibull_adstock": "F1", "almon_pdl": "F1", "dual_adstock": "F1",
                 "koyck": "F2", "ardl": "F2", "finite_dl": "F2"}
    framework_recoveries = {"F1": [], "F2": [], "F3": []}
    for model in ["bsts", "kalman_dlm", "mcmc_stock", "geo_adstock",
                  "weibull_adstock", "almon_pdl", "dual_adstock",
                  "koyck", "ardl", "finite_dl"]:
        scenarios = agg.get(model, {})
        s14 = [scenarios.get(s, 0.0) for s in ["S1", "S2", "S3", "S4"]]
        avg14 = sum(s14) / 4
        var14 = sum((x - avg14) ** 2 for x in s14) / 4
        std14 = math.sqrt(var14)
        # Robustness Score per paper: Mean Recovery / (1 + Cross-Scenario StdDev)
        rob_score = avg14 / (1 + std14 / 100) if avg14 > 0 else avg14
        # Alternative: paper might use percentage points directly
        rob_score_pp = avg14 / (1 + std14)
        print(f"{model:<16} {scenarios.get('S1', 0):>6.1f} {scenarios.get('S2', 0):>6.1f} {scenarios.get('S3', 0):>6.1f} {scenarios.get('S4', 0):>6.1f} {scenarios.get('S5', 0):>6.1f} {avg14:>9.1f} {std14:>9.2f} {rob_score_pp:>9.2f}")
        framework_recoveries[framework[model]].append(avg14)

    print()
    print("=" * 80)
    print("Framework-Level Averages (S1-S4)")
    print("=" * 80)
    for fw in ["F1", "F2", "F3"]:
        recs = framework_recoveries[fw]
        avg = sum(recs) / len(recs) if recs else 0
        print(f"  {fw}: average={avg:.1f}% (models: {recs})")

    print()
    print("=" * 80)
    print("S1 ONLY Framework Average (Section 4 paper claim: F3=75.7%, F2=32.2%, F1=30.8%)")
    print("=" * 80)
    f1_s1 = [agg[m].get("S1", 0) for m in ["geo_adstock", "weibull_adstock", "almon_pdl", "dual_adstock"]]
    f2_s1 = [agg[m].get("S1", 0) for m in ["koyck", "ardl", "finite_dl"]]
    f3_s1 = [agg[m].get("S1", 0) for m in ["bsts", "kalman_dlm", "mcmc_stock"]]
    print(f"  F1 S1 avg: {sum(f1_s1)/len(f1_s1):.2f}% (values: {f1_s1})")
    print(f"  F2 S1 avg: {sum(f2_s1)/len(f2_s1):.2f}% (values: {f2_s1})")
    print(f"  F3 S1 avg: {sum(f3_s1)/len(f3_s1):.2f}% (values: {f3_s1})")

    print()
    print("=" * 80)
    print("S1-S4 Framework Average (Section 9: F3=78.4%, F2=42.8%, F1=22.4%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
