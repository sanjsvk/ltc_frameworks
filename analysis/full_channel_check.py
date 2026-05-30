"""Show channel-level rec for all key models per scenario."""
import csv

CSV_PATH = "C:/github/ltc/validation/04_CHANNEL_LEVEL_METRICS.csv"


def main():
    rows = list(csv.DictReader(open(CSV_PATH)))
    print(f"{'Model':<16} {'Scen':<4} {'TV':<8} {'Video':<8} {'Social':<8} {'Display':<8} {'Search':<8} {'Total':<8}")
    print("-" * 80)
    for model in ["bsts", "kalman_dlm", "mcmc_stock", "geo_adstock",
                  "weibull_adstock", "almon_pdl", "dual_adstock",
                  "koyck", "ardl", "finite_dl"]:
        for scen in ["S1", "S2", "S3", "S4", "S5"]:
            d = {r["channel"]: r["recovery_accuracy"] for r in rows
                 if r["model"] == model and r["scenario"] == scen}
            if not d:
                continue
            print(f"{model:<16} {scen:<4} {d.get('tv','-'):<8} {d.get('video','-'):<8} {d.get('social','-'):<8} {d.get('display','-'):<8} {d.get('search','-'):<8} {d.get('total','-'):<8}")
        print()


if __name__ == "__main__":
    main()
