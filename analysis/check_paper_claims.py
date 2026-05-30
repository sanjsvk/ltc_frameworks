"""Validate Section 6 paper claims against extracted channel-level metrics."""
import csv

CSV_PATH = "C:/github/ltc/validation/04_CHANNEL_LEVEL_METRICS.csv"


def get_row(rows, model, scenario, channel):
    for r in rows:
        if r["model"] == model and r["scenario"] == scenario and r["channel"] == channel:
            return r
    return None


def fmt(r):
    if not r:
        return "MISSING"
    return f"rec={r['recovery_accuracy']:<7} mape={r['mape']}"


def main():
    rows = list(csv.DictReader(open(CSV_PATH)))

    print("=" * 80)
    print("SECTION 6.1: ARDL S2 (paper: agg 68.8%, all channels 0%)")
    print("=" * 80)
    for ch in ["tv", "video", "social", "display", "search", "total"]:
        r = get_row(rows, "ardl", "S2", ch)
        print(f"  {ch:<8} {fmt(r)}")

    print()
    print("=" * 80)
    print("SECTION 6.2: Koyck S2 (paper: TV 2.2, Video 14.9, Social 50.4, Display 59.3, Search 0.0, agg 43.0)")
    print("=" * 80)
    for ch in ["tv", "video", "social", "display", "search", "total"]:
        r = get_row(rows, "koyck", "S2", ch)
        print(f"  {ch:<8} {fmt(r)}")

    print()
    print("=" * 80)
    print("SECTION 6.3: Video LTC per scenario (paper claims)")
    print("=" * 80)
    print("Paper claims:")
    print("  mcmc_stock:    S3=56%, S4=46%, S5=71%")
    print("  kalman_dlm:    S3=0%,  S4=0%,  S5=0%")
    print("  bsts:          S3=0%,  S4=0%,  S5=0%")
    print("  koyck:         S3=5%,  S4=0%,  S5=0%")
    print("  ardl:          S3=0%,  S4=0%,  S5=0%")
    print("  geo_adstock:   S3=0%,  S4=5%,  S5=0%")
    print()
    for model in ["mcmc_stock", "kalman_dlm", "bsts", "koyck", "ardl", "geo_adstock"]:
        s3 = get_row(rows, model, "S3", "video")
        s4 = get_row(rows, model, "S4", "video")
        s5 = get_row(rows, model, "S5", "video")
        print(f"  {model:<14} S3 video: {fmt(s3)}")
        print(f"  {model:<14} S4 video: {fmt(s4)}")
        print(f"  {model:<14} S5 video: {fmt(s5)}")
        print()

    print("=" * 80)
    print("SECTION 6.4: MCMC channel rankings S1-S4")
    print("=" * 80)
    print("Paper claims:")
    print("  S1: TV(92) > Video(77) > Social(61) > Display(14) > Search(0)")
    print("  S2: TV(79) > Video(68) > Social(45) > Display(9)  > Search(0)")
    print("  S3: TV(92) > Social(70) > Video(56) > Display(9)  > Search(0)")
    print("  S4: TV(87) > Social(51) > Video(46) > Display(5)  > Search(0)")
    print()
    for scen in ["S1", "S2", "S3", "S4"]:
        print(f"--- MCMC {scen} ---")
        for ch in ["tv", "video", "social", "display", "search"]:
            r = get_row(rows, "mcmc_stock", scen, ch)
            print(f"  {ch:<8} {fmt(r)}")
        print()


if __name__ == "__main__":
    main()
