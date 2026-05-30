"""Verify S5 claims in paper - all models 0% recovery with frozen, MCMC 88.5% with priors."""
import json

models = ["bsts", "kalman_dlm", "mcmc_stock", "geo_adstock",
          "weibull_adstock", "almon_pdl", "dual_adstock",
          "koyck", "ardl", "finite_dl"]

print(f"{'Model':<16} {'recovery':>10} {'mape':>10} {'total_recov_ratio':>20} {'100-mape':>10}")
print("-" * 80)
for m in models:
    fp = f"C:/github/ltc/outputs/results/{m}_S5.json"
    raw = open(fp).read().replace("NaN", "null").replace("Infinity", "null").replace("-null", "null")
    data = json.loads(raw)
    total = data["ltc"]["total"]
    rec = total.get("recovery_accuracy")
    mp = total.get("mape")
    trr = total.get("total_recovery_ratio")
    not_capped = 100 - mp if mp is not None else None
    print(f"{m:<16} {rec:>10.2f} {mp:>10.2f} {trr:>20.4f} {not_capped:>10.2f}")
