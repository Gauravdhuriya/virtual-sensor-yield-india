"""
Run the tree-model experiments. Results are cached in out/ so the script can be stopped and restarted.

  python run_all.py --quick     2-5 minute smoke test (1 seed, XGBoost, temporal hold-out) -> out_quick/
  python run_all.py             full run: XGBoost comparison, ablation, sensitivity, then Random Forest -> out/
  python run_all.py --skip-rf   everything except the slow Random Forest stage
  python run_all.py --stage main|abl|sens|rf    run one stage only
"""
import argparse, itertools, json, os, pickle, time
import adt_india as adt

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--stage", default="all", choices=["all", "main", "abl", "sens", "rf"])
ap.add_argument("--skip-rf", action="store_true", help="skip the slow Random Forest stage")
args = ap.parse_args()

OUT = "out_quick" if args.quick else "out"
os.makedirs(OUT, exist_ok=True)
REGIMES = ["level_env", "level_ymem", "anom"]
SEEDS = dict(xgb=[0] if args.quick else [0, 1, 2, 3, 4], rf=[0, 1, 2])
ABL_MODEL = "xgb"                      # learner for ablation and sensitivity ("rf" is slower)
ABL_SEEDS = [0] if args.quick else [0, 1, 2]
ABL_SCEN = ["temporal"] if args.quick else ["temporal", "multi_state"]

t_start = time.time()
s, wx, info = adt.load_all()
json.dump(info, open(f"{OUT}/info.json", "w"), indent=1)
print(json.dumps(info, indent=1))
bank = adt.make_feature_bank(wx)
_vs = adt.virtual_sensors(wx)
info["sensor_stats"] = {n: dict(mean=float(_vs[n].mean()), sd=float(_vs[n].std()), frac_at_0=float((_vs[n] <= 0).mean()),
                                frac_at_1=float((_vs[n] >= 1).mean())) for n in adt.SENSORS}
json.dump(info, open(f"{OUT}/info.json", "w"), indent=1)
del _vs


def cache(name):
    p = f"{OUT}/{name}.pkl"
    return p, (pickle.load(open(p, "rb")) if os.path.exists(p) else {})


def run(res, path, key, fn):
    if key in res:
        return
    t = time.time()
    res[key] = fn()
    pickle.dump(res, open(path, "wb"))
    S = adt.summarise(res[key]["per_seed"])
    print(f"{str(key):95s} RMSE {S['rmse'][0]:.4f}  R2 {S['r2'][0]:.3f}  [{time.time()-t:.0f}s, total {(time.time()-t_start)/60:.0f} min]", flush=True)


def run_main(model):
    path, res = cache(f"main_{model}")
    for scenario in (["temporal"] if args.quick else adt.SCENARIOS):
        for regime in REGIMES:
            for arm in (("raw", "latent", "both") if model == "xgb" else ("raw", "latent")):
                run(res, path, (model, scenario, regime, arm),
                    lambda: adt.run_config(s, bank, scenario, model, SEEDS[model], regime=regime, arm=arm, k=2))


if args.stage in ("all", "main"):
    run_main("xgb")

if args.stage in ("all", "abl"):
    path, res = cache("abl")
    configs = []
    for regime in ("level_env", "anom"):
        for r in (1, 2):
            for sub in itertools.combinations(adt.SENSORS, r):
                configs.append((regime, "sensors:" + "+".join(sub), dict(arm="latent", k=2, sensors=sub)))
        for k in (0, 1, 3):
            configs.append((regime, f"latent k={k}", dict(arm="latent", k=k)))
            configs.append((regime, f"raw k={k}", dict(arm="raw", k=k)))
        configs.append((regime, "covariates only", dict(arm="latent", k=2, climate=False)))
        for arm in ("raw", "latent"):
            configs.append((regime, f"{arm} k=2", dict(arm=arm, k=2)))
    for k in (1, 2, 3):
        configs.append(("level_ymem", f"latent+YA k={k}", dict(arm="latent", k=k)))
    configs.append(("level_ymem", "raw+YA k=2", dict(arm="raw", k=2)))
    configs.append(("level_ymem", "YA only k=2", dict(arm="latent", k=2, climate=False)))
    for scenario in ABL_SCEN:
        for regime, name, feat in configs:
            run(res, path, (scenario, regime, name),
                lambda: adt.run_config(s, bank, scenario, ABL_MODEL, ABL_SEEDS, regime=regime, **feat))

if args.stage in ("all", "sens"):
    path, res = cache("sens")
    N = adt.NOMINAL
    for name in ("alpha", "beta", "lam", "tau", "delta", "omega1"):
        for f in ((0.5, 1.5) if args.quick else (0.5, 0.8, 1.2, 1.5)):
            v = N[name] * f
            p = {name: v}
            if name == "omega1":                         # keep the weights summing to 1, omega2:omega3 fixed
                p["omega2"] = (1 - v) * N["omega2"] / (1 - N["omega1"])
            b = None
            for scenario in ABL_SCEN:
                for regime in ("level_env", "anom"):
                    key = (scenario, regime, name, f)
                    if key in res:
                        continue
                    if b is None:
                        b = adt.make_feature_bank(wx, p)
                    run(res, path, key, lambda: {"per_seed": adt.run_config(
                        s, b, scenario, ABL_MODEL, ABL_SEEDS, regime=regime, arm="latent", k=2)["per_seed"]})
    for scenario in ABL_SCEN:
        for regime in ("level_env", "anom"):
            run(res, path, (scenario, regime, "nominal", 1.0), lambda: {"per_seed": adt.run_config(
                s, bank, scenario, ABL_MODEL, ABL_SEEDS, regime=regime, arm="latent", k=2)["per_seed"]})

if args.stage in ("all", "rf") and not args.quick and not args.skip_rf:
    print("\nRandom Forest main comparison (the slowest stage; safe to stop and resume) ...")
    run_main("rf")

print(f"\nFinished in {(time.time()-t_start)/60:.0f} min. Next: python make_report.py" + (" --quick" if args.quick else ""))
