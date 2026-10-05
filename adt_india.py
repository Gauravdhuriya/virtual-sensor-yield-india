"""
Core pipeline for the Indian district experiments (ICRISAT yields + NASA POWER daily weather + SoilGrids pH).

  load_all()          -> samples table, daily weather, soil pH
  virtual_sensors()   -> daily VSSM / VSST / VSCS for every district (Eq. 4-7 of the paper, as written)
  season_features()   -> growing-season summaries of daily series (three sub-periods per season)
  build_features()    -> raw / latent / raw+latent feature matrices with temporal memory
  splits()            -> temporal, cross-district, multi-state, multi-crop and combined hold-outs
"""
import glob, os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- configuration
ICRISAT_CSV = "ICRISAT-District Level Data.csv"     # path to the ICRISAT file (edit if needed)
WEATHER_DIR = "india_data"                           # folder written by fetch_india_data.py
N_JOBS = -1                                          # CPU cores for RF / XGBoost (-1 = all)

CROPS = {  # name: (ICRISAT column prefix, season)
    "Rice": ("RICE", "kharif"), "Maize": ("MAIZE", "kharif"), "Kharif sorghum": ("KHARIF SORGHUM", "kharif"),
    "Pearl millet": ("PEARL MILLET", "kharif"), "Wheat": ("WHEAT", "rabi"), "Chickpea": ("CHICKPEA", "rabi"),
}
SEASON = {"kharif": ((6, 1), (10, 31), 0), "rabi": ((11, 1), (3, 31), 1)}   # start, end, year offset of end
Y0 = 1984                 # first weather year
MAXLAG = 3                # every sample needs 3 prior years of yield, so all variants share the same samples
MIN_AREA = 1.0            # minimum crop area (1000 ha) for a district-year to be used
TRAIN_END = 2007          # temporal hold-out: train <= 2007, test >= 2008

# Nominal virtual-sensor parameters. None of them is tuned on yield.
NOMINAL = dict(
    alpha=0.8 / 150.0,    # infiltration fraction 0.8 / root-zone storage 150 mm   -> per mm of rain
    beta=10.0 / 150.0,    # maximum evaporative demand 10 mm/day / storage 150 mm  -> per unit of ET proxy
    lam=0.7,              # weight of temperature vs wind in the ET proxy (Eq. 5)
    tau=5.0,              # soil thermal time constant in days; gamma = exp(-1/tau) (Eq. 6)
    delta=0.1,            # radiative heating, deg C per MJ m-2 day-1 (Eq. 6)
    omega1=0.5, omega2=0.3,   # water and thermal stress weights; omega3 = 1 - omega1 - omega2 (Eq. 7)
    ph_opt=6.5, vssm0=0.5, t_ref=45.0, w_ref=10.0,
)
RAW_VARS = ["rain", "temp", "wind", "solar", "rh"]
SENSORS = ("VSSM", "VSST", "VSCS")


# ----------------------------------------------------------------------------- data
def load_weather():
    files = sorted(glob.glob(os.path.join(WEATHER_DIR, "india_weather_part*.npz")))
    if not files:
        raise FileNotFoundError(f"No india_weather_part*.npz in {WEATHER_DIR}/ - run fetch_india_data.py first")
    data, codes = [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        d = z["data"].astype(np.float32)
        d[z["data"] == int(z["missing"])] = np.nan
        data.append(d / z["scale"].astype(np.float32)); codes.append(z["codes"]); dates = z["dates"]
    W = np.concatenate(data); codes = np.concatenate(codes)
    dates = pd.to_datetime(pd.Series(dates.astype(str)), format="%Y%m%d")
    n_missing = int(np.isnan(W).sum())
    doy = dates.dt.dayofyear.to_numpy()
    for v in range(W.shape[2]):                      # fill gaps with the district's day-of-year mean
        x = W[:, :, v]
        if np.isnan(x).any():
            clim = np.stack([np.nanmean(x[:, doy == k], axis=1) if (doy == k).any() else np.full(len(x), np.nan)
                             for k in range(1, 367)], axis=1)
            fill = clim[:, doy - 1]
            x[np.isnan(x)] = fill[np.isnan(x)]
            x[np.isnan(x)] = np.nanmean(x)
    tab = pd.read_csv(os.path.join(WEATHER_DIR, "india_districts.csv")).set_index("dist_code").loc[codes]
    ph = tab["soil_ph"].astype(float)
    n_ph = int(ph.notna().sum())
    ph = ph.fillna(ph.groupby(tab["state"]).transform("median")).fillna(ph.median()).fillna(NOMINAL["ph_opt"])
    return dict(W=W, codes=codes, dates=dates, ph=ph.to_numpy(np.float32), lat=tab["lat"].to_numpy(),
                n_missing=n_missing, n_ph=n_ph)


def load_all():
    wx = load_weather()
    path = ICRISAT_CSV
    if not os.path.exists(path):                     # accept any ICRISAT*.csv in the folder
        found = sorted(glob.glob("ICRISAT*.csv"))
        if not found:
            raise FileNotFoundError("Put the ICRISAT district-level CSV in this folder (file name starting with ICRISAT)")
        path = found[0]
    df = pd.read_csv(path)
    didx = {c: i for i, c in enumerate(wx["codes"])}
    last_year = int(wx["dates"].dt.year.max())
    rows, dropped = [], 0
    for crop, (pref, season) in CROPS.items():
        y = df[["Dist Code", "Year", "State Name", "Dist Name"]].copy()
        y["yk"] = df[f"{pref} YIELD (Kg per ha)"]; y["area"] = df[f"{pref} AREA (1000 ha)"]
        y = y[(y["yk"] > 0) & (y["area"] >= MIN_AREA) & y["Dist Code"].isin(didx)]
        ok = y["yk"] <= 5 * y["yk"].median()         # implausible yields (production / tiny area artefacts)
        dropped += int((~ok).sum()); y = y[ok]
        y["crop"], y["season"], y["y"] = crop, season, y["yk"] / 1000.0
        rows.append(y)
    y = pd.concat(rows, ignore_index=True)
    key = y.set_index(["Dist Code", "crop", "Year"])["y"]
    for L in range(1, MAXLAG + 1):
        y[f"y_l{L}"] = key.reindex(pd.MultiIndex.from_arrays([y["Dist Code"], y["crop"], y["Year"] - L])).to_numpy()
    s = y.dropna(subset=[f"y_l{L}" for L in range(1, MAXLAG + 1)])
    end_ok = s["Year"] + s["season"].map({k: v[2] for k, v in SEASON.items()}) <= last_year
    s = s[(s["Year"] - MAXLAG >= Y0) & end_ok].reset_index(drop=True)
    s = s.rename(columns={"Dist Code": "dist", "State Name": "state", "Dist Name": "district", "Year": "year"})
    s["d_idx"] = s["dist"].map(didx).astype(int)
    s["crop_code"] = pd.Categorical(s["crop"], categories=list(CROPS)).codes
    s["is_rabi"] = (s["season"] == "rabi").astype(int)
    info = dict(n_samples=len(s), n_districts=int(s["dist"].nunique()), n_states=int(s["state"].nunique()),
                n_series=int(s.groupby(["dist", "crop"]).ngroups), years=(int(s["year"].min()), int(s["year"].max())),
                per_crop=s["crop"].value_counts().to_dict(), dropped_outliers=dropped,
                weather_districts=len(wx["codes"]), weather_missing_values=wx["n_missing"],
                districts_with_ph=wx["n_ph"], weather_days=len(wx["dates"]))
    return s, wx, info


# ----------------------------------------------------------------------------- virtual sensors
def virtual_sensors(wx, p=None):
    """Daily recursions for all districts at once (Eq. 4-7).

    ET(t)   = lam * T_n(t) + (1 - lam) * W_n(t)                       T_n = T/45 C, W_n = wind/10 m/s, clipped to [0, 1]
    VSSM(t) = clip(VSSM(t-1) + alpha * R(t) - beta * ET(t), 0, 1)       R in mm/day
    VSST(t) = gamma * VSST(t-1) + (1 - gamma) * (T(t) + delta * S(t))   deg C, gamma = exp(-1/tau)
    VSCS(t) = omega1 * (1 - VSSM) + omega2 * VSST/45 + omega3 * |pH - pH_opt| / pH_opt
    """
    q = dict(NOMINAL)
    if p:
        q.update(p)
    W = wx["W"]
    rain, temp, wind, solar = W[:, :, 0], W[:, :, 1], W[:, :, 2], W[:, :, 3]
    lam = min(max(q["lam"], 0.0), 1.0)
    et = lam * np.clip(temp / q["t_ref"], 0, 1) + (1 - lam) * np.clip(wind / q["w_ref"], 0, 1)
    gamma = float(np.exp(-1.0 / q["tau"]))
    D, T = rain.shape
    vssm, vsst = np.empty((D, T), np.float32), np.empty((D, T), np.float32)
    m = np.full(D, q["vssm0"], np.float32); t = temp[:, 0] + q["delta"] * solar[:, 0]
    for i in range(T):
        m = np.clip(m + q["alpha"] * rain[:, i] - q["beta"] * et[:, i], 0.0, 1.0)
        t = gamma * t + (1 - gamma) * (temp[:, i] + q["delta"] * solar[:, i])
        vssm[:, i], vsst[:, i] = m, t
    o1, o2 = q["omega1"], q["omega2"]
    o3 = max(1.0 - o1 - o2, 0.0)
    ph_term = np.abs(wx["ph"] - q["ph_opt"]) / q["ph_opt"]
    vscs = o1 * (1 - vssm) + o2 * vsst / q["t_ref"] + o3 * ph_term[:, None]
    return dict(VSSM=vssm, VSST=vsst, VSCS=vscs.astype(np.float32))


def season_bounds(dates):
    """Day-index bounds of the three sub-periods of each season and year: dict season -> (starts, ends) [n_years, 3]."""
    d = pd.DatetimeIndex(dates)
    years = np.arange(Y0, d.year.max() + 1)
    out = {}
    for name, ((m0, d0), (m1, d1), off) in SEASON.items():
        st, en = np.zeros((len(years), 3), int), np.ones((len(years), 3), int)
        for j, yr in enumerate(years):
            a, b = pd.Timestamp(yr, m0, d0), pd.Timestamp(yr + off, m1, d1)
            if b > d[-1]:
                continue
            ia, ib = d.get_loc(a), d.get_loc(b) + 1
            cuts = np.linspace(ia, ib, 4).round().astype(int)
            st[j], en[j] = cuts[:3], cuts[1:]
        out[name] = (st, en)
    return out


def season_features(series, bounds):
    """series: list of [D, T] daily arrays -> dict season -> [D, n_years, 3 * len(series)] sub-period means."""
    out = {}
    for name, (st, en) in bounds.items():
        blocks = []
        for x in series:
            c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1, dtype=np.float64)], axis=1)
            blocks.append((c[:, en] - c[:, st]) / (en - st))            # [D, Y, 3]
        out[name] = np.concatenate(blocks, axis=2).astype(np.float32)
    return out


def make_feature_bank(wx, vs_params=None):
    bounds = season_bounds(wx["dates"])
    raw = season_features([wx["W"][:, :, v] for v in range(5)], bounds)
    vs = virtual_sensors(wx, vs_params)
    lat = season_features([vs[n] for n in SENSORS], bounds)
    return dict(raw=raw, latent=lat, ph=wx["ph"])


def _gather(bank_arm, s, lag, cols=None):
    out = None
    for season, code in (("kharif", 0), ("rabi", 1)):
        F = bank_arm[season]
        if cols is not None:
            F = F[:, :, cols]
        if out is None:
            out = np.zeros((len(s), F.shape[2]), np.float32)
        m = (s["is_rabi"].to_numpy() == code)
        out[m] = F[s["d_idx"].to_numpy()[m], s["year"].to_numpy()[m] - lag - Y0]
    return out


# ----------------------------------------------------------------------------- features
def build_features(s, bank, tr_idx, arm="latent", k=2, regime="level_ymem", sensors=SENSORS, climate=True):
    """
    arm     : 'raw' (5 weather variables + soil pH) | 'latent' (virtual sensors) | 'both'
    k       : memory window - seasons of years t, t-1 .. t-k are used
    regime  : 'level_env'  target = yield (t/ha), climate representation only
              'level_ymem' target = yield, plus lagged yield anomalies YA(t-1..t-k)   (the paper's H_t, Eq. 11)
              'anom'       target = log(yield / mean of the previous 3 years), climate representation only
    Shared covariate in every arm: crop identity (one-hot). All statistics come from training rows only.
    """
    tr = s.iloc[tr_idx]
    if regime == "anom":
        mu, sd = 0.0, 1.0
        target = np.log(s["y"].to_numpy() / s[[f"y_l{L}" for L in (1, 2, 3)]].mean(axis=1).to_numpy())
    else:
        mu, sd = float(tr["y"].mean()), float(tr["y"].std())
        target = (s["y"].to_numpy() - mu) / sd
    cols, names = [], []
    if climate and arm in ("raw", "both"):
        for L in range(k + 1):
            cols.append(_gather(bank["raw"], s, L)); names += [f"{v}_p{j}_t-{L}" for v in RAW_VARS for j in (1, 2, 3)]
        cols.append(bank["ph"][s["d_idx"].to_numpy()][:, None]); names.append("soil_ph")
    if climate and arm in ("latent", "both"):
        sel = [3 * SENSORS.index(n) + j for n in sensors for j in range(3)]
        for L in range(k + 1):
            cols.append(_gather(bank["latent"], s, L, sel)); names += [f"{n}_p{j}_t-{L}" for n in sensors for j in (1, 2, 3)]
    if cols:
        X = np.column_stack(cols).astype(np.float64)
        m_, s_ = X[tr_idx].mean(0), X[tr_idx].std(0)
        X = (X - m_) / np.where(s_ > 0, s_, 1.0)
        cols = [X]
    if regime == "level_ymem":
        ya = np.column_stack([(s[f"y_l{L}"].to_numpy() - mu) / sd for L in range(1, max(k, 1) + 1)])
        cols.append(ya); names += [f"YA_t-{L}" for L in range(1, max(k, 1) + 1)]
    oh = np.eye(len(CROPS))[s["crop_code"].to_numpy()]
    cols.append(oh); names += [f"crop_{c}" for c in CROPS]
    return np.column_stack(cols), target, (mu, sd), names


# ----------------------------------------------------------------------------- hold-out protocols
SCENARIOS = ["temporal", "cross_district", "multi_state", "multi_crop", "combined"]
UNIT = dict(temporal="dist", cross_district="dist", multi_state="state", multi_crop="crop", combined="dist")
SC_NAME = dict(temporal="Temporal hold-out", cross_district="Cross-district", multi_state="Multi-state",
               multi_crop="Multi-crop", combined="Combined state-crop")


def splits(s, scenario, seed=0):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(s))
    states = np.array(sorted(s["state"].unique())); crops = np.array(list(CROPS))
    s_fold = dict(zip(rng.permutation(states), np.arange(len(states)) % 5))
    c_fold = dict(zip(rng.permutation(crops), np.arange(len(crops)) % 5))
    d_fold = {}
    for _, g in s.drop_duplicates("dist").groupby("state"):               # districts spread over folds within each state
        d_fold.update(zip(rng.permutation(g["dist"].to_numpy()), (np.arange(len(g)) + rng.randint(5)) % 5))
    sf, cf, df_ = s["state"].map(s_fold).to_numpy(), s["crop"].map(c_fold).to_numpy(), s["dist"].map(d_fold).to_numpy()
    yr = s["year"].to_numpy()
    if scenario == "temporal":
        yield idx[yr <= TRAIN_END], idx[yr > TRAIN_END]
    elif scenario == "cross_district":
        for f in range(5):
            yield idx[df_ != f], idx[df_ == f]
    elif scenario == "multi_state":
        for f in range(5):
            yield idx[sf != f], idx[sf == f]
    elif scenario == "multi_crop":
        for c in crops:
            m = (s["crop"] == c).to_numpy()
            yield idx[~m], idx[m]
    elif scenario == "combined":
        for f in range(5):
            te = idx[(sf == f) & (cf == f)]
            if len(te):
                yield idx[(sf != f) & (cf != f)], te
    else:
        raise ValueError(scenario)


# ----------------------------------------------------------------------------- models and metrics
def fit_predict_tree(model, Xtr, ytr, Xte, seed):
    if model == "rf":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=N_JOBS)
    elif model == "xgb":
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                         random_state=seed, tree_method="hist", n_jobs=N_JOBS)
    else:
        raise ValueError(model)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def metrics(y, p):
    e = p - y
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                r2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)))


def unit_rmse(units, y, p):
    return np.sqrt(pd.DataFrame(dict(u=units, se=(p - y) ** 2)).groupby("u")["se"].mean())


def collect(s, scenario, regime, preds, seeds, extra=None):
    """Pool test predictions over folds; per-seed metrics, seed-averaged predictions, per-unit RMSE."""
    te = ~np.isnan(preds[0])
    if regime == "anom":
        y = np.log(s["y"].to_numpy() / s[[f"y_l{L}" for L in (1, 2, 3)]].mean(axis=1).to_numpy())[te]
    else:
        y = s["y"].to_numpy()[te]
    per_seed = [metrics(y, preds[j, te]) for j in range(len(seeds))]
    units = s[UNIT[scenario]].to_numpy()[te]
    base = unit_rmse(units, y, preds[0, te])
    ur = pd.Series(np.mean([unit_rmse(units, y, preds[j, te]).to_numpy() for j in range(len(seeds))], axis=0), index=base.index)
    out = dict(per_seed=per_seed, pred=preds[:, te].mean(0).astype(np.float32), y=y.astype(np.float32), unit_rmse=ur)
    if extra:
        out.update(extra)
    return out


def run_config(s, bank, scenario, model, seeds, regime="level_ymem", **feat):
    preds = np.full((len(seeds), len(s)), np.nan)
    for tr, te in splits(s, scenario):
        X, yt, (mu, sd), _ = build_features(s, bank, tr, regime=regime, **feat)
        for j, seed in enumerate(seeds):
            preds[j, te] = fit_predict_tree(model, X[tr], yt[tr], X[te], seed) * sd + mu
    return collect(s, scenario, regime, preds, seeds)


def summarise(per_seed):
    out = {}
    for m in ("rmse", "mae", "r2"):
        v = np.array([d[m] for d in per_seed])
        out[m] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    return out


def paired_test(u_a, u_b, n_boot=5000, seed=0):
    """Paired comparison of per-unit RMSE (b - a): Wilcoxon signed-rank and bootstrap 95% CI of the mean difference."""
    from scipy.stats import wilcoxon
    a, b = u_a.align(u_b, join="inner")
    d = (b - a).to_numpy()
    rng = np.random.RandomState(seed)
    boots = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)]
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = float("nan")
    return dict(n=int(len(d)), mean_diff=float(d.mean()), ci=(float(np.percentile(boots, 2.5)),
                float(np.percentile(boots, 97.5))), p=p, frac_better=float((d < 0).mean()))
