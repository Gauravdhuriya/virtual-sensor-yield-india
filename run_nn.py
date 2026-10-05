"""
Neural baselines (optional stage; needs:  pip install jax optax).

  lstm : LSTM over the yearly sequence (t-2, t-1, t) of season features
  tf   : Transformer encoder (2 layers, 4 heads, d = 32) over the same sequence
  pgnn : physics-guided network. The daily virtual-sensor recursions (Eq. 4-7) are a differentiable layer whose
         parameters (alpha, beta, lambda, gamma, delta, omega) are learned end to end with an MLP head.

  python run_nn.py --quick     smoke test          -> out_quick/nn.pkl
  python run_nn.py             full run (cached)   -> out/nn.pkl
"""
import argparse, os, pickle, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import jax, jax.numpy as jnp, optax
import adt_india as adt

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
OUT = "out_quick" if args.quick else "out"; os.makedirs(OUT, exist_ok=True)
H, D, HEADS, FF, K = 32, 32, 4, 64, 2
EPOCHS, BATCH, LR, WD = (3 if args.quick else 30), 256, 2e-3, 1e-4
PG_EPOCHS, PG_BATCH = (2 if args.quick else 15), 2048      # the physics-guided net re-runs the daily recursion every step
SEEDS = [0] if args.quick else [0, 1, 2]
N = adt.NOMINAL


# ------------------------------------------------------------------ inputs
def nn_inputs(s, bank, tr, arm, regime):
    if regime == "anom":
        mu, sd = 0.0, 1.0
        y = np.log(s["y"].to_numpy() / s[[f"y_l{L}" for L in (1, 2, 3)]].mean(axis=1).to_numpy())
    else:
        mu, sd = float(s["y"].iloc[tr].mean()), float(s["y"].iloc[tr].std())
        y = (s["y"].to_numpy() - mu) / sd
    steps = np.stack([adt._gather(bank[arm], s, L) for L in range(K, -1, -1)], 1)            # (N, K+1, d): t-K .. t
    m_, s_ = steps[tr].mean((0, 1)), steps[tr].std((0, 1))
    steps = (steps - m_) / np.where(s_ > 0, s_, 1.0)
    ymem = regime == "level_ymem"
    ya = np.stack([(s[f"y_l{L}"].to_numpy() - mu) / sd for L in range(K, 0, -1)], 1) if ymem else None
    if ymem:                                              # yield anomaly of the current step is unknown -> 0 + flag
        cur = np.tile(np.r_[np.zeros(K), 1.0], (len(s), 1))
        steps = np.concatenate([steps, np.column_stack([ya, np.zeros(len(s))])[:, :, None], cur[:, :, None]], 2)
    oh = np.eye(len(adt.CROPS))[s["crop_code"].to_numpy()]
    ph = bank["ph"][s["d_idx"].to_numpy()]
    ph = ((ph - ph[tr].mean()) / (ph[tr].std() + 1e-9))[:, None]
    static = np.column_stack([oh, ph]) if arm == "raw" else oh
    pg_static = np.column_stack([oh] + ([ya] if ymem else []))
    f32 = lambda a: np.asarray(a, np.float32)
    batch = dict(seq=f32(steps), static=f32(static), y=f32(y), pg_static=f32(pg_static),
                 d=s["d_idx"].to_numpy().astype(np.int32), yi=(s["year"].to_numpy() - adt.Y0).astype(np.int32),
                 rabi=s["is_rabi"].to_numpy().astype(np.int32))
    return batch, (mu, sd)


def pg_consts(wx):
    W = wx["W"]
    b = adt.season_bounds(wx["dates"])
    c = dict(rain=jnp.asarray(W[:, :, 0].T), temp=jnp.asarray(W[:, :, 1].T), wind=jnp.asarray(W[:, :, 2].T),
             solar=jnp.asarray(W[:, :, 3].T), ph=jnp.asarray(np.abs(wx["ph"] - N["ph_opt"]) / N["ph_opt"]))
    for name in ("kharif", "rabi"):
        c[f"{name}_st"], c[f"{name}_en"] = jnp.asarray(b[name][0]), jnp.asarray(b[name][1])
    return c


# ------------------------------------------------------------------ layers
def dense_init(key, i, o):
    return dict(W=jax.random.normal(key, (i, o)) * np.sqrt(2.0 / i), b=jnp.zeros(o))

def dense(p, x):
    return x @ p["W"] + p["b"]

def head_init(key, i):
    k1, k2 = jax.random.split(key)
    return dict(h=dense_init(k1, i, 32), o=dense_init(k2, 32, 1))

def head(p, x):
    return dense(p["o"], jax.nn.relu(dense(p["h"], x)))[:, 0]

def layernorm(p, x):
    m, v = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * p["g"] + p["b"]


def lstm_init(key, d_in, d_static):
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(Wx=jax.random.normal(k1, (d_in, 4 * H)) / np.sqrt(d_in), Wh=jax.random.normal(k2, (H, 4 * H)) / np.sqrt(H),
                b=jnp.zeros(4 * H).at[H:2 * H].set(1.0), head=head_init(k3, H + d_static))

def lstm_apply(p, c, b):
    x = jnp.swapaxes(b["seq"], 0, 1)
    def step(carry, xt):
        h, cc = carry
        i, f, g, o = jnp.split(xt @ p["Wx"] + h @ p["Wh"] + p["b"], 4, axis=-1)
        cc = jax.nn.sigmoid(f) * cc + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(cc)
        return (h, cc), None
    z0 = jnp.zeros((x.shape[1], H))
    (h, _), _ = jax.lax.scan(step, (z0, z0), x)
    return head(p["head"], jnp.concatenate([h, b["static"]], 1))


def tf_init(key, d_in, d_static):
    ks = jax.random.split(key, 12)
    ln = lambda: dict(g=jnp.ones(D), b=jnp.zeros(D))
    layers = [dict(qkv=dense_init(ks[2 + 4 * l], D, 3 * D), o=dense_init(ks[3 + 4 * l], D, D),
                   f1=dense_init(ks[4 + 4 * l], D, FF), f2=dense_init(ks[5 + 4 * l], FF, D), n1=ln(), n2=ln()) for l in range(2)]
    return dict(emb=dense_init(ks[0], d_in, D), pos=jax.random.normal(ks[1], (K + 1, D)) * 0.02, layers=layers, nf=ln(),
                head=head_init(ks[10], D + d_static))

def tf_apply(p, c, b):
    x = dense(p["emb"], b["seq"]) + p["pos"]
    n, T, _ = x.shape
    sp = lambda t: t.reshape(n, T, HEADS, D // HEADS).transpose(0, 2, 1, 3)
    for l in p["layers"]:
        q, k, v = jnp.split(dense(l["qkv"], layernorm(l["n1"], x)), 3, axis=-1)
        att = jax.nn.softmax(sp(q) @ sp(k).transpose(0, 1, 3, 2) / np.sqrt(D // HEADS), -1)
        x = x + dense(l["o"], (att @ sp(v)).transpose(0, 2, 1, 3).reshape(n, T, D))
        x = x + dense(l["f2"], jax.nn.relu(dense(l["f1"], layernorm(l["n2"], x))))
    return head(p["head"], jnp.concatenate([layernorm(p["nf"], x).mean(1), b["static"]], 1))


# ------------------------------------------------- physics-guided network
def logit(v):
    return float(np.log(v / (1 - v)))

def pgnn_init(key, d_static):
    k1, k2 = jax.random.split(key)
    o3 = 1 - N["omega1"] - N["omega2"]
    phys = dict(a=jnp.array(0.0), b=jnp.array(0.0), d=jnp.array(0.0), l=jnp.array(logit(N["lam"])),
                g=jnp.array(logit(float(np.exp(-1 / N["tau"])))), w=jnp.log(jnp.array([N["omega1"], N["omega2"], o3])))
    return dict(phys=phys, h1=dense_init(k1, 9 * (K + 1) + d_static, 64), head=head_init(k2, 64))

def phys_values(ph):
    w = jax.nn.softmax(ph["w"])
    return dict(alpha=N["alpha"] * jnp.exp(ph["a"]), beta=N["beta"] * jnp.exp(ph["b"]), delta=N["delta"] * jnp.exp(ph["d"]),
                lam=jax.nn.sigmoid(ph["l"]), gamma=jax.nn.sigmoid(ph["g"]), omega1=w[0], omega2=w[1], omega3=w[2])

def pgnn_bank(ph, c):
    v = phys_values(ph)
    et = v["lam"] * jnp.clip(c["temp"] / N["t_ref"], 0, 1) + (1 - v["lam"]) * jnp.clip(c["wind"] / N["w_ref"], 0, 1)
    forc = c["temp"] + v["delta"] * c["solar"]
    def step(carry, inp):
        m, t = carry
        r, e, f = inp
        m = jnp.clip(m + v["alpha"] * r - v["beta"] * e, 0.0, 1.0)
        t = v["gamma"] * t + (1 - v["gamma"]) * f
        return (m, t), (m, t)
    Dn = c["rain"].shape[1]
    _, (m, t) = jax.lax.scan(step, (jnp.full(Dn, N["vssm0"]), forc[0]), (c["rain"], et, forc))
    tn = t / N["t_ref"]
    cs = v["omega1"] * (1 - m) + v["omega2"] * tn + v["omega3"] * c["ph"][None, :]
    out = {}
    for name in ("kharif", "rabi"):
        st, en = c[f"{name}_st"], c[f"{name}_en"]
        blocks = []
        for x in (m, tn, cs):
            cum = jnp.concatenate([jnp.zeros((1, Dn)), jnp.cumsum(x, 0)], 0)
            blocks.append(jnp.transpose((cum[en] - cum[st]) / (en - st)[:, :, None], (2, 0, 1)))   # [D, Y, 3]
        out[name] = jnp.concatenate(blocks, 2)
    return out

def pgnn_apply(p, c, b):
    bank = pgnn_bank(p["phys"], c)
    feats = [jnp.where(b["rabi"][:, None] == 1, bank["rabi"][b["d"], b["yi"] - L], bank["kharif"][b["d"], b["yi"] - L])
             for L in range(K + 1)]
    x = (jnp.concatenate(feats, 1) - c["f_mean"]) / c["f_sd"]
    return head(p["head"], jax.nn.relu(dense(p["h1"], jnp.concatenate([x, b["pg_static"]], 1))))


APPLY = dict(lstm=lstm_apply, tf=tf_apply, pgnn=pgnn_apply)
_opt = optax.adamw(LR, weight_decay=WD)

def _make(kind):
    apply = APPLY[kind]
    def loss(p, c, b):
        return jnp.mean((apply(p, c, b) - b["y"]) ** 2)
    @jax.jit
    def step(p, o, c, b):
        l, g = jax.value_and_grad(loss)(p, c, b)
        u, o = _opt.update(g, o, p)
        return optax.apply_updates(p, u), o, l
    return step, jax.jit(apply)

_FNS = {k: _make(k) for k in APPLY}


def predict(kind, p, c, batch, idx, chunk):
    apply = _FNS[kind][1]
    out = []
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        pad = np.r_[j, np.full(chunk - len(j), j[0])]
        out.append(np.asarray(apply(p, c, {k: jnp.asarray(v[pad]) for k, v in batch.items()}))[:len(j)])
    return np.concatenate(out)


def fit_predict(kind, batch, c, tr, te, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(tr)
    nv = max(len(perm) // 10, 1)
    va, trn = perm[:nv], perm[nv:]
    key = jax.random.PRNGKey(seed)
    bs = PG_BATCH if kind == "pgnn" else BATCH
    if kind == "lstm":
        p = lstm_init(key, batch["seq"].shape[2], batch["static"].shape[1])
    elif kind == "tf":
        p = tf_init(key, batch["seq"].shape[2], batch["static"].shape[1])
    else:
        p = pgnn_init(key, batch["pg_static"].shape[1])
        c = dict(c, f_mean=jnp.zeros(9 * (K + 1)), f_sd=jnp.ones(9 * (K + 1)))       # standardise with nominal sensors
        bank = pgnn_bank(p["phys"], c)
        j = jnp.asarray(trn)
        f0 = jnp.concatenate([jnp.where(jnp.asarray(batch["rabi"])[j][:, None] == 1,
                              bank["rabi"][jnp.asarray(batch["d"])[j], jnp.asarray(batch["yi"])[j] - L],
                              bank["kharif"][jnp.asarray(batch["d"])[j], jnp.asarray(batch["yi"])[j] - L]) for L in range(K + 1)], 1)
        c["f_mean"], c["f_sd"] = f0.mean(0), jnp.where(f0.std(0) > 1e-6, f0.std(0), 1.0)
    step = _FNS[kind][0]
    o = _opt.init(p)
    best, best_p = np.inf, p
    for ep in range(PG_EPOCHS if kind == "pgnn" else EPOCHS):
        sh = rng.permutation(trn)
        for i in range(0, len(sh) - bs + 1, bs):
            p, o, _ = step(p, o, c, {k: jnp.asarray(v[sh[i:i + bs]]) for k, v in batch.items()})
        v = float(np.mean((predict(kind, p, c, batch, va, bs) - batch["y"][va]) ** 2))
        if v < best:
            best, best_p = v, p
    phys = {k: float(x) for k, x in phys_values(best_p["phys"]).items()} if kind == "pgnn" else None
    return predict(kind, best_p, c, batch, te, bs), phys


def run_nn(s, bank, consts, scenario, kind, arm, regime):
    preds = np.full((len(SEEDS), len(s)), np.nan)
    phys_log = []
    for tr, te in adt.splits(s, scenario):
        batch, (mu, sd) = nn_inputs(s, bank, tr, arm, regime)
        for j, seed in enumerate(SEEDS):
            pz, phys = fit_predict(kind, batch, consts, tr, te, seed)
            preds[j, te] = pz * sd + mu
            if phys:
                phys_log.append(phys)
    return adt.collect(s, scenario, regime, preds, SEEDS, extra=dict(phys=phys_log))


if __name__ == "__main__":
    t_start = time.time()
    s, wx, info = adt.load_all()
    bank = adt.make_feature_bank(wx)
    consts = pg_consts(wx)
    path = f"{OUT}/nn.pkl"
    res = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    for scenario in (["temporal"] if args.quick else ["temporal", "combined"]):
        for regime in ("level_ymem", "anom"):
            for kind, arm in (("lstm", "raw"), ("lstm", "latent"), ("tf", "raw"), ("tf", "latent"), ("pgnn", "raw")):
                key = (kind, scenario, regime, arm)
                if key in res:
                    continue
                t = time.time()
                res[key] = run_nn(s, bank, consts, scenario, kind, arm, regime)
                pickle.dump(res, open(path, "wb"))
                S = adt.summarise(res[key]["per_seed"])
                print(f"{str(key):55s} RMSE {S['rmse'][0]:.4f}  R2 {S['r2'][0]:.3f}  [{time.time()-t:.0f}s, total {(time.time()-t_start)/60:.0f} min]", flush=True)
    print("Finished. Next: python make_report.py" + (" --quick" if args.quick else ""))
