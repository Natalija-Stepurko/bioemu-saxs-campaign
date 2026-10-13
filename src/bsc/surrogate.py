"""Stage `surrogate` (ACQUISITION_DESIGN §3, E5): a small correction to BioEmu-1 with parameters shared
across proteins, the model that acquired SAXS profiles update.

For protein p with conformer curves I_pk(q) and conformer radii R_pk (Guinier, from each curve):

    population shift   w_pk ∝ exp(−λ_p z_pk),  z_pk = (ln R_pk − mean_k ln R_pk) / s_z,  λ_p = θ_λ · x_p
    size scaling       Î_p(q) = Σ_k w_pk I_pk(q·s_p),  ln s_p = b·tanh(θ_s · x_p / b),  b = ln 1.25

so s_p stays inside [0.8, 1.25] and θ = 0 gives the uniform average of the conformer curves, the raw
BioEmu-1 prediction scored in `score`. s_z is one constant, the pooled within-ensemble SD of ln R over the
clean entries, so that λ is in units of that spread. x_p = (1, disorder score, ln length, u_p), the three
non-constant features standardised over the clean entries, where u_p is the log ratio of the model's own
ensemble Rg to the scaling-law Rg for the chain's length and disorder score (features.FOLDED_LAW and
DISORDERED_LAW blended by the disorder score, as in the `predict` stage).

I_pk(q·s) is read from a cubic Hermite interpolant in ln q through the computed points (node slopes by
three-point differences), built once per protein; it passes through every computed point, so s = 1
reproduces the computed curves exactly. Only q points whose rescaled value q·s stays inside the computed
grid are scored (with a linear taper over one grid spacing beyond either end, so the score is continuous in s;
ACQUISITION_DESIGN §10). Each profile's scale and constant background are solved inside χ² by weighted least
squares, as in `score`; the reduced χ² divides by the number of scored points minus two.

Fit: θ̂ minimises Σ_p χ²_p(θ)/2 + |θ|²/(2τ²) (L-BFGS, analytic gradient). τ is chosen once, by grouped
5-fold cross-validation inside the first split's acquisition pool.

Output: results/acq_surrogate.json (θ = 0 check over every entry, τ selection, E5 on the five splits,
timing of one fit).
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import logsumexp

from bsc import analyse, features, saxs, score
from bsc import config as C

N_THETA = 8  # θ_λ (4) then θ_s (4)
LOG_S_BOUND = float(np.log(1.25))
TAU_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
N_SPLITS, HELD_OUT = 5, 0.30
SPLIT_SEED = 20261013
E5_THRESHOLD = 0.10
MAX_RESTARTS = 2
CACHE = C.DATA / "acquisition_cache.pkl"


# ---------------------------------------------------------------------------- pre-measurement data
@dataclass
class Computed:
    """What is known about a protein before its profile is measured: computed curves and features."""

    label: str
    q: np.ndarray  # computed q grid, as in the archive (may repeat a value)
    node_u: np.ndarray  # sorted unique ln q
    node_of: np.ndarray  # index of each q point into node_u
    Y: np.ndarray  # conformer curves at the nodes, (K, n_nodes)
    z: np.ndarray  # standardised conformer log size relative to the ensemble mean, (K,)
    lnR: np.ndarray  # conformer ln Rg, (K,)
    x: np.ndarray  # features (1, disorder, ln length, u), standardised, (4,)
    dl: np.ndarray  # node-slope stencil: m_i = dl_i F_{i-1} + dc_i F_i + du_i F_{i+1}
    dc: np.ndarray
    du: np.ndarray

    @property
    def h(self) -> np.ndarray:
        return np.diff(self.node_u)


@dataclass
class Experiment:
    """A measured profile on the computed grid: the points with a positive error."""

    label: str
    pts: np.ndarray  # indices of the kept points into Computed.q
    I: np.ndarray
    sigma: np.ndarray


def slope_stencil(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coefficients of the three-point derivative on a non-uniform grid (one-sided at the ends)."""
    n = len(u)
    dl, dc, du = np.zeros(n), np.zeros(n), np.zeros(n)
    h = np.diff(u)
    h0, h1 = h[:-1], h[1:]
    dl[1:-1] = -h1 / (h0 * (h0 + h1))
    dc[1:-1] = (h1 - h0) / (h0 * h1)
    du[1:-1] = h0 / (h1 * (h0 + h1))
    dc[0], du[0] = -1 / h[0], 1 / h[0]
    dl[-1], dc[-1] = -1 / h[-1], 1 / h[-1]
    return dl, dc, du


def node_slopes(F: np.ndarray, c: Computed) -> np.ndarray:
    """Slopes dF/d(ln q) at the nodes for each row of F, (m, n_nodes)."""
    M = c.dc * F
    M[:, 1:] += c.dl[1:] * F[:, :-1]
    M[:, :-1] += c.du[:-1] * F[:, 1:]
    return M


def scaling_law_rg(length: np.ndarray, disorder: np.ndarray) -> np.ndarray:
    f = np.asarray(disorder, float)
    return (1 - f) * features.scaling_rg(length, features.FOLDED_LAW) + f * features.scaling_rg(
        length, features.DISORDERED_LAW
    )


def read_curves(p) -> tuple[np.ndarray, np.ndarray]:
    """q grid (header, read as numbers so repeated values survive) and the finite conformer curves."""
    with open(p) as f:
        q = np.array([float(v) for v in f.readline().strip().split(",")[1:]])
    Y = pd.read_csv(p, index_col=0).to_numpy(float)
    return q, Y[np.isfinite(Y).all(axis=1)]


def _load_computed_raw(label: str, model: str = C.MODEL) -> dict | None:
    """Conformer curves and per-conformer Guinier radii from the archive (no experimental data)."""
    p = C.PRED_DIR / model / f"{C.PREDICTOR}-{label}.csv"
    if not p.exists():
        return None
    q, Y = read_curves(p)
    raw = Y.mean(axis=0)
    g = saxs.guinier(q, raw, 0.01 * raw + 1e-9, C.GUINIER_QRG_MAX)
    rg = np.array([saxs.guinier(q, y, 0.01 * y + 1e-9, C.GUINIER_QRG_MAX)["rg"] for y in Y])
    return {"label": label, "q": q, "Y": Y, "rg_conf": rg, "rg_model": g["rg"]}


def model_rg(label: str, model: str) -> float:
    """Guinier Rg of a model's uniform ensemble curve over the whole computed grid (for O3)."""
    p = C.PRED_DIR / model / f"{C.PREDICTOR}-{label}.csv"
    if not p.exists():
        return float("nan")
    q, Y = read_curves(p)
    if len(Y) == 0:
        return float("nan")
    raw = Y.mean(axis=0)
    return float(saxs.guinier(q, raw, 0.01 * raw + 1e-9, C.GUINIER_QRG_MAX)["rg"])


def pool_table() -> pd.DataFrame:
    """The clean entries with a BioEmu-1 ensemble (the 399 of the first study), with their sequence data.
    Cleanliness is the quality gate any campaign applies before a profile is used."""
    ent, d = analyse.load()
    m = d[(d["model"] == C.MODEL) & d["clean"]]
    t = ent.set_index("label").loc[m["label"]].reset_index()
    groups = pd.read_csv(C.RESULTS / "clusters.csv").set_index("label")["group"]
    t["group"] = groups.loc[t["label"]].to_numpy()
    return t[["label", "sequence", "length", "disorder_mean", "disorder_class", "group"]].reset_index(
        drop=True
    )


def build_cache(table: pd.DataFrame | None = None, n_jobs: int | None = None) -> dict:
    """Read every pool protein's computed curves once and derive the pre-measurement quantities."""
    table = pool_table() if table is None else table
    raws = Parallel(n_jobs=n_jobs or C.N_JOBS)(delayed(_load_computed_raw)(lab) for lab in table["label"])
    keep = [r is not None for r in raws]
    table, raws = table[keep].reset_index(drop=True), [r for r in raws if r is not None]
    lnr = [np.log(r["rg_conf"]) for r in raws]
    dev = np.concatenate([v[np.isfinite(v)] - np.nanmean(v) for v in lnr])
    s_z = float(np.std(dev))
    length = table["length"].to_numpy(float)
    dis = table["disorder_mean"].to_numpy(float)
    rg_model = np.array([r["rg_model"] for r in raws])
    u = np.log(rg_model / scaling_law_rg(length, dis))
    F = np.column_stack([dis, np.log(length), u])
    mu, sd = F.mean(axis=0), F.std(axis=0)
    X = np.column_stack([np.ones(len(F)), (F - mu) / sd])
    comps = {}
    for i, r in enumerate(raws):
        q = r["q"]
        uq = np.log(q)
        node_u, node_of = np.unique(uq, return_inverse=True)
        first = np.array([np.flatnonzero(node_of == j)[0] for j in range(len(node_u))])
        zz = lnr[i] - np.nanmean(lnr[i])
        zz = np.where(np.isfinite(zz), zz, 0.0) / s_z
        dl, dc, du = slope_stencil(node_u)
        comps[r["label"]] = Computed(
            label=r["label"],
            q=q,
            node_u=node_u,
            node_of=node_of,
            Y=np.ascontiguousarray(r["Y"][:, first]),
            z=zz,
            lnR=lnr[i],
            x=X[i],
            dl=dl,
            dc=dc,
            du=du,
        )
    feats = table.assign(
        rg_model=rg_model,
        u=u,
        rg_conformer_median=[float(np.nanmedian(r["rg_conf"])) for r in raws],
        rg_conformer_iqr=[
            float(np.nanpercentile(r["rg_conf"], 75) - np.nanpercentile(r["rg_conf"], 25)) for r in raws
        ],
    )
    return {
        "table": feats,
        "computed": comps,
        "s_z": s_z,
        "x_mean": mu.tolist(),
        "x_sd": sd.tolist(),
        "x_names": ["intercept", "disorder score", "ln length", "u (ensemble Rg over scaling law)"],
    }


def load_cache(rebuild: bool = False) -> dict:
    if CACHE.exists() and not rebuild:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    cache = build_cache()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(cache, f)
    return cache


def read_experiment(c: Computed) -> Experiment:
    """The measured profile of an acquired protein, read from disk only when it is acquired."""
    e = score.read_experiment(c.label)
    if len(e) != len(c.q):
        raise ValueError(f"{c.label}: {len(e)} measured points, {len(c.q)} computed")
    s = e["sigma"].to_numpy(float)
    keep = np.flatnonzero(np.isfinite(s) & (s > 0))
    return Experiment(c.label, keep, e["I"].to_numpy(float)[keep], s[keep])


class Experiments:
    """Reads profiles lazily, so nothing that is not asked for is ever opened."""

    def __init__(self, comps: dict[str, Computed]):
        self.comps, self._d = comps, {}

    def __getitem__(self, label: str) -> Experiment:
        if label not in self._d:
            self._d[label] = read_experiment(self.comps[label])
        return self._d[label]


# ---------------------------------------------------------------------------- the corrected curve
def lam_eta(theta: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    return float(theta[:4] @ x), float(theta[4:] @ x)


def log_scale(eta: float) -> tuple[float, float]:
    """ln s and d ln s / d eta for the bounded size scaling."""
    ls = LOG_S_BOUND * np.tanh(eta / LOG_S_BOUND)
    return float(ls), float(1.0 - (ls / LOG_S_BOUND) ** 2)


def weights(z: np.ndarray, lam: float) -> np.ndarray:
    a = -lam * z
    return np.exp(a - logsumexp(a))


def curve(
    c: Computed,
    lam: float,
    eta: float,
    pts: np.ndarray | None = None,
    grad: bool = True,
    weights_override: np.ndarray | None = None,
) -> dict:
    """Î at the points `pts` (indices into c.q; all by default) that are scored after scaling, with dÎ/dλ
    and dÎ/dη. A point whose rescaled q stays inside the computed grid has weight 1; the weight falls
    linearly to 0 over one grid spacing beyond either end (the last Hermite segment is extended that far),
    so that the scored set changes continuously with s. Returns the kept mask over `pts`, the weights of
    the kept points and their derivative with respect to η."""
    pts = np.arange(len(c.q)) if pts is None else pts
    ls, dls = log_scale(eta)
    w = weights(c.z, lam) if weights_override is None else weights_override
    rows = [w]
    if grad:
        rows.append(-w * (c.z - w @ c.z))
    F = np.vstack(rows) @ c.Y
    U = c.node_u
    t = U[c.node_of[pts]] + ls
    h_lo, h_hi = U[1] - U[0], U[-1] - U[-2]
    over, under = t - U[-1], U[0] - t
    wq = 1.0 - np.clip(over / h_hi, 0, None) - np.clip(under / h_lo, 0, None)
    inside = wq > 0
    t, wq, over, under = t[inside], wq[inside], over[inside], under[inside]
    dwq = (np.where(under > 0, 1.0 / h_lo, 0.0) - np.where(over > 0, 1.0 / h_hi, 0.0)) * dls
    n = len(U)
    i = np.clip(np.searchsorted(U, t, side="right") - 1, 0, n - 2)
    h = U[i + 1] - U[i]
    s = (t - U[i]) / h
    s2, s3 = s * s, s * s * s
    h00, h10, h01, h11 = 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s, -2 * s3 + 3 * s2, s3 - s2
    M = node_slopes(F, c)
    val = h00 * F[:, i] + h10 * h * M[:, i] + h01 * F[:, i + 1] + h11 * h * M[:, i + 1]
    out = {"I": val[0], "inside": inside, "wq": wq, "dwq_deta": dwq}
    if grad:
        dval = (
            (6 * s2 - 6 * s) * (F[0, i] - F[0, i + 1]) / h
            + (3 * s2 - 4 * s + 1) * M[0, i]
            + (3 * s2 - 2 * s) * M[0, i + 1]
        )
        out["dI_dlam"] = val[1]
        out["dI_deta"] = dval * dls
    return out


def _fit_scale(A: np.ndarray, e: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Weighted least-squares scale and background on standardised arrays."""
    aa, ae, ee, ay, ey = A @ A, A @ e, e @ e, A @ y, e @ y
    det = aa * ee - ae * ae
    return (ay * ee - ae * ey) / det, (aa * ey - ae * ay) / det


def chi2_terms(
    c: Computed, e: Experiment, theta: np.ndarray, grad: bool = True
) -> tuple[float, np.ndarray | None]:
    """Reduced χ² of the corrected curve against the profile, with its gradient over θ."""
    lam, eta = lam_eta(theta, c.x)
    cv = curve(c, lam, eta, e.pts, grad=grad)
    m, wq = cv["inside"], cv["wq"]
    sig = e.sigma[m]
    A, ones, y = cv["I"] / sig, 1.0 / sig, e.I[m] / sig
    rw = np.sqrt(wq)
    sc, bg = _fit_scale(A * rw, ones * rw, y * rw)
    r = sc * A + bg * ones - y
    Q, D = float(wq @ (r * r)), float(wq.sum() - 2)
    chi2 = Q / D
    if not grad:
        return chi2, None
    g_lam = 2.0 * sc * ((wq * r) @ (cv["dI_dlam"] / sig)) / D
    dw = cv["dwq_deta"]
    g_eta = (2.0 * sc * ((wq * r) @ (cv["dI_deta"] / sig)) + dw @ (r * r)) / D - Q / D**2 * dw.sum()
    return chi2, np.concatenate([g_lam * c.x, g_eta * c.x])


def raw_chi2(c: Computed, e: Experiment) -> float:
    return chi2_terms(c, e, np.zeros(N_THETA), grad=False)[0]


def objective(
    theta: np.ndarray, items: list[tuple[Computed, Experiment]], tau: float
) -> tuple[float, np.ndarray]:
    val = 0.5 * theta @ theta / tau**2
    g = theta / tau**2
    for c, e in items:
        x2, gx = chi2_terms(c, e, theta)
        val += 0.5 * x2
        g = g + 0.5 * gx
    return float(val), g


def fit(items: list[tuple[Computed, Experiment]], tau: float, theta0: np.ndarray | None = None) -> np.ndarray:
    if not items:
        return np.zeros(N_THETA)
    x = np.zeros(N_THETA) if theta0 is None else np.asarray(theta0, float)
    for _ in range(MAX_RESTARTS + 1):
        res = minimize(
            objective,
            x,
            args=(items, tau),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 500, "gtol": 1e-6, "ftol": 1e-10},
        )
        x = res.x
        if res.success:  # a line-search stop at a kink of the taper is restarted with fresh memory
            break
    return x


def chi2_many(theta: np.ndarray, items: list[tuple[Computed, Experiment]]) -> np.ndarray:
    return np.array([chi2_terms(c, e, theta, grad=False)[0] for c, e in items])


def improvement(theta: np.ndarray, held: list[tuple[Computed, Experiment]], base: np.ndarray) -> float:
    """Median over held-out profiles of the log10 χ² reduction against θ = 0 (the primary score)."""
    return float(np.median(np.log10(base) - np.log10(chi2_many(theta, held))))


def corrected_curve(
    c: Computed, e: Experiment, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(q, Î, I, σ) on the points inside the grid after scaling, for the secondary scores."""
    lam, eta = lam_eta(theta, c.x)
    cv = curve(c, lam, eta, e.pts, grad=False)
    m = np.flatnonzero(cv["inside"])
    full = cv["wq"] >= 1.0
    return c.q[e.pts][m][full], cv["I"][full], e.I[m][full], e.sigma[m][full]


# ---------------------------------------------------------------------------- splits and E5
def splits(
    table: pd.DataFrame, n_splits: int = N_SPLITS, held_out: float = HELD_OUT, seed: int = SPLIT_SEED
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped splits: whole sequence groups are held out, in random order, until 30% of the profiles."""
    groups = table["group"].to_numpy()
    uniq = np.unique(groups)
    target = int(round(held_out * len(table)))
    out = []
    for k in range(n_splits):
        order = np.random.default_rng(seed + k).permutation(uniq)
        held, n = [], 0
        for g in order:
            if n >= target:
                break
            held.append(g)
            n += int((groups == g).sum())
        te = np.flatnonzero(np.isin(groups, held))
        tr = np.flatnonzero(~np.isin(groups, held))
        out.append((tr, te))
    return out


def kfold_groups(groups: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    uniq = np.random.default_rng(seed).permutation(np.unique(groups))
    return [np.flatnonzero(np.isin(groups, uniq[i::k])) for i in range(k)]


def choose_tau(
    items: list[tuple[Computed, Experiment]],
    groups: np.ndarray,
    base: np.ndarray,
    tau_grid=TAU_GRID,
    k: int = 5,
    seed: int = SPLIT_SEED,
    n_jobs: int = 1,
) -> dict:
    folds = kfold_groups(groups, k, seed)

    def one(tau, f):
        val = set(folds[f].tolist())
        tr = [it for j, it in enumerate(items) if j not in val]
        te = [items[j] for j in folds[f]]
        th = fit(tr, tau)
        return improvement(th, te, base[folds[f]])

    res = Parallel(n_jobs=n_jobs)(delayed(one)(t, f) for t in tau_grid for f in range(k))
    scores = np.array(res).reshape(len(tau_grid), k)
    best = int(np.argmax(scores.mean(axis=1)))
    return {
        "tau_grid": list(tau_grid),
        "cv_mean_improvement": scores.mean(axis=1).tolist(),
        "cv_per_fold": scores.tolist(),
        "tau": float(tau_grid[best]),
        "n_folds": k,
    }


def secondary(theta: np.ndarray, held: list[tuple[Computed, Experiment]], table: pd.DataFrame) -> dict:
    """NRMSD and the median Rg ratio of disordered held-out proteins, raw and corrected."""
    cls = table.set_index("label")["disorder_class"]
    ent = pd.read_csv(C.RESULTS / "entries.csv").set_index("label")
    out = {"nrmsd": {}, "rg_ratio_disordered": {}}
    for name, th in (("raw", np.zeros(N_THETA)), ("corrected", theta)):
        nr, rr = [], []
        for c, e in held:
            q, Ih, I, s = corrected_curve(c, e, th)
            nr.append(saxs.nrmsd_log(Ih, I, s))
            if cls[c.label] == "disordered":
                g = saxs.guinier(q, Ih, 0.01 * Ih + 1e-9, C.GUINIER_QRG_MAX)
                rr.append(g["rg"] / ent.loc[c.label, "rg_exp"])
        out["nrmsd"][name] = float(np.nanmedian(nr))
        out["rg_ratio_disordered"][name] = float(np.nanmedian(rr))
        out["n_disordered"] = len(rr)
    return out


def items_for(labels, comps: dict[str, Computed], exps: Experiments) -> list[tuple[Computed, Experiment]]:
    return [(comps[lab], exps[lab]) for lab in labels]


def main() -> None:
    cache = load_cache()
    table, comps = cache["table"], cache["computed"]
    exps = Experiments(comps)
    labels = table["label"].to_numpy()
    all_items = items_for(labels, comps, exps)
    # θ = 0 reproduces the raw χ² of every entry
    base = chi2_many(np.zeros(N_THETA), all_items)
    sc = pd.read_csv(C.RESULTS / "scores.csv")
    ref = sc[sc["model"] == C.MODEL].set_index("label").loc[labels, "chi2_raw"].to_numpy()
    rel = np.abs(base - ref) / ref
    check = {
        "n": int(len(labels)),
        "max_relative_difference": float(rel.max()),
        "median_raw_chi2": float(np.median(base)),
    }
    print(
        f"  θ = 0 against the scored raw χ² on {len(labels)} entries: max relative difference "
        f"{rel.max():.2e}",
        flush=True,
    )
    sp = splits(table)
    tr0, te0 = sp[0]
    t0 = time.time()
    tau_sel = choose_tau(
        [all_items[i] for i in tr0], table["group"].to_numpy()[tr0], base[tr0], n_jobs=C.N_JOBS
    )
    tau = tau_sel["tau"]
    print(
        f"  τ by grouped CV in the first pool: {tau} (CV improvement "
        f"{dict(zip(tau_sel['tau_grid'], np.round(tau_sel['cv_mean_improvement'], 4), strict=True))}; "
        f"{time.time() - t0:.0f} s)",
        flush=True,
    )

    def per_split(k, tr, te):
        t1 = time.time()
        th = fit([all_items[i] for i in tr], tau)
        sec = time.time() - t1
        held = [all_items[i] for i in te]
        x2 = chi2_many(th, held)
        rec = {
            "split": k,
            "n_pool": int(len(tr)),
            "n_held_out": int(len(te)),
            "theta": th.tolist(),
            "fit_seconds": sec,
            "median_chi2_raw": float(np.median(base[te])),
            "median_chi2_corrected": float(np.median(x2)),
            "median_log10_reduction": float(np.median(np.log10(base[te]) - np.log10(x2))),
            "share_improved": float(np.mean(x2 < base[te])),
            "secondary": secondary(th, held, table),
        }
        rec["relative_reduction_median_chi2"] = 1 - rec["median_chi2_corrected"] / rec["median_chi2_raw"]
        cls = table["disorder_class"].to_numpy()[te]
        rec["median_log10_reduction_by_class"] = {
            c: float(np.median(np.log10(base[te][cls == c]) - np.log10(x2[cls == c])))
            for c in analyse.CLASS_ORDER
        }
        rec["_rows"] = pd.DataFrame(
            {
                "split": k,
                "label": labels[te],
                "disorder_class": cls,
                "chi2_raw": base[te],
                "chi2_corrected": x2,
            }
        )
        return rec

    recs = Parallel(n_jobs=min(C.N_JOBS, N_SPLITS))(
        delayed(per_split)(k, tr, te) for k, (tr, te) in enumerate(sp)
    )
    held_rows = pd.concat([r.pop("_rows") for r in recs], ignore_index=True)
    # is the whole-pool optimum the one L-BFGS finds from θ = 0? Four random starts on the first split
    pool0 = [all_items[i] for i in tr0]
    starts = [np.zeros(N_THETA)] + [
        np.r_[rng.normal(0, 0.5, 4), rng.normal(0, 0.05, 4)]
        for rng in [np.random.default_rng(SPLIT_SEED + 50 + j) for j in range(4)]
    ]

    def from_start(x0):
        th = fit(pool0, tau, theta0=x0)
        return {
            "objective": objective(th, pool0, tau)[0],
            "theta": th.tolist(),
            "held_out_median_log10_reduction": improvement(th, [all_items[i] for i in te0], base[te0]),
        }

    multistart = Parallel(n_jobs=min(C.N_JOBS, len(starts)))(delayed(from_start)(x0) for x0 in starts)
    held_rows["log10_reduction"] = np.log10(held_rows["chi2_raw"]) - np.log10(held_rows["chi2_corrected"])
    held_rows.to_csv(C.RESULTS / "acq_surrogate_heldout.csv", index=False)
    pooled = {}
    for i, c in enumerate(analyse.CLASS_ORDER):
        v = held_rows.loc[held_rows["disorder_class"] == c, "log10_reduction"].to_numpy()
        pooled[c] = {
            "n": int(len(v)),
            "median_log10_reduction": float(np.median(v)),
            "ci95": analyse.ci(v, seed_offset=800 + i),
            "share_held_out": float(len(v) / len(held_rows)),
        }
    rr = np.array([r["relative_reduction_median_chi2"] for r in recs])
    e5 = {
        "statement": "fitted on the whole pool, the surrogate improves the held-out median chi2 "
        "by at least 10%",
        "per_split": recs,
        "mean_relative_reduction": float(rr.mean()),
        "min_relative_reduction": float(rr.min()),
        "max_relative_reduction": float(rr.max()),
        "mean_median_log10_reduction": float(np.mean([r["median_log10_reduction"] for r in recs])),
        "threshold": E5_THRESHOLD,
        "by_class_pooled": pooled,
        "rg_ratio_disordered_raw_mean": float(
            np.mean([r["secondary"]["rg_ratio_disordered"]["raw"] for r in recs])
        ),
        "rg_ratio_disordered_corrected_mean": float(
            np.mean([r["secondary"]["rg_ratio_disordered"]["corrected"] for r in recs])
        ),
        "met": bool(rr.mean() >= E5_THRESHOLD),
    }
    out = {
        "theta_zero_check": check,
        "s_z": cache["s_z"],
        "features": cache["x_names"],
        "feature_mean": cache["x_mean"],
        "feature_sd": cache["x_sd"],
        "tau_selection": tau_sel,
        "tau": tau,
        "splits": {
            "n": N_SPLITS,
            "held_out": HELD_OUT,
            "seed": SPLIT_SEED,
            "sizes": [[int(len(tr)), int(len(te))] for tr, te in sp],
        },
        "fit_seconds_full_pool": [r["fit_seconds"] for r in recs],
        "multistart_first_split": multistart,
        "E5": e5,
    }
    json.dump(out, open(C.RESULTS / "acq_surrogate.json", "w"), indent=1)
    print(
        f"  E5: held-out median χ² reduced by {rr.mean():.1%} on average over {N_SPLITS} splits "
        f"(range {rr.min():.1%} to {rr.max():.1%}); {'met' if e5['met'] else 'not met'}. One full-pool fit "
        f"{np.mean(out['fit_seconds_full_pool']):.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
