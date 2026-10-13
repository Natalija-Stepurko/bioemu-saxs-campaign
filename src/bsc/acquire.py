"""Stage `acquire` (ACQUISITION_DESIGN §5–§6): the retrospective acquisition simulation.

Five grouped splits (70% pool, 30% held out), ten seed sets of ten pool proteins per split, then eight
rounds of ten acquisitions per policy; after each round θ̂ is refitted on everything acquired and the
held-out profiles are scored. The primary metric is the area under the held-out improvement curve (median
log10 χ² reduction against θ = 0, from 10 to 90 acquired, divided by the width so it is a mean improvement),
compared with random choice on the same split and seed set, with 95% intervals from a bootstrap over splits
and seed sets.

Policies: random; greedy on O1–O5 and on the predicted error, batches filled with a coverage penalty (a
candidate sharing a sequence group with an acquired protein or an earlier pick of the batch waits until no
other candidate is left); Pareto, non-dominated sorting on (O1, O4, O5) with batches built greedily by
hypervolume contribution, and its feasibility-constrained variant (only candidates with O5 above the pool
median); a sweep of Chebyshev weightings of (O1, O4, O5); the oracle, which ranks by realised value
(`bsc value`, which reads the hidden profiles) and is an upper reference, not a deployable policy.

The decision rule of E5 is applied by the page and the report: when the surrogate fitted on the whole pool
does not improve the held-out median χ² by 10%, this comparison is reported as uninformative.
Output: results/acq_curves.csv (one row per split × seed × policy × round), results/acq_acquire.json,
results/acq_pareto_snapshot.csv.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata

from bsc import config as C
from bsc import objectives as O
from bsc import surrogate as S

N_SEEDS, N_INITIAL, BATCH, N_ROUNDS = 10, 10, 10, 8
CHEBYSHEV_ALPHA = (0.0, 0.25, 0.5, 0.75, 1.0)  # weight on O5; O1 and O4 share the rest equally
SINGLE = ("O1", "O2", "O3", "O4", "O5", "predicted error")
POLICIES = (
    "random",
    *SINGLE,
    "pareto",
    "pareto constrained",
    *[f"chebyshev {a:.2f}" for a in CHEBYSHEV_ALPHA],
    "oracle",
)
N_BOOT = 2000
_CTX = None


def context() -> O.Context:
    """One context per worker process, loaded once."""
    global _CTX
    if _CTX is None:
        cache = S.load_cache()
        _CTX = O.Context(cache, pd.read_csv(C.RESULTS / "acq_static.csv"))
    return _CTX


# ---------------------------------------------------------------------------- batch builders
def fill_static(score: np.ndarray, cand: np.ndarray, groups: np.ndarray, blocked: set, k: int) -> list[int]:
    """The k best candidates by score (largest first), with the coverage penalty."""
    order = cand[np.lexsort((cand, -score))]
    picked, late = [], []
    blocked = set(blocked)
    for i in order:
        if groups[i] in blocked:
            late.append(i)
            continue
        picked.append(int(i))
        blocked.add(groups[i])
        if len(picked) == k:
            return picked
    return picked + [int(i) for i in late[: k - len(picked)]]


def fill_o1(
    V: np.ndarray, M: np.ndarray, cand: np.ndarray, groups: np.ndarray, blocked: set, k: int
) -> list[int]:
    """Greedy on the batch log det: each pick conditions the information of the next."""
    M = M.copy()
    blocked = set(blocked)
    free = np.ones(len(cand), bool)
    picked = []
    for _ in range(k):
        gain = O.logdet_gain(V, np.linalg.inv(M))
        ok = free & np.array([groups[i] not in blocked for i in cand])
        use = ok if ok.any() else free
        j = int(np.flatnonzero(use)[np.argmax(gain[use])])
        picked.append(int(cand[j]))
        free[j] = False
        blocked.add(groups[cand[j]])
        M = M + V[j] @ V[j].T
    return picked


def fill_o4(
    cont: np.ndarray, cand: np.ndarray, acquired: np.ndarray, groups: np.ndarray, blocked: set, k: int
) -> list[int]:
    """Greedy on coverage: each pick becomes part of what the next must be far from."""
    acq = list(acquired)
    blocked = set(blocked)
    free = np.ones(len(cand), bool)
    picked = []
    for _ in range(k):
        cov = O.coverage(cont, cand, np.array(acq, int))
        ok = free & np.array([groups[i] not in blocked for i in cand])
        use = ok if ok.any() else free
        j = int(np.flatnonzero(use)[np.argmax(cov[use])])
        picked.append(int(cand[j]))
        acq.append(int(cand[j]))
        free[j] = False
        blocked.add(groups[cand[j]])
    return picked


def nondominated_fronts(F: np.ndarray) -> np.ndarray:
    """Front index (0 = non-dominated) of each row of F, all objectives maximised."""
    n = len(F)
    ge = (F[:, None, :] >= F[None, :, :]).all(axis=2)
    gt = (F[:, None, :] > F[None, :, :]).any(axis=2)
    dom = ge & gt  # dom[i, j]: i dominates j
    front = np.full(n, -1)
    left = np.ones(n, bool)
    f = 0
    while left.any():
        cur = np.zeros(n, bool)
        idx = np.flatnonzero(left)
        sub = dom[np.ix_(idx, idx)]
        cur[idx[~sub.any(axis=0)]] = True
        front[cur] = f
        left &= ~cur
        f += 1
    return front


def hypervolume(P: np.ndarray) -> float:
    """Hypervolume dominated by points in [0, 1]^3 (maximisation, reference at the origin)."""
    if len(P) == 0:
        return 0.0
    zs = np.unique(P[:, 2])[::-1]
    vol = 0.0
    for i, z in enumerate(zs):
        z_next = zs[i + 1] if i + 1 < len(zs) else 0.0
        Q = P[P[:, 2] >= z][:, :2]
        Q = Q[np.argsort(-Q[:, 0])]
        area, ymax = 0.0, 0.0
        for j in range(len(Q)):
            ymax = max(ymax, Q[j, 1])
            x_next = Q[j + 1, 0] if j + 1 < len(Q) else 0.0
            area += (Q[j, 0] - x_next) * ymax
        vol += (z - z_next) * area
    return vol


def ranks01(v: np.ndarray) -> np.ndarray:
    return rankdata(v) / len(v)


def fill_pareto(F: np.ndarray, cand: np.ndarray, groups: np.ndarray, blocked: set, k: int) -> list[int]:
    """Fronts in order; inside a front, the point adding the most hypervolume to the batch so far (ties:
    largest rank sum). The coverage penalty defers points that share a group with the batch or the
    acquired set."""
    R = np.column_stack([ranks01(F[:, j]) for j in range(F.shape[1])])
    front = nondominated_fronts(F)
    blocked = set(blocked)
    chosen: list[int] = []
    free = np.ones(len(cand), bool)
    for relax in (False, True):
        for f in range(front.max() + 1):
            while len(chosen) < k:
                pool = free & (front == f)
                if not relax:
                    pool &= np.array([groups[i] not in blocked for i in cand])
                idx = np.flatnonzero(pool)
                if len(idx) == 0:
                    break
                base = hypervolume(R[chosen]) if chosen else 0.0
                gain = np.array([hypervolume(R[chosen + [j]]) - base for j in idx])
                best = idx[np.lexsort((-R[idx].sum(axis=1), -gain))[0]]
                chosen.append(int(best))
                free[best] = False
                blocked.add(groups[cand[best]])
            if len(chosen) == k:
                return [int(cand[j]) for j in chosen]
    return [int(cand[j]) for j in chosen]


def fill_chebyshev(
    F: np.ndarray, w: np.ndarray, cand: np.ndarray, groups: np.ndarray, blocked: set, k: int
) -> list[int]:
    """Smallest weighted Chebyshev distance to the ideal point in rank space."""
    R = np.column_stack([ranks01(F[:, j]) for j in range(F.shape[1])])
    dist = (w[None, :] * (1 - R)).max(axis=1) + 1e-6 * (1 - R).sum(axis=1)
    return fill_static(-dist, cand, groups, blocked, k)


# ---------------------------------------------------------------------------- one run
def choose(
    policy: str,
    ctx: O.Context,
    theta: np.ndarray,
    acquired: np.ndarray,
    cand: np.ndarray,
    tau: float,
    rng: np.random.Generator,
    o5_median: float,
    value: pd.Series | None,
) -> list[int]:
    groups = ctx.groups
    blocked = set(groups[acquired])
    if policy == "random":
        return [int(i) for i in rng.choice(cand, BATCH, replace=False)]
    if policy == "oracle":
        return fill_static(
            value.reindex(ctx.labels[cand]).fillna(-np.inf).to_numpy(), cand, groups, blocked, BATCH
        )
    need = {policy} if policy in ("O2", "predicted error") else {"O1", "O4", "O5"}
    ob = ctx.objectives(theta, acquired, cand, tau, need=need)
    if policy == "O1":
        return fill_o1(ob["V"], ob["M"], cand, groups, blocked, BATCH)
    if policy == "O4":
        return fill_o4(ctx.cont, cand, acquired, groups, blocked, BATCH)
    if policy in ("O2", "O3", "O5", "predicted error"):
        return fill_static(ob[policy], cand, groups, blocked, BATCH)
    F = np.column_stack([ob["O1"], ob["O4"], ob["O5"]])
    if policy == "pareto":
        return fill_pareto(F, cand, groups, blocked, BATCH)
    if policy == "pareto constrained":
        keep = ob["O5"] > o5_median
        return fill_pareto(F[keep], cand[keep], groups, blocked, BATCH)
    if policy.startswith("chebyshev"):
        a = float(policy.split()[1])
        return fill_chebyshev(F, np.array([(1 - a) / 2, (1 - a) / 2, a]), cand, groups, blocked, BATCH)
    raise ValueError(policy)


def run(split: int, seed: int, policy: str, tau: float, value: pd.Series | None = None) -> list[dict]:
    ctx = context()
    tr, te = S.splits(ctx.table)[split]
    held = [ctx.item(i) for i in te]
    base = np.array([ctx.raw_chi2(i) for i in te])
    o5_median = float(np.median(ctx.O5[tr]))
    init = np.random.default_rng([S.SPLIT_SEED, split, seed]).choice(tr, N_INITIAL, replace=False)
    rng = np.random.default_rng([S.SPLIT_SEED, split, seed, POLICIES.index(policy)])
    acquired = np.array(init, int)
    theta = S.fit([ctx.item(i) for i in acquired], tau)
    rows = []
    t0 = time.time()
    for r in range(N_ROUNDS + 1):
        x2 = S.chi2_many(theta, held)
        red = np.log10(base) - np.log10(x2)
        rows.append(
            {
                "split": split,
                "seed": seed,
                "policy": policy,
                "round": r,
                "n_acquired": len(acquired),
                "improvement": float(np.median(red)),
                "relative_reduction_median_chi2": float(1 - np.median(x2) / np.median(base)),
                "mean_O5_acquired": float(ctx.O5[acquired].mean()),
                "mean_O5_chosen": float(ctx.O5[acquired[N_INITIAL:]].mean()) if r else np.nan,
                "acquired": " ".join(ctx.labels[acquired[-BATCH:]]) if r else " ".join(ctx.labels[acquired]),
                "seconds": time.time() - t0,
            }
        )
        if r == N_ROUNDS:
            break
        cand = np.setdiff1d(tr, acquired)
        batch = choose(policy, ctx, theta, acquired, cand, tau, rng, o5_median, value)
        acquired = np.concatenate([acquired, batch])
        theta = S.fit([ctx.item(i) for i in acquired], tau, theta0=theta)
    return rows


def pareto_snapshot(tau: float) -> pd.DataFrame:
    """Candidates' (O1, O4, O5) at the first round of split 0, seed set 0, with the fronts and the first
    batch of each multi-objective policy, for the figure."""
    ctx = context()
    tr, _ = S.splits(ctx.table)[0]
    init = np.random.default_rng([S.SPLIT_SEED, 0, 0]).choice(tr, N_INITIAL, replace=False)
    theta = S.fit([ctx.item(i) for i in init], tau)
    cand = np.setdiff1d(tr, init)
    ob = ctx.objectives(theta, init, cand, tau, need={"O1", "O4", "O5"})
    F = np.column_stack([ob["O1"], ob["O4"], ob["O5"]])
    df = pd.DataFrame(
        {
            "label": ctx.labels[cand],
            "O1": ob["O1"],
            "O4": ob["O4"],
            "O5": ob["O5"],
            "front": nondominated_fronts(F),
            "disorder_class": ctx.table["disorder_class"].to_numpy()[cand],
        }
    )
    rng = np.random.default_rng(0)
    for pol in ("pareto", "pareto constrained", "chebyshev 0.00", "chebyshev 1.00"):
        picked = choose(pol, ctx, theta, init, cand, tau, rng, float(np.median(ctx.O5[tr])), None)
        df[pol] = df["label"].isin(ctx.labels[picked])
    return df


# ---------------------------------------------------------------------------- summary
def auc(curve: pd.DataFrame) -> pd.Series:
    """Mean held-out improvement over 10–90 acquired (trapezoid area divided by the width)."""

    def one(g):
        g = g.sort_values("n_acquired")
        x, y = g["n_acquired"].to_numpy(float), g["improvement"].to_numpy()
        return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)) / (x[-1] - x[0]))

    return curve.groupby(["split", "seed", "policy"]).apply(one, include_groups=False)


def hierarchical_ci(diff: pd.Series, seed: int = 20261013, n_boot: int = N_BOOT) -> list[float]:
    """Bootstrap over splits, then seed sets within each drawn split; diff indexed by (split, seed)."""
    rng = np.random.default_rng(seed)
    by_split = [g.to_numpy() for _, g in diff.groupby(level="split")]
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(by_split), len(by_split))
        stats.append(np.mean([rng.choice(by_split[k], len(by_split[k])).mean() for k in pick]))
    return [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))]


def summarise(curves: pd.DataFrame, comp: pd.DataFrame, e5_met: bool) -> dict:
    A = auc(curves).unstack("policy")
    out = {
        "informative": bool(e5_met),
        "n_splits": int(curves["split"].nunique()),
        "n_seeds": int(curves["seed"].nunique()),
        "batch": BATCH,
        "rounds": N_ROUNDS,
        "policies": {},
    }
    for p in A.columns:
        d = A[p] - A["random"]
        rec = {
            "auc_mean": float(A[p].mean()),
            "auc_minus_random_mean": float(d.mean()),
            "auc_minus_random_ci95": hierarchical_ci(d) if p != "random" else [0.0, 0.0],
            "final_improvement_mean": float(
                curves[(curves["policy"] == p) & (curves["round"] == N_ROUNDS)]["improvement"].mean()
            ),
            "mean_O5_chosen": float(
                curves[(curves["policy"] == p) & (curves["round"] == N_ROUNDS)]["mean_O5_chosen"].mean()
            ),
        }
        rec.update(comp.loc[p].to_dict() if p in comp.index else {})
        out["policies"][p] = rec
    P = out["policies"]
    lo = {p: P[p]["auc_minus_random_ci95"][0] for p in P}
    out["E7"] = {
        "statement": "Pareto and O1-greedy batches beat random choice; the predicted-error policy does not",
        "pareto_above_random": lo["pareto"] > 0,
        "o1_above_random": lo["O1"] > 0,
        "predicted_error_above_random": lo["predicted error"] > 0,
        "met": bool(lo["pareto"] > 0 and lo["O1"] > 0 and not lo["predicted error"] > 0),
        "evaluable": bool(e5_met),
    }
    a_un, a_con = P["pareto"]["auc_mean"], P["pareto constrained"]["auc_mean"]
    cost = (a_un - a_con) / a_un if a_un > 0 else float("nan")
    out["E8"] = {
        "statement": "restricting to the more feasible half costs less than a quarter of the improvement",
        "auc_unconstrained": a_un,
        "auc_constrained": a_con,
        "cost_share": cost,
        "met": bool(np.isfinite(cost) and cost < 0.25),
        "evaluable": bool(e5_met and a_un > 0),
    }
    out["tradeoff"] = [
        {
            "policy": p,
            "auc": P[p]["auc_mean"],
            "mean_O5_chosen": P[p]["mean_O5_chosen"],
            "auc_minus_random_ci95": P[p]["auc_minus_random_ci95"],
        }
        for p in P
        if p.startswith("chebyshev") or p.startswith("pareto")
    ]
    return out


def composition(curves: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Class shares, median length and disorder of what each policy chose (seed sets excluded)."""
    t = table.set_index("label")
    rows = []
    for p, g in curves[curves["round"] > 0].groupby("policy"):
        labs = " ".join(g["acquired"]).split()
        sub = t.loc[labs]
        rec = {
            "policy": p,
            "median_length": float(sub["length"].median()),
            "median_disorder": float(sub["disorder_mean"].median()),
        }
        for c in ("folded", "partly disordered", "disordered"):
            rec[f"share_{c}"] = float((sub["disorder_class"] == c).mean())
        rows.append(rec)
    return pd.DataFrame(rows).set_index("policy")


def main(policies: list[str] | None = None) -> None:
    sur = json.load(open(C.RESULTS / "acq_surrogate.json"))
    tau, e5_met = sur["tau"], sur["E5"]["met"]
    policies = policies or list(POLICIES)
    vp = C.RESULTS / "acq_value.csv"
    value = None
    if "oracle" in policies:
        if not vp.exists():
            raise SystemExit("the oracle needs realised values: run `bsc value` first")
        v = pd.read_csv(vp)
        value = {k: g.set_index("label")["value"] for k, g in v.groupby("split")}
    t0 = time.time()
    jobs = [(k, s, p) for k in range(S.N_SPLITS) for s in range(N_SEEDS) for p in policies]
    res = Parallel(n_jobs=C.N_JOBS, verbose=0)(
        delayed(run)(k, s, p, tau, value[k] if (value is not None and p == "oracle") else None)
        for k, s, p in jobs
    )
    wall = time.time() - t0
    curves = pd.DataFrame([r for rr in res for r in rr])
    curves.to_csv(C.RESULTS / "acq_curves.csv", index=False)
    table = S.load_cache()["table"]
    comp = composition(curves, table)
    out = summarise(curves, comp, e5_met)
    out["n_fits"] = int(len(jobs) * (N_ROUNDS + 1))
    out["wall_seconds"] = wall
    out["cpu_seconds_per_run_median"] = float(
        curves.groupby(["split", "seed", "policy"])["seconds"].max().median()
    )
    out["n_jobs"] = C.N_JOBS
    json.dump(out, open(C.RESULTS / "acq_acquire.json", "w"), indent=1, default=float)
    pareto_snapshot(tau).to_csv(C.RESULTS / "acq_pareto_snapshot.csv", index=False)
    P = out["policies"]
    print(
        f"  {len(jobs)} runs, {out['n_fits']} fits in {wall / 60:.1f} min; "
        f"{'informative' if e5_met else 'UNINFORMATIVE (E5 not met)'}",
        flush=True,
    )
    for p in policies:
        print(
            f"    {p:20s} AUC {P[p]['auc_mean']:+.4f}  minus random {P[p]['auc_minus_random_mean']:+.4f} "
            f"[{P[p]['auc_minus_random_ci95'][0]:+.4f}, {P[p]['auc_minus_random_ci95'][1]:+.4f}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
