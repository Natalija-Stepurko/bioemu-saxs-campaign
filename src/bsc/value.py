"""Stage `value` (ACQUISITION_DESIGN §6, E6, E9): the realised value of each pool protein and whether the
objectives, computed before the measurement, predict it.

Realised value: for each split, 20 random permutations of the pool; the first 10 and the first 30 proteins
of each form a subset S. For every pool protein p outside S, the gain is the held-out improvement (median
log10 χ² reduction against θ = 0) after refitting θ̂ on S ∪ {p}, minus that after fitting on S. The value of
p is its mean gain over the subsets it is not part of (Monte-Carlo marginal contributions).

Validity: at each subset S every objective is computed for every candidate as it would be in a round
(θ̂ fitted on S, the noise model and the predicted-error model fitted on S's profiles only, coverage to S),
converted to a rank within S, and averaged per protein. The Spearman correlation between that mean rank and
the realised value is computed per split and averaged over the splits; 95% intervals come from a bootstrap
over sequence groups (2,000 resamples).

Output: results/acq_value.csv (split × protein: value and mean objective ranks), results/acq_value_gains.csv
(every marginal contribution), results/acq_value.json.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from bsc import analyse
from bsc import config as C
from bsc import objectives as O
from bsc import surrogate as S
from bsc.acquire import context, ranks01

N_PERMUTATIONS = 20
SUBSET_SIZES = (10, 30)
N_BOOT = 2000


def marginal(split: int, perm: int, size: int, tau: float) -> list[dict]:
    ctx = context()
    tr, te = S.splits(ctx.table)[split]
    held = [ctx.item(i) for i in te]
    base = np.array([ctx.raw_chi2(i) for i in te])
    order = np.random.default_rng([S.SPLIT_SEED, 99, split, perm]).permutation(tr)
    sub = order[:size]
    items = [ctx.item(i) for i in sub]
    theta_s = S.fit(items, tau)
    score_s = S.improvement(theta_s, held, base)
    cand = np.setdiff1d(tr, sub)
    ob = ctx.objectives(theta_s, sub, cand, tau)
    rows = []
    for j, p in enumerate(cand):
        th = S.fit(items + [ctx.item(p)], tau, theta0=theta_s)
        rows.append(
            {
                "split": split,
                "perm": perm,
                "size": size,
                "label": ctx.labels[p],
                "gain": S.improvement(th, held, base) - score_s,
                **{f"rank_{k}": float(ranks01(ob[k])[j]) for k in O.OBJECTIVES},
                **{f"raw_{k}": float(ob[k][j]) for k in O.OBJECTIVES},
            }
        )
    return rows


def spearman_by_split(v: pd.DataFrame, obj: str) -> float:
    return float(
        np.mean([stats.spearmanr(g[f"rank_{obj}"], g["value"]).statistic for _, g in v.groupby("split")])
    )


def grouped_bootstrap(v: pd.DataFrame, fn, seed: int = analyse.BOOT_SEED, n_boot: int = N_BOOT) -> np.ndarray:
    """fn evaluated on resamples of whole sequence groups (the same groups drawn for every split)."""
    rng = np.random.default_rng(seed)
    groups = v["group"].unique()
    idx = {g: np.flatnonzero(v["group"].to_numpy() == g) for g in groups}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(groups, len(groups))
        rows = np.concatenate([idx[g] for g in pick])
        out.append(fn(v.iloc[rows]))
    return np.array(out)


def main() -> None:
    sur = json.load(open(C.RESULTS / "acq_surrogate.json"))
    tau = sur["tau"]
    t0 = time.time()
    jobs = [(k, p, n) for k in range(S.N_SPLITS) for p in range(N_PERMUTATIONS) for n in SUBSET_SIZES]
    res = Parallel(n_jobs=C.N_JOBS)(delayed(marginal)(k, p, n, tau) for k, p, n in jobs)
    wall = time.time() - t0
    gains = pd.DataFrame([r for rr in res for r in rr])
    gains.to_csv(C.RESULTS / "acq_value_gains.csv", index=False)
    table = S.load_cache()["table"].set_index("label")
    rank_cols = [f"rank_{k}" for k in O.OBJECTIVES]
    v = (
        gains.groupby(["split", "label"])
        .agg(value=("gain", "mean"), n_subsets=("gain", "size"), **{c: (c, "mean") for c in rank_cols})
        .reset_index()
    )
    v["group"] = table.loc[v["label"], "group"].to_numpy()
    v["disorder_class"] = table.loc[v["label"], "disorder_class"].to_numpy()
    v["length"] = table.loc[v["label"], "length"].to_numpy()
    v.to_csv(C.RESULTS / "acq_value.csv", index=False)

    validity = {}
    for k in O.OBJECTIVES:
        boot = grouped_bootstrap(v, lambda d, k=k: spearman_by_split(d, k))
        validity[k] = {
            "spearman": spearman_by_split(v, k),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        }
    diff = grouped_bootstrap(
        v, lambda d: spearman_by_split(d, "O1") - spearman_by_split(d, "O2"), seed=analyse.BOOT_SEED + 1
    )
    e6 = {
        "statement": "O1 correlates positively with realised value (interval excluding 0), "
        "more strongly than O2",
        "rho_O1": validity["O1"]["spearman"],
        "ci95_O1": validity["O1"]["ci95"],
        "rho_O2": validity["O2"]["spearman"],
        "ci95_O2": validity["O2"]["ci95"],
        "rho_O1_minus_O2": validity["O1"]["spearman"] - validity["O2"]["spearman"],
        "ci95_O1_minus_O2": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
        "met": bool(
            validity["O1"]["ci95"][0] > 0 and validity["O1"]["spearman"] > validity["O2"]["spearman"]
        ),
    }
    # E9: value per protein averaged over the splits in which it is in the pool
    per = v.groupby("label").agg(
        value=("value", "mean"), disorder_class=("disorder_class", "first"), group=("group", "first")
    )
    by_cls = {}
    for i, c in enumerate(analyse.CLASS_ORDER):
        x = per.loc[per["disorder_class"] == c, "value"].to_numpy()
        by_cls[c] = {
            "n": int(len(x)),
            "median": float(np.median(x)),
            "ci95": analyse.ci(x, seed_offset=900 + i),
            "share_positive": float(np.mean(x > 0)),
        }
    dis = per.loc[per["disorder_class"] == "disordered", "value"]
    fol = per.loc[per["disorder_class"] == "folded", "value"]
    u = stats.mannwhitneyu(dis, fol, alternative="two-sided")
    e9 = {
        "statement": "realised value is highest for disordered proteins (median, disordered above folded)",
        "median_disordered": by_cls["disordered"]["median"],
        "median_folded": by_cls["folded"]["median"],
        "p_two_sided_mannwhitney": float(u.pvalue),
        "met": bool(by_cls["disordered"]["median"] > by_cls["folded"]["median"]),
    }
    top = per.sort_values("value", ascending=False).head(max(1, len(per) // 10))
    comp_top = {
        "n": int(len(top)),
        **{f"share_{c}": float((top["disorder_class"] == c).mean()) for c in analyse.CLASS_ORDER},
        "median_length": float(table.loc[top.index, "length"].median()),
    }
    sizes = {
        str(n): {
            "median_gain": float(gains.loc[gains["size"] == n, "gain"].median()),
            "share_positive": float((gains.loc[gains["size"] == n, "gain"] > 0).mean()),
            "sd_gain": float(gains.loc[gains["size"] == n, "gain"].std()),
        }
        for n in SUBSET_SIZES
    }
    out = {
        "n_permutations": N_PERMUTATIONS,
        "subset_sizes": list(SUBSET_SIZES),
        "n_fits": int(len(gains) + len(jobs)),
        "wall_seconds": wall,
        "n_jobs": C.N_JOBS,
        "n_proteins": int(per.shape[0]),
        "value_median": float(per["value"].median()),
        "value_iqr": [float(per["value"].quantile(0.25)), float(per["value"].quantile(0.75))],
        "share_value_positive": float((per["value"] > 0).mean()),
        "gain_by_subset_size": sizes,
        "validity": validity,
        "E6": e6,
        "value_by_class": by_cls,
        "E9": e9,
        "top_decile_composition": comp_top,
    }
    json.dump(out, open(C.RESULTS / "acq_value.json", "w"), indent=1)
    print(
        f"  {len(gains)} marginal contributions in {wall / 60:.1f} min; Spearman with realised value: "
        + ", ".join(
            f"{k} {r['spearman']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]"
            for k, r in validity.items()
        )
        + f"; E6 {'met' if e6['met'] else 'not met'}; E9 {'met' if e9['met'] else 'not met'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
