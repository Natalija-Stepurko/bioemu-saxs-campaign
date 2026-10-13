"""Stage `objectives` (ACQUISITION_DESIGN §4): quantities computable before a profile is measured, used to
choose which profiles to acquire.

    O1  information about the shared model: profiled Fisher information of θ (scale and background
        projected out), with the predicted noise; batch value
        log det(Λ0 + J_acq + Σ_B J_p) − log det(Λ0 + J_acq)
    O2  information about the protein's own ensemble: mean reduced χ² between the ensemble average and its
        larger-half and smaller-half sub-ensembles, under the predicted noise
    O3  model disagreement: SD of ln Rg across six generators' ensemble curves
    O4  coverage: 1 − 5-mer containment to the nearest acquired protein
    O5  feasibility (heuristic, fixed, not validated): from length, disorder score, cysteines, hydrophobic
        fraction and the isoelectric point's distance from pH 7.4
    baseline: predicted raw error (ridge regression of log10 raw χ² on the `predict` features), refitted on
        the acquired profiles only

No objective reads a profile that has not been acquired: the noise of an unmeasured profile comes from a
model of the error relative to the low-angle intensity against q and molecular weight, fitted on the
acquired profiles only, and the predicted-error baseline is fitted on them too.

This stage writes the static objectives (O3, O5, the features) and two checks: whether O5 separates the
deposited profiles that failed the quality flags, and how well the noise model predicts held-out errors.
Output: results/acq_static.csv, results/acq_objectives.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from bsc import analyse, features, predict, robust, score
from bsc import config as C
from bsc import surrogate as S

O3_MODELS = ("bioemu", "peptron", "idpfold", "boltz2", "alphafold", "esmfold")
OBJECTIVES = ("O1", "O2", "O3", "O4", "O5", "predicted error")
N_LOW = 5  # points that define the low-angle reference intensity
NOISE_POINTS = 60  # points per acquired profile in the noise regression (equal weight)

# isoelectric point: pKa values of EMBOSS
PKA_POS = {"Nterm": 8.6, "K": 10.8, "R": 12.5, "H": 6.5}
PKA_NEG = {"Cterm": 3.6, "D": 3.9, "E": 4.1, "C": 8.5, "Y": 10.1}


# ---------------------------------------------------------------------------- O5, feasibility
def isoelectric_point(seq: str) -> float:
    s = seq.upper()
    n = {a: s.count(a) for a in "KRHDECY"}

    def charge(ph):
        pos = 1 / (1 + 10 ** (ph - PKA_POS["Nterm"])) + sum(
            n[a] / (1 + 10 ** (ph - PKA_POS[a])) for a in "KRH"
        )
        neg = 1 / (1 + 10 ** (PKA_NEG["Cterm"] - ph)) + sum(
            n[a] / (1 + 10 ** (PKA_NEG[a] - ph)) for a in "DECY"
        )
        return pos - neg

    lo, hi = 0.0, 14.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if charge(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def feasibility(seq: str, length: int, disorder: float) -> dict:
    """O5: one minus the mean of five penalties in [0, 1], fixed before any acquisition result:
    length outside 50–300 residues, the disorder score, cysteine count (saturating at six), hydrophobic
    fraction above 0.40 (saturating at 0.55) and an isoelectric point within 1.5 units of pH 7.4."""
    s = seq.upper()
    n_cys = s.count("C")
    hyd = sum(s.count(a) for a in "AVILMFWY") / max(1, len(s))
    pi = isoelectric_point(s)
    pen = {
        "length": (50 - length) / 50
        if length < 50
        else min(1.0, (length - 300) / 300)
        if length > 300
        else 0.0,
        "disorder": float(np.clip(disorder, 0, 1)),
        "cysteine": min(1.0, n_cys / 6),
        "hydrophobic": float(np.clip((hyd - 0.40) / 0.15, 0, 1)),
        "pI": float(np.clip(1 - abs(pi - 7.4) / 1.5, 0, 1)),
    }
    return {
        "O5": 1 - float(np.mean(list(pen.values()))),
        "pI": pi,
        "n_cys": n_cys,
        "frac_hydrophobic": hyd,
        **{f"pen_{k}": v for k, v in pen.items()},
    }


# ---------------------------------------------------------------------------- O4, coverage
def containment_matrix(seqs: list[str]) -> np.ndarray:
    sets = [robust.kmer_set(s) for s in seqs]
    n = len(sets)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = robust.containment(sets[i], sets[j])
    return M


def coverage(cont: np.ndarray, cand: np.ndarray, acquired: np.ndarray) -> np.ndarray:
    """O4: 1 − containment to the nearest acquired protein (1 when nothing is acquired)."""
    if len(acquired) == 0:
        return np.ones(len(cand))
    return 1 - cont[np.ix_(cand, acquired)].max(axis=1)


# ---------------------------------------------------------------------------- the noise model
def low_angle_reference(I: np.ndarray) -> float:
    v = I[:N_LOW]
    v = v[v > 0]
    return float(v.mean()) if len(v) else float("nan")


class NoiseModel:
    """log10(σ / I_low) = β · (1, log10 q, (log10 q)², log10 mass), fitted on acquired profiles only."""

    def __init__(self):
        self.beta = None

    @staticmethod
    def design(q: np.ndarray, mass_kda: float) -> np.ndarray:
        lq = np.log10(q)
        return np.column_stack([np.ones_like(lq), lq, lq**2, np.full_like(lq, np.log10(mass_kda))])

    def fit(self, rows: list[tuple[S.Computed, S.Experiment, float]]) -> NoiseModel:
        X, y = [], []
        for c, e, mass in rows:
            ref = low_angle_reference(e.I)
            if not np.isfinite(ref) or ref <= 0:
                continue
            take = np.unique(np.linspace(0, len(e.pts) - 1, min(NOISE_POINTS, len(e.pts))).astype(int))
            X.append(self.design(c.q[e.pts][take], mass))
            y.append(np.log10(e.sigma[take] / ref))
        X, y = np.vstack(X), np.concatenate(y)
        self.beta = np.linalg.lstsq(X, y, rcond=None)[0]
        self.rmse_ = float(np.sqrt(np.mean((X @ self.beta - y) ** 2)))
        return self

    def relative(self, q: np.ndarray, mass_kda: float) -> np.ndarray:
        return 10 ** (self.design(q, mass_kda) @ self.beta)


# ---------------------------------------------------------------------------- O1 and O2
def _project_out(G: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Columns of G minus their least-squares projection onto the columns of Q."""
    coef = np.linalg.lstsq(Q, G, rcond=None)[0]
    return G - Q @ coef


def info_factor(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """V with V Vᵀ = kron(a, x xᵀ), the per-protein information in θ, for a 2×2 a."""
    ev, U = np.linalg.eigh(0.5 * (a + a.T))
    L = U * np.sqrt(np.clip(ev, 0, None))
    return np.kron(L, x[:, None])  # (8, 2)


def candidate_terms(c: S.Computed, theta: np.ndarray, noise: NoiseModel, mass_kda: float) -> dict:
    """O1's information factor and O2 for a protein not yet measured, under the predicted noise."""
    lam, eta = S.lam_eta(theta, c.x)
    cv = S.curve(c, lam, eta)
    q = c.q[cv["inside"]]
    I = cv["I"]
    sig = low_angle_reference(I) * noise.relative(q, mass_kda) / np.sqrt(cv["wq"])
    G = np.column_stack([cv["dI_dlam"], cv["dI_deta"]]) / sig[:, None]
    Q = np.column_stack([I / sig, 1 / sig])
    Gt = _project_out(G, Q)
    dof = cv["wq"].sum() - 2
    a = Gt.T @ Gt / dof
    # O2: the uniform ensemble against its larger and smaller halves (raw ensemble, all computed points)
    big = c.z > np.median(c.z)
    Ib = S.curve(c, 0.0, 0.0, grad=False, weights_override=big / big.sum())["I"]
    Is = S.curve(c, 0.0, 0.0, grad=False, weights_override=~big / (~big).sum())["I"]
    Iu = S.curve(c, 0.0, 0.0, grad=False)["I"]
    sig0 = low_angle_reference(Iu) * noise.relative(c.q, mass_kda)
    o2 = []
    for half in (Ib, Is):
        A, e, y = half / sig0, 1 / sig0, Iu / sig0
        sc, bg = S._fit_scale(A, e, y)
        r = sc * A + bg * e - y
        o2.append(r @ r / (len(r) - 2))
    return {"V": info_factor(a, c.x), "O2": float(np.mean(o2))}


def acquired_factor(c: S.Computed, e: S.Experiment, theta: np.ndarray) -> np.ndarray:
    """The information factor of an acquired profile, with its measured errors and fitted scale."""
    lam, eta = S.lam_eta(theta, c.x)
    cv = S.curve(c, lam, eta, e.pts)
    m = cv["inside"]
    sig = e.sigma[m] / np.sqrt(cv["wq"])
    A, ones, y = cv["I"] / sig, 1 / sig, e.I[m] / sig
    sc, _ = S._fit_scale(A, ones, y)
    G = sc * np.column_stack([cv["dI_dlam"], cv["dI_deta"]]) / sig[:, None]
    Gt = _project_out(G, np.column_stack([A, ones]))
    a = Gt.T @ Gt / (cv["wq"].sum() - 2)
    return info_factor(a, c.x)


def logdet_gain(V: np.ndarray, Minv: np.ndarray) -> np.ndarray:
    """log det(M + V_p V_pᵀ) − log det(M) for each candidate's factor V_p, stacked (n, 8, 2)."""
    K = np.einsum("nia,ij,njb->nab", V, Minv, V)
    K = K + np.eye(V.shape[2])[None]
    return np.linalg.slogdet(K)[1]


# ---------------------------------------------------------------------------- predicted error
def feature_table(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The `predict` stage's features, from the computed curves and the sequence alone."""
    d = table.assign(
        model=C.MODEL, clean=True, chi2_raw=np.nan, phi_at_chi2_1=np.nan, phi_at_chi2_2=np.nan, kind=None
    )
    X = features.build(d, table[["label", "sequence"]])
    return X, features.feature_columns(X)


def predicted_error(
    X: np.ndarray, acquired: np.ndarray, y_acq: np.ndarray, cand: np.ndarray, seed: int = C.SEED
) -> np.ndarray:
    mdl = predict.regressors(seed)["ridge"].fit(X[acquired], y_acq)
    return mdl.predict(X[cand])


# ---------------------------------------------------------------------------- the state of a round
class Context:
    """Everything the objectives need that does not depend on the round: the cache, static objectives,
    the containment matrix and the feature table, indexed like the pool table."""

    def __init__(self, cache: dict, static: pd.DataFrame | None = None):
        self.table = cache["table"]
        self.comps = cache["computed"]
        self.labels = self.table["label"].to_numpy()
        self.exps = S.Experiments(self.comps)
        self.mass = np.array([score.sequence_mass_kda(s) for s in self.table["sequence"]])
        self.cont = containment_matrix(self.table["sequence"].tolist())
        self.groups = self.table["group"].to_numpy()
        static = static if static is not None else static_objectives(self.table)
        static = static.set_index("label").loc[self.labels]
        self.O3 = static["O3"].to_numpy()
        self.O5 = static["O5"].to_numpy()
        ft, cols = feature_table(self.table)
        self.X = ft.set_index("label").loc[self.labels, cols].to_numpy(float)
        self._raw = {}

    def item(self, i: int) -> tuple[S.Computed, S.Experiment]:
        lab = self.labels[i]
        return self.comps[lab], self.exps[lab]

    def raw_chi2(self, i: int) -> float:
        if i not in self._raw:
            self._raw[i] = S.raw_chi2(*self.item(i))
        return self._raw[i]

    def objectives(
        self, theta: np.ndarray, acquired: np.ndarray, cand: np.ndarray, tau: float, need=OBJECTIVES
    ) -> dict:
        """Every objective for the candidates, given θ̂ and the acquired set (reads acquired profiles only)."""
        out = {"cand": cand}
        acq_rows = [(*self.item(i), self.mass[i]) for i in acquired]
        if "O1" in need or "O2" in need:
            noise = NoiseModel().fit(acq_rows)
            terms = [candidate_terms(self.comps[self.labels[i]], theta, noise, self.mass[i]) for i in cand]
            V = np.stack([t["V"] for t in terms])
            M = np.eye(S.N_THETA) / tau**2
            for c, e, _ in acq_rows:
                Va = acquired_factor(c, e, theta)
                M = M + Va @ Va.T
            out["V"], out["M"] = V, M
            out["O1"] = logdet_gain(V, np.linalg.inv(M))
            out["O2"] = np.array([t["O2"] for t in terms])
        out["O3"] = self.O3[cand]
        out["O4"] = coverage(self.cont, cand, acquired)
        out["O5"] = self.O5[cand]
        if "predicted error" in need:
            y = np.log10([self.raw_chi2(i) for i in acquired])
            out["predicted error"] = predicted_error(self.X, acquired, y, cand)
        return out


# ---------------------------------------------------------------------------- the stage
def static_objectives(table: pd.DataFrame, n_jobs: int = 1) -> pd.DataFrame:
    rg = Parallel(n_jobs=n_jobs)(delayed(S.model_rg)(lab, m) for lab in table["label"] for m in O3_MODELS)
    rg = np.array(rg).reshape(len(table), len(O3_MODELS))
    rows = []
    for i, r in table.reset_index(drop=True).iterrows():
        f = feasibility(r["sequence"], int(r["length"]), float(r["disorder_mean"]))
        lr = np.log(rg[i][np.isfinite(rg[i])])
        rows.append(
            {
                "label": r["label"],
                "O3": float(np.std(lr, ddof=1)) if len(lr) >= 3 else np.nan,
                "n_models_O3": int(len(lr)),
                **f,
                **{f"rg_{m}": rg[i][k] for k, m in enumerate(O3_MODELS)},
            }
        )
    return pd.DataFrame(rows)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    u = stats.mannwhitneyu(pos, neg).statistic
    return float(u / (len(pos) * len(neg)))


def feasibility_check(ent: pd.DataFrame) -> dict:
    """Does O5 separate the deposited profiles that failed the quality flags from the rest? Among the
    entries with a BioEmu-1 ensemble (the denominator of the first study)."""
    sc = pd.read_csv(C.RESULTS / "scores.csv")
    e = ent[ent["label"].isin(sc.loc[sc["model"] == C.MODEL, "label"])].copy()
    flagged = (
        e["flag_aggregation"]
        | e["flag_negative_I"]
        | e["flag_bad_errors"]
        | e["flag_oligomer"]
        | ~e["guinier_valid"]
    )
    e["failed"] = flagged.to_numpy()
    o5 = np.array(
        [
            feasibility(s, int(n), float(dm))["O5"]
            for s, n, dm in zip(e["sequence"], e["length"], e["disorder_mean"], strict=True)
        ]
    )
    fail, ok = o5[e["failed"].to_numpy()], o5[~e["failed"].to_numpy()]
    a = auc(ok, fail)  # probability that a passing profile scores higher than a failed one
    rng = np.random.default_rng(analyse.BOOT_SEED)
    boot = [auc(rng.choice(ok, len(ok)), rng.choice(fail, len(fail))) for _ in range(analyse.N_BOOT)]
    return {
        "n_failed": int(len(fail)),
        "n_passed": int(len(ok)),
        "auc_pass_above_fail": a,
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "median_O5_failed": float(np.median(fail)),
        "median_O5_passed": float(np.median(ok)),
        "p_two_sided": float(stats.mannwhitneyu(ok, fail).pvalue),
    }


def noise_check(ctx: Context, seed: int = S.SPLIT_SEED) -> dict:
    """Fit the noise model on each split's pool and predict the errors of its held-out profiles."""
    out = []
    for k, (tr, te) in enumerate(S.splits(ctx.table)):
        nm = NoiseModel().fit([(*ctx.item(i), ctx.mass[i]) for i in tr])
        err = []
        for i in te:
            c, e = ctx.item(i)
            pred = low_angle_reference(e.I) * nm.relative(c.q[e.pts], ctx.mass[i])
            err.append(np.median(np.abs(np.log10(pred) - np.log10(e.sigma))))
        out.append(
            {
                "split": k,
                "beta": nm.beta.tolist(),
                "train_rmse_log10": nm.rmse_,
                "held_out_median_abs_log10_error": float(np.median(err)),
            }
        )
    return {
        "per_split": out,
        "median_abs_log10_error": float(np.median([o["held_out_median_abs_log10_error"] for o in out])),
    }


def main() -> None:
    cache = S.load_cache(rebuild=True)
    table = cache["table"]
    static = static_objectives(table, n_jobs=C.N_JOBS)
    static.to_csv(C.RESULTS / "acq_static.csv", index=False)
    ent = pd.read_csv(C.RESULTS / "entries.csv")
    ctx = Context(cache, static)
    feas = feasibility_check(ent)
    noise = noise_check(ctx)
    out = {
        "objectives": list(OBJECTIVES),
        "o3_models": list(O3_MODELS),
        "n_pool": int(len(table)),
        "feasibility_check": feas,
        "noise_model": noise,
        "O5_median": float(static["O5"].median()),
        "spearman_static": {
            f"{a}~{b}": float(stats.spearmanr(static[a], static[b], nan_policy="omit").statistic)
            for a, b in (("O3", "O5"),)
        },
    }
    json.dump(out, open(C.RESULTS / "acq_objectives.json", "w"), indent=1)
    print(
        f"  {len(table)} pool proteins; O5 ranks passing above failed profiles with AUC "
        f"{feas['auc_pass_above_fail']:.2f} ({feas['ci95'][0]:.2f}–{feas['ci95'][1]:.2f}; "
        f"{feas['n_failed']} failed); "
        f"noise model held-out median |log10 error| {noise['median_abs_log10_error']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
