"""The acquisition extension (ACQUISITION_DESIGN §9): θ = 0 reproduces the raw χ², the analytic gradient
matches finite differences, the information objective finds a known optimal design, and no objective
reads a profile that has not been acquired."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bsc import acquire, saxs
from bsc import objectives as O
from bsc import surrogate as S


def sphere_curve(q, R):
    x = q * R
    return (3 * (np.sin(x) - x * np.cos(x)) / x**3) ** 2


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """Eight proteins laid out like the archive: measured profiles and per-conformer computed curves."""
    from bsc import config as C
    data, results = tmp_path / "data", tmp_path / "results"
    saxs_dir = data / "PeptoneDB-SAXS" / "sasbdb-clean_data"
    pred = data / "Predictions" / "PeptoneDB-SAXS-expt" / "bioemu"
    saxs_dir.mkdir(parents=True)
    pred.mkdir(parents=True)
    results.mkdir()
    rng = np.random.default_rng(1)
    rows = []
    for i in range(8):
        label = f"SASDY{i:02d}"
        q = np.linspace(0.012, 0.3, 90 + 10 * i)
        if i == 3:
            q[40] = q[39]                                   # a repeated q value, as in two archive entries
        R_true, R_model = 18 + 2 * i, 18 + 2 * i + (4 if i % 2 else -1)
        truth = np.mean([sphere_curve(q, r) for r in R_true + rng.normal(0, 1.5, 100)], axis=0)
        sigma = 0.02 * truth + 2e-4
        if i == 5:
            sigma[-3:] = 0.0                                # points without an error are not scored
        exp = truth + rng.normal(0, np.where(sigma > 0, sigma, 1e-4))
        pd.DataFrame({"q": q, "I": exp, "s": sigma}).to_csv(saxs_dir / f"{label}-bift.dat", sep="\t",
                                                            index=False, header=["# q", "I(q)", "sigma"])
        radii = R_model + rng.normal(0, 2.0, 40)
        curves = pd.DataFrame(np.array([sphere_curve(q, r) for r in radii]), columns=[f"{v:.5f}" for v in q])
        curves.to_csv(pred / f"Pepsi-{label}.csv")
        seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 40 + 15 * i))
        dis = 0.1 + 0.1 * i
        cls = "folded" if dis < 0.2 else "partly disordered" if dis < 0.6 else "disordered"
        rows.append({"label": label, "sequence": seq, "length": len(seq), "disorder_mean": dis,
                     "disorder_class": cls, "group": i})
    monkeypatch.setattr(C, "DATA", data)
    monkeypatch.setattr(C, "RESULTS", results)
    monkeypatch.setattr(C, "SAXS_DIR", saxs_dir)
    monkeypatch.setattr(C, "PRED_DIR", pred.parent)
    monkeypatch.setattr(C, "N_JOBS", 1)
    table = pd.DataFrame(rows)
    cache = S.build_cache(table, n_jobs=1)
    return {"cache": cache, "saxs_dir": saxs_dir, "table": table}


def test_theta_zero_reproduces_the_raw_chi2(synthetic):
    from bsc import score
    comps = synthetic["cache"]["computed"]
    for lab, c in comps.items():
        e = score.read_experiment(lab)
        keep = e["sigma"].to_numpy() > 0
        Y = score.read_prediction("bioemu", lab)
        ref = saxs.chi2(Y[:, keep].mean(axis=0), e["I"].to_numpy()[keep], e["sigma"].to_numpy()[keep])
        assert S.raw_chi2(c, S.read_experiment(c)) == pytest.approx(ref, rel=1e-10)


ARCHIVE_CACHE = Path(os.environ.get("BSC_ROOT", "/nonexistent")) / "data" / "acquisition_cache.pkl"


@pytest.mark.skipif(not ARCHIVE_CACHE.exists(),
                    reason="needs the archive (BSC_ROOT with the acquisition cache)")
def test_theta_zero_reproduces_the_scored_raw_chi2_of_every_archive_entry():
    from bsc import config as C
    cache = S.load_cache()
    comps, labels = cache["computed"], cache["table"]["label"].to_numpy()
    sc = pd.read_csv(C.RESULTS / "scores.csv")
    ref = sc[sc["model"] == C.MODEL].set_index("label").loc[labels, "chi2_raw"].to_numpy()
    got = np.array([S.raw_chi2(comps[lab], S.read_experiment(comps[lab])) for lab in labels])
    assert len(labels) == 399
    np.testing.assert_allclose(got, ref, rtol=1e-10)


def test_analytic_gradient_matches_finite_differences(synthetic):
    comps = synthetic["cache"]["computed"]
    rng = np.random.default_rng(3)
    items = [(c, S.read_experiment(c)) for c in comps.values()]
    for theta in (rng.normal(0, 0.05, S.N_THETA), rng.normal(0, 0.3, S.N_THETA)):
        _, g = S.objective(theta, items, tau=1.0)
        eps = 1e-6
        fd = np.array([(S.objective(theta + eps * np.eye(S.N_THETA)[k], items, 1.0)[0]
                        - S.objective(theta - eps * np.eye(S.N_THETA)[k], items, 1.0)[0]) / (2 * eps)
                       for k in range(S.N_THETA)])
        np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6 * np.abs(g).max())


def test_size_scaling_stays_in_bounds_and_scores_points_inside_the_grid(synthetic):
    c = next(iter(synthetic["cache"]["computed"].values()))
    for eta in (-5.0, -0.1, 0.0, 0.1, 5.0):
        ls, _ = S.log_scale(eta)
        assert np.log(0.8) - 1e-12 <= ls <= np.log(1.25) + 1e-12
        cv = S.curve(c, 0.0, eta, grad=False)
        u = np.log(c.q[cv["inside"]]) + ls
        full = cv["wq"] >= 1
        # weight 1 inside the grid, a taper of at most one grid spacing beyond it, nothing further out
        assert (u[full] >= np.log(c.q.min()) - 1e-12).all() and (u[full] <= np.log(c.q.max()) + 1e-12).all()
        h = np.diff(np.unique(np.log(c.q)))
        assert u.min() > np.log(c.q.min()) - h[0] and u.max() < np.log(c.q.max()) + h[-1]
        assert ((cv["wq"] > 0) & (cv["wq"] <= 1)).all()
    assert S.curve(c, 0.0, 0.0, grad=False)["inside"].all()


def test_fitting_recovers_a_known_size_scaling(synthetic):
    """Profiles computed from the model's own conformers shrunk by 10%: the fit finds s = 0.9."""
    comps = synthetic["cache"]["computed"]
    items = []
    for c in comps.values():
        full = S.curve(c, 0.0, S.LOG_S_BOUND * np.arctanh(np.log(0.9) / S.LOG_S_BOUND), grad=False)
        pts = np.flatnonzero(full["inside"])
        I = full["I"]
        items.append((c, S.Experiment(c.label, pts, I, 0.01 * I + 1e-6)))
        assert full["wq"].min() < 1                         # the lowest point sits in the taper
    theta = S.fit(items, tau=100.0)
    for c, _ in items:
        lam, eta = S.lam_eta(theta, c.x)
        assert np.exp(S.log_scale(eta)[0]) == pytest.approx(0.9, abs=0.01)


def test_information_objective_recovers_the_d_optimal_design():
    """A straight line in one feature: the D-optimal pair of design points sits at the two ends."""
    t = np.linspace(-1, 1, 11)
    a = np.array([[1.0, 0.0], [0.0, 0.0]])
    V = np.stack([O.info_factor(a, np.array([1.0, ti, 0.0, 0.0])) for ti in t])
    M = 1e-3 * np.eye(S.N_THETA)
    cand = np.arange(len(t))
    picked = acquire.fill_o1(V, M, cand, groups=cand, blocked=set(), k=2)
    assert sorted(t[picked]) == [-1.0, 1.0]
    # the greedy pair is the exhaustive optimum of the batch log det
    best = max(((i, j) for i in cand for j in cand if i < j),
               key=lambda ij: np.linalg.slogdet(M + V[ij[0]] @ V[ij[0]].T + V[ij[1]] @ V[ij[1]].T)[1])
    assert sorted(picked) == sorted(best)
    # and the single-candidate gain is smallest at the centre, where the slope is not informed
    g = O.logdet_gain(V, np.linalg.inv(M + V[0] @ V[0].T))
    assert np.argmax(g) == len(t) - 1


def test_objectives_are_identical_when_hidden_profiles_are_deleted(synthetic):
    cache, saxs_dir, table = synthetic["cache"], synthetic["saxs_dir"], synthetic["table"]
    static = O.static_objectives(table)
    acquired, cand = np.array([0, 2, 5]), np.array([1, 3, 4, 6, 7])
    theta = np.linspace(-0.05, 0.05, S.N_THETA)

    def compute():
        ctx = O.Context(cache, static)
        return ctx.objectives(theta, acquired, cand, tau=1.0)

    before = compute()
    hidden = [lab for i, lab in enumerate(table["label"]) if i not in acquired]
    for lab in hidden:
        (saxs_dir / f"{lab}-bift.dat").unlink()
    after = compute()
    for k in (*O.OBJECTIVES, "M"):
        np.testing.assert_array_equal(before[k], after[k])
    with pytest.raises(FileNotFoundError):
        S.read_experiment(cache["computed"][hidden[0]])


def test_pareto_tools():
    F = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]], float)
    assert acquire.nondominated_fronts(F).tolist() == [0, 0, 0, 0, 1]
    assert acquire.hypervolume(np.array([[1.0, 1.0, 1.0]])) == pytest.approx(1.0)
    assert acquire.hypervolume(np.array([[1.0, 0.5, 1.0], [0.5, 1.0, 1.0]])) == pytest.approx(0.75)
    assert acquire.hypervolume(np.array([[1.0, 1.0, 0.5], [0.5, 0.5, 1.0]])) == pytest.approx(0.625)


def test_feasibility_score_is_fixed_and_bounded():
    f = O.feasibility("MKT" * 40, 120, 0.1)
    assert 0 <= f["O5"] <= 1
    assert O.feasibility("C" * 20 + "A" * 100, 120, 0.9)["O5"] < f["O5"]
    assert O.isoelectric_point("DDDDEEEE" + "G" * 20) < 4.5 and O.isoelectric_point("KKKKRRRR") > 10


def test_hierarchical_interval_brackets_a_constant_difference():
    idx = pd.MultiIndex.from_product([range(5), range(10)], names=["split", "seed"])
    d = pd.Series(0.3, index=idx)
    assert acquire.hierarchical_ci(d, n_boot=50) == pytest.approx([0.3, 0.3])
    _ = json.dumps(acquire.POLICIES)


def test_score_is_continuous_as_points_leave_the_grid(synthetic):
    """The taper keeps χ² continuous in the size scaling (a hard cut would jump at every grid point)."""
    comps = synthetic["cache"]["computed"]
    c = comps["SASDY02"]
    e = S.read_experiment(c)
    etas = np.linspace(0.0, 0.05, 2001)
    x2 = np.array([S.chi2_terms(c, e, np.r_[np.zeros(4), eta / c.x[0], 0, 0, 0], grad=False)[0]
                   for eta in etas])
    steps = np.abs(np.diff(x2))
    assert steps.max() < 20 * np.median(steps) + 1e-12
