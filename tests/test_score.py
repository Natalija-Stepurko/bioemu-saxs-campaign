"""The score stage on a synthetic archive laid out like the PeptoneBench predictions."""
import json

import numpy as np
import pandas as pd
import pytest


def sphere_curve(q, R):
    x = q * R
    return (3 * (np.sin(x) - x * np.cos(x)) / x**3) ** 2


@pytest.fixture
def synthetic_root(tmp_path, monkeypatch):
    from bsc import config as C
    data, results = tmp_path / "data", tmp_path / "results"
    saxs_dir = data / "PeptoneDB-SAXS" / "sasbdb-clean_data"
    pred = data / "Predictions" / "PeptoneDB-SAXS" / "bioemu"
    saxs_dir.mkdir(parents=True)
    pred.mkdir(parents=True)
    rng = np.random.default_rng(0)
    q = np.linspace(0.01, 0.3, 120)
    rows = []
    for i, (label, R_true, R_model, dis) in enumerate(
            [("SASDX01", 20.0, 20.0, 0.1), ("SASDX02", 30.0, 22.0, 0.7), ("SASDX03", 25.0, 25.0, 0.4)]):
        # the "experiment" is itself an ensemble average, as a real solution measurement is
        truth = np.mean([sphere_curve(q, r) for r in R_true + rng.normal(0, 1.5, 200)], axis=0)
        sigma = 0.02 * truth + 1e-4
        exp = truth + rng.normal(0, sigma)
        pd.DataFrame({"q": q, "I": exp, "s": sigma}).to_csv(saxs_dir / f"{label}-bift.dat", sep="\t",
                                                                index=False, header=["# q", "I(q)", "sigma"])
        radii = R_model + rng.normal(0, 1.5, 60)
        curves = pd.DataFrame(np.array([sphere_curve(q, r) for r in radii]), columns=[f"{v:.5f}" for v in q])
        curves.to_csv(pred / f"Pepsi-{label}.csv")
        rows.append({"label": label, "sequence": "A" * (50 + 30 * i), "length": 50 + 30 * i, "pH": 7.0,
                     "mean_gscore_adopt2": dis, "gscores_adopt2": "[]"})
    pd.DataFrame(rows).to_csv(data / "PeptoneDB-SAXS" / "PeptoneDB-SAXS.csv", index=False)
    monkeypatch.setattr(C, "DATA", data)
    monkeypatch.setattr(C, "RESULTS", results)
    monkeypatch.setattr(C, "SAXS_TABLE", data / "PeptoneDB-SAXS" / "PeptoneDB-SAXS.csv")
    monkeypatch.setattr(C, "SAXS_DIR", saxs_dir)
    monkeypatch.setattr(C, "PRED_DIR", pred.parent)
    monkeypatch.setattr(C, "N_JOBS", 1)
    monkeypatch.setattr(C, "THETA_GRID", 10.0 ** np.linspace(-1, 5, 9))
    return tmp_path


def test_score_stage_end_to_end(synthetic_root):
    from bsc import score
    score.main(["bioemu"])
    ent = pd.read_csv(synthetic_root / "results" / "entries.csv")
    sc = pd.read_csv(synthetic_root / "results" / "scores.csv")
    assert list(ent["disorder_class"]) == ["folded", "disordered", "partly disordered"]
    assert ent["guinier_valid"].all()
    s = sc.set_index("label")
    # the correct model fits; the mis-sized one does not, and reweighting cannot rescue it fully
    assert s.loc["SASDX01", "chi2_raw"] < 3
    assert s.loc["SASDX02", "chi2_raw"] > 20
    assert s.loc["SASDX02", "chi2_best"] > s.loc["SASDX01", "chi2_best"]
    assert (s["phi_at_best"] <= 1.0).all() and (s["n_conformers"] == 60).all()
    paths = json.load(open(synthetic_root / "results" / "paths" / "bioemu.json"))
    assert {p["label"] for p in paths} == set(s.index)
    # the ensemble Rg tracks the model radius, and the experimental Rg the true one
    assert abs(s.loc["SASDX02", "rg_model"] - 22 * np.sqrt(3 / 5)) < 1.5
    assert abs(ent.set_index("label").loc["SASDX02", "rg_exp"] - 30 * np.sqrt(3 / 5)) < 1.5
