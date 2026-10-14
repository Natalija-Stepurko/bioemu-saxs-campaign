"""The coverage stage: structure groups, intervals and the stratified test, on synthetic data."""
import numpy as np
import pandas as pd
from scipy import stats

from bsc import coverage


def test_structure_groups_from_the_sifts_flags():
    mp = pd.DataFrame({"map_method": ["SASBDB", "SASBDB", "SASBDB", "unmapped"],
                       "has_pdb_50": [True, False, "False", np.nan],
                       "has_pdb_any_overlap": [True, True, False, np.nan]})
    g = coverage.structure_group(mp)
    assert g.iloc[:3].tolist() == ["pdb50", "partial", "none"] and pd.isna(g.iloc[3])


def test_wilson_interval_matches_known_values_and_stays_in_range():
    lo, hi = coverage.wilson(118, 290)
    assert abs(lo - 0.352) < 0.002 and abs(hi - 0.464) < 0.002
    assert coverage.wilson(0, 11)[0] == 0.0 and 0 < coverage.wilson(0, 11)[1] < 0.3


def test_van_elteren_with_one_stratum_is_the_rank_sum_test():
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(0, 1, 40), rng.normal(0.6, 1, 50)]
    grp = np.r_[np.ones(40, bool), np.zeros(50, bool)]
    ve = coverage.van_elteren(x, grp, np.zeros(90))
    mw = stats.mannwhitneyu(x[grp], x[~grp], alternative="two-sided", method="asymptotic",
                            use_continuity=False)
    assert abs(ve["p_two_sided"] - mw.pvalue) < 1e-9


def test_a_confounded_pooled_difference_vanishes_when_stratified():
    """Structure availability tracks the class, the class sets the error, structure has no effect."""
    rng = np.random.default_rng(7)
    rows = []
    for cls, n, p_struct, mu in (("folded", 300, 0.85, 0.3), ("disordered", 120, 0.35, 1.0)):
        for i in range(n):
            rows.append({"label": f"{cls[0]}{i}", "disorder_class": cls, "has": rng.random() < p_struct,
                         "chi2_raw": 10 ** rng.normal(mu, 0.3)})
    d = pd.DataFrame(rows)
    d["model"], d["clean"] = "bioemu", True
    d["rg_ratio"], d["nrmsd_raw"] = 1.0 + rng.normal(0, 0.05, len(d)), 0.03
    d["resolvability"] = np.where(d["chi2_raw"] <= 2, "raw fit", "strongly reweightable")
    mp = pd.DataFrame({"label": d["label"], "map_method": "SASBDB", "has_pdb_50": d["has"],
                       "has_pdb_any_overlap": d["has"]})
    out = coverage.structure_split(d.drop(columns="has"), mp)
    assert out["pooled_tests"]["chi2_raw_mannwhitney_p"] < 1e-6
    assert out["stratified_by_class"]["chi2_raw"]["p_two_sided"] > 0.01
    assert out["pooled"]["pdb50"]["n"] + out["pooled"]["under50"]["n"] == len(d)
    adj = [out["by_class"][c]["tests"]["chi2_raw_holm_p"] for c in ("folded", "disordered")]
    raw = [out["by_class"][c]["tests"]["chi2_raw_mannwhitney_p"] for c in ("folded", "disordered")]
    assert all(a >= r for a, r in zip(adj, raw, strict=True))


def test_holm_adjustment():
    adj = coverage.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}
