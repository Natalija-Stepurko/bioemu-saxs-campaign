"""Small units of the coordrg and envelopes stages that need no data."""
import numpy as np
import pandas as pd

from bsc import coordrg, envelopes


def test_coordrg_selection_is_seeded_and_covers_the_examples():
    n = 120
    rng = np.random.default_rng(0)
    cls = np.array(["folded"] * 70 + ["partly disordered"] * 30 + ["disordered"] * 20)
    d = pd.DataFrame({"model": "bioemu", "clean": True, "label": [f"S{i:03d}" for i in range(n)],
                      "disorder_class": cls, "chi2_raw": rng.lognormal(1, 0.5, n),
                      "length": rng.integers(50, 400, n)})
    d.loc[5, "clean"] = False
    sel = coordrg.selection(d)
    assert sel == coordrg.selection(d)                      # deterministic
    assert sum(v == "disordered" for v in sel.values()) == 20
    assert sum(v == "folded sample" for v in sel.values()) == coordrg.N_FOLDED_SAMPLE
    assert "S005" not in sel                                 # flagged entries are not analysed
    members = coordrg.archive_members(sorted(sel), ["S001"])
    assert len(members) == 2 * len(sel) + 1 and members[-1].endswith("alphafold/S001.pdb")


def test_denss_logs_are_parsed(tmp_path):
    log = tmp_path / "x_0.log"
    log.write_text("... - Final Chi2: 1.529e+00\n... - Final Rg: 21.353\n"
                   "... - Final Support Volume: 55507.452\n")
    rec = envelopes.parse_final_log(log)
    assert rec == {"final_chi2": 1.529, "final_rg": 21.353, "support_volume": 55507.452}
    avg = tmp_path / "x_final.log"
    avg.write_text("Mean of correlation scores: 0.858\nNumber of aligned maps accepted: 9\n"
                   "Resolution = 25.3 +- 3.7 A\n")
    rec = envelopes.parse_average_log(avg)
    assert rec == {"resolution_A": 25.3, "resolution_sd_A": 3.7, "mean_correlation": 0.858,
                   "maps_accepted": 9}
