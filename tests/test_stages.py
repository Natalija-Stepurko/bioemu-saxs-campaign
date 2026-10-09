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


def test_operating_point_is_the_first_point_reaching_the_target():
    assert envelopes.operating_point([5.0, 3.0, 1.9, 1.2], raw_chi2=6.0) == 2
    assert envelopes.operating_point([1.5, 1.2], raw_chi2=1.8) == -1          # the raw ensemble already fits
    assert envelopes.operating_point([5.0, 4.0, 3.0], raw_chi2=6.0) is None   # never reached


def test_secondary_structure_records_follow_the_pdb_columns():
    residues = [("ALA", "A", i) for i in range(12)]
    rec = envelopes.ss_records(residues, "CHHHHCCEEECH")
    helix, sheet = rec
    assert helix.startswith("HELIX ") and sheet.startswith("SHEET ")
    # the columns 3Dmol.js and other readers take the residue ranges from
    assert helix[19] == "A" and int(helix[21:25]) == 1 and helix[31] == "A" and int(helix[33:37]) == 4
    assert sheet[21] == "A" and int(sheet[22:26]) == 7 and sheet[32] == "A" and int(sheet[33:37]) == 9
    assert len(rec) == 2                                       # the one-residue helix at the end stays coil


def test_autocrop_trims_white_and_reports_edge_contact(tmp_path):
    from PIL import Image
    im = Image.new("RGB", (100, 80), "white")
    for x in range(40, 60):
        for y in range(30, 50):
            im.putpixel((x, y), (0, 0, 0))
    p = tmp_path / "r.png"
    im.save(p)
    assert envelopes.autocrop(p, margin=5) is False
    assert Image.open(p).size == (30, 30)
    assert envelopes.ink_extent(p, (30, 30)) == (20, 20)
