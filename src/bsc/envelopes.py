"""Stage `envelopes` (DESIGN §7, A3): an ab initio electron-density envelope of each worked example
from its experimental curve, with DENSS (Grant 2018), and a ribbon rendering of the highest-weight
BioEmu-1 conformer inside it, for the worked-examples figure.

DENSS needs numpy < 2 and lives in its own environment (BSC_DENSS_ENV, default /scratch/.venv-denss);
when that environment is absent the reconstruction step prints a notice and is skipped, so `bsc
report` still runs elsewhere with the tracked figure. Per example: `denss-all` with N_MAPS
reconstructions in FAST mode and Dmax from the SASBDB record (pddf_dmax, nm, converted to Å), aligned
and averaged.

The ribbon step docks the conformer into the averaged map by principal axes, writes it as PDB with
HELIX/SHEET records from DSSP (mdtraj), and renders the cartoon with the envelope isosurfaces in
3Dmol.js inside headless Chromium (playwright). The Python that has playwright is BSC_RENDER_PY
(default: this interpreter if it can import playwright); without it the step prints a notice.

Output: results/denss/<label>/ (maps, logs and the ribbon rendering, not tracked) and
results/denss_stats.json (fit statistics, tracked).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import requests

from bsc import analyse, saxs, score
from bsc import config as C

DENSS_ENV = Path(os.environ.get("BSC_DENSS_ENV", "/scratch/.venv-denss"))
N_MAPS = 10
MODE = "FAST"
OUT = C.RESULTS / "denss"


def dmax_angstrom(label: str) -> float | None:
    """Dmax from the SASBDB record: the summary written by `bsc fetch`, else the REST API."""
    summ = json.load(open(C.SASBDB_SUMMARY)) if C.SASBDB_SUMMARY.exists() else {}
    v = summ.get(label, {}).get("pddf_dmax")
    if v is None:
        try:
            r = requests.get(C.SASBDB_API.format(label=label), timeout=60)
            r.raise_for_status()
            v = r.json().get("pddf_dmax")
        except (requests.RequestException, ValueError):
            v = None
        if v is not None and C.SASBDB_SUMMARY.exists():
            summ.setdefault(label, {})["pddf_dmax"] = v
            json.dump(summ, open(C.SASBDB_SUMMARY, "w"), indent=1)
    return 10.0 * float(v) if v is not None else None


def write_dat(label: str, folder: Path) -> Path:
    exp = score.read_experiment(label)
    exp = exp[exp["sigma"] > 0]
    p = folder / f"{label}.dat"
    exp.to_csv(p, sep=" ", header=False, index=False, float_format="%.6e")
    return p


def parse_final_log(p: Path) -> dict:
    txt = p.read_text()
    out = {}
    for key, pat in (("final_chi2", r"Final Chi2:\s*([0-9.eE+-]+)"),
                     ("final_rg", r"Final Rg:\s*([0-9.eE+-]+)"),
                     ("support_volume", r"Final Support Volume:\s*([0-9.eE+-]+)")):
        m = re.findall(pat, txt)
        if m:
            out[key] = float(m[-1])
    return out


def parse_average_log(p: Path) -> dict:
    txt = p.read_text()
    out = {}
    m = re.search(r"Resolution = ([0-9.]+) \+- ([0-9.]+)", txt)
    if m:
        out["resolution_A"], out["resolution_sd_A"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"Mean of correlation scores: ([0-9.]+)", txt)
    if m:
        out["mean_correlation"] = float(m.group(1))
    m = re.search(r"Number of aligned maps accepted: (\d+)", txt)
    if m:
        out["maps_accepted"] = int(m.group(1))
    return out


def run_denss(label: str, dmax: float, cores: int) -> dict:
    folder = OUT / label
    folder.mkdir(parents=True, exist_ok=True)
    dat = write_dat(label, folder)
    cmd = [str(DENSS_ENV / "bin" / "denss-all"), "-f", dat.name, "-d", f"{dmax:.1f}", "-m", MODE,
           "-nm", str(N_MAPS), "-j", str(cores), "-o", label, "--plot_off"]
    with open(folder / "denss-all.log", "w") as log:
        subprocess.run(cmd, cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True)
    maps = folder / label
    logs = [p for p in sorted(maps.glob(f"{label}_[0-9]*.log")) if "_aligned" not in p.name]
    per_map = [parse_final_log(p) for p in logs]
    per_map = [m for m in per_map if "final_chi2" in m]
    rec = {"label": label, "dmax_A": dmax, "n_maps": N_MAPS, "mode": MODE,
           "avg_map": str((maps / f"{label}_avg.mrc").relative_to(C.RESULTS)),
           "chi2_per_map": [m["final_chi2"] for m in per_map],
           "chi2_mean": float(np.mean([m["final_chi2"] for m in per_map])),
           "chi2_median": float(np.median([m["final_chi2"] for m in per_map])),
           "rg_per_map_mean": float(np.mean([m["final_rg"] for m in per_map])),
           "support_volume_mean_A3": float(np.mean([m["support_volume"] for m in per_map]))}
    rec.update(parse_average_log(maps / f"{label}_final.log"))
    return rec


# ---- worked examples: fit at the operating point, docking and ribbon rendering --------------------
POROD_A3_PER_DA = 1.7          # protein volume from mass, for the isovalue that encloses the expected volume
RAW_FIT_CHI2 = 2.0             # the operating point of the kinds: the first point on the path with chi2 <= 2
ENVELOPE_COLOUR, ENVELOPE_OPACITY = "#6F98C4", 0.50
CA_BREAK_A = 4.2               # consecutive C-alpha atoms further apart than this: a break in the chain
HELIX_COLOUR, COIL_COLOUR = "#2F5D8A", "#3A4048"
THREE_DMOL = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"


def ensembles_dir() -> Path:
    return C.DATA / "Predictions" / "PeptoneDB-SAXS-ensembles"


def operating_point(path_chi2: list[float], raw_chi2: float, target: float = RAW_FIT_CHI2) -> int | None:
    """Where the kinds read the reweighting path: -1 when the raw ensemble already fits (chi2 <= target),
    else the index of the first point along the path (decreasing prior strength) with chi2 <= target,
    else None (the target is not reached)."""
    if raw_chi2 <= target:
        return -1
    for i, c in enumerate(path_chi2):
        if c <= target:
            return i
    return None


def example_fit(label: str) -> dict | None:
    """Data, the raw and reweighted BioEmu-1 curves and the AlphaFold2 curve for one entry, each scaled
    to the data. The reweighted curve is taken at the operating point (first point along the path with
    chi2 <= 2); the chi2 minimum of the path is recorded beside it, and its weights pick the conformer
    drawn in the envelope."""
    exp = score.read_experiment(label)
    keep = (exp["sigma"] > 0).to_numpy()
    q, I, s = (exp[c].to_numpy()[keep] for c in ("q", "I", "sigma"))
    curves = score.read_prediction(C.MODEL, label)
    af = score.read_prediction("alphafold", label)
    if curves is None:
        return None
    ok = np.isfinite(curves).all(axis=1)
    idx = np.flatnonzero(ok)
    curves = curves[ok][:, keep]
    raw = saxs.ensemble_curve(curves)
    path, ws = saxs.reweighting_curve(curves, I, s, C.THETA_GRID, return_weights=True)
    chi_path = [p["chi2"] for p in path]
    i_min = int(np.argmin(chi_path))
    i_op = operating_point(chi_path, saxs.chi2(raw, I, s))
    w_min = ws[i_min]
    w_op = np.full(len(curves), 1.0 / len(curves)) if i_op in (-1, None) else ws[i_op]
    op, mn = saxs.ensemble_curve(curves, w_op), saxs.ensemble_curve(curves, w_min)

    def rg(curve):
        return saxs.guinier(q, curve, 0.01 * curve + 1e-9, C.GUINIER_QRG_MAX)["rg"]

    out = {"label": label, "q": q, "I": I, "sigma": s, "curves": {},
           "operating_reached": i_op is not None, "operating_is_raw": i_op == -1,
           "phi_operating": saxs.kish_fraction(w_op), "phi_minimum": saxs.kish_fraction(w_min),
           "chi2_minimum": float(chi_path[i_min]),
           "top_conformer": int(idx[int(np.argmax(w_min))]), "top_weight": float(w_min.max()),
           "rg_raw": rg(raw), "rg_operating": rg(op), "rg_minimum": rg(mn),
           "rg_exp": saxs.guinier(q, I, s, C.GUINIER_QRG_MAX)["rg"]}
    for name, m in (("raw", raw), ("operating", op), ("alphafold", af[0][keep] if af is not None else None)):
        if m is None:
            continue
        c, b = saxs.scale_and_background(m, I, s)
        out["curves"][name] = {"fit": c * m + b, "chi2": saxs.chi2(m, I, s)}
    return out


def principal_frame(x: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and rotation (columns: principal axes, longest first, right-handed)."""
    wts = np.ones(len(x)) if weights is None else weights
    c = (wts[:, None] * x).sum(0) / wts.sum()
    cov = ((x - c) * wts[:, None]).T @ (x - c) / wts.sum()
    vals, vecs = np.linalg.eigh(cov)
    R = vecs[:, ::-1]
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return c, R


def envelope_mesh(mrc_path, volume_A3: float, upsample: int = 3):
    """Marching-cubes mesh (vertices, faces, vertex normals) of the averaged DENSS map at the isovalue
    enclosing `volume_A3`, with the upsampled map, its voxel size and the isovalue."""
    import mrcfile
    from scipy import ndimage
    from skimage import measure
    with mrcfile.open(mrc_path) as m:
        rho = np.asarray(m.data, float)
        vox = float(m.voxel_size.x)
    rho = np.clip(ndimage.zoom(rho, upsample, order=3), 0, None)
    vox /= upsample
    frac = min(0.95, volume_A3 / (rho.size * vox**3))
    iso = float(np.quantile(rho, 1 - frac))
    verts, faces, normals, _ = measure.marching_cubes(rho, level=iso, spacing=(vox, vox, vox))
    # scikit-image's normals point up the density gradient, into the particle; the renderer lights
    # the side the normal points to, so they are turned outwards
    return verts, faces, -normals, rho, vox, iso


class Docking:
    """A conformer superposed on an envelope by principal axes. The four proper axis-sign choices and
    the mirror image of the envelope (SAXS does not fix handedness) are tried; the one placing most
    C-alpha atoms inside the isosurface is kept. The displayed frame is the envelope's principal-axes
    frame (x longest, then y, then z); the conformer keeps its own handedness."""

    def __init__(self, ca: np.ndarray, rho: np.ndarray, vox: float, iso: float):
        self.rho, self.vox = rho, vox
        inside = np.argwhere(rho > iso) * vox
        self.c_map, self.R_map = principal_frame(inside, rho[rho > iso])
        self.c_ca, self.R_ca = principal_frame(ca)
        x_frame = (ca - self.c_ca) @ self.R_ca
        best = None
        for mirror in (1, -1):
            for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                S = np.diag([sx, sy, sx * sy])
                M = np.array([mirror, 1, 1])
                frac = self.inside_fraction(x_frame @ S, iso, M)
                if best is None or frac > best[0]:
                    best = (frac, S, M)
        self.inside, self.S, self.M = best

    def conformer(self, x: np.ndarray) -> np.ndarray:
        """Conformer coordinates (Å, any atoms) -> displayed frame."""
        return ((x - self.c_ca) @ self.R_ca) @ self.S

    def map_points(self, v: np.ndarray) -> np.ndarray:
        """Map-grid coordinates (Å) -> displayed frame."""
        return ((v - self.c_map) @ self.R_map) * self.M

    def map_vectors(self, n: np.ndarray) -> np.ndarray:
        return (n @ self.R_map) * self.M

    def grid(self, x: np.ndarray, M: np.ndarray | None = None) -> np.ndarray:
        M = self.M if M is None else M
        g = ((x * M) @ self.R_map.T + self.c_map) / self.vox
        return np.clip(np.round(g).astype(int), 0, np.array(self.rho.shape) - 1)

    def inside_fraction(self, x_display: np.ndarray, iso: float, M: np.ndarray | None = None) -> float:
        ijk = self.grid(x_display, M)
        return float(np.mean(self.rho[ijk[:, 0], ijk[:, 1], ijk[:, 2]] > iso))


def load_conformer(label: str, conformer: int):
    import mdtraj as md
    d = ensembles_dir() / C.MODEL
    return md.load(str(d / f"{label}.xtc"), top=str(d / f"{label}.pdb"))[conformer]


def ss_records(residues: list[tuple[str, str, int]], ss: str, min_len: int = 3) -> list[str]:
    """PDB HELIX and SHEET records for runs of 'H' and 'E' in a simplified DSSP string; `residues` holds
    (residue name, chain id, residue number) in chain order. Runs shorter than `min_len` are left as coil."""
    out, i, n_h, n_e = [], 0, 0, 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if ss[i] in "HE" and j - i + 1 >= min_len:
            (rn0, ch0, r0), (rn1, ch1, r1) = residues[i], residues[j]
            if ss[i] == "H":
                n_h += 1
                out.append(f"HELIX  {n_h:>3} {n_h:>3} {rn0:>3} {ch0} {r0:>4}  {rn1:>3} {ch1} {r1:>4}  1"
                           f"{'':30}{j - i + 1:>5}")
            else:
                n_e += 1
                out.append(f"SHEET  {n_e:>3} S{n_e:<2} 1 {rn0:>3} {ch0}{r0:>4}  {rn1:>3} {ch1}{r1:>4}  0")
        i = j + 1
    return out


def cartoon_pdb(frame, xyz_A: np.ndarray) -> tuple[str, str]:
    """One conformer as PDB text with secondary-structure records, at the given coordinates (Å), and
    its simplified DSSP string."""
    import mdtraj as md
    ss = "".join(md.compute_dssp(frame, simplified=True)[0]).replace("NA", "C")
    top = frame.topology
    chain_id = "A"
    residues = [(r.name, chain_id, r.resSeq) for r in top.residues]
    lines = ss_records(residues, ss)
    for k, a in enumerate(top.atoms):
        name = a.name if len(a.name) == 4 else f" {a.name:<3}"
        x, y, z = xyz_A[k]
        lines.append(f"ATOM  {k + 1:>5} {name} {a.residue.name:>3} {chain_id}{a.residue.resSeq:>4}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {a.element.symbol:>2}")
    lines.append("END")
    return "\n".join(lines) + "\n", ss


RENDER_HTML = """<!doctype html><html><head><meta charset="utf-8">
<script src="__LIB__"></script>
<style>html,body{margin:0;background:#fff}#v{width:__W__px;height:__H__px;position:relative}</style></head>
<body><div id="v"></div><script>
const pdb = __PDB__; const surfaces = __SURF__;
const v = $3Dmol.createViewer(document.getElementById('v'), {backgroundColor: 'white', antialias: true});
v.setProjection('orthographic');
const m = v.addModel(pdb, 'pdb');
m.setStyle({}, {cartoon: {color: '__COIL__', thickness: 0.5}});
m.setStyle({ss: 'h'}, {cartoon: {color: '__HELIX__'}});
const box = v.addModel(__BOX__, 'xyz');   // the corners of everything drawn, hidden, so zoomTo frames it all
box.setStyle({}, {});
for (const s of surfaces) {
  v.addCustom({vertexArr: s.v.map(p => ({x: p[0], y: p[1], z: p[2]})),
               normalArr: s.n.map(p => ({x: p[0], y: p[1], z: p[2]})),
               faceArr: s.f, color: s.color, opacity: s.opacity});
}
v.zoomTo(); v.zoom(__ZOOM__); v.render(); window.renderDone = true;
</script></body></html>"""

SHOOT = """import sys
from playwright.sync_api import sync_playwright
html, png, w, h = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
    pg.goto("file://" + html)
    pg.wait_for_function("window.renderDone === true", timeout=120000)
    pg.wait_for_timeout(500)
    pg.locator("#v").screenshot(path=png)
    b.close()
"""


def render_python() -> str | None:
    """The interpreter that runs playwright: BSC_RENDER_PY, else this one if it can import playwright."""
    env = os.environ.get("BSC_RENDER_PY")
    if env:
        return env if Path(env).exists() else None
    try:
        import playwright  # noqa: F401
        return sys.executable
    except ImportError:
        return None


def render_ribbon(pdb: str, surfaces: list[dict], png: Path, size=(1800, 1200), fill: float = 0.94) -> bool:
    """Cartoon of `pdb` with translucent surfaces, rendered by 3Dmol.js in headless Chromium. A first
    pass at the default zoom measures the drawing; the second zooms it to `fill` of the frame."""
    py = render_python()
    if py is None:
        print("  no Python with playwright (set BSC_RENDER_PY); ribbon rendering skipped", flush=True)
        return False
    pts = np.vstack([np.array([[float(x) for x in ln[30:54].split()] for ln in pdb.splitlines()
                               if ln.startswith("ATOM")])] + [s["verts"] for s in surfaces])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    box = "8\nbox\n" + "".join(f"X {x:.2f} {y:.2f} {z:.2f}\n" for x, y, z in corners)
    surf = [{"v": np.round(s["verts"], 2).tolist(), "n": np.round(s["normals"], 3).tolist(),
             "f": s["faces"].ravel().astype(int).tolist(), "color": s["color"], "opacity": s["opacity"]}
            for s in surfaces]
    page = png.with_suffix(".html")
    script = png.with_name("render_ribbon.py")
    script.write_text(SHOOT)
    zoom = 1.0
    for _ in range(2):
        subs = {"__LIB__": THREE_DMOL, "__PDB__": json.dumps(pdb), "__SURF__": json.dumps(surf),
                "__W__": str(size[0]), "__H__": str(size[1]), "__COIL__": COIL_COLOUR,
                "__HELIX__": HELIX_COLOUR, "__ZOOM__": f"{zoom:.3f}", "__BOX__": json.dumps(box)}
        html = RENDER_HTML
        for k, v in subs.items():
            html = html.replace(k, v)
        page.write_text(html)
        r = subprocess.run([py, str(script), str(page.resolve()), str(png.resolve()), str(size[0]),
                            str(size[1])], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ribbon rendering failed: {r.stderr.strip()[-400:]}", flush=True)
            return False
        w, h = ink_extent(png, size)
        zoom *= fill / max(w / size[0], h / size[1])
    if autocrop(png):
        print(f"  {png.name}: the rendering touches the frame", flush=True)
    return True


def ink_extent(png: Path, size: tuple[int, int]) -> tuple[int, int]:
    """Width and height, symmetric about the frame centre (zoom acts about it), of the non-white part
    of an image."""
    from PIL import Image
    a = np.asarray(Image.open(png).convert("RGB")).astype(int)
    ink = np.argwhere((255 - a).sum(axis=2) > 12)
    if len(ink) == 0:
        return size
    (y0, x0), (y1, x1) = ink.min(axis=0), ink.max(axis=0)
    cx, cy = size[0] / 2, size[1] / 2
    return int(2 * max(cx - x0, x1 - cx)), int(2 * max(cy - y0, y1 - cy))


def autocrop(png: Path, margin: int = 12) -> bool:
    """Trim the white border around a rendering, keeping `margin` pixels; True if the drawing reaches
    the edge of the frame (it may be cut off)."""
    from PIL import Image
    im = Image.open(png).convert("RGB")
    a = np.asarray(im).astype(int)
    ink = np.argwhere((255 - a).sum(axis=2) > 12)
    if len(ink) == 0:
        return False
    (y0, x0), (y1, x1) = ink.min(axis=0), ink.max(axis=0)
    touches = bool(y0 == 0 or x0 == 0 or y1 == a.shape[0] - 1 or x1 == a.shape[1] - 1)
    box = (max(0, x0 - margin), max(0, y0 - margin), min(a.shape[1], x1 + margin + 1),
           min(a.shape[0], y1 + margin + 1))
    im.crop(box).save(png)
    return touches


def docked_example(label: str, conformer: int, avg_map: Path, volume: float, support: float) -> dict:
    """The envelope surface at the isovalue enclosing the protein's expected volume and the conformer,
    both in the displayed frame, with the share of C-alpha atoms inside that surface and inside the
    larger surface at the volume DENSS assigned to the particle (when that is much larger). One surface
    is drawn: a second translucent surface around it hides the first in the renderer."""
    verts, faces, normals, rho, vox, iso = envelope_mesh(avg_map, volume)
    frame = load_conformer(label, conformer)
    xyz = 10.0 * frame.xyz[0]
    ca = xyz[frame.topology.select("name CA")]
    dock = Docking(ca, rho, vox, iso)
    ca_show = dock.conformer(ca)
    if np.prod(dock.M) < 0:          # a mirrored envelope turns its triangles inside out; restore the winding
        faces = faces[:, [0, 2, 1]]
    out = {"frame": frame, "xyz": dock.conformer(xyz), "ca": ca_show, "inside": dock.inside,
           "chain_breaks": int(np.sum(np.linalg.norm(np.diff(ca, axis=0), axis=1) > CA_BREAK_A)),
           "surfaces": [{"verts": dock.map_points(verts), "faces": faces,
                         "normals": dock.map_vectors(normals), "color": ENVELOPE_COLOUR,
                         "opacity": ENVELOPE_OPACITY}]}
    if support > 1.3 * volume:
        *_, iso2 = envelope_mesh(avg_map, support)
        out["inside_outer"] = dock.inside_fraction(ca_show, iso2)
    else:
        out["inside_outer"] = dock.inside
    return out


def ribbon_png(label: str) -> Path:
    return OUT / label / f"{label}_ribbon.png"


def render_examples(stats: dict) -> None:
    """Ribbon renderings of the three worked examples (conformer and envelopes as in the figure)."""
    _, d = analyse.load()
    ent = pd_read_entries()
    for cls, label in analyse.choose_examples(d).items():
        if label not in stats or not (C.RESULTS / stats[label]["avg_map"]).exists():
            continue
        f = example_fit(label)
        if f is None:
            continue
        mw = float(ent.loc[label, "mw_seq_kda"])
        volume = POROD_A3_PER_DA * 1000.0 * mw
        support = float(stats[label].get("support_volume_mean_A3", volume))
        dk = docked_example(label, f["top_conformer"], C.RESULTS / stats[label]["avg_map"], volume, support)
        pdb, ss = cartoon_pdb(dk["frame"], dk["xyz"])
        png = ribbon_png(label)
        if render_ribbon(pdb, dk["surfaces"], png):
            print(f"  {cls}: {label} ribbon rendered ({ss.count('H')} helix, {ss.count('E')} strand "
                  f"residues; {100 * dk['inside']:.0f}% of Cα inside the inner envelope)", flush=True)


def pd_read_entries():
    import pandas as pd
    return pd.read_csv(C.RESULTS / "entries.csv").set_index("label")

def main(cores: int | None = None) -> None:
    stats_path = C.RESULTS / "denss_stats.json"
    stats = json.load(open(stats_path)) if stats_path.exists() else {}
    if not (DENSS_ENV / "bin" / "denss-all").exists():
        print(f"  DENSS environment {DENSS_ENV} not found; reconstructions skipped (set BSC_DENSS_ENV)",
              flush=True)
    else:
        _, d = analyse.load()
        cores = cores or min(C.N_JOBS, 5)
        for cls, label in analyse.choose_examples(d).items():
            if label in stats and (C.RESULTS / stats[label]["avg_map"]).exists():
                print(f"  {cls}: {label} already reconstructed", flush=True)
                continue
            dmax = dmax_angstrom(label)
            if dmax is None:
                print(f"  {cls}: {label} has no Dmax in its SASBDB record; skipped", flush=True)
                continue
            print(f"  {cls}: {label}, Dmax {dmax:.0f} Å, {N_MAPS} maps in {MODE} mode", flush=True)
            rec = run_denss(label, dmax, cores)
            rec["disorder_class"] = cls
            stats[label] = rec
            json.dump(stats, open(stats_path, "w"), indent=1)
            print(f"    mean chi2 of the maps {rec['chi2_mean']:.2f}; "
                  f"resolution {rec.get('resolution_A', float('nan')):.0f} Å", flush=True)
    if stats and ensembles_dir().exists():
        render_examples(stats)


if __name__ == "__main__":
    main()
