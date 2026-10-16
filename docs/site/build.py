"""Build docs/index.html from results/. Every number on the page is read from results/ (analysis.json,
by_class.csv, candidates.csv, predict.json, select_eval.json, examples.json, campaign_arms.json);
checks stop the build if the prose and the numbers disagree, including every sentence that states a
direction or a comparison.

    python3 docs/site/build.py
"""
from __future__ import annotations

import base64
import json
import os
import struct
import sys
from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd

PIXEL_DENSITY = 3   # must match bsc.report.PIXEL_DENSITY; checked below

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("BSC_ROOT", ROOT)) / "results"
OUT = Path(os.environ.get("BSC_PAGE_OUT", ROOT / "docs" / "index.html"))
REPO = "https://github.com/Natalija-Stepurko/bioemu-saxs-campaign"
SITE = "https://natalija-stepurko.com"

A = json.load(open(RESULTS / "analysis.json"))
BC = pd.read_csv(RESULTS / "by_class.csv")
CAND = pd.read_csv(RESULTS / "candidates.csv")
P = json.load(open(RESULTS / "predict.json"))
S = json.load(open(RESULTS / "select_eval.json"))
EX = json.load(open(RESULTS / "examples.json"))
ARMS = json.load(open(RESULTS / "campaign_arms.json"))
B, E, R, H = A["bioemu"], A["expectations"], A["rg_robustness"], A["headline_robustness"]
CLS = {r["disorder_class"]: r for r in A["by_class"] if r["model"] == "bioemu"}
RES = B["resolvability_counts"]
KIND_RAW, KIND_MODEST, KIND_STRONG, KIND_NOTFIT = A["kinds"]
MODEL_NAME = {"bioemu": "BioEmu-1", "alphafold": "AlphaFold2", "esmfold": "ESMFold", "boltz2": "Boltz-2",
              "idpfold": "IDPFold", "peptron": "PepTron", "boltz1x": "Boltz-1x", "esmflow": "ESMFlow",
              "idpsam": "idpSAM", "idpgan": "idpGAN", "idp-o": "IDP-o"}
GENERATORS = ("boltz2", "idpfold", "idpsam", "peptron")
SINGLE = ("alphafold", "esmfold")
CLASSES = ("folded", "partly disordered", "disordered")
FEATURE_LABEL = {"rg_conformer_spread": "spread of the conformers' Rg (interquartile range over median)",
                 "rg_model": "ensemble Rg", "rg_model_over_law": "ensemble Rg over the scaling-law Rg",
                 "rg_model_over_law_disordered": "ensemble Rg over the disordered scaling-law Rg",
                 "ncpr": "net charge per residue", "fcr": "fraction of charged residues", "disorder_mean": "disorder score",
                 "log_length": "chain length (log)", "length": "chain length"}

INK, MUTED, RULE, PANEL, ACC, GREY = "#16191D", "#5B646E", "#DDE1E4", "#FFFFFF", "#2F5D8A", "#EEF1F2"
SANS = "ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def check(cond, msg):
    if not cond:
        sys.exit(f"page text no longer matches results/: {msg}")


def pct(x):
    return f"{100 * x:.0f}%"


def p1(x):
    """Share as a percentage with one decimal (for shares under about 10%)."""
    return f"{100 * x:.1f}%"


def f1(x):
    return f"{x:.1f}"


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def pv(p):
    return "< 0.001" if p < 1e-3 else f"{p:.2g}" if p < 0.01 else f"{p:.2f}"


def ci(v, fmt=f2):
    return f"{fmt(v[0])}–{fmt(v[1])}"


def data_uri(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def png_width(p: Path) -> int:
    """Pixel width from the PNG header."""
    with open(p, "rb") as f:
        head = f.read(24)
    return struct.unpack(">I", head[16:20])[0]


def fig(name, caption, what, read=None, cls="result"):
    """A figure capped at its design width: the PNG carries PIXEL_DENSITY pixels per display pixel."""
    p = RESULTS / "figures" / f"{name}.png"
    check(p.exists(), f"figure {name} exists")
    w = png_width(p) // PIXEL_DENSITY
    check(w <= 860, f"figure {name} is at most 860 px wide")
    body = f'<p><strong>What it shows.</strong> {what}</p>'
    if read:
        body += f'<p><strong>How to read it.</strong> {read}</p>'
    return (f'<figure class="{cls}" style="max-width:{w + 26}px"><figcaption>{caption}</figcaption>'
            f'<img src="{data_uri(p)}" alt="{caption}" width="{w}" style="max-width:{w}px"><div class="rtext">{body}</div></figure>')


def met(k):
    return "met" if E[k]["met"] else "not met"


def short_title(t, n=60):
    t = t or ""
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"


def details(summary, body):
    return f'<details class="more"><summary>{summary}</summary><div class="inner">{body}</div></details>'


REFS = [
    ("lewis", 'Lewis S, Hempel T, Jiménez-Luna J, et al. Scalable emulation of protein equilibrium ensembles with '
              'generative deep learning. <em>Science</em> 2025.', "https://doi.org/10.1126/science.adv9817"),
    ("haitin", 'Haitin Y, et al. Use of biomolecular emulator for characterizing flexible proteins by small-angle '
               'x-ray scattering. <em>Protein Science</em> 2026.', "https://doi.org/10.1002/pro.70646"),
    ("invernizzi", 'Invernizzi M, Bottaro S, Streit JO, et al. Advancing protein ensemble predictions across the '
                   'order–disorder continuum. bioRxiv 2025.', "https://doi.org/10.1101/2025.10.18.680935"),
    ("kikhney", 'Kikhney AG, Borges CR, Molodenskiy DS, Jeffries CM, Svergun DI. SASBDB: towards an automatically '
                'curated and validated repository for biological scattering data. <em>Protein Science</em> 2020.',
     "https://doi.org/10.1002/pro.3731"),
    ("svergun", 'Svergun D, Barberato C, Koch MHJ. CRYSOL – a program to evaluate X-ray solution scattering of '
                'biological macromolecules from atomic coordinates. <em>J Appl Cryst</em> 1995.',
     "https://doi.org/10.1107/S0021889895007047"),
    ("grudinin", 'Grudinin S, Garkavenko M, Kazennov A. Pepsi-SAXS: an adaptive method for rapid and accurate '
                 'computation of small-angle X-ray scattering profiles. <em>Acta Cryst D</em> 2017.',
     "https://doi.org/10.1107/S2059798317005745"),
    ("bottaro", 'Bottaro S, Bengtsen T, Lindorff-Larsen K. Integrating molecular simulation and experimental data: '
                'a Bayesian/maximum entropy reweighting approach. <em>Methods Mol Biol</em> 2020.',
     "https://doi.org/10.1007/978-1-0716-0270-6_15"),
    ("grant", 'Grant TD. Ab initio electron density determination directly from solution scattering data. '
              '<em>Nature Methods</em> 2018.', "https://doi.org/10.1038/nmeth.4581"),
    ("kohn", 'Kohn JE, Millett IS, Jacob J, et al. Random-coil behavior and the dimensions of chemically unfolded '
             'proteins. <em>PNAS</em> 2004.', "https://doi.org/10.1073/pnas.0403643101"),
    ("bulow", 'Bülow S, et al. AF-CALVADOS: ensembles of folded and disordered proteins from AlphaFold and a '
              'coarse-grained model. <em>Protein Science</em> 2026.', "https://doi.org/10.1002/pro.70694"),
    ("tsuboyama", 'Tsuboyama K, Dauparas J, Chen J, et al. Mega-scale experimental analysis of protein folding stability '
                  'in biology and design. <em>Nature</em> 2023.', "https://doi.org/10.1038/s41586-023-06328-6"),
    ("emenecker", 'Emenecker RJ, Griffith D, Holehouse AS. Metapredict: a fast, accurate, and easy-to-use predictor of '
                  'consensus disorder and structure. <em>Biophys J</em> 2021.', "https://doi.org/10.1016/j.bpj.2021.08.039"),
    ("steinegger", 'Steinegger M, Söding J. MMseqs2 enables sensitive protein sequence searching for the analysis of '
                   'massive data sets. <em>Nat Biotechnol</em> 2017.', "https://doi.org/10.1038/nbt.3988"),
]
REFNO = {k: i + 1 for i, (k, _, _) in enumerate(REFS)}


def cite(*keys):
    return "[" + ", ".join(f'<a href="#ref-{k}">{REFNO[k]}</a>' for k in keys) + "]"


# ----------------------------------------------------------------------------- svg helpers
def T(x, y, s, size=12, fill=INK, anchor="start", weight=400, mono=False, italic=False):
    fam = MONO if mono else SANS
    s = s.replace("& ", "&amp; ")
    st = f"font-family:{fam};font-size:{size}px;fill:{fill};font-weight:{weight}"
    if italic:
        st += ";font-style:italic"
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="middle" style="{st}">{s}</text>')


def box(x, y, w, h, fill=PANEL, stroke=RULE, r=3, sw=1):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def arrow(x1, y1, x2, y2, stroke=MUTED, sw=1.2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" '
            f'marker-end="url(#arr)"/>')


def svg(w, h, body, label):
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>{body}</svg>')


def method_diagram(n_profiles, n_conf):
    """Two bands: the public inputs (grey) and what this study computes from them (blue)."""
    W, H = 720, 300
    grey_fill, grey_stroke, blue_fill = "#E9ECEE", "#B9C0C6", "#E6EEF6"
    b = [T(16, 16, "PUBLIC INPUTS", 10.5, MUTED, mono=True, weight=600)]
    cards1 = [("SASBDB profiles", [f"{n_profiles} measured SAXS curves", "with sequence and disorder score"],
               "curated in PeptoneDB-SAXS"),
              ("BioEmu-1 conformers and curves", [f"{n_conf} conformers per protein; one", "Pepsi-SAXS curve per conformer"],
               "made by the PeptoneBench authors")]
    for i, (title, lines, who) in enumerate(cards1):
        x = 16 + i * 352
        b.append(box(x, 30, 336, 86, fill=grey_fill, stroke=grey_stroke, sw=1.2))
        b.append(T(x + 12, 50, title, 13, INK, weight=600))
        for k, ln in enumerate(lines):
            b.append(T(x + 12, 69 + 15 * k, ln, 11.5, INK))
        b.append(T(x + 12, 105, who, 10, MUTED, mono=True))
    b.append(T(16, 152, "COMPUTED IN THIS STUDY", 10.5, ACC, mono=True, weight=600))
    cards2 = [("Raw errors", ["χ² and NRMSD of the", "unweighted ensemble;", "Rg of data and model"]),
              ("Reweighting", ["populations moved", "towards each curve;", "φ and four kinds"]),
              ("Predictability", ["error predicted before", "measuring; acquisition", "policies compared"]),
              ("Experimental design", ["replication arm and", "hypothesis-test arm;", "QC and success criteria"])]
    for i, (title, lines) in enumerate(cards2):
        x = 16 + i * 176
        b.append(box(x, 166, 160, 98, fill=blue_fill, stroke=ACC, sw=1.2))
        b.append(T(x + 10, 186, title, 13, INK, weight=600))
        for k, ln in enumerate(lines):
            b.append(T(x + 10, 208 + 16 * k, ln, 11.5, INK))
        if i < 3:
            b.append(arrow(x + 160, 215, x + 174, 215))
    for x in (184, 536):
        b.append(arrow(x, 117, x, 163))
    b.append(T(16, H - 14, "Grey: taken as published. Blue: computed here.", 11, MUTED))
    return svg(W, H, "".join(b), "Method: public SAXS profiles and BioEmu-1 curves feed the raw errors, reweighting, "
                                 "the predictability tests and the experimental design")


def decision_diagram(n_arm1, n_test, n_ctrl):
    """The two arms of the campaign as a decision diagram."""
    W, H = 720, 262
    b = []
    b.append(box(200, 10, 320, 50, fill="#EEF2F6", stroke=ACC, sw=1.2))
    b.append(T(360, 28, "Error map of 399 measured profiles", 13, INK, anchor="middle", weight=600))
    b.append(T(360, 46, "the error on an unmeasured protein is not predictable", 11, MUTED, anchor="middle"))
    arms = [(16, "Arm 1 · replication", ["Are known discrepancies reproducible", "under controlled conditions?"],
             [f"{n_arm1} high-error, reweightable monomers", "re-measured in one buffer, one", "temperature, tags removed"]),
            (372, "Arm 2 · hypothesis test", ["Are disordered ensembles", "systematically too extended?"],
             [f"{n_test} test proteins chosen from the model,", f"{n_ctrl} scaling-law-matched controls;", "outcome and refutation fixed in advance"])]
    for x, title, q, how in arms:
        b.append(arrow(360, 60, x + 166, 84))
        b.append(box(x, 86, 332, 166, fill=PANEL, stroke=RULE, sw=1.2))
        b.append(T(x + 14, 108, title, 13, INK, weight=600))
        for k, ln in enumerate(q):
            b.append(T(x + 14, 132 + 17 * k, ln, 12, ACC, italic=True))
        for k, ln in enumerate(how):
            b.append(T(x + 14, 184 + 16 * k, ln, 11.5, INK))
    return svg(W, H, "".join(b), "Decision diagram: the error map leads to a replication arm and a hypothesis-test arm")


FAVICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path d="M3 5 C 9 6, 11 18, 17 22 S 26 27, 30 27" '
           'fill="none" stroke="#2F5D8A" stroke-width="3" stroke-linecap="round"/><circle cx="6" cy="7.5" r="2" '
           'fill="#2F5D8A"/><circle cx="13" cy="16" r="2" fill="#2F5D8A"/><circle cx="22" cy="24.5" r="2" '
           'fill="#2F5D8A"/></svg>')


CSS = """
:root{--paper:#F7F8F9;--panel:#FFFFFF;--ink:#16191D;--muted:#5B646E;--rule:#DDE1E4;--acc:#2F5D8A;--bioemu:#1F6FE5;--feas-bg:#FBF3E6;--feas-rule:#C08A3E;--feas-head:#7A4E12;--band:#EEF1F2;
  --sans:ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);line-height:1.6;padding-inline:16px}
.wrap{max-width:980px;margin:0 auto;padding:30px 0 90px}
h2[id]{scroll-margin-top:74px}
.topnav{position:sticky;top:0;z-index:10;background:color-mix(in srgb,var(--paper) 90%,transparent);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--rule);margin-inline:-16px;padding-inline:16px}
.topnav .inner{max-width:980px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;gap:16px;height:54px;position:relative}
.topnav .brand{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:12px;color:var(--ink);text-decoration:none;white-space:nowrap}
.topnav ul{list-style:none;margin:0;padding:0;display:flex;gap:13px;font-family:var(--mono);font-size:10.5px;letter-spacing:.05em;text-transform:uppercase}
.topnav ul a{display:block;color:var(--muted);text-decoration:none;white-space:nowrap;padding:17px 0 15px;border-bottom:2px solid transparent}
.topnav ul a:hover{color:var(--ink)}.topnav ul a.active{color:var(--ink);border-bottom-color:var(--ink)}
.navtoggle{display:none;font:inherit;font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink);
  background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:6px 10px;cursor:pointer;max-width:60vw;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.navtoggle::before{content:"\\2261\\00a0\\00a0";font-size:13px}
@media (max-width:1060px){.navtoggle{display:block}.topnav ul{display:none;position:absolute;top:54px;right:0;flex-direction:column;gap:0;min-width:240px;
  background:var(--panel);border:1px solid var(--rule);border-radius:3px;box-shadow:0 8px 24px rgba(22,25,29,.12);padding:6px 0}
  .topnav.open ul{display:flex}.topnav ul a{padding:9px 16px;border-bottom:0;border-left:2px solid transparent}.topnav ul a.active{border-left-color:var(--ink);background:var(--band)}}
header{border-bottom:2px solid var(--ink);padding-bottom:22px;margin-bottom:30px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
h1{font-size:clamp(26px,3.4vw,36px);line-height:1.14;letter-spacing:-.02em;margin:0 0 10px;text-wrap:balance;font-weight:640}
.scope{font-size:16px;color:var(--muted);margin:0 0 10px}
.byline{font-family:var(--mono);font-size:12.5px;color:var(--muted);margin:0 0 18px}
.abstract{font-size:15px;line-height:1.6;margin:0 0 18px}
.repo{display:flex;align-items:center;gap:12px;flex-wrap:wrap;text-decoration:none;color:inherit;background:var(--panel);border:1px solid var(--rule);
  border-left:3px solid var(--ink);border-radius:0 3px 3px 0;padding:14px 18px}
.repo span{font-size:.92rem;color:var(--muted);line-height:1.45;flex:1 1 24ch}.repo b{color:var(--ink)}
.repo .repo-path{flex:0 0 auto;font-family:var(--mono);font-size:.72rem;color:var(--ink);border:1px solid var(--rule);border-radius:2px;padding:4px 9px;white-space:nowrap}
@media(max-width:640px){.repo .repo-path{flex-basis:100%;text-align:center}}
h2{font-size:21px;letter-spacing:-.01em;margin:52px 0 8px;font-weight:640;text-wrap:balance}
h3{font-size:14.5px;font-weight:640;margin:22px 0 8px}
p{margin:0 0 14px}
.sub{color:var(--muted);font-size:15px;margin:0 0 18px}
.result{margin:18px auto;background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:12px 12px 10px}
.result figcaption{font-family:var(--mono);font-size:11.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
.result img,.result svg{width:100%;height:auto;display:block;margin:0 auto}
.result.diagram svg{min-width:600px}
.result .rtext p{font-size:12.5px;color:var(--muted);margin:8px 0 0}.result .rtext strong{color:var(--ink);font-weight:600}
.steps{counter-reset:s;padding-left:0;list-style:none;margin:14px 0 0;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 22px}
@media (max-width:860px){.steps{grid-template-columns:minmax(0,1fr)}}
.steps li{counter-increment:s;position:relative;padding-left:30px;font-size:13.5px;color:var(--ink)}
.steps li::before{content:counter(s);position:absolute;left:0;top:2px;width:20px;height:20px;border-radius:50%;background:var(--ink);color:#fff;font-family:var(--mono);font-size:11px;display:grid;place-items:center}
.steps li .who{font-family:var(--mono);font-size:10.5px;color:var(--muted);letter-spacing:.04em}
.steps li .who.here{color:var(--acc)}
.findings{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:18px 0}
@media (max-width:860px){.findings{grid-template-columns:minmax(0,1fr)}}
.findings>p{background:var(--panel);border:1px solid var(--rule);border-top:3px solid var(--ink);border-radius:3px;padding:14px 16px;margin:0;font-size:14px}
.call{border-left:3px solid var(--acc);background:#EEF2F6;padding:14px 20px;margin:18px 0;border-radius:0 3px 3px 0}
.feas{border:1px solid #EAD8BC;border-left:4px solid var(--feas-rule);background:var(--feas-bg);padding:12px 20px 6px;margin:18px 0;border-radius:0 3px 3px 0}
.feas .feas-title{font-family:var(--mono);font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--feas-head);font-weight:600;margin:0 0 6px}
.feas ul{margin:0 0 6px;padding-left:20px}.feas li{font-size:14.5px;margin:0 0 6px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule)}
td{padding:8px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
td.num{font-family:var(--mono);font-size:12.5px;text-align:right;white-space:nowrap}
td.dim{color:var(--muted);font-size:12.5px}
table.wide{min-width:680px}
tr.base td{color:var(--muted)}
.note{color:var(--muted);font-size:13px}
details.more{margin:10px 0 18px;border:1px solid var(--rule);border-radius:3px;background:var(--panel)}
details.more summary{cursor:pointer;padding:10px 14px;font-family:var(--mono);font-size:11.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
details.more[open] summary{border-bottom:1px solid var(--rule)}
details.more .inner{padding:4px 14px 12px}details.more .inner>p{margin:10px 0}
details.more .result{border:0;padding:0}
.tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:4px 0 16px}
@media (max-width:760px){.tiles{grid-template-columns:minmax(0,1fr)}}
.tile{background:var(--panel);border:1px solid var(--rule);border-top:3px solid var(--bioemu);border-radius:3px;padding:12px 14px}
.tile .v{font-size:26px;font-weight:640;letter-spacing:-.01em;line-height:1.1}
.nb{white-space:nowrap}
.tile .l{font-size:14px;margin:4px 0 2px}.tile .s{font-family:var(--mono);font-size:11.5px;color:var(--muted);line-height:1.45}
.flow{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;font-size:13.5px;margin:0 0 16px;color:var(--muted)}
.flow b{color:var(--ink);font-size:15px;font-family:var(--mono)}.flow .arr{color:var(--muted)}
.lead{font-size:15.5px}
@media (max-width:640px){table.exp,table.exp tbody,table.exp tr,table.exp td{display:block}table.exp thead{display:none}table.exp tr{border-bottom:1px solid var(--rule);padding:8px 0}table.exp td{border:0;padding:2px 0}}
.outcome{font-family:var(--mono);font-size:11.5px;letter-spacing:.03em;text-transform:uppercase;white-space:nowrap}
.outcome.yes{color:var(--acc)}.outcome.no{color:var(--muted)}.outcome.rev{color:#A0521A}
ol.refs{padding-left:22px;font-size:13.5px;color:var(--muted)}ol.refs li{margin:0 0 6px}ol.refs a{color:var(--ink);word-break:break-all}
ul.facts{padding-left:20px;margin:0 0 16px}ul.facts li{margin:0 0 9px}
tr.grp td{font-family:var(--mono);font-size:11px;letter-spacing:.04em;color:var(--muted);background:var(--band);padding-top:6px;padding-bottom:6px}
footer{margin-top:60px;padding-top:16px;border-top:1px solid var(--rule);font-family:var(--mono);font-size:11.5px;color:var(--muted);line-height:1.8}
footer a{color:var(--ink)}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
"""

NAV_JS = """
(()=>{const nav=document.querySelector('.topnav'),btn=nav.querySelector('.navtoggle');
const links=[...nav.querySelectorAll('ul a')];const byId=new Map(links.map(a=>[a.getAttribute('href').slice(1),a]));
const setOpen=o=>{nav.classList.toggle('open',o);btn.setAttribute('aria-expanded',String(o));};
btn.addEventListener('click',()=>setOpen(!nav.classList.contains('open')));
links.forEach(a=>a.addEventListener('click',()=>setOpen(false)));
document.addEventListener('keydown',e=>{if(e.key==='Escape')setOpen(false);});
document.addEventListener('click',e=>{if(!nav.contains(e.target))setOpen(false);});
const heads=[...byId.keys()].map(id=>document.getElementById(id)).filter(Boolean);
const mark=()=>{const y=nav.offsetHeight+24;let cur=null;for(const h of heads){if(h.getBoundingClientRect().top<=y)cur=h;else break;}
links.forEach(a=>{a.classList.remove('active');a.removeAttribute('aria-current');});
if(cur){const a=byId.get(cur.id);a.classList.add('active');a.setAttribute('aria-current','true');btn.textContent=a.textContent;}else btn.textContent='Sections';};
addEventListener('scroll',mark,{passive:true});addEventListener('resize',mark);mark();})();
"""

GH = ('<svg viewBox="0 0 16 16" aria-hidden="true" width="17" height="17"><path fill="currentColor" d="M8 0C3.58 0 0 '
      '3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23'
      '-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07'
      '-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2'
      '-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 '
      '3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58'
      '-8-8-8Z"/></svg>')


def coverage_section(n_bioemu: int) -> str:
    """'Where the SAXS set sits in protein structure space': every number from results/coverage/,
    with a check for every directional or comparative sentence."""
    CV = json.load(open(RESULTS / "coverage" / "summary.json"))
    SP = json.load(open(RESULTS / "coverage" / "structure_split.json"))
    RC = json.load(open(RESULTS / "coverage" / "rank_coverage.json"))
    cath, cov, comp = CV["cath_reference"], CV["coverage"], CV["composition"]
    g3, mapc = cov["gene3d"], CV["mapping"]["clean"]
    check(CV["n_clean"] == n_bioemu and SP["n"] + SP["n_unmapped_excluded"] == n_bioemu,
          "coverage analysis covers the same clean BioEmu-1 profiles")
    check(mapc["mapped"]["k"] == SP["n"] and mapc["by_method"]["unmapped"] == SP["n_unmapped_excluded"],
          "mapped and unmapped counts agree between the coverage outputs")
    ver = cath["version"].lstrip("v")
    check(ver.startswith("4.4"), "reference is CATH v4.4")
    # -- finding 1: common superfamilies covered, rare ones not
    t50, t200 = cov["top50_superfamilies_present_gene3d"], cov["top200_superfamilies_present_gene3d"]
    tail = RC["bins"][-1]
    check(t50["share"] > t200["share"] > 3 * tail["share"] and tail["share"] < 0.02,
          "superfamily coverage falls with rank")
    check(all(a["share"] >= b["share"] for a, b in zip(RC["bins"][1:-1], RC["bins"][2:], strict=False)),
          "coverage decreases monotonically beyond the top ten")
    check(g3["share_of_cath_s35_clusters_in_represented_superfamilies"] > 5 * g3["share_of_cath_superfamilies"],
          "the superfamilies present hold a much larger share of S35 clusters than of superfamilies")
    # -- finding 2: class skew
    cls = {r["class"]: r for r in comp["class"]}
    a_, b_, ab = cls[1], cls[2], cls[3]
    check(b_["enr_lo"] > 1 and ab["enr_hi"] < 1 and a_["enr_lo"] < 1 < a_["enr_hi"],
          "mainly beta enriched, alpha/beta depleted, mainly alpha within its interval of CATH")
    top_under = {r["id"]: r for r in CV["over_under"]["topology_under"]}
    sf_under = {r["id"]: r for r in CV["over_under"]["superfamily_under"]}
    sf_over = {r["id"]: r for r in CV["over_under"]["superfamily_over"]}
    ross, tim, nadp = top_under["3.40.50"], top_under["3.20.20"], sf_under["3.40.50.720"]
    check(ross["observed"] < ross["expected"] and tim["observed"] < tim["expected"] and nadp["observed"] == 0,
          "Rossmann and TIM-barrel folds under-represented; NAD(P)-binding Rossmann absent")
    sig = ["2.30.30.40", "1.10.238.10", "1.10.510.10", "2.30.29.30"]
    check(all(s in sf_over and sf_over[s]["observed"] > 2 * sf_over[s]["expected"] for s in sig),
          "SH3, EF-hand, kinase and PH superfamilies over-represented")
    MP = pd.read_csv(RESULTS / "coverage" / "per_protein_mapping.csv")
    MP = MP[MP["clean"].astype(str) == "True"]
    has_sig = MP["gene3d_superfamilies"].fillna("").str.split(";").map(lambda x: bool(set(x) & set(sig)))
    n_sig, n_sig_human = int(has_sig.sum()), int((MP.loc[has_sig, "org_group"] == "Human").sum())
    check(n_sig_human > n_sig / 2, "most signalling-domain profiles are human")
    # -- finding 3: organism
    org = {r["group"]: r for r in CV["organism"]["groups"]}
    hu = org["Human"]
    check(hu["ours_lo"] > hu["pdb_nr30_share"] > hu["swissprot_share"], "human proteins enriched over the PDB and Swiss-Prot")
    check(max(org, key=lambda k: org[k]["ours_share"]) == "Human", "human is the largest organism group")
    # -- finding 4: fragments of long multidomain proteins
    L, D = CV["length"], CV["domains_per_protein"]
    check(L["ours_share_gt500"]["k"] == 0 and L["pdb_nr30_ge30_share_gt500"] > 0.1 and L["swissprot_share_gt500"] > 0.1,
          "no construct above 500 residues; over a tenth of the references are")
    check(L["ours_fulllength_parent"]["median"] > 1.5 * L["ours_construct"]["median"], "parents much longer than constructs")
    check(D["ours_parent_fulllength_gene3d_mean"] > 2 * D["swissprot_fulllength_gene3d_mean"] > 2 * D["ours_construct_gene3d_mean"],
          "parents carry more than twice the domains of a Swiss-Prot protein; constructs fewer than Swiss-Prot")
    check(D["ours_parent_fulllength_gene3d"]["4+"] > 3 * D["swissprot_fulllength_gene3d"]["4+"], "parents: four or more domains far more often")
    # -- finding 5: disorder
    U = CV["disorder"]["uniprot_disordered_annotation"]
    MOST, LONG = "mostly disordered (>=50% residues)", "long IDR (>=30 aa), <50%"
    mc, sp_most = U["ours_construct"][MOST], U["swissprot_all"][MOST]
    check(mc["ci95"][0] > 2 * sp_most, "mostly disordered constructs over twice as frequent as in Swiss-Prot")
    par_idr = U["ours_parent_fulllength"][MOST]["share"] + U["ours_parent_fulllength"][LONG]["share"]
    sp_idr = U["swissprot_all"][MOST] + U["swissprot_all"][LONG]
    con_idr = U["ours_construct"][MOST]["share"] + U["ours_construct"][LONG]["share"]
    check(par_idr > 2.5 * sp_idr and par_idr > 1.8 * con_idr, "parents carry long disordered regions that the constructs mostly leave out")
    # -- finding 6: structure availability
    SA = CV["structure_availability"]
    nos, by = SA["no_pdb_ge50pct_of_construct"], SA["by_disorder_class_no_pdb_ge50"]
    check(0.2 < nos["share"] < 0.3, "about a quarter without an experimental structure of the construct")
    check(by["disordered"]["share"] > by["partly disordered"]["share"] > by["folded"]["share"]
          and SA["fisher_disordered_or_partly_vs_folded"]["p"] < 0.001, "structure missing more often the more disordered")

    # -- the structure split
    P, BC = SP["pooled"], SP["by_class"]
    check(SP["groups"] == ["pdb50", "partial", "none"], "three structure groups")
    check(P["pdb50"]["n"] + P["partial"]["n"] + P["none"]["n"] == SP["n"], "groups add up")
    check(P["pdb50"]["raw_fit"]["share"] > P["none"]["raw_fit"]["share"] > P["partial"]["raw_fit"]["share"]
          and P["pdb50"]["chi2_raw_median"] < P["none"]["chi2_raw_median"] < P["partial"]["chi2_raw_median"],
          "pooled: best fits with a structure, worst with partial overlap")
    pt, st = SP["pooled_tests"], SP["stratified_by_class"]
    check(pt["chi2_raw_mannwhitney_p"] < 0.05 and pt["rg_ratio_mannwhitney_p"] < 0.05, "pooled differences significant")
    check(pt["nrmsd_raw_mannwhitney_p"] > 0.05, "pooled: no difference under NRMSD")
    check(pt["chi2_raw_pdb50_vs_none_p"] > 0.05, "pooled: structure against no overlap not significant")
    check(st["chi2_raw"]["p_two_sided"] > 0.05 and st["rg_ratio"]["p_two_sided"] > 0.05, "no difference stratified by class")
    share50 = SP["pdb50_share_by_class"]
    check(share50["folded"]["share"] > share50["partly disordered"]["share"] > share50["disordered"]["share"],
          "structure coverage falls with disorder")
    for c in ("folded", "partly disordered"):
        check(BC[c]["tests"]["chi2_raw_mannwhitney_p"] > 0.05 and BC[c]["tests"]["rg_ratio_mannwhitney_p"] > 0.05,
              f"{c}: no difference")
    dz = BC["disordered"]
    check(dz["tests"]["chi2_raw_mannwhitney_p"] < 0.05 < dz["tests"]["chi2_raw_holm_p"], "disordered: nominal only")
    check(dz["pdb50"]["chi2_raw_median"] < dz["under50"]["chi2_raw_median"], "disordered: lower chi2 with a structure")
    check(dz["tests"]["nrmsd_raw_mannwhitney_p"] > 0.05 and dz["tests"]["rg_ratio_mannwhitney_p"] > 0.05,
          "disordered: no difference in NRMSD or Rg ratio")
    check(1 < dz["pdb50"]["rg_ratio_median"] < dz["under50"]["rg_ratio_median"] and dz["pdb50"]["rg_ratio_ci95"][0] < 1
          and dz["under50"]["rg_ratio_ci95"][0] > 1, "disordered: still above 1 with a structure, interval reaching 1")
    check(BC["partly disordered"]["pdb50"]["chi2_raw_median"] > BC["partly disordered"]["under50"]["chi2_raw_median"],
          "partly disordered: median higher with a structure (stated)")

    def ps(r):          # share with Wilson interval
        return f'{pct(r["share"])} ({pct(r["ci95"][0])}–{pct(r["ci95"][1])})'

    def row(name, r, cls_=""):
        nf = r["target_not_reached"]
        return (f'<tr{cls_}><td>{name}</td><td class="num">{r["n"]}</td><td class="num">{ps(r["raw_fit"])}</td>'
                f'<td class="num">{f1(r["chi2_raw_median"])} ({ci(r["chi2_raw_ci95"], f1)})</td>'
                f'<td class="num">{f3(r["nrmsd_raw_median"])}</td>'
                f'<td class="num">{f2(r["rg_ratio_median"])} ({ci(r["rg_ratio_ci95"])})</td>'
                f'<td class="num">{pct(nf["share"])}</td></tr>')
    GL = {"pdb50": "PDB ≥ 50%", "partial": "partial", "none": "no PDB"}
    rows = '<tr class="grp"><td colspan="7">all classes</td></tr>' + "".join(row(GL[g], P[g]) for g in SP["groups"])
    for c in CLASSES:
        t = BC[c]["tests"]
        rows += (f'<tr class="grp"><td colspan="7">{c} · Mann–Whitney, PDB ≥ 50% against the rest: χ² p {pv(t["chi2_raw_mannwhitney_p"])}, '
                 f'Rg ratio p {pv(t["rg_ratio_mannwhitney_p"])}</td></tr>' + "".join(row(GL[g], BC[c][g]) for g in SP["groups"]))
    split_table = ('<div class="scroll"><table class="wide split"><thead><tr><th>construct structure</th><th>n</th><th>χ² ≤ 2 (95%)</th>'
                   '<th>median raw χ² (95%)</th><th>median NRMSD</th><th>median Rg ratio (95%)</th><th>target fit not reached</th></tr></thead><tbody>'
                   + rows + '</tbody></table></div><p class="note">PDB ≥ 50%: experimental chains (X-ray, electron microscopy or NMR) cover at least half of the construct; partial: some overlap, under half; no PDB: no overlap. '
                   f'Shares: Wilson 95% intervals; medians: percentile bootstrap over entries (2,000 resamples). {SP["n_unmapped_excluded"]} profiles without a UniProt mapping are left out.</p>')

    # -- details: composition tables
    def ou_rows(lst, k=8):
        return "".join(f'<tr><td>{r["id"]}</td><td class="dim">{r["name"] or "unnamed"}</td><td class="num">{r["observed"]}</td>'
                       f'<td class="num">{r["expected"]:.1f}</td><td class="num">{r["cath_s35"]:,}</td><td class="num">{pv(r["p"])}</td></tr>' for r in lst[:k])
    ou_head = '<thead><tr><th>CATH id</th><th>name</th><th>observed</th><th>expected</th><th>CATH S35 clusters</th><th>Poisson p</th></tr></thead>'
    ou = "".join(f'<p class="note">{title}</p><div class="scroll"><table class="wide">{ou_head}<tbody>{ou_rows(CV["over_under"][key])}</tbody></table></div>'
                 for title, key in (("Superfamilies, over-represented", "superfamily_over"), ("Superfamilies, under-represented", "superfamily_under"),
                                    ("Topologies, over-represented", "topology_over"), ("Topologies, under-represented", "topology_under")))
    ou_note = (f'<p class="note">Units: {CV["over_under"]["n_units"]} unique pairs of UniProt accession and CATH-Gene3D superfamily inside the constructs; '
               'expected = units × the CATH S35 share; the Poisson tails are uncorrected and serve for ranking only.</p>')
    cls_rows = "".join(f'<tr><td>{r["name"]}</td><td class="num">{r["ours_units"]}</td><td class="num">{pct(r["ours_share"])} ({pct(r["ours_lo"])}–{pct(r["ours_hi"])})</td>'
                       f'<td class="num">{r["cath_s35"]:,}</td><td class="num">{pct(r["cath_share"])}</td><td class="num">{f2(r["enrichment"])} ({ci([r["enr_lo"], r["enr_hi"]])})</td></tr>'
                       for r in comp["class"])
    cls_table = ('<div class="scroll"><table><thead><tr><th>CATH class</th><th>SAXS units</th><th>SAXS share (95%)</th><th>CATH S35 clusters</th>'
                 '<th>CATH share</th><th>ratio (95%)</th></tr></thead><tbody>' + cls_rows + '</tbody></table></div>'
                 f'<p class="note">Intervals: bootstrap over UniProt accessions (2,000 resamples). {comp["n_units"]} units in classes 1–4 from {comp["n_proteins"]} proteins.</p>')
    LQ = (("SAXS constructs", "ours_construct"), ("their full-length proteins", "ours_fulllength_parent"),
          ("PDB 30% clusters, ≥ 30 residues", "pdb_nr30_ge30"), ("Swiss-Prot", "swissprot"))
    len_rows = "".join(f'<tr><td>{nm}</td>' + "".join(f'<td class="num">{L[k][q]:.0f}</td>' for q in ("p10", "p25", "median", "p75", "p90")) + '</tr>' for nm, k in LQ)
    len_table = ('<div class="scroll"><table><thead><tr><th>chain length (residues)</th><th>10th</th><th>25th</th><th>median</th><th>75th</th><th>90th</th></tr></thead><tbody>'
                 + len_rows + '</tbody></table></div>')
    DIS = ((MOST, "≥ 50% of residues disordered"), (LONG, "a disordered region of ≥ 30 residues, under 50%"))
    dis_rows = "".join(f'<tr><td>{lab}</td><td class="num">{pct(U["ours_construct"][k]["share"])}</td><td class="num">{pct(U["ours_parent_fulllength"][k]["share"])}</td>'
                       f'<td class="num">{U["swissprot_all"][k] * 100:.1f}%</td><td class="num">{pct(U["swissprot_with_pdb_xref"][k])}</td></tr>' for k, lab in DIS)
    dis_table = ('<div class="scroll"><table><thead><tr><th>UniProt "Disordered" annotation</th><th>SAXS constructs</th><th>full-length proteins</th>'
                 '<th>Swiss-Prot</th><th>Swiss-Prot with a PDB entry</th></tr></thead><tbody>' + dis_rows + '</tbody></table></div>'
                 '<p class="note">UniProt REGION features with the note "Disordered", mostly MobiDB-lite predictions.</p>')
    org_rows = "".join(f'<tr><td>{g}</td><td class="num">{org[g]["ours"]}</td><td class="num">{pct(org[g]["ours_share"])}</td>'
                       f'<td class="num">{pct(org[g]["pdb_nr30_share"])}</td><td class="num">{org[g]["swissprot_share"] * 100:.1f}%</td></tr>' for g in org)
    org_table = ('<div class="scroll"><table><thead><tr><th>organism</th><th>SAXS entries</th><th>SAXS share</th><th>PDB 30% clusters</th><th>Swiss-Prot</th></tr></thead><tbody>'
                 + org_rows + '</tbody></table></div>'
                 f'<p class="note">{CV["organism"]["n_pdb_nr30_ge30"]:,} PDB clusters of chains of at least 30 residues with an experimental entity; {CV["organism"]["n_swissprot"]:,} Swiss-Prot entries.</p>')
    by_class_struct = "; ".join(f'{c} {by[c]["k"]} of {by[c]["n"]} ({pct(by[c]["share"])})' for c in CLASSES)
    VER = json.load(open(RESULTS / "coverage" / "versions.json"))
    ver_rows = "".join(f'<tr><td>{k}</td><td>{v.get("version") or v.get("release") or v.get("file_last_modified") or "REST API"}</td></tr>'
                       for k, v in VER.items())
    unm = mapc["unmapped_labels"]
    repl = MP["map_detail"].fillna("").str.contains("SASBDB accession")
    check(int(repl.sum()) == mapc["sasbdb_accession_replaced"], "corrected-accession count")
    n_replaced_unmapped = int((repl & (MP["map_method"] == "unmapped")).sum())
    check(len(unm) == SP["n_unmapped_excluded"], "unmapped list complete")
    mapping_html = (
        f'<p>{mapc["mapped"]["k"]} of {mapc["n"]} profiles map to a UniProtKB entry: {mapc["by_method"]["SASBDB"]} through the accession in the SASBDB record, '
        f'{mapc["by_method"]["RCSB seq search + SIFTS"]} through a PDB sequence search (identity ≥ 90%) and the SIFTS reference of the hit, and '
        f'{mapc["by_method"]["UniProt peptide search"]} through a UniProt peptide search; a local alignment to the canonical sequence then defines the construct region '
        f'(median identity {f2(mapc["median_alignment_identity"])}). For {mapc["sasbdb_accession_replaced"]} profiles the SASBDB accession was set aside because it pointed to a '
        f'UniProtKB entry deleted since deposition or aligned to a small part of the construct; all but {n_replaced_unmapped} of them were mapped again by sequence search. The {len(unm)} unmapped profiles are '
        + ", ".join(f'{lab} ({short_title(nm, 46)})' for lab, nm in unm)
        + '. Domains are CATH-Gene3D assignments whose residues lie at least half inside the construct region; PDB coverage counts residues of '
          'atomic-model chains listed by SIFTS.</p>'
        f'<div class="scroll"><table><thead><tr><th>source</th><th>version</th></tr></thead><tbody>{ver_rows}</tbody></table></div>'
        f'<p class="note">CATH {cath["version"]}: {cath["n_domains"]:,} domains, {cath["n_s35_clusters"]:,} S35 clusters, {cath["n_superfamilies"]:,} superfamilies, '
        f'{cath["n_topologies"]:,} topologies, {cath["n_architectures"]} architectures. CATH-Gene3D in InterPro is release 4.3.0; every superfamily it assigns here is present in {cath["version"]}.</p>')
    check(g3["superfamilies_not_in_cath_v4.4"] == [], "every Gene3D superfamily present in CATH v4.4")

    # -- prose
    lead = (
        f"<p>As the reference for experimentally known structure space I use CATH {ver} at the S35 level (domains clustered at 35% sequence identity) for folds, "
        f"and non-redundant PDB clusters (30% identity) and Swiss-Prot for organism, length and disorder. CATH and the PDB describe what has been solved; "
        f"the comparison with the training data is above.</p>")
    findings = (
        f'<ul class="facts">'
        f'<li><strong>Common folds are covered, rare ones are not.</strong> {pct(cov["has_gene3d_superfamily"]["share"])} of the {n_bioemu} profiles carry a CATH superfamily inside the construct. '
        f'The {g3["distinct_superfamilies"]} superfamilies present are {g3["share_of_cath_superfamilies"] * 100:.1f}% of CATH\'s {cath["n_superfamilies"]:,} but hold {pct(g3["share_of_cath_s35_clusters_in_represented_superfamilies"])} of its S35 clusters: '
        f'{t50["k"]} of the 50 most populated superfamilies are present ({pct(t50["share"])}), {t200["k"]} of the top 200, and {tail["k"]} of the {tail["n"]:,} ranked beyond {tail["first"] - 1:,} ({tail["share"] * 100:.1f}%).</li>'
        f'<li><strong>Mainly-β domains are enriched and α/β enzyme folds depleted.</strong> Mainly-β domains make up {pct(b_["ours_share"])} of the set against {pct(b_["cath_share"])} of CATH (ratio {f2(b_["enrichment"])}, 95% interval {ci([b_["enr_lo"], b_["enr_hi"]])}), '
        f'α/β domains {pct(ab["ours_share"])} against {pct(ab["cath_share"])} ({f2(ab["enrichment"])}, {ci([ab["enr_lo"], ab["enr_hi"]])}). The Rossmann fold appears {ross["observed"]} times where {ross["expected"]:.0f} are expected, '
        f'the TIM barrel {tim["observed"]} times against {tim["expected"]:.0f}, and the NAD(P)-binding Rossmann superfamily not at all. The most over-represented superfamilies include SH3, EF-hand, protein kinase and PH domains; {n_sig_human} of the {n_sig} profiles that carry one are of human proteins.</li>'
        f'<li><strong>Human proteins dominate.</strong> {pct(hu["ours_share"])} of the profiles are of human proteins ({pct(hu["ours_lo"])}–{pct(hu["ours_hi"])}), against {pct(hu["pdb_nr30_share"])} of PDB clusters and {hu["swissprot_share"] * 100:.1f}% of Swiss-Prot.</li>'
        f'<li><strong>The constructs are fragments of long multidomain proteins.</strong> No construct is longer than 500 residues ({pct(L["pdb_nr30_ge30_share_gt500"])} of PDB clusters and {pct(L["swissprot_share_gt500"])} of Swiss-Prot proteins are); '
        f'the median construct has {L["ours_construct"]["median"]:.0f} residues, the full-length protein it comes from {L["ours_fulllength_parent"]["median"]:.0f}. The full-length proteins carry {f1(D["ours_parent_fulllength_gene3d_mean"])} CATH domains on average, '
        f'{pct(D["ours_parent_fulllength_gene3d"]["4+"])} of them four or more, against {f1(D["swissprot_fulllength_gene3d_mean"])} and {pct(D["swissprot_fulllength_gene3d"]["4+"])} for Swiss-Prot; the constructs carry {f1(D["ours_construct_gene3d_mean"])}.</li>'
        f'<li><strong>Disorder is over-represented.</strong> By UniProt annotation {mc["share"] * 100:.1f}% of the constructs are mostly disordered ({mc["ci95"][0] * 100:.1f}–{mc["ci95"][1] * 100:.1f}%), against {sp_most * 100:.1f}% of Swiss-Prot; '
        f'{pct(par_idr)} of the full-length proteins carry a long disordered region or are mostly disordered, against {pct(sp_idr)} of Swiss-Prot, and the constructs keep part of it ({pct(con_idr)}).</li>'
        f'<li><strong>A quarter of the measured constructs have no experimental structure.</strong> For {nos["k"]} of {nos["n"]} mapped profiles ({pct(nos["share"])}, {pct(nos["ci95"][0])}–{pct(nos["ci95"][1])}) no PDB structure covers half of the construct, '
        f'and for {SA["no_pdb_overlap_at_all"]["k"]} none overlaps it. The share rises with disorder: {by_class_struct}.</li></ul>')
    T = json.load(open(RESULTS / "coverage" / "training_summary.json"))
    td, to_, tc_, tl = T["disorder"], T["organism"], T["composition"], T["length"]
    t_most = {k: td[k]["mostly disordered"]["share"] for k in ("SAXS", "AFDB", "CATH MD")}
    t_hu = {k: to_[k]["Human"]["share"] for k in ("SAXS", "AFDB", "CATH MD")}
    t_b = {k: tc_[k]["class"]["Mainly beta"]["share"] for k in ("SAXS", "AFDB", "CATH MD", "CATH S35")}
    t_r = {k: tc_[k]["folds"]["3.40.50 Rossmann fold"]["share"] for k in ("SAXS", "AFDB", "CATH MD", "CATH S35")}
    check(t_most["SAXS"] > 5 * t_most["AFDB"] and t_hu["SAXS"] > 2 * t_hu["CATH MD"] > 10 * t_hu["AFDB"]
          and t_b["SAXS"] > t_b["CATH S35"] > t_b["CATH MD"] > t_b["AFDB"], "skews where training is thin")
    check(t_r["AFDB"] > t_r["CATH S35"] > t_r["SAXS"] and t_r["CATH MD"] > t_r["CATH S35"], "Rossmann skew not a training gap")
    interp = (
        f"<p><strong>Interpretation.</strong> Set against the measured composition of the training data, three of the archive's skews fall where the training data are thinnest: "
        f"disorder ({p1(t_most['AFDB'])} of AFDB input chains and {p1(t_most['CATH MD'])} of simulated domains predicted mostly disordered, against {p1(t_most['SAXS'])} of the constructs), "
        f"human proteins ({p1(t_hu['AFDB'])} of AFDB clusters, {pct(t_hu['CATH MD'])} of simulated domains, {pct(t_hu['SAXS'])} of the profiles) and mainly-β domains "
        f"({pct(t_b['AFDB'])} of AFDB units, {pct(t_b['CATH MD'])} of simulated ones, {pct(t_b['SAXS'])} of the set). The depletion of α/β enzyme folds is a gap of the archive only: "
        f"the Rossmann fold is {pct(t_r['AFDB'])} of AFDB units and {pct(t_r['CATH MD'])} of simulated ones, above its CATH share. "
        f"Long chains are present in pre-training as single predicted structures ({pct(tl['AFDB']['share_gt500'])} of AFDB inputs above 500 residues) and absent from both the simulations and the archive.</p>")
    p50, pp_, pn = P["pdb50"], P["partial"], P["none"]
    split_prose = (
        f"<p>Of the {SP['n']} mapped profiles, {p50['n']} have an experimental structure covering at least half of the construct, {pp_['n']} a partial one and {pn['n']} none. "
        f"Pooled, BioEmu-1's raw ensemble meets χ² ≤ 2 for {ps(p50['raw_fit'])} of the first group, {ps(pp_['raw_fit'])} of the second and {ps(pn['raw_fit'])} of the third, "
        f"with median raw χ² {f1(p50['chi2_raw_median'])}, {f1(pp_['chi2_raw_median'])} and {f1(pn['chi2_raw_median'])} "
        f"(at least half covered against the rest, two-sided Mann–Whitney p {pv(pt['chi2_raw_mannwhitney_p'])}). The groups differ in disorder: "
        f"a structure covers at least half of the construct for {pct(share50['folded']['share'])} of folded, {pct(share50['partly disordered']['share'])} of partly disordered and "
        f"{pct(share50['disordered']['share'])} of disordered profiles. Within classes the difference vanishes for folded (p {pv(BC['folded']['tests']['chi2_raw_mannwhitney_p'])}) "
        f"and partly disordered proteins (p {pv(BC['partly disordered']['tests']['chi2_raw_mannwhitney_p'])}; here the median χ² is higher with a structure, "
        f"{f1(BC['partly disordered']['pdb50']['chi2_raw_median'])} against {f1(BC['partly disordered']['under50']['chi2_raw_median'])}), and a rank test stratified by class finds no difference "
        f"in raw χ² (van Elteren p {pv(st['chi2_raw']['p_two_sided'])}) or in the Rg ratio (p {pv(st['rg_ratio']['p_two_sided'])}). NRMSD does not differ even pooled "
        f"(p {pv(pt['nrmsd_raw_mannwhitney_p'])}).</p>"
        f"<p>Among disordered proteins, raw χ² is lower when a structure exists (median {f1(dz['pdb50']['chi2_raw_median'])} against {f1(dz['under50']['chi2_raw_median'])}, "
        f"p {pv(dz['tests']['chi2_raw_mannwhitney_p'])}, n = {dz['pdb50']['n']} against {dz['under50']['n']}), but the difference does not survive a Holm correction over the six "
        f"within-class tests (p {pv(dz['tests']['chi2_raw_holm_p'])}) and is absent under NRMSD (p {pv(dz['tests']['nrmsd_raw_mannwhitney_p'])}). At the median these ensembles remain "
        f"larger than measured: Rg ratio {f2(dz['pdb50']['rg_ratio_median'])} with a structure (95% interval {ci(dz['pdb50']['rg_ratio_ci95'])}, which reaches 1) and "
        f"{f2(dz['under50']['rg_ratio_median'])} ({ci(dz['under50']['rg_ratio_ci95'])}) without; the two do not differ (p {pv(dz['tests']['rg_ratio_mannwhitney_p'])}). Whether the construct has an experimental structure does not explain BioEmu-1's "
        f"raw error beyond what the disorder class explains.</p>")

    space_html = f"""
<h3 id="space">Where the SAXS archive sits against the training data and CATH</h3>
{lead}
{findings}
{fig("fig_coverage_space", "Figure 7 · fold coverage of the SAXS set",
     f"A: CATH {ver} superfamilies ranked by their number of S35 clusters, in bins; the bar is the share present in the SAXS set (a CATH-Gene3D domain inside a construct), with Wilson 95% intervals and counts. B: CATH class composition of the {comp['n_units']} unique protein–superfamily units in the set against the CATH S35 clusters, with bootstrap intervals over proteins; the few-secondary-structure class (about 1% of both) is not drawn.",
     "In A a set that sampled fold space evenly by cluster would show bars of similar height. In B a ratio above 1 marks a class more frequent in the SAXS set than among solved domains.")}
{fig("fig_coverage_sources", "Figure 8 · organism and domain architecture",
     "A: source organism of the SAXS profiles (with Wilson 95% intervals), of PDB 30% clusters of chains of at least 30 residues, and of Swiss-Prot entries. B: CATH-Gene3D domains per protein in the SAXS construct, in the full-length UniProt protein it was cut from, and in Swiss-Prot proteins.",
     None)}
{interp}
{details("Architectures, over- and under-represented superfamilies, class table", fig("fig_coverage_architecture", "CATH architectures", "Ratio of the SAXS share to the CATH S35 share for each architecture with a CATH share of at least 1% or at least three units in the set; units in brackets; intervals from the bootstrap over proteins.") + cls_table + ou_note + ou)}
{details("Length, disorder and organism tables", len_table + dis_table + org_table)}
{details("Mapping, definitions and database versions", mapping_html)}
"""
    split_html = f"""
<h3 id="solved">Does a solved structure matter?</h3>
{split_prose}
{split_table}
{fig("fig_structure_split", "Figure 11 · raw χ² by availability of an experimental structure, within each class",
     "One point per profile, raw reduced χ² of the BioEmu-1 ensemble (log scale), by disorder class and by PDB coverage of the construct: at least half (filled), partial (light) and none (open). Bars: group median and its 95% bootstrap interval (groups of five or more). The p value above each class compares at least half covered against the rest.",
     "Within a class, a solved structure would help if the filled group sat lower than the other two.")}
"""
    return {"space": space_html, "split": split_html}


def training_section(n_bioemu: int) -> dict:
    """The training-data parts of 'What should be measured next?': the composition of BioEmu-1's training data
    (a), the closeness of the measured proteins to it and the error by closeness (c), and the categories of
    protein data to collect (d). Every number from results/coverage/, with a check for every directional
    sentence."""
    T = json.load(open(RESULTS / "coverage" / "training_summary.json"))
    TC = json.load(open(RESULTS / "coverage" / "training_closeness.json"))
    CE = json.load(open(RESULTS / "coverage" / "category_evidence.json"))
    NN = pd.read_csv(RESULTS / "coverage" / "training_nearest.csv")
    CV = json.load(open(RESULTS / "coverage" / "summary.json"))
    cm, comp, dis, org, L = T["components"], T["composition"], T["disorder"], T["organism"], T["length"]
    sc, ov = T["superfamily_coverage"], T["overlap"]
    MOST, LONG = "mostly disordered", "long disordered region"
    check(ov["n"] == n_bioemu and TC["n"] == n_bioemu and CE["n"] == n_bioemu, "training analysis covers the clean profiles")
    check(len(NN) == 439 and int(NN["clean"].astype(str).eq("True").sum()) == n_bioemu, "nearest-neighbour table has all 439")
    check(cm["afdb"] == T["sets"]["AFDB"] and cm["cath1"] + cm["cath2"] == T["sets"]["CATH MD"], "set sizes agree")
    # -- (a) composition
    a_m, a_l = dis["AFDB"][MOST]["share"], dis["AFDB"][LONG]["share"]
    m_m, m_l = dis["CATH MD"][MOST]["share"], dis["CATH MD"][LONG]["share"]
    s_m, s_l = dis["SAXS"][MOST]["share"], dis["SAXS"][LONG]["share"]
    c_m = dis["CATH S35"][MOST]["share"]
    check(s_m > 5 * a_m and s_m > 20 * m_m and m_m + m_l < 0.01 and s_l > 2 * a_l, "disorder: thin in training, frequent in SAXS")
    check(a_m < 2 * c_m, "AFDB inputs no more often mostly disordered than twice the CATH S35 share")
    check(L["CATH MD"]["max"] < 200 and L["MEGAscale"]["max"] < 100 and L["SAXS"]["max"] <= 500 and L["AFDB"]["share_gt500"] > 0.15,
          "simulated chains below 200, constructs at most 500, a fifth of AFDB inputs above 500")
    hu = {k: org[k]["Human"]["share"] for k in ("SAXS", "AFDB", "CATH MD", "PDB 30%")}
    ob = {k: org[k]["Other bacteria"]["share"] for k in ("SAXS", "AFDB", "CATH MD", "PDB 30%")}
    check(hu["AFDB"] < 0.01 and hu["SAXS"] > 2 * hu["CATH MD"] and ob["AFDB"] > 0.75 and ob["AFDB"] > 2 * ob["PDB 30%"],
          "AFDB clusters bacterial, human rare; SAXS human-rich")
    cls = {k: {c: comp[k]["class"][c]["share"] for c in ("Mainly alpha", "Mainly beta", "Alpha beta")} for k in comp}
    check(cls["AFDB"]["Mainly beta"] < 0.6 * cls["CATH S35"]["Mainly beta"] < cls["SAXS"]["Mainly beta"]
          and cls["CATH MD"]["Mainly beta"] < cls["CATH S35"]["Mainly beta"], "mainly beta thin in training, rich in SAXS")
    ross = {k: comp[k]["folds"]["3.40.50 Rossmann fold"]["share"] for k in comp}
    tim = {k: comp[k]["folds"]["3.20.20 TIM barrel"]["share"] for k in comp}
    check(ross["AFDB"] > ross["CATH S35"] and ross["CATH MD"] > ross["CATH S35"] > ross["SAXS"], "Rossmann folds not thin in training")
    check(tim["CATH MD"] == 0 and tim["AFDB"] < tim["CATH S35"], "TIM barrels absent from MD, below CATH in AFDB")
    si = sc["saxs_in"]
    n_none = sc["saxs_superfamilies"] - si["any training component"]["k"]
    check(sc["saxs_superfamilies"] == CV["coverage"]["gene3d"]["distinct_superfamilies"], "215 SAXS superfamilies")
    check(len(sc["saxs_not_in_training"]) == n_none, "superfamilies outside training counted")
    purpose = ("<p class=\"lead\">What BioEmu-1 was trained on, what the SAXS archive covers and where the two leave gaps are the evidence "
               "for which protein data to collect. This section measures all three, relates BioEmu-1's error to closeness to its training "
               "data, and then lists the categories of proteins the evidence supports, before the two arms of the campaign.</p>")
    training_lead = (
        f"<p>BioEmu-1 learned in three stages {cite('lewis')}. Pre-training used {cm['afdb']:,} AlphaFold Database sequence clusters "
        f"({cm['afdb_member']:,} predicted structures; the cluster definitions are published with the model), from which clusters whose structures were all more than half coil had been removed. "
        f"Fine-tuning used molecular dynamics: {cm['cath1'] + cm['cath2']:,} CATH domains of up to {L['CATH MD']['max']} residues, "
        f"{cm['megasim']} MEGAscale wild types with {cm['megasim_mutants']:,} point mutants, {cm['octapeptides']:,} octapeptides, and published simulations of a few named proteins "
        f"(DDR1 kinases, SETD8, coronavirus proteins, MHC class II, barnase–barstar). A last stage fitted measured folding stabilities of 361 MEGAscale wild-type domains and their mutants {cite('tsuboyama')}. "
        f"The DESRES fast-folder simulations trained a separate model only. I read the sequences of each part (the simulated topologies, the cluster members from UniProt) and described them as I described the SAXS set: "
        f"CATH superfamilies, chain length, disorder predicted from sequence by one predictor for every set {cite('emenecker')}, and organism.</p>")
    training_facts = (
        '<ul class="facts">'
        f'<li><strong>Disorder is nearly absent from the training data.</strong> {p1(a_m)} of the AFDB input chains are predicted mostly disordered and {p1(a_l)} carry a disordered run of at least 30 residues; '
        f'of the simulated CATH domains, {p1(m_m)} and {p1(m_l)}. The SAXS constructs: {p1(s_m)} and {p1(s_l)}.</li>'
        f'<li><strong>Simulated chains are short.</strong> The simulated CATH domains have a median of {L["CATH MD"]["median"]:.0f} residues and none reaches 200; the MEGAscale domains reach {L["MEGAscale"]["max"]}. '
        f'The AFDB inputs are full-length chains (median {L["AFDB"]["median"]:.0f}), {pct(L["AFDB"]["share_gt500"])} of them longer than 500 residues, each seen as predicted structures without dynamics. No SAXS construct exceeds 500 residues.</li>'
        f'<li><strong>Pre-training is bacterial.</strong> {pct(ob["AFDB"])} of the AFDB clusters come from bacteria other than <em>E. coli</em> and {p1(hu["AFDB"])} from human proteins; '
        f'human domains make up {pct(hu["CATH MD"])} of the simulated set and {pct(hu["SAXS"])} of the SAXS profiles.</li>'
        f'<li><strong>Mainly-β domains are thin, α/β enzyme folds are not.</strong> Mainly-β domains are {pct(cls["AFDB"]["Mainly beta"])} of the AFDB units and {pct(cls["CATH MD"]["Mainly beta"])} of the simulated ones, against {pct(cls["CATH S35"]["Mainly beta"])} of CATH and {pct(cls["SAXS"]["Mainly beta"])} of the SAXS set. '
        f'The Rossmann fold is {pct(ross["AFDB"])} of AFDB units and {pct(ross["CATH MD"])} of simulated ones, above CATH ({pct(ross["CATH S35"])}); the TIM barrel is {p1(tim["AFDB"])} of AFDB units and absent from the simulations ({p1(tim["CATH S35"])} of CATH).</li>'
        f'<li><strong>Most of the archive\'s superfamilies occur in training, few in the simulations.</strong> Of the {sc["saxs_superfamilies"]} CATH superfamilies in the SAXS constructs, {si["AFDB"]["k"]} occur in the AFDB clusters, '
        f'{si["MD (CATH MD and MegaSim)"]["k"]} in the simulated domains and {si["any training component"]["k"]} in some part of the training data; {n_none} occur in none.</li></ul>')
    comp_fig = fig("fig_training_composition", "Figure 6 · BioEmu-1's training data, the SAXS set and the CATH reference",
                   f"A: CATH class of protein–superfamily units: SAXS constructs (CATH-Gene3D), AFDB pre-training clusters (CATH-Gene3D of the input chain, {comp['AFDB']['n_units']:,} units), simulated CATH domains ({comp['CATH MD']['n_units']:,}) and CATH S35 clusters. "
                   "B: chain length. C: predicted disorder (metapredict), share of chains mostly disordered and share with a disordered run of at least 30 residues. D: source organism; the reference is the PDB 30% clusters. MEGAscale domains are in the tables below.",
                   "A bar of the SAXS set far from the AFDB and simulated bars marks a region the archive measures and the training data barely contain.")
    sets_t = ("SAXS", "AFDB", "CATH MD", "MEGAscale", "CATH S35")
    names_t = {"SAXS": "SAXS constructs", "AFDB": "AFDB clusters", "CATH MD": "simulated CATH domains", "MEGAscale": "MEGAscale domains", "CATH S35": "CATH S35"}
    rows_c = "".join(f'<tr><td>{names_t[k]}</td><td class="num">{T["sets"][k]:,}</td><td class="num">{comp[k]["n_superfamilies"]:,}</td>'
                     + "".join(f'<td class="num">{pct(comp[k]["class"][c]["share"])}</td>' for c in ("Mainly alpha", "Mainly beta", "Alpha beta"))
                     + f'<td class="num">{p1(ross[k])}</td><td class="num">{p1(tim[k])}</td>'
                     + f'<td class="num">{L[k]["median"]:.0f}</td><td class="num">{p1(L[k]["share_gt500"])}</td>'
                     + f'<td class="num">{p1(dis[k][MOST]["share"])}</td><td class="num">{p1(dis[k][LONG]["share"])}</td></tr>' for k in sets_t)
    comp_table = ('<div class="scroll"><table class="wide"><thead><tr><th>set</th><th>chains</th><th>superfamilies</th><th>mainly α</th><th>mainly β</th><th>α/β</th>'
                  '<th>Rossmann</th><th>TIM barrel</th><th>median length</th><th>&gt; 500</th><th>mostly disordered</th><th>disordered run ≥ 30</th></tr></thead><tbody>'
                  + rows_c + '</tbody></table></div>'
                  f'<p class="note">Class and fold shares are over protein–superfamily units (CATH S35: clusters). MEGAscale domains: the {cm["megascale"]} curated wild types of the MEGAscale study and the MegaSim wild types, one row per distinct sequence, superfamilies from the best sequence hit in CATH; '
                  f'{pct(org["MEGAscale"]["Designed"]["share"])} of them are de novo designs. Octapeptides ({cm["octapeptides"]:,} systems, 8 residues) are counted and not compared.</p>')
    org_rows = "".join(f'<tr><td>{g}</td>' + "".join(f'<td class="num">{p1(org[k][g]["share"])}</td>' for k in ("SAXS", "AFDB", "CATH MD", "MEGAscale", "PDB 30%")) + '</tr>'
                       for g in ("Human", "Mouse", "E. coli", "Yeast", "Other bacteria", "Viruses", "Plants", "Other", "Designed"))
    org_table = ('<div class="scroll"><table><thead><tr><th>organism</th><th>SAXS</th><th>AFDB</th><th>simulated CATH</th><th>MEGAscale</th><th>PDB 30%</th></tr></thead><tbody>'
                 + org_rows + '</tbody></table></div>')
    sf_rows = "".join(f'<tr><td>{k}</td><td class="num">{v["k"]} of {v["n"]}</td></tr>' for k, v in si.items())
    rev = sc["training_in_saxs"]
    sf_table = ('<div class="scroll"><table><thead><tr><th>SAXS-set superfamilies found in</th><th>superfamilies</th></tr></thead><tbody>' + sf_rows + '</tbody></table></div>'
                f'<p class="note">The other way round, the SAXS set contains {rev["AFDB"]["k"]} of the {rev["AFDB"]["n"]:,} superfamilies of the AFDB clusters ({p1(rev["AFDB"]["share"])}) and {rev["CATH MD"]["k"]} of the {rev["CATH MD"]["n"]} of the simulated domains ({pct(rev["CATH MD"]["share"])}).</p>')
    training_html = (f'<h3 id="training">What BioEmu-1 was trained on</h3>{training_lead}{training_facts}{comp_fig}'
                     + details("Composition, organism and superfamily tables", comp_table + org_table + sf_table))

    # -- (c) closeness
    al = ov["at_least"]
    bc = ov["bins_by_class"]
    dis_lt30 = bc["disordered"]["<30"]
    n_dis = sum(bc["disordered"].values())
    md = ov["by_component"]["md"]
    check(al["30"]["share"] < 0.5 and al["90"]["share"] < 0.1, "most constructs far from training data")
    check(md["30"]["k"] < al["30"]["k"] / 4, "simulated systems rarely close")
    check(al["50"]["share"] < 0.2, "below one in five at 50%")
    check(dis_lt30 / n_dis > 0.7 > bc["folded"]["<30"] / sum(bc["folded"].values()), "disordered constructs further from training data")
    near = [r for r in ov["near_identical"] if r["clean"]]
    check(len(near) == al["90"]["k"], "near-identical list complete")
    comp_name = {"afdb_member": "AFDB cluster member", "cath2": "simulated CATH domain", "megasim": "MegaSim wild type",
                 "megascale": "MEGAscale domain", "other_md": "named MD protein", "cath1": "simulated CATH domain"}
    n_near_corona = sum(1 for r in near if r["component"] == "other_md")
    check(all("sarscov2" in r["hit"] for r in near if r["component"] == "other_md"), "near-identical MD proteins are coronavirus proteins")
    acc = ov["uniprot_in_afdb_clusters"]
    sp, spc = TC["spearman_pooled"], TC["spearman_by_class"]
    P = TC["pooled"]
    check(P[">=90"]["raw_fit"]["share"] > P["<30"]["raw_fit"]["share"] and sp["raw_fit"]["p"] < 0.05 and sp["rg_ratio"]["p"] < 0.05
          and sp["chi2_raw"]["p"] > 0.05, "pooled: raw-fit share and Rg ratio trend with identity, chi2 not significant")
    within = [spc[c][k] for c in CLASSES for k in ("chi2_raw", "raw_fit", "nrmsd_raw", "rg_ratio")]
    min_holm = min(x["holm_p"] for x in within)
    min_raw = min(within, key=lambda x: x["p"])
    min_key = next((c, k) for c in CLASSES for k in ("chi2_raw", "raw_fit", "nrmsd_raw", "rg_ratio") if spc[c][k] is min_raw)
    check(min_holm > 0.05, "no within-class association after Holm")
    strat = TC["stratified_identity_ge50"]["chi2_raw"]["p_two_sided"]
    check(strat > 0.05, "stratified test null")
    sg, sgm = TC["sf_group"], TC["sf_group_md"]
    check(sg["pooled_tests"]["chi2_raw_mannwhitney_p"] > 0.05 and sg["stratified_by_class"]["chi2_raw"]["p_two_sided"] > 0.05
          and sgm["pooled_tests"]["chi2_raw_mannwhitney_p"] > 0.05 and sgm["stratified_by_class"]["chi2_raw"]["p_two_sided"] > 0.05,
          "superfamily presence: no difference")
    check(all(sg["by_class"][c]["tests"].get("chi2_raw_holm_p", 1) > 0.05 and sgm["by_class"][c]["tests"].get("chi2_raw_holm_p", 1) > 0.05 for c in CLASSES),
          "superfamily presence: none within class")
    n_dis_close = n_dis - dis_lt30
    close_lead = (
        f"<p>For each construct I searched every part of the training data for the most similar sequence over a local alignment that covers at least half of the construct {cite('steinegger')}, "
        f"and checked whether the construct's CATH superfamilies occur there. BioEmu-1's training data were filtered against the paper's own test proteins only, so overlap with these proteins was possible.</p>")
    close_facts = (
        '<ul class="facts">'
        f'<li><strong>Most measured proteins are far from the training data.</strong> Of the {ov["n"]} profiles, {al["30"]["k"]} ({pct(al["30"]["share"])}) have a training sequence of at least 30% identity, '
        f'{al["50"]["k"]} ({pct(al["50"]["share"])}) of at least 50%, {al["90"]["k"]} ({pct(al["90"]["share"])}) of at least 90% and {al["100"]["k"]} an identical stretch covering at least half the construct. '
        f'The close neighbours are AFDB cluster members and named proteins of the published simulations ({n_near_corona} of the {len(near)} at 90% or more, all coronavirus proteins); '
        f'only {md["30"]["k"]} constructs reach 30% to a simulated CATH domain or MegaSim wild type, and {acc["k"]} of {acc["n"]} UniProt accessions are AFDB cluster members.</li>'
        f'<li><strong>Disordered proteins are the furthest.</strong> {dis_lt30} of the {n_dis} disordered constructs have no training sequence at 30% identity.</li>'
        f'<li><strong>Superfamilies:</strong> the construct carries a CATH superfamily that occurs in the training data for {sg["pooled"]["present"]["n"]} profiles, one that does not for {sg["pooled"]["absent"]["n"]}, and none for {sg["pooled"]["none"]["n"]}; '
        f'restricted to the simulated superfamilies, {sgm["pooled"]["present"]["n"]} and {sgm["pooled"]["absent"]["n"]}.</li></ul>')
    overlap_fig = fig("fig_training_overlap", "Figure 9 · how close the measured proteins are to BioEmu-1's training data",
                      f"A: share of the {ov['n']} profiles whose nearest training sequence (local alignment covering at least half of the construct) reaches each identity, for all training data, the AFDB cluster members and the simulated systems; constructs without such an alignment count as below every threshold. B: the profiles by identity bin and disorder class.",
                      "A curve that stays high to the right would mean many measured proteins were nearly in the training data; at 50% identity the share is below one in five.")
    BCc = TC["by_class"]
    close_prose = (
        f"<p>Pooled, the raw fit improves with closeness: χ² ≤ 2 is met by {pct(P['<30']['raw_fit']['share'])} of profiles below 30% identity and {pct(P['>=90']['raw_fit']['share'])} at 90% or more "
        f"(Spearman ρ {sp['raw_fit']['rho']:+.2f}, p {pv(sp['raw_fit']['p'])}), and the Rg ratio falls slightly (ρ {sp['rg_ratio']['rho']:+.2f}, p {pv(sp['rg_ratio']['p'])}); raw χ² itself does not reach significance (p {pv(sp['chi2_raw']['p'])}). "
        f"The bins differ in disorder class: {dis_lt30} of the {P['<30']['n']} profiles below 30% are disordered and {bc['disordered']['>=90']} of the {P['>=90']['n']} at 90% or more. "
        f"Within classes no measure varies with closeness: across the {len(within)} within-class Spearman tests (three classes, raw χ², raw fit, NRMSD and Rg ratio) the smallest nominal p is {pv(min_raw['p'])} ({min_key[0]}, {'NRMSD' if min_key[1] == 'nrmsd_raw' else min_key[1].replace('_', ' ')}) and every Holm-adjusted p is at least {f2(min_holm)}; "
        f"a rank test stratified by class finds no difference in raw χ² between constructs with a training neighbour of at least 50% identity and the rest (van Elteren p {pv(strat)}). "
        f"Superfamily presence does not separate the errors either (pooled Mann–Whitney p {pv(sg['pooled_tests']['chi2_raw_mannwhitney_p'])}; restricted to simulated superfamilies p {pv(sgm['pooled_tests']['chi2_raw_mannwhitney_p'])}, stratified by class p {pv(sgm['stratified_by_class']['chi2_raw']['p_two_sided'])}).</p>"
        f"<p>In this archive, closeness to the training data does not explain BioEmu-1's raw error beyond the disorder class. The over-extended disordered ensembles are also the proteins furthest from the training data, "
        f"and the archive cannot separate the two: only {n_dis_close} disordered constructs have a training neighbour at 30% identity or more.</p>")
    close_fig = fig("fig_closeness", "Figure 10 · BioEmu-1's error by closeness to the training data, within each class",
                    "One point per profile: raw reduced χ² (top, log scale) and the ratio of ensemble Rg to measured Rg (bottom), by the identity of the nearest training sequence, within each disorder class. Bars: group medians (groups of three or more); dashed lines χ² = 2 and ratio 1.",
                    "If closeness to the training data mattered, the points would step down (top) or towards 1 (bottom) from left to right within a class.")

    def bin_row(name, r):
        if not r["n"]:
            return f'<tr><td>{name}</td><td class="num">0</td><td class="num">–</td><td class="num">–</td><td class="num">–</td></tr>'
        return (f'<tr><td>{name}</td><td class="num">{r["n"]}</td><td class="num">{pct(r["raw_fit"]["share"])} ({pct(r["raw_fit"]["ci95"][0])}–{pct(r["raw_fit"]["ci95"][1])})</td>'
                f'<td class="num">{f1(r["chi2_raw_median"])}</td><td class="num">{f2(r["rg_ratio_median"])}</td></tr>')
    BL = {"<30": "below 30%", "30-50": "30–50%", "50-90": "50–90%", ">=90": "90% or more"}
    brows = '<tr class="grp"><td colspan="5">all classes</td></tr>' + "".join(bin_row(BL[b], P[b]) for b in TC["bins"])
    for c in CLASSES:
        t = spc[c]
        brows += (f'<tr class="grp"><td colspan="5">{c} · Spearman with identity: χ² p {pv(t["chi2_raw"]["p"])}, Rg ratio p {pv(t["rg_ratio"]["p"])} (Holm {f2(t["chi2_raw"]["holm_p"])}, {f2(t["rg_ratio"]["holm_p"])})</td></tr>'
                  + "".join(bin_row(BL[b], BCc[c][b]) for b in TC["bins"]))
    bin_table = ('<div class="scroll"><table class="wide"><thead><tr><th>nearest training identity</th><th>n</th><th>χ² ≤ 2 (95%)</th><th>median raw χ²</th><th>median Rg ratio</th></tr></thead><tbody>'
                 + brows + '</tbody></table></div>')
    sg_rows = ""
    for key, nm in (("sf_group", "any part of the training data"), ("sf_group_md", "the simulated domains")):
        g = TC[key]
        for grp, lab in (("present", "superfamily present"), ("absent", "superfamily absent"), ("none", "no CATH superfamily")):
            r = g["pooled"][grp]
            sg_rows += (f'<tr><td>{nm}</td><td>{lab}</td><td class="num">{r["n"]}</td><td class="num">{pct(r["raw_fit"]["share"])}</td>'
                        f'<td class="num">{f1(r["chi2_raw_median"])}</td><td class="num">{f2(r["rg_ratio_median"])}</td></tr>')
    sg_table = ('<div class="scroll"><table><thead><tr><th>superfamily in</th><th>group</th><th>n</th><th>χ² ≤ 2</th><th>median raw χ²</th><th>median Rg ratio</th></tr></thead><tbody>'
                + sg_rows + '</tbody></table></div><p class="note">Present against absent, two-sided Mann–Whitney on raw χ², NRMSD and Rg ratio, pooled and within each class (Holm over the within-class tests with both groups present).</p>')
    near_rows = "".join(f'<tr><td><a href="https://www.sasbdb.org/data/{r["label"]}/">{r["label"]}</a></td><td class="num">{r["construct_length"]}</td><td>{r["disorder_class"]}</td>'
                        f'<td>{comp_name.get(r["component"], r["component"])}</td><td>{r["hit"].split(":")[-1] if r["component"] != "other_md" else r["hit"].split(":", 1)[1]}</td>'
                        f'<td class="num">{pct(r["identity"])}</td><td class="num">{pct(r["coverage"])}</td></tr>' for r in near)
    near_table = ('<div class="scroll"><table class="wide"><thead><tr><th>SASBDB</th><th>residues</th><th>class</th><th>nearest training set</th><th>sequence</th><th>identity</th><th>construct covered</th></tr></thead><tbody>'
                  + near_rows + '</tbody></table></div>'
                  '<p class="note">Profiles with a training sequence of at least 90% identity over at least half of the construct. AFDB sequences are named by UniProt accession; the named MD proteins are full UniProt chains, so their match can extend beyond what was simulated.</p>')
    closeness_html = (f'<h3 id="closeness">How close are these proteins to BioEmu-1\'s training data?</h3>{close_lead}{close_facts}{overlap_fig}{close_prose}{close_fig}'
                      + details("Error by identity bin and by superfamily presence", bin_table + sg_table)
                      + details("The near-identical cases", near_table))

    # -- (d) which protein data to collect
    cat = CE["categories"]
    dm, dl, big = cat["mostly disordered (metapredict)"], cat["long disordered region, under half"], cat["construct over 500 residues"]
    check(dm["in"]["raw_fit"]["share"] < 0.5 * dm["rest"]["raw_fit"]["share"] and dm["in"]["rg_ratio_median"] > 1.1 > dm["rest"]["rg_ratio_median"]
          and dm["rg_ratio_mannwhitney_p"] < 0.001, "row 1: disordered: worse fit, too extended")
    check(dl["in"]["raw_fit"]["share"] < dl["rest"]["raw_fit"]["share"] and dl["chi2_raw_mannwhitney_p"] < 0.05 and dl["rg_ratio_mannwhitney_p"] > 0.05,
          "row 2: long disordered region: higher chi2, size as the rest")
    check(big["in"]["n"] == 0 and CE["max_construct_length"] <= 500, "row 3: no construct above 500")
    dropped = {"alpha/beta enzyme folds": "α/β enzyme folds (Rossmann, TIM barrel)", "bacteria other than E. coli": "bacteria other than <em>E. coli</em>",
               "superfamily beyond the top 200": "superfamilies beyond the 200 most populated", "no experimental structure": "proteins with no experimental structure",
               "human proteins": "human proteins", "mainly-beta domains": "mainly-β domains",
               "two or more CATH superfamilies": "constructs with two or more domains (up to 500 residues)"}
    for k in dropped:
        c_ = cat[k]
        check((c_["chi2_raw_mannwhitney_p"] > 0.05 or c_["in"]["chi2_raw_median"] < c_["rest"]["chi2_raw_median"])
              and c_["in"]["raw_fit"]["share"] >= c_["rest"]["raw_fit"]["share"] - 0.12, f"dropped category {k}: error not larger than the rest")

    def err(c):
        r, q = c["in"], c["rest"]
        return (f'{pct(r["raw_fit"]["share"])} meet χ² ≤ 2 (rest {pct(q["raw_fit"]["share"])}); Rg ratio {f2(r["rg_ratio_median"])} (rest {f2(q["rg_ratio_median"])}); n = {r["n"]}')
    mb = T["multi_superfamily"]
    rows_d = [
        ("Disordered proteins (at least half of the residues predicted disordered)",
         f"thin: {p1(a_m)} of AFDB inputs, {p1(m_m)} of simulated domains",
         f"{dm['in']['n']} profiles ({pct(dm['in']['n'] / CE['n'])})", err(dm),
         "the size error: ensembles too extended, read from the Guinier region",
         "usually soluble and monomeric; low mass gives weak scattering, so a concentration series matters"),
        ("Folded proteins with a long disordered region (a run of at least 30 residues, under half the chain)",
         f"thin: {p1(a_l)} of AFDB inputs, {p1(m_l)} of simulated domains",
         f"{dl['in']['n']} profiles ({pct(dl['in']['n'] / CE['n'])})", err(dl),
         f"a shape error (raw χ² higher, Mann–Whitney p {pv(dl['chi2_raw_mannwhitney_p'])}) with the size close to measured",
         "tails and linkers are prone to proteolysis; intact mass by SDS-PAGE or mass spectrometry before measuring"),
        ("Full-length multidomain proteins above 500 residues",
         f"as predicted structures only: {pct(L['AFDB']['share_gt500'])} of AFDB inputs exceed 500 residues, {pct(mb['AFDB']['share'])} carry two or more superfamilies; no simulated chain exceeds {L['CATH MD']['max']} residues",
         f"none: the longest construct has {CE['max_construct_length']} residues",
         "not measured: no profile in the archive",
         "domain arrangement and interdomain flexibility, at low resolution",
         "expression and stability of long constructs; size-exclusion-coupled SAXS to separate aggregates"),
    ]
    collect_rows = "".join(f'<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e}</td><td>{f}</td></tr>' for a, b, c, d, e, f in rows_d)
    collect_table = ('<div class="scroll"><table class="wide collect"><thead><tr><th>protein category</th><th>in BioEmu-1\'s training data</th><th>in the SAXS archive</th>'
                     '<th>BioEmu-1\'s raw SAXS error here</th><th>what SAXS shows</th><th>feasibility</th></tr></thead><tbody>' + collect_rows + '</tbody></table></div>'
                     '<p class="note">Disorder of the training chains and of the constructs from the same sequence predictor (metapredict); "rest" is every other clean profile. Feasibility notes are judgement.</p>')

    def drop_row(k):
        c = cat[k]
        return (f'<tr><td>{dropped[k]}</td><td class="num">{c["in"]["n"]}</td><td class="num">{pct(c["in"]["raw_fit"]["share"])} / {pct(c["rest"]["raw_fit"]["share"])}</td>'
                f'<td class="num">{f2(c["in"]["rg_ratio_median"])}</td><td class="num">{pv(c["chi2_raw_mannwhitney_p"])}</td></tr>')
    drop_train = (f'<p class="note">Training side of the dropped categories: Rossmann folds are {pct(ross["AFDB"])} of AFDB units and {pct(ross["CATH MD"])} of simulated ones against {pct(ross["CATH S35"])} of CATH; '
                  f'{pct(ob["AFDB"])} of AFDB clusters are from bacteria other than <em>E. coli</em>; superfamilies beyond the 200 most populated are {pct(comp["AFDB"]["rank_beyond_200"]["share"])} of AFDB units and {pct(comp["CATH MD"]["rank_beyond_200"]["share"])} of simulated ones ({pct(comp["CATH S35"]["rank_beyond_200"]["share"])} of CATH); '
                  f'the AFDB clusters need no experimental structure; human proteins are {p1(hu["AFDB"])} of AFDB clusters and {pct(hu["CATH MD"])} of the simulated domains; '
                  f'mainly-β domains are {pct(cls["AFDB"]["Mainly beta"])} of AFDB units and {pct(cls["CATH MD"]["Mainly beta"])} of simulated ones ({pct(cls["CATH S35"]["Mainly beta"])} of CATH); '
                  f'{pct(mb["AFDB"]["share"])} of AFDB inputs and no simulated domain carry two or more superfamilies.</p>')
    drop_table = ('<div class="scroll"><table><thead><tr><th>category</th><th>SAXS profiles</th><th>χ² ≤ 2, in / rest</th><th>median Rg ratio</th><th>raw χ², Mann–Whitney p</th></tr></thead><tbody>'
                  + "".join(drop_row(k) for k in dropped) + '</tbody></table></div>' + drop_train)
    collect_prose = (
        "<p>Each row joins three measurements: how much of the category the training data contain, how much the archive measures, and BioEmu-1's raw error on it here. "
        "A category is kept when the training data are thin in it and either the error on it exceeds the error on the rest of the archive or the archive holds no measurement of it. "
        "The order is my judgement from that evidence: the first row combines the thinnest training data with the largest error and a direction SAXS reads directly; the third is a gap in both the simulations and the archive, so its error is unknown. "
        "I give no numeric priority: the heuristic priority score tested above picked high-discrepancy profiles no better than random choice.</p>")
    dropped_names = ", ".join(dropped[k] for k in ("alpha/beta enzyme folds", "bacteria other than E. coli", "superfamily beyond the top 200", "no experimental structure"))
    collect_drop = (f"<p>Four candidates from the coverage analysis are not supported: {dropped_names}. Either the training data are not thin in them "
                    f"(the Rossmann fold is more frequent in pre-training and simulation than in CATH; {pct(ob['AFDB'])} of the pre-training clusters are bacterial) "
                    "or BioEmu-1's raw error on them matches or is below the error on the rest of the archive. Human proteins and mainly-β domains are thin in pre-training, "
                    "but the error on them matches the rest; so does the error on constructs with two or more domains up to 500 residues.</p>")
    use_prose = (
        f"<p><strong>How such data would be used.</strong> BioEmu-1's last training stage fitted measured folding stabilities by property-prediction fine-tuning: a foldedness observable computed from sampled structures is matched to the measurement by back-propagating through the sampler {cite('lewis')}. "
        "A SAXS observable could enter the same way, as the radius of gyration of the sampled conformers or, through a differentiable forward model, as the scattering curve. "
        "Reweighting the ensemble toward a measured profile, as in the reweighting section, gives a second route: the reweighted conformers serve as conformer-level targets. "
        "Neither route was tried in this study; the rows above are where either would add information the training data lack.</p>")
    collect_html = (f'<h3 id="collect">Which protein data to collect</h3>{collect_prose}{collect_table}{collect_drop}'
                    + details("The categories that were evaluated and dropped", drop_table) + use_prose)
    arms_link = ("Arm 2 tests the size bias in the category this analysis ranks first, disordered proteins; Arm 1 checks that deposited profiles reproduce under controlled conditions, "
                 "which any profile needs before it can serve as a training label.")
    limits = ("BioEmu-1's training data were measured from their public releases: the AFDB cluster definitions and the simulated systems are those published with the model, "
              "the 361 MEGAscale wild types used for fine-tuning are a subset of the curated set the paper does not list, and the other published simulations are represented by the full UniProt chains of the named proteins, an upper bound on their overlap. "
              "Disorder of the training chains is predicted from sequence, without examining the training structures, and AFDB superfamilies are the CATH-Gene3D assignments of the full chain.")
    return {"purpose": purpose, "training": training_html, "closeness": closeness_html, "collect": collect_html,
            "arms_link": arms_link, "limits": limits}


def build():
    n_bioemu = sum(RES.values())
    n_clean = A["n_clean"]
    n_entries = A["n_entries"]
    check(n_entries == 439 and n_bioemu == 399 and n_clean == 403, "denominators: 439 archive, 403 clean, 399 with BioEmu-1")
    share_fit = B["share_raw_fit_chi2_2"]
    fits, mod, strong, notfit = (RES.get(k, 0) for k in (KIND_RAW, KIND_MODEST, KIND_STRONG, KIND_NOTFIT))
    check(fits + mod + strong + notfit == n_bioemu, "kind counts sum")
    check(KIND_NOTFIT == "target fit not reached", "fourth kind is named 'target fit not reached'")
    check(abs(fits / n_bioemu - share_fit) < 1e-9, "raw-fit count and share agree")
    comps = {c["model"]: c for c in A["model_comparison"]}
    gens = [m for m in GENERATORS if m in comps]
    singles = [m for m in SINGLE if m in comps]
    check(len(comps) == 6 and set(comps) == set(GENERATORS) | set(SINGLE), "six comparators")
    try:                                     # the page's BioEmu-1 colour is the figures' role colour
        from bsc.report import PIXEL_DENSITY as FIG_DENSITY
        from bsc.report import ROLE_COL
        check(f"--bioemu:{ROLE_COL['bioemu']};" in CSS, "page BioEmu-1 colour matches the figure role colour")
        check(FIG_DENSITY == PIXEL_DENSITY, "page and figures agree on the pixel density")
    except ImportError:
        pass
    RV = pd.read_csv(RESULTS / "resolvability.csv")
    RV = RV[RV["model"] == "bioemu"].set_index("label")

    # ---- result 1: classes under two scores
    check(min(CLS, key=lambda k: CLS[k]["chi2_raw_median"]) == "folded"
          and max(CLS, key=lambda k: CLS[k]["chi2_raw_median"]) == "disordered", "class ordering of the raw error by chi2")
    check(E["E1"]["met"] and not E["E2"]["met"] and not E["E3"]["met"] and not E["E4"]["met"], "E1 met, E2–E4 not")
    check(E["E2"]["share_fit_le_100"] > E["E2"]["share_fit_gt_100"] and abs(E["E2"]["spearman_rho"]) < 0.2,
          "E2: a step at 100 residues, no slope")
    NR = H["nrmsd"]
    nr_cls = NR["by_class"]
    check(not NR["class_order_holds"] and NR["model_order_holds"] and NR["bioemu_first_under_nrmsd"],
          "NRMSD: class order does not hold, model order does")
    check(all(nr_cls[c]["nrmsd_raw_ci95"][0] <= nr_cls["folded"]["nrmsd_raw_median"] <= nr_cls[c]["nrmsd_raw_ci95"][1]
              or nr_cls["folded"]["nrmsd_raw_ci95"][0] <= nr_cls[c]["nrmsd_raw_median"] <= nr_cls["folded"]["nrmsd_raw_ci95"][1]
              for c in ("partly disordered", "disordered")), "NRMSD class medians overlap in their intervals")
    nr_med = [nr_cls[c]["nrmsd_raw_median"] for c in CLASSES]
    check(max(nr_med) / min(nr_med) < 1.2, "NRMSD class medians are similar (within 20%)")
    check(CLS["disordered"]["chi2_raw_median"] / CLS["folded"]["chi2_raw_median"] > 3, "chi2 medians differ several-fold")

    # ---- result 2: size
    rg_dis = E["E4"]["median_rg_ratio_disordered"]
    rg_ci = E["E4"]["ci95_median_rg_ratio"]
    check(rg_dis > 1 and rg_ci[0] > 1 and E["E4"]["p_two_sided"] < 0.05, "E4 reversed: ratio above 1 with interval and test")
    others_rg = {m: v for m, v in A["rg_ratio_disordered_by_model"].items() if m != "bioemu" and m in comps}
    check(all(v < rg_dis for v in others_rg.values()), "BioEmu-1 has the largest disordered Rg ratio of the seven")
    check(rg_dis > 1.1 and all(v <= 1.1 for v in others_rg.values()), "BioEmu-1 the only model above 1.1")
    trimmed = R["disordered_drop_5_most_extreme"]
    check(trimmed["median"] > 1 and trimmed["ci95"][0] > 1 and trimmed["p_two_sided_wilcoxon_log"] < 0.05, "trimmed E4 holds")
    collapsed = R["near_duplicates"]["collapsed_curve_rg_ratio_by_class"]["disordered"]
    check(collapsed["median"] > 1 and collapsed["ci95"][0] > 1, "E4 holds with near-duplicates collapsed")
    rho_len = R["disordered_ratio_vs_length"]
    check(rho_len["spearman_rho"] > 0 and rho_len["p"] < 0.05, "longer disordered chains are more over-extended")
    CR = R["coordinate_rg"]["by_class"]
    cr_d, cr_f = CR["disordered"], CR["folded"]
    check(cr_d["coord_over_exp_median"] > cr_d["curve_over_exp_median"] > 1 and cr_d["coord_over_exp_ci95"][0] > 1,
          "the size discrepancy is also present in coordinate-derived radii")
    check(cr_f["coord_over_exp_median"] < cr_f["curve_over_exp_median"] < 1, "folded: Cα radius below the curve radius")
    ext_curve, ext_coord = rg_dis - 1, cr_d["coord_over_exp_median"] - 1

    # ---- result 3: reweighting
    rew = mod + strong
    check(rew > strong > 0 and notfit > 0, "reweighting reaches the target for more entries, some only with phi < 0.5")
    KS = H["kind_counts_by_phi_threshold"]
    check(len({KS[t][KIND_RAW] for t in KS}) == 1 and len({KS[t][KIND_NOTFIT] for t in KS}) == 1,
          "phi threshold moves only the two reweightable kinds")
    rep = A["representative_path"]
    check(RV.loc[rep, "resolvability"] == KIND_STRONG, "the representative path is strongly reweightable")

    # ---- worked examples: nearest the class median raw chi2
    for cls in CLASSES:
        lab = A["examples"][cls]
        ex = EX[lab]
        g = RV[(RV["disorder_class"] == cls) & RV["clean"]]
        dev = (g["chi2_raw"] - g["chi2_raw"].median()).abs()
        check(abs(dev.loc[lab] - dev.min()) < 1e-9, f"{lab}: nearest the class median raw chi2")
        reached = RV.loc[lab, "resolvability"] in (KIND_RAW, KIND_MODEST, KIND_STRONG)
        check(ex["operating_reached"] == reached, f"{lab}: reweighted curve at the operating point iff chi2 <= 2 is reached")
        if reached and not ex["operating_is_raw"]:
            check(abs(ex["phi_operating"] - RV.loc[lab, "phi_at_chi2_2"]) < 1e-6 and ex["chi2"]["operating"] <= 2,
                  f"{lab}: operating phi equals the stored phi at chi2 = 2")
        else:
            check(abs(ex["chi2"]["operating"] - ex["chi2_minimum"]) < 1e-9, f"{lab}: not reached, curve at the chi2 minimum")
        lv = ex["density_levels"]
        check([x["name"] for x in lv] == ["particle", "protein"] and lv[0]["volume_A3"] > lv[1]["volume_A3"]
              and lv[0]["iso_over_max"] < lv[1]["iso_over_max"] and lv[0]["ca_inside"] >= lv[1]["ca_inside"]
              and abs(lv[1]["ca_inside"] - ex["ca_inside_envelope"]) < 1e-12,
              f"{lab}: two nested density levels; the protein level is the docking level")
    ex_f, ex_p, ex_d = (EX[A["examples"][c]] for c in CLASSES)
    lab_d = A["examples"]["disordered"]
    ratio_d = RV.loc[lab_d, "rg_ratio"]
    check(ex_d["rg_raw"] > ex_d["rg_operating"] > ex_d["rg_exp"] and ex_d["operating_reached"],
          "disordered example: larger than measured raw; reweighting to chi2 <= 2 moves Rg toward the measured value")
    check(1 < ratio_d < rg_dis, "disordered example: size ratio above 1 but below the class median (caption says so)")

    # ---- result 4: prediction and selection
    r2_best = max(P["regression"]["ridge"]["r2"]["mean"], P["regression"]["hgb"]["r2"]["mean"])
    auc_hgb = P["classification"]["hgb"]["auc"]["mean"]
    auc_best = max(P["classification"]["logistic"]["auc"]["mean"], auc_hgb)
    check(r2_best < 0.1 and auc_best < 0.7, "the error is not predictable from pre-measurement features")
    check(not ARMS["arm1"]["novel_acquisition_possible"], "no novel-acquisition list")
    k10 = str(S["top_k"][0])
    pol = S["policies"][k10]
    rnd_sd = S["random_single_draw_sd"][k10]["strong_or_notfit"]
    y = {p: pol[p]["strong_or_notfit"] for p in ("random", "predicted error", "predicted priority", "observed priority (ceiling)")}
    gain_err = y["predicted error"]["minus_random_mean"]
    gain_prio = y["predicted priority"]["minus_random_mean"]
    check(0 < gain_err < rnd_sd and gain_err > gain_prio and abs(gain_prio) < rnd_sd,
          "predicted error above random by less than one sd of a single pick; the policy fed predictions closer to random")
    check(y["observed priority (ceiling)"]["mean"] > 0.95, "oracle ranking near 1")
    for p in y:
        check(abs(pol[p]["strongly_reweightable"]["mean"] + pol[p]["target_not_reached"]["mean"] - y[p]["mean"]) < 1e-9,
              f"{p}: the two yield components add up")
    top_imp = P["permutation_importance_hgb"][0]

    # ---- campaign and models
    arm2 = ARMS["arm2"]
    check(arm2["test_observed_median_ratio"] > arm2["control_observed_median_ratio"] > 1.1, "arm 2: controls also above experiment in the archive")
    n_better = sum(1 for c in comps.values() if c["median_chi2_raw"] < c["bioemu_median_chi2_raw_same_entries"])
    check(n_better == 0, "BioEmu-1 has the lowest median raw chi2 of the models compared")
    pep = comps["peptron"]
    check(pep["share_better_than_bioemu"] == max(c["share_better_than_bioemu"] for c in comps.values())
          and pep["share_better_than_bioemu"] < 0.5 and pep["ci95_share_better"][1] < 0.5, "PepTron closest, below half")

    # the quality criteria the page lists are the numbered list of the campaign proposal
    import re
    camp = (ROOT / "docs" / "CAMPAIGN.md").read_text()
    qc = camp.split("## 6. QC criteria", 1)[1].split("## 7.", 1)[0]
    n_qc_int = len(re.findall(r"^\d+\. ", qc, flags=re.M))
    check(n_qc_int == 8, "eight scripted quality criteria in the campaign proposal")
    n_qc = {8: "eight"}[n_qc_int]

    def tile(v, label, sub):
        return f'<div class="tile"><div class="v">{v}</div><div class="l">{label}</div><div class="s">{sub}</div></div>'

    # ---- hero
    headline = "Where does BioEmu-1 fail against solution scattering, and what should be measured next?"
    hero = (
        f"BioEmu-1 {cite('lewis')} predicts the set of shapes a protein adopts in solution from its sequence. "
        f"I asked where these predicted ensembles disagree with measurements, and which new measurements would test "
        f"and improve such a model. As a benchmark, I compared raw BioEmu-1 ensembles with small-angle X-ray "
        f"scattering (SAXS) profiles from SASBDB {cite('kikhney')}, using the ensembles and computed curves that the "
        f"PeptoneBench authors published {cite('invernizzi')}. The clearest systematic error ran opposite to my "
        f"expectation: for disordered proteins the ensembles are too extended. The obstacle for choosing new "
        f"experiments is prediction: the size of the error on a protein could not be predicted before measuring it. "
        f"The proposed campaign therefore has two arms, a controlled replication of known discrepancies and a "
        f"prospective test of the extension bias.")
    tiles = (tile(pct(share_fit), "raw ensembles meet the χ² ≤ 2 threshold", f"{fits} of {n_bioemu} profiles")
             + tile(pct(rg_dis - 1), "larger than measured: BioEmu-1's ensembles of disordered proteins",
                    f"median radius of gyration {f2(rg_dis)} times the measured value, {E['E4']['n']} proteins "
                    f'<span class="nb">(95% interval {ci(rg_ci)})</span>')
             + tile(pct(r2_best), "of BioEmu-1's error can be predicted before measuring",
                    f'from sequence and model features, on held-out proteins <span class="nb">(R² = {f2(r2_best)})</span>'))
    check(pct(rg_dis - 1) == f"{round(100 * (rg_dis - 1))}%" == pct(ext_curve), "size tile: same rounding as Result 2")
    check(pct(r2_best) == f"{round(100 * r2_best)}%", "prediction tile: R² as a percentage")
    flow = (f'<div class="flow" aria-label="data flow"><span><b>{n_entries}</b> archive profiles</span><span class="arr">→</span>'
            f'<span><b>{n_clean}</b> pass quality checks</span><span class="arr">→</span>'
            f'<span><b>{n_bioemu}</b> with a BioEmu-1 ensemble</span></div>')

    # ---- method steps (details)
    steps = [
        ("public input", f"{n_entries} measured SAXS profiles with sequence, pH and per-residue disorder score: PeptoneDB-SAXS, curated from SASBDB {cite('kikhney', 'invernizzi')}."),
        ("public input", f"{int(B['n_conformers_median'])} BioEmu-1 conformers per protein, sampled by the PeptoneBench authors from the sequence alone."),
        ("public input", f"One scattering curve per conformer, computed by the PeptoneBench authors with Pepsi-SAXS {cite('grudinin')} on each profile's q grid."),
        ("computed here", f"Average the conformer curves with equal weights; fit one scale factor {cite('svergun')} and one constant background; report reduced χ² against the measured errors, and NRMSD, a second score that does not weight by them."),
        ("computed here", "Guinier radius of gyration of the data and of the ensemble curve; flags for aggregation (low-angle upturn), over-subtraction (intensity below −3σ) and oligomers (SASBDB measured mass against sequence mass)."),
        ("computed here", f"Maximum-entropy reweighting {cite('bottaro')} along a path of decreasing prior strength; the effective sample fraction φ at the first point with χ² ≤ 2 sorts each entry into one of four kinds."),
        ("computed here", f"The same scoring for {len(comps)} comparator models on the shared entries: four ensemble generators and two single-structure predictors as reference baselines."),
        ("computed here", "Predict the raw error from features known before measuring; compare acquisition policies on held-out folds; select replication systems by stated rules and write the two-arm campaign."),
    ]
    steps_html = "".join(
        f'<li><span class="who{" here" if who == "computed here" else ""}">{who}</span><br>{txt}</li>' for who, txt in steps)

    # ---- expectations
    def outcome(k):
        if k == "E4":
            return '<span class="outcome rev">Opposite direction observed</span>'
        return '<span class="outcome yes">Supported</span>' if E[k]["met"] else '<span class="outcome no">Not supported</span>'
    exp_rows = "".join(
        f'<tr><td><strong>{k}</strong> {stmt}</td><td class="dim">{why}</td><td>{outcome(k)}</td></tr>'
        for k, stmt, why in (
            ("E1", "Raw χ² is worse for disordered than for folded proteins.", "The model was trained mostly on folded structures and simulations of them."),
            ("E2", "Among folded proteins, raw χ² rises with chain length.", "More domains, more ways to misplace them."),
            ("E3", "Most entries reach χ² ≈ 1 while keeping φ > 0.3.", "The right conformers are usually present, only mis-weighted."),
            ("E4", "Disordered ensembles are too compact (Rg below experiment).", "The known over-compaction of models trained on folded data.")))
    exp_stats = (
        f"<p><strong>E1</strong> median χ² {f1(E['E1']['median_disordered'])} for disordered against {f1(E['E1']['median_folded'])} for folded proteins; one-sided Mann–Whitney p {pv(E['E1']['p_one_sided'])}.</p>"
        f"<p><strong>E2</strong> Spearman ρ = {E['E2']['spearman_rho']:+.2f} (p = {pv(E['E2']['p'])}, n = {E['E2']['n']}). The data show a step: folded chains of up to 100 residues meet χ² ≤ 2 in {pct(E['E2']['share_fit_le_100'])} of cases (n = {E['E2']['n_le_100']}), longer ones in {pct(E['E2']['share_fit_gt_100'])} (n = {E['E2']['n_gt_100']}), with no trend above that.</p>"
        f"<p><strong>E3</strong> {pct(E['E3']['share_reaching_chi2_1'])} of entries reach χ² ≈ 1 at all; {pct(E['E3']['share_phi_above_0.3'])} do so keeping φ &gt; 0.3.</p>"
        f"<p><strong>E4</strong> median Rg ratio {f2(rg_dis)} (95% interval {ci(rg_ci)}), two-sided Wilcoxon on the log ratio p {pv(E['E4']['p_two_sided'])}; {pct(1 - E['E4']['share_below_1'])} of disordered entries above 1.</p>")

    # ---- tables
    class_rows = "".join(
        f'<tr><td>{k}</td><td class="num">{CLS[k]["n"]}</td><td class="num">{f1(CLS[k]["chi2_raw_median"])}</td>'
        f'<td class="num">{f1(CLS[k]["chi2_raw_q25"])}–{f1(CLS[k]["chi2_raw_q75"])}</td>'
        f'<td class="num">{pct(CLS[k]["share_raw_fit"])} ({pct(H["share_raw_fit_by_class"][k]["ci95"][0])}–{pct(H["share_raw_fit_by_class"][k]["ci95"][1])})</td>'
        f'<td class="num">{f3(nr_cls[k]["nrmsd_raw_median"])} ({ci(nr_cls[k]["nrmsd_raw_ci95"], f3)})</td>'
        f'<td class="num">{pct(CLS[k]["share_not_fit"])}</td></tr>' for k in CLASSES)
    class_table = ('<div class="scroll"><table><thead><tr><th>class</th><th>n</th><th>median raw χ²</th><th>quartiles</th>'
                   '<th>χ² ≤ 2 (95%)</th><th>median NRMSD (95%)</th><th>target fit not reached</th></tr></thead><tbody>'
                   + class_rows + '</tbody></table></div><p class="note">NRMSD: root-mean-square deviation of ln I between the scaled model and the data over the points with I &gt; 0 and I ≥ 3σ, divided by the range of ln I of the data over those points. Intervals are percentile bootstraps over entries (2,000 resamples).</p>')

    kind_rows = "".join(
        f'<tr><td>{k}</td><td>{desc}</td>' + "".join(f'<td class="num">{KS[t][k]}</td>' for t in ("0.3", "0.5", "0.7")) + "</tr>"
        for k, desc in ((KIND_RAW, "raw χ² ≤ 2"), (KIND_MODEST, "χ² ≤ 2 reached while keeping φ ≥ threshold"),
                        (KIND_STRONG, "χ² ≤ 2 reached only with φ below the threshold"),
                        (KIND_NOTFIT, "χ² ≤ 2 not reached by the tested reweighting procedure")))
    kind_table = ('<div class="scroll"><table><thead><tr><th>kind</th><th>definition</th><th>φ threshold 0.3</th>'
                  '<th>0.5 (used)</th><th>0.7</th></tr></thead><tbody>' + kind_rows + '</tbody></table></div>'
                  '<p class="note">The threshold moves entries between the two reweightable kinds only; the raw-fit and target-not-reached counts do not depend on it.</p>')

    def comp_row(m, base=False):
        c = comps[m]
        return (f'<tr{" class=base" if base else ""}><td>{MODEL_NAME[m]}{" (single structure)" if base else ""}</td>'
                f'<td class="num">{c["n"]}</td><td class="num">{f1(c["median_chi2_raw"])}</td>'
                f'<td class="num">{f1(c["bioemu_median_chi2_raw_same_entries"])}</td>'
                f'<td class="num">{c["median_log10_chi2_ratio_vs_bioemu"]:+.2f} ({c["ci95_median_log10_chi2_ratio"][0]:+.2f}, {c["ci95_median_log10_chi2_ratio"][1]:+.2f})</td>'
                f'<td class="num">{pct(c["share_better_than_bioemu"])} ({pct(c["ci95_share_better"][0])}–{pct(c["ci95_share_better"][1])})</td>'
                f'<td class="num">{pct(c["share_better_nrmsd"])}</td><td class="num">{f2(others_rg[m])}</td></tr>')
    comp_html = ('<div class="scroll"><table class="wide"><thead><tr><th>model</th><th>shared entries</th><th>median raw χ²</th>'
                 '<th>BioEmu-1, same entries</th><th>median log₁₀ χ² ratio (95%)</th><th>entries where it beats BioEmu-1 (95%)</th>'
                 '<th>same, by NRMSD</th><th>disordered Rg ratio</th></tr></thead><tbody>'
                 + "".join(comp_row(m) for m in sorted(gens, key=lambda m: comps[m]["median_chi2_raw"]))
                 + "".join(comp_row(m, base=True) for m in sorted(singles, key=lambda m: comps[m]["median_chi2_raw"]))
                 + '</tbody></table></div><p class="note">Intervals: percentile bootstrap over the shared entries. The log₁₀ χ² ratio is comparator over BioEmu-1, positive when BioEmu-1 fits better. BioEmu-1 disordered Rg ratio: '
                 + f2(rg_dis) + '.</p>')

    arm1 = ARMS["arm1"]["replication"]
    check(len(arm1) == 12 and {r["label"] for r in arm1} == set(CAND.head(len(arm1))["label"]),
          "arm 1 lists the same twelve systems the priority score ranks first")
    check(all(0.6 < r["mw_ratio"] < 1.6 for r in arm1), "arm 1 candidates are monomers by measured mass")
    check(all(r["length"] <= ARMS["arm1"]["max_length"] and r["disorder_class"] != "disordered"
              and r["resolvability"] in ("strongly reweightable", "modestly reweightable") for r in arm1),
          "arm 1 systems meet the stated eligibility rules")
    check([r["chi2_raw"] for r in arm1] == sorted((r["chi2_raw"] for r in arm1), reverse=True),
          "arm 1 listed by raw chi2")
    cand_rows = "".join(
        f'<tr><td><a href="https://www.sasbdb.org/data/{r["label"]}/">{r["label"]}</a></td><td class="num">{int(r["length"])}</td>'
        f'<td>{r["disorder_class"]}</td><td class="num">{f1(r["chi2_raw"])}</td><td>{r["resolvability"]}</td>'
        f'<td>{r["rg_direction"].replace("model ", "")}</td><td class="dim">{short_title(r["sasbdb_title"], 48)}</td></tr>'
        for r in arm1[:5])

    def arm2_rows(rows):
        return "".join(
            f'<tr><td><a href="https://www.sasbdb.org/data/{r["label"]}/">{r["label"]}</a></td><td class="num">{r["length"]}</td>'
            f'<td class="num">{r["ncpr"]:+.2f}</td><td class="num">{r["fcr"]:.2f}</td><td class="num">{r["rg_model"]}</td>'
            f'<td class="num">{r["rg_law_disordered"]}</td><td class="num">{r["ratio_model_over_law"]:.2f}</td>'
            f'<td class="num">{r["ratio_model_over_exp_observed"]:.2f}</td></tr>' for r in rows)
    arm2_head = ('<thead><tr><th>SASBDB</th><th>length</th><th>net charge / residue</th><th>charged fraction</th>'
                 '<th>ensemble Rg (Å)</th><th>scaling-law Rg (Å)</th><th>model / law</th><th>model / measured (archive)</th></tr></thead>')

    pred_rows = "".join(
        f'<tr><td>{name}</td><td class="num">{P["regression"][rk]["r2"]["mean"]:.2f} ± {P["regression"][rk]["r2"]["sd"]:.2f}</td>'
        f'<td class="num">{P["regression"][rk]["mae"]["mean"]:.3f}</td>'
        f'<td class="num">{P["classification"][ck]["auc"]["mean"]:.2f} ± {P["classification"][ck]["auc"]["sd"]:.2f}</td>'
        f'<td class="num">{P["classification"][ck]["brier"]["mean"]:.3f}</td></tr>'
        for name, rk, ck in (("constant / base rate", "constant", "rate"), ("class mean / class rate", "class_mean", "class_rate"),
                             ("ridge / logistic regression", "ridge", "logistic"), ("gradient-boosted trees (depth 3)", "hgb", "hgb")))
    POL = (("random", "random"), ("predicted error", "predicted error"),
           ("predicted priority", "acquisition policy fed predictions"), ("observed priority (ceiling)", "oracle ranking (uses measured SAXS)"))
    sel_rows = "".join(
        f'<tr><td>{name}</td>' + "".join(
            f'<td class="num">{S["policies"][str(k)][p]["strong_or_notfit"]["mean"]:.2f} ± {S["policies"][str(k)][p]["strong_or_notfit"]["sd"]:.2f}</td>'
            f'<td class="num">{S["policies"][str(k)][p]["strongly_reweightable"]["mean"]:.2f}</td>'
            f'<td class="num">{S["policies"][str(k)][p]["target_not_reached"]["mean"]:.2f}</td>' for k in S["top_k"]) + "</tr>"
        for p, name in POL)
    sel_table = ('<div class="scroll"><table class="wide"><thead><tr><th>policy</th><th>top 10: yield</th><th>strongly reweightable</th>'
                 '<th>target fit not reached</th><th>top 20: yield</th><th>strongly reweightable</th><th>target fit not reached</th></tr></thead><tbody>'
                 + sel_rows + f'</tbody></table></div><p class="note">Mean ± sd over five repeats of the grouped 5-fold split; random is averaged over {S["n_random_draws_per_fold"]} draws per fold, and one random pick of ten spreads by ±{f2(rnd_sd)}. Pool shares: {pct(S["pool_mean"]["strong_or_notfit"])} high-discrepancy, {pct(S["pool_mean"]["strongly_reweightable"])} strongly reweightable, {pct(S["pool_mean"]["target_not_reached"])} target fit not reached.</p>')

    nd = R["near_duplicates"]
    st_d = R["curve_rg_ratio_by_class"]["disordered"]
    robust_rows = "".join(f'<tr><td>{a}</td><td class="num">{b}</td><td class="num">{c}</td><td class="num">{d}</td></tr>' for a, b, c, d in (
        ("all clean disordered entries", st_d["n"], f"{f2(st_d['median'])} ({ci(st_d['ci95'])})", pv(st_d["p_two_sided_wilcoxon_log"])),
        ("five most extreme ratios dropped", trimmed["n"], f"{f2(trimmed['median'])} ({ci(trimmed['ci95'])})", pv(trimmed["p_two_sided_wilcoxon_log"])),
        (f"near-duplicates collapsed (5-mer Jaccard ≥ {nd['threshold_jaccard_5mer']})", collapsed["n"], f"{f2(collapsed['median'])} ({ci(collapsed['ci95'])})", pv(collapsed["p_two_sided_wilcoxon_log"])),
        ("Cα coordinates of the conformers", cr_d["n"], f"{f2(cr_d['coord_over_exp_median'])} ({ci(cr_d['coord_over_exp_ci95'])})", pv(cr_d["p_two_sided_wilcoxon_log_coord_over_exp"]))))
    robust_table = ('<div class="scroll"><table><thead><tr><th>disordered entries</th><th>n</th><th>median ratio (95%)</th><th>Wilcoxon p, log ratio</th></tr></thead><tbody>'
                    + robust_rows + f'</tbody></table></div><p class="note">Within the class the ratio grows with chain length (Spearman ρ = {rho_len["spearman_rho"]:+.2f}, p = {pv(rho_len["p"])}, n = {rho_len["n"]}). The {A["n_entries"]} profiles contain no identical sequences; {nd["n_pairs"]} pairs are near-duplicates (about 93% identity or closer), and collapsing each cluster leaves the raw-fit share at {pct(nd["collapsed_share_raw_fit"])}.</p>')
    dup_rows = "".join(f'<tr><td>{p["a"]}</td><td>{p["b"]}</td><td class="num">{p["jaccard"]:.2f}</td><td class="num">{p["containment"]:.2f}</td></tr>'
                       for p in nd["pairs"])

    def ex_row(cls):
        lab = A["examples"][cls]
        ex = EX[lab]
        rew = (f'{ex["chi2"]["operating"]:.2f} (φ {ex["phi_operating"]:.2f})' if ex["operating_reached"]
               else f'minimum {ex["chi2"]["operating"]:.2f} (φ {ex["phi_operating"]:.2f})')
        lv = {x["name"]: x for x in ex["density_levels"]}
        return (f'<tr><td>{cls}</td><td><a href="https://www.sasbdb.org/data/{lab}/">{lab}</a></td><td class="num">{ex["length"]}</td>'
                f'<td>{RV.loc[lab, "resolvability"]}</td><td class="num">{ex["chi2"]["raw"]:.2f}</td><td class="num">{rew}</td>'
                f'<td class="num">{ex["chi2"]["alphafold"]:.1f}</td><td class="num">{ex["rg_exp"]:.1f}</td><td class="num">{ex["rg_raw"]:.1f}</td>'
                f'<td class="num">{ex["rg_operating"]:.1f}</td><td class="num">{pct(lv["protein"]["ca_inside"])} / {pct(lv["particle"]["ca_inside"])}</td>'
                f'<td class="num">{ex["chain_breaks"]}</td><td class="num">{ex["denss"]["chi2_median"]:.2f}</td>'
                f'<td class="num">{ex["denss"]["resolution_A"]:.0f}</td></tr>')
    ex_table = ('<div class="scroll"><table class="wide"><thead><tr><th>class</th><th>entry</th><th>residues</th><th>kind</th>'
                '<th>χ² raw</th><th>χ² reweighted</th><th>χ² AlphaFold2</th><th>Rg measured (Å)</th><th>Rg raw</th>'
                '<th>Rg reweighted</th><th>Cα inside, protein / particle level</th><th>chain breaks</th>'
                '<th>DENSS χ²</th><th>resolution (Å)</th></tr></thead><tbody>' + "".join(ex_row(c) for c in CLASSES)
                + '</tbody></table></div>')
    cov = coverage_section(n_bioemu)
    trn = training_section(n_bioemu)
    nav = "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t in [
        ("method", "Method"), ("expectations", "Expectations"), ("fit", "Raw fit"), ("examples", "Examples"),
        ("size", "Size"), ("reweight", "Reweighting"), ("guide", "Guidance"), ("next", "Next"),
        ("collect", "Data to collect"),
        ("models", "Other models"), ("limits", "Limits"), ("repro", "Code")])
    refs_html = "".join(f'<li id="ref-{k}">{t} <a href="{u}">{u.replace("https://", "")}</a></li>' for k, t, u in REFS)
    today = date.today()
    diagram = method_diagram(n_entries, int(B["n_conformers_median"]))
    decision = decision_diagram(len(arm1), len(arm2["test"]), len(arm2["controls"]))
    favicon = "data:image/svg+xml," + quote(FAVICON)

    exs = {c: EX[A["examples"][c]] for c in CLASSES}
    labs = {c: A["examples"][c] for c in CLASSES}
    reached = [c for c in CLASSES if exs[c]["operating_reached"]]
    not_reached = [c for c in CLASSES if not exs[c]["operating_reached"]]
    rew_sentence = "The reweighted curve is taken where the four kinds are read, at the first point along the reweighting path with χ² ≤ 2"
    if reached:
        rew_sentence += " (" + ", ".join("φ {:.2f} for {}".format(exs[c]["phi_operating"], labs[c]) for c in reached) + ")"
    if not_reached:
        rew_sentence += ("; where that target is not reached (" + ", ".join(labs[c] for c in not_reached)
                         + "), it is the curve at the χ² minimum of the path, labelled as such")
    rew_sentence += (". φ is the effective sample fraction the weights keep, 1 for uniform weights. The χ² minimum of each path is "
                     + ", ".join("{:.2f} at φ {:.3f}".format(exs[c]["chi2_minimum"], exs[c]["phi_minimum"]) for c in CLASSES)
                     + " (folded, partly disordered, disordered); a minimum at small φ rests on a few conformers and overfits.")
    breaks = [c for c in CLASSES if exs[c]["chain_breaks"] > 0]
    breaks_sentence = ""
    if breaks:
        breaks_sentence = (" The drawn conformer has breaks in the sampled chain (consecutive Cα more than 4.2 Å apart): "
                           + ", ".join("{} in {}".format(exs[c]["chain_breaks"], labs[c]) for c in breaks) + ".")
    uniform = [c for c in CLASSES if exs[c]["top_weight"] <= 1.5 / 100]
    uniform_sentence = ""
    if uniform:
        uniform_sentence = (" Where the weights stay uniform along the whole path (" + ", ".join(labs[c] for c in uniform)
                            + "), no conformer carries more weight than another and the one drawn is the first of the ensemble.")
    how_to_read = (
        "<p><strong>Residuals.</strong> (I<sub>model</sub> − I<sub>exp</sub>)/σ against q, with ±3 guides. A residual that wanders outside ±3 over a range of q is a shape error at that length scale (about 1/q); a flat band within ±3 is a fit at the noise level.</p>"
        f"<p><strong>Reweighted curve.</strong> {rew_sentence}</p>"
        f"<p><strong>Envelope.</strong> DENSS {cite('grant')} reconstructions from the measured curve alone (ten maps, aligned and averaged; Dmax from the SASBDB record), drawn as filled projections of two density levels of the averaged map: the isosurface enclosing the protein's expected volume at 1.7 Å³ per Da (darker) and the one enclosing the volume DENSS assigned to the particle (lighter). The conformer is the highest-weight one at the χ² minimum of the path, docked into the protein-volume level by principal axes; the share of its Cα atoms inside that level is given under each panel and, for both levels, in the table.{uniform_sentence}{breaks_sentence}</p>"
        f"<p><strong>How typical the examples are.</strong> Each is the entry nearest its class median of raw χ². {lab_d} is not typical in size: its size ratio ({ratio_d:.2f}) is below the disordered median ({f2(rg_dis)}), so this column understates the class-wide excess. It is {ex_d['length']} residues long, and the drawn conformer has {ex_d['chain_breaks']} chain breaks.</p>"
        "<p><strong>What the envelope is.</strong> An illustration from an ensemble-averaged measurement, not evidence about any single conformation; for a disordered protein one conformer cannot fill it.</p>")
    pp = y["predicted priority"]

    html = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="description" content="Where BioEmu-1 disagrees with solution scattering, and which proteins to measure next.">
<title>BioEmu against SAXS</title>
<link rel="icon" href="{favicon}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{CSS}</style>
<nav class="topnav" aria-label="Sections"><div class="inner">
<a class="brand" href="#top">{GH}<span>bioemu → saxs → campaign</span></a>
<button class="navtoggle" type="button" aria-label="Jump to section" aria-expanded="false" aria-controls="navlist">Sections</button>
<ul id="navlist">{nav}</ul></div></nav>
<span id="top"></span>
<div class="wrap">
<header>
<p class="eyebrow">Biomolecular emulators · solution scattering · experimental design</p>
<h1>{headline}</h1>
<p class="byline">Natalija Stepurko · {today:%B %Y} · <a href="{SITE}">natalija-stepurko.com</a></p>
<p class="abstract">{hero}</p>
<div class="tiles">{tiles}</div>
{flow}
<a class="repo" href="{REPO}">{GH}<span><b>All the code is public.</b> Every number and figure on this page comes from the pipeline in this repository, which downloads the public inputs, scores them and writes the campaign proposal.</span><span class="repo-path">Natalija-Stepurko/bioemu-saxs-campaign</span></a>
</header>

<h2 id="method">How the benchmark works</h2>
<p>SAXS reports the overall size and shape of a protein in solution, averaged over every conformation it adopts, so it tests an ensemble as a whole. BioEmu-1 was trained on predicted structures, molecular-dynamics simulations and measured stabilities {cite('lewis')}. BioEmu conformers have served as a pool for SAXS ensemble selection on a few dozen proteins, where the unweighted pool did not reproduce the data {cite('haitin')}, and PeptoneBench ranks generators by their fit after reweighting {cite('invernizzi')}. This study asks where the raw model fails by protein class and what that implies for the next measurements.</p>
<figure class="result diagram" style="max-width:746px"><figcaption>Method · what is taken as published and what this study computes</figcaption><div class="scroll">{diagram}</div></figure>
<p>This project re-analyses BioEmu-1 ensembles sampled by the PeptoneBench authors; it does not retrain or re-sample the model. The archive records the BioEmu code version but not the checkpoint.</p>
{details("The eight steps of the pipeline", f'<ol class="steps">{steps_html}</ol>')}

<h2 id="expectations">Expectations set before scoring</h2>
<p>Four expectations were written down, each with its reasoning, before any score was computed.</p>
<div class="scroll"><table class="exp"><thead><tr><th>expectation</th><th>reasoning</th><th>outcome</th></tr></thead><tbody>{exp_rows}</tbody></table></div>
{details("Test statistics for each expectation", exp_stats)}

<h2 id="fit">Raw SAXS fit is worse for disordered proteins, but the size of the gap depends on how error is measured</h2>
<p>Each raw ensemble is the uniform average of its conformer curves, scaled to the profile with one factor and one constant background. Of the {n_bioemu} profiles, {fits} meet χ² ≤ 2 ({pct(share_fit)}, 95% interval {pct(H['share_raw_fit']['ci95'][0])}–{pct(H['share_raw_fit']['ci95'][1])}). The median raw χ² is {f1(CLS['disordered']['chi2_raw_median'])} for disordered and {f1(CLS['folded']['chi2_raw_median'])} for folded proteins ({f1(CLS['partly disordered']['chi2_raw_median'])} for partly disordered). Under NRMSD, a deviation of ln I that does not weight points by their reported errors, the class medians are similar: {f3(nr_cls['folded']['nrmsd_raw_median'])} folded, {f3(nr_cls['partly disordered']['nrmsd_raw_median'])} partly disordered and {f3(nr_cls['disordered']['nrmsd_raw_median'])} disordered, with overlapping intervals. A poor reduced χ² can reflect both the disagreement with the curve and the precision assigned to the measurement. The size error is examined separately below.</p>
{fig("fig_error_map", "Figure 1 · raw fit by protein class under two scores",
     f"Each point is one of the {n_bioemu} profiles. Left: reduced χ² of the uniform BioEmu-1 ensemble, log scale, with the class median (bar and value) and the χ² = 2 threshold (dashed). Right: NRMSD of the same fits, with the class median and its 95% bootstrap interval.",
     "Under χ² the disordered median is several times the folded one; under NRMSD the three medians lie within each other's intervals.")}
{details("Class table and the chain-length test (E2)", class_table + fig("fig_length", "Raw χ² against chain length", f"Raw reduced χ² of every profile against chain length, coloured by class. Among folded proteins the rank correlation is weak (ρ = {E['E2']['spearman_rho']:+.2f}); chains of up to 100 residues meet χ² ≤ 2 more often ({pct(E['E2']['share_fit_le_100'])} against {pct(E['E2']['share_fit_gt_100'])})."))}

<h2 id="examples">What a fit looks like</h2>
<p class="lead">One protein per class, the one nearest its class median of raw χ², shows the fits on measured curves.</p>
{fig("fig_examples", "Figure 2 · one worked example per class",
     "Top: every measured point with its error band and three computed curves scaled to it, BioEmu-1 raw (solid), reweighted (dotted) and AlphaFold2 (dashed), with their χ² above each panel. Middle: residuals. Bottom: the ab initio envelope reconstructed from the measured curve, with the Cα trace of a BioEmu-1 conformer docked inside.",
     f"In the disordered column the raw ensemble is larger than the protein in solution (Rg {ex_d['rg_raw']:.0f} Å against {ex_d['rg_exp']:.0f} Å measured), and reweighting to χ² ≤ 2 moves it to {ex_d['rg_operating']:.0f} Å, toward the measured value.")}
{details("How to read Figure 2", how_to_read + ex_table)}

<h2 id="size">Disordered ensembles are systematically too extended</h2>
<p>The radius of gyration, Rg, is the root-mean-square distance of a protein's mass from its centre, a single number for overall size; SAXS measures it from the lowest-angle part of the curve. For disordered proteins the median ensemble size is {pct(ext_curve)} above experiment from the scattering curves and {pct(ext_coord)} above from the conformer coordinates.</p>
{fig("fig_rg_robustness", "Figure 3 · ensemble size against measured size",
     f"Left: ensemble Rg over measured Rg for every profile, from the computed and measured curves, by class; bars are the median and its 95% bootstrap interval. Right: for the {cr_d['n']} disordered and {cr_f['n']} folded entries whose conformer coordinates were analysed, the same ratio from the curves (circles) and from the Cα coordinates of the conformers (diamonds).",
     "Above the dashed line the model's ensemble is larger than the protein in solution. The expectation was over-compaction (below the line); the disordered median lies above it.")}
<p>E4 expected the usual failure of models trained on folded data, over-compaction of disordered chains. The data show the opposite: the median ratio is {f2(rg_dis)} (95% interval {ci(rg_ci)}, two-sided Wilcoxon p {pv(E['E4']['p_two_sided'])}, n = {E['E4']['n']}), and {pct(1 - E['E4']['share_below_1'])} of disordered entries lie above 1; folded proteins sit at {f2(CLS['folded']['rg_ratio_median'])}. The size discrepancy is also present in coordinate-derived radii: the Cα radius of the sampled conformers is {f2(cr_d['coord_over_exp_median'])} times the measured value ({ci(cr_d['coord_over_exp_ci95'])}). Cα, full-atom and SAXS-derived radii are different observables, however: for folded proteins the Cα ratio is {f2(cr_f['coord_over_exp_median'])} where the curve gives {f2(cr_f['curve_over_exp_median'])}, an offset because the curve includes side chains and a hydration layer that the Cα radius leaves out. The {pct(ext_coord)} is therefore not to be read as an exact physical excess; it shows that the extension is present in the conformers before any curve is computed.</p>
{details("Robustness of the size finding and absolute radii", robust_table + fig("fig_rg", "Ensemble Rg against measured Rg", "Radius of gyration of the BioEmu-1 ensemble curve against the Guinier radius of the measured profile, one point per profile, log axes.") + '<div class="scroll"><table><thead><tr><th>entry</th><th>entry</th><th>5-mer Jaccard</th><th>containment</th></tr></thead><tbody>' + dup_rows + '</tbody></table></div>')}

<h2 id="reweight">Can the disagreement be corrected by changing conformer populations?</h2>
<p>Reweighting keeps every conformer fixed and adjusts only how often each one occurs, until the averaged curve matches the profile. Maximum-entropy reweighting {cite('bottaro')} does this while keeping the weights as close to uniform as the data allow; relaxing that constraint step by step traces a path from the raw ensemble towards a full fit. The effective sample fraction φ measures how much of the ensemble still carries weight: 1 for uniform weights, near 0 when a few conformers carry it all. The four kinds are read where the path first reaches χ² ≤ 2: <em>raw fit</em> (already there), <em>modestly reweightable</em> (reached with φ ≥ 0.5), <em>strongly reweightable</em> (reached only with φ &lt; 0.5, which requires substantial population changes) and <em>target fit not reached</em> (by the tested reweighting procedure).</p>
{fig("fig_kinds", "Figure 4 · the four kinds, and one reweighting path",
     f"Left: the {n_bioemu} profiles by kind. Right: χ² against φ along the path of {rep}, the strongly reweightable entry with the median φ of its kind; the open circle marks where χ² first falls to 2.",
     None)}
<p>Reweighting brings {rew} more profiles to χ² ≤ 2, {strong} of them only with φ &lt; 0.5; for {notfit} the target is not reached.</p>
<div class="call"><p><strong>Two cautions.</strong> Reweightability is not information gain: SAXS is low-dimensional, and distinct ensembles can give the same curve, so a reweighted fit does not identify the true populations. Not reaching the target does not establish that compatible conformations are absent from BioEmu's distribution; it describes this finite ensemble of {int(B['n_conformers_median'])} conformers, this forward model and this procedure.</p></div>
{details("Kind counts at other φ thresholds, every path, and the kinds by class", kind_table + fig("fig_resolvability", "Every reweighting path", "Left: χ² against φ for every profile, coloured by class. Right: the share of each class in the four kinds."))}

<h2 id="guide">Can the error map guide experiments on proteins that have not been measured?</h2>
<p>An error map needs the measurement. Choosing a new experiment needs the error predicted before the measurement exists. I fitted predictors of the raw error (log₁₀ χ²) from quantities known before a profile exists: chain length, disorder score and class, sequence composition, the Rg expected from length by scaling laws {cite('kohn')}, and the model's own ensemble Rg and conformer spread. Evaluation is 5-fold cross-validation grouped by sequence cluster, repeated five times. I then asked which entries each acquisition policy would pick from held-out folds and scored the picks by the high-discrepancy target yield: the share that are strongly reweightable or whose target fit is not reached. The yield is a proxy for where the model is wrong, not a measure of training value.</p>
{fig("fig_predict", "Figure 5 · predicting the error, and what each policy picks",
     f"A: observed against predicted log₁₀ raw χ² on held-out folds ({'ridge regression' if P['best_regressor'] == 'ridge' else 'boosted trees'}), coloured by class. B: high-discrepancy target yield among the top {k10} picks; error bars are the sd over five repeats, and for random the sd of a single random pick of {k10}. The oracle ranking uses the measured SAXS and is shown as a ceiling; it is not a deployable policy.",
     "In A a predictive model would spread the points along the diagonal; here they form a vertical band. In B a policy is useful when it stands clear of the random bar and its spread.")}
<p>The best model explains {pct(r2_best)} of held-out variation in the raw error (R² = {f2(r2_best)}; class means alone {f2(P['regression']['class_mean']['r2']['mean'])}), and the tree classifier of the high-discrepancy kinds reaches an AUC of {f2(auc_hgb)}. Random picks of ten give a yield of {f2(y['random']['mean'])}, with a spread of ±{f2(rnd_sd)} for any single pick. Ranking by predicted error gives {f2(y['predicted error']['mean'])} ± {f2(y['predicted error']['sd'])}, {gain_err:+.2f} over random, within about one sd of a single random pick. The acquisition policy fed predictions, the campaign's priority rule with predicted inputs, gives {f2(pp['mean'])} ± {f2(pp['sd'])}. The oracle ranking, which uses the measured SAXS, reaches {f2(y['observed priority (ceiling)']['mean'])}: the rule separates the kinds when it sees the measurement, and the missing piece is a predictor of them. The map therefore supports controlled replication of known failures; it does not support a new-protein campaign driven by predicted error.</p>
<p class="call"><strong>The priority score is not a basis for recommending measurements.</strong> The design proposed ranking systems by log₁₀ raw χ² × a reweighting weight × a tractability weight, with weights set by judgement. Fed the quantities known before a measurement, it picks high-discrepancy profiles at {f2(pp['mean'])} against {f2(y['random']['mean'])} for random choice, a difference inside the spread of a single random pick. I report the score and its test, and the campaign below does not rest on it. Arm 1 re-measures deposited systems, so it can use their measured discrepancy directly: it lists the systems that meet stated eligibility rules and have the largest observed raw χ², which are the same twelve the score ranks first; what is dropped is the judgement-set weighting. Arm 2 splits test proteins from controls by a quantity computed from the model and the sequence alone, within the archive's disordered entries whose deposited profiles pass the quality checks.</p>
{details("Prediction metrics, the two components of the yield, and feature importance", '<div class="scroll"><table><thead><tr><th>model</th><th>R² (log₁₀ χ²)</th><th>MAE</th><th>AUC (high-discrepancy kind)</th><th>Brier</th></tr></thead><tbody>' + pred_rows + f'</tbody></table></div><p class="note">Mean ± sd over five repeats; n = {P["n"]} entries in {P["n_groups"]} sequence groups. The feature the tree model uses most is the {FEATURE_LABEL.get(top_imp["feature"], top_imp["feature"].replace("_", " "))} (importance {top_imp["mean"]:.3f} ± {top_imp["sd"]:.3f}).</p>' + sel_table + fig("fig_predict_detail", "Calibration and feature importance", "Left: calibration of the tree classifier, quintiles of predicted probability against the observed share. Right: drop in held-out R² when each feature is permuted, eight most used features."))}

<h2 id="next">What should be measured next?</h2>
{trn["purpose"]}
{trn["training"]}
{cov["space"]}
{trn["closeness"]}
{cov["split"]}
{trn["collect"]}
<h3 id="arms">The campaign</h3>
<p>The error map identifies where BioEmu-1 disagrees with deposited data; it cannot yet say where it will disagree on a new protein. The campaign has two arms that follow from that. {trn["arms_link"]} The <a href="{REPO}/blob/main/docs/CAMPAIGN.md">full proposal</a> gives constructs, assay, scripted quality criteria, success criteria, contingencies and work packages for an external provider.</p>
<figure class="result diagram" style="max-width:746px"><figcaption>Campaign · two arms</figcaption><div class="scroll">{decision}</div></figure>
<h3>Arm 1 · replication: are known discrepancies reproducible under controlled conditions?</h3>
<p>High-error, reweightable monomers from the archive are re-measured by SEC-SAXS in one standard buffer and temperature, with tags removed and a concentration series. Standardised measurement reduces condition heterogeneity and tests whether the discrepancy is reproducible. One common buffer and temperature can itself shift some ensembles, so each new profile is compared with the deposited one before it is compared with the model. The five eligible systems with the largest raw χ²:</p>
<div class="scroll"><table class="wide"><thead><tr><th>SASBDB</th><th>length</th><th>class</th><th>raw χ²</th><th>kind</th><th>model Rg</th><th>deposited study</th></tr></thead><tbody>{cand_rows}</tbody></table></div>
<p class="note">Eligible: reweighting reaches χ² ≤ 2, measured mass matches a monomer, at most {ARMS['arm1']['max_length']} residues, folded or partly disordered ({ARMS['arm1']['n_eligible']} entries). Listed by raw χ², largest first: the order sorts by the size of the known discrepancy, which is what a replication tests, and is not an estimate of what each measurement is worth; the final choice needs a feasibility review. The first {len(arm1)} are in the <a href="{REPO}/blob/main/docs/CAMPAIGN.md">campaign proposal</a>.</p>
<h3>Arm 2 · hypothesis test: are BioEmu-1's disordered ensembles systematically too extended?</h3>
<p>The hypothesis came from this archive, so the arm is a prospective replication test with pre-specified model-based selection and new SAXS measurements. Test proteins are disordered proteins whose BioEmu-1 ensemble Rg exceeds the scaling-law Rg for disordered chains (1.927 N<sup>0.598</sup> Å {cite('kohn')}) by more than 10%, a quantity computed from the model and the sequence alone, drawn from the archive's disordered entries whose deposited profiles pass the quality checks, spanning chain length and net charge ({arm2['n_test_candidates']} of {arm2['n_disordered_pool']} disordered entries qualify). The scaling-law-matched controls are disordered proteins of the same length bins whose ensemble Rg is within 10% of the scaling law ({arm2['n_control_candidates']} qualify). In the archive their ratio of ensemble Rg to measured Rg is {f2(arm2['control_observed_median_ratio'])} at the median, against {f2(arm2['test_observed_median_ratio'])} for the test group, so the controls are not known to agree with experiment. Pre-specified outcome: the median Rg<sub>exp</sub>/Rg<sub>model</sub> across the test arm from Guinier fits of the new profiles. The bias is refuted if that median is at least 0.95; it is confirmed if it is at most 0.90 in the test arm and at least 0.95 in the controls.</p>
{details("Test proteins and controls", '<div class="scroll"><table class="wide">' + arm2_head + '<tbody>' + arm2_rows(arm2["test"]) + '</tbody></table></div><p class="note">Controls</p><div class="scroll"><table class="wide">' + arm2_head + '<tbody>' + arm2_rows(arm2["controls"]) + '</tbody></table></div>')}
<div class="feas" role="note"><p class="feas-title">Feasibility judgement</p><ul>
<li>A single common buffer and temperature is the aim, but it may not hold for every protein: some may be insoluble, unstable or aggregate in it.</li>
<li>Gate before any SAXS: solubility in the standard buffer; the A280 trace of purification and analytical SEC (one symmetric peak at the expected elution volume) with the A260/A280 ratio for nucleic acid; SDS-PAGE (one band at the expected mass); circular dichroism where the fold is in question.</li>
<li>Aggregation signs on the SAXS data: a low-angle upturn, a non-linear Guinier region, Rg or I(0)/c rising with concentration, a mass from I(0) above the monomer, or Rg changing across the SEC peak; all are among the scripted quality criteria.</li>
<li>Fallback: a protein that fails in the standard buffer is measured in the nearest buffer in which it is monodisperse, or in its deposited buffer, with the deviation recorded, and analysed separately; one that fails in every buffer is replaced by the next eligible system.</li>
</ul></div>
<p>Both arms use size-exclusion-coupled SAXS at three concentrations with a protein standard in every session, and {n_qc} scripted quality criteria (Guinier linearity, no low-angle upturn and Rg agreeing across concentrations, no radiation damage across frames, molecular weight from I(0) within 20% of the sequence mass, no negative intensities, a repeated deposited entry reproduced at χ² &lt; 2, I(0)/c constant across concentrations, and a constant Rg across the SEC peak). Entries whose target fit is not reached get hydrogen–deuterium exchange mass spectrometry on the same batch of protein. Every batch of twelve constructs re-runs this pipeline, so the error map is updated before the next batch is chosen.</p>

<h2 id="models">How does BioEmu-1 compare with other ensemble generators?</h2>
<p>The archive carries computed curves from other models for the same entries, scored here identically and without reweighting: the ensemble generators {", ".join(MODEL_NAME[m] for m in gens)}, and the single-structure predictors {" and ".join(MODEL_NAME[m] for m in singles)} as reference baselines. On the shared entries BioEmu-1 has the lowest median raw χ² of the seven, and the ordering is kept under NRMSD. PepTron comes closest: it beats BioEmu-1 on {pct(pep['share_better_than_bioemu'])} of the {pep['n']} shared entries (95% interval {pct(pep['ci95_share_better'][0])}–{pct(pep['ci95_share_better'][1])}). BioEmu-1 is the only tested model with a disordered median Rg ratio above 1.1; PepTron sits at {f2(others_rg['peptron'])}.</p>
{fig("fig_models", "Figure 12 · seven models on the same profiles",
     "Left: median raw reduced χ² of each model's unweighted ensemble by disorder class, on the profiles that pass quality checks; BioEmu-1 and PepTron emphasised, single-structure predictors grouped below. Right: median Rg ratio of each model on the disordered entries; solid line 1, dashed line 1.1.",
     None)}
{details("Paired comparison table", comp_html + f'<p class="note">Five further folders in the archive are not used: Boltz-1x and ESMFlow are superseded by models already included, idpGAN and IDP-o are disorder-only generators represented here by idpSAM, and PepTron-base is the un-finetuned PepTron. AF-CALVADOS {cite("bulow")} was evaluated on this benchmark but its predictions were not in the archive.</p>')}

<h2 id="limits">Limits</h2>
<p>The ensembles are those in the PeptoneBench archive, which records the BioEmu code version but not the checkpoint; re-sampling with the current v1.2 checkpoint is the first follow-up. SASBDB profiles differ in buffer, temperature and construct, none of which the model sees, so part of the raw error is condition mismatch. The forward model is treated as exact; its hydration-shell parameters bias Rg by a few per cent, which is why modestly reweightable cases are read with care. Disorder classes come from a sequence-based predictor. The model comparison is restricted to the six comparators on the shared entries. CATH and the PDB serve as the reference for experimentally known structure space. {trn["limits"]} PDB coverage of a construct comes from SIFTS residue ranges, which count residues listed in the structure record and can include residues not modelled in the coordinates. The prediction and selection tests are retrospective and use {n_bioemu} entries; a predictor that fails at this size could succeed with more data or with richer features of the model's own ensemble. Profiles with a low-angle upturn above {int(100 * A['thresholds']['aggregation_upturn'])}%, an intensity more than three standard errors below zero, no valid Guinier region, or a measured mass above 1.6 times the sequence mass are kept out of every statistic ({A['n_flag_aggregation']}, {A['n_flag_negative']}, {A['n_guinier_invalid']} and {A['n_flag_oligomer']} profiles).</p>

<h2 id="repro">Reproducibility</h2>
<p>The pipeline runs in named stages (download, score, coordinate radii, analysis, selection test, prediction, envelopes, report) from a locked environment, with unit tests on synthetic data. Derived results are tracked in the repository, and the page build reads every number from them and stops if a sentence that states a direction or a comparison no longer matches. Key code: <a href="{REPO}/blob/main/src/bsc/coordrg.py">coordinate radii</a>, <a href="{REPO}/blob/main/src/bsc/select_eval.py">acquisition policies</a>, <a href="{REPO}/blob/main/src/bsc/predict.py">error prediction</a>. The ensembles and computed curves are those published by the PeptoneBench authors {cite('invernizzi')} (Zenodo 17306061, CC-BY 4.0).</p>

<h2 id="refs">References</h2>
<ol class="refs">{refs_html}</ol>
<footer>
<div>Data: PeptoneDB-SAXS and PeptoneBench predictions (Zenodo 17306061, CC-BY 4.0), curated from SASBDB. Downloaded at run time and not redistributed.</div>
<div>Code: <a href="{REPO}">bioemu-saxs-campaign</a> · design: <a href="{REPO}/blob/main/docs/DESIGN.md">design document</a> · proposal: <a href="{REPO}/blob/main/docs/CAMPAIGN.md">campaign</a>.</div>
<div>More of what I am working on is at <a href="https://natalija-stepurko.com">natalija-stepurko.com</a>.</div>
</footer>
</div>
<script>{NAV_JS}</script>
</html>
"""
    low = html.lower()
    for bad in ("rather than", "instead of", "by contrast", "not assumed"):
        check(bad not in low, f"forbidden phrase: {bad}")
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    check(not re.search(r"\bwe\b", text, re.I), "first person singular only")
    check(not re.search(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,40}\b", re.sub(r"data:[^\"']+", "", text)),
          "no hashes on the page")
    OUT.write_text(html)
    print(f"wrote {OUT}  {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build()
