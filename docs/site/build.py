"""Build docs/index.html from results/. Every number on the page is read from results/ (analysis.json,
by_class.csv, candidates.csv, predict.json, select_eval.json, examples.json, campaign_arms.json and the
acquisition results acq_surrogate.json, acq_objectives.json, acq_value.json, acq_acquire.json);
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


# ----------------------------------------------------------------------------- the acquisition extension
SUR = json.load(open(RESULTS / "acq_surrogate.json"))
OBJ = json.load(open(RESULTS / "acq_objectives.json"))
VAL = json.load(open(RESULTS / "acq_value.json"))
ACQ = json.load(open(RESULTS / "acq_acquire.json"))
OBJ_LABEL = {"O1": "O1 · information about the shared parameters", "O2": "O2 · information about the protein's own ensemble",
             "O3": "O3 · disagreement between generators", "O4": "O4 · sequence coverage", "O5": "O5 · feasibility (heuristic)",
             "predicted error": "predicted raw error (baseline)"}
POLICY_LABEL = {"random": "random", "O1": "greedy on O1", "O2": "greedy on O2", "O3": "greedy on O3", "O4": "greedy on O4",
                "O5": "greedy on O5", "predicted error": "greedy on predicted error", "pareto": "Pareto on (O1, O4, O5)",
                "pareto constrained": "Pareto, more feasible half only", "oracle": "oracle (reads the profiles)"}


def sgn(x, nd=2):
    t = f"{x:+.{nd}f}"
    return f"{0:.{nd}f}" if float(t) == 0 else t.replace("-", "−")


def sci(v, nd=2):
    return f"{sgn(v[0], nd)} to {sgn(v[1], nd)}"


def policy_label(p):
    return f"Chebyshev, weight {float(p.split()[1]):.2f} on O5" if p.startswith("chebyshev") else POLICY_LABEL[p]


def acquisition_section(n_bioemu, r2_best):
    e5, val, e6, e9 = SUR["E5"], VAL["validity"], VAL["E6"], VAL["E9"]
    splits = e5["per_split"]
    n_pool, n_held = SUR["splits"]["sizes"][0]
    bc = e5["by_class_pooled"]
    rr = [r["relative_reduction_median_chi2"] for r in splits]
    rg_raw, rg_cor = e5["rg_ratio_disordered_raw_mean"], e5["rg_ratio_disordered_corrected_mean"]
    feas = OBJ["feasibility_check"]
    wc = VAL["validity_within_class"]
    # every sentence below that states a direction is checked here
    check(SUR["theta_zero_check"]["n"] == n_bioemu and SUR["theta_zero_check"]["max_relative_difference"] < 1e-13,
          "θ = 0 reproduces the raw χ² of every entry")
    check(not e5["met"] and e5["mean_relative_reduction"] < e5["threshold"], "E5 not met: the held-out median χ² does not fall by 10%")
    check(max(rr) < e5["threshold"], "no split reaches the 10% threshold")
    check(abs(rg_cor - 1) < abs(rg_raw - 1) and rg_raw > 1.1, "the correction brings disordered sizes toward the measured value")
    check(bc["disordered"]["ci95"][0] > 0 and bc["folded"]["ci95"][0] < 0 < bc["folded"]["ci95"][1],
          "disordered held-out profiles improve; folded ones do not move")
    check(bc["folded"]["share_held_out"] > 0.5 and bc["disordered"]["share_held_out"] < 0.15,
          "folded profiles are the majority of the held-out set; disordered about a tenth")
    check(not ACQ["informative"] and not ACQ["E7"]["evaluable"] and not ACQ["E8"]["evaluable"],
          "the acquisition comparison is flagged uninformative")
    check(val["O1"]["ci95"][0] < 0 < val["O1"]["ci95"][1] and not e6["met"], "O1 uncorrelated with realised value; E6 not met")
    neg = [k for k in ("O2", "O3", "predicted error") if val[k]["ci95"][1] < 0]
    check(neg == ["O2", "O3", "predicted error"], "O2, O3 and predicted error correlate negatively, intervals below 0")
    check(val["O4"]["ci95"][0] < 0 < val["O4"]["ci95"][1], "O4 uncorrelated")
    check(val["O5"]["spearman"] < 0.2 and all(val[k]["spearman"] <= val["O5"]["spearman"] for k in val), "O5 the only positive, weak")
    check(val["O5"]["ci95"][0] > 0 and all(wc[c]["O5"] > 0 for c in CLASSES), "O5 positive overall and within each class")
    check(all(wc[c]["O2"] < 0 for c in CLASSES), "O2 negative within each class")
    check(not e9["met"] and e9["median_disordered"] < e9["median_folded"] and e9["p_two_sided_mannwhitney"] < 0.05,
          "E9 opposite: realised value lower for disordered than folded")
    check(feas["n_failed"] == 36 and feas["ci95"][0] < 0.5 < feas["ci95"][1], "O5 does not separate the 36 failed profiles")
    check(VAL["top_decile_composition"]["share_disordered"] == 0, "no disordered protein in the top decile of realised value")
    o2wc = [wc[c]["O2"] for c in CLASSES]
    ms = SUR["multistart_first_split"]
    check(max(m["objective"] for m in ms) - min(m["objective"] for m in ms) < 1e-3 * min(m["objective"] for m in ms),
          "random starts reach the same optimum")
    fit_s = sum(SUR["fit_seconds_full_pool"]) / len(SUR["fit_seconds_full_pool"])
    k3 = 1e3

    rise = -e5["mean_relative_reduction"]
    check(rise > 0, "on average the held-out median χ² rose")
    lead = (
        f"<p class=\"lead\">Which profiles, chosen before they are measured, would most improve a model's predictions for other proteins? "
        f"I tested this retrospectively on the {n_bioemu} profiles, with a design fixed before any result: in five splits, 30% are held out and the "
        f"other {n_pool} are revealed ten at a time. The model they update is a small correction to BioEmu-1 with eight parameters shared across "
        f"proteins, which reweights each ensemble by conformer size and rescales it; it is not BioEmu-1's network. Fitted on a whole pool, the "
        f"correction brings disordered ensembles to their measured size but also worsens the profiles that already fit, and the held-out median "
        f"χ² rose by {pct(rise)} on average where the design "
        f"required a 10% fall, so by the rule fixed in advance the comparison of acquisition policies is uninformative. Of the objectives computable "
        f"before a measurement, none predicted a profile's realised value beyond a weak positive correlation for feasibility "
        f"(Spearman ρ = {sgn(val['O5']['spearman'])}); the information objective was uncorrelated with it (ρ = {sgn(val['O1']['spearman'])}).</p>")
    hb = e5["heldout_by_raw_fit"]
    fit_, mis = hb["raw_fit"], hb["raw_misfit"]
    band = {(b["lo"], b["hi"]): b for b in hb["bands"]}
    b23, b10 = band[(2.0, 3.0)], band[(10.0, None)]
    meds = hb["median_raw_chi2_by_split"]
    fc, dc = hb["by_class"]["folded"], hb["by_class"]["disordered"]
    check(mis["mean_ci95"][0] > 0 and mis["median_ci95"][0] > 0, "profiles with raw χ² > 2 improve")
    check(fit_["mean_ci95"][1] < 0 and fit_["median_ci95"][1] < 0, "profiles with raw χ² ≤ 2 get worse")
    check(b23["mean_ci95"][1] < 0 and b10["mean_ci95"][0] > 0, "the 2–3 band gets worse, the band above 10 improves")
    check(all(2 < m <= 5 for m in meds), "every split's held-out median raw χ² lies between 2 and 5")
    check(band[(3.0, 5.0)]["mean_ci95"][0] < 0 < band[(3.0, 5.0)]["mean_ci95"][1], "the 3–5 band does not change")
    check(fc["share_better"] > 0.3 and fc["share_worse"] > 0.3 and abs(fc["share_better"] - fc["share_worse"]) < 0.05,
          "folded profiles move both ways in about equal shares")
    check(dc["share_better"] > 0.75, "most disordered profiles improve")
    check(r2_best < 0.1, "the first study could not predict the raw error")
    fig6 = fig("fig_acq_surrogate", "Figure 6 · what the shared correction does to held-out profiles",
               f"A: for every held-out profile of the five splits ({n_held} per split), the log₁₀ reduction of χ² when the correction fitted on the pool "
               f"replaces raw BioEmu-1, against the profile's raw χ², coloured by class; black bars are the mean in bands of raw χ² (up to 2, 2–3, 3–5, "
               f"5–10, above 10) with 95% bootstrap intervals, the dotted line is χ² = 2, and the grey band spans the five splits' held-out median raw χ². "
               f"Values beyond the axis are drawn at its edge. "
               f"B: the reduction of the held-out median χ² in each split, the quantity E5 tested, against its 10% threshold; a negative reduction is "
               f"a rise. Disordered held-out ensembles move from a median Rg ratio of {f2(rg_raw)} to {f2(rg_cor)} (mean over the splits).",
               f"Points above zero are profiles the correction fits better. The shared correction improves the profiles that raw BioEmu-1 fits badly "
               f"(raw χ² above 2: mean log₁₀ reduction {sgn(mis['mean'], 3)}, 95% interval {sci(mis['mean_ci95'], 3)}, n = {mis['n']}; above 10: "
               f"{sgn(b10['mean'], 3)}) and worsens those it already fits (raw χ² up to 2: {sgn(fit_['mean'], 3)}, {sci(fit_['mean_ci95'], 3)}, "
               f"n = {fit_['n']}; 2–3: {sgn(b23['mean'], 3)}). The held-out median raw χ² lies between {f1(min(meds))} and {f1(max(meds))} in the five "
               f"splits, where profiles get worse or do not change, so the held-out median rises. Folded profiles move both ways: "
               f"{pct(fc['share_better'])} get better and {pct(fc['share_worse'])} worse by more than {hb['move']} in log₁₀ χ²; {pct(dc['share_better'])} "
               f"of disordered profiles improve. The correction is driven only by features known before measuring, and in the first study those features "
               f"could not predict which proteins BioEmu-1 fits badly (R² = {f2(r2_best)} for log₁₀ raw χ²), so it cannot aim its change at the "
               f"proteins that need it.")
    value_p = (
        f"<p>The realised value of a profile is what it actually does: the change in the held-out score (the median log₁₀ χ² reduction) when the profile "
        f"is added to a random set of 10 or 30 pool profiles and the correction is refitted, averaged over 20 such sets per split, "
        f"{VAL['n_fits']:,} fits in all. Each objective was computed at the same sets, as a deployed policy would compute it, from computed curves, sequence "
        f"and the profiles already in the set, and compared with realised value by rank.</p>")
    fig7 = fig("fig_acq_validity", "Figure 7 · do the objectives predict what a profile is worth?",
               "A: Spearman correlation between each objective and realised value, averaged over the five splits, with 95% intervals from a bootstrap over "
               "sequence groups. B: realised value of each pool protein (averaged over the splits in which it is in the pool), by class, with the median and "
               "its 95% bootstrap interval.",
               "In A an objective that identifies useful profiles sits right of zero with its interval clear of it; only feasibility does, weakly.")
    after = (
        f"<p>One reading of the negative correlations: a profile whose own ensemble is broad (O2), on which the generators disagree (O3), or whose raw "
        f"error is predicted to be large pulls the shared correction toward the disordered minority, which the held-out median barely registers; "
        f"O2 stays negative within each class (ρ from {sgn(min(o2wc))} to {sgn(max(o2wc))}). Realised value is accordingly lower for disordered than for "
        f"folded proteins (median {f1(k3 * e9['median_disordered'])} against {f1(k3 * e9['median_folded'])} × 10⁻³ log₁₀ χ²; two-sided Mann–Whitney "
        f"p {pv(e9['p_two_sided_mannwhitney'])}), and none of the {VAL['top_decile_composition']['n']} proteins of highest value is disordered: the "
        f"opposite of E9. The feasibility heuristic correlates weakly and positively, also within each class, but it does not separate the "
        f"{feas['n_failed']} deposited profiles that failed the quality flags from the rest (AUC {f2(feas['auc_pass_above_fail'])}, 95% interval "
        f"{ci(feas['ci95'])}). The result is specific to this correction and this score: a correction with more freedom for folded proteins, or a score "
        f"that weights the classes equally, could value data differently.</p>")

    def outc(kind):
        return {"no": '<span class="outcome no">Not supported</span>', "rev": '<span class="outcome rev">Opposite direction observed</span>',
                "un": '<span class="outcome no">Uninformative (E5 not met)</span>'}[kind]
    rows = [
        ("E5", "Fitted on the whole pool, the correction improves the held-out median χ² by at least 10%.", outc("no"),
         f"median χ² rose by {pct(rise)} on average (change across splits {sgn(-100 * max(rr), 0)}% to {sgn(-100 * min(rr), 0)}%)"),
        ("E6", "O1 correlates positively with realised value, more strongly than O2.", outc("no"),
         f"ρ O1 {sgn(e6['rho_O1'])} ({sci(e6['ci95_O1'])}); O2 {sgn(e6['rho_O2'])} ({sci(e6['ci95_O2'])})"),
        ("E7", "Pareto and O1-greedy batches beat random choice; the predicted-error policy does not.", outc("un"), "not evaluated"),
        ("E8", "Restricting to the more feasible half costs less than a quarter of the improvement.", outc("un"), "not evaluated"),
        ("E9", "Realised value is highest for disordered proteins.", outc("rev"),
         f"median {f1(k3 * e9['median_disordered'])} disordered, {f1(k3 * e9['median_folded'])} folded (× 10⁻³)"),
    ]
    table = ('<div class="scroll"><table class="exp"><thead><tr><th>expectation</th><th>outcome</th><th>numbers</th></tr></thead><tbody>'
             + "".join(f'<tr><td><strong>{k}</strong> {s}</td><td>{o}</td><td class="dim">{n}</td></tr>' for k, s, o, n in rows)
             + '</tbody></table></div>')

    # ---- details
    split_rows = "".join(
        f'<tr><td>{r["split"] + 1}</td><td class="num">{f2(r["median_chi2_raw"])}</td><td class="num">{f2(r["median_chi2_corrected"])}</td>'
        f'<td class="num">{sgn(100 * r["relative_reduction_median_chi2"], 0)}%</td><td class="num">{sgn(r["median_log10_reduction"], 3)}</td>'
        f'<td class="num">{pct(r["share_improved"])}</td><td class="num">{f2(r["secondary"]["rg_ratio_disordered"]["raw"])} → '
        f'{f2(r["secondary"]["rg_ratio_disordered"]["corrected"])}</td><td class="num">{f3(r["secondary"]["nrmsd"]["raw"])} → '
        f'{f3(r["secondary"]["nrmsd"]["corrected"])}</td></tr>' for r in splits)
    sur_detail = (
        '<p>The correction: each conformer\'s weight is multiplied by exp(−λ z), z being its log radius of gyration relative to the ensemble mean, '
        'and the averaged curve is read at q·s; λ and ln s are linear in four features known before a measurement (a constant, the disorder score, '
        'log chain length, and the log ratio of the ensemble\'s Rg to the scaling-law Rg for its length), and s is bounded to 0.8–1.25. With all '
        f'eight parameters at zero it reproduces the raw ensemble: the χ² of all {SUR["theta_zero_check"]["n"]} entries agrees with the scoring stage to '
        f'within a relative difference of 10⁻¹³. Each profile\'s scale and background are fitted inside χ², and the '
        f'parameters minimise the sum of reduced χ² plus a ridge penalty whose width, τ = {SUR["tau"]:g}, was chosen by grouped cross-validation inside '
        f'the first pool. A fit on a whole pool takes {fit_s:.0f} s on one core; from five starting points it reaches the same optimum.</p>'
        '<div class="scroll"><table class="wide"><thead><tr><th>split</th><th>held-out median χ², raw</th><th>corrected</th><th>reduction</th>'
        '<th>median log₁₀ χ² reduction</th><th>profiles improved</th><th>disordered Rg ratio</th><th>median NRMSD</th></tr></thead><tbody>'
        + split_rows + f'</tbody></table></div><p class="note">{n_pool} pool and {n_held} held-out profiles per split; whole sequence groups are held out.</p>')
    val_rows = "".join(
        f'<tr><td>{OBJ_LABEL[k]}</td><td class="num">{sgn(val[k]["spearman"])} ({sci(val[k]["ci95"])})</td>'
        + "".join(f'<td class="num">{sgn(wc[c][k])}</td>' for c in CLASSES) + '</tr>' for k in val)
    nm = OBJ["noise_model"]
    val_detail = (
        '<div class="scroll"><table class="wide"><thead><tr><th>objective</th><th>Spearman ρ (95%)</th><th>within folded</th><th>within partly disordered</th>'
        '<th>within disordered</th></tr></thead><tbody>' + val_rows + '</tbody></table></div>'
        '<p class="note">O1: expected gain in information about the eight shared parameters (log determinant of the posterior precision, Laplace approximation, '
        'scale and background profiled out). O2: how strongly the curve separates the protein\'s larger and smaller conformers. O3: spread of ln Rg across six '
        'generators. O4: 1 − 5-mer containment to the nearest acquired sequence. O5: one minus the mean of five penalties (length outside 50–300, disorder '
        'score, cysteines, hydrophobic fraction, isoelectric point near 7.4), fixed in advance and not validated. The noise of an unmeasured profile is predicted '
        'from q and molecular weight by a model fitted on acquired profiles only; on held-out profiles it is off by a factor of '
        f'{10 ** nm["median_abs_log10_error"]:.1f} at the median. The predicted error is a ridge regression of log₁₀ raw χ² on the features of the prediction test, '
        'refitted on the acquired profiles.</p>')
    P = ACQ["policies"]
    pols = sorted([p for p in P if p != "random"], key=lambda p: -P[p]["auc_minus_random_mean"])
    pol_rows = "".join(
        f'<tr><td>{policy_label(p)}</td><td class="num">{sgn(1000 * P[p]["auc_minus_random_mean"], 1)} ({sgn(1000 * P[p]["auc_minus_random_ci95"][0], 1)} to '
        f'{sgn(1000 * P[p]["auc_minus_random_ci95"][1], 1)})</td><td class="num">{f2(P[p]["mean_O5_chosen"])}</td>'
        f'<td class="num">{pct(P[p]["share_folded"])} / {pct(P[p]["share_partly disordered"])} / {pct(P[p]["share_disordered"])}</td></tr>' for p in pols)
    sim_detail = (
        '<p>Because E5 was not met, even the correction fitted on a whole pool does not lower the held-out median χ², so the simulation cannot show '
        'which data would; the differences below are not evidence for or against any policy. They are listed for completeness.</p>'
        + fig("fig_acq_curves", "Acquisition simulation · reported as uninformative",
              f"A: held-out median log₁₀ χ² reduction against the number of profiles acquired, mean over {ACQ['n_splits']} splits × {ACQ['n_seeds']} seed sets of ten, "
              "for random choice, O1, predicted error, the two Pareto policies and the oracle (other policies grey); the dash-dotted line is the correction "
              "fitted on the whole pool. B: area under each curve minus that of random choice on the same split and seed set, with 95% intervals from a "
              "bootstrap over splits and seed sets.", cls="result")
        + '<div class="scroll"><table class="wide"><thead><tr><th>policy</th><th>area minus random, × 10⁻³ (95%)</th><th>mean O5 of the proteins chosen</th>'
        '<th>chosen: folded / partly / disordered</th></tr></thead><tbody>' + pol_rows + '</tbody></table></div>'
        f'<p class="note">Random choice: mean O5 {f2(P["random"]["mean_O5_chosen"])}, chosen {pct(P["random"]["share_folded"])} / {pct(P["random"]["share_partly disordered"])} / '
        f'{pct(P["random"]["share_disordered"])}. {ACQ["n_fits"]:,} fits.</p>')
    pareto_detail = fig("fig_acq_pareto", "The candidate objectives in one round",
                        "The pool candidates of the first round (first split, first seed set) in the three pairwise projections of O1, O4 and O5; the first "
                        "non-dominated front is ringed, and the first batch of the Pareto policy and of its feasible-half variant is marked.")
    trade_detail = fig("fig_acq_tradeoff", "Feasibility against held-out improvement · uninformative",
                       "Area under the held-out improvement curve against the mean feasibility of the proteins chosen, along the Chebyshev sweep of weights on "
                       "(O1, O4, O5) and for the two Pareto policies, random choice and greedy feasibility; bars are 95% intervals.")
    return (lead + fig6 + value_p + fig7 + after + table
            + details("The correction, its fit and the E5 test per split", sur_detail)
            + details("Every objective against realised value", val_detail)
            + details("The acquisition simulation, reported as uninformative", sim_detail + trade_detail)
            + details("The candidate objectives and a Pareto batch", pareto_detail))


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
    nav = "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t in [
        ("method", "Method"), ("expectations", "Expectations"), ("fit", "Raw fit"), ("examples", "Examples"),
        ("size", "Size"), ("reweight", "Reweighting"), ("guide", "Guidance"), ("next", "Next"), ("choose", "Choosing data"),
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
    acq_html = acquisition_section(n_bioemu, r2_best)

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
<p>The error map identifies where BioEmu-1 disagrees with deposited data; it cannot yet say where it will disagree on a new protein. The campaign has two arms that follow from that. The <a href="{REPO}/blob/main/docs/CAMPAIGN.md">full proposal</a> gives constructs, assay, scripted quality criteria, success criteria, contingencies and work packages for an external provider.</p>
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

<h2 id="choose">Choosing data for the next model</h2>
{acq_html}

<h2 id="models">How does BioEmu-1 compare with other ensemble generators?</h2>
<p>The archive carries computed curves from other models for the same entries, scored here identically and without reweighting: the ensemble generators {", ".join(MODEL_NAME[m] for m in gens)}, and the single-structure predictors {" and ".join(MODEL_NAME[m] for m in singles)} as reference baselines. On the shared entries BioEmu-1 has the lowest median raw χ² of the seven, and the ordering is kept under NRMSD. PepTron comes closest: it beats BioEmu-1 on {pct(pep['share_better_than_bioemu'])} of the {pep['n']} shared entries (95% interval {pct(pep['ci95_share_better'][0])}–{pct(pep['ci95_share_better'][1])}). BioEmu-1 is the only tested model with a disordered median Rg ratio above 1.1; PepTron sits at {f2(others_rg['peptron'])}.</p>
{fig("fig_models", "Figure 8 · seven models on the same profiles",
     "Left: median raw reduced χ² of each model's unweighted ensemble by disorder class, on the profiles that pass quality checks; BioEmu-1 and PepTron emphasised, single-structure predictors grouped below. Right: median Rg ratio of each model on the disordered entries; solid line 1, dashed line 1.1.",
     None)}
{details("Paired comparison table", comp_html + f'<p class="note">Five further folders in the archive are not used: Boltz-1x and ESMFlow are superseded by models already included, idpGAN and IDP-o are disorder-only generators represented here by idpSAM, and PepTron-base is the un-finetuned PepTron. AF-CALVADOS {cite("bulow")} was evaluated on this benchmark but its predictions were not in the archive.</p>')}

<h2 id="limits">Limits</h2>
<p>The ensembles are those in the PeptoneBench archive, which records the BioEmu code version but not the checkpoint; re-sampling with the current v1.2 checkpoint is the first follow-up. SASBDB profiles differ in buffer, temperature and construct, none of which the model sees, so part of the raw error is condition mismatch. The forward model is treated as exact; its hydration-shell parameters bias Rg by a few per cent, which is why modestly reweightable cases are read with care. Disorder classes come from a sequence-based predictor. The model comparison is restricted to the six comparators on the shared entries. The prediction and selection tests are retrospective and use {n_bioemu} entries; a predictor that fails at this size could succeed with more data or with richer features of the model's own ensemble. Profiles with a low-angle upturn above {int(100 * A['thresholds']['aggregation_upturn'])}%, an intensity more than three standard errors below zero, no valid Guinier region, or a measured mass above 1.6 times the sequence mass are kept out of every statistic ({A['n_flag_aggregation']}, {A['n_flag_negative']}, {A['n_guinier_invalid']} and {A['n_flag_oligomer']} profiles).</p>

<h2 id="repro">Reproducibility</h2>
<p>The pipeline runs in named stages (download, score, coordinate radii, analysis, selection test, prediction, envelopes, the four acquisition stages, report) from a locked environment, with unit tests on synthetic data. Derived results are tracked in the repository, and the page build reads every number from them and stops if a sentence that states a direction or a comparison no longer matches. Key code: <a href="{REPO}/blob/main/src/bsc/coordrg.py">coordinate radii</a>, <a href="{REPO}/blob/main/src/bsc/select_eval.py">acquisition policies</a>, <a href="{REPO}/blob/main/src/bsc/predict.py">error prediction</a>, <a href="{REPO}/blob/main/src/bsc/surrogate.py">the shared correction</a>, <a href="{REPO}/blob/main/src/bsc/acquire.py">the acquisition simulation</a>. The ensembles and computed curves are those published by the PeptoneBench authors {cite('invernizzi')} (Zenodo 17306061, CC-BY 4.0).</p>

<h2 id="refs">References</h2>
<ol class="refs">{refs_html}</ol>
<footer>
<div>Data: PeptoneDB-SAXS and PeptoneBench predictions (Zenodo 17306061, CC-BY 4.0), curated from SASBDB. Downloaded at run time and not redistributed.</div>
<div>Code: <a href="{REPO}">bioemu-saxs-campaign</a> · design: <a href="{REPO}/blob/main/docs/DESIGN.md">design document</a> · <a href="{REPO}/blob/main/docs/ACQUISITION_DESIGN.md">acquisition design</a> · proposal: <a href="{REPO}/blob/main/docs/CAMPAIGN.md">campaign</a>.</div>
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
