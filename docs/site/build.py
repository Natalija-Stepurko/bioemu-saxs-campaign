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
import sys
from datetime import date
from pathlib import Path

import pandas as pd

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


def fig(name, caption, what, read, narrow=False):
    p = RESULTS / "figures" / f"{name}.png"
    check(p.exists(), f"figure {name} exists")
    cls = "result narrow" if narrow else "result"
    return (f'<figure class="{cls}"><figcaption>{caption}</figcaption><img src="{data_uri(p)}" alt="{caption}">'
            f'<div class="rtext"><p><strong>What it shows.</strong> {what}</p><p><strong>How to read it.</strong> '
            f'{read}</p></div></figure>')


def met(k):
    return "met" if E[k]["met"] else "not met"


def short_title(t, n=60):
    t = t or ""
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"


def details(summary, body):
    return f'<details class="more"><summary>{summary}</summary>{body}</details>'


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


def method_diagram(n_profiles, n_conf, n_bioemu, n_comp):
    """Two rows: off-the-shelf inputs (grey) and what this study computes (blue)."""
    W, H = 960, 330
    BW, BH = 220, 86
    grey_fill, blue_fill, blue_stroke = "#E9ECEE", "#E6EEF6", ACC
    b = []

    def card(x, y, title, lines, who, fill, stroke):
        b.append(box(x, y, BW, BH, fill=fill, stroke=stroke, sw=1.2))
        b.append(T(x + 12, y + 18, title, 12.5, INK, weight=600))
        for i, ln in enumerate(lines):
            b.append(T(x + 12, y + 38 + 15 * i, ln, 10.5, INK))
        b.append(T(x + 12, y + BH - 11, who, 9.5, MUTED, mono=True))

    # row 1: inputs
    y1 = 36
    xs = [20, 20 + BW + 60, 20 + 2 * (BW + 60)]
    card(xs[0], y1, "SASBDB", [f"{n_profiles} measured SAXS profiles", "with sequence and disorder score"],
         "curated by PeptoneBench", grey_fill, "#B9C0C6")
    card(xs[1], y1, "BioEmu-1 ensembles", [f"{n_conf} conformers per protein", "sampled from sequence"],
         "sampled by the PeptoneBench authors", grey_fill, "#B9C0C6")
    card(xs[2], y1, "Scattering curves", ["one curve per conformer", "Pepsi-SAXS forward model"],
         "computed by the PeptoneBench authors", grey_fill, "#B9C0C6")
    for i in range(2):
        b.append(arrow(xs[i] + BW, y1 + BH / 2, xs[i + 1] - 3, y1 + BH / 2))
    b.append(T(20, 18, "OFF-THE-SHELF INPUTS", 10, MUTED, mono=True, weight=600))
    # row 2: computed here
    y2 = 190
    BW2 = 212
    xs2 = [20 + k * (BW2 + 24) for k in range(4)]
    cards2 = [
        ("Raw fit", ["average the 100 curves; one scale", "and one background; reduced χ²", "against the measured errors"]),
        ("Size and data checks", ["Guinier Rg of data and ensemble;", "aggregation, over-subtraction,", "oligomer flags (SASBDB mass)"]),
        ("Reweighting path", ["prior strength relaxed stepwise;", "effective sample fraction φ at χ²", "targets → four kinds"]),
        ("Rank and design", [f"same for {n_comp} comparator models;", "candidates ranked; two-arm", "campaign; two validation tests"]),
    ]
    for (title, lines), x in zip(cards2, xs2, strict=True):
        b.append(box(x, y2, BW2, BH + 10, fill=blue_fill, stroke=blue_stroke, sw=1.2))
        b.append(T(x + 12, y2 + 18, title, 12.5, INK, weight=600))
        for i, ln in enumerate(lines):
            b.append(T(x + 12, y2 + 38 + 15 * i, ln, 10.5, INK))
        b.append(T(x + 12, y2 + BH - 1, "computed in this study", 9.5, ACC, mono=True))
    for i in range(3):
        b.append(arrow(xs2[i] + BW2, y2 + (BH + 10) / 2, xs2[i + 1] - 3, y2 + (BH + 10) / 2))
    # curves feed the raw fit; the data feed the checks
    b.append(arrow(xs[2] + BW / 2, y1 + BH, xs2[0] + BW2 / 2 + 60, y2 - 3))
    b.append(arrow(xs[0] + BW / 2, y1 + BH, xs2[0] + 40, y2 - 3))
    b.append(T(xs2[-1] + BW2, y2 - 18, f"COMPUTED IN THIS STUDY · {n_bioemu} CLEAN MONOMER PROFILES", 10, MUTED,
               anchor="end", mono=True, weight=600))
    # legend
    ly = H - 16
    b.append(box(20, ly - 7, 14, 14, fill=grey_fill, stroke="#B9C0C6"))
    b.append(T(42, ly, "off-the-shelf input (who made it)", 10.5, MUTED))
    b.append(box(260, ly - 7, 14, 14, fill=blue_fill, stroke=blue_stroke))
    b.append(T(282, ly, "computed in this study", 10.5, MUTED))
    return svg(W, H, "".join(b), "Method: public profiles, ensembles and curves feed the raw fit, the size checks, "
                                 "the reweighting path and the campaign design")


CSS = """
:root{--paper:#F7F8F9;--panel:#FFFFFF;--ink:#16191D;--muted:#5B646E;--rule:#DDE1E4;--acc:#2F5D8A;--band:#EEF1F2;
  --sans:ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);line-height:1.6;padding-inline:16px}
.wrap{max-width:1140px;margin:0 auto;padding:30px 0 90px}
h2[id]{scroll-margin-top:74px}
.topnav{position:sticky;top:0;z-index:10;background:color-mix(in srgb,var(--paper) 90%,transparent);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--rule);margin-inline:-16px;padding-inline:16px}
.topnav .inner{max-width:1140px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;gap:16px;height:54px;position:relative}
.topnav .brand{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:12px;color:var(--ink);text-decoration:none;white-space:nowrap}
.topnav ul{list-style:none;margin:0;padding:0;display:flex;gap:13px;font-family:var(--mono);font-size:10.5px;letter-spacing:.05em;text-transform:uppercase}
.topnav ul a{display:block;color:var(--muted);text-decoration:none;white-space:nowrap;padding:17px 0 15px;border-bottom:2px solid transparent}
.topnav ul a:hover{color:var(--ink)}.topnav ul a.active{color:var(--ink);border-bottom-color:var(--ink)}
.navtoggle{display:none;font:inherit;font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink);
  background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:6px 10px;cursor:pointer;max-width:60vw;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.navtoggle::before{content:"\\2261\\00a0\\00a0";font-size:13px}
@media (max-width:1100px){.navtoggle{display:block}.topnav ul{display:none;position:absolute;top:54px;right:0;flex-direction:column;gap:0;min-width:240px;
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
.result{margin:18px 0;background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:12px 12px 10px}
.result figcaption{font-family:var(--mono);font-size:11.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
.result img,.result svg{width:100%;height:auto;display:block}
.result.narrow img{max-width:760px;margin:0 auto}
.result.diagram svg{min-width:760px}
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
details.more .scroll,details.more p{padding:0 14px 10px}
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


def build():
    n_bioemu = sum(RES.values())
    n_clean = A["n_clean"]
    check(n_bioemu == 399 and n_clean == 403, "denominators: 399 BioEmu-scored clean entries, 403 clean entries")
    share_fit = B["share_raw_fit_chi2_2"]
    fits, mod, strong, notfit = (RES.get(k, 0) for k in (KIND_RAW, KIND_MODEST, KIND_STRONG, KIND_NOTFIT))
    check(fits + mod + strong + notfit == n_bioemu, "kind counts sum")
    comps = {c["model"]: c for c in A["model_comparison"]}
    gens = [m for m in GENERATORS if m in comps]
    singles = [m for m in SINGLE if m in comps]
    check(len(comps) == 6 and set(comps) == set(GENERATORS) | set(SINGLE), "six comparators")

    # ---- directions and comparisons the text states
    worst = max(CLS, key=lambda k: CLS[k]["chi2_raw_median"])
    best_cls = min(CLS, key=lambda k: CLS[k]["chi2_raw_median"])
    check(best_cls == "folded" and worst == "disordered", "class ordering of the raw error by chi2")
    check(E["E1"]["met"] and not E["E2"]["met"] and not E["E3"]["met"] and not E["E4"]["met"], "E1 met, E2–E4 not")
    check(E["E2"]["share_fit_le_100"] > E["E2"]["share_fit_gt_100"] and abs(E["E2"]["spearman_rho"]) < 0.2,
          "E2: a step at 100 residues, no slope")
    rg_dis = E["E4"]["median_rg_ratio_disordered"]
    rg_ci = E["E4"]["ci95_median_rg_ratio"]
    check(rg_dis > 1 and rg_ci[0] > 1 and E["E4"]["p_two_sided"] < 0.05, "E4 reversed: ratio above 1 with interval and test")
    others_rg = {m: v for m, v in A["rg_ratio_disordered_by_model"].items() if m != "bioemu" and m in comps}
    check(all(v < rg_dis for v in others_rg.values()), "BioEmu-1 has the largest disordered Rg ratio of the seven")
    trimmed = R["disordered_drop_5_most_extreme"]
    check(trimmed["median"] > 1 and trimmed["ci95"][0] > 1 and trimmed["p_two_sided_wilcoxon_log"] < 0.05, "trimmed E4 holds")
    collapsed = R["near_duplicates"]["collapsed_curve_rg_ratio_by_class"]["disordered"]
    check(collapsed["median"] > 1 and collapsed["ci95"][0] > 1, "E4 holds with near-duplicates collapsed")
    rho_len = R["disordered_ratio_vs_length"]
    check(rho_len["spearman_rho"] > 0, "longer disordered chains are more over-extended")
    CR = R["coordinate_rg"]["by_class"]
    check(CR["disordered"]["coord_over_exp_median"] > CR["disordered"]["curve_over_exp_median"] > 1
          and CR["disordered"]["coord_over_exp_ci95"][0] > 1, "the excess extension is in the conformers")
    check(CR["folded"]["coord_over_exp_median"] < 1 and CR["folded"]["curve_over_coord_median"] > 1,
          "folded: coordinates below experiment, forward model adds a hydration layer")
    check(CR["disordered"]["curve_over_coord_median"] < 1, "disordered: the curve Rg sits below the coordinate Rg")
    NR = H["nrmsd"]
    check(not NR["class_order_holds"] and NR["model_order_holds"] and NR["bioemu_first_under_nrmsd"],
          "NRMSD: class order does not hold, model order does")
    nr_cls = NR["by_class"]
    check(all(nr_cls[c]["nrmsd_raw_ci95"][0] <= nr_cls["folded"]["nrmsd_raw_median"] <= nr_cls[c]["nrmsd_raw_ci95"][1]
              or nr_cls["folded"]["nrmsd_raw_ci95"][0] <= nr_cls[c]["nrmsd_raw_median"] <= nr_cls["folded"]["nrmsd_raw_ci95"][1]
              for c in ("partly disordered", "disordered")), "NRMSD class medians overlap in their intervals")
    KS = H["kind_counts_by_phi_threshold"]
    check(len({KS[t][KIND_RAW] for t in KS}) == 1 and len({KS[t][KIND_NOTFIT] for t in KS}) == 1,
          "phi threshold moves only the two reweightable kinds")
    n_better = sum(1 for c in comps.values() if c["median_chi2_raw"] < c["bioemu_median_chi2_raw_same_entries"])
    check(n_better == 0, "BioEmu-1 has the lowest median raw chi2 of the models compared")
    pep = comps["peptron"]
    check(pep["share_better_than_bioemu"] == max(c["share_better_than_bioemu"] for c in comps.values())
          and pep["share_better_than_bioemu"] < 0.5 and pep["ci95_share_better"][1] < 0.5, "PepTron closest, below half")
    # prediction and selection
    r2_best = max(P["regression"]["ridge"]["r2"]["mean"], P["regression"]["hgb"]["r2"]["mean"])
    auc_best = max(P["classification"]["logistic"]["auc"]["mean"], P["classification"]["hgb"]["auc"]["mean"])
    check(r2_best < 0.2 and auc_best < 0.7, "the error is not predictable from sequence-level features (page text says so)")
    check(not ARMS["arm1"]["novel_acquisition_possible"], "no novel-acquisition list")
    top_imp = P["permutation_importance_hgb"][0]
    k10 = str(S["top_k"][0])
    pol = S["policies"][k10]
    rnd_sd = S["random_single_draw_sd"][k10]["strong_or_notfit"]
    gain_prio = pol["predicted priority"]["strong_or_notfit"]["minus_random_mean"]
    gain_err = pol["predicted error"]["strong_or_notfit"]["minus_random_mean"]
    ceil = pol["observed priority (ceiling)"]["strong_or_notfit"]["mean"]
    check(abs(gain_prio) < rnd_sd and gain_err < rnd_sd and ceil > 0.9, "selection rule: within chance with predictions, ceiling high")
    check(pol["predicted priority"]["phi_reduction_chi2_1"]["mean"] <= pol["random"]["phi_reduction_chi2_1"]["mean"] + 0.01,
          "predicted priority does not raise the phi-reduction objective")
    # examples
    for lab, ex in EX.items():
        check(ex["chi2"]["reweighted"] < ex["chi2"]["raw"], f"{lab}: reweighting lowers chi2")
    check(EX[A["examples"]["folded"]]["chi2"]["alphafold"] < EX[A["examples"]["folded"]]["chi2"]["raw"], "folded example: AF2 fits as well as the ensemble")
    for cls in ("partly disordered", "disordered"):
        check(EX[A["examples"][cls]]["chi2"]["alphafold"] > EX[A["examples"][cls]]["chi2"]["raw"], f"{cls} example: AF2 worse")
    arm2 = ARMS["arm2"]
    check(arm2["test_observed_median_ratio"] > arm2["control_observed_median_ratio"] > 1, "arm 2 archive ratios")

    def class_sentence():
        parts = [f"{f1(CLS[k]['chi2_raw_median'])} for {k} proteins (n = {CLS[k]['n']})" for k in CLASSES]
        return ", ".join(parts[:-1]) + " and " + parts[-1]

    headline = "Where does BioEmu-1 fail against solution scattering, and what should be measured next?"
    subline = (f"Raw ensemble predictions against {n_bioemu} SAXS profiles, what reweighting says about each miss, "
               f"and a two-arm measurement design.")
    abstract = (
        f"BioEmu-1 {cite('lewis')} emulates a protein's conformational ensemble from its sequence. I scored its raw "
        f"ensembles, with no reweighting, against {A['n_entries']} small-angle X-ray scattering (SAXS) profiles from "
        f"SASBDB {cite('kikhney')}, using the public ensembles and back-calculated curves of PeptoneBench "
        f"{cite('invernizzi')}; {n_clean} profiles pass data-quality checks and {n_bioemu} of those have a BioEmu-1 "
        f"ensemble. The ensemble fits {pct(share_fit)} of them at reduced χ² ≤ 2 (95% interval "
        f"{pct(H['share_raw_fit']['ci95'][0])}–{pct(H['share_raw_fit']['ci95'][1])}; median χ² "
        f"{f1(B['chi2_raw_median'])}, {f1(CLS['folded']['chi2_raw_median'])} for folded and "
        f"{f1(CLS['disordered']['chi2_raw_median'])} for disordered proteins). Reweighting sorts the misses: {mod} are "
        f"modestly reweightable, {strong} strongly reweightable and {notfit} are not fit along the path. One "
        f"expectation came out reversed: for disordered proteins the ensemble radius of gyration is {f2(rg_dis)} times "
        f"the measured value at the median ({ci(rg_ci)}), and the conformers themselves are "
        f"{f2(CR['disordered']['coord_over_exp_median'])} times too large, so the excess is in the sampled chains. "
        f"Two validation experiments test the campaign logic: the raw error cannot be predicted from sequence-level "
        f"features (R² = {f2(r2_best)}), and the selection rule, fed with such predictions, picks informative profiles no "
        f"better than chance on held-out folds. The campaign therefore has a replication arm under one condition and a "
        f"pre-specified test of the extension bias with matched controls chosen from the model alone.")

    # ---- method steps
    steps = [
        ("off-the-shelf", f"{A['n_entries']} measured SAXS profiles with sequence, pH and per-residue disorder score: PeptoneDB-SAXS, curated from SASBDB {cite('kikhney', 'invernizzi')}."),
        ("off-the-shelf", f"{int(B['n_conformers_median'])} BioEmu-1 conformers per protein, sampled by the PeptoneBench authors from the sequence alone."),
        ("off-the-shelf", f"One scattering curve per conformer, computed by the PeptoneBench authors with Pepsi-SAXS {cite('grudinin')} on each profile's q grid."),
        ("computed here", f"Average the conformer curves with equal weights; fit one scale factor {cite('svergun')} and one constant background; report reduced χ² against the measured errors, and a second score that does not weight by them."),
        ("computed here", "Guinier radius of gyration of the data and of the ensemble curve; flags for aggregation (low-angle upturn), over-subtraction (intensity below −3σ) and oligomers (SASBDB measured mass against sequence mass)."),
        ("computed here", f"Maximum-entropy reweighting {cite('bottaro')} along a path of decreasing prior strength; the effective sample fraction φ kept at χ² targets sorts each entry into one of four kinds."),
        ("computed here", f"The same scoring for {len(comps)} comparator models on the shared entries: four ensemble generators and two single-structure predictors as reference baselines."),
        ("computed here", "Rank candidates for measurement; test whether the error can be predicted before measuring and whether the selection rule beats chance; write the two-arm campaign."),
    ]
    steps_html = "".join(
        f'<li><span class="who{" here" if who == "computed here" else ""}">{who}</span><br>{txt}</li>' for who, txt in steps)

    # ---- expectations table with results
    exp_rows = "".join(
        f'<tr><td><strong>{k}</strong></td><td>{stmt}</td><td class="dim">{why}</td><td>{res}</td></tr>'
        for k, stmt, why, res in (
            ("E1", "Raw χ² is worse for disordered than for folded proteins.",
             "The model was trained mostly on folded structures and molecular dynamics of them.",
             f"<strong>met</strong> · median χ² {f1(E['E1']['median_disordered'])} against {f1(E['E1']['median_folded'])}, one-sided p {pv(E['E1']['p_one_sided'])}"),
            ("E2", "Among folded proteins, raw χ² rises with chain length.",
             "More domains, more ways to misplace them.",
             f"<strong>not met</strong> · Spearman ρ = {E['E2']['spearman_rho']:+.2f} (p = {pv(E['E2']['p'])}); a step: raw fit in {pct(E['E2']['share_fit_le_100'])} of chains ≤ 100 residues and {pct(E['E2']['share_fit_gt_100'])} of longer ones, no slope above that"),
            ("E3", "Most entries reach χ² ≈ 1 while keeping φ > 0.3; a minority need φ < 0.1.",
             "The right conformers are usually present, only mis-weighted.",
             f"<strong>not met</strong> · {pct(E['E3']['share_reaching_chi2_1'])} reach χ² ≈ 1 at all; {pct(E['E3']['share_phi_above_0.3'])} do so keeping φ > 0.3"),
            ("E4", "The ensemble Rg is below the measured Rg for disordered proteins (over-compaction).",
             "The known over-compaction of models trained on folded data.",
             f"<strong>reversed</strong> · median ratio {f2(rg_dis)} (95% interval {ci(rg_ci)}), two-sided Wilcoxon p {pv(E['E4']['p_two_sided'])}; {pct(1 - E['E4']['share_below_1'])} of disordered entries above 1")))

    # ---- tables
    class_rows = "".join(
        f'<tr><td>{k}</td><td class="num">{CLS[k]["n"]}</td><td class="num">{f1(CLS[k]["chi2_raw_median"])}</td>'
        f'<td class="num">{f1(CLS[k]["chi2_raw_q25"])}–{f1(CLS[k]["chi2_raw_q75"])}</td>'
        f'<td class="num">{pct(CLS[k]["share_raw_fit"])} ({pct(H["share_raw_fit_by_class"][k]["ci95"][0])}–{pct(H["share_raw_fit_by_class"][k]["ci95"][1])})</td>'
        f'<td class="num">{f3(nr_cls[k]["nrmsd_raw_median"])} ({ci(nr_cls[k]["nrmsd_raw_ci95"], f3)})</td>'
        f'<td class="num">{CLS[k]["chi2_best_median"]:.2f}</td>'
        f'<td class="num">{pct(CLS[k]["share_not_fit"])}</td><td class="num">{f2(CLS[k]["rg_ratio_median"])}</td></tr>' for k in CLASSES)

    kind_rows = "".join(
        f'<tr><td>{k}</td><td>{desc}</td><td class="num">{RES.get(k, 0)}</td>'
        + "".join(f'<td class="num">{KS[t][k]}</td>' for t in ("0.3", "0.7")) + "</tr>"
        for k, desc in ((KIND_RAW, "raw χ² ≤ 2"), (KIND_MODEST, "χ² ≤ 2 reached while keeping φ ≥ 0.5"),
                        (KIND_STRONG, "χ² ≤ 2 reached only with φ < 0.5"), (KIND_NOTFIT, "χ² ≤ 2 never reached")))

    def comp_row(m, base=False):
        c = comps[m]
        return (f'<tr{" class=base" if base else ""}><td>{MODEL_NAME[m]}{" (single structure)" if base else ""}</td>'
                f'<td class="num">{c["n"]}</td><td class="num">{f1(c["median_chi2_raw"])}</td>'
                f'<td class="num">{f1(c["bioemu_median_chi2_raw_same_entries"])}</td>'
                f'<td class="num">{c["median_log10_chi2_ratio_vs_bioemu"]:+.2f} ({c["ci95_median_log10_chi2_ratio"][0]:+.2f}, {c["ci95_median_log10_chi2_ratio"][1]:+.2f})</td>'
                f'<td class="num">{pct(c["share_better_than_bioemu"])} ({pct(c["ci95_share_better"][0])}–{pct(c["ci95_share_better"][1])})</td>'
                f'<td class="num">{pct(c["share_better_nrmsd"])}</td><td class="num">{f2(others_rg[m])}</td></tr>')
    comp_html = ('<div class="scroll"><table><thead><tr><th>model</th><th>shared entries</th><th>median raw χ²</th>'
                 '<th>BioEmu-1, same entries</th><th>median log₁₀ χ² ratio (95%)</th><th>entries where it beats BioEmu-1 (95%)</th>'
                 '<th>same, by NRMSD</th><th>disordered Rg ratio</th></tr></thead><tbody>'
                 + "".join(comp_row(m) for m in sorted(gens, key=lambda m: comps[m]["median_chi2_raw"]))
                 + "".join(comp_row(m, base=True) for m in sorted(singles, key=lambda m: comps[m]["median_chi2_raw"]))
                 + '</tbody></table></div>')

    arm1 = ARMS["arm1"]["replication"]
    cand_rows = "".join(
        f'<tr><td><a href="https://www.sasbdb.org/data/{r["label"]}/">{r["label"]}</a></td><td class="num">{int(r["length"])}</td>'
        f'<td>{r["disorder_class"]}</td><td class="num">{f1(r["chi2_raw"])}</td><td class="num">{r["chi2_best"]:.2f}</td>'
        f'<td>{r["resolvability"]}</td><td>{r["rg_direction"]}</td><td class="dim">{short_title(r["sasbdb_title"])}</td></tr>'
        for r in arm1)
    check(all(0.6 < r["mw_ratio"] < 1.6 for r in arm1), "arm 1 candidates are monomers by measured mass")

    def arm2_rows(rows):
        return "".join(
            f'<tr><td><a href="https://www.sasbdb.org/data/{r["label"]}/">{r["label"]}</a></td><td class="num">{r["length"]}</td>'
            f'<td class="num">{r["ncpr"]:+.2f}</td><td class="num">{r["fcr"]:.2f}</td><td class="num">{r["rg_model"]}</td>'
            f'<td class="num">{r["rg_law_disordered"]}</td><td class="num">{r["ratio_model_over_law"]:.2f}</td></tr>' for r in rows)
    arm2_head = ('<thead><tr><th>SASBDB</th><th>length</th><th>net charge / residue</th><th>charged fraction</th>'
                 '<th>ensemble Rg (Å)</th><th>scaling-law Rg (Å)</th><th>ratio</th></tr></thead>')

    pred_rows = "".join(
        f'<tr><td>{name}</td><td class="num">{P["regression"][rk]["r2"]["mean"]:.2f} ± {P["regression"][rk]["r2"]["sd"]:.2f}</td>'
        f'<td class="num">{P["regression"][rk]["mae"]["mean"]:.3f}</td>'
        f'<td class="num">{P["classification"][ck]["auc"]["mean"]:.2f} ± {P["classification"][ck]["auc"]["sd"]:.2f}</td>'
        f'<td class="num">{P["classification"][ck]["brier"]["mean"]:.3f}</td></tr>'
        for name, rk, ck in (("constant / base rate", "constant", "rate"), ("class mean / class rate", "class_mean", "class_rate"),
                             ("ridge / logistic regression", "ridge", "logistic"), ("gradient-boosted trees (depth 3)", "hgb", "hgb")))

    sel_rows = "".join(
        f'<tr><td>{p}</td>' + "".join(
            f'<td class="num">{S["policies"][str(k)][p]["strong_or_notfit"]["mean"]:.2f} ± {S["policies"][str(k)][p]["strong_or_notfit"]["sd"]:.2f}</td>'
            f'<td class="num">{S["policies"][str(k)][p]["phi_reduction_chi2_1"]["mean"]:.2f}</td>'
            f'<td class="num">{S["policies"][str(k)][p]["phi_reduction_chi2_2"]["mean"]:.2f}</td>' for k in S["top_k"]) + "</tr>"
        for p in ("random", "predicted error", "predicted priority", "observed priority (ceiling)"))

    dup_rows = "".join(f'<tr><td>{p["a"]}</td><td>{p["b"]}</td><td class="num">{p["jaccard"]:.2f}</td><td class="num">{p["containment"]:.2f}</td></tr>'
                       for p in R["near_duplicates"]["pairs"])

    ex_cells = []
    for cls in CLASSES:
        lab = A["examples"][cls]
        ex = EX[lab]
        ex_cells.append(
            f'<tr><td>{cls}</td><td><a href="https://www.sasbdb.org/data/{lab}/">{lab}</a></td><td class="num">{ex["length"]}</td>'
            f'<td class="num">{ex["chi2"]["raw"]:.2f} → {ex["chi2"]["reweighted"]:.2f}</td><td class="num">{ex["chi2"]["alphafold"]:.1f}</td>'
            f'<td class="num">{ex["rg_exp"]:.1f}</td><td class="num">{ex["rg_raw"]:.1f}</td><td class="num">{ex["rg_reweighted"]:.1f}</td>'
            f'<td class="num">{ex["phi_reweighted"]:.2f}</td><td class="num">{ex["denss"]["chi2_median"]:.2f}</td>'
            f'<td class="num">{ex["denss"]["resolution_A"]:.0f}</td></tr>')
    ex_table = ('<div class="scroll"><table><thead><tr><th>class</th><th>entry</th><th>residues</th><th>χ² raw → reweighted</th>'
                '<th>χ² AlphaFold2</th><th>Rg exp (Å)</th><th>Rg raw</th><th>Rg reweighted</th><th>φ at the minimum</th>'
                '<th>DENSS χ² (median of maps)</th><th>resolution (Å)</th></tr></thead><tbody>' + "".join(ex_cells) + '</tbody></table></div>')

    nav = "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t in [
        ("method", "Method"), ("expectations", "Expectations"), ("map", "Error map"), ("rg", "Size"),
        ("reweight", "Reweighting"), ("models", "Other models"), ("predict", "Predictable?"), ("select", "Selection rule"),
        ("campaign", "Campaign"), ("limits", "Limits"), ("refs", "References")])
    refs_html = "".join(f'<li id="ref-{k}">{t} <a href="{u}">{u.replace("https://", "")}</a></li>' for k, t, u in REFS)
    today = date.today()
    diagram = method_diagram(A["n_entries"], int(B["n_conformers_median"]), n_bioemu, len(comps))
    nd = R["near_duplicates"]
    cr_d, cr_f = CR["disordered"], CR["folded"]

    html = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="description" content="Where BioEmu-1 disagrees with solution scattering, and which proteins to measure next.">
<title>BioEmu against SAXS</title>
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
<p class="scope">{subline}</p>
<p class="byline">Natalija Stepurko · {today:%B %Y} · <a href="{SITE}">natalija-stepurko.com</a></p>
<p class="abstract">{abstract}</p>
<a class="repo" href="{REPO}">{GH}<span><b>All the code is public.</b> Every number and figure on this page comes from the pipeline in this repository, which downloads the public inputs, scores them and writes the campaign proposal.</span><span class="repo-path">Natalija-Stepurko/bioemu-saxs-campaign</span></a>
</header>

<h2 id="method">Method</h2>
<p>A biomolecular emulator predicts the set of shapes a protein adopts in solution, and how often, from its sequence alone. BioEmu-1 was trained on predicted structures, molecular-dynamics simulations and measured stabilities {cite('lewis')}; its successors will be trained with new experimental data, and choosing those experiments is the job this study rehearses. SAXS reports the overall size and shape distribution of a protein in solution, and {A['n_entries']} curated profiles with sequences are public {cite('kikhney', 'invernizzi')}. BioEmu conformers have served as a pool for SAXS ensemble selection on a few dozen proteins, where the unweighted pool did not reproduce the data {cite('haitin')}, and PeptoneBench ranks generators by their fit after reweighting {cite('invernizzi')}; neither asked where the raw model fails by protein class, or what that implies for the next measurements.</p>
<figure class="result diagram"><figcaption>Method · what is taken as given and what this study computes</figcaption><div class="scroll">{diagram}</div></figure>
<ol class="steps">{steps_html}</ol>

<h2 id="expectations">Expectations set before scoring</h2>
<p>Four expectations were written down before any score was computed, each with its reasoning; the result column is filled from the results and the later sections refer back to it.</p>
<div class="scroll"><table class="wide"><thead><tr><th></th><th>expectation</th><th>reasoning</th><th>result</th></tr></thead><tbody>{exp_rows}</tbody></table></div>

<h2 id="map">Where the raw ensemble fails</h2>
{fig("fig_error_map", "Figure 1 · raw fit by protein class and chain length",
     f"Reduced χ² of the uniform BioEmu-1 ensemble average against each SAXS profile, after fitting one scale factor and one constant background, on the {n_bioemu} clean monomer profiles with an ensemble. Left: by disorder class (bars are medians). Right: against chain length, coloured by class.",
     "Points below the dashed line (χ² = 2) are profiles the raw ensemble describes. The median χ² is " + class_sentence() + ". Among folded proteins only the smallest, up to 100 residues, fit more often; above that the error does not change with length (E2).")}
<div class="findings">
<p><strong>The raw ensemble fits {pct(share_fit)} of profiles</strong> (95% interval {pct(H['share_raw_fit']['ci95'][0])}–{pct(H['share_raw_fit']['ci95'][1])}; median χ² {f1(B['chi2_raw_median'])}, quartiles {f1(B['chi2_raw_q25'])}–{f1(B['chi2_raw_q75'])}). By class, {pct(H['share_raw_fit_by_class']['folded']['value'])} of folded, {pct(H['share_raw_fit_by_class']['partly disordered']['value'])} of partly disordered and {pct(H['share_raw_fit_by_class']['disordered']['value'])} of disordered profiles fit (E1 met).</p>
<p><strong>A second score tells a different story about the classes.</strong> χ² weights every point by its reported error. A normalised root-mean-square deviation of ln I over the usable q range, which does not, gives {f3(nr_cls['folded']['nrmsd_raw_median'])} for folded, {f3(nr_cls['partly disordered']['nrmsd_raw_median'])} for partly disordered and {f3(nr_cls['disordered']['nrmsd_raw_median'])} for disordered proteins, with overlapping intervals: the class ordering of the χ² medians is partly a property of the reported errors. The ordering of the seven models is the same under both scores.</p>
<p><strong>Error and chain length.</strong> E2 is not met: the rank correlation among folded proteins is weak (ρ = {E['E2']['spearman_rho']:+.2f}). What the data show is a step: folded proteins of up to 100 residues fit {pct(E['E2']['share_fit_le_100'])} of the time (n = {E['E2']['n_le_100']}), longer ones {pct(E['E2']['share_fit_gt_100'])} (n = {E['E2']['n_gt_100']}).</p>
</div>
<div class="scroll"><table><thead><tr><th>class</th><th>n</th><th>median raw χ²</th><th>quartiles</th><th>raw fit, χ² ≤ 2 (95%)</th><th>median NRMSD (95%)</th><th>median best χ²</th><th>not fit along the path</th><th>median Rg ratio</th></tr></thead><tbody>{class_rows}</tbody></table></div>
<p class="note">NRMSD: root-mean-square deviation of ln I between the scaled model and the data over the points with I &gt; 0 and I ≥ 3σ, divided by the range of ln I of the data over those points. Intervals are percentile bootstraps over entries (2,000 resamples).</p>
{fig("fig_examples", "Figure 2 · one worked example per class, the entry at the class median of raw χ²",
     f"Top: ln I(q) of the measured profile (open circles, every {{n}}th point with its error bar) with three computed curves scaled to it: the raw BioEmu-1 ensemble average (black), the ensemble reweighted to the χ² minimum of its path (blue) and the AlphaFold2 single structure (grey), each labelled with its χ². Middle: the residual (I_model − I_exp)/σ on the same q axis, with a zero line and ±3 guides. Bottom: an ab initio electron-density envelope reconstructed from the measured curve alone with DENSS {cite('grant')} (ten maps in fast mode, aligned and averaged; Dmax from the SASBDB record), drawn as two surfaces: the inner one encloses the protein's expected volume (1.7 Å³ per Da), the fainter outer one the volume DENSS assigned to the particle. The Cα trace of the highest-weight conformer after reweighting is superposed by principal axes; the blue dot marks its first residue.",
     f"A residual that wanders outside ±3 over a stretch of q is a shape error at that length scale (1/q), a flat band within ±3 is a fit at the noise level. For the folded example ({A['examples']['folded']}) all three curves sit near the noise and the conformer fills the envelope. For the partly disordered example ({A['examples']['partly disordered']}) the single structure misses by χ² {EX[A['examples']['partly disordered']]['chi2']['alphafold']:.0f} where the ensemble reaches {EX[A['examples']['partly disordered']]['chi2']['raw']:.1f} raw and {EX[A['examples']['partly disordered']]['chi2']['reweighted']:.2f} reweighted; the envelope shows a globular part with a tail that the top conformer places differently. For the disordered example ({A['examples']['disordered']}) the envelope is an average over many chain conformations, so a single conformer cannot fill it and only {pct(EX[A['examples']['disordered']]['ca_inside_envelope'])} of its Cα atoms fall inside the inner surface; the raw ensemble misses at low q (too large) and reweighting to φ = {EX[A['examples']['disordered']]['phi_reweighted']:.2f} brings the size down to the measured value.").replace("{n}", "20")}
{ex_table}
<p class="note">Rg in Å from Guinier fits (q·Rg ≤ 1.3) of the measured curve, of the raw ensemble curve and of the reweighted curve. φ: effective sample fraction at the χ² minimum of the reweighting path. DENSS χ² is the reconstruction's own fit to the curve.</p>

<h2 id="rg">The size finding: disordered ensembles are too extended</h2>
<p>E4 predicted over-compaction of disordered proteins, the usual failure of models trained on folded structures. The data show the reverse. The ensemble radius of gyration is {f2(rg_dis)} times the measured value at the median for disordered proteins (95% interval {ci(rg_ci)}, two-sided Wilcoxon on the log ratio p {pv(E['E4']['p_two_sided'])}, n = {E['E4']['n']}), above 1 in {pct(1 - E['E4']['share_below_1'])} of them, against {f2(CLS['folded']['rg_ratio_median'])} for folded proteins (interval {ci(R['curve_rg_ratio_by_class']['folded']['ci95'])}). Dropping the five most extreme ratios leaves a median of {f2(trimmed['median'])} ({ci(trimmed['ci95'])}, p {pv(trimmed['p_two_sided_wilcoxon_log'])}). The ratio grows with chain length within the class (Spearman ρ = {rho_len['spearman_rho']:+.2f}, p = {pv(rho_len['p'])}).</p>
<p>Is the extension in the conformers or in the forward model? For every clean disordered entry and {cr_f['n']} random folded ones I computed the radius of gyration from the Cα coordinates of the sampled conformers, and compared it with the measured Rg. For disordered proteins the coordinate Rg is {f2(cr_d['coord_over_exp_median'])} times the measured value ({ci(cr_d['coord_over_exp_ci95'])}), larger than the curve-derived {f2(cr_d['curve_over_exp_median'])}: the sampled chains themselves are too large, and the forward model takes the ratio down, not up (curve over coordinates {f2(cr_d['curve_over_coord_median'])}). For folded proteins the coordinates give {f2(cr_f['coord_over_exp_median'])} and the curves {f2(cr_f['curve_over_exp_median'])}; the difference is the hydration layer that the forward model adds to a compact particle.</p>
{fig("fig_rg_robustness", "Figure 3 · robustness of the size finding",
     f"Left: ensemble Rg over measured Rg for every clean entry, from the curves, by class; bars are the median and its 95% bootstrap interval, disordered points coloured by chain length. Right: the same ratio for the {cr_d['n']} disordered and {cr_f['n']} folded entries whose conformer coordinates were analysed, from the curves and from the Cα coordinates.",
     "A median above 1 means the model's ensemble is larger than the protein in solution. The coordinate-derived ratio for disordered proteins is the direct test: it is above the curve-derived one, so the excess is in the sampled conformers.")}
<p>Among the {A['n_entries']} profiles there are no two identical sequences; {nd['n_pairs']} pairs are near-duplicates (5-mer Jaccard ≥ {nd['threshold_jaccard_5mer']}, about 93% identity or closer), and collapsing each cluster to one value leaves the disordered median at {f2(collapsed['median'])} ({ci(collapsed['ci95'])}) and the raw-fit share at {pct(nd['collapsed_share_raw_fit'])}.</p>
{details(f"the {nd['n_pairs']} near-duplicate pairs", '<div class="scroll"><table><thead><tr><th>entry</th><th>entry</th><th>5-mer Jaccard</th><th>containment</th></tr></thead><tbody>' + dup_rows + '</tbody></table></div>')}
{fig("fig_rg", "Figure 4 · ensemble size against measured size",
     "Radius of gyration of the BioEmu-1 ensemble curve against the Guinier radius of gyration of the experimental profile, both on the experimental q grid, one point per profile.",
     "Points above the diagonal are ensembles that are too extended. A systematic offset within a class is a calibration problem; scatter around the line is a per-protein one.", narrow=True)}

<h2 id="reweight">What reweighting says about each miss</h2>
<p>The ensemble's conformer weights are moved towards the data by maximum-entropy reweighting {cite('bottaro')}, with decreasing strength of the prior, and χ² is tracked against the effective sample fraction φ that survives. Four kinds follow from the path: <em>raw fit</em> (raw χ² ≤ 2), <em>modestly reweightable</em> (χ² ≤ 2 reached while keeping φ ≥ 0.5), <em>strongly reweightable</em> (χ² ≤ 2 reached only with φ &lt; 0.5) and <em>not fit along the path</em>. Of the {n_bioemu} profiles, {fits} are raw fits, {mod} modestly reweightable, {strong} strongly reweightable and {notfit} not fit.</p>
{fig("fig_resolvability", "Figure 5 · reweighting paths and the kinds by class",
     "Left: for every profile, reduced χ² against the effective sample fraction kept as the prior is relaxed (left to right is more reweighting), coloured by disorder class. Right: the share of each class in the four kinds.",
     f"Curves that cross the dashed line early, while φ is still near 1, are modestly reweightable; curves that cross it only far to the right are strongly reweightable; curves that never cross it are not fit. E3 expected most entries to reach χ² ≈ 1 keeping φ > 0.3; {pct(E['E3']['share_phi_above_0.3'])} do.")}
<div class="scroll"><table><thead><tr><th>kind</th><th>definition</th><th>n (φ threshold 0.5)</th><th>threshold 0.3</th><th>threshold 0.7</th></tr></thead><tbody>{kind_rows}</tbody></table></div>
<p class="note">The threshold moves entries between the two reweightable kinds only; the raw-fit and not-fit counts do not depend on it.</p>
<div class="call"><p><strong>Interpretation.</strong> A strongly reweightable entry is read as one where the right conformers are present and mis-weighted, so a profile of it carries a usable training signal; a not-fit entry needs a different measurement or a different model, because SAXS says the ensemble is wrong without saying how; a modestly reweightable entry needs a small, consistent correction, often of the Rg scale. Two cautions limit this reading: φ measures how concentrated the weights become, not information content; and SAXS is low-dimensional, so distinct ensembles can give the same curve.</p></div>

<h2 id="models">Other models on the same profiles</h2>
<p>The same archive carries back-calculated curves from other models for the same entries, scored here identically and without reweighting. The primary comparison is with the ensemble generators {", ".join(MODEL_NAME[m] for m in gens)}; the single-structure predictors {" and ".join(MODEL_NAME[m] for m in singles)} are reference baselines, since one structure cannot be reweighted and its best fit is its raw fit. Five further folders are not used: Boltz-1x and ESMFlow are superseded by models already included, idpGAN and IDP-o are disorder-only generators represented here by idpSAM, and PepTron-base is the un-finetuned PepTron. AF-CALVADOS {cite('bulow')} is a further ensemble method evaluated on this benchmark whose predictions were not in the archive.</p>
<p>Among the six comparators on the shared entries, BioEmu-1 has the lowest median raw χ² and the ranking is the same under the error-free score. PepTron comes closest: it beats BioEmu-1 on {pct(pep['share_better_than_bioemu'])} of the {pep['n']} shared entries (95% interval {pct(pep['ci95_share_better'][0])}–{pct(pep['ci95_share_better'][1])}), with a median log₁₀ χ² ratio of {pep['median_log10_chi2_ratio_vs_bioemu']:+.2f}. On disordered proteins, BioEmu-1 is the only one of the seven whose median Rg ratio is above 1.1; {", ".join(f"{MODEL_NAME[m]} {f2(others_rg[m])}" for m in sorted(others_rg, key=others_rg.get, reverse=True))} on the same entries.</p>
{fig("fig_models", "Figure 6 · raw fit of each model by class", "Median raw reduced χ² (bars: quartiles) of the unweighted ensemble of each model, by disorder class, on the profiles that pass quality checks; hatched bars are the single-structure baselines.", "Lower is better. Differences within a class smaller than the quartile ranges are not read.")}
{comp_html}
<p class="note">Intervals: percentile bootstrap over the shared entries. The log₁₀ χ² ratio is comparator over BioEmu-1, positive when BioEmu-1 fits better.</p>

<h2 id="predict">Can the error be predicted before measuring?</h2>
<p>A campaign that chooses what to measure needs to know, before measuring, where the model is likely to be wrong. I fitted predictors of the raw error (log₁₀ χ²) and of the kind (strongly reweightable or not fit) from quantities known before a profile exists: chain length, disorder score and class, sequence composition (charge, hydrophobic, aromatic, proline and glycine fractions), the Rg expected from length by the scaling laws for folded and for disordered chains {cite('kohn')}, and the model's own ensemble Rg and conformer spread. Nothing from the experimental curve enters. Evaluation is 5-fold cross-validation grouped by sequence cluster so that constructs of one protein never straddle a fold, repeated five times.</p>
<div class="scroll"><table><thead><tr><th>model</th><th>R² (log₁₀ χ²)</th><th>MAE</th><th>AUC (kind)</th><th>Brier</th></tr></thead><tbody>{pred_rows}</tbody></table></div>
<p class="note">Mean ± sd over the five repeats; n = {P['n']} entries in {P['n_groups']} sequence groups; positive rate {pct(P['positive_rate'])}.</p>
<p>The error is not predictable from these features at this sample size: the best regression reaches R² = {f2(r2_best)} (class means alone give {f2(P['regression']['class_mean']['r2']['mean'])}), and the best classifier an AUC of {f2(auc_best)} with a Brier score no better than the base rate. The one feature the tree model leans on is the {FEATURE_LABEL.get(top_imp['feature'], top_imp['feature'].replace('_', ' '))} (importance {top_imp['mean']:.3f} ± {top_imp['sd']:.3f}), and the calibration curve is close to the diagonal only because the predictions hardly leave the base rate.</p>
{fig("fig_predict", "Figure 7 · predicting the error, and what the selection rule picks",
     "a: observed against predicted log₁₀ raw χ², out of fold, coloured by class. b: calibration of the tree classifier, quintiles of predicted probability against the observed share. c: permutation importance of the ten most used features. d: the share of strongly reweightable or not-fit entries among the top 10 and top 20 entries chosen by each acquisition policy on held-out folds; the band is the pool mean ± the sd of one random pick of ten.",
     "In a, a predictive model would spread the points along the diagonal; here they form a vertical band. In d, a policy is useful when its bar stands clear of the random band; the ceiling bar shows what the rule does when fed the observed quantities, which is what the current candidate ranking does.")}

<h2 id="select">Does the selection rule pick informative measurements?</h2>
<p>The campaign ranks candidates by log₁₀ raw χ² × a kind weight × a tractability weight. To test the rule without circularity I treated the {n_bioemu} profiles as a pool and asked, on held-out folds, whether the rule fed with <em>predicted</em> quantities (from the models above, fitted on the training folds) picks entries whose profiles would move the ensemble. Three policies: random; largest predicted raw error; the priority rule with predicted error and predicted kind weight. The same rule with the observed quantities is the ceiling. Objectives among the top k: the share of strongly reweightable or not-fit entries, and the mean φ-reduction (1 − φ at the χ² target, 1 when never reached).</p>
<div class="scroll"><table><thead><tr><th>policy</th><th>top 10: share strongly reweightable or not fit</th><th>φ-reduction at χ² = 1</th><th>at χ² = 2</th><th>top 20: share</th><th>φ-reduction at χ² = 1</th><th>at χ² = 2</th></tr></thead><tbody>{sel_rows}</tbody></table></div>
<p class="note">Mean ± sd over five repeats of the grouped 5-fold split; random is averaged over {S['n_random_draws_per_fold']} draws per fold. The pool shares are {pct(S['pool_mean']['strong_or_notfit'])} strongly reweightable or not fit and a φ-reduction of {f2(S['pool_mean']['phi_reduction_chi2_1'])} at χ² = 1 ({pct(1 - E['E3']['share_reaching_chi2_1'])} of entries never reach it, which is why that objective hardly moves).</p>
<p>With predicted quantities the rule is no better than chance: {f2(pol['predicted priority']['strong_or_notfit']['mean'])} of the top ten are strongly reweightable or not fit against {f2(pol['random']['strong_or_notfit']['mean'])} for random choice, a difference of {gain_prio:+.2f} where a single random pick of ten spreads by ±{f2(rnd_sd)}. Ranking by predicted error alone does slightly better ({f2(pol['predicted error']['strong_or_notfit']['mean'])}, {gain_err:+.2f} ± {f2(pol['predicted error']['strong_or_notfit']['minus_random_sd'])} over repeats), still within that spread. With the observed quantities the rule picks {f2(ceil)}, so the rule itself separates the kinds; what is missing is a predictor of them. This evaluation is retrospective: a profile that already exists carries no new information for the model, and the test says only how far the rule can be trusted to rank unmeasured proteins from what is known about them.</p>

<h2 id="campaign">The measurement campaign</h2>
<p>The <a href="{REPO}/blob/main/docs/CAMPAIGN.md">proposal</a> has two arms, each with constructs, assay, scripted quality criteria, success criteria, contingencies and work packages for an external provider.</p>
<h3>Arm 1 · model improvement: high-error, reweightable monomers under one standard condition</h3>
<p>The top of the ranking, restricted to the two reweightable kinds:</p>
<div class="scroll"><table><thead><tr><th>SASBDB</th><th>length</th><th>class</th><th>raw χ²</th><th>best χ²</th><th>kind</th><th>Rg</th><th>deposited study</th></tr></thead><tbody>{cand_rows}</tbody></table></div>
<p class="note">Every candidate's measured molecular weight matches its sequence mass within the monomer range.</p>
<p>Re-measuring deposited systems under one buffer, one temperature, with tags removed and a concentration series controls the conditions the model does not see, and so removes the part of the raw error that is condition mismatch; it adds no new region of sequence space. A novel-acquisition sub-list, sequences without a SASBDB entry chosen by the error predictor, is not offered, because that predictor is not predictive (R² = {f2(r2_best)}); only the replication list stands. The kind and tractability weights of the ranking are heuristics set by judgement; the section above says how far they were validated: the rule separates the kinds when fed observed quantities and does not when fed predictions.</p>
{fig("fig_candidates", "Figure 8 · the ranked candidates", "Measurement priority of the top twenty entries: log₁₀ raw χ² × kind weight (strongly reweightable 1.0, modestly reweightable 0.6, not fit 0.3) × tractability weight (chain ≤ 350 residues; folded or partly disordered).", "Each bar is a protein whose SAXS profile the model currently misses and whose measurement under one standard condition would constrain it.", narrow=True)}
<h3>Arm 2 · hypothesis test: are BioEmu-1's disordered ensembles too extended?</h3>
<p>The test arm takes disordered proteins whose BioEmu-1 ensemble Rg exceeds the scaling-law Rg for disordered chains (1.927 N<sup>0.598</sup> Å {cite('kohn')}) by more than 10%, a quantity computed from the model alone, spanning chain length and net charge ({arm2['n_test_candidates']} of {arm2['n_disordered_pool']} disordered entries qualify); matched controls are disordered proteins of the same length bins whose ratio is within 10% of 1 ({arm2['n_control_candidates']} qualify). Pre-specified outcome: the median of Rg<sub>exp</sub>/Rg<sub>model</sub> across the test arm from Guinier fits of the new profiles. The bias is refuted if that median is at least 0.95; it is confirmed if it is at most 0.90 in the test arm and at least 0.95 in the controls. In the archive the test candidates have a median observed Rg<sub>model</sub>/Rg<sub>exp</sub> of {f2(arm2['test_observed_median_ratio'])} and the controls {f2(arm2['control_observed_median_ratio'])}; the campaign repeats this under one condition with the selection made before any curve is seen.</p>
{details("test arm and controls", '<div class="scroll"><table>' + arm2_head + '<tbody>' + arm2_rows(arm2["test"]) + '</tbody></table></div><p class="note">Controls</p><div class="scroll"><table>' + arm2_head + '<tbody>' + arm2_rows(arm2["controls"]) + '</tbody></table></div>')}
<p>The assay is size-exclusion-coupled SAXS at three concentrations with a protein standard in every session, tags removed, one standard buffer recorded with each dataset, and a repeat of one deposited entry per batch as a cross-site control. Six quality criteria (Guinier linearity, no low-angle upturn, no radiation damage across frames, molecular weight from I(0) within 20% of the sequence mass, no negative intensities, control reproduced at χ² &lt; 2) decide pass or fail from recorded values. Not-fit cases get hydrogen–deuterium exchange mass spectrometry on the same batch of protein. Every batch of twelve constructs re-runs this pipeline on its new profiles, so the error map is updated before the next batch is chosen.</p>

<h2 id="limits">Limits</h2>
<p>The ensembles are from the PeptoneBench archive, generated with BioEmu at code commit ac7455d; the archive does not record the checkpoint version, and re-sampling with the current v1.2 checkpoint is the first follow-up. SASBDB profiles differ in buffer, temperature and construct, none of which the model sees, so part of the raw error is condition mismatch, which a campaign under one standard condition removes. The forward model is treated as exact; its hydration-shell parameters bias Rg by a few per cent, which is why modestly reweightable cases are read with care, and the coordinate-derived Rg of the disordered conformers stands apart from it. Disorder classes come from a sequence-based predictor, not from the data. The model comparison is restricted to the six comparators on the shared entries. The validation experiments are retrospective and use {n_bioemu} entries; a predictor that fails at this size could succeed with more data or with features from the model's own ensemble beyond its Rg and spread. Profiles with an upturn above {int(100 * A['thresholds']['aggregation_upturn'])}%, an intensity more than three standard errors below zero, no valid Guinier region, or a measured mass above 1.6 times the sequence mass are kept out of every statistic ({A['n_flag_aggregation']}, {A['n_flag_negative']}, {A['n_guinier_invalid']} and {A['n_flag_oligomer']} profiles; the oligomers' median raw χ² is {f1(A['oligomer_median_chi2_raw'])}).</p>

<h2 id="refs">References</h2>
<ol class="refs">{refs_html}</ol>
<footer>
<div>Data: PeptoneDB-SAXS and PeptoneBench predictions (Zenodo 17306061, CC-BY 4.0), curated from SASBDB. Downloaded at run time and not redistributed.</div>
<div>Code: <a href="{REPO}">bioemu-saxs-campaign</a> · design: <a href="{REPO}/blob/main/docs/DESIGN.md">design document</a> · proposal: <a href="{REPO}/blob/main/docs/CAMPAIGN.md">campaign</a>.</div>
</footer>
</div>
<script>{NAV_JS}</script>
</html>
"""
    low = html.lower()
    for bad in ("rather than", "instead of", "by contrast", "not assumed"):
        check(bad not in low, f"forbidden phrase: {bad}")
    OUT.write_text(html)
    print(f"wrote {OUT}  {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build()
