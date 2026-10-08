"""Build docs/index.html from results/. Every number on the page is read from results/analysis.json,
results/candidates.csv and results/by_class.csv; checks stop the build if the prose and the numbers
disagree.

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
B, E = A["bioemu"], A["expectations"]
CLS = {r["disorder_class"]: r for r in A["by_class"] if r["model"] == "bioemu"}
RES = B["resolvability_counts"]
MODEL_NAME = {"bioemu": "BioEmu-1", "alphafold2": "AlphaFold2", "boltz2": "Boltz-2", "idpfold2": "IDPFold2",
              "peptron": "PepTron", "boltz1x": "Boltz-1x", "esmflow": "ESMFlow", "idpsam": "idpSAM"}


def check(cond, msg):
    if not cond:
        sys.exit(f"page text no longer matches results/: {msg}")


def pct(x):
    return f"{100 * x:.0f}%"


def f1(x):
    return f"{x:.1f}"


def pv(p):
    return "< 0.001" if p < 1e-3 else f"{p:.2g}" if p < 0.01 else f"{p:.2f}"


def data_uri(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def fig(name, caption, what, read):
    p = RESULTS / "figures" / f"{name}.png"
    check(p.exists(), f"figure {name} exists")
    return (f'<figure class="result"><figcaption>{caption}</figcaption><img src="{data_uri(p)}" alt="{caption}">'
            f'<div class="rtext"><p><strong>What it shows.</strong> {what}</p><p><strong>How to read it.</strong> '
            f'{read}</p></div></figure>')


def met(k):
    return "met" if E[k]["met"] else "not met"


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
]
REFNO = {k: i + 1 for i, (k, _, _) in enumerate(REFS)}


def cite(*keys):
    return "[" + ", ".join(f'<a href="#ref-{k}">{REFNO[k]}</a>' for k in keys) + "]"


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
.topnav ul{list-style:none;margin:0;padding:0;display:flex;gap:15px;font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase}
.topnav ul a{display:block;color:var(--muted);text-decoration:none;white-space:nowrap;padding:17px 0 15px;border-bottom:2px solid transparent}
.topnav ul a:hover{color:var(--ink)}.topnav ul a.active{color:var(--ink);border-bottom-color:var(--ink)}
.navtoggle{display:none;font:inherit;font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink);
  background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:6px 10px;cursor:pointer;max-width:60vw;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.navtoggle::before{content:"\\2261\\00a0\\00a0";font-size:13px}
@media (max-width:1000px){.navtoggle{display:block}.topnav ul{display:none;position:absolute;top:54px;right:0;flex-direction:column;gap:0;min-width:220px;
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
.result img{width:100%;height:auto;display:block}
.result .rtext p{font-size:12.5px;color:var(--muted);margin:8px 0 0}.result .rtext strong{color:var(--ink);font-weight:600}
.findings{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:18px 0}
@media (max-width:860px){.findings{grid-template-columns:minmax(0,1fr)}}
.findings>p{background:var(--panel);border:1px solid var(--rule);border-top:3px solid var(--ink);border-radius:3px;padding:14px 16px;margin:0;font-size:14px}
.call{border-left:3px solid var(--acc);background:#EEF2F6;padding:14px 20px;margin:18px 0;border-radius:0 3px 3px 0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule)}
td{padding:8px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
td.num{font-family:var(--mono);font-size:12.5px;text-align:right;white-space:nowrap}
.note{color:var(--muted);font-size:13px}
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
    n_clean = A["n_clean"]
    cls_n = {k: CLS[k]["n"] for k in CLS}
    share_fit = B["share_raw_fit_chi2_2"]
    pop, cal, unres, fits = (RES.get(k, 0) for k in ("population", "calibration", "unresolved", "fits"))
    check(pop + cal + unres + fits == sum(RES.values()), "resolvability counts sum")
    comps = A["model_comparison"]
    top = CAND.head(12)

    def class_sentence():
        parts = [f"{f1(CLS[k]['chi2_raw_median'])} for {k} proteins (n = {cls_n[k]})"
                 for k in ("folded", "partly disordered", "disordered")]
        return ", ".join(parts[:-1]) + " and " + parts[-1]

    worst = max(CLS, key=lambda k: CLS[k]["chi2_raw_median"])
    best_cls = min(CLS, key=lambda k: CLS[k]["chi2_raw_median"])
    headline = (f"BioEmu-1 ensembles fit {pct(share_fit)} of {n_clean} solution-scattering profiles as they are, "
                f"and the misses concentrate in {worst} proteins")
    abstract = (
        f"BioEmu-1 {cite('lewis')} emulates a protein's conformational ensemble from its sequence. The next model "
        f"will be trained with new measurements, and that campaign should start where the current model is wrong "
        f"and where a measurement can tell. I scored the raw BioEmu-1 ensemble, with no reweighting, against "
        f"{A['n_entries']} small-angle X-ray scattering (SAXS) profiles from SASBDB {cite('kikhney')}, using the "
        f"public ensembles and back-calculated curves of PeptoneBench {cite('invernizzi')}; {n_clean} profiles pass "
        f"data-quality checks. The ensemble fits {pct(share_fit)} of them at reduced χ² ≤ 2 (median χ² "
        f"{f1(B['chi2_raw_median'])}); the median is {f1(CLS[best_cls]['chi2_raw_median'])} for {best_cls} and "
        f"{f1(CLS[worst]['chi2_raw_median'])} for {worst} proteins. Reweighting each ensemble towards its profile "
        f"sorts the misses: {cal} need a modest correction, {pop} need their populations moved, and {unres} cannot "
        f"be fitted from the conformers the model proposes. The campaign proposal built from this map names "
        f"the systems, the assay, scripted quality criteria and the work packages for an external provider.")

    comp_rows = "".join(
        f'<tr><td>{MODEL_NAME.get(c["model"], c["model"])}</td><td class="num">{c["n"]}</td>'
        f'<td class="num">{f1(c["median_chi2_raw"])}</td><td class="num">{f1(c["bioemu_median_chi2_raw_same_entries"])}</td>'
        f'<td class="num">{pct(c["share_better_than_bioemu"])}</td><td class="num">{pv(c["p_two_sided"])}</td></tr>'
        for c in comps)
    comp_html = (f'<div class="scroll"><table><thead><tr><th>model</th><th>shared entries</th><th>median raw χ²</th>'
                 f'<th>BioEmu-1 on the same entries</th><th>entries where it beats BioEmu-1</th><th>p (Wilcoxon)</th>'
                 f'</tr></thead><tbody>{comp_rows}</tbody></table></div>') if comps else \
        '<p class="note">The archive provided no comparator ensembles for these entries.</p>'

    cand_rows = "".join(
        f'<tr><td><a href="https://www.sasbdb.org/data/{r.label}/">{r.label}</a></td><td class="num">{int(r.length)}</td>'
        f'<td>{r.disorder_class}</td><td class="num">{f1(r.chi2_raw)}</td><td class="num">{r.chi2_best:.2f}</td>'
        f'<td>{r.resolvability}</td><td>{r.rg_direction}</td></tr>' for r in top.itertuples())

    class_rows = "".join(
        f'<tr><td>{k}</td><td class="num">{CLS[k]["n"]}</td><td class="num">{f1(CLS[k]["chi2_raw_median"])}</td>'
        f'<td class="num">{f1(CLS[k]["chi2_raw_q25"])}–{f1(CLS[k]["chi2_raw_q75"])}</td>'
        f'<td class="num">{pct(CLS[k]["share_raw_fit"])}</td><td class="num">{CLS[k]["chi2_best_median"]:.2f}</td>'
        f'<td class="num">{CLS[k]["phi_at_chi2_1_median"]:.2f}</td><td class="num">{pct(CLS[k]["share_unresolved"])}</td>'
        f'<td class="num">{CLS[k]["rg_ratio_median"]:.2f}</td></tr>' for k in ("folded", "partly disordered", "disordered"))

    exp_rows = "".join(
        f'<tr><td>{k}</td><td>{E[k]["statement"]}</td><td>{detail}</td><td><strong>{met(k)}</strong></td></tr>'
        for k, detail in (
            ("E1", f"median χ² {f1(E['E1']['median_disordered'])} against {f1(E['E1']['median_folded'])}; "
                   f"one-sided p = {pv(E['E1']['p_one_sided'])}"),
            ("E2", f"Spearman ρ = {E['E2']['spearman_rho']:+.2f}, p = {pv(E['E2']['p'])}, n = {E['E2']['n']}"),
            ("E3", f"{pct(E['E3']['share_phi_above_0.3'])} reach χ² ≈ 1 keeping φ > 0.3; "
                   f"{pct(E['E3']['share_phi_below_0.1'])} need φ < 0.1"),
            ("E4", f"median Rg ratio {E['E4']['median_rg_ratio_disordered']:.2f}, {pct(E['E4']['share_below_1'])} below 1; "
                   f"one-sided p = {pv(E['E4']['p_one_sided'])}")))

    nav = "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t in [
        ("question", "Question"), ("map", "Error map"), ("resolve", "What SAXS resolves"), ("models", "Other models"),
        ("campaign", "Campaign"), ("approach", "Approach"), ("expectations", "Expectations"), ("limits", "Limits"),
        ("refs", "References")])
    refs_html = "".join(f'<li id="ref-{k}">{t} <a href="{u}">{u.replace("https://", "")}</a></li>' for k, t, u in REFS)
    today = date.today()

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
<p class="scope">Raw BioEmu-1 ensembles against {n_clean} SASBDB profiles · reweighting to measure what the data can resolve · a measurement campaign built from the map</p>
<p class="byline">Natalija Stepurko · {today:%B %Y} · <a href="{SITE}">natalija-stepurko.com</a></p>
<p class="abstract">{abstract}</p>
<a class="repo" href="{REPO}">{GH}<span><b>All the code is public.</b> Every number and figure on this page comes from the pipeline in this repository, which downloads the public inputs, scores them and writes the campaign proposal.</span><span class="repo-path">Natalija-Stepurko/bioemu-saxs-campaign</span></a>
</header>

<h2 id="question">The question</h2>
<p>A biomolecular emulator predicts the set of shapes a protein adopts in solution, and how often, from its sequence alone. BioEmu-1 was trained on predicted structures, molecular-dynamics simulations and measured stabilities {cite('lewis')}. Its successors will be trained with new experimental data. Choosing those experiments is the job this study rehearses: where is the model wrong, where can a measurement tell, and what should be commissioned first. SAXS is the measurement examined here, because it reports the overall size and shape distribution of a protein in solution and because {A['n_entries']} curated profiles with sequences are public {cite('kikhney', 'invernizzi')}.</p>
<p>Two things were known. BioEmu conformers have served as a pool for SAXS ensemble selection on a few dozen proteins, where the unweighted pool did not reproduce the data and selection was needed {cite('haitin')}. And PeptoneBench ranks ensemble generators, BioEmu among them, by their fit after reweighting {cite('invernizzi')}. What neither asked is where the raw model fails, by protein class, and what that implies for the next measurements.</p>

<h2 id="map">The error map</h2>
{fig("fig_error_map", "Figure 1 · raw fit by protein class and chain length",
     f"Reduced χ² of the uniform BioEmu-1 ensemble average against each SAXS profile, after fitting one scale factor and one constant background, on the {n_clean} profiles that pass data-quality checks. Left: by disorder class (bars are medians). Right: against chain length, coloured by class.",
     "Points below the dashed line (χ² = 2) are profiles the raw ensemble describes. The median χ² is " + class_sentence() + ".")}
<div class="findings">
<p><strong>The raw ensemble fits {pct(share_fit)} of profiles.</strong> Median reduced χ² is {f1(B['chi2_raw_median'])} (quartiles {f1(B['chi2_raw_q25'])}–{f1(B['chi2_raw_q75'])}). Expectation E1, that disordered proteins fit worse than folded ones, is {met('E1')} (p = {pv(E['E1']['p_one_sided'])}).</p>
<p><strong>Error and chain length.</strong> Among folded proteins the raw χ² {'rises' if E['E2']['spearman_rho'] > 0 else 'does not rise'} with length (Spearman ρ = {E['E2']['spearman_rho']:+.2f}, p = {pv(E['E2']['p'])}); expectation E2 is {met('E2')}.</p>
<p><strong>Size.</strong> For disordered proteins the ensemble radius of gyration is {E['E4']['median_rg_ratio_disordered']:.2f} times the experimental value at the median, below 1 in {pct(E['E4']['share_below_1'])} of cases; expectation E4 (over-compaction) is {met('E4')}.</p>
</div>
{fig("fig_rg", "Figure 2 · ensemble size against measured size",
     "Radius of gyration of the BioEmu-1 ensemble curve against the Guinier radius of gyration of the experimental profile, both on the experimental q grid, one point per profile.",
     "Points below the diagonal are ensembles that are too compact. A systematic offset within a class is a calibration problem; scatter around the line is a per-protein one.")}
<div class="scroll"><table><thead><tr><th>class</th><th>n</th><th>median raw χ²</th><th>quartiles</th><th>raw fit (χ² ≤ 2)</th><th>median best χ²</th><th>median φ at χ² = 1</th><th>unresolved</th><th>median Rg ratio</th></tr></thead><tbody>{class_rows}</tbody></table></div>
<p class="note">φ: the effective sample fraction kept when the ensemble is reweighted until it fits at χ² = 1 (1 = no reweighting). Unresolved: χ² ≤ 2 is never reached along the reweighting path.</p>

<h2 id="resolve">What SAXS can and cannot resolve</h2>
<p>A poor raw fit has three possible meanings, and the reweighting path tells them apart. The ensemble's conformer weights are moved towards the data by maximum-entropy reweighting {cite('bottaro')}, with decreasing strength of the prior, and χ² is tracked against the effective sample fraction φ that survives. An entry that reaches χ² ≤ 2 while keeping half its effective sample needs a <em>calibration</em>; one that reaches it only after φ falls below 0.5 needs its <em>populations</em> moved; one that never reaches it is <em>unresolved</em> by SAXS from the conformers the model proposes.</p>
{fig("fig_resolvability", "Figure 3 · reweighting paths and the resolvability of each class",
     "Left: for every profile, reduced χ² against the effective sample fraction kept as the prior is relaxed (left to right is more reweighting), coloured by disorder class. Right: the share of each class in the four kinds.",
     f"Of {sum(RES.values())} profiles, {fits} fit as they are, {cal} are calibration cases, {pop} are population cases and {unres} are unresolved. Population cases are where a new SAXS profile carries the most training signal: the right conformers are in the ensemble, and the measurement says how to weight them.")}
<div class="call"><p><strong>For the campaign.</strong> New SAXS data moves the model most on population cases, and least on calibration cases, which a consistent correction handles. Unresolved cases need a measurement that reports local structure, because SAXS says the ensemble is wrong without saying how. Expectation E3 is {met('E3')}: {pct(E['E3']['share_phi_above_0.3'])} of profiles reach χ² ≈ 1 while keeping more than 30% of the effective sample.</p></div>

<h2 id="models">Other models on the same profiles</h2>
<p>The same archive carries ensembles from other generators for the same entries, scored here identically and without reweighting.</p>
{fig("fig_models", "Figure 4 · raw fit of each model by class", "Median raw reduced χ² (bars: quartiles) of the unweighted ensemble of each model, by disorder class, on the profiles that pass quality checks.", "Lower is better. Differences within a class smaller than the quartile ranges are not read.") if comps else ""}
{comp_html}

<h2 id="campaign">The measurement campaign</h2>
<p>The <a href="{REPO}/blob/main/docs/CAMPAIGN.md">proposal</a> ranks candidate systems by the model's error, by whether SAXS can resolve it and by tractability (chain length, disorder class), then sets out the assay, the quality criteria that a script applies to every dataset, the success criteria, the contingencies and the work packages for an external provider. The top of the ranking:</p>
<div class="scroll"><table><thead><tr><th>SASBDB</th><th>length</th><th>class</th><th>raw χ²</th><th>best χ²</th><th>kind</th><th>Rg</th></tr></thead><tbody>{cand_rows}</tbody></table></div>
{fig("fig_candidates", "Figure 5 · the ranked candidates", "Measurement priority of the top twenty entries: log₁₀ raw χ² × resolvability weight (population 1.0, calibration 0.6, unresolved 0.3) × tractability weight (chain ≤ 350 residues; folded or partly disordered).", "Each bar is a protein whose SAXS profile the model currently misses and whose measurement under one standard condition would constrain it.")}
<p>The assay is size-exclusion-coupled SAXS at three concentrations with a protein standard in every session, tags removed, one standard buffer recorded with each dataset, and a repeat of one deposited entry per batch as a cross-site control. Six quality criteria (Guinier linearity, no low-angle upturn, no radiation damage across frames, molecular weight from I(0) within 20% of the sequence mass, no negative intensities, control reproduced at χ² &lt; 2) decide pass or fail from recorded values. Unresolved cases get hydrogen–deuterium exchange mass spectrometry on the same batch of protein. Every batch of twelve constructs re-runs this pipeline on its new profiles, so the error map is updated before the next batch is chosen.</p>

<h2 id="approach">Approach</h2>
<p>The {A['n_entries']} profiles, with sequences, pH and per-residue disorder scores, are PeptoneDB-SAXS {cite('invernizzi')}, curated from SASBDB {cite('kikhney')}. The BioEmu-1 ensembles and their back-calculated curves (Pepsi-SAXS {cite('grudinin')}) are the PeptoneBench predictions; this study re-analyses them and does not re-sample. Each raw fit scales the ensemble average to the data with one factor {cite('svergun')} and one constant background and reports reduced χ² against the experimental errors. The Guinier radius of gyration is fitted on the low-angle region (q·Rg ≤ 1.3), iterated, with the fit taken from the upper part of the window, so a low-angle upturn registers as an aggregation flag and does not inflate Rg. Profiles with an upturn above {int(100 * A['thresholds']['aggregation_upturn'])}%, negative intensities or no valid Guinier region are reported but kept out of every statistic ({A['n_flag_aggregation']}, {A['n_flag_negative']} and {A['n_guinier_invalid']} profiles). Disorder classes use the mean per-residue score: folded below 0.2, disordered above 0.6.</p>

<h2 id="expectations">Expectations set before scoring</h2>
<div class="scroll"><table><thead><tr><th></th><th>expectation</th><th>result</th><th></th></tr></thead><tbody>{exp_rows}</tbody></table></div>

<h2 id="limits">Limits</h2>
<p>The ensembles were generated by the PeptoneBench authors with BioEmu-1 at their settings; a re-sampling with the current checkpoint is the first check once GPU time is available. SASBDB profiles differ in buffer, temperature and construct, none of which the model sees, so part of the raw error is condition mismatch, which a campaign under one standard condition removes. The forward model is treated as exact; its hydration-shell parameters bias Rg by a few per cent, which is why calibration cases are read with care. Disorder classes come from a sequence-based predictor, not from the data. The comparison between models uses only the entries each archive folder provides.</p>

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
