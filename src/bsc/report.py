"""Stage `report`: figures from results/ and the campaign proposal (docs/CAMPAIGN.md) whose
numbers and candidate table are read from results/analysis.json and results/candidates.csv.
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bsc import config as C  # noqa: E402

INK, MUTED, RULE = "#16191D", "#5B646E", "#DDE1E4"
CLS_COL = {"folded": "#2F5D8A", "partly disordered": "#8A6FA8", "disordered": "#C2681A"}
MODEL_NAME = {"bioemu": "BioEmu-1", "alphafold2": "AlphaFold2", "boltz2": "Boltz-2", "idpfold2": "IDPFold2",
              "peptron": "PepTron", "boltz1x": "Boltz-1x", "esmflow": "ESMFlow", "idpsam": "idpSAM"}
CLASS_ORDER = [c[0] for c in C.DISORDER_CLASSES]
OUT = C.RESULTS / "figures"


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": RULE,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def fig_error_map(d: pd.DataFrame, A: dict) -> None:
    """Raw chi2 of BioEmu-1 by disorder class and length: the error map."""
    m = d[(d["model"] == C.MODEL) & d["clean"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.1, 1.6]})
    ax = axes[0]
    for i, cls in enumerate(CLASS_ORDER):
        v = np.log10(m.loc[m["disorder_class"] == cls, "chi2_raw"])
        x = i + np.random.default_rng(i).uniform(-0.22, 0.22, len(v))
        ax.scatter(x, v, s=14, color=CLS_COL[cls], alpha=0.65, lw=0)
        ax.hlines(v.median(), i - 0.3, i + 0.3, color=INK, lw=2)
        ax.text(i, 3.55, f"n = {len(v)}", ha="center", fontsize=9, color=MUTED)
    ax.axhline(np.log10(2), color=RULE, lw=1, ls="--")
    ax.text(2.45, np.log10(2) + 0.04, "χ² = 2", fontsize=8.5, color=MUTED, ha="right")
    ax.set_xticks(range(3))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylabel("raw reduced χ² (log10)")
    ax.set_title("Unweighted BioEmu-1 ensemble against each SAXS profile", fontsize=10, loc="left")
    ax.set_ylim(-0.6, 3.7)
    ax = axes[1]
    for cls in CLASS_ORDER:
        g = m[m["disorder_class"] == cls]
        ax.scatter(g["length"], np.log10(g["chi2_raw"]), s=14, color=CLS_COL[cls], alpha=0.7, lw=0, label=cls)
    ax.axhline(np.log10(2), color=RULE, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("chain length (residues)")
    ax.set_ylabel("raw reduced χ² (log10)")
    ax.set_ylim(-0.6, 3.7)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    e2 = A["expectations"]["E2"]
    ax.set_title(f"Folded proteins: Spearman ρ = {e2['spearman_rho']:+.2f}, p = {e2['p']:.2g}", fontsize=10,
                 loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_error_map.png")
    plt.close(fig)


def fig_rg(d: pd.DataFrame) -> None:
    """Ensemble Rg against experimental Guinier Rg."""
    m = d[(d["model"] == C.MODEL) & d["clean"]].dropna(subset=["rg_exp", "rg_model"])
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = [0.8 * min(m["rg_exp"].min(), m["rg_model"].min()), 1.1 * max(m["rg_exp"].max(), m["rg_model"].max())]
    ax.plot(lim, lim, color=RULE, lw=1)
    for cls in CLASS_ORDER:
        g = m[m["disorder_class"] == cls]
        ax.scatter(g["rg_exp"], g["rg_model"], s=16, color=CLS_COL[cls], alpha=0.75, lw=0, label=cls)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("experimental Rg (Å, Guinier)")
    ax.set_ylabel("BioEmu-1 ensemble Rg (Å)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Points below the line: the ensemble is too compact", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_rg.png")
    plt.close(fig)


def fig_resolvability(d: pd.DataFrame, paths: list[dict]) -> None:
    """Reweighting paths (chi2 against effective sample fraction) and the class breakdown."""
    m = d[(d["model"] == C.MODEL) & d["clean"]].set_index("label")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    P = pd.DataFrame(paths)
    for lab, g in P.groupby("label"):
        if lab not in m.index:
            continue
        g = g.sort_values("phi", ascending=False)
        ax.plot(g["phi"], g["chi2"], color=CLS_COL[m.loc[lab, "disorder_class"]], alpha=0.25, lw=0.9)
    ax.axhline(2, color=RULE, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.05, 0.005)
    ax.set_xlabel("effective sample fraction φ kept  (← more reweighting)")
    ax.set_ylabel("reduced χ²")
    ax.set_title("How far each ensemble must be reweighted to fit", fontsize=10, loc="left")
    ax = axes[1]
    order = ["fits", "calibration", "population", "unresolved"]
    tab = (m.groupby(["disorder_class", "resolvability"]).size().unstack(fill_value=0)
           .reindex(index=CLASS_ORDER, columns=order, fill_value=0))
    share = tab.div(tab.sum(axis=1), axis=0)
    left = np.zeros(3)
    greys = ["#B9C0C6", "#8A939B", "#5B646E", "#16191D"]
    for col, c in zip(order, greys, strict=True):
        ax.barh(range(3), share[col], left=left, color=c, label=col, height=0.6)
        left += share[col].to_numpy()
    ax.set_yticks(range(3))
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of entries")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "fig_resolvability.png")
    plt.close(fig)


def fig_models(A: dict, d: pd.DataFrame) -> None:
    """Raw chi2 by class for every model in the archive."""
    models = [C.MODEL] + [r["model"] for r in A["model_comparison"]]
    bc = pd.DataFrame(A["by_class"])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.8 / len(models)
    for j, mo in enumerate(models):
        g = bc[bc["model"] == mo].set_index("disorder_class").reindex(CLASS_ORDER)
        x = np.arange(3) + (j - (len(models) - 1) / 2) * w
        ax.bar(x, g["chi2_raw_median"], width=w * 0.92, color=plt.cm.tab10(j), label=MODEL_NAME.get(mo, mo))
        ax.errorbar(x, g["chi2_raw_median"], yerr=[g["chi2_raw_median"] - g["chi2_raw_q25"],
                                                   g["chi2_raw_q75"] - g["chi2_raw_median"]],
                    fmt="none", ecolor=INK, lw=0.8, capsize=2)
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylabel("median raw reduced χ² (bars: quartiles)")
    ax.axhline(2, color=RULE, lw=1, ls="--")
    ax.legend(frameon=False, fontsize=9, ncol=3)
    ax.set_title("Unweighted ensembles of each model against the same profiles", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_models.png")
    plt.close(fig)


def fig_candidates(cand: pd.DataFrame) -> None:
    top = cand.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    cols = top["disorder_class"].map(CLS_COL)
    ax.barh(range(len(top)), top["priority"], color=cols, height=0.65)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{r.label}  ·  {int(r.length)} aa  ·  {r.rg_direction}" for r in top.itertuples()],
                       fontsize=8.5)
    ax.set_xlabel("measurement priority (log10 raw χ² × resolvability × tractability)")
    ax.set_title("Top candidates for new measurements", fontsize=10, loc="left")
    for cls, c in CLS_COL.items():
        ax.scatter([], [], color=c, label=cls)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_candidates.png")
    plt.close(fig)


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def write_campaign(A: dict, cand: pd.DataFrame) -> None:
    """docs/CAMPAIGN.md: the proposal, with every number read from the analysis."""
    b = A["bioemu"]
    E = A["expectations"]
    res = b["resolvability_counts"]
    n_clean = A["n_clean"]
    cls = {r["disorder_class"]: r for r in A["by_class"] if r["model"] == C.MODEL}
    top = cand.head(12)
    rows = "\n".join(
        f"| {r.label} | {int(r.length)} | {r.disorder_class} | {r.chi2_raw:.1f} | {r.chi2_best:.2f} | "
        f"{r.resolvability} | {r.rg_direction} |" for r in top.itertuples())
    text = f"""# Experimental campaign proposal: SAXS data for the next BioEmu

*Generated by `bsc report` from `results/analysis.json` and `results/candidates.csv`. The reasoning
is fixed in `docs/DESIGN.md` §6; the numbers below are read from the results.*

## 1. What the error map says

Scored without reweighting against {n_clean} quality-filtered SASBDB profiles, the BioEmu-1 ensemble
reaches reduced χ² ≤ 2 for {pct(b['share_raw_fit_chi2_2'])} of entries (median χ² {b['chi2_raw_median']:.1f};
quartiles {b['chi2_raw_q25']:.1f}–{b['chi2_raw_q75']:.1f}). By class, the median raw χ² is
{cls['folded']['chi2_raw_median']:.1f} for folded proteins (n = {cls['folded']['n']}),
{cls['partly disordered']['chi2_raw_median']:.1f} for partly disordered (n = {cls['partly disordered']['n']}) and
{cls['disordered']['chi2_raw_median']:.1f} for disordered (n = {cls['disordered']['n']}).
Expectation E1 (disordered worse than folded) is {'met' if E['E1']['met'] else 'not met'}
(one-sided p = {E['E1']['p_one_sided']:.2g}); E2 (error rising with length among folded proteins) is
{'met' if E['E2']['met'] else 'not met'} (ρ = {E['E2']['spearman_rho']:+.2f}); E4 (over-compaction of
disordered proteins) is {'met' if E['E4']['met'] else 'not met'} (median Rg ratio
{E['E4']['median_rg_ratio_disordered']:.2f}, {pct(E['E4']['share_below_1'])} below 1).

## 2. What SAXS can and cannot resolve

Reweighting each ensemble towards its profile sorts the entries into four kinds
({res.get('fits', 0)} fit as they are; {res.get('calibration', 0)} need a modest shift, keeping at least half the
effective sample; {res.get('population', 0)} need their populations moved substantially; {res.get('unresolved', 0)}
cannot be brought to χ² ≤ 2 from the conformers the model proposes). Expectation E3 is
{'met' if E['E3']['met'] else 'not met'}: {pct(E['E3']['share_phi_above_0.3'])} of entries reach χ² ≈ 1 while
keeping more than 30% of the effective sample.

- **Population cases** are where new SAXS data would move the model most: the right conformers exist in
  the ensemble, and the measurement says how to weight them. A fine-tuning signal built from such
  profiles is informative.
- **Unresolved cases** need a different measurement or a different model. SAXS says the ensemble is wrong
  but not how; the follow-up is a measurement that reports local structure or dynamics (NMR relaxation
  or chemical shifts, HDX-MS, single-molecule FRET on labelled constructs).
- **Calibration cases** are cheap wins: a small, consistent correction, often of the Rg scale. For these,
  repeat measurements at several concentrations matter more than new systems.

## 3. Candidate systems

The ranking multiplies the model's error (log₁₀ raw χ²) by a resolvability weight (population 1.0,
calibration 0.6, unresolved 0.3) and a tractability weight (chain ≤ 350 residues; folded or partly
disordered). The top of the list:

| SASBDB | length | class | raw χ² | best χ² | kind | Rg |
|---|---|---|---|---|---|---|
{rows}

Each candidate carries a construct note in `results/candidates.csv` (length, disorder class, Rg direction).
Before commissioning, each is checked against its SASBDB entry for buffer, concentration series and
oligomeric state, and against UniProt for tags and disordered termini.

## 4. Assay design

- **Primary assay.** Size-exclusion-coupled SAXS (SEC-SAXS) at a synchrotron beamline, so that
  aggregates and oligomers are separated from the monomer before scattering is recorded. Three
  concentrations across the elution peak are integrated and compared; Guinier Rg agreement across them is
  the first QC gate.
- **Constructs.** Expressed with a cleavable tag; the tag is removed before measurement, because a 2–3 kDa
  tag changes Rg of a small protein measurably. Disordered termini longer than ~15 residues are either
  kept (if present in the deposited entry) or trimmed in a second construct, so the model's handling of
  termini can be tested directly.
- **Buffer.** Matched buffer from the final SEC step, with the exact dialysate recorded; BioEmu does not
  model pH or ionic strength, so buffer is held at one standard condition (20 mM HEPES pH 7.4, 150 mM NaCl,
  2 mM TCEP) and recorded with each dataset.
- **Standards and controls.** A protein standard (bovine serum albumin or glucose isomerase) in every
  session for absolute scale; water for intensity calibration; a buffer-only frame bracketing each
  sample; a repeat of one previously measured SASBDB entry per batch as a cross-site control.
- **Companion measurement.** For the unresolved kind, HDX-MS on the same batch of protein: it reports
  per-segment exchange that distinguishes a locally unfolded region from a globally wrong fold.

## 5. QC criteria, applied by script

Every dataset passes or fails on recorded values, not on inspection:

1. Guinier region: at least 12 points with q·Rg ≤ 1.3 and a linear fit whose residuals show no trend.
2. No aggregation: lowest-angle intensity within 5% of the Guinier line; Rg from the three concentrations
   agree within 3%.
3. Radiation damage: frame-to-frame χ² across the exposure series below 1.5; otherwise frames are
   discarded from the end until it is.
4. Molecular weight from I(0) against the standard within 20% of the sequence mass of the monomer.
5. Buffer subtraction sanity: no negative intensities within the usable range; the high-q plateau is
   flat.
6. Reproducibility: the cross-site control reproduces its deposited profile with χ² < 2 after scaling.

Datasets that fail are repeated once; a second failure retires the construct and triggers the
contingency below.

## 6. Success criteria and contingencies

- **Dataset-level success:** QC passed, and the profile constrains the BioEmu ensemble (reweighting to
  χ² ≤ 1 changes the effective sample fraction by at least 0.2). A profile that the raw ensemble already
  fits is still deposited, but counted as a confirmation, not as training signal.
- **Campaign-level success:** at least 60% of commissioned systems yield a usable profile; the set
  covers the three disorder classes in roughly the proportions of the error map; the population kind
  is over-represented, because that is where the model can learn.
- **Contingencies.** Poor expression → switch to a homologue from the same family with a deposited
  SASBDB entry. Aggregation at SEC → lower concentration and add 5% glycerol; if that fails, retire.
  Beamtime loss → a laboratory SAXS instrument for the smallest, most concentrated samples, accepting
  the lower q range and recording it. Ambiguous SAXS (the unresolved kind) → HDX-MS first, SAXS second.

## 7. Work packages for an external provider

| Package | Deliverable | Review point |
|---|---|---|
| WP1 Constructs | Expression plasmids for the ranked list, sequence-verified, with and without tags | Design review before synthesis; sequence files checked by script |
| WP2 Protein production | ≥ 2 mg per construct at ≥ 95% purity by SEC and SDS-PAGE, monodisperse by DLS | Purity and DLS reports per batch; a batch failing DLS does not proceed |
| WP3 SEC-SAXS | Three-concentration SEC-SAXS per sample, with standards and buffer frames; raw frames and reduced curves | Scripted QC on delivery; results joined to the candidate table |
| WP4 HDX-MS (unresolved kind only) | Deuterium uptake per peptide at four time points, with a fully deuterated control | Peptide coverage ≥ 85%; back-exchange reported |
| WP5 Deposition | SASBDB deposition of every QC-passed dataset with full metadata | Accession codes recorded against each candidate |

Milestones are set per batch of twelve constructs: constructs at week 2, protein at week 6, SAXS at
week 8, QC and model re-scoring at week 9. The re-scoring repeats this pipeline on the new profiles, so
every batch updates the error map before the next batch is chosen.

## 8. What this proposal rests on, and what it does not claim

The error map uses ensembles generated by the PeptoneBench authors with BioEmu-1 at their settings; a
re-sampling with the current checkpoint is the first check once GPU time is available. SASBDB profiles
differ in buffer, temperature and construct, none of which the model sees, so part of the raw error is
condition mismatch, which a campaign under one standard condition removes. The scoring treats the
Pepsi-SAXS forward model as exact; its hydration-shell parameters are a known source of Rg bias of a
few per cent, which is why calibration cases are read with care.
"""
    (C.REPO / "docs" / "CAMPAIGN.md").write_text(text)


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    A = json.load(open(C.RESULTS / "analysis.json"))
    d = pd.read_csv(C.RESULTS / "resolvability.csv")
    cand = pd.read_csv(C.RESULTS / "candidates.csv")
    paths = json.load(open(C.RESULTS / "paths" / f"{C.MODEL}.json"))
    fig_error_map(d, A)
    fig_rg(d)
    fig_resolvability(d, paths)
    if A["model_comparison"]:
        fig_models(A, d)
    fig_candidates(cand)
    write_campaign(A, cand)
    print(f"  figures -> {OUT}; proposal -> docs/CAMPAIGN.md", flush=True)


if __name__ == "__main__":
    main()
