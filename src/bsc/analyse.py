"""Stage `analyse`: the error map by protein class, the four pre-specified expectations, the
resolvability classification of every entry, model comparisons, and the ranked list of candidate
systems for new measurements.

Output: results/analysis.json (every number the page reports), results/by_class.csv,
results/resolvability.csv, results/candidates.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from bsc import config as C

CLASS_ORDER = [c[0] for c in C.DISORDER_CLASSES]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    ent = pd.read_csv(C.RESULTS / "entries.csv")
    sc = pd.read_csv(C.RESULTS / "scores.csv")
    df = sc.merge(ent.drop(columns=["sequence"]), on="label", how="left")
    df["log_chi2_raw"] = np.log10(df["chi2_raw"])
    df["rg_ratio"] = df["rg_model"] / df["rg_exp"]
    flagged = df["flag_aggregation"] | df["flag_negative_I"] | df["flag_bad_errors"] | df["flag_oligomer"]
    df["clean"] = ~flagged & df["guinier_valid"]
    return ent, df


def resolvability(row: pd.Series) -> str:
    """One label per entry from the raw fit and the reweighting path (thresholds fixed in DESIGN §4).

    fits          raw chi2 <= 2: the prior ensemble already describes the data
    calibration   raw chi2 > 2 but chi2 <= 2 reached while keeping phi >= 0.5: a modest shift
    population    chi2 <= 2 needs phi < 0.5 but is reached: the data move the populations substantially
    unresolved    chi2 <= 2 never reached: outside the ensemble's reach, or a data problem
    """
    if row["chi2_raw"] <= 2:
        return "fits"
    if np.isfinite(row["phi_at_chi2_2"]) and row["phi_at_chi2_2"] >= 0.5:
        return "calibration"
    if np.isfinite(row["phi_at_chi2_2"]):
        return "population"
    return "unresolved"


def expectations(d: pd.DataFrame) -> dict:
    """E1–E4 of DESIGN §5 on the model under study, clean entries only."""
    m = d[(d["model"] == C.MODEL) & d["clean"]]
    out = {}
    f, dis = m[m["disorder_class"] == "folded"], m[m["disorder_class"] == "disordered"]
    u = stats.mannwhitneyu(dis["log_chi2_raw"], f["log_chi2_raw"], alternative="greater")
    out["E1"] = {"statement": "raw chi2 worse for disordered than folded",
                 "median_folded": float(f["chi2_raw"].median()),
                 "median_disordered": float(dis["chi2_raw"].median()),
                 "n_folded": len(f), "n_disordered": len(dis), "p_one_sided": float(u.pvalue),
                 "met": bool(dis["chi2_raw"].median() > f["chi2_raw"].median() and u.pvalue < 0.05)}
    rho = stats.spearmanr(f["length"], f["log_chi2_raw"])
    small, large = f[f["length"] <= 100], f[f["length"] > 100]
    out["E2"] = {"statement": "among folded proteins raw chi2 rises with length",
                 "spearman_rho": float(rho.statistic), "p": float(rho.pvalue), "n": len(f),
                 "share_fit_le_100": float((small["chi2_raw"] <= 2).mean()), "n_le_100": len(small),
                 "share_fit_gt_100": float((large["chi2_raw"] <= 2).mean()), "n_gt_100": len(large),
                 "median_chi2_le_100": float(small["chi2_raw"].median()),
                 "median_chi2_gt_100": float(large["chi2_raw"].median()),
                 "met": bool(rho.statistic > 0 and rho.pvalue < 0.05)}
    reached = m["phi_at_chi2_1"].notna()
    out["E3"] = {"statement": "most entries reach chi2 ~1 with phi above 0.3; a minority need phi below 0.1",
                 "share_reaching_chi2_1": float(reached.mean()),
                 "share_phi_above_0.3": float((m["phi_at_chi2_1"] > 0.3).mean()),
                 "share_phi_below_0.1": float((m["phi_at_chi2_1"] < 0.1).mean()),
                 "met": bool((m["phi_at_chi2_1"] > 0.3).mean() > 0.5)}
    ratio = dis["rg_ratio"].dropna()
    w = stats.wilcoxon(np.log(ratio) if len(ratio) > 5 else [0, 0, 0, 0, 0, 0], alternative="less")
    out["E4"] = {"statement": "ensemble Rg below experimental Rg for disordered proteins",
                 "median_rg_ratio_disordered": float(ratio.median()), "n": int(len(ratio)),
                 "share_below_1": float((ratio < 1).mean()), "p_one_sided": float(w.pvalue),
                 "met": bool(ratio.median() < 1 and w.pvalue < 0.05)}
    return out


def by_class(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, cls), g in d[d["clean"]].groupby(["model", "disorder_class"]):
        rows.append({"model": model, "disorder_class": cls, "n": len(g),
                     "chi2_raw_median": g["chi2_raw"].median(),
                     "chi2_raw_q25": g["chi2_raw"].quantile(0.25),
                     "chi2_raw_q75": g["chi2_raw"].quantile(0.75),
                     "share_raw_fit": (g["chi2_raw"] <= 2).mean(),
                     "chi2_best_median": g["chi2_best"].median(),
                     "phi_at_chi2_1_median": g["phi_at_chi2_1"].median(),
                     "share_unresolved": (g["resolvability"] == "unresolved").mean(),
                     "rg_ratio_median": g["rg_ratio"].median()})
    out = pd.DataFrame(rows)
    out["disorder_class"] = pd.Categorical(out["disorder_class"], CLASS_ORDER, ordered=True)
    return out.sort_values(["model", "disorder_class"])


def model_comparison(d: pd.DataFrame) -> list[dict]:
    """Paired comparison of each comparator with the model under study on the entries both have."""
    base = d[(d["model"] == C.MODEL) & d["clean"]].set_index("label")
    out = []
    for m in sorted(d["model"].unique()):
        if m == C.MODEL:
            continue
        o = d[(d["model"] == m) & d["clean"]].set_index("label")
        common = base.index.intersection(o.index)
        if len(common) < 10:
            continue
        diff = np.log10(o.loc[common, "chi2_raw"]) - np.log10(base.loc[common, "chi2_raw"])
        w = stats.wilcoxon(diff)
        out.append({"model": m, "n": int(len(common)),
                    "median_log10_chi2_ratio_vs_bioemu": float(diff.median()),
                    "share_better_than_bioemu": float((diff < 0).mean()), "p_two_sided": float(w.pvalue),
                    "median_chi2_raw": float(o.loc[common, "chi2_raw"].median()),
                    "bioemu_median_chi2_raw_same_entries": float(base.loc[common, "chi2_raw"].median())})
    return out


def candidates(d: pd.DataFrame) -> pd.DataFrame:
    """Rank entries for new measurement: the model is wrong (large raw chi2 after clean data), the
    data could tell (reweighting reaches a fit), and the system is tractable (length, folded or
    partly disordered). Score = log10 raw chi2 × resolvability weight × tractability weight."""
    m = d[(d["model"] == C.MODEL) & d["clean"]].copy()
    res_w = m["resolvability"].map({"population": 1.0, "calibration": 0.6, "unresolved": 0.3, "fits": 0.0})
    tract = np.where(m["length"] <= 350, 1.0, 0.5) * m["disorder_class"].map(
        {"folded": 1.0, "partly disordered": 1.0, "disordered": 0.6}).to_numpy()
    m["priority"] = m["log_chi2_raw"].clip(lower=0) * res_w * tract
    m["rg_direction"] = np.where(m["rg_ratio"] < 0.95, "model too compact",
                                 np.where(m["rg_ratio"] > 1.05, "model too extended", "Rg agrees"))
    cols = ["label", "length", "disorder_class", "chi2_raw", "chi2_best", "phi_at_chi2_2",
            "resolvability", "rg_exp", "rg_model", "rg_direction", "mw_ratio", "sasbdb_title", "priority"]
    return m.sort_values("priority", ascending=False)[cols].reset_index(drop=True)


def main() -> None:
    ent, d = load()
    d["resolvability"] = d.apply(resolvability, axis=1)
    bc = by_class(d)
    bc.to_csv(C.RESULTS / "by_class.csv", index=False)
    keep = ["model", "label", "disorder_class", "length", "chi2_raw", "chi2_best", "phi_at_chi2_1",
            "phi_at_chi2_2", "resolvability", "rg_exp", "rg_model", "rg_ratio", "clean"]
    d[keep].to_csv(C.RESULTS / "resolvability.csv", index=False)
    cand = candidates(d)
    cand.to_csv(C.RESULTS / "candidates.csv", index=False)
    m = d[(d["model"] == C.MODEL)]
    mc = m[m["clean"]]
    summary = {
        "n_entries": int(len(ent)), "n_clean": int(ent.shape[0] - (~m["clean"]).sum()),
        "n_flag_aggregation": int(ent["flag_aggregation"].sum()),
        "n_flag_negative": int(ent["flag_negative_I"].sum()),
        "n_flag_bad_errors": int(ent["flag_bad_errors"].sum()),
        "n_flag_oligomer": int(ent["flag_oligomer"].sum()),
        "oligomer_median_chi2_raw": float(m.loc[m["flag_oligomer"], "chi2_raw"].median()),
        "n_guinier_invalid": int((~ent["guinier_valid"]).sum()),
        "class_counts": ent["disorder_class"].value_counts().to_dict(),
        "bioemu": {
            "n_scored": int(len(m)), "n_conformers_median": float(m["n_conformers"].median()),
            "chi2_raw_median": float(mc["chi2_raw"].median()),
            "chi2_raw_q25": float(mc["chi2_raw"].quantile(0.25)),
            "chi2_raw_q75": float(mc["chi2_raw"].quantile(0.75)),
            "share_raw_fit_chi2_2": float((mc["chi2_raw"] <= 2).mean()),
            "share_raw_fit_chi2_1": float((mc["chi2_raw"] <= 1).mean()),
            "chi2_best_median": float(mc["chi2_best"].median()),
            "phi_at_chi2_1_median": float(mc["phi_at_chi2_1"].median()),
            "resolvability_counts": mc["resolvability"].value_counts().to_dict(),
            "rg_ratio_median_by_class": mc.groupby("disorder_class")["rg_ratio"].median().to_dict(),
        },
        "expectations": expectations(d),
        "rg_ratio_disordered_by_model": {
            m: float(g["rg_ratio"].median())
            for m, g in d[d["clean"] & (d["disorder_class"] == "disordered")].groupby("model")},
        "model_comparison": model_comparison(d),
        "by_class": json.loads(bc.to_json(orient="records")),
        "top_candidates": json.loads(cand.head(25).to_json(orient="records")),
        "thresholds": {"raw_fit_chi2": 2.0, "resolvability_phi": 0.5, "chi2_target": C.CHI2_TARGET,
                       "aggregation_upturn": C.AGGREGATION_UPTURN,
                       "theta_grid": [float(x) for x in C.THETA_GRID]},
    }
    json.dump(summary, open(C.RESULTS / "analysis.json", "w"), indent=1, default=float)
    b = summary["bioemu"]
    print(f"  {summary['n_clean']} clean entries; BioEmu raw chi2 median {b['chi2_raw_median']:.2f}; "
          f"raw fit (chi2<=2) in {b['share_raw_fit_chi2_2']:.0%}; "
          f"expectations met: {[k for k, v in summary['expectations'].items() if v['met']]}", flush=True)


if __name__ == "__main__":
    main()
