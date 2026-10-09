"""Stage `envelopes` (DESIGN §7, A3): an ab initio electron-density envelope of each worked example
from its experimental curve, with DENSS (Grant 2018), for the worked-examples figure.

DENSS needs numpy < 2 and lives in its own environment (BSC_DENSS_ENV, default /scratch/.venv-denss);
when that environment is absent the stage prints a notice and returns, so `bsc report` still runs
elsewhere with the tracked figure. Per example: `denss-all` with N_MAPS reconstructions in FAST mode
and Dmax from the SASBDB record (pddf_dmax, nm, converted to Å), aligned and averaged.

Output: results/denss/<label>/ (maps and logs, not tracked) and results/denss_stats.json (fit
statistics, tracked).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import requests

from bsc import analyse, score
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


def main(cores: int | None = None) -> None:
    if not (DENSS_ENV / "bin" / "denss-all").exists():
        print(f"  DENSS environment {DENSS_ENV} not found; envelopes skipped (set BSC_DENSS_ENV)", flush=True)
        return
    _, d = analyse.load()
    examples = analyse.choose_examples(d)
    stats_path = C.RESULTS / "denss_stats.json"
    stats = json.load(open(stats_path)) if stats_path.exists() else {}
    cores = cores or min(C.N_JOBS, 5)
    for cls, label in examples.items():
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


if __name__ == "__main__":
    main()
