"""Shared paths and helpers for the SAXS coverage analysis."""
import json, os, time, pathlib, datetime
import requests

# Locations. COVERAGE_WORK holds the downloads and the intermediate tables (about 0.9 GB, not tracked);
# COVERAGE_DOWNLOADS can point the download cache elsewhere. The SAXS table is read from the main
# pipeline's data directory (BSC_ROOT, as for `bsc fetch`); the clean-entry flags from the tracked results.
REPO = pathlib.Path(__file__).resolve().parents[2]
ROOT = pathlib.Path(os.environ.get("COVERAGE_WORK", REPO / "analysis" / "coverage" / "work"))
DL = pathlib.Path(os.environ.get("COVERAGE_DOWNLOADS", ROOT / "downloads"))
TAB = ROOT / "tables"
FIG = ROOT / "figures"
BSC_ROOT = pathlib.Path(os.environ.get("BSC_ROOT", REPO))
SAXS_TABLE = BSC_ROOT / "data" / "PeptoneDB-SAXS" / "PeptoneDB-SAXS.csv"
RESOLV = REPO / "results" / "resolvability.csv"
ENTRIES = REPO / "results" / "entries.csv"
EXPORT = REPO / "results" / "coverage"   # the small outputs the page and the `coverage` stage read
VERSIONS = DL / "versions.json"
for _d in [TAB, FIG] + [DL / x for x in ("cath", "sasbdb", "uniprot", "rcsb", "interpro", "pdbe", "ted")]:
    _d.mkdir(parents=True, exist_ok=True)

S = requests.Session()
S.headers["User-Agent"] = "saxs-coverage-analysis/1.0 (python-requests)"


def get(url, params=None, tries=5, timeout=60, ok404=False, **kw):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=timeout, **kw)
            if r.status_code == 404 and ok404:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 * (i + 1)); continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed {url}")


def cached_json(path, url, params=None, ok404=False, **kw):
    path = pathlib.Path(path)
    if path.exists():
        return json.loads(path.read_text())
    r = get(url, params=params, ok404=ok404, **kw)
    if r.status_code == 404 or r.status_code == 204 or not r.content:
        data = {"__status__": r.status_code}
    else:
        data = r.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def record_version(key, info):
    """Record the release of a source the first time it is fetched. An existing record is kept: the
    versions describe the cached downloads, and deleting the download directory removes both."""
    v = json.loads(VERSIONS.read_text()) if VERSIONS.exists() else {}
    if key in v:
        return
    info = dict(info)
    info.setdefault("accessed", datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    v[key] = info
    VERSIONS.write_text(json.dumps(v, indent=2))


ORG_CATS = ["Human", "Mouse", "E. coli", "Yeast", "Other bacteria", "Viruses", "Plants", "Other"]


def org_category(taxid, sci_name, lineage_names):
    """Map an organism to the eight comparison groups. lineage_names: iterable of names."""
    name = sci_name.strip() if isinstance(sci_name, str) else ""
    lin = {x.strip() for x in lineage_names if isinstance(x, str) and x}
    try:
        taxid = int(taxid)
    except (TypeError, ValueError):
        taxid = None
    if taxid == 9606 or name.startswith("Homo sapiens"):
        return "Human"
    if taxid == 10090 or name.startswith("Mus musculus"):
        return "Mouse"
    if name.startswith("Escherichia coli") or "Escherichia coli" in lin:
        return "E. coli"
    if (name.startswith("Saccharomyces cerevisiae") or name.startswith("Schizosaccharomyces pombe")
            or "Saccharomyces cerevisiae" in lin or "Schizosaccharomyces pombe" in lin):
        return "Yeast"
    if "Bacteria" in lin:
        return "Other bacteria"
    if "Viruses" in lin:
        return "Viruses"
    if "Viridiplantae" in lin:
        return "Plants"
    return "Other"
