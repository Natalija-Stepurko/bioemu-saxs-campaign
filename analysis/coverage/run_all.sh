#!/usr/bin/env bash
# Rerun the whole coverage analysis and export its small outputs to results/coverage/.
#   COVERAGE_WORK       downloads and intermediate tables (default analysis/coverage/work, not tracked)
#   COVERAGE_DOWNLOADS  the download cache, if it lives elsewhere (default $COVERAGE_WORK/downloads)
#   BSC_ROOT            where `bsc fetch` put data/PeptoneDB-SAXS (default: the repository)
#   COVERAGE_PY         a Python with the extra requirements (requirements.txt)
# Downloads are cached; delete the download directory to refetch everything (versions.json goes with it).
set -euo pipefail
cd "$(dirname "$0")"
export COVERAGE_WORK=${COVERAGE_WORK:-$PWD/work}
PY=${COVERAGE_PY:-python3}
mkdir -p "$COVERAGE_WORK/logs"
./00_download_cath.sh
$PY 01_fetch_sasbdb.py
$PY 02_map_uniprot.py
./03_download_reference_sets.sh
$PY 04_fetch_annotations.py
$PY 05_fetch_rcsb_nr.py
$PY 06_parse_swissprot.py
$PY 07_build_protein_table.py
$PY 08_compare.py > "$COVERAGE_WORK/logs/08_compare.log"
$PY 09_figures.py
$PY 10_export.py
