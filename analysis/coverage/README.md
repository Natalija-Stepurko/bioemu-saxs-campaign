# Coverage of known structure space by the SAXS set

This analysis places the 399 clean PeptoneDB-SAXS entries with a BioEmu-1 ensemble in experimentally known
protein structure space. It maps every construct to a UniProtKB region, annotates the region with CATH
superfamilies, organism, disorder and PDB coverage, and compares the set with three references: CATH v4.4.0
at the S35 sequence-cluster level (fold space), the non-redundant PDB (RCSB 30%-identity polymer-entity
clusters, experimental entries only) and UniProtKB/Swiss-Prot (organism, length, disorder). Its small outputs
are tracked in `results/coverage/`; the `bsc coverage` stage and the project page read only those.

## Steps

| Script | Does |
|---|---|
| `00_download_cath.sh` | CATH latest-release classification files (v4.4.0) and the v4.3.0 S35 list for a sensitivity check |
| `01_fetch_sasbdb.py` | SASBDB entry summaries for the 439 labels |
| `02_map_uniprot.py` | construct → UniProtKB accession: the SASBDB `uniprot_code` where it aligns (identity ≥ 0.5, construct coverage ≥ 0.3), else an RCSB sequence search (≥ 90% identity) with the SIFTS UniProt reference of the hit, else a UniProt peptide search on three internal 20-mers; then a local alignment to the canonical sequence defines the construct region |
| `03_download_reference_sets.sh` | the Swiss-Prot flat file and the RCSB 30%-identity cluster file |
| `04_fetch_annotations.py` | per accession: InterPro CATH-Gene3D and Pfam locations, PDBe SIFTS best structures, TED domains |
| `05_fetch_rcsb_nr.py` | length and source organism of one experimental representative per RCSB 30% cluster |
| `06_parse_swissprot.py` | one row per Swiss-Prot entry: length, organism group, UniProt "Disordered" regions, Gene3D matches, PDB cross-references |
| `07_build_protein_table.py` | the per-entry table: domains whose residues lie at least half inside the construct, SIFTS coverage of the construct by atomic-model chains (`has_pdb_50`: at least 50%), disorder content, organism group |
| `08_compare.py` | all comparisons into `summary.json`: superfamily coverage by rank, class and architecture composition (cluster bootstrap over accessions), over- and under-represented superfamilies (Poisson tails, uncorrected, for ranking), organism, length, domains per protein, disorder, structure availability, redundancy |
| `09_figures.py` | diagnostic figures in the work directory (the page figures are drawn by `bsc coverage`) |
| `10_export.py` | copies the tracked outputs into `results/coverage/` |

Units: the domain-level comparisons count unique (UniProt accession, CATH superfamily) pairs inside the
constructs, the protein-family analogue of a CATH S35 cluster. Single proportions carry Wilson 95% intervals.

## Mapping notes

390 of the 399 clean entries map to a UniProtKB entry (353 by the SASBDB accession, 23 by RCSB sequence search,
14 by UniProt peptide search). The SASBDB accession was set aside for 17 entries: 16 pointed to UniProtKB
entries deleted since deposition and one aligned to 6% of the construct; 16 of the 17 were mapped again by
sequence search. Nine entries stay unmapped: four antibody-derived fragments (a Fab, an scFv and two
single-domain antibodies), a synthetic construct, a filamin two-domain fragment, an antimicrobial peptide, a
bacterial harpin and a lanthionine synthetase C-like protein whose UniProtKB entry was deleted. They are left
out of every comparison that needs a mapping. SIFTS residue ranges follow the deposited sequence, so PDB coverage counts residues listed in a
structure record, which can include residues not modelled in the coordinates.

## Database versions

Recorded in `results/coverage/versions.json` when each source was first fetched (October 2026):

| Source | Version |
|---|---|
| CATH | v4.4.0 (601,328 domains, 37,350 S35 clusters) |
| CATH-Gene3D in InterPro | 4.3.0 (InterPro 110.0) |
| Pfam | 38.2 |
| UniProtKB / Swiss-Prot | release 2026_03 (575,748 Swiss-Prot entries) |
| RCSB clusters-by-entity-30 | file of 4 October 2026 (47,069 clusters with an experimental entity) |
| PDBe SIFTS best_structures | live API, weekly update |
| TED | v1 (AFDB v4 models with CATH labels) |
| SASBDB | REST API entry summaries |

The live APIs (SIFTS, InterPro, TED, RCSB search) move between releases, so a rerun after deleting the download
cache can differ slightly from the tracked outputs; with the cache in place the rerun reproduces them byte for
byte.

## Rerun

```bash
uv venv /path/to/venv-cov && uv pip install --python /path/to/venv-cov -r analysis/coverage/requirements.txt
COVERAGE_PY=/path/to/venv-cov/bin/python BSC_ROOT=<where bsc fetch put data/> \
  COVERAGE_WORK=/path/with/1GB/free analysis/coverage/run_all.sh
uv run bsc coverage      # structure split, rank coverage and the page figures from results/coverage/
```

Needs network access, about 0.9 GB of disk for the downloads (the Swiss-Prot flat file is 0.7 GB) and the
extra requirements in `requirements.txt`, which the main environment does not carry. The SAXS table comes from
`bsc fetch` (`BSC_ROOT/data/PeptoneDB-SAXS/`) and the clean-entry flags from `results/resolvability.csv`.
Downloads are cached in `$COVERAGE_WORK/downloads` (`COVERAGE_DOWNLOADS` moves them); delete that directory
to refetch everything, which also starts a new `versions.json`.
