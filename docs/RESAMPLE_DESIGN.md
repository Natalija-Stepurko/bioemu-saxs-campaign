# Design: BioEmu v1.2 re-sample against the archive's ensembles and the measured profiles

Written 2026-10-16, after the samples were downloaded and before any radius of gyration, curve or χ² was
computed from them. Deviations are added, dated, in the last section; nothing above it is edited after the
first result is computed.

## 1. Question

The archive's BioEmu ensembles (PeptoneBench, Zenodo 17306061) were generated with BioEmu at code commit
ac7455d; the checkpoint was not recorded. Their disordered ensembles are larger than the measured proteins,
in the computed curves and in the Cα coordinates of the conformers (`results/coord_rg.csv`). Two questions:

1. Does the current checkpoint, bioemu-v1.2, still produce disordered ensembles that are too large?
2. Does bioemu-v1.2 fit the measured SAXS profiles better than the archive's ensembles?

## 2. Samples and proteins

The proteins are those listed in `notebooks/resample_list.csv`: the clean disordered entries of the study,
the worked examples and folded controls. The notebook `notebooks/bioemu_resample.ipynb` was run on a Colab T4
GPU with `model_name="bioemu-v1.2"`, `num_samples=100` and `filter_samples=True`; the BioEmu package was
installed unpinned (`pip install bioemu[cuda]`), so its version was not recorded. The output folder holds,
per protein, the raw sample batches (`batch_*.npz`: Cα positions and frame orientations), the backbone
topology, the sequence, and either `samples.xtc` or, for some proteins, only `samples_unphysical.xtc`.

**What the files contain, established in code before use.** For each protein: the number of samples in the
npz batches, the frames in each xtc file, the model name and timing from `progress.json`. The meaning of
`samples_unphysical.xtc` is read from BioEmu's source (`bioemu.convert_chemgraph.save_pdb_and_xtc`).

**Recovery of every sample.** BioEmu's own conversion (`save_pdb_and_xtc` with `filter_samples=False`, from
the BioEmu release on PyPI at the time the notebook ran, installed CPU-only) turns all samples of the npz
batches into backbone conformers (N, Cα, C, O, Cβ). The conversion is checked against the Colab output:
the filter recomputed on the recovered conformers must retain exactly as many conformers as `samples.xtc`
holds, and each retained conformer must match its `samples.xtc` frame (RMSD after superposition within the
xtc storage precision); where only `samples_unphysical.xtc` exists, its frames must match the recovered
ones one to one. If any check fails, the analysis stops and the failure is reported.

**Archive ensembles.** The BioEmu conformers of the archive (`Predictions/PeptoneDB-SAXS-ensembles/bioemu`,
all atoms), extracted for the listed proteins from `Predictions.tar.gz` in one pass where missing, and the
archive's computed curves (`Predictions/PeptoneDB-SAXS-expt/bioemu/Pepsi-<label>.csv`).

**Measured data.** The PeptoneDB-SAXS profiles and the Guinier Rg of each profile already in
`results/entries.csv`; the disorder class from `results/resolvability.csv`.

## 3. Primary measure: size

For each protein and each side (v1.2, archive): the radius of gyration of the Cα atoms of every conformer
(equal weights), and the ensemble value Rg_ens = √(mean Rg²) over conformers, the convention of the
`coordrg` stage. Both sides use the Cα atoms, so the observable is identical although the archive carries
all atoms and v1.2 the backbone. The ratio is r = Rg_ens / Rg_exp, with the Guinier Rg of the measured
profile.

Reported for the disordered class and for the folded class (the single partly disordered protein is
reported on its own): the median r of v1.2 and of the archive on the same proteins, each with a percentile
bootstrap 95% interval (2,000 resamples over proteins, seed 20261016); the paired difference
Δ = log r_v1.2 − log r_archive, its median with a bootstrap interval, and a two-sided Wilcoxon signed-rank
test of Δ; the number of proteins whose v1.2 ensemble is smaller than the archive's. As a check of the
wiring, the archive values computed here are compared with `results/coord_rg.csv` where that file has the
protein.

## 4. Secondary measure: fit

Reduced χ² of the uniform ensemble-average curve against the measured profile, by three paths:

- **A, v1.2.** The Cα atoms of each conformer, back-mapped to all atoms with PULCHRA 3.06 (default options),
  then Pepsi-SAXS 3.0 with the measured profile and the flags of the simulation analysis and the
  PeptoneBench wrapper (`--maximum_scattering_vector 10`; on a non-zero exit, once more with
  `--nCoeff 50`); the fitted curve of each conformer (fourth output column) is kept.
- **B, archive through the same path.** The Cα atoms of each archive conformer, through the identical
  PULCHRA and Pepsi-SAXS steps. A and B differ only in the conformers.
- **C, archive as published.** The archive's own computed curves, as given.

Each path is scored with the repository's raw-fit scoring (`bsc.score.score_entry`): conformer curves with a
non-finite value are dropped and counted, points with σ ≤ 0 are left out, the uniform average is scaled to
the data with one factor and one constant background (`bsc.saxs.chi2`). NRMSD (`bsc.saxs.nrmsd_log`) is
reported beside χ², without an expectation. As a check of the wiring, path C must reproduce the raw χ² of
`results/scores.csv`.

Reported: the median χ² of each path over all proteins and by class; the paired log ratio
log χ²_A − log χ²_B (and A against C), its median with a bootstrap interval, and a two-sided Wilcoxon
signed-rank test; the difference of the medians (A − B) with a paired bootstrap interval.

## 5. The physical filter

BioEmu's filter (`bioemu.convert_chemgraph.filter_unphysical_traj`, default criteria: Cα–Cα distance of
sequence neighbours below 4.5 Å, C–N distance of sequence neighbours below 2.0 Å, no two atoms of residues
more than two apart closer than 1.0 Å) is evaluated per conformer with BioEmu's own code, on the atoms
BioEmu's output carries (N, Cα, C, O, Cβ). For the archive the same atoms are taken from the all-atom
conformers, so both sides are judged by the same criteria on the same atoms.

- **Primary analysis:** all samples on both sides, unfiltered, because the archive's scoring used all of its
  conformers.
- **Sensitivity:** §3 and §4 recomputed on the conformers that pass the filter, on both sides. A protein with
  fewer than 10 passing conformers on either side is left out of the paired filtered comparison and counted.
- **Rejected share:** per protein, the share of v1.2 samples the filter rejects (and the share of archive
  conformers it would reject), with the share failing each criterion; by class, the median and mean share and
  the number of proteins with every sample rejected. Those proteins are named with their length and class.

## 6. Expectations, fixed in advance

- **E12.** v1.2's median Cα-Rg ratio over the disordered proteins, all samples, remains above 1.1.
  Reasoning: the archive's excess size is in the conformers, and the coverage analysis found disordered
  chains nearly absent from BioEmu-1's public training data; I have no record that a later checkpoint added
  disordered training data.
- **E13.** v1.2's median raw χ² over all listed proteins, all samples, path A, is lower than the archive's
  on the same proteins through the same path (B). Reasoning: a later checkpoint released by the developers
  is expected to fit at least as well; the expectation is directional only.

Each is judged on the point estimate, reported with its interval and paired test.

## 7. Compute

Conversion runs CPU-only in a separate environment with the BioEmu package. PULCHRA and Pepsi-SAXS run per
conformer, cached per conformer and resumable, in at most three worker processes pinned to three cores
(Pepsi-SAXS uses every core it can see). Wall times of each step are recorded.

## 8. Limits

- One sampling run of 100 samples per protein, on one GPU type (Colab T4); seed-to-seed variation of the
  ensemble statistics is not measured.
- The side chains of both sides are rebuilt by PULCHRA from the Cα trace for paths A and B; the archive's
  own curves (path C) came from its own all-atom conformers, so C differs from A and B in the side-chain
  route as well as in the Pepsi-SAXS run.
- The archive's BioEmu checkpoint is not recorded, so a difference between v1.2 and the archive is a
  difference between two checkpoints of unknown distance, possibly also in sampler settings.
- The BioEmu package version on Colab was not recorded; the conversion used here is checked against the
  Colab files (§2).

## 9. Deviations
