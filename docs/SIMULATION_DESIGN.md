# Design: coarse-grained simulation of the archive's disordered proteins

Written 2026-10-15, before any simulation was run. Deviations are added, dated, in the last section; nothing
above it is edited after the first simulation starts.

## 1. Question

BioEmu-1's ensembles of the archive's clean disordered proteins are too large: the median ratio of ensemble
Rg (Guinier fit of the ensemble's computed curve) to the measured Guinier Rg is 1.21 (95% interval
1.12–1.28, n = 41). Does a coarse-grained model tuned for disorder, CALVADOS 3 (Cao et al. 2024), run with
OpenMM, reproduce the measured size of the same 41 proteins?

## 2. Proteins

The 41 profiles of the disordered class (mean disorder score ≥ 0.6) that pass the quality checks and have a
BioEmu-1 ensemble (`results/resolvability.csv`, model `bioemu`, `clean`, class `disordered`). The simulated
sequence is the PeptoneDB-SAXS construct sequence (`results/entries.csv`), the one BioEmu-1 was given.

## 3. Measures

**Primary.** For each protein, the ratio r = Rg_sim / Rg_exp, where Rg_sim is the root-mean-square over
frames of the mass-weighted radius of gyration of the residue beads (CALVADOS's own convention, residue
masses with +2 Da on the first and +16 Da on the last bead) and Rg_exp is the Guinier Rg of the measured
profile already in `results/entries.csv`. Reported: the median of r over proteins with a percentile
bootstrap interval (2,000 resamples over proteins, seed 20261015), and, paired on the same proteins,
BioEmu-1's ratio as reported on the page (Guinier Rg of the computed ensemble curve over Rg_exp, column
`rg_ratio`). Paired tests: Wilcoxon signed-rank on log r_CALVADOS − log r_BioEmu, and a sign test on
|log r_CALVADOS| < |log r_BioEmu|.

**Like-for-like sensitivity analyses** (the primary compares a coordinate radius with a curve radius):
(a) CALVADOS bead Rg against BioEmu-1's Cα coordinate Rg (`results/coord_rg.csv`, `ratio_coord_exp`);
(b) the Guinier Rg of CALVADOS's computed ensemble curve (secondary measure below) against BioEmu-1's
curve ratio, the same forward model and the same Guinier fit on both sides.

**Secondary: reduced χ² of a computed SAXS curve.** Forward model: back-mapping of each analysed frame to
all atoms with PULCHRA 3.06 (Rotkiewicz & Skolnick 2008; the tool the CALVADOS authors use), then
Pepsi-SAXS 3.0 (Grudinin et al. 2017) on each frame with the flags the PeptoneBench authors used to build the
archive's curves (`--maximum_scattering_vector 10`, the measured profile given so that Pepsi fits its two
hydration parameters per frame, angular units automatic). The conformer curves are averaged uniformly and
scored exactly as the archive's BioEmu-1 curves were scored here (`bsc.saxs`: one scale and one constant
background, reduced χ², NRMSD, Guinier Rg of the ensemble curve). 100 frames per protein, evenly spaced over
the production run, the median number of BioEmu-1 conformers in the archive. χ² is not strictly comparable
with the archive's: the Pepsi-SAXS version the archive used is not recorded, and back-mapped coarse-grained
frames carry reconstruction error that BioEmu-1's all-atom conformers do not. χ² is reported, paired with
BioEmu-1's raw χ², without a pre-declared expectation.

## 4. Simulation settings

From the CALVADOS papers and the published single-chain protocol of the IDRome study (Tesei et al. 2024),
which the package's examples follow:

| Setting | Value |
|---|---|
| Model | CALVADOS 3 residue parameters (`residues_CALVADOS3.csv` of the package's examples), Ashbaugh–Hatch ε 0.2 kcal/mol, cutoff 2.0 nm; Debye–Hückel electrostatics, cutoff 4.0 nm; charged termini; histidine charge from pH |
| Package | CALVADOS 0.9.0, github.com/KULL-Centre/CALVADOS at commit 6981542; OpenMM 8.4, CPU platform |
| Integrator | Langevin, 10 fs time step, friction 0.01 ps⁻¹ (package defaults) |
| Box | cubic, side (N − 1) × 0.38 + 4 nm, periodic; the chain starts as the package builds it (a compact grid at the box centre) and is energy-minimised |
| Length | 1,010 frames, one every 7,000 steps for N < 150, else every ⌈3 × 10⁻⁴ N²⌉ × 1,000 steps (70.7 ns for N < 150; 687 ns for N = 475) |
| Equilibration | the first 10 frames are discarded |
| Frames analysed | 1,000 for Rg; 100 evenly spaced for the SAXS curve |
| Seed | 20261015 + the protein's position in the input table |

**Conditions from the SASBDB record of each entry** (`results/coverage/sasbdb_metadata.csv`):

- Temperature: the recorded cell temperature in kelvin; 293.15 K where none is recorded.
- pH: the recorded buffer pH; 7.0 where none is recorded.
- Ionic strength, from the buffer description by a fixed rule: 1:1 salts (NaCl, KCl, NaNO₃, LiCl) add their
  concentration; 2:1 salts (CaCl₂, MgCl₂, Na₂SO₄, (NH₄)₂SO₄) add three times their concentration; buffer
  agents add the charged fraction at the recorded pH (phosphate: c(1 + 2α) with α = 1/(1 + 10^(7.2 − pH));
  Tris, pKa 8.1, and Bis-Tris, pKa 6.5: the protonated fraction times c; HEPES, pKa 7.5, and MES, pKa 6.1:
  the deprotonated fraction times c); PBS is 0.17 M plus any salt listed beside it. A number without a unit
  before a salt name is read as mM. EDTA, DTT, TCEP, glycerol and sugars add nothing. Where the rule finds no
  ionic species, 0.15 M. The minimum used is 0.01 M.
- Denaturants (urea, guanidinium chloride) cannot be represented and are recorded; guanidinium chloride
  counts towards ionic strength as a 1:1 salt.

Every value and its source (recorded, parsed, default) is written to the input table.

## 5. Convergence

For each protein the 1,000 analysed frames of Rg² are cut into five consecutive blocks. Reported: the block
means, the standard error of the mean from the blocks, and the relative difference between the means of the
first and second halves. A run counts as converged when the block standard error is below 2% of the mean Rg
and the halves differ by less than 5%. Runs that are not converged are reported and kept, and the primary
result is also given without them.

## 6. Expectations, fixed in advance

- **E10.** CALVADOS 3's median Rg ratio over the simulated proteins lies within 0.90–1.10.
  Reasoning: the model's residue parameters were fitted to SAXS radii of disordered proteins and to NMR
  paramagnetic relaxation enhancement data, so the radii of disordered proteins measured by SAXS are the
  quantity it was built to reproduce.
- **E11.** CALVADOS 3's ratio is closer to 1 than BioEmu-1's for most proteins: the share of proteins with
  |log r_CALVADOS| < |log r_BioEmu| exceeds one half, with a two-sided sign test p < 0.05.
  Reasoning: BioEmu-1's training data contain almost no disordered chains (the coverage analysis), and its
  disordered ensembles are 21% too large at the median.

Both are evaluated on all simulated proteins and, separately, without the proteins flagged by the
circularity check and without the proteins measured in a denaturant.

## 7. Circularity check

CALVADOS's residue parameters were fitted to SAXS radii and NMR data of disordered proteins, some of which
may be in this archive. I obtain the protein lists with sequences from the authors' published data:
CALVADOS 1 (Tesei et al. 2021, github.com/KULL-Centre/papers, 2021/CG-IDPs-Tesei-et-al, optimisation inputs),
CALVADOS 2 (Tesei & Lindorff-Larsen 2022, 2022/CG-cutoffs-Tesei-et-al, optimisation inputs) and CALVADOS 3
(Cao et al. 2024, github.com/KULL-Centre/_2024_Cao_CALVADOSCOM: the training set `allproteins.pkl` and the
validation sets `*_validate.pkl`), each at a pinned commit. Every simulated protein is aligned to every listed
protein (local alignment, BLOSUM62, gap open −10, extend −0.5); identity is the number of identical aligned
positions over the length of the shorter sequence. A protein is flagged when identity ≥ 90% to any training
or validation protein, with the set named. Results are reported with and without flagged proteins.

## 8. Compute budget and subset rule

The pilot runs the full protocol on two proteins of different length (51 and 180 residues) and the first 2%
of the steps of the longest (475 residues), on one thread each, and times them. The cost of every protein is
then projected from the measured steps per second, with a power law in N fitted to the pilot timings. The run
uses a pool of at most six single-thread processes. If the projected wall time for all 41 exceeds 24 hours:
first, proteins whose projected time exceeds 12 hours run 505 frames at the same save interval (the first 5
discarded); if the total still exceeds 24 hours, the proteins above the longest length that fits are left out,
and BioEmu-1's ratio is reported on the same subset beside its value on all 41. Any such step is recorded in
§10 with the pilot numbers.

## 9. Limits

- **Resolution.** One bead per residue; side-chain packing, hydration and backbone geometry are absent and
  enter the SAXS curve only through back-mapping.
- **Transferability.** The parameters were fitted on disordered proteins at near-physiological conditions;
  short peptides, proteins with residual structure, and extreme temperature or pH lie outside the fitted
  range. The disordered class here comes from a sequence-based predictor, and some of its members may hold
  folded parts that CALVADOS 3 without restraints treats as disordered.
- **Conditions.** The buffer is reduced to an ionic strength by a rule; denaturants, cosolvents, D₂O and
  specific ion effects are not modelled; temperature enters the potential only through the Debye length and
  the thermostat.
- **Observables.** Rg_sim is a bead radius; the experimental Rg includes side chains and the hydration layer.
  CALVADOS was fitted to experimental radii in this convention, so the primary comparison uses the
  convention the model was built for; the sensitivity analyses in §3 put both models on the same footing.
- **Pepsi-SAXS** is distributed for academic use; it is downloaded at run time and not redistributed.

## 10. Deviations

- **2026-10-15, pilot and compute plan (§8 applied).** The pilot ran SASDJH6 (51 residues) in full, SASDJT7
  (180) and the first 20 frames of SASDLK9 (475), one thread each, with the production settings and seeds, so
  their frames are part of the production runs. Measured speeds: 2,828, 1,124 and 430 steps per second; the
  power law fitted to them projects 209 single-thread hours for all 41 proteins at full length and 42 hours of
  wall time on six workers. By the rule fixed in §8, the three proteins projected over 12 hours (404, 410 and 439
  residues) run 505 frames (the first 5 discarded), and SASDLK9, the longest (475 residues), is left out; the
  remaining 40 are projected at 20.6 hours of wall time. Every number here is in `results/simulations/plan.json`.
  The speed of SASDJT7 entered the fit after 350 of its 1,010 frames.
- **2026-10-15, conformers without a curve.** Pepsi-SAXS can return a curve of NaN values with a zero exit
  code (seen on one frame of SASDJH6). Such conformers are dropped before averaging, as the score stage drops
  non-finite conformer curves of the archive; the number dropped per protein is in `per_protein.csv`.
- **2026-10-15, an analysis added when one protein had been analysed: the Guinier offset of a chain.** The archive's
  Guinier fit (q·Rg up to 1.3) applied to the Debye function of a Gaussian chain returns 0.92 of the true radius
  (`guinier_over_true_rg_gaussian_chain` in `summary.json`; a unit test checks it on simulated random coils). The
  measured radii come from that fit and the primary measure divides a bead radius by them, so a correct chain
  model can sit above 1 by up to that factor. The primary measure and E10 stay as written; the curve-against-curve
  sensitivity analysis of §3 (b), in which both radii come from the same fit, is reported beside it.
- **2026-10-16, proteins run at reduced length in the SAXS and analysis steps.** The SAXS and analysis scripts
  first took the frame count and the frames to discard from the input table (1,010 and 10) for every protein;
  before any result was read they were changed to take both from the compute plan (505 and 5 for the three
  proteins the plan shortened).

## References

Cao F, von Bülow S, Tesei G, Lindorff-Larsen K. A coarse-grained model for disordered and multi-domain
proteins. *Protein Science* 2024. doi:10.1002/pro.5172

Tesei G, Schulze TK, Crehuet R, Lindorff-Larsen K. Accurate model of liquid–liquid phase behavior of
intrinsically disordered proteins from optimization of single-chain properties. *PNAS* 2021.
doi:10.1073/pnas.2111696118

Tesei G, Lindorff-Larsen K. Improved predictions of phase behaviour of intrinsically disordered proteins by
tuning the interaction range. *Open Research Europe* 2022. doi:10.12688/openreseurope.14967.2

Tesei G, Trolle AI, Jonsson N, et al. Conformational ensembles of the human intrinsically disordered
proteome. *Nature* 2024. doi:10.1038/s41586-023-07004-5

von Bülow S, Yasuda I, Cao F, et al. Software package for simulations using the coarse-grained CALVADOS
model. arXiv 2025. doi:10.48550/arXiv.2504.10408

Rotkiewicz P, Skolnick J. Fast procedure for reconstruction of full-atom protein models from reduced
representations. *J Comput Chem* 2008. doi:10.1002/jcc.20906

Grudinin S, Garkavenko M, Kazennov A. Pepsi-SAXS: an adaptive method for rapid and accurate computation of
small-angle X-ray scattering profiles. *Acta Cryst D* 2017. doi:10.1107/S2059798317005745
