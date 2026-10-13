# Design: choosing SAXS measurements for the next ensemble model

*Multi-objective acquisition with Pareto analysis, evaluated retrospectively. Written on 2026-10-13, before any
acquisition result was computed. Sections 1–8 are fixed; later sections, if any, are dated additions.*

## 1. Questions and the boundary of the claim

The first study mapped where raw BioEmu-1 ensembles disagree with 399 SASBDB profiles, and found that the size of
that disagreement cannot be predicted from sequence-level features (R² = 0.05). This extension asks a different
question: which measurements, chosen before they are made, would most improve a model's predictions for *other*
proteins?

- **Q1.** Which objectives, computable before a measurement, predict how much a SAXS profile improves a shared
  model's predictions on held-out proteins?
- **Q2.** Do batches chosen on the Pareto front of those objectives improve held-out predictions faster than random
  choice and than any single objective?
- **Q3.** How much held-out improvement is given up by restricting choices to proteins that are likely to be
  tractable (the trade-off curve)?

**Boundary.** The model that the acquired data update is a small *surrogate correction* to BioEmu-1 with parameters
shared across proteins (§3), not BioEmu-1's network. Results are evidence about which objectives identify useful
data for that surrogate; they do not establish that fine-tuning BioEmu-1 would rank data the same way. Nothing is
re-sampled: all ensembles and computed curves are the public PeptoneBench ones, and the "acquisitions" reveal
profiles that already exist.

## 2. Prior work

- Choosing proteins to label so that a structure model improves: DEWDROP (Xue et al. 2025/2026, *J Comput Biol*,
  doi:10.1177/15578666251405823) selects proteins by Monte-Carlo-dropout batch active learning to fine-tune EquiFold,
  tested by simulated acquisition from an already-labelled pool. Closest in setup; static structures, no SAXS.
- Choosing *measurements within one system* by expected information: FRET label pairs (Dimura et al. 2020,
  *Nat Commun*, doi:10.1038/s41467-020-19023-1), EPR distances (Kazmier et al. 2011, doi:10.1016/j.jsb.2010.11.003),
  reflectometry settings by expected information gain (Treece et al. 2019, *J Appl Cryst*,
  doi:10.1107/S1600576718017016).
- Information content of SAXS for ensembles: Potrzebowski et al. 2018 (*PLoS Comput Biol*,
  doi:10.1371/journal.pcbi.1006641); Shannon channels (Konarev & Svergun 2015, doi:10.1107/S2052252515005163); curve
  discrimination (Hura et al. 2013, doi:10.1038/nmeth.2453).
- Retraining models on solution data that already existed: BioEmu's fine-tuning on folding stabilities (Lewis et
  al. 2025, *Science*); CALVADOS parameters learned from SAXS and PRE data (Tesei et al. 2021, *PNAS*,
  doi:10.1073/pnas.2111696118; von Bülow et al. 2026, *Protein Sci*, doi:10.1002/pro.70694).
- Selection machinery: query-by-committee (Smith et al. 2018, doi:10.1063/1.5023802), BatchBALD (arXiv:1906.08158),
  BADGE (arXiv:1906.03671), core-sets (arXiv:1708.00489), multi-objective Bayesian optimisation (qEHVI,
  arXiv:2006.05078); data valuation (Data Shapley, arXiv:1904.02868).

The literature search found no study that selects proteins for SAXS (or another solution measurement) to improve a
generative ensemble model or a coarse-grained force field across proteins. The combination is the contribution;
each ingredient is established.

## 3. The model the data update

For protein *p* with conformers *k = 1…K* the archive provides one computed curve per conformer, I_pk(q). From each
curve I take the conformer's radius of gyration R_pk (Guinier). Two components, both with parameters θ shared
across proteins:

- **Population shift** (changes weights within the ensemble): w_pk ∝ exp(−λ_p z_pk), with z_pk the conformer's log
  size relative to its ensemble mean and λ_p = θ_λ · x_p.
- **Size scaling** (moves the whole ensemble): Î_p(q) = Σ_k w_pk I_pk(q·s_p), with log s_p = θ_s · x_p and s_p
  bounded to [0.8, 1.25]; only q points whose rescaled value stays inside the computed grid are scored.

x_p holds features known before any measurement: an intercept, the sequence-predicted disorder fraction, log chain
length, and u_p, the log ratio of the model's own ensemble size to the polymer scaling-law size for its length.
θ has eight parameters. θ = 0 reproduces raw BioEmu-1 exactly.

The two components answer a known objection to information-based selection. A protein whose conformers give very
different curves is informative about *weights*; a protein with a narrow but wrongly sized ensemble is not, yet it
can carry the more useful training signal. The population shift covers the first case and the size scaling the
second, so an objective defined on θ (§4, O1) can value both.

**Fitting.** θ̂ minimises Σ_{p acquired} χ²_p(θ)/2 + |θ|²/(2τ²), with each profile's scale and constant background
solved inside χ² exactly as in the scoring stage; τ is fixed before the simulation by cross-validation on the first
split's pool only. **Scoring.** The held-out score is the median reduced χ² of the corrected predictions; NRMSD and
the median Rg ratio of disordered proteins are secondary scores. Improvement is measured against θ = 0.

## 4. Objectives (all computable before the measurement)

No objective reads a hidden profile. The noise of a profile not yet measured is predicted from a model of relative
error against q and molecular weight, fitted only on profiles already acquired.

- **O1 · information about the shared model.** The expected reduction in posterior uncertainty of θ from measuring
  *p*, under a Laplace approximation: the Fisher information J_p = G_pᵀ Σ_p⁻¹ G_p, with G_p = ∂Î_p/∂θ at the current
  θ̂ and Σ_p the predicted noise. Batch value: log det(Λ₀ + J_acquired + Σ_{p∈B} J_p) − log det(Λ₀ + J_acquired), the
  Gaussian form of batch mutual information. This is the training-value objective.
- **O2 · information about the protein's own ensemble.** How strongly a curve could separate the protein's own
  conformers: the expected χ² between the ensemble average and its larger-half and smaller-half sub-ensembles under
  the predicted noise. Kept to test the objection in §3 directly.
- **O3 · model disagreement.** Spread of the predicted Rg across the generators in the archive (BioEmu-1, PepTron,
  IDPFold, Boltz-2, AlphaFold2, ESMFold) for *p*.
- **O4 · coverage.** Sequence distance (3-mer containment, as in the cross-validation groups) from *p* to the nearest
  already-acquired protein; at batch level, the minimum pairwise distance.
- **O5 · feasibility (heuristic).** A fixed score from sequence: chain length, predicted disorder fraction, cysteine
  count, hydrophobic fraction, distance of the isoelectric point from pH 7.4. It is not validated; the only
  retrospective check available is whether it separates the 36 deposited profiles that failed the quality flags
  (aggregation, invalid Guinier region, oligomeric mass) from the rest.
- **Baseline · predicted error**, from the existing `predict` stage (known to be weak).

## 5. Selection policies

- Random.
- Greedy on each single objective (O1–O5, predicted error), batches filled with a coverage penalty.
- **Pareto:** non-dominated sorting of candidates on (O1, O4, O5), batches built greedily by hypervolume
  contribution; a sweep of Chebyshev weightings traces batches across the front; a constrained variant keeps only
  candidates with O5 above the pool median.
- **Oracle:** realised marginal value (§6), which reads the hidden profiles; an upper bound, not deployable.

With a few hundred candidates and cheap explicit objectives, exhaustive non-dominated sorting is sufficient;
surrogate-based multi-objective Bayesian optimisation (qEHVI) is built for expensive black-box objectives and is not
needed.

## 6. Evaluation

- **Splits.** Five grouped splits of the 399 profiles (sequence containment groups): 70% acquisition pool, 30%
  held out. Ten random seed sets of 10 initial proteins per split.
- **Rounds.** Batches of 10, eight rounds (up to 90 acquired); after each round refit θ̂ and score the held-out set.
- **Primary metric.** Area under the held-out improvement curve (median log χ² reduction against the number
  acquired), per policy; differences from random with 95% intervals over splits and seeds.
- **Realised value of each protein.** The average held-out gain from adding *p* to random subsets of 10 and 30
  pool proteins (Monte-Carlo marginal contributions, 20 permutations per split).
- **Objective validity (Q1).** Spearman correlation between each objective and realised value, with grouped
  bootstrap intervals. This tests whether the objectives predict value, independently of how batches are built.
- **Trade-off (Q3).** Held-out improvement against mean feasibility of the acquired proteins, along the Pareto
  sweep.
- **Composition.** Class, length and disorder of what each policy selects, and of the highest realised values.

## 7. Expectations, fixed before running

- **E5.** Fitted on the whole pool, the surrogate improves the held-out median χ² by at least 10% over raw
  BioEmu-1. *Decision rule:* if it does not, the data cannot improve this model, the acquisition comparison is
  reported as uninformative, and only the objective-validity analysis is reported.
- **E6.** O1 correlates positively with realised value (interval excluding 0), and more strongly than O2.
- **E7.** Pareto batches and O1-greedy batches beat random choice on the primary metric (interval of the difference
  excluding 0); the predicted-error policy does not.
- **E8.** Restricting choices to the more feasible half of the pool costs less than a quarter of the unconstrained
  improvement.
- **E9.** Realised value is highest for disordered proteins (median realised value, disordered above folded),
  consistent with the size bias.

Every outcome is reported as found, including a null result for E6 or E7.

## 8. What this would show, and the prospective step

- **Would show:** whether objectives computable before a measurement identify SAXS profiles that improve a shared
  correction to BioEmu-1 on unseen proteins, and what a feasibility constraint costs.
- **Would not show:** that fine-tuning BioEmu-1 itself ranks data the same way; anything about proteins unlike
  SASBDB's composition; whether the feasibility heuristic is right.
- **Prospective step:** a candidate pool of proteins without a SASBDB entry (for example reviewed human proteins or
  domains of at most 350 residues); BioEmu-1 sampling on a GPU and computed curves to evaluate O1–O3; selection on
  the Pareto front; measurement through the campaign's sample-quality gate. Arm 1's repeat measurements supply the
  noise model for O1 and O2, and Arm 2 tests the bias that the size-scaling component encodes.

## 9. Implementation plan

- **Stages:** `bsc objectives` (O1–O5 and the noise model), `bsc surrogate` (fit, E5 check), `bsc acquire` (the
  simulation), `bsc value` (realised values and objective validity); figures and a page section from `report`.
- **Tests:** θ = 0 reproduces the raw χ² of every entry; analytic gradients of the surrogate match finite
  differences; the Fisher-information objective recovers the known optimum on a synthetic design; objectives are
  identical when the hidden profiles are deleted from disk (no leakage).
- **Compute:** CPU only. One surrogate fit over the acquired set takes seconds; the simulation is about 4,000 fits
  (5 splits × 10 seeds × 10 policies × 8 rounds) and the value estimation about 10,000; a few hours on six cores.
- **Page:** one new section, "Choosing data for the next model", with the learning curves, the Pareto front and the
  objective-validity plot; detail collapsed.

## 10. Deviations (added 2026-10-13, after the run)

Departures from §1–§9, each with its reason:

- **A taper at the ends of the computed grid.** §3 scores only the q points whose rescaled value stays inside
  the computed grid. Taken literally, every point that leaves the grid as s changes removes its residual from
  χ² at once, so the objective is a staircase in θ: in a first run the fits stopped at these steps (L-BFGS
  line-search failures, with a gradient norm of about 50 at the reported optimum, and 16 of 40 refits in a
  probe of the marginal contributions not moving from their starting point). A point now keeps weight 1
  while its rescaled q lies inside the grid, and its weight falls linearly to 0 over one grid spacing beyond
  either end, where the last interpolation segment is extended; the reduced χ² is the weighted sum over the
  weights' total minus two. At s = 1 every weight is 1, so θ = 0 still reproduces the raw χ² exactly. All
  reported results come from the second run. The first run also missed E5 (the held-out median χ² rose by 8% on
  average) and gave the same signs for the six objective correlations; its numbers are not used.
- **O4 uses 5-mer containment.** §4 says "3-mer containment, as in the cross-validation groups"; the
  cross-validation groups are built on 5-mer containment (threshold 0.3), so O4 and the coverage penalty use
  the same 5-mers, and the splits, the penalty and O4 share one notion of sequence similarity.
- **`bsc value` runs before `bsc acquire`.** §9 lists the stages in the order objectives, surrogate, acquire,
  value; the oracle policy of §5 ranks by realised value, so the value stage has to exist first.
- **Number of value fits.** §9 estimated about 10,000 fits for the value estimation. Marginal contributions
  for every pool protein outside each subset of 10 and 30, over 20 permutations and five splits, take about
  52,000 fits (each warm-started and small); the definition in §6 is unchanged.
- **A fifth figure.** E5 was not met, so by the decision rule the learning curves, the trade-off curve and
  the Pareto projections are reported as uninformative or descriptive and sit in collapsed panels; the page
  leads with a figure of what the whole-pool correction does to held-out profiles by class, which is what
  the E5 outcome rests on, and with the objective-validity plot. The simulation was run in full and its
  results are kept, labelled as uninformative.

Choices that §1–§9 left open, fixed before the first run except the restarts of the fit, which were added
with the taper:

- *Surrogate.* z_pk is divided by one constant, the pooled within-ensemble SD of ln R over the 399 entries,
  so that λ is in units of that spread; the three non-constant features are standardised over the same
  entries; u_p uses the scaling-law radius blended by the disorder score (the folded and disordered laws of
  the `predict` stage); the bound on s_p is ln s_p = ln 1.25 · tanh(θ_s·x_p / ln 1.25); I_pk(q·s) is read from
  a cubic Hermite interpolant in ln q through the computed points, so s = 1 reproduces them exactly; τ was
  chosen from (0.03, 0.1, 0.3, 1, 3, 10) and came out as 1. Fits use L-BFGS with the analytic gradient,
  restarted with fresh memory (at most twice) when a line search stops at a kink of the taper; on the first
  split's pool, four random starts reach the optimum found from θ = 0.
- *O1.* The Fisher information is profiled: the derivatives are projected orthogonal to the scale and
  background directions (both are solved inside χ²) and divided by the profile's degrees of freedom, which is
  the Laplace approximation of the reduced-χ² objective of §3; Λ₀ = I/τ².
- *Noise model.* log10(σ/I_low) is linear in (1, log10 q, (log10 q)², log10 sequence mass), I_low being the mean
  of the five lowest-angle intensities; fitted on up to 60 evenly spaced points per acquired profile.
- *O2* uses the raw ensemble split at the median conformer size; *O3* is the SD of ln Rg of the uniform
  ensemble curves of the six named generators; *O5* is one minus the mean of five penalties in [0, 1]: length
  outside 50–300 residues, the disorder score, cysteines (saturating at six), hydrophobic fraction above 0.40
  (saturating at 0.55), isoelectric point within 1.5 units of 7.4. The quality-flag check uses the 435 entries
  with a BioEmu-1 ensemble, of which 36 failed.
- *Policies.* The coverage penalty defers a candidate that shares a sequence group with an acquired protein or
  an earlier pick of the batch. O1-greedy conditions each pick on the earlier ones; O4-greedy adds each pick
  to the set the next must be far from. Pareto batches fill from the first front, by hypervolume contribution
  in rank-normalised objective space with the reference at the origin. Chebyshev weights on (O1, O4, O5) are
  ((1 − a)/2, (1 − a)/2, a) for a = 0, 0.25, 0.5, 0.75, 1. The oracle ranks by each protein's realised value in
  the same split.
- *Statistics.* E5 is met when the mean over the five splits of 1 − (held-out median χ² corrected)/(raw) is at
  least 0.10. The area under the curve is the trapezoid over 10–90 acquired divided by 80. Intervals for the
  difference from random resample splits, then seed sets within each split. Objective validity: at every
  subset each objective is ranked among the candidates, the ranks are averaged per protein, and the Spearman
  correlation with realised value is computed per split and averaged; the interval resamples sequence
  groups. E7 is met when the lower bounds for Pareto and O1-greedy exceed 0 and that for predicted error does
  not; E8 when (AUC_Pareto − AUC_constrained)/AUC_Pareto < 0.25 with a positive AUC_Pareto.
