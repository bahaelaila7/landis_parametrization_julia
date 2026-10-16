# HBAD v2 — biomass-free, process-based cohort model (branch `hbad-v2`, worktree /workspace/landis_hbad2)

Goal (user, 2026-10-16): leave Biomass Succession aside; build HBAD v2 that is **completely biomass independent**:
Biomass-Succession-inspired structure, but **mean height, QMD and TPA drive competition** (QMD–stocking charts: BA, TPA and
stocking are constrained); each species has its own curves of **height vs diameter growth over cohort age**; **TPA only decreases**
(new trees = new cohorts). Work autonomously, iterate on the concept, verify empirically.

Why biomass-free: FIA AGB is not measured, it is an allometric estimate from DIA (+HT) (model on model; it changed with the NSVB
switch), so a biomass-scored model is fitted to an estimate of an estimate. DIA (0.1 in), tree counts and remeasured survival are
measured; HT is measured but noisy (see §1).

## 1. Data audit (tools/hbad2/audit_trees.jl; GA+FL, slim_r4.duckdb read-only)
* Tree-visits: 868,568 live macro trees (TPA_UNADJ 6.018, DIA ≥ 5 in), 124,152 live microplot saplings (74.965, DIA 1–4.9 in).
* Tree identity (plot, SUBP, TREE) is stable across visits: 581,401 live→live, 168,532 live→dead/removed consecutive pairs (dt ≈ 5.4 yr).
* `estimated_age` advances exactly with elapsed time for every tracked tree (age2 − age1 − dt: mean −7e−17), i.e. each tree has a
  fixed birth year → cohort identity by (species, birth-year class) is stable.
* **TPA of an observed cohort is not monotone, although the true cohort TPA is**:
  - 8,118 live saplings cross 5 in and switch from TPA 74.965 to 6.018 (a −92 % "drop" that is a sampling-design change);
  - 143,661 trees are first tallied at a later visit with DIA ≥ 5 in, median estimated age **20 yr** (q10–q90 12.7–30) — ingrowth
    trees join birth-year classes that already exist ("newer trees = younger cohorts" does not hold at the tally threshold).
  ⇒ Sim A scores **closed sets of tracked trees**: a cohort = the trees alive at v1; at v2 its survivors keep their v1 expansion
  weight, so observed N can only decrease and mortality is a binomial count of tree fates. Ingrowth is a separate process.
* Height: HT present for all live trees, = ACTUALHT for 98.5 %; **16 % of remeasured live trees get shorter** (sd ln(H2/H1) 0.10–0.31
  by size) → measured H is noisy (see E1b).
* Plots by number of visits: 1:131, 2:1212, 3:1354, 4:2458, 5:1127, 6:3021.

Data product (tools/hbad2/build_intervals.jl → results/hbad2/, not committed): `intervals.arrow` (30,703 plot intervals; flags trt/dst/harv
at v2, planted at v1), `interval_trees.arrow` (766,365 tracked trees: alive 581,359, dead 51,606, harvest 114,146, missing 19,254),
`interval_ingrowth.arrow` (183,698). Undisturbed intervals (no TRTCD1, DSTRBCD1, harvest fate) with dt > 2 yr are used for fitting.

## 2. Concept checks on trees (tools/hbad2/empirical_checks{,2}.jl; fit TRAIN, score VAL; 403,800 trees)
E1 — competition index (species + ln D, D, ln age always in; VAL):

| index set | R² ln dD | R² dH | mort NLL | mort AUC |
|---|---|---|---|---|
| none | 0.207 | 0.186 | 0.2352 | 0.678 |
| BAL (v1) | 0.266 | 0.200 | 0.2347 | 0.682 |
| SDIH (stand density of TALLER trees) | 0.270 | 0.185 | 0.2337 | 0.690 |
| BAL+RD | 0.266 | 0.205 | 0.2343 | 0.684 |
| SDIH+RD | 0.270 | 0.186 | **0.2322** | **0.696** |
| SDIĤ+SDI (taller by species H–D curve height) | 0.269 | **0.204** | | |

E1b — measured-height ranking creates a regression-to-the-mean artefact: corr(ln H/Ĥ at v1, dH) = −0.28 (trees measured too short
"grow" more). Ranked by Ĥ (species Chapman–Richards height at the tree's D) the effects are: overtopping −30 % ln dD per SD and
−0.072 m/yr per SD of dH; **symmetric crowding (stand SDI) ≈ 0 on dD (+2 %) but +0.040 m/yr (+14 %) on dH** ⇒ crowding shifts growth
from diameter to height (H/D rises with density), suppression reduces both. Implication for the model: rank cohorts by MEAN height
(cohort means average the noise), height growth needs its own process (not an H–D curve).

E3 — species timing (free-growing CCLCD 1–3; mean dH m/yr / dD cm/yr by age class; results/hbad2/timing_free_growing.csv):
pines (PITA 0.86/0.86 at 5 yr → 0.31/0.56 at 70 yr) grow height and diameter together early and height growth collapses with age;
oaks (QUNI 0.42/0.37 → 0.24/0.65; QULA3 similar) keep **increasing** diameter growth while height growth falls (dH:dD 1.1 → 0.4);
LITU/LIST2 height-first; NYBI/TAAS slow, diameter growth rises with age. ⇒ species-specific H and D curves with different timing.

## 3. Model (src/plugins/HBAD2/HBAD2.jl; stand-alone prototype kernel)
See the module header for the equations: cohort state (N, D, H, A); overtopping O from SDI of taller cohorts (+½ own),
Chapman–Richards free-growth curves for H and D in state form, competition modifiers exp(−gO·O + gR·RD) per variable, hazard =
background(size) + suppression (1 − e^{−gOD·O}) + Biomass-Succession age term, self-thinning to SDImax removing suppressed cohorts first.
14 parameters per species (13 fitted + LONGEVITY from species_longevity_ref), 1 global (size slope of background mortality).

## 4. Sim A harness and likelihood (tools/hbad2/common.jl, fit_simA.jl, cmaes.jl)
Cohort = (plot, species key, 10-yr birth class) of the tracked trees; species keys = species with ≥ 3000 train trees (10) + _S/_H.
Each interval is an independent site initialised from the observed (N, D, H, A) and integrated dt years (whole years + one fractional step).
Objective = joint NLL: binomial survival of tree counts (p = N2/N1) + Gaussian ln D2 and ln H2 of the survivors (weight = survivors, σ profiled out).

## 5. Results log
