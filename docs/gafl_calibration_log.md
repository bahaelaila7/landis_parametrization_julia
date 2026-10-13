# Georgia + Florida BiomassSuccession calibration — decision log

Goal (user, 2026-10-13): parameterize LANDIS-II Biomass Succession for the plots of **Georgia and Florida** and get the
statistical equivalence tests to pass for as many species as possible. Run locally (8 cores, 32 GB). Document every
decision.

Conventions used throughout:
- **Pass criterion**: TOST equivalence (`test/tost_sim_obs.jl`) of simulated vs observed biomass of surviving cohorts
  (matched by plot, year, species, age; pre-injection), per species group, at margin ±10 % (also reported ±20 %),
  α = 0.05. Groups follow the run's species tiering.
- **Model selection on val, report on test**: candidates are chosen on the validation split; the final pass count is
  reported on the untouched test split.
- Code/config live on branch `region4` (`/workspace/landis_region4`); runs in `runs_local/` (git-excluded).

---

## D1. Scope: GA + FL plots only (2026-10-13)
`configs/region4/gafl_states.yml` = the GA/FL test config with `filter_states: [12, 13]`. 7 L3s contain GA/FL plots:
8.3.5, 8.3.4, 8.5.3, 8.4.4, 8.4.9, 8.4.1, 15.4.1. Split pass: 8,985 remeasured plots; 209 species → 29 exact + 11
groups; species-balanced training cap (user decision, k = 4) train 4,418 → **2,822**; val 3,113; test 3,050.
Floors rebuilt from these train plots: `data_r4/floor_gaflst_{bmax,anpp}.csv` (624 rows).
Why GA+FL only: user instruction ("parameterize those two states"). Consequence: 8.4.9 has only 11 train plots, so its
species pool to L2/region.

## D2. Diagnosis of the previous (whole-ecoregion) test run — the search never left its start (2026-10-13)
Evidence: the best-aggregate candidate of `r4_gafl_test` has S = 0.5, D = 15, B_MAX/ANPP ratio = 27.5 and
B_MAX = 23,500 for almost every species — exactly the midpoints of the parameter boxes ([0.01,1]… / [5,25] / [20,35] /
[12000,35000]); the only deviations are B_MAX values raised to the data floors. Cause: the first point of a Sobol
sequence is the box centre (u = 0.5 everywhere); it beat every other Sobol sample, and afterwards the (1+1)-CMA-ES
offspring with σ0 = 0.3 perturb all ~500 coordinates by ~±0.3 simultaneously, so essentially every offspring was worse
and 20 generations were not enough for the step size to shrink. The across-the-board under-prediction reported in
`docs/region4_pnet.md` §8 is therefore the bias of the **default (prior-median) parameters**, not of a fitted model.

## D3. Search strategy: coarse-to-fine, small initial step (2026-10-13)
- **Stage A (coarse)**: pool every growth parameter at the **region** level (one B_MAX / ANPP-ratio / S per species
  category; PITA and PIEL keep per-site-cell values, PIEL A+B merged) → ≈ 190 dims instead of ≈ 354. Rationale:
  per-species biomass bias — what TOST measures — is first-order a species-level level/shape problem; fewer dims
  converge much faster, and the coarse solution is a good start for refinement.
- **σ0 = 0.08** (was 0.3): step norm ≈ σ·√d; 0.3 in ~200–500 dims makes almost every offspring a large jump.
- **Start at the box centre plus a little spread**: `igel_sobol_init` with `pool_k 1` (plain μ Sobol points; the first is
  the centre) — no 300-candidate screening (it found nothing better than the centre).
- **igel_mu 16, n_reps 1** for Stage A: more generations per hour (each lineage takes more steps); stochastic
  replication matters less than progress at this stage. Stage B returns to n_reps 2.
- **Stage B (fine)**: `start_from` the Stage-A selected candidate, L3 → L2 → region pooling as in D1, σ0 small again.
- Candidate selection from the Pareto archive: by **validation TOST pass count** (not by the aggregate loss), since the
  goal is the tests.
