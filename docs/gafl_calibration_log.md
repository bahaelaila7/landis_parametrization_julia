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

## D4. Stage A (CMA-ES, region pooling) stopped; switch to per-group coordinate calibration (2026-10-13)
Evidence (Stage A, ~190 dims, σ0 0.08, μ 16, 8.5 s/gen): train loss 2.6985 → 2.6821 in 241 generations (−0.6 %), val flat
(4.45); the archive's best candidate still had S = 0.5 for every species. Diagnostics on that candidate (`runs_local/
diag_bounds/`, scored with `tools/tost_select.jl`):
- Raising all B_MAX to the 35,000 cap and ANPP to its max (ratio 20) fixed only growth-limited groups (val: QUNI −48 →
  +5 %, QUAL −28 → −3 %, QULA3 −31 → −2 %, PITA·B −12 → +2 %); PITA·A/D, PIEL·AB, PIPA2, LIST2, ACRU stayed at −43…−58 %.
- Bias is **size-structured**: matched cohorts with observed AGB < 100 g/m² are under-predicted by 34–58 %, 100–1,000 by
  0–44 %, while cohorts > 1,000 g/m² are near 0 or over-predicted (PITA +1 %, QUNI +25…49 %). Small cohorts dominate the
  unweighted TOST mean. In Biomass Succession young-cohort growth ∝ b_ap^S·e^(1−b_ap^S): S = 0.5 throttles small cohorts.
- On TRAIN, S ∈ {0.5, 0.25, 0.10} (with B_MAX 35k, ratio 20) moves every group's bias monotonically upward as S drops
  (e.g. PITA·C −42 / −9 / +25 %, LIST2 −47 / +28 / +69 %); each group's zero crossing lies at a different S (mostly
  0.25–0.5) ⇒ the response is close to separable by species.
Decision: stop the CMA-ES Stage A and calibrate per TOST group by coordinate-wise root-finding on TRAIN
(`tools/calib_coord.jl`): evaluate a shared grid of values for one parameter, interpolate each group's bias(value) to its
zero crossing, assemble the per-group values, re-evaluate, iterate (S, then ANPP ratio, then S), because species interact
through competition. Val is used only for selection; test only for the final report. The result seeds a CMA-ES
refinement (`start_from`) on the actual W/AGB objectives.

## D5. Calibration round set 1 (mean-bias, train) → size-targeted calibration (2026-10-13)
`tools/calib_coord.jl` from the Stage-A best, schedule S, ratio, S, ratio (`runs_local/calib1/`):
| step | train pass ±10 % | ±20 % | median abs bias |
|---|---|---|---|
| start (≈ prior medians) | 2/46 | 4/46 | 39.4 % |
| round 1 (S) | 20/46 | 30/46 | 4.2 % |
| round 2 (ratio) | 28/46 | 41/46 | 2.8 % |
| round 3 (S) | 25/46 | 41/46 | 3.5 % |
| round 4 (ratio) | 27/46 | 42/46 | 2.8 % |
VAL (selection split): rounds 1–4 pass ±10 % = 12 / **15** / 14 / 13 of 48 (±20 %: 26 / 28 / 32 / 29), median |bias|
≈ 8 %. Round 2 selected so far (val 15/48 at ±10 %), vs 1/48 for the Stage-A best.
Diagnosis of the train→val gap (round 2, per group): (i) QUNI −46 % on both splits — S/ratio cannot reach zero; it
responds to B_MAX (D4 diagnostic: −48 → −10 % at B_MAX 35k) ⇒ add B_MAX as a lever. (ii) groups that pass on train but
not val (PIEL·C +4 → +22 %, PIEL·D −0.4 → +26 %, LITU −1.5 → −14 %, QUVI +1 → −14 %, PITA·C −2.6 → +11 %): calibrating
only the MEAN lets S and ANPP compensate each other for train's particular mix of small and large cohorts; val (pine-
heavy surplus of the train cap) has a different size mix, so the compensation does not transfer.
Decision: **size-targeted calibration** — S is root-found on small cohorts (observed AGB < 1,000 g/m²), ANPP ratio and
B_MAX on large cohorts (≥ 1,000 g/m²); schedule S, ratio, bmax, S, ratio. Getting both size regimes right should make
the mean bias robust to composition differences between splits. Threshold 1,000 g/m² (10 t/ha) = where D4 showed the
bias regime change.

## D6. Size-targeted calibration rejected; calibrate on train ∪ val (2026-10-13)
Size-targeted calibration (`runs_local/calib2/`, schedule S, ratio, bmax ×2 with S on small and ratio/B_MAX on large
cohorts) gave train 21/46 (vs 28/46 mean-based) and val 16/48 at ±10 % but only 26/48 at ±20 % and median |bias| 10.3 %
(vs 8.2 %) — not a win; the train→val gap is composition (val/test hold the pine-heavy plots moved out of train by the cap),
not mainly size mix. Decision: calibrate the per-group parameters on **train ∪ val** (mean bias), keep **test untouched**
for the out-of-sample report. Rationale: each group's 1–3 parameters are fitted to its own hundreds–thousands of matched
pairs, so the calibration is separable by species — pine-dominated plots cannot drag hardwood parameters, which was the
purpose of the user's species-balanced leveling (still used for the CMA-ES objective). `tools/tost_common.jl` now pools
several splits (`SPLIT="train+val"`, plot ids offset per split).
Result (`runs_local/calib3/`, absolute grids S, ratio, bmax ×2): train∪val pass ±10 % 1 → 32/48, ±20 % 5 → 45/48,
median |bias| 40.4 → 2.5 %.

## D7. Relative refinement (2026-10-13)
Residuals of −13…+7 % remained (coarse shared grid + species interactions). Added relative schedule entries (`S~`,
`ratio~`, `bmax~`): each group's current value × shared factors {0.8, 0.9, 1, 1.1, 1.25}, root-found per group.
Result (`runs_local/calib4/`, from calib3 round 6, schedule ratio~, S~, bmax~ ×2): train∪val **41/48 at ±10 %, 46/48 at
±20 %, median |bias| 0.1 %**. All 7 remaining ±10 % failures have |bias| < 2 % (PITA·D, _GRP_28_LST, _S_LST, _H_NA,
PIEL·X, _H, _S_NA; n = 7–236): variance-limited, not bias. Parameters stay inside bounds (S 0.11–0.73, none at a bound);
110/508 (eco, species) targets sit at the maximum ANPP (ratio 20) and several B_MAX near 35,000 — GA/FL forests want more
productivity than the current box allows (possible future widening of the bounds).

## D8. Out-of-sample result and final procedure: all-data fit + 3-fold CV over the existing splits (2026-10-13)
TEST (untouched until now, 3,050 plots), calib4 params: **19/46 at ±10 %, 30/46 at ±20 %, median |bias| 7.0 %**
(Stage-A start: 2/46, 4/46, 40.1 %). In-sample biases are ≈ 0 for every group, but on test they scatter ±5–18 % with signs
that flip between splits (e.g. GOLA +18 % on val in calib1 vs −18 % on test) ⇒ dominated by **sampling variability**
(cohort pairs within a plot are correlated, so a group mean varies by several % between plot samples), not systematic
bias. Groups with n_test ≳ 1,400 mostly pass; of the 31 groups within ±10 % on test, 12 fail only on CI width.
Decision: (i) the deployable GA+FL parameter set is calibrated on **all** plots (train ∪ val ∪ test) with the identical
procedure (more data per group ⇒ smaller parameter error); (ii) its generalization is estimated by **3-fold CV over the
existing splits** (calibrate on two, test on the third: train+val→test, train+test→val, val+test→train), procedure
fixed in advance: start = Stage-A best, schedule S, ratio, bmax, S, ratio, bmax, ratio~, S~, bmax~, ratio~, S~, bmax~.
