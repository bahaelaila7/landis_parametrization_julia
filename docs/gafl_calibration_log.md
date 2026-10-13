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

### D8 results — 3-fold CV over the existing splits (all pairs), all-data fit
| held-out | calibrated on | pass ±10 % | pass ±20 % | median abs bias |
|---|---|---|---|---|
| val | train+test | 20/48 | 37/48 | 3.9 % |
| train | val+test | 17/46 | 35/46 | 6.0 % |
| test | train+val | 19/46 | 31/46 | 6.9 % |
All-data fit (`runs_local/final_calibrated_gafl.jld2`): in-sample 42/48 (±10 %), 45/48 (±20 %). Held-out biases average ≈ 0
across folds with alternating signs (sampling variance; 13 groups pass ±10 % in all folds, 26 pass ±20 % in all folds).
Training objectives (warm-start eval, new `igel_warm_start`): W 0.091–0.094 (Stage-A best 0.0955, better) but AGB objective
2.70 vs 2.59 (worse): the calibration zeroes the MEAN log bias by over-predicting mid-size cohorts (+12 % at 1–5 kg m⁻²)
while still under-predicting the smallest (−14 % < 100 g m⁻²); total absolute error +19 %.

## D9. Phantom-backfill cohorts are synthetic — evaluate and calibrate on REAL records only (2026-10-13)
`curated_cohorts_landis_stdorg` contains `intro_type = phantom_backfill` rows (FIA_DATA_PREP/cohorts_landis.py: pre-debut
cohort biomass extrapolated from a sapling growth model, not measured). GA+FL share by cohort size: < 100 g m⁻² **46.5 %**,
100–1,000 3.5 %, 1–5 kg 1.2 %, > 5 kg 0.8 %. The "under-prediction of small cohorts" (D4) — and the phantom → first-real-
tally jump — is therefore largely a data artifact, and D4–D8 calibrated toward it. Code: `make_splots` now carries an
AGB-weighted `phantom_frac` per aggregated cohort into splots and injection cohorts (evaluation-only; not read by the
simulator or the loss — gate byte-identical); `TOST_REAL_ONLY=1` keeps a matched pair only if the observed cohort AND the
same cohort at the plot's previous visit (where the simulated interval started) are majority-real.
Evidence: the D8 all-data parameters, evaluated real-only on all GA+FL plots: **5/47 at ±10 %**, 33/47 at ±20 %, median
|bias| 13.2 %, with 45/47 groups OVER-predicted by a near-uniform +7…+16 % in every size class (no size structure left) —
the real growth shape is captured; only the level is off because it was fitted to inflated phantom growth.
Decision: redo the calibration and the 3-fold CV with `TOST_REAL_ONLY=1` (same fixed procedure, from the Stage-A start).
The real-only TOST is the reported criterion from here on (all-pairs numbers kept for reference).

## D10. Final result — GA+FL BiomassSuccession parameters, real-only calibration (2026-10-13)
Procedure (fixed in advance, identical for every fold): start = Stage-A best (≈ prior medians); `tools/calib_coord.jl` with
`TOST_REAL_ONLY=1`, schedule `S,ratio,bmax,S,ratio,bmax,ratio~,S~,bmax~,ratio~,S~,bmax~` (`runs_local/run_cv_real.sh`).

**Out-of-sample (3-fold CV over the existing splits, real records only):**
| held-out | calibrated on | pass ±10 % | pass ±20 % | median abs bias |
|---|---|---|---|---|
| val | train+test | 23/47 | 38/47 | 4.3 % |
| train | val+test | 18/46 | 37/46 | 4.8 % |
| test | train+val | 19/46 | 32/46 | 6.3 % |
| mean | | ≈ 20/46 (43 %) | ≈ 36/46 (77 %) | ≈ 5.1 % |
Robust groups (held-out ±10 % in all 3 folds, 14/47): LIST2, ACRU, QUNI, _H_HST, NYBI, LITU, QULA3, QUAL, PIEL·AB, PIPA2,
CAGL8, TAAS, QUST, QUPR2. Held-out ±20 % in all 3 folds: 29/47. Remaining held-out misses are sampling variance (biases
average ≈ 0 across folds with alternating signs; cohort pairs are clustered within plots, so a group mean varies by several
% between plot samples), concentrated in groups with < ~1,500 pairs.

**Deployable parameter set (calibrated on all GA+FL plots):** `results/gafl/params_gafl_biomass_real.jld2`
(+ readable `params_gafl_biomass_real.csv`: eco, species, S, B_MAX, ANPP_MAX, ratio, D, LONGEVITY, SHADE_TOL).
In-sample (all plots, real records): **42/47 groups at ±10 %, 45/47 at ±20 %**, median |bias| 0.1 %.
- Size structure on real records is flat: median bias +2 % (< 100 g m⁻²), 0 % (100–1,000), +6 % (1–5 kg), +1 % (> 5 kg)
  (defaults: −42 / −31 / −16 / −12 %); total absolute error 92.3 M g m⁻² (phantom-fitted D8 set: 117 M).
- Training objectives (warm-start eval on the leveled train split): W 0.0952 (Stage-A 0.0955), AGB 2.644 (Stage-A 2.587;
  the training AGB loss still scores phantom cohorts, which this calibration deliberately no longer chases).
- Parameters: S 0.085–0.84 (median 0.20), ratio 20–35 (19 targets at 20, 24 at 35), B_MAX median 24,116; 110/508 targets at
  ≥ 34,900 — where the data floor (20 × p99 growth) exceeds the 35,000 box the floor wins (same as Pan's decode) ⇒ the B_MAX
  box is too tight for GA/FL fast growers (candidate follow-up: widen the upper bound).
- Files: `results/gafl/` — per_group_summary.csv (CV + final per group), cv_heldout_{val,train,test}_groups.csv,
  final_insample_tost_groups.csv, final_calib_rounds.csv.

Not done / possible next iterations: (a) widen the B_MAX/ANPP box and re-calibrate the 110 pinned targets; (b) a warm-started
CMA-ES refinement (`igel_warm_start`) on W/AGB with the training loss restricted to real records; (c) the training
pipeline itself still includes phantom cohorts in its W/AGB objectives — excluding them there is a data-pipeline decision
for the user.

## D11. Sim A follow-up: widened B_MAX/ANPP box (2026-10-13) — no out-of-sample gain; v1 stays deployable
Motivation (D10 limitation a): 110/508 B_MAX targets sat at the 35,000 box and the ANPP ratio at its bounds (19 at 20, 24 at 35).
Change: yaml `bmax_upper` (upper B_MAX bound; lower stays 12,000) and `anpp_ratio_bounds` (`BSP.BMAX_BOUNDS` /
`BSP.ANPP_RATIO_BOUNDS`, also read by `tools/calib_coord.jl` for its grids); `configs/region4/gafl_A2.yml` = gafl_A +
`bmax_upper: 60000`, `anpp_ratio_bounds: [15, 40]`. Same fixed procedure (`tools/gafl_cv.sh`, real-only TOST, D10 schedule).
Also added: yaml `drop_phantom` (SQL `intro_type <> 'phantom_backfill'` — training data without phantom cohorts; D10 limitation c;
default off ⇒ unchanged) and the roster margin ±5 % (all tools now report ±5/10/20 %).

| held-out | v2 (wide box) ±5 / ±10 / ±20 % | v1 (D10) ±10 / ±20 % |
|---|---|---|
| val | 8 / 22 / 36 of 47 | 23 / 38 |
| train | 5 / 14 / 34 of 46 | 18 / 37 |
| test | 9 / 18 / 31 of 46 | 19 / 32 |
| mean | ≈ 7 / 18 / 34 | ≈ 20 / 36 |
In-sample (all plots): v2 33/47 ±5 %, 41/47 ±10 %, 44/47 ±20 % (v1: 42/47, 45/47). Robust in all 3 folds: v2 3 (±5 %), 12 (±10 %), 26 (±20 %).
The wider box WAS used (35 targets at the new 60,000 cap, 61 ratios at 15, 58 at 40) but bought nothing: per-group AGB
bias is matched by many (B_MAX, ratio, S) combinations, so extra room only adds variance. **Decision: keep v1
(`results/gafl/params_gafl_biomass_real.jld2`) as the deployable Sim A set**; v2 kept for reference in
`results/gafl/A2_bounds/`. The pinned v1 targets are not a binding constraint on the held-out AGB fit.

## D12. Sim B setup (GA+FL, timber-only free simulation) (2026-10-13)
Config `configs/region4/gafl_B.yml`: table `curated_cohorts_landis_simB` (real cohorts with DIA ≥ 5 in; phantoms and
saplings dropped), `dual_mode: b` (free run from the first visit; no injection; observed partial-disturbance drops applied to
matching cohorts, `simB_disturb_only`), establishment ON, SHADE_TOL/MATURITY from data, tier-4 per-plot-mean objective.
Start: `tools/seed_simB_start.jl` remaps the Sim A v1 growth parameters by (eco cell, species) NAME (Sim B has its own
species/eco lists: 504/504 cells matched), sets SHADE_TOL/MATURITY/PROB_ESTAB from data, MIN_REL pinned.
Evaluation (`tools/tost_common.jl`, Sim B mode): sim vs obs AGB from `Pan.resim_simB_paired` restricted to age bins ≥ 3
(age > 20 yr, `SIMB_MIN_BIN`): observed cohorts are timber-only while the simulation has no size cut-off, so the youngest
bins are not comparable (regeneration/ingrowth below 5 in is unobserved).

## D13. Two Sim B defects found and fixed before calibrating (2026-10-13)
First evaluation: 0/47 groups at any margin, sim ≈ 0.3 × obs from the first simulated year. Diagnosis by a per-cohort
probe (year-0 SoA Σ c_bio = observed Σ, 22.96 M vs 22.98 M ⇒ initialisation fine; median cohort retained 28 % after ONE year;
many cohorts at ~1e-6 of their biomass):
1. **Seeding bug** — `seed_simB_start.jl` left `PROB_MORT_SPP` at the random prior draw of `generate_biomass_params`
   (median 0.56 yr⁻¹ catastrophic death); Sim A has 0. Fixed: PROB_MORT is remapped with the growth parameters (new cells 0).
2. **Sim A never scores whole-cohort death.** With PROB_MORT fixed, the free run OVER-predicts (+28 % at cycle 1, +57 % at
   cycle 2). The Sim A TOST pairs are an inner join on (plot, year, species, age) — survivors only — so Sim A growth is growth
   *conditional on survival*. In the data, 13–16 % of cohort AGB present at a visit is gone at the next one (whole cohort
   absent; ±0–3 yr age tolerance), and 5–6 % of next-visit AGB is ingrowth/age-mismatch. BiomassSuccession's only
   whole-cohort death process besides longevity is `PROB_MORT_SPP`, so Sim B calibrates it (new `pmort` lever in
   `calib_coord.jl`, grid 0–0.045 yr⁻¹, then relative refinement), keeping Sim A's survivor growth as the start.
3. **Test statistic for a free run (D14 below).**

## D14. Sim B equivalence test: plot-cluster bootstrap on the ratio of sums (2026-10-13)
A free stochastic run cannot reproduce WHICH cohorts die on a plot, only how much biomass a group keeps on average, so the
Sim A statistic (mean per-pair log difference) is the wrong estimator: per-pair differences are dominated by
dead-in-sim/alive-in-obs pairs (|log| ≈ 5–10). A first calibration on per-(cycle, eco, bin) cells reached median |bias| 6 %
but 0/47 groups passed (CIs ±30–80 %). Sim B statistic per group: log(Σ sim / Σ obs) over all (plot, cycle) pairs of the
comparable bins; 90 % CI = percentile bootstrap over PLOTS (B = 1000, fixed seed; `SIMB_BOOT`); equivalent at margin m iff the
CI lies within ±log(1+m) (TOST at α = 0.05, same margins as Sim A). The calibrator root-finds the same statistic.
