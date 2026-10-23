# Nested cross-validated calibration of the five Pan models, GA+FL (EPA L3)

Branch `nested-cv` (from `fe-engine` 5f53ec6), worktree `landis_ncv`. HBAD v2 recruitment and HBAD3P stages run from branch `hbad3-ncv`
(from `hbad3`), worktree `landis_hbad3_ncv`. User request 2026-10-23: calibrate Biomass Succession (BS), PnET, HBAD v1, HBAD v2, HBAD3P on
GA+FL with nested CV, resumable, pipeline double-checked before the long run. Campaign root: `runs_local/ncv/` (git-excluded).

## 1. Protocol
* **Plots.** All FIA plots of GA+FL (STATECD 12, 13) in `slim_r4.duckdb` `PLOT`. Every remeasurement of a plot stays in one fold.
* **Outer folds (5), spatially blocked.** Plots are grouped in 0.25° lat/lon blocks (~25 km; FIA public coordinates are fuzzed ≤ 1.6 km). Blocks are
  dealt to the folds in a seeded random order, each block to the fold with the fewest plots of the block's dominant L3 (ties: fewest plots
  overall), so every fold holds every L3 in proportion and neighbouring plots never straddle train and test. `tools/ncv/make_folds.py`.
* **Roles per outer fold k (nested hold-out).** test = fold k (report only), inner val = fold k mod 5 + 1 (selection only), train = the other 3.
  **Final fit** (the deployed set): train = folds 2–5, val = fold 1, no test; its performance estimate is the pooled outer-fold result.
* **Split files.** `runs_local/ncv/folds/split_<role>.csv` (`statecd,unitcd,countycd,plot,set`), roles k1…k5, final. BS / PnET / HBAD v1 read it
  through the new top-level key `split_csv`; HBAD v2 through `hbad2_calib.split_csv`; HBAD3P through a per-fold copy of the interval product whose
  `set` column is rewritten from it. The species-balanced train cap (`train_cap: auto`, k 4) still applies inside train; its surplus goes to val.
* **Data-derived inputs are rebuilt per fold from that fold's train plots:** BS B_MAX / ANPP floors (`tools/build_floors_region.jl` on the
  `split_only` output), HBAD v1 empirical priors + translation (`tools/hbad_{cohorts,translation,priors}.jl` with `PAN_HBAD_SPLIT`; `PAN_HBAD_TRAIN_ONLY=1`
  so the SDImax statistics — the plugin's SDIMAX default and range, the HBAD v2 start — use train plot-visits only), HBAD v2 recruitment data fit,
  HBAD3P Hmax / recruitment (inside its stages).
* **Not rebuilt (accepted, documented; structural, no test VALUES enter a fitted number):** species tiering (which species get own parameters; all
  plots); the HBAD v1 translation STRUCTURE (`PAN_HBAD_FIX_SELECTION` = the deployed `results/hbad/translation_selection.csv`, chosen on the old random
  split's val: the model definition is kept, its coefficients are refit per fold — an automatic re-selection on fold k1 picked a structure the plugin
  does not implement); the stratum of a plot for the train cap (its rarest species, counted over all plots, as the drawn split); shade tolerance /
  longevity trait tables (HBAD v2: a pooled key's longevity is the median over the symbols present in any set, as the prototype); the species-key labels
  of HBAD3P (keys of species big in TRAIN; a test-only rare species falls into the existing _S / _H key).

## 2. Calibration per fold
### BS, PnET, HBAD v1 (Sim A, cohort AGB): two candidate families, inner val picks
* **Family bias**: gdigel `objective: bias` (μ 16) → best solution → `calib_coord.jl` root refinement (TOST_REAL_ONLY=1, CALIB_SIZE_THR=0).
* **Family mo**: gdigel `objective: mo` (μ 16, 8 weights, patience 15, rel_tol 0.005) → archive picks knee / min_A / min_W → `calib_coord.jl`
  `CALIB_MODE=minimize` with each pick's λ and refs.
* Trials and refinement schedules = the Oct-14 production recipe (`runs_local/mogd_full.sh` of `mo-gdigel`): BS 6400, HBAD v1 6400, PnET 4800;
  schedules BS `S,ratio,bmax,S,ratio,bmax,ratio~,S~,bmax~,ratio~,S~,bmax~` (root) / `ratio~,S~,bmax~,ratio~,S~,bmax~` (minimize), HBAD v1
  `gmult,gmult~,mmult~,gmult~,mmult~,gmult~` / `gmult~,mmult~,gmult~,mmult~`, PnET `foln,foln~,foln~,foln~,foln~` / `foln~,foln~,foln~,foln~`.
* **Selection (inner val only)**: 8 candidates (bias raw + refined, 3 mo picks raw + refined), `tools/tost_select.jl … val` ranked by groups
  equivalent at ±20 %, then ±10 %, then ±5 %, then lower val mean |log1p(sim) − log1p(obs)| (`TOST_RANK=20,10,5,acc`).
### HBAD v2 (tracked trees + free-run stand table)
v2j recipe per fold: recruitment Poisson fit on train sapling ingrowth → Sim A fit with vigour (`objective: sim_a`, cold start from the data-based
`init_theta`, i.e. no start fitted on other splits) → joint fit (`objective: joint`, `fb: margins`, λ 0.75, `recruitment: kappa`, mortality frozen at
the Sim A optimum). The joint scalarisation's ideal/nadir (`tools/ncv/hbad2_refs.jl`): fA anchored at the start (this fold's Sim A optimum keeps j15's
normalised start position −0.011, span scaled by the Sim A train cohorts); fB ideal and nadir scaled by the Sim B train sites (a sum of per-site terms; not
anchored, so a poorer cold start does not move the target; the nadir is raised to the start if the start lies above it). Both rules return j15's ideal /
nadir exactly on j15's own data (checked). No candidate family → inner val is reported, not used to choose.
### HBAD3P (process-grounded; tracked trees)
Cold start (σ 0.15, 6000 evals, as `P3_APP`) `fit3p` → `recruit3p` → `freerun3p` (val and test).

## 3. Evaluation (outer test only, pooled over the 5 folds)
* BS / PnET / HBAD v1: the val-selected set on test: per-group TOST ±5/10/20 % (pooled out-of-fold pairs: one TOST per group on the union of
  the 5 test folds' pairs, each predicted by the fold that never saw it) + mean |log ratio| and W per group; per-fold numbers and spread.
* HBAD v2 / HBAD3P: test Sim A NLL (gain over the null per tracked tree) and free-run BA / N≥12.7 / QMD / Lorey-H bias.
* Equifinality / stability: spread of each fitted parameter across the 5 folds' selected sets.
* (Stage E, after calibration) common free-run comparison of all five deployed per-fold sets on their test plots through the scenario engine.

## 4. Resumability
`tools/ncv/campaign.sh <models|all> [roles]` runs every step in order; each step writes `<step_dir>/.done` only after success and is skipped
when the marker exists; gdigel and HBAD v2 searches restart from their second-newest numbered checkpoint (`gd_state@N.jld2`, `search_state@N.jld2`; the
newest may be truncated by a kill during its write) when the directory exists without a marker; short steps rerun from scratch. A resumed search is not
bit-identical to an uninterrupted one (the RNG stream restarts) — statistically equivalent. One campaign at a time (`flock` on `<root>/.lock`). The driver
runs from a snapshot copy, so editing `campaign.sh` during a campaign is safe (it takes effect at the next start). Split checks
(`tools/ncv/verify_splits.py`) run right after each model's first fit of every role.
`tools/ncv/status.sh` prints done / running / pending steps. After a container restart: rerun the same `campaign.sh` command.

## 5. Checks before the long run
1. `tools/ncv/verify_folds.py`: folds partition the plots, blocks are whole, L3 shares per fold, role files consistent (test ∩ train = ∅, …).
2. Smoke campaign (`NCV_SMOKE=1`: tiny budgets) through EVERY step of all five models for role k1 and final, including an interrupted-and-resumed step.
3. `tools/ncv/verify_splits.jl`: the train / val / test plots each model actually loaded (its `split_plots.csv` / interval sets) equal the role file.
4. Independent review of the driver and code changes; findings verified before launch.
