# Tools

`tools/` holds the command-line tools of the current workflows (the 2026-10 GA+FL / Region 4 campaign: data builders for the slim
database and the PnET / HBAD inputs, igelmo and gdigel calibration, TOST selection, Sim A / Sim B evaluation, the scenario engine).
Tools that no current config, doc, test or active tool uses are in `tools/legacy/` (list and reasons: `tools/legacy/README.md`); they are
available unchanged at the tag `refactor-step7`.

All tools run from the repository root as `./julia_gdal.sh --project=. [--threads=N] tools/<tool>.jl <args>`; `PAN_DATA` / `PAN_OUT`
select the data and output directories where a config uses `${PAN_DATA}` / `${PAN_OUT}`. Usage lines are the first comment lines of each file.

## How a tool was checked (Step 8)

Two checks, recorded per tool below:

* **static** (every `.jl` tool): `./julia_gdal.sh --project=. tools/check_tool_api.jl tools/<tool>.jl` lowers the file, resolves every free
  global and every `Pan.X` / `P.X` / `P.Data.X` path against the loaded `Pan`, and reports what does not resolve. All `tools/*.jl` print
  `OK` except the three fragments that are `include`d by a parent (`calib_levers.jl`, `calib_minimize.jl`, `tost_common.jl` need the
  globals of `calib_coord.jl` / `tost_select.jl`; they resolve when checked through the parent). `.sh` / `.sbatch`: `bash -n`. `.py`: `ast.parse`.
  A static check proves names, not signatures: it did not catch the two real breaks found by the runs (below).
* **run**: the tool executed on a tiny input and completed (exit 0). The input is the golden-harness configuration (80 fixed GA+FL plots,
  `test/golden/plots.csv`, seed 1337, 2-fold split, merged from `configs/region4/gafl_{A,B}.yml` / `gafl_pnet.yml` / `gafl_hbad.yml`
  with the `common()` overrides of `test/golden/scenarios.jl`; `tiny_*.yml`) and the outputs of the golden scenarios `bs_igelmo`
  (params `best_params@2.jld2`, run dir), `bs_gd_mo` (gdigel dir) and `bs_simB` (Sim B run dir), copied to a scratch directory.

Real breaks found and fixed by the runs: `seed_center_bs.jl` (called `generate_biomass_params` without the `BSTables` argument, a MethodError
since Step 3d) and `w_decompose.jl` (one `InjectionPlan` was reused for the legacy and the new scoring order; since Step 3b a plan is only
valid under the scoring rule it was built with, so `fit_params` refused it). `plot_spinup_process.jl` (YAML read without expanding
`${PAN_DATA}`) and `simB_regen_diff.jl` (hand-rolled data preparation that no longer matches a run's species lists) failed and moved to legacy.

## Active tools

Status: **RUN** = executed on a tiny input and completed; **STATIC** = names resolved, not executed (reason given).

### Data builders (inputs of the configs)

| Tool | What | Status and evidence |
|---|---|---|
| `build_slim_db_region.jl` | Slim DuckDB of a state x L3 region from FIADB + the Sim B cohort DB (`--out`, `--states`, `--ecos`) | RUN: `--states 13 --ecos 8.5.3`, wrote a new DB |
| `build_floors_region.jl` | Region B_MAX / ANPP data floors from a `split_only` run dir (`split_plots.csv`, `species_map.csv`) | RUN: `--run <gd_out> --db slim_r4.duckdb --out <scratch>/floor_x` wrote `_bmax.csv`, `_anpp.csv` |
| `build_pnet_climate.jl` | PRISM L4 monthly climate cache (JLD2) for PnET | RUN: `--y0 2000 --y1 2002` wrote the cache |
| `build_plot_soil.jl` | Per-plot SOLUS soil hydraulics for PnET | RUN: on `slim_r4.duckdb`, wrote `plot_soil` CSV |
| `export_cohort_disturbance.py` | Per-cohort disturbance drops (parquet, read by `disturbance_full_removals`) | RUN: `--out <scratch>/disturb_x.parquet 13` (python `-I`) |
| `build_treemap_donors.py` | TreeMap donor table for the scenario engine | STATIC: `ast.parse`; writes to the hard-coded `/workspace/landis_scen/data/` (outside this worktree), not run |
| `fit_volume_eqs.py` | Region-4 volume equations (`volume_eqs_r4.csv`) | STATIC: `ast.parse`; same hard-coded output location, not run |
| `region_inventory.jl` | Per-L3 inventory (remeasured plots, site classes, dominant species) | RUN: on `slim_r4.duckdb` |
| `hbad_common.jl` | Shared helpers of the HBAD tools | STATIC (library); used by the three HBAD tools below |
| `hbad_cohorts.jl`, `hbad_priors.jl`, `hbad_translation.jl` | HBAD cohort table, priors CSV, empirical cohort translation (write `results/hbad/*.csv`, which the HBAD plugin reads) | RUN: the chain `hbad_cohorts.jl` -> `hbad_translation.jl` -> `hbad_priors.jl` with the new `PAN_HBAD_OUT=<scratch>` override of `HBAD_OUT` (`hbad_common.jl`), so the tracked `results/hbad` stayed untouched. All three completed. Compared with the tracked `results/hbad`: `cohorts_gafl_head.csv`, `translation_selection.csv`, `priors_dgrowth_by_dbh.csv`, `priors_mortality_table.csv` are byte-identical; the other CSVs differ in the trailing digits (fitted coefficients up to 3e-4 relative, e.g. `g_b1_lnD`; same shapes) and `species_group.csv` lists the same rows in another order, so the fit is not bit-reproducible run to run |
| `hbad_translate_lib.jl` | Stand-alone translation library (CSV + DataFrames), loaded by `src/plugins/HBAD/priors.jl` | RUN: loaded by `test/hbad_unit.jl` (part of `test/runtests.jl`) |

### HBAD2 calibration (docs/hbad2_calibration.md)

| Tool | What | Status and evidence |
|---|---|---|
| `hbad2_intervals_check.jl` | Rebuilds the remeasurement-interval product from the database and compares it with the prototype's Arrow files (order independent) | RUN: 30,703 intervals / 766,365 tracked trees / 183,698 ingrowth rows identical |
| `hbad2_calib_parity.jl` | Pan's Sim A / Sim B objectives against the prototype harness (read-only, same process) for `fit_v2j` and `fit_v2fv5`, train and val | RUN: max relative difference 1.3e-14, `CALIB PARITY PASS` |

### Calibration and selection

| Tool | What | Status and evidence |
|---|---|---|
| `train_resumable.sh` | Resumable `run_from_yaml` launcher (checkpoint scan, rollback, ramdisk option) | RUN: `PAN_EXTRA_OVERRIDES='"trials"=>6' tools/train_resumable.sh tiny_A.yml <scratch>/tr_out 2` wrote `.train_done` and `fold_1/best_params@*.jld2` |
| `gd_run.jl` | `search_mode: gdigel` from a config | RUN: 4 trials, `mu=4 ckpt_every=2 min_pairs=3` |
| `gd_check.jl` | gdigel in-loop group bias at a params set | RUN (tiny gdigel config) |
| `gd_equifinality.jl` | Equifinality report of a gdigel run | RUN: on the `bs_gd_mo` dir, `train` split |
| `gd_select_distinct.jl` | K behaviourally distinct sets from a gdigel run | RUN: K=2 |
| `mo_archive_pick.jl` | Knee and extremes of a gdigel `mo_archive.csv` | RUN: printed the three picks |
| `calib_coord.jl` (+ `calib_levers.jl`, `calib_minimize.jl`) | Coordinate-wise per-group calibration (`root` and `CALIB_MODE=minimize`) | RUN: schedule `S,ratio` (root) and `S` with `CALIB_MODE=minimize` |
| `tost_select.jl` (+ `tost_common.jl`) | Rank archive candidates by TOST equivalence | RUN: 4 candidates of the `bs_igelmo` run, `val` split |
| `cmaes_diag.jl` | Landscape / noise diagnostic around a params set | RUN |
| `w_decompose.jl` | Decomposition of the A and W terms (legacy vs new scoring order) | RUN after the fix above |
| `shock_stats.jl` | Disturbance-shock matching statistics | RUN |
| `check_rankw.jl` | Per-stratum RANKW weights sanity check | RUN |
| `scatter_archive.jl` | Sim-vs-obs scatter / R2 / slope per archive candidate | RUN: 2 candidates |
| `analyze_run.jl` | Post-hoc analysis of a finished MO run | RUN: on the `bs_igelmo` run dir |
| `gafl_cv.sh`, `gafl_run_cv_real.sh`, `rescore_v1_5pct.sh`, `simB_lowvar.sh` | Multi-stage CV / re-score drivers that call `calib_coord.jl` + `tost_select.jl` (each stage is a tool listed here) | STATIC: `bash -n`; every `tools/` / `docs/` / `configs/` path they name exists; the stages they chain were run individually. Not run end to end: 4 calibration fits per call |
| `regress_gate.sh` | Fixed-seed regression gate of the default plugin (`docs/region4_pnet.md` section 7) | RUN, completes (40 evaluations, `REGRESS_DONE n_arch=4`), but does NOT match `docs/regress/baseline_853_split-productive-species.txt` (4 archive rows vs 3, different objectives). Expected: the baseline was recorded on the pre-refactor branch, before the behaviour fixes of Steps 1-6 (listed in the refactor WORKLOG); it has to be re-recorded on this tree before the gate can pass again. Needs `/workspace/landis_parametrization_julia/runs/fl853_igelmo_4cell_shade_n1000_s_anpp.yml` |
| `submit.sbatch`, `submit_simB_freeze.sbatch` | Slurm wrappers around `train_resumable.sh` / `select_simB_seeds.jl` | STATIC: `bash -n` (no Slurm here); the two commands they wrap were run |

### Sim A / Sim B campaign

| Tool | What | Status and evidence |
|---|---|---|
| `seed_center_bs.jl` | Neutral Biomass Succession start (box centre) | RUN after the fix above |
| `seed_simB_start.jl` | Sim B start remapped from a Sim A set | RUN |
| `seed_pnet_start.jl`, `seed_hbad_start.jl` | Deterministic PnET / HBAD starts | RUN |
| `select_simB_seeds.jl` | Seed DB (5 front positions + K crowding picks) from a Sim A run | RUN: `select_simB_seeds.jl <bs_igelmo run dir> seeds.duckdb --mu 6 --warmup 0` (`--mu` must exceed 5) |
| `simab_runB.jl` | Sim B igelmo run with `key=value` overrides | RUN: 4 trials |
| `simab_pick.jl` | `knee` / `seeds` / `pickB` helpers of the campaign script | RUN: `knee`, `seeds`, `pickB` |
| `simab_eval.jl` (+ `simab_eval_tost.jl`, `simab_eval_ctx.jl`) | Pass-criteria evaluation of a Sim A (+ Sim B) result | RUN: `simab_eval.jl tiny_A3.yml bs.jld2 simB_start.jld2 out.md` (A and B; `tiny_*3.yml` = the tiny config with `n_folds: 1, test_frac: 0.25, train_cap: auto` because the tool scores val and test) wrote the full report (sections 1-6 + verdict). Environment used: `SIMAB_THREADS=2 SIMAB_REF_PARAMS=<params> SIMAB_REF_CACHE=<scratch>`; the defaults point to a params file in `/workspace/landis_scen` and cache into `<repo>/runs_local/` |
| `export_params_yaml.jl` | Export a calibrated JLD2 as a self-contained YAML roster for the scenario engine | RUN: wrote `exp.yaml` |

### Scenario engine and checks

| Tool | What | Status and evidence |
|---|---|---|
| `pan_worker.jl` | JSON-lines worker of the scenario engine (`docs/scenario_campaign_plan.md`) | RUN: `ping`, `list_params`, `default_params`, `simulate` (2 plots, 1 harvest), `shutdown` over stdin/stdout |
| `compare_param_sets.jl` | Project the same plots with several roster YAMLs | RUN: the three `r4_*_v1.yaml` sets, 2 cycles, 2 plots |
| `scenario_plugin_check.jl` | Scenario-engine hash of the three r4 rosters (regression gate) | RUN: part of the Step 8 gates (hashes below in the WORKLOG) |
| `check_tool_api.jl` | Static API check of `tools/*.jl` | RUN on every tool (this page) |

## Legacy

See `tools/legacy/README.md`. The Pan-calling legacy tools were not run; `tools/check_tool_api.jl` still works on them.
