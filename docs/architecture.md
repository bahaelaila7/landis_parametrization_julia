# Pan architecture (after the Step 1-8 refactor)

Pan fits the parameters of a forest-succession model to FIA remeasurement data with stochastic / evolutionary search, and the same
simulation core is reused by the scenario engine. This page is a map: every statement names the function or file (and line, where the
file is stable) that implements it. Longer design notes live next to it in `docs/` (see the index at the end).

Conventions: paths are relative to the repository root; `file:line` is the line of the definition at the time of writing; names in
backticks are Julia identifiers. `PU` = `Pan.Parametrization`, `Data` = `Pan.Data`.

## 1. Module and file map

`src/Pan.jl` is the module: it includes the sub-modules and then the flat files in dependency order (`src/Pan.jl`, include list).

| Group | Files | What is there |
|---|---|---|
| Engine | `src/PanCore.jl`, `src/types.jl` | `SiteSoA`, `SiteView`, plugin hooks (`scalar_arrays`, `csr_fields`, `csr_arrays`, `process_plugin!`), `@maybe_threads`, `copy_and_reseed_soa`, `reset_soa!`. `FloatType = Float32`, `UIntType = UInt32` (`src/types.jl`). |
| Plugins | `src/Plugins.jl`, `src/plugins/BaseSitePlugin.jl`, `src/plugins/BiomassSuccession/`, `src/plugins/PnET/`, `src/plugins/HBAD/` | The three succession models (module `Plugins`). Each has `params.jl`, `plugin.jl` (kernel + `process_plugin!`) and, for PnET/HBAD, a `context.jl` (immutable data context). |
| HBAD2 calibration | `src/hbad2_calib/*.jl` (module `HBAD2Calib`), `src/data/hbad2_intervals.jl`, `src/search/PlainCMA.jl`, `src/drivers/hbad2_calib.jl` | Calibration of the hbad2 plugin on the remeasurement-interval product: Sim A / Sim B likelihoods, parameter vector, CMA-ES / igelmo through `run_search_loop` (`docs/hbad2_calibration.md`). |
| Plugin interface | `src/succession_api.jl` | Generic functions the fit loop and the drivers call (`step!`, `generate_params`, `make_param_dists`, `make_ctx`, `spinup!`, `add_cohort!`, `plant!`, ...) with one method set per plugin context; `PluginConfig` (`src/succession_api.jl:54`). |
| Parameter space | `src/Parametrization.jl`, `src/parametrize/utils.jl`, `src/parametrize/slots.jl`, `src/parametrize/{BiomassSuccession,PnET,HBAD}.jl`, `src/config/param_space.jl` | `MutableParam`, `ParamDists`, `ParamSpace` (`src/parametrize/utils.jl:161`), `u_to_params` / `params_to_u` (`src/parametrize/slots.jl:81`, `:137`), per-plugin parameter distributions, `make_param_space` / `build_param_tying` / `cma_blocks` (`src/config/param_space.jl:36`, `:125`, `:54`). |
| Loss | `src/loss/{types,kernels,smoothing,site_loss,aggregate,scales}.jl` | `LossParams` (`src/loss/types.jl:20`), `LossScales` (`:90`), `SiteLoss` (`:112`); Wasserstein / AGB kernels; `compute_loss_scales` (`src/loss/scales.jl:10`), `build_split_scales` (`:185`). `types`, `kernels`, `smoothing`, `site_loss` are included from `src/Parametrization.jl` (module `Parametrization`). |
| Config | `src/config/{schema,options,params_yaml,plugin_config,tables,from_yaml,param_space}.jl` | Strict YAML reader (`load_config`, `src/config/schema.jl:623`), option structs, YAML parameter rosters, plugin construction, table loaders, YAML to `DataConfig` / `ScoringOptions` bridges. |
| Data | `src/Data.jl`, `src/data/utils.jl`, `src/SpatialData.jl`, `src/data/spatial.jl`, `src/Spatial.jl` | Cohort database loading, split, strata, tiering, reference-table helpers (`Data`); raster / LANDIS loaders for the spatial runs (`SpatialData`, `Spatial`). |
| Reference + sites | `src/reference.jl`, `src/sites.jl` | `Reference` / `build_reference` (`src/reference.jl:38`); `make_sites` / `_make_sites` (`src/sites.jl:64`). |
| Scoring rule | `src/scoring/{injection,shock,exclusion,death,wdiag,plan}.jl` | Observed-cohort injection modes, disturbance shock, cohort-level exclusion, death term, W diagnostics, `InjectionPlan` (`src/scoring/plan.jl:16`). |
| Simulation | `src/sim/fit_params.jl`, `src/sim/tier_acc.jl` | `fit_params` (`src/sim/fit_params.jl:131`), the tier accumulators (`src/sim/tier_acc.jl`). |
| Search algorithms | `src/Search.jl`, `src/search/*.jl` | Pure optimizers (module `Search`): `common.jl` (non-dominated sort, crowding, Pareto archive, (1+1)-CMA kernels, `AbstractSearch`), `LBSA`, `SA`, `MOLBSA`, `MOSA`, `CMAES`, `MOCMAES`, `IgelMOCMAES`, `CMAMAE`, `NSGA2`, `CCIgel`, `GDIgel`. |
| Search drivers | `src/drivers/*.jl` | One `parametrize_<Name>` per algorithm that wires the optimizer to `fit_params`; shared parts in `common.jl`, `mo_common.jl`, `search_loop.jl`. |
| I/O | `src/io/{writer,outputs,checkpoints,losses_db,plots}.jl` | Background checkpoint writer, `split_plots.csv`, JLD2 / DuckDB readers, `record_best_losses!`, plots. |
| Entry | `src/entry.jl` | `parametrize`, `run_from_yaml`, `main`, `julia_main`. |
| Analysis / export | `src/diagnostics.jl`, `src/spatial_sim.jl`, `src/landis_export.jl`, `src/species_composition.jl` | `simulate_and_test`, spatial simulation and LANDIS export, species composition. |
| Scenario engine | `src/climate_scenario.jl`, `src/scenario.jl`, `src/scenario_site.jl`, `src/scenario_volume.jl` | `ScenarioConfig`, `load_paramset`, `run_scenario`, climate scenarios, volume equations (see `docs/scenario_engine.md`). Served by `tools/pan_worker.jl`. |

## 2. Data flow of a run

```
YAML --load_config--> PanConfig --data_config--> Data.DataConfig --Data.prepare_parametrization_data--> splots (train) + val/test tables
   |                      |                                                          |
   |                      +--scoring_from_config--> ScoringOptions, SimBOptions      +--build_reference--> Reference (spdf, site years, injection cohorts, ref SiteSoA)
   |                                                                                   |
 run_from_yaml --parametrize--> plugin_config (PluginConfig) + make_param_space (ParamSpace) + LossParams / LossScales + InjectionPlan
                                   |
                                   +--parametrize_<driver>--> run_search_loop { ask -> fit_params (tier accumulator) -> tell } --> outputs/
```

1. **Config.** `run_from_yaml(yaml_path; overrides, exec, post_search_hook)` (`src/entry.jl:368`) reads the YAML with `load_config`
   (`src/config/schema.jl:623`): the key table is `SCHEMA_LIST` (`:38`, `KeySpec` at `:22`), unknown keys are an error with a
   spell-suggestion (`_unknown_key_error`), retired keys are listed in `RETIRED_KEYS` (`:310`), `${VAR}` / `$VAR` / `~` are expanded by
   `_expandenv` (`:338`; `PAN_DATA` and `PAN_OUT` default to `runs` / `runs/outputs`, `_ENV_FALLBACK`). The result is a `PanConfig`
   (`:587`, an `AbstractDict` with checked `getindex`). `config_docs()` (`:654`) prints the key table.
2. **Options.** `scoring_from_config` (`src/config/from_yaml.jl:159`) builds `ScoringOptions` (`src/config/options.jl:21`) and `SimBOptions`
   (`:51`); `exec_options_from_env` (`:105`) builds `ExecOptions` (`:84`: parallel mode, timing, validation switches, read once from
   `PAN_*` env variables); `search_options_from_config` (`:184`) builds `SearchOptions` (LBSA / SA / Igel / MO-CMA-ES knobs).
3. **Data.** `data_config(cfg)` (`src/config/from_yaml.jl:109`) is the only YAML to `Data.DataConfig` builder (`src/data/utils.jl:202`:
   selection, split, strata, tiering, disturbance parquet, `on_missing` policy). `Data.prepare_parametrization_data(dc)`
   (`src/data/utils.jl:915`) returns the train cohort table `splots`, `eco_list`, `species_list`, `eco_species_ids`, the val and test tables
   and a `Data.DataOutputs` (`:166`). `_write_split_plots` (`src/io/outputs.jl`) persists the exact split to `split_plots.csv` and the species
   categories to `species_map.csv`.
4. **Reference.** `build_reference` (`src/reference.jl:38`) turns one split into a `Reference` (`src/reference.jl:14`): smoothed observed age
   distributions (`reference_tables`, `src/reference.jl:25`, via `PU.smoothen_ref_years` and `Data.make_spdf_dict`), the site measurement
   years, the Sim B spinup cohorts, the injection cohorts, and the initial `SiteSoA` from `make_sites` (`src/sites.jl:64`).
   `prepare_from_cfg(cfg)` (`src/config/from_yaml.jl:173`) is the same data preparation for tools.
5. **Plugin and parameter space.** `plugin_config` (`src/config/plugin_config.jl:41`) builds the immutable `PluginConfig` (`BSTables` |
   `PnETContext` | `HBADContext`). `make_param_space` (`src/config/param_space.jl:36`) builds the `ParamSpace`: distributions, slot layout
   (`_build_slots`, `src/parametrize/slots.jl`), tying / pooling (`build_param_tying`, `:125`), data floors (`PU.with_floors`), CMA blocks
   (`cma_blocks`, `:54`). Searches work on a vector `u` (the slots); `u_to_params` and `params_to_u` convert.
6. **Scoring plan.** `build_injection_plan` (`src/scoring/plan.jl:44`) builds one immutable `InjectionPlan` per reference (injection
   dictionary, disturbance shock, exclusion plan); `compute_loss_scales` / `build_split_scales` give each reference its `LossScales`.
7. **Evaluation.** `fit_params` (`src/sim/fit_params.jl:131`) simulates the reference SoA for each seed (rep) and returns
   `[(SiteLoss, cohort rows, per-eco SiteLoss)]`. See section 3.
8. **Search.** `parametrize` (`src/entry.jl:8`) picks the driver from `search_mode` (`SEARCH_MODES`, `src/entry.jl:2`: `lbsa`, `sa`,
   `molbsa`, `cmaes`, `mocmaes`, `igelmo`, `cmame`, `ccigel`, `nsga2`, `gdigel`; `sobol` returns early; `plot_only*` are handled in
   `run_from_yaml`). Drivers share `MOEnv` / `mo_env` (`src/drivers/search_loop.jl:84`, `:115`), `run_candidate` / `candidate_fitness`
   (`:158`), `mo_val!` (`:218`) and the generation-loop wrapper `run_search_loop` (`:291`), which owns the writer task, the interrupt
   handling and the exit checkpoint. `igelmo` and `gdigel` implement the `AbstractSearch` interface (`ask`, `tell!`, `state`,
   `save_state`, `load_state`; `src/search/common.jl:24`, engines `IgelSearch` `src/search/IgelMOCMAES.jl:333`, `GDSearch`
   `src/search/GDIgel.jl:279`).
9. **Outputs** (written to `output_dir`, or `output_dir/fold_<k>` when `n_folds > 1`, `run_from_yaml`): `split_plots.csv`,
   `species_map.csv`, `pool_map.csv`; `best_params@<iter>.{jld2,json}`; `search_state@<iter>.jld2` plus the `search_state_latest.jld2`
   link (`save_exit_checkpoint`, `src/drivers/search_loop.jl:276`); `losses.duckdb` with `total_loss`, `ecoregion_loss`,
   `species_loss` (`open_losses_db`, `src/drivers/search_loop.jl:24`; rows by `record_best_losses!`, `src/io/losses_db.jl`); for the
   multi-objective drivers `metrics.csv`, `cv_val_cache.csv`, `best_val_params@<iter>`; gdigel adds `gd_*.csv`, `mo_archive*`, `solutions/`.
   Checkpoints are written by the background writer (`start_writer`, `start_mo_writer`, `src/io/writer.jl`).

## 3. `fit_params` and the tier accumulators

`fit_params` runs one evaluation. Per rep it calls `_fit_rep` (`src/sim/fit_params.jl:97`), which copies and reseeds the reference SoA
(`copy_and_reseed_soa`, `src/PanCore.jl:433`, or the in-place `reset_soa!`, `:451`, when the caller owns a work SoA) and calls
`_run_rep!` (`:53`), the function barrier that receives the concrete accumulator type. The accumulator comes from `make_tier_acc`
(`src/sim/tier_acc.jl:380`), the only place the runtime tier is looked at (`search_tier`, config key `tier`). Per simulated year the loop
does, in this order (`src/sim/fit_params.jl`, `_run_rep!`): plugin step, free disturbance, disturbance shock, pre-injection snapshot,
legacy-order injection, death / survivors filtering, W diagnostics, scoring at the sites measured that year (`_score_sites!`,
`src/sim/tier_acc.jl:410`, threaded with `@maybe_threads soa.parallel`), and the new-order injection of the observed cohorts.

| Tier | Accumulator | Scores |
|---|---|---|
| 1 | `Tier1Acc` | biomass-by-age bins per (eco, species) over all measurements |
| 2 | `Tier2Acc` | per-site bin biomasses per (eco, species, bin) |
| 3 | `Tier3Acc` | per-site Wasserstein (age CDF) + AGB loss, summed into per-thread per-eco `SiteLoss` (allocation-free kernel); the production Sim A tier |
| 4 | `Tier4Acc` | per-(eco, cycle) AGB-by-age histograms of the free simulation (Sim B) |
| 5 | `Tier5Acc` | tier 3 and tier 4 in one pass |

Sim B (free simulation with establishment) is a second `fit_params` call on `dual_b.ref_soa` with distinct seeds, combined with Sim A by
`SimBOptions.dual_mode` (`:off`, `:joint`, `:b`; `src/config/options.jl:51`).

## 4. Scoring rule and injection modes

The scoring rule is data, not code paths: `ScoringOptions` (`src/config/options.jl:21`) is built once per run and passed to
`build_injection_plan`, `fit_params` and the drivers. `fit_params` refuses an `InjectionPlan` built under different options
(`_check_plan_options`, `src/scoring/plan.jl:57`).

* **Injection mode** is a singleton type chosen once per call by `_inject_mode(override, replace, sync)` (`src/scoring/injection.jl:46`;
  `InjectMode` at `:41`): `InjAppend` (override off), `InjReplace` (wipe each measured site and rebuild it from the observed cohorts),
  `InjSync` (the cohort set tracks the data, matched survivors keep their simulated biomass; the production setting), `InjOverride`
  (overwrite the matching cohort's biomass). `_inject_observed_cohorts!` (`:246`) dispatches on it; `noise` jitters the injected biomass.
* **Order.** `loss_before_injection` (default true, resolved by `_resolve_loss_before_injection`, `src/config/from_yaml.jl:65`): score the
  simulated state before injecting the observed cohorts for the next interval. Together with `override` + `sync` (and not `replace`) this is the "shock order" (`shock_order`,
  `src/config/options.jl:62`), the only order in which the disturbance shock, the cohort exclusion plan and the death term apply.
* **Disturbance** (`ScoringOptions.disturbance`): `:off`, `:scale` (matched partially disturbed survivors scaled by `1 - drop`, cohort
  stays in the loss), `:exclude_overwrite` (overwrite with the observed reduced biomass and exclude the cohort at that year; production),
  `:exclude_noscale`. Cohorts fully removed by disturbance are removed from the simulation before scoring (`src/scoring/shock.jl`,
  data from `Data._plot_shocks!`); the cohort-level exclusion plan is `_ExclPlan` (`src/scoring/exclusion.jl:9`), which also excludes
  recruits at entry (`exclude_recruits_at_entry`).
* **Natural-death part** (`src/scoring/death.jl`): `w_scope` `:all` | `:survivors` and `death_term` `:none` | `:likelihood`
  (survive / die negative log-likelihood given `PROB_MORT`; `PROB_MORT` is not applied in the Sim A simulation then, `DEATH_PMIN` at
  `:8`).
* **Diagnostics**: `w_diag` (`:none`, `:stats`, `:nodeath`, `:oracle`; `WDiag`, `src/scoring/wdiag.jl:10`; used by `tools/w_decompose.jl`).
* **Loss knobs** (`LossParams`, `src/loss/types.jl:20`): hinge / softplus / normalisation / count-balance / rank weights; every knob of the
  run lives in this immutable struct, built in `parametrize` (`_loss_params`), plus per-reference `LossScales` (`:90`).

## 5. Plugin interface

A succession model is a type `<:AbstractPlugin` (`src/PanCore.jl`) that provides

1. its SoA layout: `PanCore.scalar_arrays(::Type{P}, n)` (one value per site) and `PanCore.csr_fields(::Type{P})` /
   `PanCore.csr_arrays(::Type{P}, nnz)` (variable-length per site, compressed sparse row), e.g. `src/plugins/PnET/plugin.jl:7`, `:25`;
2. the annual kernel `PanCore.process_plugin!(soa, ::Type{P}, year; ctx)` (e.g. `src/plugins/HBAD/plugin.jl:320`);
3. a parameter struct and its distributions (`src/plugins/*/params.jl`, `src/parametrize/{BiomassSuccession,PnET,HBAD}.jl`);
4. methods of the generics in `src/succession_api.jl` for its context type (`BSTables`, `PnETContext` `src/plugins/PnET/context.jl:15`,
   `HBADContext` `src/plugins/HBAD/context.jl:28`): `plugin_type`, `params_type`, `make_param_dists`, `generate_params`,
   `generate_eco_params`, `make_ctx`, `step!`, `spinup!`, `add_cohort!`, `plant!`, `plant_slot_needed`, `per_eco_groups`, `init_sites!`,
   `set_cohort_bio!`.

`SiteView{S}(soa, i)` (`src/PanCore.jl:57`) resolves `site.field` to the right array through an `@generated` `getproperty` /
`setproperty!`, so plugin kernels are written against fields and compile to direct array indexing. `succession_plugin_of`
(`src/succession_api.jl:10`) extracts the single non-`BaseSite` plugin from the SoA type, so the fit loop dispatches statically (no dynamic
dispatch in `fit_params`; `test/unit_hotpath.jl` checks type stability and zero allocations of the kernels). `PluginConfig{P,C,F}`
(`src/succession_api.jl:54`) bundles the context with the builder of plugin-specific site attributes (PnET: climate id, latitude, soil).

## 6. Determinism rules

Runs are reproducible for a fixed seed and independent of the thread count. The rules, each with the code that enforces it:

* **Per-site RNG.** `_make_sites` draws one `base_seed` from the run rng and seeds site `i` with `RNGType(hash((base_seed, i)))`
  (`src/sites.jl:89`, `:119`). `copy_and_reseed_soa` / `reset_soa!` (`src/PanCore.jl:433`, `:451`) reseed every site per rep from the rep seed.
  No RNG is shared between sites or taken from a per-thread RNG. `RNGType = Random.Xoshiro` (`src/Pan.jl`).
* **Fixed, common random numbers for scoring.** `mo_env` draws the rep seeds once (`_fixed_seeds`, `MOEnv.fixed_seeds`,
  `src/drivers/search_loop.jl:92`, `:136`); every training and validation evaluation uses those same seeds (`mo_val!(...; seeds=:fixed)`,
  `:218`), so the validation evaluation never draws from the run rng. Diagnostic
  simulations derive their own rng from the fixed seeds (`_diag_rng`, `src/drivers/common.jl:43`), so switching diagnostics on or off does
  not change the search trajectory.
* **Thread independence.** Site loops are `@maybe_threads soa.parallel` loops (`src/PanCore.jl:11`) whose body touches only site `i`
  and per-thread accumulators that are merged in a fixed order. The parallelism flag is a field of the SoA (`SiteSoA.parallel`,
  `src/PanCore.jl:45`), not process state: in candidate-parallel mode (igelmo, gdigel) the outer loop runs candidates on per-thread work SoAs
  (`thread_work_soa`, `src/drivers/search_loop.jl:165`) with the site loops serial.
* **Ordered data.** Every SQL query whose rows feed a float accumulation, an index assignment or a Dict has a total `ORDER BY` (several
  with `ORDER BY ALL`; the audit is in `WORKLOG.md`), and `_make_sites` sorts a plot's cohorts on a full key (age descending, eco-species id, biomass; `src/sites.jl`, `sort!(... [:age_calc, :eco_species_id, :agb_sum] ...)`), so the site
  and cohort order does not depend on DuckDB's parallel scan order. The scenario engine reads plots and cohorts in `CN` order
  (`test/unit_scenario_order.jl`).
* **Checkpoints continue exactly.** `load_state(T, save_state(s, path))` continues as `s` would (`AbstractSearch`, `src/search/common.jl:24`;
  `test/resume_search.jl`, `test/unit_search_sa.jl`).

The golden harness (`test/golden/`) pins all of this: 46 scenarios, one Julia process each with one thread, outputs compared bit for
bit (see `test/golden/README.md`).

## 7. Where each former global went

Before Step 3 the run state lived in about 156 module-level `Ref`s. None is left (only immutable tables, the log-once `_WARNED_ENV`
set in `src/config/schema.jl:323` and an atomic counter). The replacements:

| Former global | Now |
|---|---|
| `ACTIVE_SUCCESSION`, `SITE_ATTRS_FN` | `PluginConfig` (`src/succession_api.jl:54`), built by `plugin_config` (`src/config/plugin_config.jl:41`) and passed to drivers, `make_sites`, `fit_params` |
| Biomass Succession `SHADE_TOL_TABLE` / `MATURITY_TABLE` and other plugin tables | `BSTables` (`src/plugins/BiomassSuccession/BiomassSuccessionPlugin.jl:20`), carried in the context; `bs_tables_from_cfg` (`src/config/plugin_config.jl:12`) |
| HBAD `ensure_configured!` module state | `HBADContext` (`src/plugins/HBAD/context.jl:28`) with `check_context` (`:89`) raising an explicit error |
| PnET climate / constants / scratch pool | `PnETContext` (`src/plugins/PnET/context.jl:15`) owns climate, constants and the per-thread scratch |
| About 40 loss `Ref`s in `Parametrization`, `SCALES_LOCKED`, `CELL_NORM_FREEZE`, `T4_CELL_PLOTS` | `PU.LossParams` (`src/loss/types.jl:20`) and `PU.LossScales` per reference (`compute_loss_scales`, `build_split_scales`); `cell_plots` travel in `dual_b` |
| Scoring flags (`apply_scoring_flags!`), `_SHOCK_REGISTRY`, `_EXCL_CACHE` | `ScoringOptions`, `SimBOptions` (`src/config/options.jl`) and one `InjectionPlan` per reference (`src/scoring/plan.jl:16`) |
| Parameter distributions, tying, floors, CMA blocks | `PU.ParamSpace` (`src/parametrize/utils.jl:161`) |
| `PARALLEL_SITES`, `PARALLEL_MODE`, `PAN_TIMING`, `SKIP_VAL`, `NEWBEST_STATS`, `STORE_VAL_OBJ`, `CV_RESELECT` | `SiteSoA.parallel` and `ExecOptions` / `RunTimer` (`src/config/options.jl:84`, `:119`) |
| `POST_SEARCH_HOOK` | keyword `post_search_hook` of `run_from_yaml` / `parametrize` (`src/entry.jl`) |
| LBSA / Igel / MO-CMA-ES module knobs | `SearchOptions` (`src/config/options.jl:176`) |
| `Data.*` module state | `Data.DataConfig`, `Data.DataOutputs`, `Data.CycleOptions` (`src/data/utils.jl:202`, `:166`, `:178`) |
| Scenario engine lazy tables | `ScenarioConfig` and the lock-guarded `ScenarioCache` it owns (`src/scenario.jl:59`, `:31`) |

## 8. Entry points, tools, tests

* Run a config: `make yaml RUN=<config.yml>` (Makefile) = `Pan.run_from_yaml`; `Pan.main()` takes a `.yml` argument too. Build the
  fast-start images with `make sysimage` / `make exec` (`scripts/precompile_exec.jl`).
* Tools (`tools/`, status table in `docs/tools.md`); legacy tools are in `tools/legacy/`.
* Tests: `test/runtests.jl` (unit suites in-process, long scripts as subprocesses); the golden harness in `test/golden/`.

## 9. Design notes in `docs/`

`scenario_engine.md`, `scenario_campaign_plan.md` (scenario engine and worker protocol), `hbad_plugin.md`, `hbad2_plugin.md`, `hbad2_calibration.md`, `pnet_spec.md`,
`region4_pnet.md`, `volume_equations.md` (plugins and data), `mo_gdigel.md`, `cmaes_gdigel.md` (search), `gafl_calibration_log.md`,
`gafl_model_comparison.md`, `gafl_pnet_calibration.md`, `simAB_options_experiment.md` (calibration protocol and results),
`campaign_2026-10-14.md`, `treemap_donors.md`, `plan_deck_and_mo_gdigel.md`.
