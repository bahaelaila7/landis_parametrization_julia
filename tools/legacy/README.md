# Legacy tools

These tools are not part of the current workflows (the 2026-10 GA+FL / Region 4 campaign: igelmo / gdigel calibration, TOST
selection, scenario engine). They moved here from `tools/` in Step 8 and are NOT maintained against the current API. The last tree in
which they ran (as far as they ran then) is the tag `refactor-step7`; the thesis-era versions are at `thesis-2026-07`:

    git show refactor-step7:tools/<name>          # or: git worktree add <dir> refactor-step7

Edited when moved: the `include(.../src/FVS.jl)` lines of the FVS tools now point at `tools/legacy/FVS.jl` (the file moved with them).
Other relative paths (`@__DIR__/..`, `runs/`, `results/`) were not touched; run them from a checkout of the tag.
`tools/check_tool_api.jl` can still statically check them: `./julia_gdal.sh --project=. tools/check_tool_api.jl tools/legacy/<file>.jl`.

| Tool | Why legacy |
|---|---|
| `cv_aggregate.jl`, `cv_convergence.jl`, `cv_designate.jl`, `cv_tost.jl`, `cv_two_fronts.jl`, `cv_two_fronts_val.jl`, `sweep_fold_percentiles.jl`, `sweep_fold_percentiles_val.jl`, `build_val_fronts.jl`, `demo_front_area.jl`, `cv_analysis.sh`, `run_cv.sh` | Thesis (2026-07) 5-fold MO-CMA-ES cross-validation chain: front aggregation over folds and `cv_reselect` re-scoring of MO-CMA-ES archives. The 2026-10 campaign cross-validates with `tools/gafl_cv.sh` + gdigel + `tools/tost_select.jl`. (`DEADCODE.md` D2.) |
| `extract_p101.jl`, `extract_candidates.sh`, `front_positions.sh`, `p101_pipeline.sh`, `run_p101_candidates.sh`, `resume_master.sh`, `run_853_chain.sh` | The p101 front-position pipeline and the 8.5.3 production chain of the thesis runs. Superseded by `tools/select_simB_seeds.jl` and the configs in `configs/region4`. |
| `sanity_3cand.sh`, `sanity_bestagg.sh`, `sanity_preinject.sh`, `sanity_scatter_sc.sh` | One-off sanity runs of the thesis Sim A scoring (pre-injection state, best aggregate); the scoring rule changed on 2026-10-15 and the golden harness now covers these paths. |
| `fvs_benchmark.jl`, `fvs_bsdimax_diag.jl`, `fvs_compare100.jl`, `fvs_loss.jl`, `fvs_loss_validate.jl`, `fvs_pan_regimpute_compare.jl`, `fvs_regen_trace.jl`, `fvs_strata_regimpute.jl`, `compare_fvs_pan.jl`, `compare_fvs_pan_rescale.jl`, `test_fvs.jl`, `test_fvs_spatial.jl`, `run_test_fvs.sh`, `run_test_fvs_spatial.sh`, `FVS.jl`, `regimpute/` | FVS (Forest Vegetation Simulator) comparison and forward-simulator experiments. `FVS.jl` (formerly `src/FVS.jl`) is not loaded by Pan; it needs the FVS executables and the REGIMPUTE keyword files in `regimpute/`. |
| `pan_project100.jl`, `pan_project100_autoes.jl` | 100-year projections of the thesis calibrated sets (pre-scenario-engine); `tools/compare_param_sets.jl` and the scenario engine replace them. |
| `plot_spinup_process.jl` | Thesis spinup-process figure (writes `spinup_process*.png`); reads the YAML raw (`YAML.load_file`, no `${PAN_DATA}` expansion), so it fails on the current configs (found by the Step 8 run check: `cohort database not found: ${PAN_DATA}/slim_r4.duckdb`). The `capture` argument of `BiomassSuccessionPlugin.spinup_cohorts!` it used is still in the plugin. |
| `extract_archive_seeds.jl` | Hard-coded paths to one July run (`runs/fl5_l4cover_mocmaes_Aonly_Sglobal_ipop_v2_outputs`); `tools/select_simB_seeds.jl` is the general seed extractor and writes the same `sobol_results` schema. |
| `plot_hinge_softplus.jl`, `plot_wasserstein_illustration.jl`, `plot_wasserstein_one_plot_simA.jl`, `plot_wasserstein_simA_diverge.jl`, `plot_simB_cohorts_spinup.jl`, `split_counts_latex.jl`, `archive_export.jl`, `rankw_multiplier.jl` and the `*.png` figures | Figure / table generators for the thesis and deck; outputs are the committed PNGs. `rankw_multiplier.jl` is an exploration of the pre-RANKW-mode weight. |
| `analyze.sbatch`, `gen_sobol.sbatch`, `warmup.sbatch` | Cluster batch scripts of the thesis runs (orphans: no current script or doc calls them). `tools/submit.sbatch` is the generic current one. |
