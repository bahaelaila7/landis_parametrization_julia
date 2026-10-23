# Parse the data-prep inputs from a loaded config (`load_config`) into a NamedTuple whose names match `parametrize`'s kwargs.
function _cfg_data_inputs(cfg0::AbstractDict)
  cfg = _as_config(cfg0)
  # Plots to DROP from the loaded set (outlier exclusion): the inline `exclude_plots:` list and/or the rows of an
  # `exclude_plots_csv:` file with the four plot columns; merged (inline first).
  exclude_plots = copy(cfg["exclude_plots"])
  if cfg["exclude_plots_csv"] !== nothing
    for r in eachrow(CSV.read(cfg["exclude_plots_csv"], DataFrame))
      push!(exclude_plots, (Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot)))
    end
  end
  return (
    cohorts_db_path=cfg["cohorts_db_path"],   # (the FIADB env override is applied by load_config)
    filter_eco_field=cfg["filter_eco_field"],
    eco_field=cfg["eco_field"],
    tablename=cfg["tablename"],
    filter_ecos=cfg["filter_ecos"],
    filter_plots=cfg["filter_plots"],
    exclude_plots=exclude_plots,
    filter_species=cfg["filter_species"],
    skip_disturbances=cfg["skip_disturbances"],
    spinup=cfg["spinup"],
    by_subplot=cfg["by_subplot"],
    val_frac=cfg["val_frac"],
    split_seed=cfg["split_seed"],
    n_folds=cfg["n_folds"],
    fold_index=cfg["fold_index"],
    test_frac=cfg["test_frac"],
    split_csv=cfg["split_csv"],
    min_trees=cfg["min_trees"],
    min_agb_frac=cfg["min_agb_frac"],
    stratify_eco_mixed=cfg["stratify_eco_mixed"],
    single_ecoregion=cfg["single_ecoregion"],
    stratify_landuse=cfg["stratify_landuse"],
    site_class_strata=cfg["site_class_strata"],
    siteclass_hi_max=cfg["siteclass_hi_max"],
    siteclass_scheme=cfg["siteclass_scheme"],
    shade_tier_csv=cfg["shade_tier_csv"],
    filter_extent=cfg["filter_extent"],
    filter_states=cfg["filter_states"],
    siteclass_l3s=cfg["siteclass_l3s"],
    max_exact=cfg["max_exact"], max_group=cfg["max_group"],
    drop_phantom=cfg["drop_phantom"],
    train_cap=cfg["train_cap"],
    train_cap_k=cfg["train_cap_k"], train_cap_min=cfg["train_cap_min"],
  )
end

# yaml `dual_mode` → :off | :joint | :b. Accepts Bool (true = joint, the pre-June spelling) and strings in any case
# ("off"/"false", "joint"/"true", "b"). The single parser for every reader (run_from_yaml, the disturbance-shock setup, tools).
function _parse_dual_mode(v)::Symbol
  v === true && return :joint
  (v === false || v === nothing) && return :off
  s = lowercase(strip(String(v)))
  s in ("off", "false") && return :off
  s in ("joint", "true") && return :joint
  s == "b" && return :b
  error("dual_mode must be off | joint | b (or true/false), got $(repr(v))")
end

# yaml `loss_before_injection` (B14). The 2026-10-15 scoring rule (shock → score → sync injection) is defined for sync
# injection only (override_injection + override_injection_sync, not replace). There the default is true; for the other
# injection modes (append, replace) the default is the legacy order (false) and an explicit `true` is an error: those modes
# used to score pre-injection silently, with the species-level exclusion and no recruit exclusion. Without injection the
# flag has no effect. `g(key, default)` reads the already-coerced config values (called by `load_config`).
function _resolve_loss_before_injection(g)::Bool
  v = g("loss_before_injection", nothing)
  inj = Bool(g("override_injection", false))
  rule = inj && Bool(g("override_injection_sync", false)) && !Bool(g("override_injection_replace", false))
  v === nothing && return !inj || rule
  b = Bool(v)
  b && inj && !rule && error("loss_before_injection: true needs override_injection_sync (and not override_injection_replace); " *
                             "the other injection modes score in the legacy order — set loss_before_injection: false or omit it")
  return b
end

# Disturbance-shock data source (yaml `disturbance_full_removals: <parquet>`, default data/cohort_disturbance_full.parquet in the
# repo if present). Active only with the new scoring order + sync injection; otherwise "" (legacy behaviour byte-identical: no
# partial-drop attach, no shock rows). Pure: returns the path, sets nothing.
function _disturbance_shock_path(cfg0::AbstractDict)
  cfg = _as_config(cfg0)
  lbi = cfg["loss_before_injection"]   # resolved by load_config (_resolve_loss_before_injection)
  dual = cfg["dual_mode"] !== :off
  # Sim A needs sync injection; Sim B (dual mode / simB_disturb_only) uses the full removals + partial drops on its own
  on = lbi && ((cfg["override_injection"] && cfg["override_injection_sync"]) || dual || cfg["simB_disturb_only"])
  explicit = cfg["disturbance_full_removals"]   # nothing = default parquet | "" = disabled | path
  default = joinpath(dirname(dirname(@__DIR__)), "data", "cohort_disturbance_full.parquet")
  path = ""
  if on
    if explicit === nothing
      isfile(default) ? (path = default) : @warn "override_injection_sync with loss_before_injection: no disturbance data (set `disturbance_full_removals:` or build data/cohort_disturbance_full.parquet with tools/export_cohort_disturbance.py) — fully-removed cohorts are NOT shocked before scoring"
    elseif explicit == ""
      path = ""
    else
      path = explicit
      isfile(path) || error("disturbance_full_removals: $(path) not found (set `disturbance_full_removals: \"\"` to run without disturbance shocks)")   # used to warn and silently disable the shocks
    end
  end
  path
end

"""
    data_config(cfg; split_seed=nothing) -> Data.DataConfig

The ONE place a loaded config becomes the data-preparation inputs of a run: the selection / split / strata / tiering keys (`_cfg_data_inputs`:
`\${VAR}`/`~` expansion, the `FIADB` env override, `exclude_plots` + `exclude_plots_csv`), the disturbance-shock parquet and `data_on_missing`.
`split_seed` replaces the config's `split_seed` (per-fold scripts share the shuffle of the run they re-derive). Pure.
`run_from_yaml` passes the result to `parametrize(data=...)`; the analysis scripts get the same struct from `prepare_from_cfg`.
"""
function data_config(cfg0::AbstractDict; split_seed=nothing)::Data.DataConfig
  cfg = _as_config(cfg0)
  din = _cfg_data_inputs(cfg)
  return Data.DataConfig(; din...,
    split_seed=split_seed === nothing ? din.split_seed : Int(split_seed),
    disturbance_path=_disturbance_shock_path(cfg),
    on_missing=Symbol(cfg["data_on_missing"]))
end

"""
    scoring_options(cfg) -> ScoringOptions

The scoring rule of a run, read from a loaded config: the injection mode (`override_injection[_replace|_sync|_noise|_disturbance]`),
`loss_before_injection`, `exclude_recruits_at_entry`, `w_scope`, `death_term` and the initial-biomass perturbation (`init_perturb_frac/_cap`).
Pure: it touches no module state.
"""
function scoring_options(cfg0::AbstractDict)::ScoringOptions
  cfg = _as_config(cfg0)
  w_scope = cfg["w_scope"]; w_scope in (:all, :survivors) || error("w_scope must be all | survivors")
  death_term = cfg["death_term"]; death_term in (:none, :likelihood) || error("death_term must be none | likelihood")
  return ScoringOptions(; override=cfg["override_injection"], replace=cfg["override_injection_replace"], sync=cfg["override_injection_sync"],
    noise=cfg["override_injection_noise"], disturbance=cfg["override_injection_disturbance"],
    loss_before_injection=cfg["loss_before_injection"], exclude_recruits_at_entry=cfg["exclude_recruits_at_entry"],
    w_scope=w_scope, death_term=death_term,
    init_perturb_frac=cfg["init_perturb_frac"], init_perturb_cap=FloatType(cfg["init_perturb_cap"]),
    cycle=Data.CycleOptions(use_fia_cycle=cfg["fia_cycle"], merge=cfg["fia_cycle_merge"]))
end

"""
    simb_options(cfg) -> SimBOptions

Sim B / dual-mode options of a run, read from a loaded config. `t4_primary` is true for a dual_mode `b` run and for a tier-4 run without a dual.
Pure: it touches no module state.
"""
function simb_options(cfg0::AbstractDict)::SimBOptions
  cfg = _as_config(cfg0)
  dual = cfg["dual_mode"]
  return SimBOptions(; dual_mode=dual, tier_b=cfg["tier_b"], t4_primary=(dual === :b || (dual === :off && cfg["tier"] == 4)),
    simb_spinup=cfg["simB_spinup"], simb_disturb_only=cfg["simB_disturb_only"],
    t4_per_plot_mean=cfg["simB_per_plot_mean"], t4_mean_over_reps=cfg["simB_rep_mean"])
end

"""
    scoring_from_config(cfg) -> (scoring::ScoringOptions, simb::SimBOptions)

The ONE place the scoring and Sim B options are built. `run_from_yaml` and the analysis tools (tost, scatter, plots) all call it, so a tool
scores exactly the way the run was trained; the options are then passed explicitly to `parametrize`, `fit_params` and the helpers.

The data-preparation inputs live in `data_config`, the parameter-space switches in `param_space_options`.
"""
function scoring_from_config(cfg0::AbstractDict)
  cfg = _as_config(cfg0)
  scoring = scoring_options(cfg)
  simb = simb_options(cfg)
  return scoring, simb
end

"""
    prepare_from_cfg(cfg::AbstractDict; split_rng_seed_override=nothing, kw...)

`Data.prepare_parametrization_data(data_config(cfg; split_seed=split_rng_seed_override))` — EXACTLY the data the training path prepares (same struct) —
returns `(splots, eco_list, species_list, eco_species_ids, splots_val, splots_test, outs::Data.DataOutputs)`. Extra `kw` replace individual `DataConfig`
fields (e.g. `fold_index=k`); an unknown field is an error.
"""
function prepare_from_cfg(cfg::AbstractDict; split_rng_seed_override=nothing, kw...)
  dc = data_config(cfg; split_seed=split_rng_seed_override)
  return Data.prepare_parametrization_data(isempty(kw) ? dc : Data.with_data(dc; kw...))
end
