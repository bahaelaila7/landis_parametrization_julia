# Shared machinery of the search drivers (Step 5a). Before this file every driver re-implemented, in 100-300 lines of its own, the same
# preamble (tier references, scales, fixed seeds, work SoAs, the train/val evaluators), the background writer + losses.duckdb DDL,
# the interrupt handling and the exit checkpoint. They now build a `MOEnv`, open a `SearchLoop` and run their generation loop inside
# `run_search_loop`; what stays in a driver is only what makes it a different algorithm (initial population, ask/evaluate/tell, restarts).
#
# RNG order is part of a driver's behaviour (common random numbers, checkpoints, golden): the builder draws in the order of the drivers it
# replaces (plot samples → fixed seeds → Sim B reference) and `val_dual_early` reproduces the one driver family (MOCMAES, MOLBSA) that
# built the validation Sim B before the val plot sample.

# ---- interrupt handling ------------------------------------------------------------------------------------------------------
# A Ctrl-C inside a @threads region (e.g. mid-resize in readjust_soa!) surfaces as a TaskFailedException wrapping an InterruptException,
# not a bare InterruptException: recurse through the task/composite wrappers to recognise it.
caused_by_interrupt(e) = e isa InterruptException ? true :
  e isa TaskFailedException ? any(en -> caused_by_interrupt(en.exception), Base.current_exceptions(e.task)) :
  e isa CompositeException ? any(caused_by_interrupt, e.exceptions) : false

# ---- losses.duckdb ------------------------------------------------------------------------------------------------------------
"""
    open_losses_db(output_dir; archive_size=true) -> (file, connection)

Open `<output_dir>/losses.duckdb` and create its three tables if missing. `total_loss` carries an `archive_size` column in the multi-objective
drivers only. The caller closes `file`.
"""
function open_losses_db(output_dir::AbstractString; archive_size::Bool=true)
  file = DuckDB.DB(joinpath(output_dir, "losses.duckdb"))
  db = DuckDB.connect(file)
  DuckDB.execute(db, "CREATE TABLE IF NOT EXISTS total_loss (iteration INTEGER, n_sites INTEGER, n_obs INTEGER, total_loss DOUBLE, " *
                     (archive_size ? "archive_size INTEGER, " : "") * "params_blob BLOB)")
  DuckDB.execute(db, "CREATE TABLE IF NOT EXISTS ecoregion_loss (iteration INTEGER, ecoregion VARCHAR, eco_num_sites INTEGER, eco_num_obs INTEGER, ecoregion_total_loss DOUBLE)")
  DuckDB.execute(db, "CREATE TABLE IF NOT EXISTS species_loss (iteration INTEGER, ecoregion VARCHAR, species VARCHAR, age_dist_loss DOUBLE, agb_loss DOUBLE)")
  return file, db
end

# ---- reference tables of the search tiers ---------------------------------------------------------------------------------------
# tier 4/5 and every dual run: per-(eco, cycle) age-bin AGB histogram of the observed cohorts.
function t4_reference(splots, spdf_plts, loss_params, eco_species_ids, eco_list; cycle_years, scoring::ScoringOptions)
  n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
  cycle_map, n_cycles = Data.build_cycle_map(splots; cycle_years=cycle_years, cycle=scoring.cycle)
  t4_ref = [[zeros(FloatType, length(eco_species_ids[eco_id]), n_bins) for _ in 1:n_cycles] for eco_id in eachindex(eco_list)]
  for ((plot_id, eco_id), year_dict) in spdf_plts, (sy, spdf_gt) in year_dict
    cyc = get(cycle_map, (Int(plot_id), Int(sy)), 0); cyc == 0 && continue
    for (sp_eco, rec) in spdf_gt.records; t4_ref[eco_id][cyc][sp_eco, :] .+= diff([0f0; rec.sp_age_cdf]) .* rec.sp_agb_sum; end
  end
  return t4_ref, cycle_map, n_cycles
end

"The per-tier reference tables of a run: `(t1_ref, t2_ref, t4_ref, cycle_map, n_cycles)` (tier 3 needs none)."
function tier_references(search_tier::Int, simb::SimBOptions, splots, spdf_plts, loss_params, eco_species_ids, eco_list; cycle_years, scoring::ScoringOptions)
  t4_ref = nothing; cycle_map = nothing; n_cycles = 0; t1_ref = nothing; t2_ref = nothing
  if search_tier == 1
    n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    t1_ref = [zeros(FloatType, length(eco_species_ids[eco_id]), n_bins) for eco_id in eachindex(eco_list)]
    for ((_, eco_id), year_dict) in spdf_plts, (_, spdf_gt) in year_dict, (sp_eco, rec) in spdf_gt.records
      t1_ref[eco_id][sp_eco, :] .+= diff([0f0; rec.sp_age_cdf]) .* rec.sp_agb_sum
    end
  elseif search_tier == 2
    n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    t2_ref_bins = [[FloatType[] for sp in 1:length(eco_species_ids[eco_id]), b in 1:n_bins] for eco_id in eachindex(eco_list)]
    t2_ref_total = [[FloatType[] for _ in 1:length(eco_species_ids[eco_id])] for eco_id in eachindex(eco_list)]
    for ((_, eco_id), year_dict) in spdf_plts, (_, spdf_gt) in year_dict, (sp_eco, rec) in spdf_gt.records
      bin_probs = diff([0f0; rec.sp_age_cdf])
      for b in 1:n_bins; push!(t2_ref_bins[eco_id][Int(sp_eco), b], bin_probs[b] * rec.sp_agb_sum); end
      push!(t2_ref_total[eco_id][Int(sp_eco)], rec.sp_agb_sum)
    end
    t2_ref = (bins=t2_ref_bins, total=t2_ref_total)
  elseif search_tier == 4 || search_tier == 5 || simb.dual_mode != :off
    t4_ref, cycle_map, n_cycles = t4_reference(splots, spdf_plts, loss_params, eco_species_ids, eco_list; cycle_years, scoring)
  end
  return (; t1_ref, t2_ref, t4_ref, cycle_map, n_cycles)
end

# ---- the evaluation environment of the multi-objective drivers --------------------------------------------------------------------
"""
    MOEnv

Everything the multi-objective drivers share about HOW a parameter set is evaluated: the train reference with its injection plan, tier
references, fixed seeds (common random numbers), Sim B reference and scales; reusable work SoAs (one per thread in candidate mode); and the
held-out validation evaluator `val_fit` (one function for every driver, see [`mo_val!`](@ref)).

Parallelism: the train evaluator runs sites serially exactly when `train_sites_parallel` is false (candidate-parallel generation loops).
Validation scoring of archive members is concurrent in candidate mode (`mo_gen_finalize!`), so the validation evaluator ALWAYS runs its
sites serially there (`val_sites_parallel`) and uses per-thread work SoAs.
"""
Base.@kwdef mutable struct MOEnv
  ref_soa::Any; plugin::PluginConfig; splots::Any; spdf_plts::Any; spinup_cohorts::Any; site_sim_years::Any
  species_list::Vector{String}; eco_list::Vector{String}; eco_species_ids::Vector{Vector{Int}}
  loss_params::Any; scoring::ScoringOptions; simb::SimBOptions; spinup::Bool; rng::Any; debug::Bool; search_tier::Int
  n_reps::Int; n_output_plots::Int; no_establishment::Bool; injection_cohorts::Any; cycle_years::Real; scale_seed::PU.LossScales; exec::ExecOptions
  n_species::Int; max_sim_year::Int
  injection::Any; sampled_ids::Any; emp_sample::Any
  t1_ref::Any = nothing; t2_ref::Any = nothing; t4_ref::Any = nothing; cycle_map::Any = nothing; n_cycles::Int = 0
  fixed_seeds::Any; dual_b::Any = nothing; scales::PU.LossScales
  work_soa::Any = nothing; work_soa_t::Any = nothing          # reusable working SoAs: one (serial) or one per thread (candidate mode)
  train_sites_parallel::Bool = true; timer::Any = nothing
  # ---- held-out validation split
  have_val::Bool = false
  val_splots::Any = nothing; val_ref_soa::Any = nothing; val_spdf_plts::Any = nothing; val_site_sim_years::Any = nothing
  val_spinup_cohorts::Any = nothing; val_injection_cohorts::Any = nothing
  inj_val::Any = nothing; sampled_ids_val::Any = Set{UIntType}(); emp_sample_val::Any = nothing
  val_dual_b0::Any = nothing; val_dual_b0_built::Bool = false   # Sim B of the val split, when built before mo_val! (val_dual_early)
  val_scales::Any = nothing; val_dual_b::Any = nothing
  val_tier::Int = 3
  val_t4_ref::Any = nothing; val_cycle_map::Any = nothing; val_n_cycles::Int = 0
  val_work::Any = nothing; val_work_t::Any = nothing; val_sites_parallel::Bool = true
  val_fit::Any = nothing
end

"""
    mo_env(; ...) -> MOEnv

Build the evaluation environment (the shared preamble of the multi-objective drivers). `train_sites_parallel=false` ⇔ the driver evaluates its
candidates concurrently. `val_dual_early=true` builds the validation Sim B before the val plot sample (RNG order of MOCMAES / MOLBSA); otherwise
[`mo_val!`](@ref) builds it, at the point of the driver's choosing.
"""
function mo_env(; ref_soa, plugin::PluginConfig, splots, spdf_plts, spinup_cohorts, site_sim_years, species_list, eco_list, eco_species_ids,
    loss_params, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, rng, debug::Bool, search_tier::Int, n_reps::Int, n_output_plots::Int,
    no_establishment::Bool, injection_cohorts, cycle_years, scale_seed::PU.LossScales, exec::ExecOptions,
    val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing,
    train_sites_parallel::Bool=true, val_dual_early::Bool=false)
  splots.sim_year .= Dates.value.(Dates.Day.(splots.measdate - splots.start_measdate)) ./ 365.25 .|> round .|> Int
  # Fixed plot sample chosen once at startup so progress is comparable across iterations
  all_plot_ids = UIntType.(unique(splots.plot_id))
  sampled_ids = _sample_plot_ids(all_plot_ids, n_output_plots, rng; injection_cohorts=injection_cohorts)
  emp_sample = n_output_plots > 0 ? _make_emp_df(splots, sampled_ids) : nothing
  n_species = length(species_list)
  max_sim_year = site_sim_years.sim_years .|> maximum |> maximum
  injection = build_injection_plan(injection_cohorts, ref_soa, eco_species_ids, loss_params, scoring)   # this reference's injection dict, shock and exclusions
  have_val = !isnothing(val_ref_soa) && !exec.skip_val   # ExecOptions.skip_val (PAN_SKIP_VAL) disables all held-out val re-sim during training
  inj_val = have_val ? build_injection_plan(val_injection_cohorts, val_ref_soa, eco_species_ids, loss_params, scoring) : nothing   # the val reference's own plan
  val_dual_b0 = (val_dual_early && have_val) ?
    _build_val_dual_b(val_splots, eco_species_ids, eco_list, val_spdf_plts, loss_params, cycle_years, val_injection_cohorts, val_spinup_cohorts, rng, no_establishment; scoring, simb, plugin=plugin) : nothing
  sampled_ids_val = (have_val && n_output_plots > 0) ? _sample_plot_ids(UIntType.(unique(val_splots.plot_id)), n_output_plots, rng; injection_cohorts=val_injection_cohorts) : Set{UIntType}()
  emp_sample_val = (have_val && n_output_plots > 0) ? _make_emp_df(val_splots, sampled_ids_val) : nothing
  refs = tier_references(search_tier, simb, splots, spdf_plts, loss_params, eco_species_ids, eco_list; cycle_years, scoring)
  # Common random numbers: fix the per-rep seed set once so every candidate is scored on the *same* stochastic realization.
  fixed_seeds = _fixed_seeds(rng, n_reps, scoring)
  dual_b0 = simb.dual_mode == :off ? nothing : _build_dual_b(splots, eco_species_ids, eco_list, spdf_plts, loss_params, refs.t4_ref, refs.cycle_map, refs.n_cycles, injection_cohorts, spinup_cohorts, rng, no_establishment; b_only=(simb.dual_mode == :b), scoring, simb, plugin=plugin)
  scales, dual_b = build_split_scales(loss_params, scale_seed, search_tier, spdf_plts, refs.t4_ref, eco_species_ids, n_species; dual_b=dual_b0, simb)   # the TRAIN reference scales (RANKW defined here)
  # ONE reusable working SoA per reference (reset_soa! reuses the buffers each eval instead of deep-copying the reference); candidate mode: one per thread
  cand = candidate_parallel(exec)
  return MOEnv(; ref_soa, plugin, splots, spdf_plts, spinup_cohorts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, spinup, rng, debug,
    search_tier, n_reps, n_output_plots, no_establishment, injection_cohorts, cycle_years, scale_seed, exec, n_species, max_sim_year, injection, sampled_ids, emp_sample,
    t1_ref=refs.t1_ref, t2_ref=refs.t2_ref, t4_ref=refs.t4_ref, cycle_map=refs.cycle_map, n_cycles=refs.n_cycles, fixed_seeds, dual_b, scales,
    work_soa=deepcopy(ref_soa), work_soa_t=(cand ? [deepcopy(ref_soa) for _ in 1:Threads.maxthreadid()] : nothing),
    train_sites_parallel, timer=run_timer(exec),
    have_val, val_splots, val_ref_soa, val_spdf_plts, val_site_sim_years, val_spinup_cohorts, val_injection_cohorts, inj_val, sampled_ids_val, emp_sample_val,
    val_dual_b0, val_dual_b0_built=val_dual_early)
end

"One training evaluation: the per-rep results of parameter set `p` (on the work SoA `ws`)."
run_candidate(env::MOEnv, p; ws=env.work_soa) =
  fit_params(env.ref_soa, p, env.max_sim_year, env.n_species, env.eco_species_ids, env.spdf_plts, env.site_sim_years, env.spinup, env.spinup_cohorts, env.loss_params;
    debug=env.debug, search_tier=env.search_tier, t1_ref=env.t1_ref, t2_ref=env.t2_ref, t4_ref=env.t4_ref, cycle_map=env.cycle_map, n_cycles=env.n_cycles, seeds=env.fixed_seeds,
    injection=env.injection, dual_b=env.dual_b, work_soa=ws, scales=env.scales, scoring=env.scoring, simb=env.simb, plugin=env.plugin, collect_cohorts=env.n_output_plots > 0,
    parallel_sites=env.train_sites_parallel, timer=(env.train_sites_parallel ? env.timer : nothing))

"Score a candidate's repetitions into one `MOFitness` (eco losses summed over reps): `(fitness, run_result, eco_losses)`."
function candidate_fitness(env::MOEnv, rep_results)
  run_result, eco_losses, _ = _agg_reps(rep_results, env.loss_params, env.scoring)
  objs = _mo_objectives(eco_losses, env.eco_species_ids, env.loss_params, env.scales, env.dual_b)
  MOLBSA.MOFitness(objs, _mo_aggregate(objs, run_result, env.loss_params)), run_result, eco_losses
end

"The work SoA for the current thread: the per-thread one in candidate mode, else the shared one."
@inline thread_work_soa(env::MOEnv) = env.work_soa_t === nothing ? env.work_soa : env.work_soa_t[Threads.threadid()]

# ---- validation ------------------------------------------------------------------------------------------------------------------
"""
    val_fit_params(x; ...) -> per-rep fit_params results of parameter set `x` on a validation split

The ONE call through which every driver simulates a held-out split (the multi-objective drivers via [`val_reps`](@ref), CMA-ES / LBSA directly).
"""
val_fit_params(x; val_ref_soa, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params,
    tier::Int=3, t4_ref=nothing, cycle_map=nothing, n_cycles::Int=0, injection, dual_b=nothing, seeds, work_soa=nothing, scales, scoring, simb, plugin,
    collect_cohorts::Bool=false, parallel_sites::Bool=true) =
  fit_params(val_ref_soa, x, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params;
    debug=false, search_tier=tier, t4_ref, cycle_map, n_cycles, injection, dual_b, seeds, work_soa, scales, scoring, simb, plugin, collect_cohorts, parallel_sites)

"""
    val_reps(env, x) -> per-rep fit_params results of parameter set `x` on the validation split

THE validation evaluation of the multi-objective drivers. Same reference, injection plan, scales and (when the run is dual) Sim B as the train
split's objective; the fixed common-random-number seeds of the train split (B29: the val evaluation never draws from the run rng);
the per-thread work SoA; and sites serial whenever candidates are evaluated concurrently (`env.val_sites_parallel`) so concurrent val scoring never
nests threads.
"""
function val_reps(env::MOEnv, x)
  ws = env.val_work_t === nothing ? env.val_work : env.val_work_t[Threads.threadid()]
  val_fit_params(x; val_ref_soa=env.val_ref_soa, max_sim_year=env.max_sim_year, n_species=env.n_species, eco_species_ids=env.eco_species_ids, val_spdf_plts=env.val_spdf_plts,
    val_site_sim_years=env.val_site_sim_years, spinup=env.spinup, val_spinup_cohorts=env.val_spinup_cohorts, loss_params=env.loss_params, tier=env.val_tier,
    t4_ref=env.val_t4_ref, cycle_map=env.val_cycle_map, n_cycles=env.val_n_cycles, injection=env.inj_val, dual_b=env.val_dual_b, seeds=env.fixed_seeds, work_soa=ws,
    scales=env.val_scales, scoring=env.scoring, simb=env.simb, plugin=env.plugin, collect_cohorts=env.n_output_plots > 0, parallel_sites=env.val_sites_parallel)
end

"`(run_result, cached_sites, eco_losses)` of `x` on the validation split, aggregated over the reps like train."
function val_eval(env::MOEnv, x)
  vreps = val_reps(env, x)
  v = _agg_reps(vreps, env.loss_params, env.scoring)
  return (v[1], vreps[v[3]][2], v[2])
end

"""
    val_sites_parallel(exec) -> Bool

Whether a validation evaluation may thread over sites. Validation scoring of archive members is concurrent in candidate mode (`mo_gen_finalize!`
runs `val_fit` of several members at once), so there each evaluation runs its sites serially: `soa.parallel` must be false inside a
candidate-parallel batch, else the site loops nest `@threads` regions inside the candidate-level ones.
"""
val_sites_parallel(exec::ExecOptions) = !candidate_parallel(exec)

"""
    mo_val!(env; dual=true, tier=3) -> env

Build the validation side of `env` (scales, Sim B, work buffers, `val_fit`). Called at the point of the driver's choosing because building the
val Sim B draws from the run rng (see `val_dual_early`). `dual`: the val objective includes Sim B in dual runs (as train does); `tier`: the tier the
val split is scored at (a higher tier needs the val split's own tier-4 tables).
"""
function mo_val!(env::MOEnv; dual::Bool=true, tier::Int=3, seeds::Symbol=:fixed)
  env.have_val || return env
  env.val_tier = tier
  if tier == 4 || tier == 5   # tier 4/5 need per-cycle bins built from the validation split
    env.val_t4_ref, env.val_cycle_map, env.val_n_cycles = t4_reference(env.val_splots, env.val_spdf_plts, env.loss_params, env.eco_species_ids, env.eco_list; cycle_years=env.cycle_years, scoring=env.scoring)
  end
  if dual
    val_dual_b0 = env.val_dual_b0_built ? env.val_dual_b0 :
      _build_val_dual_b(env.val_splots, env.eco_species_ids, env.eco_list, env.val_spdf_plts, env.loss_params, env.cycle_years, env.val_injection_cohorts, env.val_spinup_cohorts, env.rng, env.no_establishment; scoring=env.scoring, simb=env.simb, plugin=env.plugin)
    env.val_scales, env.val_dual_b = build_split_scales(env.loss_params, env.scale_seed, tier, env.val_spdf_plts, env.val_t4_ref, env.eco_species_ids, env.n_species; dual_b=val_dual_b0, train=env.scales, simb=env.simb)
  else
    env.val_scales = first(build_split_scales(env.loss_params, env.scale_seed, tier, env.val_spdf_plts, env.val_t4_ref, env.eco_species_ids, env.n_species; train=env.scales, simb=env.simb))
    env.val_dual_b = nothing
  end
  cand = candidate_parallel(env.exec)
  env.val_sites_parallel = val_sites_parallel(env.exec)
  env.val_work = deepcopy(env.val_ref_soa)
  env.val_work_t = cand ? [deepcopy(env.val_ref_soa) for _ in 1:Threads.maxthreadid()] : nothing   # per-thread buffers: val_fit runs concurrently across archive members
  env.val_fit = x -> val_eval(env, x)
  return env
end

# ---- the generation loop wrapper ----------------------------------------------------------------------------------------------------
"""
    SearchLoop

The I/O side of a running search: the background checkpoint/plot writer, `losses.duckdb` and, for the multi-objective drivers, the per-generation
bookkeeping `MOGenIO` (metrics.csv, val cache, best_val_params). Opened by [`mo_loop_open`](@ref) (multi-objective) or [`scalar_loop_open`](@ref)
(CMA-ES, LBSA), driven by [`run_search_loop`](@ref).
"""
struct SearchLoop
  output_dir::String
  writer_ch::Any
  writer_task::Any
  losses_file::Any
  losses_db::Any
  gio::Union{Nothing,MOGenIO}
end

"Start the writer, open `losses.duckdb` and the generation bookkeeping for `state` (the fully built search state)."
function mo_loop_open(env::MOEnv, state; output_dir::AbstractString, resuming::Bool)
  writer_ch, writer_task = start_mo_writer(typeof(state), output_dir)
  losses_file, losses_db = open_losses_db(output_dir)
  gio = mo_gen_open(state; output_dir=output_dir, writer_ch=writer_ch, losses_db=losses_db, resuming=resuming, splots=env.splots, eco_list=env.eco_list, species_list=env.species_list,
    eco_species_ids=env.eco_species_ids, n_species=env.n_species, site_sim_years=env.site_sim_years, loss_params=env.loss_params, no_establishment=env.no_establishment,
    n_reps=env.n_reps, rng=env.rng, have_val=env.have_val, n_output_plots=env.n_output_plots, sampled_ids=env.sampled_ids, sampled_ids_val=env.sampled_ids_val,
    emp_sample=env.emp_sample, emp_sample_val=env.emp_sample_val, val_fit=env.val_fit, val_scales=env.val_scales, val_dual_b=env.val_dual_b, plugin=env.plugin, exec=env.exec, timer=env.timer)
  return SearchLoop(String(output_dir), writer_ch, writer_task, losses_file, losses_db, gio)
end

"Start the writer and open `losses.duckdb` for a single-objective driver (CMA-ES, LBSA): `(loop, losses_db)`."
function scalar_loop_open(state; output_dir::AbstractString)
  writer_ch, writer_task = start_writer(typeof(state), output_dir)
  losses_file, losses_db = open_losses_db(output_dir; archive_size=false)
  return SearchLoop(String(output_dir), writer_ch, writer_task, losses_file, losses_db, nothing), losses_db
end

"The default exit checkpoint: `search_state@i.jld2` and the `search_state_latest.jld2` link."
function save_exit_checkpoint(state, output_dir::AbstractString)
  mkpath(output_dir); fname = "search_state@$(state.i).jld2"
  JLD2.save_object(joinpath(output_dir, fname), state)
  link_path = joinpath(output_dir, "search_state_latest.jld2"); islink(link_path) && rm(link_path); symlink(fname, link_path)
  @info "Search state saved @ $(state.i)"
end

"""
    run_search_loop(body, loop; get_state, save_exit=..., close_extra=..., unit="gen")

Run the driver's generation loop `body()` with the shared life cycle: a user interrupt (also when wrapped by a threaded region) ends the
loop cleanly, any other exception propagates; in every case the writer is drained and stopped, `losses.duckdb` and metrics.csv are closed
(`close_extra()` closes driver-owned files), the exit checkpoint is written (`save_exit()`, default: `get_state()` as `search_state@i.jld2`)
and `after_exit()` runs (end-of-run evaluations; it runs after an interrupt or an error as well).
"""
function run_search_loop(body, loop::SearchLoop; get_state, save_exit=nothing, close_extra=() -> nothing, after_exit=() -> nothing, unit::AbstractString="gen")
  try
    body()
  catch e
    caused_by_interrupt(e) ? @info("Search interrupted by user @ $unit $(get_state().i); finalizing checkpoint…") : rethrow()
  finally
    stop_writer(loop.writer_ch, loop.writer_task); close(loop.losses_file); loop.gio === nothing || try; close(loop.gio.metrics_io); catch; end
    try; close_extra(); catch; end
    try
      save_exit === nothing ? save_exit_checkpoint(get_state(), loop.output_dir) : save_exit()
    catch e; @error "Failed to save search state on exit" exception = (e, catch_backtrace()); end
    after_exit()
  end
  return nothing
end
