# Simulated-annealing driver (Step 5b rebuild; B22). Classic SA with a geometric cooling schedule on the scalar Sim-A loss, built on the shared
# machinery like the other drivers: `mo_env` (injection plan, tier references, fixed seeds, scales, Sim B), `run_candidate` (= fit_params),
# the one validation evaluation (`mo_val!` / `val_eval`, fixed common-random-number seeds), `scalar_loop_open` / `run_search_loop` (writer,
# losses.duckdb, interrupt handling, exit checkpoint) and checkpoint/resume (`resume_from`: the state carries the temperature, the counters and
# the run rng, so a resumed run continues exactly like an uninterrupted one). Proposals mutate the CURRENT point (the retired driver mutated its
# last proposal, accepted or not, which made it a random walk). Score = the LBSA score (cell-normalised aggregate under CELL_NORM, else the raw
# scalar). Knobs: yaml `sa_*` (SAOptions).
function parametrize_SA(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=3, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, scale_seed::PU.LossScales=PU.LossScales(), exec::ExecOptions=ExecOptions(), search::SearchOptions=SearchOptions())
  search_tier in (1, 2, 3, 4, 5) || error("parametrize_SA: unknown search_tier=$search_tier (expected 1, 2, 3, 4, or 5)")
  env = mo_env(; ref_soa, plugin, splots, spdf_plts, spinup_cohorts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, spinup, rng, debug,
    search_tier, n_reps, n_output_plots, no_establishment, injection_cohorts, cycle_years, scale_seed, exec,
    val_splots, val_ref_soa, val_spdf_plts, val_site_sim_years, val_spinup_cohorts, val_injection_cohorts)
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  n_species = env.n_species
  # the SAME scalar as LBSA's: the cell-normalised aggregate under CELL_NORM, else the raw total
  _score(rr, el) = loss_params.cell_norm ? Float64(sum(_mo_objectives(el, eco_species_ids, loss_params, env.scales, env.dual_b))) : convert(Float64, PU.get_total_loss(rr, loss_params))
  function _eval(p)
    rep_results = run_candidate(env, p)
    run_result, eco_losses, _ = _agg_reps(rep_results, loss_params, scoring)
    return _score(run_result, eco_losses), run_result, eco_losses, rep_results
  end
  sa = search.sa
  if isnothing(resume_from)
    bio_params = if !isnothing(start_from)
      @info "Seeding initial candidate from $start_from (fresh search_state)"
      p = _load_params_from_path(start_from, plugin.ctx)
      (p.SPECIES_LIST == species_list && p.ECO_LIST == eco_list) ||
        error("start_from params are incompatible with this run: their SPECIES_LIST/ECO_LIST differ from the loaded data (e.g. different stratify_eco_mixed, species tiering, or eco/plot filters). Seeding requires matching ecoregions and species.")
      p
    elseif !isempty(_sobol_cands)
      @info "Using Sobol candidate 1/$(length(_sobol_cands)) as initial point"
      _sobol_cands[1]
    else
      generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    end
    f0 = _eval(bio_params)[1]
    cur = SA.SACandidate(bio_params, f0)
    search_state = SA.SAState(cur, cur, rng; max_iter=TRIALS, initial_t=sa.initial_t, t=sa.initial_t, min_t=sa.min_t, alpha=sa.alpha, acceptance_rule=sa.acceptance,
                              reheat_prob_threshold=sa.reheat_prob_threshold, reheat_after_iters=sa.reheat_after_iters)
    push!(search_state.best_iterations, (0, f0, f0))
  else
    @info "Resuming from $resume_from"
    search_state = JLD2.load_object(resume_from)
    search_state.max_iter = TRIALS
    rng = search_state.rng   # the checkpoint's rng IS the run's rng from here on (as in the uninterrupted run, where state.rng === rng)
  end
  if TRIALS < 1 || SA.is_search_over(search_state)
    return search_state
  end

  mo_val!(env)
  loop, losses_db = scalar_loop_open(search_state; output_dir)
  writer_ch = loop.writer_ch
  run_search_loop(loop; get_state=() -> search_state, unit="trial") do
    TProgress.@track for trial in (search_state.i+1):TRIALS
      bio_params, _ = PU.mutate_params(search_state.current.x, space; rng=rng)
      next_loss, run_result, eco_losses, rep_results = _eval(bio_params)
      is_new_best = SA.search_cmp!(SA.SACandidate(bio_params, next_loss), search_state)
      val_sim_sample = nothing
      if is_new_best
        iter = search_state.best_iteration
        total = convert(Float64, search_state.best.fx)
        @info "New best @ $iter | loss=$total"
        if env.have_val
          try
            val_result = env.val_fit(search_state.best.x)
            @info "Val loss @ $iter | loss=$(convert(Float64, PU.get_total_loss(val_result[1], loss_params)))"
            val_sim_sample = n_output_plots > 0 ? _filter_cached_to_df(val_result[2], env.sampled_ids_val) : nothing
          catch e
            @warn "val fit_params failed" exception = (e, catch_backtrace())
          end
        end
        record_best_losses!(losses_db, iter, total, run_result, eco_losses, search_state.best.x; eco_list, species_list, n_species, loss_params, print_breakdown=true)
      end
      if is_new_best || search_state.i % 50 == 0
        cached_sites_state = _median_rep_cached(rep_results, loss_params, scoring)
        sim_sample = (is_new_best && n_output_plots > 0) ? _filter_cached_to_df(cached_sites_state, env.sampled_ids) : nothing
        put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state), splots, _cohorts_df(cached_sites_state), env.emp_sample, sim_sample, is_new_best ? env.emp_sample_val : nothing, val_sim_sample))
      end
      SA.search_update_rule!(search_state) && break
    end
  end
  return search_state
end
