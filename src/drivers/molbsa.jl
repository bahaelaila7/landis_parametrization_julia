# Multi-objective LBSA driver. Mirrors parametrize_LBSA, but scores each candidate
# by the per-(eco,species,{w,agb}) loss vector and compares candidates by net
# objective win-count (MOLBSA.mo_delta) rather than by the scalar total loss.
# All tiers (1, 2, 3, 4) expose the per-ecoregion breakdown that the objective
# vector is built from.
function parametrize_MOLBSA(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, seed_archive_from::Union{Nothing,String}=nothing, scale_seed::PU.LossScales=PU.LossScales(), exec::ExecOptions=ExecOptions(), search::SearchOptions=SearchOptions())
  search_tier in (1, 2, 3, 4, 5) || error("parametrize_MOLBSA: unknown search_tier=$search_tier (expected 1, 2, 3, 4, or 5)")
  env = mo_env(; ref_soa, plugin, splots, spdf_plts, spinup_cohorts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, spinup, rng, debug,
    search_tier, n_reps, n_output_plots, no_establishment, injection_cohorts, cycle_years, scale_seed, exec,
    val_splots, val_ref_soa, val_spdf_plts, val_site_sim_years, val_spinup_cohorts, val_injection_cohorts,
    val_dual_early=true)   # this driver built the validation Sim B before the val plot sample (RNG order)
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  n_species = env.n_species
  _run(p; ws=env.work_soa) = run_candidate(env, p; ws)
  _fitness(rep_results) = candidate_fitness(env, rep_results)

  if isnothing(resume_from) && !isnothing(seed_archive_from)
    # Local-refinement mode: pour a prior run's (e.g. CMA-MAE) sorted archive into MOLBSA's Pareto archive,
    # then let MOLBSA refine it locally. The CMA-MAE state's .archive is Vector{MOLBSA.MOCandidate} already.
    @info "Seeding MOLBSA archive from $seed_archive_from (local refinement)"
    seed_st = JLD2.load_object(seed_archive_from)
    seed_arch = hasproperty(seed_st, :archive) ? seed_st.archive : MOLBSA.MOCandidate[]
    rep0 = hasproperty(seed_st, :representative) ? seed_st.representative.x : seed_st.best.x
    (rep0.SPECIES_LIST == species_list && rep0.ECO_LIST == eco_list) ||
      error("seed_archive_from incompatible: its SPECIES_LIST/ECO_LIST differ from the loaded data.")
    bio_params = rep0
    fx0, _, _ = _fitness(_run(bio_params))
    cur = MOLBSA.MOCandidate(bio_params, fx0)
    search_state = MOLBSA.MOLBSAState(cur, cur, rng; max_iter=TRIALS, archive_cap=archive_cap)
    for c in seed_arch
      MOLBSA.update_archive!(search_state, c)
    end
    @info "  seeded $(length(seed_arch)) elites → $(length(search_state.archive)) non-dominated (cap $archive_cap); rep agg=$(round(convert(Float64, cur.fx.aggregate), digits=2))"
    search_state.sobol_cand_idx = 2
  elseif isnothing(resume_from)
    bio_params = if !isnothing(start_from)
      @info "Seeding initial candidate from $start_from (fresh search_state)"
      p = _load_params_from_path(start_from, plugin.ctx)
      (p.SPECIES_LIST == species_list && p.ECO_LIST == eco_list) ||
        error("start_from params are incompatible with this run: their SPECIES_LIST/ECO_LIST differ from the loaded data.")
      p
    elseif !isempty(_sobol_cands)
      @info "Using Sobol candidate 1/$(length(_sobol_cands)) as initial point"
      _sobol_cands[1]
    else
      generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    end
    fx0, _, _ = _fitness(_run(bio_params))
    cur = MOLBSA.MOCandidate(bio_params, fx0)
    search_state = MOLBSA.MOLBSAState(cur, cur, rng; max_iter=TRIALS, archive_cap=archive_cap)
    search_state.sobol_cand_idx = 2
  else
    @info "Resuming from $resume_from"
    search_state = JLD2.load_object(resume_from)
    search_state.max_iter = TRIALS
    bio_params = search_state.current.x
    force_restart_from_random && (search_state._should_restart = true)
  end
  if TRIALS < 1 || MOLBSA.is_search_over(search_state)
    return search_state
  end
  next_candidate() =
    if search_state.sobol_cand_idx <= length(_sobol_cands)
      p = _sobol_cands[search_state.sobol_cand_idx]
      search_state.sobol_cand_idx += 1
      p
    else
      generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    end

  mo_val!(env)
  loop = mo_loop_open(env, search_state; output_dir, resuming=!isnothing(resume_from))
  gio = loop.gio

  run_search_loop(loop; get_state=() -> search_state, unit="trial") do
      TProgress.@track for trial in (search_state.i+1):TRIALS
        bio_params, _ = PU.mutate_params(bio_params, space; rng=rng, mutation_mode=PU.BothMutations)
        for _ in 0:rand(rng, 0:2)
          bio_params, _ = PU.mutate_params(bio_params, space; rng=rng, mutation_mode=PU.BothMutations)
        end

        rep_results = _run(bio_params)
        fx, run_result, eco_losses = _fitness(rep_results)
        cached_sites_state = _median_rep_cached(rep_results, loss_params, scoring)

        next = MOLBSA.MOCandidate(bio_params, fx)
        is_new_best = MOLBSA.search_cmp!(next, search_state)
        if MOLBSA.should_restart(search_state)
          @info "Restarting @ $(search_state.i)"
          bio_params = next_candidate()
          rfx, _, _ = _fitness(_run(bio_params))
          is_new_best = MOLBSA.restart(search_state, MOLBSA.MOCandidate(bio_params, rfx))
        end

        mo_gen_finalize!(gio, search_state, 1, is_new_best, run_result, eco_losses, cached_sites_state)   # shared: ckpt on ANY archive change (train+val)
        MOLBSA.is_search_over(search_state) && break
      end
  end

  return search_state
end
