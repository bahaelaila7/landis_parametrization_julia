function parametrize_NSGA2(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, nsga2_pop::Int=48, nsga2_offspring::Union{Nothing,Int}=nothing, nsga2_eta_c::Float64=20.0, nsga2_eta_m::Float64=20.0, nsga2_pc::Float64=0.9, nsga2_pm::Float64=-1.0, sobol_init::Bool=true, scale_seed::PU.LossScales=PU.LossScales(), exec::ExecOptions=ExecOptions(), search::SearchOptions=SearchOptions())
  env = mo_env(; ref_soa, plugin, splots, spdf_plts, spinup_cohorts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, spinup, rng, debug,
    search_tier, n_reps, n_output_plots, no_establishment, injection_cohorts, cycle_years, scale_seed, exec,
    val_splots, val_ref_soa, val_spdf_plts, val_site_sim_years, val_spinup_cohorts, val_injection_cohorts)
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  n_species = env.n_species
  _run(p; ws=env.work_soa) = run_candidate(env, p; ws)
  _fitness(rep_results) = candidate_fitness(env, rep_results)
  next_param() = !isempty(_sobol_cands) && length(_sobol_cands) >= 1 ? popfirst!(_sobol_cands) :
                 generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)

  λ = isnothing(nsga2_offspring) ? nsga2_pop : nsga2_offspring
  if isnothing(resume_from)
    template = !isnothing(start_from) ? _load_params_from_path(start_from, plugin.ctx) : generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    slots = space.slots
    # initial μ population: Sobol candidate DB if given, else a Sobol space-filling design, else random draws.
    init_params = !isempty(_sobol_cands) ? [next_param() for _ in 1:nsga2_pop] :
                  sobol_init ? PU.sobol_samples(space, template, nsga2_pop) :
                  [generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment) for _ in 1:nsga2_pop]
    init_us = [PU.params_to_u(p, space, slots) for p in init_params]
    init_cands = [MOLBSA.MOCandidate(init_params[k], _fitness(_run(init_params[k]))[1]) for k in 1:nsga2_pop]
    rep0 = init_cands[argmin([Float64(c.fx.aggregate) for c in init_cands])]
    @info "NSGA-II: μ=$nsga2_pop parents, λ=$λ offspring/gen, d=$(length(slots)), ηc=$nsga2_eta_c ηm=$nsga2_eta_m pc=$nsga2_pc pm=$(nsga2_pm < 0 ? "1/d" : string(nsga2_pm))"
    search_state = NSGA2.NSGA2State(; representative=rep0, current=rep0, archive=MOLBSA.MOCandidate{typeof(template)}[], best_iterations=Tuple{Int,Float64,MOLBSA.MOCandidate{typeof(template)}}[], archive_cap=archive_cap, rng=rng, i=0, n_evals=nsga2_pop, max_iter=typemax(Int), pop_u=init_us, pop=init_cands, n=length(slots), mu=nsga2_pop, lambda=λ, eta_c=nsga2_eta_c, eta_m=nsga2_eta_m, p_c=nsga2_pc, p_m=nsga2_pm)
    for c in init_cands; NSGA2._archive!(search_state, c); end
    bio_params = template
  else
    @info "Resuming from $resume_from"
    search_state = JLD2.load_object(resume_from)
    bio_params = search_state.representative.x
    slots = space.slots
  end
  if TRIALS < 1 || search_state.n_evals >= TRIALS
    return search_state
  end

  mo_val!(env)
  loop = mo_loop_open(env, search_state; output_dir, resuming=!isnothing(resume_from))
  gio = loop.gio

  evals_done = search_state.n_evals
  est_gens = max(1, cld(TRIALS - evals_done, λ))
  run_search_loop(loop; get_state=() -> search_state) do
      TProgress.@track for _gen in 1:est_gens
        evals_done >= TRIALS && break
        offs = NSGA2.ask(search_state)                  # λ offspring u-vectors (tournament + SBX + polynomial mutation)
        off_fxs = Vector{MOLBSA.MOFitness}(undef, λ)
        off_params = Vector{typeof(bio_params)}(undef, λ)
        gen_best_agg = Inf; local gb_run, gb_eco, gb_cached, gb_idx
        for k in 1:λ                                     # SERIAL (fit_params threads internally)
          p = PU.u_to_params(offs[k], space, slots, bio_params)
          rep_results = _run(p); fx_k, run_k, eco_k = _fitness(rep_results)
          off_params[k] = p; off_fxs[k] = fx_k; evals_done += 1
          if k == 1 || isless(fx_k.aggregate, gen_best_agg)   # k==1 ⇒ gb_* always defined (all-NaN gen); NaN ranks worst
            gen_best_agg = fx_k.aggregate; gb_idx = k; gb_run = run_k; gb_eco = eco_k; gb_cached = _median_rep_cached(rep_results, loss_params, scoring)
          end
        end
        is_new_best = NSGA2.tell!(search_state, off_fxs, off_params, offs)   # (μ+λ) non-dom-sort + crowding selection
        search_state.n_evals = evals_done
        mo_gen_finalize!(gio, search_state, search_state.mu, is_new_best, gb_run, gb_eco, gb_cached)   # shared: ckpt on ANY archive change (train+val)
      end
  end
  return search_state
end
