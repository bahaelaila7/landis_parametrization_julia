# Cooperative-Coevolutionary Igel MO-(1+1)-CMA-ES ("ccigel"). Clone of parametrize_IgelMOCMAES with a CC
# layer (see search/CCIgel.jl): population = n_species × K whole individuals in per-species groups. It
# alternates SPECIALIZATION phases (each species group runs a (1+1)-CMA restricted to that species' u-slots,
# scored on the species' own 2-objective loss, NSGA-II select K within the group) with a RECOMBINATION
# (whole individuals rebuilt block-by-block from random specialists) and INTEGRATION phases (verbatim Igel:
# whole-vector (1+1) on the aggregate 2-obj, NSGA-II selection, Pareto archive). Archive / metrics /
# checkpoints (mo_gen_finalize!) fire ONLY in integration; specialization phases are warmups (they advance
# n_evals and log a light progress line). Schedule: cc_cycles × [spec cc_spec_gens + integ cc_integ_gens],
# then integration until the TRIALS budget is exhausted.
function parametrize_CCIgel(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, igel_sigma0::Float64=0.3, igel_sobol_init::Bool=true, igel_niche_radius::Float64=0.0, igel_reseed_sigma::Float64=0.0, igel_maturity::Int=0, single_cov::Bool=false, cc_group_size::Int=2, cc_spec_gens::Int=20, cc_integ_gens::Int=20, cc_cycles::Int=5, cc_fix_others::Bool=false, scale_seed::PU.LossScales=PU.LossScales(), exec::ExecOptions=ExecOptions(), search::SearchOptions=SearchOptions())
  env = mo_env(; ref_soa, plugin, splots, spdf_plts, spinup_cohorts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, spinup, rng, debug,
    search_tier, n_reps, n_output_plots, no_establishment, injection_cohorts, cycle_years, scale_seed, exec,
    val_splots, val_ref_soa, val_spdf_plts, val_site_sim_years, val_spinup_cohorts, val_injection_cohorts)
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  n_species = env.n_species
  _run(p; ws=env.work_soa) = run_candidate(env, p; ws)
  _fitness(rep_results) = candidate_fitness(env, rep_results)
  # per-species objective (specialization scoring): score species g only, aggregate = Σ of its 2 objectives.
  function _fitness_species(rep_results, gsp::Int)
    _, eco_losses, _ = _agg_reps(rep_results, loss_params, scoring)
    objs = _mo_objectives_species(eco_losses, eco_species_ids, loss_params, env.scales, env.dual_b, gsp)
    MOLBSA.MOFitness(objs, Float64(sum(objs)))
  end
  next_param() = !isempty(_sobol_cands) && length(_sobol_cands) >= 1 ? popfirst!(_sobol_cands) :
                 generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)

  K = cc_group_size
  cc_mu = n_species * K                          # whole-individual population size = generation eval count
  CCIgel.register_sampler_types!(PU.SpeciesSampler, PU.EcoSpeciesSampler)
  if isnothing(resume_from)
    template = !isnothing(start_from) ? _load_params_from_path(start_from, plugin.ctx) : generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    slots = space.slots
    # initial population of cc_mu whole individuals: Sobol candidate DB / Sobol design / random draws.
    init_params = !isempty(_sobol_cands) ? [next_param() for _ in 1:cc_mu] :
                  igel_sobol_init ? PU.sobol_samples(space, template, cc_mu) :
                  [generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment) for _ in 1:cc_mu]
    init_us = [PU.params_to_u(p, space, slots) for p in init_params]
    init_cands = [MOLBSA.MOCandidate(init_params[k], _fitness(_run(init_params[k]))[1]) for k in 1:cc_mu]
    groups = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(space, slots))
    species_slots = CCIgel.species_slot_map(space.dists, slots, eco_species_ids, n_species)
    n_specialized = count(!isempty, species_slots)
    @info "CC-Igel: $n_species species × K=$K = $cc_mu individuals; $n_specialized species have specialization slots; integration blocks=$(length(groups))"
    search_state = IgelMOCMAES.IgelState(init_us, init_cands, rng; sigma0=igel_sigma0, archive_cap=archive_cap, max_iter=typemax(Int), niche_radius=igel_niche_radius, reseed_sigma=igel_reseed_sigma, maturity_period=igel_maturity, blocks=groups)
    search_state.n_evals = cc_mu
    bio_params = template
  else
    @info "Resuming from $resume_from"
    search_state = JLD2.load_object(resume_from)
    bio_params = search_state.representative.x
    slots = space.slots
    species_slots = CCIgel.species_slot_map(space.dists, slots, eco_species_ids, n_species)
  end
  if TRIALS < 1 || search_state.n_evals >= TRIALS
    return search_state
  end

  mo_val!(env)
  loop = mo_loop_open(env, search_state; output_dir, resuming=!isnothing(resume_from))
  gio = loop.gio

  evals_done = search_state.n_evals

  # ---- one INTEGRATION generation: verbatim Igel over the whole vector (archive/metrics/ckpt) ----
  function integ_gen!()
    offs = IgelMOCMAES.ask(search_state, search.igel)
    m = search_state.mu
    off_fxs = Vector{MOLBSA.MOFitness}(undef, m)
    off_params = Vector{typeof(bio_params)}(undef, m)
    gen_best_agg = Inf; local gb_run, gb_eco, gb_cached
    for k in 1:m
      p = PU.u_to_params(offs[k], space, slots, bio_params)
      rep_results = _run(p); fx_k, run_k, eco_k = _fitness(rep_results)
      off_params[k] = p; off_fxs[k] = fx_k; evals_done += 1
      if k == 1 || isless(fx_k.aggregate, gen_best_agg)   # k==1 ⇒ gb_* always defined (all-NaN gen); NaN ranks worst
        gen_best_agg = fx_k.aggregate; gb_run = run_k; gb_eco = eco_k; gb_cached = _median_rep_cached(rep_results, loss_params, scoring)
      end
    end
    is_new_best = IgelMOCMAES.tell!(search_state, off_fxs, off_params, search.igel)
    search_state.n_evals = evals_done
    mo_gen_finalize!(gio, search_state, m, is_new_best, gb_run, gb_eco, gb_cached)
  end

  # ---- build per-species specialization sub-states from the current integration population ----
  # Each species group g gets an IgelState over the SAME whole u-vectors but with blocks=[species_slots[g]]
  # so ask/tell perturb/update ONLY species g's slots. Group members = a K-slice of the current population.
  function build_cc_groups()
    cur_us = [copy(ind.x) for ind in search_state.pop]                  # Individual.x IS the u-vector
    S = IgelMOCMAES.IgelState
    states = Vector{Union{Nothing,S}}(undef, n_species)
    for s in 1:n_species
      if isempty(species_slots[s])
        states[s] = nothing; continue
      end
      lo = (s - 1) * K + 1; hi = min(s * K, length(cur_us))
      idxs = collect(lo:hi); length(idxs) < K && append!(idxs, rand(rng, 1:length(cur_us), K - length(idxs)))
      grp_us = [copy(cur_us[i]) for i in idxs]
      grp_cands = [MOLBSA.MOCandidate(PU.u_to_params(cur_us[i], space, slots, bio_params), search_state.pop[i].fx) for i in idxs]
      states[s] = IgelMOCMAES.IgelState(grp_us, grp_cands, rng; sigma0=igel_sigma0, archive_cap=archive_cap,
        max_iter=typemax(Int), blocks=Vector{Int}[species_slots[s]])
    end
    return CCIgel.CCGroups(states, species_slots)
  end

  # ---- one SPECIALIZATION generation over ALL species groups (warmup: advances n_evals, no archive) ----
  # `bg` (cc_fix_others=true) is a SHARED background u-vector (the current-best representative) held for the
  # phase: an offspring is evaluated by overwriting only species s's dims of `bg`. `bg=nothing` (default)
  # keeps each individual's own (drifting) other-blocks — Igel.ask perturbs only s's dims of ind.x.
  function spec_gen!(cc, bg::Union{Nothing,Vector{Float64}})
    tot = 0.0; nsp = 0
    for s in 1:n_species
      st = cc.states[s]; st === nothing && continue
      offs = IgelMOCMAES.ask(st, search.igel)
      off_fxs = Vector{MOLBSA.MOFitness}(undef, st.mu)
      off_params = Vector{typeof(bio_params)}(undef, st.mu)
      for k in 1:st.mu
        u = offs[k]
        if bg !== nothing                                # splice species s's dims onto the shared background
          u = copy(bg); @inbounds for d in species_slots[s]; u[d] = offs[k][d]; end
        end
        p = PU.u_to_params(u, space, slots, bio_params)   # decode WHOLE vector
        off_params[k] = p; off_fxs[k] = _fitness_species(_run(p), s); evals_done += 1
      end
      IgelMOCMAES.tell!(st, off_fxs, off_params, search.igel)                      # NSGA-II select K within group g
      tot += minimum(fx.aggregate for fx in off_fxs); nsp += 1
    end
    search_state.n_evals = evals_done
    return nsp == 0 ? 0.0 : tot / nsp
  end

  # one RECOMBINATION step: rebuild the integration population block-by-block from random specialists, then
  # swap in a fresh integration IgelState over it (keeping the accumulated archive / representative).
  function recombine!(cc)
    base_us = [copy(ind.x) for ind in search_state.pop]                 # Individual.x IS the u-vector
    new_us = CCIgel.recombine_us(cc, n_species, K, base_us, rng)
    new_params = [PU.u_to_params(u, space, slots, bio_params) for u in new_us]
    new_cands = [MOLBSA.MOCandidate(new_params[k], _fitness(_run(new_params[k]))[1]) for k in 1:cc_mu]
    evals_done += cc_mu
    groups2 = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(space, slots))
    old_archive = search_state.archive; old_rep = search_state.representative
    old_best_iter = search_state.best_iteration; old_best_iters = search_state.best_iterations
    gen_i = search_state.i
    search_state = IgelMOCMAES.IgelState(new_us, new_cands, rng; sigma0=igel_sigma0, archive_cap=archive_cap,
      max_iter=typemax(Int), niche_radius=igel_niche_radius, reseed_sigma=igel_reseed_sigma, maturity_period=igel_maturity, blocks=groups2)
    search_state.archive = old_archive; search_state.representative = old_rep
    search_state.best_iteration = old_best_iter; search_state.best_iterations = old_best_iters
    search_state.i = gen_i; search_state.n_evals = evals_done
    return nothing
  end

  # Phase SCHEDULE as a list of (:integ | :spec) tags. cc_fix_others=true needs a current-best before the
  # first specialization → a LEADING integration phase. After the scheduled cycles, keep integrating to budget.
  schedule = Symbol[]
  cc_fix_others && push!(schedule, :integ)                 # leading aggregate phase (provides the background)
  for _c in 1:cc_cycles; push!(schedule, :spec); push!(schedule, :integ); end

  run_search_loop(loop; get_state=() -> search_state) do
      _cyc = 0
      TProgress.@track for _phase in 1:length(schedule)
        evals_done >= TRIALS && break
        if schedule[_phase] == :spec
          _cyc += 1
          cc = build_cc_groups()
          bg = cc_fix_others ? PU.params_to_u(search_state.representative.x, space, slots) : nothing
          for _g in 1:cc_spec_gens
            evals_done >= TRIALS && break
            m = spec_gen!(cc, bg)
            @info "CC-Igel spec cycle $_cyc gen $_g | mean-species-best=$(round(m; sigdigits=4)) | evals=$evals_done | fix_others=$cc_fix_others"
          end
          recombine!(cc)                                     # → fresh integration population
        else
          for _g in 1:cc_integ_gens
            evals_done >= TRIALS && break
            integ_gen!()
          end
        end
      end
      # remainder: pure integration until the budget is exhausted
      while evals_done < TRIALS
        integ_gen!()
      end
  end
  return search_state
end
