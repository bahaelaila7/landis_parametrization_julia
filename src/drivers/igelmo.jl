# Igel/Hansen/Roth (2007) population-based MO-CMA-ES driver. Same MO objective vector, archive and
# writer as parametrize_MOCMAES, but the engine is a population of μ (1+1)-CMA-ES individuals
# (IgelMOCMAES) rather than one distribution — better front spread/extreme coverage. μ candidates
# are evaluated per generation (serially); the initial population is a Sobol design.
function parametrize_IgelMOCMAES(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, igel_mu::Int=20, igel_sigma0::Float64=0.3, igel_sobol_init::Bool=true, igel_sobol_pool_k::Int=1, igel_sobol_raw_mult::Int=2, igel_niche_radius::Float64=0.0, igel_reseed_sigma::Float64=0.0, igel_reseed_random_frac::Float64=0.0, igel_maturity::Int=0, igel_seed_maturity::Bool=false, igel_warm_start::Bool=false, igel_freeze_seed_growth::Bool=false, single_cov::Bool=false, scale_seed::PU.LossScales=PU.LossScales(), exec::ExecOptions=ExecOptions(), search::SearchOptions=SearchOptions())
  env = mo_env(; ref_soa, plugin, splots, spdf_plts, spinup_cohorts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, spinup, rng, debug,
    search_tier, n_reps, n_output_plots, no_establishment, injection_cohorts, cycle_years, scale_seed, exec,
    val_splots, val_ref_soa, val_spdf_plts, val_site_sim_years, val_spinup_cohorts, val_injection_cohorts,
    train_sites_parallel=!candidate_parallel(exec))   # candidate mode: sites go serial for EVERY eval of this driver (init population, gen loop, val)
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  n_species = env.n_species; _work_soa_t = env.work_soa_t; _timer = env.timer
  _run(p; ws=env.work_soa) = run_candidate(env, p; ws)
  _fitness(rep_results) = candidate_fitness(env, rep_results)
  if isnothing(resume_from)
    template = !isnothing(start_from) ? _load_params_from_path(start_from, plugin.ctx) : generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    slots = space.slots
    # initial population — three sources, all feeding the same "best μ·pool_k by dominance → RANDOM μ" thinning
    # (pool_k>1) that gives a good-but-spread start instead of seeding all μ at the single best spot (→ collapse):
    #   • sobol_candidates_db → params PRE-evaluated + dominance-sorted best-first by load_sobol_candidates
    #     (generated offline by search_mode="sobol", which runs the SAME fit_params simulator + _mo_objectives as
    #     this run). Cut uses the stored ranking — NO re-eval of the whole pool; only the selected μ are evaluated.
    #   • igel_sobol_init     → a fresh Sobol design generated here; OVERSAMPLED to (μ·pool_k)·raw_mult, all
    #     evaluated, then dominance-cut + random-thinned below (inline_cut).
    #   • else                → random generated params (no pooling).
    do_pool = igel_sobol_pool_k > 1
    elite_n = igel_mu * igel_sobol_pool_k
    inline_cut = false
    pool_params =
      if !isempty(_sobol_cands)
        best = _sobol_cands[1:min(elite_n, length(_sobol_cands))]                      # already dominance-sorted best-first
        do_pool ? Random.shuffle(rng, best)[1:min(igel_mu, length(best))] : _sobol_cands[1:min(igel_mu, length(_sobol_cands))]
      elseif igel_warm_start && !isnothing(start_from)
        # warm start: the start_from params themselves + (μ−1) Gaussian perturbations around them in u-space
        u0 = PU.params_to_u(template, space, slots)
        @info "igelmo warm start from $(basename(start_from)): μ=$igel_mu around it (σ=$igel_sigma0)"
        vcat([template], [PU.u_to_params(clamp.(u0 .+ igel_sigma0 .* randn(rng, length(u0)), 0.0, 1.0), space, slots, template) for _ in 2:igel_mu])
      elseif igel_sobol_init
        inline_cut = do_pool
        PU.sobol_samples(space, template, do_pool ? elite_n * igel_sobol_raw_mult : igel_mu)
      else
        [generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment) for _ in 1:igel_mu]
      end
    # Seed the ESTABLISHMENT params from the data tables (prob_estab_from_data / maturity_from_data) and pin
    # MIN_REL (fix_min_rel) — mirroring parametrize_MOCMAES. Applied to the template AND every pool member BEFORE
    # eval/params_to_u, so the encoded u and the frozen `bases` carry the seeded/pinned values. Growth
    # ({D,S,B_MAX,ANPP_MAX}) is untouched. Each fn is a no-op unless its flag/table is set.
    for _p in Iterators.flatten(((template,), pool_params))
      _seed_prob_estab!(bs_tables(plugin), _p, eco_list, species_list, eco_species_ids)
      _seed_maturity!(bs_tables(plugin), _p, species_list)
      _seed_min_rel!(_p, eco_list, space.opts)
    end
    # evaluate under THIS run's exact loss config: the raw pool if inline_cut, else the already-selected μ.
    np = length(pool_params); pool_fx = Vector{Any}(undef, np)
    if candidate_parallel(exec)     # candidate-parallelize the eval (sites already serial here)
      Threads.@threads :static for k in 1:np
        pool_fx[k] = _fitness(_run(pool_params[k]; ws=_work_soa_t[Threads.threadid()]))[1]
      end
    else
      for k in 1:np; pool_fx[k] = _fitness(_run(pool_params[k]))[1]; end
    end
    # inline Sobol only: cut the freshly-evaluated raw pool to best μ·pool_k by dominance, then UNIFORM-RANDOM μ.
    # (DB / random / non-pool sources already hold exactly the μ starting individuals.)
    sel = if inline_cut
      elite = NSGA2._select(Vector{FloatType}[fx.objectives for fx in pool_fx], min(elite_n, np))
      chosen = Random.shuffle(rng, elite)[1:min(igel_mu, length(elite))]
      @info "igelmo Sobol-pool init: evaluated $np → kept best $(length(elite)) by dominance → random-selected $(length(chosen)) starting individuals"
      chosen
    else
      collect(1:np)
    end
    init_params = pool_params[sel]
    init_us = [PU.params_to_u(p, space, slots) for p in init_params]
    init_cands = [MOLBSA.MOCandidate(init_params[k], pool_fx[sel[k]]) for k in eachindex(sel)]
    groups = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(space, slots))   # block-diagonal per-individual (1+1)-CMA
    @info "Igel MO-CMA-ES block-diagonal: $(length(groups)) covariance blocks per individual (sizes $(length.(groups)))"
    search_state = IgelMOCMAES.IgelState(init_us, init_cands, rng; sigma0=igel_sigma0, archive_cap=archive_cap, max_iter=typemax(Int), niche_radius=igel_niche_radius, reseed_sigma=igel_reseed_sigma, maturity_period=igel_maturity, init_mature=igel_seed_maturity, blocks=groups)
    igel_seed_maturity && igel_maturity > 0 && @info "igelmo phase-1: $igel_mu seeds shielded for $igel_maturity gens, then (μ+μ) competition"
    search_state.n_evals = np
    bio_params = template
    # PER-LINEAGE FROZEN GROWTH: each seed keeps its OWN {D,S,B_MAX,ANPP_MAX} (fix_growth removes them from the
    # slots ⇒ they come from the per-lineage `base` handed to u_to_params). `bases[k]` tracks lineage k's seed;
    # aligned to search_state.pop[k] here (pop is built from init_us in order) and re-aligned by object identity
    # after every tell! (below). Sidecar-persisted for resume.
    bases = igel_freeze_seed_growth ? copy(init_params) : nothing
    if igel_freeze_seed_growth
      space.opts.fix_growth || @warn "igel_freeze_seed_growth is on but fix_growth is OFF → growth is in the search slots and will NOT stay frozen"
      # _sobol_cands is already drained into init_params by here, so check the seed SOURCE instead.
      (isnothing(sobol_candidates_db) && isnothing(start_from)) && @warn "igel_freeze_seed_growth is on but no seed source (sobol_candidates_db/start_from) → freezing RANDOM growth"
      @info "igelmo per-lineage frozen growth: $igel_mu lineages each keep their seed's {D,S,B_MAX,ANPP_MAX}"
    end
  else
    @info "Resuming from $resume_from"
    # (RANKW is the TRAIN reference's, carried explicitly by `scales`/`val_scales` from the start — no throwaway train eval needed on resume.)
    _resumed = IgelMOCMAES.load_state(IgelMOCMAES.IgelSearch, resume_from; dir=output_dir, bases=igel_freeze_seed_growth)   # state (+ per-lineage bases sidecar from output_dir)
    search_state = IgelMOCMAES.state(_resumed); bases = _resumed.bases
    bio_params = search_state.representative.x
    slots = space.slots
    bases === nothing || @info "igelmo per-lineage frozen growth: reloaded $(length(bases)) lineage bases from igel_bases@$(search_state.i).jld2"
  end

  # Engine options of this run (explicit, not state fields → they apply on fresh AND resume): the config-derived ones of `search.igel` + this
  # driver's reseed fraction + parameter-space niching in native units / semantic quanta when igel_niche_radius>0 (the typed maps close over
  # the decode template: bio_params is the u_to_params template on both fresh & resume — only slot dims are read ⇒ immaterial, frozen-growth-safe).
  igel_opts = let base = IgelMOCMAES.with_reseed_random_frac(search.igel, igel_reseed_random_frac), bp = bio_params
    igel_niche_radius > 0 ?
      IgelMOCMAES.with_niching(base, PU.slot_quanta(space, slots),
        u -> PU.params_to_nativevec(PU.u_to_params(u, space, slots, bp), space, slots),
        p -> PU.params_to_nativevec(p, space, slots)) : base
  end
  isearch = IgelMOCMAES.IgelSearch(search_state, igel_opts; bases=bases)   # the resumable search: engine state + run options + per-lineage bases
  igel_niche_radius > 0 && @info "igelmo parameter-space niching ON: radius=$(igel_niche_radius) native-quanta (L∞) over $(length(slots)) slots"

  if TRIALS < 1 || search_state.n_evals >= TRIALS
    return search_state
  end

  # Held-out val must measure the SAME objective as train: Sim B (dual_b) when dual_mode≠off, else Sim A.
  mo_val!(env)
  loop = mo_loop_open(env, search_state; output_dir, resuming=!isnothing(resume_from))
  gio = loop.gio

  evals_done = search_state.n_evals
  est_gens = max(1, cld(TRIALS - evals_done, igel_mu))
  done_gens = isnothing(resume_from) ? 0 : search_state.i    # gens already completed (search_state.i == checkpoint/metrics counter)
  total_gens = done_gens + est_gens                          # size the bar to the WHOLE run so resume continues the counter/bar
  # resume-safe reseed/σ trace → a SEPARATE csv (NOT metrics.csv, which appends under a fixed header): a stop→resume
  # just appends; header is written only when the file is new/empty. Columns: gen, mean σ, min σ, #below reseed floor, #reseeds.
  _reseed_path = joinpath(output_dir, "reseed_trace.csv"); _reseed_new = !isfile(_reseed_path) || filesize(_reseed_path) == 0
  _reseed_io = open(_reseed_path, _reseed_new ? "w" : "a")
  _reseed_new && (println(_reseed_io, "iteration,mean_sigma,min_sigma,n_below_floor,n_reseed"); flush(_reseed_io))
  run_search_loop(loop; get_state=() -> search_state, save_exit=() -> (IgelMOCMAES.save_state(isearch, output_dir); @info "Search state saved @ $(search_state.i)"), close_extra=() -> close(_reseed_io)) do
      _first_gen = true
      TProgress.@track for _gen in 1:total_gens                # _gen = ABSOLUTE generation number
        _gen <= done_gens && continue                          # RESUME: fast-skip completed gens (bar jumps to the resumed position, then continues)
        evals_done >= TRIALS && break
        if _timer !== nothing
          _first_gen && (println("[timing] startup→first gen: $(_tsec(time_ns()-exec.t0))s"); _first_gen = false)
          reset!(_timer)
        end
        _tg = time_ns()
        offs = IgelMOCMAES.ask(isearch)
        _t_ask = time_ns() - _tg
        off_fxs = Vector{MOLBSA.MOFitness}(undef, igel_mu)
        off_params = Vector{typeof(bio_params)}(undef, igel_mu)
        off_run = Vector{Any}(undef, igel_mu); off_eco = Vector{Any}(undef, igel_mu)   # retained per-candidate loss/eco (cheap) → gen-best needs no serial re-sim
        gen_best_agg = Inf; local gb_run, gb_eco, gb_cached, gb_idx
        _te = time_ns()
        if candidate_parallel(exec)
          # candidate×rep is the parallel unit: each candidate's sim runs single-threaded on a thread-local SoA
          # (NUMA-local, no shared-array contention). Sites are serial here (`_sites_parallel` is false in candidate mode).
          Threads.@threads :static for k in 1:igel_mu
            p = PU.u_to_params(offs[k], space, slots, isearch.bases === nothing ? bio_params : isearch.bases[k])
            off_params[k] = p
            fx_k, run_k, eco_k = _fitness(_run(p; ws=_work_soa_t[Threads.threadid()]))
            off_fxs[k] = fx_k; off_run[k] = run_k; off_eco[k] = eco_k   # keep the batch's run/eco (self-contained loss structs; no SoA aliasing)
          end
          evals_done += igel_mu
          for k in 1:igel_mu                              # serial best-pick (gen argmin aggregate)
            (k == 1 || isless(off_fxs[k].aggregate, gen_best_agg)) && (gen_best_agg = off_fxs[k].aggregate; gb_idx = k)
          end
          gb_run = off_run[gb_idx]; gb_eco = off_eco[gb_idx]  # already computed in the parallel batch (identical to a re-sim: fixed seeds) — no serial full-sim
          gb_cached = n_output_plots > 0 ? _median_rep_cached(_run(off_params[gb_idx]; ws=_work_soa_t[1]), loss_params, scoring) : nothing  # cohort cache only feeds output plots; unused otherwise
        else
          for k in 1:igel_mu                              # SoA mode: serial candidates, sites parallel (fit_params threads)
            p = PU.u_to_params(offs[k], space, slots, isearch.bases === nothing ? bio_params : isearch.bases[k])
            rep_results = _run(p); fx_k, run_k, eco_k = _fitness(rep_results)
            off_params[k] = p; off_fxs[k] = fx_k; evals_done += 1
            if k == 1 || isless(fx_k.aggregate, gen_best_agg)   # k==1 ⇒ gb_* always defined (all-NaN gen); NaN ranks worst
              gen_best_agg = fx_k.aggregate; gb_idx = k; gb_run = run_k; gb_eco = eco_k; gb_cached = _median_rep_cached(rep_results, loss_params, scoring)
            end
          end
        end
        _t_eval = time_ns() - _te; _sim = _timer === nothing ? UInt64(0) : _timer.sim
        _tt = time_ns()
        is_new_best = IgelMOCMAES.tell!(isearch, off_fxs, off_params)   # also realigns the per-lineage frozen-growth bases with the survivors
        _t_tell = time_ns() - _tt
        search_state.n_evals = evals_done
        _tf = time_ns()
        _save_ckpt = mo_gen_finalize!(gio, search_state, igel_mu, is_new_best, gb_run, gb_eco, gb_cached)   # shared: ckpt on ANY archive change (train+val)
        # sidecar the per-lineage bases WHENEVER a checkpoint is saved, tagged with the same gen so resume loads
        # the exact pop↔bases alignment (search_state.pop is not mutated between here and the next ask).
        _save_ckpt && try; IgelMOCMAES.save_aux(isearch, output_dir); catch e; @warn "igel_bases sidecar save failed" exception=e; end
        let _sg = Float64[ind.sigma for ind in search_state.pop], _nr = count(search_state._reseed)   # reseed/σ signal
          println(_reseed_io, string(search_state.i, ",", round(search_state.t, sigdigits=6), ",", round(minimum(_sg), sigdigits=6), ",", count(<(igel_reseed_sigma), _sg), ",", _nr))
          (_nr > 0 || search_state.i % 100 == 0) && flush(_reseed_io)
          _nr > 0 && @info "igelmo gen $(search_state.i): $_nr reseed(s) — mean_σ=$(round(search_state.t, sigdigits=3)) min_σ=$(round(minimum(_sg), sigdigits=3))"
        end
        if _timer !== nothing
          println("[timing] gen $_gen: $(_tsec(time_ns()-_tg))s | ask=$(_tsec(_t_ask)) eval=$(_tsec(_t_eval))(sim=$(_tsec(_sim)) loss+oth=$(_tsec(_t_eval-_sim))) tell=$(_tsec(_t_tell)) final=$(_tsec(time_ns()-_tf))(io=$(_tsec(_timer.io)))")
        end
      end
  end
  return search_state
end
