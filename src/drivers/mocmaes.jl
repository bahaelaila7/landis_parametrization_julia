function parametrize_MOCMAES(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, cmaes_lambda::Union{Nothing,Int}=nothing, cmaes_sigma0::Float64=0.3, cmaes_warmstart_seeds::Int=20, ipop::Bool=false, ipop_stagnation::Int=20, ipop_max_barren_restarts::Int=4, integer_handling::Bool=false, integer_std_factor::Float64=0.3, single_cov::Bool=false, seed_archive_from::Union{Nothing,String}=nothing, scale_seed::PU.LossScales=PU.LossScales(), exec::ExecOptions=ExecOptions(), search::SearchOptions=SearchOptions())
  env = mo_env(; ref_soa, plugin, splots, spdf_plts, spinup_cohorts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, spinup, rng, debug,
    search_tier, n_reps, n_output_plots, no_establishment, injection_cohorts, cycle_years, scale_seed, exec,
    val_splots, val_ref_soa, val_spdf_plts, val_site_sim_years, val_spinup_cohorts, val_injection_cohorts,
    val_dual_early=true)   # this driver built the validation Sim B before the val plot sample (RNG order)
  _sobol_cands = if !isnothing(seed_archive_from)
    _seed_st = JLD2.load_object(seed_archive_from)
    _arch = hasproperty(_seed_st, :archive) ? [c.x for c in _seed_st.archive] : []
    println("MO-CMA-ES: init+restart pool = ", length(_arch), " archive members from ", seed_archive_from)
    _arch
  elseif isnothing(sobol_candidates_db)
    []
  else
    load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  end
  n_species = env.n_species; max_sim_year = env.max_sim_year
  mo_val!(env)

  if exec.cv_reselect   # skip the search; just re-score the 10 reselection positions (5 per archive) on the (frozen-norm) val set
    env.have_val || (@warn "ExecOptions.cv_reselect set but no val split — nothing to do"; return)
    _cv_reselect_dump(output_dir, eco_list, eco_species_ids, n_species, max_sim_year,
      val_ref_soa, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params, scoring, simb,
      env.inj_val, env.val_dual_b, env.val_scales, env.fixed_seeds, val_splots, species_list, plugin)
    return
  end

  _run(p; ws=env.work_soa) = run_candidate(env, p; ws)
  _fitness(rep_results) = candidate_fitness(env, rep_results)

  sobol_idx = Ref(1)
  next_candidate() =
    if sobol_idx[] <= length(_sobol_cands)
      p = _sobol_cands[sobol_idx[]]
      sobol_idx[] += 1
      p
    else
      generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    end

  if isnothing(resume_from)
    bio_params = if !isnothing(start_from)
      @info "Seeding initial candidate from $start_from (fresh search_state)"
      p = _load_params_from_path(start_from, plugin.ctx)
      (p.SPECIES_LIST == species_list && p.ECO_LIST == eco_list) ||
        error("start_from params are incompatible with this run: their SPECIES_LIST/ECO_LIST differ from the loaded data.")
      p
    else
      next_candidate()
    end
    _seed_prob_estab!(bs_tables(plugin), bio_params, eco_list, species_list, eco_species_ids)   # calibration seed (in-place; stays in search)
    _seed_maturity!(bs_tables(plugin), bio_params, species_list)
    _seed_min_rel!(bio_params, eco_list, space.opts)   # pin MIN_REL out of search (fix_min_rel)
    _floor_bmax_seed!(bio_params, space)          # raise any seed B_MAX below its per-(eco,species) data floor
    frozen_ref = deepcopy(bio_params)   # frozen-growth reference for IPOP restarts (fix_growth must survive restart)
    slots = space.slots
    fx0, _, _ = _fitness(_run(bio_params))
    mean0 = PU.params_to_u(bio_params, space, slots)
    rep = MOLBSA.MOCandidate(bio_params, fx0)
    groups = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(space, slots))   # block-diagonal MO-CMA-ES
    @info "MO-CMA-ES block-diagonal: $(length(groups)) covariance blocks (sizes $(length.(groups)))"
    search_state = MOCMAES.MOCMAESState(mean0, cmaes_sigma0, rep, rng; lambda=cmaes_lambda, blocks=groups, max_iter=typemax(Int), archive_cap=archive_cap)
    search_state.n_evals = 1
    # WARM-START the distribution from the top-K Sobol seeds instead of the single best point: mean =
    # log-weighted recombination of the top-K, C₀ = their (shrunk) empirical covariance shape. K=1 ⇒ no-op
    # (keeps the single-point mean + C=I already set above).
    if cmaes_warmstart_seeds > 1 && isnothing(start_from) && length(_sobol_cands) >= 2
      K = min(cmaes_warmstart_seeds, length(_sobol_cands))
      us = [PU.params_to_u(_sobol_cands[i], space, slots) for i in 1:K]
      wr = [log(K + 1) - log(i) for i in 1:K]; wr ./= sum(wr)   # POSITIVE rank-decaying weights (all i≤K) → convex-combo mean (in [0,1]) + PSD covariance
      CMAES.warmstart_blocks!(search_state.blocks, us, wr)
      @info "MO-CMA-ES warm-start: mean = log-weighted recombination of top-$K Sobol seeds; C₀ = shrunk empirical covariance (shape only, σ₀ carries scale)"
    end
    if !isnothing(seed_archive_from)
      # Pre-seed the archive: evaluate each pooled member under THIS (dual) sim and offer it to the archive.
      # They aren't inserted otherwise — they're only restart centers — so without this the archive starts at 1.
      n0 = length(search_state.archive)
      for p in _sobol_cands
        fxp, _, _ = _fitness(_run(p))
        MOCMAES.update_archive!(search_state, MOLBSA.MOCandidate(p, fxp))
      end
      search_state.n_evals += length(_sobol_cands)
      println("MO-CMA-ES: pre-seeded archive with ", length(_sobol_cands), " members → ", length(search_state.archive), " non-dominated (was ", n0, ")")
    end
  else
    @info "Resuming from $resume_from"
    search_state = JLD2.load_object(resume_from)
    bio_params = search_state.representative.x
    frozen_ref = isnothing(start_from) ? nothing : _load_params_from_path(start_from, plugin.ctx)   # frozen-growth reference
    slots = space.slots
    if force_restart_from_random
      bio_params = next_candidate()
      _apply_frozen_growth!(bio_params, frozen_ref, space.opts)
      _seed_prob_estab!(bs_tables(plugin), bio_params, eco_list, species_list, eco_species_ids); _seed_maturity!(bs_tables(plugin), bio_params, species_list); _seed_min_rel!(bio_params, eco_list, space.opts)
      _floor_bmax_seed!(bio_params, space)
      MOCMAES.restart!(search_state, PU.params_to_u(bio_params, space, slots), cmaes_sigma0)
    end
  end
  # Hansen-style mixed-integer handling (honors the flag each run, incl. on resume).
  search_state.u_min_std = integer_handling ? PU.integer_u_min_std(space, slots, integer_std_factor) : Float64[]
  if TRIALS < 1 || search_state.n_evals >= TRIALS
    return search_state
  end

  if env.have_val && get(ENV, "VAL_REPLAY", "") == "1"
    ckpts = filter(f -> occursin(r"^best_params@\d+\.jld2$", f), readdir(output_dir))
    sort!(ckpts, by = f -> parse(Int, match(r"@(\d+)", f).captures[1]))
    @info "VAL_REPLAY: recomputing val for $(length(ckpts)) checkpoints in $output_dir"
    open(joinpath(output_dir, "val_replay.csv"), "w") do io
      println(io, "iteration,val_loss")
      for f in ckpts
        it = parse(Int, match(r"@(\d+)", f).captures[1])
        try
          p = JLD2.load_object(joinpath(output_dir, f))
          vl = _mo_val_loss(val_eval(env, p), eco_species_ids, loss_params, env.val_scales, env.val_dual_b)
          println(io, "$it,$vl"); flush(io); @info "val-replay @ $it = $vl"
        catch e
          @warn "val-replay failed @ $it" exception = (e, catch_backtrace())
        end
      end
    end
    @info "VAL_REPLAY done → $(joinpath(output_dir, "val_replay.csv"))"
    return nothing
  end

  stagnation = 0
  barren_restarts = 0             # consecutive IPOP restarts with no new best (barren-restart stop)
  improved_since_restart = false  # any new best found in the current IPOP epoch?
  loop = mo_loop_open(env, search_state; output_dir, resuming=!isnothing(resume_from))
  gio = loop.gio
  evals_done = search_state.n_evals
  est_gens = max(1, cld(TRIALS - evals_done, search_state.lambda))
  run_search_loop(loop; get_state=() -> search_state) do
      TProgress.@track for _gen in 1:est_gens
        evals_done >= TRIALS && break
        xs_u = CMAES.ask(search_state)
        λ = search_state.lambda
        fxs = Vector{MOLBSA.MOFitness}(undef, λ)
        cands = Vector{typeof(bio_params)}(undef, λ)
        gen_best_agg = Inf
        local gb_run, gb_eco, gb_cached, gb_idx
        # SERIAL: fit_params already threads internally over sites (see parametrize_CMAES note).
        for k in 1:λ
          cand = PU.u_to_params(xs_u[k], space, slots, bio_params)
          rep_results = _run(cand)
          fx_k, run_k, eco_k = _fitness(rep_results)
          cands[k] = cand
          fxs[k] = fx_k
          evals_done += 1
          if k == 1 || isless(fx_k.aggregate, gen_best_agg)   # k==1 ⇒ gb_* always defined; isless ⇒ NaN ranks worst
            gen_best_agg = fx_k.aggregate
            gb_idx = k
            gb_run = run_k
            gb_eco = eco_k
            gb_cached = _median_rep_cached(rep_results, loss_params, scoring)
          end
        end

        MOCMAES.tell!(search_state, fxs, xs_u; nds=search.mocmaes_nds)
        search_state.n_evals = evals_done

        is_new_best = false
        for k in 1:λ
          _upd = MOCMAES.update_archive!(search_state, MOLBSA.MOCandidate(cands[k], fxs[k]))
          is_new_best |= _upd.improved
        end
        search_state.current = MOLBSA.MOCandidate(cands[gb_idx], fxs[gb_idx])
        stagnation = is_new_best ? 0 : stagnation + 1
        improved_since_restart |= is_new_best
        mo_gen_finalize!(gio, search_state, search_state.lambda, is_new_best, gb_run, gb_eco, gb_cached)   # shared: ckpt on ANY archive change (train+val)

        if ipop && (search_state.sigma < 1e-11 || stagnation >= ipop_stagnation)
          # Barren-restart stop: count consecutive IPOP restarts that produced no new best; stop after N.
          barren_restarts = improved_since_restart ? 0 : barren_restarts + 1
          improved_since_restart = false
          if ipop_max_barren_restarts > 0 && barren_restarts >= ipop_max_barren_restarts
            @info "Stopping search @ gen $(search_state.i): $barren_restarts consecutive IPOP restarts with no improvement (≥ $ipop_max_barren_restarts)"
            break
          end
          new_lambda = search_state.lambda * 2
          @info "IPOP restart @ gen $(search_state.i): λ $(search_state.lambda) → $new_lambda"
          bio_params = next_candidate()
          _apply_frozen_growth!(bio_params, frozen_ref, space.opts)   # keep fix_growth frozen at the Sim-A seed across restarts
          _seed_prob_estab!(bs_tables(plugin), bio_params, eco_list, species_list, eco_species_ids)   # re-seed the calibration params
          _seed_maturity!(bs_tables(plugin), bio_params, species_list)
          _seed_min_rel!(bio_params, eco_list, space.opts)   # keep MIN_REL pinned across IPOP restarts
          MOCMAES.restart!(search_state, PU.params_to_u(bio_params, space, slots), cmaes_sigma0; lambda=new_lambda)
          stagnation = 0
        end
      end
  end

  env.have_val && try   # CV: held-out front metrics under THIS fold's live (train-frozen) loss — see _dump_cv_front
    _dump_cv_front(search_state, output_dir, eco_list, eco_species_ids, n_species, max_sim_year,
      val_ref_soa, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params, scoring, simb,
      env.inj_val, env.val_dual_b, env.val_scales, env.fixed_seeds, val_splots, val_injection_cohorts, plugin)
  catch e; @error "cv front dump failed" exception = (e, catch_backtrace()); end

  return search_state
end
