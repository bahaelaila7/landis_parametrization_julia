# CMA-ES driver. Mirrors parametrize_LBSA's setup and checkpoint/writer machinery, but optimizes
# the Biomass Succession params with Covariance Matrix Adaptation in the vector bridge's [0,1]^d
# u-space. Each generation samples λ candidates and evaluates them SERIALLY (fit_params already
# threads internally), then updates the CMA-ES distribution. TRIALS is the total fit_params budget
# (so it's comparable to LBSA); λ defaults to Hansen's 4+⌊3·ln(n)⌋. `ipop=true` enables IPOP
# restarts (doubling λ from a fresh mean on σ-collapse / stagnation); default is a single run.
function parametrize_CMAES(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=3, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, cmaes_lambda::Union{Nothing,Int}=nothing, cmaes_sigma0::Float64=0.3, ipop::Bool=false, ipop_stagnation::Int=20, ipop_max_barren_restarts::Int=4, integer_handling::Bool=false, integer_std_factor::Float64=0.3, single_cov::Bool=false, scale_seed::PU.LossScales=PU.LossScales(), exec::ExecOptions=ExecOptions(), search::SearchOptions=SearchOptions())
  splots.sim_year .= Dates.value.(Dates.Day.(splots.measdate - splots.start_measdate)) ./ 365.25 .|> round .|> Int

  # Fixed plot sample chosen once at startup so progress is comparable across iterations
  all_plot_ids = UIntType.(unique(splots.plot_id))
  sampled_ids = _sample_plot_ids(all_plot_ids, n_output_plots, rng; injection_cohorts=injection_cohorts)
  emp_sample = n_output_plots > 0 ? _make_emp_df(splots, sampled_ids) : nothing

  n_species = length(species_list)
  max_sim_year = site_sim_years.sim_years .|> maximum |> maximum
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  injection = build_injection_plan(injection_cohorts, ref_soa, eco_species_ids, loss_params, scoring)   # this reference's injection dict, shock and exclusions

  # Validation setup
  have_val = !isnothing(val_ref_soa) && !exec.skip_val   # ExecOptions.skip_val (PAN_SKIP_VAL) disables all held-out val re-sim during training
  inj_val = have_val ? build_injection_plan(val_injection_cohorts, val_ref_soa, eco_species_ids, loss_params, scoring) : nothing   # the val reference's own plan
  val_dual_b = have_val ? _build_val_dual_b(val_splots, eco_species_ids, eco_list, val_spdf_plts, loss_params, cycle_years, val_injection_cohorts, val_spinup_cohorts, rng, no_establishment; scoring, simb, plugin=plugin) : nothing
  sampled_ids_val = (have_val && n_output_plots > 0) ?
                    _sample_plot_ids(UIntType.(unique(val_splots.plot_id)), n_output_plots, rng; injection_cohorts=val_injection_cohorts) :
                    Set{UIntType}()
  emp_sample_val = (have_val && n_output_plots > 0) ? _make_emp_df(val_splots, sampled_ids_val) : nothing

  t4_ref = nothing
  cycle_map = nothing
  n_cycles = 0
  if search_tier == 1
    n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    t1_ref = [zeros(FloatType, length(eco_species_ids[eco_id]), n_bins) for eco_id in eachindex(eco_list)]
    for ((_, eco_id), year_dict) in spdf_plts
      for (_, spdf_gt) in year_dict
        for (sp_eco, rec) in spdf_gt.records
          t1_ref[eco_id][sp_eco, :] .+= diff([0f0; rec.sp_age_cdf]) .* rec.sp_agb_sum
        end
      end
    end
    t2_ref = nothing
  elseif search_tier == 2
    t1_ref = nothing
    n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    t2_ref_bins = [[FloatType[] for sp in 1:length(eco_species_ids[eco_id]), b in 1:n_bins] for eco_id in eachindex(eco_list)]
    t2_ref_total = [[FloatType[] for _ in 1:length(eco_species_ids[eco_id])] for eco_id in eachindex(eco_list)]
    for ((_, eco_id), year_dict) in spdf_plts
      for (_, spdf_gt) in year_dict
        for (sp_eco, rec) in spdf_gt.records
          bin_probs = diff([0f0; rec.sp_age_cdf])
          for b in 1:n_bins
            push!(t2_ref_bins[eco_id][Int(sp_eco), b], bin_probs[b] * rec.sp_agb_sum)
          end
          push!(t2_ref_total[eco_id][Int(sp_eco)], rec.sp_agb_sum)
        end
      end
    end
    t2_ref = (bins=t2_ref_bins, total=t2_ref_total)
  elseif search_tier == 4 || search_tier == 5 || simb.dual_mode != :off
    n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    cycle_map, n_cycles = Data.build_cycle_map(splots; cycle_years=cycle_years, cycle=scoring.cycle)
    t4_ref = [[zeros(FloatType, length(eco_species_ids[eco_id]), n_bins) for _ in 1:n_cycles] for eco_id in eachindex(eco_list)]
    for ((plot_id, eco_id), year_dict) in spdf_plts
      for (sy, spdf_gt) in year_dict
        cyc = get(cycle_map, (Int(plot_id), Int(sy)), 0)
        cyc == 0 && continue
        for (sp_eco, rec) in spdf_gt.records
          t4_ref[eco_id][cyc][sp_eco, :] .+= diff([0f0; rec.sp_age_cdf]) .* rec.sp_agb_sum
        end
      end
    end
    t1_ref = nothing
    t2_ref = nothing
  else
    t1_ref = nothing
    t2_ref = nothing
  end
  # Common random numbers: fix the per-rep seed set once so every candidate in a generation is
  # scored on the *same* stochastic realization (required for a fair CMA-ES ranking — a noisier
  # candidate must not win on a lucky seed). Fixing it for the whole run smooths the surface;
  # refreshing per generation to avoid overfitting one realization is a later knob.
  fixed_seeds = _fixed_seeds(rng, n_reps, scoring)
  scales, _ = build_split_scales(loss_params, scale_seed, search_tier, spdf_plts, t4_ref, eco_species_ids, n_species; simb)   # the TRAIN reference scales (RANKW defined here)
  val_scales = have_val ? first(build_split_scales(loss_params, scale_seed, 3, val_spdf_plts, nothing, eco_species_ids, n_species; train=scales, simb)) : nothing

  # Sobol-or-random source for the initial mean and any IPOP restart mean.
  sobol_idx = Ref(1)
  next_candidate() =
    if sobol_idx[] <= length(_sobol_cands)
      p = _sobol_cands[sobol_idx[]]
      @info "Using Sobol candidate $(sobol_idx[])/$(length(_sobol_cands))"
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
        error("start_from params are incompatible with this run: their SPECIES_LIST/ECO_LIST differ from the loaded data (e.g. different stratify_eco_mixed, species tiering, or eco/plot filters). Seeding requires matching ecoregions and species.")
      p
    else
      next_candidate()
    end
    slots = space.slots
    init_run = _agg_reps(fit_params(ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, t4_ref, cycle_map, n_cycles, seeds=fixed_seeds, injection=injection, scales=scales, scoring, simb, plugin=plugin), loss_params, scoring)[1]
    mean0 = PU.params_to_u(bio_params, space, slots)
    init_best = CMAES.CMAESCandidate(bio_params, convert(Float64, PU.get_total_loss(init_run, loss_params)))
    groups = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(space, slots))   # block-diagonal CMA-ES
    @info "CMA-ES block-diagonal: $(length(groups)) covariance blocks (sizes $(length.(groups)))"
    search_state = CMAES.CMAESState(mean0, cmaes_sigma0, init_best, rng; lambda=cmaes_lambda, blocks=groups, max_iter=typemax(Int))
    search_state.n_evals = 1
  else
    @info "Resuming from $resume_from"
    search_state = JLD2.load_object(resume_from)
    bio_params = search_state.best.x
    slots = space.slots
    if force_restart_from_random
      bio_params = next_candidate()
      CMAES.restart!(search_state, PU.params_to_u(bio_params, space, slots), cmaes_sigma0)
    end
  end
  # Hansen-style mixed-integer handling (honors the flag each run, incl. on resume).
  search_state.u_min_std = integer_handling ? PU.integer_u_min_std(space, slots, integer_std_factor) : Float64[]
  if TRIALS < 1 || search_state.n_evals >= TRIALS
    return search_state
  end

  loop, losses_db = scalar_loop_open(search_state; output_dir)
  writer_ch = loop.writer_ch

  stagnation = 0
  barren_restarts = 0             # consecutive IPOP restarts with no new best (barren-restart stop)
  improved_since_restart = false  # any new best found in the current IPOP epoch?
  evals_done = search_state.n_evals
  # Upper bound on generations needed to exhaust the budget (λ never shrinks: IPOP only grows it,
  # which only spends the budget faster). The `evals_done >= TRIALS` break is the real stop.
  est_gens = max(1, cld(TRIALS - evals_done, search_state.lambda))
  run_search_loop(loop; get_state=() -> search_state) do
      TProgress.@track for _gen in 1:est_gens
        evals_done >= TRIALS && break
        xs_u = CMAES.ask(search_state)
        λ = search_state.lambda
        fitnesses = Vector{Float64}(undef, λ)
        gen_best_f = Inf
        local gen_best_params, gen_best_run, gen_best_eco, gen_best_cached
        # SERIAL on purpose: fit_params already parallelizes internally with Threads.@threads :static
        # over sites and deep-copies ref_soa per call. Wrapping this loop in @spawn/@threads would
        # nest :static regions and oversubscribe cores; one fit_params call already saturates threads.
        for k in 1:λ
          cand = PU.u_to_params(xs_u[k], space, slots, bio_params)
          rep_results = fit_params(ref_soa, cand, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, t4_ref, cycle_map, n_cycles, seeds=fixed_seeds, injection=injection, scales=scales, scoring, simb, plugin=plugin, collect_cohorts=n_output_plots>0)
          run_result, eco_losses_k, _ = _agg_reps(rep_results, loss_params, scoring)
          fitnesses[k] = convert(Float64, PU.get_total_loss(run_result, loss_params))
          evals_done += 1
          if fitnesses[k] < gen_best_f
            gen_best_f = fitnesses[k]
            gen_best_params = cand
            gen_best_run = run_result
            gen_best_eco = eco_losses_k
            gen_best_cached = _median_rep_cached(rep_results, loss_params, scoring)
          end
        end

        CMAES.tell!(search_state, fitnesses, xs_u)
        search_state.n_evals = evals_done

        is_new_best = CMAES.note_best!(search_state, CMAES.CMAESCandidate(gen_best_params, gen_best_f))
        stagnation = is_new_best ? 0 : stagnation + 1
        improved_since_restart |= is_new_best

        val_sim_sample = nothing
        if is_new_best
          iter = search_state.best_iteration
          total = convert(Float64, search_state.best.fx)
          @info "New best @ gen $iter | loss=$total | σ=$(search_state.sigma)"
          try
            test_df = simulate_and_test(; splots=splots, bio_params=search_state.best.x, eco_list=eco_list, species_list=species_list, eco_species_ids=eco_species_ids, loss_params=loss_params, site_sim_years=site_sim_years, M=n_reps, no_establishment=no_establishment, rng=_diag_rng(fixed_seeds, :train), plugin=plugin)
            println("Train stats:")
            show(test_df; allrows=true, allcols=true)
            println()
          catch e
            @warn "simulate_and_test (train) failed" exception = (e, catch_backtrace())
          end
          if have_val
            try
              val_reps_ = val_fit_params(search_state.best.x; val_ref_soa, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params,
              injection=inj_val, seeds=fixed_seeds, scales=val_scales, scoring, simb, plugin, collect_cohorts=n_output_plots > 0)   # B29: the fixed common-random-number seeds, like every other driver
              val_agg_ = _agg_reps(val_reps_, loss_params, scoring)
              val_result = (val_agg_[1], val_reps_[val_agg_[3]][2], val_agg_[2])
              val_total = convert(Float64, PU.get_total_loss(val_result[1], loss_params))
              @info "Val loss @ gen $iter | loss=$val_total"
              try
                val_test_df = simulate_and_test(; splots=val_splots, bio_params=search_state.best.x, eco_list=eco_list, species_list=species_list, eco_species_ids=eco_species_ids, loss_params=loss_params, site_sim_years=val_site_sim_years, M=n_reps, no_establishment=no_establishment, rng=_diag_rng(fixed_seeds, :val), plugin=plugin)
                println("Val stats:")
                show(val_test_df; allrows=true, allcols=true)
                println()
              catch e
                @warn "simulate_and_test (val) failed" exception = (e, catch_backtrace())
              end
              val_sim_sample = n_output_plots > 0 ? _filter_cached_to_df(val_result[2], sampled_ids_val) : nothing
            catch e
              @warn "val fit_params failed" exception = (e, catch_backtrace())
            end
          end
          record_best_losses!(losses_db, iter, total, gen_best_run, gen_best_eco, search_state.best.x; eco_list, species_list, n_species, loss_params)
        end
        cached_sites_state_df = _cohorts_df(gen_best_cached)
        sim_sample = (is_new_best && n_output_plots > 0) ? _filter_cached_to_df(gen_best_cached, sampled_ids) : nothing
        put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state), splots, cached_sites_state_df, emp_sample, sim_sample, is_new_best ? emp_sample_val : nothing, val_sim_sample))

        # IPOP restart: re-seed the distribution (with doubled λ) on σ-collapse or stagnation.
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
          CMAES.restart!(search_state, PU.params_to_u(bio_params, space, slots), cmaes_sigma0; lambda=new_lambda)
          stagnation = 0
        end
      end
  end

  return search_state
end
