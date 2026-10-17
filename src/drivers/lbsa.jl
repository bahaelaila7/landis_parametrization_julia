function parametrize_LBSA(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=3, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, scale_seed::PU.LossScales=PU.LossScales(), exec::ExecOptions=ExecOptions(), search::SearchOptions=SearchOptions())
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
  # Common random numbers: fix the per-rep seed set once so every candidate is scored on
  # the *same* stochastic realization. The RNG term then cancels in LBSA's accept/reject
  # comparison, smoothing the surface — vs drawing fresh seeds per trial (high variance).
  # (On resume this is a new realization; the incumbent self-heals after the first accepted
  # move. Refreshing periodically to avoid overfitting one realization is a later knob.)
  fixed_seeds = _fixed_seeds(rng, n_reps, scoring)
  scales, _ = build_split_scales(loss_params, scale_seed, search_tier, spdf_plts, t4_ref, eco_species_ids, n_species; simb)   # the TRAIN reference scales (RANKW defined here)
  val_scales = have_val ? first(build_split_scales(loss_params, scale_seed, 3, val_spdf_plts, nothing, eco_species_ids, n_species; train=scales, simb)) : nothing
  # Score by the SAME Sim-A loss as the (MO-)CMA-ES drivers: under CELL_NORM the cell-normalized aggregate
  # (Σ of _mo_objectives — per-cell ÷scale, per-stratum RANKW), else the raw scalar. So LBSA optimizes the
  # identical objective and its convergence is directly comparable.
  _norm_loss(rr, el) = loss_params.cell_norm ? Float64(sum(_mo_objectives(el, eco_species_ids, loss_params, scales))) : convert(Float64, PU.get_total_loss(rr, loss_params))
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
    init_rr, init_el, _ = _agg_reps(fit_params(ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, t4_ref, cycle_map, n_cycles, seeds=fixed_seeds, injection=injection, scales=scales, scoring, simb, plugin=plugin), loss_params, scoring)
    cur = LBSA.LBSACandidate(bio_params, _norm_loss(init_rr, init_el))
    search_state = LBSA.LBSAState(cur, cur, rng; max_iter=TRIALS, stretch_len=search.lbsa.stretch_len, up_attempt_stale_ratio=search.lbsa.stale_ratio, temp_list_len=search.lbsa.temp_list_len, replace_oldest_instead_of_max=search.lbsa.replace_oldest, cooling_only_schedule=search.lbsa.cooling_only)
    search_state.sobol_cand_idx = 2
  else
    @info "Resuming from $resume_from"
    search_state = JLD2.load_object(resume_from)
    search_state.max_iter = TRIALS
    bio_params = search_state.current.x
    if force_restart_from_random
      search_state._should_restart = true
    end
  end
  if TRIALS < 1 || LBSA.is_search_over(search_state)
    return search_state
  end
  next_candidate() =
    if search_state.sobol_cand_idx <= length(_sobol_cands)
      p = _sobol_cands[search_state.sobol_cand_idx]
      @info "Using Sobol candidate $(search_state.sobol_cand_idx)/$(length(_sobol_cands))"
      search_state.sobol_cand_idx += 1
      p
    else
      generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    end

  loop, losses_db = scalar_loop_open(search_state; output_dir)
  writer_ch = loop.writer_ch

  # Dense convergence + restart/reheat trace: one row per trial — best-so-far & incumbent loss, max
  # temperature, the frozen-reheat counter, and a restart flag (1 when a restart fired this trial).
  # On resume, APPEND (keep the prior trace) and skip the header.
  conv_io = open(joinpath(output_dir, "convergence.csv"), isnothing(resume_from) ? "w" : "a")
  isnothing(resume_from) && (println(conv_io, "trial,best_loss,current_loss,t_max,frozen_reheats,restart"); flush(conv_io))

  baseline_weights = diff([0.0; space.dists.weights_cumsum])
  param_sensitivities = zeros(length(baseline_weights))
  sensitivity_decay = 0.999      # per-trial decay toward 0 for all params
  sensitivity_ema_alpha = 0.1    # EMA weight for chosen param's sensitivity update
  sensitivity_lambda = 3.0       # max weight boost at full sensitivity (baseline × (1 + λ))

  run_search_loop(loop; get_state=() -> search_state, close_extra=() -> close(conv_io), unit="trial") do
      TProgress.@track for trial in (search_state.i+1):TRIALS
        dynamic_weights = baseline_weights .* (1.0 .+ sensitivity_lambda .* param_sensitivities)
        dynamic_weights ./= sum(dynamic_weights)
        dynamic_cumsum = cumsum(dynamic_weights)
        bio_params, chosen_param_idx = PU.mutate_params(bio_params, space; rng=rng, mutation_mode=PU.BothMutations)
        #ctx=PU.SamplingContext(search_state.current.fx.sp_w_loss .+ search_state.current.fx.sp_agb_loss .+ (search_state.current.fx.sp_w_loss .* search_state.current.fx.sp_agb_loss)),
        #ctx=PU.SamplingContext(search_state.current.fx.sp_w_loss),
        # dynamic_cumsum=dynamic_cumsum)
        for _ in 0:rand(rng, 0:2)
          bio_params, _ = PU.mutate_params(bio_params, space; rng=rng, mutation_mode=PU.BothMutations)#,  #BothMutations,
        end

        rep_results = fit_params(ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, t4_ref, cycle_map, n_cycles, seeds=fixed_seeds, injection=injection, scales=scales, scoring, simb, plugin=plugin, collect_cohorts=n_output_plots>0)
        run_result, eco_losses, _ = _agg_reps(rep_results, loss_params, scoring)
        cached_sites_state = _median_rep_cached(rep_results, loss_params, scoring)
        next_loss = _norm_loss(run_result, eco_losses)

        current_loss = convert(Float64, search_state.current.fx)
        delta_loss = abs(next_loss - current_loss)
        norm_delta = delta_loss / (current_loss + 1e-10)
        param_sensitivities .*= sensitivity_decay
        param_sensitivities[chosen_param_idx] = (1.0 - sensitivity_ema_alpha) * param_sensitivities[chosen_param_idx] + sensitivity_ema_alpha * min(norm_delta, 1.0)

        next = LBSA.LBSACandidate(bio_params, next_loss)

        is_new_best = LBSA.search_cmp!(next, search_state)
        #if is_new_best
        #  put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state), merged_sites_state))
        #  plot(cached_sites_state)
        #end
        did_restart = LBSA.should_restart(search_state)
        if did_restart
          @info "Restarting @ $(search_state.i)"
          bio_params = next_candidate()
          rst_rr, rst_el, _ = _agg_reps(fit_params(ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, t4_ref, cycle_map, n_cycles, seeds=fixed_seeds, injection=injection, scales=scales, scoring, simb, plugin=plugin), loss_params, scoring)
          is_new_best = LBSA.restart(search_state, LBSA.LBSACandidate(bio_params, _norm_loss(rst_rr, rst_el)))
        end
        val_sim_sample = nothing
        if is_new_best
          iter = search_state.best_iteration
          total = convert(Float64, search_state.best.fx)
          @info "New best @ $iter | loss=$total"
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
              @info "Val loss @ $iter | loss=$val_total"
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
          record_best_losses!(losses_db, iter, total, run_result, eco_losses, search_state.best.x; eco_list, species_list, n_species, loss_params, print_breakdown=true)
        end
        let tmax = (isempty(search_state._t_list) || search_state._t_max_idx > length(search_state._t_list)) ? NaN : search_state._t_list[search_state._t_max_idx]
          println(conv_io, string(trial, ",", convert(Float64, search_state.best.fx), ",", convert(Float64, search_state.current.fx), ",", tmax, ",", search_state._frozen_no_best_reheats, ",", did_restart ? 1 : 0))
          (trial % 50 == 0 || did_restart) && flush(conv_io)
        end
        if is_new_best || search_state.i % 50 == 0
          cached_sites_state_df = _cohorts_df(cached_sites_state)
          sim_sample = (is_new_best && n_output_plots > 0) ? _filter_cached_to_df(cached_sites_state, sampled_ids) : nothing
          put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state), splots, cached_sites_state_df, emp_sample, sim_sample, is_new_best ? emp_sample_val : nothing, val_sim_sample))
        end
        if LBSA.is_search_over(search_state)
          break
        end
        #println(convert(Float64,run_result))
      end
  end

  return search_state
end
