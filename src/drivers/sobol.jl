function parametrize_sobol(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, rng::Random.AbstractRNG, debug::Bool, N::Int=100, M::Int=5, eval_tier::Int=3, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, cycle_years::Real=8, saltelli::Bool=false, scale_seed::PU.LossScales=PU.LossScales())
  n_species = length(species_list)
  n_ecoregions = length(eco_list)
  max_sim_year = site_sim_years.sim_years .|> maximum |> maximum

  t4_ref = nothing
  cycle_map = nothing
  n_cycles = 0
  if eval_tier == 1
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
  elseif eval_tier == 2
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
  elseif eval_tier == 4 || eval_tier == 5 || simb.dual_mode != :off
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

  initial_params = generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
  injection = build_injection_plan(injection_cohorts, ref_soa, eco_species_ids, loss_params, scoring)   # this reference's injection dict, shock and exclusions
  samples, sal_tags = saltelli ? PU.saltelli_design(space, initial_params, N) :
                                  (PU.sobol_samples(space, initial_params, N), nothing)
  n_samples = length(samples)
  saltelli && @info "Saltelli design: N=$N base, $n_samples total evals (A+B+ABₖ over $(div(n_samples,N)-2) param-type groups)"

  # IDENTICAL eval to the downstream optimizers: dual Sim A⊕B, init-perturbation reps (best-of for A),
  # B median-over-distinct-seeds — i.e. a candidate's loss here == its CMA-MAE/CMA-ES quality, so the
  # seed pool ranks on the same loss surface. (M = n_reps perturbation reps via _fixed_seeds.)
  fixed_seeds = _fixed_seeds(rng, M, scoring)
  dual_b0 = simb.dual_mode == :off ? nothing : _build_dual_b(splots, eco_species_ids, eco_list, spdf_plts, loss_params, t4_ref, cycle_map, n_cycles, injection_cohorts, spinup_cohorts, rng, no_establishment; b_only=(simb.dual_mode == :b), scoring, simb, plugin=plugin)
  scales, dual_b = build_split_scales(loss_params, scale_seed, eval_tier, spdf_plts, t4_ref, eco_species_ids, n_species; dual_b=dual_b0, simb)   # the TRAIN reference scales (RANKW defined here)
  _run(p) = fit_params(ref_soa, p, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params;
    debug=debug, search_tier=eval_tier, t1_ref=t1_ref, t2_ref=t2_ref, t4_ref=t4_ref, cycle_map=cycle_map, n_cycles=n_cycles,
    seeds=fixed_seeds, injection=injection, dual_b=dual_b, scales=scales, scoring, simb, plugin=plugin)

  ResultT = @NamedTuple{params::typeof(initial_params), mean_loss::FloatType, std_loss::FloatType, median_loss::FloatType, losses::Vector{FloatType}}
  results = ResultT[]

  losses_db_file = DuckDB.DB(joinpath(output_dir, "losses.duckdb"))
  losses_db = DuckDB.connect(losses_db_file)
  # store ΣW and ΣAGB (raw sums over all eco×species) so W/AGB-only UMAPs need no re-evaluation.
  if saltelli
    DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS saltelli_results (run_id VARCHAR, idx INTEGER, tag VARCHAR, mean_loss DOUBLE, sumW DOUBLE, sumAGB DOUBLE, params_blob BLOB)")
  else
    DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS sobol_results (run_id VARCHAR, sobol_idx INTEGER, mean_loss DOUBLE, std_loss DOUBLE, median_loss DOUBLE, sumW DOUBLE, sumAGB DOUBLE, params_blob BLOB, objs_blob BLOB)")
  end
  run_id = string(Dates.now())

  try
    TProgress.@track for i in 1:n_samples
      bio_params = samples[i]
      run_result, eco_losses, _ = _agg_reps(_run(bio_params), loss_params, scoring)    # A best-of-perturbation + B median → one aggregate
      objs = _mo_objectives(eco_losses, eco_species_ids, loss_params, scales, dual_b)   # EXACT downstream objective (cell-norm ÷scale, rank, power)
      loss = FloatType(_mo_aggregate(objs, run_result, loss_params))   # = Σobjs under cell-norm; get_total_loss otherwise
      sumW = sum(@view objs[1:2:end]); sumA = sum(@view objs[2:2:end])   # W = odd objs, AGB = even (A_W,A_AGB,B_W,B_AGB)
      push!(results, (; params=bio_params, mean_loss=loss, std_loss=zero(FloatType), median_loss=loss, losses=FloatType[loss]))
      buf = IOBuffer(); Serialization.serialize(buf, bio_params)
      obuf = IOBuffer(); Serialization.serialize(obuf, Float64.(objs))   # full objective vector (A_W,A_AGB[,B_W,B_AGB])
      if saltelli
        DuckDB.execute(losses_db, "INSERT INTO saltelli_results VALUES (?, ?, ?, ?, ?, ?, ?)",
          [run_id, i, sal_tags[i], Float64(loss), Float64(sumW), Float64(sumA), take!(buf)])
      else
        DuckDB.execute(losses_db, "INSERT INTO sobol_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
          [run_id, i, Float64(loss), 0.0, Float64(loss), Float64(sumW), Float64(sumA), take!(buf), take!(obuf)])
      end
    end
  finally
    close(losses_db_file)
    isempty(results) && return results
    sort!(results, by=x -> x.mean_loss)
    out_path = joinpath(output_dir, "sobol_results@$(length(results))of$(N)x$(M).jld2")
    JLD2.save_object(out_path, results)
    @info "Saved sobol results → $out_path (best mean_loss = $(results[1].mean_loss) ± $(results[1].std_loss), median = $(results[1].median_loss))"

    if n_output_plots > 0 && !isempty(results)
      splots.sim_year .= Dates.value.(Dates.Day.(splots.measdate .- splots.start_measdate)) ./ 365.25 .|> round .|> Int
      sampled_ids = _sample_plot_ids(UIntType.(unique(splots.plot_id)), n_output_plots, rng; injection_cohorts=injection_cohorts)
      emp_sample = _make_emp_df(splots, sampled_ids)
      best_params = results[1].params
      best_soa = make_sites_like(plugin, splots, eco_species_ids; rng, spinup, no_establishment=no_establishment)
      best_result = only(fit_params(best_soa, best_params, site_sim_years.sim_years .|> maximum |> maximum,
        length(species_list), eco_species_ids, spdf_plts,
        site_sim_years, spinup, spinup_cohorts, loss_params;
        debug, search_tier=eval_tier, t1_ref=t1_ref, t2_ref=t2_ref, t4_ref=t4_ref, cycle_map=cycle_map, n_cycles=n_cycles,
        seeds=[rand(rng, UInt64)],
        injection=injection, scales=scales, scoring, simb, plugin=plugin, collect_cohorts=true))
      sim_sample = _filter_cached_to_df(best_result[2], sampled_ids)
      generate_plots(emp_sample, sim_sample, "sobol_best", convert(Float64, PU.get_total_loss(best_result[1], loss_params)), output_dir)
    end
  end
  return results
end
