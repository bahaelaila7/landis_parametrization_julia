include("tier_acc.jl")

# Initial-biomass perturbation of one rep (sim-year-0 population; not scored, only propagated): perturbed = max(2, bio + clamp(bio*(sc-1), +-cap)),
# the 2-floor hard-coded, sc = 1 + pct.
function _perturb_init!(pc, soa, sc::FloatType, cap)
  @maybe_threads soa.parallel for i in 1:soa.n     # per-site independent -> parallel in SoA mode
    site = getsite(soa, i)
    site.active || continue
    @inbounds for j in 1:Int(site.live)
      b = site.c_bio[j]
      set_cohort_bio!(pc, site, j, max(FloatType(2), b + clamp(b * (sc - one(FloatType)), -cap, cap)))
    end
  end
  nothing
end

# Sim B (free) disturbance: apply the observed exogenous biomass drop to the free sim — reduce a free-sim cohort's biomass by the observed drop
# fraction ONLY IF it matches the disturbed cohort in SPECIES AND AGE (per (plot,year,species,age)). The model responds to disturbance but does not
# predict it; no cohort injection (stays free regen). Unmatched disturbed cohorts (the model didn't grow them) are not applied.
function _apply_free_disturbance!(pc, soa, yd)
  @maybe_threads soa.parallel for i in 1:soa.n
    @inbounds begin
      site = getsite(soa, i)
      spd = get(yd, Int(site.mapcode), nothing)
      spd === nothing && continue
      j = 1; removed = false
      while j <= Int(site.live)
        d = get(spd, (site.c_species[j], UIntType(round(site.c_age[j]))), zero(FloatType))
        if d >= one(FloatType)        # FULL removal by disturbance (export): the cohort is gone, stop growing it
          _shock_remove_cohort!(site, j); removed = true
          continue                    # slot j now holds the swapped-in cohort
        elseif d > zero(FloatType)
          drop = site.c_bio[j] * d
          set_cohort_bio!(pc, site, j, site.c_bio[j] - drop)
          site.B -= drop
        end
        j += 1
      end
      if removed
        o = zero(UIntType)
        for q in 1:Int(site.live); site.c_age[q] > one(FloatType) && (o += one(UIntType)); end
        site.old = o
      end
    end
  end
  nothing
end

# The simulated years of one rep with the tier accumulator `acc` and the cohort logs `pre_log` (pre-injection snapshot) / `post_log` (after scoring).
# This is the function barrier of fit_params: `acc`, the logs and `sc` are concrete here, so the site loop (`_score_sites!`) is specialised on the tier and
# on whether cohorts are logged. Order of a year: step, Sim B disturbance, shock, snapshot, legacy injection, death / W diagnostics, scoring, new-order
# injection. Returns the rep's (run_result, cohorts, eco_losses) and adds the rep's tier-4 histogram to `mor` when scoring the mean over reps.
function _run_rep!(acc::TierAcc, pre_log, post_log, SP, soa, ctx, pc, sc::YearScoring, mor, r::NamedTuple)
  scoring = r.scoring; injection_dict = r.injection_dict; injection_years = r.injection_years
  for current_sim_year in r.starting_sim_year:r.max_sim_year
    if r.timer !== nothing && soa.parallel   # only in soa mode: candidate mode runs fit_params on many threads -> racy timer.sim
      _ts = time_ns()
      step!(SP, soa, current_sim_year, ctx)
      r.timer.sim += time_ns() - _ts
    else
      step!(SP, soa, current_sim_year, ctx)
    end
    if r.disturbance_dict !== nothing && current_sim_year in r.disturbance_years
      _apply_free_disturbance!(pc, soa, r.disturbance_dict[current_sim_year])
    end
    if r.do_shock                              # disturbance shock: removals + (:scale) partial drops, BEFORE the preinject snapshot and all scoring
      _shy = r.shock === nothing ? nothing : get(r.shock, current_sim_year, nothing)
      _inj = current_sim_year in injection_years ? injection_dict[current_sim_year] : nothing
      (_shy === nothing && _inj === nothing) || _apply_disturbance_shock!(pc, soa, _shy, _inj, scoring)
    end
    _snapshot_pre!(pre_log, soa, current_sim_year, sc.sim_years)    # pure model prediction BEFORE any injection/override
    if !scoring.loss_before_injection && !isnothing(injection_dict) && current_sim_year in injection_years   # legacy order: inject, then score
      _inject_observed_cohorts!(pc, soa, injection_dict[current_sim_year], scoring)
    end
    if r.do_shock && current_sim_year in injection_years
      r.death_acc === nothing || _death_collect!(r.death_acc, soa, current_sim_year, injection_dict, r.excl_plan)
      scoring.w_scope === :survivors && _survivors_only!(soa, current_sim_year, injection_dict)
    end
    (scoring.w_diag !== :none && scoring.loss_before_injection && r.do_shock && current_sim_year in injection_years) && _w_diag_apply!(pc, soa, current_sim_year, injection_dict, r.excl_plan, scoring, r.w_diag === nothing ? WDiag() : r.w_diag)
    begin_year!(acc, soa)
    _score_sites!(acc, post_log, soa, current_sim_year, sc)
    end_year!(acc, soa, current_sim_year)
    if scoring.loss_before_injection && !isnothing(injection_dict) && current_sim_year in injection_years   # score the simulated state first, then inject for the next interval
      _inject_observed_cohorts!(pc, soa, injection_dict[current_sim_year], scoring; skip_scale=r.skip_scale)
    end
  end
  run_result, eco_losses = finish!(acc)
  _mor_add!(mor, acc)
  (run_result, _cohorts(pre_log, post_log), eco_losses)
end

"The parameters with the fitted, score-only PROB_MORT zeroed: the (deterministic) simulation of a Sim A run with a death term applies no stochastic mortality."
_without_mortality(bp) = PU.Setfield.@set(bp.PROB_MORT_SPP = [zero.(v) for v in bp.PROB_MORT_SPP])

# One rep: its SoA (copy / reset of the reference, optional initial-biomass perturbation), the plugin context, the spinup, then `_rep_sim`.
# `ec` bundles the call's invariants (a NamedTuple, so every field is concretely typed in the specialised callee).
function _fit_rep(ec::NamedTuple, seed, init_scale::FloatType)
  pc = ec.pc
  # this evaluation's site-loop parallelism (false inside a candidate-parallel batch) is set on the SoA it owns BEFORE anything threads over it
  # (reset_soa! loops over the CSR arrays with dst.parallel); no process state.
  ec.work_soa === nothing || (ec.work_soa.parallel = ec.parallel_sites)
  soa0 = ec.work_soa === nothing ? copy_and_reseed_soa(ec.ref_soa, seed) : reset_soa!(ec.work_soa, ec.ref_soa; seed=seed)
  soa0.parallel = ec.parallel_sites
  init_scale != one(FloatType) && _perturb_init!(pc, soa0, init_scale, ec.scoring.init_perturb_cap)   # not scored, only propagated
  SP = succession_plugin_of(ec.ref_soa)
  # death_acc !== nothing => Sim A with a death term: PROB_MORT is a fitted score-only parameter, NOT applied in the (deterministic) simulation
  eco_params = generate_eco_params(pc, ec.death_acc === nothing || !hasproperty(ec.bio_params, :PROB_MORT_SPP) ? ec.bio_params : _without_mortality(ec.bio_params))
  ctx = make_ctx(pc, eco_params)
  soa = ec.spinup ? spinup!(SP, soa0, ec.spinup_cohorts, eco_params) : soa0
  _rep_sim(ec, SP, soa, ctx, eco_params)
end
function _rep_sim(ec::NamedTuple, SP, soa, ctx, eco_params)
  starting_sim_year = ec.spinup ? 0 : 1   # with spinup, even the first year of vegetation data is to be matched and compared
  r = merge(ec.run, (; starting_sim_year))
  acc = make_tier_acc(ec.tier, soa, ec.sc, merge(ec.acc_args, (; starting_sim_year)))
  pre_log = ec.cache_preinject ? CohortLog(soa.n) : NoCohortLog()
  post_log = (!ec.cache_preinject && ec.collect_cohorts) ? CohortLog(soa.n) : NoCohortLog()
  _run_rep!(acc, pre_log, post_log, SP, soa, ctx, ec.pc, ec.sc, ec.mor, r)
end

const RepResult = Tuple{PU.SiteLoss,Vector{CohortRow},Vector{PU.SiteLoss}}

"""
    fit_params(ref_soa, bio_params, ...; search_tier, seeds, injection, ...) -> Vector{(run_result, cohorts, eco_losses)}

One evaluation of `bio_params`: per seed (rep) a simulation of `ref_soa` scored at the tier `search_tier`; `[(run_result::SiteLoss, cohorts, eco_losses)]`.
`cohorts` is the simulated state at the measured years as `(ref_cn, year, species, age, biomass)` rows: the PRE-injection snapshot with
`cache_preinject=true`, else the post-scoring state with `collect_cohorts=true`, else empty (building it costs a push per cohort per measurement,
so only the evaluations that plot / diagnose ask for it).
"""
function fit_params(ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier::Int=3, t1_ref::Union{Nothing,Vector{Matrix{FloatType}}}=nothing, t2_ref=nothing, seeds::AbstractVector=[nothing], injection::Union{Nothing,InjectionPlan}=nothing, injection_cohorts=nothing, t4_ref=nothing, cycle_map=nothing, n_cycles::Int=0, dual_b=nothing, disturbance_dict=nothing, disturbance_years=Set{Int}(), cache_preinject::Bool=false, collect_cohorts::Bool=false, work_soa=nothing, death_acc::Union{Nothing,DeathAcc}=nothing, cell_plots=nothing, scales::Union{Nothing,PU.LossScales}=nothing, scoring::ScoringOptions, simb::SimBOptions, plugin::PluginConfig, parallel_sites::Bool=true, timer::Union{Nothing,RunTimer}=nothing, w_diag::Union{Nothing,WDiag}=nothing, capture_t4::Union{Nothing,T4Capture}=nothing)
  pc = plugin.ctx   # the plugin context (data tables / priors / climate): explicit, never module state
  # `scales`: the normalisation of THIS reference (see compute_loss_scales / build_split_scales); computed here from the reference when the caller
  # has none (one-off evaluations, analysis tools). Sim B's (dual_b.scales) are derived from it the same way when absent.
  # tier-4 per-plot-mean counts of THIS split: explicit, else those of its Sim B (the counts of the tier-4 reference the primary shares cycle bins with)
  cell_plots_ = cell_plots === nothing && dual_b !== nothing ? get(dual_b, :cell_plots, nothing) : cell_plots
  scales_ = scales === nothing ? compute_loss_scales(loss_params, search_tier, spdf_plts, t4_ref, eco_species_ids, n_species; cell_plots=cell_plots_, per_plot_mean=simb.t4_per_plot_mean) : scales
  dual_b_ = dual_b !== nothing && !haskey(dual_b, :scales) ?
    merge(dual_b, (scales=compute_loss_scales(dual_b.loss_params, dual_b.tier, dual_b.spdf_plts, dual_b.t4_ref, eco_species_ids, n_species; cell_plots=dual_b.cell_plots, per_plot_mean=simb.t4_per_plot_mean, rankw=scales_.rankw, softplus_cell_from=scales_),)) : dual_b
  search_tier == 4 && simb.t4_per_plot_mean && !isempty(get(ENV, "PAN_LOG_CELLPLOTS", "")) && @info "tier-4 per-plot-mean cell plots: Σ = $(sum(sum, something(cell_plots_, [Int[]])))"
  # The injection of the observed cohorts, the disturbance shock and the loss exclusions of THIS reference come from its InjectionPlan
  # (built once per reference by `build_injection_plan`); callers that have only the raw injection cohorts (tools, probes) may pass them
  # as `injection_cohorts` and the plan is built here.
  plan = injection === nothing ? build_injection_plan(injection_cohorts, ref_soa, eco_species_ids, loss_params, scoring) : injection
  plan === nothing || _check_plan_options(plan, scoring)
  injection_dict = plan === nothing ? nothing : plan.dict
  injection_years = plan === nothing ? Set{Int}() : plan.years
  # Disturbance shock (new order only): needs the sync injection (the observed set defines which cohorts were disturbed).
  do_shock = shock_order(scoring) && injection_dict !== nothing
  shock = do_shock ? plan.shock : nothing
  excl_plan = do_shock ? plan.excl : nothing            # cohort-level exclusion (new order)
  legacy_excl = plan === nothing ? nothing : plan.legacy_excl   # species-level exclusion (legacy order)
  # The kwargs a recursive Sim B evaluation shares with this call.
  b_common = (; debug, scoring, simb, plugin, parallel_sites, timer, w_diag, capture_t4)
  # B-only mode: skip Sim A entirely and evaluate only Sim B (free process), seeded from A's params.
  if dual_b_ !== nothing && get(dual_b_, :b_only, false)
    b_seeds = UInt64[hash((UInt64(seeds[1]), i)) for i in eachindex(seeds)]   # distinct seeds for the stochastic free sim
    resB = fit_params(dual_b_.ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, dual_b_.spdf_plts,
      site_sim_years, dual_b_.spinup, dual_b_.spinup_cohorts, dual_b_.loss_params; b_common..., search_tier=dual_b_.tier,
      t4_ref=dual_b_.t4_ref, cycle_map=dual_b_.cycle_map, n_cycles=dual_b_.n_cycles, seeds=b_seeds,
      injection=dual_b_.injection, collect_cohorts=(collect_cohorts || cache_preinject),   # B is the result here: its post-scoring cohorts (cache_preinject is not passed on)
      disturbance_dict=dual_b_.disturbance_dict, disturbance_years=dual_b_.disturbance_years, dual_b=nothing, cell_plots=dual_b_.cell_plots, scales=dual_b_.scales)
    b_losses = Float64[convert(Float64, PU.get_total_loss(r[1], loss_params)) for r in resB]
    bi = argmin(abs.(b_losses .- Statistics.median(b_losses)))   # B's median rep
    return [resB[bi]]
  end
  init_scales = _init_scales(length(seeds), scoring)   # per-rep initial-biomass scale (identity unless perturbing)
  # Mean-over-reps (tier-4 free sim): accumulate every rep's sim histogram, then score ONE loss on the mean
  # histogram (loss-of-mean = the model's EXPECTED distribution) instead of a median-rep pick. Reps run
  # sequentially, so the shared accumulator is race-free.
  mor = simb.t4_mean_over_reps && search_tier == 4 && dual_b_ === nothing && length(seeds) > 1 ?
    MeanOverReps([[zeros(FloatType, length(eco_species_ids[e]), size(t4_ref[1][1], 2)) for _ in 1:n_cycles] for e in eachindex(eco_species_ids)], nothing) : nothing
  sc = YearScoring(site_sim_years.sim_years, spdf_plts, cycle_map, legacy_excl, excl_plan, eco_species_ids, n_species, loss_params, scales_, debug)
  ec = (; pc, ref_soa, work_soa, parallel_sites, scoring, death_acc, bio_params, spinup, spinup_cohorts, tier=search_tier, cache_preinject, collect_cohorts, sc, mor,
    acc_args=(; max_sim_year, t1_ref, t2_ref, t4_ref, n_cycles, cell_plots=cell_plots_, per_plot_mean=simb.t4_per_plot_mean, capture_t4),
    run=(; scoring, injection_dict, injection_years, do_shock, shock, excl_plan, skip_scale=do_shock, disturbance_dict, disturbance_years, death_acc, w_diag, timer, max_sim_year))
  resA = RepResult[]
  for ri in eachindex(seeds)
    push!(resA, _fit_rep(ec, seeds[ri], init_scales[ri]))
  end
  # Dual-mode: also run Sim B (free process, no_establish baked into dual_b.ref_soa, tier-4) and stack
  # its per-eco losses after Sim A's via vcat — _mo_objectives turns this into A⊕B objectives.
  # Sim B is STOCHASTIC (free regen via site.rng), so it gets DISTINCT seeds per rep (A's perturbation
  # reps share one seed; reusing it would make B's reps identical and its spinup ignores the perturbation).
  # Aggregate A best-of-perturbation (via _agg_reps), B by the MEDIAN total (W+AGB-bin) loss over reps,
  # then combine A_best + B_median into ONE result (the SO loss = A_best + B_median).
  if mor !== nothing   # replace the per-rep results with ONE loss scored on the mean sim histogram
    ns = length(seeds)
    for e in eachindex(eco_species_ids), cyc in 1:n_cycles
      mor.hist[e][cyc] ./= ns
    end
    sc_, oc_ = mor.counts
    eco_losses_m = calculate_t4_loss(mor.hist, t4_ref, loss_params, scales_, n_species, eco_species_ids, sc_, oc_, n_cycles; cell_plots=cell_plots_, per_plot_mean=simb.t4_per_plot_mean, capture=capture_t4)
    resA = RepResult[(sum(eco_losses_m), _median_rep_cached(resA, loss_params, scoring), eco_losses_m)]
  end
  dual_b_ === nothing && return resA
  b_seeds = UInt64[hash((UInt64(seeds[1]), i)) for i in eachindex(seeds)]
  resB = fit_params(dual_b_.ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, dual_b_.spdf_plts,
    site_sim_years, dual_b_.spinup, dual_b_.spinup_cohorts, dual_b_.loss_params; b_common..., search_tier=dual_b_.tier, t4_ref=dual_b_.t4_ref,
    cycle_map=dual_b_.cycle_map, n_cycles=dual_b_.n_cycles, seeds=b_seeds, injection=dual_b_.injection,
    disturbance_dict=dual_b_.disturbance_dict,
    disturbance_years=dual_b_.disturbance_years, dual_b=nothing, cell_plots=dual_b_.cell_plots, scales=dual_b_.scales)
  a_run, a_eco, a_idx = _agg_reps(resA, loss_params, scoring)
  b_losses = Float64[convert(Float64, PU.get_total_loss(r[1], loss_params)) for r in resB]
  bi = argmin(abs.(b_losses .- Statistics.median(b_losses)))   # B's median rep (robust to a lucky/unlucky draw)
  return RepResult[(a_run + resB[bi][1], resA[a_idx][2], vcat(a_eco, resB[bi][3]))]
end
