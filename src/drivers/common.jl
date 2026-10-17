function _sample_plot_ids(all_ids::Vector{UIntType}, n::Int, rng;
  injection_cohorts::Union{Nothing,DataFrame}=nothing)
  n == 0 && return Set{UIntType}()
  if isnothing(injection_cohorts) || isempty(injection_cohorts)
    return Set(Random.shuffle(rng, all_ids)[1:min(n, length(all_ids))])
  end
  inj_set = Set(UIntType.(injection_cohorts.plot_id))
  with_inj = filter(id -> id in inj_set, all_ids)
  without_inj = filter(id -> !(id in inj_set), all_ids)
  n_inj = min(n ÷ 2, length(with_inj))
  n_clean = min(n - n_inj, length(without_inj))
  n_inj = min(n - n_clean, length(with_inj))   # backfill if clean side was small
  Set(vcat(
    Random.shuffle(rng, with_inj)[1:n_inj],
    Random.shuffle(rng, without_inj)[1:n_clean],
  ))
end

function _make_emp_df(splots::DataFrame, sampled_ids::Set)
  extra = hasproperty(splots, :subp) ? [:subp] : Symbol[]
  select(
    filter(row -> row.plot_id in sampled_ids, splots),  # include sim_year=0 (initial condition)
    :plot_id, :sim_year, :eco_species_id => :species_id, :age_calc, :agb_sum,
    :statecd, :unitcd, :countycd, :plot, extra..., :effective_species,
  )
end

# Initial-condition perturbation (ScoringOptions.init_perturb_frac / _cap). When init_perturb_frac > 0 the n_reps "reps" are NOT RNG
# re-seeds but evenly-spaced scalings of the INITIAL cohort biomass over [1-frac, 1+frac] (the sim-year-0
# population only — year 0 is never scored, so this is pure boundary-condition slack), and an
# individual's fitness is the BEST (min) rep, not the sum. frac=0 ⇒ legacy behavior (seed-varied reps,
# summed). fit_params reads the options off the number of seeds, so callers need no new args.
# init_perturb_cap is an optional absolute cap (g/m²) on the per-cohort biomass CHANGE: the perturbation is bio*(scale-1)
# clamped to ±cap, so e.g. 5% on a 4000 g/m² cohort is limited to ±100 instead of ±200. Inf ⇒ uncapped (pure multiplicative scaling).
# n evenly-spaced initial-biomass scale factors across [1-frac, 1+frac]; identity when off or n≤1.
_init_scales(n::Int, opts::ScoringOptions) = (opts.init_perturb_frac <= 0 || n <= 1) ? fill(one(FloatType), n) :
  FloatType[1 - opts.init_perturb_frac + 2 * opts.init_perturb_frac * (k - 1) / (n - 1) for k in 1:n]
# Per-rep RNG seeds. Perturb mode shares ONE seed across reps so the only thing that varies is the
# initial-biomass scaling (sync path has no RNG influence anyway, but this makes it exact); legacy
# draws an independent seed per rep.
# B33: the rng of the diagnostic `simulate_and_test` (new-best stats) — derived from the fixed seeds, never the run rng,
# so the search trajectory does not depend on whether diagnostics / val are enabled.
_diag_rng(seeds::AbstractVector{UInt64}, tag::Symbol) = Random.MersenneTwister(hash((String(tag), Tuple(seeds))))
_fixed_seeds(rng, n::Int, opts::ScoringOptions) = opts.init_perturb_frac > 0 ? fill(rand(rng, UInt64), n) : [rand(rng, UInt64) for _ in 1:n]

# Aggregate per-rep results into (run_result, eco_losses, picked_idx). Legacy (frac=0): SUM across reps.
# Perturb mode (frac>0 & >1 rep): pick the SINGLE BEST rep (min total loss).
function _agg_reps(rep_results, loss_params::PU.LossParams, opts::ScoringOptions)
  if opts.init_perturb_frac > 0 && length(rep_results) > 1
    b = argmin(Float64[convert(Float64, PU.get_total_loss(r[1], loss_params)) for r in rep_results])
    return rep_results[b][1], rep_results[b][3], b
  end
  run_result = sum(r[1] for r in rep_results)
  eco_losses = isnothing(rep_results[1][3]) ? nothing : [sum(r[3][e] for r in rep_results) for e in eachindex(rep_results[1][3])]
  return run_result, eco_losses, 1
end

function _median_rep_cached(rep_results, loss_params::PU.LossParams, opts::ScoringOptions)
  losses = Float64[convert(Float64, PU.get_total_loss(r[1], loss_params)) for r in rep_results]
  # perturb mode caches the BEST rep's trajectory (matching _agg_reps); legacy caches the median rep.
  (opts.init_perturb_frac > 0 && length(rep_results) > 1) && return rep_results[argmin(losses)][2]
  rep_results[argmin(abs.(losses .- Statistics.median(losses)))][2]
end

# Dual-mode (Sim A sync + Sim B free; SimBOptions.dual_mode). When dual_mode is not :off: establishment params are fit, and each
# candidate is evaluated with a second simulation (free process: model regenerates natural plots, planting
# handed in for artificial plots) whose tier-4 per-eco losses are stacked after Sim A's via fit_params'
# dual_b. Errors attribute to the process each param governs (growth←A, introduction←B).
# dual_mode: :off (Sim A only) · :joint (A⊕B) · :b (Sim B only, seed from A). t4_primary: see SimBOptions (src/options.jl).
# SimBOptions.simb_spinup (yaml simB_spinup): Sim B start; true = spin up from a deficit (back-cast);
#                       false = start from the OBSERVED year-0 cohorts (anchored), still free establishment
# SimBOptions.simb_disturb_only (yaml simB_disturb_only): Sim B is PURELY FREE — NO cohort injection anywhere (natural AND artificial run free
# from the year-0 initial cohorts, model establishment via PROB_ESTAB stays ON), and the observed disturbance
# drops are applied as scale-reductions to EVERY plot's matching-species cohorts (no artificial exclusion, no
# planting sync). Contrast the default: natural free + artificial planting-synced + disturbance excluded on artificial.
# Tier-4 (Sim B) loss options. t4_per_plot_mean (simB_per_plot_mean): divide each (eco,cycle,species) cell histogram by its plot count →
# AGB term is mean biomass-per-plot-by-age (intensive; W is CDF-normalized so unaffected). t4_mean_over_reps (simB_rep_mean): for
# the stochastic free sim, average the n_reps' tier-4 sim histograms and score the loss ONCE on the mean
# histogram (loss-of-mean, the model's EXPECTED distribution) instead of picking a median rep.

# ── Optional wall-time instrumentation (ExecOptions.timing, env PAN_TIMING=1; see RunTimer in options.jl). With timing off no RunTimer exists
# and every probe is a `timer === nothing` branch; the only always-on work is a handful of time_ns() per GENERATION (not per site).
# Prints, per generation, where the wall goes: startup→first-gen, then ask / eval(sim vs loss+other) / tell /
# finalize(io = time BLOCKED handing checkpoints to the writer — a large io means writer backpressure, i.e. the
# ramdisk-output lever would help). sim is accumulated across the gen's candidate evals in fit_params.
_tsec(ns) = round(max(0, signed(UInt64(ns))) / 1e9, digits=3)   # signed+clamp: never throws on an underflowed diff
# Per-(eco,cycle) distinct-plot counts of the tier-4 reference are computed once in _build_dual_b and carried as dual_b.cell_plots (read by BOTH the AGB
# scale `_cell_scales` and the loss `calculate_t4_loss`, so per-plot-mean stays consistent).
# Build Sim B's dual_b NamedTuple: free-process SoA + injection restricted to artificial ecos (natural
# plots run free), sharing A's t4_ref/cycle bins (Sim B is tier-4).
function _build_dual_b(splots, eco_species_ids, eco_list, spdf_plts, loss_params, t4_ref, cycle_map, n_cycles, injection_cohorts, spinup_cohorts, rng, no_establishment; b_only::Bool=false, scoring::ScoringOptions, simb::SimBOptions, plugin::PluginConfig)
  # Per-(eco,cycle) distinct-plot count, matching the tier-4 ref build (all measured years per cycle). Used
  # by SimBOptions.t4_per_plot_mean to turn the AGB level into mean biomass-per-plot-by-age (consistently in scale+loss).
  cell_plots = [zeros(Int, max(1, n_cycles)) for _ in eachindex(eco_species_ids)]
  for ((plot_id, eco_id), year_dict) in spdf_plts
    cycs = Set{Int}()
    for (sy, _) in year_dict
      cyc = get(cycle_map, (Int(plot_id), Int(sy)), 0)
      cyc == 0 || push!(cycs, cyc)
    end
    for cyc in cycs; cell_plots[eco_id][cyc] += 1; end
  end
  art = Set(e for e in eachindex(eco_list) if occursin("artificial", lowercase(eco_list[e])))
  art_plots = Set(Int(r.plot_id) for r in eachrow(splots) if Int(r.eco_id) in art)
  # simb_disturb_only: no injection at all (purely free everywhere); else default = plant/sync artificial plots.
  inj_b = (simb.simb_disturb_only || isnothing(injection_cohorts)) ? nothing : filter(r -> Int(r.plot_id) in art_plots, injection_cohorts)
  # Sim B start: simb_spinup=true spins up from a deficit (model establishes the stand from scratch via the
  # back-cast); false starts from the OBSERVED year-0 cohorts (anchored to data, no invented history) while
  # still running free establishment forward.
  ref_b = make_sites_like(plugin, splots, eco_species_ids; rng=rng, spinup=simb.simb_spinup, no_establishment=false)
  injection_b = build_injection_plan(inj_b, ref_b, eco_species_ids, loss_params, scoring)
  # Disturbance-apply for the free (natural) plots: at each disturbance year, reduce the species' free-sim
  # biomass by the observed drop fraction (exogenous event; the model responds but does not predict it).
  # Artificial plots are excluded here — their disturbances are handled by the planting/sync injection.
  # Per (site,year,species) the fraction is the biomass-weighted average of the cohorts' disturbance_drop_pct.
  # ddict: year → site(mapcode) → (species, age) → biomass-weighted drop fraction. Keyed by (species,age)
  # so the free-sim apply scales only a cohort matching BOTH species and age at the disturbance year.
  ddict = Dict{Int,Dict{Int,Dict{Tuple{UIntType,UIntType},FloatType}}}()
  if !isnothing(injection_cohorts) && hasproperty(injection_cohorts, :disturbance_drop_pct)
    plot_to_site = Dict{Int,Int}(Int(getsite(ref_b, i).ref_cn) => i for i in 1:ref_b.n)
    acc = Dict{Tuple{Int,Int,UIntType,UIntType},Tuple{FloatType,FloatType}}()   # (year,site,species,age)→(Σ bio·drop, Σ bio)
    for r in eachrow(injection_cohorts)
      (!simb.simb_disturb_only && Int(r.plot_id) in art_plots) && continue   # disturb-only: apply to ALL plots
      r.disturbance_drop_pct > 0 || continue
      si = get(plot_to_site, Int(r.plot_id), 0); si == 0 && continue
      k = (Int(r.sim_year), si, UIntType(r.eco_species_id), UIntType(round(r.age_calc)))
      bio = FloatType(r.agb_sum); n0, d0 = get(acc, k, (zero(FloatType), zero(FloatType)))
      acc[k] = (n0 + bio * FloatType(r.disturbance_drop_pct), d0 + bio)
    end
    for ((yr, si, sp, age), (num, den)) in acc
      den > 0 || continue
      sd = get!(get!(ddict, yr, Dict{Int,Dict{Tuple{UIntType,UIntType},FloatType}}()), si, Dict{Tuple{UIntType,UIntType},FloatType}())
      sd[(sp, age)] = num / den
    end
    # Plot-level FULL removals by disturbance (Data._plot_shocks!, carried as metadata of the injection cohorts): drop = 1 ⇒ the free-sim
    # cohort (species, age) is removed at that year by the apply loop. Same artificial/natural rule as above.
    if "disturbance_shock" in metadatakeys(injection_cohorts)
      nfull_b = 0
      for r in eachrow(metadata(injection_cohorts, "disturbance_shock"))
        (!simb.simb_disturb_only && Int(r.plot_id) in art_plots) && continue
        si = get(plot_to_site, Int(r.plot_id), 0); si == 0 && continue
        sd = get!(get!(ddict, Int(r.sim_year), Dict{Int,Dict{Tuple{UIntType,UIntType},FloatType}}()), si, Dict{Tuple{UIntType,UIntType},FloatType}())
        sd[(UIntType(r.eco_species_id), UIntType(r.age_calc))] = one(FloatType); nfull_b += 1
      end
      @info "Sim B: $(nfull_b) full-removal disturbance entries"
    end
  end
  @info "Sim B dual_b: $(ref_b.n) sites, $(length(art_plots)) artificial plots | injection=$(injection_b === nothing ? "OFF" : "ON ($(length(injection_b.years)) yrs, artificial)") | disturbance-scale=$(isempty(ddict) ? "OFF" : "ON ($(length(ddict)) yrs, $(simb.simb_disturb_only ? "ALL plots" : "natural only"))") | establishment=ON"
  return (ref_soa=ref_b, spdf_plts=spdf_plts, loss_params=loss_params, t4_ref=t4_ref,
    cycle_map=cycle_map, n_cycles=n_cycles, injection=injection_b,
    spinup=simb.simb_spinup, spinup_cohorts=spinup_cohorts, b_only=b_only, tier=simb.tier_b, t4_primary=simb.t4_primary,
    disturbance_dict=(isempty(ddict) ? nothing : ddict), disturbance_years=Set(keys(ddict)), cell_plots=cell_plots)
end

# ANALYSIS helper: re-simulate ONE params set through the FREE Sim-B path (dual_mode:b) EXACTLY as the run
# does — set the Sim-B dynamics globals from the config, build the tier-4 ref + dual_b (disturbance-scale /
# year-0 vs spinup / establishment ON / per-lineage frozen growth already baked into `params`), run
# fit_params(b_only), then bin the simulated cohorts AND the observed cohorts into the tier-4 age-bins and
# pair them. Returns a DataFrame(cyc, eco, esp, bin, sp, obs_agb, sim_agb) — the common input for the Sim-B
# scatter / sMAPE / TOST plots (so all three reflect the actual free-regen sim, not the Sim-A tier-3 path).
# by_plot=true keeps the plot id (pairs per (plot, cycle, eco, species, bin)) — used by the plot-cluster bootstrap TOST.
# preexist=true keeps only cohorts that existed at the plot's first visit (age > sim_year) on BOTH sides: drops simulated
# recruits and observed cohorts younger than the run (timber-only data cannot observe recruits until they reach 5 in).
function resim_simB_paired(cfg0, params, sp, spdf_plts, site_sim_years, spinup_cohorts, loss_params,
                           eco_list, species_list, eco_species_ids, rng; by_plot::Bool=false, preexist::Bool=false,
                           scoring::ScoringOptions=scoring_options(cfg0), plugin::PluginConfig)
  cfg = _as_config(cfg0)
  # Sim B only (dual_mode b), whatever the config says; the other Sim B options follow the config.
  simb = SimBOptions(; dual_mode=:b, tier_b=cfg["tier_b"], t4_primary=true, simb_spinup=cfg["simB_spinup"],
    simb_disturb_only=cfg["simB_disturb_only"], t4_per_plot_mean=cfg["simB_per_plot_mean"], t4_mean_over_reps=cfg["simB_rep_mean"])
  n_species = length(species_list)
  inj = (simb.simb_disturb_only || scoring.override) ? Data.get_injection_cohorts(sp; all_cohorts=true) : nothing
  cycle_map, n_cycles = Data.build_cycle_map(sp; cycle_years=cfg["cycle_years"], cycle=scoring.cycle)
  n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
  t4_ref = [[zeros(FloatType, length(eco_species_ids[e]), n_bins) for _ in 1:n_cycles] for e in eachindex(eco_list)]
  for ((pid, eid), yd) in spdf_plts, (sy, gt) in yd
    cyc = get(cycle_map, (Int(pid), Int(sy)), 0); cyc == 0 && continue
    for (spe, rec) in gt.records; t4_ref[eid][cyc][spe, :] .+= diff([0f0; rec.sp_age_cdf]) .* rec.sp_agb_sum; end
  end
  dual_b = _build_dual_b(sp, eco_species_ids, eco_list, spdf_plts, loss_params, t4_ref, cycle_map, n_cycles, inj, spinup_cohorts, rng, false; b_only=true, scoring, simb, plugin=plugin)
  msy = maximum(maximum.(filter(!isempty, site_sim_years.sim_years)))
  res = fit_params(dual_b.ref_soa, params, msy, n_species, eco_species_ids, spdf_plts, site_sim_years, false, spinup_cohorts, loss_params;
    debug=false, search_tier=3, dual_b=dual_b, seeds=[rand(rng, UInt64)], scoring, simb, plugin=plugin, collect_cohorts=true)
  cached = res[1][2]; ab = loss_params.age_bins
  plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(DataFrames.select(sp, [:plot_id, :eco_id]))))
  simdf = DataFrames.DataFrame(plot=Int[], cyc=Int[], eco=Int[], esp=Int[], bin=Int[], agb=Float64[])
  for (pid, sy, esp, age, bio) in cached
    haskey(plot2eco, Int(pid)) || continue
    cyc = get(cycle_map, (Int(pid), Int(sy)), 0); cyc == 0 && continue
    (preexist && Float64(age) <= sy) && continue
    b = PU.find_age_bin(Int(ceil(Float64(age))), ab); b == 0 && continue
    push!(simdf, (by_plot ? Int(pid) : 0, cyc, plot2eco[Int(pid)], Int(esp), Int(b), Float64(bio)))
  end
  keys_ = [:plot, :cyc, :eco, :esp, :bin]
  sim_agg = DataFrames.combine(DataFrames.groupby(simdf, keys_), :agb => sum => :sim_agb)
  obsdf = DataFrames.DataFrame(plot=Int[], cyc=Int[], eco=Int[], esp=Int[], bin=Int[], agb=Float64[])
  for r in eachrow(DataFrames.subset(sp, :sim_year => DataFrames.ByRow(>(0))))
    cyc = get(cycle_map, (Int(r.plot_id), Int(r.sim_year)), 0); cyc == 0 && continue
    (preexist && Float64(r.age_calc) <= r.sim_year) && continue
    b = PU.find_age_bin(Int(ceil(Float64(r.age_calc))), ab); b == 0 && continue
    push!(obsdf, (by_plot ? Int(r.plot_id) : 0, cyc, Int(r.eco_id), Int(r.eco_species_id), Int(b), Float64(r.agb_sum)))
  end
  obs_agg = DataFrames.combine(DataFrames.groupby(obsdf, keys_), :agb => sum => :obs_agb)
  paired = DataFrames.outerjoin(obs_agg, sim_agg, on=keys_)
  paired.obs_agb = coalesce.(paired.obs_agb, 0.0); paired.sim_agb = coalesce.(paired.sim_agb, 0.0)
  especo2sp = Dict((Int(r.eco_id), Int(r.eco_species_id)) => Int(r.species_id) for r in eachrow(unique(DataFrames.select(sp, [:eco_id, :eco_species_id, :species_id]))))
  paired.sp = [especo2sp[(e, esp)] for (e, esp) in zip(paired.eco, paired.esp)]
  by_plot || DataFrames.select!(paired, DataFrames.Not(:plot))
  return paired, cycle_map, n_cycles
end

# Build Sim B for the VALIDATION split, so held-out loss measures the SAME A⊕B objective as training
# (otherwise val = Sim A only, which is not comparable to a dual train loss). Builds val tier-4 refs +
# cycle map, then the dual_b. Returns nothing when dual mode is off.
function _build_val_dual_b(val_splots, eco_species_ids, eco_list, val_spdf_plts, loss_params, cycle_years,
    val_injection_cohorts, val_spinup_cohorts, rng, no_establishment; scoring::ScoringOptions, simb::SimBOptions, plugin::PluginConfig)
  simb.dual_mode == :off && return nothing
  vcm, vnc = Data.build_cycle_map(val_splots; cycle_years=cycle_years, cycle=scoring.cycle)
  nb = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
  vt4 = [[zeros(FloatType, length(eco_species_ids[e]), nb) for _ in 1:vnc] for e in eachindex(eco_list)]
  for ((pid, eid), yd) in val_spdf_plts, (sy, gt) in yd
    cyc = get(vcm, (Int(pid), Int(sy)), 0); cyc == 0 && continue
    for (sp_eco, rec) in gt.records; vt4[eid][cyc][sp_eco, :] .+= diff([0f0; rec.sp_age_cdf]) .* rec.sp_agb_sum; end
  end
  _build_dual_b(val_splots, eco_species_ids, eco_list, val_spdf_plts, loss_params, vt4, vcm, vnc,
    val_injection_cohorts, val_spinup_cohorts, rng, no_establishment; b_only=(simb.dual_mode == :b), scoring, simb, plugin=plugin)
end

function _filter_cached_to_df(cached, sampled_ids::Set)
  filtered = filter(t -> UIntType(t[1]) in sampled_ids, cached)
  isempty(filtered) && return DataFrame(plot_id=UIntType[], sim_year=Int[], species_id=UIntType[], age=UIntType[], agb=FloatType[])
  DataFrame(filtered, [:plot_id, :sim_year, :species_id, :age, :agb])
end
