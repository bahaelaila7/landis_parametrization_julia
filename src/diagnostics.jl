# ---------------------------------------------------------------------------
# Standalone plot helpers
# ---------------------------------------------------------------------------
function _load_params_from_path(params_path::String, pc=nothing)
  raw = JLD2.load_object(params_path)
  if raw isa SA.SAState
    @info "Loaded SAState — using best params" loss = convert(Float64, raw.best.fx) iter = raw.best_iteration
    return raw.best.x
  elseif raw isa LBSA.LBSAState
    @info "Loaded LBSAState — using best params" loss = convert(Float64, raw.best.fx) iter = raw.best_iteration
    return raw.best.x
  elseif raw isa MOLBSA.MOLBSAState
    @info "Loaded MOLBSAState — using representative params" agg = raw.representative.fx.aggregate iter = raw.best_iteration
    return raw.representative.x
  elseif raw isa CMAES.CMAESState
    @info "Loaded CMAESState — using best params" loss = convert(Float64, raw.best.fx) iter = raw.best_iteration
    return raw.best.x
  elseif raw isa MOCMAES.MOCMAESState
    @info "Loaded MOCMAESState — using representative params" agg = raw.representative.fx.aggregate iter = raw.best_iteration archive = length(raw.archive)
    return raw.representative.x
  elseif raw isa IgelMOCMAES.IgelState
    @info "Loaded IgelState — using representative params" agg = raw.representative.fx.aggregate iter = raw.best_iteration archive = length(raw.archive)
    return raw.representative.x
  elseif raw isa CMAMAE.CMAMAEMOState
    @info "Loaded CMAMAEMOState — using representative params" agg = raw.representative.fx.aggregate iter = raw.best_iteration archive = length(raw.archive)
    return raw.representative.x
  elseif raw isa JLD2.ReconstructedMutable && occursin("HBADParams", String(typeof(raw).parameters[1]))
    @info "Loaded HBADParams saved before the establishment fields existed - upgraded with the defaults"
    return HBADPlugin.upgrade_params(raw, pc isa HBADPlugin.HBADContext ? pc : nothing)
  elseif raw isa Vector  # sobol results
    @info "Loaded sobol results — using rank-1 params" mean_loss = raw[1].mean_loss
    return raw[1].params
  else
    @info "Loaded params directly"
    return raw
  end
end

# One simulation of `bio_params` on the training context `ctx` (parametrize(context_only=true)), then the sampled plots' figures.
function _run_and_plot(bio_params, label, ctx; sampled_ids, emp_sample, output_dir, rng)
  injection = build_injection_plan(ctx.injection_cohorts, ctx.ref_soa, ctx.eco_species_ids, ctx.loss_params, ctx.scoring)
  max_sim_year = maximum(maximum.(filter(!isempty, ctx.site_sim_years.sim_years)))
  result = only(fit_params(ctx.ref_soa, bio_params, max_sim_year, ctx.n_species, ctx.eco_species_ids,
    ctx.spdf_plts, ctx.site_sim_years, ctx.spinup, ctx.spinup_cohorts, ctx.loss_params;
    debug=false, search_tier=3, seeds=[rand(rng, UInt64)], cache_preinject=!isnothing(injection), collect_cohorts=true,
    injection=injection,
    scales=compute_loss_scales(ctx.loss_params, 3, ctx.spdf_plts, nothing, ctx.eco_species_ids, ctx.n_species; per_plot_mean=ctx.simb.t4_per_plot_mean, cbal=ctx.scale_seed.cbal, ncoh=ctx.scale_seed.ncoh, dom=ctx.scale_seed.dom), scoring=ctx.scoring, simb=ctx.simb, plugin=ctx.plugin))
  loss = convert(Float64, PU.get_total_loss(result[1], ctx.loss_params))
  sim_sample = _filter_cached_to_df(result[2], sampled_ids)
  generate_plots(emp_sample, sim_sample, label, loss, output_dir)
  @info "Plots saved [$label]" loss n_plots = length(sampled_ids)
end

"""
plot_sample — run one simulation with given params and plot n_output_plots forest plots.
Accepts best_params@X.jld2, search_state@X.jld2, or sobol_results@X.jld2.
"""
function plot_sample(ctx; params_path::String, n_output_plots::Int=20, rng_seed::Int=1337)
  output_dir = ctx.output_dir; mkpath(output_dir)
  rng = RNGType(UInt64(rng_seed))
  bio_params = _load_params_from_path(params_path, ctx.plugin.ctx)
  all_ids = UIntType.(unique(ctx.splots.plot_id))
  sampled_ids = _sample_plot_ids(all_ids, n_output_plots, rng; injection_cohorts=ctx.injection_cohorts)
  emp_sample = _make_emp_df(ctx.splots, sampled_ids)
  _run_and_plot(bio_params, "plot_only", ctx; sampled_ids, emp_sample, output_dir, rng)
end

"""
plot_sample_sobol — load saved sobol results and plot a stratified sample of param sets.
Selection: 1 absolute top + 60% of remaining from top half + rest from bottom half.
All param sets are visualised on the same fixed set of n_output_plots forest plots.
"""
function plot_sample_sobol(ctx; sobol_candidates_db::String, n_output_plots::Int=10, n_sobol_params_to_plot::Int=5, rng_seed::Int=1337)
  output_dir = ctx.output_dir
  mkpath(output_dir)
  rng = RNGType(UInt64(rng_seed))

  db = DuckDB.DB(sobol_candidates_db)
  con = DuckDB.connect(db)
  raw = DuckDB.execute(con, "SELECT mean_loss, params_blob FROM sobol_results ORDER BY mean_loss ASC, sobol_idx ASC") |> DataFrame
  close(db)
  isempty(raw) && (@warn "No sobol results found in $sobol_candidates_db"; return)
  results = [(params=Serialization.deserialize(IOBuffer(row.params_blob)), mean_loss=row.mean_loss)
             for row in eachrow(raw)]
  N = length(results)
  @info "Loaded $N sobol results from $sobol_candidates_db"

  # Stratified param-set selection
  n_extra = n_sobol_params_to_plot - 1
  n_top = round(Int, 0.6 * n_extra)
  n_bot = n_extra - n_top
  top_half = results[2:max(2, N ÷ 2)]
  bot_half = results[max(2, N ÷ 2)+1:end]
  selected = vcat(
    [results[1]],
    Random.shuffle(rng, top_half)[1:min(n_top, length(top_half))],
    Random.shuffle(rng, bot_half)[1:min(n_bot, length(bot_half))],
  )
  @info "plot_sample_sobol: plotting $(length(selected)) param sets from $N sobol results"

  all_ids = UIntType.(unique(ctx.splots.plot_id))
  sampled_ids = _sample_plot_ids(all_ids, n_output_plots, rng; injection_cohorts=ctx.injection_cohorts)
  emp_sample = _make_emp_df(ctx.splots, sampled_ids)

  for (rank, res) in enumerate(selected)
    label = rank == 1 ? "sobol_rank1_loss$(round(res.mean_loss,digits=4))" :
            "sobol_rank$(rank)_loss$(round(res.mean_loss,digits=4))"
    _run_and_plot(res.params, label, ctx; sampled_ids, emp_sample, output_dir, rng)
  end
end

# ---------------------------------------------------------------------------
# Statistical testing: simulated vs reference plot distributions
# ---------------------------------------------------------------------------


function simulate_and_test(;
  splots::DataFrame,
  bio_params,
  plugin::PluginConfig,
  eco_list::Vector{String},
  species_list::Vector{String},
  eco_species_ids::Vector{Vector{Int}},
  loss_params::PU.LossParams,
  site_sim_years,
  M::Int=10,
  equiv_margin::Float64=0.2,   # equivalence band = ±equiv_margin · mean(ref)
  tost_alpha::Float64=0.05,
  no_establishment::Bool=false,
  rng::Random.AbstractRNG,
)::DataFrame
  n_ecos = length(eco_list)
  n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
  max_sim_year = maximum(maximum.(filter(!isempty, site_sim_years.sim_years)))
  ref_soa = make_sites(splots, eco_species_ids; plugin=plugin, rng=rng, spinup=false, no_establishment=no_establishment)
  SP = succession_plugin_of(ref_soa)
  eco_params = generate_eco_params(plugin.ctx, bio_params)
  ctx = make_ctx(plugin.ctx, eco_params)

  # Reference: per (eco_id, sp_eco, bin) → [AGB per (plot, measurement_year)], excluding the
  # initial year (sim_year=0) since that's the state we initialize from.
  ref_vals = [[[FloatType[] for _ in 1:n_bins] for _ in 1:length(eco_species_ids[e])] for e in 1:n_ecos]
  for gdf in groupby(splots, [:plot_id, :eco_id, :eco_species_id, :measdate])
    r = gdf[1, :]
    sim_year = Int(round(Dates.value(Dates.Day(r.measdate - r.start_measdate)) / 365.25))
    sim_year > 0 || continue
    eco_id = Int(r.eco_id)
    sp_eco = Int(r.eco_species_id)
    (1 <= eco_id <= n_ecos && 1 <= sp_eco <= length(eco_species_ids[eco_id])) || continue
    bin_agbs = zeros(FloatType, n_bins)
    for row in eachrow(gdf)
      b = PU.find_age_bin(max(1, Int(round(Float64(row.age_calc)))), loss_params.age_bins)
      b == 0 && continue
      bin_agbs[b] += FloatType(row.agb_sum)
    end
    for b in 1:n_bins
      push!(ref_vals[eco_id][sp_eco][b], bin_agbs[b])
    end
  end

  # Simulate M times; accumulate per (site_idx, eco_id, sp_eco, bin) to average across reps.
  sim_sum = Dict{NTuple{4,Int},Float64}()
  sim_cnt = Dict{NTuple{4,Int},Int}()

  for seed in (rand(rng, UInt64) for _ in 1:M)
    soa = copy_and_reseed_soa(ref_soa, seed)
    for current_sim_year in 1:max_sim_year
      step!(SP, soa, current_sim_year, ctx)
      for i in 1:soa.n
        site = getsite(soa, i)
        !site.active && continue
        sim_years = site_sim_years.sim_years[site.mapcode]
        if current_sim_year in sim_years
          eco_id = Int(site.eco_id)
          n_sp_eco = length(eco_species_ids[eco_id])
          bin_agbs = zeros(FloatType, n_sp_eco, n_bins)
          for j in 1:site.live
            sp_eco = Int(site.c_species[j])
            b = PU.find_age_bin(max(1, Int(ceil(Float64(site.c_age[j])))), loss_params.age_bins)
            b == 0 && continue
            bin_agbs[sp_eco, b] += site.c_bio[j]
          end
          for sp_eco in 1:n_sp_eco
            for b in 1:n_bins
              key = (i, eco_id, sp_eco, b)
              sim_sum[key] = get(sim_sum, key, 0.0) + Float64(bin_agbs[sp_eco, b])
              sim_cnt[key] = get(sim_cnt, key, 0) + 1
            end
          end
        end
        if current_sim_year == last(sim_years)
          site.active = false
        end
      end
    end
  end

  # Per (eco_id, sp_eco, bin): one mean-across-M value per (site, measurement-year) pair.
  sim_vals = [[[FloatType[] for _ in 1:n_bins] for _ in 1:length(eco_species_ids[e])] for e in 1:n_ecos]
  for (key, total) in sim_sum
    (_, eco_id, sp_eco, b) = key
    push!(sim_vals[eco_id][sp_eco][b], FloatType(total / sim_cnt[key]))
  end

  # TOST (two one-sided tests) for equivalence per (eco, species, bin), on log1p(AGB)
  # so the band is multiplicative (handles skew + zeros). Δ = log1p(equiv_margin) is a
  # fixed log-scale band ≈ a factor-(1+equiv_margin) tolerance on the original scale.
  # Equivalent at tost_alpha iff BOTH one-sided Welch tests reject — i.e. the geometric-
  # mean ratio sim/ref is provably inside [1/(1+m), (1+m)] (approximate under log1p).
  bin_label(b) = b <= length(loss_params.age_bins.bins_idx) ?
                 "<$(loss_params.age_bins.bins_idx[b])" : ">=$(loss_params.age_bins.bins_idx[end])"

  Δ = log1p(equiv_margin)   # fixed band on the log1p scale

  rows = NamedTuple[]
  for eco_id in 1:n_ecos
    for sp_eco in eachindex(eco_species_ids[eco_id])
      gsp = eco_species_ids[eco_id][sp_eco]
      for b in 1:n_bins
        rawref = Float64.(ref_vals[eco_id][sp_eco][b])
        rawsim = Float64.(sim_vals[eco_id][sp_eco][b])
        (length(rawref) < 2 || length(rawsim) < 2) && continue   # need a variance per side

        ref = log1p.(rawref)
        sim = log1p.(rawsim)
        d = Statistics.mean(sim) - Statistics.mean(ref)

        if Statistics.var(ref) == 0 && Statistics.var(sim) == 0
          # both bins constant → no sampling uncertainty; decide directly (Welch SE would be 0)
          equiv = abs(d) < Δ
          p_tost, ci_lo, ci_hi = (equiv ? 0.0 : 1.0), d, d
        else
          # two one-sided Welch tests against the shifted nulls ∓Δ (SE > 0 here)
          p_lower = HypothesisTests.pvalue(HypothesisTests.UnequalVarianceTTest(sim .+ Δ, ref); tail=:right)  # H1: d > −Δ
          p_upper = HypothesisTests.pvalue(HypothesisTests.UnequalVarianceTTest(sim .- Δ, ref); tail=:left)   # H1: d <  Δ
          p_tost = max(p_lower, p_upper)
          ci = HypothesisTests.confint(HypothesisTests.UnequalVarianceTTest(sim, ref); level=1 - 2 * tost_alpha)
          ci_lo, ci_hi = ci[1], ci[2]
        end

        push!(rows, (
          eco=eco_list[eco_id],
          species=species_list[gsp],
          bin=b,
          age_class=bin_label(b),
          n_ref=length(rawref),
          n_sim=length(rawsim),
          mean_ref=Statistics.mean(rawref),   # original scale, for context
          log_diff=d,                          # mean(log1p sim) − mean(log1p ref)
          ratio=exp(d),                        # ≈ geometric-mean ratio sim/ref
          margin=Δ,                            # log-scale band
          ci_lo=ci_lo, ci_hi=ci_hi,            # CI for log_diff
          p_tost=p_tost,
          equivalent=(p_tost < tost_alpha),
        ))
      end
    end
  end

  return isempty(rows) ? DataFrame() : DataFrame(rows)
end
