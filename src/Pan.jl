module Pan
include("PanCore.jl")
include("Plugins.jl")
include("Parametrization.jl")
include("Search.jl")
include("Data.jl")
include("Spatial.jl")
#for file in filter(f -> endswith(f,"Plugin.jl"), readdir("src/plugins";join=false))
#    println("including plugin $(file)")
#    include(joinpath("plugins", file))
#end
#include("plugins/BiomassSuccessionPlugin.jl")
using .PanCore
using .Plugins: BaseSitePlugin, BiomassSuccessionPlugin
import .Parametrization as PU
import .Parametrization.BiomassSuccessionParametrization as BSP
using .Search: SA, LBSA
import .Data as Data
import .Spatial
import Dates
import CSV

import Random
import Distributions as Dists
import Term.Progress as TProgress
import JLD2
using DataFrames



const RNGType = Random.MersenneTwister
const ActivePlugins = (BaseSitePlugin.BaseSite, BiomassSuccessionPlugin.BiomassSuccession)
const ActiveSoA = SiteSoA{Tuple{(ActivePlugins)...}}

function make_sites(splots::DataFrame, eco_species_ids::Vector{Vector{Int}}; rng::Random.AbstractRNG, spinup::Bool=true)
  #eco_ids = unique(select(splots, [:eco_id]))
  #n_species = maximum(splots.species_id)
  #plot_ids = unique(select(splots, [:plot_id]))
  #plot_eco_ids = unique(select(splots, [:plot_id, :eco_id]))
  #plot_eco_ids = sort!(plot_eco_ids, [:plot_id])
  df = unique(select(splots, [:plot_id, :eco_id]))
  n = nrow(df)
  #plot_eco_ids = sort!(plot_eco_ids, [:plot_id])
  #@assert (nrow(plot_ids) == nrow(plot_eco_ids)) "Error: plot_ids and plot_eco_ids are not of equal length!"

  #index = eco_id * plot_id

  #splots_dict::Dict{Int64,DataFrame} = Dict(
  #    plt_key.plt_cn => DataFrame(plt_df)
  #    for (plt_key, plt_df) in pairs(groupby(splots, :plt_cn, sort=false))
  #)
  species_counts = [Int32(length(eco_species_ids[row.eco_id])) for row in eachrow(df)]
  if spinup
    cohort_counts = fill(Int32(2), n)
    @debug cohort_counts
    soa = ActiveSoA((cohort=cohort_counts, species=species_counts))
    tRNGs = [RNGType(rand(rng, UInt64)) for _ in 1:Threads.maxthreadid()]
    Threads.@threads :static for i in 1:nrow(df)
      @inbounds begin
        site = getsite(soa, i)
        site.active = false
        site.rng = RNGType(rand(tRNGs[Threads.threadid()], UInt64))
        site.ecocode = UIntType(df.eco_id[i])# no ecocode coming from raster, relying on eco_id
        site.eco_id = df.eco_id[i]
        site.mapcode = UIntType(df.plot_id[i]) # for parametrization, plot_id is global index, no raster
        site.ref_cn = UIntType(df.plot_id[i])
        site.old = zero(UIntType)
        site.live = zero(UIntType)
        site.B = zero(FloatType)
        site.AGNPP = zero(FloatType)
        site.harvestCapacityReduction = zero(FloatType)
        site.growthReduction = zero(FloatType)
        site.prevYearMortality = zero(FloatType)
        site.shade_class = one(UIntType)
        site.c_species .= zero(UIntType)
        site.c_age .= zero(FloatType)
        site.c_bio .= zero(FloatType)
        site.c_m_tot .= zero(FloatType)
        site.c_comp .= zero(FloatType)
        site.no_establish = false
        site.sp_mature .= false
        site.sp_sprout .= false
        site.sp_seed .= false
        site.sp_serotiny .= false
        site.sp_plant .= false
      end
    end
    return soa
  else
    splots_dict = Dict(
      (plt_key.plot_id, plt_key.eco_id) => begin
        plt_df = sort!(DataFrame(plt_df), :age_calc, rev=true)
        plt_df.sim_year .= Dates.value.(Dates.Day.(plt_df.measdate - plt_df.start_measdate)) ./ 365.25 .|> round .|> Int
        Data.get_initial_cohorts(plt_df)
      end
      for (plt_key, plt_df) in pairs(groupby(splots, [:plot_id, :eco_id], sort=false))
    )
    df_keys = sort!(collect(keys(splots_dict)))

    cohort_counts = [Int32(nrow(splots_dict[key])) for key in df_keys]
    soa = ActiveSoA((cohort=cohort_counts, species=species_counts))
    tRNGs = [RNGType(rand(rng, UInt64)) for _ in 1:Threads.maxthreadid()]
    Threads.@threads :static for i in 1:length(df_keys)
      @inbounds begin
        key = df_keys[i]
        plt_df = splots_dict[key]
        (plot_id, eco_id) = key
        initial_cohorts = plt_df
        #n_cohorts = nrow(initial_cohorts)
        #cap = UIntType(2^ceil(log2(n_cohorts)))
        site = getsite(soa, i)
        site.active = true
        site.rng = RNGType(rand(tRNGs[Threads.threadid()], UInt64))
        site.ecocode = UIntType(eco_id) #UIntType(getproperty(p, ecocode_field)),
        site.eco_id = eco_id
        site.mapcode = UIntType(plot_id)
        site.ref_cn = plot_id
        site.old = zero(UIntType)
        site.live = zero(UIntType)
        site.B = zero(FloatType)
        site.AGNPP = zero(FloatType)
        site.harvestCapacityReduction = zero(FloatType)
        site.growthReduction = zero(FloatType)
        site.prevYearMortality = zero(FloatType)
        site.shade_class = one(UIntType)
        site.c_species .= zero(UIntType)
        site.c_age .= zero(FloatType)
        site.c_bio .= zero(FloatType)
        site.c_m_tot .= zero(FloatType)
        site.c_comp .= zero(FloatType)
        site.no_establish = false
        site.sp_mature .= false
        site.sp_sprout .= false
        site.sp_seed .= false
        site.sp_serotiny .= false
        site.sp_plant .= false
        for row in eachrow(initial_cohorts)
          BiomassSuccessionPlugin.add_cohort!(site, UIntType(row.eco_species_id), FloatType(row.age_calc), FloatType(row.agb_sum))
        end
      end
    end
    return soa
  end
end

struct WriterJob{State}
  is_new_best::Bool
  state::State
end

const STOP = :stop

function start_writer(::Type{State}, output_dir::AbstractString; buffer_size::Int=8) where {State}
  ch = Channel{Union{WriterJob{State},Symbol}}(buffer_size)

  task = Threads.@spawn begin
    try
      for job in ch
        job === STOP && break
        state = job.state
        if job.is_new_best
          try
            @info "New best: $(convert(Float64,state.best.fx))" iter = state.i
            mkpath(output_dir)
            JLD2.save_object(joinpath(output_dir, "search_state@$(state.i).jld2"), state)
            PU.save_json(joinpath(output_dir, "best_params@$(state.i).json"), state.best.x)
            JLD2.save_object(joinpath(output_dir, "best_params@$(state.i).jld2"), state.best.x)
          catch e
            @error "writer: save failed" iter = state.i exception = (e, catch_backtrace())
          end
        else
          @info ("Best@$(state.best_iteration): $(convert(Float64, state.best.fx)), Avg diff: $(state.diff_avg), Temp: $(state.t), ratio $(state.diff_avg/state.t), Prob: $(state.prob_avg)")
        end
      end
    catch e
      @error "writer: fatal" exception = (e, catch_backtrace())
      rethrow()
    end
  end

  return ch, task
end

function stop_writer(ch::Channel, task::Task)
  put!(ch, STOP)
  close(ch)
  wait(task)
end
function test_spdf(df, n_species, eco_species_ids, loss_params::PU.LossParams)
  site_losses = []
  for (plt_key, plt_dict) in pairs(df)
    for (sim_key, spdf_plt) in pairs(plt_dict)
      #plot_id = plt_key.plot_id

      plot_id, eco_id = plt_key
      eco_n_species = length(eco_species_ids[eco_id])
      species_id_map = eco_species_ids[eco_id]
      @assert eco_n_species == length(spdf_plt.keys) "eco species numbers do not match"
      @assert length(species_id_map) == eco_n_species "eco species numbers do not match"

      insite = falses(eco_n_species)

      sp_w_loss = zeros(FloatType, n_species)
      sp_agb_loss = zeros(FloatType, n_species)
      site_agb_loss = zero(FloatType)
      for sp in 1:eco_n_species
        if !(sp in keys(spdf_plt.records))
          continue
        end
        insite[sp] = true
        sim_agb_sum = 0.0f0 #spdf_plt.records[sp].sp_agb_sum
        gsp = species_id_map[sp]

        log_diff = log10(sim_agb_sum + loss_params.EPS)
        sp_agb_loss[gsp] = sim_agb_sum
        site_agb_loss += sim_agb_sum
        if spdf_plt.keys[sp]
          rec = @inbounds spdf_plt.records[sp]
          #ages .= zero(FloatType)
          #for a in @view p[sp_start_idx:sp_end_idx]
          #    ages[UIntType(site.c_age[a])] = site.c_bio[a]
          #end
          #sim_age_cdf = smoothen_bin_cdf(ages; w=loss_params.smoothing_weights, age_bins=loss_params.age_bins)
          #@assert !any(isnan.(sim_age_cdf)) "cdf NaN"
          #@assert length(sim_age_cdf) == length(rec.sp_age_cdf) "cdf bins are not the same size"
          sp_w_loss[gsp] = sum(loss_params.age_bins.bin_widths .* abs.(rec.sp_age_cdf)[begin:end-1])
          @assert !any(isnan.(sp_w_loss[gsp])) "NaN"
          log_diff -= log10(rec.sp_agb_sum + loss_params.EPS)
          sp_agb_loss[gsp] = abs(sim_agb_sum - rec.sp_agb_sum)
          site_agb_loss -= rec.sp_agb_sum
        end
        sp_w_loss[gsp] = (1.0f0 + sp_w_loss[gsp]) * (abs(log_diff)^2)

      end
      for sp in (1:length(spdf_plt.keys))[spdf_plt.keys.&(.!insite)]
        @inbounds rec = spdf_plt.records[UIntType(sp)]
        @inbounds gsp = species_id_map[sp]
        sp_agb_loss[gsp] = rec.sp_agb_sum
        sp_w_loss[gsp] += loss_params.lambda * abs(log10(rec.sp_agb_sum + loss_params.EPS))
        site_agb_loss -= rec.sp_agb_sum
      end


      push!(site_losses, PU.SiteLoss(sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=abs(site_agb_loss), num_sites=1))
    end
  end
  return sum(site_losses)
end
function parametrize(; cohorts_db_path::String,
  eco_field=:epa_l4,
  tablename::String="data_eco_cohorts_g",
  output_dir::String="./outputs",
  filter_ecos::Vector{String}=String[],
  skip_disturbances::Bool=true,
  spinup::Bool=true,
  TRIALS::Int=30000,
  rng::Random.AbstractRNG)

  # [5, 10, 20, 40, 60, 80]
  #bins_idx = vcat(5:5:30, 40:10:80, 100:20:160)
  bins_idx = vcat(10:10:40, 60:20:120)
  smoothing_window = PU.get_smoothing_window(; smoothing_window=2, smoothing_variance=FloatType(1.2f0))
  @info bins_idx
  @info smoothing_window
  loss_params = PU.LossParams(
    age_bins=PU.AgeBins(
      bins_idx=bins_idx .|> Int,
      last_bin_open=true
    ),
    smoothing_weights=smoothing_window
  )
  splots, eco_list, species_list, eco_species_ids = Data.prepare_parametrization_data(; cohorts_db_path=cohorts_db_path,
    eco_field=eco_field,
    tablename=tablename,
    output_dir=tablename,
    skip_disturbances=skip_disturbances,
    filter_ecos=filter_ecos,
    RNG=rng)
  n_species = length(species_list)
  n_ecoregions = length(eco_list)
  n_plots = maximum(splots.plot_id)
  #println(eco_list)
  #println(species_list)
  #println(eco_species_ids)
  #println(splots)
  #return
  println("Plots:$n_plots, Ecos:$n_ecoregions, Species:$n_species, Measurements: $(size(splots))")
  println("Initiating param distributions")
  #greet()
  #println(splots)
  #println(splots)
  max_age = Int(maximum(splots.age_calc))
  #precomupte loss for missing entries
  println("Preprocessing plot results (smoothing and binning)")
  debug = (get(ENV, "JULIA_DEBUG", "") ∈ ["Pan", "all"])
  @time spdf = PU.smoothen_ref_years(splots, loss_params, max_age; debug=debug)
  @assert minimum(spdf.sim_year) == 0 "$(spdf.sim_year)"
  #show(spdf)
  println("Creating comparison years")
  @time spdf_plts = Data.make_spdf_dict(spdf, eco_species_ids)
  @time total_err = test_spdf(spdf_plts, n_species, eco_species_ids, loss_params)
  println(total_err)
  println(PU.get_total_loss(total_err))
  println(sum(length(xs) for xs in eco_species_ids))
  println("Marking sim years")
  @time site_sim_years = Data.get_site_sim_years(spdf)
  #println(site_sim_years)
  #return
  #show(site_sim_years)

  println("Marking spinup cohorts")
  @time spinup_cohorts = Data.get_spinup_cohorts(splots)
  #initial_cohorts = get_initial_cohorts(splots)
  println("beginning trials")
  #SITES_PER_RUN = Int(round(nrow(site_sim_years) * 0.33))
  #Profile.clear()
  #Profile.init(n=10^7, delay=0.001)
  #SITES_PER_RUN = Int(round(nrow(site_sim_years) * 0.33))



  ref_soa = make_sites(splots, eco_species_ids; rng=rng, spinup=spinup)
  parametrize_LBSA(; ref_soa=ref_soa,
    output_dir=output_dir,
    spdf_plts=spdf_plts,
    spinup_cohorts=spinup_cohorts,
    site_sim_years=site_sim_years,
    species_list=species_list,
    eco_list=eco_list,
    eco_species_ids=eco_species_ids,
    spinup=spinup,
    TRIALS=TRIALS,
    loss_params=loss_params,
    rng=rng,
    debug=debug)
end
function fit_params(soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug)
  eco_params = BiomassSuccessionPlugin.generate_eco_params(bio_params)
  ctx = (BiomassSuccession=(eco_params=eco_params,),)
  years_results = Vector{PU.SiteLoss}(undef, max_sim_year + 1)
  if spinup
    soa = BiomassSuccessionPlugin.spinup_cohorts!(soa, spinup_cohorts, eco_params)
  end
  for current_sim_year in 0:max_sim_year
    #println("\ttimestep $(t)")
    sites_results = Vector{PU.SiteLoss}(undef, soa.n)
    PanCore.process_plugin!(soa, BiomassSuccessionPlugin.BiomassSuccession, current_sim_year; ctx=ctx.BiomassSuccession)
    Threads.@threads :static for i in 1:soa.n
      @inbounds begin
        site = getsite(soa, i)
        !site.active && continue
        spdf_plt = spdf_plts[(site.ref_cn, site.eco_id)]
        sim_years = site_sim_years.sim_years[site.mapcode]
        if current_sim_year in sim_years
          sloss = PU.calculate_site_loss2(current_sim_year, site, n_species, eco_species_ids, spdf_plt[current_sim_year], loss_params; debug=debug)
          sites_results[i] = sloss
        end
      end
    end
    year_results_no_missing = collect(PU.skipundef(sites_results))
    if length(year_results_no_missing) > 0
      current_year_results = sum(year_results_no_missing)
      years_results[current_sim_year+1] = current_year_results
    end
  end
  run_result = sum(PU.skipundef(years_results))
  #@assert !any(isnan.(run_result.sp_w_loss)) "run NaN"
  return run_result
end
function parametrize_LBSA(; ref_soa::ActiveSoA, output_dir::AbstractString, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool)
  n_species = length(species_list)
  max_sim_year = site_sim_years.sim_years .|> maximum |> maximum
  param_dists = BSP.make_biomass_param_dists(length(species_list), length(eco_list), eco_species_ids)
  bio_params = BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng)
  best_result = fit_params(deepcopy(ref_soa), bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug)
  #PU.SiteLoss(FloatType[], FloatType[], FloatType(Inf), 1)
  cur = LBSA.LBSACandidate(bio_params, best_result)
  _best = cur
  search_state = LBSA.LBSAState(_best, cur, rng; max_iter=TRIALS)
  if TRIALS < 1
    return search_state #best_loss, best_result, best_params
  end

  writer_ch, writer_task = start_writer(typeof(search_state), output_dir)

  try

    TProgress.@track for trial in 1:TRIALS
      soa = deepcopy(ref_soa)

      bio_params = PU.mutate_params(bio_params, param_dists; rng=rng, mutation_mode=PU.BothMutations,
        ctx=PU.SamplingContext(search_state.current.fx.sp_w_loss .+ search_state.current.fx.sp_agb_loss .+ (search_state.current.fx.sp_w_loss .* search_state.current.fx.sp_agb_loss)))
      run_result = fit_params(soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug)
      next = LBSA.LBSACandidate(bio_params, run_result)

      is_new_best = LBSA.search_cmp!(next, search_state)
      if is_new_best
        put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state)))
      end
      if LBSA.should_restart(search_state)
        @info "Restarting @ $(search_state.i)"
        bio_params = BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng)
        cur_result = fit_params(deepcopy(ref_soa), bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug)
        is_new_best = LBSA.restart(search_state, LBSA.LBSACandidate(bio_params, cur_result))
      end
      if is_new_best || search_state.i % 50 == 0
        put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state)))
      end
      if LBSA.is_search_over(search_state)
        break
      end
      #println(convert(Float64,run_result))
    end
  finally
    stop_writer(writer_ch, writer_task)
  end

end
function parametrize_SA(; ref_soa::ActiveSoA, output_dir::AbstractString, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool)
  n_species = length(species_list)
  param_dists = BSP.make_biomass_param_dists(length(species_list), length(eco_list), eco_species_ids)
  bio_params = BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng)
  best_result = nothing # PU.SiteLoss(FloatType[], FloatType[], FloatType(Inf), 1)
  cur = SA.SACandidate(bio_params, best_result)
  _best = cur
  search_state = SA.SAState(_best, cur, rng; max_iter=TRIALS, initial_t=1e3, t=1e3)
  if TRIALS < 1
    return search_state #best_loss, best_result, best_params
  end
  max_sim_year = site_sim_years.sim_years .|> maximum |> maximum

  writer_ch, writer_task = start_writer(typeof(search_state), output_dir)

  try

    TProgress.@track for trial in 1:TRIALS
      search_state.i = trial
      soa = deepcopy(ref_soa)

      bio_params = PU.mutate_params(bio_params, param_dists; rng=rng)
      eco_params = BiomassSuccessionPlugin.generate_eco_params(bio_params)
      ctx = (BiomassSuccession=(eco_params=eco_params,),)
      years_results = Vector{PU.SiteLoss}(undef, max_sim_year + 1)
      if spinup
        soa = BiomassSuccessionPlugin.spinup_cohorts!(soa, spinup_cohorts, eco_params)
      end
      for current_sim_year in 0:max_sim_year
        #println("\ttimestep $(t)")
        sites_results = Vector{PU.SiteLoss}(undef, soa.n)
        PanCore.process_plugin!(soa, BiomassSuccessionPlugin.BiomassSuccession, current_sim_year; ctx=ctx.BiomassSuccession)
        Threads.@threads :static for i in 1:soa.n
          @inbounds begin
            site = getsite(soa, i)
            !site.active && continue
            spdf_plt = spdf_plts[(site.ref_cn, site.eco_id)]
            sim_years = site_sim_years.sim_years[site.mapcode]
            if current_sim_year in sim_years
              sloss = PU.calculate_site_loss2(current_sim_year, site, n_species, eco_species_ids, spdf_plt[current_sim_year], loss_params; debug=debug)
              sites_results[i] = sloss
            end
          end
        end
        year_results_no_missing = collect(PU.skipundef(sites_results))
        if length(year_results_no_missing) > 0
          current_year_results = sum(year_results_no_missing)
          years_results[current_sim_year+1] = current_year_results
        end
      end
      run_result = sum(PU.skipundef(years_results))
      @assert !any(isnan.(run_result.sp_w_loss)) "run NaN"
      next = SA.SACandidate(bio_params, run_result)

      is_new_best = SA.search_cmp!(next, search_state)
      if is_new_best || search_state.i % 50 == 0
        put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state)))
      end
      if SA.search_update_rule!(search_state)
        break
      end
      #println(convert(Float64,run_result))
    end
  finally
    stop_writer(writer_ch, writer_task)
  end

end
function main()
  seed = 20200102#1337
  Random.seed!(seed)
  rng = RNGType(rand(UInt64))
  #filter_ecos = String["8.5.3.75e", "8.5.3.75f", "8.5.3.75a", "8.5.3.75c", "8.5.3.75g", "8.3.5.65o", "8.5.3.75d", "8.5.3.75h", "8.3.5.65h", "8.3.5.65f", "8.3.5.65g", "15.4.1.76b", "8.5.3.75b", "8.5.3.75i", "9.4.7.32b", "8.3.7.35b", "8.3.7.35e", "8.5.1.63h", "8.3.7.35g", "8.3.7.35f", "8.3.5.65l", "8.3.5.65c", "9.5.1.34a"]
  #filter_ecos=String["8.3.5.65o", "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"]
  #filter_ecos=String["8.3.5.65o", "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"]
  filter_ecos = ["8.5.3.75g"]
  #filter_ecos = ["8.5.3"]
  parametrize(;
    cohorts_db_path="../data_eco_cohorts.duckdb",
    eco_field="epa_l4",
    tablename="data_eco_cohorts",
    output_dir="./outputs",
    filter_ecos=filter_ecos,
    skip_disturbances=true,
    spinup=false,
    TRIALS=1000000, rng=rng)
end

function julia_main()::Cint
  main()
  return 0
end

# ---------------------------------------------------------------------------
# Spatial (forward) simulation
# ---------------------------------------------------------------------------

function make_sites_from_communities(
  splots::DataFrame,
  eco_species_ids::Vector{Vector{Int}},
  rng::Random.AbstractRNG,
)
  site_df = unique(select(splots, [:mapcode, :eco_id]))
  sort!(site_df, :mapcode)
  n = nrow(site_df)

  splots_dict = Dict(
    key.mapcode => DataFrame(rows)
    for (key, rows) in pairs(groupby(splots, :mapcode, sort=false))
  )

  cohort_counts = Int32[nrow(splots_dict[row.mapcode]) for row in eachrow(site_df)]
  species_counts = Int32[length(eco_species_ids[row.eco_id]) for row in eachrow(site_df)]

  soa = ActiveSoA((cohort=cohort_counts, species=species_counts))
  tRNGs = [RNGType(rand(rng, UInt64)) for _ in 1:Threads.maxthreadid()]

  Threads.@threads :static for i in 1:n
    @inbounds begin
      row = site_df[i, :]
      cohorts = splots_dict[row.mapcode]
      site = getsite(soa, i)

      site.active = true
      site.rng = RNGType(rand(tRNGs[Threads.threadid()], UInt64))
      site.mapcode = row.mapcode
      site.ecocode = 0
      site.eco_id = row.eco_id
      site.ref_cn = row.mapcode
      site.old = 0
      site.live = 0
      site.B = zero(FloatType)
      site.AGNPP = zero(FloatType)
      site.harvestCapacityReduction = zero(FloatType)
      site.growthReduction = zero(FloatType)
      site.prevYearMortality = zero(FloatType)
      site.shade_class = 1
      site.c_species .= zero(UIntType)
      site.c_age .= zero(FloatType)
      site.c_bio .= zero(FloatType)
      site.c_m_tot .= zero(FloatType)
      site.c_comp .= zero(FloatType)
      site.no_establish = false
      site.sp_mature .= false
      site.sp_sprout .= false
      site.sp_seed .= false
      site.sp_serotiny .= false
      site.sp_plant .= false

      for cohort_row in eachrow(cohorts)
        BiomassSuccessionPlugin.add_cohort!(
          site,
          UIntType(cohort_row.eco_species_id),
          FloatType(cohort_row.age_calc),
          FloatType(cohort_row.agb_sum),
        )
      end
    end
  end
  return soa
end

function simulate_spatial_treemap(;
  data_dir::String,
  output_dir::String,
  eco_raster::String,
  eco_ecocode_mapping::String,
  biomass_params_path::String,
  treemap_raster::Union{String,Nothing}=nothing,
  communities_csv::Union{String,Nothing}=nothing,
  communities_db::Union{String,Nothing}=nothing,
  treemap_version::Int=2022,
  treemap_db_path::String="../data_eco_cohorts.duckdb",
  rng_seed::Int=1337,
  timehorizon_years::Int=50,
  output_every_years::Int=5,
)
  rng = RNGType(UInt64(rng_seed))

  println("Loading params: $biomass_params_path")
  params = JLD2.load_object(joinpath(data_dir, biomass_params_path))

  println("Loading eco raster")
  eco_raster_data = Data.load_eco_raster(joinpath(data_dir, eco_raster))

  eco_mapping_path = joinpath(data_dir, eco_ecocode_mapping)
  eco_mapping_df = CSV.read(eco_mapping_path, DataFrame)

  local splots_pixels, mod_params, eco_species_ids

  if !isnothing(treemap_raster)
    println("Loading treemap raster: $treemap_raster")
    @time cn_raster, _ = Data.load_treemap_raster(
      joinpath(data_dir, treemap_raster); treemap_version=treemap_version)
    @assert size(cn_raster) == size(eco_raster_data) "Raster size mismatch: treemap $(size(cn_raster)) ≠ eco $(size(eco_raster_data))"

    println("Extracting cohorts from DuckDB (treemap path)")
    @time splots, eco_list, eff_eco_list, species_list = Data.load_treemap_cohorts(
      cn_raster, eco_raster_data, treemap_db_path, eco_mapping_path)
    println("Plots: $(length(unique(splots.plt_cn))), Ecos: $(length(eco_list)), Species: $(length(species_list))")

    println("Remapping params to data eco/species")
    @time mod_params, mapped_splots, eco_species_ids = Data.map_params_to_data_treemap(
      params, eco_list, eff_eco_list, species_list, splots)

    println("Expanding cohorts to raster pixels")
    @time splots_pixels = Data.expand_to_pixels(
      mapped_splots, cn_raster, eco_raster_data)

  elseif !isnothing(communities_csv)
    println("Loading initial communities from CSV: $communities_csv")
    ic_df = Data.load_csv_communities(joinpath(data_dir, communities_csv))
    splots_pixels, mod_params, eco_species_ids = Data.prepare_general_splots(
      ic_df, eco_raster_data, eco_mapping_df, params)

  elseif !isnothing(communities_db)
    println("Loading initial communities from DuckDB: $communities_db")
    ic_df = Data.load_duckdb_communities(communities_db)
    splots_pixels, mod_params, eco_species_ids = Data.prepare_general_splots(
      ic_df, eco_raster_data, eco_mapping_df, params)

  else
    error("Specify one of: treemap_raster, communities_csv, communities_db")
  end

  n_sites = length(unique(splots_pixels.mapcode))
  println("Initializing SoA: $n_sites sites")
  @time ref_soa = make_sites_from_communities(splots_pixels, eco_species_ids, rng)

  eco_params = BiomassSuccessionPlugin.generate_eco_params(mod_params)

  mkpath(output_dir)
  writer_ch, writer_task = Spatial.start_spatial_writer(output_dir)

  println("Running simulation: $timehorizon_years years, output every $output_every_years")
  try
    Spatial.run_spatial!(ref_soa, eco_params, writer_ch;
      timehorizon=timehorizon_years, output_every=output_every_years)
  finally
    Spatial.stop_spatial_writer(writer_ch, writer_task)
  end

  println("Generating output rasters")
  ref_raster_path = !isnothing(treemap_raster) ?
                    joinpath(data_dir, treemap_raster) :
                    joinpath(data_dir, eco_raster)
  Spatial.generate_rasters_from_output(output_dir, ref_raster_path)
end

function spatial_main()
  simulate_spatial_treemap(
    data_dir="../",
    output_dir="./outputs/spatial",
    eco_raster="eco_raster.tif",
    eco_ecocode_mapping="eco_ecocode_mapping.csv",
    biomass_params_path="landis_parametrization_julia/outputs/best_params.jld2",
    treemap_raster="treemap.tif",
    treemap_version=2022,
    treemap_db_path="../data_eco_cohorts.duckdb",
    timehorizon_years=50,
    output_every_years=5,
  )
end

function simulate_spatial_landis(;
  output_dir::String,
  initial_communities_tif::String,
  ecoregion_tif::String,
  initial_communities_csv::String,
  core_species_data::String,
  spp_ecoregion_data::String,
  species_data::String,
  eco_ecocode_mapping::String,
  spp_eco_year::Int=0,
  min_rel_biomass::Vector{Float32}=Float32[0.15, 0.25, 0.50, 0.75, 0.85],
  rng_seed::Int=1337,
  timehorizon_years::Int=50,
  output_every_years::Int=5,
)
  rng = RNGType(UInt64(rng_seed))

  println("Loading LANDIS rasters")
  @time communities_raster = Data.load_landis_mapcode_raster(initial_communities_tif)
  println(communities_raster)
  @time eco_raster = Data.load_eco_raster(ecoregion_tif)
  println(eco_raster)
  @assert size(communities_raster) == size(eco_raster) "Raster size mismatch"

  println("Loading LANDIS tables")
  eco_ecocode_df = CSV.read(eco_ecocode_mapping, DataFrame)
  ic_df = CSV.read(initial_communities_csv, DataFrame)
  core_sp_df = Data.load_landis_core_species(core_species_data)
  spp_eco_df = Data.load_landis_spp_ecoregion(spp_ecoregion_data; year=spp_eco_year)
  species_df = CSV.read(species_data, DataFrame)

  println("Building BiomassSuccessionParams from LANDIS files")
  @time params = Data.make_landis_params(
    core_sp_df, species_df, spp_eco_df;
    min_rel_biomass=min_rel_biomass,
  )
  println("Ecos: $(length(params.ECO_LIST)), Species: $(length(params.SPECIES_LIST))")

  println("Expanding IC to pixels")
  @time splots = Data.expand_landis_pixels(
    communities_raster, eco_raster, ic_df, eco_ecocode_df, params)
  println("Sites: $(length(unique(splots.mapcode))), Cohort rows: $(nrow(splots))")

  eco_species_ids = Vector{Vector{Int}}(params.ECO_SPECIES_IDS)

  println("Initializing SoA")
  @time ref_soa = make_sites_from_communities(splots, eco_species_ids, rng)

  eco_params = BiomassSuccessionPlugin.generate_eco_params(params)

  mkpath(output_dir)
  writer_ch, writer_task = Spatial.start_spatial_writer(output_dir)

  println("Running simulation: $timehorizon_years years, output every $output_every_years")
  try
    Spatial.run_spatial!(ref_soa, eco_params, writer_ch;
      timehorizon=timehorizon_years, output_every=output_every_years)
  finally
    Spatial.stop_spatial_writer(writer_ch, writer_task)
  end

  println("Generating output rasters")
  Spatial.generate_rasters_from_output(output_dir, ecoregion_tif)
end

function landis_main()
  prefix = joinpath("..", "landis_data")
  simulate_spatial_landis(
    output_dir="./outputs/landis",
    initial_communities_tif=joinpath(prefix, "initial_communities.tif"),
    ecoregion_tif=joinpath(prefix, "ecoregions.tif"),
    initial_communities_csv=joinpath(prefix, "initial_communities.csv"),
    core_species_data=joinpath(prefix, "CoreSpeciesData.txt"),
    spp_ecoregion_data=joinpath(prefix, "SppEcoregionData.csv"),
    species_data=joinpath(prefix, "SpeciesData.csv"),
    eco_ecocode_mapping=joinpath(prefix, "eco_ecocode_mapping.csv"),
    spp_eco_year=0,
    timehorizon_years=50,
    output_every_years=5,
  )
end

end
