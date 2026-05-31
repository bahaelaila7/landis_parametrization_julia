module Pan
ENV["GKSwstype"] = "nul" # this is to suppress window opening upon plotting
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
import Statistics
import Distributions as Dists
import Term.Progress as TProgress
import JLD2
import Serialization
import YAML
using DataFrames
import CairoMakie
import DuckDB
import HypothesisTests




const RNGType = Random.Xoshiro #Random.MersenneTwister
const ActivePlugins = (BaseSitePlugin.BaseSite, BiomassSuccessionPlugin.BiomassSuccession)
const ActiveSoA = SiteSoA{Tuple{(ActivePlugins)...}}

function make_sites(splots::DataFrame, eco_species_ids::Vector{Vector{Int}}; rng::Random.AbstractRNG, spinup::Bool=true, no_establishment::Bool=false)
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
        site.no_establish = no_establishment
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
        site.no_establish = no_establishment
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


const _SiteInjectionYear = Vector{Tuple{Int, Vector{Tuple{UIntType,FloatType,FloatType}}}}
const _SiteInjectionDict = Dict{Int, _SiteInjectionYear}

# Only touches the sites that actually have injections (typically << n_sites).
# _new_cohort_counts is already == site.live for all other sites after process_plugin!.
function _inject_observed_cohorts!(soa, site_cohorts::_SiteInjectionYear)
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    site._new_cohort_counts = site.live + length(cohorts)
  end
  PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    for (sp, age, bio) in cohorts
      BiomassSuccessionPlugin.add_cohort!(site, sp, age, bio)
      site.B += bio
    end
  end
end

# Pre-index by sim_year → [(site_idx, cohorts)] using the ref_soa's plot→site mapping.
# Called once at setup; eliminates the per-site dict lookup inside _inject_observed_cohorts!.
function _build_injection_dict(injection_cohorts::DataFrame, ref_soa)::_SiteInjectionDict
  plot_to_site = Dict{Int,Int}(Int(getsite(ref_soa, i).ref_cn) => i for i in 1:ref_soa.n)
  by_year = _SiteInjectionDict()
  site_year_pos = Dict{Tuple{Int,Int},Int}()
  for row in eachrow(injection_cohorts)
    site_idx = get(plot_to_site, Int(row.plot_id), 0)
    site_idx == 0 && continue
    year = Int(row.sim_year)
    year_list = get!(by_year, year, _SiteInjectionYear())
    pos = get(site_year_pos, (site_idx, year), 0)
    if pos == 0
      push!(year_list, (site_idx, Tuple{UIntType,FloatType,FloatType}[]))
      pos = length(year_list)
      site_year_pos[(site_idx, year)] = pos
    end
    push!(year_list[pos][2], (UIntType(row.eco_species_id), FloatType(row.age_calc), FloatType(row.agb_sum)))
  end
  by_year
end

function generate_plots(empirical_df::DataFrame, simulated_df::DataFrame, iteration, loss, outdir="./outputs/")
  outdir = joinpath(outdir, "plots")

  mkpath(outdir)

  DictData = Dict{Tuple{UIntType,UIntType,UIntType},Tuple{Vector{UIntType},Vector{FloatType}}}
  empirical_dict::DictData = Dict()
  simulated_dict::DictData = Dict()
  for (key, group_df) in pairs(groupby(empirical_df, [:plot_id, :sim_year, :species_id]))
    empirical_dict[(UIntType(key.plot_id), UIntType(key.sim_year), UIntType(key.species_id))] = ([UIntType(row.age_calc) for row in eachrow(group_df)], [FloatType(row.agb_sum) for row in eachrow(group_df)])
  end
  for (key, group_df) in pairs(groupby(simulated_df, [:plot_id, :sim_year, :species_id]))
    simulated_dict[(UIntType(key.plot_id), UIntType(key.sim_year), UIntType(key.species_id))] = ([UIntType(row.age) for row in eachrow(group_df)], [FloatType(row.agb) for row in eachrow(group_df)])
  end

  all_keys = union(keys(empirical_dict), keys(simulated_dict))
  all_keys_df = DataFrame(all_keys, [:plot_id, :sim_year, :species_id])

  # Build plot_id → human-readable label
  plot_labels = Dict{UIntType, String}()
  if hasproperty(empirical_df, :statecd)
    has_subp = hasproperty(empirical_df, :subp)
    id_cols  = has_subp ? [:plot_id, :statecd, :unitcd, :countycd, :plot, :subp] :
                          [:plot_id, :statecd, :unitcd, :countycd, :plot]
    for row in eachrow(unique(select(empirical_df, id_cols)))
      plot_labels[UIntType(row.plot_id)] = has_subp ?
        "($(row.statecd), $(row.unitcd), $(row.countycd), $(row.plot), subp=$(row.subp))" :
        "($(row.statecd), $(row.unitcd), $(row.countycd), $(row.plot))"
    end
  end

  # Build (plot_id, species_id) → effective_species label
  species_labels = Dict{Tuple{UIntType,UIntType}, String}()
  if hasproperty(empirical_df, :effective_species)
    for row in eachrow(unique(select(empirical_df, [:plot_id, :species_id, :effective_species])))
      species_labels[(UIntType(row.plot_id), UIntType(row.species_id))] =
        string(row.effective_species)
    end
  end

  markersize_fun(x) = 3 .+ 4 .* log10.(x .+ 1)

  for (key, sim_year_df) in pairs(groupby(all_keys_df, [:plot_id, :species_id]))
    (plot_id, species_id) = key
    plot_label = get(plot_labels, plot_id, "plot_id=$(Int(plot_id))")

    sp_label  = get(species_labels, (plot_id, species_id), "sp$(Int(species_id))")
    id_safe   = replace(plot_label, r"[\(\), ]+" => "_")
    file_name = "$(id_safe)$(sp_label)@$(iteration)_$(round(loss,digits=4)).png"
    n_panels  = nrow(sim_year_df)
    f = CairoMakie.Figure(size=(800, 40 + 400 * n_panels))
    CairoMakie.Label(f[0, 1],
      "plot=$(plot_label)  species=$(sp_label)  iter=$(iteration)  loss=$(round(loss,digits=4))";
      fontsize=12, halign=:left, tellwidth=false)
    axes = []
    axi = 0
    xmin = 1000000.0
    xmax = -1.0
    for sim_year in sort(sim_year_df.sim_year)
      axi += 1
      dict_key = (plot_id, sim_year, species_id)
      em_points = get(empirical_dict, dict_key, nothing)
      sim_points = get(simulated_dict, dict_key, nothing)
      @assert !isnothing(em_points) || !isnothing(sim_points) "dict_key = $(dict_key) not in $(all_keys_df)"



      ax = CairoMakie.Axis(f[axi, 1],
        xlabel="Age (yr)",
        ylabel="Biomass (g/m²)",
        title="$(sp_label)  —  sim_year=$(sim_year)"
      )



      #p = Plots.scatter(
      #  xlabel="Age",
      #  ylabel="Biomass",
      #  title="Species: $(species_id)",
      #  legend=:bottomright,
      #  size=(800, 600)
      #)

      if !isnothing(em_points)
        xmax = max(xmax, maximum(em_points[1]))
        xmin = min(xmin, minimum(em_points[1]))
        #Plots.scatter!(
        #  p,
        #  em_points[1],
        #  em_points[2];
        #  markersize=markersize_fun(em_points[2]),
        #  markercolor=:blue,
        #  markeralpha=0.4,
        #  markerstrokewidth=0,
        #  label="Empirical"
        #)

        CairoMakie.scatter!(
          ax,
          em_points[1],
          em_points[2];
          color=(:blue, 0.4),
          markersize=markersize_fun(em_points[2]),
          strokewidth=0,
          label="Empirical"
        )

      end

      if !isnothing(sim_points)
        xmax = max(xmax, maximum(sim_points[1]))
        xmin = min(xmin, minimum(sim_points[1]))
        #Plots.scatter!(
        #  p,
        #  sim_points[1],
        #  sim_points[2];
        #  markersize=markersize_fun(sim_points[2]),
        #  markercolor=:red,
        #  markeralpha=0.4,
        #  markerstrokewidth=0,
        #  label="Simulated"
        #)
        CairoMakie.scatter!(
          ax,
          sim_points[1],
          sim_points[2];
          markersize=markersize_fun(sim_points[2]),
          color=(:red, 0.4),
          strokewidth=0,
          label="Simulated"
        )
      end

      push!(axes, ax)
    end
    # Tight vertical spacing
    CairoMakie.rowgap!(f.layout, 5)

    # Share x-axis
    CairoMakie.linkxaxes!(axes...)

    # Optional: hide repeated x decorations
    for ax in axes[1:end-1]
      CairoMakie.hidexdecorations!(ax, grid=true)
    end
    for (i, ax) in enumerate(axes)
      i == 1 && CairoMakie.axislegend(ax, position=:rb)
      ax.xminorticks = xmin:xmax
      ax.xminorgridvisible = true
    end

    filename = joinpath(
      outdir,
      file_name
    )

    CairoMakie.save(filename, f)

    println("Saved: $filename")

  end

end

struct WriterJob{State}
  is_new_best::Bool
  state::State
  splots::DataFrame
  merged_sites_state::DataFrame
  emp_sample::Union{Nothing,DataFrame}
  sim_sample::Union{Nothing,DataFrame}
  emp_sample_val::Union{Nothing,DataFrame}
  sim_sample_val::Union{Nothing,DataFrame}
end

const STOP = :stop

function start_writer(::Type{State}, output_dir::AbstractString; buffer_size::Int=8) where {State}
  ch = Channel{Union{WriterJob{State},Symbol}}(buffer_size)

  task = Threads.@spawn begin
    #ENV["MPLBACKEND"] = "Agg"
    #Plots.gr(show = false)
    #function __init__()
    #  Plots.pythonplot()
    #end
    #Plots.default(show = false)
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

          if !isnothing(job.emp_sample) && !isnothing(job.sim_sample)
            try
              generate_plots(job.emp_sample, job.sim_sample, "training_$(state.i)", convert(Float64, state.best.fx), output_dir)
            catch e
              @error "writer: generate_plots (train) failed" iter=state.i exception=(e, catch_backtrace())
            end
          end
          if !isnothing(job.emp_sample_val) && !isnothing(job.sim_sample_val)
            try
              generate_plots(job.emp_sample_val, job.sim_sample_val, "validation_$(state.i)", convert(Float64, state.best.fx), output_dir)
            catch e
              @error "writer: generate_plots (val) failed" iter=state.i exception=(e, catch_backtrace())
            end
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
function plot_biomass_bin_deltas(splots::DataFrame, loss_params::PU.LossParams; output_dir::String)
  n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
  bin_labels = [i <= length(loss_params.age_bins.bins_idx) ?
                "<$(loss_params.age_bins.bins_idx[i])" :
                "≥$(loss_params.age_bins.bins_idx[end])"
                for i in 1:n_bins]

  df = transform(splots,
    :age_calc => ByRow(a -> PU.find_age_bin(max(1, Int(round(Float64(a)))), loss_params.age_bins)) => :bin)
  filter!(row -> row.bin > 0, df)

  agg = combine(groupby(df, [:plot_id, :effective_species, :sim_year, :bin]),
                :agb_sum => sum => :agb_bin)

  delta_bins       = Int[]
  delta_agbs       = FloatType[]
  delta_species    = String[]
  contributing_ids = Set{Int}()

  for gdf in groupby(sort(agg, [:plot_id, :effective_species, :sim_year]), [:plot_id, :effective_species])
    sim_years = sort(unique(gdf.sim_year))
    length(sim_years) < 2 && continue
    push!(contributing_ids, Int(gdf.plot_id[1]))
    sp = gdf.effective_species[1]
    for si in 2:length(sim_years)
      sy_prev, sy_curr = sim_years[si-1], sim_years[si]
      mask_prev = gdf.sim_year .== sy_prev
      mask_curr = gdf.sim_year .== sy_curr
      prev = Dict(gdf.bin[i] => gdf.agb_bin[i] for i in findall(mask_prev))
      curr = Dict(gdf.bin[i] => gdf.agb_bin[i] for i in findall(mask_curr))
      for b in 1:n_bins
        agb_p = get(prev, b, 0f0)
        agb_c = get(curr, b, 0f0)
        agb_p == 0 && agb_c == 0 && continue
        push!(delta_bins, b)
        push!(delta_agbs, FloatType(agb_c - agb_p))
        push!(delta_species, sp)
      end
    end
  end

  isempty(delta_bins) && (@warn "plot_biomass_bin_deltas: no deltas found"; return)

  diag_dir = joinpath(output_dir, "diagnostics")
  mkpath(diag_dir)

  all_species = sort(unique(delta_species))
  palette     = CairoMakie.Makie.wong_colors()
  sp_colors   = [palette[mod1(i, length(palette))] for i in eachindex(all_species)]

  f  = CairoMakie.Figure(size=(1000, 600))
  ax = CairoMakie.Axis(f[1, 1];
    xlabel="Age bin",
    ylabel="ΔAGB (g/m²)",
    title="Biomass change per age bin (n=$(length(contributing_ids)) plots)",
    xticks=(1:n_bins, bin_labels))
  for (i, sp) in enumerate(all_species)
    mask = delta_species .== sp
    CairoMakie.scatter!(ax, delta_bins[mask], delta_agbs[mask];
      color=sp_colors[i], markersize=8, strokewidth=0, label=sp)
  end
  CairoMakie.hlines!(ax, [0f0]; color=:black, linewidth=1)
  CairoMakie.axislegend(ax; position=:rt)
  fname = joinpath(diag_dir, "bin_delta_all_species.png")
  CairoMakie.save(fname, f)
  println("Saved diagnostic: $fname")
end

function parametrize(; cohorts_db_path::String,
  eco_field=:epa_l4,
  tablename::String="data_eco_cohorts_g",
  output_dir::String="./outputs",
  filter_eco_field::String,
  filter_ecos::Vector{String}=String[],
  filter_plots::Vector{NTuple{4,Int}}=NTuple{4,Int}[],
  filter_species::Vector{String}=String[],
  filter_planted::Bool=false,
  skip_disturbances::Bool=true,
  bins_idx::Vector{Int64}=1:180 .|> Int64,
  smoothing_window::Vector{FloatType}=FloatType[one(FloatType)],
  spinup::Bool=true,
  search_mode::String="lbsa",
  tier::Int=3,
  TRIALS::Int=30000,
  resume_from::Union{Nothing,String}=nothing,
  force_restart_from_random::Bool=false,
  sobol_n::Int=100,
  n_reps::Int=5,
  sobol_candidates_db::Union{Nothing,String}=nothing,
  sobol_top_frac::Float64=0.5,
  n_output_plots::Int=0,
  by_subplot::Bool=false,
  no_establishment::Bool=false,
  val_frac::Float64=0.0,
  split_seed::Int=42,
  min_trees::Int=100,
  min_agb_frac::Float64=0.05,
  diagnose::Bool=false,
  rng::Random.AbstractRNG)

  mkpath(output_dir)
  # [5, 10, 20, 40, 60, 80]
  #bins_idx = vcat(5:5:30, 40:10:80, 100:20:160)
  @info bins_idx
  @info smoothing_window
  loss_params = PU.LossParams(
    age_bins=PU.AgeBins(
      bins_idx=bins_idx .|> Int,
      last_bin_open=true
    ),
    smoothing_weights=smoothing_window
  )
  split_rng = val_frac > 0.0 ? RNGType(UInt64(split_seed)) : nothing
  splots, eco_list, species_list, eco_species_ids, splots_val_raw =
    Data.prepare_parametrization_data(; cohorts_db_path=cohorts_db_path,
      eco_field=eco_field,
      tablename=tablename,
      output_dir=tablename,
      skip_disturbances=skip_disturbances,
      spinup=spinup,
      by_subplot=by_subplot,
      val_frac=val_frac,
      split_rng=split_rng,
      min_trees=min_trees,
      min_agb_frac=min_agb_frac,
      filter_eco_field=filter_eco_field,
      filter_ecos=filter_ecos,
      filter_plots=filter_plots,
      filter_species=filter_species,
      filter_planted=filter_planted,
      RNG=rng)
  n_species = length(species_list)
  n_ecoregions = length(eco_list)
  n_plots = maximum(splots.plot_id)
  println("Plots:$n_plots, Ecos:$n_ecoregions, Species:$n_species, Measurements: $(size(splots))")

  if diagnose
    plot_biomass_bin_deltas(splots, loss_params; output_dir=output_dir)
  end

  # Val preprocessing — each half is self-contained with its own contiguous plot_ids
  val_splots = nothing; val_ref_soa = nothing; val_spdf_plts = nothing
  val_site_sim_years = nothing; val_spinup_cohorts = nothing; val_injection_cohorts = nothing
  if !isnothing(splots_val_raw)
    val_max_age = Int(maximum(splots_val_raw.age_calc))
    val_spdf = PU.smoothen_ref_years(splots_val_raw, loss_params, val_max_age; debug=false)
    val_spdf_plts = Data.make_spdf_dict(val_spdf, eco_species_ids)
    val_site_sim_years = Data.get_site_sim_years(val_spdf)
    val_spinup_cohorts = DataFrame()
    val_injection_cohorts = no_establishment ? Data.get_injection_cohorts(splots_val_raw) : nothing
    val_ref_soa = make_sites(splots_val_raw, eco_species_ids; rng=rng, spinup=spinup, no_establishment=no_establishment)
    val_splots = splots_val_raw
    println("Val: $(length(unique(splots_val_raw.plot_id))) plots, $(nrow(splots_val_raw)) rows")
  end

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



  injection_cohorts = no_establishment ? Data.get_injection_cohorts(splots) : nothing
  ref_soa = make_sites(splots, eco_species_ids; rng=rng, spinup=spinup, no_establishment=no_establishment)
  if search_mode == "sobol"
    return parametrize_sobol(; ref_soa=ref_soa,
      output_dir=output_dir,
      splots=splots,
      spdf_plts=spdf_plts,
      spinup_cohorts=spinup_cohorts,
      site_sim_years=site_sim_years,
      species_list=species_list,
      eco_list=eco_list,
      eco_species_ids=eco_species_ids,
      loss_params=loss_params,
      spinup=spinup,
      no_establishment=no_establishment,
      injection_cohorts=injection_cohorts,
      rng=rng,
      debug=debug,
      N=sobol_n,
      M=n_reps,
      eval_tier=tier,
      n_output_plots=n_output_plots)
  end
  parametrize_LBSA(; ref_soa=ref_soa,
    output_dir=output_dir,
    splots=splots,
    spdf_plts=spdf_plts,
    spinup_cohorts=spinup_cohorts,
    site_sim_years=site_sim_years,
    species_list=species_list,
    eco_list=eco_list,
    eco_species_ids=eco_species_ids,
    spinup=spinup,
    search_tier=tier,
    TRIALS=TRIALS,
    resume_from=resume_from,
    force_restart_from_random=force_restart_from_random,
    n_reps=n_reps,
    sobol_candidates_db=sobol_candidates_db,
    sobol_top_frac=sobol_top_frac,
    loss_params=loss_params,
    no_establishment=no_establishment,
    injection_cohorts=injection_cohorts,
    n_output_plots=n_output_plots,
    val_splots=val_splots,
    val_ref_soa=val_ref_soa,
    val_spdf_plts=val_spdf_plts,
    val_site_sim_years=val_site_sim_years,
    val_spinup_cohorts=val_spinup_cohorts,
    val_injection_cohorts=val_injection_cohorts,
    rng=rng,
    debug=debug)
end
function accumulate_site_bins!(t1_sim_eco::Matrix{FloatType}, site, loss_params::PU.LossParams, scratch_perm::Vector{Int}, scratch_ages::Vector{FloatType})
  site.live == 0 && return
  nlive = site.live
  c_species = @view site.c_species[1:nlive]
  max_age = Int(ceil(maximum(@view site.c_age[1:nlive]))) + length(loss_params.smoothing_weights) >> 1
  if length(scratch_perm) < nlive
    resize!(scratch_perm, nlive)
  end
  if length(scratch_ages) < max_age
    resize!(scratch_ages, max_age)
  end
  p = @view scratch_perm[1:nlive]
  sortperm!(p, c_species)
  ages = @view scratch_ages[1:max_age]
  sp_start = 1
  prev_sp = c_species[p[1]]
  function conclude!(sp, s, e)
    fill!(ages, 0f0)
    for a in @view p[s:e]
      ages[clamp(Int(ceil(site.c_age[a])), 1, max_age)] += site.c_bio[a]
    end
    agb = sum(ages)
    cdf = PU.smoothen_bin_cdf(ages; w=loss_params.smoothing_weights, age_bins=loss_params.age_bins)
    t1_sim_eco[sp, 1] += cdf[1] * agb
    for k in 2:length(cdf)
      t1_sim_eco[sp, k] += (cdf[k] - cdf[k-1]) * agb
    end
  end
  for i in eachindex(p)
    sp = c_species[p[i]]
    if sp != prev_sp
      conclude!(prev_sp, sp_start, i - 1)
      prev_sp = sp
      sp_start = i
    end
    if i == length(p)
      conclude!(sp, sp_start, i)
    end
  end
end

function calculate_aggregate_loss(t1_sim::Vector{Matrix{FloatType}}, t1_ref::Vector{Matrix{FloatType}}, loss_params::PU.LossParams, n_species::Int, eco_species_ids::Vector{Vector{Int}}, eco_site_counts::Vector{Int}, eco_obs_counts::Vector{Int})::Vector{PU.SiteLoss}
  eco_losses = Vector{PU.SiteLoss}(undef, length(t1_sim))
  for eco_id in eachindex(t1_sim)
    sp_w_loss = zeros(FloatType, n_species)
    sp_agb_loss = zeros(FloatType, n_species)
    sp_map = eco_species_ids[eco_id]
    for sp_eco in eachindex(sp_map)
      gsp = sp_map[sp_eco]
      sim_row = @view t1_sim[eco_id][sp_eco, :]
      ref_row = @view t1_ref[eco_id][sp_eco, :]
      let bw = loss_params.age_bins.bin_widths
        s = zero(FloatType)
        acc_sim = zero(FloatType)
        acc_ref = zero(FloatType)
        @inbounds for k in eachindex(bw)
          acc_sim += sim_row[k]
          acc_ref += ref_row[k]
          s += bw[k] * abs(acc_sim - acc_ref)
        end
        sp_w_loss[gsp] = s
      end
      sp_agb_loss[gsp] = abs(sum(sim_row) - sum(ref_row))
    end
    eco_losses[eco_id] = PU.SiteLoss(sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=zero(FloatType), num_sites=eco_site_counts[eco_id], num_obs=eco_obs_counts[eco_id])
  end
  return eco_losses
end


function calculate_t2_loss(t2_sim_bins, t2_sim_total, t2_ref, n_species, eco_species_ids, eco_site_counts, eco_obs_counts)
  eco_losses = Vector{PU.SiteLoss}(undef, length(eco_species_ids))
  n_bins = size(t2_ref.bins[1], 2)
  for eco_id in eachindex(eco_species_ids)
    sp_w_loss = zeros(FloatType, n_species)
    sp_agb_loss = zeros(FloatType, n_species)
    sp_map = eco_species_ids[eco_id]
    for sp_eco in eachindex(sp_map)
      gsp = sp_map[sp_eco]
      for b in 1:n_bins
        sp_w_loss[gsp] += PU.wasserstein1d(t2_sim_bins[eco_id][sp_eco, b], t2_ref.bins[eco_id][sp_eco, b])
      end
      sp_agb_loss[gsp] = PU.wasserstein1d(t2_sim_total[eco_id][sp_eco], t2_ref.total[eco_id][sp_eco])
    end
    eco_losses[eco_id] = PU.SiteLoss(sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=zero(FloatType), num_sites=eco_site_counts[eco_id], num_obs=eco_obs_counts[eco_id])
  end
  return eco_losses
end

function fit_params(ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier::Int=3, t1_ref::Union{Nothing,Vector{Matrix{FloatType}}}=nothing, t2_ref=nothing, seeds::AbstractVector=[nothing], injection_dict=nothing, injection_years=Set{Int}())
  return map(seeds) do seed
    soa = copy_and_reseed_soa(ref_soa, seed)
    eco_params = BiomassSuccessionPlugin.generate_eco_params(bio_params)
    ctx = (BiomassSuccession=(eco_params=eco_params,),)
    years_results = Vector{PU.SiteLoss}(undef, max_sim_year + 1)
    starting_sim_year = 1
    if spinup
      soa = BiomassSuccessionPlugin.spinup_cohorts!(soa, spinup_cohorts, eco_params)
      starting_sim_year = 0 # with spinup, even the first year of vegetation data is to be matched and compared
    end
    sites_data = [Tuple{UIntType,Int,UIntType,UIntType,FloatType}[] for _ in 1:soa.n]
    if search_tier == 1
      eco_site_counts = zeros(Int, length(eco_species_ids))
      eco_obs_counts = zeros(Int, length(eco_species_ids))
      for i in 1:soa.n
        site = getsite(soa, i)
        !site.active && continue
        eco_site_counts[site.eco_id] += 1
        eco_obs_counts[site.eco_id] += count(>=(starting_sim_year), site_sim_years.sim_years[site.mapcode])
      end
      n_bins = size(t1_ref[1], 2)
      t1_sim_t = [[zeros(FloatType, length(eco_species_ids[eco_id]), n_bins) for eco_id in eachindex(eco_species_ids)] for _ in 1:Threads.maxthreadid()]
      max_cohorts_scratch = Int(maximum(soa.refs.cohort[i+1] - soa.refs.cohort[i] for i in 1:soa.n))
      max_age_scratch = max_sim_year + max(1, length(loss_params.smoothing_weights) >> 1) + 5
      scratch_perm_t = [Vector{Int}(undef, max_cohorts_scratch) for _ in 1:Threads.maxthreadid()]
      scratch_ages_t = [Vector{FloatType}(undef, max_age_scratch) for _ in 1:Threads.maxthreadid()]
    elseif search_tier == 2
      n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
      eco_site_counts = zeros(Int, length(eco_species_ids))
      eco_obs_counts = zeros(Int, length(eco_species_ids))
      for i in 1:soa.n
        site = getsite(soa, i)
        !site.active && continue
        eco_site_counts[site.eco_id] += 1
        eco_obs_counts[site.eco_id] += count(>=(starting_sim_year), site_sim_years.sim_years[site.mapcode])
      end
      t2_sim_bins_t = [[[FloatType[] for sp in 1:length(eco_species_ids[eco_id]), b in 1:n_bins] for eco_id in eachindex(eco_species_ids)] for _ in 1:Threads.maxthreadid()]
      t2_sim_total_t = [[[FloatType[] for _ in 1:length(eco_species_ids[eco_id])] for eco_id in eachindex(eco_species_ids)] for _ in 1:Threads.maxthreadid()]
    end
    for current_sim_year in starting_sim_year:max_sim_year
      #println("\ttimestep $(t)")
      PanCore.process_plugin!(soa, BiomassSuccessionPlugin.BiomassSuccession, current_sim_year; ctx=ctx.BiomassSuccession)
      if !isnothing(injection_dict) && current_sim_year in injection_years
        _inject_observed_cohorts!(soa, injection_dict[current_sim_year])
      end
      if search_tier == 1
        Threads.@threads :static for i in 1:soa.n
          @inbounds begin
            site = getsite(soa, i)
            !site.active && continue
            sim_years = site_sim_years.sim_years[site.mapcode]
            if current_sim_year in sim_years
              accumulate_site_bins!(t1_sim_t[Threads.threadid()][site.eco_id], site, loss_params, scratch_perm_t[Threads.threadid()], scratch_ages_t[Threads.threadid()])
              for j in 1:site.live
                push!(sites_data[i], (site.ref_cn, current_sim_year, site.c_species[j], UIntType(site.c_age[j]), site.c_bio[j]))
              end
            end
            if current_sim_year == last(sim_years)
              site.active = false
            end
          end
        end
      elseif search_tier == 2
        Threads.@threads :static for i in 1:soa.n
          @inbounds begin
            site = getsite(soa, i)
            !site.active && continue
            sim_years = site_sim_years.sim_years[site.mapcode]
            if current_sim_year in sim_years
              eco_id = Int(site.eco_id)
              tid = Threads.threadid()
              n_sp_eco = length(eco_species_ids[eco_id])
              sp_bin_agbs = zeros(FloatType, n_sp_eco, n_bins)
              for j in 1:site.live
                sp_eco = Int(site.c_species[j])
                age = Int(ceil(Float64(site.c_age[j])))
                b = PU.find_age_bin(age, loss_params.age_bins)
                b == 0 && continue
                sp_bin_agbs[sp_eco, b] += site.c_bio[j]
              end
              for sp_eco in 1:n_sp_eco
                sp_total = zero(FloatType)
                for b in 1:n_bins
                  sp_total += sp_bin_agbs[sp_eco, b]
                  push!(t2_sim_bins_t[tid][eco_id][sp_eco, b], sp_bin_agbs[sp_eco, b])
                end
                push!(t2_sim_total_t[tid][eco_id][sp_eco], sp_total)
              end
              for j in 1:site.live
                push!(sites_data[i], (site.ref_cn, current_sim_year, site.c_species[j], UIntType(site.c_age[j]), site.c_bio[j]))
              end
            end
            if current_sim_year == last(sim_years)
              site.active = false
            end
          end
        end
      else
        sites_results = Vector{PU.SiteLoss}(undef, soa.n)
        Threads.@threads :static for i in 1:soa.n
          @inbounds begin
            site = getsite(soa, i)
            !site.active && continue
            spdf_plt = spdf_plts[(site.ref_cn, site.eco_id)]
            sim_years = site_sim_years.sim_years[site.mapcode]
            if current_sim_year in sim_years
              sloss = PU.calculate_site_loss2(current_sim_year, site, n_species, eco_species_ids, spdf_plt[current_sim_year], loss_params; debug=debug)
              sites_results[i] = sloss
              for j in 1:site.live
                push!(sites_data[i], (site.ref_cn, current_sim_year, site.c_species[j], UIntType(site.c_age[j]), site.c_bio[j]))
              end
            end
            if current_sim_year == last(sim_years)
              site.active = false
            end
          end
        end
        year_results_no_missing = PU.skipundef(sites_results)
        if length(year_results_no_missing) > 0
          years_results[current_sim_year+1] = sum(year_results_no_missing)
        end
      end
    end
    if search_tier == 1
      for tid in 2:Threads.maxthreadid()
        for eco_id in eachindex(eco_species_ids)
          t1_sim_t[1][eco_id] .+= t1_sim_t[tid][eco_id]
        end
      end
      eco_losses = calculate_aggregate_loss(t1_sim_t[1], t1_ref, loss_params, n_species, eco_species_ids, eco_site_counts, eco_obs_counts)
      run_result = sum(eco_losses)
    elseif search_tier == 2
      t2_sim_bins = [[vcat((t2_sim_bins_t[tid][eco_id][sp, b] for tid in 1:Threads.maxthreadid())...) for sp in 1:length(eco_species_ids[eco_id]), b in 1:n_bins] for eco_id in eachindex(eco_species_ids)]
      t2_sim_total = [[vcat((t2_sim_total_t[tid][eco_id][sp] for tid in 1:Threads.maxthreadid())...) for sp in 1:length(eco_species_ids[eco_id])] for eco_id in eachindex(eco_species_ids)]
      eco_losses = calculate_t2_loss(t2_sim_bins, t2_sim_total, t2_ref, n_species, eco_species_ids, eco_site_counts, eco_obs_counts)
      run_result = sum(eco_losses)
    else
      eco_losses = nothing
      run_result = sum(PU.skipundef(years_results))
    end
    #@assert !any(isnan.(run_result.sp_w_loss)) "run NaN"
    cached_sites_state = [cohort for cohorts in sites_data for cohort in cohorts]
    (run_result, cached_sites_state, eco_losses)
  end
end
function parametrize_sobol(; ref_soa::ActiveSoA, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, rng::Random.AbstractRNG, debug::Bool, N::Int=100, M::Int=5, eval_tier::Int=3, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing)
  n_species = length(species_list)
  n_ecoregions = length(eco_list)
  max_sim_year = site_sim_years.sim_years .|> maximum |> maximum

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
  else
    t1_ref = nothing
    t2_ref = nothing
  end

  param_dists = BSP.make_biomass_param_dists(n_species, n_ecoregions, eco_species_ids; no_establishment=no_establishment)
  initial_params = BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
  injection_dict  = isnothing(injection_cohorts) ? nothing : _build_injection_dict(injection_cohorts, ref_soa)
  injection_years = isnothing(injection_cohorts) ? Set{Int}() : Set(Int.(injection_cohorts.sim_year))
  samples = PU.sobol_samples(param_dists, initial_params, N)
  #println(samples)

  ResultT = @NamedTuple{params::typeof(initial_params), mean_loss::FloatType, std_loss::FloatType, median_loss::FloatType, losses::Vector{FloatType}}
  results = ResultT[]

  losses_db_file = DuckDB.DB(joinpath(output_dir, "losses.duckdb"))
  losses_db = DuckDB.connect(losses_db_file)
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS sobol_results (run_id VARCHAR, sobol_idx INTEGER, mean_loss DOUBLE, std_loss DOUBLE, median_loss DOUBLE, params_blob BLOB)")
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS sobol_eval_losses (run_id VARCHAR, sobol_idx INTEGER, eval_idx INTEGER, loss DOUBLE)")
  run_id = string(Dates.now())
  eval_seeds = [rand(rng, UInt64) for _ in 1:M]

  try
    TProgress.@track for i in 1:N
      bio_params = samples[i]
      rep_results = fit_params(ref_soa, bio_params, max_sim_year, n_species,
        eco_species_ids, spdf_plts, site_sim_years,
        spinup, spinup_cohorts, loss_params;
        debug=debug, search_tier=eval_tier, t1_ref=t1_ref, t2_ref=t2_ref, seeds=eval_seeds,
        injection_dict=injection_dict, injection_years=injection_years)
      losses = FloatType[PU.get_total_loss(r[1]) for r in rep_results]
      μ = FloatType(sum(losses) / M)
      σ = FloatType(sqrt(sum((l - μ)^2 for l in losses) / max(M - 1, 1)))
      sorted = sort(losses)
      med = FloatType(M % 2 == 1 ? sorted[M÷2+1] : (sorted[M÷2] + sorted[M÷2+1]) / 2)
      push!(results, (; params=bio_params, mean_loss=μ, std_loss=σ, median_loss=med, losses=losses))
      buf = IOBuffer()
      Serialization.serialize(buf, bio_params)
      DuckDB.execute(losses_db, "INSERT INTO sobol_results VALUES (?, ?, ?, ?, ?, ?)",
        [run_id, i, Float64(μ), Float64(σ), Float64(med), take!(buf)])
      for (ei, loss) in enumerate(losses)
        DuckDB.execute(losses_db, "INSERT INTO sobol_eval_losses VALUES (?, ?, ?, ?)",
          [run_id, i, ei, Float64(loss)])
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
      emp_sample  = _make_emp_df(splots, sampled_ids)
      best_params = results[1].params
      best_soa    = make_sites(splots, eco_species_ids; rng, spinup, no_establishment=no_establishment)
      best_result = only(fit_params(best_soa, best_params, site_sim_years.sim_years .|> maximum |> maximum,
                                    length(species_list), eco_species_ids, spdf_plts,
                                    site_sim_years, spinup, spinup_cohorts, loss_params;
                                    debug, search_tier=eval_tier, t1_ref=t1_ref, t2_ref=t2_ref,
                                    seeds=[rand(rng, UInt64)],
                                    injection_dict=injection_dict, injection_years=injection_years))
      sim_sample  = _filter_cached_to_df(best_result[2], sampled_ids)
      generate_plots(emp_sample, sim_sample, "sobol_best", convert(Float64, PU.get_total_loss(best_result[1])), output_dir)
    end
  end
  return results
end

function _sample_plot_ids(all_ids::Vector{UIntType}, n::Int, rng;
                          injection_cohorts::Union{Nothing,DataFrame}=nothing)
  n == 0 && return Set{UIntType}()
  if isnothing(injection_cohorts) || isempty(injection_cohorts)
    return Set(Random.shuffle(rng, all_ids)[1:min(n, length(all_ids))])
  end
  inj_set    = Set(UIntType.(injection_cohorts.plot_id))
  with_inj   = filter(id ->  id in inj_set, all_ids)
  without_inj = filter(id -> !(id in inj_set), all_ids)
  n_inj   = min(n ÷ 2, length(with_inj))
  n_clean = min(n - n_inj, length(without_inj))
  n_inj   = min(n - n_clean, length(with_inj))   # backfill if clean side was small
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

function _median_rep_cached(rep_results)
  losses = Float64[convert(Float64, PU.get_total_loss(r[1])) for r in rep_results]
  rep_results[argmin(abs.(losses .- Statistics.median(losses)))][2]
end

function _filter_cached_to_df(cached, sampled_ids::Set)
  filtered = filter(t -> UIntType(t[1]) in sampled_ids, cached)
  isempty(filtered) && return DataFrame(plot_id=UIntType[], sim_year=Int[], species_id=UIntType[], age=UIntType[], agb=FloatType[])
  DataFrame(filtered, [:plot_id, :sim_year, :species_id, :age, :agb])
end

function parametrize_LBSA(; ref_soa::ActiveSoA, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=3, resume_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing)
  splots.sim_year .= Dates.value.(Dates.Day.(splots.measdate - splots.start_measdate)) ./ 365.25 .|> round .|> Int

  # Fixed plot sample chosen once at startup so progress is comparable across iterations
  all_plot_ids = UIntType.(unique(splots.plot_id))
  sampled_ids  = _sample_plot_ids(all_plot_ids, n_output_plots, rng; injection_cohorts=injection_cohorts)
  emp_sample   = n_output_plots > 0 ? _make_emp_df(splots, sampled_ids) : nothing

  n_species = length(species_list)
  max_sim_year = site_sim_years.sim_years .|> maximum |> maximum
  param_dists = BSP.make_biomass_param_dists(length(species_list), length(eco_list), eco_species_ids; no_establishment=no_establishment)
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  injection_dict  = isnothing(injection_cohorts) ? nothing : _build_injection_dict(injection_cohorts, ref_soa)
  injection_years = isnothing(injection_cohorts) ? Set{Int}() : Set(Int.(injection_cohorts.sim_year))

  # Validation setup
  have_val       = !isnothing(val_ref_soa)
  inj_dict_val   = (have_val && !isnothing(val_injection_cohorts)) ? _build_injection_dict(val_injection_cohorts, val_ref_soa) : nothing
  inj_years_val  = (have_val && !isnothing(val_injection_cohorts)) ? Set(Int.(val_injection_cohorts.sim_year)) : Set{Int}()
  sampled_ids_val = (have_val && n_output_plots > 0) ?
    _sample_plot_ids(UIntType.(unique(val_splots.plot_id)), n_output_plots, rng; injection_cohorts=val_injection_cohorts) :
    Set{UIntType}()
  emp_sample_val = (have_val && n_output_plots > 0) ? _make_emp_df(val_splots, sampled_ids_val) : nothing

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
  else
    t1_ref = nothing
    t2_ref = nothing
  end
  if isnothing(resume_from)
    bio_params = if !isempty(_sobol_cands)
      @info "Using Sobol candidate 1/$(length(_sobol_cands)) as initial point"
      _sobol_cands[1]
    else
      BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    end
    best_result = sum(r[1] for r in fit_params(ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, seeds=[rand(rng, UInt64) for _ in 1:n_reps], injection_dict=injection_dict, injection_years=injection_years))
    cur = LBSA.LBSACandidate(bio_params, best_result)
    search_state = LBSA.LBSAState(cur, cur, rng; max_iter=TRIALS)
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
  next_candidate() = if search_state.sobol_cand_idx <= length(_sobol_cands)
    p = _sobol_cands[search_state.sobol_cand_idx]
    @info "Using Sobol candidate $(search_state.sobol_cand_idx)/$(length(_sobol_cands))"
    search_state.sobol_cand_idx += 1
    p
  else
    BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
  end

  writer_ch, writer_task = start_writer(typeof(search_state), output_dir)

  losses_db_file = DuckDB.DB(joinpath(output_dir, "losses.duckdb"))
  losses_db = DuckDB.connect(losses_db_file)
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS total_loss (iteration INTEGER, n_sites INTEGER, n_obs INTEGER, total_loss DOUBLE, params_blob BLOB)")
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS ecoregion_loss (iteration INTEGER, ecoregion VARCHAR, eco_num_sites INTEGER, eco_num_obs INTEGER, ecoregion_total_loss DOUBLE)")
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS species_loss (iteration INTEGER, ecoregion VARCHAR, species VARCHAR, age_dist_loss DOUBLE, agb_loss DOUBLE)")

  baseline_weights = diff([0.0; param_dists.weights_cumsum])
  param_sensitivities = zeros(length(baseline_weights))
  sensitivity_decay = 0.999      # per-trial decay toward 0 for all params
  sensitivity_ema_alpha = 0.1    # EMA weight for chosen param's sensitivity update
  sensitivity_lambda = 3.0       # max weight boost at full sensitivity (baseline × (1 + λ))

  try

    TProgress.@track for trial in (search_state.i+1):TRIALS
      #dynamic_weights = baseline_weights .* (1.0 .+ sensitivity_lambda .* param_sensitivities)
      #dynamic_weights ./= sum(dynamic_weights)
      #dynamic_cumsum = cumsum(dynamic_weights)
      bio_params, chosen_param_idx = PU.mutate_params(bio_params, param_dists; rng=rng, mutation_mode=PU.BothMutations)
      #ctx=PU.SamplingContext(search_state.current.fx.sp_w_loss .+ search_state.current.fx.sp_agb_loss .+ (search_state.current.fx.sp_w_loss .* search_state.current.fx.sp_agb_loss)),
      #dynamic_cumsum=dynamic_cumsum)
      for _ in 0:rand(rng, 0:2)
        bio_params, _ = PU.mutate_params(bio_params, param_dists; rng=rng, mutation_mode=PU.BothMutations)#,  #BothMutations,
      end

      rep_results = fit_params(ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, seeds=[rand(rng, UInt64) for _ in 1:n_reps], injection_dict=injection_dict, injection_years=injection_years)
      run_result = sum(r[1] for r in rep_results)
      cached_sites_state = _median_rep_cached(rep_results)
      eco_losses = isnothing(rep_results[1][3]) ? nothing : [sum(r[3][e] for r in rep_results) for e in eachindex(rep_results[1][3])]

      current_loss = convert(Float64, search_state.current.fx)
      delta_loss = abs(convert(Float64, run_result) - current_loss)
      norm_delta = delta_loss / (current_loss + 1e-10)
      param_sensitivities .*= sensitivity_decay
      param_sensitivities[chosen_param_idx] = (1.0 - sensitivity_ema_alpha) * param_sensitivities[chosen_param_idx] + sensitivity_ema_alpha * min(norm_delta, 1.0)

      next = LBSA.LBSACandidate(bio_params, run_result)

      is_new_best = LBSA.search_cmp!(next, search_state)
      #if is_new_best
      #  put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state), merged_sites_state))
      #  plot(cached_sites_state)
      #end
      if LBSA.should_restart(search_state)
        @info "Restarting @ $(search_state.i)"
        bio_params = next_candidate()
        cur_result = sum(r[1] for r in fit_params(ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, seeds=[rand(rng, UInt64) for _ in 1:n_reps], injection_dict=injection_dict, injection_years=injection_years))
        is_new_best = LBSA.restart(search_state, LBSA.LBSACandidate(bio_params, cur_result))
      end
      val_sim_sample = nothing
      if is_new_best
        iter = search_state.best_iteration
        total = convert(Float64, search_state.best.fx)
        @info "New best @ $iter | loss=$total"
        try
          test_df = simulate_and_test(; splots=splots, bio_params=search_state.best.x, eco_list=eco_list, species_list=species_list, eco_species_ids=eco_species_ids, loss_params=loss_params, site_sim_years=site_sim_years, M=n_reps, no_establishment=no_establishment, rng=rng)
          println("Train stats:"); show(test_df; allrows=true, allcols=true); println()
        catch e
          @warn "simulate_and_test (train) failed" exception=(e, catch_backtrace())
        end
        if have_val
          try
            val_result = only(fit_params(val_ref_soa, search_state.best.x, max_sim_year, n_species,
              eco_species_ids, val_spdf_plts, val_site_sim_years,
              spinup, val_spinup_cohorts, loss_params;
              debug=false, search_tier=3,
              injection_dict=inj_dict_val, injection_years=inj_years_val,
              seeds=[rand(rng, UInt64)]))
            val_total = convert(Float64, PU.get_total_loss(val_result[1]))
            @info "Val loss @ $iter | loss=$val_total"
            try
              val_test_df = simulate_and_test(; splots=val_splots, bio_params=search_state.best.x, eco_list=eco_list, species_list=species_list, eco_species_ids=eco_species_ids, loss_params=loss_params, site_sim_years=val_site_sim_years, M=n_reps, no_establishment=no_establishment, rng=rng)
              println("Val stats:"); show(val_test_df; allrows=true, allcols=true); println()
            catch e
              @warn "simulate_and_test (val) failed" exception=(e, catch_backtrace())
            end
            val_sim_sample = n_output_plots > 0 ? _filter_cached_to_df(val_result[2], sampled_ids_val) : nothing
          catch e
            @warn "val fit_params failed" exception=(e, catch_backtrace())
          end
        end
        let buf = IOBuffer()
          Serialization.serialize(buf, search_state.best.x)
          DuckDB.execute(losses_db, "INSERT INTO total_loss VALUES (?, ?, ?, ?, ?)", [iter, run_result.num_sites, run_result.num_obs, total, take!(buf)])
        end
        if !isnothing(eco_losses)
          for (eco_id, eco_loss) in enumerate(eco_losses)
            eco_name = eco_list[eco_id]
            eco_total = convert(Float64, PU.get_total_loss(eco_loss))
            DuckDB.execute(losses_db, "INSERT INTO ecoregion_loss VALUES (?, ?, ?, ?, ?)", [iter, eco_name, eco_loss.num_sites, eco_loss.num_obs, eco_total])
            n = max(1, eco_loss.num_sites)
            for gsp in 1:n_species
              eco_loss.sp_w_loss[gsp] == 0f0 && continue
              DuckDB.execute(losses_db, "INSERT INTO species_loss VALUES (?, ?, ?, ?, ?)", [iter, eco_name, species_list[gsp], eco_loss.sp_w_loss[gsp] / n, eco_loss.sp_agb_loss[gsp] / n])
            end
          end
        end
      end
      if is_new_best || search_state.i % 50 == 0
        cached_sites_state_df = DataFrame(cached_sites_state, [:plot_id, :sim_year, :species_id, :age, :agb])
        sim_sample = (is_new_best && n_output_plots > 0) ? _filter_cached_to_df(cached_sites_state, sampled_ids) : nothing
        put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state), splots, cached_sites_state_df, emp_sample, sim_sample, is_new_best ? emp_sample_val : nothing, val_sim_sample))
      end
      if LBSA.is_search_over(search_state)
        break
      end
      #println(convert(Float64,run_result))
    end
  finally
    stop_writer(writer_ch, writer_task)
    close(losses_db_file)
    try
      mkpath(output_dir)
      fname = "search_state@$(search_state.i).jld2"
      JLD2.save_object(joinpath(output_dir, fname), search_state)
      link_path = joinpath(output_dir, "search_state_latest.jld2")
      islink(link_path) && rm(link_path)
      symlink(fname, link_path)
      @info "Search state saved @ $(search_state.i)"
    catch e
      @error "Failed to save search state on exit" exception = (e, catch_backtrace())
    end
  end

end

function load_best_params(db_path::String; iteration::Union{Nothing,Int}=nothing)
  db = DuckDB.DB(db_path)
  con = DuckDB.connect(db)
  sql = isnothing(iteration) ?
        "SELECT params_blob FROM total_loss ORDER BY total_loss ASC LIMIT 1" :
        "SELECT params_blob FROM total_loss WHERE iteration = $iteration LIMIT 1"
  result = DuckDB.execute(con, sql) |> DataFrame
  close(db)
  isempty(result) && return nothing
  return Serialization.deserialize(IOBuffer(result.params_blob[1]))
end

function load_sobol_candidates(db_path::String; top_frac::Float64=0.5, run_id::Union{Nothing,String}=nothing)
  isfile(db_path) || return []
  db = DuckDB.DB(db_path)
  con = DuckDB.connect(db)
  try
    sql = isnothing(run_id) ?
          "SELECT params_blob FROM sobol_results ORDER BY mean_loss ASC" :
          "SELECT params_blob FROM sobol_results WHERE run_id = '$(run_id)' ORDER BY mean_loss ASC"
    result = DuckDB.execute(con, sql) |> DataFrame
    isempty(result) && return []
    n_keep = max(1, round(Int, nrow(result) * top_frac))
    @info "Loaded $(n_keep) Sobol candidates (top $(round(Int, top_frac*100))% of $(nrow(result)))"
    return [Serialization.deserialize(IOBuffer(row.params_blob)) for row in eachrow(result[1:n_keep, :])]
  catch
    return []
  finally
    close(db)
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

      bio_params, _ = PU.mutate_params(bio_params, param_dists; rng=rng)
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
              #cache_site_data()
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

function run_from_yaml(yaml_path::String)
  cfg = YAML.load_file(yaml_path)
  get_cfg(key, default) = get(cfg, key, default)

  seed = get_cfg("seed", 404)
  Random.seed!(seed)
  rng = RNGType(rand(UInt64))

  filter_plots = NTuple{4,Int}[
    NTuple{4,Int}([x isa AbstractString ? parse(Int, x) : Int(x) for x in p])
    for p in get_cfg("filter_plots", [])
  ]

  sw_size = get_cfg("smoothing_window_size", 0)
  smoothing_window = sw_size > 0 ?
                     PU.get_smoothing_window(; smoothing_window=sw_size, smoothing_variance=FloatType(get_cfg("smoothing_variance", 1.0))) :
                     FloatType[one(FloatType)]

  resume_from = get_cfg("resume_from", nothing)
  resume_from = (resume_from === nothing || resume_from == "null") ? nothing : String(resume_from)
  sobol_candidates_db = get_cfg("sobol_candidates_db", nothing)
  sobol_candidates_db = (sobol_candidates_db === nothing || sobol_candidates_db == "null") ? nothing : String(sobol_candidates_db)

  # Fields shared by parametrize, plot_sample, and plot_sample_sobol
  common_kw = (
    cohorts_db_path   = get_cfg("cohorts_db_path", "../data_eco_cohorts.duckdb"),
    filter_eco_field  = get_cfg("filter_eco_field", "epa_l3"),
    eco_field         = get_cfg("eco_field", "epa_l3"),
    tablename         = get_cfg("tablename", "data_eco_cohorts"),
    output_dir        = get_cfg("output_dir", "./outputs"),
    filter_ecos       = String.(get_cfg("filter_ecos", String[])),
    filter_plots      = filter_plots,
    filter_species    = String.(get_cfg("filter_species", String[])),
    skip_disturbances = get_cfg("skip_disturbances", true),
    spinup            = get_cfg("spinup", false),
    by_subplot        = get_cfg("by_subplot", false),
    no_establishment  = get_cfg("no_establishment", false),
    min_trees         = Int(get_cfg("min_trees", 100)),
    min_agb_frac      = Float64(get_cfg("min_agb_frac", 0.05)),
    bins_idx          = Int.(get_cfg("bins_idx", vcat(10:10:40, 60:20:120))),
  )
  n_output_plots = get_cfg("n_output_plots", 0)
  sw_kw = (smoothing_window_size = get_cfg("smoothing_window_size", 0),
           smoothing_variance    = Float64(get_cfg("smoothing_variance", 1.2)))

  search_mode = get_cfg("search_mode", "lbsa")

  if search_mode == "plot_only"
    params_path = String(get_cfg("params_path", ""))
    isempty(params_path) && error("search_mode=plot_only requires params_path in yaml")
    return plot_sample(; common_kw..., sw_kw...,
                         params_path=params_path,
                         n_output_plots=max(1, n_output_plots),
                         rng_seed=seed)
  end

  if search_mode == "plot_only_sobol"
    sobol_db = String(get_cfg("sobol_candidates_db", ""))
    isempty(sobol_db) && error("search_mode=plot_only_sobol requires sobol_candidates_db in yaml")
    return plot_sample_sobol(; common_kw..., sw_kw...,
                               sobol_candidates_db=sobol_db,
                               n_output_plots=max(1, n_output_plots),
                               n_sobol_params_to_plot=max(1, get_cfg("n_sobol_params_to_plot", 5)),
                               rng_seed=seed)
  end

  parametrize(;
    common_kw...,
    search_mode         = search_mode,
    tier                = get_cfg("tier", 1),
    smoothing_window    = smoothing_window,
    TRIALS              = get_cfg("trials", 1000000),
    resume_from         = resume_from,
    force_restart_from_random = get_cfg("force_restart_from_random", false),
    sobol_n             = get_cfg("sobol_n", 100),
    n_reps              = get_cfg("n_reps", 5),
    sobol_candidates_db = sobol_candidates_db,
    sobol_top_frac      = Float64(get_cfg("sobol_top_frac", 0.5)),
    n_output_plots      = n_output_plots,
    val_frac            = Float64(get_cfg("val_frac", 0.0)),
    split_seed          = Int(get_cfg("split_seed", 42)),
    diagnose            = get_cfg("diagnose", false),
    rng                 = rng)
end

function main()
  if length(ARGS) >= 1 && (endswith(ARGS[1], ".yaml") || endswith(ARGS[1], ".yml"))
    return run_from_yaml(ARGS[1])
  end
  seed = 12312315 #404 #1337
  Random.seed!(seed)
  rng = RNGType(rand(UInt64))
  #filter_ecos = String["8.5.3.75e", "8.5.3.75f", "8.5.3.75a", "8.5.3.75c", "8.5.3.75g", "8.3.5.65o", "8.5.3.75d", "8.5.3.75h", "8.3.5.65h", "8.3.5.65f", "8.3.5.65g", "15.4.1.76b", "8.5.3.75b", "8.5.3.75i", "9.4.7.32b", "8.3.7.35b", "8.3.7.35e", "8.5.1.63h", "8.3.7.35g", "8.3.7.35f", "8.3.5.65l", "8.3.5.65c", "9.5.1.34a"]
  #filter_ecos = String["8.5.3", "8.3.5", "15.4.1", "9.4.7", "8.3.7", "8.5.1", "9.5.1"]
  #filter_ecos=String["8.3.5.65o", "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"]
  #filter_ecos = String["8.3.5.65o", "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"]
  #filter_ecos = String["8.5.3.75e", "8.5.3.75f", "8.5.3.75g"]
  #filter_ecos = String["8.5.3.75f"]
  #filter_ecos = String["8.5.3.75g"]
  #filter_ecos = ["8.5.3.75g"]
  filter_ecos = ["8.5.3"]
  filter_plots = NTuple{4,Int}[]#(13, 1, 49, 26)]  # e.g. [(statecd,unitcd,countycd,plot), ...]
  filter_species = String[] #String["PIEL"]        # e.g. ["ACRU", "QURU"]
  # bins_idx = vcat(10:10:40, 60:20:120)
  t1_bins_idx = [20, 60, 120]
  # smoothing_window = PU.get_smoothing_window(; smoothing_window=2, smoothing_variance=FloatType(1.2f0))
  t1_smoothing_window = PU.get_smoothing_window(; smoothing_window=2, smoothing_variance=FloatType(1.2f0))

  resume_from = "./outputs/search_state_latest.jld2" #nothing             # set to e.g. "./outputs/search_state_latest.jld2" to resume
  force_restart_from_random = false # set to true to restart from random params when resuming
  parametrize(;
    cohorts_db_path="../data_eco_cohorts.duckdb",
    filter_eco_field="epa_l3",
    eco_field="epa_l3",
    tablename="data_eco_cohorts",
    output_dir="./outputs",
    filter_ecos=filter_ecos,
    filter_plots=filter_plots,
    filter_species=filter_species,
    skip_disturbances=true,
    spinup=false,
    search_mode="lbsa",
    tier=1,
    bins_idx=t1_bins_idx,
    smoothing_window=t1_smoothing_window,
    TRIALS=1000000,
    resume_from=resume_from,
    force_restart_from_random=force_restart_from_random,
    rng=rng)
end

function julia_main()::Cint
  main()
  return 0
end

# ---------------------------------------------------------------------------
# Standalone plot helpers
# ---------------------------------------------------------------------------

function _load_params_from_path(params_path::String)
  raw = JLD2.load_object(params_path)
  if raw isa LBSA.LBSAState
    @info "Loaded LBSAState — using best params" loss=convert(Float64, raw.best.fx) iter=raw.best_iteration
    return raw.best.x
  elseif raw isa Vector  # sobol results
    @info "Loaded sobol results — using rank-1 params" mean_loss=raw[1].mean_loss
    return raw[1].params
  else
    @info "Loaded params directly"
    return raw
  end
end

struct PlotContext
  splots::DataFrame
  species_list::Vector{String}
  eco_species_ids::Vector{Vector{Int}}
  site_sim_years
  spinup_cohorts::DataFrame
  spdf_plts
  loss_params::PU.LossParams
  max_sim_year::Int
  no_establishment::Bool
  injection_cohorts::Union{Nothing,DataFrame}
end

# Shared data-loading boilerplate used by both plot functions.
function _load_plot_context(; cohorts_db_path, eco_field, tablename, output_dir,
                               skip_disturbances, spinup, by_subplot=false, no_establishment=false, filter_eco_field,
                               filter_ecos, filter_plots, filter_species,
                               bins_idx, smoothing_window_size, smoothing_variance, rng)
  smoothing_window = smoothing_window_size > 0 ?
    PU.get_smoothing_window(; smoothing_window=smoothing_window_size,
                              smoothing_variance=FloatType(smoothing_variance)) :
    FloatType[one(FloatType)]
  loss_params = PU.LossParams(
    age_bins          = PU.AgeBins(bins_idx=bins_idx .|> Int, last_bin_open=true),
    smoothing_weights = smoothing_window,
  )
  splots, _, species_list, eco_species_ids, _ =
    Data.prepare_parametrization_data(;
      cohorts_db_path, eco_field, tablename, output_dir,
      skip_disturbances, spinup, by_subplot, filter_eco_field,
      filter_ecos, filter_plots, filter_species, RNG=rng)
  splots.sim_year .= Dates.value.(Dates.Day.(splots.measdate .- splots.start_measdate)) ./ 365.25 .|> round .|> Int
  max_sim_year   = maximum(splots.sim_year)
  max_age        = Int(maximum(splots.age_calc))
  site_sim_years = Data.get_site_sim_years(splots)
  spinup_cohorts = Data.get_spinup_cohorts(splots)
  spdf           = PU.smoothen_ref_years(splots, loss_params, max_age; debug=false)
  spdf_plts      = Data.make_spdf_dict(spdf, eco_species_ids)
  PlotContext(splots, species_list, eco_species_ids, site_sim_years,
              spinup_cohorts, spdf_plts, loss_params, max_sim_year,
              no_establishment,
              no_establishment ? Data.get_injection_cohorts(splots) : nothing)
end

function _run_and_plot(bio_params, label, ctx::PlotContext;
                       sampled_ids, emp_sample, output_dir, spinup, rng)
  ref_soa = make_sites(ctx.splots, ctx.eco_species_ids; rng, spinup, no_establishment=ctx.no_establishment)
  inj_dict  = isnothing(ctx.injection_cohorts) ? nothing : _build_injection_dict(ctx.injection_cohorts, ref_soa)
  inj_years = isnothing(ctx.injection_cohorts) ? Set{Int}() : Set(Int.(ctx.injection_cohorts.sim_year))
  result  = only(fit_params(ref_soa, bio_params, ctx.max_sim_year,
                            length(ctx.species_list), ctx.eco_species_ids,
                            ctx.spdf_plts, ctx.site_sim_years, spinup,
                            ctx.spinup_cohorts, ctx.loss_params;
                            debug=false, search_tier=3, seeds=[rand(rng, UInt64)],
                            injection_dict=inj_dict, injection_years=inj_years))
  loss       = convert(Float64, PU.get_total_loss(result[1]))
  sim_sample = _filter_cached_to_df(result[2], sampled_ids)
  generate_plots(emp_sample, sim_sample, label, loss, output_dir)
  @info "Plots saved [$label]" loss n_plots=length(sampled_ids)
end

"""
plot_sample — run one simulation with given params and plot n_output_plots forest plots.
Accepts best_params@X.jld2, search_state@X.jld2, or sobol_results@X.jld2.
"""
function plot_sample(;
  cohorts_db_path::String,
  params_path::String,
  output_dir::String                   = "./outputs",
  filter_eco_field::String             = "epa_l3",
  eco_field::String                    = "epa_l3",
  tablename::String                    = "curated_cohorts_landis",
  filter_ecos::Vector{String}          = String[],
  filter_plots::Vector{NTuple{4,Int}}  = NTuple{4,Int}[],
  filter_species::Vector{String}       = String[],
  skip_disturbances::Bool              = true,
  spinup::Bool                         = false,
  by_subplot::Bool                     = false,
  no_establishment::Bool               = false,
  bins_idx::Vector{Int64}              = vcat(10:10:40, 60:20:120) .|> Int64,
  smoothing_window_size::Int           = 0,
  smoothing_variance::Float64          = 1.2,
  n_output_plots::Int                  = 20,
  rng_seed::Int                        = 1337,
)
  mkpath(output_dir)
  rng        = RNGType(UInt64(rng_seed))
  bio_params = _load_params_from_path(params_path)
  ctx = _load_plot_context(; cohorts_db_path, eco_field, tablename, output_dir,
                              skip_disturbances, spinup, by_subplot, no_establishment, filter_eco_field,
                              filter_ecos, filter_plots, filter_species,
                              bins_idx, smoothing_window_size, smoothing_variance, rng)
  all_ids     = UIntType.(unique(ctx.splots.plot_id))
  sampled_ids = _sample_plot_ids(all_ids, n_output_plots, rng; injection_cohorts=ctx.injection_cohorts)
  emp_sample  = _make_emp_df(ctx.splots, sampled_ids)
  _run_and_plot(bio_params, "plot_only", ctx; sampled_ids, emp_sample, output_dir, spinup, rng)
end

"""
plot_sample_sobol — load saved sobol results and plot a stratified sample of param sets.
Selection: 1 absolute top + 60% of remaining from top half + rest from bottom half.
All param sets are visualised on the same fixed set of n_output_plots forest plots.
"""
function plot_sample_sobol(;
  sobol_candidates_db::String,
  cohorts_db_path::String,
  output_dir::String                   = "./outputs",
  filter_eco_field::String             = "epa_l3",
  eco_field::String                    = "epa_l3",
  tablename::String                    = "curated_cohorts_landis",
  filter_ecos::Vector{String}          = String[],
  filter_plots::Vector{NTuple{4,Int}}  = NTuple{4,Int}[],
  filter_species::Vector{String}       = String[],
  skip_disturbances::Bool              = true,
  spinup::Bool                         = false,
  by_subplot::Bool                     = false,
  no_establishment::Bool               = false,
  bins_idx::Vector{Int64}              = vcat(10:10:40, 60:20:120) .|> Int64,
  smoothing_window_size::Int           = 0,
  smoothing_variance::Float64          = 1.2,
  n_output_plots::Int                  = 10,              # fixed forest-plot sample size
  n_sobol_params_to_plot::Int          = 5,               # sobol param sets to simulate
  rng_seed::Int                        = 1337,
)
  mkpath(output_dir)
  rng = RNGType(UInt64(rng_seed))

  db  = DuckDB.DB(sobol_candidates_db)
  con = DuckDB.connect(db)
  raw = DuckDB.execute(con, "SELECT mean_loss, params_blob FROM sobol_results ORDER BY mean_loss ASC") |> DataFrame
  close(db)
  isempty(raw) && (@warn "No sobol results found in $sobol_candidates_db"; return)
  results = [(params=Serialization.deserialize(IOBuffer(row.params_blob)), mean_loss=row.mean_loss)
             for row in eachrow(raw)]
  N = length(results)
  @info "Loaded $N sobol results from $sobol_candidates_db"

  # Stratified param-set selection
  n_extra   = n_sobol_params_to_plot - 1
  n_top     = round(Int, 0.6 * n_extra)
  n_bot     = n_extra - n_top
  top_half  = results[2:max(2, N÷2)]
  bot_half  = results[max(2,N÷2)+1:end]
  selected  = vcat(
    [results[1]],
    Random.shuffle(rng, top_half)[1:min(n_top, length(top_half))],
    Random.shuffle(rng, bot_half)[1:min(n_bot, length(bot_half))],
  )
  @info "plot_sample_sobol: plotting $(length(selected)) param sets from $N sobol results"

  ctx = _load_plot_context(; cohorts_db_path, eco_field, tablename, output_dir,
                              skip_disturbances, spinup, by_subplot, no_establishment, filter_eco_field,
                              filter_ecos, filter_plots, filter_species,
                              bins_idx, smoothing_window_size, smoothing_variance, rng)
  all_ids     = UIntType.(unique(ctx.splots.plot_id))
  sampled_ids = _sample_plot_ids(all_ids, n_output_plots, rng; injection_cohorts=ctx.injection_cohorts)
  emp_sample  = _make_emp_df(ctx.splots, sampled_ids)

  for (rank, res) in enumerate(selected)
    label = rank == 1 ? "sobol_rank1_loss$(round(res.mean_loss,digits=4))" :
                        "sobol_rank$(rank)_loss$(round(res.mean_loss,digits=4))"
    _run_and_plot(res.params, label, ctx; sampled_ids, emp_sample, output_dir, spinup, rng)
  end
end

# ---------------------------------------------------------------------------
# Statistical testing: simulated vs reference plot distributions
# ---------------------------------------------------------------------------


function simulate_and_test(;
  splots::DataFrame,
  bio_params,
  eco_list::Vector{String},
  species_list::Vector{String},
  eco_species_ids::Vector{Vector{Int}},
  loss_params::PU.LossParams,
  site_sim_years,
  M::Int=10,
  no_establishment::Bool=false,
  rng::Random.AbstractRNG,
)::DataFrame
  n_ecos = length(eco_list)
  n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
  max_sim_year = maximum(maximum.(filter(!isempty, site_sim_years.sim_years)))
  eco_params = BiomassSuccessionPlugin.generate_eco_params(bio_params)
  ctx = (BiomassSuccession=(eco_params=eco_params,),)

  ref_soa = make_sites(splots, eco_species_ids; rng=rng, spinup=false, no_establishment=no_establishment)

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
  sim_sum = Dict{NTuple{4,Int}, Float64}()
  sim_cnt = Dict{NTuple{4,Int}, Int}()

  for seed in (rand(rng, UInt64) for _ in 1:M)
    soa = copy_and_reseed_soa(ref_soa, seed)
    for current_sim_year in 1:max_sim_year
      PanCore.process_plugin!(soa, BiomassSuccessionPlugin.BiomassSuccession, current_sim_year; ctx=ctx.BiomassSuccession)
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

  # Mann-Whitney U per (eco, species, bin).
  bin_label(b) = b <= length(loss_params.age_bins.bins_idx) ?
    "<$(loss_params.age_bins.bins_idx[b])" : ">=$(loss_params.age_bins.bins_idx[end])"

  rows = NamedTuple[]
  for eco_id in 1:n_ecos
    for sp_eco in eachindex(eco_species_ids[eco_id])
      gsp = eco_species_ids[eco_id][sp_eco]
      for b in 1:n_bins
        ref = ref_vals[eco_id][sp_eco][b]
        sim = sim_vals[eco_id][sp_eco][b]
        (isempty(ref) || isempty(sim)) && continue
        test = HypothesisTests.MannWhitneyUTest(Float64.(ref), Float64.(sim))
        push!(rows, (
          eco=eco_list[eco_id],
          species=species_list[gsp],
          bin=b,
          age_class=bin_label(b),
          n_ref=length(ref),
          n_sim=length(sim),
          U_statistic=test.U,
          p_value=HypothesisTests.pvalue(test),
        ))
      end
    end
  end

  return isempty(rows) ? DataFrame() : DataFrame(rows)
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

  @info "Pre-SoA RSS: $(round(Sys.maxrss()/1e9, digits=2)) GB"
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
      cn_raster, eco_raster_data, treemap_db_path, eco_mapping_path, params)
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

  println("Running simulation: $timehorizon_years years, output every $output_every_years")
  let ctx = (eco_params=eco_params,),
    bufs = [BiomassSuccessionPlugin._new_buf() for _ in 1:Threads.maxthreadid()],
    chunks = zeros(Int, Threads.maxthreadid())

    Spatial.run_spatial!(ref_soa, BiomassSuccessionPlugin.BiomassSuccession, ctx, output_dir,
      (soa, year) -> BiomassSuccessionPlugin.emit_year!(bufs, chunks, soa, year, output_dir);
      timehorizon=timehorizon_years, output_every=output_every_years)
  end

  println("Generating output rasters")
  ref_raster_path = !isnothing(treemap_raster) ?
                    joinpath(data_dir, treemap_raster) :
                    joinpath(data_dir, eco_raster)
  BiomassSuccessionPlugin.generate_rasters_from_output(; output_dir=output_dir, ref_raster_path=ref_raster_path)

  println("Coalescing Arrow files to DuckDB")
  BiomassSuccessionPlugin.coalesce_to_duckdb(; output_dir=output_dir, db_path=joinpath(output_dir, "cohorts.duckdb"))
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

  println("Running simulation: $timehorizon_years years, output every $output_every_years")
  let ctx = (eco_params=eco_params,),
    bufs = [BiomassSuccessionPlugin._new_buf() for _ in 1:Threads.maxthreadid()],
    chunks = zeros(Int, Threads.maxthreadid())

    Spatial.run_spatial!(ref_soa, BiomassSuccessionPlugin.BiomassSuccession, ctx, output_dir,
      (soa, year) -> BiomassSuccessionPlugin.emit_year!(bufs, chunks, soa, year, output_dir);
      timehorizon=timehorizon_years, output_every=output_every_years)
  end

  println("Generating output rasters")
  BiomassSuccessionPlugin.generate_rasters_from_output(; output_dir=output_dir, ref_raster_path=ecoregion_tif)

  println("Coalescing Arrow files to DuckDB")
  BiomassSuccessionPlugin.coalesce_to_duckdb(; output_dir=output_dir, db_path=joinpath(output_dir, "cohorts.duckdb"))
end

function export_landis_main(; output_dir::String="/workspace/best_params_landis")
  jld2_files = filter(f -> startswith(f, "best_params@") && endswith(f, ".jld2"),
    readdir("./outputs"))
  isempty(jld2_files) && error("No best_params JLD2 files found in ./outputs/")
  latest = last(sort(jld2_files))
  jld2_path = joinpath("./outputs", latest)
  println("Exporting from $jld2_path")
  params = JLD2.load_object(jld2_path)
  BiomassSuccessionPlugin.export_landis_params(params; output_dir=output_dir)
end

# ---------------------------------------------------------------------------
# Export LANDIS-II scenario files (same loading path as spatial simulation)
# ---------------------------------------------------------------------------

# Copy a biomass-climate config file and any data files it references (ClimateFile /
# SpinUpClimateFile lines) into output_dir. Returns the basename for use in
# biomass_succession.txt. Does not read CSV content — just copies bytes.
function _copy_climate_files(src_path::String, output_dir::String)::String
  src_abs = abspath(src_path)
  src_dir = dirname(src_abs)
  dst_name = basename(src_abs)
  cp(src_abs, joinpath(output_dir, dst_name); force=true)
  for line in eachline(src_abs)
    m = match(r"^\s*(?:ClimateFile|SpinUpClimateFile)\s+(\S+)", line)
    isnothing(m) && continue
    ref = m.captures[1]
    ref_src = isabspath(ref) ? ref : joinpath(src_dir, ref)
    isfile(ref_src) || continue
    ref_dst = joinpath(output_dir, basename(ref))
    isfile(ref_dst) || cp(ref_src, ref_dst)
    println("  $(basename(ref))")
  end
  return dst_name
end

function export_landis_scenario(;
  data_dir::String,
  output_dir::String,
  eco_raster::String,
  eco_ecocode_mapping::String,
  biomass_params_path::String,
  climate_config_file::String,
  treemap_raster::Union{String,Nothing}=nothing,
  communities_csv::Union{String,Nothing}=nothing,
  communities_db::Union{String,Nothing}=nothing,
  treemap_version::Int=2022,
  treemap_db_path::String="../data_eco_cohorts.duckdb",
  duration_years::Int=40,
  cell_length_m::Int=30,
  timestep::Int=5,
  # Ignored parameters (for signature compatibility with simulate_spatial_treemap)
  rng_seed::Int=1337,
  timehorizon_years::Int=50,
  output_every_years::Int=5,
)
  mkpath(output_dir)

  println("Loading params: $biomass_params_path")
  params = JLD2.load_object(joinpath(data_dir, biomass_params_path))

  println("Loading eco raster")
  eco_raster_path = joinpath(data_dir, eco_raster)
  eco_raster_data = Data.load_eco_raster(eco_raster_path)

  eco_mapping_path = joinpath(data_dir, eco_ecocode_mapping)
  eco_mapping_df = CSV.read(eco_mapping_path, DataFrame)

  local communities_df, combo_to_mapcode, mod_params, eco_species_ids, cn_raster

  if !isnothing(treemap_raster)
    println("Loading treemap raster: $treemap_raster")
    @time cn_raster, _ = Data.load_treemap_raster(
      joinpath(data_dir, treemap_raster); treemap_version=treemap_version)
    @assert size(cn_raster) == size(eco_raster_data) "Raster size mismatch: treemap $(size(cn_raster)) ≠ eco $(size(eco_raster_data))"

    println("Extracting cohorts from DuckDB (treemap path)")
    @time splots, eco_list, eff_eco_list, species_list = Data.load_treemap_cohorts(
      cn_raster, eco_raster_data, treemap_db_path, eco_mapping_path, params)
    println("Plots: $(length(unique(splots.plt_cn))), Ecos: $(length(eco_list)), Species: $(length(species_list))")

    println("Remapping params to data eco/species")
    @time mod_params, mapped_splots, eco_species_ids = Data.map_params_to_data_treemap(
      params, eco_list, eff_eco_list, species_list, splots)

    println("Deduplicating cohorts by (plt_cn, ecocode)")
    @time communities_df, combo_to_mapcode = Data.deduplicate_for_export(
      mapped_splots, cn_raster, eco_raster_data)

  elseif !isnothing(communities_csv) || !isnothing(communities_db)
    if !isnothing(communities_csv)
      println("Loading initial communities from CSV: $communities_csv")
      ic_df = Data.load_csv_communities(joinpath(data_dir, communities_csv))
    else
      println("Loading initial communities from DuckDB: $communities_db")
      ic_df = Data.load_duckdb_communities(communities_db)
    end
    # prepare_general_splots maps to internal IDs; for export we use ic_df directly.
    # We still call it to get mod_params and eco_species_ids.
    _, mod_params, eco_species_ids = Data.prepare_general_splots(
      ic_df, eco_raster_data, eco_mapping_df, params)
    cn_raster = nothing
    combo_to_mapcode = nothing

    # ic_df already has semantic columns (mapcode, species, age/CohortAge, biomass/CohortBiomass).
    # Normalize column names and filter to species present in mod_params.
    sp_set = Set(string.(mod_params.SPECIES_LIST))
    sp_col = hasproperty(ic_df, :SpeciesName) ? ic_df.SpeciesName : ic_df.species
    age_col = hasproperty(ic_df, :CohortAge) ? ic_df.CohortAge : ic_df.age
    bio_col = hasproperty(ic_df, :CohortBiomass) ? ic_df.CohortBiomass : ic_df.biomass
    mc_col = hasproperty(ic_df, :MapCode) ? ic_df.MapCode : ic_df.mapcode
    valid = [s in sp_set for s in sp_col]
    communities_df = DataFrame(
      mapcode=Int.(mc_col[valid]),
      species=String.(sp_col[valid]),
      age_calc=FloatType.(age_col[valid]),
      agb_sum=FloatType.(bio_col[valid]),
    )

  else
    error("Specify one of: treemap_raster, communities_csv, communities_db")
  end

  n_sites = length(unique(communities_df.mapcode))
  println("Prepared $n_sites unique mapcodes for export")

  # ---------------------------------------------------------------------------
  println("\nExporting LANDIS-II scenario files to $output_dir ...")

  # 1. scenario.txt
  println("Exporting scenario.txt")
  BiomassSuccessionPlugin.export_scenario_file(;
    output_path=joinpath(output_dir, "scenario.txt"),
    duration_years=duration_years,
    cell_length_m=cell_length_m,
    rng_seed=rng_seed,
  )

  # 2. ecoregion.txt + copy ecoregion.tif (resolve symlinks for a real file copy)
  println("Exporting ecoregion.txt")
  BiomassSuccessionPlugin.export_ecoregions_txt(
    mod_params, eco_mapping_df;
    output_path=joinpath(output_dir, "ecoregion.txt"))
  println("Copying ecoregion.tif")
  cp(realpath(eco_raster_path), joinpath(output_dir, "ecoregion.tif"); force=true)
  println("  ecoregion.tif")

  # 3. Climate config file + any data files it references
  climate_ref = nothing  # filename to embed in biomass_succession.txt
  if !isnothing(climate_config_file)
    println("Copying climate files")
    climate_ref = _copy_climate_files(climate_config_file, output_dir)
    println("  $climate_ref")
  end

  # 4. CoreSpeciesData.txt, SpeciesData.csv, SppEcoregionData.csv, biomass_succession.txt
  BiomassSuccessionPlugin.export_landis_params(mod_params;
    output_dir=output_dir,
    climate_config_file=climate_ref
  )

  # 4. initial_communities.csv
  println("Exporting initial_communities.csv")
  BiomassSuccessionPlugin.export_initial_communities_csv(
    communities_df;
    output_path=joinpath(output_dir, "initial_communities.csv"))

  # 5. initial_communities.tif (treemap path only; mapcode raster)
  if !isnothing(combo_to_mapcode)
    println("Exporting initial_communities.tif")
    ref_raster = joinpath(data_dir, treemap_raster)
    BiomassSuccessionPlugin.export_initial_communities_tif(
      combo_to_mapcode, cn_raster, eco_raster_data, ref_raster;
      output_path=joinpath(output_dir, "initial_communities.tif"))
  end

  # 6. eco_ecocode_mapping.csv (lookup table, not read by LANDIS directly)
  BiomassSuccessionPlugin.export_eco_ecocode_mapping(
    mod_params, eco_mapping_df;
    output_path=joinpath(output_dir, "eco_ecocode_mapping.csv"))
  println("  eco_ecocode_mapping.csv")

  println("\nLANDIS-II scenario exported to: $output_dir")
  println("Files:")
  for f in sort(readdir(output_dir))
    println("  $f")
  end
end

function export_landis_scenario_main()
  export_landis_scenario(
    data_dir="../",
    output_dir="./outputs/landis_scenario",
    eco_raster="eco_raster.tif",
    eco_ecocode_mapping="eco_ecocode_mapping.csv",
    biomass_params_path="landis_parametrization_julia/outputs/best_params.jld2",
    treemap_raster="treemap.tif",
    treemap_version=2022,
    treemap_db_path="../data_eco_cohorts.duckdb",
    climate_config_file="biomass-climate.txt",
  )
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
