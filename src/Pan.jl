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
using .Search: SA, LBSA, MOLBSA, CMAES, MOCMAES, IgelMOCMAES, CMAMAE
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


# Each cohort tuple = (eco_species_id, age, observed_biomass, disturbance_drop_pct).
const _SiteInjectionYear = Vector{Tuple{Int,Vector{Tuple{UIntType,FloatType,FloatType,FloatType}}}}
const _SiteInjectionDict = Dict{Int,_SiteInjectionYear}

# Diagnostic toggle. When true, the injection path is switched to OVERRIDE mode:
# get_injection_cohorts returns the full observed state at every measurement year,
# and _inject_observed_cohorts! forces each observed cohort onto the site — overwriting
# the biomass of the matching (species, age) cohort in place, or adding it if absent —
# so sim matches obs at every loss point. Flip from the REPL: `Pan.OVERRIDE_INJECTION[] = false`.
# Mirrors the BiomassSuccessionPlugin.CALIBRATE[] pattern. For the sensitivity test, set
# OVERRIDE_INJECTION_NOISE[] > 0 (multiplicative jitter sd on injected biomass).
# OVERRIDE_INJECTION_REPLACE[] = true switches override to FULL REPLACE: each measured
# site is wiped (live/old/B reset) and rebuilt from exactly the observed cohorts, dropping
# sim-only cohorts (ones that should have died but didn't). Gives an exact multi-cohort
# state for stress-testing the CSR indexing. Requires OVERRIDE_INJECTION[] (for the data).
const OVERRIDE_INJECTION = Ref(false)
const OVERRIDE_INJECTION_NOISE = Ref(0.0)
const OVERRIDE_INJECTION_REPLACE = Ref(false)
# OVERRIDE_INJECTION_SYNC[] = true: SET-SYNC mode (choice B) — the injection manages the cohort
# SET to track the data (remove sim cohorts whose (species,age) isn't observed = "died"; add
# observed cohorts the sim lacks = "recruited", with observed biomass) but LEAVES matched
# survivors' sim biomass untouched, so the simulator fits their biomass deterministically.
# Requires OVERRIDE_INJECTION[]; mutually exclusive with REPLACE (REPLACE takes precedence).
const OVERRIDE_INJECTION_SYNC = Ref(false)

# Partial-disturbance handling in SYNC mode. Each observed cohort carries a disturbance_drop_pct
# (from curation). Modes:
#   :off              — ignore drop (current behaviour).
#   :scale            — scale the matched (survivor) sim cohort's biomass by (1-drop) at the
#                       disturbance year (one-time shock); the cohort STAYS in the loss, so growth
#                       params fit the undisturbed trajectory with the disturbance applied.
#   :exclude_overwrite— overwrite the matched cohort with the observed (already-reduced) biomass and
#                       EXCLUDE it from the site loss at that year.
#   :exclude_noscale  — leave the sim biomass and EXCLUDE the cohort from the site loss at that year.
const OVERRIDE_INJECTION_DISTURBANCE = Ref(:off)

# Only touches the sites that actually have injections (typically << n_sites).
# _new_cohort_counts is already == site.live for all other sites after process_plugin!.
# override=false (default): append every listed cohort as a new cohort — unchanged.
# override=true: for each observed cohort, overwrite the biomass of the matching
#   (species, age) cohort in place; add it only if no match exists. Sim cohorts with
#   no observed counterpart are left untouched; capacity grows only by the unmatched count.
function _inject_observed_cohorts!(soa, site_cohorts::_SiteInjectionYear; override::Bool=false, replace::Bool=false, sync::Bool=false)
  if !override
    for (site_idx, cohorts) in site_cohorts
      site = getsite(soa, site_idx)
      site.active || continue
      site._new_cohort_counts = Int(site.live) + length(cohorts)
    end
    PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))
    for (site_idx, cohorts) in site_cohorts
      site = getsite(soa, site_idx)
      site.active || continue
      for (sp, age, bio, _drop) in cohorts
        BiomassSuccessionPlugin.add_cohort!(site, sp, age, bio)
        site.B += bio
      end
    end
    return
  end

  if replace
    # Full replace: wipe each measured site and rebuild it from exactly the observed
    # cohorts (raw biomass, no trunc; optional noise). Drops sim-only cohorts so the
    # site state equals the data even when some sim cohorts should have died but didn't.
    for (site_idx, cohorts) in site_cohorts
      site = getsite(soa, site_idx)
      site.active || continue
      site._new_cohort_counts = length(cohorts)
    end
    PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))
    noise = FloatType(OVERRIDE_INJECTION_NOISE[])
    for (site_idx, cohorts) in site_cohorts
      site = getsite(soa, site_idx)
      site.active || continue
      site.live = zero(UIntType)
      site.old = zero(UIntType)
      site.B = zero(FloatType)
      cbio = site.c_bio
      for (sp, age, bio, _drop) in cohorts
        b = noise > zero(FloatType) ? abs(bio * (one(FloatType) + noise * randn(site.rng, FloatType))) : bio
        BiomassSuccessionPlugin.add_cohort!(site, sp, age, b)
        cbio[Int(site.live)] = b   # overwrite add_cohort!'s trunc with the raw value
        site.B += b
      end
    end
    return
  end

  if sync
    # Set-sync (choice B): cohort SET tracks the data; matched survivors keep their sim biomass.
    # Pass A: remove sim cohorts whose (species,age) isn't in the observed set ("died"), in place.
    for (site_idx, cohorts) in site_cohorts
      site = getsite(soa, site_idx)
      site.active || continue
      keep = Set{Tuple{UIntType,FloatType}}((sp, age) for (sp, age, _, _) in cohorts)
      csp = site.c_species
      cage = site.c_age
      cbio = site.c_bio
      i = 1
      while i <= Int(site.live)
        if (csp[i], cage[i]) in keep
          i += 1
        else
          site.B -= cbio[i]
          lastidx = Int(site.live)
          if i != lastidx
            csp[i] = csp[lastidx]
            cage[i] = cage[lastidx]
            cbio[i] = cbio[lastidx]
          end
          site.live -= one(UIntType)
        end
      end
    end
    # Disturbance handling for matched (survivor) cohorts at the disturbance year. Only survivors
    # (present after Pass A) are touched; recruits added in Pass C keep their observed biomass.
    #   :scale            — biomass *= (1-drop): a known shock; cohort STAYS in the loss (growth fits it).
    #   :exclude_overwrite— biomass := observed survivor biomass; cohort EXCLUDED from loss (loss side).
    #   :exclude_noscale  — biomass unchanged; cohort EXCLUDED from loss (loss side).
    let dmode = OVERRIDE_INJECTION_DISTURBANCE[]
      if dmode === :scale || dmode === :exclude_overwrite
        for (site_idx, cohorts) in site_cohorts
          site = getsite(soa, site_idx)
          site.active || continue
          csp = site.c_species; cage = site.c_age; cbio = site.c_bio
          for (sp, age, bio, drop) in cohorts
            drop > zero(FloatType) || continue
            @inbounds for j in 1:Int(site.live)
              if csp[j] == sp && cage[j] == age
                newb = dmode === :scale ? cbio[j] * (one(FloatType) - drop) : bio
                site.B += newb - cbio[j]
                cbio[j] = newb
                break
              end
            end
          end
        end
      end
    end
    # Pass B: size each site to survivors + (# observed (species,age) absent from the sim).
    for (site_idx, cohorts) in site_cohorts
      site = getsite(soa, site_idx)
      site.active || continue
      csp = site.c_species
      cage = site.c_age
      live = Int(site.live)
      n_new = 0
      for (sp, age, _, _) in cohorts
        matched = false
        for j in 1:live
          if csp[j] == sp && cage[j] == age
            matched = true
            break
          end
        end
        matched || (n_new += 1)
      end
      site._new_cohort_counts = live + n_new
    end
    PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))
    # Pass C: append the unmatched observed cohorts (recruits) with observed biomass; recompute old.
    noise = FloatType(OVERRIDE_INJECTION_NOISE[])
    for (site_idx, cohorts) in site_cohorts
      site = getsite(soa, site_idx)
      site.active || continue
      csp = site.c_species
      cage = site.c_age
      cbio = site.c_bio
      orig_live = Int(site.live)
      for (sp, age, bio, _drop) in cohorts
        matched = false
        for j in 1:orig_live
          if csp[j] == sp && cage[j] == age
            matched = true
            break
          end
        end
        if !matched
          b = noise > zero(FloatType) ? abs(bio * (one(FloatType) + noise * randn(site.rng, FloatType))) : bio
          BiomassSuccessionPlugin.add_cohort!(site, sp, age, b)
          cbio[Int(site.live)] = b
          site.B += b
        end
      end
      # Pass A removals + Pass C adds left site.old stale — recompute (# cohorts with age > 1).
      o = zero(UIntType)
      for j in 1:Int(site.live)
        cage[j] > one(FloatType) && (o += one(UIntType))
      end
      site.old = o
    end
    return
  end

  # Pass 1: size each site to live + (# observed cohorts with no (species,age) match).
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    csp = site.c_species
    cage = site.c_age
    live = Int(site.live)
    n_new = 0
    for (sp, age, _) in cohorts
      matched = false
      for j in 1:live
        if csp[j] == sp && cage[j] == age
          matched = true
          break
        end
      end
      matched || (n_new += 1)
    end
    site._new_cohort_counts = live + n_new
  end
  PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))

  # Pass 2: overwrite matches in place (searching only the original live range),
  # append the unmatched. Biomass is written RAW (no trunc) so the injected state
  # equals the reference exactly — only this diagnostic path skips the integer-valued
  # truncation that add_cohort!/succession use everywhere else (C# parity).
  noise = FloatType(OVERRIDE_INJECTION_NOISE[])
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    csp = site.c_species
    cage = site.c_age
    cbio = site.c_bio
    orig_live = Int(site.live)
    for (sp, age, bio) in cohorts
      b = noise > zero(FloatType) ? abs(bio * (one(FloatType) + noise * randn(site.rng, FloatType))) : bio
      matched = false
      for j in 1:orig_live
        if csp[j] == sp && cage[j] == age
          site.B += b - cbio[j]
          cbio[j] = b
          matched = true
          break
        end
      end
      if !matched
        BiomassSuccessionPlugin.add_cohort!(site, sp, age, b)
        cbio[Int(site.live)] = b   # overwrite add_cohort!'s trunc with the raw value
        site.B += b
      end
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
      push!(year_list, (site_idx, Tuple{UIntType,FloatType,FloatType,FloatType}[]))
      pos = length(year_list)
      site_year_pos[(site_idx, year)] = pos
    end
    drop = hasproperty(injection_cohorts, :disturbance_drop_pct) ? FloatType(row.disturbance_drop_pct) : zero(FloatType)
    push!(year_list[pos][2], (UIntType(row.eco_species_id), FloatType(row.age_calc), FloatType(row.agb_sum), drop))
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
  plot_labels = Dict{UIntType,String}()
  if hasproperty(empirical_df, :statecd)
    has_subp = hasproperty(empirical_df, :subp)
    id_cols = has_subp ? [:plot_id, :statecd, :unitcd, :countycd, :plot, :subp] :
              [:plot_id, :statecd, :unitcd, :countycd, :plot]
    for row in eachrow(unique(select(empirical_df, id_cols)))
      plot_labels[UIntType(row.plot_id)] = has_subp ?
                                           "($(row.statecd), $(row.unitcd), $(row.countycd), $(row.plot), subp=$(row.subp))" :
                                           "($(row.statecd), $(row.unitcd), $(row.countycd), $(row.plot))"
    end
  end

  # Build (plot_id, species_id) → effective_species label
  species_labels = Dict{Tuple{UIntType,UIntType},String}()
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

    sp_label = get(species_labels, (plot_id, species_id), "sp$(Int(species_id))")
    id_safe = replace(plot_label, r"[\(\), ]+" => "_")
    file_name = "$(id_safe)$(sp_label)@$(iteration)_$(round(loss,digits=4)).png"
    n_panels = nrow(sim_year_df)
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
              @error "writer: generate_plots (train) failed" iter = state.i exception = (e, catch_backtrace())
            end
          end
          if !isnothing(job.emp_sample_val) && !isnothing(job.sim_sample_val)
            try
              generate_plots(job.emp_sample_val, job.sim_sample_val, "validation_$(state.i)", convert(Float64, state.best.fx), output_dir)
            catch e
              @error "writer: generate_plots (val) failed" iter = state.i exception = (e, catch_backtrace())
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

  delta_bins = Int[]
  delta_agbs = FloatType[]
  delta_species = String[]
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
  palette = CairoMakie.Makie.wong_colors()
  sp_colors = [palette[mod1(i, length(palette))] for i in eachindex(all_species)]

  f = CairoMakie.Figure(size=(1000, 600))
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
  unbinned_w::Bool=false,   # Sim A: compute W as the EXACT per-year EMD (no binning, no smoothing). bins_idx applies to Sim B only.
  w_count_balance::Bool=false,          # count-balance reweight of the W1 term (survivorship under-representation fix)
  w_count_balance_mode::String="both",  # which sim(s) get reweighted: "a" / "b" / "both"
  w_count_beta::Float64=0.99,           # effective-number-of-samples temper (→1 ≈ 1/n, →0 ≈ uniform)
  spinup::Bool=true,
  search_mode::String="lbsa",
  tier::Int=3,
  TRIALS::Int=30000,
  resume_from::Union{Nothing,String}=nothing,
  start_from::Union{Nothing,String}=nothing,
  force_restart_from_random::Bool=false,
  sobol_n::Int=100,
  saltelli::Bool=false,
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
  stratify_eco_mixed::Bool=false,
  single_ecoregion::Bool=false,
  stratify_landuse::Bool=false,
  filter_extent::Union{Nothing,String}=nothing,
  loss_lambda::Float64=1.0,
  loss_alpha::Float64=1.0,
  agb_hinge::Bool=false,
  agb_hinge_threshold::Float64=10.0,
  agb_hinge_pct::Float64=0.0,
  agb_hinge_pct_min::Float64=0.0,
  agb_hinge_pct_max::Float64=Inf,
  agb_hinge_l2::Bool=false,
  agb_hinge_p::Float64=2.0,
  agb_hinge_beta::Float64=1.0,
  agb_normalize::Bool=false,
  w_normalize::Bool=false,
  cell_normalize::Bool=false,
  w_scale_factor::Float64=1.0,   # multiplies W_SCALE_{A,B}; <1 shrinks the W divisor → amplifies ΣW (rebalance vs AGB)
  w_smooth::Bool=false,
  w_smooth_band::Float64=0.05,
  w_smooth_conc::Float64=0.5,
  w_smooth_beta::Float64=1.0,          # fixed β for the W softplus (used when w_smooth_auto_knee=false, the default)
  w_smooth_auto_knee::Bool=false,      # true → derive β_W from band·conc·wmax (reweight-aware per-cell); W only, AGB unaffected
  w_p::Float64=1.0,
  loss_piecewise::Bool=false,
  w_pivot::Float64=1.0,
  agb_pivot::Float64=1.0,
  init_perturb_frac::Float64=0.0,
  init_perturb_cap::Float64=50.0,   # max |absolute biomass deviation| per cohort
  diagnose::Bool=false,
  cycle_years::Real=8,
  cmaes_lambda::Union{Nothing,Int}=nothing,
  cmaes_sigma0::Float64=0.3,
  cmaes_warmstart_seeds::Int=20,   # MO-CMA-ES: # top Sobol seeds to warm-start mean(recombination)+C(covariance shape); 1 ⇒ single-point + C=I (old behavior)
  ipop::Bool=false,
  ipop_stagnation::Int=20,
  cmaes_archive_cap::Int=200,
  seed_archive_from::Union{Nothing,String}=nothing,
  cmaes_integer_handling::Bool=false,
  cmaes_integer_std_factor::Float64=0.3,
  cmaes_single_cov::Bool=false,   # CMA-ES family: one full covariance over ALL params (no eco×lu block split)
  igel_mu::Int=20,
  igel_sigma0::Float64=0.3,
  igel_sobol_init::Bool=true,
  igel_niche_radius::Float64=0.0,
  igel_reseed_sigma::Float64=0.0,
  igel_maturity::Int=0,
  cmame_alpha::Float64=0.02,
  cmame_grid::Int=15,
  cmame_reseed_explore::Float64=1.0,
  cmame_restart_patience::Int=6,
  cmame_sobol_reseed::Bool=false,
  cmame_mo_rank::Bool=false,
  balanced_quality::Bool=false,
  archive_by_sp::Bool=false,
  bounds_by_sobol::Bool=false,
  top_seeds::Float64=0.2,
  dominance_species::String="exact",
  rng::Random.AbstractRNG)

  mkpath(output_dir)
  # [5, 10, 20, 40, 60, 80]
  #bins_idx = vcat(5:5:30, 40:10:80, 100:20:160)
  @info bins_idx
  @info smoothing_window
  PU.LOSS_ALPHA[] = FloatType(loss_alpha)   # 0 → optimize L2/AGB-level only (zero Wasserstein)
  PU.AGB_HINGE[] = agb_hinge                 # AGB term: hinge-L1 (tolerance band) vs sqrt-difference
  PU.AGB_HINGE_THRESHOLD[] = FloatType(agb_hinge_threshold)
  PU.AGB_HINGE_PCT[] = FloatType(agb_hinge_pct)          # >0 ⇒ band = clamp(obs*pct, min, max)
  PU.AGB_HINGE_PCT_MIN[] = FloatType(agb_hinge_pct_min)
  PU.AGB_HINGE_PCT_MAX[] = FloatType(agb_hinge_pct_max)
  PU.AGB_HINGE_L2[] = agb_hinge_l2                       # legacy
  PU.AGB_HINGE_P[] = FloatType(agb_hinge_p)              # hinge exponent pen(h)=h^p (default 2 = L2)
  PU.AGB_HINGE_BETA[] = FloatType(agb_hinge_beta)        # softplus sharpness (constant; 0 ⇒ hard hinge)
  PU.AGB_NORMALIZE[] = agb_normalize             # AGB term = relative error (÷ observed total)
  PU.W_NORMALIZE[] = w_normalize                 # W1 term = ÷ per-reference max → [0,1] ratio
  PU.CELL_NORM[] = cell_normalize                # per-(eco×lu×sp) ÷scale + AGB-rank rescale (overrides global)
  PU.W_SCALE_FACTOR[] = FloatType(w_scale_factor)  # scale the W divisor (rebalance ΣW vs ΣAGB)
  if cell_normalize                              # reset tables so THIS run's train reference (re)populates them
    PU.RANKW[] = zeros(FloatType, 0, 0); PU.CELL_NORM_FREEZE[] = false
  end
  PU.W_SOFTPLUS[] = w_smooth                     # softplus-smooth the per-term W1
  PU.W_SMOOTH_BAND[] = FloatType(w_smooth_band); PU.W_SMOOTH_CONC[] = FloatType(w_smooth_conc)
  PU.W_SMOOTH_BETA[] = FloatType(w_smooth_beta); PU.W_SMOOTH_AUTO_KNEE[] = w_smooth_auto_knee
  PU.W_P[] = FloatType(w_p)                       # square the W term (aggression)
  PU.LOSS_PIECEWISE[] = loss_piecewise; PU.W_PIVOT[] = FloatType(w_pivot); PU.AGB_PIVOT[] = FloatType(agb_pivot)
  INIT_PERTURB_FRAC[] = init_perturb_frac    # >0 ⇒ n_reps initial-biomass perturbations, scored by best
  INIT_PERTURB_CAP[] = FloatType(init_perturb_cap)   # cap on |absolute biomass change| per cohort
  init_perturb_frac > 0 && @info "Initial-condition perturbation: ±$(round(100*init_perturb_frac;digits=2))% over $(n_reps) reps (best-of), shared seed; perturbed = max(2, bio + clamp(bio·pct, ±$(init_perturb_cap))) [max deviation $(init_perturb_cap), floor 2]"
  # unbinned_w (Sim A): per-year age grid (1:400) + no age-smoothing → W₁ is the EXACT per-year EMD.
  # Setting the bins at the SOURCE means W_SOFTPLUS_BETA (_set_loss_scales!) and the cell-norm W_SCALE
  # (_set_cell_scales!) both recompute on this same grid — W₁, its normalization, and the softplus knee
  # stay consistent (no per-year-W₁ ÷ stale-binned-scale mismatch). bins_idx then only governs Sim B (tier-4).
  loss_params = unbinned_w ?
    PU.LossParams(age_bins=PU.AgeBins(bins_idx=collect(1:400), last_bin_open=true),
      smoothing_weights=FloatType[one(FloatType)], lambda=FloatType(loss_lambda)) :
    PU.LossParams(age_bins=PU.AgeBins(bins_idx=bins_idx .|> Int, last_bin_open=true),
      smoothing_weights=smoothing_window, lambda=FloatType(loss_lambda))
  unbinned_w && @info "Sim-A W = UNBINNED per-year EMD (bins 1:400, no age-smoothing); softplus β + cell W_SCALE recompute on this grid"
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
      stratify_eco_mixed=stratify_eco_mixed,
      single_ecoregion=single_ecoregion,
      stratify_landuse=stratify_landuse,
      filter_extent=filter_extent,
      filter_eco_field=filter_eco_field,
      filter_ecos=filter_ecos,
      filter_plots=filter_plots,
      filter_species=filter_species,
      filter_planted=filter_planted,
      RNG=rng)
  n_species = length(species_list)
  n_ecoregions = length(eco_list)
  n_plots = maximum(splots.plot_id)
  # count-balance reweight: build per-agebin W1 weights ONCE from the TRAIN reference (frozen), on the COARSE
  # age_idx bins; Sim A (per-year) looks them up via CBAL_COARSE. Applied in calculate_species_loss!.
  PU.CBAL_ON[] = w_count_balance
  PU.CBAL_MODE[] = Symbol(lowercase(w_count_balance_mode))
  PU.CBAL_BETA[] = w_count_beta
  if w_count_balance
    PU._set_cbal_weights!(splots, PU.AgeBins(bins_idx=bins_idx .|> Int, last_bin_open=true), loss_params.age_bins, eco_species_ids, n_species; beta=w_count_beta)
    @info "Count-balance reweight ON (mode=$(PU.CBAL_MODE[]) β=$(w_count_beta)); coarse=$(length(bins_idx)) bins+open, Sim-A W bins=$(length(loss_params.age_bins.bin_widths))"
  end
  println("Plots:$n_plots, Ecos:$n_ecoregions, Species:$n_species, Measurements: $(size(splots))")
  # Which species drive DOMINANCE (MO Pareto / CMA-MAE measure / breadth axis). Default exact (SPCD only).
  DOMINANCE_GSP[] = lowercase(dominance_species) == "all" ? Set{Int}() :
                    Set{Int}(s for s in 1:n_species if _dom_include_name(species_list[s], lowercase(dominance_species)))
  dominance_species != "all" && @info "Dominance species ($dominance_species): $(length(DOMINANCE_GSP[]))/$n_species → $([species_list[s] for s in sort(collect(DOMINANCE_GSP[]))])"
  println(species_list)

  if diagnose
    Data.print_cycle_coverage(splots; cycle_years=cycle_years, eco_list=eco_list)
    plot_biomass_bin_deltas(splots, loss_params; output_dir=output_dir)
  end

  # Val preprocessing — each half is self-contained with its own contiguous plot_ids
  val_splots = nothing
  val_ref_soa = nothing
  val_spdf_plts = nothing
  val_site_sim_years = nothing
  val_spinup_cohorts = nothing
  val_injection_cohorts = nothing
  if !isnothing(splots_val_raw)
    val_max_age = Int(maximum(splots_val_raw.age_calc))
    val_spdf = PU.smoothen_ref_years(splots_val_raw, loss_params, val_max_age; debug=false)
    val_spdf_plts = Data.make_spdf_dict(val_spdf, eco_species_ids)
    val_site_sim_years = Data.get_site_sim_years(val_spdf)
    val_spinup_cohorts = Data.get_spinup_cohorts(splots_val_raw)   # needed for the val Sim-B spinup (dual val)
    val_injection_cohorts = (no_establishment || OVERRIDE_INJECTION[] || SIMB_DISTURB_ONLY[]) ? Data.get_injection_cohorts(splots_val_raw; all_cohorts=(OVERRIDE_INJECTION[] || SIMB_DISTURB_ONLY[])) : nothing
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



  # SIMB_DISTURB_ONLY builds injection_cohorts purely as the disturbance-data source (all_cohorts=true so the
  # drops cover the whole stand), even with no_establishment=false + override_injection=false — it never injects.
  injection_cohorts = (no_establishment || OVERRIDE_INJECTION[] || SIMB_DISTURB_ONLY[]) ?
    Data.get_injection_cohorts(splots; all_cohorts=(OVERRIDE_INJECTION[] || SIMB_DISTURB_ONLY[])) : nothing
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
      saltelli=saltelli,
      M=n_reps,
      eval_tier=tier,
      n_output_plots=n_output_plots,
      cycle_years=cycle_years)
  end
  driver = search_mode == "molbsa" ? parametrize_MOLBSA :
           search_mode == "cmaes" ? parametrize_CMAES :
           search_mode == "mocmaes" ? parametrize_MOCMAES :
           search_mode == "igelmo" ? parametrize_IgelMOCMAES :
           search_mode == "cmame" ? parametrize_CMAMAE : parametrize_LBSA
  # CMA-ES-only knobs; LBSA/MOLBSA don't accept these, so only splat them for the (MO)CMA-ES drivers.
  cmaes_kw = search_mode == "molbsa" ? (archive_cap=cmaes_archive_cap, seed_archive_from=seed_archive_from) :
             search_mode == "cmaes" ? (cmaes_lambda=cmaes_lambda, cmaes_sigma0=cmaes_sigma0, ipop=ipop, ipop_stagnation=ipop_stagnation, integer_handling=cmaes_integer_handling, integer_std_factor=cmaes_integer_std_factor, single_cov=cmaes_single_cov) :
             search_mode == "mocmaes" ? (cmaes_lambda=cmaes_lambda, cmaes_sigma0=cmaes_sigma0, cmaes_warmstart_seeds=cmaes_warmstart_seeds, ipop=ipop, ipop_stagnation=ipop_stagnation, archive_cap=cmaes_archive_cap, integer_handling=cmaes_integer_handling, integer_std_factor=cmaes_integer_std_factor, single_cov=cmaes_single_cov, seed_archive_from=seed_archive_from) :
             search_mode == "igelmo" ? (archive_cap=cmaes_archive_cap, igel_mu=igel_mu, igel_sigma0=igel_sigma0, igel_sobol_init=igel_sobol_init, igel_niche_radius=igel_niche_radius, igel_reseed_sigma=igel_reseed_sigma, igel_maturity=igel_maturity, single_cov=cmaes_single_cov) :
             search_mode == "cmame" ? (cmaes_lambda=cmaes_lambda, cmaes_sigma0=cmaes_sigma0, cmame_alpha=cmame_alpha, cmame_grid=cmame_grid, cmame_reseed_explore=cmame_reseed_explore, cmame_restart_patience=cmame_restart_patience, cmame_sobol_reseed=cmame_sobol_reseed, balanced_quality=balanced_quality, cmame_mo_rank=cmame_mo_rank, archive_by_sp=archive_by_sp, bounds_by_sobol=bounds_by_sobol, top_seeds=top_seeds, integer_handling=cmaes_integer_handling, integer_std_factor=cmaes_integer_std_factor, single_cov=cmaes_single_cov) :
             NamedTuple()
  driver(; ref_soa=ref_soa,
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
    start_from=start_from,
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
    cycle_years=cycle_years,
    rng=rng,
    debug=debug,
    cmaes_kw...)
end
function accumulate_site_bins!(t1_sim_eco::Matrix{FloatType}, site, loss_params::PU.LossParams, scratch_perm::Vector{Int}, scratch_ages::Vector{FloatType})
  site.live == 0 && return
  nlive = site.live
  c_species = @view site.c_species[1:nlive]
  max_age = Int(ceil(maximum(@view site.c_age[1:nlive]))) + length(loss_params.smoothing_weights) >> 1
  if length(scratch_perm) < nlive
    resize!(scratch_perm, nlive)
  end
  resize!(scratch_ages, max_age)   # exact length so `ages` is a dense Vector — imfilter (via smoothen_bin_cdf) crashes natively on a SubArray view
  p = @view scratch_perm[1:nlive]
  sortperm!(p, c_species)
  ages = scratch_ages
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
      tot_sim = sum(sim_row)
      tot_ref = sum(ref_row)
      let bw = loss_params.age_bins.bin_widths
        s = zero(FloatType)
        acc_sim = zero(FloatType)
        acc_ref = zero(FloatType)
        @inbounds for k in eachindex(bw)
          acc_sim += sim_row[k]
          acc_ref += ref_row[k]
          # W1 on the NORMALIZED biomass-by-age CDF → shape only, invariant to AGB level.
          cdf_sim = tot_sim > 0 ? acc_sim / tot_sim : zero(FloatType)
          cdf_ref = tot_ref > 0 ? acc_ref / tot_ref : zero(FloatType)
          s += bw[k] * (cdf_sim - cdf_ref)^2
        end
        sp_w_loss[gsp] = s
      end
      # AGB LEVEL as a separate, gentle sqrt-difference term (weighted by loss_params.lambda).
      sp_agb_loss[gsp] = loss_params.lambda * (sqrt(tot_sim) - sqrt(tot_ref))^2
    end
    eco_losses[eco_id] = PU.SiteLoss(sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=zero(FloatType), num_sites=eco_site_counts[eco_id], num_obs=eco_obs_counts[eco_id])
  end
  return eco_losses
end


function calculate_t2_loss(t2_sim_bins, t2_sim_total, t2_ref, n_species, eco_species_ids, eco_site_counts, eco_obs_counts, loss_params)
  eco_losses = Vector{PU.SiteLoss}(undef, length(eco_species_ids))
  n_bins = size(t2_ref.bins[1], 2)
  _mean(v) = isempty(v) ? zero(FloatType) : sum(v) / length(v)
  for eco_id in eachindex(eco_species_ids)
    sp_w_loss = zeros(FloatType, n_species)
    sp_agb_loss = zeros(FloatType, n_species)
    sp_map = eco_species_ids[eco_id]
    for sp_eco in eachindex(sp_map)
      gsp = sp_map[sp_eco]
      for b in 1:n_bins
        sp_w_loss[gsp] += PU.wasserstein1d(t2_sim_bins[eco_id][sp_eco, b], t2_ref.bins[eco_id][sp_eco, b])
      end
      # AGB LEVEL as a gentle sqrt-difference of mean totals (weighted by loss_params.lambda).
      sp_agb_loss[gsp] = loss_params.lambda * (sqrt(_mean(t2_sim_total[eco_id][sp_eco])) - sqrt(_mean(t2_ref.total[eco_id][sp_eco])))^2
    end
    eco_losses[eco_id] = PU.SiteLoss(sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=zero(FloatType), num_sites=eco_site_counts[eco_id], num_obs=eco_obs_counts[eco_id])
  end
  return eco_losses
end

# Tier 4: population-level loss. t4_sim/t4_ref are [eco][cycle] matrices of per-bin biomass
# (summed across the plots measured in that calendar cycle). Wasserstein-1 on the biomass-by-age
# CDF is computed per (eco, cycle, species); cycles are summed into one SiteLoss per eco
# (num_obs = total measurements in the eco, so get_total_loss is per-measurement-averaged W1).
const CAPTURE_T4SIM = Ref{Any}(nothing)   # diagnostics: set non-nothing → next calculate_t4_loss stashes (sim,ref,n_cycles)
function calculate_t4_loss(t4_sim, t4_ref, loss_params, n_species, eco_species_ids, t4_site_counts, t4_obs_counts, n_cycles)
  CAPTURE_T4SIM[] === nothing || (CAPTURE_T4SIM[] = (sim=deepcopy(t4_sim), ref=t4_ref, n_cycles=n_cycles, site_counts=deepcopy(t4_site_counts)))
  eco_losses = Vector{PU.SiteLoss}(undef, length(eco_species_ids))
  for eco_id in eachindex(eco_species_ids)
    sp_w_loss = zeros(FloatType, n_species)
    sp_agb_loss = zeros(FloatType, n_species)
    sp_map = eco_species_ids[eco_id]
    bw = loss_params.age_bins.bin_widths
    ncell = 0   # # of (cycle,species) cells with data → tier-4 loss is AVERAGED per cell (dataset-size invariant)
    for cyc in 1:n_cycles
      sim_c = t4_sim[eco_id][cyc]
      ref_c = t4_ref[eco_id][cyc]
      # per-plot-mean: divide the AGB level by the # of distinct plots in this (eco,cycle) so the loss is
      # mean biomass-per-plot-by-age (invariant to plot count). W is CDF-normalized → pcinv cancels there.
      # Uses the SAME T4_CELL_PLOTS the AGB scale divides by, so numerator/denominator stay consistent.
      pcinv = (T4_PER_PLOT_MEAN[] && T4_CELL_PLOTS[] !== nothing) ?
        one(FloatType) / FloatType(max(1, T4_CELL_PLOTS[][eco_id][cyc])) : one(FloatType)
      for sp_eco in eachindex(sp_map)
        gsp = sp_map[sp_eco]
        sim_row = @view sim_c[sp_eco, :]
        ref_row = @view ref_c[sp_eco, :]
        tot_sim = sum(sim_row)
        tot_ref = sum(ref_row)
        (tot_sim > 0 || tot_ref > 0) && (ncell += 1)
        s = zero(FloatType)
        a = zero(FloatType)
        acc_sim = zero(FloatType)
        acc_ref = zero(FloatType)
        # count-balance reweight (Sim B / tier-4): coarse bins ARE the age_idx bins → direct CBAL_W lookup.
        usew = PU.CBAL_ON[] && PU.CBAL_MODE[] !== :a && !isempty(PU.CBAL_W[])
        cbw = usew ? (@inbounds PU.CBAL_W[][gsp, eco_id]) : FloatType[]
        @inbounds for k in eachindex(bw)
          acc_sim += sim_row[k]
          acc_ref += ref_row[k]
          # W1 (L1) on the NORMALIZED biomass-by-age CDF → shape only, invariant to AGB level.
          cdf_sim = tot_sim > 0 ? acc_sim / tot_sim : zero(FloatType)
          cdf_ref = tot_ref > 0 ? acc_ref / tot_ref : zero(FloatType)
          s += (usew ? cbw[k] : one(FloatType)) * bw[k] * abs(cdf_sim - cdf_ref)
          # AGB LEVEL per age bin: ABSOLUTE biomass diff (size-respecting); the global ÷Σobs makes it
          # intensive (numerator & Σobs both scale with #plots → no train/val artifact, no /cnt needed).
          sm = sim_row[k] * pcinv; rm = ref_row[k] * pcinv   # per-plot-mean AGB level (pcinv=1 when off)
          a += PU.AGB_HINGE[] ?
            PU._hinge_relu(abs(sm - rm) - PU._agb_hinge_thresh(rm)) :   # linear excess; power after norm
            (sqrt(sm) - sqrt(rm))^2
        end
        sp_w_loss[gsp] += PU._w_finish(s, Int(gsp), Int(eco_id))  # raw (softplus) W1 when CELL_NORM; else (W1/scale)^p
        sp_agb_loss[gsp] += loss_params.lambda * PU._agb_finish(a)   # raw hinge excess when CELL_NORM; else (excess/scale)^p
      end
    end
    # num_obs = # cells (not measurements): tier-4 loss is averaged PER CELL so it doesn't scale with #plots.
    eco_losses[eco_id] = PU.SiteLoss(sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=zero(FloatType),
      num_sites=max(1, sum(t4_site_counts[eco_id])), num_obs=max(1, ncell))
  end
  return eco_losses
end

# Set the GLOBAL loss-normalization denominators from the reference of THIS evaluation (ratio-of-sums):
# AGB_SCALE = Σ observed AGB; W_SCALE = Σ per-reference-max W1 (Σ bw·max(cdf_ref,1−cdf_ref)). tier-4 reads
# the aggregated t4_ref; all other tiers read the per-measurement spdf records. No-op unless a flag is on.
function _set_loss_scales!(search_tier, spdf_plts, t4_ref, loss_params, eco_species_ids, n_species)
  bw = loss_params.age_bins.bin_widths
  if PU.W_SOFTPLUS[]
    if !PU.W_SMOOTH_AUTO_KNEE[]
      # Knee adjustment OFF (default): fixed β for W, independent of the grid's theoretical max — the
      # band·conc·wmax derivation over-inflates the knee on fine grids (unbinned Σbw≈399) and mutes real W.
      PU.W_SOFTPLUS_BETA[] = PU.W_SMOOTH_BETA[]
      isempty(PU.W_SOFTPLUS_BETA_CELL[]) || (PU.W_SOFTPLUS_BETA_CELL[] = zeros(FloatType, 0, 0))
    else
      # Knee adjustment ON — PER-CELL, bounded by each cell's REFERENCE AGE STRETCH. The theoretical W1 max
      # for a (species,eco) cell is Σ bw only up to K* = the cell's OLDEST reference-cohort bin; past K* the
      # reference CDF is saturated (F_ref=1), so |ΔF| can't realistically be 1 there — summing the whole grid
      # (Σbw−bw[end]=399 unbinned) counts a long empty tail and over-inflates the knee. Reweighted by CBAL_W
      # when the count-balance reweight is on. All frozen across trials → compute once (guarded by size).
      band = PU.W_SMOOTH_BAND[]; conc = PU.W_SMOOTH_CONC[]
      nb = length(bw); ng = n_species; ne = length(eco_species_ids)
      use_cbal = PU.CBAL_ON[] && !isempty(PU.CBAL_W[]) && !isempty(PU.CBAL_COARSE[])
      coarse = use_cbal ? PU.CBAL_COARSE[] : Int[]
      CW = use_cbal ? PU.CBAL_W[] : Matrix{Vector{FloatType}}(undef, 0, 0)
      wmax_full = sum(bw) - bw[end]
      PU.W_SOFTPLUS_BETA[] = wmax_full > 0 ? FloatType(1.0 / (band * conc * wmax_full)) : zero(FloatType)  # scalar fallback
      if size(PU.W_SOFTPLUS_BETA_CELL[]) != (ng, ne)
        # K*[gsp,eco] = oldest reference-cohort bin across the cell's plots (last bin with incremental mass).
        Kstar = zeros(Int, ng, ne)
        for ((_, eid), yd) in spdf_plts, (_, gt) in yd, (sp_eco, rec) in gt.records
          gsp = Int(eco_species_ids[eid][sp_eco]); cdf = rec.sp_age_cdf; kk = 0
          @inbounds for k in 1:nb
            p = cdf[k] - (k == 1 ? zero(FloatType) : cdf[k-1])
            p > FloatType(1e-6) && (kk = k)
          end
          kk > Kstar[gsp, eid] && (Kstar[gsp, eid] = kk)
        end
        βcell = fill(PU.W_SOFTPLUS_BETA[], ng, ne)
        for eco in 1:ne, gsp in 1:ng
          K = Kstar[gsp, eco]; K <= 1 && continue                # no reference / single bin → keep scalar
          w = use_cbal ? (@inbounds CW[gsp, eco]) : FloatType[]
          rew = use_cbal && !isempty(w)
          m = 0.0
          @inbounds for k in 1:K-1; m += (rew ? Float64(w[coarse[k]]) : 1.0) * Float64(bw[k]); end  # Σ bw over the ref stretch (drop saturating bin K*)
          m > 0 && (βcell[gsp, eco] = FloatType(1.0 / (band * conc * m)))
        end
        PU.W_SOFTPLUS_BETA_CELL[] = βcell
      end
    end
  end
  PU.CELL_NORM[] && return _set_cell_scales!(search_tier, spdf_plts, t4_ref, loss_params, eco_species_ids, n_species)
  (PU.W_NORMALIZE[] || PU.AGB_NORMALIZE[]) || return
  wsum = 0.0; asum = 0.0
  if search_tier == 4 && t4_ref !== nothing
    for eco in t4_ref, cycmat in eco, sp in 1:size(cycmat, 1)
      row = @view cycmat[sp, :]; tot = sum(row); asum += tot
      tot > 0 || continue
      acc = 0.0
      @inbounds for k in eachindex(bw); acc += row[k]; cdf = acc / tot; wsum += bw[k] * max(cdf, 1.0 - cdf); end
    end
  else
    for (_, yd) in spdf_plts, (_, gt) in yd, (_, rec) in gt.records
      asum += Float64(rec.sp_agb_sum)
      @inbounds for k in eachindex(bw); wsum += bw[k] * max(rec.sp_age_cdf[k], one(FloatType) - rec.sp_age_cdf[k]); end
    end
  end
  PU.W_SCALE[] = wsum > 0 ? FloatType(wsum) : one(FloatType)
  PU.AGB_SCALE[] = asum > 0 ? FloatType(asum) : one(FloatType)
end

# Per-cell (eco×lu×sp) scales for CELL_NORM, computed from the reference (constant across evals). Each
# cell (eco e, global species gsp) gets W_SCALE = Σ per-ref-max W1 and AGB_SCALE = Σ observed AGB. Sim A
# (tier≠4) reads the per-plot spdf records → A tables (+ sets the shared RANKW by AGB); Sim B (tier 4)
# reads the aggregated t4_ref → B tables (sets RANKW only if A hasn't, i.e. B-only). RANKW[gsp,e] =
# (1/log(rank+1))/Σ over dominance-included cells, rank by observed AGB descending.
function _set_cell_scales!(search_tier, spdf_plts, t4_ref, loss_params, eco_species_ids, n_species)
  # Scales are recomputed every eval from the CURRENT split's reference — they're intensive ratio-of-sums
  # (Σ|sim−obs|/Σobs), so train and val are on the same scale by construction (no plot-count deflation).
  # Only the RANK is frozen after the first (train) eval, so train and val weight the same cells the same way.
  bw = loss_params.age_bins.bin_widths
  ne = length(eco_species_ids)
  W = zeros(FloatType, n_species, ne); A = zeros(FloatType, n_species, ne)
  if search_tier == 4 && t4_ref !== nothing
    useB = PU.CBAL_ON[] && PU.CBAL_MODE[] !== :a && !isempty(PU.CBAL_W[])   # reweight the ref-scale to match calculate_t4_loss (Sim B): direct cbw[k]
    cp = (T4_PER_PLOT_MEAN[] && T4_CELL_PLOTS[] !== nothing) ? T4_CELL_PLOTS[] : nothing
    for (e, eco) in enumerate(t4_ref)
      sp_map = eco_species_ids[e]
      for (cyc, cycmat) in enumerate(eco), sp in 1:size(cycmat, 1)
        gsp = sp_map[sp]
        # AGB scale gets the SAME per-plot-mean division as the loss (W scale is CDF-based → left as-is).
        pcinv = cp === nothing ? one(FloatType) : one(FloatType) / FloatType(max(1, cp[e][cyc]))
        row = @view cycmat[sp, :]; tot = sum(row); A[gsp, e] += tot * pcinv
        tot > 0 || continue
        cbw = useB ? (@inbounds PU.CBAL_W[][gsp, e]) : FloatType[]
        rew = !isempty(cbw) && length(cbw) == length(bw)
        acc = zero(FloatType)
        @inbounds for k in eachindex(bw); acc += row[k]; cdf = acc / tot; W[gsp, e] += (rew ? cbw[k] : one(FloatType)) * bw[k] * max(cdf, one(FloatType) - cdf); end
      end
    end
    PU.W_SCALE_FACTOR[] != one(FloatType) && (W .*= PU.W_SCALE_FACTOR[])
    PU.W_SCALE_B[] = W; PU.AGB_SCALE_B[] = A
    if !PU.CELL_NORM_FREEZE[]
      isempty(PU.RANKW[]) && _set_rankw!(A, eco_species_ids)   # B-only: rank from B (dual sets it in tier-3)
      PU.CELL_NORM_FREEZE[] = true   # first (train) dual eval done → freeze the RANK for the run
    end
  else
    useA = PU.CBAL_ON[] && PU.CBAL_MODE[] !== :b && !isempty(PU.CBAL_W[])   # reweight the ref-scale to match _w1_sum (Sim A): coarse[k] lookup
    coarse = PU.CBAL_COARSE[]
    for ((_, eco_id), yd) in spdf_plts, (_, gt) in yd, (sp_eco, rec) in gt.records
      gsp = eco_species_ids[eco_id][sp_eco]
      A[gsp, eco_id] += rec.sp_agb_sum
      cbw = useA ? (@inbounds PU.CBAL_W[][gsp, eco_id]) : FloatType[]
      rew = !isempty(cbw)
      @inbounds for k in eachindex(bw); W[gsp, eco_id] += (rew ? cbw[coarse[k]] : one(FloatType)) * bw[k] * max(rec.sp_age_cdf[k], one(FloatType) - rec.sp_age_cdf[k]); end
    end
    PU.W_SCALE_FACTOR[] != one(FloatType) && (W .*= PU.W_SCALE_FACTOR[])
    PU.W_SCALE_A[] = W; PU.AGB_SCALE_A[] = A
    if !PU.CELL_NORM_FREEZE[]
      nplots = zeros(Int, ne)                                  # plots per stratum (train) → split-size weight
      for ((_, eco_id), _) in spdf_plts; nplots[eco_id] += 1; end
      _set_rankw!(A, eco_species_ids; split_size=nplots)       # rank from train A's per-plot ref, once
    end
    DUAL_MODE[] == :off && (PU.CELL_NORM_FREEZE[] = true)      # single-sim (A-only): no tier-4 to freeze → freeze rank here
  end
end

# Rank dominance-included species WITHIN EACH STRATUM (eco_id) by observed AGB (desc), weight
# 1/√(log1p(rank)) (√ softens the decay so rarer species aren't crushed); normalise each stratum to 1,
# then weight strata by the ORDER OF MAGNITUDE of their split size — RANKW[:,e] ∝ log10(1+n_plots[e]).
# This sits between the global ranking (largest split owns everything, small splits starved) and equal
# per-stratum (small noisy splits over-weighted): a 939-plot split outweighs a 34-plot one by ~log10,
# not 28× and not 1×. split_size = nothing → equal strata (fallback for B-only). Total Σ RANKW = 1.
function _set_rankw!(agb::Matrix{FloatType}, eco_species_ids; split_size::Union{Nothing,AbstractVector}=nothing)
  R = zeros(FloatType, size(agb)); sw = zeros(Float64, length(eco_species_ids))
  for e in eachindex(eco_species_ids)
    cells = Tuple{FloatType,Int}[]
    for gsp in eco_species_ids[e]
      _dom_included(gsp) || continue
      push!(cells, (agb[gsp, e], gsp))
    end
    isempty(cells) && continue
    sort!(cells; by=c -> -c[1])     # descending AGB within this stratum → rank 1 = most common here
    tot = zero(FloatType)
    for (rank, (_, gsp)) in enumerate(cells)
      w = FloatType(1.0 / sqrt(log1p(rank))); R[gsp, e] = w; tot += w
    end
    tot > 0 && (@views R[:, e] ./= tot)                                   # within-stratum sums to 1
    sw[e] = split_size === nothing ? 1.0 : log10(1.0 + Float64(split_size[e]))  # order of magnitude of split
  end
  tw = sum(sw)
  tw > 0 && for e in eachindex(eco_species_ids); @views R[:, e] .*= FloatType(sw[e] / tw); end  # Σ = 1
  PU.RANKW[] = R
end

function fit_params(ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier::Int=3, t1_ref::Union{Nothing,Vector{Matrix{FloatType}}}=nothing, t2_ref=nothing, seeds::AbstractVector=[nothing], injection_dict=nothing, injection_years=Set{Int}(), t4_ref=nothing, cycle_map=nothing, n_cycles::Int=0, dual_b=nothing, disturbance_dict=nothing, disturbance_years=Set{Int}())
  _set_loss_scales!(search_tier, spdf_plts, t4_ref, loss_params, eco_species_ids, n_species)   # global ratio-of-sums or per-cell denominators
  # Disturbance exclude-modes derive a (year → site_idx → excluded eco_species) lookup from the
  # injection cohorts with drop>0, so those (site, species) are skipped from the loss at that year.
  exclusion_dict = nothing
  if OVERRIDE_INJECTION_DISTURBANCE[] in (:exclude_overwrite, :exclude_noscale) && injection_dict !== nothing
    exclusion_dict = Dict{Int,Dict{Int,Set{UIntType}}}()
    for (yr, sites) in injection_dict
      sd = Dict{Int,Set{UIntType}}()
      for (site_idx, cohorts) in sites
        s = Set{UIntType}(sp for (sp, _age, _bio, drop) in cohorts if drop > zero(FloatType))
        isempty(s) || (sd[site_idx] = s)
      end
      isempty(sd) || (exclusion_dict[yr] = sd)
    end
  end
  # B-only mode: skip Sim A entirely and evaluate only Sim B (free process), seeded from A's params.
  if dual_b !== nothing && get(dual_b, :b_only, false)
    b_seeds = UInt64[hash((UInt64(seeds[1]), i)) for i in eachindex(seeds)]   # distinct seeds for the stochastic free sim
    resB = fit_params(dual_b.ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, dual_b.spdf_plts,
      site_sim_years, dual_b.spinup, dual_b.spinup_cohorts, dual_b.loss_params; debug, search_tier=dual_b.tier,
      t4_ref=dual_b.t4_ref, cycle_map=dual_b.cycle_map, n_cycles=dual_b.n_cycles, seeds=b_seeds,
      injection_dict=dual_b.injection_dict, injection_years=dual_b.injection_years,
      disturbance_dict=dual_b.disturbance_dict, disturbance_years=dual_b.disturbance_years, dual_b=nothing)
    b_losses = Float64[convert(Float64, PU.get_total_loss(r[1])) for r in resB]
    bi = argmin(abs.(b_losses .- Statistics.median(b_losses)))   # B's median rep
    return [resB[bi]]
  end
  init_scales = _init_scales(length(seeds))   # per-rep initial-biomass scale (identity unless perturbing)
  # Mean-over-reps (tier-4 free sim): accumulate every rep's sim histogram, then score ONE loss on the mean
  # histogram (loss-of-mean = the model's EXPECTED distribution) instead of a median-rep pick. Reps run
  # sequentially (plain map), so the shared accumulator is race-free.
  _mor = T4_MEAN_OVER_REPS[] && search_tier == 4 && dual_b === nothing && length(seeds) > 1
  _mor_hist = _mor ? [[zeros(FloatType, length(eco_species_ids[e]), size(t4_ref[1][1], 2)) for _ in 1:n_cycles] for e in eachindex(eco_species_ids)] : nothing
  _mor_counts = Ref{Any}(nothing)
  resA = map(eachindex(seeds)) do ri
    soa = copy_and_reseed_soa(ref_soa, seeds[ri])
    if init_scales[ri] != one(FloatType)       # perturb the sim-year-0 population (not scored, only propagated)
      sc = init_scales[ri]; cap = INIT_PERTURB_CAP[]
      for i in 1:soa.n
        site = getsite(soa, i)
        site.active || continue
        @inbounds for j in 1:Int(site.live)
          b = site.c_bio[j]
          # perturbed = max(2, bio + sign(pct)·min(bio·|pct|, max_perturb)) ≡ max(2, bio + clamp(bio·pct, ±cap));
          # max_perturb (cap) bounds the DEVIATION, 2-floor hardcoded, sc = 1+pct.
          site.c_bio[j] = max(FloatType(2), b + clamp(b * (sc - one(FloatType)), -cap, cap))
        end
      end
    end
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
    elseif search_tier == 4 || search_tier == 5
      # tier 4 (and tier 5, which is tier 3 + tier 4) need per-cycle bin accumulators.
      n_bins = size(t4_ref[1][1], 2)
      # per (eco, cycle): #measurements (obs) and #distinct plots, mirroring tier-1 counts
      t4_site_counts = [zeros(Int, n_cycles) for _ in eachindex(eco_species_ids)]
      t4_obs_counts = [zeros(Int, n_cycles) for _ in eachindex(eco_species_ids)]
      for i in 1:soa.n
        site = getsite(soa, i)
        !site.active && continue
        seen = Set{Int}()
        for sy in site_sim_years.sim_years[site.mapcode]
          sy < starting_sim_year && continue
          cyc = get(cycle_map, (Int(site.mapcode), Int(sy)), 0)
          cyc == 0 && continue
          t4_obs_counts[site.eco_id][cyc] += 1
          (cyc in seen) || (push!(seen, cyc); t4_site_counts[site.eco_id][cyc] += 1)
        end
      end
      t4_sim_t = [[[zeros(FloatType, length(eco_species_ids[eco_id]), n_bins) for _ in 1:n_cycles] for eco_id in eachindex(eco_species_ids)] for _ in 1:Threads.maxthreadid()]
      max_cohorts_scratch = Int(maximum(soa.refs.cohort[i+1] - soa.refs.cohort[i] for i in 1:soa.n))
      max_age_scratch = max_sim_year + max(1, length(loss_params.smoothing_weights) >> 1) + 5
      scratch_perm_t = [Vector{Int}(undef, max_cohorts_scratch) for _ in 1:Threads.maxthreadid()]
      scratch_ages_t = [Vector{FloatType}(undef, max_age_scratch) for _ in 1:Threads.maxthreadid()]
    end
    if search_tier == 3 || search_tier == 5
      # tier 3 (and tier 5): per-ecoregion accumulators so the MO search can read a per-(eco,species) loss
      # breakdown. These regroup the same per-site SiteLoss values run_result is summed from.
      eco3_w = [zeros(FloatType, n_species) for _ in eachindex(eco_species_ids)]
      eco3_agb = [zeros(FloatType, n_species) for _ in eachindex(eco_species_ids)]
      eco3_site_agb = zeros(FloatType, length(eco_species_ids))
      eco3_obs = zeros(Int, length(eco_species_ids))
    end
    for current_sim_year in starting_sim_year:max_sim_year
      #println("\ttimestep $(t)")
      PanCore.process_plugin!(soa, BiomassSuccessionPlugin.BiomassSuccession, current_sim_year; ctx=ctx.BiomassSuccession)
      # Sim B (free) disturbance: apply the observed exogenous biomass drop to the free sim — reduce a free-sim
      # cohort's biomass by the observed drop fraction ONLY IF it matches the disturbed cohort in SPECIES AND
      # AGE (per (plot,year,species,age)). The model responds to disturbance but does not predict it; no cohort
      # injection (stays free regen). Unmatched disturbed cohorts (the model didn't grow them) are not applied.
      if disturbance_dict !== nothing && current_sim_year in disturbance_years
        yd = disturbance_dict[current_sim_year]
        Threads.@threads :static for i in 1:soa.n
          @inbounds begin
            site = getsite(soa, i)
            spd = get(yd, Int(site.mapcode), nothing)
            spd === nothing && continue
            for j in 1:site.live
              d = get(spd, (site.c_species[j], UIntType(round(site.c_age[j]))), zero(FloatType))
              if d > zero(FloatType)
                drop = site.c_bio[j] * d
                site.c_bio[j] -= drop
                site.B -= drop
              end
            end
          end
        end
      end
      if !isnothing(injection_dict) && current_sim_year in injection_years
        _inject_observed_cohorts!(soa, injection_dict[current_sim_year]; override=OVERRIDE_INJECTION[], replace=OVERRIDE_INJECTION_REPLACE[], sync=OVERRIDE_INJECTION_SYNC[])
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
      elseif search_tier == 4
        Threads.@threads :static for i in 1:soa.n
          @inbounds begin
            site = getsite(soa, i)
            !site.active && continue
            sim_years = site_sim_years.sim_years[site.mapcode]
            if current_sim_year in sim_years
              cyc = get(cycle_map, (Int(site.mapcode), current_sim_year), 0)
              if cyc != 0
                accumulate_site_bins!(t4_sim_t[Threads.threadid()][site.eco_id][cyc], site, loss_params, scratch_perm_t[Threads.threadid()], scratch_ages_t[Threads.threadid()])
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
      elseif search_tier == 5
        # tier 5 = tier 3 (per-measurement per-site loss) + tier 4 (per-cycle bins) in ONE sim pass.
        sites_results = Vector{PU.SiteLoss}(undef, soa.n)
        Threads.@threads :static for i in 1:soa.n
          @inbounds begin
            site = getsite(soa, i)
            !site.active && continue
            sim_years = site_sim_years.sim_years[site.mapcode]
            if current_sim_year in sim_years
              # tier-4 contribution: accumulate this measurement into its cycle's bins
              cyc = get(cycle_map, (Int(site.mapcode), current_sim_year), 0)
              if cyc != 0
                accumulate_site_bins!(t4_sim_t[Threads.threadid()][site.eco_id][cyc], site, loss_params, scratch_perm_t[Threads.threadid()], scratch_ages_t[Threads.threadid()])
              end
              # tier-3 contribution: per-site loss at this measurement year
              spdf_plt = spdf_plts[(site.ref_cn, site.eco_id)]
              excluded = nothing
              if exclusion_dict !== nothing
                yd = get(exclusion_dict, current_sim_year, nothing)
                yd === nothing || (excluded = get(yd, i, nothing))
              end
              sites_results[i] = PU.calculate_site_loss2(current_sim_year, site, n_species, eco_species_ids, spdf_plt[current_sim_year], loss_params; debug=debug, excluded=excluded)
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
        for i in 1:soa.n
          isassigned(sites_results, i) || continue
          eco_id = Int(getsite(soa, i).eco_id)
          sl = sites_results[i]
          eco3_w[eco_id] .+= sl.sp_w_loss
          eco3_agb[eco_id] .+= sl.sp_agb_loss
          eco3_site_agb[eco_id] += sl.site_agb_loss
          eco3_obs[eco_id] += 1
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
              excluded = nothing
              if exclusion_dict !== nothing
                yd = get(exclusion_dict, current_sim_year, nothing)
                yd === nothing || (excluded = get(yd, i, nothing))
              end
              sloss = PU.calculate_site_loss2(current_sim_year, site, n_species, eco_species_ids, spdf_plt[current_sim_year], loss_params; debug=debug, excluded=excluded)
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
        for i in 1:soa.n
          isassigned(sites_results, i) || continue
          eco_id = Int(getsite(soa, i).eco_id)
          sl = sites_results[i]
          eco3_w[eco_id] .+= sl.sp_w_loss
          eco3_agb[eco_id] .+= sl.sp_agb_loss
          eco3_site_agb[eco_id] += sl.site_agb_loss
          eco3_obs[eco_id] += 1
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
      eco_losses = calculate_t2_loss(t2_sim_bins, t2_sim_total, t2_ref, n_species, eco_species_ids, eco_site_counts, eco_obs_counts, loss_params)
      run_result = sum(eco_losses)
    elseif search_tier == 4
      for tid in 2:Threads.maxthreadid()
        for eco_id in eachindex(eco_species_ids)
          for cyc in 1:n_cycles
            t4_sim_t[1][eco_id][cyc] .+= t4_sim_t[tid][eco_id][cyc]
          end
        end
      end
      eco_losses = calculate_t4_loss(t4_sim_t[1], t4_ref, loss_params, n_species, eco_species_ids, t4_site_counts, t4_obs_counts, n_cycles)
      run_result = sum(eco_losses)
      if _mor_hist !== nothing   # accumulate this rep's sim histogram for the mean-over-reps pass
        for eco_id in eachindex(eco_species_ids), cyc in 1:n_cycles
          _mor_hist[eco_id][cyc] .+= t4_sim_t[1][eco_id][cyc]
        end
        _mor_counts[] === nothing && (_mor_counts[] = (t4_site_counts, t4_obs_counts))
      end
    elseif search_tier == 5
      # tier 5 = tier 3 + tier 4. SO: run_result is the SUM of the two scalar SiteLosses.
      # MO: eco_losses = [tier-3 per-eco …; tier-4 per-eco …] (length 2·n_eco) → _mo_objectives
      # then yields 4·n_ess objectives (W,AGB)×{tier3,tier4} (see its modulo over eco_species_ids).
      eco_losses_3 = [PU.SiteLoss(sp_w_loss=eco3_w[e], sp_agb_loss=eco3_agb[e], site_agb_loss=eco3_site_agb[e], num_sites=eco3_obs[e], num_obs=eco3_obs[e]) for e in eachindex(eco_species_ids)]
      for tid in 2:Threads.maxthreadid(), eco_id in eachindex(eco_species_ids), cyc in 1:n_cycles
        t4_sim_t[1][eco_id][cyc] .+= t4_sim_t[tid][eco_id][cyc]
      end
      eco_losses_4 = calculate_t4_loss(t4_sim_t[1], t4_ref, loss_params, n_species, eco_species_ids, t4_site_counts, t4_obs_counts, n_cycles)
      run_result = sum(PU.skipundef(years_results)) + sum(eco_losses_4)
      eco_losses = vcat(eco_losses_3, eco_losses_4)
    else
      # Per-ecoregion breakdown (num_sites mirrors num_obs here, as in the scalar tier-3 sum).
      eco_losses = [PU.SiteLoss(sp_w_loss=eco3_w[e], sp_agb_loss=eco3_agb[e], site_agb_loss=eco3_site_agb[e], num_sites=eco3_obs[e], num_obs=eco3_obs[e]) for e in eachindex(eco_species_ids)]
      run_result = sum(PU.skipundef(years_results))
    end
    #@assert !any(isnan.(run_result.sp_w_loss)) "run NaN"
    cached_sites_state = [cohort for cohorts in sites_data for cohort in cohorts]
    (run_result, cached_sites_state, eco_losses)
  end
  # Dual-mode: also run Sim B (free process, no_establish baked into dual_b.ref_soa, tier-4) and stack
  # its per-eco losses after Sim A's via vcat — _mo_objectives turns this into A⊕B objectives.
  # Sim B is STOCHASTIC (free regen via site.rng), so it gets DISTINCT seeds per rep (A's perturbation
  # reps share one seed; reusing it would make B's reps identical and its spinup ignores the perturbation).
  # Aggregate A best-of-perturbation (via _agg_reps), B by the MEDIAN total (W+AGB-bin) loss over reps,
  # then combine A_best + B_median into ONE result (the SO loss = A_best + B_median).
  if _mor_hist !== nothing   # replace the per-rep results with ONE loss scored on the mean sim histogram
    ns = length(seeds)
    for e in eachindex(eco_species_ids), cyc in 1:n_cycles
      _mor_hist[e][cyc] ./= ns
    end
    sc, oc = _mor_counts[]
    eco_losses_m = calculate_t4_loss(_mor_hist, t4_ref, loss_params, n_species, eco_species_ids, sc, oc, n_cycles)
    resA = [(sum(eco_losses_m), _median_rep_cached(resA), eco_losses_m)]
  end
  dual_b === nothing && return resA
  b_seeds = UInt64[hash((UInt64(seeds[1]), i)) for i in eachindex(seeds)]
  resB = fit_params(dual_b.ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, dual_b.spdf_plts,
    site_sim_years, dual_b.spinup, dual_b.spinup_cohorts, dual_b.loss_params; debug, search_tier=dual_b.tier, t4_ref=dual_b.t4_ref,
    cycle_map=dual_b.cycle_map, n_cycles=dual_b.n_cycles, seeds=b_seeds, injection_dict=dual_b.injection_dict,
    injection_years=dual_b.injection_years, disturbance_dict=dual_b.disturbance_dict,
    disturbance_years=dual_b.disturbance_years, dual_b=nothing)
  a_run, a_eco, a_idx = _agg_reps(resA)
  b_losses = Float64[convert(Float64, PU.get_total_loss(r[1])) for r in resB]
  bi = argmin(abs.(b_losses .- Statistics.median(b_losses)))   # B's median rep (robust to a lucky/unlucky draw)
  return [(a_run + resB[bi][1], resA[a_idx][2], vcat(a_eco, resB[bi][3]))]
end
function parametrize_sobol(; ref_soa::ActiveSoA, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, rng::Random.AbstractRNG, debug::Bool, N::Int=100, M::Int=5, eval_tier::Int=3, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, cycle_years::Real=8, saltelli::Bool=false)
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
  elseif eval_tier == 4 || eval_tier == 5 || DUAL_MODE[] != :off
    n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    cycle_map, n_cycles = Data.build_cycle_map(splots; cycle_years=cycle_years)
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

  param_dists = BSP.make_biomass_param_dists(n_species, n_ecoregions, eco_species_ids; no_establishment=no_establishment, fit_establishment=(DUAL_MODE[] != :off))
  initial_params = BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
  injection_dict = isnothing(injection_cohorts) ? nothing : _build_injection_dict(injection_cohorts, ref_soa)
  injection_years = isnothing(injection_cohorts) ? Set{Int}() : Set(Int.(injection_cohorts.sim_year))
  samples, sal_tags = saltelli ? PU.saltelli_design(param_dists, initial_params, N) :
                                  (PU.sobol_samples(param_dists, initial_params, N), nothing)
  n_samples = length(samples)
  saltelli && @info "Saltelli design: N=$N base, $n_samples total evals (A+B+ABₖ over $(div(n_samples,N)-2) param-type groups)"

  # IDENTICAL eval to the downstream optimizers: dual Sim A⊕B, init-perturbation reps (best-of for A),
  # B median-over-distinct-seeds — i.e. a candidate's loss here == its CMA-MAE/CMA-ES quality, so the
  # seed pool ranks on the same loss surface. (M = n_reps perturbation reps via _fixed_seeds.)
  fixed_seeds = _fixed_seeds(rng, M)
  dual_b = DUAL_MODE[] == :off ? nothing : _build_dual_b(splots, eco_species_ids, eco_list, spdf_plts, loss_params, t4_ref, cycle_map, n_cycles, injection_cohorts, spinup_cohorts, rng, no_establishment; b_only=(DUAL_MODE[] == :b))
  _run(p) = fit_params(ref_soa, p, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params;
    debug=debug, search_tier=eval_tier, t1_ref=t1_ref, t2_ref=t2_ref, t4_ref=t4_ref, cycle_map=cycle_map, n_cycles=n_cycles,
    seeds=fixed_seeds, injection_dict=injection_dict, injection_years=injection_years, dual_b=dual_b)

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
      run_result, eco_losses, _ = _agg_reps(_run(bio_params))    # A best-of-perturbation + B median → one aggregate
      objs = _mo_objectives(eco_losses, eco_species_ids)         # EXACT downstream objective (cell-norm ÷scale, rank, power)
      loss = FloatType(_mo_aggregate(objs, run_result))          # = Σobjs under cell-norm; get_total_loss otherwise
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
      best_soa = make_sites(splots, eco_species_ids; rng, spinup, no_establishment=no_establishment)
      best_result = only(fit_params(best_soa, best_params, site_sim_years.sim_years .|> maximum |> maximum,
        length(species_list), eco_species_ids, spdf_plts,
        site_sim_years, spinup, spinup_cohorts, loss_params;
        debug, search_tier=eval_tier, t1_ref=t1_ref, t2_ref=t2_ref, t4_ref=t4_ref, cycle_map=cycle_map, n_cycles=n_cycles,
        seeds=[rand(rng, UInt64)],
        injection_dict=injection_dict, injection_years=injection_years))
      sim_sample = _filter_cached_to_df(best_result[2], sampled_ids)
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

# Initial-condition perturbation. When INIT_PERTURB_FRAC[] > 0 the n_reps "reps" are NOT RNG re-seeds
# but evenly-spaced scalings of the INITIAL cohort biomass over [1-frac, 1+frac] (the sim-year-0
# population only — year 0 is never scored, so this is pure boundary-condition slack), and an
# individual's fitness is the BEST (min) rep, not the sum. frac=0 ⇒ legacy behavior (seed-varied reps,
# summed). fit_params reads this Ref directly off the number of seeds, so callers need no new args.
const INIT_PERTURB_FRAC = Ref{Float64}(0.0)
# Optional absolute cap (g/m²) on the per-cohort biomass CHANGE: the perturbation is bio*(scale-1)
# clamped to ±INIT_PERTURB_CAP, so e.g. 5% on a 4000 g/m² cohort is limited to ±100 instead of ±200.
# Inf ⇒ uncapped (pure multiplicative scaling).
const INIT_PERTURB_CAP = Ref{FloatType}(FloatType(Inf))
# n evenly-spaced initial-biomass scale factors across [1-frac, 1+frac]; identity when off or n≤1.
_init_scales(n::Int) = (INIT_PERTURB_FRAC[] <= 0 || n <= 1) ? fill(one(FloatType), n) :
  FloatType[1 - INIT_PERTURB_FRAC[] + 2 * INIT_PERTURB_FRAC[] * (k - 1) / (n - 1) for k in 1:n]
# Per-rep RNG seeds. Perturb mode shares ONE seed across reps so the only thing that varies is the
# initial-biomass scaling (sync path has no RNG influence anyway, but this makes it exact); legacy
# draws an independent seed per rep.
_fixed_seeds(rng, n::Int) = INIT_PERTURB_FRAC[] > 0 ? fill(rand(rng, UInt64), n) : [rand(rng, UInt64) for _ in 1:n]

# Aggregate per-rep results into (run_result, eco_losses, picked_idx). Legacy (frac=0): SUM across reps.
# Perturb mode (frac>0 & >1 rep): pick the SINGLE BEST rep (min total loss).
function _agg_reps(rep_results)
  if INIT_PERTURB_FRAC[] > 0 && length(rep_results) > 1
    b = argmin(Float64[convert(Float64, PU.get_total_loss(r[1])) for r in rep_results])
    return rep_results[b][1], rep_results[b][3], b
  end
  run_result = sum(r[1] for r in rep_results)
  eco_losses = isnothing(rep_results[1][3]) ? nothing : [sum(r[3][e] for r in rep_results) for e in eachindex(rep_results[1][3])]
  return run_result, eco_losses, 1
end

function _median_rep_cached(rep_results)
  losses = Float64[convert(Float64, PU.get_total_loss(r[1])) for r in rep_results]
  # perturb mode caches the BEST rep's trajectory (matching _agg_reps); legacy caches the median rep.
  (INIT_PERTURB_FRAC[] > 0 && length(rep_results) > 1) && return rep_results[argmin(losses)][2]
  rep_results[argmin(abs.(losses .- Statistics.median(losses)))][2]
end

# Dual-mode (Sim A sync + Sim B free). When DUAL_MODE[] is set: establishment params are fit, and each
# candidate is evaluated with a second simulation (free process: model regenerates natural plots, planting
# handed in for artificial plots) whose tier-4 per-eco losses are stacked after Sim A's via fit_params'
# dual_b. Errors attribute to the process each param governs (growth←A, introduction←B).
const DUAL_MODE = Ref{Symbol}(:off)   # :off (Sim A only) · :joint (A⊕B) · :b (Sim B only, seed from A)
const TIER_B = Ref{Int}(4)            # Sim B's tier (yaml tier_b; A's tier is the config `tier`)
const LBSA_STRETCH_LEN = Ref{Int}(150)  # LBSA stretch length (yaml lbsa_stretch_len); shorter → faster freeze detection
const LBSA_STALE_RATIO = Ref{Float64}(0.95)  # LBSA up_attempt_stale_ratio (yaml lbsa_stale_ratio); lower → freezes/reheats sooner
const LBSA_TEMP_LIST_LEN = Ref{Int}(150)     # LBSA temperature-list length L (yaml lbsa_temp_list_len)
const LBSA_REPLACE_OLDEST = Ref{Bool}(true)  # LBSA replace_oldest_instead_of_max (yaml lbsa_replace_oldest); false → overwrite the MAX each stretch (temp adapts every stretch instead of a full L-cycle)
const LBSA_COOLING_ONLY = Ref{Bool}(false)   # LBSA cooling_only_schedule (yaml lbsa_cooling_only); true → max temp can only decrease (monotone cooling) so the search actually freezes → freeze/reheat/restart chain fires. false → temp self-regulates to accepted-uphill mean and never freezes.
const SIMB_SPINUP = Ref{Bool}(true)   # Sim B start (yaml simB_spinup): true = spin up from a deficit (back-cast);
                                      # false = start from the OBSERVED year-0 cohorts (anchored), still free establishment
# yaml simB_disturb_only: Sim B is PURELY FREE — NO cohort injection anywhere (natural AND artificial run free
# from the year-0 initial cohorts, model establishment via PROB_ESTAB stays ON), and the observed disturbance
# drops are applied as scale-reductions to EVERY plot's matching-species cohorts (no artificial exclusion, no
# planting sync). Contrast the default: natural free + artificial planting-synced + disturbance excluded on artificial.
const SIMB_DISTURB_ONLY = Ref{Bool}(false)
# Tier-4 (Sim B) loss options. PER_PLOT_MEAN: divide each (eco,cycle,species) cell histogram by its plot count →
# AGB term is mean biomass-per-plot-by-age (intensive; W is CDF-normalized so unaffected). MEAN_OVER_REPS: for
# the stochastic free sim, average the n_reps' tier-4 sim histograms and score the loss ONCE on the mean
# histogram (loss-of-mean, the model's EXPECTED distribution) instead of picking a median rep.
const T4_PER_PLOT_MEAN = Ref{Bool}(false)   # yaml simB_per_plot_mean
const T4_MEAN_OVER_REPS = Ref{Bool}(false)  # yaml simB_rep_mean
# Per-(eco,cycle) distinct-plot counts for the tier-4 ref, computed once in _build_dual_b. Read by BOTH
# the AGB scale (_set_cell_scales!) and the loss (calculate_t4_loss) so per-plot-mean stays consistent.
const T4_CELL_PLOTS = Ref{Union{Nothing,Vector{Vector{Int}}}}(nothing)
# Build Sim B's dual_b NamedTuple: free-process SoA + injection restricted to artificial ecos (natural
# plots run free), sharing A's t4_ref/cycle bins (Sim B is tier-4).
function _build_dual_b(splots, eco_species_ids, eco_list, spdf_plts, loss_params, t4_ref, cycle_map, n_cycles, injection_cohorts, spinup_cohorts, rng, no_establishment; b_only::Bool=false)
  # Per-(eco,cycle) distinct-plot count, matching the tier-4 ref build (all measured years per cycle). Used
  # by T4_PER_PLOT_MEAN to turn the AGB level into mean biomass-per-plot-by-age (consistently in scale+loss).
  cell_plots = [zeros(Int, max(1, n_cycles)) for _ in eachindex(eco_species_ids)]
  for ((plot_id, eco_id), year_dict) in spdf_plts
    cycs = Set{Int}()
    for (sy, _) in year_dict
      cyc = get(cycle_map, (Int(plot_id), Int(sy)), 0)
      cyc == 0 || push!(cycs, cyc)
    end
    for cyc in cycs; cell_plots[eco_id][cyc] += 1; end
  end
  T4_CELL_PLOTS[] = cell_plots
  art = Set(e for e in eachindex(eco_list) if occursin("artificial", lowercase(eco_list[e])))
  art_plots = Set(Int(r.plot_id) for r in eachrow(splots) if Int(r.eco_id) in art)
  # SIMB_DISTURB_ONLY: no injection at all (purely free everywhere); else default = plant/sync artificial plots.
  inj_b = (SIMB_DISTURB_ONLY[] || isnothing(injection_cohorts)) ? nothing : filter(r -> Int(r.plot_id) in art_plots, injection_cohorts)
  # Sim B start: SIMB_SPINUP[]=true spins up from a deficit (model establishes the stand from scratch via the
  # back-cast); false starts from the OBSERVED year-0 cohorts (anchored to data, no invented history) while
  # still running free establishment forward.
  ref_b = make_sites(splots, eco_species_ids; rng=rng, spinup=SIMB_SPINUP[], no_establishment=false)
  idict_b = isnothing(inj_b) ? nothing : _build_injection_dict(inj_b, ref_b)
  iyears_b = isnothing(inj_b) ? Set{Int}() : Set(Int.(inj_b.sim_year))
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
      (!SIMB_DISTURB_ONLY[] && Int(r.plot_id) in art_plots) && continue   # disturb-only: apply to ALL plots
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
  end
  @info "Sim B dual_b: $(ref_b.n) sites, $(length(art_plots)) artificial plots | injection=$(idict_b === nothing ? "OFF" : "ON ($(length(iyears_b)) yrs, artificial)") | disturbance-scale=$(isempty(ddict) ? "OFF" : "ON ($(length(ddict)) yrs, $(SIMB_DISTURB_ONLY[] ? "ALL plots" : "natural only"))") | establishment=ON"
  return (ref_soa=ref_b, spdf_plts=spdf_plts, loss_params=loss_params, t4_ref=t4_ref,
    cycle_map=cycle_map, n_cycles=n_cycles, injection_dict=idict_b, injection_years=iyears_b,
    spinup=SIMB_SPINUP[], spinup_cohorts=spinup_cohorts, b_only=b_only, tier=TIER_B[],
    disturbance_dict=(isempty(ddict) ? nothing : ddict), disturbance_years=Set(keys(ddict)))
end

# Build Sim B for the VALIDATION split, so held-out loss measures the SAME A⊕B objective as training
# (otherwise val = Sim A only, which is not comparable to a dual train loss). Builds val tier-4 refs +
# cycle map, then the dual_b. Returns nothing when dual mode is off.
function _build_val_dual_b(val_splots, eco_species_ids, eco_list, val_spdf_plts, loss_params, cycle_years,
    val_injection_cohorts, val_spinup_cohorts, rng, no_establishment)
  DUAL_MODE[] == :off && return nothing
  vcm, vnc = Data.build_cycle_map(val_splots; cycle_years=cycle_years)
  nb = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
  vt4 = [[zeros(FloatType, length(eco_species_ids[e]), nb) for _ in 1:vnc] for e in eachindex(eco_list)]
  for ((pid, eid), yd) in val_spdf_plts, (sy, gt) in yd
    cyc = get(vcm, (Int(pid), Int(sy)), 0); cyc == 0 && continue
    for (sp_eco, rec) in gt.records; vt4[eid][cyc][sp_eco, :] .+= diff([0f0; rec.sp_age_cdf]) .* rec.sp_agb_sum; end
  end
  _build_dual_b(val_splots, eco_species_ids, eco_list, val_spdf_plts, loss_params, vt4, vcm, vnc,
    val_injection_cohorts, val_spinup_cohorts, rng, no_establishment; b_only=(DUAL_MODE[] == :b))
end

function _filter_cached_to_df(cached, sampled_ids::Set)
  filtered = filter(t -> UIntType(t[1]) in sampled_ids, cached)
  isempty(filtered) && return DataFrame(plot_id=UIntType[], sim_year=Int[], species_id=UIntType[], age=UIntType[], agb=FloatType[])
  DataFrame(filtered, [:plot_id, :sim_year, :species_id, :age, :agb])
end

function parametrize_LBSA(; ref_soa::ActiveSoA, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=3, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8)
  splots.sim_year .= Dates.value.(Dates.Day.(splots.measdate - splots.start_measdate)) ./ 365.25 .|> round .|> Int

  # Fixed plot sample chosen once at startup so progress is comparable across iterations
  all_plot_ids = UIntType.(unique(splots.plot_id))
  sampled_ids = _sample_plot_ids(all_plot_ids, n_output_plots, rng; injection_cohorts=injection_cohorts)
  emp_sample = n_output_plots > 0 ? _make_emp_df(splots, sampled_ids) : nothing

  n_species = length(species_list)
  max_sim_year = site_sim_years.sim_years .|> maximum |> maximum
  param_dists = BSP.make_biomass_param_dists(length(species_list), length(eco_list), eco_species_ids; no_establishment=no_establishment, fit_establishment=(DUAL_MODE[] != :off))
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  injection_dict = isnothing(injection_cohorts) ? nothing : _build_injection_dict(injection_cohorts, ref_soa)
  injection_years = isnothing(injection_cohorts) ? Set{Int}() : Set(Int.(injection_cohorts.sim_year))

  # Validation setup
  have_val = !isnothing(val_ref_soa)
  inj_dict_val = (have_val && !isnothing(val_injection_cohorts)) ? _build_injection_dict(val_injection_cohorts, val_ref_soa) : nothing
  inj_years_val = (have_val && !isnothing(val_injection_cohorts)) ? Set(Int.(val_injection_cohorts.sim_year)) : Set{Int}()
  val_dual_b = have_val ? _build_val_dual_b(val_splots, eco_species_ids, eco_list, val_spdf_plts, loss_params, cycle_years, val_injection_cohorts, val_spinup_cohorts, rng, no_establishment) : nothing
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
  elseif search_tier == 4 || search_tier == 5 || DUAL_MODE[] != :off
    n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    cycle_map, n_cycles = Data.build_cycle_map(splots; cycle_years=cycle_years)
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
  fixed_seeds = _fixed_seeds(rng, n_reps)
  # Score by the SAME Sim-A loss as the (MO-)CMA-ES drivers: under CELL_NORM the cell-normalized aggregate
  # (Σ of _mo_objectives — per-cell ÷scale, per-stratum RANKW), else the raw scalar. So LBSA optimizes the
  # identical objective and its convergence is directly comparable.
  _norm_loss(rr, el) = PU.CELL_NORM[] ? Float64(sum(_mo_objectives(el, eco_species_ids))) : convert(Float64, rr)
  if isnothing(resume_from)
    bio_params = if !isnothing(start_from)
      @info "Seeding initial candidate from $start_from (fresh search_state)"
      p = _load_params_from_path(start_from)
      (p.SPECIES_LIST == species_list && p.ECO_LIST == eco_list) ||
        error("start_from params are incompatible with this run: their SPECIES_LIST/ECO_LIST differ from the loaded data (e.g. different stratify_eco_mixed, species tiering, or eco/plot filters). Seeding requires matching ecoregions and species.")
      p
    elseif !isempty(_sobol_cands)
      @info "Using Sobol candidate 1/$(length(_sobol_cands)) as initial point"
      _sobol_cands[1]
    else
      BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    end
    init_rr, init_el, _ = _agg_reps(fit_params(ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, t4_ref, cycle_map, n_cycles, seeds=fixed_seeds, injection_dict=injection_dict, injection_years=injection_years))
    cur = LBSA.LBSACandidate(bio_params, _norm_loss(init_rr, init_el))
    search_state = LBSA.LBSAState(cur, cur, rng; max_iter=TRIALS, stretch_len=LBSA_STRETCH_LEN[], up_attempt_stale_ratio=LBSA_STALE_RATIO[], temp_list_len=LBSA_TEMP_LIST_LEN[], replace_oldest_instead_of_max=LBSA_REPLACE_OLDEST[], cooling_only_schedule=LBSA_COOLING_ONLY[])
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
      BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    end

  writer_ch, writer_task = start_writer(typeof(search_state), output_dir)

  losses_db_file = DuckDB.DB(joinpath(output_dir, "losses.duckdb"))
  losses_db = DuckDB.connect(losses_db_file)
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS total_loss (iteration INTEGER, n_sites INTEGER, n_obs INTEGER, total_loss DOUBLE, params_blob BLOB)")
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS ecoregion_loss (iteration INTEGER, ecoregion VARCHAR, eco_num_sites INTEGER, eco_num_obs INTEGER, ecoregion_total_loss DOUBLE)")
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS species_loss (iteration INTEGER, ecoregion VARCHAR, species VARCHAR, age_dist_loss DOUBLE, agb_loss DOUBLE)")

  # Dense convergence + restart/reheat trace: one row per trial — best-so-far & incumbent loss, max
  # temperature, the frozen-reheat counter, and a restart flag (1 when a restart fired this trial).
  # On resume, APPEND (keep the prior trace) and skip the header.
  conv_io = open(joinpath(output_dir, "convergence.csv"), isnothing(resume_from) ? "w" : "a")
  isnothing(resume_from) && (println(conv_io, "trial,best_loss,current_loss,t_max,frozen_reheats,restart"); flush(conv_io))

  baseline_weights = diff([0.0; param_dists.weights_cumsum])
  param_sensitivities = zeros(length(baseline_weights))
  sensitivity_decay = 0.999      # per-trial decay toward 0 for all params
  sensitivity_ema_alpha = 0.1    # EMA weight for chosen param's sensitivity update
  sensitivity_lambda = 3.0       # max weight boost at full sensitivity (baseline × (1 + λ))

  # A Ctrl-C inside a @threads region (e.g. mid-resize in readjust_soa!) surfaces as a
  # TaskFailedException wrapping an InterruptException, not a bare InterruptException.
  # Recurse through the task/composite wrappers to recognize it.
  caused_by_interrupt(e) =
    e isa InterruptException ? true :
    e isa TaskFailedException ? any(en -> caused_by_interrupt(en.exception), Base.current_exceptions(e.task)) :
    e isa CompositeException ? any(caused_by_interrupt, e.exceptions) :
    false

  try

    TProgress.@track for trial in (search_state.i+1):TRIALS
      dynamic_weights = baseline_weights .* (1.0 .+ sensitivity_lambda .* param_sensitivities)
      dynamic_weights ./= sum(dynamic_weights)
      dynamic_cumsum = cumsum(dynamic_weights)
      bio_params, chosen_param_idx = PU.mutate_params(bio_params, param_dists; rng=rng, mutation_mode=PU.BothMutations)
      #ctx=PU.SamplingContext(search_state.current.fx.sp_w_loss .+ search_state.current.fx.sp_agb_loss .+ (search_state.current.fx.sp_w_loss .* search_state.current.fx.sp_agb_loss)),
      #ctx=PU.SamplingContext(search_state.current.fx.sp_w_loss),
      # dynamic_cumsum=dynamic_cumsum)
      for _ in 0:rand(rng, 0:2)
        bio_params, _ = PU.mutate_params(bio_params, param_dists; rng=rng, mutation_mode=PU.BothMutations)#,  #BothMutations,
      end

      rep_results = fit_params(ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, t4_ref, cycle_map, n_cycles, seeds=fixed_seeds, injection_dict=injection_dict, injection_years=injection_years)
      run_result, eco_losses, _ = _agg_reps(rep_results)
      cached_sites_state = _median_rep_cached(rep_results)
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
        rst_rr, rst_el, _ = _agg_reps(fit_params(ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, t4_ref, cycle_map, n_cycles, seeds=fixed_seeds, injection_dict=injection_dict, injection_years=injection_years))
        is_new_best = LBSA.restart(search_state, LBSA.LBSACandidate(bio_params, _norm_loss(rst_rr, rst_el)))
      end
      val_sim_sample = nothing
      if is_new_best
        iter = search_state.best_iteration
        total = convert(Float64, search_state.best.fx)
        @info "New best @ $iter | loss=$total"
        try
          test_df = simulate_and_test(; splots=splots, bio_params=search_state.best.x, eco_list=eco_list, species_list=species_list, eco_species_ids=eco_species_ids, loss_params=loss_params, site_sim_years=site_sim_years, M=n_reps, no_establishment=no_establishment, rng=rng)
          println("Train stats:")
          show(test_df; allrows=true, allcols=true)
          println()
        catch e
          @warn "simulate_and_test (train) failed" exception = (e, catch_backtrace())
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
        let buf = IOBuffer()
          Serialization.serialize(buf, search_state.best.x)
          DuckDB.execute(losses_db, "INSERT INTO total_loss VALUES (?, ?, ?, ?, ?)", [iter, run_result.num_sites, run_result.num_obs, total, take!(buf)])
        end
        if !isnothing(eco_losses)
          for (eco_id, eco_loss) in enumerate(eco_losses)
            eco_name = _eco_name(eco_list, eco_id)
            eco_total = convert(Float64, PU.get_total_loss(eco_loss))
            DuckDB.execute(losses_db, "INSERT INTO ecoregion_loss VALUES (?, ?, ?, ?, ?)", [iter, eco_name, eco_loss.num_sites, eco_loss.num_obs, eco_total])
            n = max(1, eco_loss.num_sites)
            for gsp in 1:n_species
              eco_loss.sp_w_loss[gsp] == 0f0 && continue
              DuckDB.execute(losses_db, "INSERT INTO species_loss VALUES (?, ?, ?, ?, ?)", [iter, eco_name, species_list[gsp], eco_loss.sp_w_loss[gsp] / n, eco_loss.sp_agb_loss[gsp] / n])
            end
          end

          # Raw per-eco / per-species SiteLoss breakdown (ecoregions and species sorted by
          # contribution) so you can see where the remaining error concentrates.
          println("Raw SiteLoss breakdown @ $iter (total=$(round(total, sigdigits=6))):")
          for eco_id in sort(collect(eachindex(eco_losses)); by=e -> -convert(Float64, PU.get_total_loss(eco_losses[e])))
            el = eco_losses[eco_id]
            println("  [$(_eco_name(eco_list, eco_id))] eco_total=$(round(convert(Float64, PU.get_total_loss(el)), sigdigits=5))  sites=$(el.num_sites) obs=$(el.num_obs)")
            for gsp in sort([g for g in 1:n_species if el.sp_w_loss[g] != 0f0]; by=g -> -el.sp_w_loss[g])
              println("      $(rpad(species_list[gsp], 10)) w=$(round(el.sp_w_loss[gsp], sigdigits=4))  agb=$(round(el.sp_agb_loss[gsp], sigdigits=4))")
            end
          end
        end
      end
      let tmax = (isempty(search_state._t_list) || search_state._t_max_idx > length(search_state._t_list)) ? NaN : search_state._t_list[search_state._t_max_idx]
        println(conv_io, string(trial, ",", convert(Float64, search_state.best.fx), ",", convert(Float64, search_state.current.fx), ",", tmax, ",", search_state._frozen_no_best_reheats, ",", did_restart ? 1 : 0))
        (trial % 50 == 0 || did_restart) && flush(conv_io)
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
  catch e
    # Swallow a user interrupt (possibly wrapped by a @threads region) and let `finally`
    # save the checkpoint; rethrow anything that isn't an interrupt.
    if caused_by_interrupt(e)
      @info "Search interrupted by user @ trial $(search_state.i); finalizing checkpoint…"
    else
      rethrow()
    end
  finally
    stop_writer(writer_ch, writer_task)
    close(losses_db_file); close(conv_io)
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

  return search_state
end

# CMA-ES driver. Mirrors parametrize_LBSA's setup and checkpoint/writer machinery, but optimizes
# the Biomass Succession params with Covariance Matrix Adaptation in the vector bridge's [0,1]^d
# u-space. Each generation samples λ candidates and evaluates them SERIALLY (fit_params already
# threads internally), then updates the CMA-ES distribution. TRIALS is the total fit_params budget
# (so it's comparable to LBSA); λ defaults to Hansen's 4+⌊3·ln(n)⌋. `ipop=true` enables IPOP
# restarts (doubling λ from a fresh mean on σ-collapse / stagnation); default is a single run.
function parametrize_CMAES(; ref_soa::ActiveSoA, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=3, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, cmaes_lambda::Union{Nothing,Int}=nothing, cmaes_sigma0::Float64=0.3, ipop::Bool=false, ipop_stagnation::Int=20, integer_handling::Bool=false, integer_std_factor::Float64=0.3, single_cov::Bool=false)
  splots.sim_year .= Dates.value.(Dates.Day.(splots.measdate - splots.start_measdate)) ./ 365.25 .|> round .|> Int

  # Fixed plot sample chosen once at startup so progress is comparable across iterations
  all_plot_ids = UIntType.(unique(splots.plot_id))
  sampled_ids = _sample_plot_ids(all_plot_ids, n_output_plots, rng; injection_cohorts=injection_cohorts)
  emp_sample = n_output_plots > 0 ? _make_emp_df(splots, sampled_ids) : nothing

  n_species = length(species_list)
  max_sim_year = site_sim_years.sim_years .|> maximum |> maximum
  param_dists = BSP.make_biomass_param_dists(length(species_list), length(eco_list), eco_species_ids; no_establishment=no_establishment, fit_establishment=(DUAL_MODE[] != :off))
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  injection_dict = isnothing(injection_cohorts) ? nothing : _build_injection_dict(injection_cohorts, ref_soa)
  injection_years = isnothing(injection_cohorts) ? Set{Int}() : Set(Int.(injection_cohorts.sim_year))

  # Validation setup
  have_val = !isnothing(val_ref_soa)
  inj_dict_val = (have_val && !isnothing(val_injection_cohorts)) ? _build_injection_dict(val_injection_cohorts, val_ref_soa) : nothing
  inj_years_val = (have_val && !isnothing(val_injection_cohorts)) ? Set(Int.(val_injection_cohorts.sim_year)) : Set{Int}()
  val_dual_b = have_val ? _build_val_dual_b(val_splots, eco_species_ids, eco_list, val_spdf_plts, loss_params, cycle_years, val_injection_cohorts, val_spinup_cohorts, rng, no_establishment) : nothing
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
  elseif search_tier == 4 || search_tier == 5 || DUAL_MODE[] != :off
    n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    cycle_map, n_cycles = Data.build_cycle_map(splots; cycle_years=cycle_years)
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
  fixed_seeds = _fixed_seeds(rng, n_reps)

  # Sobol-or-random source for the initial mean and any IPOP restart mean.
  sobol_idx = Ref(1)
  next_candidate() =
    if sobol_idx[] <= length(_sobol_cands)
      p = _sobol_cands[sobol_idx[]]
      @info "Using Sobol candidate $(sobol_idx[])/$(length(_sobol_cands))"
      sobol_idx[] += 1
      p
    else
      BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    end

  if isnothing(resume_from)
    bio_params = if !isnothing(start_from)
      @info "Seeding initial candidate from $start_from (fresh search_state)"
      p = _load_params_from_path(start_from)
      (p.SPECIES_LIST == species_list && p.ECO_LIST == eco_list) ||
        error("start_from params are incompatible with this run: their SPECIES_LIST/ECO_LIST differ from the loaded data (e.g. different stratify_eco_mixed, species tiering, or eco/plot filters). Seeding requires matching ecoregions and species.")
      p
    else
      next_candidate()
    end
    slots = PU.build_slots(param_dists, bio_params)
    init_run = _agg_reps(fit_params(ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, t4_ref, cycle_map, n_cycles, seeds=fixed_seeds, injection_dict=injection_dict, injection_years=injection_years))[1]
    mean0 = PU.params_to_u(bio_params, param_dists, slots)
    init_best = CMAES.CMAESCandidate(bio_params, convert(Float64, PU.get_total_loss(init_run)))
    groups = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(param_dists, slots, BSP.BIOMASS_PER_ECO_GROUPS))   # block-diagonal CMA-ES
    @info "CMA-ES block-diagonal: $(length(groups)) covariance blocks (sizes $(length.(groups)))"
    search_state = CMAES.CMAESState(mean0, cmaes_sigma0, init_best, rng; lambda=cmaes_lambda, blocks=groups, max_iter=typemax(Int))
    search_state.n_evals = 1
  else
    @info "Resuming from $resume_from"
    search_state = JLD2.load_object(resume_from)
    bio_params = search_state.best.x
    slots = PU.build_slots(param_dists, bio_params)
    if force_restart_from_random
      bio_params = next_candidate()
      CMAES.restart!(search_state, PU.params_to_u(bio_params, param_dists, slots), cmaes_sigma0)
    end
  end
  # Hansen-style mixed-integer handling (honors the flag each run, incl. on resume).
  search_state.u_min_std = integer_handling ? PU.integer_u_min_std(param_dists, slots, integer_std_factor) : Float64[]
  if TRIALS < 1 || search_state.n_evals >= TRIALS
    return search_state
  end

  writer_ch, writer_task = start_writer(typeof(search_state), output_dir)

  losses_db_file = DuckDB.DB(joinpath(output_dir, "losses.duckdb"))
  losses_db = DuckDB.connect(losses_db_file)
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS total_loss (iteration INTEGER, n_sites INTEGER, n_obs INTEGER, total_loss DOUBLE, params_blob BLOB)")
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS ecoregion_loss (iteration INTEGER, ecoregion VARCHAR, eco_num_sites INTEGER, eco_num_obs INTEGER, ecoregion_total_loss DOUBLE)")
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS species_loss (iteration INTEGER, ecoregion VARCHAR, species VARCHAR, age_dist_loss DOUBLE, agb_loss DOUBLE)")

  # A Ctrl-C inside a @threads region (e.g. mid-resize in readjust_soa!) surfaces as a
  # TaskFailedException wrapping an InterruptException, not a bare InterruptException.
  caused_by_interrupt(e) =
    e isa InterruptException ? true :
    e isa TaskFailedException ? any(en -> caused_by_interrupt(en.exception), Base.current_exceptions(e.task)) :
    e isa CompositeException ? any(caused_by_interrupt, e.exceptions) :
    false

  stagnation = 0
  evals_done = search_state.n_evals
  # Upper bound on generations needed to exhaust the budget (λ never shrinks: IPOP only grows it,
  # which only spends the budget faster). The `evals_done >= TRIALS` break is the real stop.
  est_gens = max(1, cld(TRIALS - evals_done, search_state.lambda))
  try
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
        cand = PU.u_to_params(xs_u[k], param_dists, slots, bio_params)
        rep_results = fit_params(ref_soa, cand, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, t4_ref, cycle_map, n_cycles, seeds=fixed_seeds, injection_dict=injection_dict, injection_years=injection_years)
        run_result, eco_losses_k, _ = _agg_reps(rep_results)
        fitnesses[k] = convert(Float64, PU.get_total_loss(run_result))
        evals_done += 1
        if fitnesses[k] < gen_best_f
          gen_best_f = fitnesses[k]
          gen_best_params = cand
          gen_best_run = run_result
          gen_best_eco = eco_losses_k
          gen_best_cached = _median_rep_cached(rep_results)
        end
      end

      CMAES.tell!(search_state, fitnesses, xs_u)
      search_state.n_evals = evals_done

      is_new_best = CMAES.note_best!(search_state, CMAES.CMAESCandidate(gen_best_params, gen_best_f))
      stagnation = is_new_best ? 0 : stagnation + 1

      val_sim_sample = nothing
      if is_new_best
        iter = search_state.best_iteration
        total = convert(Float64, search_state.best.fx)
        @info "New best @ gen $iter | loss=$total | σ=$(search_state.sigma)"
        try
          test_df = simulate_and_test(; splots=splots, bio_params=search_state.best.x, eco_list=eco_list, species_list=species_list, eco_species_ids=eco_species_ids, loss_params=loss_params, site_sim_years=site_sim_years, M=n_reps, no_establishment=no_establishment, rng=rng)
          println("Train stats:")
          show(test_df; allrows=true, allcols=true)
          println()
        catch e
          @warn "simulate_and_test (train) failed" exception = (e, catch_backtrace())
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
            @info "Val loss @ gen $iter | loss=$val_total"
            try
              val_test_df = simulate_and_test(; splots=val_splots, bio_params=search_state.best.x, eco_list=eco_list, species_list=species_list, eco_species_ids=eco_species_ids, loss_params=loss_params, site_sim_years=val_site_sim_years, M=n_reps, no_establishment=no_establishment, rng=rng)
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
        let buf = IOBuffer()
          Serialization.serialize(buf, search_state.best.x)
          DuckDB.execute(losses_db, "INSERT INTO total_loss VALUES (?, ?, ?, ?, ?)", [iter, gen_best_run.num_sites, gen_best_run.num_obs, total, take!(buf)])
        end
        if !isnothing(gen_best_eco)
          for (eco_id, eco_loss) in enumerate(gen_best_eco)
            eco_name = _eco_name(eco_list, eco_id)
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
      cached_sites_state_df = DataFrame(gen_best_cached, [:plot_id, :sim_year, :species_id, :age, :agb])
      sim_sample = (is_new_best && n_output_plots > 0) ? _filter_cached_to_df(gen_best_cached, sampled_ids) : nothing
      put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state), splots, cached_sites_state_df, emp_sample, sim_sample, is_new_best ? emp_sample_val : nothing, val_sim_sample))

      # IPOP restart: re-seed the distribution (with doubled λ) on σ-collapse or stagnation.
      if ipop && (search_state.sigma < 1e-11 || stagnation >= ipop_stagnation)
        new_lambda = search_state.lambda * 2
        @info "IPOP restart @ gen $(search_state.i): λ $(search_state.lambda) → $new_lambda"
        bio_params = next_candidate()
        CMAES.restart!(search_state, PU.params_to_u(bio_params, param_dists, slots), cmaes_sigma0; lambda=new_lambda)
        stagnation = 0
      end
    end
  catch e
    if caused_by_interrupt(e)
      @info "Search interrupted by user @ gen $(search_state.i); finalizing checkpoint…"
    else
      rethrow()
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

  return search_state
end

# Multi-objective CMA-ES driver. Stands to parametrize_CMAES as parametrize_MOLBSA stands to
# parametrize_LBSA: same generational CMA-ES loop and u-space bridge, but each candidate is scored by
# the per-(eco,species,{w,agb}) objective VECTOR (MOLBSA.MOFitness via _mo_objectives), offspring are
# ranked multi-objectively (MOCMAES.mo_sortperm) for the distribution update, and "best" is a Pareto
# archive + min-aggregate representative (reusing start_mo_writer). TRIALS is the total eval budget.
function parametrize_MOCMAES(; ref_soa::ActiveSoA, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, cmaes_lambda::Union{Nothing,Int}=nothing, cmaes_sigma0::Float64=0.3, cmaes_warmstart_seeds::Int=20, ipop::Bool=false, ipop_stagnation::Int=20, integer_handling::Bool=false, integer_std_factor::Float64=0.3, single_cov::Bool=false, seed_archive_from::Union{Nothing,String}=nothing)
  splots.sim_year .= Dates.value.(Dates.Day.(splots.measdate - splots.start_measdate)) ./ 365.25 .|> round .|> Int

  all_plot_ids = UIntType.(unique(splots.plot_id))
  sampled_ids = _sample_plot_ids(all_plot_ids, n_output_plots, rng; injection_cohorts=injection_cohorts)
  emp_sample = n_output_plots > 0 ? _make_emp_df(splots, sampled_ids) : nothing

  n_species = length(species_list)
  max_sim_year = site_sim_years.sim_years .|> maximum |> maximum
  param_dists = BSP.make_biomass_param_dists(length(species_list), length(eco_list), eco_species_ids; no_establishment=no_establishment, fit_establishment=(DUAL_MODE[] != :off))
  # Candidate pool for the initial point + IPOP restarts. seed_archive_from pours a PRIOR run's Pareto
  # archive (its members' params) into this pool — e.g. seed Sim B's MO-CMA-ES from the single-cov Sim A
  # archive so each restart re-centers on a diverse good-A solution (params are re-evaluated on use).
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
  injection_dict = isnothing(injection_cohorts) ? nothing : _build_injection_dict(injection_cohorts, ref_soa)
  injection_years = isnothing(injection_cohorts) ? Set{Int}() : Set(Int.(injection_cohorts.sim_year))

  have_val = !isnothing(val_ref_soa)
  inj_dict_val = (have_val && !isnothing(val_injection_cohorts)) ? _build_injection_dict(val_injection_cohorts, val_ref_soa) : nothing
  inj_years_val = (have_val && !isnothing(val_injection_cohorts)) ? Set(Int.(val_injection_cohorts.sim_year)) : Set{Int}()
  val_dual_b = have_val ? _build_val_dual_b(val_splots, eco_species_ids, eco_list, val_spdf_plts, loss_params, cycle_years, val_injection_cohorts, val_spinup_cohorts, rng, no_establishment) : nothing
  sampled_ids_val = (have_val && n_output_plots > 0) ?
                    _sample_plot_ids(UIntType.(unique(val_splots.plot_id)), n_output_plots, rng; injection_cohorts=val_injection_cohorts) :
                    Set{UIntType}()
  emp_sample_val = (have_val && n_output_plots > 0) ? _make_emp_df(val_splots, sampled_ids_val) : nothing

  # Per-tier reference setup (identical to the other drivers).
  t4_ref = nothing
  cycle_map = nothing
  n_cycles = 0
  t1_ref = nothing
  t2_ref = nothing
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
  elseif search_tier == 2
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
  elseif search_tier == 4 || search_tier == 5 || DUAL_MODE[] != :off
    n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    cycle_map, n_cycles = Data.build_cycle_map(splots; cycle_years=cycle_years)
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
  end
  fixed_seeds = _fixed_seeds(rng, n_reps)

  dual_b = DUAL_MODE[] == :off ? nothing : _build_dual_b(splots, eco_species_ids, eco_list, spdf_plts, loss_params, t4_ref, cycle_map, n_cycles, injection_cohorts, spinup_cohorts, rng, no_establishment; b_only=(DUAL_MODE[] == :b))
  _run(p) = fit_params(ref_soa, p, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, t4_ref, cycle_map, n_cycles, seeds=fixed_seeds, injection_dict=injection_dict, injection_years=injection_years, dual_b=dual_b)
  # Score a candidate's repetitions into one MOFitness (eco_losses summed over reps) + aux.
  function _fitness(rep_results)
    run_result, eco_losses, _ = _agg_reps(rep_results)
    objs = _mo_objectives(eco_losses, eco_species_ids)
    MOLBSA.MOFitness(objs, _mo_aggregate(objs, run_result)), run_result, eco_losses
  end

  sobol_idx = Ref(1)
  next_candidate() =
    if sobol_idx[] <= length(_sobol_cands)
      p = _sobol_cands[sobol_idx[]]
      sobol_idx[] += 1
      p
    else
      BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    end

  if isnothing(resume_from)
    bio_params = if !isnothing(start_from)
      @info "Seeding initial candidate from $start_from (fresh search_state)"
      p = _load_params_from_path(start_from)
      (p.SPECIES_LIST == species_list && p.ECO_LIST == eco_list) ||
        error("start_from params are incompatible with this run: their SPECIES_LIST/ECO_LIST differ from the loaded data.")
      p
    else
      next_candidate()
    end
    _seed_prob_estab!(bio_params, eco_list, species_list, eco_species_ids)   # calibration seed (in-place; stays in search)
    _seed_maturity!(bio_params, species_list)
    _seed_min_rel!(bio_params, eco_list)   # pin MIN_REL out of search (fix_min_rel)
    frozen_ref = deepcopy(bio_params)   # frozen-growth reference for IPOP restarts (fix_growth must survive restart)
    slots = PU.build_slots(param_dists, bio_params)
    fx0, _, _ = _fitness(_run(bio_params))
    mean0 = PU.params_to_u(bio_params, param_dists, slots)
    rep = MOLBSA.MOCandidate(bio_params, fx0)
    groups = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(param_dists, slots, BSP.BIOMASS_PER_ECO_GROUPS))   # block-diagonal MO-CMA-ES
    @info "MO-CMA-ES block-diagonal: $(length(groups)) covariance blocks (sizes $(length.(groups)))"
    search_state = MOCMAES.MOCMAESState(mean0, cmaes_sigma0, rep, rng; lambda=cmaes_lambda, blocks=groups, max_iter=typemax(Int), archive_cap=archive_cap)
    search_state.n_evals = 1
    # WARM-START the distribution from the top-K Sobol seeds instead of the single best point: mean =
    # log-weighted recombination of the top-K, C₀ = their (shrunk) empirical covariance shape. K=1 ⇒ no-op
    # (keeps the single-point mean + C=I already set above).
    if cmaes_warmstart_seeds > 1 && isnothing(start_from) && length(_sobol_cands) >= 2
      K = min(cmaes_warmstart_seeds, length(_sobol_cands))
      us = [PU.params_to_u(_sobol_cands[i], param_dists, slots) for i in 1:K]
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
    frozen_ref = isnothing(start_from) ? nothing : _load_params_from_path(start_from)   # frozen-growth reference
    slots = PU.build_slots(param_dists, bio_params)
    if force_restart_from_random
      bio_params = next_candidate()
      _apply_frozen_growth!(bio_params, frozen_ref)
      _seed_prob_estab!(bio_params, eco_list, species_list, eco_species_ids); _seed_maturity!(bio_params, species_list); _seed_min_rel!(bio_params, eco_list)
      MOCMAES.restart!(search_state, PU.params_to_u(bio_params, param_dists, slots), cmaes_sigma0)
    end
  end
  # Hansen-style mixed-integer handling (honors the flag each run, incl. on resume).
  search_state.u_min_std = integer_handling ? PU.integer_u_min_std(param_dists, slots, integer_std_factor) : Float64[]
  if TRIALS < 1 || search_state.n_evals >= TRIALS
    return search_state
  end

  writer_ch, writer_task = start_mo_writer(typeof(search_state), output_dir)

  losses_db_file = DuckDB.DB(joinpath(output_dir, "losses.duckdb"))
  losses_db = DuckDB.connect(losses_db_file)
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS total_loss (iteration INTEGER, n_sites INTEGER, n_obs INTEGER, total_loss DOUBLE, archive_size INTEGER, params_blob BLOB)")
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS ecoregion_loss (iteration INTEGER, ecoregion VARCHAR, eco_num_sites INTEGER, eco_num_obs INTEGER, ecoregion_total_loss DOUBLE)")
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS species_loss (iteration INTEGER, ecoregion VARCHAR, species VARCHAR, age_dist_loss DOUBLE, agb_loss DOUBLE)")

  # Per-generation trajectory CSV (best representative train + val loss), like the CMA-MAE driver.
  # On resume: APPEND (don't truncate generations 1..i from the prior run) and skip re-writing the header.
  _resuming = !isnothing(resume_from)
  metrics_io = open(joinpath(output_dir, "metrics.csv"), _resuming ? "a" : "w")
  _resuming || println(metrics_io, "iteration,best_train_loss,best_val_loss,pop_size,archive_size")
  # Seed rep_val_loss from the initial representative so every row carries a val loss until the next
  # new-best (otherwise it stays Inf → blank, because val is only recomputed at a new-best — and a run
  # that hasn't found a new-best yet would show an all-blank val column).
  rep_val_loss = Inf
  if have_val
    try
      _vr0 = only(fit_params(val_ref_soa, search_state.representative.x, max_sim_year, n_species,
        eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params;
        debug=false, search_tier=3, injection_dict=inj_dict_val, injection_years=inj_years_val, dual_b=val_dual_b, seeds=fixed_seeds))
      rep_val_loss = _mo_val_loss(_vr0, eco_species_ids)
    catch e; @warn "initial val-loss init failed" exception = (e, catch_backtrace()); end
  end
  # Track the best-VAL candidate separately from the best-TRAIN representative, and export its params on any
  # val improvement (best_val_params@N) — so the pre-overfit / best-generalizing candidate stays pullable and
  # comparable to the (overfitting) train-best. Seed the tracker + a @0 checkpoint from the initial rep.
  rep_val_best = rep_val_loss
  if have_val && isfinite(rep_val_best)
    try
      mkpath(output_dir)
      PU.save_json(joinpath(output_dir, "best_val_params@0.json"), search_state.representative.x)
      JLD2.save_object(joinpath(output_dir, "best_val_params@0.jld2"), search_state.representative.x)
    catch e; @warn "initial best_val save failed" exception = (e, catch_backtrace()); end
  end

  caused_by_interrupt(e) =
    e isa InterruptException ? true :
    e isa TaskFailedException ? any(en -> caused_by_interrupt(en.exception), Base.current_exceptions(e.task)) :
    e isa CompositeException ? any(caused_by_interrupt, e.exceptions) :
    false

  stagnation = 0
  evals_done = search_state.n_evals
  est_gens = max(1, cld(TRIALS - evals_done, search_state.lambda))
  try
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
        cand = PU.u_to_params(xs_u[k], param_dists, slots, bio_params)
        rep_results = _run(cand)
        fx_k, run_k, eco_k = _fitness(rep_results)
        cands[k] = cand
        fxs[k] = fx_k
        evals_done += 1
        if k == 1 || fx_k.aggregate < gen_best_agg   # k==1 guarantees gb_* are always defined
          gen_best_agg = fx_k.aggregate
          gb_idx = k
          gb_run = run_k
          gb_eco = eco_k
          gb_cached = _median_rep_cached(rep_results)
        end
      end

      MOCMAES.tell!(search_state, fxs, xs_u)
      search_state.n_evals = evals_done

      is_new_best = false
      archive_changed = false
      for k in 1:λ
        _upd = MOCMAES.update_archive!(search_state, MOLBSA.MOCandidate(cands[k], fxs[k]))
        is_new_best |= _upd.improved
        archive_changed |= _upd.changed
      end
      search_state.current = MOLBSA.MOCandidate(cands[gb_idx], fxs[gb_idx])
      stagnation = is_new_best ? 0 : stagnation + 1

      val_sim_sample = nothing
      if is_new_best
        iter = search_state.best_iteration
        total = search_state.representative.fx.aggregate
        @info "New best @ gen $iter | agg=$total | archive=$(length(search_state.archive)) | σ=$(search_state.sigma)"
        try
          test_df = simulate_and_test(; splots=splots, bio_params=search_state.representative.x, eco_list=eco_list, species_list=species_list, eco_species_ids=eco_species_ids, loss_params=loss_params, site_sim_years=site_sim_years, M=n_reps, no_establishment=no_establishment, rng=rng)
          println("Train stats:")
          show(test_df; allrows=true, allcols=true)
          println()
        catch e
          @warn "simulate_and_test (train) failed" exception = (e, catch_backtrace())
        end
        if have_val
          try
            # Score val the SAME way as train: n_reps mean-over-reps (fixed_seeds → _mor), NOT a single noisy
            # rep. A single rep sits above the 5-rep mean (Jensen + variance), which was inflating val and
            # masquerading as overfitting.
            val_result = only(fit_params(val_ref_soa, search_state.representative.x, max_sim_year, n_species,
              eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params;
              debug=false, search_tier=3, injection_dict=inj_dict_val, injection_years=inj_years_val, dual_b=val_dual_b, seeds=fixed_seeds))
            rep_val_loss = _mo_val_loss(val_result, eco_species_ids); @info "Val loss @ gen $iter | loss=$rep_val_loss"
            # Export on VAL improvement too (best_val_params@N), independent of the train-best export below.
            if rep_val_loss < rep_val_best
              rep_val_best = rep_val_loss
              try
                mkpath(output_dir)
                PU.save_json(joinpath(output_dir, "best_val_params@$(search_state.i).json"), search_state.representative.x)
                JLD2.save_object(joinpath(output_dir, "best_val_params@$(search_state.i).jld2"), search_state.representative.x)
                @info "New best VAL @ gen $(search_state.i) | val=$rep_val_loss  train=$(search_state.representative.fx.aggregate) → best_val_params@$(search_state.i)"
              catch e
                @warn "best_val checkpoint failed" exception = (e, catch_backtrace())
              end
            end
            val_sim_sample = n_output_plots > 0 ? _filter_cached_to_df(val_result[2], sampled_ids_val) : nothing
          catch e
            @warn "val fit_params failed" exception = (e, catch_backtrace())
          end
        end
        let buf = IOBuffer()
          Serialization.serialize(buf, search_state.representative.x)
          DuckDB.execute(losses_db, "INSERT INTO total_loss VALUES (?, ?, ?, ?, ?, ?)", [iter, gb_run.num_sites, gb_run.num_obs, total, length(search_state.archive), take!(buf)])
        end
        for (eco_id, eco_loss) in enumerate(gb_eco)
          eco_name = _eco_name(eco_list, eco_id)
          DuckDB.execute(losses_db, "INSERT INTO ecoregion_loss VALUES (?, ?, ?, ?, ?)", [iter, eco_name, eco_loss.num_sites, eco_loss.num_obs, convert(Float64, PU.get_total_loss(eco_loss))])
          n = max(1, eco_loss.num_sites)
          for gsp in 1:n_species
            eco_loss.sp_w_loss[gsp] == 0f0 && continue
            DuckDB.execute(losses_db, "INSERT INTO species_loss VALUES (?, ?, ?, ?, ?)", [iter, eco_name, species_list[gsp], eco_loss.sp_w_loss[gsp] / n, eco_loss.sp_agb_loss[gsp] / n])
          end
        end
      end
      println(metrics_io, string(search_state.i, ",", convert(Float64, search_state.representative.fx.aggregate), ",", isfinite(rep_val_loss) ? rep_val_loss : "", ",", search_state.lambda, ",", length(search_state.archive))); flush(metrics_io)
      cached_sites_state_df = DataFrame(gb_cached, [:plot_id, :sim_year, :species_id, :age, :agb])
      sim_sample = (is_new_best && n_output_plots > 0) ? _filter_cached_to_df(gb_cached, sampled_ids) : nothing
      put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state), splots, cached_sites_state_df, emp_sample, sim_sample, is_new_best ? emp_sample_val : nothing, val_sim_sample))
      # Checkpoint the archive on ANY change (adds OR swaps), not just representative improvements, so the
      # params of every archive state are pullable. is_new_best already saves via the writer above (improved
      # ⟹ changed), so only handle the changed-but-not-improved case here. Synchronous (search thread), but
      # small: state is ~13KB/member and archive changes become rare as the distribution converges.
      if archive_changed && !is_new_best
        try
          mkpath(output_dir)
          JLD2.save_object(joinpath(output_dir, "search_state@$(search_state.i).jld2"), search_state)
        catch e
          @warn "archive-change checkpoint failed" iter = search_state.i exception = (e, catch_backtrace())
        end
      end

      if ipop && (search_state.sigma < 1e-11 || stagnation >= ipop_stagnation)
        new_lambda = search_state.lambda * 2
        @info "IPOP restart @ gen $(search_state.i): λ $(search_state.lambda) → $new_lambda"
        bio_params = next_candidate()
        _apply_frozen_growth!(bio_params, frozen_ref)   # keep fix_growth frozen at the Sim-A seed across restarts
        _seed_prob_estab!(bio_params, eco_list, species_list, eco_species_ids)   # re-seed the calibration params
        _seed_maturity!(bio_params, species_list)
        _seed_min_rel!(bio_params, eco_list)   # keep MIN_REL pinned across IPOP restarts
        MOCMAES.restart!(search_state, PU.params_to_u(bio_params, param_dists, slots), cmaes_sigma0; lambda=new_lambda)
        stagnation = 0
      end
    end
  catch e
    if caused_by_interrupt(e)
      @info "Search interrupted by user @ gen $(search_state.i); finalizing checkpoint…"
    else
      rethrow()
    end
  finally
    stop_writer(writer_ch, writer_task)
    close(losses_db_file); close(metrics_io)
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

  return search_state
end

# Igel/Hansen/Roth (2007) population-based MO-CMA-ES driver. Same MO objective vector, archive and
# writer as parametrize_MOCMAES, but the engine is a population of μ (1+1)-CMA-ES individuals
# (IgelMOCMAES) rather than one distribution — better front spread/extreme coverage. μ candidates
# are evaluated per generation (serially); the initial population is a Sobol design.
function parametrize_IgelMOCMAES(; ref_soa::ActiveSoA, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, igel_mu::Int=20, igel_sigma0::Float64=0.3, igel_sobol_init::Bool=true, igel_niche_radius::Float64=0.0, igel_reseed_sigma::Float64=0.0, igel_maturity::Int=0, single_cov::Bool=false)
  splots.sim_year .= Dates.value.(Dates.Day.(splots.measdate - splots.start_measdate)) ./ 365.25 .|> round .|> Int
  all_plot_ids = UIntType.(unique(splots.plot_id))
  sampled_ids = _sample_plot_ids(all_plot_ids, n_output_plots, rng; injection_cohorts=injection_cohorts)
  emp_sample = n_output_plots > 0 ? _make_emp_df(splots, sampled_ids) : nothing
  n_species = length(species_list)
  max_sim_year = site_sim_years.sim_years .|> maximum |> maximum
  param_dists = BSP.make_biomass_param_dists(length(species_list), length(eco_list), eco_species_ids; no_establishment=no_establishment, fit_establishment=(DUAL_MODE[] != :off))
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  injection_dict = isnothing(injection_cohorts) ? nothing : _build_injection_dict(injection_cohorts, ref_soa)
  injection_years = isnothing(injection_cohorts) ? Set{Int}() : Set(Int.(injection_cohorts.sim_year))
  have_val = !isnothing(val_ref_soa)
  inj_dict_val = (have_val && !isnothing(val_injection_cohorts)) ? _build_injection_dict(val_injection_cohorts, val_ref_soa) : nothing
  inj_years_val = (have_val && !isnothing(val_injection_cohorts)) ? Set(Int.(val_injection_cohorts.sim_year)) : Set{Int}()
  sampled_ids_val = (have_val && n_output_plots > 0) ? _sample_plot_ids(UIntType.(unique(val_splots.plot_id)), n_output_plots, rng; injection_cohorts=val_injection_cohorts) : Set{UIntType}()
  emp_sample_val = (have_val && n_output_plots > 0) ? _make_emp_df(val_splots, sampled_ids_val) : nothing
  t4_ref = nothing; cycle_map = nothing; n_cycles = 0; t1_ref = nothing; t2_ref = nothing
  if search_tier == 1
    n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    t1_ref = [zeros(FloatType, length(eco_species_ids[eco_id]), n_bins) for eco_id in eachindex(eco_list)]
    for ((_, eco_id), year_dict) in spdf_plts, (_, spdf_gt) in year_dict, (sp_eco, rec) in spdf_gt.records
      t1_ref[eco_id][sp_eco, :] .+= diff([0f0; rec.sp_age_cdf]) .* rec.sp_agb_sum
    end
  elseif search_tier == 2
    n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    t2_ref_bins = [[FloatType[] for sp in 1:length(eco_species_ids[eco_id]), b in 1:n_bins] for eco_id in eachindex(eco_list)]
    t2_ref_total = [[FloatType[] for _ in 1:length(eco_species_ids[eco_id])] for eco_id in eachindex(eco_list)]
    for ((_, eco_id), year_dict) in spdf_plts, (_, spdf_gt) in year_dict, (sp_eco, rec) in spdf_gt.records
      bin_probs = diff([0f0; rec.sp_age_cdf])
      for b in 1:n_bins; push!(t2_ref_bins[eco_id][Int(sp_eco), b], bin_probs[b] * rec.sp_agb_sum); end
      push!(t2_ref_total[eco_id][Int(sp_eco)], rec.sp_agb_sum)
    end
    t2_ref = (bins=t2_ref_bins, total=t2_ref_total)
  elseif search_tier == 4 || search_tier == 5 || DUAL_MODE[] != :off
    n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    cycle_map, n_cycles = Data.build_cycle_map(splots; cycle_years=cycle_years)
    t4_ref = [[zeros(FloatType, length(eco_species_ids[eco_id]), n_bins) for _ in 1:n_cycles] for eco_id in eachindex(eco_list)]
    for ((plot_id, eco_id), year_dict) in spdf_plts, (sy, spdf_gt) in year_dict
      cyc = get(cycle_map, (Int(plot_id), Int(sy)), 0); cyc == 0 && continue
      for (sp_eco, rec) in spdf_gt.records; t4_ref[eco_id][cyc][sp_eco, :] .+= diff([0f0; rec.sp_age_cdf]) .* rec.sp_agb_sum; end
    end
  end
  fixed_seeds = _fixed_seeds(rng, n_reps)
  dual_b = DUAL_MODE[] == :off ? nothing : _build_dual_b(splots, eco_species_ids, eco_list, spdf_plts, loss_params, t4_ref, cycle_map, n_cycles, injection_cohorts, spinup_cohorts, rng, no_establishment; b_only=(DUAL_MODE[] == :b))
  _run(p) = fit_params(ref_soa, p, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, t4_ref, cycle_map, n_cycles, seeds=fixed_seeds, injection_dict=injection_dict, injection_years=injection_years, dual_b=dual_b)
  function _fitness(rep_results)
    run_result, eco_losses, _ = _agg_reps(rep_results)
    objs = _mo_objectives(eco_losses, eco_species_ids)
    MOLBSA.MOFitness(objs, _mo_aggregate(objs, run_result)), run_result, eco_losses
  end
  next_param() = !isempty(_sobol_cands) && length(_sobol_cands) >= 1 ? popfirst!(_sobol_cands) :
                 BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)

  if isnothing(resume_from)
    template = !isnothing(start_from) ? _load_params_from_path(start_from) : BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    slots = PU.build_slots(param_dists, template)
    # initial population: Sobol candidate DB if given, else a Sobol space-filling design (igel_sobol_init),
    # else random generated params. Sobol spreads the μ individuals across the space → better coverage.
    init_params = !isempty(_sobol_cands) ? [next_param() for _ in 1:igel_mu] :
                  igel_sobol_init ? PU.sobol_samples(param_dists, template, igel_mu) :
                  [BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment) for _ in 1:igel_mu]
    init_us = [PU.params_to_u(p, param_dists, slots) for p in init_params]
    init_cands = [MOLBSA.MOCandidate(init_params[k], _fitness(_run(init_params[k]))[1]) for k in 1:igel_mu]
    groups = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(param_dists, slots, BSP.BIOMASS_PER_ECO_GROUPS))   # block-diagonal per-individual (1+1)-CMA
    @info "Igel MO-CMA-ES block-diagonal: $(length(groups)) covariance blocks per individual (sizes $(length.(groups)))"
    search_state = IgelMOCMAES.IgelState(init_us, init_cands, rng; sigma0=igel_sigma0, archive_cap=archive_cap, max_iter=typemax(Int), niche_radius=igel_niche_radius, reseed_sigma=igel_reseed_sigma, maturity_period=igel_maturity, blocks=groups)
    search_state.n_evals = igel_mu
    bio_params = template
  else
    @info "Resuming from $resume_from"
    search_state = JLD2.load_object(resume_from)
    bio_params = search_state.representative.x
    slots = PU.build_slots(param_dists, bio_params)
  end
  if TRIALS < 1 || search_state.n_evals >= TRIALS
    return search_state
  end

  writer_ch, writer_task = start_mo_writer(typeof(search_state), output_dir)
  losses_db_file = DuckDB.DB(joinpath(output_dir, "losses.duckdb"))
  losses_db = DuckDB.connect(losses_db_file)
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS total_loss (iteration INTEGER, n_sites INTEGER, n_obs INTEGER, total_loss DOUBLE, archive_size INTEGER, params_blob BLOB)")
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS ecoregion_loss (iteration INTEGER, ecoregion VARCHAR, eco_num_sites INTEGER, eco_num_obs INTEGER, ecoregion_total_loss DOUBLE)")
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS species_loss (iteration INTEGER, ecoregion VARCHAR, species VARCHAR, age_dist_loss DOUBLE, agb_loss DOUBLE)")
  caused_by_interrupt(e) = e isa InterruptException ? true :
    e isa TaskFailedException ? any(en -> caused_by_interrupt(en.exception), Base.current_exceptions(e.task)) :
    e isa CompositeException ? any(caused_by_interrupt, e.exceptions) : false

  evals_done = search_state.n_evals
  est_gens = max(1, cld(TRIALS - evals_done, igel_mu))
  try
    TProgress.@track for _gen in 1:est_gens
      evals_done >= TRIALS && break
      offs = IgelMOCMAES.ask(search_state)
      off_fxs = Vector{MOLBSA.MOFitness}(undef, igel_mu)
      off_params = Vector{typeof(bio_params)}(undef, igel_mu)
      gen_best_agg = Inf; local gb_run, gb_eco, gb_cached, gb_idx
      for k in 1:igel_mu                               # SERIAL (fit_params threads internally)
        p = PU.u_to_params(offs[k], param_dists, slots, bio_params)
        rep_results = _run(p); fx_k, run_k, eco_k = _fitness(rep_results)
        off_params[k] = p; off_fxs[k] = fx_k; evals_done += 1
        if fx_k.aggregate < gen_best_agg
          gen_best_agg = fx_k.aggregate; gb_idx = k; gb_run = run_k; gb_eco = eco_k; gb_cached = _median_rep_cached(rep_results)
        end
      end
      is_new_best = IgelMOCMAES.tell!(search_state, off_fxs, off_params)
      search_state.n_evals = evals_done
      val_sim_sample = nothing
      if is_new_best
        iter = search_state.best_iteration; total = search_state.representative.fx.aggregate
        @info "New best @ gen $iter | agg=$total | archive=$(length(search_state.archive))"
        try
          test_df = simulate_and_test(; splots=splots, bio_params=search_state.representative.x, eco_list=eco_list, species_list=species_list, eco_species_ids=eco_species_ids, loss_params=loss_params, site_sim_years=site_sim_years, M=n_reps, no_establishment=no_establishment, rng=rng)
          println("Train stats:"); show(test_df; allrows=true, allcols=true); println()
        catch e; @warn "simulate_and_test (train) failed" exception = (e, catch_backtrace()); end
        if have_val
          try
            val_result = only(fit_params(val_ref_soa, search_state.representative.x, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params; debug=false, search_tier=3, injection_dict=inj_dict_val, injection_years=inj_years_val, dual_b=val_dual_b, seeds=[rand(rng, UInt64)]))
            @info "Val loss @ gen $iter | loss=$(_mo_val_loss(val_result, eco_species_ids))"
            val_sim_sample = n_output_plots > 0 ? _filter_cached_to_df(val_result[2], sampled_ids_val) : nothing
          catch e; @warn "val fit_params failed" exception = (e, catch_backtrace()); end
        end
        let buf = IOBuffer()
          Serialization.serialize(buf, search_state.representative.x)
          DuckDB.execute(losses_db, "INSERT INTO total_loss VALUES (?, ?, ?, ?, ?, ?)", [iter, gb_run.num_sites, gb_run.num_obs, total, length(search_state.archive), take!(buf)])
        end
        for (eco_id, eco_loss) in enumerate(gb_eco)
          eco_name = _eco_name(eco_list, eco_id)
          DuckDB.execute(losses_db, "INSERT INTO ecoregion_loss VALUES (?, ?, ?, ?, ?)", [iter, eco_name, eco_loss.num_sites, eco_loss.num_obs, convert(Float64, PU.get_total_loss(eco_loss))])
          n = max(1, eco_loss.num_sites)
          for gsp in 1:n_species
            eco_loss.sp_w_loss[gsp] == 0f0 && continue
            DuckDB.execute(losses_db, "INSERT INTO species_loss VALUES (?, ?, ?, ?, ?)", [iter, eco_name, species_list[gsp], eco_loss.sp_w_loss[gsp] / n, eco_loss.sp_agb_loss[gsp] / n])
          end
        end
      end
      cached_sites_state_df = DataFrame(gb_cached, [:plot_id, :sim_year, :species_id, :age, :agb])
      sim_sample = (is_new_best && n_output_plots > 0) ? _filter_cached_to_df(gb_cached, sampled_ids) : nothing
      put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state), splots, cached_sites_state_df, emp_sample, sim_sample, is_new_best ? emp_sample_val : nothing, val_sim_sample))
    end
  catch e
    caused_by_interrupt(e) ? @info("Search interrupted by user @ gen $(search_state.i); finalizing checkpoint…") : rethrow()
  finally
    stop_writer(writer_ch, writer_task); close(losses_db_file)
    try
      mkpath(output_dir); fname = "search_state@$(search_state.i).jld2"
      JLD2.save_object(joinpath(output_dir, fname), search_state)
      link_path = joinpath(output_dir, "search_state_latest.jld2"); islink(link_path) && rm(link_path); symlink(fname, link_path)
      @info "Search state saved @ $(search_state.i)"
    catch e; @error "Failed to save search state on exit" exception = (e, catch_backtrace()); end
  end
  return search_state
end

# CMA-MAE driver (Fontaine & Nikolaidis 2022): one archive-driven CMA-ES emitter over a 2-D MAP-Elites
# grid. Behaviour descriptor = (Σ Wasserstein loss, Σ AGB loss) — the two halves of the MO objective
# vector; quality = the aggregate loss. The archive keeps one elite parameter set per behaviour cell
# (a quality-diversity set of fits), the representative is the lowest-aggregate elite. Same MO objective
# vector, writer, archive logging and losses.duckdb schema as parametrize_MOCMAES; the emitter re-seeds
# itself on convergence/stagnation (no external IPOP). Grid bounds are auto-scaled from the seed loss.
function parametrize_CMAMAE(; ref_soa::ActiveSoA, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, cmaes_lambda::Union{Nothing,Int}=nothing, cmaes_sigma0::Float64=0.3, cmame_alpha::Float64=0.02, cmame_grid::Int=15, cmame_reseed_explore::Float64=1.0, cmame_restart_patience::Int=6, cmame_sobol_reseed::Bool=false, balanced_quality::Bool=false, cmame_mo_rank::Bool=false, archive_by_sp::Bool=false, bounds_by_sobol::Bool=false, top_seeds::Float64=0.2, integer_handling::Bool=false, integer_std_factor::Float64=0.3, single_cov::Bool=false)
  splots.sim_year .= Dates.value.(Dates.Day.(splots.measdate - splots.start_measdate)) ./ 365.25 .|> round .|> Int
  all_plot_ids = UIntType.(unique(splots.plot_id))
  sampled_ids = _sample_plot_ids(all_plot_ids, n_output_plots, rng; injection_cohorts=injection_cohorts)
  emp_sample = n_output_plots > 0 ? _make_emp_df(splots, sampled_ids) : nothing
  n_species = length(species_list)
  max_sim_year = site_sim_years.sim_years .|> maximum |> maximum
  param_dists = BSP.make_biomass_param_dists(length(species_list), length(eco_list), eco_species_ids; no_establishment=no_establishment, fit_establishment=(DUAL_MODE[] != :off))
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  injection_dict = isnothing(injection_cohorts) ? nothing : _build_injection_dict(injection_cohorts, ref_soa)
  injection_years = isnothing(injection_cohorts) ? Set{Int}() : Set(Int.(injection_cohorts.sim_year))
  have_val = !isnothing(val_ref_soa)
  inj_dict_val = (have_val && !isnothing(val_injection_cohorts)) ? _build_injection_dict(val_injection_cohorts, val_ref_soa) : nothing
  inj_years_val = (have_val && !isnothing(val_injection_cohorts)) ? Set(Int.(val_injection_cohorts.sim_year)) : Set{Int}()
  sampled_ids_val = (have_val && n_output_plots > 0) ? _sample_plot_ids(UIntType.(unique(val_splots.plot_id)), n_output_plots, rng; injection_cohorts=val_injection_cohorts) : Set{UIntType}()
  emp_sample_val = (have_val && n_output_plots > 0) ? _make_emp_df(val_splots, sampled_ids_val) : nothing
  t4_ref = nothing; cycle_map = nothing; n_cycles = 0; t1_ref = nothing; t2_ref = nothing
  if search_tier == 1
    n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    t1_ref = [zeros(FloatType, length(eco_species_ids[eco_id]), n_bins) for eco_id in eachindex(eco_list)]
    for ((_, eco_id), year_dict) in spdf_plts, (_, spdf_gt) in year_dict, (sp_eco, rec) in spdf_gt.records
      t1_ref[eco_id][sp_eco, :] .+= diff([0f0; rec.sp_age_cdf]) .* rec.sp_agb_sum
    end
  elseif search_tier == 2
    n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    t2_ref_bins = [[FloatType[] for sp in 1:length(eco_species_ids[eco_id]), b in 1:n_bins] for eco_id in eachindex(eco_list)]
    t2_ref_total = [[FloatType[] for _ in 1:length(eco_species_ids[eco_id])] for eco_id in eachindex(eco_list)]
    for ((_, eco_id), year_dict) in spdf_plts, (_, spdf_gt) in year_dict, (sp_eco, rec) in spdf_gt.records
      bin_probs = diff([0f0; rec.sp_age_cdf])
      for b in 1:n_bins; push!(t2_ref_bins[eco_id][Int(sp_eco), b], bin_probs[b] * rec.sp_agb_sum); end
      push!(t2_ref_total[eco_id][Int(sp_eco)], rec.sp_agb_sum)
    end
    t2_ref = (bins=t2_ref_bins, total=t2_ref_total)
  elseif search_tier == 4 || search_tier == 5 || DUAL_MODE[] != :off
    n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    cycle_map, n_cycles = Data.build_cycle_map(splots; cycle_years=cycle_years)
    t4_ref = [[zeros(FloatType, length(eco_species_ids[eco_id]), n_bins) for _ in 1:n_cycles] for eco_id in eachindex(eco_list)]
    for ((plot_id, eco_id), year_dict) in spdf_plts, (sy, spdf_gt) in year_dict
      cyc = get(cycle_map, (Int(plot_id), Int(sy)), 0); cyc == 0 && continue
      for (sp_eco, rec) in spdf_gt.records; t4_ref[eco_id][cyc][sp_eco, :] .+= diff([0f0; rec.sp_age_cdf]) .* rec.sp_agb_sum; end
    end
  end
  fixed_seeds = _fixed_seeds(rng, n_reps)
  dual_b = DUAL_MODE[] == :off ? nothing : _build_dual_b(splots, eco_species_ids, eco_list, spdf_plts, loss_params, t4_ref, cycle_map, n_cycles, injection_cohorts, spinup_cohorts, rng, no_establishment; b_only=(DUAL_MODE[] == :b))
  _run(p) = fit_params(ref_soa, p, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, t4_ref, cycle_map, n_cycles, seeds=fixed_seeds, injection_dict=injection_dict, injection_years=injection_years, dual_b=dual_b)
  function _fitness(rep_results)
    run_result, eco_losses, _ = _agg_reps(rep_results)
    objs = _mo_objectives(eco_losses, eco_species_ids)
    MOLBSA.MOFitness(objs, _mo_aggregate(objs, run_result)), run_result, eco_losses
  end
  sobol_idx = Ref(1)
  next_candidate() =
    if sobol_idx[] <= length(_sobol_cands)
      p = _sobol_cands[sobol_idx[]]; sobol_idx[] += 1; p
    else
      BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    end

  # ---- per-species-breadth archive (archive_by_sp) + Sobol-calibrated (ΣW,ΣAGB) bounds (bounds_by_sobol) ----
  # 3rd grid axis = breadth LEVEL 1..n_sp: a candidate non-dominated on b species fills its (W,AGB) column
  # up to level b (each cell keeps min-aggregate). breadth = #species whose running (W_s,AGB_s) Pareto
  # front the candidate would join. A broad+low-aggregate candidate wins many levels.
  n_sp = length(species_list)
  _sp_wa(eco_losses) = begin                          # per-global-species (ΣW_s, ΣAGB_s) over all eco blocks
    W = zeros(Float64, n_sp); A = zeros(Float64, n_sp)
    @inbounds for el in eco_losses, s in 1:n_sp
      W[s] += Float64(el.sp_w_loss[s]); A[s] += Float64(el.sp_agb_loss[s])
    end
    (W, A)
  end
  _nd_on(front, w, a) = !any(p -> p[1] <= w && p[2] <= a && (p[1] < w || p[2] < a), front)
  _ins_front!(front, w, a) = (_nd_on(front, w, a) && (filter!(p -> !(w <= p[1] && a <= p[2] && (w < p[1] || a < p[2])), front); push!(front, (w, a))); nothing)
  inc_sp = isempty(DOMINANCE_GSP[]) ? collect(1:n_sp) : sort(collect(DOMINANCE_GSP[]))   # dominance species (excl _H/_S etc.)
  n_inc = length(inc_sp)
  sp_fronts = [Tuple{Float64,Float64}[] for _ in 1:n_inc]   # one running (W_s,A_s) Pareto front per dominance species
  gwlo = 0.0; gwhi = 1.0; galo = 0.0; gahi = 1.0; _calibrated = false
  if (bounds_by_sobol || archive_by_sp) && !isnothing(sobol_candidates_db)
    cal = load_sobol_candidates(sobol_candidates_db; top_frac=top_seeds)
    if !isempty(cal)
      @info "CMA-MAE: calibrating (ΣW,ΣAGB) bounds from $(length(cal)) top-$(round(Int,100*top_seeds))% Sobol seeds…"
      Wg = Float64[]; Ag = Float64[]
      for cc in cal
        fxc, _, _ = _fitness(_run(cc)); mc = CMAMAE.measure(fxc)
        push!(Wg, Float64(mc[1])); push!(Ag, Float64(mc[2]))
      end
      gwlo, gwhi = minimum(Wg), max(maximum(Wg), minimum(Wg) + 1e-9)
      galo, gahi = minimum(Ag), max(maximum(Ag), minimum(Ag) + 1e-9)
      _calibrated = true
      @info "  bounds ΣW∈[$(round(gwlo,sigdigits=3)),$(round(gwhi,sigdigits=3))] ΣAGB∈[$(round(galo,sigdigits=3)),$(round(gahi,sigdigits=3))]"
    end
  end

  if isnothing(resume_from)
    bio_params = if !isnothing(start_from)
      @info "Seeding initial candidate from $start_from (fresh search_state)"
      p = _load_params_from_path(start_from)
      (p.SPECIES_LIST == species_list && p.ECO_LIST == eco_list) ||
        error("start_from params are incompatible with this run: their SPECIES_LIST/ECO_LIST differ from the loaded data.")
      p
    else
      next_candidate()
    end
    slots = PU.build_slots(param_dists, bio_params)
    fx0, _, _ = _fitness(_run(bio_params))
    mean0 = PU.params_to_u(bio_params, param_dists, slots)
    rep = MOLBSA.MOCandidate(bio_params, fx0)
    # auto-scale the 2-D measure grid from the seed candidate's loss (worse than any fit we'll keep),
    # so the archive cells bracket the reachable (Wasserstein, AGB) range; quality ceiling t0 above it.
    m0 = CMAMAE.measure(fx0)
    if !_calibrated                                   # no Sobol bounds → fall back to 2× the seed's measure
      gwlo, galo = 0.0, 0.0
      gwhi = max(2.0 * Float64(m0[1]), 1e-9); gahi = max(2.0 * Float64(m0[2]), 1e-9)
    end
    t0 = 1.5 * convert(Float64, fx0.aggregate)
    groups = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(param_dists, slots, BSP.BIOMASS_PER_ECO_GROUPS))   # block-diagonal CMA-MAE emitter
    @info "CMA-MAE block-diagonal: $(length(groups)) covariance blocks (sizes $(length.(groups)))"
    if archive_by_sp
      grid_dims_v = [cmame_grid, cmame_grid, n_inc]   # (ΣW, ΣAGB, breadth-level 1..n_inc over dominance species)
      meas_lo_v = [0.0, 0.0, 0.0]; meas_hi_v = [1.0, 1.0, 1.0]   # position normalized in the tell loop; level∈[0,1)
      @info "CMA-MAE per-species-breadth archive: $(cmame_grid)×$(cmame_grid)×$(n_inc) (ΣW,ΣAGB,breadth over $(n_inc) dominance species); ideal cell (1,1,$(n_inc))"
    else
      grid_dims_v = [cmame_grid, cmame_grid]
      meas_lo_v = [gwlo, galo]; meas_hi_v = [gwhi, gahi]
      @info "CMA-MAE grid $(cmame_grid)×$(cmame_grid) over ΣW∈[$(round(gwlo,sigdigits=3)),$(round(gwhi,sigdigits=3))] ΣAGB∈[$(round(galo,sigdigits=3)),$(round(gahi,sigdigits=3))], λ=auto, α=$(cmame_alpha)"
    end
    search_state = CMAMAE.CMAMAEMOState(mean0, cmaes_sigma0, rep, rng; meas_lo=meas_lo_v, meas_hi=meas_hi_v,
      grid_dims=grid_dims_v, lambda=cmaes_lambda, alpha=cmame_alpha, t0=t0, reseed_explore=cmame_reseed_explore,
      restart_patience=cmame_restart_patience, sobol_reseed=cmame_sobol_reseed, mo_rank=cmame_mo_rank, balanced=balanced_quality, blocks=groups, max_iter=typemax(Int))
    search_state.n_evals = 1
  else
    @info "Resuming from $resume_from"
    search_state = JLD2.load_object(resume_from)
    bio_params = search_state.representative.x
    slots = PU.build_slots(param_dists, bio_params)
  end
  search_state.engine.emitter.u_min_std = integer_handling ? PU.integer_u_min_std(param_dists, slots, integer_std_factor) : Float64[]   # Hansen integer handling on the CMA-MAE emitter
  if TRIALS < 1 || search_state.n_evals >= TRIALS
    return search_state
  end

  writer_ch, writer_task = start_mo_writer(typeof(search_state), output_dir)
  losses_db_file = DuckDB.DB(joinpath(output_dir, "losses.duckdb"))
  losses_db = DuckDB.connect(losses_db_file)
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS total_loss (iteration INTEGER, n_sites INTEGER, n_obs INTEGER, total_loss DOUBLE, archive_size INTEGER, params_blob BLOB)")
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS ecoregion_loss (iteration INTEGER, ecoregion VARCHAR, eco_num_sites INTEGER, eco_num_obs INTEGER, ecoregion_total_loss DOUBLE)")
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS species_loss (iteration INTEGER, ecoregion VARCHAR, species VARCHAR, age_dist_loss DOUBLE, agb_loss DOUBLE)")
  # Per-generation trajectory: best (representative) train+val loss, emitter population size, archive size.
  metrics_io = open(joinpath(output_dir, "metrics.csv"), "w")
  println(metrics_io, "iteration,best_train_loss,best_val_loss,pop_size,archive_size")
  # Validation mirrors the TRAINING tier so the held-out loss ranks candidates by the SAME objective
  # (tier 5 ⇒ tier-5 val). Tier 4/5 need per-cycle bins built from the validation split.
  val_t4_ref = nothing; val_cycle_map = nothing; val_n_cycles = 0
  if have_val && (search_tier == 4 || search_tier == 5)
    _vnb = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    val_cycle_map, val_n_cycles = Data.build_cycle_map(val_splots; cycle_years=cycle_years)
    val_t4_ref = [[zeros(FloatType, length(eco_species_ids[eco_id]), _vnb) for _ in 1:val_n_cycles] for eco_id in eachindex(eco_list)]
    for ((plot_id, eco_id), year_dict) in val_spdf_plts, (sy, spdf_gt) in year_dict
      cyc = get(val_cycle_map, (Int(plot_id), Int(sy)), 0); cyc == 0 && continue
      for (sp_eco, rec) in spdf_gt.records; val_t4_ref[eco_id][cyc][sp_eco, :] .+= diff([0f0; rec.sp_age_cdf]) .* rec.sp_agb_sum; end
    end
  end
  # Validation Sim B so held-out loss measures the SAME A⊕B objective as training (dual modes only).
  val_dual_b = have_val ? _build_val_dual_b(val_splots, eco_species_ids, eco_list, val_spdf_plts, loss_params, cycle_years, val_injection_cohorts, val_spinup_cohorts, rng, no_establishment) : nothing
  rep_val_loss = Inf
  if have_val
    try
      vr0 = only(fit_params(val_ref_soa, search_state.representative.x, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params; debug=false, search_tier=search_tier, t4_ref=val_t4_ref, cycle_map=val_cycle_map, n_cycles=val_n_cycles, injection_dict=inj_dict_val, injection_years=inj_years_val, dual_b=val_dual_b, seeds=[rand(rng, UInt64)]))
      rep_val_loss = _mo_val_loss(vr0, eco_species_ids)
    catch; end
  end
  caused_by_interrupt(e) = e isa InterruptException ? true :
    e isa TaskFailedException ? any(en -> caused_by_interrupt(en.exception), Base.current_exceptions(e.task)) :
    e isa CompositeException ? any(caused_by_interrupt, e.exceptions) : false

  evals_done = search_state.n_evals
  est_gens = max(1, cld(TRIALS - evals_done, search_state.engine.emitter.lambda))
  try
    TProgress.@track for _gen in 1:est_gens
      evals_done >= TRIALS && break
      xs_u = CMAMAE.ask(search_state)
      λ = search_state.engine.emitter.lambda
      fxs = Vector{MOLBSA.MOFitness}(undef, λ)
      cands = Vector{MOLBSA.MOCandidate{typeof(bio_params)}}(undef, λ)
      meas = Vector{Tuple{Float64,Float64}}(undef, λ)
      sp_meas = Vector{Vector{Vector{Float64}}}(undef, λ)   # archive_by_sp: per-candidate breadth-column cells
      sp_qual = Vector{Vector{Float64}}(undef, λ)
      gen_best_agg = Inf
      local gb_run, gb_eco, gb_cached, gb_idx
      for k in 1:λ                                    # SERIAL (fit_params threads internally over sites)
        cand = PU.u_to_params(xs_u[k], param_dists, slots, bio_params)
        rep_results = _run(cand)
        fx_k, run_k, eco_k = _fitness(rep_results)
        cands[k] = MOLBSA.MOCandidate(cand, fx_k)
        fxs[k] = fx_k
        evals_done += 1
        if archive_by_sp
          Ws, As = _sp_wa(eco_k)
          b = 0
          @inbounds for (p, s) in enumerate(inc_sp); _nd_on(sp_fronts[p], Ws[s], As[s]) && (b += 1); end   # breadth vs running fronts
          @inbounds for (p, s) in enumerate(inc_sp); _ins_front!(sp_fronts[p], Ws[s], As[s]); end          # then update the fronts
          gW = sum(Ws[s] for s in inc_sp); gA = sum(As[s] for s in inc_sp)   # position over the dominance species (matches measure())
          wn = clamp((gW - gwlo) / (gwhi - gwlo), 0.0, 1 - 1e-9); an = clamp((gA - galo) / (gahi - galo), 0.0, 1 - 1e-9)
          # cell quality: balanced (rescaled ΣW+ΣAGB ≡ wn+an, scale-fair) OR raw aggregate (legacy)
          aggq = balanced_quality ? (wn + an) : Float64(fx_k.aggregate)
          sp_meas[k] = [[wn, an, (j - 0.5) / n_inc] for j in 1:b]  # fill breadth-levels 1..b at this (W,AGB) cell
          sp_qual[k] = fill(aggq, b)                               # quality at every filled level
        else
          m = CMAMAE.measure(fx_k); meas[k] = (Float64(m[1]), Float64(m[2]))
        end
        if fx_k.aggregate < gen_best_agg
          gen_best_agg = fx_k.aggregate; gb_idx = k; gb_run = run_k; gb_eco = eco_k; gb_cached = _median_rep_cached(rep_results)
        end
      end

      is_new_best = archive_by_sp ? CMAMAE.tell_species_mo!(search_state, cands, sp_meas, sp_qual, xs_u) :
                    CMAMAE.tell_mo!(search_state, cands, meas, xs_u)
      search_state.n_evals = evals_done
      search_state.current = cands[gb_idx]

      val_sim_sample = nothing
      if is_new_best
        iter = search_state.best_iteration
        total = search_state.representative.fx.aggregate
        @info "New best @ gen $iter | agg=$total | archive=$(length(search_state.archive)) | re-seeds=$(search_state.engine.n_restarts)"
        try
          test_df = simulate_and_test(; splots=splots, bio_params=search_state.representative.x, eco_list=eco_list, species_list=species_list, eco_species_ids=eco_species_ids, loss_params=loss_params, site_sim_years=site_sim_years, M=n_reps, no_establishment=no_establishment, rng=rng)
          println("Train stats:"); show(test_df; allrows=true, allcols=true); println()
        catch e; @warn "simulate_and_test (train) failed" exception = (e, catch_backtrace()); end
        if have_val
          try
            val_result = only(fit_params(val_ref_soa, search_state.representative.x, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params; debug=false, search_tier=search_tier, t4_ref=val_t4_ref, cycle_map=val_cycle_map, n_cycles=val_n_cycles, injection_dict=inj_dict_val, injection_years=inj_years_val, dual_b=val_dual_b, seeds=[rand(rng, UInt64)]))
            rep_val_loss = _mo_val_loss(val_result, eco_species_ids)
            @info "Val loss @ gen $iter | loss=$rep_val_loss"
            val_sim_sample = n_output_plots > 0 ? _filter_cached_to_df(val_result[2], sampled_ids_val) : nothing
          catch e; @warn "val fit_params failed" exception = (e, catch_backtrace()); end
        end
        let buf = IOBuffer()
          Serialization.serialize(buf, search_state.representative.x)
          DuckDB.execute(losses_db, "INSERT INTO total_loss VALUES (?, ?, ?, ?, ?, ?)", [iter, gb_run.num_sites, gb_run.num_obs, total, length(search_state.archive), take!(buf)])
        end
        for (eco_id, eco_loss) in enumerate(gb_eco)
          eco_name = _eco_name(eco_list, eco_id)
          DuckDB.execute(losses_db, "INSERT INTO ecoregion_loss VALUES (?, ?, ?, ?, ?)", [iter, eco_name, eco_loss.num_sites, eco_loss.num_obs, convert(Float64, PU.get_total_loss(eco_loss))])
          n = max(1, eco_loss.num_sites)
          for gsp in 1:n_species
            eco_loss.sp_w_loss[gsp] == 0f0 && continue
            DuckDB.execute(losses_db, "INSERT INTO species_loss VALUES (?, ?, ?, ?, ?)", [iter, eco_name, species_list[gsp], eco_loss.sp_w_loss[gsp] / n, eco_loss.sp_agb_loss[gsp] / n])
          end
        end
      end
      cached_sites_state_df = DataFrame(gb_cached, [:plot_id, :sim_year, :species_id, :age, :agb])
      sim_sample = (is_new_best && n_output_plots > 0) ? _filter_cached_to_df(gb_cached, sampled_ids) : nothing
      put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state), splots, cached_sites_state_df, emp_sample, sim_sample, is_new_best ? emp_sample_val : nothing, val_sim_sample))
      println(metrics_io, "$(search_state.i),$(Float64(search_state.representative.fx.aggregate)),$(isfinite(rep_val_loss) ? rep_val_loss : ""),$(search_state.engine.emitter.lambda),$(length(search_state.archive))")
      flush(metrics_io)
    end
  catch e
    caused_by_interrupt(e) ? @info("Search interrupted by user @ gen $(search_state.i); finalizing checkpoint…") : rethrow()
  finally
    try; close(metrics_io); catch; end
    stop_writer(writer_ch, writer_task); close(losses_db_file)
    try
      mkpath(output_dir); fname = "search_state@$(search_state.i).jld2"
      JLD2.save_object(joinpath(output_dir, fname), search_state)
      link_path = joinpath(output_dir, "search_state_latest.jld2"); islink(link_path) && rm(link_path); symlink(fname, link_path)
      @info "Search state saved @ $(search_state.i)"
    catch e; @error "Failed to save search state on exit" exception = (e, catch_backtrace()); end
    # Re-evaluate every archive elite on the held-out validation set so each carries train+val loss
    # (aligned to the saved archive order). train_loss is the elite's stored aggregate.
    if have_val
      try
        open(joinpath(output_dir, "archive_eval.csv"), "w") do io
          println(io, "candidate,train_loss,val_loss")
          for (ci, m) in enumerate(search_state.archive)
            vr = only(fit_params(val_ref_soa, m.x, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params; debug=false, search_tier=search_tier, t4_ref=val_t4_ref, cycle_map=val_cycle_map, n_cycles=val_n_cycles, injection_dict=inj_dict_val, injection_years=inj_years_val, dual_b=val_dual_b, seeds=[rand(rng, UInt64)]))
            println(io, "$ci,$(Float64(m.fx.aggregate)),$(_mo_val_loss(vr, eco_species_ids))")
          end
        end
        @info "Wrote archive_eval.csv (train+val loss per archive elite)"
      catch e; @warn "archive validation eval failed" exception = (e, catch_backtrace()); end
    end
  end
  return search_state
end

# Flatten the per-ecoregion SiteLoss breakdown into the multi-objective vector:
# for each (ecoregion, species-in-eco) two coordinates — the age-distribution
# (Wasserstein) loss and the AGB-level loss. Order is fixed across candidates, so
# the vector is coordinate-comparable in MOLBSA.mo_delta / dominance.
# Eco name for a per-eco loss index. Tier 5 returns 2·n_eco losses (tier-3 block then tier-4 block);
# indices past n_eco name the tier-4 copy with a suffix so per-eco logs stay unique.
function _eco_name(eco_list, idx::Integer)   # per-eco loss vectors can be k·n_eco long (tier-5: 2×; dual: 3×)
  ne = length(eco_list); blk = (idx - 1) ÷ ne; base = eco_list[(idx - 1) % ne + 1]
  blk == 0 ? base : base * "·blk$blk"
end

# Global-species mask for DOMINANCE/diversity: which global species enter the MO objective vector (and
# thus Pareto dominance in MO-CMA-ES/MOLBSA/Igel, the CMA-MAE measure descriptor, and the per-species
# breadth axis). Empty Set = all species. Set by parametrize() from the `dominance_species` flag
# (exact = SPCD only · exact+grp = + _GRP groups · all = incl. _H/_S catch-alls). The scalar aggregate
# (get_total_loss) is unaffected — _H/_S still count toward overall fit, just not toward dominance.
const DOMINANCE_GSP = Ref{Set{Int}}(Set{Int}())
@inline _dom_included(gsp::Int) = isempty(DOMINANCE_GSP[]) || gsp in DOMINANCE_GSP[]
# Decide a species name's membership for a dominance_species mode.
_dom_include_name(name::AbstractString, mode::AbstractString) =
  mode == "all" ? true :
  startswith(name, "_GRP") ? (mode == "exact+grp") :
  startswith(name, "_") ? false :          # _H, _S (and other "_" catch-alls) — excluded unless "all"
  true                                      # SPCD exact species

# Safe per-cell table lookup: 0 when the matrix isn't sized for this cell (e.g. an empty A table under a
# B-only dual) → the caller then falls back to the raw (un-normalized) term instead of indexing out of range.
@inline _cell(M::Matrix{FloatType}, gsp::Int, e::Int)::FloatType =
  (size(M, 1) >= gsp && size(M, 2) >= e) ? @inbounds(M[gsp, e]) : zero(FloatType)

function _mo_objectives(eco_losses::Vector{PU.SiteLoss}, eco_species_ids::Vector{Vector{Int}})::Vector{FloatType}
  # 4 GROUPED objectives for the dual: [ΣA_W, ΣA_AGB, ΣB_W, ΣB_AGB] — sum each over its eco-blocks and
  # dominance species. eco_losses = [A's n_eco blocks ; B's n_eco blocks] (2·n_eco ⇒ ngroups=2 ⇒ 4 obj);
  # a single sim (n_eco blocks) ⇒ ngroups=1 ⇒ 2 obj [ΣW, ΣAGB]. Keeping the MO vector low-dimensional
  # makes A↔B and W↔AGB first-class objectives (clean 4-D Pareto / net-win) instead of diluting them
  # across ~n_ess per-species entries. Species breadth is handled POST-HOC (the sweep), not here.
  ne = length(eco_species_ids)
  ngroups = max(1, length(eco_losses) ÷ ne)
  objs = zeros(FloatType, 2 * ngroups)
  cell = PU.CELL_NORM[]
  for (idx, el) in enumerate(eco_losses)
    g = (idx - 1) ÷ ne                       # 0-based group: 0 = Sim A (tier-3), 1 = Sim B (tier-4)
    e = (idx - 1) % ne + 1
    if cell
      # Per-cell: divide each (eco,lu,sp) cell by its OWN reference scale, apply the power, rescale by
      # the AGB-rank weight, then sum into the group. A and B keep separate scales; RANKW is shared.
      Ws = g == 0 ? PU.W_SCALE_A[] : PU.W_SCALE_B[]
      As = g == 0 ? PU.AGB_SCALE_A[] : PU.AGB_SCALE_B[]
      Rw = PU.RANKW[]
      @inbounds for gsp in eco_species_ids[e]
        _dom_included(gsp) || continue
        rw = _cell(Rw, gsp, e); ws = _cell(Ws, gsp, e); as = _cell(As, gsp, e)
        objs[2 * g + 1] += rw * PU._w_pow(ws > 0 ? el.sp_w_loss[gsp] / ws : el.sp_w_loss[gsp])
        objs[2 * g + 2] += rw * PU._agb_pow(as > 0 ? el.sp_agb_loss[gsp] / as : el.sp_agb_loss[gsp])
      end
    else
      @inbounds for gsp in eco_species_ids[e]
        _dom_included(gsp) || continue         # restrict to the selected dominance species
        objs[2 * g + 1] += el.sp_w_loss[gsp]   # ΣW   for this group
        objs[2 * g + 2] += el.sp_agb_loss[gsp] # ΣAGB for this group
      end
    end
  end
  return objs
end

# Scalar aggregate paired with the MO objective vector (for the representative / archive eviction). With
# CELL_NORM the per-cell scales/rank live only in _mo_objectives, so the aggregate = Σ of the objectives;
# otherwise fall back to the established get_total_loss on the (dual-combined) SiteLoss.
@inline _mo_aggregate(objs::Vector{FloatType}, run_result)::Float64 =
  PU.CELL_NORM[] ? Float64(sum(objs)) : convert(Float64, PU.get_total_loss(run_result))

# Validation aggregate from a fit_params result tuple (run_result, cached, eco_losses). Uses the per-cell
# objective aggregate when CELL_NORM (the val fit_params just re-set the per-cell scales to the val
# reference), else the established get_total_loss — so val stays comparable to the train aggregate.
@inline _mo_val_loss(val_result, eco_species_ids)::Float64 =
  _mo_aggregate(_mo_objectives(val_result[3], eco_species_ids), val_result[1])

# Background writer for MOLBSA: checkpoints the full state (archive included) and
# the representative params, and generates plots, off the search thread. Mirrors
# start_writer but reads `state.representative` instead of `state.best`.
function start_mo_writer(::Type{State}, output_dir::AbstractString; buffer_size::Int=8) where {State}
  ch = Channel{Union{WriterJob{State},Symbol}}(buffer_size)
  task = Threads.@spawn begin
    try
      for job in ch
        job === STOP && break
        state = job.state
        if job.is_new_best
          try
            @info "New best: $(state.representative.fx.aggregate)" iter = state.i archive = length(state.archive)
            mkpath(output_dir)
            JLD2.save_object(joinpath(output_dir, "search_state@$(state.i).jld2"), state)
            PU.save_json(joinpath(output_dir, "best_params@$(state.i).json"), state.representative.x)
            JLD2.save_object(joinpath(output_dir, "best_params@$(state.i).jld2"), state.representative.x)
          catch e
            @error "mo_writer: save failed" iter = state.i exception = (e, catch_backtrace())
          end
          if !isnothing(job.emp_sample) && !isnothing(job.sim_sample)
            try
              generate_plots(job.emp_sample, job.sim_sample, "training_$(state.i)", state.representative.fx.aggregate, output_dir)
            catch e
              @error "mo_writer: generate_plots (train) failed" iter = state.i exception = (e, catch_backtrace())
            end
          end
          if !isnothing(job.emp_sample_val) && !isnothing(job.sim_sample_val)
            try
              generate_plots(job.emp_sample_val, job.sim_sample_val, "validation_$(state.i)", state.representative.fx.aggregate, output_dir)
            catch e
              @error "mo_writer: generate_plots (val) failed" iter = state.i exception = (e, catch_backtrace())
            end
          end
        else
          @info ("Best@$(state.best_iteration): agg=$(state.representative.fx.aggregate), archive=$(length(state.archive)), Avg diff: $(state.diff_avg), Temp: $(state.t), Prob: $(state.prob_avg)")
        end
      end
    catch e
      @error "mo_writer: fatal" exception = (e, catch_backtrace())
      rethrow()
    end
  end
  return ch, task
end

# Multi-objective LBSA driver. Mirrors parametrize_LBSA, but scores each candidate
# by the per-(eco,species,{w,agb}) loss vector and compares candidates by net
# objective win-count (MOLBSA.mo_delta) rather than by the scalar total loss.
# All tiers (1, 2, 3, 4) expose the per-ecoregion breakdown that the objective
# vector is built from.
function parametrize_MOLBSA(; ref_soa::ActiveSoA, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, seed_archive_from::Union{Nothing,String}=nothing)
  search_tier in (1, 2, 3, 4, 5) || error("parametrize_MOLBSA: unknown search_tier=$search_tier (expected 1, 2, 3, 4, or 5)")
  splots.sim_year .= Dates.value.(Dates.Day.(splots.measdate - splots.start_measdate)) ./ 365.25 .|> round .|> Int

  all_plot_ids = UIntType.(unique(splots.plot_id))
  sampled_ids = _sample_plot_ids(all_plot_ids, n_output_plots, rng; injection_cohorts=injection_cohorts)
  emp_sample = n_output_plots > 0 ? _make_emp_df(splots, sampled_ids) : nothing

  n_species = length(species_list)
  max_sim_year = site_sim_years.sim_years .|> maximum |> maximum
  param_dists = BSP.make_biomass_param_dists(length(species_list), length(eco_list), eco_species_ids; no_establishment=no_establishment, fit_establishment=(DUAL_MODE[] != :off))
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  injection_dict = isnothing(injection_cohorts) ? nothing : _build_injection_dict(injection_cohorts, ref_soa)
  injection_years = isnothing(injection_cohorts) ? Set{Int}() : Set(Int.(injection_cohorts.sim_year))

  have_val = !isnothing(val_ref_soa)
  inj_dict_val = (have_val && !isnothing(val_injection_cohorts)) ? _build_injection_dict(val_injection_cohorts, val_ref_soa) : nothing
  inj_years_val = (have_val && !isnothing(val_injection_cohorts)) ? Set(Int.(val_injection_cohorts.sim_year)) : Set{Int}()
  val_dual_b = have_val ? _build_val_dual_b(val_splots, eco_species_ids, eco_list, val_spdf_plts, loss_params, cycle_years, val_injection_cohorts, val_spinup_cohorts, rng, no_establishment) : nothing
  sampled_ids_val = (have_val && n_output_plots > 0) ?
                    _sample_plot_ids(UIntType.(unique(val_splots.plot_id)), n_output_plots, rng; injection_cohorts=val_injection_cohorts) :
                    Set{UIntType}()
  emp_sample_val = (have_val && n_output_plots > 0) ? _make_emp_df(val_splots, sampled_ids_val) : nothing

  # Per-tier reference setup (mirrors parametrize_LBSA; tier 3 needs no reference and falls through).
  t4_ref = nothing
  cycle_map = nothing
  n_cycles = 0
  t1_ref = nothing
  t2_ref = nothing
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
  elseif search_tier == 2
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
  elseif search_tier == 4 || search_tier == 5 || DUAL_MODE[] != :off
    n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
    cycle_map, n_cycles = Data.build_cycle_map(splots; cycle_years=cycle_years)
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
  end
  fixed_seeds = _fixed_seeds(rng, n_reps)

  dual_b = DUAL_MODE[] == :off ? nothing : _build_dual_b(splots, eco_species_ids, eco_list, spdf_plts, loss_params, t4_ref, cycle_map, n_cycles, injection_cohorts, spinup_cohorts, rng, no_establishment; b_only=(DUAL_MODE[] == :b))
  _run(p) = fit_params(ref_soa, p, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, t4_ref, cycle_map, n_cycles, seeds=fixed_seeds, injection_dict=injection_dict, injection_years=injection_years, dual_b=dual_b)
  # Score a candidate's repetitions into one MOFitness (eco_losses summed over reps).
  function _fitness(rep_results)
    run_result, eco_losses, _ = _agg_reps(rep_results)
    objs = _mo_objectives(eco_losses, eco_species_ids)
    MOLBSA.MOFitness(objs, _mo_aggregate(objs, run_result)), run_result, eco_losses
  end

  if isnothing(resume_from) && !isnothing(seed_archive_from)
    # Local-refinement mode: pour a prior run's (e.g. CMA-MAE) sorted archive into MOLBSA's Pareto archive,
    # then let MOLBSA refine it locally. The CMA-MAE state's .archive is Vector{MOLBSA.MOCandidate} already.
    @info "Seeding MOLBSA archive from $seed_archive_from (local refinement)"
    seed_st = JLD2.load_object(seed_archive_from)
    seed_arch = hasproperty(seed_st, :archive) ? seed_st.archive : MOLBSA.MOCandidate[]
    rep0 = hasproperty(seed_st, :representative) ? seed_st.representative.x : seed_st.best.x
    (rep0.SPECIES_LIST == species_list && rep0.ECO_LIST == eco_list) ||
      error("seed_archive_from incompatible: its SPECIES_LIST/ECO_LIST differ from the loaded data.")
    bio_params = rep0
    fx0, _, _ = _fitness(_run(bio_params))
    cur = MOLBSA.MOCandidate(bio_params, fx0)
    search_state = MOLBSA.MOLBSAState(cur, cur, rng; max_iter=TRIALS, archive_cap=archive_cap)
    for c in seed_arch
      MOLBSA.update_archive!(search_state, c)
    end
    @info "  seeded $(length(seed_arch)) elites → $(length(search_state.archive)) non-dominated (cap $archive_cap); rep agg=$(round(convert(Float64, cur.fx.aggregate), digits=2))"
    search_state.sobol_cand_idx = 2
  elseif isnothing(resume_from)
    bio_params = if !isnothing(start_from)
      @info "Seeding initial candidate from $start_from (fresh search_state)"
      p = _load_params_from_path(start_from)
      (p.SPECIES_LIST == species_list && p.ECO_LIST == eco_list) ||
        error("start_from params are incompatible with this run: their SPECIES_LIST/ECO_LIST differ from the loaded data.")
      p
    elseif !isempty(_sobol_cands)
      @info "Using Sobol candidate 1/$(length(_sobol_cands)) as initial point"
      _sobol_cands[1]
    else
      BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    end
    fx0, _, _ = _fitness(_run(bio_params))
    cur = MOLBSA.MOCandidate(bio_params, fx0)
    search_state = MOLBSA.MOLBSAState(cur, cur, rng; max_iter=TRIALS, archive_cap=archive_cap)
    search_state.sobol_cand_idx = 2
  else
    @info "Resuming from $resume_from"
    search_state = JLD2.load_object(resume_from)
    search_state.max_iter = TRIALS
    bio_params = search_state.current.x
    force_restart_from_random && (search_state._should_restart = true)
  end
  if TRIALS < 1 || MOLBSA.is_search_over(search_state)
    return search_state
  end
  next_candidate() =
    if search_state.sobol_cand_idx <= length(_sobol_cands)
      p = _sobol_cands[search_state.sobol_cand_idx]
      search_state.sobol_cand_idx += 1
      p
    else
      BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    end

  writer_ch, writer_task = start_mo_writer(typeof(search_state), output_dir)

  losses_db_file = DuckDB.DB(joinpath(output_dir, "losses.duckdb"))
  losses_db = DuckDB.connect(losses_db_file)
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS total_loss (iteration INTEGER, n_sites INTEGER, n_obs INTEGER, total_loss DOUBLE, archive_size INTEGER, params_blob BLOB)")
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS ecoregion_loss (iteration INTEGER, ecoregion VARCHAR, eco_num_sites INTEGER, eco_num_obs INTEGER, ecoregion_total_loss DOUBLE)")
  DuckDB.execute(losses_db, "CREATE TABLE IF NOT EXISTS species_loss (iteration INTEGER, ecoregion VARCHAR, species VARCHAR, age_dist_loss DOUBLE, agb_loss DOUBLE)")

  caused_by_interrupt(e) =
    e isa InterruptException ? true :
    e isa TaskFailedException ? any(en -> caused_by_interrupt(en.exception), Base.current_exceptions(e.task)) :
    e isa CompositeException ? any(caused_by_interrupt, e.exceptions) :
    false

  try
    TProgress.@track for trial in (search_state.i+1):TRIALS
      bio_params, _ = PU.mutate_params(bio_params, param_dists; rng=rng, mutation_mode=PU.BothMutations)
      for _ in 0:rand(rng, 0:2)
        bio_params, _ = PU.mutate_params(bio_params, param_dists; rng=rng, mutation_mode=PU.BothMutations)
      end

      rep_results = _run(bio_params)
      fx, run_result, eco_losses = _fitness(rep_results)
      cached_sites_state = _median_rep_cached(rep_results)

      next = MOLBSA.MOCandidate(bio_params, fx)
      is_new_best = MOLBSA.search_cmp!(next, search_state)
      if MOLBSA.should_restart(search_state)
        @info "Restarting @ $(search_state.i)"
        bio_params = next_candidate()
        rfx, _, _ = _fitness(_run(bio_params))
        is_new_best = MOLBSA.restart(search_state, MOLBSA.MOCandidate(bio_params, rfx))
      end

      val_sim_sample = nothing
      if is_new_best
        iter = search_state.best_iteration
        total = search_state.representative.fx.aggregate
        @info "New best @ $iter | agg=$total | archive=$(length(search_state.archive))"
        try
          test_df = simulate_and_test(; splots=splots, bio_params=search_state.representative.x, eco_list=eco_list, species_list=species_list, eco_species_ids=eco_species_ids, loss_params=loss_params, site_sim_years=site_sim_years, M=n_reps, no_establishment=no_establishment, rng=rng)
          println("Train stats:")
          show(test_df; allrows=true, allcols=true)
          println()
        catch e
          @warn "simulate_and_test (train) failed" exception = (e, catch_backtrace())
        end
        if have_val
          try
            val_result = only(fit_params(val_ref_soa, search_state.representative.x, max_sim_year, n_species,
              eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params;
              debug=false, search_tier=3, injection_dict=inj_dict_val, injection_years=inj_years_val, dual_b=val_dual_b, seeds=[rand(rng, UInt64)]))
            @info "Val loss @ $iter | loss=$(_mo_val_loss(val_result, eco_species_ids))"
            val_sim_sample = n_output_plots > 0 ? _filter_cached_to_df(val_result[2], sampled_ids_val) : nothing
          catch e
            @warn "val fit_params failed" exception = (e, catch_backtrace())
          end
        end
        let buf = IOBuffer()
          Serialization.serialize(buf, search_state.representative.x)
          DuckDB.execute(losses_db, "INSERT INTO total_loss VALUES (?, ?, ?, ?, ?, ?)", [iter, run_result.num_sites, run_result.num_obs, total, length(search_state.archive), take!(buf)])
        end
        for (eco_id, eco_loss) in enumerate(eco_losses)
          eco_name = _eco_name(eco_list, eco_id)
          DuckDB.execute(losses_db, "INSERT INTO ecoregion_loss VALUES (?, ?, ?, ?, ?)", [iter, eco_name, eco_loss.num_sites, eco_loss.num_obs, convert(Float64, PU.get_total_loss(eco_loss))])
          n = max(1, eco_loss.num_sites)
          for gsp in 1:n_species
            eco_loss.sp_w_loss[gsp] == 0f0 && continue
            DuckDB.execute(losses_db, "INSERT INTO species_loss VALUES (?, ?, ?, ?, ?)", [iter, eco_name, species_list[gsp], eco_loss.sp_w_loss[gsp] / n, eco_loss.sp_agb_loss[gsp] / n])
          end
        end
      end
      if is_new_best || search_state.i % 50 == 0
        cached_sites_state_df = DataFrame(cached_sites_state, [:plot_id, :sim_year, :species_id, :age, :agb])
        sim_sample = (is_new_best && n_output_plots > 0) ? _filter_cached_to_df(cached_sites_state, sampled_ids) : nothing
        put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state), splots, cached_sites_state_df, emp_sample, sim_sample, is_new_best ? emp_sample_val : nothing, val_sim_sample))
      end
      MOLBSA.is_search_over(search_state) && break
    end
  catch e
    if caused_by_interrupt(e)
      @info "Search interrupted by user @ trial $(search_state.i); finalizing checkpoint…"
    else
      rethrow()
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

  return search_state
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

# Non-dominated sort of (W,AGB) candidates → ordering by Pareto front, then aggregate within front.
function _dominance_order(W::Vector{Float64}, A::Vector{Float64}, G::Vector{Float64})
  n = length(W)
  dom(i, j) = (W[i] <= W[j] && A[i] <= A[j]) && (W[i] < W[j] || A[i] < A[j])
  front = fill(0, n); remaining = Set(1:n); fr = 0
  while !isempty(remaining)
    fr += 1
    nd = [i for i in remaining if !any(j -> dom(j, i), remaining)]
    for i in nd; front[i] = fr; delete!(remaining, i); end
  end
  sortperm([(front[i], G[i]) for i in 1:n])   # front rank, then aggregate within front
end

# rank = :dominance (non-dominated sort over ΣW,ΣAGB — exchange-rate-free, keeps the whole tradeoff;
#        the scalar aggregate is ~pure-AGB so don't rank by it), :balanced (min-max ΣW+ΣAGB), or :aggregate.
function load_sobol_candidates(db_path::String; top_frac::Float64=0.5, run_id::Union{Nothing,String}=nothing, rank::Symbol=:dominance)
  isfile(db_path) || return []
  db = DuckDB.DB(db_path)
  con = DuckDB.connect(db)
  try
    cols = Set(string(r.name) for r in (DuckDB.execute(con, "PRAGMA table_info('sobol_results')") |> DataFrame |> eachrow))
    hasWA = ("sumW" in cols) && ("sumAGB" in cols)
    sel = hasWA ? "mean_loss, sumW, sumAGB, params_blob" : "mean_loss, params_blob"
    where = isnothing(run_id) ? "" : " WHERE run_id = '$(run_id)'"
    result = DuckDB.execute(con, "SELECT $sel FROM sobol_results$where ORDER BY mean_loss ASC") |> DataFrame
    isempty(result) && return []
    G = Float64.(result.mean_loss)
    order = if rank != :aggregate && hasWA
      W = Float64.(result.sumW); A = Float64.(result.sumAGB)
      if rank == :balanced
        rs(x) = (lo = minimum(x); hi = maximum(x); hi > lo ? (x .- lo) ./ (hi - lo) : zero(x))
        sortperm(rs(W) .+ rs(A))
      else
        _dominance_order(W, A, G)   # :dominance
      end
    else
      sortperm(G)                   # :aggregate
    end
    n_keep = max(1, round(Int, nrow(result) * top_frac))
    keep = order[1:n_keep]
    @info "Loaded $(n_keep) Sobol candidates (top $(round(Int, top_frac*100))% of $(nrow(result)), rank=$rank)"
    return [Serialization.deserialize(IOBuffer(result.params_blob[i])) for i in keep]
  catch e
    @warn "load_sobol_candidates failed" exception = e
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

# Load the data-derived per-species LONGEVITY table (species_symbol -> years) from species_longevity_ref
# in the cohorts DB. Returns nothing (with a warning) if the table is absent. See species_longevity_ref_README.md.
function _load_longevity_table(db_path::String)
  # READ-ONLY attach (in-memory main + ATTACH … READ_ONLY) — never open the cohorts DB read-write just to
  # SELECT; a read-write open here risks corrupting the DB if the process is killed mid-load.
  con = DuckDB.connect(DuckDB.DB())
  DuckDB.execute(con, "ATTACH '$(db_path)' AS src (READ_ONLY);")
  has = (DuckDB.execute(con, "SELECT count(*) AS n FROM information_schema.tables WHERE table_name='species_longevity_ref'") |> DataFrame).n[1]
  if has == 0
    @warn "longevity_from_data=true but species_longevity_ref not found in $db_path — LONGEVITY stays in the search"
    return nothing
  end
  df = DuckDB.execute(con, "SELECT species_symbol, longevity FROM src.species_longevity_ref WHERE longevity IS NOT NULL") |> DataFrame
  d = Dict{String,Float64}(String(r.species_symbol) => Float64(r.longevity) for r in eachrow(df))
  @info "longevity_from_data: pinned LONGEVITY for $(length(d)) species from species_longevity_ref (out of search)"
  d
end

# Load per-species SHADE_TOL (shade class 1-5) from a CSV (cols: sym, shade_class), e.g.
# runs/shadetol_all_species.csv. Pins SHADE_TOL out of the search. Missing/absent → SHADE_TOL_DEFAULT.
function _load_shade_tol_table(csv_path::String)
  if !isfile(csv_path)
    @warn "shade_tol_from_data=true but $csv_path not found — SHADE_TOL stays in the search"
    return nothing
  end
  df = CSV.read(csv_path, DataFrame)
  d = Dict{String,Int}(uppercase(strip(String(r.sym))) => Int(r.shade_class) for r in eachrow(df))
  @info "shade_tol_from_data: pinned SHADE_TOL for $(length(d)) species from $csv_path (out of search)"
  d
end

# Load per-(category, L3, land_use) PROB_ESTAB from the natural + artificial rollup CSVs (cols: category,
# l3, prob_estab). natural CSV → natural strata, artificial CSV → artificial strata. Seeds (not pins) the
# initial PROB_ESTAB_SPP → the run becomes a calibration around the data-derived rates.
function _load_prob_estab_table(csv_nat::String, csv_art::String)
  d = Dict{Tuple{String,String,String},Float64}()
  for (path, lu) in ((csv_nat, "natural"), (csv_art, "artificial"))
    (isempty(path) || !isfile(path)) && (path == csv_nat && @warn "prob_estab_from_data: $path not found"; continue)
    df = CSV.read(path, DataFrame)
    for r in eachrow(df)
      d[(uppercase(strip(String(r.category))), String(r.l3), lu)] = Float64(r.prob_estab)
    end
  end
  isempty(d) && (@warn "prob_estab_from_data: no rows loaded — PROB_ESTAB left as-is"; return nothing)
  @info "prob_estab_from_data: loaded $(length(d)) (category,l3,lu) PROB_ESTAB values"
  d
end

_parse_eco_lu(s) = (p = split(String(s), "|lu="); length(p) == 2 ? (String(p[1]), String(p[2])) : (String(s), ""))

# Seed the initial candidate's PROB_ESTAB_SPP[eco][sp] from the data table (in place; stays in the search).
function _seed_prob_estab!(params, eco_list, species_list, eco_species_ids)
  tbl = BiomassSuccessionPlugin.PROB_ESTAB_TABLE[]
  isnothing(tbl) && return params
  n = 0
  for (eco_id, sp_ids) in enumerate(eco_species_ids)
    l3, lu = _parse_eco_lu(eco_list[eco_id])
    for (j, gsp) in enumerate(sp_ids)
      v = get(tbl, (uppercase(species_list[gsp]), l3, lu), nothing)
      v === nothing && continue
      params.PROB_ESTAB_SPP[eco_id][j] = FloatType(v); n += 1
    end
  end
  @info "prob_estab_from_data: seeded PROB_ESTAB for $n eco×species cells (calibration start)"
  params
end

# Per-category MATURITY (SONA age) from prob_estab_all_species.csv (cols: category, maturity). One value per
# category (constant across L3); median if a category has several. Seeds (not pins) MATURITY.
function _load_maturity_table(csv_path::String)
  if !isfile(csv_path)
    @warn "maturity_from_data=true but $csv_path not found — MATURITY left as-is"; return nothing
  end
  df = CSV.read(csv_path, DataFrame)
  d = Dict{String,Int}()
  for g in groupby(df, :category)
    d[uppercase(strip(String(g.category[1])))] = round(Int, Statistics.median(Float64.(g.maturity)))
  end
  @info "maturity_from_data: loaded MATURITY for $(length(d)) categories from $csv_path"
  d
end

# Seed the initial candidate's per-species MATURITY from the table (in place; stays in the search).
function _seed_maturity!(params, species_list)
  tbl = BiomassSuccessionPlugin.MATURITY_TABLE[]
  isnothing(tbl) && return params
  n = 0
  for gsp in eachindex(species_list)
    v = get(tbl, uppercase(species_list[gsp]), nothing)
    v === nothing && continue
    params.MATURITY[gsp] = FloatType(v); n += 1
  end
  @info "maturity_from_data: seeded MATURITY for $n species (calibration start)"
  params
end

# Data-derived MIN_REL_BIOMASS default (base 0.10 + fixed 0.175 spacing; see FIA_DATA_PREP shade analysis).
const MIN_REL_PINNED = FloatType[0.10, 0.275, 0.45, 0.625, 0.80]
# PIN MIN_REL_BIOMASS out of the search: overwrite every ecoregion's thresholds with MIN_REL_PINNED IN PLACE.
# next_candidate() draws a random gradient, so (like _apply_frozen_growth!) this must be re-applied on restart.
function _seed_min_rel!(params, eco_list)
  BSP.FIX_MIN_REL[] || return params
  for e in eachindex(params.MIN_REL_BIOMASS)
    params.MIN_REL_BIOMASS[e] = copy(MIN_REL_PINNED)
  end
  @info "fix_min_rel: pinned MIN_REL_BIOMASS = $(MIN_REL_PINNED) for $(length(params.MIN_REL_BIOMASS)) ecoregions (out of search)"
  params
end

# Re-pin the fix_growth-frozen params {D,S,ANPP_MAX,B_MAX} to a reference (start_from / Sim A) IN PLACE. IPOP
# restart draws a fresh next_candidate() with RANDOM growth, which silently breaks fix_growth (frozen params
# must stay at the seed). No-op when fix_growth is off or no reference is given.
function _apply_frozen_growth!(cand, ref)
  (BSP.FIX_GROWTH[] && ref !== nothing) || return cand
  cand.D .= ref.D
  cand.S .= ref.S
  for e in eachindex(cand.ANPP_MAX_SPP); cand.ANPP_MAX_SPP[e] .= ref.ANPP_MAX_SPP[e]; end
  for e in eachindex(cand.B_MAX_SPP); cand.B_MAX_SPP[e] .= ref.B_MAX_SPP[e]; end
  cand
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
  start_from = get_cfg("start_from", nothing)
  start_from = (start_from === nothing || start_from == "null") ? nothing : String(start_from)
  filter_extent = get_cfg("filter_extent", nothing)
  filter_extent = (filter_extent === nothing || filter_extent == "null") ? nothing : String(filter_extent)
  sobol_candidates_db = get_cfg("sobol_candidates_db", nothing)
  sobol_candidates_db = (sobol_candidates_db === nothing || sobol_candidates_db == "null") ? nothing : String(sobol_candidates_db)

  # Fields shared by parametrize, plot_sample, and plot_sample_sobol
  common_kw = (
    cohorts_db_path=get_cfg("cohorts_db_path", "../data_eco_cohorts.duckdb"),
    filter_eco_field=get_cfg("filter_eco_field", "epa_l3"),
    eco_field=get_cfg("eco_field", "epa_l3"),
    tablename=get_cfg("tablename", "data_eco_cohorts"),
    output_dir=get_cfg("output_dir", "./outputs"),
    filter_ecos=String.(get_cfg("filter_ecos", String[])),
    filter_plots=filter_plots,
    filter_species=String.(get_cfg("filter_species", String[])),
    skip_disturbances=get_cfg("skip_disturbances", true),
    spinup=get_cfg("spinup", false),
    by_subplot=get_cfg("by_subplot", false),
    no_establishment=get_cfg("no_establishment", false),
    min_trees=Int(get_cfg("min_trees", 100)),
    min_agb_frac=Float64(get_cfg("min_agb_frac", 0.05)),
    stratify_eco_mixed=Bool(get_cfg("stratify_eco_mixed", false)),
    single_ecoregion=Bool(get_cfg("single_ecoregion", false)),
    stratify_landuse=Bool(get_cfg("stratify_landuse", false)),
    filter_extent=filter_extent,
    bins_idx=Int.(get_cfg("bins_idx", vcat(10:10:40, 60:20:120))),
  )
  n_output_plots = get_cfg("n_output_plots", 0)
  sw_kw = (smoothing_window_size=get_cfg("smoothing_window_size", 0),
    smoothing_variance=Float64(get_cfg("smoothing_variance", 1.2)))

  search_mode = get_cfg("search_mode", "lbsa")

  # Injection-override flags (module Refs read at injection time). Default = the current Ref
  # value, so omitting a key keeps the in-code default; set them in yaml to control a run.
  OVERRIDE_INJECTION[] = Bool(get_cfg("override_injection", OVERRIDE_INJECTION[]))
  OVERRIDE_INJECTION_REPLACE[] = Bool(get_cfg("override_injection_replace", OVERRIDE_INJECTION_REPLACE[]))
  OVERRIDE_INJECTION_SYNC[] = Bool(get_cfg("override_injection_sync", OVERRIDE_INJECTION_SYNC[]))
  OVERRIDE_INJECTION_NOISE[] = Float64(get_cfg("override_injection_noise", OVERRIDE_INJECTION_NOISE[]))
  OVERRIDE_INJECTION_DISTURBANCE[] = Symbol(get_cfg("override_injection_disturbance", String(OVERRIDE_INJECTION_DISTURBANCE[])))
  Data.USE_FIA_CYCLE[] = Bool(get_cfg("fia_cycle", false))   # tier-4/5 cycles from FIA `cycle` vs computed split
  Data.FIA_CYCLE_MERGE[] = Int(get_cfg("fia_cycle_merge", 3))   # merge N consecutive FIA cycles into one group
  DUAL_MODE[] = let v = get_cfg("dual_mode", "off")          # off (A only) | joint (A⊕B) | b (B only, seed from A)
    v === true ? :joint : v === false ? :off : Symbol(lowercase(String(v)))
  end
  TIER_B[] = Int(get_cfg("tier_b", 4))                       # Sim B's tier (A's tier = config `tier`)
  LBSA_STRETCH_LEN[] = Int(get_cfg("lbsa_stretch_len", 150)) # LBSA stretch length
  LBSA_STALE_RATIO[] = Float64(get_cfg("lbsa_stale_ratio", 0.95)) # LBSA freeze-detection uphill-attempt ratio
  LBSA_TEMP_LIST_LEN[] = Int(get_cfg("lbsa_temp_list_len", 150))   # LBSA temperature-list length L
  LBSA_REPLACE_OLDEST[] = Bool(get_cfg("lbsa_replace_oldest", true)) # false → overwrite the max each stretch (responsive temp)
  LBSA_COOLING_ONLY[] = Bool(get_cfg("lbsa_cooling_only", false))    # true → monotone cooling → search actually freezes → reheat/restart fires
  SIMB_SPINUP[] = Bool(get_cfg("simB_spinup", true))         # false = Sim B starts from observed cohorts (no spinup)
  T4_PER_PLOT_MEAN[] = Bool(get_cfg("simB_per_plot_mean", false))   # tier-4: AGB as mean biomass-per-plot-by-age
  T4_MEAN_OVER_REPS[] = Bool(get_cfg("simB_rep_mean", false))       # tier-4: score loss on the mean-over-reps sim histogram
  SIMB_DISTURB_ONLY[] = Bool(get_cfg("simB_disturb_only", false))   # Sim B purely free (no injection); disturbance-scale on ALL plots
  MOCMAES.USE_NDS[] = Bool(get_cfg("mocmaes_true_nds", false)) # true = MO-CMA-ES offspring ranked by NSGA-II Pareto fronts (vs default win-count)
  MOCMAES.USE_NDS[] && @info "MO-CMA-ES offspring ranking: TRUE NSGA-II non-dominated sorting (Pareto fronts + crowding)"
  BiomassSuccessionPlugin.FIXED_LONGEVITY[] = (let v = get_cfg("fix_longevity", nothing); isnothing(v) ? nothing : Float64(v) end)
  BiomassSuccessionPlugin.LONGEVITY_TABLE[] = Bool(get_cfg("longevity_from_data", false)) ?
    _load_longevity_table(String(get_cfg("cohorts_db_path", "../data_eco_cohorts.duckdb"))) : nothing
  BiomassSuccessionPlugin.SHADE_TOL_TABLE[] = Bool(get_cfg("shade_tol_from_data", false)) ?
    _load_shade_tol_table(String(get_cfg("shade_tol_csv", "./runs/shadetol_all_species.csv"))) : nothing
  BiomassSuccessionPlugin.PROB_ESTAB_TABLE[] = Bool(get_cfg("prob_estab_from_data", false)) ?
    _load_prob_estab_table(String(get_cfg("prob_estab_csv", "./runs/prob_estab_from_data.csv")),
                           String(get_cfg("prob_estab_csv_artificial", "./runs/prob_estab_from_data_artificial.csv"))) : nothing
  BiomassSuccessionPlugin.MATURITY_TABLE[] = Bool(get_cfg("maturity_from_data", false)) ?
    _load_maturity_table(String(get_cfg("maturity_csv", "./runs/prob_estab_all_species.csv"))) : nothing
  BSP.FIX_GROWTH[] = Bool(get_cfg("fix_growth", false))     # stage-B: fix {D,S,ANPP_MAX,B_MAX}, fit establishment only
  BSP.FIX_MATURITY[] = Bool(get_cfg("fix_maturity", false)) # pin MATURITY out of the search (at MATURITY_TABLE/SONA)
  BSP.FIX_MIN_REL[] = Bool(get_cfg("fix_min_rel", false))   # pin MIN_REL_BIOMASS out of the search (at MIN_REL_PINNED)

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
    search_mode=search_mode,
    tier=get_cfg("tier", 1),
    smoothing_window=smoothing_window,
    unbinned_w=Bool(get_cfg("unbinned_w", false)),   # Sim A: exact per-year EMD (no binning/age-smoothing)
    w_count_balance=Bool(get_cfg("w_count_balance", false)),
    w_count_balance_mode=String(get_cfg("w_count_balance_mode", "both")),
    w_count_beta=Float64(get_cfg("w_count_beta", 0.99)),
    TRIALS=get_cfg("trials", 1000000),
    resume_from=resume_from,
    start_from=start_from,
    force_restart_from_random=get_cfg("force_restart_from_random", false),
    sobol_n=get_cfg("sobol_n", 100),
    saltelli=get_cfg("saltelli", false),
    n_reps=get_cfg("n_reps", 5),
    sobol_candidates_db=sobol_candidates_db,
    sobol_top_frac=Float64(get_cfg("sobol_top_frac", 0.5)),
    n_output_plots=n_output_plots,
    val_frac=Float64(get_cfg("val_frac", 0.0)),
    split_seed=Int(get_cfg("split_seed", 42)),
    diagnose=get_cfg("diagnose", false),
    cycle_years=get_cfg("cycle_years", 8),
    loss_lambda=Float64(get_cfg("loss_lambda", 1.0)),
    loss_alpha=Float64(get_cfg("loss_alpha", 1.0)),
    agb_hinge=Bool(get_cfg("agb_hinge", false)),
    agb_hinge_threshold=Float64(get_cfg("agb_hinge_threshold", 10.0)),
    agb_hinge_pct=Float64(get_cfg("agb_hinge_pct", 0.0)),
    agb_hinge_pct_min=Float64(get_cfg("agb_hinge_pct_min", 0.0)),
    agb_hinge_pct_max=Float64(get_cfg("agb_hinge_pct_max", Inf)),
    agb_hinge_l2=Bool(get_cfg("agb_hinge_l2", false)),
    agb_hinge_p=Float64(get_cfg("agb_hinge_p", 2.0)),   # hinge exponent pen(h)=h^p, default 2 (L2)
    agb_hinge_beta=Float64(get_cfg("agb_hinge_beta", 1.0)),
    agb_normalize=Bool(get_cfg("agb_normalize", false)),
    w_normalize=Bool(get_cfg("w_normalize", false)),
    cell_normalize=Bool(get_cfg("cell_normalize", false)),
    w_scale_factor=Float64(get_cfg("w_scale_factor", 1.0)),
    w_smooth=Bool(get_cfg("w_smooth", false)),
    w_smooth_band=Float64(get_cfg("w_smooth_band", 0.05)),
    w_smooth_conc=Float64(get_cfg("w_smooth_conc", 0.5)),
    w_smooth_beta=Float64(get_cfg("w_smooth_beta", 1.0)),
    w_smooth_auto_knee=Bool(get_cfg("w_smooth_auto_knee", false)),
    w_p=Float64(get_cfg("w_p", 1.0)),
    loss_piecewise=Bool(get_cfg("loss_piecewise", false)),
    w_pivot=Float64(get_cfg("w_pivot", 1.0)),
    agb_pivot=Float64(get_cfg("agb_pivot", 1.0)),
    init_perturb_frac=Float64(get_cfg("init_perturb_frac", 0.0)),
    init_perturb_cap=Float64(get_cfg("init_perturb_cap", 50.0)),
    cmaes_lambda=(haskey(cfg, "cmaes_lambda") ? Int(get_cfg("cmaes_lambda", 0)) : nothing),
    cmaes_sigma0=Float64(get_cfg("cmaes_sigma0", 0.3)),
    cmaes_warmstart_seeds=Int(get_cfg("cmaes_warmstart_seeds", 20)),
    ipop=get_cfg("ipop", false),
    ipop_stagnation=Int(get_cfg("ipop_stagnation", 20)),
    cmaes_archive_cap=Int(get_cfg("cmaes_archive_cap", 200)),
    seed_archive_from=(let v = get_cfg("seed_archive_from", nothing); (v === nothing || v == "null") ? nothing : String(v) end),
    cmaes_integer_handling=get_cfg("cmaes_integer_handling", false),
    cmaes_integer_std_factor=Float64(get_cfg("cmaes_integer_std_factor", 0.3)),
    cmaes_single_cov=Bool(get_cfg("cmaes_single_cov", false)),
    igel_mu=Int(get_cfg("igel_mu", 20)),
    igel_sigma0=Float64(get_cfg("igel_sigma0", 0.3)),
    igel_sobol_init=get_cfg("igel_sobol_init", true),
    igel_niche_radius=Float64(get_cfg("igel_niche_radius", 0.0)),
    igel_reseed_sigma=Float64(get_cfg("igel_reseed_sigma", 0.0)),
    igel_maturity=Int(get_cfg("igel_maturity", 0)),
    cmame_alpha=Float64(get_cfg("cmame_alpha", 0.02)),
    cmame_grid=Int(get_cfg("cmame_grid", 15)),
    cmame_reseed_explore=Float64(get_cfg("cmame_reseed_explore", 1.0)),
    cmame_restart_patience=Int(get_cfg("cmame_restart_patience", 6)),
    cmame_sobol_reseed=Bool(get_cfg("cmame_sobol_reseed", false)),
    cmame_mo_rank=Bool(get_cfg("cmame_mo_rank", false)),
    balanced_quality=Bool(get_cfg("balanced_quality", false)),
    archive_by_sp=Bool(get_cfg("archive_by_sp", false)),
    bounds_by_sobol=Bool(get_cfg("bounds_by_sobol", false)),
    top_seeds=Float64(get_cfg("top_seeds", 0.2)),
    dominance_species=String(get_cfg("dominance_species", "exact")),
    rng=rng)
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
  elseif raw isa Vector  # sobol results
    @info "Loaded sobol results — using rank-1 params" mean_loss = raw[1].mean_loss
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
  min_trees=100, min_agb_frac=0.05, stratify_eco_mixed=false, filter_extent=nothing,
  bins_idx, smoothing_window_size, smoothing_variance, rng)
  smoothing_window = smoothing_window_size > 0 ?
                     PU.get_smoothing_window(; smoothing_window=smoothing_window_size,
    smoothing_variance=FloatType(smoothing_variance)) :
                     FloatType[one(FloatType)]
  loss_params = PU.LossParams(
    age_bins=PU.AgeBins(bins_idx=bins_idx .|> Int, last_bin_open=true),
    smoothing_weights=smoothing_window,
  )
  splots, _, species_list, eco_species_ids, _ =
    Data.prepare_parametrization_data(;
      cohorts_db_path, eco_field, tablename, output_dir,
      skip_disturbances, spinup, by_subplot, filter_eco_field,
      filter_ecos, filter_plots, filter_species, min_trees, min_agb_frac,
      stratify_eco_mixed, filter_extent, RNG=rng)
  splots.sim_year .= Dates.value.(Dates.Day.(splots.measdate .- splots.start_measdate)) ./ 365.25 .|> round .|> Int
  max_sim_year = maximum(splots.sim_year)
  max_age = Int(maximum(splots.age_calc))
  site_sim_years = Data.get_site_sim_years(splots)
  spinup_cohorts = Data.get_spinup_cohorts(splots)
  spdf = PU.smoothen_ref_years(splots, loss_params, max_age; debug=false)
  spdf_plts = Data.make_spdf_dict(spdf, eco_species_ids)
  PlotContext(splots, species_list, eco_species_ids, site_sim_years,
    spinup_cohorts, spdf_plts, loss_params, max_sim_year,
    no_establishment,
    (no_establishment || OVERRIDE_INJECTION[]) ? Data.get_injection_cohorts(splots; all_cohorts=OVERRIDE_INJECTION[]) : nothing)
end

function _run_and_plot(bio_params, label, ctx::PlotContext;
  sampled_ids, emp_sample, output_dir, spinup, rng)
  ref_soa = make_sites(ctx.splots, ctx.eco_species_ids; rng, spinup, no_establishment=ctx.no_establishment)
  inj_dict = isnothing(ctx.injection_cohorts) ? nothing : _build_injection_dict(ctx.injection_cohorts, ref_soa)
  inj_years = isnothing(ctx.injection_cohorts) ? Set{Int}() : Set(Int.(ctx.injection_cohorts.sim_year))
  result = only(fit_params(ref_soa, bio_params, ctx.max_sim_year,
    length(ctx.species_list), ctx.eco_species_ids,
    ctx.spdf_plts, ctx.site_sim_years, spinup,
    ctx.spinup_cohorts, ctx.loss_params;
    debug=false, search_tier=3, seeds=[rand(rng, UInt64)],
    injection_dict=inj_dict, injection_years=inj_years))
  loss = convert(Float64, PU.get_total_loss(result[1]))
  sim_sample = _filter_cached_to_df(result[2], sampled_ids)
  generate_plots(emp_sample, sim_sample, label, loss, output_dir)
  @info "Plots saved [$label]" loss n_plots = length(sampled_ids)
end

"""
plot_sample — run one simulation with given params and plot n_output_plots forest plots.
Accepts best_params@X.jld2, search_state@X.jld2, or sobol_results@X.jld2.
"""
function plot_sample(;
  cohorts_db_path::String,
  params_path::String,
  output_dir::String="./outputs",
  filter_eco_field::String="epa_l3",
  eco_field::String="epa_l3",
  tablename::String="curated_cohorts_landis",
  filter_ecos::Vector{String}=String[],
  filter_plots::Vector{NTuple{4,Int}}=NTuple{4,Int}[],
  filter_species::Vector{String}=String[],
  skip_disturbances::Bool=true,
  spinup::Bool=false,
  by_subplot::Bool=false,
  no_establishment::Bool=false,
  bins_idx::Vector{Int64}=vcat(10:10:40, 60:20:120) .|> Int64,
  smoothing_window_size::Int=0,
  smoothing_variance::Float64=1.2,
  min_trees::Int=100,
  min_agb_frac::Float64=0.05,
  stratify_eco_mixed::Bool=false,
  filter_extent::Union{Nothing,String}=nothing,
  n_output_plots::Int=20,
  rng_seed::Int=1337,
)
  mkpath(output_dir)
  rng = RNGType(UInt64(rng_seed))
  bio_params = _load_params_from_path(params_path)
  ctx = _load_plot_context(; cohorts_db_path, eco_field, tablename, output_dir,
    skip_disturbances, spinup, by_subplot, no_establishment, filter_eco_field,
    filter_ecos, filter_plots, filter_species,
    min_trees, min_agb_frac, stratify_eco_mixed, filter_extent,
    bins_idx, smoothing_window_size, smoothing_variance, rng)
  all_ids = UIntType.(unique(ctx.splots.plot_id))
  sampled_ids = _sample_plot_ids(all_ids, n_output_plots, rng; injection_cohorts=ctx.injection_cohorts)
  emp_sample = _make_emp_df(ctx.splots, sampled_ids)
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
  output_dir::String="./outputs",
  filter_eco_field::String="epa_l3",
  eco_field::String="epa_l3",
  tablename::String="curated_cohorts_landis",
  filter_ecos::Vector{String}=String[],
  filter_plots::Vector{NTuple{4,Int}}=NTuple{4,Int}[],
  filter_species::Vector{String}=String[],
  skip_disturbances::Bool=true,
  spinup::Bool=false,
  by_subplot::Bool=false,
  no_establishment::Bool=false,
  bins_idx::Vector{Int64}=vcat(10:10:40, 60:20:120) .|> Int64,
  smoothing_window_size::Int=0,
  smoothing_variance::Float64=1.2,
  min_trees::Int=100,
  min_agb_frac::Float64=0.05,
  stratify_eco_mixed::Bool=false,
  filter_extent::Union{Nothing,String}=nothing,
  n_output_plots::Int=10,              # fixed forest-plot sample size
  n_sobol_params_to_plot::Int=5,               # sobol param sets to simulate
  rng_seed::Int=1337,
)
  mkpath(output_dir)
  rng = RNGType(UInt64(rng_seed))

  db = DuckDB.DB(sobol_candidates_db)
  con = DuckDB.connect(db)
  raw = DuckDB.execute(con, "SELECT mean_loss, params_blob FROM sobol_results ORDER BY mean_loss ASC") |> DataFrame
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

  ctx = _load_plot_context(; cohorts_db_path, eco_field, tablename, output_dir,
    skip_disturbances, spinup, by_subplot, no_establishment, filter_eco_field,
    filter_ecos, filter_plots, filter_species,
    min_trees, min_agb_frac, stratify_eco_mixed, filter_extent,
    bins_idx, smoothing_window_size, smoothing_variance, rng)
  all_ids = UIntType.(unique(ctx.splots.plot_id))
  sampled_ids = _sample_plot_ids(all_ids, n_output_plots, rng; injection_cohorts=ctx.injection_cohorts)
  emp_sample = _make_emp_df(ctx.splots, sampled_ids)

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
  equiv_margin::Float64=0.2,   # equivalence band = ±equiv_margin · mean(ref)
  tost_alpha::Float64=0.05,
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
  sim_sum = Dict{NTuple{4,Int},Float64}()
  sim_cnt = Dict{NTuple{4,Int},Int}()

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
  rng_seed::Int=113387,
  timehorizon_years::Int=50,
  output_every_years::Int=5,
  simple_match::Bool=true,
  stratify_eco_mixed::Bool=false,
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
    @time splots, eco_list, eff_eco_list, species_list = (simple_match ? Data.load_treemap_cohorts_simple : Data.load_treemap_cohorts)(
      cn_raster, eco_raster_data, treemap_db_path, eco_mapping_path, params; stratify_eco_mixed=stratify_eco_mixed)
    println("Plots: $(length(unique(splots.plt_cn))), Ecos: $(length(eco_list)), Species: $(length(species_list))")
    println(species_list)

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
  BiomassSuccessionPlugin.generate_rasters_from_output(; output_dir=output_dir, ef_raster_path=ref_raster_path)

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
  min_rel_biomass::Vector{Float32}=Float32[0.25, 0.45, 0.56, 0.70, 0.90],
  rng_seed::Int=1337,
  timehorizon_years::Int=50,
  output_every_years::Int=1,
)
  rng = RNGType(UInt64(rng_seed))

  println("Loading LANDIS rasters")
  @time communities_raster = Data.load_landis_mapcode_raster(initial_communities_tif)
  @time eco_raster = Data.load_eco_raster(ecoregion_tif)
  @assert size(communities_raster) == size(eco_raster) "Raster size mismatch"

  println("Loading LANDIS tables")
  println("eco ecocode mapping")
  eco_ecocode_df = CSV.read(eco_ecocode_mapping, DataFrame)
  println("eco ecocode mapping, done")

  println("initial communities csv")
  ic_df = CSV.read(initial_communities_csv, DataFrame)
  println("initial communities csv: done")

  println("Core species data")
  core_sp_df = Data.load_landis_core_species(core_species_data)
  println("Core species data done")
  println("Core biomass spp")
  spp_eco_df = Data.load_landis_spp_ecoregion(spp_ecoregion_data; year=spp_eco_year)
  println("Core biomass spp done")
  println("species data")
  species_df = CSV.read(species_data, DataFrame)
  println("species data done")

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
  downsample::Int=1,
  # Ignored parameters (for signature compatibility with simulate_spatial_treemap)
  rng_seed::Int=1337,
  timehorizon_years::Int=50,
  output_every_years::Int=5,
  simple_match::Bool=true,
  stratify_eco_mixed::Bool=false,
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
  local mapcode_raster, coarse_eco_raster
  # When stratifying, the exported ecoregion files encode the stratum in the ecocode
  # (base*10 + mixed?1:2); otherwise these stay at the base raster/mapping (copy as-is).
  strat_eco_raster = nothing
  eco_export_mapping = eco_mapping_df

  if !isnothing(treemap_raster)
    println("Loading treemap raster: $treemap_raster")
    @time cn_raster, _ = Data.load_treemap_raster(
      joinpath(data_dir, treemap_raster); treemap_version=treemap_version)
    @assert size(cn_raster) == size(eco_raster_data) "Raster size mismatch: treemap $(size(cn_raster)) ≠ eco $(size(eco_raster_data))"

    # Optional spatial downsample: group n×n fine pixels into one coarse cell. The eco raster
    # used for species matching is replaced by its block-majority so a coarse cell has a single
    # ecoregion. With stratify_eco_mixed, the per-plot stratum is baked into the ecocode FIRST
    # (base*10 + mixed?1:2) and then majority-voted together with the base eco — so the cell's
    # stratum is a majority vote too, and the loader matches every cohort in the block against
    # that single stratified ecoregion (consistent base+stratum majority vote).
    eco_match_raster = eco_raster_data
    eco_loader_override = nothing      # stratified mapping df handed to the loader
    loader_stratify = stratify_eco_mixed
    mapcode_raster = nothing
    coarse_eco_raster = nothing
    if downsample > 1
      println("Downsampling rasters $(downsample)x ($(cell_length_m)m → $(cell_length_m * downsample)m), eco by majority vote")
      if stratify_eco_mixed
        cn_mixed = Data.load_plot_mixed(cn_raster, treemap_db_path)
        strat_fine = Data.stratify_eco_raster(eco_raster_data, cn_raster, cn_mixed)
        eco_export_mapping = Data.stratified_eco_mapping(eco_mapping_df)
        eco_match_raster, coarse_eco_raster = Data.block_majority_eco(strat_fine, downsample, Set(Int.(eco_export_mapping.ecocode)))
        eco_loader_override = eco_export_mapping  # ecocodes already stratified → loader runs unstratified
        loader_stratify = false
      else
        eco_match_raster, coarse_eco_raster = Data.block_majority_eco(eco_raster_data, downsample, Set(Int.(eco_mapping_df.ecocode)))
      end
    end

    println("Extracting cohorts from DuckDB (treemap path)")
    @time splots, eco_list, eff_eco_list, species_list = (simple_match ? Data.load_treemap_cohorts_simple : Data.load_treemap_cohorts)(
      cn_raster, eco_match_raster, treemap_db_path, eco_mapping_path, params;
      stratify_eco_mixed=loader_stratify, eco_ecocode_override=eco_loader_override)
    println("Plots: $(length(unique(splots.plt_cn))), Ecos: $(length(eco_list)), Species: $(length(species_list))")
    println(species_list)

    println("Remapping params to data eco/species")
    @time mod_params, mapped_splots, eco_species_ids = Data.map_params_to_data_treemap(
      params, eco_list, eff_eco_list, species_list, splots)

    if downsample > 1
      println("Aggregating cohorts into $(downsample)x coarse cells (combine species×age, biomass / $(downsample^2))")
      @time communities_df, mapcode_raster = Data.aggregate_coarse_communities(
        mapped_splots, cn_raster, eco_match_raster, downsample)
      combo_to_mapcode = nothing
    else
      println("Deduplicating cohorts by (plt_cn, ecocode)")
      @time communities_df, combo_to_mapcode = Data.deduplicate_for_export(
        mapped_splots, cn_raster, eco_raster_data)
    end

    # Non-downsample stratification: build a full-res stratified ecoregion raster + mapping.
    # (The downsample+stratify case already produced coarse_eco_raster/eco_export_mapping above.)
    if stratify_eco_mixed && downsample == 1
      println("Building stratified ecoregion raster + mapping (base*10 + mixed?1:2)")
      cn_mixed = Data.load_plot_mixed(cn_raster, treemap_db_path)
      strat_eco_raster = Data.stratify_eco_raster(eco_raster_data, cn_raster, cn_mixed)
      eco_export_mapping = Data.stratified_eco_mapping(eco_mapping_df)
    end

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
    mapcode_raster = nothing
    coarse_eco_raster = nothing

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
  eff_cell_length = downsample > 1 ? cell_length_m * downsample : cell_length_m
  println("Exporting scenario.txt")
  BiomassSuccessionPlugin.export_scenario_file(;
    output_path=joinpath(output_dir, "scenario.txt"),
    duration_years=duration_years,
    cell_length_m=eff_cell_length,
    rng_seed=rng_seed,
  )

  # 2. ecoregion.txt + ecoregion.tif (stratified, downsampled, or copied as-is)
  println("Exporting ecoregion.txt")
  BiomassSuccessionPlugin.export_ecoregions_txt(
    mod_params, eco_export_mapping;
    output_path=joinpath(output_dir, "ecoregion.txt"))
  if !isnothing(strat_eco_raster)
    println("Writing stratified ecoregion.tif")
    BiomassSuccessionPlugin.export_coarse_raster(
      strat_eco_raster, eco_raster_path, 1;
      output_path=joinpath(output_dir, "ecoregion.tif"),
      dtype=Int16, nodata=0)
  elseif downsample > 1 && !isnothing(coarse_eco_raster)
    println("Writing downsampled ecoregion.tif")
    BiomassSuccessionPlugin.export_coarse_raster(
      coarse_eco_raster, eco_raster_path, downsample;
      output_path=joinpath(output_dir, "ecoregion.tif"),
      dtype=Int16, nodata=0)
  else
    println("Copying ecoregion.tif")
    cp(realpath(eco_raster_path), joinpath(output_dir, "ecoregion.tif"); force=true)
    println("  ecoregion.tif")
  end

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
  if downsample > 1 && !isnothing(mapcode_raster)
    println("Exporting downsampled initial_communities.tif")
    ref_raster = joinpath(data_dir, treemap_raster)
    BiomassSuccessionPlugin.export_coarse_raster(
      mapcode_raster, ref_raster, downsample;
      output_path=joinpath(output_dir, "initial_communities.tif"),
      dtype=Int32, nodata=0)
  elseif !isnothing(combo_to_mapcode)
    println("Exporting initial_communities.tif")
    ref_raster = joinpath(data_dir, treemap_raster)
    BiomassSuccessionPlugin.export_initial_communities_tif(
      combo_to_mapcode, cn_raster, eco_raster_data, ref_raster;
      output_path=joinpath(output_dir, "initial_communities.tif"))
  end

  # 6. eco_ecocode_mapping.csv (lookup table, not read by LANDIS directly)
  BiomassSuccessionPlugin.export_eco_ecocode_mapping(
    mod_params, eco_export_mapping;
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
    data_dir="../poster/fl5_strat4_prob_spinup/",
    output_dir="../poster/fl5_strat4_prob_spinup/landis_90_2",
    eco_raster="ecoregion.tif",
    eco_ecocode_mapping="eco_ecocode_l3_mapping.csv",
    biomass_params_path="params.jld2",
    treemap_raster="FL5_22.tif",
    treemap_version=2022,
    treemap_db_path="../FIASQLITE2PGSQL/FIADB.duckdb",
    climate_config_file="../poster/fl5/biomass-climate.txt",
    downsample=3,
    stratify_eco_mixed=true,
  )
end

function landis_poster_main()
  prefix = joinpath("../poster/fl5_strat4_prob_spinup/landis_90_2")
  simulate_spatial_landis(
    output_dir="../poster/fl5_strat4_prob_spinup/pan_outputs_90_2",
    initial_communities_tif=joinpath(prefix, "initial_communities.tif"),
    ecoregion_tif=joinpath(prefix, "ecoregion.tif"),
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

# ---------------------------------------------------------------------------
# Figure: forest species composition by ecoregion (stacked bars) from the
# coalesced cohorts.duckdb. One subplot per ecoregion; a stacked bar per year
# (default 0/25/50), stacked by species. Species labelled with FIA common names.
#
# The coalesced cohorts table stores integer (eco_id, species_id) where species_id
# is ECO-LOCAL; we rebuild the sim's params from the scenario dir to resolve
# species_id → ECO_SPECIES_IDS[eco_id][species_id] → SPECIES_LIST → symbol, then
# map symbols to common names (REF_SPECIES / REF_SPECIES_GROUP).
# ---------------------------------------------------------------------------
function plot_species_composition(;
  cohorts_db::String,
  scenario_dir::String,          # exported LANDIS files (CoreSpeciesData.txt, SpeciesData.csv, SppEcoregionData.csv)
  fia_db::String,                # DB holding REF_SPECIES (+ REF_SPECIES_GROUP)
  output_path::String,
  years::Vector{Int}=[0, 25, 50],
  spp_eco_year::Int=0,
  relative::Bool=false,          # normalize each bar to fractions of total biomass
  species_order::Vector{String}=String[],  # canonical symbol order for stable colors across plots
)
  # 1. Reconstruct the sim's id scheme.
  core_sp_df = Data.load_landis_core_species(joinpath(scenario_dir, "CoreSpeciesData.txt"))
  species_df = CSV.read(joinpath(scenario_dir, "SpeciesData.csv"), DataFrame)
  spp_eco_df = Data.load_landis_spp_ecoregion(joinpath(scenario_dir, "SppEcoregionData.csv"); year=spp_eco_year)
  params = Data.make_landis_params(core_sp_df, species_df, spp_eco_df)
  SPECIES_LIST = params.SPECIES_LIST
  ECO_LIST = params.ECO_LIST
  eco_species_ids = params.ECO_SPECIES_IDS

  # 2. Aggregate biomass per (eco_id, species_id, year).
  db = DuckDB.DB(cohorts_db)
  con = DuckDB.connect(db)
  agg = DuckDB.execute(
    con,
    """
  SELECT year, eco_id, species_id, SUM(biomass) AS biomass
  FROM cohorts WHERE year IN ($(join(years, ",")))
  GROUP BY year, eco_id, species_id
"""
  ) |> DataFrame
  DuckDB.close(db)
  isempty(agg) && error("No cohorts for years $(years) in $(cohorts_db)")
  present_years = sort(unique(Int.(agg.year)))
  miss = setdiff(years, present_years)
  isempty(miss) || @warn "requested years absent in cohorts.duckdb (skipped)" absent = miss

  # 3. Common-name labels + softwood/hardwood class (REF_SPECIES_GROUP.CLASS).
  fdb = DuckDB.DB(fia_db)
  fcon = DuckDB.connect(fdb)
  ref_sp = DuckDB.execute(fcon, "SELECT r.SPECIES_SYMBOL AS sym, r.COMMON_NAME AS cn, r.SFTWD_HRDWD AS sh FROM REF_SPECIES r") |> DataFrame
  grp = DataFrame(spgrpcd=Int[], name=String[], class=String[])
  try
    grp = DuckDB.execute(fcon, "SELECT g.SPGRPCD AS spgrpcd, g.NAME AS name, g.CLASS AS class FROM REF_SPECIES_GROUP g") |> DataFrame
  catch e
    @warn "REF_SPECIES_GROUP unavailable; _GRP_ labels stay raw and their soft/hard defaults to hardwood" exception = e
  end
  DuckDB.close(fdb)
  common = Dict(uppercase(strip(String(r.sym))) => String(r.cn) for r in eachrow(ref_sp))
  sym_sh = Dict(uppercase(strip(String(r.sym))) => uppercase(strip(String(r.sh))) for r in eachrow(ref_sp) if !ismissing(r.sh))
  grpname = Dict(Int(r.spgrpcd) => String(r.name) for r in eachrow(grp))
  grpclass = Dict(Int(r.spgrpcd) => String(r.class) for r in eachrow(grp))

  # softwood (blue–green) vs hardwood (yellow–red):
  #   real species → REF_SPECIES.SFTWD_HRDWD ; _GRP_<spgrpcd> → REF_SPECIES_GROUP.CLASS ; _S/_H explicit.
  function is_soft(sym::AbstractString)
    s = uppercase(strip(sym))
    s == "_S" && return true
    s == "_H" && return false
    if startswith(s, "_GRP_")
      g = tryparse(Int, s[6:end])
      return startswith(uppercase(isnothing(g) ? "" : get(grpclass, g, "")), "S")
    end
    return get(sym_sh, s, "H") == "S"   # default (unknown) → hardwood
  end

  function label_for(sym::AbstractString)
    if startswith(sym, "_GRP_")
      g = tryparse(Int, sym[6:end])
      return (!isnothing(g) && haskey(grpname, g)) ? "Other $(grpname[g])" : "Other group $(sym[6:end])"
    elseif sym == "_H"
      return "Other hardwood"
    elseif sym == "_S"
      return "Other softwood"
    else
      cn = get(common, uppercase(strip(sym)), nothing)
      return isnothing(cn) ? sym : "$(cn) ($(sym))"
    end
  end

  # 4. id (eco-local) → global symbol → label; eco_id → name. Drop any out-of-range ids.
  neco = length(ECO_LIST)
  keep = [1 <= Int(r.eco_id) <= neco && 1 <= Int(r.species_id) <= length(eco_species_ids[Int(r.eco_id)])
          for r in eachrow(agg)]
  agg = agg[keep, :]
  agg.symbol = [string(SPECIES_LIST[Int(eco_species_ids[Int(r.eco_id)][Int(r.species_id)])]) for r in eachrow(agg)]
  agg.label = label_for.(agg.symbol)
  agg.eco = [ECO_LIST[Int(r.eco_id)] for r in eachrow(agg)]
  agg = combine(groupby(agg, [:eco, :year, :symbol, :label]), :biomass => sum => :biomass)

  # 5. Color & stack by each species' CANONICAL position so the same species gets the same
  #    color across different plots. Default order = SPECIES_LIST (identical across runs of the
  #    same scenario); pass `species_order` (a shared symbol list) for cross-scenario consistency.
  #    Any present symbol not in the order is appended deterministically (sorted) at the end.
  order = String[string(s) for s in (isempty(species_order) ? SPECIES_LIST : species_order)]
  seen = Set(order)
  for s in sort(unique(agg.symbol))
    s in seen || (push!(order, s); push!(seen, s))
  end
  # Split canonically into softwoods (blue→green) and hardwoods (yellow→red); shade each
  # species by its position WITHIN its class (over the full order, so shades are plot-stable).
  soft_all = [s for s in order if is_soft(s)]
  hard_all = [s for s in order if !is_soft(s)]
  soft_grad = CairoMakie.cgrad([:navy, :dodgerblue, :darkturquoise, :seagreen, :limegreen])
  hard_grad = CairoMakie.cgrad([:gold, :orange, :orangered, :red, :darkred])
  shade(i, n) = n <= 1 ? 0.5 : (i - 1) / (n - 1)
  color_of = Dict{String,CairoMakie.RGBAf}()
  for (i, s) in enumerate(soft_all)
    color_of[s] = CairoMakie.RGBAf(soft_grad[shade(i, length(soft_all))])
  end
  for (i, s) in enumerate(hard_all)
    color_of[s] = CairoMakie.RGBAf(hard_grad[shade(i, length(hard_all))])
  end
  # stack order: softwoods (bottom) then hardwoods, each in canonical order.
  stack_order = vcat(soft_all, hard_all)
  sym_idx = Dict(s => i for (i, s) in enumerate(stack_order))

  # 6. figure: one axis per ecoregion + shared legend column.
  ecos = sort(unique(agg.eco))
  yx = Dict(y => i for (i, y) in enumerate(present_years))
  ncols = max(1, ceil(Int, sqrt(length(ecos))))   # square-ish grid (4 ecoregions → 2×2)
  nrows = cld(length(ecos), ncols)
  f = CairoMakie.Figure(size=(360 * ncols + 340, 80 + 320 * nrows))
  CairoMakie.Label(f[0, 1:(ncols+1)],
    "Forest species composition by ecoregion" * (relative ? " (relative)" : " (biomass g/m²)");
    fontsize=16, font=:bold, tellwidth=false)

  for (ei, eco) in enumerate(ecos)
    r = cld(ei, ncols)
    c = mod1(ei, ncols)
    ax = CairoMakie.Axis(f[r, c]; title="Ecoregion: $(eco)", xlabel="year",
      ylabel=relative ? "biomass fraction" : "biomass (g/m²)",
      xticks=(collect(1:length(present_years)), string.(present_years)))
    sub = agg[agg.eco.==eco, :]
    ymap = relative ? Dict(row.year => row.ytot for row in eachrow(combine(groupby(sub, :year), :biomass => sum => :ytot))) : Dict()
    xs = Int[]
    ys = Float64[]
    stk = Int[]
    cs = CairoMakie.RGBAf[]
    for row in eachrow(sub)
      haskey(yx, Int(row.year)) || continue
      h = relative ? Float64(row.biomass) / max(ymap[row.year], eps()) : Float64(row.biomass)
      push!(xs, yx[Int(row.year)])
      push!(ys, h)
      push!(stk, sym_idx[row.symbol])
      push!(cs, color_of[row.symbol])
    end
    isempty(xs) || CairoMakie.barplot!(ax, xs, ys; stack=stk, color=cs)
  end

  # Legend: present species in stack order (softwoods then hardwoods), each with its fixed color.
  present = sort(unique(agg.symbol); by=s -> sym_idx[s])
  label_of = Dict(r.symbol => r.label for r in eachrow(agg))
  elems = [CairoMakie.PolyElement(color=color_of[s]) for s in present]
  CairoMakie.Legend(f[1:nrows, ncols+1], elems, [label_of[s] for s in present],
    "Species  (softwood: blue–green, hardwood: yellow–red)"; framevisible=false, labelsize=10)

  mkpath(dirname(output_path))
  CairoMakie.save(output_path, f)
  println("Wrote $output_path  ($(length(ecos)) ecoregions, $(length(present)) species, years $(present_years))")
  return output_path
end

function species_composition_main()
  prefix = joinpath("../poster/fl5_strat4/landis_90_2")
  plot_species_composition(
    cohorts_db=joinpath("../poster/fl5_strat4_prob_spinup/pan_outputs_90_2", "cohorts.duckdb"),
    scenario_dir=prefix,
    fia_db="../FIASQLITE2PGSQL/FIADB.duckdb",
    output_path=joinpath("../poster/fl5_strat4_prob_spinup/pan_outputs_90_2", "species_composition.png"),
    years=[0, 25, 50],
  )
end

function landis_main()
  prefix = joinpath("/home/bahaa/Downloads/", "Extension-Biomass-Succession-master/testings/CoreV8.0-BiomassSuccession7.0/")
  simulate_spatial_landis(
    output_dir="./outputs/landis_test",
    initial_communities_tif=joinpath(prefix, "initial-communities.tif"),
    ecoregion_tif=joinpath(prefix, "ecoregions.tif"),
    initial_communities_csv=joinpath(prefix, "biomass-succession_InitialCommunities.csv"),
    core_species_data=joinpath(prefix, "CoreSpeciesData.txt"),
    spp_ecoregion_data=joinpath(prefix, "SppEcoregionData.csv"),
    species_data=joinpath(prefix, "SpeciesData.csv"),
    eco_ecocode_mapping=joinpath(prefix, "eco_ecocode_mapping.csv"),
    spp_eco_year=0,
    timehorizon_years=50,
    output_every_years=1,
  )
end

end
