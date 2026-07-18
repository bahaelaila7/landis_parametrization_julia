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
using .Search: SA, LBSA, MOLBSA, CMAES, MOCMAES, IgelMOCMAES, CMAMAE, NSGA2, CCIgel
import .Data as Data
import .Spatial
import Dates
import LinearAlgebra
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
    base_seed = rand(rng, UInt64)   # per-site RNG = RNGType(hash((base_seed, i))) below → thread-count-invariant & reproducible (was seeded from per-thread RNGs, which made results depend on --threads)
    Threads.@threads :static for i in 1:nrow(df)
      @inbounds begin
        site = getsite(soa, i)
        site.active = false
        site.rng = RNGType(hash((base_seed, i)))
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
    base_seed = rand(rng, UInt64)   # per-site RNG = RNGType(hash((base_seed, i))) below → thread-count-invariant & reproducible (was seeded from per-thread RNGs, which made results depend on --threads)
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
        site.rng = RNGType(hash((base_seed, i)))
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
  save_ckpt::Bool          # persist a checkpoint this generation (MO-CMA-ES: on ANY archive change; others: on new best)
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
        if job.save_ckpt
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
  exclude_plots::Vector{NTuple{4,Int}}=NTuple{4,Int}[],
  filter_species::Vector{String}=String[],
  filter_planted::Bool=false,
  skip_disturbances::Bool=true,
  bins_idx::Vector{Int64}=1:180 .|> Int64,
  smoothing_window::Vector{FloatType}=FloatType[one(FloatType)],
  unbinned_w::Bool=false,   # Sim A: compute W as the EXACT per-year EMD (no binning, no smoothing). bins_idx applies to Sim B only.
  w_count_balance::Bool=false,          # count-balance reweight of the W1 term (survivorship under-representation fix)
  w_count_balance_mode::String="both",  # which sim(s) get reweighted: "a" / "b" / "both"
  w_count_beta::Float64=0.99,           # effective-number-of-samples temper (→1 ≈ 1/n, →0 ≈ uniform)
  rankw_mode::String="rank",            # per-(species,stratum) objective weight: "rank" (1/√ln(rank) by AGB) or "cbal_pct" (percentile-floored class-balanced by cohort count)
  rankw_beta::Float64=0.999,            # β for rankw_mode=cbal_pct
  param_split_species::AbstractDict=Dict{String,Vector{String}}(),  # {param_name => [species]} fit per-eco; others tied across ecos
  param_tier_merge::AbstractDict=Dict{String,Any}(),  # {param_name => {species => {cell => group}}} tie split site-cells per species
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
  n_folds::Int=1,
  fold_index::Int=1,
  test_frac::Float64=0.0,               # >0 ⇒ 3-way train/val/test single split (held-out test); mutually exclusive with n_folds>1 CV

  min_trees::Int=100,
  min_agb_frac::Float64=0.05,
  stratify_eco_mixed::Bool=false,
  single_ecoregion::Bool=false,
  stratify_landuse::Bool=false,
  site_class_strata::Bool=false,        # replace land_use with a per-plot site-productivity tier (from COND.SITECLCD)
  siteclass_hi_max::Int=4,              # hi = SITECLCD ≤ this, lo = above (2-way default)
  siteclass_scheme::String="2way",      # "2way" (hi/lo) or "4cell" (A=1-3,B=4,C=5,D=6-7) site-tier label
  shade_tier_csv::Union{Nothing,String}=nothing,  # species→shade_class CSV; splits grouping key by LST(1-3)/HST(4-5)
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
  w_hinge::Bool=false,
  w_hinge_pct::Float64=0.01,
  w_hinge_pct_min::Float64=2.0,
  w_hinge_pct_max::Float64=5.0,
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
  ipop_max_barren_restarts::Int=4,   # stop after this many consecutive IPOP restarts with no new best (0 ⇒ disabled)
  cmaes_archive_cap::Int=200,
  seed_archive_from::Union{Nothing,String}=nothing,
  cmaes_integer_handling::Bool=false,
  cmaes_integer_std_factor::Float64=0.3,
  cmaes_single_cov::Bool=false,   # CMA-ES family: one full covariance over ALL params (no eco×lu block split)
  nsga2_pop::Int=48, nsga2_offspring::Union{Nothing,Int}=nothing, nsga2_eta_c::Float64=20.0,
  nsga2_eta_m::Float64=20.0, nsga2_pc::Float64=0.9, nsga2_pm::Float64=-1.0, nsga2_sobol_init::Bool=true,
  igel_mu::Int=20,
  igel_sigma0::Float64=0.3,
  igel_sobol_init::Bool=true,
  igel_sobol_pool_k::Int=1,
  igel_sobol_raw_mult::Int=2,
  igel_niche_radius::Float64=0.0,
  igel_reseed_sigma::Float64=0.0,
  igel_reseed_random_frac::Float64=0.0,
  igel_maturity::Int=0,
  igel_seed_maturity::Bool=false,   # igelmo phase-1: shield the μ seeds for `igel_maturity` gens before they compete
  igel_freeze_seed_growth::Bool=false,   # igelmo: each seed lineage keeps its own frozen {D,S,B_MAX,ANPP_MAX} (needs fix_growth + seeds)
  cc_group_size::Int=2,
  cc_spec_gens::Int=20,
  cc_integ_gens::Int=20,
  cc_cycles::Int=5,
  cc_fix_others::Bool=false,
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

  # Pin BLAS to 1 thread: the search does hundreds of tiny per-offspring covariance eigendecomps per gen —
  # multi-threaded OpenBLAS spawns all cores per call and oversubscribes catastrophically at high --threads.
  LinearAlgebra.BLAS.set_num_threads(1)
  @info "search starting" julia_threads=Threads.nthreads() blas_threads=LinearAlgebra.BLAS.get_num_threads()
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
  PU.W_HINGE[] = w_hinge                          # Sim A: ±band "benefit of the doubt" forgiveness before the age CDF
  PU.W_HINGE_PCT[] = FloatType(w_hinge_pct); PU.W_HINGE_PCT_MIN[] = FloatType(w_hinge_pct_min); PU.W_HINGE_PCT_MAX[] = FloatType(w_hinge_pct_max)
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
  split_rng = (val_frac > 0.0 || n_folds > 1 || test_frac > 0.0) ? RNGType(UInt64(split_seed)) : nothing   # seeded from split_seed only ⇒ all folds share the shuffle
  splots, eco_list, species_list, eco_species_ids, splots_val_raw, splots_test_raw =
    Data.prepare_parametrization_data(; cohorts_db_path=cohorts_db_path,
      eco_field=eco_field,
      tablename=tablename,
      output_dir=tablename,
      skip_disturbances=skip_disturbances,
      spinup=spinup,
      by_subplot=by_subplot,
      val_frac=val_frac,
      split_rng=split_rng,
      n_folds=n_folds,
      fold_index=fold_index,
      test_frac=test_frac,
      min_trees=min_trees,
      min_agb_frac=min_agb_frac,
      stratify_eco_mixed=stratify_eco_mixed,
      single_ecoregion=single_ecoregion,
      stratify_landuse=stratify_landuse,
      site_class_strata=site_class_strata,
      siteclass_hi_max=siteclass_hi_max,
      siteclass_scheme=siteclass_scheme,
      shade_tier_csv=shade_tier_csv,
      filter_extent=filter_extent,
      filter_eco_field=filter_eco_field,
      filter_ecos=filter_ecos,
      filter_plots=filter_plots,
      exclude_plots=exclude_plots,
      filter_species=filter_species,
      filter_planted=filter_planted,
      RNG=rng)
  n_species = length(species_list)
  n_ecoregions = length(eco_list)
  n_plots = maximum(splots.plot_id)
  # count-balance reweight: build per-agebin W1 weights ONCE from the TRAIN reference (frozen), on the COARSE
  # age_idx bins; Sim A (per-year) looks them up via CBAL_COARSE. Applied in calculate_species_loss!.
  # RANKW mode: :rank (default, 1/√ln(rank) by AGB) or :cbal_pct (percentile-floored class-balanced by cohort
  # count — species play the role of age-bins in _set_cbal_weights!; 10% floor caps rare/empty cells).
  PU.RANKW_MODE[] = Symbol(lowercase(rankw_mode)); PU.RANKW_BETA[] = rankw_beta
  if PU.RANKW_MODE[] === :cbal_pct
    _nc = zeros(Int, n_species, length(eco_species_ids))
    for r in eachrow(splots)
      (1 <= Int(r.species_id) <= n_species && 1 <= Int(r.eco_id) <= length(eco_species_ids)) && (_nc[Int(r.species_id), Int(r.eco_id)] += 1)
    end
    PU.CELL_NCOH[] = _nc
    @info "RANKW mode = cbal_pct (percentile-floored class-balanced, β=$rankw_beta)"
  end
  PU.CBAL_ON[] = w_count_balance
  PU.CBAL_MODE[] = Symbol(lowercase(w_count_balance_mode))
  PU.CBAL_BETA[] = w_count_beta
  if w_count_balance
    PU._set_cbal_weights!(splots, PU.AgeBins(bins_idx=bins_idx .|> Int, last_bin_open=true), loss_params.age_bins, eco_species_ids, n_species; beta=w_count_beta)
    @info "Count-balance reweight ON (mode=$(PU.CBAL_MODE[]) β=$(w_count_beta)); coarse=$(length(bins_idx)) bins+open, Sim-A W bins=$(length(loss_params.age_bins.bin_widths))"
  end
  # per-(param, species) SPLIT SETS: named params are fit per-eco (split) only for the listed species;
  # every other species shares one value across all ecoregions (tied). Empty ⇒ everything split (default).
  PU.PARAM_SPLIT_SETS[] = Dict{Symbol,Set{Int}}()
  for (pname, syms) in param_split_species
    ids = Set{Int}()
    for sym in syms
      i = findfirst(==(String(sym)), species_list)
      i === nothing ? (@warn "split-set species not in tiering — skipped" param=pname species=sym) : push!(ids, i)
    end
    isempty(ids) || (PU.PARAM_SPLIT_SETS[][Symbol(pname)] = ids)
  end
  isempty(PU.PARAM_SPLIT_SETS[]) ||
    @info "Param split sets (species fit per-eco; others tied across ecos)" splits=Dict(string(k) => [species_list[i] for i in sort(collect(v))] for (k, v) in PU.PARAM_SPLIT_SETS[])
  # per-species site-cell merges: {param => {species => {cell => group_label}}} — split cells sharing a label are tied
  PU.PARAM_TIER_MERGE[] = Dict{Symbol,Dict{Int,Dict{String,String}}}()
  for (pname, spmap) in param_tier_merge
    m = Dict{Int,Dict{String,String}}()
    for (sym, cellmap) in spmap
      i = findfirst(==(String(sym)), species_list)
      i === nothing ? (@warn "tier-merge species not in tiering — skipped" param=pname species=sym) :
        (m[i] = Dict{String,String}(String(c) => String(g) for (c, g) in cellmap))
    end
    isempty(m) || (PU.PARAM_TIER_MERGE[][Symbol(pname)] = m)
  end
  isempty(PU.PARAM_TIER_MERGE[]) ||
    @info "Param tier merges (split cells tied per species)" merges=Dict(string(k) => Dict(species_list[i] => cm for (i, cm) in v) for (k, v) in PU.PARAM_TIER_MERGE[])
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
  # 3-way split: the test set is carved out of the data (so train is the intended fraction) and HELD OUT of
  # training + val monitoring. It is not fed to the optimizer; evaluate the final model on it post-hoc
  # (the diagnostic scripts reproduce the same split via split_seed/val_frac/test_frac and target the test set).
  if !isnothing(splots_test_raw)
    println("Test (HELD OUT — not used in training/val): $(length(unique(splots_test_raw.plot_id))) plots, $(nrow(splots_test_raw)) rows")
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
  _build_bmax_floor!(eco_list, species_list, eco_species_ids)   # per-(eco,species) B_MAX data floor → PU.BMAX_FLOOR (nothing if off)
  _build_anpp_floor!(eco_list, species_list, eco_species_ids)   # per-(eco,species) ANPP data floor → PU.ANPP_FLOOR (nothing if off)
  _build_prob_estab_floor!(eco_list, species_list, eco_species_ids)   # per-(eco,species) PROB_ESTAB data LOWER bound → PU.PROB_ESTAB_FLOOR (nothing if off)
  driver = search_mode == "molbsa" ? parametrize_MOLBSA :
           search_mode == "cmaes" ? parametrize_CMAES :
           search_mode == "mocmaes" ? parametrize_MOCMAES :
           search_mode == "igelmo" ? parametrize_IgelMOCMAES :
           search_mode == "cmame" ? parametrize_CMAMAE :
           search_mode == "ccigel" ? parametrize_CCIgel :
           search_mode == "nsga2" ? parametrize_NSGA2 : parametrize_LBSA
  # CMA-ES-only knobs; LBSA/MOLBSA don't accept these, so only splat them for the (MO)CMA-ES drivers.
  cmaes_kw = search_mode == "molbsa" ? (archive_cap=cmaes_archive_cap, seed_archive_from=seed_archive_from) :
             search_mode == "cmaes" ? (cmaes_lambda=cmaes_lambda, cmaes_sigma0=cmaes_sigma0, ipop=ipop, ipop_stagnation=ipop_stagnation, ipop_max_barren_restarts=ipop_max_barren_restarts, integer_handling=cmaes_integer_handling, integer_std_factor=cmaes_integer_std_factor, single_cov=cmaes_single_cov) :
             search_mode == "mocmaes" ? (cmaes_lambda=cmaes_lambda, cmaes_sigma0=cmaes_sigma0, cmaes_warmstart_seeds=cmaes_warmstart_seeds, ipop=ipop, ipop_stagnation=ipop_stagnation, ipop_max_barren_restarts=ipop_max_barren_restarts, archive_cap=cmaes_archive_cap, integer_handling=cmaes_integer_handling, integer_std_factor=cmaes_integer_std_factor, single_cov=cmaes_single_cov, seed_archive_from=seed_archive_from) :
             search_mode == "igelmo" ? (archive_cap=cmaes_archive_cap, igel_mu=igel_mu, igel_sigma0=igel_sigma0, igel_sobol_init=igel_sobol_init, igel_sobol_pool_k=igel_sobol_pool_k, igel_sobol_raw_mult=igel_sobol_raw_mult, igel_niche_radius=igel_niche_radius, igel_reseed_sigma=igel_reseed_sigma, igel_reseed_random_frac=igel_reseed_random_frac, igel_maturity=igel_maturity, igel_seed_maturity=igel_seed_maturity, igel_freeze_seed_growth=igel_freeze_seed_growth, single_cov=cmaes_single_cov) :
             search_mode == "ccigel" ? (archive_cap=cmaes_archive_cap, igel_sigma0=igel_sigma0, igel_sobol_init=igel_sobol_init, igel_niche_radius=igel_niche_radius, igel_reseed_sigma=igel_reseed_sigma, igel_maturity=igel_maturity, single_cov=cmaes_single_cov, cc_group_size=cc_group_size, cc_spec_gens=cc_spec_gens, cc_integ_gens=cc_integ_gens, cc_cycles=cc_cycles, cc_fix_others=cc_fix_others) :
             search_mode == "nsga2" ? (archive_cap=cmaes_archive_cap, nsga2_pop=nsga2_pop, nsga2_offspring=nsga2_offspring, nsga2_eta_c=nsga2_eta_c, nsga2_eta_m=nsga2_eta_m, nsga2_pc=nsga2_pc, nsga2_pm=nsga2_pm, sobol_init=nsga2_sobol_init) :
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
  cbal = PU.RANKW_MODE[] === :cbal_pct && !isempty(PU.CELL_NCOH[])        # percentile-floored class-balanced (by cohort count)
  for e in eachindex(eco_species_ids)
    ids = Int[]; for gsp in eco_species_ids[e]; _dom_included(gsp) && push!(ids, gsp); end
    isempty(ids) && continue
    if cbal
      β = PU.RANKW_BETA[]; nc = PU.CELL_NCOH[]; tot = sum(nc[gsp, e] for gsp in ids); totw = 0.0
      for gsp in ids
        pct = tot > 0 ? max(round(10 * nc[gsp, e] / tot) / 10, 0.10) : 0.10  # count-share → nearest 10%, floored at 10% (caps rare/empty cells; mirrors _set_cbal_weights!)
        w = (1 - β) / (1 - β^(pct * tot)); R[gsp, e] = FloatType(w); totw += w
      end
      totw > 0 && (@views R[:, e] ./= totw)
    else
      cells = sort!([(agb[gsp, e], gsp) for gsp in ids]; by=c -> -c[1])   # descending AGB within this stratum → rank 1 = most common
      tot = zero(FloatType)
      for (rank, (_, gsp)) in enumerate(cells)
        w = FloatType(1.0 / sqrt(log1p(rank))); R[gsp, e] = w; tot += w
      end
      tot > 0 && (@views R[:, e] ./= tot)                                # within-stratum sums to 1
    end
    sw[e] = split_size === nothing ? 1.0 : log10(1.0 + Float64(split_size[e]))  # order of magnitude of split
  end
  tw = sum(sw)
  tw > 0 && for e in eachindex(eco_species_ids); @views R[:, e] .*= FloatType(sw[e] / tw); end  # Σ = 1
  PU.RANKW[] = R
end

function fit_params(ref_soa, bio_params, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier::Int=3, t1_ref::Union{Nothing,Vector{Matrix{FloatType}}}=nothing, t2_ref=nothing, seeds::AbstractVector=[nothing], injection_dict=nothing, injection_years=Set{Int}(), t4_ref=nothing, cycle_map=nothing, n_cycles::Int=0, dual_b=nothing, disturbance_dict=nothing, disturbance_years=Set{Int}(), cache_preinject::Bool=false, work_soa=nothing)
  PU.SCALES_LOCKED[] || _set_loss_scales!(search_tier, spdf_plts, t4_ref, loss_params, eco_species_ids, n_species)   # global ratio-of-sums or per-cell denominators (skipped inside a candidate-parallel batch — set once serially by the caller)
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
    soa = work_soa === nothing ? copy_and_reseed_soa(ref_soa, seeds[ri]) : reset_soa!(work_soa, ref_soa; seed=seeds[ri])
    if init_scales[ri] != one(FloatType)       # perturb the sim-year-0 population (not scored, only propagated)
      sc = init_scales[ri]; cap = INIT_PERTURB_CAP[]
      @maybe_threads PARALLEL_SITES[] for i in 1:soa.n     # per-site independent → parallel in SoA mode
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
    # PRE-INJECTION cache (diagnostics only): pure model-grown cohorts BEFORE _inject_observed_cohorts! —
    # excludes handed values (injected recruits + disturbance-overwrites) so sim↔obs isn't circular.
    preinject_data = cache_preinject ? [Tuple{UIntType,Int,UIntType,UIntType,FloatType}[] for _ in 1:soa.n] : nothing
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
    if search_tier == 3
      # Allocation-free tier-3 loss path: per-thread per-eco accumulators (merged once after the year loop) +
      # per-thread scratch, so calculate_site_loss2_acc! allocates nothing per site/species (kills GC pressure at
      # high thread counts). Tier 5 keeps the per-site SiteLoss path above; only tier 3 uses these.
      nt = Threads.maxthreadid()
      eco3_w_t = [[zeros(FloatType, n_species) for _ in eachindex(eco_species_ids)] for _ in 1:nt]
      eco3_agb_t = [[zeros(FloatType, n_species) for _ in eachindex(eco_species_ids)] for _ in 1:nt]
      eco3_site_agb_t = [zeros(FloatType, length(eco_species_ids)) for _ in 1:nt]
      eco3_obs_t = [zeros(Int, length(eco_species_ids)) for _ in 1:nt]
      max_cohorts_scratch3 = Int(maximum(soa.refs.cohort[i+1] - soa.refs.cohort[i] for i in 1:soa.n))
      max_age_scratch3 = max_sim_year + max(1, length(loss_params.smoothing_weights) >> 1) + 5
      max_eco_nsp3 = maximum(length(e) for e in eco_species_ids)
      nbins3 = length(loss_params.age_bins.bins_idx) + (loss_params.age_bins.last_bin_open ? 1 : 0)
      perm_t3 = [Vector{Int}(undef, max_cohorts_scratch3) for _ in 1:nt]
      ages_t3 = [Vector{FloatType}(undef, max_age_scratch3) for _ in 1:nt]
      insite_t3 = [Vector{Bool}(undef, max_eco_nsp3) for _ in 1:nt]
      cdf_t3 = [Vector{FloatType}(undef, nbins3) for _ in 1:nt]
      agb_bins_t3 = [Vector{FloatType}(undef, nbins3) for _ in 1:nt]
    end
    for current_sim_year in starting_sim_year:max_sim_year
      #println("\ttimestep $(t)")
      if PAN_TIMING[] && PARALLEL_SITES[]   # only in soa mode: candidate mode runs fit_params on many threads → racy _tm_sim
        _ts = time_ns()
        PanCore.process_plugin!(soa, BiomassSuccessionPlugin.BiomassSuccession, current_sim_year; ctx=ctx.BiomassSuccession)
        _tm_sim[] += time_ns() - _ts
      else
        PanCore.process_plugin!(soa, BiomassSuccessionPlugin.BiomassSuccession, current_sim_year; ctx=ctx.BiomassSuccession)
      end
      # Sim B (free) disturbance: apply the observed exogenous biomass drop to the free sim — reduce a free-sim
      # cohort's biomass by the observed drop fraction ONLY IF it matches the disturbed cohort in SPECIES AND
      # AGE (per (plot,year,species,age)). The model responds to disturbance but does not predict it; no cohort
      # injection (stays free regen). Unmatched disturbed cohorts (the model didn't grow them) are not applied.
      if disturbance_dict !== nothing && current_sim_year in disturbance_years
        yd = disturbance_dict[current_sim_year]
        @maybe_threads PARALLEL_SITES[] for i in 1:soa.n
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
      if cache_preinject                       # snapshot pure model prediction BEFORE any injection/override
        for i in 1:soa.n
          @inbounds begin
            site = getsite(soa, i)
            site.active || continue
            if current_sim_year in site_sim_years.sim_years[site.mapcode]
              for j in 1:site.live
                push!(preinject_data[i], (site.ref_cn, current_sim_year, site.c_species[j], UIntType(site.c_age[j]), site.c_bio[j]))
              end
            end
          end
        end
      end
      if !isnothing(injection_dict) && current_sim_year in injection_years
        _inject_observed_cohorts!(soa, injection_dict[current_sim_year]; override=OVERRIDE_INJECTION[], replace=OVERRIDE_INJECTION_REPLACE[], sync=OVERRIDE_INJECTION_SYNC[])
      end
      if search_tier == 1
        @maybe_threads PARALLEL_SITES[] for i in 1:soa.n
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
        @maybe_threads PARALLEL_SITES[] for i in 1:soa.n
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
        @maybe_threads PARALLEL_SITES[] for i in 1:soa.n
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
        @maybe_threads PARALLEL_SITES[] for i in 1:soa.n
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
        # tier 3: accumulate each site's per-species (W, AGB) loss straight into the calling thread's per-eco
        # accumulators via calculate_site_loss2_acc! — no per-site SiteLoss, no per-species temporaries.
        @maybe_threads PARALLEL_SITES[] for i in 1:soa.n
          @inbounds begin
            tid = Threads.threadid()
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
              eco_id = Int(site.eco_id)
              sag = PU.calculate_site_loss2_acc!(eco3_w_t[tid][eco_id], eco3_agb_t[tid][eco_id], current_sim_year, site,
                      n_species, eco_species_ids, spdf_plt[current_sim_year], loss_params,
                      perm_t3[tid], ages_t3[tid], insite_t3[tid], cdf_t3[tid], agb_bins_t3[tid]; excluded=excluded)
              eco3_site_agb_t[tid][eco_id] += sag
              eco3_obs_t[tid][eco_id] += 1
              for j in 1:site.live
                push!(sites_data[i], (site.ref_cn, current_sim_year, site.c_species[j], UIntType(site.c_age[j]), site.c_bio[j]))
              end
            end
            if current_sim_year == last(sim_years)
              site.active = false
            end
          end
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
      # tier 3: merge per-thread per-eco accumulators, then build the per-eco breakdown. run_result is the sum
      # of the per-eco SiteLosses (= the same total the old per-year `sum(years_results)` produced, regrouped).
      for tid in 1:Threads.maxthreadid(), e in eachindex(eco_species_ids)
        eco3_w[e] .+= eco3_w_t[tid][e]
        eco3_agb[e] .+= eco3_agb_t[tid][e]
        eco3_site_agb[e] += eco3_site_agb_t[tid][e]
        eco3_obs[e] += eco3_obs_t[tid][e]
      end
      eco_losses = [PU.SiteLoss(sp_w_loss=eco3_w[e], sp_agb_loss=eco3_agb[e], site_agb_loss=eco3_site_agb[e], num_sites=eco3_obs[e], num_obs=eco3_obs[e]) for e in eachindex(eco_species_ids)]
      run_result = sum(eco_losses)
    end
    #@assert !any(isnan.(run_result.sp_w_loss)) "run NaN"
    cached_sites_state = [cohort for cohorts in (cache_preinject ? preinject_data : sites_data) for cohort in cohorts]
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

# ── Optional wall-time instrumentation (env PAN_TIMING=1). Zero cost when off: every probe is behind a single
# PAN_TIMING[] Ref read, and the only always-on work is a handful of time_ns() per GENERATION (not per site).
# Prints, per generation, where the wall goes: startup→first-gen, then ask / eval(sim vs loss+other) / tell /
# finalize(io = time BLOCKED handing checkpoints to the writer — a large io means writer backpressure, i.e. the
# ramdisk-output lever would help). sim is accumulated across the gen's candidate evals in fit_params.
const PAN_TIMING = Ref(false)
# Parallelism strategy (env PAN_PARALLEL): :soa = parallelize WITHIN each eval over sites (default; needed for
# single-eval spatial/raster runs). :candidate = parallelize the search's candidate×rep evals (each single-threaded
# on a thread-local SoA — NUMA-local, no shared-array contention, scales past one socket). Drives PARALLEL_SITES.
const PARALLEL_MODE = Ref(:soa)
const NEWBEST_STATS = Ref(false)    # env PAN_NEWBEST_STATS=1: run the diagnostic simulate_and_test(train) "Train stats"
                                    # print on each new best. Default OFF — it's a whole extra train sim per new best.
const _tm_start  = Ref(UInt64(0))   # ns at parametrize entry — for the STARTUP→first-gen span
const _tm_sim    = Ref(UInt64(0))   # ns in process_plugin! (sim), accumulated within the current generation
const _tm_io     = Ref(UInt64(0))   # ns BLOCKED on the writer put! within the current generation
_tsec(ns) = round(max(0, signed(UInt64(ns))) / 1e9, digits=3)   # signed+clamp: never throws on an underflowed diff
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

# ANALYSIS helper: re-simulate ONE params set through the FREE Sim-B path (dual_mode:b) EXACTLY as the run
# does — set the Sim-B dynamics globals from the config, build the tier-4 ref + dual_b (disturbance-scale /
# year-0 vs spinup / establishment ON / per-lineage frozen growth already baked into `params`), run
# fit_params(b_only), then bin the simulated cohorts AND the observed cohorts into the tier-4 age-bins and
# pair them. Returns a DataFrame(cyc, eco, esp, bin, sp, obs_agb, sim_agb) — the common input for the Sim-B
# scatter / sMAPE / TOST plots (so all three reflect the actual free-regen sim, not the Sim-A tier-3 path).
function resim_simB_paired(cfg, params, sp, spdf_plts, site_sim_years, spinup_cohorts, loss_params,
                           eco_list, species_list, eco_species_ids, rng)
  g(k, d) = get(cfg, k, d)
  DUAL_MODE[] = :b
  SIMB_DISTURB_ONLY[] = Bool(g("simB_disturb_only", false))
  SIMB_SPINUP[] = Bool(g("simB_spinup", true))
  TIER_B[] = Int(g("tier_b", 4))
  T4_PER_PLOT_MEAN[] = Bool(g("simB_per_plot_mean", false))
  T4_MEAN_OVER_REPS[] = Bool(g("simB_rep_mean", false))
  n_species = length(species_list)
  inj = (SIMB_DISTURB_ONLY[] || OVERRIDE_INJECTION[]) ? Data.get_injection_cohorts(sp; all_cohorts=true) : nothing
  cycle_map, n_cycles = Data.build_cycle_map(sp; cycle_years=Float64(g("cycle_years", 8)))
  n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
  t4_ref = [[zeros(FloatType, length(eco_species_ids[e]), n_bins) for _ in 1:n_cycles] for e in eachindex(eco_list)]
  for ((pid, eid), yd) in spdf_plts, (sy, gt) in yd
    cyc = get(cycle_map, (Int(pid), Int(sy)), 0); cyc == 0 && continue
    for (spe, rec) in gt.records; t4_ref[eid][cyc][spe, :] .+= diff([0f0; rec.sp_age_cdf]) .* rec.sp_agb_sum; end
  end
  dual_b = _build_dual_b(sp, eco_species_ids, eco_list, spdf_plts, loss_params, t4_ref, cycle_map, n_cycles, inj, spinup_cohorts, rng, false; b_only=true)
  msy = maximum(maximum.(filter(!isempty, site_sim_years.sim_years)))
  res = fit_params(dual_b.ref_soa, params, msy, n_species, eco_species_ids, spdf_plts, site_sim_years, false, spinup_cohorts, loss_params;
    debug=false, search_tier=3, dual_b=dual_b, seeds=[rand(rng, UInt64)])
  cached = res[1][2]; ab = loss_params.age_bins
  plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(DataFrames.select(sp, [:plot_id, :eco_id]))))
  simdf = DataFrames.DataFrame(cyc=Int[], eco=Int[], esp=Int[], bin=Int[], agb=Float64[])
  for (pid, sy, esp, age, bio) in cached
    haskey(plot2eco, Int(pid)) || continue
    cyc = get(cycle_map, (Int(pid), Int(sy)), 0); cyc == 0 && continue
    b = PU.find_age_bin(Int(ceil(Float64(age))), ab); b == 0 && continue
    push!(simdf, (cyc, plot2eco[Int(pid)], Int(esp), Int(b), Float64(bio)))
  end
  sim_agg = DataFrames.combine(DataFrames.groupby(simdf, [:cyc, :eco, :esp, :bin]), :agb => sum => :sim_agb)
  obsdf = DataFrames.DataFrame(cyc=Int[], eco=Int[], esp=Int[], bin=Int[], agb=Float64[])
  for r in eachrow(DataFrames.subset(sp, :sim_year => DataFrames.ByRow(>(0))))
    cyc = get(cycle_map, (Int(r.plot_id), Int(r.sim_year)), 0); cyc == 0 && continue
    b = PU.find_age_bin(Int(ceil(Float64(r.age_calc))), ab); b == 0 && continue
    push!(obsdf, (cyc, Int(r.eco_id), Int(r.eco_species_id), Int(b), Float64(r.agb_sum)))
  end
  obs_agg = DataFrames.combine(DataFrames.groupby(obsdf, [:cyc, :eco, :esp, :bin]), :agb => sum => :obs_agb)
  paired = DataFrames.outerjoin(obs_agg, sim_agg, on=[:cyc, :eco, :esp, :bin])
  paired.obs_agb = coalesce.(paired.obs_agb, 0.0); paired.sim_agb = coalesce.(paired.sim_agb, 0.0)
  especo2sp = Dict((Int(r.eco_id), Int(r.eco_species_id)) => Int(r.species_id) for r in eachrow(unique(DataFrames.select(sp, [:eco_id, :eco_species_id, :species_id]))))
  paired.sp = [especo2sp[(e, esp)] for (e, esp) in zip(paired.eco, paired.esp)]
  return paired, cycle_map, n_cycles
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
  have_val = !isnothing(val_ref_soa) && !SKIP_VAL[]   # PAN_SKIP_VAL disables all held-out val re-sim during training
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
function parametrize_CMAES(; ref_soa::ActiveSoA, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=3, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, cmaes_lambda::Union{Nothing,Int}=nothing, cmaes_sigma0::Float64=0.3, ipop::Bool=false, ipop_stagnation::Int=20, ipop_max_barren_restarts::Int=4, integer_handling::Bool=false, integer_std_factor::Float64=0.3, single_cov::Bool=false)
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
  have_val = !isnothing(val_ref_soa) && !SKIP_VAL[]   # PAN_SKIP_VAL disables all held-out val re-sim during training
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
  barren_restarts = 0             # consecutive IPOP restarts with no new best (barren-restart stop)
  improved_since_restart = false  # any new best found in the current IPOP epoch?
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
      improved_since_restart |= is_new_best

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
# End-of-run CV dump: collapse the archive to best / extreme-W / extreme-AGB / knee, then evaluate each on the
# HELD-OUT val set using the LIVE (this fold's train-frozen) cell-norm / RANKW / count-balance metric — so the
# held-out loss is on the SAME objective the candidates were selected under. Writes per-fold cv_front_metrics.csv
# (rep × W/AGB/total), cv_front_plots.csv (per-plot sim vs ref AGB, for TOST) and cv_eco_map.csv.
function _dump_cv_front(search_state, output_dir, eco_list, eco_species_ids, n_species, max_sim_year,
    val_ref_soa, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params,
    inj_dict_val, inj_years_val, val_dual_b, fixed_seeds, val_splots, val_injection_cohorts)
  arch = collect(search_state.archive)
  isempty(arch) && (@warn "cv dump: empty archive"; return)
  wtot(c) = sum(@view c.fx.objectives[1:2:end]); atot(c) = sum(@view c.fx.objectives[2:2:end])   # ΣW (odd) / ΣAGB (even)
  best = argmin(c -> c.fx.aggregate, arch); eW = argmin(wtot, arch); eA = argmin(atot, arch)
  ws = wtot.(arch); as = atot.(arch)
  nrm(x, v) = (hi = maximum(v); lo = minimum(v); hi > lo ? (x - lo) / (hi - lo) : 0.0)
  p1 = (nrm(wtot(eW), ws), nrm(atot(eW), as)); p2 = (nrm(wtot(eA), ws), nrm(atot(eA), as))
  d12 = hypot(p2[1] - p1[1], p2[2] - p1[2])   # knee = max perpendicular distance from the eW–eA line (normalized space)
  perp(c) = d12 <= 0 ? 0.0 : abs((p2[1]-p1[1])*(p1[2]-nrm(atot(c),as)) - (p1[1]-nrm(wtot(c),ws))*(p2[2]-p1[2])) / d12
  knee = argmax(perp, arch)
  reps = [("best", best), ("extW", eW), ("extAGB", eA), ("knee", knee)]
  obs = combine(groupby(subset(val_splots, :sim_year => ByRow(>(0))), [:plot_id, :sim_year]), :agb_sum => sum => :ref_agb)
  plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(select(val_splots, [:plot_id, :eco_id]))))
  excl = (OVERRIDE_INJECTION_DISTURBANCE[] == :exclude_overwrite && !isnothing(val_injection_cohorts) && hasproperty(val_injection_cohorts, :disturbance_drop_pct)) ?
    Set((Int(r.plot_id), Int(r.sim_year)) for r in eachrow(val_injection_cohorts) if r.disturbance_drop_pct > 0) : Set{Tuple{Int,Int}}()
  mrows = NamedTuple[]; prows = NamedTuple[]
  for (kind, c) in reps
    rr = fit_params(val_ref_soa, c.x, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years,
      spinup, val_spinup_cohorts, loss_params; debug=false, search_tier=3, injection_dict=inj_dict_val,
      injection_years=inj_years_val, dual_b=val_dual_b, seeds=fixed_seeds)
    run_result, eco_losses, idx = _agg_reps(rr)
    objs = _mo_objectives(eco_losses, eco_species_ids)
    push!(mrows, (rep_kind=kind, W=Float64(sum(objs[1:2:end])), AGB=Float64(sum(objs[2:2:end])), total=Float64(_mo_aggregate(objs, run_result)), front_size=length(arch)))
    sim = DataFrame(plot_id=Int[], sim_year=Int[], agb=Float64[])
    for (pid, sy, _esp, _a, bio) in rr[idx][2]; sy > 0 && push!(sim, (Int(pid), Int(sy), Float64(bio))); end
    paired = innerjoin(obs, combine(groupby(sim, [:plot_id, :sim_year]), :agb => sum => :sim_agb), on=[:plot_id, :sim_year])
    for row in eachrow(paired)
      (Int(row.plot_id), Int(row.sim_year)) in excl && continue
      push!(prows, (rep_kind=kind, plot_id=Int(row.plot_id), sim_year=Int(row.sim_year), eco_id=get(plot2eco, Int(row.plot_id), 0), sim_agb=Float64(row.sim_agb), ref_agb=Float64(row.ref_agb)))
    end
  end
  CSV.write(joinpath(output_dir, "cv_front_metrics.csv"), DataFrame(mrows))
  CSV.write(joinpath(output_dir, "cv_front_plots.csv"), DataFrame(prows))
  CSV.write(joinpath(output_dir, "cv_eco_map.csv"), DataFrame(eco_id=collect(1:length(eco_list)), eco_label=eco_list))
  @info "cv dump: front held-out metrics ($(length(reps)) reps, $(length(arch))-member front) → $output_dir"
end

const CV_RESELECT = Ref(false)   # when set, parametrize_MOCMAES skips the search and runs _cv_reselect_dump instead
const STORE_VAL_OBJ = Ref(true)  # when have_val, score EVERY admitted archive candidate on val (train-argmin ≠ val-argmin) → cv_val_cache.csv
const SKIP_VAL = Ref(false)      # env PAN_SKIP_VAL=1: skip ALL held-out val re-simulation during training (the per-new-best/
                                 # per-archive-member val fit_params). Speeds the search; leaves best_val_loss blank.

# Post-hoc reselection RANKED ON VALIDATION. Evaluate EVERY archive candidate (deduped by train-objective id
# across checkpoints) on the held-out val set → val (A_W,A_AGB) with frozen-train normalization. Rank each
# checkpoint's archive by its VALIDATION front area (shared-rectangle rule: min area, tie max knee, tie max
# count) → p100 = best-on-val archive; last = final checkpoint. Designate 5 positions {extreme_w, extreme_agb,
# knee, "median", best_aggregate} on each archive's VAL front. Writes cv_reselect_metrics.csv (10 rows, val + train objectives).
function _cv_reselect_dump(output_dir, eco_list, eco_species_ids, n_species, max_sim_year,
    val_ref_soa, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params,
    inj_dict_val, inj_years_val, val_dual_b, fixed_seeds, val_splots, species_list)
  ckpts = Tuple{Int,Any}[]
  for f in readdir(output_dir; join=true)
    m = match(r"search_state@(\d+)\.jld2$", f); isnothing(m) && continue
    push!(ckpts, (parse(Int, m.captures[1]), JLD2.load_object(f)))
  end
  isempty(ckpts) && (@warn "reselect: no checkpoints in $output_dir"; return)
  sort!(ckpts, by=first)
  wtot(c)=Float64(sum(@view c.fx.objectives[1:2:end])); atot(c)=Float64(sum(@view c.fx.objectives[2:2:end]))
  ckey(c)=Tuple(round.(Float64.(collect(c.fx.objectives)), digits=10))   # dedup id (non-dominated ⇒ distinct)
  val_obj(x) = begin
    rr = fit_params(val_ref_soa, x, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years,
      spinup, val_spinup_cohorts, loss_params; debug=false, search_tier=3, injection_dict=inj_dict_val,
      injection_years=inj_years_val, dual_b=val_dual_b, seeds=fixed_seeds)
    _, eco_losses, _ = _agg_reps(rr); objs = _mo_objectives(eco_losses, eco_species_ids)
    (Float64(sum(objs[1:2:end])), Float64(sum(objs[2:2:end])))
  end
  # evaluate EVERY unique archive candidate on val once
  vcache = Dict{Any,Tuple{Float64,Float64}}()
  for (_, st) in ckpts, c in collect(st.archive)
    k = ckey(c); haskey(vcache, k) || (vcache[k] = val_obj(c.x))
  end
  allv = collect(values(vcache)); Wm = minimum(p[1] for p in allv); Am = minimum(p[2] for p in allv)
  vnd(arch) = begin                                                   # val-NON-DOMINATED members, (candidate,valpt), sorted by val W asc
    prs = [(c, vcache[ckey(c)]) for c in arch]
    nd = filter(p -> !any(q -> q[2][1] <= p[2][1] && q[2][2] <= p[2][2] && q[2] != p[2], prs), prs)
    sort(nd, by = p -> (p[2][1], -p[2][2]))
  end
  function area_pts(pts)                                               # shared-rectangle area on val points
    p = sort(pts, by=q->(q[1], -q[2])); p1=p[1]; a=(p1[1]-Wm)*(p1[2]-Am)
    for i in 1:length(p)-1; a += (p[i+1][1]-p[i][1])*((p[i][2]-Am)+(p[i+1][2]-Am))/2; end
    a
  end
  function knee_pts(pts)
    length(pts) < 3 && return 0.0
    ws=[q[1] for q in pts]; as=[q[2] for q in pts]
    nw(w)=(hi=maximum(ws);lo=minimum(ws); hi>lo ? (w-lo)/(hi-lo) : 0.0); na(a)=(hi=maximum(as);lo=minimum(as); hi>lo ? (a-lo)/(hi-lo) : 0.0)
    eW=pts[argmin(ws)]; eA=pts[argmin(as)]; p1=(nw(eW[1]),na(eW[2])); pk=(nw(eA[1]),na(eA[2])); d12=hypot(pk[1]-p1[1],pk[2]-p1[2])
    d12<=0 ? 0.0 : maximum(abs((pk[1]-p1[1])*(p1[2]-na(q[2])) - (p1[1]-nw(q[1]))*(pk[2]-p1[2]))/d12 for q in pts)
  end
  ranked = sort(ckpts, by = kv -> (pts=[p[2] for p in vnd(collect(kv[2].archive))]; (area_pts(pts), -knee_pts(pts), -length(pts))))
  p100_arch = collect(ranked[1][2].archive); p100_gen = ranked[1][1]
  last_arch  = collect(ckpts[end][2].archive); last_gen = ckpts[end][1]
  function positions(arch)                       # designate 5 positions on the archive's val-NON-DOMINATED front
    nd = vnd(arch); vp = [p[2] for p in nd]; cs = [p[1] for p in nd]
    ws=[q[1] for q in vp]; as=[q[2] for q in vp]; iW=argmin(ws); iA=argmin(as); iB=argmin(ws .+ as)   # extremes + best aggregate
    n = length(nd)
    n == 1 && return [("extreme_w",cs[1],vp[1]), ("extreme_agb",cs[1],vp[1]), ("knee",cs[1],vp[1]), ("median",cs[1],vp[1]), ("best_aggregate",cs[1],vp[1])]
    nw(w)=(hi=maximum(ws);lo=minimum(ws); hi>lo ? (w-lo)/(hi-lo) : 0.0); na(a)=(hi=maximum(as);lo=minimum(as); hi>lo ? (a-lo)/(hi-lo) : 0.0)
    p1=(nw(vp[iW][1]),na(vp[iW][2])); pk=(nw(vp[iA][1]),na(vp[iA][2])); d12=hypot(pk[1]-p1[1],pk[2]-p1[2])
    M=((p1[1]+pk[1])/2,(p1[2]+pk[2])/2)
    segi(i)=(qn=(nw(vp[i][1]),na(vp[i][2])); ABx=-M[1];ABy=-M[2];d2=ABx^2+ABy^2; t=d2<=0 ? 0.0 : clamp(((qn[1]-M[1])*ABx+(qn[2]-M[2])*ABy)/d2,0,1); hypot(qn[1]-(M[1]+t*ABx),qn[2]-(M[2]+t*ABy)))
    im = argmin(segi(i) for i in 1:n)                                     # "median" = closest to midpoint→corner line
    n == 2 && return [("extreme_w",cs[iW],vp[iW]), ("extreme_agb",cs[iA],vp[iA]), ("knee",cs[im],vp[im]), ("median",cs[im],vp[im]), ("best_aggregate",cs[iB],vp[iB])]  # knee=median with 2 pts
    perpi(i)= d12<=0 ? 0.0 : abs((pk[1]-p1[1])*(p1[2]-na(vp[i][2])) - (p1[1]-nw(vp[i][1]))*(pk[2]-p1[2]))/d12
    ik = argmax(perpi(i) for i in 1:n)
    [("extreme_w",cs[iW],vp[iW]), ("extreme_agb",cs[iA],vp[iA]), ("knee",cs[ik],vp[ik]), ("median",cs[im],vp[im]), ("best_aggregate",cs[iB],vp[iB])]
  end
  n_val = length(unique(val_splots.plot_id))
  rows = NamedTuple[]
  for (atype, arch, agen) in (("p100", p100_arch, p100_gen), ("last", last_arch, last_gen))
    for (nm, c, vp) in positions(arch)
      push!(rows, (archive=atype, position=nm, A_W=vp[1], A_AGB=vp[2],
                   A_W_train=wtot(c), A_AGB_train=atot(c), n_val=n_val, sel_gen=agen, front_size=length(arch)))
    end
  end
  # p101 = union non-dominated front over EVERY unique candidate across ALL checkpoints (on val). Its members are
  # REAL candidates (each from some checkpoint) — extracted to p101_candidates/candidate_<i>/params.jld2 (ordered
  # by val A_W) for the per-candidate scatter/sMAPE/TOST scripts. cv_p101_positions.csv maps the 5 positions →
  # candidate index. sel_gen=-1 marks the synthetic union front.
  seen101 = Set{Any}(); uniq101 = Any[]
  for (_, st) in ckpts, c in collect(st.archive)
    k = ckey(c); k in seen101 && continue; push!(seen101, k); push!(uniq101, c)
  end
  p101_nd = vnd(uniq101)
  idx_of = Dict(ckey(p[1]) => i for (i, p) in enumerate(p101_nd))
  p101dir = joinpath(output_dir, "p101_candidates"); isdir(p101dir) && rm(p101dir, recursive=true)
  for (i, (c, _vp)) in enumerate(p101_nd)
    d = joinpath(p101dir, "candidate_$i"); mkpath(d); JLD2.save_object(joinpath(d, "params.jld2"), c.x)
  end
  posrows = NamedTuple[]
  for (nm, c, vp) in positions(uniq101)
    ci = get(idx_of, ckey(c), 0)
    push!(rows, (archive="p101", position=nm, A_W=vp[1], A_AGB=vp[2],
                 A_W_train=wtot(c), A_AGB_train=atot(c), n_val=n_val, sel_gen=-1, front_size=length(p101_nd)))
    push!(posrows, (position=nm, candidate=ci, A_W=vp[1], A_AGB=vp[2], A_W_train=wtot(c), A_AGB_train=atot(c)))
  end
  CSV.write(joinpath(output_dir, "cv_p101_positions.csv"), DataFrame(posrows))
  @info "p101: $(length(p101_nd)) union non-dominated candidates → $p101dir  (5 positions → cv_p101_positions.csv)"
  CSV.write(joinpath(output_dir, "cv_reselect_metrics.csv"), DataFrame(rows))
  # every checkpoint's per-candidate val objectives (for the val-ranked percentile sweep plots)
  frows = NamedTuple[]
  for (g, st) in ckpts, c in collect(st.archive)
    vp = vcache[ckey(c)]; push!(frows, (gen=g, A_W=vp[1], A_AGB=vp[2], A_W_train=wtot(c), A_AGB_train=atot(c)))
  end
  CSV.write(joinpath(output_dir, "cv_val_fronts.csv"), DataFrame(frows))
  # Save the p100 front's 5 representative candidate params (JLD2 + JSON) so per-species sims are runnable post-hoc.
  for (nm, c, vp) in positions(p100_arch)
    try
      JLD2.save_object(joinpath(output_dir, "cv_p100_$(nm).jld2"), c.x)
      PU.save_json(joinpath(output_dir, "cv_p100_$(nm).json"), c.x)
    catch e; @warn "cv_p100 param save failed" position=nm exception=(e, catch_backtrace()); end
  end
  # Per-(plot, species, year) simulated vs observed AGB for the p100 front's 5 positions on the held-out set —
  # the backbone for per-species linear plots / MAPE / per-(species,stratum) TOST. esp is eco-LOCAL; the global
  # species is eco_species_ids[eco_id][esp]. Uses the SAME representative rep _agg_reps selects (matches objective).
  try
    obs = combine(groupby(filter(r -> r.sim_year > 0, val_splots),
      [:plot_id, :sim_year, :eco_species_id, :eco_id]), :agb_sum => sum => :obs_agb)
    prows = NamedTuple[]
    for (nm, c, _vp) in positions(p100_arch)
      rr = fit_params(val_ref_soa, c.x, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years,
        spinup, val_spinup_cohorts, loss_params; debug=false, search_tier=3, injection_dict=inj_dict_val,
        injection_years=inj_years_val, dual_b=val_dual_b, seeds=fixed_seeds)
      av = _agg_reps(rr); cached = rr[av[3]][2]
      simdf = DataFrame(plot_id=Int[], sim_year=Int[], eco_species_id=Int[], sim_agb=Float64[])
      for (pid, sy, esp, _age, bio) in cached
        sy > 0 && push!(simdf, (Int(pid), Int(sy), Int(esp), Float64(bio)))
      end
      sim_agg = combine(groupby(simdf, [:plot_id, :sim_year, :eco_species_id]), :sim_agb => sum => :sim_agb)
      paired = innerjoin(obs, sim_agg, on=[:plot_id, :sim_year, :eco_species_id])
      for r in eachrow(paired)
        gsp = eco_species_ids[r.eco_id][r.eco_species_id]
        push!(prows, (position=nm, plot_id=r.plot_id, sim_year=r.sim_year, eco_id=r.eco_id,
          eco_label=eco_list[r.eco_id], species=species_list[gsp], sim_agb=r.sim_agb, obs_agb=Float64(r.obs_agb)))
      end
    end
    CSV.write(joinpath(output_dir, "cv_p100_perspecies.csv"), DataFrame(prows))
    @info "cv_p100_perspecies.csv: $(length(prows)) (position,plot,species,year) rows on val"
  catch e
    @warn "per-species dump failed" exception = (e, catch_backtrace())
  end
  @info "cv reselect (VAL-ranked): p100=@$p100_gen (best val-area), last=@$last_gen, $(length(vcache)) unique cands scored on val, n_val=$n_val → $output_dir/cv_reselect_metrics.csv (+cv_val_fronts.csv, +cv_p100_*.jld2/json)"
end

# Order-invariant signature of a MO archive: hash of its members' objective vectors, sorted so archive
# ordering doesn't matter. Two archives with the same SET of objective points hash equal; ANY membership
# change (add / drop / swap) flips it. parametrize_MOCMAES checkpoints whenever this changes, so every
# distinct archive state is pullable for the CV reselection — not just representative improvements.
_mo_archive_sig(state) = hash(sort!([collect(c.fx.objectives) for c in state.archive]))

# ── Shared per-generation checkpoint + val bookkeeping for EVERY MO search engine ────────────────────────
# MOCMAES / IgelMOCMAES / CMAMAE / MOLBSA all drive their per-generation output through `mo_gen_open` (once)
# + `mo_gen_finalize!` (every generation) so they behave identically:
#   • checkpoint search_state@N.jld2 + best_params@N.* on ANY change to the (train) Pareto archive — an add,
#     drop OR swap — not merely when the representative improves;
#   • when a val split exists, val-score every newly admitted archive member into cv_val_cache.csv (so every
#     checkpoint's val front is recoverable) and export best_val_params@N whenever the representative's val
#     loss improves;
#   • append the per-generation metrics.csv row (iteration,best_train_loss,best_val_loss,pop_size,archive_size).
# `pop_size` and the val-fit closure are the only per-engine differences and are passed in by each driver.
_vkey(c) = Tuple(round.(Float64.(collect(c.fx.objectives)), digits=10))          # dedup archive members by train objectives
_twt(c) = Float64(sum(@view c.fx.objectives[1:2:end]))                            # train W-half (Σ Wasserstein objectives)
_tat(c) = Float64(sum(@view c.fx.objectives[2:2:end]))                            # train AGB-half (Σ AGB objectives)
# val_cache key = the (Σ train-W, Σ train-AGB) pair — which IS what's persisted (cv_val_cache's first two columns)
# and what analyze_run matches on, so the cache round-trips through the CSV and can be reloaded on resume.
_rkey(c) = (round(_twt(c), digits=10), round(_tat(c), digits=10))

mutable struct MOGenIO
  output_dir::String
  writer_ch::Any
  losses_db::Any
  metrics_io::IO
  splots::Any
  eco_list::Vector{String}
  species_list::Vector{String}
  eco_species_ids::Vector{Vector{Int}}
  n_species::Int
  site_sim_years::Any
  loss_params::Any
  no_establishment::Bool
  n_reps::Int
  rng::Any
  have_val::Bool
  n_output_plots::Int
  sampled_ids::Any
  sampled_ids_val::Any
  emp_sample::Any
  emp_sample_val::Any
  val_fit::Any                      # x -> (run_result, cached_sites, eco_losses) on the val split (nothing when !have_val)
  last_ckpt_sig::UInt
  val_cache::Dict{Any,NTuple{4,Float64}}
  rep_val_loss::Float64
  rep_val_best::Float64
end

_persist_val_cache!(gio::MOGenIO) = try
  vs = collect(values(gio.val_cache))
  isempty(vs) || CSV.write(joinpath(gio.output_dir, "cv_val_cache.csv"),
    DataFrame(A_W_train=[v[1] for v in vs], A_AGB_train=[v[2] for v in vs], A_W=[v[3] for v in vs], A_AGB=[v[4] for v in vs]))
catch e; @warn "persist val cache failed" exception=(e, catch_backtrace()); end

# Reload a persisted cv_val_cache.csv into the in-memory cache on RESUME (else _persist_val_cache!'s overwrite would
# drop every pre-resume entry → the val curve would only start at the resume generation). Keyed like _rkey.
function _load_val_cache(output_dir, losses_db=nothing)
  d = Dict{Any,NTuple{4,Float64}}(); f = joinpath(output_dir, "cv_val_cache.csv")
  if isfile(f)
    try
      for r in CSV.File(f)
        d[(round(Float64(r.A_W_train), digits=10), round(Float64(r.A_AGB_train), digits=10))] =
          (Float64(r.A_W_train), Float64(r.A_AGB_train), Float64(r.A_W), Float64(r.A_AGB))
      end
      @info "Loaded $(length(d)) val-cache entries from cv_val_cache.csv (resume)"
    catch e; @warn "load val cache failed" exception=(e, catch_backtrace()); end
  end
  if isempty(d) && losses_db !== nothing       # CSV missing/empty → recover from the losses.duckdb mirror
    try
      for r in DuckDB.execute(losses_db, "SELECT A_W_train, A_AGB_train, A_W, A_AGB FROM val_cache")
        d[(round(Float64(r.A_W_train), digits=10), round(Float64(r.A_AGB_train), digits=10))] =
          (Float64(r.A_W_train), Float64(r.A_AGB_train), Float64(r.A_W), Float64(r.A_AGB))
      end
      isempty(d) || @info "Recovered $(length(d)) val-cache entries from losses.duckdb (cv_val_cache.csv absent)"
    catch e; @warn "val_cache DB recover failed (table may not exist yet)" exception=(e, catch_backtrace()); end
  end
  d
end

# Open metrics.csv (append on resume), seed rep val-loss + best_val_params@0 from the initial representative,
# and stamp the initial archive signature. `state` is the fully-built search_state; `val_fit` may be nothing.
function mo_gen_open(state; output_dir, writer_ch, losses_db, resuming::Bool, splots, eco_list, species_list,
    eco_species_ids, n_species, site_sim_years, loss_params, no_establishment, n_reps, rng, have_val,
    n_output_plots, sampled_ids, sampled_ids_val, emp_sample, emp_sample_val, val_fit)
  metrics_io = open(joinpath(output_dir, "metrics.csv"), resuming ? "a" : "w")
  resuming || println(metrics_io, "iteration,best_train_loss,best_val_loss,pop_size,archive_size")
  gio = MOGenIO(output_dir, writer_ch, losses_db, metrics_io, splots, eco_list, species_list, eco_species_ids,
    n_species, site_sim_years, loss_params, no_establishment, n_reps, rng, have_val, n_output_plots,
    sampled_ids, sampled_ids_val, emp_sample, emp_sample_val, val_fit, _mo_archive_sig(state),
    resuming ? _load_val_cache(output_dir, losses_db) : Dict{Any,NTuple{4,Float64}}(), Inf, Inf)   # RESUME: keep the pre-resume val cache (CSV, or losses.duckdb mirror if the CSV is gone)
  if have_val && val_fit !== nothing
    try
      _r, _c, _el = val_fit(state.representative.x)
      gio.rep_val_loss = _mo_val_loss((_r, nothing, _el), eco_species_ids)
      gio.rep_val_best = gio.rep_val_loss
      if isfinite(gio.rep_val_best)
        mkpath(output_dir)
        PU.save_json(joinpath(output_dir, "best_val_params@0.json"), state.representative.x)
        JLD2.save_object(joinpath(output_dir, "best_val_params@0.jld2"), state.representative.x)
      end
    catch e; @warn "initial val-loss init failed" exception=(e, catch_backtrace()); end
  end
  return gio
end

# The per-generation tail every MO engine shares. `gb_run/gb_eco/gb_cached` describe this generation's best
# offspring (for losses.duckdb + the train sim sample). Returns `save_ckpt`.
function mo_gen_finalize!(gio::MOGenIO, state, pop_size::Int, is_new_best::Bool, gb_run, gb_eco, gb_cached)
  _sig = _mo_archive_sig(state)
  save_ckpt = _sig != gio.last_ckpt_sig               # ANY archive change (add/drop/swap); improved ⟹ changed
  save_ckpt && (gio.last_ckpt_sig = _sig)
  archive_changed = save_ckpt
  _newvals = NTuple{4,Float64}[]         # val entries scored THIS gen → the writer mirrors them into losses.duckdb.val_cache
  if gio.have_val && STORE_VAL_OBJ[] && (archive_changed || is_new_best)   # val-score any NEW archive members
    _newc = [c for c in collect(state.archive) if !haskey(gio.val_cache, _rkey(c))]
    _vscore = c -> try
        _r, _c, _el = gio.val_fit(c.x)
        _o = _mo_objectives(_el, gio.eco_species_ids)
        (_rkey(c), (_twt(c), _tat(c), Float64(sum(@view _o[1:2:end])), Float64(sum(@view _o[2:2:end]))))
      catch e; @warn "val score failed" exception=(e, catch_backtrace()); nothing; end
    if PARALLEL_MODE[] == :candidate      # score new members concurrently (each val_fit is sites-serial); store after
      _vout = Vector{Any}(undef, length(_newc))
      if !isempty(_newc)
        _vout[1] = _vscore(_newc[1])      # SERIAL first: sets VAL scales (val_fit → _set_loss_scales!) before the batch
        PU.SCALES_LOCKED[] = true
        try
          Threads.@threads :static for i in 2:length(_newc); _vout[i] = _vscore(_newc[i]); end
        finally; PU.SCALES_LOCKED[] = false; end
      end
      for r in _vout; r === nothing || (gio.val_cache[r[1]] = r[2]; push!(_newvals, r[2])); end
    else
      for c in _newc; r = _vscore(c); r === nothing || (gio.val_cache[r[1]] = r[2]; push!(_newvals, r[2])); end
    end
  end
  val_sim_sample = nothing
  _losses_payload = nothing            # snapshot of per-new-best losses; the DB write is offloaded to the writer
  if is_new_best
    iter = state.best_iteration
    total = Float64(state.representative.fx.aggregate)
    @info "New best @ gen $iter | agg=$total | archive=$(length(state.archive))"
    if NEWBEST_STATS[]   # opt-in diagnostic only (PAN_NEWBEST_STATS=1) — otherwise skip this whole extra train sim
      try
        test_df = simulate_and_test(; splots=gio.splots, bio_params=state.representative.x, eco_list=gio.eco_list,
          species_list=gio.species_list, eco_species_ids=gio.eco_species_ids, loss_params=gio.loss_params,
          site_sim_years=gio.site_sim_years, M=gio.n_reps, no_establishment=gio.no_establishment, rng=gio.rng)
        println("Train stats:"); show(test_df; allrows=true, allcols=true); println()
      catch e; @warn "simulate_and_test (train) failed" exception=(e, catch_backtrace()); end
    end
    if gio.have_val && gio.val_fit !== nothing
      try
        # The representative is an archive member, so STORE_VAL_OBJ already val-scored it into val_cache. Under
        # cell-norm the aggregate is Σobjs = val_W + val_AGB, so reuse the cached score and SKIP the redundant sim —
        # unless we need the val plot sample (n_output_plots>0) or can't reconstruct (non-cell-norm): then re-sim.
        _rk = _rkey(state.representative)
        if PU.CELL_NORM[] && gio.n_output_plots == 0 && haskey(gio.val_cache, _rk)
          _cv = gio.val_cache[_rk]; gio.rep_val_loss = _cv[3] + _cv[4]
          @info "Val loss @ gen $iter | loss=$(gio.rep_val_loss) (cached)"
        else
          _r, _c, _el = gio.val_fit(state.representative.x)
          gio.rep_val_loss = _mo_val_loss((_r, nothing, _el), gio.eco_species_ids)
          @info "Val loss @ gen $iter | loss=$(gio.rep_val_loss)"
          val_sim_sample = gio.n_output_plots > 0 ? _filter_cached_to_df(_c, gio.sampled_ids_val) : nothing
        end
        if gio.rep_val_loss < gio.rep_val_best
          gio.rep_val_best = gio.rep_val_loss
          try
            mkpath(gio.output_dir)
            PU.save_json(joinpath(gio.output_dir, "best_val_params@$(state.i).json"), state.representative.x)
            JLD2.save_object(joinpath(gio.output_dir, "best_val_params@$(state.i).jld2"), state.representative.x)
            @info "New best VAL @ gen $(state.i) | val=$(gio.rep_val_loss) train=$total → best_val_params@$(state.i)"
          catch e; @warn "best_val checkpoint failed" exception=(e, catch_backtrace()); end
        end
      catch e; @warn "val fit_params failed" exception=(e, catch_backtrace()); end
    end
    # snapshot the losses as plain values on the MAIN thread; the batched DB write runs on the writer thread
    _blob = let buf = IOBuffer(); Serialization.serialize(buf, state.representative.x); take!(buf) end
    _losses_payload = (iter=iter, ns=gb_run.num_sites, no=gb_run.num_obs, total=total, arch=length(state.archive), blob=_blob,
      eco=[(name=_eco_name(gio.eco_list, eco_id), ns=el.num_sites, no=el.num_obs, tot=convert(Float64, PU.get_total_loss(el)),
            sp=[(s=gio.species_list[gsp], w=el.sp_w_loss[gsp]/max(1, el.num_sites), a=el.sp_agb_loss[gsp]/max(1, el.num_sites))
                for gsp in 1:gio.n_species if el.sp_w_loss[gsp] != 0f0])
           for (eco_id, el) in enumerate(gb_eco)])
  end
  # OFFLOAD to the async writer: metrics line (every gen) + BATCHED losses insert (one transaction, new-best only).
  # No fsync on the search's critical path; drained/flushed on normal exit AND interrupt by stop_writer's finally.
  _mline = string(state.i, ",", Float64(state.representative.fx.aggregate), ",", isfinite(gio.rep_val_loss) ? gio.rep_val_loss : "", ",", pop_size, ",", length(state.archive))
  _lossjob = MOLossesJob(gio.losses_db, gio.metrics_io, _mline, _losses_payload, _newvals)
  if PAN_TIMING[]; _tio = time_ns(); put!(gio.writer_ch, _lossjob); _tm_io[] += time_ns() - _tio; else; put!(gio.writer_ch, _lossjob); end
  cached_sites_state_df = gb_cached === nothing ?    # nothing when n_output_plots==0 (candidate mode skips the best re-sim); the MO writer never reads this field anyway
    DataFrame(plot_id=Int[], sim_year=Int[], species_id=Int[], age=Int[], agb=Float64[]) :
    DataFrame(gb_cached, [:plot_id, :sim_year, :species_id, :age, :agb])
  sim_sample = (is_new_best && gio.n_output_plots > 0) ? _filter_cached_to_df(gb_cached, gio.sampled_ids) : nothing
  # The async writer above persists search_state@N + best_params@N whenever save_ckpt is set, and
  # save_ckpt == archive_changed, so EVERY archive-changed state is already saved by the writer. (A former
  # synchronous save for the changed-but-not-improved case was redundant and raced the writer on the same
  # path — removed.)
  _ckptjob = WriterJob(save_ckpt, deepcopy(state), gio.splots, cached_sites_state_df, gio.emp_sample, sim_sample, is_new_best ? gio.emp_sample_val : nothing, val_sim_sample)  # deepcopy (CPU) built before the timed put! so io = channel BLOCK only
  if PAN_TIMING[]; _tio = time_ns(); put!(gio.writer_ch, _ckptjob); _tm_io[] += time_ns() - _tio; else; put!(gio.writer_ch, _ckptjob); end
  (gio.have_val && STORE_VAL_OBJ[] && (is_new_best || archive_changed)) && _persist_val_cache!(gio)
  return save_ckpt
end

function parametrize_MOCMAES(; ref_soa::ActiveSoA, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, cmaes_lambda::Union{Nothing,Int}=nothing, cmaes_sigma0::Float64=0.3, cmaes_warmstart_seeds::Int=20, ipop::Bool=false, ipop_stagnation::Int=20, ipop_max_barren_restarts::Int=4, integer_handling::Bool=false, integer_std_factor::Float64=0.3, single_cov::Bool=false, seed_archive_from::Union{Nothing,String}=nothing)
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

  have_val = !isnothing(val_ref_soa) && !SKIP_VAL[]   # PAN_SKIP_VAL disables all held-out val re-sim during training
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

  if CV_RESELECT[]   # skip the search; just re-score the 10 reselection positions (5 per archive) on the (frozen-norm) val set
    have_val || (@warn "CV_RESELECT set but no val split — nothing to do"; return)
    _cv_reselect_dump(output_dir, eco_list, eco_species_ids, n_species, max_sim_year,
      val_ref_soa, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params,
      inj_dict_val, inj_years_val, val_dual_b, fixed_seeds, val_splots, species_list)
    return
  end

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
    _floor_bmax_seed!(bio_params)          # raise any seed B_MAX below its per-(eco,species) data floor
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
      _floor_bmax_seed!(bio_params)
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
  # metrics.csv, initial rep val-loss seeding, best_val_params@0 and the per-generation checkpoint/val bookkeeping
  # are all handled by the shared mo_gen_open / mo_gen_finalize! (below), same as every other MO engine.
  # VAL REPLAY (ENV VAL_REPLAY=1): recompute val for every saved best_params@N.jld2 in output_dir — recovers
  # val for runs where it was never computed live (the Sim-A val regression). Reuses the val setup above; NO
  # search, does not touch losses.duckdb. Writes val_replay.csv (iteration,val_loss) and returns.
  if have_val && get(ENV, "VAL_REPLAY", "") == "1"
    ckpts = filter(f -> occursin(r"^best_params@\d+\.jld2$", f), readdir(output_dir))
    sort!(ckpts, by = f -> parse(Int, match(r"@(\d+)", f).captures[1]))
    @info "VAL_REPLAY: recomputing val for $(length(ckpts)) checkpoints in $output_dir"
    open(joinpath(output_dir, "val_replay.csv"), "w") do io
      println(io, "iteration,val_loss")
      for f in ckpts
        it = parse(Int, match(r"@(\d+)", f).captures[1])
        try
          p = JLD2.load_object(joinpath(output_dir, f))
          _vr = _agg_reps(fit_params(val_ref_soa, p, max_sim_year, n_species, eco_species_ids,
            val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params;
            debug=false, search_tier=3, injection_dict=inj_dict_val, injection_years=inj_years_val,
            dual_b=val_dual_b, seeds=fixed_seeds))
          vl = _mo_val_loss((_vr[1], nothing, _vr[2]), eco_species_ids)
          println(io, "$it,$vl"); flush(io); @info "val-replay @ $it = $vl"
        catch e
          @warn "val-replay failed @ $it" exception = (e, catch_backtrace())
        end
      end
    end
    @info "VAL_REPLAY done → $(joinpath(output_dir, "val_replay.csv"))"
    return nothing
  end

  caused_by_interrupt(e) =
    e isa InterruptException ? true :
    e isa TaskFailedException ? any(en -> caused_by_interrupt(en.exception), Base.current_exceptions(e.task)) :
    e isa CompositeException ? any(caused_by_interrupt, e.exceptions) :
    false

  stagnation = 0
  barren_restarts = 0             # consecutive IPOP restarts with no new best (barren-restart stop)
  improved_since_restart = false  # any new best found in the current IPOP epoch?
  val_fit = have_val ? function (x)            # (run, cached, eco_losses) on the val split, aggregated over n_reps like train
      vreps = fit_params(val_ref_soa, x, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params; debug=false, search_tier=3, injection_dict=inj_dict_val, injection_years=inj_years_val, dual_b=val_dual_b, seeds=fixed_seeds)
      v = _agg_reps(vreps); (v[1], vreps[v[3]][2], v[2])
    end : nothing
  gio = mo_gen_open(search_state; output_dir=output_dir, writer_ch=writer_ch, losses_db=losses_db,
    resuming=!isnothing(resume_from), splots=splots, eco_list=eco_list, species_list=species_list,
    eco_species_ids=eco_species_ids, n_species=n_species, site_sim_years=site_sim_years, loss_params=loss_params,
    no_establishment=no_establishment, n_reps=n_reps, rng=rng, have_val=have_val, n_output_plots=n_output_plots,
    sampled_ids=sampled_ids, sampled_ids_val=sampled_ids_val, emp_sample=emp_sample, emp_sample_val=emp_sample_val, val_fit=val_fit)
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
    close(losses_db_file); close(gio.metrics_io)
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

  have_val && try   # CV: held-out front metrics under THIS fold's live (train-frozen) loss — see _dump_cv_front
    _dump_cv_front(search_state, output_dir, eco_list, eco_species_ids, n_species, max_sim_year,
      val_ref_soa, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params,
      inj_dict_val, inj_years_val, val_dual_b, fixed_seeds, val_splots, val_injection_cohorts)
  catch e; @error "cv front dump failed" exception = (e, catch_backtrace()); end

  return search_state
end

# Igel/Hansen/Roth (2007) population-based MO-CMA-ES driver. Same MO objective vector, archive and
# writer as parametrize_MOCMAES, but the engine is a population of μ (1+1)-CMA-ES individuals
# (IgelMOCMAES) rather than one distribution — better front spread/extreme coverage. μ candidates
# are evaluated per generation (serially); the initial population is a Sobol design.
function parametrize_IgelMOCMAES(; ref_soa::ActiveSoA, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, igel_mu::Int=20, igel_sigma0::Float64=0.3, igel_sobol_init::Bool=true, igel_sobol_pool_k::Int=1, igel_sobol_raw_mult::Int=2, igel_niche_radius::Float64=0.0, igel_reseed_sigma::Float64=0.0, igel_reseed_random_frac::Float64=0.0, igel_maturity::Int=0, igel_seed_maturity::Bool=false, igel_freeze_seed_growth::Bool=false, single_cov::Bool=false)
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
  have_val = !isnothing(val_ref_soa) && !SKIP_VAL[]   # PAN_SKIP_VAL disables all held-out val re-sim during training
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
  # ONE reusable working SoA per reference — candidates/reps run serially, so reset_soa! reuses these
  # buffers each eval instead of deep-copying the reference (removes the serial per-eval deepcopy + its churn).
  _work_soa = deepcopy(ref_soa)
  _work_soa_val = have_val ? deepcopy(val_ref_soa) : nothing
  # candidate mode: one work SoA PER THREAD (each candidate eval runs single-threaded on its own, NUMA-local)
  _work_soa_t = PARALLEL_MODE[] == :candidate ? [deepcopy(ref_soa) for _ in 1:Threads.maxthreadid()] : nothing
  _run(p; ws=_work_soa) = fit_params(ref_soa, p, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params; debug, search_tier, t1_ref, t2_ref, t4_ref, cycle_map, n_cycles, seeds=fixed_seeds, injection_dict=injection_dict, injection_years=injection_years, dual_b=dual_b, work_soa=ws)
  # Candidate mode: sites go serial for EVERY eval in this driver (init population, RANKW freeze, gen loop, val),
  # so each eval is single-threaded/deterministic and the gen loop parallelizes over candidates. Restored in `finally`.
  PARALLEL_MODE[] == :candidate && (PARALLEL_SITES[] = false)
  function _fitness(rep_results)
    run_result, eco_losses, _ = _agg_reps(rep_results)
    objs = _mo_objectives(eco_losses, eco_species_ids)
    MOLBSA.MOFitness(objs, _mo_aggregate(objs, run_result)), run_result, eco_losses
  end
  if isnothing(resume_from)
    template = !isnothing(start_from) ? _load_params_from_path(start_from) : BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    slots = PU.build_slots(param_dists, template)
    # initial population — three sources, all feeding the same "best μ·pool_k by dominance → RANDOM μ" thinning
    # (pool_k>1) that gives a good-but-spread start instead of seeding all μ at the single best spot (→ collapse):
    #   • sobol_candidates_db → params PRE-evaluated + dominance-sorted best-first by load_sobol_candidates
    #     (generated offline by search_mode="sobol", which runs the SAME fit_params simulator + _mo_objectives as
    #     this run). Cut uses the stored ranking — NO re-eval of the whole pool; only the selected μ are evaluated.
    #   • igel_sobol_init     → a fresh Sobol design generated here; OVERSAMPLED to (μ·pool_k)·raw_mult, all
    #     evaluated, then dominance-cut + random-thinned below (inline_cut).
    #   • else                → random generated params (no pooling).
    do_pool = igel_sobol_pool_k > 1
    elite_n = igel_mu * igel_sobol_pool_k
    inline_cut = false
    pool_params =
      if !isempty(_sobol_cands)
        best = _sobol_cands[1:min(elite_n, length(_sobol_cands))]                      # already dominance-sorted best-first
        do_pool ? Random.shuffle(rng, best)[1:min(igel_mu, length(best))] : _sobol_cands[1:min(igel_mu, length(_sobol_cands))]
      elseif igel_sobol_init
        inline_cut = do_pool
        PU.sobol_samples(param_dists, template, do_pool ? elite_n * igel_sobol_raw_mult : igel_mu)
      else
        [BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment) for _ in 1:igel_mu]
      end
    # Seed the ESTABLISHMENT params from the data tables (prob_estab_from_data / maturity_from_data) and pin
    # MIN_REL (fix_min_rel) — mirroring parametrize_MOCMAES. Applied to the template AND every pool member BEFORE
    # eval/params_to_u, so the encoded u and the frozen `bases` carry the seeded/pinned values. Growth
    # ({D,S,B_MAX,ANPP_MAX}) is untouched. Each fn is a no-op unless its flag/table is set.
    for _p in Iterators.flatten(((template,), pool_params))
      _seed_prob_estab!(_p, eco_list, species_list, eco_species_ids)
      _seed_maturity!(_p, species_list)
      _seed_min_rel!(_p, eco_list)
    end
    # evaluate under THIS run's exact loss config: the raw pool if inline_cut, else the already-selected μ.
    np = length(pool_params); pool_fx = Vector{Any}(undef, np)
    if PARALLEL_MODE[] == :candidate     # candidate-parallelize the eval (sites already serial here)
      _set_loss_scales!(search_tier, spdf_plts, t4_ref, loss_params, eco_species_ids, n_species)  # SERIAL: freeze RANKW + set scales before the parallel batch (no race on the scale/RANKW globals)
      PU.SCALES_LOCKED[] = true
      try
        Threads.@threads :static for k in 1:np
          pool_fx[k] = _fitness(_run(pool_params[k]; ws=_work_soa_t[Threads.threadid()]))[1]
        end
      finally; PU.SCALES_LOCKED[] = false; end
    else
      for k in 1:np; pool_fx[k] = _fitness(_run(pool_params[k]))[1]; end
    end
    # inline Sobol only: cut the freshly-evaluated raw pool to best μ·pool_k by dominance, then UNIFORM-RANDOM μ.
    # (DB / random / non-pool sources already hold exactly the μ starting individuals.)
    sel = if inline_cut
      elite = NSGA2._select(Vector{FloatType}[fx.objectives for fx in pool_fx], min(elite_n, np))
      chosen = Random.shuffle(rng, elite)[1:min(igel_mu, length(elite))]
      @info "igelmo Sobol-pool init: evaluated $np → kept best $(length(elite)) by dominance → random-selected $(length(chosen)) starting individuals"
      chosen
    else
      collect(1:np)
    end
    init_params = pool_params[sel]
    init_us = [PU.params_to_u(p, param_dists, slots) for p in init_params]
    init_cands = [MOLBSA.MOCandidate(init_params[k], pool_fx[sel[k]]) for k in eachindex(sel)]
    groups = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(param_dists, slots, BSP.BIOMASS_PER_ECO_GROUPS))   # block-diagonal per-individual (1+1)-CMA
    @info "Igel MO-CMA-ES block-diagonal: $(length(groups)) covariance blocks per individual (sizes $(length.(groups)))"
    search_state = IgelMOCMAES.IgelState(init_us, init_cands, rng; sigma0=igel_sigma0, archive_cap=archive_cap, max_iter=typemax(Int), niche_radius=igel_niche_radius, reseed_sigma=igel_reseed_sigma, maturity_period=igel_maturity, init_mature=igel_seed_maturity, blocks=groups)
    igel_seed_maturity && igel_maturity > 0 && @info "igelmo phase-1: $igel_mu seeds shielded for $igel_maturity gens, then (μ+μ) competition"
    search_state.n_evals = np
    bio_params = template
    # PER-LINEAGE FROZEN GROWTH: each seed keeps its OWN {D,S,B_MAX,ANPP_MAX} (fix_growth removes them from the
    # slots ⇒ they come from the per-lineage `base` handed to u_to_params). `bases[k]` tracks lineage k's seed;
    # aligned to search_state.pop[k] here (pop is built from init_us in order) and re-aligned by object identity
    # after every tell! (below). Sidecar-persisted for resume.
    bases = igel_freeze_seed_growth ? copy(init_params) : nothing
    if igel_freeze_seed_growth
      BSP.FIX_GROWTH[] || @warn "igel_freeze_seed_growth is on but fix_growth is OFF → growth is in the search slots and will NOT stay frozen"
      # _sobol_cands is already drained into init_params by here, so check the seed SOURCE instead.
      (isnothing(sobol_candidates_db) && isnothing(start_from)) && @warn "igel_freeze_seed_growth is on but no seed source (sobol_candidates_db/start_from) → freezing RANDOM growth"
      @info "igelmo per-lineage frozen growth: $igel_mu lineages each keep their seed's {D,S,B_MAX,ANPP_MAX}"
    end
  else
    @info "Resuming from $resume_from"
    search_state = JLD2.load_object(resume_from)
    bio_params = search_state.representative.x
    slots = PU.build_slots(param_dists, bio_params)
    # Freeze the cbal RANKW / CELL_NORM on the TRAIN reference (a throwaway train eval), mirroring what the
    # init-population evals do on a fresh run. Without this, mo_gen_open's val-score below runs the FIRST eval
    # on the VAL reference and freezes RANKW on val's plot counts → a different per-cell weighting than the
    # original run → identical offspring evaluate differently → the Pareto front collapses on resume.
    PU.CELL_NORM_FREEZE[] || (_run(search_state.representative.x); @info "RANKW frozen on TRAIN reference (resume)")
    if igel_freeze_seed_growth
      _bpath = joinpath(output_dir, "igel_bases@$(search_state.i).jld2")
      isfile(_bpath) || error("igel_freeze_seed_growth resume: missing sidecar $_bpath (per-lineage frozen growth cannot be reconstructed from the checkpoint alone)")
      bases = JLD2.load_object(_bpath)
      length(bases) == length(search_state.pop) || error("igel_bases sidecar has $(length(bases)) entries but pop has $(length(search_state.pop))")
      @info "igelmo per-lineage frozen growth: reloaded $(length(bases)) lineage bases from $(basename(_bpath))"
    else
      bases = nothing
    end
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
  # Held-out val must measure the SAME objective as train: Sim B (dual_b) when dual_mode≠off, else Sim A.
  # (val_dual_b = nothing when DUAL_MODE==:off ⇒ val_fit falls back to the tier-3 Sim-A path.) Previously this
  # driver hardcoded dual_b=nothing here, so a dual_mode:b run measured VAL on Sim A — inconsistent with the
  # Sim-B train loss (and sensitive to init_perturb via the Sim-A rep-combination). Fixed to use val_dual_b.
  val_dual_b = have_val ? _build_val_dual_b(val_splots, eco_species_ids, eco_list, val_spdf_plts, loss_params, cycle_years, val_injection_cohorts, val_spinup_cohorts, rng, no_establishment) : nothing
  # candidate mode: per-thread val buffers so val_fit can run concurrently across archive members (val is sites-serial)
  _val_work_t = (have_val && PARALLEL_MODE[] == :candidate) ? [deepcopy(val_ref_soa) for _ in 1:Threads.maxthreadid()] : nothing
  val_fit = have_val ? function (x)            # (run, cached, eco_losses) on the val split, aggregated over n_reps like train
      _vws = _val_work_t === nothing ? _work_soa_val : _val_work_t[Threads.threadid()]
      vreps = fit_params(val_ref_soa, x, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params; debug=false, search_tier=3, injection_dict=inj_dict_val, injection_years=inj_years_val, dual_b=val_dual_b, seeds=fixed_seeds, work_soa=_vws)
      v = _agg_reps(vreps); (v[1], vreps[v[3]][2], v[2])
    end : nothing
  gio = mo_gen_open(search_state; output_dir=output_dir, writer_ch=writer_ch, losses_db=losses_db,
    resuming=!isnothing(resume_from), splots=splots, eco_list=eco_list, species_list=species_list,
    eco_species_ids=eco_species_ids, n_species=n_species, site_sim_years=site_sim_years, loss_params=loss_params,
    no_establishment=no_establishment, n_reps=n_reps, rng=rng, have_val=have_val, n_output_plots=n_output_plots,
    sampled_ids=sampled_ids, sampled_ids_val=sampled_ids_val, emp_sample=emp_sample, emp_sample_val=emp_sample_val, val_fit=val_fit)
  caused_by_interrupt(e) = e isa InterruptException ? true :
    e isa TaskFailedException ? any(en -> caused_by_interrupt(en.exception), Base.current_exceptions(e.task)) :
    e isa CompositeException ? any(caused_by_interrupt, e.exceptions) : false

  evals_done = search_state.n_evals
  est_gens = max(1, cld(TRIALS - evals_done, igel_mu))
  done_gens = isnothing(resume_from) ? 0 : search_state.i    # gens already completed (search_state.i == checkpoint/metrics counter)
  total_gens = done_gens + est_gens                          # size the bar to the WHOLE run so resume continues the counter/bar
  IgelMOCMAES.RESEED_RANDOM_FRAC[] = igel_reseed_random_frac  # module Ref (not a state field) → applies on fresh AND resume
  # resume-safe reseed/σ trace → a SEPARATE csv (NOT metrics.csv, which appends under a fixed header): a stop→resume
  # just appends; header is written only when the file is new/empty. Columns: gen, mean σ, min σ, #below reseed floor, #reseeds.
  _reseed_path = joinpath(output_dir, "reseed_trace.csv"); _reseed_new = !isfile(_reseed_path) || filesize(_reseed_path) == 0
  _reseed_io = open(_reseed_path, _reseed_new ? "w" : "a")
  _reseed_new && (println(_reseed_io, "iteration,mean_sigma,min_sigma,n_below_floor,n_reseed"); flush(_reseed_io))
  try
    _first_gen = true
    TProgress.@track for _gen in 1:total_gens                # _gen = ABSOLUTE generation number
      _gen <= done_gens && continue                          # RESUME: fast-skip completed gens (bar jumps to the resumed position, then continues)
      evals_done >= TRIALS && break
      if PAN_TIMING[]
        _first_gen && (println("[timing] startup→first gen: $(_tsec(time_ns()-_tm_start[]))s"); _first_gen = false)
        _tm_sim[] = 0; _tm_io[] = 0
      end
      _tg = time_ns()
      offs = IgelMOCMAES.ask(search_state)
      _t_ask = time_ns() - _tg
      off_fxs = Vector{MOLBSA.MOFitness}(undef, igel_mu)
      off_params = Vector{typeof(bio_params)}(undef, igel_mu)
      off_run = Vector{Any}(undef, igel_mu); off_eco = Vector{Any}(undef, igel_mu)   # retained per-candidate loss/eco (cheap) → gen-best needs no serial re-sim
      gen_best_agg = Inf; local gb_run, gb_eco, gb_cached, gb_idx
      _te = time_ns()
      if PARALLEL_MODE[] == :candidate
        # candidate×rep is the parallel unit: each candidate's sim runs single-threaded on a thread-local SoA
        # (NUMA-local, no shared-array contention). Sites are already serial (PARALLEL_SITES=false, set above).
        _set_loss_scales!(search_tier, spdf_plts, t4_ref, loss_params, eco_species_ids, n_species)  # SERIAL: set TRAIN scales before the batch (no race on scale globals)
        PU.SCALES_LOCKED[] = true
        try
          Threads.@threads :static for k in 1:igel_mu
            p = PU.u_to_params(offs[k], param_dists, slots, bases === nothing ? bio_params : bases[k])
            off_params[k] = p
            fx_k, run_k, eco_k = _fitness(_run(p; ws=_work_soa_t[Threads.threadid()]))
            off_fxs[k] = fx_k; off_run[k] = run_k; off_eco[k] = eco_k   # keep the batch's run/eco (self-contained loss structs; no SoA aliasing)
          end
        finally; PU.SCALES_LOCKED[] = false; end
        evals_done += igel_mu
        for k in 1:igel_mu                              # serial best-pick (gen argmin aggregate)
          off_fxs[k].aggregate < gen_best_agg && (gen_best_agg = off_fxs[k].aggregate; gb_idx = k)
        end
        gb_run = off_run[gb_idx]; gb_eco = off_eco[gb_idx]  # already computed in the parallel batch (identical to a re-sim: fixed seeds) — no serial full-sim
        gb_cached = n_output_plots > 0 ? _median_rep_cached(_run(off_params[gb_idx]; ws=_work_soa_t[1])) : nothing  # cohort cache only feeds output plots; unused otherwise
      else
        for k in 1:igel_mu                              # SoA mode: serial candidates, sites parallel (fit_params threads)
          p = PU.u_to_params(offs[k], param_dists, slots, bases === nothing ? bio_params : bases[k])
          rep_results = _run(p); fx_k, run_k, eco_k = _fitness(rep_results)
          off_params[k] = p; off_fxs[k] = fx_k; evals_done += 1
          if fx_k.aggregate < gen_best_agg
            gen_best_agg = fx_k.aggregate; gb_idx = k; gb_run = run_k; gb_eco = eco_k; gb_cached = _median_rep_cached(rep_results)
          end
        end
      end
      _t_eval = time_ns() - _te; _sim = _tm_sim[]
      _tt = time_ns()
      # capture the parent Individual objects BEFORE tell! reorders the population; offspring k descends from
      # parent k, so both carry lineage k's frozen growth (bases[k]). After selection the same objects survive
      # by reference, so we realign bases by object identity — no change to IgelMOCMAES needed.
      _pre_pop = bases === nothing ? nothing : copy(search_state.pop)
      is_new_best = IgelMOCMAES.tell!(search_state, off_fxs, off_params)
      if bases !== nothing
        objbase = IdDict{Any,Any}()
        for k in 1:igel_mu; objbase[_pre_pop[k]] = bases[k]; objbase[search_state._off[k]] = bases[k]; end
        bases = Any[objbase[ind] for ind in search_state.pop]   # survivors → their lineage's frozen growth
      end
      _t_tell = time_ns() - _tt
      search_state.n_evals = evals_done
      _tf = time_ns()
      _save_ckpt = mo_gen_finalize!(gio, search_state, igel_mu, is_new_best, gb_run, gb_eco, gb_cached)   # shared: ckpt on ANY archive change (train+val)
      # sidecar the per-lineage bases WHENEVER a checkpoint is saved, tagged with the same gen so resume loads
      # the exact pop↔bases alignment (search_state.pop is not mutated between here and the next ask).
      bases !== nothing && _save_ckpt && try; JLD2.save_object(joinpath(output_dir, "igel_bases@$(search_state.i).jld2"), bases); catch e; @warn "igel_bases sidecar save failed" exception=e; end
      let _sg = Float64[ind.sigma for ind in search_state.pop], _nr = count(search_state._reseed)   # reseed/σ signal
        println(_reseed_io, string(search_state.i, ",", round(search_state.t, sigdigits=6), ",", round(minimum(_sg), sigdigits=6), ",", count(<(igel_reseed_sigma), _sg), ",", _nr))
        (_nr > 0 || search_state.i % 100 == 0) && flush(_reseed_io)
        _nr > 0 && @info "igelmo gen $(search_state.i): $_nr reseed(s) — mean_σ=$(round(search_state.t, sigdigits=3)) min_σ=$(round(minimum(_sg), sigdigits=3))"
      end
      if PAN_TIMING[]
        println("[timing] gen $_gen: $(_tsec(time_ns()-_tg))s | ask=$(_tsec(_t_ask)) eval=$(_tsec(_t_eval))(sim=$(_tsec(_sim)) loss+oth=$(_tsec(_t_eval-_sim))) tell=$(_tsec(_t_tell)) final=$(_tsec(time_ns()-_tf))(io=$(_tsec(_tm_io[])))")
      end
    end
  catch e
    caused_by_interrupt(e) ? @info("Search interrupted by user @ gen $(search_state.i); finalizing checkpoint…") : rethrow()
  finally
    PARALLEL_SITES[] = true   # restore site-parallelism (candidate mode set it false for this driver)
    stop_writer(writer_ch, writer_task); close(losses_db_file); try; close(gio.metrics_io); catch; end
    try; close(_reseed_io); catch; end
    try
      mkpath(output_dir); fname = "search_state@$(search_state.i).jld2"
      JLD2.save_object(joinpath(output_dir, fname), search_state)
      bases !== nothing && JLD2.save_object(joinpath(output_dir, "igel_bases@$(search_state.i).jld2"), bases)   # keep the sidecar aligned with the exit checkpoint
      link_path = joinpath(output_dir, "search_state_latest.jld2"); islink(link_path) && rm(link_path); symlink(fname, link_path)
      @info "Search state saved @ $(search_state.i)"
    catch e; @error "Failed to save search state on exit" exception = (e, catch_backtrace()); end
  end
  return search_state
end

# Cooperative-Coevolutionary Igel MO-(1+1)-CMA-ES ("ccigel"). Clone of parametrize_IgelMOCMAES with a CC
# layer (see search/CCIgel.jl): population = n_species × K whole individuals in per-species groups. It
# alternates SPECIALIZATION phases (each species group runs a (1+1)-CMA restricted to that species' u-slots,
# scored on the species' own 2-objective loss, NSGA-II select K within the group) with a RECOMBINATION
# (whole individuals rebuilt block-by-block from random specialists) and INTEGRATION phases (verbatim Igel:
# whole-vector (1+1) on the aggregate 2-obj, NSGA-II selection, Pareto archive). Archive / metrics /
# checkpoints (mo_gen_finalize!) fire ONLY in integration; specialization phases are warmups (they advance
# n_evals and log a light progress line). Schedule: cc_cycles × [spec cc_spec_gens + integ cc_integ_gens],
# then integration until the TRIALS budget is exhausted.
function parametrize_CCIgel(; ref_soa::ActiveSoA, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, igel_sigma0::Float64=0.3, igel_sobol_init::Bool=true, igel_niche_radius::Float64=0.0, igel_reseed_sigma::Float64=0.0, igel_maturity::Int=0, single_cov::Bool=false, cc_group_size::Int=2, cc_spec_gens::Int=20, cc_integ_gens::Int=20, cc_cycles::Int=5, cc_fix_others::Bool=false)
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
  have_val = !isnothing(val_ref_soa) && !SKIP_VAL[]   # PAN_SKIP_VAL disables all held-out val re-sim during training
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
  # per-species objective (specialization scoring): score species g only, aggregate = Σ of its 2 objectives.
  function _fitness_species(rep_results, gsp::Int)
    _, eco_losses, _ = _agg_reps(rep_results)
    objs = _mo_objectives_species(eco_losses, eco_species_ids, gsp)
    MOLBSA.MOFitness(objs, Float64(sum(objs)))
  end
  next_param() = !isempty(_sobol_cands) && length(_sobol_cands) >= 1 ? popfirst!(_sobol_cands) :
                 BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)

  K = cc_group_size
  cc_mu = n_species * K                          # whole-individual population size = generation eval count
  CCIgel.register_sampler_types!(PU.SpeciesSampler, PU.EcoSpeciesSampler)
  if isnothing(resume_from)
    template = !isnothing(start_from) ? _load_params_from_path(start_from) : BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    slots = PU.build_slots(param_dists, template)
    # initial population of cc_mu whole individuals: Sobol candidate DB / Sobol design / random draws.
    init_params = !isempty(_sobol_cands) ? [next_param() for _ in 1:cc_mu] :
                  igel_sobol_init ? PU.sobol_samples(param_dists, template, cc_mu) :
                  [BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment) for _ in 1:cc_mu]
    init_us = [PU.params_to_u(p, param_dists, slots) for p in init_params]
    init_cands = [MOLBSA.MOCandidate(init_params[k], _fitness(_run(init_params[k]))[1]) for k in 1:cc_mu]
    groups = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(param_dists, slots, BSP.BIOMASS_PER_ECO_GROUPS))
    species_slots = CCIgel.species_slot_map(param_dists, slots, eco_species_ids, n_species)
    n_specialized = count(!isempty, species_slots)
    @info "CC-Igel: $n_species species × K=$K = $cc_mu individuals; $n_specialized species have specialization slots; integration blocks=$(length(groups))"
    search_state = IgelMOCMAES.IgelState(init_us, init_cands, rng; sigma0=igel_sigma0, archive_cap=archive_cap, max_iter=typemax(Int), niche_radius=igel_niche_radius, reseed_sigma=igel_reseed_sigma, maturity_period=igel_maturity, blocks=groups)
    search_state.n_evals = cc_mu
    bio_params = template
  else
    @info "Resuming from $resume_from"
    search_state = JLD2.load_object(resume_from)
    bio_params = search_state.representative.x
    slots = PU.build_slots(param_dists, bio_params)
    species_slots = CCIgel.species_slot_map(param_dists, slots, eco_species_ids, n_species)
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
  val_fit = have_val ? function (x)
      vreps = fit_params(val_ref_soa, x, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params; debug=false, search_tier=3, injection_dict=inj_dict_val, injection_years=inj_years_val, dual_b=nothing, seeds=fixed_seeds)
      v = _agg_reps(vreps); (v[1], vreps[v[3]][2], v[2])
    end : nothing
  gio = mo_gen_open(search_state; output_dir=output_dir, writer_ch=writer_ch, losses_db=losses_db,
    resuming=!isnothing(resume_from), splots=splots, eco_list=eco_list, species_list=species_list,
    eco_species_ids=eco_species_ids, n_species=n_species, site_sim_years=site_sim_years, loss_params=loss_params,
    no_establishment=no_establishment, n_reps=n_reps, rng=rng, have_val=have_val, n_output_plots=n_output_plots,
    sampled_ids=sampled_ids, sampled_ids_val=sampled_ids_val, emp_sample=emp_sample, emp_sample_val=emp_sample_val, val_fit=val_fit)
  caused_by_interrupt(e) = e isa InterruptException ? true :
    e isa TaskFailedException ? any(en -> caused_by_interrupt(en.exception), Base.current_exceptions(e.task)) :
    e isa CompositeException ? any(caused_by_interrupt, e.exceptions) : false

  evals_done = search_state.n_evals

  # ---- one INTEGRATION generation: verbatim Igel over the whole vector (archive/metrics/ckpt) ----
  function integ_gen!()
    offs = IgelMOCMAES.ask(search_state)
    m = search_state.mu
    off_fxs = Vector{MOLBSA.MOFitness}(undef, m)
    off_params = Vector{typeof(bio_params)}(undef, m)
    gen_best_agg = Inf; local gb_run, gb_eco, gb_cached
    for k in 1:m
      p = PU.u_to_params(offs[k], param_dists, slots, bio_params)
      rep_results = _run(p); fx_k, run_k, eco_k = _fitness(rep_results)
      off_params[k] = p; off_fxs[k] = fx_k; evals_done += 1
      if fx_k.aggregate < gen_best_agg
        gen_best_agg = fx_k.aggregate; gb_run = run_k; gb_eco = eco_k; gb_cached = _median_rep_cached(rep_results)
      end
    end
    is_new_best = IgelMOCMAES.tell!(search_state, off_fxs, off_params)
    search_state.n_evals = evals_done
    mo_gen_finalize!(gio, search_state, m, is_new_best, gb_run, gb_eco, gb_cached)
  end

  # ---- build per-species specialization sub-states from the current integration population ----
  # Each species group g gets an IgelState over the SAME whole u-vectors but with blocks=[species_slots[g]]
  # so ask/tell perturb/update ONLY species g's slots. Group members = a K-slice of the current population.
  function build_cc_groups()
    cur_us = [copy(ind.x) for ind in search_state.pop]                  # Individual.x IS the u-vector
    S = IgelMOCMAES.IgelState
    states = Vector{Union{Nothing,S}}(undef, n_species)
    for s in 1:n_species
      if isempty(species_slots[s])
        states[s] = nothing; continue
      end
      lo = (s - 1) * K + 1; hi = min(s * K, length(cur_us))
      idxs = collect(lo:hi); length(idxs) < K && append!(idxs, rand(rng, 1:length(cur_us), K - length(idxs)))
      grp_us = [copy(cur_us[i]) for i in idxs]
      grp_cands = [MOLBSA.MOCandidate(PU.u_to_params(cur_us[i], param_dists, slots, bio_params), search_state.pop[i].fx) for i in idxs]
      states[s] = IgelMOCMAES.IgelState(grp_us, grp_cands, rng; sigma0=igel_sigma0, archive_cap=archive_cap,
        max_iter=typemax(Int), blocks=Vector{Int}[species_slots[s]])
    end
    return CCIgel.CCGroups(states, species_slots)
  end

  # ---- one SPECIALIZATION generation over ALL species groups (warmup: advances n_evals, no archive) ----
  # `bg` (cc_fix_others=true) is a SHARED background u-vector (the current-best representative) held for the
  # phase: an offspring is evaluated by overwriting only species s's dims of `bg`. `bg=nothing` (default)
  # keeps each individual's own (drifting) other-blocks — Igel.ask perturbs only s's dims of ind.x.
  function spec_gen!(cc, bg::Union{Nothing,Vector{Float64}})
    tot = 0.0; nsp = 0
    for s in 1:n_species
      st = cc.states[s]; st === nothing && continue
      offs = IgelMOCMAES.ask(st)
      off_fxs = Vector{MOLBSA.MOFitness}(undef, st.mu)
      off_params = Vector{typeof(bio_params)}(undef, st.mu)
      for k in 1:st.mu
        u = offs[k]
        if bg !== nothing                                # splice species s's dims onto the shared background
          u = copy(bg); @inbounds for d in species_slots[s]; u[d] = offs[k][d]; end
        end
        p = PU.u_to_params(u, param_dists, slots, bio_params)   # decode WHOLE vector
        off_params[k] = p; off_fxs[k] = _fitness_species(_run(p), s); evals_done += 1
      end
      IgelMOCMAES.tell!(st, off_fxs, off_params)                      # NSGA-II select K within group g
      tot += minimum(fx.aggregate for fx in off_fxs); nsp += 1
    end
    search_state.n_evals = evals_done
    return nsp == 0 ? 0.0 : tot / nsp
  end

  # one RECOMBINATION step: rebuild the integration population block-by-block from random specialists, then
  # swap in a fresh integration IgelState over it (keeping the accumulated archive / representative).
  function recombine!(cc)
    base_us = [copy(ind.x) for ind in search_state.pop]                 # Individual.x IS the u-vector
    new_us = CCIgel.recombine_us(cc, n_species, K, base_us, rng)
    new_params = [PU.u_to_params(u, param_dists, slots, bio_params) for u in new_us]
    new_cands = [MOLBSA.MOCandidate(new_params[k], _fitness(_run(new_params[k]))[1]) for k in 1:cc_mu]
    evals_done += cc_mu
    groups2 = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(param_dists, slots, BSP.BIOMASS_PER_ECO_GROUPS))
    old_archive = search_state.archive; old_rep = search_state.representative
    old_best_iter = search_state.best_iteration; old_best_iters = search_state.best_iterations
    gen_i = search_state.i
    search_state = IgelMOCMAES.IgelState(new_us, new_cands, rng; sigma0=igel_sigma0, archive_cap=archive_cap,
      max_iter=typemax(Int), niche_radius=igel_niche_radius, reseed_sigma=igel_reseed_sigma, maturity_period=igel_maturity, blocks=groups2)
    search_state.archive = old_archive; search_state.representative = old_rep
    search_state.best_iteration = old_best_iter; search_state.best_iterations = old_best_iters
    search_state.i = gen_i; search_state.n_evals = evals_done
    return nothing
  end

  # Phase SCHEDULE as a list of (:integ | :spec) tags. cc_fix_others=true needs a current-best before the
  # first specialization → a LEADING integration phase. After the scheduled cycles, keep integrating to budget.
  schedule = Symbol[]
  cc_fix_others && push!(schedule, :integ)                 # leading aggregate phase (provides the background)
  for _c in 1:cc_cycles; push!(schedule, :spec); push!(schedule, :integ); end

  try
    _cyc = 0
    TProgress.@track for _phase in 1:length(schedule)
      evals_done >= TRIALS && break
      if schedule[_phase] == :spec
        _cyc += 1
        cc = build_cc_groups()
        bg = cc_fix_others ? PU.params_to_u(search_state.representative.x, param_dists, slots) : nothing
        for _g in 1:cc_spec_gens
          evals_done >= TRIALS && break
          m = spec_gen!(cc, bg)
          @info "CC-Igel spec cycle $_cyc gen $_g | mean-species-best=$(round(m; sigdigits=4)) | evals=$evals_done | fix_others=$cc_fix_others"
        end
        recombine!(cc)                                     # → fresh integration population
      else
        for _g in 1:cc_integ_gens
          evals_done >= TRIALS && break
          integ_gen!()
        end
      end
    end
    # remainder: pure integration until the budget is exhausted
    while evals_done < TRIALS
      integ_gen!()
    end
  catch e
    caused_by_interrupt(e) ? @info("Search interrupted by user @ gen $(search_state.i); finalizing checkpoint…") : rethrow()
  finally
    PARALLEL_SITES[] = true   # restore site-parallelism (candidate mode set it false for this driver)
    stop_writer(writer_ch, writer_task); close(losses_db_file); try; close(gio.metrics_io); catch; end
    try
      mkpath(output_dir); fname = "search_state@$(search_state.i).jld2"
      JLD2.save_object(joinpath(output_dir, fname), search_state)
      link_path = joinpath(output_dir, "search_state_latest.jld2"); islink(link_path) && rm(link_path); symlink(fname, link_path)
      @info "Search state saved @ $(search_state.i)"
    catch e; @error "Failed to save search state on exit" exception = (e, catch_backtrace()); end
  end
  return search_state
end

function parametrize_NSGA2(; ref_soa::ActiveSoA, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, nsga2_pop::Int=48, nsga2_offspring::Union{Nothing,Int}=nothing, nsga2_eta_c::Float64=20.0, nsga2_eta_m::Float64=20.0, nsga2_pc::Float64=0.9, nsga2_pm::Float64=-1.0, sobol_init::Bool=true)
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
  have_val = !isnothing(val_ref_soa) && !SKIP_VAL[]   # PAN_SKIP_VAL disables all held-out val re-sim during training
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

  λ = isnothing(nsga2_offspring) ? nsga2_pop : nsga2_offspring
  if isnothing(resume_from)
    template = !isnothing(start_from) ? _load_params_from_path(start_from) : BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    slots = PU.build_slots(param_dists, template)
    # initial μ population: Sobol candidate DB if given, else a Sobol space-filling design, else random draws.
    init_params = !isempty(_sobol_cands) ? [next_param() for _ in 1:nsga2_pop] :
                  sobol_init ? PU.sobol_samples(param_dists, template, nsga2_pop) :
                  [BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment) for _ in 1:nsga2_pop]
    init_us = [PU.params_to_u(p, param_dists, slots) for p in init_params]
    init_cands = [MOLBSA.MOCandidate(init_params[k], _fitness(_run(init_params[k]))[1]) for k in 1:nsga2_pop]
    rep0 = init_cands[argmin([Float64(c.fx.aggregate) for c in init_cands])]
    @info "NSGA-II: μ=$nsga2_pop parents, λ=$λ offspring/gen, d=$(length(slots)), ηc=$nsga2_eta_c ηm=$nsga2_eta_m pc=$nsga2_pc pm=$(nsga2_pm < 0 ? "1/d" : string(nsga2_pm))"
    search_state = NSGA2.NSGA2State(; representative=rep0, current=rep0, archive=MOLBSA.MOCandidate{typeof(template)}[], best_iterations=Tuple{Int,Float64,MOLBSA.MOCandidate{typeof(template)}}[], archive_cap=archive_cap, rng=rng, i=0, n_evals=nsga2_pop, max_iter=typemax(Int), pop_u=init_us, pop=init_cands, n=length(slots), mu=nsga2_pop, lambda=λ, eta_c=nsga2_eta_c, eta_m=nsga2_eta_m, p_c=nsga2_pc, p_m=nsga2_pm)
    for c in init_cands; NSGA2._archive!(search_state, c); end
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
  val_fit = have_val ? function (x)            # (run, cached, eco_losses) on the val split, aggregated over n_reps like train
      vreps = fit_params(val_ref_soa, x, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params; debug=false, search_tier=3, injection_dict=inj_dict_val, injection_years=inj_years_val, dual_b=nothing, seeds=fixed_seeds)
      v = _agg_reps(vreps); (v[1], vreps[v[3]][2], v[2])
    end : nothing
  gio = mo_gen_open(search_state; output_dir=output_dir, writer_ch=writer_ch, losses_db=losses_db,
    resuming=!isnothing(resume_from), splots=splots, eco_list=eco_list, species_list=species_list,
    eco_species_ids=eco_species_ids, n_species=n_species, site_sim_years=site_sim_years, loss_params=loss_params,
    no_establishment=no_establishment, n_reps=n_reps, rng=rng, have_val=have_val, n_output_plots=n_output_plots,
    sampled_ids=sampled_ids, sampled_ids_val=sampled_ids_val, emp_sample=emp_sample, emp_sample_val=emp_sample_val, val_fit=val_fit)
  caused_by_interrupt(e) = e isa InterruptException ? true :
    e isa TaskFailedException ? any(en -> caused_by_interrupt(en.exception), Base.current_exceptions(e.task)) :
    e isa CompositeException ? any(caused_by_interrupt, e.exceptions) : false

  evals_done = search_state.n_evals
  est_gens = max(1, cld(TRIALS - evals_done, λ))
  try
    TProgress.@track for _gen in 1:est_gens
      evals_done >= TRIALS && break
      offs = NSGA2.ask(search_state)                  # λ offspring u-vectors (tournament + SBX + polynomial mutation)
      off_fxs = Vector{MOLBSA.MOFitness}(undef, λ)
      off_params = Vector{typeof(bio_params)}(undef, λ)
      gen_best_agg = Inf; local gb_run, gb_eco, gb_cached, gb_idx
      for k in 1:λ                                     # SERIAL (fit_params threads internally)
        p = PU.u_to_params(offs[k], param_dists, slots, bio_params)
        rep_results = _run(p); fx_k, run_k, eco_k = _fitness(rep_results)
        off_params[k] = p; off_fxs[k] = fx_k; evals_done += 1
        if fx_k.aggregate < gen_best_agg
          gen_best_agg = fx_k.aggregate; gb_idx = k; gb_run = run_k; gb_eco = eco_k; gb_cached = _median_rep_cached(rep_results)
        end
      end
      is_new_best = NSGA2.tell!(search_state, off_fxs, off_params, offs)   # (μ+λ) non-dom-sort + crowding selection
      search_state.n_evals = evals_done
      mo_gen_finalize!(gio, search_state, search_state.mu, is_new_best, gb_run, gb_eco, gb_cached)   # shared: ckpt on ANY archive change (train+val)
    end
  catch e
    caused_by_interrupt(e) ? @info("Search interrupted by user @ gen $(search_state.i); finalizing checkpoint…") : rethrow()
  finally
    PARALLEL_SITES[] = true   # restore site-parallelism (candidate mode set it false for this driver)
    stop_writer(writer_ch, writer_task); close(losses_db_file); try; close(gio.metrics_io); catch; end
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
  have_val = !isnothing(val_ref_soa) && !SKIP_VAL[]   # PAN_SKIP_VAL disables all held-out val re-sim during training
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
  val_fit = have_val ? (x -> only(fit_params(val_ref_soa, x, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params; debug=false, search_tier=search_tier, t4_ref=val_t4_ref, cycle_map=val_cycle_map, n_cycles=val_n_cycles, injection_dict=inj_dict_val, injection_years=inj_years_val, dual_b=val_dual_b, seeds=[rand(rng, UInt64)]))) : nothing
  gio = mo_gen_open(search_state; output_dir=output_dir, writer_ch=writer_ch, losses_db=losses_db,
    resuming=false, splots=splots, eco_list=eco_list, species_list=species_list, eco_species_ids=eco_species_ids,
    n_species=n_species, site_sim_years=site_sim_years, loss_params=loss_params, no_establishment=no_establishment,
    n_reps=n_reps, rng=rng, have_val=have_val, n_output_plots=n_output_plots, sampled_ids=sampled_ids,
    sampled_ids_val=sampled_ids_val, emp_sample=emp_sample, emp_sample_val=emp_sample_val, val_fit=val_fit)
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

      mo_gen_finalize!(gio, search_state, search_state.engine.emitter.lambda, is_new_best, gb_run, gb_eco, gb_cached)   # shared: ckpt on ANY archive change (train+val)
    end
  catch e
    caused_by_interrupt(e) ? @info("Search interrupted by user @ gen $(search_state.i); finalizing checkpoint…") : rethrow()
  finally
    try; close(gio.metrics_io); catch; end
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
      # Per-cell: divide each (eco,lu,sp) cell by its OWN reference scale, rescale by the AGB-rank weight,
      # then sum into the group. A and B keep separate scales; RANKW is shared. NOTE: the AGB exponent is
      # applied ONCE, per-cohort in calculate_species_loss! (_agb_pow on each hinged excess) — do NOT
      # re-apply it here or it double-powers for p≠1 (W's power stays here since W is not pre-powered).
      Ws = g == 0 ? PU.W_SCALE_A[] : PU.W_SCALE_B[]
      As = g == 0 ? PU.AGB_SCALE_A[] : PU.AGB_SCALE_B[]
      Rw = PU.RANKW[]
      @inbounds for gsp in eco_species_ids[e]
        _dom_included(gsp) || continue
        rw = _cell(Rw, gsp, e); ws = _cell(Ws, gsp, e); as = _cell(As, gsp, e)
        objs[2 * g + 1] += rw * PU._w_pow(ws > 0 ? el.sp_w_loss[gsp] / ws : el.sp_w_loss[gsp])
        objs[2 * g + 2] += rw * (as > 0 ? el.sp_agb_loss[gsp] / as : el.sp_agb_loss[gsp])
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

# Per-species variant of _mo_objectives for CC-Igel specialization: IDENTICAL to _mo_objectives but the
# inner per-species loop is restricted to `target_gsp` (still summed over ecoregions / dual groups). Returns
# the 2-element [ΣW_g, ΣA_g] (single-sim) or 2·ngroups if dual — but for CC specialization only sim A is
# used so it is the 2-vector. Same cell / non-cell branches, scales, rank and w-power as the aggregate.
function _mo_objectives_species(eco_losses::Vector{PU.SiteLoss}, eco_species_ids::Vector{Vector{Int}}, target_gsp::Int)::Vector{FloatType}
  ne = length(eco_species_ids)
  ngroups = max(1, length(eco_losses) ÷ ne)
  objs = zeros(FloatType, 2 * ngroups)
  cell = PU.CELL_NORM[]
  for (idx, el) in enumerate(eco_losses)
    g = (idx - 1) ÷ ne
    e = (idx - 1) % ne + 1
    if cell
      Ws = g == 0 ? PU.W_SCALE_A[] : PU.W_SCALE_B[]
      As = g == 0 ? PU.AGB_SCALE_A[] : PU.AGB_SCALE_B[]
      Rw = PU.RANKW[]
      @inbounds for gsp in eco_species_ids[e]
        gsp == target_gsp || continue
        _dom_included(gsp) || continue
        rw = _cell(Rw, gsp, e); ws = _cell(Ws, gsp, e); as = _cell(As, gsp, e)
        objs[2 * g + 1] += rw * PU._w_pow(ws > 0 ? el.sp_w_loss[gsp] / ws : el.sp_w_loss[gsp])
        objs[2 * g + 2] += rw * (as > 0 ? el.sp_agb_loss[gsp] / as : el.sp_agb_loss[gsp])
      end
    else
      @inbounds for gsp in eco_species_ids[e]
        gsp == target_gsp || continue
        _dom_included(gsp) || continue
        objs[2 * g + 1] += el.sp_w_loss[gsp]
        objs[2 * g + 2] += el.sp_agb_loss[gsp]
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

# Per-generation losses/metrics write, OFFLOADED to the writer thread so the search never blocks on I/O.
# `payload` (a NamedTuple, only on a new best) is inserted into losses.duckdb in ONE transaction — a single
# fsync instead of ~1+n_eco+n_eco·n_species autocommits (the cluster-vs-laptop slowdown). `metrics_line` is
# written every generation. Drained/flushed on normal exit AND interrupt via stop_writer's finally.
struct MOLossesJob
  db::Any
  io::IO
  metrics_line::String
  payload::Any            # NamedTuple(iter, ns, no, total, arch, blob, eco[...]) on new-best; else nothing
  val_rows::Vector{NTuple{4,Float64}}   # (A_W_train,A_AGB_train,A_W,A_AGB) scored this gen → mirrored into losses.duckdb.val_cache
end

function _mo_write_losses!(job::MOLossesJob)
  p = job.payload
  if p !== nothing
    try
      DuckDB.execute(job.db, "BEGIN TRANSACTION")
      DuckDB.execute(job.db, "INSERT INTO total_loss VALUES (?, ?, ?, ?, ?, ?)", [p.iter, p.ns, p.no, p.total, p.arch, p.blob])
      for er in p.eco
        DuckDB.execute(job.db, "INSERT INTO ecoregion_loss VALUES (?, ?, ?, ?, ?)", [p.iter, er.name, er.ns, er.no, er.tot])
        for sr in er.sp
          DuckDB.execute(job.db, "INSERT INTO species_loss VALUES (?, ?, ?, ?, ?)", [p.iter, er.name, sr.s, sr.w, sr.a])
        end
      end
      DuckDB.execute(job.db, "COMMIT")
    catch e
      try; DuckDB.execute(job.db, "ROLLBACK"); catch; end
      @error "mo_writer: losses insert failed" exception = (e, catch_backtrace())
    end
  end
  if !isempty(job.val_rows)              # durable mirror of cv_val_cache into losses.duckdb (survives resumes / a lost CSV)
    try
      DuckDB.execute(job.db, "CREATE TABLE IF NOT EXISTS val_cache (A_W_train DOUBLE, A_AGB_train DOUBLE, A_W DOUBLE, A_AGB DOUBLE, PRIMARY KEY (A_W_train, A_AGB_train))")
      DuckDB.execute(job.db, "BEGIN TRANSACTION")
      for v in job.val_rows
        DuckDB.execute(job.db, "INSERT INTO val_cache VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING", [v[1], v[2], v[3], v[4]])
      end
      DuckDB.execute(job.db, "COMMIT")
    catch e
      try; DuckDB.execute(job.db, "ROLLBACK"); catch; end
      @error "mo_writer: val_cache insert failed" exception = (e, catch_backtrace())
    end
  end
  try; println(job.io, job.metrics_line); flush(job.io); catch e; @error "mo_writer: metrics write failed" exception = (e, catch_backtrace()); end
end

# Background writer for MOLBSA: checkpoints the full state (archive included) and
# the representative params, and generates plots, off the search thread. Mirrors
# start_writer but reads `state.representative` instead of `state.best`.
function start_mo_writer(::Type{State}, output_dir::AbstractString; buffer_size::Int=8) where {State}
  ch = Channel{Union{WriterJob{State},MOLossesJob,Symbol}}(buffer_size)
  task = Threads.@spawn begin
    try
      for job in ch
        job === STOP && break
        job isa MOLossesJob && (_mo_write_losses!(job); continue)   # batched losses + metrics, off the search thread
        state = job.state
        if job.save_ckpt
          try
            @info "Checkpoint (archive changed): agg=$(state.representative.fx.aggregate)" iter = state.i archive = length(state.archive)
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

  have_val = !isnothing(val_ref_soa) && !SKIP_VAL[]   # PAN_SKIP_VAL disables all held-out val re-sim during training
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
  val_fit = have_val ? (x -> only(fit_params(val_ref_soa, x, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params; debug=false, search_tier=3, injection_dict=inj_dict_val, injection_years=inj_years_val, dual_b=val_dual_b, seeds=[rand(rng, UInt64)]))) : nothing
  gio = mo_gen_open(search_state; output_dir=output_dir, writer_ch=writer_ch, losses_db=losses_db,
    resuming=!isnothing(resume_from), splots=splots, eco_list=eco_list, species_list=species_list,
    eco_species_ids=eco_species_ids, n_species=n_species, site_sim_years=site_sim_years, loss_params=loss_params,
    no_establishment=no_establishment, n_reps=n_reps, rng=rng, have_val=have_val, n_output_plots=n_output_plots,
    sampled_ids=sampled_ids, sampled_ids_val=sampled_ids_val, emp_sample=emp_sample, emp_sample_val=emp_sample_val, val_fit=val_fit)

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

      mo_gen_finalize!(gio, search_state, 1, is_new_best, run_result, eco_losses, cached_sites_state)   # shared: ckpt on ANY archive change (train+val)
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
    close(losses_db_file); try; close(gio.metrics_io); catch; end
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
    lu in ("natural", "artificial") || (lu = "natural")   # site_class_strata puts a site-tier (A/B/C/D) in |lu=; establishment prob is site-tier-independent → use the natural model input
    for (j, gsp) in enumerate(sp_ids)
      v = get(tbl, (uppercase(species_list[gsp]), l3, lu), nothing)
      v === nothing && continue
      params.PROB_ESTAB_SPP[eco_id][j] = FloatType(v); n += 1
    end
  end
  @info "prob_estab_from_data: seeded PROB_ESTAB for $n eco×species cells (calibration start)"
  params
end

# Per-(category, L3, land_use) B_MAX floor (g/m²) from runs/bmax_floor.csv (cols: category,l3,lu,bmax_floor).
function _load_bmax_floor_table(csv_path::String)
  if !isfile(csv_path)
    @warn "bmax_floor_from_data=true but $csv_path not found — B_MAX floor left flat [12000,35000]"; return nothing
  end
  df = CSV.read(csv_path, DataFrame)
  d = Dict{Tuple{String,String,String},Float64}()
  for r in eachrow(df)
    d[(uppercase(strip(String(r.category))), String(r.l3), String(r.lu))] = Float64(r.bmax_floor)
  end
  isempty(d) && (@warn "bmax_floor: no rows in $csv_path"; return nothing)
  @info "bmax_floor_from_data: loaded $(length(d)) (category,l3,lu) B_MAX floors from $csv_path"
  d
end

# Build the per-(eco_id, sp_local) B_MAX lower bound = max(12000, table[(species,l3,lu)]) and install it into
# PU.BMAX_FLOOR (read by u_to_params/params_to_u/mutate_params). Installs `nothing` when the feature is off.
function _build_bmax_floor!(eco_list, species_list, eco_species_ids)
  tbl = BiomassSuccessionPlugin.BMAX_FLOOR_TABLE[]
  if isnothing(tbl)
    PU.BMAX_FLOOR[] = nothing; return nothing
  end
  d = Dict{Tuple{Int,Int},FloatType}()
  for (eco_id, sp_ids) in enumerate(eco_species_ids)
    l3, lu = _parse_eco_lu(eco_list[eco_id])
    for (j, gsp) in enumerate(sp_ids)
      d[(eco_id, j)] = FloatType(max(12000.0, get(tbl, (uppercase(species_list[gsp]), l3, lu), 12000.0)))
    end
  end
  PU.BMAX_FLOOR[] = d
  @info "bmax_floor_from_data: B_MAX lower bound set for $(length(d)) eco×species cells ($(count(>(FloatType(12000)), values(d))) above the 12000 default); window [12000,35000]"
  nothing
end

# Raise any seed B_MAX below its floor up to the floor, so the initial candidate is feasible / consistent
# with the rescaled decode. No-op when the floor feature is off.
function _floor_bmax_seed!(params)
  d = PU.BMAX_FLOOR[]
  isnothing(d) && return params
  for (eco_id, sp_ids) in enumerate(params.ECO_SPECIES_IDS)
    for j in eachindex(sp_ids)
      lo = get(d, (eco_id, j), FloatType(12000))
      params.B_MAX_SPP[eco_id][j] < lo && (params.B_MAX_SPP[eco_id][j] = lo)
    end
  end
  params
end

# yaml `prob_estab_lower_bound`: use the per-species data PROB_ESTAB as a LOWER bound on the search (search
# window [data, 1.0]) instead of only seeding it — so the optimizer can't suppress establishment below the
# empirically-observed rate (it may only go higher). Requires prob_estab_from_data (the table).
const PROB_ESTAB_LOWER_BOUND = Ref{Bool}(false)
const PROB_ESTAB_LOWER_MULT  = Ref{Float64}(1.0)   # yaml `prob_estab_lower_bound_mult`: floor = mult × data (e.g. 0.8)
# Per-(eco_id, sp_local) PROB_ESTAB lower bound = mult × data value → PU.PROB_ESTAB_FLOOR (read by
# u_to_params/params_to_u). Cells with no table entry keep the param's own lower bound. nothing when off.
function _build_prob_estab_floor!(eco_list, species_list, eco_species_ids)
  tbl = BiomassSuccessionPlugin.PROB_ESTAB_TABLE[]
  if !PROB_ESTAB_LOWER_BOUND[] || isnothing(tbl)
    PU.PROB_ESTAB_FLOOR[] = nothing; return nothing
  end
  mult = PROB_ESTAB_LOWER_MULT[]
  d = Dict{Tuple{Int,Int},FloatType}()
  for (eco_id, sp_ids) in enumerate(eco_species_ids)
    l3, lu = _parse_eco_lu(eco_list[eco_id])
    lu in ("natural", "artificial") || (lu = "natural")   # site_class_strata puts a site-tier in |lu= ⇒ use natural
    for (j, gsp) in enumerate(sp_ids)
      v = get(tbl, (uppercase(species_list[gsp]), l3, lu), nothing)
      v === nothing || (d[(eco_id, j)] = FloatType(clamp(mult * Float64(v), 0.0, 1.0)))
    end
  end
  PU.PROB_ESTAB_FLOOR[] = isempty(d) ? nothing : d
  @info "prob_estab_lower_bound: PROB_ESTAB search floored at $(mult)×data for $(length(d)) eco×species cells (window [floor,1.0])"
  nothing
end

# ANPP_MAX data floor — parallel to _load_bmax_floor_table. CSV cols: category,l3,lu,anpp_floor (+ extras
# ignored). anpp_floor = p99 of per-cohort agb/age (a valid lower bound on ANPP_MAX; see plugin growth eq).
function _load_anpp_floor_table(csv_path::String)
  if !isfile(csv_path)
    @warn "anpp_floor_from_data=true but $csv_path not found — ANPP floor off"; return nothing
  end
  df = CSV.read(csv_path, DataFrame)
  col = "anpp_floor" in names(df) ? :anpp_floor : (:anpp_floor_pai in propertynames(df) ? :anpp_floor_pai : nothing)
  isnothing(col) && (@warn "anpp_floor: no anpp_floor column in $csv_path"; return nothing)
  d = Dict{Tuple{String,String,String},Float64}()
  for r in eachrow(df)
    v = getproperty(r, col)
    ismissing(v) && continue
    d[(uppercase(strip(String(r.category))), String(r.l3), String(r.lu))] = Float64(v)
  end
  isempty(d) && (@warn "anpp_floor: no rows in $csv_path"; return nothing)
  @info "anpp_floor_from_data: loaded $(length(d)) (category,l3,lu) ANPP floors from $csv_path (col=$col)"
  d
end

# Build the per-(eco_id, sp_local) ANPP lower bound = table[(species,l3,lu)] (0 = no floor) and install into
# PU.ANPP_FLOOR (read by u_to_params). Installs `nothing` when the feature is off. Parallel to _build_bmax_floor!.
function _build_anpp_floor!(eco_list, species_list, eco_species_ids)
  tbl = BiomassSuccessionPlugin.ANPP_FLOOR_TABLE[]
  if isnothing(tbl)
    PU.ANPP_FLOOR[] = nothing; return nothing
  end
  d = Dict{Tuple{Int,Int},FloatType}()
  for (eco_id, sp_ids) in enumerate(eco_species_ids)
    l3, lu = _parse_eco_lu(eco_list[eco_id])
    for (j, gsp) in enumerate(sp_ids)
      d[(eco_id, j)] = FloatType(get(tbl, (uppercase(species_list[gsp]), l3, lu), 0.0))
    end
  end
  PU.ANPP_FLOOR[] = d
  @info "anpp_floor_from_data: ANPP lower bound set for $(length(d)) eco×species cells ($(count(>(FloatType(0)), values(d))) with a nonzero floor)"
  nothing
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

# Expand ${VAR}/$VAR (from ENV) and a leading ~ in a config path string, so yaml paths are portable across
# machines (e.g. cohorts_db_path: "${FIADB}", output_dir: "${SCRATCH}/out", "~/data.duckdb"). Missing vars →
# "". Plain/relative paths pass through unchanged. Same treatment the FIADB override gets, applied in-yaml.
function _expandenv(s::AbstractString)
  isempty(s) && return String(s)
  out = replace(String(s), r"\$\{(\w+)\}" => m -> get(ENV, m[3:end-1], ""))   # ${VAR}
  out = replace(out, r"\$(\w+)" => m -> get(ENV, m[2:end], ""))               # $VAR
  return expanduser(out)
end

function run_from_yaml(yaml_path::String; overrides::AbstractDict=Dict{String,Any}())
  PAN_TIMING[] = get(ENV, "PAN_TIMING", "") in ["1", "true", "yes"]   # per-gen wall-time breakdown to stdout
  SKIP_VAL[] = get(ENV, "PAN_SKIP_VAL", "") in ["1", "true", "yes"]   # skip held-out val re-sim during training
  PARALLEL_MODE[] = Symbol(get(ENV, "PAN_PARALLEL", "soa"))           # :soa (site-parallel) | :candidate (candidate-parallel)
  NEWBEST_STATS[] = get(ENV, "PAN_NEWBEST_STATS", "") in ["1", "true", "yes"]   # opt-in diagnostic train-stats sim on new best
  _tm_start[] = time_ns()                                             # startup clock (spans data load → first gen)
  get!(ENV, "PAN_DATA", "runs")          # data dir for ${PAN_DATA}/*.csv|*.duckdb in the yaml; default = repo runs/ (local),
                                         # set PAN_DATA=<dir> on the cluster (e.g. $SCRATCH/runs/data) → one yaml, both layouts
  get!(ENV, "PAN_OUT", "runs/outputs")   # output root for a ${PAN_OUT}/... output_dir in the yaml; default local, set on cluster.
                                         # (When launched via train_resumable.sh, output_dir is overridden anyway — this is the bare-run fallback.)
  cfg = YAML.load_file(yaml_path)
  for (k, v) in overrides; cfg[String(k)] = v; end   # runner/CLI overrides (e.g. per-fold fold_index/output_dir)
  # String config values get ${VAR}/$VAR/~ expanded from the environment (paths like output_dir, *_csv,
  # cohorts_db_path become portable); non-strings and plain strings pass through unchanged.
  get_cfg(key, default) = (v = get(cfg, key, default); v isa AbstractString ? _expandenv(v) : v)

  seed = get_cfg("seed", 404)
  Random.seed!(seed)
  rng = RNGType(rand(UInt64))

  filter_plots = NTuple{4,Int}[
    NTuple{4,Int}([x isa AbstractString ? parse(Int, x) : Int(x) for x in p])
    for p in get_cfg("filter_plots", [])
  ]

  # Plots to DROP from the loaded set (outlier exclusion). Accepts an inline `exclude_plots:` list of
  # [statecd,unitcd,countycd,plot] and/or an `exclude_plots_csv:` file with those four columns; merged.
  exclude_plots = NTuple{4,Int}[
    NTuple{4,Int}([x isa AbstractString ? parse(Int, x) : Int(x) for x in p])
    for p in get_cfg("exclude_plots", [])
  ]
  let epc = get_cfg("exclude_plots_csv", nothing)
    if !(epc === nothing || epc == "null")
      for r in eachrow(CSV.read(String(epc), DataFrame))
        push!(exclude_plots, (Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot)))
      end
    end
  end
  isempty(exclude_plots) || @info "exclude_plots: dropping $(length(exclude_plots)) plots from the loaded set"

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

  # K-fold: each fold writes to <output_dir>/fold_<fold_index> so the 5 fold runs don't collide.
  _n_folds = Int(get_cfg("n_folds", 1)); _fold_index = Int(get_cfg("fold_index", 1))
  _output_dir = String(get_cfg("output_dir", "./outputs"))
  _n_folds > 1 && (_output_dir = joinpath(_output_dir, "fold_$(_fold_index)"))
  isdir(_output_dir) || mkpath(_output_dir)

  # DB path: env FIADB wins (portable across machines); else the yaml value (get_cfg already expands ${VAR}/~).
  _db_path = get(ENV, "FIADB", "") != "" ? ENV["FIADB"] : String(get_cfg("cohorts_db_path", "../data_eco_cohorts.duckdb"))

  # Fields shared by parametrize, plot_sample, and plot_sample_sobol
  common_kw = (
    cohorts_db_path=_db_path,
    filter_eco_field=get_cfg("filter_eco_field", "epa_l3"),
    eco_field=get_cfg("eco_field", "epa_l3"),
    tablename=get_cfg("tablename", "data_eco_cohorts"),
    output_dir=_output_dir,
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
    site_class_strata=Bool(get_cfg("site_class_strata", false)),
    siteclass_hi_max=Int(get_cfg("siteclass_hi_max", 4)),
    siteclass_scheme=String(get_cfg("siteclass_scheme", "2way")),
    shade_tier_csv=(haskey(cfg, "shade_tier_csv") ? String(get_cfg("shade_tier_csv", "")) : nothing),
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
    _load_longevity_table(_db_path) : nothing
  # Per-species LONGEVITY overrides from yaml (`longevity_overrides: {PIEL: 275}`) — applied on top of the
  # data table so hand corrections don't require rewriting the read-only species_longevity_ref DB table.
  let ov = get_cfg("longevity_overrides", nothing)
    if !isnothing(ov) && !isempty(ov)
      tbl = BiomassSuccessionPlugin.LONGEVITY_TABLE[]
      isnothing(tbl) && (tbl = Dict{String,Float64}(); BiomassSuccessionPlugin.LONGEVITY_TABLE[] = tbl)
      for (k, v) in ov; tbl[String(k)] = Float64(v); end
      @info "longevity_overrides applied: $(Dict(String(k)=>Float64(v) for (k,v) in ov))"
    end
  end
  BiomassSuccessionPlugin.SHADE_TOL_TABLE[] = Bool(get_cfg("shade_tol_from_data", false)) ?
    _load_shade_tol_table(String(get_cfg("shade_tol_csv", "./runs/shadetol_all_species.csv"))) : nothing
  BiomassSuccessionPlugin.PROB_ESTAB_TABLE[] = Bool(get_cfg("prob_estab_from_data", false)) ?
    _load_prob_estab_table(String(get_cfg("prob_estab_csv", "./runs/prob_estab_from_data.csv")),
                           String(get_cfg("prob_estab_csv_artificial", "./runs/prob_estab_from_data_artificial.csv"))) : nothing
  BiomassSuccessionPlugin.MATURITY_TABLE[] = Bool(get_cfg("maturity_from_data", false)) ?
    _load_maturity_table(String(get_cfg("maturity_csv", "./runs/prob_estab_all_species.csv"))) : nothing
  BiomassSuccessionPlugin.BMAX_FLOOR_TABLE[] = Bool(get_cfg("bmax_floor_from_data", false)) ?
    _load_bmax_floor_table(String(get_cfg("bmax_floor_csv", "./runs/bmax_floor.csv"))) : nothing
  BiomassSuccessionPlugin.ANPP_FLOOR_TABLE[] = Bool(get_cfg("anpp_floor_from_data", false)) ?
    _load_anpp_floor_table(String(get_cfg("anpp_floor_csv", "./runs/anpp_floor_853.csv"))) : nothing
  IgelMOCMAES.USE_CHOLESKY[] = Bool(get_cfg("igel_cholesky", false))   # igelmo sampling sqrt: Cholesky (cheaper) vs eigen; Ref → resume-safe
  BSP.FIX_GROWTH[] = Bool(get_cfg("fix_growth", false))     # stage-B: fix {D,S,ANPP_MAX,B_MAX}, fit establishment only
  BSP.FIX_MATURITY[] = Bool(get_cfg("fix_maturity", false)) # pin MATURITY out of the search (at MATURITY_TABLE/SONA)
  BSP.FIX_MIN_REL[] = Bool(get_cfg("fix_min_rel", false))   # pin MIN_REL_BIOMASS out of the search (at MIN_REL_PINNED)
  PROB_ESTAB_LOWER_BOUND[] = Bool(get_cfg("prob_estab_lower_bound", false))   # PROB_ESTAB data value as a search LOWER bound (needs prob_estab_from_data)
  PROB_ESTAB_LOWER_MULT[] = Float64(get_cfg("prob_estab_lower_bound_mult", 1.0))   # floor = mult × data (e.g. 0.8)

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
    exclude_plots=exclude_plots,
    search_mode=search_mode,
    tier=get_cfg("tier", 1),
    smoothing_window=smoothing_window,
    unbinned_w=Bool(get_cfg("unbinned_w", false)),   # Sim A: exact per-year EMD (no binning/age-smoothing)
    w_count_balance=Bool(get_cfg("w_count_balance", false)),
    w_count_balance_mode=String(get_cfg("w_count_balance_mode", "both")),
    w_count_beta=Float64(get_cfg("w_count_beta", 0.99)),
    rankw_mode=String(get_cfg("rankw_mode", "rank")),
    rankw_beta=Float64(get_cfg("rankw_beta", 0.999)),
    param_split_species=Dict{String,Vector{String}}(String(k) => String.(v) for (k, v) in get_cfg("param_split_species", Dict())),
    param_tier_merge=Dict{String,Any}(String(k) => Dict{String,Any}(String(sp) => Dict{String,String}(String(c) => String(g) for (c, g) in cm) for (sp, cm) in v) for (k, v) in get_cfg("param_tier_merge", Dict())),
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
    n_folds=Int(get_cfg("n_folds", 1)),
    fold_index=Int(get_cfg("fold_index", 1)),
    test_frac=Float64(get_cfg("test_frac", 0.0)),
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
    w_hinge=Bool(get_cfg("w_hinge", false)),
    w_hinge_pct=Float64(get_cfg("w_hinge_pct", 0.01)),
    w_hinge_pct_min=Float64(get_cfg("w_hinge_pct_min", 2.0)),
    w_hinge_pct_max=Float64(get_cfg("w_hinge_pct_max", 5.0)),
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
    ipop_max_barren_restarts=Int(get_cfg("ipop_max_barren_restarts", 4)),
    cmaes_archive_cap=Int(get_cfg("cmaes_archive_cap", 200)),
    seed_archive_from=(let v = get_cfg("seed_archive_from", nothing); (v === nothing || v == "null") ? nothing : String(v) end),
    cmaes_integer_handling=get_cfg("cmaes_integer_handling", false),
    cmaes_integer_std_factor=Float64(get_cfg("cmaes_integer_std_factor", 0.3)),
    cmaes_single_cov=Bool(get_cfg("cmaes_single_cov", false)),
    nsga2_pop=Int(get_cfg("nsga2_pop", 48)),
    nsga2_offspring=(let v = get_cfg("nsga2_offspring", nothing); (v === nothing || v == "null") ? nothing : Int(v) end),
    nsga2_eta_c=Float64(get_cfg("nsga2_eta_c", 20.0)),
    nsga2_eta_m=Float64(get_cfg("nsga2_eta_m", 20.0)),
    nsga2_pc=Float64(get_cfg("nsga2_pc", 0.9)),
    nsga2_pm=Float64(get_cfg("nsga2_pm", -1.0)),
    nsga2_sobol_init=Bool(get_cfg("nsga2_sobol_init", true)),
    igel_mu=Int(get_cfg("igel_mu", 20)),
    igel_sigma0=Float64(get_cfg("igel_sigma0", 0.3)),
    igel_sobol_init=get_cfg("igel_sobol_init", true),
    igel_sobol_pool_k=Int(get_cfg("igel_sobol_pool_k", 1)),        # >1 ⇒ oversample: keep best μ·k by dominance, random-pick μ
    igel_sobol_raw_mult=Int(get_cfg("igel_sobol_raw_mult", 2)),    # raw Sobol pool = (μ·pool_k)·raw_mult candidates evaluated
    igel_niche_radius=Float64(get_cfg("igel_niche_radius", 0.0)),
    igel_reseed_sigma=Float64(get_cfg("igel_reseed_sigma", 0.0)),
    igel_reseed_random_frac=Float64(get_cfg("igel_reseed_random_frac", 0.0)),
    igel_maturity=Int(get_cfg("igel_maturity", 0)),
    igel_seed_maturity=Bool(get_cfg("igel_seed_maturity", false)),
    igel_freeze_seed_growth=Bool(get_cfg("igel_freeze_seed_growth", false)),
    cc_group_size=Int(get_cfg("cc_group_size", 2)),
    cc_spec_gens=Int(get_cfg("cc_spec_gens", 20)),
    cc_integ_gens=Int(get_cfg("cc_integ_gens", 20)),
    cc_cycles=Int(get_cfg("cc_cycles", 5)),
    cc_fix_others=Bool(get_cfg("cc_fix_others", false)),
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
  base_seed = rand(rng, UInt64)   # per-site RNG = RNGType(hash((base_seed, i))) below → thread-count-invariant & reproducible

  Threads.@threads :static for i in 1:n
    @inbounds begin
      row = site_df[i, :]
      cohorts = splots_dict[row.mapcode]
      site = getsite(soa, i)

      site.active = true
      site.rng = RNGType(hash((base_seed, i)))
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
