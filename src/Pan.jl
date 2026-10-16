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
using .Plugins: BaseSitePlugin, BiomassSuccessionPlugin, PnETPlugin, HBADPlugin
import .Parametrization as PU
import .Parametrization.BiomassSuccessionParametrization as BSP
import .Parametrization.PnETParametrization as PNP
import .Parametrization.HBADParametrization as HBP
using .Search: SA, LBSA, MOLBSA, CMAES, MOCMAES, IgelMOCMAES, CMAMAE, NSGA2, CCIgel, GDIgel, Common
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
include("succession_api.jl")
include("params_yaml.jl")
include("config.jl")
include("options.jl")
include("param_space.jl")

succession_plugin_type(name::AbstractString) = lowercase(name) in ("biomass", "biomasssuccession", "bs") ? _BS :
                                                lowercase(name) in ("pnet", "pnet-succession") ? _PN :
                                                lowercase(name) in ("hbad", "hbad-succession") ? _HB :
                                                error("unknown succession_plugin '$name' (biomass | pnet | hbad)")

"""
    bs_tables_from_cfg(cfg, db_path) -> BSTables

The Biomass Succession data tables of a run (pinned LONGEVITY / SHADE_TOL, seeds for PROB_ESTAB / MATURITY, B_MAX / ANPP / PROB_ESTAB floors), loaded from the
config's csv paths and the cohort database. Pure function of its inputs: nothing is stored in module state, the caller passes the result on explicitly.
"""
function bs_tables_from_cfg(cfg0::AbstractDict, db_path::AbstractString)
  cfg = _as_config(cfg0)
  longevity = cfg["longevity_from_data"] ? _load_longevity_table(db_path) : nothing
  # Per-species LONGEVITY overrides from yaml (`longevity_overrides: {PIEL: 275}`) - applied on top of the
  # data table so hand corrections don't require rewriting the read-only species_longevity_ref DB table.
  let ov = cfg["longevity_overrides"]
    if !isnothing(ov) && !isempty(ov)
      tbl = longevity === nothing ? Dict{String,Float64}() : longevity
      for (k, v) in ov; tbl[String(k)] = Float64(v); end
      longevity = tbl
      @info "longevity_overrides applied: $(Dict(String(k)=>Float64(v) for (k,v) in ov))"
    end
  end
  return BiomassSuccessionPlugin.BSTables(;
    fixed_longevity=(let v = cfg["fix_longevity"]; isnothing(v) ? nothing : Float64(v) end),
    longevity=longevity,
    shade_tol=cfg["shade_tol_from_data"] ? _load_shade_tol_table(cfg["shade_tol_csv"]) : nothing,
    prob_estab=cfg["prob_estab_from_data"] ? _load_prob_estab_table(cfg["prob_estab_csv"], cfg["prob_estab_csv_artificial"]) : nothing,
    maturity=cfg["maturity_from_data"] ? _load_maturity_table(cfg["maturity_csv"]) : nothing,
    bmax_floor=cfg["bmax_floor_from_data"] ? _load_bmax_floor_table(cfg["bmax_floor_csv"]) : nothing,
    anpp_floor=cfg["anpp_floor_from_data"] ? _load_anpp_floor_table(cfg["anpp_floor_csv"]) : nothing)
end

"""
    plugin_config(succession_plugin, species_list, eco_species_ids, eco_list; tables, cohorts_db_path, tablename, pnet_*, hbad_*) -> PluginConfig

Build the immutable plugin context of a run (`BSTables` | `PnETContext` | `HBADContext`) and the site-attribute builder (PnET: per-plot climate id / latitude / soil
from the cohort database). The ONE place `parametrize` and `setup_plugin_from_cfg` construct a plugin; nothing is written to module state.
"""
function plugin_config(succession_plugin::AbstractString, species_list, eco_species_ids, eco_list::Vector{String};
    tables::BiomassSuccessionPlugin.BSTables=BiomassSuccessionPlugin.BSTables(), cohorts_db_path::AbstractString="", tablename::AbstractString="",
    pnet_climate_cache::AbstractString="", pnet_co2_csv::AbstractString="", pnet_generic::AbstractDict=Dict{String,Any}(),
    pnet_soil_texture::AbstractDict=Dict{String,String}(), pnet_soil_csv::AbstractString="",
    hbad_priors_csv::AbstractString="", hbad_coefs_csv::AbstractString="", hbad_init::AbstractString="jitter", hbad_seed_rain::Bool=false,
    hbad_prob_estab_csv::AbstractString="", hbad_cycle_years::Real=8)
  SUCC = succession_plugin_type(succession_plugin)
  if SUCC === PnETPlugin.PnET
    (isempty(pnet_climate_cache) || isempty(pnet_co2_csv)) && error("succession_plugin=pnet requires pnet_climate_cache and pnet_co2_csv")
    ctx = PnETPlugin.PnETContext(pnet_climate_cache, pnet_co2_csv; generic=pnet_generic, db_path=cohorts_db_path, tables=tables)
    @info "PnET succession: climate cache $(basename(pnet_climate_cache)) ($(length(ctx.climate.ids)) L4s, $(ctx.climate.y1)-$(ctx.climate.y2)); fixed constants" ctx.fixed
    soil_map = Dict{String,String}(String(k) => String(v) for (k, v) in pnet_soil_texture)
    return PluginConfig(ctx; site_attrs=sp -> PnETPlugin.build_site_attrs(ctx.climate, cohorts_db_path, tablename, sp, eco_list; soil_map=soil_map, soil_csv=pnet_soil_csv))
  elseif SUCC === HBADPlugin.HBAD
    (species_list === nothing || eco_species_ids === nothing) && error("the hbad plugin needs species_list and eco_species_ids")
    return PluginConfig(HBADPlugin.HBADContext(species_list, eco_species_ids; priors_csv=hbad_priors_csv, coefs_csv=hbad_coefs_csv,
      init_mode=Symbol(lowercase(hbad_init)), estab_prob_csv=hbad_prob_estab_csv, cycle_years=hbad_cycle_years, seed_rain=hbad_seed_rain, tables=tables))
  end
  return PluginConfig(tables)
end

# Tools/diagnostics (tools/tost_common.jl, calib_coord.jl): select and configure the succession plugin from a yaml cfg
# exactly as `parametrize` does. Returns the `PluginConfig` (pass it to `make_sites`, `fit_params`, `generate_params`, ...).
function setup_plugin_from_cfg(cfg0::AbstractDict, eco_list::Vector{String}; species_list=nothing, eco_species_ids=nothing,
                               tables::BiomassSuccessionPlugin.BSTables=BiomassSuccessionPlugin.BSTables())
  cfg = _as_config(cfg0)
  return plugin_config(cfg["succession_plugin"], species_list, eco_species_ids, eco_list; tables=tables,
    cohorts_db_path=cfg["cohorts_db_path"], tablename=cfg["tablename"],
    pnet_climate_cache=cfg["pnet_climate_cache"], pnet_co2_csv=cfg["pnet_co2_csv"], pnet_generic=Dict{String,Any}(cfg["pnet_generic"]),
    pnet_soil_texture=cfg["pnet_soil_texture"], pnet_soil_csv=cfg["pnet_soil_csv"],
    hbad_priors_csv=cfg["hbad_priors_csv"], hbad_coefs_csv=cfg["hbad_coefs_csv"], hbad_init=cfg["hbad_init"], hbad_seed_rain=cfg["hbad_seed_rain"],
    hbad_prob_estab_csv=cfg["hbad_prob_estab_csv"], hbad_cycle_years=cfg["cycle_years"])
end

# SoA type for a given succession plugin P (BaseSite + P). For BiomassSuccession this is exactly `ActiveSoA`.
soa_type(::Type{P}) where {P} = SiteSoA{Tuple{BaseSitePlugin.BaseSite,P}}

# Plugin-specific per-site initialisation inside make_sites (BiomassSuccession: the statements previously inlined there, same order).
function _init_site_plugin!(::Type{_BS}, site, no_establishment::Bool)
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
  return nothing
end
function _init_site_plugin!(::Type{_PN}, site, no_establishment::Bool)
  site.B = zero(FloatType)
  site.c_species .= zero(UIntType)
  site.c_age .= zero(FloatType)
  site.c_bio .= zero(FloatType)
  site.c_tbio .= zero(FloatType)
  site.sp_mature .= false
  return nothing
end

function _init_site_plugin!(::Type{_HB}, site, no_establishment::Bool)
  site.B = zero(FloatType)
  site.BA = zero(FloatType)
  site.SDI = zero(FloatType)
  site.RD = zero(FloatType)
  site.thin_tph = zero(FloatType)
  site.shade_class = zero(UIntType)
  site.no_establish = no_establishment
  site.c_species .= zero(UIntType)
  site.c_age .= zero(FloatType)
  site.c_bio .= zero(FloatType)
  site.c_tph .= zero(FloatType)
  site.c_dq .= zero(FloatType)
  site.c_h .= zero(FloatType)
  site.c_bal .= zero(FloatType)
  site.sp_mature .= false
  site.sp_estab .= false
  return nothing
end

"""
    make_sites(splots, eco_species_ids; plugin=bs_plugin(), rng, spinup, no_establishment, site_attrs=nothing) -> SiteSoA

Initial SoA of a run for the succession plugin of `plugin` (a `PluginConfig`). `site_attrs` (PnET: per-plot climate/soil table) defaults to
`plugin.site_attrs(splots)`.
"""
make_sites(splots::DataFrame, eco_species_ids::Vector{Vector{Int}}; plugin::PluginConfig=bs_plugin(), kw...) =
  _make_sites(plugin, splots, eco_species_ids; kw...)

"""
    Reference

The scoring reference of one split (train, val, or a tool's plot set), built by `build_reference`:

  - `splots`: the split's cohort table.
  - `spdf`: the smoothed, binned observed age distributions (`PU.smoothen_ref_years`).
  - `spdf_plts`: `(plot_id, eco_id) → sim_year → SPDFGroundTruth`, the loss reference.
  - `site_sim_years`: the measurement years of every site.
  - `spinup_cohorts`: the year-deficit cohorts of the Sim B spinup.
  - `injection_cohorts`: the observed cohorts of the injection (and the disturbance data), or `nothing` when the run does not inject.
  - `ref_soa`: the initial `SiteSoA` (`nothing` for `make_soa=false`).
"""
struct Reference{SY,SC,I,R}
  splots::DataFrame
  spdf::DataFrame
  spdf_plts::Dict{Tuple{UIntType,UIntType},Dict{Int,PU.SPDFGroundTruth}}
  site_sim_years::SY
  spinup_cohorts::SC
  injection_cohorts::I
  ref_soa::R
end

"The smoothed observed distributions of `rows` and their `(plot_id, eco_id) → sim_year → SPDFGroundTruth` index (the smoothing / binning step of a reference)."
function reference_tables(rows::DataFrame, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams; debug::Bool=false)
  spdf = PU.smoothen_ref_years(rows, loss_params, Int(maximum(rows.age_calc)); debug=debug)
  (spdf, Data.make_spdf_dict(spdf, eco_species_ids))
end

"""
    build_reference(splots, eco_species_ids, loss_params; scoring=nothing, simb=nothing, no_establishment=false, plugin=nothing,
                    rng=nothing, spinup=true, debug=false) -> Reference

The one place that builds the scoring reference of a split: smoothing and binning, the `(plot, eco) → year` loss index, the site years, the
spinup cohorts, the injection cohorts (only when the run injects: `no_establishment`, `scoring.override` or `simb.simb_disturb_only`) and, when a
`plugin` is given, the initial `SiteSoA` (`make_sites`; consumes `rng`).
"""
function build_reference(splots::DataFrame, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams;
                         scoring::Union{Nothing,ScoringOptions}=nothing, simb::Union{Nothing,SimBOptions}=nothing, no_establishment::Bool=false,
                         plugin::Union{Nothing,PluginConfig}=nothing, rng::Union{Nothing,Random.AbstractRNG}=nothing, spinup::Bool=true, debug::Bool=false)
  spdf, spdf_plts = reference_tables(splots, eco_species_ids, loss_params; debug=debug)
  ssy = Data.get_site_sim_years(spdf)
  spc = Data.get_spinup_cohorts(splots)
  # simb_disturb_only builds the injection cohorts purely as the disturbance-data source (all_cohorts=true so the drops cover the whole stand);
  # it never injects.
  all_coh = (scoring !== nothing && scoring.override) || (simb !== nothing && simb.simb_disturb_only)
  inj = (no_establishment || all_coh) ? Data.get_injection_cohorts(splots; all_cohorts=all_coh) : nothing
  soa = plugin === nothing ? nothing : make_sites(splots, eco_species_ids; rng=rng, spinup=spinup, no_establishment=no_establishment, plugin=plugin)
  Reference(splots, spdf, spdf_plts, ssy, spc, inj, soa)
end

function _make_sites(plugin::PluginConfig{P}, splots::DataFrame, eco_species_ids::Vector{Vector{Int}}; rng::Random.AbstractRNG, spinup::Bool=true, no_establishment::Bool=false,
                     site_attrs=nothing) where {P}
  pc = plugin.ctx
  site_attrs === nothing && (site_attrs = plugin.site_attrs(splots))
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
    soa = soa_type(P)((cohort=cohort_counts, species=species_counts))
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
        _init_site_plugin!(P, site, no_establishment)
      end
    end
    init_sites!(pc, soa, site_attrs)
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
    soa = soa_type(P)((cohort=cohort_counts, species=species_counts))
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
        _init_site_plugin!(P, site, no_establishment)
        for row in eachrow(initial_cohorts)
          add_cohort!(pc, site, UIntType(row.eco_species_id), FloatType(row.age_calc), FloatType(row.agb_sum))
        end
      end
    end
    init_sites!(pc, soa, site_attrs)
    return soa
  end
end


# Each cohort tuple = (eco_species_id, age, observed_biomass, disturbance_drop_pct).
const _SiteInjectionYear = Vector{Tuple{Int,Vector{Tuple{UIntType,FloatType,FloatType,FloatType}}}}
const _SiteInjectionDict = Dict{Int,_SiteInjectionYear}

# Injection modes (ScoringOptions, built once per run by `scoring_from_config`; src/options.jl):
#   override=true: get_injection_cohorts returns the full observed state at every measurement year, and _inject_observed_cohorts!
#     forces each observed cohort onto the site — overwriting the biomass of the matching (species, age) cohort in place, or adding
#     it if absent — so sim matches obs at every loss point. For the sensitivity test, noise > 0 is a multiplicative jitter sd on
#     injected biomass.
#   replace=true switches override to FULL REPLACE: each measured site is wiped (live/old/B reset) and rebuilt from exactly the
#     observed cohorts, dropping sim-only cohorts (ones that should have died but didn't). Gives an exact multi-cohort state for
#     stress-testing the CSR indexing. Requires override (for the data).
#   sync=true: SET-SYNC mode (choice B) — the injection manages the cohort SET to track the data (remove sim cohorts whose
#     (species,age) isn't observed = "died"; add observed cohorts the sim lacks = "recruited", with observed biomass) but LEAVES
#     matched survivors' sim biomass untouched, so the simulator fits their biomass deterministically. Requires override; mutually
#     exclusive with replace (replace takes precedence).
#   loss_before_injection=true (the default for sync): at a measurement year the loss (every search tier) is computed on the SIMULATED
#     state — i.e. BEFORE the observed cohorts are injected for the next interval. The previous order (inject, then score) compared a
#     state whose cohort SET had already been forced to the observed one (sync: unobserved sim cohorts removed, missing observed
#     cohorts added with observed biomass), so the age-structure (Wasserstein) term largely scored observed against observed.
#     `loss_before_injection: false` reproduces the previous behaviour (and earlier results).
#
# Partial-disturbance handling in SYNC mode (ScoringOptions.disturbance). Each observed cohort carries a disturbance_drop_pct
# (from curation). Modes:
#   :off              — ignore drop (current behaviour).
#   :scale            — scale the matched (survivor) sim cohort's biomass by (1-drop) at the
#                       disturbance year (one-time shock); the cohort STAYS in the loss, so growth
#                       params fit the undisturbed trajectory with the disturbance applied.
#   :exclude_overwrite— overwrite the matched cohort with the observed (already-reduced) biomass and
#                       EXCLUDE it from the site loss at that year.
#   :exclude_noscale  — leave the sim biomass and EXCLUDE the cohort from the site loss at that year.

# Disturbance SHOCK (new scoring order, loss_before_injection + override sync = `shock_order(opts)`). Applied at every measurement year T
# AFTER the growth step and BEFORE scoring, on the simulated state:
#   (a) sim cohorts matching (eco_species, age) of a cohort FULLY removed by disturbance in (T_prev→T) are removed;
#   (b) :scale — matched partially-disturbed survivors (observed drop > 0) are scaled by (1-drop).
# Undisturbed cohorts (including ones that died naturally in the data: the sim must kill them) are untouched and scored as simulated.
# Data: plot-level full removals from Data._plot_shocks! (tools/export_cohort_disturbance.py parquet).
const _SiteShockYear = Vector{Tuple{Int,Vector{Tuple{UIntType,FloatType}}}}
const _SiteShockDict = Dict{Int,_SiteShockYear}
# diagnostics (tools/shock_stats.jl): [full-removal rows tried, matched+removed sim cohorts, partial (drop>0) cohorts, matched+scaled]
const SHOCK_STATS = [Threads.Atomic{Int}(0) for _ in 1:4]

function _build_shock_dict(shock_rows::DataFrame, ref_soa)::_SiteShockDict
  plot_to_site = Dict{Int,Int}(Int(getsite(ref_soa, i).ref_cn) => i for i in 1:ref_soa.n)
  by_year = _SiteShockDict()
  pos = Dict{Tuple{Int,Int},Int}()
  for row in eachrow(shock_rows)
    site_idx = get(plot_to_site, Int(row.plot_id), 0)
    site_idx == 0 && continue
    year = Int(row.sim_year)
    yl = get!(by_year, year, _SiteShockYear())
    p = get(pos, (site_idx, year), 0)
    if p == 0
      push!(yl, (site_idx, Tuple{UIntType,FloatType}[]))
      p = length(yl); pos[(site_idx, year)] = p
    end
    push!(yl[p][2], (UIntType(row.eco_species_id), FloatType(row.age_calc)))
  end
  by_year
end

# ---- cohort-level exclusion at a measurement year (new order; replaces the species-level exclusion_dict) ----------------------
# A cohort (eco_species, age) at (site, T) is EXCLUDED from the loss on BOTH sides when it is
#   * partially disturbed (drop > 0) and the mode is :exclude_overwrite / :exclude_noscale, or
#   * a RECRUIT at entry (ScoringOptions.exclude_recruits_at_entry): observed at T but with no record of the same cohort (species, age-(T-T_prev))
#     at the previous visit of that plot — Sim A has no establishment, so it cannot produce it; it enters via the sync afterwards.
# Sim side: the cohort is skipped (PU._excl_perm!). Reference side: an alternative SPDFGroundTruth for that site-year is built from the
# observed cohorts WITHOUT the excluded ones (same smoothing/binning code: PU.smoothen_ref_years). A species left with no cohort on
# either side contributes nothing (no "missing species" penalty).
struct _ExclPlan
  mode::Symbol
  recruits::Bool
  coh::Dict{Int,Dict{Int,Vector{Tuple{UIntType,FloatType}}}}   # year → site_idx → excluded (eco_species, age)
  ref::Dict{Int,Dict{Int,PU.SPDFGroundTruth}}                  # year → site_idx → reference without those cohorts
  prev::Dict{Int,Dict{Int,Int}}                                # year → site_idx → previous visit year (absent for a site's first visit)
end

function _build_excl_plan(injection_dict, ref_soa, eco_species_ids, loss_params, opts::ScoringOptions)::_ExclPlan
  mode = opts.disturbance; rec_on = opts.exclude_recruits_at_entry
  dist_on = mode in (:exclude_overwrite, :exclude_noscale)
  site_years = Dict{Int,Vector{Int}}()
  site_set = Dict{Tuple{Int,Int},Set{Tuple{UIntType,Int}}}()
  for (yr, sites) in injection_dict, (si, cohorts) in sites
    push!(get!(site_years, si, Int[]), yr)
    site_set[(si, yr)] = Set{Tuple{UIntType,Int}}((sp, Int(age)) for (sp, age, _, _) in cohorts)
  end
  for v in values(site_years); sort!(v); end
  coh = Dict{Int,Dict{Int,Vector{Tuple{UIntType,FloatType}}}}()
  rows = DataFrame(plot_id=UIntType[], eco_id=UIntType[], measdate=Dates.Date[], start_measdate=Dates.Date[], eco_species_id=UIntType[], age_calc=Int[], agb_sum=FloatType[])
  d0 = Dates.Date(2000, 1, 1)
  for yr in sort!(collect(keys(injection_dict))), (si, cohorts) in injection_dict[yr]
    ys = site_years[si]; k = searchsortedfirst(ys, yr)
    py = k > 1 ? ys[k-1] : -1   # previous visit year (year 0 is a valid previous visit)
    ex = Tuple{UIntType,FloatType}[]
    for (sp, age, bio, drop) in cohorts
      isdist = dist_on && drop > zero(FloatType)
      isrec = rec_on && py >= 0 && !((sp, Int(age) - (yr - py)) in site_set[(si, py)])
      (isdist || isrec) && push!(ex, (sp, age))
    end
    isempty(ex) && continue
    get!(coh, yr, Dict{Int,Vector{Tuple{UIntType,FloatType}}}())[si] = ex
    eco = UIntType(getsite(ref_soa, si).eco_id)
    for (sp, age, bio, _) in cohorts
      (sp, age) in ex && continue
      push!(rows, (UIntType(si), eco, d0 + Dates.Day(round(Int, yr * 365.25)), d0, sp, Int(age), bio))
    end
  end
  ref = Dict{Int,Dict{Int,PU.SPDFGroundTruth}}()
  alt = Dict{Tuple{Int,Int},PU.SPDFGroundTruth}()
  if nrow(rows) > 0
    for ((pid, eco), byyr) in reference_tables(rows, eco_species_ids, loss_params)[2], (yr, gt) in byyr
      alt[(Int(pid), yr)] = gt
    end
  end
  for (yr, d) in coh, si in keys(d)
    nsp = length(eco_species_ids[Int(getsite(ref_soa, si).eco_id)])
    gt = get(alt, (si, yr), nothing)
    gt === nothing && (gt = PU.SPDFGroundTruth(keys=falses(nsp), records=Dict{UIntType,PU.SPDFRecord}()))
    get!(ref, yr, Dict{Int,PU.SPDFGroundTruth}())[si] = gt
  end
  prev = Dict{Int,Dict{Int,Int}}()
  for (si, ys) in site_years, k in 2:length(ys); get!(prev, ys[k], Dict{Int,Int}())[si] = ys[k-1]; end
  _ExclPlan(mode, rec_on, coh, ref, prev)
end

# ---- Sim A scoring modes for the natural-death part (docs/simAB_options_experiment.md) ------------------------------------
# w_scope     :all       — W on every simulated cohort (sim-only natural deaths included; current behaviour)
#             :survivors — W on cohorts present in both sim and obs at T: sim cohorts absent from the observed set are removed before scoring
#                          (the sync injection removes them right after anyway). Needs the new order + sync.
# death_term  :none | :likelihood — per-cohort survive/die negative log-likelihood given PROB_MORT p over the interval Δt (years between
#                          the two visits): survivor −Δt·log(1−p); natural death −log(1−(1−p)^Δt). PROB_MORT is NOT applied in the Sim A
#                          simulation (zeroed inside fit_params when a DeathAcc is passed); it only enters the death term.
const DEATH_PMIN = 1e-6
const DEATH_PMAX = 0.5
const DEATH_MAXDT = 80
"Per-cell (eco e, local species j) survivor/death statistics: surv_dt = Σ_survivors Δt, n_surv, dhist[k] = # natural deaths with Δt = k."
struct DeathAcc
  surv_dt::Vector{Vector{Float64}}
  n_surv::Vector{Vector{Int}}
  dhist::Vector{Vector{Vector{Int}}}
end
DeathAcc(eco_species_ids) = DeathAcc([zeros(length(ids)) for ids in eco_species_ids], [zeros(Int, length(ids)) for ids in eco_species_ids],
                                      [[zeros(Int, DEATH_MAXDT) for _ in ids] for ids in eco_species_ids])
@inline _pclamp(p) = clamp(Float64(p), DEATH_PMIN, DEATH_PMAX)
"Negative log-likelihood of one cohort-interval outcome."
@inline death_nll_survivor(p, dt) = -dt * log1p(-_pclamp(p))
@inline death_nll_death(p, dt) = -log(-expm1(dt * log1p(-_pclamp(p))))     # −log(1−(1−p)^Δt)
"Per-cell (Σ NLL, # cohorts) given the cell's p."
function death_cell(acc::DeathAcc, e::Int, j::Int, p)
  n = acc.n_surv[e][j]; s = death_nll_survivor(p, 1.0) * acc.surv_dt[e][j]
  for k in 1:DEATH_MAXDT
    c = acc.dhist[e][j][k]; c > 0 && (s += c * death_nll_death(p, Float64(k)); n += c)
  end
  s, n
end

# Collect survive/die statistics at year T (before survivors-only removal). Survivor = sim cohort present in obs at T; natural death = sim
# cohort absent from obs at T (disturbance-removed cohorts were already removed by the shock; excluded cohorts are skipped).
function _death_collect!(acc::DeathAcc, soa, year, injection_dict, plan)
  pv = get(plan.prev, year, nothing); pv === nothing && return
  for (si, cohorts) in injection_dict[year]
    site = getsite(soa, si); site.active || continue
    py = get(pv, si, -1); py < 0 && continue
    dt = min(year - py, DEATH_MAXDT)
    exl = (yd = get(plan.coh, year, nothing); yd === nothing ? nothing : get(yd, si, nothing))
    e = Int(site.eco_id); csp = site.c_species; cage = site.c_age
    for j in 1:Int(site.live)
      sp = csp[j]; age = cage[j]
      exl !== nothing && any(t -> t[1] == sp && t[2] == age, exl) && continue
      inobs = false
      for (osp, oage, _, _) in cohorts; (osp == sp && oage == age) && (inobs = true; break); end
      if inobs; acc.surv_dt[e][Int(sp)] += dt; acc.n_surv[e][Int(sp)] += 1
      else; acc.dhist[e][Int(sp)][dt] += 1; end
    end
  end
  nothing
end

# Survivors-only scoring: remove sim cohorts absent from the observed set at T (see ScoringOptions.w_scope).
function _survivors_only!(soa, year, injection_dict)
  for (si, cohorts) in injection_dict[year]
    site = getsite(soa, si); site.active || continue
    csp = site.c_species; cage = site.c_age; n0 = Int(site.live)
    j = 1
    while j <= Int(site.live)
      inobs = false
      for (sp, age, _, _) in cohorts; (csp[j] == sp && cage[j] == age) && (inobs = true; break); end
      inobs ? (j += 1) : _shock_remove_cohort!(site, j)
    end
    if Int(site.live) != n0
      o = zero(UIntType)
      for q in 1:Int(site.live); cage[q] > one(FloatType) && (o += one(UIntType)); end
      site.old = o
    end
  end
  nothing
end

# DIAGNOSTIC ONLY (default :none = no effect). ScoringOptions.w_diag at a measurement year, new order, just before scoring:
#   :stats   — accumulate cohort-set statistics sim-vs-obs into the WDiag collector (no state change)
#   :nodeath — additionally REMOVE sim cohorts that are absent from the observed set (oracle natural deaths)
#   :oracle  — run the sync injection BEFORE scoring (sim set := obs set; matched keep sim biomass): the shape-only W
# The statistics are collected into a `WDiag` that the caller passes to fit_params(...; w_diag=WDiag(...)) and reads afterwards (tools/w_decompose.jl);
# without one, the :nodeath / :oracle state changes still apply and the statistics go to a throw-away collector.
#   stats: 16 counters (see the S[...] updates below); cat: rows 1 both-exact 2 both-mismatched 3 ref-only(missing species) 4 both-but-nothing-matched
#   5/6 ref-only without / with a previous visit; cols ΣW, count. ctx (optional, for the per-species W categories of :stats):
#   (spdf_plts, n_species, eco_species_ids, loss_params, scales).
struct WDiag{C}
  stats::Vector{Float64}
  cat::Matrix{Float64}
  ctx::C
end
WDiag(; ctx=nothing) = WDiag(zeros(Float64, 16), zeros(Float64, 6, 2), ctx)
function _w_diag_apply!(pc, soa, year, injection_dict, plan, opts::ScoringOptions, wd::WDiag)
  mode = opts.w_diag
  S = wd.stats
  for (si, cohorts) in injection_dict[year]
    site = getsite(soa, si); site.active || continue
    exl = plan === nothing ? nothing : (yd = get(plan.coh, year, nothing); yd === nothing ? nothing : get(yd, si, nothing))
    isex(sp, age) = exl !== nothing && any(e -> e[1] == sp && e[2] == age, exl)
    csp = site.c_species; cage = site.c_age; cbio = site.c_bio
    S[15] += 1
    obs_sp = Set{UIntType}(); sim_sp = Set{UIntType}()
    for (sp, age, bio, _) in cohorts
      isex(sp, age) && (S[13] += bio; continue)
      push!(obs_sp, sp)
      found = false
      for j in 1:Int(site.live); csp[j] == sp && cage[j] == age && (found = true; break); end
      if found; S[1] += 1; S[3] += bio; else
        S[6] += 1; S[7] += bio
      end
    end
    for j in 1:Int(site.live)
      isex(csp[j], cage[j]) && (S[12] += 1; continue)
      push!(sim_sp, csp[j])
      inobs = false
      for (sp, age, _, _) in cohorts; (csp[j] == sp && cage[j] == age) && (inobs = true; break); end
      if inobs; S[2] += cbio[j]; else
        S[4] += 1; S[5] += cbio[j]
        for (sp, age, _, _) in cohorts   # age near-miss: same species, |age diff| == 1 present in obs
          (csp[j] == sp && abs(cage[j] - age) == 1) && (S[14] += 1; break)
        end
      end
    end
    for (sp, age, bio, _) in cohorts; isex(sp, age) && continue; sp in sim_sp || (S[8] += 1; S[9] += bio); end
    for j in 1:Int(site.live); isex(csp[j], cage[j]) && continue; csp[j] in obs_sp || (S[10] += 1; S[11] += cbio[j]); end
    if wd.ctx !== nothing && mode === :stats
      (spdf_plts, n_species, esids, lpar, lsc) = wd.ctx
      gt = spdf_plts[(site.ref_cn, site.eco_id)][year]; ecl = nothing
      if exl !== nothing; gt = plan.ref[year][si]; ecl = exl; end
      sl = PU.calculate_site_loss2(year, site, n_species, esids, gt, lpar, lsc; excl_c=ecl)
      smap = esids[Int(site.eco_id)]
      for spj in 1:length(gt.keys)
        gt.keys[spj] || continue
        w = Float64(sl.sp_w_loss[smap[spj]])
        simset = Set{Float64}(Float64(cage[j]) for j in 1:Int(site.live) if csp[j] == spj && !isex(csp[j], cage[j]))
        obsset = Set{Float64}(Float64(a) for (sp, a, _, _) in cohorts if sp == spj && !isex(sp, a))
        c = isempty(simset) ? 3 : (simset == obsset ? 1 : (isempty(intersect(simset, obsset)) ? 4 : 2))
        wd.cat[c, 1] += w; wd.cat[c, 2] += 1
        if c == 3   # why is the species absent in the sim? look at the site's previous observed visit
          py = -1
          for (y2, lst) in injection_dict; y2 < year && y2 > py && any(t -> t[1] == si, lst) && (py = y2); end
          if py < 0; wd.cat[5, 2] += 1; wd.cat[5, 1] += w   # no previous visit
          else
            prev = first(t[2] for t in injection_dict[py] if t[1] == si)
            if any(t -> t[1] == spj, prev); wd.cat[6, 2] += 1; wd.cat[6, 1] += w   # species WAS observed at the previous visit
            else; wd.cat[2, 1] += 0; end
          end
        end
      end
    end
    if mode === :nodeath
      j = 1
      while j <= Int(site.live)
        inobs = false
        for (sp, age, _, _) in cohorts; (csp[j] == sp && cage[j] == age) && (inobs = true; break); end
        inobs ? (j += 1) : _shock_remove_cohort!(site, j)
      end
    end
  end
  mode === :oracle && _inject_observed_cohorts!(pc, soa, injection_dict[year], opts; override=true, replace=false, sync=true, skip_scale=true)
  nothing
end

# Swap-remove sim cohort i (same mechanics as the sync's Pass A).
@inline function _shock_remove_cohort!(site, i::Int)
  cbio = site.c_bio
  site.B -= cbio[i]
  lastidx = Int(site.live)
  i != lastidx && PanCore.swap_cohort!(site, i, lastidx)
  site.live -= one(UIntType)
  nothing
end

# Shock step (a): remove the sim cohorts matching (eco_species, age) of a cohort FULLY removed by disturbance since the previous visit.
function _shock_full_removals!(soa, shock_year::_SiteShockYear)
  for (site_idx, lst) in shock_year
    site = getsite(soa, site_idx)
    site.active || continue
    csp = site.c_species; cage = site.c_age
    removed = false
    for (sp, age) in lst
      Threads.atomic_add!(SHOCK_STATS[1], 1)
      @inbounds for j in 1:Int(site.live)
        if csp[j] == sp && cage[j] == age
          _shock_remove_cohort!(site, j)
          Threads.atomic_add!(SHOCK_STATS[2], 1)
          removed = true
          break
        end
      end
    end
    if removed
      o = zero(UIntType)
      for j in 1:Int(site.live)
        cage[j] > one(FloatType) && (o += one(UIntType))
      end
      site.old = o
      site._new_cohort_counts = Int(site.live)
    end
  end
  nothing
end
# Shock step (b), mode :scale: scale the matched, partially disturbed survivors (observed drop > 0) by (1-drop).
function _shock_scale_partial!(pc, soa, site_cohorts::_SiteInjectionYear)
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    csp = site.c_species; cage = site.c_age; cbio = site.c_bio
    for (sp, age, _bio, drop) in cohorts
      drop > zero(FloatType) || continue
      Threads.atomic_add!(SHOCK_STATS[3], 1)
      @inbounds for j in 1:Int(site.live)
        if csp[j] == sp && cage[j] == age
          Threads.atomic_add!(SHOCK_STATS[4], 1)
          newb = cbio[j] * (one(FloatType) - drop)
          site.B += newb - cbio[j]
          set_cohort_bio!(pc, site, j, newb)
          break
        end
      end
    end
  end
  nothing
end
# The shock step of the scoring rule: (a) then (b), on the simulated state, before scoring.
function _apply_disturbance_shock!(pc, soa, shock_year::Union{Nothing,_SiteShockYear}, site_cohorts::Union{Nothing,_SiteInjectionYear}, opts::ScoringOptions)
  shock_year === nothing || _shock_full_removals!(soa, shock_year)
  (site_cohorts !== nothing && opts.disturbance === :scale) && _shock_scale_partial!(pc, soa, site_cohorts)
  nothing
end

# ---- Observed-cohort injection, one function per mode -----------------------------------------------------------------------------
# Only touches the sites that actually have injections (typically << n_sites).
# _new_cohort_counts is already == site.live for all other sites after process_plugin!.
# The mode is a singleton type chosen once per call by `_inject_mode(override, replace, sync)` (replace beats sync):
#   InjAppend   (override=false): append every listed cohort as a new cohort.
#   InjReplace  (override, replace): wipe each measured site and rebuild it from exactly the observed cohorts.
#   InjSync     (override, sync): the cohort SET tracks the data, matched survivors keep their sim biomass (steps below).
#   InjOverride (override only): overwrite the biomass of the matching (species, age) cohort in place; add it only if no match exists.
abstract type InjectMode end
struct InjAppend   <: InjectMode end
struct InjReplace  <: InjectMode end
struct InjSync     <: InjectMode end
struct InjOverride <: InjectMode end
@inline _inject_mode(override::Bool, replace::Bool, sync::Bool)::Union{InjAppend,InjReplace,InjSync,InjOverride} =
  !override ? InjAppend() : replace ? InjReplace() : sync ? InjSync() : InjOverride()

# Sim cohort (species, age) lookup over the first `n` cohorts of the site.
@inline function _has_cohort(csp, cage, n::Int, sp, age)::Bool
  for j in 1:n
    (csp[j] == sp && cage[j] == age) && return true
  end
  false
end
# Biomass of an injected cohort: the observed value, with the multiplicative noise of the sensitivity test when noise > 0.
@inline _noisy_bio(site, bio::FloatType, noise::FloatType)::FloatType =
  noise > zero(FloatType) ? abs(bio * (one(FloatType) + noise * randn(site.rng, FloatType))) : bio
# Append (species, age) with RAW biomass b (overwrites add_cohort!'s integer truncation, which only the model proper uses; C# parity).
@inline function _add_cohort_raw!(pc, site, sp, age, b::FloatType)
  add_cohort!(pc, site, sp, age, b)
  set_cohort_bio!(pc, site, Int(site.live), b)
  site.B += b
  nothing
end
# Recompute site.old (# cohorts with age > 1) after cohorts were removed / added.
@inline function _recompute_old!(site)
  cage = site.c_age
  o = zero(UIntType)
  for j in 1:Int(site.live)
    cage[j] > one(FloatType) && (o += one(UIntType))
  end
  site.old = o
  nothing
end

function _inject!(::InjAppend, pc, soa, site_cohorts::_SiteInjectionYear, opts::ScoringOptions, skip_scale::Bool)
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
      add_cohort!(pc, site, sp, age, bio)
      site.B += bio
    end
  end
  nothing
end

# Full replace: wipe each measured site and rebuild it from exactly the observed
# cohorts (raw biomass, no trunc; optional noise). Drops sim-only cohorts so the
# site state equals the data even when some sim cohorts should have died but didn't.
function _inject!(::InjReplace, pc, soa, site_cohorts::_SiteInjectionYear, opts::ScoringOptions, skip_scale::Bool)
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    site._new_cohort_counts = length(cohorts)
  end
  PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))
  noise = FloatType(opts.noise)
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    site.live = zero(UIntType)
    site.old = zero(UIntType)
    site.B = zero(FloatType)
    for (sp, age, bio, _drop) in cohorts
      _add_cohort_raw!(pc, site, sp, age, _noisy_bio(site, bio, noise))
    end
  end
  nothing
end

# Set-sync (choice B): cohort SET tracks the data; matched survivors keep their sim biomass. Four steps, in this order.
# Step A: remove sim cohorts whose (species,age) isn't in the observed set ("died"), in place.
function _sync_remove_unobserved!(soa, site_cohorts::_SiteInjectionYear)
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
          PanCore.swap_cohort!(site, i, lastidx)   # moves every per-cohort CSR field
        end
        site.live -= one(UIntType)
      end
    end
  end
  nothing
end
# Disturbance handling for matched (survivor) cohorts at the disturbance year. Only survivors
# (present after step A) are touched; recruits added in step C keep their observed biomass.
#   :scale            — biomass *= (1-drop): a known shock; cohort STAYS in the loss (growth fits it).
#   :exclude_overwrite— biomass := observed survivor biomass; cohort EXCLUDED from loss (loss side).
#   :exclude_noscale  — biomass unchanged; cohort EXCLUDED from loss (loss side).
# `skip_scale`: the (1-drop) shock was already applied before scoring.
function _sync_disturbed_survivors!(pc, soa, site_cohorts::_SiteInjectionYear, dmode::Symbol, skip_scale::Bool)
  ((dmode === :scale && !skip_scale) || dmode === :exclude_overwrite) || return nothing
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
          set_cohort_bio!(pc, site, j, newb)
          break
        end
      end
    end
  end
  nothing
end
# Step B: size each site to survivors + (# observed (species,age) absent from the sim).
function _sync_size_sites!(soa, site_cohorts::_SiteInjectionYear)
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    csp = site.c_species
    cage = site.c_age
    live = Int(site.live)
    n_new = 0
    for (sp, age, _, _) in cohorts
      _has_cohort(csp, cage, live, sp, age) || (n_new += 1)
    end
    site._new_cohort_counts = live + n_new
  end
  PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))
  nothing
end
# Step C: append the unmatched observed cohorts (recruits) with observed biomass; recompute old.
function _sync_append_recruits!(pc, soa, site_cohorts::_SiteInjectionYear, noise::FloatType)
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    csp = site.c_species
    cage = site.c_age
    orig_live = Int(site.live)
    for (sp, age, bio, _drop) in cohorts
      _has_cohort(csp, cage, orig_live, sp, age) && continue
      _add_cohort_raw!(pc, site, sp, age, _noisy_bio(site, bio, noise))
    end
    # step A removals + step C adds left site.old stale
    _recompute_old!(site)
  end
  nothing
end
function _inject!(::InjSync, pc, soa, site_cohorts::_SiteInjectionYear, opts::ScoringOptions, skip_scale::Bool)
  _sync_remove_unobserved!(soa, site_cohorts)
  _sync_disturbed_survivors!(pc, soa, site_cohorts, opts.disturbance, skip_scale)
  _sync_size_sites!(soa, site_cohorts)
  _sync_append_recruits!(pc, soa, site_cohorts, FloatType(opts.noise))
  nothing
end

# Override in place (no sync): overwrite matches in place (searching only the original live range), append the unmatched.
# Biomass is written RAW (no trunc) so the injected state equals the reference exactly — only this diagnostic path skips the
# integer-valued truncation that add_cohort!/succession use everywhere else (C# parity).
function _inject!(::InjOverride, pc, soa, site_cohorts::_SiteInjectionYear, opts::ScoringOptions, skip_scale::Bool)
  # Pass 1: size each site to live + (# observed cohorts with no (species,age) match).
  _sync_size_sites!(soa, site_cohorts)
  # Pass 2
  noise = FloatType(opts.noise)
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    csp = site.c_species
    cage = site.c_age
    cbio = site.c_bio
    orig_live = Int(site.live)
    for (sp, age, bio, _drop) in cohorts
      b = _noisy_bio(site, bio, noise)
      matched = false
      for j in 1:orig_live
        if csp[j] == sp && cage[j] == age
          site.B += b - cbio[j]
          set_cohort_bio!(pc, site, j, b)
          matched = true
          break
        end
      end
      matched || _add_cohort_raw!(pc, site, sp, age, b)
    end
  end
  nothing
end

# Entry point of the injection step of the scoring rule (after the shock and the exclusion plan have been applied / consulted).
function _inject_observed_cohorts!(pc, soa, site_cohorts::_SiteInjectionYear, opts::ScoringOptions; override::Bool=opts.override, replace::Bool=opts.replace, sync::Bool=opts.sync, skip_scale::Bool=false)
  _inject!(_inject_mode(override, replace, sync), pc, soa, site_cohorts, opts, skip_scale)
  nothing
end

# Pre-index by sim_year → [(site_idx, cohorts)] using the ref_soa's plot→site mapping.
# Called once per reference by build_injection_plan; eliminates the per-site dict lookup inside _inject_observed_cohorts!.
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

"""
    InjectionPlan

Everything the observed-cohort injection needs for ONE reference (train Sim A, val Sim A, Sim B's `dual_b`, val `dual_b`), built once by
`build_injection_plan` and passed with that reference (`fit_params(...; injection=plan)`):

  - `dict`, `years`: the injection cohorts by sim year → [(site, cohorts)], and the set of measurement years that inject.
  - `shock`: the disturbance-shock dict (full removals by disturbance, from the injection data's `disturbance_shock` metadata), or `nothing`.
  - `excl`: the cohort-level exclusion plan of the shock order (disturbed cohorts and recruits at entry), or `nothing` outside that order.
  - `legacy_excl`: the species-level exclusion of the legacy order with `disturbance = :exclude_*`, or `nothing`.
  - `opts`: the `ScoringOptions` the plan was built under; `fit_params` refuses a plan built under a different shock order, disturbance
    mode or recruit rule (the exclusion data depend on them).

The plan is immutable and read-only during a search, so the site loop of any thread (and the candidate-parallel batches) can share it.
"""
struct InjectionPlan
  dict::_SiteInjectionDict
  years::Set{Int}
  shock::Union{Nothing,_SiteShockDict}
  excl::Union{Nothing,_ExclPlan}
  legacy_excl::Union{Nothing,Dict{Int,Dict{Int,Set{UIntType}}}}
  opts::ScoringOptions
end

# species-level exclusion (legacy order): year → site_idx → species with a partially disturbed cohort
function _legacy_exclusion(injection_dict)
  exclusion_dict = Dict{Int,Dict{Int,Set{UIntType}}}()
  for (yr, sites) in injection_dict
    sd = Dict{Int,Set{UIntType}}()
    for (site_idx, cohorts) in sites
      s = Set{UIntType}(sp for (sp, _age, _bio, drop) in cohorts if drop > zero(FloatType))
      isempty(s) || (sd[site_idx] = s)
    end
    isempty(sd) || (exclusion_dict[yr] = sd)
  end
  exclusion_dict
end

"""
    build_injection_plan(injection_cohorts, ref_soa, eco_species_ids, loss_params, opts) -> Union{Nothing,InjectionPlan}

The injection plan of the reference `ref_soa` from its observed injection cohorts (`Data.get_injection_cohorts`); `nothing` when there are none.
"""
function build_injection_plan(injection_cohorts, ref_soa, eco_species_ids, loss_params, opts::ScoringOptions)::Union{Nothing,InjectionPlan}
  injection_cohorts === nothing && return nothing
  dict = _build_injection_dict(injection_cohorts, ref_soa)
  years = Set(Int.(injection_cohorts.sim_year))
  shock = "disturbance_shock" in metadatakeys(injection_cohorts) ?
    _build_shock_dict(metadata(injection_cohorts, "disturbance_shock"), ref_soa) : nothing
  do_shock = shock_order(opts)
  excl = do_shock ? _build_excl_plan(dict, ref_soa, eco_species_ids, loss_params, opts) : nothing   # cohort-level exclusion (new order)
  legacy = (!do_shock && opts.disturbance in (:exclude_overwrite, :exclude_noscale)) ? _legacy_exclusion(dict) : nothing
  return InjectionPlan(dict, years, shock, excl, legacy, opts)
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

# Persist the exact train/val/test plot assignment (+ eco cell, + dominant species when train_cap ran) so floor
# builders / analysis scripts reuse the run's split instead of re-deriving it (scripts drift from training kwargs).
function _write_split_plots(output_dir::AbstractString, eco_list, splots, sv, st, outs::Data.DataOutputs)
  mkpath(output_dir)
  ids = [:statecd, :unitcd, :countycd, :plot]
  parts = DataFrame[]
  for (set, d) in (("train", splots), ("val", sv), ("test", st))
    (d === nothing || nrow(d) == 0) && continue
    u = unique(select(d, vcat(ids, [:eco_id])))
    u.eco = [eco_list[i] for i in u.eco_id]; u.set .= set
    push!(parts, select(u, vcat(ids, [:eco, :set])))
  end
  isempty(parts) && return
  out = vcat(parts...)
  dd = outs.split_dom
  out = dd === nothing ? out : leftjoin(out, select(dd, vcat(ids, [:dom])), on=ids)
  sort!(out, vcat([:set], ids))
  CSV.write(joinpath(output_dir, "split_plots.csv"), out)
  CSV.write(joinpath(output_dir, "species_map.csv"), outs.species_map)   # symbol → category (for floor builders)
  @info "split_plots.csv: $(nrow(out)) plots → $(output_dir)"
end

# search drivers reachable from `parametrize` (sobol returns early; plot_only* are handled in run_from_yaml)
const SEARCH_MODES = ("lbsa", "sa", "molbsa", "cmaes", "mocmaes", "igelmo", "cmame", "ccigel", "nsga2", "gdigel")
function _check_search_mode(m::AbstractString)
  (m in SEARCH_MODES || m == "sobol") && return nothing
  error("unknown search_mode $(repr(m)); valid: $(join(SEARCH_MODES, ", ")), sobol, plot_only, plot_only_sobol")
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
  w_count_pct_floor::Float64=0.10,      # floor on the per-bin count-share → effective #cohorts (neff=pct·tot); lower ⇒ rare bins get MORE weight
  rankw_mode::String="rank",            # per-(species,stratum) objective weight: "rank" (1/√ln(rank) by AGB) or "cbal_pct" (percentile-floored class-balanced by cohort count)
  rankw_beta::Float64=0.999,            # β for rankw_mode=cbal_pct
  rankw_pct_floor::Float64=0.10,        # cohort-count-share floor+grid for rankw_mode=cbal_pct (species×strata; twin of w_count_pct_floor)
  param_split_species::AbstractDict=Dict{String,Vector{String}}(),  # {param_name => [species]} fit per-eco; others tied across ecos
  param_tier_merge::AbstractDict=Dict{String,Any}(),  # {param_name => {species => {cell => group}}} tie split site-cells per species
  param_pooling::AbstractDict=Dict{String,Any}(),     # region runs: L3→L2→region pooling of EcoSpecies params (see _build_pool_labels)
  succession_plugin::String="biomass",                # "biomass" (BiomassSuccession, default) | "pnet" (PnET ecophysiology)
  tables::BiomassSuccessionPlugin.BSTables=BiomassSuccessionPlugin.BSTables(),   # Biomass Succession data tables (see bs_tables_from_cfg); shared by every plugin's generators
  pnet_climate_cache::String="",                       # pnet: PRISM L4 monthly cache (JLD2)
  pnet_co2_csv::String="",                             # pnet: annual CO2 csv (year,co2_ppm)
  pnet_generic::AbstractDict=Dict{String,Any}(),       # pnet: PnETConst field overrides
  pnet_soil_texture::AbstractDict=Dict{String,Any}(),  # pnet: {L3 => Saxton-Rawls texture code}; default LOAM
  pnet_soil_csv::String="",                            # pnet: per-plot SOLUS soil (tools/build_plot_soil.jl); "" = texture path (legacy)
  hbad_priors_csv::String="",                          # hbad: fixed priors (results/hbad/priors.csv); "" = repo default
  hbad_coefs_csv::String="",                           # hbad: translation coefficients (results/hbad/translation_coefs.csv); "" = repo default
  hbad_seed_rain::Bool=false,                          # hbad: every species of the ecoregion is a seed source (ESTAB_SEED_RAIN)
  hbad_prob_estab_csv::String="",                      # hbad: per-cycle (species, L3) PROB_ESTAB table (prob_estab_all_species.csv); "" = constant fallback
  hbad_cycle_years::Float64=8.0,                       # hbad: remeasurement interval for the per-cycle -> annual PROB_ESTAB conversion
  hbad_init::String="jitter",                          # hbad: start candidates default | jitter | prior (see HBADPlugin.INIT_MODE)
  cmaes_blocks::String="eco",                          # CMA-ES covariance blocks: "eco" (per cell, legacy) | "l3" (per L3 + tail)
  spinup::Bool=true,
  search_mode::String="lbsa",
  context_only::Bool=false,   # prepare the training context exactly as a search would, hand it to `post_search_hook` (if any), skip the search
  post_search_hook=nothing,   # optional diagnostic callback `f(ctx)`: called with the prepared training context right after the search driver returns (tools, test/pnet_smoke_diag.jl); nothing ⇒ no call
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
  filter_states::Vector{Int}=Int[],        # region = state ∩ filter_ecos (FIPS, e.g. EPA Region 4); empty ⇒ no state filter
  siteclass_l3s::Vector{String}=String[],  # site-class cells only in these ecos; others → one cell "X" (empty ⇒ all)
  max_exact::Int=12, max_group::Int=4,     # species tiering caps (own-param species / pooled groups)
  drop_phantom::Bool=false,                # drop intro_type=phantom_backfill rows (synthetic pre-debut biomass) at load
  train_cap::Union{Nothing,Int,String}=nothing,  # species-balanced TRAIN: cap plots per (eco cell × dominant species); Int | "auto"; surplus → val/test
  train_cap_k::Float64=1.5, train_cap_min::Int=20, # "auto" cap = max(train_cap_min, ceil(k × median stratum size per eco cell))
  split_only::Bool=false,                  # write split_plots.csv + species_map.csv (+ pool_map.csv) and stop (floor-building pass)
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
  data_opts::Data.DataOptions=Data.DataOptions(),   # data-preparation inputs (disturbance-shock parquet); see `data_options`
  scoring::ScoringOptions=ScoringOptions(),   # the scoring rule + initial-biomass perturbation (see `scoring_from_config`)
  simb::SimBOptions=SimBOptions(),            # Sim B / dual-mode options
  space_opts::PU.ParamSpaceOptions=PU.ParamSpaceOptions(),   # which parameters are searched and their boxes (see `param_space_options`)
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
  igel_warm_start::Bool=false,      # igelmo: initial population = the start_from params + (μ−1) Gaussian perturbations (σ=igel_sigma0, u-space)
  igel_freeze_seed_growth::Bool=false,   # igelmo: each seed lineage keeps its own frozen {D,S,B_MAX,ANPP_MAX} (needs fix_growth + seeds)
  cc_group_size::Int=2,
  cc_spec_gens::Int=20,
  cc_integ_gens::Int=20,
  cc_cycles::Int=5,
  cc_fix_others::Bool=false,
  gdigel::AbstractDict=Dict{String,Any}(),   # search_mode gdigel: group-decomposed (1+1)-CMA options (docs/cmaes_gdigel.md)
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
  exec::ExecOptions=ExecOptions(),   # how the run executes (parallel mode, timing, validation switches); see ExecOptions
  search::SearchOptions=SearchOptions(),   # config-derived knobs of the search engines (LBSA, Igel, MO-CMA-ES ranking); see SearchOptions
  rng::Random.AbstractRNG)
  _check_search_mode(search_mode)   # fail fast, before data preparation

  # Pin BLAS to 1 thread: the search does hundreds of tiny per-offspring covariance eigendecomps per gen —
  # multi-threaded OpenBLAS spawns all cores per call and oversubscribes catastrophically at high --threads.
  LinearAlgebra.BLAS.set_num_threads(1)
  @info "search starting" julia_threads=Threads.nthreads() blas_threads=LinearAlgebra.BLAS.get_num_threads()
  mkpath(output_dir)
  # [5, 10, 20, 40, 60, 80]
  #bins_idx = vcat(5:5:30, 40:10:80, 100:20:160)
  @info bins_idx
  @info smoothing_window
  scoring.init_perturb_frac > 0 && @info "Initial-condition perturbation: ±$(round(100*scoring.init_perturb_frac;digits=2))% over $(n_reps) reps (best-of), shared seed; perturbed = max(2, bio + clamp(bio·pct, ±$(scoring.init_perturb_cap))) [max deviation $(scoring.init_perturb_cap), floor 2]"
  # unbinned_w (Sim A): per-year age grid (1:400) + no age-smoothing → W₁ is the EXACT per-year EMD.
  # Setting the bins at the SOURCE means the softplus β and the cell-norm W scales
  # (compute_loss_scales) both recompute on this same grid — W₁, its normalization, and the softplus knee
  # stay consistent (no per-year-W₁ ÷ stale-binned-scale mismatch). bins_idx then only governs Sim B (tier-4).
  # The ONE place the loss options are read: every loss knob of the run lives in this immutable LossParams (and the per-reference,
  # data-derived normalisation in LossScales, see compute_loss_scales). No module-level loss state.
  _loss_params(age_bins, smoothing_weights) = PU.LossParams(; age_bins=age_bins, smoothing_weights=smoothing_weights, lambda=FloatType(loss_lambda),
    loss_alpha=FloatType(loss_alpha),                         # 0 → optimize L2/AGB-level only (zero Wasserstein)
    agb_hinge=agb_hinge,                                      # AGB term: hinge-L1 (tolerance band) vs sqrt-difference
    agb_hinge_threshold=FloatType(agb_hinge_threshold),
    agb_hinge_pct=FloatType(agb_hinge_pct),                   # >0 ⇒ band = clamp(obs*pct, min, max)
    agb_hinge_pct_min=FloatType(agb_hinge_pct_min), agb_hinge_pct_max=FloatType(agb_hinge_pct_max),
    agb_hinge_p=FloatType(agb_hinge_p),                       # hinge exponent pen(h)=h^p (default 2 = L2)
    agb_hinge_beta=FloatType(agb_hinge_beta),                 # softplus sharpness (constant; 0 ⇒ hard hinge)
    agb_normalize=agb_normalize,                              # AGB term = relative error (÷ observed total)
    w_normalize=w_normalize,                                  # W1 term = ÷ per-reference max → [0,1] ratio
    cell_norm=cell_normalize,                                 # per-(eco×lu×sp) ÷scale + AGB-rank rescale (overrides global)
    w_scale_factor=FloatType(w_scale_factor),                 # scale the W divisor (rebalance ΣW vs ΣAGB)
    w_softplus=w_smooth,                                      # softplus-smooth the per-term W1
    w_smooth_band=FloatType(w_smooth_band), w_smooth_conc=FloatType(w_smooth_conc),
    w_smooth_beta=FloatType(w_smooth_beta), w_smooth_auto_knee=w_smooth_auto_knee,
    w_p=FloatType(w_p),                                       # square the W term (aggression)
    w_hinge=w_hinge,                                          # Sim A: ±band "benefit of the doubt" forgiveness before the age CDF
    w_hinge_pct=FloatType(w_hinge_pct), w_hinge_pct_min=FloatType(w_hinge_pct_min), w_hinge_pct_max=FloatType(w_hinge_pct_max),
    loss_piecewise=loss_piecewise, w_pivot=FloatType(w_pivot), agb_pivot=FloatType(agb_pivot),
    rankw_mode=Symbol(lowercase(rankw_mode)), rankw_beta=rankw_beta, rankw_pct_floor=rankw_pct_floor,
    cbal_on=w_count_balance, cbal_mode=Symbol(lowercase(w_count_balance_mode)), cbal_beta=w_count_beta, cbal_pct_floor=w_count_pct_floor)
  loss_params = unbinned_w ?
    _loss_params(PU.AgeBins(bins_idx=collect(1:400), last_bin_open=true), FloatType[one(FloatType)]) :
    _loss_params(PU.AgeBins(bins_idx=bins_idx .|> Int, last_bin_open=true), smoothing_window)
  unbinned_w && @info "Sim-A W = UNBINNED per-year EMD (bins 1:400, no age-smoothing); softplus β + cell W_SCALE recompute on this grid"
  splots, eco_list, species_list, eco_species_ids, splots_val_raw, splots_test_raw, data_outs =
    Data.prepare_parametrization_data(; data_opts, _prep_kwargs(; cohorts_db_path, eco_field, tablename, filter_eco_field, filter_ecos,
      filter_plots, exclude_plots, filter_species, filter_planted, skip_disturbances, spinup, by_subplot,
      val_frac, split_seed, n_folds, fold_index, test_frac, min_trees, min_agb_frac, stratify_eco_mixed,
      single_ecoregion, stratify_landuse, site_class_strata, siteclass_hi_max, siteclass_scheme, shade_tier_csv,
      filter_extent, filter_states, siteclass_l3s, max_exact, max_group, train_cap, train_cap_k, train_cap_min, rng, drop_phantom)...)
  _write_split_plots(output_dir, eco_list, splots, splots_val_raw, splots_test_raw, data_outs)
  n_species = length(species_list)
  # --- succession plugin selection (static type; everything downstream dispatches on succession_plugin_of(soa)) ---
  SUCC = succession_plugin_type(succession_plugin)
  if SUCC === PnETPlugin.PnET
    spinup && error("succession_plugin=pnet requires spinup: false (cohorts start from the observed state)")
    simb.dual_mode === :off || error("succession_plugin=pnet does not support dual_mode yet (Sim B needs establishment)")
  end
  if SUCC === HBADPlugin.HBAD
    spinup && error("succession_plugin=hbad requires spinup: false (cohorts start from the observed state)")
    # Establishment (natural recruitment, docs/hbad_plugin.md): ON when no_establishment is false (Sim B / free runs); Sim A keeps no_establishment: true.
    no_establishment || @info "succession_plugin=hbad: natural establishment is ON (PROB_ESTAB x SUFFICIENT_LIGHT(shade class from RD), seed rain=$(hbad_seed_rain))"
  end
  # The plugin of the run: an immutable context (BS tables | PnET climate + constants | HBAD priors + options) and the site-attribute builder, passed
  # explicitly to the drivers, `make_sites` and `fit_params` - no module-level plugin state.
  plugin = plugin_config(succession_plugin, species_list, eco_species_ids, eco_list; tables=tables, cohorts_db_path=cohorts_db_path, tablename=tablename,
    pnet_climate_cache=pnet_climate_cache, pnet_co2_csv=pnet_co2_csv, pnet_generic=pnet_generic, pnet_soil_texture=pnet_soil_texture, pnet_soil_csv=pnet_soil_csv,
    hbad_priors_csv=hbad_priors_csv, hbad_coefs_csv=hbad_coefs_csv, hbad_init=hbad_init, hbad_seed_rain=hbad_seed_rain,
    hbad_prob_estab_csv=hbad_prob_estab_csv, hbad_cycle_years=hbad_cycle_years)
  n_species = length(species_list)
  n_ecoregions = length(eco_list)
  n_plots = maximum(splots.plot_id)
  # count-balance reweight: build per-agebin W1 weights ONCE from the TRAIN reference (frozen), on the COARSE
  # age_idx bins; Sim A (per-year) looks them up via CBAL_COARSE. Applied in calculate_species_loss_acc!.
  # RANKW mode: :rank (default, 1/√ln(rank) by AGB) or :cbal_pct (percentile-floored class-balanced by cohort
  # count — species play the role of age-bins in build_cbal_tables; 10% floor caps rare/empty cells).
  # The run-wide TRAIN tables the reference scales are built from: per-cell cohort counts (rankw_mode = :cbal_pct) and the count-balance
  # weights. They travel as `scale_seed` (a LossScales holding only these) to every search driver, which derives the scales of each reference.
  _nc = zeros(Int, 0, 0)
  if loss_params.rankw_mode === :cbal_pct
    _nc = _cell_ncoh(splots, n_species, length(eco_species_ids))
    @info "RANKW mode = cbal_pct (percentile-floored class-balanced, β=$rankw_beta)"
  end
  _cbal = PU.NO_CBAL
  if w_count_balance
    _cbal = PU.build_cbal_tables(splots, PU.AgeBins(bins_idx=bins_idx .|> Int, last_bin_open=true), loss_params.age_bins, eco_species_ids, n_species; beta=w_count_beta, pct_floor=w_count_pct_floor)
    @info "Count-balance reweight ON (mode=$(loss_params.cbal_mode) β=$(w_count_beta) pct_floor=$(w_count_pct_floor)); coarse=$(length(bins_idx)) bins+open, Sim-A W bins=$(length(loss_params.age_bins.bin_widths))"
  end
  # Which species drive DOMINANCE (MO Pareto / CMA-MAE measure / breadth axis / RANKW). Default exact (SPCD only). Travels in the scale seed.
  _dom = dominance_mask(species_list, dominance_species)
  scale_seed = PU.LossScales(cbal=_cbal, ncoh=_nc, dom=_dom)
  # What shares a u-dimension (split sets, tier merges, pooling); writes pool_map.csv. Part of the run's ParamSpace (see param_space.jl).
  tying = build_param_tying(param_split_species, param_tier_merge, param_pooling, splots, eco_list, species_list, eco_species_ids, output_dir)
  if split_only
    @info "split_only: wrote split_plots.csv / species_map.csv / pool_map.csv to $(output_dir) — stopping before search"
    return nothing
  end
  println("Plots:$n_plots, Ecos:$n_ecoregions, Species:$n_species, Measurements: $(size(splots))")
  dominance_species != "all" && @info "Dominance species ($dominance_species): $(count(_dom))/$n_species → $([species_list[s] for s in findall(_dom)])"
  # The parameter space of the run: distributions + slot layout + pooling + covariance blocks (data floors are added below, after the sobol branch).
  space = make_param_space(plugin.ctx, species_list, eco_list, eco_species_ids; no_establishment=no_establishment, fit_establishment=(simb.dual_mode != :off),
    opts=space_opts, tying=tying, blocks=cma_blocks(cmaes_blocks, eco_list))
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
    vref = build_reference(splots_val_raw, eco_species_ids, loss_params; scoring=scoring, simb=simb, no_establishment=no_establishment, plugin=plugin, rng=rng, spinup=spinup)
    val_spdf_plts = vref.spdf_plts
    val_site_sim_years = vref.site_sim_years
    val_spinup_cohorts = vref.spinup_cohorts   # needed for the val Sim-B spinup (dual val)
    val_injection_cohorts = vref.injection_cohorts
    val_ref_soa = vref.ref_soa
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
  println("Preprocessing plot results (smoothing and binning)")
  debug = (get(ENV, "JULIA_DEBUG", "") ∈ ["Pan", "all"])
  ref = build_reference(splots, eco_species_ids, loss_params; scoring=scoring, simb=simb, no_establishment=no_establishment, plugin=plugin, rng=rng, spinup=spinup, debug=debug)
  @assert minimum(ref.spdf.sim_year) == 0 "$(ref.spdf.sim_year)"
  spdf_plts = ref.spdf_plts; site_sim_years = ref.site_sim_years; spinup_cohorts = ref.spinup_cohorts
  injection_cohorts = ref.injection_cohorts; ref_soa = ref.ref_soa
  total_err = test_spdf(spdf_plts, n_species, eco_species_ids, loss_params)
  println(total_err)
  println(PU.get_total_loss(total_err, loss_params))
  println(sum(length(xs) for xs in eco_species_ids))
  println("beginning trials")
  if search_mode == "sobol"
    return parametrize_sobol(; ref_soa=ref_soa, space=space, plugin=plugin,
      output_dir=output_dir,
      splots=splots,
      spdf_plts=spdf_plts,
      spinup_cohorts=spinup_cohorts,
      site_sim_years=site_sim_years,
      species_list=species_list,
      eco_list=eco_list,
      eco_species_ids=eco_species_ids,
      loss_params=loss_params,
      scoring=scoring,
      simb=simb,
      scale_seed=scale_seed,
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
  # per-(eco,species) data floors of B_MAX / ANPP / PROB_ESTAB (each nothing if off); the sobol branch above searches the unfloored space
  space = PU.with_floors(space, build_param_floors(bs_tables(plugin), space_opts, eco_list, species_list, eco_species_ids))
  if context_only   # plot_only / plot_only_sobol and context tools: the training context, no search
    ctx = (; space, plugin, ref_soa, splots, spdf_plts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, scale_seed, injection_cohorts, spinup_cohorts, output_dir, n_species, tier, spinup, no_establishment)
    post_search_hook === nothing || post_search_hook(ctx)
    return ctx
  end
  driver = search_mode == "sa" ? parametrize_SA :
           search_mode == "molbsa" ? parametrize_MOLBSA :
           search_mode == "cmaes" ? parametrize_CMAES :
           search_mode == "mocmaes" ? parametrize_MOCMAES :
           search_mode == "igelmo" ? parametrize_IgelMOCMAES :
           search_mode == "cmame" ? parametrize_CMAMAE :
           search_mode == "ccigel" ? parametrize_CCIgel :
           search_mode == "nsga2" ? parametrize_NSGA2 :
           search_mode == "gdigel" ? parametrize_GDIgel : parametrize_LBSA   # "lbsa"
  # CMA-ES-only knobs; LBSA/MOLBSA don't accept these, so only splat them for the (MO)CMA-ES drivers.
  cmaes_kw = search_mode == "molbsa" ? (archive_cap=cmaes_archive_cap, seed_archive_from=seed_archive_from) :
             search_mode == "cmaes" ? (cmaes_lambda=cmaes_lambda, cmaes_sigma0=cmaes_sigma0, ipop=ipop, ipop_stagnation=ipop_stagnation, ipop_max_barren_restarts=ipop_max_barren_restarts, integer_handling=cmaes_integer_handling, integer_std_factor=cmaes_integer_std_factor, single_cov=cmaes_single_cov) :
             search_mode == "mocmaes" ? (cmaes_lambda=cmaes_lambda, cmaes_sigma0=cmaes_sigma0, cmaes_warmstart_seeds=cmaes_warmstart_seeds, ipop=ipop, ipop_stagnation=ipop_stagnation, ipop_max_barren_restarts=ipop_max_barren_restarts, archive_cap=cmaes_archive_cap, integer_handling=cmaes_integer_handling, integer_std_factor=cmaes_integer_std_factor, single_cov=cmaes_single_cov, seed_archive_from=seed_archive_from) :
             search_mode == "igelmo" ? (archive_cap=cmaes_archive_cap, igel_mu=igel_mu, igel_sigma0=igel_sigma0, igel_sobol_init=igel_sobol_init, igel_sobol_pool_k=igel_sobol_pool_k, igel_sobol_raw_mult=igel_sobol_raw_mult, igel_niche_radius=igel_niche_radius, igel_reseed_sigma=igel_reseed_sigma, igel_reseed_random_frac=igel_reseed_random_frac, igel_maturity=igel_maturity, igel_seed_maturity=igel_seed_maturity, igel_warm_start=igel_warm_start, igel_freeze_seed_growth=igel_freeze_seed_growth, single_cov=cmaes_single_cov) :
             search_mode == "ccigel" ? (archive_cap=cmaes_archive_cap, igel_sigma0=igel_sigma0, igel_sobol_init=igel_sobol_init, igel_niche_radius=igel_niche_radius, igel_reseed_sigma=igel_reseed_sigma, igel_maturity=igel_maturity, single_cov=cmaes_single_cov, cc_group_size=cc_group_size, cc_spec_gens=cc_spec_gens, cc_integ_gens=cc_integ_gens, cc_cycles=cc_cycles, cc_fix_others=cc_fix_others) :
             search_mode == "gdigel" ? (gd_opts=gdigel,) :
             search_mode == "nsga2" ? (archive_cap=cmaes_archive_cap, nsga2_pop=nsga2_pop, nsga2_offspring=nsga2_offspring, nsga2_eta_c=nsga2_eta_c, nsga2_eta_m=nsga2_eta_m, nsga2_pc=nsga2_pc, nsga2_pm=nsga2_pm, sobol_init=nsga2_sobol_init) :
             search_mode == "cmame" ? (cmaes_lambda=cmaes_lambda, cmaes_sigma0=cmaes_sigma0, cmame_alpha=cmame_alpha, cmame_grid=cmame_grid, cmame_reseed_explore=cmame_reseed_explore, cmame_restart_patience=cmame_restart_patience, cmame_sobol_reseed=cmame_sobol_reseed, balanced_quality=balanced_quality, cmame_mo_rank=cmame_mo_rank, archive_by_sp=archive_by_sp, bounds_by_sobol=bounds_by_sobol, top_seeds=top_seeds, integer_handling=cmaes_integer_handling, integer_std_factor=cmaes_integer_std_factor, single_cov=cmaes_single_cov) :
             NamedTuple()
  _driver_result = driver(; ref_soa=ref_soa, space=space, plugin=plugin,
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
      scoring=scoring,
      simb=simb,
      scale_seed=scale_seed,
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
      exec=exec,
      search=search,
      cmaes_kw...)
  post_search_hook === nothing || post_search_hook((; space, plugin, ref_soa, splots, spdf_plts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, scale_seed, injection_cohorts, spinup_cohorts, output_dir, n_species, tier, spinup, no_establishment))
  return _driver_result
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
# Diagnostics: fit_params(...; capture_t4=T4Capture()) → every calculate_t4_loss of that call stashes `(sim, ref, n_cycles, site_counts)` into `.data`
# (the last one wins: Sim B's when a dual_b follows Sim A). tools/simB_regen_diff.jl.
mutable struct T4Capture
  data::Union{Nothing,NamedTuple}
end
T4Capture() = T4Capture(nothing)
function calculate_t4_loss(t4_sim, t4_ref, loss_params::PU.LossParams, scales::PU.LossScales, n_species, eco_species_ids, t4_site_counts, t4_obs_counts, n_cycles; cell_plots=nothing, per_plot_mean::Bool, capture::Union{Nothing,T4Capture}=nothing)
  capture === nothing || (capture.data = (sim=deepcopy(t4_sim), ref=t4_ref, n_cycles=n_cycles, site_counts=deepcopy(t4_site_counts)))
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
      # Uses the SAME cell plot counts the AGB scale divides by, so numerator/denominator stay consistent.
      pcinv = (per_plot_mean && cell_plots !== nothing) ?
        one(FloatType) / FloatType(max(1, cell_plots[eco_id][cyc])) : one(FloatType)
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
        usew = loss_params.cbal_on && loss_params.cbal_mode !== :a && !isempty(scales.cbal.w)
        cbw = usew ? (@inbounds scales.cbal.w[gsp, eco_id]) : FloatType[]
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
          a += loss_params.agb_hinge ?
            PU._hinge_relu(loss_params, abs(sm - rm) - PU._agb_hinge_thresh(loss_params, rm)) :   # linear excess; power after norm
            (sqrt(sm) - sqrt(rm))^2
        end
        sp_w_loss[gsp] += PU._w_finish(loss_params, scales, s, Int(gsp), Int(eco_id))  # raw (softplus) W1 when cell_norm; else (W1/scale)^p
        sp_agb_loss[gsp] += loss_params.lambda * PU._agb_finish(loss_params, scales, a)   # raw hinge excess when cell_norm; else (excess/scale)^p
      end
    end
    # num_obs = # cells (not measurements): tier-4 loss is averaged PER CELL so it doesn't scale with #plots.
    eco_losses[eco_id] = PU.SiteLoss(sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=zero(FloatType),
      num_sites=max(1, sum(t4_site_counts[eco_id])), num_obs=max(1, ncell))
  end
  return eco_losses
end

# ── Loss scales: the normalisation derived from ONE reference set (pure; no module state). ────────────────────────────────────────────────
#
# `compute_loss_scales` is called once per reference (train Sim A, train Sim B, val Sim A, val Sim B, …) and the result is passed explicitly to
# `fit_params` and `_mo_objectives`:
#   * global ratio-of-sums (agb_normalize / w_normalize): Σ observed AGB, Σ per-reference-max W1 → `w_global`/`agb_global`;
#   * per-cell (cell_norm): `w_cell`/`agb_cell`[gsp,eco] (Sim A reads the per-plot spdf records, Sim B the aggregated t4_ref), × w_scale_factor on W;
#   * the softplus β (fixed, or per-cell auto-knee computed ONCE from the first reference and shared by the others);
#   * RANKW: computed from the TRAIN reference only (`rankw=nothing` marks the train reference) and shared (same matrix) by every other reference,
#     val included, so train and val weight the same eco×lu×sp cells the same way. The scales themselves are intensive ratio-of-sums, so they are
#     recomputed per reference.
# `cbal`/`ncoh` are the run-wide TRAIN tables (count-balance weights, per-cell cohort counts), carried inside the result.

"Per-(species, eco) TRAIN cohort-row counts: the input of the `rankw_mode = :cbal_pct` weights."
function _cell_ncoh(splots, n_species::Int, n_eco::Int)::Matrix{Int}
  nc = zeros(Int, n_species, n_eco)
  for r in eachrow(splots)
    (1 <= Int(r.species_id) <= n_species && 1 <= Int(r.eco_id) <= n_eco) && (nc[Int(r.species_id), Int(r.eco_id)] += 1)
  end
  return nc
end

function compute_loss_scales(lp::PU.LossParams, search_tier::Int, spdf_plts, t4_ref, eco_species_ids, n_species::Int;
    cell_plots=nothing, per_plot_mean::Bool, cbal::PU.CbalTables=PU.NO_CBAL, ncoh::Matrix{Int}=zeros(Int, 0, 0), dom::Vector{Bool}=Bool[],
    rankw::Union{Nothing,Matrix{FloatType}}=nothing, softplus_cell_from::Union{Nothing,PU.LossScales}=nothing)::PU.LossScales
  bw = lp.age_bins.bin_widths
  ne = length(eco_species_ids)
  # --- softplus β ---
  sp_beta = zero(FloatType); sp_beta_cell = zeros(FloatType, 0, 0)
  if lp.w_softplus
    if !lp.w_smooth_auto_knee
      # Knee adjustment OFF (default): fixed β for W, independent of the grid's theoretical max — the
      # band·conc·wmax derivation over-inflates the knee on fine grids (unbinned Σbw≈399) and mutes real W.
      sp_beta = lp.w_smooth_beta
    else
      # Knee adjustment ON — PER-CELL, bounded by each cell's REFERENCE AGE STRETCH. The theoretical W1 max
      # for a (species,eco) cell is Σ bw only up to K* = the cell's OLDEST reference-cohort bin; past K* the
      # reference CDF is saturated (F_ref=1), so |ΔF| can't realistically be 1 there — summing the whole grid
      # (Σbw−bw[end]=399 unbinned) counts a long empty tail and over-inflates the knee. Reweighted by the
      # count-balance weights when that is on. All frozen across trials → computed once (by the first reference; later references reuse it).
      band = lp.w_smooth_band; conc = lp.w_smooth_conc
      nb = length(bw); ng = n_species
      use_cbal = lp.cbal_on && !isempty(cbal.w) && !isempty(cbal.coarse)
      coarse = use_cbal ? cbal.coarse : Int[]
      CW = use_cbal ? cbal.w : Matrix{Vector{FloatType}}(undef, 0, 0)
      wmax_full = sum(bw) - bw[end]
      sp_beta = wmax_full > 0 ? FloatType(1.0 / (band * conc * wmax_full)) : zero(FloatType)  # scalar fallback
      if softplus_cell_from !== nothing && size(softplus_cell_from.softplus_beta_cell) == (ng, ne)
        sp_beta_cell = softplus_cell_from.softplus_beta_cell
      else
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
        βcell = fill(sp_beta, ng, ne)
        for eco in 1:ne, gsp in 1:ng
          K = Kstar[gsp, eco]; K <= 1 && continue                # no reference / single bin → keep scalar
          w = use_cbal ? (@inbounds CW[gsp, eco]) : FloatType[]
          rew = use_cbal && !isempty(w)
          m = 0.0
          @inbounds for k in 1:K-1; m += (rew ? Float64(w[coarse[k]]) : 1.0) * Float64(bw[k]); end  # Σ bw over the ref stretch (drop saturating bin K*)
          m > 0 && (βcell[gsp, eco] = FloatType(1.0 / (band * conc * m)))
        end
        sp_beta_cell = βcell
      end
    end
  end
  # --- per-cell scales (+ RANKW) ---
  if lp.cell_norm
    W, A, R = _cell_scales(lp, search_tier, spdf_plts, t4_ref, eco_species_ids, n_species, cbal, ncoh, cell_plots, rankw, per_plot_mean, dom)
    return PU.LossScales(w_cell=W, agb_cell=A, rankw=R, softplus_beta=sp_beta, softplus_beta_cell=sp_beta_cell, cbal=cbal, ncoh=ncoh, dom=dom)
  end
  rk = rankw === nothing ? zeros(FloatType, 0, 0) : rankw
  (lp.w_normalize || lp.agb_normalize) || return PU.LossScales(rankw=rk, softplus_beta=sp_beta, softplus_beta_cell=sp_beta_cell, cbal=cbal, ncoh=ncoh, dom=dom)
  # --- global ratio-of-sums ---
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
  return PU.LossScales(w_global=wsum > 0 ? FloatType(wsum) : one(FloatType), agb_global=asum > 0 ? FloatType(asum) : one(FloatType),
    rankw=rk, softplus_beta=sp_beta, softplus_beta_cell=sp_beta_cell, cbal=cbal, ncoh=ncoh, dom=dom)
end

# Per-cell (eco×lu×sp) scales for cell_norm, computed from the reference (constant across evals). Each
# cell (eco e, global species gsp) gets W scale = Σ per-ref-max W1 and AGB scale = Σ observed AGB. Sim A
# (tier≠4) reads the per-plot spdf records; Sim B (tier 4) reads the aggregated t4_ref. RANKW[gsp,e] =
# (1/log(rank+1))/Σ over dominance-included cells, rank by observed AGB descending. `rankw === nothing` ⇒ this is the TRAIN reference
# and the rank is computed from it (Sim A: per-stratum plot counts as split size; Sim B-only: equal strata); otherwise the given train rank is used.
function _cell_scales(lp::PU.LossParams, search_tier::Int, spdf_plts, t4_ref, eco_species_ids, n_species::Int, cbal::PU.CbalTables,
    ncoh::Matrix{Int}, cell_plots, rankw::Union{Nothing,Matrix{FloatType}}, per_plot_mean::Bool, dom::Vector{Bool})
  bw = lp.age_bins.bin_widths
  ne = length(eco_species_ids)
  W = zeros(FloatType, n_species, ne); A = zeros(FloatType, n_species, ne)
  if search_tier == 4 && t4_ref !== nothing
    useB = lp.cbal_on && lp.cbal_mode !== :a && !isempty(cbal.w)   # reweight the ref-scale to match calculate_t4_loss (Sim B): direct cbw[k]
    cp = (per_plot_mean && cell_plots !== nothing) ? cell_plots : nothing
    for (e, eco) in enumerate(t4_ref)
      sp_map = eco_species_ids[e]
      for (cyc, cycmat) in enumerate(eco), sp in 1:size(cycmat, 1)
        gsp = sp_map[sp]
        # AGB scale gets the SAME per-plot-mean division as the loss (W scale is CDF-based → left as-is).
        pcinv = cp === nothing ? one(FloatType) : one(FloatType) / FloatType(max(1, cp[e][cyc]))
        row = @view cycmat[sp, :]; tot = sum(row); A[gsp, e] += tot * pcinv
        tot > 0 || continue
        cbw = useB ? (@inbounds cbal.w[gsp, e]) : FloatType[]
        rew = !isempty(cbw) && length(cbw) == length(bw)
        acc = zero(FloatType)
        @inbounds for k in eachindex(bw); acc += row[k]; cdf = acc / tot; W[gsp, e] += (rew ? cbw[k] : one(FloatType)) * bw[k] * max(cdf, one(FloatType) - cdf); end
      end
    end
    lp.w_scale_factor != one(FloatType) && (W .*= lp.w_scale_factor)
    R = rankw === nothing ? _rankw(lp, A, eco_species_ids; ncoh=ncoh, dom=dom) : rankw   # B-only: rank from B
  else
    useA = lp.cbal_on && lp.cbal_mode !== :b && !isempty(cbal.w)   # reweight the ref-scale to match _w1_sum (Sim A): coarse[k] lookup
    coarse = cbal.coarse
    for ((_, eco_id), yd) in spdf_plts, (_, gt) in yd, (sp_eco, rec) in gt.records
      gsp = eco_species_ids[eco_id][sp_eco]
      A[gsp, eco_id] += rec.sp_agb_sum
      cbw = useA ? (@inbounds cbal.w[gsp, eco_id]) : FloatType[]
      rew = !isempty(cbw)
      @inbounds for k in eachindex(bw); W[gsp, eco_id] += (rew ? cbw[coarse[k]] : one(FloatType)) * bw[k] * max(rec.sp_age_cdf[k], one(FloatType) - rec.sp_age_cdf[k]); end
    end
    lp.w_scale_factor != one(FloatType) && (W .*= lp.w_scale_factor)
    if rankw === nothing
      nplots = zeros(Int, ne)                                  # plots per stratum (train) → split-size weight
      for ((_, eco_id), _) in spdf_plts; nplots[eco_id] += 1; end
      R = _rankw(lp, A, eco_species_ids; split_size=nplots, ncoh=ncoh, dom=dom)   # rank from train A's per-plot ref
    else
      R = rankw
    end
  end
  return W, A, R
end

# Rank dominance-included species WITHIN EACH STRATUM (eco_id) by observed AGB (desc), weight
# 1/√(log1p(rank)) (√ softens the decay so rarer species aren't crushed); normalise each stratum to 1,
# then weight strata by the ORDER OF MAGNITUDE of their split size — RANKW[:,e] ∝ log10(1+n_plots[e]).
# This sits between the global ranking (largest split owns everything, small splits starved) and equal
# per-stratum (small noisy splits over-weighted): a 939-plot split outweighs a 34-plot one by ~log10,
# not 28× and not 1×. split_size = nothing → equal strata (fallback for B-only). Total Σ RANKW = 1.
# rankw_mode = :cbal_pct (with the per-cell train cohort counts `ncoh`) replaces the AGB rank by percentile-floored class-balanced weights.
function _rankw(lp::PU.LossParams, agb::Matrix{FloatType}, eco_species_ids; split_size::Union{Nothing,AbstractVector}=nothing,
    ncoh::Matrix{Int}=zeros(Int, 0, 0), dom::Vector{Bool}=Bool[])::Matrix{FloatType}
  R = zeros(FloatType, size(agb)); sw = zeros(Float64, length(eco_species_ids))
  cbal = lp.rankw_mode === :cbal_pct && !isempty(ncoh)        # percentile-floored class-balanced (by cohort count)
  for e in eachindex(eco_species_ids)
    ids = Int[]; for gsp in eco_species_ids[e]; _dom_included(dom, gsp) && push!(ids, gsp); end
    isempty(ids) && continue
    if cbal
      β = lp.rankw_beta; nc = ncoh; tot = sum(nc[gsp, e] for gsp in ids); totw = 0.0
      # a stratum without training cohorts carries no loss; its class-balanced weight would be (1-β)/(1-β^0) = Inf → Inf/Inf = NaN,
      # which poisons the aggregate (NaN·0). Weight 0 instead (B7).
      if tot == 0
        sw[e] = split_size === nothing ? 1.0 : log10(1.0 + Float64(split_size[e]))
        continue
      end
      qf = round(Int, 1 / lp.rankw_pct_floor)                              # grid steps (10/20/50 for 0.10/0.05/0.02); mirrors build_cbal_tables
      for gsp in ids
        pct = tot > 0 ? max(round(qf * nc[gsp, e] / tot) / qf, 1 / qf) : 1 / qf  # count-share on a 1/qf grid, floored at 1/qf (=rankw_pct_floor); qf=10 ≡ old round(10·share)/10 ⌊0.10⌋
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
  return R
end

# Scales of the references of ONE split (train, or val) of a run: the primary reference's (`search_tier`, `spdf_plts`, `t4_ref`) and, when
# `dual_b` is given, Sim B's (returned inside a copy of the dual_b NamedTuple as `.scales`, which `fit_params` and `_mo_objectives` read).
# `train` is the TRAIN LossScales for a val split (its rankw and softplus cells are reused, so train and val weight cells the same way);
# `train === nothing` marks the train split itself, whose first reference defines RANKW. `seed` carries the run-wide `cbal`/`ncoh` tables.
function build_split_scales(lp::PU.LossParams, seed::PU.LossScales, search_tier::Int, spdf_plts, t4_ref, eco_species_ids, n_species::Int;
    dual_b=nothing, cell_plots=nothing, train::Union{Nothing,PU.LossScales}=nothing, simb::SimBOptions)
  cell_plots === nothing && dual_b !== nothing && (cell_plots = get(dual_b, :cell_plots, nothing))   # the split's own tier-4 counts (see fit_params)
  main = compute_loss_scales(lp, search_tier, spdf_plts, t4_ref, eco_species_ids, n_species; cell_plots=cell_plots, per_plot_mean=simb.t4_per_plot_mean, cbal=seed.cbal, ncoh=seed.ncoh, dom=seed.dom,
    rankw = train === nothing ? nothing : train.rankw, softplus_cell_from = train)
  if train !== nothing && dual_b === nothing && simb.t4_primary && !(search_tier == 4 && t4_ref !== nothing)
    # Tier-4-only run (t4_primary, no Sim B) whose val reference is evaluated as Sim A (tier 3): its per-cell objective has always been
    # normalised by the TRAIN tier-4 tables (the val Sim A reference never produced tier-4 tables of its own). Kept as is: the loss
    # kernels use the val reference's own fields, `_mo_objectives` reads the train cell scales.
    main = PU.LossScales(w_global=main.w_global, agb_global=main.agb_global, w_cell=train.w_cell, agb_cell=train.agb_cell, rankw=main.rankw,
      softplus_beta=main.softplus_beta, softplus_beta_cell=main.softplus_beta_cell, cbal=main.cbal, ncoh=main.ncoh, dom=main.dom)
  end
  dual_b === nothing && return main, nothing
  bs = compute_loss_scales(dual_b.loss_params, dual_b.tier, dual_b.spdf_plts, dual_b.t4_ref, eco_species_ids, n_species; cell_plots=dual_b.cell_plots, per_plot_mean=simb.t4_per_plot_mean,
    cbal=seed.cbal, ncoh=seed.ncoh, dom=seed.dom, rankw=main.rankw, softplus_cell_from=(train === nothing ? main : train))
  return main, merge(dual_b, (scales=bs,))
end

# The two scale sets `_mo_objectives` indexes by group: group 0 = the primary reference — Sim A, except when Sim B is primary (SimBOptions.t4_primary, carried by dual_b: a
# dual_mode b run, whose only block is Sim B, or a tier-4-only run whose single reference already carries the tier-4 scales); group 1 = Sim B.
@inline function _group_scales(scales::PU.LossScales, dual_b)
  # without a Sim B (tier 5 stacks a tier-4 half after the tier-3 half) group 1 has NO scales of its own: the empty LossScales makes _mo_objectives
  # refuse (B11) instead of normalising tier-4 losses with another reference's scales
  dual_b === nothing && return scales, PU.LossScales()
  bsc = dual_b.scales::PU.LossScales
  return (dual_b.t4_primary ? bsc : scales), bsc
end

# A plan carries data derived under a given shock order / disturbance mode / recruit rule: it must be used with the same scoring rule.
function _check_plan_options(plan::InjectionPlan, scoring::ScoringOptions)
  p = plan.opts
  (shock_order(p) == shock_order(scoring) && p.disturbance === scoring.disturbance && p.exclude_recruits_at_entry == scoring.exclude_recruits_at_entry) ||
    error("InjectionPlan was built under a different scoring rule (shock order $(shock_order(p)), disturbance $(p.disturbance), recruits $(p.exclude_recruits_at_entry)) " *
          "than the one it is evaluated with ($(shock_order(scoring)), $(scoring.disturbance), $(scoring.exclude_recruits_at_entry)); rebuild it with build_injection_plan")
  nothing
end

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

"""
    record_best_losses!(losses_db, iter, total, run_result, eco_losses, params; eco_list, species_list, n_species, loss_params, print_breakdown=false)

The `losses.duckdb` rows of a new best of a single-objective driver (LBSA, CMA-ES, SA): `total_loss` (with the serialized parameters), one
`ecoregion_loss` row per ecoregion and one `species_loss` row per (ecoregion, species) with a nonzero age-distribution loss. `print_breakdown`
also prints the raw per-ecoregion / per-species breakdown, sorted by contribution, to see where the remaining error concentrates.
"""
function record_best_losses!(losses_db, iter, total, run_result, eco_losses, params; eco_list, species_list, n_species, loss_params, print_breakdown::Bool=false)
  let buf = IOBuffer()
    Serialization.serialize(buf, params)
    DuckDB.execute(losses_db, "INSERT INTO total_loss VALUES (?, ?, ?, ?, ?)", [iter, run_result.num_sites, run_result.num_obs, total, take!(buf)])
  end
  isnothing(eco_losses) && return nothing
  for (eco_id, eco_loss) in enumerate(eco_losses)
    eco_name = _eco_name(eco_list, eco_id)
    eco_total = convert(Float64, PU.get_total_loss(eco_loss, loss_params))
    DuckDB.execute(losses_db, "INSERT INTO ecoregion_loss VALUES (?, ?, ?, ?, ?)", [iter, eco_name, eco_loss.num_sites, eco_loss.num_obs, eco_total])
    n = max(1, eco_loss.num_sites)
    for gsp in 1:n_species
      eco_loss.sp_w_loss[gsp] == 0f0 && continue
      DuckDB.execute(losses_db, "INSERT INTO species_loss VALUES (?, ?, ?, ?, ?)", [iter, eco_name, species_list[gsp], eco_loss.sp_w_loss[gsp] / n, eco_loss.sp_agb_loss[gsp] / n])
    end
  end
  if print_breakdown
    println("Raw SiteLoss breakdown @ $iter (total=$(round(total, sigdigits=6))):")
    for eco_id in sort(collect(eachindex(eco_losses)); by=e -> -convert(Float64, PU.get_total_loss(eco_losses[e], loss_params)))
      el = eco_losses[eco_id]
      println("  [$(_eco_name(eco_list, eco_id))] eco_total=$(round(convert(Float64, PU.get_total_loss(el, loss_params)), sigdigits=5))  sites=$(el.num_sites) obs=$(el.num_obs)")
      for gsp in sort([g for g in 1:n_species if el.sp_w_loss[g] != 0f0]; by=g -> -el.sp_w_loss[g])
        println("      $(rpad(species_list[gsp], 10)) w=$(round(el.sp_w_loss[gsp], sigdigits=4))  agb=$(round(el.sp_agb_loss[gsp], sigdigits=4))")
      end
    end
  end
  return nothing
end

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
            test_df = simulate_and_test(; splots=splots, bio_params=search_state.best.x, eco_list=eco_list, species_list=species_list, eco_species_ids=eco_species_ids, loss_params=loss_params, site_sim_years=site_sim_years, M=n_reps, no_establishment=no_establishment, rng=rng, plugin=plugin)
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
                val_test_df = simulate_and_test(; splots=val_splots, bio_params=search_state.best.x, eco_list=eco_list, species_list=species_list, eco_species_ids=eco_species_ids, loss_params=loss_params, site_sim_years=val_site_sim_years, M=n_reps, no_establishment=no_establishment, rng=rng, plugin=plugin)
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
            test_df = simulate_and_test(; splots=splots, bio_params=search_state.best.x, eco_list=eco_list, species_list=species_list, eco_species_ids=eco_species_ids, loss_params=loss_params, site_sim_years=site_sim_years, M=n_reps, no_establishment=no_establishment, rng=rng, plugin=plugin)
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
                val_test_df = simulate_and_test(; splots=val_splots, bio_params=search_state.best.x, eco_list=eco_list, species_list=species_list, eco_species_ids=eco_species_ids, loss_params=loss_params, site_sim_years=val_site_sim_years, M=n_reps, no_establishment=no_establishment, rng=rng, plugin=plugin)
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
    val_ref_soa, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params, scoring, simb,
    inj_val, val_dual_b, val_scales, fixed_seeds, val_splots, val_injection_cohorts, plugin)
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
  excl = (scoring.disturbance == :exclude_overwrite && !isnothing(val_injection_cohorts) && hasproperty(val_injection_cohorts, :disturbance_drop_pct)) ?
    Set((Int(r.plot_id), Int(r.sim_year)) for r in eachrow(val_injection_cohorts) if r.disturbance_drop_pct > 0) : Set{Tuple{Int,Int}}()
  mrows = NamedTuple[]; prows = NamedTuple[]
  for (kind, c) in reps
    rr = val_fit_params(c.x; val_ref_soa, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params,
      injection=inj_val, dual_b=val_dual_b, seeds=fixed_seeds, scales=val_scales, scoring, simb, plugin, collect_cohorts=true)
    run_result, eco_losses, idx = _agg_reps(rr, loss_params, scoring)
    objs = _mo_objectives(eco_losses, eco_species_ids, loss_params, val_scales, val_dual_b)
    push!(mrows, (rep_kind=kind, W=Float64(sum(objs[1:2:end])), AGB=Float64(sum(objs[2:2:end])), total=Float64(_mo_aggregate(objs, run_result, loss_params)), front_size=length(arch)))
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

# Post-hoc reselection RANKED ON VALIDATION. Evaluate EVERY archive candidate (deduped by train-objective id
# across checkpoints) on the held-out val set → val (A_W,A_AGB) with frozen-train normalization. Rank each
# checkpoint's archive by its VALIDATION front area (shared-rectangle rule: min area, tie max knee, tie max
# count) → p100 = best-on-val archive; last = final checkpoint. Designate 5 positions {extreme_w, extreme_agb,
# knee, "median", best_aggregate} on each archive's VAL front. Writes cv_reselect_metrics.csv (10 rows, val + train objectives).
function _cv_reselect_dump(output_dir, eco_list, eco_species_ids, n_species, max_sim_year,
    val_ref_soa, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params, scoring, simb,
    inj_val, val_dual_b, val_scales, fixed_seeds, val_splots, species_list, plugin)
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
    rr = val_fit_params(x; val_ref_soa, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params,
      injection=inj_val, dual_b=val_dual_b, seeds=fixed_seeds, scales=val_scales, scoring, simb, plugin)
    _, eco_losses, _ = _agg_reps(rr, loss_params, scoring); objs = _mo_objectives(eco_losses, eco_species_ids, loss_params, val_scales, val_dual_b)
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
      rr = val_fit_params(c.x; val_ref_soa, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params,
      injection=inj_val, dual_b=val_dual_b, seeds=fixed_seeds, scales=val_scales, scoring, simb, plugin, collect_cohorts=true)
      av = _agg_reps(rr, loss_params, scoring); cached = rr[av[3]][2]
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
  val_scales::Any                   # LossScales of the val reference (RANKW = the train one); nothing when !have_val
  val_dual_b::Any                   # val Sim B (carries its own .scales) or nothing
  val_fit::Any                      # x -> (run_result, cached_sites, eco_losses) on the val split (nothing when !have_val)
  plugin::PluginConfig              # the run's plugin (diagnostic simulate_and_test)
  exec::ExecOptions                 # execution options of the run (store_val_obj, newbest_stats, parallel_mode)
  timer::Union{Nothing,RunTimer}    # io-time accounting of the writer hand-offs (nothing ⇒ timing off)
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
    n_output_plots, sampled_ids, sampled_ids_val, emp_sample, emp_sample_val, val_fit, plugin::PluginConfig, exec::ExecOptions=ExecOptions(), timer::Union{Nothing,RunTimer}=nothing, val_scales=nothing, val_dual_b=nothing)
  metrics_io = open(joinpath(output_dir, "metrics.csv"), resuming ? "a" : "w")
  resuming || println(metrics_io, "iteration,best_train_loss,best_val_loss,pop_size,archive_size")
  gio = MOGenIO(output_dir, writer_ch, losses_db, metrics_io, splots, eco_list, species_list, eco_species_ids,
    n_species, site_sim_years, loss_params, no_establishment, n_reps, rng, have_val, n_output_plots,
    sampled_ids, sampled_ids_val, emp_sample, emp_sample_val, val_scales, val_dual_b, val_fit, plugin, exec, timer, _mo_archive_sig(state),
    resuming ? _load_val_cache(output_dir, losses_db) : Dict{Any,NTuple{4,Float64}}(), Inf, Inf)   # RESUME: keep the pre-resume val cache (CSV, or losses.duckdb mirror if the CSV is gone)
  if have_val && val_fit !== nothing
    try
      _r, _c, _el = val_fit(state.representative.x)
      gio.rep_val_loss = _mo_val_loss((_r, nothing, _el), eco_species_ids, loss_params, val_scales, val_dual_b)
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
  if gio.have_val && gio.exec.store_val_obj && (archive_changed || is_new_best)   # val-score any NEW archive members
    _newc = [c for c in collect(state.archive) if !haskey(gio.val_cache, _rkey(c))]
    _vscore = c -> try
        _r, _c, _el = gio.val_fit(c.x)
        _o = _mo_objectives(_el, gio.eco_species_ids, gio.loss_params, gio.val_scales, gio.val_dual_b)
        (_rkey(c), (_twt(c), _tat(c), Float64(sum(@view _o[1:2:end])), Float64(sum(@view _o[2:2:end]))))
      catch e; @warn "val score failed" exception=(e, catch_backtrace()); nothing; end
    if candidate_parallel(gio.exec)      # score new members concurrently (each val_fit is sites-serial); store after
      _vout = Vector{Any}(undef, length(_newc))
      if !isempty(_newc)
        _vout[1] = _vscore(_newc[1])      # the first member serially, the rest concurrently (the val scales are explicit, nothing to race on)
        Threads.@threads :static for i in 2:length(_newc); _vout[i] = _vscore(_newc[i]); end
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
    if gio.exec.newbest_stats   # opt-in diagnostic only (PAN_NEWBEST_STATS=1) — otherwise skip this whole extra train sim
      try
        test_df = simulate_and_test(; splots=gio.splots, bio_params=state.representative.x, eco_list=gio.eco_list,
          species_list=gio.species_list, eco_species_ids=gio.eco_species_ids, loss_params=gio.loss_params,
          site_sim_years=gio.site_sim_years, M=gio.n_reps, no_establishment=gio.no_establishment, rng=gio.rng, plugin=gio.plugin)
        println("Train stats:"); show(test_df; allrows=true, allcols=true); println()
      catch e; @warn "simulate_and_test (train) failed" exception=(e, catch_backtrace()); end
    end
    if gio.have_val && gio.val_fit !== nothing
      try
        # The representative is an archive member, so ExecOptions.store_val_obj already val-scored it into val_cache. Under
        # cell-norm the aggregate is Σobjs = val_W + val_AGB, so reuse the cached score and SKIP the redundant sim —
        # unless we need the val plot sample (n_output_plots>0) or can't reconstruct (non-cell-norm): then re-sim.
        _rk = _rkey(state.representative)
        if gio.loss_params.cell_norm && gio.n_output_plots == 0 && haskey(gio.val_cache, _rk)
          _cv = gio.val_cache[_rk]; gio.rep_val_loss = _cv[3] + _cv[4]
          @info "Val loss @ gen $iter | loss=$(gio.rep_val_loss) (cached)"
        else
          _r, _c, _el = gio.val_fit(state.representative.x)
          gio.rep_val_loss = _mo_val_loss((_r, nothing, _el), gio.eco_species_ids, gio.loss_params, gio.val_scales, gio.val_dual_b)
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
      eco=[(name=_eco_name(gio.eco_list, eco_id), ns=el.num_sites, no=el.num_obs, tot=convert(Float64, PU.get_total_loss(el, gio.loss_params)),
            sp=[(s=gio.species_list[gsp], w=el.sp_w_loss[gsp]/max(1, el.num_sites), a=el.sp_agb_loss[gsp]/max(1, el.num_sites))
                for gsp in 1:gio.n_species if el.sp_w_loss[gsp] != 0f0])
           for (eco_id, el) in enumerate(gb_eco)])
  end
  # OFFLOAD to the async writer: metrics line (every gen) + BATCHED losses insert (one transaction, new-best only).
  # No fsync on the search's critical path; drained/flushed on normal exit AND interrupt by stop_writer's finally.
  _mline = string(state.i, ",", Float64(state.representative.fx.aggregate), ",", isfinite(gio.rep_val_loss) ? gio.rep_val_loss : "", ",", pop_size, ",", length(state.archive))
  _lossjob = MOLossesJob(gio.losses_db, gio.metrics_io, _mline, _losses_payload, _newvals)
  timed_put!(gio.timer, gio.writer_ch, _lossjob)
  cached_sites_state_df = gb_cached === nothing ?    # nothing when n_output_plots==0 (candidate mode skips the best re-sim); the MO writer never reads this field anyway
    DataFrame(plot_id=Int[], sim_year=Int[], species_id=Int[], age=Int[], agb=Float64[]) :
    _cohorts_df(gb_cached)
  sim_sample = (is_new_best && gio.n_output_plots > 0) ? _filter_cached_to_df(gb_cached, gio.sampled_ids) : nothing
  # The async writer above persists search_state@N + best_params@N whenever save_ckpt is set, and
  # save_ckpt == archive_changed, so EVERY archive-changed state is already saved by the writer. (A former
  # synchronous save for the changed-but-not-improved case was redundant and raced the writer on the same
  # path — removed.)
  _ckptjob = WriterJob(save_ckpt, deepcopy(state), gio.splots, cached_sites_state_df, gio.emp_sample, sim_sample, is_new_best ? gio.emp_sample_val : nothing, val_sim_sample)  # deepcopy (CPU) built before the timed put! so io = channel BLOCK only
  timed_put!(gio.timer, gio.writer_ch, _ckptjob)
  (gio.have_val && gio.exec.store_val_obj && (is_new_best || archive_changed)) && _persist_val_cache!(gio)
  return save_ckpt
end

function parametrize_MOCMAES(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, cmaes_lambda::Union{Nothing,Int}=nothing, cmaes_sigma0::Float64=0.3, cmaes_warmstart_seeds::Int=20, ipop::Bool=false, ipop_stagnation::Int=20, ipop_max_barren_restarts::Int=4, integer_handling::Bool=false, integer_std_factor::Float64=0.3, single_cov::Bool=false, seed_archive_from::Union{Nothing,String}=nothing, scale_seed::PU.LossScales=PU.LossScales(), exec::ExecOptions=ExecOptions(), search::SearchOptions=SearchOptions())
  env = mo_env(; ref_soa, plugin, splots, spdf_plts, spinup_cohorts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, spinup, rng, debug,
    search_tier, n_reps, n_output_plots, no_establishment, injection_cohorts, cycle_years, scale_seed, exec,
    val_splots, val_ref_soa, val_spdf_plts, val_site_sim_years, val_spinup_cohorts, val_injection_cohorts,
    val_dual_early=true)   # this driver built the validation Sim B before the val plot sample (RNG order)
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
  n_species = env.n_species; max_sim_year = env.max_sim_year
  mo_val!(env)

  if exec.cv_reselect   # skip the search; just re-score the 10 reselection positions (5 per archive) on the (frozen-norm) val set
    env.have_val || (@warn "ExecOptions.cv_reselect set but no val split — nothing to do"; return)
    _cv_reselect_dump(output_dir, eco_list, eco_species_ids, n_species, max_sim_year,
      val_ref_soa, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params, scoring, simb,
      env.inj_val, env.val_dual_b, env.val_scales, env.fixed_seeds, val_splots, species_list, plugin)
    return
  end

  _run(p; ws=env.work_soa) = run_candidate(env, p; ws)
  _fitness(rep_results) = candidate_fitness(env, rep_results)

  sobol_idx = Ref(1)
  next_candidate() =
    if sobol_idx[] <= length(_sobol_cands)
      p = _sobol_cands[sobol_idx[]]
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
        error("start_from params are incompatible with this run: their SPECIES_LIST/ECO_LIST differ from the loaded data.")
      p
    else
      next_candidate()
    end
    _seed_prob_estab!(bs_tables(plugin), bio_params, eco_list, species_list, eco_species_ids)   # calibration seed (in-place; stays in search)
    _seed_maturity!(bs_tables(plugin), bio_params, species_list)
    _seed_min_rel!(bio_params, eco_list, space.opts)   # pin MIN_REL out of search (fix_min_rel)
    _floor_bmax_seed!(bio_params, space)          # raise any seed B_MAX below its per-(eco,species) data floor
    frozen_ref = deepcopy(bio_params)   # frozen-growth reference for IPOP restarts (fix_growth must survive restart)
    slots = space.slots
    fx0, _, _ = _fitness(_run(bio_params))
    mean0 = PU.params_to_u(bio_params, space, slots)
    rep = MOLBSA.MOCandidate(bio_params, fx0)
    groups = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(space, slots))   # block-diagonal MO-CMA-ES
    @info "MO-CMA-ES block-diagonal: $(length(groups)) covariance blocks (sizes $(length.(groups)))"
    search_state = MOCMAES.MOCMAESState(mean0, cmaes_sigma0, rep, rng; lambda=cmaes_lambda, blocks=groups, max_iter=typemax(Int), archive_cap=archive_cap)
    search_state.n_evals = 1
    # WARM-START the distribution from the top-K Sobol seeds instead of the single best point: mean =
    # log-weighted recombination of the top-K, C₀ = their (shrunk) empirical covariance shape. K=1 ⇒ no-op
    # (keeps the single-point mean + C=I already set above).
    if cmaes_warmstart_seeds > 1 && isnothing(start_from) && length(_sobol_cands) >= 2
      K = min(cmaes_warmstart_seeds, length(_sobol_cands))
      us = [PU.params_to_u(_sobol_cands[i], space, slots) for i in 1:K]
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
    frozen_ref = isnothing(start_from) ? nothing : _load_params_from_path(start_from, plugin.ctx)   # frozen-growth reference
    slots = space.slots
    if force_restart_from_random
      bio_params = next_candidate()
      _apply_frozen_growth!(bio_params, frozen_ref, space.opts)
      _seed_prob_estab!(bs_tables(plugin), bio_params, eco_list, species_list, eco_species_ids); _seed_maturity!(bs_tables(plugin), bio_params, species_list); _seed_min_rel!(bio_params, eco_list, space.opts)
      _floor_bmax_seed!(bio_params, space)
      MOCMAES.restart!(search_state, PU.params_to_u(bio_params, space, slots), cmaes_sigma0)
    end
  end
  # Hansen-style mixed-integer handling (honors the flag each run, incl. on resume).
  search_state.u_min_std = integer_handling ? PU.integer_u_min_std(space, slots, integer_std_factor) : Float64[]
  if TRIALS < 1 || search_state.n_evals >= TRIALS
    return search_state
  end

  if env.have_val && get(ENV, "VAL_REPLAY", "") == "1"
    ckpts = filter(f -> occursin(r"^best_params@\d+\.jld2$", f), readdir(output_dir))
    sort!(ckpts, by = f -> parse(Int, match(r"@(\d+)", f).captures[1]))
    @info "VAL_REPLAY: recomputing val for $(length(ckpts)) checkpoints in $output_dir"
    open(joinpath(output_dir, "val_replay.csv"), "w") do io
      println(io, "iteration,val_loss")
      for f in ckpts
        it = parse(Int, match(r"@(\d+)", f).captures[1])
        try
          p = JLD2.load_object(joinpath(output_dir, f))
          vl = _mo_val_loss(val_eval(env, p), eco_species_ids, loss_params, env.val_scales, env.val_dual_b)
          println(io, "$it,$vl"); flush(io); @info "val-replay @ $it = $vl"
        catch e
          @warn "val-replay failed @ $it" exception = (e, catch_backtrace())
        end
      end
    end
    @info "VAL_REPLAY done → $(joinpath(output_dir, "val_replay.csv"))"
    return nothing
  end

  stagnation = 0
  barren_restarts = 0             # consecutive IPOP restarts with no new best (barren-restart stop)
  improved_since_restart = false  # any new best found in the current IPOP epoch?
  loop = mo_loop_open(env, search_state; output_dir, resuming=!isnothing(resume_from))
  gio = loop.gio
  evals_done = search_state.n_evals
  est_gens = max(1, cld(TRIALS - evals_done, search_state.lambda))
  run_search_loop(loop; get_state=() -> search_state) do
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
          cand = PU.u_to_params(xs_u[k], space, slots, bio_params)
          rep_results = _run(cand)
          fx_k, run_k, eco_k = _fitness(rep_results)
          cands[k] = cand
          fxs[k] = fx_k
          evals_done += 1
          if k == 1 || isless(fx_k.aggregate, gen_best_agg)   # k==1 ⇒ gb_* always defined; isless ⇒ NaN ranks worst
            gen_best_agg = fx_k.aggregate
            gb_idx = k
            gb_run = run_k
            gb_eco = eco_k
            gb_cached = _median_rep_cached(rep_results, loss_params, scoring)
          end
        end

        MOCMAES.tell!(search_state, fxs, xs_u; nds=search.mocmaes_nds)
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
          _apply_frozen_growth!(bio_params, frozen_ref, space.opts)   # keep fix_growth frozen at the Sim-A seed across restarts
          _seed_prob_estab!(bs_tables(plugin), bio_params, eco_list, species_list, eco_species_ids)   # re-seed the calibration params
          _seed_maturity!(bs_tables(plugin), bio_params, species_list)
          _seed_min_rel!(bio_params, eco_list, space.opts)   # keep MIN_REL pinned across IPOP restarts
          MOCMAES.restart!(search_state, PU.params_to_u(bio_params, space, slots), cmaes_sigma0; lambda=new_lambda)
          stagnation = 0
        end
      end
  end

  env.have_val && try   # CV: held-out front metrics under THIS fold's live (train-frozen) loss — see _dump_cv_front
    _dump_cv_front(search_state, output_dir, eco_list, eco_species_ids, n_species, max_sim_year,
      val_ref_soa, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params, scoring, simb,
      env.inj_val, env.val_dual_b, env.val_scales, env.fixed_seeds, val_splots, val_injection_cohorts, plugin)
  catch e; @error "cv front dump failed" exception = (e, catch_backtrace()); end

  return search_state
end

# Igel/Hansen/Roth (2007) population-based MO-CMA-ES driver. Same MO objective vector, archive and
# writer as parametrize_MOCMAES, but the engine is a population of μ (1+1)-CMA-ES individuals
# (IgelMOCMAES) rather than one distribution — better front spread/extreme coverage. μ candidates
# are evaluated per generation (serially); the initial population is a Sobol design.
function parametrize_IgelMOCMAES(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, igel_mu::Int=20, igel_sigma0::Float64=0.3, igel_sobol_init::Bool=true, igel_sobol_pool_k::Int=1, igel_sobol_raw_mult::Int=2, igel_niche_radius::Float64=0.0, igel_reseed_sigma::Float64=0.0, igel_reseed_random_frac::Float64=0.0, igel_maturity::Int=0, igel_seed_maturity::Bool=false, igel_warm_start::Bool=false, igel_freeze_seed_growth::Bool=false, single_cov::Bool=false, scale_seed::PU.LossScales=PU.LossScales(), exec::ExecOptions=ExecOptions(), search::SearchOptions=SearchOptions())
  env = mo_env(; ref_soa, plugin, splots, spdf_plts, spinup_cohorts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, spinup, rng, debug,
    search_tier, n_reps, n_output_plots, no_establishment, injection_cohorts, cycle_years, scale_seed, exec,
    val_splots, val_ref_soa, val_spdf_plts, val_site_sim_years, val_spinup_cohorts, val_injection_cohorts,
    train_sites_parallel=!candidate_parallel(exec))   # candidate mode: sites go serial for EVERY eval of this driver (init population, gen loop, val)
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  n_species = env.n_species; _work_soa_t = env.work_soa_t; _timer = env.timer
  _run(p; ws=env.work_soa) = run_candidate(env, p; ws)
  _fitness(rep_results) = candidate_fitness(env, rep_results)
  if isnothing(resume_from)
    template = !isnothing(start_from) ? _load_params_from_path(start_from, plugin.ctx) : generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    slots = space.slots
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
      elseif igel_warm_start && !isnothing(start_from)
        # warm start: the start_from params themselves + (μ−1) Gaussian perturbations around them in u-space
        u0 = PU.params_to_u(template, space, slots)
        @info "igelmo warm start from $(basename(start_from)): μ=$igel_mu around it (σ=$igel_sigma0)"
        vcat([template], [PU.u_to_params(clamp.(u0 .+ igel_sigma0 .* randn(rng, length(u0)), 0.0, 1.0), space, slots, template) for _ in 2:igel_mu])
      elseif igel_sobol_init
        inline_cut = do_pool
        PU.sobol_samples(space, template, do_pool ? elite_n * igel_sobol_raw_mult : igel_mu)
      else
        [generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment) for _ in 1:igel_mu]
      end
    # Seed the ESTABLISHMENT params from the data tables (prob_estab_from_data / maturity_from_data) and pin
    # MIN_REL (fix_min_rel) — mirroring parametrize_MOCMAES. Applied to the template AND every pool member BEFORE
    # eval/params_to_u, so the encoded u and the frozen `bases` carry the seeded/pinned values. Growth
    # ({D,S,B_MAX,ANPP_MAX}) is untouched. Each fn is a no-op unless its flag/table is set.
    for _p in Iterators.flatten(((template,), pool_params))
      _seed_prob_estab!(bs_tables(plugin), _p, eco_list, species_list, eco_species_ids)
      _seed_maturity!(bs_tables(plugin), _p, species_list)
      _seed_min_rel!(_p, eco_list, space.opts)
    end
    # evaluate under THIS run's exact loss config: the raw pool if inline_cut, else the already-selected μ.
    np = length(pool_params); pool_fx = Vector{Any}(undef, np)
    if candidate_parallel(exec)     # candidate-parallelize the eval (sites already serial here)
      Threads.@threads :static for k in 1:np
        pool_fx[k] = _fitness(_run(pool_params[k]; ws=_work_soa_t[Threads.threadid()]))[1]
      end
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
    init_us = [PU.params_to_u(p, space, slots) for p in init_params]
    init_cands = [MOLBSA.MOCandidate(init_params[k], pool_fx[sel[k]]) for k in eachindex(sel)]
    groups = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(space, slots))   # block-diagonal per-individual (1+1)-CMA
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
      space.opts.fix_growth || @warn "igel_freeze_seed_growth is on but fix_growth is OFF → growth is in the search slots and will NOT stay frozen"
      # _sobol_cands is already drained into init_params by here, so check the seed SOURCE instead.
      (isnothing(sobol_candidates_db) && isnothing(start_from)) && @warn "igel_freeze_seed_growth is on but no seed source (sobol_candidates_db/start_from) → freezing RANDOM growth"
      @info "igelmo per-lineage frozen growth: $igel_mu lineages each keep their seed's {D,S,B_MAX,ANPP_MAX}"
    end
  else
    @info "Resuming from $resume_from"
    # (RANKW is the TRAIN reference's, carried explicitly by `scales`/`val_scales` from the start — no throwaway train eval needed on resume.)
    _resumed = IgelMOCMAES.load_state(IgelMOCMAES.IgelSearch, resume_from; dir=output_dir, bases=igel_freeze_seed_growth)   # state (+ per-lineage bases sidecar from output_dir)
    search_state = IgelMOCMAES.state(_resumed); bases = _resumed.bases
    bio_params = search_state.representative.x
    slots = space.slots
    bases === nothing || @info "igelmo per-lineage frozen growth: reloaded $(length(bases)) lineage bases from igel_bases@$(search_state.i).jld2"
  end

  # Engine options of this run (explicit, not state fields → they apply on fresh AND resume): the config-derived ones of `search.igel` + this
  # driver's reseed fraction + parameter-space niching in native units / semantic quanta when igel_niche_radius>0 (the typed maps close over
  # the decode template: bio_params is the u_to_params template on both fresh & resume — only slot dims are read ⇒ immaterial, frozen-growth-safe).
  igel_opts = let base = IgelMOCMAES.with_reseed_random_frac(search.igel, igel_reseed_random_frac), bp = bio_params
    igel_niche_radius > 0 ?
      IgelMOCMAES.with_niching(base, PU.slot_quanta(space, slots),
        u -> PU.params_to_nativevec(PU.u_to_params(u, space, slots, bp), space, slots),
        p -> PU.params_to_nativevec(p, space, slots)) : base
  end
  isearch = IgelMOCMAES.IgelSearch(search_state, igel_opts; bases=bases)   # the resumable search: engine state + run options + per-lineage bases
  igel_niche_radius > 0 && @info "igelmo parameter-space niching ON: radius=$(igel_niche_radius) native-quanta (L∞) over $(length(slots)) slots"

  if TRIALS < 1 || search_state.n_evals >= TRIALS
    return search_state
  end

  # Held-out val must measure the SAME objective as train: Sim B (dual_b) when dual_mode≠off, else Sim A.
  mo_val!(env)
  loop = mo_loop_open(env, search_state; output_dir, resuming=!isnothing(resume_from))
  gio = loop.gio

  evals_done = search_state.n_evals
  est_gens = max(1, cld(TRIALS - evals_done, igel_mu))
  done_gens = isnothing(resume_from) ? 0 : search_state.i    # gens already completed (search_state.i == checkpoint/metrics counter)
  total_gens = done_gens + est_gens                          # size the bar to the WHOLE run so resume continues the counter/bar
  # resume-safe reseed/σ trace → a SEPARATE csv (NOT metrics.csv, which appends under a fixed header): a stop→resume
  # just appends; header is written only when the file is new/empty. Columns: gen, mean σ, min σ, #below reseed floor, #reseeds.
  _reseed_path = joinpath(output_dir, "reseed_trace.csv"); _reseed_new = !isfile(_reseed_path) || filesize(_reseed_path) == 0
  _reseed_io = open(_reseed_path, _reseed_new ? "w" : "a")
  _reseed_new && (println(_reseed_io, "iteration,mean_sigma,min_sigma,n_below_floor,n_reseed"); flush(_reseed_io))
  run_search_loop(loop; get_state=() -> search_state, save_exit=() -> (IgelMOCMAES.save_state(isearch, output_dir); @info "Search state saved @ $(search_state.i)"), close_extra=() -> close(_reseed_io)) do
      _first_gen = true
      TProgress.@track for _gen in 1:total_gens                # _gen = ABSOLUTE generation number
        _gen <= done_gens && continue                          # RESUME: fast-skip completed gens (bar jumps to the resumed position, then continues)
        evals_done >= TRIALS && break
        if _timer !== nothing
          _first_gen && (println("[timing] startup→first gen: $(_tsec(time_ns()-exec.t0))s"); _first_gen = false)
          reset!(_timer)
        end
        _tg = time_ns()
        offs = IgelMOCMAES.ask(isearch)
        _t_ask = time_ns() - _tg
        off_fxs = Vector{MOLBSA.MOFitness}(undef, igel_mu)
        off_params = Vector{typeof(bio_params)}(undef, igel_mu)
        off_run = Vector{Any}(undef, igel_mu); off_eco = Vector{Any}(undef, igel_mu)   # retained per-candidate loss/eco (cheap) → gen-best needs no serial re-sim
        gen_best_agg = Inf; local gb_run, gb_eco, gb_cached, gb_idx
        _te = time_ns()
        if candidate_parallel(exec)
          # candidate×rep is the parallel unit: each candidate's sim runs single-threaded on a thread-local SoA
          # (NUMA-local, no shared-array contention). Sites are serial here (`_sites_parallel` is false in candidate mode).
          Threads.@threads :static for k in 1:igel_mu
            p = PU.u_to_params(offs[k], space, slots, isearch.bases === nothing ? bio_params : isearch.bases[k])
            off_params[k] = p
            fx_k, run_k, eco_k = _fitness(_run(p; ws=_work_soa_t[Threads.threadid()]))
            off_fxs[k] = fx_k; off_run[k] = run_k; off_eco[k] = eco_k   # keep the batch's run/eco (self-contained loss structs; no SoA aliasing)
          end
          evals_done += igel_mu
          for k in 1:igel_mu                              # serial best-pick (gen argmin aggregate)
            (k == 1 || isless(off_fxs[k].aggregate, gen_best_agg)) && (gen_best_agg = off_fxs[k].aggregate; gb_idx = k)
          end
          gb_run = off_run[gb_idx]; gb_eco = off_eco[gb_idx]  # already computed in the parallel batch (identical to a re-sim: fixed seeds) — no serial full-sim
          gb_cached = n_output_plots > 0 ? _median_rep_cached(_run(off_params[gb_idx]; ws=_work_soa_t[1]), loss_params, scoring) : nothing  # cohort cache only feeds output plots; unused otherwise
        else
          for k in 1:igel_mu                              # SoA mode: serial candidates, sites parallel (fit_params threads)
            p = PU.u_to_params(offs[k], space, slots, isearch.bases === nothing ? bio_params : isearch.bases[k])
            rep_results = _run(p); fx_k, run_k, eco_k = _fitness(rep_results)
            off_params[k] = p; off_fxs[k] = fx_k; evals_done += 1
            if k == 1 || isless(fx_k.aggregate, gen_best_agg)   # k==1 ⇒ gb_* always defined (all-NaN gen); NaN ranks worst
              gen_best_agg = fx_k.aggregate; gb_idx = k; gb_run = run_k; gb_eco = eco_k; gb_cached = _median_rep_cached(rep_results, loss_params, scoring)
            end
          end
        end
        _t_eval = time_ns() - _te; _sim = _timer === nothing ? UInt64(0) : _timer.sim
        _tt = time_ns()
        is_new_best = IgelMOCMAES.tell!(isearch, off_fxs, off_params)   # also realigns the per-lineage frozen-growth bases with the survivors
        _t_tell = time_ns() - _tt
        search_state.n_evals = evals_done
        _tf = time_ns()
        _save_ckpt = mo_gen_finalize!(gio, search_state, igel_mu, is_new_best, gb_run, gb_eco, gb_cached)   # shared: ckpt on ANY archive change (train+val)
        # sidecar the per-lineage bases WHENEVER a checkpoint is saved, tagged with the same gen so resume loads
        # the exact pop↔bases alignment (search_state.pop is not mutated between here and the next ask).
        _save_ckpt && try; IgelMOCMAES.save_aux(isearch, output_dir); catch e; @warn "igel_bases sidecar save failed" exception=e; end
        let _sg = Float64[ind.sigma for ind in search_state.pop], _nr = count(search_state._reseed)   # reseed/σ signal
          println(_reseed_io, string(search_state.i, ",", round(search_state.t, sigdigits=6), ",", round(minimum(_sg), sigdigits=6), ",", count(<(igel_reseed_sigma), _sg), ",", _nr))
          (_nr > 0 || search_state.i % 100 == 0) && flush(_reseed_io)
          _nr > 0 && @info "igelmo gen $(search_state.i): $_nr reseed(s) — mean_σ=$(round(search_state.t, sigdigits=3)) min_σ=$(round(minimum(_sg), sigdigits=3))"
        end
        if _timer !== nothing
          println("[timing] gen $_gen: $(_tsec(time_ns()-_tg))s | ask=$(_tsec(_t_ask)) eval=$(_tsec(_t_eval))(sim=$(_tsec(_sim)) loss+oth=$(_tsec(_t_eval-_sim))) tell=$(_tsec(_t_tell)) final=$(_tsec(time_ns()-_tf))(io=$(_tsec(_timer.io)))")
        end
      end
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
function parametrize_CCIgel(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, igel_sigma0::Float64=0.3, igel_sobol_init::Bool=true, igel_niche_radius::Float64=0.0, igel_reseed_sigma::Float64=0.0, igel_maturity::Int=0, single_cov::Bool=false, cc_group_size::Int=2, cc_spec_gens::Int=20, cc_integ_gens::Int=20, cc_cycles::Int=5, cc_fix_others::Bool=false, scale_seed::PU.LossScales=PU.LossScales(), exec::ExecOptions=ExecOptions(), search::SearchOptions=SearchOptions())
  env = mo_env(; ref_soa, plugin, splots, spdf_plts, spinup_cohorts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, spinup, rng, debug,
    search_tier, n_reps, n_output_plots, no_establishment, injection_cohorts, cycle_years, scale_seed, exec,
    val_splots, val_ref_soa, val_spdf_plts, val_site_sim_years, val_spinup_cohorts, val_injection_cohorts)
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  n_species = env.n_species
  _run(p; ws=env.work_soa) = run_candidate(env, p; ws)
  _fitness(rep_results) = candidate_fitness(env, rep_results)
  # per-species objective (specialization scoring): score species g only, aggregate = Σ of its 2 objectives.
  function _fitness_species(rep_results, gsp::Int)
    _, eco_losses, _ = _agg_reps(rep_results, loss_params, scoring)
    objs = _mo_objectives_species(eco_losses, eco_species_ids, loss_params, env.scales, env.dual_b, gsp)
    MOLBSA.MOFitness(objs, Float64(sum(objs)))
  end
  next_param() = !isempty(_sobol_cands) && length(_sobol_cands) >= 1 ? popfirst!(_sobol_cands) :
                 generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)

  K = cc_group_size
  cc_mu = n_species * K                          # whole-individual population size = generation eval count
  CCIgel.register_sampler_types!(PU.SpeciesSampler, PU.EcoSpeciesSampler)
  if isnothing(resume_from)
    template = !isnothing(start_from) ? _load_params_from_path(start_from, plugin.ctx) : generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    slots = space.slots
    # initial population of cc_mu whole individuals: Sobol candidate DB / Sobol design / random draws.
    init_params = !isempty(_sobol_cands) ? [next_param() for _ in 1:cc_mu] :
                  igel_sobol_init ? PU.sobol_samples(space, template, cc_mu) :
                  [generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment) for _ in 1:cc_mu]
    init_us = [PU.params_to_u(p, space, slots) for p in init_params]
    init_cands = [MOLBSA.MOCandidate(init_params[k], _fitness(_run(init_params[k]))[1]) for k in 1:cc_mu]
    groups = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(space, slots))
    species_slots = CCIgel.species_slot_map(space.dists, slots, eco_species_ids, n_species)
    n_specialized = count(!isempty, species_slots)
    @info "CC-Igel: $n_species species × K=$K = $cc_mu individuals; $n_specialized species have specialization slots; integration blocks=$(length(groups))"
    search_state = IgelMOCMAES.IgelState(init_us, init_cands, rng; sigma0=igel_sigma0, archive_cap=archive_cap, max_iter=typemax(Int), niche_radius=igel_niche_radius, reseed_sigma=igel_reseed_sigma, maturity_period=igel_maturity, blocks=groups)
    search_state.n_evals = cc_mu
    bio_params = template
  else
    @info "Resuming from $resume_from"
    search_state = JLD2.load_object(resume_from)
    bio_params = search_state.representative.x
    slots = space.slots
    species_slots = CCIgel.species_slot_map(space.dists, slots, eco_species_ids, n_species)
  end
  if TRIALS < 1 || search_state.n_evals >= TRIALS
    return search_state
  end

  mo_val!(env)
  loop = mo_loop_open(env, search_state; output_dir, resuming=!isnothing(resume_from))
  gio = loop.gio

  evals_done = search_state.n_evals

  # ---- one INTEGRATION generation: verbatim Igel over the whole vector (archive/metrics/ckpt) ----
  function integ_gen!()
    offs = IgelMOCMAES.ask(search_state, search.igel)
    m = search_state.mu
    off_fxs = Vector{MOLBSA.MOFitness}(undef, m)
    off_params = Vector{typeof(bio_params)}(undef, m)
    gen_best_agg = Inf; local gb_run, gb_eco, gb_cached
    for k in 1:m
      p = PU.u_to_params(offs[k], space, slots, bio_params)
      rep_results = _run(p); fx_k, run_k, eco_k = _fitness(rep_results)
      off_params[k] = p; off_fxs[k] = fx_k; evals_done += 1
      if k == 1 || isless(fx_k.aggregate, gen_best_agg)   # k==1 ⇒ gb_* always defined (all-NaN gen); NaN ranks worst
        gen_best_agg = fx_k.aggregate; gb_run = run_k; gb_eco = eco_k; gb_cached = _median_rep_cached(rep_results, loss_params, scoring)
      end
    end
    is_new_best = IgelMOCMAES.tell!(search_state, off_fxs, off_params, search.igel)
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
      grp_cands = [MOLBSA.MOCandidate(PU.u_to_params(cur_us[i], space, slots, bio_params), search_state.pop[i].fx) for i in idxs]
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
      offs = IgelMOCMAES.ask(st, search.igel)
      off_fxs = Vector{MOLBSA.MOFitness}(undef, st.mu)
      off_params = Vector{typeof(bio_params)}(undef, st.mu)
      for k in 1:st.mu
        u = offs[k]
        if bg !== nothing                                # splice species s's dims onto the shared background
          u = copy(bg); @inbounds for d in species_slots[s]; u[d] = offs[k][d]; end
        end
        p = PU.u_to_params(u, space, slots, bio_params)   # decode WHOLE vector
        off_params[k] = p; off_fxs[k] = _fitness_species(_run(p), s); evals_done += 1
      end
      IgelMOCMAES.tell!(st, off_fxs, off_params, search.igel)                      # NSGA-II select K within group g
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
    new_params = [PU.u_to_params(u, space, slots, bio_params) for u in new_us]
    new_cands = [MOLBSA.MOCandidate(new_params[k], _fitness(_run(new_params[k]))[1]) for k in 1:cc_mu]
    evals_done += cc_mu
    groups2 = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(space, slots))
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

  run_search_loop(loop; get_state=() -> search_state) do
      _cyc = 0
      TProgress.@track for _phase in 1:length(schedule)
        evals_done >= TRIALS && break
        if schedule[_phase] == :spec
          _cyc += 1
          cc = build_cc_groups()
          bg = cc_fix_others ? PU.params_to_u(search_state.representative.x, space, slots) : nothing
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
  end
  return search_state
end

function parametrize_NSGA2(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, nsga2_pop::Int=48, nsga2_offspring::Union{Nothing,Int}=nothing, nsga2_eta_c::Float64=20.0, nsga2_eta_m::Float64=20.0, nsga2_pc::Float64=0.9, nsga2_pm::Float64=-1.0, sobol_init::Bool=true, scale_seed::PU.LossScales=PU.LossScales(), exec::ExecOptions=ExecOptions(), search::SearchOptions=SearchOptions())
  env = mo_env(; ref_soa, plugin, splots, spdf_plts, spinup_cohorts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, spinup, rng, debug,
    search_tier, n_reps, n_output_plots, no_establishment, injection_cohorts, cycle_years, scale_seed, exec,
    val_splots, val_ref_soa, val_spdf_plts, val_site_sim_years, val_spinup_cohorts, val_injection_cohorts)
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  n_species = env.n_species
  _run(p; ws=env.work_soa) = run_candidate(env, p; ws)
  _fitness(rep_results) = candidate_fitness(env, rep_results)
  next_param() = !isempty(_sobol_cands) && length(_sobol_cands) >= 1 ? popfirst!(_sobol_cands) :
                 generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)

  λ = isnothing(nsga2_offspring) ? nsga2_pop : nsga2_offspring
  if isnothing(resume_from)
    template = !isnothing(start_from) ? _load_params_from_path(start_from, plugin.ctx) : generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    slots = space.slots
    # initial μ population: Sobol candidate DB if given, else a Sobol space-filling design, else random draws.
    init_params = !isempty(_sobol_cands) ? [next_param() for _ in 1:nsga2_pop] :
                  sobol_init ? PU.sobol_samples(space, template, nsga2_pop) :
                  [generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment) for _ in 1:nsga2_pop]
    init_us = [PU.params_to_u(p, space, slots) for p in init_params]
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
    slots = space.slots
  end
  if TRIALS < 1 || search_state.n_evals >= TRIALS
    return search_state
  end

  mo_val!(env)
  loop = mo_loop_open(env, search_state; output_dir, resuming=!isnothing(resume_from))
  gio = loop.gio

  evals_done = search_state.n_evals
  est_gens = max(1, cld(TRIALS - evals_done, λ))
  run_search_loop(loop; get_state=() -> search_state) do
      TProgress.@track for _gen in 1:est_gens
        evals_done >= TRIALS && break
        offs = NSGA2.ask(search_state)                  # λ offspring u-vectors (tournament + SBX + polynomial mutation)
        off_fxs = Vector{MOLBSA.MOFitness}(undef, λ)
        off_params = Vector{typeof(bio_params)}(undef, λ)
        gen_best_agg = Inf; local gb_run, gb_eco, gb_cached, gb_idx
        for k in 1:λ                                     # SERIAL (fit_params threads internally)
          p = PU.u_to_params(offs[k], space, slots, bio_params)
          rep_results = _run(p); fx_k, run_k, eco_k = _fitness(rep_results)
          off_params[k] = p; off_fxs[k] = fx_k; evals_done += 1
          if k == 1 || isless(fx_k.aggregate, gen_best_agg)   # k==1 ⇒ gb_* always defined (all-NaN gen); NaN ranks worst
            gen_best_agg = fx_k.aggregate; gb_idx = k; gb_run = run_k; gb_eco = eco_k; gb_cached = _median_rep_cached(rep_results, loss_params, scoring)
          end
        end
        is_new_best = NSGA2.tell!(search_state, off_fxs, off_params, offs)   # (μ+λ) non-dom-sort + crowding selection
        search_state.n_evals = evals_done
        mo_gen_finalize!(gio, search_state, search_state.mu, is_new_best, gb_run, gb_eco, gb_cached)   # shared: ckpt on ANY archive change (train+val)
      end
  end
  return search_state
end

# CMA-MAE driver (Fontaine & Nikolaidis 2022): one archive-driven CMA-ES emitter over a 2-D MAP-Elites
# grid. Behaviour descriptor = (Σ Wasserstein loss, Σ AGB loss) — the two halves of the MO objective
# vector; quality = the aggregate loss. The archive keeps one elite parameter set per behaviour cell
# (a quality-diversity set of fits), the representative is the lowest-aggregate elite. Same MO objective
# vector, writer, archive logging and losses.duckdb schema as parametrize_MOCMAES; the emitter re-seeds
# itself on convergence/stagnation (no external IPOP). Grid bounds are auto-scaled from the seed loss.
function parametrize_CMAMAE(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, cmaes_lambda::Union{Nothing,Int}=nothing, cmaes_sigma0::Float64=0.3, cmame_alpha::Float64=0.02, cmame_grid::Int=15, cmame_reseed_explore::Float64=1.0, cmame_restart_patience::Int=6, cmame_sobol_reseed::Bool=false, balanced_quality::Bool=false, cmame_mo_rank::Bool=false, archive_by_sp::Bool=false, bounds_by_sobol::Bool=false, top_seeds::Float64=0.2, integer_handling::Bool=false, integer_std_factor::Float64=0.3, single_cov::Bool=false, scale_seed::PU.LossScales=PU.LossScales(), exec::ExecOptions=ExecOptions(), search::SearchOptions=SearchOptions())
  env = mo_env(; ref_soa, plugin, splots, spdf_plts, spinup_cohorts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, spinup, rng, debug,
    search_tier, n_reps, n_output_plots, no_establishment, injection_cohorts, cycle_years, scale_seed, exec,
    val_splots, val_ref_soa, val_spdf_plts, val_site_sim_years, val_spinup_cohorts, val_injection_cohorts)
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  n_species = env.n_species
  _run(p; ws=env.work_soa) = run_candidate(env, p; ws)
  _fitness(rep_results) = candidate_fitness(env, rep_results)
  sobol_idx = Ref(1)
  next_candidate() =
    if sobol_idx[] <= length(_sobol_cands)
      p = _sobol_cands[sobol_idx[]]; sobol_idx[] += 1; p
    else
      generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
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
  inc_sp = isempty(scale_seed.dom) ? collect(1:n_sp) : findall(scale_seed.dom)   # dominance species (excl _H/_S etc.)
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
      p = _load_params_from_path(start_from, plugin.ctx)
      (p.SPECIES_LIST == species_list && p.ECO_LIST == eco_list) ||
        error("start_from params are incompatible with this run: their SPECIES_LIST/ECO_LIST differ from the loaded data.")
      p
    else
      next_candidate()
    end
    slots = space.slots
    fx0, _, _ = _fitness(_run(bio_params))
    mean0 = PU.params_to_u(bio_params, space, slots)
    rep = MOLBSA.MOCandidate(bio_params, fx0)
    # auto-scale the 2-D measure grid from the seed candidate's loss (worse than any fit we'll keep),
    # so the archive cells bracket the reachable (Wasserstein, AGB) range; quality ceiling t0 above it.
    m0 = CMAMAE.measure(fx0)
    if !_calibrated                                   # no Sobol bounds → fall back to 2× the seed's measure
      gwlo, galo = 0.0, 0.0
      gwhi = max(2.0 * Float64(m0[1]), 1e-9); gahi = max(2.0 * Float64(m0[2]), 1e-9)
    end
    t0 = 1.5 * convert(Float64, fx0.aggregate)
    groups = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(space, slots))   # block-diagonal CMA-MAE emitter
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
    slots = space.slots
  end
  search_state.engine.emitter.u_min_std = integer_handling ? PU.integer_u_min_std(space, slots, integer_std_factor) : Float64[]   # Hansen integer handling on the CMA-MAE emitter
  if TRIALS < 1 || search_state.n_evals >= TRIALS
    return search_state
  end

  # Validation mirrors the TRAINING tier so the held-out loss ranks candidates by the SAME objective (tier 5 ⇒ tier-5 val; tier 4/5 need
  # per-cycle bins built from the validation split) and, in dual modes, the same A⊕B objective.
  mo_val!(env; dual=true, tier=search_tier)
  loop = mo_loop_open(env, search_state; output_dir, resuming=!isnothing(resume_from))
  gio = loop.gio

  evals_done = search_state.n_evals
  est_gens = max(1, cld(TRIALS - evals_done, search_state.engine.emitter.lambda))
  run_search_loop(loop; get_state=() -> search_state, after_exit=() -> begin
    # Re-evaluate every archive elite on the held-out validation set so each carries train+val loss
    # (aligned to the saved archive order). train_loss is the elite's stored aggregate.
    if env.have_val
      try
        open(joinpath(output_dir, "archive_eval.csv"), "w") do io
          println(io, "candidate,train_loss,val_loss")
          for (ci, m) in enumerate(search_state.archive)
            vr = env.val_fit(m.x)
            println(io, "$ci,$(Float64(m.fx.aggregate)),$(_mo_val_loss(vr, eco_species_ids, loss_params, env.val_scales, env.val_dual_b))")
          end
        end
        @info "Wrote archive_eval.csv (train+val loss per archive elite)"
      catch e; @warn "archive validation eval failed" exception = (e, catch_backtrace()); end
    end
  end) do
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
          cand = PU.u_to_params(xs_u[k], space, slots, bio_params)
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
          if k == 1 || isless(fx_k.aggregate, gen_best_agg)   # k==1 ⇒ gb_* always defined (all-NaN gen); NaN ranks worst
            gen_best_agg = fx_k.aggregate; gb_idx = k; gb_run = run_k; gb_eco = eco_k; gb_cached = _median_rep_cached(rep_results, loss_params, scoring)
          end
        end

        is_new_best = archive_by_sp ? CMAMAE.tell_species_mo!(search_state, cands, sp_meas, sp_qual, xs_u) :
                      CMAMAE.tell_mo!(search_state, cands, meas, xs_u; nds=search.mocmaes_nds)
        search_state.n_evals = evals_done
        search_state.current = cands[gb_idx]

        mo_gen_finalize!(gio, search_state, search_state.engine.emitter.lambda, is_new_best, gb_run, gb_eco, gb_cached)   # shared: ckpt on ANY archive change (train+val)
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
# thus Pareto dominance in MO-CMA-ES/MOLBSA/Igel, the CMA-MAE measure descriptor, the per-species breadth axis and the RANKW weights).
# Empty mask = all species. Built once by parametrize() from the `dominance_species` flag (exact = SPCD only · exact+grp = + _GRP groups ·
# all = incl. _H/_S catch-alls) and carried in `LossScales.dom`. The scalar aggregate (get_total_loss) is unaffected — _H/_S still count
# toward overall fit, just not toward dominance.
function dominance_mask(species_list, mode::AbstractString)::Vector{Bool}
  lowercase(mode) == "all" && return Bool[]
  m = Bool[_dom_include_name(sp, lowercase(mode)) for sp in species_list]
  return any(m) ? m : Bool[]            # an empty selection means "all species" (as the former empty Set did)
end
@inline _dom_included(dom::Vector{Bool}, gsp::Int) = isempty(dom) || (gsp <= length(dom) && @inbounds dom[gsp])
# Decide a species name's membership for a dominance_species mode.
_dom_include_name(name::AbstractString, mode::AbstractString) =
  mode == "all" ? true :
  startswith(name, "_GRP") ? (mode == "exact+grp") :
  startswith(name, "_") ? false :          # _H, _S (and other "_" catch-alls) — excluded unless "all"
  true                                      # SPCD exact species

# Safe per-cell table lookup: 0 when the matrix isn't sized for this cell (e.g. an empty A table under a
# B-only dual) → the caller then falls back to the raw (un-normalized) term instead of indexing out of range.
# per-(species, eco) entry of a scale / rank-weight matrix. A matrix that does not cover the cell means the scales were not
# set for this reference (B11: this used to return 0, silently mixing normalised and raw terms or zeroing the objective).
@inline function _cell(M::Matrix{FloatType}, gsp::Int, e::Int)::FloatType
  (size(M, 1) >= gsp && size(M, 2) >= e) || _cell_error(M, gsp, e)
  @inbounds M[gsp, e]
end
@noinline _cell_error(M, gsp, e) = error("loss scale/weight matrix of size $(size(M)) does not cover (species=$gsp, eco=$e): " *
                                         "the cell scales / RANKW were not set for this reference (call _set_loss_scales! first)")   # (wording kept: the golden status.txt of the tier-5 scenario records it)

# The one accumulation of the MO objective vector. `target_gsp == 0` sums every dominance species; `target_gsp > 0` restricts the inner loop to that
# global species (CC-Igel specialization: still summed over ecoregions / dual groups).
function _mo_accumulate(eco_losses::Vector{PU.SiteLoss}, eco_species_ids::Vector{Vector{Int}}, lp::PU.LossParams, scales::PU.LossScales, dual_b, target_gsp::Int)::Vector{FloatType}
  # 4 GROUPED objectives for the dual: [ΣA_W, ΣA_AGB, ΣB_W, ΣB_AGB] — sum each over its eco-blocks and
  # dominance species. eco_losses = [A's n_eco blocks ; B's n_eco blocks] (2·n_eco ⇒ ngroups=2 ⇒ 4 obj);
  # a single sim (n_eco blocks) ⇒ ngroups=1 ⇒ 2 obj [ΣW, ΣAGB]. Keeping the MO vector low-dimensional
  # makes A↔B and W↔AGB first-class objectives (clean 4-D Pareto / net-win) instead of diluting them
  # across ~n_ess per-species entries. Species breadth is handled POST-HOC (the sweep), not here.
  # `scales` are those of the primary reference and `dual_b.scales` Sim B's (see _group_scales); each block is normalised by the scales of the
  # reference it was scored on, and RANKW is the train one carried by both.
  ne = length(eco_species_ids)
  ngroups = max(1, length(eco_losses) ÷ ne)
  objs = zeros(FloatType, 2 * ngroups)
  cell = lp.cell_norm
  sc0, sc1 = _group_scales(scales, dual_b)
  dom = scales.dom                          # dominance species mask (empty = all), a field of the reference scales: allocation-free
  for (idx, el) in enumerate(eco_losses)
    g = (idx - 1) ÷ ne                       # 0-based group: 0 = Sim A (tier-3), 1 = Sim B (tier-4)
    e = (idx - 1) % ne + 1
    if cell
      # Per-cell: divide each (eco,lu,sp) cell by its OWN reference scale, rescale by the AGB-rank weight,
      # then sum into the group. A and B keep separate scales; RANKW is shared. NOTE: the AGB exponent is
      # applied ONCE, per-cohort in the species loss (_agb_pow on each hinged excess) — do NOT
      # re-apply it here or it double-powers for p≠1 (W's power stays here since W is not pre-powered).
      sc = g == 0 ? sc0 : sc1
      Ws = sc.w_cell; As = sc.agb_cell; Rw = sc.rankw
      @inbounds for gsp in eco_species_ids[e]
        (target_gsp == 0 || gsp == target_gsp) || continue
        _dom_included(dom, gsp) || continue
        rw = _cell(Rw, gsp, e); ws = _cell(Ws, gsp, e); as = _cell(As, gsp, e)
        objs[2 * g + 1] += rw * PU._w_pow(lp, ws > 0 ? el.sp_w_loss[gsp] / ws : el.sp_w_loss[gsp])
        objs[2 * g + 2] += rw * (as > 0 ? el.sp_agb_loss[gsp] / as : el.sp_agb_loss[gsp])
      end
    else
      @inbounds for gsp in eco_species_ids[e]
        (target_gsp == 0 || gsp == target_gsp) || continue
        _dom_included(dom, gsp) || continue         # restrict to the selected dominance species
        objs[2 * g + 1] += el.sp_w_loss[gsp]   # ΣW   for this group
        objs[2 * g + 2] += el.sp_agb_loss[gsp] # ΣAGB for this group
      end
    end
  end
  return objs
end
_mo_objectives(eco_losses::Vector{PU.SiteLoss}, eco_species_ids::Vector{Vector{Int}}, lp::PU.LossParams, scales::PU.LossScales, dual_b=nothing)::Vector{FloatType} =
  _mo_accumulate(eco_losses, eco_species_ids, lp, scales, dual_b, 0)
# Per-species variant for CC-Igel specialization: the same accumulation restricted to `target_gsp`.
_mo_objectives_species(eco_losses::Vector{PU.SiteLoss}, eco_species_ids::Vector{Vector{Int}}, lp::PU.LossParams, scales::PU.LossScales, dual_b, target_gsp::Int)::Vector{FloatType} =
  _mo_accumulate(eco_losses, eco_species_ids, lp, scales, dual_b, target_gsp)

# Scalar aggregate paired with the MO objective vector (for the representative / archive eviction). With
# cell_norm the per-cell scales/rank live only in _mo_objectives, so the aggregate = Σ of the objectives;
# otherwise fall back to the established get_total_loss on the (dual-combined) SiteLoss.
@inline _mo_aggregate(objs::Vector{FloatType}, run_result, lp::PU.LossParams)::Float64 =
  lp.cell_norm ? Float64(sum(objs)) : convert(Float64, PU.get_total_loss(run_result, lp))

# Validation aggregate from a fit_params result tuple (run_result, cached, eco_losses), scored with the VAL reference's scales
# (`scales`/`val_dual_b.scales`; RANKW is the train one inside them), so val stays comparable to the train aggregate.
@inline _mo_val_loss(val_result, eco_species_ids, lp::PU.LossParams, scales::PU.LossScales, dual_b=nothing)::Float64 =
  _mo_aggregate(_mo_objectives(val_result[3], eco_species_ids, lp, scales, dual_b), val_result[1], lp)

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
function parametrize_MOLBSA(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, archive_cap::Int=200, seed_archive_from::Union{Nothing,String}=nothing, scale_seed::PU.LossScales=PU.LossScales(), exec::ExecOptions=ExecOptions(), search::SearchOptions=SearchOptions())
  search_tier in (1, 2, 3, 4, 5) || error("parametrize_MOLBSA: unknown search_tier=$search_tier (expected 1, 2, 3, 4, or 5)")
  env = mo_env(; ref_soa, plugin, splots, spdf_plts, spinup_cohorts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, spinup, rng, debug,
    search_tier, n_reps, n_output_plots, no_establishment, injection_cohorts, cycle_years, scale_seed, exec,
    val_splots, val_ref_soa, val_spdf_plts, val_site_sim_years, val_spinup_cohorts, val_injection_cohorts,
    val_dual_early=true)   # this driver built the validation Sim B before the val plot sample (RNG order)
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  n_species = env.n_species
  _run(p; ws=env.work_soa) = run_candidate(env, p; ws)
  _fitness(rep_results) = candidate_fitness(env, rep_results)

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
      p = _load_params_from_path(start_from, plugin.ctx)
      (p.SPECIES_LIST == species_list && p.ECO_LIST == eco_list) ||
        error("start_from params are incompatible with this run: their SPECIES_LIST/ECO_LIST differ from the loaded data.")
      p
    elseif !isempty(_sobol_cands)
      @info "Using Sobol candidate 1/$(length(_sobol_cands)) as initial point"
      _sobol_cands[1]
    else
      generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
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
      generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    end

  mo_val!(env)
  loop = mo_loop_open(env, search_state; output_dir, resuming=!isnothing(resume_from))
  gio = loop.gio

  run_search_loop(loop; get_state=() -> search_state, unit="trial") do
      TProgress.@track for trial in (search_state.i+1):TRIALS
        bio_params, _ = PU.mutate_params(bio_params, space; rng=rng, mutation_mode=PU.BothMutations)
        for _ in 0:rand(rng, 0:2)
          bio_params, _ = PU.mutate_params(bio_params, space; rng=rng, mutation_mode=PU.BothMutations)
        end

        rep_results = _run(bio_params)
        fx, run_result, eco_losses = _fitness(rep_results)
        cached_sites_state = _median_rep_cached(rep_results, loss_params, scoring)

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

# Simulated-annealing driver (Step 5b rebuild; B22). Classic SA with a geometric cooling schedule on the scalar Sim-A loss, built on the shared
# machinery like the other drivers: `mo_env` (injection plan, tier references, fixed seeds, scales, Sim B), `run_candidate` (= fit_params),
# the one validation evaluation (`mo_val!` / `val_eval`, fixed common-random-number seeds), `scalar_loop_open` / `run_search_loop` (writer,
# losses.duckdb, interrupt handling, exit checkpoint) and checkpoint/resume (`resume_from`: the state carries the temperature, the counters and
# the run rng, so a resumed run continues exactly like an uninterrupted one). Proposals mutate the CURRENT point (the retired driver mutated its
# last proposal, accepted or not, which made it a random walk). Score = the LBSA score (cell-normalised aggregate under CELL_NORM, else the raw
# scalar). Knobs: yaml `sa_*` (SAOptions).
function parametrize_SA(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=3, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, scale_seed::PU.LossScales=PU.LossScales(), exec::ExecOptions=ExecOptions(), search::SearchOptions=SearchOptions())
  search_tier in (1, 2, 3, 4, 5) || error("parametrize_SA: unknown search_tier=$search_tier (expected 1, 2, 3, 4, or 5)")
  env = mo_env(; ref_soa, plugin, splots, spdf_plts, spinup_cohorts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, spinup, rng, debug,
    search_tier, n_reps, n_output_plots, no_establishment, injection_cohorts, cycle_years, scale_seed, exec,
    val_splots, val_ref_soa, val_spdf_plts, val_site_sim_years, val_spinup_cohorts, val_injection_cohorts)
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  n_species = env.n_species
  # the SAME scalar as LBSA's: the cell-normalised aggregate under CELL_NORM, else the raw total
  _score(rr, el) = loss_params.cell_norm ? Float64(sum(_mo_objectives(el, eco_species_ids, loss_params, env.scales, env.dual_b))) : convert(Float64, PU.get_total_loss(rr, loss_params))
  function _eval(p)
    rep_results = run_candidate(env, p)
    run_result, eco_losses, _ = _agg_reps(rep_results, loss_params, scoring)
    return _score(run_result, eco_losses), run_result, eco_losses, rep_results
  end
  sa = search.sa
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
    f0 = _eval(bio_params)[1]
    cur = SA.SACandidate(bio_params, f0)
    search_state = SA.SAState(cur, cur, rng; max_iter=TRIALS, initial_t=sa.initial_t, t=sa.initial_t, min_t=sa.min_t, alpha=sa.alpha, acceptance_rule=sa.acceptance,
                              reheat_prob_threshold=sa.reheat_prob_threshold, reheat_after_iters=sa.reheat_after_iters)
    push!(search_state.best_iterations, (0, f0, f0))
  else
    @info "Resuming from $resume_from"
    search_state = JLD2.load_object(resume_from)
    search_state.max_iter = TRIALS
    rng = search_state.rng   # the checkpoint's rng IS the run's rng from here on (as in the uninterrupted run, where state.rng === rng)
  end
  if TRIALS < 1 || SA.is_search_over(search_state)
    return search_state
  end

  mo_val!(env)
  loop, losses_db = scalar_loop_open(search_state; output_dir)
  writer_ch = loop.writer_ch
  run_search_loop(loop; get_state=() -> search_state, unit="trial") do
    TProgress.@track for trial in (search_state.i+1):TRIALS
      bio_params, _ = PU.mutate_params(search_state.current.x, space; rng=rng)
      next_loss, run_result, eco_losses, rep_results = _eval(bio_params)
      is_new_best = SA.search_cmp!(SA.SACandidate(bio_params, next_loss), search_state)
      val_sim_sample = nothing
      if is_new_best
        iter = search_state.best_iteration
        total = convert(Float64, search_state.best.fx)
        @info "New best @ $iter | loss=$total"
        if env.have_val
          try
            val_result = env.val_fit(search_state.best.x)
            @info "Val loss @ $iter | loss=$(convert(Float64, PU.get_total_loss(val_result[1], loss_params)))"
            val_sim_sample = n_output_plots > 0 ? _filter_cached_to_df(val_result[2], env.sampled_ids_val) : nothing
          catch e
            @warn "val fit_params failed" exception = (e, catch_backtrace())
          end
        end
        record_best_losses!(losses_db, iter, total, run_result, eco_losses, search_state.best.x; eco_list, species_list, n_species, loss_params, print_breakdown=true)
      end
      if is_new_best || search_state.i % 50 == 0
        cached_sites_state = _median_rep_cached(rep_results, loss_params, scoring)
        sim_sample = (is_new_best && n_output_plots > 0) ? _filter_cached_to_df(cached_sites_state, env.sampled_ids) : nothing
        put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state), splots, _cohorts_df(cached_sites_state), env.emp_sample, sim_sample, is_new_best ? env.emp_sample_val : nothing, val_sim_sample))
      end
      SA.search_update_rule!(search_state) && break
    end
  end
  return search_state
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
function _seed_prob_estab!(tables::BiomassSuccessionPlugin.BSTables, params, eco_list, species_list, eco_species_ids)
  tbl = tables.prob_estab
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
function _seed_maturity!(tables::BiomassSuccessionPlugin.BSTables, params, species_list)
  tbl = tables.maturity
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
# ---------------------------------------------------------------------------------------------------------------
# Single source of truth for the DATA-PREP keyword arguments (train/val/test split, region filters, site-class
# tiers, species tiering, outlier exclusions, ...). Used by `run_from_yaml` (training) AND by the analysis scripts
# (`prepare_from_cfg`), so a script re-derives EXACTLY the plots/split the run trained on.
# ---------------------------------------------------------------------------------------------------------------

# Parse the data-prep inputs from a loaded config (`load_config`) into a NamedTuple whose names match `parametrize`'s kwargs.
function _cfg_data_inputs(cfg0::AbstractDict)
  cfg = _as_config(cfg0)
  # Plots to DROP from the loaded set (outlier exclusion): the inline `exclude_plots:` list and/or the rows of an
  # `exclude_plots_csv:` file with the four plot columns; merged (inline first).
  exclude_plots = copy(cfg["exclude_plots"])
  if cfg["exclude_plots_csv"] !== nothing
    for r in eachrow(CSV.read(cfg["exclude_plots_csv"], DataFrame))
      push!(exclude_plots, (Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot)))
    end
  end
  return (
    cohorts_db_path=cfg["cohorts_db_path"],   # (the FIADB env override is applied by load_config)
    filter_eco_field=cfg["filter_eco_field"],
    eco_field=cfg["eco_field"],
    tablename=cfg["tablename"],
    filter_ecos=cfg["filter_ecos"],
    filter_plots=cfg["filter_plots"],
    exclude_plots=exclude_plots,
    filter_species=cfg["filter_species"],
    skip_disturbances=cfg["skip_disturbances"],
    spinup=cfg["spinup"],
    by_subplot=cfg["by_subplot"],
    val_frac=cfg["val_frac"],
    split_seed=cfg["split_seed"],
    n_folds=cfg["n_folds"],
    fold_index=cfg["fold_index"],
    test_frac=cfg["test_frac"],
    min_trees=cfg["min_trees"],
    min_agb_frac=cfg["min_agb_frac"],
    stratify_eco_mixed=cfg["stratify_eco_mixed"],
    single_ecoregion=cfg["single_ecoregion"],
    stratify_landuse=cfg["stratify_landuse"],
    site_class_strata=cfg["site_class_strata"],
    siteclass_hi_max=cfg["siteclass_hi_max"],
    siteclass_scheme=cfg["siteclass_scheme"],
    shade_tier_csv=cfg["shade_tier_csv"],
    filter_extent=cfg["filter_extent"],
    filter_states=cfg["filter_states"],
    siteclass_l3s=cfg["siteclass_l3s"],
    max_exact=cfg["max_exact"], max_group=cfg["max_group"],
    drop_phantom=cfg["drop_phantom"],
    train_cap=cfg["train_cap"],
    train_cap_k=cfg["train_cap_k"], train_cap_min=cfg["train_cap_min"],
  )
end

# parametrize-style data-prep inputs → keyword arguments of `Data.prepare_parametrization_data` (quirks live HERE only:
# `output_dir=tablename`, split RNG seeded from `split_seed` alone so all folds share the shuffle).
function _prep_kwargs(; cohorts_db_path::String, eco_field, tablename::String, filter_eco_field::String,
  filter_ecos::Vector{String}, filter_plots::Vector{NTuple{4,Int}}, exclude_plots::Vector{NTuple{4,Int}},
  filter_species::Vector{String}, filter_planted::Bool=false, skip_disturbances::Bool, spinup::Bool, by_subplot::Bool,
  val_frac::Float64, split_seed::Int, n_folds::Int, fold_index::Int, test_frac::Float64,
  min_trees::Int, min_agb_frac::Float64, stratify_eco_mixed::Bool, single_ecoregion::Bool, stratify_landuse::Bool,
  site_class_strata::Bool, siteclass_hi_max::Int, siteclass_scheme::String, shade_tier_csv::Union{Nothing,String},
  filter_extent::Union{Nothing,String}, filter_states::Vector{Int}, siteclass_l3s::Vector{String},
  max_exact::Int, max_group::Int, train_cap::Union{Nothing,Int,String}, train_cap_k::Float64, train_cap_min::Int,
  rng::Random.AbstractRNG, drop_phantom::Bool=false)
  split_rng = (val_frac > 0.0 || n_folds > 1 || test_frac > 0.0) ? RNGType(UInt64(split_seed)) : nothing   # seeded from split_seed only ⇒ all folds share the shuffle
  return (cohorts_db_path=cohorts_db_path,
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
    filter_states=filter_states, siteclass_l3s=siteclass_l3s, max_exact=max_exact, max_group=max_group,
    train_cap=train_cap, train_cap_k=train_cap_k, train_cap_min=train_cap_min, drop_phantom=drop_phantom,
    filter_eco_field=filter_eco_field,
    filter_ecos=filter_ecos,
    filter_plots=filter_plots,
    exclude_plots=exclude_plots,
    filter_species=filter_species,
    filter_planted=filter_planted,
    RNG=rng)
end

# yaml `dual_mode` → :off | :joint | :b. Accepts Bool (true = joint, the pre-June spelling) and strings in any case
# ("off"/"false", "joint"/"true", "b"). The single parser for every reader (run_from_yaml, the disturbance-shock setup, tools).
function _parse_dual_mode(v)::Symbol
  v === true && return :joint
  (v === false || v === nothing) && return :off
  s = lowercase(strip(String(v)))
  s in ("off", "false") && return :off
  s in ("joint", "true") && return :joint
  s == "b" && return :b
  error("dual_mode must be off | joint | b (or true/false), got $(repr(v))")
end

# yaml `loss_before_injection` (B14). The 2026-10-15 scoring rule (shock → score → sync injection) is defined for sync
# injection only (override_injection + override_injection_sync, not replace). There the default is true; for the other
# injection modes (append, replace) the default is the legacy order (false) and an explicit `true` is an error: those modes
# used to score pre-injection silently, with the species-level exclusion and no recruit exclusion. Without injection the
# flag has no effect. `g(key, default)` reads the already-coerced config values (called by `load_config`).
function _resolve_loss_before_injection(g)::Bool
  v = g("loss_before_injection", nothing)
  inj = Bool(g("override_injection", false))
  rule = inj && Bool(g("override_injection_sync", false)) && !Bool(g("override_injection_replace", false))
  v === nothing && return !inj || rule
  b = Bool(v)
  b && inj && !rule && error("loss_before_injection: true needs override_injection_sync (and not override_injection_replace); " *
                             "the other injection modes score in the legacy order — set loss_before_injection: false or omit it")
  return b
end

# Disturbance-shock data source (yaml `disturbance_full_removals: <parquet>`, default data/cohort_disturbance_full.parquet in the
# repo if present). Active only with the new scoring order + sync injection; otherwise "" (legacy behaviour byte-identical: no
# partial-drop attach, no shock rows). Pure: returns the path, sets nothing.
function _disturbance_shock_path(cfg0::AbstractDict)
  cfg = _as_config(cfg0)
  lbi = cfg["loss_before_injection"]   # resolved by load_config (_resolve_loss_before_injection)
  dual = cfg["dual_mode"] !== :off
  # Sim A needs sync injection; Sim B (dual mode / simB_disturb_only) uses the full removals + partial drops on its own
  on = lbi && ((cfg["override_injection"] && cfg["override_injection_sync"]) || dual || cfg["simB_disturb_only"])
  explicit = cfg["disturbance_full_removals"]   # nothing = default parquet | "" = disabled | path
  default = joinpath(dirname(@__DIR__), "data", "cohort_disturbance_full.parquet")
  path = ""
  if on
    if explicit === nothing
      isfile(default) ? (path = default) : @warn "override_injection_sync with loss_before_injection: no disturbance data (set `disturbance_full_removals:` or build data/cohort_disturbance_full.parquet with tools/export_cohort_disturbance.py) — fully-removed cohorts are NOT shocked before scoring"
    elseif explicit == ""
      path = ""
    else
      path = explicit
      isfile(path) || (@warn "disturbance_full_removals: $(path) not found — disturbance shocks disabled"; path = "")
    end
  end
  path
end

"""
    data_options(cfg) -> Data.DataOptions

The data-preparation inputs of a run, read from a loaded config (the disturbance-shock parquet). Pure; pass the result as
`parametrize(data_opts=...)` / `Data.prepare_parametrization_data(data_opts=...)` (`prepare_from_cfg` does it for you).
"""
data_options(cfg0::AbstractDict)::Data.DataOptions = Data.DataOptions(disturbance_path=_disturbance_shock_path(cfg0))

"""
    scoring_options(cfg) -> ScoringOptions

The scoring rule of a run, read from a loaded config: the injection mode (`override_injection[_replace|_sync|_noise|_disturbance]`),
`loss_before_injection`, `exclude_recruits_at_entry`, `w_scope`, `death_term` and the initial-biomass perturbation (`init_perturb_frac/_cap`).
Pure: it touches no module state.
"""
function scoring_options(cfg0::AbstractDict)::ScoringOptions
  cfg = _as_config(cfg0)
  w_scope = cfg["w_scope"]; w_scope in (:all, :survivors) || error("w_scope must be all | survivors")
  death_term = cfg["death_term"]; death_term in (:none, :likelihood) || error("death_term must be none | likelihood")
  return ScoringOptions(; override=cfg["override_injection"], replace=cfg["override_injection_replace"], sync=cfg["override_injection_sync"],
    noise=cfg["override_injection_noise"], disturbance=cfg["override_injection_disturbance"],
    loss_before_injection=cfg["loss_before_injection"], exclude_recruits_at_entry=cfg["exclude_recruits_at_entry"],
    w_scope=w_scope, death_term=death_term,
    init_perturb_frac=cfg["init_perturb_frac"], init_perturb_cap=FloatType(cfg["init_perturb_cap"]),
    cycle=Data.CycleOptions(use_fia_cycle=cfg["fia_cycle"], merge=cfg["fia_cycle_merge"]))
end

"""
    simb_options(cfg) -> SimBOptions

Sim B / dual-mode options of a run, read from a loaded config. `t4_primary` is true for a dual_mode `b` run and for a tier-4 run without a dual.
Pure: it touches no module state.
"""
function simb_options(cfg0::AbstractDict)::SimBOptions
  cfg = _as_config(cfg0)
  dual = cfg["dual_mode"]
  return SimBOptions(; dual_mode=dual, tier_b=cfg["tier_b"], t4_primary=(dual === :b || (dual === :off && cfg["tier"] == 4)),
    simb_spinup=cfg["simB_spinup"], simb_disturb_only=cfg["simB_disturb_only"],
    t4_per_plot_mean=cfg["simB_per_plot_mean"], t4_mean_over_reps=cfg["simB_rep_mean"])
end

"""
    scoring_from_config(cfg) -> (scoring::ScoringOptions, simb::SimBOptions)

The ONE place the scoring and Sim B options are built. `run_from_yaml` and the analysis tools (tost, scatter, plots) all call it, so a tool
scores exactly the way the run was trained; the options are then passed explicitly to `parametrize`, `fit_params` and the helpers.

The data-preparation inputs live in `data_options`, the parameter-space switches in `param_space_options`.
"""
function scoring_from_config(cfg0::AbstractDict)
  cfg = _as_config(cfg0)
  scoring = scoring_options(cfg)
  simb = simb_options(cfg)
  return scoring, simb
end

"""
    prep_kwargs_from_cfg(cfg::AbstractDict; split_rng_seed_override=nothing, rng=nothing) -> NamedTuple

Keyword arguments for `Data.prepare_parametrization_data`, computed from a YAML-loaded config EXACTLY as the training
path (`run_from_yaml` → `parametrize`) computes them: `\${VAR}`/`~` expansion of string values, the `FIADB` env override,
`exclude_plots` + `exclude_plots_csv`, `filter_plots`/`filter_species`/`filter_extent`/`filter_ecos`, region keys
(`filter_states`, `siteclass_l3s`, `max_exact`, `max_group`, `train_cap[_k|_min]`), site-class/shade-tier keys, and the
split RNG (seeded from `split_seed` when val/test/folds are used). `split_rng_seed_override` replaces `split_seed`;
`rng` is the (currently unused by prepare) `RNG` argument — default is derived from `seed` without touching global RNG state.
Per-fold scripts can override with `prepare_from_cfg(cfg; fold_index=k)`.
"""
function prep_kwargs_from_cfg(cfg0::AbstractDict; split_rng_seed_override=nothing, rng::Union{Nothing,Random.AbstractRNG}=nothing)
  cfg = _as_config(cfg0)
  din = _cfg_data_inputs(cfg)
  split_seed = split_rng_seed_override === nothing ? din.split_seed : Int(split_rng_seed_override)
  rng_ = rng === nothing ? RNGType(rand(Random.Xoshiro(cfg["seed"]), UInt64)) : rng   # == run_from_yaml's RNGType(rand(UInt64)) after Random.seed!(seed)
  return merge(_prep_kwargs(; din..., split_seed=split_seed, rng=rng_), (; data_opts=data_options(cfg)))
end

"""
    prepare_from_cfg(cfg::AbstractDict; split_rng_seed_override=nothing, kw...)

`Data.prepare_parametrization_data(; prep_kwargs_from_cfg(cfg)..., kw...)` — returns
`(splots, eco_list, species_list, eco_species_ids, splots_val, splots_test, outs::Data.DataOutputs)`. Extra `kw` override individual arguments.
"""
function prepare_from_cfg(cfg::AbstractDict; split_rng_seed_override=nothing, kw...)
  return Data.prepare_parametrization_data(; prep_kwargs_from_cfg(cfg; split_rng_seed_override=split_rng_seed_override)..., kw...)
end

function run_from_yaml(yaml_path::String; overrides::AbstractDict=Dict{String,Any}(), exec::Union{Nothing,ExecOptions}=nothing, post_search_hook=nothing)
  exec = exec === nothing ? exec_options_from_env() : exec   # env PAN_TIMING / PAN_SKIP_VAL / PAN_PARALLEL / PAN_NEWBEST_STATS, read once for the whole run
  get!(ENV, "PAN_DATA", "runs")          # data dir for ${PAN_DATA}/*.csv|*.duckdb in the yaml; default = repo runs/ (local),
                                         # set PAN_DATA=<dir> on the cluster (e.g. $SCRATCH/runs/data) → one yaml, both layouts
  get!(ENV, "PAN_OUT", "runs/outputs")   # output root for a ${PAN_OUT}/... output_dir in the yaml; default local, set on cluster.
                                         # (When launched via train_resumable.sh, output_dir is overridden anyway — this is the bare-run fallback.)
  # The one validated reader: unknown/retired keys are errors, values are coerced, defaults filled, ${VAR}/$VAR/~ expanded.
  # `overrides` are runner/CLI values (e.g. per-fold fold_index/output_dir).
  cfg = load_config(yaml_path; overrides=overrides)

  seed = cfg["seed"]
  Random.seed!(seed)
  rng = RNGType(rand(UInt64))

  # Data-prep inputs (filter_plots, exclude_plots[_csv], split/region/site-class keys, ...) are parsed by the SAME
  # function the analysis scripts use (`prep_kwargs_from_cfg`), so training and re-derivations can never drift.
  din = _cfg_data_inputs(cfg)
  filter_plots = din.filter_plots
  exclude_plots = din.exclude_plots
  isempty(exclude_plots) || @info "exclude_plots: dropping $(length(exclude_plots)) plots from the loaded set"

  sw_size = cfg["smoothing_window_size"]
  smoothing_window = sw_size > 0 ?
                     PU.get_smoothing_window(; smoothing_window=sw_size, smoothing_variance=FloatType(cfg["smoothing_variance"])) :
                     FloatType[one(FloatType)]

  # K-fold: each fold writes to <output_dir>/fold_<fold_index> so the 5 fold runs don't collide.
  _n_folds = cfg["n_folds"]; _fold_index = cfg["fold_index"]
  _output_dir = cfg["output_dir"]
  _n_folds > 1 && (_output_dir = joinpath(_output_dir, "fold_$(_fold_index)"))
  isdir(_output_dir) || mkpath(_output_dir)

  _db_path = din.cohorts_db_path   # (yaml value, with the FIADB env override applied by load_config)

  # Fields shared by parametrize, plot_sample, and plot_sample_sobol
  data_opts = data_options(cfg)   # data-preparation inputs (disturbance-shock parquet), passed to parametrize
  common_kw = (
    data_opts=data_opts,
    cohorts_db_path=_db_path,
    filter_eco_field=din.filter_eco_field,
    eco_field=din.eco_field,
    tablename=din.tablename,
    output_dir=_output_dir,
    filter_ecos=din.filter_ecos,
    filter_plots=filter_plots,
    filter_species=din.filter_species,
    skip_disturbances=din.skip_disturbances,
    spinup=din.spinup,
    by_subplot=din.by_subplot,
    no_establishment=cfg["no_establishment"],
    min_trees=din.min_trees,
    min_agb_frac=din.min_agb_frac,
    stratify_eco_mixed=din.stratify_eco_mixed,
    single_ecoregion=din.single_ecoregion,
    stratify_landuse=din.stratify_landuse,
    site_class_strata=din.site_class_strata,
    siteclass_hi_max=din.siteclass_hi_max,
    siteclass_scheme=din.siteclass_scheme,
    shade_tier_csv=din.shade_tier_csv,
    filter_extent=din.filter_extent,
    filter_states=din.filter_states,
    siteclass_l3s=din.siteclass_l3s,
    max_exact=din.max_exact, max_group=din.max_group, drop_phantom=din.drop_phantom,
    train_cap=din.train_cap,
    train_cap_k=din.train_cap_k, train_cap_min=din.train_cap_min,
    split_only=cfg["split_only"],
    bins_idx=cfg["bins_idx"],
  )
  n_output_plots = cfg["n_output_plots"]
  search_mode = cfg["search_mode"]

  scoring, simb = scoring_from_config(cfg)   # scoring rule + Sim B options (shared with the analysis tools), passed explicitly below
  search = search_options_from_config(cfg)   # LBSA / Igel / MO-CMA-ES-ranking knobs of the run, passed to the drivers explicitly
  search.mocmaes_nds && @info "MO-CMA-ES offspring ranking: TRUE NSGA-II non-dominated sorting (Pareto fronts + crowding)"
  tables = bs_tables_from_cfg(cfg, _db_path)   # the Biomass Succession data tables of the run, passed explicitly (parametrize(tables=...))

  # plot modes (B8): the plots use the training context (same data prep, splits, strata, loss and scoring flags as a search)
  plot_mode = search_mode in ("plot_only", "plot_only_sobol")
  plot_params_path = plot_mode && search_mode == "plot_only" ? cfg["params_path"] : ""
  plot_sobol_db = search_mode == "plot_only_sobol" ? something(cfg["sobol_candidates_db"], "") : ""
  search_mode == "plot_only" && isempty(plot_params_path) && error("search_mode=plot_only requires params_path in yaml")
  search_mode == "plot_only_sobol" && isempty(plot_sobol_db) && error("search_mode=plot_only_sobol requires sobol_candidates_db in yaml")

  _res = parametrize(;
    common_kw...,
    exclude_plots=exclude_plots,
    search_mode=plot_mode ? "igelmo" : search_mode,
    context_only=plot_mode,
    tier=cfg["tier"],
    smoothing_window=smoothing_window,
    unbinned_w=cfg["unbinned_w"],   # Sim A: exact per-year EMD (no binning/age-smoothing)
    w_count_balance=cfg["w_count_balance"],
    w_count_balance_mode=cfg["w_count_balance_mode"],
    w_count_beta=cfg["w_count_beta"],
    w_count_pct_floor=cfg["w_count_pct_floor"],
    rankw_mode=cfg["rankw_mode"],
    rankw_beta=cfg["rankw_beta"],
    rankw_pct_floor=cfg["rankw_pct_floor"],
    param_split_species=cfg["param_split_species"],
    param_pooling=cfg["param_pooling"],
    succession_plugin=cfg["succession_plugin"],
    tables=tables,
    pnet_climate_cache=cfg["pnet_climate_cache"],
    pnet_co2_csv=cfg["pnet_co2_csv"],
    pnet_generic=cfg["pnet_generic"],
    pnet_soil_texture=cfg["pnet_soil_texture"],
    pnet_soil_csv=cfg["pnet_soil_csv"],
    hbad_priors_csv=cfg["hbad_priors_csv"],
    hbad_coefs_csv=cfg["hbad_coefs_csv"],
    hbad_init=cfg["hbad_init"],
    hbad_seed_rain=cfg["hbad_seed_rain"],
    hbad_prob_estab_csv=cfg["hbad_prob_estab_csv"],
    hbad_cycle_years=cfg["cycle_years"],
    cmaes_blocks=cfg["cmaes_blocks"],
    param_tier_merge=cfg["param_tier_merge"],
    TRIALS=cfg["trials"],
    resume_from=cfg["resume_from"],
    start_from=cfg["start_from"],
    force_restart_from_random=cfg["force_restart_from_random"],
    sobol_n=cfg["sobol_n"],
    saltelli=cfg["saltelli"],
    n_reps=cfg["n_reps"],
    sobol_candidates_db=cfg["sobol_candidates_db"],
    sobol_top_frac=cfg["sobol_top_frac"],
    n_output_plots=n_output_plots,
    val_frac=din.val_frac,
    split_seed=din.split_seed,
    n_folds=din.n_folds,
    fold_index=din.fold_index,
    test_frac=din.test_frac,
    diagnose=cfg["diagnose"],
    cycle_years=cfg["cycle_years"],
    loss_lambda=cfg["loss_lambda"],
    loss_alpha=cfg["loss_alpha"],
    agb_hinge=cfg["agb_hinge"],
    agb_hinge_threshold=cfg["agb_hinge_threshold"],
    agb_hinge_pct=cfg["agb_hinge_pct"],
    agb_hinge_pct_min=cfg["agb_hinge_pct_min"],
    agb_hinge_pct_max=cfg["agb_hinge_pct_max"],
    agb_hinge_l2=cfg["agb_hinge_l2"],
    agb_hinge_p=cfg["agb_hinge_p"],   # hinge exponent pen(h)=h^p, default 2 (L2)
    agb_hinge_beta=cfg["agb_hinge_beta"],
    agb_normalize=cfg["agb_normalize"],
    w_normalize=cfg["w_normalize"],
    cell_normalize=cfg["cell_normalize"],
    w_scale_factor=cfg["w_scale_factor"],
    w_smooth=cfg["w_smooth"],
    w_smooth_band=cfg["w_smooth_band"],
    w_smooth_conc=cfg["w_smooth_conc"],
    w_smooth_beta=cfg["w_smooth_beta"],
    w_smooth_auto_knee=cfg["w_smooth_auto_knee"],
    w_p=cfg["w_p"],
    w_hinge=cfg["w_hinge"],
    w_hinge_pct=cfg["w_hinge_pct"],
    w_hinge_pct_min=cfg["w_hinge_pct_min"],
    w_hinge_pct_max=cfg["w_hinge_pct_max"],
    loss_piecewise=cfg["loss_piecewise"],
    w_pivot=cfg["w_pivot"],
    agb_pivot=cfg["agb_pivot"],
    scoring=scoring,
    simb=simb,
    space_opts=param_space_options(cfg),   # which parameters are searched and their boxes
    cmaes_lambda=cfg["cmaes_lambda"],
    cmaes_sigma0=cfg["cmaes_sigma0"],
    cmaes_warmstart_seeds=cfg["cmaes_warmstart_seeds"],
    ipop=cfg["ipop"],
    ipop_stagnation=cfg["ipop_stagnation"],
    ipop_max_barren_restarts=cfg["ipop_max_barren_restarts"],
    cmaes_archive_cap=cfg["cmaes_archive_cap"],
    seed_archive_from=cfg["seed_archive_from"],
    cmaes_integer_handling=cfg["cmaes_integer_handling"],
    cmaes_integer_std_factor=cfg["cmaes_integer_std_factor"],
    cmaes_single_cov=cfg["cmaes_single_cov"],
    nsga2_pop=cfg["nsga2_pop"],
    nsga2_offspring=cfg["nsga2_offspring"],
    nsga2_eta_c=cfg["nsga2_eta_c"],
    nsga2_eta_m=cfg["nsga2_eta_m"],
    nsga2_pc=cfg["nsga2_pc"],
    nsga2_pm=cfg["nsga2_pm"],
    nsga2_sobol_init=cfg["nsga2_sobol_init"],
    igel_mu=cfg["igel_mu"],
    igel_sigma0=cfg["igel_sigma0"],
    igel_sobol_init=cfg["igel_sobol_init"],
    igel_sobol_pool_k=cfg["igel_sobol_pool_k"],        # >1 ⇒ oversample: keep best μ·k by dominance, random-pick μ
    igel_sobol_raw_mult=cfg["igel_sobol_raw_mult"],    # raw Sobol pool = (μ·pool_k)·raw_mult candidates evaluated
    igel_niche_radius=cfg["igel_niche_radius"],
    igel_reseed_sigma=cfg["igel_reseed_sigma"],
    igel_reseed_random_frac=cfg["igel_reseed_random_frac"],
    igel_maturity=cfg["igel_maturity"],
    igel_seed_maturity=cfg["igel_seed_maturity"],
    igel_warm_start=cfg["igel_warm_start"],
    igel_freeze_seed_growth=cfg["igel_freeze_seed_growth"],
    cc_group_size=cfg["cc_group_size"],
    cc_spec_gens=cfg["cc_spec_gens"],
    cc_integ_gens=cfg["cc_integ_gens"],
    cc_cycles=cfg["cc_cycles"],
    cc_fix_others=cfg["cc_fix_others"],
    gdigel=cfg["gdigel"],
    cmame_alpha=cfg["cmame_alpha"],
    cmame_grid=cfg["cmame_grid"],
    cmame_reseed_explore=cfg["cmame_reseed_explore"],
    cmame_restart_patience=cfg["cmame_restart_patience"],
    cmame_sobol_reseed=cfg["cmame_sobol_reseed"],
    cmame_mo_rank=cfg["cmame_mo_rank"],
    balanced_quality=cfg["balanced_quality"],
    archive_by_sp=cfg["archive_by_sp"],
    bounds_by_sobol=cfg["bounds_by_sobol"],
    top_seeds=cfg["top_seeds"],
    dominance_species=cfg["dominance_species"],
    exec=exec,
    search=search,
    post_search_hook=post_search_hook,
    rng=rng)
  plot_mode || return _res
  return search_mode == "plot_only" ?
    plot_sample(_res; params_path=plot_params_path, n_output_plots=max(1, n_output_plots), rng_seed=seed) :
    plot_sample_sobol(_res; sobol_candidates_db=plot_sobol_db, n_output_plots=max(1, n_output_plots),
                      n_sobol_params_to_plot=max(1, cfg["n_sobol_params_to_plot"]), rng_seed=seed)
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

include("search_loop.jl")
include("gdigel_driver.jl")

include("climate_scenario.jl")
include("scenario.jl")
include("scenario_site.jl")
include("scenario_volume.jl")

end
