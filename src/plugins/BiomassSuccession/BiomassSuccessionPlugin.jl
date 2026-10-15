module BiomassSuccessionPlugin
using ..PanCore
using DataFrames
import Random, Distributions as Dists
import Arrow, DuckDB, ArchGDAL

"""
    BSTables(; fixed_longevity, longevity, longevity_default, shade_tol, shade_tol_default, prob_estab, maturity, bmax_floor, anpp_floor)

The data tables of a Biomass Succession run (immutable; built once from the config by `Pan.bs_tables_from_cfg` and passed explicitly to everything that reads them:
`generate_biomass_params`, `make_biomass_param_dists`, `build_param_floors`, the PnET/HBAD generators). Each table is `nothing` when its feature is off.

  * `fixed_longevity`: pin LONGEVITY out of the search at one value (`nothing` = optimize normally). `longevity` (per-species, from `species_longevity_ref`) ALSO pins it
    and takes precedence; species absent from it use `longevity_default`.
  * `shade_tol` (species_symbol -> class 1-5, from shadetol_all_species.csv): drops SHADE_TOL from the search and seeds each species' class (absent -> `shade_tol_default`).
  * `prob_estab` ((category, L3, land_use) -> p): SEEDS PROB_ESTAB_SPP (stays in the search); with `prob_estab_lower_bound` also its lower bound.
  * `maturity` (category -> SONA age): SEEDS MATURITY (stays in the search).
  * `bmax_floor` / `anpp_floor` ((category, L3, land_use) -> g/m2 resp. g/m2/yr): per-(eco,species) lower bounds of B_MAX / ANPP_MAX applied at decode.
"""
Base.@kwdef struct BSTables
  fixed_longevity::Union{Nothing,Float64} = nothing
  longevity::Union{Nothing,Dict{String,Float64}} = nothing
  longevity_default::Float64 = 275.0
  shade_tol::Union{Nothing,Dict{String,Int}} = nothing
  shade_tol_default::Int = 3
  prob_estab::Union{Nothing,Dict{Tuple{String,String,String},Float64}} = nothing
  maturity::Union{Nothing,Dict{String,Int}} = nothing
  bmax_floor::Union{Nothing,Dict{Tuple{String,String,String},Float64}} = nothing
  anpp_floor::Union{Nothing,Dict{Tuple{String,String,String},Float64}} = nothing
end

include("plugin.jl")   # structs, process_plugin!, simulation steps — always needed
include("params.jl")   # generate_eco_params, generate_biomass_params
include("export.jl")   # export_landis_params (writes LANDIS-II input files)

export BSTables, generate_biomass_params, generate_eco_params, spinup_cohorts!, export_landis_params,
       export_initial_communities_csv, export_initial_communities_tif,
       export_ecoregions_txt, export_scenario_file, export_eco_ecocode_mapping
end
