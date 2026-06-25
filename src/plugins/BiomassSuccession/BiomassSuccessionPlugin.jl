module BiomassSuccessionPlugin
using ..PanCore
using DataFrames
import Random, Distributions as Dists
import Arrow, DuckDB, ArchGDAL

# Pin a parameter OUT of optimization (set from config; nothing = optimize normally). When set,
# make_biomass_param_dists drops it from the mutable set and generate_biomass_params seeds the value.
const FIXED_LONGEVITY = Ref{Union{Nothing,Float64}}(nothing)
# Per-species fixed LONGEVITY (species_symbol -> years), from the data-derived species_longevity_ref
# table. When set it ALSO pins LONGEVITY out of the search (same as FIXED_LONGEVITY). Takes precedence
# over FIXED_LONGEVITY. Species not in the table fall back to LONGEVITY_DEFAULT.
const LONGEVITY_TABLE = Ref{Union{Nothing,Dict{String,Float64}}}(nothing)
const LONGEVITY_DEFAULT = Ref{Float64}(275.0)

include("plugin.jl")   # structs, process_plugin!, simulation steps — always needed
include("params.jl")   # generate_eco_params, generate_biomass_params
include("export.jl")   # export_landis_params (writes LANDIS-II input files)

export FIXED_LONGEVITY, LONGEVITY_TABLE, LONGEVITY_DEFAULT,
       generate_biomass_params, generate_eco_params, spinup_cohorts!, export_landis_params,
       export_initial_communities_csv, export_initial_communities_tif,
       export_ecoregions_txt, export_scenario_file, export_eco_ecocode_mapping
end
