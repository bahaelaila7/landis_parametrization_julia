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
# Per-species fixed SHADE_TOL (species_symbol -> shade class 1-5), from the data-derived
# shadetol_all_species.csv. When set, make_biomass_param_dists drops SHADE_TOL from the search and
# generate_biomass_params seeds each species' class from the table. Species absent → SHADE_TOL_DEFAULT.
const SHADE_TOL_TABLE = Ref{Union{Nothing,Dict{String,Int}}}(nothing)
const SHADE_TOL_DEFAULT = Ref{Int}(3)
# Per-(category, L3, land_use) data-derived PROB_ESTAB, from prob_estab_from_data.csv (+ _artificial).
# When set, the initial candidate's PROB_ESTAB_SPP is SEEDED from it (calibration start); it STAYS in the
# search (unlike SHADE_TOL_TABLE which pins). nothing = leave PROB_ESTAB at its generated/loaded value.
const PROB_ESTAB_TABLE = Ref{Union{Nothing,Dict{Tuple{String,String,String},Float64}}}(nothing)
# Per-category data-derived MATURITY (SONA age), from the prob_estab_all_species.csv `maturity` column.
# When set, the initial candidate's MATURITY is SEEDED per species (calibration start); stays in the search.
const MATURITY_TABLE = Ref{Union{Nothing,Dict{String,Int}}}(nothing)

include("plugin.jl")   # structs, process_plugin!, simulation steps — always needed
include("params.jl")   # generate_eco_params, generate_biomass_params
include("export.jl")   # export_landis_params (writes LANDIS-II input files)

export FIXED_LONGEVITY, LONGEVITY_TABLE, LONGEVITY_DEFAULT, SHADE_TOL_TABLE, SHADE_TOL_DEFAULT, PROB_ESTAB_TABLE, MATURITY_TABLE,
       generate_biomass_params, generate_eco_params, spinup_cohorts!, export_landis_params,
       export_initial_communities_csv, export_initial_communities_tif,
       export_ecoregions_txt, export_scenario_file, export_eco_ecocode_mapping
end
