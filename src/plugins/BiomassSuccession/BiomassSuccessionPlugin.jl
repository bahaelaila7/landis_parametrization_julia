module BiomassSuccessionPlugin
using ..PanCore
using DataFrames
import Random, Distributions as Dists
import Arrow, DuckDB, ArchGDAL

include("plugin.jl")   # structs, process_plugin!, simulation steps — always needed
include("params.jl")   # generate_eco_params, generate_biomass_params
include("export.jl")   # export_landis_params (writes LANDIS-II input files)

export generate_biomass_params, generate_eco_params, spinup_cohorts!, export_landis_params,
       export_initial_communities_csv, export_initial_communities_tif,
       export_ecoregions_txt, export_scenario_file, export_eco_ecocode_mapping
end
