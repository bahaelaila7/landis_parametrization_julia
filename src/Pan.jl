module Pan
ENV["GKSwstype"] = "nul" # this is to suppress window opening upon plotting
include("PanCore.jl")
include("Plugins.jl")
include("Parametrization.jl")
include("Search.jl")
include("Data.jl")
include("SpatialData.jl")
include("Spatial.jl")
using .PanCore
using .Plugins: BaseSitePlugin, BiomassSuccessionPlugin, PnETPlugin, HBADPlugin
import .Parametrization as PU
import .Parametrization.BiomassSuccessionParametrization as BSP
import .Parametrization.PnETParametrization as PNP
import .Parametrization.HBADParametrization as HBP
using .Search: SA, LBSA, MOLBSA, CMAES, MOCMAES, IgelMOCMAES, CMAMAE, NSGA2, CCIgel, GDIgel, Common
import .Data as Data
import .SpatialData
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

# Layout (definition order matters for consts/structs used in signatures):
# config/ -> sites, reference -> scoring/ -> loss/ -> sim/ -> io/ -> drivers/ -> entry -> diagnostics/exports -> scenario engine

include("succession_api.jl")
include("config/params_yaml.jl")
include("config/schema.jl")
include("config/options.jl")
include("config/param_space.jl")
include("config/plugin_config.jl")
include("config/tables.jl")
include("config/from_yaml.jl")
include("sites.jl")
include("reference.jl")
include("scoring/injection.jl")
include("scoring/shock.jl")
include("scoring/exclusion.jl")
include("scoring/death.jl")
include("scoring/wdiag.jl")
include("scoring/plan.jl")
include("loss/aggregate.jl")
include("loss/scales.jl")
include("sim/fit_params.jl")
include("io/plots.jl")
include("io/writer.jl")
include("io/outputs.jl")
include("io/losses_db.jl")
include("io/checkpoints.jl")
include("drivers/common.jl")
include("drivers/mo_common.jl")
include("drivers/search_loop.jl")
include("drivers/sobol.jl")
include("drivers/lbsa.jl")
include("drivers/cmaes.jl")
include("drivers/mocmaes.jl")
include("drivers/igelmo.jl")
include("drivers/ccigel.jl")
include("drivers/nsga2.jl")
include("drivers/cmame.jl")
include("drivers/molbsa.jl")
include("drivers/sa.jl")
include("drivers/gdigel.jl")
include("entry.jl")
include("diagnostics.jl")
include("spatial_sim.jl")
include("landis_export.jl")
include("species_composition.jl")
include("climate_scenario.jl")
include("scenario.jl")
include("scenario_site.jl")
include("scenario_volume.jl")
end
