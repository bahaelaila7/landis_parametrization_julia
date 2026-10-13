# HBAD (Height / Basal Area / Density) succession as a Pan plugin: cohort = (species, age, trees/ha, quadratic mean diameter, mean height); fixed
# empirical priors from results/hbad (GA+FL FIA); fitted multipliers on diameter growth and mortality + per-species SDImax. Documentation: docs/hbad_plugin.md.
# Structure-of-arrays: every per-cohort quantity is a CSR (:cohort) array moved by PanCore.swap_cohort!; per-site state is scalar arrays.
# The annual kernel is allocation-free and uses no dynamic dispatch (isbits SpPrior table, concrete eco-parameter struct).
module HBADPlugin
using ..PanCore
import ..BiomassSuccessionPlugin
using DataFrames
import CSV
import Random

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DEFAULT_PRIORS_CSV = joinpath(REPO_ROOT, "results", "hbad", "priors.csv")
const DEFAULT_COEFS_CSV = joinpath(REPO_ROOT, "results", "hbad", "translation_coefs.csv")

include("priors.jl")   # SpPrior, PRIORS, species->key resolution, translation, configure!
include("params.jl")   # HBADParams, HBADEcoParams, generate_*
include("plugin.jl")   # SoA layout, kernel, process_plugin!, add_cohort!, set_cohort_bio!

export HBAD, HBADParams, HBADEcoParams, HBADConst, SpPrior
end
