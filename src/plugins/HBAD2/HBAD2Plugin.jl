# HBAD v2 (biomass-free, process-based cohort model) as a Pan succession plugin. Port of the prototype kernel of branch `hbad-v2`
# (src/plugins/HBAD2/HBAD2.jl there) with the selected parameter set v2g (docs/hbad_v2.md 4.1 / 5.17); documentation: docs/hbad2_plugin.md.
# Cohort state (Float64): species, age A, trees/ha N, quadratic mean diameter D (cm), mean height H (m), vigour quantile zq. Biomass never
# enters the dynamics; `c_bio` is a REPORTED allometric AGB (HBAD v1 priors). Structure-of-arrays: every per-cohort quantity is a CSR (:cohort)
# array; the annual kernel is allocation-free in the steady state, uses isbits parameter structs and no dynamic dispatch, and is deterministic
# (expected-value dynamics, no random numbers at all).
module HBAD2Plugin
using ..PanCore
import ..BiomassSuccessionPlugin
import ..HBADPlugin
using DataFrames
import CSV

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DEFAULT_PARAMS_DIR = joinpath(REPO_ROOT, "results", "hbad2")

include("params.jl")   # SpP, GlobP, RecP, Allo, HBAD2Params, HBAD2EcoParams, CSV loader, species -> key mapping
include("context.jl")  # HBAD2Const, HBAD2Context, generate_*
include("kernel.jl")   # step_cohorts! (prototype step_site!), recruitment rates / application, site reporting
include("plugin.jl")   # SoA layout, process_plugin!, add_cohort!, set_cohort_bio!, plant!

export HBAD2, HBAD2Params, HBAD2EcoParams, HBAD2Const, HBAD2Context, SpP, GlobP, RecP
end
