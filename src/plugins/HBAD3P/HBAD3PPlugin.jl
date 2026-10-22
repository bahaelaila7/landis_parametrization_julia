# HBAD3P (process-grounded HBAD3 cohort model) as a Pan succession plugin. Port of the reference kernel of the prototype tree /workspace/landis_hbad3
# (src/plugins/HBAD3/HBAD3P.jl: header = the equations; docs/hbad3p_pipeline.md = the fit pipeline) with the GA+FL parameter set P3_GAFL; documentation: docs/hbad3p_plugin.md.
# Cohort state (Float64): species, age A, trees/ha N (per ha of the FORESTED subplot when farea = 1), quadratic mean diameter D (cm), mean height H (m), crown ratio CR,
# vigour node zq (5-point Gauss-Hermite). Per site: water index W (period AET/PET) and its plot normal Wn. Biomass never enters the dynamics; `c_bio` is a REPORTED allometric
# AGB (HBAD v1 priors, as HBAD2). Structure-of-arrays: every per-cohort quantity is a CSR (:cohort) array; the annual kernel is allocation-free in the steady state, uses isbits
# parameter structs and no dynamic dispatch, and is deterministic (expected-value dynamics, no random numbers).
module HBAD3PPlugin
using ..PanCore
import ..BiomassSuccessionPlugin
import ..HBADPlugin
using DataFrames
import CSV
import Statistics

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DEFAULT_PARAMS_DIR = joinpath(REPO_ROOT, "results", "hbad3p", "P3_GAFL")

include("params.jl")   # SpPP, GlobPP, RecGP, Allo, HBAD3PParams, HBAD3PEcoParams, csv loader, species -> key mapping
include("context.jl")  # HBAD3PConst, HBAD3PContext, generate_*
include("kernel.jl")   # step_cohorts! (reference step_site!), recruitment rates / application, allometric AGB
include("plugin.jl")   # SoA layout, process_plugin!, add_cohort!, plant!, scale_cohort!

export HBAD3P, HBAD3PParams, HBAD3PEcoParams, HBAD3PConst, HBAD3PContext, SpPP, GlobPP, RecGP
end
