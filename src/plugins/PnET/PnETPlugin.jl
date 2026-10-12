# PnET-Succession (v2 cohort ecophysiology) as a Pan succession plugin. Science: docs/pnet_spec.md.
# Structure-of-arrays: every per-cohort quantity is a CSR (:cohort) array moved by PanCore.swap_cohort!; per-site state is scalar arrays.
# Hot loop is allocation-free: per-thread scratch (PnETScratch) lives in a global pool, indexed by Threads.threadid().
module PnETPlugin
using ..PanCore
import ..BiomassSuccessionPlugin
using DataFrames
import Random, JLD2, DuckDB, Dates

const IMAX = 5     # sublayers per cohort (E2 IMAX). Compile-time constant: scratch/loop sizes depend on it.
const MAXL = 3     # max canopy layers supported (E2 MaxCanopyLayers = 3)

include("params.jl")   # PFT tables, PnETParams, PnETEcoParams, Saxton-Rawls, generate_*
const FIXED = Ref(PnETConst())   # global fixed constants (settable from yaml/tests)
include("climate.jl")  # PRISM cache -> derived array, daylength / PAR
include("plugin.jl")   # SoA layout, kernels, process_plugin!, add_cohort!, set_cohort_bio!
include("site_attrs.jl")   # DB -> per-site attributes (clim_id, cal_year0, lat, soil), init_sites!

__init__() = (scratch_pool(); nothing)   # pre-size the per-thread scratch pool for the running thread count

export PnET, PnETParams, PnETEcoParams, PnETConst, PnETClimate
end
