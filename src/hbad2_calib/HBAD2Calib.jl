# Calibration of the HBAD2 plugin (docs/hbad2_calibration.md): data (Sim A tracked-tree cohorts, Sim B free-run stand tables), parameter vector, objectives and the
# problem a search driver minimises. The search itself (engine, writer, checkpoints) is src/drivers/hbad2_calib.jl.
module HBAD2Calib
import ..PanCore
import ..PanCore: UIntType
import ..Data
import ..Plugins
const HBAD2Plugin = Plugins.HBAD2Plugin
const Pan = parentmodule(@__MODULE__)
using DataFrames
import CSV, Dates, Random, Statistics, Distributions

include("spec.jl")
include("data.jl")
include("objective.jl")
include("problem.jl")
include("export.jl")

end
