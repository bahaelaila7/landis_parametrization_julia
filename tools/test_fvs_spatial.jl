# Test the FVS raster projection pipeline.
#   julia --project=. tools/test_fvs_spatial.jl TREEMAP.tif [DB] [HORIZON] [EVERY] [VERSION]
# Run from an env where libgfortran.so.5 is on LD_LIBRARY_PATH (mamba env).
#
# Reads a TreeMap raster (clipped to your AOI), projects each unique plot-CN
# forward, and writes total + per-species AGB GeoTIFFs every EVERY years.

using DataFrames
include(joinpath(@__DIR__, "..", "src", "FVS.jl"))
using .FVS

treemap = ARGS[1]
db      = length(ARGS) >= 2 ? ARGS[2] : "/workspace/FIASQLITE2PGSQL/FIADB.duckdb"
horizon = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 50
every   = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 5
version = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 2022

outdir = joinpath(@__DIR__, "..", "tmp", "fvs_spatial")
println("treemap=$treemap db=$db horizon=$horizon every=$every version=$version")

res = FVS.simulate_spatial(; treemap_raster=treemap, db_path=db, output_dir=outdir,
    timehorizon=horizon, output_every=every, treemap_version=version,
    variant="sn")

println("\nunique CNs: ", length(res.cns), "  stands: ", length(res.stands),
        "  species: ", res.species)
println("rasters written: ", length(res.files))
for f in first(res.files, 12)
    println("  ", f)
end

# quick sanity: total AGB at year 0 vs final year, summed over CNs
using Statistics
for off in (0, horizon)
    vals = [v for ((cn, o), v) in res.total if o == off]
    isempty(vals) && continue
    println("year $off: $(length(vals)) CNs, mean total AGB = ",
            round(mean(vals), digits=1), " g/m²")
end
