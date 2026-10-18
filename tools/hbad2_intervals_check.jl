# Rebuild the HBAD2 interval product with Pan's Data layer and compare it with the prototype's Arrow files (read-only), order-independently.
#   PAN_DATA=/workspace/landis_region4/data_r4 ./julia_gdal.sh --project=. --threads=2 tools/hbad2_intervals_check.jl [--proto /workspace/landis_hbad2] [--split <split_plots.csv>] [--cache <dir>] [--db <duckdb>]
using Pan, DataFrames, Printf
function argval(name, default); i = findfirst(==(name), ARGS); i === nothing ? default : ARGS[i+1]; end
const PROTO = argval("--proto", "/workspace/landis_hbad2")
const SPLIT = argval("--split", "/workspace/landis_region4/runs_local/gaflst_split/split_plots.csv")
const DB = argval("--db", joinpath(get(ENV, "PAN_DATA", "/workspace/landis_region4/data_r4"), "slim_r4.duckdb"))
const CACHE = argval("--cache", mktempdir())
split = Pan.Data.read_split_plots(SPLIT)
t0 = time(); mine = Pan.Data.load_hbad2_intervals(CACHE, DB, split; rebuild=true); @printf("built in %.1f s (cache %s)\n", time() - t0, CACHE)
proto = Pan.Data.read_hbad2_intervals(joinpath(PROTO, "results", "hbad2"))
@printf("rows  intervals %d / %d   trees %d / %d   ingrowth %d / %d   (Pan / prototype)\n", nrow(mine.intervals), nrow(proto.intervals), nrow(mine.trees), nrow(proto.trees), nrow(mine.ingrowth), nrow(proto.ingrowth))
cmp = Pan.Data.compare_interval_products(mine, proto)
println(cmp)
again = Pan.Data.read_hbad2_intervals(CACHE)
println("cache round trip identical: ", Pan.Data.compare_interval_products(again, mine))
ok = cmp.intervals.only_a == 0 && cmp.intervals.only_b == 0 && cmp.intervals.differing == 0 && cmp.trees.only_a == 0 && cmp.trees.only_b == 0 && cmp.ingrowth.only_a == 0 && cmp.ingrowth.only_b == 0
println(ok ? "INTERVALS IDENTICAL (order-independent)" : "INTERVALS DIFFER")
exit(ok ? 0 : 1)
