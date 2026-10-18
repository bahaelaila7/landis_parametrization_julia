# HBAD v2 Region 4 extension: the R4 plot split = the GA+FL split (kept as is, so GA+FL held-out plots stay held out) ∪ the split of the other R4 states
# drawn by configs/region4/hbad2_r4_split.yml (same procedure). Refuses overlapping plots.
#   ./julia_gdal.sh --project=. tools/hbad2_r4_split.jl <gafl split_plots.csv> <ext split_plots.csv> <out csv>
using CSV, DataFrames
g = CSV.read(ARGS[1], DataFrame); e = CSV.read(ARGS[2], DataFrame)
k = [:statecd, :unitcd, :countycd, :plot]
nrow(innerjoin(select(g, k), select(e, k); on=k)) == 0 || error("overlapping plots between the two splits")
cols = [:statecd, :unitcd, :countycd, :plot, :eco, :set, :dom]
m = sort!(vcat(select(g, cols), select(e, cols)), k)
CSV.write(ARGS[3], m)
println("plots ", nrow(m), " (GA+FL ", nrow(g), ", other ", nrow(e), "); sets ", Dict(s => count(==(s), m.set) for s in unique(m.set)))
