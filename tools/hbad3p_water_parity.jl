# Parity of the Julia water balance (HBAD3PPlugin.water_index / water_normal) with the reference (landis_hbad3 scripts/hbad3/water_balance.py):
# recompute the plot normal Wn of a sample of plots from the same inputs (plot_climate_r4.csv: L4, AWC; latitude = the L4 mean of the table's plots,
# as the reference; PRISM monthly tmean / ppt by L4 from 1980) and compare with the roster's plot_normal_water.csv. Also reports the effect of the
# scenario engine's choices (tmean = (tmin + tmax)/2; the plot's own latitude).
#   source wt_env.sh && ./julia_gdal.sh --project=. tools/hbad3p_water_parity.jl [n_plots=400]
using Pan, CSV, DataFrames, DuckDB, Statistics, Random
const H = Pan.HBAD3PPlugin
ws = get(ENV, "WORKSPACE", normpath(joinpath(@__DIR__, "..", "..")))
pcl = CSV.read(joinpath(ws, "landis_hbad3", "results", "hbad3", "plot_climate_r4.csv"), DataFrame)
wn = CSV.read(joinpath(@__DIR__, "..", "results", "hbad3p", "P3_GAFL", "plot_normal_water.csv"), DataFrame)
latl4 = Dict(g.l4[1] => mean(g.lat) for g in groupby(dropmissing(pcl, [:l4, :lat]), :l4))
d = innerjoin(dropmissing(pcl, [:l4, :lat, :awc_mm]), wn, on=[:st, :u, :c, :p])
n = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 400
d = d[randperm(MersenneTwister(1), nrow(d))[1:min(n, nrow(d))], :]
l4s = unique(String.(d.l4)); lst = join(("'" * l * "'" for l in l4s), ",")
con = DBInterface.connect(DuckDB.DB, joinpath(ws, "pan_climate", "pan_climate.duckdb"); readonly=true)
m = DataFrame(DBInterface.execute(con, "SELECT l4, var, year, month, value FROM observed_monthly WHERE l4 IN ($lst) AND var IN ('tmean','tmin','tmax','ppt') AND year BETWEEN 1980 AND 2024"))
yrs = collect(1980:2024); cl = Dict{Tuple{String,String},Matrix{Float64}}()
for r in eachrow(m)
  M = get!(cl, (String(r.l4), String(r.var))) do; fill(NaN, 12, length(yrs)); end
  M[Int(r.month), Int(r.year) - 1979] = Float64(r.value)
end
e_ref = Float64[]; e_tmm = Float64[]; e_lat = Float64[]
for r in eachrow(d)
  l = String(r.l4); P = cl[(l, "ppt")]; T = cl[(l, "tmean")]; T2 = 0.5 .* (cl[(l, "tmin")] .+ cl[(l, "tmax")])
  push!(e_ref, H.water_normal(H.water_index(T, P, latl4[l], r.awc_mm), yrs) - r.wn)
  push!(e_tmm, H.water_normal(H.water_index(T2, P, latl4[l], r.awc_mm), yrs) - r.wn)
  push!(e_lat, H.water_normal(H.water_index(T2, P, r.lat, r.awc_mm), yrs) - r.wn)
end
q(x) = round.((maximum(abs.(x)), mean(x), quantile(abs.(x), 0.95)); sigdigits=3)
println("plots $(nrow(d)) | Wn mean $(round(mean(d.wn); digits=4))")
println("reference inputs (PRISM tmean, L4-mean latitude): max |err|, mean err, p95 |err| = ", q(e_ref))
println("tmean = (tmin+tmax)/2:                                                ", q(e_tmm))
println("+ the plot's own latitude (the scenario engine):                      ", q(e_lat))
