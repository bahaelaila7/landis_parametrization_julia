# Project the same FIA plots with several parameter sets (YAML) and compare the trajectories — the test of whether
# equally-well-fitting parameter sets are MEANINGFULLY distinct (they should agree over the calibration horizon and may
# diverge in the long run).
#   ./julia_gdal.sh --project=. --threads=4 tools/compare_param_sets.jl <out.csv> <cycles> <period> <n_plots> <set1.yaml> <set2.yaml> ...
# Plots: a fixed sample of GA+FL plots measured 2018–2020 (FIA PLOT.CN, deterministic order). Grow-only plan.
using Pan, DuckDB, DataFrames, CSV, Statistics, Printf
const P = Pan
out = ARGS[1]; cycles = parse(Int, ARGS[2]); period = parse(Int, ARGS[3]); nplots = parse(Int, ARGS[4]); files = ARGS[5:end]
db = DuckDB.DB(P.SCEN_FIA_DB[]; readonly=true); con = DuckDB.DBInterface.connect(db)
cns = DataFrame(DuckDB.DBInterface.execute(con, """
  SELECT p.CN cn FROM PLOT p WHERE p.STATECD IN (12, 13) AND p.INVYR BETWEEN 2018 AND 2020
    AND (p.STATECD || '_' || p.UNITCD || '_' || p.COUNTYCD || '_' || p.PLOT) IN
        (SELECT DISTINCT plt_cn FROM curated_cohorts_landis_stdorg WHERE statecd IN (12,13) AND year(measdate) BETWEEN 2018 AND 2020)
  ORDER BY hash(p.CN) LIMIT $(nplots)""")).cn
plots = [(cn=String(c), acres=1.0) for c in cns]
plan = Dict{String,Any}("cycles" => cycles, "period" => period, "actions" => Any[])
rows = DataFrame(set=String[], elapsed=Int[], agb=Float64[], ba=Float64[], tpa=Float64[], qmd=Float64[], carbon=Float64[], nplots=Int[])
for f in files
  ps = P.load_paramset(read(f, String))
  res, w = P.run_scenario(ps, plots, plan)
  ok = [v for v in values(res) if v.status == "ok"]
  for c in 1:(cycles + 1)
    m = [v.metrics[c] for v in ok]
    push!(rows, (basename(f), (c - 1) * period, mean(x.agb for x in m), mean(x.ba for x in m), mean(x.tpa for x in m), mean(x.qmd for x in m), mean(x.carbon_live_ag for x in m), length(ok)))
  end
end
CSV.write(out, rows)
for e in unique(rows.elapsed)
  r = rows[rows.elapsed .== e, :]
  @printf("year +%3d  AGB %s  | CV %.1f %%   BA ft2/ac %s   TPA %s\n", e, join([@sprintf("%6.0f", x) for x in r.agb], " "), 100 * std(r.agb) / mean(r.agb),
          join([@sprintf("%5.0f", x) for x in r.ba], " "), join([@sprintf("%5.0f", x) for x in r.tpa], " "))
end
