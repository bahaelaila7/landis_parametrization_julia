# Cross-check (2026-10-22): run tools/hbad3p_crosscheck_freerun_proto.jl in the HBAD3P prototype repo (landis_hbad3, pipeline env, HBAD3_REPO=<that repo>) -> csv; then this script in this repo on the csv.
# Result: 30 GA+FL val plots, plot BA at year 0 identical (30/30, 1e-6), at year 10 identical to 1e-6 relative (30/30).
# Pan run_scenario (hbad3p roster, no-op plan, 1 cycle of 10 years) on the same plots: plot BA at year 0 and 10 vs the prototype free run (W = Wn)
using Pan, DuckDB, DataFrames, CSV, Dates, Printf, Statistics
const P = Pan
proto = CSV.read(ARGS[1], DataFrame)
SCEN = P.ScenarioConfig(); db = DuckDB.DB(SCEN.fia_db; readonly=true); con = DuckDB.DBInterface.connect(db)
cns = String[]
for r in eachrow(proto)
  q = DataFrame(DuckDB.DBInterface.execute(con, "SELECT CAST(CN AS VARCHAR) cn FROM PLOT WHERE STATECD=$(r.st) AND UNITCD=$(r.u) AND COUNTYCD=$(r.c) AND PLOT=$(r.p) AND make_date(MEASYEAR::INT, MEASMON::INT, MEASDAY::INT) = DATE '$(r.d1)'"))
  push!(cns, nrow(q) == 1 ? String(q.cn[1]) : "")
end
ps = P.load_paramset(read(joinpath(pwd(), "results", "params_yaml", "r4_hbad3p_gafl.yaml"), String), SCEN)
ok = findall(!isempty, cns); println("plots with a PLT_CN: ", length(ok), " of ", nrow(proto))
plots = [(cn=cns[i], acres=1.0) for i in ok]
plan = Dict{String,Any}("cycles" => 1, "period" => 10, "actions" => Any[])
res, warn = P.run_scenario(ps, plots, plan)
isempty(warn) || println("warnings: ", join(unique(warn), " | "))
d = DataFrame(i=Int[], ba0_pan=Float64[], ba10_pan=Float64[], ba0_proto=Float64[], ba10_proto=Float64[])
for (j, i) in enumerate(ok)
  r = res[cns[i]]; m = r.metrics
  push!(d, (i, m[1].ba / P.SC_M2HA_TO_FT2AC, m[end].ba / P.SC_M2HA_TO_FT2AC, proto.ba0[i], proto.ba10[i]))
end
d.r0 = d.ba0_pan ./ d.ba0_proto; d.r10 = d.ba10_pan ./ d.ba10_proto
show(stdout, first(d, 8)); println()
@printf("BA year 0  Pan / prototype: median %.4f, range %.4f .. %.4f, exact (1e-6) %d of %d\n", median(d.r0), minimum(d.r0), maximum(d.r0), count(x -> abs(x - 1) < 1e-6, d.r0), nrow(d))
@printf("BA year 10 Pan / prototype: median %.4f, range %.4f .. %.4f, within 1 %%: %d of %d\n", median(d.r10), minimum(d.r10), maximum(d.r10), count(x -> abs(x - 1) < 0.01, d.r10), nrow(d))
