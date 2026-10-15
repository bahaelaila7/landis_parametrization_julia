# Scenario-engine regression check (Step 3d): runs the r4 parameter sets of all three plugins through run_scenario (harvest + plant) on 4 FIA plots and prints a hash of the metrics; compare two trees.
#   ./julia_gdal.sh --project=. --threads=2 tools/scenario_plugin_check.jl
using Pan, DuckDB, DataFrames, SHA
const P = Pan
db = DuckDB.DB(P.SCEN_FIA_DB[]; readonly=true); con = DuckDB.DBInterface.connect(db)
cns = String.(DataFrame(DuckDB.DBInterface.execute(con, """SELECT p.CN cn FROM PLOT p JOIN (SELECT DISTINCT plt_cn k FROM $(P.SCEN_COHORT_TABLE[]) WHERE epa_l3 IN ('8.3.5','8.5.3')) c
  ON c.k = p.STATECD || '_' || p.UNITCD || '_' || p.COUNTYCD || '_' || p.PLOT WHERE p.INVYR >= 2015 ORDER BY p.CN LIMIT 4""")).cn)
plots = [(cn=c, acres=10.0 + i) for (i, c) in enumerate(cns)]
plan = Dict("cycles" => 3, "period" => 5, "actions" => Any[
  Dict("kind" => "harvest", "cycle" => 1, "selection" => Dict("species" => "all"), "removal" => Dict("type" => "percent_biomass", "value" => 30.0), "stands" => Dict("rank" => "all", "area_pct" => 100)),
  Dict("kind" => "plant", "cycle" => 2, "species" => "PITA,QUAL", "tpa" => 200.0, "survival" => 80.0)])
for f in ("r4_biomass_v1", "r4_hbad_v1", "r4_pnet_v1")
  ps = P.load_paramset(read(joinpath("results", "params_yaml", f * ".yaml"), String))
  res, warns = P.run_scenario(ps, plots, plan; seed=3)
  io = IOBuffer()
  for cn in cns
    r = res[cn]; println(io, cn, " ", r.status)
    for m in r.metrics; println(io, m); end
  end
  println(f, " ", bytes2hex(sha256(take!(io))), "  plots=", length(res), " status=", join([res[c].status for c in cns], ","))
end
