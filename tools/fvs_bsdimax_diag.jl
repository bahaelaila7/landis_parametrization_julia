# Diagnostic: is BSDIMAX actually 0? Runs a few stands (NOAUTOES) with a COMPUTE addfile that records,
# every cycle, BSDIMAX / ASDIMAX / current stand-SDI [SpMcDBH(11,ALL)] / gate (BSDIMAX*0.75) / seed source
# [SpMcDBH(1,ALL,dbh>=5,ht>=10)] into the FVS_Compute table, then prints them.
#   Run: ./julia_gdal.sh --project=. tools/fvs_bsdimax_diag.jl [n_stands=6]
using DataFrames, DuckDB, Printf, Dates
include(joinpath(@__DIR__, "..", "src", "FVS.jl")); using .FVS
const FIADB  = "/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"
const BINDIR = "/workspace/FVStest"
const LDPATH = "/home/node/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia"
const HORIZONS = [25, 50, 75, 100]
const DIAG_KCP = joinpath(@__DIR__, "regimpute", "diag_bsdimax.kcp")
const NSTANDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 6

con = DuckDB.connect(DuckDB.DB())
DuckDB.execute(con, "ATTACH '$(FIADB)' AS s (READ_ONLY);")
df = DataFrame(DBInterface.execute(con, """
  SELECT t.statecd,t.unitcd,t.countycd,t.plot, t.measdate, t.spcd, t.dia, t.ht, t.cr, t.tpa_unadj,
         t.estimated_age, t.site_slope, t.site_aspect, t.site_elev
  FROM s.curated_trees_fvs t
  WHERE t.epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f')
    AND t.statuscd=1 AND t.dia IS NOT NULL AND t.tpa_unadj>0
"""))
stands = FVS.StandSpec[]
for gp in groupby(df, [:statecd,:unitcd,:countycd,:plot])
  length(stands) >= NSTANDS && break
  fmd = minimum(gp.measdate); iv = year(fmd); init = gp[gp.measdate .== fmd, :]; ki = first(init)
  trees = FVS.TreeRec[(spcd=Int(r.spcd), dbh=Float64(r.dia), ht=ismissing(r.ht) ? 0.0 : Float64(r.ht),
    cr=ismissing(r.cr) ? 0.0 : Float64(r.cr), tpa=Float64(r.tpa_unadj), damage=(0,0,0,0,0,0),
    birth_age=ismissing(r.estimated_age) ? 0.0 : Float64(r.estimated_age)) for r in eachrow(init)]
  push!(stands, FVS.StandSpec(id=join((ki.statecd,ki.unitcd,ki.countycd,ki.plot),"_"), inv_year=iv,
    target_years=iv .+ HORIZONS,
    slope=ismissing(ki.site_slope) ? nothing : Float64(ki.site_slope),
    aspect=ismissing(ki.site_aspect) ? nothing : Float64(ki.site_aspect),
    elev_ft=ismissing(ki.site_elev) ? nothing : Float64(ki.site_elev), trees=trees))
end
println("diagnostic: $(length(stands)) stands")
dir = joinpath(@__DIR__, "..", "tmp", "fvs_bsdimax_diag")
keypath, _, dbpath = FVS.write_run(stands; dir=dir, fiavbc=true, estab=:noauto, ffe=false,
                                   regimpute=DIAG_KCP, compute_db=true)
ok, log = FVS.run_fvs(keypath; variant="sn", bindir=BINDIR, ld_library_path=LDPATH)
println("FVS ok=$ok  (db=$dbpath)")

c2 = DuckDB.connect(DuckDB.DB())
try DuckDB.execute(c2, "LOAD sqlite;") catch; try DuckDB.execute(c2, "INSTALL sqlite;"); DuckDB.execute(c2, "LOAD sqlite;") catch end end
DuckDB.execute(c2, "ATTACH '$(dbpath)' AS fvs (TYPE sqlite, READ_ONLY);")
tabs = DataFrame(DuckDB.execute(c2, "SELECT name FROM fvs.sqlite_master WHERE type='table' ORDER BY name"))
println("tables in FVSOut.db: ", join(tabs.name, ", "))
if "FVS_Compute" in tabs.name
  comp = DataFrame(DuckDB.execute(c2, "SELECT * FROM fvs.\"FVS_Compute\""))
  println("\n=== FVS_Compute (per stand × year) ===")
  show(comp, allrows=true, allcols=true); println()
else
  println("!! FVS_Compute table NOT created — the COMPUTE db keyword didn't take.")
end
