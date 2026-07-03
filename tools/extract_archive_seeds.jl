# Export the finished A-only MO-CMA-ES archive (front-1, strict-Pareto non-dominated set) as a
# sobol-candidates DB, so the stage-B run can seed start + each IPOP restart from a front-1 candidate
# (growth fixed at that candidate via FIX_GROWTH, establishment fit around it).
#   Run: ./julia_gdal.sh --project=. tools/extract_archive_seeds.jl
using Pan, JLD2, DuckDB, DataFrames, Serialization
const STATE = "runs/fl5_l4cover_mocmaes_Aonly_Sglobal_ipop_v2_outputs/search_state_latest.jld2"
const OUT   = "runs/fl5_l4cover_mocmaes_Aonly_Sglobal_ipop_v2_outputs/archive_seeds.duckdb"
state = JLD2.load_object(STATE)
arch = collect(state.archive)
println("A-only archive front-1 size: ", length(arch), "  best aggregate=", round(minimum(c.fx.aggregate for c in arch), sigdigits=5))
isfile(OUT) && rm(OUT)
db = DuckDB.DB(OUT); con = DuckDB.connect(db)
DuckDB.execute(con, "CREATE TABLE sobol_results (run_id VARCHAR, sobol_idx INTEGER, mean_loss DOUBLE, std_loss DOUBLE, median_loss DOUBLE, sumW DOUBLE, sumAGB DOUBLE, params_blob BLOB)")
for (i, c) in enumerate(arch)
  buf = IOBuffer(); Serialization.serialize(buf, c.x)
  agg = Float64(c.fx.aggregate)
  DuckDB.execute(con, "INSERT INTO sobol_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    ["archive", i, agg, 0.0, agg, 0.0, 0.0, take!(buf)])
end
close(db)
println("wrote ", OUT, " — ", length(arch), " front-1 seeds (load with sobol_candidates_db + sobol_top_frac:1.0)")
