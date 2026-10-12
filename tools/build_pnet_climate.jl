# Monthly climate cache for the PnET plugin: PRISM (EPA L4 polygon means) → one dense Float32 array
# CLIM[var, month, year, l4] (+ ids/years/vars), so runs never touch the 1.4 GB PRISM DuckDB.
# PRISM_LT_EPA values are °C / mm / hPa (verified; the README's °F/inches is wrong) and stored as text → CAST.
#   ./julia_gdal.sh --project=. tools/build_pnet_climate.jl --l4-from <slim.duckdb> --out <cache.jld2> [--y0 1980 --y1 2025]
import DuckDB, DataFrames, JLD2
args = Dict{String,String}(); let i = 1; while i <= length(ARGS); args[ARGS[i][3:end]] = ARGS[i+1]; i += 2; end; end
y0 = parse(Int, get(args, "y0", "1980")); y1 = parse(Int, get(args, "y1", "2025"))
prism = get(args, "prism", "/workspace/PRISM_LT/PRISM_LT_EPA/prism_lt_epa.duckdb")
const VARS = ["tmin", "tmax", "tmean", "ppt", "vpdmin", "vpdmax", "tdmean"]
con = DuckDB.DBInterface.connect(DuckDB.DB, ":memory:")
q(s) = DataFrames.DataFrame(DuckDB.DBInterface.execute(con, s))
q("ATTACH '$prism' AS p (READ_ONLY)"); q("ATTACH '$(args["l4-from"])' AS s (READ_ONLY)")
ids = sort(String.(q("SELECT DISTINCT epa_l4 FROM s.curated_cohorts_landis_stdorg WHERE epa_l4 IS NOT NULL").epa_l4))
idl = join(["'$i'" for i in ids], ",")
cast(v) = "CAST(CAST($(v)_mean AS VARCHAR) AS DOUBLE) $v"
df = q("SELECT EPA_L4_ID id, YEAR y, MONTH m, $(join(cast.(VARS), ", ")) FROM p.EPA_L4_monthly WHERE EPA_L4_ID IN ($idl) AND YEAR BETWEEN $y0 AND $y1")
years = collect(y0:y1); ix = Dict(id => k for (k, id) in enumerate(ids))
CLIM = fill(NaN32, length(VARS), 12, length(years), length(ids))
for r in eachrow(df), (v, name) in enumerate(VARS)
  CLIM[v, r.m, r.y - y0 + 1, ix[r.id]] = Float32(r[Symbol(name)])
end
nmiss = count(isnan, CLIM)
println("ids=$(length(ids)) years=$(y0)-$(y1) missing=$nmiss (last year may be partial)")
# fill any trailing missing months (partial final year) with that id/month's climatology over the last 30 yr
for k in axes(CLIM, 4), m in 1:12, v in axes(CLIM, 1), y in axes(CLIM, 3)
  if isnan(CLIM[v, m, y, k])
    ref = filter(!isnan, CLIM[v, m, max(1, y - 30):y, k]); CLIM[v, m, y, k] = isempty(ref) ? NaN32 : Float32(sum(ref) / length(ref))
  end
end
@assert !any(isnan, CLIM) "unfillable gaps remain"
JLD2.jldsave(args["out"]; CLIM, ids, years, vars=VARS, source="PRISM_LT EPA L4 monthly means (°C, mm, hPa)")
println("wrote $(args["out"])  size=$(size(CLIM))  ", round(filesize(args["out"]) / 2^20, digits=2), " MiB")
