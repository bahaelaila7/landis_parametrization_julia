# Slim, self-contained DuckDB for a multi-ecoregion REGION run (Julia port of FIA_DATA_PREP/build_slim_db.py,
# + state filter + PLOT coordinates). Region = state ∩ eco: plots whose statecd ∈ --states AND eco ∈ --ecos.
# All sources opened READ-ONLY; refuses to overwrite.
#   ./julia_gdal.sh --project=. tools/build_slim_db_region.jl --out runs/slim_r4.duckdb \
#       --states 1,12,13,21,28,37,45,47 --ecos 8.3.5,8.5.3,8.3.4,... [--eco-field epa_l3] [--fiadb ...] [--simb ...]
import DuckDB, DataFrames
args = Dict{String,String}()
let i = 1
  while i <= length(ARGS); args[ARGS[i][3:end]] = ARGS[i+1]; i += 2; end
end
out    = args["out"]
ef     = get(args, "eco-field", "epa_l3")
fiadb  = get(args, "fiadb", "/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb")
simb   = get(args, "simb", "/workspace/FIA/FIASQLITE2PGSQL/cohorts_simB.duckdb")
ecos   = String.(strip.(split(args["ecos"], ",")))
states = parse.(Int, split(args["states"], ","))
isfile(out) && error("refusing to overwrite existing $out — rm it first")
ecol = join(["'" * replace(e, "'" => "''") * "'" for e in ecos], ","); stl = join(states, ",")
filt = "$ef IN ($ecol) AND statecd IN ($stl)"
key = "statecd, unitcd, countycd, plot"

con = DuckDB.DBInterface.connect(DuckDB.DB, out)
q(s) = DuckDB.DBInterface.execute(con, s)
q("ATTACH '$fiadb' AS fia (READ_ONLY)")
have_simb = isfile(simb); have_simb && q("ATTACH '$simb' AS simb (READ_ONLY)")
exists(alias, tab) = (alias == "simb" && !have_simb) ? false :
  first(DataFrames.DataFrame(q("SELECT count(*) n FROM information_schema.tables WHERE table_catalog='$alias' AND lower(table_name)=lower('$tab')")).n) > 0

parts = String[]
exists("fia", "curated_cohorts_landis_stdorg") && push!(parts, "SELECT DISTINCT $key FROM fia.curated_cohorts_landis_stdorg WHERE $filt")
exists("simb", "curated_cohorts_landis_simB")  && push!(parts, "SELECT DISTINCT $key FROM simb.curated_cohorts_landis_simB WHERE $filt")
q("CREATE TEMP TABLE _plots AS " * join(parts, " UNION "))
println("region states=$states ecos=$(length(ecos)): ", first(DataFrames.DataFrame(q("SELECT count(*) n FROM _plots")).n), " plots")
jon = join(["t.$k=p.$k" for k in split(key, ", ")], " AND ")
# (dest, alias, src, mode): eco = WHERE region filter; plots = join plot universe; all = whole table
for (dst, alias, tab, mode) in [
    ("curated_cohorts_landis_stdorg", "fia", "curated_cohorts_landis_stdorg", "eco"),
    ("curated_cohorts_landis_simB", "simb", "curated_cohorts_landis_simB", "eco"),
    ("curated_trees", "fia", "curated_trees", "eco"),
    ("COND", "fia", "COND", "plots"),
    ("PLOT", "fia", "PLOT", "plots"),                 # LAT/LON/ELEV for filter_extent + PnET climate/soil lookup
    ("cohort_disturbance", "simb", "cohort_disturbance", "plots"),
    ("REF_SPECIES", "fia", "REF_SPECIES", "all"),
    ("species_longevity_ref", "fia", "species_longevity_ref", "all")]
  exists(alias, tab) || (println("  SKIP $dst ($alias.$tab absent)"); continue)
  sel = mode == "eco" ? "SELECT * FROM $alias.$tab WHERE $filt" :
        mode == "plots" ? "SELECT t.* FROM $alias.$tab t JOIN _plots p ON $jon" : "SELECT * FROM $alias.$tab"
  q("CREATE TABLE $dst AS $sel")
  println(rpad("  $dst", 34), first(DataFrames.DataFrame(q("SELECT count(*) n FROM $dst")).n), " rows")
end
q("DROP TABLE _plots"); DuckDB.DBInterface.close!(con)
println("wrote $out  ($(round(filesize(out)/2^20, digits=1)) MiB)")
