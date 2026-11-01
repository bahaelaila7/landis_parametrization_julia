# Region inventory: per L3 → remeasured plots, 4-cell site-class split, top dominant species (AGB at last visit),
# share of commercial-pine-dominated plots. Drives siteclass_l3s / param_pooling / train_cap choices.
#   ./julia_gdal.sh --project=. tools/region_inventory.jl data_r4/fia_truth_r4.duckdb [eco_field]
import DuckDB, DataFrames, CSV
db = ARGS[1]; ef = length(ARGS) > 1 ? ARGS[2] : "epa_l3"
con = DuckDB.DBInterface.connect(DuckDB.DB, ":memory:")
q(s) = DataFrames.DataFrame(DuckDB.DBInterface.execute(con, s))
q("ATTACH '$db' AS src (READ_ONLY)")
COMMERCIAL = "('PITA','PIEL','PIPA2','PIEC2')"
df = q("""
WITH c AS (SELECT * FROM src.curated_cohorts_landis_stdorg WHERE plot_meas_num > 1),
last AS (SELECT statecd,unitcd,countycd,plot, max(measdate) md FROM c GROUP BY ALL),
sp AS (SELECT c.statecd,c.unitcd,c.countycd,c.plot, any_value(c.$ef) eco, c.species_symbol sym, sum(c.agb) a
       FROM c JOIN last l USING (statecd,unitcd,countycd,plot) WHERE c.measdate = l.md GROUP BY c.statecd,c.unitcd,c.countycd,c.plot,c.species_symbol),
dom AS (SELECT statecd,unitcd,countycd,plot, eco, arg_max(sym, a) dom FROM sp GROUP BY ALL),
cr AS (SELECT STATECD s,UNITCD u,COUNTYCD cc,PLOT p, SITECLCD sc, SUM(CONDPROP_UNADJ) w FROM src.COND WHERE COND_STATUS_CD=1 AND SITECLCD IS NOT NULL GROUP BY ALL),
rk AS (SELECT s,u,cc,p, CASE WHEN sc<=3 THEN 'A' WHEN sc=4 THEN 'B' WHEN sc=5 THEN 'C' ELSE 'D' END cell,
       ROW_NUMBER() OVER (PARTITION BY s,u,cc,p ORDER BY w DESC, sc) rn FROM cr)
SELECT d.*, coalesce(r.cell,'NA') cell FROM dom d LEFT JOIN rk r ON r.s=d.statecd AND r.u=d.unitcd AND r.cc=d.countycd AND r.p=d.plot AND r.rn=1
""")
using DataFrames: groupby, combine, nrow
rows = NamedTuple[]
for g in groupby(df, :eco)
  n = nrow(g); cells = Dict(c => count(==(c), g.cell) for c in ("A","B","C","D","NA"))
  dc = sort(combine(groupby(g, :dom), nrow => :n), :n, rev=true)
  pine = count(in(("PITA","PIEL","PIPA2","PIEC2")), g.dom) / n
  push!(rows, (eco=g.eco[1], plots=n, A=cells["A"], B=cells["B"], C=cells["C"], D=cells["D"], NA=cells["NA"],
    pine_dom=round(pine, digits=2), top_dom=join(["$(r.dom):$(r.n)" for r in first(eachrow(dc), 6)], " ")))
end
out = sort(DataFrames.DataFrame(rows), :plots, rev=true)
show(stdout, MIME("text/plain"), out; allrows=true, allcols=true, truncate=200); println()
println("TOTAL remeasured plots: ", sum(out.plots), "   commercial-pine-dominated: ", round(count(in(("PITA","PIEL","PIPA2","PIEC2")), df.dom)/nrow(df), digits=3))
CSV.write(replace(db, ".duckdb" => "_inventory.csv"), out)
