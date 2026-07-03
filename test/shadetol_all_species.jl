# SHADE_TOL for ALL species, everywhere (natural stands only), from FIA crown class. Per species_symbol:
# shaded_frac = fraction of live stems with CCLCD > 3 (intermediate + overtopped = grew up below canopy),
# NATURAL land_use only (plantation understory is managed, so excluded). Classify 1–5 by value-quintile of
# shaded_frac across species with ≥ MINN natural live stems (small samples dropped as noisy). Reusable table.
#   Run: ./julia_gdal.sh --project=. test/shadetol_all_species.jl [MINN=200]
using DataFrames, DuckDB, Printf, CSV, Statistics
con = DBInterface.connect(DuckDB.DB("/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"))
MINN = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200

raw = DataFrame(DBInterface.execute(con, """
  SELECT UPPER(TRIM(tt.species_symbol)) sym, any_value(tt.spgrpcd) spgrpcd,
    count(*) n_live,
    count(*) FILTER (WHERE ct.CCLCD > 3) n_shaded,
    count(*) FILTER (WHERE ct.CCLCD = 5) n_over
  FROM curated_trees ct
  JOIN tree_trajectories tt
    ON tt.statecd=ct.STATECD AND tt.unitcd=ct.UNITCD AND tt.countycd=ct.COUNTYCD
   AND tt.plot=ct.PLOT AND tt.subp=ct.SUBP AND tt.tree=ct.TREE AND tt.measdate=ct.MEASDATE
  WHERE ct.STATUSCD=1 AND ct.CCLCD BETWEEN 1 AND 5
    AND NOT EXISTS (SELECT 1 FROM COND co WHERE co.statecd=tt.statecd AND co.unitcd=tt.unitcd
                    AND co.countycd=tt.countycd AND co.plot=tt.plot AND co.stdorgcd=1)   -- STDORGCD 2-way natural (never planted at any remeasurement)
    AND tt.intro_type='real'
  GROUP BY 1
"""))
cov = DataFrame(DBInterface.execute(con, "SELECT count(DISTINCT epa_l3) neco, count(DISTINCT statecd) nstate FROM tree_trajectories tt WHERE NOT EXISTS (SELECT 1 FROM COND co WHERE co.statecd=tt.statecd AND co.unitcd=tt.unitcd AND co.countycd=tt.countycd AND co.plot=tt.plot AND co.stdorgcd=1)"))
println("coverage: $(cov.nstate[1]) states, $(cov.neco[1]) L3 ecoregions (natural); species w/ any CCLCD stems: $(nrow(raw)); stems: $(sum(raw.n_live))")

ref = DataFrame(DBInterface.execute(con, "SELECT UPPER(TRIM(SPECIES_SYMBOL)) sym, SPCD, COMMON_NAME FROM REF_SPECIES"))
raw = leftjoin(raw, unique(ref, :sym), on=:sym)
raw.shaded_frac = raw.n_shaded ./ raw.n_live
raw.overtopped_frac = raw.n_over ./ raw.n_live
q = sort(raw[raw.n_live .>= MINN, :], :shaded_frac)
qs = Statistics.quantile(q.shaded_frac, [0.2, 0.4, 0.6, 0.8])
q.shade_class = [1 + count(<(f), qs) for f in q.shaded_frac]
println("classified $(nrow(q)) species (≥ $MINN natural live stems); value-quintile thresholds (CCLCD>3 %): ",
  join((x -> @sprintf("%.1f", 100x)).(qs), "  "))

mkpath("runs")
CSV.write("runs/shadetol_all_species.csv", select(q, :sym, :SPCD, :COMMON_NAME, :spgrpcd, :n_live, :shaded_frac, :overtopped_frac, :shade_class))
println("wrote runs/shadetol_all_species.csv  ($(nrow(q)) species)\n")
show(combine(groupby(sort(q, :shade_class), :shade_class), nrow => :n_species, :shaded_frac => (x -> @sprintf("%.0f–%.0f", 100minimum(x), 100maximum(x))) => :range_pct); allrows=true)
println("\n\n--- 12 most INTOLERANT ---")
for r in eachrow(first(q, 12)); println(rpad(r.sym, 8), rpad(coalesce(r.COMMON_NAME, ""), 30), rpad(string(r.n_live), 8), @sprintf("%.1f", 100r.shaded_frac), "%  class ", r.shade_class); end
println("--- 12 most TOLERANT ---")
for r in eachrow(last(q, 12)); println(rpad(r.sym, 8), rpad(coalesce(r.COMMON_NAME, ""), 30), rpad(string(r.n_live), 8), @sprintf("%.1f", 100r.shaded_frac), "%  class ", r.shade_class); end
