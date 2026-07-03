# SHADE_TOL (per-species shade-tolerance class 1–5) estimated from FIA crown class. A shade-tolerant
# species survives in the understory; FIA CCLCD encodes the crown position: 1=open-grown, 2=dominant,
# 3=codominant, 4=intermediate, 5=overtopped. Understory = CCLCD ∈ {4,5}. Per tiered species (the 14),
# understory_frac = live overtopped/intermediate stems / all live stems.
# NATURAL STANDS ONLY (land_use='natural'): in plantations the understory can be pruned/thinned, so crown
# class there does NOT reflect true tolerance. CCLCD is in curated_trees but land_use is in
# tree_trajectories, so we join them on the tree+measdate keys and filter to natural L3 8.3.5/8.5.3.
# Class = rank of understory_frac into 5 quantile bins (1 = most intolerant … 5 = most tolerant).
#   Run: ./julia_gdal.sh --project=. test/shadetol_from_data.jl
using DataFrames, DuckDB, Printf, CSV, Statistics
con = DBInterface.connect(DuckDB.DB("/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"))

raw = DataFrame(DBInterface.execute(con, """
  SELECT UPPER(TRIM(tt.species_symbol)) sym, tt.spgrpcd, UPPER(TRIM(tt.sftwd_hrdwd)) sh,
    count(*) n_live,
    count(*) FILTER (WHERE ct.CCLCD > 3) n_shaded,   -- CCLCD 4,5 = intermediate + overtopped (below-canopy)
    count(*) FILTER (WHERE ct.CCLCD = 5)  n_over
  FROM curated_trees ct
  JOIN tree_trajectories tt
    ON tt.statecd=ct.STATECD AND tt.unitcd=ct.UNITCD AND tt.countycd=ct.COUNTYCD
   AND tt.plot=ct.PLOT AND tt.subp=ct.SUBP AND tt.tree=ct.TREE AND tt.measdate=ct.MEASDATE
  WHERE ct.STATUSCD = 1 AND ct.CCLCD BETWEEN 1 AND 5
    AND tt.land_use = 'natural' AND tt.epa_l3 IN ('8.3.5','8.5.3') AND tt.intro_type = 'real'
  GROUP BY 1,2,3
"""))
println("raw (species×spgrp) rows: ", nrow(raw), "  total live stems: ", sum(raw.n_live))

# tier to the analysis 14 species (same mapping as establishment_from_data.jl)
EXACT = Set(["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS"]); GRPS = Set([41,43])
tier(sym, grp, sh) = sym in EXACT ? sym : (grp in GRPS ? "_GRP_$grp" : "_" * (sh in ("S","H") ? sh : "H"))
raw.eff = [tier(String(r.sym), Int(r.spgrpcd), ismissing(r.sh) ? "H" : String(r.sh)) for r in eachrow(raw)]

agg = combine(groupby(raw, :eff), :n_live => sum => :n_live, :n_shaded => sum => :n_shaded, :n_over => sum => :n_over)
agg.shaded_frac = agg.n_shaded ./ agg.n_live       # CCLCD>2 (not open-grown/dominant) = the shade-tolerance signal
agg.overtopped_frac = agg.n_over ./ agg.n_live
sort!(agg, :shaded_frac)

# classify into 1–5 by VALUE quintile (20/40/60/80th percentile thresholds of the shaded_frac spread)
qs = Statistics.quantile(agg.shaded_frac, [0.2, 0.4, 0.6, 0.8])
agg.shade_class = [1 + count(<(f), qs) for f in agg.shaded_frac]

println("\n=== SHADE_TOL from FIA crown class (CCLCD>3, ie intermediate+overtopped), NATURAL L3 8.3.5/8.5.3, live stems ===")
println("value-quintile thresholds (CCLCD>3 %): ", join((x->@sprintf("%.1f", 100x)).(qs), "  "))
println(rpad("species", 10), rpad("n_live", 9), rpad("CCLCD>3 %", 12), rpad("overtopped%", 13), "shade_class")
for r in eachrow(agg)
  println(rpad(r.eff, 10), rpad(string(r.n_live), 9),
    rpad(@sprintf("%.1f", 100r.shaded_frac), 12), rpad(@sprintf("%.1f", 100r.overtopped_frac), 13), r.shade_class)
end
mkpath("runs"); CSV.write("runs/shadetol_from_data.csv", select(agg, :eff, :n_live, :shaded_frac, :overtopped_frac, :shade_class))
println("\nwrote runs/shadetol_from_data.csv")
