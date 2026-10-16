# HBAD v2 data audit (GA+FL, read-only): what does FIA measure directly about cohorts?
#   ./julia_gdal.sh --project=. tools/hbad2/audit_trees.jl
using DuckDB, DataFrames, Statistics, Printf
con = DBInterface.connect(DuckDB.DB, ":memory:")
DBInterface.execute(con, "ATTACH '/workspace/landis_region4/data_r4/slim_r4.duckdb' AS s (READ_ONLY)")
q(sql) = DataFrame(DBInterface.execute(con, sql))
W = "t.STATECD IN (12,13)"
println("== status / TPA_UNADJ classes")
println(q("SELECT STATUSCD, round(TPA_UNADJ,3) tpa, DIA>=5 is_macro, count(*) n FROM s.curated_trees t WHERE $W GROUP BY ALL ORDER BY n DESC LIMIT 12"))
println("== height: HT vs ACTUALHT (live), share with HT, HT==ACTUALHT")
println(q("SELECT count(*) n, avg((HT>0)::INT) has_ht, avg((ACTUALHT>0)::INT) has_aht, avg((HT=ACTUALHT)::INT) ht_eq_aht FROM s.curated_trees t WHERE $W AND STATUSCD=1"))
println("== tree_fate, age_source")
println(q("SELECT tree_fate, count(*) n FROM s.curated_trees t WHERE $W GROUP BY ALL ORDER BY n DESC"))
println(q("SELECT age_source, count(*) n FROM s.curated_trees t WHERE $W AND STATUSCD=1 GROUP BY ALL ORDER BY n DESC"))
# linked remeasurements of the same tree
DBInterface.execute(con, """CREATE TEMP TABLE tv AS SELECT STATECD st, UNITCD u, COUNTYCD c, PLOT p, SUBP sb, TREE tr, MEASDATE md, STATUSCD s, DIA dia, HT ht, TPA_UNADJ tpa, estimated_age age, spcd_resolved spcd,
  row_number() over (partition by STATECD,UNITCD,COUNTYCD,PLOT,SUBP,TREE order by MEASDATE) k FROM s.curated_trees t WHERE $W""")
DBInterface.execute(con, """CREATE TEMP TABLE pr AS SELECT a.*, b.md md2, b.s s2, b.dia dia2, b.ht ht2, b.tpa tpa2, b.age age2, date_diff('day', a.md, b.md)/365.25 dt
  FROM tv a JOIN tv b ON a.st=b.st AND a.u=b.u AND a.c=b.c AND a.p=b.p AND a.sb=b.sb AND a.tr=b.tr AND b.k=a.k+1""")
println("== linked consecutive pairs by (status1 -> status2)")
println(q("SELECT s, s2, count(*) n, avg(dt) dt FROM pr GROUP BY ALL ORDER BY n DESC LIMIT 8"))
println("== live->live: TPA_UNADJ transition (sapling->tree) ")
println(q("SELECT round(tpa,2) tpa1, round(tpa2,2) tpa2, count(*) n, avg(dia) d1, avg(dia2) d2 FROM pr WHERE s=1 AND s2=1 GROUP BY ALL ORDER BY n DESC LIMIT 6"))
println("== live->live: age advance vs elapsed time; dia & ht increments")
println(q("SELECT count(*) n, avg(age2-age-dt) age_minus_dt_mean, quantile_cont(age2-age-dt,[0.05,0.5,0.95]) q, avg((dia2-dia)/dt) ddia_in_yr, avg((ht2-ht)/dt) dht_ft_yr, avg((ht2<ht)::INT) frac_ht_down, avg((dia2<dia)::INT) frac_dia_down FROM pr WHERE s=1 AND s2=1 AND ht>0 AND ht2>0"))
println("== first-appearance trees after the plot's first visit (ingrowth): size + age")
DBInterface.execute(con, "CREATE TEMP TABLE pv AS SELECT st,u,c,p, min(md) md0 FROM tv GROUP BY ALL")
println(q("SELECT (tv.dia>=5) is_macro, count(*) n, avg(dia) dia, quantile_cont(age,[0.1,0.5,0.9]) age_q FROM tv JOIN pv USING (st,u,c,p) WHERE tv.k=1 AND tv.md>pv.md0 AND tv.s=1 GROUP BY ALL"))
println("== live trees per plot-visit subplot-species-age cohort (cohort singletons)")
println(q("SELECT avg((n=1)::INT) frac_single, avg(n) mean_n FROM (SELECT count(*) n FROM tv WHERE s=1 GROUP BY st,u,c,p,sb,md,spcd,round(age))"))
println(q("SELECT avg((n=1)::INT) frac_single, avg(n) mean_n, quantile_cont(n,[0.5,0.9]) q FROM (SELECT count(*) n FROM tv WHERE s=1 AND dia>=5 GROUP BY st,u,c,p,md,spcd,floor(age/10))"))
