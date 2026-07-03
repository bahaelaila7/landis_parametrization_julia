# How many NON-release recruits remain once we (a) include direct ≥5" debuts (outside-microplot) and
# (b) exclude canopy-release (planting-after-harvest + suppressed) at the PLOT level — i.e. enough for flux?
using DataFrames, DuckDB, Printf
con = DBInterface.connect(DuckDB.DB("/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"))
df = DataFrame(DBInterface.execute(con, """
WITH base AS (
  SELECT statecd,unitcd,countycd,plot,subp,tree,measdate,
    UPPER(TRIM(species_symbol)) sym, spgrpcd, UPPER(TRIM(sftwd_hrdwd)) sh, epa_l3, land_use,
    dia, tpa, cut_event_type, subp_has_dstrb
  FROM tree_trajectories
  WHERE intro_type='real' AND epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f') AND dia>0
),
plotev AS ( SELECT statecd,unitcd,countycd,plot,measdate,
    max(CASE WHEN cut_event_type IS NOT NULL OR subp_has_dstrb THEN 1 ELSE 0 END) pcut,
    sum(0.005454154*dia*dia*tpa) ba FROM base GROUP BY 1,2,3,4,5),
pel AS ( SELECT *, lag(pcut) OVER w ppcut, lag(ba) OVER w pba FROM plotev
         WINDOW w AS (PARTITION BY statecd,unitcd,countycd,plot ORDER BY measdate)),
pf AS ( SELECT statecd,unitcd,countycd,plot, min(measdate) fmd FROM base GROUP BY 1,2,3,4),
tw AS ( SELECT *, lag(dia) OVER win pdia, lag(measdate) OVER win pmd FROM base
        WINDOW win AS (PARTITION BY statecd,unitcd,countycd,plot,subp,tree ORDER BY measdate))
SELECT tw.sym, tw.spgrpcd, tw.sh, tw.epa_l3, tw.land_use, tw.tpa,
  CASE WHEN tw.pdia<5 THEN 'through' ELSE 'direct' END kind,
  CASE WHEN COALESCE(pe.pcut,0)=1 OR COALESCE(pe.ppcut,0)=1 OR (pe.pba>0 AND pe.ba<0.75*pe.pba) THEN 1 ELSE 0 END plot_rel
FROM tw JOIN pel pe ON tw.statecd=pe.statecd AND tw.unitcd=pe.unitcd AND tw.countycd=pe.countycd AND tw.plot=pe.plot AND tw.measdate=pe.measdate
        JOIN pf ON tw.statecd=pf.statecd AND tw.unitcd=pf.unitcd AND tw.countycd=pf.countycd AND tw.plot=pf.plot
WHERE tw.dia>=5 AND (tw.pdia<5 OR tw.pdia IS NULL) AND (tw.pmd IS NOT NULL OR tw.measdate>pf.fmd)
"""))
EXACT = Set(["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS"]); GRPS = Set([41,43])
tier(s,g,h) = s in EXACT ? s : (g in GRPS ? "_GRP_$g" : "_" * (h in ("S","H") ? h : "H"))
df.eff = [tier(String(r.sym), Int(r.spgrpcd), ismissing(r.sh) ? "H" : String(r.sh)) for r in eachrow(df)]
println("TOTAL recruits (through+direct): ", nrow(df), "  through=", sum(df.kind .== "through"), "  direct=", sum(df.kind .== "direct"))
println("plot-release flagged=", sum(df.plot_rel), "  NON-release=", sum(df.plot_rel .== 0))
nr = df[df.plot_rel .== 0, :]
println("\nNON-release recruits per species (total / through / direct):")
for s in sort(unique(df.eff))
  t = sum((nr.eff .== s) .& (nr.kind .== "through")); d = sum((nr.eff .== s) .& (nr.kind .== "direct"))
  @printf("  %-10s %4d  (thr=%d dir=%d)\n", s, t + d, t, d)
end
println("\nNON-release recruits per stratum:")
for st in sort(unique(nr.epa_l3 .* " | " .* nr.land_use))
  @printf("  %-22s %d\n", st, sum((nr.epa_l3 .* " | " .* nr.land_use) .== st))
end
