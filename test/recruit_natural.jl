# Natural-stand establishment only (SEP is a natural-stand question). Compare the FL5 4-L4 natural sample
# vs the broader L3 (8.3.5 / 8.5.3) natural pool, and the natural-vs-artificial release fraction.
using DataFrames, DuckDB, Printf
con = DBInterface.connect(DuckDB.DB("/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"))
df = DataFrame(DBInterface.execute(con, """
WITH base AS (
  SELECT statecd,unitcd,countycd,plot,subp,tree,measdate,
    UPPER(TRIM(species_symbol)) sym, spgrpcd, UPPER(TRIM(sftwd_hrdwd)) sh, epa_l3, epa_l4, land_use,
    dia, tpa, cut_event_type, subp_has_dstrb
  FROM tree_trajectories
  WHERE intro_type='real' AND epa_l3 IN ('8.3.5','8.5.3') AND dia>0
),
plotev AS ( SELECT statecd,unitcd,countycd,plot,measdate,
    max(CASE WHEN cut_event_type IS NOT NULL OR subp_has_dstrb THEN 1 ELSE 0 END) pcut,
    sum(0.005454154*dia*dia*tpa) ba FROM base GROUP BY 1,2,3,4,5),
pel AS ( SELECT *, lag(pcut) OVER w ppcut, lag(ba) OVER w pba FROM plotev
         WINDOW w AS (PARTITION BY statecd,unitcd,countycd,plot ORDER BY measdate)),
pf AS ( SELECT statecd,unitcd,countycd,plot, min(measdate) fmd FROM base GROUP BY 1,2,3,4),
tw AS ( SELECT *, lag(dia) OVER win pdia, lag(measdate) OVER win pmd FROM base
        WINDOW win AS (PARTITION BY statecd,unitcd,countycd,plot,subp,tree ORDER BY measdate))
SELECT tw.sym, tw.spgrpcd, tw.sh, tw.epa_l3, tw.epa_l4, tw.land_use,
  CASE WHEN tw.epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f') THEN 1 ELSE 0 END in_fl5,
  CASE WHEN COALESCE(pe.pcut,0)=1 OR COALESCE(pe.ppcut,0)=1 OR (pe.pba>0 AND pe.ba<0.75*pe.pba) THEN 1 ELSE 0 END plot_rel
FROM tw JOIN pel pe ON tw.statecd=pe.statecd AND tw.unitcd=pe.unitcd AND tw.countycd=pe.countycd AND tw.plot=pe.plot AND tw.measdate=pe.measdate
        JOIN pf ON tw.statecd=pf.statecd AND tw.unitcd=pf.unitcd AND tw.countycd=pf.countycd AND tw.plot=pf.plot
WHERE tw.dia>=5 AND (tw.pdia<5 OR tw.pdia IS NULL) AND (tw.pmd IS NOT NULL OR tw.measdate>pf.fmd)
"""))
EXACT = Set(["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS"]); GRPS = Set([41,43])
tier(s,g,h) = s in EXACT ? s : (g in GRPS ? "_GRP_$g" : "_" * (h in ("S","H") ? h : "H"))
df.eff = [tier(String(r.sym), Int(r.spgrpcd), ismissing(r.sh) ? "H" : String(r.sh)) for r in eachrow(df)]

# release fraction natural vs artificial (within the L3 region)
for lu in ["artificial","natural"]
  s = df[df.land_use .== lu, :]
  @printf("%-11s recruits=%d  release-flagged=%d (%.0f%%)  NON-release=%d\n", lu, nrow(s), sum(s.plot_rel), 100*sum(s.plot_rel)/max(nrow(s),1), sum(s.plot_rel .== 0))
end
nat = df[(df.land_use .== "natural") .& (df.plot_rel .== 0), :]
println("\nNATURAL non-release recruits per species — FL5-4L4 vs broader L3 pool:")
@printf("  %-10s %8s %8s\n", "species", "FL5", "L3pool")
for s in sort(unique(df.eff))
  @printf("  %-10s %8d %8d\n", s, sum((nat.eff .== s) .& (nat.in_fl5 .== 1)), sum(nat.eff .== s))
end
@printf("\n  TOTAL natural non-release:  FL5=%d   L3pool=%d\n", sum(nat.in_fl5 .== 1), nrow(nat))
