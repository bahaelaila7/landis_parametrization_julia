# Natural-stand Species Establishment Probability from FIA, per species × L3 ecoregion (the stratum).
# Universe: all natural stands in L3 8.3.5 / 8.5.3. Recruit = real stem crossing into ≥5" (throughgrowth
# OR direct debut), canopy-release EXCLUDED at plot level (planting-after-harvest + suppressed + BA-drop).
#   SEP_annual = 1-(1-p_interval)^(1/mean_intv),  p_interval = #(plot-intervals w/ recruit of s)/#(natural plot-intervals)
#   flux       = Σ TPA(recruits) / (natural plot-interval-years)        [stems·ac⁻¹·yr⁻¹]
#   Run: ./julia_gdal.sh --project=. test/establishment_natural_sep.jl
using DataFrames, DuckDB, Printf, CSV
con = DBInterface.connect(DuckDB.DB("/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"))

# --- non-release ≥5" recruits in natural L3 stands (one row per debut) ---
rec = DataFrame(DBInterface.execute(con, """
WITH base AS (
  SELECT statecd,unitcd,countycd,plot,subp,tree,measdate,
    UPPER(TRIM(species_symbol)) sym, spgrpcd, UPPER(TRIM(sftwd_hrdwd)) sh, epa_l3, dia, tpa, cut_event_type, subp_has_dstrb
  FROM tree_trajectories
  WHERE intro_type='real' AND epa_l3 IN ('8.3.5','8.5.3') AND land_use='natural' AND dia>0
),
plotev AS ( SELECT statecd,unitcd,countycd,plot,measdate,
    max(CASE WHEN cut_event_type IS NOT NULL OR subp_has_dstrb THEN 1 ELSE 0 END) pcut,
    sum(0.005454154*dia*dia*tpa) ba FROM base GROUP BY 1,2,3,4,5),
pel AS ( SELECT *, lag(pcut) OVER w ppcut, lag(ba) OVER w pba FROM plotev
         WINDOW w AS (PARTITION BY statecd,unitcd,countycd,plot ORDER BY measdate)),
pf AS ( SELECT statecd,unitcd,countycd,plot, min(measdate) fmd FROM base GROUP BY 1,2,3,4),
tw AS ( SELECT *, lag(dia) OVER win pdia, lag(measdate) OVER win pmd FROM base
        WINDOW win AS (PARTITION BY statecd,unitcd,countycd,plot,subp,tree ORDER BY measdate))
SELECT tw.epa_l3, tw.sym, tw.spgrpcd, tw.sh, tw.tpa, datediff('year',tw.pmd,tw.measdate) intv,
       tw.statecd, tw.unitcd, tw.countycd, tw.plot, tw.measdate
FROM tw JOIN pel pe ON tw.statecd=pe.statecd AND tw.unitcd=pe.unitcd AND tw.countycd=pe.countycd AND tw.plot=pe.plot AND tw.measdate=pe.measdate
        JOIN pf ON tw.statecd=pf.statecd AND tw.unitcd=pf.unitcd AND tw.countycd=pf.countycd AND tw.plot=pf.plot
WHERE tw.dia>=5 AND (tw.pdia<5 OR tw.pdia IS NULL) AND (tw.pmd IS NOT NULL OR tw.measdate>pf.fmd)
  AND NOT (COALESCE(pe.pcut,0)=1 OR COALESCE(pe.ppcut,0)=1 OR (pe.pba>0 AND pe.ba<0.75*pe.pba))
"""))
# --- exposure: natural plot-intervals per L3 (denominator) ---
exp = DataFrame(DBInterface.execute(con, """
WITH pm AS ( SELECT DISTINCT statecd,unitcd,countycd,plot, epa_l3, measdate
             FROM tree_trajectories WHERE epa_l3 IN ('8.3.5','8.5.3') AND land_use='natural' )
SELECT epa_l3, count(*) n_int, sum(yr) plot_years, avg(yr) mean_int FROM (
  SELECT epa_l3, datediff('year', lag(measdate) OVER (PARTITION BY statecd,unitcd,countycd,plot ORDER BY measdate), measdate) yr FROM pm
) WHERE yr IS NOT NULL AND yr>0 GROUP BY 1
"""))
expd = Dict(String(r.epa_l3) => (r.n_int, Float64(r.plot_years), Float64(r.mean_int)) for r in eachrow(exp))
println("natural plot-interval exposure: ", [(k, v[1], round(Int,v[2])) for (k,v) in expd])

EXACT = Set(["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS"]); GRPS = Set([41,43])
tier(s,g,h) = s in EXACT ? s : (g in GRPS ? "_GRP_$g" : "_" * (h in ("S","H") ? h : "H"))
rec.eff = [tier(String(r.sym), Int(r.spgrpcd), ismissing(r.sh) ? "H" : String(r.sh)) for r in eachrow(rec)]
rec.l3 = String.(rec.epa_l3)
rec.pmk = string.(rec.statecd,"_",rec.unitcd,"_",rec.countycd,"_",rec.plot,"_",rec.measdate)  # plot-interval key (by debut meas)

rows = NamedTuple[]
for l3 in sort(unique(rec.l3)), s in sort(unique(rec.eff))
  sub = rec[(rec.l3 .== l3) .& (rec.eff .== s), :]
  isempty(sub) && continue
  n_int, py, mi = expd[l3]
  n_rec_int = length(unique(sub.pmk))                 # plot-intervals with ≥1 recruit of s
  p_int = n_rec_int / max(n_int, 1)
  p_ann = 1 - (1 - min(p_int, 0.999))^(1 / max(mi, 1))
  flux = sum(Float64.(sub.tpa)) / max(py, 1)          # stems·ac⁻¹·yr⁻¹
  push!(rows, (l3=l3, species=s, n_recruits=nrow(sub), p_interval=round(p_int,digits=4),
               sep_annual=round(p_ann,digits=4), flux_tpa_yr=round(flux,digits=5)))
end
out = DataFrame(rows)
for l3 in sort(unique(out.l3))
  println("\n=== L3 ", l3, " natural — SEP_annual (P recruit/plot-yr), flux, n ===")
  for r in eachrow(sort(out[out.l3 .== l3, :], :sep_annual, rev=true))
    @printf("  %-10s SEP=%.4f  flux=%.5f  (n=%d, p_int=%.4f)\n", r.species, r.sep_annual, r.flux_tpa_yr, r.n_recruits, r.p_interval)
  end
end
mkpath("runs"); CSV.write("runs/establishment_natural_l3.csv", out)
println("\nwrote runs/establishment_natural_l3.csv  (apply to FL5 L4s: 8.3.5.65*←8.3.5, 8.5.3.75*←8.5.3; floor 0.2)")
