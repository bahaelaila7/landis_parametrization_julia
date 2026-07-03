# Natural-stand SEP conditioned on SEED PRESENCE (seedlings at interval start), per species × L3.
# SEP = P(non-release ≥5" recruit of s at interval end | s had seedlings at interval start).
# Seedling presence via SEEDLING joined through PLOT on the actual MEASUREMENT date (MEASYEAR/MEASMON) —
# NOT INVYR. Presence = TREECOUNT_CALC>0 (cap-/sentinel-robust). Seedling tiering pools to the 14 groups.
#   Run: ./julia_gdal.sh --project=. test/establishment_natural_sep_conditioned.jl
using DataFrames, DuckDB, Printf, CSV
con = DBInterface.connect(DuckDB.DB("/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"))
EXACT = Set(["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS"]); GRPS = Set([41,43])
tier(sym, grp, sh) = sym in EXACT ? sym : (grp in GRPS ? "_GRP_$grp" : "_" * (sh in ("S","H") ? sh : "H"))
# SPCD → symbol / softwood (for seedling tiering; SEEDLING carries SPCD+SPGRPCD, needs symbol+SH from REF)
ref = DataFrame(DBInterface.execute(con, "SELECT SPCD spcd, UPPER(TRIM(SPECIES_SYMBOL)) sym, UPPER(TRIM(SFTWD_HRDWD)) sh FROM REF_SPECIES"))
spcd_sym = Dict(Int(r.spcd) => String(r.sym) for r in eachrow(ref) if !ismissing(r.spcd))
spcd_sh  = Dict(Int(r.spcd) => (ismissing(r.sh) ? "H" : String(r.sh)) for r in eachrow(ref) if !ismissing(r.spcd))
tier_spcd(spcd, grp) = (sym = get(spcd_sym, spcd, ""); tier(sym, grp, get(spcd_sh, spcd, "H")))

# --- intervals: natural L3 plot-measurements bridged to PLOT.CN by MEASYEAR/MEASMON (the real PLT_CN) ---
iv = DataFrame(DBInterface.execute(con, """
WITH meas AS (
  SELECT DISTINCT t.statecd,t.unitcd,t.countycd,t.plot, t.epa_l3, t.measdate, CAST(p.CN AS VARCHAR) cn
  FROM tree_trajectories t
  JOIN PLOT p ON t.statecd=p.STATECD AND t.unitcd=p.UNITCD AND t.countycd=p.COUNTYCD AND t.plot=p.PLOT
              AND year(t.measdate)=p.MEASYEAR AND month(t.measdate)=p.MEASMON
  WHERE t.epa_l3 IN ('8.3.5','8.5.3') AND t.land_use='natural'
)
SELECT * FROM (
  SELECT epa_l3 l3, statecd,unitcd,countycd,plot, measdate m_end,
    lag(cn) OVER w cn_start, lag(measdate) OVER w m_start,
    datediff('year', lag(measdate) OVER w, measdate) yr
  FROM meas WINDOW w AS (PARTITION BY statecd,unitcd,countycd,plot ORDER BY measdate)
) WHERE m_start IS NOT NULL AND yr>0
"""))
println("natural L3 intervals (bridged): ", nrow(iv))
# --- seedling presence at interval start: PLT_CN(=cn_start) → tiered species set ---
seed = DataFrame(DBInterface.execute(con, """
  SELECT DISTINCT CAST(PLT_CN AS VARCHAR) cn, SPCD spcd, SPGRPCD grp FROM SEEDLING WHERE TREECOUNT_CALC>0
    AND CAST(PLT_CN AS VARCHAR) IN (SELECT DISTINCT cn_start FROM (
      WITH meas AS (SELECT DISTINCT t.statecd,t.unitcd,t.countycd,t.plot,t.measdate, CAST(p.CN AS VARCHAR) cn
        FROM tree_trajectories t JOIN PLOT p ON t.statecd=p.STATECD AND t.unitcd=p.UNITCD AND t.countycd=p.COUNTYCD AND t.plot=p.PLOT
          AND year(t.measdate)=p.MEASYEAR AND month(t.measdate)=p.MEASMON
        WHERE t.epa_l3 IN ('8.3.5','8.5.3') AND t.land_use='natural')
      SELECT lag(cn) OVER (PARTITION BY statecd,unitcd,countycd,plot ORDER BY measdate) cn_start FROM meas))
"""))
seed.eff = [tier_spcd(Int(r.spcd), Int(r.grp)) for r in eachrow(seed)]
seed_by_cn = Dict{String,Set{String}}()
for r in eachrow(seed); push!(get!(seed_by_cn, String(r.cn), Set{String}()), r.eff); end

# --- non-release ≥5" recruits at interval end (plot keys + measdate + species) ---
rec = DataFrame(DBInterface.execute(con, """
WITH base AS (
  SELECT statecd,unitcd,countycd,plot,subp,tree,measdate,
    UPPER(TRIM(species_symbol)) sym, spgrpcd, UPPER(TRIM(sftwd_hrdwd)) sh, dia, cut_event_type, subp_has_dstrb, tpa
  FROM tree_trajectories WHERE intro_type='real' AND epa_l3 IN ('8.3.5','8.5.3') AND land_use='natural' AND dia>0
),
plotev AS ( SELECT statecd,unitcd,countycd,plot,measdate,
    max(CASE WHEN cut_event_type IS NOT NULL OR subp_has_dstrb THEN 1 ELSE 0 END) pcut,
    sum(0.005454154*dia*dia*tpa) ba FROM base GROUP BY 1,2,3,4,5),
pel AS ( SELECT *, lag(pcut) OVER w ppcut, lag(ba) OVER w pba FROM plotev WINDOW w AS (PARTITION BY statecd,unitcd,countycd,plot ORDER BY measdate)),
pf AS ( SELECT statecd,unitcd,countycd,plot, min(measdate) fmd FROM base GROUP BY 1,2,3,4),
tw AS ( SELECT *, lag(dia) OVER win pdia, lag(measdate) OVER win pmd FROM base WINDOW win AS (PARTITION BY statecd,unitcd,countycd,plot,subp,tree ORDER BY measdate))
SELECT tw.statecd,tw.unitcd,tw.countycd,tw.plot, tw.measdate, UPPER(TRIM(tw.sym)) sym, tw.spgrpcd, tw.sh
FROM tw JOIN pel pe ON tw.statecd=pe.statecd AND tw.unitcd=pe.unitcd AND tw.countycd=pe.countycd AND tw.plot=pe.plot AND tw.measdate=pe.measdate
        JOIN pf ON tw.statecd=pf.statecd AND tw.unitcd=pf.unitcd AND tw.countycd=pf.countycd AND tw.plot=pf.plot
WHERE tw.dia>=5 AND (tw.pdia<5 OR tw.pdia IS NULL) AND (tw.pmd IS NOT NULL OR tw.measdate>pf.fmd)
  AND NOT (COALESCE(pe.pcut,0)=1 OR COALESCE(pe.ppcut,0)=1 OR (pe.pba>0 AND pe.ba<0.75*pe.pba))
"""))
rec.eff = [tier(String(r.sym), Int(r.spgrpcd), ismissing(r.sh) ? "H" : String(r.sh)) for r in eachrow(rec)]
pmk(s,u,c,p,m) = string(s,"_",u,"_",c,"_",p,"_",m)
rec_by_pm = Dict{String,Set{String}}()
for r in eachrow(rec); push!(get!(rec_by_pm, pmk(r.statecd,r.unitcd,r.countycd,r.plot,r.measdate), Set{String}()), r.eff); end

# --- assemble: for each interval, every seed-present species is a denominator trial; a recruit is a hit ---
denom = Dict{Tuple{String,String},Int}(); num = Dict{Tuple{String,String},Int}()
yrsum = Dict{Tuple{String,String},Float64}()
for row in eachrow(iv)
  elig = get(seed_by_cn, ismissing(row.cn_start) ? "" : String(row.cn_start), nothing); elig === nothing && continue
  hits = get(rec_by_pm, pmk(row.statecd,row.unitcd,row.countycd,row.plot,row.m_end), Set{String}())
  for e in elig
    k = (String(row.l3), e); denom[k] = get(denom,k,0)+1; yrsum[k]=get(yrsum,k,0.0)+row.yr
    e in hits && (num[k] = get(num,k,0)+1)
  end
end
rows = NamedTuple[]
for k in sort(collect(keys(denom)))
  d = denom[k]; d >= 10 || continue
  nn = get(num,k,0); mi = yrsum[k]/d
  p = nn/d; pa = 1-(1-min(p,0.999))^(1/max(mi,1))
  push!(rows, (l3=k[1], species=k[2], n_eligible=d, n_recruit=nn, p_cond=round(p,digits=4), sep_annual=round(pa,digits=4)))
end
out = DataFrame(rows)
for l3 in sort(unique(out.l3))
  println("\n=== L3 ", l3, " natural — SEP conditioned on seedling presence (sorted) ===")
  for r in eachrow(sort(out[out.l3 .== l3, :], :sep_annual, rev=true))
    @printf("  %-10s SEP=%.4f  (recruit %d/%d eligible, p_int=%.3f)\n", r.species, r.sep_annual, r.n_recruit, r.n_eligible, r.p_cond)
  end
end
mkpath("runs"); CSV.write("runs/establishment_natural_l3_conditioned.csv", out)
println("\nwrote runs/establishment_natural_l3_conditioned.csv")
