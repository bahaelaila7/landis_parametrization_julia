# ≥5" ingrowth recruitment per tiered-species × eco×lu, comparing canopy-release exclusions:
#   none | excl immediate-interval release | excl ever-since-sapling release | excl cond_balive drop>25%
# and two SEP forms: per-plot-interval P(recruit, annualized) and flux (stems·ac⁻¹·yr⁻¹). "Does it matter?"
#   Run: ./julia_gdal.sh --project=. test/recruitment_compare.jl
using DataFrames, DuckDB, Printf, Statistics
const FIADB = "/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"
con = DBInterface.connect(DuckDB.DB(FIADB))

# recruit-level rows with release flags (a stem crossing into ≥5")
rec = DataFrame(DBInterface.execute(con, """
WITH base AS (
  SELECT statecd,unitcd,countycd,plot,subp,tree,measdate,
    UPPER(TRIM(species_symbol)) sym, spgrpcd, UPPER(TRIM(sftwd_hrdwd)) sh, epa_l3, land_use,
    dia, tpa, cut_event_type, subp_has_dstrb
  FROM tree_trajectories
  WHERE intro_type='real' AND epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f') AND dia>0
),
plotba AS (   -- plot live BA (sq ft/ac) per measurement from the trees, with prior-measurement BA
  SELECT statecd,unitcd,countycd,plot,measdate,
    sum(0.005454154*dia*dia*tpa) ba,
    lag(sum(0.005454154*dia*dia*tpa)) OVER (PARTITION BY statecd,unitcd,countycd,plot ORDER BY measdate) pba
  FROM base GROUP BY 1,2,3,4,5
),
w AS (
  SELECT *,
    lag(dia)      OVER win pdia,
    lag(measdate) OVER win pmd,
    CASE WHEN lag(cut_event_type) OVER win IS NOT NULL OR coalesce(lag(subp_has_dstrb) OVER win,false) THEN 1 ELSE 0 END rel_imm,
    coalesce(max(CASE WHEN cut_event_type IS NOT NULL OR subp_has_dstrb THEN 1 ELSE 0 END) OVER wp, 0) rel_ever
  FROM base
  WINDOW win AS (PARTITION BY statecd,unitcd,countycd,plot,subp,tree ORDER BY measdate),
         wp  AS (PARTITION BY statecd,unitcd,countycd,plot,subp,tree ORDER BY measdate ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT w.sym, w.spgrpcd, w.sh, w.epa_l3, w.land_use, w.tpa, datediff('year',w.pmd,w.measdate) intv,
       w.rel_imm, w.rel_ever,
       CASE WHEN pb.pba>0 AND pb.ba < 0.75*pb.pba THEN 1 ELSE 0 END rel_ba,
       CASE WHEN w.pdia IS NULL THEN 1 ELSE 0 END direct_debut
FROM w JOIN plotba pb ON w.statecd=pb.statecd AND w.unitcd=pb.unitcd AND w.countycd=pb.countycd AND w.plot=pb.plot AND w.measdate=pb.measdate
WHERE w.dia>=5 AND (w.pdia IS NULL OR w.pdia<5) AND w.pmd IS NOT NULL
"""))
# exposure: plot-intervals (consecutive measurements) per eco×lu, with interval length
exp = DataFrame(DBInterface.execute(con, """
WITH pm AS (
  SELECT DISTINCT statecd,unitcd,countycd,plot, epa_l3, land_use, measdate FROM tree_trajectories
  WHERE epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f')
)
SELECT epa_l3, land_use, count(*) n_intervals, sum(datediff('year', pmd, measdate)) plot_years
FROM (SELECT epa_l3, land_use, measdate, lag(measdate) OVER (PARTITION BY statecd,unitcd,countycd,plot ORDER BY measdate) pmd FROM pm)
WHERE pmd IS NOT NULL GROUP BY 1,2
"""))
exp.stratum = String.(exp.epa_l3) .* " | " .* String.(exp.land_use)
expd = Dict(r.stratum => (r.n_intervals, r.plot_years) for r in eachrow(exp))

EXACT = Set(["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS"]); GRPS = Set([41,43])
tier(s,g,h) = s in EXACT ? s : (g in GRPS ? "_GRP_$g" : "_" * (h in ("S","H") ? h : "H"))
rec.eff = [tier(String(r.sym), Int(r.spgrpcd), ismissing(r.sh) ? "H" : String(r.sh)) for r in eachrow(rec)]
rec.stratum = String.(rec.epa_l3) .* " | " .* String.(rec.land_use)

variants = ["all" => (r->true), "excl_imm" => (r->r.rel_imm==0), "excl_ever" => (r->r.rel_ever==0), "excl_ba" => (r->r.rel_ba==0)]
println("recruits total: ", nrow(rec), "  direct-debut (no prior sapling): ", sum(rec.direct_debut),
        "  flagged: imm=", sum(rec.rel_imm), " ever=", sum(rec.rel_ever), " ba=", sum(rec.rel_ba))

# pooled per species: flux (TPA/yr) and per-interval recruit count, under each variant
tot_py = sum(v[2] for v in values(expd)); tot_ni = sum(v[1] for v in values(expd))
println("\n=== pooled per species: P(recruit/plot-interval, annualized) under each release exclusion ===")
println(rpad("species",12), join([rpad(v[1],11) for v in variants]))
for s in sort(unique(rec.eff))
  cells = String[]
  for (name,f) in variants
    sub = rec[(rec.eff .== s) .& f.(eachrow(rec)), :]
    # P(recruit per plot-interval) ≈ #recruit-events / #plot-intervals ; annualize by mean interval
    p_int = nrow(sub)/max(tot_ni,1); mi = isempty(sub.intv) ? 5.0 : mean(skipmissing(sub.intv))
    push!(cells, @sprintf("%.4f", 1-(1-min(p_int,0.999))^(1/max(mi,1))))
  end
  println(rpad(s,12), join([rpad(c,11) for c in cells]))
end
println("\n(pooled plot-intervals=", tot_ni, ", plot-years=", round(Int,tot_py), ")")
using Statistics
