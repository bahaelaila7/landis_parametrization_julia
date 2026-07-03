# Data-derived establishment (≈ Species Establishment Probability) from FIA remeasurement.
# Anchor = SAPLING (1.0–4.9" DBH) survival past stem-exclusion: of live saplings at t, fraction alive at
# t+1 (harvest/disturbance CENSORED, only natural death counts), per tiered-species × eco×lu, annualized
# p_annual = p_interval^(1/interval). Also a coarse PROB_MORT = natural catastrophic cohort loss rate.
#   Run: ./julia_gdal.sh --project=. test/establishment_from_data.jl
using DataFrames, DuckDB, Printf, CSV
const FIADB = "/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"
const ECOS = ("8.3.5.65o","8.5.3.75g","8.5.3.75e","8.5.3.75f")
con = DBInterface.connect(DuckDB.DB(FIADB))

# per (species, eco×lu) sapling-interval outcomes via per-tree LEAD over measurements
sql = """
WITH o AS (
  SELECT species_symbol sym, spgrpcd, UPPER(TRIM(sftwd_hrdwd)) sh, epa_l3, land_use, dia, measdate,
    lead(measdate)   OVER w nmd,
    lead(is_terminal)OVER w nterm,
    lead(death_cause)OVER w ndc
  FROM tree_trajectories
  WHERE epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f') AND dia>0
  WINDOW w AS (PARTITION BY statecd,unitcd,countycd,plot,subp,tree ORDER BY measdate)
)
SELECT UPPER(TRIM(sym)) sym, spgrpcd, sh, epa_l3, land_use,
  count(*) FILTER (WHERE outcome='surv') surv,
  count(*) FILTER (WHERE outcome='died') died,
  avg(intv) FILTER (WHERE outcome<>'cens') mean_intv
FROM (
  SELECT sym, spgrpcd, sh, epa_l3, land_use, datediff('year', measdate, nmd) intv,
    CASE WHEN nterm AND ndc='natural' THEN 'died'
         WHEN nterm AND ndc IN ('harvest_explicit','harvest_inferred','disturbance') THEN 'cens'
         ELSE 'surv' END outcome
  FROM o WHERE dia>=1.0 AND dia<5.0 AND nmd IS NOT NULL
) GROUP BY 1,2,3,4,5
"""
raw = DataFrame(DBInterface.execute(con, sql))
println("rows (species×eco×lu sapling cells): ", nrow(raw))

# tier to the analysis 14 species
EXACT = Set(["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS"]); GRPS = Set([41,43])
tier(sym, grp, sh) = sym in EXACT ? sym : (grp in GRPS ? "_GRP_$grp" : "_" * (sh in ("S","H") ? sh : "H"))
raw.eff = [tier(String(r.sym), Int(r.spgrpcd), ismissing(r.sh) ? "H" : String(r.sh)) for r in eachrow(raw)]
raw.stratum = String.(raw.epa_l3) .* " | " .* String.(raw.land_use)

# aggregate survived/died over the tier, recompute p; weight mean interval by at-risk n
agg = combine(groupby(raw, [:stratum, :eff]),
  :surv => sum => :surv, :died => sum => :died,
  [:mean_intv, :surv, :died] => ((mi, s, d) -> sum(skipmissing(mi .* (s .+ d))) / max(sum(s .+ d), 1)) => :mean_intv)
agg.n = agg.surv .+ agg.died
agg.p_interval = agg.surv ./ max.(agg.n, 1)
agg.p_annual = agg.p_interval .^ (1 ./ max.(agg.mean_intv, 1e-6))

println("\n=== sapling annual survival (establishment proxy) — stratum | species : p_annual (n at-risk) ===")
sort!(agg, [:stratum, :eff])
for st in unique(agg.stratum)
  println("• ", st)
  for r in eachrow(agg[agg.stratum .== st, :])
    r.n >= 20 || continue
    @printf("    %-12s p_ann=%.3f  (p_%.0fyr=%.3f, n=%d)\n", r.eff, r.p_annual, r.mean_intv, r.p_interval, r.n)
  end
end
# also a pooled-over-eco fallback (for sparse cells)
pool = combine(groupby(raw, :eff), :surv => sum => :surv, :died => sum => :died,
  [:mean_intv, :surv, :died] => ((mi, s, d) -> sum(skipmissing(mi .* (s .+ d))) / max(sum(s .+ d), 1)) => :mi)
pool.n = pool.surv .+ pool.died; pool.p_annual = (pool.surv ./ max.(pool.n, 1)) .^ (1 ./ max.(pool.mi, 1e-6))
println("\n=== pooled-over-eco fallback ===")
for r in eachrow(sort(pool, :eff)); r.n >= 20 && @printf("    %-12s p_ann=%.3f  n=%d\n", r.eff, r.p_annual, r.n); end

mkpath("runs"); CSV_OUT = "runs/establishment_rates.csv"
CSV.write(CSV_OUT, agg)
println("\nwrote ", CSV_OUT)
