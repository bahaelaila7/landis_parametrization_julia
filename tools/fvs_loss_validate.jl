# Validate the tier-3+hinge+cell-norm loss replica: feed the OBSERVED curated cohorts as the "sim".
# Expect W = 0 exactly (identical age-CDFs) and total ≈ the softplus-floor baseline (tiny). Then a
# perturbation (sim AGB ×1.5, ages shifted) must make W and AGB jump — proving the metric responds.
#   Run: ./julia_gdal.sh --project=. tools/fvs_loss_validate.jl
using DataFrames, DuckDB, Printf
include(joinpath(@__DIR__, "fvs_loss.jl")); using .FVSLoss
const FIADB = "/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"
con = DBInterface.connect(DuckDB.DB(FIADB))
EXACT = Set(["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS"]); GRPS = Set([41,43])
tier(s,g,h) = s in EXACT ? s : (g in GRPS ? "_GRP_$g" : "_" * (h in ("S","H") ? h : "H"))

# plot-level cohorts (SUM agb over subplots = corrected scale, no /nsub), per (plot, measdate, species, age)
df = DataFrame(DBInterface.execute(con, """
  WITH base AS (
    SELECT statecd,unitcd,countycd,plot, epa_l3, land_use, measdate, age_calc,
           UPPER(TRIM(species_symbol)) sym, spgrpcd grp, UPPER(TRIM(sftwd_hrdwd)) sh, SUM(agb) agb
    FROM curated_cohorts_landis
    WHERE epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f') AND agb>0
    GROUP BY statecd,unitcd,countycd,plot,epa_l3,land_use,measdate,age_calc,
             UPPER(TRIM(species_symbol)),spgrpcd,UPPER(TRIM(sftwd_hrdwd)))
  SELECT *, CAST(round(datediff('day', min(measdate) OVER
           (PARTITION BY statecd,unitcd,countycd,plot), measdate)/365.25) AS INTEGER) yr
  FROM base
"""))
df.plotkey = string.(df.statecd,"_",df.unitcd,"_",df.countycd,"_",df.plot)
df.year = Int.(df.yr)
df.eff = [tier(String(r.sym), Int(r.grp), ismissing(r.sh) ? "H" : String(r.sh)) for r in eachrow(df)]
df.agebin = FVSLoss.binidx.(round.(Int, Float64.(df.age_calc)))
df.agb = Float64.(df.agb)
df.eco = string.(df.epa_l3, "|", df.land_use)
println("obs cohort rows: ", nrow(df), "  plots: ", length(unique(df.plotkey)), "  (plot,year) cells: ", length(unique(zip(df.plotkey, df.year))))

obs = FVSLoss.index_cohorts(df)
cell_of = Dict((r.plotkey, r.year, r.eff) => r.eco for r in eachrow(df))

L0 = FVSLoss.loss(obs, obs, cell_of)
@printf("\nobs-as-sim:    W=%.6g  AGB=%.6g  total=%.6g   (nobs=%d)\n", L0.W, L0.AGB, L0.total, L0.nobs)

# perturbation: sim = obs with AGB ×1.5 and one age-bin shift → W and AGB should both rise
dpert = copy(df); dpert.agb = dpert.agb .* 1.5; dpert.agebin = clamp.(dpert.agebin .+ 1, 1, FVSLoss.NB)
sim = FVSLoss.index_cohorts(dpert)
L1 = FVSLoss.loss(obs, sim, cell_of)
@printf("sim=obs×1.5,+1bin:  W=%.6g  AGB=%.6g  total=%.6g\n", L1.W, L1.AGB, L1.total)
println("\n✓ pass if  W≈0 for obs-as-sim  and  both W,AGB jump under perturbation")
