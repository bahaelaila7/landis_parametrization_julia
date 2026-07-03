# FVS FL5 benchmark — runs FVSsn yearly for 50 yr, aggregates trees → cohort (tiered-species × age-bin),
# dry AGB = Aboveground_Total_Live(carbon)×2 apportioned by Σ(TCuFt·TPA). 12-run matrix:
#   layout {aspatial(4 EPA-L4 plots, first meas) | spatial(FL5 TreeMap CNs)}
#   × estab {auto | noauto}  × condition {raw(TREE+PLOT) | curated | curated_nodamage}
# Stage 1 here: aspatial × curated_nodamage × noauto, end-to-end + cohort×age×AGB sanity.
#   Run: ./julia_gdal.sh --project=. tools/fvs_benchmark.jl <layout> <estab> <condition>
using DataFrames, DuckDB, Printf, Statistics, Dates
include(joinpath(@__DIR__, "..", "src", "FVS.jl"))
using .FVS
include(joinpath(@__DIR__, "fvs_loss.jl")); using .FVSLoss

const FIADB   = "/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"
const BINDIR  = "/workspace/FVStest"                # our self-built canonical FVSsn
const LDPATH  = "/home/node/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia"  # libgfortran.so.5
const L4S     = ["8.3.5.65o", "8.5.3.75g", "8.5.3.75e", "8.5.3.75f"]
const HORIZON = 50
const BINS    = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]
const TONS_ACRE_TO_G_M2 = 2000.0 * 453.592 / 4046.86   # short ton/acre → g/m² (Pan native)
const FFE_C2B = 2.0                                     # aboveground biomass = FFE carbon / 0.5

layout    = length(ARGS) >= 1 ? ARGS[1] : "aspatial"
estabarg  = length(ARGS) >= 2 ? ARGS[2] : "noauto"
condition = length(ARGS) >= 3 ? ARGS[3] : "curated_nodamage"
estab = Symbol(estabarg)
println("=== FVS benchmark: layout=$layout  estab=$estab  condition=$condition ===")
con = DBInterface.connect(DuckDB.DB(FIADB))

# --- species tiering: FIA SPCD → effective species (EXACT 10 / _GRP_41,43 / _S,_H) ---
EXACT = Set(["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS"]); GRPS = Set([41,43])
ref = DataFrame(DBInterface.execute(con, "SELECT SPCD spcd, UPPER(TRIM(SPECIES_SYMBOL)) sym, UPPER(TRIM(SFTWD_HRDWD)) sh FROM REF_SPECIES"))
spcd_sym = Dict(Int(r.spcd) => String(r.sym) for r in eachrow(ref) if !ismissing(r.spcd))
spcd_sh  = Dict(Int(r.spcd) => (ismissing(r.sh) ? "H" : String(r.sh)) for r in eachrow(ref) if !ismissing(r.spcd))
grp_of   = Dict(String(r.sym) => Int(r.grp) for r in eachrow(DataFrame(DBInterface.execute(con,
  "SELECT DISTINCT UPPER(TRIM(species_symbol)) sym, spgrpcd grp FROM curated_cohorts_landis"))))
function tier_spcd(spcd::Int)
  sym = get(spcd_sym, spcd, ""); sym in EXACT && return sym
  get(grp_of, sym, -1) in GRPS && return "_GRP_$(grp_of[sym])"
  "_" * (get(spcd_sh, spcd, "H") in ("S","H") ? spcd_sh[spcd] : "H")
end
binidx(a) = (for (i,b) in enumerate(BINS); a < b && return i; end; length(BINS)+1)
# FVS caps at MAXCYC=40 cycles. To reach year 50 while keeping yearly resolution through the FIA
# remeasure window (loss years ≤~20) + year 25, use yearly 0–30 then 5-yr to 50 = 34 cycles (≤40).
bench_target_years(iv) = vcat(collect((iv+1):(iv+30)), collect((iv+35):5:(iv+HORIZON)))

# --- build StandSpecs for aspatial curated (first measurement only) ---
function aspatial_curated_stands(; damage::Bool)
  df = DataFrame(DBInterface.execute(con, """
    SELECT t.statecd,t.unitcd,t.countycd,t.plot, t.measdate, t.spcd, t.dia, t.ht, t.cr, t.tpa_unadj,
           t.estimated_age, t.site_slope, t.site_aspect, t.site_elev
    FROM curated_trees_fvs t
    WHERE t.epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f')
      AND t.statuscd=1 AND t.dia IS NOT NULL AND t.tpa_unadj>0
  """))
  stands = FVS.StandSpec[]
  for g in groupby(df, [:statecd,:unitcd,:countycd,:plot])
    fmd = minimum(g.measdate); iv = year(fmd); init = g[g.measdate .== fmd, :]; ki = first(init)
    # target years = later-remeasurement offsets (for the loss) ∪ {25,50} (figures); cycle_plan(maxlen=10)
    # fills the big 25→50 gap with intermediate cycles. ~6–10 cycles vs 34 yearly.
    offs = round.(Int, Dates.value.(Dates.Day.(g.measdate .- fmd)) ./ 365.25)
    targets = sort(unique(filter(t -> 0 < t <= HORIZON, vcat(offs, 25, 50))))
    trees = FVS.TreeRec[(spcd=Int(r.spcd), dbh=Float64(r.dia), ht=ismissing(r.ht) ? 0.0 : Float64(r.ht),
      cr=ismissing(r.cr) ? 0.0 : Float64(r.cr), tpa=Float64(r.tpa_unadj),
      damage=(0,0,0,0,0,0), birth_age=ismissing(r.estimated_age) ? 0.0 : Float64(r.estimated_age)) for r in eachrow(init)]
    push!(stands, FVS.StandSpec(id=join((ki.statecd,ki.unitcd,ki.countycd,ki.plot),"_"), inv_year=iv,
      target_years=iv .+ targets,
      slope=ismissing(ki.site_slope) ? nothing : Float64(ki.site_slope),
      aspect=ismissing(ki.site_aspect) ? nothing : Float64(ki.site_aspect),
      elev_ft=ismissing(ki.site_elev) ? nothing : Float64(ki.site_elev), trees=trees))
  end
  stands
end

# --- aggregate FVS output → per (stand, year, eff, agebin) AGB (g/m²) ---
function aggregate_cohorts(res)
  tl = res.treelist; fb = res.fiavbc
  (tl === nothing || fb === nothing) && error("missing FVS_TreeList / FVS_FIAVBC_Summary")
  # stand AGB (g/m²) per (StandID, Year) = FIAVBC AbvGrdBio (FIA-consistent aboveground biomass; no FFE)
  standagb = Dict{Tuple{String,Int},Float64}()
  for i in 1:nrow(fb)
    standagb[(String(fb.StandID[i]), Int(fb.Year[i]))] = Float64(fb.AbvGrdBio[i]) * TONS_ACRE_TO_G_M2
  end
  # per-tree volume share → cohort AGB
  rows = NamedTuple[]
  tl.eff = [tier_spcd(parse(Int, String(s))) for s in tl.SpeciesFIA]
  tl.agebin = binidx.(round.(Int, Float64.(tl.TreeAge)))
  tl.vol = Float64.(tl.TCuFt) .* Float64.(tl.TPA)
  for g in groupby(tl, [:StandID, :Year])
    sid = String(first(g.StandID)); yr = Int(first(g.Year))
    sagb = get(standagb, (sid, yr), 0.0); vtot = sum(g.vol)
    vtot <= 0 && continue
    for c in groupby(g, [:eff, :agebin])
      push!(rows, (standid=sid, year=yr, eff=first(c.eff), agebin=first(c.agebin),
        agb=sagb * sum(c.vol)/vtot, tpa=sum(Float64.(c.TPA))))
    end
  end
  DataFrame(rows)
end

# --- run one config ---
stands = layout == "aspatial" ?
  (condition == "raw" ? error("raw loader: stage 2") : aspatial_curated_stands(damage = (condition == "curated"))) :
  error("spatial loader: stage 2")
println("stands: ", length(stands), "  (yearly 0..$HORIZON)")
dir = joinpath(@__DIR__, "..", "tmp", "fvs_bench", "$(layout)_$(estab)_$(condition)")
keypath, _, dbpath = FVS.write_run(stands; dir=dir, fiavbc=true, estab=estab, ffe=false)
ok, log = FVS.run_fvs(keypath; variant="sn", bindir=BINDIR, ld_library_path=LDPATH)
println("FVS ok=$ok  db=$dbpath")
res = FVS.read_fvs_sqlite(dbpath)
coh = aggregate_cohorts(res)
println("cohort rows: ", nrow(coh))
# sanity: total AGB (g/m², mean over stands) at sim-year offset 0, 25, 50 (per-stand inv year)
invyr = Dict(s => minimum(coh.year[coh.standid .== s]) for s in unique(coh.standid))
coh.offset = [coh.year[i] - invyr[coh.standid[i]] for i in 1:nrow(coh)]
for off in (0, 25, 50)
  s = coh[coh.offset .== off, :]; isempty(s) && continue
  byst = combine(groupby(s, :standid), :agb => sum => :tot)
  @printf("  sim-year %2d: %d stands, mean stand AGB = %.0f g/m²  (%.1f short-ton/acre)\n",
    off, nrow(byst), mean(byst.tot), mean(byst.tot) / TONS_ACRE_TO_G_M2)
end
# --- benchmark loss vs curated ground truth (tier-3 + hinge + cell-norm, all ref data, eco×lu×sp) ---
# Curated source only (FVS init == curated cohorts ⇒ apples-to-apples). Same replica scores Pan/Landis later.
if startswith(condition, "curated")
  obsdf = DataFrame(DBInterface.execute(con, """
    WITH base AS (
      SELECT statecd,unitcd,countycd,plot, epa_l3, land_use, measdate, age_calc,
             UPPER(TRIM(species_symbol)) sym, spgrpcd grp, UPPER(TRIM(sftwd_hrdwd)) sh, SUM(agb) agb
      FROM curated_cohorts_landis WHERE epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f') AND agb>0
      GROUP BY statecd,unitcd,countycd,plot,epa_l3,land_use,measdate,age_calc,
               UPPER(TRIM(species_symbol)),spgrpcd,UPPER(TRIM(sftwd_hrdwd)))
    SELECT *, CAST(round(datediff('day', min(measdate) OVER
             (PARTITION BY statecd,unitcd,countycd,plot), measdate)/365.25) AS INTEGER) yr FROM base
  """))
  otier(s, g, h) = s in EXACT ? s : (g in GRPS ? "_GRP_$g" : "_" * (h in ("S","H") ? h : "H"))
  obsdf.plotkey = string.(obsdf.statecd,"_",obsdf.unitcd,"_",obsdf.countycd,"_",obsdf.plot)
  obsdf.year = Int.(obsdf.yr)
  obsdf.eff = [otier(String(r.sym), Int(r.grp), ismissing(r.sh) ? "H" : String(r.sh)) for r in eachrow(obsdf)]
  obsdf.agebin = FVSLoss.binidx.(round.(Int, Float64.(obsdf.age_calc)))
  obsdf.agb = Float64.(obsdf.agb); obsdf.eco = string.(obsdf.epa_l3,"|",obsdf.land_use)
  obs = FVSLoss.index_cohorts(obsdf)
  cell_of = Dict((r.plotkey, r.year, r.eff) => r.eco for r in eachrow(obsdf))
  # FVS sim keyed by sim-year offset (= calendar Year − stand inv year), so it aligns with obs offsets
  iv = Dict(s => minimum(coh.year[coh.standid .== s]) for s in unique(coh.standid))
  simdf = DataFrame(plotkey=coh.standid, year=[coh.year[i] - iv[coh.standid[i]] for i in 1:nrow(coh)],
                    eff=coh.eff, agebin=coh.agebin, agb=coh.agb)
  sim = FVSLoss.index_cohorts(simdf)
  L = FVSLoss.loss(obs, sim, cell_of)
  @printf("\n=== FVS benchmark loss (vs ALL ref, eco×lu×sp): total=%.5g  A_W=%.5g  A_AGB=%.5g  (nobs=%d) ===\n",
    L.total, L.W, L.AGB, L.nobs)
end
mkpath("runs");
println("OK — cohort×age×AGB + loss for $(layout)/$(estab)/$(condition)")
