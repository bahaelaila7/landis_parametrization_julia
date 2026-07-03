# FVS aspatial 100-yr NOAUTOES projection of the 4 EPA-L4 study plots, from CURATED TREES (with damage),
# → per (plotkey, sim-year offset, eff-species, agebin) AGB (g/m²). Writes fvs_cohorts_100.csv.
# birth_age = estimated_age so FVS TreeAge tracks initial-tree age → agebin is real. Compare offsets 25/50/75/100.
#   Run: ./julia_gdal.sh --project=. tools/fvs_compare100.jl
using DataFrames, DuckDB, Printf, Statistics, Dates, CSV
include(joinpath(@__DIR__, "..", "src", "FVS.jl")); using .FVS
const FIADB  = "/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"
const BINDIR = "/workspace/FVStest"
const LDPATH = "/home/node/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia"
const L4S    = ["8.3.5.65o", "8.5.3.75g", "8.5.3.75e", "8.5.3.75f"]
const HORIZON = 100
const HORIZONS = [25, 50, 75, 100]
const BINS   = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]
const TONS_ACRE_TO_G_M2 = 2000.0 * 453.592 / 4046.86
const ESTAB = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :noauto     # :noauto (NOAUTOES) | :auto (AUTOES)
const TAG = String(ESTAB)
# REGIMPUTE SN shade-tolerance addfile — supplies FIA-imputed natural regen so AUTOES actually establishes
# (SN's partial establishment model adds none on its own). Attached only for :auto runs.
const REGIMPUTE_SN = joinpath(@__DIR__, "regimpute", "REGIMPUTE", "Regen_ShadeTolerance_Method_SN.kcp")
con = DuckDB.connect(DuckDB.DB())
DuckDB.execute(con, "ATTACH '$(FIADB)' AS s (READ_ONLY);")

EXACT = Set(["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS"]); GRPS = Set([41,43])
ref = DataFrame(DBInterface.execute(con, "SELECT SPCD spcd, UPPER(TRIM(SPECIES_SYMBOL)) sym, UPPER(TRIM(SFTWD_HRDWD)) sh FROM s.REF_SPECIES"))
spcd_sym = Dict(Int(r.spcd) => String(r.sym) for r in eachrow(ref) if !ismissing(r.spcd))
spcd_sh  = Dict(Int(r.spcd) => (ismissing(r.sh) ? "H" : String(r.sh)) for r in eachrow(ref) if !ismissing(r.spcd))
grp_of   = Dict(String(r.sym) => Int(r.grp) for r in eachrow(DataFrame(DBInterface.execute(con,
  "SELECT DISTINCT UPPER(TRIM(species_symbol)) sym, spgrpcd grp FROM s.curated_cohorts_landis"))))
function tier_spcd(spcd::Int)
  sym = get(spcd_sym, spcd, ""); sym in EXACT && return sym
  get(grp_of, sym, -1) in GRPS && return "_GRP_$(grp_of[sym])"
  "_" * (get(spcd_sh, spcd, "H") in ("S","H") ? spcd_sh[spcd] : "H")
end
binidx(a) = (for (i,b) in enumerate(BINS); a < b && return i; end; length(BINS)+1)
binlabel(i) = i == 1 ? "≤$(BINS[1])" : i <= length(BINS) ? "$(BINS[i-1])–$(BINS[i])" : "$(BINS[end])+"

# aspatial curated stands (first measurement). Damage attributes come from curated_trees (curated_trees_fvs
# lacks them) via a per-tree-key join; FIA (DAMTYP,DAMSEV) pairs → FVS IDAMCD slots (direct pass, not a
# validated FIA→FVS crosswalk — flag if refining).
df = DataFrame(DBInterface.execute(con, """
  WITH dam AS (
    SELECT statecd,unitcd,countycd,plot,subp,tree,invyr,
           max(DAMTYP1) damtyp1, max(DAMSEV1) damsev1, max(DAMTYP2) damtyp2, max(DAMSEV2) damsev2
    FROM s.curated_trees GROUP BY 1,2,3,4,5,6,7)
  SELECT t.statecd,t.unitcd,t.countycd,t.plot, t.measdate, t.spcd, t.dia, t.ht, t.cr, t.tpa_unadj,
         t.estimated_age, t.site_slope, t.site_aspect, t.site_elev,
         d.damtyp1, d.damsev1, d.damtyp2, d.damsev2
  FROM s.curated_trees_fvs t
  LEFT JOIN dam d ON t.statecd=d.statecd AND t.unitcd=d.unitcd AND t.countycd=d.countycd
    AND t.plot=d.plot AND t.subp=d.subp AND t.tree=d.tree AND t.invyr=d.invyr
  WHERE t.epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f')
    AND t.statuscd=1 AND t.dia IS NOT NULL AND t.tpa_unadj>0
"""))
dmg(r) = (Int(coalesce(r.damtyp1,0)), Int(coalesce(r.damsev1,0)),
          Int(coalesce(r.damtyp2,0)), Int(coalesce(r.damsev2,0)), 0, 0)
stands = FVS.StandSpec[]
for gp in groupby(df, [:statecd,:unitcd,:countycd,:plot])
  fmd = minimum(gp.measdate); iv = year(fmd); init = gp[gp.measdate .== fmd, :]; ki = first(init)
  trees = FVS.TreeRec[(spcd=Int(r.spcd), dbh=Float64(r.dia), ht=ismissing(r.ht) ? 0.0 : Float64(r.ht),
    cr=ismissing(r.cr) ? 0.0 : Float64(r.cr), tpa=Float64(r.tpa_unadj), damage=dmg(r),
    birth_age=ismissing(r.estimated_age) ? 0.0 : Float64(r.estimated_age)) for r in eachrow(init)]
  push!(stands, FVS.StandSpec(id=join((ki.statecd,ki.unitcd,ki.countycd,ki.plot),"_"), inv_year=iv,
    target_years=iv .+ HORIZONS,
    slope=ismissing(ki.site_slope) ? nothing : Float64(ki.site_slope),
    aspect=ismissing(ki.site_aspect) ? nothing : Float64(ki.site_aspect),
    elev_ft=ismissing(ki.site_elev) ? nothing : Float64(ki.site_elev), trees=trees))
end
# FVS is stochastic (mortality allocation, regen/ingrowth) → run NREP replicates with distinct RANNSEEDs
# and average, mirroring the Pan reps. NREP=1 → FVS default seed (no RANNSEED emitted).
const NREP = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
println("FVS stands: ", length(stands), " (100-yr, $(uppercase(TAG))ES, curated+damage; NREP=$NREP)")
dir = joinpath(@__DIR__, "..", "tmp", "fvs_compare100_$(TAG)")
# one replicate → per (plotkey, year, eff, agebin) cohort AGB
function run_rep(ranseed)
  keypath, _, dbpath = FVS.write_run(stands; dir=dir, fiavbc=true, estab=ESTAB, ffe=false, ranseed=ranseed,
    regimpute = ESTAB === :auto ? REGIMPUTE_SN : nothing)
  ok, log = FVS.run_fvs(keypath; variant="sn", bindir=BINDIR, ld_library_path=LDPATH)
  res = FVS.read_fvs_sqlite(dbpath); tl = res.treelist; fb = res.fiavbc
  (tl === nothing || fb === nothing) && error("missing FVS_TreeList / FVS_FIAVBC_Summary")
  standagb = Dict{Tuple{String,Int},Float64}()
  for i in 1:nrow(fb); standagb[(String(fb.StandID[i]), Int(fb.Year[i]))] = Float64(fb.AbvGrdBio[i]) * TONS_ACRE_TO_G_M2; end
  tl.eff = [tier_spcd(parse(Int, String(s))) for s in tl.SpeciesFIA]
  tl.agebin = binidx.(round.(Int, Float64.(tl.TreeAge))); tl.vol = Float64.(tl.TCuFt) .* Float64.(tl.TPA)
  d = Dict{Tuple{String,Int,String,Int},Float64}()
  for gp in groupby(tl, [:StandID, :Year])
    sid = String(first(gp.StandID)); yr = Int(first(gp.Year)); sagb = get(standagb, (sid, yr), 0.0)
    vtot = sum(gp.vol); vtot <= 0 && continue
    for c in groupby(gp, [:eff, :agebin]); d[(sid, yr, String(first(c.eff)), Int(first(c.agebin)))] = sagb*sum(c.vol)/vtot; end
  end
  d
end
seeds = NREP == 1 ? [nothing] : [2*r - 1 for r in 1:NREP]   # distinct odd RANNSEEDs
acc = Dict{Tuple{String,Int,String,Int},Float64}(); reptot = Dict(off => Float64[] for off in HORIZONS)
nstands = length(stands)
for (ri, sd) in enumerate(seeds)
  d = run_rep(sd)
  iv = Dict{String,Int}(); for (pk, yr, _, _) in keys(d); iv[pk] = haskey(iv, pk) ? min(iv[pk], yr) : yr; end
  rt = Dict(off => 0.0 for off in HORIZONS)
  for ((pk, yr, eff, ab), v) in d
    acc[(pk, yr, eff, ab)] = get(acc, (pk, yr, eff, ab), 0.0) + v
    off = yr - iv[pk]; haskey(rt, off) && (rt[off] += v)
  end
  for off in HORIZONS; push!(reptot[off], rt[off] / nstands); end
  print("rep $ri(seed=$(sd===nothing ? "default" : sd)) "); flush(stdout)
end
println()
rows = [(plotkey=pk, year=yr, eff=eff, agebin=ab, agb=v / NREP) for ((pk, yr, eff, ab), v) in acc]
coh = DataFrame(rows)
iv = Dict(s => minimum(coh.year[coh.plotkey .== s]) for s in unique(coh.plotkey))
coh.offset = [coh.year[i] - iv[coh.plotkey[i]] for i in 1:nrow(coh)]; coh.agebin_label = binlabel.(coh.agebin)
out = joinpath(dir, "fvs_cohorts_100_$(TAG).csv"); CSV.write(out, coh)
rs = DataFrame(offset=HORIZONS, mean_agb=[mean(reptot[o]) for o in HORIZONS],
  std_agb=[length(reptot[o]) > 1 ? std(reptot[o]) : 0.0 for o in HORIZONS], nreps=NREP)
CSV.write(joinpath(dir, "fvs_repstats_$(TAG).csv"), rs)
println("wrote $out  ($(nrow(coh)) rows; mean over $NREP reps)")
for r in eachrow(rs); @printf("  yr %3d: mean stand AGB = %.0f ± %.0f g/m²  (over %d RANNSEEDs)\n", r.offset, r.mean_agb, r.std_agb, NREP); end
