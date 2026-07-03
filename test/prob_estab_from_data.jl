# PROB_ESTAB from FIA, per species × L3 (natural). For EVERY species in the study L3 region we compute:
#   PROB_ESTAB(s,eco) = #(seed-source visits with a seedling of s) / #(seed-source visits), over the species'
#   own LIGHT window: ALSTKCD ∈ [max(1,4-SHADE_TOL_s), 4]  (shade_class=6-ALSTKCD; ALSTKCD 1=overstocked…4=poorly
#   stocked; 5=nonstocked EXCLUDED). "seed source" = a seedling of s OR a live s ≥ SONA maturity age.
# Each species uses ITS OWN SHADE_TOL (from runs/shadetol_all_species.csv) — so the LANDIS aggregates
#   (_GRP_41/_GRP_43/_S/_H) are pooled from member species each windowed correctly, not with one group SHADE_TOL.
#   MATURITY = SONA for the 10 study species, else sw=10/hw=20. Natural stands, L3 8.3.5/8.5.3.
#   Run: ./julia_gdal.sh --project=. test/prob_estab_from_data.jl
using DataFrames, DuckDB, Printf, CSV
con = DBInterface.connect(DuckDB.DB("/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"; readonly=true))  # READ-ONLY (never lock/write FIADB)

# per-species SHADE_TOL (global, from the all-species CCLCD>3 table); fallback by softwood/hardwood
stdf = CSV.read("runs/shadetol_all_species.csv", DataFrame)
SHADE = Dict(uppercase(strip(String(r.sym))) => Int(r.shade_class) for r in eachrow(stdf))
shadetol(sym, sh) = get(SHADE, sym, sh == "S" ? 2 : 3)
# SONA maturity for the 10 study species; else sw=10/hw=20
SONA = Dict("PIEL"=>10,"PIPA2"=>20,"PITA"=>15,"TAAS"=>30,"LIST2"=>20,"QUNI"=>20,"QULA3"=>15,"QUVI"=>20,"NYBI"=>30,"ACRU"=>5)
maturity(sym, sh) = get(SONA, sym, sh == "S" ? 10 : 20)
# LANDIS category (which of the 14 model "species" this maps to)
EXACT = Set(keys(SONA)); GRPS = Set([41,43])
cat(sym, g, sh) = sym in EXACT ? sym : (g in GRPS ? "_GRP_$g" : "_" * (sh in ("S","H") ? sh : "H"))
vk(st,un,co,pl,iv) = string(Int(st),"|",Int(un),"|",Int(co),"|",Int(pl),"|",Int(iv))
# STDORGCD 2-way stratum (NOT the old 4-way tree_trajectories.land_use). natural = never planted at any
# remeasurement/condition (NOT EXISTS stdorgcd=1); artificial = planted at some point (EXISTS) — exact complement.
#   Run: ./julia_gdal.sh --project=. test/prob_estab_from_data.jl [natural|artificial]   (default natural)
const MODE = length(ARGS) >= 1 ? lowercase(ARGS[1]) : "natural"
@assert MODE in ("natural", "artificial") "MODE must be natural or artificial"
const EXPRED = MODE == "natural" ? "NOT EXISTS" : "EXISTS"
const SUFFIX = MODE == "natural" ? "" : "_artificial"
const NP = "SELECT DISTINCT tt.statecd,tt.unitcd,tt.countycd,tt.plot,tt.epa_l3 FROM tree_trajectories tt WHERE $EXPRED (SELECT 1 FROM COND co WHERE co.statecd=tt.statecd AND co.unitcd=tt.unitcd AND co.countycd=tt.countycd AND co.plot=tt.plot AND co.stdorgcd=1) AND tt.epa_l3 IN ('8.3.5','8.5.3')"

# 1) visit universe + dominant LIVE_CANOPY_CVR_PCT (crown closure); only plot-measurements that HAVE it
plots = DataFrame(DBInterface.execute(con, """
  WITH np AS ($NP)
  SELECT c.STATECD st,c.UNITCD un,c.COUNTYCD co,c.PLOT pl,c.INVYR iv, any_value(np.epa_l3) l3, arg_max(c.LIVE_CANOPY_CVR_PCT,c.CONDPROP_UNADJ) cc
  FROM COND c JOIN np ON c.STATECD=np.statecd AND c.UNITCD=np.unitcd AND c.COUNTYCD=np.countycd AND c.PLOT=np.plot
  WHERE c.COND_STATUS_CD=1 AND c.LIVE_CANOPY_CVR_PCT IS NOT NULL AND c.CONDPROP_UNADJ IS NOT NULL GROUP BY 1,2,3,4,5"""))
plots.cc = Float64.(plots.cc); plots.vkey = [vk(r.st,r.un,r.co,r.pl,r.iv) for r in eachrow(plots)]
# Crown-closure light window: a species establishes where LIVE_CANOPY_CVR_PCT ≤ 40 + (SHADE_TOL-1)*15
#   (intolerant SHADE_TOL=1 → ≤40% open crown; tolerant =5 → ≤100% any). Keyed by SHADE_TOL 1..5.
eligset = Dict{Tuple{String,Int},Set{String}}()
for l3 in ["8.3.5","8.5.3"], stol in 1:5
  thr = 40 + (stol-1)*15
  eligset[(l3,stol)] = Set(plots[(plots.l3.==l3) .& (plots.cc .<= thr),:].vkey)
end

# 2) seedlings & 3) mature — keep species-level (sym, E_SPGRPCD, sftwd)
seed = DataFrame(DBInterface.execute(con, """
  WITH np AS ($NP)
  SELECT DISTINCT sd.STATECD st,sd.UNITCD un,sd.COUNTYCD co,sd.PLOT pl,sd.INVYR iv,
    UPPER(TRIM(rs.SPECIES_SYMBOL)) sym, rs.E_SPGRPCD g, UPPER(TRIM(rs.SFTWD_HRDWD)) sh
  FROM SEEDLING sd JOIN np ON sd.STATECD=np.statecd AND sd.UNITCD=np.unitcd AND sd.COUNTYCD=np.countycd AND sd.PLOT=np.plot
  JOIN REF_SPECIES rs ON rs.SPCD=sd.SPCD WHERE sd.TREECOUNT>0"""))
mat = DataFrame(DBInterface.execute(con, """
  WITH np AS ($NP)
  SELECT ct.STATECD st,ct.UNITCD un,ct.COUNTYCD co,ct.PLOT pl,ct.INVYR iv,
    UPPER(TRIM(rs.SPECIES_SYMBOL)) sym, rs.E_SPGRPCD g, UPPER(TRIM(rs.SFTWD_HRDWD)) sh, max(ct.estimated_age) maxage
  FROM curated_trees ct JOIN np ON ct.STATECD=np.statecd AND ct.UNITCD=np.unitcd AND ct.COUNTYCD=np.countycd AND ct.PLOT=np.plot
  JOIN REF_SPECIES rs ON rs.SPCD=ct.spcd_resolved WHERE ct.STATUSCD=1 AND ct.estimated_age IS NOT NULL GROUP BY 1,2,3,4,5,6,7,8"""))
sh_of = Dict{String,String}(); g_of = Dict{String,Int}()
for df in (seed,mat), r in eachrow(df); s=String(r.sym); sh_of[s]=ismissing(r.sh) ? "H" : String(r.sh); g_of[s]=Int(coalesce(r.g,0)); end
seed_set = Set((vk(r.st,r.un,r.co,r.pl,r.iv), String(r.sym)) for r in eachrow(seed))
mat_max = Dict{Tuple{String,String},Float64}()
for r in eachrow(mat); k=(vk(r.st,r.un,r.co,r.pl,r.iv),String(r.sym)); mat_max[k]=max(get(mat_max,k,0.0),Float64(r.maxage)); end

# 4) per (species, L3): counts over the species' own light window
species = sort(collect(union(Set(String.(seed.sym)), Set(String.(mat.sym)))))
persp = DataFrame(species=String[], category=String[], l3=String[], shade_tol=Int[], maturity=Int[], n_seedsource=Int[], n_seedling=Int[], prob_estab=Float64[])
for s in species, l3 in ["8.3.5","8.5.3"]
  sh = get(sh_of,s,"H"); stol=shadetol(s,sh); matr=maturity(s,sh)
  nden=0; nnum=0
  for p in eligset[(l3,stol)]
    hs=(p,s) in seed_set; hm=get(mat_max,(p,s),0.0)>=matr
    (hs||hm) || continue; nden+=1; hs && (nnum+=1)
  end
  nden>0 && push!(persp, (s, cat(s,get(g_of,s,0),sh), l3, stol, matr, nden, nnum, nnum/nden))
end
CSV.write("runs/prob_estab_all_species$SUFFIX.csv", sort(persp,[:l3,:category,:species]))

# 5) aggregate to the 14 LANDIS categories (pool member seed-source + seedling counts)
agg = combine(groupby(persp, [:category,:l3]), :n_seedsource=>sum=>:seedsrc, :n_seedling=>sum=>:seedling, :species=>length=>:n_species)
agg.prob_estab = agg.seedling ./ agg.seedsrc
CATS = vcat(sort(collect(EXACT)), ["_GRP_41","_GRP_43","_H","_S"])
for l3 in ["8.3.5","8.5.3"]
  println("\n=== L3 $l3 $MODE — PROB_ESTAB per LANDIS category (pooled from per-species estimates) ===")
  @printf("%-9s %8s %8s %8s %9s\n","category","seedsrc","seedlng","P_estab","#member_sp")
  for c in CATS
    r = agg[(agg.category.==c).&(agg.l3.==l3),:]
    nrow(r)==0 && continue
    @printf("%-9s %8d %8d %8.3f %9d\n", c, r.seedsrc[1], r.seedling[1], r.prob_estab[1], r.n_species[1])
  end
end
CSV.write("runs/prob_estab_from_data$SUFFIX.csv", agg)
println("\n[$MODE] wrote runs/prob_estab_all_species$SUFFIX.csv ($(nrow(persp)) species×L3 rows) + runs/prob_estab_from_data$SUFFIX.csv (categories)")
