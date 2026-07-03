# FVS 4-strata (5-stand) REGIMPUTE test. Group the study plots into eco(L3)×land_use stands — each FIA plot
# becomes one FVS point — run AUTOES + REGIMPUTE, and check whether the young age-classes finally regenerate
# (they were all 0 with single-plot stands). Splits any stratum >MAXPLT=500 plots into chunks.
using DataFrames, DuckDB, Printf, Statistics, Dates
include(joinpath(@__DIR__, "..", "src", "FVS.jl")); using .FVS
const FIADB="/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"; const BINDIR="/workspace/FVStest"
const LDPATH="/home/node/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia"
const HORIZONS=[25,50,75,100]; const BINS=[10,20,30,40,50,60,80,100,120,150]
const REGIMPUTE_SN=get(ENV,"REGIMPUTE_KCP",joinpath(@__DIR__,"regimpute","REGIMPUTE","Regen_ShadeTolerance_Method_SN.kcp"))
const MAXPLT=500; const TREE_CAP=2900; const INVYEAR=2000   # FVS caps MAXTRE=3000 trees & MAXPLT=500 plots/stand
binidx(a)=(for (i,b) in enumerate(BINS); a<b && return i end; length(BINS)+1)

con=DuckDB.connect(DuckDB.DB()); DuckDB.execute(con,"ATTACH '$(FIADB)' AS s (READ_ONLY);")
# plot -> land_use
lu=Dict{NTuple{4,Int},String}()
for r in eachrow(DataFrame(DBInterface.execute(con,"SELECT DISTINCT statecd,unitcd,countycd,plot,land_use FROM s.curated_cohorts_landis_stdorg WHERE epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f') AND land_use IS NOT NULL")))
  lu[(Int(r.statecd),Int(r.unitcd),Int(r.countycd),Int(r.plot))]=String(r.land_use)
end
df=DataFrame(DBInterface.execute(con,"""
  SELECT t.statecd,t.unitcd,t.countycd,t.plot,t.measdate,t.spcd,t.dia,t.ht,t.cr,t.tpa_unadj,
         t.estimated_age,t.site_slope,t.site_aspect,t.site_elev,t.epa_l4
  FROM s.curated_trees_fvs t
  WHERE t.epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f') AND t.statuscd=1 AND t.dia IS NOT NULL AND t.tpa_unadj>0"""))
l3of(e)=join(split(String(e),".")[1:3],".")
strata=Dict{Tuple{String,String},Vector{Any}}()
for gp in groupby(df,[:statecd,:unitcd,:countycd,:plot])
  ki=first(gp); pk=(Int(ki.statecd),Int(ki.unitcd),Int(ki.countycd),Int(ki.plot)); haskey(lu,pk)||continue
  l3=l3of(ki.epa_l4); luv=lu[pk]
  fmd=minimum(gp.measdate); init=gp[gp.measdate.==fmd,:]
  trees=[(spcd=Int(r.spcd),dbh=Float64(r.dia),ht=ismissing(r.ht) ? 0.0 : Float64(r.ht),cr=ismissing(r.cr) ? 0.0 : Float64(r.cr),tpa=Float64(r.tpa_unadj),birth_age=ismissing(r.estimated_age) ? 0.0 : Float64(r.estimated_age)) for r in eachrow(init)]
  push!(get!(strata,(l3,luv),Any[]),trees)
end
stands=FVS.StandSpec[]; meta=[]
for ((l3,luv),plots) in sort(collect(strata),by=first)
  # greedily pack plots into chunks of ≤TREE_CAP trees and ≤MAXPLT plots (each chunk = one FVS stand)
  chunks=Vector{Vector{Any}}(); cur=Any[]; ct=0
  for ptrees in plots
    if !isempty(cur) && (ct+length(ptrees) > TREE_CAP || length(cur) >= MAXPLT); push!(chunks,cur); cur=Any[]; ct=0; end
    push!(cur,ptrees); ct+=length(ptrees)
  end
  !isempty(cur) && push!(chunks,cur)
  for (ci,chunk) in enumerate(chunks)
    alltrees=FVS.TreeRec[]; pts=Int[]
    for (pi,ptrees) in enumerate(chunk), t in ptrees
      push!(alltrees,(spcd=t.spcd,dbh=t.dbh,ht=t.ht,cr=t.cr,tpa=t.tpa,damage=(0,0,0,0,0,0),birth_age=t.birth_age)); push!(pts,pi)
    end
    id="$(replace(l3,"."=>"_"))_$(luv)"*(length(chunks)>1 ? "_$ci" : "")
    push!(stands,FVS.StandSpec(id=id,inv_year=INVYEAR,target_years=INVYEAR.+HORIZONS,trees=alltrees,n_plots=length(chunk),points=pts))
    push!(meta,(id=id,np=length(chunk),nt=length(alltrees)))
  end
end
println("built $(length(stands)) stands:"); for m in meta; println("  $(m.id): $(m.np) plots, $(m.nt) trees"); end
dir=joinpath(@__DIR__,"..","tmp","fvs_strata_regimpute")
keypath,_,dbpath=FVS.write_run(stands;dir=dir,fiavbc=true,estab=:auto,ffe=false,regimpute=REGIMPUTE_SN)
ok,log=FVS.run_fvs(keypath;variant="sn",bindir=BINDIR,ld_library_path=LDPATH)
println("FVS ok=$ok")
res=FVS.read_fvs_sqlite(dbpath); tl=res.treelist
if tl===nothing; println("!! NO TREELIST"); else
  tl.age=round.(Int,Float64.(tl.TreeAge)); tl.ab=binidx.(tl.age); tl.tpa=Float64.(tl.TPA)
  println("\n=== REGEN CHECK: trees-per-acre in young bins (age≤10 / ≤20) per stand × offset ===")
  for gp in groupby(tl,:StandID)
    sid=String(first(gp.StandID)); print(rpad(sid,20))
    for y in HORIZONS
      s=gp[gp.Year.==(INVYEAR+y),:]; y10=sum(s.tpa[s.age.<=10]); y20=sum(s.tpa[s.age.<=20])
      @printf(" yr%d[≤10=%.1f ≤20=%.1f]",y,y10,y20)
    end
    println()
  end
end
