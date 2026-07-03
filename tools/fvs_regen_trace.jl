# Trace WHY REGIMPUTE adds ~0 regen: run 2 multi-plot stands with REGIMPUTE + a diagnostic COMPUTE that
# dumps, every cycle, RGNST, expected saplings STxMR, present saplings SAPSTx, seed-source SpCountx, the
# adjusted per-species regen STxMRA, and SDI vs the gate. Reads the values back from run.out CMPU echoes.
using DataFrames, DuckDB, Dates
include(joinpath(@__DIR__,"..","src","FVS.jl")); using .FVS
const FIADB="/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"; const BINDIR="/workspace/FVStest"
const LDPATH="/home/node/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia"
const HORIZONS=[25,50,75,100]; const REGIMPUTE_SN=joinpath(@__DIR__,"regimpute","Regen_ShadeTolerance_Method_SN_fixed.kcp")  # FIXED: RGNST computes, regen ON
const INVYEAR=2000; const TREE_CAP=2900; const MAXPLT=500
con=DuckDB.connect(DuckDB.DB()); DuckDB.execute(con,"ATTACH '$(FIADB)' AS s (READ_ONLY);")
lu=Dict{NTuple{4,Int},String}()
for r in eachrow(DataFrame(DBInterface.execute(con,"SELECT DISTINCT statecd,unitcd,countycd,plot,land_use FROM s.curated_cohorts_landis_stdorg WHERE epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f') AND land_use IS NOT NULL")))
  lu[(Int(r.statecd),Int(r.unitcd),Int(r.countycd),Int(r.plot))]=String(r.land_use)
end
df=DataFrame(DBInterface.execute(con,"""SELECT t.statecd,t.unitcd,t.countycd,t.plot,t.measdate,t.spcd,t.dia,t.ht,t.cr,t.tpa_unadj,t.estimated_age,t.epa_l4
  FROM s.curated_trees_fvs t WHERE t.epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f') AND t.statuscd=1 AND t.dia IS NOT NULL AND t.tpa_unadj>0"""))
l3of(e)=join(split(String(e),".")[1:3],".")
strata=Dict{Tuple{String,String},Vector{Any}}()
for gp in groupby(df,[:statecd,:unitcd,:countycd,:plot])
  ki=first(gp); pk=(Int(ki.statecd),Int(ki.unitcd),Int(ki.countycd),Int(ki.plot)); haskey(lu,pk)||continue
  fmd=minimum(gp.measdate); init=gp[gp.measdate.==fmd,:]
  trees=[(spcd=Int(r.spcd),dbh=Float64(r.dia),ht=ismissing(r.ht) ? 0.0 : Float64(r.ht),cr=ismissing(r.cr) ? 0.0 : Float64(r.cr),tpa=Float64(r.tpa_unadj),birth_age=ismissing(r.estimated_age) ? 0.0 : Float64(r.estimated_age)) for r in eachrow(init)]
  push!(get!(strata,(l3of(ki.epa_l4),lu[pk]),Any[]),trees)
end
stands=FVS.StandSpec[]
for ((l3,luv),plots) in sort(collect(strata),by=first)
  length(stands)>=2 && break
  chunk=plots[1:min(MAXPLT,length(plots))]; ct=0; keep=Any[]
  for p in chunk; ct+length(p)>TREE_CAP && break; push!(keep,p); ct+=length(p); end
  alltrees=FVS.TreeRec[]; pts=Int[]
  for (pi,pt) in enumerate(keep), t in pt; push!(alltrees,(spcd=t.spcd,dbh=t.dbh,ht=t.ht,cr=t.cr,tpa=t.tpa,damage=(0,0,0,0,0,0),birth_age=t.birth_age)); push!(pts,pi); end
  push!(stands,FVS.StandSpec(id="$(replace(l3,"."=>"_"))_$(luv)",inv_year=INVYEAR,target_years=INVYEAR.+collect(10:10:100),trees=alltrees,n_plots=length(keep),points=pts))  # clean 10-yr cycles (match REGIMPUTE's IF-10 min-delay)
  println("stand $(replace(l3,"."=>"_"))_$(luv): $(length(keep)) plots, $(length(alltrees)) trees")
end
dir=joinpath(@__DIR__,"..","tmp","fvs_regen_trace"); mkpath(dir)
diag="""
IF         0
CYCLE GE 1
THEN
Compute            0
DRGNST=RGNST
DSDI=SPMCDBH(11,ALL,0,1.0,999,1.0,999,0)
DGATE=BSDIMAX*0.75
DST3MR=ST3MR
DSAP3=SAPST3
DSPC3=SpCount3
DMRA3=ST3MRA
DST4MR=ST4MR
DSAP4=SAPST4
DSPC4=SpCount4
DMRA4=ST4MRA
End
ENDIF
"""
combined=joinpath(dir,"regimpute_trace.kcp"); open(combined,"w") do io; write(io,read(REGIMPUTE_SN,String)); write(io,diag); end
keypath,_,dbpath=FVS.write_run(stands;dir=dir,fiavbc=true,estab=:auto,ffe=false,regimpute=combined)
ok,_=FVS.run_fvs(keypath;variant="sn",bindir=BINDIR,ld_library_path=LDPATH)
println("FVS ok=$ok  run.out=$(joinpath(dir,"run.out"))")
