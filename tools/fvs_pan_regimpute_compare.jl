# FVS (4 strata, AUTOES + FIXED REGIMPUTE) vs Pan candidate-6 (AUTOES): mean plot AGB (short tons/acre) by
# tiered species × coarse age-bin per stratum at 25/50/75/100 yr. Groups plots into eco×land_use stands
# (sub-stands under FVS MAXTRE=3000/MAXPLT=500), runs regen-enabled FVS, aggregates sub-stands→strata,
# aggregates Pan candidate-6 AUTOES→strata, and writes one heatmap PNG per stratum (FVS | Pan | FVS−Pan).
using DataFrames, DuckDB, Printf, Statistics, Dates, CSV
import CairoMakie; const MK=CairoMakie
include(joinpath(@__DIR__,"..","src","FVS.jl")); using .FVS
const FIADB="/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"; const BINDIR="/workspace/FVStest"
const LDPATH="/home/node/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia"
const HORIZONS=[25,50,75,100]; const BINS=[10,20,30,40,50,60,80,100,120,150]; const nb=length(BINS)+1
const KCP=joinpath(@__DIR__,"regimpute","Regen_ShadeTolerance_Method_SN_fixed.kcp")
const MAXPLT=500; const TREE_CAP=2900; const INVYEAR=2000
const GM2_TPA=2000.0*453.592/4046.86; tpa(x)=x/GM2_TPA
const PANCSV=get(ENV,"PAN_CSV","runs/fl5_l4cover_mocmaes_simA_wsf01_longpin_stdorg_outputs/candidates/candidate_6/pan_cohorts_100_autoes.csv")
const C6DIR=dirname(PANCSV)   # comparison PNGs land next to the Pan candidate
const PAN_LABEL=get(ENV,"PAN_LABEL",replace(basename(dirname(PANCSV)),"candidate_"=>"cand-"))   # title label for the Pan candidate
binidx(a)=(for (i,b) in enumerate(BINS); a<b && return i end; length(BINS)+1)
binlabel(i)=i==1 ? "≤$(BINS[1])" : i<=length(BINS) ? "$(BINS[i-1])-$(BINS[i])" : "$(BINS[end])+"

con=DuckDB.connect(DuckDB.DB()); DuckDB.execute(con,"ATTACH '$(FIADB)' AS s (READ_ONLY);")
# species tiering (same 14 as fvs_compare100)
EXACT=Set(["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS"]); GRPS=Set([41,43])
ref=DataFrame(DBInterface.execute(con,"SELECT SPCD spcd,UPPER(TRIM(SPECIES_SYMBOL)) sym,UPPER(TRIM(SFTWD_HRDWD)) sh FROM s.REF_SPECIES"))
spcd_sym=Dict(Int(r.spcd)=>String(r.sym) for r in eachrow(ref) if !ismissing(r.spcd))
spcd_sh=Dict(Int(r.spcd)=>(ismissing(r.sh) ? "H" : String(r.sh)) for r in eachrow(ref) if !ismissing(r.spcd))
grp_of=Dict(String(r.sym)=>Int(r.grp) for r in eachrow(DataFrame(DBInterface.execute(con,"SELECT DISTINCT UPPER(TRIM(species_symbol)) sym,spgrpcd grp FROM s.curated_cohorts_landis"))))
function tier(spcd)
  sym=get(spcd_sym,spcd,""); sym in EXACT && return sym
  get(grp_of,sym,-1) in GRPS && return "_GRP_$(grp_of[sym])"
  "_"*(get(spcd_sh,spcd,"H") in ("S","H") ? spcd_sh[spcd] : "H")
end
# plot -> land_use
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
# build sub-stands, remember stratum + plot count of each
stands=FVS.StandSpec[]; smeta=Dict{String,Tuple{String,Int}}()  # standid -> (stratumkey, nplots)
for ((l3,luv),plots) in sort(collect(strata),by=first)
  chunks=Vector{Vector{Any}}(); cur=Any[]; ct=0
  for pt in plots; if !isempty(cur)&&(ct+length(pt)>TREE_CAP||length(cur)>=MAXPLT); push!(chunks,cur); cur=Any[]; ct=0 end; push!(cur,pt); ct+=length(pt) end
  !isempty(cur)&&push!(chunks,cur)
  for (ci,chunk) in enumerate(chunks)
    at=FVS.TreeRec[]; pts=Int[]
    for (pi,ptr) in enumerate(chunk), t in ptr; push!(at,(spcd=t.spcd,dbh=t.dbh,ht=t.ht,cr=t.cr,tpa=t.tpa,damage=(0,0,0,0,0,0),birth_age=t.birth_age)); push!(pts,pi) end
    id="$(replace(l3,"."=>"_"))_$(luv)"*(length(chunks)>1 ? "_$ci" : "")
    push!(stands,FVS.StandSpec(id=id,inv_year=INVYEAR,target_years=INVYEAR.+HORIZONS,trees=at,n_plots=length(chunk),points=pts))
    smeta[id]=("$l3|lu=$luv",length(chunk))
  end
end
strata_keys=sort(unique(v[1] for v in values(smeta)))
println("FVS: $(length(stands)) sub-stands across $(length(strata_keys)) strata: $strata_keys")
dir=joinpath(@__DIR__,"..","tmp","fvs_pan_regimpute"); mkpath(dir); dbpath=joinpath(dir,"FVSOut.db")
if get(ENV,"REUSE_FVS","")=="1" && isfile(dbpath)
  println("REUSE_FVS: reading cached FVS-REGIMPUTE output $dbpath (FVS side is Pan-candidate-independent)")
else
  keypath,_,dbpath=FVS.write_run(stands;dir=dir,fiavbc=true,estab=:auto,ffe=false,regimpute=KCP)
  ok,_=FVS.run_fvs(keypath;variant="sn",bindir=BINDIR,ld_library_path=LDPATH); println("FVS ok=$ok")
end
res=FVS.read_fvs_sqlite(dbpath); tl=res.treelist; fb=res.fiavbc
@assert tl!==nothing && fb!==nothing "missing FVS output"
# per sub-stand: mean-plot AGB (g/m²) per (offset, tier, agebin) — same recipe as fvs_compare100
standagb=Dict{Tuple{String,Int},Float64}()
for i in 1:nrow(fb); standagb[(String(fb.StandID[i]),Int(fb.Year[i]))]=Float64(fb.AbvGrdBio[i])*GM2_TPA; end
tl.eff=[tier(parse(Int,String(s))) for s in tl.SpeciesFIA]; tl.ab=binidx.(round.(Int,Float64.(tl.TreeAge))); tl.vol=Float64.(tl.TCuFt).*Float64.(tl.TPA)
# FVS strata accumulator: (stratum, offset, eff, agebin) -> (Σ agb×nplots, Σ nplots-per-offset tracked separately)
FVS_agb=Dict{Tuple{String,Int,String,Int},Float64}(); STRN=Dict{String,Int}()
for (sk,(_,np)) in smeta; end
for gp in groupby(tl,[:StandID,:Year])
  sid=String(first(gp.StandID)); yr=Int(first(gp.Year)); off=yr-INVYEAR; off in HORIZONS || continue
  haskey(smeta,sid) || continue; (stk,np)=smeta[sid]
  sagb=get(standagb,(sid,yr),0.0); vtot=sum(gp.vol); vtot<=0 && continue
  for c in groupby(gp,[:eff,:ab])
    e=String(first(c.eff)); ab=Int(first(c.ab)); a=sagb*sum(c.vol)/vtot   # per-acre g/m² for this sub-stand
    FVS_agb[(stk,off,e,ab)]=get(FVS_agb,(stk,off,e,ab),0.0)+a*np           # weight by plot count
  end
end
# per-stratum total plot count (for the /n)
for (sid,(stk,np)) in smeta; STRN[stk]=get(STRN,stk,0)+np; end
# ---- Pan candidate-6 AUTOES → strata ----
pan=CSV.read(PANCSV,DataFrame)
# plot(statecd_unitcd_countycd_plot) -> stratum: need eco (l3) + land_use for each pan plot
p2strat=Dict{String,String}()
for r in eachrow(DataFrame(DBInterface.execute(con,"""SELECT DISTINCT statecd,unitcd,countycd,plot,epa_l4,land_use FROM s.curated_cohorts_landis_stdorg
   WHERE epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f') AND land_use IS NOT NULL""")))
  p2strat[join((Int(r.statecd),Int(r.unitcd),Int(r.countycd),Int(r.plot)),"_")]="$(l3of(r.epa_l4))|lu=$(r.land_use)"
end
Pan_agb=Dict{Tuple{String,Int,String,Int},Float64}(); PANN=Dict{String,Set{String}}()
for r in eachrow(pan)
  pk=String(r.plotkey); haskey(p2strat,pk)||continue; stk=p2strat[pk]; off=Int(r.offset); off in HORIZONS||continue
  Pan_agb[(stk,off,String(r.eff),Int(r.agebin))]=get(Pan_agb,(stk,off,String(r.eff),Int(r.agebin)),0.0)+Float64(r.agb)
  push!(get!(PANN,stk,Set{String}()),pk)
end
# ---- plot: one PNG per stratum (rows=horizons, cols=FVS|Pan|diff), tons/acre, shared log scale ----
allsp=sort(unique(vcat([k[3] for k in keys(FVS_agb)],[k[3] for k in keys(Pan_agb)])))
spidx=Dict(s=>i for (i,s) in enumerate(allsp)); ns=length(allsp)
matof(D,stk,off,n)= (M=zeros(ns,nb); for ((s,o,e,ab),v) in D; (s==stk&&o==off)||continue; M[spidx[e],ab]+=tpa(v)/n end; M)
# global vmax for shared scale
vmax=Ref(0.0)
for stk in strata_keys, off in HORIZONS
  fn=get(STRN,stk,1); pn=max(length(get(PANN,stk,Set{String}())),1)
  vmax[]=max(vmax[], maximum(matof(FVS_agb,stk,off,fn)), maximum(matof(Pan_agb,stk,off,pn)))
end
vmax=ceil(vmax[]/5)*5
vmax=ceil(vmax/5)*5; VMIN=0.1; DMAX=1000.0/GM2_TPA
println("strata: $strata_keys ; shared AGB vmax=$vmax t/ac ; err ±$(round(DMAX,digits=2))")
seq=MK.cgrad(:viridis); div=MK.cgrad(:RdBu;rev=true)
for stk in strata_keys
  fn=get(STRN,stk,1); pn=max(length(get(PANN,stk,Set{String}())),1)
  fig=MK.Figure(size=(240+3*(90+18*nb),90+4*(60+15*ns)))
  MK.Label(fig[0,1:4],"FVS (AUTOES+REGIMPUTE) vs Pan $(PAN_LABEL) (AUTOES) — $stk — mean plot AGB (t/ac), log; FVS n=$fn plots, Pan n=$pn";fontsize=13,font=:bold)
  MK.Label(fig[-1,1:4],"REGIMPUTE fixed (dead _DomCC line removed) → regen ON. Shared log $(VMIN)–$(round(Int,vmax)) t/ac (gray=0); err ±$(round(DMAX,digits=2)) t/ac";fontsize=10,color=:gray30)
  hmref=Ref{Any}(nothing); dref=Ref{Any}(nothing)
  for (ri,off) in enumerate(HORIZONS)
    FM=matof(FVS_agb,stk,off,fn); PM=matof(Pan_agb,stk,off,pn)
    for (ci,(M,lab)) in enumerate([(FM,"FVS+REGIMPUTE"),(PM,"Pan $(PAN_LABEL)"),(FM.-PM,"FVS − Pan")])
      ax=MK.Axis(fig[ri,ci];title="yr $off — $lab",xlabel=(ri==4 ? "age bin" : ""),ylabel=(ci==1 ? "species" : ""),
        xticks=(1:nb,binlabel.(1:nb)),yticks=(1:ns,allsp),xticklabelrotation=π/4,xticklabelsize=7,yticklabelsize=7,titlesize=10)
      if ci<=2
        Mp=map(x->x<=0 ? NaN : x,M); hm=MK.heatmap!(ax,1:nb,1:ns,permutedims(Mp);colormap=seq,colorscale=log10,colorrange=(VMIN,vmax),nan_color=:gray85); hmref[]=hm
      else
        hm=MK.heatmap!(ax,1:nb,1:ns,permutedims(M);colormap=div,colorrange=(-DMAX,DMAX)); dref[]=hm
      end
      tot=round(sum(M),digits=1); MK.text!(ax,0.99,0.01;text="Σ=$tot",space=:relative,align=(:right,:bottom),fontsize=8,color=:white)
    end
  end
  MK.Colorbar(fig[1:4,4],hmref[];label="AGB t/ac (log; gray=0)")
  MK.Colorbar(fig[1:4,5],dref[];label="FVS−Pan t/ac (±$(round(DMAX,digits=2)))")
  out=joinpath(C6DIR,"fvs_regimpute_vs_pan_$(replace(stk,"|lu="=>"_","."=>"_")).png"); MK.save(out,fig); println("wrote $out")
end
println("=== DONE ===")
