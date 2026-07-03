# Third FL5 species-composition grid (4 strata × 3 sources), like *_perplot.jl but with the FIA columns
# put on the SAME single-snapshot footing as TreeMap: each plot restricted to its measurement CLOSEST to
# 2022 within 2018–2024 (each plot sampled once — the ~2022 inventory cycle). This removes the
# all-measurements inflation that made the FIA columns ~2× TreeMap.
#   Cols: EPA FIA (national 4-L4) · FL5 FIA (FL5 raster extent) · FL5 TreeMap (tons/pixel).
#   Per-plot normalization: FIA ÷ unique plots; TreeMap ÷ Σnpix. y linked within each column.
#   Run: ./julia_gdal.sh --project=. test/fig_fl5_species_panels_combined_2022.jl
using DataFrames, CairoMakie, DuckDB, Statistics
import ArchGDAL as AG
const MK = CairoMakie
const DB = "/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"
const TREEMAP = "/workspace/poster/fl5/FL5_22.tif"
const TAB = "curated_cohorts_landis_stdorg"
const L4 = "('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f')"
const FL5_BOX = (lon0=-83.42, lon1=-81.982, lat0=29.758, lat1=30.699)
const TONS_PER_PIXEL = 900/1e6*1.10231
const BINS = [10,20,30,40,50,60,80,100,120,150]; const nb = length(BINS)+1
const OUT = "runs/fig_fl5_species_panels_combined_2022.png"
binidx(a) = (for (i,b) in enumerate(BINS); a<b && return i; end; length(BINS)+1)
binlabels = vcat("<$(BINS[1])", ["$(BINS[i-1])–$(BINS[i])" for i in 2:length(BINS)], "≥$(BINS[end])")
const SPECIES = ["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS","_GRP_41","_GRP_43","_H","_S"]
const EXACT = Set(["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS"]); const GRPS = Set([41,43])
tier(sym,grp,sh) = sym in EXACT ? sym : (grp in GRPS ? "_GRP_$grp" : "_"*(sh in ("S","H") ? sh : "H"))
fmtpx(x) = x >= 1e6 ? "$(round(x/1e6,digits=2))M" : "$(round(Int,x/1e3))k"   # pixel-count formatter

# in-memory main + ATTACH read-only (registrations land in memory; never touches the file)
con = DuckDB.connect(DuckDB.DB())
DuckDB.execute(con, "ATTACH '$DB' AS src (READ_ONLY);")

# --- colours: REF_SPECIES softwood/hardwood → blue→green / yellow→red ---
ref = DataFrame(DuckDB.execute(con, "SELECT UPPER(TRIM(SPECIES_SYMBOL)) sym, UPPER(TRIM(SFTWD_HRDWD)) sh, COMMON_NAME cn FROM src.REF_SPECIES"))
sym_sh = Dict(String(r.sym)=>String(r.sh) for r in eachrow(ref) if !ismissing(r.sh))
common = Dict(String(r.sym)=>String(r.cn) for r in eachrow(ref) if !ismissing(r.cn))
grpclass = Dict{Int,String}()
try
  gg = DataFrame(DuckDB.execute(con, "SELECT SPGRPCD spgrpcd, UPPER(TRIM(CLASS)) class FROM src.REF_SPECIES_GROUP"))
  global grpclass = Dict(Int(r.spgrpcd)=>String(r.class) for r in eachrow(gg))
catch; end
function is_soft(s)
  s = uppercase(strip(s)); s=="_S" && return true; s=="_H" && return false
  if startswith(s,"_GRP_"); n=tryparse(Int,s[6:end]); return startswith(get(grpclass,something(n,-1),""),"S"); end
  return get(sym_sh,s,"H")=="S"
end
soft = [s for s in SPECIES if is_soft(s)]; hard = [s for s in SPECIES if !is_soft(s)]
sgrad = MK.cgrad([:navy,:dodgerblue,:darkturquoise,:seagreen,:limegreen]); hgrad = MK.cgrad([:gold,:orange,:orangered,:red,:darkred])
shade(i,n) = n<=1 ? 0.5 : (i-1)/(n-1)
color_of = Dict{String,MK.RGBAf}()
for (i,s) in enumerate(soft); color_of[s]=MK.RGBAf(sgrad[shade(i,length(soft))]); end
for (i,s) in enumerate(hard); color_of[s]=MK.RGBAf(hgrad[shade(i,length(hard))]); end
stack_order = vcat(soft, hard)
function label_for(s)
  su=uppercase(strip(s)); startswith(su,"_GRP_") && return "Group $(s[6:end]) ($s)"
  su=="_H" && return "Other Hardwoods (_H)"; su=="_S" && return "Other Softwoods (_S)"
  cn=get(common,su,nothing); isnothing(cn) ? s : "$cn ($s)"
end

# comp (l3,lu,species,bin)→value + per-stratum plot counts, from a FIA cohort DataFrame (per-plot, 2022-restricted)
function fia_comp(df)
  d = Dict{Tuple{String,String,String,Int},Float64}(); np = Dict{Tuple{String,String},Set{NTuple{4,Int}}}()
  for r in eachrow(df)
    eff = tier(String(r.sym), Int(r.grp), String(r.sh)); b = binidx(Int(round(r.age)))
    k = (String(r.l3), String(r.lu), eff, b); d[k] = get(d,k,0.0) + Float64(r.agb)
    pk = (String(r.l3), String(r.lu)); push!(get!(np, pk, Set{NTuple{4,Int}}()), (Int(r.sc),Int(r.uc),Int(r.cc),Int(r.pl)))
  end
  d, Dict(k=>length(v) for (k,v) in np)
end

# 2022-restricted per-plot cohorts: each plot → its measurement CLOSEST to 2022 within 2018–2024.
fia_sql(extra) = """
  WITH pm AS (
    SELECT statecd,unitcd,countycd,plot, arg_min(measdate, abs(year(measdate)-2022)) chosen
    FROM src.$TAB WHERE epa_l4 IN $L4 AND year(measdate) BETWEEN 2018 AND 2024 $extra
    GROUP BY 1,2,3,4)
  SELECT c.statecd sc, c.unitcd uc, c.countycd cc, c.plot pl, c.epa_l3 l3, c.land_use lu,
         UPPER(TRIM(c.species_symbol)) sym, c.spgrpcd grp, UPPER(TRIM(c.sftwd_hrdwd)) sh,
         c.age_calc age, SUM(c.agb) agb
  FROM src.$TAB c JOIN pm ON c.statecd=pm.statecd AND c.unitcd=pm.unitcd AND c.countycd=pm.countycd
       AND c.plot=pm.plot AND c.measdate=pm.chosen
  WHERE c.agb>0
  GROUP BY 1,2,3,4,5,6,UPPER(TRIM(c.species_symbol)),c.spgrpcd,UPPER(TRIM(c.sftwd_hrdwd)),c.age_calc"""

# Col1: EPA FIA (national footprint of the 4 L4 codes)
comp1, nplots1 = fia_comp(DataFrame(DuckDB.execute(con, fia_sql(""))))
println("COL1 plots: ", nplots1)

# Col2: FL5 FIA (plots inside the FL5 raster extent)
inbox = DataFrame(DuckDB.execute(con, """
  SELECT DISTINCT statecd,unitcd,countycd,plot FROM src.data_plot_eco
  WHERE CAST(lon AS DOUBLE) BETWEEN $(FL5_BOX.lon0) AND $(FL5_BOX.lon1)
    AND CAST(lat AS DOUBLE) BETWEEN $(FL5_BOX.lat0) AND $(FL5_BOX.lat1)"""))
DuckDB.register_data_frame(con, inbox, "fl5box")
extra2 = " AND (statecd,unitcd,countycd,plot) IN (SELECT statecd,unitcd,countycd,plot FROM fl5box)"
comp2, nplots2 = fia_comp(DataFrame(DuckDB.execute(con, fia_sql(extra2))))
println("COL2 plots: ", nplots2)

# Col3: FL5 TreeMap (pixel-weighted tons/pixel), single-snapshot recent measurement.
function parse_pltcn(aux)
  d = Dict{UInt32,String}()
  for ch in split(read(aux,String), "<Row index=")[2:end]
    fs=[m.captures[1] for m in eachmatch(r"<F>([^<]*)</F>", ch)]; length(fs)>=3 || continue
    v=tryparse(Int,fs[1]); cn=strip(fs[3]); (v===nothing||isempty(cn)) && continue; d[UInt32(v)]=String(cn)
  end; d
end
val2cn = parse_pltcn(TREEMAP*".aux.xml")
band = AG.read(AG.getband(AG.read(TREEMAP),1)); cnt=Dict{UInt32,Int}(); for v in band; cnt[v]=get(cnt,v,0)+1; end
pcn = Dict{String,Int}(); for (v,c) in cnt; haskey(val2cn,v) && (pcn[val2cn[v]]=get(pcn,val2cn[v],0)+c); end
cns = collect(keys(pcn))
DuckDB.register_data_frame(con, DataFrame(cn=cns, npix=[pcn[c] for c in cns]), "fl5cn")
bridge = DataFrame(DuckDB.execute(con, "SELECT p.STATECD statecd,p.UNITCD unitcd,p.COUNTYCD countycd,p.PLOT plot,f.npix FROM src.PLOT p JOIN fl5cn f ON p.CN=f.cn"))
DuckDB.register_data_frame(con, bridge, "fl5pl")
fl5 = DataFrame(DuckDB.execute(con, """
  WITH fpix AS (SELECT statecd,unitcd,countycd,plot, SUM(npix) npix FROM fl5pl GROUP BY 1,2,3,4),
       recent AS (SELECT statecd,unitcd,countycd,plot, MAX(measdate) md FROM src.$TAB GROUP BY 1,2,3,4)
  SELECT c.statecd sc,c.unitcd uc,c.countycd cc,c.plot pl, c.epa_l3 l3, c.land_use lu,
         UPPER(TRIM(c.species_symbol)) sym, c.spgrpcd grp, UPPER(TRIM(c.sftwd_hrdwd)) sh,
         c.age_calc age, SUM(c.agb) agb, MAX(fpix.npix) npix
  FROM src.$TAB c
  JOIN recent r ON c.statecd=r.statecd AND c.unitcd=r.unitcd AND c.countycd=r.countycd AND c.plot=r.plot AND c.measdate=r.md
  JOIN fpix ON c.statecd=fpix.statecd AND c.unitcd=fpix.unitcd AND c.countycd=fpix.countycd AND c.plot=fpix.plot
  WHERE c.agb>0
  GROUP BY 1,2,3,4,5,6,UPPER(TRIM(c.species_symbol)),c.spgrpcd,UPPER(TRIM(c.sftwd_hrdwd)),c.age_calc"""))
comp3 = Dict{Tuple{String,String,String,Int},Float64}(); nplots3 = Dict{Tuple{String,String},Int}(); npix3 = Dict{Tuple{String,String},Float64}()
for sub in groupby(fl5,[:l3,:lu])
  key=(String(sub.l3[1]),String(sub.lu[1])); seen=Dict{NTuple{4,Int},Int}()
  for r in eachrow(sub); seen[(Int(r.sc),Int(r.uc),Int(r.cc),Int(r.pl))]=Int(r.npix); end
  nplots3[key]=length(seen); npix3[key]=Float64(sum(values(seen)))
end
for r in eachrow(fl5)
  # keep AGB DENSITY (g/m²): agb×npix, then ÷Σnpix below = npix-weighted mean plot density — SAME unit as
  # the FIA columns (mean per-plot AGB, g/m²), so all three columns are directly comparable. (No tons_per_pixel.)
  eff=tier(String(r.sym),Int(r.grp),String(r.sh)); dens=Float64(r.agb)*r.npix
  k=(String(r.l3),String(r.lu),eff,binidx(Int(round(r.age)))); comp3[k]=get(comp3,k,0.0)+dens
end
println("COL3 TreeMap plots: ", nplots3)

# --- render 4×3 ---
STRATA = [("8.3.5","artificial"),("8.3.5","natural"),("8.5.3","artificial"),("8.5.3","natural")]
COLS = [("EPA FIA  (Σ AGB/plot, 2022)", comp1, nplots1, nplots1),
        ("FL5 FIA  (Σ AGB/plot, 2022)", comp2, nplots2, nplots2),
        ("FL5 TreeMap  (AGB/plot, g/m², npix-wt)", comp3, npix3, nplots3)]
prow(i) = i<=2 ? i+1 : i+2
fig = MK.Figure(size=(1550,1300))
MK.Label(fig[0,1:5], "FL5 species composition — FIA restricted to the ~2022 cycle (each plot once, 2018–2024 nearest-2022)"; fontsize=16, font=:bold)
for (ci,(cname,_,_,_)) in enumerate(COLS); MK.Label(fig[1,ci+1], cname; fontsize=18, font=:bold, tellwidth=false); end
rowaxes = [MK.Axis[] for _ in eachindex(STRATA)]   # per-stratum-ROW axes → link the 3 sources within a row, each row its own scale
for (i,(l3,lu)) in enumerate(STRATA)
  r = prow(i); l3disp = l3=="8.5.3" ? "8.5.3 (Coastal)" : l3
  MK.Label(fig[r,1], "$l3disp  |  $lu"; rotation=π/2, fontsize=16, font=:bold, tellheight=false)
  for (ci,(_,comp,denomdict,npd)) in enumerate(COLS)
    n = get(npd,(l3,lu),0); dv = get(denomdict,(l3,lu),0.0); denom = dv>0 ? Float64(dv) : 1.0
    ttl = ci==3 ? "n=$(fmtpx(dv)) px (unique=$n)" : "n=$n plots"   # TreeMap: total landscape pixels + unique donor plots
    ax = MK.Axis(fig[r,ci+1]; xticks=(1:nb,binlabels), xticklabelrotation=π/4, ylabel = ci==1 ? "AGB / plot" : "", title=ttl, titlesize=11)
    xs=Int[]; ys=Float64[]; stk=Int[]; cs=MK.RGBAf[]
    for (si,s) in enumerate(stack_order), b in 1:nb
      y = get(comp,(l3,lu,s,b),0.0)/denom; y>0 || continue
      push!(xs,b); push!(ys,y); push!(stk,si); push!(cs,color_of[s])
    end
    isempty(xs) || MK.barplot!(ax, xs, ys; stack=stk, color=cs); MK.xlims!(ax, 0.3, nb+0.7)
    (i==2||i==4) ? (ax.xlabel="age (yr)") : MK.hidexdecorations!(ax; grid=false)
    push!(rowaxes[i], ax)
  end
end
MK.Box(fig[4,2:4]; color=:black); MK.rowsize!(fig.layout, 4, MK.Fixed(6)); MK.colsize!(fig.layout, 1, MK.Fixed(34))
MK.Legend(fig[2:6,5], [MK.PolyElement(color=color_of[s]) for s in stack_order], [label_for(s) for s in stack_order], "Species (soft→hard)"; framevisible=true)
for i in eachindex(STRATA); MK.linkyaxes!(rowaxes[i]...); end   # per-row link: 3 sources comparable within a stratum; artificial not squished by natural
MK.save(OUT, fig; px_per_unit=2)
println("wrote ", OUT)
