# Compare FVS vs Pan (candidate) 100-yr Sim-A projections: mean plot AGB (g/m²) by tiered-species × coarse
# age-bin, at 25/50/75/100 yr, over the COMMON plots. Grid: rows=horizons, cols={FVS, Pan, FVS−Pan}.
#   Run: ./julia_gdal.sh --project=. tools/compare_fvs_pan.jl <fvs_cohorts_100.csv> <pan_cohorts_100.csv> <out.png>
import CairoMakie, DataFrames, CSV, Statistics, DuckDB
const MK = CairoMakie; const DF = DataFrames
fvs = CSV.read(ARGS[1], DF.DataFrame); pan = CSV.read(ARGS[2], DF.DataFrame); outpng = ARGS[3]
# optional 6th arg: land-use filter ("natural" | "artificial") — restrict to plots of that STDORGCD class
const LU = length(ARGS) >= 6 ? String(ARGS[6]) : "all"
if LU in ("natural", "artificial")
  con = DuckDB.connect(DuckDB.DB())
  DuckDB.execute(con, "ATTACH '/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb' AS s (READ_ONLY);")
  m = DF.DataFrame(DuckDB.execute(con, """
    SELECT DISTINCT statecd||'_'||unitcd||'_'||countycd||'_'||plot AS plotkey, land_use
    FROM s.curated_cohorts_landis_stdorg
    WHERE epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f') AND land_use='$LU'"""))
  keep = Set(String.(m.plotkey))
  fvs = fvs[in.(string.(fvs.plotkey), Ref(keep)), :]; pan = pan[in.(string.(pan.plotkey), Ref(keep)), :]
  println("land-use=$LU → $(length(keep)) plots in that class")
end
# optional rep-stats CSVs (offset,mean_agb,std_agb,nreps): ARGS[4]=Pan, ARGS[5]=FVS → annotate Σ = mean±std
pan_rs = length(ARGS) >= 4 && isfile(ARGS[4]) ? CSV.read(ARGS[4], DF.DataFrame) : nothing
fvs_rs = length(ARGS) >= 5 && isfile(ARGS[5]) ? CSV.read(ARGS[5], DF.DataFrame) : nothing
stdnote(rs, off) = rs === nothing ? "" : (let r = rs[rs.offset .== off, :]; DF.nrow(r) == 0 ? "" : "  [$(Int(rs.nreps[1]))-rep: $(round(Int,r.mean_agb[1]))±$(round(Int,r.std_agb[1]))]" end)
nrep = pan_rs !== nothing ? Int(pan_rs.nreps[1]) : (fvs_rs !== nothing ? Int(fvs_rs.nreps[1]) : 0)
HORIZONS = [25, 50, 75, 100]; BINS = [10,20,30,40,50,60,80,100,120,150]; nb = length(BINS)+1
binlabel(i) = i == 1 ? "≤$(BINS[1])" : i <= length(BINS) ? "$(BINS[i-1])-$(BINS[i])" : "$(BINS[end])+"
# common plots (fair landscape mean)
common = intersect(Set(string.(fvs.plotkey)), Set(string.(pan.plotkey)))
n = length(common)
println("common plots = $n  (FVS $(length(unique(fvs.plotkey))), Pan $(length(unique(pan.plotkey))))")
fvs = fvs[in.(string.(fvs.plotkey), Ref(common)), :]; pan = pan[in.(string.(pan.plotkey), Ref(common)), :]
species = sort(unique(vcat(String.(fvs.eff), String.(pan.eff))))
spidx = Dict(s => i for (i, s) in enumerate(species)); ns = length(species)
# mean plot AGB per (offset, eff, agebin) = Σ agb over common plots / n
function mat(df, off)
  M = zeros(ns, nb)
  s = df[df.offset .== off, :]
  for r in eachrow(s); M[spidx[String(r.eff)], Int(r.agebin)] += Float64(r.agb); end
  M ./ n
end
FM = Dict(off => mat(fvs, off) for off in HORIZONS); PM = Dict(off => mat(pan, off) for off in HORIZONS)
vmax = maximum(maximum(vcat(vec(FM[o]), vec(PM[o]))) for o in HORIZONS)
dmax = maximum(maximum(abs.(FM[o] .- PM[o])) for o in HORIZONS)
println("AGB colour max=$(round(vmax)) g/m²; diff max=±$(round(dmax))")

fig = MK.Figure(size=(220 + 3*(90+18*nb), 80 + 4*(60+15*ns)))
MK.Label(fig[0, 1:4], "FVS vs Pan candidate-6 — mean plot AGB (g/m²) by species × age-bin — land-use: $(uppercase(LU)) ($n common plots)"; fontsize=15, font=:bold)
(pan_rs !== nothing || fvs_rs !== nothing) && MK.Label(fig[-1, 1:4],
  "Both FVS and Pan are means of $nrep stochastic replicates (FVS: distinct RANNSEEDs; Pan: reseeded establishment). Per-panel Σ shows [$nrep-rep landscape-mean±std g/m²]."; fontsize=11, color=:gray30)
seq = MK.cgrad(:viridis); div = MK.cgrad(:RdBu; rev=true)
hmref = Ref{Any}(nothing); dref = Ref{Any}(nothing)
for (ri, off) in enumerate(HORIZONS)
  for (ci, (M, lab, cm, cr, rs)) in enumerate([(FM[off],"FVS",seq,(0,vmax),fvs_rs), (PM[off],"Pan cand-6",seq,(0,vmax),pan_rs), (FM[off].-PM[off],"FVS − Pan",div,(-dmax,dmax),nothing)])
    ax = MK.Axis(fig[ri, ci]; title="yr $off — $lab", xlabel=(ri==4 ? "age bin" : ""), ylabel=(ci==1 ? "species" : ""),
      xticks=(1:nb, binlabel.(1:nb)), yticks=(1:ns, species), xticklabelrotation=π/4, xticklabelsize=7, yticklabelsize=7, titlesize=10)
    hm = MK.heatmap!(ax, 1:nb, 1:ns, permutedims(M); colormap=cm, colorrange=cr)
    ci <= 2 ? (hmref[] = hm) : (dref[] = hm)
    tot = round(Int, sum(M)); MK.text!(ax, 0.99, 0.01; text="Σ=$tot$(stdnote(rs, off))", space=:relative, align=(:right,:bottom), fontsize=8, color=:white)
  end
end
MK.Colorbar(fig[1:4, 4], hmref[]; label="AGB g/m² (FVS, Pan)")
MK.Colorbar(fig[1:4, 5], dref[]; label="FVS−Pan g/m²")
MK.save(outpng, fig); println("wrote $outpng")
