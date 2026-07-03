# Re-plot the 6 candidate-6 FVS-vs-Pan heatmaps with a COMMON biomass scale, in SHORT TONS/ACRE.
#  - AGB (FVS, Pan) colour: shared (0 .. global vmax) across ALL 6 figures.
#  - error (FVS-Pan) colour: fixed ±(1000 g/m²) = ±4.461 short tons/acre, centred on 0.
# One process: reads cached CSVs (no re-sim), renders all 6. Overwrites the existing PNGs.
import CairoMakie, DataFrames, CSV, Statistics, DuckDB
const MK = CairoMakie; const DF = DataFrames

const GM2_PER_TPA = 2000.0 * 453.592 / 4046.86   # 224.16985 g/m² per short-ton/acre (matches fvs_compare100.jl)
tpa(x) = x / GM2_PER_TPA                          # g/m² → short tons/acre
const DMAX = 1000.0 / GM2_PER_TPA                 # error range: ±1000 g/m² expressed in tons/acre (±4.461)
const VMIN = 0.1                                  # log-scale floor (t/ac); true-zero cells drawn as "absent" gray

const C6 = "runs/fl5_l4cover_mocmaes_simA_wsf01_longpin_stdorg_outputs/candidates/candidate_6"
const CONFIGS = [
  (tag="AUTOES",   fvs="tmp/fvs_compare100_auto/fvs_cohorts_100_auto.csv",     pan="$C6/pan_cohorts_100_autoes.csv", rs="tmp/fvs_compare100_auto/fvs_repstats_auto.csv"),
  (tag="NOAUTOES", fvs="tmp/fvs_compare100_noauto/fvs_cohorts_100_noauto.csv", pan="$C6/pan_cohorts_100.csv",         rs="tmp/fvs_compare100_noauto/fvs_repstats_noauto.csv"),
]
const LUS = ["all", "natural", "artificial"]
const HORIZONS = [25, 50, 75, 100]
const BINS = [10,20,30,40,50,60,80,100,120,150]; const nb = length(BINS)+1
binlabel(i) = i == 1 ? "≤$(BINS[1])" : i <= length(BINS) ? "$(BINS[i-1])-$(BINS[i])" : "$(BINS[end])+"

# plots of a given land-use class (nothing = all)
function lu_keep(lu)
  lu == "all" && return nothing
  con = DuckDB.connect(DuckDB.DB())
  DuckDB.execute(con, "ATTACH '/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb' AS s (READ_ONLY);")
  m = DF.DataFrame(DuckDB.execute(con, """
    SELECT DISTINCT statecd||'_'||unitcd||'_'||countycd||'_'||plot AS plotkey
    FROM s.curated_cohorts_landis_stdorg
    WHERE epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f') AND land_use='$lu'"""))
  Set(String.(m.plotkey))
end

# build per-(config,lu): FVS & Pan mean-plot-AGB matrices (tons/acre), species axis, n, repstats
function build(cfg, lu)
  fvs = CSV.read(cfg.fvs, DF.DataFrame); pan = CSV.read(cfg.pan, DF.DataFrame)
  keep = lu_keep(lu)
  if keep !== nothing
    fvs = fvs[in.(string.(fvs.plotkey), Ref(keep)), :]; pan = pan[in.(string.(pan.plotkey), Ref(keep)), :]
  end
  common = intersect(Set(string.(fvs.plotkey)), Set(string.(pan.plotkey))); n = length(common)
  fvs = fvs[in.(string.(fvs.plotkey), Ref(common)), :]; pan = pan[in.(string.(pan.plotkey), Ref(common)), :]
  species = sort(unique(vcat(String.(fvs.eff), String.(pan.eff))))
  spidx = Dict(s => i for (i, s) in enumerate(species)); ns = length(species)
  matf(df, off) = (M = zeros(ns, nb); s = df[df.offset .== off, :];
                   for r in eachrow(s); M[spidx[String(r.eff)], Int(r.agebin)] += tpa(Float64(r.agb)); end; M ./ n)
  FM = Dict(off => matf(fvs, off) for off in HORIZONS); PM = Dict(off => matf(pan, off) for off in HORIZONS)
  rs = isfile(cfg.rs) ? CSV.read(cfg.rs, DF.DataFrame) : nothing
  (FM=FM, PM=PM, n=n, species=species, ns=ns, rs=rs)
end

data = Dict{Tuple{String,String},Any}()
for cfg in CONFIGS, lu in LUS
  data[(cfg.tag, lu)] = build(cfg, lu)
  println("built $(cfg.tag)/$lu : n=$(data[(cfg.tag,lu)].n) plots, $(data[(cfg.tag,lu)].ns) species")
end
vmax_raw = maximum(maximum(vcat(vec(d.FM[o]), vec(d.PM[o]))) for d in values(data) for o in HORIZONS)
vmax = ceil(vmax_raw / 5) * 5   # round up to a clean tons/acre value
println("GLOBAL AGB vmax = $(round(vmax_raw,digits=2)) → $(vmax) short tons/acre ; error fixed ±$(round(DMAX,digits=3)) tons/acre (=±1000 g/m²)")

outname(tag, lu) = lu == "all" ?
  (tag == "AUTOES" ? "fvs_vs_pan_agb_species_agebin_AUTOES.png" : "fvs_vs_pan_agb_species_agebin.png") :
  "fvs_vs_pan_$(tag)_$(lu).png"

function render(tag, lu, d, out)
  seq = MK.cgrad(:viridis); div = MK.cgrad(:RdBu; rev=true)
  stdnote(rs, off) = rs === nothing ? "" : (let r = rs[rs.offset .== off, :]
      DF.nrow(r) == 0 ? "" : "  [$(Int(rs.nreps[1]))-rep: $(round(tpa(r.mean_agb[1]),digits=1))±$(round(tpa(r.std_agb[1]),digits=1))]" end)
  fig = MK.Figure(size=(240 + 3*(90+18*nb), 90 + 4*(60+15*d.ns)))
  MK.Label(fig[0, 1:4], "FVS vs Pan cand-6 [$tag] — mean plot AGB (short tons/acre) by species × age-bin — land-use: $(uppercase(lu)) ($(d.n) common plots)"; fontsize=14, font=:bold)
  MK.Label(fig[-1, 1:4], "Shared LOG AGB scale $(VMIN)–$(round(Int,vmax)) t/ac across all 6 figures (gray = truly absent, 0 t/ac) · error (FVS−Pan) linear, fixed ±$(round(DMAX,digits=2)) t/ac (=±1000 g/m²), centred 0"; fontsize=10, color=:gray30)
  hmref = Ref{Any}(nothing); dref = Ref{Any}(nothing)
  for (ri, off) in enumerate(HORIZONS)
    for (ci, (M, lab, cm, cr, rs)) in enumerate([(d.FM[off],"FVS",seq,(0,vmax),d.rs), (d.PM[off],"Pan cand-6",seq,(0,vmax),nothing), (d.FM[off].-d.PM[off],"FVS − Pan",div,(-DMAX,DMAX),nothing)])
      ax = MK.Axis(fig[ri, ci]; title="yr $off — $lab", xlabel=(ri==4 ? "age bin" : ""), ylabel=(ci==1 ? "species" : ""),
        xticks=(1:nb, binlabel.(1:nb)), yticks=(1:d.ns, d.species), xticklabelrotation=π/4, xticklabelsize=7, yticklabelsize=7, titlesize=10)
      if ci <= 2   # absolute AGB: shared LOG scale; true-zero → NaN → gray "absent"
        Mp = map(x -> x <= 0 ? NaN : x, M)
        hm = MK.heatmap!(ax, 1:nb, 1:d.ns, permutedims(Mp); colormap=cm, colorscale=log10, colorrange=(VMIN, vmax), nan_color=:gray85)
        hmref[] = hm
      else         # FVS − Pan: linear, fixed ±DMAX
        hm = MK.heatmap!(ax, 1:nb, 1:d.ns, permutedims(M); colormap=cm, colorrange=cr)
        dref[] = hm
      end
      tot = round(sum(M), digits=1); MK.text!(ax, 0.99, 0.01; text="Σ=$tot$(stdnote(rs, off))", space=:relative, align=(:right,:bottom), fontsize=8, color=:white)
    end
  end
  MK.Colorbar(fig[1:4, 4], hmref[]; label="AGB (short tons/acre, log; gray=0)")
  MK.Colorbar(fig[1:4, 5], dref[]; label="FVS−Pan (short tons/acre), ±$(round(DMAX,digits=2))")
  MK.save(out, fig); println("wrote $out")
end

for cfg in CONFIGS, lu in LUS
  render(cfg.tag, lu, data[(cfg.tag, lu)], joinpath(C6, outname(cfg.tag, lu)))
end
println("=== RESCALE DONE ===")
