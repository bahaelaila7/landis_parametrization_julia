# Fig 1: FL5-by-TreeMap species composition (eco×lu×age stacked bars, TONS).
# Fig 3a: FL5-by-TreeMap AGGREGATE per species (TONS).
# TreeMap pixel (30×30 m) → PLT_CN → FIA PLOT (statecd/county/plot/INVYR) → analysis plot-level cohorts
# (prepare_parametrization_data splots: plot-level per-acre biomass, tiered to the same 14 species).
# total tons = Σ_plots Σ_cohorts agb(lbs/acre) × npix × (900/4046.86 acre/pixel)/2000.
#   Run: ./julia_gdal.sh --project=. test/fig_fl5_treemap.jl runs/fl5_l4cover_mocmaes_Aonly_Seco.yml
using Pan, DataFrames, CairoMakie, DuckDB, Statistics, Dates, YAML
import ArchGDAL as AG
const MK = CairoMakie; const D = Pan.Data
const FIADB = "/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"
const TREEMAP = "/workspace/poster/fl5/FL5_22.tif"
const BINS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]
const TONS_PER_PIXEL = 900 / 1e6 * 1.10231   # agb is g/m² (LANDIS); × 900 m²/pixel → g → tonnes → short tons
cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)

# --- 1. TreeMap: Value→PLT_CN (RAT field 2, as string) + pixel count per PLT_CN (band1 histogram) ---
function parse_pltcn(aux)
  d = Dict{UInt32,String}()
  for ch in split(read(aux, String), "<Row index=")[2:end]
    fs = [m.captures[1] for m in eachmatch(r"<F>([^<]*)</F>", ch)]
    length(fs) >= 3 || continue
    v = tryparse(Int, fs[1]); cn = strip(fs[3])
    (v === nothing || isempty(cn)) && continue
    d[UInt32(v)] = String(cn)
  end
  d
end
val2cn = parse_pltcn(TREEMAP * ".aux.xml")
band = AG.read(AG.getband(AG.read(TREEMAP), 1))
cnt = Dict{UInt32,Int}(); for v in band; cnt[v] = get(cnt, v, 0) + 1; end
pcn = Dict{String,Int}()
for (v, c) in cnt; haskey(val2cn, v) && (pcn[val2cn[v]] = get(pcn, val2cn[v], 0) + c); end
println("FL5 distinct PLT_CN: ", length(pcn), "  forested pixels: ", sum(values(pcn)))

# --- 2. PLOT bridge: CN → (statecd,unitcd,countycd,plot,INVYR) ---
con = DBInterface.connect(DuckDB.DB(FIADB))
cns = collect(keys(pcn))
DuckDB.register_data_frame(con, DataFrame(cn=cns, npix=[pcn[c] for c in cns]), "fl5cn")
bridge = DataFrame(DBInterface.execute(con, """
  SELECT p.STATECD statecd, p.UNITCD unitcd, p.COUNTYCD countycd, p.PLOT plot, p.INVYR invyr, f.npix
  FROM PLOT p JOIN fl5cn f ON p.CN=f.cn"""))
println("PLOT bridge: ", nrow(bridge), "/", length(cns), " CNs → plot; states ", sort(unique(bridge.statecd)))
# --- 3. join curated DIRECTLY (national; 88% of FL5 pixels). TPA_UNADJ already carries the plot-level
#        (4-subplot) expansion in its denominator, so Σ cohort agb = plot per-acre density — no /nsub. ---
DuckDB.register_data_frame(con, bridge, "fl5pl")
fl5 = DataFrame(DBInterface.execute(con, """
  WITH fpix AS (SELECT statecd,unitcd,countycd,plot, SUM(npix) npix FROM fl5pl GROUP BY 1,2,3,4),
       recent AS (SELECT statecd,unitcd,countycd,plot, MAX(measdate) md FROM curated_cohorts_landis GROUP BY 1,2,3,4),
       stdorg AS (SELECT STATECD,UNITCD,COUNTYCD,PLOT,
                    CASE WHEN MAX(STDORGCD)=1 THEN 'artificial' ELSE 'natural' END lu
                  FROM COND WHERE STDORGCD IS NOT NULL GROUP BY 1,2,3,4)
  SELECT c.epa_l3 l3, s.lu lu, UPPER(TRIM(c.species_symbol)) sym, c.spgrpcd grp,
         UPPER(TRIM(c.sftwd_hrdwd)) sh, c.age_calc age, SUM(c.agb) agb, MAX(fpix.npix) npix
  FROM curated_cohorts_landis c
  JOIN recent r ON c.statecd=r.statecd AND c.unitcd=r.unitcd AND c.countycd=r.countycd AND c.plot=r.plot AND c.measdate=r.md
  JOIN fpix ON c.statecd=fpix.statecd AND c.unitcd=fpix.unitcd AND c.countycd=fpix.countycd AND c.plot=fpix.plot
  JOIN stdorg s ON c.statecd=s.STATECD AND c.unitcd=s.UNITCD AND c.countycd=s.COUNTYCD AND c.plot=s.PLOT
  WHERE c.agb>0
  GROUP BY c.statecd,c.unitcd,c.countycd,c.plot,c.epa_l3,s.lu,
           UPPER(TRIM(c.species_symbol)),c.spgrpcd,UPPER(TRIM(c.sftwd_hrdwd)),c.age_calc"""))
println("FL5 cohort rows: ", nrow(fl5), "  pixels represented: ", "(via npix)")
# tier to the analysis 14 species (exact 10 + _GRP_41/43 + _S/_H)
EXACT = Set(["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS"]); GRPS = Set([41,43])
tier(sym, grp, sh) = sym in EXACT ? sym : (grp in GRPS ? "_GRP_$grp" : "_" * (sh in ("S","H") ? sh : "H"))
fl5.eff = [tier(String(r.sym), Int(r.grp), String(r.sh)) for r in eachrow(fl5)]
fl5.tons = Float64.(fl5.agb) .* fl5.npix .* TONS_PER_PIXEL
# canonical 14-species list + eco×lu strata (donor plot's eco; restrict panels to the 4 analysis strata)
species_list = ["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS","_GRP_41","_GRP_43","_H","_S"]
fl5.stratum = String.(fl5.l3) .* " | " .* String.(fl5.lu)
const STRATA = ["8.3.5 | artificial", "8.3.5 | natural", "8.5.3 | artificial", "8.5.3 | natural"]

# --- 4. shared color/label scheme (soft blue→green, hard yellow→red; common names) ---
ref = DataFrame(DBInterface.execute(con, "SELECT UPPER(TRIM(SPECIES_SYMBOL)) sym, UPPER(TRIM(SFTWD_HRDWD)) sh, COMMON_NAME cn FROM REF_SPECIES"))
sym_sh = Dict(String(r.sym) => String(r.sh) for r in eachrow(ref) if !ismissing(r.sh))
common = Dict(String(r.sym) => String(r.cn) for r in eachrow(ref) if !ismissing(r.cn))
isoft(s) = (su = uppercase(strip(s)); su == "_S" ? true : su == "_H" ? false : startswith(su, "_GRP_") ? false : get(sym_sh, su, "H") == "S")
soft = [s for s in species_list if isoft(s)]; hard = [s for s in species_list if !isoft(s)]
sgrad = MK.cgrad([:navy, :dodgerblue, :darkturquoise, :seagreen, :limegreen]); hgrad = MK.cgrad([:gold, :orange, :orangered, :red, :darkred])
shade(i, n) = n <= 1 ? 0.5 : (i - 1) / (n - 1)
color_of = Dict{String,MK.RGBAf}()
for (i, s) in enumerate(soft); color_of[s] = MK.RGBAf(sgrad[shade(i, length(soft))]); end
for (i, s) in enumerate(hard); color_of[s] = MK.RGBAf(hgrad[shade(i, length(hard))]); end
stack_order = vcat(soft, hard)
function label_for(s)
  su = uppercase(strip(s)); startswith(su, "_GRP_") && return "Group $(s[6:end]) ($s)"
  su == "_H" && return "Other Hardwoods (_H)"; su == "_S" && return "Other Softwoods (_S)"
  cn = get(common, su, nothing); isnothing(cn) ? s : "$cn ($s)"
end
binidx(a) = (for (i, b) in enumerate(BINS); a < b && return i; end; length(BINS) + 1)
fl5.bin = binidx.(Int.(fl5.age)); nb = length(BINS) + 1
binlabels = vcat("<$(BINS[1])", ["$(BINS[i-1])–$(BINS[i])" for i in 2:length(BINS)], "≥$(BINS[end])")

# --- Fig 1: panels by eco×lu (4 analysis strata), stacked species×age (tons) ---
fig1 = MK.Figure(size = (1180, 860))
for (pidx, st) in enumerate(STRATA)
  row, col = (pidx - 1) ÷ 2 + 1, (pidx - 1) % 2 + 1
  ax = MK.Axis(fig1[row, col]; title = st, xticks = (1:nb, binlabels), xticklabelrotation = π/4,
    ylabel = "aggregate AGB (tons)", xlabel = "age (yr)")
  sub = fl5[fl5.stratum .== st, :]
  xs = Int[]; ys = Float64[]; stk = Int[]; cs = MK.RGBAf[]
  for (si, s) in enumerate(stack_order)
    ss = sub[sub.eff .== s, :]
    for b in 1:nb
      y = sum(ss.tons[ss.bin .== b]; init = 0.0); y > 0 || continue
      push!(xs, b); push!(ys, y); push!(stk, si); push!(cs, color_of[s])
    end
  end
  isempty(xs) || MK.barplot!(ax, xs, ys; stack = stk, color = cs); MK.xlims!(ax, 0.3, nb + 0.7)
end
MK.Legend(fig1[1:2, 3], [MK.PolyElement(color = color_of[s]) for s in stack_order],
  [label_for(s) for s in stack_order], "Species (soft→hard)"; framevisible = true)
MK.save("runs/fig1_fl5_treemap_panels_CORRECTED.png", fig1; px_per_unit = 3)

# --- Fig 3a: aggregate per species over ALL matched FL5 plots (tons) ---
totd = Dict(s => sum(fl5.tons[fl5.eff .== s]; init = 0.0) for s in species_list)
ord = [s for s in stack_order if get(totd, s, 0.0) > 0]
fig3 = MK.Figure(size = (860, 540))
ax = MK.Axis(fig3[1, 1]; xticks = (1:length(ord), [label_for(s) for s in ord]), xticklabelrotation = π/3,
  ylabel = "aggregate AGB (tons)", title = "FL5 by TreeMap — total live AGB by species")
MK.barplot!(ax, 1:length(ord), [totd[s] for s in ord]; color = [color_of[s] for s in ord])
MK.save("runs/fig3a_fl5_treemap_aggregate_CORRECTED.png", fig3; px_per_unit = 3)
println("FL5 total live AGB (TreeMap, all matched): ", round(Int, sum(fl5.tons)), " tons | in 4 strata: ",
        round(Int, sum(fl5.tons[in.(fl5.stratum, Ref(STRATA))])), " tons")
println("wrote runs/fig1_fl5_treemap_panels_CORRECTED.png + runs/fig3a_fl5_treemap_aggregate_CORRECTED.png")
