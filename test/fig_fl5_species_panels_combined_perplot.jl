# Combined species-composition grid: 4 rows × 3 columns.
#   Rows  = the 4 strata, grouped 2-per-EPA-L3 (artificial / natural), with a horizontal divider
#           between 8.3.5 and 8.5.3.
#   Cols  = three data sources for the SAME stacked species×age composition:
#             1) EPA FIA    — plot cohorts over the national footprint of the 4 study L4 codes
#             2) FL5 FIA    — the same, restricted to plots inside the FL5 TreeMap raster extent
#             3) FL5 TreeMap— TreeMap-pixel-weighted tons (imputed FL5 raster composition)
#   land_use = STDORGCD ground truth (curated_cohorts_landis_stdorg). Shared 14-species colour/legend.
#   Run: ./julia_gdal.sh --project=. test/fig_fl5_species_panels_combined.jl
using Pan, DataFrames, CairoMakie, DuckDB, YAML, Statistics
import ArchGDAL as AG
const MK = CairoMakie; const D = Pan.Data
const FIADB = "/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"
const TREEMAP = "/workspace/poster/fl5/FL5_22.tif"
const CFG = "runs/fl5_species_panels_stdorg.yml"                 # stdorg table + 4-strata knobs
const FL5_BOX = (lon0 = -83.42, lon1 = -81.982, lat0 = 29.758, lat1 = 30.699)  # FL5_22.tif 4326 bbox
const TONS_PER_PIXEL = 900 / 1e6 * 1.10231                       # g/m² × 900 m²/pixel → g → short tons
const OUT = "runs/fig_fl5_species_panels_combined_perplot.png"
cfg = YAML.load_file(CFG); g(k, d) = get(cfg, k, d)

const BINS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]; const nb = length(BINS) + 1
binidx(a) = (for (i, b) in enumerate(BINS); a < b && return i; end; length(BINS) + 1)
binlabels = vcat("<$(BINS[1])", ["$(BINS[i-1])–$(BINS[i])" for i in 2:length(BINS)], "≥$(BINS[end])")

# --- shared 14-species colour scheme + labels (soft blue→green, hard yellow→red) ---
const SPECIES = ["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS","_GRP_41","_GRP_43","_H","_S"]
con = DBInterface.connect(DuckDB.DB(FIADB))
ref = DataFrame(DBInterface.execute(con, "SELECT UPPER(TRIM(SPECIES_SYMBOL)) sym, UPPER(TRIM(SFTWD_HRDWD)) sh, COMMON_NAME cn FROM REF_SPECIES"))
sym_sh = Dict(String(r.sym) => String(r.sh) for r in eachrow(ref) if !ismissing(r.sh))
common = Dict(String(r.sym) => String(r.cn) for r in eachrow(ref) if !ismissing(r.cn))
grpclass = Dict{Int,String}()
try
  gg = DataFrame(DBInterface.execute(con, "SELECT SPGRPCD spgrpcd, UPPER(TRIM(CLASS)) class FROM REF_SPECIES_GROUP"))
  global grpclass = Dict(Int(r.spgrpcd) => String(r.class) for r in eachrow(gg))
catch; end
function is_soft(s)
  s = uppercase(strip(s)); s == "_S" && return true; s == "_H" && return false
  if startswith(s, "_GRP_"); n = tryparse(Int, s[6:end]); return startswith(get(grpclass, something(n, -1), ""), "S"); end
  return get(sym_sh, s, "H") == "S"
end
soft = [s for s in SPECIES if is_soft(s)]; hard = [s for s in SPECIES if !is_soft(s)]
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

# comp Dict keyed (l3, lu, species, bin) → aggregate value
parse_strat(en) = (p = split(String(en), "|lu="); (String(p[1]), String(p[2])))
function splots_comp(splots, eco_list, sp_list)
  d = Dict{Tuple{String,String,String,Int},Float64}()
  for r in eachrow(splots)
    (l3, lu) = parse_strat(eco_list[r.eco_id]); sym = String(sp_list[r.species_id]); b = binidx(Int(round(r.age_calc)))
    k = (l3, lu, sym, b); d[k] = get(d, k, 0.0) + Float64(r.agb_sum)
  end
  d
end

# --- Column 1: EPA FIA (national footprint of the 4 L4 codes) ---
splots1, eco1, spc1, _, _ = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], filter_eco_field=String(g("filter_eco_field", "epa_l4")),
  eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]), output_dir=String(cfg["tablename"]),
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false, val_frac=0.0,
  min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
  single_ecoregion=false, stratify_landuse=true, filter_ecos=String.(get(cfg, "filter_ecos", String[])),
  RNG=Pan.RNGType(UInt64(Int(g("seed", 1)))))
comp1 = splots_comp(splots1, eco1, spc1)
# plots per (l3,lu) stratum → per-plot normalization
splots_nplots(splots, eco_list) = Dict{Tuple{String,String},Int}(
  parse_strat(String(eco_list[e])) => length(unique(splots.plot_id[splots.eco_id .== e])) for e in eachindex(eco_list))
nplots1 = splots_nplots(splots1, eco1)
println("COL1 EPA FIA plots:", length(unique(splots1.plot_id)), " per-stratum:", nplots1)

# --- Column 2: FL5 FIA (plots inside the FL5 raster extent) ---
inbox = DataFrame(DBInterface.execute(con, """
  SELECT DISTINCT statecd, unitcd, countycd, plot FROM data_plot_eco
  WHERE CAST(lon AS DOUBLE) BETWEEN $(FL5_BOX.lon0) AND $(FL5_BOX.lon1)
    AND CAST(lat AS DOUBLE) BETWEEN $(FL5_BOX.lat0) AND $(FL5_BOX.lat1)"""))
fplots = [(Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot)) for r in eachrow(inbox)]
splots2, eco2, spc2, _, _ = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], filter_eco_field=String(g("filter_eco_field", "epa_l4")),
  eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]), output_dir=String(cfg["tablename"]),
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false, val_frac=0.0,
  min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
  single_ecoregion=false, stratify_landuse=true, filter_ecos=String.(get(cfg, "filter_ecos", String[])),
  filter_plots=fplots, RNG=Pan.RNGType(UInt64(Int(g("seed", 1)))))
comp2 = splots_comp(splots2, eco2, spc2)
nplots2 = splots_nplots(splots2, eco2)
println("COL2 FL5 FIA plots:", length(unique(splots2.plot_id)), " per-stratum:", nplots2)

# --- Column 3: FL5 TreeMap (pixel-weighted tons) ---
function parse_pltcn(aux)
  d = Dict{UInt32,String}()
  for ch in split(read(aux, String), "<Row index=")[2:end]
    fs = [m.captures[1] for m in eachmatch(r"<F>([^<]*)</F>", ch)]; length(fs) >= 3 || continue
    v = tryparse(Int, fs[1]); cn = strip(fs[3]); (v === nothing || isempty(cn)) && continue
    d[UInt32(v)] = String(cn)
  end
  d
end
val2cn = parse_pltcn(TREEMAP * ".aux.xml")
band = AG.read(AG.getband(AG.read(TREEMAP), 1)); cnt = Dict{UInt32,Int}(); for v in band; cnt[v] = get(cnt, v, 0) + 1; end
pcn = Dict{String,Int}(); for (v, c) in cnt; haskey(val2cn, v) && (pcn[val2cn[v]] = get(pcn, val2cn[v], 0) + c); end
cns = collect(keys(pcn))
DuckDB.register_data_frame(con, DataFrame(cn=cns, npix=[pcn[c] for c in cns]), "fl5cn")
bridge = DataFrame(DBInterface.execute(con, "SELECT p.STATECD statecd,p.UNITCD unitcd,p.COUNTYCD countycd,p.PLOT plot,p.INVYR invyr,f.npix FROM PLOT p JOIN fl5cn f ON p.CN=f.cn"))
DuckDB.register_data_frame(con, bridge, "fl5pl")
fl5 = DataFrame(DBInterface.execute(con, """
  WITH fpix AS (SELECT statecd,unitcd,countycd,plot, SUM(npix) npix FROM fl5pl GROUP BY 1,2,3,4),
       recent AS (SELECT statecd,unitcd,countycd,plot, MAX(measdate) md FROM curated_cohorts_landis_stdorg GROUP BY 1,2,3,4)
  SELECT c.statecd sc, c.unitcd uc, c.countycd cc, c.plot pl,
         c.epa_l3 l3, c.land_use lu, UPPER(TRIM(c.species_symbol)) sym, c.spgrpcd grp,
         UPPER(TRIM(c.sftwd_hrdwd)) sh, c.age_calc age, SUM(c.agb) agb, MAX(fpix.npix) npix
  FROM curated_cohorts_landis_stdorg c
  JOIN recent r ON c.statecd=r.statecd AND c.unitcd=r.unitcd AND c.countycd=r.countycd AND c.plot=r.plot AND c.measdate=r.md
  JOIN fpix ON c.statecd=fpix.statecd AND c.unitcd=fpix.unitcd AND c.countycd=fpix.countycd AND c.plot=fpix.plot
  WHERE c.agb>0
  GROUP BY c.statecd,c.unitcd,c.countycd,c.plot,c.epa_l3,c.land_use,UPPER(TRIM(c.species_symbol)),c.spgrpcd,UPPER(TRIM(c.sftwd_hrdwd)),c.age_calc"""))
const EXACT = Set(["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS"]); const GRPS = Set([41, 43])
tier(sym, grp, sh) = sym in EXACT ? sym : (grp in GRPS ? "_GRP_$grp" : "_" * (sh in ("S", "H") ? sh : "H"))
comp3 = Dict{Tuple{String,String,String,Int},Float64}()
for r in eachrow(fl5)
  eff = tier(String(r.sym), Int(r.grp), String(r.sh)); tons = Float64(r.agb) * r.npix * TONS_PER_PIXEL
  k = (String(r.l3), String(r.lu), eff, binidx(Int(round(r.age)))); comp3[k] = get(comp3, k, 0.0) + tons
end
# TreeMap tons already carry each plot's npix multiplier → normalize by Σ npix (tons/pixel), NOT unique-plot count.
nplots3 = Dict{Tuple{String,String},Int}(); npix3 = Dict{Tuple{String,String},Float64}()
for sub in groupby(fl5, [:l3, :lu])
  key = (String(sub.l3[1]), String(sub.lu[1]))
  seen = Dict{NTuple{4,Int},Int}()                      # distinct plot → its (constant) total npix
  for r in eachrow(sub); seen[(Int(r.sc), Int(r.uc), Int(r.cc), Int(r.pl))] = Int(r.npix); end
  nplots3[key] = length(seen); npix3[key] = Float64(sum(values(seen)))
end
println("COL3 FL5 TreeMap: per-stratum plots:", nplots3, "  Σnpix:", npix3)

# --- render 4×3 grid ---
STRATA = [("8.3.5", "artificial"), ("8.3.5", "natural"), ("8.5.3", "artificial"), ("8.5.3", "natural")]
# tuple = (header, comp, denom-dict, plot-count-dict). Divisor = plots for FIA cols, Σnpix for TreeMap.
COLS = [("EPA FIA  (Σ AGB/plot)", comp1, nplots1, nplots1),
        ("FL5 FIA  (Σ AGB/plot)", comp2, nplots2, nplots2),
        ("FL5 TreeMap  (tons/pixel)", comp3, npix3, nplots3)]
prow(i) = i <= 2 ? i + 1 : i + 2      # layout rows: 1=headers, 2-3=8.3.5, 4=divider, 5-6=8.5.3
fig = MK.Figure(size = (1550, 1300))
for (ci, (cname, _, _, _)) in enumerate(COLS); MK.Label(fig[1, ci+1], cname; fontsize = 18, font = :bold, tellwidth = false); end
colaxes = [MK.Axis[] for _ in eachindex(COLS)]
for (i, (l3, lu)) in enumerate(STRATA)
  r = prow(i)
  l3disp = l3 == "8.5.3" ? "8.5.3 (Coastal)" : l3   # distinguish from 8.3.5 (close numerals)
  MK.Label(fig[r, 1], "$l3disp  |  $lu"; rotation = π/2, fontsize = 16, font = :bold, tellheight = false)
  for (ci, (_, comp, denomdict, npd)) in enumerate(COLS)
    n = get(npd, (l3, lu), 0)                                    # plot count (title)
    dv = get(denomdict, (l3, lu), 0.0); denom = dv > 0 ? Float64(dv) : 1.0   # divisor: plots (FIA) or Σnpix (TreeMap)
    ax = MK.Axis(fig[r, ci+1]; xticks = (1:nb, binlabels), xticklabelrotation = π/4,
      ylabel = ci == 1 ? "AGB / plot" : "", title = "n=$n plots", titlesize = 11)
    xs = Int[]; ys = Float64[]; stk = Int[]; cs = MK.RGBAf[]
    for (si, s) in enumerate(stack_order), b in 1:nb
      y = get(comp, (l3, lu, s, b), 0.0) / denom; y > 0 || continue
      push!(xs, b); push!(ys, y); push!(stk, si); push!(cs, color_of[s])
    end
    isempty(xs) || MK.barplot!(ax, xs, ys; stack = stk, color = cs); MK.xlims!(ax, 0.3, nb + 0.7)
    (i == 2 || i == 4) ? (ax.xlabel = "age (yr)") : MK.hidexdecorations!(ax; grid = false)
    push!(colaxes[ci], ax)
  end
end
for ci in eachindex(COLS); MK.linkyaxes!(colaxes[ci]...); end   # share y within each column → per-plot magnitudes comparable across strata
MK.Box(fig[4, 2:4]; color = :black)                     # horizontal divider between the two L3 ecoregions
MK.rowsize!(fig.layout, 4, MK.Fixed(6))
MK.colsize!(fig.layout, 1, MK.Fixed(34))                # thin row-label column
MK.Legend(fig[2:6, 5], [MK.PolyElement(color = color_of[s]) for s in stack_order],
  [label_for(s) for s in stack_order], "Species (soft→hard)"; framevisible = true)
MK.save(OUT, fig; px_per_unit = 2)
println("wrote ", OUT)
