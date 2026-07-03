# 4 panels (eco × land-use) of species composition over age bins: stacked bars of aggregate AGB by the
# ANALYSIS species (tiered 14). Colors = softwood (blue→green) → hardwood (yellow→red), shaded within
# class, stacked soft-bottom→hard-top (scheme ported from Pan.jl plot_landis_species_composition).
#   Run: ./julia_gdal.sh --project=. test/fig_fl5_species_panels.jl runs/fl5_l4cover_mocmaes_Aonly_Seco.yml
using Pan, DataFrames, CairoMakie, YAML, DuckDB
const MK = CairoMakie; const D = Pan.Data
cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
const BINS = Int.(g("bins_idx", [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]))
const OUT = "runs/fig_fl5_species_panels_CORRECTED.png"

rng = Pan.RNGType(UInt64(Int(g("seed", 1))))
splots, eco_list, species_list, eco_species_ids, _ = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], filter_eco_field=String(g("filter_eco_field", "epa_l4")),
  eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]), output_dir=String(cfg["tablename"]),
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false, val_frac=0.0,
  min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
  single_ecoregion=Bool(g("single_ecoregion", false)), stratify_landuse=Bool(g("stratify_landuse", false)),
  filter_ecos=String.(get(cfg, "filter_ecos", String[])), RNG=rng)
println("plots:", length(unique(splots.plot_id)), " ecos:", length(eco_list), " species:", length(species_list))

# --- softwood/hardwood classification (REF_SPECIES + REF_SPECIES_GROUP) → blue→green / yellow→red ---
fcon = DBInterface.connect(DuckDB.DB(cfg["cohorts_db_path"]))
ref = DataFrame(DBInterface.execute(fcon, "SELECT UPPER(TRIM(SPECIES_SYMBOL)) sym, UPPER(TRIM(SFTWD_HRDWD)) sh, COMMON_NAME cn FROM REF_SPECIES"))
sym_sh = Dict(String(r.sym) => String(r.sh) for r in eachrow(ref) if !ismissing(r.sh))
common = Dict(String(r.sym) => String(r.cn) for r in eachrow(ref) if !ismissing(r.cn))
# label: "Common Name (SYM)"; _GRP_N → "Group N (_GRP_N)"; _H/_S → "Other Hardwoods/Softwoods (_H/_S)"
function label_for(s)
  su = uppercase(strip(s))
  startswith(su, "_GRP_") && return "Group $(s[6:end]) ($s)"
  su == "_H" && return "Other Hardwoods (_H)"
  su == "_S" && return "Other Softwoods (_S)"
  cn = get(common, su, nothing)
  isnothing(cn) ? s : "$cn ($s)"
end
grpclass = Dict{Int,String}()
try
  gg = DataFrame(DBInterface.execute(fcon, "SELECT SPGRPCD spgrpcd, UPPER(TRIM(CLASS)) class FROM REF_SPECIES_GROUP"))
  global grpclass = Dict(Int(r.spgrpcd) => String(r.class) for r in eachrow(gg))
catch; end
function is_soft(s)
  s = uppercase(strip(s))
  s == "_S" && return true; s == "_H" && return false
  if startswith(s, "_GRP_"); n = tryparse(Int, s[6:end]); return startswith(get(grpclass, something(n, -1), ""), "S"); end
  return get(sym_sh, s, "H") == "S"   # unknown → hardwood
end
soft = [s for s in species_list if is_soft(s)]
hard = [s for s in species_list if !is_soft(s)]
soft_grad = MK.cgrad([:navy, :dodgerblue, :darkturquoise, :seagreen, :limegreen])
hard_grad = MK.cgrad([:gold, :orange, :orangered, :red, :darkred])
shade(i, n) = n <= 1 ? 0.5 : (i - 1) / (n - 1)
color_of = Dict{String,MK.RGBAf}()
for (i, s) in enumerate(soft); color_of[s] = MK.RGBAf(soft_grad[shade(i, length(soft))]); end
for (i, s) in enumerate(hard); color_of[s] = MK.RGBAf(hard_grad[shade(i, length(hard))]); end
stack_order = vcat(soft, hard)                              # softwood bottom → hardwood top
spidx = Dict(s => i for (i, s) in enumerate(species_list))  # symbol → species_id
println("soft: ", soft, "\nhard: ", hard)

binidx(a) = (for (i, b) in enumerate(BINS); a < b && return i; end; length(BINS) + 1)
splots.bin = binidx.(Int.(splots.age_calc))
nb = length(BINS) + 1
binlabels = vcat("<$(BINS[1])", ["$(BINS[i-1])–$(BINS[i])" for i in 2:length(BINS)], "≥$(BINS[end])")
pretty(e) = replace(e, "|lu=" => " | ")

fig = MK.Figure(size = (1180, 860))
for (e, ename) in enumerate(eco_list)
  row, col = (e - 1) ÷ 2 + 1, (e - 1) % 2 + 1
  ax = MK.Axis(fig[row, col]; title = pretty(ename), xticks = (1:nb, binlabels),
    xticklabelrotation = π/4, ylabel = "aggregate AGB", xlabel = "age (yr)")
  sub = splots[splots.eco_id .== e, :]
  xs = Int[]; ys = Float64[]; stk = Int[]; cs = MK.RGBAf[]
  for (si, s) in enumerate(stack_order)
    ss = sub[sub.species_id .== spidx[s], :]
    for b in 1:nb
      y = sum(ss.agb_sum[ss.bin .== b]; init = 0.0)
      y > 0 || continue
      push!(xs, b); push!(ys, Float64(y)); push!(stk, si); push!(cs, color_of[s])
    end
  end
  isempty(xs) || MK.barplot!(ax, xs, ys; stack = stk, color = cs)
  MK.xlims!(ax, 0.3, nb + 0.7)
end
elems = [MK.PolyElement(color = color_of[s]) for s in stack_order]
MK.Legend(fig[1:2, 3], elems, [label_for(s) for s in stack_order], "Species (soft→hard)"; framevisible = true, nbanks = 1)
MK.save(OUT, fig; px_per_unit = 3)
println("wrote ", OUT)
