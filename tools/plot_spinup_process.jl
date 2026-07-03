# Spinup-process figure (as defined by the `spinup` flag). For ONE plot we run the REAL spinup_cohorts!:
# it goes back in time to the oldest cohort's birth year, plants a young (age-1) cohort each time the clock
# hits an observed cohort's birth year, and grows everything with model dynamics until sim_year 0 — then the
# simulation continues forward to the plot's LAST measurement year. We capture the per-year stand via the
# BSP.SPINUP_CAPTURE hook (back-cast, sim_year < 0) + our own forward loop (sim_year ≥ 0), and draw each
# cohort's AGB trajectory vs sim_year, coloured by species, with observed cohorts overlaid as dots and a
# DASHED VERTICAL LINE at sim_year = 0 marking the end of the spinup (back-cast) years.
#   Run: [PAN_PLOT=409] [PAN_PARAMS=params.jld2] ./julia_gdal.sh --project=. tools/plot_spinup_process.jl <config.yml>
using Pan
import JLD2, YAML, CairoMakie, DataFrames, DuckDB, Setfield
const MK = CairoMakie; const P = Pan; const D = P.Data; const BSP = P.BiomassSuccessionPlugin; const DF = DataFrames

cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
D.USE_FIA_CYCLE[] = Bool(g("fia_cycle", false))
P.INIT_PERTURB_FRAC[] = 0.0
rng = P.RNGType(UInt64(Int(g("seed", 1))))
val_frac = Float64(g("val_frac", 0.0)); split_rng = val_frac > 0 ? P.RNGType(UInt64(Int(g("split_seed", 42)))) : nothing
splots, eco_list, species_list, eco_species_ids, splots_val = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], filter_eco_field=String(g("filter_eco_field", "epa_l4")),
  eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]), output_dir=String(cfg["tablename"]),
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false, val_frac=val_frac, split_rng=split_rng,
  min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
  stratify_eco_mixed=Bool(g("stratify_eco_mixed", false)),
  single_ecoregion=Bool(g("single_ecoregion", false)), stratify_landuse=Bool(g("stratify_landuse", false)),
  filter_extent=(haskey(cfg, "filter_extent") ? String(cfg["filter_extent"]) : nothing),
  filter_ecos=String.(get(cfg, "filter_ecos", String[])),
  filter_plots=NTuple{4,Int}[NTuple{4,Int}(Int.(p)) for p in get(cfg, "filter_plots", [])], RNG=rng)
best = if haskey(ENV, "PAN_PARAMS")
  JLD2.load_object(ENV["PAN_PARAMS"])
else
  st = JLD2.load_object(joinpath(cfg["output_dir"], "search_state_latest.jld2"))
  hasproperty(st, :representative) ? st.representative.x : st.best.x
end
SMF = parse(Float32, get(ENV, "PAN_SPINUP_MORT", "0.0"))    # spinup-only extra age-mortality (0 disables it; 0.15 = old default)
best = Setfield.@set best.SPINUP_MORTALITY_FRACTION = SMF
println("SPINUP_MORTALITY_FRACTION = ", best.SPINUP_MORTALITY_FRACTION)
sp = splots                                                # train-only (matches the other Sim-A/B figures)

# --- canonical species colour scheme (ported from the trajectory script) ---
let fcon = DuckDB.connect(DuckDB.DB(cfg["cohorts_db_path"]))
  ref = DF.DataFrame(DuckDB.execute(fcon, "SELECT UPPER(TRIM(SPECIES_SYMBOL)) sym, UPPER(TRIM(SFTWD_HRDWD)) sh FROM REF_SPECIES"))
  global sym_sh = Dict(String(r.sym) => String(r.sh) for r in eachrow(ref) if !ismissing(r.sh))
  global grpclass = Dict{Int,String}()
  try
    gg = DF.DataFrame(DuckDB.execute(fcon, "SELECT SPGRPCD spgrpcd, UPPER(TRIM(CLASS)) class FROM REF_SPECIES_GROUP"))
    global grpclass = Dict(Int(r.spgrpcd) => String(r.class) for r in eachrow(gg))
  catch; end
end
function is_soft(s)
  s = uppercase(strip(s)); s == "_S" && return true; s == "_H" && return false
  startswith(s, "_GRP_") && (n = tryparse(Int, s[6:end]); return startswith(get(grpclass, something(n, -1), ""), "S"))
  return get(sym_sh, s, "H") == "S"
end
soft = [s for s in species_list if is_soft(s)]; hard = [s for s in species_list if !is_soft(s)]
soft_grad = MK.cgrad([:navy, :dodgerblue, :darkturquoise, :seagreen, :limegreen])
hard_grad = MK.cgrad([:gold, :orange, :orangered, :red, :darkred])
shade(i, n) = n <= 1 ? 0.5 : (i - 1) / (n - 1)
color_of = Dict{String,MK.RGBAf}()
for (i, s) in enumerate(soft); color_of[s] = MK.RGBAf(soft_grad[shade(i, length(soft))]); end
for (i, s) in enumerate(hard); color_of[s] = MK.RGBAf(hard_grad[shade(i, length(hard))]); end

# --- pick the plot + its observed cohorts (with species) ---
obs_sp = DF.combine(DF.groupby(sp, [:plot_id, :sim_year, :eco_species_id, :age_calc]), :agb_sum => sum => :agb)
plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(DF.select(sp, [:plot_id, :eco_id]))))
LU = get(ENV, "PAN_LU", "natural")                            # land-use filter for auto-selection (eco name substring)
function pick_plot()
  bp = 0; bn = -1
  for p in unique(obs_sp.plot_id)
    occursin(LU, eco_list[plot2eco[p]]) || continue           # only plots in a `LU` land-use stratum
    oy = DF.subset(obs_sp, :plot_id => DF.ByRow(==(p))); ys = unique(oy.sim_year)
    (length(ys) < 2 || minimum(ys) != 0) && continue
    n = DF.nrow(DF.subset(oy, :sim_year => DF.ByRow(==(maximum(ys))))); n > bn && (bn = n; bp = p)
  end
  @assert bp != 0 "no plot matching land-use '$LU' with a seed year + later measurement"
  bp
end
pid = Int(haskey(ENV, "PAN_PLOT") ? parse(Int, ENV["PAN_PLOT"]) : pick_plot())   # plot_id col is UInt32; getsite wants Int
ecop = plot2eco[pid]
loc2name(loc) = species_list[eco_species_ids[ecop][loc]]      # local per-eco species → global name
colsp(loc) = color_of[loc2name(loc)]
oy = DF.subset(obs_sp, :plot_id => DF.ByRow(==(pid)))
last_meas = maximum(oy.sim_year)
println("plot #$pid ($(eco_list[ecop])): observed years = $(sort(unique(oy.sim_year))), spinup→$(last_meas)")

# --- run the REAL spinup with the per-year capture hook, then the forward sim to last_meas ---
snaps = Tuple{Int,Vector{Tuple{Int,Float64,Float64}}}[]       # (sim_year, [(species_loc, age, bio)...]) for plot pid
record!(yr, soa) = (s = P.getsite(soa, pid);                  # mapcode == site index invariant holds in spinup
  push!(snaps, (Int(yr), [(Int(s.c_species[j]), Float64(s.c_age[j]), Float64(s.c_bio[j])) for j in 1:Int(s.live)])))
eco_params = BSP.generate_eco_params(best)
soa = P.make_sites(sp, eco_species_ids; rng=P.RNGType(1), spinup=true, no_establishment=false)
spinup_all = D.get_spinup_cohorts(sp)
BSP.SPINUP_CAPTURE[] = record!                                # capture the back-cast years (sim_year < 0)
soa = BSP.spinup_cohorts!(soa, spinup_all, eco_params)
BSP.SPINUP_CAPTURE[] = nothing
for y in 0:last_meas                                          # forward sim (sim_year ≥ 0), as in fit_params
  P.PanCore.process_plugin!(soa, BSP.BiomassSuccession, y; ctx=(eco_params=eco_params,))
  record!(y, soa)
end
# spinup_cohorts! runs ALL plots together, so it starts at the GLOBAL oldest birth year — but THIS plot's
# site is empty until its own oldest cohort is planted. Use plot #pid's first non-empty year for the axis.
glob0 = minimum(first.(snaps))
yr0 = minimum(yr for (yr, cohs) in snaps if !isempty(cohs))
println("global spinup start=$(glob0); plot #$pid first cohort at sim_year $(yr0); captured ..$(last_meas)  ($(length(snaps)) snapshots)")

# --- build per-cohort tracks: (species_loc, birth_year) → [(sim_year, AGB)] (merge same-age dups) ---
tracks = Dict{Tuple{Int,Int},Vector{Tuple{Float64,Float64}}}()
for (yr, cohs) in snaps
  agg = Dict{Tuple{Int,Int},Float64}()
  for (loc, age, bio) in cohs; k = (loc, Int(round(age))); agg[k] = get(agg, k, 0.0) + bio; end
  for ((loc, ra), bio) in agg; push!(get!(tracks, (loc, yr - ra), Tuple{Float64,Float64}[]), (Float64(yr), bio)); end
end
for v in values(tracks); sort!(v); end

# --- plot ---
allv = Float64[]; for seg in values(tracks); append!(allv, last.(seg)); end; append!(allv, Float64.(oy.agb))
posv = filter(>(0), allv); ymin = max(1.0, minimum(posv) * 0.7); ymax = maximum(allv) * 1.3   # log-axis bounds
fig = MK.Figure(size=(1180, 600))
ax = MK.Axis(fig[1, 1]; xlabel="sim_year  (← spinup back-cast | forward sim →)", ylabel="cohort AGB (g/m², log)",
  yscale=log10, limits=(yr0 - 3, last_meas + 3, ymin, ymax),
  title="Spinup process — plot #$(pid) ($(eco_list[ecop])), SPINUP_MORTALITY_FRACTION=$(SMF): planted at birth years (oldest sim_year $(yr0)) → grown to 0 → +$(last_meas) yr")
MK.vspan!(ax, yr0 - 3, 0; color=(:gray, 0.10))                # shade the spinup (back-cast) region
MK.vlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1.8)
MK.text!(ax, yr0 / 2, ymin * 1.4; text="spinup (back-cast)", color=:gray35, fontsize=11, align=(:center, :bottom))
for ((loc, _by), seg) in tracks
  MK.lines!(ax, first.(seg), last.(seg); color=colsp(loc), linewidth=1.6)
end
present = sort(unique(first.(keys(tracks))) ∪ unique(Int.(oy.eco_species_id)))   # species in legend
for r in eachrow(oy)                                          # observed cohorts as dots at their measured years
  MK.scatter!(ax, [Float64(r.sim_year)], [Float64(r.agb)]; color=colsp(Int(r.eco_species_id)), markersize=11, strokecolor=:black, strokewidth=0.7)
end
MK.Legend(fig[1, 2],
  vcat([MK.LineElement(color=colsp(l), linewidth=3) for l in present],
       [MK.MarkerElement(color=:gray, marker=:circle, strokecolor=:black), MK.LineElement(color=:gray)]),
  vcat([loc2name(l) for l in present], ["observed cohort", "simulated cohort"]); framevisible=true, labelsize=10)
out = "tools/spinup_process.png"
MK.save(out, fig); println("wrote $out")
