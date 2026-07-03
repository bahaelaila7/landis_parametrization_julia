# Sim B cohort illustration (NO Wasserstein), the counterpart to the Sim-A Wasserstein figure. Sim B does
# NOT inject the observed cohorts — it SPINS UP a stand from bare ground (make_sites(spinup=true) →
# spinup_cohorts! with free establishment), then runs forward. So instead of an age-CDF + W1 area we just
# show the raw cohorts as (age, AGB) dots: observed (navy) vs simulated (red). Two panels for ONE plot:
#   left  = year 0, right after spinup — the model-assembled stand vs the observed starting cohorts (they
#           need NOT match: Sim B regenerates rather than seeds, unlike Sim A's W1=0 aligned start);
#   right = a later measured year (aligned) — Sim-B cohorts vs observed cohorts at that same year.
#   Run: [PAN_PLOT=409] [PAN_PARAMS=params.jld2] ./julia_gdal.sh --project=. tools/plot_simB_cohorts_spinup.jl <config.yml>
using Pan
import JLD2, YAML, CairoMakie, DataFrames
const MK = CairoMakie; const P = Pan; const D = P.Data; const BSP = P.BiomassSuccessionPlugin; const DF = DataFrames

cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
D.USE_FIA_CYCLE[] = Bool(g("fia_cycle", false))
P.INIT_PERTURB_FRAC[] = 0.0
rng = P.RNGType(UInt64(Int(g("seed", 1))))
val_frac = Float64(g("val_frac", 0.0)); split_rng = val_frac > 0 ? P.RNGType(UInt64(Int(g("split_seed", 42)))) : nothing
# IDENTICAL data setup to tools/plot_wasserstein_one_plot_simA.jl so plot_id (#409) means the same plot
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
# train-only (matches the Sim-A figure's plot_id numbering; the spinup path asserts mapcode==site index,
# which requires the contiguous in-order plot_id of a single prepared set — vcat'ing val breaks it)
sp = splots

# observed cohorts (age, AGB) per (plot, year), summed over species
obs_age = DF.combine(DF.groupby(sp, [:plot_id, :sim_year, :age_calc]), :agb_sum => sum => :agb)
DF.rename!(obs_age, :sim_year => :yr, :age_calc => :age)
plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(DF.select(sp, [:plot_id, :eco_id]))))

# pick the plot: PAN_PLOT override, else the one with the most observed cohorts at its latest measured year
function pick_plot()
  haskey(ENV, "PAN_PLOT") && return parse(Int, ENV["PAN_PLOT"])
  best_pid = 0; best_n = -1
  for pid in unique(obs_age.plot_id)
    oy = DF.subset(obs_age, :plot_id => DF.ByRow(==(pid))); years = unique(oy.yr)
    (length(years) < 2 || minimum(years) != 0) && continue
    n = DF.nrow(DF.subset(oy, :yr => DF.ByRow(==(maximum(years)))))
    n > best_n && (best_n = n; best_pid = pid)
  end
  best_pid
end
pid = pick_plot()
oy = DF.subset(obs_age, :plot_id => DF.ByRow(==(pid)))
@assert DF.nrow(oy) > 0 "plot $pid not in data"
y1 = maximum(oy.yr)
econame = eco_list[plot2eco[pid]]
println("plot #$pid ($econame): observed years = $(sort(unique(oy.yr))), comparing at y1=+$(y1)yr")

# --- Sim B: spin up from bare ground, then step to y1. Sites are independent (local competition only), so
#     we run the FULL set and read plot #pid's site — a single-plot soa hits a mapcode-indexing bug. ---
soa = P.make_sites(sp, eco_species_ids; rng=P.RNGType(1), spinup=true, no_establishment=false)
spinup_all = D.get_spinup_cohorts(sp)
eco_params = BSP.generate_eco_params(best)
soa = BSP.spinup_cohorts!(soa, spinup_all, eco_params)          # regenerate stand → year-0 state
si = findfirst(i -> Int(P.getsite(soa, i).mapcode) == pid, 1:soa.n); @assert si !== nothing
read_cohorts(soa, si) = (s = P.getsite(soa, si); [(Float64(s.c_age[j]), Float64(s.c_bio[j])) for j in 1:Int(s.live)])

simB0 = read_cohorts(soa, si)                                   # cohorts right after spinup (year 0)
for y in 1:y1                                                   # step the spun-up stand forward to the aligned year
  P.PanCore.process_plugin!(soa, BSP.BiomassSuccession, y; ctx=(eco_params=eco_params,))
end
simB1 = read_cohorts(soa, si)

obs0 = [(Float64(r.age), Float64(r.agb)) for r in eachrow(DF.subset(oy, :yr => DF.ByRow(==(0))))]
obs1 = [(Float64(r.age), Float64(r.agb)) for r in eachrow(DF.subset(oy, :yr => DF.ByRow(==(y1))))]
allpts = vcat(obs0, obs1, simB0, simB1)
xmax = maximum(first.(allpts)) + 6; ymax = maximum(last.(allpts)) * 1.08
println("cohort counts — obs0=$(length(obs0)) simB0=$(length(simB0)) | obs1=$(length(obs1)) simB1=$(length(simB1))")

# --- plot: raw cohorts as (age, AGB) dots + stems; navy obs / red sim. NO CDF, NO Wasserstein band ---
panels = [(obs0, simB0, "year 0 — after spinup: model-assembled stand vs observed start"),
          (obs1, simB1, "+$(y1) yr (aligned year) — Sim-B cohorts vs observed")]
fig = MK.Figure(size=(1200, 520))
MK.Label(fig[0, 1:2], "Sim B (no Wasserstein): ONE plot, stand SPUN UP from bare ground (free establishment) — cohorts (age, AGB), obs vs sim  (plot #$(pid), $(econame))"; fontsize=13, font=:bold)
stem!(ax, pts, col; mk) = begin
  for (a, b) in pts; MK.lines!(ax, [a, a], [0.0, b]; color=(col, 0.35), linewidth=1.3); end
  MK.scatter!(ax, first.(pts), last.(pts); color=col, markersize=11, marker=mk)
end
for (i, (obs, simb, lbl)) in enumerate(panels)
  ax = MK.Axis(fig[1, i]; title="$(lbl)\nΣAGB obs=$(round(Int,sum(last.(obs)))) · sim=$(round(Int,sum(last.(simb)))) g/m²",
    xlabel="cohort age (yr)", ylabel="cohort AGB (g/m²)", limits=(0, xmax, -0.02 * ymax, ymax))
  stem!(ax, obs, :navy; mk=:circle)
  stem!(ax, simb, :firebrick; mk=:diamond)
  MK.Legend(fig[2, i], [MK.MarkerElement(color=:navy, marker=:circle), MK.MarkerElement(color=:firebrick, marker=:diamond)],
    ["observed cohorts", "Sim-B cohorts (spun up)"]; orientation=:horizontal, framevisible=false)
end
out = "tools/simB_cohorts_spinup.png"
MK.save(out, fig); println("wrote $out")
