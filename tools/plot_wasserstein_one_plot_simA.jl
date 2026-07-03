# Wasserstein illustration on ONE real plot, the Sim-A way. Sim A seeds the simulation from the plot's
# OBSERVED cohorts at its FIRST measured year (so sim ≡ obs there, W1 = 0), runs forward, and at a LATER
# measured year compares the simulated vs observed AGB-weighted age-CDF AT THE SAME (aligned) year. Dots =
# cohorts, dashed step lines = cumulative AGB fraction (cumsum/CDF), shaded area between = W1. Uses the real
# fit_params (tier-3) Sim-A pipeline + the single-cov best params, so the right-panel sim is genuine drift.
#   Run: [PAN_PARAMS=params.jld2] ./julia_gdal.sh --project=. tools/plot_wasserstein_one_plot_simA.jl <config.yml>
using Pan
import JLD2, YAML, CairoMakie, Statistics, DataFrames
const MK = CairoMakie; const P = Pan; const PU = P.PU; const D = P.Data; const DF = DataFrames

cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
P.OVERRIDE_INJECTION[] = Bool(g("override_injection", true))
P.OVERRIDE_INJECTION_SYNC[] = Bool(g("override_injection_sync", true))
P.OVERRIDE_INJECTION_REPLACE[] = Bool(g("override_injection_replace", false))
P.OVERRIDE_INJECTION_DISTURBANCE[] = Symbol(g("override_injection_disturbance", "off"))
D.USE_FIA_CYCLE[] = Bool(g("fia_cycle", false))
P.INIT_PERTURB_FRAC[] = 0.0
no_estab = Bool(g("no_establishment", false))
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
n_species = length(species_list)
bins = Int.(g("bins_idx", [20, 60, 120]))
loss_params = PU.LossParams(age_bins=PU.AgeBins(bins_idx=bins, last_bin_open=true), smoothing_weights=P.FloatType[1.0], lambda=P.FloatType(g("loss_lambda", 1.0)))
best = if haskey(ENV, "PAN_PARAMS")
  JLD2.load_object(ENV["PAN_PARAMS"])
else
  st = JLD2.load_object(joinpath(cfg["output_dir"], "search_state_latest.jld2"))
  hasproperty(st, :representative) ? st.representative.x : st.best.x
end

# --- run the real Sim-A pipeline (tier 3) and capture per-cohort (plot, year, age, AGB) ---
sp = splots
max_age = Int(maximum(sp.age_calc)); spdf = PU.smoothen_ref_years(sp, loss_params, max_age; debug=false)
spdf_plts = D.make_spdf_dict(spdf, eco_species_ids); ssy = D.get_site_sim_years(spdf); spc = D.get_spinup_cohorts(sp)
inj = (no_estab || P.OVERRIDE_INJECTION[]) ? D.get_injection_cohorts(sp; all_cohorts=P.OVERRIDE_INJECTION[]) : nothing
rs = P.make_sites(sp, eco_species_ids; rng=rng, spinup=false, no_establishment=no_estab)
idict = isnothing(inj) ? nothing : P._build_injection_dict(inj, rs); iyears = isnothing(inj) ? Set{Int}() : Set(Int.(inj.sim_year))
msy = maximum(maximum.(filter(!isempty, ssy.sim_years)))
res = P.fit_params(rs, best, msy, n_species, eco_species_ids, spdf_plts, ssy, false, spc, loss_params;
  debug=false, search_tier=3, injection_dict=idict, injection_years=iyears, seeds=[rand(rng, UInt64)])
cached = res[1][2]

# sim cohorts: AGB per (plot, year, age), summed over species
simdf = DF.DataFrame(plot_id=Int[], yr=Int[], age=Float64[], agb=Float64[])
for (pid, sy, _esp, a, bio) in cached
  push!(simdf, (Int(pid), Int(sy), Float64(a), Float64(bio)))
end
sim_age = DF.combine(DF.groupby(simdf, [:plot_id, :yr, :age]), :agb => sum => :agb)
# obs cohorts: AGB per (plot, year, age) — includes the seed year (sim_year == 0)
obs_age = DF.combine(DF.groupby(sp, [:plot_id, :sim_year, :age_calc]), :agb_sum => sum => :agb)
DF.rename!(obs_age, :sim_year => :yr, :age_calc => :age)
# plot → eco name
plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(DF.select(sp, [:plot_id, :eco_id]))))

# --- AGB-weighted step-CDF helpers + W1 ---
GRID = collect(0.0:0.5:200.0); DG = GRID[2] - GRID[1]
stepF(xa, xc, a) = (i = searchsortedlast(xa, a); i == 0 ? 0.0 : xc[i])
function cdf_of(df)                       # df rows: age, agb (one plot-year) → (ages, cumAGBfrac)
  d = DF.sort(df, :age); w = Float64.(d.agb); s = sum(w); s <= 0 && return (Float64[], Float64[])
  (Float64.(d.age), cumsum(w ./ s))
end
function w1_between(ages_o, cum_o, ages_s, cum_s)
  Fo = [stepF(ages_o, cum_o, a) for a in GRID]; Fs = [stepF(ages_s, cum_s, a) for a in GRID]
  (sum(abs.(Fo .- Fs)) * DG, Fo, Fs)
end

# --- pick ONE plot: has a seed year (0) + a later obs year with a matching sim year; many obs cohorts; W1 illustrative ---
cands = NamedTuple[]
for pid in sort(unique(obs_age.plot_id))
  oy = DF.subset(obs_age, :plot_id => DF.ByRow(==(pid)))
  years = sort(unique(oy.yr))
  (length(years) < 2 || years[1] != 0) && continue          # need a seed (yr 0) + at least one later obs
  y1 = maximum(years)                                        # latest measured year (most drift)
  so = DF.subset(sim_age, :plot_id => DF.ByRow(==(pid)), :yr => DF.ByRow(==(y1)))
  DF.nrow(so) == 0 && continue                               # need a simulated set at the aligned year
  o1 = DF.subset(oy, :yr => DF.ByRow(==(y1)))
  ages_o, cum_o = cdf_of(o1); ages_s, cum_s = cdf_of(so)
  (isempty(ages_o) || isempty(ages_s)) && continue
  w1, = w1_between(ages_o, cum_o, ages_s, cum_s)
  push!(cands, (pid=pid, y1=y1, nobs=DF.nrow(o1), nsim=DF.nrow(so), w1=w1))
end
# prefer plenty of obs cohorts and an illustrative (not tiny, not extreme) divergence
sort!(cands, by=c -> (-(min(c.nobs, 25)) , abs(c.w1 - 15.0)))
@assert !isempty(cands) "no plot with a seed year + later aligned obs/sim found"
println("top candidate plots (pid, y1, nobs, nsim, W1):")
for c in cands[1:min(8, end)]; println("  $(c.pid)  y1=$(c.y1)  nobs=$(c.nobs)  nsim=$(c.nsim)  W1=$(round(c.w1;digits=1))"); end
sel = cands[1]; pid = sel.pid; y1 = sel.y1
econame = eco_list[plot2eco[pid]]

# --- build the two panels: seed year y0=0 (sim ≡ obs, W1=0) and aligned year y1 (real drift) ---
oy = DF.subset(obs_age, :plot_id => DF.ByRow(==(pid)))
o0 = DF.subset(oy, :yr => DF.ByRow(==(0))); o1 = DF.subset(oy, :yr => DF.ByRow(==(y1)))
s1 = DF.subset(sim_age, :plot_id => DF.ByRow(==(pid)), :yr => DF.ByRow(==(y1)))
ages_o0, cum_o0 = cdf_of(o0)                       # seed: sim is identical to obs (Sim A injection)
ages_o1, cum_o1 = cdf_of(o1); ages_s1, cum_s1 = cdf_of(s1)
xmax = max(maximum(ages_o0), maximum(ages_o1), maximum(ages_s1)) + 8

panels = [(ages_o0, cum_o0, ages_o0, cum_o0, "y0 = first measured year — Sim A seed (sim ≡ obs)"),
          (ages_o1, cum_o1, ages_s1, cum_s1, "y1 = +$(y1) yr — sim run forward vs obs (aligned year)")]
fig = MK.Figure(size=(1200, 510))
MK.Label(fig[0, 1:2], "Wasserstein-1 on ONE plot, Sim-A across years — sim seeded from obs at y0 (W1=0), drifts by y1 (plot #$(pid), $(econame))"; fontsize=13, font=:bold)
for (i, (ao, co, as, cs, lbl)) in enumerate(panels)
  Fo = [stepF(ao, co, a) for a in GRID]; Fs = [stepF(as, cs, a) for a in GRID]
  w1 = sum(abs.(Fo .- Fs)) * DG
  ax = MK.Axis(fig[1, i]; title="$(lbl):  W1 = $(round(w1; digits=1)) yr", xlabel="cohort age (yr)",
    ylabel="cumulative AGB fraction (CDF)", limits=(0, xmax, -0.03, 1.05))
  MK.band!(ax, GRID, min.(Fo, Fs), max.(Fo, Fs); color=(:darkorange, 0.30))
  MK.stairs!(ax, [0.0; ao; xmax], [0.0; co; 1.0]; color=:navy, linestyle=:dash, linewidth=2.0, step=:post)
  MK.scatter!(ax, ao, co; color=:navy, markersize=11)
  MK.stairs!(ax, [0.0; as; xmax], [0.0; cs; 1.0]; color=:firebrick, linestyle=:dash, linewidth=2.0, step=:post)
  MK.scatter!(ax, as, cs; color=:firebrick, markersize=11, marker=:diamond)
  MK.Legend(fig[2, i], [MK.MarkerElement(color=:navy, marker=:circle), MK.MarkerElement(color=:firebrick, marker=:diamond), MK.PolyElement(color=(:darkorange, 0.45))],
    ["observed cohorts (+CDF)", "simulated cohorts (+CDF)", "W1 (area between CDFs)"]; orientation=:horizontal, framevisible=false)
end
out = "tools/wasserstein_one_plot_simA.png"
MK.save(out, fig); println("wrote $out  (plot #$pid, $econame, y1=+$(y1)yr, W1=$(round(sel.w1;digits=1)))")
