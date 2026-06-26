# Sim-vs-obs AGB scatter per (effective ecoregion × species), with a least-squares line + R².
# Reproduces the run's TRAIN/VAL split (same val_frac + split_seed) and emits a scatter for EACH:
#   scatter_sim_obs_train.png / scatter_sim_obs_val.png  (val = held-out generalization).
# Matches sim cohorts to observations by (plot, sim_year, eco_species), sums AGB per species, and
# EXCLUDES the first measurement (sim_year 0 = the init state, which is never predicted).
#   Run:  ./julia_gdal.sh --project=. test/scatter_sim_obs.jl <config.yml>
using Pan
import JLD2, YAML, CairoMakie, Statistics, Dates, DataFrames
const MK = CairoMakie; const P = Pan; const PU = P.PU; const D = P.Data; const DF = DataFrames

cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
outdir = cfg["output_dir"]
P.OVERRIDE_INJECTION[] = Bool(g("override_injection", true))
P.OVERRIDE_INJECTION_SYNC[] = Bool(g("override_injection_sync", true))
P.OVERRIDE_INJECTION_REPLACE[] = Bool(g("override_injection_replace", false))
P.OVERRIDE_INJECTION_DISTURBANCE[] = Symbol(g("override_injection_disturbance", "off"))
D.USE_FIA_CYCLE[] = Bool(g("fia_cycle", false))
P.INIT_PERTURB_FRAC[] = 0.0                       # nominal sim (no perturbation) for the scatter
no_estab = Bool(g("no_establishment", false))
rng = P.RNGType(UInt64(Int(g("seed", 1))))

val_frac = Float64(g("val_frac", 0.0))
split_rng = val_frac > 0 ? P.RNGType(UInt64(Int(g("split_seed", 42)))) : nothing
splots, eco_list, species_list, eco_species_ids, splots_val = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], filter_eco_field=String(g("filter_eco_field", "epa_l4")),
  eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]), output_dir=String(cfg["tablename"]),
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false, val_frac=val_frac, split_rng=split_rng,
  min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
  stratify_eco_mixed=Bool(g("stratify_eco_mixed", false)),
  single_ecoregion=Bool(g("single_ecoregion", false)), stratify_landuse=Bool(g("stratify_landuse", false)),
  filter_extent=(haskey(cfg, "filter_extent") ? String(cfg["filter_extent"]) : nothing),
  filter_ecos=String.(get(cfg, "filter_ecos", String[])),
  filter_plots=NTuple{4,Int}[NTuple{4,Int}(Int.(p)) for p in get(cfg, "filter_plots", [])],
  RNG=rng)
n_species = length(species_list)
bins = Int.(g("bins_idx", [20, 60, 120]))
loss_params = PU.LossParams(age_bins=PU.AgeBins(bins_idx=bins, last_bin_open=true),
  smoothing_weights=P.FloatType[1.0], lambda=P.FloatType(g("loss_lambda", 1.0)))

st = JLD2.load_object(joinpath(outdir, "search_state_latest.jld2"))
best = hasproperty(st, :representative) ? st.representative.x : st.best.x
(best.SPECIES_LIST == species_list && best.ECO_LIST == eco_list) ||
  @warn "params eco/species differ from re-derived data — scatter may be mismatched" params_ecos=best.ECO_LIST data_ecos=eco_list

linfit(x, y) = begin
  mx, my = Statistics.mean(x), Statistics.mean(y); vx = sum((x .- mx) .^ 2)
  b = vx > 0 ? sum((x .- mx) .* (y .- my)) / vx : 0.0; a = my - b * mx
  ss_res = sum((y .- (a .+ b .* x)) .^ 2); ss_tot = sum((y .- my) .^ 2)
  (a, b, ss_tot > 0 ? 1 - ss_res / ss_tot : NaN)
end

function make_scatter(sp, label)
  (isnothing(sp) || DF.nrow(sp) == 0) && (println("($label: no data)"); return)
  max_age = Int(maximum(sp.age_calc))
  spdf = PU.smoothen_ref_years(sp, loss_params, max_age; debug=false)
  spdf_plts = D.make_spdf_dict(spdf, eco_species_ids)
  ssy = D.get_site_sim_years(spdf)
  spc = D.get_spinup_cohorts(sp)
  inj = (no_estab || P.OVERRIDE_INJECTION[]) ? D.get_injection_cohorts(sp; all_cohorts=P.OVERRIDE_INJECTION[]) : nothing
  rs = P.make_sites(sp, eco_species_ids; rng=rng, spinup=false, no_establishment=no_estab)
  idict = isnothing(inj) ? nothing : P._build_injection_dict(inj, rs)
  iyears = isnothing(inj) ? Set{Int}() : Set(Int.(inj.sim_year))
  msy = maximum(maximum.(filter(!isempty, ssy.sim_years)))
  res = P.fit_params(rs, best, msy, n_species, eco_species_ids, spdf_plts, ssy, false, spc, loss_params;
    debug=false, search_tier=3, injection_dict=idict, injection_years=iyears, seeds=[rand(rng, UInt64)])
  cached = res[1][2]
  simdf = DF.DataFrame(plot_id=Int[], sim_year=Int[], esp=Int[], agb=Float64[])
  for (pid, sy, esp, _a, bio) in cached; push!(simdf, (Int(pid), Int(sy), Int(esp), Float64(bio))); end
  sim_agg = DF.combine(DF.groupby(simdf, [:plot_id, :sim_year, :esp]), :agb => sum => :sim_agb)
  obs = DF.combine(DF.groupby(DF.subset(sp, :sim_year => DF.ByRow(>(0))),
      [:plot_id, :sim_year, :eco_id, :eco_species_id]), :agb_sum => sum => :obs_agb)
  DF.rename!(obs, :eco_species_id => :esp)
  paired = DF.innerjoin(obs, sim_agg, on=[:plot_id, :sim_year, :esp])
  # In :exclude_overwrite the disturbed (drop>0) cohorts are OVERWRITTEN with the observed biomass
  # (supplied, not predicted) and excluded from the loss — strip those (plot, year, species) points
  # too, exactly like the first measurement, so the scatter only shows genuine predictions.
  if P.OVERRIDE_INJECTION_DISTURBANCE[] == :exclude_overwrite && !isnothing(inj)
    excl = Set((Int(r.plot_id), Int(r.sim_year), Int(r.eco_species_id)) for r in eachrow(inj) if r.disturbance_drop_pct > 0)
    n0 = DF.nrow(paired)
    paired = DF.filter(row -> !((row.plot_id, row.sim_year, row.esp) in excl), paired)
    println("  $label: stripped $(n0 - DF.nrow(paired)) supplied/overwritten points (exclude_overwrite)")
  end
  # eco_species_id is LOCAL to each eco, so eco must come from the PLOT (each plot → one eco);
  # species is then (eco_id, local esp) → global species_id.
  plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(DF.select(sp, [:plot_id, :eco_id]))))
  especo2sp = Dict((Int(r.eco_id), Int(r.eco_species_id)) => Int(r.species_id)
                   for r in eachrow(unique(DF.select(sp, [:eco_id, :eco_species_id, :species_id]))))
  paired.eco = [plot2eco[p] for p in paired.plot_id]
  paired.sp = [especo2sp[(e, esp)] for (e, esp) in zip(paired.eco, paired.esp)]
  # 3 fit modes × per-ecoregion figure: linear (OLS, SE-of-line band), log (log-log OLS, slope=exponent),
  # weighted (WLS w=1/AGB, variance∝AGB → band = σ·√AGB fanning out with AGB).
  for mode in (:linear, :log, :weighted, :weightedlog)
    for e in sort(unique(paired.eco))
      pe = DF.subset(paired, :eco => DF.ByRow(==(e)))
      econame = replace(split(eco_list[e], "=")[end], r"[^A-Za-z0-9]" => "_")
      r2all = overall_r2(Float64.(pe.obs_agb), Float64.(pe.sim_agb), mode)
      sps = [s for s in sort(unique(pe.sp)) if DF.nrow(DF.subset(pe, :sp => DF.ByRow(==(s)))) >= 3]
      isempty(sps) && continue
      println("$label/$econame [$mode]: $(DF.nrow(pe)) pts / $(length(unique(pe.plot_id))) plots / $(length(sps)) species  overall R²=$(round(r2all,digits=3))")
      ncol = min(4, max(1, length(sps))); nr = cld(length(sps), ncol)
      fig = MK.Figure(size=(330 * ncol, 300 * nr + 30))
      MK.Label(fig[0, 1:ncol], "Sim vs obs AGB [$mode fit] — $(label) · $(eco_list[e]) — overall R²=$(round(r2all,digits=3))"; fontsize=14, font=:bold)
      for (i, s) in enumerate(sps)
        d = DF.subset(pe, :sp => DF.ByRow(==(s)))
        r, c = fldmod1(i, ncol)
        draw_panel!(fig[r, c], Float64.(d.obs_agb), Float64.(d.sim_agb), species_list[s], mode)
      end
      out = joinpath(outdir, "scatter_sim_obs_$(label)_$(econame)_$(mode).png"); MK.save(out, fig); println("  wrote $out")
    end
  end
end

# overall R² in the mode's own space
function overall_r2(x, y, mode)
  if mode == :log
    return linfit(log10.(x .+ 1), log10.(y .+ 1))[3]
  elseif mode == :weightedlog
    n = length(x); lx = log10.(x .+ 1); ly = log10.(y .+ 1); wn = (x .+ 1) ./ Statistics.mean(x .+ 1)
    xw = sum(wn .* lx) / n; yw = sum(wn .* ly) / n; Sxxw = sum(wn .* (lx .- xw) .^ 2)
    b = Sxxw > 0 ? sum(wn .* (lx .- xw) .* (ly .- yw)) / Sxxw : 0.0
    return 1 - sum(wn .* (ly .- (yw .+ b .* (lx .- xw))) .^ 2) / max(sum(wn .* (ly .- yw) .^ 2), eps())
  elseif mode == :weighted
    w = 1.0 ./ max.(x, 1.0); sw = sum(w); xw = sum(w .* x) / sw; yw = sum(w .* y) / sw
    sxx = sum(w .* (x .- xw) .^ 2); b = sxx > 0 ? sum(w .* (x .- xw) .* (y .- yw)) / sxx : 0.0
    return 1 - sum(w .* (y .- (yw .+ b .* (x .- xw))) .^ 2) / max(sum(w .* (y .- yw) .^ 2), eps())
  end
  return linfit(x, y)[3]
end

function draw_panel!(cell, x, y, title, mode)
  n = length(x)
  if mode == :log
    ε = 1.0; lx = log10.(x .+ ε); ly = log10.(y .+ ε); a, b, r2 = linfit(lx, ly)
    xb = Statistics.mean(lx); Sxx = sum((lx .- xb) .^ 2)
    σr = n > 2 ? sqrt(sum((ly .- (a .+ b .* lx)) .^ 2) / (n - 2)) : 0.0
    ax = MK.Axis(cell; xlabel="observed AGB", ylabel="simulated AGB", xscale=log10, yscale=log10, title="$title  (n=$n)")
    lxs = range(minimum(lx), maximum(lx); length=60); ylf = a .+ b .* lxs
    seb = Sxx > 0 ? σr .* sqrt.(1 / n .+ (lxs .- xb) .^ 2 ./ Sxx) : zeros(length(lxs))
    xp = 10 .^ lxs
    lo = max(1.0, minimum(vcat(x, y) .+ ε)); hi = maximum(vcat(x, y) .+ ε)
    MK.lines!(ax, [lo, hi], [lo, hi]; color=:gray70, linestyle=:dash)               # y = x
    MK.band!(ax, xp, 10 .^ (ylf .- seb), 10 .^ (ylf .+ seb); color=(:firebrick, 0.18))
    MK.lines!(ax, xp, 10 .^ ylf; color=:firebrick, linewidth=2)
    MK.scatter!(ax, x .+ ε, y .+ ε; color=:steelblue, markersize=7, strokewidth=0.3, strokecolor=:black)
    MK.text!(ax, 0.04, 0.96; text="R²=$(round(r2,digits=3))\nexp=$(round(b,digits=2))", space=:relative, align=(:left, :top), fontsize=9)
  elseif mode == :weightedlog
    ε = 1.0; lx = log10.(x .+ ε); ly = log10.(y .+ ε); wn = (x .+ ε) ./ Statistics.mean(x .+ ε)  # weight by AGB
    xw = sum(wn .* lx) / n; yw = sum(wn .* ly) / n; Sxxw = sum(wn .* (lx .- xw) .^ 2)
    b = Sxxw > 0 ? sum(wn .* (lx .- xw) .* (ly .- yw)) / Sxxw : 0.0; a = yw - b * xw
    resid = ly .- (a .+ b .* lx); r2 = 1 - sum(wn .* resid .^ 2) / max(sum(wn .* (ly .- yw) .^ 2), eps())
    σr = n > 2 ? sqrt(sum(wn .* resid .^ 2) / (n - 2)) : 0.0
    ax = MK.Axis(cell; xlabel="observed AGB", ylabel="simulated AGB", xscale=log10, yscale=log10, title="$title  (n=$n)")
    lxs = range(minimum(lx), maximum(lx); length=60); ylf = a .+ b .* lxs
    seb = Sxxw > 0 ? σr .* sqrt.(1 / n .+ (lxs .- xw) .^ 2 ./ Sxxw) : zeros(length(lxs))
    xp = 10 .^ lxs; lo = max(1.0, minimum(vcat(x, y) .+ ε)); hi = maximum(vcat(x, y) .+ ε)
    MK.lines!(ax, [lo, hi], [lo, hi]; color=:gray70, linestyle=:dash)
    MK.band!(ax, xp, 10 .^ (ylf .- seb), 10 .^ (ylf .+ seb); color=(:firebrick, 0.18))
    MK.lines!(ax, xp, 10 .^ ylf; color=:firebrick, linewidth=2)
    MK.scatter!(ax, x .+ ε, y .+ ε; color=:steelblue, markersize=7, strokewidth=0.3, strokecolor=:black)
    MK.text!(ax, 0.04, 0.96; text="R²=$(round(r2,digits=3))\nexp=$(round(b,digits=2))", space=:relative, align=(:left, :top), fontsize=9)
  elseif mode == :weighted
    w = 1.0 ./ max.(x, 1.0); sw = sum(w); xw = sum(w .* x) / sw; yw = sum(w .* y) / sw
    sxx = sum(w .* (x .- xw) .^ 2); b = sxx > 0 ? sum(w .* (x .- xw) .* (y .- yw)) / sxx : 0.0; a = yw - b * xw
    resid = y .- (a .+ b .* x); r2 = 1 - sum(w .* resid .^ 2) / max(sum(w .* (y .- yw) .^ 2), eps())
    σ2 = n > 2 ? sum(resid .^ 2 ./ max.(x, 1.0)) / (n - 2) : 0.0     # residual variance per unit AGB
    ax = MK.Axis(cell; xlabel="observed AGB", ylabel="simulated AGB", title="$title  (n=$n)")
    m = max(maximum(x), maximum(y), 1.0) * 1.05
    xs = range(minimum(x), maximum(x); length=60); yl = a .+ b .* xs; band = sqrt(σ2) .* sqrt.(xs)  # σ·√AGB
    MK.lines!(ax, [0, m], [0, m]; color=:gray70, linestyle=:dash)
    MK.band!(ax, xs, max.(yl .- band, 0.0), yl .+ band; color=(:firebrick, 0.18))
    MK.lines!(ax, xs, yl; color=:firebrick, linewidth=2)
    MK.scatter!(ax, x, y; color=:steelblue, markersize=7, strokewidth=0.3, strokecolor=:black)
    MK.text!(ax, 0.04, 0.96; text="R²=$(round(r2,digits=3))\nslope=$(round(b,digits=2))", space=:relative, align=(:left, :top), fontsize=9)
    MK.xlims!(ax, 0, m); MK.ylims!(ax, 0, m)
  else  # linear
    a, b, r2 = linfit(x, y)
    ax = MK.Axis(cell; xlabel="observed AGB", ylabel="simulated AGB", title="$title  (n=$n)")
    m = max(maximum(x), maximum(y), 1.0) * 1.05; xb = Statistics.mean(x); Sxx = sum((x .- xb) .^ 2)
    σr = n > 2 ? sqrt(sum((y .- (a .+ b .* x)) .^ 2) / (n - 2)) : 0.0
    xs = range(minimum(x), maximum(x); length=60); yl = a .+ b .* xs
    seb = Sxx > 0 ? σr .* sqrt.(1 / n .+ (xs .- xb) .^ 2 ./ Sxx) : zeros(length(xs))
    MK.lines!(ax, [0, m], [0, m]; color=:gray70, linestyle=:dash)
    MK.band!(ax, xs, max.(yl .- seb, 0.0), yl .+ seb; color=(:firebrick, 0.18))
    MK.lines!(ax, xs, yl; color=:firebrick, linewidth=2)
    MK.scatter!(ax, x, y; color=:steelblue, markersize=7, strokewidth=0.3, strokecolor=:black)
    MK.text!(ax, 0.04, 0.96; text="R²=$(round(r2,digits=3))\nslope=$(round(b,digits=2))", space=:relative, align=(:left, :top), fontsize=9)
    MK.xlims!(ax, 0, m); MK.ylims!(ax, 0, m)
  end
end

make_scatter(splots, "train")
make_scatter(splots_val, "val")
