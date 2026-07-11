# TIERED variant of scatter_sim_obs.jl (original left untouched). Panels are grouped by per-species site-tier
# instead of per-ecoregion: TIERED species (in param_split_species) get one panel per site-cell group
# (e.g. PITA A/B/C/D, PIEL AB/C/D); POOLED species get one panel each with their cells combined. One figure
# per (split × mode): scatter_sim_obs_train_<mode>.png / _val_<mode>.png. Tiered species first, pooled after.
#   Run:  ./julia_gdal.sh --project=. test/scatter_sim_obs_tiered.jl <config.yml>
using Pan
import JLD2, YAML, CairoMakie, Statistics, Dates, DataFrames, CSV
const MK = CairoMakie; const P = Pan; const PU = P.PU; const D = P.Data; const DF = DataFrames

cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
base_outdir = get(ENV, "PAN_OUT", "") != "" ? joinpath(ENV["PAN_OUT"], basename(String(cfg["output_dir"]))) : String(cfg["output_dir"])  # outputs under $PAN_OUT (scratch) if set
# PAN_PARAMS = a specific params .jld2 to plot (else the run's representative); PAN_OUTSUB = output subfolder.
outdir = haskey(ENV, "PAN_OUTSUB") ? (let d = joinpath(base_outdir, ENV["PAN_OUTSUB"]); mkpath(d); d end) : base_outdir
TAG = haskey(ENV, "PAN_OUTSUB") ? replace(basename(ENV["PAN_OUTSUB"]), "candidate_" => "cand ") * " — " : ""   # candidate id in titles
P.OVERRIDE_INJECTION[] = Bool(g("override_injection", true))
P.OVERRIDE_INJECTION_SYNC[] = Bool(g("override_injection_sync", true))
P.OVERRIDE_INJECTION_REPLACE[] = Bool(g("override_injection_replace", false))
P.OVERRIDE_INJECTION_DISTURBANCE[] = Symbol(g("override_injection_disturbance", "off"))
D.USE_FIA_CYCLE[] = Bool(g("fia_cycle", false))
P.INIT_PERTURB_FRAC[] = 0.0                       # nominal sim (no perturbation) for the scatter
no_estab = Bool(g("no_establishment", false))
rng = P.RNGType(UInt64(Int(g("seed", 1))))

val_frac = Float64(g("val_frac", 0.0))
test_frac = Float64(g("test_frac", 0.0))               # 3-way split: reproduce the run's held-out TEST set too
n_folds = Int(g("n_folds", 1)); fold_index = parse(Int, get(ENV, "PAN_FOLD", string(g("fold_index", 1))))  # PAN_FOLD picks the CV fold
split_rng = (val_frac > 0 || n_folds > 1 || test_frac > 0) ? P.RNGType(UInt64(Int(g("split_seed", 42)))) : nothing
excl_plots = NTuple{4,Int}[]                       # honor the run's outlier exclusions so the fold split matches
let epc = g("exclude_plots_csv", nothing)
  if !(epc === nothing || epc == "null")
    for ln in Iterators.drop(eachline(String(epc)), 1); isempty(strip(ln)) && continue; v = parse.(Int, split(ln, ",")); push!(excl_plots, (v[1], v[2], v[3], v[4])); end
  end
end
splots, eco_list, species_list, eco_species_ids, splots_val, splots_test = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], filter_eco_field=String(g("filter_eco_field", "epa_l4")),
  eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]), output_dir=String(cfg["tablename"]),
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false, val_frac=val_frac, split_rng=split_rng, test_frac=test_frac,
  n_folds=n_folds, fold_index=fold_index, exclude_plots=excl_plots,
  min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
  stratify_eco_mixed=Bool(g("stratify_eco_mixed", false)),
  single_ecoregion=Bool(g("single_ecoregion", false)), stratify_landuse=Bool(g("stratify_landuse", false)),
  site_class_strata=Bool(g("site_class_strata", false)), siteclass_hi_max=Int(g("siteclass_hi_max", 4)),
  siteclass_scheme=String(g("siteclass_scheme", "2way")),
  shade_tier_csv=(haskey(cfg, "shade_tier_csv") ? String(cfg["shade_tier_csv"]) : nothing),
  filter_extent=(haskey(cfg, "filter_extent") ? String(cfg["filter_extent"]) : nothing),
  filter_ecos=String.(get(cfg, "filter_ecos", String[])),
  filter_plots=NTuple{4,Int}[NTuple{4,Int}(Int.(p)) for p in get(cfg, "filter_plots", [])],
  RNG=rng)
n_species = length(species_list)
# ── panel grouping: tiered species (in a param split set) get ONE panel per site-cell group (e.g. PITA A/B/C/D,
# PIEL AB/C/D); pooled species get ONE panel with all their cells combined. Mirrors param_split_species + param_tier_merge.
_tiered_set = Set{String}()
for (_, syms) in get(cfg, "param_split_species", Dict()), s in syms; push!(_tiered_set, uppercase(strip(String(s)))); end
_merge_map = Dict{String,Dict{String,String}}()   # species → (cell → group label)
for (_, spmap) in get(cfg, "param_tier_merge", Dict()), (sp, cm) in spmap
  d = get!(_merge_map, uppercase(strip(String(sp))), Dict{String,String}())
  for (c, gp) in cm; d[String(c)] = String(gp); end
end
_eco_cell(e) = (p = split(String(eco_list[e]), "|lu="); length(p) == 2 ? String(p[2]) : "")   # eco → site-cell
_is_tiered(s) = uppercase(species_list[s]) in _tiered_set
_panel_grp(s, e) = _is_tiered(s) ? get(get(_merge_map, uppercase(species_list[s]), Dict{String,String}()), _eco_cell(e), _eco_cell(e)) : "pooled"
# species abundance = distinct plots over all splits → rank panels by it
_allsp = DF.DataFrame(plot_id=Int[], species_id=Int[])
for s in (splots, splots_val, splots_test); isnothing(s) && continue; append!(_allsp, DF.DataFrame(plot_id=Int.(s.plot_id), species_id=Int.(s.species_id))); end
spabund = Dict{Int,Int}(); for gdf in DF.groupby(unique(_allsp), :species_id); spabund[Int(gdf.species_id[1])] = DF.nrow(gdf); end
bins = Int.(g("bins_idx", [20, 60, 120]))
loss_params = PU.LossParams(age_bins=PU.AgeBins(bins_idx=bins, last_bin_open=true),
  smoothing_weights=P.FloatType[1.0], lambda=P.FloatType(g("loss_lambda", 1.0)))

best = if haskey(ENV, "PAN_PARAMS")
  JLD2.load_object(ENV["PAN_PARAMS"])
else
  st = JLD2.load_object(joinpath(base_outdir, "search_state_latest.jld2"))
  hasproperty(st, :representative) ? st.representative.x : st.best.x
end
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
    debug=false, search_tier=3, injection_dict=idict, injection_years=iyears, seeds=[rand(rng, UInt64)], cache_preinject=true)  # pre-inject = pure model state (REQUIRED for the cohort match below)
  cached = res[1][2]
  # ONLY genuinely-simulated biomass ends up in the scatter. Match the pre-inject model cohorts to the
  # OBSERVED injection set by (plot, year, species, AGE) — exactly the sync survivor rule. This drops
  # (a) INJECTED cohorts (observed-only, never grown by the model) and (b) SYNC-REMOVED cohorts (sim-only,
  # absent from the data), then drops disturbance-OVERWRITTEN cohorts (drop>0: biomass supplied, not predicted).
  isnothing(inj) && error("$label: scatter needs injection cohorts (override_injection) to separate simulated vs injected biomass")
  simc = DF.DataFrame(plot_id=Int[], sim_year=Int[], esp=Int[], age=Int[], sim_agb=Float64[])
  for (pid, sy, esp, age, bio) in cached; push!(simc, (Int(pid), Int(sy), Int(esp), Int(age), Float64(bio))); end
  simc = DF.combine(DF.groupby(DF.subset(simc, :sim_year => DF.ByRow(>(0))), [:plot_id, :sim_year, :esp, :age]), :sim_agb => sum => :sim_agb)
  injc = DF.DataFrame(plot_id=Int.(inj.plot_id), sim_year=Int.(inj.sim_year), esp=Int.(inj.eco_species_id),
                      age=Int.(inj.age_calc), obs_agb=Float64.(inj.agb_sum), drop=Float64.(inj.disturbance_drop_pct))
  injc = DF.combine(DF.groupby(DF.subset(injc, :sim_year => DF.ByRow(>(0))), [:plot_id, :sim_year, :esp, :age]),
                    :obs_agb => sum => :obs_agb, :drop => maximum => :drop)
  matched = DF.innerjoin(simc, injc, on=[:plot_id, :sim_year, :esp, :age])   # sync survivors: simulated ∩ observed, same (species,age)
  n_sim = DF.nrow(simc)
  P.OVERRIDE_INJECTION_DISTURBANCE[] == :exclude_overwrite && (matched = DF.filter(r -> r.drop <= 0.0, matched))
  paired = DF.combine(DF.groupby(matched, [:plot_id, :sim_year, :esp]), :sim_agb => sum => :sim_agb, :obs_agb => sum => :obs_agb)
  println("  $label: $(DF.nrow(paired)) (plot,yr,sp) points from $(DF.nrow(matched))/$(n_sim) simulated cohorts matched to observed (dropped sync-removed + injected + disturbance-overwritten)")
  # eco_species_id is LOCAL to each eco, so eco must come from the PLOT (each plot → one eco);
  # species is then (eco_id, local esp) → global species_id.
  plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(DF.select(sp, [:plot_id, :eco_id]))))
  especo2sp = Dict((Int(r.eco_id), Int(r.eco_species_id)) => Int(r.species_id)
                   for r in eachrow(unique(DF.select(sp, [:eco_id, :eco_species_id, :species_id]))))
  paired.eco = [plot2eco[p] for p in paired.plot_id]
  paired.sp = [especo2sp[(e, esp)] for (e, esp) in zip(paired.eco, paired.esp)]
  # CACHE the raw matched sim-obs pairs (per plot × species) so aggregated scatters can be re-plotted w/o re-sim
  CSV.write(joinpath(outdir, "sim_obs_pairs_$(label).csv"),
    DF.DataFrame(plot_id=paired.plot_id, sim_year=paired.sim_year, stratum=[eco_list[e] for e in paired.eco],
                 species=[species_list[s] for s in paired.sp], obs_agb=Float64.(paired.obs_agb), sim_agb=Float64.(paired.sim_agb)))
  # ONE figure per (split × mode): panels = per-site-cell for TIERED species (PITA A/B/C/D, PIEL AB/C/D) +
  # ONE pooled panel per pooled species (its cells combined). Tiered species first, pooled after — no per-eco split.
  fitrows = DF.DataFrame(split=String[], stratum=String[], species=String[], n=Int[], slope=Float64[], R2=Float64[])
  _modes = haskey(ENV, "PAN_SCATTER_MODES") ? Tuple(Symbol.(strip.(split(ENV["PAN_SCATTER_MODES"], ",")))) : (:linear, :log, :weighted, :weightedlog)
  paired.pgrp = [_panel_grp(s, e) for (s, e) in zip(paired.sp, paired.eco)]
  panel_of(s, gp) = DF.subset(paired, [:sp, :pgrp] => DF.ByRow((a, b) -> a == s && b == gp))
  panels = sort([k for k in unique([(r.sp, r.pgrp) for r in eachrow(paired)]) if DF.nrow(panel_of(k...)) >= 3];
                by = k -> (-get(spabund, k[1], 0), k[1], k[2]))   # rank species by abundance (#plots); tiered cells together
  for mode in _modes
    isempty(panels) && continue
    r2all = overall_r2(Float64.(paired.obs_agb), Float64.(paired.sim_agb), mode)
    ncol = min(4, max(1, length(panels))); nr = cld(length(panels), ncol)
    fig = MK.Figure(size=(330 * ncol, 300 * nr + 30))
    MK.Label(fig[0, 1:ncol], "$(TAG)Sim vs obs AGB [$mode fit] — $(label) — overall R²=$(round(r2all,digits=3))"; fontsize=14, font=:bold)
    for (i, (s, gp)) in enumerate(panels)
      d = panel_of(s, gp); r, c = fldmod1(i, ncol)
      ttl = (_is_tiered(s) ? "$(species_list[s]) · $(gp)" : "$(species_list[s]) (pooled)") * "  [$(get(spabund, s, 0))p]"
      draw_panel!(fig[r, c], Float64.(d.obs_agb), Float64.(d.sim_agb), ttl, mode)
      mode == :linear && (fv = linfit(Float64.(d.obs_agb), Float64.(d.sim_agb)); push!(fitrows, (label, gp, species_list[s], DF.nrow(d), fv[2], fv[3])))
    end
    out = joinpath(outdir, "scatter_sim_obs_$(label)_$(mode).png"); MK.save(out, fig)
    println("$label [$mode]: $(length(panels)) panels ($(count(k->_is_tiered(k[1]),panels)) tiered + $(count(k->!_is_tiered(k[1]),panels)) pooled), overall R²=$(round(r2all,digits=3)) → wrote $out")
  end
  CSV.write(joinpath(outdir, "scatter_fit_$(label).csv"), fitrows)
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

function draw_panel!(cell, x, y, title, mode; cvals=nothing, crange=nothing, show_n::Bool=true)   # cvals (age-bin) → gradient colour
  n = length(x)
  cr = cvals === nothing ? (0.0, 1.0) : (crange === nothing ? (Float64(minimum(cvals)), Float64(maximum(cvals)) + 1e-9) : crange)
  if mode == :log
    ε = 1.0; lx = log10.(x .+ ε); ly = log10.(y .+ ε); a, b, r2 = linfit(lx, ly)
    xb = Statistics.mean(lx); Sxx = sum((lx .- xb) .^ 2)
    σr = n > 2 ? sqrt(sum((ly .- (a .+ b .* lx)) .^ 2) / (n - 2)) : 0.0
    ax = MK.Axis(cell; xlabel="observed AGB", ylabel="simulated AGB", xscale=log10, yscale=log10, title=(show_n ? "$title  (n=$n)" : title))
    lxs = range(minimum(lx), maximum(lx); length=60); ylf = a .+ b .* lxs
    seb = Sxx > 0 ? σr .* sqrt.(1 / n .+ (lxs .- xb) .^ 2 ./ Sxx) : zeros(length(lxs))
    xp = 10 .^ lxs
    lo = max(1.0, minimum(vcat(x, y) .+ ε)); hi = maximum(vcat(x, y) .+ ε)
    MK.lines!(ax, [lo, hi], [lo, hi]; color=:gray70, linestyle=:dash)               # y = x
    MK.band!(ax, xp, 10 .^ (ylf .- seb), 10 .^ (ylf .+ seb); color=(:firebrick, 0.18))
    MK.lines!(ax, xp, 10 .^ ylf; color=:firebrick, linewidth=2)
    MK.scatter!(ax, x .+ ε, y .+ ε; color=(cvals === nothing ? :steelblue : cvals), colormap=:viridis, colorrange=cr, markersize=9, strokewidth=0.3, strokecolor=:black)
    MK.text!(ax, 0.04, 0.96; text="R²=$(round(r2,digits=3))\nexp=$(round(b,digits=2))", space=:relative, align=(:left, :top), fontsize=9)
  elseif mode == :weightedlog
    ε = 1.0; lx = log10.(x .+ ε); ly = log10.(y .+ ε); wn = (x .+ ε) ./ Statistics.mean(x .+ ε)  # weight by AGB
    xw = sum(wn .* lx) / n; yw = sum(wn .* ly) / n; Sxxw = sum(wn .* (lx .- xw) .^ 2)
    b = Sxxw > 0 ? sum(wn .* (lx .- xw) .* (ly .- yw)) / Sxxw : 0.0; a = yw - b * xw
    resid = ly .- (a .+ b .* lx); r2 = 1 - sum(wn .* resid .^ 2) / max(sum(wn .* (ly .- yw) .^ 2), eps())
    σr = n > 2 ? sqrt(sum(wn .* resid .^ 2) / (n - 2)) : 0.0
    ax = MK.Axis(cell; xlabel="observed AGB", ylabel="simulated AGB", xscale=log10, yscale=log10, title=(show_n ? "$title  (n=$n)" : title))
    lxs = range(minimum(lx), maximum(lx); length=60); ylf = a .+ b .* lxs
    seb = Sxxw > 0 ? σr .* sqrt.(1 / n .+ (lxs .- xw) .^ 2 ./ Sxxw) : zeros(length(lxs))
    xp = 10 .^ lxs; lo = max(1.0, minimum(vcat(x, y) .+ ε)); hi = maximum(vcat(x, y) .+ ε)
    MK.lines!(ax, [lo, hi], [lo, hi]; color=:gray70, linestyle=:dash)
    MK.band!(ax, xp, 10 .^ (ylf .- seb), 10 .^ (ylf .+ seb); color=(:firebrick, 0.18))
    MK.lines!(ax, xp, 10 .^ ylf; color=:firebrick, linewidth=2)
    MK.scatter!(ax, x .+ ε, y .+ ε; color=(cvals === nothing ? :steelblue : cvals), colormap=:viridis, colorrange=cr, markersize=9, strokewidth=0.3, strokecolor=:black)
    MK.text!(ax, 0.04, 0.96; text="R²=$(round(r2,digits=3))\nexp=$(round(b,digits=2))", space=:relative, align=(:left, :top), fontsize=9)
  elseif mode == :weighted
    w = 1.0 ./ max.(x, 1.0); sw = sum(w); xw = sum(w .* x) / sw; yw = sum(w .* y) / sw
    sxx = sum(w .* (x .- xw) .^ 2); b = sxx > 0 ? sum(w .* (x .- xw) .* (y .- yw)) / sxx : 0.0; a = yw - b * xw
    resid = y .- (a .+ b .* x); r2 = 1 - sum(w .* resid .^ 2) / max(sum(w .* (y .- yw) .^ 2), eps())
    σ2 = n > 2 ? sum(resid .^ 2 ./ max.(x, 1.0)) / (n - 2) : 0.0     # residual variance per unit AGB
    ax = MK.Axis(cell; xlabel="observed AGB", ylabel="simulated AGB", title=(show_n ? "$title  (n=$n)" : title))
    m = max(maximum(x), maximum(y), 1.0) * 1.05
    xs = range(minimum(x), maximum(x); length=60); yl = a .+ b .* xs; band = sqrt(σ2) .* sqrt.(xs)  # σ·√AGB
    MK.lines!(ax, [0, m], [0, m]; color=:gray70, linestyle=:dash)
    MK.band!(ax, xs, max.(yl .- band, 0.0), yl .+ band; color=(:firebrick, 0.18))
    MK.lines!(ax, xs, yl; color=:firebrick, linewidth=2)
    MK.scatter!(ax, x, y; color=(cvals === nothing ? :steelblue : cvals), colormap=:viridis, colorrange=cr, markersize=9, strokewidth=0.3, strokecolor=:black)
    MK.text!(ax, 0.04, 0.96; text="R²=$(round(r2,digits=3))\nslope=$(round(b,digits=2))", space=:relative, align=(:left, :top), fontsize=9)
    MK.xlims!(ax, 0, m); MK.ylims!(ax, 0, m)
  else  # linear
    a, b, r2 = linfit(x, y)
    ax = MK.Axis(cell; xlabel="observed AGB", ylabel="simulated AGB", title=(show_n ? "$title  (n=$n)" : title))
    m = max(maximum(x), maximum(y), 1.0) * 1.05; xb = Statistics.mean(x); Sxx = sum((x .- xb) .^ 2)
    σr = n > 2 ? sqrt(sum((y .- (a .+ b .* x)) .^ 2) / (n - 2)) : 0.0
    xs = range(minimum(x), maximum(x); length=60); yl = a .+ b .* xs
    seb = Sxx > 0 ? σr .* sqrt.(1 / n .+ (xs .- xb) .^ 2 ./ Sxx) : zeros(length(xs))
    MK.lines!(ax, [0, m], [0, m]; color=:gray70, linestyle=:dash)
    MK.band!(ax, xs, max.(yl .- seb, 0.0), yl .+ seb; color=(:firebrick, 0.18))
    MK.lines!(ax, xs, yl; color=:firebrick, linewidth=2)
    MK.scatter!(ax, x, y; color=(cvals === nothing ? :steelblue : cvals), colormap=:viridis, colorrange=cr, markersize=9, strokewidth=0.3, strokecolor=:black)
    MK.text!(ax, 0.04, 0.96; text="R²=$(round(r2,digits=3))\nslope=$(round(b,digits=2))", space=:relative, align=(:left, :top), fontsize=9)
    MK.xlims!(ax, 0, m); MK.ylims!(ax, 0, m)
  end
end

# Sim B (free spin-up process) scatter — cohorts aren't paired one-to-one, so aggregate AGB by AGE BIN
# per (plot, year, species, bin) and pair the bins. Natural plots regenerate; artificial plots planted.
function make_scatter_B(sp, label)
  (isnothing(sp) || DF.nrow(sp) == 0) && return
  max_age = Int(maximum(sp.age_calc))
  spdf = PU.smoothen_ref_years(sp, loss_params, max_age; debug=false)
  spdf_plts = D.make_spdf_dict(spdf, eco_species_ids); ssy = D.get_site_sim_years(spdf); spc = D.get_spinup_cohorts(sp)
  art = Set(e for e in eachindex(eco_list) if occursin("artificial", lowercase(eco_list[e])))
  plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(DF.select(sp, [:plot_id, :eco_id]))))
  art_plots = Set(p for (p, e) in plot2eco if e in art)
  inj_all = D.get_injection_cohorts(sp; all_cohorts=true)
  inj_b = DF.filter(r -> Int(r.plot_id) in art_plots, inj_all)
  rs_b = P.make_sites(sp, eco_species_ids; rng=rng, spinup=true, no_establishment=false)
  idict_b = DF.nrow(inj_b) == 0 ? nothing : P._build_injection_dict(inj_b, rs_b)
  iyears_b = DF.nrow(inj_b) == 0 ? Set{Int}() : Set(Int.(inj_b.sim_year))
  msy = maximum(maximum.(filter(!isempty, ssy.sim_years)))
  res = P.fit_params(rs_b, best, msy, n_species, eco_species_ids, spdf_plts, ssy, true, spc, loss_params;
    debug=false, search_tier=3, injection_dict=idict_b, injection_years=iyears_b, seeds=[rand(rng, UInt64)])
  cached = res[1][2]; ab = loss_params.age_bins
  # FOREST age-distribution, exactly as Sim B's tier-4 objective: aggregate AGB by
  # (cycle × eco/land-use × species × age-bin) across all plots — sim age-bins vs obs age-bins.
  cyc_map, _ = D.build_cycle_map(sp; cycle_years=Float64(g("cycle_years", 8)))
  simdf = DF.DataFrame(cyc=Int[], eco=Int[], esp=Int[], bin=Int[], agb=Float64[])
  for (pid, sy, esp, age, bio) in cached
    haskey(plot2eco, Int(pid)) || continue
    cyc = get(cyc_map, (Int(pid), Int(sy)), 0); cyc == 0 && continue
    b = PU.find_age_bin(Int(ceil(Float64(age))), ab); b == 0 && continue
    push!(simdf, (cyc, plot2eco[Int(pid)], Int(esp), Int(b), Float64(bio)))
  end
  sim_agg = DF.combine(DF.groupby(simdf, [:cyc, :eco, :esp, :bin]), :agb => sum => :sim_agb)
  obsdf = DF.DataFrame(cyc=Int[], eco=Int[], esp=Int[], bin=Int[], agb=Float64[])
  for r in eachrow(DF.subset(sp, :sim_year => DF.ByRow(>(0))))
    cyc = get(cyc_map, (Int(r.plot_id), Int(r.sim_year)), 0); cyc == 0 && continue
    b = PU.find_age_bin(Int(ceil(Float64(r.age_calc))), ab); b == 0 && continue
    push!(obsdf, (cyc, Int(r.eco_id), Int(r.eco_species_id), Int(b), Float64(r.agb_sum)))
  end
  obs_agg = DF.combine(DF.groupby(obsdf, [:cyc, :eco, :esp, :bin]), :agb => sum => :obs_agb)
  paired = DF.outerjoin(obs_agg, sim_agg, on=[:cyc, :eco, :esp, :bin])   # keep bins present in either (missing→0)
  paired.obs_agb = coalesce.(paired.obs_agb, 0.0); paired.sim_agb = coalesce.(paired.sim_agb, 0.0)
  especo2sp = Dict((Int(r.eco_id), Int(r.eco_species_id)) => Int(r.species_id) for r in eachrow(unique(DF.select(sp, [:eco_id, :eco_species_id, :species_id]))))
  paired.sp = [especo2sp[(e, esp)] for (e, esp) in zip(paired.eco, paired.esp)]
  # SPLIT BY CYCLE: one figure per (cycle × eco × mode); species panels; points = age bins (colour = bin).
  for mode in (:linear, :log, :weighted, :weightedlog), cy in sort(unique(paired.cyc)), e in sort(unique(paired.eco))
    pe = DF.subset(paired, :cyc => DF.ByRow(==(cy)), :eco => DF.ByRow(==(e)))
    DF.nrow(pe) == 0 && continue
    econame = replace(eco_list[e], r"[^A-Za-z0-9]" => "_")
    r2all = overall_r2(Float64.(pe.obs_agb), Float64.(pe.sim_agb), mode)
    sps = [s for s in sort(unique(pe.sp)) if DF.nrow(DF.subset(pe, :sp => DF.ByRow(==(s)))) >= 3]
    isempty(sps) && continue
    println("$label/$econame/cyc$cy [B-binned/$mode]: $(DF.nrow(pe)) bin-pts / $(length(sps)) species  R²=$(round(r2all,digits=3))")
    ncol = min(4, max(1, length(sps))); nr = cld(length(sps), ncol)
    # consistent age-bin colour scale + labelled colorbar across all species panels of this figure
    nbk = length(bins) + 1                                   # # age classes (last bin open)
    cr_fig = (1.0, Float64(nbk))
    agelabels = vcat(["<$(b)" for b in bins], ["≥$(bins[end])"])
    fig = MK.Figure(size=(330 * ncol + 95, 300 * nr + 30))
    MK.Label(fig[0, 1:ncol], "SIM B age-bins vs obs [$mode] — $(label) · cycle $cy · $(eco_list[e]) — R²=$(round(r2all,digits=3)) (colour = age bin)"; fontsize=13, font=:bold)
    for (i, s) in enumerate(sps)
      d = DF.subset(pe, :sp => DF.ByRow(==(s))); r, c = fldmod1(i, ncol)
      draw_panel!(fig[r, c], Float64.(d.obs_agb), Float64.(d.sim_agb), species_list[s], mode; cvals=Float64.(d.bin), crange=cr_fig, show_n=false)
    end
    MK.Colorbar(fig[1:nr, ncol + 1]; colormap=:viridis, colorrange=cr_fig, label="age bin (yr)", ticks=(1:nbk, agelabels))
    out = joinpath(outdir, "scatter_simB_$(label)_$(econame)_cyc$(cy)_$(mode).png"); MK.save(out, fig); println("  wrote $out")
  end
end

make_scatter(splots, "train")          # Sim A — exact-cohort paired
make_scatter(splots_val, "val")
(!isnothing(splots_test) && get(ENV, "PAN_EVAL_TEST", "0") == "1") && make_scatter(splots_test, "test")   # 3-way test: held out unless PAN_EVAL_TEST=1
let dm = g("dual_mode", "off")          # Sim B — age-bin paired (only for dual runs)
  if dm === true || (dm isa AbstractString && lowercase(dm) in ("joint", "b"))
    make_scatter_B(splots, "train"); make_scatter_B(splots_val, "val")
  end
end
