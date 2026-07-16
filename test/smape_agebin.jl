# AGB error by COARSE age bin — symmetric MAPE (sMAPE), per species, per stratum (eco×lu).
# Age-RESOLVED: matches sim cohort ↔ obs cohort by (plot, sim_year, eco_species, coarse-age-bin), so we
# see WHERE on the age axis a species' biomass is over/under-predicted. sMAPE(cell) = mean over matched
# (plot,year) pairs of 200·|sim−obs|/(sim+obs) [%]. First measurement (sim_year 0) excluded; supplied/
# overwritten (drop>0) cohorts stripped in :exclude_overwrite — same pairs as the scatter/TOST.
#   Run: PAN_PARAMS=<params.jld2> PAN_OUTSUB=<sub> ./julia_gdal.sh --project=. test/smape_agebin.jl <config.yml>
using Pan
import JLD2, YAML, CairoMakie, Statistics, Dates, DataFrames, Printf
const MK = CairoMakie; const P = Pan; const PU = P.PU; const D = P.Data; const DF = DataFrames
cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
base_outdir = get(ENV, "PAN_OUT", "") != "" ? joinpath(ENV["PAN_OUT"], basename(String(cfg["output_dir"]))) : String(cfg["output_dir"])  # outputs under $PAN_OUT (scratch) if set
outdir = haskey(ENV, "PAN_OUTSUB") ? (let d = joinpath(base_outdir, ENV["PAN_OUTSUB"]); mkpath(d); d end) : base_outdir
TAG = haskey(ENV, "PAN_OUTSUB") ? replace(basename(ENV["PAN_OUTSUB"]), "candidate_" => "cand ") * " — " : ""   # candidate id in titles
P.OVERRIDE_INJECTION[] = Bool(g("override_injection", true)); P.OVERRIDE_INJECTION_SYNC[] = Bool(g("override_injection_sync", true))
P.OVERRIDE_INJECTION_REPLACE[] = Bool(g("override_injection_replace", false))
P.OVERRIDE_INJECTION_DISTURBANCE[] = Symbol(g("override_injection_disturbance", "off"))
D.USE_FIA_CYCLE[] = Bool(g("fia_cycle", false)); D.FIA_CYCLE_MERGE[] = Int(g("fia_cycle_merge", 1))
P.INIT_PERTURB_FRAC[] = 0.0; no_estab = Bool(g("no_establishment", false)); rng = P.RNGType(UInt64(Int(g("seed", 1))))
val_frac = Float64(g("val_frac", 0.0))
test_frac = Float64(g("test_frac", 0.0))               # 3-way split: reproduce the run's held-out TEST set too
n_folds = Int(g("n_folds", 1)); fold_index = parse(Int, get(ENV, "PAN_FOLD", string(g("fold_index", 1))))  # PAN_FOLD picks the CV fold
split_rng = (val_frac > 0 || n_folds > 1 || test_frac > 0) ? P.RNGType(UInt64(Int(g("split_seed", 42)))) : nothing
excl_plots = NTuple{4,Int}[]
let epc = g("exclude_plots_csv", nothing)
  if !(epc === nothing || epc == "null")
    for ln in Iterators.drop(eachline(String(epc)), 1); isempty(strip(ln)) && continue; v = parse.(Int, split(ln, ",")); push!(excl_plots, (v[1], v[2], v[3], v[4])); end
  end
end
splots, eco_list, species_list, eco_species_ids, splots_val, splots_test = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]),
  output_dir=String(cfg["tablename"]), filter_eco_field=String(g("filter_eco_field", "epa_l4")),
  filter_ecos=String.(get(cfg, "filter_ecos", String[])),
  n_folds=n_folds, fold_index=fold_index, exclude_plots=excl_plots,
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false, min_trees=Int(g("min_trees", 100)),
  min_agb_frac=Float64(g("min_agb_frac", 0.05)), single_ecoregion=Bool(g("single_ecoregion", false)),
  stratify_landuse=Bool(g("stratify_landuse", false)), val_frac=val_frac, split_rng=split_rng, test_frac=test_frac,
  site_class_strata=Bool(g("site_class_strata", false)), siteclass_hi_max=Int(g("siteclass_hi_max", 4)),
  siteclass_scheme=String(g("siteclass_scheme", "2way")),
  shade_tier_csv=(haskey(cfg, "shade_tier_csv") ? String(cfg["shade_tier_csv"]) : nothing),
  filter_extent=(haskey(cfg, "filter_extent") ? String(cfg["filter_extent"]) : nothing), RNG=rng)
n_species = length(species_list)
# species-tier grouping (mirror the scatter): tiered species per site-cell, pooled species one row; NO productivity
# facet. Species ranked by abundance = distinct plots over all splits.
_tiered_set = Set{String}()
for (_, syms) in get(cfg, "param_split_species", Dict()), s in syms; push!(_tiered_set, uppercase(strip(String(s)))); end
_merge_map = Dict{String,Dict{String,String}}()
for (_, spmap) in get(cfg, "param_tier_merge", Dict()), (sp_, cm) in spmap
  d = get!(_merge_map, uppercase(strip(String(sp_))), Dict{String,String}()); for (c, gp) in cm; d[String(c)] = String(gp); end
end
_eco_cell(e) = (p = split(String(eco_list[e]), "|lu="); length(p) == 2 ? String(p[2]) : "")
_is_tiered(s) = uppercase(species_list[s]) in _tiered_set
_panel_grp(s, e) = _is_tiered(s) ? get(get(_merge_map, uppercase(species_list[s]), Dict{String,String}()), _eco_cell(e), _eco_cell(e)) : "pooled"
_allsp = DF.DataFrame(plot_id=Int[], species_id=Int[])
for s in (splots, splots_val, splots_test); isnothing(s) && continue; append!(_allsp, DF.DataFrame(plot_id=Int.(s.plot_id), species_id=Int.(s.species_id))); end
spabund = Dict{Int,Int}(); for gdf in DF.groupby(unique(_allsp), :species_id); spabund[Int(gdf.species_id[1])] = DF.nrow(gdf); end
bins = Int.(g("bins_idx", [20, 60, 120])); coarse = PU.AgeBins(bins_idx=bins, last_bin_open=true)
binlabels = vcat(["≤$(bins[1])"], ["$(bins[i-1])–$(bins[i])" for i in 2:length(bins)], ["$(bins[end])+"])
nb = length(bins) + 1
loss_params = PU.LossParams(age_bins=coarse, smoothing_weights=P.FloatType[1.0], lambda=one(P.FloatType))

best = haskey(ENV, "PAN_PARAMS") ? JLD2.load_object(ENV["PAN_PARAMS"]) :
  (let st = JLD2.load_object(joinpath(base_outdir, "search_state_latest.jld2")); hasproperty(st, :representative) ? st.representative.x : st.best.x end)

# age-resolved matched pairs per (plot, sim_year, eco_species, coarse-bin): sim_agb vs obs_agb
function paired_agebin(sp, label)
  (isnothing(sp) || DF.nrow(sp) == 0) && return nothing
  max_age = Int(maximum(sp.age_calc)); spdf = PU.smoothen_ref_years(sp, loss_params, max_age; debug=false)
  spdf_plts = D.make_spdf_dict(spdf, eco_species_ids); ssy = D.get_site_sim_years(spdf); spc = D.get_spinup_cohorts(sp)
  inj = (no_estab || P.OVERRIDE_INJECTION[]) ? D.get_injection_cohorts(sp; all_cohorts=P.OVERRIDE_INJECTION[]) : nothing
  rs = P.make_sites(sp, eco_species_ids; rng=rng, spinup=false, no_establishment=no_estab)
  idict = isnothing(inj) ? nothing : P._build_injection_dict(inj, rs); iyears = isnothing(inj) ? Set{Int}() : Set(Int.(inj.sim_year))
  msy = maximum(maximum.(filter(!isempty, ssy.sim_years)))
  res = P.fit_params(rs, best, msy, n_species, eco_species_ids, spdf_plts, ssy, false, spc, loss_params;
    debug=false, search_tier=3, injection_dict=idict, injection_years=iyears, seeds=[rand(rng, UInt64)], cache_preinject=(get(ENV,"PAN_PREINJECT","1")!="0"))
  cached = res[1][2]
  # ONLY genuinely-simulated biomass: match pre-inject model cohorts to the OBSERVED injection set by
  # (plot, year, species, exact AGE) = sync survivor rule, drop disturbance-overwritten (drop>0), THEN bin by
  # coarse age. Drops injected (obs-only) & sync-removed (sim-only) cohorts (they'd otherwise distort sMAPE).
  isnothing(inj) && error("smape needs injection cohorts (override_injection) to separate simulated vs injected biomass")
  simc = DF.DataFrame(plot_id=Int[], sim_year=Int[], esp=Int[], age=Int[], sim_agb=Float64[])
  for (pid, sy, esp, age, bio) in cached; push!(simc, (Int(pid), Int(sy), Int(esp), Int(age), Float64(bio))); end
  simc = DF.combine(DF.groupby(DF.subset(simc, :sim_year => DF.ByRow(>(0))), [:plot_id, :sim_year, :esp, :age]), :sim_agb => sum => :sim_agb)
  injc = DF.DataFrame(plot_id=Int.(inj.plot_id), sim_year=Int.(inj.sim_year), esp=Int.(inj.eco_species_id),
                      age=Int.(inj.age_calc), obs_agb=Float64.(inj.agb_sum), drop=Float64.(inj.disturbance_drop_pct))
  injc = DF.combine(DF.groupby(DF.subset(injc, :sim_year => DF.ByRow(>(0))), [:plot_id, :sim_year, :esp, :age]),
                    :obs_agb => sum => :obs_agb, :drop => maximum => :drop)
  matched = DF.innerjoin(simc, injc, on=[:plot_id, :sim_year, :esp, :age])   # sync survivors, exact (species,age)
  P.OVERRIDE_INJECTION_DISTURBANCE[] == :exclude_overwrite && (matched = DF.filter(r -> r.drop <= 0.0, matched))
  matched.cbin = [PU.find_age_bin(a, coarse) for a in matched.age]
  matched = DF.subset(matched, :cbin => DF.ByRow(>=(1)))
  paired = DF.combine(DF.groupby(matched, [:plot_id, :sim_year, :esp, :cbin]), :sim_agb => sum => :sim_agb, :obs_agb => sum => :obs_agb)
  plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(DF.select(sp, [:plot_id, :eco_id]))))
  especo2sp = Dict((Int(r.eco_id), Int(r.eco_species_id)) => Int(r.species_id) for r in eachrow(unique(DF.select(sp, [:eco_id, :eco_species_id, :species_id]))))
  paired.eco = [plot2eco[p] for p in paired.plot_id]
  paired.sp = [get(especo2sp, (e, esp), 0) for (e, esp) in zip(paired.eco, paired.esp)]
  DF.subset(paired, :sp => DF.ByRow(>(0)))
end

smape_pair(s, o) = (s + o) > 0 ? 200.0 * abs(s - o) / (s + o) : 0.0   # symmetric % ; both-zero → 0
mape_pair(s, o)  = o > 0 ? 100.0 * abs(s - o) / o : NaN               # standard % (rel. to obs); obs=0 → undefined

const GYR = MK.cgrad(:RdYlGn; rev=true)         # green (low err) → yellow → red (high err)
const DIV = MK.cgrad(:RdBu; rev=true)            # signed bias: blue (sim over-predicts) → white(0) → red (sim under-predicts)
# heatmap of an [species × agebin] error matrix M, values annotated
function draw_err!(outpath, M, labels, title, cbarlabel; cmap=GYR, crange=(0.0, 100.0))
  fig = MK.Figure(size=(180 + 70 * nb, 130 + 26 * length(labels)))
  ax = MK.Axis(fig[1, 1]; title=title, xlabel="coarse age bin (yr)", ylabel="species (ranked by #plots)",
    xticks=(1:nb, binlabels), yticks=(1:length(labels), labels), xticklabelrotation=π / 4)
  hm = MK.heatmap!(ax, 1:nb, 1:length(labels), permutedims(M); colormap=cmap, colorrange=crange, nan_color=(:gray85))
  for si in 1:length(labels), b in 1:nb
    isnan(M[si, b]) && continue
    MK.text!(ax, b, si; text=string(round(Int, M[si, b])), align=(:center, :center), fontsize=8, color=:black)
  end
  MK.Colorbar(fig[1, 2], hm; label=cbarlabel)
  MK.save(outpath, fig)
end

rows = DF.DataFrame(split=String[], group=String[], species=String[], nplots=Int[], agebin=String[], n=Int[],
  sim_agb=Float64[], obs_agb=Float64[], avg_ref=Float64[], avg_sim=Float64[], sMAPE=Float64[], MAPE=Float64[], forestMAPE=Float64[])
_only_test = get(ENV, "PAN_ONLY_TEST", "0") == "1"     # --test mode: only the held-out test split
for (sp, label) in ((splots, "train"), (splots_val, "val"), (splots_test, "test"))
  isnothing(sp) && continue                     # 3-way: held-out test set (nothing unless test_frac>0)
  _only_test && label != "test" && continue     # --test: skip train/val
  label == "test" && !(_only_test || get(ENV, "PAN_EVAL_TEST", "0") == "1") && continue   # hold test out unless enabled
  pr = paired_agebin(sp, label); isnothing(pr) && continue
  # ONE heatmap per split: y-axis = species×tier-group (all productivities together), ranked by abundance
  pr.pgrp = [_panel_grp(s, e) for (s, e) in zip(pr.sp, pr.eco)]
  gkeys = sort(unique([(r.sp, r.pgrp) for r in eachrow(pr)]); by = k -> (-get(spabund, k[1], 0), k[1], k[2]))
  labels = [(_is_tiered(s) ? "$(species_list[s])·$(gp)" : species_list[s]) * "  ($(get(spabund, s, 0))p)" for (s, gp) in gkeys]
  ng = length(gkeys)
  Ms = fill(NaN, ng, nb); Mm = fill(NaN, ng, nb); Mf = fill(NaN, ng, nb)  # sMAPE, MAPE, forest-level MAPE [group × agebin]
  for (gi, (s, gp)) in enumerate(gkeys), b in 1:nb
    d = DF.subset(pr, [:sp, :pgrp] => DF.ByRow((a, c) -> a == s && c == gp), :cbin => DF.ByRow(==(b)))
    DF.nrow(d) == 0 && continue
    sm = Statistics.mean(smape_pair.(d.sim_agb, d.obs_agb))
    mp = let v = filter(!isnan, mape_pair.(d.sim_agb, d.obs_agb)); isempty(v) ? NaN : Statistics.mean(v) end
    # FOREST-LEVEL MAPE: average biomass across the plots (the "forest") FIRST, then the % error —
    # signed 100·(avg_ref − avg_sim)/avg_ref, so per-plot over/under-predictions cancel (landscape total).
    aref = Statistics.mean(d.obs_agb); asim = Statistics.mean(d.sim_agb)
    fm = aref > 0 ? 100.0 * (aref - asim) / aref : NaN
    Ms[gi, b] = sm; Mm[gi, b] = mp; Mf[gi, b] = fm
    lbl = _is_tiered(s) ? "$(species_list[s])·$(gp)" : species_list[s]
    push!(rows, (label, lbl, species_list[s], get(spabund, s, 0), binlabels[b], DF.nrow(d), sum(d.sim_agb), sum(d.obs_agb), aref, asim, sm, mp, fm))
  end
  draw_err!(joinpath(outdir, "smape_agebin_$(label).png"), Ms, labels,
    "$(TAG)AGB sMAPE by age bin — $label  (species ranked by #plots)", "sMAPE %  (green=low, red≥100)"; cmap=GYR, crange=(0.0, 100.0))
  draw_err!(joinpath(outdir, "mape_agebin_$(label).png"), Mm, labels,
    "$(TAG)AGB MAPE by age bin — $label", "MAPE %  (green=low, red≥100)"; cmap=GYR, crange=(0.0, 100.0))
  draw_err!(joinpath(outdir, "forest_mape_agebin_$(label).png"), Mf, labels,
    "$(TAG)Forest-level MAPE by age bin — $label  [100·(avgRef−avgSim)/avgRef]",
    "signed %  (blue: sim over-predicts | red: sim under-predicts)"; cmap=DIV, crange=(-100.0, 100.0))
  println("$label: $ng groups × $nb bins → smape+mape+forestMAPE png")
end
csv = joinpath(outdir, "smape_agebin.csv")
open(csv, "w") do io
  println(io, "split,group,species,nplots,agebin,n_pairs,sim_agb,obs_agb,avg_ref,avg_sim,sMAPE_pct,MAPE_pct,forestMAPE_pct")
  for r in eachrow(rows); println(io, "$(r.split),$(r.group),$(r.species),$(r.nplots),$(r.agebin),$(r.n),$(r.sim_agb),$(r.obs_agb),$(r.avg_ref),$(r.avg_sim),$(r.sMAPE),$(r.MAPE),$(r.forestMAPE)"); end
end
println("wrote $csv  ($(DF.nrow(rows)) cells)")
