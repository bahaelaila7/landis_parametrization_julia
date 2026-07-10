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
base_outdir = cfg["output_dir"]
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
  filter_extent=(haskey(cfg, "filter_extent") ? String(cfg["filter_extent"]) : nothing), RNG=rng)
n_species = length(species_list)
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
  # sim: bin each cohort's AGE into the coarse grid, sum bio per (plot, year, esp, cbin)
  sim = DF.DataFrame(plot_id=Int[], sim_year=Int[], esp=Int[], cbin=Int[], sim_agb=Float64[])
  for (pid, sy, esp, age, bio) in cached
    b = PU.find_age_bin(Int(round(age)), coarse); b >= 1 && push!(sim, (Int(pid), Int(sy), Int(esp), b, Float64(bio)))
  end
  sim = DF.combine(DF.groupby(sim, [:plot_id, :sim_year, :esp, :cbin]), :sim_agb => sum => :sim_agb)
  # obs: same binning on observed cohorts (exclude first measurement sim_year 0)
  obs = DF.subset(sp, :sim_year => DF.ByRow(>(0)))
  obs = DF.DataFrame(plot_id=Int.(obs.plot_id), sim_year=Int.(obs.sim_year), esp=Int.(obs.eco_species_id),
    cbin=[PU.find_age_bin(Int(round(a)), coarse) for a in obs.age_calc], obs_agb=Float64.(obs.agb_sum))
  obs = DF.subset(obs, :cbin => DF.ByRow(>=(1)))
  obs = DF.combine(DF.groupby(obs, [:plot_id, :sim_year, :esp, :cbin]), :obs_agb => sum => :obs_agb)
  paired = DF.outerjoin(obs, sim, on=[:plot_id, :sim_year, :esp, :cbin])  # outer: penalize sim-only or obs-only bins
  paired.obs_agb = coalesce.(paired.obs_agb, 0.0); paired.sim_agb = coalesce.(paired.sim_agb, 0.0)
  if P.OVERRIDE_INJECTION_DISTURBANCE[] == :exclude_overwrite && !isnothing(inj)
    excl = Set((Int(r.plot_id), Int(r.sim_year), Int(r.eco_species_id)) for r in eachrow(inj) if r.disturbance_drop_pct > 0)
    paired = DF.filter(row -> !((row.plot_id, row.sim_year, row.esp) in excl), paired)
  end
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
function draw_err!(outpath, M, sps, title, cbarlabel; cmap=GYR, crange=(0.0, 100.0))
  fig = MK.Figure(size=(150 + 70 * nb, 130 + 26 * length(sps)))
  ax = MK.Axis(fig[1, 1]; title=title, xlabel="coarse age bin (yr)", ylabel="species",
    xticks=(1:nb, binlabels), yticks=(1:length(sps), species_list[sps]), xticklabelrotation=π / 4)
  hm = MK.heatmap!(ax, 1:nb, 1:length(sps), permutedims(M); colormap=cmap, colorrange=crange, nan_color=(:gray85))
  for si in 1:length(sps), b in 1:nb
    isnan(M[si, b]) && continue
    MK.text!(ax, b, si; text=string(round(Int, M[si, b])), align=(:center, :center), fontsize=8, color=:black)
  end
  MK.Colorbar(fig[1, 2], hm; label=cbarlabel)
  MK.save(outpath, fig)
end

rows = DF.DataFrame(split=String[], stratum=String[], species=String[], agebin=String[], n=Int[],
  sim_agb=Float64[], obs_agb=Float64[], avg_ref=Float64[], avg_sim=Float64[], sMAPE=Float64[], MAPE=Float64[], forestMAPE=Float64[])
for (sp, label) in ((splots, "train"), (splots_val, "val"), (splots_test, "test"))
  isnothing(sp) && continue                     # 3-way: held-out test set (nothing unless test_frac>0)
  label == "test" && get(ENV, "PAN_EVAL_TEST", "0") != "1" && continue   # hold test out unless enabled
  pr = paired_agebin(sp, label); isnothing(pr) && continue
  for e in sort(unique(pr.eco))
    pe = DF.subset(pr, :eco => DF.ByRow(==(e))); econame = replace(eco_list[e], r"[^A-Za-z0-9]" => "_")
    sps = sort(unique(pe.sp))
    Ms = fill(NaN, length(sps), nb); Mm = fill(NaN, length(sps), nb); Mf = fill(NaN, length(sps), nb)  # sMAPE, MAPE, forest-level MAPE [species × agebin]
    for (si, s) in enumerate(sps), b in 1:nb
      d = DF.subset(pe, :sp => DF.ByRow(==(s)), :cbin => DF.ByRow(==(b)))
      DF.nrow(d) == 0 && continue
      sm = Statistics.mean(smape_pair.(d.sim_agb, d.obs_agb))
      mp = let v = filter(!isnan, mape_pair.(d.sim_agb, d.obs_agb)); isempty(v) ? NaN : Statistics.mean(v) end
      # FOREST-LEVEL MAPE: average biomass across the plots (the "forest") FIRST, then the % error —
      # signed 100·(avg_ref − avg_sim)/avg_ref, so per-plot over/under-predictions cancel (landscape total).
      aref = Statistics.mean(d.obs_agb); asim = Statistics.mean(d.sim_agb)
      fm = aref > 0 ? 100.0 * (aref - asim) / aref : NaN
      Ms[si, b] = sm; Mm[si, b] = mp; Mf[si, b] = fm
      push!(rows, (label, eco_list[e], species_list[s], binlabels[b], DF.nrow(d), sum(d.sim_agb), sum(d.obs_agb), aref, asim, sm, mp, fm))
    end
    draw_err!(joinpath(outdir, "smape_agebin_$(label)_$(econame).png"), Ms, sps,
      "$(TAG)AGB sMAPE by age bin — $label · $(eco_list[e])", "sMAPE %  (green=low, red≥100)"; cmap=GYR, crange=(0.0, 100.0))
    draw_err!(joinpath(outdir, "mape_agebin_$(label)_$(econame).png"), Mm, sps,
      "$(TAG)AGB MAPE by age bin — $label · $(eco_list[e])", "MAPE %  (green=low, red≥100)"; cmap=GYR, crange=(0.0, 100.0))
    draw_err!(joinpath(outdir, "forest_mape_agebin_$(label)_$(econame).png"), Mf, sps,
      "$(TAG)Forest-level MAPE by age bin — $label · $(eco_list[e])  [100·(avgRef−avgSim)/avgRef]",
      "signed %  (blue: sim over-predicts | red: sim under-predicts)"; cmap=DIV, crange=(-100.0, 100.0))
    println("$label/$(eco_list[e]): $(length(sps)) species × $nb bins → smape+mape+forestMAPE png")
  end
end
csv = joinpath(outdir, "smape_agebin.csv")
open(csv, "w") do io
  println(io, "split,stratum,species,agebin,n_pairs,sim_agb,obs_agb,avg_ref,avg_sim,sMAPE_pct,MAPE_pct,forestMAPE_pct")
  for r in eachrow(rows); println(io, "$(r.split),$(r.stratum),$(r.species),$(r.agebin),$(r.n),$(r.sim_agb),$(r.obs_agb),$(r.avg_ref),$(r.avg_sim),$(r.sMAPE),$(r.MAPE),$(r.forestMAPE)"); end
end
println("wrote $csv  ($(DF.nrow(rows)) cells)")
