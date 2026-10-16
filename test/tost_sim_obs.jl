# Per-species equivalence test (paired TOST) of simulated vs observed cohort AGB, per (split, eco).
# Uses the SAME matched/stripped pairs as the scatter: sim cohort vs its observed cohort, first
# measurement excluded, overwritten (supplied) cohorts stripped in :exclude_overwrite mode.
# Test: on d = log1p(sim) − log1p(obs) (≈ log-ratio), H1 "equivalent" iff the (1−2α) CI of mean d
# lies inside ±Δ, Δ = log1p(equiv_margin). Writes tost_sim_obs.csv + a forest plot per (split, eco).
#   Run:  ./julia_gdal.sh --project=. test/tost_sim_obs.jl <config.yml> [equiv_margin=0.2] [alpha=0.05]
using Pan
import JLD2, YAML, CairoMakie, Statistics, Dates, DataFrames, HypothesisTests, Printf
const MK = CairoMakie; const P = Pan; const PU = P.PU; const D = P.Data; const DF = DataFrames; const HT = HypothesisTests

cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
MARGINS = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : [0.2]   # one or more equivalence margins
ALPHA = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.05
base_outdir = get(ENV, "PAN_OUT", "") != "" ? joinpath(ENV["PAN_OUT"], basename(String(cfg["output_dir"]))) : String(cfg["output_dir"])  # outputs under $PAN_OUT (scratch) if set
outdir = haskey(ENV, "PAN_OUTSUB") ? (let d = joinpath(base_outdir, ENV["PAN_OUTSUB"]); mkpath(d); d end) : base_outdir
TAG = haskey(ENV, "PAN_OUTSUB") ? replace(basename(ENV["PAN_OUTSUB"]), "candidate_" => "cand ") * " — " : ""
scoring = P.ScoringOptions(; override=Bool(g("override_injection", true)), sync=Bool(g("override_injection_sync", true)), replace=Bool(g("override_injection_replace", false)), disturbance=Symbol(g("override_injection_disturbance", "off"))); simb = P.SimBOptions()
scoring = P.with_options(scoring; cycle=D.CycleOptions(use_fia_cycle=Bool(g("fia_cycle", false))))   # tier-4/5 cycle bucketing (was the Data.USE_FIA_CYCLE/FIA_CYCLE_MERGE globals)
no_estab = Bool(g("no_establishment", false))
rng = P.RNGType(UInt64(Int(g("seed", 1))))
val_frac = Float64(g("val_frac", 0.0))
test_frac = Float64(g("test_frac", 0.0))               # 3-way split: reproduce the run's held-out TEST set too
n_folds = Int(g("n_folds", 1)); fold_index = parse(Int, get(ENV, "PAN_FOLD", string(g("fold_index", 1))))  # PAN_FOLD picks the CV fold
# Data prep via the SAME code path as training (Pan.prepare_from_cfg → prep_kwargs_from_cfg): identical split, region/site-class
# filters, outlier exclusions, species tiering caps. (PAN_FOLD picks the CV fold.)
splots, eco_list, species_list, eco_species_ids, splots_val, splots_test = P.prepare_from_cfg(cfg; fold_index=fold_index)
n_species = length(species_list)
# species-tier grouping (mirror the scatter): tiered species get one row per site-cell, pooled species one row;
# productivity is NOT a separate facet. Species ranked by abundance = distinct plots over all splits.
_tiered_set = Set{String}()
for (_, syms) in get(cfg, "param_split_species", Dict()), s in syms; push!(_tiered_set, uppercase(strip(String(s)))); end
_merge_map = Dict{String,Dict{String,String}}()
for (_, spmap) in get(cfg, "param_tier_merge", Dict()), (sp_, cm) in spmap
  d = get!(_merge_map, uppercase(strip(String(sp_))), Dict{String,String}()); for (c, gp) in cm; d[String(c)] = String(gp); end
end
let pp = get(cfg, "param_pooling", Dict())     # param_pooling cell_split species are per-cell too (+ its cell_merge)
  for (_, spec) in get(pp, "params", Dict()), s in get(spec, "cell_split", String[]); push!(_tiered_set, uppercase(strip(String(s)))); end
  for (sp_, cm) in get(pp, "cell_merge", Dict())
    d = get!(_merge_map, uppercase(strip(String(sp_))), Dict{String,String}()); for (c, gp) in cm; d[String(c)] = String(gp); end
  end
end
_eco_cell(e) = (p = split(String(eco_list[e]), "|lu="); length(p) == 2 ? String(p[2]) : "")
_is_tiered(s) = uppercase(species_list[s]) in _tiered_set
_panel_grp(s, e) = _is_tiered(s) ? get(get(_merge_map, uppercase(species_list[s]), Dict{String,String}()), _eco_cell(e), _eco_cell(e)) : "pooled"
_allsp = DF.DataFrame(plot_id=Int[], species_id=Int[])
for s in (splots, splots_val, splots_test); isnothing(s) && continue; append!(_allsp, DF.DataFrame(plot_id=Int.(s.plot_id), species_id=Int.(s.species_id))); end
spabund = Dict{Int,Int}(); for gdf in DF.groupby(unique(_allsp), :species_id); spabund[Int(gdf.species_id[1])] = DF.nrow(gdf); end
bins = Int.(g("bins_idx", [20, 60, 120]))
loss_params = PU.LossParams(age_bins=PU.AgeBins(bins_idx=bins, last_bin_open=true), smoothing_weights=P.FloatType[1.0], lambda=P.FloatType(g("loss_lambda", 1.0)))
best = if haskey(ENV, "PAN_PARAMS")
  JLD2.load_object(ENV["PAN_PARAMS"])
else
  st = JLD2.load_object(joinpath(base_outdir, "search_state_latest.jld2"))
  hasproperty(st, :representative) ? st.representative.x : st.best.x
end

_dm = (let d = g("dual_mode", "off"); d === true ? "joint" : lowercase(string(d)); end)   # "off"/"joint"/"b"
function paired_for(sp, label)
  spdf, spdf_plts = P.reference_tables(sp, eco_species_ids, loss_params)
  ssy = D.get_site_sim_years(spdf); spc = D.get_spinup_cohorts(sp)
  if _dm == "b"   # Sim-B free-regen: age-bin-aggregated sim vs obs via the shared helper (per-cell log-diff test)
    paired, _, _ = P.resim_simB_paired(cfg, best, sp, spdf_plts, ssy, spc, loss_params, eco_list, species_list, eco_species_ids, rng; scoring=scoring, plugin=P.bs_plugin())
    return DF.subset(paired, :sp => DF.ByRow(>(0)))
  end
  inj = (no_estab || scoring.override) ? D.get_injection_cohorts(sp; all_cohorts=scoring.override) : nothing
  rs = P.make_sites(sp, eco_species_ids; rng=rng, spinup=false, no_establishment=no_estab)
  injection = P.build_injection_plan(inj, rs, eco_species_ids, loss_params, scoring)
  msy = maximum(maximum.(filter(!isempty, ssy.sim_years)))
  res = P.fit_params(rs, best, msy, n_species, eco_species_ids, spdf_plts, ssy, false, spc, loss_params;
    debug=false, search_tier=3, injection=injection, seeds=[rand(rng, UInt64)], cache_preinject=true, scoring, simb, plugin=P.bs_plugin())
  cached = res[1][2]
  # ONLY genuinely-simulated biomass: match pre-inject model cohorts to the OBSERVED injection set by
  # (plot, year, species, AGE) = sync survivor rule; drop injected (obs-only), sync-removed (sim-only) and
  # disturbance-overwritten (drop>0) cohorts. Otherwise injection/overwrite inflate the equivalence test.
  isnothing(inj) && error("tost needs injection cohorts (override_injection) to separate simulated vs injected biomass")
  simc = DF.DataFrame(plot_id=Int[], sim_year=Int[], esp=Int[], age=Int[], sim_agb=Float64[])
  for (pid, sy, esp, age, bio) in cached; push!(simc, (Int(pid), Int(sy), Int(esp), Int(age), Float64(bio))); end
  simc = DF.combine(DF.groupby(DF.subset(simc, :sim_year => DF.ByRow(>(0))), [:plot_id, :sim_year, :esp, :age]), :sim_agb => sum => :sim_agb)
  injc = DF.DataFrame(plot_id=Int.(inj.plot_id), sim_year=Int.(inj.sim_year), esp=Int.(inj.eco_species_id),
                      age=Int.(inj.age_calc), obs_agb=Float64.(inj.agb_sum), drop=Float64.(inj.disturbance_drop_pct))
  injc = DF.combine(DF.groupby(DF.subset(injc, :sim_year => DF.ByRow(>(0))), [:plot_id, :sim_year, :esp, :age]),
                    :obs_agb => sum => :obs_agb, :drop => maximum => :drop)
  matched = DF.innerjoin(simc, injc, on=[:plot_id, :sim_year, :esp, :age])
  scoring.disturbance == :exclude_overwrite && (matched = DF.filter(r -> r.drop <= 0.0, matched))
  paired = DF.combine(DF.groupby(matched, [:plot_id, :sim_year, :esp]), :sim_agb => sum => :sim_agb, :obs_agb => sum => :obs_agb)
  plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(DF.select(sp, [:plot_id, :eco_id]))))
  especo2sp = Dict((Int(r.eco_id), Int(r.eco_species_id)) => Int(r.species_id) for r in eachrow(unique(DF.select(sp, [:eco_id, :eco_species_id, :species_id]))))
  paired.eco = [plot2eco[p] for p in paired.plot_id]; paired.sp = [especo2sp[(e, esp)] for (e, esp) in zip(paired.eco, paired.esp)]
  paired
end

function tost(d, Δ)
  n = length(d); md = Statistics.mean(d)
  n < 3 && return (n=n, mean=md, lo=NaN, hi=NaN, p=NaN, equiv=false)
  if Statistics.std(d) < 1e-9
    return (n=n, mean=md, lo=md, hi=md, p=(abs(md) < Δ ? 0.0 : 1.0), equiv=(abs(md) < Δ))
  end
  pl = HT.pvalue(HT.OneSampleTTest(d, -Δ); tail=:right)   # H1: mean > −Δ
  pu = HT.pvalue(HT.OneSampleTTest(d, Δ); tail=:left)     # H1: mean <  Δ
  ci = HT.confint(HT.OneSampleTTest(d); level=1 - 2 * ALPHA)
  (n=n, mean=md, lo=ci[1], hi=ci[2], p=max(pl, pu), equiv=(max(pl, pu) < ALPHA))
end

# compute the matched sim/obs pairs ONCE per split (the expensive fit_params step), reuse for every margin
_only_test = get(ENV, "PAN_ONLY_TEST", "0") == "1"     # --test mode: only the held-out test split
_eval_test = _only_test || get(ENV, "PAN_EVAL_TEST", "0") == "1"     # hold test out unless explicitly enabled
paired_splits = [(paired_for(sp, label), label) for (sp, label) in ((splots, "train"), (splots_val, "val"), (splots_test, "test"))
                 if !(isnothing(sp) || DF.nrow(sp) == 0) && (label != "test" || _eval_test) && !(_only_test && label != "test")]

for EQM in MARGINS
  Δ = log1p(EQM); pct = round(Int, 100EQM)
  rows = DF.DataFrame(split=String[], group=String[], species=String[], nplots=Int[], n=Int[], mean_logdiff=Float64[],
    ci_lo=Float64[], ci_hi=Float64[], pct_bias=Float64[], p_tost=Float64[], equivalent=Bool[])
  for (pr, label) in paired_splits
    pr.pgrp = [_panel_grp(s, e) for (s, e) in zip(pr.sp, pr.eco)]
    grpkeys = sort(unique([(r.sp, r.pgrp) for r in eachrow(pr)]);
                   by = k -> (-get(spabund, k[1], 0), k[1], k[2]))         # rank species by #plots; tiered cells together
    res = NamedTuple[]; names_ = String[]
    for (s, gp) in grpkeys
      d = DF.subset(pr, [:sp, :pgrp] => DF.ByRow((a, b) -> a == s && b == gp))
      length(d.obs_agb) < 3 && continue
      dd = log1p.(Float64.(d.sim_agb)) .- log1p.(Float64.(d.obs_agb))
      t = tost(dd, Δ); push!(res, t)
      lbl = _is_tiered(s) ? "$(species_list[s])·$(gp)" : species_list[s]
      push!(names_, "$(lbl)  (n=$(t.n), $(get(spabund, s, 0))p)")
      push!(rows, (label, lbl, species_list[s], get(spabund, s, 0), t.n, t.mean, t.lo, t.hi, expm1(t.mean) * 100, t.p, t.equiv))
    end
    isempty(res) && continue
    # ONE forest plot per split: rows = species×tier-group (all productivities together), ranked by abundance
    fig = MK.Figure(size=(780, 90 + 24 * length(res)))
    neq = count(r -> r.equiv, res)
    ax = MK.Axis(fig[1, 1]; xlabel="mean  log1p(sim) − log1p(obs)   (← sim low | sim high →)",
      title="$(TAG)TOST equivalence — $label — ±$(pct)% band, α=$ALPHA — $neq/$(length(res)) equivalent  (species ranked by #plots)",
      yticks=(1:length(res), names_))
    MK.vspan!(ax, -Δ, Δ; color=(:seagreen, 0.10))                       # equivalence band
    MK.vlines!(ax, [-Δ, Δ]; color=:seagreen, linestyle=:dash); MK.vlines!(ax, [0.0]; color=:gray60)
    for (i, r) in enumerate(res)
      col = r.equiv ? :seagreen : :firebrick
      isnan(r.lo) || MK.lines!(ax, [r.lo, r.hi], [i, i]; color=col, linewidth=2)
      MK.scatter!(ax, [r.mean], [i]; color=col, markersize=11)
      MK.text!(ax, Δ * 1.05, i; text=Printf.@sprintf("%+.0f%%  p=%.3f", expm1(r.mean) * 100, r.p), align=(:left, :center), fontsize=9, color=col)
    end
    MK.xlims!(ax, min(-2Δ, minimum(r -> isnan(r.lo) ? r.mean : r.lo, res) * 1.1), max(3Δ, maximum(r -> isnan(r.hi) ? r.mean : r.hi, res) * 1.3))
    out = joinpath(outdir, "tost_$(pct)pct_$(label).png"); MK.save(out, fig)
    println("$label: ±$(pct)% → $neq/$(length(res)) groups equivalent → $out")
  end
  DF.sort!(rows, [:split, DF.order(:nplots, rev=true), :group])
  csv = joinpath(outdir, "tost_sim_obs_$(pct)pct.csv")
  open(csv, "w") do io
    println(io, "split,group,species,nplots,n,mean_logdiff,ci_lo,ci_hi,pct_bias,p_tost,equivalent")
    for r in eachrow(rows)
      println(io, join([r.split, r.group, r.species, r.nplots, r.n, round(r.mean_logdiff, digits=4), round(r.ci_lo, digits=4),
        round(r.ci_hi, digits=4), round(r.pct_bias, digits=1), round(r.p_tost, digits=4), r.equivalent], ","))
    end
  end
  println("wrote $csv  (±$(pct)%: $(DF.nrow(rows)) rows; $(count(rows.equivalent))/$(DF.nrow(rows)) equivalent)")
end
