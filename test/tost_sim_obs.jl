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
base_outdir = cfg["output_dir"]
outdir = haskey(ENV, "PAN_OUTSUB") ? (let d = joinpath(base_outdir, ENV["PAN_OUTSUB"]); mkpath(d); d end) : base_outdir
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
  st = JLD2.load_object(joinpath(base_outdir, "search_state_latest.jld2"))
  hasproperty(st, :representative) ? st.representative.x : st.best.x
end

function paired_for(sp, label)
  max_age = Int(maximum(sp.age_calc)); spdf = PU.smoothen_ref_years(sp, loss_params, max_age; debug=false)
  spdf_plts = D.make_spdf_dict(spdf, eco_species_ids); ssy = D.get_site_sim_years(spdf); spc = D.get_spinup_cohorts(sp)
  inj = (no_estab || P.OVERRIDE_INJECTION[]) ? D.get_injection_cohorts(sp; all_cohorts=P.OVERRIDE_INJECTION[]) : nothing
  rs = P.make_sites(sp, eco_species_ids; rng=rng, spinup=false, no_establishment=no_estab)
  idict = isnothing(inj) ? nothing : P._build_injection_dict(inj, rs); iyears = isnothing(inj) ? Set{Int}() : Set(Int.(inj.sim_year))
  msy = maximum(maximum.(filter(!isempty, ssy.sim_years)))
  res = P.fit_params(rs, best, msy, n_species, eco_species_ids, spdf_plts, ssy, false, spc, loss_params;
    debug=false, search_tier=3, injection_dict=idict, injection_years=iyears, seeds=[rand(rng, UInt64)])
  cached = res[1][2]
  simdf = DF.DataFrame(plot_id=Int[], sim_year=Int[], esp=Int[], agb=Float64[])
  for (pid, sy, esp, _a, bio) in cached; push!(simdf, (Int(pid), Int(sy), Int(esp), Float64(bio))); end
  sim_agg = DF.combine(DF.groupby(simdf, [:plot_id, :sim_year, :esp]), :agb => sum => :sim_agb)
  obs = DF.combine(DF.groupby(DF.subset(sp, :sim_year => DF.ByRow(>(0))), [:plot_id, :sim_year, :eco_id, :eco_species_id]), :agb_sum => sum => :obs_agb)
  DF.rename!(obs, :eco_species_id => :esp)
  paired = DF.innerjoin(obs, sim_agg, on=[:plot_id, :sim_year, :esp])
  if P.OVERRIDE_INJECTION_DISTURBANCE[] == :exclude_overwrite && !isnothing(inj)
    excl = Set((Int(r.plot_id), Int(r.sim_year), Int(r.eco_species_id)) for r in eachrow(inj) if r.disturbance_drop_pct > 0)
    paired = DF.filter(row -> !((row.plot_id, row.sim_year, row.esp) in excl), paired)
  end
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
paired_splits = [(paired_for(sp, label), label) for (sp, label) in ((splots, "train"), (splots_val, "val"))
                 if !(isnothing(sp) || DF.nrow(sp) == 0)]

for EQM in MARGINS
  Δ = log1p(EQM); pct = round(Int, 100EQM)
  rows = DF.DataFrame(split=String[], eco=String[], species=String[], n=Int[], mean_logdiff=Float64[],
    ci_lo=Float64[], ci_hi=Float64[], pct_bias=Float64[], p_tost=Float64[], equivalent=Bool[])
  for (pr, label) in paired_splits
    for e in sort(unique(pr.eco))
      pe = DF.subset(pr, :eco => DF.ByRow(==(e))); econame = replace(eco_list[e], r"[^A-Za-z0-9]" => "_")
      res = NamedTuple[]; names_ = String[]
      for s in sort(unique(pe.sp))
        d = DF.subset(pe, :sp => DF.ByRow(==(s)))
        length(d.obs_agb) < 3 && continue
        dd = log1p.(Float64.(d.sim_agb)) .- log1p.(Float64.(d.obs_agb))
        t = tost(dd, Δ); push!(res, t); push!(names_, "$(species_list[s])  (n=$(t.n))")
        push!(rows, (label, eco_list[e], species_list[s], t.n, t.mean, t.lo, t.hi, expm1(t.mean) * 100, t.p, t.equiv))
      end
      isempty(res) && continue
      # forest plot: species rows, mean log-diff ± CI, ±Δ equivalence band
      fig = MK.Figure(size=(760, 90 + 26 * length(res)))
      neq = count(r -> r.equiv, res)
      ax = MK.Axis(fig[1, 1]; xlabel="mean  log1p(sim) − log1p(obs)   (← sim low | sim high →)",
        title="TOST equivalence — $label · $(eco_list[e]) — ±$(pct)% band, α=$ALPHA — $neq/$(length(res)) equivalent",
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
      out = joinpath(outdir, "tost_$(pct)pct_$(label)_$(econame).png"); MK.save(out, fig)
      println("$label/$econame: ±$(pct)% → $neq/$(length(res)) species equivalent → $out")
    end
  end
  DF.sort!(rows, [:split, :eco, :species])
  csv = joinpath(outdir, "tost_sim_obs_$(pct)pct.csv")
  open(csv, "w") do io
    println(io, "split,eco,species,n,mean_logdiff,ci_lo,ci_hi,pct_bias,p_tost,equivalent")
    for r in eachrow(rows)
      println(io, join([r.split, r.eco, r.species, r.n, round(r.mean_logdiff, digits=4), round(r.ci_lo, digits=4),
        round(r.ci_hi, digits=4), round(r.pct_bias, digits=1), round(r.p_tost, digits=4), r.equivalent], ","))
    end
  end
  println("wrote $csv  (±$(pct)%: $(DF.nrow(rows)) rows; $(count(rows.equivalent))/$(DF.nrow(rows)) equivalent)")
end
