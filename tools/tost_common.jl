# Shared by tools/tost_select.jl and tools/calib_coord.jl: data prep (identical to training via Pan.prepare_from_cfg),
# matched sim/obs cohort pairs (same rule as test/tost_sim_obs.jl), species-group labels, TOST.
using Pan
import JLD2, YAML, Statistics, DataFrames, HypothesisTests, CSV
const P = Pan; const PU = P.PU; const D = P.Data; const DF = DataFrames; const HT = HypothesisTests
# caller defines: cfg (Dict), SPLIT ("train"|"val"|"test") before including this file
g(k, d) = get(cfg, k, d)
ALPHA = 0.05
P.OVERRIDE_INJECTION[] = Bool(g("override_injection", true)); P.OVERRIDE_INJECTION_SYNC[] = Bool(g("override_injection_sync", true))
P.OVERRIDE_INJECTION_REPLACE[] = Bool(g("override_injection_replace", false))
P.OVERRIDE_INJECTION_DISTURBANCE[] = Symbol(g("override_injection_disturbance", "off"))
D.USE_FIA_CYCLE[] = Bool(g("fia_cycle", false)); P.INIT_PERTURB_FRAC[] = 0.0
no_estab = Bool(g("no_establishment", false)); rng0 = Int(g("seed", 1))
splots, eco_list, species_list, eco_species_ids, splots_val, splots_test = P.prepare_from_cfg(cfg)
# SPLIT: "train" | "val" | "test", or several joined by "+" (e.g. "train+val"): pairs of every listed split are pooled
# (plot ids offset per split so they stay unique). Each split keeps its own simulation inputs.
_split_df(nm) = nm == "train" ? splots : nm == "val" ? splots_val : nm == "test" ? splots_test : error("unknown split $nm")
n_species = length(species_list)
# grouping identical to tost_sim_obs.jl (tiered = param_split_species ∪ param_pooling cell_split, with merges)
tiered = Set{String}(); merge_map = Dict{String,Dict{String,String}}()
for (_, syms) in get(cfg, "param_split_species", Dict()), s in syms; push!(tiered, uppercase(strip(String(s)))); end
for (_, spmap) in get(cfg, "param_tier_merge", Dict()), (s_, cm) in spmap
  d = get!(merge_map, uppercase(strip(String(s_))), Dict{String,String}()); for (c, gp) in cm; d[String(c)] = String(gp); end
end
let pp = get(cfg, "param_pooling", Dict())
  for (_, spec) in get(pp, "params", Dict()), s in get(spec, "cell_split", String[]); push!(tiered, uppercase(strip(String(s)))); end
  for (s_, cm) in get(pp, "cell_merge", Dict())
    d = get!(merge_map, uppercase(strip(String(s_))), Dict{String,String}()); for (c, gp) in cm; d[String(c)] = String(gp); end
  end
end
eco_cell(e) = (q = split(String(eco_list[e]), "|lu="); length(q) == 2 ? String(q[2]) : "")
is_tiered(s) = uppercase(species_list[s]) in tiered
grp(s, e) = is_tiered(s) ? get(get(merge_map, uppercase(species_list[s]), Dict{String,String}()), eco_cell(e), eco_cell(e)) : "pooled"
label(s, gp) = is_tiered(s) ? "$(species_list[s])·$(gp)" : species_list[s]
bins = Int.(g("bins_idx", [20, 60, 120]))
loss_params = PU.LossParams(age_bins=PU.AgeBins(bins_idx=bins, last_bin_open=true), smoothing_weights=P.FloatType[1.0], lambda=P.FloatType(g("loss_lambda", 1.0)))
function _split_ctx(sp)
  max_age = Int(maximum(sp.age_calc)); spdf = PU.smoothen_ref_years(sp, loss_params, max_age; debug=false)
  spdf_plts = D.make_spdf_dict(spdf, eco_species_ids); ssy = D.get_site_sim_years(spdf); spc = D.get_spinup_cohorts(sp)
  inj = D.get_injection_cohorts(sp; all_cohorts=P.OVERRIDE_INJECTION[])
  msy = maximum(maximum.(filter(!isempty, ssy.sim_years)))
  plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(DF.select(sp, [:plot_id, :eco_id]))))
  especo2sp = Dict((Int(r.eco_id), Int(r.eco_species_id)) => Int(r.species_id) for r in eachrow(unique(DF.select(sp, [:eco_id, :eco_species_id, :species_id]))))
  ph = hasproperty(inj, :phantom_frac) ? Float64.(inj.phantom_frac) : zeros(DF.nrow(inj))
  injc = DF.DataFrame(plot_id=Int.(inj.plot_id), sim_year=Int.(inj.sim_year), esp=Int.(inj.eco_species_id), age=Int.(inj.age_calc),
                      obs_agb=Float64.(inj.agb_sum), drop=Float64.(inj.disturbance_drop_pct), phantom=ph)
  # REAL_ONLY support: phantom share of every observed cohort (all years incl. 0) + each plot's previous visit year
  phmap = Dict((r.plot_id, r.sim_year, r.esp, r.age) => r.phantom for r in eachrow(injc))
  prevyr = Dict{Tuple{Int,Int},Int}()
  for gd in DF.groupby(unique(DF.select(injc, [:plot_id, :sim_year])), :plot_id)
    ys = sort(gd.sim_year); for k in 2:length(ys); prevyr[(gd.plot_id[1], ys[k])] = ys[k-1]; end
  end
  injc = DF.combine(DF.groupby(DF.subset(injc, :sim_year => DF.ByRow(>(0))), [:plot_id, :sim_year, :esp, :age]),
                    :obs_agb => sum => :obs_agb, :drop => maximum => :drop, :phantom => maximum => :phantom)
  (; sp, spdf_plts, ssy, spc, inj, msy, plot2eco, especo2sp, injc, phmap, prevyr)
end
SPLITS = String.(split(SPLIT, "+"))
REAL_ONLY = get(ENV, "TOST_REAL_ONLY", "0") == "1"
REAL_ONLY && println("TOST_REAL_ONLY: pairs restricted to real (non-phantom) observed cohorts with a real previous-visit state")
CTXS = [_split_ctx(_split_df(nm)) for nm in SPLITS]
sp = CTXS[1].sp   # (first split; used by callers for counts)
function _paired_one(params, c, offset)
  rng = P.RNGType(UInt64(rng0))
  rs = P.make_sites(c.sp, eco_species_ids; rng=rng, spinup=false, no_establishment=no_estab)
  idict = P._build_injection_dict(c.inj, rs); iyears = Set(Int.(c.inj.sim_year))
  res = P.fit_params(rs, params, c.msy, n_species, eco_species_ids, c.spdf_plts, c.ssy, false, c.spc, loss_params;
    debug=false, search_tier=3, injection_dict=idict, injection_years=iyears, seeds=[rand(rng, UInt64)], cache_preinject=true)
  simc = DF.DataFrame(plot_id=Int[], sim_year=Int[], esp=Int[], age=Int[], sim_agb=Float64[])
  for (pid, sy, esp, age, bio) in res[1][2]; push!(simc, (Int(pid), Int(sy), Int(esp), Int(age), Float64(bio))); end
  simc = DF.combine(DF.groupby(DF.subset(simc, :sim_year => DF.ByRow(>(0))), [:plot_id, :sim_year, :esp, :age]), :sim_agb => sum => :sim_agb)
  m = DF.innerjoin(simc, c.injc, on=[:plot_id, :sim_year, :esp, :age])
  P.OVERRIDE_INJECTION_DISTURBANCE[] == :exclude_overwrite && (m = DF.filter(r -> r.drop <= 0.0, m))
  if REAL_ONLY   # D9: compare only REAL measurements whose simulated interval also STARTED from a real record
    keep = map(eachrow(m)) do r
      r.phantom >= 0.5 && return false
      py = get(c.prevyr, (r.plot_id, r.sim_year), nothing); py === nothing && return true
      get(c.phmap, (r.plot_id, py, r.esp, r.age - (r.sim_year - py)), 0.0) < 0.5
    end
    m = m[keep, :]
  end
  pr = DF.combine(DF.groupby(m, [:plot_id, :sim_year, :esp]), :sim_agb => sum => :sim_agb, :obs_agb => sum => :obs_agb)
  pr.eco = [c.plot2eco[p] for p in pr.plot_id]; pr.sp = [c.especo2sp[(e, esp)] for (e, esp) in zip(pr.eco, pr.esp)]
  pr.pgrp = [grp(s, e) for (s, e) in zip(pr.sp, pr.eco)]
  pr.plot_id .+= offset
  pr
end
paired(params) = reduce(vcat, [_paired_one(params, c, 10_000_000 * (k - 1)) for (k, c) in enumerate(CTXS)])
function tost(d, Δ)
  n = length(d); md = Statistics.mean(d)
  n < 3 && return (n=n, mean=md, lo=NaN, hi=NaN, p=NaN, equiv=false)
  Statistics.std(d) < 1e-9 && return (n=n, mean=md, lo=md, hi=md, p=(abs(md) < Δ ? 0.0 : 1.0), equiv=(abs(md) < Δ))
  pl = HT.pvalue(HT.OneSampleTTest(d, -Δ); tail=:right); pu = HT.pvalue(HT.OneSampleTTest(d, Δ); tail=:left)
  ci = HT.confint(HT.OneSampleTTest(d); level=1 - 2 * ALPHA)
  (n=n, mean=md, lo=ci[1], hi=ci[2], p=max(pl, pu), equiv=(max(pl, pu) < ALPHA))
end

# per-group mean log1p-difference and TOST for one params set → Dict(label => (n, mean, lo, hi, equiv per margin))
function group_stats(pr, margins)
  out = Dict{String,Any}()
  for (s, gp) in sort(unique([(r.sp, r.pgrp) for r in eachrow(pr)]))
    d = DF.subset(pr, [:sp, :pgrp] => DF.ByRow((a, b) -> a == s && b == gp)); DF.nrow(d) < 3 && continue
    dd = log1p.(Float64.(d.sim_agb)) .- log1p.(Float64.(d.obs_agb))
    out[label(s, gp)] = (s=s, gp=gp, n=length(dd), mean=Statistics.mean(dd), tests=[tost(dd, log1p(m)) for m in margins])
  end
  out
end
