# Rank Pareto-archive candidates by TOST equivalence pass count on one split (default val) — model selection for the
# "pass as many species as possible" goal. Same pairing/test as test/tost_sim_obs.jl (surviving cohorts matched by
# plot, year, species, age; pre-injection; d = log1p(sim) − log1p(obs); equivalent iff (1−2α) CI ⊂ ±log1p(margin)).
# Data/split prepared ONCE via Pan.prepare_from_cfg (identical to training).
#   ./julia_gdal.sh --project=. --threads=N tools/tost_select.jl <config.yml> <run_dir> [K=16] [split=val] [margins=0.1,0.2]
# Writes <run_dir>/tost_select_<split>.csv (one row per candidate) and tost_select_<split>_groups.csv, and saves the
# top candidate (most passes at the first margin; ties → smaller mean |bias|) to <run_dir>/selected_<split>/params.jld2.
using Pan
import JLD2, YAML, Statistics, DataFrames, HypothesisTests
const P = Pan; const PU = P.PU; const D = P.Data; const DF = DataFrames; const HT = HypothesisTests
cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d); run_dir = ARGS[2]
K = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 16
SPLIT = length(ARGS) >= 4 ? ARGS[4] : "val"
MARGINS = length(ARGS) >= 5 ? parse.(Float64, split(ARGS[5], ",")) : [0.1, 0.2]
ALPHA = 0.05
P.OVERRIDE_INJECTION[] = Bool(g("override_injection", true)); P.OVERRIDE_INJECTION_SYNC[] = Bool(g("override_injection_sync", true))
P.OVERRIDE_INJECTION_REPLACE[] = Bool(g("override_injection_replace", false))
P.OVERRIDE_INJECTION_DISTURBANCE[] = Symbol(g("override_injection_disturbance", "off"))
D.USE_FIA_CYCLE[] = Bool(g("fia_cycle", false)); P.INIT_PERTURB_FRAC[] = 0.0
no_estab = Bool(g("no_establishment", false)); rng0 = Int(g("seed", 1))
splots, eco_list, species_list, eco_species_ids, splots_val, splots_test = P.prepare_from_cfg(cfg)
sp = SPLIT == "train" ? splots : SPLIT == "val" ? splots_val : splots_test
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
# split-invariant simulation inputs, built once
max_age = Int(maximum(sp.age_calc)); spdf = PU.smoothen_ref_years(sp, loss_params, max_age; debug=false)
spdf_plts = D.make_spdf_dict(spdf, eco_species_ids); ssy = D.get_site_sim_years(spdf); spc = D.get_spinup_cohorts(sp)
inj = D.get_injection_cohorts(sp; all_cohorts=P.OVERRIDE_INJECTION[])
msy = maximum(maximum.(filter(!isempty, ssy.sim_years)))
plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(DF.select(sp, [:plot_id, :eco_id]))))
especo2sp = Dict((Int(r.eco_id), Int(r.eco_species_id)) => Int(r.species_id) for r in eachrow(unique(DF.select(sp, [:eco_id, :eco_species_id, :species_id]))))
injc = DF.DataFrame(plot_id=Int.(inj.plot_id), sim_year=Int.(inj.sim_year), esp=Int.(inj.eco_species_id), age=Int.(inj.age_calc),
                    obs_agb=Float64.(inj.agb_sum), drop=Float64.(inj.disturbance_drop_pct))
injc = DF.combine(DF.groupby(DF.subset(injc, :sim_year => DF.ByRow(>(0))), [:plot_id, :sim_year, :esp, :age]), :obs_agb => sum => :obs_agb, :drop => maximum => :drop)
function paired(params)
  rng = P.RNGType(UInt64(rng0))
  rs = P.make_sites(sp, eco_species_ids; rng=rng, spinup=false, no_establishment=no_estab)   # BiomassSuccession (region4 branch)
  idict = P._build_injection_dict(inj, rs); iyears = Set(Int.(inj.sim_year))
  res = P.fit_params(rs, params, msy, n_species, eco_species_ids, spdf_plts, ssy, false, spc, loss_params;
    debug=false, search_tier=3, injection_dict=idict, injection_years=iyears, seeds=[rand(rng, UInt64)], cache_preinject=true)
  simc = DF.DataFrame(plot_id=Int[], sim_year=Int[], esp=Int[], age=Int[], sim_agb=Float64[])
  for (pid, sy, esp, age, bio) in res[1][2]; push!(simc, (Int(pid), Int(sy), Int(esp), Int(age), Float64(bio))); end
  simc = DF.combine(DF.groupby(DF.subset(simc, :sim_year => DF.ByRow(>(0))), [:plot_id, :sim_year, :esp, :age]), :sim_agb => sum => :sim_agb)
  m = DF.innerjoin(simc, injc, on=[:plot_id, :sim_year, :esp, :age])
  P.OVERRIDE_INJECTION_DISTURBANCE[] == :exclude_overwrite && (m = DF.filter(r -> r.drop <= 0.0, m))
  pr = DF.combine(DF.groupby(m, [:plot_id, :sim_year, :esp]), :sim_agb => sum => :sim_agb, :obs_agb => sum => :obs_agb)
  pr.eco = [plot2eco[p] for p in pr.plot_id]; pr.sp = [especo2sp[(e, esp)] for (e, esp) in zip(pr.eco, pr.esp)]
  pr.pgrp = [grp(s, e) for (s, e) in zip(pr.sp, pr.eco)]
  pr
end
function tost(d, Δ)
  n = length(d); md = Statistics.mean(d)
  n < 3 && return (n=n, mean=md, lo=NaN, hi=NaN, p=NaN, equiv=false)
  Statistics.std(d) < 1e-9 && return (n=n, mean=md, lo=md, hi=md, p=(abs(md) < Δ ? 0.0 : 1.0), equiv=(abs(md) < Δ))
  pl = HT.pvalue(HT.OneSampleTTest(d, -Δ); tail=:right); pu = HT.pvalue(HT.OneSampleTTest(d, Δ); tail=:left)
  ci = HT.confint(HT.OneSampleTTest(d); level=1 - 2 * ALPHA)
  (n=n, mean=md, lo=ci[1], hi=ci[2], p=max(pl, pu), equiv=(max(pl, pu) < ALPHA))
end
# candidates: latest checkpoint archive; K spread along the front (sorted by AGB objective) incl. both extremes + best aggregate
ck = sort(filter(f -> occursin(r"search_state@\d+\.jld2", f), readdir(run_dir)); by=f -> parse(Int, match(r"@(\d+)", f)[1]))
st = JLD2.load_object(joinpath(run_dir, last(ck))); arch = collect(st.archive)
ord = sortperm([Float64(sum(c.fx.objectives[2:2:end])) for c in arch])          # by AGB objective
pick = unique(vcat(ord[round.(Int, range(1, length(ord); length=min(K, length(ord))))], argmin([sum(Float64.(c.fx.objectives)) for c in arch])))
println("$(last(ck)): archive $(length(arch)); scoring $(length(pick)) candidates on $SPLIT ($(DF.nrow(unique(DF.select(sp, :plot_id)))) plots)")
crow = NamedTuple[]; grows = NamedTuple[]
for (k, i) in enumerate(pick)
  c = arch[i]; pr = paired(c.x)
  keys_ = sort(unique([(r.sp, r.pgrp) for r in eachrow(pr)]))
  npass = zeros(Int, length(MARGINS)); ngrp = 0; absb = Float64[]
  for (s, gp) in keys_
    d = DF.subset(pr, [:sp, :pgrp] => DF.ByRow((a, b) -> a == s && b == gp)); DF.nrow(d) < 3 && continue
    dd = log1p.(Float64.(d.sim_agb)) .- log1p.(Float64.(d.obs_agb)); ngrp += 1
    ts = [tost(dd, log1p(mg)) for mg in MARGINS]; npass .+= [t.equiv for t in ts]
    push!(absb, abs(expm1(ts[1].mean)) * 100)
    push!(grows, (cand=i, group=label(s, gp), n=ts[1].n, pct_bias=round(expm1(ts[1].mean) * 100, digits=1),
                  ci_lo=round(expm1(ts[1].lo) * 100, digits=1), ci_hi=round(expm1(ts[1].hi) * 100, digits=1),
                  pass1=ts[1].equiv, pass2=length(ts) > 1 ? ts[2].equiv : false))
  end
  obj = Float64.(c.fx.objectives)
  push!(crow, (cand=i, W=round(sum(obj[1:2:end]), digits=5), AGB=round(sum(obj[2:2:end]), digits=4), groups=ngrp,
               pass1=npass[1], pass2=length(npass) > 1 ? npass[2] : 0, median_abs_bias=round(Statistics.median(absb), digits=1)))
  println("  [$k/$(length(pick))] cand $i  AGB=$(crow[end].AGB)  pass±$(round(Int,100MARGINS[1]))%=$(npass[1])/$ngrp  " *
          (length(npass) > 1 ? "pass±$(round(Int,100MARGINS[2]))%=$(npass[2])/$ngrp  " : "") * "median|bias|=$(crow[end].median_abs_bias)%")
end
C = sort(DF.DataFrame(crow), [DF.order(:pass1, rev=true), DF.order(:pass2, rev=true), :median_abs_bias])
G = DF.DataFrame(grows)
import CSV
CSV.write(joinpath(run_dir, "tost_select_$(SPLIT).csv"), C); CSV.write(joinpath(run_dir, "tost_select_$(SPLIT)_groups.csv"), G)
best = C.cand[1]; mkpath(joinpath(run_dir, "selected_$(SPLIT)"))
JLD2.save_object(joinpath(run_dir, "selected_$(SPLIT)", "params.jld2"), arch[best].x)
println("SELECTED cand $best: pass±$(round(Int,100MARGINS[1]))% $(C.pass1[1])/$(C.groups[1]), median|bias| $(C.median_abs_bias[1])% → selected_$(SPLIT)/params.jld2")
show(stdout, MIME("text/plain"), G[G.cand .== best, :]; allrows=true); println()
