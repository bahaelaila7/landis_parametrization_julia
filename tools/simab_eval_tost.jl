# Held-out evaluation for tools/simab_eval.jl: group TOST (Sim A: per-pair TOST; Sim B: plot-cluster bootstrap on log Σsim/Σobs) on val and test
# with the UNCHANGED tools/tost_common.jl, plus (Sim B) the total-AGB bias per FIA cycle.
#   ./julia_gdal.sh --project=. tools/simab_eval_tost.jl <config.yml> <params.jld2> <out.jld2>
using Pan
cfg = Pan.load_config(ARGS[1]); params_path = ARGS[2]; out = ARGS[3]
SPLIT = "val+test"; MARGINS = [0.05, 0.1, 0.2]
include(joinpath(@__DIR__, "tost_common.jl"))
p = JLD2.load_object(params_path)
res = Dict{String,Any}()
for (k, nm) in enumerate(SPLITS)
  pr = _paired_one(p, CTXS[k], 0)
  gs = group_stats(pr, MARGINS)
  mab = Statistics.median([abs(expm1(r.mean)) * 100 for r in values(gs)])
  r = Dict{String,Any}("ng" => length(gs), "pass05" => count(x -> x.tests[1].equiv, values(gs)), "pass10" => count(x -> x.tests[2].equiv, values(gs)),
                       "pass20" => count(x -> x.tests[3].equiv, values(gs)), "median_abs_bias" => mab,
                       "groups" => [(label=l, n=x.n, pct_bias=expm1(x.mean) * 100, eq05=x.tests[1].equiv, eq10=x.tests[2].equiv, eq20=x.tests[3].equiv) for (l, x) in gs])
  if SIMB && hasproperty(pr, :cyc)   # total AGB bias per cycle (Σ over all pairs of the split)
    cb = Dict{Int,Any}()
    for dd in DF.groupby(pr, :cyc); cb[Int(dd.cyc[1])] = (n=DF.nrow(dd), pct=expm1(log(sum(dd.sim_agb) + 1) - log(sum(dd.obs_agb) + 1)) * 100); end
    r["cycle_bias"] = cb
  end
  res[nm] = r
  println("$nm: groups $(r["ng"]) ±5/10/20% = $(r["pass05"])/$(r["pass10"])/$(r["pass20"]), median|bias| $(round(mab, digits=1))%" * (haskey(r, "cycle_bias") ? "  cycle bias % " * string(Dict(c => round(v.pct, digits=1) for (c, v) in r["cycle_bias"])) : ""))
end
JLD2.save_object(out, res)
