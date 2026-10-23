# Rank Pareto-archive candidates (or given params files) by TOST equivalence pass count on one split (default val).
# Same pairing/test as test/tost_sim_obs.jl; data/split prepared ONCE (tools/tost_common.jl → Pan.prepare_from_cfg).
#   ./julia_gdal.sh --project=. --threads=N tools/tost_select.jl <config.yml> <run_dir | a.jld2,b.jld2,...> [K=16] [split=val] [margins=0.1,0.2]
# Writes tost_select_<split>.csv (per candidate) + tost_select_<split>_groups.csv next to the candidates and saves the top
# candidate (most passes at the first margin; ties → more passes at the 2nd, then smaller median |bias|) to selected_<split>/.
# TOST_DUMP=<dir> also writes the matched pairs of every candidate.
# TOST_RANK=<m1,m2,...[,acc]> (e.g. 0.2,0.1,0.05,acc; nested CV, tools/ncv): rank by passes at these margins in this order, then (acc) by the lower
#   mean over groups of the group's mean |log1p(sim) - log1p(obs)| (column mean_abs_lr); default: the rule above.
using Pan
cfg = Pan.load_config(ARGS[1]); run_dir = ARGS[2]
K = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 16
SPLIT = length(ARGS) >= 4 ? ARGS[4] : "val"
MARGINS = length(ARGS) >= 5 ? parse.(Float64, split(ARGS[5], ",")) : [0.05, 0.1, 0.2]
include(joinpath(@__DIR__, "tost_common.jl"))
struct _FileCand; x; fx; end
if occursin(".jld2", run_dir)
  files = String.(split(run_dir, ","))
  arch = [_FileCand(JLD2.load_object(f), (objectives=Float32[NaN, NaN],)) for f in files]
  pick = collect(eachindex(arch)); run_dir = dirname(files[1])
  println("scoring $(length(files)) params files on $SPLIT")
else
  ck = sort(filter(f -> occursin(r"search_state@\d+\.jld2", f), readdir(run_dir)); by=f -> parse(Int, match(r"@(\d+)", f)[1]))
  st = JLD2.load_object(joinpath(run_dir, last(ck))); arch = collect(st.archive)
  ord = sortperm([Float64(sum(c.fx.objectives[2:2:end])) for c in arch])          # by AGB objective
  pick = unique(vcat(ord[round.(Int, range(1, length(ord); length=min(K, length(ord))))], argmin([sum(Float64.(c.fx.objectives)) for c in arch])))
  println("$(last(ck)): archive $(length(arch)); scoring $(length(pick)) candidates on $SPLIT")
end
col(m) = Symbol("pass_$(round(Int, 100m))")
PRIMARY = findfirst(≈(0.1), MARGINS) === nothing ? 1 : findfirst(≈(0.1), MARGINS)   # rank by ±10 % first if present
crow = DF.DataFrame(); grows = DF.DataFrame()
for (k, i) in enumerate(pick)
  c = arch[i]; pr = paired(c.x)
  haskey(ENV, "TOST_DUMP") && CSV.write(joinpath(ENV["TOST_DUMP"], "pairs_cand$(i).csv"),
    DF.transform(pr, :sp => DF.ByRow(s -> species_list[s]) => :species, :eco => DF.ByRow(e -> String(eco_list[e])) => :eco_name))
  gs = group_stats(pr, MARGINS)
  for (lbl, r) in gs
    t = r.tests[PRIMARY]
    row = Dict{Symbol,Any}(:cand => i, :group => lbl, :n => r.n, :pct_bias => round(expm1(r.mean) * 100, digits=1), :mean_abs_lr => round(r.mabs, digits=5),
      :ci_lo => round(expm1(t.lo) * 100, digits=1), :ci_hi => round(expm1(t.hi) * 100, digits=1))
    for (mi, m) in enumerate(MARGINS); row[col(m)] = r.tests[mi].equiv; end
    push!(grows, NamedTuple(row); cols=:union)
  end
  np = [count(r -> r.tests[m].equiv, values(gs)) for m in eachindex(MARGINS)]
  obj = Float64.(c.fx.objectives); mab = Statistics.median([abs(expm1(r.mean)) * 100 for r in values(gs)])
  row = Dict{Symbol,Any}(:cand => i, :W => round(sum(obj[1:2:end]), digits=5), :AGB => round(sum(obj[2:2:end]), digits=4),
                         :groups => length(gs), :median_abs_bias => round(mab, digits=1),
                         :mean_abs_lr => round(Statistics.mean([r.mabs for r in values(gs)]), digits=5))
  for (mi, m) in enumerate(MARGINS); row[col(m)] = np[mi]; end
  push!(crow, NamedTuple(row); cols=:union)
  println("  [$k/$(length(pick))] cand $i  " * join(["pass±$(round(Int,100m))%=$(np[mi])/$(length(gs))" for (mi, m) in enumerate(MARGINS)], "  ") *
          "  median|bias|=$(round(mab, digits=1))%")
end
order_ = if haskey(ENV, "TOST_RANK")
  rk = strip.(split(ENV["TOST_RANK"], ","))
  vcat([(m = parse(Float64, x); any(≈(m), MARGINS) || error("TOST_RANK margin $x is not among the margins $(MARGINS)"); DF.order(col(m), rev=true)) for x in rk if x != "acc"],
       any(==("acc"), rk) ? [:mean_abs_lr] : Symbol[], [:median_abs_bias])
else
  vcat([DF.order(col(MARGINS[PRIMARY]), rev=true)], [DF.order(col(m), rev=true) for (mi, m) in enumerate(MARGINS) if mi != PRIMARY], [:median_abs_bias])
end
C = sort(crow, order_)
CSV.write(joinpath(run_dir, "tost_select_$(SPLIT).csv"), C); CSV.write(joinpath(run_dir, "tost_select_$(SPLIT)_groups.csv"), grows)
best = C.cand[1]; mkpath(joinpath(run_dir, "selected_$(SPLIT)"))
JLD2.save_object(joinpath(run_dir, "selected_$(SPLIT)", "params.jld2"), arch[best].x)
println("SELECTED cand $best: " * join(["±$(round(Int,100m))% $(C[1, col(m)])/$(C.groups[1])" for m in MARGINS], ", ") * ", median|bias| $(C.median_abs_bias[1])%, mean|log ratio| $(C.mean_abs_lr[1])")
open(io -> println(io, best <= length(pick) && occursin(".jld2", ARGS[2]) ? String(split(ARGS[2], ",")[best]) : "cand $best"), joinpath(run_dir, "selected_$(SPLIT)", "source.txt"), "w")
