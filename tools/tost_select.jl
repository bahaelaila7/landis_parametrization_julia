# Rank Pareto-archive candidates (or given params files) by TOST equivalence pass count on one split (default val).
# Same pairing/test as test/tost_sim_obs.jl; data/split prepared ONCE (tools/tost_common.jl → Pan.prepare_from_cfg).
#   ./julia_gdal.sh --project=. --threads=N tools/tost_select.jl <config.yml> <run_dir | a.jld2,b.jld2,...> [K=16] [split=val] [margins=0.1,0.2]
# Writes tost_select_<split>.csv (per candidate) + tost_select_<split>_groups.csv next to the candidates and saves the top
# candidate (most passes at the first margin; ties → more passes at the 2nd, then smaller median |bias|) to selected_<split>/.
# TOST_DUMP=<dir> also writes the matched pairs of every candidate.
import YAML
cfg = YAML.load_file(ARGS[1]); run_dir = ARGS[2]
K = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 16
SPLIT = length(ARGS) >= 4 ? ARGS[4] : "val"
MARGINS = length(ARGS) >= 5 ? parse.(Float64, split(ARGS[5], ",")) : [0.1, 0.2]
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
crow = NamedTuple[]; grows = NamedTuple[]
for (k, i) in enumerate(pick)
  c = arch[i]; pr = paired(c.x)
  haskey(ENV, "TOST_DUMP") && CSV.write(joinpath(ENV["TOST_DUMP"], "pairs_cand$(i).csv"),
    DF.transform(pr, :sp => DF.ByRow(s -> species_list[s]) => :species, :eco => DF.ByRow(e -> String(eco_list[e])) => :eco_name))
  gs = group_stats(pr, MARGINS)
  for (lbl, r) in gs
    t1 = r.tests[1]
    push!(grows, (cand=i, group=lbl, n=r.n, pct_bias=round(expm1(r.mean) * 100, digits=1), ci_lo=round(expm1(t1.lo) * 100, digits=1),
                  ci_hi=round(expm1(t1.hi) * 100, digits=1), pass1=t1.equiv, pass2=length(r.tests) > 1 ? r.tests[2].equiv : false))
  end
  np = [count(r -> r.tests[m].equiv, values(gs)) for m in eachindex(MARGINS)]
  obj = Float64.(c.fx.objectives); mab = Statistics.median([abs(expm1(r.mean)) * 100 for r in values(gs)])
  push!(crow, (cand=i, W=round(sum(obj[1:2:end]), digits=5), AGB=round(sum(obj[2:2:end]), digits=4), groups=length(gs),
               pass1=np[1], pass2=length(np) > 1 ? np[2] : 0, median_abs_bias=round(mab, digits=1)))
  println("  [$k/$(length(pick))] cand $i  pass±$(round(Int,100MARGINS[1]))%=$(np[1])/$(length(gs))  " *
          (length(np) > 1 ? "pass±$(round(Int,100MARGINS[2]))%=$(np[2])/$(length(gs))  " : "") * "median|bias|=$(round(mab, digits=1))%")
end
C = sort(DF.DataFrame(crow), [DF.order(:pass1, rev=true), DF.order(:pass2, rev=true), :median_abs_bias])
G = DF.DataFrame(grows)
CSV.write(joinpath(run_dir, "tost_select_$(SPLIT).csv"), C); CSV.write(joinpath(run_dir, "tost_select_$(SPLIT)_groups.csv"), G)
best = C.cand[1]; mkpath(joinpath(run_dir, "selected_$(SPLIT)"))
JLD2.save_object(joinpath(run_dir, "selected_$(SPLIT)", "params.jld2"), arch[best].x)
println("SELECTED cand $best: pass±$(round(Int,100MARGINS[1]))% $(C.pass1[1])/$(C.groups[1]), median|bias| $(C.median_abs_bias[1])%")
