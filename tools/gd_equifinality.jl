# Equifinality report for a gdigel run: how differently can each species group be parameterized and still pass?
#   ./julia_gdal.sh --project=. --threads=8 tools/gd_equifinality.jl <config.yml> <gd_out_dir> [split=train] [margin=0.1]
# 1. Scores every solution (gd_out_dir/solutions/*.jld2) with the TOST tooling on `split` (tools/tost_common.jl: same pairs,
#    groups and test as every other table) → per-solution pass counts and per-(solution, group) pass flags.
# 2. For each TOST group, over the solutions in which that group passes at `margin`: the native value range of every
#    parameter slot that affects only that group (e.g. S, B_MAX, ANPP for Biomass Succession; FolN for PnET; G_MULT, M_MULT
#    for HBAD), and the number of DISTINCT parameterizations (greedy clusters, L∞ distance > 0.1 in the search's u-space).
# Writes <gd_out_dir>/equifinality_<split>.csv (per group × param) and equifinality_<split>_solutions.csv.
using Pan
cfg = Pan.load_config(ARGS[1]); gdd = ARGS[2]
SPLIT = length(ARGS) >= 3 ? ARGS[3] : "train"; MARGIN = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.1
MARGINS = [0.05, 0.1, 0.2]; mi = findfirst(==(MARGIN), MARGINS)
include(joinpath(@__DIR__, "tost_common.jl"))
files = sort(filter(f -> endswith(f, ".jld2"), readdir(joinpath(gdd, "solutions"); join=true)))
SPT = P.succession_plugin_for(JLD2.load_object(files[1]))
spc = P.param_space_from_cfg(cfg, PLUGIN.ctx, splots, eco_list, species_list, eco_species_ids; no_establishment=true, fit_establishment=false)   # the run's parameter space (options, pooling, blocks) over the Sim A subset
pd = spc.dists
sols = [JLD2.load_object(f) for f in files]
slots = spc.slots
# tost per solution
srows = DF.DataFrame(file=String[], n05=Int[], n10=Int[], n20=Int[], median_abs_bias_pct=Float64[])
passmap = Dict{Tuple{Int,String},Bool}()
for (k, p) in enumerate(sols)
  gs = group_stats(paired(p), MARGINS)
  push!(srows, (basename(files[k]), count(r -> r.tests[1].equiv, values(gs)), count(r -> r.tests[2].equiv, values(gs)), count(r -> r.tests[3].equiv, values(gs)),
    round(Statistics.median([abs(expm1(r.mean)) * 100 for r in values(gs)]), digits=2)))
  for (l, r) in gs; passmap[(k, l)] = r.tests[mi].equiv; end
  println("  [$k/$(length(sols))] $(basename(files[k])): ±5% $(srows.n05[end]) ±10% $(srows.n10[end]) ±20% $(srows.n20[end])")
end
CSV.write(joinpath(gdd, "equifinality_$(SPLIT)_solutions.csv"), srows)
# slot → TOST group label (slots touching exactly one TOST group)
cells_of(pi, t) = P._gd_slot_cells(pd.params[pi].sampler, t, eco_species_ids)
lab(e, j) = (s = Int(eco_species_ids[e][j]); label(s, grp(s, e)))
U = [PU.params_to_u(p, spc, slots) for p in sols]; NV = [PU.params_to_nativevec(p, spc, slots) for p in sols]
# HBAD SDIMAX is sampled in a wide box and clamped per species at decode (sdimax_range): report the EFFECTIVE value
if SPT === P.HBADPlugin.HBAD
  for (i, (pi, t)) in enumerate(slots)
    String(pd.params[pi].name) == "SDIMAX" || continue
    lo, hi = P.HBADPlugin.sdimax_range(PLUGIN.ctx.priors.sp[Int(t)])
    for v in NV; v[i] = clamp(v[i], lo, hi); end
  end
end
out = DF.DataFrame(group=String[], param=String[], n_pass=Int[], n_distinct=Int[], min=Float64[], median=Float64[], max=Float64[], max_over_min=Float64[])
glabels = sort(unique(l for (_, l) in keys(passmap)))
for gl in glabels
  ks = [k for k in eachindex(sols) if get(passmap, (k, gl), false)]
  sidx = [i for (i, (pi, t)) in enumerate(slots) if unique(lab(c...) for c in cells_of(pi, t)) == [gl]]
  isempty(sidx) && continue
  # distinct parameterizations among passing solutions (greedy L∞ clustering in u on this group's slots)
  reps = Int[]
  for k in ks; any(r -> maximum(abs.(U[k][sidx] .- U[r][sidx])) <= 0.1, reps) || push!(reps, k); end
  for i in sidx
    nm = String(pd.params[slots[i][1]].name)
    v = [NV[k][i] for k in ks]
    isempty(v) ? push!(out, (gl, nm, 0, 0, NaN, NaN, NaN, NaN)) :
      push!(out, (gl, nm, length(ks), length(reps), minimum(v), Statistics.median(v), maximum(v), minimum(v) > 0 ? maximum(v) / minimum(v) : NaN))
  end
end
CSV.write(joinpath(gdd, "equifinality_$(SPLIT).csv"), out)
println("EQUIFINALITY_DONE: $(length(sols)) solutions; groups with ≥2 distinct passing parameterizations: ",
  length(unique(out.group[out.n_distinct .>= 2])), " / ", length(unique(out.group)))
