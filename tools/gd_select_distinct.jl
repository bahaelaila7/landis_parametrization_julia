# Pick K meaningfully distinct, behavioural parameter sets from a gdigel run (anchor-and-explore, docs/cmaes_gdigel.md §4).
#   ./julia_gdal.sh --project=. tools/gd_select_distinct.jl <config.yml> <gd_out_dir> <anchor_params.jld2> <K> <out_dir> [min_pass10_drop=3]
# Candidates: recorded solutions + final lineage points + the behavioural archive. A candidate is behavioural when its
# in-loop ±10 % pass count (train, approximate TOST) is ≥ the anchor's minus `min_pass10_drop`.
# Selection: farthest-point sampling in the search's u-space (L2, every dimension in [0,1]) seeded with the anchor, so each
# pick is the behavioural candidate farthest from everything already chosen. Writes <out_dir>/alt_<k>.jld2 + selection.csv
# (distance to anchor and to the nearest other pick, pass counts, per-parameter-family median |log ratio| vs the anchor).
using Pan, JLD2, DataFrames, CSV, Statistics, YAML
const P = Pan; const PU = P.PU
cfg = YAML.load_file(ARGS[1]); gdd = ARGS[2]; anchor_path = ARGS[3]; K = parse(Int, ARGS[4]); outd = ARGS[5]; mkpath(outd)
drop = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 3
# the u-space layout depends on the run's parameter pooling, which is configured by the data preparation — so decode
# inside the same preparation path (POST_SEARCH_HOOK after a 1-trial run), exactly like the search did
function select_hook(d)
  global anchor = P._load_params_from_path(anchor_path)
  SP = P.succession_plugin_for(anchor)
  global pd = P.make_param_dists(SP, length(anchor.SPECIES_LIST), length(anchor.ECO_LIST), [Int.(ids) for ids in anchor.ECO_SPECIES_IDS]; no_establishment=true, fit_establishment=false)
  global slots = PU.build_slots(pd, anchor)
  ck = sort(filter(f -> occursin(r"^gd_state@\d+\.jld2$", f), readdir(gdd)); by=f -> parse(Int, match(r"@(\d+)", f)[1]))
  S = JLD2.load_object(joinpath(gdd, last(ck))); st = S.st
  ua = PU.params_to_u(anchor, pd, slots)
  gpass(m, se, M) = count(g -> !isnan(m[g]) && abs(m[g]) + 1.645 * se[g] < log1p(M), eachindex(m))
  cands = NamedTuple[]
  for a in S.archive; push!(cands, (u=a.u, n05=a.n05, n10=a.n10, n20=a.n20, src="archive g$(a.gen) L$(a.lineage)")); end
  anchor_n10 = maximum((a.n10 for a in S.archive if maximum(abs.(a.u .- ua)) < 0.05); init=0)
  anchor_n10 == 0 && (anchor_n10 = maximum((c.n10 for c in cands); init=0))
  thr = anchor_n10 - drop
  beh = [c for c in cands if c.n10 >= thr]
  println("candidates $(length(cands)), anchor ±10 % ≈ $(anchor_n10) (in-loop), behavioural (≥ $thr): $(length(beh))")
  isempty(beh) && error("no behavioural candidates")
  chosen = Vector{Float64}[ua]; picks = NamedTuple[]
  dist(a, b) = sqrt(sum(abs2, a .- b))
  for k in 1:K
    best = argmax([minimum(dist(c.u, x) for x in chosen) for c in beh])
    c = beh[best]; push!(chosen, c.u)
    push!(picks, merge(c, (dist_anchor=dist(c.u, ua), dist_nearest=minimum(dist(c.u, x) for x in chosen[1:end-1]))))
  end
  # per parameter family: median |log(value / anchor value)| over slots
  nva = PU.params_to_nativevec(anchor, pd, slots)
  fam = [String(pd.params[pi].name) for (pi, _) in slots]
  rows = DataFrame(alt=Int[], file=String[], source=String[], n05=Int[], n10=Int[], n20=Int[], dist_anchor=Float64[], dist_nearest=Float64[], family_change=String[])
  for (k, c) in enumerate(picks)
    p = PU.u_to_params(clamp.(c.u, 0, 1), pd, slots, anchor)
    JLD2.save_object(joinpath(outd, "alt_$(k).jld2"), p)
    nv = PU.params_to_nativevec(p, pd, slots)
    fc = join(["$(f) $(round(Statistics.median([abs(log(max(nv[i], 1e-9) / max(nva[i], 1e-9))) for i in eachindex(fam) if fam[i] == f]) * 100, digits=1))%" for f in unique(fam)], "; ")
    push!(rows, (k, "alt_$(k).jld2", c.src, c.n05, c.n10, c.n20, round(c.dist_anchor, digits=3), round(c.dist_nearest, digits=3), fc))
  end
  CSV.write(joinpath(outd, "selection.csv"), rows)
  println(rows)

end
P.POST_SEARCH_HOOK[] = select_hook
P.run_from_yaml(ARGS[1]; overrides=Dict{String,Any}("output_dir" => joinpath(outd, "_prep"), "trials" => 1, "igel_mu" => 2, "igel_sobol_init" => false,
  "search_mode" => "igelmo", "n_reps" => 1, "start_from" => anchor_path))
