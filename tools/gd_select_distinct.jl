# Pick K meaningfully distinct, behavioural parameter sets from a gdigel run (anchor-and-explore, docs/cmaes_gdigel.md §4).
#   ./julia_gdal.sh --project=. tools/gd_select_distinct.jl <config.yml> <gd_out_dir> <anchor_params.jld2> <K> <out_dir> [max_groups_above_tau=2]
# Candidates: recorded solutions + final lineage points + the behavioural archive. A candidate is behavioural when at most
# `max_groups_above_tau` species groups exceed the exploration's per-group tolerance tau_g (the anchor has none).
# Selection: farthest-point sampling in the search's u-space (L2, every dimension in [0,1]) seeded with the anchor, so each
# pick is the behavioural candidate farthest from everything already chosen. Writes <out_dir>/alt_<k>.jld2 + selection.csv
# (distance to anchor and to the nearest other pick, pass counts, per-parameter-family median |log ratio| vs the anchor).
using Pan, JLD2, DataFrames, CSV, Statistics, YAML
const P = Pan; const PU = P.PU
cfg = YAML.load_file(ARGS[1]); gdd = ARGS[2]; anchor_path = ARGS[3]; K = parse(Int, ARGS[4]); outd = ARGS[5]; mkpath(outd)
drop = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 2
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
  # candidates: every recorded solution, the final lineage points and the behavioural archive (u + per-group losses f).
  # Behavioural = at most `drop` groups above their explore threshold tau_g (the criterion the exploration itself used;
  # the anchor has 0 by construction). Duplicates (identical u) are removed.
  tau = st.tau; nbad(f) = isempty(tau) ? 0 : count(g -> !isnan(f[g]) && f[g] > tau[g], eachindex(f))
  arch_n10 = Dict(a.u => (a.n05, a.n10, a.n20) for a in S.archive)
  cands = NamedTuple[]; seen = Set{UInt64}()
  addc(u, f, src) = (h = hash(round.(u; digits=6)); h in seen || (push!(seen, h); push!(cands, (u=u, f=f, nbad=nbad(f), src=src,
      n05=get(arch_n10, u, (-1, -1, -1))[1], n10=get(arch_n10, u, (-1, -1, -1))[2], n20=get(arch_n10, u, (-1, -1, -1))[3]))))
  for sl in st.solutions; addc(sl.x, sl.f, "solution L$(sl.lineage) g$(sl.gen)"); end
  for L in st.lineages; addc(L.x, L.f, "final L$(L.id)"); end
  for a in S.archive; addc(a.u, a.f, "archive g$(a.gen) L$(a.lineage)"); end
  beh = [c for c in cands if c.nbad <= drop]
  println("candidates $(length(cands)) (solutions $(length(st.solutions)), lineages $(length(st.lineages)), archive $(length(S.archive))); behavioural (≤ $drop groups above tau): $(length(beh))")
  isempty(beh) && error("no behavioural candidates")
  # selection space: "params" (u-space, default) or "projection" (SELECT_SPACE=projection, ANCHOR_YAML=<anchor .yaml>):
  # each behavioural candidate is projected 100 years (10 × 10 yr, grow-only) on PROJ_PLOTS FIA plots with the scenario
  # engine, and picks maximize the distance between PROJECTED outcomes — log mean AGB at +20/+50/+100 yr, log mean TPA at
  # +50/+100 yr and the BA share of the anchor's top species at +100 yr — so the alternatives differ in what they predict.
  space = get(ENV, "SELECT_SPACE", "params")
  feat = Dict{Int,Vector{Float64}}()   # candidate index (0 = anchor) => feature vector
  if space == "projection"
    meta = P.params_from_yaml(read(ENV["ANCHOR_YAML"], String))[2]
    nplots = parse(Int, get(ENV, "PROJ_PLOTS", "60"))
    db = P.DuckDB.DB(P.SCEN_FIA_DB[]; readonly=true); con = P.DuckDB.DBInterface.connect(db)
    cns = DataFrame(P.DuckDB.DBInterface.execute(con, """
      SELECT p.CN cn FROM PLOT p WHERE p.STATECD IN (12, 13) AND p.INVYR BETWEEN 2018 AND 2020
        AND (p.STATECD || '_' || p.UNITCD || '_' || p.COUNTYCD || '_' || p.PLOT) IN
            (SELECT DISTINCT plt_cn FROM curated_cohorts_landis_stdorg WHERE statecd IN (12,13) AND year(measdate) BETWEEN 2018 AND 2020)
      ORDER BY hash(p.CN) LIMIT $(nplots)""")).cn
    plots = [(cn=String(c), acres=1.0) for c in cns]
    plan = Dict{String,Any}("cycles" => 10, "period" => 10, "actions" => Any[])
    topsp = Ref{Vector{Int16}}(Int16[])
    function project(p)
      ps = P.load_paramset(P.params_to_yaml(p; name="candidate", species_map=meta.species_map, eco_mapping=meta.eco_mapping))
      res, _ = P.run_scenario(ps, plots, plan)
      ok = [v for v in values(res) if v.status == "ok"]
      agb(c) = mean(v.metrics[c].agb for v in ok); tpa(c) = mean(v.metrics[c].tpa for v in ok)
      ba = Dict{Int16,Float64}()
      for v in ok, ((sp, _), (_, b)) in v.dist[v.metrics[11].year]; ba[sp] = get(ba, sp, 0.0) + b; end
      tot = sum(values(ba); init=1e-9)
      isempty(topsp[]) && (topsp[] = first.(sort(collect(ba); by=x -> -x[2]))[1:min(6, length(ba))])
      vcat(log.([agb(3), agb(6), agb(11)]), log.([tpa(6), tpa(11)]), [get(ba, s, 0.0) / tot for s in topsp[]])
    end
    feat[0] = project(PU.u_to_params(ua, pd, slots, anchor))
    for (i, c) in enumerate(beh)
      feat[i] = project(PU.u_to_params(clamp.(c.u, 0, 1), pd, slots, anchor))
      println("  projected candidate $i/$(length(beh)) ($(c.src)): ", round.(feat[i] .- feat[0]; digits=3))
    end
  end
  fv(i) = space == "projection" ? feat[i] : (i == 0 ? ua : beh[i].u)
  chosen = Int[0]; picks = NamedTuple[]
  dist(a, b) = sqrt(sum(abs2, a .- b))
  for k in 1:K
    cand = [i for i in eachindex(beh) if !(i in chosen)]
    isempty(cand) && break
    best = cand[argmax([minimum(dist(fv(i), fv(j)) for j in chosen) for i in cand])]
    c = beh[best]; push!(chosen, best)
    push!(picks, merge(c, (dist_anchor=dist(fv(best), fv(0)), dist_nearest=minimum(dist(fv(best), fv(j)) for j in chosen[1:end-1]))))
  end
  println("selection space: $space")
  # per parameter family: median |log(value / anchor value)| over slots
  nva = PU.params_to_nativevec(anchor, pd, slots)
  fam = [String(pd.params[pi].name) for (pi, _) in slots]
  rows = DataFrame(alt=Int[], file=String[], source=String[], groups_above_tau=Int[], n05=Int[], n10=Int[], n20=Int[], dist_anchor=Float64[], dist_nearest=Float64[], family_change=String[])
  for (k, c) in enumerate(picks)
    p = PU.u_to_params(clamp.(c.u, 0, 1), pd, slots, anchor)
    JLD2.save_object(joinpath(outd, "alt_$(k).jld2"), p)
    nv = PU.params_to_nativevec(p, pd, slots)
    fc = join(["$(f) $(round(Statistics.median([abs(log(max(nv[i], 1e-9) / max(nva[i], 1e-9))) for i in eachindex(fam) if fam[i] == f]) * 100, digits=1))%" for f in unique(fam)], "; ")
    push!(rows, (k, "alt_$(k).jld2", c.src, c.nbad, c.n05, c.n10, c.n20, round(c.dist_anchor, digits=3), round(c.dist_nearest, digits=3), fc))
  end
  CSV.write(joinpath(outd, "selection.csv"), rows)
  println(rows)

end
P.POST_SEARCH_HOOK[] = select_hook
P.run_from_yaml(ARGS[1]; overrides=Dict{String,Any}("output_dir" => joinpath(outd, "_prep"), "trials" => 1, "igel_mu" => 2, "igel_sobol_init" => false,
  "search_mode" => "igelmo", "n_reps" => 1, "start_from" => anchor_path))
