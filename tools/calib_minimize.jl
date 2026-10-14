# CALIB_MODE=minimize of tools/calib_coord.jl (docs/mo_gdigel.md §4) — accuracy/structure refinement, NO bias root-finding.
# Included by calib_coord.jl (which provides cfg, start, outd, SCHED). Environment:
#   CALIB_LAMBDA  λ of the Chebyshev scalarization (default 0.5)       CALIB_AREF / CALIB_WREF  reference TOTALS (default: the start set)
# For every scheduled parameter the shared grid (absolute values, or the relative factors for "S~"/"ratio~"/... entries) is
# evaluated as in the root mode, but each GROUP picks the grid value that minimises ITS Chebyshev objective
#     obj_g = max( λ·a_g / a_ref_g , (1−λ)·w_g / w_ref_g ),   a_g = mean |log1p(sim)−log1p(obs)|, w_g = W1 age-structure distance
# (a_ref_g, w_ref_g = the group's values at the start set, floored at 10 % of the median so tiny references cannot dominate) and
# accepts it only if it improves the group's current objective. The assembled set is re-evaluated; if its total
# F = max(λ ΣA/A_ref, (1−λ) ΣW/W_ref) is worse than the previous set's (cross-group interactions) the round is reverted.
# Groups, matched pairs, the W term and the evaluation are exactly those of gdigel (shared code in src/gdigel_driver.jl); the
# training context is prepared by run_from_yaml (so every loss flag of the config is active), CALIB_SPLIT must be train.
# Train TOST pass counts (approximate normal-CI version) are logged for information only and never used.
using Pan, JLD2, CSV, DataFrames, Statistics, Random
import YAML
const P = Pan; const PU = P.PU
g(k, d) = get(cfg, k, d)
LAM = parse(Float64, get(ENV, "CALIB_LAMBDA", "0.5")); (0.0 <= LAM <= 1.0) || error("CALIB_LAMBDA must be in [0,1]")
get(ENV, "CALIB_SPLIT", "train") == "train" || error("CALIB_MODE=minimize works on the train split only")
CTX = Ref{Any}(nothing)
P.POST_SEARCH_HOOK[] = d -> (CTX[] = d; nothing)
P.run_from_yaml(ARGS[1]; overrides=Dict{String,Any}("output_dir" => joinpath(outd, "_ctx"), "trials" => 1, "igel_mu" => 2, "igel_sobol_init" => false,
  "search_mode" => "igelmo", "n_reps" => 1, "start_from" => start))
d = CTX[]; d === nothing && error("training context was not captured")
eco_list = d.eco_list; species_list = d.species_list; eco_species_ids = d.eco_species_ids
include(joinpath(@__DIR__, "calib_levers.jl"))
gdo = get(cfg, "gdigel", Dict()); spinup = Bool(g("spinup", false))
SP = P.succession_plugin_of(d.ref_soa)
pd = P.make_param_dists(SP, d.n_species, length(d.eco_list), d.eco_species_ids; no_establishment=Bool(g("no_establishment", false)), fit_establishment=false)
cur = JLD2.load_object(start); slots = PU.build_slots(pd, cur)
cell_group, labels = P._gd_groups(pd, slots, d.eco_species_ids, d.species_list, d.eco_list; group_param=String(get(gdo, "group_param", "")))
ix = P._gd_index(d.injection_cohorts, d.splots, cell_group, labels; real_only=Bool(get(gdo, "real_only", true)))
groupcells = P._gd_groupcells(ix.cell_group, ix.ngroups)
targets = Dict{String,Vector{Tuple{Int,Int}}}(labels[k] => groupcells[k] for k in 1:ix.ngroups)
nvis = P._gd_cell_visits(d.spdf_plts, d.eco_species_ids, spinup ? 0 : 1)
idict = P._build_injection_dict(d.injection_cohorts, d.ref_soa); iyears = Set(Int.(d.injection_cohorts.sim_year))
msy = d.site_sim_years.sim_years .|> maximum |> maximum
seeds = P._fixed_seeds(P.RNGType(UInt64(Int(g("seed", 1)))), 1)
function ev(p)
  r = P.fit_params(d.ref_soa, p, msy, d.n_species, d.eco_species_ids, d.spdf_plts, d.site_sim_years, spinup, d.spinup_cohorts, d.loss_params;
    debug=false, search_tier=d.tier, seeds=seeds, injection_dict=idict, injection_years=iyears, cache_preinject=true)
  _, _, k = P._agg_reps(r)
  res = P._gd_losses(ix, r[k][2])
  num, den = P._gd_cell_w(r[k][3], d.eco_species_ids, nvis)
  (a=res.a, w=P._gd_group_w(num, den, groupcells), m=res.m, se=res.se, n=res.n)
end
sums(r) = (sum(v for v in r.a if !isnan(v); init=0.0), sum(v for v in r.w if !isnan(v); init=0.0))
cur = enforce!(cur)
r0 = ev(cur); A0, W0 = sums(r0)
AREF = haskey(ENV, "CALIB_AREF") ? parse(Float64, ENV["CALIB_AREF"]) : A0
WREF = haskey(ENV, "CALIB_WREF") ? parse(Float64, ENV["CALIB_WREF"]) : W0
Fof(A, W) = max(LAM * A / AREF, (1 - LAM) * W / WREF)
floor_a = 0.1 * median(filter(!isnan, r0.a)); floor_w = 0.1 * median(filter(!isnan, r0.w))
aref_g = [max(v, floor_a) for v in r0.a]; wref_g = [isnan(v) ? NaN : max(v, floor_w) for v in r0.w]
function gobj(a, w, k)   # group Chebyshev objective; NaN when the group has no accuracy data
  isnan(a[k]) && return NaN
  max(LAM * a[k] / aref_g[k], isnan(w[k]) ? 0.0 : (1 - LAM) * w[k] / wref_g[k])
end
pass(r) = (n05=P._gd_pass(r.m, r.se, 0.05), n10=P._gd_pass(r.m, r.se, 0.1), n20=P._gd_pass(r.m, r.se, 0.2))
log_rows = NamedTuple[]
function record!(rnd, name, r, nset, nacc, reverted)
  A, W = sums(r); p = pass(r)
  println("round $rnd ($name): ΣA=$(round(A, digits=4)) ΣW=$(round(W, digits=4)) F=$(round(Fof(A, W), digits=4)) [λ=$LAM, A_ref=$(round(AREF, sigdigits=5)), W_ref=$(round(WREF, sigdigits=5))]; " *
          "groups improved $nacc/$nset$(reverted ? " (REVERTED: total F worse)" : ""); train approx-TOST (info only) ±5/10/20: $(p.n05)/$(p.n10)/$(p.n20) of $(count(!isnan, r.a))")
  push!(log_rows, (round=rnd, param=name, lambda=LAM, sumA=A, sumW=W, F=Fof(A, W), groups_set=nset, groups_improved=nacc, reverted=reverted,
                   info_n05=p.n05, info_n10=p.n10, info_n20=p.n20, ng=count(!isnan, r.a), A_ref=AREF, W_ref=WREF))
end
record!(0, "start", r0, 0, 0, false)
rcur = r0
for (rnd, P0) in enumerate(SCHED)
  global cur, rcur
  rel = endswith(P0, "~"); P_ = rel ? P0[1:end-1] : P0
  grid = rel ? REL_FACTORS : GRID[P_]
  base = Dict(lbl => cur_value(cur, P_, ts) for (lbl, ts) in targets)
  O = fill(NaN, ix.ngroups, length(grid))
  for (vi, v) in enumerate(grid)
    p = deepcopy(cur); for (lbl, ts) in targets; setv!(p, P_, ts, rel ? base[lbl] * v : v); end
    r = ev(p); for k in 1:ix.ngroups; O[k, vi] = gobj(r.a, r.w, k); end
    println("  round $rnd ($P_) grid $v done")
  end
  new = deepcopy(cur); nset = 0; nacc = 0
  for k in 1:ix.ngroups
    isnan(gobj(rcur.a, rcur.w, k)) && continue
    ok = findall(!isnan, view(O, k, :)); isempty(ok) && continue
    nset += 1; vb = ok[argmin(O[k, ok])]
    O[k, vb] < gobj(rcur.a, rcur.w, k) - 1e-12 || continue     # accept only a strict improvement of the group's own objective
    setv!(new, P_, targets[labels[k]], rel ? base[labels[k]] * grid[vb] : grid[vb]); nacc += 1
  end
  rnew = ev(new); An, Wn = sums(rnew); Ac, Wc = sums(rcur); rev = Fof(An, Wn) > Fof(Ac, Wc)
  if !rev; cur = new; rcur = rnew; end
  record!(rnd, P0, rev ? rcur : rnew, nset, nacc, rev)
  tag = rel ? "$(P_)rel" : P_
  JLD2.save_object(joinpath(outd, "round$(rnd)_$(tag).jld2"), cur)
  CSV.write(joinpath(outd, "round$(rnd)_$(tag)_groups.csv"), DataFrame(group=labels, n=rcur.n, a=rcur.a, w=rcur.w, obj=[gobj(rcur.a, rcur.w, k) for k in 1:ix.ngroups],
    bias_pct=round.(expm1.(rcur.m) .* 100, digits=1)))
end
CSV.write(joinpath(outd, "calib_log.csv"), DataFrame(log_rows))
JLD2.save_object(joinpath(outd, "calibrated.jld2"), cur)
println("DONE (minimize, λ=$LAM) → $(outd)/calibrated.jld2")
