# Coordinate-wise per-group calibration of the TOST bias on TRAIN (docs/gafl_calibration_log.md D4).
# For one EcoSpecies parameter at a time: evaluate a shared grid of values (applied to all groups), interpolate each
# TOST group's mean log-bias m_g(v) to its zero crossing (nearest bracket to the current value; no sign change ⇒ grid
# value with min |m|), assemble the per-group values, evaluate the assembled set, repeat over the schedule.
# Groups = tools/tost_common.jl labels (species, or species·cell for cell-split species) ⇒ representable in the
# region-level pooling of configs/region4/gafl_A.yml. Bounds/floors enforced as in u_to_params: S∈[0.01,1];
# ratio=B_MAX/ANPP∈[20,35]; B_MAX∈[floor,35000]; ANPP≥data floor.
#   ./julia_gdal.sh --project=. --threads=N tools/calib_coord.jl <config.yml> <start.jld2> <out_dir> [schedule=S,ratio,S,ratio] [mode=root|minimize]
using Pan
cfg = Pan.load_config(ARGS[1]); start = ARGS[2]; outd = ARGS[3]; mkpath(outd)
SCHED = length(ARGS) >= 4 ? String.(split(ARGS[4], ",")) : ["S", "ratio", "S", "ratio"]
SPLIT = get(ENV, "CALIB_SPLIT", "train"); MARGINS = [0.05, 0.1, 0.2]   # e.g. CALIB_SPLIT=train+val (D6)
# CALIB_MODE (env, or 5th argument): "root" (default, the TOST-bias root-finding below) | "minimize" (docs/mo_gdigel.md: per-group
# Chebyshev minimisation of accuracy a_g and age-structure w_g; CALIB_LAMBDA, CALIB_AREF/WREF) — see tools/calib_minimize.jl.
MODE = lowercase(get(ENV, "CALIB_MODE", length(ARGS) >= 5 ? ARGS[5] : "root"))
MODE in ("root", "minimize") || error("CALIB_MODE must be root | minimize")
MODE == "minimize" && (include(joinpath(@__DIR__, "calib_minimize.jl")); exit(0))
include(joinpath(@__DIR__, "tost_common.jl"))
include(joinpath(@__DIR__, "calib_levers.jl"))   # levers: GRID/NESTED/REL_FACTORS, data floors, cur_value/setv!/enforce! (shared with the minimize mode)
# Size-targeted calibration (D5): S governs young/small-cohort growth, ANPP ratio and B_MAX the large cohorts. Each
# parameter's root-finding uses only the matched pairs of its size regime (obs AGB < / ≥ SIZE_THR g/m²); groups with
# < 3 pairs in the regime fall back to all pairs. SIZE_THR=0 disables (all pairs, the D4 behaviour).
SIZE_THR = parse(Float64, get(ENV, "CALIB_SIZE_THR", "1000"))
REGIME = Dict("S" => :small, "ratio" => :large, "bmax" => :large, "pestab" => :small, "pmort" => :large,
              "foln" => :large, "gmult" => :large, "mmult" => :large)
function regime_stats(pr, reg)
  SIZE_THR <= 0 && return group_stats(pr, MARGINS)
  sub = reg === :small ? pr[pr.obs_agb .< SIZE_THR, :] : pr[pr.obs_agb .>= SIZE_THR, :]
  gsub = group_stats(sub, MARGINS); gall = group_stats(pr, MARGINS)
  Dict(l => (haskey(gsub, l) && gsub[l].n >= 3 ? gsub[l] : r) for (l, r) in gall)
end
# targets of each group: all (e, j) with that species and group cell
targets = Dict{String,Vector{Tuple{Int,Int}}}()
for (e, ids) in enumerate(eco_species_ids), (j, s) in enumerate(ids)
  push!(get!(targets, label(Int(s), grp(Int(s), e)), Tuple{Int,Int}[]), (e, j))
end
summ(gs) = (n05=count(r -> r.tests[1].equiv, values(gs)), n10=count(r -> r.tests[2].equiv, values(gs)), n20=count(r -> r.tests[3].equiv, values(gs)), ng=length(gs),
            mab=Statistics.median([abs(expm1(r.mean)) * 100 for r in values(gs)]))
log_rows = NamedTuple[]
cur = enforce!(deepcopy(JLD2.load_object(start)))
gs = group_stats(paired(cur), MARGINS); sm = summ(gs)
println("start: $SPLIT pass±5% $(sm.n05)/$(sm.ng), ±10% $(sm.n10)/$(sm.ng), ±20% $(sm.n20)/$(sm.ng), median|bias| $(round(sm.mab, digits=1))%")
push!(log_rows, (round=0, param="start", n05=sm.n05, n10=sm.n10, n20=sm.n20, ng=sm.ng, median_abs_bias=round(sm.mab, digits=1)))
for (rnd, P0) in enumerate(SCHED)
  rel = endswith(P0, "~"); P_ = rel ? P0[1:end-1] : P0
  grid = rel ? REL_FACTORS : GRID[P_]; M = Dict{String,Vector{Float64}}()
  base = Dict(lbl => cur_value(cur, P_, ts) for (lbl, ts) in targets)
  for v in grid
    p = deepcopy(cur); for (lbl, ts) in targets; setv!(p, P_, ts, rel ? base[lbl] * v : v); end
    gv = SIZE_THR > 0 ? regime_stats(paired(p), REGIME[P_]) : group_stats(paired(p), MARGINS)
    for (lbl, r) in gv; get!(M, lbl, fill(NaN, length(grid)))[findfirst(==(v), grid)] = r.mean; end
    println("  round $rnd ($P_) grid $v done")
  end
  new = deepcopy(cur); nset = 0
  for (lbl, ts) in targets
    haskey(M, lbl) || continue
    m = M[lbl]; c0 = rel ? 1.0 : cur_value(cur, P_, ts); best = nothing
    for k in 1:length(grid)-1   # brackets with a sign change; pick the one nearest the current value
      (isnan(m[k]) || isnan(m[k+1]) || sign(m[k]) == sign(m[k+1])) && continue
      x = grid[k] + (0 - m[k]) * (grid[k+1] - grid[k]) / (m[k+1] - m[k])
      (best === nothing || abs(x - c0) < abs(best - c0)) && (best = x)
    end
    if best === nothing
      ok = findall(!isnan, m); isempty(ok) && continue
      best = grid[ok[argmin(abs.(m[ok]))]]
    end
    setv!(new, P_, ts, rel ? base[lbl] * best : best); nset += 1
  end
  gn = group_stats(paired(new), MARGINS); sn = summ(gn)
  println("round $rnd ($P0): set $nset groups → $SPLIT pass±5% $(sn.n05)/$(sn.ng), ±10% $(sn.n10)/$(sn.ng), ±20% $(sn.n20)/$(sn.ng), median|bias| $(round(sn.mab, digits=1))%")
  push!(log_rows, (round=rnd, param=P0, n05=sn.n05, n10=sn.n10, n20=sn.n20, ng=sn.ng, median_abs_bias=round(sn.mab, digits=1)))
  global cur = new
  tag = rel ? "$(P_)rel" : P_
  JLD2.save_object(joinpath(outd, "round$(rnd)_$(tag).jld2"), cur)
  CSV.write(joinpath(outd, "round$(rnd)_$(tag)_groups.csv"),
    DF.DataFrame([(group=l, n=r.n, pct_bias=round(expm1(r.mean) * 100, digits=1), pass05=r.tests[1].equiv, pass10=r.tests[2].equiv, pass20=r.tests[3].equiv) for (l, r) in gn]))
end
CSV.write(joinpath(outd, "calib_log.csv"), DF.DataFrame(log_rows))
JLD2.save_object(joinpath(outd, "calibrated.jld2"), cur)
println("DONE → $(outd)/calibrated.jld2")
