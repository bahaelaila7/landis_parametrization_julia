# Coordinate-wise per-group calibration of the TOST bias on TRAIN (docs/gafl_calibration_log.md D4).
# For one EcoSpecies parameter at a time: evaluate a shared grid of values (applied to all groups), interpolate each
# TOST group's mean log-bias m_g(v) to its zero crossing (nearest bracket to the current value; no sign change ⇒ grid
# value with min |m|), assemble the per-group values, evaluate the assembled set, repeat over the schedule.
# Groups = tools/tost_common.jl labels (species, or species·cell for cell-split species) ⇒ representable in the
# region-level pooling of configs/region4/gafl_A.yml. Bounds/floors enforced as in u_to_params: S∈[0.01,1];
# ratio=B_MAX/ANPP∈[20,35]; B_MAX∈[floor,35000]; ANPP≥data floor.
#   ./julia_gdal.sh --project=. --threads=N tools/calib_coord.jl <config.yml> <start.jld2> <out_dir> [schedule=S,ratio,S,ratio]
import YAML
cfg = YAML.load_file(ARGS[1]); start = ARGS[2]; outd = ARGS[3]; mkpath(outd)
SCHED = length(ARGS) >= 4 ? String.(split(ARGS[4], ",")) : ["S", "ratio", "S", "ratio"]
SPLIT = get(ENV, "CALIB_SPLIT", "train"); MARGINS = [0.05, 0.1, 0.2]   # e.g. CALIB_SPLIT=train+val (D6)
include(joinpath(@__DIR__, "tost_common.jl"))
# search boxes from the config (same keys as training: bmax_upper, anpp_ratio_bounds)
BU = Float64(g("bmax_upper", 35000)); RB = Float64.(g("anpp_ratio_bounds", [20.0, 35.0]))
GRID = Dict("pestab" => [0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8],
            "pmort" => [0.0, 0.003, 0.007, 0.012, 0.02, 0.03, 0.045, 0.07, 0.1, 0.15],   # Sim B whole-cohort death prob/yr (D13, D15)
            "foln" => [0.7, 1.0, 1.4, 1.8, 2.3, 3.0],                    # PnET foliar N % (EcoSpecies)
            "gmult" => [0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.3],              # HBAD diameter-growth multiplier (EcoSpecies)
            "mmult" => [0.3, 0.5, 0.75, 1.0, 1.5, 2.2, 3.5],             # HBAD mortality-hazard multiplier (EcoSpecies)
            "S" => [0.12, 0.18, 0.25, 0.33, 0.42, 0.55, 0.75],
            "ratio" => collect(range(RB[1], RB[2]; length=6)),
            "bmax" => collect(range(14000.0, BU; length=6)))
# Size-targeted calibration (D5): S governs young/small-cohort growth, ANPP ratio and B_MAX the large cohorts. Each
# parameter's root-finding uses only the matched pairs of its size regime (obs AGB < / ≥ SIZE_THR g/m²); groups with
# < 3 pairs in the regime fall back to all pairs. SIZE_THR=0 disables (all pairs, the D4 behaviour).
SIZE_THR = parse(Float64, get(ENV, "CALIB_SIZE_THR", "1000"))
REGIME = Dict("S" => :small, "ratio" => :large, "bmax" => :large, "pestab" => :small, "pmort" => :large,
              "foln" => :large, "gmult" => :large, "mmult" => :large)
# Generic per-(eco, species) nested levers: name => (params field, lower, upper)
NESTED = Dict("pestab" => (:PROB_ESTAB_SPP, 0.0, 1.0), "pmort" => (:PROB_MORT_SPP, 0.0, 0.2),
              "foln" => (:FolN, 0.6, 3.0), "gmult" => (:G_MULT, 0.3, 3.0), "mmult" => (:M_MULT, 0.2, 5.0))
# Relative refinement (D7): a schedule entry "S~" / "ratio~" / "bmax~" multiplies EACH group's current value by a shared
# factor (one eval still covers every group) and root-finds the factor per group — fine, local steps around the
# group's own value instead of a coarse absolute grid shared by all groups.
REL_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.25]
function regime_stats(pr, reg)
  SIZE_THR <= 0 && return group_stats(pr, MARGINS)
  sub = reg === :small ? pr[pr.obs_agb .< SIZE_THR, :] : pr[pr.obs_agb .>= SIZE_THR, :]
  gsub = group_stats(sub, MARGINS); gall = group_stats(pr, MARGINS)
  Dict(l => (haskey(gsub, l) && gsub[l].n >= 3 ? gsub[l] : r) for (l, r) in gall)
end
# data floors (same tables as training), keyed by (CATEGORY, l3, lu)
bmax_tbl = g("bmax_floor_from_data", false) ? P._load_bmax_floor_table(P._expandenv(String(g("bmax_floor_csv", "")))) : nothing
anpp_tbl = g("anpp_floor_from_data", false) ? P._load_anpp_floor_table(P._expandenv(String(g("anpp_floor_csv", "")))) : nothing
l3lu(e) = (q = split(String(eco_list[e]), "|lu="); (String(q[1]), length(q) == 2 ? String(q[2]) : ""))
fl(tbl, s, e, d) = tbl === nothing ? d : get(tbl, (uppercase(species_list[s]), l3lu(e)...), d)
# targets of each group: all (e, j) with that species and group cell
targets = Dict{String,Vector{Tuple{Int,Int}}}()
for (e, ids) in enumerate(eco_species_ids), (j, s) in enumerate(ids)
  push!(get!(targets, label(Int(s), grp(Int(s), e)), Tuple{Int,Int}[]), (e, j))
end
function cur_value(p, P_, ts)
  e, j = first(ts)
  haskey(NESTED, P_) && return Float64(getproperty(p, NESTED[P_][1])[e][j])
  P_ == "S" ? Float64(p.S[e][j]) : P_ == "bmax" ? Float64(p.B_MAX_SPP[e][j]) :
    Float64(p.B_MAX_SPP[e][j] / p.ANPP_MAX_SPP[e][j])
end
function setv!(p, P_, ts, v)
  for (e, j) in ts
    s = Int(eco_species_ids[e][j])
    if haskey(NESTED, P_)   # pestab (Sim B establishment), pmort (annual whole-cohort death), foln, gmult, mmult
      f, lo, hi = NESTED[P_]; getproperty(p, f)[e][j] = Float32(clamp(v, lo, hi))
    elseif P_ == "S"
      p.S[e][j] = Float32(clamp(v, 0.01, 1.0))
    elseif P_ == "ratio"
      r = clamp(v, RB[1], RB[2]); p.ANPP_MAX_SPP[e][j] = Float32(max(p.B_MAX_SPP[e][j] / r, fl(anpp_tbl, s, e, 0.0)))
    else  # bmax: keep the ratio
      r = p.B_MAX_SPP[e][j] / p.ANPP_MAX_SPP[e][j]
      p.B_MAX_SPP[e][j] = Float32(clamp(v, max(12000.0, fl(bmax_tbl, s, e, 12000.0)), BU))
      p.ANPP_MAX_SPP[e][j] = Float32(max(p.B_MAX_SPP[e][j] / r, fl(anpp_tbl, s, e, 0.0)))
    end
  end
  p
end
function enforce!(p)   # make a start point feasible (floors) before calibrating
  hasproperty(p, :B_MAX_SPP) || return p   # non-biomass plugins (pnet/hbad): their levers carry their own bounds
  for (lbl, ts) in targets, (e, j) in ts
    s = Int(eco_species_ids[e][j]); r = clamp(p.B_MAX_SPP[e][j] / p.ANPP_MAX_SPP[e][j], RB[1], RB[2])
    p.B_MAX_SPP[e][j] = Float32(clamp(p.B_MAX_SPP[e][j], max(12000.0, fl(bmax_tbl, s, e, 12000.0)), BU))
    p.ANPP_MAX_SPP[e][j] = Float32(max(p.B_MAX_SPP[e][j] / r, fl(anpp_tbl, s, e, 0.0)))
    p.S[e][j] = Float32(clamp(p.S[e][j], 0.01, 1.0))
  end
  p
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
