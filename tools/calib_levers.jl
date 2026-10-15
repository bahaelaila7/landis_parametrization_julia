# Parameter levers shared by tools/calib_coord.jl (root-finding and minimize modes). Expects globals: cfg (Pan.load_config), P, eco_list,
# species_list, eco_species_ids, and (for enforce!/the root mode) `targets` : label => [(e, j) cells].
# search boxes from the config (same keys as training: bmax_upper, anpp_ratio_bounds)
BU = Float64(cfg["bmax_upper"]); RB = cfg["anpp_ratio_bounds"]
GRID = Dict("pestab" => [0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8],
            "pmort" => [0.0, 0.003, 0.007, 0.012, 0.02, 0.03, 0.045, 0.07, 0.1, 0.15],   # Sim B whole-cohort death prob/yr (D13, D15)
            "foln" => [0.7, 1.0, 1.4, 1.8, 2.3, 3.0],                    # PnET foliar N % (EcoSpecies)
            "gmult" => [0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.3],              # HBAD diameter-growth multiplier (EcoSpecies)
            "mmult" => [0.3, 0.5, 0.75, 1.0, 1.5, 2.2, 3.5],             # HBAD mortality-hazard multiplier (EcoSpecies)
            "S" => [0.12, 0.18, 0.25, 0.33, 0.42, 0.55, 0.75],
            "ratio" => collect(range(RB[1], RB[2]; length=6)),
            "bmax" => collect(range(14000.0, BU; length=6)))
# Generic per-(eco, species) nested levers: name => (params field, lower, upper)
NESTED = Dict("pestab" => (:PROB_ESTAB_SPP, 0.0, 1.0), "pmort" => (:PROB_MORT_SPP, 0.0, 0.2),
              "foln" => (:FolN, 0.6, 3.0), "gmult" => (:G_MULT, 0.3, 3.0), "mmult" => (:M_MULT, 0.2, 5.0))
# Relative refinement (D7): a schedule entry "S~" / "ratio~" / "bmax~" multiplies EACH group's current value by a shared
# factor (one eval still covers every group) and root-finds the factor per group — fine, local steps around the
# group's own value instead of a coarse absolute grid shared by all groups.
REL_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.25]
# data floors (same tables as training), keyed by (CATEGORY, l3, lu)
bmax_tbl = cfg["bmax_floor_from_data"] ? P._load_bmax_floor_table(cfg["bmax_floor_csv"]) : nothing
anpp_tbl = cfg["anpp_floor_from_data"] ? P._load_anpp_floor_table(cfg["anpp_floor_csv"]) : nothing
l3lu(e) = (q = split(String(eco_list[e]), "|lu="); (String(q[1]), length(q) == 2 ? String(q[2]) : ""))
fl(tbl, s, e, d) = tbl === nothing ? d : get(tbl, (uppercase(species_list[s]), l3lu(e)...), d)
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
