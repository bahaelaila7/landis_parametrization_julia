# ---- Sim A scoring modes for the natural-death part (docs/simAB_options_experiment.md) ------------------------------------
# w_scope     :all       — W on every simulated cohort (sim-only natural deaths included; current behaviour)
#             :survivors — W on cohorts present in both sim and obs at T: sim cohorts absent from the observed set are removed before scoring
#                          (the sync injection removes them right after anyway). Needs the new order + sync.
# death_term  :none | :likelihood — per-cohort survive/die negative log-likelihood given PROB_MORT p over the interval Δt (years between
#                          the two visits): survivor −Δt·log(1−p); natural death −log(1−(1−p)^Δt). PROB_MORT is NOT applied in the Sim A
#                          simulation (zeroed inside fit_params when a DeathAcc is passed); it only enters the death term.
const DEATH_PMIN = 1e-6
const DEATH_PMAX = 0.5
const DEATH_MAXDT = 80
"Per-cell (eco e, local species j) survivor/death statistics: surv_dt = Σ_survivors Δt, n_surv, dhist[k] = # natural deaths with Δt = k."
struct DeathAcc
  surv_dt::Vector{Vector{Float64}}
  n_surv::Vector{Vector{Int}}
  dhist::Vector{Vector{Vector{Int}}}
end
DeathAcc(eco_species_ids) = DeathAcc([zeros(length(ids)) for ids in eco_species_ids], [zeros(Int, length(ids)) for ids in eco_species_ids],
                                      [[zeros(Int, DEATH_MAXDT) for _ in ids] for ids in eco_species_ids])
@inline _pclamp(p) = clamp(Float64(p), DEATH_PMIN, DEATH_PMAX)
"Negative log-likelihood of one cohort-interval outcome."
@inline death_nll_survivor(p, dt) = -dt * log1p(-_pclamp(p))
@inline death_nll_death(p, dt) = -log(-expm1(dt * log1p(-_pclamp(p))))     # −log(1−(1−p)^Δt)
"Per-cell (Σ NLL, # cohorts) given the cell's p."
function death_cell(acc::DeathAcc, e::Int, j::Int, p)
  n = acc.n_surv[e][j]; s = death_nll_survivor(p, 1.0) * acc.surv_dt[e][j]
  for k in 1:DEATH_MAXDT
    c = acc.dhist[e][j][k]; c > 0 && (s += c * death_nll_death(p, Float64(k)); n += c)
  end
  s, n
end

# Collect survive/die statistics at year T (before survivors-only removal). Survivor = sim cohort present in obs at T; natural death = sim
# cohort absent from obs at T (disturbance-removed cohorts were already removed by the shock; excluded cohorts are skipped).
function _death_collect!(acc::DeathAcc, soa, year, injection_dict, plan)
  pv = get(plan.prev, year, nothing); pv === nothing && return
  for (si, cohorts) in injection_dict[year]
    site = getsite(soa, si); site.active || continue
    py = get(pv, si, -1); py < 0 && continue
    dt = min(year - py, DEATH_MAXDT)
    exl = (yd = get(plan.coh, year, nothing); yd === nothing ? nothing : get(yd, si, nothing))
    e = Int(site.eco_id); csp = site.c_species; cage = site.c_age
    for j in 1:Int(site.live)
      sp = csp[j]; age = cage[j]
      exl !== nothing && any(t -> t[1] == sp && t[2] == age, exl) && continue
      inobs = false
      for (osp, oage, _, _) in cohorts; (osp == sp && oage == age) && (inobs = true; break); end
      if inobs; acc.surv_dt[e][Int(sp)] += dt; acc.n_surv[e][Int(sp)] += 1
      else; acc.dhist[e][Int(sp)][dt] += 1; end
    end
  end
  nothing
end

# Survivors-only scoring: remove sim cohorts absent from the observed set at T (see ScoringOptions.w_scope).
function _survivors_only!(soa, year, injection_dict)
  for (si, cohorts) in injection_dict[year]
    site = getsite(soa, si); site.active || continue
    csp = site.c_species; cage = site.c_age; n0 = Int(site.live)
    j = 1
    while j <= Int(site.live)
      inobs = false
      for (sp, age, _, _) in cohorts; (csp[j] == sp && cage[j] == age) && (inobs = true; break); end
      inobs ? (j += 1) : _shock_remove_cohort!(site, j)
    end
    if Int(site.live) != n0
      o = zero(UIntType)
      for q in 1:Int(site.live); cage[q] > one(FloatType) && (o += one(UIntType)); end
      site.old = o
    end
  end
  nothing
end
