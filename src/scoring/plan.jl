"""
    InjectionPlan

Everything the observed-cohort injection needs for ONE reference (train Sim A, val Sim A, Sim B's `dual_b`, val `dual_b`), built once by
`build_injection_plan` and passed with that reference (`fit_params(...; injection=plan)`):

  - `dict`, `years`: the injection cohorts by sim year → [(site, cohorts)], and the set of measurement years that inject.
  - `shock`: the disturbance-shock dict (full removals by disturbance, from the injection data's `disturbance_shock` metadata), or `nothing`.
  - `excl`: the cohort-level exclusion plan of the shock order (disturbed cohorts and recruits at entry), or `nothing` outside that order.
  - `legacy_excl`: the species-level exclusion of the legacy order with `disturbance = :exclude_*`, or `nothing`.
  - `opts`: the `ScoringOptions` the plan was built under; `fit_params` refuses a plan built under a different shock order, disturbance
    mode or recruit rule (the exclusion data depend on them).

The plan is immutable and read-only during a search, so the site loop of any thread (and the candidate-parallel batches) can share it.
"""
struct InjectionPlan
  dict::_SiteInjectionDict
  years::Set{Int}
  shock::Union{Nothing,_SiteShockDict}
  excl::Union{Nothing,_ExclPlan}
  legacy_excl::Union{Nothing,Dict{Int,Dict{Int,Set{UIntType}}}}
  opts::ScoringOptions
end

# species-level exclusion (legacy order): year → site_idx → species with a partially disturbed cohort
function _legacy_exclusion(injection_dict)
  exclusion_dict = Dict{Int,Dict{Int,Set{UIntType}}}()
  for (yr, sites) in injection_dict
    sd = Dict{Int,Set{UIntType}}()
    for (site_idx, cohorts) in sites
      s = Set{UIntType}(sp for (sp, _age, _bio, drop) in cohorts if drop > zero(FloatType))
      isempty(s) || (sd[site_idx] = s)
    end
    isempty(sd) || (exclusion_dict[yr] = sd)
  end
  exclusion_dict
end

"""
    build_injection_plan(injection_cohorts, ref_soa, eco_species_ids, loss_params, opts) -> Union{Nothing,InjectionPlan}

The injection plan of the reference `ref_soa` from its observed injection cohorts (`Data.get_injection_cohorts`); `nothing` when there are none.
"""
function build_injection_plan(injection_cohorts, ref_soa, eco_species_ids, loss_params, opts::ScoringOptions)::Union{Nothing,InjectionPlan}
  injection_cohorts === nothing && return nothing
  dict = _build_injection_dict(injection_cohorts, ref_soa)
  years = Set(Int.(injection_cohorts.sim_year))
  shock = "disturbance_shock" in metadatakeys(injection_cohorts) ?
    _build_shock_dict(metadata(injection_cohorts, "disturbance_shock"), ref_soa) : nothing
  do_shock = shock_order(opts)
  excl = do_shock ? _build_excl_plan(dict, ref_soa, eco_species_ids, loss_params, opts) : nothing   # cohort-level exclusion (new order)
  legacy = (!do_shock && opts.disturbance in (:exclude_overwrite, :exclude_noscale)) ? _legacy_exclusion(dict) : nothing
  return InjectionPlan(dict, years, shock, excl, legacy, opts)
end

# A plan carries data derived under a given shock order / disturbance mode / recruit rule: it must be used with the same scoring rule.
function _check_plan_options(plan::InjectionPlan, scoring::ScoringOptions)
  p = plan.opts
  (shock_order(p) == shock_order(scoring) && p.disturbance === scoring.disturbance && p.exclude_recruits_at_entry == scoring.exclude_recruits_at_entry) ||
    error("InjectionPlan was built under a different scoring rule (shock order $(shock_order(p)), disturbance $(p.disturbance), recruits $(p.exclude_recruits_at_entry)) " *
          "than the one it is evaluated with ($(shock_order(scoring)), $(scoring.disturbance), $(scoring.exclude_recruits_at_entry)); rebuild it with build_injection_plan")
  nothing
end
