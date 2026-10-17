# Each cohort tuple = (eco_species_id, age, observed_biomass, disturbance_drop_pct).
const _SiteInjectionYear = Vector{Tuple{Int,Vector{Tuple{UIntType,FloatType,FloatType,FloatType}}}}
const _SiteInjectionDict = Dict{Int,_SiteInjectionYear}

# Injection modes (ScoringOptions, built once per run by `scoring_from_config`; src/options.jl):
#   override=true: get_injection_cohorts returns the full observed state at every measurement year, and _inject_observed_cohorts!
#     forces each observed cohort onto the site — overwriting the biomass of the matching (species, age) cohort in place, or adding
#     it if absent — so sim matches obs at every loss point. For the sensitivity test, noise > 0 is a multiplicative jitter sd on
#     injected biomass.
#   replace=true switches override to FULL REPLACE: each measured site is wiped (live/old/B reset) and rebuilt from exactly the
#     observed cohorts, dropping sim-only cohorts (ones that should have died but didn't). Gives an exact multi-cohort state for
#     stress-testing the CSR indexing. Requires override (for the data).
#   sync=true: SET-SYNC mode (choice B) — the injection manages the cohort SET to track the data (remove sim cohorts whose
#     (species,age) isn't observed = "died"; add observed cohorts the sim lacks = "recruited", with observed biomass) but LEAVES
#     matched survivors' sim biomass untouched, so the simulator fits their biomass deterministically. Requires override; mutually
#     exclusive with replace (replace takes precedence).
#   loss_before_injection=true (the default for sync): at a measurement year the loss (every search tier) is computed on the SIMULATED
#     state — i.e. BEFORE the observed cohorts are injected for the next interval. The previous order (inject, then score) compared a
#     state whose cohort SET had already been forced to the observed one (sync: unobserved sim cohorts removed, missing observed
#     cohorts added with observed biomass), so the age-structure (Wasserstein) term largely scored observed against observed.
#     `loss_before_injection: false` reproduces the previous behaviour (and earlier results).
#
# Partial-disturbance handling in SYNC mode (ScoringOptions.disturbance). Each observed cohort carries a disturbance_drop_pct
# (from curation). Modes:
#   :off              — ignore drop (current behaviour).
#   :scale            — scale the matched (survivor) sim cohort's biomass by (1-drop) at the
#                       disturbance year (one-time shock); the cohort STAYS in the loss, so growth
#                       params fit the undisturbed trajectory with the disturbance applied.
#   :exclude_overwrite— overwrite the matched cohort with the observed (already-reduced) biomass and
#                       EXCLUDE it from the site loss at that year.
#   :exclude_noscale  — leave the sim biomass and EXCLUDE the cohort from the site loss at that year.

# ---- Observed-cohort injection, one function per mode -----------------------------------------------------------------------------
# Only touches the sites that actually have injections (typically << n_sites).
# _new_cohort_counts is already == site.live for all other sites after process_plugin!.
# The mode is a singleton type chosen once per call by `_inject_mode(override, replace, sync)` (replace beats sync):
#   InjAppend   (override=false): append every listed cohort as a new cohort.
#   InjReplace  (override, replace): wipe each measured site and rebuild it from exactly the observed cohorts.
#   InjSync     (override, sync): the cohort SET tracks the data, matched survivors keep their sim biomass (steps below).
#   InjOverride (override only): overwrite the biomass of the matching (species, age) cohort in place; add it only if no match exists.
abstract type InjectMode end
struct InjAppend   <: InjectMode end
struct InjReplace  <: InjectMode end
struct InjSync     <: InjectMode end
struct InjOverride <: InjectMode end
@inline _inject_mode(override::Bool, replace::Bool, sync::Bool)::Union{InjAppend,InjReplace,InjSync,InjOverride} =
  !override ? InjAppend() : replace ? InjReplace() : sync ? InjSync() : InjOverride()

# Sim cohort (species, age) lookup over the first `n` cohorts of the site.
@inline function _has_cohort(csp, cage, n::Int, sp, age)::Bool
  for j in 1:n
    (csp[j] == sp && cage[j] == age) && return true
  end
  false
end
# Biomass of an injected cohort: the observed value, with the multiplicative noise of the sensitivity test when noise > 0.
@inline _noisy_bio(site, bio::FloatType, noise::FloatType)::FloatType =
  noise > zero(FloatType) ? abs(bio * (one(FloatType) + noise * randn(site.rng, FloatType))) : bio
# Append (species, age) with RAW biomass b (overwrites add_cohort!'s integer truncation, which only the model proper uses; C# parity).
@inline function _add_cohort_raw!(pc, site, sp, age, b::FloatType)
  add_cohort!(pc, site, sp, age, b)
  set_cohort_bio!(pc, site, Int(site.live), b)
  site.B += b
  nothing
end
# Recompute site.old (# cohorts with age > 1) after cohorts were removed / added.
@inline function _recompute_old!(site)
  cage = site.c_age
  o = zero(UIntType)
  for j in 1:Int(site.live)
    cage[j] > one(FloatType) && (o += one(UIntType))
  end
  site.old = o
  nothing
end

function _inject!(::InjAppend, pc, soa, site_cohorts::_SiteInjectionYear, opts::ScoringOptions, skip_scale::Bool)
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    site._new_cohort_counts = Int(site.live) + length(cohorts)
  end
  PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    for (sp, age, bio, _drop) in cohorts
      add_cohort!(pc, site, sp, age, bio)
      site.B += bio
    end
  end
  nothing
end

# Full replace: wipe each measured site and rebuild it from exactly the observed
# cohorts (raw biomass, no trunc; optional noise). Drops sim-only cohorts so the
# site state equals the data even when some sim cohorts should have died but didn't.
function _inject!(::InjReplace, pc, soa, site_cohorts::_SiteInjectionYear, opts::ScoringOptions, skip_scale::Bool)
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    site._new_cohort_counts = length(cohorts)
  end
  PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))
  noise = FloatType(opts.noise)
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    site.live = zero(UIntType)
    site.old = zero(UIntType)
    site.B = zero(FloatType)
    for (sp, age, bio, _drop) in cohorts
      _add_cohort_raw!(pc, site, sp, age, _noisy_bio(site, bio, noise))
    end
  end
  nothing
end

# Set-sync (choice B): cohort SET tracks the data; matched survivors keep their sim biomass. Four steps, in this order.
# Step A: remove sim cohorts whose (species,age) isn't in the observed set ("died"), in place.
function _sync_remove_unobserved!(soa, site_cohorts::_SiteInjectionYear)
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    keep = Set{Tuple{UIntType,FloatType}}((sp, age) for (sp, age, _, _) in cohorts)
    csp = site.c_species
    cage = site.c_age
    cbio = site.c_bio
    i = 1
    while i <= Int(site.live)
      if (csp[i], cage[i]) in keep
        i += 1
      else
        site.B -= cbio[i]
        lastidx = Int(site.live)
        if i != lastidx
          PanCore.swap_cohort!(site, i, lastidx)   # moves every per-cohort CSR field
        end
        site.live -= one(UIntType)
      end
    end
  end
  nothing
end
# Disturbance handling for matched (survivor) cohorts at the disturbance year. Only survivors
# (present after step A) are touched; recruits added in step C keep their observed biomass.
#   :scale            — biomass *= (1-drop): a known shock; cohort STAYS in the loss (growth fits it).
#   :exclude_overwrite— biomass := observed survivor biomass; cohort EXCLUDED from loss (loss side).
#   :exclude_noscale  — biomass unchanged; cohort EXCLUDED from loss (loss side).
# `skip_scale`: the (1-drop) shock was already applied before scoring.
function _sync_disturbed_survivors!(pc, soa, site_cohorts::_SiteInjectionYear, dmode::Symbol, skip_scale::Bool)
  ((dmode === :scale && !skip_scale) || dmode === :exclude_overwrite) || return nothing
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    csp = site.c_species; cage = site.c_age; cbio = site.c_bio
    for (sp, age, bio, drop) in cohorts
      drop > zero(FloatType) || continue
      @inbounds for j in 1:Int(site.live)
        if csp[j] == sp && cage[j] == age
          newb = dmode === :scale ? cbio[j] * (one(FloatType) - drop) : bio
          site.B += newb - cbio[j]
          set_cohort_bio!(pc, site, j, newb)
          break
        end
      end
    end
  end
  nothing
end
# Step B: size each site to survivors + (# observed (species,age) absent from the sim).
function _sync_size_sites!(soa, site_cohorts::_SiteInjectionYear)
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    csp = site.c_species
    cage = site.c_age
    live = Int(site.live)
    n_new = 0
    for (sp, age, _, _) in cohorts
      _has_cohort(csp, cage, live, sp, age) || (n_new += 1)
    end
    site._new_cohort_counts = live + n_new
  end
  PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))
  nothing
end
# Step C: append the unmatched observed cohorts (recruits) with observed biomass; recompute old.
function _sync_append_recruits!(pc, soa, site_cohorts::_SiteInjectionYear, noise::FloatType)
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    csp = site.c_species
    cage = site.c_age
    orig_live = Int(site.live)
    for (sp, age, bio, _drop) in cohorts
      _has_cohort(csp, cage, orig_live, sp, age) && continue
      _add_cohort_raw!(pc, site, sp, age, _noisy_bio(site, bio, noise))
    end
    # step A removals + step C adds left site.old stale
    _recompute_old!(site)
  end
  nothing
end
function _inject!(::InjSync, pc, soa, site_cohorts::_SiteInjectionYear, opts::ScoringOptions, skip_scale::Bool)
  _sync_remove_unobserved!(soa, site_cohorts)
  _sync_disturbed_survivors!(pc, soa, site_cohorts, opts.disturbance, skip_scale)
  _sync_size_sites!(soa, site_cohorts)
  _sync_append_recruits!(pc, soa, site_cohorts, FloatType(opts.noise))
  nothing
end

# Override in place (no sync): overwrite matches in place (searching only the original live range), append the unmatched.
# Biomass is written RAW (no trunc) so the injected state equals the reference exactly — only this diagnostic path skips the
# integer-valued truncation that add_cohort!/succession use everywhere else (C# parity).
function _inject!(::InjOverride, pc, soa, site_cohorts::_SiteInjectionYear, opts::ScoringOptions, skip_scale::Bool)
  # Pass 1: size each site to live + (# observed cohorts with no (species,age) match).
  _sync_size_sites!(soa, site_cohorts)
  # Pass 2
  noise = FloatType(opts.noise)
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    csp = site.c_species
    cage = site.c_age
    cbio = site.c_bio
    orig_live = Int(site.live)
    for (sp, age, bio, _drop) in cohorts
      b = _noisy_bio(site, bio, noise)
      matched = false
      for j in 1:orig_live
        if csp[j] == sp && cage[j] == age
          site.B += b - cbio[j]
          set_cohort_bio!(pc, site, j, b)
          matched = true
          break
        end
      end
      matched || _add_cohort_raw!(pc, site, sp, age, b)
    end
  end
  nothing
end

# Entry point of the injection step of the scoring rule (after the shock and the exclusion plan have been applied / consulted).
function _inject_observed_cohorts!(pc, soa, site_cohorts::_SiteInjectionYear, opts::ScoringOptions; override::Bool=opts.override, replace::Bool=opts.replace, sync::Bool=opts.sync, skip_scale::Bool=false)
  _inject!(_inject_mode(override, replace, sync), pc, soa, site_cohorts, opts, skip_scale)
  nothing
end

# Pre-index by sim_year → [(site_idx, cohorts)] using the ref_soa's plot→site mapping.
# Called once per reference by build_injection_plan; eliminates the per-site dict lookup inside _inject_observed_cohorts!.
function _build_injection_dict(injection_cohorts::DataFrame, ref_soa)::_SiteInjectionDict
  plot_to_site = Dict{Int,Int}(Int(getsite(ref_soa, i).ref_cn) => i for i in 1:ref_soa.n)
  by_year = _SiteInjectionDict()
  site_year_pos = Dict{Tuple{Int,Int},Int}()
  for row in eachrow(injection_cohorts)
    site_idx = get(plot_to_site, Int(row.plot_id), 0)
    site_idx == 0 && continue
    year = Int(row.sim_year)
    year_list = get!(by_year, year, _SiteInjectionYear())
    pos = get(site_year_pos, (site_idx, year), 0)
    if pos == 0
      push!(year_list, (site_idx, Tuple{UIntType,FloatType,FloatType,FloatType}[]))
      pos = length(year_list)
      site_year_pos[(site_idx, year)] = pos
    end
    drop = hasproperty(injection_cohorts, :disturbance_drop_pct) ? FloatType(row.disturbance_drop_pct) : zero(FloatType)
    push!(year_list[pos][2], (UIntType(row.eco_species_id), FloatType(row.age_calc), FloatType(row.agb_sum), drop))
  end
  by_year
end
