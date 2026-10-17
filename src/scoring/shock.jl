# Disturbance SHOCK (new scoring order, loss_before_injection + override sync = `shock_order(opts)`). Applied at every measurement year T
# AFTER the growth step and BEFORE scoring, on the simulated state:
#   (a) sim cohorts matching (eco_species, age) of a cohort FULLY removed by disturbance in (T_prev→T) are removed;
#   (b) :scale — matched partially-disturbed survivors (observed drop > 0) are scaled by (1-drop).
# Undisturbed cohorts (including ones that died naturally in the data: the sim must kill them) are untouched and scored as simulated.
# Data: plot-level full removals from Data._plot_shocks! (tools/export_cohort_disturbance.py parquet).
const _SiteShockYear = Vector{Tuple{Int,Vector{Tuple{UIntType,FloatType}}}}
const _SiteShockDict = Dict{Int,_SiteShockYear}
# diagnostics (tools/shock_stats.jl): [full-removal rows tried, matched+removed sim cohorts, partial (drop>0) cohorts, matched+scaled]
const SHOCK_STATS = [Threads.Atomic{Int}(0) for _ in 1:4]

function _build_shock_dict(shock_rows::DataFrame, ref_soa)::_SiteShockDict
  plot_to_site = Dict{Int,Int}(Int(getsite(ref_soa, i).ref_cn) => i for i in 1:ref_soa.n)
  by_year = _SiteShockDict()
  pos = Dict{Tuple{Int,Int},Int}()
  for row in eachrow(shock_rows)
    site_idx = get(plot_to_site, Int(row.plot_id), 0)
    site_idx == 0 && continue
    year = Int(row.sim_year)
    yl = get!(by_year, year, _SiteShockYear())
    p = get(pos, (site_idx, year), 0)
    if p == 0
      push!(yl, (site_idx, Tuple{UIntType,FloatType}[]))
      p = length(yl); pos[(site_idx, year)] = p
    end
    push!(yl[p][2], (UIntType(row.eco_species_id), FloatType(row.age_calc)))
  end
  by_year
end

# Swap-remove sim cohort i (same mechanics as the sync's Pass A).
@inline function _shock_remove_cohort!(site, i::Int)
  cbio = site.c_bio
  site.B -= cbio[i]
  lastidx = Int(site.live)
  i != lastidx && PanCore.swap_cohort!(site, i, lastidx)
  site.live -= one(UIntType)
  nothing
end

# Shock step (a): remove the sim cohorts matching (eco_species, age) of a cohort FULLY removed by disturbance since the previous visit.
function _shock_full_removals!(soa, shock_year::_SiteShockYear)
  for (site_idx, lst) in shock_year
    site = getsite(soa, site_idx)
    site.active || continue
    csp = site.c_species; cage = site.c_age
    removed = false
    for (sp, age) in lst
      Threads.atomic_add!(SHOCK_STATS[1], 1)
      @inbounds for j in 1:Int(site.live)
        if csp[j] == sp && cage[j] == age
          _shock_remove_cohort!(site, j)
          Threads.atomic_add!(SHOCK_STATS[2], 1)
          removed = true
          break
        end
      end
    end
    if removed
      o = zero(UIntType)
      for j in 1:Int(site.live)
        cage[j] > one(FloatType) && (o += one(UIntType))
      end
      site.old = o
      site._new_cohort_counts = Int(site.live)
    end
  end
  nothing
end
# Shock step (b), mode :scale: scale the matched, partially disturbed survivors (observed drop > 0) by (1-drop).
function _shock_scale_partial!(pc, soa, site_cohorts::_SiteInjectionYear)
  for (site_idx, cohorts) in site_cohorts
    site = getsite(soa, site_idx)
    site.active || continue
    csp = site.c_species; cage = site.c_age; cbio = site.c_bio
    for (sp, age, _bio, drop) in cohorts
      drop > zero(FloatType) || continue
      Threads.atomic_add!(SHOCK_STATS[3], 1)
      @inbounds for j in 1:Int(site.live)
        if csp[j] == sp && cage[j] == age
          Threads.atomic_add!(SHOCK_STATS[4], 1)
          newb = cbio[j] * (one(FloatType) - drop)
          site.B += newb - cbio[j]
          set_cohort_bio!(pc, site, j, newb)
          break
        end
      end
    end
  end
  nothing
end
# The shock step of the scoring rule: (a) then (b), on the simulated state, before scoring.
function _apply_disturbance_shock!(pc, soa, shock_year::Union{Nothing,_SiteShockYear}, site_cohorts::Union{Nothing,_SiteInjectionYear}, opts::ScoringOptions)
  shock_year === nothing || _shock_full_removals!(soa, shock_year)
  (site_cohorts !== nothing && opts.disturbance === :scale) && _shock_scale_partial!(pc, soa, site_cohorts)
  nothing
end
