# ---- cohort-level exclusion at a measurement year (new order; replaces the species-level exclusion_dict) ----------------------
# A cohort (eco_species, age) at (site, T) is EXCLUDED from the loss on BOTH sides when it is
#   * partially disturbed (drop > 0) and the mode is :exclude_overwrite / :exclude_noscale, or
#   * a RECRUIT at entry (ScoringOptions.exclude_recruits_at_entry): observed at T but with no record of the same cohort (species, age-(T-T_prev))
#     at the previous visit of that plot — Sim A has no establishment, so it cannot produce it; it enters via the sync afterwards.
# Sim side: the cohort is skipped (PU._excl_perm!). Reference side: an alternative SPDFGroundTruth for that site-year is built from the
# observed cohorts WITHOUT the excluded ones (same smoothing/binning code: PU.smoothen_ref_years). A species left with no cohort on
# either side contributes nothing (no "missing species" penalty).
struct _ExclPlan
  mode::Symbol
  recruits::Bool
  coh::Dict{Int,Dict{Int,Vector{Tuple{UIntType,FloatType}}}}   # year → site_idx → excluded (eco_species, age)
  ref::Dict{Int,Dict{Int,PU.SPDFGroundTruth}}                  # year → site_idx → reference without those cohorts
  prev::Dict{Int,Dict{Int,Int}}                                # year → site_idx → previous visit year (absent for a site's first visit)
end

function _build_excl_plan(injection_dict, ref_soa, eco_species_ids, loss_params, opts::ScoringOptions)::_ExclPlan
  mode = opts.disturbance; rec_on = opts.exclude_recruits_at_entry
  dist_on = mode in (:exclude_overwrite, :exclude_noscale)
  site_years = Dict{Int,Vector{Int}}()
  site_set = Dict{Tuple{Int,Int},Set{Tuple{UIntType,Int}}}()
  for (yr, sites) in injection_dict, (si, cohorts) in sites
    push!(get!(site_years, si, Int[]), yr)
    site_set[(si, yr)] = Set{Tuple{UIntType,Int}}((sp, Int(age)) for (sp, age, _, _) in cohorts)
  end
  for v in values(site_years); sort!(v); end
  coh = Dict{Int,Dict{Int,Vector{Tuple{UIntType,FloatType}}}}()
  rows = DataFrame(plot_id=UIntType[], eco_id=UIntType[], measdate=Dates.Date[], start_measdate=Dates.Date[], eco_species_id=UIntType[], age_calc=Int[], agb_sum=FloatType[])
  d0 = Dates.Date(2000, 1, 1)
  for yr in sort!(collect(keys(injection_dict))), (si, cohorts) in injection_dict[yr]
    ys = site_years[si]; k = searchsortedfirst(ys, yr)
    py = k > 1 ? ys[k-1] : -1   # previous visit year (year 0 is a valid previous visit)
    ex = Tuple{UIntType,FloatType}[]
    for (sp, age, bio, drop) in cohorts
      isdist = dist_on && drop > zero(FloatType)
      isrec = rec_on && py >= 0 && !((sp, Int(age) - (yr - py)) in site_set[(si, py)])
      (isdist || isrec) && push!(ex, (sp, age))
    end
    isempty(ex) && continue
    get!(coh, yr, Dict{Int,Vector{Tuple{UIntType,FloatType}}}())[si] = ex
    eco = UIntType(getsite(ref_soa, si).eco_id)
    for (sp, age, bio, _) in cohorts
      (sp, age) in ex && continue
      push!(rows, (UIntType(si), eco, d0 + Dates.Day(round(Int, yr * 365.25)), d0, sp, Int(age), bio))
    end
  end
  ref = Dict{Int,Dict{Int,PU.SPDFGroundTruth}}()
  alt = Dict{Tuple{Int,Int},PU.SPDFGroundTruth}()
  if nrow(rows) > 0
    for ((pid, eco), byyr) in reference_tables(rows, eco_species_ids, loss_params)[2], (yr, gt) in byyr
      alt[(Int(pid), yr)] = gt
    end
  end
  for (yr, d) in coh, si in keys(d)
    nsp = length(eco_species_ids[Int(getsite(ref_soa, si).eco_id)])
    gt = get(alt, (si, yr), nothing)
    gt === nothing && (gt = PU.SPDFGroundTruth(keys=falses(nsp), records=Dict{UIntType,PU.SPDFRecord}()))
    get!(ref, yr, Dict{Int,PU.SPDFGroundTruth}())[si] = gt
  end
  prev = Dict{Int,Dict{Int,Int}}()
  for (si, ys) in site_years, k in 2:length(ys); get!(prev, ys[k], Dict{Int,Int}())[si] = ys[k-1]; end
  _ExclPlan(mode, rec_on, coh, ref, prev)
end
