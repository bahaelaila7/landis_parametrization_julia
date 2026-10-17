# ------------------------------------------------------------------------------------------------------------------------------------------------
# Tier accumulators: the scoring half of `fit_params` (Step 4a).
#
# `fit_params` simulates a rep year by year and, at every measurement year of a site, hands the site to ONE accumulator that owns the tier's
# state (per-thread histograms / per-eco loss sums / scratch buffers):
#
#   Tier1Acc  biomass-by-age bins per (eco, species), one set for all measurements          -> `calculate_aggregate_loss`
#   Tier2Acc  per-site bin biomasses per (eco, species, bin) and per-species totals         -> `calculate_t2_loss`
#   Tier3Acc  per-site Wasserstein / AGB loss summed straight into per-thread per-eco sums  (allocation-free kernel)
#   Tier4Acc  per-(eco, cycle) bin histograms of the free sim                               -> `calculate_t4_loss`
#   Tier5Acc  tier 3 (per-site SiteLoss, serial per-year reduction) + tier 4 in ONE sim pass
#
# Interface (all methods are statically dispatched on the accumulator type; `make_tier_acc` is the only place the runtime tier is looked at, once
# per rep, and `_run_rep!` is the function barrier that gets the concrete accumulator type):
#   site_visit!(acc, sc, site, i, year)   the per-site, per-measurement-year work (inside the threaded site loop)
#   begin_year!(acc, soa) / end_year!(acc, soa, year)   per-year hooks (serial; no-ops except tier 5)
#   finish!(acc) -> (run_result::SiteLoss, eco_losses::Vector{SiteLoss})   merge the threads and score
# ------------------------------------------------------------------------------------------------------------------------------------------------

# --- optional per-measurement cohort log (what fit_params returns as its second element; see `collect_cohorts` / `cache_preinject`) -------------
const CohortRow = Tuple{UIntType,Int,UIntType,UIntType,FloatType}   # (ref_cn, sim year, eco species, age, biomass)

"No cohort log: the evaluation does not need the simulated cohorts (the default)."
struct NoCohortLog end
"Every cohort of every site at every measured year, grouped by site (sites are visited by one thread each, so the push is race-free)."
struct CohortLog
  rows::Vector{Vector{CohortRow}}
end
CohortLog(n::Int) = CohortLog([CohortRow[] for _ in 1:n])

@inline _log_site!(::NoCohortLog, i::Int, site, year::Int) = nothing
@inline function _log_site!(l::CohortLog, i::Int, site, year::Int)
  v = l.rows[i]
  for j in 1:site.live
    push!(v, (site.ref_cn, year, site.c_species[j], UIntType(site.c_age[j]), site.c_bio[j]))
  end
  nothing
end
_flat(l::CohortLog) = CohortRow[c for cs in l.rows for c in cs]
_flat(::NoCohortLog) = CohortRow[]
# the writer's `merged_sites_state` frame (empty when the evaluation did not collect cohorts)
_cohorts_df(rows::AbstractVector{<:Tuple}) = isempty(rows) ? DataFrame(plot_id=Int[], sim_year=Int[], species_id=Int[], age=Int[], agb=Float64[]) :
  DataFrame(rows, [:plot_id, :sim_year, :species_id, :age, :agb])
# the cohorts a rep returns: the pre-injection snapshot when one was taken, else the post-scoring log, else nothing
_cohorts(pre::CohortLog, ::Union{NoCohortLog,CohortLog}) = _flat(pre)
_cohorts(::NoCohortLog, post) = _flat(post)

# pure model state BEFORE the injection / override of this year (diagnostics: sim <-> obs without circularity)
_snapshot_pre!(::NoCohortLog, soa, year::Int, sim_years) = nothing
function _snapshot_pre!(l::CohortLog, soa, year::Int, sim_years)
  for i in 1:soa.n
    @inbounds begin
      site = getsite(soa, i)
      site.active || continue
      year in sim_years[site.mapcode] && _log_site!(l, i, site, year)
    end
  end
  nothing
end

# --- the read-only inputs of the per-site scoring --------------------------------------------------------------------------------------------
"""
    YearScoring

What the site visit reads (never written, shared by all threads): the sim years of each site, the reference of each (plot, eco), the tier-4 cycle map,
the species-level (legacy) and cohort-level exclusions, and the loss definition. Parametric so every field access is concretely typed.
"""
struct YearScoring{SY,SP,CM,EX,EP,LP,SC}
  sim_years::SY                       # site_sim_years.sim_years (indexed by site.mapcode)
  spdf_plts::SP                       # (ref_cn, eco_id) -> sim year -> SPDFGroundTruth
  cycle_map::CM                       # (mapcode, sim year) -> tier-4 cycle (tiers 4 / 5) or nothing
  legacy_excl::EX                     # species-level exclusion of the legacy order, or nothing
  excl_plan::EP                       # cohort-level exclusion plan of the shock order, or nothing
  eco_species_ids::Vector{Vector{Int}}
  n_species::Int
  loss_params::LP
  scales::SC
  debug::Bool
end

# the (reference, species exclusion, cohort exclusion) of one site at one year; the cohort exclusion swaps in its trimmed reference
@inline function _site_exclusions(sc::YearScoring, year::Int, i::Int, gt)
  excluded = nothing
  if sc.legacy_excl !== nothing
    yd = get(sc.legacy_excl, year, nothing)
    yd === nothing || (excluded = get(yd, i, nothing))
  end
  excl_c = nothing
  if sc.excl_plan !== nothing
    ycl = get(sc.excl_plan.coh, year, nothing)
    ec = ycl === nothing ? nothing : get(ycl, i, nothing)
    ec === nothing || (excl_c = ec; gt = sc.excl_plan.ref[year][i])
  end
  return gt, excluded, excl_c
end

abstract type TierAcc end
begin_year!(::TierAcc, soa) = nothing
end_year!(::TierAcc, soa, year::Int) = nothing

# --- shared bookkeeping ------------------------------------------------------------------------------------------------------------------------
# (#active sites, #measurements) per ecoregion (tiers 1 / 2)
function _eco_counts(soa, sim_years, starting_sim_year::Int, n_eco::Int)
  site_counts = zeros(Int, n_eco); obs_counts = zeros(Int, n_eco)
  for i in 1:soa.n
    site = getsite(soa, i)
    !site.active && continue
    site_counts[site.eco_id] += 1
    obs_counts[site.eco_id] += count(>=(starting_sim_year), sim_years[site.mapcode])
  end
  site_counts, obs_counts
end
# per (eco, cycle): #measurements (obs) and #distinct plots, mirroring the tier-1 counts (tiers 4 / 5)
function _t4_counts(soa, sim_years, cycle_map, starting_sim_year::Int, n_eco::Int, n_cycles::Int)
  t4_site_counts = [zeros(Int, n_cycles) for _ in 1:n_eco]
  t4_obs_counts = [zeros(Int, n_cycles) for _ in 1:n_eco]
  for i in 1:soa.n
    site = getsite(soa, i)
    !site.active && continue
    seen = Set{Int}()
    for sy in sim_years[site.mapcode]
      sy < starting_sim_year && continue
      cyc = get(cycle_map, (Int(site.mapcode), Int(sy)), 0)
      cyc == 0 && continue
      t4_obs_counts[site.eco_id][cyc] += 1
      (cyc in seen) || (push!(seen, cyc); t4_site_counts[site.eco_id][cyc] += 1)
    end
  end
  t4_site_counts, t4_obs_counts
end
# per-thread scratch of accumulate_site_bins! (sorted cohort permutation, age histogram)
function _bin_scratch(soa, max_sim_year::Int, loss_params)
  nt = Threads.maxthreadid()
  max_cohorts = Int(maximum(soa.refs.cohort[i+1] - soa.refs.cohort[i] for i in 1:soa.n))
  max_age = max_sim_year + max(1, length(loss_params.smoothing_weights) >> 1) + 5
  [Vector{Int}(undef, max_cohorts) for _ in 1:nt], [Vector{FloatType}(undef, max_age) for _ in 1:nt]
end
_n_age_bins(loss_params) = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)

# --- tier 1 ------------------------------------------------------------------------------------------------------------------------------------
struct Tier1Acc{LP} <: TierAcc
  sim_t::Vector{Vector{Matrix{FloatType}}}      # [thread][eco] bins x species
  perm_t::Vector{Vector{Int}}
  ages_t::Vector{Vector{FloatType}}
  ref::Vector{Matrix{FloatType}}
  loss_params::LP
  n_species::Int
  eco_species_ids::Vector{Vector{Int}}
  site_counts::Vector{Int}
  obs_counts::Vector{Int}
end
function Tier1Acc(soa, sim_years, starting_sim_year::Int, max_sim_year::Int, t1_ref, loss_params, n_species::Int, eco_species_ids)
  site_counts, obs_counts = _eco_counts(soa, sim_years, starting_sim_year, length(eco_species_ids))
  n_bins = size(t1_ref[1], 2)
  sim_t = [[zeros(FloatType, length(eco_species_ids[eco_id]), n_bins) for eco_id in eachindex(eco_species_ids)] for _ in 1:Threads.maxthreadid()]
  perm_t, ages_t = _bin_scratch(soa, max_sim_year, loss_params)
  Tier1Acc(sim_t, perm_t, ages_t, t1_ref, loss_params, n_species, eco_species_ids, site_counts, obs_counts)
end
@inline function site_visit!(acc::Tier1Acc, sc::YearScoring, site, i::Int, year::Int)
  tid = Threads.threadid()
  accumulate_site_bins!(acc.sim_t[tid][site.eco_id], site, acc.loss_params, acc.perm_t[tid], acc.ages_t[tid])
end
function finish!(acc::Tier1Acc)
  for tid in 2:Threads.maxthreadid()
    for eco_id in eachindex(acc.eco_species_ids)
      acc.sim_t[1][eco_id] .+= acc.sim_t[tid][eco_id]
    end
  end
  eco_losses = calculate_aggregate_loss(acc.sim_t[1], acc.ref, acc.loss_params, acc.n_species, acc.eco_species_ids, acc.site_counts, acc.obs_counts)
  sum(eco_losses), eco_losses
end

# --- tier 2 ------------------------------------------------------------------------------------------------------------------------------------
struct Tier2Acc{LP,R} <: TierAcc
  bins_t::Vector{Vector{Matrix{Vector{FloatType}}}}   # [thread][eco] species x bin -> the sites' bin biomasses
  total_t::Vector{Vector{Vector{Vector{FloatType}}}}  # [thread][eco][species] -> the sites' total biomasses
  sp_bin_t::Vector{Vector{Matrix{FloatType}}}         # [thread][eco] per-visit scratch (species x bin)
  ref::R
  loss_params::LP
  n_species::Int
  eco_species_ids::Vector{Vector{Int}}
  site_counts::Vector{Int}
  obs_counts::Vector{Int}
  n_bins::Int
end
function Tier2Acc(soa, sim_years, starting_sim_year::Int, t2_ref, loss_params, n_species::Int, eco_species_ids)
  n_bins = _n_age_bins(loss_params)
  site_counts, obs_counts = _eco_counts(soa, sim_years, starting_sim_year, length(eco_species_ids))
  nt = Threads.maxthreadid(); ne = eachindex(eco_species_ids)
  bins_t = [[[FloatType[] for sp in 1:length(eco_species_ids[eco_id]), b in 1:n_bins] for eco_id in ne] for _ in 1:nt]
  total_t = [[[FloatType[] for _ in 1:length(eco_species_ids[eco_id])] for eco_id in ne] for _ in 1:nt]
  sp_bin_t = [[zeros(FloatType, length(eco_species_ids[eco_id]), n_bins) for eco_id in ne] for _ in 1:nt]
  Tier2Acc(bins_t, total_t, sp_bin_t, t2_ref, loss_params, n_species, eco_species_ids, site_counts, obs_counts, n_bins)
end
@inline function site_visit!(acc::Tier2Acc, sc::YearScoring, site, i::Int, year::Int)
  eco_id = Int(site.eco_id)
  tid = Threads.threadid()
  n_sp_eco = length(acc.eco_species_ids[eco_id])
  sp_bin_agbs = acc.sp_bin_t[tid][eco_id]; fill!(sp_bin_agbs, zero(FloatType))
  for j in 1:site.live
    sp_eco = Int(site.c_species[j])
    age = Int(ceil(Float64(site.c_age[j])))
    b = PU.find_age_bin(age, acc.loss_params.age_bins)
    b == 0 && continue
    sp_bin_agbs[sp_eco, b] += site.c_bio[j]
  end
  for sp_eco in 1:n_sp_eco
    sp_total = zero(FloatType)
    for b in 1:acc.n_bins
      sp_total += sp_bin_agbs[sp_eco, b]
      push!(acc.bins_t[tid][eco_id][sp_eco, b], sp_bin_agbs[sp_eco, b])
    end
    push!(acc.total_t[tid][eco_id][sp_eco], sp_total)
  end
end
function finish!(acc::Tier2Acc)
  esids = acc.eco_species_ids; nt = Threads.maxthreadid()
  t2_sim_bins = [[vcat((acc.bins_t[tid][eco_id][sp, b] for tid in 1:nt)...) for sp in 1:length(esids[eco_id]), b in 1:acc.n_bins] for eco_id in eachindex(esids)]
  t2_sim_total = [[vcat((acc.total_t[tid][eco_id][sp] for tid in 1:nt)...) for sp in 1:length(esids[eco_id])] for eco_id in eachindex(esids)]
  eco_losses = calculate_t2_loss(t2_sim_bins, t2_sim_total, acc.ref, acc.n_species, esids, acc.site_counts, acc.obs_counts, acc.loss_params)
  sum(eco_losses), eco_losses
end

# --- tier 3 ------------------------------------------------------------------------------------------------------------------------------------
# Allocation-free: per-thread per-eco accumulators (merged once after the year loop) + per-thread scratch, so `calculate_site_loss2_acc!` allocates
# nothing per site / species (kills GC pressure at high thread counts).
struct Tier3Acc{LP,SC} <: TierAcc
  w_t::Vector{Vector{Vector{FloatType}}}        # [thread][eco][global species]
  agb_t::Vector{Vector{Vector{FloatType}}}
  site_agb_t::Vector{Vector{FloatType}}         # [thread][eco]
  obs_t::Vector{Vector{Int}}
  perm_t::Vector{Vector{Int}}
  ages_t::Vector{Vector{FloatType}}
  insite_t::Vector{Vector{Bool}}
  cdf_t::Vector{Vector{FloatType}}
  agb_bins_t::Vector{Vector{FloatType}}
  loss_params::LP
  scales::SC
  n_species::Int
  eco_species_ids::Vector{Vector{Int}}
end
function Tier3Acc(soa, max_sim_year::Int, loss_params, scales, n_species::Int, eco_species_ids)
  nt = Threads.maxthreadid(); ne = eachindex(eco_species_ids)
  max_cohorts = Int(maximum(soa.refs.cohort[i+1] - soa.refs.cohort[i] for i in 1:soa.n))
  max_age = max_sim_year + max(1, length(loss_params.smoothing_weights) >> 1) + 5
  max_eco_nsp = maximum(length(e) for e in eco_species_ids)
  nbins = length(loss_params.age_bins.bins_idx) + (loss_params.age_bins.last_bin_open ? 1 : 0)
  Tier3Acc([[zeros(FloatType, n_species) for _ in ne] for _ in 1:nt], [[zeros(FloatType, n_species) for _ in ne] for _ in 1:nt],
    [zeros(FloatType, length(eco_species_ids)) for _ in 1:nt], [zeros(Int, length(eco_species_ids)) for _ in 1:nt],
    [Vector{Int}(undef, max_cohorts) for _ in 1:nt], [Vector{FloatType}(undef, max_age) for _ in 1:nt], [Vector{Bool}(undef, max_eco_nsp) for _ in 1:nt],
    [Vector{FloatType}(undef, nbins) for _ in 1:nt], [Vector{FloatType}(undef, nbins) for _ in 1:nt],
    loss_params, scales, n_species, eco_species_ids)
end
@inline function site_visit!(acc::Tier3Acc, sc::YearScoring, site, i::Int, year::Int)
  tid = Threads.threadid()
  eco_id = Int(site.eco_id)
  gt_now, excluded, excl_c = _site_exclusions(sc, year, i, sc.spdf_plts[(site.ref_cn, site.eco_id)][year])
  sag = PU.calculate_site_loss2_acc!(acc.w_t[tid][eco_id], acc.agb_t[tid][eco_id], year, site,
          acc.n_species, acc.eco_species_ids, gt_now, acc.loss_params, acc.scales,
          acc.perm_t[tid], acc.ages_t[tid], acc.insite_t[tid], acc.cdf_t[tid], acc.agb_bins_t[tid]; excluded=excluded, excl_c=excl_c)
  acc.site_agb_t[tid][eco_id] += sag
  acc.obs_t[tid][eco_id] += 1
  nothing
end
# merge the per-thread per-eco accumulators, then build the per-eco breakdown; the run result is the sum of the per-eco SiteLosses
function finish!(acc::Tier3Acc)
  esids = acc.eco_species_ids; ne = eachindex(esids)
  eco3_w = [zeros(FloatType, acc.n_species) for _ in ne]; eco3_agb = [zeros(FloatType, acc.n_species) for _ in ne]
  eco3_site_agb = zeros(FloatType, length(esids)); eco3_obs = zeros(Int, length(esids))
  for tid in 1:Threads.maxthreadid(), e in ne
    eco3_w[e] .+= acc.w_t[tid][e]
    eco3_agb[e] .+= acc.agb_t[tid][e]
    eco3_site_agb[e] += acc.site_agb_t[tid][e]
    eco3_obs[e] += acc.obs_t[tid][e]
  end
  eco_losses = [PU.SiteLoss(sp_w_loss=eco3_w[e], sp_agb_loss=eco3_agb[e], site_agb_loss=eco3_site_agb[e], num_sites=eco3_obs[e], num_obs=eco3_obs[e]) for e in ne]
  sum(eco_losses), eco_losses
end

# --- tier 4 ------------------------------------------------------------------------------------------------------------------------------------
struct Tier4Acc{R,LP,SC,CP,CAP} <: TierAcc
  sim_t::Vector{Vector{Vector{Matrix{FloatType}}}}   # [thread][eco][cycle] bins x species
  perm_t::Vector{Vector{Int}}
  ages_t::Vector{Vector{FloatType}}
  ref::R
  loss_params::LP
  scales::SC
  n_species::Int
  eco_species_ids::Vector{Vector{Int}}
  site_counts::Vector{Vector{Int}}
  obs_counts::Vector{Vector{Int}}
  n_cycles::Int
  cell_plots::CP
  per_plot_mean::Bool
  capture::CAP
end
function Tier4Acc(soa, sim_years, cycle_map, starting_sim_year::Int, max_sim_year::Int, t4_ref, n_cycles::Int, loss_params, scales, n_species::Int, eco_species_ids,
                  cell_plots, per_plot_mean::Bool, capture)
  n_bins = size(t4_ref[1][1], 2)
  site_counts, obs_counts = _t4_counts(soa, sim_years, cycle_map, starting_sim_year, length(eco_species_ids), n_cycles)
  sim_t = [[[zeros(FloatType, length(eco_species_ids[eco_id]), n_bins) for _ in 1:n_cycles] for eco_id in eachindex(eco_species_ids)] for _ in 1:Threads.maxthreadid()]
  perm_t, ages_t = _bin_scratch(soa, max_sim_year, loss_params)
  Tier4Acc(sim_t, perm_t, ages_t, t4_ref, loss_params, scales, n_species, eco_species_ids, site_counts, obs_counts, n_cycles, cell_plots, per_plot_mean, capture)
end
# accumulate this measurement into its cycle's bins (shared with tier 5)
@inline function _t4_visit!(acc::Tier4Acc, sc::YearScoring, site, year::Int)
  cyc = get(sc.cycle_map, (Int(site.mapcode), year), 0)
  if cyc != 0
    tid = Threads.threadid()
    accumulate_site_bins!(acc.sim_t[tid][site.eco_id][cyc], site, acc.loss_params, acc.perm_t[tid], acc.ages_t[tid])
  end
end
@inline site_visit!(acc::Tier4Acc, sc::YearScoring, site, i::Int, year::Int) = _t4_visit!(acc, sc, site, year)
function _t4_score!(acc::Tier4Acc)   # merge the threads into thread 1, then the tier-4 loss per eco
  for tid in 2:Threads.maxthreadid(), eco_id in eachindex(acc.eco_species_ids), cyc in 1:acc.n_cycles
    acc.sim_t[1][eco_id][cyc] .+= acc.sim_t[tid][eco_id][cyc]
  end
  calculate_t4_loss(acc.sim_t[1], acc.ref, acc.loss_params, acc.scales, acc.n_species, acc.eco_species_ids, acc.site_counts, acc.obs_counts, acc.n_cycles;
    cell_plots=acc.cell_plots, per_plot_mean=acc.per_plot_mean, capture=acc.capture)
end
function finish!(acc::Tier4Acc)
  eco_losses = _t4_score!(acc)
  sum(eco_losses), eco_losses
end

# --- tier 5 = tier 3 (per-measurement per-site loss) + tier 4 (per-cycle bins) in ONE sim pass ------------------------------------------------------
mutable struct Tier5Acc{T4<:Tier4Acc,LP,SC} <: TierAcc
  t4::T4
  sites_results::Vector{PU.SiteLoss}              # this year's per-site losses (undef where the site was not measured)
  years_results::Vector{PU.SiteLoss}              # per sim year: the sum of the sites' losses
  eco_w::Vector{Vector{FloatType}}
  eco_agb::Vector{Vector{FloatType}}
  eco_site_agb::Vector{FloatType}
  eco_obs::Vector{Int}
  loss_params::LP
  scales::SC
  n_species::Int
  eco_species_ids::Vector{Vector{Int}}
  debug::Bool
end
function Tier5Acc(t4::Tier4Acc, soa, max_sim_year::Int, loss_params, scales, n_species::Int, eco_species_ids, debug::Bool)
  ne = eachindex(eco_species_ids)
  Tier5Acc(t4, Vector{PU.SiteLoss}(undef, 0), Vector{PU.SiteLoss}(undef, max_sim_year + 1),
    [zeros(FloatType, n_species) for _ in ne], [zeros(FloatType, n_species) for _ in ne], zeros(FloatType, length(eco_species_ids)), zeros(Int, length(eco_species_ids)),
    loss_params, scales, n_species, eco_species_ids, debug)
end
begin_year!(acc::Tier5Acc, soa) = (acc.sites_results = Vector{PU.SiteLoss}(undef, soa.n); nothing)
@inline function site_visit!(acc::Tier5Acc, sc::YearScoring, site, i::Int, year::Int)
  _t4_visit!(acc.t4, sc, site, year)
  gt_now, excluded, excl_c = _site_exclusions(sc, year, i, sc.spdf_plts[(site.ref_cn, site.eco_id)][year])
  acc.sites_results[i] = PU.calculate_site_loss2(year, site, acc.n_species, acc.eco_species_ids, gt_now, acc.loss_params, acc.scales; debug=acc.debug, excluded=excluded, excl_c=excl_c)
  nothing
end
function end_year!(acc::Tier5Acc, soa, year::Int)
  year_results_no_missing = PU.skipundef(acc.sites_results)
  if length(year_results_no_missing) > 0
    acc.years_results[year+1] = sum(year_results_no_missing)
  end
  for i in 1:soa.n
    isassigned(acc.sites_results, i) || continue
    eco_id = Int(getsite(soa, i).eco_id)
    sl = acc.sites_results[i]
    acc.eco_w[eco_id] .+= sl.sp_w_loss
    acc.eco_agb[eco_id] .+= sl.sp_agb_loss
    acc.eco_site_agb[eco_id] += sl.site_agb_loss
    acc.eco_obs[eco_id] += 1
  end
  nothing
end
# run_result = the SUM of the two scalar SiteLosses; eco_losses = [tier-3 per-eco ...; tier-4 per-eco ...] (length 2 * n_eco) -> `_mo_objectives` then
# yields 4 * n_ess objectives (W, AGB) x {tier3, tier4} (see its modulo over eco_species_ids).
function finish!(acc::Tier5Acc)
  ne = eachindex(acc.eco_species_ids)
  eco_losses_3 = [PU.SiteLoss(sp_w_loss=acc.eco_w[e], sp_agb_loss=acc.eco_agb[e], site_agb_loss=acc.eco_site_agb[e], num_sites=acc.eco_obs[e], num_obs=acc.eco_obs[e]) for e in ne]
  eco_losses_4 = _t4_score!(acc.t4)
  sum(PU.skipundef(acc.years_results)) + sum(eco_losses_4), vcat(eco_losses_3, eco_losses_4)
end

# the accumulator of a tier; the one place where the runtime tier is looked at (once per rep)
function make_tier_acc(tier::Int, soa, sc::YearScoring, a::NamedTuple)
  if tier == 1
    Tier1Acc(soa, sc.sim_years, a.starting_sim_year, a.max_sim_year, a.t1_ref, sc.loss_params, sc.n_species, sc.eco_species_ids)
  elseif tier == 2
    Tier2Acc(soa, sc.sim_years, a.starting_sim_year, a.t2_ref, sc.loss_params, sc.n_species, sc.eco_species_ids)
  elseif tier == 4 || tier == 5
    t4 = Tier4Acc(soa, sc.sim_years, sc.cycle_map, a.starting_sim_year, a.max_sim_year, a.t4_ref, a.n_cycles, sc.loss_params, sc.scales, sc.n_species, sc.eco_species_ids,
      a.cell_plots, a.per_plot_mean, a.capture_t4)
    tier == 4 ? t4 : Tier5Acc(t4, soa, a.max_sim_year, sc.loss_params, sc.scales, sc.n_species, sc.eco_species_ids, sc.debug)
  else
    Tier3Acc(soa, a.max_sim_year, sc.loss_params, sc.scales, sc.n_species, sc.eco_species_ids)
  end
end

# --- mean-over-reps (tier-4 free sim): every rep's sim histogram is accumulated, ONE loss is scored on the mean ----------------------------------
mutable struct MeanOverReps
  hist::Vector{Vector{Matrix{FloatType}}}
  counts::Union{Nothing,Tuple{Vector{Vector{Int}},Vector{Vector{Int}}}}
end
_mor_add!(::Nothing, ::TierAcc) = nothing
function _mor_add!(m::MeanOverReps, acc::Tier4Acc)
  for eco_id in eachindex(acc.eco_species_ids), cyc in 1:acc.n_cycles
    m.hist[eco_id][cyc] .+= acc.sim_t[1][eco_id][cyc]
  end
  m.counts === nothing && (m.counts = (acc.site_counts, acc.obs_counts))
  nothing
end

# --- the per-year site loop -------------------------------------------------------------------------------------------------------------------
# ONE threaded loop over the sites for every tier; `acc` and `log` are concrete, so `site_visit!` / `_log_site!` are resolved at compile time.
function _score_sites!(acc::TierAcc, log, soa, year::Int, sc::YearScoring)
  sim_years_all = sc.sim_years
  @maybe_threads soa.parallel for i in 1:soa.n
    @inbounds begin
      site = getsite(soa, i)
      !site.active && continue
      sim_years = sim_years_all[site.mapcode]
      if year in sim_years
        site_visit!(acc, sc, site, i, year)
        _log_site!(log, i, site, year)
      end
      if year == last(sim_years)
        site.active = false
      end
    end
  end
  nothing
end
