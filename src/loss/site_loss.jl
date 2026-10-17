# Sort `perm` (a permutation of cohort indices) by species, ties by index — the unique stable order, so it equals `sortperm(species)` and a stable
# sort of an ascending index list. In-place QuickSort on the (species, index) key: no scratch buffer, so the site accumulator allocates nothing.
@inline function _sort_perm_by_species!(perm::AbstractVector{Int}, csp)
  sort!(perm; alg=QuickSort, lt=(a, b) -> @inbounds(csp[a] < csp[b] || (csp[a] == csp[b] && a < b)))
  return perm
end

# Cohort-level exclusion (disturbance exclude-modes / recruits at entry, new scoring order): fill `perm` with the indices of the
# site's live cohorts that are NOT in excl_c (a list of (eco_species, age)), sorted by species; returns the count. The matching
# observed cohorts are removed from the reference by Pan._build_excl_plan (alternative SPDFGroundTruth for that site-year).
function _excl_perm!(perm::Vector{Int}, site, excl_c)::Int
  live = Int(site.live); resize!(perm, live); n = 0
  csp = site.c_species; cage = site.c_age
  @inbounds for j in 1:live
    ex = false
    for (esp, age) in excl_c
      if csp[j] == esp && cage[j] == age
        ex = true; break
      end
    end
    ex || (n += 1; perm[n] = j)
  end
  resize!(perm, n)
  _sort_perm_by_species!(perm, csp)
  n
end

"""
    calculate_site_loss2(year, site, n_species, eco_species_ids, spdf_plt, loss_params, scales; excluded, excl_c) -> SiteLoss

The per-site loss as a freshly allocated `SiteLoss` (tier 5 / diagnostics): the allocation-free accumulator `calculate_site_loss2_acc!` run on
zeroed per-species accumulators and private scratch. There is ONE implementation of the loss arithmetic.
`excluded` (eco-species to skip) and `excl_c` (cohort-level exclusions) as in `calculate_site_loss2_acc!`. `debug` logs the per-species losses.
"""
function calculate_site_loss2(current_year::Int, site::SiteView, n_species::Int, eco_species_ids::Vector{Vector{Int}}, spdf_plt::SPDFGroundTruth, loss_params::LossParams, scales::LossScales; lp::FloatType=one(FloatType), debug::Bool=false, excluded::Union{Nothing,Set{UIntType}}=nothing, excl_c::Union{Nothing,Vector{Tuple{UIntType,FloatType}}}=nothing)::SiteLoss
  sp_w_loss = zeros(FloatType, n_species)
  sp_agb_loss = zeros(FloatType, n_species)
  site_agb_loss = calculate_site_loss2_acc!(sp_w_loss, sp_agb_loss, current_year, site, n_species, eco_species_ids, spdf_plt, loss_params, scales,
    Int[], FloatType[], Bool[], FloatType[], FloatType[]; lp=lp, excluded=excluded, excl_c=excl_c)
  debug && @info "site loss" ref_cn=site.ref_cn year=current_year sp_w_loss sp_agb_loss site_agb_loss
  return SiteLoss(sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=site_agb_loss, num_sites=1)
end

@inline function find_age_bin(age::Int, age_bins::AgeBins)::Int
  for k in eachindex(age_bins.bins_idx)
    age < age_bins.bins_idx[k] && return k
  end
  age_bins.last_bin_open ? length(age_bins.bins_idx) + 1 : 0
end

function wasserstein1d(a::AbstractVector, b::AbstractVector)::FloatType
  (isempty(a) || isempty(b)) && return zero(FloatType)
  sa = sort(Float64.(a))
  sb = sort(Float64.(b))
  na, nb = length(sa), length(sb)
  ia, ib = 1, 1
  ca, cb = 0, 0
  prev_x = min(sa[1], sb[1])
  w1 = 0.0
  while ia <= na || ib <= nb
    va = ia <= na ? sa[ia] : Inf
    vb = ib <= nb ? sb[ib] : Inf
    x = min(va, vb)
    w1 += abs(ca / na - cb / nb) * (x - prev_x)
    while ia <= na && sa[ia] == x
      ca += 1
      ia += 1
    end
    while ib <= nb && sb[ib] == x
      cb += 1
      ib += 1
    end
    prev_x = x
  end
  return FloatType(w1)
end

# ─────────────────────────────────────────────────────────────────────────────────────────────────────
# Allocation-free loss path: THE loss implementation (calculate_site_loss2 wraps it). Arithmetic of /
# bin_ages / smoothen_bin_cdf_forgive, but every per-site/per-species temporary is a caller-owned per-thread
# scratch buffer, and the per-species (W, AGB) losses are accumulated straight into per-eco accumulator arrays
# (w_acc/agb_acc) instead of a freshly-allocated per-site SiteLoss. Eliminates the hot-path GC pressure that
# hurts most at high thread counts. The originals are kept as the numerical reference (and for tier-5/other).
# ─────────────────────────────────────────────────────────────────────────────────────────────────────

# In-place bin_ages: `out` is resized to the bin count and overwritten. Bit-identical to bin_ages.
@inline function bin_ages!(out::Vector{FloatType}, ages::AbstractVector{FloatType}; age_bins::Vector{Int}, last_bin_open::Bool)
  nb = length(age_bins) + (last_bin_open ? 1 : 0)
  length(out) == nb || resize!(out, nb)
  fill!(out, zero(FloatType))
  current_bin = 1
  current_age = age_bins[current_bin]
  @inbounds for i in eachindex(ages)
    if i >= current_age
      current_bin += 1
      if current_bin > length(age_bins)
        last_bin_open ? (current_age = typemax(Int)) : break   # typemax(Int) ≡ Inf for realistic ages
      else
        current_age = age_bins[current_bin]
      end
    end
    out[current_bin] += ages[i]
  end
  return out
end

# In-place smoothen_bin_cdf_forgive: bins `p` into `cdf`, applies the ±band forgiveness toward `obs_bin`, then
# turns it into a normalized CDF in place. Bit-identical to smoothen_bin_cdf_forgive.
@inline function smoothen_bin_cdf_forgive!(cdf::Vector{FloatType}, p::AbstractVector{FloatType}, obs_bin, lp::LossParams; age_bins::AgeBins)
  bin_ages!(cdf, p; age_bins=age_bins.bins_idx, last_bin_open=age_bins.last_bin_open)
  @inbounds for i in eachindex(cdf)
    t = _w_hinge_thresh(lp, obs_bin[i])
    cdf[i] -= clamp(cdf[i] - obs_bin[i], -t, t)
    cdf[i] < zero(FloatType) && (cdf[i] = zero(FloatType))
  end
  cumsum!(cdf, cdf)   # forward running sum — aliasing src===dst is safe (each element depends only on priors)
  @inbounds (cdf[end] > zero(FloatType)) && (cdf ./= cdf[end])
  return cdf
end

# Accumulating per-species loss: adds this species' (W, AGB) loss into w_acc[gsp]/agb_acc[gsp] and returns the
# updated running site_agb_loss. `ages` (site-level, length max_age) is zeroed and refilled here; `cdf`/`agb_bins`
# are per-thread bin-sized scratch. 
@inline function calculate_species_loss_acc!(; sp, gsp, site, ages, p, sp_start_idx, sp_end_idx, spdf_plt, loss_params, scales,
    w_acc, agb_acc, site_agb_loss, cdf, agb_bins, lp::FloatType=one(FloatType))
  sim_agb_sum = zero(FloatType)   # left fold over the species' cohorts (== sum of the indexed view, without materialising the index vector)
  @inbounds for a in @view p[sp_start_idx:sp_end_idx]; sim_agb_sum += site.c_bio[a]; end
  agb_val = sim_agb_sum
  w_val = zero(FloatType)
  site_agb_loss += sim_agb_sum
  if spdf_plt.keys[sp]
    rec = @inbounds spdf_plt.records[sp]
    fill!(ages, zero(FloatType))
    @inbounds for a in @view p[sp_start_idx:sp_end_idx]
      ages[UIntType(site.c_age[a])] += site.c_bio[a]
    end
    if loss_params.w_hinge   # ±band "benefit of the doubt" (Sim A) — allocation-free forgive path
      sim_age_cdf = smoothen_bin_cdf_forgive!(cdf, ages, rec.sp_age_agb, loss_params; age_bins=loss_params.age_bins)
    else           # non-hinge path is rare (config uses w_hinge) — keep the allocating helper
      sim_age_cdf = smoothen_bin_cdf(ages; w=loss_params.smoothing_weights, age_bins=loss_params.age_bins)
    end
    let bw = loss_params.age_bins.bin_widths
      w_val = _w_finish(loss_params, scales, _w1_sum(loss_params, scales, rec.sp_age_cdf, sim_age_cdf, bw, Int(gsp), Int(site.eco_id)), Int(gsp), Int(site.eco_id))
    end
    bin_ages!(agb_bins, ages; age_bins=loss_params.age_bins.bins_idx, last_bin_open=loss_params.age_bins.last_bin_open)
    acc = zero(FloatType)
    if loss_params.agb_hinge
      @inbounds for i in eachindex(rec.sp_age_agb)
        acc += _agb_pow(loss_params, _hinge_relu(loss_params, abs(agb_bins[i] - rec.sp_age_agb[i]) - _agb_hinge_thresh(loss_params, rec.sp_age_agb[i])))
      end
      agb_val = loss_params.lambda * _agb_finish(loss_params, scales, acc)
    else
      @inbounds for i in eachindex(rec.sp_age_agb)
        acc += (sqrt(max(agb_bins[i], zero(FloatType))) - sqrt(max(rec.sp_age_agb[i], zero(FloatType))))^2
      end
      agb_val = loss_params.lambda * acc
    end
    site_agb_loss -= rec.sp_agb_sum
  end
  @inbounds w_acc[gsp] += w_val
  @inbounds agb_acc[gsp] += agb_val
  return site_agb_loss
end

# Accumulating per-site loss: adds every species' (W, AGB) loss for this site into the per-eco accumulators
# w_acc/agb_acc, and returns this site's |site_agb_loss| (the caller adds it to eco3_site_agb and bumps obs).
# `perm`/`ages`/`insite`/`cdf`/`agb_bins` are per-thread scratch (resized in place). 
function calculate_site_loss2_acc!(w_acc::Vector{FloatType}, agb_acc::Vector{FloatType}, current_year::Int, site::SiteView,
    n_species::Int, eco_species_ids::Vector{Vector{Int}}, spdf_plt::SPDFGroundTruth, loss_params::LossParams, scales::LossScales,
    perm::Vector{Int}, ages::Vector{FloatType}, insite::Vector{Bool}, cdf::Vector{FloatType}, agb_bins::Vector{FloatType};
    lp::FloatType=one(FloatType), excluded::Union{Nothing,Set{UIntType}}=nothing, excl_c::Union{Nothing,Vector{Tuple{UIntType,FloatType}}}=nothing)::FloatType
  _excl(sp) = excluded !== nothing && UIntType(sp) in excluded
  eco_n_species = length(site.sp_mature)
  species_id_map = eco_species_ids[site.eco_id]
  length(insite) == eco_n_species || resize!(insite, eco_n_species)
  fill!(insite, false)
  site_agb_loss = zero(FloatType)

  nkeep = excl_c === nothing ? Int(site.live) : _excl_perm!(perm, site, excl_c)
  if site.live > 0 && nkeep > 0
    c_species = @view site.c_species[1:site.live]
    max_age = UIntType(ceil(maximum(@view site.c_age[1:site.live])))
    max_age += UIntType(length(loss_params.smoothing_weights) >> 1)
    if excl_c === nothing
      resize!(perm, site.live)
      @inbounds for i in eachindex(perm); perm[i] = i; end
      _sort_perm_by_species!(perm, c_species)
    end
    resize!(ages, max_age)
    p = perm
    sp_start_idx = 1
    prev_sp = @inbounds site.c_species[p[sp_start_idx]]
    for i in eachindex(p)
      sp = @inbounds c_species[p[i]]
      @inbounds insite[sp] = true
      if sp != prev_sp
        sp_end_idx = i - 1
        if !_excl(prev_sp)
          site_agb_loss = calculate_species_loss_acc!(; sp=prev_sp, gsp=species_id_map[prev_sp],
            site=site, ages=ages, p=p, sp_start_idx=sp_start_idx, sp_end_idx=sp_end_idx,
            spdf_plt=spdf_plt, loss_params=loss_params, scales=scales, w_acc=w_acc, agb_acc=agb_acc,
            site_agb_loss=site_agb_loss, cdf=cdf, agb_bins=agb_bins, lp=lp)
        end
        prev_sp = sp
        sp_start_idx = i
      end
      if i == length(p)
        sp_end_idx = i
        if !_excl(sp)
          site_agb_loss = calculate_species_loss_acc!(; sp=sp, gsp=species_id_map[sp],
            site=site, ages=ages, p=p, sp_start_idx=sp_start_idx, sp_end_idx=sp_end_idx,
            spdf_plt=spdf_plt, loss_params=loss_params, scales=scales, w_acc=w_acc, agb_acc=agb_acc,
            site_agb_loss=site_agb_loss, cdf=cdf, agb_bins=agb_bins, lp=lp)
        end
      end
    end
  end

  # species present in REF but absent from SIM (plain loop — avoids the `keys .& .!insite` BitVector allocs)
  @inbounds for sp in 1:length(spdf_plt.keys)
    (spdf_plt.keys[sp] && !insite[sp]) || continue
    _excl(sp) && continue
    rec = spdf_plt.records[UIntType(sp)]
    gsp = species_id_map[sp]
    let bw = loss_params.age_bins.bin_widths
      w_acc[gsp] += _w_finish(loss_params, scales, _w1_sum(loss_params, scales, rec.sp_age_cdf, nothing, bw, Int(gsp), Int(site.eco_id)), Int(gsp), Int(site.eco_id))
    end
    if loss_params.agb_hinge
      agb_acc0 = zero(FloatType)
      for i in eachindex(rec.sp_age_agb)
        agb_acc0 += _agb_pow(loss_params, _hinge_relu(loss_params, rec.sp_age_agb[i] - _agb_hinge_thresh(loss_params, rec.sp_age_agb[i])))
      end
      agb_acc[gsp] += loss_params.lambda * _agb_finish(loss_params, scales, agb_acc0)
    else
      agb_acc[gsp] += loss_params.lambda * rec.sp_agb_sum
    end
    site_agb_loss -= rec.sp_agb_sum
  end

  return abs(site_agb_loss)
end
