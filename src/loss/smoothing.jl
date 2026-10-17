#StructTypes.StructType(::Type{SAState}) = StructTypes.Struct()
#StructTypes.StructType(::Type{SACandidate}) = StructTypes.Struct()
#StructTypes.StructType(::Type{SiteLoss}) = StructTypes.Struct()
#StructTypes.StructType(::Type{Random.Xoshiro}) = StructTypes.Struct()
#
function save_json(path::String, s)
  open(path, "w") do io
    JSON3.write(io, s)
  end
end


@inline function get_smoothing_window(; smoothing_window::Int=Int(3), smoothing_variance::FloatType=FloatType(1.0f0))
  w = ((-smoothing_window:smoothing_window) ./ smoothing_variance) .^ FloatType(2.0f0) .* FloatType(-0.5f0) .|> exp
  w ./= sum(w)
  return w
end

@inline function smoothen_bin_cdf(p; w::Vector{FloatType}, age_bins::AgeBins)::Vector{FloatType}
  pc = smooth_ages(; ages=p, smoothing_window=w)
  @assert !any(isnan.(pc)) "smooth NaN"
  pc_bin = bin_ages(pc; age_bins=age_bins.bins_idx, last_bin_open=age_bins.last_bin_open)
  pc_bin_cdf = cumsum(pc_bin)
  @assert !any(isnan.(pc_bin_cdf)) "cumsum NaN"
  if pc_bin_cdf[end] > zero(FloatType)
    pc_bin_cdf ./= pc_bin_cdf[end]
  end
  return pc_bin_cdf

end

# W-hinge variant: forgive the (absolute) binned sim biomass TOWARD obs_bin by up to _w_hinge_thresh(obs_i) per
# bin (sim_i -= clamp(sim_i − obs_i, −t, t)), THEN cumsum + normalize → CDF. obs_bin = rec.sp_age_agb (absolute
# binned obs). Uses raw (unsmoothed) binning so the ±band stays in absolute g/m² — for smoothing_window_size=1
# (this config) that equals smoothen_bin_cdf's binning; with age-smoothing on, the forgiveness path stays unsmoothed.
@inline function smoothen_bin_cdf_forgive(p, obs_bin, lp::LossParams; age_bins::AgeBins)::Vector{FloatType}
  pc_bin = bin_ages(p; age_bins=age_bins.bins_idx, last_bin_open=age_bins.last_bin_open)
  @inbounds for i in eachindex(pc_bin)
    t = _w_hinge_thresh(lp, obs_bin[i])
    pc_bin[i] -= clamp(pc_bin[i] - obs_bin[i], -t, t)
    pc_bin[i] < zero(FloatType) && (pc_bin[i] = zero(FloatType))
  end
  pc_bin_cdf = cumsum(pc_bin)
  pc_bin_cdf[end] > zero(FloatType) && (pc_bin_cdf ./= pc_bin_cdf[end])
  return pc_bin_cdf
end



@inline function smooth_ages(; ages::Vector{FloatType}, smoothing_window::Vector{FloatType})::Vector{FloatType}
  if length(smoothing_window) < 2
    return ages[:]
  end
  smoothed_ages = ImageFiltering.imfilter(ages, smoothing_window, "symmetric")
  @assert !any(isnan.(smoothed_ages)) "filter NaN"
  s = sum(smoothed_ages)
  if s > zero(FloatType)
    smoothed_ages ./= s
  end
  return smoothed_ages
end

function smoothen_ref_years(df::DataFrame, loss_params::LossParams, max_age::Int; debug=false)::DataFrame
  spdf = combine(groupby(df, [:plot_id, :eco_id, :measdate, :start_measdate, :eco_species_id])) do rows
    ages = zeros(FloatType, max_age + (UIntType(length(loss_params.smoothing_weights) >> 1))
    )
    for row in eachrow(rows)
      ages[row.age_calc] += row.agb_sum
    end
    row = rows[1, :]
    sim_year = Dates.value.(Dates.Day.(row.measdate - row.start_measdate)) ./ 365.25 .|> round .|> Int
    @assert sim_year >= 0 "negative sim_year $row"
    cdf = smoothen_bin_cdf(ages; w=loss_params.smoothing_weights, age_bins=loss_params.age_bins)
    # binned ABSOLUTE observed biomass per age bin (unnormalized) — the per-cohort AGB reference (obs_i)
    age_agb = bin_ages(ages; age_bins=loss_params.age_bins.bins_idx, last_bin_open=loss_params.age_bins.last_bin_open)
    if debug
      smoothed_ages = smooth_ages(; ages=ages, smoothing_window=loss_params.smoothing_weights)
      binned_ages = bin_ages(smoothed_ages; age_bins=loss_params.age_bins.bins_idx, last_bin_open=loss_params.age_bins.last_bin_open)
      (; sim_year=[sim_year], data_agb_sum=[sum(rows.agb_sum)], data_agbs_cdf=[cdf], data_age_agb=[age_agb],
        ages=[ages], smoothed_ages=[smoothed_ages], binned_ages=[binned_ages])
    else
      (; sim_year=[sim_year], data_agb_sum=[sum(rows.agb_sum)], data_agbs_cdf=[cdf], data_age_agb=[age_agb])
    end
  end
  return spdf


  #plot_id x eco_id, measdate, sim_year, swhd, spgrpcd, species_id -> smoothened(biomass by age)
  #plot_id x eco_id, measdate, sim_year, swhd, spgrpcd -> smoothened_binned(biomass by age)
  #plot_id x eco_id, measdate, sim_year, swhd -> biomass by age

end

function get_bin_widths(; age_bins::Vector{Int}, last_bin_open::Bool)
  # returns the bin widths for wasser1 (ie K-1 widths)
  local bin_widths
  if last_bin_open
    @assert length(age_bins) > 0 "insufficint bins, must be at least 1"
    bin_widths = age_bins .- [0; age_bins[begin:end-1]]
  else
    @assert length(age_bins) > 1 "insufficint bins, must be at least 2"
    bin_widths = age_bins[begin:end-1] .- [0; age_bins[begin:end-2]]
  end
  #return ones(FloatType, length(bin_widths))
  return bin_widths

end

@inline function bin_ages(ages::Vector{FloatType}; age_bins::Vector{Int}, last_bin_open::Bool)::Vector{FloatType}
  bins = length(age_bins)
  if last_bin_open
    bins += 1
  end
  bs = zeros(FloatType, bins)
  current_bin = 1
  current_age = age_bins[current_bin]
  for (i, a) in enumerate(ages)
    if i >= current_age
      current_bin += 1
      if current_bin > length(age_bins)
        if last_bin_open
          current_age = Inf
        else
          break
        end
      else
        current_age = age_bins[current_bin]
      end
    end
    bs[current_bin] += ages[i]
  end
  return bs
end



# Weighted W1(L1) sum over bins. `sim_cdf===nothing` ⇒ sim absent (|0−ref| = ref). Applies the count-balance
# weight for Sim A (tier-3) when enabled and MODE∈{:a,:both}: per-year bin k → coarse bin sc.cbal.coarse[k].
@inline function _w1_sum(lp::LossParams, sc::LossScales, ref_cdf, sim_cdf, bw, gsp::Int, eco::Int)::FloatType
  use = lp.cbal_on && lp.cbal_mode !== :b && !isempty(sc.cbal.w)
  s = zero(FloatType)
  if use
    w = @inbounds sc.cbal.w[gsp, eco]; coarse = sc.cbal.coarse
    @inbounds for k in eachindex(bw)
      d = sim_cdf === nothing ? ref_cdf[k] : abs(sim_cdf[k] - ref_cdf[k])
      s += w[coarse[k]] * bw[k] * d
    end
  else
    @inbounds for k in eachindex(bw)
      d = sim_cdf === nothing ? ref_cdf[k] : abs(sim_cdf[k] - ref_cdf[k])
      s += bw[k] * d
    end
  end
  return s
end

# Build the count-balance tables (per gsp×eco weight over COARSE age_idx bins + per-year→coarse map) from the TRAIN reference cohort records.
# Returns a CbalTables, built once at setup and shared (frozen) by every reference of the run.
function build_cbal_tables(splots, coarse_bins::AgeBins, per_year_bins::AgeBins, eco_species_ids, n_species::Int; beta::Float64, pct_floor::Float64=0.10)::CbalTables
  ne = length(eco_species_ids)
  nb = length(coarse_bins.bins_idx) + (coarse_bins.last_bin_open ? 1 : 0)
  counts = [zeros(Int, nb) for _ in 1:n_species, _ in 1:ne]
  for r in eachrow(splots)
    gsp = Int(r.species_id); eco = Int(r.eco_id)
    (1 <= gsp <= n_species && 1 <= eco <= ne) || continue
    b = find_age_bin(Int(round(r.age_calc)), coarse_bins)
    b >= 1 && (counts[gsp, eco][b] += 1)
  end
  W = Matrix{Vector{FloatType}}(undef, n_species, ne)
  for gsp in 1:n_species, eco in 1:ne
    c = counts[gsp, eco]; tot = sum(c); w = ones(FloatType, nb)
    if tot > 0
      occ = Int[]
      for b in 1:nb
        if c[b] > 0
          q = round(Int, 1 / pct_floor)                          # integer grid steps: 10 / 20 / 50 for floor 0.10 / 0.05 / 0.02
          pct = max(round(q * c[b] / tot) / q, 1 / q)            # count-share on a 1/q grid, floored at 1/q(=pct_floor). q=10 is EXACTLY the old round(10·share)/10 ⌊0.10⌋
          #   grid step and floor move together (no gap); using integer q (not share/pct_floor) avoids the 0.1-not-representable rounding drift. yaml w_count_pct_floor.
          neff = pct * tot                                        # effective #cohorts = percentile × total (caps rare-bin weight)
          w[b] = FloatType((1 - beta) / (1 - beta^neff)); push!(occ, b)
        end
        # empty bins keep weight 1 (neutral): NOT zero — else a sim cohort placed there goes unpenalized
      end
      m = sum(w[b] for b in occ) / length(occ)                     # normalize occupied → mean 1 (redistribute)
      m > 0 && for b in occ; w[b] /= m; end
    end
    W[gsp, eco] = w
  end
  # map each Sim-A W bin k to its coarse age_idx bin via the bin's upper age edge (works for per-year OR coarse)
  coarse = [find_age_bin(per_year_bins.bins_idx[k] - 1, coarse_bins) for k in eachindex(per_year_bins.bin_widths)]
  return CbalTables(W, coarse)
end
