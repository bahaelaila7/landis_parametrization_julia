function accumulate_site_bins!(t1_sim_eco::Matrix{FloatType}, site, loss_params::PU.LossParams, scratch_perm::Vector{Int}, scratch_ages::Vector{FloatType})
  site.live == 0 && return
  nlive = site.live
  c_species = @view site.c_species[1:nlive]
  max_age = Int(ceil(maximum(@view site.c_age[1:nlive]))) + length(loss_params.smoothing_weights) >> 1
  if length(scratch_perm) < nlive
    resize!(scratch_perm, nlive)
  end
  resize!(scratch_ages, max_age)   # exact length so `ages` is a dense Vector — imfilter (via smoothen_bin_cdf) crashes natively on a SubArray view
  p = @view scratch_perm[1:nlive]
  sortperm!(p, c_species)
  ages = scratch_ages
  sp_start = 1
  prev_sp = c_species[p[1]]
  function conclude!(sp, s, e)
    fill!(ages, 0f0)
    for a in @view p[s:e]
      ages[clamp(Int(ceil(site.c_age[a])), 1, max_age)] += site.c_bio[a]
    end
    agb = sum(ages)
    cdf = PU.smoothen_bin_cdf(ages; w=loss_params.smoothing_weights, age_bins=loss_params.age_bins)
    t1_sim_eco[sp, 1] += cdf[1] * agb
    for k in 2:length(cdf)
      t1_sim_eco[sp, k] += (cdf[k] - cdf[k-1]) * agb
    end
  end
  for i in eachindex(p)
    sp = c_species[p[i]]
    if sp != prev_sp
      conclude!(prev_sp, sp_start, i - 1)
      prev_sp = sp
      sp_start = i
    end
    if i == length(p)
      conclude!(sp, sp_start, i)
    end
  end
end

function calculate_aggregate_loss(t1_sim::Vector{Matrix{FloatType}}, t1_ref::Vector{Matrix{FloatType}}, loss_params::PU.LossParams, n_species::Int, eco_species_ids::Vector{Vector{Int}}, eco_site_counts::Vector{Int}, eco_obs_counts::Vector{Int})::Vector{PU.SiteLoss}
  eco_losses = Vector{PU.SiteLoss}(undef, length(t1_sim))
  for eco_id in eachindex(t1_sim)
    sp_w_loss = zeros(FloatType, n_species)
    sp_agb_loss = zeros(FloatType, n_species)
    sp_map = eco_species_ids[eco_id]
    for sp_eco in eachindex(sp_map)
      gsp = sp_map[sp_eco]
      sim_row = @view t1_sim[eco_id][sp_eco, :]
      ref_row = @view t1_ref[eco_id][sp_eco, :]
      tot_sim = sum(sim_row)
      tot_ref = sum(ref_row)
      let bw = loss_params.age_bins.bin_widths
        s = zero(FloatType)
        acc_sim = zero(FloatType)
        acc_ref = zero(FloatType)
        @inbounds for k in eachindex(bw)
          acc_sim += sim_row[k]
          acc_ref += ref_row[k]
          # W1 on the NORMALIZED biomass-by-age CDF → shape only, invariant to AGB level.
          cdf_sim = tot_sim > 0 ? acc_sim / tot_sim : zero(FloatType)
          cdf_ref = tot_ref > 0 ? acc_ref / tot_ref : zero(FloatType)
          s += bw[k] * (cdf_sim - cdf_ref)^2
        end
        sp_w_loss[gsp] = s
      end
      # AGB LEVEL as a separate, gentle sqrt-difference term (weighted by loss_params.lambda).
      sp_agb_loss[gsp] = loss_params.lambda * (sqrt(tot_sim) - sqrt(tot_ref))^2
    end
    eco_losses[eco_id] = PU.SiteLoss(sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=zero(FloatType), num_sites=eco_site_counts[eco_id], num_obs=eco_obs_counts[eco_id])
  end
  return eco_losses
end


function calculate_t2_loss(t2_sim_bins, t2_sim_total, t2_ref, n_species, eco_species_ids, eco_site_counts, eco_obs_counts, loss_params)
  eco_losses = Vector{PU.SiteLoss}(undef, length(eco_species_ids))
  n_bins = size(t2_ref.bins[1], 2)
  _mean(v) = isempty(v) ? zero(FloatType) : sum(v) / length(v)
  for eco_id in eachindex(eco_species_ids)
    sp_w_loss = zeros(FloatType, n_species)
    sp_agb_loss = zeros(FloatType, n_species)
    sp_map = eco_species_ids[eco_id]
    for sp_eco in eachindex(sp_map)
      gsp = sp_map[sp_eco]
      for b in 1:n_bins
        sp_w_loss[gsp] += PU.wasserstein1d(t2_sim_bins[eco_id][sp_eco, b], t2_ref.bins[eco_id][sp_eco, b])
      end
      # AGB LEVEL as a gentle sqrt-difference of mean totals (weighted by loss_params.lambda).
      sp_agb_loss[gsp] = loss_params.lambda * (sqrt(_mean(t2_sim_total[eco_id][sp_eco])) - sqrt(_mean(t2_ref.total[eco_id][sp_eco])))^2
    end
    eco_losses[eco_id] = PU.SiteLoss(sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=zero(FloatType), num_sites=eco_site_counts[eco_id], num_obs=eco_obs_counts[eco_id])
  end
  return eco_losses
end

# Tier 4: population-level loss. t4_sim/t4_ref are [eco][cycle] matrices of per-bin biomass
# (summed across the plots measured in that calendar cycle). Wasserstein-1 on the biomass-by-age
# CDF is computed per (eco, cycle, species); cycles are summed into one SiteLoss per eco
# (num_obs = total measurements in the eco, so get_total_loss is per-measurement-averaged W1).
# Diagnostics: fit_params(...; capture_t4=T4Capture()) → every calculate_t4_loss of that call stashes `(sim, ref, n_cycles, site_counts)` into `.data`
# (the last one wins: Sim B's when a dual_b follows Sim A). tools/simB_regen_diff.jl.
mutable struct T4Capture
  data::Union{Nothing,NamedTuple}
end
T4Capture() = T4Capture(nothing)
function calculate_t4_loss(t4_sim, t4_ref, loss_params::PU.LossParams, scales::PU.LossScales, n_species, eco_species_ids, t4_site_counts, t4_obs_counts, n_cycles; cell_plots=nothing, per_plot_mean::Bool, capture::Union{Nothing,T4Capture}=nothing)
  capture === nothing || (capture.data = (sim=deepcopy(t4_sim), ref=t4_ref, n_cycles=n_cycles, site_counts=deepcopy(t4_site_counts)))
  eco_losses = Vector{PU.SiteLoss}(undef, length(eco_species_ids))
  for eco_id in eachindex(eco_species_ids)
    sp_w_loss = zeros(FloatType, n_species)
    sp_agb_loss = zeros(FloatType, n_species)
    sp_map = eco_species_ids[eco_id]
    bw = loss_params.age_bins.bin_widths
    ncell = 0   # # of (cycle,species) cells with data → tier-4 loss is AVERAGED per cell (dataset-size invariant)
    for cyc in 1:n_cycles
      sim_c = t4_sim[eco_id][cyc]
      ref_c = t4_ref[eco_id][cyc]
      # per-plot-mean: divide the AGB level by the # of distinct plots in this (eco,cycle) so the loss is
      # mean biomass-per-plot-by-age (invariant to plot count). W is CDF-normalized → pcinv cancels there.
      # Uses the SAME cell plot counts the AGB scale divides by, so numerator/denominator stay consistent.
      pcinv = (per_plot_mean && cell_plots !== nothing) ?
        one(FloatType) / FloatType(max(1, cell_plots[eco_id][cyc])) : one(FloatType)
      for sp_eco in eachindex(sp_map)
        gsp = sp_map[sp_eco]
        sim_row = @view sim_c[sp_eco, :]
        ref_row = @view ref_c[sp_eco, :]
        tot_sim = sum(sim_row)
        tot_ref = sum(ref_row)
        (tot_sim > 0 || tot_ref > 0) && (ncell += 1)
        s = zero(FloatType)
        a = zero(FloatType)
        acc_sim = zero(FloatType)
        acc_ref = zero(FloatType)
        # count-balance reweight (Sim B / tier-4): coarse bins ARE the age_idx bins → direct CBAL_W lookup.
        usew = loss_params.cbal_on && loss_params.cbal_mode !== :a && !isempty(scales.cbal.w)
        cbw = usew ? (@inbounds scales.cbal.w[gsp, eco_id]) : FloatType[]
        @inbounds for k in eachindex(bw)
          acc_sim += sim_row[k]
          acc_ref += ref_row[k]
          # W1 (L1) on the NORMALIZED biomass-by-age CDF → shape only, invariant to AGB level.
          cdf_sim = tot_sim > 0 ? acc_sim / tot_sim : zero(FloatType)
          cdf_ref = tot_ref > 0 ? acc_ref / tot_ref : zero(FloatType)
          s += (usew ? cbw[k] : one(FloatType)) * bw[k] * abs(cdf_sim - cdf_ref)
          # AGB LEVEL per age bin: ABSOLUTE biomass diff (size-respecting); the global ÷Σobs makes it
          # intensive (numerator & Σobs both scale with #plots → no train/val artifact, no /cnt needed).
          sm = sim_row[k] * pcinv; rm = ref_row[k] * pcinv   # per-plot-mean AGB level (pcinv=1 when off)
          a += loss_params.agb_hinge ?
            PU._hinge_relu(loss_params, abs(sm - rm) - PU._agb_hinge_thresh(loss_params, rm)) :   # linear excess; power after norm
            (sqrt(sm) - sqrt(rm))^2
        end
        sp_w_loss[gsp] += PU._w_finish(loss_params, scales, s, Int(gsp), Int(eco_id))  # raw (softplus) W1 when cell_norm; else (W1/scale)^p
        sp_agb_loss[gsp] += loss_params.lambda * PU._agb_finish(loss_params, scales, a)   # raw hinge excess when cell_norm; else (excess/scale)^p
      end
    end
    # num_obs = # cells (not measurements): tier-4 loss is averaged PER CELL so it doesn't scale with #plots.
    eco_losses[eco_id] = PU.SiteLoss(sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=zero(FloatType),
      num_sites=max(1, sum(t4_site_counts[eco_id])), num_obs=max(1, ncell))
  end
  return eco_losses
end

# ── Loss scales: the normalisation derived from ONE reference set (pure; no module state). ────────────────────────────────────────────────
#
# `compute_loss_scales` is called once per reference (train Sim A, train Sim B, val Sim A, val Sim B, …) and the result is passed explicitly to
# `fit_params` and `_mo_objectives`:
#   * global ratio-of-sums (agb_normalize / w_normalize): Σ observed AGB, Σ per-reference-max W1 → `w_global`/`agb_global`;
#   * per-cell (cell_norm): `w_cell`/`agb_cell`[gsp,eco] (Sim A reads the per-plot spdf records, Sim B the aggregated t4_ref), × w_scale_factor on W;
#   * the softplus β (fixed, or per-cell auto-knee computed ONCE from the first reference and shared by the others);
#   * RANKW: computed from the TRAIN reference only (`rankw=nothing` marks the train reference) and shared (same matrix) by every other reference,
#     val included, so train and val weight the same eco×lu×sp cells the same way. The scales themselves are intensive ratio-of-sums, so they are
#     recomputed per reference.
# `cbal`/`ncoh` are the run-wide TRAIN tables (count-balance weights, per-cell cohort counts), carried inside the result.
