"Per-(species, eco) TRAIN cohort-row counts: the input of the `rankw_mode = :cbal_pct` weights."
function _cell_ncoh(splots, n_species::Int, n_eco::Int)::Matrix{Int}
  nc = zeros(Int, n_species, n_eco)
  for r in eachrow(splots)
    (1 <= Int(r.species_id) <= n_species && 1 <= Int(r.eco_id) <= n_eco) && (nc[Int(r.species_id), Int(r.eco_id)] += 1)
  end
  return nc
end

function compute_loss_scales(lp::PU.LossParams, search_tier::Int, spdf_plts, t4_ref, eco_species_ids, n_species::Int;
    cell_plots=nothing, per_plot_mean::Bool, cbal::PU.CbalTables=PU.NO_CBAL, ncoh::Matrix{Int}=zeros(Int, 0, 0), dom::Vector{Bool}=Bool[],
    rankw::Union{Nothing,Matrix{FloatType}}=nothing, softplus_cell_from::Union{Nothing,PU.LossScales}=nothing)::PU.LossScales
  bw = lp.age_bins.bin_widths
  ne = length(eco_species_ids)
  # --- softplus β ---
  sp_beta = zero(FloatType); sp_beta_cell = zeros(FloatType, 0, 0)
  if lp.w_softplus
    if !lp.w_smooth_auto_knee
      # Knee adjustment OFF (default): fixed β for W, independent of the grid's theoretical max — the
      # band·conc·wmax derivation over-inflates the knee on fine grids (unbinned Σbw≈399) and mutes real W.
      sp_beta = lp.w_smooth_beta
    else
      # Knee adjustment ON — PER-CELL, bounded by each cell's REFERENCE AGE STRETCH. The theoretical W1 max
      # for a (species,eco) cell is Σ bw only up to K* = the cell's OLDEST reference-cohort bin; past K* the
      # reference CDF is saturated (F_ref=1), so |ΔF| can't realistically be 1 there — summing the whole grid
      # (Σbw−bw[end]=399 unbinned) counts a long empty tail and over-inflates the knee. Reweighted by the
      # count-balance weights when that is on. All frozen across trials → computed once (by the first reference; later references reuse it).
      band = lp.w_smooth_band; conc = lp.w_smooth_conc
      nb = length(bw); ng = n_species
      use_cbal = lp.cbal_on && !isempty(cbal.w) && !isempty(cbal.coarse)
      coarse = use_cbal ? cbal.coarse : Int[]
      CW = use_cbal ? cbal.w : Matrix{Vector{FloatType}}(undef, 0, 0)
      wmax_full = sum(bw) - bw[end]
      sp_beta = wmax_full > 0 ? FloatType(1.0 / (band * conc * wmax_full)) : zero(FloatType)  # scalar fallback
      if softplus_cell_from !== nothing && size(softplus_cell_from.softplus_beta_cell) == (ng, ne)
        sp_beta_cell = softplus_cell_from.softplus_beta_cell
      else
        # K*[gsp,eco] = oldest reference-cohort bin across the cell's plots (last bin with incremental mass).
        Kstar = zeros(Int, ng, ne)
        for ((_, eid), yd) in spdf_plts, (_, gt) in yd, (sp_eco, rec) in gt.records
          gsp = Int(eco_species_ids[eid][sp_eco]); cdf = rec.sp_age_cdf; kk = 0
          @inbounds for k in 1:nb
            p = cdf[k] - (k == 1 ? zero(FloatType) : cdf[k-1])
            p > FloatType(1e-6) && (kk = k)
          end
          kk > Kstar[gsp, eid] && (Kstar[gsp, eid] = kk)
        end
        βcell = fill(sp_beta, ng, ne)
        for eco in 1:ne, gsp in 1:ng
          K = Kstar[gsp, eco]; K <= 1 && continue                # no reference / single bin → keep scalar
          w = use_cbal ? (@inbounds CW[gsp, eco]) : FloatType[]
          rew = use_cbal && !isempty(w)
          m = 0.0
          @inbounds for k in 1:K-1; m += (rew ? Float64(w[coarse[k]]) : 1.0) * Float64(bw[k]); end  # Σ bw over the ref stretch (drop saturating bin K*)
          m > 0 && (βcell[gsp, eco] = FloatType(1.0 / (band * conc * m)))
        end
        sp_beta_cell = βcell
      end
    end
  end
  # --- per-cell scales (+ RANKW) ---
  if lp.cell_norm
    W, A, R = _cell_scales(lp, search_tier, spdf_plts, t4_ref, eco_species_ids, n_species, cbal, ncoh, cell_plots, rankw, per_plot_mean, dom)
    return PU.LossScales(w_cell=W, agb_cell=A, rankw=R, softplus_beta=sp_beta, softplus_beta_cell=sp_beta_cell, cbal=cbal, ncoh=ncoh, dom=dom)
  end
  rk = rankw === nothing ? zeros(FloatType, 0, 0) : rankw
  (lp.w_normalize || lp.agb_normalize) || return PU.LossScales(rankw=rk, softplus_beta=sp_beta, softplus_beta_cell=sp_beta_cell, cbal=cbal, ncoh=ncoh, dom=dom)
  # --- global ratio-of-sums ---
  wsum = 0.0; asum = 0.0
  if search_tier == 4 && t4_ref !== nothing
    for eco in t4_ref, cycmat in eco, sp in 1:size(cycmat, 1)
      row = @view cycmat[sp, :]; tot = sum(row); asum += tot
      tot > 0 || continue
      acc = 0.0
      @inbounds for k in eachindex(bw); acc += row[k]; cdf = acc / tot; wsum += bw[k] * max(cdf, 1.0 - cdf); end
    end
  else
    for (_, yd) in spdf_plts, (_, gt) in yd, (_, rec) in gt.records
      asum += Float64(rec.sp_agb_sum)
      @inbounds for k in eachindex(bw); wsum += bw[k] * max(rec.sp_age_cdf[k], one(FloatType) - rec.sp_age_cdf[k]); end
    end
  end
  return PU.LossScales(w_global=wsum > 0 ? FloatType(wsum) : one(FloatType), agb_global=asum > 0 ? FloatType(asum) : one(FloatType),
    rankw=rk, softplus_beta=sp_beta, softplus_beta_cell=sp_beta_cell, cbal=cbal, ncoh=ncoh, dom=dom)
end

# Per-cell (eco×lu×sp) scales for cell_norm, computed from the reference (constant across evals). Each
# cell (eco e, global species gsp) gets W scale = Σ per-ref-max W1 and AGB scale = Σ observed AGB. Sim A
# (tier≠4) reads the per-plot spdf records; Sim B (tier 4) reads the aggregated t4_ref. RANKW[gsp,e] =
# (1/log(rank+1))/Σ over dominance-included cells, rank by observed AGB descending. `rankw === nothing` ⇒ this is the TRAIN reference
# and the rank is computed from it (Sim A: per-stratum plot counts as split size; Sim B-only: equal strata); otherwise the given train rank is used.
function _cell_scales(lp::PU.LossParams, search_tier::Int, spdf_plts, t4_ref, eco_species_ids, n_species::Int, cbal::PU.CbalTables,
    ncoh::Matrix{Int}, cell_plots, rankw::Union{Nothing,Matrix{FloatType}}, per_plot_mean::Bool, dom::Vector{Bool})
  bw = lp.age_bins.bin_widths
  ne = length(eco_species_ids)
  W = zeros(FloatType, n_species, ne); A = zeros(FloatType, n_species, ne)
  if search_tier == 4 && t4_ref !== nothing
    useB = lp.cbal_on && lp.cbal_mode !== :a && !isempty(cbal.w)   # reweight the ref-scale to match calculate_t4_loss (Sim B): direct cbw[k]
    cp = (per_plot_mean && cell_plots !== nothing) ? cell_plots : nothing
    for (e, eco) in enumerate(t4_ref)
      sp_map = eco_species_ids[e]
      for (cyc, cycmat) in enumerate(eco), sp in 1:size(cycmat, 1)
        gsp = sp_map[sp]
        # AGB scale gets the SAME per-plot-mean division as the loss (W scale is CDF-based → left as-is).
        pcinv = cp === nothing ? one(FloatType) : one(FloatType) / FloatType(max(1, cp[e][cyc]))
        row = @view cycmat[sp, :]; tot = sum(row); A[gsp, e] += tot * pcinv
        tot > 0 || continue
        cbw = useB ? (@inbounds cbal.w[gsp, e]) : FloatType[]
        rew = !isempty(cbw) && length(cbw) == length(bw)
        acc = zero(FloatType)
        @inbounds for k in eachindex(bw); acc += row[k]; cdf = acc / tot; W[gsp, e] += (rew ? cbw[k] : one(FloatType)) * bw[k] * max(cdf, one(FloatType) - cdf); end
      end
    end
    lp.w_scale_factor != one(FloatType) && (W .*= lp.w_scale_factor)
    R = rankw === nothing ? _rankw(lp, A, eco_species_ids; ncoh=ncoh, dom=dom) : rankw   # B-only: rank from B
  else
    useA = lp.cbal_on && lp.cbal_mode !== :b && !isempty(cbal.w)   # reweight the ref-scale to match _w1_sum (Sim A): coarse[k] lookup
    coarse = cbal.coarse
    for ((_, eco_id), yd) in spdf_plts, (_, gt) in yd, (sp_eco, rec) in gt.records
      gsp = eco_species_ids[eco_id][sp_eco]
      A[gsp, eco_id] += rec.sp_agb_sum
      cbw = useA ? (@inbounds cbal.w[gsp, eco_id]) : FloatType[]
      rew = !isempty(cbw)
      @inbounds for k in eachindex(bw); W[gsp, eco_id] += (rew ? cbw[coarse[k]] : one(FloatType)) * bw[k] * max(rec.sp_age_cdf[k], one(FloatType) - rec.sp_age_cdf[k]); end
    end
    lp.w_scale_factor != one(FloatType) && (W .*= lp.w_scale_factor)
    if rankw === nothing
      nplots = zeros(Int, ne)                                  # plots per stratum (train) → split-size weight
      for ((_, eco_id), _) in spdf_plts; nplots[eco_id] += 1; end
      R = _rankw(lp, A, eco_species_ids; split_size=nplots, ncoh=ncoh, dom=dom)   # rank from train A's per-plot ref
    else
      R = rankw
    end
  end
  return W, A, R
end

# Rank dominance-included species WITHIN EACH STRATUM (eco_id) by observed AGB (desc), weight
# 1/√(log1p(rank)) (√ softens the decay so rarer species aren't crushed); normalise each stratum to 1,
# then weight strata by the ORDER OF MAGNITUDE of their split size — RANKW[:,e] ∝ log10(1+n_plots[e]).
# This sits between the global ranking (largest split owns everything, small splits starved) and equal
# per-stratum (small noisy splits over-weighted): a 939-plot split outweighs a 34-plot one by ~log10,
# not 28× and not 1×. split_size = nothing → equal strata (fallback for B-only). Total Σ RANKW = 1.
# rankw_mode = :cbal_pct (with the per-cell train cohort counts `ncoh`) replaces the AGB rank by percentile-floored class-balanced weights.
function _rankw(lp::PU.LossParams, agb::Matrix{FloatType}, eco_species_ids; split_size::Union{Nothing,AbstractVector}=nothing,
    ncoh::Matrix{Int}=zeros(Int, 0, 0), dom::Vector{Bool}=Bool[])::Matrix{FloatType}
  R = zeros(FloatType, size(agb)); sw = zeros(Float64, length(eco_species_ids))
  cbal = lp.rankw_mode === :cbal_pct && !isempty(ncoh)        # percentile-floored class-balanced (by cohort count)
  for e in eachindex(eco_species_ids)
    ids = Int[]; for gsp in eco_species_ids[e]; _dom_included(dom, gsp) && push!(ids, gsp); end
    isempty(ids) && continue
    if cbal
      β = lp.rankw_beta; nc = ncoh; tot = sum(nc[gsp, e] for gsp in ids); totw = 0.0
      # a stratum without training cohorts carries no loss; its class-balanced weight would be (1-β)/(1-β^0) = Inf → Inf/Inf = NaN,
      # which poisons the aggregate (NaN·0). Weight 0 instead (B7).
      if tot == 0
        sw[e] = split_size === nothing ? 1.0 : log10(1.0 + Float64(split_size[e]))
        continue
      end
      qf = round(Int, 1 / lp.rankw_pct_floor)                              # grid steps (10/20/50 for 0.10/0.05/0.02); mirrors build_cbal_tables
      for gsp in ids
        pct = tot > 0 ? max(round(qf * nc[gsp, e] / tot) / qf, 1 / qf) : 1 / qf  # count-share on a 1/qf grid, floored at 1/qf (=rankw_pct_floor); qf=10 ≡ old round(10·share)/10 ⌊0.10⌋
        w = (1 - β) / (1 - β^(pct * tot)); R[gsp, e] = FloatType(w); totw += w
      end
      totw > 0 && (@views R[:, e] ./= totw)
    else
      cells = sort!([(agb[gsp, e], gsp) for gsp in ids]; by=c -> -c[1])   # descending AGB within this stratum → rank 1 = most common
      tot = zero(FloatType)
      for (rank, (_, gsp)) in enumerate(cells)
        w = FloatType(1.0 / sqrt(log1p(rank))); R[gsp, e] = w; tot += w
      end
      tot > 0 && (@views R[:, e] ./= tot)                                # within-stratum sums to 1
    end
    sw[e] = split_size === nothing ? 1.0 : log10(1.0 + Float64(split_size[e]))  # order of magnitude of split
  end
  tw = sum(sw)
  tw > 0 && for e in eachindex(eco_species_ids); @views R[:, e] .*= FloatType(sw[e] / tw); end  # Σ = 1
  return R
end

# Scales of the references of ONE split (train, or val) of a run: the primary reference's (`search_tier`, `spdf_plts`, `t4_ref`) and, when
# `dual_b` is given, Sim B's (returned inside a copy of the dual_b NamedTuple as `.scales`, which `fit_params` and `_mo_objectives` read).
# `train` is the TRAIN LossScales for a val split (its rankw and softplus cells are reused, so train and val weight cells the same way);
# `train === nothing` marks the train split itself, whose first reference defines RANKW. `seed` carries the run-wide `cbal`/`ncoh` tables.
function build_split_scales(lp::PU.LossParams, seed::PU.LossScales, search_tier::Int, spdf_plts, t4_ref, eco_species_ids, n_species::Int;
    dual_b=nothing, cell_plots=nothing, train::Union{Nothing,PU.LossScales}=nothing, simb::SimBOptions)
  cell_plots === nothing && dual_b !== nothing && (cell_plots = get(dual_b, :cell_plots, nothing))   # the split's own tier-4 counts (see fit_params)
  main = compute_loss_scales(lp, search_tier, spdf_plts, t4_ref, eco_species_ids, n_species; cell_plots=cell_plots, per_plot_mean=simb.t4_per_plot_mean, cbal=seed.cbal, ncoh=seed.ncoh, dom=seed.dom,
    rankw = train === nothing ? nothing : train.rankw, softplus_cell_from = train)
  if train !== nothing && dual_b === nothing && simb.t4_primary && !(search_tier == 4 && t4_ref !== nothing)
    # Tier-4-only run (t4_primary, no Sim B) whose val reference is evaluated as Sim A (tier 3): its per-cell objective has always been
    # normalised by the TRAIN tier-4 tables (the val Sim A reference never produced tier-4 tables of its own). Kept as is: the loss
    # kernels use the val reference's own fields, `_mo_objectives` reads the train cell scales.
    main = PU.LossScales(w_global=main.w_global, agb_global=main.agb_global, w_cell=train.w_cell, agb_cell=train.agb_cell, rankw=main.rankw,
      softplus_beta=main.softplus_beta, softplus_beta_cell=main.softplus_beta_cell, cbal=main.cbal, ncoh=main.ncoh, dom=main.dom)
  end
  dual_b === nothing && return main, nothing
  bs = compute_loss_scales(dual_b.loss_params, dual_b.tier, dual_b.spdf_plts, dual_b.t4_ref, eco_species_ids, n_species; cell_plots=dual_b.cell_plots, per_plot_mean=simb.t4_per_plot_mean,
    cbal=seed.cbal, ncoh=seed.ncoh, dom=seed.dom, rankw=main.rankw, softplus_cell_from=(train === nothing ? main : train))
  return main, merge(dual_b, (scales=bs,))
end

# The two scale sets `_mo_objectives` indexes by group: group 0 = the primary reference — Sim A, except when Sim B is primary (SimBOptions.t4_primary, carried by dual_b: a
# dual_mode b run, whose only block is Sim B, or a tier-4-only run whose single reference already carries the tier-4 scales); group 1 = Sim B.
@inline function _group_scales(scales::PU.LossScales, dual_b)
  # without a Sim B (tier 5 stacks a tier-4 half after the tier-3 half) group 1 has NO scales of its own: the empty LossScales makes _mo_objectives
  # refuse (B11) instead of normalising tier-4 losses with another reference's scales
  dual_b === nothing && return scales, PU.LossScales()
  bsc = dual_b.scales::PU.LossScales
  return (dual_b.t4_primary ? bsc : scales), bsc
end
