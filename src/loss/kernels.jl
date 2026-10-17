# ── Loss kernels. They read the run's options from `lp::LossParams` and the reference's normalisation from `sc::LossScales`; there is no
# module-level loss state. All are `@inline`, branch on immutable fields, and allocate nothing.

# Per-observation AGB hinge tolerance for an observed AGB level `obs`: pct > 0 ⇒ clamp(obs*pct, pct_min, pct_max), else the flat threshold.
@inline _agb_hinge_thresh(lp::LossParams, obs::FloatType)::FloatType =
  lp.agb_hinge_pct > zero(FloatType) ?
    clamp(obs * lp.agb_hinge_pct, lp.agb_hinge_pct_min, lp.agb_hinge_pct_max) :
    lp.agb_hinge_threshold
# Exponent on the NORMALIZED AGB relative error (applied AFTER ÷scale, so it is (excess/scale)^p — a squared RATIO that stays comparable to
# W's (W1/scale)^p; applying it to the raw excess pre-norm would reintroduce the scale gap).
@inline _agb_pow(lp::LossParams, x::FloatType)::FloatType =
  lp.agb_hinge_p == FloatType(2) ? x * x : lp.agb_hinge_p == one(FloatType) ? x : x^lp.agb_hinge_p
# GLOBAL ratio-of-sums normalization: divide the ABSOLUTE per-term error by a single constant of the reference (Σ observed AGB; Σ per-reference-max W1).
@inline _agb_norm(lp::LossParams, sc::LossScales, term::FloatType)::FloatType =
  (lp.agb_normalize && sc.agb_global > zero(FloatType)) ? term / sc.agb_global : term
@inline _w_norm(lp::LossParams, sc::LossScales, term::FloatType)::FloatType =
  (lp.w_normalize && sc.w_global > zero(FloatType)) ? term / sc.w_global : term
# Softplus smoothing of the per-term W1 (de-weights small shape errors smoothly), pedestal-subtracted so a perfect match (s=0) → 0. β is the
# reweight-aware per-cell one when the reference has it (`sc.softplus_beta_cell`), else the scalar.
@inline function _w_smooth(lp::LossParams, sc::LossScales, s::FloatType, gsp::Int, eco::Int)::FloatType
  lp.w_softplus || return s
  βc = sc.softplus_beta_cell
  β = !isempty(βc) ? (@inbounds βc[gsp, eco]) : sc.softplus_beta
  β > zero(FloatType) || return s
  s + log1p(exp(-β * s)) / β - log(FloatType(2)) / β   # softplus_β(s) − log2/β  (0 at s=0, ≤ s)
end
# Exponent on the (normalized) W term — square it for aggression (w_p=2 ⇒ (W1/scale)²). Default 1 (L1).
@inline _w_pow(lp::LossParams, x::FloatType)::FloatType = lp.w_p == FloatType(2) ? x * x : (lp.w_p == one(FloatType) ? x : x^lp.w_p)
# W "benefit of the doubt" tolerance for an observed binned biomass `obs`.
@inline _w_hinge_thresh(lp::LossParams, obs::FloatType)::FloatType = clamp(obs * lp.w_hinge_pct, lp.w_hinge_pct_min, lp.w_hinge_pct_max)
# Per-term finishers: with cell_norm the per-cell loss stores the RAW (softplus-W / hinge-AGB) error and the ÷scale, power and rank apply at
# aggregation (`Pan._mo_objectives`); otherwise the global ÷scale and the power are applied here.
@inline _w_finish(lp::LossParams, sc::LossScales, s::FloatType, gsp::Int, eco::Int)::FloatType =
  lp.cell_norm ? _w_smooth(lp, sc, s, gsp, eco) : _w_pow(lp, _w_norm(lp, sc, _w_smooth(lp, sc, s, gsp, eco)))
@inline _agb_finish(lp::LossParams, sc::LossScales, h::FloatType)::FloatType = lp.cell_norm ? h : _agb_pow(lp, _agb_norm(lp, sc, h))

# Median-pivoted piecewise (applied to the AGGREGATE ΣW, ΣAGB in get_total_loss): f(x) = sqrt(x/m) if x<m else (x/m)². Pivot m = per-objective
# median (w_pivot/agb_pivot). Lifts better-than-typical (sqrt, avoids suppressing small) and aggressively penalizes worse-than-typical (square).
@inline function _piecewise(x::FloatType, m::FloatType)::FloatType
  m <= zero(FloatType) && return x
  r = x / m
  r < one(FloatType) ? sqrt(r) : r * r
end
# Softplus (smooth ReLU) replacing the hard hinge max(0,x), so the loss surface isn't sharp at the band edge: max(0, |sim-obs| - thresh(obs))^p
# becomes softplus(|sim-obs| - thresh(obs))^p. β is a CONSTANT sharpness (agb_hinge_beta); β ≤ 0 ⇒ the hard ReLU. Stable form.
@inline function _hinge_relu(lp::LossParams, x::FloatType)::FloatType
  β = lp.agb_hinge_beta
  β <= zero(FloatType) && return max(zero(FloatType), x)
  return max(zero(FloatType), x) + log1p(exp(-β * abs(x))) / β
end

@inline function get_total_loss(loss::SiteLoss, lp::LossParams, beta::FloatType=FloatType(1.0f0))::FloatType
  alpha = lp.loss_alpha
  # sp_w_loss = age-distribution SHAPE (W1, AGB-invariant); sp_agb_loss = AGB LEVEL.
  # When normalizing, each sp_* term is already ÷ its global scale → the sum IS the intensive ratio-of-sums
  # (Σabs/Σref), so skip the /num_obs averaging; otherwise average per measurement as before.
  if lp.loss_piecewise   # median-pivoted piecewise on the aggregates → W & AGB centered at 1, comparable
    return alpha * _piecewise(sum(loss.sp_w_loss), lp.w_pivot) + beta * _piecewise(sum(loss.sp_agb_loss), lp.agb_pivot)
  end
  denom = (lp.w_normalize || lp.agb_normalize) ? one(FloatType) : FloatType(loss.num_obs)
  return (alpha * sum(loss.sp_w_loss) + beta * sum(loss.sp_agb_loss)) / denom
end
