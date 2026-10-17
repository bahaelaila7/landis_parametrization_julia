struct AgeBins
  bins_idx::Vector{Int}
  bin_widths::Vector{FloatType}
  last_bin_open::Bool
  function AgeBins(; bins_idx::Vector{Int}, last_bin_open::Bool)
    new(bins_idx,
      get_bin_widths(; age_bins=bins_idx, last_bin_open=last_bin_open),
      last_bin_open)
  end
end

"""
    LossParams

Everything that defines the loss and is constant for a run: the age bins, the smoothing window, the AGB weight `lambda`, and the loss
OPTIONS (set once from the config by `Pan.parametrize`). Field defaults equal the former module-level `Ref` defaults, so a hand-built
`LossParams(age_bins=…, smoothing_weights=…)` is the plain unnormalised loss. The per-reference, data-derived normalisation lives in
[`LossScales`](@ref) (one per reference set), never here.
"""
Base.@kwdef struct LossParams
  age_bins::AgeBins
  smoothing_weights::Vector{FloatType}
  lambda::FloatType = FloatType(1.0f0)   # weight of the AGB-level (sqrt-difference) term vs the shape W1
  EPS::FloatType = FloatType(1.0f-7)
  # Global weight on the Wasserstein (shape) term in the optimized scalar (0 keeps only the L2/AGB-level term).
  loss_alpha::FloatType = one(FloatType)
  # AGB-level loss form: agb_hinge=true replaces the gentle sqrt-difference AGB term with a HINGE: a tolerance band below which AGB
  # differences are not penalized (threshold flat, or `pct` of the observed AGB clamped to [pct_min, pct_max] when pct > 0). The hinge is
  # softplus-smoothed (sharpness agb_hinge_beta, <=0 ⇒ hard ReLU) and raised to agb_hinge_p AFTER the ÷scale (squared ratio, comparable to W).
  agb_hinge::Bool = false
  agb_hinge_threshold::FloatType = FloatType(10.0)
  agb_hinge_pct::FloatType = zero(FloatType)
  agb_hinge_pct_min::FloatType = zero(FloatType)
  agb_hinge_pct_max::FloatType = FloatType(Inf)
  agb_hinge_p::FloatType = FloatType(2.0)
  agb_hinge_beta::FloatType = FloatType(1.0)
  # GLOBAL ratio-of-sums normalisation (divide by Σ observed AGB / Σ per-reference-max W1 of the reference: see LossScales.agb_global/w_global).
  agb_normalize::Bool = false
  w_normalize::Bool = false
  # PER-CELL (eco×lu×sp) normalisation: the per-cell loss keeps the RAW shape/level error (softplus-W and hinge-AGB kept); the per-cell ÷scale,
  # the power and the RANKW rescale are applied at aggregation (`Pan._mo_objectives`). Overrides the global normalisation.
  cell_norm::Bool = false
  w_scale_factor::FloatType = FloatType(1.0)   # multiplies the W scales (<1 shrinks the divisor → amplifies ΣW vs AGB)
  # Softplus smoothing of the per-term W1. w_smooth_auto_knee=false: fixed β = w_smooth_beta; true: β derived per cell from band·conc·wmax.
  w_softplus::Bool = false
  w_smooth_band::FloatType = FloatType(0.05)
  w_smooth_conc::FloatType = FloatType(0.5)
  w_smooth_beta::FloatType = FloatType(1.0)
  w_smooth_auto_knee::Bool = false
  w_p::FloatType = FloatType(1.0)              # exponent on the (normalised) W term
  # W "benefit of the doubt" (Sim A only): forgive each sim cohort's binned biomass toward the observed by a pct/min/max band before the CDF.
  w_hinge::Bool = false
  w_hinge_pct::FloatType = FloatType(0.01)
  w_hinge_pct_min::FloatType = FloatType(2.0)
  w_hinge_pct_max::FloatType = FloatType(5.0)
  # Median-pivoted piecewise on the aggregate ΣW, ΣAGB (get_total_loss): sqrt(x/m) below the pivot m, (x/m)^2 above.
  loss_piecewise::Bool = false
  w_pivot::FloatType = FloatType(1.0)
  agb_pivot::FloatType = FloatType(1.0)
  # Per-stratum RANKW weights: :rank (1/√ln(rank) by AGB) or :cbal_pct (percentile-floored class-balanced by cohort count).
  rankw_mode::Symbol = :rank
  rankw_beta::Float64 = 0.999
  rankw_pct_floor::Float64 = 0.10
  # COUNT-BALANCE reweight of the per-agebin W1 (counteracts survivorship): cbal_mode ∈ :a (Sim A) / :b (Sim B) / :both.
  cbal_on::Bool = false
  cbal_mode::Symbol = :both
  cbal_beta::Float64 = 0.99
  cbal_pct_floor::Float64 = 0.10
end

"""
    CbalTables

Count-balance weights built once from the TRAIN reference (`Pan.build_cbal_tables`) and shared by every reference of the run:
`w[gsp,eco]` = weight per COARSE age bin, `coarse[k]` = the coarse bin of Sim-A per-year bin k.
"""
struct CbalTables
  w::Matrix{Vector{FloatType}}
  coarse::Vector{Int}
end
const NO_CBAL = CbalTables(Matrix{Vector{FloatType}}(undef, 0, 0), Int[])

"""
    LossScales

The normalisation derived from ONE reference set (train Sim A, train Sim B, val Sim A, …), computed by `Pan.compute_loss_scales` and
passed explicitly to `fit_params` / `_mo_objectives`. Defaults are neutral (no normalisation), so a hand-built `LossScales()` is valid.
`rankw`, `cbal` and `dom` are the run-wide TRAIN tables, shared (same object) by every reference including val.
"""
Base.@kwdef struct LossScales
  w_global::FloatType = one(FloatType)           # Σ per-reference-max W1 (agb/w_normalize)
  agb_global::FloatType = one(FloatType)         # Σ observed AGB
  w_cell::Matrix{FloatType} = zeros(FloatType, 0, 0)    # [gsp,eco] cell-norm W scale (× w_scale_factor)
  agb_cell::Matrix{FloatType} = zeros(FloatType, 0, 0)  # [gsp,eco] cell-norm AGB scale
  rankw::Matrix{FloatType} = zeros(FloatType, 0, 0)     # [gsp,eco] stratum×species weights, Σ=1 (frozen from the train reference)
  softplus_beta::FloatType = zero(FloatType)
  softplus_beta_cell::Matrix{FloatType} = zeros(FloatType, 0, 0)  # [gsp,eco] reweight-aware β (auto-knee); empty ⇒ use the scalar
  cbal::CbalTables = NO_CBAL
  ncoh::Matrix{Int} = zeros(Int, 0, 0)                   # per-(species,eco) TRAIN cohort counts (input of rankw_mode = :cbal_pct)
  dom::Vector{Bool} = Bool[]                             # [gsp] species that drive DOMINANCE (MO objectives, RANKW); empty ⇒ all species
end

Base.@kwdef struct SPDFRecord
  sp_agb_sum::FloatType
  sp_age_cdf::Vector{FloatType}
  sp_age_agb::Vector{FloatType}   # binned ABSOLUTE observed biomass per age bin — the per-cohort AGB reference
end
Base.@kwdef struct SPDFGroundTruth
  keys::BitVector
  records::Dict{UIntType,SPDFRecord}
end
Base.@kwdef struct SiteLoss
  sp_w_loss::Vector{FloatType}
  sp_agb_loss::Vector{FloatType}
  site_agb_loss::FloatType
  num_sites::Int = 1
  num_obs::Int = 1  # (site, measurement_year) pairs regressed against
end
@inline function Base.:+(loss1::SiteLoss, loss2::SiteLoss)
  SiteLoss(sp_w_loss=loss1.sp_w_loss .+ loss2.sp_w_loss,
    sp_agb_loss=loss1.sp_agb_loss .+ loss2.sp_agb_loss,
    site_agb_loss=loss1.site_agb_loss + loss2.site_agb_loss,
    num_sites=loss1.num_sites + loss2.num_sites,
    num_obs=loss1.num_obs + loss2.num_obs)
end

function Base.sum(losses::AbstractVector{SiteLoss})
  isempty(losses) && error("sum of empty SiteLoss vector")
  sp_w = copy(losses[1].sp_w_loss)
  sp_agb = copy(losses[1].sp_agb_loss)
  site_agb = losses[1].site_agb_loss
  n_sites = losses[1].num_sites
  n_obs = losses[1].num_obs
  @inbounds for k in 2:length(losses)
    sp_w .+= losses[k].sp_w_loss
    sp_agb .+= losses[k].sp_agb_loss
    site_agb += losses[k].site_agb_loss
    n_sites += losses[k].num_sites
    n_obs += losses[k].num_obs
  end
  SiteLoss(sp_w_loss=sp_w, sp_agb_loss=sp_agb, site_agb_loss=site_agb, num_sites=n_sites, num_obs=n_obs)
end
