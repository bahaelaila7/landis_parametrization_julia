using ..PanCore
export MutableParam, SpeciesSampler, EcoSampler, GlobalSampler, EcoSpeciesSampler, GradientApplier, ScalarApplier, IndexApplier, NestedIndexApplier, ParamDists, SamplingContext, LossParams, SiteLoss, AgeBins, get_smoothing_window, calculate_site_loss2, skipundef, MutationType, sobol_samples, saltelli_design, wasserstein1d, find_age_bin, build_slots, u_to_params, params_to_u

import Setfield
import Sobol
import Random
import ImageFiltering
import Dates
import JSON3
using DataFrames
import Distributions as Dists

struct MutableParam{S,T}
  name::Symbol
  dist::Dists.Distribution
  bounds::Tuple{Union{Nothing,T},Union{Nothing,T}}
  sigma::T
  type::Type
  sampler::S             # how to pick the target element
  applier::Any           # how to write the sampled value back
  quantum::FloatType     # snap sampled values to a multiple of this; 0 = no quantization (continuous)
  group::Symbol          # block-diagonal CMA-ES group of (assumed-)correlated params (see build_groups)
end
# Back-compat 7-arg form (no quantization). Pass `quantum=…` to put a parameter on a coarse grid
# (e.g. B_MAX_SPP by 100). The quantum is honored project-wide by every value-producing path:
# mutate_params (LBSA/MOLBSA), u_to_params (CMA-ES/MOCMAES) and sobol_samples. `group` tags the
# parameter's block-diagonal CMA-ES group (default :default ⇒ all params in one block = plain CMA-ES).
MutableParam(name::Symbol, dist::Dists.Distribution, bounds::Tuple, sigma::T, type::Type, sampler::S, applier; quantum::Real=0, group::Symbol=:default) where {S,T} =
  MutableParam{S,T}(name, dist, bounds, sigma, type, sampler, applier, FloatType(quantum), group)

# Partition the u-coordinates (one per `slots` entry) into BLOCK-DIAGONAL CMA-ES groups by each
# parameter's `group`. Groups whose symbol is in `per_eco_groups` are FURTHER split per ecoregion (a
# separate block per eco) — so eco-scoped params (Eco/EcoSpecies) yield one block per ecoregion, giving
# the "2 global + #ecoregion matrices" layout. Returns a Vector of u-index vectors (a partition of 1:d),
# in first-seen order. A single group (the default) ⇒ one block = plain CMA-ES.
function build_groups(param_dists, slots::Vector{Tuple{Int,Any}}, per_eco_groups)
  keyed = Dict{Tuple{Symbol,Int},Vector{Int}}()
  order = Tuple{Symbol,Int}[]
  for (dim, (pi, target)) in enumerate(slots)
    g = param_dists.params[pi].group
    eco = (g in per_eco_groups) ? (target isa Tuple ? Int(target[1]) : (target isa Integer ? Int(target) : 0)) : 0
    key = (g, eco)
    if !haskey(keyed, key)
      keyed[key] = Int[]; push!(order, key)
    end
    push!(keyed[key], dim)
  end
  return Vector{Int}[keyed[k] for k in order]
end

# Snap `v` to the nearest multiple of `q` (q == 0 → unchanged), returning the parameter's type.
@inline _quantize(v, q::FloatType, ::Type{T}) where {T} = q > zero(FloatType) ? T(round(Float64(v) / q) * q) : T(v)
struct ParamDists{T} # subtyping here to make different plugins have different ParamDists
  params::Vector{MutableParam}
  weights_cumsum::Vector{Float64}
end

@enum MutationType GaussianMutation RandomMutation BothMutations

abstract type AbstractSampler end
struct GlobalSampler <: AbstractSampler end       # scalar field, no index needed
struct SpeciesSampler <: AbstractSampler end      # pick a random species
struct EcoSampler <: AbstractSampler end          # pick a random ecoregion
struct EcoSpeciesSampler <: AbstractSampler end   # pick a random (eco, species) pair

abstract type AbstractApplier end
struct ScalarApplier <: AbstractApplier end                          # field[] = val
struct IndexApplier <: AbstractApplier end                           # field[i] = val
struct NestedIndexApplier <: AbstractApplier end                     # field[i][j] = val
struct GradientApplier <: AbstractApplier
  step::FloatType
end         # field[i] = [val, val+step, ...]

# Loss-weighted sampling context: sp_w_loss[global_species_id] drives proportional selection.
struct SamplingContext
  sp_w_loss::Vector{FloatType}
end

# Returns a 1-based index sampled proportional to weights; falls back to uniform when all zero.
function _weighted_sample(weights::AbstractVector{FloatType}, rng::Random.AbstractRNG)::Int
  n = length(weights)
  total = sum(weights)
  total <= zero(FloatType) && return rand(rng, 1:n)
  r = rand(rng, FloatType) * total
  cumw = zero(FloatType)
  for i in 1:n
    cumw += weights[i]
    cumw >= r && return i
  end
  return n
end

sample_target(s::GlobalSampler, p, rng::Random.AbstractRNG) = nothing
sample_target(s::SpeciesSampler, p, rng::Random.AbstractRNG) = rand(rng, 1:length(p.SPECIES_LIST))
sample_target(s::EcoSampler, p, rng::Random.AbstractRNG) = rand(rng, 1:length(p.ECO_LIST))
function sample_target(s::EcoSpeciesSampler, p, rng::Random.AbstractRNG)
  eco_id = rand(rng, 1:length(p.ECO_SPECIES_IDS))
  species_id = rand(rng, 1:length(p.ECO_SPECIES_IDS[eco_id]))
  (eco_id, species_id)
end

# Context-aware overloads: species/eco-species selection proportional to sp_w_loss.
sample_target(::GlobalSampler, p, rng::Random.AbstractRNG, ::SamplingContext) = nothing
sample_target(::EcoSampler, p, rng::Random.AbstractRNG, ::SamplingContext) = rand(rng, 1:length(p.ECO_LIST))

function sample_target(::SpeciesSampler, p, rng::Random.AbstractRNG, ctx::SamplingContext)
  _weighted_sample(ctx.sp_w_loss, rng)
end

function sample_target(::EcoSpeciesSampler, p, rng::Random.AbstractRNG, ctx::SamplingContext)
  # Stage 1: sample eco proportional to summed loss of its species.
  eco_weights = FloatType[
    sum(ctx.sp_w_loss[Int(gsp)] for gsp in p.ECO_SPECIES_IDS[eco_id])
    for eco_id in 1:length(p.ECO_SPECIES_IDS)
  ]
  eco_id = _weighted_sample(eco_weights, rng)
  # Stage 2: sample local species index proportional to its loss.
  sp_weights = FloatType[ctx.sp_w_loss[Int(gsp)] for gsp in p.ECO_SPECIES_IDS[eco_id]]
  sp_id = _weighted_sample(sp_weights, rng)
  (eco_id, sp_id)
end

get_field_val(::ScalarApplier, field, idx) = field
get_field_val(::IndexApplier, field, idx) = field[idx]
function get_field_val(::NestedIndexApplier, field, idx)
  field[idx[1]][idx[2]]
end
function get_field_val(a::GradientApplier, field, idx)
  field[idx][1]
end
apply_mutation(::ScalarApplier, field, idx, val) = val
apply_mutation(::IndexApplier, field, idx, val) = setindex!(copy(field), val, idx)
function apply_mutation(::NestedIndexApplier, field, idx, val)
  nv = copy(field)
  nv[idx[1]] = copy(nv[idx[1]])
  nv[idx[1]][idx[2]] = val
  nv
end
function apply_mutation(a::GradientApplier, field, idx, val)
  nv = copy(field)
  nv[idx] = [val + k * a.step for k in 0:length(nv[idx])-1]
  nv
end


function mutate_params(p::T, param_dists::ParamDists{T}; rng::Random.AbstractRNG, mutation_mode::MutationType=RandomMutation, ctx::Union{Nothing,SamplingContext}=nothing, dynamic_cumsum::Union{Nothing,Vector{Float64}}=nothing) where {T}
  s = rand(rng, Float64)
  wcumsum = isnothing(dynamic_cumsum) ? param_dists.weights_cumsum : dynamic_cumsum
  param_idx = something(findlast(wcumsum .<= s), 1)
  param = param_dists.params[param_idx]
  _mutation_mode = mutation_mode != BothMutations ? mutation_mode : (rand(rng) > 0.5 ? GaussianMutation : RandomMutation)

  idx = isnothing(ctx) ? sample_target(param.sampler, p, rng) :
        sample_target(param.sampler, p, rng, ctx)
  field = getproperty(p, param.name)
  val = begin
    cur_val = get_field_val(param.applier, field, idx)
    r = cur_val
    min_, max_ = param.bounds
    if (bf = BMAX_FLOOR[]) !== nothing && param.name === :B_MAX_SPP   # per-(eco,species) data floor (LBSA/SA path)
      min_ = FloatType(get(bf, idx, 12000.0))
    end
    while r == cur_val
      r = if _mutation_mode == GaussianMutation
        sigma = param.sigma
        #println(param, r, cur_val)
        begin
          if param.type == UIntType
            if !isnothing(min_) && cur_val == min_
              cur_val + param.sigma
            elseif !isnothing(max_) && cur_val == max_
              cur_val - param.sigma
            else
              rand(rng) > 0.5 ? cur_val + param.sigma : cur_val - param.sigma
            end
          else
            rand(rng, Dists.Normal(FloatType(cur_val), sigma))
          end
        end |> param.type
      else
        rand(rng, param.dist) |> param.type
      end
      r = if !isnothing(min_) && FloatType(r) <= FloatType(min_)
        FloatType(min_)
      elseif !isnothing(max_) && FloatType(r) >= FloatType(max_)
        FloatType(max_)
      else
        r
      end |> param.type
      r = _quantize(r, param.quantum, param.type)   # snap to the param's grid (e.g. B_MAX by 100)
    end
    r
  end

  new_field = apply_mutation(param.applier, field, idx, val)
  #println(field)
  #println(typeof(val))
  #println(new_field)

  Setfield.@set(p.$(param.name) = new_field), param_idx
end



@inline skipundef(xs::AbstractArray) = xs[filter(i -> isassigned(xs, i), eachindex(xs))]

BandType = Union{String,Real}
ValType = Union{String,Real}
AttrDict = Dict{String,BandType}
AttrTable = Dict{BandType,AttrDict}
#using Profile, ProfileSVG
#using ProgressBars

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

Base.@kwdef struct LossParams
  age_bins::AgeBins
  smoothing_weights::Vector{FloatType}
  lambda::FloatType = FloatType(1.0f0)   # weight of the AGB-level (sqrt-difference) term vs the shape W1
  EPS::FloatType = FloatType(1.0f-7)
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
# Global weight on the Wasserstein (shape) term in the optimized scalar. Set to 0 to keep only
# the L2/AGB-level term (sp_w_loss is still computed — it drives loss-weighted sampling and the
# per-species breakdown). Wired from `loss_alpha` (yaml / parametrize).
const LOSS_ALPHA = Ref{FloatType}(one(FloatType))
# AGB-level loss form. AGB_HINGE[]=true replaces the gentle sqrt-difference AGB term with a HINGE-L1:
# loss = lambda * max(0, |sim_agb - obs_agb| - threshold) — a tolerance band below which AGB differences
# are not penalized; threshold 0 ⇒ plain L1. Wired from agb_hinge / agb_hinge_threshold.
const AGB_HINGE = Ref{Bool}(false)
const AGB_HINGE_THRESHOLD = Ref{FloatType}(FloatType(10.0))
# Percentage tolerance: when AGB_HINGE_PCT[] > 0 the band is a fraction of the OBSERVED AGB,
# clamped to [AGB_HINGE_PCT_MIN, AGB_HINGE_PCT_MAX] — e.g. pct=0.04, min=10, max=200 ⇒
# threshold = clamp(obs*0.04, 10, 200). pct ≤ 0 ⇒ fall back to the flat AGB_HINGE_THRESHOLD.
const AGB_HINGE_PCT = Ref{FloatType}(zero(FloatType))
const AGB_HINGE_PCT_MIN = Ref{FloatType}(zero(FloatType))
const AGB_HINGE_PCT_MAX = Ref{FloatType}(FloatType(Inf))
# Per-observation hinge tolerance for an observed AGB level `obs`.
@inline _agb_hinge_thresh(obs::FloatType)::FloatType =
  AGB_HINGE_PCT[] > zero(FloatType) ?
    clamp(obs * AGB_HINGE_PCT[], AGB_HINGE_PCT_MIN[], AGB_HINGE_PCT_MAX[]) :
    AGB_HINGE_THRESHOLD[]
# Hinge penalty exponent p: pen(h) = h^p, applied to the (softplus) hinge excess. Default p=2 (L2/MSE);
# p=1 ⇒ L1. Wired from agb_hinge_p. (AGB_HINGE_L2 is legacy, kept only for back-compat default mapping.)
const AGB_HINGE_L2 = Ref{Bool}(false)
const AGB_HINGE_P = Ref{FloatType}(FloatType(2.0))
# Exponent on the NORMALIZED AGB relative error (applied AFTER ÷scale, so it's (excess/scale)^p — a
# squared RATIO that stays comparable to W's (W1/scale)^p; applying it to the raw excess pre-norm would
# reintroduce the scale gap). _agb_hinge_pen kept as an alias for back-compat.
@inline _agb_pow(x::FloatType)::FloatType =
  AGB_HINGE_P[] == FloatType(2) ? x * x : AGB_HINGE_P[] == one(FloatType) ? x : x^AGB_HINGE_P[]
@inline _agb_hinge_pen(h::FloatType)::FloatType = h   # identity now (power moved outside _agb_norm)
# GLOBAL ratio-of-sums normalization: divide the ABSOLUTE per-term error by a single global constant
# (set once per evaluation from the reference). Because the divisor is a fixed total, every relative
# magnitude is preserved → cohort SIZE is respected (a 100× bigger cohort contributes 100× more error),
# while the objective lands at O(1) comparable to W. NOT per-term (that neutralizes size) and NOT min-max
# (that amplifies the tame measure's noise). AGB_SCALE = Σ observed AGB; W_SCALE = Σ per-reference-max W1.
const AGB_NORMALIZE = Ref{Bool}(false)
const AGB_SCALE = Ref{FloatType}(FloatType(1.0))
@inline _agb_norm(term::FloatType)::FloatType =
  (AGB_NORMALIZE[] && AGB_SCALE[] > zero(FloatType)) ? term / AGB_SCALE[] : term
const W_NORMALIZE = Ref{Bool}(false)
const W_SCALE = Ref{FloatType}(FloatType(1.0))
@inline _w_norm(term::FloatType)::FloatType =
  (W_NORMALIZE[] && W_SCALE[] > zero(FloatType)) ? term / W_SCALE[] : term
# Softplus smoothing of the per-term W1 (de-weights small shape errors smoothly). β set per run from the
# bins (W_SOFTPLUS_BETA, computed in _set_loss_scales!): knee ≈ W_SMOOTH_BAND·W_SMOOTH_CONC·(Σbw−bw_N).
# Pedestal-subtracted so a perfect match (s=0) → 0; for s>0, slope ramps 0.5→1 over the knee. Applied to
# RAW W1 before _w_norm (constant ÷ after preserves the shape, same as the AGB softplus).
# REWEIGHT-AWARE: when the count-balance reweight is on, the per-term W1 that reaches here is the REWEIGHTED
# sum Σ CBAL_W·bw·|ΔF|, whose theoretical max is Σ CBAL_W·bw (per species×eco cell), NOT Σbw. So the knee
# is calibrated per-cell via W_SOFTPLUS_BETA_CELL[gsp,eco]; the scalar W_SOFTPLUS_BETA is the fallback
# (no reweight, or cells with no reference). Both computed in _set_loss_scales!.
const W_SOFTPLUS = Ref{Bool}(false)
const W_SOFTPLUS_BETA = Ref{FloatType}(zero(FloatType))
const W_SOFTPLUS_BETA_CELL = Ref{Matrix{FloatType}}(zeros(FloatType, 0, 0))  # [gsp,eco] reweight-aware β; empty ⇒ use scalar
const W_SMOOTH_BAND = Ref{FloatType}(FloatType(0.05))   # smoothing region = this fraction of the realistic range (auto-knee only)
const W_SMOOTH_CONC = Ref{FloatType}(FloatType(0.5))    # realistic range = this fraction of the theoretical max (auto-knee only)
const W_SMOOTH_BETA = Ref{FloatType}(FloatType(1.0))    # FIXED β for the W softplus when auto-knee is OFF (the default)
const W_SMOOTH_AUTO_KNEE = Ref{Bool}(false)             # false (default): use W_SMOOTH_BETA; true: derive β from band·conc·wmax (reweight-aware per-cell). AGB hinge is unaffected either way.
@inline function _w_smooth(s::FloatType, gsp::Int, eco::Int)::FloatType
  W_SOFTPLUS[] || return s
  βc = W_SOFTPLUS_BETA_CELL[]
  β = !isempty(βc) ? (@inbounds βc[gsp, eco]) : W_SOFTPLUS_BETA[]
  β > zero(FloatType) || return s
  s + log1p(exp(-β * s)) / β - log(FloatType(2)) / β   # softplus_β(s) − log2/β  (0 at s=0, ≤ s)
end
# Exponent on the (normalized) W term — square it for aggression (W_P=2 ⇒ (W1/scale)²). Default 1 (L1).
const W_P = Ref{FloatType}(FloatType(1.0))
@inline _w_pow(x::FloatType)::FloatType = W_P[] == FloatType(2) ? x * x : (W_P[] == one(FloatType) ? x : x^W_P[])

# W "benefit of the doubt" (Sim A only): BEFORE the age CDF, forgive each sim cohort's binned biomass TOWARD
# the observed by up to _w_hinge_thresh(obs_i) g/m² — i.e. sim_i -= clamp(sim_i − obs_i, −t_i, +t_i), so inside
# the band sim_i := obs_i and small per-cohort biomass errors don't perturb the age SHAPE. Threshold has the
# same pct/min/max form as the AGB hinge. W_HINGE off ⇒ plain CDF (unchanged).
const W_HINGE = Ref{Bool}(false)
const W_HINGE_PCT = Ref{FloatType}(FloatType(0.01))       # default 1% of obs cohort biomass
const W_HINGE_PCT_MIN = Ref{FloatType}(FloatType(2.0))    # floored at 2 g/m²
const W_HINGE_PCT_MAX = Ref{FloatType}(FloatType(5.0))    # capped at 5 g/m²
@inline _w_hinge_thresh(obs::FloatType)::FloatType = clamp(obs * W_HINGE_PCT[], W_HINGE_PCT_MIN[], W_HINGE_PCT_MAX[])

# PER-CELL (eco×lu×sp) normalization. When on, the per-site/per-cell loss stores the RAW shape/level
# error (softplus-W and hinge-AGB kept — they smooth per observation — but NOT the global ÷scale or the
# power); the per-cell ÷scale, the power, and a rank rescale are applied at aggregation (_mo_objectives).
# Each cell is normalized by its OWN reference magnitude (W: Σ per-ref-max W1; AGB: Σ observed AGB), so
# every (eco,lu,sp) cell is judged on its own terms — but that makes a tiny catch-all cell as influential
# as a dominant one, so each cell is then rescaled by RANKW = 1/log(rank+1) (rank by observed AGB across
# all cells, normalized Σ=1): errors from more common species count more, noisy _GRP/_H/_S less.
# Tables are [gsp, eco]; A and B keep separate scales (different references); RANKW is shared.
const CELL_NORM   = Ref{Bool}(false)
# Set true after the first (train) eval populates the RANK. The scales recompute per split (intensive
# ratio-of-sums, so train/val share a scale anyway), but the rank is frozen from TRAIN so train and val
# weight the same eco×lu×sp cells the same way (val is otherwise free to re-rank on its thinner split).
const CELL_NORM_FREEZE = Ref{Bool}(false)
# When true, fit_params SKIPS its per-eval _set_loss_scales! (the caller has set the reference-derived loss scales
# once, serially). Candidate-parallel batches set this so concurrent evals don't race on the W_SCALE/AGB_SCALE/
# RANKW globals (those scales are candidate-independent, so one serial computation before the batch is correct).
const SCALES_LOCKED = Ref{Bool}(false)
const W_SCALE_A   = Ref{Matrix{FloatType}}(zeros(FloatType, 0, 0))   # Sim A: Σ per-ref-max W1
const AGB_SCALE_A = Ref{Matrix{FloatType}}(zeros(FloatType, 0, 0))   # Sim A: Σ observed AGB
const W_SCALE_FACTOR = Ref{FloatType}(FloatType(1.0))                # multiplies W_SCALE_{A,B} (yaml w_scale_factor); <1 shrinks the divisor → amplifies the ΣW objective (rebalance vs AGB)
const W_SCALE_B   = Ref{Matrix{FloatType}}(zeros(FloatType, 0, 0))   # Sim B (tier-4 ref)
const AGB_SCALE_B = Ref{Matrix{FloatType}}(zeros(FloatType, 0, 0))
const RANKW       = Ref{Matrix{FloatType}}(zeros(FloatType, 0, 0))   # 1/log(rank+1), Σ=1, rank by AGB
const RANKW_MODE  = Ref{Symbol}(:rank)                               # :rank (1/√ln(rank) by AGB) or :cbal_pct (percentile-floored class-balanced by cohort count)
const RANKW_BETA  = Ref{Float64}(0.999)                              # β for :cbal_pct RANKW (effective-number temper)
const CELL_NCOH   = Ref{Matrix{Int}}(zeros(Int, 0, 0))               # per-(species,stratum) TRAIN cohort counts, for :cbal_pct RANKW

# --- COUNT-BALANCE reweight: per-agebin W1 weight ∝ 1/effective-number-of-samples of the cohort COUNT in
# that bin, to counteract survivorship (old bins are under-REPRESENTED, not under-massed). The count is
# bucketed into deciles-of-total so n=1 vs n=2 (noise) don't split; the decile is the ENS exponent d, and
# w = (1−β)/(1−β^d). Computed once from the TRAIN reference on the COARSE (age_idx) bins, frozen. Applied to
# the W1 term WITHOUT changing the binning — Sim A (per-year grid) looks up the coarse bin via CBAL_COARSE.
const CBAL_ON     = Ref{Bool}(false)                                 # yaml w_count_balance
const CBAL_MODE   = Ref{Symbol}(:both)                               # :a (Sim A) / :b (Sim B) / :both
const CBAL_BETA   = Ref{Float64}(0.99)                               # ENS temper; →1 ≈ 1/n, →0 ≈ uniform
const CBAL_W      = Ref{Matrix{Vector{FloatType}}}(Matrix{Vector{FloatType}}(undef, 0, 0))  # [gsp,eco]→weight per coarse bin
const CBAL_COARSE = Ref{Vector{Int}}(Int[])                          # Sim-A per-year bin k → coarse (age_idx) bin
@inline _w_finish(s::FloatType, gsp::Int, eco::Int)::FloatType = CELL_NORM[] ? _w_smooth(s, gsp, eco) : _w_pow(_w_norm(_w_smooth(s, gsp, eco)))
@inline _agb_finish(h::FloatType)::FloatType = CELL_NORM[] ? h            : _agb_pow(_agb_norm(h))

# Median-pivoted piecewise (applied to the AGGREGATE ΣW, ΣAGB in get_total_loss): f(x) = sqrt(x/m) if
# x<m else (x/m)². Pivot m = per-objective median (W_PIVOT/AGB_PIVOT). Lifts better-than-typical (sqrt,
# avoids suppressing small) and aggressively penalizes worse-than-typical (square); both objectives
# centered at 1 ⇒ comparable. LOSS_PIECEWISE toggles it.
const LOSS_PIECEWISE = Ref{Bool}(false)
const W_PIVOT = Ref{FloatType}(FloatType(1.0))
const AGB_PIVOT = Ref{FloatType}(FloatType(1.0))
@inline function _piecewise(x::FloatType, m::FloatType)::FloatType
  m <= zero(FloatType) && return x
  r = x / m
  r < one(FloatType) ? sqrt(r) : r * r
end
# Softplus (smooth ReLU) replacing the hard hinge max(0,x), so the loss surface isn't sharp at the band
# edge: the hinge max(0, |sim-obs| - thresh(obs))^p becomes softplus(|sim-obs| - thresh(obs))^p. β is a
# CONSTANT sharpness (it does NOT depend on the threshold); the knee LOCATION is set by the argument
# x = |sim-obs| - thresh(obs), so it sits at deviation = thresh. In-band leakage is ≈ e^(-β·thresh)/β,
# negligible for thresh ≳ a few. Knee width ~1/β (AGB units). β ≤ 0 ⇒ the hard ReLU. Stable form.
const AGB_HINGE_BETA = Ref{FloatType}(FloatType(1.0))
@inline function _hinge_relu(x::FloatType)::FloatType
  β = AGB_HINGE_BETA[]
  β <= zero(FloatType) && return max(zero(FloatType), x)
  return max(zero(FloatType), x) + log1p(exp(-β * abs(x))) / β
end

@inline function get_total_loss(loss::SiteLoss, alpha::FloatType=LOSS_ALPHA[], beta::FloatType=FloatType(1.0f0))::FloatType
  # sp_w_loss = age-distribution SHAPE (W1, AGB-invariant); sp_agb_loss = AGB LEVEL.
  # When normalizing, each sp_* term is already ÷ its global scale → the sum IS the intensive ratio-of-sums
  # (Σabs/Σref), so skip the /num_obs averaging; otherwise average per measurement as before.
  if LOSS_PIECEWISE[]   # median-pivoted piecewise on the aggregates → W & AGB centered at 1, comparable
    return alpha * _piecewise(sum(loss.sp_w_loss), W_PIVOT[]) + beta * _piecewise(sum(loss.sp_agb_loss), AGB_PIVOT[])
  end
  denom = (W_NORMALIZE[] || AGB_NORMALIZE[]) ? one(FloatType) : FloatType(loss.num_obs)
  return (alpha * sum(loss.sp_w_loss) + beta * sum(loss.sp_agb_loss)) / denom
end
@inline Base.convert(::Type{Float64}, a::SiteLoss) = Float64(get_total_loss(a))
#@inline Base.promote_rule(::Type{SiteLoss}, ::Type{Float64}) = Float64

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
@inline function smoothen_bin_cdf_forgive(p, obs_bin; age_bins::AgeBins)::Vector{FloatType}
  pc_bin = bin_ages(p; age_bins=age_bins.bins_idx, last_bin_open=age_bins.last_bin_open)
  @inbounds for i in eachindex(pc_bin)
    t = _w_hinge_thresh(obs_bin[i])
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
# weight for Sim A (tier-3) when enabled and MODE∈{:a,:both}: per-year bin k → coarse bin CBAL_COARSE[k].
@inline function _w1_sum(ref_cdf, sim_cdf, bw, gsp::Int, eco::Int)::FloatType
  use = CBAL_ON[] && CBAL_MODE[] !== :b && !isempty(CBAL_W[])
  s = zero(FloatType)
  if use
    w = @inbounds CBAL_W[][gsp, eco]; coarse = CBAL_COARSE[]
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

# Build CBAL_W (per gsp×eco weight over COARSE age_idx bins) + CBAL_COARSE (per-year→coarse map) from the
# TRAIN reference cohort records. Frozen after this call (call once at setup, like RANKW).
function _set_cbal_weights!(splots, coarse_bins::AgeBins, per_year_bins::AgeBins, eco_species_ids, n_species::Int; beta::Float64)
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
          pct = max(round(10 * c[b] / tot) / 10, 0.10)            # count-share → nearest 10%, floored at 10%
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
  CBAL_W[] = W
  # map each Sim-A W bin k to its coarse age_idx bin via the bin's upper age edge (works for per-year OR coarse)
  CBAL_COARSE[] = [find_age_bin(per_year_bins.bins_idx[k] - 1, coarse_bins) for k in eachindex(per_year_bins.bin_widths)]
  return nothing
end

@inline function calculate_species_loss!(; sp, gsp, site, ages, p, sp_start_idx, sp_end_idx, spdf_plt, loss_params, sp_w_loss, sp_agb_loss, site_agb_loss, lp::FloatType=one(FloatType), debug::Bool=false)
  sim_agb_sum = sum(@view site.c_bio[p[sp_start_idx:sp_end_idx]])
  #log_diff = log10(1 + sim_agb_sum) #+ loss_params.EPS)
  sp_agb_loss[gsp] = sim_agb_sum
  site_agb_loss += sim_agb_sum
  if spdf_plt.keys[sp]
    rec = @inbounds spdf_plt.records[sp]
    ages .= zero(FloatType)
    for a in @view p[sp_start_idx:sp_end_idx]
      ages[UIntType(site.c_age[a])] += site.c_bio[a]
    end
    sim_age_cdf = W_HINGE[] ?   # ±band "benefit of the doubt": forgive sim toward obs before the CDF (Sim A)
      smoothen_bin_cdf_forgive(ages, rec.sp_age_agb; age_bins=loss_params.age_bins) :
      smoothen_bin_cdf(ages; w=loss_params.smoothing_weights, age_bins=loss_params.age_bins)
    #@assert !any(isnan.(sim_age_cdf)) "cdf NaN"
    #@assert length(sim_age_cdf) == length(rec.sp_age_cdf) "cdf bins are not the same size"
    let bw = loss_params.age_bins.bin_widths
      # W1 (L1) on the NORMALIZED age CDF → distribution SHAPE only (÷ global W_SCALE), optionally
      # count-balance-reweighted per coarse age bin (CBAL) to counteract survivorship under-representation.
      sp_w_loss[gsp] = _w_finish(_w1_sum(rec.sp_age_cdf, sim_age_cdf, bw, Int(gsp), Int(site.eco_id)), Int(gsp), Int(site.eco_id))
    end
    #@assert !any(isnan.(sp_w_loss[gsp])) "NaN"
    # PER-COHORT AGB term (weighted by lambda): for each age bin i, softplus_β(|sim_i − obs_i| − thresh_i)^p,
    # obs_i = rec.sp_age_agb[i], thresh_i = _agb_hinge_thresh(obs_i); SUMMED over bins so age errors no longer
    # cancel. Stored raw — the per-species ÷scale, cell-norm rank and exponent apply downstream (for the L1 run
    # ^p is identity, so the per-bin _agb_pow and the downstream one are both no-ops). Fallback = Σ_i sqrt-diff².
    sim_agb_bins = bin_ages(ages; age_bins=loss_params.age_bins.bins_idx, last_bin_open=loss_params.age_bins.last_bin_open)
    agb_acc = zero(FloatType)
    if AGB_HINGE[]
      @inbounds for i in eachindex(rec.sp_age_agb)
        agb_acc += _agb_pow(_hinge_relu(abs(sim_agb_bins[i] - rec.sp_age_agb[i]) - _agb_hinge_thresh(rec.sp_age_agb[i])))
      end
      sp_agb_loss[gsp] = loss_params.lambda * _agb_finish(agb_acc)
    else
      @inbounds for i in eachindex(rec.sp_age_agb)
        agb_acc += (sqrt(max(sim_agb_bins[i], zero(FloatType))) - sqrt(max(rec.sp_age_agb[i], zero(FloatType))))^2
      end
      sp_agb_loss[gsp] = loss_params.lambda * agb_acc
    end
    site_agb_loss -= rec.sp_agb_sum
    if debug
      println("smoothing_weights $(loss_params.smoothing_weights)")
      smoothed_ages = smooth_ages(; ages=ages, smoothing_window=loss_params.smoothing_weights)
      binned_ages = bin_ages(smoothed_ages; age_bins=loss_params.age_bins.bins_idx, last_bin_open=loss_params.age_bins.last_bin_open)
      println("sim_sp: $(sp)")
      println("sim_agb: $(sim_agb_sum)")
      println("ref_agb: $(rec.sp_agb_sum)")
      println("sim_cdf: $(sim_age_cdf)")
      println("ref_cdf: $(rec.sp_age_cdf)")
      println("sim__ages: $(ages)")
      println("sim_sages: $(smoothed_ages)")
      println("sim_bages: $(binned_ages)")
      println("sp_w_loss: $(sp_w_loss)")


    end

  end
  #sp_w_loss[gsp] = sp_w_loss[gsp] #(1.0f0 + sp_w_loss[gsp]) * (abs(log_diff)^2)
  return site_agb_loss
end

function calculate_site_loss2(current_year::Int, site::SiteView, n_species::Int, eco_species_ids::Vector{Vector{Int}}, spdf_plt::SPDFGroundTruth, loss_params::LossParams; lp::FloatType=one(FloatType), debug::Bool=false, excluded::Union{Nothing,Set{UIntType}}=nothing)::SiteLoss
  # `excluded`: eco-species (local sp index) to SKIP from this site's loss at this year — used by the
  # disturbance exclude-modes (a partially-disturbed cohort the growth model can't fairly reproduce).
  _excl(sp) = excluded !== nothing && UIntType(sp) in excluded
  #Sort by species
  #println("-----------current_year $(current_year) -------------")
  #println("-----------current_site $(site.ref_cn) -------------")
  eco_n_species = length(site.sp_mature)
  species_id_map = eco_species_ids[site.eco_id]
  #@assert eco_n_species == length(spdf_plt.keys) "eco species numbers do not match"
  #@assert length(species_id_map) == eco_n_species "eco species numbers do not match"

  insite = falses(eco_n_species)

  #global loss
  sp_w_loss = zeros(FloatType, n_species)
  sp_agb_loss = zeros(FloatType, n_species)
  site_agb_loss = zero(FloatType)

  if site.live > 0

    c_species = @view site.c_species[1:site.live]
    max_age = UIntType(ceil(maximum(@view site.c_age[1:site.live])))
    max_age += UIntType(length(loss_params.smoothing_weights) >> 1)
    #get indices of species sorted
    # traversing c_species[p[1..end]] is equivalent to traversing sorted_c_species[1..end]
    # but now useful so that I don't need to sort c_age, c_bio
    p = sortperm(c_species)
    @debug c_species
    @debug p

    ages = Vector{FloatType}(undef, max_age)
    sp_start_idx = 1
    prev_sp = site.c_species[p[sp_start_idx]]


    #initialize losses
    # process sp's in site, sorted by species
    for i in eachindex(p)

      # get the
      sp = @inbounds c_species[p[i]]
      insite[sp] = true
      # keep looping until you find the end of sp segment
      # then use p[sp_start_index:sp_end_index] to gather from c_bio, c_age
      # conclude sp
      if sp != prev_sp
        @debug "concluding_species: $(prev_sp)"
        sp_end_idx = i - 1
        #println("here1: $(sp_start_idx):$(sp_end_idx), $(@view p[sp_start_idx:sp_end_idx])")
        # The segment [sp_start_idx:i-1] just finished belongs to prev_sp, NOT the new sp.
        if !_excl(prev_sp)
          site_agb_loss = calculate_species_loss!(; sp=prev_sp, gsp=species_id_map[prev_sp],
            site=site, ages=ages, p=p, sp_start_idx=sp_start_idx, sp_end_idx=sp_end_idx,
            spdf_plt=spdf_plt, loss_params=loss_params, sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=site_agb_loss, lp=lp, debug=debug)
        end
        prev_sp = sp
        sp_start_idx = i
      end
      @debug ("processing_species: $(sp)")
      if i == length(p)
        @debug ("concluding_species: $(sp)")
        sp_end_idx = i
        #println("here2: $(sp_start_idx):$(sp_end_idx), $(@view p[sp_start_idx:sp_end_idx])")
        if !_excl(sp)
          site_agb_loss = calculate_species_loss!(; sp=sp, gsp=species_id_map[sp],
            site=site, ages=ages, p=p, sp_start_idx=sp_start_idx, sp_end_idx=sp_end_idx,
            spdf_plt=spdf_plt, loss_params=loss_params, sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=site_agb_loss, lp=lp, debug=debug)
        end
      end
    end
  end

  for sp in (1:length(spdf_plt.keys))[spdf_plt.keys.&(.!insite)]
    _excl(sp) && continue   # disturbance-excluded species: don't penalize as "missing" either
    @inbounds rec = spdf_plt.records[UIntType(sp)]
    @inbounds gsp = species_id_map[sp]
    # Species present in REF but absent in SIM: sim CDF = 0, sim AGB = 0.
    # Shape penalty = full ref CDF²; level penalty = (sqrt(ref AGB))² = ref AGB (sim sqrt = 0).
    let bw = loss_params.age_bins.bin_widths
      sp_w_loss[gsp] = _w_finish(_w1_sum(rec.sp_age_cdf, nothing, bw, Int(gsp), Int(site.eco_id)), Int(gsp), Int(site.eco_id))  # sim absent → |0−ref|=ref
    end
    if AGB_HINGE[]   # per-cohort, sim_i = 0 ⇒ |sim_i − obs_i| = obs_i = rec.sp_age_agb[i]
      agb_acc0 = zero(FloatType)
      @inbounds for i in eachindex(rec.sp_age_agb)
        agb_acc0 += _agb_pow(_hinge_relu(rec.sp_age_agb[i] - _agb_hinge_thresh(rec.sp_age_agb[i])))
      end
      sp_agb_loss[gsp] = loss_params.lambda * _agb_finish(agb_acc0)
    else
      sp_agb_loss[gsp] = loss_params.lambda * rec.sp_agb_sum   # Σ_i (√obs_i)² = Σ_i obs_i = ref total
    end
    site_agb_loss -= rec.sp_agb_sum
  end


  return SiteLoss(sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=abs(site_agb_loss), num_sites=1)
  #return SiteLoss(sp_w_loss=zeros(FloatType, length(sp_w_loss)), sp_agb_loss=zeros(FloatType, length(sp_w_loss)), site_agb_loss=zero(FloatType))
  #return SiteLoss(sp_w_loss=sp_w_loss, sp_agb_loss=zeros(FloatType, length(sp_w_loss)), site_agb_loss=zero(FloatType))

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
# Allocation-free tier-3 loss path. Same arithmetic as calculate_site_loss2 / calculate_species_loss! /
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
@inline function smoothen_bin_cdf_forgive!(cdf::Vector{FloatType}, p::AbstractVector{FloatType}, obs_bin; age_bins::AgeBins)
  bin_ages!(cdf, p; age_bins=age_bins.bins_idx, last_bin_open=age_bins.last_bin_open)
  @inbounds for i in eachindex(cdf)
    t = _w_hinge_thresh(obs_bin[i])
    cdf[i] -= clamp(cdf[i] - obs_bin[i], -t, t)
    cdf[i] < zero(FloatType) && (cdf[i] = zero(FloatType))
  end
  cumsum!(cdf, cdf)   # forward running sum — aliasing src===dst is safe (each element depends only on priors)
  @inbounds (cdf[end] > zero(FloatType)) && (cdf ./= cdf[end])
  return cdf
end

# Accumulating per-species loss: adds this species' (W, AGB) loss into w_acc[gsp]/agb_acc[gsp] and returns the
# updated running site_agb_loss. `ages` (site-level, length max_age) is zeroed and refilled here; `cdf`/`agb_bins`
# are per-thread bin-sized scratch. Mirrors calculate_species_loss! exactly.
@inline function calculate_species_loss_acc!(; sp, gsp, site, ages, p, sp_start_idx, sp_end_idx, spdf_plt, loss_params,
    w_acc, agb_acc, site_agb_loss, cdf, agb_bins, lp::FloatType=one(FloatType))
  sim_agb_sum = sum(@view site.c_bio[p[sp_start_idx:sp_end_idx]])
  agb_val = sim_agb_sum
  w_val = zero(FloatType)
  site_agb_loss += sim_agb_sum
  if spdf_plt.keys[sp]
    rec = @inbounds spdf_plt.records[sp]
    fill!(ages, zero(FloatType))
    @inbounds for a in @view p[sp_start_idx:sp_end_idx]
      ages[UIntType(site.c_age[a])] += site.c_bio[a]
    end
    if W_HINGE[]   # ±band "benefit of the doubt" (Sim A) — allocation-free forgive path
      sim_age_cdf = smoothen_bin_cdf_forgive!(cdf, ages, rec.sp_age_agb; age_bins=loss_params.age_bins)
    else           # non-hinge path is rare (config uses w_hinge) — keep the allocating helper
      sim_age_cdf = smoothen_bin_cdf(ages; w=loss_params.smoothing_weights, age_bins=loss_params.age_bins)
    end
    let bw = loss_params.age_bins.bin_widths
      w_val = _w_finish(_w1_sum(rec.sp_age_cdf, sim_age_cdf, bw, Int(gsp), Int(site.eco_id)), Int(gsp), Int(site.eco_id))
    end
    bin_ages!(agb_bins, ages; age_bins=loss_params.age_bins.bins_idx, last_bin_open=loss_params.age_bins.last_bin_open)
    acc = zero(FloatType)
    if AGB_HINGE[]
      @inbounds for i in eachindex(rec.sp_age_agb)
        acc += _agb_pow(_hinge_relu(abs(agb_bins[i] - rec.sp_age_agb[i]) - _agb_hinge_thresh(rec.sp_age_agb[i])))
      end
      agb_val = loss_params.lambda * _agb_finish(acc)
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
# `perm`/`ages`/`insite`/`cdf`/`agb_bins` are per-thread scratch (resized in place). Mirrors calculate_site_loss2.
function calculate_site_loss2_acc!(w_acc::Vector{FloatType}, agb_acc::Vector{FloatType}, current_year::Int, site::SiteView,
    n_species::Int, eco_species_ids::Vector{Vector{Int}}, spdf_plt::SPDFGroundTruth, loss_params::LossParams,
    perm::Vector{Int}, ages::Vector{FloatType}, insite::Vector{Bool}, cdf::Vector{FloatType}, agb_bins::Vector{FloatType};
    lp::FloatType=one(FloatType), excluded::Union{Nothing,Set{UIntType}}=nothing)::FloatType
  _excl(sp) = excluded !== nothing && UIntType(sp) in excluded
  eco_n_species = length(site.sp_mature)
  species_id_map = eco_species_ids[site.eco_id]
  length(insite) == eco_n_species || resize!(insite, eco_n_species)
  fill!(insite, false)
  site_agb_loss = zero(FloatType)

  if site.live > 0
    c_species = @view site.c_species[1:site.live]
    max_age = UIntType(ceil(maximum(@view site.c_age[1:site.live])))
    max_age += UIntType(length(loss_params.smoothing_weights) >> 1)
    resize!(perm, site.live)
    sortperm!(perm, c_species)
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
            spdf_plt=spdf_plt, loss_params=loss_params, w_acc=w_acc, agb_acc=agb_acc,
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
            spdf_plt=spdf_plt, loss_params=loss_params, w_acc=w_acc, agb_acc=agb_acc,
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
      w_acc[gsp] += _w_finish(_w1_sum(rec.sp_age_cdf, nothing, bw, Int(gsp), Int(site.eco_id)), Int(gsp), Int(site.eco_id))
    end
    if AGB_HINGE[]
      agb_acc0 = zero(FloatType)
      for i in eachindex(rec.sp_age_agb)
        agb_acc0 += _agb_pow(_hinge_relu(rec.sp_age_agb[i] - _agb_hinge_thresh(rec.sp_age_agb[i])))
      end
      agb_acc[gsp] += loss_params.lambda * _agb_finish(agb_acc0)
    else
      agb_acc[gsp] += loss_params.lambda * rec.sp_agb_sum
    end
    site_agb_loss -= rec.sp_agb_sum
  end

  return abs(site_agb_loss)
end

# Flat enumeration of every tunable scalar dimension as (param_idx, target_idx), where
# target_idx is nothing (Global), an Int (Species/Eco), or an (eco,sp) tuple (EcoSpecies).
# `template` provides ECO_SPECIES_IDS, SPECIES_LIST, ECO_LIST. `length(build_slots(...))`
# is the search-space dimensionality `d` shared by sobol_samples and the CMA-ES bridge.
_eco_lu(s) = (p = split(String(s), "|lu="); length(p) == 2 ? String(p[2]) : "")  # eco "l3|lu=CELL" → "CELL"

function build_slots(param_dists::ParamDists{T}, template::T)::Vector{Tuple{Int,Any}} where T
  slots = Tuple{Int,Any}[]
  for (pi, param) in enumerate(param_dists.params)
    if param.sampler isa GlobalSampler
      push!(slots, (pi, nothing))
    elseif param.sampler isa SpeciesSampler
      for i in 1:length(template.SPECIES_LIST)
        push!(slots, (pi, i))
      end
    elseif param.sampler isa EcoSampler
      for i in 1:length(template.ECO_LIST)
        push!(slots, (pi, i))
      end
    elseif param.sampler isa EcoSpeciesSampler
      ss = get(PARAM_SPLIT_SETS[], param.name, nothing)   # global-species ids kept per-eco; others shared
      if ss === nothing
        for (eco_id, sp_ids) in enumerate(template.ECO_SPECIES_IDS)
          for sp_id in eachindex(sp_ids)
            push!(slots, (pi, (eco_id, sp_id)))
          end
        end
      else
        pos = Dict{Int,Vector{Tuple{Int,Int}}}()          # global species → all its (eco, sp_local) positions
        for (eco_id, sp_ids) in enumerate(template.ECO_SPECIES_IDS), sp_local in eachindex(sp_ids)
          push!(get!(pos, Int(sp_ids[sp_local]), Tuple{Int,Int}[]), (eco_id, sp_local))
        end
        mp = get(PARAM_TIER_MERGE[], param.name, nothing)
        for gsp in sort!(collect(keys(pos)))
          if gsp in ss
            mg = mp === nothing ? nothing : get(mp, gsp, nothing)
            if mg === nothing
              for t in pos[gsp]; push!(slots, (pi, t)); end   # split: one u-dim per (eco, species)
            else
              groups = Dict{String,Vector{Tuple{Int,Int}}}()  # merge the site-cells that share a label
              for t in pos[gsp]
                cell = _eco_lu(template.ECO_LIST[t[1]])
                push!(get!(groups, get(mg, cell, cell), Tuple{Int,Int}[]), t)
              end
              for lbl in sort!(collect(keys(groups)))
                g = groups[lbl]
                push!(slots, (pi, length(g) == 1 ? g[1] : g))  # merged cells → one broadcast u-dim
              end
            end
          else
            push!(slots, (pi, pos[gsp]))                    # shared: one u-dim broadcast across its ecos
          end
        end
      end
    end
  end
  return slots
end

# Map a u-vector in [0,1]^d to a params struct via each prior's quantile, then bounds-clip and
# Per-(eco_id, sp_local) LOWER bound for :B_MAX_SPP — the data-derived floor (≥12000), set by
# Pan._build_bmax_floor!. When non-nothing, u_to_params / params_to_u RESCALE the uniform prior onto
# [floor, upper] for each B_MAX slot (so there is no probability pile-up at the floor) and mutate_params
# clips to it. nothing = flat [12000,35000] window from the param bounds (feature off).
const BMAX_FLOOR = Ref{Union{Nothing,Dict{Tuple{Int,Int},FloatType}}}(nothing)

# Per-(eco_id, sp_local) LOWER bound for the DERIVED ANPP (g/m²/yr), set by Pan._build_anpp_floor!. Since
# ANPP = B_MAX/ratio under the reparam, this floors it at decode: ANPP = max(B_MAX/ratio, floor). Equivalent
# to a per-cell UPPER bound on the ratio (r ≤ B_MAX/floor). The params_to_u inverse is unchanged: recovering
# r = B_MAX/ANPP and re-decoding re-applies the floor (round-trip stable). nothing = no floor (feature off).
const ANPP_FLOOR = Ref{Union{Nothing,Dict{Tuple{Int,Int},FloatType}}}(nothing)

# Per-parameter SPLIT SET: for an EcoSpecies param, the set of GLOBAL species ids whose value is fit
# separately per ecoregion (stratum); species NOT in the set share ONE value across all ecoregions.
# `param.name ∉ keys` ⇒ every species split per-eco (the default, byte-identical to before). This lets
# e.g. B_MAX vary by site-tier only for the responsive species while the rest are tied — transparent to
# the plugin: u_to_params still fills every B_MAX_SPP[eco][sp] slot (shared species get the same value
# broadcast across their ecos). A shared slot's target is the Vector of all its (eco, sp_local) positions.
const PARAM_SPLIT_SETS = Ref{Dict{Symbol,Set{Int}}}(Dict{Symbol,Set{Int}}())

# Per-parameter, per-species SITE-CELL MERGE map for a split (stratified) species: gsp → (cell → group label).
# A split species' per-eco slots are grouped by their site-cell's label, so cells sharing a label share ONE
# fit value (e.g. PIEL {A,B}→"AB" ties SITECLCD 1-3 and 4). Absent species/param ⇒ every cell its own group
# (plain per-eco split). Lets different species use different site-productivity partitions off one plot label.
const PARAM_TIER_MERGE = Ref{Dict{Symbol,Dict{Int,Dict{String,String}}}}(Dict{Symbol,Dict{Int,Dict{String,String}}}())

# cast to the param's type — identical to the sobol_samples inner mapping. Bounds and discrete
# priors (DiscreteUniform) are handled automatically by quantile, so any u is representable.
function u_to_params(u::AbstractVector{<:Real}, param_dists::ParamDists{T}, slots::Vector{Tuple{Int,Any}}, template::T)::T where T
  p = template
  bf = BMAX_FLOOR[]
  af = ANPP_FLOOR[]
  for (dim, (pi, idx)) in enumerate(slots)
    param = param_dists.params[pi]
    min_, max_ = param.bounds
    tgts = idx isa Vector ? idx : (idx,)                 # >1 target ⇒ shared u-dim broadcast across ecos
    rep = first(tgts)                                    # representative target for floor / B_MAX lookup
    if param.name === :ANPP_MAX_SPP   # ratio reparam: u → ratio ∈ [20,35]; ANPP_MAX_SPP stores derived ANPP = B_MAX/ratio per eco (B_MAX_SPP decoded first). Shared ⇒ one ratio, ANPP re-derived per eco from that eco's B_MAX.
      ratio = Dists.quantile(param.dist, clamp(Float64(u[dim]), 1e-10, 1 - 1e-10))
      f = getproperty(p, :ANPP_MAX_SPP)
      for t in tgts
        bmax = Float64(get_field_val(param.applier, getproperty(p, :B_MAX_SPP), t))
        anpp = bmax / ratio
        af !== nothing && (anpp = max(anpp, Float64(get(af, t, 0.0))))   # data floor: ANPP ≥ p99(agb/age)
        f = apply_mutation(param.applier, f, t, param.type(anpp))
      end
      p = Setfield.@set p.ANPP_MAX_SPP = f
      continue
    end
    raw = if bf !== nothing && param.name === :B_MAX_SPP
      lo = Float64(get(bf, rep, 12000.0)); hi = Float64(max_)   # rescale uniform u onto [floor, upper]
      lo + clamp(Float64(u[dim]), 0.0, 1.0) * (hi - lo)
    else
      Dists.quantile(param.dist, clamp(Float64(u[dim]), 1e-10, 1 - 1e-10))
    end
    val = if !isnothing(min_) && raw < min_
      param.type(min_)
    elseif !isnothing(max_) && raw > max_
      param.type(max_)
    else
      param.type(raw)
    end
    val = _quantize(val, param.quantum, param.type)   # snap to the param's grid (e.g. B_MAX by 100)
    f = getproperty(p, param.name)
    for t in tgts                                     # single target ⇒ one write (identical to before)
      f = apply_mutation(param.applier, f, t, val)
    end
    p = Setfield.@set p.$(param.name) = f
  end
  return p
end

# Inverse of u_to_params (on the prior-grid): map each tunable value to u = cdf(prior, value).
# Used once to seed the CMA-ES mean from an initial params struct. Clamped off the open
# endpoints so the round-trip through quantile stays inside the bounds.
function params_to_u(params::T, param_dists::ParamDists{T}, slots::Vector{Tuple{Int,Any}})::Vector{Float64} where T
  u = Vector{Float64}(undef, length(slots))
  bf = BMAX_FLOOR[]
  for (dim, (pi, idx)) in enumerate(slots)
    param = param_dists.params[pi]
    rep = idx isa Vector ? idx[1] : idx                 # shared slot ⇒ read the representative eco (all equal)
    cur = get_field_val(param.applier, getproperty(params, param.name), rep)
    u[dim] = if param.name === :ANPP_MAX_SPP   # ratio reparam inverse: recover ratio = B_MAX/ANPP, map via the ratio prior. Handles legacy seeds (real ANPP) by clamping their implied ratio into [20,35].
      anpp = Float64(cur); bmax = Float64(get_field_val(param.applier, getproperty(params, :B_MAX_SPP), rep))
      ratio = anpp > 0 ? bmax / anpp : Float64(Dists.mean(param.dist))
      clamp(Float64(Dists.cdf(param.dist, ratio)), 1e-6, 1 - 1e-6)
    elseif bf !== nothing && param.name === :B_MAX_SPP
      lo = Float64(get(bf, rep, 12000.0)); hi = Float64(param.bounds[2])   # inverse of the u_to_params rescale
      clamp((Float64(cur) - lo) / (hi - lo), 1e-6, 1 - 1e-6)
    else
      clamp(Float64(Dists.cdf(param.dist, Float64(cur))), 1e-6, 1 - 1e-6)
    end
  end
  return u
end

# Hansen-style mixed-integer handling: a per-u-coordinate lower bound on the CMA-ES sampling
# std-dev, so discrete coordinates keep flipping integers even as σ shrinks (otherwise a coarse
# discrete like SHADE_TOL freezes once σ·√C_ii drops below its plateau width and all offspring
# round to the same value). Each DiscreteUniform prior with L levels has a u-space step width of
# 1/L; the floor is `factor/L`. Continuous priors get 0 (no floor). The floor self-targets coarse
# discretes — for fine ones (large L) it is tiny and effectively never binds. Returns a length-d
# vector aligned with `slots` (used to top up the marginal std in CMAES.ask).
function integer_u_min_std(param_dists::ParamDists{T}, slots::Vector{Tuple{Int,Any}}, factor::Float64=0.3)::Vector{Float64} where T
  s = zeros(Float64, length(slots))
  for (dim, (pi, _)) in enumerate(slots)
    p = param_dists.params[pi]
    if p.dist isa Dists.DiscreteUniform
      span = Float64(Dists.maximum(p.dist) - Dists.minimum(p.dist))   # value-range width
      step = p.quantum > zero(FloatType) ? Float64(p.quantum) : 1.0   # effective grid step
      L = floor(span / step) + 1                                      # number of reachable levels
      L > 1 && (s[dim] = factor / L)                                  # u-width of one step ≈ 1/L
    end
  end
  return s
end

# Generate N quasi-random Sobol samples covering the full parameter space.
# `param_dists` comes from e.g. BSP.make_biomass_param_dists(...).
# `initial_params` is the template struct (provides ECO_SPECIES_IDS, SPECIES_LIST, ECO_LIST).
function sobol_samples(param_dists::ParamDists{T}, initial_params::T, N::Int)::Vector{T} where T
  slots = build_slots(param_dists, initial_params)
  d = length(slots)
  seq = Sobol.SobolSeq(d)
  u = zeros(Float64, d)
  results = Vector{T}(undef, N)
  for n in 1:N
    Sobol.next!(seq, u)
    results[n] = u_to_params(u, param_dists, slots, initial_params)
  end
  return results
end

# Saltelli A/B/ABₖ design for GROUPED (param-type) Sobol sensitivity. Factors = parameter TYPES: one
# group per MutableParam name, whose eco/species/eco-species u-dims move together. Returns the N(K+2)
# param structs in evaluation order plus a matching tag vector ("A", "B", "AB:<name>"), so the existing
# sobol eval loop yields f(A), f(B), f(ABₖ) — enough for first-order Sᵢ and total-effect Sₜᵢ per type.
function saltelli_design(param_dists::ParamDists{T}, initial_params::T, N::Int) where T
  slots = build_slots(param_dists, initial_params)
  d = length(slots)
  groups = Tuple{Symbol,Vector{Int}}[]      # (param-name, u-dim indices), first-seen order
  gpos = Dict{Symbol,Int}()
  for (dim, (pi, _)) in enumerate(slots)
    nm = param_dists.params[pi].name
    haskey(gpos, nm) || (push!(groups, (nm, Int[])); gpos[nm] = length(groups))
    push!(groups[gpos[nm]][2], dim)
  end
  seq = Sobol.SobolSeq(d)                    # one d-dim sequence split: A = draws 1:N, B = draws N+1:2N
  A = Matrix{Float64}(undef, N, d); B = Matrix{Float64}(undef, N, d)
  u = zeros(Float64, d)
  for n in 1:N; Sobol.next!(seq, u); @views A[n, :] .= u; end
  for n in 1:N; Sobol.next!(seq, u); @views B[n, :] .= u; end
  samples = Vector{T}(undef, N * (length(groups) + 2))
  tags = Vector{String}(undef, length(samples))
  k = 0
  for n in 1:N; k += 1; samples[k] = u_to_params(@view(A[n, :]), param_dists, slots, initial_params); tags[k] = "A"; end
  for n in 1:N; k += 1; samples[k] = u_to_params(@view(B[n, :]), param_dists, slots, initial_params); tags[k] = "B"; end
  for (nm, cols) in groups
    for n in 1:N
      u2 = Vector{Float64}(A[n, :]); @views u2[cols] .= B[n, cols]   # ABₖ: A with group-k cols from B
      k += 1; samples[k] = u_to_params(u2, param_dists, slots, initial_params); tags[k] = "AB:" * String(nm)
    end
  end
  return samples, tags
end
