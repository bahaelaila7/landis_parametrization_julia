using ..PanCore
export MutableParam, SpeciesSampler, EcoSampler, GlobalSampler, EcoSpeciesSampler, GradientApplier, ScalarApplier, IndexApplier, NestedIndexApplier, ParamDists, SamplingContext, LossParams, LossScales, CbalTables, SiteLoss, AgeBins, get_smoothing_window, calculate_site_loss2, skipundef, MutationType, sobol_samples, saltelli_design, wasserstein1d, find_age_bin, build_slots, u_to_params, params_to_u, ParamSpace, ParamSpaceOptions, ParamFloors, ParamTying, CmaBlocks, with_floors, with_dists, slot_quanta, integer_u_min_std, params_to_nativevec, build_groups

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
# parameter's `group`. Groups whose symbol is in `blocks.per_eco_groups` are FURTHER split per ecoregion (a
# separate block per eco) — so eco-scoped params (Eco/EcoSpecies) yield one block per ecoregion, giving
# the "2 global + #ecoregion matrices" layout. Returns a Vector of u-index vectors (a partition of 1:d),
# in first-seen order. A single group (the default) ⇒ one block = plain CMA-ES.
# Covariance blocking mode (`CmaBlocks.mode`). :eco (legacy) = one block per (group, eco cell); :l3 = one block per (group, L3)
# where `eco_l3[eco_id]` gives the cell's L3 index — pooled slots spanning several L3s go to a tail block (-1).
"""
    CmaBlocks

The block-diagonal CMA-ES covariance layout of a parameter space: `mode` (`:eco` | `:l3`), the L3 / L2 index of every ecoregion cell
(`eco_l3`, `eco_l2`; empty in `:eco` mode; pooled slots spanning several L3s of ONE L2 get an L2 block) and the plugin's per-eco groups
(`per_eco_groups`, split once per ecoregion/L3; every other group is one global block).
"""
Base.@kwdef struct CmaBlocks
  mode::Symbol = :eco
  eco_l3::Vector{Int} = Int[]
  eco_l2::Vector{Int} = Int[]
  per_eco_groups::Set{Symbol} = Set{Symbol}()
end
function _block_eco(b::CmaBlocks, target)
  if b.mode === :l3 && !isempty(b.eco_l3)
    target isa Tuple && return b.eco_l3[Int(target[1])]
    target isa Integer && return b.eco_l3[Int(target)]
    if target isa AbstractVector && !isempty(target)
      ls = unique(b.eco_l3[Int(t[1])] for t in target)
      length(ls) == 1 && return ls[1]
      if !isempty(b.eco_l2)
        l2 = unique(b.eco_l2[Int(t[1])] for t in target)
        length(l2) == 1 && return 10_000 + l2[1]   # L2 block (offset keeps it apart from L3 indices)
      end
      return -1                                   # spans several L2s → region tail block
    end
    return 0
  end
  target isa Tuple ? Int(target[1]) : (target isa Integer ? Int(target) : 0)
end
function _build_groups(param_dists, slots::Vector{Tuple{Int,Any}}, blocks::CmaBlocks)
  keyed = Dict{Tuple{Symbol,Int},Vector{Int}}()
  order = Tuple{Symbol,Int}[]
  for (dim, (pi, target)) in enumerate(slots)
    g = param_dists.params[pi].group
    eco = (g in blocks.per_eco_groups) ? _block_eco(blocks, target) : 0
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

"""
    ParamSpaceOptions

The options that decide WHICH parameters a plugin searches and in what box (Step 3c; they were module `Ref`s of the Biomass Succession
parametrization). Immutable, built once per run by `Pan.param_space_options(cfg)`; the defaults are the legacy defaults.

  - `fix_growth`: stage B, pin the growth/decay params {D, S, ANPP_MAX, B_MAX} (drop them from the search); `fix_maturity`, `fix_min_rel`: pin MATURITY / MIN_REL_BIOMASS.
  - `death_pmort_sima`: Sim A with the death term fits PROB_MORT_SPP (score-only there); `fix_pmort`: Sim B pins PROB_MORT (frozen from the seed).
  - `pmort_upper_a`, `pmort_upper_b`: upper bound of the PROB_MORT_SPP box in Sim A (death term) / Sim B.
  - `bmax_bounds`, `anpp_ratio_bounds`: the B_MAX box and the B_MAX/ANPP ratio box.
  - `prob_estab_lower_bound`, `prob_estab_lower_mult`: use `mult` × the data PROB_ESTAB as the search's lower bound (builds `ParamFloors.prob_estab`).
"""
Base.@kwdef struct ParamSpaceOptions
  fix_growth::Bool = false
  fix_maturity::Bool = false
  fix_min_rel::Bool = false
  death_pmort_sima::Bool = false
  fix_pmort::Bool = false
  pmort_upper_a::Float32 = 0.2f0
  pmort_upper_b::Float32 = 0.02f0
  bmax_bounds::Tuple{Int,Int} = (12000, 35000)
  anpp_ratio_bounds::Tuple{Float32,Float32} = (20.0f0, 35.0f0)
  prob_estab_lower_bound::Bool = false
  prob_estab_lower_mult::Float64 = 1.0
end

"""
    ParamFloors

Per-(eco_id, sp_local) data bounds of the decode (each `nothing` = feature off), built by `Pan.build_param_floors`:
  - `bmax`: LOWER bound of `:B_MAX_SPP` (≥12000). `u_to_params` / `params_to_u` RESCALE the uniform prior onto [floor, upper] (no probability
    pile-up at the floor). A floor above the upper bound pins the slot at the floor for every u (B18).
  - `anpp`: LOWER bound of the DERIVED ANPP (g/m²/yr): ANPP = B_MAX/ratio is floored at decode, i.e. a per-cell UPPER bound on the ratio.
    `params_to_u` is unchanged (recovering r = B_MAX/ANPP and re-decoding re-applies the floor: round-trip stable).
  - `prob_estab`: LOWER bound of `:PROB_ESTAB_SPP` (window [floor, 1.0]); cells absent from the dict fall back to the param's own lower bound.
"""
Base.@kwdef struct ParamFloors
  bmax::Union{Nothing,Dict{Tuple{Int,Int},FloatType}} = nothing
  anpp::Union{Nothing,Dict{Tuple{Int,Int},FloatType}} = nothing
  prob_estab::Union{Nothing,Dict{Tuple{Int,Int},FloatType}} = nothing
end

"""
    ParamTying

Which (eco, species) targets of an EcoSpecies parameter share ONE u-dimension (empty = every cell its own dim, the default):
  - `labels` (param_pooling, region runs): param → (eco_id, sp_local) → pooling label ("cell:8.5.3|AB|PIEL", "l3:8.3.4|QURU", "l2:8.4|OXAR", "reg:R|_H_HST");
    takes precedence over `split_sets` / `tier_merge` for that param. `floor_pooled_max`: a pooled slot's B_MAX / PROB_ESTAB data floor is the MAX
    over its members (else the first member's: legacy, harmless when the members share one floor).
  - `split_sets`: param → the GLOBAL species ids fit separately per ecoregion (stratum); every other species shares ONE value across all ecoregions.
  - `tier_merge`: param → gsp → (site-cell → group label): a split species' per-eco slots are grouped by their site-cell's label, so cells sharing a
    label share one fit value (e.g. PIEL {A,B}→"AB").
"""
Base.@kwdef struct ParamTying
  labels::Dict{Symbol,Dict{Tuple{Int,Int},String}} = Dict{Symbol,Dict{Tuple{Int,Int},String}}()
  floor_pooled_max::Bool = false
  split_sets::Dict{Symbol,Set{Int}} = Dict{Symbol,Set{Int}}()
  tier_merge::Dict{Symbol,Dict{Int,Dict{String,String}}} = Dict{Symbol,Dict{Int,Dict{String,String}}}()
end

"""
    ParamSpace{T}

Everything that defines the SEARCH SPACE of one run, built once from (config, species/eco lists, `eco_species_ids`, data tables) and passed
explicitly to every driver, decoder and tool (no module state):
`dists` (the plugin's `ParamDists`), `opts` (`ParamSpaceOptions`), `tying` (pooling / split sets / merges), `floors` (`ParamFloors`),
`blocks` (`CmaBlocks`), the `slots` layout (one u-dimension each, see `build_slots`) and the lists the layout was derived from.
Construct with `ParamSpace(dists, n_species, eco_list, eco_species_ids; opts, tying, floors, blocks)`; `with_floors` / `with_dists` derive variants.
"""
struct ParamSpace{T}
  dists::ParamDists{T}
  opts::ParamSpaceOptions
  tying::ParamTying
  floors::ParamFloors
  blocks::CmaBlocks
  slots::Vector{Tuple{Int,Any}}
  n_species::Int
  eco_list::Vector{String}
  eco_species_ids::Vector{Vector{Int}}
end
function ParamSpace(dists::ParamDists{T}, n_species::Int, eco_list::AbstractVector{<:AbstractString}, eco_species_ids::Vector{Vector{Int}};
                    opts::ParamSpaceOptions=ParamSpaceOptions(), tying::ParamTying=ParamTying(), floors::ParamFloors=ParamFloors(),
                    blocks::CmaBlocks=CmaBlocks()) where {T}
  el = String[String(e) for e in eco_list]
  ParamSpace{T}(dists, opts, tying, floors, blocks, _build_slots(dists, tying, n_species, el, eco_species_ids), n_species, el, eco_species_ids)
end
"The same space with other data floors (the slot layout does not depend on them)."
with_floors(sp::ParamSpace{T}, floors::ParamFloors) where {T} =
  ParamSpace{T}(sp.dists, sp.opts, sp.tying, floors, sp.blocks, sp.slots, sp.n_species, sp.eco_list, sp.eco_species_ids)
"The same space over other distributions (e.g. a driver that fits another parameter set); the slot layout is re-derived."
with_dists(sp::ParamSpace{T}, dists::ParamDists{T}) where {T} =
  ParamSpace{T}(dists, sp.opts, sp.tying, sp.floors, sp.blocks, _build_slots(dists, sp.tying, sp.n_species, sp.eco_list, sp.eco_species_ids),
                sp.n_species, sp.eco_list, sp.eco_species_ids)

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


# loss mass of a slot's targets for loss-weighted slot choice (SamplingContext): species slots → sp_w_loss[species];
# eco×species slots → Σ over their (eco, local species) targets; global / eco slots → 1 (uniform among themselves)
function _slot_loss(ctx::SamplingContext, p, slot, sampler)::FloatType
  idx = slot[2]
  sampler isa SpeciesSampler && return ctx.sp_w_loss[idx]
  sampler isa EcoSpeciesSampler || return one(FloatType)
  tg = idx isa Vector ? idx : (idx,)
  sum(FloatType(ctx.sp_w_loss[Int(p.ECO_SPECIES_IDS[e][j])]) for (e, j) in tg)
end

# One annealing move (LBSA / MOLBSA / SA): pick a parameter by its sampling weight, one of its u-slots uniformly, and move that
# slot's u coordinate — Gaussian (σ = the parameter's native sigma over its box width) or a uniform redraw — then decode with
# the SAME u_to_params the CMA-ES drivers use, so pooling, the ANPP ratio reparam, data floors and quantization all hold.
# Mutating a B_MAX slot re-derives the ANPP of its targets at the unchanged ratio (as a full u decode would). Returns
# (params, param_idx); a slot that cannot change (pinned by a floor, single level) returns `p` unchanged after max_tries.
# B9: the previous version picked parameter i+1's weight for i (findlast(cumsum .<= s); the LAST parameter was never mutated —
# ANPP_MAX_SPP in Sim A), mutated single cells of pooled parameters, wrote the ANPP ratio prior's value (20–35) into ANPP,
# and looped forever on pinned slots.
function mutate_params(p::T, space::ParamSpace{T}; rng::Random.AbstractRNG, mutation_mode::MutationType=RandomMutation,
                       slots::Union{Nothing,Vector{Tuple{Int,Any}}}=nothing, dynamic_cumsum::Union{Nothing,Vector{Float64}}=nothing,
                       ctx::Union{Nothing,SamplingContext}=nothing, max_tries::Int=50) where {T}
  param_dists = space.dists
  wc = isnothing(dynamic_cumsum) ? param_dists.weights_cumsum : dynamic_cumsum
  s = rand(rng, Float64) * wc[end]
  param_idx = something(findfirst(>(s), wc), length(wc))
  param = param_dists.params[param_idx]
  sl = isnothing(slots) ? space.slots : slots
  dims = findall(t -> t[1] == param_idx, sl)
  isempty(dims) && return p, param_idx
  d = isnothing(ctx) ? dims[rand(rng, 1:length(dims))] : dims[_weighted_sample(FloatType[_slot_loss(ctx, p, sl[k], param.sampler) for k in dims], rng)]
  mode = mutation_mode != BothMutations ? mutation_mode : (rand(rng) > 0.5 ? GaussianMutation : RandomMutation)
  # the slot plus, for B_MAX, the ANPP slots sharing its targets (decoded after it, at their current ratio)
  tg(t) = t[2] isa Vector ? t[2] : (t[2],)
  sub = [d]
  if param.name === :B_MAX_SPP
    ai = findfirst(q -> q.name === :ANPP_MAX_SPP, param_dists.params)
    if ai !== nothing
      mine = Set(tg(sl[d]))
      append!(sub, [k for k in eachindex(sl) if sl[k][1] == ai && any(in(mine), tg(sl[k]))])
    end
  end
  subslots = sl[sub]
  u0 = params_to_u(p, space, subslots)
  lo, hi = param.bounds
  σu = (isnothing(lo) || isnothing(hi) || hi == lo) ? 0.1 : Float64(param.sigma) / Float64(hi - lo)
  cur = getproperty(p, param.name)
  for _ in 1:max_tries
    u = copy(u0)
    u[1] = mode == GaussianMutation ? clamp(u0[1] + σu * randn(rng), 0.0, 1.0) : rand(rng)
    q = u_to_params(u, space, subslots, p)
    getproperty(q, param.name) != cur && return q, param_idx
  end
  return p, param_idx
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

@inline function calculate_species_loss!(; sp, gsp, site, ages, p, sp_start_idx, sp_end_idx, spdf_plt, loss_params, scales, sp_w_loss, sp_agb_loss, site_agb_loss, lp::FloatType=one(FloatType), debug::Bool=false)
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
    sim_age_cdf = loss_params.w_hinge ?   # ±band "benefit of the doubt": forgive sim toward obs before the CDF (Sim A)
      smoothen_bin_cdf_forgive(ages, rec.sp_age_agb, loss_params; age_bins=loss_params.age_bins) :
      smoothen_bin_cdf(ages; w=loss_params.smoothing_weights, age_bins=loss_params.age_bins)
    #@assert !any(isnan.(sim_age_cdf)) "cdf NaN"
    #@assert length(sim_age_cdf) == length(rec.sp_age_cdf) "cdf bins are not the same size"
    let bw = loss_params.age_bins.bin_widths
      # W1 (L1) on the NORMALIZED age CDF → distribution SHAPE only (÷ global W_SCALE), optionally
      # count-balance-reweighted per coarse age bin (CBAL) to counteract survivorship under-representation.
      sp_w_loss[gsp] = _w_finish(loss_params, scales, _w1_sum(loss_params, scales, rec.sp_age_cdf, sim_age_cdf, bw, Int(gsp), Int(site.eco_id)), Int(gsp), Int(site.eco_id))
    end
    #@assert !any(isnan.(sp_w_loss[gsp])) "NaN"
    # PER-COHORT AGB term (weighted by lambda): for each age bin i, softplus_β(|sim_i − obs_i| − thresh_i)^p,
    # obs_i = rec.sp_age_agb[i], thresh_i = _agb_hinge_thresh(obs_i); SUMMED over bins so age errors no longer
    # cancel. Stored raw — the per-species ÷scale, cell-norm rank and exponent apply downstream (for the L1 run
    # ^p is identity, so the per-bin _agb_pow and the downstream one are both no-ops). Fallback = Σ_i sqrt-diff².
    sim_agb_bins = bin_ages(ages; age_bins=loss_params.age_bins.bins_idx, last_bin_open=loss_params.age_bins.last_bin_open)
    agb_acc = zero(FloatType)
    if loss_params.agb_hinge
      @inbounds for i in eachindex(rec.sp_age_agb)
        agb_acc += _agb_pow(loss_params, _hinge_relu(loss_params, abs(sim_agb_bins[i] - rec.sp_age_agb[i]) - _agb_hinge_thresh(loss_params, rec.sp_age_agb[i])))
      end
      sp_agb_loss[gsp] = loss_params.lambda * _agb_finish(loss_params, scales, agb_acc)
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

function calculate_site_loss2(current_year::Int, site::SiteView, n_species::Int, eco_species_ids::Vector{Vector{Int}}, spdf_plt::SPDFGroundTruth, loss_params::LossParams, scales::LossScales; lp::FloatType=one(FloatType), debug::Bool=false, excluded::Union{Nothing,Set{UIntType}}=nothing, excl_c::Union{Nothing,Vector{Tuple{UIntType,FloatType}}}=nothing)::SiteLoss
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

  p0 = excl_c === nothing ? nothing : (pp = Int[]; _excl_perm!(pp, site, excl_c); pp)
  if site.live > 0 && (p0 === nothing || !isempty(p0))

    c_species = @view site.c_species[1:site.live]
    max_age = UIntType(ceil(maximum(@view site.c_age[1:site.live])))
    max_age += UIntType(length(loss_params.smoothing_weights) >> 1)
    #get indices of species sorted
    # traversing c_species[p[1..end]] is equivalent to traversing sorted_c_species[1..end]
    # but now useful so that I don't need to sort c_age, c_bio
    p = p0 === nothing ? sortperm(c_species) : p0
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
            spdf_plt=spdf_plt, loss_params=loss_params, scales=scales, sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=site_agb_loss, lp=lp, debug=debug)
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
            spdf_plt=spdf_plt, loss_params=loss_params, scales=scales, sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=site_agb_loss, lp=lp, debug=debug)
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
      sp_w_loss[gsp] = _w_finish(loss_params, scales, _w1_sum(loss_params, scales, rec.sp_age_cdf, nothing, bw, Int(gsp), Int(site.eco_id)), Int(gsp), Int(site.eco_id))  # sim absent → |0−ref|=ref
    end
    if loss_params.agb_hinge   # per-cohort, sim_i = 0 ⇒ |sim_i − obs_i| = obs_i = rec.sp_age_agb[i]
      agb_acc0 = zero(FloatType)
      @inbounds for i in eachindex(rec.sp_age_agb)
        agb_acc0 += _agb_pow(loss_params, _hinge_relu(loss_params, rec.sp_age_agb[i] - _agb_hinge_thresh(loss_params, rec.sp_age_agb[i])))
      end
      sp_agb_loss[gsp] = loss_params.lambda * _agb_finish(loss_params, scales, agb_acc0)
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
# are per-thread bin-sized scratch. Mirrors calculate_species_loss! exactly.
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
# `perm`/`ages`/`insite`/`cdf`/`agb_bins` are per-thread scratch (resized in place). Mirrors calculate_site_loss2.
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

# Flat enumeration of every tunable scalar dimension as (param_idx, target_idx), where
# target_idx is nothing (Global), an Int (Species/Eco), or an (eco,sp) tuple (EcoSpecies).
# Built once per ParamSpace from the species count, the eco list and `eco_species_ids`; `length(space.slots)`
# is the search-space dimensionality `d` shared by sobol_samples and the CMA-ES bridge.
_eco_lu(s) = (p = split(String(s), "|lu="); length(p) == 2 ? String(p[2]) : "")  # eco "l3|lu=CELL" → "CELL"

function _build_slots(param_dists::ParamDists, tying::ParamTying, n_species::Int, eco_list::AbstractVector{<:AbstractString}, eco_species_ids::AbstractVector{<:AbstractVector{<:Integer}})::Vector{Tuple{Int,Any}}
  slots = Tuple{Int,Any}[]
  for (pi, param) in enumerate(param_dists.params)
    if param.sampler isa GlobalSampler
      push!(slots, (pi, nothing))
    elseif param.sampler isa SpeciesSampler
      for i in 1:n_species
        push!(slots, (pi, i))
      end
    elseif param.sampler isa EcoSampler
      for i in 1:length(eco_list)
        push!(slots, (pi, i))
      end
    elseif param.sampler isa EcoSpeciesSampler
      pl = get(tying.labels, param.name, nothing)          # param_pooling: one u-dim per pooling label
      ss = get(tying.split_sets, param.name, nothing)   # global-species ids kept per-eco; others shared
      if pl !== nothing
        groups = Dict{String,Vector{Tuple{Int,Int}}}()
        for (eco_id, sp_ids) in enumerate(eco_species_ids), sp_local in eachindex(sp_ids)
          push!(get!(groups, pl[(eco_id, sp_local)], Tuple{Int,Int}[]), (eco_id, sp_local))
        end
        for lbl in sort!(collect(keys(groups)))
          g = groups[lbl]
          push!(slots, (pi, length(g) == 1 ? g[1] : g))
        end
      elseif ss === nothing
        for (eco_id, sp_ids) in enumerate(eco_species_ids)
          for sp_id in eachindex(sp_ids)
            push!(slots, (pi, (eco_id, sp_id)))
          end
        end
      else
        pos = Dict{Int,Vector{Tuple{Int,Int}}}()          # global species → all its (eco, sp_local) positions
        for (eco_id, sp_ids) in enumerate(eco_species_ids), sp_local in eachindex(sp_ids)
          push!(get!(pos, Int(sp_ids[sp_local]), Tuple{Int,Int}[]), (eco_id, sp_local))
        end
        mp = get(tying.tier_merge, param.name, nothing)
        for gsp in sort!(collect(keys(pos)))
          if gsp in ss
            mg = mp === nothing ? nothing : get(mp, gsp, nothing)
            if mg === nothing
              for t in pos[gsp]; push!(slots, (pi, t)); end   # split: one u-dim per (eco, species)
            else
              groups = Dict{String,Vector{Tuple{Int,Int}}}()  # merge the site-cells that share a label
              for t in pos[gsp]
                cell = _eco_lu(eco_list[t[1]])
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

"The slot layout of `space` re-derived for the lists of `template` (a params struct); equals `space.slots` when the template belongs to the same data."
build_slots(space::ParamSpace{T}, template::T) where {T} =
  _build_slots(space.dists, space.tying, length(template.SPECIES_LIST), template.ECO_LIST, template.ECO_SPECIES_IDS)
"The CMA-ES block-diagonal covariance groups over `slots` (default `space.slots`): a partition of the u-coordinates, see `CmaBlocks`."
build_groups(space::ParamSpace, slots::Vector{Tuple{Int,Any}}=space.slots) = _build_groups(space.dists, slots, space.blocks)

_slot_floor(fl, tgts, dflt, pooled_max::Bool) = pooled_max ? maximum(Float64(get(fl, t, dflt)) for t in tgts) : Float64(get(fl, first(tgts), dflt))

# Map a u-vector in [0,1]^d to a params struct via each prior's quantile, then bounds-clip and
# cast to the param's type — identical to the sobol_samples inner mapping. Bounds and discrete
# priors (DiscreteUniform) are handled automatically by quantile, so any u is representable.
function u_to_params(u::AbstractVector{<:Real}, space::ParamSpace{T}, slots::Vector{Tuple{Int,Any}}, template::T)::T where T
  param_dists = space.dists
  p = template
  bf = space.floors.bmax
  af = space.floors.anpp
  pef = space.floors.prob_estab
  pmax = space.tying.floor_pooled_max
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
    lo_eff = min_   # effective lower bound for the clamp (raised to a per-cell data floor for B_MAX / PROB_ESTAB)
    raw = if bf !== nothing && param.name === :B_MAX_SPP
      lo = _slot_floor(bf, tgts, 12000.0, pmax); lo_eff = lo; hi = Float64(max_)   # rescale uniform u onto [floor, upper]
      lo + clamp(Float64(u[dim]), 0.0, 1.0) * (hi - lo)
    elseif pef !== nothing && param.name === :PROB_ESTAB_SPP
      lo = _slot_floor(pef, tgts, Float64(min_), pmax); lo_eff = lo; hi = Float64(max_)   # data lower bound → [floor, 1.0]
      lo + clamp(Float64(u[dim]), 0.0, 1.0) * (hi - lo)
    else
      Dists.quantile(param.dist, clamp(Float64(u[dim]), 1e-10, 1 - 1e-10))
    end
    # clip to [lo_eff, max_]; a data floor above the box upper bound wins (B18: the slot is pinned at its floor for every u —
    # previously u ≤ 0 decoded to the upper bound, BELOW the floor, and every u > 0 to the floor)
    v = raw
    !isnothing(max_) && v > max_ && (v = max_)
    !isnothing(lo_eff) && v < lo_eff && (v = lo_eff)
    val = param.type(v)
    val = _quantize(val, param.quantum, param.type)   # snap to the param's grid (e.g. B_MAX by 100)
    f = getproperty(p, param.name)
    for t in tgts                                     # single target ⇒ one write (identical to before)
      f = apply_mutation(param.applier, f, t, val)
    end
    p = Setfield.@set p.$(param.name) = f
  end
  return p
end

"`u_to_params(u, space, space.slots, template)`."
u_to_params(u::AbstractVector{<:Real}, space::ParamSpace{T}, template::T) where {T} = u_to_params(u, space, space.slots, template)

# Inverse of u_to_params (on the prior-grid): map each tunable value to u = cdf(prior, value).
# Used once to seed the CMA-ES mean from an initial params struct. Clamped off the open
# endpoints so the round-trip through quantile stays inside the bounds.
function params_to_u(params::T, space::ParamSpace{T}, slots::Vector{Tuple{Int,Any}})::Vector{Float64} where T
  param_dists = space.dists
  u = Vector{Float64}(undef, length(slots))
  bf = space.floors.bmax
  pef = space.floors.prob_estab
  pmax = space.tying.floor_pooled_max
  for (dim, (pi, idx)) in enumerate(slots)
    param = param_dists.params[pi]
    rep = idx isa Vector ? idx[1] : idx                 # shared slot ⇒ read the representative eco (all equal)
    tgts = idx isa Vector ? idx : (idx,)
    cur = get_field_val(param.applier, getproperty(params, param.name), rep)
    u[dim] = if param.name === :ANPP_MAX_SPP   # ratio reparam inverse: recover ratio = B_MAX/ANPP, map via the ratio prior. Handles legacy seeds (real ANPP) by clamping their implied ratio into [20,35].
      anpp = Float64(cur); bmax = Float64(get_field_val(param.applier, getproperty(params, :B_MAX_SPP), rep))
      ratio = anpp > 0 ? bmax / anpp : Float64(Dists.mean(param.dist))
      clamp(Float64(Dists.cdf(param.dist, ratio)), 1e-6, 1 - 1e-6)
    elseif bf !== nothing && param.name === :B_MAX_SPP
      lo = _slot_floor(bf, tgts, 12000.0, pmax); hi = Float64(param.bounds[2])   # inverse of the u_to_params rescale
      clamp((Float64(cur) - lo) / (hi - lo), 1e-6, 1 - 1e-6)
    elseif pef !== nothing && param.name === :PROB_ESTAB_SPP
      lo = _slot_floor(pef, tgts, Float64(param.bounds[1]), pmax); hi = Float64(param.bounds[2])   # inverse of the [floor,1] rescale
      clamp((Float64(cur) - lo) / (hi - lo), 1e-6, 1 - 1e-6)
    elseif param.dist isa Dists.DiscreteUnivariateDistribution
      # integer prior: encode to the MIDDLE of level k's mass interval (cdf(k-1), cdf(k)]. cdf(k) itself sits on the quantile
      # boundary, and floating point decoded it to k+1 for some levels (SHADE_TOL 3, MATURITY 7/14/28) — B17.
      k = Float64(cur)
      clamp(0.5 * (Float64(Dists.cdf(param.dist, k - 1)) + Float64(Dists.cdf(param.dist, k))), 1e-6, 1 - 1e-6)
    else
      clamp(Float64(Dists.cdf(param.dist, Float64(cur))), 1e-6, 1 - 1e-6)
    end
  end
  return u
end

"`params_to_u(params, space, space.slots)`."
params_to_u(params::T, space::ParamSpace{T}) where {T} = params_to_u(params, space, space.slots)

# native parameter VALUE per slot — same layout as `params_to_u`, but WITHOUT the prior-CDF transform.
# Used by parameter-space niching so decision distances are measured in native units (and thus in the
# per-parameter `quantum`s), not in the CDF-warped u-space. Reads only slot dims, so it is identical
# regardless of the `template`/`base` used to decode (frozen-growth-safe).
function params_to_nativevec(params::T, space::ParamSpace{T}, slots::Vector{Tuple{Int,Any}})::Vector{Float64} where T
  param_dists = space.dists
  v = Vector{Float64}(undef, length(slots))
  for (dim, (pi, idx)) in enumerate(slots)
    param = param_dists.params[pi]
    rep = idx isa Vector ? idx[1] : idx                 # shared slot ⇒ representative target (all equal)
    v[dim] = Float64(get_field_val(param.applier, getproperty(params, param.name), rep))
  end
  return v
end

# per-slot semantic quantum in NATIVE units: the parameter's `quantum` if set, else its Gaussian `sigma`
# (quantum-first, sigma-fallback). The scale below which a difference is deemed semantically irrelevant.
slot_quanta(space::ParamSpace, slots::Vector{Tuple{Int,Any}}=space.slots) =
  Float64[(p = space.dists.params[pi]; Float64(p.quantum > zero(FloatType) ? p.quantum : p.sigma)) for (pi, _) in slots]

# Hansen-style mixed-integer handling: a per-u-coordinate lower bound on the CMA-ES sampling
# std-dev, so discrete coordinates keep flipping integers even as σ shrinks (otherwise a coarse
# discrete like SHADE_TOL freezes once σ·√C_ii drops below its plateau width and all offspring
# round to the same value). Each DiscreteUniform prior with L levels has a u-space step width of
# 1/L; the floor is `factor/L`. Continuous priors get 0 (no floor). The floor self-targets coarse
# discretes — for fine ones (large L) it is tiny and effectively never binds. Returns a length-d
# vector aligned with `slots` (used to top up the marginal std in CMAES.ask).
function integer_u_min_std(space::ParamSpace, slots::Vector{Tuple{Int,Any}}=space.slots, factor::Float64=0.3)::Vector{Float64}
  s = zeros(Float64, length(slots))
  for (dim, (pi, _)) in enumerate(slots)
    p = space.dists.params[pi]
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
# `space` is the run's ParamSpace (see Pan.build_param_space); `initial_params` is the decode template.
function sobol_samples(space::ParamSpace{T}, initial_params::T, N::Int)::Vector{T} where T
  slots = space.slots
  d = length(slots)
  seq = Sobol.SobolSeq(d)
  u = zeros(Float64, d)
  results = Vector{T}(undef, N)
  for n in 1:N
    Sobol.next!(seq, u)
    results[n] = u_to_params(u, space, slots, initial_params)
  end
  return results
end

# Saltelli A/B/ABₖ design for GROUPED (param-type) Sobol sensitivity. Factors = parameter TYPES: one
# group per MutableParam name, whose eco/species/eco-species u-dims move together. Returns the N(K+2)
# param structs in evaluation order plus a matching tag vector ("A", "B", "AB:<name>"), so the existing
# sobol eval loop yields f(A), f(B), f(ABₖ) — enough for first-order Sᵢ and total-effect Sₜᵢ per type.
function saltelli_design(space::ParamSpace{T}, initial_params::T, N::Int) where T
  slots = space.slots
  d = length(slots)
  groups = Tuple{Symbol,Vector{Int}}[]      # (param-name, u-dim indices), first-seen order
  gpos = Dict{Symbol,Int}()
  for (dim, (pi, _)) in enumerate(slots)
    nm = space.dists.params[pi].name
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
  for n in 1:N; k += 1; samples[k] = u_to_params(@view(A[n, :]), space, slots, initial_params); tags[k] = "A"; end
  for n in 1:N; k += 1; samples[k] = u_to_params(@view(B[n, :]), space, slots, initial_params); tags[k] = "B"; end
  for (nm, cols) in groups
    for n in 1:N
      u2 = Vector{Float64}(A[n, :]); @views u2[cols] .= B[n, cols]   # ABₖ: A with group-k cols from B
      k += 1; samples[k] = u_to_params(u2, space, slots, initial_params); tags[k] = "AB:" * String(nm)
    end
  end
  return samples, tags
end
