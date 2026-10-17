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
