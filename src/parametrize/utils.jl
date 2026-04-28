using ..PanCore
export MutableParam, SpeciesSampler, EcoSampler, GlobalSampler, EcoSpeciesSampler, GradientApplier, ScalarApplier, IndexApplier, NestedIndexApplier, ParamDists, SamplingContext, LossParams, SiteLoss, AgeBins, get_smoothing_window, calculate_site_loss2, skipundef, MutationType

import Setfield
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
end
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
  nv = deepcopy(field)
  nv[idx[1]][idx[2]] = val
  nv
end
function apply_mutation(a::GradientApplier, field, idx, val)
  nv = deepcopy(field)
  nv[idx] = [val + k * a.step for k in 0:length(nv[idx])-1]
  nv
end


function mutate_params(p::T, param_dists::ParamDists{T}; rng::Random.AbstractRNG, mutation_mode::MutationType=RandomMutation, ctx::Union{Nothing,SamplingContext}=nothing) where {T}
  s = rand(rng, Float64)
  param_idx = something(findlast(param_dists.weights_cumsum .<= s), 1)
  param = param_dists.params[param_idx]
  _mutation_mode = mutation_mode != BothMutations ? mutation_mode : (rand(rng) > 0.5 ? GaussianMutation : RandomMutation)

  idx = isnothing(ctx) ? sample_target(param.sampler, p, rng) :
        sample_target(param.sampler, p, rng, ctx)
  field = getproperty(p, param.name)
  val = begin
    cur_val = get_field_val(param.applier, field, idx)
    r = cur_val
    min_, max_ = param.bounds
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
    end
    r
  end

  new_field = apply_mutation(param.applier, field, idx, val)
  #println(field)
  #println(typeof(val))
  #println(new_field)

  Setfield.@set p.$(param.name) = new_field
end



@inline skipundef(xs::AbstractArray) = (xs[i] for i in eachindex(xs) if isassigned(xs, i))

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
  lambda::FloatType = FloatType(1.0f-2)
  EPS::FloatType = FloatType(1.0f-7)
end

Base.@kwdef struct SPDFRecord
  sp_agb_sum::FloatType
  sp_age_cdf::Vector{FloatType}
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
end
@inline function Base.:+(loss1::SiteLoss, loss2::SiteLoss)
  SiteLoss(sp_w_loss=loss1.sp_w_loss .+ loss2.sp_w_loss,
    sp_agb_loss=loss1.sp_agb_loss .+ loss2.sp_agb_loss,
    site_agb_loss=loss1.site_agb_loss + loss2.site_agb_loss,
    num_sites=loss1.num_sites + loss2.num_sites)
end
@inline function get_total_loss(loss::SiteLoss, alpha::FloatType=FloatType(1.0f0), beta::FloatType=FloatType(1.0f0))::FloatType

  w = (alpha * loss.sp_w_loss)
  sp = (beta * log.(1 .+ loss.sp_agb_loss))
  #site = log(1+loss.site_agb_loss)
  all = sum(w .+ sp .+ (w .* sp))
  #all = w + sp #+ site
  return all / loss.num_sites

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



@inline function smooth_ages(; ages::Vector{FloatType}, smoothing_window::Vector{FloatType})::Vector{FloatType}
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
    ages = zeros(FloatType, max_age)
    for row in eachrow(rows)
      ages[row.age_calc] += row.agb_sum
    end
    row = rows[1, :]
    sim_year = Dates.value.(Dates.Day.(row.measdate - row.start_measdate)) ./ 365.25 .|> round .|> Int
    @assert sim_year >= 0 "negative sim_year $row"
    cdf = smoothen_bin_cdf(ages; w=loss_params.smoothing_weights, age_bins=loss_params.age_bins)
    if debug
      smoothed_ages = smooth_ages(; ages=ages, smoothing_window=loss_params.smoothing_weights)
      binned_ages = bin_ages(smoothed_ages; age_bins=loss_params.age_bins.bins_idx, last_bin_open=loss_params.age_bins.last_bin_open)
      (; sim_year=[sim_year], data_agb_sum=[sum(rows.agb_sum)], data_agbs_cdf=[cdf],
        ages=[ages], smoothed_ages=[smoothed_ages], binned_ages=[binned_ages])
    else
      (; sim_year=[sim_year], data_agb_sum=[sum(rows.agb_sum)], data_agbs_cdf=[cdf])
    end
  end
  return spdf


  #plot_id x eco_id, measdate, sim_year, swhd, spgrpcd, species_id -> smoothened(biomass by age)
  #plot_id x eco_id, measdate, sim_year, swhd, spgrpcd -> smoothened_binned(biomass by age)
  #plot_id x eco_id, measdate, sim_year, swhd -> biomass by age

end

function get_bin_widths(; age_bins::Vector{Int}, last_bin_open::Bool)
  # returns the bin widths for wasser1 (ie K-1 widths)
  if last_bin_open
    @assert length(age_bins) > 0 "insufficint bins, must be at least 1"
    return age_bins .- [0; age_bins[begin:end-1]]
  else
    @assert length(age_bins) > 1 "insufficint bins, must be at least 2"
    return age_bins[begin:end-1] .- [0; age_bins[begin:end-2]]
  end

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



@inline function calculate_species_loss!(; sp, gsp, site, ages, p, sp_start_idx, sp_end_idx, spdf_plt, loss_params, sp_w_loss, sp_agb_loss, site_agb_loss, debug::Bool=false)
  sim_agb_sum = sum(@view site.c_bio[p[sp_start_idx:sp_end_idx]])
  log_diff = log10(1 + sim_agb_sum) #+ loss_params.EPS)
  sp_agb_loss[gsp] = sim_agb_sum
  site_agb_loss += sim_agb_sum
  if spdf_plt.keys[sp]
    rec = @inbounds spdf_plt.records[sp]
    ages .= zero(FloatType)
    for a in @view p[sp_start_idx:sp_end_idx]
      ages[UIntType(site.c_age[a])] = site.c_bio[a]
    end
    sim_age_cdf = smoothen_bin_cdf(ages; w=loss_params.smoothing_weights, age_bins=loss_params.age_bins)
    #@assert !any(isnan.(sim_age_cdf)) "cdf NaN"
    #@assert length(sim_age_cdf) == length(rec.sp_age_cdf) "cdf bins are not the same size"
    sp_w_loss[gsp] = sum(loss_params.age_bins.bin_widths .* abs.(sim_age_cdf - rec.sp_age_cdf)[begin:end-1])
    #@assert !any(isnan.(sp_w_loss[gsp])) "NaN"
    log_diff -= log10(1 + rec.sp_agb_sum) #+ loss_params.EPS)
    sp_agb_loss[gsp] = abs(sim_agb_sum - rec.sp_agb_sum)
    site_agb_loss -= rec.sp_agb_sum
    if debug
      smoothed_ages = smooth_ages(; ages=ages, smoothing_window=loss_params.smoothing_weights)
      binned_ages = bin_ages(smoothed_ages; age_bins=loss_params.age_bins.bins_idx, last_bin_open=loss_params.age_bins.last_bin_open)
      println("sim_sp: $(sp)")
      println("sim_agb: $(sim_agb_sum)")
      println("sim_cdf: $(sim_age_cdf)")
      println("sim__ages: $(ages)")
      println("sim_sages: $(smoothed_ages)")
      println("sim_bages: $(binned_ages)")

    end
  end
  #sp_w_loss[gsp] = sp_w_loss[gsp] #(1.0f0 + sp_w_loss[gsp]) * (abs(log_diff)^2)
  return site_agb_loss
end

function calculate_site_loss2(current_year::Int, site::SiteView, n_species::Int, eco_species_ids::Vector{Vector{Int}}, spdf_plt::SPDFGroundTruth, loss_params::LossParams; debug::Bool=false)::SiteLoss
  #Sort by species
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
        @debug "concluding_species: $(sp)"
        sp_end_idx = i - 1
        site_agb_loss = calculate_species_loss!(; sp=sp, gsp=species_id_map[sp],
          site=site, ages=ages, p=p, sp_start_idx=sp_start_idx, sp_end_idx=sp_end_idx,
          spdf_plt=spdf_plt, loss_params=loss_params, sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=site_agb_loss, debug=debug)
        prev_sp = sp
        sp_start_idx = i
      end
      @debug ("processing_species: $(sp)")
      if i == length(p)
        @debug ("concluding_species: $(sp)")
        sp_end_idx = i
        site_agb_loss = calculate_species_loss!(; sp=sp, gsp=species_id_map[sp],
          site=site, ages=ages, p=p, sp_start_idx=sp_start_idx, sp_end_idx=sp_end_idx,
          spdf_plt=spdf_plt, loss_params=loss_params, sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=site_agb_loss, debug=debug)
      end
    end
  end

  for sp in (1:length(spdf_plt.keys))[spdf_plt.keys.&(.!insite)]
    @inbounds rec = spdf_plt.records[UIntType(sp)]
    @inbounds gsp = species_id_map[sp]
    sp_agb_loss[gsp] = rec.sp_agb_sum
    #@assert sp_w_loss[gsp] == 0
    sp_w_loss[gsp] = sum(loss_params.age_bins.bin_widths .* rec.sp_age_cdf[begin:end-1])
    #sp_w_loss[gsp] = loss_params.lambda * abs(log10(1+rec.sp_agb_sum)) # + loss_params.EPS))
    site_agb_loss -= rec.sp_agb_sum
  end


  return SiteLoss(sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=abs(site_agb_loss), num_sites=1)

end
