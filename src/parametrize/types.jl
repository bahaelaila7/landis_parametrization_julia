using ..PanCore
export MutableParam, SpeciesSampler, EcoSampler,  GlobalSampler, EcoSpeciesSampler, GradientApplier, ScalarApplier, IndexApplier, NestedIndexApplier, ParamDists, SamplingContext 
import Setfield
import Random

struct MutableParam{S}
    name     :: Symbol
    dist     :: Any           # Distributions.jl distribution
    type     :: Type
    sampler  :: S             # how to pick the target element
    applier  :: Any           # how to write the sampled value back
end
struct ParamDists{T}
    params          :: Vector{MutableParam}
    weights_cumsum  :: Vector{Float64}
end
struct GlobalSampler end       # scalar field, no index needed
struct SpeciesSampler end      # pick a random species
struct EcoSampler end          # pick a random ecoregion
struct EcoSpeciesSampler end   # pick a random (eco, species) pair

struct ScalarApplier end                          # field[] = val
struct IndexApplier end                           # field[i] = val
struct NestedIndexApplier end                     # field[i][j] = val
struct GradientApplier
    step::FloatType
end         # field[i] = [val, val+step, ...]

sample_target(s::GlobalSampler,     p, rng::Random.AbstractRNG) = nothing
sample_target(s::SpeciesSampler,    p, rng::Random.AbstractRNG) = rand(rng, 1:length(p.SPECIES_LIST))
sample_target(s::EcoSampler,        p, rng::Random.AbstractRNG) = rand(rng, 1:length(p.ECO_LIST))
function sample_target(s::EcoSpeciesSampler, p, rng::Random.AbstractRNG)
    eco_id     = rand(rng, 1:length(p.ECO_SPECIES_IDS))
    species_id = rand(rng, 1:length(p.ECO_SPECIES_IDS[eco_id]))
    (eco_id, species_id)
end

apply_mutation(::ScalarApplier,       field, idx, val) = val
apply_mutation(::IndexApplier,        field, idx, val) = setindex!(copy(field), val, idx)
function apply_mutation(::NestedIndexApplier,  field, idx, val)
    nv = deepcopy(field)
    nv[idx[1]][idx[2]] = val
    nv
end
function apply_mutation(a::GradientApplier, field, idx, val)
    nv = deepcopy(field)
    nv[idx] = [val + k * a.step for k in 0:length(nv[idx])-1]
    nv
end


function mutate_params(p::T, param_dists::ParamDists{T}; rng::Random.AbstractRNG) where {T}
    s          = rand(rng, Float64)
    param_idx  = something(findlast(param_dists.weights_cumsum .<= s), 1)
    param      = param_dists.params[param_idx]

    val        = rand(rng, param.dist) |> param.type
    idx        = sample_target(param.sampler, p, rng)
    field      = getproperty(p, param.name)
    new_field  = apply_mutation(param.applier, field, idx, val)
    #println(field)
    #println(typeof(val))
    #println(new_field)

    Setfield.@set p.$(param.name) = new_field
end

