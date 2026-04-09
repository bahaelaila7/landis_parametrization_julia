module BiomassSuccessionParametrization
export make_biomass_param_dists
using ..Parametrization
using ..PanCore
using ....Plugins.BiomassSuccessionPlugin: BiomassSuccessionParams
import Random
using Distributions
function make_biomass_param_dists(n_species::Int, n_ecoregions::Int, eco_species_ids::Vector{Vector{Int}})
    n_ess = sum(length(s) for s in eco_species_ids)
    params = [
        MutableParam(:SPINUP_MORTALITY_FRACTION, Uniform(0.0f0, 0.2f0), FloatType, GlobalSampler(), ScalarApplier()),
        #MutableParam(:S, truncated(Normal(0.5, 1.0), 0.01, 1.0), FloatType, SpeciesSampler(), IndexApplier()),
        MutableParam(:S, Uniform(0.01f0, 1.0f0), FloatType, SpeciesSampler(), IndexApplier()),
        #MutableParam(:D, truncated(Normal(15, 10), 5, 25), FloatType, SpeciesSampler(), IndexApplier()),
        MutableParam(:D, Uniform(5.0f0, 25.0f0), FloatType, SpeciesSampler(), IndexApplier()),
        #MutableParam(:LONGEVITY, truncated(Normal(200, 100), 100, 300), FloatType, SpeciesSampler(), IndexApplier()),
        MutableParam(:LONGEVITY, DiscreteUniform(100, 400), FloatType, SpeciesSampler(), IndexApplier()),
        MutableParam(:SHADE_TOL, DiscreteUniform(1, 5), UIntType, SpeciesSampler(), IndexApplier()),
        MutableParam(:MATURITY, DiscreteUniform(3, 40), FloatType, SpeciesSampler(), IndexApplier()),
        MutableParam(:PROB_MORT_SPP, Uniform(), FloatType, EcoSpeciesSampler(), NestedIndexApplier()),
        MutableParam(:PROB_ESTAB_SPP, Uniform(), FloatType, EcoSpeciesSampler(), NestedIndexApplier()),
        #MutableParam(:ANPP_MAX_SPP, truncated(Normal(2500, 100), 2400, 2500), FloatType, EcoSpeciesSampler(), NestedIndexApplier()),
        MutableParam(:ANPP_MAX_SPP, DiscreteUniform(2400, 2500), FloatType, EcoSpeciesSampler(), NestedIndexApplier()),
        #MutableParam(:B_MAX_SPP, truncated(Normal(2500, 100), 2400, 2500), FloatType, EcoSpeciesSampler(), NestedIndexApplier()),
        MutableParam(:B_MAX_SPP, DiscreteUniform(2400, 35000), FloatType, EcoSpeciesSampler(), NestedIndexApplier()),
        MutableParam(:MIN_REL_BIOMASS, Uniform(0.0f0, 0.5f0), FloatType, EcoSampler(), GradientApplier(0.10f0)),
    ]
    weights = Float64[15 + (p.sampler isa EcoSpeciesSampler ? n_ess :
                            p.sampler isa SpeciesSampler ? n_species :
                            p.sampler isa EcoSampler ? n_ecoregions : 1)
                      for p in params]
    weights ./= sum(weights)
    ParamDists{BiomassSuccessionParams}(params, cumsum(weights))
end
end
