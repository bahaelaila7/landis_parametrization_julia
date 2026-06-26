module BiomassSuccessionParametrization
export make_biomass_param_dists, BIOMASS_PER_ECO_GROUPS
# Block-diagonal CMA-ES groups that are split per ecoregion (one covariance matrix per eco); other
# groups get one matrix each. Passed to PU.build_groups by the CMA-ES drivers.
const BIOMASS_PER_ECO_GROUPS = Set([:growth_biomass, :min_rel])
using ..Parametrization
using ..PanCore
using ....Plugins.BiomassSuccessionPlugin: BiomassSuccessionParams, FIXED_LONGEVITY, LONGEVITY_TABLE
import Random
using Distributions
function make_biomass_param_dists(n_species::Int, n_ecoregions::Int, eco_species_ids::Vector{Vector{Int}}; no_establishment::Bool=false)
  n_ess = sum(length(s) for s in eco_species_ids)
  params = [
    #MutableParam(:SPINUP_MORTALITY_FRACTION, Uniform(0.0f0, 0.2f0), (0.0f0, 0.2f0),0.05f0,FloatType, GlobalSampler(), ScalarApplier()),
    #MutableParam(:S, truncated(Normal(0.5, 1.0), 0.01, 1.0), FloatType, SpeciesSampler(), IndexApplier()),
    #MutableParam(:D, truncated(Normal(15, 10), 5, 25), FloatType, SpeciesSampler(), IndexApplier()),
    #MutableParam(:S, Uniform(0.01f0, 1.0f0), (0.01f0, 1.0f0), 0.01f0, FloatType, SpeciesSampler(), IndexApplier()),
    # Block-diagonal CMA-ES groups (see PU.build_groups). With establishment ON → 3 GLOBAL + 2×#eco blocks:
    #   GLOBAL: {LONGEVITY, D} · {MATURITY} · {SHADE_TOL}   (per-species params)
    #   PER-ECO: {S, ANPP_MAX, B_MAX, PROB_MORT, PROB_ESTAB} (growth/biomass) · {MIN_REL_BIOMASS} (separate)
    # Growth curve S is per-(eco,species). MIN_REL_BIOMASS gets its own per-eco block.
    # no_establishment = sync/manual-injection mode: introduction AND removal are done by the injection,
    # so the only parameters that affect the simulation are the deterministic growth/decay ones. We fit
    # ONLY {D, LONGEVITY, S, ANPP_MAX, B_MAX} and drop the rest, which are either reproduction-only
    # (MATURITY, SHADE_TOL, MIN_REL_BIOMASS, PROB_ESTAB — all inert when establishment is off) or the
    # random "act-of-god" mortality (PROB_MORT, pinned to 0 in generate_biomass_params). → 1 GLOBAL
    # {LONGEVITY, D} + 1×#eco {S, ANPP_MAX, B_MAX} blocks.
    MutableParam(:D, Uniform(5.0f0, 25.0f0), (5.0f0, 25.0f0), 0.2f0, FloatType, SpeciesSampler(), IndexApplier(); group=:longevity_decay),
    ((isnothing(FIXED_LONGEVITY[]) && isnothing(LONGEVITY_TABLE[])) ? (MutableParam(:LONGEVITY, DiscreteUniform(100, 600), (100, 600), 50, FloatType, SpeciesSampler(), IndexApplier(); quantum=50, group=:longevity_decay),) : ())...,
    (no_establishment ? () : (MutableParam(:MATURITY, DiscreteUniform(1, 50), (1, 50), 3, FloatType, SpeciesSampler(), IndexApplier(); group=:maturity),))...,
    (no_establishment ? () : (MutableParam(:SHADE_TOL, DiscreteUniform(1, 5), (1, 5), 1, UIntType, SpeciesSampler(), IndexApplier(); group=:shade),))...,
    MutableParam(:S, Uniform(0.01f0, 1.0f0), (0.01f0, 1.0f0), 0.01f0, FloatType, SpeciesSampler(), IndexApplier(); group=:longevity_decay),
    MutableParam(:ANPP_MAX_SPP, DiscreteUniform(100, 1500), (100, 1500), 100, FloatType, EcoSpeciesSampler(), NestedIndexApplier(); group=:growth_biomass),
    MutableParam(:B_MAX_SPP, DiscreteUniform(15000, 30000), (15000, 30000), 1000, FloatType, EcoSpeciesSampler(), NestedIndexApplier(); quantum=100, group=:growth_biomass),
    (no_establishment ? () : (MutableParam(:PROB_MORT_SPP, Uniform(0.0f0, 0.05f0), (0.0f0, 0.05f0), 0.01f0, FloatType, EcoSpeciesSampler(), NestedIndexApplier(); group=:growth_biomass),))...,
    (no_establishment ? () : (MutableParam(:PROB_ESTAB_SPP, Uniform(0.0f0, 1.0f0), (0.0f0, 1.0f0), 0.1f0, FloatType, EcoSpeciesSampler(), NestedIndexApplier(); group=:growth_biomass),))...,
    (no_establishment ? () : (MutableParam(:MIN_REL_BIOMASS, Uniform(0.0f0, 0.5f0), (0.0f0, 0.5f0), 1f0, FloatType, EcoSampler(), GradientApplier(0.10f0); group=:min_rel),))...,
  ]
  weights = Float64[15 + (p.sampler isa EcoSpeciesSampler ? n_ess :
                          p.sampler isa SpeciesSampler ? n_species :
                          p.sampler isa EcoSampler ? n_ecoregions : 1)
                    for p in params]
  weights ./= sum(weights)
  ParamDists{BiomassSuccessionParams}(params, cumsum(weights))
end
end
