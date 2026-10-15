module PnETParametrization
export make_pnet_param_dists, PNET_PER_ECO_GROUPS
# Block-diagonal CMA-ES groups split per ecoregion/L3: FolN (EcoSpecies) and the eco rooting-depth multiplier. The species-level
# params share one GLOBAL block.
const PNET_PER_ECO_GROUPS = Set([:growth_pnet, :soil])
using ..Parametrization
using ..PanCore
using ....Plugins.PnETPlugin: PnETParams, BOUNDS_FOLN, BOUNDS_MAXFRACFOL, BOUNDS_FRACTWD, BOUNDS_SLWMAX, BOUNDS_TOWOOD, BOUNDS_PSNAGERED, BOUNDS_RDMULT
using Distributions

function make_pnet_param_dists(n_species::Int, n_ecoregions::Int, eco_species_ids::Vector{Vector{Int}}; no_establishment::Bool=false, fit_establishment::Bool=false, opts::ParamSpaceOptions=ParamSpaceOptions())   # `opts` (BS switches) is unused by this plugin
  n_ess = sum(length(s) for s in eco_species_ids)
  U(b) = Uniform(b[1], b[2])
  sp(name, b, sigma, group) = MutableParam(name, U(b), b, sigma, FloatType, SpeciesSampler(), IndexApplier(); group=group)
  params = [
    # EcoSpecies: foliar N, per (eco, species) -- pooled across ecoregions by `param_pooling` exactly like B_MAX_SPP
    MutableParam(:FolN, U(BOUNDS_FOLN), BOUNDS_FOLN, 0.2f0, FloatType, EcoSpeciesSampler(), NestedIndexApplier(); group=:growth_pnet),
    # Species-level (global block): foliage allocation, active wood decay, leaf mass per area, wood turnover, age decline
    sp(:MaxFracFol, BOUNDS_MAXFRACFOL, 0.02f0, :pnet_species),
    sp(:FrActWd, BOUNDS_FRACTWD, 1.0f-5, :pnet_species),
    sp(:SLWmax, BOUNDS_SLWMAX, 20.0f0, :pnet_species),
    sp(:TOwood, BOUNDS_TOWOOD, 0.004f0, :pnet_species),
    sp(:PsnAgeRed, BOUNDS_PSNAGERED, 0.5f0, :pnet_species),
    # Eco: rooting depth / water-holding multiplier
    MutableParam(:RD_MULT, U(BOUNDS_RDMULT), BOUNDS_RDMULT, 0.15f0, FloatType, EcoSampler(), IndexApplier(); group=:soil),
  ]
  weights = Float64[15 + (p.sampler isa EcoSpeciesSampler ? n_ess :
                          p.sampler isa SpeciesSampler ? n_species :
                          p.sampler isa EcoSampler ? n_ecoregions : 1)
                    for p in params]
  weights ./= sum(weights)
  ParamDists{PnETParams}(params, cumsum(weights))
end
end
