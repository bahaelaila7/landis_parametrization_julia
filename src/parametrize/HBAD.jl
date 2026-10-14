module HBADParametrization
export make_hbad_param_dists, HBAD_PER_ECO_GROUPS
# Block-diagonal CMA-ES groups split per ecoregion/L3: the growth + mortality multipliers (EcoSpecies) and, when establishment is searched, PROB_ESTAB_SPP. SDIMAX (per species) shares one GLOBAL block.
const HBAD_PER_ECO_GROUPS = Set([:growth_hbad, :estab_hbad])
using ..Parametrization
using ..PanCore
using ....Plugins.HBADPlugin: HBADParams, BOUNDS_GMULT, BOUNDS_MMULT, BOUNDS_SDIMAX
using Distributions

function make_hbad_param_dists(n_species::Int, n_ecoregions::Int, eco_species_ids::Vector{Vector{Int}}; no_establishment::Bool=false, fit_establishment::Bool=false)
  n_ess = sum(length(s) for s in eco_species_ids)
  LU(b) = LogUniform(Float64(b[1]), Float64(b[2]))
  est = !no_establishment || fit_establishment
  params = [
    # EcoSpecies multipliers (log-uniform: symmetric in ln-space; default 1.0 sits at u ~ 0.5)
    MutableParam(:G_MULT, LU(BOUNDS_GMULT), BOUNDS_GMULT, 0.1f0, FloatType, EcoSpeciesSampler(), NestedIndexApplier(); group=:growth_hbad),
    MutableParam(:M_MULT, LU(BOUNDS_MMULT), BOUNDS_MMULT, 0.2f0, FloatType, EcoSpeciesSampler(), NestedIndexApplier(); group=:growth_hbad),
    # Species-level (global block). Global window; the plugin clamps each species to [0.7 p90, 1.3 p99] at decode (HBADPlugin.sdimax_range)
    MutableParam(:SDIMAX, Uniform(Float64(BOUNDS_SDIMAX[1]), Float64(BOUNDS_SDIMAX[2])), BOUNDS_SDIMAX, 40.0f0, FloatType, SpeciesSampler(), IndexApplier(); group=:hbad_species),
    # Establishment (Sim B): annual PROB_ESTAB per eco x species. SHADE_TOL / MATURITY / SUFFICIENT_LIGHT / MIN_REL_RD / ESTAB_TPH are fixed (data tables / defaults).
    (est ? (MutableParam(:PROB_ESTAB_SPP, Uniform(0.0, 1.0), (0.0f0, 1.0f0), 0.05f0, FloatType, EcoSpeciesSampler(), NestedIndexApplier(); group=:estab_hbad),) : ())...,
  ]
  weights = Float64[15 + (p.sampler isa EcoSpeciesSampler ? n_ess :
                          p.sampler isa SpeciesSampler ? n_species :
                          p.sampler isa EcoSampler ? n_ecoregions : 1)
                    for p in params]
  weights ./= sum(weights)
  ParamDists{HBADParams}(params, cumsum(weights))
end
end
