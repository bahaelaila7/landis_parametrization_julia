module BiomassSuccessionParametrization
export make_biomass_param_dists, BIOMASS_PER_ECO_GROUPS
# Block-diagonal CMA-ES groups that are split per ecoregion (one covariance matrix per eco); other
# groups get one matrix each. Passed to PU.build_groups by the CMA-ES drivers.
const BIOMASS_PER_ECO_GROUPS = Set([:growth_biomass, :min_rel])
# Stage-B: when true, FIX the growth/decay params {D, S, ANPP_MAX, B_MAX} (drop from the search) and
# fit only the establishment params {MATURITY, SHADE_TOL, PROB_ESTAB, MIN_REL, PROB_MORT}. LONGEVITY is
# fixed separately via longevity_from_data. Set from the `fix_growth` yaml knob.
const FIX_GROWTH = Ref{Bool}(false)
# Stage-B: PIN MATURITY out of the search (fixed at the seeded MATURITY_TABLE / SONA value). Set from the
# `fix_maturity` yaml knob. Requires MATURITY_TABLE (maturity_from_data) to supply the per-species values.
const FIX_MATURITY = Ref{Bool}(false)
# Stage-B: PIN MIN_REL_BIOMASS out of the search (fixed at the data-derived default [0.10, 0.275, 0.45,
# 0.625, 0.80], seeded via _seed_min_rel! in Pan.jl). Set from the `fix_min_rel` yaml knob.
const FIX_MIN_REL = Ref{Bool}(false)
using ..Parametrization
using ..PanCore
using ....Plugins.BiomassSuccessionPlugin: BiomassSuccessionParams, FIXED_LONGEVITY, LONGEVITY_TABLE, SHADE_TOL_TABLE
import Random
using Distributions
function make_biomass_param_dists(n_species::Int, n_ecoregions::Int, eco_species_ids::Vector{Vector{Int}}; no_establishment::Bool=false, fit_establishment::Bool=false)
  n_ess = sum(length(s) for s in eco_species_ids)
  # dual_mode: even with no_establishment (sync Sim A), FIT the introduction params for the free Sim B.
  # PROB_MORT stays pinned to 0; the rest {MATURITY, SHADE_TOL, PROB_ESTAB, MIN_REL} are searched.
  est = !no_establishment || fit_establishment
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
    (FIX_GROWTH[] ? () : (MutableParam(:D, Uniform(5.0f0, 25.0f0), (5.0f0, 25.0f0), 0.2f0, FloatType, SpeciesSampler(), IndexApplier(); group=:longevity_decay),))...,
    ((isnothing(FIXED_LONGEVITY[]) && isnothing(LONGEVITY_TABLE[])) ? (MutableParam(:LONGEVITY, DiscreteUniform(100, 600), (100, 600), 50, FloatType, SpeciesSampler(), IndexApplier(); quantum=50, group=:longevity_decay),) : ())...,
    (est && !FIX_MATURITY[] ? (MutableParam(:MATURITY, DiscreteUniform(1, 50), (1, 50), 3, FloatType, SpeciesSampler(), IndexApplier(); group=:maturity),) : ())...,
    (est && isnothing(SHADE_TOL_TABLE[]) ? (MutableParam(:SHADE_TOL, DiscreteUniform(1, 5), (1, 5), 1, UIntType, SpeciesSampler(), IndexApplier(); group=:shade),) : ())...,
    (FIX_GROWTH[] ? () : (MutableParam(:S, Uniform(0.01f0, 1.0f0), (0.01f0, 1.0f0), 0.01f0, FloatType, SpeciesSampler(), IndexApplier(); group=:longevity_decay),))...,   # GLOBAL per-species
    (FIX_GROWTH[] ? () : (MutableParam(:B_MAX_SPP, DiscreteUniform(12000, 35000), (12000, 35000), 1000, FloatType, EcoSpeciesSampler(), NestedIndexApplier(); quantum=100, group=:growth_biomass),))...,   # window widened 15000→12000 lower / 30000→35000 upper; per-(eco,species) data floor applied at decode via PU.BMAX_FLOOR. DECODED BEFORE ANPP so the ratio reparam reads the floored B_MAX.
    (FIX_GROWTH[] ? () : (MutableParam(:ANPP_MAX_SPP, Uniform(20.0f0, 35.0f0), (20.0f0, 35.0f0), 1.0f0, FloatType, EcoSpeciesSampler(), NestedIndexApplier(); group=:growth_biomass),))...,   # FITTED AS the B_MAX/ANPP ratio ∈ [20,35] (per eco×species); the ANPP_MAX_SPP FIELD stores the DERIVED ANPP = B_MAX_SPP/ratio (see u_to_params/params_to_u special-case). Enforces the eco ~25-30× relationship — no high-B_MAX/low-ANPP combos. Plugin unchanged (still reads ANPP_MAX_SPP).
    (no_establishment ? () : (MutableParam(:PROB_MORT_SPP, Uniform(0.0f0, 0.02f0), (0.0f0, 0.02f0), 0.005f0, FloatType, EcoSpeciesSampler(), NestedIndexApplier(); group=:growth_biomass),))...,   # bound 0.05→0.02: mortality was the overfit lever (cranked to ceiling to sculpt age-dist)
    (est ? (MutableParam(:PROB_ESTAB_SPP, Uniform(0.2f0, 1.0f0), (0.2f0, 1.0f0), 0.1f0, FloatType, EcoSpeciesSampler(), NestedIndexApplier(); group=:growth_biomass),) : ())...,   # floor 0.2: only sampled over present eco×lu×sp (absent ones aren't optimized → 0)
    (est && !FIX_MIN_REL[] ? (MutableParam(:MIN_REL_BIOMASS, Uniform(0.05f0, 0.2f0), (0.05f0, 0.2f0), 1f0, FloatType, EcoSampler(), GradientApplier(0.175f0); group=:min_rel),) : ())...,   # base ∈ [0.05,0.2], 0.175 spacing → last bucket 0.75–0.90
  ]
  weights = Float64[15 + (p.sampler isa EcoSpeciesSampler ? n_ess :
                          p.sampler isa SpeciesSampler ? n_species :
                          p.sampler isa EcoSampler ? n_ecoregions : 1)
                    for p in params]
  weights ./= sum(weights)
  ParamDists{BiomassSuccessionParams}(params, cumsum(weights))
end
end
