# HBAD parameter containers. Fitted: G_MULT, M_MULT [EcoSpecies], SDIMAX [Species]. Fixed: LONGEVITY (as BiomassSuccession) and every
# coefficient of PRIORS[] (results/hbad). Search bounds live here so params + parametrize agree.

const BOUNDS_GMULT = (0.3f0, 3.0f0)        # diameter-growth multiplier (log-uniform prior, centre ~0.95, default 1)
const BOUNDS_MMULT = (0.2f0, 5.0f0)        # mortality-hazard multiplier (log-uniform prior, median exactly 1)
const BOUNDS_SDIMAX = (200.0f0, 1500.0f0)  # global search window of SDImax; clamped per species to [0.7 p90, 1.3 p99] at decode
const SDIMAX_LO_MULT = 0.7
const SDIMAX_HI_MULT = 1.3

"""Starting-point mode of `generate_hbad_params`: :default (G=M=1, SDImax=p95), :jitter (lognormal around the default; diverse initial populations), :prior (draws from the search priors)."""
const INIT_MODE = Ref{Symbol}(:jitter)

"""Fixed constants of the annual kernel (settable from tests)."""
Base.@kwdef struct HBADConst
  dmin_cov::Float64 = 2.54       # cm: diameters below this are evaluated at this value in the growth / mortality regressions (FIA tallies trees >= 1 in)
  gmax::Float64 = 3.0            # cm/yr: cap of the diameter increment
  haz_max::Float64 = 0.5         # /yr: cap of the mortality hazard
  rd_max::Float64 = 1.5          # cap of the relative density entering the mortality model
  min_tph::Float64 = 1.0         # trees/ha: cohorts below this density are removed
end
const FIXED = Ref(HBADConst())

"""HBAD parameters, indexed like BiomassSuccessionParams (global species ids; ECO_SPECIES_IDS maps eco-local -> global)."""
Base.@kwdef struct HBADParams
  ECO_LIST::Vector{String}
  SPECIES_LIST::Vector{String}
  ECO_SPECIES_IDS::Vector{Vector{UIntType}}
  LONGEVITY::Vector{FloatType}               # fixed (years), per global species
  SDIMAX::Vector{FloatType}                  # fitted, per global species (Reineke SDImax, trees/ha at 25.4 cm)
  G_MULT::Vector{Vector{FloatType}}          # fitted, eco x species (eco-local species index)
  M_MULT::Vector{Vector{FloatType}}          # fitted, eco x species
end

"""Per-ecoregion, per-candidate resolved parameters (eco-local species index; concrete isbits element types)."""
struct HBADEcoParams
  pri::Vector{SpPrior}
  gmult::Vector{Float64}
  mmult::Vector{Float64}
  longevity::Vector{Float64}
  sdimax::Vector{Float64}
end

@inline sdimax_range(p::SpPrior) = (SDIMAX_LO_MULT * p.sdi_p90, SDIMAX_HI_MULT * p.sdi_p99)

function generate_hbad_eco_params(params::HBADParams)::Vector{HBADEcoParams}
  ensure_configured!(params.SPECIES_LIST, params.ECO_SPECIES_IDS)
  pri = PRIORS[]
  out = Vector{HBADEcoParams}(undef, length(params.ECO_SPECIES_IDS))
  for (e, gids) in enumerate(params.ECO_SPECIES_IDS)
    n = length(gids)
    sp = Vector{SpPrior}(undef, n)
    gm = Vector{Float64}(undef, n); mm = Vector{Float64}(undef, n); lg = Vector{Float64}(undef, n); sd = Vector{Float64}(undef, n)
    for j in 1:n
      g = Int(gids[j])
      sp[j] = pri.sp[g]
      gm[j] = params.G_MULT[e][j]; mm[j] = params.M_MULT[e][j]; lg[j] = params.LONGEVITY[g]
      lo, hi = sdimax_range(sp[j])
      sd[j] = clamp(Float64(params.SDIMAX[g]), lo, hi)
    end
    out[e] = HBADEcoParams(sp, gm, mm, lg, sd)
  end
  return out
end

@inline _jit(rng, med, lo, hi, sd) = clamp(med * exp(sd * randn(rng)), lo, hi)

function generate_hbad_params(species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}};
                              rng::Random.AbstractRNG, no_establishment::Bool=false, mode::Symbol=INIT_MODE[])
  mode in (:default, :jitter, :prior) || error("HBAD init mode must be default|jitter|prior, got $mode")
  ensure_configured!(species_list, eco_species_ids)
  pri = PRIORS[]
  n_species = length(species_list)
  LONG = if !isnothing(BiomassSuccessionPlugin.LONGEVITY_TABLE[])
    tbl = BiomassSuccessionPlugin.LONGEVITY_TABLE[]
    FloatType[get(tbl, sym, BiomassSuccessionPlugin.LONGEVITY_DEFAULT[]) for sym in species_list]
  else
    fill(FloatType(something(BiomassSuccessionPlugin.FIXED_LONGEVITY[], 400.0)), n_species)
  end
  function sd_of(g)
    p = pri.sp[g]; lo, hi = sdimax_range(p)
    return mode === :default ? p.sdi_p95 : mode === :jitter ? _jit(rng, p.sdi_p95, lo, hi, 0.10) : lo + (hi - lo) * rand(rng)
  end
  SD = FloatType[sd_of(g) for g in 1:n_species]
  gen(lo, hi, sd) = mode === :default ? 1.0 : mode === :jitter ? _jit(rng, 1.0, lo, hi, sd) : exp(log(lo) + (log(hi) - log(lo)) * rand(rng))
  GM = [FloatType[gen(BOUNDS_GMULT[1], BOUNDS_GMULT[2], 0.25) for _ in ids] for ids in eco_species_ids]
  MM = [FloatType[gen(BOUNDS_MMULT[1], BOUNDS_MMULT[2], 0.25) for _ in ids] for ids in eco_species_ids]
  HBADParams(ECO_LIST=eco_list, SPECIES_LIST=species_list, ECO_SPECIES_IDS=[UIntType.(ids) for ids in eco_species_ids], LONGEVITY=LONG, SDIMAX=SD,
    G_MULT=GM, M_MULT=MM)
end
