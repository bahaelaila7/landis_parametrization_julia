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
  estab_dq::Float64 = 1.0        # cm: quadratic mean diameter of a newly established / planted cohort (age 1)
end
const FIXED = Ref(HBADConst())
"""Default of `ESTAB_SEED_RAIN` for generated parameter sets (yaml `hbad_seed_rain`)."""
const ESTAB_SEED_RAIN_DEFAULT = Ref(false)

# ---- establishment defaults (docs/hbad_plugin.md, "Establishment and planting") -------------------------------------
"""BiomassSuccession's SUFFICIENT_LIGHT matrix (rows = shade tolerance 1..5, columns = shade class 0..5), stored as BS does: one vector over tolerance per shade class. Corner (tol 5, class 5) = 0.50."""
const SUFFICIENT_LIGHT_MATRIX = FloatType[
  0.75 0.50 0.25 0.00 0.00 0.00;
  1.00 0.75 0.50 0.25 0.00 0.00;
  1.00 1.00 0.75 0.50 0.25 0.00;
  1.00 1.00 1.00 0.75 0.50 0.25;
  1.00 1.00 1.00 1.00 0.75 0.50]
default_sufficient_light() = [FloatType.(SUFFICIENT_LIGHT_MATRIX[:, c]) for c in axes(SUFFICIENT_LIGHT_MATRIX, 2)]
"""Default relative-density thresholds of the 5 shade classes (shade class 0..5 = number exceeded)."""
const DEFAULT_MIN_REL_RD = (0.15, 0.30, 0.45, 0.60, 0.75)
const DEFAULT_ESTAB_TPH = 750.0            # trees/ha of a new cohort (see docs)
const DEFAULT_MATURITY = 20.0              # years (mode of prob_estab_all_species.csv: 236 of 264 rows)
const DEFAULT_SHADE_TOL = 3
const DEFAULT_PROB_ESTAB_CYCLE = 0.60      # per-cycle occurrence used when a species/L3 is not in the table (median of prob_estab_all_species.csv = 0.603)
"""Per-cycle -> annual probability: a seedling occurrence p_cycle over `cycle_years` years = 1 - (1 - p_annual)^cycle_years."""
@inline annual_prob(p_cycle::Real, cycle_years::Real=ESTAB_CYCLE_YEARS[]) = 1 - (1 - clamp(Float64(p_cycle), 0.0, 1.0))^(1 / Float64(cycle_years))
"""Remeasurement interval (years) used to convert the per-cycle occurrence table into annual probabilities (yaml `cycle_years`; GA+FL FIA ~ 6-8)."""
const ESTAB_CYCLE_YEARS = Ref{Float64}(8.0)
"""Optional per-(species symbol, L3) per-cycle PROB_ESTAB table (prob_estab_all_species.csv), set by `load_estab_table!`."""
const ESTAB_PROB_TABLE = Ref{Union{Nothing,Dict{Tuple{String,String},Float64}}}(nothing)

"""Load prob_estab_all_species.csv (cols species, l3, prob_estab) into `ESTAB_PROB_TABLE[]`; returns the dict (nothing when absent)."""
function load_estab_table!(csv_path::AbstractString)
  if !isfile(csv_path)
    @warn "HBAD: PROB_ESTAB table $csv_path not found - using the constant fallback $(DEFAULT_PROB_ESTAB_CYCLE) per cycle"
    ESTAB_PROB_TABLE[] = nothing
    return nothing
  end
  df = CSV.read(csv_path, DataFrame)
  d = Dict{Tuple{String,String},Float64}((uppercase(strip(String(r.species))), String(r.l3)) => Float64(r.prob_estab) for r in eachrow(df))
  ESTAB_PROB_TABLE[] = d
  @info "HBAD: loaded per-cycle PROB_ESTAB for $(length(d)) (species, L3) cells from $csv_path (annual p = 1-(1-p)^(1/$(ESTAB_CYCLE_YEARS[])))"
  return d
end

_eco_l3(name::AbstractString) = String(first(split(String(name), "|lu=")))

"""Annual PROB_ESTAB default of (eco name, species symbol): table value of (species, L3) else the species' median over L3s else the constant; converted per-cycle -> annual."""
function default_prob_estab(eco_name::AbstractString, sym::AbstractString)
  tbl = ESTAB_PROB_TABLE[]
  pc = DEFAULT_PROB_ESTAB_CYCLE
  if tbl !== nothing
    k = (uppercase(sym), _eco_l3(eco_name))
    if haskey(tbl, k)
      pc = tbl[k]
    else
      vs = [v for ((s, _), v) in tbl if s == uppercase(sym)]
      isempty(vs) || (pc = sort(vs)[(length(vs) + 1) ÷ 2])
    end
  end
  return annual_prob(pc)
end

default_shade_tol(species_list) = UIntType[UIntType(clamp(get(something(BiomassSuccessionPlugin.SHADE_TOL_TABLE[], Dict{String,Int}()), sym, BiomassSuccessionPlugin.SHADE_TOL_DEFAULT[]), 1, 5)) for sym in species_list]
default_maturity(species_list) = FloatType[FloatType(get(something(BiomassSuccessionPlugin.MATURITY_TABLE[], Dict{String,Int}()), uppercase(sym), DEFAULT_MATURITY)) for sym in species_list]

"""HBAD parameters, indexed like BiomassSuccessionParams (global species ids; ECO_SPECIES_IDS maps eco-local -> global).
The establishment block (PROB_ESTAB_SPP ... ESTAB_SEED_RAIN) has defaults so that Sim-A-only constructions keep working; it is inert when sites carry `no_establish`."""
Base.@kwdef struct HBADParams
  ECO_LIST::Vector{String}
  SPECIES_LIST::Vector{String}
  ECO_SPECIES_IDS::Vector{Vector{UIntType}}
  LONGEVITY::Vector{FloatType}               # fixed (years), per global species
  SDIMAX::Vector{FloatType}                  # fitted, per global species (Reineke SDImax, trees/ha at 25.4 cm)
  G_MULT::Vector{Vector{FloatType}}          # fitted, eco x species (eco-local species index)
  M_MULT::Vector{Vector{FloatType}}          # fitted, eco x species
  # --- establishment (only used when the site is not flagged no_establish) ---
  PROB_ESTAB_SPP::Vector{Vector{FloatType}} = [FloatType[default_prob_estab(ECO_LIST[e], SPECIES_LIST[Int(g)]) for g in ids] for (e, ids) in enumerate(ECO_SPECIES_IDS)]  # ANNUAL probability, eco x species
  SHADE_TOL::Vector{UIntType} = default_shade_tol(SPECIES_LIST)                     # 1 (intolerant) .. 5 (tolerant), per global species
  MATURITY::Vector{FloatType} = default_maturity(SPECIES_LIST)                      # years: age at which a cohort can seed
  SUFFICIENT_LIGHT::Vector{Vector{FloatType}} = default_sufficient_light()          # [shade class 0..5][tolerance 1..5]
  MIN_REL_RD::Vector{Vector{FloatType}} = [FloatType[DEFAULT_MIN_REL_RD...] for _ in ECO_LIST]   # per eco: 5 ascending RD thresholds
  ESTAB_TPH::Vector{FloatType} = fill(FloatType(DEFAULT_ESTAB_TPH), length(SPECIES_LIST))        # trees/ha of a new cohort, per global species
  ESTAB_SEED_RAIN::Bool = false                                                     # true: every species of the ecoregion is a seed source
end

"""Per-ecoregion, per-candidate resolved parameters (eco-local species index; concrete isbits element types)."""
struct HBADEcoParams
  pri::Vector{SpPrior}
  gmult::Vector{Float64}
  mmult::Vector{Float64}
  longevity::Vector{Float64}
  sdimax::Vector{Float64}
  # establishment (eco-local species index)
  prob_estab::Vector{Float64}
  shade_tol::Vector{Int}
  maturity::Vector{Float64}
  estab_tph::Vector{Float64}
  light::Vector{Float64}             # flat [(shade class)*5 + tolerance], shade class 0..5, tolerance 1..5
  min_rel_rd::NTuple{5,Float64}
  seed_rain::Bool
end

@inline sdimax_range(p::SpPrior) = (SDIMAX_LO_MULT * p.sdi_p90, SDIMAX_HI_MULT * p.sdi_p99)

"""
    upgrade_params(raw) -> HBADParams

Params saved by an older version (JLD2 loads them as a `ReconstructedMutable` lacking the establishment fields) become a current `HBADParams`;
the missing establishment fields take their defaults (inert while sites carry `no_establish`). A current `HBADParams` is returned unchanged.
"""
function upgrade_params(raw)
  raw isa HBADParams && return raw
  have = propertynames(raw)
  return HBADParams(; (f => getproperty(raw, f) for f in fieldnames(HBADParams) if f in have)...)
end
generate_hbad_eco_params(params) = generate_hbad_eco_params(upgrade_params(params))

function generate_hbad_eco_params(params::HBADParams)::Vector{HBADEcoParams}
  ensure_configured!(params.SPECIES_LIST, params.ECO_SPECIES_IDS)
  pri = PRIORS[]
  out = Vector{HBADEcoParams}(undef, length(params.ECO_SPECIES_IDS))
  for (e, gids) in enumerate(params.ECO_SPECIES_IDS)
    n = length(gids)
    sp = Vector{SpPrior}(undef, n)
    gm = Vector{Float64}(undef, n); mm = Vector{Float64}(undef, n); lg = Vector{Float64}(undef, n); sd = Vector{Float64}(undef, n)
    pe = Vector{Float64}(undef, n); st = Vector{Int}(undef, n); mt = Vector{Float64}(undef, n); et = Vector{Float64}(undef, n)
    for j in 1:n
      g = Int(gids[j])
      sp[j] = pri.sp[g]
      gm[j] = params.G_MULT[e][j]; mm[j] = params.M_MULT[e][j]; lg[j] = params.LONGEVITY[g]
      lo, hi = sdimax_range(sp[j])
      sd[j] = clamp(Float64(params.SDIMAX[g]), lo, hi)
      pe[j] = clamp(Float64(params.PROB_ESTAB_SPP[e][j]), 0.0, 1.0); st[j] = clamp(Int(params.SHADE_TOL[g]), 1, 5)
      mt[j] = Float64(params.MATURITY[g]); et[j] = Float64(params.ESTAB_TPH[g])
    end
    length(params.SUFFICIENT_LIGHT) == 6 && all(v -> length(v) == 5, params.SUFFICIENT_LIGHT) || error("HBAD: SUFFICIENT_LIGHT must be 6 shade classes x 5 tolerance classes")
    light = Float64[Float64(params.SUFFICIENT_LIGHT[c][t]) for c in 1:6 for t in 1:5]
    length(params.MIN_REL_RD[e]) == 5 || error("HBAD: MIN_REL_RD[$e] must have 5 thresholds")
    mr = Tuple(Float64.(sort(params.MIN_REL_RD[e])))
    out[e] = HBADEcoParams(sp, gm, mm, lg, sd, pe, st, mt, et, light, mr, params.ESTAB_SEED_RAIN)
  end
  return out
end

@inline _jit(rng, med, lo, hi, sd) = clamp(med * exp(sd * randn(rng)), lo, hi)

function generate_hbad_params(species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}};
                              rng::Random.AbstractRNG, no_establishment::Bool=false, mode::Symbol=INIT_MODE[], seed_rain::Bool=ESTAB_SEED_RAIN_DEFAULT[])
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
    G_MULT=GM, M_MULT=MM, ESTAB_SEED_RAIN=seed_rain)
end
