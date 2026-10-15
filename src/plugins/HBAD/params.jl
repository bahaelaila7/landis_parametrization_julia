# HBAD parameter containers. Fitted: G_MULT, M_MULT [EcoSpecies], SDIMAX [Species]. Fixed: LONGEVITY (as BiomassSuccession) and every
# coefficient of the context priors (results/hbad). Search bounds live here so params + parametrize agree.

const BOUNDS_GMULT = (0.3f0, 3.0f0)        # diameter-growth multiplier (log-uniform prior, centre ~0.95, default 1)
const BOUNDS_MMULT = (0.2f0, 5.0f0)        # mortality-hazard multiplier (log-uniform prior, median exactly 1)
const BOUNDS_SDIMAX = (200.0f0, 1500.0f0)  # global search window of SDImax; clamped per species to [0.7 p90, 1.3 p99] at decode
const SDIMAX_LO_MULT = 0.7
const SDIMAX_HI_MULT = 1.3

"""Starting-point modes of `generate_hbad_params` (`HBADContext.init_mode`): :default (G=M=1, SDImax=p95), :jitter (lognormal around the default; diverse initial populations), :prior (draws from the search priors)."""
const INIT_MODES = (:default, :jitter, :prior)

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
@inline annual_prob(p_cycle::Real, cycle_years::Real=DEFAULT_CYCLE_YEARS) = 1 - (1 - clamp(Float64(p_cycle), 0.0, 1.0))^(1 / Float64(cycle_years))

_eco_l3(name::AbstractString) = String(first(split(String(name), "|lu=")))

"""Annual PROB_ESTAB default of (eco name, species symbol): table value of (species, L3) else the species' median over L3s else the constant; converted per-cycle -> annual."""
function default_prob_estab(eco_name::AbstractString, sym::AbstractString; tbl::Union{Nothing,AbstractDict}=nothing, cycle_years::Real=DEFAULT_CYCLE_YEARS)
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
  return annual_prob(pc, cycle_years)
end

default_shade_tol(species_list, tables::BiomassSuccessionPlugin.BSTables=BiomassSuccessionPlugin.BSTables()) =
  UIntType[UIntType(clamp(get(something(tables.shade_tol, Dict{String,Int}()), sym, tables.shade_tol_default), 1, 5)) for sym in species_list]
default_maturity(species_list, tables::BiomassSuccessionPlugin.BSTables=BiomassSuccessionPlugin.BSTables()) =
  FloatType[FloatType(get(something(tables.maturity, Dict{String,Int}()), uppercase(sym), DEFAULT_MATURITY)) for sym in species_list]
"""The establishment defaults of a context: per-eco PROB_ESTAB_SPP (annual), SHADE_TOL and MATURITY per global species (from the BS tables)."""
establishment_defaults(ctx::HBADContext, eco_list, species_list, eco_species_ids) =
  (PROB_ESTAB_SPP=[FloatType[default_prob_estab(eco_list[e], species_list[Int(g)]; tbl=ctx.estab_prob_table, cycle_years=ctx.estab_cycle_years) for g in ids] for (e, ids) in enumerate(eco_species_ids)],
   SHADE_TOL=default_shade_tol(species_list, ctx.tables), MATURITY=default_maturity(species_list, ctx.tables))

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
    upgrade_params(raw[, ctx]) -> HBADParams

Params saved by an older version (JLD2 loads them as a `ReconstructedMutable` lacking the establishment fields) become a current `HBADParams`;
the missing establishment fields take their defaults (the context's establishment defaults when `ctx` is given; inert while sites carry `no_establish`).
A current `HBADParams` is returned unchanged.
"""
function upgrade_params(raw, ctx::Union{Nothing,HBADContext}=nothing)
  raw isa HBADParams && return raw
  have = propertynames(raw)
  kw = Dict{Symbol,Any}(f => getproperty(raw, f) for f in fieldnames(HBADParams) if f in have)
  if ctx !== nothing
    dflt = establishment_defaults(ctx, kw[:ECO_LIST], kw[:SPECIES_LIST], kw[:ECO_SPECIES_IDS])
    for (f, v) in pairs(dflt)
      haskey(kw, f) || (kw[f] = v)
    end
  end
  return HBADParams(; kw...)
end

"""Per-ecoregion resolved parameters of `params` under the context's priors (an old-format `params` is upgraded first)."""
generate_hbad_eco_params(ctx::HBADContext, params) = generate_hbad_eco_params(ctx, upgrade_params(params, ctx))

function generate_hbad_eco_params(ctx::HBADContext, params::HBADParams)::Vector{HBADEcoParams}
  check_context(ctx, params.SPECIES_LIST, params.ECO_SPECIES_IDS)
  pri = ctx.priors
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

function generate_hbad_params(ctx::HBADContext, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}};
                              rng::Random.AbstractRNG, no_establishment::Bool=false, mode::Symbol=ctx.init_mode, seed_rain::Bool=ctx.seed_rain)
  mode in INIT_MODES || error("HBAD init mode must be default|jitter|prior, got $mode")
  check_context(ctx, species_list, eco_species_ids)
  pri = ctx.priors
  n_species = length(species_list)
  LONG = BiomassSuccessionPlugin.default_longevity(ctx.tables, species_list)
  function sd_of(g)
    p = pri.sp[g]; lo, hi = sdimax_range(p)
    return mode === :default ? p.sdi_p95 : mode === :jitter ? _jit(rng, p.sdi_p95, lo, hi, 0.10) : lo + (hi - lo) * rand(rng)
  end
  SD = FloatType[sd_of(g) for g in 1:n_species]
  gen(lo, hi, sd) = mode === :default ? 1.0 : mode === :jitter ? _jit(rng, 1.0, lo, hi, sd) : exp(log(lo) + (log(hi) - log(lo)) * rand(rng))
  GM = [FloatType[gen(BOUNDS_GMULT[1], BOUNDS_GMULT[2], 0.25) for _ in ids] for ids in eco_species_ids]
  MM = [FloatType[gen(BOUNDS_MMULT[1], BOUNDS_MMULT[2], 0.25) for _ in ids] for ids in eco_species_ids]
  HBADParams(ECO_LIST=eco_list, SPECIES_LIST=species_list, ECO_SPECIES_IDS=[UIntType.(ids) for ids in eco_species_ids], LONGEVITY=LONG, SDIMAX=SD,
    G_MULT=GM, M_MULT=MM, ESTAB_SEED_RAIN=seed_rain; establishment_defaults(ctx, eco_list, species_list, eco_species_ids)...)
end
