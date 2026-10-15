# The immutable per-run context of the HBAD plugin (Step 3d): everything the plugin used to read from module-level state.
# Built once per (species list, eco -> species table) by `HBADContext(...)`; passed explicitly to `generate_hbad_params`,
# `generate_hbad_eco_params`, `add_cohort!`, `set_cohort_bio!`, `plant!` and carried in the `ctx` of `Pan.step!` (`make_ctx`).

"""Fixed constants of the annual kernel (settable from tests)."""
Base.@kwdef struct HBADConst
  dmin_cov::Float64 = 2.54       # cm: diameters below this are evaluated at this value in the growth / mortality regressions (FIA tallies trees >= 1 in)
  gmax::Float64 = 3.0            # cm/yr: cap of the diameter increment
  haz_max::Float64 = 0.5         # /yr: cap of the mortality hazard
  rd_max::Float64 = 1.5          # cap of the relative density entering the mortality model
  min_tph::Float64 = 1.0         # trees/ha: cohorts below this density are removed
  estab_dq::Float64 = 1.0        # cm: quadratic mean diameter of a newly established / planted cohort (age 1)
end

"""Default remeasurement interval (years) of the per-cycle -> annual PROB_ESTAB conversion."""
const DEFAULT_CYCLE_YEARS = 8.0

"""
    HBADContext

  * `priors`: fixed per-species coefficients (index = global species id of the run), resolved from priors.csv / translation_coefs.csv;
  * `eco_gids`: eco-local -> global species id table (`add_cohort!` only sees (eco, local species));
  * `fixed`: kernel constants; `init_mode`: start of generated parameter sets (`:default` | `:jitter` | `:prior`);
  * `estab_prob_table`: optional per-(species symbol, L3) per-cycle PROB_ESTAB (prob_estab_all_species.csv); `estab_cycle_years`: per-cycle -> annual conversion;
  * `seed_rain`: default of `ESTAB_SEED_RAIN` of generated parameter sets;
  * `tables`: the BiomassSuccession data tables (LONGEVITY / SHADE_TOL / MATURITY seeds) shared with the other plugins.
"""
struct HBADContext
  priors::HBADPriors
  eco_gids::EcoGids
  fixed::HBADConst
  init_mode::Symbol
  estab_prob_table::Union{Nothing,Dict{Tuple{String,String},Float64}}
  estab_cycle_years::Float64
  seed_rain::Bool
  tables::BiomassSuccessionPlugin.BSTables
end

"""Load prob_estab_all_species.csv (cols species, l3, prob_estab) -> dict (nothing when the file is absent or `csv_path` is empty)."""
function load_estab_table(csv_path::AbstractString; cycle_years::Real=DEFAULT_CYCLE_YEARS)
  isempty(csv_path) && return nothing
  if !isfile(csv_path)
    @warn "HBAD: PROB_ESTAB table $csv_path not found - using the constant fallback $(DEFAULT_PROB_ESTAB_CYCLE) per cycle"
    return nothing
  end
  df = CSV.read(csv_path, DataFrame)
  d = Dict{Tuple{String,String},Float64}((uppercase(strip(String(r.species))), String(r.l3)) => Float64(r.prob_estab) for r in eachrow(df))
  @info "HBAD: loaded per-cycle PROB_ESTAB for $(length(d)) (species, L3) cells from $csv_path (annual p = 1-(1-p)^(1/$(Float64(cycle_years))))"
  return d
end

"""
    HBADContext(species_list, eco_species_ids; priors_csv, coefs_csv, gmax, hmax, verbose=true, fixed, init_mode, estab_prob_csv, cycle_years, seed_rain, tables)

Resolve the fixed priors of `species_list` (logging the mapping when `verbose`) and bundle them with the eco -> species table and the run's HBAD options.
"""
function HBADContext(species_list::AbstractVector{<:AbstractString}, eco_species_ids::AbstractVector;
                     priors_csv::AbstractString=DEFAULT_PRIORS_CSV, coefs_csv::AbstractString=DEFAULT_COEFS_CSV,
                     gmax::Float64=3.0, hmax::Float64=0.3, verbose::Bool=true,
                     fixed::HBADConst=HBADConst(), init_mode::Symbol=:jitter,
                     estab_prob_csv::AbstractString="", cycle_years::Real=DEFAULT_CYCLE_YEARS, seed_rain::Bool=false,
                     tables::BiomassSuccessionPlugin.BSTables=BiomassSuccessionPlugin.BSTables())
  priors = load_priors(species_list; priors_csv=isempty(priors_csv) ? DEFAULT_PRIORS_CSV : priors_csv,
                       coefs_csv=isempty(coefs_csv) ? DEFAULT_COEFS_CSV : coefs_csv, gmax=gmax, hmax=hmax)
  verbose && _log_priors(priors, priors_csv, coefs_csv)
  return HBADContext(priors, eco_gids(eco_species_ids), fixed, init_mode,
                     load_estab_table(estab_prob_csv; cycle_years=cycle_years), Float64(cycle_years), seed_rain, tables)
end

function _log_priors(pri::HBADPriors, priors_csv, coefs_csv)
  exact_t = count(i -> pri.trans_key[i] == pri.species_list[i], eachindex(pri.species_list))
  exact_p = count(i -> pri.prior_key[i] == pri.species_list[i], eachindex(pri.species_list))
  guard_g = [pri.species_list[i] => pri.growth_key[i] for i in eachindex(pri.species_list) if pri.growth_key[i] != pri.prior_key[i]]
  guard_m = [pri.species_list[i] => pri.mort_key[i] for i in eachindex(pri.species_list) if pri.mort_key[i] != pri.prior_key[i]]
  @info "HBAD priors: $(length(pri.species_list)) species; translation exact=$(exact_t) group-fallback=$(length(pri.species_list) - exact_t); priors exact=$(exact_p) group-fallback=$(length(pri.species_list) - exact_p)" priors_csv coefs_csv
  for (i, s) in enumerate(pri.species_list)
    @debug "HBAD map $s -> translation[$(pri.trans_key[i])] priors[$(pri.prior_key[i])] growth[$(pri.growth_key[i])] mort[$(pri.mort_key[i])] SDImax p95=$(pri.sp[i].sdi_p95)"
  end
  isempty(guard_g) || @info "HBAD sanity guard: growth block replaced by group row for" guard_g
  isempty(guard_m) || @info "HBAD sanity guard: mortality block replaced by group row for" guard_m
  @info "HBAD species -> keys" mapping = join(["$(s)=>t:$(pri.trans_key[i])/p:$(pri.prior_key[i])" for (i, s) in enumerate(pri.species_list)], "  ")
  return nothing
end

"""A copy of `ctx` with other field values (`with(ctx; fixed=..., tables=...)`)."""
with(ctx::HBADContext; kw...) = HBADContext((get(kw, f, getfield(ctx, f)) for f in fieldnames(HBADContext))...)

"""Throw unless `ctx` was built for exactly this species list and eco -> species table (the explicit replacement of the old silent re-configuration)."""
function check_context(ctx::HBADContext, species_list::AbstractVector{<:AbstractString}, eco_species_ids::AbstractVector)
  ctx.priors.species_list == species_list ||
    error("HBAD context was built for another species list ($(length(ctx.priors.species_list)) species, first $(first(ctx.priors.species_list, 3))) than the parameters' ($(length(species_list)) species): build one with HBADContext(species_list, eco_species_ids)")
  ctx.eco_gids == eco_gids(eco_species_ids) ||
    error("HBAD context was built for another eco -> species table than the parameters' ECO_SPECIES_IDS: build one with HBADContext(species_list, eco_species_ids)")
  return nothing
end

"""Prior of eco-local species `esp` in ecoregion `eco`."""
@inline prior_of(ctx::HBADContext, eco::Integer, esp::Integer) = @inbounds ctx.priors.sp[global_species(ctx.eco_gids, eco, esp)]
