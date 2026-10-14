# Succession-plugin interface. The fit loop and drivers dispatch on the succession plugin TYPE (obtained from the
# SoA's plugin tuple via `succession_plugin_of`), so a second succession model (e.g. PnET) only has to add methods
# to these generics. BiomassSuccession is currently the only implementation; each method simply delegates.

"""
    succession_plugin_of(soa) -> Type{<:AbstractPlugin}

The (single) non-`BaseSite` plugin type in the SoA's plugin tuple. Pure type computation (constant-folded).
"""
@generated function succession_plugin_of(::Type{<:SiteSoA{Ps}}) where {Ps<:Tuple}
  cand = [P for P in Ps.parameters if P !== BaseSitePlugin.BaseSite]
  length(cand) == 1 || error("expected exactly one succession plugin besides BaseSite, got $(cand)")
  return :($(cand[1]))
end
succession_plugin_of(soa::SiteSoA) = succession_plugin_of(typeof(soa))

# --- generic functions (methods per plugin below) ---
function params_type end
function make_param_dists end
function generate_params end
function generate_eco_params end
function make_ctx end
function step! end
function spinup! end
function add_cohort! end
function per_eco_groups end
function plant! end            # plant!(P, site, species_local, tph; kw...): add planted trees to one site (caller handles CSR capacity, see plant_sites!)
function plant_slot_needed end # plant_slot_needed(P, site, species_local, age): 1 if plant! appends a cohort, 0 if it merges
# Plugin-specific hooks with BiomassSuccession-neutral defaults:
function init_sites! end       # init_sites!(P, soa, site_attrs): per-site static attributes (BS: nothing needed)
function set_cohort_bio! end   # set_cohort_bio!(P, site, k, newbio): write a cohort's reported biomass (BS: plain assignment)

# Plugin type from a params struct (for tools that only hold params, not a SoA).
function succession_plugin_for end
# Builds plugin site attributes for a splots DataFrame (PnET: per-plot climate/lat/soil); set by `parametrize`.
const SITE_ATTRS_FN = Ref{Any}(sp -> nothing)
make_sites_like(P::Type, splots, eco_species_ids; kw...) = make_sites(splots, eco_species_ids; plugin=P, site_attrs=SITE_ATTRS_FN[](splots), kw...)

# --- BiomassSuccession ---
const _BS = BiomassSuccessionPlugin.BiomassSuccession

params_type(::Type{_BS}) = BiomassSuccessionPlugin.BiomassSuccessionParams
succession_plugin_for(::BiomassSuccessionPlugin.BiomassSuccessionParams) = _BS

make_param_dists(::Type{_BS}, args...; kw...) = BSP.make_biomass_param_dists(args...; kw...)

generate_params(::Type{_BS}, args...; kw...) = BiomassSuccessionPlugin.generate_biomass_params(args...; kw...)

generate_eco_params(::Type{_BS}, params) = BiomassSuccessionPlugin.generate_eco_params(params)

# Per-plugin ctx NamedTuple, keyed by plugin name (consumed by `step!`).
make_ctx(::Type{_BS}, eco_params) = (BiomassSuccession=(eco_params=eco_params,),)

step!(::Type{_BS}, soa, year::Integer, ctx) = PanCore.process_plugin!(soa, _BS, year; ctx=ctx.BiomassSuccession)

spinup!(::Type{_BS}, soa, spinup_cohorts, eco_params) = BiomassSuccessionPlugin.spinup_cohorts!(soa, spinup_cohorts, eco_params)

add_cohort!(::Type{_BS}, site, args...) = BiomassSuccessionPlugin.add_cohort!(site, args...)

per_eco_groups(::Type{_BS}) = BSP.BIOMASS_PER_ECO_GROUPS

# BiomassSuccession: no site attributes; external biomass writes are plain assignments (byte-identical to the former inline code).
init_sites!(::Type{_BS}, soa, site_attrs) = soa
@inline set_cohort_bio!(::Type{_BS}, site, k, newbio) = (site.c_bio[k] = newbio; nothing)

# BiomassSuccession planting: a new cohort of the species with the LANDIS initial biomass (calculate_initial_biomass: exp(-1.6 B/B_MAX_ECO) * ANPP_MAX_SPP,
# floor 2 g/m2) at `age` (default 1). `tph` is ignored (BiomassSuccession carries no stem density). Needs the ecoregion's resolved parameters
# (`eco_params=generate_eco_params(SP, params)[site.eco_id]`) for ANPP_MAX_SPP / B_MAX_ECO. A free cohort slot is the caller's job (`plant_sites!`).
function plant!(::Type{_BS}, site, species_local::Integer, tph::Real=0; eco_params, age::Real=1, kw...)
  nbio = BiomassSuccessionPlugin.calculate_initial_biomass(eco_params.ANPP_MAX_SPP[species_local], FloatType(site.B), eco_params.B_MAX_ECO)
  Int(site.live) < length(site.c_species) || error("BiomassSuccession plant!: site has no free cohort slot (capacity $(length(site.c_species))); use plant_sites!")
  BiomassSuccessionPlugin.add_cohort!(site, UIntType(species_local), FloatType(age), nbio)
  site.B += nbio
  return Int(site.live)
end
plant_slot_needed(::Type{_BS}, site, species_local::Integer, age::Real=1) = 1

# --- PnET ---
const _PN = PnETPlugin.PnET

params_type(::Type{_PN}) = PnETPlugin.PnETParams

make_param_dists(::Type{_PN}, args...; kw...) = PNP.make_pnet_param_dists(args...; kw...)

generate_params(::Type{_PN}, args...; kw...) = PnETPlugin.generate_pnet_params(args...; kw...)

generate_eco_params(::Type{_PN}, params) = PnETPlugin.generate_pnet_eco_params(params)

make_ctx(::Type{_PN}, eco_params) = begin
  clim = PnETPlugin.CLIMATE[]
  clim === nothing && error("PnET climate not loaded: PnETPlugin.configure!(climate_cache=..., co2_csv=...) (yaml pnet_climate_cache / pnet_co2_csv)")
  (PnET=(eco=eco_params, cst=PnETPlugin.FIXED[], clim=clim, scratch=PnETPlugin.scratch_pool()),)
end

step!(::Type{_PN}, soa, year::Integer, ctx) = PanCore.process_plugin!(soa, _PN, year; ctx=ctx.PnET)

spinup!(::Type{_PN}, soa, spinup_cohorts, eco_params) = error("PnET has no spinup path yet: run with spinup: false (cohorts start from the observed state)")

add_cohort!(::Type{_PN}, site, args...) = PnETPlugin.add_cohort!(site, args...)

plant!(::Type{_PN}, site, species_local::Integer, tph::Real; kw...) = error("PnET: plant! is not supported yet (no planting / establishment in the PnET plugin)")
plant_slot_needed(::Type{_PN}, site, species_local::Integer, age::Real=1) = error("PnET: plant! is not supported yet")

per_eco_groups(::Type{_PN}) = PNP.PNET_PER_ECO_GROUPS

init_sites!(::Type{_PN}, soa, site_attrs) = (site_attrs === nothing ? error("PnET needs site attributes (clim_id, cal_year0, lat, soil)") : PnETPlugin.init_sites!(soa, site_attrs))
@inline set_cohort_bio!(::Type{_PN}, site, k, newbio) = PnETPlugin.set_cohort_bio!(site, k, newbio)

succession_plugin_for(::PnETPlugin.PnETParams) = PnETPlugin.PnET

# --- HBAD (height / basal area / density) ---
const _HB = HBADPlugin.HBAD

params_type(::Type{_HB}) = HBADPlugin.HBADParams

make_param_dists(::Type{_HB}, args...; kw...) = HBP.make_hbad_param_dists(args...; kw...)

generate_params(::Type{_HB}, args...; kw...) = HBADPlugin.generate_hbad_params(args...; kw...)

generate_eco_params(::Type{_HB}, params) = HBADPlugin.generate_hbad_eco_params(params)

make_ctx(::Type{_HB}, eco_params) = (HBAD=(eco=eco_params, cst=HBADPlugin.FIXED[]),)

step!(::Type{_HB}, soa, year::Integer, ctx) = PanCore.process_plugin!(soa, _HB, year; ctx=ctx.HBAD)

spinup!(::Type{_HB}, soa, spinup_cohorts, eco_params) = error("HBAD has no spinup path: run with spinup: false (cohorts start from the observed state)")

add_cohort!(::Type{_HB}, site, args...) = HBADPlugin.add_cohort!(site, args...)

per_eco_groups(::Type{_HB}) = HBP.HBAD_PER_ECO_GROUPS

init_sites!(::Type{_HB}, soa, site_attrs) = soa
@inline set_cohort_bio!(::Type{_HB}, site, k, newbio) = HBADPlugin.set_cohort_bio!(site, k, newbio)

plant!(::Type{_HB}, site, species_local::Integer, tph::Real; eco_params=nothing, kw...) = HBADPlugin.plant!(site, species_local, tph; kw...)
plant_slot_needed(::Type{_HB}, site, species_local::Integer, age::Real=1) = HBADPlugin.plant_slot_needed(site, species_local, age)

# Tools that only hold params (post-hoc sims) get the fixed priors re-resolved for the params' species list when needed.
succession_plugin_for(p::HBADPlugin.HBADParams) = (HBADPlugin.ensure_configured!(p.SPECIES_LIST, p.ECO_SPECIES_IDS); HBADPlugin.HBAD)

"""
    plant_sites!(soa, plantings; age=1, eco_params=nothing, kw...)

Scenario planting harness for any succession plugin. `plantings` = iterable of `(site_index, species_local, tph)`. Reserves one CSR cohort slot per planting
(`_new_cohort_counts = live + #plantings of that site`, one `PanCore.readjust_soa!`, the same pattern as `_inject_observed_cohorts!`), then calls the plugin's
`plant!(SP, site, species_local, tph; age, kw...)` for each planting in order. `eco_params` (BiomassSuccession only) is the vector from `generate_eco_params`;
it is indexed by the site's `eco_id`. Inactive sites are skipped. Call it between annual steps (never inside one).
"""
function plant_sites!(soa, plantings; age::Real=1, eco_params=nothing, kw...)
  SP = succession_plugin_of(soa)
  cnt = soa.scalar._new_cohort_counts
  @inbounds for i in 1:soa.n
    cnt[i] = soa.scalar.live[i]
  end
  any_ = false
  for (si, _, _) in plantings
    soa.scalar.active[si] || continue
    cnt[si] += 1; any_ = true
  end
  any_ && PanCore.readjust_soa!(soa, (cohort=cnt,))
  for (si, sp, tph) in plantings
    site = getsite(soa, si)
    site.active || continue
    if eco_params === nothing
      plant!(SP, site, sp, tph; age=age, kw...)
    else
      plant!(SP, site, sp, tph; age=age, eco_params=eco_params[site.eco_id], kw...)
    end
  end
  return soa
end
