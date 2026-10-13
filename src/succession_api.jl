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

# Tools that only hold params (post-hoc sims) get the fixed priors re-resolved for the params' species list when needed.
succession_plugin_for(p::HBADPlugin.HBADParams) = (HBADPlugin.ensure_configured!(p.SPECIES_LIST, p.ECO_SPECIES_IDS); HBADPlugin.HBAD)
