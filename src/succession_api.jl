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
# Two kinds of first argument: the plugin TYPE (`Type{P}`: pure type-level facts and the annual step, dispatched on the SoA's plugin tuple) and the plugin
# CONTEXT `pc` (an immutable per-run object that holds the plugin's data: `BSTables` / `HBADContext` / `PnETContext`; its concrete type selects the
# method statically, so kernels never dispatch dynamically). `PluginConfig` bundles a context with the site-attribute builder of the run.
function params_type end
function make_param_dists end   # make_param_dists(pc, n_species, n_eco, eco_species_ids; ...)
function generate_params end    # generate_params(pc, species_list, eco_list, eco_species_ids; rng, no_establishment)
function generate_eco_params end # generate_eco_params(pc, params)
function make_ctx end           # make_ctx(pc, eco_params) -> the NamedTuple `step!` consumes
function step! end              # step!(P, soa, year, ctx)
function spinup! end
function add_cohort! end        # add_cohort!(pc, site, species_local, age, agb)
function per_eco_groups end
function plant! end            # plant!(pc, site, species_local, tph; kw...): add planted trees to one site (caller handles CSR capacity, see plant_sites!)
function plant_slot_needed end # plant_slot_needed(pc, site, species_local, age): 1 if plant! appends a cohort, 0 if it merges
function plant_slots_reserved end # plant_slots_reserved(pc): CSR slots `plant_sites!` reserves per planting (1 for every plugin but HBAD2: its vigour sub-cohorts)
# Plugin-specific hooks with BiomassSuccession-neutral defaults:
function init_sites! end       # init_sites!(pc, soa, site_attrs): per-site static attributes (BS: nothing needed)
function set_cohort_bio! end   # set_cohort_bio!(pc, site, k, newbio): write a cohort's reported biomass (BS: plain assignment)

# Plugin type from a params struct (for tools that only hold params, not a SoA).
function succession_plugin_for end

"""
    plugin_type(pc) -> Type{<:AbstractPlugin}

The succession plugin a context (or `PluginConfig`) belongs to. A pure type-level map (constant-folded).
"""
function plugin_type end

"""
    PluginConfig(ctx; site_attrs = splots -> nothing)

The succession plugin of a run: its immutable context `ctx` (`BSTables` | `HBADContext` | `PnETContext`) and the builder of the plugin's site attributes
for a splots table (PnET: climate id / latitude / soil per plot; others: `nothing`). Replaces the old `ACTIVE_SUCCESSION` / `SITE_ATTRS_FN` module Refs:
`setup_plugin_from_cfg` / `parametrize` build one and pass it to the drivers, `make_sites`, `fit_params` and the tools. The plugin type `P` is a type
parameter, so everything that takes a `PluginConfig` specialises on it.
"""
struct PluginConfig{P,C,F}
  ctx::C
  site_attrs::F
end
_no_site_attrs(splots) = nothing
PluginConfig(ctx; site_attrs=_no_site_attrs) = PluginConfig{plugin_type(ctx),typeof(ctx),typeof(site_attrs)}(ctx, site_attrs)
plugin_type(::PluginConfig{P}) where {P} = P
bs_tables(plugin::PluginConfig) = bs_tables(plugin.ctx)

"""The Biomass Succession data tables carried by a plugin context (every plugin's context holds them: LONGEVITY / SHADE_TOL / MATURITY seeds, floors)."""
function bs_tables end

"""The Biomass Succession plugin config (plain `make_sites` / analysis scripts that work on the default plugin), with `tables` (default: no pins, no floors)."""
bs_plugin(tables::BiomassSuccessionPlugin.BSTables=BiomassSuccessionPlugin.BSTables()) = PluginConfig(tables)

make_sites_like(plugin::PluginConfig, splots, eco_species_ids; kw...) = make_sites(splots, eco_species_ids; plugin=plugin, kw...)

plant_slots_reserved(pc) = 1

# --- BiomassSuccession ---
const _BS = BiomassSuccessionPlugin.BiomassSuccession

params_type(::Type{_BS}) = BiomassSuccessionPlugin.BiomassSuccessionParams
succession_plugin_for(::BiomassSuccessionPlugin.BiomassSuccessionParams) = _BS

const _BSTables = BiomassSuccessionPlugin.BSTables
plugin_type(::_BSTables) = _BS
bs_tables(pc::_BSTables) = pc

make_param_dists(pc::_BSTables, args...; kw...) = BSP.make_biomass_param_dists(pc, args...; kw...)

generate_params(pc::_BSTables, args...; kw...) = BiomassSuccessionPlugin.generate_biomass_params(pc, args...; kw...)

generate_eco_params(::_BSTables, params) = BiomassSuccessionPlugin.generate_eco_params(params)

# Per-plugin ctx NamedTuple, keyed by plugin name (consumed by `step!`).
make_ctx(::_BSTables, eco_params) = (BiomassSuccession=(eco_params=eco_params,),)

step!(::Type{_BS}, soa, year::Integer, ctx) = PanCore.process_plugin!(soa, _BS, year; ctx=ctx.BiomassSuccession)

spinup!(::Type{_BS}, soa, spinup_cohorts, eco_params) = BiomassSuccessionPlugin.spinup_cohorts!(soa, spinup_cohorts, eco_params)

add_cohort!(::_BSTables, site, args...) = BiomassSuccessionPlugin.add_cohort!(site, args...)

per_eco_groups(::Type{_BS}) = BSP.BIOMASS_PER_ECO_GROUPS

# BiomassSuccession: no site attributes; external biomass writes are plain assignments (byte-identical to the former inline code).
init_sites!(::_BSTables, soa, site_attrs) = soa
@inline set_cohort_bio!(::_BSTables, site, k, newbio) = (site.c_bio[k] = newbio; nothing)

# BiomassSuccession planting: a new cohort of the species with the LANDIS initial biomass (calculate_initial_biomass: exp(-1.6 B/B_MAX_ECO) * ANPP_MAX_SPP,
# floor 2 g/m2) at `age` (default 1). `tph` is ignored (BiomassSuccession carries no stem density). Needs the ecoregion's resolved parameters
# (`eco_params=generate_eco_params(SP, params)[site.eco_id]`) for ANPP_MAX_SPP / B_MAX_ECO. A free cohort slot is the caller's job (`plant_sites!`).
function plant!(::_BSTables, site, species_local::Integer, tph::Real=0; eco_params, age::Real=1, kw...)
  nbio = BiomassSuccessionPlugin.calculate_initial_biomass(eco_params.ANPP_MAX_SPP[species_local], FloatType(site.B), eco_params.B_MAX_ECO)
  Int(site.live) < length(site.c_species) || error("BiomassSuccession plant!: site has no free cohort slot (capacity $(length(site.c_species))); use plant_sites!")
  BiomassSuccessionPlugin.add_cohort!(site, UIntType(species_local), FloatType(age), nbio)
  site.B += nbio
  return Int(site.live)
end
plant_slot_needed(::_BSTables, site, species_local::Integer, age::Real=1) = 1

# --- PnET ---
const _PN = PnETPlugin.PnET

params_type(::Type{_PN}) = PnETPlugin.PnETParams

const _PNContext = PnETPlugin.PnETContext
plugin_type(::_PNContext) = _PN
bs_tables(pc::_PNContext) = pc.tables

make_param_dists(::_PNContext, args...; kw...) = PNP.make_pnet_param_dists(args...; kw...)

generate_params(pc::_PNContext, args...; kw...) = PnETPlugin.generate_pnet_params(pc, args...; kw...)

generate_eco_params(pc::_PNContext, params) = PnETPlugin.generate_pnet_eco_params(pc, params)

make_ctx(pc::_PNContext, eco_params) = (PnET=(eco=eco_params, cst=pc.fixed, clim=pc.climate, scratch=pc.scratch),)

step!(::Type{_PN}, soa, year::Integer, ctx) = PanCore.process_plugin!(soa, _PN, year; ctx=ctx.PnET)

spinup!(::Type{_PN}, soa, spinup_cohorts, eco_params) = error("PnET has no spinup path yet: run with spinup: false (cohorts start from the observed state)")

add_cohort!(pc::_PNContext, site, args...) = PnETPlugin.add_cohort!(site, args...; cst=pc.fixed)

plant!(::_PNContext, site, species_local::Integer, tph::Real; kw...) = error("PnET: plant! is not supported yet (no planting / establishment in the PnET plugin)")
plant_slot_needed(::_PNContext, site, species_local::Integer, age::Real=1) = error("PnET: plant! is not supported yet")

per_eco_groups(::Type{_PN}) = PNP.PNET_PER_ECO_GROUPS

init_sites!(pc::_PNContext, soa, site_attrs) = (site_attrs === nothing ? error("PnET needs site attributes (clim_id, cal_year0, lat, soil)") : PnETPlugin.init_sites!(soa, site_attrs, pc.fixed))
@inline set_cohort_bio!(::_PNContext, site, k, newbio) = PnETPlugin.set_cohort_bio!(site, k, newbio)

succession_plugin_for(::PnETPlugin.PnETParams) = PnETPlugin.PnET

# --- HBAD (height / basal area / density) ---
const _HB = HBADPlugin.HBAD

params_type(::Type{_HB}) = HBADPlugin.HBADParams

const _HBContext = HBADPlugin.HBADContext
plugin_type(::_HBContext) = _HB
bs_tables(pc::_HBContext) = pc.tables

make_param_dists(::_HBContext, args...; kw...) = HBP.make_hbad_param_dists(args...; kw...)

generate_params(pc::_HBContext, args...; kw...) = HBADPlugin.generate_hbad_params(pc, args...; kw...)

generate_eco_params(pc::_HBContext, params) = HBADPlugin.generate_hbad_eco_params(pc, params)

make_ctx(pc::_HBContext, eco_params) = (HBAD=(eco=eco_params, cst=pc.fixed),)

step!(::Type{_HB}, soa, year::Integer, ctx) = PanCore.process_plugin!(soa, _HB, year; ctx=ctx.HBAD)

spinup!(::Type{_HB}, soa, spinup_cohorts, eco_params) = error("HBAD has no spinup path: run with spinup: false (cohorts start from the observed state)")

add_cohort!(pc::_HBContext, site, args...) = HBADPlugin.add_cohort!(pc, site, args...)

per_eco_groups(::Type{_HB}) = HBP.HBAD_PER_ECO_GROUPS

init_sites!(::_HBContext, soa, site_attrs) = soa
@inline set_cohort_bio!(pc::_HBContext, site, k, newbio) = HBADPlugin.set_cohort_bio!(pc, site, k, newbio)

plant!(pc::_HBContext, site, species_local::Integer, tph::Real; eco_params=nothing, kw...) = HBADPlugin.plant!(pc, site, species_local, tph; kw...)
plant_slot_needed(::_HBContext, site, species_local::Integer, age::Real=1) = HBADPlugin.plant_slot_needed(site, species_local, age)

succession_plugin_for(::HBADPlugin.HBADParams) = HBADPlugin.HBAD

# --- HBAD2 (HBAD v2: biomass-free process-based cohort model; stage 1 = fixed v2g parameters) ---
const _H2 = HBAD2Plugin.HBAD2

params_type(::Type{_H2}) = HBAD2Plugin.HBAD2Params

const _H2Context = HBAD2Plugin.HBAD2Context
plugin_type(::_H2Context) = _H2
bs_tables(pc::_H2Context) = pc.tables

make_param_dists(::_H2Context, args...; kw...) = error("hbad2 is calibrated by tools/hbad2 (stage 1: fixed parameters)")

generate_params(pc::_H2Context, args...; kw...) = HBAD2Plugin.generate_hbad2_params(pc, args...; kw...)

generate_eco_params(pc::_H2Context, params) = HBAD2Plugin.generate_hbad2_eco_params(pc, params)

make_ctx(pc::_H2Context, eco_params) = (HBAD2=(eco=eco_params, cst=pc.fixed),)

step!(::Type{_H2}, soa, year::Integer, ctx) = PanCore.process_plugin!(soa, _H2, year; ctx=ctx.HBAD2)

spinup!(::Type{_H2}, soa, spinup_cohorts, eco_params) = error("HBAD2 has no spinup path: run with spinup: false (cohorts start from the observed state)")

add_cohort!(pc::_H2Context, site, args...) = HBAD2Plugin.add_cohort!(pc, site, args...)

per_eco_groups(::Type{_H2}) = Set{Symbol}()

init_sites!(::_H2Context, soa, site_attrs) = soa
@inline set_cohort_bio!(pc::_H2Context, site, k, newbio) = HBAD2Plugin.set_cohort_bio!(pc, site, k, newbio)

plant!(pc::_H2Context, site, species_local::Integer, tph::Real; eco_params=nothing, kw...) = HBAD2Plugin.plant!(pc, site, species_local, tph; kw...)
plant_slot_needed(pc::_H2Context, site, species_local::Integer, age::Real=1) = HBAD2Plugin.plant_slot_needed(pc, site, species_local, age)
plant_slots_reserved(pc::_H2Context) = HBAD2Plugin.plant_slots_reserved(pc.fixed)

succession_plugin_for(::HBAD2Plugin.HBAD2Params) = HBAD2Plugin.HBAD2

"""
    plant_sites!(pc, soa, plantings; age=1, eco_params=nothing, kw...)

Scenario planting harness for any succession plugin. `plantings` = iterable of `(site_index, species_local, tph)`. Reserves one CSR cohort slot per planting
(`_new_cohort_counts = live + #plantings of that site`, one `PanCore.readjust_soa!`, the same pattern as `_inject_observed_cohorts!`), then calls the plugin's
`plant!(SP, site, species_local, tph; age, kw...)` for each planting in order. `eco_params` (BiomassSuccession only) is the vector from `generate_eco_params`;
it is indexed by the site's `eco_id`. Inactive sites are skipped. Call it between annual steps (never inside one).
"""
function plant_sites!(pc, soa, plantings; age::Real=1, eco_params=nothing, kw...)
  cnt = soa.scalar._new_cohort_counts
  @inbounds for i in 1:soa.n
    cnt[i] = soa.scalar.live[i]
  end
  any_ = false
  for (si, _, _) in plantings
    soa.scalar.active[si] || continue
    cnt[si] += plant_slots_reserved(pc); any_ = true
  end
  any_ && PanCore.readjust_soa!(soa, (cohort=cnt,))
  for (si, sp, tph) in plantings
    site = getsite(soa, si)
    site.active || continue
    if eco_params === nothing
      plant!(pc, site, sp, tph; age=age, kw...)
    else
      plant!(pc, site, sp, tph; age=age, eco_params=eco_params[site.eco_id], kw...)
    end
  end
  return soa
end
