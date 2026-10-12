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

# --- BiomassSuccession ---
const _BS = BiomassSuccessionPlugin.BiomassSuccession

params_type(::Type{_BS}) = BiomassSuccessionPlugin.BiomassSuccessionParams

make_param_dists(::Type{_BS}, args...; kw...) = BSP.make_biomass_param_dists(args...; kw...)

generate_params(::Type{_BS}, args...; kw...) = BiomassSuccessionPlugin.generate_biomass_params(args...; kw...)

generate_eco_params(::Type{_BS}, params) = BiomassSuccessionPlugin.generate_eco_params(params)

# Per-plugin ctx NamedTuple, keyed by plugin name (consumed by `step!`).
make_ctx(::Type{_BS}, eco_params) = (BiomassSuccession=(eco_params=eco_params,),)

step!(::Type{_BS}, soa, year::Integer, ctx) = PanCore.process_plugin!(soa, _BS, year; ctx=ctx.BiomassSuccession)

spinup!(::Type{_BS}, soa, spinup_cohorts, eco_params) = BiomassSuccessionPlugin.spinup_cohorts!(soa, spinup_cohorts, eco_params)

add_cohort!(::Type{_BS}, site, args...) = BiomassSuccessionPlugin.add_cohort!(site, args...)

per_eco_groups(::Type{_BS}) = BSP.BIOMASS_PER_ECO_GROUPS
