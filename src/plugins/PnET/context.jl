# The immutable per-run context of the PnET plugin (Step 3d): what used to be the module-level FIXED / CLIMATE / SPECIES_SFTWD Refs and the global
# per-thread scratch pool. Built by `PnETContext(...)` (from a climate cache + yaml `pnet_generic`), carried in the `ctx` of `Pan.step!` (`make_ctx`)
# and passed explicitly to the generators, `add_cohort!`, `set_cohort_bio!` and `init_sites!`.

"""
    PnETContext

  * `fixed`: the non-fitted ecophysiology constants (`PnETConst`, yaml `pnet_generic` overrides applied);
  * `climate`: the derived PRISM climate + CO2 series (`PnETClimate`);
  * `species_sftwd`: FIA REF_SPECIES.SFTWD_HRDWD per species symbol, for plant-functional-type assignment (empty => genus heuristic);
  * `tables`: the BiomassSuccession data tables (LONGEVITY seed), shared with the other plugins;
  * `scratch`: per-thread working memory of the annual kernel, indexed by `Threads.threadid()`; owned by this context (shared by its `with_*` copies),
    never part of any result.
"""
struct PnETContext
  fixed::PnETConst
  climate::PnETClimate
  species_sftwd::Dict{String,String}
  tables::BiomassSuccessionPlugin.BSTables
  scratch::Vector{PnETScratch}
end
PnETContext(; fixed::PnETConst=PnETConst(), climate::PnETClimate, species_sftwd::AbstractDict=Dict{String,String}(),
            tables::BiomassSuccessionPlugin.BSTables=BiomassSuccessionPlugin.BSTables(), scratch::Vector{PnETScratch}=new_scratch_pool()) =
  PnETContext(fixed, climate, Dict{String,String}(species_sftwd), tables, scratch)

"""
    PnETContext(climate_cache, co2_csv; generic=Dict(), db_path=nothing, tables=BSTables())

Load the PRISM climate cache + CO2 series, apply the `generic` overrides to the constants and (if `db_path` is given) read REF_SPECIES.SFTWD_HRDWD.
"""
PnETContext(climate_cache::AbstractString, co2_csv::AbstractString; generic::AbstractDict=Dict{String,Any}(), db_path::Union{Nothing,AbstractString}=nothing,
            tables::BiomassSuccessionPlugin.BSTables=BiomassSuccessionPlugin.BSTables()) =
  PnETContext(; fixed=pnet_const(generic), climate=load_climate(climate_cache, co2_csv), species_sftwd=read_species_sftwd(db_path), tables=tables)

"""A copy of `ctx` with another climate / constants / tables (same scratch pool)."""
with_climate(ctx::PnETContext, climate::PnETClimate) = PnETContext(ctx.fixed, climate, ctx.species_sftwd, ctx.tables, ctx.scratch)
with_fixed(ctx::PnETContext, fixed::PnETConst) = PnETContext(fixed, ctx.climate, ctx.species_sftwd, ctx.tables, ctx.scratch)
with_tables(ctx::PnETContext, tables::BiomassSuccessionPlugin.BSTables) = PnETContext(ctx.fixed, ctx.climate, ctx.species_sftwd, tables, ctx.scratch)

"""Fresh PnET parameters under the context's species table and BS tables."""
generate_pnet_params(ctx::PnETContext, args...; kw...) = generate_pnet_params(ctx.tables, ctx.species_sftwd, args...; kw...)
"""Per-ecoregion resolved parameters under the context's constants."""
generate_pnet_eco_params(ctx::PnETContext, params::PnETParams) = generate_pnet_eco_params(params, ctx.fixed)
