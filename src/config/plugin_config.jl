succession_plugin_type(name::AbstractString) = lowercase(name) in ("biomass", "biomasssuccession", "bs") ? _BS :
                                                lowercase(name) in ("pnet", "pnet-succession") ? _PN :
                                                lowercase(name) in ("hbad", "hbad-succession") ? _HB :
                                                lowercase(name) in ("hbad2", "hbad-v2") ? _H2 :
                                                lowercase(name) in ("hbad3p", "hbad-3p") ? _H3 :
                                                error("unknown succession_plugin '$name' (biomass | pnet | hbad | hbad2 | hbad3p)")

"""
    bs_tables_from_cfg(cfg, db_path) -> BSTables

The Biomass Succession data tables of a run (pinned LONGEVITY / SHADE_TOL, seeds for PROB_ESTAB / MATURITY, B_MAX / ANPP / PROB_ESTAB floors), loaded from the
config's csv paths and the cohort database. Pure function of its inputs: nothing is stored in module state, the caller passes the result on explicitly.
"""
function bs_tables_from_cfg(cfg0::AbstractDict, db_path::AbstractString)
  cfg = _as_config(cfg0)
  longevity = cfg["longevity_from_data"] ? _load_longevity_table(db_path) : nothing
  # Per-species LONGEVITY overrides from yaml (`longevity_overrides: {PIEL: 275}`) - applied on top of the
  # data table so hand corrections don't require rewriting the read-only species_longevity_ref DB table.
  let ov = cfg["longevity_overrides"]
    if !isnothing(ov) && !isempty(ov)
      tbl = longevity === nothing ? Dict{String,Float64}() : longevity
      for (k, v) in ov; tbl[String(k)] = Float64(v); end
      longevity = tbl
      @info "longevity_overrides applied: $(Dict(String(k)=>Float64(v) for (k,v) in ov))"
    end
  end
  return BiomassSuccessionPlugin.BSTables(;
    fixed_longevity=(let v = cfg["fix_longevity"]; isnothing(v) ? nothing : Float64(v) end),
    longevity=longevity,
    shade_tol=cfg["shade_tol_from_data"] ? _load_shade_tol_table(cfg["shade_tol_csv"]) : nothing,
    prob_estab=cfg["prob_estab_from_data"] ? _load_prob_estab_table(cfg["prob_estab_csv"], cfg["prob_estab_csv_artificial"]) : nothing,
    maturity=cfg["maturity_from_data"] ? _load_maturity_table(cfg["maturity_csv"]) : nothing,
    bmax_floor=cfg["bmax_floor_from_data"] ? _load_bmax_floor_table(cfg["bmax_floor_csv"]) : nothing,
    anpp_floor=cfg["anpp_floor_from_data"] ? _load_anpp_floor_table(cfg["anpp_floor_csv"]) : nothing)
end

"""
    plugin_config(succession_plugin, species_list, eco_species_ids, eco_list; tables, cohorts_db_path, tablename, pnet_*, hbad_*) -> PluginConfig

Build the immutable plugin context of a run (`BSTables` | `PnETContext` | `HBADContext`) and the site-attribute builder (PnET: per-plot climate id / latitude / soil
from the cohort database). The ONE place `parametrize` and `setup_plugin_from_cfg` construct a plugin; nothing is written to module state.
"""
function plugin_config(succession_plugin::AbstractString, species_list, eco_species_ids, eco_list::Vector{String};
    tables::BiomassSuccessionPlugin.BSTables=BiomassSuccessionPlugin.BSTables(), cohorts_db_path::AbstractString="", tablename::AbstractString="",
    pnet_climate_cache::AbstractString="", pnet_co2_csv::AbstractString="", pnet_generic::AbstractDict=Dict{String,Any}(),
    pnet_soil_texture::AbstractDict=Dict{String,String}(), pnet_soil_csv::AbstractString="",
    hbad_priors_csv::AbstractString="", hbad_coefs_csv::AbstractString="", hbad_init::AbstractString="jitter", hbad_seed_rain::Bool=false,
    hbad_prob_estab_csv::AbstractString="", hbad_cycle_years::Real=8,
    hbad2_params_dir::AbstractString="", hbad2_vigour::Bool=true,
    hbad3p_params_dir::AbstractString="", hbad3p_vigour::Bool=true)
  SUCC = succession_plugin_type(succession_plugin)
  if SUCC === PnETPlugin.PnET
    (isempty(pnet_climate_cache) || isempty(pnet_co2_csv)) && error("succession_plugin=pnet requires pnet_climate_cache and pnet_co2_csv")
    ctx = PnETPlugin.PnETContext(pnet_climate_cache, pnet_co2_csv; generic=pnet_generic, db_path=cohorts_db_path, tables=tables)
    @info "PnET succession: climate cache $(basename(pnet_climate_cache)) ($(length(ctx.climate.ids)) L4s, $(ctx.climate.y1)-$(ctx.climate.y2)); fixed constants" ctx.fixed
    soil_map = Dict{String,String}(String(k) => String(v) for (k, v) in pnet_soil_texture)
    return PluginConfig(ctx; site_attrs=sp -> PnETPlugin.build_site_attrs(ctx.climate, cohorts_db_path, tablename, sp, eco_list; soil_map=soil_map, soil_csv=pnet_soil_csv))
  elseif SUCC === HBADPlugin.HBAD
    (species_list === nothing || eco_species_ids === nothing) && error("the hbad plugin needs species_list and eco_species_ids")
    return PluginConfig(HBADPlugin.HBADContext(species_list, eco_species_ids; priors_csv=hbad_priors_csv, coefs_csv=hbad_coefs_csv,
      init_mode=Symbol(lowercase(hbad_init)), estab_prob_csv=hbad_prob_estab_csv, cycle_years=hbad_cycle_years, seed_rain=hbad_seed_rain, tables=tables))
  elseif SUCC === HBAD2Plugin.HBAD2
    (species_list === nothing || eco_species_ids === nothing) && error("the hbad2 plugin needs species_list and eco_species_ids")
    return PluginConfig(HBAD2Plugin.HBAD2Context(species_list, eco_species_ids; params_dir=hbad2_params_dir, priors_csv=hbad_priors_csv, coefs_csv=hbad_coefs_csv,
      vigour=hbad2_vigour, tables=tables))
  elseif SUCC === HBAD3PPlugin.HBAD3P
    (species_list === nothing || eco_species_ids === nothing) && error("the hbad3p plugin needs species_list and eco_species_ids")
    return PluginConfig(HBAD3PPlugin.HBAD3PContext(species_list, eco_species_ids; params_dir=hbad3p_params_dir, priors_csv=hbad_priors_csv, coefs_csv=hbad_coefs_csv,
      vigour=hbad3p_vigour, tables=tables))
  end
  return PluginConfig(tables)
end

# Tools/diagnostics (tools/tost_common.jl, calib_coord.jl): select and configure the succession plugin from a yaml cfg
# exactly as `parametrize` does. Returns the `PluginConfig` (pass it to `make_sites`, `fit_params`, `generate_params`, ...).
function setup_plugin_from_cfg(cfg0::AbstractDict, eco_list::Vector{String}; species_list=nothing, eco_species_ids=nothing,
                               tables::BiomassSuccessionPlugin.BSTables=BiomassSuccessionPlugin.BSTables())
  cfg = _as_config(cfg0)
  return plugin_config(cfg["succession_plugin"], species_list, eco_species_ids, eco_list; tables=tables,
    cohorts_db_path=cfg["cohorts_db_path"], tablename=cfg["tablename"],
    pnet_climate_cache=cfg["pnet_climate_cache"], pnet_co2_csv=cfg["pnet_co2_csv"], pnet_generic=Dict{String,Any}(cfg["pnet_generic"]),
    pnet_soil_texture=cfg["pnet_soil_texture"], pnet_soil_csv=cfg["pnet_soil_csv"],
    hbad_priors_csv=cfg["hbad_priors_csv"], hbad_coefs_csv=cfg["hbad_coefs_csv"], hbad_init=cfg["hbad_init"], hbad_seed_rain=cfg["hbad_seed_rain"],
    hbad_prob_estab_csv=cfg["hbad_prob_estab_csv"], hbad_cycle_years=cfg["cycle_years"],
    hbad2_params_dir=cfg["hbad2_params_dir"], hbad2_vigour=cfg["hbad2_vigour"],
    hbad3p_params_dir=cfg["hbad3p_params_dir"], hbad3p_vigour=cfg["hbad3p_vigour"])
end
