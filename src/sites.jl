# SoA type for a given succession plugin P (BaseSite + P). For BiomassSuccession this is exactly `ActiveSoA`.
soa_type(::Type{P}) where {P} = SiteSoA{Tuple{BaseSitePlugin.BaseSite,P}}

# Plugin-specific per-site initialisation inside make_sites (BiomassSuccession: the statements previously inlined there, same order).
function _init_site_plugin!(::Type{_BS}, site, no_establishment::Bool)
  site.B = zero(FloatType)
  site.AGNPP = zero(FloatType)
  site.harvestCapacityReduction = zero(FloatType)
  site.growthReduction = zero(FloatType)
  site.prevYearMortality = zero(FloatType)
  site.shade_class = one(UIntType)
  site.c_species .= zero(UIntType)
  site.c_age .= zero(FloatType)
  site.c_bio .= zero(FloatType)
  site.c_m_tot .= zero(FloatType)
  site.c_comp .= zero(FloatType)
  site.no_establish = no_establishment
  site.sp_mature .= false
  site.sp_sprout .= false
  site.sp_seed .= false
  site.sp_serotiny .= false
  site.sp_plant .= false
  return nothing
end
function _init_site_plugin!(::Type{_PN}, site, no_establishment::Bool)
  site.B = zero(FloatType)
  site.c_species .= zero(UIntType)
  site.c_age .= zero(FloatType)
  site.c_bio .= zero(FloatType)
  site.c_tbio .= zero(FloatType)
  site.sp_mature .= false
  return nothing
end

function _init_site_plugin!(::Type{_HB}, site, no_establishment::Bool)
  site.B = zero(FloatType)
  site.BA = zero(FloatType)
  site.SDI = zero(FloatType)
  site.RD = zero(FloatType)
  site.thin_tph = zero(FloatType)
  site.shade_class = zero(UIntType)
  site.no_establish = no_establishment
  site.c_species .= zero(UIntType)
  site.c_age .= zero(FloatType)
  site.c_bio .= zero(FloatType)
  site.c_tph .= zero(FloatType)
  site.c_dq .= zero(FloatType)
  site.c_h .= zero(FloatType)
  site.c_bal .= zero(FloatType)
  site.sp_mature .= false
  site.sp_estab .= false
  return nothing
end

"""
    make_sites(splots, eco_species_ids; plugin=bs_plugin(), rng, spinup, no_establishment, site_attrs=nothing) -> SiteSoA

Initial SoA of a run for the succession plugin of `plugin` (a `PluginConfig`). `site_attrs` (PnET: per-plot climate/soil table) defaults to
`plugin.site_attrs(splots)`.
"""
make_sites(splots::DataFrame, eco_species_ids::Vector{Vector{Int}}; plugin::PluginConfig=bs_plugin(), kw...) =
  _make_sites(plugin, splots, eco_species_ids; kw...)

function _make_sites(plugin::PluginConfig{P}, splots::DataFrame, eco_species_ids::Vector{Vector{Int}}; rng::Random.AbstractRNG, spinup::Bool=true, no_establishment::Bool=false,
                     site_attrs=nothing) where {P}
  pc = plugin.ctx
  site_attrs === nothing && (site_attrs = plugin.site_attrs(splots))
  #eco_ids = unique(select(splots, [:eco_id]))
  #n_species = maximum(splots.species_id)
  #plot_ids = unique(select(splots, [:plot_id]))
  #plot_eco_ids = unique(select(splots, [:plot_id, :eco_id]))
  #plot_eco_ids = sort!(plot_eco_ids, [:plot_id])
  df = unique(select(splots, [:plot_id, :eco_id]))
  n = nrow(df)
  #plot_eco_ids = sort!(plot_eco_ids, [:plot_id])
  #@assert (nrow(plot_ids) == nrow(plot_eco_ids)) "Error: plot_ids and plot_eco_ids are not of equal length!"

  #index = eco_id * plot_id

  #splots_dict::Dict{Int64,DataFrame} = Dict(
  #    plt_key.plt_cn => DataFrame(plt_df)
  #    for (plt_key, plt_df) in pairs(groupby(splots, :plt_cn, sort=false))
  #)
  species_counts = [Int32(length(eco_species_ids[row.eco_id])) for row in eachrow(df)]
  if spinup
    cohort_counts = fill(Int32(2), n)
    @debug cohort_counts
    soa = soa_type(P)((cohort=cohort_counts, species=species_counts))
    base_seed = rand(rng, UInt64)   # per-site RNG = RNGType(hash((base_seed, i))) below → thread-count-invariant & reproducible (was seeded from per-thread RNGs, which made results depend on --threads)
    Threads.@threads :static for i in 1:nrow(df)
      @inbounds begin
        site = getsite(soa, i)
        site.active = false
        site.rng = RNGType(hash((base_seed, i)))
        site.ecocode = UIntType(df.eco_id[i])# no ecocode coming from raster, relying on eco_id
        site.eco_id = df.eco_id[i]
        site.mapcode = UIntType(df.plot_id[i]) # for parametrization, plot_id is global index, no raster
        site.ref_cn = UIntType(df.plot_id[i])
        site.old = zero(UIntType)
        site.live = zero(UIntType)
        _init_site_plugin!(P, site, no_establishment)
      end
    end
    init_sites!(pc, soa, site_attrs)
    return soa
  else
    splots_dict = Dict(
      (plt_key.plot_id, plt_key.eco_id) => begin
        plt_df = sort!(DataFrame(plt_df), [:age_calc, :eco_species_id, :agb_sum], rev=[true, false, false])   # B32 follow-up: full deterministic tie-break (equal-age cohorts no longer keep the input order)
        plt_df.sim_year .= Dates.value.(Dates.Day.(plt_df.measdate - plt_df.start_measdate)) ./ 365.25 .|> round .|> Int
        Data.get_initial_cohorts(plt_df)
      end
      for (plt_key, plt_df) in pairs(groupby(splots, [:plot_id, :eco_id], sort=false))
    )
    df_keys = sort!(collect(keys(splots_dict)))

    cohort_counts = [Int32(nrow(splots_dict[key])) for key in df_keys]
    soa = soa_type(P)((cohort=cohort_counts, species=species_counts))
    base_seed = rand(rng, UInt64)   # per-site RNG = RNGType(hash((base_seed, i))) below → thread-count-invariant & reproducible (was seeded from per-thread RNGs, which made results depend on --threads)
    Threads.@threads :static for i in 1:length(df_keys)
      @inbounds begin
        key = df_keys[i]
        plt_df = splots_dict[key]
        (plot_id, eco_id) = key
        initial_cohorts = plt_df
        #n_cohorts = nrow(initial_cohorts)
        #cap = UIntType(2^ceil(log2(n_cohorts)))
        site = getsite(soa, i)
        site.active = true
        site.rng = RNGType(hash((base_seed, i)))
        site.ecocode = UIntType(eco_id) #UIntType(getproperty(p, ecocode_field)),
        site.eco_id = eco_id
        site.mapcode = UIntType(plot_id)
        site.ref_cn = plot_id
        site.old = zero(UIntType)
        site.live = zero(UIntType)
        _init_site_plugin!(P, site, no_establishment)
        for row in eachrow(initial_cohorts)
          add_cohort!(pc, site, UIntType(row.eco_species_id), FloatType(row.age_calc), FloatType(row.agb_sum))
        end
      end
    end
    init_sites!(pc, soa, site_attrs)
    return soa
  end
end
