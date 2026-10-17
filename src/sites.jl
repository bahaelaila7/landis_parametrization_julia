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

function _init_site_plugin!(::Type{_H2}, site, no_establishment::Bool)
  site.B = zero(FloatType)
  site.BA = zero(FloatType)
  site.SDI = zero(FloatType)
  site.RD = zero(FloatType)
  site.ryear = Int32(0)
  site.planted = false
  site.no_establish = no_establishment
  site.c_species .= zero(UIntType)
  site.c_age .= 0.0
  site.c_tph .= 0.0
  site.c_dq .= 0.0
  site.c_h .= 0.0
  site.c_zq .= 0.0
  site.c_bio .= zero(FloatType)
  site.c_O .= 0.0
  site.c_SDI .= 0.0
  site.c_V .= 0.0
  site.c_ropen .= Int32(0)
  site.sp_mature .= false
  site.sp_estab .= false
  site.sp_rate .= 0.0
  site.sp_ba .= 0.0
  site.sp_oy .= Int32(0)
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
  df = unique(select(splots, [:plot_id, :eco_id]))
  n = nrow(df)


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

"""
    make_sites_hbad2(plugin, tc::Data.TreeCohorts; rng, no_establishment=false, eco_id=1, planted=nothing) -> SiteSoA

Initial HBAD2 SoA of tree-based sites (`Data.load_tree_cohorts`): one active site per subplot site of `tc`, one ecoregion `eco_id` whose species are all species of the
plugin context in order (the cohort's key index IS the eco-local species index, so `plugin.ctx` must have been built with `tc.keys` as its species list). See `make_sites_hbad2_arrays`.
"""
function make_sites_hbad2(plugin::PluginConfig{P}, tc::Data.TreeCohorts; kw...) where {P}
  plugin.ctx.priors.species_list == tc.keys || error("make_sites_hbad2: the plugin context's species list must equal the tree cohorts' keys")
  return make_sites_hbad2_arrays(plugin, tc.off, tc.key, tc.A, tc.N, tc.D, tc.H, tc.zq; kw...)
end

"""
    make_sites_hbad2_arrays(plugin, off, species, age, N, D, H, zq; rng, no_establishment=false, eco_id=1, planted=nothing) -> SiteSoA

HBAD2 SoA from per-site cohort arrays (CSR `off`, 1-based, `off[end]` = total + 1; `species` = eco-local species index). State is copied exactly (Float64); `c_bio` is the reported
allometric AGB; `planted` (Bool per site) sets the planted-stand flag; `eco_id` is one ecoregion for all sites or one per site (the eco-local species index must be the key index in
every ecoregion, i.e. all ecoregions list all species in order); `mapcode` (default the site index) is the caller's plot id (the scenario engine groups the subplot sites of a plot by it). Per-site RNG as in `make_sites` (unused by the deterministic kernel). Empty sites get capacity 1.
"""
function make_sites_hbad2_arrays(plugin::PluginConfig{P}, off::AbstractVector{<:Integer}, species::AbstractVector{<:Integer}, age::AbstractVector{<:Real}, N::AbstractVector{<:Real},
                                 D::AbstractVector{<:Real}, H::AbstractVector{<:Real}, zq::AbstractVector{<:Real};
                                 rng::Random.AbstractRNG, no_establishment::Bool=false, eco_id::Union{Integer,AbstractVector{<:Integer}}=1,
                                 planted::Union{Nothing,AbstractVector{Bool}}=nothing, mapcode::Union{Nothing,AbstractVector{<:Integer}}=nothing) where {P}
  P === HBAD2Plugin.HBAD2 || error("make_sites_hbad2 needs a succession_plugin=hbad2 PluginConfig")
  pc = plugin.ctx
  ns = length(off) - 1
  nsp = length(pc.priors.species_list)
  counts = Int32[max(off[s+1] - off[s], 1) for s in 1:ns]
  soa = soa_type(P)((cohort=counts, species=fill(Int32(nsp), ns)))
  base_seed = rand(rng, UInt64)
  ep = generate_eco_params(pc, generate_params(pc, pc.priors.species_list, ["eco"], [collect(1:nsp)]))
  for i in 1:ns
    site = getsite(soa, i)
    site.active = true
    site.rng = RNGType(hash((base_seed, i)))
    e = eco_id isa Integer ? Int(eco_id) : Int(eco_id[i])
    site.ecocode = UIntType(e); site.eco_id = e; site.mapcode = UIntType(mapcode === nothing ? i : mapcode[i]); site.ref_cn = i
    site.old = zero(UIntType); site.live = zero(UIntType)
    _init_site_plugin!(P, site, no_establishment)
    planted === nothing || (site.planted = planted[i])
    n = off[i+1] - off[i]
    for (q, c) in enumerate(off[i]:off[i+1]-1)
      HBAD2Plugin._set_cohort!(site, q, species[c], age[c], N[c], D[c], H[c], zq[c], HBAD2Plugin.agb_of(ep[1].allo[Int(species[c])], Float64(N[c]), Float64(D[c])))
    end
    HBAD2Plugin.report_site!(soa.csr, soa.scalar, ep[1], i, Int(soa.refs.cohort[i]) - 1, n)
  end
  init_sites!(pc, soa, nothing)
  return soa
end
