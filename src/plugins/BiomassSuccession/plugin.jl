struct BiomassSuccession <: AbstractPlugin end
const PluginType = BiomassSuccession


PanCore.scalar_arrays(::Type{PluginType}, n::Int) =
  (
    B=Vector{FloatType}(undef, n),
    AGNPP=Vector{FloatType}(undef, n),
    harvestCapacityReduction=Vector{FloatType}(undef, n),
    growthReduction=Vector{FloatType}(undef, n),
    prevYearMortality=Vector{FloatType}(undef, n),
    shade_class=Vector{UIntType}(undef, n),
    no_establish=Vector{Bool}(undef, n),
    #eco_params=Vector{BiomassSuccessionEcoParams}(undef, n)
  )

PanCore.csr_fields(::Type{PluginType}) =
  (
    c_species=:cohort,
    c_age=:cohort,
    c_bio=:cohort,
    c_m_tot=:cohort, #scratch space to avoid adhoc allocations
    c_comp=:cohort, #scratch space to avoid adhoc allocations
    sp_mature=:species,
    sp_initial_biomass=:species,
    sp_sprout=:species,
    sp_plant=:species,
    sp_serotiny=:species,
    sp_seed=:species,
  )
PanCore.csr_arrays(::Type{PluginType}, nnz::NamedTuple) =
  (
    c_species=Vector{UIntType}(undef, nnz.cohort),
    c_age=Vector{FloatType}(undef, nnz.cohort),
    c_bio=Vector{FloatType}(undef, nnz.cohort),
    c_m_tot=Vector{FloatType}(undef, nnz.cohort),
    c_comp=Vector{FloatType}(undef, nnz.cohort),
    sp_mature=Vector{Bool}(undef, nnz.species),
    sp_initial_biomass=Vector{FloatType}(undef, nnz.species),
    sp_sprout=Vector{Bool}(undef, nnz.species),
    sp_plant=Vector{Bool}(undef, nnz.species),
    sp_serotiny=Vector{Bool}(undef, nnz.species),
    sp_seed=Vector{Bool}(undef, nnz.species),
  )

function PanCore.process_plugin!(soa::PanCore.AnySoA, ::Type{PluginType}, current_time::Int; ctx::NamedTuple)
  eco_params = ctx.eco_params
  Threads.@threads :static for i in 1:soa.n
    @inbounds site = getsite(soa, i)
    succession_step!(current_time, site, eco_params[site.eco_id])
    reproduction_check_step!(current_time, site, eco_params[site.eco_id])
    @inbounds site._new_cohort_counts = sum(site.sp_sprout) + site.live
  end
  PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))
  Threads.@threads :static for i in 1:soa.n
    @inbounds site = getsite(soa, i)
    #sprouting_step!(current_time, site, eco_params[site.eco_id])
    reproduction_commit_step!(current_time, site, eco_params[site.eco_id])
  end

end

Base.@kwdef struct BiomassSuccessionEcoParams
  SPINUP_MORTALITY_FRACTION::FloatType

  D::Vector{FloatType}
  S::Vector{FloatType}

  LONGEVITY::Vector{FloatType}
  SHADE_TOL::Vector{UIntType}
  MATURITY::Vector{FloatType}
  PROB_RESPROUT::Vector{FloatType}

  ANPP_MAX_SPP::Vector{FloatType}
  B_MAX_SPP::Vector{FloatType}
  B_MAX_ECO::FloatType
  PROB_MORT_SPP::Vector{FloatType}
  PROB_ESTAB_SPP::Vector{FloatType}

  MIN_REL_BIOMASS::Vector{FloatType}
  SUFFICIENT_LIGHT::Vector{Vector{FloatType}}

end
Base.@kwdef struct BiomassSuccessionParams
  # Metadata
  # eco -> species_ids (ids of the species in ecoregions)
  ECO_LIST::Vector{String}
  SPECIES_LIST::Vector{String}
  ECO_SPECIES_IDS::Vector{Vector{UIntType}}

  # Global
  SPINUP_MORTALITY_FRACTION::FloatType
  SUFFICIENT_LIGHT::Vector{Vector{FloatType}}

  # ecoregion specific
  MIN_REL_BIOMASS::Vector{Vector{FloatType}}

  # Species Specific
  D::Vector{FloatType}
  S::Vector{FloatType}
  LONGEVITY::Vector{FloatType}
  SHADE_TOL::Vector{UIntType}
  MATURITY::Vector{FloatType}
  PROB_RESPROUT::Vector{FloatType}


  # ecoregion x species 
  B_MAX_SPP::Vector{Vector{FloatType}}
  ANPP_MAX_SPP::Vector{Vector{FloatType}}
  PROB_MORT_SPP::Vector{Vector{FloatType}}
  PROB_ESTAB_SPP::Vector{Vector{FloatType}}

end

abstract type SeedDispersal end
struct NoDispersal <: SeedDispersal end
struct UniversalDispersal <: SeedDispersal end
struct WardDispersal <: SeedDispersal end

@generated function _dimslice(A::AbstractArray{T,N}, indices::Vararg{Any,M}) where {T,N,M}
  colons = ntuple(_ -> :, N - M)
  #println("A[$(indices)..., $(colons)...]\n")
  #println("N=$N, M=$M, indices=$indices, colons=$colons, A[$(indices)..., $(colons)...] \n")
  quote
    A[indices..., $(colons)...]
    #begin
    #    println("A[$(indices)..., $($(colons))...]\n")
    #    kk = A[indices..., $(colons)...]
    #    println(kk)
    #    kk
    #end
  end
end

macro dimslice(A, indices...)
  #@show indices
  quote
    _dimslice($(esc(A)), $(map(esc, indices)...))
  end
end

@inline function calculate_initial_biomass(sp_max_anpp::FloatType, site_b::FloatType, b_max_eco::FloatType)::FloatType
  b = exp(-FloatType(1.6f0) * site_b / b_max_eco)
  if b < one(FloatType)
    b = one(FloatType)
  end
  b *= sp_max_anpp
  if b < FloatType(2.0f0)
    b = FloatType(2.0f0)
  end
  return b
end

@inline function add_cohort!(site::SiteView, species::UIntType, age::FloatType, biomass::FloatType)
  #println(site.live)
  site.live += one(UIntType)
  if age > one(FloatType)
    site.old += 1
  end
  #println(site.live)
  site.c_species[site.live] = species
  site.c_age[site.live] = age
  site.c_bio[site.live] = biomass
end
function spinup_cohorts!(empty_soa::PanCore.AnySoA, spinup_cohorts::DataFrame, eco_params::Vector{BiomassSuccessionEcoParams})
  #show(spinup_cohorts.year_deficit)
  # year deficit is establisment year.
  # however, I cannot add with age = 0, therefore it'll have to show up the year after with age=1
  # therefore year_age_one = year_deficit + 1
  @debug spinup_cohorts
  current_year = minimum(spinup_cohorts.year_deficit) + 1
  soa = empty_soa
  # max_current_year = max(year_deficit) + 1 = -2 + 1 = -1
  # will have to be careful with simulation not to trigger succession year 0 twice
  ## current_year will go down to -1, since the last estab cohort
  ## would be 1 year old, so a year before the last start_measdate
  #pbar = ProgressBar(total = -current_year)
  #println("current_year, $(current_year), year_deficit+1, $(first(spinup_cohorts).year_deficit + 1)")
  year_groups = groupby(spinup_cohorts, :year_deficit, sort=true)
  for (year_deficit_key, rows) in pairs(year_groups)#eachrow(spinup_cohorts)
    year_deficit = year_deficit_key.year_deficit
    #println(row)
    while current_year < year_deficit + 1
      #println("current_year, $(current_year), year_deficit+1, $(year_deficit + 1), succession: $(current_year < year_deficit+1) ")
      #@debug ("year $(current_year), before recounting $((soa.refs.cohort))")
      Threads.@threads :static for i in 1:soa.n
        @inbounds site = getsite(soa, i)
        @assert Int(site.mapcode) == Int(i)
        if i == 18
          @debug "thread recount after marking s18: live=$(site.live), sp_sprout=$(site.sp_sprout), species_refs=$(pointer(soa.refs.species))[$(soa.refs.species[18]):$(soa.refs.species[19]-1)]"
        end
        @inbounds site._new_cohort_counts = sum(site.sp_sprout) + site.live
      end
      @debug ("year $(current_year), after recounting $((soa.refs.cohort))")
      @debug (soa.scalar._new_cohort_counts)
      @debug ("adjusting")
      soa = #PanCore.with_thread_sync() do
        PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))
      #end
      @debug ("sprouting")
      Threads.@threads :static for i in 1:soa.n
        @inbounds site = getsite(soa, i)
        #sprouting_step!(current_year, site, eco_params[site.eco_id])
        reproduction_commit_step!(current_year, site, eco_params[site.eco_id])
      end
      @debug ("succession")
      Threads.@threads :static for i in 1:soa.n
        @inbounds site = getsite(soa, i)
        succession_step!(current_year, site, eco_params[site.eco_id])
        reproduction_check_step!(current_year, site, eco_params[site.eco_id])
        if i == 18
          @debug "thread recount after succession+repro s18: live=$(site.live), sp_sprout=$(site.sp_sprout), species_refs=$(pointer(soa.refs.species))[$(soa.refs.species[18]):$(soa.refs.species[19]-1)]"
        end
        @inbounds site._new_cohort_counts = sum(site.sp_sprout) + site.live
      end
      @debug ("year $(current_year), recounting after repro $((soa.refs.cohort))")
      @debug (soa.scalar._new_cohort_counts)
      #grow all active
      #PanCore.process_plugin!(soa, PluginType, t; ctx=(eco_params=eco_params,))
      #Threads.@threads :static for site in sites #
      #    if site.active
      #        #println(site.mapcode)
      #        succession_step!(current_year, site, eco_params)
      #        reproduction_step!(current_year, site, eco_params)
      #    end
      #end
      current_year += 1
    end
    @debug ("Done catching up $(current_year)")
    #check and add cohort

    plots = combine(groupby(rows, [:plot_id, :eco_id], sort=false)) do rs
      (; eco_species_ids=[rs.eco_species_id])
    end
    Threads.@threads :static for row in eachrow(plots)
      site = PanCore.getsite(soa, Int(row.plot_id))
      site.active = true
      for sp in row.eco_species_ids
        if site.sp_sprout[sp]
          continue
        end
        site.sp_sprout[sp] = !any(site.c_species[young_idx] == sp for young_idx in (site.old+1):site.live)
        if site.sp_sprout[sp]

          @assert objectid(soa) == objectid(site.soa)
          #@assert site.soa.csr.sp_sprout === soa.csr.sp_sprout
          #println(site.sp_sprout)
          soa_sp_sprout = @view soa.csr.sp_sprout[soa.refs.species[Int(row.plot_id)]:soa.refs.species[Int(row.plot_id)+1]-1]
          #println(soa_sp_sprout)
          @assert pointer(site.sp_sprout) == pointer(soa_sp_sprout)
          @assert all(site.sp_sprout .== soa_sp_sprout)
          @assert pointer(soa.refs.species) == pointer(site.soa.refs.species)
          @assert length(soa.refs.species) == length(site.soa.refs.species)
          @assert (soa.refs.species[Int(row.plot_id)]:soa.refs.species[Int(row.plot_id)+1]-1) == (site.soa.refs.species[Int(row.plot_id)]:site.soa.refs.species[Int(row.plot_id)+1]-1)
          @debug ("year $(current_year), marking plot_id=$(row.plot_id), mapcode=$(site.mapcode), sp=$(sp), sum(sp_sprout)+live=$(sum(site.sp_sprout)+site.live), sp_sprout=$(site.sp_sprout), soa=$(objectid(soa)), site.soa=$(objectid(site.soa)), soa.sp_sprout[lo:hi]=$(soa.csr.sp_sprout[soa.refs.species[Int(row.plot_id)]:soa.refs.species[Int(row.plot_id)+1]-1]), soa.species_refs=$(pointer(soa.refs.species))[$(soa.refs.species[Int(row.plot_id)]):$(soa.refs.species[Int(row.plot_id)+1]-1)] , site.soa.species_refs=$(pointer(site.soa.refs.species))[$(site.soa.refs.species[Int(row.plot_id)]):$(site.soa.refs.species[Int(row.plot_id)+1]-1)]")
          #params = eco_params[site.eco_id]
          #try
          #initial_biomass = calculate_initial_biomass(params.B_MAX_SPP[sp], site.B, params.B_MAX_ECO)
          #add_new_cohort!(site, UIntType(sp), one(FloatType), initial_biomass)

          #catch e
          #    println(row)
          #    println(site)
          #    println(params)
          #    rethrow(e)
          #end
        end
      end
      @debug "post-mark verify s=$(row.plot_id): live=$(site.live), sp_sprout=$(site.sp_sprout), species_refs=$(pointer(soa.refs.species))[$(soa.refs.species[Int(row.plot_id)]):$(soa.refs.species[Int(row.plot_id)+1]-1)]"
    end


    #print(site)
    # make it active if not already
    # check if site has a young cohort of species
    # if not, add one with initial biomass calculated
    #add_new_cohort = true
    #if site.old < site.live  # there are young cohorts
    #    for young_idx in (site.old+1):site.live
    #        if site.c_species[young_idx] == sp #found one, no need to add
    #            add_new_cohort = false
    #            break
    #        end
    #    end
    #end
    #println("Need to added cohort: $(add_new_cohort), current_year= $(current_year)")
    #println("Adding cohort $(row.species_symbol_map) to ", row.plot_id)
  end
  @debug ("year $(current_year), final before recounting $((soa.refs.cohort))")
  Threads.@threads :static for i in 1:soa.n
    @inbounds site = getsite(soa, i)
    @inbounds site._new_cohort_counts = sum(site.sp_sprout) + site.live
  end
  @debug ("year $(current_year), final after recounting $((soa.refs.cohort))")
  @debug (soa.scalar._new_cohort_counts)
  soa = #PanCore.with_thread_sync() do
    PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))
  #end
  Threads.@threads :static for i in 1:soa.n
    @inbounds site = getsite(soa, i)
    #sprouting_step!(current_year, site, eco_params[site.eco_id])
    reproduction_commit_step!(current_year, site, eco_params[site.eco_id])
  end
  @assert current_year == -1 "$(current_year)"
  #update(pbar)
  # cohorts with year_deficit = 0 will have been added but not succeeded yet
  return soa


end

function reproduction_commit_step!(current_time::Int, site::SiteView, params::BiomassSuccessionEcoParams)
  B = site.B #before adding any new cohorts
  new_B = zero(FloatType)
  for sp in 1:length(site.sp_sprout)
    if site.sp_sprout[sp]
      nbio = calculate_initial_biomass(params.ANPP_MAX_SPP[sp], B, params.B_MAX_ECO)
      add_cohort!(site, UIntType(sp), one(FloatType), nbio)
      new_B += nbio
    end
  end
  site.B += new_B
  site.sp_sprout .= false
  site.sp_plant .= false
  site.sp_serotiny .= false
  site.sp_seed .= false
end

function _sprouting_step!(current_time::Int, site::SiteView, params::BiomassSuccessionEcoParams)
  for sp in 1:length(site.sp_sprout)
    if site.sp_sprout[sp]
      new_biomass = calculate_initial_biomass(params.ANPP_MAX_SPP[sp],
        site.B, params.B_MAX_ECO)
      add_cohort!(site, UIntType(sp), one(FloatType), new_biomass)
      site.B += new_biomass
      site.sp_sprout[sp] = false
    end
  end
end

@inline function plant_establish(sp::Int, site::SiteView, params::BiomassSuccessionEcoParams; rng::Random.AbstractRNG)
  return params.PROB_ESTAB_SPP[sp] > zero(FloatType)
end
@inline function prob_eco_establish(sp::Int, site::SiteView, params::BiomassSuccessionEcoParams; rng::Random.AbstractRNG)
  return params.PROB_ESTAB_SPP[sp] == one(FloatType) || params.PROB_ESTAB_SPP[sp] >= rand(rng, FloatType)
end
@inline serotiny_establish = prob_eco_establish
@inline function resprout_establish(sp::Int, site::SiteView, params::BiomassSuccessionEcoParams; rng::Random.AbstractRNG)
  return params.PROB_RESPROUT[sp] == one(FloatType) || params.PROB_RESPROUT[sp] >= rand(rng, FloatType)
end
@inline function sufficient_light(sp::Int, shade_probs::Vector{FloatType}, params::BiomassSuccessionEcoParams; rng::Random.AbstractRNG)
  sp_light_prob = shade_probs[params.SHADE_TOL[sp]]
  sp_light_prob == zero(FloatType) && return false
  return sp_light_prob >= rand(rng, FloatType)
end


@inline function do_seeding!(::NoDispersal, site::SiteView, params::BiomassSuccessionEcoParams; rng::Random.AbstractRNG)
  shade_probs = params.SUFFICIENT_LIGHT[site.shade_class+1] #julia is 1-indexed
  n_species = length(site.sp_mature)
  for sp in 1:n_species
    if site.sp_mature[sp]
      site.sp_seed[sp] = sufficient_light(sp, shade_probs, params; rng=rng) && prob_eco_establish(sp, site, params; rng=rng)
    end
  end
end

function reproduction_check_step!(current_time::Int, site::SiteView, params::BiomassSuccessionEcoParams; seeding::SeedDispersal=NoDispersal())
  # reproduction if live cohorts
  # the purpose of this function is to turn "shoudl try" flags of sp_plant, sp_serotiny, and sp_resprout into "succeeded" or not
  # after all trials are done, site.sp_sprout represents which species will have a new cohort added
  # adding the actual cohort based on success is in reproduction_commit_step!
  if !site.active || site.no_establish
    return
  end
  rng = site.rng
  n_species = length(site.sp_mature)

  # check "try planting flags"
  planting = false
  for sp in 1:n_species
    if !site.sp_plant[sp]
      continue
    end
    site.sp_plant[s] = plant_establish(sp, site, params; rng=rng)
    planting |= site.sp_plant[sp]
  end

  shade_probs = params.SUFFICIENT_LIGHT[site.shade_class+1] #julia is 1-indexed

  serotiny = false
  #try serotiny if no planting
  if !planting
    for sp in 1:n_species
      if !site.sp_serotiny[sp]
        continue
      end
      site.sp_serotiny[s] = sufficient_light(sp, shade_probs, params; rng=rng) && serotiny_establish(sp, site, params; rng=rng)
      serotiny |= site.sp_serotiny[sp]
    end
  end

  resprout = false
  #resprout only if no serotiny
  if !serotiny
    for sp in 1:n_species
      if !site.sp_sprout[sp]
        continue
      end
      site.sp_sprout[s] = sufficient_light(sp, shade_probs, params; rng=rng) && resprout_establish(sp, site, params; rng=rng)
      resprout |= site.sp_sprout[sp]
    end
    # can sprout even when plant
  else
    site.sp_sprout .= site.sp_serotiny
  end

  if resprout
    site.sp_sprout .|= site.sp_plant
  end

  if !(planting || serotiny || resprout)
    do_seeding!(seeding, site, params; rng=rng)
    site.sp_sprout .= site.sp_seed
  end
end

function succession_step!(current_time::Int, site::SiteView, params::BiomassSuccessionEcoParams)
  if !site.active
    return
  end
  B = zero(FloatType)
  C = zero(FloatType)
  capacityReduction = (one(FloatType) - site.harvestCapacityReduction)
  #B = sum(c_bio)
  #


  #replicate serial competition:
  # sort older to young
  # B = sum their bio,
  # increment age, and remove senescent
  # 
  # older to 
  # 
  # advancing age, summing site biomass, computing competition, mortality due to age or random act of god
  for i in 1:site.live
    site.c_age[i] += one(FloatType)
    age = site.c_age[i]
    sp = site.c_species[i]
    bio = site.c_bio[i]
    B += bio
    comp = bio^FloatType(0.95f0)
    #println("Bio $bio, Comp $(comp)")
    if comp < one(FloatType)
      comp = one(FloatType)
    end
    C += comp
    #@assert !isnan(C) "$bio"
    site.c_comp[i] = comp

    # this is calculating age_mortality + longevity mortality
    # starting with all biomass dead
    site.c_m_tot[i] = bio
    max_age = params.LONGEVITY[sp]
    #keeping all biomass dead if age >= max_age
    if age < max_age
      # not max age yet
      m_age_factor = exp(params.D[sp] * (age / max_age - one(FloatType)))
      if current_time <= 0
        m_age_factor += params.SPINUP_MORTALITY_FRACTION
      end
      if m_age_factor < one(FloatType) # worth multiplying, so at max m_age_factor = 1.0 effectively
        site.c_m_tot[i] *= m_age_factor
      end
    end
  end

  new_B = zero(FloatType)
  AGNPP = zero(FloatType)
  M_TOT = zero(FloatType)
  B_ACT = zero(FloatType)
  site.sp_mature .= false

  last = site.live
  i = 1
  while i <= last
    age = site.c_age[i]
    bio = site.c_bio[i]
    sp = site.c_species[i]
    b_max = params.B_MAX_SPP[sp] * capacityReduction
    b_pot = b_max - B + bio
    if b_pot < one(FloatType)
      b_pot = one(FloatType)
    end
    if capacityReduction >= one(FloatType) && b_pot < site.prevYearMortality
      b_pot = site.prevYearMortality
    end
    b_ap = bio / b_pot
    b_ap_s = b_ap^params.S[sp]
    anpp_act = b_ap_s * exp(one(FloatType) - b_ap_s)
    #@assert !isnan(anpp_act) "$b_ap_s, $(params.S[sp])"
    if anpp_act > one(FloatType)
      anpp_act = one(FloatType)
    end
    site.c_comp[i] /= C

    anpp_max_c = params.ANPP_MAX_SPP[sp] * site.c_comp[i]
    #@assert !isnan(anpp_max_c) "$C, $(site.c_comp[i]), $(params.ANPP_MAX_SPP[site.ecocode, sp])"
    anpp_act *= anpp_max_c

    if site.growthReduction > zero(FloatType)
      anpp_act *= one(FloatType) - site.growthReduction
    end
    AGNPP += anpp_act # in landis code it calculates previous year's anpp, not sure if that's correct

    # growth mortality
    m_bio = anpp_max_c
    if b_ap <= one(FloatType)
      m_bio *= (FloatType(2.0f0) * b_ap) / (one(FloatType) + b_ap)
    end
    if m_bio > bio
      m_bio = bio
    end
    if m_bio > anpp_max_c
      m_bio = anpp_max_c
    end
    if site.growthReduction > zero(FloatType)
      m_bio *= one(FloatType) - site.growthReduction
    end

    # remove age mortality from anpp and growth mortality
    m_age = site.c_m_tot[i]
    anpp_act -= m_age
    if anpp_act < one(FloatType)
      anpp_act = one(FloatType)
    end
    m_bio -= m_age
    if m_bio < zero(FloatType)
      m_bio = zero(FloatType)
    end
    if m_bio > anpp_act
      m_bio = anpp_act
    end
    m_tot = m_age + m_bio
    # remove rounding errors if they exit
    if m_tot > bio
      m_tot = bio
    end
    #@assert bio - m_tot >= -FloatType(1.0f-8)
    # keep all bio dead if random mortality (not allowed during spinup)
    if current_time > 0 && rand(site.rng, FloatType) > params.PROB_MORT_SPP[sp]
      m_tot = bio
    end

    M_TOT += m_tot
    site.c_m_tot[i] = m_tot

    nbio = bio + anpp_act - m_tot
    #@assert !isnan(nbio) "$bio, mtot  $m_tot, $m_age, $m_bio anpp_act $anpp_act, $anpp_max_c, $C, $(site.c_comp[i])"

    site.c_bio[i] = nbio
    senescent = (nbio <= FloatType(1.0f-8))
    if !senescent
      new_B += nbio
      if age > FloatType(5.0f0)
        B_ACT += nbio
      end

      #updating species maturity
      if age >= params.MATURITY[sp]
        site.sp_mature[sp] = true
      end

      i += 1
    else
      if i != last
        site.c_age[i] = site.c_age[last]
        site.c_bio[i] = site.c_bio[last]
        site.c_species[i] = site.c_species[last]
        #site.c_comp[i] = site.c_comp[last]
        #site.c_m_tot[i] = site.c_m_tot[last]
      end
      last -= 1
    end
    #TODO slot in the right place
  end
  site.live = last



  # calculating shade class
  site_b_max = params.B_MAX_ECO
  site_b_pot = site_b_max - site.prevYearMortality
  #println(typeof(B_ACT), B_ACT)
  #println(typeof(site_b_pot), site_b_pot)
  if B_ACT > site_b_pot
    B_ACT = site_b_pot
  end
  b_am = B_ACT / site_b_max
  #shade_classes = @view params.MIN_REL_BIOMASS[:, site.ecocode]

  shade_class = zero(UIntType)
  for sc_threshold in params.MIN_REL_BIOMASS
    if b_am > sc_threshold
      #clears the threshold, so at least has this shade_class
      shade_class += 1
    else
      break

    end
  end

  ######################
  # Updating site data
  #####################
  # before reproduction, all cohorts on site are now old
  site.old = site.live
  site.B = new_B
  site.AGNPP = AGNPP
  #site.defoliationLoss = defoliationLoss_ij.sum()
  site.prevYearMortality = M_TOT #M_TOT_ij.sum()
  site.shade_class = shade_class


  #println(current_time, "done")


end
