module BiomassSuccessionPlugin
using ..PanCore
using DataFrames

export generate_biomass_params, generate_eco_params, spinup_cohorts!

import Random, Distributions as Dists

struct BiomassSuccession <: AbstractPlugin end
const PluginType = BiomassSuccession


PanCore.scalar_arrays(::Type{PluginType}, n::Int) =
    (
        B=Vector{FloatType}(undef, n),
        AGNPP=Vector{FloatType}(undef, n),
        capacityReduction=Vector{FloatType}(undef, n),
        growthReduction=Vector{FloatType}(undef, n),
        prevYearMortality=Vector{FloatType}(undef, n),
        shade_class=Vector{UIntType}(undef, n),
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
        sp_sprout=:species,
    )
PanCore.csr_arrays(::Type{PluginType}, nnz::NamedTuple) =
    (
        c_species=Vector{UIntType}(undef, nnz.cohort),
        c_age=Vector{FloatType}(undef, nnz.cohort),
        c_bio=Vector{FloatType}(undef, nnz.cohort),
        c_m_tot=Vector{FloatType}(undef, nnz.cohort),
        c_comp=Vector{FloatType}(undef, nnz.cohort),
        sp_mature=Vector{Bool}(undef, nnz.species),
        sp_sprout=Vector{Bool}(undef, nnz.species),
    )

function PanCore.process_plugin!(soa::PanCore.AnySoA, ::Type{PluginType}, current_time::Int; ctx::NamedTuple)
    eco_params = ctx.eco_params
    new_cohort_counts = zeros(Int32, soa.n)
    Threads.@threads :static for i in 1:soa.n
        @inbounds site = getsite(soa, i)
        succession_step!(current_time, site, eco_params[site.eco_id])
        reproduction_step!(current_time, site, eco_params[site.eco_id])
        @inbounds new_cohort_counts[i] = sum(site.sp_sprout) + site.live
    end
    PanCore.readjust_soa!(soa, (cohort=new_cohort_counts,))
    Threads.@threads :static for i in 1:soa.n
        @inbounds site = getsite(soa, i)
        sprouting_step!(current_time, site, eco_params[site.eco_id])
    end

end

Base.@kwdef struct BiomassSuccessionEcoParams
    SPINUP_MORTALITY_FRACTION::FloatType

    D::Vector{FloatType}
    S::Vector{FloatType}

    LONGEVITY::Vector{FloatType}
    SHADE_TOL::Vector{UIntType}
    MATURITY::Vector{FloatType}

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


    # ecoregion x species 
    B_MAX_SPP::Vector{Vector{FloatType}}
    ANPP_MAX_SPP::Vector{Vector{FloatType}}
    PROB_MORT_SPP::Vector{Vector{FloatType}}
    PROB_ESTAB_SPP::Vector{Vector{FloatType}}

end

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

function generate_eco_params(params::BiomassSuccessionParams)::Array{BiomassSuccessionEcoParams}

    return [
        BiomassSuccessionEcoParams(
            SPINUP_MORTALITY_FRACTION=params.SPINUP_MORTALITY_FRACTION,
            SUFFICIENT_LIGHT=params.SUFFICIENT_LIGHT,
            D=(@dimslice params.D species),
            S=(@dimslice params.S species),
            LONGEVITY=(@dimslice params.LONGEVITY species),
            MATURITY=(@dimslice params.MATURITY species),
            SHADE_TOL=(@dimslice params.SHADE_TOL species), ANPP_MAX_SPP=params.ANPP_MAX_SPP[eco_id],
            B_MAX_SPP=params.B_MAX_SPP[eco_id],
            B_MAX_ECO=maximum(params.B_MAX_SPP[eco_id]), PROB_MORT_SPP=(params.PROB_MORT_SPP[eco_id]),
            PROB_ESTAB_SPP=(params.PROB_ESTAB_SPP[eco_id]),
            MIN_REL_BIOMASS=params.MIN_REL_BIOMASS[eco_id],
        )




        for (eco_id, species) in enumerate(params.ECO_SPECIES_IDS)]


end
function generate_biomass_params(species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Array{Array{Int}}; rng::Random.AbstractRNG)
    n_species = length(species_list) |> UIntType
    n_ecoregions = length(eco_list) |> UIntType
    # TODO: species that do not show up for a specific ecoregion, make all their prob_estab = 0
    SPINUP_MORTALITY_FRACTION = 0.15f0 #rand(Dists.Uniform(0f0,0.20f0))
    #println(typeof(SPINUP_MORTALITY_FRACTION))

    S = rand(rng, Dists.truncated(Dists.Normal(0.5, 1.0), 0.01, 1.0), n_species) .|> FloatType #Random.rand(rng, FloatType, n_species),#
    #println(typeof(S))
    D = rand(rng, Dists.truncated(Dists.Normal(15, 10), 5, 25), n_species) .|> FloatType
    #println(typeof(D))
    LONGEVITY = rand(rng, Dists.truncated(Dists.Normal(200, 100), 100, 300), n_species) .|> FloatType
    #println(typeof(LONGEVITY))
    SHADE_TOL = rand(rng, Dists.DiscreteUniform(1, 5), n_species) .|> UIntType # ::Vector{FloatType}
    #println(typeof(SHADE_TOL))
    MATURITY = rand(rng, Dists.DiscreteUniform(3, 40), n_species) .|> FloatType #::Vector{FloatType}
    #println(typeof(MATURITY))

    PROB_MORT_SPP = [rand(rng, Dists.Uniform(), length(eco_species)) .|> FloatType  #::Matrix{FloatType}
                     for eco_species in eco_species_ids]
    #println(typeof(PROB_MORT_SPP))
    PROB_ESTAB_SPP = [rand(rng, Dists.Uniform(), length(eco_species)) .|> FloatType  #::Matrix{FloatType}
                      for eco_species in eco_species_ids]
    #println(typeof(PROB_ESTAB_SPP))
    ANPP_MAX_SPP = [rand(rng, Dists.truncated(Dists.Normal(2500, 100), 2400, 2500), length(eco_species)) .|> FloatType
                    for eco_species in eco_species_ids]
    #println(typeof(ANPP_MAX_SPP))
    B_MAX_SPP = [rand(rng, Dists.truncated(Dists.Normal(2500, 100), 2400, 2500), length(eco_species)) .|> FloatType
                 for eco_species in eco_species_ids]


    #println(typeof(B_MAX_SPP))
    #println(B_MAX_SPP)
    #B_MAX_ECO = [maximum(b_max_spp_eco) for b_max_spp_eco in B_MAX_SPP] #FloatType[0.0,0.0],#::Vector{FloatType}
    #println(typeof(B_MAX_ECO))
    #println(B_MAX_ECO)
    # (species.shade_tol 1-5 X (site.shade_class+1) 1-6) -> prob of sufficient light
    # shade_tol: 1 (least tolerant to shade) to 5 (most tolerant to shade)
    # shade_class: 1 (no shade on site) to 6 (full shade)
    # Biomass Documentation shadeclass 0 (no shade) to 5 (full shade) but julia arrays are 1-indexed
    # julia is column major, much faster to pickout the site's shade class as one chunk
    # then reference the species' shade_tol within
    SUFFICIENT_LIGHT_MATRIX = FloatType[
        1.00 0.50 0.25 0.00 0.00 0.00;
        1.00 1.00 0.50 0.25 0.00 0.00;
        1.00 1.00 1.00 0.50 0.25 0.00;
        1.00 1.00 1.00 1.00 0.50 0.25;
        1.00 1.00 1.00 1.00 1.00 0.50]
    # transforming to (shade_class -> shade_tol) to be able to save to json Vector{Vector{FloatType}}, Matrix{FloatType} is not easily serde'd
    SUFFICIENT_LIGHT = [vec(SUFFICIENT_LIGHT_MATRIX[:, shade_class]) for shade_class in axes(SUFFICIENT_LIGHT_MATRIX, 2)]
    #[1.0, 1.0, 1.0, 1.0, 1.0]  for shade_class = 1 (no shade), plants of all shade tolerance can reproduce
    #[0.5, 1.0, 1.0, 1.0, 1.0]
    #[0.25, 0.5, 1.0, 1.0, 1.0]
    #[0.0, 0.25, 0.5, 1.0, 1.0]
    #[0.0, 0.0, 0.25, 0.5, 1.0]
    #[0.0, 0.0, 0.0, 0.25, 0.5] for shade_class = 6 (full shade), only plants with highest shade tol (4,5) have a chance

    #println(typeof(SUFFICIENT_LIGHT))

    # by ecoregion
    firstMINRel = rand(rng, Dists.Uniform(0.0, 0.5), n_ecoregions) .|> FloatType
    #MIN_REL_BIOMASS = FloatType[[0.25, 0.45, 0.56, 0.70, 0.90] for _ in 1:2]
    # (shade_class x eco) -> bio percent
    # again, column major, pickout the relevant column for ecoregion
    #MIN_REL_BIOMASS = [fmin + k * 0.10f0
    #                   for k in 0:4,
    #                   fmin in firstMINRel]
    # transforming to eco -> shade_class
    MIN_REL_BIOMASS = [[fmin + k * 0.10f0 for k in 0:4]
                       for fmin in firstMINRel]
    #println(typeof(MIN_REL_BIOMASS))
    #println(MIN_REL_BIOMASS)

    p = BiomassSuccessionParams(
        SPINUP_MORTALITY_FRACTION=SPINUP_MORTALITY_FRACTION,
        SUFFICIENT_LIGHT=SUFFICIENT_LIGHT, D=D,
        S=S,
        LONGEVITY=LONGEVITY,
        SHADE_TOL=SHADE_TOL,
        MATURITY=MATURITY, ANPP_MAX_SPP=ANPP_MAX_SPP,
        B_MAX_SPP=B_MAX_SPP,
        PROB_MORT_SPP=PROB_MORT_SPP,
        PROB_ESTAB_SPP=PROB_ESTAB_SPP,
        MIN_REL_BIOMASS=MIN_REL_BIOMASS, SPECIES_LIST=species_list,
        ECO_LIST=eco_list,
        ECO_SPECIES_IDS=eco_species_ids,
    )
    return p


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
function spinup_cohorts!(empty_soa::PanCore.AnySoA, spinup_cohorts::DataFrame, eco_params::Array{BiomassSuccessionEcoParams})
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
            #@debug (new_cohort_counts)
            #new_cohort_counts .= Int32(0)
            new_cohort_counts = zeros(Int32, soa.n)
            Threads.@threads for i in 1:soa.n
                @inbounds site = getsite(soa, i)
                @assert Int(site.mapcode) == Int(i)
                if i == 18
                    @debug "thread recount after marking s18: live=$(site.live), sp_sprout=$(site.sp_sprout), species_refs=$(pointer(soa.refs.species))[$(soa.refs.species[18]):$(soa.refs.species[19]-1)]"
                end
                @inbounds new_cohort_counts[i] = sum(site.sp_sprout) + site.live
            end
            @debug ("year $(current_year), after recounting $((soa.refs.cohort))")
            @debug (new_cohort_counts)
            @debug ("adjusting")
            soa = PanCore.with_thread_sync() do
                PanCore.readjust_soa!(soa, (cohort=new_cohort_counts,))
            end
            @debug ("sprouting")
            Threads.@threads for i in 1:soa.n
                @inbounds site = getsite(soa, i)
                sprouting_step!(current_year, site, eco_params[site.eco_id])
            end
            @debug ("succession")
            new_cohort_counts = zeros(Int32, soa.n)
            Threads.@threads for i in 1:soa.n
                @inbounds site = getsite(soa, i)
                succession_step!(current_year, site, eco_params[site.eco_id])
                reproduction_step!(current_year, site, eco_params[site.eco_id])
                if i == 18
                    @debug "thread recount after succession+repro s18: live=$(site.live), sp_sprout=$(site.sp_sprout), species_refs=$(pointer(soa.refs.species))[$(soa.refs.species[18]):$(soa.refs.species[19]-1)]"
                end
                @inbounds new_cohort_counts[i] = sum(site.sp_sprout) + site.live
            end
            @debug ("year $(current_year), recounting after repro $((soa.refs.cohort))")
            @debug (new_cohort_counts)
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
        Threads.@threads for row in eachrow(plots)
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
    new_cohort_counts = zeros(Int32, soa.n)
    Threads.@threads for i in 1:soa.n
        @inbounds site = getsite(soa, i)
        @inbounds new_cohort_counts[i] = sum(site.sp_sprout) + site.live
    end
    @debug ("year $(current_year), final after recounting $((soa.refs.cohort))")
    @debug (new_cohort_counts)
    soa = PanCore.with_thread_sync() do
        PanCore.readjust_soa!(soa, (cohort=new_cohort_counts,))
    end
    Threads.@threads for i in 1:soa.n
        @inbounds site = getsite(soa, i)
        sprouting_step!(current_year, site, eco_params[site.eco_id])
    end
    @assert current_year == -1 "$(current_year)"
    #update(pbar)
    # cohorts with year_deficit = 0 will have been added but not succeeded yet
    return soa


end

function sprouting_step!(current_time::Int, site::SiteView, params::BiomassSuccessionEcoParams)
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

function reproduction_step!(current_time::Int, site::SiteView, params::BiomassSuccessionEcoParams)
    # reproduction if live cohorts
    if site.active && site.live > zero(UIntType)
        #println(shade_class, params.SUFFICIENT_LIGHT)
        #shade_probs = @view params.SUFFICIENT_LIGHT[:, site.shade_class]
        shade_probs = params.SUFFICIENT_LIGHT[site.shade_class+1] #julia is 1-indexed
        #println(shade_probs)
        for sp in 1:length(site.sp_mature)
            if site.sp_mature[sp]
                sp_light_prob = shade_probs[params.SHADE_TOL[sp]]
                light_rng = rand(site.rng, FloatType)
                if light_rng <= sp_light_prob
                    sp_estab_prob = params.PROB_ESTAB_SPP[sp]
                    sp_estab_rng = rand(site.rng, FloatType)
                    if sp_estab_rng <= sp_estab_prob
                        #new_biomass = calculate_initial_biomass(params.ANPP_MAX_SPP[sp],
                        #    site.B, params.B_MAX_ECO)
                        #add_new_cohort!(site, UIntType(sp), one(FloatType), new_biomass)
                        site.sp_sprout[sp] = true
                        #site.B += new_biomass
                    end
                end
            end

        end
    end
end

function succession_step!(current_time::Int, site::SiteView, params::BiomassSuccessionEcoParams)
    if !site.active
        return
    end
    B = zero(FloatType)
    C = zero(FloatType)
    #RNG = site.rng Random.seed!(site.rng_state)
    site.sp_mature .= false

    # advancing age, summing site biomass, computing competition, mortality due to age or random act of god
    for i in 1:site.live
        site.c_age[i] += one(FloatType)
        age = site.c_age[i]
        sp = site.c_species[i]
        if age >= params.MATURITY[sp]
            site.sp_mature[sp] = true
        end
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
        site.c_m_tot[i] = bio
        max_age = params.LONGEVITY[sp]
        if age < max_age
            # not max age yet
            mort_rng = rand(site.rng, FloatType)
            if mort_rng > params.PROB_MORT_SPP[sp]
                m_age_factor = exp(params.D[sp] * (age / max_age - one(FloatType)))
                if current_time <= 0
                    m_age_factor += params.SPINUP_MORTALITY_FRACTION
                end
                if m_age_factor < one(FloatType)
                    site.c_m_tot[i] *= m_age_factor
                end
            end
        end
    end

    new_B = zero(FloatType)
    AGNPP = zero(FloatType)
    M_TOT = zero(FloatType)
    B_ACT = zero(FloatType)

    last = site.live
    i = 1
    while i <= last
        age = site.c_age[i]
        bio = site.c_bio[i]
        sp = site.c_species[i]
        b_max = params.B_MAX_SPP[sp] * site.capacityReduction
        b_pot = b_max - B - bio
        if b_pot < one(FloatType)
            b_pot = one(FloatType)
        end
        # TODO: check this condition
        if site.capacityReduction >= one(FloatType) && b_pot < site.prevYearMortality
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
        AGNPP += anpp_act

        # growth mortality
        m_bio = anpp_max_c
        if m_bio <= one(FloatType)
            m_bio *= (FloatType(2.0f0) * b_ap) / (one(FloatType) + b_ap)
        end
        if m_bio > bio
            m_bio = bio
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
        if m_bio < anpp_act
            m_bio = anpp_act
        end

        m_tot = m_age + m_bio
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
        site.live = last
    end


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
end
