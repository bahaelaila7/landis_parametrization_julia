module SuccessionModule
# prepare parameters for one run
# prepare cohorts
#   spin up
# report outcome
# compare outcome
export Site, BiomassSuccessionParams, BiomassSuccessionEcoParams, succession_step!, add_new_cohort!, reproduction_step!
using Base
using Random

FloatType = Float32
UIntType = UInt32
#struct FloatType
#    val::__FloatType
#end
#struct UIntType
#    val::__UIntType
#end

#@inline UIntType(val::__UIntType) = UIntType(val)
##@inline __UIntType(a::UIntType) = a.val
#@inline Base.convert(::Type{UIntType}, val::__UIntType) = UIntType(val)
#@inline Base.convert(::Type{__UIntType}, a::UIntType) = a.val

#@inline FloatType(val::__FloatType) = FloatType(val)
##@inline __FloatType(a::FloatType) = a.val
#@inline Base.convert(::Type{FloatType}, val::__FloatType) = FloatType(val)
#@inline Base.convert(::Type{__FloatType}, a::FloatType) = a.val

Base.@kwdef mutable struct Site
    active::Bool
    rng::Random.Xoshiro

    mapcode::UIntType # index from raster
    ecocode::UIntType # value from eco_raster

    eco_id::Int # internal index into defined ecoregions
    ref_cn::UIntType # for debug purposes

    cap::UIntType
    old::UIntType
    live::UIntType
    B::FloatType
    AGNPP::FloatType
    capacityReduction::FloatType
    growthReduction::FloatType
    prevYearMortality::FloatType
    shade_class::UIntType

    c_species::Vector{UIntType}
    c_age::Vector{FloatType}
    c_bio::Vector{FloatType}
    c_m_tot::Vector{FloatType}
    c_comp::Vector{FloatType}
    sp_mature::Vector{Bool}
end

Base.@kwdef struct BiomassSuccessionEcoParams
    SPINUP_MORTALITY_FRACTION::Vector{FloatType}

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
    SPINUP_MORTALITY_FRACTION::Vector{FloatType}
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
@inline function ensure_site_cap!(site::Site, new_cap::UIntType)
    cap = site.cap
    if cap < new_cap
        #print("RESIZING $cap to ")
        cap *= 2
        #println("$cap")
        #println("$(site.c_age)")
        resize!(site.c_age, cap)
        #println("$(site.c_age)")
        resize!(site.c_bio, cap)
        resize!(site.c_species, cap)
        resize!(site.c_m_tot, cap)
        resize!(site.c_comp, cap)
        site.cap = cap
    end
end

@inline function compact_site!(site::Site)
    new_cap = max(site.live, 2)
    if new_cap < site.cap
            
            site.c_age = copy(resize!(site.c_age, new_cap))
            site.c_bio = copy(resize!(site.c_bio, new_cap))
            site.c_species = copy(resize!(site.c_species, new_cap))
            site.c_m_tot = copy(resize!(site.c_m_tot, new_cap))
            site.c_comp = copy(resize!(site.c_comp, new_cap))
            site.cap = new_cap
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

@inline function add_new_cohort!(site::Site, species::UIntType, age::FloatType, biomass::FloatType)
    ensure_site_cap!(site, UIntType(site.live + one(UIntType)))
    site.live += one(UIntType)
    site.c_species[site.live] = species
    site.c_age[site.live] = one(FloatType)
    site.c_bio[site.live] = biomass
end


function reproduction_step!(current_time::Int, eco_params::Array{BiomassSuccessionEcoParams}, site::Site)
    params = eco_params[site.ecocode]
    # reproduction if live cohorts
    if site.live > zero(UIntType)
        #println(shade_class, params.SUFFICIENT_LIGHT)
        #shade_probs = @view params.SUFFICIENT_LIGHT[:, site.shade_class]
        shade_probs = params.SUFFICIENT_LIGHT[site.shade_class + 1] #julia is 1-indexed
        #println(shade_probs)
        for sp in 1:length(site.sp_mature)
            if site.sp_mature[sp]
                sp_light_prob = shade_probs[params.SHADE_TOL[sp]]
                light_rng = rand(site.rng, FloatType)
                if light_rng <= sp_light_prob
                    sp_estab_prob = params.PROB_ESTAB_SPP[sp]
                    sp_estab_rng = rand(site.rng, FloatType)
                    if sp_estab_rng <= sp_estab_prob
                        new_biomass = calculate_initial_biomass(params.ANPP_MAX_SPP[sp],
                            site.B, params.B_MAX_ECO)
                        add_new_cohort!(site, UIntType(sp), one(FloatType), new_biomass)
                        site.B += new_biomass
                    end
                end
            end

        end
    end
end

function succession_step!(current_time::Int, eco_params::Array{BiomassSuccessionEcoParams}, site::Site)
    params = eco_params[site.ecocode]
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
        @assert !isnan(C) "$bio"
        site.c_comp[i] = comp
        site.c_m_tot[i] = bio
        max_age = params.LONGEVITY[sp]
        if age < max_age
            # not max age yet
            mort_rng = rand(site.rng, FloatType)
            if mort_rng > params.PROB_MORT_SPP[sp]
                m_age_factor = exp(params.D[sp] * (age / max_age - one(FloatType)))
                if current_time <= 0
                    m_age_factor += params.SPINUP_MORTALITY_FRACTION[]
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
        @assert !isnan(nbio) "$bio, mtot  $m_tot, $m_age, $m_bio anpp_act $anpp_act, $anpp_max_c, $C, $(site.c_comp[i])"

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
            shade_class +=1
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
