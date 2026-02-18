module SuccessionModule
    # prepare parameters for one run
    # prepare cohorts
    #   spin up
    # report outcome
    # compare outcome
    export Site, BiomassSuccessionParams, succession_step!, add_new_cohort!;
    using Base
    using Random

    Base.@kwdef mutable struct Site
        active::Bool
        rng::Random.Xoshiro
        ecocode::UInt
        mapcode::UInt

        cap:: UInt
        old::UInt
        live:: UInt
        B::Float32
        AGNPP::Float32
        capacityReduction::Float32
        growthReduction::Float32
        prevYearMortality::Float32
        shade_class::UInt

        c_species::Vector{UInt32}
        c_age::Vector{Float32}
        c_bio::Vector{Float32}
        c_m_tot::Vector{Float32}
        c_comp::Vector{Float32}
        sp_mature::Vector{Bool}

        
        
    end
    Base.@kwdef struct BiomassSuccessionParams
        SPINUP_MORTALITY_FRACTION::Float32

        D::Vector{Float32}
        S::Vector{Float32}
        
        LONGEVITY::Vector{Float32}
        SHADE_TOL::Vector{UInt32}
        MATURITY::Vector{Float32}
        B_MAX_ECO::Vector{Float32}
        
        
        ANPP_MAX_SPP::Matrix{Float32}
        B_MAX_SPP::Matrix{Float32}
        PROB_MORT_SPP::Matrix{Float32}
        PROB_ESTAB_SPP::Matrix{Float32}

        SUFFICIENT_LIGHT::Matrix{Float32}
        MIN_REL_BIOMASS::Matrix{Float32}
        
        
    end
    function ensure_site_cap!(site::Site, live::UInt)
        cap = site.cap
        if cap < live
            #print("RESIZING $cap to ")
            cap *= 2
            #println("$cap")
            #println("$(site.c_age)")
            site.c_age = resize!(site.c_age, cap)
            #println("$(site.c_age)")
            site.c_bio = resize!(site.c_bio, cap)
            site.c_species = resize!(site.c_species, cap)
            site.c_m_tot = resize!(site.c_m_tot, cap)
            site.c_comp = resize!(site.c_comp, cap)
            site.cap = cap
        end
    end

    function calculate_initial_biomass(sp_max_anpp::Float32, site_b::Float32, b_max_eco::Float32)::Float32
        b = exp(-1.6f0 * site_b / b_max_eco)
        if b < 1.0f0
            b = 1.0f0
        end
        b *= sp_max_anpp
        if b < 2.0f0
            b = 2.0f0
        end
        return b
    end
    
    function add_new_cohort!(site::Site, species::UInt32, age::Float32, biomass::Float32)
        ensure_site_cap!(site,site.live + 1)
        site.live += 1
        site.c_species[site.live] = species
        site.c_age[site.live] = 1.0f0
        site.c_bio[site.live] = biomass
    end

    function succession_step!(current_time::Int, params::BiomassSuccessionParams, site::Site)
        B = 0.0f0
        C = 0.0f0
        #RNG = site.rng Random.seed!(site.rng_state)
        site.sp_mature .= false

        # advancing age, summing site biomass, computing competition, mortality due to age or random act of god
        for i in 1:site.live
            site.c_age[i] += 1.0f0
            age = site.c_age[i]
            sp = site.c_species[i]
            if age >= params.MATURITY[sp]
                site.sp_mature[sp] = true
            end
            bio = site.c_bio[i]
            B += bio
            comp = bio ^ 0.95f0
            #println("Bio $bio, Comp $(comp)")
            if comp < 1.0f0
                comp = 1.0f0
            end
            C += comp
            @assert !isnan(C) "$bio"
            site.c_comp[i] = comp
            site.c_m_tot[i] = bio
            max_age = params.LONGEVITY[sp]
            if age < max_age
                # not max age yet
                mort_rng = rand(site.rng, Float32)
                if mort_rng > params.PROB_MORT_SPP[site.ecocode, sp]
                    m_age_factor = exp(params.D[sp] * (age/max_age - 1.0f0))
                    if current_time <= 0
                        m_age_factor += params.SPINUP_MORTALITY_FRACTION
                    end
                    if m_age_factor < 1.0f0
                        site.c_m_tot[i] *= m_age_factor
                    end
                end
            end
        end

        new_B = 0.0f0
        AGNPP = 0.0f0
        M_TOT = 0.0f0
        B_ACT = 0.0f0

        last = site.live
        i = 1
        while i <= last
            age = site.c_age[i]
            bio = site.c_bio[i]
            sp = site.c_species[i]
            b_max = params.B_MAX_SPP[site.ecocode, sp] * site.capacityReduction
            b_pot = b_max - B - bio
            if b_pot < 1.0f0
                b_pot = 1.0f0
            end
            # TODO: check this condition
            if site.capacityReduction >= 1.0f0 && b_pot < site.prevYearMortality
                b_pot = site.prevYearMortality
            end
            
            b_ap = bio / b_pot
            b_ap_s = b_ap ^ params.S[sp]
            anpp_act = b_ap_s * exp(1.0f0 - b_ap_s)
            #@assert !isnan(anpp_act) "$b_ap_s, $(params.S[sp])"
            if anpp_act > 1.0f0
                anpp_act = 1.0f0
            end
            site.c_comp[i] /= C

            anpp_max_c = params.ANPP_MAX_SPP[site.ecocode, sp] * site.c_comp[i]
            #@assert !isnan(anpp_max_c) "$C, $(site.c_comp[i]), $(params.ANPP_MAX_SPP[site.ecocode, sp])"
            anpp_act *= anpp_max_c

            if site.growthReduction > 0.0f0
                anpp_act *= 1.0f0 - site.growthReduction
            end
            AGNPP += anpp_act

            # growth mortality
            m_bio = anpp_max_c
            if m_bio <=1.0f0
                m_bio *= (2.0f0*b_ap)/(1.0f0+b_ap)
            end
            if m_bio > bio
                m_bio = bio
            end
            if site.growthReduction > 0.0f0
                m_bio *= 1.0f0 - site.growthReduction
            end

            # remove age mortality from anpp and growth mortality
            m_age = site.c_m_tot[i]
            anpp_act -= m_age
            if anpp_act < 1.0f0
                anpp_act = 1.0f0
            end
            m_bio -= m_age
            if m_bio < 0.0f0
                m_bio = 0.0f0
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
            senescent = (nbio <= 1.0f-8)
            if !senescent
                new_B += nbio
                if age > 5.0f0
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
        site_b_max = params.B_MAX_ECO[site.ecocode]
        site_b_pot = site_b_max - site.prevYearMortality
        #println(typeof(B_ACT), B_ACT)
        #println(typeof(site_b_pot), site_b_pot)
        if B_ACT > site_b_pot
            B_ACT = site_b_pot
        end
        b_am = B_ACT / site_b_max
        shade_classes = @view params.MIN_REL_BIOMASS[:, site.ecocode]
        shade_class = 1
        for sc in 1:(length(shade_classes) - 1)
            if b_am > shade_classes[sc]
                shade_class += 1
            else
                break
            end
        end

        # before reproduction, all cohorts on site are now old
        site.old = site.live
        
        # reproduction if live cohorts
        if site.live > 0
            #println(shade_class, params.SUFFICIENT_LIGHT)
            shade_probs = @view params.SUFFICIENT_LIGHT[:, shade_class]
            #println(shade_probs)
            for sp in 1:length(site.sp_mature)
                if site.sp_mature[sp]
                    sp_light_prob = shade_probs[params.SHADE_TOL[sp]]
                    light_rng = rand(site.rng, Float32)
                    if light_rng <= sp_light_prob
                        sp_estab_prob = params.PROB_ESTAB_SPP[site.ecocode, sp]
                        sp_estab_rng = rand(site.rng, Float32)
                        if sp_estab_rng <= sp_estab_prob
                            new_biomass= calculate_initial_biomass(params.ANPP_MAX_SPP[site.ecocode, sp],
                                                                              new_B, site_b_max)
                            add_new_cohort!(site, UInt32(sp),1f0,new_biomass)
                        end
                    end
                end

            end
        end
            
    #println(current_time, "done")
    end
end
