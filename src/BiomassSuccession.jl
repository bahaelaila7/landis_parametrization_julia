module BiomassSuccession


include("biomass_succession.jl")
import .SuccessionModule: Site, BiomassSuccessionParams, succession_step!, calculate_initial_biomass, add_new_cohort!
import CSV, Random, Dates, Distributions as Dists
using ProgressBars, DataFrames

function load_cohorts()
    all_df = CSV.read("../data_eco_cohorts.csv", DataFrame)
    cdf = all_df[all_df.eco .== "8.3.4.45a", :]

    #println(species)
    plots = combine(groupby(cdf, [:statecd, :unitcd, :countycd, :plot, :eco, :measdate, :start_measdate, :species_symbol_map, :age_calc]), nrow => :count, :agb => sum =>:agb_sum)
    splots = sort!(plots, [:measdate,:statecd, :unitcd, :countycd, :plot, :age_calc, :species_symbol_map ])
    splots.plot_id .= groupindices(groupby(splots,[:statecd, :unitcd, :countycd, :plot])) .|> UInt32
    splots.eco_id .=groupindices(groupby(splots,[:eco])) .|> UInt32
    splots.species_id .=groupindices(groupby(splots,[:species_symbol_map])) .|> UInt32

    n_species = maximum(splots.species_id)
    n_plots = maximum(splots.plot_id)
    n_ecoregions = maximum(splots.eco_id)
    #println(splots.species_id)
    return splots, n_plots, n_species, n_ecoregions

    #println(sort!(plots, :count, rev=true))
    #starting_plots = splots[splots.measdate .== splots.start_measdate, :]
    #println(starting_plots)
end
function make_sites(splots::DataFrame, rng::Random.AbstractRNG)
    eco_ids = unique(select(splots, [:eco_id]))
    n_species = maximum(splots.species_id)
    plot_ids = unique(select(splots, [:plot_id]))
    plot_eco_ids = unique(select(splots, [:plot_id, :eco_id]))
    plot_eco_ids = sort!(plot_eco_ids, [:plot_id])
    @assert (nrow(plot_ids) == nrow(plot_eco_ids)) "Error: plot_ids and plot_eco_ids are not of equal length!"

    #index = eco_id * plot_id
    sites = [ Site(
            active = false,
            rng_state = Random.rand(rng, UInt64),
            ecocode = UInt(row.eco_id),
            mapcode = UInt(row.plot_id),

            cap = UInt(2),
            old = UInt(0),
            live = UInt(0),
            B = 0.0f0,
            AGNPP = 0.0f0,
            capacityReduction = 1.0f0,
            growthReduction = 1.0f0,
            prevYearMortality = 0.0f0,
            shade_class = UInt(1),

            c_species = UInt32[0,0],
            c_age = Float32[0.0,0.0],
            c_bio = Float32[0.0,0.0],
            c_m_tot = Float32[0.0,0.0],
            c_comp = Float32[0.0,0.0],
            sp_mature = falses(n_species),
        )
        for (i, row) in enumerate(eachrow(plot_eco_ids))]
    return sites
end
function spinup_cohorts!(df::DataFrame, sites::Vector{Site}, params::BiomassSuccessionParams)
    println(minimum(df.age_calc))
    #show(df[df.age_calc .< 0,:]) #.age_calc .= 1
    year_estab = df.measdate .- (df.age_calc .|> Dates.Year)
    oldest = minimum(year_estab)
    last = maximum(df.start_measdate)
    println("Oldest cohort established: $oldest")
    println("First measurement date: $last")
    df.year_deficit .= Dates.value.(year_estab .- last)./365.25 .|> round .|> Int

    spinup_cohorts = df[df.year_deficit .<= 0, :]
    spinup_cohorts = sort!(spinup_cohorts, :year_deficit)
    #show(spinup_cohorts.year_deficit)
    current_year = minimum(df.year_deficit)

    

    ## will go down to -1, since the last estab cohort
    ## would be 1 year old, so a year before the last start_measdate
    pbar = ProgressBar(total = -current_year)
    for row in ProgressBar(eachrow(spinup_cohorts))
        while current_year < row.year_deficit
            #grow all active
            Threads.@threads for site in sites
                if site.active
                    #println(site.mapcode)
                    succession_step!(current_year, params, site)
                end
            end
            current_year += 1
            update(pbar)
        end
        #check and add cohort
        site = sites[row.plot_id]
        #print(site)
        # make it active if not already
        site.active = true
        # check if site has a young cohort of species
        # if not, add one with initial biomass calculated
        add_new_cohort = true
        sp = row.species_id
        if site.old < site.live  # there are young cohorts
            for idx in (site.old+1):site.live
                if site.c_species[idx] == sp
                    add_new_cohort = false
                end
            end
        end
        if add_new_cohort
            initial_biomass = calculate_initial_biomass(params.B_MAX_SPP[site.ecocode, sp], site.B, params.B_MAX_ECO[site.ecocode])
            add_new_cohort!(site, sp, 1f0, initial_biomass)
        end
        #println("Adding cohort $(row.species_symbol_map) to ", row.plot_id)
    end

    
end


function generate_biomass_params(rng::Random.AbstractRNG, n_species::UInt, n_ecoregions::UInt)
    # TODO: species that do not show up for a specific ecoregion, make all their prob_estab = 0
    SPINUP_MORTALITY_FRACTION = 0.15f0 #rand(Dists.Uniform(0f0,0.20f0))
    println(typeof(SPINUP_MORTALITY_FRACTION))

    S = rand(rng, Dists.truncated(Dists.Normal(0.5,1.0), 0.01,1.0), n_species) .|> Float32 #Random.rand(rng, Float32, n_species),#
    println(typeof(S))
    D = rand(rng, Dists.truncated(Dists.Normal(15,10),5,25), n_species) .|> Float32
    println(typeof(D))
    LONGEVITY = rand(rng, Dists.truncated(Dists.Normal(200,100), 100,300), n_species) .|> Float32
    println(typeof(LONGEVITY))
    SHADE_TOL = rand(rng, Dists.DiscreteUniform(1,5), n_species) .|> UInt32 # ::Vector{Float32}
    println(typeof(SHADE_TOL))
    MATURITY = rand(rng, Dists.DiscreteUniform(3,40), n_species) .|> Float32 #::Vector{Float32}
    println(typeof(MATURITY))

    PROB_MORT_SPP = rand(rng, Dists.Uniform(), n_ecoregions, n_species) .|> Float32  #::Matrix{Float32}
    println(typeof(PROB_MORT_SPP))
    PROB_ESTAB_SPP = rand(rng, Dists.Uniform(), n_ecoregions, n_species) .|> Float32  #::Matrix{Float32}
    println(typeof(PROB_ESTAB_SPP))
    ANPP_MAX_SPP = rand(rng, Dists.truncated(Dists.Normal(2500, 100), 2400, 2500), n_ecoregions, n_species) .|> Float32 
    println(typeof(ANPP_MAX_SPP))
    B_MAX_SPP = rand(rng, Dists.truncated(Dists.Normal(2500, 100), 2400, 2500), n_ecoregions, n_species) .|> Float32 
    println(typeof(B_MAX_SPP))
    #println(B_MAX_SPP)
    B_MAX_ECO = vec(maximum(B_MAX_SPP; dims=2)) #Float32[0.0,0.0],#::Vector{Float32}
    println(typeof(B_MAX_ECO))
    #println(B_MAX_ECO)
    # (shade_tol x shade_class) -> prob of sufficient light
    # julia is column major, much faster to pickout the site's shade class as one chunk
    # then reference the species' shade_tol within
    SUFFICIENT_LIGHT = Float32[
                        1.00 0.50 0.25 0.00 0.00 0.00;
                        1.00 1.00 0.50 0.25 0.00 0.00;
                        1.00 1.00 1.00 0.50 0.25 0.00;
                        1.00 1.00 1.00 1.00 0.50 0.25;
                        1.00 1.00 1.00 1.00 1.00 0.50 ]
    println(typeof(SUFFICIENT_LIGHT))

    # by ecoregion
    firstMINRel = rand(rng, Dists.Uniform(0.0,0.5), n_ecoregions) .|> Float32 
    #MIN_REL_BIOMASS = Float32[[0.25, 0.45, 0.56, 0.70, 0.90] for _ in 1:2]
    # (shade_class x eco) -> bio percent
    # again, column major, pickout the relevant column for ecoregion
    MIN_REL_BIOMASS = [fmin+k*0.10f0
                        for k in 0:5,
                        fmin in firstMINRel]
    println(typeof(MIN_REL_BIOMASS))
    #println(MIN_REL_BIOMASS)

    p = BiomassSuccessionParams(
        SPINUP_MORTALITY_FRACTION = SPINUP_MORTALITY_FRACTION ,
        D = D,
        S = S,
        LONGEVITY = LONGEVITY,
        SHADE_TOL = SHADE_TOL,
        MATURITY = MATURITY,

        ANPP_MAX_SPP = ANPP_MAX_SPP,
        B_MAX_SPP = B_MAX_SPP,
        B_MAX_ECO = B_MAX_ECO,
        PROB_MORT_SPP = PROB_MORT_SPP,
        PROB_ESTAB_SPP = PROB_ESTAB_SPP,

        SUFFICIENT_LIGHT = SUFFICIENT_LIGHT,
        MIN_REL_BIOMASS = MIN_REL_BIOMASS,
    )
    return p


end

function mutate_biomass_params(p)
    # Only mutate for species in ecoregion, no point in mutating others
    # schema: variable, element-wise pdf, final type, species-specific, ecoregion-specific
    #
    # schemes to mutate:
    # - pick one variable, pick one element or more, 
    # - types of mutation: creep, redraw
    # EA:
    # - bundle mutation rate, mutation angle.

    d = [
    (:SPINUP_MORTALITY_FRACTION, Dists.Uniform(0f0,0.20f0), Float32, false, false),
    (:S, Dists.truncated(Dists.Normal(0.5,1.0), 0.01,1.0), Float32, true, false),
    (:D, Dists.truncated(Dists.Normal(15,10),5,25), Float32, true, false),
    (:LONGEVITY, Dists.truncated(Dists.Normal(200,100), 100,300), Float32, true, false),
    (:SHADE_TOL, Dists.DiscreteUniform(1,5), Float32, true, false),
     (:MATURITY, Dists.DiscreteUniform(3,40), Float32, true, false),
    (:PROB_MORT_SPP, Dists.Uniform(), Float32, true, true),
    (:PROB_ESTAB_SPP, Dists.Uniform(), Float32, true, true),
    (:ANPP_MAX_SPP, Dists.truncated(Dists.Normal(2500, 100), 2400, 2500), Float32, true, true),
    (:B_MAX_SPP, Dists.truncated(Dists.Normal(2500, 100), 2400, 2500), Float32, true, true),
    (:firstMINRel, Dists.Uniform(0.0,0.5), Float32, false, true),
    ]
    println(d)

end



#actual entry
function main(args)
    #greet()
    RNG = Random.Xoshiro(1337)
    #return
    println("###loading data")
    splots, n_plots, n_species, n_ecoregions =load_cohorts()
    println("Plots:$n_plots, Ecos:$n_ecoregions, Species:$n_species, Measurements: $(size(splots))")
    p = generate_biomass_params(RNG, UInt(n_species), UInt(n_ecoregions))
    println(p)
    println("###making sites")
    sites = make_sites(splots, RNG)
    println("###Sites made, beginning spinup")
    spinup_cohorts!(splots, sites, p ) #[splots.measdate .== splots.start_measdate,:])
    for site in sites
        @assert site.old <= site.live <= site.cap "$site"
    end
    return
    
    println("Hello ", p, args)
    Threads.@threads for current_time in ProgressBar(0:100)
        succession_step!(current_time,p,s)
    end

end


#stub C entry
function julia_main()::Cint
    try
        main(ARGS)
        return 0
    catch err

        showerror(stderr, err, catch_backtrace())
        return 1
    end
end

end # module BiomassSuccession
