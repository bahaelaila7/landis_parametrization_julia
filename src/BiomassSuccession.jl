module BiomassSuccession


include("biomass_succession.jl")
import .SuccessionModule: Site, BiomassSuccessionParams, succession_step
import CSV, Random, Dates
using ProgressBars, DataFrames

function load_cohorts()
    all_df = CSV.read("../data_eco_cohorts.csv", DataFrame)
    cdf = all_df[all_df.eco .== "8.3.4.45a", :]
    species = unique(cdf.species_symbol_map)
    #println(species)
    plots = combine(groupby(cdf, [:statecd, :unitcd, :countycd, :plot, :eco, :measdate, :start_measdate, :species_symbol_map, :age_calc]), nrow => :count, :agb => sum =>:agb_sum)
    splots = sort!(plots, [:measdate,:statecd, :unitcd, :countycd, :plot, :age_calc, :species_symbol_map ])
    return splots

    #println(sort!(plots, :count, rev=true))
    #starting_plots = splots[splots.measdate .== splots.start_measdate, :]
    #println(starting_plots)
end
function make_sites(splots::DataFrame, rng::Random.AbstractRNG)
    splots.plot_id .= groupindices(groupby(splots,[:statecd, :unitcd, :countycd, :plot]))
    splots.eco_id .=groupindices(groupby(splots,[:eco]))


    eco_ids = unique(select(splots, [:eco_id]))
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
            live = UInt(0),
            B = 0.0f0,
            AGNPP = 0.0f0,
            capacityReduction = 1.0f0,
            growthReduction = 1.0f0,
            prevYearMortality = 0.0f0,
            shade_class = UInt(1),

            c_species = UInt32[0,0],
            c_age = Float32[0.0,0.0],
            c_biomass = Float32[0.0,0.0],
            c_m_tot = Float32[0.0,0.0],
            c_comp = Float32[0.0,0.0],
            sp_mature = Bool[false,false],
        )
        for (i, row) in enumerate(eachrow(plot_eco_ids))]
    return sites
end
function spinup_cohorts(df::DataFrame, sites::Vector{Site}, params::BiomassSuccessionParams)
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
                    succession_step(current_year, params, site)
                end
            end
            current_year += 1
            update(pbar)
        end
        #check and add cohort
        site = sites[row.plot_id]
        #println("Adding cohort $(row.species_symbol_map) to ", row.plot_id)
        site.active = true
    end

    
end
#actual entry
function main(args)
    #greet()
    RNG = Random.Xoshiro(1337)
    splots =load_cohorts()
    sites = make_sites(splots, RNG)
    p = BiomassSuccessionParams(
        SPINUP_MORTALITY_FRACTION = 0.15f0,

        D = Float32[0.0,0.0],#
        S = Float32[0.0,0.0],#::Vector{Float32}
        LONGEVITY = Float32[0.0,0.0],#::Vector{Float32}
        SHADE_TOL = Float32[0.0,0.0],#::Vector{Float32}
        MATURITY = Float32[0.0,0.0],#::Vector{Float32}
        B_MAX_ECO = Float32[0.0,0.0],#::Vector{Float32}
        
        
        ANPP_MAX_SPP = Float32[0.0 0.0; 0.0 0.0],#::Matrix{Float32}
        B_MAX_SPP = Float32[0.0 0.0; 0.0 0.0],#::Matrix{Float32}
        PROB_MORT_SPP = Float32[0.0 0.0; 0.0 0.0],#::Matrix{Float32}
        PROB_ESTAB_SPP = Float32[0.0 0.0; 0.0 0.0],#::Matrix{Float32}

        SUFFICIENT_LIGHT = Float32[0.0 0.0; 0.0 0.0],#::Matrix{Float32}
        MIN_REL_BIOMASS = Float32[0.0 0.0; 0.0 0.0],#::Matrix{Float32}

    )
    spinup_cohorts(splots, sites, p ) #[splots.measdate .== splots.start_measdate,:])
    return
    
    println("Hello ", p, args)
    Threads.@threads for current_time in ProgressBar(0:100)
        succession_step(current_time,p,s)
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
