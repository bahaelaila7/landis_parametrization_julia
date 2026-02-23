module BiomassSuccession


include("biomass_succession.jl")
import .SuccessionModule: Site, BiomassSuccessionParams, succession_step!, calculate_initial_biomass, add_new_cohort!
import CSV, Random, Dates, Distributions as Dists, ImageFiltering
using ProgressBars, DataFrames

function load_cohorts()
    all_df = CSV.read("../data_eco_cohorts.csv", DataFrame)
    FL5_counties_ecos = ["8.3.5.65o", "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"]
    filtered_plots = in(FL5_counties_ecos).(all_df.eco)
    cdf = all_df[filtered_plots,:]

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
            rng = Random.Xoshiro(rand(rng, UInt64)),
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

function get_smoothing_window(;smoothing_window::Int = Int(3), smoothing_variance::Float32 =1.0f0)
    w = ((-smoothing_window:smoothing_window) ./ smoothing_variance) .^ 2.0f0 .* -0.5f0 .|> exp
    w ./= sum(w)
    return w
end

function smoothen_bin_cdf(p; w::Vector{Float32}=get_smoothing_window(), age_bins::Vector{Int}= [5,8,13,20,25,40,60,80])
    pc = smooth_ages(;ages = p, smoothing_window = w)
    pc_bin = bin_ages(pc; age_bins = age_bins)
    pc_bin_cdf = cumsum(pc_bin)
    if pc_bin_cdf[end] > 0.0f0
        pc_bin_cdf ./= pc_bin_cdf[end]
    end
    return pc_bin_cdf
    
end

function wasser1(pc_bin_cdf::Vector{Float32}, qc_bin_cdf::Vector{Float32}; age_bins ::Vector{Int}= [5,8,13,20,25,40,60,80])
    bin_widths = age_bins .- [0;age_bins[begin:end-1]]
    wasser_1 = bin_widths .* abs.(qc_bin_cdf - pc_bin_cdf)[begin:end-1]
    return sum(wasser_1)
end
#function smoothen_ages(;smoothing_window,
function smooth_ages(;ages, smoothing_window)
    smoothed_ages = ImageFiltering.imfilter(ages, smoothing_window, "symmetric")
    smoothed_ages ./= sum(smoothed_ages)
end

function bin_ages(ages; age_bins=[5,8,13,20,25,40,60,80], open_last=true)
    bins = length(age_bins)
    if open_last
        bins += 1
    end
    bs = zeros(bins)
    current_bin = 1
    current_age = age_bins[current_bin]
    for (i, a) in enumerate(ages)
        if i >= current_age
            current_bin += 1
            if current_bin > length(age_bins)
                if open_last
                    current_age = Inf
                else
                    break
                end
            else
                current_age = age_bins[current_bin]
            end
        end
        bs[current_bin] += ages[i]
    end
    return bs
end

function get_spinup_cohorts(df::DataFrame)
    spinup_cohorts = df[df.year_deficit .<= 0, :]
    spinup_cohorts = sort!(spinup_cohorts, :year_deficit)
    return spinup_cohorts
end
function get_initial_cohorts(df::DataFrame)
    return df[df.year_deficit .== 0, :]
end
function initialize_sites!(initial_cohorts::DataFrame, sites::Vector{Site})
    ## current_year will go down to -1, since the last estab cohort
    ## would be 1 year old, so a year before the last start_measdate
    for row in eachrow(initial_cohorts)
        #check and add cohort
        site = sites[row.plot_id]
        #print(site)
        # make it active if not already
        site.active = true
        add_new_cohort!(site, row.species_id, row.age_calc, row.agb_sum)
        #println("Adding cohort $(row.species_symbol_map) to ", row.plot_id)
    end

end
function mark_estab_year!(df::DataFrame)
    println(minimum(df.age_calc))
    #show(df[df.age_calc .< 0,:]) #.age_calc .= 1
    year_estab = df.measdate .- (df.age_calc .|> Dates.Year)
    oldest = minimum(year_estab)
    last = maximum(df.start_measdate)
    println("Oldest cohort established: $oldest")
    println("First measurement date: $last")
    df.year_deficit .= Dates.value.(year_estab .- last)./365.25 .|> round .|> Int
end

function spinup_cohorts!(spinup_cohorts::DataFrame, sites::Vector{Site}, params::BiomassSuccessionParams)
    #show(spinup_cohorts.year_deficit)
    current_year = minimum(spinup_cohorts.year_deficit)
    ## current_year will go down to -1, since the last estab cohort
    ## would be 1 year old, so a year before the last start_measdate
    #pbar = ProgressBar(total = -current_year)
    for row in eachrow(spinup_cohorts)
        while current_year < row.year_deficit
            #grow all active
            Threads.@threads for site in sites
                if site.active
                    #println(site.mapcode)
                    succession_step!(current_year, params, site)
                end
            end
            current_year += 1
            #update(pbar)
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
    #update(pbar)
    # cohorts with year_deficit = 0 will have been added but not succeeded yet

    
end


function generate_biomass_params(rng::Random.AbstractRNG, n_species::UInt, n_ecoregions::UInt)
    # TODO: species that do not show up for a specific ecoregion, make all their prob_estab = 0
    SPINUP_MORTALITY_FRACTION = 0.15f0 #rand(Dists.Uniform(0f0,0.20f0))
    #println(typeof(SPINUP_MORTALITY_FRACTION))

    S = rand(rng, Dists.truncated(Dists.Normal(0.5,1.0), 0.01,1.0), n_species) .|> Float32 #Random.rand(rng, Float32, n_species),#
    #println(typeof(S))
    D = rand(rng, Dists.truncated(Dists.Normal(15,10),5,25), n_species) .|> Float32
    #println(typeof(D))
    LONGEVITY = rand(rng, Dists.truncated(Dists.Normal(200,100), 100,300), n_species) .|> Float32
    #println(typeof(LONGEVITY))
    SHADE_TOL = rand(rng, Dists.DiscreteUniform(1,5), n_species) .|> UInt32 # ::Vector{Float32}
    #println(typeof(SHADE_TOL))
    MATURITY = rand(rng, Dists.DiscreteUniform(3,40), n_species) .|> Float32 #::Vector{Float32}
    #println(typeof(MATURITY))

    PROB_MORT_SPP = rand(rng, Dists.Uniform(), n_ecoregions, n_species) .|> Float32  #::Matrix{Float32}
    #println(typeof(PROB_MORT_SPP))
    PROB_ESTAB_SPP = rand(rng, Dists.Uniform(), n_ecoregions, n_species) .|> Float32  #::Matrix{Float32}
    #println(typeof(PROB_ESTAB_SPP))
    ANPP_MAX_SPP = rand(rng, Dists.truncated(Dists.Normal(2500, 100), 2400, 2500), n_ecoregions, n_species) .|> Float32 
    #println(typeof(ANPP_MAX_SPP))
    B_MAX_SPP = rand(rng, Dists.truncated(Dists.Normal(2500, 100), 2400, 2500), n_ecoregions, n_species) .|> Float32 
    #println(typeof(B_MAX_SPP))
    #println(B_MAX_SPP)
    B_MAX_ECO = vec(maximum(B_MAX_SPP; dims=2)) #Float32[0.0,0.0],#::Vector{Float32}
    #println(typeof(B_MAX_ECO))
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
    #println(typeof(SUFFICIENT_LIGHT))

    # by ecoregion
    firstMINRel = rand(rng, Dists.Uniform(0.0,0.5), n_ecoregions) .|> Float32 
    #MIN_REL_BIOMASS = Float32[[0.25, 0.45, 0.56, 0.70, 0.90] for _ in 1:2]
    # (shade_class x eco) -> bio percent
    # again, column major, pickout the relevant column for ecoregion
    MIN_REL_BIOMASS = [fmin+k*0.10f0
                        for k in 0:5,
                        fmin in firstMINRel]
    #println(typeof(MIN_REL_BIOMASS))
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
    mark_estab_year!(splots)
    spinup_cohorts = get_spinup_cohorts(splots)
    initial_cohorts = get_initial_cohorts(splots)
    println("beginning trials")
    for trials in ProgressBar(1:100)
        params = generate_biomass_params(RNG, UInt(n_species), UInt(n_ecoregions))
        #println(params)
        #println("###making sites")
        sites = make_sites(splots, RNG)
        #println("###Sites made, beginning spinup")
        spinup_cohorts!(spinup_cohorts, sites, params ) #[splots.measdate .== splots.start_measdate,:])
        #println("Sites spun up")
        
        #
        ## 
        ##
        ##
        ##
        ##
        #
        for current_year in 0:50 #ProgressBar(0:50)
            Threads.@threads for site in sites
                if site.active
                    #println(site.mapcode)
                    succession_step!(current_year, params, site)
                    # what years to check for this site
                    aggregate_results(current_year, site)
                    
                end
            end
        end
        for site in sites
            @assert site.old <= site.live <= site.cap "$site"
        end
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
