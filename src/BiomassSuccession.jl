module BiomassSuccession


include("biomass_succession.jl")
import .SuccessionModule: Site, BiomassSuccessionParams, succession_step!, reproduction_step!, calculate_initial_biomass, add_new_cohort!, FloatType, UIntType
import CSV, Random, Dates, Distributions as Dists, ImageFiltering, StatsBase, Term.Progress as TProgress
using DataFrames
#using Profile, ProfileSVG
#using ProgressBars

struct AgeBins
    bins_idx::Vector{Int}
    bin_widths::Vector{FloatType}
    last_bin_open::Bool
    function AgeBins(; bins_idx::Vector{Int}, last_bin_open::Bool)
        new(bins_idx,
            get_bin_widths(; age_bins=bins_idx, last_bin_open=last_bin_open),
            last_bin_open)
    end
end

Base.@kwdef struct LossParams
    age_bins::AgeBins
    smoothing_weights::Vector{FloatType}
    lambda::FloatType = 1.0f-2
    EPS::FloatType = 1.0f-8
end

Base.@kwdef struct SPDFRecord
    sp_agb_sum::FloatType
    sp_age_cdf::Vector{FloatType}
end
Base.@kwdef struct SPDFGroundTruth
    keys::BitVector
    records::Dict{UIntType,SPDFRecord}
end
Base.@kwdef struct SiteLoss
    sp_w_loss::Vector{FloatType}
    sp_age_loss::Vector{FloatType}
    site_agb_loss::FloatType
end

function load_cohorts()
    all_df = CSV.read("../data_eco_cohorts_cn.csv", DataFrame)
    FL5_counties_ecos = ["8.3.5.65o", "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"]
    filtered_plots = in(FL5_counties_ecos).(all_df.eco)
    cdf = all_df[filtered_plots, :]

    plots = combine(groupby(cdf, [:plt_cn, :statecd, :unitcd, :countycd, :plot, :eco, :measdate, :species_symbol_map, :age_calc], sort=false), nrow => :count, :agb => sum => :agb_sum)
    # start_measdate = 
    start_measdates = combine(groupby(plots, [:statecd, :unitcd, :countycd, :plot], sort=false)) do rows
        (; start_measdate=[minimum(rows.measdate)])
    end
    plots_measdate = innerjoin(plots, start_measdates, on=[:statecd, :unitcd, :countycd, :plot])
    splots = sort!(plots_measdate, [:measdate, :statecd, :unitcd, :countycd, :plot, :age_calc, :species_symbol_map])

    splots.plot_id .= groupindices(groupby(splots, [:statecd, :unitcd, :countycd, :plot])) .|> UIntType
    splots.eco_id .= groupindices(groupby(splots, [:eco])) .|> UIntType
    splots.species_id .= groupindices(groupby(splots, [:species_symbol_map])) .|> UIntType

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
    sites = [Site(
        active=false,
        rng=Random.Xoshiro(rand(rng, UInt64)),
        ecocode=UIntType(row.eco_id),
        mapcode=UIntType(row.plot_id), cap=UInt(2),
        ref_cn=UIntType(row.plot_id),
        old=UIntType(0),
        live=UIntType(0),
        B=0.0f0,
        AGNPP=0.0f0,
        capacityReduction=1.0f0,
        growthReduction=1.0f0,
        prevYearMortality=0.0f0,
        shade_class=UIntType(1), c_species=UIntType[0, 0],
        c_age=FloatType[0.0, 0.0],
        c_bio=FloatType[0.0, 0.0],
        c_m_tot=FloatType[0.0, 0.0],
        c_comp=FloatType[0.0, 0.0],
        sp_mature=falses(n_species),
    )
             for (i, row) in enumerate(eachrow(plot_eco_ids))]
    return sites
end

@inline function get_smoothing_window(; smoothing_window::Int=Int(3), smoothing_variance::FloatType=1.0f0)
    w = ((-smoothing_window:smoothing_window) ./ smoothing_variance) .^ 2.0f0 .* -0.5f0 .|> exp
    w ./= sum(w)
    return w
end

@inline function smoothen_bin_cdf(p; w::Vector{FloatType}, age_bins::AgeBins)::Vector{FloatType}
    pc = smooth_ages(; ages=p, smoothing_window=w)
    pc_bin = bin_ages(pc; age_bins=age_bins.bins_idx, last_bin_open=age_bins.last_bin_open)
    pc_bin_cdf = cumsum(pc_bin)
    if pc_bin_cdf[end] > 0.0f0
        pc_bin_cdf ./= pc_bin_cdf[end]
    end
    return pc_bin_cdf

end

@inline function bins_loss2(pc_bin_cdf::Vector{FloatType}, qc_bin_cdf::Vector{FloatType}; loss_params::LossParams)::Float32
    # warning: 
    # Technically W1 is not defined if one or both distribution collapsed (0 everywhere)
    # 0 distance if both are collapsed while +Inf if only one is sensible,
    # BUT Inf will make all aggregates useless and will make the search directionless
    # tagging along a small log(m+eps) such that if aggregate is 0 eps penalize it heavily
    wasser1 = 0.0f0
    a, b = 0.0f0, 0.0f0
    if pc_bin_cdf[end] > loss_params.EPS
        a = pc_bin_cdf[end]
    end

    if qc_bin_cdf[end] > loss_params.EPS
        b = qc_bin_cdf[end]
    end

    if a > loss_params.EPS && b > loss_params.EPS
        wasser1 += sum(loss_params.age_bins.bin_widths .* abs.(qc_bin_cdf - pc_bin_cdf)[begin:end-1])
    end

    if a > loss_params.EPS || b > loss_params.EPS
        wasser1 += loss_params.lambda * abs(log10(a + loss_params.EPS) - log10(b + loss_params.EPS))
    end

    return wasser1
end
@inline function bins_loss(pc_bin_cdf::Union{Missing,Vector{FloatType}}, qc_bin_cdf::Union{Missing,Vector{FloatType}}; loss_params::LossParams)::Float32
    # warning: 
    # Technically W1 is not defined if one or both distribution collapsed (0 everywhere)
    # 0 distance if both are collapsed while +Inf if only one is sensible,
    # BUT Inf will make all aggregates useless and will make the search directionless
    # tagging along a small log(m+eps) such that if aggregate is 0 eps penalize it heavily
    wasser1 = 0.0f0
    a, b = 0.0f0, 0.0f0
    if pc_bin_cdf !== missing && pc_bin_cdf[end] > loss_params.EPS
        a = pc_bin_cdf[end]
    end

    if qc_bin_cdf !== missing && qc_bin_cdf[end] > loss_params.EPS
        b = qc_bin_cdf[end]
    end

    if a > loss_params.EPS && b > loss_params.EPS
        wasser1 += sum(loss_params.age_bins.bin_widths .* abs.(qc_bin_cdf - pc_bin_cdf)[begin:end-1])
    end

    if a > loss_params.EPS || b > loss_params.EPS
        wasser1 += loss_params.lambda * abs(log10(a + loss_params.EPS) - log10(b + loss_params.EPS))
    end

    return wasser1
end
#function smoothen_ages(;smoothing_window,
@inline function smooth_ages(; ages::Vector{FloatType}, smoothing_window::Vector{FloatType})::Vector{Float32}
    smoothed_ages = ImageFiltering.imfilter(ages, smoothing_window, "symmetric")
    smoothed_ages ./= sum(smoothed_ages)
    return smoothed_ages
end

function smoothen_ref_years(df::DataFrame, loss_params::LossParams, max_age::Int)::DataFrame
    spdf = combine(groupby(df, [:plot_id, :eco_id, :measdate, :start_measdate, :species_id])) do rows
        ages = zeros(FloatType, max_age)
        for row in eachrow(rows)
            ages[row.age_calc] += row.agb_sum
        end
        row = rows[1, :]
        sim_year = Dates.value(row.measdate - row.start_measdate) ./ 365.25 .|> round .|> Int
        @assert sim_year >= 0 "negative sim_year $row"
        cdf = smoothen_bin_cdf(ages; w=loss_params.smoothing_weights, age_bins=loss_params.age_bins)
        (; sim_year=[sim_year], data_agb_sum=[sum(rows.agb_sum)], data_agbs_cdf=[cdf])
    end
    return spdf


    #plot_id x eco_id, measdate, sim_year, swhd, spgrpcd, species_id -> smoothened(biomass by age)
    #plot_id x eco_id, measdate, sim_year, swhd, spgrpcd -> smoothened_binned(biomass by age)
    #plot_id x eco_id, measdate, sim_year, swhd -> biomass by age

end
function get_bin_widths(; age_bins::Vector{Int}, last_bin_open::Bool)
    # returns the bin widths for wasser1 (ie K-1 widths)
    if last_bin_open
        @assert length(age_bins) > 0 "insufficint bins, must be at least 1"
        return age_bins .- [0; age_bins[begin:end-1]]
    else
        @assert length(age_bins) > 1 "insufficint bins, must be at least 2"
        return age_bins[begin:end-1] .- [0; age_bins[begin:end-2]]
    end

end

@inline function bin_ages(ages::Vector{FloatType}; age_bins::Vector{Int}, last_bin_open::Bool)::Vector{FloatType}
    bins = length(age_bins)
    if last_bin_open
        bins += 1
    end
    bs = zeros(FloatType, bins)
    current_bin = 1
    current_age = age_bins[current_bin]
    for (i, a) in enumerate(ages)
        if i >= current_age
            current_bin += 1
            if current_bin > length(age_bins)
                if last_bin_open
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
    spinup_cohorts = df[df.year_deficit.<=0, :]
    spinup_cohorts = sort!(spinup_cohorts, :year_deficit)
    return spinup_cohorts
end
function get_initial_cohorts(df::DataFrame)
    return df[df.year_deficit.==0, :]
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
function get_site_sim_years(df::DataFrame)::DataFrame
    return combine(groupby(df, [:plot_id, :eco_id])) do rows
        (; sim_years=[unique(sort(rows.sim_year))])
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
    df.year_deficit .= Dates.value.(year_estab .- last) ./ 365.25 .|> round .|> Int
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
                    reproduction_step!(current_year, params, site)
                end
            end
            current_year += 1
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
            add_new_cohort!(site, sp, 1.0f0, initial_biomass)
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

    S = rand(rng, Dists.truncated(Dists.Normal(0.5, 1.0), 0.01, 1.0), n_species) .|> FloatType #Random.rand(rng, FloatType, n_species),#
    #println(typeof(S))
    D = rand(rng, Dists.truncated(Dists.Normal(15, 10), 5, 25), n_species) .|> FloatType
    #println(typeof(D))
    LONGEVITY = rand(rng, Dists.truncated(Dists.Normal(200, 100), 100, 300), n_species) .|> FloatType
    #println(typeof(LONGEVITY))
    SHADE_TOL = rand(rng, Dists.DiscreteUniform(1, 5), n_species) .|> UInt32 # ::Vector{FloatType}
    #println(typeof(SHADE_TOL))
    MATURITY = rand(rng, Dists.DiscreteUniform(3, 40), n_species) .|> FloatType #::Vector{FloatType}
    #println(typeof(MATURITY))

    PROB_MORT_SPP = rand(rng, Dists.Uniform(), n_ecoregions, n_species) .|> FloatType  #::Matrix{FloatType}
    #println(typeof(PROB_MORT_SPP))
    PROB_ESTAB_SPP = rand(rng, Dists.Uniform(), n_ecoregions, n_species) .|> FloatType  #::Matrix{FloatType}
    #println(typeof(PROB_ESTAB_SPP))
    ANPP_MAX_SPP = rand(rng, Dists.truncated(Dists.Normal(2500, 100), 2400, 2500), n_ecoregions, n_species) .|> FloatType
    #println(typeof(ANPP_MAX_SPP))
    B_MAX_SPP = rand(rng, Dists.truncated(Dists.Normal(2500, 100), 2400, 2500), n_ecoregions, n_species) .|> FloatType
    #println(typeof(B_MAX_SPP))
    #println(B_MAX_SPP)
    B_MAX_ECO = vec(maximum(B_MAX_SPP; dims=2)) #FloatType[0.0,0.0],#::Vector{FloatType}
    #println(typeof(B_MAX_ECO))
    #println(B_MAX_ECO)
    # (shade_tol x shade_class) -> prob of sufficient light
    # julia is column major, much faster to pickout the site's shade class as one chunk
    # then reference the species' shade_tol within
    SUFFICIENT_LIGHT = FloatType[
        1.00 0.50 0.25 0.00 0.00 0.00;
        1.00 1.00 0.50 0.25 0.00 0.00;
        1.00 1.00 1.00 0.50 0.25 0.00;
        1.00 1.00 1.00 1.00 0.50 0.25;
        1.00 1.00 1.00 1.00 1.00 0.50]
    #println(typeof(SUFFICIENT_LIGHT))

    # by ecoregion
    firstMINRel = rand(rng, Dists.Uniform(0.0, 0.5), n_ecoregions) .|> FloatType
    #MIN_REL_BIOMASS = FloatType[[0.25, 0.45, 0.56, 0.70, 0.90] for _ in 1:2]
    # (shade_class x eco) -> bio percent
    # again, column major, pickout the relevant column for ecoregion
    MIN_REL_BIOMASS = [fmin + k * 0.10f0
                       for k in 0:5,
                       fmin in firstMINRel]
    #println(typeof(MIN_REL_BIOMASS))
    #println(MIN_REL_BIOMASS)

    p = BiomassSuccessionParams(
        SPINUP_MORTALITY_FRACTION=SPINUP_MORTALITY_FRACTION,
        D=D,
        S=S,
        LONGEVITY=LONGEVITY,
        SHADE_TOL=SHADE_TOL,
        MATURITY=MATURITY, ANPP_MAX_SPP=ANPP_MAX_SPP,
        B_MAX_SPP=B_MAX_SPP,
        B_MAX_ECO=B_MAX_ECO,
        PROB_MORT_SPP=PROB_MORT_SPP,
        PROB_ESTAB_SPP=PROB_ESTAB_SPP, SUFFICIENT_LIGHT=SUFFICIENT_LIGHT,
        MIN_REL_BIOMASS=MIN_REL_BIOMASS,
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
        (:SPINUP_MORTALITY_FRACTION, Dists.Uniform(0.0f0, 0.20f0), FloatType, false, false),
        (:S, Dists.truncated(Dists.Normal(0.5, 1.0), 0.01, 1.0), FloatType, true, false),
        (:D, Dists.truncated(Dists.Normal(15, 10), 5, 25), FloatType, true, false),
        (:LONGEVITY, Dists.truncated(Dists.Normal(200, 100), 100, 300), FloatType, true, false),
        (:SHADE_TOL, Dists.DiscreteUniform(1, 5), FloatType, true, false),
        (:MATURITY, Dists.DiscreteUniform(3, 40), FloatType, true, false),
        (:PROB_MORT_SPP, Dists.Uniform(), FloatType, true, true),
        (:PROB_ESTAB_SPP, Dists.Uniform(), FloatType, true, true),
        (:ANPP_MAX_SPP, Dists.truncated(Dists.Normal(2500, 100), 2400, 2500), FloatType, true, true),
        (:B_MAX_SPP, Dists.truncated(Dists.Normal(2500, 100), 2400, 2500), FloatType, true, true),
        (:firstMINRel, Dists.Uniform(0.0, 0.5), FloatType, false, true),
    ]
    println(d)

end
function calculate_site_loss2(current_year::Int, site::Site, spdf_plt::SPDFGroundTruth, loss_params::LossParams)::SiteLoss
    #Sort by species
    c_species = @view site.c_species[1:site.live]
    max_age = maximum(@view site.c_age[1:site.live])
    #get indices of species sorted
    p = sortperm(c_species)
    n_species = length(site.sp_mature)

    @assert n_species == length(spdf_plt.keys) "species numbers do not match"

    ages = Vector{FloatType}(undef, max_age)
    sp_start_idx = 1
    last_sp = sps[sp_start_idx]

    insite = falses(n_species)

    #initialize losses 
    sp_w_loss = zeros(FloatType, n_species)
    sp_agb_loss = zeros(FloatType, n_species)
    site_agb_loss = 0.0f0
    # process sp's in site, sorted by species
    for i in 1:length(p)

        # get the 
        sp = @inbounds c_species[i]
        insite[sp] = true
        if i == length(p) && last_sp != sp
            last_sp = sp
            sp_start_idx = i
        end

        if sp != last_sp || i == length(p)
            # conclude sp 
            sp_end_idx = i - 1
            if sp == last_sp
                # when length(p) was the cause only
                sp_end_idx += 1
            end
            sim_agb_sum = sum(@view site.agbs[p])
            log_diff = log10(sim_agb_sum + loss_params.EPS)
            sp_agb_loss[sp] = sim_agb_sum
            site_agb_loss += sim_agb_sum
            if spdf_plt.keys[sp]
                rec = @inbounds spdf_plt.records[sp]
                ages .= 0.0f0
                for a in @view p[sp_start_idx:i]
                    ages[UIntType(site.c_age[a])] = site.c_bio[a]
                end
                sim_age_cdf = smoothen_bin_cdf(ages; w=loss_params.smoothing_weights, age_bins=loss_params.age_bins)
                @assert length(sim_age_cdf) == length(rec.sp_age_cdf) "cdf bins are not the same size"
                sp_w_loss[sp] = sum(loss_params.age_bins.bin_widths .* abs.(sim_age_cdf - rec.sp_age_cdf)[begin:end-1])
                log_diff -= log10(rec.sp_agb_sum + loss_params.EPS)
                sp_agb_loss[sp] = abs(sim_agb_sum - rec.sp_agb_sum)
                site_agb_loss -= rec.sp_agb_sum
            end
            sp_w_loss[sp] += loss_params.lambda * abs(log_diff)

            last_sp = sp
            sp_start_idx = i
        end
    end

    for sp in (1:length(spdf_plt.keys))[spdf_plt.keys&!insite]
        @inbounds rec = spdf_plt.records[UIntType(sp)]
        sp_agb_loss[sp] = rec.sp_agb_sum
        sp_w_loss -= loss_params.lambda * abs(log10(rec.sp_agb_sum + loss_params.EPS))
        site_agb_loss -= rec.sp_agb_sum
    end

    return SiteLoss(sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=abs(site_agb_loss))

end

function process_site_results(current_year::Int, site::Site, loss_params::LossParams)::DataFrame

    #if site.live == 0
    #    return DataFrame(plot_id=UInt32[], eco_id=UInt32[], species_id = UInt32, sim_agb_sum=FloatType[], sim_agbs_cdf=FloatType[])
    #end

    c_age = (@view site.c_age[1:site.live]) .|> Int
    c_bio = @view site.c_bio[1:site.live]
    c_species = @view site.c_species[1:site.live]
    @assert abs(site.B - sum(c_bio)) < 1.0f-2 "Site[$(site.mapcode)]: B $(site.B) not equal sum(c_bio) $(sum(c_bio)), $(site)"
    df = DataFrame(:species_id => c_species, :sim_age => c_age, :sim_agb => c_bio)
    sort!(df, [:sim_age])

    if nrow(df) == 0
        return DataFrame(plot_id=UInt32, eco_id=UInt32, sim_year=Int, species_id=UInt32, agb_total=FloatType, sim_agb_sum=FloatType, sim_agbs_cdf=Float32[])
    end

    site_df = combine(groupby(df, [:species_id])) do rows

        max_age = maximum(rows.sim_age)

        ages = zeros(FloatType, max_age)
        for row in eachrow(rows)
            ages[row.sim_age] += row.sim_agb
        end
        cdf = smoothen_bin_cdf(ages; w=loss_params.smoothing_weights, age_bins=loss_params.age_bins)
        (; sim_agb_sum=[sum(rows.sim_agb)], sim_agbs_cdf=[cdf])
    end
    site_df.plot_id .= site.mapcode
    site_df.eco_id .= site.ecocode
    site_df.sim_year .= current_year
    site_df.agb_total .= site.B

    return site_df
end

function calculate_site_loss(current_sim_year::Int, spdf::DataFrame, site::Site, loss_params::LossParams)
    #if nrow(site_results) == 0
    #    println(site_results)
    #end
    site_results = process_site_results(current_sim_year, site, loss_params)
    df = outerjoin(spdf, site_results, on=[:plot_id, :eco_id, :sim_year, :species_id])
    #println(df)
    by_species_id = combine(groupby(df, [:plot_id, :eco_id, :sim_year, :species_id])) do rows
        w_loss = 0.0f0
        agb_loss = 0.0f0
        for row in eachrow(rows)
            #println(row)

            w_loss += bins_loss(row.data_agbs_cdf, row.sim_agbs_cdf; loss_params=loss_params)
            agb_loss += abs(coalesce(row.data_agb_sum, 0) - coalesce(row.sim_agb_sum, 0))
        end
        (; w_loss=[w_loss], agb_loss=[agb_loss])

    end
    return by_species_id

end

@inline skipundef(xs::AbstractArray) = (xs[i] for i in eachindex(xs) if isassigned(xs, i))

function reset_pjob!(pbar::TProgress.ProgressBar, job::TProgress.ProgressJob; N::Int, desc::Union{Nothing,String}=nothing)
    pbar.paused = true

    # critical: reset i first so render never sees i > N
    job.i = 0
    job.N = N
    job.finished = false
    job.stoptime = nothing
    job.startime = Dates.now()
    desc === nothing || (job.description = desc)

    for k in eachindex(job.columns)
        c = job.columns[k]
        if c isa TProgress.CompletedColumn
            job.columns[k] = TProgress.CompletedColumn(job; style=c.style)
        elseif c isa TProgress.DownloadedColumn
            job.columns[k] = TProgress.DownloadedColumn(job)  # if you ever use this column
        end
    end

    spaces = length(job.columns) - 1
    colwidths = sum(c.measure.w for c in job.columns if !(c isa TProgress.ProgressColumn))
    bcol_width = max(1, job.width - colwidths - spaces)

    for c in job.columns
        c isa TProgress.ProgressColumn && TProgress.setwidth!(c, bcol_width)

    end

    pbar.paused = false
    return job
end
#@inline function reset_pjob!(pjob::TProgress.ProgressJob;N::Int)
#    pjob.N = N
#    pjob.finished=false
#    pjob.stoptime=nothing
#    pjob.startime=TProgress.now()
#    TProgress.update!(pjob; i = 0)
#end

function inspect(x)
    println(x)
    println(typeof(x))
    x
end
function make_spdf_dict(spdf::DataFrame)::Dict{UIntType,Dict{Int,SPDFGroundTruth}}
    n_species = length(unique(spdf.species_id))
    spdf_plts = Dict( #{Int, Dict{Int,DataFrame}}()
        plt_key.plot_id => Dict( #( {Int, DataFrame}(
            year_key.sim_year => SPDFGroundTruth(keys=begin
                    a = falses(n_species)
                    a[unique(year_df.species_id)] .= true
                    a
                end,
                records=Dict(
                    sp_key.species_id => begin
                        @assert nrow(sp_df) == 1 "more than 1 sp $(sp_df)"
                        rec = first(sp_df)
                        SPDFRecord(sp_agb_sum=rec.data_agb_sum, sp_age_cdf=rec.data_agbs_cdf)
                    end

                    for (sp_key, sp_df) in pairs(groupby(year_df, :species_id, sort=false))
                )
            )
            for (year_key, year_df) in pairs(groupby(plt_df, [:sim_year], sort=false))
        )
        for (plt_key, plt_df) in pairs(groupby(spdf, [:plot_id], sort=false))
    )

end

#actual entry
function main(args)
    #greet()
    RNG = Random.Xoshiro(1337)
    age_bins = loss_params = LossParams(
        age_bins=AgeBins(
            bins_idx=[5, 8, 13, 20, 25, 40, 60, 80] .|> Int,
            last_bin_open=true
        ),
        smoothing_weights=get_smoothing_window(; smoothing_window=1, smoothing_variance=1.0f0)
    )
    #return
    println("###loading data")
    splots, n_plots, n_species, n_ecoregions = load_cohorts()
    println("Plots:$n_plots, Ecos:$n_ecoregions, Species:$n_species, Measurements: $(size(splots))")
    mark_estab_year!(splots)
    max_age = maximum(splots.age_calc)
    #precomupte loss for missing entries
    spdf = smoothen_ref_years(splots, loss_params, max_age)
    #show(spdf)
    spdf_plts = make_spdf_dict(spdf)
    site_sim_years = get_site_sim_years(spdf)
    #return
    #show(site_sim_years)

    spinup_cohorts = get_spinup_cohorts(splots)
    initial_cohorts = get_initial_cohorts(splots)
    println("beginning trials")
    SITES_PER_RUN = Int(round(nrow(site_sim_years) * 0.33))
    #Profile.clear()
    #Profile.init(n=10^7, delay=0.001)
    TRIALS = 100
    first_run = true
    pbar = TProgress.ProgressBar(; expand=true)
    trials_pbar = TProgress.addjob!(pbar; N=TRIALS, description="Trials")
    run_pbar = TProgress.addjob!(pbar; N=1, description="Years")
    TProgress.with(pbar) do
        for trial in 1:TRIALS #ProgressBar(1:100)
            params = generate_biomass_params(RNG, UInt(n_species), UInt(n_ecoregions))
            #println(params)
            #println("###making sites")
            sites = make_sites(splots, RNG)
            chosen_sites = StatsBase.sample(RNG, 1:length(sites), SITES_PER_RUN, replace=false, ordered=true)
            max_sim_year = site_sim_years.sim_years[chosen_sites] .|> maximum |> maximum
            #println("###Sites made, beginning spinup")
            spinup_cohorts!(spinup_cohorts, sites, params) #[splots.measdate .== splots.start_measdate,:])
            #println("Sites spun up")

            #
            ## 
            ##
            ##
            ##
            ##
            #



            reset_pjob!(pbar, run_pbar; N=max_sim_year + 1)

            years_results = [DataFrame() for _ in 0:max_sim_year] #Vector{MDataFrame}(missing,max_sim_year+1)
            for current_sim_year in 0:max_sim_year #ProgressBar(0:max_sim_year) #ProgressBar(0:50)
                sites_results = [DataFrame() for _ in 1:length(chosen_sites)] #Vector{MDataFrame}(missing, length(chosen_sites))
                #any_site_results = falses(Threads.nthreads())
                Threads.@threads for i in eachindex(chosen_sites) #
                    @inbounds mapcode = chosen_sites[i]
                    @inbounds site = sites[mapcode]
                    @inbounds spdf_plt = spdf_plts[site.ref_cn]
                    @inbounds sim_years = site_sim_years.sim_years[mapcode]
                    if site.active
                        #println(site.mapcode)
                        succession_step!(current_sim_year, params, site)
                        reproduction_step!(current_sim_year, params, site)
                        # what years to check for this site
                        if current_sim_year in sim_years
                            sloss = calculate_site_loss2(current_sim_year, site, spdf_plt[current_sim_year] , loss_params)

                            #if first_run
                            #    first_run = false
                            #else
                            #    begin
                            #        sloss = calculate_site_loss(current_sim_year, spdf_plt[current_sim_year], site, loss_params)
                            #    end
                            #end
                            @inbounds sites_results[i] = sloss
                        end
                    end
                    @assert site.old <= site.live <= site.cap "$site"
                end
                #current_year_results = DataFrame()
                #if any(any_site_results)
                #current_year_results=reduce(vcat, collect(skipundef(sites_results)))
                current_year_results = reduce(vcat, sites_results)
                years_results[current_sim_year+1] = current_year_results
                #end

                TProgress.update!(run_pbar)

            end
            run_result = reduce(vcat, years_results)
            #show(run_result)
            TProgress.update!(trials_pbar)
            #ProfileSVG.save("profile_$(trial).svg")
        end
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
