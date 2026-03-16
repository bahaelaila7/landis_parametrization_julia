module BiomassSuccession


include("biomass_succession.jl")
import .SuccessionModule: Site, BiomassSuccessionParams, succession_step!, reproduction_step!, calculate_initial_biomass, add_new_cohort!, compact_site!, FloatType, UIntType
import CSV, Random, Dates, Distributions as Dists, ImageFiltering, StatsBase, Term.Progress as TProgress
using DataFrames
import SQLite
import Rasters, ArchGDAL, CairoMakie, GeoMakie
import Parquet2
import Setfield

const AG = ArchGDAL
BandType = Union{String,Real}
ValType = Union{String,Real}
AttrDict = Dict{String,BandType}
AttrTable = Dict{BandType,AttrDict}
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
    lambda::FloatType = FloatType(1.0f-2)
    EPS::FloatType = FloatType(1.0f-7)
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
    sp_agb_loss::Vector{FloatType}
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
        mapcode=UIntType(row.plot_id), cap=UIntType(2),
        ref_cn=UIntType(row.plot_id),
        old=zero(UIntType),
        live=zero(UIntType),
        B=zero(FloatType),
        AGNPP=zero(FloatType),
        capacityReduction=one(FloatType),
        growthReduction=one(FloatType),
        prevYearMortality=zero(FloatType),
        shade_class=one(UIntType), c_species=zeros(UIntType, 2),
        c_age=zeros(FloatType, 2),
        c_bio=zeros(FloatType, 2),
        c_m_tot=zeros(FloatType, 2),
        c_comp=zeros(FloatType, 2),
        sp_mature=falses(n_species),
    )
             for (i, row) in enumerate(eachrow(plot_eco_ids))]
    return sites
end

@inline function get_smoothing_window(; smoothing_window::Int=Int(3), smoothing_variance::FloatType=FloatType(1.0f0))
    w = ((-smoothing_window:smoothing_window) ./ smoothing_variance) .^ FloatType(2.0f0) .* FloatType(-0.5f0) .|> exp
    w ./= sum(w)
    return w
end

@inline function smoothen_bin_cdf(p; w::Vector{FloatType}, age_bins::AgeBins)::Vector{FloatType}
    pc = smooth_ages(; ages=p, smoothing_window=w)
    @assert !any(isnan.(pc)) "smooth NaN"
    pc_bin = bin_ages(pc; age_bins=age_bins.bins_idx, last_bin_open=age_bins.last_bin_open)
    pc_bin_cdf = cumsum(pc_bin)
    @assert !any(isnan.(pc_bin_cdf)) "cumsum NaN"
    if pc_bin_cdf[end] > zero(FloatType)
        pc_bin_cdf ./= pc_bin_cdf[end]
    end
    return pc_bin_cdf

end

@inline function bins_loss2(pc_bin_cdf::Vector{FloatType}, qc_bin_cdf::Vector{FloatType}; loss_params::LossParams)::FloatType
    # warning: 
    # Technically W1 is not defined if one or both distribution collapsed (0 everywhere)
    # 0 distance if both are collapsed while +Inf if only one is sensible,
    # BUT Inf will make all aggregates useless and will make the search directionless
    # tagging along a small log(m+eps) such that if aggregate is 0 eps penalize it heavily
    wasser1 = zero(FloatType)
    a, b = zero(FloatType), zero(FloatType)
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

@inline function bins_loss(pc_bin_cdf::Union{Missing,Vector{FloatType}}, qc_bin_cdf::Union{Missing,Vector{FloatType}}; loss_params::LossParams)::FloatType
    # warning: 
    # Technically W1 is not defined if one or both distribution collapsed (0 everywhere)
    # 0 distance if both are collapsed while +Inf if only one is sensible,
    # BUT Inf will make all aggregates useless and will make the search directionless
    # tagging along a small log(m+eps) such that if aggregate is 0 eps penalize it heavily
    wasser1 = zero(FloatType)
    a, b = zero(FloatType), zero(FloatType)
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

@inline function smooth_ages(; ages::Vector{FloatType}, smoothing_window::Vector{FloatType})::Vector{FloatType}
    smoothed_ages = ImageFiltering.imfilter(ages, smoothing_window, "symmetric")
    @assert !any(isnan.(smoothed_ages)) "filter NaN"
    s = sum(smoothed_ages)
    if s > zero(FloatType)
        smoothed_ages ./= s
    end
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
            Threads.@threads for site in sites # 
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
            add_new_cohort!(site, sp, one(FloatType), initial_biomass)
        end
        #println("Adding cohort $(row.species_symbol_map) to ", row.plot_id)
    end
    #update(pbar)
    # cohorts with year_deficit = 0 will have been added but not succeeded yet


end


function generate_biomass_params(n_species::UIntType, n_ecoregions::UIntType; rng::Random.AbstractRNG)
    # TODO: species that do not show up for a specific ecoregion, make all their prob_estab = 0
    SPINUP_MORTALITY_FRACTION = [0.15f0] #rand(Dists.Uniform(0f0,0.20f0))
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

struct BiomassParam
    name
    dist
    type::Type
    species::Bool
    ecoregion::Bool
end
struct BiomassParamDists
    params::Array{BiomassParam}
    weights_cumsum::Array{Float64}
    n_species::UIntType
    n_ecoregions::UIntType
    interaction_matrix::Matrix{UIntType}
end

function make_biomass_param_dists(n_species::UIntType, n_ecoregions::UIntType)
    BIOMASS_PARAM_DISTS = [
        BiomassParam(:SPINUP_MORTALITY_FRACTION, Dists.Uniform(0.0f0, 0.20f0), FloatType, false, false),
        BiomassParam(:S, Dists.truncated(Dists.Normal(0.5, 1.0), 0.01, 1.0), FloatType, true, false),
        BiomassParam(:D, Dists.truncated(Dists.Normal(15, 10), 5, 25), FloatType, true, false),
        BiomassParam(:LONGEVITY, Dists.truncated(Dists.Normal(200, 100), 100, 300), FloatType, true, false),
        BiomassParam(:SHADE_TOL, Dists.DiscreteUniform(1, 5), FloatType, true, false),
        BiomassParam(:MATURITY, Dists.DiscreteUniform(3, 40), FloatType, true, false),
        BiomassParam(:PROB_MORT_SPP, Dists.Uniform(), FloatType, true, true),
        BiomassParam(:PROB_ESTAB_SPP, Dists.Uniform(), FloatType, true, true),
        BiomassParam(:ANPP_MAX_SPP, Dists.truncated(Dists.Normal(2500, 100), 2400, 2500), FloatType, true, true),
        BiomassParam(:B_MAX_SPP, Dists.truncated(Dists.Normal(2500, 100), 2400, 2500), FloatType, true, true),
        BiomassParam(:firstMINRel, Dists.Uniform(0.0, 0.5), FloatType, false, true),
    ]
    interaction_matrix = [1 n_ecoregions; n_species n_species*n_ecoregions]
    BIOMASS_PARAM_WEIGHTS_RAW = [
        begin
            a, b = UIntType(UIntType(p.species) + 1), UIntType(UIntType(p.ecoregion) + 1)
            dims = interaction_matrix[(a, b)...]
            15.0 + FloatType(dims)
        end
        for p in BIOMASS_PARAM_DISTS]
    BIOMASS_PARAM_WEIGHTS = BIOMASS_PARAM_WEIGHTS_RAW / sum(BIOMASS_PARAM_WEIGHTS_RAW)

    return BiomassParamDists(BIOMASS_PARAM_DISTS, cumsum(BIOMASS_PARAM_WEIGHTS), n_species, n_ecoregions, interaction_matrix)

end

function mutate_biomass_params(p::BiomassSuccessionParams, param_dists::BiomassParamDists; rng::Random.AbstractRNG)
    # Only mutate for species in ecoregion, no point in mutating others
    # schema: variable, element-wise pdf, final type, species-specific, ecoregion-specific
    #
    # schemes to mutate:
    # - pick one variable, pick one element or more, 
    # - types of mutation: creep, redraw
    # EA:
    # - bundle mutation rate, mutation angle.

    # select one of the params
    s = rand(rng, Float64)
    param_idx = something(findlast(param_dists.weights_cumsum .<= s), 1)
    selected_param = param_dists.params[param_idx]
    v = getproperty(p, selected_param.name)
    dims = length(v)
    param_val = rand(rng, selected_param.dist) |> selected_param.type


    param_idx = dims == 1 ? 1 : rand(rng, 1:dims)
    nv = similar(v)
    nv .= v
    nv[param_idx...] = param_val
    np = Setfield.@set p.$(selected_param.name) = nv
    return np
end

function calculate_site_loss2(current_year::Int, site::Site, spdf_plt::SPDFGroundTruth, loss_params::LossParams)::SiteLoss
    #Sort by species
    n_species = length(site.sp_mature)
    insite = falses(n_species)
    @assert n_species == length(spdf_plt.keys) "species numbers do not match"
    sp_w_loss = zeros(FloatType, n_species)
    sp_agb_loss = zeros(FloatType, n_species)
    site_agb_loss = zero(FloatType)

    if site.live > 0

        c_species = @view site.c_species[1:site.live]
        max_age = UIntType(ceil(maximum(@view site.c_age[1:site.live])))
        #get indices of species sorted
        p = sortperm(c_species)

        ages = Vector{FloatType}(undef, max_age)
        sp_start_idx = 1
        last_sp = site.c_species[p[sp_start_idx]]


        #initialize losses 
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
                sim_agb_sum = sum(@view site.c_bio[p[sp_start_idx:sp_end_idx]])
                log_diff = log10(sim_agb_sum + loss_params.EPS)
                sp_agb_loss[sp] = sim_agb_sum
                site_agb_loss += sim_agb_sum
                if spdf_plt.keys[sp]
                    rec = @inbounds spdf_plt.records[sp]
                    ages .= zero(FloatType)
                    for a in @view p[sp_start_idx:sp_end_idx]
                        ages[UIntType(site.c_age[a])] = site.c_bio[a]
                    end
                    sim_age_cdf = smoothen_bin_cdf(ages; w=loss_params.smoothing_weights, age_bins=loss_params.age_bins)
                    @assert !any(isnan.(sim_age_cdf)) "cdf NaN"
                    @assert length(sim_age_cdf) == length(rec.sp_age_cdf) "cdf bins are not the same size"
                    sp_w_loss[sp] = sum(loss_params.age_bins.bin_widths .* abs.(sim_age_cdf - rec.sp_age_cdf)[begin:end-1])
                    @assert !any(isnan.(sp_w_loss[sp])) "NaN"
                    log_diff -= log10(rec.sp_agb_sum + loss_params.EPS)
                    sp_agb_loss[sp] = abs(sim_agb_sum - rec.sp_agb_sum)
                    site_agb_loss -= rec.sp_agb_sum
                end
                sp_w_loss[sp] += loss_params.lambda * abs(log_diff)

                last_sp = sp
                sp_start_idx = i
            end
        end
    end

    for sp in (1:length(spdf_plt.keys))[spdf_plt.keys.&(.!insite)]
        @inbounds rec = spdf_plt.records[UIntType(sp)]
        sp_agb_loss[sp] = rec.sp_agb_sum
        sp_w_loss[sp] += loss_params.lambda * abs(log10(rec.sp_agb_sum + loss_params.EPS))
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
        return DataFrame(plot_id=UIntType, eco_id=UIntType, sim_year=Int, species_id=UIntType, agb_total=FloatType, sim_agb_sum=FloatType, sim_agbs_cdf=FloatType[])
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
        w_loss = zero(FloatType)
        agb_loss = zero(FloatType)
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

@inline function get_total_loss(loss::SiteLoss, alpha::FloatType=FloatType(1.0f0), beta::FloatType=FloatType(1.0f0))::FloatType

    w = (alpha * sum(loss.sp_w_loss))
    sp = (beta * sum(loss.sp_agb_loss))
    site = (loss.site_agb_loss)
    all = w + sp + site
    return all

end

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
@inline function Base.:+(loss1::SiteLoss, loss2::SiteLoss)
    SiteLoss(sp_w_loss=loss1.sp_w_loss .+ loss2.sp_w_loss,
        sp_agb_loss=loss1.sp_agb_loss .+ loss2.sp_agb_loss,
        site_agb_loss=loss1.site_agb_loss .+ loss2.site_agb_loss)
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
                        nrow(sp_df) > 1 && @warn "more than 1 sp $(sp_df)"
                        rec = last(sp_df)
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

function export_sites!(db::SQLite.DB, current_year::Int, sites)

    sqls = Array{Union{Missing,SQLite.DB}}(missing, Threads.maxthreadid())
    DBI = SQLite.DBInterface


    Threads.@threads for I in eachindex(sites)
        tid = Threads.threadid()
        @inbounds con = sqls[tid]
        if ismissing(con)
            con = SQLite.DB(":memory:")
            @inbounds sql[tid] = con
            stmt = """CREATE TABLE IF NOT EXISTS output_communities(
                                    year INTEGER,
                                    mapcode INTEGER,
                                    ecocode INTEGER,
                                    species_symbol INTEGER,
                                    AGE INTEGER,
                                    BIOMASS REAL
                                 );
                                 """
            DBI.execute(con, stmt)
        end

        @inbounds site = sites[I]
        if !ismissing(site) && site.active
            for i in 1:site.live
                stmt = """INSERT INTO output_communities(year, mapcode, ecocode, species_symbol, age, biomass)
                                VALUES (
                                $(current_year),
                                $(site.mapcode),
                                $(site.ecocode),
                                $(site.c_species[i]),
                                $(Int(site.c_age[i])),
                                $(site.c_bio[i])
                                );"""
                DBI.execute(con, stmt)
            end
        end
    end
end

function parametrize(loss_params::LossParams, splots::DataFrame, n_plots::UIntType, n_species::UIntType, n_ecoregions::UIntType; RNG::Union{Nothing,Random.AbstractRNG}, TRIALS::Int=5)::Tuple{FloatType,SiteLoss,BiomassSuccessionParams}
    if isnothing(RNG)
        RNG = Random.default_rng()
    end
    #greet()
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

    best_loss = FloatType(Inf)
    best_result = SiteLoss(FloatType[], FloatType[], FloatType(Inf))
    best_params = generate_biomass_params(RNG, n_species, n_ecoregions)
    if TRIALS < 1
        return best_loss, best_result, best_params
    end

    pbar = TProgress.ProgressBar(; expand=true)
    trials_pbar = TProgress.addjob!(pbar; N=TRIALS, description="Trials")
    run_pbar = TProgress.addjob!(pbar; N=1, description="Years")
    TProgress.with(pbar) do
        for trial in 1:TRIALS #ProgressBar(1:100)
            params = generate_biomass_params(RNG, n_species, n_ecoregions)
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

            #years_results = [DataFrame() for _ in 0:max_sim_year] #Vector{MDataFrame}(missing,max_sim_year+1)
            years_results = Vector{SiteLoss}(undef, max_sim_year + 1)
            for current_sim_year in 0:max_sim_year #ProgressBar(0:max_sim_year) #ProgressBar(0:50)
                sites_results = Vector{SiteLoss}(undef, length(chosen_sites)) #[DataFrame() for _ in 1:length(chosen_sites)] #Vector{MDataFrame}(missing, length(chosen_sites))
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
                            sloss = calculate_site_loss2(current_sim_year, site, spdf_plt[current_sim_year], loss_params)

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
                #current_year_results = reduce(vcat, skipundef(sites_results))
                year_results_no_missing = collect(skipundef(sites_results))
                if length(year_results_no_missing) > 0
                    current_year_results = sum(year_results_no_missing)
                    years_results[current_sim_year+1] = current_year_results
                end
                #end

                TProgress.update!(run_pbar)

            end
            #run_result = reduce(vcat, years_results)
            run_result = sum(skipundef(years_results))
            @assert !any(isnan.(run_result.sp_w_loss)) "run NaN"

            run_loss::FloatType = get_total_loss(run_result)
            if run_loss <= best_loss
                best_loss = run_loss
                best_result = run_result
                best_params = params
            end

            #show(run_result)
            #println("Run $(trial): $(run_result)")
            TProgress.update!(trials_pbar)
            #ProfileSVG.save("profile_$(trial).svg")
        end
    end

    #println(typeof(best_loss), best_loss)
    #println(best_loss, best_result, best_params)
    return best_loss, best_result, best_params
end

function load_treemap_raster(raster_path::String; CN_FIELD_NAME::String="PLT_CN")::Tuple{Array{Union{Missing,Int64}},AttrTable}
    #CairoMakie.activate!()
    #load raster

    #labels = AG.read(path) do ds
    #    band = AG.getband(ds, 3)
    #    AG.getcategorynames(band)   # Vector of strings (value 0 at index 1)
    #end
    #    println(labels)

    AG.readraster(raster_path) do ds
        band = AG.getband(ds, 1)
        rat = AG.getdefaultRAT(band)
        nrows = AG.nrow(rat)
        ncols = AG.ncolumn(rat)

        colnames = [AG.columnname(rat, c) for c in 0:ncols-1]

        valcol = findfirst(==("Value"), colnames) #AG.findcolumnindex(rat, AG.GFU_MinMax)
        isnothing(valcol) && error("No GFU_MinMax/'Value' column in RAT.")
        valcol -= 1 # gdal is 0 based
        bandType = AG.pixeltype(band)
        function rat_get(r::Int, c::Int)
            t = AG.columntype(rat, c)
            if t == AG.GFT_Integer
                return AG.asint(rat, r, c)
            elseif t == AG.GFT_Real
                return AG.asdouble(rat, r, c)
            else
                return AG.asstring(rat, r, c)
            end
        end
        val_att_dict = AttrTable()


        for r in 0:nrows-1
            px = bandType(rat_get(r, valcol))
            attrs = AttrDict()
            for c in 0:ncols-1
                attrs[colnames[c+1]] = rat_get(r, c)
            end
            val_att_dict[px] = attrs
        end


        A = AG.read(band)
        NO_DATA = AG.getnodatavalue(band)
        println(size(A), typeof(A))
        println(length(A))

        outA = Array{Union{Missing,Int64}}(missing, size(A))

        #AA = map(A) do cell
        #rand(RNG, UInt64)
        # skip NODATA
        begin
            Threads.@threads for i in eachindex(A, outA)
                @inbounds cell = A[i]
                if !ismissing(cell) && cell != NO_DATA
                    plt_attr = get(val_att_dict, cell, missing)
                    if !ismissing(plt_attr)
                        plt_cn = get(plt_attr, CN_FIELD_NAME, missing)
                        @inbounds outA[i] = UInt64(plt_cn)
                    end
                end
            end
        end
        return outA, val_att_dict
    end
end

function populate_initial_treemap_communities(plt_cn_raster::Array{Union{Missing,Int64}}, splots_dict::Dict{Int64,DataFrame}, n_species::UIntType; RNG::Union{Nothing,Random.AbstractRNG})::Array{Union{Missing,Site}}
    if isnothing(RNG)
        RNG = Random.default_rng()
    end

    tRNGs = [Random.Xoshiro(rand(RNG, UInt64)) for _ in 1:Threads.maxthreadid()]
    totalpixels = zeros(UInt, Threads.maxthreadid())
    totalmissing = zeros(UInt, Threads.maxthreadid())
    outA = Array{Union{Missing,Site}}(missing, size(plt_cn_raster))

    begin
        Threads.@threads for i in eachindex(plt_cn_raster, outA)
            tid = Threads.threadid()
            @inbounds plt_cn = plt_cn_raster[i]

            if !ismissing(plt_cn)
                @inbounds totalpixels[tid] += 1
                @inbounds totalmissing[tid] += 1
                initial_cohorts = get(splots_dict, plt_cn, missing)
                if !ismissing(initial_cohorts) && nrow(initial_cohorts) > 0
                    @inbounds tRNG = tRNGs[tid]
                    p = first(initial_cohorts)
                    n_cohorts = nrow(initial_cohorts)
                    cap = UIntType(2^ceil(log2(n_cohorts)))
                    site = Site(
                        active=true,
                        rng=Random.Xoshiro(rand(tRNG, UInt64)),
                        ecocode=UIntType(p.eco_id),
                        mapcode=UIntType(p.plot_id), cap=UIntType(cap),
                        ref_cn=UIntType(p.plot_id),
                        old=zero(UIntType),
                        live=zero(UIntType),
                        B=zero(FloatType),
                        AGNPP=zero(FloatType),
                        capacityReduction=one(FloatType),
                        growthReduction=one(FloatType),
                        prevYearMortality=zero(FloatType),
                        shade_class=one(UIntType),
                        c_species=zeros(UIntType, cap),
                        c_age=zeros(FloatType, cap),
                        c_bio=zeros(FloatType, cap),
                        c_m_tot=zeros(FloatType, cap),
                        c_comp=zeros(FloatType, cap),
                        sp_mature=falses(n_species),
                    )
                    for row in eachrow(initial_cohorts)
                        add_new_cohort!(site, row.species_id, FloatType(row.age_calc), FloatType(row.agb_sum))
                    end

                    @inbounds outA[i] = site
                    @inbounds totalmissing[tid] -= 1
                end

                #ismissing(plt_cn) && error("cannot get CN out of $(plt_attr)")
            end
        end
    end
    println("Missing CNs: $(sum(totalmissing)/sum(totalpixels) * 100)")
    return outA
end

struct SiteRecord
    year::UIntType
    mapcode::UIntType
    ecocode::UIntType
    species::UIntType
    age::UIntType
    biomass::FloatType
end
function run_simulation(site_raster::Array{Union{Missing,Site}}, params::BiomassSuccessionParams; years::Int=30, RNG=RNG)
    FLUSH_THRESHOLD = 1000
    writer_buffer = Vector{SiteRecord}()
    sizehint!(writer_buffer, FLUSH_THRESHOLD)
    flush_task = nothing
    chunk = 1

    thread_buffers = [Vector{SiteRecord}() for _ in 1:Threads.maxthreadid()]
    TProgress.@track for current_sim_year in 1:years
        Threads.@threads for I in eachindex(site_raster)
            @inbounds site = site_raster[I]
            if !ismissing(site)
                #println(I)
                succession_step!(current_sim_year, params, site)
                reproduction_step!(current_sim_year, params, site)
                if current_sim_year % 5 == 0
                    compact_site!(site)
                end
                if current_sim_year > 0
                    tid = Threads.threadid()
                    @inbounds buf = thread_buffers[tid]

                    for i in 1:site.live
                        push!(buf, SiteRecord(current_sim_year, site.mapcode, site.ecocode, site.c_species[i], site.c_age[i], site.c_bio[i]))
                    end
                end
            end
        end
        for buf in thread_buffers
            append!(writer_buffer, buf)
            empty!(buf)
        end
        if length(writer_buffer) > FLUSH_THRESHOLD
            payload = writer_buffer
            flush_task = @async begin
                Parquet2.writefile("chunk_$(chunk).parquet", payload)
                @info "Flushed chunk $(chunk): $(length(payload))"
                chunk += 1
            end
            writer_buffer = Vector{SiteRecord}()
            sizehint!(writer_buffer, FLUSH_THRESHOLD)
        end
    end
    for buf in thread_buffers
        append!(writer_buffer, buf)
    end
    flush_task !== nothing && wait(flush_task)
    if !isempty(writer_buffer)
        Parquet2.writefile("chunk_$(chunk).parquet", writer_buffer)
        @info "Flushed final chunk $(chunk): $(length(writer_buffer))"
    end
end
function run_simulation2(site_raster::Array{Union{Missing,Site}}, params::BiomassSuccessionParams; years::Int=30, RNG=RNG)
    FLUSH_THRESHOLD = 1000
    BUFF_LEN = 2048
    ch = Channel{Vector{SiteRecord}}(BUFF_LEN)
    writer_thread = Threads.@spawn begin
        writer_buffer = Vector{SiteRecord}()
        sizehint!(writer_buffer, FLUSH_THRESHOLD)
        chunk = 1

        for rec_buff in ch
            append!(writer_buffer, rec_buff)
            if length(writer_buffer) > FLUSH_THRESHOLD
                Parquet2.writefile("chunk_$(chunk).parquet", writer_buffer)
                @info "Flushed chunk $(chunk)"
                chunk += 1
                empty!(writer_buffer)
            end
        end

        if !isempty(writer_buffer)
            Parquet2.writefile("chunk_$(chunk).parquet", writer_buffer)
            @info "Flushed final chunk $(chunk)"
        end
    end

    thread_buffers = [Vector{SiteRecord}() for _ in 1:Threads.maxthreadid()]
    TProgress.@track for current_sim_year in 1:years
        Threads.@threads for I in eachindex(site_raster)
            @inbounds site = site_raster[I]
            if !ismissing(site)
                #println(I)
                succession_step!(current_sim_year, params, site)
                reproduction_step!(current_sim_year, params, site)
                if current_sim_year % 5 == 0
                    compact_site!(site)
                end
                if current_sim_year > 0
                    tid = Threads.threadid()
                    @inbounds buf = thread_buffers[tid]

                    for i in 1:site.live
                        push!(buf, SiteRecord(current_sim_year, site.mapcode, site.ecocode, site.c_species[i], site.c_age[i], site.c_bio[i]))
                    end
                    if length(buf) >= FLUSH_THRESHOLD / 10

                        put!(ch, buf)
                        @inbounds thread_buffers[tid] = Vector{SiteRecord}()
                    end
                end
            end
        end
    end
    thread_buffers .|> x -> put!(ch, x)
    close(ch)
    wait(writer_thread)
end





#outR = Rasters.Raster(outA, Rasters.dims(r); name=Rasters.name(r), metadata=Rasters.metadata(r))
#return outR


#fig = GeoMakie.Figure()
#ga = GeoMakie.GeoAxis(fig[1, 1], aspect = GeoMakie.DataAspect()) # Create a geographic axis
#GeoMakie.heatmap!(ga, r)
#Rasters.plot(fig)




#load CN
#match with available CN
#grow for 30 years
#save to raster agb

#actual entry
function main(args)
    RNG = Random.Xoshiro(1337)
    all = true
    best_params = nothing
    splots, n_plots, n_species, n_ecoregions = DataFrame(), UIntType(400), UIntType(2), UIntType(1)
    BIOMASS_PARAM_DISTS = make_biomass_param_dists(n_species, n_ecoregions)
    println(BIOMASS_PARAM_DISTS)
    pp = generate_biomass_params(n_species, n_ecoregions; rng=RNG)
    println(pp)
    ppp = pp
    for _ in 1:400
        ppp = mutate_biomass_params(ppp, BIOMASS_PARAM_DISTS; rng=RNG)
    end
    println(pp)
    println(ppp)
    return
    if all
        loss_params = LossParams(
            age_bins=AgeBins(
                bins_idx=[5, 8, 13, 20, 25, 40, 60, 80] .|> Int,
                last_bin_open=true
            ),
            smoothing_weights=get_smoothing_window(; smoothing_window=1, smoothing_variance=FloatType(1.0f0))
        )
        #return
        println("###loading data")
        splots, n_plots, n_species, n_ecoregions = load_cohorts()
        println("Plots:$n_plots, Ecos:$n_ecoregions, Species:$n_species, Measurements: $(size(splots))")
        BIOMASS_PARAM_DISTS = make_biomass_param_dists(n_species, n_ecoregions)
        @time best_loss, best_result, best_params = parametrize(loss_params, splots, n_plots, n_species, n_ecoregions; RNG=RNG, TRIALS=2)
        println("Best Loss: $(best_loss)")
    end
    raster_path = "/home/bahaa/Downloads/FL_extents/FL5_extent_shapefile/FL_Baker22.tif"

    splots_dict = Dict(
        plt_key.plt_cn => DataFrame(plt_df)
        for (plt_key, plt_df) in pairs(groupby(splots, :plt_cn, sort=false))
    )
    println("Loading Raster")
    @time cn_raster, vat = load_treemap_raster(raster_path)
    println("Populating Raster")
    @time site_raster = populate_initial_treemap_communities(cn_raster, splots_dict, n_species; RNG=RNG)
    println("Running simulation")
    @time run_simulation(site_raster, best_params; RNG=RNG)


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
