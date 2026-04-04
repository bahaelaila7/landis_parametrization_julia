module BiomassSuccession


include("biomass_succession.jl")
include("SA.jl")
import .SuccessionModule: Site, BiomassSuccessionParams, BiomassSuccessionEcoParams, succession_step!, reproduction_step!, calculate_initial_biomass, add_new_cohort!, compact_site!, FloatType, UIntType
import .SA: SACandidate, SAState, simulated_annealing_acceptance_rule, threshold_accepting_acceptance_rule, search_cmp!, search_update_rule!
import CSV, Random, Dates, Distributions as Dists, ImageFiltering, StatsBase, Term.Progress as TProgress
using DataFrames
import SQLite
import Rasters, ArchGDAL, CairoMakie, GeoMakie
import Parquet2
import Setfield
import JLD2
import StructTypes
import JSON3
import Glob


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
    num_sites::Int = 1
end
@inline function get_total_loss(loss::SiteLoss, alpha::FloatType=FloatType(1.0f0), beta::FloatType=FloatType(1.0f0))::FloatType

    w = (alpha * sum(loss.sp_w_loss))
    sp = (beta * sum(loss.sp_agb_loss))
    site = (loss.site_agb_loss)
    all = w + sp + site
    return all / loss.num_sites

end
@inline Base.convert(::Type{Float64}, a::SiteLoss) = Float64(get_total_loss(a))
#@inline Base.promote_rule(::Type{SiteLoss}, ::Type{Float64}) = Float64

#StructTypes.StructType(::Type{SAState}) = StructTypes.Struct()
#StructTypes.StructType(::Type{SACandidate}) = StructTypes.Struct()
#StructTypes.StructType(::Type{SiteLoss}) = StructTypes.Struct()
#StructTypes.StructType(::Type{Random.Xoshiro}) = StructTypes.Struct()
#
function save_json(path::String, s)
    open(path, "w") do io
        JSON3.write(io, s)
    end
end
#function load_succession_params(path::String)::BiomassSuccessionParams
#    data = open(path, "r") do io
#        read(io, String)
#    end
#    JSON3.read(data, BiomassSuccessionParams)
#end
#function load_search_state(path::String)::SAState
#    data = open(path, "r") do io
#        read(io, String)
#    end
#    JSON3.read(data, SAState; allow_inf=true)
#end
function process_cohorts_csv(csv_path::String="../data_eco_cohorts.csv", output_db::String="../data_eco_cohorts.db"; filter_ecos::Array{String}=String[])
    all_df = CSV.read(csv_path, DataFrame)
    cdf = all_df
    if length(filter_ecos) > 0
        filtered_plots = in(filter_ecos).(all_df.eco)
        cdf = all_df[filtered_plots, :]
    end

    plots = combine(groupby(cdf, [:plt_cn, :statecd, :unitcd, :countycd, :plot, :eco, :measdate, :species_symbol, :age_calc], sort=false), nrow => :count, :agb => sum => :agb_sum)
    # start_measdate =
    start_measdates = combine(groupby(plots, [:statecd, :unitcd, :countycd, :plot], sort=false)) do rows
        (; start_measdate=[minimum(rows.measdate)])
    end
    plots_measdate = innerjoin(plots, start_measdates, on=[:statecd, :unitcd, :countycd, :plot])
    splots = sort!(plots_measdate, [:measdate, :statecd, :unitcd, :countycd, :plot, :age_calc, :species_symbol])

    splots.measdate = string.(splots.measdate)
    splots.start_measdate = string.(splots.start_measdate)
    splots.plot_id .= groupindices(groupby(splots, [:statecd, :unitcd, :countycd, :plot])) .|> Int
    splots.eco_id .= groupindices(groupby(splots, [:eco])) .|> Int
    splots.species_id .= groupindices(groupby(splots, [:species_symbol])) .|> Int

    db = SQLite.DB(output_db)
    SQLite.drop!(db, "splots", ifexists=true)
    splots |> SQLite.load!(db, "splots")
    SQLite.close(db)


    #n_species = maximum(splots.species_id)
    #n_plots = maximum(splots.plot_id)
    #n_ecoregions = maximum(splots.eco_id)
    #println(splots.species_id)
    #return splots, n_plots, n_species, n_ecoregions

    #println(sort!(plots, :count, rev=true))
    #starting_plots = splots[splots.measdate .== splots.start_measdate, :]
    #println(starting_plots)
end

function load_cohorts_sqlite(db_path::String="../data_eco_cohorts.db", tablename::String="data_eco_cohorts"; filter_ecos::Array{String}=String[])::DataFrame
    db = SQLite.DB(db_path)
    sql = "SELECT * FROM $(tablename)"
    if length(filter_ecos) > 0
        sql *= " WHERE eco in ('$(join(filter_ecos,"','"))')"
    end
    println(sql)
    df = SQLite.DBInterface.execute(db, sql) |> DataFrame
    SQLite.close(db)
    return df

end
function load_cohorts_csv(csv_path::String="../data_eco_cohorts_cn.csv"; filter_ecos::Array{String}=String[])
    all_df = CSV.read(csv_path, DataFrame)
    cdf = all_df
    if length(filter_ecos) > 0
        filtered_plots = in(filter_ecos).(all_df.eco)
        cdf = all_df[filtered_plots, :]
    end

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

function make_sites(splots::DataFrame, eco_species_ids::Array{Array{Int}}; rng::Random.AbstractRNG)
    #eco_ids = unique(select(splots, [:eco_id]))
    #n_species = maximum(splots.species_id)
    #plot_ids = unique(select(splots, [:plot_id]))
    #plot_eco_ids = unique(select(splots, [:plot_id, :eco_id]))
    #plot_eco_ids = sort!(plot_eco_ids, [:plot_id])
    plot_eco_ids = unique(select(splots, [:plot_id, :eco_id]))
    #plot_eco_ids = sort!(plot_eco_ids, [:plot_id])
    #@assert (nrow(plot_ids) == nrow(plot_eco_ids)) "Error: plot_ids and plot_eco_ids are not of equal length!"

    #index = eco_id * plot_id

    #splots_dict::Dict{Int64,DataFrame} = Dict(
    #    plt_key.plt_cn => DataFrame(plt_df)
    #    for (plt_key, plt_df) in pairs(groupby(splots, :plt_cn, sort=false))
    #)

    sites = [Site(
        active=false,
        rng=Random.Xoshiro(rand(rng, UInt64)),
        ecocode=UIntType(row.eco_id), # no ecocode coming from raster, relying on eco_id
        eco_id=row.eco_id,
        mapcode=UIntType(row.plot_id), # for parametrization, plot_id is global index, no raster
        cap=UIntType(2),
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
        c_comp=zeros(FloatType, 2), sp_mature=falses(length(eco_species_ids[row.eco_id])),
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
    spdf = combine(groupby(df, [:plot_id, :eco_id, :measdate, :start_measdate, :eco_species_id])) do rows
        ages = zeros(FloatType, max_age)
        for row in eachrow(rows)
            ages[row.age_calc] += row.agb_sum
        end
        row = rows[1, :]
        sim_year = Dates.value.(Dates.Day.(row.measdate - row.start_measdate)) ./ 365.25 .|> round .|> Int
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
    #println(minimum(df.age_calc))
    #show(df[df.age_calc .< 0,:]) #.age_calc .= 1
    year_estab = df.measdate .- (df.age_calc .|> Dates.Year)
    oldest = minimum(year_estab)
    last = maximum(df.start_measdate)
    println("Oldest cohort established: $oldest")
    println("First measurement date: $last")
    df.year_deficit .= Dates.value.(Dates.Day.(year_estab .- last)) ./ 365.25 .|> round .|> Int
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

function generate_eco_params1(params::BiomassSuccessionParams)::Array{BiomassSuccessionEcoParams}
    #eco_id-> species_ids
    eco_params = [
        BiomassSuccessionEcoParams(
            SPINUP_MORTALITY_FRACTION=params.SPINUP_MORTALITY_FRACTION,
            D=(@dimslice params.D species),
            S=(@dimslice params.S species),
            LONGEVITY=(@dimslice params.LONGEVITY species),
            MATURITY=(@dimslice params.MATURITY species),
            SHADE_TOL=(@dimslice params.SHADE_TOL species), ANPP_MAX_SPP=(@dimslice params.ANPP_MAX_SPP eco_id species),
            B_MAX_SPP=(@dimslice params.B_MAX_SPP eco_id species),
            B_MAX_ECO=max((@dimslice params.B_MAX_SPP eco_id species)),
            PROB_MORT_SPP=(@dimslice params.PROB_MORT_SPP eco_id species),
            PROB_ESTAB_SPP=(@dimslice params.PROB_ESTAB_SPP eco_id species),
            SUFFICIENT_LIGHT=params.SUFFICIENT_LIGHT,
            MIN_REL_BIOMASS=(@dimslice params.MIN_REL_BIOMASS eco_id),
        )
        for (eco_id, species) in enumerate(params.ECO_SPECIES_IDS)]
    return eco_params

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

function spinup_cohorts!(spinup_cohorts::DataFrame, sites::Vector{Site}, eco_params::Array{BiomassSuccessionEcoParams})
    #show(spinup_cohorts.year_deficit)
    current_year = minimum(spinup_cohorts.year_deficit)
    ## current_year will go down to -1, since the last estab cohort
    ## would be 1 year old, so a year before the last start_measdate
    #pbar = ProgressBar(total = -current_year)
    for row in eachrow(spinup_cohorts)
        #println(row)
        while current_year < row.year_deficit
            #grow all active
            Threads.@threads :static for site in sites #
                if site.active
                    #println(site.mapcode)
                    succession_step!(current_year, eco_params, site)
                    reproduction_step!(current_year, eco_params, site)
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
        sp = row.eco_species_id
        if site.old < site.live  # there are young cohorts
            for idx in (site.old+1):site.live
                if site.c_species[idx] == sp
                    add_new_cohort = false
                end
            end
        end
        if add_new_cohort
            params = eco_params[site.eco_id]
            try
                initial_biomass = calculate_initial_biomass(params.B_MAX_SPP[sp], site.B, params.B_MAX_ECO)
                add_new_cohort!(site, UIntType(sp), one(FloatType), initial_biomass)
            catch e
                println(row)
                println(site)
                println(params)
                rethrow(e)
            end
        end
        #println("Adding cohort $(row.species_symbol_map) to ", row.plot_id)
    end
    #update(pbar)
    # cohorts with year_deficit = 0 will have been added but not succeeded yet


end

#TODO: use BiomassParamDists
function generate_biomass_params(species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Array{Array{Int}}; rng::Random.AbstractRNG)
    n_species = length(species_list) |> UIntType
    n_ecoregions = length(eco_list) |> UIntType
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

struct BiomassParam
    name
    dist
    type::Type
    species_specific::Bool
    ecoregion_specific::Bool
end
struct BiomassParamDists
    params::Array{BiomassParam}
    weights_cumsum::Array{Float64}
    n_species::UIntType
    n_ecoregions::UIntType
    #interaction_matrix::Matrix{UIntType}
end

function make_biomass_param_dists(n_species::Int, n_ecoregions::Int, eco_species_ids::Array{Array{Int}})
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
    #interaction_matrix = [1 n_ecoregions; n_species n_species*n_ecoregions]
    n_species_ecoregions = sum(length(species) for species in eco_species_ids)
    BIOMASS_PARAM_WEIGHTS_RAW = [
        Float64(15 + begin
            if p.species_specific && p.ecoregion_specific
                n_species_ecoregions
            elseif p.species_specific
                n_species
            elseif p.ecoregion_specific
                n_ecoregions
            else
                1
            end
        end)
        for p in BIOMASS_PARAM_DISTS]
    BIOMASS_PARAM_WEIGHTS = BIOMASS_PARAM_WEIGHTS_RAW / sum(BIOMASS_PARAM_WEIGHTS_RAW)

    return BiomassParamDists(BIOMASS_PARAM_DISTS, cumsum(BIOMASS_PARAM_WEIGHTS), n_species, n_ecoregions)#, interaction_matrix)

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
    #TODO: redo cumsum to match eco -> species
    param_idx = something(findlast(param_dists.weights_cumsum .<= s), 1)
    selected_param = param_dists.params[param_idx]
    param_val = rand(rng, selected_param.dist) |> selected_param.type
    np = if selected_param.name == :firstMINRel
        eco_id = rand(rng, 1:length(p.ECO_SPECIES_IDS))
        nv = deepcopy(p.MIN_REL_BIOMASS)
        nv[i] = [param_val + k * 0.10f0 for k in 0:4]
        Setfield.@set p.MIN_REL_BIOMASS = nv
    else
        v = getproperty(p, selected_param.name)
        nv = deepcopy(v)
        if selected_param.ecoregion_specific
            eco_id = rand(rng, 1:length(p.ECO_SPECIES_IDS))
            if selected_param.species_specific
                species_id = rand(rng, 1:length(p.ECO_SPECIES_IDS[eco_id]))
                nv[eco_id][species_id] = param_val
            else
                nv[eco_id] = param_val
            end
        elseif selected_param.species_specific
            species_id = rand(rng, 1:length(p.SPECIES_LIST))
            nv[species_id] = param_val
        else
            nv[] = param_val
        end

        Setfield.@set p.$(selected_param.name) = nv
    end
    return np
end

@inline function calculate_species_loss!(; sp, gsp, site, ages, p, sp_start_idx, sp_end_idx, spdf_plt, loss_params, sp_w_loss, sp_agb_loss, site_agb_loss)
    sim_agb_sum = sum(@view site.c_bio[p[sp_start_idx:sp_end_idx]])
    log_diff = log10(sim_agb_sum + loss_params.EPS)
    sp_agb_loss[gsp] = sim_agb_sum
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
        sp_w_loss[gsp] = sum(loss_params.age_bins.bin_widths .* abs.(sim_age_cdf - rec.sp_age_cdf)[begin:end-1])
        @assert !any(isnan.(sp_w_loss[gsp])) "NaN"
        log_diff -= log10(rec.sp_agb_sum + loss_params.EPS)
        sp_agb_loss[gsp] = abs(sim_agb_sum - rec.sp_agb_sum)
        site_agb_loss -= rec.sp_agb_sum
    end
    sp_w_loss[gsp] = (1.0f0 + sp_w_loss[gsp]) * (abs(log_diff)^2)
    return site_agb_loss
end

function calculate_site_loss2(current_year::Int, site::Site, n_species::Int, eco_species_ids::Array{Array{Int}}, spdf_plt::SPDFGroundTruth, loss_params::LossParams; debug=false)::SiteLoss
    #Sort by species
    eco_n_species = length(site.sp_mature)
    species_id_map = eco_species_ids[site.eco_id]
    @assert eco_n_species == length(spdf_plt.keys) "eco species numbers do not match"
    @assert length(species_id_map) == eco_n_species "eco species numbers do not match"

    insite = falses(eco_n_species)

    #global loss
    sp_w_loss = zeros(FloatType, n_species)
    sp_agb_loss = zeros(FloatType, n_species)
    site_agb_loss = zero(FloatType)

    if site.live > 0

        c_species = @view site.c_species[1:site.live]
        max_age = UIntType(ceil(maximum(@view site.c_age[1:site.live])))
        #get indices of species sorted
        # traversing c_species[p[1..end]] is equivalent to traversing sorted_c_species[1..end]
        # but now useful so that I don't need to sort c_age, c_bio
        p = sortperm(c_species)
        if debug
            println(c_species)
            println(p)
        end

        ages = Vector{FloatType}(undef, max_age)
        sp_start_idx = 1
        prev_sp = site.c_species[p[sp_start_idx]]


        #initialize losses
        # process sp's in site, sorted by species
        for i in eachindex(p)

            # get the
            sp = @inbounds c_species[p[i]]
            insite[sp] = true
            # keep looping until you find the end of sp segment
            # then use p[sp_start_index:sp_end_index] to gather from c_bio, c_age
            # conclude sp
            if sp != prev_sp
                if debug
                    println("concluding_species: $(sp)")
                end
                sp_end_idx = i - 1
                site_agb_loss = calculate_species_loss!(; sp=sp, gsp=species_id_map[sp],
                    site=site, ages=ages, p=p, sp_start_idx=sp_start_idx, sp_end_idx=sp_end_idx,
                    spdf_plt=spdf_plt, loss_params=loss_params, sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=site_agb_loss)
                prev_sp = sp
                sp_start_idx = i
            end
            if debug
                println("processing_species: $(sp)")
            end
            if i == length(p)
                if debug
                    println("concluding_species: $(sp)")
                end
                sp_end_idx = i
                site_agb_loss = calculate_species_loss!(; sp=sp, gsp=species_id_map[sp],
                    site=site, ages=ages, p=p, sp_start_idx=sp_start_idx, sp_end_idx=sp_end_idx,
                    spdf_plt=spdf_plt, loss_params=loss_params, sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=site_agb_loss)
            end
        end
    end

    for sp in (1:length(spdf_plt.keys))[spdf_plt.keys.&(.!insite)]
        @inbounds rec = spdf_plt.records[UIntType(sp)]
        @inbounds gsp = species_id_map[sp]
        sp_agb_loss[gsp] = rec.sp_agb_sum
        sp_w_loss[gsp] += loss_params.lambda * abs(log10(rec.sp_agb_sum + loss_params.EPS))
        site_agb_loss -= rec.sp_agb_sum
    end


    return SiteLoss(sp_w_loss=sp_w_loss, sp_agb_loss=sp_agb_loss, site_agb_loss=abs(site_agb_loss), num_sites=1)

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
    site_df.eco_id .= site.eco_id
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
        site_agb_loss=loss1.site_agb_loss + loss2.site_agb_loss,
        num_sites=loss1.num_sites + loss2.num_sites)
end
function make_spdf_dict(spdf::DataFrame, eco_species_ids::Array{Array{Int}})::Dict{UIntType,Dict{Int,SPDFGroundTruth}}
    #n_species = length(unique(spdf.species_id))
    return Dict( #{Int, Dict{Int,DataFrame}}()
        plt_key.plot_id => Dict( #( {Int, DataFrame}(
            year_key.sim_year => SPDFGroundTruth(keys=begin
                    n_species = length(eco_species_ids[first(year_df).eco_id])
                    #println("----")
                    #println(eco_species_ids)
                    #println(n_species)
                    #println(year_df)
                    #println(unique(year_df.eco_species_id))
                    #println("----")
                    a = falses(n_species)
                    a[unique(year_df.eco_species_id)] .= true
                    a
                end,
                records=Dict(
                    sp_key.eco_species_id => begin
                        nrow(sp_df) > 1 && @warn "more than 1 sp $(sp_df)"
                        rec = last(sp_df)
                        SPDFRecord(sp_agb_sum=rec.data_agb_sum, sp_age_cdf=rec.data_agbs_cdf)
                    end

                    for (sp_key, sp_df) in pairs(groupby(year_df, :eco_species_id, sort=false))
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

function parametrize(; cohorts_db_path::String, tablename::String, output_dir::String, loss_params::LossParams, skip_disturbances=true, filter_ecos::Array{String}=String[], RNG::Union{Nothing,Random.AbstractRNG}, TRIALS::Int=5)::SAState#Tuple{FloatType,SiteLoss,BiomassSuccessionParams}
    if isnothing(RNG)
        RNG = Random.default_rng()
    end
    #cohorts_df = load_cohorts_sqlite(db_path, tablename; filter_ecos=filter_ecos)
    println("Connecting to: $(cohorts_db_path) ")
    db = SQLite.DB(cohorts_db_path)
    println("Creating index if necessary")
    SQLite.execute(db, "CREATE INDEX IF NOT EXISTS PLT_ECO_IDX ON data_eco_cohorts(ECO);")
    sql = "SELECT * FROM $(tablename) WHERE true"
    if length(filter_ecos) > 0
        sql *= " AND eco in ('$(join(filter_ecos,"','"))')"
    end
    if skip_disturbances
        sql *= " AND subp_has_dstrb='f'"
    end
    println(sql)
    cohorts_df = SQLite.DBInterface.execute(db, sql) |> DataFrame
    SQLite.close(db)
    println("Closing db. $(nrow(cohorts_df)) rows loaded.")

    @time splots, eco_list, species_list, eco_species_ids = make_splots(cohorts_df)
    n_species = length(species_list)
    n_ecoregions = length(eco_list)
    n_plots = maximum(splots.plot_id)
    #println(eco_list)
    #println(species_list)
    #println(eco_species_ids)
    #println(splots)
    #return
    println("Plots:$n_plots, Ecos:$n_ecoregions, Species:$n_species, Measurements: $(size(splots))")
    println("Initiating param distributions")
    @time params_dist = make_biomass_param_dists(n_species, n_ecoregions, eco_species_ids)
    #greet()
    #println(splots)
    mark_estab_year!(splots)
    #println(splots)

    max_age = maximum(splots.age_calc)
    #precomupte loss for missing entries
    println("Preprocessing plot results (smoothing and binning)")
    @time spdf = smoothen_ref_years(splots, loss_params, max_age)
    #println(spdf)
    #show(spdf)
    println("Creating comparison years")
    @time spdf_plts = make_spdf_dict(spdf, eco_species_ids)
    println("Marking sim years")
    @time site_sim_years = get_site_sim_years(spdf)
    #println(site_sim_years)
    #return
    #show(site_sim_years)

    println("Marking spinup cohorts")
    @time spinup_cohorts = get_spinup_cohorts(splots)
    #initial_cohorts = get_initial_cohorts(splots)
    println("beginning trials")
    SITES_PER_RUN = Int(round(nrow(site_sim_years) * 0.33))
    #Profile.clear()
    #Profile.init(n=10^7, delay=0.001)

    #best_loss = FloatType(Inf)
    best_result = SiteLoss(FloatType[], FloatType[], FloatType(Inf), 1)
    best_params = generate_biomass_params(species_list, eco_list, eco_species_ids; rng=RNG)
    cur = SACandidate(best_params, best_result)
    search_state = SAState(best=cur, current=cur, rng=RNG, max_iter=TRIALS,
        initial_t=1e3, t=1e3)
    if TRIALS < 1
        return search_state #best_loss, best_result, best_params
    end

    pbar = TProgress.ProgressBar(; expand=true)
    trials_pbar = TProgress.addjob!(pbar; N=TRIALS, description="Trials")
    #run_pbar = TProgress.addjob!(pbar; N=1, description="Years")
    canceled = Threads.Atomic{Bool}(false)
    println("Threads: $(Threads.nthreads())")
    try
        TProgress.with(pbar) do
            iter = 0
            while true #for trial in 1:TRIALS #ProgressBar(1:100)
                iter += 1
                search_state.i = iter
                #params = generate_biomass_params(RNG, n_species, n_ecoregions)
                params = mutate_biomass_params(search_state.current.x, params_dist; rng=RNG)
                eco_params = generate_eco_params(params)
                #println(params)
                #println("###making sites")
                sites = make_sites(splots, eco_species_ids; rng=RNG)
                chosen_sites = 1:length(sites)#StatsBase.sample(RNG, 1:length(sites), SITES_PER_RUN, replace=false, ordered=true)
                max_sim_year = site_sim_years.sim_years[chosen_sites] .|> maximum |> maximum
                #println(sites[chosen_sites])
                #println(max_sim_year)
                #println("###Sites made, beginning spinup")
                spinup_cohorts!(spinup_cohorts, sites, eco_params) #[splots.measdate .== splots.start_measdate,:])
                #println("Sites spun up")

                #
                ##
                ##
                ##
                ##
                ##
                #



                #reset_pjob!(pbar, run_pbar; N=max_sim_year + 1)

                #years_results = [DataFrame() for _ in 0:max_sim_year] #Vector{MDataFrame}(missing,max_sim_year+1)
                years_results = Vector{SiteLoss}(undef, max_sim_year + 1)
                for current_sim_year in 0:max_sim_year #ProgressBar(0:max_sim_year) #ProgressBar(0:50)
                    sites_results = Vector{SiteLoss}(undef, length(chosen_sites)) #[DataFrame() for _ in 1:length(chosen_sites)] #Vector{MDataFrame}(missing, length(chosen_sites))
                    #any_site_results = falses(Threads.nthreads())
                    Threads.@threads :static for i in eachindex(chosen_sites) #
                        canceled[] && break
                        @inbounds mapcode = chosen_sites[i]
                        @inbounds site = sites[mapcode]
                        @inbounds spdf_plt = spdf_plts[site.ref_cn]
                        @inbounds sim_years = site_sim_years.sim_years[mapcode]
                        if site.active
                            #println(site.mapcode)
                            succession_step!(current_sim_year, eco_params, site)
                            reproduction_step!(current_sim_year, eco_params, site)
                            # what years to check for this site
                            if current_sim_year in sim_years
                                #println(site.mapcode)
                                debug = false
                                if false && site.live > 0 #&& (current_sim_year == sim_years[1] || current_sim_year == sim_years[end])
                                    println("Thread: $(Threads.threadid())")
                                    println("Current_sim_year = $(current_sim_year)")
                                    println("Site: $(site.mapcode)")
                                    println(site)
                                    debug = true
                                end
                                sloss = calculate_site_loss2(current_sim_year, site, n_species, eco_species_ids, spdf_plt[current_sim_year], loss_params; debug=debug)
                                if false && site.live > 0 #&& (current_sim_year == sim_years[1] || current_sim_year == sim_years[end])
                                    println(site)
                                    println(spdf_plt[current_sim_year])
                                    println(sloss)
                                    println("----------------------------")
                                end

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

                    #TProgress.update!(run_pbar)

                end
                #run_result = reduce(vcat, years_results)
                run_result = sum(skipundef(years_results))
                @assert !any(isnan.(run_result.sp_w_loss)) "run NaN"



                next = SACandidate(params, run_result)

                if search_cmp!(next, search_state)
                    println(convert(Float64, search_state.best.fx))
                end
                if search_state.i % 10 == 0
                    println("Best@$(search_state.best_iteration): $(convert(Float64, search_state.best.fx)), Avg diff: $(search_state.diff_avg), Temp: $(search_state.t), ratio $(search_state.diff_avg/search_state.t), Prob: $(search_state.prob_avg)")
                    save_json(joinpath(output_dir, "best_params.json"), search_state.best.x)
                    JLD2.save_object(joinpath(output_dir, "best_params.jld2"), search_state.best.x)
                    JLD2.save_object(joinpath(output_dir, "search_state.jld2"), search_state)
                end
                if search_update_rule!(search_state)
                    break
                end
                #run_loss::FloatType = get_total_loss(run_result)
                #if run_loss <= best_loss
                #    best_loss = run_loss
                #    best_result = run_result
                #    best_params = params
                #end

                #show(run_result)
                #println("Run $(trial): $(run_result)")
                TProgress.update!(trials_pbar)
                #ProfileSVG.save("profile_$(trial).svg")
            end
        end
    catch e
        e isa InterruptException || rethrow(e)
        canceled[] = true
    finally
        #save_json("best_params.json", search_state.best)
        #save_json("search_state.json", search_state)
        #JLD2.save_object("best_params.jld2", search_state.best)
        JLD2.save_object(joinpath(output_dir, "search_state.jld2"), search_state)
        #println(search_state)
    end

    #println(typeof(best_loss), best_loss)
    #println(best_loss, best_result, best_params)
    return search_state #best_loss, best_result, best_params
end

function load_treemap_raster(raster_path::String; treemap_version::Int=2022)::Tuple{Array{Union{Missing,Int64}},Array{Union{Missing,Int64}},AttrTable}

    CN_FIELD_NAME = (treemap_version == 2016 ? "CN" : "PLT_CN")
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
            Threads.@threads :static for i in eachindex(A, outA)
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
        return outA, outA, val_att_dict
    end
end

function populate_initial_treemap_communities(plt_cn_raster::Array{Union{Missing,Int64}}, eco_raster::Matrix{Int16}, splots::DataFrame, eco_species_ids::Vector{Vector{UIntType}}; ecocode_field=:effective_ecocode, RNG::Union{Nothing,Random.AbstractRNG})::Array{Union{Missing,Site}}
    if isnothing(RNG)
        RNG = Random.default_rng()
    end

    splots_dict = Dict(
        (plt_key.plt_cn, plt_key.raster_ecocode) => DataFrame(plt_df)
        for (plt_key, plt_df) in pairs(groupby(splots, [:plt_cn, :raster_ecocode], sort=false))
    )
    #println(typeof(splots_dict))

    tRNGs = [Random.Xoshiro(rand(RNG, UInt64)) for _ in 1:Threads.maxthreadid()]
    totalpixels = zeros(UInt, Threads.maxthreadid())
    totalmissing = zeros(UInt, Threads.maxthreadid())
    outA = Array{Union{Missing,Site}}(missing, size(plt_cn_raster))

    begin
        Threads.@threads for i in eachindex(plt_cn_raster, outA)
            tid = Threads.threadid()
            @inbounds plt_cn = plt_cn_raster[i]
            @inbounds ecocode = eco_raster[i]

            if !ismissing(plt_cn) && !ismissing(ecocode)
                @inbounds totalpixels[tid] += 1
                @inbounds totalmissing[tid] += 1
                initial_cohorts = get(splots_dict, (plt_cn, Int64(ecocode)), missing)
                if !ismissing(initial_cohorts) && nrow(initial_cohorts) > 0
                    @inbounds tRNG = tRNGs[tid]
                    p = first(initial_cohorts)
                    n_cohorts = nrow(initial_cohorts)
                    cap = UIntType(2^ceil(log2(n_cohorts)))
                    site = Site(
                        active=true,
                        rng=Random.Xoshiro(rand(tRNG, UInt64)),
                        ecocode=UIntType(p.raster_ecocode), #UIntType(getproperty(p, ecocode_field)),
                        eco_id=p.eco_id,
                        mapcode=UIntType(i),
                        ref_cn=p.plt_cn,
                        cap=UIntType(cap),
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
                        sp_mature=falses(length(eco_species_ids[p.eco_id])),
                    )
                    for row in eachrow(initial_cohorts)
                        #@assert row.eco_species_id <= length(eco_species_ids[p.eco_id]) "$(length(eco_species_ids[p.eco_id])),\n$(p),\n$(row),\n$(initial_cohorts)"
                        add_new_cohort!(site, UIntType(row.eco_species_id), FloatType(row.age_calc), FloatType(row.agb_sum))
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
    eco_id::UIntType
    species_id::UIntType
    age::UIntType
    biomass::FloatType
end
function run_simulation(; site_raster::Array{Union{Missing,Site}}, params::BiomassSuccessionParams, output_dir::String, timehorizon::Int=30, RNG=Random.AbstractRNG)
    eco_params = generate_eco_params(params)
    FLUSH_THRESHOLD = 1000
    writer_buffer = Vector{SiteRecord}()
    sizehint!(writer_buffer, FLUSH_THRESHOLD)
    flush_task = nothing
    chunk = 1


    thread_buffers = [Vector{SiteRecord}() for _ in 1:Threads.maxthreadid()]
    TProgress.@track for current_sim_year in 1:timehorizon
        Threads.@threads :static for I in eachindex(site_raster)
            @inbounds site = site_raster[I]
            if !ismissing(site)
                #println(I)
                succession_step!(current_sim_year, eco_params, site)
                reproduction_step!(current_sim_year, eco_params, site)
                if current_sim_year % 5 == 0
                    compact_site!(site)
                end
                if current_sim_year > 0
                    tid = Threads.threadid()
                    @inbounds buf = thread_buffers[tid]

                    for i in 1:site.live
                        push!(buf, SiteRecord(current_sim_year, site.mapcode, site.eco_id, site.c_species[i], site.c_age[i], site.c_bio[i]))
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
                Parquet2.writefile(joinpath(output_dir, "year_$(current_sim_year)_chunk_$(chunk).parquet"), payload)
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
        Parquet2.writefile(joinpath(output_dir, "year_$(timehorizon)_chunk_$(chunk).parquet"), writer_buffer)
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
                Parquet2.writefile("./outputs/chunk_$(chunk).parquet", writer_buffer)
                @info "Flushed chunk $(chunk)"
                chunk += 1
                empty!(writer_buffer)
            end
        end

        if !isempty(writer_buffer)
            Parquet2.writefile("./outputs/chunk_$(chunk).parquet", writer_buffer)
            @info "Flushed final chunk $(chunk)"
        end
    end

    thread_buffers = [Vector{SiteRecord}() for _ in 1:Threads.maxthreadid()]
    TProgress.@track for current_sim_year in 1:years
        Threads.@threads :static for I in eachindex(site_raster)
            cancelled[] && break
            @inbounds site = site_raster[I]
            if !ismissing(site)
                #println(I)
                succession_step!(current_sim_year, eco_params, site)
                reproduction_step!(current_sim_year, eco_params, site)
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

function write_raster(src_path::String, output_path::String, output::Array{Union{Missing,Float32}}; nodata::Float32=-9999.0f0)
    ArchGDAL.read(src_path) do src
        data = replace(output, missing => nodata)
        ArchGDAL.create(
            output_path,
            driver=ArchGDAL.getdriver("GTiff"),
            width=ArchGDAL.width(src),
            height=ArchGDAL.height(src),
            nbands=1,
            dtype=Float32,
        ) do dst
            ArchGDAL.setgeotransform!(dst, ArchGDAL.getgeotransform(src))
            ArchGDAL.setproj!(dst, ArchGDAL.getproj(src))
            ArchGDAL.write!(ArchGDAL.getband(dst, 1), data)
        end
    end
end
function generate_output_mapcodes(; output_dir::String, ref_raster_path::String, site_raster)
    AG.read(ref_raster_path) do src
        src_band = AG.getband(src, 1)
        src_data = AG.read(src_band)
        dst_data = zeros(Int32, size(src_data))
        Threads.@threads :static for i in eachindex(site_raster)
            @inbounds site = site_raster[i]
            if !ismissing(site)
                @inbounds dst_data[Int(site.mapcode)] = Int32(site.mapcode)
            end
        end
        AG.create(
            joinpath(output_dir, "output_communities.tif"),
            driver=AG.getdriver("GTiff"),
            width=AG.width(src),
            height=AG.height(src),
            nbands=1,
            dtype=Int32,
        ) do dst
            AG.setgeotransform!(dst, AG.getgeotransform(src))
            dst_band = AG.getband(dst, 1)
            AG.setnodatavalue!(dst_band, Int32(0))
            AG.setproj!(dst, AG.getproj(src))
            AG.write!(AG.getband(dst, 1), dst_data)
        end
    end
end
function generate_rasters_from_output(; data_dir::String, ref_raster_path::String, output_dir::String)
    ref_raster_path = joinpath(data_dir, ref_raster_path)
    files = Glob.glob("year_*_chunk_*.parquet", output_dir)
    re = r"year_(\d+)_chunk_\d+\.parquet"
    @inline extract_year = filename::String -> parse(Int, match(re, filename).captures[1])
    years_files = Dict{Int,Vector{String}}()
    for f in files
        year = extract_year(f)
        push!(get!(years_files, year, String[]), f)
    end
    years_files = sort(collect(years_files), by=first)

    AG.read(ref_raster_path) do src
        src_band = AG.getband(src, 1)
        src_data = AG.read(src_band)
        TProgress.@track for (year, files) in years_files
            dst_data = zeros(size(src_data))
            for f in files
                ds = Parquet2.Dataset(f)
                for chunk in Parquet2.Tables.partitions(ds)
                    mapcode = Parquet2.Tables.getcolumn(chunk, :mapcode)
                    biomass = Parquet2.Tables.getcolumn(chunk, :biomass)
                    for i in eachindex(mapcode)
                        @inbounds dst_data[mapcode[i]] += Float32(biomass[i])
                    end
                end
            end
            #dst_data = replace(output)
            AG.create(
                joinpath(output_dir, "agb_$(year).tif"),
                driver=AG.getdriver("GTiff"),
                width=AG.width(src),
                height=AG.height(src),
                nbands=1,
                dtype=Float32,
            ) do dst
                AG.setgeotransform!(dst, AG.getgeotransform(src))
                dst_band = AG.getband(dst, 1)
                AG.setnodatavalue!(dst_band, 0.0f0)
                AG.setproj!(dst, AG.getproj(src))
                AG.write!(AG.getband(dst, 1), dst_data)
            end
        end



    end

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
    #splots, n_plots, n_species, n_ecoregions = DataFrame(), UIntType(400), UIntType(15), UIntType(5)
    #load_cohorts_sqlite(;filter_ecos=["8.3.5.65o", "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"])
    #return

    #eco_species_ids = [sort!(Random.randperm(rng, n_species)[1:rand(rng,1:n_species)])
    #                        for _ in 1:n_ecoregions]
    #params = generate_biomass_params(n_species, n_ecoregions, eco_species_ids; rng=RNG)
    #generate_eco_params(params)

    #return
    #BIOMASS_PARAM_DISTS = make_biomass_param_dists(n_species, n_ecoregions)
    loss_params = LossParams(
        age_bins=AgeBins(
            bins_idx=[5, 10, 20, 40, 60, 80] .|> Int,
            last_bin_open=true
        ),
        smoothing_weights=get_smoothing_window(; smoothing_window=1, smoothing_variance=FloatType(1.0f0))
    )
    #return
    #splots, n_plots, n_species, n_ecoregions = load_cohorts_csv(filter_ecos=["8.3.5.65o", "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"])
    #filter_ecos = ["8.5.3.75e", "8.5.3.75f", "8.5.3.75a", "8.5.3.75c", "8.5.3.75g", "8.3.5.65o", "8.5.3.75d", "8.5.3.75h", "8.3.5.65h", "8.3.5.65f", "8.3.5.65g", "15.4.1.76b", "8.5.3.75b", "8.5.3.75i", "9.4.7.32b", "8.3.7.35b", "8.3.7.35e", "8.5.1.63h", "8.3.7.35g", "8.3.7.35f", "8.3.5.65l", "8.3.5.65c", "9.5.1.34a"]
    #filter_ecos = ["8.3.5.65o", "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"]
    #filter_ecos = ["8.3.5.65o"#, "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"]
    #filter_ecos = ["8.3.5.65o", "8.5.3.75a", "8.5.3.75c", "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"] 
    filter_ecos = ["8.5.3.75g"]#, "8.5.3.75a", "8.5.3.75c", "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"] 
    println("Filtering ecos: $(filter_ecos)")
    search_state = parametrize(; cohorts_db_path="../data_eco_l4_cohorts.db", tablename="data_eco_cohorts_g", output_dir="./outputs", loss_params=loss_params, filter_ecos=filter_ecos, skip_disturbances=true, RNG=RNG, TRIALS=20000)
    println("Best Loss: $(search_state.best.fx)")

end
function load_eco_raster(raster_eco_path::String)::Matrix{Int16}

    ds = AG.readraster(raster_eco_path)
    band = AG.getband(ds, 1)
    #nodataval = AG.getnodatavalue(band)
    A = AG.read(band)
    AG.destroy(ds)
    return A
end
function make_effective_splots(df::DataFrame)#::Tuple{DataFrame,Array{String},Array{String}, Array{String},Array{Array{(Int,Int)}}}#, effective_eco_field::Union{Nothing,Symbol}=:effective_eco, ecocode_fields::Union{Nothing,Array{Symbol}}=nothing)::Tuple{DataFrame,Array{String},Array{String},Array{Array{Int}}}


    #df.species_id = groupindices(groupby(df,:species_field))
    #df.eco_id = groupindices(groupby(df,:eco))

    #println(df)

    eco_vals = sort(unique(df.raster_eco))
    eco_dict = Dict(eco => i for (i, eco) in enumerate(eco_vals))
    # to keep same ids for eco_vals

    effective_eco_vals = sort(unique(df.effective_eco))#vcat(sort(setdiff(eco_vals,unique(df.effective_eco))) , sort(setdiff(unique(df.effective_eco), eco_vals)))
    effective_eco_dict = Dict(eco => i for (i, eco) in enumerate(effective_eco_vals))

    effective_species_symbol_map_vals = sort(unique(df.effective_species_symbol_map))
    effective_species_symbol_map_dict = Dict(ssm => i for (i, ssm) in enumerate(effective_species_symbol_map_vals))
    #println(eco_vals)
    #println(effective_eco_vals)
    #println(effective_species_symbol_map_vals)

    df.species_id = getindex.(Ref(effective_species_symbol_map_dict), df.effective_species_symbol_map)
    df.eco_id = getindex.(Ref(eco_dict), df.raster_eco)
    df.effective_eco_id = getindex.(Ref(effective_eco_dict), df.effective_eco)
    #println(df)
    #
    #


    #maps list species text
    #maps list eco_text

    df.measdate = Dates.DateTime.(df.measdate, Dates.dateformat"yyyy-mm-dd")

    fields = [:plt_cn, :statecd, :unitcd, :countycd, :plot, :raster_ecocode, :eco_id, :effective_eco_id, :measdate, :species_id, :age_calc]
    plots = combine(groupby(df, fields, sort=false), nrow => :count, :agb => sum => :agb_sum)
    # start_measdate =
    start_measdates = combine(groupby(plots, [:statecd, :unitcd, :countycd, :plot], sort=false)) do rows
        (; start_measdate=[minimum(rows.measdate)])
    end
    plots_measdate = innerjoin(plots, start_measdates, on=[:statecd, :unitcd, :countycd, :plot])
    splots = sort!(plots_measdate, [:measdate, :statecd, :unitcd, :countycd, :plot, :age_calc, :species_id])

    splots.plot_id .= groupindices(groupby(splots, [:statecd, :unitcd, :countycd, :plot, :raster_ecocode])) .|> UIntType


    # make indices for the data structure
    # eco_id -> (effective_eco_id, effective_species_id)
    #eco_species_id_map = Dict((eco_id, species_id) => eco_species_id
    #                          for (eco_id, species_ids) in enumerate(ddf.species_ids)
    #                          for (eco_species_id, species_id) in enumerate(species_ids))
    #splots.eco_species_id .= getindex.(Ref(eco_species_id_map), zip(splots.eco_id, splots.species_id))
    #println(effective_species_symbol_map_vals)
    #println(eco_vals)
    #ddf = combine(groupby(df, eco_id_fields, sort=true)) do rows
    #(; effective_species_ids=[sort(unique(zip(rows.species_id,rows.effective_eco_id)))])
    #(; species_ids=[sort(unique(rows.species_id))])
    #end

    #eco_species_id_fields = [:eco_id, :effective_eco_id, :species_id]
    #eco_species_id_df = select(splots, eco_species_id_fields) |> unique |> sort
    #eco_species_id_df = combine(groupby(eco_species_id_df, :eco_id, sort=true)) do rows
    #    (; effective_eco_id=rows.effective_eco_id, species_id=rows.species_id, eco_species_id=1:nrow(rows))
    #end
    #println(eco_species_id_df)
    #eco_species_id_dict = Dict(
    #    (row.eco_id, row.effective_eco_id, row.species_id) => row.eco_species_id
    #    for row in eachrow(eco_species_id_df)
    #)

    #println(eco_species_id_dict)
    #println("uuuOK?")
    #splots = innerjoin(splots, eco_species_id_df, on=eco_species_id_fields)
    ##.eco_species_id .= getindex.(Ref(eco_species_id_dict), eachrow(splots))
    ##disallowmissing!(splots)
    return splots, eco_vals, effective_eco_vals, effective_species_symbol_map_vals

end
function make_splots(df::DataFrame)::Tuple{DataFrame,Array{String},Array{String},Array{Array{Int}}}


    #df.species_id = groupindices(groupby(df,:species_field))
    #df.eco_id = groupindices(groupby(df,:eco))

    #println(df)

    eco_vals = sort(unique(df.eco))
    eco_dict = Dict(eco => i for (i, eco) in enumerate(eco_vals))
    species_symbol_map_vals = sort(unique(df.species_symbol_map))
    species_symbol_map_dict = Dict(ssm => i for (i, ssm) in enumerate(species_symbol_map_vals))

    df.species_id = getindex.(Ref(species_symbol_map_dict), df.species_symbol_map)
    df.eco_id = getindex.(Ref(eco_dict), df.eco)
    #println(df)
    #
    #

    ddf = combine(groupby(df, :eco_id, sort=true)) do rows
        (; species_ids=[sort(unique(rows.species_id))])
    end

    #maps list species text
    #maps list eco_text

    df.measdate = Dates.DateTime.(df.measdate, Dates.dateformat"yyyy-mm-dd")

    fields = [:plt_cn, :statecd, :unitcd, :countycd, :plot, :eco_id, :measdate, :species_id, :age_calc]
    plots = combine(groupby(df, fields, sort=false), nrow => :count, :agb => sum => :agb_sum)
    # start_measdate =
    start_measdates = combine(groupby(plots, [:statecd, :unitcd, :countycd, :plot], sort=false)) do rows
        (; start_measdate=[minimum(rows.measdate)])
    end
    plots_measdate = innerjoin(plots, start_measdates, on=[:statecd, :unitcd, :countycd, :plot])
    splots = sort!(plots_measdate, [:measdate, :statecd, :unitcd, :countycd, :plot, :age_calc, :species_id])

    splots.plot_id .= groupindices(groupby(splots, [:statecd, :unitcd, :countycd, :plot])) .|> UIntType

    eco_species_id_map = Dict((eco_id, species_id) => eco_species_id
                              for (eco_id, species_ids) in enumerate(ddf.species_ids)
                              for (eco_species_id, species_id) in enumerate(species_ids))
    splots.eco_species_id .= getindex.(Ref(eco_species_id_map), zip(splots.eco_id, splots.species_id))
    #println(effective_species_symbol_map_vals)
    #println(eco_vals)
    return splots, eco_vals, species_symbol_map_vals, ddf.species_ids

end

function get_treemap_cohorts(cn_raster, eco_raster, cohorts_db, eco_ecocode_mapping_csv)
    eco_ecocode_mapping_df = CSV.read(eco_ecocode_mapping_csv, DataFrame)

    cn_eco::Dict{Tuple{Int64,Int16},Int64} = StatsBase.countmap((cn, eco) for (cn, eco) in zip(cn_raster, eco_raster) if !ismissing(cn))
    cn_eco_df = DataFrame(CN=[k[1] for k in keys(cn_eco)],
        ecocode=[Int64(k[2]) for k in keys(cn_eco)], #sqlite freaks out if not int64
        count=collect(values(cn_eco)))

    #splots, n_plots, n_species, n_ecoregions = load_raster_cohorts()
    #df (cn, eco) => join splots O on O.PLT_CN = df.CN => Left outer join species_mapping M on df.eco = M.eco AND O.species = M.species (some species_symbol_map nulls) => actual_eco =if M.species_symbol_map is null (O.eco, O.species_symbol_map) else (M.eco, M.species_symbol_map)
    #load_cohorts_sqlite/
    db = SQLite.DB(cohorts_db)
    SQLite.execute(db, "PRAGMA temp_store=MEMORY")

    SQLite.execute(db, "CREATE INDEX IF NOT EXISTS PLT_CN_IDX ON data_eco_cohorts(PLT_CN);")
    SQLite.execute(db, "CREATE INDEX IF NOT EXISTS SPECIES_ECO_IDX ON data_species_eco_map(ECO);")

    SQLite.execute(db, "CREATE TABLE IF NOT EXISTS data_all_species AS SELECT DISTINCT species_symbol_map FROM data_eco_cohorts;")
    SQLite.execute(db, "CREATE UNIQUE INDEX IF NOT EXISTS ALL_SPECIES_IDX ON data_all_species(species_symbol_map);")

    SQLite.load!(cn_eco_df, db, "cn_eco"; temp=true)
    SQLite.load!(eco_ecocode_mapping_df, db, "eco_ecocode_map"; temp=true)

    SQLite.execute(db, "CREATE UNIQUE INDEX IF NOT EXISTS ECOCODE_IDX ON eco_ecocode_map(ecocode);")
    SQLite.execute(db, "CREATE UNIQUE INDEX IF NOT EXISTS ECO_IDX ON eco_ecocode_map(eco);")
    #TODO:  Actually not all species_symbol_maps are available neither in extent eco (disturb) nor the original eco (not parametrized or disturbed)
    #Probably also have another step of adding the species to target eco's catch all eco_H/S
    # if sp in eco -> sp
    # else catch all eco_H/S -> catch all
    # else bring from original eco, but mark effective_{eco,ecocode,sp} from original
    #
    # ultimately there should be only one (ecocode, sp) pair because they could be
    # coming from many different ecos, therefore will be ranked by sum(tree_count) and only is taken
    #
    sql = "
            With full_table AS (
        SELECT df.CN,df.ecocode raster_ecocode, e.eco raster_eco,  o.*,
            (COALESCE(m.species_symbol_map,b.species_symbol_map) IS NULL) borrowed,
            (CASE WHEN COALESCE(m.species_symbol_map,b.species_symbol_map) IS NULL THEN o.eco ELSE e.eco END) effective_eco,
            (CASE WHEN COALESCE(m.species_symbol_map,b.species_symbol_map) IS NULL THEN eo.ecocode ELSE e.ecocode END) effective_ecocode,
            (CASE WHEN COALESCE(m.species_symbol_map,b.species_symbol_map) IS NULL THEN o.species_symbol_map ELSE COALESCE(m.species_symbol_map,b.species_symbol_map) END) effective_species_symbol_map

            FROM cn_eco df
            JOIN eco_ecocode_map e ON df.ecocode = e.ecocode
            JOIN data_eco_cohorts o ON df.CN = o.PLT_CN
            JOIN eco_ecocode_map eo ON o.eco = eo.eco
            LEFT OUTER JOIN data_species_eco_map m ON m.eco = e.eco AND m.species_symbol = o.species_symbol
            LEFT OUTER JOIN data_all_species b ON b.species_symbol_map = concat(e.eco,'_',o.sftwd_hrdwd)
         ),group_totals AS (
            SELECT
                raster_ecocode,
                effective_eco,
                effective_ecocode,
                effective_species_symbol_map,
                SUM(tree_count) AS group_count
            FROM full_table
            GROUP BY raster_ecocode, effective_species_symbol_map, effective_eco, effective_ecocode
            ),
        dominant AS (
        SELECT
                raster_ecocode,
                effective_eco,
                effective_ecocode,
                effective_species_symbol_map
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY raster_ecocode, effective_species_symbol_map ORDER BY group_count DESC) AS rn
            FROM group_totals
        ) WHERE rn = 1
)
        SELECT
        d.effective_eco,
        d.effective_ecocode,
        d.effective_species_symbol_map,
        t.*
FROM full_table t
JOIN dominant d
    ON t.raster_ecocode = d.raster_ecocode
    AND t.effective_species_symbol_map = d.effective_species_symbol_map;

        "

    #println(sql)
    df = SQLite.DBInterface.execute(db, sql) |> DataFrame
    #println(df)
    #println(unique(df.raster_ecocode))
    #println(unique(df.effective_ecocode))
    #println(unique(df.raster_eco))
    #println(unique(df.effective_eco))
    SQLite.close(db)
    @time return make_effective_splots(df)

end

function simulate_treemap_raster(; data_dir::String, output_dir::String, cohorts_db::String, treemap_raster::String, eco_raster::String, eco_ecocode_mapping::String, biomass_succession_parameters::String, RNG_seed=1337, timehorizon_years::Int=50, treemap_version=2022)
    biomass_succession_parameters_path = joinpath(data_dir, biomass_succession_parameters)
    treemap_raster_path = joinpath(data_dir, treemap_raster)
    cohorts_db_path = joinpath(data_dir, cohorts_db)
    eco_raster_path = joinpath(data_dir, eco_raster)
    eco_ecocode_mapping_path = joinpath(data_dir, eco_ecocode_mapping)
    #println("Extracting plots")
    #@time splots, eco_list, species_list, eco_species_ids = get_treemap_cohorts1(cohorts_db_path)
    #return

    RNG = Random.Xoshiro(RNG_seed)
    println("Loading parametrs: $(biomass_succession_parameters_path)")
    params = JLD2.load_object(biomass_succession_parameters_path)
    println("Loading Raster: $(treemap_raster_path)")
    @time cn_raster, vat = load_treemap_raster(treemap_raster_path, treemap_version=treemap_version)

    println("Loading Eco Raster: $(eco_raster_path)")
    @time eco_raster = load_eco_raster(eco_raster_path)

    @assert size(cn_raster) == size(eco_raster) "Size mismatch treemap Raster $(size(cn_raster)) != Eco raster $(size(eco_raster))"
    println("Extracting plots: $(cohorts_db_path)")
    @time splots, eco_list, effective_eco_list, species_list = begin
        cohorts_file = "./cohorts.jld2"
        skip_cohort_extraction = false
        if !skip_cohort_extraction
            splots, eco_list, effective_eco_list, species_list = get_treemap_cohorts(cn_raster, eco_raster, cohorts_db_path, eco_ecocode_mapping_path)
            JLD2.save_object(cohorts_file, (splots, eco_list, effective_eco_list, species_list))
            splots, eco_list, effective_eco_list, species_list
        else
            JLD2.load_object(cohorts_file)
        end
    end



    n_species = length(species_list)
    n_ecoregions = length(eco_list)
    n_plots = maximum(splots.plot_id)
    println("Plots:$n_plots, Ecos:$n_ecoregions, Species:$n_species, Measurements: $(size(splots))")
    #println(splots)
    #println("ok?")

    #reorder eco_species_ids

    #remapping
    #println(length(params.SPECIES_LIST))
    #println(length(species_list))
    params_species_df = DataFrame(param_species=params.SPECIES_LIST, param_species_id=1:length(params.SPECIES_LIST))
    data_species_df = DataFrame(species=species_list, data_species_id=1:length(species_list))
    joint_species_df = innerjoin(data_species_df, params_species_df, on=:species => :param_species)
    #species_ids = Dict(species => i for i, species in sort(unique(joint_species_df.speceis)))
    joint_species_df.species_id .= groupindices(groupby(joint_species_df, :species, sort=true))
    # species_symbol_map -> species_id (from data) and param_species_id (in param)
    #println(joint_species_df)

    params_eco_df = DataFrame(param_eco=params.ECO_LIST, param_eco_id=1:length(params.ECO_LIST))
    data_eco_df = DataFrame(eco=eco_list, eco_id=1:length(eco_list))
    data_effective_eco_df = DataFrame(effective_eco=effective_eco_list, effective_eco_id=1:length(effective_eco_list))
    joint_eco_df = innerjoin(data_eco_df, params_eco_df, on=:eco => :param_eco)
    joint_effective_eco_df = innerjoin(data_effective_eco_df, params_eco_df, on=:effective_eco => :param_eco)
    param_eco_species_id_df = DataFrame([(e, i, s) for (e, ss) in enumerate(params.ECO_SPECIES_IDS) for (i, s) in enumerate(ss)], [:param_eco_id, :param_eco_species_id, :param_species_id])

    #joint_splots = splots & joint_specie_df (ON species_id) & joint_effective_eco_df (ON effective_eco_id)
    #       (eco_id, species_id, effective_eco_id,  param_species_id, param_eco_id)
    # joint_splots & joint param_eco_species_id (ON param_eco_id &  param_species_id)
    #       (eco_id, species_id, effective_eco_id,  param_species_id, param_eco_id, param_eco_species_id)
    #       now we have (eco_id, param_eco_id, param_eco_species_id) to pick out (eco_id, effective_eco_id, species_id)
    #       #
    mapped_splots_df = innerjoin(
        innerjoin(
            innerjoin(
                splots, joint_species_df, on=:species_id),
            joint_effective_eco_df, on=:effective_eco_id),
        param_eco_species_id_df, on=[:param_eco_id, :param_species_id])
    eco_species_id_fields = [:eco_id, :effective_eco_id, :species_id, :param_eco_id, :param_eco_species_id]
    eco_species_id_df = select(mapped_splots_df, eco_species_id_fields) |> unique |> sort
    eco_species_id_df = combine(groupby(eco_species_id_df, :eco_id, sort=true)) do rows
        (; effective_eco_id=rows.effective_eco_id,
            species_id=rows.species_id,
            param_eco_id=rows.param_eco_id,
            param_eco_species_id=rows.param_eco_species_id,
            eco_species_id=1:nrow(rows))
    end
    #println(eco_species_id_df)
    #eco_species_id_dict = Dict(
    #    (row.eco_id, row.effective_eco_id, row.species_id) => row.eco_species_id
    #    for row in eachrow(eco_species_id_df)
    #)

    #println(eco_species_id_dict)
    mapped_splots_df = innerjoin(mapped_splots_df, eco_species_id_df, on=eco_species_id_fields)
    sort!(mapped_splots_df, [:measdate, :statecd, :unitcd, :countycd, :plot, :age_calc])
    disallowmissing!(mapped_splots_df)
    #println("uuuOK?")
    #println(mapped_splots_df)


    eco_param_species_ids_df = combine(groupby(mapped_splots_df, [:eco_id], sort=true)) do rows
        (; param_eco_species_id_selector=[sort(unique(zip(rows.species_id, rows.param_eco_id, rows.param_eco_species_id)))])
    end

    #println(eco_param_species_ids_df.param_eco_species_id_selector)
    #println("Ok?")

    #for eee in eco_param_species_ids_df.param_eco_species_id_selector
    #    @assert (length(eee) == 0 || length(eee) == eee[end][1]) "$(length(eee)) $(eee[end]) $(eee[end][1])"
    #end
    #readline()



    ## joint joint eco
    #println(length(params.ECO_LIST))
    #println(length(eco_list))
    #println(joint_eco_df)
    #println(joint_effective_eco_df)


    ## join effective eco + param_eco, now we know what (eco="8.9.1.1a") has eco_id for both
    ## now we need to know how the eco_species_ids line up
    ## we need each species_id with its corresponding param_species_id (now we know "PIEL" species_id for both)
    ## # now we need param_eco_species_id which is the index of param_species_id within param_eco param
    ## for each data eco param -> [(effective_eco_id, species_id)] -> [(param_eco_id, param_eco_species_id)]
    ## join effective
    ## do not need effective_eco_id, only eco_id
    ## eco,effective_eco, param_eco, param_eco_id, param_sp_id, effective_sp_id , effective_param_sp_id
    ##join

    #data_effective_eco_species_id_df = DataFrame([(e, s) for (e, ss) in enumerate(eco_species_ids) for s in ss], [:eco_id, :species_id])
    #param_eco_species_id_df = DataFrame([(e, i, s) for (e, ss) in enumerate(params.ECO_SPECIES_IDS) for (i, s) in enumerate(ss)], [:param_eco_id, :param_eco_species_index, :param_species_id])
    #println(param_eco_species_id_df)
    #joint_eco_species_id_df = innerjoin(innerjoin(param_eco_species_id_df, joint_effective_eco_df, on=:param_eco_id), joint_species_df, on=:param_species_id)
    #println(joint_eco_species_id_df)
    #narrow_joint_eco_species_id_df = innerjoin(joint_eco_species_id_df, data_eco_species_id_df, on=[:eco_id, :species_id])
    #println(narrow_joint_eco_species_id_df)


    #mapped_df = combine(groupby(narrow_joint_eco_species_id_df, [:eco_id, :param_eco_id], sort=true)) do rows
    #    (; param_selectors=[rows.param_eco_species_index], species_ids=[rows.species_id])
    #end
    #println(mapped_df)
    #println(eco_species_ids)






    #param2data_species_map = Dict(psid => sid
    #                              for (sid, psid) in zip(joint_species_df.species_id, joint_species_df.param_species_id))
    #println(param2data_species_map)

    #println(eco_species_ids)

    ##SPINUP_MORTALITY_FRACTION = params.SPINUP_MORTALITY_FRACTION
    ##SUFFICIENT_LIGHT = params.SUFFICIENT_LIGHT
    ##ECO_LIST = eco_list
    ##SPECIES_LIST = species_list
    ##ECO_SPECIES_IDS = mapped_df.species_ids
    ##MIN_REL_BIOMASS = params.MIN_REL_BIOMASS[joint_eco_df.param_eco_id]

    ##D = params.D[joint_species_df.param_species_id]
    ##S = params.S[joint_species_df.param_species_id]
    ##LONGEVITY = params.LONGEVITY[joint_species_df.param_species_id]
    ##SHADE_TOL = params.SHADE_TOL[joint_species_df.param_species_id]
    ##MATURITY = params.MATURITY[joint_species_df.param_species_id]

    ##B_MAX_SPP = [params.B_MAX_SPP[param_eco_id][param_selector] for (param_eco_id, param_selector) in zip(mapped_df.param_eco_id, mapped_df.param_selectors)]
    ##ANPP_MAX_SPP = [params.ANPP_MAX_SPP[param_eco_id][param_selector] for (param_eco_id, param_selector) in zip(mapped_df.param_eco_id, mapped_df.param_selectors)]
    ##PROB_MORT_SPP = [params.PROB_MORT_SPP[param_eco_id][param_selector] for (param_eco_id, param_selector) in zip(mapped_df.param_eco_id, mapped_df.param_selectors)]
    ##PROB_ESTAB_SPP = [params.PROB_ESTAB_SPP[param_eco_id][param_selector] for (param_eco_id, param_selector) in zip(mapped_df.param_eco_id, mapped_df.param_selectors)]
    @inline function eco_species_selector(param)
        [[getproperty(params, param)[param_eco_id][param_eco_species_id]
          for (species_id, param_eco_id, param_eco_species_id) in
          param_stuff]


         for param_stuff in
         eco_param_species_ids_df.param_eco_species_id_selector]

    end
    new_params_eco_species_id = [[species_id
                                  for (species_id, param_eco_id, param_eco_species_id) in
                                  param_stuff]


                                 for param_stuff in
                                 eco_param_species_ids_df.param_eco_species_id_selector]
    println(new_params_eco_species_id)
    println([maximum(eco_species_id) for eco_species_id in new_params_eco_species_id])
    println(length(joint_species_df.species))

    mod_params = BiomassSuccessionParams(
        SPINUP_MORTALITY_FRACTION=params.SPINUP_MORTALITY_FRACTION,
        SUFFICIENT_LIGHT=params.SUFFICIENT_LIGHT,
        ECO_LIST=joint_eco_df.eco,
        SPECIES_LIST=joint_species_df.species, #species_list,
        ECO_SPECIES_IDS=new_params_eco_species_id, #mapped_df.species_ids,
        MIN_REL_BIOMASS=params.MIN_REL_BIOMASS[joint_eco_df.param_eco_id],
        D=params.D[joint_species_df.param_species_id],
        S=params.S[joint_species_df.param_species_id],
        LONGEVITY=params.LONGEVITY[joint_species_df.param_species_id],
        SHADE_TOL=params.SHADE_TOL[joint_species_df.param_species_id],
        MATURITY=params.MATURITY[joint_species_df.param_species_id],
        B_MAX_SPP=eco_species_selector(:B_MAX_SPP),
        ANPP_MAX_SPP=eco_species_selector(:ANPP_MAX_SPP),
        PROB_MORT_SPP=eco_species_selector(:PROB_MORT_SPP),
        PROB_ESTAB_SPP=eco_species_selector(:PROB_ESTAB_SPP),
        #[
        #B_MAX_SPP=[params.B_MAX_SPP[param_eco_id][param_selector] for (param_eco_id, param_selector) in zip(mapped_df.param_eco_id, mapped_df.param_selectors)],
        #ANPP_MAX_SPP=[params.ANPP_MAX_SPP[param_eco_id][param_selector] for (param_eco_id, param_selector) in zip(mapped_df.param_eco_id, mapped_df.param_selectors)],
        #PROB_MORT_SPP=[params.PROB_MORT_SPP[param_eco_id][param_selector] for (param_eco_id, param_selector) in zip(mapped_df.param_eco_id, mapped_df.param_selectors)],
        #PROB_ESTAB_SPP=[params.PROB_ESTAB_SPP[param_eco_id][param_selector] for (param_eco_id, param_selector) in zip(mapped_df.param_eco_id, mapped_df.param_selectors)],
    )

    @assert joint_eco_df.eco == eco_list "eco prob, $(joint_eco_df.eco) != $(eco_list)"
    #@assert joint_species_df.species == species_list "species prob, $(joint_species_df.species) == $(species_list)"
    l1 = [length(xs) for xs in mod_params.B_MAX_SPP]
    l2 = [length(xs) for xs in new_params_eco_species_id]
    ###println(species_list[eco_species_ids[15]])
    @assert l1 == l2 "eco x sp, $(l1) != $(l2)"
    #mod_params = Setfield.setproperties(params, (
    #    ECO_LIST=eco_list,
    #    SPECIES_LIST=species_list,
    #    ECO_SPECIES_IDS = mapped_df.species_ids,
    #    MIN_REL_BIOMASS = params.MIN_REL_BIOMASS[joint_eco_df.param_eco_id...],

    #    D=params.D[joint_species_df.param_species_id...],
    #    S=params.S[joint_species_df.param_species_id...],
    #    LONGEVITY=params.LONGEVITY[joint_species_df.param_species_id...],
    #    SHADE_TOL=params.SHADE_TOL[joint_species_df.param_species_id...],
    #    MATURITY=params.MATURITY[joint_species_df.param_species_id...],

    #    B_MAX_SPP = [params.B_MAX_SPP[param_selector...] for param_selector in mapped_df.param_selectors],
    #    ANPP_MAX_SPP = [params.ANPP_MAX_SPP[param_selector...] for param_selector in mapped_df.param_selectors],
    #    PROB_MORT_SPP = [params.PROB_MORT_SPP[param_selector...] for param_selector in mapped_df.param_selectors],
    #    PROB_ESTAB_SPP = [params.PROB_ESTAB_SPP[param_selector...] for param_selector in mapped_df.param_selectors],



    #    #ECO_SPECIES_IDS=[get.(Ref(param2data_species_map), params.ECO_SPECIES_IDS[e], missing)
    #    #                 for e in joint_eco_df.param_eco_id]
    #))
    #load right params
    println(mod_params)



    # extract ecoregion map for raster, extract plots
    #splots, n_plots, n_species, n_ecoregions = load_cohorts_csv()
    println("Populating Raster")
    @time site_raster = populate_initial_treemap_communities(cn_raster, eco_raster, mapped_splots_df, mod_params.ECO_SPECIES_IDS; RNG=RNG)
    println("Generating output mapcodes raster")
    @time generate_output_mapcodes(; output_dir=output_dir, ref_raster_path=treemap_raster_path, site_raster=site_raster)
    # generate_eco_params
    println("Running simulation")
    @time run_simulation(; site_raster=site_raster, params=mod_params, output_dir=output_dir, RNG=RNG, timehorizon=timehorizon_years)
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
