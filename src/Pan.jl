module Pan
include("PanCore.jl")
include("Plugins.jl")
include("Parametrization.jl")
include("Search.jl")
include("Data.jl")
#for file in filter(f -> endswith(f,"Plugin.jl"), readdir("src/plugins";join=false))
#    println("including plugin $(file)")
#    include(joinpath("plugins", file))
#end
#include("plugins/BiomassSuccessionPlugin.jl")
using .PanCore
using .Plugins: BaseSitePlugin, BiomassSuccessionPlugin
import .Parametrization as PU
import .Parametrization.BiomassSuccessionParametrization as BSP
using .Search: SA, LBSA
import .Data as Data
import Dates


import Random
import Distributions as Dists
import Term.Progress as TProgress
import JLD2
using DataFrames



const ActivePlugins = (BaseSitePlugin.BaseSite, BiomassSuccessionPlugin.BiomassSuccession)
const ActiveSoA = SiteSoA{Tuple{(ActivePlugins)...}}

function make_sites(splots::DataFrame, eco_species_ids::Vector{Vector{Int}}; rng::Random.AbstractRNG, spinup::Bool=true)
    #eco_ids = unique(select(splots, [:eco_id]))
    #n_species = maximum(splots.species_id)
    #plot_ids = unique(select(splots, [:plot_id]))
    #plot_eco_ids = unique(select(splots, [:plot_id, :eco_id]))
    #plot_eco_ids = sort!(plot_eco_ids, [:plot_id])
    df = unique(select(splots, [:plot_id, :eco_id]))
    n = nrow(df)
    #plot_eco_ids = sort!(plot_eco_ids, [:plot_id])
    #@assert (nrow(plot_ids) == nrow(plot_eco_ids)) "Error: plot_ids and plot_eco_ids are not of equal length!"

    #index = eco_id * plot_id

    #splots_dict::Dict{Int64,DataFrame} = Dict(
    #    plt_key.plt_cn => DataFrame(plt_df)
    #    for (plt_key, plt_df) in pairs(groupby(splots, :plt_cn, sort=false))
    #)
    species_counts = [Int32(length(eco_species_ids[row.eco_id])) for row in eachrow(df)]
    if spinup
        cohort_counts = fill(Int32(2), n)
        @debug cohort_counts
        soa = ActiveSoA((cohort=cohort_counts, species=species_counts))
        Threads.@threads :static for i in 1:nrow(df)
            @inbounds begin
                site = getsite(soa, i)
                site.active = false
                site.rng = Random.Xoshiro(rand(rng, UInt64))
                site.ecocode = UIntType(df.eco_id[i])# no ecocode coming from raster, relying on eco_id
                site.eco_id = df.eco_id[i]
                site.mapcode = UIntType(df.plot_id[i]) # for parametrization, plot_id is global index, no raster
                site.ref_cn = UIntType(df.plot_id[i])
                site.old = zero(UIntType)
                site.live = zero(UIntType)
                site.B = zero(FloatType)
                site.AGNPP = zero(FloatType)
                site.capacityReduction = one(FloatType)
                site.growthReduction = one(FloatType)
                site.prevYearMortality = zero(FloatType)
                site.shade_class = one(UIntType)
                site.c_species .= zero(UIntType)
                site.c_age .= zero(FloatType)
                site.c_bio .= zero(FloatType)
                site.c_m_tot .= zero(FloatType)
                site.c_comp .= zero(FloatType)
                site.sp_mature .= false
                site.sp_sprout .= false
            end
        end
        return soa
    else
        splots_dict = Dict(
            (plt_key.plot_id, plt_key.eco_id) => begin
                plt_df = sort!(DataFrame(plt_df), :age_calc, rev=true)
                plt_df.sim_year .= Dates.value.(Dates.Day.(plt_df.measdate - plt_df.start_measdate)) ./ 365.25 .|> round .|> Int
                Data.get_initial_cohorts(plt_df)
            end
            for (plt_key, plt_df) in pairs(groupby(splots, [:plot_id, :eco_id], sort=false))
        )
        df_keys = sort!(collect(keys(splots_dict)))

        cohort_counts = [Int32(nrow(splots_dict[key])) for key in df_keys]
        soa = ActiveSoA((cohort=cohort_counts, species=species_counts))
        Threads.@threads :static for i in 1:length(df_keys)
            @inbounds begin
                key = df_keys[i]
                plt_df = splots_dict[key]
                (plot_id, eco_id) = key
                initial_cohorts = plt_df
                #n_cohorts = nrow(initial_cohorts)
                #cap = UIntType(2^ceil(log2(n_cohorts)))
                site = getsite(soa, i)
                site.active = true
                site.rng = Random.Xoshiro(rand(rng, UInt64))
                site.ecocode = UIntType(eco_id) #UIntType(getproperty(p, ecocode_field)),
                site.eco_id = eco_id
                site.mapcode = UIntType(plot_id)
                site.ref_cn = plot_id
                site.old = zero(UIntType)
                site.live = zero(UIntType)
                site.B = zero(FloatType)
                site.AGNPP = zero(FloatType)
                site.capacityReduction = one(FloatType)
                site.growthReduction = one(FloatType)
                site.prevYearMortality = zero(FloatType)
                site.shade_class = one(UIntType)
                site.c_species .= zero(UIntType)
                site.c_age .= zero(FloatType)
                site.c_bio .= zero(FloatType)
                site.c_m_tot .= zero(FloatType)
                site.c_comp .= zero(FloatType)
                site.sp_mature .= false
                site.sp_sprout .= false
                for row in eachrow(initial_cohorts)
                    BiomassSuccessionPlugin.add_cohort!(site, UIntType(row.eco_species_id), FloatType(row.age_calc), FloatType(row.agb_sum))
                end
            end
        end
        return soa
    end
end

struct WriterJob{State}
    is_new_best::Bool
    state::State
end

const STOP = :stop

function start_writer(::Type{State}, output_dir::AbstractString; buffer_size::Int=8) where {State}
    ch = Channel{Union{WriterJob{State},Symbol}}(buffer_size)

    task = Threads.@spawn begin
        try
            for job in ch
                job === STOP && break
                state = job.state
                if job.is_new_best
                    try
                        @info "New best: $(convert(Float64,state.best.fx))" iter = state.i
                        mkpath(output_dir)
                        JLD2.save_object(joinpath(output_dir, "search_state@$(state.i).jld2"), state)
                        PU.save_json(joinpath(output_dir, "best_params@$(state.i).json"), state.best.x)
                        JLD2.save_object(joinpath(output_dir, "best_params@$(state.i).jld2"), state.best.x)
                    catch e
                        @error "writer: save failed" iter = state.i exception = (e, catch_backtrace())
                    end
                else
                    @info ("Best@$(state.best_iteration): $(convert(Float64, state.best.fx)), Avg diff: $(state.diff_avg), Temp: $(state.t), ratio $(state.diff_avg/state.t), Prob: $(state.prob_avg)")
                end
            end
        catch e
            @error "writer: fatal" exception = (e, catch_backtrace())
            rethrow()
        end
    end

    return ch, task
end

function stop_writer(ch::Channel, task::Task)
    put!(ch, STOP)
    close(ch)
    wait(task)
end
function parametrize(; cohorts_db_path::String="../data_eco_l4_cohorts.db",
    tablename::String="data_eco_cohorts_g",
    output_dir::String="./outputs",
    filter_ecos::Vector{String}=String[],
    skip_disturbances::Bool=true,
    spinup::Bool=true,
    TRIALS::Int=30000,
    rng::Random.AbstractRNG)

    loss_params = PU.LossParams(
        age_bins=PU.AgeBins(
            bins_idx=[5, 10, 20, 40, 60, 80] .|> Int,
            last_bin_open=true
        ),
        smoothing_weights=PU.get_smoothing_window(; smoothing_window=1, smoothing_variance=FloatType(1.0f0))
    )
    println(Data)
    splots, eco_list, species_list, eco_species_ids = Data.prepare_parametrization_data(; cohorts_db_path=cohorts_db_path,
        tablename=tablename,
        output_dir=tablename,
        skip_disturbances=skip_disturbances,
        filter_ecos=filter_ecos,
        RNG=rng)
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
    #greet()
    #println(splots)
    #println(splots)
    max_age = maximum(splots.age_calc)
    #precomupte loss for missing entries
    println("Preprocessing plot results (smoothing and binning)")
    debug = (get(ENV, "JULIA_DEBUG", "") ∈ ["Pan", "all"])
    @time spdf = PU.smoothen_ref_years(splots, loss_params, max_age; debug=debug)
    @assert minimum(spdf.sim_year) == 0 "$(spdf.sim_year)"
    #show(spdf)
    println("Creating comparison years")
    @time spdf_plts = Data.make_spdf_dict(spdf, eco_species_ids)
    println("Marking sim years")
    @time site_sim_years = Data.get_site_sim_years(spdf)
    #println(site_sim_years)
    #return
    #show(site_sim_years)

    println("Marking spinup cohorts")
    @time spinup_cohorts = Data.get_spinup_cohorts(splots)
    #initial_cohorts = get_initial_cohorts(splots)
    println("beginning trials")
    #SITES_PER_RUN = Int(round(nrow(site_sim_years) * 0.33))
    #Profile.clear()
    #Profile.init(n=10^7, delay=0.001)
    #SITES_PER_RUN = Int(round(nrow(site_sim_years) * 0.33))



    ref_soa = make_sites(splots, eco_species_ids; rng=rng, spinup=spinup)
    parametrize_LBSA(; ref_soa = ref_soa,
                   output_dir = output_dir,
                   spdf_plts = spdf_plts,
                   spinup_cohorts = spinup_cohorts,
                   site_sim_years = site_sim_years,
                   species_list = species_list,
                   eco_list = eco_list,
                   eco_species_ids = eco_species_ids,
                   spinup = spinup,
                   TRIALS = TRIALS,
                   loss_params = loss_params,
                   rng = rng,
                   debug = debug)
end
function parametrize_LBSA(;ref_soa::ActiveSoA,output_dir::AbstractString, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool)
    n_species = length(species_list)
    param_dists = BSP.make_biomass_param_dists(length(species_list), length(eco_list), eco_species_ids)
    bio_params = BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng)
    best_result = PU.SiteLoss(FloatType[], FloatType[], FloatType(Inf), 1)
    cur = LBSA.LBSACandidate(bio_params, best_result)
    _best = cur
    search_state = LBSA.LBSAState(_best, cur, rng; max_iter=TRIALS)
    if TRIALS < 1
        return search_state #best_loss, best_result, best_params
    end
    max_sim_year = site_sim_years.sim_years .|> maximum |> maximum

    writer_ch, writer_task = start_writer(typeof(search_state), output_dir)

    try

        TProgress.@track for trial in 1:TRIALS
            soa = deepcopy(ref_soa)

            bio_params = PU.mutate_params(bio_params, param_dists; rng=rng)
            eco_params = BiomassSuccessionPlugin.generate_eco_params(bio_params)
            ctx = (BiomassSuccession=(eco_params=eco_params,),)
            years_results = Vector{PU.SiteLoss}(undef, max_sim_year + 1)
            if spinup
                soa = BiomassSuccessionPlugin.spinup_cohorts!(soa, spinup_cohorts, eco_params)
            end
            for current_sim_year in 0:max_sim_year
                #println("\ttimestep $(t)")
                sites_results = Vector{PU.SiteLoss}(undef, soa.n)
                PanCore.process_plugin!(soa, BiomassSuccessionPlugin.BiomassSuccession, current_sim_year; ctx=ctx.BiomassSuccession)
                Threads.@threads :static for i in 1:soa.n
                    @inbounds begin
                        site = getsite(soa, i)
                        !site.active && continue
                        spdf_plt = spdf_plts[site.ref_cn]
                        sim_years = site_sim_years.sim_years[site.mapcode]
                        if current_sim_year in sim_years
                            sloss = PU.calculate_site_loss2(current_sim_year, site, n_species, eco_species_ids, spdf_plt[current_sim_year], loss_params; debug=debug)
                            sites_results[i] = sloss
                        end
                    end
                end
                year_results_no_missing = collect(PU.skipundef(sites_results))
                if length(year_results_no_missing) > 0
                    current_year_results = sum(year_results_no_missing)
                    years_results[current_sim_year+1] = current_year_results
                end
            end
            run_result = sum(PU.skipundef(years_results))
            @assert !any(isnan.(run_result.sp_w_loss)) "run NaN"
            next = LBSA.LBSACandidate(bio_params, run_result)

            is_new_best = LBSA.search_cmp!(next, search_state)
            if is_new_best || search_state.i % 50 == 0
                put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state)))
            end
            if LBSA.is_search_over(search_state)
                break
            end
            #println(convert(Float64,run_result))
        end
    finally
        stop_writer(writer_ch, writer_task)
    end

end
function parametrize_SA(;ref_soa::ActiveSoA,output_dir::AbstractString, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool)
    n_species = length(species_list)
    param_dists = BSP.make_biomass_param_dists(length(species_list), length(eco_list), eco_species_ids)
    bio_params = BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng)
    best_result = PU.SiteLoss(FloatType[], FloatType[], FloatType(Inf), 1)
    cur = SA.SACandidate(bio_params, best_result)
    _best = cur
    search_state = SA.SAState(_best, cur, rng; max_iter=TRIALS, initial_t=1e3, t=1e3)
    if TRIALS < 1
        return search_state #best_loss, best_result, best_params
    end
    max_sim_year = site_sim_years.sim_years .|> maximum |> maximum

    writer_ch, writer_task = start_writer(typeof(search_state), output_dir)

    try

        TProgress.@track for trial in 1:TRIALS
            search_state.i = trial
            soa = deepcopy(ref_soa)

            bio_params = PU.mutate_params(bio_params, param_dists; rng=rng)
            eco_params = BiomassSuccessionPlugin.generate_eco_params(bio_params)
            ctx = (BiomassSuccession=(eco_params=eco_params,),)
            years_results = Vector{PU.SiteLoss}(undef, max_sim_year + 1)
            if spinup
                soa = BiomassSuccessionPlugin.spinup_cohorts!(soa, spinup_cohorts, eco_params)
            end
            for current_sim_year in 0:max_sim_year
                #println("\ttimestep $(t)")
                sites_results = Vector{PU.SiteLoss}(undef, soa.n)
                PanCore.process_plugin!(soa, BiomassSuccessionPlugin.BiomassSuccession, current_sim_year; ctx=ctx.BiomassSuccession)
                Threads.@threads :static for i in 1:soa.n
                    @inbounds begin
                        site = getsite(soa, i)
                        !site.active && continue
                        spdf_plt = spdf_plts[site.ref_cn]
                        sim_years = site_sim_years.sim_years[site.mapcode]
                        if current_sim_year in sim_years
                            sloss = PU.calculate_site_loss2(current_sim_year, site, n_species, eco_species_ids, spdf_plt[current_sim_year], loss_params; debug=debug)
                            sites_results[i] = sloss
                        end
                    end
                end
                year_results_no_missing = collect(PU.skipundef(sites_results))
                if length(year_results_no_missing) > 0
                    current_year_results = sum(year_results_no_missing)
                    years_results[current_sim_year+1] = current_year_results
                end
            end
            run_result = sum(PU.skipundef(years_results))
            @assert !any(isnan.(run_result.sp_w_loss)) "run NaN"
            next = SA.SACandidate(bio_params, run_result)

            is_new_best = SA.search_cmp!(next, search_state)
            if is_new_best || search_state.i % 50 == 0
                put!(writer_ch, WriterJob(is_new_best, deepcopy(search_state)))
            end
            if SA.search_update_rule!(search_state)
                break
            end
            #println(convert(Float64,run_result))
        end
    finally
        stop_writer(writer_ch, writer_task)
    end

end
function main()
    seed=123
    Random.seed!(seed)
    rng = Random.Xoshiro(rand(UInt64))
    filter_ecos=String["8.5.3.75e", "8.5.3.75f", "8.5.3.75a", "8.5.3.75c", "8.5.3.75g", "8.3.5.65o", "8.5.3.75d", "8.5.3.75h", "8.3.5.65h", "8.3.5.65f", "8.3.5.65g", "15.4.1.76b", "8.5.3.75b", "8.5.3.75i", "9.4.7.32b", "8.3.7.35b", "8.3.7.35e", "8.5.1.63h", "8.3.7.35g", "8.3.7.35f", "8.3.5.65l", "8.3.5.65c", "9.5.1.34a"]
    #filter_ecos=String["8.3.5.65o", "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"]
    #filter_ecos=String["8.3.5.65o", "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"]
    #filter_ecos=["8.5.3.75g"]
    parametrize(;
     cohorts_db_path="../data_eco_l4_cohorts.db",
    tablename="data_eco_cohorts",
    output_dir="./outputs",
    filter_ecos=filter_ecos,
    skip_disturbances=false,
    spinup=false,
    TRIALS=1000000, rng=rng)
end

function julia_main()::Cint
	main()
	return 0
end
end
