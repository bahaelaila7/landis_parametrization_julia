using ..PanCore
export prepare_parametrization_data, get_site_sim_years, get_spinup_cohorts, make_spdf_dict, get_initial_cohorts
import SQLite
import Random
using DataFrames
import Dates
import ....Parametrization.SPDFGroundTruth
import ....Parametrization.SPDFRecord


function make_spdf_dict(spdf::DataFrame, eco_species_ids::Vector{Vector{Int}})::Dict{Tuple{UIntType,UIntType},Dict{Int,SPDFGroundTruth}}
    #n_species = length(unique(spdf.species_id))
    return Dict( #{Int, Dict{Int,DataFrame}}()
        (plt_key.plot_id, plt_key.eco_id) => Dict( #( {Int, DataFrame}(
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
        for (plt_key, plt_df) in pairs(groupby(spdf, [:plot_id, :eco_id], sort=false))
    )
end
function get_initial_cohorts(df::DataFrame)
    #return df[df.year_deficit.==0, :]
    return df[df.sim_year .== 0, :]
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
    #last = maximum(df.start_measdate)
    #println("Oldest cohort established: $oldest")
    #println("First measurement date: $last")
    #df.year_deficit .= Dates.value.(Dates.Day.(year_estab .- last)) ./ 365.25 .|> round .|> Int
    df.year_deficit .= Dates.value.(Dates.Day.(year_estab .- df.start_measdate)) ./ 365.25 .|> round .|> Int
end
function make_splots(df::DataFrame; eco::String="epa_l4")::Tuple{DataFrame,Vector{String},Vector{String},Vector{Vector{Int}}}
    eco_field, species_field = begin
                        if eco == "epa_l4"
                            (:epa_l4, :species_symbol_map_l4)
                        elseif eco == "epa_l3"
                            (:epa_l3, :species_symbol_map_l3)
                        else
                            (:ecosubcd, :species_symbol_map_ecosubcd)
                        end
    end


    #df.species_id = groupindices(groupby(df,:species_field))
    #df.eco_id = groupindices(groupby(df,:eco))

    #println(df)

    eco_vals = sort(unique(getproperty(df, eco_field)))
    eco_dict = Dict(eco => i for (i, eco) in enumerate(eco_vals))
    species_symbol_map_vals = sort(unique(getproperty(df,species_field)))
    species_symbol_map_dict = Dict(ssm => i for (i, ssm) in enumerate(species_symbol_map_vals))

    df.species_id = getindex.(Ref(species_symbol_map_dict), getproperty(df,species_field))
    df.eco_id = getindex.(Ref(eco_dict), getproperty(df,eco_field))
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
function get_spinup_cohorts(df::DataFrame)
    spinup_cohorts = df[df.year_deficit.<-1, :]
    spinup_cohorts = unique(select(spinup_cohorts, [:year_deficit, :plot_id, :eco_id, :eco_species_id ]))
    spinup_cohorts = sort!(spinup_cohorts, [:year_deficit, :plot_id, :eco_id, :eco_species_id ])
    return spinup_cohorts
end

function prepare_parametrization_data(; cohorts_db_path::String, eco_field::String, tablename::String, output_dir::String, skip_disturbances=true, filter_ecos::Vector{String}=String[], RNG::Union{Nothing,Random.AbstractRNG})
        #cohorts_df = load_cohorts_sqlite(db_path, tablename; filter_ecos=filter_ecos)
    println("Connecting to: $(cohorts_db_path) ")
    db = SQLite.DB(cohorts_db_path)
    println("Creating index if necessary")
    SQLite.execute(db, "CREATE INDEX IF NOT EXISTS PLT_ECO_IDX_$(eco_field) ON data_eco_cohorts($(eco_field));")
    sql = "SELECT * FROM $(tablename) WHERE true"
    if length(filter_ecos) > 0
        sql *= " AND $(eco_field) in ('$(join(filter_ecos,"','"))')"
    end
    if skip_disturbances
        sql *= " AND subp_has_dstrb= false"
    end
    println(sql)
    cohorts_df = SQLite.DBInterface.execute(db, sql) |> DataFrame
    SQLite.close(db)
    println("Closing db. $(nrow(cohorts_df)) rows loaded.")

    @time splots, eco_list, species_list, eco_species_ids = make_splots(cohorts_df, eco =eco_field)
    mark_estab_year!(splots)


    return splots, eco_list, species_list, eco_species_ids

end
