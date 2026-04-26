using ..PanCore
using DataFrames
import ArchGDAL
import DuckDB
import CSV
import StatsBase
import Dates

const AG = ArchGDAL
const AttrDict = Dict{String,Any}
const AttrTable = Dict{Int64,AttrDict}

export load_eco_raster, load_treemap_raster, load_treemap_cohorts,
       map_params_to_data_treemap, expand_to_pixels,
       load_csv_communities, load_duckdb_communities, prepare_general_splots

function load_eco_raster(path::String)::Matrix{Int16}
    ds = AG.readraster(path)
    band = AG.getband(ds, 1)
    A = AG.read(band)
    AG.destroy(ds)
    return A
end

function load_treemap_raster(path::String; treemap_version::Int=2022)
    CN_FIELD_NAME = treemap_version == 2016 ? "CN" : "PLT_CN"
    AG.readraster(path) do ds
        band = AG.getband(ds, 1)
        rat  = AG.getdefaultRAT(band)
        nrows  = AG.nrow(rat)
        ncols  = AG.ncolumn(rat)
        colnames = [AG.columnname(rat, c) for c in 0:ncols-1]

        valcol = findfirst(==("Value"), colnames)
        isnothing(valcol) && error("No 'Value' column in RAT of $path")
        valcol -= 1  # GDAL is 0-based

        function rat_get(r, c)
            t = AG.columntype(rat, c)
            if t == AG.GFT_Integer
                return AG.asint(rat, r, c)
            elseif t == AG.GFT_Real
                return AG.asdouble(rat, r, c)
            else
                return AG.asstring(rat, r, c)
            end
        end

        vat = AttrTable()
        for r in 0:nrows-1
            px = AG.pixeltype(band)(rat_get(r, valcol))
            attrs = AttrDict()
            for c in 0:ncols-1
                attrs[colnames[c+1]] = rat_get(r, c)
            end
            vat[Int64(px)] = attrs
        end

        A = AG.read(band)
        NO_DATA = AG.getnodatavalue(band)
        out = Array{Union{Missing,Int64}}(missing, size(A))
        Threads.@threads :static for i in eachindex(A, out)
            @inbounds cell = A[i]
            if !ismissing(cell) && cell != NO_DATA
                plt_attr = get(vat, Int64(cell), missing)
                if !ismissing(plt_attr)
                    plt_cn = get(plt_attr, CN_FIELD_NAME, missing)
                    !ismissing(plt_cn) && (@inbounds out[i] = Int64(plt_cn))
                end
            end
        end
        return out, vat
    end
end

function load_treemap_cohorts(
    cn_raster::Array{Union{Missing,Int64}},
    eco_raster::Matrix{Int16},
    db_path::String,
    eco_ecocode_mapping_csv::String,
)
    eco_ecocode_df = CSV.read(eco_ecocode_mapping_csv, DataFrame)

    cn_eco_counts = StatsBase.countmap(
        (cn, Int64(eco))
        for (cn, eco) in zip(cn_raster, eco_raster)
        if !ismissing(cn)
    )
    cn_eco_df = DataFrame(
        CN      = Int64[k[1] for k in keys(cn_eco_counts)],
        ecocode = Int64[k[2] for k in keys(cn_eco_counts)],
        count   = collect(values(cn_eco_counts)),
    )

    db  = DuckDB.DB(db_path)
    con = DuckDB.connect(db)
    DuckDB.register_data_frame(con, cn_eco_df,     "cn_eco")
    DuckDB.register_data_frame(con, eco_ecocode_df, "eco_ecocode_map")

    sql = """
        WITH all_species AS (
            SELECT DISTINCT species_symbol_map FROM data_eco_cohorts
        ),
        full_table AS (
            SELECT
                df.CN,
                df.ecocode                                          AS raster_ecocode,
                e.eco                                               AS raster_eco,
                o.*,
                (COALESCE(m.species_symbol_map, b.species_symbol_map) IS NULL) AS borrowed,
                CASE WHEN COALESCE(m.species_symbol_map, b.species_symbol_map) IS NULL
                     THEN o.eco   ELSE e.eco   END                 AS effective_eco,
                CASE WHEN COALESCE(m.species_symbol_map, b.species_symbol_map) IS NULL
                     THEN eo.ecocode ELSE e.ecocode END            AS effective_ecocode,
                CASE WHEN COALESCE(m.species_symbol_map, b.species_symbol_map) IS NULL
                     THEN o.species_symbol_map
                     ELSE COALESCE(m.species_symbol_map, b.species_symbol_map) END
                                                                    AS effective_species_symbol_map
            FROM cn_eco df
            JOIN eco_ecocode_map  e  ON df.ecocode = e.ecocode
            JOIN data_eco_cohorts o  ON df.CN = o.PLT_CN
            JOIN eco_ecocode_map  eo ON o.eco = eo.eco
            LEFT OUTER JOIN data_species_eco_map m
                ON m.eco = e.eco AND m.species_symbol = o.species_symbol
            LEFT OUTER JOIN all_species b
                ON b.species_symbol_map = e.eco || '_' || o.sftwd_hrdwd
        ),
        group_totals AS (
            SELECT raster_ecocode, effective_eco, effective_ecocode,
                   effective_species_symbol_map,
                   SUM(tree_count) AS group_count
            FROM full_table
            GROUP BY raster_ecocode, effective_species_symbol_map,
                     effective_eco, effective_ecocode
        ),
        dominant AS (
            SELECT raster_ecocode, effective_eco, effective_ecocode,
                   effective_species_symbol_map
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY raster_ecocode, effective_species_symbol_map
                    ORDER BY group_count DESC
                ) AS rn
                FROM group_totals
            ) WHERE rn = 1
        )
        SELECT d.effective_eco, d.effective_ecocode,
               d.effective_species_symbol_map, t.*
        FROM full_table t
        JOIN dominant d
            ON  t.raster_ecocode            = d.raster_ecocode
            AND t.effective_species_symbol_map = d.effective_species_symbol_map
    """

    df = DuckDB.execute(con, sql) |> DataFrame
    DuckDB.close(con)
    return _make_effective_splots(df)
end

function _make_effective_splots(df::DataFrame)
    eco_vals = sort(unique(df.raster_eco))
    eco_dict = Dict(eco => i for (i, eco) in enumerate(eco_vals))

    eff_eco_vals = sort(unique(df.effective_eco))
    eff_eco_dict = Dict(eco => i for (i, eco) in enumerate(eff_eco_vals))

    eff_sp_vals = sort(unique(df.effective_species_symbol_map))
    eff_sp_dict = Dict(ssm => i for (i, ssm) in enumerate(eff_sp_vals))

    df.species_id      = getindex.(Ref(eff_sp_dict),  df.effective_species_symbol_map)
    df.eco_id          = getindex.(Ref(eco_dict),      df.raster_eco)
    df.effective_eco_id = getindex.(Ref(eff_eco_dict), df.effective_eco)

    if !(eltype(df.measdate) <: Dates.TimeType)
        df.measdate = Dates.DateTime.(df.measdate, Dates.dateformat"yyyy-mm-dd")
    end

    fields = [:plt_cn, :statecd, :unitcd, :countycd, :plot,
              :raster_ecocode, :eco_id, :effective_eco_id, :measdate,
              :species_id, :age_calc]
    plots = combine(groupby(df, fields, sort=false),
                    nrow => :count, :agb => sum => :agb_sum)
    start_measdates = combine(
        groupby(plots, [:statecd, :unitcd, :countycd, :plot], sort=false)
    ) do rows
        (; start_measdate=[minimum(rows.measdate)])
    end
    plots_md = innerjoin(plots, start_measdates,
                         on=[:statecd, :unitcd, :countycd, :plot])
    splots = sort!(plots_md,
                   [:measdate, :statecd, :unitcd, :countycd, :plot,
                    :age_calc, :species_id])
    splots.plot_id .= groupindices(
        groupby(splots, [:statecd, :unitcd, :countycd, :plot, :raster_ecocode])
    ) .|> UIntType

    return splots, eco_vals, eff_eco_vals, eff_sp_vals
end

function map_params_to_data_treemap(params, eco_list, effective_eco_list, species_list, splots)
    params_species_df = DataFrame(
        param_species    = params.SPECIES_LIST,
        param_species_id = 1:length(params.SPECIES_LIST),
    )
    data_species_df = DataFrame(
        species         = species_list,
        data_species_id = 1:length(species_list),
    )
    joint_species_df = innerjoin(data_species_df, params_species_df,
                                 on=:species => :param_species)

    params_eco_df = DataFrame(
        param_eco    = params.ECO_LIST,
        param_eco_id = 1:length(params.ECO_LIST),
    )
    data_eco_df = DataFrame(
        eco    = eco_list,
        eco_id = 1:length(eco_list),
    )
    data_eff_eco_df = DataFrame(
        effective_eco    = effective_eco_list,
        effective_eco_id = 1:length(effective_eco_list),
    )

    joint_eco_df     = innerjoin(data_eco_df,     params_eco_df,
                                 on=:eco => :param_eco)
    joint_eff_eco_df = innerjoin(data_eff_eco_df, params_eco_df,
                                 on=:effective_eco => :param_eco)

    param_eco_species_df = DataFrame(
        [(e, i, s)
         for (e, ss) in enumerate(params.ECO_SPECIES_IDS)
         for (i, s) in enumerate(ss)],
        [:param_eco_id, :param_eco_species_id, :param_species_id],
    )

    eco_species_id_fields = [:eco_id, :effective_eco_id, :species_id,
                             :param_eco_id, :param_eco_species_id]

    mapped = innerjoin(
        innerjoin(
            innerjoin(splots, joint_species_df,    on=:species_id => :data_species_id),
            joint_eff_eco_df,                      on=:effective_eco_id,
        ),
        param_eco_species_df, on=[:param_eco_id, :param_species_id],
    )

    eco_sp_id_df = select(mapped, eco_species_id_fields) |> unique |> sort
    eco_sp_id_df = combine(groupby(eco_sp_id_df, :eco_id, sort=true)) do rows
        (; effective_eco_id      = rows.effective_eco_id,
           species_id            = rows.species_id,
           param_eco_id          = rows.param_eco_id,
           param_eco_species_id  = rows.param_eco_species_id,
           eco_species_id        = 1:nrow(rows))
    end
    mapped = innerjoin(mapped, eco_sp_id_df, on=eco_species_id_fields)

    eco_param_sp_df = combine(groupby(mapped, :eco_id, sort=true)) do rows
        (; selector=[sort(unique(zip(
            rows.species_id, rows.param_eco_id, rows.param_eco_species_id
        )))])
    end

    eco_species_ids = [[s for (s, _pe, _pes) in row.selector]
                       for row in eachrow(eco_param_sp_df)]

    function slice_eco_sp(param)
        [[getproperty(params, param)[pe][pes]
          for (_s, pe, pes) in row.selector]
         for row in eachrow(eco_param_sp_df)]
    end

    @assert joint_eco_df.eco == eco_list[joint_eco_df.eco_id] "Eco ordering mismatch after join. Not all raster ecos are present in params."

    mod_params = typeof(params)(
        SPINUP_MORTALITY_FRACTION = params.SPINUP_MORTALITY_FRACTION,
        SUFFICIENT_LIGHT          = params.SUFFICIENT_LIGHT,
        ECO_LIST                  = joint_eco_df.eco,
        SPECIES_LIST              = joint_species_df.species,
        ECO_SPECIES_IDS           = eco_species_ids,
        MIN_REL_BIOMASS           = params.MIN_REL_BIOMASS[joint_eco_df.param_eco_id],
        D                         = params.D[joint_species_df.param_species_id],
        S                         = params.S[joint_species_df.param_species_id],
        LONGEVITY                 = params.LONGEVITY[joint_species_df.param_species_id],
        SHADE_TOL                 = params.SHADE_TOL[joint_species_df.param_species_id],
        MATURITY                  = params.MATURITY[joint_species_df.param_species_id],
        B_MAX_SPP                 = slice_eco_sp(:B_MAX_SPP),
        ANPP_MAX_SPP              = slice_eco_sp(:ANPP_MAX_SPP),
        PROB_MORT_SPP             = slice_eco_sp(:PROB_MORT_SPP),
        PROB_ESTAB_SPP            = slice_eco_sp(:PROB_ESTAB_SPP),
    )

    return mod_params, mapped, eco_species_ids
end

function expand_to_pixels(
    mapped_splots::DataFrame,
    cn_raster::Array{Union{Missing,Int64}},
    eco_raster::Matrix{Int16},
)
    cohorts_dict = Dict(
        (Int64(key.plt_cn), Int64(key.raster_ecocode)) => (
            n              = nrow(rows),
            eco_id         = rows.eco_id,
            eco_species_id = rows.eco_species_id,
            age_calc       = rows.age_calc,
            agb_sum        = rows.agb_sum,
        )
        for (key, rows) in pairs(groupby(mapped_splots,
                                         [:plt_cn, :raster_ecocode], sort=false))
    )

    mapcodes       = Int[]
    eco_ids        = Int[]
    eco_sp_ids     = Int[]
    ages           = FloatType[]
    agbs           = FloatType[]

    for i in eachindex(cn_raster)
        plt_cn = cn_raster[i]
        ismissing(plt_cn) && continue
        eco = Int64(eco_raster[i])
        c = get(cohorts_dict, (plt_cn, eco), nothing)
        isnothing(c) && continue
        append!(mapcodes,   fill(i,   c.n))
        append!(eco_ids,    c.eco_id)
        append!(eco_sp_ids, c.eco_species_id)
        append!(ages,       FloatType.(c.age_calc))
        append!(agbs,       FloatType.(c.agb_sum))
    end

    return DataFrame(
        mapcode       = mapcodes,
        eco_id        = eco_ids,
        eco_species_id = eco_sp_ids,
        age_calc      = ages,
        agb_sum       = agbs,
    )
end

function load_csv_communities(path::String)::DataFrame
    CSV.read(path, DataFrame)
end

function load_duckdb_communities(db_path::String; tablename::String="communities")::DataFrame
    con = DuckDB.connect(DuckDB.DB(db_path))
    df  = DuckDB.execute(con, "SELECT * FROM $(tablename)") |> DataFrame
    DuckDB.close(con)
    return df
end

function prepare_general_splots(
    ic_df::DataFrame,
    eco_raster::Matrix{Int16},
    eco_ecocode_df::DataFrame,
    params,
)
    ecocode_to_eco  = Dict(row.ecocode => row.eco for row in eachrow(eco_ecocode_df))
    eco_to_param_id = Dict(eco => i for (i, eco) in enumerate(params.ECO_LIST))
    sp_to_param_id  = Dict(sp  => i for (i, sp)  in enumerate(params.SPECIES_LIST))

    eco_sp_id_map = Dict(
        (eco_id, sp_id) => local_idx
        for (eco_id, species_ids) in enumerate(params.ECO_SPECIES_IDS)
        for (local_idx, sp_id) in enumerate(species_ids)
    )

    ecocodes  = [Int64(eco_raster[row.mapcode]) for row in eachrow(ic_df)]
    eco_ids   = [get(eco_to_param_id,  get(ecocode_to_eco, ec, ""), 0) for ec in ecocodes]
    param_sp_ids = [get(sp_to_param_id, row.species, 0) for row in eachrow(ic_df)]
    eco_sp_ids   = [get(eco_sp_id_map, (eid, sid), 0)
                    for (eid, sid) in zip(eco_ids, param_sp_ids)]

    valid = (eco_ids .> 0) .& (param_sp_ids .> 0) .& (eco_sp_ids .> 0)

    age_calcs = hasproperty(ic_df, :age) ?
        FloatType.(ic_df.age[valid]) :
        zeros(FloatType, sum(valid))

    splots = DataFrame(
        mapcode        = ic_df.mapcode[valid],
        eco_id         = eco_ids[valid],
        eco_species_id = eco_sp_ids[valid],
        age_calc       = age_calcs,
        agb_sum        = FloatType.(ic_df.biomass[valid]),
    )

    return splots, params, Vector{Vector{Int}}(params.ECO_SPECIES_IDS)
end
