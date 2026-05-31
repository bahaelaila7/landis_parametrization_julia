using ..PanCore
using ..Plugins: BiomassSuccessionPlugin
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
  map_params_to_data_treemap, expand_to_pixels, deduplicate_for_export,
  load_csv_communities, load_duckdb_communities, prepare_general_splots,
  load_landis_mapcode_raster, load_landis_core_species,
  load_landis_spp_ecoregion, make_landis_params, expand_landis_pixels

function load_eco_raster(path::String)::Matrix{Int16}
  isfile(path) || error("Raster not found: $path")
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
    rat = AG.getdefaultRAT(band)
    nrows = AG.nrow(rat)
    ncols = AG.ncolumn(rat)
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
  params;
  eco_field::String="epa_l4",
)
  species_field = if eco_field == "epa_l4"
    "species_symbol_map_l4"
  elseif eco_field == "epa_l3"
    "species_symbol_map_l3"
  else
    "species_symbol_map_ecosubcd"
  end

  eco_ecocode_df = CSV.read(eco_ecocode_mapping_csv, DataFrame)

  cn_eco_counts = StatsBase.countmap(
    (cn, Int64(eco))
    for (cn, eco) in zip(cn_raster, eco_raster)
    if !ismissing(cn)
  )
  cn_eco_df = DataFrame(
    CN=Int64[k[1] for k in keys(cn_eco_counts)],
    ecocode=Int64[k[2] for k in keys(cn_eco_counts)],
    count=collect(values(cn_eco_counts)),
  )

  # Flatten params into a (eco, species) table registered in DuckDB so the
  # 4-case matching can be expressed as declarative SQL joins.
  params_eco_sp_df = DataFrame(
    eco=[params.ECO_LIST[eid]
         for eid in eachindex(params.ECO_LIST)
         for _ in params.ECO_SPECIES_IDS[eid]],
    species=[params.SPECIES_LIST[Int(sid)]
             for eid in eachindex(params.ECO_LIST)
             for sid in params.ECO_SPECIES_IDS[eid]],
  )

  db = DuckDB.DB(db_path)
  con = DuckDB.connect(db)
  DuckDB.register_data_frame(con, cn_eco_df, "cn_eco")
  DuckDB.register_data_frame(con, eco_ecocode_df, "eco_ecocode_map")
  DuckDB.register_data_frame(con, params_eco_sp_df, "params_eco_species")

  # Create a reusable view that resolves each cohort row to its effective
  # species/eco using the 4-case priority:
  #   1. species exists in target eco params           → (species,     target_eco, borrowed=false)
  #   2. softwood/hardwood catchall in target eco      → (eco_S/H,    target_eco, borrowed=false)
  #   3a. species exists in original eco params        → (species,    orig_eco,   borrowed=true)
  #   3b. catchall in original eco params              → (orig_eco_SH, orig_eco,  borrowed=true)
  #   4. none of the above                             → NULL (dropped)
  DuckDB.execute(
    con,
    """
    CREATE OR REPLACE TEMP VIEW all_cohorts_matched AS
    SELECT
        df.CN,
        df.ecocode                          AS raster_ecocode,
        e.eco                               AS raster_eco,
        o.plt_cn, o.statecd, o.unitcd, o.countycd, o.plot, o.subp,
        o.$(eco_field)                      AS original_eco,
        o.$(species_field)                  AS species_symbol_map,
        o.sftwd_hrdwd,
        o.agb, o.tree_count, o.measdate, o.age_calc,
        -- case 1: exact species in target eco
        COALESCE(p1.species, p2.species, p3a.species, p3b.species)
                                            AS effective_species_symbol_map,
        CASE
            WHEN p1.species IS NOT NULL OR p2.species IS NOT NULL THEN e.eco
            WHEN p3a.species IS NOT NULL OR p3b.species IS NOT NULL THEN o.$(eco_field)
        END                                 AS effective_eco,
        CASE
            WHEN p1.species IS NOT NULL OR p2.species IS NOT NULL THEN FALSE
            WHEN p3a.species IS NOT NULL OR p3b.species IS NOT NULL THEN TRUE
        END                                 AS borrowed
    FROM cn_eco df
    JOIN eco_ecocode_map  e  ON df.ecocode = e.ecocode
    JOIN data_eco_cohorts o  ON df.CN      = o.PLT_CN
    -- case 1
    LEFT JOIN params_eco_species p1
        ON p1.eco = e.eco AND p1.species = o.$(species_field)
    -- case 2: softwood/hardwood catchall in target eco
    LEFT JOIN params_eco_species p2
        ON p2.eco = e.eco AND p2.species = e.eco || '_' || o.sftwd_hrdwd
    -- case 3a: exact species in original eco
    LEFT JOIN params_eco_species p3a
        ON p3a.eco = o.$(eco_field) AND p3a.species = o.$(species_field)
    -- case 3b: catchall in original eco
    LEFT JOIN params_eco_species p3b
        ON p3b.eco = o.$(eco_field)
       AND p3b.species = o.$(eco_field) || '_' || o.sftwd_hrdwd
"""
  )

  # Print a concise summary of matching outcomes.
  stats = DuckDB.execute(
    con,
    """
    SELECT
        SUM(tree_count)                                                         AS total_trees,
        SUM(agb)                                                                AS total_agb,
        SUM(CASE WHEN effective_species_symbol_map IS NULL THEN tree_count ELSE 0 END)
                                                                                AS dropped_trees,
        SUM(CASE WHEN effective_species_symbol_map IS NULL THEN agb     ELSE 0 END)
                                                                                AS dropped_agb,
        SUM(CASE WHEN borrowed = TRUE THEN tree_count ELSE 0 END)               AS borrowed_trees,
        SUM(CASE WHEN borrowed = TRUE THEN agb        ELSE 0 END)               AS borrowed_agb
    FROM all_cohorts_matched
"""
  ) |> DataFrame
  s = first(stats)
  tot_t, tot_a = s.total_trees, s.total_agb
  if s.dropped_trees > 0
    pt = round(100 * s.dropped_trees / tot_t, digits=1)
    pa = round(100 * s.dropped_agb / tot_a, digits=1)
    println("  Dropped (case 4): $(s.dropped_trees) trees ($pt%), $pa% AGB")
    top_drop = DuckDB.execute(
      con,
      """
    SELECT original_eco, species_symbol_map,
           SUM(tree_count) AS trees, SUM(agb) AS agb_sum
    FROM all_cohorts_matched
    WHERE effective_species_symbol_map IS NULL
    GROUP BY original_eco, species_symbol_map
    ORDER BY trees DESC LIMIT 5
"""
    ) |> DataFrame
    for r in eachrow(top_drop)
      println("    $(r.original_eco)/$(r.species_symbol_map): $(r.trees) trees")
    end
  end
  if s.borrowed_trees > 0
    pt = round(100 * s.borrowed_trees / tot_t, digits=1)
    pa = round(100 * s.borrowed_agb / tot_a, digits=1)
    println("  Borrowed (case 3): $(s.borrowed_trees) trees ($pt%), $pa% AGB")
    top_borr = DuckDB.execute(
      con,
      """
    SELECT original_eco, species_symbol_map,
           effective_eco, effective_species_symbol_map,
           SUM(tree_count) AS trees
    FROM all_cohorts_matched
    WHERE borrowed = TRUE
    GROUP BY original_eco, species_symbol_map,
             effective_eco, effective_species_symbol_map
    ORDER BY trees DESC LIMIT 5
"""
    ) |> DataFrame
    for r in eachrow(top_borr)
      println("    $(r.original_eco)/$(r.species_symbol_map) → $(r.effective_eco)/$(r.effective_species_symbol_map): $(r.trees) trees")
    end
  end

  # Select only matched rows, resolving ties in effective_eco by picking the
  # one with the highest total tree_count per (raster_ecocode, effective_species).
  df = DuckDB.execute(
    con,
    """
    WITH group_totals AS (
        SELECT raster_ecocode, effective_eco, effective_species_symbol_map,
               SUM(tree_count) AS group_count
        FROM all_cohorts_matched
        WHERE effective_species_symbol_map IS NOT NULL
        GROUP BY raster_ecocode, effective_species_symbol_map, effective_eco
    ),
    dominant AS (
        SELECT raster_ecocode, effective_eco, effective_species_symbol_map
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY raster_ecocode, effective_species_symbol_map
                ORDER BY group_count DESC
            ) AS rn
            FROM group_totals
        ) WHERE rn = 1
    )
    SELECT m.*
    FROM all_cohorts_matched m
    JOIN dominant d
        ON  m.raster_ecocode               = d.raster_ecocode
        AND m.effective_species_symbol_map  = d.effective_species_symbol_map
        AND m.effective_eco                 = d.effective_eco
    WHERE m.effective_species_symbol_map IS NOT NULL
"""
  ) |> DataFrame

  #DuckDB.close(con)
  return _make_effective_splots(df)
end

function _make_effective_splots(df::DataFrame)
  eco_vals = sort(unique(df.raster_eco))
  eco_dict = Dict(eco => i for (i, eco) in enumerate(eco_vals))

  eff_eco_vals = sort(unique(df.effective_eco))
  eff_eco_dict = Dict(eco => i for (i, eco) in enumerate(eff_eco_vals))

  eff_sp_vals = sort(unique(df.effective_species_symbol_map))
  eff_sp_dict = Dict(ssm => i for (i, ssm) in enumerate(eff_sp_vals))

  df.species_id = getindex.(Ref(eff_sp_dict), df.effective_species_symbol_map)
  df.eco_id = getindex.(Ref(eco_dict), df.raster_eco)
  df.effective_eco_id = getindex.(Ref(eff_eco_dict), df.effective_eco)

  if !(eltype(df.measdate) <: Dates.TimeType)
    df.measdate = Dates.DateTime.(df.measdate, Dates.dateformat"yyyy-mm-dd")
  end

  subp_counts = combine(groupby(df, :plt_cn), :subp => (x -> length(unique(x))) => :subp_count)
  fields = [:plt_cn, :statecd, :unitcd, :countycd, :plot,
    :raster_ecocode, :eco_id, :effective_eco_id, :measdate,
    :species_id, :age_calc]
  raw_plots = combine(groupby(df, fields, sort=false),
    nrow => :count, :agb => sum => :agb_sum)
  plots = leftjoin(raw_plots, subp_counts, on=:plt_cn)
  plots.agb_sum ./= plots.subp_count
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
    param_species=params.SPECIES_LIST,
    param_species_id=1:length(params.SPECIES_LIST),
  )
  data_species_df = DataFrame(
    species=species_list,
    data_species_id=1:length(species_list),
  )
  joint_species_df = innerjoin(data_species_df, params_species_df,
    on=:species => :param_species)

  params_eco_df = DataFrame(
    param_eco=params.ECO_LIST,
    param_eco_id=1:length(params.ECO_LIST),
  )
  data_eco_df = DataFrame(
    eco=eco_list,
    eco_id=1:length(eco_list),
  )
  data_eff_eco_df = DataFrame(
    effective_eco=effective_eco_list,
    effective_eco_id=1:length(effective_eco_list),
  )

  joint_eco_df = innerjoin(data_eco_df, params_eco_df,
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
      innerjoin(splots, joint_species_df, on=:species_id => :data_species_id),
      joint_eff_eco_df, on=:effective_eco_id,
    ),
    param_eco_species_df, on=[:param_eco_id, :param_species_id],
  )

  eco_sp_id_df = select(mapped, eco_species_id_fields) |> unique |> sort
  eco_sp_id_df = combine(groupby(eco_sp_id_df, :eco_id, sort=true)) do rows
    (; effective_eco_id=rows.effective_eco_id,
      species_id=rows.species_id,
      param_eco_id=rows.param_eco_id,
      param_eco_species_id=rows.param_eco_species_id,
      eco_species_id=1:nrow(rows))
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
    SPINUP_MORTALITY_FRACTION=params.SPINUP_MORTALITY_FRACTION,
    SUFFICIENT_LIGHT=params.SUFFICIENT_LIGHT,
    ECO_LIST=joint_eco_df.eco,
    SPECIES_LIST=joint_species_df.species,
    ECO_SPECIES_IDS=eco_species_ids,
    MIN_REL_BIOMASS=params.MIN_REL_BIOMASS[joint_eco_df.param_eco_id],
    D=params.D[joint_species_df.param_species_id],
    S=params.S[joint_species_df.param_species_id],
    LONGEVITY=params.LONGEVITY[joint_species_df.param_species_id],
    SHADE_TOL=params.SHADE_TOL[joint_species_df.param_species_id],
    MATURITY=params.MATURITY[joint_species_df.param_species_id],
    PROB_RESPROUT=params.PROB_RESPROUT[joint_species_df.param_species_id],
    B_MAX_SPP=slice_eco_sp(:B_MAX_SPP),
    ANPP_MAX_SPP=slice_eco_sp(:ANPP_MAX_SPP),
    PROB_MORT_SPP=slice_eco_sp(:PROB_MORT_SPP),
    PROB_ESTAB_SPP=slice_eco_sp(:PROB_ESTAB_SPP),
  )

  return mod_params, mapped, eco_species_ids
end

function expand_to_pixels(
  mapped_splots::DataFrame,
  cn_raster::Array{Union{Missing,Int64}},
  eco_raster::Matrix{Int16},
)
  cohorts_dict = Dict(
    (key.plt_cn, Int64(key.raster_ecocode)) => (
      n=nrow(rows),
      eco_id=rows.eco_id,
      eco_species_id=rows.eco_species_id,
      age_calc=rows.age_calc,
      agb_sum=rows.agb_sum,
    )
    for (key, rows) in pairs(groupby(mapped_splots,
      [:plt_cn, :raster_ecocode], sort=false))
  )

  mapcodes = Int[]
  eco_ids = Int[]
  eco_sp_ids = Int[]
  ages = FloatType[]
  agbs = FloatType[]

  for i in eachindex(cn_raster)
    plt_cn = cn_raster[i]
    ismissing(plt_cn) && continue
    eco = Int64(eco_raster[i])
    c = get(cohorts_dict, ("$(plt_cn)", eco), nothing)
    isnothing(c) && continue
    append!(mapcodes, fill(i, c.n))
    append!(eco_ids, c.eco_id)
    append!(eco_sp_ids, c.eco_species_id)
    append!(ages, FloatType.(c.age_calc))
    append!(agbs, FloatType.(c.agb_sum))
  end

  return DataFrame(
    mapcode=mapcodes,
    eco_id=eco_ids,
    eco_species_id=eco_sp_ids,
    age_calc=ages,
    agb_sum=agbs,
  )
end

function deduplicate_for_export(
  mapped_splots::DataFrame,
  cn_raster::Array{Union{Missing,Int64}},
  eco_raster::Matrix{Int16},
)
  # Assign a sequential mapcode to each unique (plt_cn, raster_ecocode) combination.
  # All raster pixels sharing the same plot/eco get the same mapcode, so cohort rows
  # are written once per combo (no duplication).
  combo_to_mapcode = Dict{Tuple{String,Int64},Int}()
  mc = 0
  for row in eachrow(mapped_splots)
    key = (row.plt_cn, Int64(row.raster_ecocode))
    if !haskey(combo_to_mapcode, key)
      mc += 1
      combo_to_mapcode[key] = mc
    end
  end

  communities_df = DataFrame(
    mapcode=Int[combo_to_mapcode[(row.plt_cn, Int64(row.raster_ecocode))] for row in eachrow(mapped_splots)],
    species=mapped_splots.species,    # semantic string — not an internal _id
    age_calc=FloatType.(mapped_splots.age_calc),
    agb_sum=FloatType.(mapped_splots.agb_sum),
  )

  return communities_df, combo_to_mapcode
end

function load_csv_communities(path::String)::DataFrame
  CSV.read(path, DataFrame)
end

function load_duckdb_communities(db_path::String; tablename::String="communities")::DataFrame
  con = DuckDB.connect(DuckDB.DB(db_path))
  df = DuckDB.execute(con, "SELECT * FROM $(tablename)") |> DataFrame
  DuckDB.close(con)
  return df
end

function prepare_general_splots(
  ic_df::DataFrame,
  eco_raster::Matrix{Int16},
  eco_ecocode_df::DataFrame,
  params,
)
  ecocode_to_eco = Dict(row.ecocode => string(row.eco) for row in eachrow(eco_ecocode_df))
  eco_to_param_id = Dict(eco => i for (i, eco) in enumerate(params.ECO_LIST))
  sp_to_param_id = Dict(sp => i for (i, sp) in enumerate(params.SPECIES_LIST))

  eco_sp_id_map = Dict(
    (eco_id, sp_id) => local_idx
    for (eco_id, species_ids) in enumerate(params.ECO_SPECIES_IDS)
    for (local_idx, sp_id) in enumerate(species_ids)
  )

  ecocodes = [Int64(eco_raster[row.mapcode]) for row in eachrow(ic_df)]
  eco_ids = [get(eco_to_param_id, get(ecocode_to_eco, ec, ""), 0) for ec in ecocodes]
  param_sp_ids = [get(sp_to_param_id, row.species, 0) for row in eachrow(ic_df)]
  eco_sp_ids = [get(eco_sp_id_map, (eid, sid), 0)
                for (eid, sid) in zip(eco_ids, param_sp_ids)]

  valid = (eco_ids .> 0) .& (param_sp_ids .> 0) .& (eco_sp_ids .> 0)

  age_calcs = hasproperty(ic_df, :age) ?
              FloatType.(ic_df.age[valid]) :
              zeros(FloatType, sum(valid))

  splots = DataFrame(
    mapcode=ic_df.mapcode[valid],
    eco_id=eco_ids[valid],
    eco_species_id=eco_sp_ids[valid],
    age_calc=age_calcs,
    agb_sum=FloatType.(ic_df.biomass[valid]),
  )

  return splots, params, Vector{Vector{Int}}(params.ECO_SPECIES_IDS)
end

# ---------------------------------------------------------------------------
# LANDIS-native input loaders
# ---------------------------------------------------------------------------

function load_landis_mapcode_raster(path::String)::Matrix{Int32}
  isfile(path) || error("Raster not found: $path")
  ds = AG.readraster(path)
  band = AG.getband(ds, 1)
  A = Int32.(AG.read(band))
  AG.destroy(ds)
  return A
end

function load_landis_core_species(path::String)::DataFrame
  lines = readlines(path)
  data_rows = String[]
  header_skipped = false
  for line in lines
    stripped = strip(line)
    isempty(stripped) && continue
    startswith(stripped, ">>") && continue
    if !header_skipped
      # first substantive line is "LandisData  Species"
      header_skipped = true
      continue
    end
    push!(data_rows, stripped)
  end

  cols = (
    species_name=String[],
    longevity=Float32[],
    maturity=Float32[],
    seed_eff=Float32[],
    seed_max=Float32[],
    veg_reprod_prob=Float32[],
    sprout_min=Int32[],
    sprout_max=Int32[],
    post_fire=String[],
  )
  for row in data_rows
    parts = split(row)
    length(parts) < 9 && continue
    push!(cols.species_name, parts[1])
    push!(cols.longevity, parse(Float32, parts[2]))
    push!(cols.maturity, parse(Float32, parts[3]))
    push!(cols.seed_eff, parse(Float32, parts[4]))
    push!(cols.seed_max, parse(Float32, parts[5]))
    push!(cols.veg_reprod_prob, parse(Float32, parts[6]))
    push!(cols.sprout_min, parse(Int32, parts[7]))
    push!(cols.sprout_max, parse(Int32, parts[8]))
    push!(cols.post_fire, parts[9])
  end
  return DataFrame(cols)
end

function load_landis_spp_ecoregion(path::String; year::Int=0)::DataFrame
  df = CSV.read(path, DataFrame)
  df.EcoregionName = string.(df.EcoregionName)
  df.SpeciesCode = string.(df.SpeciesCode)
  avail = sort(unique(df.Year))
  candidates = filter(y -> y <= year, avail)
  use_year = isempty(candidates) ? first(avail) : last(candidates)
  return df[df.Year.==use_year, :]
end

const _SUFFICIENT_LIGHT_MATRIX = FloatType[
  1.00 0.50 0.25 0.00 0.00 0.00;
  1.00 1.00 0.50 0.25 0.00 0.00;
  1.00 1.00 1.00 0.50 0.25 0.00;
  1.00 1.00 1.00 1.00 0.50 0.25;
  1.00 1.00 1.00 1.00 1.00 0.50]

function make_landis_params(
  core_sp_df::DataFrame,
  species_df::DataFrame,
  spp_eco_df::DataFrame;
  min_rel_biomass::Vector{Float32}=Float32[0.15, 0.25, 0.50, 0.75, 0.85],
)
  ECO_LIST = sort(unique(spp_eco_df.EcoregionName))
  SPECIES_LIST = sort(unique(spp_eco_df.SpeciesCode))
  eco_to_id = Dict(eco => i for (i, eco) in enumerate(ECO_LIST))
  sp_to_id = Dict(sp => i for (i, sp) in enumerate(SPECIES_LIST))

  # Join core species + species data on species name
  sp_joined = innerjoin(
    rename(core_sp_df, :species_name => :SpeciesCode),
    species_df,
    on=:SpeciesCode,
  )
  sp_row = Dict(row.SpeciesCode => row for row in eachrow(sp_joined))

  # Global per-species arrays (in SPECIES_LIST order)
  n_sp = length(SPECIES_LIST)
  D = zeros(FloatType, n_sp)
  S = zeros(FloatType, n_sp)
  LONGEVITY = zeros(FloatType, n_sp)
  SHADE_TOL = ones(UIntType, n_sp)
  PROB_RESPROUT = ones(UIntType, n_sp)
  MATURITY = zeros(FloatType, n_sp)
  for (i, sp) in enumerate(SPECIES_LIST)
    r = get(sp_row, sp, nothing)
    isnothing(r) && continue
    D[i] = FloatType(r.MortalityCurve)
    S[i] = FloatType(r.GrowthCurve)
    LONGEVITY[i] = FloatType(r.longevity)
    SHADE_TOL[i] = UIntType(r.ShadeTolerance)
    MATURITY[i] = FloatType(r.maturity)
    PROB_RESPROUT[i] = FloatType(r.veg_reprod_prob)
  end

  # Per-eco species membership and eco×species params
  ECO_SPECIES_IDS = Vector{Vector{UIntType}}(undef, length(ECO_LIST))
  B_MAX_SPP = Vector{Vector{FloatType}}(undef, length(ECO_LIST))
  ANPP_MAX_SPP = Vector{Vector{FloatType}}(undef, length(ECO_LIST))
  PROB_MORT_SPP = Vector{Vector{FloatType}}(undef, length(ECO_LIST))
  PROB_ESTAB_SPP = Vector{Vector{FloatType}}(undef, length(ECO_LIST))

  eco_groups = groupby(spp_eco_df, :EcoregionName, sort=true)
  for (eco_key, rows) in pairs(eco_groups)
    eco_id = eco_to_id[eco_key.EcoregionName]
    # sort species by global id for deterministic ordering
    sp_rows = sort(collect(eachrow(rows)), by=r -> get(sp_to_id, r.SpeciesCode, typemax(Int)))
    valid = filter(r -> haskey(sp_to_id, r.SpeciesCode), sp_rows)
    ECO_SPECIES_IDS[eco_id] = UIntType[sp_to_id[r.SpeciesCode] for r in valid]
    B_MAX_SPP[eco_id] = FloatType[r.BiomassMax for r in valid]
    ANPP_MAX_SPP[eco_id] = FloatType[r.ANPPmax for r in valid]
    PROB_MORT_SPP[eco_id] = FloatType[r.ProbMortality for r in valid]
    PROB_ESTAB_SPP[eco_id] = FloatType[r.ProbEstablish for r in valid]
  end

  SUFFICIENT_LIGHT = [vec(_SUFFICIENT_LIGHT_MATRIX[:, sc])
                      for sc in axes(_SUFFICIENT_LIGHT_MATRIX, 2)]
  MIN_REL_BIOMASS = [copy(min_rel_biomass) for _ in ECO_LIST]

  return BiomassSuccessionPlugin.BiomassSuccessionParams(
    ECO_LIST=ECO_LIST,
    SPECIES_LIST=SPECIES_LIST,
    ECO_SPECIES_IDS=ECO_SPECIES_IDS,
    SPINUP_MORTALITY_FRACTION=0.15f0,
    SUFFICIENT_LIGHT=SUFFICIENT_LIGHT,
    MIN_REL_BIOMASS=MIN_REL_BIOMASS,
    D=D,
    S=S,
    LONGEVITY=LONGEVITY,
    SHADE_TOL=SHADE_TOL,
    PROB_RESPROUT=PROB_RESPROUT,
    MATURITY=MATURITY,
    B_MAX_SPP=B_MAX_SPP,
    ANPP_MAX_SPP=ANPP_MAX_SPP,
    PROB_MORT_SPP=PROB_MORT_SPP,
    PROB_ESTAB_SPP=PROB_ESTAB_SPP,
  )
end

function expand_landis_pixels(
  communities_raster::Matrix{Int32},
  eco_raster::Matrix{Int16},
  ic_df::DataFrame,
  eco_ecocode_df::DataFrame,
  params,
)
  ecocode_to_eco = Dict(row.ecocode => string(row.eco) for row in eachrow(eco_ecocode_df))
  eco_to_id = Dict(eco => i for (i, eco) in enumerate(params.ECO_LIST))
  sp_to_id = Dict(sp => i for (i, sp) in enumerate(params.SPECIES_LIST))
  eco_sp_id_map = Dict(
    (eco_id, sp_id) => local_idx
    for (eco_id, species_ids) in enumerate(params.ECO_SPECIES_IDS)
    for (local_idx, sp_id) in enumerate(species_ids)
  )

  cohorts_by_mc = Dict(
    Int32(key.MapCode) => DataFrame(rows)
    for (key, rows) in pairs(groupby(ic_df, :MapCode, sort=false))
  )
  # ensure SpeciesName is String regardless of CSV parsing
  for sub in values(cohorts_by_mc)
    sub.SpeciesName = string.(sub.SpeciesName)
  end

  mapcodes = Int[]
  eco_ids = Int[]
  eco_sp_ids = Int[]
  ages = FloatType[]
  agbs = FloatType[]

  for i in eachindex(communities_raster)
    lmc = communities_raster[i]
    ecocode = Int64(eco_raster[i])
    eco_name = get(ecocode_to_eco, ecocode, nothing)
    isnothing(eco_name) && continue
    eco_id = get(eco_to_id, eco_name, 0)
    eco_id == 0 && continue
    cohorts = get(cohorts_by_mc, lmc, nothing)
    isnothing(cohorts) && continue
    for crow in eachrow(cohorts)
      sp_id = get(sp_to_id, crow.SpeciesName, 0)
      sp_id == 0 && continue
      eco_sp_id = get(eco_sp_id_map, (eco_id, sp_id), 0)
      eco_sp_id == 0 && continue
      push!(mapcodes, i)
      push!(eco_ids, eco_id)
      push!(eco_sp_ids, eco_sp_id)
      push!(ages, FloatType(crow.CohortAge))
      push!(agbs, FloatType(crow.CohortBiomass))
    end
  end

  return DataFrame(
    mapcode=mapcodes,
    eco_id=eco_ids,
    eco_species_id=eco_sp_ids,
    age_calc=ages,
    agb_sum=agbs,
  )
end
