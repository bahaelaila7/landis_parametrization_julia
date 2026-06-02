import Printf: @printf
import Arrow
import DuckDB
import ArchGDAL

# ---------------------------------------------------------------------------
# Cohort output — reads BiomassSuccession SoA fields (c_species, c_age, c_bio)
# ---------------------------------------------------------------------------

const FLUSH_THRESHOLD = 5_000_000  # records; ~120 MB per thread at 24 B/record

mutable struct ThreadBuf
  year::Vector{UInt32}
  mapcode::Vector{UInt32}
  eco_id::Vector{UInt32}
  species_id::Vector{UInt32}
  age::Vector{Float32}
  biomass::Vector{Float32}
  n::Int
end

function _new_buf(cap::Int=FLUSH_THRESHOLD)
  ThreadBuf(
    Vector{UInt32}(undef, cap),
    Vector{UInt32}(undef, cap),
    Vector{UInt32}(undef, cap),
    Vector{UInt32}(undef, cap),
    Vector{Float32}(undef, cap),
    Vector{Float32}(undef, cap),
    0,
  )
end

function _flush_buf!(buf::ThreadBuf, output_dir, year, tid, chunk)
  path = joinpath(output_dir, "cohorts_year$(year)_t$(tid)_c$(chunk).arrow")
  Arrow.write(path, (
      year=(@view buf.year[1:buf.n]),
      mapcode=(@view buf.mapcode[1:buf.n]),
      eco_id=(@view buf.eco_id[1:buf.n]),
      species_id=(@view buf.species_id[1:buf.n]),
      age=(@view buf.age[1:buf.n]),
      biomass=(@view buf.biomass[1:buf.n]),
    ); compress=:lz4)
  @info "Wrote year=$(year) thread=$(tid) chunk=$(chunk) ($(buf.n) records)"
  buf.n = 0
end

function emit_year!(thread_buffers, thread_chunks, soa, year, output_dir)
  fill!(thread_chunks, 0)
  Threads.@threads :static for i in 1:soa.n
    @inbounds begin
      site = getsite(soa, i)
      !site.active && continue
      tid = Threads.threadid()
      buf = thread_buffers[tid]
      mc = UInt32(site.mapcode)
      eid = UInt32(site.eco_id)
      yr = UInt32(year)
      for k in 1:site.live
        n = buf.n + 1
        buf.year[n] = yr
        buf.mapcode[n] = mc
        buf.eco_id[n] = eid
        buf.species_id[n] = UInt32(site.c_species[k])
        buf.age[n] = Float32(site.c_age[k])
        buf.biomass[n] = Float32(site.c_bio[k])
        buf.n = n
        if n == FLUSH_THRESHOLD
          thread_chunks[tid] += 1
          _flush_buf!(buf, output_dir, year, tid, thread_chunks[tid])
        end
      end
    end
  end
  Threads.@threads :static for tid in eachindex(thread_buffers)
    buf = thread_buffers[tid]
    if buf.n > 0
      thread_chunks[tid] += 1
      _flush_buf!(buf, output_dir, year, tid, thread_chunks[tid])
    end
  end
end

function generate_rasters_from_output(; output_dir::String, ref_raster_path::String)
  isfile(ref_raster_path) || error("Reference raster not found: $(abspath(ref_raster_path))")
  chunk_files = filter(
    f -> startswith(f, "cohorts_year") && endswith(f, ".arrow"),
    readdir(output_dir),
  )
  isempty(chunk_files) && (@warn "No cohort arrow files in $output_dir"; return)

  year_files = Dict{Int,Vector{String}}()
  for f in chunk_files
    m = match(r"cohorts_year(\d+)_", f)
    isnothing(m) && continue
    push!(get!(year_files, parse(Int, m.captures[1]), String[]), f)
  end

  ArchGDAL.read(ref_raster_path) do src
    w = ArchGDAL.width(src)
    h = ArchGDAL.height(src)
    gt = ArchGDAL.getgeotransform(src)
    proj = ArchGDAL.getproj(src)

    for year in sort(collect(keys(year_files)))
      dst_data = zeros(Float32, w, h)
      Threads.@threads :static for fname in year_files[year]
        tbl = Arrow.Table(joinpath(output_dir, fname))
        for i in eachindex(tbl.mapcode)
          @inbounds dst_data[Int(tbl.mapcode[i])] += tbl.biomass[i]
        end
      end
      out_path = joinpath(output_dir, "agb_$(year).tif")
      ArchGDAL.create(
        out_path,
        driver=ArchGDAL.getdriver("GTiff"),
        width=w, height=h, nbands=1, dtype=Float32,
      ) do dst
        ArchGDAL.setgeotransform!(dst, gt)
        ArchGDAL.setproj!(dst, proj)
        band = ArchGDAL.getband(dst, 1)
        ArchGDAL.setnodatavalue!(band, 0.0f0)
        ArchGDAL.write!(band, dst_data)
      end
      @info "Wrote agb_$(year).tif"
    end
  end
end

function coalesce_to_duckdb(; output_dir::String, db_path::String, fresh::Bool=true)
  chunk_files = sort(filter(
    f -> startswith(f, "cohorts_year") && endswith(f, ".arrow"),
    readdir(output_dir),
  ))
  isempty(chunk_files) && (@warn "No arrow files in $output_dir"; return)

  fresh && isfile(db_path) && rm(db_path)

  db = DuckDB.DB(db_path)
  con = DuckDB.connect(db)
  DuckDB.execute(
    con,
    """
  CREATE OR REPLACE TABLE cohorts (
    year       UINTEGER,
    mapcode    UINTEGER,
    eco_id     UINTEGER,
    species_id UINTEGER,
    age        FLOAT,
    biomass    FLOAT
  )
"""
  )
  app = DuckDB.Appender(con, "cohorts")
  for fname in chunk_files
    tbl = Arrow.Table(joinpath(output_dir, fname))
    n = length(tbl.year)
    for i in 1:n
      DuckDB.append(app, tbl.year[i])
      DuckDB.append(app, tbl.mapcode[i])
      DuckDB.append(app, tbl.eco_id[i])
      DuckDB.append(app, tbl.species_id[i])
      DuckDB.append(app, tbl.age[i])
      DuckDB.append(app, tbl.biomass[i])
      DuckDB.end_row(app)
    end
    DuckDB.flush(app)
    @info "Loaded $(fname): $n records"
  end
  DuckDB.close(app)
  DuckDB.close(db)
end

function export_landis_params(params::BiomassSuccessionParams;
  output_dir::String,
  climate_config_file::String,
)
  mkpath(output_dir)

  eco_list = params.ECO_LIST
  sp_list = params.SPECIES_LIST
  eco_sp_ids = params.ECO_SPECIES_IDS
  n_sp = length(sp_list)
  n_eco = length(eco_list)

  # ---- CoreSpeciesData.txt ------------------------------------------------
  open(joinpath(output_dir, "CoreSpeciesData.txt"), "w") do io
    println(io, "LandisData  Species")
    println(io, "")
    println(io, ">> Name             Longevity  Maturity  SeedDispEff  SeedDispMax  VegReprodProb  SproutMin  SproutMax  PostFireRegen")
    for i in 1:n_sp
      lon = round(Int, params.LONGEVITY[i])
      mat = round(Int, params.MATURITY[i])
      @printf(io, "   %-16s  %9d  %8d  %11d  %11d  %13s  %9d  %9d  %s\n",
        sp_list[i], lon, mat, 30, 100, "0.0", 0, 0, "none")
    end
  end

  # ---- SpeciesData.csv ----------------------------------------------------
  open(joinpath(output_dir, "SpeciesData.csv"), "w") do io
    println(io, "SpeciesCode,LeafLongevity,WoodDecayRate,MortalityCurve,GrowthCurve,LeafLignin,ShadeTolerance,FireTolerance")
    for i in 1:n_sp
      println(io, "$(sp_list[i]),1.0,0.1,$(params.D[i]),$(params.S[i]),0.1,$(params.SHADE_TOL[i]),1")
    end
  end

  # ---- SppEcoregionData.csv -----------------------------------------------
  open(joinpath(output_dir, "SppEcoregionData.csv"), "w") do io
    println(io, "Year,EcoregionName,SpeciesCode,ProbEstablish,ProbMortality,ANPPmax,BiomassMax")
    for (eco_id, eco) in enumerate(eco_list)
      for (loc_id, gsp_id) in enumerate(eco_sp_ids[eco_id])
        sp = sp_list[gsp_id]
        println(io,
          "0,$eco,$sp," *
          "$(params.PROB_ESTAB_SPP[eco_id][loc_id])," *
          "$(params.PROB_MORT_SPP[eco_id][loc_id])," *
          "$(params.ANPP_MAX_SPP[eco_id][loc_id])," *
          "$(params.B_MAX_SPP[eco_id][loc_id])")
      end
    end
  end

  # ---- biomass_succession.txt (LANDIS-II v8 Biomass Succession format) ----
  open(joinpath(output_dir, "biomass_succession.txt"), "w") do io
    println(io, "LandisData  \"Biomass Succession\"")
    println(io, "")
    println(io, "Timestep  1")
    println(io, "")
    println(io, "SeedingAlgorithm  NoDispersal")
    println(io, "")
    println(io, "InitialCommunities    ./initial_communities.csv")
    println(io, "InitialCommunitiesMap ./initial_communities.tif")
    println(io, "")

    println(io, "ClimateConfigFile $(climate_config_file)")
    println(io, "")

    println(io, "CalibrateMode  no")
    println(io, "")
    println(io, "SpinupMortalityFraction  $(params.SPINUP_MORTALITY_FRACTION)")
    println(io, "")

    # MinRelativeBiomass: rows = shade class (1-5), cols = ecoregion Name
    # Header = eco names only (no >> ShadeClass prefix); values as percentages.
    println(io, "MinRelativeBiomass")
    print(io, "")
    for eco in eco_list
      print(io, "\t$eco")
    end
    println(io)
    for sc in 1:5
      print(io, "   $sc")
      for eco_id in 1:n_eco
        @printf(io, "\t%.2f%%", params.MIN_REL_BIOMASS[eco_id][sc] * 100)
      end
      println(io)
    end
    println(io)

    # SufficientLight: transposed — rows = shade tolerance (1-5), cols = shade class (1-n)
    n_sc = length(params.SUFFICIENT_LIGHT)
    n_st = length(params.SUFFICIENT_LIGHT[1])
    println(io, "SufficientLight")
    print(io, ">> ShadeTol")
    for sc in 1:n_sc
      @printf(io, "\t%d", sc)
    end
    println(io)
    for st in 1:n_st
      print(io, "   $st")
      for sc in 1:n_sc
        @printf(io, "\t%.2f", params.SUFFICIENT_LIGHT[sc][st])
      end
      println(io)
    end
    println(io)

    println(io, "SpeciesDataFile  ./SpeciesData.csv")
    println(io, "")
    println(io, "SpeciesEcoregionDataFile  ./SppEcoregionData.csv")
    println(io, "")

    # EcoregionParameters: one row per ecoregion (Name), AET in mm
    println(io, "EcoregionParameters")
    println(io, ">> Eco\tAET (mm)")
    for eco in eco_list
      println(io, "   $eco\t600")
    end
    println(io)


    println(io, "FireReductionParameters")
    println(io, ">> Severity\tWoodLitter\tLitter")
    println(io, ">> \t\tReduct\t\tReduct")
    println(io, "   1\t\t0.0\t\t0.5")
    println(io, "   2\t\t0.0\t\t0.75")
    println(io, "   3\t\t0.0\t\t1.0")
    println(io, "")
    println(io, "HarvestReductionParameters")
    println(io, ">> Name\t\tWoodLitter\tLitter\tCohort\t\tCohort")
    println(io, ">> \t\tReduct\t\tReduct\tWoodRemoval\tLeafRemoval")
    println(io, "   MaxAgeClearcut\t0.5\t\t0.15\t0.8\t\t0.0")
    println(io, "   PatchCutting\t\t1.0\t\t1.0\t1.0\t\t0.0")
  end

  println("Exported LANDIS-II parameters to $output_dir")
  for f in ("CoreSpeciesData.txt", "SpeciesData.csv", "SppEcoregionData.csv", "biomass_succession.txt")
    println("  $f")
  end
end

# ---------------------------------------------------------------------------
# Export initial communities CSV.
# communities_df must have columns: mapcode (Int), species (String), age_calc, agb_sum.
# No internal _id fields — caller provides semantic data from deduplicate_for_export.
# ---------------------------------------------------------------------------

function export_initial_communities_csv(
  communities_df::DataFrame;
  output_path::String,
)
  open(output_path, "w") do io
    println(io, "MapCode,SpeciesName,CohortAge,CohortBiomass")
    for row in eachrow(communities_df)
      println(io, "$(Int(row.mapcode)),$(row.species),$(row.age_calc),$(row.agb_sum)")
    end
  end
  println("  initial_communities.csv")
end

# ---------------------------------------------------------------------------
# Export initial communities TIF using combo_to_mapcode lookup built by
# Data.deduplicate_for_export: key = (plt_cn::String, ecocode::Int64).
# Every raster pixel sharing the same (plt_cn, ecocode) gets the same mapcode.
# ---------------------------------------------------------------------------

function export_initial_communities_tif(
  combo_to_mapcode::Dict{Tuple{String,Int64},Int},
  cn_raster::Array{Union{Missing,Int64}},
  eco_raster::Matrix{Int16},
  ref_raster_path::String;
  output_path::String,
)
  ArchGDAL.read(ref_raster_path) do src
    w = ArchGDAL.width(src)
    h = ArchGDAL.height(src)
    gt = ArchGDAL.getgeotransform(src)
    proj = ArchGDAL.getproj(src)

    mapcode_data = zeros(Int32, w, h)
    for i in eachindex(cn_raster)
      plt_cn = cn_raster[i]
      ismissing(plt_cn) && continue
      mc = get(combo_to_mapcode, ("$(plt_cn)", Int64(eco_raster[i])), nothing)
      isnothing(mc) && continue
      mapcode_data[i] = Int32(mc)
    end

    ArchGDAL.create(
      output_path,
      driver=ArchGDAL.getdriver("GTiff"),
      width=w, height=h, nbands=1, dtype=Int32,
    ) do dst
      ArchGDAL.setgeotransform!(dst, gt)
      ArchGDAL.setproj!(dst, proj)
      band = ArchGDAL.getband(dst, 1)
      ArchGDAL.setnodatavalue!(band, 0)
      ArchGDAL.write!(band, mapcode_data)
    end
  end
  println("  initial_communities.tif")
end

# Write a coarse (downsampled n×) raster to GeoTIFF, deriving CRS/extent from a reference
# raster and scaling its pixel size by n (origin unchanged). `data` is indexed (width, height).
function export_coarse_raster(
  data::AbstractMatrix,
  ref_raster_path::String,
  n::Int;
  output_path::String,
  dtype::DataType,
  nodata,
)
  ArchGDAL.read(ref_raster_path) do src
    gt = ArchGDAL.getgeotransform(src)
    proj = ArchGDAL.getproj(src)
    gt[2] *= n  # pixel width
    gt[3] *= n  # row rotation
    gt[5] *= n  # column rotation
    gt[6] *= n  # pixel height
    wc, hc = size(data)
    out = Array{dtype}(data)
    ArchGDAL.create(
      output_path,
      driver=ArchGDAL.getdriver("GTiff"),
      width=wc, height=hc, nbands=1, dtype=dtype,
    ) do dst
      ArchGDAL.setgeotransform!(dst, gt)
      ArchGDAL.setproj!(dst, proj)
      band = ArchGDAL.getband(dst, 1)
      ArchGDAL.setnodatavalue!(band, nodata)
      ArchGDAL.write!(band, out)
    end
  end
  println("  $(basename(output_path)) (downsampled $(n)x)")
end

# ---------------------------------------------------------------------------
# Export ecoregion.txt — LANDIS ecoregion table.
# Format: Active  MapCode  Name  "Description"
# Name (eco string) is used as the identifier in biomass_succession.txt and
# SppEcoregionData.csv. MapCode (ecocode integer) matches the ecoregion TIF.
# ---------------------------------------------------------------------------

function export_ecoregions_txt(
  params::BiomassSuccessionParams,
  eco_mapping_df::DataFrame;
  output_path::String,
)
  ecocode_map = Dict(string(row.eco) => Int(row.ecocode) for row in eachrow(eco_mapping_df))
  open(output_path, "w") do io
    println(io, "LandisData\t\"Ecoregions\"")
    println(io, "")
    println(io, ">> Active\tMapCode\tName\tDescription")
    for eco in params.ECO_LIST
      eco_name = string(eco)
      ecocode = get(ecocode_map, eco_name, nothing)
      isnothing(ecocode) && continue
      println(io, "yes\t$(ecocode)\t$(eco_name)\t\"\"")
    end
    println(io, "no\t0\tnodata\t\"\"")
  end
  println("  ecoregion.txt")
end

# ---------------------------------------------------------------------------
# Export scenario.txt — LANDIS-II v8 scenario file.
# Mirrors the structure of the reference fl5_baseline scenario.
# ---------------------------------------------------------------------------

function export_scenario_file(;
  output_path::String,
  duration_years::Int=40,
  cell_length_m::Int=30,
  rng_seed::Union{Int,Nothing}=nothing,
)
  open(output_path, "w") do io
    println(io, "LandisData  \"Scenario\"")
    println(io, "")
    println(io, "")
    println(io, ">> ---------------------------------------------")
    println(io, ">> REQUIRED INPUTS")
    println(io, ">> ---------------------------------------------")
    println(io, "")
    println(io, "Duration  \t$(duration_years)")
    println(io, "")
    println(io, "Species   \tCoreSpeciesData.txt")
    println(io, "")
    println(io, "Ecoregions      ./ecoregion.txt")
    println(io, "EcoregionsMap   ./ecoregion.tif")
    println(io, "")
    println(io, "CellLength  \t$(cell_length_m) << meters")
    println(io, "")
    println(io, "")
    println(io, ">> -----------------------")
    println(io, ">> SUCCESSION EXTENSIONS")
    println(io, ">> -----------------------")
    println(io, "")
    println(io, "\t\"Biomass Succession\"\tbiomass_succession.txt")
    println(io, "")
    println(io, "")
    println(io, ">> --------------------------")
    println(io, ">> DISTURBANCE EXTENSIONS")
    println(io, ">> --------------------------")
    println(io, "")
    println(io, ">>   DisturbancesRandomOrder  yes")
    println(io, "")
    println(io, "")
    println(io, ">> ----------------------")
    println(io, ">> OUTPUT EXTENSIONS")
    println(io, ">> ----------------------")
    println(io, "")
    println(io, ">> \t\"Output Biomass\"\t\toutput_Biomass.txt")
    println(io, "")
    println(io, "")
    if !isnothing(rng_seed)
      println(io, " RandomNumberSeed  $(rng_seed)")
    else
      println(io, ">> RandomNumberSeed  1337  << uncomment for reproducibility")
    end
    println(io, "")
  end
  println("  scenario.txt")
end

# ---------------------------------------------------------------------------
# Export eco_ecocode_mapping.csv (filtered to params ecoregions)
# ---------------------------------------------------------------------------

function export_eco_ecocode_mapping(
  params::BiomassSuccessionParams,
  eco_mapping_df::DataFrame;
  output_path::String,
)
  eco_set = Set(params.ECO_LIST)
  open(output_path, "w") do io
    println(io, "ecocode,eco")
    for row in eachrow(eco_mapping_df)
      eco_name = string(row.eco)
      if eco_name in eco_set
        println(io, "$(row.ecocode),$eco_name")
      end
    end
  end
  println("  eco_ecocode_mapping.csv")
end
