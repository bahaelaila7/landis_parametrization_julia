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

function coalesce_to_duckdb(; output_dir::String, db_path::String)
  chunk_files = sort(filter(
    f -> startswith(f, "cohorts_year") && endswith(f, ".arrow"),
    readdir(output_dir),
  ))
  isempty(chunk_files) && (@warn "No arrow files in $output_dir"; return)

  con = DuckDB.connect(DuckDB.DB(db_path))
  DuckDB.execute(con, """
    CREATE TABLE IF NOT EXISTS cohorts (
      year       UINTEGER,
      mapcode    UINTEGER,
      eco_id     UINTEGER,
      species_id UINTEGER,
      age        FLOAT,
      biomass    FLOAT
    )
  """)
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
  DuckDB.close(con)
end

function export_landis_params(params::BiomassSuccessionParams; output_dir::String)
  mkpath(output_dir)

  eco_list   = params.ECO_LIST
  sp_list    = params.SPECIES_LIST
  eco_sp_ids = params.ECO_SPECIES_IDS
  n_sp  = length(sp_list)
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
    println(io, "Year,EcoregionName,SpeciesCode,ProbEstablis,ProbMortality,ANPPmax,BiomassMax")
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

  # ---- biomass_succession.txt ---------------------------------------------
  open(joinpath(output_dir, "biomass_succession.txt"), "w") do io
    println(io, "LandisData  \"Biomass Succession\"")
    println(io, "")
    println(io, "Timestep  10")
    println(io, "")
    println(io, "SeedingAlgorithm  NoDispersal")
    println(io, "")
    println(io, ">> Fill in paths to your initial communities files:")
    println(io, "InitialCommunities    ./initial_communities.txt")
    println(io, "InitialCommunitiesMap ./initial_communities.tif")
    println(io, "")
    println(io, "CalibrateMode  no")
    println(io, "")
    println(io, "SpinupMortalityFraction  $(params.SPINUP_MORTALITY_FRACTION)")
    println(io, "")

    # MinRelativeBiomass: rows = shade class (1-5), cols = ecoregion
    println(io, "MinRelativeBiomass")
    print(io, ">> ShadeClass")
    for eco in eco_list; print(io, "\t$eco"); end
    println(io)
    for sc in 1:5
      print(io, "   $sc")
      for eco_id in 1:n_eco
        @printf(io, "\t%.4f", params.MIN_REL_BIOMASS[eco_id][sc])
      end
      println(io)
    end
    println(io)

    # SufficientLight: rows = shade class (1-6), cols = shade tolerance (1-5)
    println(io, "SufficientLight")
    print(io, ">> ShadeClass")
    for st in 1:5; print(io, "\tShadeTol$st"); end
    println(io)
    for sc in 1:length(params.SUFFICIENT_LIGHT)
      print(io, "   $sc")
      for st in 1:5
        @printf(io, "\t%.2f", params.SUFFICIENT_LIGHT[sc][st])
      end
      println(io)
    end
    println(io)

    # SpeciesParameters inline table
    println(io, "SpeciesParameters")
    println(io, ">> Species           LeafLongevity  WoodDecayRate  MortalityCurve  GrowthCurve  LeafLignin  ShadeTolerance  FireTolerance")
    for i in 1:n_sp
      @printf(io, "   %-16s  %13.1f  %13.1f  %14.4f  %11.4f  %10.1f  %14d  %13d\n",
        sp_list[i], 1.0, 0.1, params.D[i], params.S[i], 0.1, params.SHADE_TOL[i], 1)
    end
    println(io)

    println(io, "DynamicInputFile  ./SppEcoregionData.csv")
  end

  println("Exported LANDIS-II parameters to $output_dir")
  for f in ("CoreSpeciesData.txt", "SpeciesData.csv", "SppEcoregionData.csv", "biomass_succession.txt")
    println("  $f")
  end
end
