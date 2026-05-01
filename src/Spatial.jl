module Spatial

using ..PanCore
using ..Plugins: BiomassSuccessionPlugin
import Term.Progress as TProgress
import Format
import ArchGDAL
import Arrow
import DuckDB

export run_spatial!, write_raster, generate_rasters_from_output,
  coalesce_to_duckdb

const FLUSH_THRESHOLD = 5_000_000  # records; 120 MB per thread at 24 B/record

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

function _emit_year!(thread_buffers, thread_chunks, soa, year, output_dir)
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

function run_spatial!(soa, eco_params, output_dir::String;
  timehorizon::Int, output_every::Int)
  ctx = (BiomassSuccession=(eco_params=eco_params,),)
  thread_buffers = [_new_buf() for _ in 1:Threads.maxthreadid()]
  thread_chunks = zeros(Int, Threads.maxthreadid())

  mkpath(output_dir)
  _emit_year!(thread_buffers, thread_chunks, soa, 0, output_dir)
  PanCore.gc_and_trim!()

  TProgress.@track for year in 1:timehorizon
    PanCore.process_plugin!(soa, BiomassSuccessionPlugin.BiomassSuccession,
      year; ctx=ctx.BiomassSuccession)
    @info "Year $year: SoA $(round(Base.summarysize(soa)/1e9, digits=2)) GB, $(Format.format(sum(soa.scalar._new_cohort_counts), commas=true)) cohorts, RSS: $(round(PanCore.current_rss_gb(), digits=2)) GB, peak RSS: $(round(Sys.maxrss() / 1e9, digits=2)) GB"

    if year % output_every == 0
      _emit_year!(thread_buffers, thread_chunks, soa, year, output_dir)
    end
  end
end

function write_raster(src_path::String, output_path::String,
  data::AbstractArray{Float32};
  nodata::Float32=-9999.0f0)
  ArchGDAL.read(src_path) do src
    out = copy(data)
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
      band = ArchGDAL.getband(dst, 1)
      ArchGDAL.setnodatavalue!(band, nodata)
      ArchGDAL.write!(band, out)
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

  # Group files by year (parse from "cohorts_year5_t3_c1.arrow")
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

    TProgress.@track for year in sort(collect(keys(year_files)))
      dst_data = zeros(Float32, w, h)

      # Files for the same year have disjoint mapcode sets (each thread wrote a
      # static partition of sites), so parallel writes to dst_data are race-free.
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
  DuckDB.execute(
    con,
    """
  CREATE TABLE IF NOT EXISTS cohorts (
    year     UINTEGER,
    mapcode  UINTEGER,
    eco_id   UINTEGER,
    species_id UINTEGER,
    age      FLOAT,
    biomass  FLOAT
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
  DuckDB.close(con)
end

end # module Spatial
