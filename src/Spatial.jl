module Spatial

using ..PanCore
using ..Plugins: BiomassSuccessionPlugin
import Term.Progress as TProgress
import ArchGDAL
import Parquet2
import DataFrames: DataFrame

export SiteRecord, start_spatial_writer, stop_spatial_writer, run_spatial!,
       write_raster, generate_rasters_from_output

struct SiteRecord
    year::UInt32
    mapcode::UInt32
    eco_id::UInt32
    species_id::UInt32
    age::Float32
    biomass::Float32
end

const _STOP = :stop
const FLUSH_THRESHOLD = 200_000

function _records_to_df(records::Vector{SiteRecord})
    DataFrame(
        year       = [r.year       for r in records],
        mapcode    = [r.mapcode    for r in records],
        eco_id     = [r.eco_id     for r in records],
        species_id = [r.species_id for r in records],
        age        = [r.age        for r in records],
        biomass    = [r.biomass    for r in records],
    )
end

function start_spatial_writer(output_dir::String; buffer_size::Int=16)
    ch = Channel{Union{Vector{SiteRecord},Symbol}}(buffer_size)
    task = Threads.@spawn begin
        chunk = 1
        buf = SiteRecord[]
        sizehint!(buf, FLUSH_THRESHOLD)
        try
            for payload in ch
                payload === _STOP && break
                append!(buf, payload)
                if length(buf) >= FLUSH_THRESHOLD
                    mkpath(output_dir)
                    path = joinpath(output_dir, "cohorts_chunk_$(chunk).parquet")
                    Parquet2.writefile(path, _records_to_df(buf))
                    @info "Spatial writer: flushed chunk $(chunk) ($(length(buf)) records)"
                    chunk += 1
                    empty!(buf)
                end
            end
        catch e
            @error "Spatial writer: fatal" exception=(e, catch_backtrace())
            rethrow()
        end
        if !isempty(buf)
            mkpath(output_dir)
            path = joinpath(output_dir, "cohorts_chunk_$(chunk).parquet")
            Parquet2.writefile(path, _records_to_df(buf))
            @info "Spatial writer: flushed final chunk $(chunk) ($(length(buf)) records)"
        end
    end
    return ch, task
end

function stop_spatial_writer(ch::Channel, task::Task)
    put!(ch, _STOP)
    close(ch)
    wait(task)
end

function run_spatial!(soa, eco_params, writer_ch::Channel;
                      timehorizon::Int, output_every::Int)
    ctx = (BiomassSuccession=(eco_params=eco_params,),)
    thread_buffers = [SiteRecord[] for _ in 1:Threads.maxthreadid()]

    TProgress.@track for year in 1:timehorizon
        PanCore.process_plugin!(soa, BiomassSuccessionPlugin.BiomassSuccession,
                                year; ctx=ctx.BiomassSuccession)

        if year % output_every == 0
            Threads.@threads :static for i in 1:soa.n
                @inbounds begin
                    site = getsite(soa, i)
                    !site.active && continue
                    buf  = thread_buffers[Threads.threadid()]
                    for k in 1:site.live
                        push!(buf, SiteRecord(
                            UInt32(year),
                            UInt32(site.mapcode),
                            UInt32(site.eco_id),
                            UInt32(site.c_species[k]),
                            Float32(site.c_age[k]),
                            Float32(site.c_bio[k]),
                        ))
                    end
                end
            end
            all_records = reduce(vcat, thread_buffers)
            foreach(empty!, thread_buffers)
            isempty(all_records) || put!(writer_ch, all_records)
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

function generate_rasters_from_output(output_dir::String, ref_raster_path::String)
    chunk_files = filter(
        f -> startswith(f, "cohorts_chunk_") && endswith(f, ".parquet"),
        readdir(output_dir),
    )
    isempty(chunk_files) && (@warn "No cohort parquet files in $output_dir"; return)

    # year → mapcode → accumulated biomass
    year_agb = Dict{Int,Dict{Int,Float32}}()

    for fname in chunk_files
        ds = Parquet2.Dataset(joinpath(output_dir, fname))
        for chunk in Parquet2.Tables.partitions(ds)
            years    = Parquet2.Tables.getcolumn(chunk, :year)
            mapcodes = Parquet2.Tables.getcolumn(chunk, :mapcode)
            biomasses = Parquet2.Tables.getcolumn(chunk, :biomass)
            for i in eachindex(years)
                y   = Int(years[i])
                mc  = Int(mapcodes[i])
                bio = Float32(biomasses[i])
                d = get!(year_agb, y, Dict{Int,Float32}())
                d[mc] = get(d, mc, 0.0f0) + bio
            end
        end
    end

    ArchGDAL.read(ref_raster_path) do src
        w   = ArchGDAL.width(src)
        h   = ArchGDAL.height(src)
        gt  = ArchGDAL.getgeotransform(src)
        proj = ArchGDAL.getproj(src)

        TProgress.@track for (year, mc_bio) in sort(collect(year_agb), by=first)
            dst_data = zeros(Float32, w, h)
            for (mc, bio) in mc_bio
                @inbounds dst_data[mc] = bio
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

end # module Spatial
