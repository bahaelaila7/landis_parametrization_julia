# Site attributes the PnET plugin needs beyond the shared cohort table: climate series (EPA L4 -> PRISM cache index), calendar year of
# sim year 0, latitude, and soil texture. Built in `Pan.parametrize` only when succession_plugin == pnet and handed to `init_sites!`.

"""
    configure!(; climate_cache, co2_csv, generic=Dict(), db_path=nothing)

Load the PRISM climate cache + CO2 series into `CLIMATE[]`, apply `generic` overrides (PnETConst field names -> values) to `FIXED[]`, and
(if `db_path` is given) read REF_SPECIES.SFTWD_HRDWD for plant-functional-type assignment.
"""
function configure!(; climate_cache::AbstractString, co2_csv::AbstractString, generic::AbstractDict=Dict{String,Any}(), db_path::Union{Nothing,AbstractString}=nothing)
  CLIMATE[] = load_climate(climate_cache, co2_csv)
  ov = Dict{Symbol,Any}()
  for (k, v) in generic
    f = Symbol(k)
    f in fieldnames(PnETConst) || error("pnet_generic: unknown PnETConst field '$k' (valid: $(collect(fieldnames(PnETConst))))")
    ov[f] = v
  end
  base = PnETConst()
  FIXED[] = PnETConst(; (f => convert(fieldtype(PnETConst, f), get(ov, f, getfield(base, f))) for f in fieldnames(PnETConst))...)
  if db_path !== nothing && isfile(db_path)
    db = DuckDB.DB(); con = DuckDB.connect(db)
    try
      DuckDB.execute(con, "ATTACH '$(db_path)' AS src (READ_ONLY);")
      df = DuckDB.execute(con, "SELECT UPPER(TRIM(SPECIES_SYMBOL)) AS s, SFTWD_HRDWD AS t FROM src.REF_SPECIES") |> DataFrame
      SPECIES_SFTWD[] = Dict{String,String}(String(r.s) => String(coalesce(r.t, "")) for r in eachrow(df) if !ismissing(r.s))
    catch e
      @warn "PnET: could not read REF_SPECIES.SFTWD_HRDWD; PFT falls back to genus heuristics" exception = e
    finally
      DuckDB.close(db)
    end
  end
  return nothing
end

_l3_of_eco(eco::AbstractString) = String(first(split(String(eco), "|lu=")))

"""
    build_site_attrs(db_path, tablename, splots, eco_list; soil_map=Dict(), default_soil="LOAM") -> DataFrame

One row per plot_id of `splots`: clim_id (index into CLIMATE[].ids from the plot's epa_l4), cal_year0 (calendar year of the plot's first
measurement = sim year 0), lat (PLOT.LAT), soil (Saxton-Rawls texture code from `soil_map[L3]`, else `default_soil`).
"""
function build_site_attrs(db_path::AbstractString, tablename::AbstractString, splots::DataFrame, eco_list::Vector{String};
                          soil_map::AbstractDict=Dict{String,String}(), default_soil::AbstractString="LOAM")
  clim = CLIMATE[]
  clim === nothing && error("PnET climate not loaded: call PnETPlugin.configure!(...) first")
  db = DuckDB.DB(); con = DuckDB.connect(db)
  l4 = Dict{NTuple{4,Int},String}(); lat = Dict{NTuple{4,Int},Float64}()
  try
    DuckDB.execute(con, "ATTACH '$(db_path)' AS src (READ_ONLY);")
    a = DuckDB.execute(con, "SELECT statecd, unitcd, countycd, plot, min(epa_l4) AS epa_l4 FROM src.$(tablename) GROUP BY ALL") |> DataFrame
    for r in eachrow(a)
      ismissing(r.epa_l4) || (l4[(Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot))] = String(r.epa_l4))
    end
    b = DuckDB.execute(con, "SELECT STATECD AS s, UNITCD AS u, COUNTYCD AS c, PLOT AS p, avg(LAT) AS lat FROM src.PLOT WHERE LAT IS NOT NULL GROUP BY ALL") |> DataFrame
    for r in eachrow(b)
      lat[(Int(r.s), Int(r.u), Int(r.c), Int(r.p))] = Float64(r.lat)
    end
  finally
    DuckDB.close(db)
  end
  idpos = Dict(id => i for (i, id) in enumerate(clim.ids))
  l3pos = Dict{String,Int}()
  for (i, id) in enumerate(clim.ids)
    p = split(id, '.'); length(p) >= 3 && get!(l3pos, join(p[1:3], '.'), i)
  end
  u = unique(select(splots, [:plot_id, :statecd, :unitcd, :countycd, :plot, :eco_id, :start_measdate]))
  g = combine(groupby(u, [:plot_id, :statecd, :unitcd, :countycd, :plot, :eco_id]), :start_measdate => minimum => :sm)
  nmiss_l4 = 0; nmiss_lat = 0
  rows = NamedTuple[]
  for r in eachrow(g)
    key = (Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot))
    code = get(l4, key, "")
    cid = get(idpos, code, 0)
    if cid == 0
      nmiss_l4 += 1
      p = split(code, '.'); l3 = length(p) >= 3 ? join(p[1:3], '.') : ""
      cid = get(l3pos, l3, 1)
    end
    la = get(lat, key, NaN)
    if isnan(la); nmiss_lat += 1; la = 33.0; end
    l3e = _l3_of_eco(eco_list[Int(r.eco_id)])
    push!(rows, (plot_id=Int(r.plot_id), clim_id=Int32(cid), cal_year0=Int16(Dates.year(r.sm)), lat=Float32(la), soil=String(get(soil_map, l3e, default_soil))))
  end
  (nmiss_l4 > 0 || nmiss_lat > 0) && @warn "PnET site attrs: $nmiss_l4 plots had no/unknown L4 in the climate cache (L3 fallback used), $nmiss_lat had no latitude (33 N used)"
  return DataFrame(rows)
end

"""Set one site's PnET static attributes + initial state (soil bucket full to field capacity, empty snow pack)."""
function init_site!(soa, i::Int; clim_id::Integer, cal_year0::Integer, lat::Real, soil::AbstractString="LOAM", root_depth::Real=FIXED[].root_depth_base)
  por, fc, wp, A, B = saxton_rawls(soil)
  sc = soa.scalar
  sc.clim_id[i] = clim_id; sc.cal_year0[i] = cal_year0; sc.lat[i] = lat
  sc.soil_por[i] = por; sc.soil_fc[i] = fc; sc.soil_wp[i] = wp; sc.soil_A[i] = A; sc.soil_B[i] = B
  sc.rooting_depth[i] = root_depth
  sc.soil_water[i] = fc; sc.snowpack[i] = 0; sc.canopy_lai_prev[i] = 0; sc.subcanopy_par[i] = 0; sc.B[i] = 0
  return nothing
end

"""Plugin hook called by make_sites with the per-plot attribute table (rows keyed by plot_id == site.mapcode)."""
function init_sites!(soa, attrs::DataFrame)
  pos = Dict{Int,Int}(Int(r.plot_id) => k for (k, r) in enumerate(eachrow(attrs)))
  for i in 1:soa.n
    k = get(pos, Int(soa.scalar.mapcode[i]), 0)
    k == 0 && error("PnET init_sites!: no site attributes for plot_id $(soa.scalar.mapcode[i])")
    r = attrs[k, :]
    init_site!(soa, i; clim_id=r.clim_id, cal_year0=r.cal_year0, lat=r.lat, soil=r.soil)
  end
  return soa
end
