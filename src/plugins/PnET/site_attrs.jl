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
  1 <= FIXED[].max_layers <= MAXL || error("pnet_generic.max_layers must be in 1:$(MAXL) (scratch arrays are sized for MAXL)")
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
    build_site_attrs(db_path, tablename, splots, eco_list; soil_map=Dict(), default_soil="LOAM", soil_csv="") -> DataFrame

One row per plot_id of `splots`: clim_id (index into CLIMATE[].ids from the plot's epa_l4), cal_year0 (calendar year of the plot's first
measurement = sim year 0), lat (PLOT.LAT), soil (Saxton-Rawls texture code from `soil_map[L3]`, else `default_soil`).

If `soil_csv` is non-empty (yaml `pnet_soil_csv`, produced by tools/build_plot_soil.jl: statecd,unitcd,countycd,plot,porosity,fieldcap,wiltpnt,
rooting_depth_mm,...) the table also carries per-plot `soil_por`, `soil_fc`, `soil_wp`, `soil_rd` (mm); `init_sites!` then uses them instead of the
texture. Plots absent from the CSV (or with non-finite values) get NaN there and fall back to the texture path (count logged). With
`soil_csv == ""` the table is exactly the legacy one (no extra columns).
"""
function build_site_attrs(db_path::AbstractString, tablename::AbstractString, splots::DataFrame, eco_list::Vector{String};
                          soil_map::AbstractDict=Dict{String,String}(), default_soil::AbstractString="LOAM", soil_csv::AbstractString="")
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
  soil_tab = isempty(soil_csv) ? nothing : load_plot_soil(soil_csv)
  idpos = Dict(id => i for (i, id) in enumerate(clim.ids))
  l3pos = Dict{String,Int}()
  for (i, id) in enumerate(clim.ids)
    p = split(id, '.'); length(p) >= 3 && get!(l3pos, join(p[1:3], '.'), i)
  end
  u = unique(select(splots, [:plot_id, :statecd, :unitcd, :countycd, :plot, :eco_id, :start_measdate]))
  g = combine(groupby(u, [:plot_id, :statecd, :unitcd, :countycd, :plot, :eco_id]), :start_measdate => minimum => :sm)
  nmiss_l4 = 0; nmiss_lat = 0; nmiss_soil = 0
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
    row = (plot_id=Int(r.plot_id), clim_id=Int32(cid), cal_year0=Int16(Dates.year(r.sm)), lat=Float32(la), soil=String(get(soil_map, l3e, default_soil)))
    if soil_tab !== nothing
      sv = get(soil_tab, key, nothing)
      sv === nothing && (nmiss_soil += 1)
      row = merge(row, sv === nothing ? (soil_por=NaN, soil_fc=NaN, soil_wp=NaN, soil_rd=NaN) : sv)
    end
    push!(rows, row)
  end
  soil_tab !== nothing && @info "PnET per-plot soil CSV $(basename(soil_csv)): $(length(rows) - nmiss_soil)/$(length(rows)) plots matched; $nmiss_soil fall back to the texture path"
  (nmiss_l4 > 0 || nmiss_lat > 0) && @warn "PnET site attrs: $nmiss_l4 plots had no/unknown L4 in the climate cache (L3 fallback used), $nmiss_lat had no latitude (33 N used)"
  return DataFrame(rows)
end

"""
    load_plot_soil(csv) -> Dict{NTuple{4,Int},NamedTuple}

Read tools/build_plot_soil.jl output: (statecd,unitcd,countycd,plot) -> (soil_por, soil_fc, soil_wp, soil_rd[mm]). Rows with any non-finite
or non-positive value are dropped (their sites use the texture path).
"""
function load_plot_soil(csv::AbstractString)
  isfile(csv) || error("pnet_soil_csv not found: $csv")
  df = CSV.read(csv, DataFrame; select=[:statecd, :unitcd, :countycd, :plot, :porosity, :fieldcap, :wiltpnt, :rooting_depth_mm])
  out = Dict{NTuple{4,Int},NamedTuple{(:soil_por, :soil_fc, :soil_wp, :soil_rd),NTuple{4,Float64}}}()
  for r in eachrow(df)
    v = (soil_por=Float64(coalesce(r.porosity, NaN)), soil_fc=Float64(coalesce(r.fieldcap, NaN)), soil_wp=Float64(coalesce(r.wiltpnt, NaN)), soil_rd=Float64(coalesce(r.rooting_depth_mm, NaN)))
    all(x -> isfinite(x) && x > 0, values(v)) || continue
    out[(Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot))] = v
  end
  return out
end

"""
Set one site's PnET static attributes + initial state (soil bucket full to field capacity, empty snow pack). Soil = texture class `soil`, or,
when `theta = (porosity, fieldcap, wiltpnt)` is given, the per-plot curve of `saxton_rawls_theta` (returns 1 if the invariant needed repair).
"""
function init_site!(soa, i::Int; clim_id::Integer, cal_year0::Integer, lat::Real, soil::AbstractString="LOAM", root_depth::Real=FIXED[].root_depth_base,
                    theta::Union{Nothing,NTuple{3,Float64}}=nothing)
  nrep = 0
  if theta === nothing
    por, fc, wp, A, B = saxton_rawls(soil)
  else
    por, fc, wp, A, B, nrep = saxton_rawls_theta(theta[1], theta[2], theta[3])
  end
  sc = soa.scalar
  sc.clim_id[i] = clim_id; sc.cal_year0[i] = cal_year0; sc.lat[i] = lat
  sc.soil_por[i] = por; sc.soil_fc[i] = fc; sc.soil_wp[i] = wp; sc.soil_A[i] = A; sc.soil_B[i] = B
  sc.rooting_depth[i] = root_depth
  sc.soil_water[i] = fc; sc.snowpack[i] = 0; sc.canopy_lai_prev[i] = 0; sc.subcanopy_par[i] = 0; sc.B[i] = 0
  return nrep
end

"""Plugin hook called by make_sites with the per-plot attribute table (rows keyed by plot_id == site.mapcode)."""
function init_sites!(soa, attrs::DataFrame)
  pos = Dict{Int,Int}(Int(r.plot_id) => k for (k, r) in enumerate(eachrow(attrs)))
  per_plot = hasproperty(attrs, :soil_por)     # only when yaml pnet_soil_csv was set (build_site_attrs soil_csv)
  nplot = 0; ntex = 0; nrep = 0
  for i in 1:soa.n
    k = get(pos, Int(soa.scalar.mapcode[i]), 0)
    k == 0 && error("PnET init_sites!: no site attributes for plot_id $(soa.scalar.mapcode[i])")
    r = attrs[k, :]
    if per_plot && isfinite(r.soil_por) && isfinite(r.soil_fc) && isfinite(r.soil_wp) && isfinite(r.soil_rd)
      # RD_MULT keeps its meaning: eff. rooting depth = rooting_depth[i] * RD_MULT (plugin.jl), now with the plot's depth instead of the constant
      nrep += init_site!(soa, i; clim_id=r.clim_id, cal_year0=r.cal_year0, lat=r.lat, root_depth=r.soil_rd, theta=(Float64(r.soil_por), Float64(r.soil_fc), Float64(r.soil_wp)))
      nplot += 1
    else
      init_site!(soa, i; clim_id=r.clim_id, cal_year0=r.cal_year0, lat=r.lat, soil=r.soil)
      ntex += 1
    end
  end
  per_plot && @info "PnET init_sites!: $nplot sites use per-plot SOLUS soil ($nrep needed invariant repair wiltpnt<fieldcap<porosity / B bounds), $ntex fall back to the texture path"
  return soa
end
