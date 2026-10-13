# Per-plot soil hydraulics for the PnET plugin from SOLUS100 (Saxton-Rawls 2006 derived rasters, hydro4.sh) + restriction depth.
#
#   ./julia_gdal.sh --project=. tools/build_plot_soil.jl --db <slim.duckdb> --out <plot_soil.csv> \
#       [--tablename curated_cohorts_landis_stdorg] [--solus /workspace/SOLUS/SOLUS100] \
#       [--base-depth-mm 1000] [--max-depth-mm 1500] [--min-depth-mm 100]
#
# For every distinct plot (statecd, unitcd, countycd, plot) of PLOT (coordinates = mean LAT/LON over visits):
#   * sample theta_s, theta_33, theta_1500, awc (m3/m3) and ksat (mm/h, hydro4.sh step 5: 1930*(theta_s-theta_33)^(3-lambda)) at the 7 SOLUS
#     depth nodes {0,5,15,30,60,100,150} cm;
#   * restriction depth = min(resdept_all_cm_p, anylithicdpt_cm_p) over the valid ones (UInt8 cm, nodata 255, max value 201 = ">200 cm");
#   * rooting depth D = clamp(restriction, min_depth, max_depth) mm; `base-depth-mm` when no restriction value is valid;
#   * rooting-zone value of each variable = mean over [0, D] of the piecewise-linear profile through the VALID depth nodes (trapezoid rule,
#     constant extrapolation outside the valid-node range); awc_mm = integral of awc over [0, D] (mm of water); no valid node -> missing.
#   * plots with no valid hydraulics are filled with the median of their EPA L4, else L3 (from the cohort table), else the LOAM
#     Saxton-Rawls default; `fill_level` = plot | l4 | l3 | default.
# Output columns: statecd,unitcd,countycd,plot,lat,lon,porosity,fieldcap,wiltpnt,awc_mm,ksat,rooting_depth_mm,n_depths,fill_level
import ArchGDAL as AG
import DuckDB, DataFrames, Statistics
using Pan: PnETPlugin

args = Dict{String,String}(); let i = 1; while i <= length(ARGS); args[ARGS[i][3:end]] = ARGS[i+1]; i += 2; end; end
const DB = args["db"]; const OUT = args["out"]
const TABLE = get(args, "tablename", "curated_cohorts_landis_stdorg")
const SOLUS = get(args, "solus", "/workspace/SOLUS/SOLUS100")
const BASE_MM = parse(Float64, get(args, "base-depth-mm", "1000"))
const MAX_MM = parse(Float64, get(args, "max-depth-mm", "1500"))
const MIN_MM = parse(Float64, get(args, "min-depth-mm", "100"))
const DEPTHS = [0, 5, 15, 30, 60, 100, 150]          # cm, SOLUS nodes
const VARS = ["theta_s", "theta_33", "theta_1500", "awc", "ksat"]
const NODATA = -9999f0
get(ENV, "GDAL_CACHE_MAX", "") == "" && AG.setconfigoption("GDAL_CACHEMAX", "1024")

"""Grid cell (col,row) (0-based) of every lon/lat in the raster at `path`; (-1,-1) outside. Also returns the grid signature."""
function grid_cells(path::String, lon::Vector{Float64}, lat::Vector{Float64})
  AG.read(path) do ds
    gt = AG.getgeotransform(ds); w = AG.width(ds); h = AG.height(ds)
    src = AG.importEPSG(4326; order=:trad); dst = AG.importWKT(AG.getproj(ds))
    cols = fill(-1, length(lon)); rows = fill(-1, length(lon))
    xs = copy(lon); ys = copy(lat); zs = zeros(length(lon))
    AG.createcoordtrans(src, dst) do ct
      AG.transform!(xs, ys, zs, ct)
    end
    for k in eachindex(lon)
      c = floor(Int, (xs[k] - gt[1]) / gt[2]); r = floor(Int, (ys[k] - gt[4]) / gt[6])
      (0 <= c < w && 0 <= r < h) && (cols[k] = c; rows[k] = r)
    end
    (cols, rows, gt, w, h)
  end
end

"""Read raster values at the given cells, visiting cells tile by tile (GDAL block cache reuse). Returns Float64 with NaN for nodata/outside."""
function sample_raster(path::String, cols, rows, order)
  out = fill(NaN, length(cols))
  AG.read(path) do ds
    b = AG.getband(ds, 1); nd = AG.getnodatavalue(b)
    for k in order
      cols[k] < 0 && continue
      v = AG.read(b, cols[k], rows[k], 1, 1)[1]
      (nd !== nothing && Float64(v) == nd) && continue
      out[k] = Float64(v)
    end
  end
  return out
end

"""Mean over [0, D_cm] of the piecewise-linear profile through the valid nodes (constant outside); NaN when no node is valid."""
function zone_mean(vals::AbstractVector, D_cm::Float64)
  idx = [j for j in eachindex(vals) if isfinite(vals[j])]
  isempty(idx) && return NaN
  z = Float64.(DEPTHS[idx]); v = Float64.(vals[idx])
  f(x) = x <= z[1] ? v[1] : x >= z[end] ? v[end] : (j = searchsortedlast(z, x); v[j] + (v[j+1] - v[j]) * (x - z[j]) / (z[j+1] - z[j]))
  pts = unique(sort(vcat(0.0, D_cm, filter(t -> 0 < t < D_cm, z))))
  s = 0.0
  for j in 1:length(pts)-1
    s += 0.5 * (f(pts[j]) + f(pts[j+1])) * (pts[j+1] - pts[j])
  end
  return s / D_cm
end

function main()
  con = DuckDB.DBInterface.connect(DuckDB.DB, ":memory:")
  q(s) = DataFrames.DataFrame(DuckDB.DBInterface.execute(con, s))
  q("ATTACH '$DB' AS s (READ_ONLY)")
  pl = q("SELECT STATECD AS statecd, UNITCD AS unitcd, COUNTYCD AS countycd, PLOT AS plot, avg(LAT) AS lat, avg(LON) AS lon FROM s.PLOT WHERE LAT IS NOT NULL AND LON IS NOT NULL GROUP BY ALL ORDER BY 1,2,3,4")
  eco = q("SELECT statecd, unitcd, countycd, plot, min(epa_l4) AS l4, min(epa_l3) AS l3 FROM s.$TABLE GROUP BY ALL")
  n = DataFrames.nrow(pl)
  @info "plots" n
  lon = Float64.(pl.lon); lat = Float64.(pl.lat)
  rpath(v, d) = joinpath(SOLUS, "hydro", "derived", "$(v)_$(d)_cm.tif")
  cols, rows, gt, w, h = grid_cells(rpath("theta_33", 30), lon, lat)
  order = sortperm(collect(zip(rows .÷ 256, cols .÷ 256)))
  data = Dict(v => fill(NaN, n, length(DEPTHS)) for v in VARS)
  for v in VARS, (j, d) in enumerate(DEPTHS)
    p = rpath(v, d)
    t = @elapsed (data[v][:, j] = sample_raster(p, cols, rows, order))
    @info "sampled" v d valid = count(isfinite, data[v][:, j]) secs = round(t, digits=1)
  end
  rd = [sample_raster(joinpath(SOLUS, f), cols, rows, order) for f in ("resdept_all_cm_p.tif", "anylithicdpt_cm_p.tif")]
  restr = [(a = filter(isfinite, (rd[1][k], rd[2][k])); isempty(a) ? NaN : minimum(a)) for k in 1:n]
  @info "restriction depth" valid = count(isfinite, restr) at201 = count(==(201.0), restr)

  por = fill(NaN, n); fcv = fill(NaN, n); wpv = fill(NaN, n); awcmm = fill(NaN, n); ks = fill(NaN, n); rdmm = fill(NaN, n); nd = zeros(Int, n)
  for k in 1:n
    Dmm = isfinite(restr[k]) ? clamp(restr[k] * 10.0, MIN_MM, MAX_MM) : clamp(BASE_MM, MIN_MM, MAX_MM)
    rdmm[k] = Dmm; Dcm = Dmm / 10
    por[k] = zone_mean(data["theta_s"][k, :], Dcm); fcv[k] = zone_mean(data["theta_33"][k, :], Dcm); wpv[k] = zone_mean(data["theta_1500"][k, :], Dcm)
    a = zone_mean(data["awc"][k, :], Dcm); awcmm[k] = a * Dmm
    ks[k] = zone_mean(data["ksat"][k, :], Dcm)
    nd[k] = count(isfinite, data["theta_33"][k, :])
  end
  df = DataFrames.DataFrame(statecd=pl.statecd, unitcd=pl.unitcd, countycd=pl.countycd, plot=pl.plot, lat=lat, lon=lon, porosity=por, fieldcap=fcv,
    wiltpnt=wpv, awc_mm=awcmm, ksat=ks, rooting_depth_mm=rdmm, n_depths=nd)
  df = DataFrames.leftjoin(df, eco; on=[:statecd, :unitcd, :countycd, :plot])
  df.fill_level = fill("plot", n)
  ok = [all(isfinite, (r.porosity, r.fieldcap, r.wiltpnt)) for r in eachrow(df)]
  cols_fill = [:porosity, :fieldcap, :wiltpnt, :awc_mm, :ksat, :rooting_depth_mm]
  med(sub, c) = (x = filter(isfinite, sub[ok[sub.__i], c]); isempty(x) ? NaN : Statistics.median(x))
  df.__i = 1:n
  l4med = Dict{String,Vector{Float64}}(); l3med = Dict{String,Vector{Float64}}()
  for (key, d, dict) in ((:l4, df, l4med), (:l3, df, l3med))
    for g in DataFrames.groupby(DataFrames.dropmissing(df, key), key)
      dict[String(g[1, key])] = [med(DataFrames.DataFrame(g), c) for c in cols_fill]
    end
  end
  por0, fc0, wp0, _, _ = PnETPlugin.saxton_rawls("LOAM")
  default = [por0, fc0, wp0, 1000 * (fc0 - wp0) * BASE_MM / 1000, NaN, BASE_MM]   # awc_mm over the default rooting depth, no ksat
  default[4] = (fc0 - wp0) * BASE_MM
  for k in 1:n
    ok[k] && continue
    v = nothing; lvl = "default"
    if !ismissing(df.l4[k]) && haskey(l4med, df.l4[k]) && all(isfinite, l4med[df.l4[k]][1:3]); v = l4med[df.l4[k]]; lvl = "l4"
    elseif !ismissing(df.l3[k]) && haskey(l3med, df.l3[k]) && all(isfinite, l3med[df.l3[k]][1:3]); v = l3med[df.l3[k]]; lvl = "l3"
    else v = default end
    for (c, x) in zip(cols_fill, v)
      df[k, c] = x
    end
    df.fill_level[k] = lvl
  end
  # rooting-depth/awc medians can be NaN if all L4 members lacked them; fall back to the default
  for k in 1:n, (ci, c) in enumerate(cols_fill)
    (c in (:ksat,)) && continue
    isfinite(df[k, c]) || (df[k, c] = default[ci])
  end
  out = df[:, [:statecd, :unitcd, :countycd, :plot, :lat, :lon, :porosity, :fieldcap, :wiltpnt, :awc_mm, :ksat, :rooting_depth_mm, :n_depths, :fill_level]]
  open(OUT, "w") do io
    println(io, join(names(out), ","))
    for r in eachrow(out)
      println(io, join((x isa AbstractFloat ? (isfinite(x) ? string(round(x, sigdigits=7)) : "") : string(x) for x in r), ","))
    end
  end
  lv = Dict(l => count(==(l), out.fill_level) for l in ("plot", "l4", "l3", "default"))
  @info "wrote $OUT" n fill_levels = lv
end
main()
