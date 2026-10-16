# Location-based site attributes for scenario runs (docs/scenario_engine.md §2.1).
# In Forest Explorer a "plot" is a TreeMap imputation: the FIA plot supplies the VEGETATION only. Everything that belongs to
# the place — EPA L3/L4 ecoregion (parameter eco cell, PnET climate), soils (PnET water holding) and FIA site class (eco cell
# split) — comes from where the AOI pixels are. The caller passes, per plot, the majority L4 of its pixels and their centroid.
import ArchGDAL

const SITE_K = 7                     # nearest FIA plots voting on the site class

"EPA L3 code of an L4 id such as \"8.3.4.45b\" → \"8.3.4\" (\"\" if malformed)."
l3_of_l4(l4::AbstractString) = (p = split(String(l4), "."); length(p) >= 4 ? join(p[1:3], ".") : "")

# ---------------------------------------------------------------------------------------------------------------------
# Soils: SOLUS100 rooting-zone hydraulics at a point (same procedure as tools/build_plot_soil.jl)
# ---------------------------------------------------------------------------------------------------------------------
const _SOLUS_DEPTHS = (0, 5, 15, 30, 60, 100, 150)       # cm

function _solus_cells(path::String, lon::Vector{Float64}, lat::Vector{Float64})
  ArchGDAL.read(path) do ds
    gt = ArchGDAL.getgeotransform(ds); w = ArchGDAL.width(ds); h = ArchGDAL.height(ds)
    xs = copy(lon); ys = copy(lat); zs = zeros(length(lon))
    ArchGDAL.createcoordtrans(ArchGDAL.importEPSG(4326; order=:trad), ArchGDAL.importWKT(ArchGDAL.getproj(ds))) do ct
      ArchGDAL.transform!(xs, ys, zs, ct)
    end
    cols = fill(-1, length(lon)); rows = fill(-1, length(lon))
    for k in eachindex(lon)
      c = floor(Int, (xs[k] - gt[1]) / gt[2]); r = floor(Int, (ys[k] - gt[4]) / gt[6])
      (0 <= c < w && 0 <= r < h) && (cols[k] = c; rows[k] = r)
    end
    cols, rows
  end
end

function _solus_sample(path::String, cols, rows)
  out = fill(NaN, length(cols))
  isfile(path) || return out
  ArchGDAL.read(path) do ds
    b = ArchGDAL.getband(ds, 1); nd = ArchGDAL.getnodatavalue(b)
    for k in eachindex(cols)
      cols[k] < 0 && continue
      v = Float64(ArchGDAL.read(b, cols[k], rows[k], 1, 1)[1])
      (nd !== nothing && v == nd) || (out[k] = v)
    end
  end
  out
end

"Mean over [0, D] cm of the piecewise-linear profile through the valid depth nodes (constant outside); NaN if none valid."
function _zone_mean(vals, D::Float64)
  idx = [j for j in eachindex(vals) if isfinite(vals[j])]
  isempty(idx) && return NaN
  z = Float64[_SOLUS_DEPTHS[j] for j in idx]; v = Float64[vals[j] for j in idx]
  f(x) = x <= z[1] ? v[1] : x >= z[end] ? v[end] : (j = searchsortedlast(z, x); v[j] + (v[j+1] - v[j]) * (x - z[j]) / (z[j+1] - z[j]))
  pts = unique(sort(vcat(0.0, D, filter(t -> 0 < t < D, z))))
  sum(0.5 * (f(pts[j]) + f(pts[j+1])) * (pts[j+1] - pts[j]) for j in 1:length(pts)-1) / D
end

"""
    soil_at(scen, lon, lat) -> Vector{Union{Nothing, (soil_por, soil_fc, soil_wp, soil_rd)}}

PnET soil water parameters (volumetric θ_s, θ_33, θ_1500 averaged over the rooting zone; rooting depth mm) from SOLUS100 at
each point; `nothing` where SOLUS has no valid hydraulics (water, urban, outside CONUS) or the rasters are not available.
Rooting depth = restriction depth (min of resdept_all / anylithicdpt) clamped to 100–1500 mm, 1000 mm if unknown.
"""
function soil_at(scen::ScenarioConfig, lon::Vector{Float64}, lat::Vector{Float64}; base_mm=1000.0, max_mm=1500.0, min_mm=100.0)
  n = length(lon)
  out = Vector{Union{Nothing,NamedTuple{(:soil_por, :soil_fc, :soil_wp, :soil_rd),NTuple{4,Float64}}}}(nothing, n)
  rp(v, d) = joinpath(scen.solus, "hydro", "derived", "$(v)_$(d)_cm.tif")
  (n == 0 || !isfile(rp("theta_33", 30))) && return out
  cols, rows = _solus_cells(rp("theta_33", 30), lon, lat)
  prof(v) = reduce(hcat, [_solus_sample(rp(v, d), cols, rows) for d in _SOLUS_DEPTHS])       # n × 7
  ts, t33, t15 = prof("theta_s"), prof("theta_33"), prof("theta_1500")
  rd = [_solus_sample(joinpath(scen.solus, f), cols, rows) for f in ("resdept_all_cm_p.tif", "anylithicdpt_cm_p.tif")]
  for k in 1:n
    r = filter(isfinite, (rd[1][k], rd[2][k]))
    Dmm = isempty(r) ? clamp(base_mm, min_mm, max_mm) : clamp(minimum(r) * 10.0, min_mm, max_mm)
    v = (soil_por=_zone_mean(ts[k, :], Dmm / 10), soil_fc=_zone_mean(t33[k, :], Dmm / 10), soil_wp=_zone_mean(t15[k, :], Dmm / 10), soil_rd=Dmm)
    all(x -> isfinite(x) && x > 0, values(v)) && (out[k] = v)
  end
  out
end

# ---------------------------------------------------------------------------------------------------------------------
# Site class: inverse-distance vote of the SITE_K nearest FIA plots (actual plot coordinates) in the same EPA L3
# ---------------------------------------------------------------------------------------------------------------------
_site_table(scen::ScenarioConfig) = get_cached!(scen.cache, :site_table) do
  db = DuckDB.DB(); con = DuckDB.DBInterface.connect(db)
  try
    DuckDB.execute(con, "ATTACH '$(scen.fia_db)' AS src (READ_ONLY)")
    df = DataFrame(DuckDB.DBInterface.execute(con, """
      WITH k AS (SELECT plt_cn AS key, any_value(epa_l3) AS l3 FROM src.$(scen.cohort_table) GROUP BY plt_cn),
           p AS (SELECT CN, STATECD||'_'||UNITCD||'_'||COUNTYCD||'_'||PLOT AS key, LAT, LON FROM src.PLOT WHERE LAT IS NOT NULL),
           pk AS (SELECT p.* FROM p JOIN k USING (key)),
           c AS (SELECT pk.key, co.SITECLCD AS sc, SUM(co.CONDPROP_UNADJ) AS w FROM src.COND co JOIN pk ON co.PLT_CN = pk.CN
                 WHERE co.COND_STATUS_CD = 1 AND co.SITECLCD IS NOT NULL GROUP BY ALL),
           r AS (SELECT key, sc, ROW_NUMBER() OVER (PARTITION BY key ORDER BY w DESC, sc) AS rn FROM c),
           ll AS (SELECT key, avg(LAT) AS lat, avg(LON) AS lon FROM pk GROUP BY key)
      SELECT k.l3, ll.lat, ll.lon, r.sc FROM k JOIN ll USING (key) JOIN r USING (key) WHERE r.rn = 1"""))
    byl3 = Dict{String,NamedTuple}()
    for g in groupby(df, :l3)
      ismissing(g.l3[1]) && continue
      byl3[String(g.l3[1])] = (lat=Float64.(g.lat), lon=Float64.(g.lon), sc=Int.(g.sc))
    end
    byl3
  finally
    DuckDB.DBInterface.close!(con); DuckDB.close(db)
  end
end

"FIA site class (SITECLCD 1–7) at a location: inverse-distance vote of the nearest FIA plots of the same L3; 0 if none."
function site_class_at(scen::ScenarioConfig, l3::AbstractString, lon::Real, lat::Real)
  t = get(_site_table(scen), String(l3), nothing)
  (t === nothing || !isfinite(lon) || !isfinite(lat)) && return 0
  kx = cosd(lat)
  d = @. (kx * (t.lon - lon))^2 + (t.lat - lat)^2
  idx = partialsortperm(d, 1:min(SITE_K, length(d)))
  votes = zeros(7)
  for i in idx
    1 <= t.sc[i] <= 7 && (votes[t.sc[i]] += 1 / (sqrt(d[i]) + 0.01))     # ~1 km floor on the distance (degrees)
  end
  maximum(votes) > 0 ? argmax(votes) : 0
end
