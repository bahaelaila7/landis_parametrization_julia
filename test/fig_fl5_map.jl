# FL5-area map: EPA L3 (solid black), L4 (dashed blue), counties (dashed white), FIA plot dots colored by
# land-use (artificial/natural), over a DRYBIO_L biomass background (TreeMap, added in stage 2). All in
# EPSG:4326. Plots span the full 4-L4 extent (the plots used in this work).
#   Run: ./julia_gdal.sh --project=. test/fig_fl5_map.jl
using DuckDB, DataFrames, CairoMakie, Statistics
import ArchGDAL as AG
const MK = CairoMakie
const DB = "/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"
const L4SHP = "/workspace/EPA_ECOREGIONS/us_eco_l4/us_eco_l4_no_st.shp"
const COUNTYSHP = "/workspace/TIGER/tl_2024_us_county/tl_2024_us_county.shp"
const L4S = Set(["65o", "75g", "75e", "75f"]); const L3S = Set(["65", "75"])
const OUT = "runs/fig_fl5_map.png"
const WGS84 = AG.importEPSG(4326)

# exterior-ring lon/lat coords of a (multi)polygon, after reprojecting to 4326
function rings4326(geom, srcsr)
  g = AG.clone(geom)
  AG.createcoordtrans(srcsr, WGS84) do trans
    AG.transform!(g, trans)
  end
  out = Vector{Vector{Point2f}}()
  function rec(h)
    t = AG.getgeomtype(h)
    if t in (AG.wkbPolygon, AG.wkbPolygon25D)
      e = AG.getgeom(h, 0)
      # target 4326 is authority order (lat,lon) → getx=lat, gety=lon → emit (lon,lat)
      push!(out, [Point2f(AG.gety(e, i), AG.getx(e, i)) for i in 0:AG.ngeom(e)-1])
    elseif t in (AG.wkbMultiPolygon, AG.wkbMultiPolygon25D, AG.wkbGeometryCollection)
      for i in 0:AG.ngeom(h)-1; rec(AG.getgeom(h, i)); end
    end
  end
  rec(g); out
end

# --- load L4 (our 4) + dissolve L3 (65,75) within a generous bbox ---
ds = AG.read(L4SHP); lyr = AG.getlayer(ds, 0); sr4 = AG.getspatialref(lyr)
l4_rings = Vector{Vector{Point2f}}(); l3_geoms = AG.IGeometry[]
function loadL4!()
  for f in lyr
    code4 = String(AG.getfield(f, "US_L4CODE")); code3 = String(AG.getfield(f, "US_L3CODE"))
    if code4 in L4S
      append!(l4_rings, rings4326(AG.getgeom(f), sr4))
    end
    code3 in L3S && push!(l3_geoms, AG.clone(AG.getgeom(f)))   # all of L3 65/75 for dissolve
  end
end
loadL4!()
println("L4 rings (our 4): ", length(l4_rings), " | L3 polys to dissolve: ", length(l3_geoms))

# extent = bbox of our 4 L4 rings (+pad)
allpts = reduce(vcat, l4_rings)
lon0, lon1 = extrema(p -> p[1], allpts); lat0, lat1 = extrema(p -> p[2], allpts)
padx = 0.04 * (lon1 - lon0); pady = 0.04 * (lat1 - lat0)
ext = (lon0 - padx, lon1 + padx, lat0 - pady, lat1 + pady)
println("extent (lon/lat): ", round.(ext, digits=3))

# dissolve L3 (union) then rings — kept SEPARATE per code so each L3 gets its own shade + legend entry.
l3rings = Dict{String,Vector{Vector{Point2f}}}()
for grp in L3S
  geoms = AG.IGeometry[]
  rewind = AG.read(L4SHP); ly = AG.getlayer(rewind, 0)
  for f in ly
    String(AG.getfield(f, "US_L3CODE")) == grp && push!(geoms, AG.clone(AG.getgeom(f)))
  end
  isempty(geoms) && continue
  u = reduce(AG.union, geoms)
  l3rings[grp] = rings4326(u, sr4)
end
# US_L3CODE → (fill colour, legend name).  65 = NA 8.3.5 Southeastern Plains, 75 = NA 8.5.3 Southern Coastal Plain
l3info = Dict("65" => (:purple, "8.3.5  Southeastern Plains"), "75" => (:darkorange, "8.5.3  Southern Coastal Plain"))
println("L3 dissolved rings: ", Dict(k => length(v) for (k, v) in l3rings))

# --- counties (dashed cyan, in extent) + states (solid cyan, dissolved by STATEFP) ---
cds = AG.read(COUNTYSHP); cly = AG.getlayer(cds, 0); csr = AG.getspatialref(cly)
county_rings = Vector{Vector{Point2f}}(); statefps = Set{String}()
for f in cly
  g = AG.getgeom(f); env = AG.envelope(g)   # native ≈ (lon,lat)
  (env.MaxX < ext[1] || env.MinX > ext[2] || env.MaxY < ext[3] || env.MinY > ext[4]) && continue
  append!(county_rings, rings4326(g, csr)); push!(statefps, String(AG.getfield(f, "STATEFP")))
end
# dissolve FULL states that appear in the extent → true state borders (clipped to view at render)
state_rings = Vector{Vector{Point2f}}()
cly2 = AG.getlayer(AG.read(COUNTYSHP), 0); sgeoms = Dict{String,Vector{AG.IGeometry}}()
for f in cly2
  st = String(AG.getfield(f, "STATEFP"))
  st in statefps && push!(get!(sgeoms, st, AG.IGeometry[]), AG.clone(AG.getgeom(f)))
end
for (st, gs) in sgeoms; append!(state_rings, rings4326(reduce(AG.union, gs), csr)); end
println("counties in extent: ", length(county_rings), " | states: ", sort(collect(keys(sgeoms))), " state rings: ", length(state_rings))

# --- FIA plots: lon/lat + land_use (our plots) ---
con = DBInterface.connect(DuckDB.DB(DB))
plots = DataFrame(DBInterface.execute(con, """
  SELECT CAST(e.lon AS DOUBLE) lon, CAST(e.lat AS DOUBLE) lat, MAX(c.land_use) land_use
  FROM curated_cohorts_landis c JOIN data_plot_eco e
    ON c.statecd=e.statecd AND c.unitcd=e.unitcd AND c.countycd=e.countycd AND c.plot=e.plot
  WHERE c.epa_l4 IN ('8.3.5.65o','8.5.3.75g','8.5.3.75e','8.5.3.75f')
  GROUP BY e.lon, e.lat
"""))
println("plots: ", nrow(plots), "  landuse: ", unique(plots.land_use))
lucol = Dict("artificial" => :crimson, "natural" => :royalblue)   # non-green → visible over the green biomass

# --- biomass background: TreeMap DRYBIO_L (field 22 in RAT), remap band1, warp Albers→4326 ---
const TREEMAP = "/workspace/poster/fl5/FL5_22.tif"
const DRYBIO_FIELD = 22   # 0=Value,1=TM_ID,2=PLT_CN,...,22=DRYBIO_L
const PIX_TONS = 900 / 4046.8564   # 30×30 m pixel = 900 m² in acres; DRYBIO_L (tons/acre) × this = tons/pixel
function parse_drybio(auxpath)
  xml = read(auxpath, String)
  d = Dict{UInt32,Float32}()
  for chunk in split(xml, "<Row index=")[2:end]
    fs = [m.captures[1] for m in eachmatch(r"<F>([^<]*)</F>", chunk)]
    length(fs) > DRYBIO_FIELD || continue
    v = tryparse(Int, fs[1]); db = tryparse(Float64, fs[DRYBIO_FIELD+1])
    (v === nothing || db === nothing || db <= 0) && continue
    d[UInt32(v)] = Float32(db * PIX_TONS)   # tons/acre → tons per 30×30 m pixel
  end
  d
end
drybio = parse_drybio(TREEMAP * ".aux.xml")
println("RAT DRYBIO_L entries: ", length(drybio))
rds = AG.read(TREEMAP)
D = 4                                   # decimate for a background
ids = AG.read(AG.getband(rds, 1))[1:D:end, 1:D:end]   # (width,height) UInt32
bio = [get(drybio, UInt32(ids[i, j]), NaN32) for i in axes(ids, 1), j in axes(ids, 2)]
gt = collect(AG.getgeotransform(rds)); gt[2] *= D; gt[6] *= D
mem = AG.create(AG.getdriver("MEM"); width = size(bio, 1), height = size(bio, 2), nbands = 1, dtype = Float32)
AG.setgeotransform!(mem, gt); AG.setproj!(mem, AG.getproj(rds))
AG.write!(mem, bio, 1); AG.setnodatavalue!(AG.getband(mem, 1), NaN)
wb, wgt = AG.gdalwarp([mem], ["-t_srs", "EPSG:4326", "-r", "near"]) do warp
  (copy(AG.read(AG.getband(warp, 1))), collect(AG.getgeotransform(warp)))
end
wnx, wny = size(wb)
blon = [wgt[1] + (i - 0.5) * wgt[2] for i in 1:wnx]
blat = [wgt[4] + (j - 0.5) * wgt[6] for j in 1:wny]
println("biomass warped: ", wnx, "x", wny, "  lon ", round(minimum(blon), digits=2), "..", round(maximum(blon), digits=2),
        "  DRYBIO range ", round(minimum(x -> isnan(x) ? Inf : x, wb), digits=1), "..", round(maximum(x -> isnan(x) ? -Inf : x, wb), digits=1))

# --- render ---
fig = MK.Figure(size = (1000, 900))
ax = MK.Axis(fig[1, 1]; xlabel = "lon", ylabel = "lat", title = "FL5 area — Aboveground Biomass (Live), EPA L3/L4, counties, FIA plots by land-use",
  aspect = MK.DataAspect())
hm = MK.heatmap!(ax, blon, blat, wb; colormap = :YlGn, colorrange = (0, quantile(filter(!isnan, vec(wb)), 0.98)))
MK.Colorbar(fig[1, 2], hm; label = "Aboveground Biomass (Live)  (tons / 30×30 m pixel)")
for (grp, rings) in l3rings, r in rings                                                            # L3 shade fill, 0.15 opacity
  MK.poly!(ax, r; color = (l3info[grp][1], 0.15), strokewidth = 0)
end
for r in state_rings; MK.lines!(ax, r; color = :cyan, linewidth = 2.5); end                       # state: solid cyan
for r in county_rings; MK.lines!(ax, r; color = (:cyan, 0.85), linewidth = 0.9, linestyle = :dash); end  # county: dashed cyan
for r in l4_rings; MK.lines!(ax, r; color = :blue, linewidth = 1.4, linestyle = :dash); end
for (grp, rings) in l3rings, r in rings; MK.lines!(ax, r; color = :black, linewidth = 2.0); end   # L3 outline
for (lu, col) in lucol
  sub = plots[plots.land_use .== lu, :]
  MK.scatter!(ax, Float64.(sub.lon), Float64.(sub.lat); color = col, markersize = 5)
end
MK.xlims!(ax, ext[1], ext[2]); MK.ylims!(ax, ext[3], ext[4])
MK.axislegend(ax,
  [MK.MarkerElement(color = lucol["artificial"], marker = :circle), MK.MarkerElement(color = lucol["natural"], marker = :circle),
   MK.PolyElement(color = (l3info["65"][1], 0.35)), MK.PolyElement(color = (l3info["75"][1], 0.35))],
  ["artificial", "natural", l3info["65"][2], l3info["75"][2]];
  position = :rt, framevisible = true, backgroundcolor = (:white, 0.85))
MK.save(OUT, fig; px_per_unit = 3)   # ~3× DPI
println("wrote ", OUT)
