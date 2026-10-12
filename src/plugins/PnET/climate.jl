# PnET climate: PRISM monthly cache -> dense derived Float32 array, CO2 series, and the lat-dependent terms
# (daylength, PAR) that PRISM does not provide.
#
# Derived array  D[v, month, yearidx, clim_id]  (Float32), v = DV_* below. Only climate-intrinsic terms are tabulated;
# daylength (needs latitude) and PAR (needs latitude + diurnal range) are computed per SITE-YEAR from `site.lat`
# (12 trig/exp evaluations -- negligible next to the 12*IMAX*ncohort sublayer steps) so that every plot keeps its own latitude
# even when many plots share one L4 climate series.
#
# Calendar year of sim year t at a site = cal_year0 + t, clamped to [first year, last year] of the cache
# (years outside the cache repeat the nearest cached year; documented deviation).

const DV_TMIN = 1
const DV_TMAX = 2
const DV_TAVE = 3     # 0.5*(Tmin+Tmax)                         [EPV.cs:313]
const DV_TDAY = 4     # 0.5*(Tmax+Tave)                         [EPV.cs:321]
const DV_PREC = 5     # mm/month
const DV_CO2 = 6      # ppm (annual series, broadcast over months)
const DV_VPD = 7      # kPa: es(Tday) - e(Tmin)                 [EPV.cs:166-196]
const DV_DTR = 8      # Tmax - Tmin (Bristow-Campbell input)
const N_DV = 8

# Mid-month (15th) day of year, non-leap, and days per month (Feb = 28, EPV.cs:149-163)
const MID_DOY = (15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349)
const DAYSPAN = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

# Radiation derivation constants (PRISM provides no radiation)
const BC_TAU_MAX = 0.7          # Bristow & Campbell (1984) clear-sky transmissivity coefficient A
const BC_TAU_B = 0.004          # B
const BC_TAU_C = 2.4            # C
const FAO_GSC = 0.0820          # solar constant, MJ m-2 min-1 (FAO-56 eq. 21)
const PAR_FRAC = 0.45           # PAR fraction of global shortwave
const PAR_UMOL_PER_J = 4.57     # umol photons per J of PAR (McCree 1972)

struct PnETClimate
  ids::Vector{String}
  y1::Int
  y2::Int
  D::Array{Float32,4}   # (N_DV, 12, ny, ncl)
end

const CLIMATE = Ref{Union{Nothing,PnETClimate}}(nothing)

@inline _vp(a, b, c, T) = a * exp(b * T / (T + c))
function vpd_kpa(tday::Real, tmin::Real)
  es = tday < 0 ? _vp(0.61078, 21.87456, 265.5, tday) : _vp(0.61078, 17.26939, 237.3, tday)
  em = tmin < 0 ? _vp(0.61078, 21.87456, 265.5, tmin) : _vp(0.61078, 17.26939, 237.3, tmin)
  return es - em
end

"""Build the derived array from raw tmin/tmax/ppt (shape [month, year, clim]) and an annual CO2 vector aligned with years."""
function build_climate(ids::Vector{String}, years::AbstractVector{<:Integer}, tmin::AbstractArray{<:Real,3}, tmax::AbstractArray{<:Real,3},
                       ppt::AbstractArray{<:Real,3}, co2_by_year::AbstractVector{<:Real})
  ny = length(years); ncl = length(ids)
  D = Array{Float32,4}(undef, N_DV, 12, ny, ncl)
  for c in 1:ncl, y in 1:ny, m in 1:12
    tn = Float64(tmin[m, y, c]); tx = Float64(tmax[m, y, c])
    ta = 0.5 * (tn + tx); td = 0.5 * (tx + ta)
    D[DV_TMIN, m, y, c] = tn; D[DV_TMAX, m, y, c] = tx; D[DV_TAVE, m, y, c] = ta; D[DV_TDAY, m, y, c] = td
    D[DV_PREC, m, y, c] = max(0.0, Float64(ppt[m, y, c])); D[DV_CO2, m, y, c] = co2_by_year[y]
    D[DV_VPD, m, y, c] = vpd_kpa(td, tn); D[DV_DTR, m, y, c] = tx - tn
  end
  return PnETClimate(ids, Int(first(years)), Int(last(years)), D)
end

"""Load the PRISM L4 cache (JLD2: CLIM[var, month, year, l4], ids, years, vars) + annual CO2 csv (year,co2_ppm)."""
function load_climate(cache_path::AbstractString, co2_csv::AbstractString)
  d = JLD2.load(String(cache_path))
  vars = d["vars"]; C = d["CLIM"]; ids = String.(d["ids"]); years = Int.(d["years"])
  iv(n) = (i = findfirst(==(n), vars); i === nothing ? error("climate cache lacks variable $n") : i)
  tmin = Array{Float32,3}(view(C, iv("tmin"), :, :, :))
  tmax = Array{Float32,3}(view(C, iv("tmax"), :, :, :))
  ppt = Array{Float32,3}(view(C, iv("ppt"), :, :, :))
  co2tbl = Dict{Int,Float64}()
  for ln in eachline(String(co2_csv))
    f = split(strip(ln), ',')
    length(f) >= 2 || continue
    y = tryparse(Int, f[1]); c = tryparse(Float64, f[2])
    (y === nothing || c === nothing) && continue
    co2tbl[y] = c
  end
  isempty(co2tbl) && error("no CO2 rows parsed from $co2_csv")
  ylo, yhi = extrema(keys(co2tbl))
  co2 = Float64[co2tbl[clamp(y, ylo, yhi)] for y in years]   # the csv has every year ylo:yhi
  return build_climate(ids, years, tmin, tmax, ppt, co2)
end

"""Latitude-dependent per-month terms for one site-year: daylength hours, and extraterrestrial radiation (MJ/m2/day)."""
function daylength_ra!(hr::AbstractVector{Float64}, ra::AbstractVector{Float64}, lat_deg::Float64)
  phi = lat_deg * 2pi / 360
  for m in 1:12
    doy = MID_DOY[m]
    z = 0.39785 * sin(4.868961 + 0.017203 * doy + 0.033446 * sin(6.224111 + 0.017202 * doy))
    decl = asin(clamp(z, -1.0, 1.0))                 # == atan(z/sqrt(1-z^2)) of EPV.cs:255-292
    z2 = -tan(decl) * tan(phi)
    h = z2 >= 1 ? 0.0 : (z2 <= -1 ? pi : acos(z2))  # sunset hour angle
    hr[m] = 24 * h / pi
    dr = 1 + 0.033 * cos(2pi * doy / 365)            # FAO-56 eq. 23
    ra[m] = (24 * 60 / pi) * FAO_GSC * dr * (h * sin(phi) * sin(decl) + cos(phi) * cos(decl) * sin(h))
  end
  return nothing
end

"""Daytime-mean PAR (umol m-2 s-1) from Bristow-Campbell transmissivity and FAO-56 extraterrestrial radiation."""
@inline function par_daytime(ra::Float64, dtr::Float64, daylen_s::Float64)
  tau = BC_TAU_MAX * (1 - exp(-BC_TAU_B * max(dtr, 0.0)^BC_TAU_C))
  rs = tau * ra
  return daylen_s > 0 ? rs * 1e6 * PAR_FRAC * PAR_UMOL_PER_J / daylen_s : 0.0
end
