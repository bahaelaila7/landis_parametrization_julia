# Water index of HBAD3P from monthly climate: port of the reference water balance (landis_hbad3 scripts/hbad3/water_balance.py, the computation
# the fit's W and Wn come from).
#   PET (mm/month): Thornthwaite. Per calendar year, heat index I = Σ_m (max(T_m, 0)/5)^1.514 and a = 6.75e-7 I³ − 7.71e-5 I² + 1.792e-2 I + 0.49239;
#                   PET_m = 16 (10 T_m / I)^a · N_m/12 · days_m/30 for T_m > 0 (else 0), N_m = day length (h) at mid-month from the latitude.
#   bucket:         capacity AWC (mm, floored at 5), full at the start; each month W += P; AET = min(PET, W); W = min(AWC, W − AET) (no runoff below capacity).
#   index:          growing season (April–October) ΣAET / ΣPET of a calendar year (1 = no water limitation).
# In the fit the period value is the mean of the annual index over the years of a remeasurement interval (median 5 yr) and the normal Wn is the
# 1991–2020 mean; `trailing_mean` gives a simulated year the same kind of value (mean of its last `window` years).

const _WB_J = (15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349)          # mid-month day of year
const _WB_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
const WB_SPINUP_FROM = 1980                                                     # the reference spins the bucket up from 1980
const WB_NORMAL = 1991:2020

"""Thornthwaite PET (mm/month) of one calendar year: `t` = 12 monthly mean temperatures (°C), `lat` (degrees)."""
function thornthwaite_pet(t::AbstractVector{<:Real}, lat::Real)
  length(t) == 12 || error("thornthwaite_pet: 12 monthly temperatures needed")
  I = sum((max(Float64(x), 0.0) / 5)^1.514 for x in t)
  a = 6.75e-7 * I^3 - 7.71e-5 * I^2 + 1.792e-2 * I + 0.49239
  pet = zeros(12); tl = tan(deg2rad(Float64(lat)))
  for m in 1:12
    tm = max(Float64(t[m]), 0.0)
    tm > 0 || continue
    dec = 0.409 * sin(2π * _WB_J[m] / 365 - 1.39)
    ws = acos(clamp(-tl * tan(dec), -1.0, 1.0)); N = 24 / π * ws
    pet[m] = 16 * (10 * tm / max(I, 1e-9))^a * N / 12 * _WB_DAYS[m] / 30
  end
  return pet
end

"""
    water_index(tmean, ppt, lat, awc) -> Vector (one value per column)

Annual growing-season AET/PET of a monthly soil bucket: `tmean`, `ppt` = 12 × n matrices (°C, mm/month) of consecutive calendar years, the first column
the start of the spin-up (the bucket starts full), `lat` in degrees, `awc` in mm (floored at 5).
"""
function water_index(tmean::AbstractMatrix{<:Real}, ppt::AbstractMatrix{<:Real}, lat::Real, awc::Real)
  size(tmean) == size(ppt) && size(tmean, 1) == 12 || error("water_index: tmean and ppt must both be 12 × n")
  cap = max(Float64(awc), 5.0); W = cap; n = size(tmean, 2); out = zeros(n)
  for y in 1:n
    pet = thornthwaite_pet(view(tmean, :, y), lat); sa = 0.0; se = 0.0
    for m in 1:12
      W += Float64(ppt[m, y]); aet = min(pet[m], W); W = min(cap, W - aet)
      if 4 <= m <= 10
        sa += aet; se += pet[m]
      end
    end
    out[y] = sa / max(se, 1e-9)
  end
  return out
end

"""Mean of the last `window` annual values ending at each year (fewer at the start of the series)."""
function trailing_mean(x::AbstractVector{<:Real}, window::Integer)
  window >= 1 || error("trailing_mean: window must be >= 1")
  out = similar(x, Float64)
  for i in eachindex(x)
    lo = max(firstindex(x), i - window + 1); out[i] = sum(@view x[lo:i]) / (i - lo + 1)
  end
  return out
end

"""Plot normal: mean of the annual index over 1991–2020 (`years` = the calendar years of `w`)."""
function water_normal(w::AbstractVector{<:Real}, years::AbstractVector{<:Integer})
  v = [w[i] for (i, y) in enumerate(years) if y in WB_NORMAL]
  isempty(v) && error("water_normal: the series does not cover 1991–2020")
  return sum(v) / length(v)
end
