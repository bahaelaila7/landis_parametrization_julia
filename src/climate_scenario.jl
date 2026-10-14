"""
    ClimateScenario

Monthly climate baselines + scenario modifications + CO2 series for PnET-style simulators.
Pure Julia, stdlib only. Monthly series are `Matrix{Float64}` of size (12, n_years); the year axis is a
separate `Vector{Int}`. Units are the caller's (the pan_climate DB uses degC and mm/month).

Exports: `baseline`, `apply_changes`, `apply_scenario`, `build_climate`, `co2_series`.
"""
module ClimateScenario

using Statistics

export baseline, apply_changes, apply_scenario, build_climate, co2_series, SSP245_CO2, SSP585_CO2

# ---------------------------------------------------------------------------------------------
# helpers: uniform access to NamedTuple / Dict (String or Symbol keys) scenario objects
# ---------------------------------------------------------------------------------------------
_get(x::NamedTuple, k::Symbol, d=nothing) = haskey(x, k) ? getfield(x, k) : d
function _get(x::AbstractDict, k::Symbol, d=nothing)
    haskey(x, String(k)) && return x[String(k)]
    haskey(x, k) && return x[k]
    return d
end
_isnothing(x) = x === nothing || x === missing

function _colindex(years::AbstractVector{<:Integer})
    Dict(Int(y) => i for (i, y) in enumerate(years))
end

# ---------------------------------------------------------------------------------------------
# 1. baseline
# ---------------------------------------------------------------------------------------------
"""
    baseline(obs, obs_years, fc, fc_years, mode; T=30, K=3, start_year, end_year) -> Matrix (12, nyears)

Build a monthly baseline for `start_year:end_year`. `obs_years[end]` is the last observed year `L`.
In every mode, years that are observed (`y <= L`, present in `obs_years`) are returned as observed.
For years `y > L`:

* `:forecast` - column from `fc` (`fc_years` must cover y; error otherwise). `fc` may be `nothing` only
  if no year > L is requested.
* `:repeat`   - repeat the last `T` observed years cyclically: year `L+j` (j >= 1) = observed year
  `L - T + 1 + mod(j-1, T)`.
* `:average`  - let window k (k = 1..K) be the T observed years `L-kT+1 : L-(k-1)T` (k=1 is the most recent).
  Year-in-window i (1..T) has value mean_k obs[window_k[i]]. Year `L+j` uses i = mod(j-1, T)+1, so the
  first future year gets the mean of the first years of the K windows, etc. If the record is shorter
  than K*T, K is reduced to floor(n/T) (error if < 1).
"""
function baseline(obs::AbstractMatrix, obs_years::AbstractVector{<:Integer},
                  fc::Union{Nothing,AbstractMatrix}, fc_years::Union{Nothing,AbstractVector{<:Integer}},
                  mode::Symbol; T::Integer=30, K::Integer=3, start_year::Integer, end_year::Integer)
    size(obs, 1) == 12 || throw(ArgumentError("obs must be (12, n_years)"))
    size(obs, 2) == length(obs_years) || throw(ArgumentError("obs columns != length(obs_years)"))
    issorted(obs_years) || throw(ArgumentError("obs_years must be sorted"))
    end_year >= start_year || throw(ArgumentError("end_year < start_year"))
    mode in (:forecast, :repeat, :average) || throw(ArgumentError("unknown baseline mode $mode"))
    oidx = _colindex(obs_years)
    L = Int(obs_years[end]); n = length(obs_years)
    fidx = (fc === nothing || fc_years === nothing) ? Dict{Int,Int}() : _colindex(fc_years)
    out = Matrix{Float64}(undef, 12, end_year - start_year + 1)

    # precompute windows for repeat/average
    local rep, avg
    if mode === :repeat
        T <= n || throw(ArgumentError("T=$T > number of observed years $n"))
        rep = obs[:, n-T+1:n]
    elseif mode === :average
        Ke = min(K, n ÷ T)
        Ke >= 1 || throw(ArgumentError("need at least T=$T observed years"))
        avg = zeros(12, T)
        for k in 1:Ke
            avg .+= obs[:, n-k*T+1:n-(k-1)*T]
        end
        avg ./= Ke
    end

    for (c, y) in enumerate(start_year:end_year)
        if haskey(oidx, y)
            out[:, c] = obs[:, oidx[y]]
        elseif y < obs_years[1]
            throw(ArgumentError("year $y precedes the observed record"))
        elseif y <= L
            throw(ArgumentError("year $y is a gap inside the observed record"))
        else
            j = y - L
            if mode === :forecast
                haskey(fidx, y) || throw(ArgumentError("forecast does not cover year $y"))
                out[:, c] = fc[:, fidx[y]]
            elseif mode === :repeat
                out[:, c] = rep[:, mod(j - 1, T) + 1]
            else
                out[:, c] = avg[:, mod(j - 1, T) + 1]
            end
        end
    end
    return out
end

# ---------------------------------------------------------------------------------------------
# 2. changes
# ---------------------------------------------------------------------------------------------
function _frac(shape::AbstractString, y::Integer, sy::Integer, ey::Integer)
    y < sy && return 0.0
    if shape == "step"
        return 1.0
    elseif shape == "ramp"
        ey <= sy && return 1.0
        return clamp((y - sy) / (ey - sy), 0.0, 1.0)
    else
        throw(ArgumentError("unknown shape '$shape' (use \"step\" or \"ramp\")"))
    end
end

function _apply_one!(clim::Dict{String,Matrix{Float64}}, years, ch)
    var   = String(_get(ch, :var))
    kind  = String(_get(ch, :kind))
    value = Float64(_get(ch, :value))
    shape = String(_get(ch, :shape, "step"))
    sy    = Int(_get(ch, :start_year, first(years)))
    ey    = Int(_get(ch, :end_year, last(years)))
    months = _get(ch, :months, nothing)
    ms = _isnothing(months) ? collect(1:12) : Int.(collect(months))
    all(1 .<= ms .<= 12) || throw(ArgumentError("months must be in 1..12"))
    kind in ("add", "mul") || throw(ArgumentError("unknown kind '$kind' (use \"add\" or \"mul\")"))
    targets = var == "tmean" ? ["tmin", "tmax"] : [var]
    if var == "tmean" && kind == "mul"
        throw(ArgumentError("kind=\"mul\" is not supported for tmean (a multiplicative change on degC is ill-defined); use tmin/tmax or add"))
    end
    var in ("tmean", "tmin", "tmax", "ppt") || throw(ArgumentError("unsupported var '$var'"))
    for t in targets
        haskey(clim, t) || throw(ArgumentError("series for '$t' missing from climate dict"))
        M = clim[t]
        for (c, y) in enumerate(years)
            f = _frac(shape, y, sy, ey)
            f == 0.0 && continue
            if kind == "add"
                @views M[ms, c] .+= value * f
            else
                @views M[ms, c] .*= 1.0 + value / 100.0 * f
            end
        end
    end
    return clim
end

"""
    apply_changes(series::Dict{String,Matrix{Float64}}, years, changes) -> Dict (new, inputs untouched)

Apply each change (NamedTuple or Dict with `var, kind, value, shape, start_year, end_year, months`) in order.
* `kind="add"`: `x += value*f`;  `kind="mul"`: `x *= 1 + value/100*f` (value in percent, +10 => x1.10 at full effect).
* `f` = 0 before `start_year`; `shape="step"`: f=1 from start_year on; `shape="ramp"`: f rises linearly from 0 at
  start_year to 1 at end_year and stays 1 afterwards (end_year<=start_year acts as a step).
* `months`: vector of 1..12 or `nothing` (all months).
* `var="tmean"` (add only) shifts tmin AND tmax by the same amount.
No physical clamping here; see `apply_scenario`.
"""
function apply_changes(series::AbstractDict, years::AbstractVector{<:Integer}, changes)
    clim = Dict{String,Matrix{Float64}}(String(k) => Matrix{Float64}(copy(v)) for (k, v) in series)
    for (k, v) in clim
        size(v, 2) == length(years) || throw(ArgumentError("series '$k' has $(size(v,2)) columns, expected $(length(years))"))
    end
    changes === nothing && return clim
    for ch in changes
        _apply_one!(clim, years, ch)
    end
    return clim
end

"""
    apply_scenario(clim::Dict{String,Matrix}, years, scenario) -> Dict

Apply `scenario.changes` (see `apply_changes`), then enforce `ppt >= 0` and `tmax >= tmin`
(tmax is raised to tmin where violated). Other keys (tmean, vpd, ...) pass through unchanged.
"""
function apply_scenario(clim::AbstractDict, years::AbstractVector{<:Integer}, scenario)
    out = apply_changes(clim, years, _get(scenario, :changes, nothing))
    haskey(out, "ppt") && (out["ppt"] .= max.(out["ppt"], 0.0))
    if haskey(out, "tmin") && haskey(out, "tmax")
        out["tmax"] .= max.(out["tmax"], out["tmin"])
    end
    return out
end

"""
    build_climate(obs, obs_years, fc, fc_years, scenario; start_year, end_year) -> (years, Dict)

Convenience: `baseline` for every variable (mode/T/K from `scenario.baseline`, default :forecast, 30, 3)
then `apply_scenario`. `obs`/`fc` are Dict var=>Matrix; `fc`/`fc_years` may be `nothing`.
"""
function build_climate(obs::AbstractDict, obs_years, fc, fc_years, scenario; start_year::Integer, end_year::Integer)
    b = _get(scenario, :baseline, Dict{String,Any}("mode" => "forecast"))
    mode = Symbol(_get(b, :mode, "forecast"))
    T = Int(_get(b, :T, 30)); K = Int(_get(b, :K, 3))
    yrs = collect(start_year:end_year)
    clim = Dict{String,Matrix{Float64}}()
    for (v, M) in obs
        f = fc === nothing ? nothing : get(fc, v, nothing)
        clim[String(v)] = baseline(M, obs_years, f, fc_years, mode; T=T, K=K, start_year=start_year, end_year=end_year)
    end
    return yrs, apply_scenario(clim, yrs, scenario)
end

# ---------------------------------------------------------------------------------------------
# 3. CO2
# ---------------------------------------------------------------------------------------------
# Global-mean atmospheric CO2 (ppm), Meinshausen et al. 2020 (GMD 13, 3571-3605) SSP concentration pathways incl.
# post-2100 extensions, as distributed in RCMIP v5.1.0 "rcmip-concentrations-annual-means-v5-1-0.csv"
# (Nicholls et al. 2020, GMD 13, 3571; model MESSAGE-GLOBIOM ssp245 / REMIND-MAGPIE ssp585, Region=World,
# Atmospheric Concentrations|CO2). Values below are the annual-mean values read from that file for the listed years
# (rounded to 0.1 ppm); intermediate years are linearly interpolated.
const SSP245_CO2 = [(2015, 399.9), (2020, 414.4), (2030, 444.1), (2040, 475.3), (2050, 506.9), (2060, 537.1),
                    (2070, 563.9), (2080, 585.1), (2090, 597.7), (2100, 602.8), (2110, 605.7), (2120, 610.1),
                    (2130, 615.2), (2140, 620.6), (2150, 626.3)]
const SSP585_CO2 = [(2015, 399.9), (2020, 414.9), (2030, 451.5), (2040, 500.0), (2050, 562.8), (2060, 643.0),
                    (2070, 743.7), (2080, 864.4), (2090, 998.3), (2100, 1135.2), (2110, 1269.3), (2120, 1397.6),
                    (2130, 1519.0), (2140, 1632.5), (2150, 1737.3)]

function _interp(tab, y::Real)
    y <= tab[1][1] && return tab[1][2]
    y >= tab[end][1] && return tab[end][2]
    i = findlast(p -> p[1] <= y, tab)
    (x0, v0), (x1, v1) = tab[i], tab[i+1]
    return v0 + (v1 - v0) * (y - x0) / (x1 - x0)
end

"""
    co2_series(years, co2_spec, observed::Dict{Int,Float64}) -> Vector{Float64}

CO2 (ppm) for each year. Years present in `observed` always return the observed value. Beyond the last observed
year `Lo` (max key of `observed`):
* `"constant"`: `observed[Lo]`
* `"linear"`: `observed[Lo] + rate_ppm_per_yr*(y-Lo)` (default rate 2.0)
* `"ssp245"` / `"ssp585"`: piecewise-linear interpolation of the table values (`SSP245_CO2`, `SSP585_CO2`);
  NOT re-anchored to the observed value (a small step of a few ppm at Lo+1 is possible).
Years before the first observed year but absent from `observed` use the first observed value; unobserved years
within the record are linearly interpolated between neighbours.
"""
function co2_series(years::AbstractVector{<:Integer}, spec, observed::AbstractDict{<:Integer,<:Real})
    isempty(observed) && throw(ArgumentError("observed CO2 dictionary is empty"))
    mode = String(_get(spec, :mode, "constant"))
    rate = Float64(_get(spec, :rate_ppm_per_yr, 2.0))
    oy = sort!(collect(Int, keys(observed))); Lo = oy[end]; vLo = Float64(observed[Lo])
    out = Vector{Float64}(undef, length(years))
    for (i, y) in enumerate(years)
        if haskey(observed, y)
            out[i] = Float64(observed[y])
        elseif y < oy[1]
            out[i] = Float64(observed[oy[1]])
        elseif y < Lo
            j = findlast(<(y), oy)
            x0, x1 = oy[j], oy[j+1]
            out[i] = observed[x0] + (observed[x1] - observed[x0]) * (y - x0) / (x1 - x0)
        elseif mode == "constant"
            out[i] = vLo
        elseif mode == "linear"
            out[i] = vLo + rate * (y - Lo)
        elseif mode == "ssp245"
            out[i] = _interp(SSP245_CO2, y)
        elseif mode == "ssp585"
            out[i] = _interp(SSP585_CO2, y)
        else
            throw(ArgumentError("unknown co2 mode '$mode'"))
        end
    end
    return out
end

end # module
