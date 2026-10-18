# HBAD2 calibration: structural switches, the parameter vector layout (theta) and its decode. Port of the prototype tools/hbad2/common.jl
# (FREE, LOGP, BOUNDS, FORM-2 bounds, decode, the 7 globals) and tools/hbad2/fit_joint.jl (recruitment coordinates, kappa, freezing).
#
# theta = [ 14 species parameters x nsp keys (FREE order, log-space for LOGP) | 7 globals ] ++ recruitment coordinates
#   globals: mD2 pH pD pM mSD sigma_v ln(sigma_w)       (decode clamps: mD2 in [-3,3], pH pD in [-1,1], pM in [-2,2], mSD in [-4,2], sigma_v -> min(|.|, sigma_v_max)
#                                                          and 0 without vigour, sigma_w = clamp(exp(.), 0.02, 0.6))
#   recruitment: `free`   3 per key (rho0 rhoS rhoL; decode clamps [-12,2] [-1,4] [-10,2]) appended to theta,
#                `fixed`  none (the data fit of the recruitment csv),
#                `kappa`  one coordinate ln(kappa) (clamped to [-3,1]) that shifts rho0 of every key.

"The per-species coordinates of theta in order (prototype `FREE`)."
const SP_FREE = (:Hmax, :rH, :cH, :Dmax, :rD, :cD, :gOD, :gRD, :gOH, :gRH, :mbg, :mD, :msup, :sdimax)
const NF = length(SP_FREE)
const NGLOB = 7
"The global coordinates of theta in order, after the species block."
const GLOBAL_NAMES = (:mD2, :pH, :pD, :pM, :mSD, :sigma_v, :ln_sigma_w)
"Fitted globals a `freeze` entry `G:<name>` may name (the prototype's `GNAMES`; sigma_w is not freezable by name there, it is by index)."
const FREEZE_GLOBALS = ("mD2", "pH", "pD", "pM", "mSD", "sigma_v")

const BOUNDS_FORM1 = Dict{Symbol,Tuple{Float64,Float64}}(
  :Hmax => (8.0, 70.0), :rH => (1e-3, 0.5), :cH => (0.3, 4.0), :Dmax => (15.0, 250.0), :rD => (1e-3, 0.5), :cD => (0.3, 4.0),
  :gOD => (-1.0, 8.0), :gRD => (-3.0, 3.0), :gOH => (-1.0, 8.0), :gRH => (-3.0, 3.0), :mbg => (-9.0, -1.0), :mD => (-4.0, 4.0),
  :msup => (1e-4, 1.0), :sdimax => (250.0, 2000.0))
"Form 2 (power-exponential) overrides: rX = increment at the reference size [cm/yr | m/yr], cX = size exponent, Xmax = decay scale."
const BOUNDS_FORM2 = Dict{Symbol,Tuple{Float64,Float64}}(
  :rD => (0.01, 3.0), :cD => (-1.5, 3.0), :Dmax => (5.0, 300.0), :rH => (0.005, 3.0), :cH => (-1.5, 3.0), :Hmax => (2.0, 200.0))
const LOGP_FORM1 = (:Hmax, :rH, :cH, :Dmax, :rD, :cD, :msup, :sdimax)

"""
    CalibSwitches

The structural switches of a calibration (the prototype's `HBAD2_*` environment variables, here yaml keys of `hbad2_calib`): growth `form` (1 Chapman-Richards, 2 power-exponential),
`size_split` (cm breaks of the initial diameter classes inside a birth class), `site_sub` (subplot sites, densities x4), `vigour` (3 vigour sub-cohorts) with its bound `sigv_max`,
`birth_w` (birth-year class width), `min_sp_trees` (train trees for an own species key), `sizecls` (size-class term in Sim A), the resolved per-parameter box `lo`/`hi` and log flags.
"""
struct CalibSwitches
  form::Int
  size_split::Vector{Float64}
  site_sub::Bool
  site_mult::Float64
  vigour::Bool
  sigv_max::Float64
  birth_w::Int
  min_sp_trees::Int
  sizecls::Bool
  lo::NTuple{NF,Float64}
  hi::NTuple{NF,Float64}
  islog::NTuple{NF,Bool}
end

"""
    CalibSwitches(; form=2, size_split=[5, 12.7, 25, 40], site="subplot", vigour=true, sigv_max=0.7, birth_w=10, min_sp_trees=3000, sizecls=false, bounds=Dict())

`bounds` maps parameter names (strings or symbols of `SP_FREE`) to `(lo, hi)` and overrides the defaults (form-1 box, form-2 overrides for the growth parameters).
In form 2 `cD` and `cH` are linear coordinates (their box is signed), in form 1 they are log coordinates.
"""
function CalibSwitches(; form::Integer=2, size_split::AbstractVector{<:Real}=[5.0, 12.7, 25.0, 40.0], site::AbstractString="subplot", vigour::Bool=true, sigv_max::Real=0.7,
                       birth_w::Integer=10, min_sp_trees::Integer=3000, sizecls::Bool=false, bounds::AbstractDict=Dict{String,Any}())
  form in (1, 2) || error("hbad2_calib: form must be 1 or 2, got $form")
  site in ("plot", "subplot") || error("hbad2_calib: site must be plot or subplot, got $site")
  b = copy(BOUNDS_FORM1)
  form == 2 && merge!(b, BOUNDS_FORM2)
  for (k, v) in bounds
    s = Symbol(k); haskey(b, s) || error("hbad2_calib.bounds: unknown species parameter $k (known: $(join(SP_FREE, ", ")))")
    (v isa AbstractVector && length(v) == 2 && v[1] < v[2]) || error("hbad2_calib.bounds.$k: need [lo, hi] with lo < hi, got $v")
    b[s] = (Float64(v[1]), Float64(v[2]))
  end
  logp = Set(LOGP_FORM1)
  form == 2 && (delete!(logp, :cD); delete!(logp, :cH))
  return CalibSwitches(Int(form), Float64.(collect(size_split)), site == "subplot", site == "subplot" ? 4.0 : 1.0, vigour, Float64(sigv_max), Int(birth_w), Int(min_sp_trees), sizecls,
                       ntuple(i -> b[SP_FREE[i]][1], NF), ntuple(i -> b[SP_FREE[i]][2], NF), ntuple(i -> SP_FREE[i] in logp, NF))
end

"Initial diameter class of a tree of diameter `D` (cm): the number of `size_split` breaks below `D` (prototype `dclass`)."
@inline dclass(sw::CalibSwitches, D) = searchsortedfirst(sw.size_split, D)

"Encode a species parameter value into its theta coordinate (log for log-space parameters)."
@inline enc_param(sw::CalibSwitches, j::Int, x) = sw.islog[j] ? log(x) : x
"Decode theta coordinate `z` of species parameter `j` (clamped to its box)."
@inline dec_param(sw::CalibSwitches, j::Int, z) = clamp(sw.islog[j] ? exp(z) : z, sw.lo[j], sw.hi[j])

"""
    ThetaLayout

Positions in the search vector. `nsp` keys; `ng = nsp*NF + NGLOB` model coordinates; `recr` = `:free` | `:fixed` | `:kappa`; `nrec` recruitment coordinates (3 nsp | 0 | 1);
`frozen` coordinates keep their start value, `free_idx` are searched (`expand` / `restrict` map between the full and the searched vector).
"""
struct ThetaLayout
  keys::Vector{String}
  nsp::Int
  ng::Int
  recr::Symbol
  nrec::Int
  frozen::Vector{Int}
  free_idx::Vector{Int}
end

function ThetaLayout(keys::Vector{String}; recr::Symbol=:fixed, freeze::AbstractVector{<:AbstractString}=String[])
  recr in (:free, :fixed, :kappa) || error("hbad2_calib: recruitment must be free | fixed | kappa, got $recr")
  nsp = length(keys); ng = nsp * NF + NGLOB
  nrec = recr == :free ? 3nsp : recr == :kappa ? 1 : 0
  frozen = Int[]
  for nm in freeze
    nm = String(strip(nm)); isempty(nm) && continue
    if startswith(nm, "G:")
      g = findfirst(==(nm[3:end]), FREEZE_GLOBALS); g === nothing && error("hbad2_calib.freeze: unknown global $(nm) (known: $(join(("G:" * x for x in FREEZE_GLOBALS), ", ")))")
      push!(frozen, nsp * NF + g)
    else
      j = findfirst(==(Symbol(nm)), SP_FREE); j === nothing && error("hbad2_calib.freeze: unknown species parameter $(nm) (known: $(join(SP_FREE, ", ")))")
      append!(frozen, [(s - 1) * NF + j for s in 1:nsp])
    end
  end
  n = ng + nrec
  return ThetaLayout(keys, nsp, ng, recr, nrec, frozen, setdiff(1:n, frozen))
end
Base.length(L::ThetaLayout) = L.ng + L.nrec
"Full vector from the searched coordinates `z` on top of the start vector `θ0`."
function expand(L::ThetaLayout, θ0::AbstractVector, z::AbstractVector)
  θ = Vector{Float64}(θ0); θ[L.free_idx] .= z; return θ
end
restrict(L::ThetaLayout, θ::AbstractVector) = Float64.(θ[L.free_idx])

"""
    decode_theta(θ, L, sw, longevity) -> (P::Vector{SpP}, G::GlobP, sigma_w::Float64)

Species parameters (decoded and clamped; LONGEVITY per key from `longevity`), the global parameter struct (`mA` = 10) and the observation spread `sigma_w`. `θ` may be longer
than the model block (recruitment coordinates follow it).
"""
function decode_theta(θ::AbstractVector, nsp::Int, sw::CalibSwitches, longevity::AbstractVector{<:Real})
  P = Vector{HBAD2Plugin.SpP}(undef, nsp)
  for s in 1:nsp
    P[s] = HBAD2Plugin.SpP(
      dec_param(sw, 1, θ[(s-1)*NF+1]), dec_param(sw, 2, θ[(s-1)*NF+2]), dec_param(sw, 3, θ[(s-1)*NF+3]), dec_param(sw, 4, θ[(s-1)*NF+4]),
      dec_param(sw, 5, θ[(s-1)*NF+5]), dec_param(sw, 6, θ[(s-1)*NF+6]), dec_param(sw, 7, θ[(s-1)*NF+7]), dec_param(sw, 8, θ[(s-1)*NF+8]),
      dec_param(sw, 9, θ[(s-1)*NF+9]), dec_param(sw, 10, θ[(s-1)*NF+10]), dec_param(sw, 11, θ[(s-1)*NF+11]), dec_param(sw, 12, θ[(s-1)*NF+12]),
      dec_param(sw, 13, θ[(s-1)*NF+13]), dec_param(sw, 14, θ[(s-1)*NF+14]), Float64(longevity[s]))
  end
  o = nsp * NF
  G = HBAD2Plugin.GlobP(clamp(θ[o+1], -3, 3), 10.0, clamp(θ[o+2], -1, 1), clamp(θ[o+3], -1, 1), clamp(θ[o+4], -2, 2), sw.form, clamp(θ[o+5], -4, 2),
                        sw.vigour ? min(abs(θ[o+6]), sw.sigv_max) : 0.0)
  return P, G, clamp(exp(θ[o+7]), 0.02, 0.6)
end

"Recruitment parameters (D0 2.54 cm, H0 3 m, A0 5 yr) from the recruitment coordinates `z` of 3 per key (prototype `decR`)."
decode_recruitment(z::AbstractVector, nsp::Int) =
  HBAD2Plugin.RecP[HBAD2Plugin.RecP(clamp(z[3k-2], -12, 2), clamp(z[3k-1], -1, 4), clamp(z[3k], -10, 2), 2.54, 3.0, 5.0) for k in 1:nsp]

"Shift rho0 of every key by `lk` (prototype `kshift`): the recruitment level coordinate."
function kshift(z::AbstractVector, lk::Real, nsp::Int)
  r = Vector{Float64}(z); for k in 1:nsp; r[3k-2] += lk; end; r
end

"""
    recruitment_of(θ, L, recr_data) -> Vector{RecP}

The recruitment parameters of a full search vector: from its own coordinates (`free`), the data fit `recr_data` (3 coordinates per key; `fixed`) or the data fit shifted by ln kappa (`kappa`).
"""
function recruitment_of(θ::AbstractVector, L::ThetaLayout, recr_data::AbstractVector)
  L.recr == :free && return decode_recruitment(view(θ, L.ng+1:length(θ)), L.nsp)
  L.recr == :kappa && return decode_recruitment(kshift(recr_data, clamp(θ[L.ng+1], -3, 1), L.nsp), L.nsp)
  return decode_recruitment(recr_data, L.nsp)
end

"Read a theta vector from a csv (column `theta`); shorter layouts of earlier fits are expanded as in the prototype (`read_theta`: missing mSD / sigma_v / sigma_w globals get 0, 0.3, ln 0.11)."
function read_theta_csv(path::AbstractString, nsp::Int)
  isfile(path) || error("theta file not found: $path")
  θi = Float64.(vec(CSV.read(path, DataFrame).theta))
  length(θi) == nsp * NF + NGLOB && return θi
  length(θi) == nsp * NF + 4 && return vcat(θi, 0.0, 0.3, log(0.11))
  length(θi) == nsp * NF + 5 && return vcat(θi, 0.3, log(0.11))
  length(θi) == nsp * NF + 6 && return vcat(θi, log(0.11))
  error("theta length $(length(θi)) of $path does not fit $nsp keys ($(nsp * NF + NGLOB) expected, shorter prototype layouts are padded)")
end
