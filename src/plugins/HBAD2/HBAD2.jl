# HBAD v2 — biomass-free, process-based cohort model (prototype kernel, stand-alone; not yet wired into Pan's plugin interface).
# Cohort state: N (trees/ha, only decreases), D (quadratic mean diameter, cm), H (mean height, m), A (age, yr). No biomass anywhere.
# Annual step per site (sites independent → threaded; SoA, CSR offsets, isbits parameters, no allocation):
#   1. snapshot   SDI_i = N_i (D_i/25.4)^1.605, SDImax_site = SDI-weighted mean of species SDIMAX, RD = ΣSDI / SDImax_site,
#                 overtopping O_i = (Σ_{H_j > H_i} SDI_j + ½ Σ_{j≠i, H_j = H_i} SDI_j + ½ SDI_i) / SDImax_site  (light competition by height)
#   2. growth     free-growth curves are Chapman–Richards in age, X(a) = Xmax (1 − e^{−r a})^c (X = H, D); the increment is taken at the
#                 cohort's CURRENT size on its curve (state form dX/dt = c r X ((Xmax/X)^{1/c} − 1)), so a suppressed cohort keeps the vigour of
#                 its size (release possible). Competition: dD = dD_pot·exp(−gOD·O + gRD·RD), dH = dH_pot·exp(−gOH·O + gRH·RD)
#                 (asymmetric overtopping vs symmetric crowding; the data say crowding shifts growth from diameter to height).
#   3. mortality  hazard h = exp(mbg + mD ln(D/20) + mD2 D/20) + msup·(1 − e^{−gOD·O}) + exp(mA (A/LONGEVITY − 1));  N ← N e^{−h·Δ}
#                 (background, growth-suppression [same overtopping that slows diameter growth], age-related as in Biomass Succession)
#   4. self-thinning  while RD > 1: remove SDI from cohorts in proportion to SDI_i·(1 − v_i), v_i = e^{−gOD·O_i} (suppressed first)
#   5. A ← A + Δ
# Δ = 1 for full years; a fractional last step (Δ < 1) scales every rate, so an interval of 5.4 yr is integrated exactly.
module HBAD2

export SpP, Sites, simulate!, N_SP_PARAMS, SP_PARAM_NAMES

const SP_PARAM_NAMES = (:Hmax, :rH, :cH, :Dmax, :rD, :cD, :gOD, :gRD, :gOH, :gRH, :mbg, :mD, :msup, :sdimax, :longevity)
const N_SP_PARAMS = length(SP_PARAM_NAMES)
struct SpP
  Hmax::Float64; rH::Float64; cH::Float64; Dmax::Float64; rD::Float64; cD::Float64
  gOD::Float64; gRD::Float64; gOH::Float64; gRH::Float64; mbg::Float64; mD::Float64; msup::Float64; sdimax::Float64; longevity::Float64
end
SpP(v::AbstractVector) = SpP(ntuple(i -> Float64(v[i]), N_SP_PARAMS)...)
struct GlobP
  mD2::Float64     # linear size term of background mortality (with the per-species ln-size slope mD: U-shaped hazard in D)
  mA::Float64      # age-mortality shape (Biomass Succession "d")
  pH::Float64      # planted cohorts: ln multiplier of height-growth rate
  pD::Float64      # planted cohorts: ln multiplier of diameter-growth rate
  pM::Float64      # planted cohorts: ln multiplier of the background + suppression hazard
end
GlobP(mD2, mA) = GlobP(mD2, mA, 0.0, 0.0, 0.0)

"SoA store: CSR over sites; initial state (N0,D0,H0,A0) and working state (N,D,H,A); per-site years (whole) + fractional remainder."
struct Sites
  off::Vector{Int}                       # cohorts of site s: off[s]:off[s+1]-1
  sp::Vector{Int32}
  N0::Vector{Float64}; D0::Vector{Float64}; H0::Vector{Float64}; A0::Vector{Float64}
  N::Vector{Float64}; D::Vector{Float64}; H::Vector{Float64}; A::Vector{Float64}
  O::Vector{Float64}; S::Vector{Float64}; V::Vector{Float64}          # scratch: overtopping, SDI, vigour
  nyr::Vector{Int}; frac::Vector{Float64}
  planted::Vector{Bool}                  # per SITE: planted stand (stand origin) → GlobP planted multipliers
end
function Sites(off, sp, N0, D0, H0, A0, nyr, frac, planted=fill(false, length(off) - 1))
  n = length(sp)
  Sites(off, Int32.(sp), N0, D0, H0, A0, copy(N0), copy(D0), copy(H0), copy(A0), zeros(n), zeros(n), zeros(n), nyr, frac, planted)
end
nsites(S::Sites) = length(S.off) - 1

@inline sdi(N, D) = N * (D / 25.4)^1.605
@inline function dpot(X, Xmax, r, c)        # Chapman–Richards increment at the current size X
  X >= Xmax && return 0.0
  X <= 0 && return 0.0
  c * r * X * ((Xmax / X)^(1 / c) - 1)
end

@inline function step_site!(S::Sites, P::AbstractVector{SpP}, G::GlobP, lo::Int, hi::Int, Δ::Float64, pl::Bool)
  N, D, H, A, O, SD, V, sp = S.N, S.D, S.H, S.A, S.O, S.S, S.V, S.sp
  tot = 0.0; wmax = 0.0
  @inbounds for i in lo:hi
    SD[i] = sdi(N[i], D[i]); tot += SD[i]; wmax += SD[i] * P[sp[i]].sdimax
  end
  tot <= 0 && return nothing
  smax = wmax / tot; RD = tot / smax
  @inbounds for i in lo:hi                   # overtopping by taller cohorts (+ half of equal-height cohorts and of itself)
    o = 0.5 * SD[i]; hi_ = H[i]
    for j in lo:hi
      j == i && continue
      H[j] > hi_ && (o += SD[j]); H[j] == hi_ && (o += 0.5 * SD[j])
    end
    O[i] = o / smax
  end
  fH = pl ? exp(G.pH) : 1.0; fD = pl ? exp(G.pD) : 1.0; fM = pl ? exp(G.pM) : 1.0
  @inbounds for i in lo:hi
    p = P[sp[i]]; o = O[i]
    v = exp(-p.gOD * o); V[i] = v
    dD = dpot(D[i], p.Dmax, p.rD * fD, p.cD) * v * exp(p.gRD * RD)
    dH = dpot(H[i], p.Hmax, p.rH * fH, p.cH) * exp(-p.gOH * o + p.gRH * RD)
    h = fM * (exp(p.mbg + p.mD * log(D[i] / 20) + G.mD2 * D[i] / 20) + p.msup * (1 - v)) + exp(G.mA * (A[i] / p.longevity - 1))
    N[i] *= exp(-h * Δ)
    D[i] += dD * Δ; H[i] += dH * Δ
  end
  for _ in 1:6                               # self-thinning to RD ≤ 1 (composition-dependent SDImax → repeat)
    tot = 0.0; wmax = 0.0; ws = 0.0
    @inbounds for i in lo:hi
      SD[i] = sdi(N[i], D[i]); tot += SD[i]; wmax += SD[i] * P[sp[i]].sdimax; ws += SD[i] * (1 - V[i]) + 1e-9 * SD[i]
    end
    tot <= 0 && break
    smax = wmax / tot; ex = tot - smax
    ex <= 1e-6 * smax && break
    @inbounds for i in lo:hi
      r = min(SD[i], ex * (SD[i] * (1 - V[i]) + 1e-9 * SD[i]) / ws)
      SD[i] > 0 && (N[i] *= 1 - r / SD[i])
    end
  end
  @inbounds for i in lo:hi; A[i] += Δ; end
  nothing
end

"Reset every site to its initial state and integrate nyr full years + the fractional remainder. Threaded over sites, no allocation."
function simulate!(S::Sites, P::AbstractVector{SpP}, G::GlobP)
  Threads.@threads :static for s in 1:nsites(S)
    lo = S.off[s]; hi = S.off[s+1] - 1
    @inbounds for i in lo:hi; S.N[i] = S.N0[i]; S.D[i] = S.D0[i]; S.H[i] = S.H0[i]; S.A[i] = S.A0[i]; end
    pl = S.planted[s]
    for _ in 1:S.nyr[s]; step_site!(S, P, G, lo, hi, 1.0, pl); end
    S.frac[s] > 0 && step_site!(S, P, G, lo, hi, S.frac[s], pl)
  end
  S
end
end # module
