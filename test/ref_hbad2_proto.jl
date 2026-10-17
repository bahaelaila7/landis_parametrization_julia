# HBAD v2 — biomass-free, process-based cohort model (prototype kernel, stand-alone; not yet wired into Pan's plugin interface).
# Cohort state: N (trees/ha, only decreases), D (quadratic mean diameter, cm), H (mean height, m), A (age, yr). No biomass anywhere.
# Annual step per site (sites independent → threaded; SoA, CSR offsets, isbits parameters, no allocation):
#   1. snapshot   SDI_i = N_i (D_i/25.4)^1.605, SDImax_site = SDI-weighted mean of species SDIMAX, RD = ΣSDI / SDImax_site,
#                 overtopping O_i = (Σ_{H_j > H_i} SDI_j + ½ Σ_{j≠i, H_j = H_i} SDI_j + ½ SDI_i) / SDImax_site  (light competition by height)
#   2. growth     free-growth curves are Chapman–Richards in age, X(a) = Xmax (1 − e^{−r a})^c (X = H, D); the increment is taken at the
#                 cohort's CURRENT size on its curve (state form dX/dt = c r X ((Xmax/X)^{1/c} − 1)), so a suppressed cohort keeps the vigour of
#                 its size (release possible). Competition: dD = dD_pot·exp(−gOD·O + gRD·RD), dH = dH_pot·exp(−gOH·O + gRH·RD)
#                 (asymmetric overtopping vs symmetric crowding; the data say crowding shifts growth from diameter to height).
#   3. mortality  hazard h = exp(mbg + mD ln(D/20) + mD2 D/20) + msup·(1 − e^{−gOD·O})·(D/20)^mSD + exp(mA (A/LONGEVITY − 1));  N ← N e^{−h·Δ}
#                 (background, growth-suppression [same overtopping that slows diameter growth], age-related as in Biomass Succession)
#   4. self-thinning  while RD > 1: remove SDI from cohorts in proportion to SDI_i·(1 − v_i), v_i = e^{−gOD·O_i} (suppressed first)
#   5. A ← A + Δ
# Δ = 1 for full years; a fractional last step (Δ < 1) scales every rate, so an interval of 5.4 yr is integrated exactly.
module HBAD2

export SpP, Sites, simulate!, N_SP_PARAMS, SP_PARAM_NAMES, RecP, FreeSites, run_free!, VIG_Z, VIG_W

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
  form::Int        # potential-growth form: 1 = Chapman–Richards state form; 2 = power-exponential r (X/X0)^c e^{−X/Xmax}
  mSD::Float64     # size exponent of the suppression hazard: msup (1 − v) (D/20)^mSD (small suppressed trees die more if < 0)
  σv::Float64      # persistent growth vigour: a sub-cohort at quantile zq grows ×e^{σv·zq}; its suppression hazard uses v·e^{σv·zq} (slow trees die)
end
GlobP(mD2, mA) = GlobP(mD2, mA, 0.0, 0.0, 0.0, 1, 0.0, 0.0)
GlobP(mD2, mA, pH, pD, pM) = GlobP(mD2, mA, pH, pD, pM, 1, 0.0, 0.0)
GlobP(mD2, mA, pH, pD, pM, form) = GlobP(mD2, mA, pH, pD, pM, form, 0.0, 0.0)
GlobP(mD2, mA, pH, pD, pM, form, mSD) = GlobP(mD2, mA, pH, pD, pM, form, mSD, 0.0)
"Gauss–Hermite (3-point) vigour sub-cohorts: quantiles z and population shares."
const VIG_Z = (-sqrt(3.0), 0.0, sqrt(3.0))
const VIG_W = (1 / 6, 2 / 3, 1 / 6)

"SoA store: CSR over sites; initial state (N0,D0,H0,A0) and working state (N,D,H,A); per-site years (whole) + fractional remainder."
struct Sites
  off::Vector{Int}                       # cohorts of site s: off[s]:off[s+1]-1
  sp::Vector{Int32}
  N0::Vector{Float64}; D0::Vector{Float64}; H0::Vector{Float64}; A0::Vector{Float64}
  N::Vector{Float64}; D::Vector{Float64}; H::Vector{Float64}; A::Vector{Float64}
  O::Vector{Float64}; S::Vector{Float64}; V::Vector{Float64}          # scratch: overtopping, SDI, vigour
  nyr::Vector{Int}; frac::Vector{Float64}
  planted::Vector{Bool}                  # per SITE: planted stand (stand origin) → GlobP planted multipliers
  zq::Vector{Float64}                    # per cohort: vigour quantile (0 = mean-field cohort)
end
function Sites(off, sp, N0, D0, H0, A0, nyr, frac, planted=fill(false, length(off) - 1), zq=zeros(length(sp)))
  n = length(sp)
  Sites(off, Int32.(sp), N0, D0, H0, A0, copy(N0), copy(D0), copy(H0), copy(A0), zeros(n), zeros(n), zeros(n), nyr, frac, planted, zq)
end
nsites(S::Sites) = length(S.off) - 1

@inline sdi(N, D) = N * (D / 25.4)^1.605
@inline function dpot(X, Xmax, r, c)        # Chapman–Richards increment at the current size X
  X >= Xmax && return 0.0
  X <= 0 && return 0.0
  c * r * X * ((Xmax / X)^(1 / c) - 1)
end

# power-exponential potential increment (form 2): r (X/X0)^c exp(−X/Xmax); peak at X = c·Xmax; X0 = 20 cm (D) / 15 m (H)
@inline dpow(X, X0, Xmax, r, c) = X <= 0 ? 0.0 : r * exp(c * log(X / X0) - X / Xmax)
@inline potD(G, D, p, f) = G.form == 1 ? dpot(D, p.Dmax, p.rD * f, p.cD) : dpow(D, 20.0, p.Dmax, p.rD * f, p.cD)
@inline potH(G, H, p, f) = G.form == 1 ? dpot(H, p.Hmax, p.rH * f, p.cH) : dpow(H, 15.0, p.Hmax, p.rH * f, p.cH)

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
    vg = exp(G.σv * S.zq[i])                 # persistent vigour of this sub-cohort (1 for a mean-field cohort)
    v = exp(-p.gOD * o); ve = min(v * vg, 1.0); V[i] = ve
    dD = potD(G, D[i], p, fD) * v * vg * exp(p.gRD * RD)
    dH = potH(G, H[i], p, fH) * vg * exp(-p.gOH * o + p.gRH * RD)
    lD = log(D[i] / 20)
    h = fM * (exp(p.mbg + p.mD * lD + G.mD2 * D[i] / 20) + p.msup * (1 - ve) * exp(G.mSD * lD)) + exp(G.mA * (A[i] / p.longevity - 1))
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

# ---------------------------------------------------------------- free runs (Sim B / scenarios): recruitment into the 1-in tally
"Recruitment parameters per species: R [trees/ha/yr] = rscale · exp(ρ0 + ρS ln(1 + BA_sp(D ≥ dseed)) + ρL·SDI/RD_REF); recruits enter at (D0, H0, A0).
The light index is stand SDI / RD_REF (the scale the coefficients were fitted on), NOT the SDImax-relative RD of the growth kernel."
struct RecP
  rho0::Float64; rhoS::Float64; rhoL::Float64; D0::Float64; H0::Float64; A0::Float64
end
const RD_REF = 600.0          # SDI scale of the recruitment light index (tools/hbad2/recruitment_check.jl)
const RSCALE = 185.2          # one FIA microplot sapling in trees/ha (rates were fitted as microplot counts)
const DSEED = 12.7            # seed-source trees: D ≥ 5 in
const RCLASS = 5              # recruits of one species within RCLASS years form one cohort (its N only decreases once the class closes)

"Free-run store: Sites with per-site capacity (off spans the capacity), live counts and the open recruit slot per (species, site)."
struct FreeSites
  S::Sites
  live::Vector{Int}
  open_slot::Matrix{Int}       # nsp × nsites, 0 = none
  open_year::Matrix{Int}
  dropped::Vector{Int}         # recruits not placed for lack of a free slot (diagnostic; should stay 0)
  vigour::Bool                 # recruits enter as 3 vigour sub-cohorts (VIG_Z / VIG_W)
end
FreeSites(S::Sites, live::Vector{Int}, nsp::Int; vigour::Bool=false) = FreeSites(S, live, zeros(Int, nsp, length(live)), zeros(Int, nsp, length(live)), zeros(Int, length(live)), vigour)

@inline function recruit_site!(F::FreeSites, R::AbstractVector{RecP}, P::AbstractVector{SpP}, s::Int, yr::Int, Δ::Float64, nsp::Int)
  S = F.S; lo = S.off[s]; cap = S.off[s+1] - S.off[s]; n = F.live[s]; hi = lo + n - 1
  tot = 0.0
  @inbounds for i in lo:hi; tot += sdi(S.N[i], S.D[i]); end
  RD = tot / RD_REF
  @inbounds for k in 1:nsp
    r = R[k]; ba = 0.0
    for i in lo:hi; (S.sp[i] == k && S.D[i] >= DSEED) && (ba += 7.853981633974483e-5 * S.D[i]^2 * S.N[i]); end
    rate = RSCALE * exp(r.rho0 + r.rhoS * log1p(ba) + r.rhoL * RD) * Δ
    j = F.open_slot[k, s]; nq = F.vigour ? 3 : 1
    if j == 0 || yr - F.open_year[k, s] >= RCLASS
      if n + nq <= cap
        j = lo + n; n += nq; F.open_slot[k, s] = j; F.open_year[k, s] = yr
        for q in 1:nq
          jj = j + q - 1; w = nq == 1 ? 1.0 : VIG_W[q]
          S.sp[jj] = k; S.N[jj] = rate * w; S.D[jj] = r.D0; S.H[jj] = r.H0; S.A[jj] = r.A0; S.N0[jj] = rate * w; S.zq[jj] = nq == 1 ? 0.0 : VIG_Z[q]
        end
      else
        F.dropped[s] += 1; F.open_slot[k, s] = 0
      end
    else
      for q in 1:nq
        jj = j + q - 1; add = rate * (nq == 1 ? 1.0 : VIG_W[q]); n0 = S.N[jj]; m = n0 + add
        S.D[jj] = sqrt((n0 * S.D[jj]^2 + add * r.D0^2) / m); S.H[jj] = (n0 * S.H[jj] + add * r.H0) / m; S.N[jj] = m; S.N0[jj] += add
      end
    end
  end
  F.live[s] = n
  nothing
end

"Free run of every site for `years[s]` years from its current state (whole years), recruitment after each annual step; `obs_at[s]`
lists the run years at which `snap(s, yr)` is called (single-threaded callback after the threaded run would need storage, so the
callback is invoked inside the site loop and must only write site-private data)."
function run_free!(F::FreeSites, P::AbstractVector{SpP}, G::GlobP, R::AbstractVector{RecP}, years::Vector{Int}, snap)
  S = F.S; nsp = length(P)
  Threads.@threads :static for s in 1:length(F.live)
    lo = S.off[s]
    for y in 1:years[s]
      n = F.live[s]
      n > 0 && step_site!(S, P, G, lo, lo + n - 1, 1.0, S.planted[s])
      recruit_site!(F, R, P, s, y, 1.0, nsp)
      snap(s, y)
    end
  end
  F
end
end # module
