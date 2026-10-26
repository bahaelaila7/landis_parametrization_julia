# HBAD4P — stand-level process cohort model (docs/hbad4p_design.md). Cohort = species key k × FIA 2-inch diameter class j × vigour lane v; state per
# cohort: N [trees / forested ha], B [basal area m² / forested ha], H [mean height m, N-weighted], A [mean age yr]. Site = plot (the stand).
# One annual step of a site (step_site!; Δ = 1 or the final fraction of a year), all rates from the START-of-step state (explicit Euler):
#   Dq = 100 √(4B / (π N)) [cm];  SDI = N (Dq/25.4)^1.605;  RD = ΣSDI / sdimax;  overtopping O = (Σ_{H' > H} SDI' + ½ Σ_{H' = H, other} SDI' + ½ SDI) / sdimax
#   growth of the lane's mean tree (HBAD3P forms): L = f0 + (1 − f0) e^{−a O};  cw = e^{gR RD} W^ω f_T f_CO2 e^{z σv}
#       f_T = exp(−((T − Topt_k) / wT)²) (T = period mean annual temperature; Topt a species trait);  f_CO2 = 1 + βC ln(C / Cref) (βC fixed)
#       hyd = max(0, 1 − min(H / 1.1 Hmax, 1)^θ);  dV = max(ε (Dq/20)^b · 122.98 · L cw hyd fD − ρ Dq² H, 0);  dD = dV Δ / (2 Dq max(H, 1.37))
#   mortality (HBAD3P hazard): gm = e^{−am O} cw;  h = fM [m0/LONGEVITY + ms e^{σST (ST − 3)} e^{−β gm} (Dq/20)^ζ] + e^{mA (A/LONGEVITY − 1)},
#       fM = (planted ? e^{pM} : 1) e^{mW (Wn − W)/0.1};  survival e^{−h Δ} applies to N and B alike (size-neutral within a cohort)
#   height (HBAD3P): dH = [H*(Dq + dD) − H*(Dq)](1 + η(1 − Lh)) + λΔ (H*(Dq)(1 + μ(1 − Lh)) − H),  Lh = e^{−kh O};  H += max(fH dH, 0)
#   upgrowth (two moments): within [L_j, U_j) the trees are spread linearly, f(x) = ½(1 + s x), x = (D − c)/(w/2) ∈ [−1, 1], s from the cohort's mean
#       square diameter (clamped to [−1, 1]); every tree grows by dD; the trees beyond U_j − dD move to class j+1 with their N and their own basal area
#       (exact moments of the linear density); heights get the allometric offset H*(D_group) − H*(D_all) and are rescaled to conserve Σ N·H
#   self-thinning after the step (≤ 6 passes while ΣSDI > sdimax): the excess SDI removed in proportion to SDI (1 − min(1, g)) (+1e−9 SDI), N and B alike
#   lane mixing: a fraction κΔ of each (k, j) is re-split over the lanes with the lane weights;  recruitment into class 1 at D0 = 2.54 cm:
#       rate = RSCALE e^{lnF_k − λ0 e^{λ1 (ST_k − 3)} RD} W^ωR (seed ? 1 : ε0) Δ, seed = a live cohort of k with D ≥ 12.7 cm on the site;  A += Δ.
module HBAD4P

export SpP4, GlobP4, Sites4, simulate!, lanes, NCLS, CLS_LO, CLS_HI

# ---------------------------------------------------------------- classes and lanes
const NCLS = 16
const CLS_LO = [2.54 * (2k - 1) for k in 1:NCLS]                 # [1, 3, 5, …, 31] in → cm
const CLS_HI = [k < NCLS ? 2.54 * (2k + 1) : Inf for k in 1:NCLS]
"Class of a diameter [cm] (0 below 1 in)."
dclass(D) = D < CLS_LO[1] ? 0 : min(NCLS, floor(Int, (D / 2.54 + 1) / 2))
"Lane nodes and weights: :sym = 0.5 / 0.25 / 0.25 at z = 0, ∓√2 (mean 0, variance 1); :fvs = FVS SN tripling 0.60 / 0.25 / 0.15 at −0.142, +1.271, −1.549."
lanes(mode::Symbol=:sym) = mode == :fvs ? ((-0.14228, 1.271, -1.549), (0.60, 0.25, 0.15)) : ((0.0, -sqrt(2.0), sqrt(2.0)), (0.5, 0.25, 0.25))
const NL = 3

struct SpP4
  lnε::Float64       # growth efficiency (calibrated)
  lnF::Float64       # fecundity (calibrated; ingrowth is part of the stand table)
  Hmax::Float64      # trait: maximum height [m]
  ST::Float64        # trait: shade-tolerance class 1-5
  longevity::Float64 # trait [yr]
  Topt::Float64      # trait: the key's mean annual temperature over its plots [°C]
end
struct GlobP4
  b::Float64; a::Float64; gR::Float64; θ::Float64; ω::Float64                # growth
  s::Float64; c::Float64; η::Float64; λ::Float64; μ::Float64; kh::Float64     # height
  m0::Float64; ms::Float64; σST::Float64; β::Float64; ζ::Float64; mW::Float64; mA::Float64   # mortality
  pD::Float64; pH::Float64; pM::Float64; σv::Float64; sdimax::Float64; am::Float64; ρ::Float64; f0::Float64
  wT::Float64        # width of the temperature response [°C] (Inf = none)
  βC::Float64        # CO2 fertilisation (fixed; 0 = none)
  κ::Float64         # lane mixing rate [1/yr]
  λ0::Float64; λ1::Float64; ε0::Float64; ωR::Float64        # recruitment: light window, background seed rain, water
end
const RSCALE = 185.2            # one FIA microplot sapling in trees / ha (scale of e^{lnF})
const DSEED = 12.7; const D_REC = 2.54; const A_REC = 5.0; const CREF = 390.1   # CO2 reference: 2010 annual mean, Mauna Loa [ppm] (landis_region4/data_r4/co2_annual.csv, the series PnET reads)
@inline sdi(N, D) = N * (D / 25.4)^1.605
@inline hstar(p::SpP4, G::GlobP4, D) = (h = p.Hmax - 1.37; 1.37 + h * (1 - exp(-G.s * D / h))^G.c)
@inline qmd(N, B) = N > 0 ? 100 * sqrt(4 * B / (π * N)) : 0.0
@inline ba(N, D) = N * π / 4 * (D / 100)^2

"""SoA store. Per site s the cells off[s]:off[s+1]-1, ordered key-major, then class, then lane (cell of (k, j, v) = base_k + (j − 1)·NL + v); each key of
a site owns classes 1:jmax (jmax = NCLS for every key in this prototype: all keys × classes × lanes are allocated). Initial (…0) and working state."""
struct Sites4
  off::Vector{Int}; nk::Int
  sp::Vector{Int32}; cl::Vector{Int8}; ln::Vector{Int8}
  N0::Vector{Float64}; B0::Vector{Float64}; H0::Vector{Float64}; A0::Vector{Float64}
  N::Vector{Float64}; B::Vector{Float64}; H::Vector{Float64}; A::Vector{Float64}
  nyr::Vector{Int}; frac::Vector{Float64}; planted::Vector{Bool}
  W::Vector{Float64}; Wn::Vector{Float64}; T::Vector{Float64}; C::Vector{Float64}   # per site: water index, its normal, temperature, CO2 (period means)
end
cell(S::Sites4, s, k, j, v) = S.off[s] + ((k - 1) * NCLS + (j - 1)) * NL + (v - 1)
nsites(S::Sites4) = length(S.off) - 1

"""Lookup tables of the fractional powers of one parameter set (built once per simulate!; linear interpolation on a 0.02 cm diameter grid to 250 cm, and
on [0, 1] in steps of 1e-4 for the hydraulic term; beyond the grid the exact function): H*(D) per key, (D/20)^b, (D/20)^ζ, (D/25.4)^1.605, x^θ."""
struct Tabs
  ht::Matrix{Float64}; pb::Vector{Float64}; pz::Vector{Float64}; ps::Vector{Float64}; hy::Vector{Float64}
end
const TD = 0.02; const TN = 12_501; const TH = 1e-4; const THN = 10_001
function Tabs(P::AbstractVector{SpP4}, G::GlobP4)
  D = [(i - 1) * TD for i in 1:TN]
  Tabs([hstar(p, G, d) for d in D, p in P], [(d / 20)^G.b for d in D], [d > 0 ? (d / 20)^G.ζ : Inf for d in D], [(d / 25.4)^1.605 for d in D],
       [((i - 1) * TH)^G.θ for i in 1:THN])
end
@inline function lut(v::AbstractVector, x, step, n, f)
  u = x / step; (u < 0 || u >= length(v) - 1) && return f(x)       # (length: an empty table = the exact function)
  i = floor(Int, u)
  w = u - i; @inbounds (1 - w) * v[i+1] + w * v[i+2]
end
@inline ht(T::Tabs, p::SpP4, G::GlobP4, k::Integer, D) = lut(view(T.ht, :, k), D, TD, TN, d -> hstar(p, G, d))
@inline pw_b(T::Tabs, G::GlobP4, D) = lut(T.pb, D, TD, TN, d -> (d / 20)^G.b)
@inline pw_z(T::Tabs, G::GlobP4, D) = (D < TD ? (D / 20)^G.ζ : lut(T.pz, D, TD, TN, d -> (d / 20)^G.ζ))
@inline sdi_t(T::Tabs, N, D) = N * lut(T.ps, D, TD, TN, d -> (d / 25.4)^1.605)
@inline hyd_t(T::Tabs, G::GlobP4, x) = lut(T.hy, x, TH, THN, y -> y^G.θ)

"Per-thread scratch of one site (sized for the largest site)."
struct Scratch
  sd::Vector{Float64}; O::Vector{Float64}; g::Vector{Float64}; dD::Vector{Float64}; dH::Vector{Float64}; sv::Vector{Float64}; ord::Vector{Int}; act::Vector{Int}
  inN::Vector{Float64}; inB::Vector{Float64}; inH::Vector{Float64}; inA::Vector{Float64}
end
Scratch(n) = Scratch(zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(Int, n), zeros(Int, n), zeros(n), zeros(n), zeros(n), zeros(n))

"Moments of the linear density ½(1 + s x) on [x0, 1]: Mn = ∫ xⁿ ½(1 + s x) dx, n = 0, 1, 2."
@inline function lin_moments(s, x0)
  m(n) = 0.5 * ((1 - x0^(n + 1)) / (n + 1) + s * (1 - x0^(n + 2)) / (n + 2))
  m(0), m(1), m(2)
end
"Slope s of the linear density on class j that reproduces the mean square diameter q2 (clamped to [−1, 1]); E[D²] = c² + c w s / 3 + w² / 12."
@inline function lin_slope(j, q2)
  c = (CLS_LO[j] + CLS_HI[j]) / 2; w = CLS_HI[j] - CLS_LO[j]
  clamp((q2 - c^2 - w^2 / 12) * 3 / (c * w), -1.0, 1.0)
end
"""Split the survivors (N, B) of a cohort of class j < NCLS growing by dD: the trees with D > U − dD move. Returns (Nm, Bm, Dm_q, Ds_q): movers' N and
basal area after growth, the quadratic mean diameters (after growth) of movers and stayers, and B1 = the basal area of all survivors after growth (stayers: B1 − Bm)."""
@inline function upgrowth(j, N, B, dD)
  c = (CLS_LO[j] + CLS_HI[j]) / 2; w = CLS_HI[j] - CLS_LO[j]; hw = w / 2
  q2 = (qmd(N, B))^2; s = lin_slope(j, q2)
  ED = c + hw * s / 3                                              # mean diameter of the linear density
  B1 = ba(N, 1.0) * (q2 + 2 * dD * ED + dD^2)                      # all trees after growth: N π/4 E[(D + dD)²] / 10⁴
  x0 = (CLS_HI[j] - dD - c) / hw
  x0 >= 1 && return (0.0, 0.0, 0.0, qmd(N, B1), B1)
  x0 = max(x0, -1.0)
  M0, M1, M2 = lin_moments(s, x0); a = c + dD
  Nm = N * M0; Em2 = a^2 * M0 + 2 * a * hw * M1 + hw^2 * M2       # Σ over movers of (D + dD)², per tree of the cohort
  Bm = ba(N, 1.0) * Em2
  Bm = min(Bm, B1)
  Ns = N - Nm; Bs = B1 - Bm
  (Nm, Bm, Nm > 0 ? qmd(Nm, Bm) : 0.0, Ns > 0 ? qmd(Ns, Bs) : 0.0, B1)
end

@inline function step_site!(S::Sites4, P::AbstractVector{SpP4}, G::GlobP4, X::Scratch, s::Int, Δ::Float64, lz, lw, T_::Tabs)
  lo = S.off[s]; hi = S.off[s+1] - 1; N, B, H, A = S.N, S.B, S.H, S.A
  na = 0; tot = 0.0
  @inbounds for i in lo:hi
    if N[i] > 1e-9
      na += 1; X.act[na] = i; X.sd[i - lo + 1] = sdi_t(T_, N[i], qmd(N[i], B[i])); tot += X.sd[i - lo + 1]
    end
  end
  RD = tot / G.sdimax
  # overtopping: cumulative SDI of strictly taller cohorts (+ half of equal heights and of the own)
  @inbounds begin
    ordv = view(X.ord, 1:na); for q in 1:na; ordv[q] = X.act[q]; end
    sort!(ordv; by=i -> -H[i])
    q = 1; above = 0.0
    while q <= na
      r = q; grp = 0.0
      while r <= na && H[ordv[r]] == H[ordv[q]]; grp += X.sd[ordv[r] - lo + 1]; r += 1; end
      for t in q:r-1; i = ordv[t]; X.O[i - lo + 1] = (above + 0.5 * grp) / G.sdimax; end   # ½ (own + equal heights)
      above += grp; q = r
    end
  end
  pl = S.planted[s]; W = S.W[s]; Wn = S.Wn[s]; T = S.T[s]
  fH = pl ? exp(G.pH) : 1.0; fD = pl ? exp(G.pD) : 1.0
  fM = (pl ? exp(G.pM) : 1.0) * exp(G.mW * (Wn - W) / 0.1)
  wω = W^G.ω; cr = exp(G.gR * RD); fC = 1 + G.βC * log(S.C[s] / CREF)
  @inbounds for i in lo:hi; li = i - lo + 1; X.g[li] = 1.0; X.inN[li] = 0.0; X.inB[li] = 0.0; X.inH[li] = 0.0; X.inA[li] = 0.0; end   # (cells empty at the start: no stress)
  # rates from the start-of-step state
  @inbounds for q in 1:na
    i = X.act[q]; li = i - lo + 1; k = S.sp[i]; p = P[k]; Dq = qmd(N[i], B[i]); Hi = H[i]; O = X.O[li]
    fT = (isinf(G.wT) || isnan(T)) ? 1.0 : exp(-((T - p.Topt) / G.wT)^2)          # (no temperature record: no response)
    cw = cr * wω * fT * fC * exp(G.σv * lz[S.ln[i]])
    g = (G.f0 + (1 - G.f0) * exp(-G.a * O)) * cw; X.g[li] = min(g, 1.0)
    gm = exp(-G.am * O) * cw
    hyd = max(0.0, 1 - hyd_t(T_, G, min(Hi / (1.1 * p.Hmax), 1.0)))
    dV = max(exp(p.lnε) * pw_b(T_, G, Dq) * 122.98 * g * hyd * fD - G.ρ * Dq^2 * Hi, 0.0)
    X.dD[li] = dV / (2 * Dq * max(Hi, 1.37)) * Δ
    Lh = exp(-G.kh * O); hs = ht(T_, p, G, k, Dq)
    X.dH[li] = max((ht(T_, p, G, k, Dq + X.dD[li]) - hs) * (1 + G.η * (1 - Lh)) + G.λ * Δ * (hs * (1 + G.μ * (1 - Lh)) - Hi), 0.0) * fH
    h = fM * (G.m0 / p.longevity + G.ms * exp(G.σST * (p.ST - 3)) * exp(-G.β * gm) * pw_z(T_, G, Dq)) + exp(G.mA * (A[i] / p.longevity - 1))
    X.sv[li] = exp(-h * Δ)
  end
  # apply: survival, growth, upgrowth (inflows collected first so that a mover is not moved twice in one step)
  @inbounds for q in 1:na
    i = X.act[q]; li = i - lo + 1; k = S.sp[i]; p = P[k]; j = Int(S.cl[i])
    n = N[i] * X.sv[li]; b = B[i] * X.sv[li]; dD = X.dD[li]; Hn = H[i] + X.dH[li]
    if j < NCLS && dD > 0
      Nm, Bm, Dm, Ds, B1 = upgrowth(j, n, b, dD)
      Dall = qmd(n, B1)
      if Nm > 0
        hD = ht(T_, p, G, k, Dall); hm = Hn + ht(T_, p, G, k, Dm) - hD; hsv = Hn + (n - Nm > 0 ? ht(T_, p, G, k, Ds) - hD : 0.0)
        tot_h = Nm * hm + (n - Nm) * hsv; f = tot_h > 0 ? n * Hn / tot_h : 1.0           # conserve Σ N·H
        t = i + NL; lt = t - lo + 1                                                       # the same key and lane, class j + 1
        X.inN[lt] += Nm; X.inB[lt] += Bm; X.inH[lt] += Nm * hm * f; X.inA[lt] += Nm * A[i]
        N[i] = n - Nm; B[i] = B1 - Bm; H[i] = hsv * f
      else
        N[i] = n; B[i] = B1; H[i] = Hn
      end
    else
      N[i] = n; B[i] = j == NCLS ? ba(n, 1.0) * (qmd(n, b)^2 + 2 * dD * qmd(n, b) + dD^2) : b; H[i] = Hn
    end
  end
  @inbounds for i in lo:hi
    li = i - lo + 1; m = X.inN[li]
    m > 0 || continue
    n0 = N[i]; nt = n0 + m
    H[i] = (n0 * H[i] + X.inH[li]) / nt; A[i] = (n0 * A[i] + X.inA[li]) / nt; N[i] = nt; B[i] += X.inB[li]
  end
  # self-thinning
  for _ in 1:6
    tot = 0.0; ws = 0.0
    @inbounds for i in lo:hi
      li = i - lo + 1; X.sd[li] = N[i] > 1e-12 ? sdi_t(T_, N[i], qmd(N[i], B[i])) : 0.0; tot += X.sd[li]; ws += X.sd[li] * (1 - X.g[li]) + 1e-9 * X.sd[li]
    end
    ex = tot - G.sdimax
    ex <= 1e-6 * G.sdimax && break
    @inbounds for i in lo:hi
      li = i - lo + 1; sd_ = X.sd[li]
      sd_ > 0 || continue
      r = min(sd_, ex * (sd_ * (1 - X.g[li]) + 1e-9 * sd_) / ws); f = 1 - r / sd_; N[i] *= f; B[i] *= f
    end
  end
  # lane mixing within (k, j)
  if G.κ > 0
    k_ = min(G.κ * Δ, 1.0)
    @inbounds for i in lo:NL:hi
      tn = N[i] + N[i+1] + N[i+2]
      tn > 0 || continue
      tb = B[i] + B[i+1] + B[i+2]; th = N[i] * H[i] + N[i+1] * H[i+1] + N[i+2] * H[i+2]; ta = N[i] * A[i] + N[i+1] * A[i+1] + N[i+2] * A[i+2]
      for v in 0:2
        t = i + v; nn = (1 - k_) * N[t] + k_ * lw[v+1] * tn
        nn > 0 || continue
        H[t] = ((1 - k_) * N[t] * H[t] + k_ * lw[v+1] * th) / nn; A[t] = ((1 - k_) * N[t] * A[t] + k_ * lw[v+1] * ta) / nn
        B[t] = (1 - k_) * B[t] + k_ * lw[v+1] * tb; N[t] = nn
      end
    end
  end
  # recruitment into class 1 (stand RD of the start of the step)
  @inbounds for k in 1:S.nk
    p = P[k]; seed = false
    for j in 3:NCLS, v in 1:NL; t = cell(S, s, k, j, v); N[t] > 1e-9 && (seed = true; break); end
    rate = RSCALE * exp(p.lnF - G.λ0 * exp(G.λ1 * (p.ST - 3)) * RD) * W^G.ωR * (seed ? 1.0 : G.ε0) * Δ
    rate > 0 || continue
    h0 = ht(T_, p, G, k, D_REC)
    for v in 1:NL
      t = cell(S, s, k, 1, v); add = rate * lw[v]; n0 = N[t]; nt = n0 + add
      H[t] = (n0 * H[t] + add * h0) / nt; A[t] = (n0 * A[t] + add * A_REC) / nt; N[t] = nt; B[t] += ba(add, D_REC)
    end
  end
  @inbounds for i in lo:hi; N[i] > 0 && (A[i] += Δ); end
  nothing
end

"Run every site from its initial state over its interval (nyr whole years + a final fraction). `lanemode` :sym | :fvs."
function simulate!(S::Sites4, P::AbstractVector{SpP4}, G::GlobP4; lanemode::Symbol=:sym, exact::Bool=false)
  lz, lw = lanes(lanemode); T_ = exact ? Tabs(zeros(0, length(P)), zeros(0), zeros(0), zeros(0), zeros(0)) : Tabs(P, G)   # exact: empty tables → every lookup falls back
  nmax = maximum(diff(S.off))
  XS = [Scratch(nmax) for _ in 1:Threads.maxthreadid()]
  Threads.@threads :static for s in 1:nsites(S)
    X = XS[Threads.threadid()]; lo = S.off[s]; hi = S.off[s+1] - 1
    @inbounds for i in lo:hi; S.N[i] = S.N0[i]; S.B[i] = S.B0[i]; S.H[i] = S.H0[i]; S.A[i] = S.A0[i]; end
    for _ in 1:S.nyr[s]; step_site!(S, P, G, X, s, 1.0, lz, lw, T_); end
    S.frac[s] > 0 && step_site!(S, P, G, X, s, S.frac[s], lz, lw, T_)
  end
  S
end
end # module
