# HBAD3P — process-grounded HBAD3 (docs/hbad3_design.md §8). Every per-species empirical equation of HBAD v2/HBAD3 is replaced by a process form whose
# parameters are global (shared by all species, consistent across R4 regions in the data checks §7.5-7.6) or species TRAITS; per species only the growth
# efficiency ε (and the fecundity F, recruitment) is calibrated. Cohort state: N, D (QMD cm), H (m), A (yr), vigour quantile z. No biomass.
#   SDI_i = N_i (D_i/25.4)^1.605; RD = ΣSDI / SDIMAX (SDIMAX constant); overtopping O_i = (Σ SDI of taller + ½ equal + ½ own) / SDIMAX
#   growth   stem-volume balance  dV = ε (D/20)^b 20^1.605 · e^{−a O} · e^{gR RD} · max(0, 1 − (H / 1.1 Hmax)^θ) · W^ω · e^{σv z}      [cm² m / yr]
#            (pipe-model leaf area ∝ D^b; Beer–Lambert light; crowding; hydraulic limitation of tall trees; water supply/demand AET/PET)
#            dD = dV / (2 D H)                                                                                     (V ∝ D² H at constant height)
#   height   trait allometry H*(D) = 1.37 + (Hmax − 1.37)(1 − exp(−s D / (Hmax − 1.37)))^c  (initial slenderness s, shape c);
#            dH = [H*(D + dD) − H*(D)](1 + η(1 − Lh)) + λ (H*(D)(1 + μ(1 − Lh)) − H),  Lh = e^{−kh O}   (shade elongation, relaxation), ≥ 0
#   mortality h = e^{pM}[m0 / LONGEVITY + ms e^{σST (ST − 3)} e^{−β g} (D/20)^ζ] · e^{mW (Wn − W)/0.1} + e^{mA (A/LONGEVITY − 1)}
#            g = e^{−a O} e^{gR RD} W^ω e^{σv z} (realised relative growth: carbon-starvation stress, Kobe/SORTIE; reserves ∝ size: ζ < 0;
#            shade-tolerant species survive low growth: σST < 0); drought ANOMALY of the period vs the plot normal (Wn − W)
#   self-thinning while RD > 1: remove SDI in proportion to SDI_i (1 − min(1, g_i))
module HBAD3P

export SpPP, GlobPP, SitesP, simulate!

struct SpPP
  lnε::Float64       # growth efficiency (calibrated)
  Hmax::Float64      # trait: maximum height [m]
  ST::Float64        # trait: shade-tolerance class 1-5
  longevity::Float64 # trait [yr]
end
struct GlobPP
  b::Float64; a::Float64; gR::Float64; θ::Float64; ω::Float64          # growth
  s::Float64; c::Float64; η::Float64; λ::Float64; μ::Float64; kh::Float64   # height
  m0::Float64; ms::Float64; σST::Float64; β::Float64; ζ::Float64; mW::Float64; mA::Float64   # mortality
  pD::Float64; pH::Float64; pM::Float64; σv::Float64; sdimax::Float64
end
"SoA store (the v2 harness layout) + per-site water index W (period AET/PET) and its plot normal Wn."
struct SitesP
  off::Vector{Int}; sp::Vector{Int32}
  N0::Vector{Float64}; D0::Vector{Float64}; H0::Vector{Float64}; A0::Vector{Float64}
  N::Vector{Float64}; D::Vector{Float64}; H::Vector{Float64}; A::Vector{Float64}
  O::Vector{Float64}; S::Vector{Float64}; V::Vector{Float64}
  nyr::Vector{Int}; frac::Vector{Float64}; planted::Vector{Bool}; zq::Vector{Float64}
  W::Vector{Float64}; Wn::Vector{Float64}
end
function SitesP(off, sp, N0, D0, H0, A0, nyr, frac, planted, zq, W, Wn)
  n = length(sp)
  SitesP(off, Int32.(sp), N0, D0, H0, A0, copy(N0), copy(D0), copy(H0), copy(A0), zeros(n), zeros(n), zeros(n), nyr, frac, planted, zq, W, Wn)
end
nsites(S::SitesP) = length(S.off) - 1
@inline sdi(N, D) = N * (D / 25.4)^1.605
@inline hstar(p::SpPP, G::GlobPP, D) = (h = p.Hmax - 1.37; 1.37 + h * (1 - exp(-G.s * D / h))^G.c)

@inline function step_site!(S::SitesP, P::AbstractVector{SpPP}, G::GlobPP, lo::Int, hi::Int, Δ::Float64, pl::Bool, W::Float64, Wn::Float64)
  N, D, H, A, O, SD, V, sp = S.N, S.D, S.H, S.A, S.O, S.S, S.V, S.sp
  tot = 0.0
  @inbounds for i in lo:hi; SD[i] = sdi(N[i], D[i]); tot += SD[i]; end
  tot <= 0 && return nothing
  RD = tot / G.sdimax
  @inbounds for i in lo:hi
    o = 0.5 * SD[i]; hi_ = H[i]
    for j in lo:hi
      j == i && continue
      H[j] > hi_ && (o += SD[j]); H[j] == hi_ && (o += 0.5 * SD[j])
    end
    O[i] = o / G.sdimax
  end
  fH = pl ? exp(G.pH) : 1.0; fD = pl ? exp(G.pD) : 1.0
  fM = (pl ? exp(G.pM) : 1.0) * exp(G.mW * (Wn - W) / 0.1)
  wω = W^G.ω; cr = exp(G.gR * RD)
  @inbounds for i in lo:hi
    p = P[sp[i]]; Di = D[i]; Hi = H[i]
    g = exp(-G.a * O[i]) * cr * wω * exp(G.σv * S.zq[i]); V[i] = min(g, 1.0)
    hyd = max(0.0, 1 - min(Hi / (1.1 * p.Hmax), 1.0)^G.θ)
    dV = exp(p.lnε) * (Di / 20)^G.b * 122.98 * g * hyd * fD           # 20^1.605 = 122.98
    dD = dV / (2 * Di * max(Hi, 1.37)) * Δ
    Lh = exp(-G.kh * O[i]); hs = hstar(p, G, Di)
    dH = (hstar(p, G, Di + dD) - hs) * (1 + G.η * (1 - Lh)) + G.λ * Δ * (hs * (1 + G.μ * (1 - Lh)) - Hi)
    h = fM * (G.m0 / p.longevity + G.ms * exp(G.σST * (p.ST - 3)) * exp(-G.β * g) * (Di / 20)^G.ζ) + exp(G.mA * (A[i] / p.longevity - 1))
    N[i] *= exp(-h * Δ); D[i] = Di + dD; H[i] = Hi + max(dH * fH, 0.0)
  end
  for _ in 1:6
    tot = 0.0; ws = 0.0
    @inbounds for i in lo:hi; SD[i] = sdi(N[i], D[i]); tot += SD[i]; ws += SD[i] * (1 - V[i]) + 1e-9 * SD[i]; end
    ex = tot - G.sdimax
    ex <= 1e-6 * G.sdimax && break
    @inbounds for i in lo:hi
      r = min(SD[i], ex * (SD[i] * (1 - V[i]) + 1e-9 * SD[i]) / ws); SD[i] > 0 && (N[i] *= 1 - r / SD[i])
    end
  end
  @inbounds for i in lo:hi; A[i] += Δ; end
  nothing
end

function simulate!(S::SitesP, P::AbstractVector{SpPP}, G::GlobPP)
  Threads.@threads :static for s in 1:nsites(S)
    lo = S.off[s]; hi = S.off[s+1] - 1
    @inbounds for i in lo:hi; S.N[i] = S.N0[i]; S.D[i] = S.D0[i]; S.H[i] = S.H0[i]; S.A[i] = S.A0[i]; end
    pl = S.planted[s]; w = S.W[s]; wn = S.Wn[s]
    for _ in 1:S.nyr[s]; step_site!(S, P, G, lo, hi, 1.0, pl, w, wn); end
    S.frac[s] > 0 && step_site!(S, P, G, lo, hi, S.frac[s], pl, w, wn)
  end
  S
end
end # module
