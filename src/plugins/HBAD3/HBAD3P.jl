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

export SpPP, GlobPP, SitesP, simulate!, RecGP, FreeSitesP, run_free!

struct SpPP
  lnε::Float64       # growth efficiency (calibrated)
  Hmax::Float64      # trait: maximum height [m]
  ST::Float64        # trait: shade-tolerance class 1-5
  longevity::Float64 # trait [yr]
  lnF::Float64       # fecundity (calibrated on sapling ingrowth; free runs only)
end
SpPP(lnε, Hmax, ST, longevity) = SpPP(lnε, Hmax, ST, longevity, -Inf)
struct GlobPP
  b::Float64; a::Float64; gR::Float64; θ::Float64; ω::Float64          # growth
  s::Float64; c::Float64; η::Float64; λ::Float64; μ::Float64; kh::Float64   # height
  m0::Float64; ms::Float64; σST::Float64; β::Float64; ζ::Float64; mW::Float64; mA::Float64   # mortality
  pD::Float64; pH::Float64; pM::Float64; σv::Float64; sdimax::Float64
  am::Float64      # light coefficient of the SURVIVAL carbon balance (stress uses e^{−am O}; am = a couples it to the growth light response)
  ρ::Float64       # maintenance respiration per unit stem volume: net dV = gross − ρ D² H (size decline of net growth); 0 = none
  f0::Float64      # diffuse-light floor: light = f0 + (1 − f0) e^{−a O} (sunflecks / diffuse light under a closed canopy); 0 = none
  φ::Float64       # crown: leaf area ∝ (CR / 0.4)^φ (pipe model: sapwood at the crown base); 0 = no crown effect
  crhi::Float64; crlo::Float64; kc::Float64       # crown-ratio target CR* = crlo + (crhi − crlo) e^{−kc O} (crown recession under shade)
  τr::Float64; τe::Float64                        # relaxation rates toward CR*: recession (CR > CR*) and re-expansion (CR < CR*) [1/yr]
end
const CR_OFF = (0.0, 0.4, 0.4, 1.0, 0.0, 0.0)    # no crown effect, constant CR
GlobPP(b, a, gR, θ, ω, s, c, η, λ, μ, kh, m0, ms, σST, β, ζ, mW, mA, pD, pH, pM, σv, sdimax) = GlobPP(b, a, gR, θ, ω, s, c, η, λ, μ, kh, m0, ms, σST, β, ζ, mW, mA, pD, pH, pM, σv, sdimax, a, 0.0, 0.0, CR_OFF...)
GlobPP(b, a, gR, θ, ω, s, c, η, λ, μ, kh, m0, ms, σST, β, ζ, mW, mA, pD, pH, pM, σv, sdimax, am) = GlobPP(b, a, gR, θ, ω, s, c, η, λ, μ, kh, m0, ms, σST, β, ζ, mW, mA, pD, pH, pM, σv, sdimax, am, 0.0, 0.0, CR_OFF...)
GlobPP(b, a, gR, θ, ω, s, c, η, λ, μ, kh, m0, ms, σST, β, ζ, mW, mA, pD, pH, pM, σv, sdimax, am, ρ, f0) = GlobPP(b, a, gR, θ, ω, s, c, η, λ, μ, kh, m0, ms, σST, β, ζ, mW, mA, pD, pH, pM, σv, sdimax, am, ρ, f0, CR_OFF...)
"SoA store (the v2 harness layout) + per-site water index W (period AET/PET) and its plot normal Wn."
struct SitesP
  off::Vector{Int}; sp::Vector{Int32}
  N0::Vector{Float64}; D0::Vector{Float64}; H0::Vector{Float64}; A0::Vector{Float64}
  N::Vector{Float64}; D::Vector{Float64}; H::Vector{Float64}; A::Vector{Float64}
  O::Vector{Float64}; S::Vector{Float64}; V::Vector{Float64}
  nyr::Vector{Int}; frac::Vector{Float64}; planted::Vector{Bool}; zq::Vector{Float64}
  W::Vector{Float64}; Wn::Vector{Float64}
  CR0::Vector{Float64}; CR::Vector{Float64}       # per cohort: crown ratio (fraction of height with live crown), initial and working
end
function SitesP(off, sp, N0, D0, H0, A0, nyr, frac, planted, zq, W, Wn, CR0=fill(0.4, length(sp)))
  n = length(sp)
  SitesP(off, Int32.(sp), N0, D0, H0, A0, copy(N0), copy(D0), copy(H0), copy(A0), zeros(n), zeros(n), zeros(n), nyr, frac, planted, zq, W, Wn, CR0, copy(CR0))
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
    fcr = (S.CR[i] / 0.4)^G.φ                                          # crown: leaf area per unit sapwood at breast height
    vgz = exp(G.σv * S.zq[i]); cw = cr * wω * vgz * fcr
    lt = G.f0 + (1 - G.f0) * exp(-G.a * O[i])
    g = lt * cw; V[i] = min(g, 1.0)
    gm = exp(-G.am * O[i]) * cw                                        # survival carbon balance (own light response)
    hyd = max(0.0, 1 - min(Hi / (1.1 * p.Hmax), 1.0)^G.θ)
    dV = max(exp(p.lnε) * (Di / 20)^G.b * 122.98 * g * hyd * fD - G.ρ * Di^2 * Hi, 0.0)   # gross (20^1.605 = 122.98) − maintenance
    dD = dV / (2 * Di * max(Hi, 1.37)) * Δ
    Lh = exp(-G.kh * O[i]); hs = hstar(p, G, Di)
    dH = (hstar(p, G, Di + dD) - hs) * (1 + G.η * (1 - Lh)) + G.λ * Δ * (hs * (1 + G.μ * (1 - Lh)) - Hi)
    h = fM * (G.m0 / p.longevity + G.ms * exp(G.σST * (p.ST - 3)) * exp(-G.β * gm) * (Di / 20)^G.ζ) + exp(G.mA * (A[i] / p.longevity - 1))
    N[i] *= exp(-h * Δ); D[i] = Di + dD; H[i] = Hi + max(dH * fH, 0.0)
    crt = G.crlo + (G.crhi - G.crlo) * exp(-G.kc * O[i]); c_ = S.CR[i]   # crown recession / re-expansion toward the light-dependent target
    S.CR[i] = clamp(c_ + (c_ > crt ? G.τr : G.τe) * (crt - c_) * Δ, 0.05, 0.95)
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
    @inbounds for i in lo:hi; S.N[i] = S.N0[i]; S.D[i] = S.D0[i]; S.H[i] = S.H0[i]; S.A[i] = S.A0[i]; S.CR[i] = S.CR0[i]; end
    pl = S.planted[s]; w = S.W[s]; wn = S.Wn[s]
    for _ in 1:S.nyr[s]; step_site!(S, P, G, lo, hi, 1.0, pl, w, wn); end
    S.frac[s] > 0 && step_site!(S, P, G, lo, hi, S.frac[s], pl, w, wn)
  end
  S
end
# ---------------------------------------------------------------- free runs: recruitment into the 1-in tally (process form, docs §7.6 R3)
"Recruitment globals: light window λ = λ0 e^{λ1 (ST − 3)} on stand RD (= SDI / sdimax); background seed rain ε0 where no conspecific seed tree is present."
struct RecGP
  λ0::Float64; λ1::Float64; ε0::Float64
end
const RSCALE = 185.2           # one FIA microplot sapling in trees/ha
const DSEED = 12.7             # seed trees: D ≥ 5 in
const RCLASS = 5               # recruits of one species within RCLASS years form one cohort
const D_REC = 2.54; const A_REC = 5.0
"Vigour quadrature of recruits (set by the caller to the run's quadrature): nodes and weights."
const VQZ = Ref{Any}((-sqrt(3.0), 0.0, sqrt(3.0))); const VQW = Ref{Any}((1 / 6, 2 / 3, 1 / 6))   # prototype: type-unstable in the recruitment step only
set_quadrature!(z, w) = (VQZ[] = Tuple(Float64.(z)); VQW[] = Tuple(Float64.(w)); nothing)
"Free-run store: SitesP with a per-site capacity, live counts, the open recruit slot per (key, site), and the plot-level seed presence (static, from the start)."
struct FreeSitesP
  S::SitesP; live::Vector{Int}; open_slot::Matrix{Int}; open_year::Matrix{Int}; dropped::Vector{Int}
  seedplot::BitMatrix            # nsp × nsites: a conspecific seed tree anywhere on the PLOT at the start (dispersal between subplots)
end
@inline function recruit_siteP!(F::FreeSitesP, P::AbstractVector{SpPP}, G::GlobPP, Rg::RecGP, s::Int, yr::Int)
  S = F.S; lo = S.off[s]; cap = S.off[s+1] - S.off[s]; n = F.live[s]; hi = lo + n - 1; nsp = length(P)
  tot = 0.0
  @inbounds for i in lo:hi; tot += sdi(S.N[i], S.D[i]); end
  RD = tot / G.sdimax
  @inbounds for k in 1:nsp
    p = P[k]; seed = F.seedplot[k, s]
    for i in lo:hi; (S.sp[i] == k && S.D[i] >= DSEED && S.N[i] > 0) && (seed = true); end
    rate = RSCALE * exp(p.lnF - Rg.λ0 * exp(Rg.λ1 * (p.ST - 3)) * RD) * (seed ? 1.0 : Rg.ε0)
    rate <= 0 && continue
    h0 = hstar(p, G, D_REC)
    j = F.open_slot[k, s]
    if j == 0 || yr - F.open_year[k, s] >= RCLASS
      zq_ = VQZ[]; wq_ = VQW[]; nq = length(zq_)
      if n + nq <= cap
        j = lo + n; n += nq; F.open_slot[k, s] = j; F.open_year[k, s] = yr
        for q in 1:nq
          jj = j + q - 1; w = wq_[q]
          S.sp[jj] = k; S.N[jj] = rate * w; S.N0[jj] = rate * w; S.D[jj] = D_REC; S.H[jj] = h0; S.A[jj] = A_REC; S.zq[jj] = zq_[q]; S.CR[jj] = G.crhi; S.CR0[jj] = G.crhi
        end
      else
        F.dropped[s] += 1; F.open_slot[k, s] = 0
      end
    else
      wq_ = VQW[]
      for q in 1:length(wq_)
        jj = j + q - 1; add = rate * wq_[q]; n0 = S.N[jj]; m = n0 + add
        S.D[jj] = sqrt((n0 * S.D[jj]^2 + add * D_REC^2) / m); S.H[jj] = (n0 * S.H[jj] + add * h0) / m; S.N[jj] = m; S.N0[jj] += add
      end
    end
  end
  F.live[s] = n
  nothing
end
"Free run of every site for years[s] years from its current state: annual step then recruitment; snap(s, yr) after each year (site-private writes only)."
function run_free!(F::FreeSitesP, P::AbstractVector{SpPP}, G::GlobPP, Rg::RecGP, years::Vector{Int}, snap)
  S = F.S
  Threads.@threads :static for s in 1:length(F.live)
    lo = S.off[s]
    for y in 1:years[s]
      n = F.live[s]
      n > 0 && step_site!(S, P, G, lo, lo + n - 1, 1.0, S.planted[s], S.W[s], S.Wn[s])
      recruit_siteP!(F, P, G, Rg, s, y)
      snap(s, y)
    end
  end
  F
end
end # module
