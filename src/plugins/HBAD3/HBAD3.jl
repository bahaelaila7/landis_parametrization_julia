# HBAD3 — process/trait-bound successor of the biomass-free HBAD v2 kernel (prototype, stand-alone; docs/hbad3_design.md §4).
# Cohort state as v2: N (trees/ha, only decreases), D (QMD, cm), H (mean height, m), A (age, yr), vigour quantile z. No biomass.
# Annual step per site (sites independent → threaded; SoA, CSR offsets, isbits parameters, no allocation):
#   1. snapshot   SDI_i = N_i (D_i/25.4)^1.605 (leaf-area proxy), SDImax_site = SDI-weighted species SDIMAX, RD = ΣSDI / SDImax_site,
#                 overtopping O_i as in v2 (SDI of taller cohorts + ½ equal + ½ own, / SDImax_site)
#   2. light      L_i = exp(−k O_i) (Beer–Lambert on the leaf area above); saturating response f_L = L (1 + K) / (L + K), K per species
#                 (from shade tolerance), f_L(1) = 1
#   3. growth     relative growth g_rel = f_L · e^{gR·RD} · W^ω   (W = site water index AET/PET of the period, ω global)
#                 dD = e^{eD} (D/20)^{cD} e^{−D/Dmax} · g_rel · e^{σv z} (· planted e^{pD})
#                 height from diameter through the species H–D allometry H*(D) = 1.37 + a (1 − e^{−b D})^c (fixed, fitted to FIA):
#                 dH = [H*(D + dD) − H*(D)] (1 + η (1 − L)) + λ (H*(D) (1 + μ (1 − L)) − H)   (shade elongation + relaxation), ≥ 0
#   4. mortality  h = e^{pM}[exp(mbg + mD ln(D/20) + mD2 D/20) + ms (1 − min(1, g_rel e^{σv z}))^γ (D/20)^mSD] + exp(mA (A/LONGEVITY − 1))
#                 (background U-shape, growth-stress [the same light / crowding / water limitation that slows growth], age)
#   5. self-thinning while RD > 1: remove SDI in proportion to SDI_i (1 − v_i), v_i = min(1, g_rel e^{σv z})
# Δ < 1 scales every rate (fractional last step of an interval).
module HBAD3

export SpP3, GlobP3, Sites3, simulate!, VIG_Z, VIG_W, SP3_NAMES

const SP3_NAMES = (:eD, :cD, :Dmax, :lnK, :gR, :mbg, :mD, :ms, :sdimax, :hd_a, :hd_b, :hd_c, :longevity)
struct SpP3
  eD::Float64; cD::Float64; Dmax::Float64; lnK::Float64; gR::Float64
  mbg::Float64; mD::Float64; ms::Float64; sdimax::Float64
  hd_a::Float64; hd_b::Float64; hd_c::Float64; longevity::Float64
end
SpP3(v::AbstractVector) = SpP3(ntuple(i -> Float64(v[i]), length(SP3_NAMES))...)
struct GlobP3
  k::Float64       # light extinction per unit overtopping O
  η::Float64       # shade elongation of the allometric height increment
  λ::Float64       # relaxation rate of H toward the (shade-adjusted) allometric height [1/yr]
  μ::Float64       # shaded trees are (1 + μ (1 − L)) × taller at the same D
  γ::Float64       # shape of the growth-stress hazard (1 − g)^γ
  mD2::Float64; mSD::Float64; mA::Float64
  pH::Float64; pD::Float64; pM::Float64
  σv::Float64
  ω::Float64       # water exponent: growth × W^ω
  sform::Int       # growth-stress hazard form: 1 = (1 − g)^γ; 2 = (e^{−γ g} − e^{−γ}) / (1 − e^{−γ}) (Kobe / SORTIE on relative growth: steep as g → 0);
                   # 3 = e^{−γ dD / 0.25} on the ABSOLUTE realised diameter increment dD [cm/yr] (SORTIE's growth-dependent mortality);
                   # 4 = diagnostic decoupling: 1 − min(1, e^{−km O} e^{σv z}) (own shade index, as v2; breaks the shared stress pathway)
  km::Float64      # sform 4 only: shade coefficient of the mortality index
  mW::Float64      # drought mortality: hazard × e^{mW (1 − W)/0.1}
  τg::Float64      # temperature on growth: × e^{τg (T − 17)/5}, T = site annual mean temperature normal [°C]
  τm::Float64      # temperature on mortality: hazard × e^{τm (T − 17)/5}
end
GlobP3(k, η, λ, μ, γ, mD2, mSD, mA, pH, pD, pM, σv, ω) = GlobP3(k, η, λ, μ, γ, mD2, mSD, mA, pH, pD, pM, σv, ω, 1, 0.0, 0.0, 0.0, 0.0)
GlobP3(k, η, λ, μ, γ, mD2, mSD, mA, pH, pD, pM, σv, ω, sform) = GlobP3(k, η, λ, μ, γ, mD2, mSD, mA, pH, pD, pM, σv, ω, sform, 0.0, 0.0, 0.0, 0.0)
GlobP3(k, η, λ, μ, γ, mD2, mSD, mA, pH, pD, pM, σv, ω, sform, km) = GlobP3(k, η, λ, μ, γ, mD2, mSD, mA, pH, pD, pM, σv, ω, sform, km, 0.0, 0.0, 0.0)
@inline function stress(G::GlobP3, ge, dDa, o, vg)
  G.sform == 1 && return (1 - ge)^G.γ
  G.sform == 2 && return (exp(-G.γ * ge) - exp(-G.γ)) / (1 - exp(-G.γ))
  G.sform == 3 && return exp(-G.γ * dDa / 0.25)
  1 - min(1.0, exp(-G.km * o) * vg)
end
const VIG_Z = (-sqrt(3.0), 0.0, sqrt(3.0))
const VIG_W = (1 / 6, 2 / 3, 1 / 6)

"SoA store (the v2 layout) + per-site water index W."
struct Sites3
  off::Vector{Int}; sp::Vector{Int32}
  N0::Vector{Float64}; D0::Vector{Float64}; H0::Vector{Float64}; A0::Vector{Float64}
  N::Vector{Float64}; D::Vector{Float64}; H::Vector{Float64}; A::Vector{Float64}
  O::Vector{Float64}; S::Vector{Float64}; V::Vector{Float64}
  nyr::Vector{Int}; frac::Vector{Float64}
  planted::Vector{Bool}; zq::Vector{Float64}
  W::Vector{Float64}
  T::Vector{Float64}                     # per site: annual mean temperature normal [°C] (17 = neutral)
end
function Sites3(off, sp, N0, D0, H0, A0, nyr, frac, planted, zq, W, T=fill(17.0, length(W)))
  n = length(sp)
  Sites3(off, Int32.(sp), N0, D0, H0, A0, copy(N0), copy(D0), copy(H0), copy(A0), zeros(n), zeros(n), zeros(n), nyr, frac, planted, zq, W, T)
end
nsites(S::Sites3) = length(S.off) - 1

@inline sdi(N, D) = N * (D / 25.4)^1.605
@inline hstar(p::SpP3, D) = 1.37 + p.hd_a * (1 - exp(-p.hd_b * D))^p.hd_c

@inline function step_site!(S::Sites3, P::AbstractVector{SpP3}, G::GlobP3, lo::Int, hi::Int, Δ::Float64, pl::Bool, W::Float64, Tn::Float64=17.0)
  N, D, H, A, O, SD, V, sp = S.N, S.D, S.H, S.A, S.O, S.S, S.V, S.sp
  tot = 0.0; wmax = 0.0
  @inbounds for i in lo:hi
    SD[i] = sdi(N[i], D[i]); tot += SD[i]; wmax += SD[i] * P[sp[i]].sdimax
  end
  tot <= 0 && return nothing
  smax = wmax / tot; RD = tot / smax
  @inbounds for i in lo:hi
    o = 0.5 * SD[i]; hi_ = H[i]
    for j in lo:hi
      j == i && continue
      H[j] > hi_ && (o += SD[j]); H[j] == hi_ && (o += 0.5 * SD[j])
    end
    O[i] = o / smax
  end
  fH = pl ? exp(G.pH) : 1.0; fD = pl ? exp(G.pD) : 1.0; fM = pl ? exp(G.pM) : 1.0
  tz = (Tn - 17) / 5
  fW = W^G.ω * exp(G.τg * tz)
  fM *= exp(G.mW * (1 - W) / 0.1 + G.τm * tz)
  @inbounds for i in lo:hi
    p = P[sp[i]]
    L = exp(-G.k * O[i]); K = exp(p.lnK); fL = L * (1 + K) / (L + K)
    g = fL * exp(p.gR * RD) * fW
    vg = exp(G.σv * S.zq[i]); ge = min(g * vg, 1.0); V[i] = ge
    Di = D[i]; lD = log(Di / 20)
    dDa = exp(p.eD + p.cD * lD - Di / p.Dmax) * g * vg * fD; dD = dDa * Δ
    hs = hstar(p, Di)
    dH = (hstar(p, Di + dD) - hs) * (1 + G.η * (1 - L)) + G.λ * Δ * (hs * (1 + G.μ * (1 - L)) - H[i])
    h = fM * (exp(p.mbg + p.mD * lD + G.mD2 * Di / 20) + p.ms * stress(G, ge, dDa, O[i], vg) * exp(G.mSD * lD)) + exp(G.mA * (A[i] / p.longevity - 1))
    N[i] *= exp(-h * Δ)
    D[i] = Di + dD; H[i] += max(dH * fH, 0.0)
  end
  for _ in 1:6
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
function simulate!(S::Sites3, P::AbstractVector{SpP3}, G::GlobP3)
  Threads.@threads :static for s in 1:nsites(S)
    lo = S.off[s]; hi = S.off[s+1] - 1
    @inbounds for i in lo:hi; S.N[i] = S.N0[i]; S.D[i] = S.D0[i]; S.H[i] = S.H0[i]; S.A[i] = S.A0[i]; end
    pl = S.planted[s]; w = S.W[s]; tn = S.T[s]
    for _ in 1:S.nyr[s]; step_site!(S, P, G, lo, hi, 1.0, pl, w, tn); end
    S.frac[s] > 0 && step_site!(S, P, G, lo, hi, S.frac[s], pl, w, tn)
  end
  S
end
end # module
