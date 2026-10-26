# HBAD4P — stand-level process cohort model (docs/hbad4p_design.md). Cohort (cell) = species key k × FIA 2-inch diameter class j × vigour lane v; state per
# cell: up to NATOM diameter ATOMS (N_m trees / forested ha at diameter D_m: the cell's N = Σ N_m, B = Σ N_m π/4 D_m² / 10⁴), a mean height H (N-weighted)
# and a mean age A. Site = plot (the stand).
# One annual step of a site (step_site!; Δ = 1 or the final fraction of a year), all rates from the START-of-step state (explicit Euler), per CELL:
#   Dq = QMD of the cell;  SDI = N (Dq/25.4)^1.605;  RD = ΣSDI / sdimax
#   overtopping O_i = Σ_j SDI_j r(H_j − H_i) / sdimax, r(Δ) = clamp(½ + Δ / (2 δH), 0, 1) (δH = 1 m: continuous in the heights; δH → 0 gives HBAD3P's
#       rule: taller cohorts count fully, equal heights and the own cohort half)
#   growth of the lane's mean tree (HBAD3P forms): L = f0 + (1 − f0) e^{−a O};  c = e^{gR RD} W^ω f_T e^{z σv};  g = L c (the realised relative growth)
#       f_T = exp(βT (T − Tref_k)): T = period mean annual temperature, Tref_k = the key's mean period temperature over its training intervals (f_T = 1 at
#       the key's typical regional climate; βT global, either sign — the data decide whether warmer sites grow faster or slower)
#       hyd = max(0, 1 − min(H / 1.1 Hmax, 1)^θ);  dV = max(ε (Dq/20)^b · 122.98 · g f_CO2 hyd fD − ρ Dq² H, 0);  dD = dV Δ / (2 Dq max(H, 1.37))
#       f_CO2 = 1 + βC ln(C / Cref) (βC fixed; the FACE constraint is on NPP, so CO2 acts on growth only — not on the stress of mortality or self-thinning)
#   mortality (HBAD3P hazard): gm = e^{−am O} c (the carbon balance of the realised growth, climate included);
#       h = fM [m0/LONGEVITY + ms e^{σST (ST − 3)} e^{−β gm} (Dq/20)^ζ] + e^{mA (A/LONGEVITY − 1)}, fM = (planted ? e^{pM} : 1) e^{mW (Wn − W)/0.1};
#       survival e^{−h Δ} applies to every atom of the cell (size-neutral within a cell)
#   height (HBAD3P): dH = [H*(Dq + dD) − H*(Dq)](1 + η(1 − Lh)) + λΔ (H*(Dq)(1 + μ(1 − Lh)) − H),  Lh = e^{−kh O};  H += max(fH dH, 0)
#   upgrowth (cohorts in bins): every atom of the cell grows by the cell's dD and lands in the class of its new diameter (any number of classes up); the
#       transfer is EXACT for the atoms (no within-class density is assumed: a cell of 1–3 tallied trees does not diffuse). When a cell holds more than
#       NATOM atoms, its two closest atoms merge (N and N·D² conserved: B is conserved). Moving groups get the allometric height offset
#       H*(D_group) − H*(D_cell), rescaled to conserve Σ N·H; the inflow carries its source cell's realised growth (the self-thinning weight).
#   self-thinning after the step (≤ 6 passes while ΣSDI > sdimax): the excess SDI removed in proportion to SDI (1 − min(1, g)) (+1e−9 SDI)
#   lane mixing: a fraction κΔ of each (k, j) is re-split over the lanes with the lane weights
#   recruitment into class 1 at D0 = 2.54 cm, at the end of the year (no growth in it): rate = RSCALE e^{lnF_k − λ0 e^{λ1 (ST_k − 3)} RD} W^ωR
#       (seed ? 1 : ε0) Δ, RD of the start of the step, seed = a key tree ≥ 12.7 cm on the plot at v1, or ≥ SEED_MIN trees/ha of the key ≥ 12.7 cm now;  A += Δ.
module HBAD4P

export SpP4, GlobP4, Sites4, simulate!, lanes, NCLS, CLS_LO, CLS_HI, NATOM

# ---------------------------------------------------------------- classes, lanes, atoms
const NCLS = 16
const CLS_LO = [2.54 * (2k - 1) for k in 1:NCLS]                 # [1, 3, 5, …, 31] in → cm
const CLS_HI = [k < NCLS ? 2.54 * (2k + 1) : Inf for k in 1:NCLS]
const NATOM = 4                                                   # diameter atoms per cell
"Class of a diameter [cm] (0 below 1 in); the 1e-9 absorbs the rounding of exact class bounds (78.74 cm = 31.0 in → class 16)."
dclass(D) = D < CLS_LO[1] - 1e-9 ? 0 : min(NCLS, floor(Int, (D / 2.54 + 1) / 2 + 1e-9))
"""Lane nodes and weights (mean 0, variance 1 of z): :sym = 0.5 / 0.25 / 0.25 at z = 0, ∓√2; :fvs = FVS SN tripling 0.60 / 0.25 / 0.15 at −0.142, +1.271,
−1.549 (sn/dgdriv.f) standardised to variance 1 (raw variance 0.776), so σv means the same in both modes."""
function lanes(mode::Symbol=:sym)
  mode == :sym && return ((0.0, -sqrt(2.0), sqrt(2.0)), (0.5, 0.25, 0.25))
  mode == :fvs || error("lane mode $mode: :sym | :fvs")
  z = (-0.14228, 1.271, -1.549); w = (0.60, 0.25, 0.15); m = sum(z .* w); sd = sqrt(sum(w .* (z .- m) .^ 2))
  (Tuple((z .- m) ./ sd), w)
end
const NL = 3

struct SpP4
  lnε::Float64       # growth efficiency (calibrated)
  lnF::Float64       # fecundity (calibrated; ingrowth is part of the stand table)
  Hmax::Float64      # trait: maximum height [m]
  ST::Float64        # trait: shade-tolerance class 1-5
  longevity::Float64 # trait [yr]
  Tref::Float64      # the key's mean period temperature over its TRAINING intervals [°C] (f_T = 1 there)
end
struct GlobP4
  b::Float64; a::Float64; gR::Float64; θ::Float64; ω::Float64                # growth
  s::Float64; c::Float64; η::Float64; λ::Float64; μ::Float64; kh::Float64     # height
  m0::Float64; ms::Float64; σST::Float64; β::Float64; ζ::Float64; mW::Float64; mA::Float64   # mortality
  pD::Float64; pH::Float64; pM::Float64; σv::Float64; sdimax::Float64; am::Float64; ρ::Float64; f0::Float64
  βT::Float64        # temperature response of growth [1/°C], log scale, either sign (0 = none)
  βC::Float64        # CO2 fertilisation of growth (fixed; 0 = none)
  κ::Float64         # lane mixing rate [1/yr]
  λ0::Float64; λ1::Float64; ε0::Float64; ωR::Float64        # recruitment: light window, background seed rain, water
end
const RSCALE = 185.2            # scale of e^{lnF} [trees / ha / yr]
const DSEED = 12.7; const SEED_MIN = 1.0                          # seed trees: D ≥ 5 in; a CURRENT seed source needs ≥ 1 tree / forested ha
const D_REC = 2.54; const A_REC = 4.5                             # recruits enter at 1 in, aged 4.5 (5 after the half year of their entry year)
const CREF = 390.1              # CO2 reference: 2010 annual mean, Mauna Loa [ppm] (landis_region4/data_r4/co2_annual.csv, the series PnET reads)
const DH_OT = 1.0               # height ramp of the overtopping rule [m]
@inline sdi(N, D) = N * (D / 25.4)^1.605
@inline hstar(p::SpP4, G::GlobP4, D) = (h = p.Hmax - 1.37; 1.37 + h * (1 - exp(-G.s * D / h))^G.c)
@inline qmd(N, B) = N > 0 ? 100 * sqrt(4 * B / (π * N)) : 0.0
@inline ba(N, D) = N * π / 4 * (D / 100)^2
@inline fT_rel(p::SpP4, G::GlobP4, T) = isnan(T) ? 1.0 : exp(G.βT * (T - p.Tref))

"""SoA store. Per site s the cells off[s]:off[s+1]-1, ordered key-major, then class, then lane (cell of (k, j, v) = off[s] + ((k − 1)·NCLS + (j − 1))·NL + v − 1;
every key × class × lane is allocated); atom m of cell i at (i − 1)·NATOM + m (N = 0: an empty slot). Initial (…0) and working state; the cell totals N / B are
kept in sync with the atoms; per site the climate (period means), the v1 seed presence per key, and the lanes."""
struct Sites4
  off::Vector{Int}; nk::Int
  sp::Vector{Int32}; cl::Vector{Int8}; ln::Vector{Int8}
  AN0::Vector{Float64}; AD0::Vector{Float64}; H0::Vector{Float64}; A0::Vector{Float64}
  AN::Vector{Float64}; AD::Vector{Float64}; H::Vector{Float64}; A::Vector{Float64}
  N0::Vector{Float64}; B0::Vector{Float64}; N::Vector{Float64}; B::Vector{Float64}     # cell totals (derived from the atoms)
  nyr::Vector{Int}; frac::Vector{Float64}; planted::Vector{Bool}
  W::Vector{Float64}; Wn::Vector{Float64}; T::Vector{Float64}; C::Vector{Float64}   # water index, its normal, temperature, CO2
  seed0::BitMatrix                                        # nk × nsites: a tree of the key ≥ 12.7 cm on the plot at v1
  lz::NTuple{3,Float64}; lw::NTuple{3,Float64}             # lane nodes and weights the initial state was split with (simulate! uses the same)
end
cell(S::Sites4, s, k, j, v) = S.off[s] + ((k - 1) * NCLS + (j - 1)) * NL + (v - 1)
nsites(S::Sites4) = length(S.off) - 1
"Cell totals (N, B) of atom arrays (an, ad) at cell i."
@inline function totals(an, ad, i)
  n = 0.0; b = 0.0
  @inbounds for m in 1:NATOM; k = (i - 1) * NATOM + m; n += an[k]; b += ba(an[k], ad[k]); end
  n, b
end

# ---------------------------------------------------------------- atom buffers
"""Add an atom (n, d) to a buffer (bn, bd, count c, capacity cap): when full, it merges into its nearest atom in diameter (N, N·D² conserved). Returns the count."""
@inline function buf_add!(bn, bd, base, c, cap, n, d)
  n > 0 || return c
  if c < cap
    bn[base+c+1] = n; bd[base+c+1] = d; return c + 1
  end
  q = 1; best = Inf
  @inbounds for t in 1:c; e = abs(bd[base+t] - d); e < best && (best = e; q = t); end
  @inbounds begin m = bn[base+q] + n; bd[base+q] = sqrt((bn[base+q] * bd[base+q]^2 + n * d^2) / m); bn[base+q] = m; end
  c
end
"Reduce a buffer of c atoms to ≤ NATOM by merging the closest pair in diameter, repeatedly (N, N·D² conserved). Returns the new count."
@inline function compact!(bn, bd, base, c)
  @inbounds while c > NATOM
    p = 1; q = 2; best = Inf
    for u in 1:c-1, t in u+1:c; e = abs(bd[base+u] - bd[base+t]); e < best && (best = e; p = u; q = t); end
    m = bn[base+p] + bn[base+q]; bd[base+p] = sqrt((bn[base+p] * bd[base+p]^2 + bn[base+q] * bd[base+q]^2) / m); bn[base+p] = m
    bn[base+q] = bn[base+c]; bd[base+q] = bd[base+c]; c -= 1
  end
  c
end
"Write a buffer of c (≤ NATOM) atoms into cell i of (an, ad) (unused slots emptied)."
@inline function put_atoms!(an, ad, i, bn, bd, base, c)
  @inbounds for m in 1:NATOM
    k = (i - 1) * NATOM + m
    if m <= c; an[k] = bn[base+m]; ad[k] = bd[base+m]; else; an[k] = 0.0; ad[k] = 0.0; end
  end
end

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

const CAPB = 4 * NATOM          # inflow buffer per cell (own stayers + movers from lower classes + lanes), merged on overflow
"Per-thread scratch of one site (sized for the largest site)."
struct Scratch
  sd::Vector{Float64}; O::Vector{Float64}; g::Vector{Float64}; dD::Vector{Float64}; dH::Vector{Float64}; sv::Vector{Float64}; ord::Vector{Int}; act::Vector{Int}
  inH::Vector{Float64}; inA::Vector{Float64}; inG::Vector{Float64}; ps::Vector{Float64}
  bn::Vector{Float64}; bd::Vector{Float64}; bc::Vector{Int}                    # inflow atoms per cell: CAPB slots, count
  gn::Vector{Float64}; gb::Vector{Float64}; gt::Vector{Int}                    # one cell's moving groups by target class (N, Σ N D², class)
end
Scratch(n) = Scratch(zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(Int, n), zeros(Int, n), zeros(n), zeros(n), zeros(n), zeros(n + 1),
                     zeros(n * CAPB), zeros(n * CAPB), zeros(Int, n), zeros(NCLS), zeros(NCLS), zeros(Int, NCLS))

"Overtopping of the active cells (X.act[1:na]) with the height ramp DH_OT: O[li] for every active cell (insertion-sorted sweep with a prefix sum; no allocation)."
@inline function overtopping!(X::Scratch, H, na::Int, lo::Int, sdimax::Float64)
  ord = X.ord
  @inbounds for q in 1:na                                              # insertion sort by height (n ~ 10²; allocation-free)
    i = X.act[q]; h = H[i]; t = q - 1
    while t >= 1 && H[ord[t]] > h; ord[t+1] = ord[t]; t -= 1; end
    ord[t+1] = i
  end
  ps = X.ps; ps[1] = 0.0
  @inbounds for q in 1:na; ps[q+1] = ps[q] + X.sd[ord[q] - lo + 1]; end
  tot = ps[na+1]; qa = 1; qb = 1
  @inbounds for q in 1:na
    hq = H[ord[q]]
    while qa <= na && H[ord[qa]] < hq - DH_OT; qa += 1; end          # first cell within the ramp below
    while qb <= na && H[ord[qb]] <= hq + DH_OT; qb += 1; end         # first cell fully above
    o = tot - ps[qb]                                                    # strictly above the ramp: full weight
    for t in qa:qb-1
      j = ord[t]; o += X.sd[j - lo + 1] * clamp(0.5 + (H[j] - hq) / (2 * DH_OT), 0.0, 1.0)
    end
    X.O[ord[q] - lo + 1] = o / sdimax
  end
  nothing
end

@inline function step_site!(S::Sites4, P::AbstractVector{SpP4}, G::GlobP4, X::Scratch, s::Int, Δ::Float64, T_::Tabs)
  lo = S.off[s]; hi = S.off[s+1] - 1; N, B, H, A, AN, AD = S.N, S.B, S.H, S.A, S.AN, S.AD; lz, lw = S.lz, S.lw
  na = 0; tot = 0.0
  @inbounds for i in lo:hi
    if N[i] > 1e-9
      na += 1; X.act[na] = i; X.sd[i - lo + 1] = sdi_t(T_, N[i], qmd(N[i], B[i])); tot += X.sd[i - lo + 1]
    end
  end
  RD = tot / G.sdimax
  overtopping!(X, H, na, lo, G.sdimax)
  pl = S.planted[s]; W = S.W[s]; Wn = S.Wn[s]; T = S.T[s]
  fH = pl ? exp(G.pH) : 1.0; fD = pl ? exp(G.pD) : 1.0
  fM = (pl ? exp(G.pM) : 1.0) * exp(G.mW * (Wn - W) / 0.1)
  wω = W^G.ω; cr = exp(G.gR * RD); fC = 1 + G.βC * log(S.C[s] / CREF)
  # inflow buffers: an inactive cell (≤ 1e-9 trees) keeps its atoms (they are added to, not overwritten)
  @inbounds for i in lo:hi
    li = i - lo + 1; X.inH[li] = 0.0; X.inA[li] = 0.0; X.inG[li] = 0.0; X.g[li] = 1.0; c = 0; base = (li - 1) * CAPB
    if N[i] <= 1e-9
      for m in 1:NATOM; k = (i - 1) * NATOM + m; c = buf_add!(X.bn, X.bd, base, c, CAPB, AN[k], AD[k]); end
      X.inH[li] = N[i] * H[i]; X.inA[li] = N[i] * A[i]; X.inG[li] = N[i]
    end
    X.bc[li] = c
  end
  # rates from the start-of-step state
  @inbounds for q in 1:na
    i = X.act[q]; li = i - lo + 1; k = S.sp[i]; p = P[k]; Dq = qmd(N[i], B[i]); Hi = H[i]; O = X.O[li]
    c_ = cr * wω * fT_rel(p, G, T) * exp(G.σv * lz[S.ln[i]])
    g = (G.f0 + (1 - G.f0) * exp(-G.a * O)) * c_; X.g[li] = min(g, 1.0)
    gm = exp(-G.am * O) * c_
    hyd = max(0.0, 1 - hyd_t(T_, G, min(Hi / (1.1 * p.Hmax), 1.0)))
    dV = max(exp(p.lnε) * pw_b(T_, G, Dq) * 122.98 * g * fC * hyd * fD - G.ρ * Dq^2 * Hi, 0.0)
    X.dD[li] = dV / (2 * Dq * max(Hi, 1.37)) * Δ
    Lh = exp(-G.kh * O); hs = ht(T_, p, G, k, Dq)
    X.dH[li] = max((ht(T_, p, G, k, Dq + X.dD[li]) - hs) * (1 + G.η * (1 - Lh)) + G.λ * Δ * (hs * (1 + G.μ * (1 - Lh)) - Hi), 0.0) * fH
    h = fM * (G.m0 / p.longevity + G.ms * exp(G.σST * (p.ST - 3)) * exp(-G.β * gm) * pw_z(T_, G, Dq)) + exp(G.mA * (A[i] / p.longevity - 1))
    X.sv[li] = exp(-h * Δ)
  end
  # apply: survival, growth, upgrowth of the atoms into the inflow buffers (no cell is grown or moved twice in one step)
  @inbounds for q in 1:na
    i = X.act[q]; li = i - lo + 1; k = S.sp[i]; p = P[k]; j = Int(S.cl[i]); sv = X.sv[li]; dD = X.dD[li]; Hn = H[i] + X.dH[li]; Ai = A[i]; gi = X.g[li]
    ng = 0; nt = 0.0; b2 = 0.0                                          # groups of this cell by target class
    for m in 1:NATOM
      kk = (i - 1) * NATOM + m; n = AN[kk] * sv; n > 0 || continue
      d = AD[kk] + dD; t = j == NCLS ? NCLS : max(j, dclass(d))
      g_ = 0; for u in 1:ng; X.gt[u] == t && (g_ = u; break); end
      g_ == 0 && (ng += 1; g_ = ng; X.gt[g_] = t; X.gn[g_] = 0.0; X.gb[g_] = 0.0)
      X.gn[g_] += n; X.gb[g_] += n * d^2; nt += n; b2 += n * d^2
      c = i + (t - j) * NL; lc = c - lo + 1; X.bc[lc] = buf_add!(X.bn, X.bd, (lc - 1) * CAPB, X.bc[lc], CAPB, n, d)
    end
    nt > 0 || continue
    hD = ht(T_, p, G, k, sqrt(b2 / nt)); sh = 0.0
    for u in 1:ng; sh += X.gn[u] * (Hn + ht(T_, p, G, k, sqrt(X.gb[u] / X.gn[u])) - hD); end
    f = sh > 0 ? nt * Hn / sh : 1.0                                     # conserve Σ N·H over the groups
    for u in 1:ng
      c = i + (X.gt[u] - j) * NL; lc = c - lo + 1; n = X.gn[u]
      X.inH[lc] += n * (Hn + ht(T_, p, G, k, sqrt(X.gb[u] / n)) - hD) * f; X.inA[lc] += n * Ai; X.inG[lc] += n * gi
    end
  end
  @inbounds for i in lo:hi
    li = i - lo + 1; c = X.bc[li]; base = (li - 1) * CAPB
    if c == 0
      for m in 1:NATOM; kk = (i - 1) * NATOM + m; AN[kk] = 0.0; AD[kk] = 0.0; end; N[i] = 0.0; B[i] = 0.0; X.g[li] = 1.0
      continue
    end
    c = compact!(X.bn, X.bd, base, c); put_atoms!(AN, AD, i, X.bn, X.bd, base, c)
    n, b = totals(AN, AD, i); N[i] = n; B[i] = b
    H[i] = X.inH[li] / n; A[i] = X.inA[li] / n; X.g[li] = X.inG[li] / n            # the self-thinning weight of the trees now in the cell
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
      r = min(sd_, ex * (sd_ * (1 - X.g[li]) + 1e-9 * sd_) / ws); f = 1 - r / sd_
      for m in 1:NATOM; AN[(i - 1) * NATOM + m] *= f; end; N[i] *= f; B[i] *= f
    end
  end
  # lane mixing within (k, j): each lane keeps (1 − κΔ) of its atoms and receives κΔ · w_v of the pooled atoms of the three lanes
  if G.κ > 0
    k_ = min(G.κ * Δ, 1.0)
    @inbounds for i in lo:NL:hi
      tn = N[i] + N[i+1] + N[i+2]
      tn > 0 || continue
      th = N[i] * H[i] + N[i+1] * H[i+1] + N[i+2] * H[i+2]; ta = N[i] * A[i] + N[i+1] * A[i+1] + N[i+2] * A[i+2]
      li0 = i - lo + 1
      for v in 0:2                                                      # build the three new lane buffers (scratch slots of the three cells)
        base = (li0 + v - 1) * CAPB; c = 0
        for m in 1:NATOM; kk = (i + v - 1) * NATOM + m; c = buf_add!(X.bn, X.bd, base, c, CAPB, (1 - k_) * AN[kk], AD[kk]); end
        for u in 0:2, m in 1:NATOM; kk = (i + u - 1) * NATOM + m; c = buf_add!(X.bn, X.bd, base, c, CAPB, k_ * lw[v+1] * AN[kk], AD[kk]); end
        X.bc[li0+v] = c
      end
      Hs = (H[i], H[i+1], H[i+2]); As = (A[i], A[i+1], A[i+2]); Ns = (N[i], N[i+1], N[i+2])
      for v in 0:2
        t = i + v; base = (li0 + v - 1) * CAPB; c = compact!(X.bn, X.bd, base, X.bc[li0+v]); put_atoms!(AN, AD, t, X.bn, X.bd, base, c)
        n, b = totals(AN, AD, t); N[t] = n; B[t] = b
        n > 0 || continue
        H[t] = ((1 - k_) * Ns[v+1] * Hs[v+1] + k_ * lw[v+1] * th) / n; A[t] = ((1 - k_) * Ns[v+1] * As[v+1] + k_ * lw[v+1] * ta) / n
      end
    end
  end
  # recruitment into class 1 (stand RD of the start of the step)
  @inbounds for k in 1:S.nk
    p = P[k]; seed = S.seed0[k, s]
    if !seed
      ns = 0.0
      for j in 3:NCLS, v in 1:NL; ns += N[cell(S, s, k, j, v)]; end
      seed = ns >= SEED_MIN
    end
    rate = RSCALE * exp(p.lnF - G.λ0 * exp(G.λ1 * (p.ST - 3)) * RD) * W^G.ωR * (seed ? 1.0 : G.ε0) * Δ
    rate > 0 || continue
    h0 = ht(T_, p, G, k, D_REC)
    for v in 1:NL
      t = cell(S, s, k, 1, v); add = rate * lw[v]; n0 = N[t]; nt = n0 + add; base = (t - lo) * CAPB; c = 0
      for m in 1:NATOM; kk = (t - 1) * NATOM + m; c = buf_add!(X.bn, X.bd, base, c, CAPB, AN[kk], AD[kk]); end
      c = buf_add!(X.bn, X.bd, base, c, CAPB, add, D_REC); c = compact!(X.bn, X.bd, base, c); put_atoms!(AN, AD, t, X.bn, X.bd, base, c)
      H[t] = (n0 * H[t] + add * h0) / nt; A[t] = (n0 * A[t] + add * (A_REC - Δ)) / nt; N[t] = nt; B[t] += ba(add, D_REC)   # (A_REC − Δ: aged A_REC after A += Δ)
    end
  end
  @inbounds for i in lo:hi; N[i] > 0 && (A[i] += Δ); end
  nothing
end

"Run every site from its initial state over its interval (nyr whole years + a final fraction). The lanes are the store's (S.lz / S.lw)."
function simulate!(S::Sites4, P::AbstractVector{SpP4}, G::GlobP4; exact::Bool=false)
  T_ = exact ? Tabs(zeros(0, length(P)), zeros(0), zeros(0), zeros(0), zeros(0)) : Tabs(P, G)   # exact: empty tables → every lookup falls back
  nmax = maximum(diff(S.off))
  XS = [Scratch(nmax) for _ in 1:Threads.maxthreadid()]
  Threads.@threads :static for s in 1:nsites(S)
    X = XS[Threads.threadid()]; lo = S.off[s]; hi = S.off[s+1] - 1
    @inbounds for i in lo:hi
      S.N[i] = S.N0[i]; S.B[i] = S.B0[i]; S.H[i] = S.H0[i]; S.A[i] = S.A0[i]
      for m in 1:NATOM; k = (i - 1) * NATOM + m; S.AN[k] = S.AN0[k]; S.AD[k] = S.AD0[k]; end
    end
    for _ in 1:S.nyr[s]; step_site!(S, P, G, X, s, 1.0, T_); end
    S.frac[s] > 0 && step_site!(S, P, G, X, s, S.frac[s], T_)
  end
  S
end
end # module
