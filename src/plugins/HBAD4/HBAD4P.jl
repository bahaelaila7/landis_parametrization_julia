# HBAD4P — stand-level process cohort model (docs/hbad4p_design.md). Cohort (cell) = species key k × FIA 2-inch diameter class j × vigour lane v; state per
# cell: up to NATOM diameter ATOMS (N_m trees / forested ha at diameter D_m: the cell's N = Σ N_m, B = Σ N_m π/4 D_m² / 10⁴), a mean height H (N-weighted)
# and a mean age A. Site = plot (the stand).
# One annual step of a site (step_site!; Δ = 1 or the final fraction of a year), all rates from the START-of-step state (explicit Euler):
#   per CELL (competition, light, climate, vigour, height):
#   Dq = QMD of the cell;  SDI = N (Dq/25.4)^1.605;  RD = ΣSDI / sdimax
#   overtopping O_i = Σ_j SDI_j r(H_j − H_i) / sdimax, r(Δ) = clamp(½ + Δ / (2 δH), 0, 1) (δH = 1 m: continuous in the heights; δH → 0 gives HBAD3P's
#       rule: taller cohorts count fully, equal heights and the own cohort half)
#   realised relative growth (HBAD3P forms): L = f0 + (1 − f0) e^{−a O};  c = e^{gR RD} W^ω f_T e^{z σv};  g = L c
#       f_T = exp(βT (T − Tref_k)): T = period mean annual temperature, Tref_k = the key's mean period temperature over its training intervals (f_T = 1 at
#       the key's typical regional climate; βT global, either sign — the data decide whether warmer sites grow faster or slower)
#       hyd = max(0, 1 − min(H / 1.1 Hmax, 1)^θ);  f_CO2 = 1 + βC ln(C / Cref) (βC fixed; the FACE constraint is on NPP, so CO2 acts on growth only — not on
#       the stress of mortality or self-thinning);  gm = e^{−am O} c (the carbon balance of the realised growth, climate included)
#   per ATOM m (diameter D_m; audit r3 I4: the size terms of growth and mortality from the atom's own diameter, one table lookup each):
#       dV_m = max(ε (D_m/20)^b · 122.98 · g f_CO2 hyd fD − ρ D_m² H, 0);  dD_m = dV_m Δ / (2 D_m max(H, 1.37))
#       h_m = fM [m0/LONGEVITY + ms e^{σST (ST − 3)} e^{−β gm} (D_m/20)^ζ] + e^{mA (A/LONGEVITY − 1)}, fM = (planted ? e^{pM} : 1) e^{mW (Wn − W)/0.1};
#       survival e^{−h_m Δ}
#   height (HBAD3P, per cell): dH = [H*(Dq') − H*(Dq)](1 + η(1 − Lh)) + λΔ (H*(Dq)(1 + μ(1 − Lh)) − H), Lh = e^{−kh O}, Dq' = the QMD of the grown atoms
#       (start-of-step N);  H += max(fH dH, 0)
#   upgrowth (cohorts in bins): every atom grows by its dD_m and lands in the class of its new diameter (any number of classes up); the transfer is EXACT for
#       the atoms. Merging (N and N·D² conserved: B is conserved): a cell's atoms are sorted by D and the closest ADJACENT pair is merged while the cell holds
#       more than NATOM atoms (in 1-D the closest pair is adjacent); distances are binned at 1e-6 cm and ties go to the first pair in D order (audit r3 I1a: a
#       merge choice decided by rounding made the objective rough); atoms closer than 1e-6 cm (duplicates) always merge. Moving groups get the allometric
#       height offset H*(D_group) − H*(D_cell), rescaled to conserve Σ N·H; the inflow carries its source cell's realised growth (the self-thinning weight).
#   self-thinning after the step (≤ 6 passes while ΣSDI > sdimax): the excess SDI removed in proportion to SDI (1 − min(1, g)) (+1e−9 SDI)
#   lane mixing: a fraction κΔ of each (k, j) is re-split over the lanes with the lane weights: lane v's atoms = its own atoms × (1 − κΔ + κΔ w_v) + the other
#       lanes' atoms × κΔ w_v (audit r3 B1: the own lane's atoms are added ONCE)
#   recruitment into class 1 at D0 = 2.54 cm, at the end of the year (no growth in it): rate = RSCALE e^{lnF_k − λ0 e^{λ1 (ST_k − 3)} RD} W^ωR
#       (seed ? 1 : ε0) Δ, RD of the start of the step, seed = a key tree ≥ 12.7 cm on the plot at v1, or ≥ SEED_MIN trees/ha of the key ≥ 12.7 cm now;  A += Δ.
# Observation space (finish_site!, the likelihood only — the dynamics are unchanged): every atom is "measured" with the FIA DBH error N(0, sdM²) (sdM fixed,
#   tools/hbad4/params4.jl): it contributes to class t with probability Φ((ub_t − D)/sdM) − Φ((lb_t − D)/sdM) (lb of class 1 = −∞: the 1-in tally threshold is
#   the model's entry definition — recruits enter AT 2.54 cm; ub of class 16 = ∞). PN / PB / PH per (site, key, class) = Σ p N, Σ p N π/4 D², Σ p N H_m (H_m = H + H*(D_m) − mean_N H*(D): the atom's allometric height, centred, audit r4 I1 / r5 I-2).
# Storage (audit r3 B1): the initial atoms are stored only for the cells occupied at v1 (CSR); the working atoms of a site live in the per-thread Scratch while
#   the site is simulated (run_site!: load_site!, step_site! per year, finish_site!), and only the occupied cells are visited.
module HBAD4P

export SpP4, GlobP4, Sites4, simulate!, lanes, NCLS, CLS_LO, CLS_HI, NATOM

# ---------------------------------------------------------------- classes, lanes, atoms
const NCLS = 16
const CLS_LO = [2.54 * (2k - 1) for k in 1:NCLS]                 # [1, 3, 5, …, 31] in → cm
const CLS_HI = [k < NCLS ? 2.54 * (2k + 1) : Inf for k in 1:NCLS]
const NATOM = 16                                                  # diameter atoms per cell (audit r3 B1: 4 → 16; 8 left 0.029 nats/site of merge error at κ = 0.1)
const MBIN = 1e6                                                  # merge distances are binned at 1 / MBIN cm (audit r3 I1a)
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
  κ::Float64         # lane mixing rate [1/yr] (0 = persistent lanes)
  λ0::Float64; λ1::Float64; ε0::Float64; ωR::Float64        # recruitment: light window, background seed rain, water
  sdM::Float64       # FIA DBH measurement error sd [cm] of the observation blur (fixed; 0 = none); likelihood only
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

# ---------------------------------------------------------------- standard normal cdf (observation blur)
"Φ(z) to ~1e-16 absolute by the all-positive erf series erf(x) = 2/√π e^{−x²} Σ 2ⁿ x^{2n+1} / (2n+1)!! (no cancellation; used to build the table)."
function Φ_series(z::Float64)
  x = abs(z) / sqrt(2.0); t = x; s = x; n = 0
  while t > 1e-18 * s; n += 1; t *= 2x^2 / (2n + 1); s += t; end
  e = 2 / sqrt(π) * exp(-x^2) * s
  z >= 0 ? 0.5 * (1 + e) : 0.5 * (1 - e)
end
const ZΦ = 8.5; const HΦ = 1 / 1024                               # cubic Hermite table of Φ on [−8.5, 8.5] (error < 1e-14; Φ(−8.5) = 9.5e-18)
const TΦ = [Φ_series(-ZΦ + (i - 1) * HΦ) for i in 1:round(Int, 2ZΦ / HΦ) + 1]
const Tφ = [exp(-0.5 * (-ZΦ + (i - 1) * HΦ)^2) / sqrt(2π) for i in 1:round(Int, 2ZΦ / HΦ) + 1]
"Standard normal cdf (C¹ cubic Hermite interpolation of TΦ with the exact density as slope; 0 / 1 beyond ±8.5)."
@inline function Φ(z::Float64)
  z <= -ZΦ && return 0.0; z >= ZΦ && return 1.0
  u = (z + ZΦ) / HΦ; i = min(floor(Int, u), length(TΦ) - 2); t = u - i
  @inbounds begin f0 = TΦ[i+1]; f1 = TΦ[i+2]; d0 = Tφ[i+1] * HΦ; d1 = Tφ[i+2] * HΦ; end
  t2 = t * t; t3 = t2 * t
  (2t3 - 3t2 + 1) * f0 + (t3 - 2t2 + t) * d0 + (-2t3 + 3t2) * f1 + (t3 - t2) * d1
end

"""SoA store. Per site s the cells off[s]:off[s+1]-1, ordered key-major, then class, then lane (cell of (k, j, v) = off[s] + ((k − 1)·NCLS + (j − 1))·NL + v − 1;
every key × class × lane has its cell totals). Initial atoms only for the cells OCCUPIED at v1: site s's occupied cells are q0[s]:q0[s+1]-1 (ascending cell
index c0[q]); cell c0[q]'s atoms are AN0 / AD0 [a0[q]:a0[q+1]-1] (sorted by D). Working atoms live in the Scratch of the thread that runs the site. Cell totals
N / B (= Σ atoms) and H, A: initial (…0) and working. PN / PB / PH: the observation-space prediction per (site, key, class) [index ((s − 1) nk + k − 1) NCLS + j]
written by finish_site! (atoms blurred by the DBH error) or obs_cells! (cell totals, unblurred: the nulls). Per site the climate (period means), the v1 seed
presence per key, and the lanes."""
struct Sites4
  off::Vector{Int}; nk::Int
  sp::Vector{Int32}; cl::Vector{Int8}; ln::Vector{Int8}
  q0::Vector{Int}; c0::Vector{Int}; a0::Vector{Int}; AN0::Vector{Float64}; AD0::Vector{Float64}
  N0::Vector{Float64}; B0::Vector{Float64}; H0::Vector{Float64}; A0::Vector{Float64}
  N::Vector{Float64}; B::Vector{Float64}; H::Vector{Float64}; A::Vector{Float64}
  PN::Vector{Float64}; PB::Vector{Float64}; PH::Vector{Float64}
  nyr::Vector{Int}; frac::Vector{Float64}; planted::Vector{Bool}
  W::Vector{Float64}; Wn::Vector{Float64}; T::Vector{Float64}; C::Vector{Float64}   # water index, its normal, temperature, CO2
  seed0::BitMatrix                                        # nk × nsites: a tree of the key ≥ 12.7 cm on the plot at v1
  lz::NTuple{3,Float64}; lw::NTuple{3,Float64}             # lane nodes and weights the initial state was split with (simulate! uses the same)
end
cell(S::Sites4, s, k, j, v) = S.off[s] + ((k - 1) * NCLS + (j - 1)) * NL + (v - 1)
nsites(S::Sites4) = length(S.off) - 1
obsix(S::Sites4, s, k, j) = ((s - 1) * S.nk + (k - 1)) * NCLS + j
"""Build a store from the initial atoms: `atoms0` = pairs (global cell index => [(N_m, D_m), …]) (≤ NATOM atoms each, any order of cells); H0 / A0 dense per
cell. N0 / B0 are the atom totals."""
function make_sites4(off, nk, sp, cl, ln, atoms0, H0, A0, nyr, frac, planted, W, Wn, T, C, seed0, lz, lw)
  ns = length(off) - 1; nc = off[end] - 1
  cs = sort([(Int(i), a) for (i, a) in atoms0 if any(x -> x[1] > 0, a)]; by=first)
  q0 = zeros(Int, ns + 1); c0 = Int[]; a0 = [1]; AN0 = Float64[]; AD0 = Float64[]; N0 = zeros(nc); B0 = zeros(nc)
  for (i, a) in cs
    s = searchsortedlast(off, i); q0[s+1] += 1; push!(c0, i)
    length(a) <= NATOM || error("make_sites4: cell $i has $(length(a)) > NATOM atoms")
    for (n, d) in sort(collect(a); by=last, alg=InsertionSort); n > 0 || continue; push!(AN0, n); push!(AD0, d); N0[i] += n; B0[i] += ba(n, d); end   # (sorted by D: the kernel keeps every cell's atoms sorted)
    push!(a0, length(AN0) + 1)
  end
  q0[1] = 1; for s in 1:ns; q0[s+1] += q0[s]; end
  no = ns * nk * NCLS
  Sites4(off, nk, sp, cl, ln, q0, c0, a0, AN0, AD0, N0, B0, H0, A0, copy(N0), copy(B0), copy(H0), copy(A0), zeros(no), zeros(no), zeros(no),
         nyr, frac, planted, W, Wn, T, C, seed0, lz, lw)
end
"Initial atoms [(N_m, D_m), …] of global cell i (empty if the cell is not occupied at v1)."
function atoms0(S::Sites4, i)
  s = searchsortedlast(S.off, i); r = S.q0[s]:S.q0[s+1]-1; q = searchsortedfirst(view(S.c0, r), i)
  (q <= length(r) && S.c0[r[q]] == i) || return Tuple{Float64,Float64}[]
  qq = r[q]; [(S.AN0[a], S.AD0[a]) for a in S.a0[qq]:S.a0[qq+1]-1]
end

# ---------------------------------------------------------------- atom buffers
@inline gapbin(x) = floor(max(x, 0.0) * MBIN)                   # (≥ 0: a merged diameter can sit an ulp outside its inputs)
@inline function mrg!(bn, bd, x, y)                              # merge slot y into slot x (N, N·D² conserved)
  @inbounds begin m = bn[x] + bn[y]; bd[x] = sqrt((bn[x] * bd[x]^2 + bn[y] * bd[y]^2) / m); bn[x] = m; end
end
"""Add an atom (n, d) to a buffer (bn, bd, count c, capacity cap): when full, it merges into its nearest atom in diameter (distance binned at 1e-6 cm, the first
in buffer order on ties; N, N·D² conserved). Returns the count."""
@inline function buf_add!(bn, bd, base, c, cap, n, d)
  n > 0 || return c
  if c < cap
    @inbounds bn[base+c+1] = n; @inbounds bd[base+c+1] = d; return c + 1
  end
  q = 1; best = Inf
  @inbounds for t in 1:c; e = gapbin(abs(bd[base+t] - d)); if e < best; best = e; q = t; e == 0 && break; end; end
  @inbounds begin m = bn[base+q] + n; bd[base+q] = sqrt((bn[base+q] * bd[base+q]^2 + n * d^2) / m); bn[base+q] = m; end
  c
end
"Workspace of compact! (the binned adjacent distances as integers, a linked list over the buffer slots; allocation-free merging)."
struct MergeWS
  g::Vector{Int}; nxt::Vector{Int}; prv::Vector{Int}
end
MergeWS(n) = MergeWS(zeros(Int, n + 1), zeros(Int, n + 1), zeros(Int, n + 1))
@inline gapint(x) = unsafe_trunc(Int, gapbin(x))
"""Sort a buffer of c atoms by D (stable insertion sort), then merge ADJACENT pairs (N, N·D² conserved; the merged diameter lies between its inputs, so the
buffer stays sorted): the pair with the smallest distance binned at 1e-6 cm (the first in D order on ties), while the buffer holds more than natom atoms or that
distance bins to 0 (duplicates). Returns the new count. Implementation: the binned distances g[left slot] over a linked list of the slots (a merged-away slot
gets typemax, no shifting); per merge one vectorised minimum pass, its first position, and the two distances next to the merge recomputed — the result is
the one of rescanning all pairs (a heap was slower at these sizes)."""
function compact!(bn, bd, base, c, natom=NATOM; ws::Union{Nothing,MergeWS}=nothing)
  @inbounds for u in 2:c
    n = bn[base+u]; d = bd[base+u]; t = u - 1
    while t >= 1 && bd[base+t] > d; bn[base+t+1] = bn[base+t]; bd[base+t+1] = bd[base+t]; t -= 1; end
    bn[base+t+1] = n; bd[base+t+1] = d
  end
  c >= 2 || return c
  W = ws === nothing ? MergeWS(c) : ws; g = W.g; nxt = W.nxt; prv = W.prv
  cc = c; M = typemax(Int); L = c - 1
  @inbounds begin
    for u in 1:L; g[u] = gapint(bd[base+u+1] - bd[base+u]); nxt[u] = u + 1; prv[u] = u - 1; end
    nxt[c] = c + 1; prv[c] = c - 1
    while cc >= 2
      best = M
      @simd for u in 1:L; best = min(best, g[u]); end
      (cc > natom || best == 0) || break
      p = 1; while g[p] != best; p += 1; end                       # the first pair (in D order) at the minimum
      w = nxt[p]; mrg!(bn, bd, base + p, base + w)
      nn = nxt[w]; nxt[p] = nn; nn <= c && (prv[nn] = p); g[w] = M; cc -= 1
      g[p] = nn <= c ? gapint(bd[base+nn] - bd[base+p]) : M
      q = prv[p]; q >= 1 && (g[q] = gapint(bd[base+p] - bd[base+q]))
      while L >= 1 && g[L] == M; L -= 1; end
    end
    cc == c && return c
    t = 0; u = 1
    while u <= c; t += 1; bn[base+t] = bn[base+u]; bd[base+t] = bd[base+u]; u = nxt[u]; end
  end
  cc
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
"Empty tables: every lookup falls back to the exact function."
exact_tabs(nk) = Tabs(zeros(0, nk), zeros(0), zeros(0), zeros(0), zeros(0))
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
"""Per-thread scratch of one site (sized for the largest site; indices are LOCAL cells li = i − off[s] + 1): the site's working atoms (an / ad, count ac;
NATOM slots per cell), the occupied cells occ[1:no] (ascending; ac > 0 exactly for these), per-cell rates, per-atom rates (adD, asv), inflow buffers."""
mutable struct Scratch
  an::Vector{Float64}; ad::Vector{Float64}; ac::Vector{Int}; occ::Vector{Int}; no::Int
  nl::Vector{Int}; nn::Int; stamp::Vector{Int}; st::Int                          # cells touched in the current phase (list + stamp)
  sd::Vector{Float64}; O::Vector{Float64}; g::Vector{Float64}; dD::Vector{Float64}; dH::Vector{Float64}; sv::Vector{Float64}; ord::Vector{Int}; act::Vector{Int}
  adD::Vector{Float64}; asv::Vector{Float64}
  inH::Vector{Float64}; inA::Vector{Float64}; inG::Vector{Float64}; ps::Vector{Float64}; ks::Vector{Float64}
  bn::Vector{Float64}; bd::Vector{Float64}; bc::Vector{Int}                    # inflow atoms per cell: CAPB slots, count
  gn::Vector{Float64}; gb::Vector{Float64}; gt::Vector{Int}                    # one cell's moving groups by target class (N, Σ N D², class)
  ws::MergeWS                                                                   # compact! workspace
end
Scratch(n) = Scratch(zeros(n * NATOM), zeros(n * NATOM), zeros(Int, n), zeros(Int, n), 0, zeros(Int, n), 0, zeros(Int, n), 0,
                     zeros(n), zeros(n), ones(n), zeros(n), zeros(n), zeros(n), zeros(Int, n), zeros(Int, n), zeros(n * NATOM), zeros(n * NATOM),
                     zeros(n), zeros(n), zeros(n), zeros(n + 1), zeros(n), zeros(n * CAPB), zeros(n * CAPB), zeros(Int, n), zeros(NCLS), zeros(NCLS), zeros(Int, NCLS), MergeWS(CAPB))
"Working atoms [(N_m, D_m), …] of global cell i of site s (the site loaded in X)."
cell_atoms(X::Scratch, S::Sites4, s, i) = (li = i - S.off[s] + 1; [(X.an[(li-1)*NATOM+m], X.ad[(li-1)*NATOM+m]) for m in 1:X.ac[li]])
@inline function touch!(X::Scratch, li)                          # first touch of a cell in this phase: empty inflow buffer and accumulators
  @inbounds if X.stamp[li] != X.st
    X.stamp[li] = X.st; X.bc[li] = 0; X.inH[li] = 0.0; X.inA[li] = 0.0; X.inG[li] = 0.0; X.nn += 1; X.nl[X.nn] = li
  end
  nothing
end
@inline function sortints!(v, n)                                  # insertion sort of v[1:n] (nearly sorted lists)
  @inbounds for u in 2:n; x = v[u]; t = u - 1; while t >= 1 && v[t] > x; v[t+1] = v[t]; t -= 1; end; v[t+1] = x; end
end

"Load site s's initial state: cell totals into S (working), the occupied cells' atoms into X."
function load_site!(S::Sites4, X::Scratch, s::Int)
  lo = S.off[s]; hi = S.off[s+1] - 1
  @inbounds for i in lo:hi; S.N[i] = S.N0[i]; S.B[i] = S.B0[i]; S.H[i] = S.H0[i]; S.A[i] = S.A0[i]; end
  @inbounds for q in 1:X.no; X.ac[X.occ[q]] = 0; end
  X.no = 0
  @inbounds for q in S.q0[s]:S.q0[s+1]-1
    li = S.c0[q] - lo + 1; c = 0
    for a in S.a0[q]:S.a0[q+1]-1; c += 1; X.an[(li-1)*NATOM+c] = S.AN0[a]; X.ad[(li-1)*NATOM+c] = S.AD0[a]; end
    X.ac[li] = c; X.no += 1; X.occ[X.no] = li
  end
  nothing
end

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
"Competition of site s's start-of-step state (X loaded): the active cells X.act[1:na] (N > 1e-9), their SDI X.sd and overtopping X.O; returns (na, RD)."
@inline function competition!(S::Sites4, G::GlobP4, X::Scratch, s::Int, T_::Tabs)
  lo = S.off[s]; N = S.N; B = S.B; na = 0; tot = 0.0
  @inbounds for q in 1:X.no
    li = X.occ[q]; i = li + lo - 1
    if N[i] > 1e-9
      na += 1; X.act[na] = i; X.sd[li] = sdi_t(T_, N[i], qmd(N[i], B[i])); tot += X.sd[li]
    end
  end
  overtopping!(X, S.H, na, lo, G.sdimax)
  na, tot / G.sdimax
end

@inline function step_site!(S::Sites4, P::AbstractVector{SpP4}, G::GlobP4, X::Scratch, s::Int, Δ::Float64, T_::Tabs)
  lo = S.off[s]; N, B, H, A = S.N, S.B, S.H, S.A; lz, lw = S.lz, S.lw; an, ad, ac = X.an, X.ad, X.ac
  na, RD = competition!(S, G, X, s, T_)
  pl = S.planted[s]; W = S.W[s]; Wn = S.Wn[s]; T = S.T[s]
  fH = pl ? exp(G.pH) : 1.0; fD = pl ? exp(G.pD) : 1.0
  fM = (pl ? exp(G.pM) : 1.0) * exp(G.mW * (Wn - W) / 0.1)
  wω = W^G.ω; cr = exp(G.gR * RD); fC = 1 + G.βC * log(S.C[s] / CREF)
  # inflow buffers: an occupied but inactive cell (≤ 1e-9 trees) keeps its atoms (they are added to, not overwritten)
  X.st += 1; X.nn = 0
  @inbounds for q in 1:X.no
    li = X.occ[q]; i = li + lo - 1
    N[i] <= 1e-9 || continue
    touch!(X, li); c = 0; base = (li - 1) * CAPB
    for m in 1:ac[li]; k = (li - 1) * NATOM + m; c = buf_add!(X.bn, X.bd, base, c, CAPB, an[k], ad[k]); end
    X.bc[li] = c; X.inH[li] = N[i] * H[i]; X.inA[li] = N[i] * A[i]; X.inG[li] = N[i]
  end
  # rates from the start-of-step state: per cell (light, climate, vigour, height) and per atom (growth, survival)
  @inbounds for q in 1:na
    i = X.act[q]; li = i - lo + 1; k = S.sp[i]; p = P[k]; Hi = H[i]; O = X.O[li]
    c_ = cr * wω * fT_rel(p, G, T) * exp(G.σv * lz[S.ln[i]])
    g = (G.f0 + (1 - G.f0) * exp(-G.a * O)) * c_; X.g[li] = min(g, 1.0)
    gm = exp(-G.am * O) * c_
    hyd = max(0.0, 1 - hyd_t(T_, G, min(Hi / (1.1 * p.Hmax), 1.0)))
    gr = exp(p.lnε) * 122.98 * g * fC * hyd * fD; hm = 2 * max(Hi, 1.37)
    h0 = fM * G.m0 / p.longevity + exp(G.mA * (A[i] / p.longevity - 1)); hs_ = fM * G.ms * exp(G.σST * (p.ST - 3)) * exp(-G.β * gm)
    n_ = 0.0; d2 = 0.0; g2 = 0.0; sv_ = 0.0
    for m in 1:ac[li]
      kk = (li - 1) * NATOM + m; d = ad[kk]; n = an[kk]
      dd = max(gr * pw_b(T_, G, d) - G.ρ * d^2 * Hi, 0.0) / (d * hm) * Δ
      sv = exp(-(h0 + hs_ * pw_z(T_, G, d)) * Δ)
      X.adD[kk] = dd; X.asv[kk] = sv; n_ += n; d2 += n * d^2; g2 += n * dd * (2d + dd); sv_ += n * sv
    end
    Dq = sqrt(d2 / n_); Dq2 = sqrt((d2 + g2) / n_); X.dD[li] = g2 / n_ / (Dq2 + Dq); X.sv[li] = sv_ / n_     # (dD: Dq' − Dq without cancellation)
    Lh = exp(-G.kh * O); hs = ht(T_, p, G, k, Dq)
    X.dH[li] = max((ht(T_, p, G, k, Dq2) - hs) * (1 + G.η * (1 - Lh)) + G.λ * Δ * (hs * (1 + G.μ * (1 - Lh)) - Hi), 0.0) * fH
  end
  # apply: survival, growth, upgrowth of the atoms into the inflow buffers (no cell is grown or moved twice in one step)
  @inbounds for q in 1:na
    i = X.act[q]; li = i - lo + 1; k = S.sp[i]; p = P[k]; j = Int(S.cl[i]); Hn = H[i] + X.dH[li]; Ai = A[i]; gi = X.g[li]
    ng = 0; nt = 0.0; b2 = 0.0                                          # groups of this cell by target class
    for m in 1:ac[li]
      kk = (li - 1) * NATOM + m; n = an[kk] * X.asv[kk]; n > 0 || continue
      d = ad[kk] + X.adD[kk]; t = j == NCLS ? NCLS : max(j, dclass(d))
      g_ = 0; for u in 1:ng; X.gt[u] == t && (g_ = u; break); end
      g_ == 0 && (ng += 1; g_ = ng; X.gt[g_] = t; X.gn[g_] = 0.0; X.gb[g_] = 0.0)
      X.gn[g_] += n; X.gb[g_] += n * d^2; nt += n; b2 += n * d^2
      lc = li + (t - j) * NL; touch!(X, lc); X.bc[lc] = buf_add!(X.bn, X.bd, (lc - 1) * CAPB, X.bc[lc], CAPB, n, d)
    end
    nt > 0 || continue
    hD = ht(T_, p, G, k, sqrt(b2 / nt)); sh = 0.0
    for u in 1:ng; sh += X.gn[u] * (Hn + ht(T_, p, G, k, sqrt(X.gb[u] / X.gn[u])) - hD); end
    f = sh > 0 ? nt * Hn / sh : 1.0                                     # conserve Σ N·H over the groups
    for u in 1:ng
      lc = li + (X.gt[u] - j) * NL; n = X.gn[u]
      X.inH[lc] += n * (Hn + ht(T_, p, G, k, sqrt(X.gb[u] / n)) - hD) * f; X.inA[lc] += n * Ai; X.inG[lc] += n * gi
    end
  end
  # the touched cells take their inflow; occupied cells that received nothing are emptied; the new occupied list (ascending)
  @inbounds for q in 1:X.no
    li = X.occ[q]
    if X.stamp[li] != X.st
      i = li + lo - 1; ac[li] = 0; N[i] = 0.0; B[i] = 0.0; X.g[li] = 1.0
    end
  end
  sortints!(X.nl, X.nn); no = 0
  @inbounds for q in 1:X.nn
    li = X.nl[q]; i = li + lo - 1; c = X.bc[li]; base = (li - 1) * CAPB
    if c == 0
      ac[li] = 0; N[i] = 0.0; B[i] = 0.0; X.g[li] = 1.0
      continue
    end
    c = compact!(X.bn, X.bd, base, c; ws=X.ws); n = 0.0; b = 0.0
    for m in 1:c; kk = (li - 1) * NATOM + m; an[kk] = X.bn[base+m]; ad[kk] = X.bd[base+m]; n += an[kk]; b += ba(an[kk], ad[kk]); end
    ac[li] = c; N[i] = n; B[i] = b; no += 1; X.occ[no] = li
    H[i] = X.inH[li] / n; A[i] = X.inA[li] / n; X.g[li] = X.inG[li] / n            # the self-thinning weight of the trees now in the cell
  end
  X.no = no
  # self-thinning
  for _ in 1:6
    tot = 0.0; ws = 0.0
    @inbounds for q in 1:X.no
      li = X.occ[q]; i = li + lo - 1; X.sd[li] = N[i] > 1e-12 ? sdi_t(T_, N[i], qmd(N[i], B[i])) : 0.0; tot += X.sd[li]; ws += X.sd[li] * (1 - X.g[li]) + 1e-9 * X.sd[li]
    end
    ex = tot - G.sdimax
    ex <= 1e-6 * G.sdimax && break
    @inbounds for q in 1:X.no
      li = X.occ[q]; i = li + lo - 1; sd_ = X.sd[li]
      sd_ > 0 || continue
      r = min(sd_, ex * (sd_ * (1 - X.g[li]) + 1e-9 * sd_) / ws); f = 1 - r / sd_
      for m in 1:ac[li]; an[(li - 1) * NATOM + m] *= f; end; N[i] *= f; B[i] *= f
    end
  end
  # lane mixing within (k, j): lane v's atoms = its own × (1 − κΔ + κΔ w_v) + the other lanes' × κΔ w_v (each (k, j) once; the occupied list is rebuilt)
  if G.κ > 0
    k_ = min(G.κ * Δ, 1.0); X.st += 1; no = 0; nold = X.no
    @inbounds for q in 1:nold
      l0 = X.occ[q] - (Int(S.ln[X.occ[q] + lo - 1]) - 1)                # the lane-1 cell of this (k, j)
      X.stamp[l0] == X.st && continue
      X.stamp[l0] = X.st; i = l0 + lo - 1
      tn = N[i] + N[i+1] + N[i+2]
      if tn > 0
        th = N[i] * H[i] + N[i+1] * H[i+1] + N[i+2] * H[i+2]; ta = N[i] * A[i] + N[i+1] * A[i+1] + N[i+2] * A[i+2]
        for v in 0:2                                                    # build the three new lane buffers (inflow slots of the three cells): a 3-way merge of
          base = (l0 + v - 1) * CAPB; c = 0                             # the lanes' sorted atoms (ties: the lower lane first = a stable sort of lane 1, 2, 3)
          w0 = k_ * lw[v+1] + (v == 0 ? 1 - k_ : 0.0); w1 = k_ * lw[v+1] + (v == 1 ? 1 - k_ : 0.0); w2 = k_ * lw[v+1] + (v == 2 ? 1 - k_ : 0.0)
          m0 = 1; m1 = 1; m2 = 1; e0 = ac[l0]; e1 = ac[l0+1]; e2 = ac[l0+2]; b0 = (l0 - 1) * NATOM; b1 = l0 * NATOM; b2 = (l0 + 1) * NATOM
          while m0 <= e0 || m1 <= e1 || m2 <= e2
            d0 = m0 <= e0 ? ad[b0+m0] : Inf; d1 = m1 <= e1 ? ad[b1+m1] : Inf; d2 = m2 <= e2 ? ad[b2+m2] : Inf
            if m0 <= e0 && d0 <= d1 && d0 <= d2
              c = buf_add!(X.bn, X.bd, base, c, CAPB, w0 * an[b0+m0], d0); m0 += 1
            elseif m1 <= e1 && d1 <= d2
              c = buf_add!(X.bn, X.bd, base, c, CAPB, w1 * an[b1+m1], d1); m1 += 1
            else
              c = buf_add!(X.bn, X.bd, base, c, CAPB, w2 * an[b2+m2], d2); m2 += 1
            end
          end
          X.bc[l0+v] = c
        end
        Hs = (H[i], H[i+1], H[i+2]); As = (A[i], A[i+1], A[i+2]); Ns = (N[i], N[i+1], N[i+2])
        for v in 0:2
          t = i + v; lt = l0 + v; base = (lt - 1) * CAPB; c = compact!(X.bn, X.bd, base, X.bc[lt]; ws=X.ws); n = 0.0; b = 0.0
          for m in 1:c; kk = (lt - 1) * NATOM + m; an[kk] = X.bn[base+m]; ad[kk] = X.bd[base+m]; n += an[kk]; b += ba(an[kk], ad[kk]); end
          ac[lt] = c; N[t] = n; B[t] = b
          n > 0 || continue
          H[t] = ((1 - k_) * Ns[v+1] * Hs[v+1] + k_ * lw[v+1] * th) / n; A[t] = ((1 - k_) * Ns[v+1] * As[v+1] + k_ * lw[v+1] * ta) / n
        end
      end
      for v in 0:2; ac[l0+v] > 0 && (no += 1; X.nl[no] = l0 + v); end
    end
    for q in 1:no; X.occ[q] = X.nl[q]; end; X.no = no                  # (triples visited in ascending order: the list stays sorted)
  end
  # recruitment into class 1 (stand RD of the start of the step)
  @inbounds for k in 1:S.nk; X.ks[k] = 0.0; end
  @inbounds for q in 1:X.no; i = X.occ[q] + lo - 1; S.cl[i] >= 3 && (X.ks[S.sp[i]] += N[i]); end
  nold = X.no
  @inbounds for k in 1:S.nk
    p = P[k]; seed = S.seed0[k, s] || X.ks[k] >= SEED_MIN
    rate = RSCALE * exp(p.lnF - G.λ0 * exp(G.λ1 * (p.ST - 3)) * RD) * W^G.ωR * (seed ? 1.0 : G.ε0) * Δ
    rate > 0 || continue
    h0 = ht(T_, p, G, k, D_REC)
    for v in 1:NL
      t = cell(S, s, k, 1, v); lt = t - lo + 1; add = rate * lw[v]; n0 = N[t]; nt = n0 + add; base = (lt - 1) * CAPB; c = 0
      ac[lt] == 0 && (X.no += 1; X.occ[X.no] = lt)
      for m in 1:ac[lt]; kk = (lt - 1) * NATOM + m; c = buf_add!(X.bn, X.bd, base, c, CAPB, an[kk], ad[kk]); end
      c = buf_add!(X.bn, X.bd, base, c, CAPB, add, D_REC); c = compact!(X.bn, X.bd, base, c; ws=X.ws)
      for m in 1:c; kk = (lt - 1) * NATOM + m; an[kk] = X.bn[base+m]; ad[kk] = X.bd[base+m]; end; ac[lt] = c
      H[t] = (n0 * H[t] + add * h0) / nt; A[t] = (n0 * A[t] + add * (A_REC - Δ)) / nt; N[t] = nt; B[t] += ba(add, D_REC)   # (A_REC − Δ: aged A_REC after A += Δ)
    end
  end
  X.no > nold && sortints!(X.occ, X.no)
  @inbounds for q in 1:X.no; i = X.occ[q] + lo - 1; N[i] > 0 && (A[i] += Δ); end
  nothing
end

"""Observation-space prediction of site s (X loaded): PN / PB / PH per (key, class) = Σ over the key's atoms of p N, p N π/4 D², p N H_m, with p the probability
that the atom is tallied in that class under the DBH measurement error N(0, sdM²) (lb of class 1 = −∞, ub of class 16 = ∞; classes beyond the neighbours of
the atom's own class are > 25 sd away and get nothing) and H_m = H + H*(D_m) − mean_N H*(D) the atom's allometric height around its cell's mean height,
CENTRED so that Σ N H_m = Σ N H (audit r5 I-2, H17: the offset H*(Dq) at the QMD biased class 1 by −2.2 %; the upgrowth conserves Σ N·H the same way, §2.4).
If an offset would put an atom below 0.5 m, the cell's H_m are clamped at 0.5 and rescaled to keep Σ N H_m = Σ N H. (audit r4 I1: with the cell mean, an
atom crossing a class bound took its new cell's height into the blurred tail of its old class.) X.adD is used as scratch for H*(D_m)."""
function finish_site!(S::Sites4, P::AbstractVector{SpP4}, G::GlobP4, X::Scratch, s::Int, T_::Tabs)
  lo = S.off[s]; nk = S.nk; b0 = (s - 1) * nk * NCLS; sd = G.sdM
  @inbounds for x in b0+1:b0+nk*NCLS; S.PN[x] = 0.0; S.PB[x] = 0.0; S.PH[x] = 0.0; end
  @inbounds for q in 1:X.no
    li = X.occ[q]; i = li + lo - 1; j = Int(S.cl[i]); k = Int(S.sp[i]); kb = b0 + (k - 1) * NCLS; Hc = S.H[i]; p = P[k]
    n_ = 0.0; hs_ = 0.0
    for m in 1:X.ac[li]; kk = (li - 1) * NATOM + m; X.adD[kk] = ht(T_, p, G, k, X.ad[kk]); n_ += X.an[kk]; hs_ += X.an[kk] * X.adD[kk]; end
    n_ > 0 || continue
    hbar = hs_ / n_; lowest = Inf
    for m in 1:X.ac[li]; kk = (li - 1) * NATOM + m; X.an[kk] > 0 && (lowest = min(lowest, Hc + X.adD[kk] - hbar)); end
    fsc = 1.0
    if lowest < 0.5                                                     # (rare: clamp and rescale, Σ N H_m = Σ N H kept)
      sh = 0.0; for m in 1:X.ac[li]; kk = (li - 1) * NATOM + m; sh += X.an[kk] * max(Hc + X.adD[kk] - hbar, 0.5); end
      fsc = n_ * Hc / sh
    end
    for m in 1:X.ac[li]
      kk = (li - 1) * NATOM + m; n = X.an[kk]; d = X.ad[kk]; n > 0 || continue; bb = ba(n, d)
      Hm = lowest < 0.5 ? max(Hc + X.adD[kk] - hbar, 0.5) * fsc : Hc + X.adD[kk] - hbar
      if sd > 0
        for t in max(1, j - 1):min(NCLS, j + 1)
          pr = (t == NCLS ? 1.0 : Φ((CLS_HI[t] - d) / sd)) - (t == 1 ? 0.0 : Φ((CLS_LO[t] - d) / sd))
          pr > 0 || continue
          S.PN[kb+t] += pr * n; S.PB[kb+t] += pr * bb; S.PH[kb+t] += pr * n * Hm
        end
      else
        S.PN[kb+j] += n; S.PB[kb+j] += bb; S.PH[kb+j] += n * Hm
      end
    end
  end
  nothing
end
"Observation-space values from the CELL TOTALS (N, B, N·H summed over the lanes; no blur) — for states without atoms (the persistence / transition-matrix nulls)."
function obs_cells!(S::Sites4)
  for s in 1:nsites(S), k in 1:S.nk, j in 1:NCLS
    x = obsix(S, s, k, j); i = cell(S, s, k, j, 1); S.PN[x] = 0.0; S.PB[x] = 0.0; S.PH[x] = 0.0
    for v in 0:NL-1; S.PN[x] += S.N[i+v]; S.PB[x] += S.B[i+v]; S.PH[x] += S.N[i+v] * S.H[i+v]; end
  end
  S
end

"Run site s from its initial state over its interval (nyr whole years + a final fraction) and write its observation-space prediction."
function run_site!(S::Sites4, P::AbstractVector{SpP4}, G::GlobP4, X::Scratch, s::Int, T_::Tabs)
  load_site!(S, X, s)
  for _ in 1:S.nyr[s]; step_site!(S, P, G, X, s, 1.0, T_); end
  S.frac[s] > 0 && step_site!(S, P, G, X, s, S.frac[s], T_)
  finish_site!(S, P, G, X, s, T_)
  nothing
end
"Run every site (threaded). The lanes are the store's (S.lz / S.lw); exact = true: no lookup tables (exact functions)."
function simulate!(S::Sites4, P::AbstractVector{SpP4}, G::GlobP4; exact::Bool=false)
  T_ = exact ? exact_tabs(length(P)) : Tabs(P, G)
  nmax = maximum(diff(S.off))
  XS = [Scratch(nmax) for _ in 1:Threads.maxthreadid()]
  Threads.@threads :static for s in 1:nsites(S)
    run_site!(S, P, G, XS[Threads.threadid()], s, T_)
  end
  S
end
end # module
