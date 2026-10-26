# Independent brute-force reference of one HBAD4P annual step (used by tools/hbad4/verify4p.jl). Written from docs/hbad4p_design.md §2, NOT from the
# kernel: exact functions (no lookup tables), O(n²) overtopping with the height ramp, and the truncated-exponential upgrowth integrated numerically in
# DIAMETER space (adaptive Gauss–Kronrod, QuadGK) instead of the kernel's closed-form partial moments in ξ space. The closure parameter a of a cohort is
# taken from a caller-supplied function (afun(j, q2)): the kernel's inversion (to test the implementation at 1e-9) or an exact BigFloat root (inversion test).
module Ref4
import Distributions
const QG = Base.loaded_modules[Base.PkgId(Base.UUID("1fd47b50-473d-5c70-9696-f719f8f3bcdc"), "QuadGK")]   # (a dependency of Distributions)
const NC = 16; const NLR = 3; const WCL = 5.08
lo_(j) = 2.54 * (2j - 1); hi_(j) = j < NC ? 2.54 * (2j + 1) : Inf
qm(N, B) = N > 0 ? sqrt(4e4 * B / (pi * N)) : 0.0              # QMD [cm] of N trees with basal area B [m²]
bam(N, D2) = N * pi * D2 / 4e4                                  # basal area [m²] of N trees of mean square diameter D2 [cm²]
hst(p, G, D) = 1.37 + (p.Hmax - 1.37) * (1 - exp(-G.s * D / (p.Hmax - 1.37)))^G.c
ramp(x, δH) = clamp(0.5 + x / (2δH), 0.0, 1.0)

"Truncated exponential on [L, U) in diameter space, f(D) ∝ exp(c (D − M)), c = a / w, M = (L + U)/2: ∫_{x0}^{x1} g(D) f(D) dD (normalised)."
function dint(g, a, L, U, x0, x1)
  x1 <= x0 && return 0.0
  c = a / (U - L); M = (L + U) / 2
  Z = QG.quadgk(D -> exp(c * (D - M)), L, U; rtol=1e-13, atol=0.0)[1]
  QG.quadgk(D -> g(D) * exp(c * (D - M)), x0, x1; rtol=1e-13, atol=0.0)[1] / Z
end
"Exact closure parameter a of class j for mean square diameter q2: BigFloat bisection on E_a[D²] (closed form in D space), clamped to ±amax."
function aexact(j, q2; amax=200.0)
  L = big(lo_(j)); w = big(WCL)
  e2(a) = begin
    a == 0 && return L^2 + L * w + w^2 / 3
    c = a / w; Z = expm1(c * w) / c
    m1 = (exp(c * w) * (w / c - 1 / c^2) + 1 / c^2) / Z; m2 = (exp(c * w) * (w^2 / c - 2w / c^2 + 2 / c^3) - 2 / c^3) / Z   # ∫ u^k e^{cu}, u = D − L
    L^2 + 2L * m1 + m2
  end
  lo = big(-amax); hi = big(amax); t = big(q2)
  t <= e2(lo) && return -amax; t >= e2(hi) && return amax
  for _ in 1:120; m = (lo + hi) / 2; e2(m) < t ? (lo = m) : (hi = m); end
  Float64((lo + hi) / 2)
end

"""Upgrowth of the survivors (n, b) of a class-j (< NC) cohort growing by dD, by numerical integration: [(t, Nt, Bt)] for every class t ≥ j reached, the stayers
(t = j) taking B1 − Σ movers' B (conservation, as the spec requires); returns (groups, B1, ED)."""
function upgrowth(j, n, b, dD, afun)
  L = lo_(j); U = hi_(j); q2 = qm(n, b)^2; a = afun(j, q2)
  ED = dint(D -> D, a, L, U, L, U)
  B1 = b + n * pi / 4e4 * (2dD * ED + dD^2)
  g = Tuple{Int,Float64,Float64}[]; Bm = 0.0
  for t in j+1:NC
    x0 = max(L, lo_(t) - dD); x1 = min(U, hi_(t) - dD)             # trees whose D + dD lands in [lo_t, hi_t)
    x1 > x0 || continue
    P = dint(D -> 1.0, a, L, U, x0, x1); P > 0 || continue
    Bt = bam(n, dint(D -> (D + dD)^2, a, L, U, x0, x1)); Bm += Bt
    push!(g, (t, n * P, Bt))
  end
  Ps = dint(D -> 1.0, a, L, U, L, min(U, U - dD))
  Ps > 0 && pushfirst!(g, (j, n * Ps, B1 - Bm))
  g, B1, ED
end

"""One step of one site straight from the documented equations (cells in the HBAD4P layout: key-major, class, lane). seed0[k]: a key tree ≥ 12.7 cm on the
plot at v1. overwrite = true mimics the kernel's overwrite of an inactive remnant (N ≤ 1e-9) by an inflow. dall = :qmd uses the kernel's E[D] ≈ QMD for the cohort's grown QMD in the height offset (:exact = the closure's E[D])."""
function step!(N, B, H, A, sp, cl, ln, nk, P, G, W, Wn, T, C, pl, Δ, lz, lw, seed0, afun; CREF=390.1, RSCALE=185.2, DREC=2.54, AREC=4.5, SEEDMIN=1.0, δH=1.0, dall=:qmd, overwrite=false)
  n = length(N); act = [i for i in 1:n if N[i] > 1e-9]
  Dq = [qm(N[i], B[i]) for i in 1:n]
  SD = zeros(n); for i in act; SD[i] = N[i] * (Dq[i] / 25.4)^1.605; end
  RD = sum(SD) / G.sdimax
  O = zeros(n)
  for i in act; O[i] = sum(SD[j] * ramp(H[j] - H[i], δH) for j in act) / G.sdimax; end
  fH = pl ? exp(G.pH) : 1.0; fD = pl ? exp(G.pD) : 1.0; fM = (pl ? exp(G.pM) : 1.0) * exp(G.mW * (Wn - W) / 0.1)
  gcap = ones(n); dD = zeros(n); dH = zeros(n); sv = ones(n)
  for i in act
    p = P[sp[i]]; D = Dq[i]; h = H[i]
    fT = (isinf(G.wT) || isnan(T)) ? 1.0 : exp(-((T - p.Topt)^2 - (p.Tref - p.Topt)^2) / G.wT^2)
    cw = exp(G.gR * RD) * W^G.ω * fT * (1 + G.βC * log(C / CREF)) * exp(G.σv * lz[ln[i]])
    g = (G.f0 + (1 - G.f0) * exp(-G.a * O[i])) * cw; gcap[i] = min(g, 1.0)
    gm = exp(-G.am * O[i]) * cw
    hyd = max(0.0, 1 - min(h / (1.1 * p.Hmax), 1.0)^G.θ)
    dV = max(exp(p.lnε) * (D / 20)^G.b * 122.98 * g * hyd * fD - G.ρ * D^2 * h, 0.0)
    dD[i] = dV / (2 * D * max(h, 1.37)) * Δ
    Lh = exp(-G.kh * O[i]); hs = hst(p, G, D)
    dH[i] = max((hst(p, G, D + dD[i]) - hs) * (1 + G.η * (1 - Lh)) + G.λ * Δ * (hs * (1 + G.μ * (1 - Lh)) - h), 0.0) * fH
    hz = fM * (G.m0 / p.longevity + G.ms * exp(G.σST * (p.ST - 3)) * exp(-G.β * gm) * (D / 20)^G.ζ) + exp(G.mA * (A[i] / p.longevity - 1))
    sv[i] = exp(-hz * Δ)
  end
  inN = zeros(n); inB = zeros(n); inH = zeros(n); inA = zeros(n)
  for i in act
    p = P[sp[i]]; j = Int(cl[i]); nn = N[i] * sv[i]; bb = B[i] * sv[i]; Hn = H[i] + dH[i]
    N[i] = 0.0; B[i] = 0.0
    if j < NC && dD[i] > 0
      grp, B1, ED = upgrowth(j, nn, bb, dD[i], afun)
      Dall = dall == :qmd ? qm(nn, bb + nn * pi / 4e4 * (2dD[i] * qm(nn, bb) + dD[i]^2)) : qm(nn, B1)
      hD = hst(p, G, Dall)
      tt = sum(Nt * (Hn + hst(p, G, qm(Nt, Bt)) - hD) for (t, Nt, Bt) in grp); f = tt > 0 ? nn * Hn / tt : 1.0
      for (t, Nt, Bt) in grp
        c = i + (t - j) * NLR
        inN[c] += Nt; inB[c] += Bt; inH[c] += Nt * (Hn + hst(p, G, qm(Nt, Bt)) - hD) * f; inA[c] += Nt * A[i]
      end
    else
      b1 = j == NC ? bb + nn * pi / 4e4 * (2dD[i] * qm(nn, bb) + dD[i]^2) : bb
      inN[i] += nn; inB[i] += b1; inH[i] += nn * Hn; inA[i] += nn * A[i]
    end
  end
  for i in 1:n
    inN[i] > 0 || continue
    overwrite && (N[i] = 0.0; B[i] = 0.0)      # (kernel: an inactive remnant N ≤ 1e-9 receiving an inflow is overwritten, HBAD4P.jl:249)
    nt = N[i] + inN[i]; H[i] = (N[i] * H[i] + inH[i]) / nt; A[i] = (N[i] * A[i] + inA[i]) / nt; B[i] += inB[i]; N[i] = nt
  end
  for _ in 1:6
    sd = [N[i] > 1e-12 ? N[i] * (qm(N[i], B[i]) / 25.4)^1.605 : 0.0 for i in 1:n]
    tot = sum(sd); ws = sum(sd[i] * (1 - gcap[i]) + 1e-9 * sd[i] for i in 1:n); ex = tot - G.sdimax
    ex <= 1e-6 * G.sdimax && break
    for i in 1:n
      sd[i] > 0 || continue
      r = min(sd[i], ex * (sd[i] * (1 - gcap[i]) + 1e-9 * sd[i]) / ws); f = 1 - r / sd[i]; N[i] *= f; B[i] *= f
    end
  end
  if G.κ > 0
    k_ = min(G.κ * Δ, 1.0)
    for i in 1:NLR:n
      tn = N[i] + N[i+1] + N[i+2]; tn > 0 || continue
      tb = B[i] + B[i+1] + B[i+2]; th = sum(N[i+v] * H[i+v] for v in 0:2); ta = sum(N[i+v] * A[i+v] for v in 0:2)
      for v in 0:2
        t = i + v; nn = (1 - k_) * N[t] + k_ * lw[v+1] * tn; nn > 0 || continue
        H[t] = ((1 - k_) * N[t] * H[t] + k_ * lw[v+1] * th) / nn; A[t] = ((1 - k_) * N[t] * A[t] + k_ * lw[v+1] * ta) / nn
        B[t] = (1 - k_) * B[t] + k_ * lw[v+1] * tb; N[t] = nn
      end
    end
  end
  for k in 1:nk
    p = P[k]; base = (k - 1) * NC * NLR
    seed = seed0[k] || sum(N[base+(j-1)*NLR+v] for j in 3:NC for v in 1:NLR) >= SEEDMIN
    rate = RSCALE * exp(p.lnF - G.λ0 * exp(G.λ1 * (p.ST - 3)) * RD) * W^G.ωR * (seed ? 1.0 : G.ε0) * Δ
    rate > 0 || continue
    h0 = hst(p, G, DREC)
    for v in 1:NLR
      t = base + v; add = rate * lw[v]; nt = N[t] + add
      H[t] = (N[t] * H[t] + add * h0) / nt; A[t] = (N[t] * A[t] + add * (AREC - Δ)) / nt; N[t] = nt; B[t] += bam(add, DREC^2)
    end
  end
  for i in 1:n; N[i] > 0 && (A[i] += Δ); end
  (O=O, dD=dD, RD=RD)
end
end # module Ref4
