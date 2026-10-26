# Independent brute-force reference of one HBAD4P annual step (used by tools/hbad4/verify4p.jl), audit round 3 (cohorts-in-bins atoms). Written from
# docs/hbad4p_design.md §2 and the kernel's documented rules (header of src/plugins/HBAD4/HBAD4P.jl), NOT from the kernel code: exact functions (no lookup
# tables), O(n²) overtopping with the height ramp, per-cell atom LISTS (Vector of (n, D)) instead of fixed slots and scratch buffers, class membership by a
# search over the class bounds, totals N / B always recomputed from the atoms.
# The merge rule is the kernel's documented one (an atom arriving at a full inflow list merges into its nearest atom in D; a list longer than natom merges
# its closest pair, repeatedly; N and N·D² conserved), with the same tie order (the first pair in list order; the last atom takes the freed place), so that
# the reference can be compared atom by atom. Switches for the diffusion studies: natom / capb (Inf = no merging at all), ward = merge the pair with the
# smallest increase of Σ N D² dispersion (n_p n_q / (n_p + n_q) (D_p − D_q)²) instead of the closest pair, peratom = growth and mortality of every atom
# from its own diameter (instead of the cell's QMD).
module Ref4
const NC = 16; const NLR = 3
lo_(j) = 2.54 * (2j - 1); hi_(j) = j < NC ? 2.54 * (2j + 1) : Inf
const TOLD = 5.08e-9                 # the kernel's class-index tolerance 1e-9 (exact class bounds of recorded 0.1-in diameters) in cm
"Class of a diameter by a search over the class bounds (0 below 1 in)."
function classof(D)
  D < lo_(1) - TOLD && return 0
  j = 1; while j < NC && D >= lo_(j + 1) - TOLD; j += 1; end
  j
end
ba1(n, d) = n * pi * d * d / 4e4                                    # basal area [m²] of n trees of diameter d [cm]
hst(p, G, D) = 1.37 + (p.Hmax - 1.37) * (1 - exp(-G.s * D / (p.Hmax - 1.37)))^G.c
ramp(x, δH) = clamp(0.5 + x / (2δH), 0.0, 1.0)
const Atoms = Vector{NTuple{2,Float64}}
totN(v) = sum(a -> a[1], v; init=0.0)
totB(v) = sum(a -> ba1(a[1], a[2]), v; init=0.0)
qm(v) = (n = totN(v); n > 0 ? sqrt(4e4 * totB(v) / (pi * n)) : 0.0)
mergepair(a, b) = (m = a[1] + b[1]; (m, sqrt((a[1] * a[2]^2 + b[1] * b[2]^2) / m)))
"Add an atom to a list of capacity cap (a full list merges it into its nearest atom in D, the first one on ties)."
const TIE = Ref(false)                # set when a merge choice was a near tie (two candidates within 1e-9 relative): the choice is then decided by rounding
function add!(v::Atoms, n, d, cap)
  n > 0 || return v
  length(v) < cap && (push!(v, (n, d)); return v)
  q = 0; best = Inf; second = Inf
  for t in eachindex(v); e = abs(v[t][2] - d); e < best ? (second = best; best = e; q = t) : (e < second && (second = e)); end
  second - best <= 1e-9 * max(best, 1e-300) && (TIE[] = true)
  v[q] = mergepair(v[q], (n, d)); v
end
"""Merge until ≤ natom atoms: the closest pair in D (first in list order), or the Ward pair; the merged atom takes the first place, the last atom the second.
sorted = true (diffusion studies with large natom; O(c) per merge): the list is sorted by D and the best ADJACENT pair is merged (in 1-D the closest pair is
adjacent; ties may resolve differently)."""
function compact!(v::Atoms, natom; ward=false, sorted=false)
  if sorted
    length(v) > natom || return v
    sort!(v; by=last)
    while length(v) > natom
      p = 0; best = Inf
      for u in 1:length(v)-1
        e = ward ? v[u][1] * v[u+1][1] / (v[u][1] + v[u+1][1]) * (v[u+1][2] - v[u][2])^2 : v[u+1][2] - v[u][2]
        e < best && (best = e; p = u)
      end
      v[p] = mergepair(v[p], v[p+1]); deleteat!(v, p + 1)
    end
    return v
  end
  while length(v) > natom
    p = 0; q = 0; best = Inf; second = Inf
    for u in 1:length(v)-1, t in u+1:length(v)
      e = ward ? v[u][1] * v[t][1] / (v[u][1] + v[t][1]) * (v[u][2] - v[t][2])^2 : abs(v[u][2] - v[t][2])
      e < best ? (second = best; best = e; p = u; q = t) : (e < second && (second = e))
    end
    best > 0 && second - best <= 1e-9 * best && (TIE[] = true)     # (exact zero distances: merging equal diameters in either order is the same atom)
    v[p] = mergepair(v[p], v[q]); v[q] = v[end]; pop!(v)
  end
  v
end

"""One step of one site from the documented equations. C[i] = atoms of cell i (HBAD4P layout: key-major, class, lane), H / A = cell mean height / age.
seed0[k]: a key tree ≥ 12.7 cm on the plot at v1. Returns per-cell rates (O, dD, sv, g) and RD; C, H, A are updated in place."""
function step!(C::Vector{Atoms}, H, A, sp, cl, ln, nk, P, G, W, Wn, T, Cppm, pl, Δ, lz, lw, seed0; natom=4, capb=16, ward=false, peratom=false, sorted=false,
               CREF=390.1, RSCALE=185.2, DREC=2.54, AREC=4.5, SEEDMIN=1.0, δH=1.0)
  n = length(C); N = totN.(C); act = [i for i in 1:n if N[i] > 1e-9]
  Dq = qm.(C); SD = zeros(n); for i in act; SD[i] = N[i] * (Dq[i] / 25.4)^1.605; end
  RD = sum(SD) / G.sdimax
  O = zeros(n); for i in act; O[i] = sum(SD[j] * ramp(H[j] - H[i], δH) for j in act) / G.sdimax; end
  fH = pl ? exp(G.pH) : 1.0; fD = pl ? exp(G.pD) : 1.0; fM = (pl ? exp(G.pM) : 1.0) * exp(G.mW * (Wn - W) / 0.1); fC = 1 + G.βC * log(Cppm / CREF)
  g = ones(n); graw = zeros(n); dD = zeros(n); dH = zeros(n); sv = ones(n); gm = zeros(n); grow(p, D, h, gi) = max(exp(p.lnε) * (D / 20)^G.b * 122.98 * gi * fC * hydf(p, h) * fD - G.ρ * D^2 * h, 0.0) / (2 * D * max(h, 1.37)) * Δ
  hydf(p, h) = max(0.0, 1 - min(h / (1.1 * p.Hmax), 1.0)^G.θ)
  haz(p, D, i) = fM * (G.m0 / p.longevity + G.ms * exp(G.σST * (p.ST - 3)) * exp(-G.β * gm[i]) * (D / 20)^G.ζ) + exp(G.mA * (A[i] / p.longevity - 1))
  for i in act
    p = P[sp[i]]; D = Dq[i]; h = H[i]
    fT = isnan(T) ? 1.0 : exp(G.βT * (T - p.Tref))
    c = exp(G.gR * RD) * W^G.ω * fT * exp(G.σv * lz[ln[i]])          # no CO2: the FACE constraint is on growth (NPP) only
    gi = (G.f0 + (1 - G.f0) * exp(-G.a * O[i])) * c; g[i] = min(gi, 1.0); graw[i] = gi; gm[i] = exp(-G.am * O[i]) * c
    dD[i] = grow(p, D, h, gi)
    Lh = exp(-G.kh * O[i]); hs = hst(p, G, D)
    dH[i] = max((hst(p, G, D + dD[i]) - hs) * (1 + G.η * (1 - Lh)) + G.λ * Δ * (hs * (1 + G.μ * (1 - Lh)) - h), 0.0) * fH
    sv[i] = exp(-haz(p, D, i) * Δ)
  end
  # inflow lists: an inactive cell (0 < N ≤ 1e-9) keeps its atoms (no growth / mortality), with the self-thinning weight 1
  buf = [Atoms() for _ in 1:n]; inH = zeros(n); inA = zeros(n); inG = zeros(n); tie = falses(n)
  for i in 1:n
    i in act && continue
    TIE[] = false; for a in C[i]; add!(buf[i], a[1], a[2], capb); end; tie[i] |= TIE[]
    inH[i] = N[i] * H[i]; inA[i] = N[i] * A[i]; inG[i] = N[i]
  end
  for i in act
    p = P[sp[i]]; j = Int(cl[i]); Hn = H[i] + dH[i]
    tg = Int[]; gn = Float64[]; gb = Float64[]; nt = 0.0; b2 = 0.0
    for a in C[i]
      if peratom                                                    # (study switch: the atom's own diameter in the growth and size-mortality terms)
        nn = a[1] * exp(-haz(p, a[2], i) * Δ); d = a[2] + grow(p, a[2], H[i], graw[i])
      else
        nn = a[1] * sv[i]; d = a[2] + dD[i]
      end
      nn > 0 || continue
      t = j == NC ? NC : max(j, classof(d))
      u = findfirst(==(t), tg); u === nothing && (push!(tg, t); push!(gn, 0.0); push!(gb, 0.0); u = length(tg))
      gn[u] += nn; gb[u] += nn * d^2; nt += nn; b2 += nn * d^2
      TIE[] = false; add!(buf[i + (t - j) * NLR], nn, d, capb); tie[i + (t - j) * NLR] |= TIE[]
    end
    nt > 0 || continue
    hD = hst(p, G, sqrt(b2 / nt)); sh = sum(gn[u] * (Hn + hst(p, G, sqrt(gb[u] / gn[u])) - hD) for u in eachindex(tg))
    f = sh > 0 ? nt * Hn / sh : 1.0
    for u in eachindex(tg)
      c = i + (tg[u] - j) * NLR
      inH[c] += gn[u] * (Hn + hst(p, G, sqrt(gb[u] / gn[u])) - hD) * f; inA[c] += gn[u] * A[i]; inG[c] += gn[u] * g[i]
    end
  end
  gw = ones(n)
  for i in 1:n
    if isempty(buf[i]); C[i] = Atoms(); continue; end
    TIE[] = false; compact!(buf[i], natom; ward=ward, sorted=sorted); tie[i] |= TIE[]; C[i] = buf[i]; m = totN(C[i]); H[i] = inH[i] / m; A[i] = inA[i] / m; gw[i] = inG[i] / m
  end
  # self-thinning (≤ 6 passes): the excess SDI removed ∝ SDI (1 − g) + 1e-9 SDI, g = the realised relative growth (capped at 1) the trees brought in
  for _ in 1:6
    sd = [totN(C[i]) > 1e-12 ? totN(C[i]) * (qm(C[i]) / 25.4)^1.605 : 0.0 for i in 1:n]
    tot = sum(sd); ws = sum(sd[i] * (1 - gw[i]) + 1e-9 * sd[i] for i in 1:n); ex = tot - G.sdimax
    ex <= 1e-6 * G.sdimax && break
    for i in 1:n
      sd[i] > 0 || continue
      r = min(sd[i], ex * (sd[i] * (1 - gw[i]) + 1e-9 * sd[i]) / ws); f = 1 - r / sd[i]
      C[i] = [(a[1] * f, a[2]) for a in C[i]]
    end
  end
  # lane mixing: lane v keeps (1 − κΔ) of its atoms and receives κΔ w_v of the atoms of all three lanes of its (key, class)
  if G.κ > 0
    k_ = min(G.κ * Δ, 1.0)
    for i in 1:NLR:n
      Ns = [totN(C[i+v]) for v in 0:2]; tn = sum(Ns); tn > 0 || continue
      th = sum(Ns[v+1] * H[i+v] for v in 0:2); ta = sum(Ns[v+1] * A[i+v] for v in 0:2)
      new = [Atoms() for _ in 0:2]; TIE[] = false
      for v in 0:2
        for a in C[i+v]; add!(new[v+1], (1 - k_) * a[1], a[2], capb); end
        for u in 0:2, a in C[i+u]; add!(new[v+1], k_ * lw[v+1] * a[1], a[2], capb); end
        compact!(new[v+1], natom; ward=ward, sorted=sorted)
      end
      TIE[] && (tie[i:i+2] .= true)
      for v in 0:2
        t = i + v; C[t] = new[v+1]; m = totN(C[t]); m > 0 || continue
        H[t] = ((1 - k_) * Ns[v+1] * H[t] + k_ * lw[v+1] * th) / m; A[t] = ((1 - k_) * Ns[v+1] * A[t] + k_ * lw[v+1] * ta) / m
      end
    end
  end
  # recruitment at the end of the year into class 1 (RD of the START of the step; seed: a v1 seed tree or ≥ SEEDMIN trees/ha ≥ 12.7 cm now)
  for k in 1:nk
    p = P[k]; base = (k - 1) * NC * NLR
    seed = seed0[k] || sum(totN(C[base+(j-1)*NLR+v]) for j in 3:NC for v in 1:NLR) >= SEEDMIN
    rate = RSCALE * exp(p.lnF - G.λ0 * exp(G.λ1 * (p.ST - 3)) * RD) * W^G.ωR * (seed ? 1.0 : G.ε0) * Δ
    rate > 0 || continue
    h0 = hst(p, G, DREC)
    for v in 1:NLR
      t = base + v; add = rate * lw[v]; n0 = totN(C[t]); nb = Atoms(); TIE[] = false; for a in C[t]; add!(nb, a[1], a[2], capb); end
      add!(nb, add, DREC, capb); compact!(nb, natom; ward=ward, sorted=sorted); tie[t] |= TIE[]; C[t] = nb
      H[t] = (n0 * H[t] + add * h0) / (n0 + add); A[t] = (n0 * A[t] + add * (AREC - Δ)) / (n0 + add)
    end
  end
  for i in 1:n; totN(C[i]) > 0 && (A[i] += Δ); end
  (O=O, dD=dD, sv=sv, g=g, RD=RD, tie=tie)
end
end # module Ref4
