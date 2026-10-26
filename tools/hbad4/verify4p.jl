# HBAD4P verification (docs/hbad4p_design.md §5): PASS / FAIL checks by code paths INDEPENDENT of the code under test —
#   kernel:      a brute-force reference step written from the documented equations (exact functions, O(n²) overtopping, the upgrowth integrals
#                re-derived in diameter space) vs the optimised kernel; Monte-Carlo samples of the linear density vs the analytic upgrowth; a
#                tree-level replay of the real v1 trees vs the two-moment class transfer; lookup tables vs exact functions; determinism / thread parity;
#   data:        SQL on the read-only raw databases (slim_r4 curated_trees + REF_SPECIES, FIADB PLOT / COND / SUBP_COND) vs the Julia builder
#                (interval selection, species keys, v1 stand table, v2 stand table, forested fractions, tally areas), CSV re-joins (climate, CO2, traits);
#   likelihood:  Distributions.jl vs the hand-written NB pmf; nll4 recomputed from its definition.
# Checks tagged [spec] test the code against the design document. Exit code 1 on any FAIL.
#   HBAD2_DATA_DIR=<fold product> ./julia_gdal.sh --project=. --threads=2 tools/hbad4/verify4p.jl [hbad3p fit dir]
# env: HBAD4_VERIFY_NSUB (sites of the brute-force checks, default 300), HBAD4_SPLIT_PLOTS, HBAD4_ROLE_SPLIT, HBAD3_SLIM_DB, HBAD4_FIADB.
include(joinpath(@__DIR__, "data4.jl")); include(joinpath(@__DIR__, "params4.jl"))
using DuckDB, Random
import Distributions
const H4 = HBAD4P
const ROLE = basename(dirname(normpath(DATA_DIR)))                                           # e.g. k1 (product = …/prep/<role>/product)
const FIT3P = length(ARGS) >= 1 ? ARGS[1] : get(ENV, "HBAD4_FIT3P", joinpath(DATA_DIR, "..", "..", "..", "hbad3p", ROLE, "fit3p_" * ROLE))
const SLIM_DB = get(ENV, "HBAD3_SLIM_DB", joinpath(WSROOT, "landis_region4", "data_r4", "slim_r4.duckdb"))
const FIADB = get(ENV, "HBAD4_FIADB", joinpath(WSROOT, "FIA", "FIASQLITE2PGSQL", "FIADB.duckdb"))
const SPLIT_PLOTS = get(ENV, "HBAD4_SPLIT_PLOTS", joinpath(DATA_DIR, "..", "split", "split_plots.csv"))
const ROLE_SPLIT = get(ENV, "HBAD4_ROLE_SPLIT", joinpath(DATA_DIR, "..", "..", "..", "folds", "split_$(ROLE).csv"))
const FOLDS_CSV = joinpath(dirname(ROLE_SPLIT), "folds.csv")
const NSUB = parse(Int, get(ENV, "HBAD4_VERIFY_NSUB", "300"))
const ACRE = 4046.8564224

# ---------------------------------------------------------------- reporting
const RES = Tuple{String,String,String}[]
function check(name, ok::Bool, detail="")
  push!(RES, (ok ? "PASS" : "FAIL", name, detail)); println(ok ? "PASS  " : "FAIL  ", name, isempty(detail) ? "" : "  — " * detail); flush(stdout); ok
end
info(name, detail) = (push!(RES, ("INFO", name, detail)); println("INFO  ", name, "  — ", detail); flush(stdout))
section(s) = (println("\n==== ", s); flush(stdout))
reld(a, b; floor=1e-12) = abs(a - b) / max(abs(a), abs(b), floor)
fmt(x) = @sprintf("%.4g", x)

# ---------------------------------------------------------------- independent reference kernel (diameter-space derivation, exact functions)
module Ref4
const NC = 16; const NLR = 3
lo_(j) = 2.54 * (2j - 1); hi_(j) = j < NC ? 2.54 * (2j + 1) : Inf
qm(N, B) = N > 0 ? sqrt(4e4 * B / (pi * N)) : 0.0              # QMD [cm] of N trees with basal area B [m²]
bam(N, D2) = N * pi * D2 / 4e4                                  # basal area [m²] of N trees of mean square diameter D2 [cm²]
hst(p, G, D) = 1.37 + (p.Hmax - 1.37) * (1 - exp(-G.s * D / (p.Hmax - 1.37)))^G.c
"∫_a^b Σ_k coefs[k] D^(k−1) dD"
pint(cf, a, b) = sum(c * (b^k - a^k) / k for (k, c) in enumerate(cf))
"Linear density on class j in diameter space f(D) = α + βD (∫ = 1), slope s ∈ [−1, 1] from the mean square diameter q2 (E[D²] is linear in s)."
function density(j, q2)
  L, U = lo_(j), hi_(j); w = U - L; c = (L + U) / 2
  e20 = pint([0.0, 0.0, 1.0], L, U) / w                         # E[D²] at s = 0 (uniform)
  de2 = 2 / w^2 * (pint([0.0, 0.0, 0.0, 1.0], L, U) - c * pint([0.0, 0.0, 1.0], L, U))   # ∂E[D²]/∂s
  s = clamp((q2 - e20) / de2, -1.0, 1.0); β = 2s / w^2; α = 1 / w - β * c
  (α=α, β=β, s=s, L=L, U=U, sraw=(q2 - e20) / de2)
end
"Movers of a class-j cohort (N, B) growing by dD (all trees the same dD): (Nm, Bm, B1) by direct integration of the density over [U − dD, U]."
function movers(j, N, B, dD)
  q2 = qm(N, B)^2; f = density(j, q2)
  ED = pint([0.0, f.α, f.β], f.L, f.U)
  B1 = bam(N, q2 + 2dD * ED + dD^2)
  Dlo = clamp(f.U - dD, f.L, f.U)
  Nm = N * pint([f.α, f.β], Dlo, f.U)
  Em2 = pint([f.α * dD^2, 2f.α * dD + f.β * dD^2, f.α + 2f.β * dD, f.β], Dlo, f.U)   # ∫ (D + dD)² f(D) dD over the movers
  Bm = min(bam(N, Em2), B1)
  (Nm, Bm, B1)
end
"One step of one site (cells in the HBAD4P layout: key-major, class, lane), straight from the documented equations."
function step!(N, B, H, A, sp, cl, ln, nk, P, G, W, Wn, T, C, pl, Δ, lz, lw; CREF=390.1, RSCALE=185.2, DREC=2.54, AREC=5.0)
  n = length(N); act = [i for i in 1:n if N[i] > 1e-9]
  Dq = [qm(N[i], B[i]) for i in 1:n]
  SD = zeros(n); for i in act; SD[i] = N[i] * (Dq[i] / 25.4)^1.605; end
  RD = sum(SD) / G.sdimax
  O = zeros(n)
  for i in act
    O[i] = (sum((SD[j] for j in act if H[j] > H[i]); init=0.0) + 0.5 * sum((SD[j] for j in act if H[j] == H[i]); init=0.0)) / G.sdimax
  end
  fH = pl ? exp(G.pH) : 1.0; fD = pl ? exp(G.pD) : 1.0; fM = (pl ? exp(G.pM) : 1.0) * exp(G.mW * (Wn - W) / 0.1)
  gcap = ones(n); dD = zeros(n); dH = zeros(n); sv = ones(n)
  for i in act
    p = P[sp[i]]; D = Dq[i]; h = H[i]
    fT = (isinf(G.wT) || isnan(T)) ? 1.0 : exp(-((T - p.Topt) / G.wT)^2)
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
    p = P[sp[i]]; j = cl[i]; nn = N[i] * sv[i]; bb = B[i] * sv[i]; Hn = H[i] + dH[i]
    if j < NC && dD[i] > 0
      Nm, Bm, B1 = movers(j, nn, bb, dD[i])
      if Nm > 0
        Ns = nn - Nm; Bs = B1 - Bm; Dall = qm(nn, B1); Dm = qm(Nm, Bm); Ds = Ns > 0 ? qm(Ns, Bs) : 0.0
        hm = Hn + hst(p, G, Dm) - hst(p, G, Dall); hs_ = Hn + (Ns > 0 ? hst(p, G, Ds) - hst(p, G, Dall) : 0.0)
        tt = Nm * hm + Ns * hs_; f = tt > 0 ? nn * Hn / tt : 1.0
        t = i + NLR
        inN[t] += Nm; inB[t] += Bm; inH[t] += Nm * hm * f; inA[t] += Nm * A[i]
        N[i] = Ns; B[i] = Bs; H[i] = hs_ * f
      else
        N[i] = nn; B[i] = B1; H[i] = Hn
      end
    else
      N[i] = nn; B[i] = j == NC ? bam(nn, (qm(nn, bb) + dD[i])^2) : bb; H[i] = Hn
    end
  end
  for i in 1:n
    inN[i] > 0 || continue
    nt = N[i] + inN[i]; H[i] = (N[i] * H[i] + inH[i]) / nt; A[i] = (N[i] * A[i] + inA[i]) / nt; N[i] = nt; B[i] += inB[i]
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
    seed = any(N[base+(j-1)*NLR+v] > 1e-9 for j in 3:NC for v in 1:NLR)
    rate = RSCALE * exp(p.lnF - G.λ0 * exp(G.λ1 * (p.ST - 3)) * RD) * W^G.ωR * (seed ? 1.0 : G.ε0) * Δ
    rate > 0 || continue
    h0 = hst(p, G, DREC)
    for v in 1:NLR
      t = base + v; add = rate * lw[v]; nt = N[t] + add
      H[t] = (N[t] * H[t] + add * h0) / nt; A[t] = (N[t] * A[t] + add * AREC) / nt; N[t] = nt; B[t] += bam(add, DREC^2)
    end
  end
  for i in 1:n; N[i] > 0 && (A[i] += Δ); end
  nothing
end
end # module Ref4

"A Sites4 of a subset of sites (all sites have the same number of cells in this layout)."
function subS(S, idx)
  nc = S.off[2] - S.off[1]; all(diff(S.off) .== nc) || error("subS: sites of unequal size")
  r = reduce(vcat, [collect(S.off[s]:S.off[s+1]-1) for s in idx])
  H4.Sites4([1 + (q - 1) * nc for q in 1:length(idx)+1], S.nk, S.sp[r], S.cl[r], S.ln[r], S.N0[r], S.B0[r], S.H0[r], S.A0[r],
            copy(S.N0[r]), copy(S.B0[r]), copy(S.H0[r]), copy(S.A0[r]), S.nyr[idx], S.frac[idx], S.planted[idx], S.W[idx], S.Wn[idx], S.T[idx], S.C[idx])
end
emptytabs(nk) = H4.Tabs(zeros(0, nk), zeros(0), zeros(0), zeros(0), zeros(0))
cellix(S, s, k, j, v) = S.off[s] + ((k - 1) * NCLS + (j - 1)) * NL + (v - 1)
withG(G; kw...) = H4.GlobP4((haskey(kw, f) ? kw[f] : getfield(G, f) for f in fieldnames(H4.GlobP4))...)
withP(p; kw...) = H4.SpP4((haskey(kw, f) ? kw[f] : getfield(p, f) for f in fieldnames(H4.SpP4))...)

println("HBAD4P verification | product $(DATA_DIR) | role $(ROLE) | threads $(Threads.nthreads()) | region $(REGION) | NSUB $(NSUB)")
t00 = time()
CS = cached_sets(["train", "val", "test"]); KEYS = CS.keys; TR = CS.TR; nk = length(KEYS)
DT = CS.sets["train"]; ST = DT.S; ns = H4.nsites(ST)
e3, f3, g3 = hbad3p_start(FIT3P, KEYS); SP = space4(KEYS, TR; start_eps=e3, start_lnF=f3, start_glob=g3); P, G, Ob = SP.decode(SP.z0)
lz, lw = H4.lanes(:sym)
rng = MersenneTwister(20261026); SUB = sort(randperm(rng, ns)[1:min(NSUB, ns)])
println("sets loaded in $(round(time() - t00; digits=1)) s; train sites $(ns), keys $(join(KEYS, ","))")

# ================================================================ 0. hand-computed single-site cases (one cohort, one year; closed forms written out here)
section("0. hand-computed one-cohort / one-year cases (closed forms, independent of the kernel and of Ref4)")
"One site, nkk keys; cohorts (k, j, v, N, QMD, H, A); Δ = 1 → one whole year, else one partial year."
function onesite(cohorts; nkk=1, Δ=1.0, W=0.9, Wn=0.95, T=18.0, C=400.0, pl=false)
  nc = nkk * NCLS * NL
  sp = Int32[k for k in 1:nkk for _ in 1:NCLS for _ in 1:NL]; cl = Int8[j for _ in 1:nkk for j in 1:NCLS for _ in 1:NL]; ln = Int8[v for _ in 1:nkk for _ in 1:NCLS for v in 1:NL]
  N0 = zeros(nc); B0 = zeros(nc); H0 = zeros(nc); A0 = zeros(nc)
  for (k, j, v, N, D, Hh, A) in cohorts; i = ((k - 1) * NCLS + (j - 1)) * NL + v; N0[i] = N; B0[i] = N * π / 4e4 * D^2; H0[i] = Hh; A0[i] = A; end
  H4.Sites4([1, nc + 1], nkk, sp, cl, ln, N0, B0, H0, A0, copy(N0), copy(B0), copy(H0), copy(A0), [Δ == 1.0 ? 1 : 0], [Δ == 1.0 ? 0.0 : Δ], [pl], [W], [Wn], [T], [C])
end
ci(k, j, v) = ((k - 1) * NCLS + (j - 1)) * NL + v
Gh = withG(G; b=2.0, a=2.0, gR=0.0, θ=8.0, ω=1.0, s=1.2, c=1.2, η=0.0, λ=0.0, μ=0.0, kh=5.0, m0=0.0, ms=0.0, σST=0.0, β=2.0, ζ=-0.5, mW=0.0, mA=1e4,
           pD=0.0, pH=0.0, pM=0.0, σv=0.3, sdimax=1500.0, am=1.0, ρ=0.0, f0=0.0, wT=Inf, βC=0.0, κ=0.0, λ0=3.0, λ1=-0.07, ε0=0.3, ωR=0.0)
Ph = H4.SpP4(0.0, -Inf, 30.0, 3.0, 1e12, 18.0)            # ε = 1, no recruitment, Hmax 30 m, ST 3, no senescence, T_opt 18
hs_(D; Hm=30.0, s=1.2, c=1.2) = 1.37 + (Hm - 1.37) * (1 - exp(-s * D / (Hm - 1.37)))^c
# 0.1 growth + upgrowth of a uniform cohort (slope 0: QMD² = c² + w²/12) in class 3, mid lane, alone on the site
let c = 15.24, w = 5.08, U = 17.78, N = 100.0, Hh = 14.0, A = 40.0
  q2 = c^2 + w^2 / 12; Dq = sqrt(q2)
  O = 0.5 * N * (Dq / 25.4)^1.605 / 1500                     # own SDI only, half of it
  g = exp(-2.0 * O) * 0.9                                    # light × W^ω (ω = 1, W = 0.9)
  hyd = 1 - (Hh / 33.0)^8                                    # 1.1 Hmax = 33 m
  dV = (Dq / 20)^2 * 122.98 * g * hyd; dD = dV / (2 * Dq * Hh)
  Nm = N * dD / w                                            # uniform density: the top dD of the class moves
  Bm = π / 4e4 * (N / w) * ((U + dD)^3 - U^3) / 3            # ∫_{U−dD}^{U} (D + dD)² dD × N / w
  B1 = π / 4e4 * N * (q2 + 2dD * c + dD^2)                   # E[(D + dD)²] with E[D] = c
  dH = hs_(Dq + dD) - hs_(Dq)
  S1 = onesite([(1, 3, 1, N, Dq, Hh, A)]); H4.simulate!(S1, [Ph], Gh; exact=true)
  i3 = ci(1, 3, 1); i4 = ci(1, 4, 1)
  ok = isapprox(S1.N[i4], Nm; rtol=1e-10) && isapprox(S1.N[i3], N - Nm; rtol=1e-10) && isapprox(S1.B[i4], Bm; rtol=1e-10) && isapprox(S1.B[i3], B1 - Bm; rtol=1e-10) &&
       isapprox(S1.N[i3] * S1.H[i3] + S1.N[i4] * S1.H[i4], N * (Hh + dH); rtol=1e-10) && S1.A[i3] == A + 1 && S1.A[i4] == A + 1
  check("0.1 one uniform class-3 cohort, one year: dD, movers N = N·dD/w, movers B, B after growth, Σ N·H = N (H + ΔH*), ages", ok,
        @sprintf("dD %.5f cm: N3 %.6f / %.6f, N4 %.6f / %.6f, B3 %.7f / %.7f, B4 %.7f / %.7f (kernel / hand)", dD, S1.N[i3], N - Nm, S1.N[i4], Nm, S1.B[i3], B1 - Bm, S1.B[i4], Bm))
  # 0.2 partial year Δ = 0.4: dD × 0.4
  S2 = onesite([(1, 3, 1, N, Dq, Hh, A)]; Δ=0.4); H4.simulate!(S2, [Ph], Gh; exact=true)
  check("0.2 partial final year Δ = 0.4: movers = N · 0.4 dD / w, age + 0.4", isapprox(S2.N[i4], N * 0.4dD / w; rtol=1e-10) && isapprox(S2.A[i3], A + 0.4; rtol=1e-14),
        @sprintf("N4 %.6f vs %.6f", S2.N[i4], N * 0.4dD / w))
  # 0.3 multiplicative growth modifiers: amplified lane (z = +√2), planted (pD), temperature (wT, T_opt), CO2 (βC): movers ∝ dD
  Gm = withG(Gh; pD=0.2, wT=5.0, βC=0.58)
  S3 = onesite([(1, 3, 3, N, Dq, Hh, A)]; pl=true, T=20.0, C=420.0); H4.simulate!(S3, [Ph], Gm; exact=true)
  fac = exp(0.3 * sqrt(2)) * exp(0.2) * exp(-((20.0 - 18.0) / 5.0)^2) * (1 + 0.58 * log(420 / 390.1))
  check("0.3 growth multipliers e^{σv z} (z = +√2) · e^{pD} · f_T · f_CO2 scale dD exactly", isapprox(S3.N[ci(1, 4, 3)], N * fac * dD / w; rtol=1e-10),
        @sprintf("movers %.6f vs hand %.6f (factor %.5f)", S3.N[ci(1, 4, 3)], N * fac * dD / w, fac))
end
# 0.4 survival (growth off): h = fM [m0/LONGEVITY + ms e^{σST (ST − 3)} e^{−β gm} (D/20)^ζ] + e^{mA (A/LONGEVITY − 1)}, fM = e^{pM} e^{mW (Wn − W)/0.1}
let N = 80.0, Dq = 25.0, A = 100.0, Lg = 200.0, Δ = 0.37
  Gs = withG(Gh; m0=1.0, ms=0.05, σST=0.3, β=2.0, am=1.0, ζ=-0.5, mA=10.0, mW=0.2, pM=0.1)
  Ps = H4.SpP4(-Inf, -Inf, 30.0, 4.0, Lg, 18.0)
  O = 0.5 * N * (Dq / 25.4)^1.605 / 1500; gm = exp(-O) * 0.9
  h = exp(0.1) * exp(0.2 * (0.95 - 0.9) / 0.1) * (1.0 / Lg + 0.05 * exp(0.3) * exp(-2gm) * (Dq / 20)^-0.5) + exp(10 * (A / Lg - 1))
  S4 = onesite([(1, 5, 1, N, Dq, 20.0, A)]; pl=true, Δ=Δ); H4.simulate!(S4, [Ps], Gs; exact=true); i = ci(1, 5, 1)
  check("0.4 survival of one cohort over a partial year (Δ = 0.37, planted, drought anomaly): N = N0 e^{−hΔ}, QMD unchanged", isapprox(S4.N[i], N * exp(-h * Δ); rtol=1e-12) &&
        isapprox(H4.qmd(S4.N[i], S4.B[i]), Dq; rtol=1e-12), @sprintf("h %.6f /yr: N %.8f vs %.8f", h, S4.N[i], N * exp(-h * Δ)))
end
# 0.5 recruitment on an empty site, and with a seed cohort (classes ≥ 3) present
let lnF = -1.0, Pr = H4.SpP4(-Inf, -1.0, 30.0, 4.0, 1e12, 18.0), Gr = withG(Gh; ωR=0.5)
  S5 = onesite(Tuple{Int,Int,Int,Float64,Float64,Float64,Float64}[]); H4.simulate!(S5, [Pr], Gr; exact=true)
  rate = 185.2 * exp(lnF) * 0.9^0.5 * 0.3
  ok = all(isapprox(S5.N[ci(1, 1, v)], rate * (0.5, 0.25, 0.25)[v]; rtol=1e-12) for v in 1:3) && isapprox(H4.qmd(sum(S5.N[1:3]), sum(S5.B[1:3])), 2.54; rtol=1e-12) &&
       isapprox(S5.H[1], hs_(2.54); rtol=1e-12) && isapprox(S5.A[1], 6.0; rtol=1e-14)
  check("0.5a recruitment, empty site (no seed tree): N1 = 185.2 e^{lnF} W^{ωR} ε0 split 0.5/0.25/0.25, D 2.54, H*(2.54), age 5 + 1", ok, @sprintf("N1 %.6f vs %.6f", sum(S5.N[1:3]), rate))
  Ns = 150.0; Ds = 16.0
  S6 = onesite([(1, 3, 1, Ns, Ds, 15.0, 30.0)]); H4.simulate!(S6, [Pr], Gr; exact=true)
  RD = Ns * (Ds / 25.4)^1.605 / 1500; rate2 = 185.2 * exp(lnF - 3.0 * exp(-0.07 * (4 - 3)) * RD) * 0.9^0.5
  check("0.5b recruitment with a seed cohort: rate = 185.2 e^{lnF − λ0 e^{λ1 (ST−3)} RD} W^{ωR} (RD of the start of the year)", isapprox(sum(S6.N[1:3]), rate2; rtol=1e-12),
        @sprintf("N1 %.6f vs %.6f", sum(S6.N[1:3]), rate2))
end
# 0.6 self-thinning of one cohort at twice sdimax (growth, mortality off): N halves, QMD unchanged
let Dq = 30.0, N = 3000 / (30.0 / 25.4)^1.605, Pz = H4.SpP4(-Inf, -Inf, 30.0, 3.0, 1e12, 18.0)
  S7 = onesite([(1, 6, 1, N, Dq, 20.0, 50.0)]); H4.simulate!(S7, [Pz], Gh; exact=true); i = ci(1, 6, 1)
  check("0.6 self-thinning: one cohort at SDI = 2 sdimax → N/2, QMD unchanged", isapprox(S7.N[i], N / 2; rtol=1e-10) && isapprox(H4.qmd(S7.N[i], S7.B[i]), Dq; rtol=1e-12),
        @sprintf("N %.6f vs %.6f", S7.N[i], N / 2))
end
# 0.7 lane mixing: (100, 0, 0) with κΔ = 0.5 → (75, 12.5, 12.5); Δ = 0.5 → κΔ = 0.25 → (87.5, 6.25, 6.25)
let Pz = H4.SpP4(-Inf, -Inf, 30.0, 3.0, 1e12, 18.0)
  S8 = onesite([(1, 3, 1, 100.0, 15.0, 12.0, 20.0)]); H4.simulate!(S8, [Pz], withG(Gh; κ=0.5); exact=true)
  S9 = onesite([(1, 3, 1, 100.0, 15.0, 12.0, 20.0)]; Δ=0.5); H4.simulate!(S9, [Pz], withG(Gh; κ=0.5); exact=true)
  a = S8.N[ci(1, 3, 1):ci(1, 3, 3)]; b = S9.N[ci(1, 3, 1):ci(1, 3, 3)]
  check("0.7 lane mixing κΔ of a cohort re-split 0.5/0.25/0.25 (heights and QMD carried)", isapprox(a, [75.0, 12.5, 12.5]; rtol=1e-12) && isapprox(b, [87.5, 6.25, 6.25]; rtol=1e-12) &&
        all(isapprox(S8.H[ci(1, 3, v)], 12.0; rtol=1e-12) for v in 1:3) && all(isapprox(H4.qmd(S8.N[ci(1, 3, v)], S8.B[ci(1, 3, v)]), 15.0; rtol=1e-12) for v in 1:3), "κ=0.5: $(a); Δ=0.5: $(b)")
end
# 0.8 overtopping of two cohorts: tall sees ½ its own SDI, short sees all of the tall + ½ its own
let Pz = H4.SpP4(0.0, -Inf, 30.0, 3.0, 1e12, 18.0)
  S10 = onesite([(1, 3, 1, 100.0, 15.31, 18.0, 30.0), (2, 3, 1, 100.0, 15.31, 10.0, 30.0)]; nkk=2)
  X = H4.Scratch(length(S10.N)); H4.step_site!(S10, [Pz, Pz], Gh, X, 1, 1.0, lz, lw, emptytabs(2))
  sd = 100 * (15.31 / 25.4)^1.605
  check("0.8 overtopping: O_tall = ½ SDI_tall / sdimax, O_short = (SDI_tall + ½ SDI_short) / sdimax", isapprox(X.O[ci(1, 3, 1)], 0.5sd / 1500; rtol=1e-12) &&
        isapprox(X.O[ci(2, 3, 1)], 1.5sd / 1500; rtol=1e-12), @sprintf("O tall %.6f (hand %.6f), short %.6f (hand %.6f)", X.O[ci(1, 3, 1)], 0.5sd / 1500, X.O[ci(2, 3, 1)], 1.5sd / 1500))
end
# 0.9 physical: a one-tree cohort at 17.5 cm (top of class 3). Truth: with dD = 0.4 cm it is at 17.9 cm (class 4, all of it); with dD = 0.16 cm at 17.66 (class 3, all)
let N = 14.87, Pq = H4.SpP4(0.0, -Inf, 30.0, 3.0, 1e12, 18.0), res = Float64[]
  S0 = onesite([(1, 3, 1, N, 17.5, 15.0, 30.0)]); X = H4.Scratch(length(S0.N)); H4.step_site!(S0, [Pq], Gh, X, 1, 1.0, lz, lw, emptytabs(1)); d0 = X.dD[ci(1, 3, 1)]
  for target in (0.4, 0.16)
    Pt = H4.SpP4(log(target / d0), -Inf, 30.0, 3.0, 1e12, 18.0)                          # dD ∝ ε (ρ = 0)
    S11 = onesite([(1, 3, 1, N, 17.5, 15.0, 30.0)]); H4.step_site!(S11, [Pt], Gh, X, 1, 1.0, lz, lw, emptytabs(1))
    push!(res, X.dD[ci(1, 3, 1)], S11.N[ci(1, 4, 1)] / N, H4.qmd(S11.N[ci(1, 3, 1)], S11.B[ci(1, 3, 1)]))
  end
  check("0.9 [physical] one tree at 17.5 cm: dD 0.4 → the whole cohort in class 4; dD 0.16 → none moves (exact for a single tree)", isapprox(res[2], 1; atol=1e-9) && res[5] < 1e-9,
        @sprintf("dD %.3f: %.1f %% moved, stayers' QMD %.2f cm (> 17.78); dD %.3f: %.1f %% moved (truth 0)", res[1], 100res[2], res[3], res[4], 100res[5]))
end

# ================================================================ 1. kernel: unit checks
section("1. kernel unit checks")
# 1.1 diameter classes at every 0.1-in diameter (FIA records DIA in tenths of an inch); exact class by integer arithmetic on inches
let bad = Tuple{Float64,Int,Int}[], incon = 0
  for d10 in 10:500
    DIA = d10 / 10; D = DIA * 2.54; jt = min(NCLS, (d10 ÷ 10 + 1) ÷ 2); j = H4.dclass(D)
    j == jt || push!(bad, (DIA, j, jt))
    (j >= 1 && !(H4.CLS_LO[j] <= D < H4.CLS_HI[j])) && (incon += 1)
  end
  check("1.1 dclass of every 0.1-in diameter 1.0–50.0 in = floor((DIA+1)/2) (HBAD4P.jl:26)", isempty(bad) && incon == 0,
        isempty(bad) ? "" : "wrong: " * join(["DIA $(b[1]) in → class $(b[2]) (expected $(b[3]))" for b in bad], ", ") * "; also D=78.74 cm lies at CLS_HI[15] (exclusive) — class bound and dclass disagree ($incon diameters)")
end
# 1.2 lane nodes / weights
for mode in (:sym, :fvs)
  z, w = H4.lanes(mode); m = sum(w .* z); v = sum(w .* z .^ 2) - m^2
  check("1.2 [spec §5] lanes :$mode: weights sum to 1, mean 0, variance 1", abs(sum(w) - 1) < 1e-12 && abs(m) < 1e-4 && abs(v - 1) < 1e-3,
        "Σw $(fmt(sum(w))) mean $(fmt(m)) var $(fmt(v))" * (mode == :fvs ? " (FVS sn/dgdriv.f nodes; σv of :fvs is not the sd of the lane deviation: E[z²] = 0.776)" : ""))
end
check("1.2b [spec §1] option HBAD4P_TRIPLE=fvs is read somewhere (data4/params4/kernel)",
      any(occursin("HBAD4P_TRIPLE", read(f, String)) for f in (joinpath(@__DIR__, "data4.jl"), joinpath(@__DIR__, "params4.jl"), joinpath(@__DIR__, "..", "..", "src", "plugins", "HBAD4", "HBAD4P.jl"))),
      "not implemented: lanemode is a keyword of build4 AND of simulate!, set independently (a :sym-built set can be simulated with :fvs nodes)")
# 1.3 lin_moments vs quadrature
let mx = 0.0
  for _ in 1:2000
    s = 2rand(rng) - 1; x0 = 2rand(rng) - 1; M = H4.lin_moments(s, x0)
    xs = range(x0, 1; length=20001); h = step(xs)
    for nn in 0:2
      fx = [x^nn * 0.5 * (1 + s * x) for x in xs]; q = h * (sum(fx) - (fx[1] + fx[end]) / 2)
      mx = max(mx, abs(q - M[nn+1]))
    end
  end
  check("1.3 lin_moments vs trapezoid quadrature (2000 random s, x0)", mx < 1e-8, "max abs diff $(fmt(mx))")
end
# 1.4 analytic upgrowth vs (a) the reference diameter-space integrals, (b) Monte-Carlo samples of the linear density (representable slopes)
let mxr = 0.0, worst_mc = 0.0
  for trial in 1:400
    j = rand(rng, 1:NCLS-1); L, U = H4.CLS_LO[j], H4.CLS_HI[j]; c = (L + U) / 2; w = U - L
    s = 1.9rand(rng) - 0.95; q2 = c^2 + c * w * s / 3 + w^2 / 12; N = 50 + 500rand(rng); B = N * π / 4e4 * q2; dD = rand(rng) * 2.5
    Nm, Bm, Dm, Ds, B1 = H4.upgrowth(j, N, B, dD); rN, rB, rB1 = Ref4.movers(j, N, B, dD)
    mxr = max(mxr, reld(Nm, rN; floor=1e-9 * N), reld(Bm, rB; floor=1e-9 * B), reld(B1, rB1))
    if trial <= 40      # Monte Carlo: rejection sampling of ½(1 + s x)
      M = 400_000; d = Float64[]; sizehint!(d, M)
      while length(d) < M; x = 2rand(rng) - 1; rand(rng) * (1 + abs(s)) <= 1 + s * x && push!(d, c + x * w / 2); end
      g = d .+ dD; mv = g .>= U
      pm = mean(mv); seN = sqrt(pm * (1 - pm) / M) * N
      bm = N * π / 4e4 * sum(g[mv] .^ 2) / M; b1 = N * π / 4e4 * mean(g .^ 2)
      seB = N * π / 4e4 * std(ifelse.(mv, g .^ 2, 0.0)) / sqrt(M)
      worst_mc = max(worst_mc, abs(Nm - pm * N) / max(seN, 1e-9), abs(Bm - bm) / max(seB, 1e-12), abs(B1 - b1) / (N * π / 4e4 * std(g .^ 2) / sqrt(M)))
    end
  end
  check("1.4a upgrowth (N movers, B movers, B after growth) vs the diameter-space integrals of the reference (400 random cases, |s| < 0.95)", mxr < 1e-9, "max rel diff $(fmt(mxr))")
  check("1.4b upgrowth vs Monte-Carlo samples of the linear density (40 cases × 4e5 trees)", worst_mc < 5.0, "worst deviation $(fmt(worst_mc)) standard errors")
end
# 1.5 the two-moment closure on the REAL v1 cells: share whose QMD the linear density cannot represent (slope clamped to ±1)
let ncl = 0, nc_ = 0, ncl1 = 0, n1 = 0, Nw = 0.0, Ncl = 0.0
  for s in 1:ns, k in 1:nk, j in 1:NCLS-1
    i = cellix(ST, s, k, j, 1); ST.N0[i] > 0 || continue
    N = ST.N0[i] + ST.N0[i+1] + ST.N0[i+2]; B = ST.B0[i] + ST.B0[i+1] + ST.B0[i+2]; f = Ref4.density(j, H4.qmd(N, B)^2)
    nc_ += 1; Nw += N; abs(f.sraw) > 1 && (ncl += 1; Ncl += N)
  end
  check("1.5 [spec §2.3] v1 cells whose QMD the linear density on the class reproduces (|slope| ≤ 1; else moments are NOT matched)", ncl == 0,
        "$(ncl) of $(nc_) train cells ($(round(100ncl / nc_; digits=1)) %, $(round(100Ncl / Nw; digits=1)) % of the trees) need |s| > 1: a linear density only spans QMDs in the central third of the class width " *
        "(class 3: $(round(sqrt(15.24^2 + 5.08^2 / 12 - 15.24 * 5.08 / 3); digits=2))–$(round(sqrt(15.24^2 + 5.08^2 / 12 + 15.24 * 5.08 / 3); digits=2)) cm of [12.70, 17.78))")
end
# 1.6 consequence on one cohort: a single tree at 17.5 cm (class 3, U = 17.78) growing 0.4 cm/yr must leave the class in year 1
let N = 14.87, D = 17.5, B = N * π / 4e4 * D^2
  Nm, Bm, Dm, Ds, B1 = H4.upgrowth(3, N, B, 0.4)
  check("1.6 [spec §2.3] single tree at 17.5 cm, dD 0.4 cm: movers = all trees, stayers' QMD inside the class", isapprox(Nm, N; rtol=1e-6),
        "kernel moves $(round(100Nm / N; digits=1)) % of the trees; stayers' QMD $(round(Ds; digits=2)) cm > U = 17.78 (B conserved, class membership not)")
end
# 1.7 zero growth, zero mortality, no recruitment / mixing / thinning: every cohort unchanged (A += dt)
let Sx = subS(ST, SUB)
  Pz = [withP(p; lnε=-Inf, lnF=-Inf, longevity=1e12) for p in P]
  Gz = withG(G; m0=0.0, ms=0.0, mA=1e4, λ=0.0, ρ=0.0, κ=0.0, sdimax=1e15)
  H4.simulate!(Sx, Pz, Gz)
  dt = Sx.nyr .+ Sx.frac
  bad = 0
  for s in 1:H4.nsites(Sx), i in Sx.off[s]:Sx.off[s+1]-1
    (reld(Sx.N[i], Sx.N0[i]) < 1e-13 && reld(Sx.B[i], Sx.B0[i]) < 1e-13 && reld(Sx.H[i], Sx.H0[i]) < 1e-13 && (Sx.N0[i] > 0 ? isapprox(Sx.A[i], Sx.A0[i] + dt[s]; atol=1e-9) : Sx.A[i] == Sx.A0[i])) || (bad += 1)
  end
  check("1.7 [spec §5] zero growth + zero mortality leave every cohort unchanged (A advances by dt)", bad == 0, "$(bad) cells changed of $(length(Sx.N))")
end
# 1.8 conservation of N (per site × key × lane) and B (= growth only) by upgrowth: mortality, recruitment, mixing, thinning off
let Sx = subS(ST, SUB)
  Pz = [withP(p; lnF=-Inf, longevity=1e12) for p in P]; Gz = withG(G; m0=0.0, ms=0.0, mA=1e4, κ=0.0, sdimax=1e15)
  H4.simulate!(Sx, Pz, Gz); mx = 0.0
  for s in 1:H4.nsites(Sx), k in 1:nk, v in 1:NL
    a = sum(Sx.N[cellix(Sx, s, k, j, v)] for j in 1:NCLS); b = sum(Sx.N0[cellix(Sx, s, k, j, v)] for j in 1:NCLS); mx = max(mx, reld(a, b; floor=1e-9))
  end
  check("1.8a [spec §5] upgrowth conserves N per (site, key, lane) over whole intervals", mx < 1e-12, "max rel diff $(fmt(mx))")
  # B: one class-3 cell, one step, B after = Σ over trees of the density of (D + dD)² (exact for the density) = Ref4 B1, split exactly into movers + stayers
  mxb = 0.0
  for _ in 1:500
    j = rand(rng, 1:NCLS-1); L, U = H4.CLS_LO[j], H4.CLS_HI[j]; N = 100rand(rng) + 1; q = L + (U - L) * rand(rng); dD = 3rand(rng)
    Nm, Bm, Dm, Ds, B1 = H4.upgrowth(j, N, N * π / 4e4 * q^2, dD)
    Ns = N - Nm; Bs = B1 - Bm; mxb = max(mxb, reld(Nm + Ns, N), reld((Nm > 0 ? π / 4e4 * Nm * Dm^2 : 0.0) + (Ns > 0 ? π / 4e4 * Ns * Ds^2 : 0.0), B1))
  end
  check("1.8b upgrowth splits N and B exactly into movers + stayers (500 random cohorts incl. clamped slopes)", mxb < 1e-10, "max rel diff $(fmt(mxb))")
end
# 1.9 lane mixing alone conserves N, B, Σ N·H, Σ N·A per (site, key, class)
let Sx = subS(ST, SUB[1:min(100, length(SUB))])
  Pz = [withP(p; lnε=-Inf, lnF=-Inf, longevity=1e12) for p in P]; Gz = withG(G; m0=0.0, ms=0.0, mA=1e4, λ=0.0, ρ=0.0, κ=0.4, sdimax=1e15)
  # unequal lanes first: perturb the initial lane split
  Sx.N0 .*= [1.0 + 0.5 * (Sx.ln[i] - 2) for i in eachindex(Sx.N0)]; Sx.H0 .+= [Sx.N0[i] > 0 ? Sx.ln[i] : 0.0 for i in eachindex(Sx.H0)]
  H4.simulate!(Sx, Pz, Gz); mx = 0.0
  for s in 1:H4.nsites(Sx), k in 1:nk, j in 1:NCLS
    i = cellix(Sx, s, k, j, 1); r = i:i+2; n0 = sum(Sx.N0[r]); n0 > 0 || continue; dt = Sx.nyr[s] + Sx.frac[s]
    mx = max(mx, reld(sum(Sx.N[r]), n0), reld(sum(Sx.B[r]), sum(Sx.B0[r])), reld(sum(Sx.N[r] .* Sx.H[r]), sum(Sx.N0[r] .* Sx.H0[r])),
             reld(sum(Sx.N[r] .* Sx.A[r]), sum(Sx.N0[r] .* (Sx.A0[r] .+ dt))))
  end
  check("1.9 lane mixing conserves N, B, Σ N·H, Σ N·A per (site, key, class)", mx < 1e-10, "max rel diff $(fmt(mx))")
end
# 1.10 dD ≤ class width (spec: asserted)
let p = withP(P[1]; lnε=6.0), j = 3, N = 100.0, B = 100 * π / 4e4 * 15.0^2
  asserted = try; H4.upgrowth(j, N, B, 7.0); false; catch; true; end
  Nm, Bm, Dm, Ds, B1 = H4.upgrowth(j, N, B, 9.0)
  check("1.10 [spec §2.3] dD > class width (5.08 cm) is asserted", asserted,
        "not asserted: with dD = 9 cm all trees move ONE class (to class 4) with QMD $(round(Dm; digits=2)) cm > CLS_HI[4] = 22.86; ε is bounded only by e^6 (params4.jl:228)")
end

# ================================================================ 2. kernel vs the brute-force reference on real sites
section("2. optimised kernel vs brute-force reference (train subset)")
# 2.1 one step at a time: the reference starts every year from the kernel's state (implementation test); the free-running divergence is reported
#     separately (exact height ties between cohorts flip with the last bit of a height: the overtopping rule is discontinuous)
let Sx = subS(ST, SUB), X = H4.Scratch(Sx.off[2] - Sx.off[1]), T0 = emptytabs(nk), mx = Dict(:N => 0.0, :B => 0.0, :H => 0.0, :A => 0.0), worst = "", nstep = 0
  mxfree = 0.0; nties = 0; ntsites = 0
  for s in 1:H4.nsites(Sx)
    r = Sx.off[s]:Sx.off[s+1]-1
    for i in r; Sx.N[i] = Sx.N0[i]; Sx.B[i] = Sx.B0[i]; Sx.H[i] = Sx.H0[i]; Sx.A[i] = Sx.A0[i]; end
    Nf = Sx.N0[r]; Bf = Sx.B0[r]; Hf = Sx.H0[r]; Af = Sx.A0[r]; tied = false
    for y in 1:Sx.nyr[s] + (Sx.frac[s] > 0)
      Δ = y <= Sx.nyr[s] ? 1.0 : Sx.frac[s]
      N = Sx.N[r]; B = Sx.B[r]; Hh = Sx.H[r]; A = Sx.A[r]
      live = [q for q in eachindex(N) if N[q] > 1e-9]
      for a in live, b in live; (Sx.sp[r[a]] < Sx.sp[r[b]] && Hh[a] == Hh[b]) && (nties += 1; tied = true); end    # exact ties between different keys
      H4.step_site!(Sx, P, G, X, s, Δ, lz, lw, T0)
      Ref4.step!(N, B, Hh, A, Sx.sp[r], Sx.cl[r], Sx.ln[r], nk, P, G, Sx.W[s], Sx.Wn[s], Sx.T[s], Sx.C[s], Sx.planted[s], Δ, lz, lw)
      Ref4.step!(Nf, Bf, Hf, Af, Sx.sp[r], Sx.cl[r], Sx.ln[r], nk, P, G, Sx.W[s], Sx.Wn[s], Sx.T[s], Sx.C[s], Sx.planted[s], Δ, lz, lw)
      nstep += 1
      for (q, i) in enumerate(r)
        fl = 1e-9 * max(1.0, sum(N)); flb = 1e-9 * max(1e-6, sum(B))           # floors: 1e-9 of the site totals (cancellation in near-empty cells)
        for (nm, a, b) in ((:N, Sx.N[i], N[q]), (:B, Sx.B[i], B[q]), (:H, N[q] > 1e-6 ? Sx.H[i] : 0.0, N[q] > 1e-6 ? Hh[q] : 0.0), (:A, N[q] > 1e-6 ? Sx.A[i] : 0.0, N[q] > 1e-6 ? A[q] : 0.0))
          d = abs(a - b) / max(abs(a), abs(b), nm == :N ? fl : nm == :B ? flb : 1e-9)
          d > mx[nm] && (mx[nm] = d; worst = "site $(SUB[s]) year $y cell $(q) $(nm) kernel $(a) ref $(b)")
        end
      end
    end
    tied && (ntsites += 1)
    mxfree = max(mxfree, maximum(abs(Sx.N[i] - Nf[q]) / max(Sx.N[i], Nf[q], 1e-9 * max(1.0, sum(Nf))) for (q, i) in enumerate(r)))
  end
  check("2.1 kernel (exact functions) = reference, one step at a time from the same state ($(nstep) steps of $(length(SUB)) sites): N, B, H, A per cell",
        all(v < 1e-8 for v in values(mx)), "max rel diff " * join(["$k $(fmt(v))" for (k, v) in mx], ", ") * (all(v < 1e-8 for v in values(mx)) ? "" : " | worst $worst"))
  check("2.1b free-running kernel and reference agree over whole intervals (no discontinuity at floating-point level)", mxfree < 1e-6,
        "max rel diff of N $(fmt(mxfree)); exact height ties between cohorts of DIFFERENT keys at step starts: $(nties) in $(ntsites) of $(length(SUB)) sites — O jumps by ½ the tied SDI when a tie breaks by 1 ulp (recruits of keys with equal Hmax get identical h0; HBAD4P.jl:138-144, 231)")
end
# 2.2 overtopping (and RD) vs brute force on a mid-interval state
let Sy = subS(ST, SUB), X = H4.Scratch(Sy.off[2] - Sy.off[1]), T0 = emptytabs(nk), mx = 0.0, nchk = 0
  H4.simulate!(Sy, P, G; exact=true)      # a state with distinct heights
  for s in 1:H4.nsites(Sy)
    lo = Sy.off[s]; hi = Sy.off[s+1] - 1; N = Sy.N[lo:hi]; B = Sy.B[lo:hi]; Hh = Sy.H[lo:hi]
    act = [q for q in eachindex(N) if N[q] > 1e-9]
    SD = [N[q] * (Ref4.qm(N[q], B[q]) / 25.4)^1.605 for q in eachindex(N)]
    Ob_ = Dict(q => (sum((SD[t] for t in act if Hh[t] > Hh[q]); init=0.0) + 0.5 * sum((SD[t] for t in act if Hh[t] == Hh[q]); init=0.0)) / G.sdimax for q in act)
    H4.step_site!(Sy, P, G, X, s, 1.0, lz, lw, T0)
    for q in act; mx = max(mx, abs(X.O[q] - Ob_[q])); nchk += 1; end
  end
  check("2.2 overtopping O of every live cohort vs O(n²) brute force", mx < 1e-12, "$(nchk) cohorts, max abs diff $(fmt(mx))")
end
# 2.3 lookup tables vs exact functions
let p = P[1], T_ = H4.Tabs(P, G), m = Dict{String,Float64}()
  Ds = vcat(2.54 .+ 247.4 .* rand(rng, 20000), [2.54, 12.7, 249.9])
  for D in Ds
    for (nm, a, b) in (("H*", H4.ht(T_, p, G, 1, D), H4.hstar(p, G, D)), ("(D/20)^b", H4.pw_b(T_, G, D), (D / 20)^G.b), ("(D/20)^ζ", H4.pw_z(T_, G, D), (D / 20)^G.ζ),
                       ("(D/25.4)^1.605", H4.sdi_t(T_, 1.0, D), (D / 25.4)^1.605))
      m[nm] = max(get(m, nm, 0.0), reld(a, b))
    end
  end
  for x in rand(rng, 20000); m["x^θ"] = max(get(m, "x^θ", 0.0), abs(H4.hyd_t(T_, G, x) - x^G.θ)); end
  check("2.3a lookup tables vs exact functions (D ∈ [2.54, 250) cm; x ∈ [0, 1])", all(v < 1e-4 for v in values(m)), join(["$k $(fmt(v))" for (k, v) in m], ", ") * " (x^θ absolute)")
  Sa = subS(ST, SUB); Sb = subS(ST, SUB)
  H4.simulate!(Sa, P, G; exact=true); H4.simulate!(Sb, P, G)
  dN = maximum(reld(sum(Sa.N[Sa.off[s]:Sa.off[s+1]-1]), sum(Sb.N[Sb.off[s]:Sb.off[s+1]-1])) for s in 1:H4.nsites(Sa))
  dB = maximum(reld(sum(Sa.B[Sa.off[s]:Sa.off[s+1]-1]), sum(Sb.B[Sb.off[s]:Sb.off[s+1]-1])) for s in 1:H4.nsites(Sa))
  check("2.3b whole intervals with tables vs exact functions: per-site N and B", dN < 1e-4 && dB < 1e-4, "max rel diff N $(fmt(dN)) B $(fmt(dB))")
end
# 2.4 determinism and thread parity (serial loop with one scratch = 1 thread), nll4 threaded vs serial
let Sa = subS(ST, SUB), Sb = subS(ST, SUB), Sc = subS(ST, SUB)
  H4.simulate!(Sa, P, G); H4.simulate!(Sb, P, G)
  X = H4.Scratch(Sc.off[2] - Sc.off[1]); T_ = H4.Tabs(P, G)
  for s in 1:H4.nsites(Sc)
    for i in Sc.off[s]:Sc.off[s+1]-1; Sc.N[i] = Sc.N0[i]; Sc.B[i] = Sc.B0[i]; Sc.H[i] = Sc.H0[i]; Sc.A[i] = Sc.A0[i]; end
    for _ in 1:Sc.nyr[s]; H4.step_site!(Sc, P, G, X, s, 1.0, lz, lw, T_); end
    Sc.frac[s] > 0 && H4.step_site!(Sc, P, G, X, s, Sc.frac[s], lz, lw, T_)
  end
  check("2.4a two simulate! runs are bitwise identical", Sa.N == Sb.N && Sa.B == Sb.B && Sa.H == Sb.H && Sa.A == Sb.A)
  check("2.4b threaded simulate! ($(Threads.nthreads()) threads) = serial site loop with one scratch (1-thread path), bitwise", Sa.N == Sc.N && Sa.B == Sc.B && Sa.H == Sc.H && Sa.A == Sc.A)
end
# 2.5 sanity of a warm-start run on all train sites
H4.simulate!(ST, P, G)
let nout = 0, nlive = 0, nh = 0, nfin = 0, outtrees = 0.0, alltrees = 0.0
  for i in eachindex(ST.N)
    isfinite(ST.N[i]) && isfinite(ST.B[i]) && isfinite(ST.H[i]) || (nfin += 1)
    ST.N[i] > 1e-6 || continue; nlive += 1; alltrees += ST.N[i]
    q = H4.qmd(ST.N[i], ST.B[i]); j = ST.cl[i]
    (H4.CLS_LO[j] - 1e-6 <= q <= H4.CLS_HI[j] + 1e-6) || (nout += 1; outtrees += ST.N[i])
    ST.H[i] >= 1.37 || (nh += 1)
  end
  check("2.5a warm-start run (train): all state finite", nfin == 0, "$(nfin) non-finite")
  check("2.5b warm-start run (train): every cohort's QMD inside its class", nout == 0, "$(nout) of $(nlive) live cells ($(round(100outtrees / alltrees; digits=2)) % of the trees) outside — consequence of 1.5")
  check("2.5c warm-start run (train): every live cohort's height ≥ 1.37 m", nh == 0, "$(nh) of $(nlive)")
end
# 2.7 [physical] annual diameter growth of the warm start at the v1 state and at the upper ε clamp: trees must not cross more than one class per year
let X = H4.Scratch(ST.off[2] - ST.off[1]), Sx = subS(ST, 1:ns), res = Dict{String,Any}()
  for (lab, PP) in (("warm start", P), ("ε = e^6 (upper clamp)", [withP(p; lnε=6.0) for p in P]))
    T_ = H4.Tabs(PP, G); dds = Float64[]
    for s in 1:ns
      lo = Sx.off[s]; hi = Sx.off[s+1] - 1
      for i in lo:hi; Sx.N[i] = Sx.N0[i]; Sx.B[i] = Sx.B0[i]; Sx.H[i] = Sx.H0[i]; Sx.A[i] = Sx.A0[i]; end
      act = [i for i in lo:hi if Sx.N0[i] > 1e-9]
      H4.step_site!(Sx, PP, G, X, s, 1.0, lz, lw, T_)
      for i in act; push!(dds, X.dD[i-lo+1]); end
    end
    res[lab] = (mx=maximum(dds), q999=quantile(dds, 0.999), n=count(>(5.08), dds), m=mean(dds), tot=length(dds))
  end
  a = res["warm start"]; b = res["ε = e^6 (upper clamp)"]
  check("2.7 [physical] no cohort grows more than one class width (5.08 cm) in a year — warm start AND anywhere in the parameter box", a.n == 0 && b.n == 0,
        @sprintf("warm start: mean dD %.3f, 99.9 %% %.3f, max %.3f cm/yr, %d > 5.08 | ε at its clamp e^6: mean %.2f, max %.2f cm/yr, %d of %d cohorts > 5.08 (moved ONE class only)", a.m, a.q999, a.mx, a.n, b.m, b.mx, b.n, b.tot))
end
# 2.8 two-moment closure at the END of the warm-start run (ST holds it, 2.5): cells whose slope is clamped, by class
let cnt = zeros(Int, 3), tot = zeros(Int, 3), low1 = 0
  for i in eachindex(ST.N)
    ST.N[i] > 1e-6 || continue; j = ST.cl[i]; j == NCLS && continue
    gq = j == 1 ? 1 : j == 2 ? 2 : 3; tot[gq] += 1
    f = Ref4.density(j, H4.qmd(ST.N[i], ST.B[i])^2); abs(f.sraw) > 1 && (cnt[gq] += 1)
    (j == 1 && f.sraw < -1) && (low1 += 1)
  end
  info("2.8 slope clamps at v2 (warm start)", "class 1: $(round(100cnt[1] / tot[1]; digits=1)) % (QMD below the representable 4.40 cm: $(low1); recruits enter at 2.54 cm = the class floor), class 2: $(round(100cnt[2] / tot[2]; digits=1)) %, classes 3–15: $(round(100cnt[3] / tot[3]; digits=1)) % of live cells")
end
# 2.9 [physical] seed sources: a key counts as seeding when ANY class ≥ 3 cell holds > 1e-9 trees/ha — fractional movers make that almost always true
let nreal = 0, ndust = 0, ndust_new = 0
  for s in 1:ns, k in 1:nk
    n3 = sum(ST.N[cellix(ST, s, k, j, v)] for j in 3:NCLS for v in 1:NL); any(ST.N[cellix(ST, s, k, j, v)] > 1e-9 for j in 3:NCLS for v in 1:NL) || continue
    s1 = any(ST.N0[cellix(ST, s, k, j, v)] > 0 for j in 3:NCLS for v in 1:NL)
    if n3 < 1.0; ndust += 1; s1 || (ndust_new += 1); else; nreal += 1; end
  end
  check("2.9 [physical] the seed-source switch (ε0 'no conspecific seed tree') is not triggered by sub-tree fractions of movers (< 1 tree/ha in classes ≥ 3)", ndust_new == 0,
        "end of the warm-start run (train): seed sources with ≥ 1 tree/ha $(nreal); with < 1 tree/ha $(ndust), of which $(ndust_new) had no seed tree at v1 (seed status created by upgrowth dust; HBAD4P.jl:225)")
end
# 2.6 upgrowth closure vs a tree-level replay of the real v1 trees (constant dD per year, no mortality): density crossing each class bound
L = load4()
let sidx = Dict(iid => s for (s, iid) in enumerate(DT.iid)), kix = Dict(k => i for (i, k) in enumerate(KEYS)), subset = Set(DT.iid[SUB]), rows = String[], worst = 0.0
  t1 = L.tr[in.(L.tr.iid, Ref(Set(DT.iid))) .& .!in.(L.tr.out, Ref(("missing", "harvest"))), :]
  for δ in (0.25, 0.5), yrs in (5,)
    above_t = zeros(NCLS); above_m = zeros(NCLS); Bt = 0.0; Bmod = 0.0
    cells = Dict{Tuple{Int,Int},Vector{Float64}}()       # (site, key) → N[j], B[j] of the tree list
    for r in eachrow(t1)
      j = H4.dclass(r.dia_cm); j == 0 && continue; s = sidx[r.iid]; k = kix[r.key]
      w = r.w_ha / (r.dia_cm >= 12.7 ? DT.f1[s] : DT.fm1[s]); Dn = r.dia_cm + yrs * δ; jn = H4.dclass(Dn)
      for b in 2:NCLS; jn >= b && (above_t[b] += w); end; Bt += w * π / 4e4 * Dn^2
      v = get!(cells, (s, k), zeros(2NCLS)); v[j] += w; v[NCLS+j] += w * π / 4e4 * r.dia_cm^2
    end
    for (_, v) in cells
      N = v[1:NCLS]; B = v[NCLS+1:end]
      for _ in 1:yrs
        inN = zeros(NCLS); inB = zeros(NCLS)
        for j in 1:NCLS
          N[j] > 0 || continue
          if j < NCLS
            Nm, Bm, _, _, B1 = H4.upgrowth(j, N[j], B[j], δ); N[j] -= Nm; B[j] = B1 - Bm; inN[j+1] += Nm; inB[j+1] += Bm
          else
            B[j] = N[j] * π / 4e4 * (H4.qmd(N[j], B[j]) + δ)^2
          end
        end
        N .+= inN; B .+= inB
      end
      for b in 2:NCLS; above_m[b] += sum(N[b:end]); end; Bmod += sum(B)
    end
    push!(rows, "dD $(δ) cm/yr × $(yrs) yr: basal area after the replay tree-level $(round(Bt; digits=1)), two-moment $(round(Bmod; digits=1)) m²/ha·sites ($(round(100 * (Bmod / Bt - 1); digits=2)) %)")
    for b in (2, 3, 4, 6, 10)
      cross_t = above_t[b] - sum(DT.S.N0[i] for i in eachindex(DT.S.N0) if DT.S.cl[i] >= b)    # trees crossing into ≥ b during the replay
      cross_m = above_m[b] - sum(DT.S.N0[i] for i in eachindex(DT.S.N0) if DT.S.cl[i] >= b)
      e = cross_m / cross_t - 1; worst = max(worst, abs(e))
      push!(rows, "dD $(δ) cm/yr × $(yrs) yr, into class ≥ $b ($(round(H4.CLS_LO[b]; digits=2)) cm): tree-level $(round(cross_t; digits=0)) /ha·sites, two-moment $(round(cross_m; digits=0)) ($(round(100e; digits=1)) %)")
    end
  end
  for r in rows; println("      ", r); end
  check("2.6 [spec §2.3] two-moment transfer vs tree-level replay of the train v1 trees: upgrowth flux across class bounds within 5 %", worst < 0.05, "worst $(round(100worst; digits=1)) % (lines above)")
end

# ================================================================ 3. spec conformance of other process details
section("3. spec conformance (process details)")
let Sx = subS(ST, SUB[1:1]), Pz = [withP(p; lnF=-Inf, longevity=1e12) for p in P], Gz = withG(G; m0=0.0, ms=0.0, mA=1e4, κ=0.0, sdimax=1e15)
  fill!(Sx.N0, 0.0); fill!(Sx.B0, 0.0); fill!(Sx.H0, 0.0); fill!(Sx.A0, 0.0)
  i = cellix(Sx, 1, 1, 3, 1); Sx.N0[i] = 200.0; Sx.B0[i] = 200 * π / 4e4 * 16.8^2; Sx.H0[i] = 14.0; Sx.A0[i] = 30.0
  Sx.nyr[1] = 1; Sx.frac[1] = 0.0
  Pz = [withP(p; lnε=p.lnε + 1.5) for p in Pz]           # enough growth to move part of the cohort in one year
  H4.simulate!(Sx, Pz, Gz; exact=true)
  hs = Sx.H[i]; hm = Sx.H[i+NL]; nm = Sx.N[i+NL]
  check("3.1 [spec §2.4] movers keep their height (receiving and staying parts carry the same N-weighted height)", nm > 0 && isapprox(hs, hm; rtol=1e-9),
        "movers $(round(nm; digits=2))/ha get H $(round(hm; digits=3)) m, stayers $(round(hs; digits=3)) m: the kernel adds the allometric offset H*(D_group) − H*(D_all) and rescales (HBAD4P.jl:175-176)")
end
let Sx = subS(ST, SUB[1:1])
  fill!(Sx.N0, 0.0); fill!(Sx.B0, 0.0); fill!(Sx.H0, 0.0); fill!(Sx.A0, 0.0); Sx.nyr[1] = 1; Sx.frac[1] = 0.0
  Pz = [withP(p; lnF=0.0) for p in P]; H4.simulate!(Sx, Pz, G; exact=true)
  i = cellix(Sx, 1, 1, 1, 1); q = H4.qmd(sum(Sx.N[i:i+2]), sum(Sx.B[i:i+2]))
  check("3.2 [spec §2.7] recruits enter at D0 = 2.54 cm + the in-year growth", q > 2.54 + 1e-9, "recruits' QMD after their first year $(round(q; digits=4)) cm (no in-year growth; HBAD4P.jl:231)")
end
check("3.3 [spec §2.2] temperature response uses the growing-season temperature T_gs and a growing-season T_opt",
      false, "kernel T = tmean_iv (period ANNUAL mean, data4.jl:128); T_opt = mean ANNUAL normal tmean_n (data4.jl:50,62); the climate table has no growing-season temperature column; both are EPA-L4 means (scripts/hbad3/climate_join.py), not plot values")
let p = P[1], Ta = TR.Topt
  info("3.4 T_opt spread", "T_opt per key " * join(["$(k) $(round(t; digits=1))" for (k, t) in zip(KEYS, Ta)], ", ") * " | train period T $(round(minimum(ST.T); digits=1))–$(round(maximum(ST.T); digits=1)) °C, mean $(round(mean(ST.T); digits=1))")
end

# ================================================================ 4. data pipeline vs raw sources (SQL)
section("4. data pipeline vs SQL on the raw databases")
con = DBInterface.connect(DuckDB.DB, ":memory:")
DBInterface.execute(con, "ATTACH '$(SLIM_DB)' AS s (READ_ONLY)"); DBInterface.execute(con, "ATTACH '$(FIADB)' AS f (READ_ONLY)")
q(sql) = DataFrame(DBInterface.execute(con, sql))
# 4.1 natural intervals of the region, from the tree records (visits, codes at v2, harvest fates, dt > 2)
ivsql = q("""
  WITH vis AS (SELECT STATECD st, UNITCD u, COUNTYCD c, PLOT p, MEASDATE md, bool_or(coalesce(TRTCD1, 0) > 0) trt, bool_or(coalesce(DSTRBCD1, 0) > 0) dst,
                      bool_or(coalesce(tree_fate, '') LIKE 'harvest%') harv
               FROM s.curated_trees WHERE STATECD IN (12, 13) GROUP BY ALL),
       pr AS (SELECT *, lead(md) OVER w md2, lead(trt) OVER w trt2, lead(dst) OVER w dst2, lead(harv) OVER w harv2 FROM vis WINDOW w AS (PARTITION BY st, u, c, p ORDER BY md))
  SELECT st, u, c, p, md d1, md2 d2 FROM pr WHERE md2 IS NOT NULL AND NOT (trt2 OR dst2 OR harv2) AND date_diff('day', md, md2) / 365.25 > 2""")
spl = CSV.read(SPLIT_PLOTS, DataFrame); setof = Dict((r.statecd, r.unitcd, r.countycd, r.plot) => String(r.set) for r in eachrow(spl))
ivsql.set = [get(setof, (r.st, r.u, r.c, r.p), "") for r in eachrow(ivsql)]; ivsql = ivsql[ivsql.set .!= "", :]
let a = Set((r.st, r.u, r.c, r.p, Date(r.d1), Date(r.d2), r.set) for r in eachrow(ivsql))
  b = Set((r.st, r.u, r.c, r.p, Date(r.d1), Date(r.d2), r.set) for nm in ("train", "val", "test") for r in eachrow(CS.sets[nm].iv))
  check("4.1 natural GA+FL intervals (plot, d1, d2, set): SQL on curated_trees = the builder's three sets", a == b,
        "SQL $(length(a)), builder $(length(b)), only SQL $(length(setdiff(a, b))), only builder $(length(setdiff(b, a)))")
end
# 4.2 sets: plot-level disjointness, = the product's split file, relation to the role split and the outer fold
let pl = Dict(nm => Set((r.st, r.u, r.c, r.p) for r in eachrow(CS.sets[nm].iv)) for nm in ("train", "val", "test"))
  dis = isempty(intersect(pl["train"], pl["val"])) && isempty(intersect(pl["train"], pl["test"])) && isempty(intersect(pl["val"], pl["test"]))
  check("4.2a train / val / test disjoint at the PLOT level", dis, join(["$k $(length(v)) plots" for (k, v) in pl], ", "))
  agree = all(get(setof, x, "") == nm for nm in keys(pl) for x in pl[nm])
  check("4.2b every site's set = $(basename(SPLIT_PLOTS)) (the source of the product's set column)", agree)
  if isfile(ROLE_SPLIT)
    rs = CSV.read(ROLE_SPLIT, DataFrame); rof = Dict((r.statecd, r.unitcd, r.countycd, r.plot) => String(r.set) for r in eachrow(rs))
    tab = Dict{Tuple{String,String},Int}(); for nm in keys(pl), x in pl[nm]; kk = (get(rof, x, "NA"), nm); tab[kk] = get(tab, kk, 0) + 1; end
    check("4.2c no plot of the role's TEST split is used for train / val, and no train/val plot is test", get(tab, ("test", "train"), 0) + get(tab, ("test", "val"), 0) + get(tab, ("train", "test"), 0) + get(tab, ("val", "test"), 0) == 0,
          "role-split → HBAD4P set: " * join(["$(a)→$(b) $(n)" for ((a, b), n) in sort(collect(tab))], ", "))
    (get(tab, ("train", "val"), 0) > 0) && info("4.2d train-cap surplus", "$(tab[("train", "val")]) natural plots of the role's TRAIN split are in HBAD4P's val (train_cap surplus → val, docs/nested_cv_gafl.md): val is not the role's inner val and is spatially interleaved with train")
    if isfile(FOLDS_CSV)
      fo = CSV.read(FOLDS_CSV, DataFrame); fd = Dict((r.statecd, r.unitcd, r.countycd, r.plot) => r.fold for r in eachrow(fo))
      tf = unique(get(fd, x, -1) for x in pl["test"]); ot = unique(get(fd, x, -1) for nm in ("train", "val") for x in pl[nm])
      check("4.2e test plots = one outer fold, never in train / val", length(tf) == 1 && !(tf[1] in ot), "test folds $(tf), train/val folds $(sort(ot))")
    end
  end
end
# 4.3 species keys: train v1 live trees (age > 0, height > 0) per species ≥ MIN_SP_TREES
DuckDB.register_data_frame(con, DataFrame(st=ivsql.st, u=ivsql.u, c=ivsql.c, p=ivsql.p, d1=Date.(ivsql.d1), d2=Date.(ivsql.d2), set=ivsql.set), "ivs")
v1 = q("""
  SELECT i.st, i.u, i.c, i.p, i.d1, i.d2, i.set, t.SUBP sb, t.TREE tr, t.DIA dia, t.HT ht, t.TPA_UNADJ tpa, t.estimated_age age,
         coalesce(r.SPECIES_SYMBOL, 'UNK') sym, coalesce(r.SFTWD_HRDWD, 'H') grp, t2.STATUSCD s2, t2.tree_fate fate2
  FROM ivs i JOIN s.curated_trees t ON t.STATECD = i.st AND t.UNITCD = i.u AND t.COUNTYCD = i.c AND t.PLOT = i.p AND t.MEASDATE = i.d1
  LEFT JOIN s.REF_SPECIES r ON r.SPCD = t.spcd_resolved
  LEFT JOIN s.curated_trees t2 ON t2.STATECD = t.STATECD AND t2.UNITCD = t.UNITCD AND t2.COUNTYCD = t.COUNTYCD AND t2.PLOT = t.PLOT AND t2.SUBP = t.SUBP AND t2.TREE = t.TREE AND t2.MEASDATE = i.d2
  WHERE t.STATUSCD = 1""")
let c = combine(groupby(v1[(v1.set .== "train") .& (coalesce.(v1.age, 0.0) .> 0) .& (coalesce.(v1.ht, 0.0) .> 0) .& (coalesce.(v1.dia, 0.0) .> 0), :], :sym), nrow => :n)
  big = sort(String.(c.sym[c.n .>= MIN_SP_TREES]))
  sqlkeys = sort(unique(vcat(big, ["_" * String(g) for g in unique(v1.grp)])))
  check("4.3a species keys from SQL (train trees, ≥ $(MIN_SP_TREES)) = the builder's keys", sqlkeys == KEYS, "SQL $(join(sqlkeys, ",")) | builder $(join(KEYS, ","))")
  k3 = String.(CSV.read(joinpath(FIT3P, "species.csv"), DataFrame).key)
  check("4.3b keys = the HBAD3P fit's keys ($(basename(FIT3P)))", k3 == KEYS)
end
keyof_(s, g) = s in Set(k for k in KEYS if k[1] != '_') ? String(s) : "_" * String(g)
v1.key = keyof_.(v1.sym, v1.grp)
dupv1 = nrow(v1) - nrow(unique(v1[:, [:st, :u, :c, :p, :d1, :sb, :tr]]))
# 4.4 forested fractions from FIADB (independent query): (plot, date, subplot) → (Σ SUBPCOND_PROP, Σ MICRCOND_PROP) over forest conditions
fsql = q("""
  SELECT p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, make_date(p.MEASYEAR::INT, p.MEASMON::INT, p.MEASDAY::INT) md, sc.SUBP sb,
         sum(sc.SUBPCOND_PROP) FILTER (WHERE k.COND_STATUS_CD = 1) fs, sum(sc.MICRCOND_PROP) FILTER (WHERE k.COND_STATUS_CD = 1) fm,
         sum(sc.SUBPCOND_PROP) ts, sum(sc.MICRCOND_PROP) tm, count(DISTINCT p.CN) ncn
  FROM f.SUBP_COND sc JOIN f.PLOT p ON p.CN = sc.PLT_CN JOIN f.COND k ON k.PLT_CN = sc.PLT_CN AND k.CONDID = sc.CONDID
  WHERE p.STATECD IN (12, 13) AND p.MEASYEAR IS NOT NULL AND p.MEASMON IS NOT NULL AND p.MEASDAY IS NOT NULL GROUP BY ALL""")
FQ = Dict((r.st, r.u, r.c, r.p, Date(r.md), r.sb) => (r.ncn == 1 && abs(coalesce(r.ts, 0.0) - 1) <= 1e-6 ? clamp(coalesce(r.fs, 0.0), 0, 1) : NaN,
                                                      r.ncn == 1 && abs(coalesce(r.tm, 0.0) - 1) <= 1e-6 ? clamp(coalesce(r.fm, 0.0), 0, 1) : NaN) for r in eachrow(fsql))
pf(v) = (x = [isnan(a) ? 1.0 : a for a in v]; m = mean(x); m <= 0 ? 1.0 : max(m, 0.25))
FF = Dict{String,Any}()
let mx = 0.0, nunk = 0, nall = 0, nzero = 0, nmin = 0
  for nm in ("train", "val", "test")
    Dn = CS.sets[nm]; f = zeros(length(Dn.iid), 4)
    for (s, r) in enumerate(eachrow(Dn.iv))
      a1 = [get(FQ, (r.st, r.u, r.c, r.p, Date(r.d1), sb), (NaN, NaN)) for sb in 1:4]; a2 = [get(FQ, (r.st, r.u, r.c, r.p, Date(r.d2), sb), (NaN, NaN)) for sb in 1:4]
      nunk += count(isnan, first.(a1)) + count(isnan, first.(a2)); nall += 8
      m2 = mean(x -> isnan(x) ? 1.0 : x, first.(a2)); m2 <= 0 && (nzero += 1); 0 < m2 < 0.25 && (nmin += 1)
      f[s, :] = [pf(first.(a1)), pf(last.(a1)), pf(first.(a2)), pf(last.(a2))]
      mx = max(mx, maximum(abs.(f[s, :] .- [Dn.f1[s], Dn.fm1[s], Dn.fs2[s], Dn.fm2[s]])))
    end
    FF[nm] = f
  end
  check("4.4a forested fractions f1, fm1, fs2, fm2 of every site: FIADB SUBP_COND × COND (independent query) = the builder", mx < 1e-9, "max abs diff $(fmt(mx))")
  check("4.4b every (interval, visit, subplot) has a known forested fraction (no silent default to 1)", nunk == 0,
        "$(nunk) of $(nall) subplot-visits unknown → treated as fully forested (plotf, data4.jl:25); v2 plot fraction 0 (→ 1.0): $(nzero); clamped up to FMIN = 0.25: $(nmin) sites")
end
# 4.5 v1 stand table: SQL tree list → (site, key, class): N = Σ w / f, B, H (N-weighted over heights), A (over ages)
let mx = Dict(:N => 0.0, :B => 0.0, :H => 0.0, :A => 0.0), nbad = 0, ncell = 0, worst = "", nmiss = 0, nharv = 0
  for nm in ("train", "val", "test")
    Dn = CS.sets[nm]; S = Dn.S; sid = Dict((r.st, r.u, r.c, r.p, Date(r.d1)) => s for (s, r) in enumerate(eachrow(Dn.iv))); kix = Dict(k => i for (i, k) in enumerate(KEYS))
    acc = Dict{Tuple{Int,Int,Int},Vector{Float64}}()
    for r in eachrow(v1[v1.set .== nm, :])
      ismissing(r.s2) && (nmiss += 1; continue); startswith(coalesce(r.fate2, ""), "harvest") && (nharv += 1; continue)
      DIA = coalesce(r.dia, 0.0); DIA >= 1.0 || continue
      j = min(NCLS, floor(Int, (DIA + 1) / 2)); s = sid[(r.st, r.u, r.c, r.p, Date(r.d1))]; k = kix[r.key]
      w = r.tpa * 2.47105 / (DIA >= 5.0 ? FF[nm][s, 1] : FF[nm][s, 2])
      a = get!(acc, (s, k, j), zeros(6)); a[1] += w; a[2] += w * π / 4e4 * (DIA * 2.54)^2
      h = coalesce(r.ht, 0.0); h > 0 && (a[3] += w * h * 0.3048; a[4] += w); ag = coalesce(r.age, 0.0); ag > 0 && (a[5] += w * ag; a[6] += w)
    end
    seen = Set{Tuple{Int,Int,Int}}()
    for s in 1:H4.nsites(S), k in 1:nk, j in 1:NCLS
      i = cellix(S, s, k, j, 1); N = S.N0[i] + S.N0[i+1] + S.N0[i+2]; a = get(acc, (s, k, j), nothing)
      (N == 0 && a === nothing) && continue; ncell += 1
      if a === nothing || N == 0
        nbad += 1; worst = "$(nm) site $(s) $(KEYS[k]) class $(j): builder N $(N), SQL $(a === nothing ? 0 : a[1])"; continue
      end
      B = S.B0[i] + S.B0[i+1] + S.B0[i+2]
      d = (N=reld(N, a[1]), B=reld(B, a[2]), H=a[4] > 0 ? reld(S.H0[i], a[3] / a[4]) : 0.0, A=a[6] > 0 ? reld(S.A0[i], a[5] / a[6]) : 0.0)
      for (kk, v) in pairs(d); mx[kk] = max(mx[kk], v); end
      maximum(values(d)) > 1e-9 && (nbad += 1; worst = "$(nm) site $(s) $(KEYS[k]) class $(j): builder N $(round(N; digits=3)) B $(round(B; digits=5)) | SQL N $(round(a[1]; digits=3)) B $(round(a[2]; digits=5))")
    end
  end
  check("4.5 v1 stand table (all sets): SQL tree list (live at v1, not missing / harvested at v2; TPA·2.47105 / forested fraction of its tally) = the builder, per (site, key, class)",
        nbad == 0, "$(ncell) cells, $(nbad) differ; max rel diff " * join(["$k $(fmt(v))" for (k, v) in mx], ", ") * (nbad > 0 ? " | e.g. $worst" : "") * " | removed: $(nmiss) missing at v2, $(nharv) harvested; duplicate v1 records $(dupv1) | v1 trees at exactly 31.0 in (class bug 1.1): $(count(==(31.0), coalesce.(v1.dia, 0.0)))")
end
# 4.6 v2 stand table = ALL live trees of the plot at v2 (FIA STATUSCD 1), classes by the v2 DIA (subplot trees shrunk below 5 in stay in class 3)
v2 = q("""
  SELECT i.st, i.u, i.c, i.p, i.d1, i.set, t.SUBP sb, t.TREE tr, t.DIA dia, t.HT ht, t.TPA_UNADJ tpa, coalesce(r.SPECIES_SYMBOL, 'UNK') sym, coalesce(r.SFTWD_HRDWD, 'H') grp,
         t1.STATUSCD s1, t1.DIA dia1, coalesce(r1.SPECIES_SYMBOL, 'UNK') sym1, coalesce(r1.SFTWD_HRDWD, 'H') grp1
  FROM ivs i JOIN s.curated_trees t ON t.STATECD = i.st AND t.UNITCD = i.u AND t.COUNTYCD = i.c AND t.PLOT = i.p AND t.MEASDATE = i.d2
  LEFT JOIN s.REF_SPECIES r ON r.SPCD = t.spcd_resolved
  LEFT JOIN s.curated_trees t1 ON t1.STATECD = t.STATECD AND t1.UNITCD = t.UNITCD AND t1.COUNTYCD = t.COUNTYCD AND t1.PLOT = t.PLOT AND t1.SUBP = t.SUBP AND t1.TREE = t.TREE AND t1.MEASDATE = i.d1
  LEFT JOIN s.REF_SPECIES r1 ON r1.SPCD = t1.spcd_resolved
  WHERE t.STATUSCD = 1""")
v2.key = [ismissing(a) || a != 1 ? keyof_(s, g) : keyof_(s1, g1) for (a, s, g, s1, g1) in zip(v2.s1, v2.sym, v2.grp, v2.sym1, v2.grp1)]   # survivors: the v1 species (as the builder)
let nres = 0, ntally = 0, ntally_ex = String[], ncell = 0, nbad = 0, nmiss_tr = 0, extra_tr = 0, mxD = 0.0, mxH = 0.0, worst = "", nkeych = 0
  for nm in ("train", "val", "test")
    Dn = CS.sets[nm]; sid = Dict((r.st, r.u, r.c, r.p, Date(r.d1)) => s for (s, r) in enumerate(eachrow(Dn.iv))); kix = Dict(k => i for (i, k) in enumerate(KEYS))
    n = zeros(Int, nk, NCLS, length(Dn.iid)); d2 = zeros(nk, NCLS, length(Dn.iid)); hs = zeros(nk, NCLS, length(Dn.iid)); nh = zeros(Int, nk, NCLS, length(Dn.iid))
    for r in eachrow(v2[v2.set .== nm, :])
      DIA = coalesce(r.dia, 0.0); DIA >= 1.0 || continue
      j = min(NCLS, floor(Int, (DIA + 1) / 2)); surv = !ismissing(r.s1) && r.s1 == 1
      (surv && coalesce(r.dia1, 0.0) >= 5.0 && DIA < 5.0) && (j = 3; DIA = 5.0)
      (!ismissing(r.s1) && r.s1 != 1) && (nres += 1)
      surv && keyof_(r.sym, r.grp) != r.key && (nkeych += 1)
      tp = coalesce(r.tpa, NaN); tpa_ok = j >= 3 ? abs(tp - 6.018046) < 1e-3 : abs(tp - 74.965282) < 1e-3
      tpa_ok || (ntally += 1; length(ntally_ex) < 3 && push!(ntally_ex, "DIA $(r.dia) in (v1 $(coalesce(r.dia1, NaN))) class $j TPA $(r.tpa)"))
      s = sid[(r.st, r.u, r.c, r.p, Date(r.d1))]; k = kix[r.key]
      n[k, j, s] += 1; d2[k, j, s] += (DIA * 2.54)^2; h = coalesce(r.ht, 0.0); h > 0 && (hs[k, j, s] += h * 0.3048; nh[k, j, s] += 1)
    end
    for s in 1:length(Dn.iid), k in 1:nk, j in 1:NCLS
      (n[k, j, s] == 0 && Dn.n[k, j, s] == 0) && continue; ncell += 1
      if n[k, j, s] != Dn.n[k, j, s] || nh[k, j, s] != Dn.nH[k, j, s]
        nbad += 1; nmiss_tr += max(n[k, j, s] - Dn.n[k, j, s], 0); extra_tr += max(Dn.n[k, j, s] - n[k, j, s], 0)
        worst = "$(nm) site $(s) $(KEYS[k]) class $(j): builder n $(Dn.n[k, j, s]) SQL n $(n[k, j, s])"
      else
        mxD = max(mxD, reld(d2[k, j, s], Dn.D2[k, j, s])); mxH = max(mxH, reld(hs[k, j, s], Dn.Hs[k, j, s]))
      end
    end
  end
  check("4.6a v2 stand table (all sets) = all live trees of the plot at v2 in curated_trees, per (site, key, class): count n, n with height, Σ D², Σ H",
        nbad == 0 && mxD < 1e-9 && mxH < 1e-9,
        "$(ncell) cells, $(nbad) differ ($(nmiss_tr) live v2 trees absent from the builder, $(extra_tr) extra) | equal cells: max rel diff ΣD² $(fmt(mxD)) ΣH $(fmt(mxH)) | live at v2 but recorded DEAD at v1: $(nres) (neither survivor nor ingrowth) | species key changed v1→v2 for $(nkeych) survivors | v2 trees at exactly 31.0 in (class bug 1.1): $(count(==(31.0), coalesce.(v2.dia, 0.0)))" * (nbad > 0 ? " | e.g. $worst" : ""))
  check("4.6b FIA design: every live v2 tree's tally (TPA_UNADJ 6.018 subplot / 74.965 microplot) = the tally area its class is scored on (μ: classes ≥ 3 subplot)",
        ntally == 0, "$(ntally) trees on a different tally than their scored class" * (isempty(ntally_ex) ? "" : ": " * join(ntally_ex, "; ") * " — subplot trees that shrank below 5 in are MICROPLOT saplings at v2 but scored as class 3 on the subplot area (data4.jl:106-111)"))
end
# 4.7 tally areas vs the FIA design expansion factors
let ws = 6.018046 * 2.47105, wm = 74.965282 * 2.47105
  e1 = ws * AREA_SUB - 1; e2 = wm * AREA_MIC - 1
  check("4.7 tally areas (data4.jl:20-21) = the inverse of the FIA expansion factors (1 tree ↔ TPA_UNADJ): w_ha × area = 1", abs(e1) < 1e-3 && abs(e2) < 1e-3,
        "subplot $(round(100 * e1; digits=3)) %, microplot $(round(100 * e2; digits=3)) % (FIA subplot r = 24 ft is 1/24.07 ac, not 1/24 ac: μ of every subplot class is 0.3 % high)")
end
# 4.8 climate joins (iid of the R4 product ↔ plot of the climate table), missing values
let cl = CSV.read(CLIM_CSV, DataFrame; select=[:iid, :st, :u, :c, :p, :tmean_iv]), wa = CSV.read(WATER_CSV, DataFrame)
  cp = Dict(r.iid => (r.st, r.u, r.c, r.p) for r in eachrow(cl)); ct = Dict(r.iid => r.tmean_iv for r in eachrow(cl))
  wd = Dict(r.iid => (r.aetpet_iv, r.aetpet_n) for r in eachrow(wa))
  nbad = 0; nmis = 0; mx = 0.0
  for nm in ("train", "val", "test")
    Dn = CS.sets[nm]
    for (s, r) in enumerate(eachrow(Dn.iv))
      get(cp, r.iid, nothing) == (r.st, r.u, r.c, r.p) || (nbad += 1)
      (haskey(wd, r.iid) && haskey(ct, r.iid) && !ismissing(wd[r.iid][1]) && !ismissing(ct[r.iid])) || (nmis += 1; continue)
      mx = max(mx, abs(Dn.S.W[s] - wd[r.iid][1]), abs(Dn.S.Wn[s] - wd[r.iid][2]), abs(Dn.S.T[s] - ct[r.iid]))
    end
  end
  check("4.8 climate: every site's iid maps to its own plot in the climate table; W, Wn, T = the tables; none missing", nbad == 0 && nmis == 0 && mx < 1e-12,
        "plot mismatches $(nbad), missing $(nmis), max abs diff $(fmt(mx))")
end
# 4.9 CO2 period mean (calendar years year(d1) … year(d2) − 1, the climate period) and a date-weighted alternative
let co = CSV.read(CO2_CSV, DataFrame), cd = Dict(Int(r.year) => Float64(r.co2_ppm) for r in eachrow(co)), mx = 0.0, mxw = 0.0
  for nm in ("train", "val", "test"), (s, r) in enumerate(eachrow(CS.sets[nm].iv))
    y1 = year(r.d1); y2 = year(r.d2); c = mean(cd[y] for y in y1:max(y2 - 1, y1)); mx = max(mx, abs(CS.sets[nm].S.C[s] - c))
    ts = range(Dates.value(r.d1), Dates.value(r.d2); length=400); cw = mean(cd[year(Date(Dates.UTD(round(Int, t))))] for t in ts)
    mxw = max(mxw, abs(c - cw))
  end
  check("4.9 CO2 period mean = mean of the annual series over year(d1) … year(d2)−1 (same period as the climate)", mx < 1e-9,
        "max abs diff $(fmt(mx)) ppm | vs the date-weighted mean over [d1, d2]: max $(round(mxw; digits=2)) ppm (f_CO2 ±$(round(100 * 0.58 * mxw / 390; digits=2)) %)")
end
# 4.10 traits from the SQL tree list (train) + the trait tables; T_opt from the climate table
let t = v1[(v1.set .== "train"), :], st_ = CSV.read(SHADETOL_CSV, DataFrame), lo = CSV.read(LONGEVITY_CSV, DataFrame)
  std_ = Dict(String(r.sym) => Float64(r.shade_class) for r in eachrow(st_)); lod = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lo))
  mx = Dict(:Hmax => 0.0, :ST => 0.0, :longevity => 0.0)
  for (k, key) in enumerate(KEYS)
    tk = t[(t.key .== key) .& (coalesce.(t.dia, 0.0) .> 0), :]; hs = 0.3048 .* Float64.(collect(skipmissing(tk.ht[coalesce.(tk.ht, 0.0) .> 0])))
    c = combine(groupby(tk, :sym), nrow => :n)
    mx[:Hmax] = max(mx[:Hmax], reld(quantile(hs, 0.995), TR.Hmax[k])); mx[:ST] = max(mx[:ST], reld(sum(get(std_, s, 3.0) * n for (s, n) in zip(c.sym, c.n)) / sum(c.n), TR.ST[k]))
    mx[:longevity] = max(mx[:longevity], reld(median([get(lod, s, 300.0) for s in c.sym]), TR.longevity[k]))
  end
  check("4.10a traits Hmax (99.5 % train height), ST (tree-weighted), LONGEVITY (median over member species) from the SQL tree list = TR", all(v < 1e-9 for v in values(mx)),
        join(["$k $(fmt(v))" for (k, v) in mx], ", "))
  ivall = DataFrame(Arrow.Table(joinpath(DATA_DIR, "intervals.arrow"))); outside = Set(ivall.iid[.!inregion.(ivall.st)])
  tall = DataFrame(Arrow.Table(joinpath(DATA_DIR, "interval_trees.arrow"))); tall = tall[in.(tall.iid, Ref(Set(ivall.iid[.!inregion.(ivall.st) .| (coalesce.(ivall.set, "") .== "train")]))), :]
  tall.key = keyof_.(tall.sym, tall.grp)
  fr = ["$(k) $(round(100 * mean(in.(tall.iid[tall.key .== k], Ref(outside))); digits=0)) %" for k in KEYS]
  info("4.10b T_opt sources", "share of the trees behind T_opt from OUTSIDE $(REGION) (other R4 states; every interval incl. harvested/disturbed): " * join(fr, ", ") *
       " — for _H / _S these are mostly species that are not even members of the GA+FL group")
  medage_used = 0
  for nm in ("train", "val", "test"); medage_used += count(r -> r.out != "missing" && r.out != "harvest" && !(r.age > 0), eachrow(L.tr[in.(L.tr.iid, Ref(Set(CS.sets[nm].iid))), :])); end
  check("4.10c no val / test information in anything derived for training (traits, keys, default ages)", medage_used == 0,
        "default age of an un-aged cell = the key's median age over ALL sets incl. val / test (data4.jl:80); v1 trees without an age: $(medage_used)")
end
# 4.11 cache key covers everything the cached sets depend on
let src = read(joinpath(@__DIR__, "data4.jl"), String)
  m = match(r"f = joinpath\(cd_, \"sets_(.*)\.jls\"\)", src)
  deps = ["HBAD4P.jl" => occursin("HBAD4P.jl", m === nothing ? "" : m.captures[1]), "MIN_SP_TREES" => occursin("MIN_SP_TREES", src[findfirst("function cached_sets", src)[1]:end][1:600]),
          "input CSV contents / paths" => false, "product file contents" => false]
  check("4.11 cache key (data4.jl:177-179) covers the kernel source used by build4 (dclass, lanes), HBAD2_MIN_SP_TREES, the climate / forest / CO2 / trait CSVs and the product contents",
        all(last.(deps)), "key = hash(product PATH) + set names + lanemode + region + hash(data4.jl); missing: " * join(first.(deps[.!last.(deps)]), ", "))
end

# ================================================================ 5. likelihood
section("5. likelihood")
let mx = 0.0
  for r in (0.05, 0.3, 1.0, 2.0, 7.5, 50.0, 100.0), μ in (1e-6, 0.01, 0.3, 1.0, 4.0, 25.0, 300.0), n in (0, 1, 2, 5, 17, 60, 400)
    a = nb_logpmf(Float64(n), μ, r); b = Distributions.logpdf(Distributions.NegativeBinomial(r, r / (r + μ)), n); mx = max(mx, abs(a - b) / max(1.0, abs(b)))
  end
  check("5.1 nb_logpmf = Distributions.NegativeBinomial(r, p = r/(r+μ)) (mean μ, variance μ + μ²/r)", mx < 1e-8, "max rel diff $(fmt(mx))")
end
# 5.2 nll4 recomputed from its definition (Distributions), warm-start simulation of the train set (ST was simulated in 2.5)
function nll_ref(D, Ob)
  S = D.S; c = 0.0; dsz = 0.0; hh = 0.0; cfl = 0.0; nfl = 0
  for s in 1:H4.nsites(S), k in 1:S.nk, j in 1:NCLS
    r = cellix(S, s, k, j, 1):cellix(S, s, k, j, 3); Np = sum(S.N[r]); n = D.n[k, j, s]
    area = j >= 3 ? 4 * ACRE / 24 / 1e4 : 4 * ACRE / 300 / 1e4; f = j >= 3 ? D.fs2[s] : D.fm2[s]; μ = max(Np * f * area, 1e-12); rr = j >= 3 ? Ob.rS : Ob.rM
    l = -Distributions.logpdf(Distributions.NegativeBinomial(rr, rr / (rr + μ)), n); c += l
    (μ < 1e-6 && n > 0) && (cfl += l; nfl += n)
    if n > 0 && Np > 1e-12
      dq = sqrt(sum(S.B[r]) / (π / 4e4 * Np)); qo = sqrt(D.D2[k, j, s] / n)
      dsz -= Distributions.logpdf(Distributions.Normal(log(dq), Ob.σD / sqrt(n)), log(qo))
      if D.nH[k, j, s] > 0
        Hp = sum(S.N[r] .* S.H[r]) / Np
        hh -= Distributions.logpdf(Distributions.Normal(log(Hp), Ob.σH / sqrt(D.nH[k, j, s])), log(D.Hs[k, j, s] / D.nH[k, j, s]))
      end
    end
  end
  (total=c + dsz + hh, count=c, size=dsz, height=hh, floor_count=cfl, floor_trees=nfl)
end
let a = nll4(DT, Ob), b = nll_ref(DT, Ob)
  check("5.2 nll4 (threaded) = independent recomputation (Distributions NB + Normal) on all $(ns) train sites after a warm-start run",
        reld(a.total, b.total) < 1e-6 && reld(a.count, b.count) < 1e-6 && reld(a.size, b.size) < 1e-9 && reld(a.height, b.height) < 1e-9,
        "nll4 $(round(a.total; digits=3)) (count $(round(a.count; digits=1)) size $(round(a.size; digits=1)) height $(round(a.height; digits=1))) | ref $(round(b.total; digits=3)) (rel $(fmt(reld(a.total, b.total))); NB log-gamma rounding 5e-10 per cell, see 5.1)")
  null_state!(DT); bn = nll_ref(DT, Ob); an = nll4(DT, Ob)
  check("5.3 persistence null: nll4 = recomputation", reld(an.total, bn.total) < 1e-5, "null $(round(an.total; digits=1)) (rel $(fmt(reld(an.total, bn.total))))")
  check("5.4 [stat] the count NLL is not dominated by the μ floor 1e-12 (cells with μ < 1e-6 holding observed trees)",
        bn.floor_count < 0.05 * bn.count && b.floor_count < 0.05 * b.count,
        "persistence null: $(round(100bn.floor_count / bn.count; digits=1)) % of its count NLL from $(bn.floor_trees) trees in μ≈0 cells (≈ 27.6 nats per tree, set by the arbitrary floor) | warm-start model: $(round(100b.floor_count / b.count; digits=1)) % from $(b.floor_trees) trees")
  H4.simulate!(ST, P, G)
end
# 5.5 ingrowth into the subplot classes that the v1 microplot tally could not see
let ing = L.ig[in.(L.ig.iid, Ref(Set(DT.iid))), :], sidx = Dict(iid => s for (s, iid) in enumerate(DT.iid)), kix = Dict(k => i for (i, k) in enumerate(KEYS)), n3 = 0, n3z = 0
  for r in eachrow(ing)
    r.dia_cm >= 12.7 || continue; s = sidx[r.iid]; k = kix[r.key]; n3 += 1
    sum(DT.S.N0[cellix(DT.S, s, k, j, v)] for j in 1:2 for v in 1:NL) == 0 && (n3z += 1)
  end
  info("5.5 subplot ingrowth without a v1 source", "$(n3) train ingrowth trees ≥ 12.7 cm at v2; $(n3z) ($(round(100n3z / n3; digits=0)) %) of a key with NO sapling in the v1 microplots (4 × 1/300 ac = 8 % of the subplot area): the model must create them by upgrowth from a zero class-2 density")
end

# ================================================================ 6. warm start and parameter space
section("6. warm start / parameter space")
let gl = CSV.read(joinpath(FIT3P, "globals.csv"), DataFrame), gd = Dict(Symbol(r.name) => r.value for r in eachrow(gl))
  check("6.1 fixed globals as documented (gR 0, mA 10, sdimax 1500, βC $(FIX_DEFAULT[:βC]))", G.gR == 0 && isapprox(G.mA, 10) && isapprox(G.sdimax, 1500) && G.βC == FIX_DEFAULT[:βC])
  # growth of the warm start vs HBAD3P's at the v1 state: HBAD4P drops HBAD3P's fitted crowding gain e^{gR RD} and crown term (CR/0.4)^φ
  RD1 = Float64[]; for s in 1:ns; r = ST.off[s]:ST.off[s+1]-1; push!(RD1, sum(H4.sdi(ST.N0[i], H4.qmd(ST.N0[i], ST.B0[i])) for i in r if ST.N0[i] > 0; init=0.0) / 1500); end
  t1 = L.tr[in.(L.tr.iid, Ref(Set(DT.iid))), :]; t1 = t1[t1.cr .> 0, :]; crm = sum(t1.w_ha .* (t1.cr ./ 100)) / sum(t1.w_ha)
  mult = exp(get(gd, :gR, 0.0) * mean(RD1)) * (max(crm, 0.05) / 0.4)^get(gd, :φ, 0.0)
  check("6.2 the warm start keeps HBAD3P's fitted growth level (HBAD3P terms absent from HBAD4P are neutral in the fit)", abs(mult - 1) < 0.05,
        "HBAD3P fit has gR = $(round(get(gd, :gR, 0.0); digits=3)) (HBAD4P fixes 0) and crown φ = $(round(get(gd, :φ, 0.0); digits=3)) (absent): at the train v1 state (mean plot RD $(round(mean(RD1); digits=3)), mean CR $(round(crm; digits=2))) HBAD3P's growth multiplier ≈ $(round(mult; digits=3)) is lost; ε was not rescaled")
end
let ok = true, msg = String[]
  for (j, sp) in enumerate(GSPEC4)
    nm = sp[1]; haskey(SP.fix, nm) && continue
    zj = findfirst(==(String(nm)), SP.names); zj === nothing && (ok = false; push!(msg, "$nm not a coordinate"); continue)
    z = copy(SP.z0); z[zj] = 1e3; v = getfield(SP.decode(z)[nm in OBS4 ? 3 : 2], nm); (v <= sp[4] + 1e-12) || (ok = false; push!(msg, "$nm upper"))
    z[zj] = -1e3; v = getfield(SP.decode(z)[nm in OBS4 ? 3 : 2], nm); (v >= sp[3] - 1e-12) || (ok = false; push!(msg, "$nm lower"))
  end
  check("6.3 decode clamps every free global to its bounds", ok, join(msg, ", "))
end

# ================================================================ 7. stand-level quantities the model is calibrated / evaluated on (warm start, train)
section("7. stand-level quantities at v2 (warm start vs observed, train; per forested ha, plot means)")
H4.simulate!(ST, P, G)
function standv(S, D, which)    # which = :sim (S.N/B/H) | :obs (tally) | :v1 (S.N0/B0/H0)
  ns_ = H4.nsites(S); N = zeros(ns_); N12 = zeros(ns_); BA = zeros(ns_); D2 = zeros(ns_); LH = zeros(ns_); BA12 = zeros(ns_)
  for s in 1:ns_, k in 1:S.nk, j in 1:NCLS
    r = cellix(S, s, k, j, 1):cellix(S, s, k, j, 3)
    if which == :obs
      a = (j >= 3 ? D.fs2[s] : D.fm2[s]) * area_class(j); n = D.n[k, j, s] / a; b = D.D2[k, j, s] * π / 4e4 / a
      h = D.nH[k, j, s] > 0 ? D.Hs[k, j, s] / D.nH[k, j, s] : 0.0
    else
      Nv = which == :sim ? S.N[r] : S.N0[r]; Bv = which == :sim ? S.B[r] : S.B0[r]; Hv = which == :sim ? S.H[r] : S.H0[r]
      n = sum(Nv); b = sum(Bv); h = n > 0 ? sum(Nv .* Hv) / n : 0.0
    end
    N[s] += n; BA[s] += b
    if j >= 3; N12[s] += n; D2[s] += b; BA12[s] += b; LH[s] += b * h; end
  end
  (N=mean(N), N12=mean(N12), BA=mean(BA), QMD12=sqrt(sum(D2) / sum(N12) * 4e4 / π), LH=sum(LH) / sum(BA12))
end
let o = standv(ST, DT, :obs), m = standv(ST, DT, :sim), z = standv(ST, DT, :v1)
  info("7.1 plot means at v2", @sprintf("N obs %.0f sim %.0f (%+.1f %%) [v1 %.0f] | N≥12.7 obs %.1f sim %.1f (%+.1f %%) | BA obs %.2f sim %.2f (%+.1f %%) [v1 %.2f] | QMD≥12.7 obs %.2f sim %.2f | Lorey H≥12.7 obs %.2f sim %.2f",
       o.N, m.N, 100 * (m.N / o.N - 1), z.N, o.N12, m.N12, 100 * (m.N12 / o.N12 - 1), o.BA, m.BA, 100 * (m.BA / o.BA - 1), z.BA, o.QMD12, m.QMD12, o.LH, m.LH))
  # ingrowth into ≥ 12.7 cm: observed = survivors crossing 12.7 + new trees ≥ 12.7 (per forested ha of the v2 subplot tally); model = N≥12.7 of the full run − of a run without classes 1–2 and recruitment
  sidx = Dict(iid => s for (s, iid) in enumerate(DT.iid)); obs = 0.0
  for r in eachrow(L.tr[in.(L.tr.iid, Ref(keys(sidx))) .& (L.tr.out .== "alive") .& (L.tr.dia_cm .< 12.7) .& (L.tr.dia2_cm .>= 12.7), :]); obs += r.w2_ha / DT.fs2[sidx[r.iid]]; end
  for r in eachrow(L.ig[in.(L.ig.iid, Ref(keys(sidx))) .& (L.ig.dia_cm .>= 12.7), :]); obs += r.w_ha / DT.fs2[sidx[r.iid]]; end
  Sb = subS(ST, 1:ns); for i in eachindex(Sb.N0); Sb.cl[i] <= 2 && (Sb.N0[i] = 0.0; Sb.B0[i] = 0.0); end
  H4.simulate!(Sb, [withP(p; lnF=-Inf) for p in P], G); mb = standv(Sb, DT, :sim)
  info("7.2 ingrowth into 12.7 cm over the interval", @sprintf("observed %.1f /ha (plot mean) | warm-start model %.1f /ha (N≥12.7 full run %.1f − without classes 1–2 %.1f)", obs / ns, m.N12 - mb.N12, m.N12, mb.N12))
end
# ================================================================ summary
DBInterface.close!(con)
nf = count(r -> r[1] == "FAIL", RES); np = count(r -> r[1] == "PASS", RES)
println("\n==== SUMMARY: $(np) PASS, $(nf) FAIL, $(count(r -> r[1] == "INFO", RES)) INFO | $(round(time() - t00; digits=0)) s")
for r in RES; r[1] == "FAIL" && println("FAIL  ", r[2]); end
exit(nf > 0 ? 1 : 0)
