# HBAD4P verification, audit round 2 (docs/hbad4p_design.md §5): PASS / FAIL checks by code paths INDEPENDENT of the code under test. Exit 1 on any FAIL.
#   0  hand-worked one-cohort / one-year cases (closed forms written out here)
#   1  kernel units: classes, lanes, closure moments vs BigFloat, closure inversion vs a BigFloat root, upgrowth vs diameter-space quadrature (QuadGK) and
#      Monte Carlo, edge cases, conservation, allocations
#   2  optimised kernel vs a brute-force reference step (tools/hbad4/verify4p_ref.jl: exact functions, O(n²) overtopping, quadrature upgrowth) on real
#      sites; lookup tables vs exact functions; determinism and thread parity
#   3  numerical diffusion of the two-moment closure vs a TREE-LEVEL replay of the real v1 trees (observed per key × class increments)
#   4  process semantics: heights of movers, recruits, seed sources, temperature response, CO2, self-thinning of movers, the open last class
#   5  data pipeline vs SQL on the read-only raw databases (slim_r4 curated_trees + REF_SPECIES, FIADB PLOT / COND / SUBP_COND / TREE)
#   6  empirical-Bayes sapling prior: independent refit, leakage, calibration on val, the remeasured-microplot question
#   7  likelihood / observation model: NB pmf, nll4 recomputed, background, within-cell spreads, sampling-variance assumption
#   8  warm start / parameter space
#   9  stand-level quantities and baselines (persistence null vs a train-estimated transition-matrix null vs the warm start)
# Tags: [spec] = code vs the design document, [phys] / [stat] = the design itself. INFO lines carry numbers that are not pass / fail.
#   HBAD2_DATA_DIR=<fold product> ./julia_gdal.sh --project=. --threads=2 tools/hbad4/verify4p.jl [hbad3p fit dir]
# env: HBAD4_VERIFY_NSUB (sites of the brute-force checks, default 150), HBAD4_VERIFY_SECTIONS (e.g. "0,1,2"; default all), HBAD3_SLIM_DB, HBAD4_FIADB.
include(joinpath(@__DIR__, "data4.jl")); include(joinpath(@__DIR__, "params4.jl")); include(joinpath(@__DIR__, "..", "hbad2", "cmaes.jl"))
include(joinpath(@__DIR__, "verify4p_ref.jl")); import .Ref4
using DuckDB, Random
import Distributions
const H4 = HBAD4P
const ROLE = basename(dirname(normpath(DATA_DIR)))                                           # e.g. k1 (product = …/prep/<role>/product)
const FIT3P = length(ARGS) >= 1 ? ARGS[1] : get(ENV, "HBAD4_FIT3P", joinpath(DATA_DIR, "..", "..", "..", "hbad3p", ROLE, "fit3p_" * ROLE))
const SLIM_DB = get(ENV, "HBAD3_SLIM_DB", joinpath(WSROOT, "landis_region4", "data_r4", "slim_r4.duckdb"))
const FIADB = get(ENV, "HBAD4_FIADB", joinpath(WSROOT, "FIA", "FIASQLITE2PGSQL", "FIADB.duckdb"))
const SPLIT_PLOTS = joinpath(DATA_DIR, "..", "split", "split_plots.csv")
const ROLE_SPLIT = joinpath(DATA_DIR, "..", "..", "..", "folds", "split_$(ROLE).csv")
const FOLDS_CSV = joinpath(dirname(ROLE_SPLIT), "folds.csv")
const NSUB = parse(Int, get(ENV, "HBAD4_VERIFY_NSUB", "150"))
const SECTIONS = Set(parse.(Int, split(get(ENV, "HBAD4_VERIFY_SECTIONS", "0,1,2,3,4,5,6,7,8,9"), ",")))
want(k) = k in SECTIONS

# ---------------------------------------------------------------- reporting
const RES = Tuple{String,String,String}[]
function check(name, ok::Bool, detail="")
  push!(RES, (ok ? "PASS" : "FAIL", name, detail)); println(ok ? "PASS  " : "FAIL  ", name, isempty(detail) ? "" : "  — " * detail); flush(stdout); ok
end
info(name, detail) = (push!(RES, ("INFO", name, detail)); println("INFO  ", name, "  — ", detail); flush(stdout))
section(s) = (println("\n==== ", s); flush(stdout))
reld(a, b; floor=1e-12) = abs(a - b) / max(abs(a), abs(b), floor)
fmt(x) = @sprintf("%.4g", x)
pct(x) = @sprintf("%.1f %%", 100x)

# ---------------------------------------------------------------- helpers
"A Sites4 of a subset of sites (every site has the same number of cells in this layout)."
function subS(S, idx)
  nc = S.off[2] - S.off[1]; all(diff(S.off) .== nc) || error("subS: sites of unequal size")
  r = reduce(vcat, [collect(S.off[s]:S.off[s+1]-1) for s in idx])
  H4.Sites4([1 + (q - 1) * nc for q in 1:length(idx)+1], S.nk, S.sp[r], S.cl[r], S.ln[r], S.N0[r], S.B0[r], S.H0[r], S.A0[r],
            copy(S.N0[r]), copy(S.B0[r]), copy(S.H0[r]), copy(S.A0[r]), S.nyr[idx], S.frac[idx], S.planted[idx], S.W[idx], S.Wn[idx], S.T[idx], S.C[idx],
            S.seed0[:, idx], S.lz, S.lw)
end
emptytabs(nk) = H4.Tabs(zeros(0, nk), zeros(0), zeros(0), zeros(0), zeros(0))
cellix(S, s, k, j, v) = S.off[s] + ((k - 1) * NCLS + (j - 1)) * NL + (v - 1)
withG(G; kw...) = H4.GlobP4((haskey(kw, f) ? kw[f] : getfield(G, f) for f in fieldnames(H4.GlobP4))...)
withP(p; kw...) = H4.SpP4((haskey(kw, f) ? kw[f] : getfield(p, f) for f in fieldnames(H4.SpP4))...)
reset!(S, s) = (for i in S.off[s]:S.off[s+1]-1; S.N[i] = S.N0[i]; S.B[i] = S.B0[i]; S.H[i] = S.H0[i]; S.A[i] = S.A0[i]; end)
kafun(j, q2) = H4.texp_a(j, q2)                    # the kernel's closure inversion (Ref4 implementation tests)
"Serial (1-thread path) whole-interval run of every site of S with one scratch."
function serial_run!(S, P, G; exact=false)
  X = H4.Scratch(maximum(diff(S.off))); T_ = exact ? emptytabs(length(P)) : H4.Tabs(P, G)
  for s in 1:H4.nsites(S)
    reset!(S, s)
    for _ in 1:S.nyr[s]; H4.step_site!(S, P, G, X, s, 1.0, T_); end
    S.frac[s] > 0 && H4.step_site!(S, P, G, X, s, S.frac[s], T_)
  end
  S
end

println("HBAD4P verification (round 2) | product $(DATA_DIR) | role $(ROLE) | threads $(Threads.nthreads()) | region $(REGION) | NSUB $(NSUB) | sections $(sort(collect(SECTIONS)))")
t00 = time()
CS = cached_sets(["train", "val", "test"]); KEYS = CS.keys; TR = CS.TR; nk = length(KEYS)
DT = CS.sets["train"]; ST = DT.S; ns = H4.nsites(ST)
# warm start exactly as smoke4.jl: HBAD3P's dropped crowding gain / crown factor at the train v1 state rescale ε
gl3 = Dict(Symbol(r.name) => Float64(r.value) for r in eachrow(CSV.read(joinpath(FIT3P, "globals.csv"), DataFrame)))
RDv1 = [sum(H4.sdi(ST.N0[i], H4.qmd(ST.N0[i], ST.B0[i])) for i in ST.off[s]:ST.off[s+1]-1 if ST.N0[i] > 0; init=0.0) / 1500 for s in 1:ns]
EPSFAC = mean(exp.(get(gl3, :gR, 0.0) .* RDv1)) * mean((CS.cr_train ./ 0.4) .^ get(gl3, :φ, 0.0))
e3, f3, g3 = hbad3p_start(FIT3P, KEYS; eps_factor=EPSFAC)
SP = space4(KEYS, TR, CS.sD, CS.sH; start_eps=e3, start_lnF=f3, start_glob=g3); P, G, Ob = SP.decode(SP.z0)
lz, lw = ST.lz, ST.lw
rng = MersenneTwister(20261026); SUB = sort(randperm(rng, ns)[1:min(NSUB, ns)])
println("sets loaded in $(round(time() - t00; digits=1)) s; train sites $(ns), keys $(join(KEYS, ",")); ε factor $(round(EPSFAC; digits=4))")

# ---------------------------------------------------------------- one-site builder for the hand cases
"One site, nkk keys; cohorts (k, j, v, N, QMD, H, A); Δ = 1 → one whole year, else one partial year."
function onesite(cohorts; nkk=1, Δ=1.0, W=0.9, Wn=0.95, T=18.0, C=400.0, pl=false, seed0=falses(nkk, 1))
  nc = nkk * NCLS * NL
  sp = Int32[k for k in 1:nkk for _ in 1:NCLS for _ in 1:NL]; cl = Int8[j for _ in 1:nkk for j in 1:NCLS for _ in 1:NL]; ln = Int8[v for _ in 1:nkk for _ in 1:NCLS for v in 1:NL]
  N0 = zeros(nc); B0 = zeros(nc); H0 = zeros(nc); A0 = zeros(nc)
  for (k, j, v, N, D, Hh, A) in cohorts; i = ((k - 1) * NCLS + (j - 1)) * NL + v; N0[i] = N; B0[i] = N * π / 4e4 * D^2; H0[i] = Hh; A0[i] = A; end
  H4.Sites4([1, nc + 1], nkk, sp, cl, ln, N0, B0, H0, A0, copy(N0), copy(B0), copy(H0), copy(A0), [Δ == 1.0 ? 1 : 0], [Δ == 1.0 ? 0.0 : Δ], [pl], [W], [Wn], [T], [C],
            BitMatrix(seed0), lz, lw)
end
ci(k, j, v) = ((k - 1) * NCLS + (j - 1)) * NL + v
Gh = withG(G; b=2.0, a=2.0, gR=0.0, θ=8.0, ω=1.0, s=1.2, c=1.2, η=0.0, λ=0.0, μ=0.0, kh=5.0, m0=0.0, ms=0.0, σST=0.0, β=2.0, ζ=-0.5, mW=0.0, mA=1e4,
           pD=0.0, pH=0.0, pM=0.0, σv=0.3, sdimax=1500.0, am=1.0, ρ=0.0, f0=0.0, wT=Inf, βC=0.0, κ=0.0, λ0=3.0, λ1=-0.07, ε0=0.3, ωR=0.0)
Ph = H4.SpP4(0.0, -Inf, 30.0, 3.0, 1e12, 18.0, 18.0)            # ε = 1, no recruitment, Hmax 30 m, ST 3, no senescence, T_opt = T_ref = 18
hs_(D; Hm=30.0, s=1.2, c=1.2) = 1.37 + (Hm - 1.37) * (1 - exp(-s * D / (Hm - 1.37)))^c
"dD of a lone cohort under (P, G) in a one-year step (read from the kernel's scratch)."
function lone_dD(coh, PP, GG; kw...)
  S0 = onesite([coh]; kw...); X = H4.Scratch(length(S0.N)); H4.step_site!(S0, PP, GG, X, 1, 1.0, emptytabs(length(PP))); X.dD[ci(coh[1], coh[2], coh[3])]
end

# ================================================================ 0. hand-computed cases
if want(0)
section("0. hand-computed one-cohort / one-year cases (closed forms, independent of the kernel and of Ref4)")
let c = 15.24, w = 5.08, U = 17.78, L = 12.7, N = 100.0, Hh = 14.0, A = 40.0
  q2 = c^2 + w^2 / 12; Dq = sqrt(q2)                          # uniform density on class 3 (closure a = 0)
  O = 0.5 * N * (Dq / 25.4)^1.605 / 1500                      # own SDI only, ramp r(0) = ½
  g = exp(-2.0 * O) * 0.9                                     # light × W^ω (ω = 1, W = 0.9)
  hyd = 1 - (Hh / 33.0)^8                                     # 1.1 Hmax = 33 m, θ = 8
  dV = (Dq / 20)^2 * 122.98 * g * hyd; dD = dV / (2 * Dq * Hh)
  Nm = N * dD / w                                             # uniform: the top dD of the class moves
  Bm = π / 4e4 * (N / w) * ((U + dD)^3 - U^3) / 3             # ∫_{U−dD}^{U} (D + dD)² dD · N / w
  B1 = π / 4e4 * N * (q2 + 2dD * c + dD^2)                    # E[(D + dD)²], E[D] = c
  dH = hs_(Dq + dD) - hs_(Dq); Hn = Hh + dH
  Dm = sqrt(((U + dD)^3 - U^3) / (3dD)); Ds = sqrt((B1 - Bm) / (π / 4e4 * (N - Nm)))
  Dall = sqrt(q2 + 2dD * Dq + dD^2)                           # kernel convention for the cohort's grown QMD (E[D] ≈ QMD; see 4.1b)
  hm = Hn + hs_(Dm) - hs_(Dall); hsv = Hn + hs_(Ds) - hs_(Dall); f = N * Hn / (Nm * hm + (N - Nm) * hsv)
  S1 = onesite([(1, 3, 1, N, Dq, Hh, A)]); H4.simulate!(S1, [Ph], Gh; exact=true); i3 = ci(1, 3, 1); i4 = ci(1, 4, 1)
  ok = isapprox(S1.N[i4], Nm; rtol=1e-9) && isapprox(S1.N[i3], N - Nm; rtol=1e-9) && isapprox(S1.B[i4], Bm; rtol=1e-9) && isapprox(S1.B[i3], B1 - Bm; rtol=1e-9) &&
       isapprox(S1.H[i4], hm * f; rtol=1e-9) && isapprox(S1.H[i3], hsv * f; rtol=1e-9) && isapprox(S1.A[i3], A + 1; rtol=1e-14) && isapprox(S1.A[i4], A + 1; rtol=1e-14)
  check("0.1 uniform class-3 cohort, one year: movers N·dD/w, their B = ∫(D+dD)², stayers B1 − Bm, heights H_n + H*(D_group) − H*(D_all) rescaled, A + 1", ok,
        @sprintf("dD %.5f cm: N3 %.6f/%.6f N4 %.6f/%.6f B3 %.7f/%.7f B4 %.7f/%.7f H3 %.5f/%.5f H4 %.5f/%.5f (kernel/hand)", dD, S1.N[i3], N - Nm, S1.N[i4], Nm, S1.B[i3], B1 - Bm,
                 S1.B[i4], Bm, S1.H[i3], hsv * f, S1.H[i4], hm * f))
  S2 = onesite([(1, 3, 1, N, Dq, Hh, A)]; Δ=0.4); H4.simulate!(S2, [Ph], Gh; exact=true)
  check("0.2 partial final year Δ = 0.4: movers N·0.4dD/w, A + 0.4", isapprox(S2.N[i4], N * 0.4dD / w; rtol=1e-9) && isapprox(S2.A[i3], A + 0.4; rtol=1e-14),
        @sprintf("N4 %.6f vs %.6f", S2.N[i4], N * 0.4dD / w))
  Gm = withG(Gh; pD=0.2, wT=5.0, βC=0.58); Pm = withP(Ph; Topt=16.0, Tref=18.0)
  S3 = onesite([(1, 3, 3, N, Dq, Hh, A)]; pl=true, T=20.0, C=420.0); H4.simulate!(S3, [Pm], Gm; exact=true)
  fac = exp(0.3 * sqrt(2)) * exp(0.2) * exp(-((20.0 - 16.0)^2 - (18.0 - 16.0)^2) / 25.0) * (1 + 0.58 * log(420 / 390.1))
  check("0.3 growth multipliers e^{σv z} (z = +√2) · e^{pD} · f_T (centred at T_ref) · f_CO2 scale dD exactly", isapprox(S3.N[ci(1, 4, 3)], N * fac * dD / w; rtol=1e-9),
        @sprintf("movers %.6f vs hand %.6f (factor %.5f)", S3.N[ci(1, 4, 3)], N * fac * dD / w, fac))
  # 0.10 multi-class: dD = 7 cm (δ = 1.378 class widths): ξ ∈ [0, 0.622) → class 4, [0.622, 1) → class 5, nothing stays
  d0 = lone_dD((1, 3, 1, N, Dq, Hh, A), [Ph], Gh); P7 = [withP(Ph; lnε=log(7.0 / d0))]
  S4 = onesite([(1, 3, 1, N, Dq, Hh, A)]); H4.simulate!(S4, P7, Gh; exact=true)
  x = 2 - 7.0 / w; Dcut = L + x * w
  B4h = π / 4e4 * (N / w) * ((Dcut + 7)^3 - (L + 7)^3) / 3; B5h = π / 4e4 * (N / w) * ((U + 7)^3 - (Dcut + 7)^3) / 3
  check("0.10 multi-class transfer, dD = 7 cm: N4 = 0.622 N, N5 = 0.378 N, none stays, B per segment = ∫ (D + 7)²", isapprox(S4.N[ci(1, 4, 1)], x * N; rtol=1e-9) &&
        isapprox(S4.N[ci(1, 5, 1)], (1 - x) * N; rtol=1e-9) && S4.N[i3] == 0 && isapprox(S4.B[ci(1, 4, 1)], B4h; rtol=1e-9) && isapprox(S4.B[ci(1, 5, 1)], B5h; rtol=1e-9),
        @sprintf("N3 %.3g N4 %.6f/%.6f N5 %.6f/%.6f B4 %.7f/%.7f B5 %.7f/%.7f", S4.N[i3], S4.N[ci(1, 4, 1)], x * N, S4.N[ci(1, 5, 1)], (1 - x) * N, S4.B[ci(1, 4, 1)], B4h, S4.B[ci(1, 5, 1)], B5h))
end
# 0.11 into the open last class + growth inside it: class 15 uniform (dD = 2 cm → movers N·2/5.08 into 16) and a class-16 cohort (B grows with QMD)
let L = 73.66, w = 5.08, U = 78.74, c = 76.2, N = 10.0, q2 = c^2 + w^2 / 12, Dq = sqrt(q2)
  Pz = withP(Ph; Hmax=40.0)
  d0 = lone_dD((1, 15, 1, N, Dq, 30.0, 80.0), [Pz], Gh); PP = [withP(Pz; lnε=log(2.0 / d0))]
  S5 = onesite([(1, 15, 1, N, Dq, 30.0, 80.0)]); H4.simulate!(S5, PP, Gh; exact=true)
  Bm = π / 4e4 * (N / w) * ((U + 2)^3 - U^3) / 3
  ok1 = isapprox(S5.N[ci(1, 16, 1)], N * 2 / w; rtol=1e-9) && isapprox(S5.B[ci(1, 16, 1)], Bm; rtol=1e-9)
  D16 = 85.0; d16 = lone_dD((1, 16, 1, N, D16, 32.0, 120.0), PP, Gh)
  S6 = onesite([(1, 16, 1, N, D16, 32.0, 120.0)]); H4.simulate!(S6, PP, Gh; exact=true)
  ok2 = isapprox(S6.B[ci(1, 16, 1)], π / 4e4 * N * (D16 + d16)^2; rtol=1e-9) && S6.N[ci(1, 16, 1)] == N
  check("0.11 class 15 → open class 16 (movers N dD/w) and growth inside class 16 (B = N π/4 (QMD + dD)², the spec's E[D] ≈ QMD)", ok1 && ok2,
        @sprintf("N16 %.6f/%.6f B16 %.7f/%.7f | class-16 cohort dD %.4f: B %.7f/%.7f", S5.N[ci(1, 16, 1)], N * 2 / w, S5.B[ci(1, 16, 1)], Bm, d16, S6.B[ci(1, 16, 1)], π / 4e4 * N * (D16 + d16)^2))
end
# 0.4 survival (growth off): h = fM [m0/LONG + ms e^{σST (ST − 3)} e^{−β gm} (D/20)^ζ] + e^{mA (A/LONG − 1)}, fM = e^{pM} e^{mW (Wn − W)/0.1}
let N = 80.0, Dq = 25.0, A = 100.0, Lg = 200.0, Δ = 0.37
  Gs = withG(Gh; m0=1.0, ms=0.05, σST=0.3, β=2.0, am=1.0, ζ=-0.5, mA=10.0, mW=0.2, pM=0.1, wT=6.0, βC=0.5)
  Ps = H4.SpP4(-Inf, -Inf, 30.0, 4.0, Lg, 17.0, 18.5)
  O = 0.5 * N * (Dq / 25.4)^1.605 / 1500; cw = 0.9 * exp(-((19.0 - 17.0)^2 - (18.5 - 17.0)^2) / 36) * (1 + 0.5 * log(410 / 390.1)); gm = exp(-O) * cw
  h = exp(0.1) * exp(0.2 * (0.95 - 0.9) / 0.1) * (1.0 / Lg + 0.05 * exp(0.3) * exp(-2gm) * (Dq / 20)^-0.5) + exp(10 * (A / Lg - 1))
  S4 = onesite([(1, 5, 1, N, Dq, 20.0, A)]; pl=true, Δ=Δ, T=19.0, C=410.0); H4.simulate!(S4, [Ps], Gs; exact=true); i = ci(1, 5, 1)
  check("0.4 survival e^{−hΔ} with the documented hazard (planted, drought anomaly, ST, size, senescence; climate and CO2 in gm) on N and B alike",
        isapprox(S4.N[i], N * exp(-h * Δ); rtol=1e-10) && isapprox(H4.qmd(S4.N[i], S4.B[i]), Dq; rtol=1e-12), @sprintf("N %.8f vs %.8f (h = %.5f /yr)", S4.N[i], N * exp(-h * Δ), h))
end
# 0.5 recruitment: (a) empty plot, no seed → ε0; (b) seed tree at v1; (c) current seed source 0.5 vs 1.5 trees/ha ≥ 12.7 cm (SEED_MIN = 1); partial year
let lnF = -2.0, Pr = H4.SpP4(-Inf, -2.0, 30.0, 4.0, 1e12, 18.0, 18.0), Gr = withG(Gh; ωR=0.5)
  rate0 = 185.2 * exp(lnF) * 0.9^0.5
  S5 = onesite(Tuple{Int,Int,Int,Float64,Float64,Float64,Float64}[]); H4.simulate!(S5, [Pr], Gr; exact=true)
  q = H4.qmd(sum(S5.N[1:3]), sum(S5.B[1:3]))
  okA = isapprox(sum(S5.N[1:3]), 0.3 * rate0; rtol=1e-12) && isapprox(S5.N[1] / sum(S5.N[1:3]), 0.5; rtol=1e-12) && isapprox(q, 2.54; rtol=1e-12) &&
        all(isapprox(S5.H[v], hs_(2.54); rtol=1e-12) for v in 1:3) && all(isapprox(S5.A[v], 4.5; rtol=1e-12) for v in 1:3)
  S5b = onesite(Tuple{Int,Int,Int,Float64,Float64,Float64,Float64}[]; Δ=0.4); H4.simulate!(S5b, [Pr], Gr; exact=true)
  okA2 = isapprox(sum(S5b.N[1:3]), 0.4 * 0.3 * rate0; rtol=1e-12) && isapprox(S5b.A[1], 4.5; rtol=1e-12)
  S5c = onesite(Tuple{Int,Int,Int,Float64,Float64,Float64,Float64}[]; seed0=trues(1, 1)); H4.simulate!(S5c, [Pr], Gr; exact=true)
  okB = isapprox(sum(S5c.N[1:3]), rate0; rtol=1e-12)
  res = Float64[]
  for n3 in (0.5, 1.5)
    S6 = onesite([(1, 3, 1, n3, 16.0, 15.0, 30.0)]); H4.simulate!(S6, [Pr], Gr; exact=true)
    RD = n3 * (16.0 / 25.4)^1.605 / 1500; push!(res, sum(S6.N[1:3]) / (185.2 * exp(lnF - 3.0 * exp(-0.07) * RD) * 0.9^0.5))
  end
  check("0.5 recruitment: rate 185.2 e^{lnF − λ0 e^{λ1(ST−3)} RD} W^{ωR} (seed ? 1 : ε0) Δ, lanes 0.5/0.25/0.25, D 2.54 cm, H*(2.54), age 4.5 after the step; seed = v1 seed tree or ≥ 1 tree/ha ≥ 12.7 cm now",
        okA && okA2 && okB && isapprox(res[1], 0.3; rtol=1e-12) && isapprox(res[2], 1.0; rtol=1e-12),
        @sprintf("no seed: %.6f vs %.6f, QMD %.6f, A %.3f; Δ=0.4: %.6f vs %.6f; seed0: %.6f vs %.6f; 0.5 tree/ha → factor %.4f, 1.5 → %.4f", sum(S5.N[1:3]), 0.3rate0, q, S5.A[1],
                 sum(S5b.N[1:3]), 0.12rate0, sum(S5c.N[1:3]), rate0, res[1], res[2]))
end
# 0.6 self-thinning of one cohort at twice sdimax (growth, mortality off): N halves, QMD unchanged
let Dq = 30.0, N = 3000 / (30.0 / 25.4)^1.605, Pz = withP(Ph; lnε=-Inf)
  S7 = onesite([(1, 6, 1, N, Dq, 20.0, 50.0)]); H4.simulate!(S7, [Pz], Gh; exact=true); i = ci(1, 6, 1)
  check("0.6 self-thinning: one cohort at SDI = 2 sdimax → N/2, QMD unchanged", isapprox(S7.N[i], N / 2; rtol=1e-10) && isapprox(H4.qmd(S7.N[i], S7.B[i]), Dq; rtol=1e-12),
        @sprintf("N %.6f vs %.6f", S7.N[i], N / 2))
end
# 0.7 lane mixing: (100, 0, 0) with κΔ = 0.5 → (75, 12.5, 12.5); Δ = 0.5 → κΔ = 0.25 → (87.5, 6.25, 6.25)
let Pz = withP(Ph; lnε=-Inf)
  S8 = onesite([(1, 3, 1, 100.0, 15.0, 12.0, 20.0)]); H4.simulate!(S8, [Pz], withG(Gh; κ=0.5); exact=true)
  S9 = onesite([(1, 3, 1, 100.0, 15.0, 12.0, 20.0)]; Δ=0.5); H4.simulate!(S9, [Pz], withG(Gh; κ=0.5); exact=true)
  a = S8.N[ci(1, 3, 1):ci(1, 3, 3)]; b = S9.N[ci(1, 3, 1):ci(1, 3, 3)]
  check("0.7 lane mixing κΔ re-split with the lane weights (heights, QMD carried)", isapprox(a, [75.0, 12.5, 12.5]; rtol=1e-12) && isapprox(b, [87.5, 6.25, 6.25]; rtol=1e-12) &&
        all(isapprox(S8.H[ci(1, 3, v)], 12.0; rtol=1e-12) for v in 1:3) && all(isapprox(H4.qmd(S8.N[ci(1, 3, v)], S8.B[ci(1, 3, v)]), 15.0; rtol=1e-12) for v in 1:3), "κ=0.5: $(a); Δ=0.5: $(b)")
end
# 0.8 overtopping with the 1 m ramp: heights 18.0 / 17.5 (Δ 0.5 m → r = 0.75 / 0.25) and 18 / 10 (→ 1 / 0), own cohort ½
let Pz = withP(Ph; lnε=-Inf), res = Bool[], msg = String[]
  for (h1, h2, rs, rt) in ((18.0, 17.5, 0.75, 0.25), (18.0, 10.0, 1.0, 0.0), (18.0, 18.0, 0.5, 0.5))
    S10 = onesite([(1, 3, 1, 100.0, 15.31, h1, 30.0), (2, 3, 1, 100.0, 15.31, h2, 30.0)]; nkk=2)
    X = H4.Scratch(length(S10.N)); H4.step_site!(S10, [Pz, Pz], Gh, X, 1, 1.0, emptytabs(2)); sd = 100 * (15.31 / 25.4)^1.605
    push!(res, isapprox(X.O[ci(1, 3, 1)], (0.5 + rt) * sd / 1500; rtol=1e-12) && isapprox(X.O[ci(2, 3, 1)], (0.5 + rs) * sd / 1500; rtol=1e-12))
    push!(msg, @sprintf("H %.1f/%.1f: O %.6f/%.6f vs %.6f/%.6f", h1, h2, X.O[ci(1, 3, 1)], X.O[ci(2, 3, 1)], (0.5 + rt) * sd / 1500, (0.5 + rs) * sd / 1500))
  end
  check("0.8 overtopping O_i = Σ SDI_j r(H_j − H_i)/sdimax, r(Δ) = clamp(½ + Δ/2δH, 0, 1), δH = 1 m (own ½)", all(res), join(msg, "; "))
end
# 0.9 diffusion of a one-tree cohort: a single tree at 17.5 cm (top of class 3). Truth: dD 0.4 → all of it in class 4; dD 0.16 → 17.66 cm, none moves
let N = 14.87, res = Float64[]
  d0 = lone_dD((1, 3, 1, N, 17.5, 15.0, 30.0), [Ph], Gh)
  for target in (0.4, 0.16)
    S11 = onesite([(1, 3, 1, N, 17.5, 15.0, 30.0)]); H4.simulate!(S11, [withP(Ph; lnε=log(target / d0))], Gh; exact=true)
    push!(res, S11.N[ci(1, 4, 1)] / N)
  end
  info("0.9 [phys] closure diffusion of a ONE-TREE cohort at 17.5 cm (r = 0.945 of class 3)",
       @sprintf("dD 0.40: %.1f %% moved (truth 100 %%); dD 0.16: %.1f %% moved (truth 0 %%) — the exponential closure spreads a point mass over the class (a ≈ %.1f)",
                100res[1], 100res[2], H4.texp_a(3, 17.5^2)))
end
end # section 0

# ================================================================ 1. kernel unit checks
if want(1)
section("1. kernel unit checks")
let bad = Tuple{Float64,Int,Int}[], incon = 0
  for d10 in 5:600
    DIA = d10 / 10; D = DIA * 2.54; jt = DIA < 1.0 ? 0 : min(NCLS, (d10 ÷ 10 + 1) ÷ 2); j = H4.dclass(D)
    j == jt || push!(bad, (DIA, j, jt))
    (j >= 1 && !(H4.CLS_LO[j] - 1e-9 <= D < H4.CLS_HI[j] - 1e-9)) && (incon += 1)
  end
  check("1.1 dclass of every 0.1-in diameter 0.5–60.0 in = floor((DIA + 1)/2) (0 below 1 in, capped at 16); consistent with CLS_LO / CLS_HI", isempty(bad) && incon == 0,
        isempty(bad) ? "$(incon) inconsistent with the bounds" : "wrong: " * join(["DIA $(b[1]) → $(b[2]) (expected $(b[3]))" for b in bad[1:min(5, end)]], ", "))
end
for mode in (:sym, :fvs)
  z, w = H4.lanes(mode); m = sum(w .* z); v = sum(w .* z .^ 2) - m^2
  check("1.2 [spec §1] lanes :$mode: weights sum to 1, mean 0, variance 1", abs(sum(w) - 1) < 1e-12 && abs(m) < 1e-12 && abs(v - 1) < 1e-12, "Σw $(fmt(sum(w))) mean $(fmt(m)) var $(fmt(v)) z $(round.(z; digits=4))")
end
check("1.2b [spec §1] the store records the lanes it was built with; build lanes = HBAD4P_TRIPLE ($(LANEMODE))", (ST.lz, ST.lw) == H4.lanes(LANEMODE) && CS.lanemode == LANEMODE)
setprecision(256)
function bigcum(a, ξ)
  a = big(a); ξ = big(ξ); a == 0 && return (ξ, ξ^2 / 2, ξ^3 / 3)
  Z = expm1(a) / a; (expm1(a * ξ) / a / Z, (exp(a * ξ) * (ξ / a - 1 / a^2) + 1 / a^2) / Z, (exp(a * ξ) * (ξ^2 / a - 2ξ / a^2 + 2 / a^3) - 2 / a^3) / Z)
end
let mx = 0.0, wh = (0.0, 0.0), mxc = 0.0
  for a in vcat(-200.0, -60.0, -7.3, -1.0, -0.0101, -0.01, -0.00999, -1e-3, -1e-7, 0.0, 1e-7, 1e-3, 0.00999, 0.01, 0.0100001, 0.05, 0.9, 13.0, 77.0, 200.0),
      ξ in (0.0, 1e-6, 0.01, 0.2, 0.5, 0.77, 0.999, 1.0)
    e = maximum(abs.(H4.texp_cum(a, ξ) .- Float64.(bigcum(a, ξ)))); e > mx && (mx = e; wh = (a, ξ))
  end
  for ξ in (0.3, 0.9, 1.0); mxc = max(mxc, maximum(abs.(H4.texp_cum(0.01 - 1e-12, ξ) .- H4.texp_cum(0.01 + 1e-12, ξ)))); end
  check("1.3 closed-form partial moments texp_cum = BigFloat integrals (a ∈ [−200, 200], incl. the series branch |a| < 0.01)", mx < 1e-8, "max abs err $(fmt(mx)) at (a, ξ) = $(wh); jump across |a| = 0.01: $(fmt(mxc))")
end
let mxa = 0.0, mxq = 0.0, mono = all(j -> issorted(view(H4.R_TAB, :, j)), 1:NCLS-1), wq = ""
  for j in 1:NCLS-1, r in range(0.01, 0.99; length=197)
    q = H4.CLS_LO[j] + r * H4.CLS_W; ak = H4.texp_a(j, q^2); ae = Ref4.aexact(j, q^2)
    _, m1, m2 = H4.texp_cum(ak, 1.0); e2 = H4.CLS_LO[j]^2 + 2H4.CLS_LO[j] * H4.CLS_W * m1 + H4.CLS_W^2 * m2
    e = abs(e2 / q^2 - 1); e > mxq && (mxq = e; wq = "class $j r $(round(r; digits=3))"); mxa = max(mxa, abs(ak - ae) / max(1.0, abs(ae)))
  end
  check("1.4 closure inversion texp_a (table + interpolation) vs an exact BigFloat root, r = QMD position ∈ [0.01, 0.99] of every class: E_a[D²] = QMD² (rel 1e-5)", mono && mxq < 1e-5,
        "R_TAB monotone $(mono); max rel error of E[D²] $(fmt(mxq)) ($(wq)); max |Δa|/max(1,|a|) $(fmt(mxa))")
end
# 1.5 upgrowth! vs the diameter-space quadrature (QuadGK) with the same closure parameter, random cohorts incl. multi-class jumps
let mx = 0.0, mxc = 0.0, nout = 0, wst = "", mxe = 0.0, wse = ""
  for trial in 1:800
    j = rand(rng, 1:NCLS-1); edge = trial <= 200; r = edge ? (isodd(trial) ? 0.004rand(rng) : 0.996 + 0.004rand(rng)) : 0.01 + 0.98rand(rng); q = H4.CLS_LO[j] + r * H4.CLS_W
    n = 1 + 300rand(rng); b = H4.ba(n, q); dD = trial % 5 == 0 ? 3 * H4.CLS_W * rand(rng) : 1.5rand(rng)
    out = Dict{Int,Tuple{Float64,Float64}}()
    Btot = H4.upgrowth!((t, Nt, Bt, Dt) -> (out[t] = get(out, t, (0.0, 0.0)) .+ (Nt, Bt)), j, n, b, dD)
    cons = max(abs(sum(first.(values(out))) - n) / n, abs(sum(last.(values(out))) - Btot) / Btot)
    if edge || dD >= H4.CLS_W
      cons > mxe && (mxe = cons; wse = "class $j r $(round(r; digits=4)) dD $(round(dD; digits=2)): Σ B out / B after growth − 1 = $(fmt(sum(last.(values(out))) / Btot - 1))"); edge && continue
    end
    grp, B1, _ = Ref4.upgrowth(j, n, b, dD, kafun)
    ref = Dict(t => (Nt, Bt) for (t, Nt, Bt) in grp)
    for t in union(keys(out), keys(ref))
      a = get(out, t, (0.0, 0.0)); c = get(ref, t, (0.0, 0.0))
      e = max(abs(a[1] - c[1]) / n, abs(a[2] - c[2]) / b) / (dD >= H4.CLS_W ? 1e2 : 1.0)    # (every tree moves: the movers' cap min(·, Btot − Bm) acts at the inversion's 1e-6)
      e > mx && (mx = e; wst = "j $j r $(round(r; digits=3)) dD $(round(dD; digits=3)) target $t: kernel $(a) ref $(c)")
      (t != j && a[1] > 1e-9 * n && !(H4.CLS_LO[t] - 1e-9 <= H4.qmd(a...) <= H4.CLS_HI[t] + 1e-9)) && (nout += 1)
    end
    dD < H4.CLS_W && (mxc = max(mxc, cons)); mxc = max(mxc, abs(Btot - B1) / B1)
  end
  check("1.5 upgrowth! = diameter-space quadrature of the truncated exponential (600 random cohorts, QMD position r ∈ [0.01, 0.99], dD up to 3 class widths), per target class N and B (1e-7 of the cohort; 1e-5 when every tree moves)", mx < 1e-7,
        "max |Δ| / (n or b) $(fmt(mx))" * (mx < 1e-7 ? "" : " | worst $(wst)"))
  check("1.5b upgrowth! conserves N and B (B after growth = b + n π/4 (2 dD E[D] + dD²), E[D] = quadrature) when some trees stay (dD < w); mover groups' QMD in their class", mxc < 1e-11 && nout == 0,
        "max rel conservation error $(fmt(mxc)); mover groups outside their class $(nout)")
  check("1.5d [spec §2.3 'N and B are conserved'] … also when EVERY tree moves (dD ≥ w) or the QMD position is outside the closure's range r ∈ [0.005, 0.995] (a clamped to ±200)", mxe < 1e-9,
        "max rel B error $(fmt(mxe)) ($(wse)): with no stayers the closure's E_a[D²] ≠ QMD² residual is dropped (HBAD4P.jl:123-128: the stayers' Btot − Bm absorbs it only when Ns > 0; the min(·, Btot − Bm) cap clips movers when the closure overshoots)")
end
let rs = Random.Xoshiro(7), worst = 0.0, msg = String[]
  for (j, r, dD) in ((3, 0.15, 0.4), (3, 0.85, 0.4), (1, 0.02, 0.3), (5, 0.5, 7.0), (14, 0.6, 9.0), (2, 0.97, 0.16))
    n = 100.0; q = H4.CLS_LO[j] + r * H4.CLS_W; b = H4.ba(n, q); a = H4.texp_a(j, q^2)
    out = Dict{Int,Float64}(); H4.upgrowth!((t, Nt, Bt, Dt) -> (out[t] = get(out, t, 0.0) + Nt), j, n, b, dD)
    u = rand(rs, 2_000_000); ξ = abs(a) < 1e-9 ? u : (a > 0 ? 1 .+ log.(u .* (1 - exp(-a)) .+ exp(-a)) ./ a : log1p.(u .* expm1(a)) ./ a)
    cl = H4.dclass.(H4.CLS_LO[j] .+ H4.CLS_W .* ξ .+ dD)
    for t in keys(out); p = mean(cl .== t); se = sqrt(p * (1 - p) / length(u)) * n; z = abs(out[t] - n * p) / max(se, 1e-9); worst = max(worst, z); end
    push!(msg, "j$j r$r dD$dD→$(sort(collect(keys(out))))")
  end
  check("1.5c upgrowth! = Monte Carlo of the closure (2e6 sampled trees per case)", worst < 5, "worst $(round(worst; digits=2)) standard errors | " * join(msg, " "))
end
let msg = String[], ok = true
  for (lab, j, r, dD) in (("dD = 1e-12", 4, 0.5, 1e-12), ("dD = w exactly (all move one class)", 4, 0.5, 5.08), ("dD = 2w", 4, 0.3, 10.16), ("class 15, dD 30 cm (all → 16)", 15, 0.4, 30.0),
                          ("QMD at the class floor (a clamped to −200)", 1, 0.0, 0.3), ("QMD 1e-6 below the class top", 6, 1 - 1e-6 / 5.08, 0.05))
    n = 50.0; q = H4.CLS_LO[j] + r * H4.CLS_W; b = H4.ba(n, q); out = Dict{Int,Tuple{Float64,Float64}}()
    Btot = H4.upgrowth!((t, Nt, Bt, Dt) -> (out[t] = get(out, t, (0.0, 0.0)) .+ (Nt, Bt)), j, n, b, dD)
    Ns = sum(first.(values(out))); Bs = sum(last.(values(out)))
    good = abs(Ns - n) < 1e-9 * n && abs(Bs - Btot) < 1e-5 * Btot && all(isfinite, Bs) && all(v -> v[1] >= 0 && v[2] >= 0, values(out))
    lab == "dD = w exactly (all move one class)" && (good &= isapprox(get(out, j + 1, (0.0, 0.0))[1], n; rtol=1e-12))
    lab == "class 15, dD 30 cm (all → 16)" && (good &= isapprox(get(out, 16, (0.0, 0.0))[1], n; rtol=1e-12))
    ok &= good; push!(msg, "$(lab): targets $(sort(collect(keys(out)))) $(good ? "ok" : "BAD $(out)") (B rel $(fmt(Bs / Btot - 1)))")
  end
  check("1.6 upgrowth! edge cases (tiny dD, dD = w and 2w, everything into the open class, QMD at a class bound): N conserved, B to 1e-5 (see 1.5d), non-negative", ok, join(msg, "; "))
end
let Sx = subS(ST, SUB)
  Pz = [withP(p; lnε=-Inf, lnF=-Inf, longevity=1e12) for p in P]
  Gz = withG(G; m0=0.0, ms=0.0, mA=1e4, λ=0.0, ρ=0.0, κ=0.0, sdimax=1e15)
  H4.simulate!(Sx, Pz, Gz); dt = Sx.nyr .+ Sx.frac; bad = 0
  for s in 1:H4.nsites(Sx), i in Sx.off[s]:Sx.off[s+1]-1
    (reld(Sx.N[i], Sx.N0[i]) < 1e-13 && reld(Sx.B[i], Sx.B0[i]) < 1e-13 && reld(Sx.H[i], Sx.H0[i]) < 1e-13 && (Sx.N0[i] > 0 ? isapprox(Sx.A[i], Sx.A0[i] + dt[s]; atol=1e-9) : Sx.A[i] == Sx.A0[i])) || (bad += 1)
  end
  check("1.7 [spec §5] zero growth + zero mortality (+ no recruitment / thinning / mixing) leave every cohort unchanged (rel 1e-13); A advances by dt", bad == 0, "$(bad) cells changed of $(length(Sx.N))")
end
let Sx = subS(ST, SUB)
  Pz = [withP(p; lnF=-Inf, longevity=1e12) for p in P]; Gz = withG(G; m0=0.0, ms=0.0, mA=1e4, κ=0.0, sdimax=1e15)
  H4.simulate!(Sx, Pz, Gz); mx = 0.0; mxa = 0.0; ntiny = count(x -> 0 < x <= 1e-9, Sx.N0)
  for s in 1:H4.nsites(Sx), k in 1:nk, v in 1:NL
    a = sum(Sx.N[cellix(Sx, s, k, j, v)] for j in 1:NCLS); b = sum(Sx.N0[cellix(Sx, s, k, j, v)] for j in 1:NCLS); mx = max(mx, reld(a, b; floor=1e-9)); mxa = max(mxa, abs(a - b))
  end
  check("1.8 [spec §5] upgrowth conserves N per (site, key, lane) over whole intervals (growth only; abs 1e-8 trees/ha)", mxa < 1e-8,
        "max abs diff $(fmt(mxa)) trees/ha (rel $(fmt(mx))): a cell with 0 < N ≤ 1e-9 is inactive and an inflow OVERWRITES it (HBAD4P.jl:249 N[i] = m, not +=); v1 cells that small: $(ntiny)")
end
let Sx = subS(ST, SUB[1:min(80, length(SUB))])
  Pz = [withP(p; lnε=-Inf, lnF=-Inf, longevity=1e12) for p in P]; Gz = withG(G; m0=0.0, ms=0.0, mA=1e4, λ=0.0, ρ=0.0, κ=0.4, sdimax=1e15)
  fl = [1.0 + 0.5 * (Sx.ln[i] - 2) for i in eachindex(Sx.N0)]; Sx.N0 .*= fl; Sx.B0 .*= fl; Sx.H0 .+= [Sx.N0[i] > 0 ? Sx.ln[i] : 0.0 for i in eachindex(Sx.H0)]
  H4.simulate!(Sx, Pz, Gz); mx = 0.0; nout = 0
  for s in 1:H4.nsites(Sx), k in 1:nk, j in 1:NCLS
    i = cellix(Sx, s, k, j, 1); r = i:i+2; n0 = sum(Sx.N0[r]); n0 > 0 || continue; dt = Sx.nyr[s] + Sx.frac[s]
    mx = max(mx, reld(sum(Sx.N[r]), n0), reld(sum(Sx.B[r]), sum(Sx.B0[r])), reld(sum(Sx.N[r] .* Sx.H[r]), sum(Sx.N0[r] .* Sx.H0[r])), reld(sum(Sx.N[r] .* Sx.A[r]), sum(Sx.N0[r] .* (Sx.A0[r] .+ dt))))
    for t in r; Sx.N[t] > 1e-9 && !(H4.CLS_LO[j] - 1e-6 <= H4.qmd(Sx.N[t], Sx.B[t]) <= H4.CLS_HI[j] + 1e-6) && (nout += 1); end
  end
  check("1.9 lane mixing conserves N, B, Σ N·H, Σ N·A per (site, key, class) and keeps lane QMDs in the class", mx < 1e-10 && nout == 0, "max rel diff $(fmt(mx)); lanes outside $(nout)")
end
let s = SUB[1], X = H4.Scratch(ST.off[2] - ST.off[1]), T_ = H4.Tabs(P, G), Sx = subS(ST, [s])
  H4.step_site!(Sx, P, G, X, 1, 1.0, T_); reset!(Sx, 1)
  al = @allocated H4.step_site!(Sx, P, G, X, 1, 1.0, T_)
  reset!(Sx, 1); al2 = @allocated H4.step_site!(Sx, P, G, X, 1, 0.37, T_)
  Sy = subS(ST, SUB); H4.simulate!(Sy, P, G); alS = @allocated H4.simulate!(Sy, P, G)
  check("1.10 no heap allocation in the hot loop (step_site! of a real site, whole and partial year)", al == 0 && al2 == 0,
        "step_site! $(al) / $(al2) bytes per call (the sort! of overtopping!, HBAD4P.jl:179: the default sort allocates a scratch vector for > 10 cells) | simulate! of $(length(SUB)) sites $(round(alS / 1e6; digits=2)) MB")
end
end # section 1

# ================================================================ 2. kernel vs the brute-force reference on real sites
if want(2)
section("2. optimised kernel vs brute-force reference (train subset of $(length(SUB)) sites)")
let Sx = subS(ST, SUB), X = H4.Scratch(Sx.off[2] - Sx.off[1]), T0 = emptytabs(nk), mx = Dict(:N => 0.0, :B => 0.0, :H => 0.0, :A => 0.0, :O => 0.0, :dD => 0.0), worst = "", nstep = 0
  mxfree = 0.0; mxe = Dict(:N => 0.0, :B => 0.0)
  for s in 1:H4.nsites(Sx)
    r = Sx.off[s]:Sx.off[s+1]-1; reset!(Sx, s)
    Nf = Sx.N0[r]; Bf = Sx.B0[r]; Hf = Sx.H0[r]; Af = Sx.A0[r]; sd0 = Sx.seed0[:, s]
    for y in 1:Sx.nyr[s] + (Sx.frac[s] > 0)
      Δ = y <= Sx.nyr[s] ? 1.0 : Sx.frac[s]
      N = Sx.N[r]; B = Sx.B[r]; Hh = Sx.H[r]; A = Sx.A[r]; Ne = copy(N); Be = copy(B); He = copy(Hh); Ae = copy(A); act0 = N .> 1e-9
      H4.step_site!(Sx, P, G, X, s, Δ, T0)
      rr = Ref4.step!(N, B, Hh, A, Sx.sp[r], Sx.cl[r], Sx.ln[r], nk, P, G, Sx.W[s], Sx.Wn[s], Sx.T[s], Sx.C[s], Sx.planted[s], Δ, lz, lw, sd0, kafun; overwrite=true)
      y <= 2 && s <= 40 && Ref4.step!(Ne, Be, He, Ae, Sx.sp[r], Sx.cl[r], Sx.ln[r], nk, P, G, Sx.W[s], Sx.Wn[s], Sx.T[s], Sx.C[s], Sx.planted[s], Δ, lz, lw, sd0, Ref4.aexact)
      Ref4.step!(Nf, Bf, Hf, Af, Sx.sp[r], Sx.cl[r], Sx.ln[r], nk, P, G, Sx.W[s], Sx.Wn[s], Sx.T[s], Sx.C[s], Sx.planted[s], Δ, lz, lw, sd0, kafun; overwrite=true)
      nstep += 1
      fl = sum(N); flb = sum(B)                 # N, B: |Δ| relative to the SITE totals (closed form vs quadrature of tail masses: texp_cum's series branch is exact to ~3e-9 of a cohort, 1.3; the reference mimics the kernel's overwrite of inactive remnants, 1.8)
      for (q, i) in enumerate(r)
        live = N[q] > 1e-4
        for (nm, a, b) in ((:N, Sx.N[i], N[q]), (:B, Sx.B[i], B[q]), (:H, live ? Sx.H[i] : 0.0, live ? Hh[q] : 0.0), (:A, live ? Sx.A[i] : 0.0, live ? A[q] : 0.0),
                           (:O, X.O[q], rr.O[q]), (:dD, X.dD[q], rr.dD[q]))
          (nm in (:O, :dD) && !act0[q]) && continue
          d = abs(a - b) / max(abs(a), abs(b), nm == :N ? fl : nm == :B ? flb : 1e-9)
          nm in (:N, :B) && (d = abs(a - b) / (nm == :N ? fl : flb))
          nm == :O && (d = abs(a - b) / max(abs(a), abs(b), 1e-6 * rr.RD))     # (O of the top cohorts = tot − prefix sum: cancellation at 1e-16 of RD)
          d > mx[nm] && (mx[nm] = d; worst = "site $(SUB[s]) year $y cell $(q) $(nm) kernel $(a) ref $(b)")
        end
        if y <= 2 && s <= 40
          mxe[:N] = max(mxe[:N], abs(Sx.N[i] - Ne[q]) / max(abs(Sx.N[i]), abs(Ne[q]), fl)); mxe[:B] = max(mxe[:B], abs(Sx.B[i] - Be[q]) / max(abs(Sx.B[i]), abs(Be[q]), flb))
        end
      end
    end
    mxfree = max(mxfree, maximum(abs(Sx.N[i] - Nf[q]) for (q, i) in enumerate(r)) / sum(Nf))
  end
  ok21 = mx[:N] < 1e-10 && mx[:B] < 1e-10 && all(mx[k] < 1e-8 for k in (:O, :dD, :H, :A))
  check("2.1 kernel (exact functions) = reference, one step at a time from the same state ($(nstep) steps): O, dD, N, B, H, A per cell (N, B: |Δ| / site total ≤ 1e-10; O, dD, H, A: rel 1e-8 on cells > 1e-4 /ha)", ok21,
        "max diff " * join(["$k $(fmt(v))" for (k, v) in mx], ", ") * (ok21 ? "" : " | worst $worst"))
  check("2.1b free-running kernel and reference agree over whole intervals (no discontinuity at floating-point level: the height ramp)", mxfree < 1e-9, "max |ΔN| per cell / site total $(fmt(mxfree))")
  info("2.1c effect of the closure-inversion error (reference with the exact BigFloat root a instead of the kernel's table)", "max rel diff per cell after one step: N $(fmt(mxe[:N])), B $(fmt(mxe[:B])) (first 2 steps of 40 sites)")
end
let p = P[1], T_ = H4.Tabs(P, G), m = Dict{String,Float64}()
  Ds = vcat(2.54 .+ 247.4 .* rand(rng, 20000), [2.54, 12.7, 249.9, 250.0, 260.0])
  for D in Ds, (nm, a, b) in (("H*", H4.ht(T_, p, G, 1, D), H4.hstar(p, G, D)), ("(D/20)^b", H4.pw_b(T_, G, D), (D / 20)^G.b), ("(D/20)^ζ", H4.pw_z(T_, G, D), (D / 20)^G.ζ),
                              ("(D/25.4)^1.605", H4.sdi_t(T_, 1.0, D), (D / 25.4)^1.605))
    m[nm] = max(get(m, nm, 0.0), reld(a, b))
  end
  for x in vcat(rand(rng, 20000), [0.0, 0.9999, 1.0]); m["x^θ"] = max(get(m, "x^θ", 0.0), abs(H4.hyd_t(T_, G, x) - x^G.θ)); end
  check("2.2 lookup tables vs exact functions (D ∈ [2.54, 260] cm; x ∈ [0, 1])", all(v < 1e-4 for v in values(m)), join(["$k $(fmt(v))" for (k, v) in m], ", ") * " (x^θ absolute)")
  Sa = subS(ST, SUB); Sb = subS(ST, SUB)
  H4.simulate!(Sa, P, G; exact=true); H4.simulate!(Sb, P, G)
  dN = maximum(reld(sum(Sa.N[Sa.off[s]:Sa.off[s+1]-1]), sum(Sb.N[Sb.off[s]:Sb.off[s+1]-1])) for s in 1:H4.nsites(Sa))
  dB = maximum(reld(sum(Sa.B[Sa.off[s]:Sa.off[s+1]-1]), sum(Sb.B[Sb.off[s]:Sb.off[s+1]-1])) for s in 1:H4.nsites(Sa))
  check("2.2b whole intervals with tables vs exact functions: per-site N and B", dN < 1e-4 && dB < 1e-4, "max rel diff N $(fmt(dN)) B $(fmt(dB))")
  H4.simulate!(DT.S, P, G; exact=true); a = nll4(DT, Ob).total; H4.simulate!(DT.S, P, G); b = nll4(DT, Ob).total
  info("2.2c NLL (train, warm start, default observation parameters) with tables vs exact functions", @sprintf("%.3f vs %.3f (Δ %.3g nats)", b, a, b - a))
end
let Sa = subS(ST, SUB), Sb = subS(ST, SUB), Sc = subS(ST, SUB)
  H4.simulate!(Sa, P, G); H4.simulate!(Sb, P, G); serial_run!(Sc, P, G)
  check("2.3 two simulate! runs are bitwise identical", Sa.N == Sb.N && Sa.B == Sb.B && Sa.H == Sb.H && Sa.A == Sb.A)
  check("2.3b threaded simulate! ($(Threads.nthreads()) threads) = serial site loop with one scratch, bitwise", Sa.N == Sc.N && Sa.B == Sc.B && Sa.H == Sc.H && Sa.A == Sc.A)
end
end # section 2

# ================================================================ 3. numerical diffusion vs a tree-level replay of the real v1 trees
L = load4()
if want(3)
section("3. two-moment closure vs TREE-LEVEL replay of the train v1 trees (each tree grows the observed mean annual increment of its key × current class)")
let t1 = v1trees(L, Set(DT.iid)), sidx = Dict(iid => s for (s, iid) in enumerate(DT.iid)), kix = Dict(k => i for (i, k) in enumerate(KEYS)), dtv = Dict(r.iid => r.dt for r in eachrow(DT.iv))
  # observed mean annual diameter increment per (key, v1 class) of train survivors (≥ 0)
  inc = zeros(nk, NCLS); cnt = zeros(nk, NCLS)
  for r in eachrow(t1); r.out == "alive" || continue; j = H4.dclass(r.dia_cm); j == 0 && continue; k = kix[r.key]; inc[k, j] += (r.dia2_cm - r.dia_cm) / dtv[r.iid]; cnt[k, j] += 1; end
  for k in 1:nk, j in 1:NCLS; inc[k, j] = cnt[k, j] >= 20 ? max(inc[k, j] / cnt[k, j], 0.0) : max(sum(inc[:, j]) / max(sum(cnt[:, j]), 1), 0.0); end
  info("3.0 observed mean annual increment (train survivors, all keys) by class [cm/yr]", join([@sprintf("%d:%.3f", j, sum(inc[:, j] .* cnt[:, j]) / max(sum(cnt[:, j]), 1)) for j in 1:NCLS], " "))
  cells = Dict{Tuple{Int,Int},Vector{Float64}}(); trees = Tuple{Int,Int,Float64,Float64}[]
  for r in eachrow(t1)
    j = H4.dclass(r.dia_cm); j == 0 && continue; s = sidx[r.iid]; k = kix[r.key]; w = r.w_ha / (ismicro(r.w_ha) ? DT.fm1[s] : DT.f1[s])
    push!(trees, (s, k, r.dia_cm, w)); v = get!(cells, (s, k), zeros(2NCLS)); v[j] += w; v[NCLS+j] += w * π / 4e4 * r.dia_cm^2
  end
  rows = String[]; worst = 0.0; worstB = 0.0
  for yrs in (5, 10, 25)
    # tree level
    Nt = zeros(NCLS); Bt = 0.0; up_t = zeros(NCLS)
    for (s, k, D, w) in trees
      d = D; for _ in 1:yrs; d += inc[k, H4.dclass(d)]; end
      jn = H4.dclass(d); Nt[jn] += w; Bt += w * π / 4e4 * d^2
    end
    # two-moment (kernel's upgrowth!), and a uniform-density (one-moment) transfer for comparison
    Nm = zeros(NCLS); Bm = 0.0; Nu = zeros(NCLS)
    for ((s, k), v) in cells
      N = v[1:NCLS]; B = v[NCLS+1:end]; Uu = copy(N)
      for _ in 1:yrs
        inN = zeros(NCLS); inB = zeros(NCLS); inU = zeros(NCLS)
        for j in 1:NCLS
          dD = inc[k, j]
          if N[j] > 0
            if j < NCLS && dD > 0
              H4.upgrowth!((t, Nn, Bn, Dn) -> (inN[t] += Nn; inB[t] += Bn), j, N[j], B[j], dD)
            else
              inN[j] += N[j]; inB[j] += j == NCLS ? N[j] * π / 4e4 * (H4.qmd(N[j], B[j]) + dD)^2 : B[j]
            end
          end
          if Uu[j] > 0
            if j < NCLS; f = min(dD / H4.CLS_W, 1.0); inU[j] += Uu[j] * (1 - f); inU[j+1] += Uu[j] * f; else; inU[j] += Uu[j]; end
          end
        end
        N = inN; B = inB; Uu = inU
      end
      Nm .+= N; Bm += sum(B); Nu .+= Uu
    end
    N0 = zeros(NCLS); for (s, k, D, w) in trees; N0[H4.dclass(D)] += w; end
    for b in (3, 4, 6, 10)
      ct = sum(Nt[b:end]) - sum(N0[b:end]); cm = sum(Nm[b:end]) - sum(N0[b:end]); cu = sum(Nu[b:end]) - sum(N0[b:end])
      e = cm / ct - 1; yrs <= 10 && (worst = max(worst, abs(e)))
      push!(rows, @sprintf("%2d yr: crossing into ≥ class %2d (%.2f cm): tree-level %9.0f, two-moment %9.0f (%+.1f %%), uniform one-moment %9.0f (%+.1f %%)  [trees/ha summed over sites]",
                           yrs, b, H4.CLS_LO[b], ct, cm, 100e, cu, 100 * (cu / ct - 1)))
    end
    tv = 0.5 * sum(abs.(Nm .- Nt)) / sum(Nt); worstB = max(worstB, abs(Bm / Bt - 1))
    push!(rows, @sprintf("%2d yr: class-distribution total variation (pooled) %.4f; basal area two-moment / tree-level − 1 = %+.3f %%", yrs, tv, 100 * (Bm / Bt - 1)))
  end
  # mechanism: the same replay with each (key) pooled over ALL sites into one cohort per class (many trees: smooth within-class densities)
  let pooled = Dict{Int,Vector{Float64}}(), Np = zeros(NCLS), Nt5 = zeros(NCLS), N0 = zeros(NCLS)
    for (s, k, D, w) in trees; v = get!(pooled, k, zeros(2NCLS)); j = H4.dclass(D); v[j] += w; v[NCLS+j] += w * π / 4e4 * D^2; N0[j] += w
      d = D; for _ in 1:5; d += inc[k, H4.dclass(d)]; end; Nt5[H4.dclass(d)] += w; end
    for (k, v) in pooled
      N = v[1:NCLS]; B = v[NCLS+1:end]
      for _ in 1:5
        inN = zeros(NCLS); inB = zeros(NCLS)
        for j in 1:NCLS
          N[j] > 0 || continue; dD = inc[k, j]
          j < NCLS && dD > 0 ? H4.upgrowth!((t, Nn, Bn, Dn) -> (inN[t] += Nn; inB[t] += Bn), j, N[j], B[j], dD) : (inN[j] += N[j]; inB[j] += B[j])
        end
        N = inN; B = inB
      end
      Np .+= N
    end
    push!(rows, "pooled per key over all sites, 5 yr: crossing into ≥ class 3 / 4 / 6 / 10: " * join([@sprintf("%+.1f %%", 100 * ((sum(Np[b:end]) - sum(N0[b:end])) / (sum(Nt5[b:end]) - sum(N0[b:end])) - 1)) for b in (3, 4, 6, 10)], " / ") *
               " vs tree level (the per-site bias comes from cells of a few trees: a point mass at r is spread over the class)")
    ncell = count(>(0), [v[j] for v in values(cells) for j in 1:NCLS]); nfew = 0
    for ((s, k), v) in cells, j in 3:NCLS; v[j] > 0 && v[j] <= 3 * 14.88 / DT.f1[s] && (nfew += 1); end
    push!(rows, @sprintf("v1 cells (site × key × class): %d; subplot cells holding ≤ 3 tallied trees: %d", ncell, nfew))
  end
  for r in rows; println("      ", r); end
  check("3.1 [spec §2.3] two-moment transfer vs tree-level replay of the real train v1 trees: flux into ≥ 12.7 / 17.8 / 27.9 / 48.3 cm within 5 % over 5 and 10 years, BA within 0.5 %",
        worst < 0.05 && worstB < 0.005, @sprintf("worst flux error %.1f %%, worst BA error %.3f %% (lines above; 25 yr shown, not scored)", 100worst, 100worstB))
end
end # section 3

# ================================================================ 4. process semantics
H4.simulate!(ST, P, G)     # warm-start state of the train set (used below)
if want(4)
section("4. process semantics (warm start, train)")
let nfin = 0, nout = 0, nlive = 0, nh = 0, outt = 0.0, allt = 0.0
  for i in eachindex(ST.N)
    isfinite(ST.N[i]) && isfinite(ST.B[i]) && isfinite(ST.H[i]) && isfinite(ST.A[i]) || (nfin += 1)
    ST.N[i] > 1e-6 || continue; nlive += 1; allt += ST.N[i]; j = ST.cl[i]; q = H4.qmd(ST.N[i], ST.B[i])
    (H4.CLS_LO[j] - 1e-6 <= q <= H4.CLS_HI[j] + 1e-6) || (nout += 1; outt += ST.N[i]); ST.H[i] >= 1.37 || (nh += 1)
  end
  check("4.0 warm-start run (all train sites): state finite, every cohort's QMD in its class, every height ≥ 1.37 m", nfin == 0 && nout == 0 && nh == 0,
        "non-finite $(nfin); QMD outside $(nout) of $(nlive) live cells; heights < 1.37 m $(nh)")
end
let Pz = [withP(p; lnF=-Inf, longevity=1e12) for p in P], Gz = withG(G; m0=0.0, ms=0.0, mA=1e4, κ=0.0, sdimax=1e15)
  Sx = onesite([(1, 3, 1, 200.0, 16.8, 14.0, 30.0)]; nkk=1); X = H4.Scratch(length(Sx.N))
  PP = [withP(Pz[1]; lnε=Pz[1].lnε + 1.5)]; H4.step_site!(Sx, PP, Gz, X, 1, 1.0, emptytabs(1)); i = ci(1, 3, 1)
  hs = Sx.H[i]; hm = Sx.H[i+NL]; nm = Sx.N[i+NL]; Hn = 14.0 + X.dH[i]
  check("4.1 [spec §2.4] movers taller than stayers (allometric offset), Σ N·H = N (H + dH) conserved by the rescaling", nm > 0 && hm > hs && isapprox(Sx.N[i] * hs + nm * hm, 200 * Hn; rtol=1e-12),
        @sprintf("movers %.2f/ha H %.3f m, stayers %.2f/ha H %.3f m; Σ N·H %.6f vs %.6f", nm, hm, Sx.N[i], hs, Sx.N[i] * hs + nm * hm, 200Hn))
  # 4.1b E[D] ≈ QMD in D_all of the height offset vs the closure's E[D] (the B growth uses the closure's)
  Sv = subS(ST, SUB[1:min(60, end)]); r = Sv.off[1]:Sv.off[2]-1; mxh = 0.0
  for s in 1:H4.nsites(Sv)
    r = Sv.off[s]:Sv.off[s+1]-1; N = Sv.N0[r]; B = Sv.B0[r]; Hh = Sv.H0[r]; A = Sv.A0[r]; N2 = copy(N); B2 = copy(B); H2 = copy(Hh); A2 = copy(A)
    Ref4.step!(N, B, Hh, A, Sv.sp[r], Sv.cl[r], Sv.ln[r], nk, P, G, Sv.W[s], Sv.Wn[s], Sv.T[s], Sv.C[s], Sv.planted[s], 1.0, lz, lw, Sv.seed0[:, s], kafun; dall=:qmd)
    Ref4.step!(N2, B2, H2, A2, Sv.sp[r], Sv.cl[r], Sv.ln[r], nk, P, G, Sv.W[s], Sv.Wn[s], Sv.T[s], Sv.C[s], Sv.planted[s], 1.0, lz, lw, Sv.seed0[:, s], kafun; dall=:exact)
    for q in eachindex(N); N[q] > 1e-6 && (mxh = max(mxh, abs(Hh[q] - H2[q]))); end
  end
  info("4.1b [spec §2.4] height offset uses D_all from E[D] ≈ QMD while the basal area uses the closure's E[D]", "max |ΔH| from the inconsistency after one step: $(fmt(mxh)) m (60 sites; negligible)")
end
let nd = 0, nd0 = 0, nreal = 0
  for s in 1:ns, k in 1:nk
    n3 = sum(ST.N[cellix(ST, s, k, j, v)] for j in 3:NCLS for v in 1:NL)
    n3 > 1e-9 || continue
    n3 < H4.SEED_MIN ? (nd += 1; ST.seed0[k, s] || (nd0 += 1)) : (nreal += 1)
  end
  info("4.2 [spec §2.7] seed sources at v2 (warm start)", "key×plot with ≥ 1 tree/ha ≥ 12.7 cm: $(nreal); with 0 < N < 1 tree/ha (closure tails, NOT seed sources unless a v1 seed tree): $(nd), of which $(nd0) without a v1 seed tree")
  check("4.2b [spec §2.7] the seed rule uses the v1 seed trees of the plot (seed0 = a key tree ≥ 12.7 cm at v1, also when the v1 tree later died)",
        all(ST.seed0[k, s] == any(ST.N0[cellix(ST, s, k, j, v)] > 0 for j in 3:NCLS for v in 1:NL) for s in 1:ns, k in 1:nk),
        "seed0 = (any v1 cell of the key in classes ≥ 3 occupied)")
end
# 4.3 temperature response: f_T = 1 at T_ref; slope at T_ref; the share of T_ref − T_opt that is the period-vs-normal offset
let cl = CSV.read(CLIM_CSV, DataFrame; select=[:iid, :tmean_iv, :tmean_n]), tiv = Dict(r.iid => r.tmean_iv for r in eachrow(cl)), tn = Dict(r.iid => r.tmean_n for r in eachrow(cl))
  off = mean(tiv[i] - tn[i] for i in DT.iid if haskey(tiv, i)); slopes = [-2 * (TR.Tref[k] - TR.Topt[k]) / G.wT^2 for k in 1:nk]
  okc = all(abs(H4.fT_rel(withP(P[k]; Tref=TR.Tref[k]), G, TR.Tref[k]) - 1) < 1e-14 for k in 1:nk)
  fT = [H4.fT_rel(P[k], G, ST.T[s]) for s in 1:ns, k in 1:nk]
  info("4.3 [phys] centred temperature response (wT = $(G.wT))", "f_T(T_ref) = 1: $(okc) | d ln f_T / dT at T_ref [%/°C]: " *
       join(["$(KEYS[k]) $(round(100 * slopes[k]; digits=1))" for k in 1:nk], ", ") * @sprintf(" | f_T over train plots: %.2f–%.2f | mean period − normal temperature of the train intervals %+.3f °C", minimum(fT), maximum(fT), off))
  check("4.3b [phys] the temperature response can represent BOTH signs of a growth–temperature gradient at a key's typical climate (some key with T_opt > T_ref)", any(TR.Topt .> TR.Tref),
        "T_ref − T_opt > 0 for all $(nk) keys (" * join(["$(KEYS[k]) $(round(TR.Tref[k] - TR.Topt[k]; digits=2))" for k in 1:nk], ", ") * @sprintf(" °C): warmer-than-typical sites ALWAYS grow less, for every wT; for _H/_S the gap is ≈ the period-vs-normal offset (%+.2f °C), not ecology; T_opt of a named species = its Region-4 centroid, the warm half of most ranges", off))
end
let co = [CS.sets[nm].S.C for nm in ("train", "val", "test")], allc = reduce(vcat, co), fC = 1 .+ G.βC .* log.(allc ./ H4.CREF)
  yr = [year(r.d1) + r.dt / 2 for r in eachrow(DT.iv)]
  info("4.4 [phys] CO2 term (βC fixed $(G.βC))", @sprintf("period CO2 %.1f–%.1f ppm → f_CO2 %.3f–%.3f (train mean %.4f); corr(C, interval mid-year) %.3f: CO2 is a deterministic time trend; it multiplies cw, hence also gm (survival) and the self-thinning weights — FACE NPP constrains growth only",
       minimum(allc), maximum(allc), minimum(fC), maximum(fC), mean(1 .+ G.βC .* log.(ST.C ./ H4.CREF)), cor(ST.C, yr)))
end
# 4.5 self-thinning of trees that moved into a cell empty at the start of the step: g of the cell = 1 → weight 1e-9 (immune)
let Pz = withP(Ph; lnε=log(4.0)), Gt = withG(Gh; sdimax=300.0, a=0.0, am=0.0)     # light off: growth (hence the movers) does not depend on sdimax; g = cw = 0.9 for every tree
  S0 = onesite([(1, 3, 1, 800.0, 16.5, 14.0, 30.0)]); H4.simulate!(S0, [Pz], Gt; exact=true)
  n3 = S0.N[ci(1, 3, 1)]; n4 = S0.N[ci(1, 4, 1)]
  Sg = onesite([(1, 3, 1, 800.0, 16.5, 14.0, 30.0)]); H4.simulate!(Sg, [Pz], withG(Gt; sdimax=1e15); exact=true)
  g3 = Sg.N[ci(1, 3, 1)]; g4 = Sg.N[ci(1, 4, 1)]
  check("4.5 [spec §2.5] self-thinning removes the excess SDI ∝ SDI (1 − g) of every tree: trees that moved into a class empty at the start keep their source cohort's g",
        abs(n4 / g4 - n3 / g3) < 0.01, @sprintf("one cohort, sdimax 300, unthinned movers %.1f + stayers %.1f /ha: stayers keep %.1f %%, movers into the (initially empty) class 4 keep %.4f %% (g = 1 of an empty cell → weight 1e-9: immune; HBAD4P.jl:210, 255)", g4, g3, 100n3 / g3, 100n4 / g4))
end
let t1 = v1trees(L, Set(DT.iid)), d = t1.dia_cm[H4.dclass.(t1.dia_cm) .== NCLS], w = t1.w_ha[H4.dclass.(t1.dia_cm) .== NCLS]
  ED = sum(w .* d) / sum(w); Q = sqrt(sum(w .* d .^ 2) / sum(w))
  info("4.6 [spec, known] open last class: B growth uses E[D] ≈ QMD", @sprintf("train v1 trees ≥ 78.74 cm: %d, E[D] %.2f cm vs QMD %.2f cm → B growth of class 16 overstated by %.2f %% of its 2 dD E[D] term", length(d), ED, Q, 100 * (Q / ED - 1)))
end
end # section 4

# ================================================================ 5. data pipeline vs SQL on the raw databases
if want(5) || want(6) || want(9)
con = DBInterface.connect(DuckDB.DB, ":memory:")
DBInterface.execute(con, "ATTACH '$(SLIM_DB)' AS s (READ_ONLY)"); DBInterface.execute(con, "ATTACH '$(FIADB)' AS f (READ_ONLY)")
q(sql) = DataFrame(DBInterface.execute(con, sql))
ivall = vcat([CS.sets[nm].iv for nm in ("train", "val", "test")]...)
DuckDB.register_data_frame(con, DataFrame(iid=ivall.iid, st=ivall.st, u=ivall.u, c=ivall.c, p=ivall.p, d1=Date.(ivall.d1), d2=Date.(ivall.d2), set=String.(ivall.set)), "ivs")
v1 = q("""
  SELECT i.iid, i.set, t.SUBP sb, t.TREE tr, t.DIA dia, t.HT ht, t.TPA_UNADJ tpa, t.estimated_age age, coalesce(r.SPECIES_SYMBOL, 'UNK') sym, coalesce(r.SFTWD_HRDWD, 'H') grp,
         t2.STATUSCD s2, t2.tree_fate fate2, t2.DIA dia2, t2.HT ht2
  FROM ivs i JOIN s.curated_trees t ON t.STATECD = i.st AND t.UNITCD = i.u AND t.COUNTYCD = i.c AND t.PLOT = i.p AND t.MEASDATE = i.d1
  LEFT JOIN s.REF_SPECIES r ON r.SPCD = t.spcd_resolved
  LEFT JOIN s.curated_trees t2 ON t2.STATECD = t.STATECD AND t2.UNITCD = t.UNITCD AND t2.COUNTYCD = t.COUNTYCD AND t2.PLOT = t.PLOT AND t2.SUBP = t.SUBP AND t2.TREE = t.TREE AND t2.MEASDATE = i.d2
  WHERE t.STATUSCD = 1""")
bigset = Set(k for k in KEYS if k[1] != '_'); keyof_(s, g) = s in bigset ? String(s) : "_" * String(g)
v1.key = keyof_.(v1.sym, v1.grp)
FQ = let fsql = q("""
  SELECT p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, make_date(p.MEASYEAR::INT, p.MEASMON::INT, p.MEASDAY::INT) md, sc.SUBP sb,
         sum(sc.SUBPCOND_PROP) FILTER (WHERE k.COND_STATUS_CD = 1) fs, sum(sc.MICRCOND_PROP) FILTER (WHERE k.COND_STATUS_CD = 1) fm,
         sum(sc.SUBPCOND_PROP) ts, sum(sc.MICRCOND_PROP) tm, count(DISTINCT p.CN) ncn
  FROM f.SUBP_COND sc JOIN f.PLOT p ON p.CN = sc.PLT_CN JOIN f.COND k ON k.PLT_CN = sc.PLT_CN AND k.CONDID = sc.CONDID
  WHERE p.STATECD IN (12, 13) AND p.MEASYEAR IS NOT NULL AND p.MEASMON IS NOT NULL AND p.MEASDAY IS NOT NULL GROUP BY ALL""")
  Dict((Int(r.st), Int(r.u), Int(r.c), Int(r.p), Date(r.md), Int(r.sb)) => (r.ncn == 1 && abs(coalesce(r.ts, 0.0) - 1) <= 1e-6 ? clamp(coalesce(r.fs, 0.0), 0, 1) : NaN,
                                                                         r.ncn == 1 && abs(coalesce(r.tm, 0.0) - 1) <= 1e-6 ? clamp(coalesce(r.fm, 0.0), 0, 1) : NaN) for r in eachrow(fsql))
end
pf(v) = (x = [isnan(a) ? 1.0 : a for a in v]; m = mean(x); m <= 0 ? 1.0 : max(m, 0.25))
FF = Dict{String,Matrix{Float64}}()
for nm in ("train", "val", "test")
  Dn = CS.sets[nm]; f = zeros(length(Dn.iid), 4)
  for (s, r) in enumerate(eachrow(Dn.iv))
    a1 = [get(FQ, (r.st, r.u, r.c, r.p, Date(r.d1), sb), (NaN, NaN)) for sb in 1:4]; a2 = [get(FQ, (r.st, r.u, r.c, r.p, Date(r.d2), sb), (NaN, NaN)) for sb in 1:4]
    f[s, :] = [pf(first.(a1)), pf(last.(a1)), pf(first.(a2)), pf(last.(a2))]
  end
  FF[nm] = f
end
end
if want(5)
section("5. data pipeline vs SQL on the raw databases")
let ivsql = q("""
    WITH vis AS (SELECT STATECD st, UNITCD u, COUNTYCD c, PLOT p, MEASDATE md, bool_or(coalesce(TRTCD1, 0) > 0) trt, bool_or(coalesce(DSTRBCD1, 0) > 0) dst,
                        bool_or(coalesce(tree_fate, '') LIKE 'harvest%') harv, bool_or(coalesce(cond_stdorgcd, 0) = 1) pl
                 FROM s.curated_trees WHERE STATECD IN (12, 13) GROUP BY ALL),
         pr AS (SELECT *, lead(md) OVER w md2, lead(trt) OVER w trt2, lead(dst) OVER w dst2, lead(harv) OVER w harv2 FROM vis WINDOW w AS (PARTITION BY st, u, c, p ORDER BY md))
    SELECT st, u, c, p, md d1, md2 d2, trt2, dst2, harv2 FROM pr WHERE md2 IS NOT NULL AND date_diff('day', md, md2) / 365.25 > 2""")
  spl = CSV.read(SPLIT_PLOTS, DataFrame); setof = Dict((r.statecd, r.unitcd, r.countycd, r.plot) => String(r.set) for r in eachrow(spl))
  nat = ivsql[.!(coalesce.(ivsql.trt2, false) .| coalesce.(ivsql.dst2, false) .| coalesce.(ivsql.harv2, false)), :]
  a0 = Set((Int(r.st), Int(r.u), Int(r.c), Int(r.p), Date(r.d1), Date(r.d2), get(setof, (r.st, r.u, r.c, r.p), "")) for r in eachrow(nat))
  a = Set(x for x in a0 if x[7] != ""); nnoset = length(a0) - length(a)        # plots outside the campaign split (filter_ecos) are in no set
  b = Set((r.st, r.u, r.c, r.p, Date(r.d1), Date(r.d2), String(r.set)) for r in eachrow(ivall))
  oa = setdiff(a, b); ob = setdiff(b, a)
  check("5.1 natural GA+FL intervals (plot, d1, d2, set), dt > 2: SQL on curated_trees (no TRTCD1 / DSTRBCD1 / harvest fate at v2) = the builder's three sets", isempty(oa) && isempty(ob),
        "SQL $(length(a)) (+ $(nnoset) on plots outside the campaign split), builder $(length(b)); only SQL $(length(oa)), only builder $(length(ob))")
  pl = Dict(nm => Set((r.st, r.u, r.c, r.p) for r in eachrow(CS.sets[nm].iv)) for nm in ("train", "val", "test"))
  dis = isempty(intersect(pl["train"], pl["val"])) && isempty(intersect(pl["train"], pl["test"])) && isempty(intersect(pl["val"], pl["test"]))
  check("5.2 train / val / test disjoint at the PLOT level; every site's set = split_plots.csv", dis && all(get(setof, x, "") == nm for nm in keys(pl) for x in pl[nm]),
        join(["$k $(length(v)) plots / $(H4.nsites(CS.sets[k].S)) intervals" for (k, v) in pl], ", "))
  if isfile(ROLE_SPLIT)
    rs = CSV.read(ROLE_SPLIT, DataFrame); rof = Dict((r.statecd, r.unitcd, r.countycd, r.plot) => String(r.set) for r in eachrow(rs))
    tab = Dict{Tuple{String,String},Int}(); for nm in keys(pl), x in pl[nm]; kk = (get(rof, x, "NA"), nm); tab[kk] = get(tab, kk, 0) + 1; end
    leak = get(tab, ("test", "train"), 0) + get(tab, ("test", "val"), 0) + get(tab, ("train", "test"), 0) + get(tab, ("val", "test"), 0)
    check("5.2b no plot of the role's TEST split in train / val and vice versa", leak == 0, "role split → HBAD4P set: " * join(["$(x)→$(y) $(n)" for ((x, y), n) in sort(collect(tab))], ", "))
    if isfile(FOLDS_CSV)
      fo = CSV.read(FOLDS_CSV, DataFrame); fd = Dict((r.statecd, r.unitcd, r.countycd, r.plot) => r.fold for r in eachrow(fo))
      tf = unique(get(fd, x, -1) for x in pl["test"]); ot = unique(get(fd, x, -1) for nm in ("train", "val") for x in pl[nm])
      check("5.2c test plots = one outer fold, never in train / val", length(tf) == 1 && !(tf[1] in ot), "test folds $(tf), train/val folds $(sort(ot))")
    end
  end
end
let c = combine(groupby(v1[(v1.set .== "train") .& (coalesce.(v1.age, 0.0) .> 0) .& (coalesce.(v1.ht, 0.0) .> 0) .& (coalesce.(v1.dia, 0.0) .> 0), :], :sym), nrow => :n)
  bigk = sort(String.(c.sym[c.n .>= MIN_SP_TREES])); sqlkeys = sort(unique(vcat(bigk, ["_" * String(g) for g in unique(v1.grp)])))
  check("5.3 species keys from SQL (train live v1 trees with age and height, ≥ $(MIN_SP_TREES)) = the builder's keys = the HBAD3P fit's keys", sqlkeys == KEYS &&
        String.(CSV.read(joinpath(FIT3P, "species.csv"), DataFrame).key) == KEYS, "SQL $(join(sqlkeys, ",")) | builder $(join(KEYS, ","))")
end
let mx = 0.0, nunk = 0, nall = 0, nmin = 0
  for nm in ("train", "val", "test"), (s, r) in enumerate(eachrow(CS.sets[nm].iv))
    Dn = CS.sets[nm]; mx = max(mx, maximum(abs.(FF[nm][s, :] .- [Dn.f1[s], Dn.fm1[s], Dn.fs2[s], Dn.fm2[s]])))
    for d in (r.d1, r.d2), sb in 1:4; nall += 1; isnan(get(FQ, (r.st, r.u, r.c, r.p, Date(d), sb), (NaN, NaN))[1]) && (nunk += 1); end
    minimum(FF[nm][s, :]) <= 0.25 + 1e-12 && (nmin += 1)
  end
  check("5.4 forested fractions f1, fm1, fs2, fm2: FIADB SUBP_COND × COND (independent query) = the builder", mx < 1e-9,
        "max abs diff $(fmt(mx)) | subplot-visits with unknown status (→ forested) $(nunk) of $(nall) | sites with a fraction at the FMIN floor 0.25: $(nmin)")
end
# 5.5 v1 stand table, classes ≥ 3 (classes 1–2 are the empirical-Bayes layer: section 6)
let mx = Dict(:N => 0.0, :B => 0.0, :H => 0.0, :A => 0.0), nbad = 0, ncell = 0, worst = "", nmiss = 0, nharv = 0, nnoage = 0
  trn = v1[(v1.set .== "train") .& (coalesce.(v1.age, 0.0) .> 0), :]; medage = Dict(k => median(trn.age[trn.key .== k]) for k in KEYS)
  for nm in ("train", "val", "test")
    Dn = CS.sets[nm]; S = Dn.S; sid = Dict(iid => s for (s, iid) in enumerate(Dn.iid)); kix = Dict(k => i for (i, k) in enumerate(KEYS))
    acc = Dict{Tuple{Int,Int,Int},Vector{Float64}}()
    for r in eachrow(v1[v1.set .== nm, :])
      ismissing(r.s2) && (nmiss += 1; continue); startswith(coalesce(r.fate2, ""), "harvest") && (nharv += 1; continue)
      DIA = coalesce(r.dia, 0.0); DIA >= 5.0 || continue
      j = min(NCLS, floor(Int, (DIA + 1) / 2)); s = sid[r.iid]; k = kix[r.key]; w = r.tpa * 2.4710538 / FF[nm][s, 1]
      a = get!(acc, (s, k, j), zeros(6)); a[1] += w; a[2] += w * π / 4e4 * (DIA * 2.54)^2
      h = coalesce(r.ht, 0.0); h > 0 && (a[3] += w * h * 0.3048; a[4] += w); ag = coalesce(r.age, 0.0); ag > 0 ? (a[5] += w * ag; a[6] += w) : (nnoage += 1)
    end
    for s in 1:H4.nsites(S), k in 1:nk, j in 3:NCLS
      i = cellix(S, s, k, j, 1); N = S.N0[i] + S.N0[i+1] + S.N0[i+2]; a = get(acc, (s, k, j), nothing)
      (N == 0 && a === nothing) && continue; ncell += 1
      (a === nothing || N == 0) && (nbad += 1; worst = "$(nm) site $s $(KEYS[k]) class $j: builder N $N SQL $(a === nothing ? 0 : a[1])"; continue)
      B = S.B0[i] + S.B0[i+1] + S.B0[i+2]; Aref = a[6] > 0 ? a[5] / a[6] : medage[KEYS[k]]
      d = (N=reld(N, a[1]), B=reld(B, a[2]), H=a[4] > 0 ? reld(S.H0[i], a[3] / a[4]) : 0.0, A=reld(S.A0[i], Aref))
      for (kk, v) in pairs(d); mx[kk] = max(mx[kk], v); end
      (max(d.N, d.B) > 1e-5 || max(d.H, d.A) > 1e-9) && (nbad += 1; worst = "$(nm) site $s $(KEYS[k]) class $j: builder N $(round(N; digits=3)) A $(S.A0[i]) | SQL N $(round(a[1]; digits=3)) A $(Aref)")
    end
  end
  check("5.5 v1 stand table classes ≥ 3 (all sets): SQL tree list (live at v1, not missing / harvested at v2; TPA·2.4710538 / forested fraction) = the builder per (site, key, class): N, B (1e-5: the product's w_ha 14.870893 vs 6.018046·2.4710538 = 14.870870), H, A (un-aged → the key's TRAIN median age)",
        nbad == 0, "$(ncell) cells, $(nbad) differ; max rel diff " * join(["$k $(fmt(v))" for (k, v) in mx], ", ") * (nbad > 0 ? " | e.g. $worst" : "") * " | removed: $(nmiss) missing at v2, $(nharv) harvested; un-aged v1 trees $(nnoage)")
end
# 5.6 v2 stand table = all live trees of the plot at v2 (FIA STATUSCD 1), class of the v2 DIA; a tree dead at v1 and alive at v2 is in neither table (product)
let v2 = q("""
    SELECT i.iid, i.set, t.SUBP sb, t.TREE tr, t.DIA dia, t.HT ht, t.TPA_UNADJ tpa, coalesce(r.SPECIES_SYMBOL, 'UNK') sym, coalesce(r.SFTWD_HRDWD, 'H') grp, t1.STATUSCD s1, t1.DIA dia1
    FROM ivs i JOIN s.curated_trees t ON t.STATECD = i.st AND t.UNITCD = i.u AND t.COUNTYCD = i.c AND t.PLOT = i.p AND t.MEASDATE = i.d2
    LEFT JOIN s.REF_SPECIES r ON r.SPCD = t.spcd_resolved
    LEFT JOIN s.curated_trees t1 ON t1.STATECD = t.STATECD AND t1.UNITCD = t.UNITCD AND t1.COUNTYCD = t.COUNTYCD AND t1.PLOT = t.PLOT AND t1.SUBP = t.SUBP AND t1.TREE = t.TREE AND t1.MEASDATE = i.d1
    WHERE t.STATUSCD = 1"""), nres = 0, ntally = 0, ex = String[], ncell = 0, nbad = 0, mxD = 0.0, mxH = 0.0, worst = "", miss = 0, extra = 0
  v2.key = keyof_.(v2.sym, v2.grp)
  for nm in ("train", "val", "test")
    Dn = CS.sets[nm]; sid = Dict(iid => s for (s, iid) in enumerate(Dn.iid)); kix = Dict(k => i for (i, k) in enumerate(KEYS)); nsn = length(Dn.iid)
    n = zeros(Int, nk, NCLS, nsn); d2 = zeros(nk, NCLS, nsn); hs = zeros(nk, NCLS, nsn); nh = zeros(Int, nk, NCLS, nsn)
    for r in eachrow(v2[v2.set .== nm, :])
      (!ismissing(r.s1) && r.s1 != 1) && (nres += 1; continue)
      DIA = coalesce(r.dia, 0.0); DIA >= 1.0 || continue; j = min(NCLS, floor(Int, (DIA + 1) / 2))
      tp = coalesce(r.tpa, NaN); (j >= 3 ? abs(tp - 6.018046) < 1e-3 : abs(tp - 74.965282) < 1e-3) || (ntally += 1; length(ex) < 3 && push!(ex, "DIA $(r.dia) in (v1 $(coalesce(r.dia1, NaN))) TPA $(r.tpa)"))
      s = sid[r.iid]; k = kix[r.key]; n[k, j, s] += 1; d2[k, j, s] += (DIA * 2.54)^2; h = coalesce(r.ht, 0.0); h > 0 && (hs[k, j, s] += h * 0.3048; nh[k, j, s] += 1)
    end
    for s in 1:nsn, k in 1:nk, j in 1:NCLS
      (n[k, j, s] == 0 && Dn.n[k, j, s] == 0) && continue; ncell += 1
      if n[k, j, s] != Dn.n[k, j, s] || nh[k, j, s] != Dn.nH[k, j, s]
        nbad += 1; miss += max(n[k, j, s] - Dn.n[k, j, s], 0); extra += max(Dn.n[k, j, s] - n[k, j, s], 0); worst = "$(nm) site $s $(KEYS[k]) class $j: builder $(Dn.n[k, j, s]) SQL $(n[k, j, s])"
      else
        mxD = max(mxD, reld(d2[k, j, s], Dn.D2[k, j, s])); mxH = max(mxH, reld(hs[k, j, s], Dn.Hs[k, j, s]))
      end
    end
  end
  check("5.6 v2 stand table (all sets) = all live trees at v2 in curated_trees (dead-at-v1 excluded), per (site, key, class): n, n with a height, Σ D², Σ H", nbad == 0 && mxD < 1e-9 && mxH < 1e-9,
        "$(ncell) cells, $(nbad) differ ($(miss) trees missing from the builder, $(extra) extra) | equal cells max rel diff ΣD² $(fmt(mxD)) ΣH $(fmt(mxH)) | live at v2 but DEAD at v1: $(nres)" * (nbad > 0 ? " | e.g. $worst" : ""))
  check("5.6b FIA design: every live v2 tree's tally (TPA_UNADJ 6.018 subplot / 74.965 microplot) = the tally area its class is scored on (classes ≥ 3 subplot)", ntally == 0,
        "$(ntally) trees on another tally" * (isempty(ex) ? "" : ": " * join(ex, "; ")))
end
let fsum = q("""
    SELECT p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, make_date(p.MEASYEAR::INT, p.MEASMON::INT, p.MEASDAY::INT) md, count(*) n, count(*) FILTER (WHERE t.DIA >= 5) n5
    FROM f.TREE t JOIN f.PLOT p ON p.CN = t.PLT_CN WHERE p.STATECD IN (12, 13) AND t.STATUSCD = 1 AND t.DIA >= 1.0 AND p.MEASYEAR IS NOT NULL AND p.MEASMON IS NOT NULL AND p.MEASDAY IS NOT NULL GROUP BY ALL""")
  fd = Dict((Int(r.st), Int(r.u), Int(r.c), Int(r.p), Date(r.md)) => (r.n, r.n5) for r in eachrow(fsum)); nd = 0; nd5 = 0; tot = 0; dif = 0; dif5 = 0; nnot = 0
  for nm in ("train", "val", "test")
    Dn = CS.sets[nm]
    for (s, r) in enumerate(eachrow(Dn.iv))
      key = (r.st, r.u, r.c, r.p, Date(r.d2)); haskey(fd, key) || (nnot += 1; continue)
      b = sum(Dn.n[:, :, s]); b5 = sum(Dn.n[:, 3:end, s]); a, a5 = fd[key]; tot += a
      a != b && (nd += 1; dif += a - b); a5 != b5 && (nd5 += 1; dif5 += a5 - b5)
    end
  end
  info("5.7 completeness of the v2 table vs FIADB TREE (live, DIA ≥ 1 in, per plot visit)", "plot visits not found in FIADB $(nnot); visits with a different count: all trees $(nd) (FIADB − builder Σ = $(dif) of $(tot)), ≥ 5 in $(nd5) (Σ $(dif5)) — includes the dead-at-v1 / alive-at-v2 trees and curated species / record fixes")
end
let ws = 6.018046 * 2.4710538, wm = 74.965282 * 2.4710538
  check("5.8 tally areas = inverse FIA expansion factors (w_ha × area = 1) and the 4-subplot / 4-microplot circle areas (24 ft, 6.8 ft radius)",
        abs(ws * AREA_SUB - 1) < 1e-12 && abs(wm * AREA_MIC - 1) < 1e-12 && abs(AREA_SUB / (4π * (24 * 0.3048)^2 / 1e4) - 1) < 3e-3 && abs(AREA_MIC / (4π * (6.8 * 0.3048)^2 / 1e4) - 1) < 3e-3,
        @sprintf("subplot %.6f ha (circles %.6f), microplot %.7f ha (circles %.7f)", AREA_SUB, 4π * (24 * 0.3048)^2 / 1e4, AREA_MIC, 4π * (6.8 * 0.3048)^2 / 1e4))
end
let cl = CSV.read(CLIM_CSV, DataFrame; select=[:iid, :st, :u, :c, :p, :tmean_iv]), wa = CSV.read(WATER_CSV, DataFrame), ivb = DataFrame(Arrow.Table(joinpath(DATA_DIR, "intervals.arrow")))
  cp = Dict(r.iid => (r.st, r.u, r.c, r.p) for r in eachrow(cl)); ct = Dict(r.iid => r.tmean_iv for r in eachrow(cl)); wd = Dict(r.iid => (r.aetpet_iv, r.aetpet_n) for r in eachrow(wa))
  nbad = 0; nmis = 0; mx = 0.0
  for nm in ("train", "val", "test"), (s, r) in enumerate(eachrow(CS.sets[nm].iv))
    get(cp, r.iid, nothing) == (r.st, r.u, r.c, r.p) || (nbad += 1)
    (haskey(wd, r.iid) && haskey(ct, r.iid) && !ismissing(wd[r.iid][1]) && !ismissing(ct[r.iid])) || (nmis += 1; continue)
    S = CS.sets[nm].S; mx = max(mx, abs(S.W[s] - wd[r.iid][1]), abs(S.Wn[s] - wd[r.iid][2]), abs(S.T[s] - ct[r.iid]))
  end
  check("5.9 climate: every site's iid maps to its own plot in the climate table (the tables were keyed on the base product's iid); W, Wn, T = the tables; none missing", nbad == 0 && nmis == 0 && mx < 1e-12,
        "plot mismatches $(nbad), missing $(nmis), max abs diff $(fmt(mx))")
  co = CSV.read(CO2_CSV, DataFrame); cd = Dict(Int(r.year) => Float64(r.co2_ppm) for r in eachrow(co)); mxc = 0.0
  for nm in ("train", "val", "test"), (s, r) in enumerate(eachrow(CS.sets[nm].iv))
    mxc = max(mxc, abs(CS.sets[nm].S.C[s] - mean(cd[y] for y in year(r.d1):max(year(r.d2) - 1, year(r.d1)))))
  end
  check("5.9b CO2 = mean of the annual Mauna Loa series over year(d1) … year(d2) − 1 (the climate period of interval_climate)", mxc < 1e-9, "max abs diff $(fmt(mxc)) ppm")
  dtok = all(abs(CS.sets[nm].S.nyr[s] + CS.sets[nm].S.frac[s] - CS.sets[nm].iv.dt[s]) < 1e-12 && 0 <= CS.sets[nm].S.frac[s] < 1 for nm in ("train", "val", "test") for s in 1:H4.nsites(CS.sets[nm].S))
  check("5.9c simulated span nyr + frac = dt of the interval (dates)", dtok)
end
let t = v1[(v1.set .== "train"), :], st_ = CSV.read(SHADETOL_CSV, DataFrame), lo = CSV.read(LONGEVITY_CSV, DataFrame), cl = CSV.read(CLIM_CSV, DataFrame; select=[:iid, :tmean_iv, :tmean_n])
  std_ = Dict(String(r.sym) => Float64(r.shade_class) for r in eachrow(st_)); lod = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lo))
  tn = Dict(r.iid => r.tmean_n for r in eachrow(cl)); tiv = Dict(r.iid => r.tmean_iv for r in eachrow(cl))
  ivb = DataFrame(Arrow.Table(joinpath(DATA_DIR, "intervals.arrow"))); okT = Set(ivb.iid[.!inregion.(ivb.st) .| (coalesce.(ivb.set, "") .== "train")])
  tall = DataFrame(Arrow.Table(joinpath(DATA_DIR, "interval_trees.arrow"))); tall = tall[in.(tall.iid, Ref(okT)) .& haskey.(Ref(tn), tall.iid), :]
  mx = Dict(:Hmax => 0.0, :ST => 0.0, :longevity => 0.0, :Topt => 0.0, :Tref => 0.0); lw_ = Float64[]
  for (k, key) in enumerate(KEYS)
    tk = t[(t.key .== key) .& (coalesce.(t.dia, 0.0) .> 0), :]; hs = 0.3048 .* Float64.(collect(skipmissing(tk.ht[coalesce.(tk.ht, 0.0) .> 0])))
    c = combine(groupby(tk, :sym), nrow => :n)
    mx[:Hmax] = max(mx[:Hmax], reld(quantile(hs, 0.995), TR.Hmax[k])); mx[:ST] = max(mx[:ST], reld(sum(get(std_, s, 3.0) * n for (s, n) in zip(c.sym, c.n)) / sum(c.n), TR.ST[k]))
    mx[:longevity] = max(mx[:longevity], reld(median([get(lod, s, 300.0) for s in c.sym]), TR.longevity[k])); push!(lw_, sum(get(lod, s, 300.0) * n for (s, n) in zip(c.sym, c.n)) / sum(c.n))
    tt = tk[haskey.(Ref(tn), tk.iid), :]
    topt = key[1] == '_' ? mean(tn[i] for i in tt.iid) : mean(tn[i] for i in tall.iid[tall.sym .== key]); mx[:Topt] = max(mx[:Topt], abs(topt - TR.Topt[k]))
    mx[:Tref] = max(mx[:Tref], abs(mean(tiv[i] for i in tt.iid) - TR.Tref[k]))
  end
  check("5.10 traits from the SQL tree list (TRAIN only) = TR: Hmax (99.5 % height), ST (tree-weighted), LONGEVITY (median), T_opt (named: outside-region + region-train trees; groups: region train), T_ref (train period mean)",
        mx[:Hmax] < 1e-9 && mx[:ST] < 1e-9 && mx[:longevity] < 1e-9 && mx[:Topt] < 1e-6 && mx[:Tref] < 1e-6, join(["$k $(fmt(v))" for (k, v) in mx], ", "))
  check("5.10b [spec §1 / data4.jl:47] LONGEVITY tree-weighted over member species as documented", all(abs.(lw_ .- TR.longevity) .< 1e-9),
        "code takes the UNWEIGHTED median over member species: " * join(["$(KEYS[k]) $(TR.longevity[k]) (tree-weighted mean $(round(lw_[k]; digits=0)))" for k in 1:nk if abs(lw_[k] - TR.longevity[k]) > 1e-9], ", "))
end
let f = joinpath(@__DIR__, "data4.jl"), src = read(f), h = hash(src), s2 = copy(src)
  s2[div(length(s2), 3)] = s2[div(length(s2), 3)] == UInt8('1') ? UInt8('2') : UInt8('1')
  srcd = read(f, String); deps = all(occursin(x, srcd[findfirst("function build_key", srcd)[1]:findfirst("function cached_sets", srcd)[1]]) for x in ("HBAD4P.jl", "REGION", "MIN_SP_TREES", "LANEMODE", "FMIN", "WATER_CSV", "CLIM_CSV", "FOREST_CSV", "CO2_CSV", "SHADETOL_CSV", "LONGEVITY_CSV"))
  check("5.11 cache key covers builder + kernel code (a one-byte edit changes it), the switches and every input file", hash(s2) != h && deps,
        "one-byte edit changes hash: $(hash(s2) != h) (Julia $(VERSION): Vector{UInt8} hashing is content-complete; it is not stable across Julia versions); inputs and switches listed: $(deps); traits4 reads DATA_DIR, not the `dir` argument of cached_sets (data4.jl:56-57)")
end
end # section 5

# ================================================================ 6. empirical-Bayes sapling prior
if want(6)
section("6. empirical-Bayes sapling layer (classes 1–2)")
# independent recomputation of the prior from the SQL tree list: groups (key, class, adult), m = Σn/Σe, α maximising the NB marginal (grid + golden refine)
EBR = let trn = ivall[ivall.set .== "train", :], sidx = Dict(iid => s for (s, iid) in enumerate(sort(trn.iid))), kix = Dict(k => i for (i, k) in enumerate(KEYS)), nst = nrow(trn)
  fm = zeros(nst); for r in eachrow(trn); s = sidx[r.iid]; fm[s] = pf([get(FQ, (r.st, r.u, r.c, r.p, Date(r.d1), sb), (NaN, NaN))[2] for sb in 1:4]); end
  n = zeros(Int, nk, 2, nst); adult = falses(nk, nst)
  for r in eachrow(v1[v1.set .== "train", :])
    (ismissing(r.s2) || startswith(coalesce(r.fate2, ""), "harvest")) && continue
    s = sidx[r.iid]; k = kix[r.key]; DIA = coalesce(r.dia, 0.0); DIA >= 5.0 && (adult[k, s] = true)
    (1.0 <= DIA < 5.0) && (n[k, DIA < 3.0 ? 1 : 2, s] += 1)
  end
  e = AREA_MIC .* fm; m = zeros(nk, 2, 2); α = fill(Inf, nk, 2, 2); llk = zeros(nk, 2, 2); llr = zeros(nk, 2, 2); llj = zeros(nk, 2, 2); ng = zeros(Int, nk, 2, 2)
  nbll(y, x, mg, r) = sum(Distributions.logpdf(Distributions.NegativeBinomial(r, r / (r + mg * xi)), yi) for (yi, xi) in zip(y, x))
  for k in 1:nk, j in 1:2, a in (false, true)
    idx = findall(==(a), adult[k, :]); isempty(idx) && continue; y = n[k, j, idx]; x = e[idx]; ng[k, j, a+1] = length(idx)
    mg = sum(y) / sum(x); m[k, j, a+1] = mg; mg > 0 || continue
    grid = range(-8, 8; length=161); v = [nbll(y, x, mg, exp(g)) for g in grid]; ib = argmax(v)
    lo_, hi_ = grid[max(ib - 1, 1)], grid[min(ib + 1, end)]
    for _ in 1:60; a1 = lo_ + 0.382 * (hi_ - lo_); a2 = lo_ + 0.618 * (hi_ - lo_); nbll(y, x, mg, exp(a1)) > nbll(y, x, mg, exp(a2)) ? (hi_ = a2) : (lo_ = a1); end
    α[k, j, a+1] = exp((lo_ + hi_) / 2); llr[k, j, a+1] = nbll(y, x, mg, α[k, j, a+1]); llk[k, j, a+1] = nbll(y, x, mg, CS.eb.α[k, j, a+1])
    best = -Inf; for lm in range(log(mg) - 0.5, log(mg) + 0.5; length=41), g in range(log(α[k, j, a+1]) - 1, log(α[k, j, a+1]) + 1; length=21); best = max(best, nbll(y, x, exp(lm), exp(g))); end
    llj[k, j, a+1] = best
  end
  (m=m, α=α, llk=llk, llr=llr, llj=llj, ng=ng, n=n, e=e, adult=adult)
end
let dm = maximum(reld(EBR.m[i], CS.eb.m[i]) for i in eachindex(EBR.m)), dll = maximum(EBR.llr .- EBR.llk), dα = maximum(abs(log(EBR.α[i]) - log(CS.eb.α[i])) for i in eachindex(EBR.α) if isfinite(EBR.α[i]))
  check("6.1 sapling prior refitted from SQL (TRAIN plots only; groups key × class × adult on the plot): m = Σn/Σe and the NB α (independent optimiser) = the cached prior",
        dm < 1e-9 && dll < 1e-6, "max rel diff m $(fmt(dm)); kernel α within $(fmt(dll)) nats of the refit's maximum (max |Δ ln α| $(fmt(dα))); α range $(fmt(minimum(CS.eb.α))) – $(fmt(maximum(filter(isfinite, CS.eb.α))))")
  gain = sum(EBR.llj .- EBR.llr)
  info("6.1b [stat] m = Σn/Σe is the Poisson MLE, not the NB profile MLE", @sprintf("joint (m, α) maximum improves the summed log-likelihood by %.2f nats over all %d groups (the doc calls α a profile-likelihood estimate)", gain, count(>(0), EBR.ng)))
end
# 6.2 v1 sapling cells of every set = the posterior mean from the refit (SQL counts, FIADB microplot fractions)
let mx = 0.0, nbad = 0, worst = ""
  for nm in ("train", "val", "test")
    Dn = CS.sets[nm]; S = Dn.S; sid = Dict(iid => s for (s, iid) in enumerate(Dn.iid)); kix = Dict(k => i for (i, k) in enumerate(KEYS)); nsn = length(Dn.iid)
    n = zeros(Int, nk, 2, nsn); adult = falses(nk, nsn)
    for r in eachrow(v1[v1.set .== nm, :])
      (ismissing(r.s2) || startswith(coalesce(r.fate2, ""), "harvest")) && continue
      s = sid[r.iid]; k = kix[r.key]; DIA = coalesce(r.dia, 0.0); DIA >= 5.0 && (adult[k, s] = true); (1.0 <= DIA < 5.0) && (n[k, DIA < 3.0 ? 1 : 2, s] += 1)
    end
    for s in 1:nsn, k in 1:nk, j in 1:2
      e = AREA_MIC * FF[nm][s, 2]; mg = EBR.m[k, j, adult[k, s]+1]; α = EBR.α[k, j, adult[k, s]+1]
      Nh = mg > 0 ? (n[k, j, s] + α) / (e + α / mg) : n[k, j, s] / e
      i = cellix(S, s, k, j, 1); Nb = S.N0[i] + S.N0[i+1] + S.N0[i+2]; d = reld(Nb, Nh; floor=1e-9)
      d > mx && (mx = d; worst = "$nm site $s $(KEYS[k]) class $j n $(n[k, j, s]): builder $(Nb), refit $(Nh)")
      d > 1e-4 && (nbad += 1)
    end
  end
  check("6.2 v1 sapling density of every (site, key, class 1–2) = posterior mean (n + α)/(e + α/m) from the refit", nbad == 0, "max rel diff $(fmt(mx)); cells > 1e-4: $(nbad)" * (nbad > 0 ? " | $worst" : ""))
end
# 6.3 calibration of the prior predictive on VAL plots (out of sample): mean count and P(n = 0) per group
let Dv = CS.sets["val"], sid = Dict(iid => s for (s, iid) in enumerate(Dv.iid)), kix = Dict(k => i for (i, k) in enumerate(KEYS)), nsn = length(Dv.iid)
  n = zeros(Int, nk, 2, nsn)
  for r in eachrow(v1[v1.set .== "val", :])
    (ismissing(r.s2) || startswith(coalesce(r.fate2, ""), "harvest")) && continue
    DIA = coalesce(r.dia, 0.0); (1.0 <= DIA < 5.0) && (n[kix[r.key], DIA < 3.0 ? 1 : 2, sid[r.iid]] += 1)
  end
  o = 0.0; ex = 0.0; z0o = 0; z0e = 0.0; ll_eb = 0.0; ll_po = 0.0
  for s in 1:nsn, k in 1:nk, j in 1:2
    a = Dv.S.seed0[k, s] + 1; mg = CS.eb.m[k, j, a]; α = CS.eb.α[k, j, a]; (mg > 0 && isfinite(α)) || continue; μ = mg * AREA_MIC * Dv.fm1[s]
    o += n[k, j, s]; ex += μ; z0o += n[k, j, s] == 0; z0e += (α / (α + μ))^α
    ll_eb += Distributions.logpdf(Distributions.NegativeBinomial(α, α / (α + μ)), n[k, j, s]); ll_po += Distributions.logpdf(Distributions.Poisson(μ), n[k, j, s])
  end
  info("6.3 [stat] prior predictive on VAL v1 microplots (fitted on train)", @sprintf("saplings observed %d vs expected %.0f (ratio %.3f); zero cells observed %d vs expected %.0f; NB log-lik %.0f vs Poisson %.0f", o, ex, o / ex, z0o, z0e, ll_eb, ll_po))
end
# 6.4 [stat] the microplots are REMEASURED: the v2 microplot tally scores the same 4 microplots. Persistence-null count NLL of classes 1–2 with the EB layer vs the raw v1 counts
let res = Dict{String,Float64}(), S = DT.S
  surv = 0; newm = 0
  for r in eachrow(L.tr[in.(L.tr.iid, Ref(Set(DT.iid))), :]); (r.out == "alive" && r.dia_cm < 12.7 && r.dia2_cm < 12.7) && (surv += 1); end
  for r in eachrow(L.ig[in.(L.ig.iid, Ref(Set(DT.iid))), :]); r.dia_cm < 12.7 && (newm += 1); end
  # raw v1 microplot densities (n / e) for classes 1–2
  raw = zeros(nk, 2, ns)
  t1 = v1trees(L, Set(DT.iid)); sidx = Dict(iid => s for (s, iid) in enumerate(DT.iid)); kix = Dict(k => i for (i, k) in enumerate(KEYS))
  for r in eachrow(t1); j = H4.dclass(r.dia_cm); j in (1, 2) || continue; raw[kix[r.key], j, sidx[r.iid]] += r.w_ha / DT.fm1[sidx[r.iid]]; end
  function cnll(Nfun, rM, bg)
    tot = 0.0
    for s in 1:ns, k in 1:nk, j in 1:2
      μ = (Nfun(s, k, j) + bg) * DT.fm2[s] * AREA_MIC; tot -= nb_logpmf(Float64(DT.n[k, j, s]), μ, rM)
    end
    tot
  end
  ebN(s, k, j) = (i = cellix(S, s, k, j, 1); S.N0[i] + S.N0[i+1] + S.N0[i+2]); rawN(s, k, j) = raw[k, j, s]
  for (lab, fn) in (("EB", ebN), ("raw", rawN))
    f(x) = cnll(fn, exp(x[1]), exp(x[2])); r = cmaes(f, [0.0, log(5.0)], 0.5; λ=10, maxevals=400, logf=s -> nothing); res[lab] = r.f
    res[lab*"_r"] = exp(r.x[1]); res[lab*"_bg"] = exp(r.x[2])
  end
  check("6.4 [stat] v1 sapling state vs the v2 microplot tally (the SAME 4 microplots, $(surv) of the $(surv + newm) train v2 microplot saplings are remeasured v1 saplings): the EB layer predicts it at least as well as the raw v1 count (persistence, own r and bg)",
        res["EB"] <= res["raw"], @sprintf("count NLL classes 1–2, persistence: EB %.1f (r %.2f, bg %.2f /ha) vs raw v1 counts %.1f (r %.2f, bg %.2f /ha): Δ %+.1f nats",
        res["EB"], res["EB_r"], res["EB_bg"], res["raw"], res["raw_r"], res["raw_bg"], res["EB"] - res["raw"]))
end
end # section 6

# ================================================================ 7. likelihood and observation model
if want(7)
section("7. likelihood / observation model")
let mx = 0.0
  for r in (0.05, 0.3, 1.0, 2.0, 7.5, 50.0, 100.0), μ in (1e-6, 0.01, 0.3, 1.0, 4.0, 25.0, 300.0), n in (0, 1, 2, 5, 17, 60, 400)
    a = nb_logpmf(Float64(n), μ, r); b = Distributions.logpdf(Distributions.NegativeBinomial(r, r / (r + μ)), n); mx = max(mx, abs(a - b) / max(1.0, abs(b)))
  end
  check("7.1 nb_logpmf = Distributions.NegativeBinomial(r, r/(r+μ)) (mean μ, variance μ + μ²/r)", mx < 1e-8, "max rel diff $(fmt(mx))")
end
function nll_ref(D, Ob)
  S = D.S; c = 0.0; dsz = 0.0; hh = 0.0; cempty = 0.0; nempty = 0; bgexp = 0.0
  for s in 1:H4.nsites(S), k in 1:S.nk, j in 1:NCLS
    r = cellix(S, s, k, j, 1):cellix(S, s, k, j, 3); Np = sum(S.N[r]); n = D.n[k, j, s]
    area = j >= 3 ? 1 / (6.018046 * 2.4710538) : 1 / (74.965282 * 2.4710538)       # 1 / (TPA_UNADJ · ac/ha)
    f = j >= 3 ? D.fs2[s] : D.fm2[s]; bg = j >= 3 ? Ob.bgS : Ob.bgM; μ = (Np + bg) * f * area; rr = j >= 3 ? Ob.rS : Ob.rM
    l = -Distributions.logpdf(Distributions.NegativeBinomial(rr, rr / (rr + μ)), n); c += l; bgexp += bg * f * area
    (Np < 1e-3 && n > 0) && (cempty += l; nempty += n)
    if n > 0 && Np > 1e-9
      dq = sqrt(sum(S.B[r]) / (π / 4e4 * Np)); qo = sqrt(D.D2[k, j, s] / n)
      dsz -= Distributions.logpdf(Distributions.Normal(log(dq), sqrt(Ob.sD[j]^2 / n + Ob.σD^2)), log(qo))
      if D.nH[k, j, s] > 0
        Hp = sum(S.N[r] .* S.H[r]) / Np
        hh -= Distributions.logpdf(Distributions.Normal(log(Hp), sqrt(Ob.sH[j]^2 / D.nH[k, j, s] + Ob.σH^2)), log(D.Hs[k, j, s] / D.nH[k, j, s]))
      end
    end
  end
  (total=c + dsz + hh, count=c, size=dsz, height=hh, cempty=cempty, nempty=nempty, bgexp=bgexp)
end
OBW = let z = copy(SP.z0), iob = [findfirst(==(String(nm)), SP.names) for nm in OBS4]
  f(x) = (zz = copy(z); zz[iob] = x; nll4(DT, SP.decode(zz)[3]).total)
  r = cmaes(f, z[iob], 0.3; λ=12, maxevals=600, logf=s -> nothing); zz = copy(z); zz[iob] = r.x; SP.decode(zz)[3]
end
let a = nll4(DT, OBW), b = nll_ref(DT, OBW)
  check("7.2 nll4 (threaded) = independent recomputation (Distributions NB with background + Normal size / height terms) on all $(ns) train sites after a warm-start run (observation parameters fitted)",
        reld(a.total, b.total) < 1e-7 && reld(a.count, b.count) < 1e-7 && reld(a.size, b.size) < 1e-9 && reld(a.height, b.height) < 1e-9,
        @sprintf("nll4 %.3f (count %.1f size %.1f height %.1f) | ref %.3f", a.total, a.count, a.size, a.height, b.total))
  ntr = sum(DT.n)
  info("7.3 [stat] fitted observation parameters at the warm start", @sprintf("rM %.3g rS %.3g bgM %.3g bgS %.3g /ha σD %.3g σH %.3g | background: expected %.1f trees over the whole train table (%.2f %% of the %d tallied), the count NLL of trees in cells the model leaves (near) empty (N < 1e-3 /ha): %.0f nats for %d trees (%.1f %% of the count NLL)",
       OBW.rM, OBW.rS, OBW.bgM, OBW.bgS, OBW.σD, OBW.σH, b.bgexp, 100b.bgexp / ntr, ntr, b.cempty, b.nempty, 100b.cempty / b.count))
  check("7.3b [stat] background densities are not at their bounds", 1.001e-3 < OBW.bgM < 499 && 1.001e-4 < OBW.bgS < 49, @sprintf("bgM %.4g (bounds 1e-3–500), bgS %.4g (1e-4–50)", OBW.bgM, OBW.bgS))
end
let wsd = within_sd(DT), sD = wsd[1], sH = wsd[2]
  check("7.4 within-cell spreads sD, sH = recomputation from the TRAIN v2 cells (cells ≥ 2 trees)", maximum(abs.(sD .- CS.sD)) < 1e-12 && maximum(abs.(sH .- CS.sH)) < 1e-12)
  ncell = [count(>=(2), view(DT.n, :, j, :)) for j in 1:NCLS]; sdmax = [j < NCLS ? log(H4.CLS_HI[j] / H4.CLS_LO[j]) / 2 : Inf for j in 1:NCLS]
  bad = [j for j in 1:NCLS-1 if CS.sD[j] > sdmax[j] + 1e-12 || ncell[j] < 30]
  check("7.4b [stat] every class's sD rests on ≥ 30 cells and does not exceed the largest sd possible inside the class (log U/L / 2)", isempty(bad),
        "classes " * join(["$j: sD $(round(CS.sD[j]; digits=3)) (max possible $(round(sdmax[j]; digits=3)), $(ncell[j]) cells)" for j in bad], "; ") * " — 0.1 is the fallback of within_sd (data4.jl:186) when no cell has ≥ 2 trees")
end
# 7.5 the size term treats log QMD_obs − log Dq as a SAMPLING error sD²/n, but the state was initialised from the same (remeasured) trees
let t1 = v1trees(L, Set(DT.iid)), sidx = Dict(iid => s for (s, iid) in enumerate(DT.iid)), kix = Dict(k => i for (i, k) in enumerate(KEYS))
  acc = Dict{Tuple{Int,Int,Int},Vector{Float64}}()
  for r in eachrow(t1)
    r.out == "alive" || continue; j1 = H4.dclass(r.dia_cm); j1 >= 3 || continue; j2 = H4.dclass(r.dia2_cm); j2 == j1 || continue
    a = get!(acc, (sidx[r.iid], kix[r.key], j1), zeros(3)); a[1] += 1; a[2] += r.dia_cm^2; a[3] += r.dia2_cm^2
  end
  res = Float64[]; pred = Float64[]
  for ((s, k, j), a) in acc
    a[1] >= 2 && DT.n[k, j, s] == a[1] || continue        # cells whose v2 trees are exactly these stayers (no movers in / out, no deaths, no ingrowth)
    push!(res, 0.5 * log(a[3] / a[2])); push!(pred, CS.sD[j]^2 / a[1])
  end
  dv = var(res); info("7.5 [stat] size-term variance vs the actual v1→v2 change of the same trees", @sprintf("%d cells (classes ≥ 3) whose v2 trees are exactly their v1 stayers: var(log QMD2 − log QMD1) = %.3g (incl. the mean growth shift) vs mean sampling variance sD²/n %.3g used by the likelihood (ratio %.3f): the likelihood treats the cell mean as a fresh sample, but the v1 state IS that sample",
       length(res), dv, mean(pred), dv / mean(pred)))
end
end # section 7

# ================================================================ 8. warm start and parameter space
if want(8)
section("8. warm start / parameter space")
let gd = gl3
  check("8.1 fixed globals as documented (gR 0, mA 10, sdimax 1500, βC $(FIX_DEFAULT[:βC]))", G.gR == 0 && isapprox(G.mA, 10) && isapprox(G.sdimax, 1500) && G.βC == FIX_DEFAULT[:βC])
  # independent recomputation: HBAD3P multiplies growth by e^{gR RD_subplot} (CR/0.4)^φ; at the v1 state, tree-weighted, per SUBPLOT RD as HBAD3P competes
  t1 = v1trees(L, Set(DT.iid)); f1d = Dict(iid => DT.f1[s] for (s, iid) in enumerate(DT.iid))
  sub = combine(groupby(t1, [:iid, :sb]), [:iid, :w_ha, :dia_cm] => ((ii, w, d) -> 4 * sum(w .* (d ./ 25.4) .^ 1.605) / f1d[ii[1]] / 1500) => :RD)   # HBAD3P: subplot density ×4 / a
  rdd = Dict((r.iid, r.sb) => r.RD for r in eachrow(sub)); t1 = t1[.!ismissing.(t1.cr) .& (coalesce.(t1.cr, 0) .> 0), :]
  mult = sum(t1.w_ha .* exp.(gd[:gR] .* [rdd[(r.iid, r.sb)] for r in eachrow(t1)]) .* (t1.cr ./ 40) .^ gd[:φ]) / sum(t1.w_ha)
  info("8.2 [spec §7 I5] ε rescaling for HBAD3P's dropped crowding gain e^{gR RD} and crown term (CR/0.4)^φ", @sprintf("used %.4f = mean_plots e^{gR RD_plot} × mean_trees (CR/0.4)^φ (product of separate unweighted means) | tree-weighted joint mean at the SUBPLOT RD HBAD3P used: %.4f (%+.1f %%)", EPSFAC, mult, 100 * (mult / EPSFAC - 1)))
  fC = mean(1 .+ G.βC .* log.(ST.C ./ H4.CREF)); fT = mean(H4.fT_rel(P[k], G, ST.T[s]) for s in 1:ns for k in 1:nk)
  check("8.3 [spec §7 I5] the warm start keeps HBAD3P's growth level: the terms HBAD3P did not have average ≈ 1 over the train plots × keys (within 2 %)", abs(fC * fT - 1) < 0.02,
        @sprintf("f_CO2 %.4f, f_T %.4f (wT %.1f; E[f_T] < 1 because f_T is a Gaussian centred off its optimum: = 1 only AT T_ref): ε is not rescaled for it; plot-level vs HBAD3P's subplot-level competition also changes O", fC, fT, G.wT))
  z3, w3 = (-sqrt(3.0), 0.0, sqrt(3.0)), (1 / 6, 2 / 3, 1 / 6)
  info("8.4 vigour lanes vs HBAD3P's quadrature", @sprintf("HBAD3P σv %.3f on 3-node Gauss–Hermite (var %.3f, kurtosis 3); HBAD4P lanes var %.3f, kurtosis %.2f, κ = %.3g (HBAD3P: persistent vigour)", gd[:σv], sum(w3 .* z3 .^ 2), sum(lw .* lz .^ 2), sum(lw .* lz .^ 4), G.κ))
end
let ok = true, msg = String[]
  for (j, sp) in enumerate(GSPEC4)
    nm = sp[1]; haskey(SP.fix, nm) && continue
    zj = findfirst(==(String(nm)), SP.names); zj === nothing && (ok = false; push!(msg, "$nm not a coordinate"); continue)
    for zv in (1e3, -1e3)
      z = copy(SP.z0); z[zj] = zv; v = getfield(SP.decode(z)[nm in OBS4 ? 3 : 2], nm); (sp[3] - 1e-12 <= v <= sp[4] + 1e-12) || (ok = false; push!(msg, "$nm $(zv)"))
    end
  end
  z = copy(SP.z0); z[1:nk] .= 50.0; ok2 = all(p.lnε == EPS_MAX for p in SP.decode(z)[1])
  check("8.5 decode clamps every free global to its bounds and ln ε to EPS_MAX", ok && ok2, join(msg, ", "))
  i = findfirst(==("σv"), SP.names); z = copy(SP.z0); z[i] = -0.3; z2 = copy(SP.z0); z2[i] = 0.3
  info("8.6 [stat] σv decoded as |z|: the objective is symmetric in its coordinate", @sprintf("σv(−0.3) = %.2f, σv(+0.3) = %.2f — a two-mode landscape for CMA-ES near σv = 0 (harmless: the :sym lanes are symmetric)", SP.decode(z)[2].σv, SP.decode(z2)[2].σv))
end
end # section 8

# ================================================================ 9. stand-level quantities and baselines
if want(9)
section("9. stand-level quantities and baselines (train in-sample; val out-of-sample)")
function standv(S, D, which)    # which = :sim (S.N/B/H) | :obs (tally) | :v1 (S.N0/B0/H0)
  ns_ = H4.nsites(S); N = zeros(ns_); N12 = zeros(ns_); BA = zeros(ns_); D2 = zeros(ns_); LH = zeros(ns_); BA12 = zeros(ns_)
  for s in 1:ns_, k in 1:S.nk, j in 1:NCLS
    r = cellix(S, s, k, j, 1):cellix(S, s, k, j, 3)
    if which == :obs
      a = (j >= 3 ? D.fs2[s] : D.fm2[s]) * area_class(j); n = D.n[k, j, s] / a; b = D.D2[k, j, s] * π / 4e4 / a; h = D.nH[k, j, s] > 0 ? D.Hs[k, j, s] / D.nH[k, j, s] : 0.0
    else
      Nv = which == :sim ? S.N[r] : S.N0[r]; Bv = which == :sim ? S.B[r] : S.B0[r]; Hv = which == :sim ? S.H[r] : S.H0[r]; n = sum(Nv); b = sum(Bv); h = n > 0 ? sum(Nv .* Hv) / n : 0.0
    end
    N[s] += n; BA[s] += b; j >= 3 && (N12[s] += n; BA12[s] += b; LH[s] += b * h)
  end
  (N=mean(N), N12=mean(N12), BA=mean(BA), QMD12=sqrt(sum(BA12) / sum(N12) * 4e4 / π), LH=sum(LH) / sum(BA12))
end
let o = standv(ST, DT, :obs), m = standv(ST, DT, :sim), z = standv(ST, DT, :v1)
  info("9.1 plot means at v2 (train, warm start)", @sprintf("N obs %.0f sim %.0f (%+.1f %%) [v1 %.0f] | N≥12.7 obs %.1f sim %.1f (%+.1f %%) | BA obs %.2f sim %.2f (%+.1f %%) [v1 %.2f] | QMD≥12.7 obs %.2f sim %.2f | Lorey H≥12.7 obs %.2f sim %.2f",
       o.N, m.N, 100 * (m.N / o.N - 1), z.N, o.N12, m.N12, 100 * (m.N12 / o.N12 - 1), o.BA, m.BA, 100 * (m.BA / o.BA - 1), z.BA, o.QMD12, m.QMD12, o.LH, m.LH))
  sidx = Dict(iid => s for (s, iid) in enumerate(DT.iid)); obs = 0.0
  for r in eachrow(L.tr[in.(L.tr.iid, Ref(keys(sidx))) .& (L.tr.out .== "alive") .& (L.tr.dia_cm .< 12.7) .& (L.tr.dia2_cm .>= 12.7), :]); obs += r.w2_ha / DT.fs2[sidx[r.iid]]; end
  for r in eachrow(L.ig[in.(L.ig.iid, Ref(keys(sidx))) .& (L.ig.dia_cm .>= 12.7), :]); obs += r.w_ha / DT.fs2[sidx[r.iid]]; end
  Sb = subS(ST, 1:ns); for i in eachindex(Sb.N0); Sb.cl[i] <= 2 && (Sb.N0[i] = 0.0; Sb.B0[i] = 0.0); end
  H4.simulate!(Sb, [withP(p; lnF=-Inf) for p in P], G); mb = standv(Sb, DT, :sim)
  info("9.2 ingrowth into 12.7 cm over the interval (train, plot mean)", @sprintf("observed %.1f /ha | warm start %.1f /ha (N≥12.7 full run %.1f − run without classes 1–2 and recruitment %.1f)", obs / ns, m.N12 - mb.N12, m.N12, mb.N12))
  Sb = nothing; GC.gc()
end
# transition-matrix null: train-estimated per (key, v1 class, dt tercile) fates, ingrowth per (key, class, seed) per year; scored with its own fitted observation parameters
TM = let t1 = v1trees(L, Set(DT.iid)), sidx = Dict(iid => s for (s, iid) in enumerate(DT.iid)), kix = Dict(k => i for (i, k) in enumerate(KEYS)), dtv = DT.iv.dt
  qs = quantile(dtv, [1 / 3, 2 / 3]); bin(dt) = dt <= qs[1] ? 1 : dt <= qs[2] ? 2 : 3
  Tn = zeros(nk, NCLS, 3, NCLS); Tb = zeros(nk, NCLS, 3, NCLS); W0 = zeros(nk, NCLS, 3); Hs = zeros(nk, NCLS); Hc = zeros(nk, NCLS)
  for r in eachrow(t1)
    j1 = H4.dclass(r.dia_cm); j1 == 0 && continue; s = sidx[r.iid]; k = kix[r.key]; w = r.w_ha / (ismicro(r.w_ha) ? DT.fm1[s] : DT.f1[s]); b = bin(dtv[s])
    W0[k, j1, b] += w
    if r.out == "alive"; j2 = H4.dclass(r.dia2_cm); j2 == 0 && continue; Tn[k, j1, b, j2] += w; Tb[k, j1, b, j2] += w * π / 4e4 * r.dia2_cm^2; r.ht2_m > 0 && (Hs[k, j2] += r.ht2_m; Hc[k, j2] += 1); end
  end
  Ig = zeros(nk, NCLS, 2); Igb = zeros(nk, NCLS, 2); Dsum = zeros(2, nk)
  for s in 1:ns, k in 1:nk; Dsum[ST.seed0[k, s]+1, k] += dtv[s]; end
  for r in eachrow(L.ig[in.(L.ig.iid, Ref(keys(sidx))), :])
    j = H4.dclass(r.dia_cm); j == 0 && continue; s = sidx[r.iid]; k = kix[r.key]; a = ST.seed0[k, s] + 1; w = r.w_ha / (ismicro(r.w_ha) ? DT.fm2[s] : DT.fs2[s])
    Ig[k, j, a] += w; Igb[k, j, a] += w * π / 4e4 * r.dia_cm^2; r.ht_m > 0 && (Hs[k, j] += r.ht_m; Hc[k, j] += 1)
  end
  (Tn=Tn, Tb=Tb, W0=W0, Ig=Ig, Igb=Igb, Dsum=Dsum, H=Hs ./ max.(Hc, 1), bin=bin)
end
"Write the transition-matrix null's v2 state of set D into D.S.N / B / H (lane 1)."
function tm_state!(D)
  S = D.S; fill!(S.N, 0.0); fill!(S.B, 0.0); fill!(S.H, 0.0)
  for s in 1:H4.nsites(S)
    b = TM.bin(D.iv.dt[s])
    for k in 1:S.nk
      n2 = zeros(NCLS); b2 = zeros(NCLS)
      for j1 in 1:NCLS
        i = cellix(S, s, k, j1, 1); N0 = S.N0[i] + S.N0[i+1] + S.N0[i+2]; (N0 > 0 && TM.W0[k, j1, b] > 0) || continue
        for j2 in 1:NCLS; n2[j2] += N0 * TM.Tn[k, j1, b, j2] / TM.W0[k, j1, b]; b2[j2] += N0 * TM.Tb[k, j1, b, j2] / TM.W0[k, j1, b]; end
      end
      a = S.seed0[k, s] + 1; rate = TM.Dsum[a, k] > 0 ? D.iv.dt[s] / TM.Dsum[a, k] : 0.0
      for j in 1:NCLS; n2[j] += TM.Ig[k, j, a] * rate; b2[j] += TM.Igb[k, j, a] * rate; end
      for j in 1:NCLS; i = cellix(S, s, k, j, 1); S.N[i] = n2[j]; S.B[i] = b2[j]; S.H[i] = TM.H[k, j] > 0 ? TM.H[k, j] : 1.37 + 0.5 * H4.CLS_LO[j]; end
    end
  end
  D
end
let res = Dict{String,Any}(), Dv = CS.sets["val"], iob = [findfirst(==(String(nm)), SP.names) for nm in OBS4]
  fitob(D) = (z = copy(SP.z0); f(x) = (zz = copy(z); zz[iob] = x; nll4(D, SP.decode(zz)[3]).total); r = cmaes(f, z[iob], 0.3; λ=12, maxevals=600, logf=s -> nothing); zz = copy(z); zz[iob] = r.x; SP.decode(zz)[3])
  for (lab, prep!) in (("persistence", D -> null_state!(D)), ("transition matrix", D -> tm_state!(D)), ("HBAD4P warm start", D -> (H4.simulate!(D.S, P, G); D)))
    prep!(DT); ob = fitob(DT); a = nll4(DT, ob); prep!(Dv); b = nll4(Dv, ob); res[lab] = (a, b)
  end
  rows = [@sprintf("%-18s train %10.1f (count %9.1f size %9.1f height %9.1f) | val %10.1f (count %9.1f)", k, v[1].total, v[1].count, v[1].size, v[1].height, v[2].total, v[2].count) for (k, v) in res]
  for r in rows; println("      ", r); end
  tm = res["transition matrix"][2]; pe = res["persistence"][2]; ws = res["HBAD4P warm start"][2]
  check("9.3 [stat] the persistence null is a fair (competitive) baseline: a train-estimated transition-matrix stand-table null (fates per key × class × dt tercile, ingrowth per key × class × seed) gains < 10 % of the persistence val NLL",
        pe.total - tm.total < 0.1 * abs(pe.total),
        @sprintf("val NLL: persistence %.0f, transition matrix %.0f (%+.0f), warm start %.0f (%+.0f) | val COUNT NLL: transition matrix %.0f vs warm start %.0f — gains over persistence mostly measure that trees grow and die", pe.total, tm.total, tm.total - pe.total, ws.total, ws.total - pe.total, tm.count, ws.count))
  H4.simulate!(DT.S, P, G)
end
end # section 9

# ================================================================ summary
@isdefined(con) && DBInterface.close!(con)
nf = count(r -> r[1] == "FAIL", RES); np = count(r -> r[1] == "PASS", RES)
println("\n==== SUMMARY: $(np) PASS, $(nf) FAIL, $(count(r -> r[1] == "INFO", RES)) INFO | $(round(time() - t00; digits=0)) s")
for r in RES; r[1] == "FAIL" && println("FAIL  ", r[2]); end
exit(nf > 0 ? 1 : 0)
