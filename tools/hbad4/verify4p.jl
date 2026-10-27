# HBAD4P verification, audit round 3 — the cohorts-in-bins ATOM kernel (docs/hbad4p_design.md §5; updated with the round-3 fixes, §9: NATOM 16, sorted adjacent merges
# with 1e-6 cm distance bins, lane mixing with the own atoms once, per-atom growth / size mortality, the DBH-error observation blur, the compact store with the
# working atoms in the per-thread scratch, the warm-start rescalings, the transition-matrix null): PASS / FAIL checks by code paths INDEPENDENT of the code under
# test (closed forms written out here, tools/hbad4/verify4p_ref.jl = a brute-force reference with atom LISTS, SQL on the read-only raw databases). Exit 1 on any FAIL.
#   0  hand-worked one-site / one-year cases (closed forms: growth, exact atom transfer, partial year, multipliers, CO2 on growth only, survival, recruitment,
#      self-thinning incl. the inflow weight, lane mixing, overtopping ramp, multi-class jumps, the open last class, merges)
#   1  kernel units: classes, lanes, merge primitives vs the reference (+ conservation), invariants (zero rates, growth only, lane mixing), allocations, CAPB overflow,
#      the state invariants after a warm-start run
#   2  kernel vs the brute-force reference step (exact functions, O(n²) overtopping, atom lists) on real sites, atom by atom (merge choices decided by rounding are
#      counted, not failed); lookup tables vs exact functions; determinism and thread parity
#   3  numerical diffusion of the atoms: (a) tree-level replay of the real train v1 trees (observed increments per key × class: the earlier review's +19 % test);
#      (b) a smooth size-dependent increment (cell-QMD rates vs each tree's own); (c) the full model at the warm start, NATOM = 4 vs 64 atoms and per-atom rates;
#      (d) objective roughness from discrete atom events
#   4  process semantics at the warm start (finite state, classes, heights, seed sources, temperature centring, CO2)
#   5  data pipeline vs SQL (slim_r4 curated_trees + REF_SPECIES, FIADB PLOT / COND / SUBP_COND / TREE): intervals, the role split, keys, forested fractions,
#      v1 / v2 stand tables, the v1 atoms, tally areas, climate, CO2, traits, cache key
#   6  empirical-Bayes sapling prior (independent refit, posterior means, out-of-sample calibration, remeasured microplots)
#   7  likelihood / observation model (NB pmf, nll4 recomputed, background, within-cell spreads)
#   8  warm start / parameter space (ε factor recomputed as HBAD3P competes, keys without an HBAD3P value, the mortality stress, decode bounds)
#   9  stand-level quantities and baselines (persistence vs a train-estimated transition-matrix null vs the warm start)
#   P  performance (timing, allocation; where the time goes is in the report)
# Tags: [spec] = code vs the design document, [phys] / [stat] = the design itself. INFO lines carry numbers that are not pass / fail.
#   HBAD4_SPLIT_CSV=<folds/split_<role>.csv> HBAD2_DATA_DIR=<fold product> ./julia_gdal.sh --project=. --threads=2 tools/hbad4/verify4p.jl [hbad3p fit dir]
# env: HBAD4_VERIFY_NSUB (sites of the brute-force checks, default 120), HBAD4_VERIFY_SECTIONS (e.g. "0,1,2"; default all incl. P), HBAD3_SLIM_DB, HBAD4_FIADB.
include(joinpath(@__DIR__, "data4.jl")); include(joinpath(@__DIR__, "params4.jl")); include(joinpath(@__DIR__, "..", "hbad2", "cmaes.jl"))
include(joinpath(@__DIR__, "verify4p_ref.jl")); import .Ref4
using DuckDB, Random
import Distributions
const H4 = HBAD4P
const NA4 = H4.NATOM
const ROLE = basename(dirname(normpath(DATA_DIR)))                                           # e.g. k1 (product = …/prep/<role>/product)
const FIT3P = length(ARGS) >= 1 ? ARGS[1] : get(ENV, "HBAD4_FIT3P", joinpath(DATA_DIR, "..", "..", "..", "hbad3p", ROLE, "fit3p_" * ROLE))
const SLIM_DB = get(ENV, "HBAD3_SLIM_DB", joinpath(WSROOT, "landis_region4", "data_r4", "slim_r4.duckdb"))
const FIADB = get(ENV, "HBAD4_FIADB", joinpath(WSROOT, "FIA", "FIASQLITE2PGSQL", "FIADB.duckdb"))
const ROLE_SPLIT = isempty(SPLIT_CSV) ? joinpath(DATA_DIR, "..", "..", "..", "folds", "split_$(ROLE).csv") : SPLIT_CSV
const FOLDS_CSV = joinpath(dirname(ROLE_SPLIT), "folds.csv")
const NSUB = parse(Int, get(ENV, "HBAD4_VERIFY_NSUB", "120"))
const SECTIONS = Set(strip.(split(get(ENV, "HBAD4_VERIFY_SECTIONS", "0,1,2,3,4,5,6,7,8,9,P"), ",")))
want(k) = string(k) in SECTIONS

# ---------------------------------------------------------------- reporting
const RES = Tuple{String,String,String}[]
function check(name, ok::Bool, detail="")
  push!(RES, (ok ? "PASS" : "FAIL", name, detail)); println(ok ? "PASS  " : "FAIL  ", name, isempty(detail) ? "" : "  — " * detail); flush(stdout); ok
end
info(name, detail) = (push!(RES, ("INFO", name, detail)); println("INFO  ", name, "  — ", detail); flush(stdout))
section(s) = (println("\n==== ", s); flush(stdout))
reld(a, b; floor=1e-12) = abs(a - b) / max(abs(a), abs(b), floor)
fmt(x) = @sprintf("%.4g", x)

# ---------------------------------------------------------------- helpers (independent of the kernel's internals: they use only the documented store layout and the
# documented run API — load_site! / step_site! / finish_site! / run_site!, cell_atoms of the scratch that ran the site, atoms0 of the store)
cellix(S, s, k, j, v) = S.off[s] + ((k - 1) * NCLS + (j - 1)) * NL + (v - 1)
"Initial atoms (n, D) of cell i (the store keeps only the cells occupied at v1)."
atoms_of(S, i; init=true) = (init || error("working atoms live in the scratch: atoms_x"); Ref4.Atoms(H4.atoms0(S, i)))
"Working atoms of global cell i of site s, from the scratch X that holds site s."
atoms_x(X, S, i; s=1) = Ref4.Atoms(H4.cell_atoms(X, S, s, i))
"A Sites4 of a subset of sites (every site has the same number of cells in this layout); `drop(i)` = true removes cell i's initial trees."
function subS(S, idx; drop=i -> false)
  nc = S.off[2] - S.off[1]; all(diff(S.off) .== nc) || error("subS: sites of unequal size")
  r = reduce(vcat, [collect(S.off[s]:S.off[s+1]-1) for s in idx])
  at = [q => atoms_of(S, r[q]) for q in eachindex(r) if !drop(r[q]) && S.N0[r[q]] > 0]
  H4.make_sites4([1 + (q - 1) * nc for q in 1:length(idx)+1], S.nk, S.sp[r], S.cl[r], S.ln[r], at, S.H0[r], S.A0[r], S.nyr[idx], S.frac[idx], S.planted[idx],
                 S.W[idx], S.Wn[idx], S.T[idx], S.C[idx], S.seed0[:, idx], S.lz, S.lw)
end
emptytabs(nk) = H4.Tabs(zeros(0, nk), zeros(0), zeros(0), zeros(0), zeros(0))
withG(G; kw...) = H4.GlobP4((haskey(kw, f) ? kw[f] : getfield(G, f) for f in fieldnames(H4.GlobP4))...)
withP(p; kw...) = H4.SpP4((haskey(kw, f) ? kw[f] : getfield(p, f) for f in fieldnames(H4.SpP4))...)
reset!(S, X, s) = H4.load_site!(S, X, s)
"Serial (1-thread path) whole-interval run of every site of S with one scratch; `each(s, X)` is called after each site (its final atoms are in X). Returns X."
function serial_run!(S, P, G; exact=false, each=nothing)
  X = H4.Scratch(maximum(diff(S.off))); T_ = exact ? emptytabs(length(P)) : H4.Tabs(P, G)
  for s in 1:H4.nsites(S)
    H4.run_site!(S, P, G, X, s, T_); each === nothing || each(s, X)
  end
  X
end
"One-site store run (exact functions): returns the scratch holding its final atoms."
sim1!(S, P, G; exact=true) = serial_run!(S, P, G; exact=exact)
"Reference whole-interval run of site s of S (Ref4 atom lists), from the initial state."
function ref_run(S, s, P, G; kw...)
  r = S.off[s]:S.off[s+1]-1; C = [atoms_of(S, i) for i in r]; Hh = S.H0[r]; A = S.A0[r]
  for y in 1:S.nyr[s] + (S.frac[s] > 0)
    Ref4.step!(C, Hh, A, S.sp[r], S.cl[r], S.ln[r], S.nk, P, G, S.W[s], S.Wn[s], S.T[s], S.C[s], S.planted[s], y <= S.nyr[s] ? 1.0 : S.frac[s], S.lz, S.lw, S.seed0[:, s]; kw...)
  end
  C, Hh, A
end
ba_(n, d) = n * π * d^2 / 4e4

println("HBAD4P verification (round 4, atoms) | product $(DATA_DIR) | role $(ROLE) | split $(ROLE_SPLIT) | threads $(Threads.nthreads()) | region $(REGION) | NSUB $(NSUB) | sections $(sort(collect(SECTIONS)))")
t00 = time()
CS = cached_sets(["train", "val", "test"]); KEYS = CS.keys; TR = CS.TR; nk = length(KEYS)
DT = CS.sets["train"]; ST = DT.S; ns = H4.nsites(ST)
# warm start exactly as smoke4.jl (params4.jl warm_start4: ε × crowding / crown × subplot / plot light, ms for the mortality stress, κ = 0)
gl3 = Dict(Symbol(r.name) => Float64(r.value) for r in eachrow(CSV.read(joinpath(FIT3P, "globals.csv"), DataFrame)))
WS = warm_start4(CS, FIT3P); SP = WS.SP; P, G, Ob = WS.P, WS.G, WS.Ob; EPSFAC = WS.fac.eps
lz, lw = ST.lz, ST.lw
# observation parameters fitted at runtime (audit r4 V: not hard-coded — they go stale with the data) at the warm start on the train set (CMA-ES on the
# six observation coordinates, as smoke4); fixed for all comparisons below
OBX = let z = copy(SP.z0), iob = [findfirst(==(String(nm)), SP.names) for nm in OBS4]
  H4.simulate!(ST, P, G); f(x) = (zz = copy(z); zz[iob] = x; nll4(DT, SP.decode(zz)[3]).total)
  r = cmaes(f, z[iob], 0.3; λ=12, maxevals=parse(Int, get(ENV, "HBAD4_VERIFY_OBS_EVALS", "500")), logf=s -> nothing); zz = copy(z); zz[iob] = r.x; SP.decode(zz)[3]
end
println("observation parameters (fitted at the warm start): rM $(round(OBX.rM; sigdigits=4)) rS $(round(OBX.rS; sigdigits=4)) bgM $(round(OBX.bgM; sigdigits=4)) bgS $(round(OBX.bgS; sigdigits=4)) σD $(round(OBX.σD; sigdigits=4)) σH $(round(OBX.σH; sigdigits=4))")
rng = MersenneTwister(20261026); SUB = sort(randperm(rng, ns)[1:min(NSUB, ns)])
println("sets loaded in $(round(time() - t00; digits=1)) s; train sites $(ns), keys $(join(KEYS, ",")); NATOM $(NA4); warm start ", WS.fac)

# ---------------------------------------------------------------- one-site builder for the hand cases
"One site with nkk keys; cells (k, j, v, [(n, D), …], H, A); Δ = 1 → one whole year, else one partial year of length Δ."
function onesite(cells; nkk=1, Δ=1.0, W=0.9, Wn=0.95, T=18.0, C=400.0, pl=false, seed0=falses(nkk, 1), nyr=nothing)
  nc = nkk * NCLS * NL
  sp = Int32[k for k in 1:nkk for _ in 1:NCLS for _ in 1:NL]; cl = Int8[j for _ in 1:nkk for j in 1:NCLS for _ in 1:NL]; ln = Int8[v for _ in 1:nkk for _ in 1:NCLS for v in 1:NL]
  H0 = zeros(nc); A0 = zeros(nc); at = Pair{Int,Vector{Tuple{Float64,Float64}}}[]
  for (k, j, v, at_, Hh, A) in cells
    i = ((k - 1) * NCLS + (j - 1)) * NL + v; push!(at, i => collect(at_)); H0[i] = Hh; A0[i] = A
  end
  ny = nyr === nothing ? (Δ == 1.0 ? 1 : 0) : nyr
  H4.make_sites4([1, nc + 1], nkk, sp, cl, ln, at, H0, A0, [ny], [Δ == 1.0 ? 0.0 : Δ], [pl], [W], [Wn], [T], [C], BitMatrix(seed0), lz, lw)
end
ci(k, j, v) = ((k - 1) * NCLS + (j - 1)) * NL + v
# hand-case globals: growth from light and water only (b = 2, a = 2, ω = 1, θ = 8), no height relaxation (η = λ = μ = 0), no mortality, no thinning effect
Gh = withG(G; b=2.0, a=2.0, gR=0.0, θ=8.0, ω=1.0, s=1.2, c=1.2, η=0.0, λ=0.0, μ=0.0, kh=5.0, m0=0.0, ms=0.0, σST=0.0, β=2.0, ζ=-0.5, mW=0.0, mA=1e4,
           pD=0.0, pH=0.0, pM=0.0, σv=0.3, sdimax=1500.0, am=1.0, ρ=0.0, f0=0.0, βT=0.0, βC=0.0, κ=0.0, λ0=3.0, λ1=-0.07, ε0=0.3, ωR=0.0)
Ph = H4.SpP4(0.0, -Inf, 30.0, 3.0, 1e12, 18.0)            # ε = 1, no recruitment (lnF = −Inf), Hmax 30 m, ST 3, no senescence, T_ref 18 °C
hs_(D; Hm=30.0, s=1.2, c=1.2) = 1.37 + (Hm - 1.37) * (1 - exp(-s * D / (Hm - 1.37)))^c
"Kernel scratch after one step_site! of a one-site store (for rates: O, dD, sv, g)."
function onestep(S0, PP, GG; Δ=1.0)
  X = H4.Scratch(length(S0.N)); H4.load_site!(S0, X, 1); H4.step_site!(S0, PP, GG, X, 1, Δ, emptytabs(length(PP))); X
end
"The first atom's increment of every cell (one-atom cells: the atom's dD) after one step from the documented state."
lone_dD(cells, PP, GG; kw...) = (S0 = onesite(cells; kw...); X = onestep(S0, PP, GG); [S0.N0[li] > 0 ? X.adD[(li-1)*NA4+1] : 0.0 for li in eachindex(S0.N0)])
"""Hand growth of the atoms of one cell (documented equations, audit r3 I4: each atom from its OWN diameter; light from the cell's O, W^ω, lane z, fT, fD, fC
multipliers; ρ = 0): per-atom increments, the cell QMD before and after growth (the height uses the latter)."""
function hand_dD(at, Hh, O; ε=1.0, b=2.0, a=2.0, W=0.9, ω=1.0, mult=1.0, Hmax=30.0, θ=8.0, Δ=1.0)
  N = sum(first, at); Dq = sqrt(sum(x -> x[1] * x[2]^2, at) / N)
  g = exp(-a * O) * W^ω * mult; hyd = 1 - min(Hh / (1.1 * Hmax), 1.0)^θ
  dd = [ε * (x[2] / 20)^b * 122.98 * g * hyd / (2 * x[2] * max(Hh, 1.37)) * Δ for x in at]
  dd, Dq, sqrt(sum(x[1] * (x[2] + d)^2 for (x, d) in zip(at, dd)) / N)
end
"Per-atom increments of cell li (one-site store S0, after ONE step from its initial state) in the scratch (the kernel's per-atom rates, in the initial atom order)."
adD_of(X, li; S0=nothing) = [X.adD[(li-1)*NA4+m] for m in 1:(S0 === nothing ? X.ac[li] : length(H4.atoms0(S0, li)))]

# ================================================================ 0. hand-computed cases
if want(0)
section("0. hand-computed one-site / one-year cases (closed forms written out here; independent of the kernel and of Ref4)")
let at = [(10.0, 13.0), (5.0, 17.70)], Hh = 14.0, A = 40.0
  N = 15.0; Dq = sqrt((10 * 13.0^2 + 5 * 17.7^2) / 15); O = 0.5 * N * (Dq / 25.4)^1.605 / 1500           # own SDI only, ramp r(0) = ½
  dd, _, Dq2 = hand_dD(at, Hh, O); d1 = 13.0 + dd[1]; d2 = 17.7 + dd[2]; j2 = d2 >= 17.78 ? 4 : 3
  dH = hs_(Dq2) - hs_(Dq); Hn = Hh + dH; hD = hs_(sqrt((10 * d1^2 + 5 * d2^2) / 15))
  h3 = Hn + hs_(d1) - hD; h4 = Hn + hs_(d2) - hD; f = 15 * Hn / (10h3 + 5h4)
  S1 = onesite([(1, 3, 1, at, Hh, A)]); X1 = sim1!(S1, [Ph], Gh); i3 = ci(1, 3, 1); i4 = ci(1, 4, 1)
  a3 = atoms_x(X1, S1, i3); a4 = atoms_x(X1, S1, i4)
  ok = j2 == 4 && length(a3) == 1 && length(a4) == 1 && isapprox(a3[1][1], 10; rtol=1e-14) && isapprox(a3[1][2], d1; rtol=1e-13) && isapprox(a4[1][1], 5; rtol=1e-14) &&
       isapprox(a4[1][2], d2; rtol=1e-13) && isapprox(S1.N[i3], 10; rtol=1e-14) && isapprox(S1.B[i4], ba_(5, d2); rtol=1e-13) && isapprox(S1.H[i3], h3 * f; rtol=1e-12) &&
       isapprox(S1.H[i4], h4 * f; rtol=1e-12) && S1.A[i3] == A + 1 && S1.A[i4] == A + 1 && isapprox(10 * S1.H[i3] + 5 * S1.H[i4], 15Hn; rtol=1e-13)
  check("0.1 [r3 I4] two atoms of a class-3 cell, one year: each atom grows by the documented growth equation at its OWN diameter (cell light / climate / height); the 17.70-cm atom lands in class 4 EXACTLY (no diffusion), the 13.0-cm atom stays; height increment from the QMD of the grown atoms; heights H_n + H*(D_group) − H*(D_all) rescaled to conserve Σ N·H; A + 1", ok,
        @sprintf("dD %.6f / %.6f cm: atoms class 3 %s / hand (10, %.6f); class 4 %s / hand (5, %.6f); H3 %.6f/%.6f H4 %.6f/%.6f", dd[1], dd[2], a3, d1, a4, d2, S1.H[i3], h3 * f, S1.H[i4], h4 * f))
  S2 = onesite([(1, 3, 1, at, Hh, A)]; Δ=0.4); X2 = sim1!(S2, [Ph], Gh); a3 = sort(atoms_x(X2, S2, i3); by=last)
  dd4, _, Dq24 = hand_dD(at, Hh, O; Δ=0.4); dH2 = hs_(Dq24) - hs_(Dq)
  check("0.2 partial final year Δ = 0.4: increments 0.4 dD_m (17.70 + 0.4 dD < 17.78: nothing moves), H + dH(QMD of the grown atoms) with no split offset, A + 0.4",
        S2.N[i4] == 0 && length(a3) == 2 && isapprox(a3[2][2], 17.7 + dd4[2]; rtol=1e-13) && isapprox(a3[1][2], 13.0 + dd4[1]; rtol=1e-13) && isapprox(S2.A[i3], A + 0.4; rtol=1e-15) &&
        isapprox(S2.H[i3], Hh + dH2; rtol=1e-12), @sprintf("atoms %s (hand D %.6f / %.6f), H %.6f vs %.6f, A %.3f", a3, 13 + dd4[1], 17.7 + dd4[2], S2.H[i3], Hh + dH2, S2.A[i3]))
  # 0.3 growth multipliers: lane 3 (z = +√2), planted (e^{pD}), temperature e^{βT (T − Tref)}, CO2 1 + βC ln(C/Cref) scale every atom's dD exactly (ρ = 0); CO2 does NOT
  # enter survival or the self-thinning weight; temperature does (realised relative growth). Survival per atom at its own D; the cell's = N-weighted mean
  Gm = withG(Gh; pD=0.2, βT=0.1, βC=0.58, ms=0.05, β=2.0, ζ=-0.5); Gb = withG(Gh; ms=0.05, β=2.0, ζ=-0.5)
  Sb_ = onesite([(1, 3, 1, at, Hh, A)]); Sm_ = onesite([(1, 3, 3, at, Hh, A)]; pl=true, T=20.0, C=420.0); Sc_ = onesite([(1, 3, 3, at, Hh, A)]; pl=true, T=20.0, C=420.0)
  Xb = onestep(Sb_, [Ph], Gb); Xm = onestep(Sm_, [Ph], Gm); Xc = onestep(Sc_, [Ph], withG(Gm; βC=0.0))
  fac = exp(0.3 * sqrt(2)) * exp(0.2) * exp(0.1 * (20.0 - 18.0)) * (1 + 0.58 * log(420 / 390.1))
  cmult = exp(0.3 * sqrt(2)) * exp(0.1 * 2.0)                       # the realised relative growth without CO2 (enters gm and g)
  hz(c_, D) = 0.05 * exp(-2.0 * exp(-1.0 * O) * 0.9 * c_) * (D / 20)^-0.5
  svh(c_) = (10 * exp(-hz(c_, 13.0)) + 5 * exp(-hz(c_, 17.7))) / 15
  i3m = ci(1, 3, 3); am_ = adD_of(Xm, i3m; S0=Sm_); ab_ = adD_of(Xb, i3; S0=Sb_); ac_ = adD_of(Xc, i3m; S0=Sc_)
  check("0.3 growth multipliers e^{σv z} (z = +√2) · e^{pD} · e^{βT (T − Tref)} · (1 + βC ln(C/Cref)) scale every atom's dD exactly; survival (per atom, own D) and the thinning weight g include the temperature, NOT the CO2 term",
        all(isapprox.(am_, ab_ .* fac; rtol=1e-12)) && Xm.sv[i3m] == Xc.sv[i3m] && Xm.g[i3m] == Xc.g[i3m] && isapprox(Xm.sv[i3m], svh(cmult); rtol=1e-12) &&
        isapprox(Xb.sv[i3], svh(1.0); rtol=1e-12) && all(isapprox.(am_ ./ ac_, 1 + 0.58 * log(420 / 390.1); rtol=1e-12)) && length(am_) == 2 && Xm.asv[(i3m-1)*NA4+1] != Xm.asv[(i3m-1)*NA4+2],
        @sprintf("dD %s vs hand %s (factor %.5f); survival %.8f (βC 0.58) = %.8f (βC 0) vs hand %.8f", string(round.(am_; digits=6)), string(round.(ab_ .* fac; digits=6)), fac, Xm.sv[i3m], Xc.sv[i3m], svh(cmult)))
  # 0.9 a one-tree cell at 17.5 cm (top of class 3): dD 0.4 → all of it in class 4; dD 0.16 → none (exact transfer; the round-2 closure moved 76 % / 43 %)
  res = Float64[]; d0 = lone_dD([(1, 3, 1, [(14.87, 17.5)], 15.0, 30.0)], [Ph], Gh)[ci(1, 3, 1)]
  for target in (0.4, 0.16)
    S11 = onesite([(1, 3, 1, [(14.87, 17.5)], 15.0, 30.0)]); H4.simulate!(S11, [withP(Ph; lnε=log(target / d0))], Gh; exact=true); push!(res, S11.N[ci(1, 4, 1)] / 14.87)
  end
  check("0.9 [spec §2.3, round-2 B1] no diffusion of a ONE-TREE cell at 17.5 cm: dD 0.40 → 100 % in class 4, dD 0.16 → 0 %", isapprox(res[1], 1; rtol=1e-14) && res[2] == 0,
        @sprintf("dD 0.40: %.4f %% moved; dD 0.16: %.4f %% moved", 100res[1], 100res[2]))
  # 0.10 multi-class jumps: ε set so the 15.0-cm atom grows 7 cm → 22.0 (class 4); the 17.0-cm atom grows 7 · dD(17)/dD(15) (= 7 · 17/15 for b = 2) → 24.93 (class 5)
  cells = [(1, 3, 1, [(8.0, 15.0), (6.0, 17.0)], 14.0, 40.0)]; S0_ = onesite(cells); X0 = onestep(S0_, [Ph], Gh); dd0 = adD_of(X0, ci(1, 3, 1); S0=S0_); e7 = 7.0 / dd0[1]
  S4 = onesite(cells); X4 = sim1!(S4, [withP(Ph; lnε=log(e7))], Gh); a4 = atoms_x(X4, S4, ci(1, 4, 1)); a5 = atoms_x(X4, S4, ci(1, 5, 1)); d17 = 17.0 + dd0[2] * e7
  check("0.10 multi-class transfer: the 15.0-cm atom grows 7 cm → 22.0 cm in class 4, the 17.0-cm atom 7 · dD(17)/dD(15) (b = 2: 7 · 17/15) → class 5, class 3 empties; B per cell = Σ n π/4 D²",
        S4.N[ci(1, 3, 1)] == 0 && length(a4) == 1 && length(a5) == 1 && isapprox(a4[1][2], 22.0; rtol=1e-12) && isapprox(a5[1][2], d17; rtol=1e-12) && isapprox(d17, 17 + 7 * 17 / 15; rtol=1e-12) &&
        a4[1][1] == 8.0 && a5[1][1] == 6.0 && isapprox(S4.B[ci(1, 5, 1)], ba_(6, d17); rtol=1e-12), "class 4 $(a4), class 5 $(a5) (hand $(round(d17; digits=6)))")
end
# 0.11 into the open last class and growth inside it: a class-15 atom at 77.0 cm + 2 cm → 79.0 (class 16); a class-16 atom at 85 cm grows inside the class
let Pz = withP(Ph; Hmax=40.0)
  c15 = [(1, 15, 1, [(10.0, 77.0)], 30.0, 80.0)]; d0 = lone_dD(c15, [Pz], Gh)[ci(1, 15, 1)]; PP = [withP(Pz; lnε=log(2.0 / d0))]
  S5 = onesite(c15); X5 = sim1!(S5, PP, Gh); a16 = atoms_x(X5, S5, ci(1, 16, 1))
  c16 = [(1, 16, 1, [(10.0, 85.0)], 32.0, 120.0)]; d16 = lone_dD(c16, PP, Gh)[ci(1, 16, 1)]
  S6 = onesite(c16); H4.simulate!(S6, PP, Gh; exact=true)
  check("0.11 class 15 → open class 16 (atom at 77 + 2 = 79 cm) and growth inside class 16 (B = N π/4 (D + dD)², exact: no E[D] ≈ QMD approximation any more)",
        length(a16) == 1 && isapprox(a16[1][2], 79.0; rtol=1e-12) && isapprox(S6.B[ci(1, 16, 1)], ba_(10, 85 + d16); rtol=1e-12) && S6.N[ci(1, 16, 1)] == 10,
        @sprintf("class 16 after the jump %s; class-16 atom dD %.4f: B %.7f vs %.7f", string(a16), d16, S6.B[ci(1, 16, 1)], ba_(10, 85 + d16)))
end
# 0.12 merge on overflow: a class-4 cell with NATOM atoms receives a mover from class 3 → sorted by D, the closest ADJACENT pair (distance binned at 1e-6 cm, the first
# in D order) merges (N and N·D² conserved)
let c4 = [(1.0, 17.9 + 4.5 / (NA4 - 1) * u) for u in 0:NA4-1], cells = [(1, 3, 1, [(2.0, 17.7)], 15.0, 30.0), (1, 4, 1, c4, 16.0, 35.0)]
  S0_ = onesite(cells); X = onestep(S0_, [Ph], Gh); d3 = adD_of(X, ci(1, 3, 1); S0=S0_); d4 = adD_of(X, ci(1, 4, 1); S0=S0_)
  S = onesite(cells); X2 = sim1!(S, [Ph], Gh); a = sort(atoms_x(X2, S, ci(1, 4, 1)); by=last)
  inc = sort(vcat([(2.0, 17.7 + d3[1])], [(c4[u][1], c4[u][2] + d4[u]) for u in 1:NA4]); by=last)
  gaps = [floor((inc[u+1][2] - inc[u][2]) * 1e6) for u in 1:NA4]; p = argmin(gaps); m = (inc[p][1] + inc[p+1][1], sqrt((inc[p][1] * inc[p][2]^2 + inc[p+1][1] * inc[p+1][2]^2) / (inc[p][1] + inc[p+1][1])))
  hand = sort(vcat(inc[1:p-1], [m], inc[p+2:end]); by=last)
  check("0.12 merge: NATOM + 1 atoms in a cell (NATOM stayers + 1 mover) → the closest adjacent pair in D merges at the N·D²-mean diameter; N and N·D² of the cell conserved",
        17.7 + d3[1] >= 17.78 && length(a) == NA4 && all(isapprox(a[u][1], hand[u][1]; rtol=1e-14) && isapprox(a[u][2], hand[u][2]; rtol=1e-13) for u in 1:NA4) &&
        isapprox(sum(x -> x[1] * x[2]^2, a), sum(x -> x[1] * x[2]^2, inc); rtol=1e-14), "kernel $(a) | hand $(hand) (pair $p)")
end
# 0.4 survival (growth off): h = fM [m0/LONG + ms e^{σST (ST − 3)} e^{−β gm} (D/20)^ζ] + e^{mA (A/LONG − 1)}, fM = e^{pM} e^{mW (Wn − W)/0.1}, gm = e^{−am O} W^ω f_T (no CO2)
let N = 80.0, Dq = 25.0, A = 100.0, Lg = 200.0, Δ = 0.37
  Gs = withG(Gh; m0=1.0, ms=0.05, σST=0.3, β=2.0, am=1.0, ζ=-0.5, mA=10.0, mW=0.2, pM=0.1, βT=-0.05, βC=0.5)
  Ps = H4.SpP4(-Inf, -Inf, 30.0, 4.0, Lg, 18.5)
  O = 0.5 * N * (Dq / 25.4)^1.605 / 1500; gm = exp(-O) * 0.9 * exp(-0.05 * (19.0 - 18.5))
  h = exp(0.1) * exp(0.2 * (0.95 - 0.9) / 0.1) * (1.0 / Lg + 0.05 * exp(0.3) * exp(-2gm) * (Dq / 20)^-0.5) + exp(10 * (A / Lg - 1))
  S4 = onesite([(1, 5, 1, [(N, Dq)], 20.0, A)]; pl=true, Δ=Δ, T=19.0, C=410.0); X4 = sim1!(S4, [Ps], Gs); i = ci(1, 5, 1)
  check("0.4 survival e^{−hΔ} with the documented hazard (planted, drought anomaly, ST, size, senescence; temperature in gm, CO2 not) on N, the atom's D unchanged",
        isapprox(S4.N[i], N * exp(-h * Δ); rtol=1e-12) && atoms_x(X4, S4, i)[1][2] == Dq, @sprintf("N %.10f vs %.10f (h = %.6f /yr)", S4.N[i], N * exp(-h * Δ), h))
end
# 0.5 recruitment: (a) empty plot, no seed → ε0; (b) seed tree at v1; (c) current seed source 0.5 vs 1.5 trees/ha ≥ 12.7 cm (SEED_MIN = 1); partial year; 5 years of recruits merge
let lnF = -2.0, Pr = H4.SpP4(-Inf, -2.0, 30.0, 4.0, 1e12, 18.0), Gr = withG(Gh; ωR=0.5)
  rate0 = 185.2 * exp(lnF) * 0.9^0.5; E = Tuple{Int,Int,Int,Vector{Tuple{Float64,Float64}},Float64,Float64}[]
  S5 = onesite(E); X5 = sim1!(S5, [Pr], Gr)
  okA = isapprox(sum(S5.N[1:3]), 0.3 * rate0; rtol=1e-12) && isapprox(S5.N[1] / sum(S5.N[1:3]), 0.5; rtol=1e-12) && all(atoms_x(X5, S5, v) == [(S5.N[v], 2.54)] for v in 1:3) &&
        all(isapprox(S5.H[v], hs_(2.54); rtol=1e-12) for v in 1:3) && all(isapprox(S5.A[v], 4.5; rtol=1e-12) for v in 1:3)
  S5b = onesite(E; Δ=0.4); H4.simulate!(S5b, [Pr], Gr; exact=true)
  okA2 = isapprox(sum(S5b.N[1:3]), 0.4 * 0.3 * rate0; rtol=1e-12) && isapprox(S5b.A[1], 4.5; rtol=1e-12)
  S5c = onesite(E; seed0=trues(1, 1)); H4.simulate!(S5c, [Pr], Gr; exact=true); okB = isapprox(sum(S5c.N[1:3]), rate0; rtol=1e-12)
  S5d = onesite(E; seed0=trues(1, 1), nyr=5); X5d = sim1!(S5d, [Pr], Gr)
  Nr = 0.0; nys = Float64[]                                         # five annual cohorts, each year's rate from the stand RD at the start of the year (the earlier recruits)
  for y in 1:5; RD = Nr * (2.54 / 25.4)^1.605 / 1500; add = rate0 * exp(-3.0 * exp(-0.07) * RD); Nr += add; push!(nys, add); end
  Ah = sum(nys[y] * (4.5 + 5 - y) for y in 1:5) / Nr
  okD = isapprox(sum(S5d.N[1:3]), Nr; rtol=1e-12) && all(length(atoms_x(X5d, S5d, v)) == 1 && all(x -> x[2] == 2.54, atoms_x(X5d, S5d, v)) for v in 1:3) &&
        isapprox(S5d.A[1], Ah; rtol=1e-12)
  res = Float64[]
  for n3 in (0.5, 1.5)
    S6 = onesite([(1, 3, 1, [(n3, 16.0)], 15.0, 30.0)]); H4.simulate!(S6, [Pr], Gr; exact=true)
    RD = n3 * (16.0 / 25.4)^1.605 / 1500; push!(res, sum(S6.N[1:3]) / (185.2 * exp(lnF - 3.0 * exp(-0.07) * RD) * 0.9^0.5))
  end
  check("0.5 recruitment: rate 185.2 e^{lnF − λ0 e^{λ1(ST−3)} RD} W^{ωR} (seed ? 1 : ε0) Δ, lanes 0.5/0.25/0.25, one atom at 2.54 cm, H*(2.54), aged 4.5 after the step; seed = v1 seed tree or ≥ 1 tree/ha ≥ 12.7 cm now; five years of recruits (each year's rate at the RD of the earlier recruits): ONE atom at 2.54 cm per lane (equal diameters merge), N-weighted mean age",
        okA && okA2 && okB && okD && isapprox(res[1], 0.3; rtol=1e-12) && isapprox(res[2], 1.0; rtol=1e-12),
        @sprintf("no seed %.6f vs %.6f; Δ = 0.4 %.6f vs %.6f; seed0 %.6f vs %.6f; 5 yr %.6f vs %.6f (A %.3f); 0.5 tree/ha → factor %.4f, 1.5 → %.4f", sum(S5.N[1:3]), 0.3rate0,
                 sum(S5b.N[1:3]), 0.12rate0, sum(S5c.N[1:3]), rate0, sum(S5d.N[1:3]), Nr, S5d.A[1], res[1], res[2]))
end
# 0.6 self-thinning: (a) one cell at 2 sdimax → N/2, D unchanged; (b) two cells: the excess removed ∝ SDI_i (1 − g_i) (+1e-9 SDI_i), g_i = e^{−a O_i} W^ω
let Pz = withP(Ph; lnε=-Inf)
  N = 3000 / (30.0 / 25.4)^1.605; S7 = onesite([(1, 6, 1, [(N, 30.0)], 20.0, 50.0)]); X7 = sim1!(S7, [Pz], Gh); i = ci(1, 6, 1)
  okA = isapprox(S7.N[i], N / 2; rtol=1e-11) && atoms_x(X7, S7, i)[1][2] == 30.0
  n1 = 900.0; n2 = 1500.0; d = 20.0; sd1 = n1 * (d / 25.4)^1.605; sd2 = n2 * (d / 25.4)^1.605
  O1 = 0.5 * sd1 / 1500; O2 = (sd1 + 0.5 * sd2) / 1500; g1 = min(exp(-2O1) * 0.9, 1.0); g2 = min(exp(-2O2) * 0.9, 1.0)
  ex = sd1 + sd2 - 1500; w1 = sd1 * (1 - g1) + 1e-9 * sd1; w2 = sd2 * (1 - g2) + 1e-9 * sd2
  S8 = onesite([(1, 5, 1, [(n1, d)], 20.0, 50.0), (2, 5, 1, [(n2, d)], 10.0, 50.0)]; nkk=2); H4.simulate!(S8, [Pz, Pz], Gh; exact=true)
  h1 = n1 * (1 - ex * w1 / (w1 + w2) / sd1); h2 = n2 * (1 - ex * w2 / (w1 + w2) / sd2)
  check("0.6 self-thinning: one cell at 2 sdimax → N/2, D unchanged; two cells (tall / overtopped): the excess SDI removed ∝ SDI (1 − g), g = e^{−a O} W^ω",
        okA && isapprox(S8.N[ci(1, 5, 1)], h1; rtol=1e-11) && isapprox(S8.N[ci(2, 5, 1)], h2; rtol=1e-11),
        @sprintf("one cell %.6f vs %.6f | tall %.5f vs %.5f, overtopped %.5f vs %.5f (g %.4f / %.4f)", S7.N[i], N / 2, S8.N[ci(1, 5, 1)], h1, S8.N[ci(2, 5, 1)], h2, g1, g2))
end
# 0.13 [spec §8 I1] the self-thinning weight of trees that moved into a class empty at the start of the step = their source cell's realised growth
let Pz = withP(Ph; lnε=log(4.0)), Gt = withG(Gh; sdimax=300.0, a=0.0, am=0.0)     # light off: growth does not depend on sdimax; g = 0.9 for every tree
  cells = [(1, 3, 1, [(400.0, 15.0), (400.0, 17.6)], 14.0, 30.0)]
  S0 = onesite(cells); H4.simulate!(S0, [Pz], Gt; exact=true); Sg = onesite(cells); H4.simulate!(Sg, [Pz], withG(Gt; sdimax=1e15); exact=true)
  k3 = S0.N[ci(1, 3, 1)] / Sg.N[ci(1, 3, 1)]; k4 = S0.N[ci(1, 4, 1)] / Sg.N[ci(1, 4, 1)]
  check("0.13 [spec §8 I1] self-thinning: movers into a class empty at the start keep their source cell's g (the same fraction survives the thinning as the stayers)",
        Sg.N[ci(1, 4, 1)] > 0 && isapprox(k3, k4; rtol=1e-9), @sprintf("unthinned movers %.1f + stayers %.1f /ha: stayers keep %.4f %%, movers %.4f %%", Sg.N[ci(1, 4, 1)], Sg.N[ci(1, 3, 1)], 100k3, 100k4))
end
# 0.7 lane mixing with atoms: lane 1 (60 at 14 cm), lane 2 (40 at 16.5 cm), κΔ = 0.5 → lane v = own atoms ONCE × (1 − κΔ + κΔ w_v) + the other lanes' × κΔ w_v
# (audit r3 B1 (C)); lane 1 (w ½): 60 × ¾ = 45 at 14, 40 × ¼ = 10 at 16.5; lane 2 (w ¼): 60 × ⅛ = 7.5 at 14, 40 × ⅝ = 25 at 16.5; lane 3: 7.5 at 14, 5 at 16.5
let Pz = withP(Ph; lnε=-Inf)
  S8 = onesite([(1, 3, 1, [(60.0, 14.0)], 12.0, 20.0), (1, 3, 2, [(40.0, 16.5)], 13.0, 30.0)]); X8 = sim1!(S8, [Pz], withG(Gh; κ=0.5))
  hand = [[(45.0, 14.0), (10.0, 16.5)], [(7.5, 14.0), (25.0, 16.5)], [(7.5, 14.0), (5.0, 16.5)]]
  hand = [sort(h; by=x -> (x[2], x[1])) for h in hand]
  th = 60 * 12.0 + 40 * 13.0; ta = 60 * 20.0 + 40 * 30.0; Hh = [(0.5 * 60 * 12 + 0.25 * th) / 55, (0.5 * 40 * 13 + 0.125 * th) / 32.5, (0.125 * th) / 12.5]
  Ah = [(0.5 * 60 * 20 + 0.25 * ta) / 55, (0.5 * 40 * 30 + 0.125 * ta) / 32.5, (0.125 * ta) / 12.5] .+ 1
  ok = all(begin a = sort(atoms_x(X8, S8, ci(1, 3, v)); by=x -> (x[2], x[1])); length(a) == length(hand[v]) && all(isapprox(a[u][1], hand[v][u][1]; rtol=1e-13) && isapprox(a[u][2], hand[v][u][2]; rtol=1e-13) for u in eachindex(a)) end for v in 1:3) &&
       all(isapprox(S8.H[ci(1, 3, v)], Hh[v]; rtol=1e-13) && isapprox(S8.A[ci(1, 3, v)], Ah[v]; rtol=1e-13) for v in 1:3)
  S9 = onesite([(1, 3, 1, [(100.0, 15.0)], 12.0, 20.0)]; Δ=0.5); H4.simulate!(S9, [Pz], withG(Gh; κ=0.5); exact=true)
  check("0.7 [r3 B1] lane mixing κΔ: each lane's own atoms enter ONCE with weight 1 − κΔ + κΔ w_v, the other lanes' with κΔ w_v (atom D kept; H and A N-weighted); Δ = 0.5 → κΔ = 0.25",
        ok && isapprox(S9.N[ci(1, 3, 1):ci(1, 3, 3)], [87.5, 6.25, 6.25]; rtol=1e-13), "lanes " * join([string(sort(atoms_x(X8, S8, ci(1, 3, v)); by=last)) for v in 1:3], " | ") * " | Δ=0.5: $(S9.N[ci(1, 3, 1):ci(1, 3, 3)])")
end
# 0.8 overtopping with the 1 m ramp: heights 18.0 / 17.5 (r = 0.75 / 0.25), 18 / 10 (1 / 0), equal (½ / ½); own cell ½
let Pz = withP(Ph; lnε=-Inf), res = Bool[], msg = String[]
  for (h1, h2, rs, rt) in ((18.0, 17.5, 0.75, 0.25), (18.0, 10.0, 1.0, 0.0), (18.0, 18.0, 0.5, 0.5))
    X = onestep(onesite([(1, 3, 1, [(100.0, 15.31)], h1, 30.0), (2, 3, 1, [(100.0, 15.31)], h2, 30.0)]; nkk=2), [Pz, Pz], Gh); sd = 100 * (15.31 / 25.4)^1.605
    push!(res, isapprox(X.O[ci(1, 3, 1)], (0.5 + rt) * sd / 1500; rtol=1e-12) && isapprox(X.O[ci(2, 3, 1)], (0.5 + rs) * sd / 1500; rtol=1e-12))
    push!(msg, @sprintf("H %.1f/%.1f: O %.6f/%.6f vs %.6f/%.6f", h1, h2, X.O[ci(1, 3, 1)], X.O[ci(2, 3, 1)], (0.5 + rt) * sd / 1500, (0.5 + rs) * sd / 1500))
  end
  check("0.8 overtopping O_i = Σ SDI_j r(H_j − H_i)/sdimax, r(Δ) = clamp(½ + Δ/2δH, 0, 1), δH = 1 m (own ½)", all(res), join(msg, "; "))
end
# 0.14 [r5 I-2, H17] the observation's atom heights: a class-3 cell of two atoms (10 at 13.0 cm, 5 at 17.0 cm), H = 14 m, no blur (sdM = 0): the class-3 PH =
# Σ n (H + H*(D_m) − Σ n H*(D) / Σ n) = 15 H exactly; with the blur, the classes' PH still sum to 15 H
let cells = [(1, 3, 1, [(10.0, 13.0), (5.0, 17.0)], 14.0, 40.0)], Pz = withP(Ph; lnε=-Inf)
  S0 = onesite(cells); X = H4.Scratch(length(S0.N)); H4.load_site!(S0, X, 1); H4.finish_site!(S0, [Pz], withG(Gh; sdM=0.0), X, 1, emptytabs(1))
  hb = (10 * hs_(13.0) + 5 * hs_(17.0)) / 15; h1 = 14 + hs_(13.0) - hb; h2 = 14 + hs_(17.0) - hb; x3 = H4.obsix(S0, 1, 1, 3)
  ok0 = isapprox(S0.PH[x3], 10h1 + 5h2; rtol=1e-13) && isapprox(S0.PH[x3], 15 * 14; rtol=1e-13) && h2 > h1
  S1 = onesite(cells); H4.load_site!(S1, X, 1); H4.finish_site!(S1, [Pz], Gh, X, 1, emptytabs(1))
  ok1 = isapprox(sum(S1.PH[H4.obsix(S1, 1, 1, j)] for j in 1:NCLS), 15 * 14; rtol=1e-12)
  check("0.14 [r5 I-2, H17] observation heights of the atoms: H_m = H + H*(D_m) − Σ N H*(D) / Σ N (hand), centred (Σ N H_m = Σ N H) without and with the DBH blur",
        ok0 && ok1, @sprintf("PH class 3 %.10f vs hand %.10f (= 15 H %.1f); H_m %.4f / %.4f; blurred Σ PH %.10f", S0.PH[x3], 10h1 + 5h2, 15 * 14.0, h1, h2, sum(S1.PH[H4.obsix(S1, 1, 1, j)] for j in 1:NCLS)))
end
end # section 0

# ================================================================ 1. kernel unit checks
if want(1)
section("1. kernel unit checks")
let bad = Tuple{Float64,Int,Int}[], incon = 0
  for d10 in 5:600
    DIA = d10 / 10; D = DIA * 2.54; jt = DIA < 1.0 ? 0 : min(NCLS, (d10 ÷ 10 + 1) ÷ 2); j = H4.dclass(D)
    j == jt || push!(bad, (DIA, j, jt)); j == Ref4.classof(D) || (incon += 1)
  end
  check("1.1 dclass of every 0.1-in diameter 0.5–60.0 in = floor((DIA + 1)/2) (0 below 1 in, capped at 16) = the reference's search over the class bounds", isempty(bad) && incon == 0,
        isempty(bad) ? "$(incon) differ from the bound search" : "wrong: " * join(["DIA $(b[1]) → $(b[2]) (expected $(b[3]))" for b in bad[1:min(5, end)]], ", "))
end
for mode in (:sym, :fvs)
  z, w = H4.lanes(mode); m = sum(w .* z); v = sum(w .* z .^ 2) - m^2
  check("1.2 [spec §1] lanes :$mode: weights sum to 1, mean 0, variance 1", abs(sum(w) - 1) < 1e-12 && abs(m) < 1e-12 && abs(v - 1) < 1e-12, "Σw $(fmt(sum(w))) mean $(fmt(m)) var $(fmt(v)) z $(round.(z; digits=4))")
end
check("1.2b [spec §1] the store records the lanes it was built with; build lanes = HBAD4P_TRIPLE ($(LANEMODE))", (ST.lz, ST.lw) == H4.lanes(LANEMODE) && CS.lanemode == LANEMODE)
# 1.3 merge primitives (buf_add! with overflow into the nearest atom, compact! of the closest pair) vs the reference lists, random buffers incl. equal diameters
let rs = Xoshiro(11), nbad = 0, ntie = 0, mxc = 0.0, span = 0
  for trial in 1:3000
    cap = rand(rs, (4, 6, 16)); nin = rand(rs, 1:24); bn = zeros(64); bd = zeros(64); c = 0; ref = Ref4.Atoms(); Ns = 0.0; S2 = 0.0
    lo_, hi_ = 2.54 + 5.08 * rand(rs, 0:14), 0.0; hi_ = lo_ + 5.08
    Ref4.TIE[] = false
    for _ in 1:nin
      d = rand(rs) < 0.2 && c > 0 ? bd[rand(rs, 1:c)] : lo_ + 5.08 * rand(rs); n = rand(rs) < 0.1 ? 1e-9 * rand(rs) : 20 * rand(rs)
      c = H4.buf_add!(bn, bd, 0, c, cap, n, d); Ref4.add!(ref, n, d, cap); Ns += n; S2 += n * d^2
    end
    c = H4.compact!(bn, bd, 0, c); Ref4.compact!(ref, NA4)
    mxc = max(mxc, reld(sum(bn[1:c]), Ns), reld(sum(bn[1:c] .* bd[1:c] .^ 2), S2))
    all(lo_ - 1e-9 .<= bd[1:c] .<= hi_ + 1e-9) || (span += 1)
    k_ = sort(collect(zip(bn[1:c], bd[1:c])); by=last); r_ = sort(ref; by=last)
    same = length(k_) == length(r_) && all(isapprox(k_[u][1], r_[u][1]; rtol=1e-12) && isapprox(k_[u][2], r_[u][2]; rtol=1e-12) for u in eachindex(k_))
    same || (Ref4.TIE[] ? (ntie += 1) : (nbad += 1))
  end
  check("1.3 merge primitives = the reference merge lists (3000 random buffers, capacities 4 / 6 / 16, 1–24 atoms, repeated diameters, near-zero atoms; sorted adjacent merging, 1e-6 cm distance bins, duplicates merged); N and N·D² conserved; merged atoms inside the class",
        nbad == 0 && mxc < 1e-13 && span == 0, "differ (no near tie) $(nbad); near-tie cases (rounding decides) $(ntie); max rel conservation error $(fmt(mxc)); atoms outside the input span $(span)")
end
let Sx = subS(ST, SUB)
  Pz = [withP(p; lnε=-Inf, lnF=-Inf, longevity=1e12) for p in P]
  Gz = withG(G; m0=0.0, ms=0.0, mA=1e4, λ=0.0, ρ=0.0, κ=0.0, sdimax=1e15)
  dt = Sx.nyr .+ Sx.frac; bad = 0; nb = zeros(Int, 5)
  serial_run!(Sx, Pz, Gz; each=(s, X) -> for i in Sx.off[s]:Sx.off[s+1]-1
    t = (atoms_x(X, Sx, i; s=s) == atoms_of(Sx, i), reld(Sx.N[i], Sx.N0[i]) < 1e-13, reld(Sx.B[i], Sx.B0[i]) < 1e-13,
         reld(Sx.H[i], Sx.H0[i]) < 1e-13, Sx.N0[i] > 0 ? isapprox(Sx.A[i], Sx.A0[i] + dt[s]; atol=1e-9) : Sx.A[i] == Sx.A0[i])
    all(t) || (bad += 1); nb .+= .!collect(t)
  end)
  check("1.4 [spec §5] zero growth + zero mortality (+ no recruitment / thinning / mixing) leave every cell's atoms bitwise unchanged; N, B, H to 1e-13; A advances by dt", bad == 0, "$(bad) cells changed of $(length(Sx.N)) (atoms / N / B / H / A: $(nb))")
end
let Sx = subS(ST, SUB)
  Pz = [withP(p; lnF=-Inf, longevity=1e12) for p in P]; Gz = withG(G; m0=0.0, ms=0.0, mA=1e4, κ=0.0, sdimax=1e15)
  mxa = 0.0; ndown = 0
  serial_run!(Sx, Pz, Gz; each=(s, X) -> for k in 1:nk, v in 1:NL
    a = sum(Sx.N[cellix(Sx, s, k, j, v)] for j in 1:NCLS); b = sum(Sx.N0[cellix(Sx, s, k, j, v)] for j in 1:NCLS); mxa = max(mxa, abs(a - b) / max(b, 1.0))
    # no tree shrinks: the N-weighted mean of D² per (site, key, lane) does not decrease
    a2 = sum(x -> x[1] * x[2]^2, reduce(vcat, [atoms_x(X, Sx, cellix(Sx, s, k, j, v); s=s) for j in 1:NCLS]); init=0.0)
    b2 = sum(x -> x[1] * x[2]^2, reduce(vcat, [atoms_of(Sx, cellix(Sx, s, k, j, v)) for j in 1:NCLS]); init=0.0)
    a2 < b2 * (1 - 1e-13) && (ndown += 1)
  end)
  check("1.5 [spec §5] growth only: N conserved per (site, key, lane) over whole intervals (rel 1e-12), Σ N D² never decreases", mxa < 1e-12 && ndown == 0, "max rel N diff $(fmt(mxa)); Σ N D² decreased in $(ndown)")
end
let S0 = subS(ST, SUB[1:min(80, length(SUB))])
  Pz = [withP(p; lnε=-Inf, lnF=-Inf, longevity=1e12) for p in P]; Gz = withG(G; m0=0.0, ms=0.0, mA=1e4, λ=0.0, ρ=0.0, κ=0.4, sdimax=1e15)
  # make the lanes differ (N × 1 / 1.5 / 2, D and H shifted) so that the mixing has something to do
  at = Pair{Int,Vector{Tuple{Float64,Float64}}}[]; H0 = copy(S0.H0)
  for i in eachindex(S0.N0)
    S0.N0[i] > 0 || continue; f = 1.0 + 0.5 * (S0.ln[i] - 1); j = S0.cl[i]
    push!(at, i => [(n * f, clamp(d + 0.3 * (S0.ln[i] - 2), H4.CLS_LO[j], j < NCLS ? H4.CLS_HI[j] - 1e-6 : Inf)) for (n, d) in atoms_of(S0, i)]); H0[i] += S0.ln[i]
  end
  Sx = H4.make_sites4(S0.off, S0.nk, S0.sp, S0.cl, S0.ln, at, H0, S0.A0, S0.nyr, S0.frac, S0.planted, S0.W, S0.Wn, S0.T, S0.C, S0.seed0, S0.lz, S0.lw)
  mx = 0.0; nout = 0
  serial_run!(Sx, Pz, Gz; each=(s, X) -> for k in 1:nk, j in 1:NCLS
    i = cellix(Sx, s, k, j, 1); r = i:i+2; n0 = sum(Sx.N0[r]); n0 > 0 || continue; dt = Sx.nyr[s] + Sx.frac[s]
    mx = max(mx, reld(sum(Sx.N[r]), n0), reld(sum(Sx.B[r]), sum(Sx.B0[r])), reld(sum(Sx.N[r] .* Sx.H[r]), sum(Sx.N0[r] .* Sx.H0[r])), reld(sum(Sx.N[r] .* Sx.A[r]), sum(Sx.N0[r] .* (Sx.A0[r] .+ dt))))
    for t in r, a in atoms_x(X, Sx, t; s=s); H4.dclass(a[2]) == j || (nout += 1); end
  end)
  check("1.6 lane mixing conserves N, B (Σ N D²), Σ N·H, Σ N·A per (site, key, class) and keeps every atom in its class", mx < 1e-11 && nout == 0, "max rel diff $(fmt(mx)); atoms outside $(nout)")
end
function alloc_step(S, P, G, X, Δ, T_)                         # (function barrier: P, G are untyped globals here)
  reset!(S, X, 1); H4.step_site!(S, P, G, X, 1, Δ, T_); H4.finish_site!(S, P, G, X, 1, T_); reset!(S, X, 1)
  @allocated begin H4.load_site!(S, X, 1); H4.step_site!(S, P, G, X, 1, Δ, T_); H4.finish_site!(S, P, G, X, 1, T_) end
end
alloc_sim(S, P, G) = (H4.simulate!(S, P, G); @allocated H4.simulate!(S, P, G))
let s = SUB[1], Sx = subS(ST, [s]), X = H4.Scratch(ST.off[2] - ST.off[1]), T_ = H4.Tabs(P, G)
  al = alloc_step(Sx, P, G, X, 1.0, T_); al2 = alloc_step(Sx, P, G, X, 0.37, T_)
  Sy = subS(ST, SUB); alS = alloc_sim(Sy, P, G)
  check("1.7 no heap allocation in the hot loop (load_site! + step_site! + finish_site! of a real site, whole and partial year)", al == 0 && al2 == 0,
        "load + step + finish $(al) / $(al2) bytes per call | simulate! of $(length(SUB)) sites $(round(alS / 1e6; digits=2)) MB (tables + per-thread scratch)")
end
# 1.8 CAPB overflow: 15 classes × 8 atoms of one key / lane jump into the open class at once (120 atoms into a buffer of CAPB slots)
let MA = 8, cells = [(1, j, 1, [(1.0 + m, H4.CLS_LO[j] + 0.6 * m) for m in 1:MA], 10.0 + j, 30.0) for j in 1:15]
  N0 = sum(sum(first, c[4]) for c in cells); S20 = sum(sum(x -> x[1] * x[2]^2, c[4]) for c in cells)
  Gb = withG(Gh; sdimax=1e15)                                             # (no self-thinning of the giant trees)
  S = onesite(cells); X = onestep(onesite(cells), [Ph], Gb); dDs = [X.adD[(ci(1, j, 1) - 1) * NA4 + m] for j in 1:15 for m in 1:MA]
  Pb = [withP(Ph; lnε=log(150.0 / minimum(dDs)))]                         # every atom grows by ≥ 150 cm → class 16
  X = onestep(S, Pb, Gb); a = atoms_x(X, S, ci(1, 16, 1))
  grown = sum(c[4][m][1] * (c[4][m][2] + X.adD[(ci(1, c[2], 1) - 1) * NA4 + m])^2 for c in cells for m in 1:MA)
  check("1.8 CAPB overflow ($(15MA) atoms into one inflow buffer of $(H4.CAPB)): N and N·D² conserved, ≤ NATOM atoms, all in class 16", length(a) <= NA4 && isapprox(sum(first, a), N0; rtol=1e-13) &&
        isapprox(sum(x -> x[1] * x[2]^2, a), grown; rtol=1e-12) && all(x -> H4.dclass(x[2]) == 16, a) && sum(S.N[1:end-NL]) == 0, "atoms $(a)")
end
# 1.9 state invariants after a warm-start run of ALL train sites (serial: the final atoms of each site are inspected in the scratch that ran it)
let nsync = 0, nover = 0, nout = 0, nneg = 0, nfin = 0, mx = 0.0, nobs = 0, nhm = 0, mxh = 0.0
  serial_run!(ST, P, G; each=(s, X) -> begin
    for i in ST.off[s]:ST.off[s+1]-1
      at = atoms_x(X, ST, i; s=s); n = sum(first, at; init=0.0); b = sum(x -> ba_(x[1], x[2]), at; init=0.0)
      d = max(reld(n, ST.N[i]; floor=1e-12), reld(b, ST.B[i]; floor=1e-14)); mx = max(mx, d); d > 1e-10 && (nsync += 1)
      length(at) > NA4 && (nover += 1)
      for (an, ad) in at
        isfinite(an) && isfinite(ad) || (nfin += 1); an < 0 && (nneg += 1); an > 0 && H4.dclass(ad) != ST.cl[i] && (nout += 1)
      end
      isfinite(ST.H[i]) && isfinite(ST.A[i]) || (nfin += 1)
    end
    for k in 1:nk                                                     # the blur moves trees between classes but conserves them per key
      a = sum(ST.PN[H4.obsix(ST, s, k, j)] for j in 1:NCLS); b = sum(ST.N[cellix(ST, s, k, j, v)] for j in 1:NCLS for v in 1:NL); reld(a, b; floor=1e-9) > 1e-12 && (nobs += 1)
      # [r5 I-2, H17] the atoms' heights in the observation are centred: Σ_classes PH = Σ_cells N·H (the blur's p sum to 1 per atom)
      ah = sum(ST.PH[H4.obsix(ST, s, k, j)] for j in 1:NCLS); bh = sum(ST.N[cellix(ST, s, k, j, v)] * ST.H[cellix(ST, s, k, j, v)] for j in 1:NCLS for v in 1:NL if ST.N[cellix(ST, s, k, j, v)] > 0; init=0.0)
      dh = reld(ah, bh; floor=1e-9); mxh = max(mxh, dh); dh > 1e-9 && (nhm += 1)
    end
  end)
  check("1.9 state invariants after a warm-start run (all $(ns) train sites): cell N / B = Σ atoms, ≤ NATOM atoms, every atom in its cell's class, no negative / non-finite value; the observation blur conserves N per (site, key); [r5 I-2] the centred atom heights conserve Σ N·H",
        nsync == 0 && nover == 0 && nout == 0 && nneg == 0 && nfin == 0 && nobs == 0 && nhm == 0, "N/B out of sync $(nsync) (max rel $(fmt(mx))); > NATOM $(nover); atoms outside their class $(nout); negative $(nneg); non-finite $(nfin); blur not conserving N $(nobs); [r5 I-2] Σ N·H_m ≠ Σ N·H (1e-9) in $(nhm) (site, key) (max rel $(fmt(mxh)))")
end
end # section 1

# ================================================================ 2. kernel vs the brute-force reference on real sites
if want(2)
section("2. optimised kernel vs brute-force reference (train subset of $(length(SUB)) sites)")
let Sx = subS(ST, SUB), X = H4.Scratch(Sx.off[2] - Sx.off[1]), T0 = emptytabs(nk), mx = Dict(:N => 0.0, :B => 0.0, :H => 0.0, :A => 0.0, :O => 0.0, :dD => 0.0, :sv => 0.0, :atom => 0.0)
  worst = ""; nstep = 0; ntiecell = 0; nmis = 0; nmis_tie = 0; ncell = 0
  for s in 1:H4.nsites(Sx)
    r = Sx.off[s]:Sx.off[s+1]-1; reset!(Sx, X, s)
    for y in 1:Sx.nyr[s] + (Sx.frac[s] > 0)
      Δ = y <= Sx.nyr[s] ? 1.0 : Sx.frac[s]
      C = [atoms_x(X, Sx, i; s=s) for i in r]; Hh = Sx.H[r]; A = Sx.A[r]; act0 = Sx.N[r] .> 1e-9
      H4.step_site!(Sx, P, G, X, s, Δ, T0)
      rr = Ref4.step!(C, Hh, A, Sx.sp[r], Sx.cl[r], Sx.ln[r], nk, P, G, Sx.W[s], Sx.Wn[s], Sx.T[s], Sx.C[s], Sx.planted[s], Δ, lz, lw, Sx.seed0[:, s]; natom=NA4, capb=H4.CAPB)
      nstep += 1; fl = sum(Sx.N[r]); flb = sum(Sx.B[r]); ntiecell += count(rr.tie)
      for (q, i) in enumerate(r)
        Nr = Ref4.totN(C[q]); live = Nr > 1e-4; (Nr > 0 || Sx.N[i] > 0) && (ncell += 1)
        for (nm, a, b, sc) in ((:N, Sx.N[i], Nr, fl), (:B, Sx.B[i], Ref4.totB(C[q]), flb), (:H, live ? Sx.H[i] : 0.0, live ? Hh[q] : 0.0, 1e-9), (:A, live ? Sx.A[i] : 0.0, live ? A[q] : 0.0, 1e-9))
          d = nm in (:N, :B) ? abs(a - b) / sc : abs(a - b) / max(abs(a), abs(b), sc)
          d > mx[nm] && (mx[nm] = d; worst = "site $(SUB[s]) year $y cell $q $nm kernel $a ref $b")
        end
        if act0[q]
          mx[:O] = max(mx[:O], abs(X.O[q] - rr.O[q]) / max(abs(rr.O[q]), 1e-6 * rr.RD)); mx[:dD] = max(mx[:dD], abs(X.dD[q] - rr.dD[q]) / max(abs(rr.dD[q]), 1e-12))
          mx[:sv] = max(mx[:sv], abs(X.sv[q] - rr.sv[q]))
        end
        ka = sort(atoms_x(X, Sx, i; s=s); by=x -> (round(x[2]; digits=9), x[1])); ra = sort(C[q]; by=x -> (round(x[2]; digits=9), x[1]))
        same = length(ka) == length(ra) && all(abs(ka[u][1] - ra[u][1]) <= 1e-10 * fl && reld(ka[u][2], ra[u][2]) <= 1e-12 for u in eachindex(ka))
        if !same
          rr.tie[q] ? (nmis_tie += 1) : (nmis += 1; worst = "site $(SUB[s]) year $y cell $q atoms kernel $(ka) ref $(ra)")
        elseif length(ka) > 0
          mx[:atom] = max(mx[:atom], maximum(max(abs(ka[u][1] - ra[u][1]) / fl, reld(ka[u][2], ra[u][2])) for u in eachindex(ka)))
        end
      end
    end
  end
  ok21 = mx[:N] < 1e-12 && mx[:B] < 1e-12 && all(mx[k] < 1e-10 for k in (:O, :dD, :H, :A, :sv)) && nmis == 0 && mx[:atom] < 1e-12
  check("2.1 kernel (exact functions) = reference, one step at a time from the same state ($(nstep) steps, $(ncell) cell-steps): O, dD (QMD increment of the per-atom growth), survival (N-weighted per-atom), N, B, H, A per cell and every ATOM (merge choices that the reference flags as near a 1e-6 cm distance-bin boundary — decided by rounding — are counted separately)",
        ok21, "max diff " * join(["$k $(fmt(v))" for (k, v) in mx], ", ") * " | atom sets differing without a near tie $(nmis), at a near tie $(nmis_tie) (cells with a near-tie merge $(ntiecell))" * (ok21 ? "" : " | worst $worst"))
end
let Sx = subS(ST, SUB), mxN = 0.0, mxB = 0.0, mx12 = 0.0
  H4.simulate!(Sx, P, G; exact=true)
  for s in 1:H4.nsites(Sx)
    C, Hh, A = ref_run(Sx, s, P, G; natom=NA4, capb=H4.CAPB); r = Sx.off[s]:Sx.off[s+1]-1
    a = sum(Sx.N[r]); b = sum(Ref4.totN, C); mxN = max(mxN, reld(a, b)); mxB = max(mxB, reld(sum(Sx.B[r]), sum(Ref4.totB, C)))
    a12 = sum(Sx.N[i] for i in r if Sx.cl[i] >= 3); b12 = sum(Ref4.totN(C[q]) for (q, i) in enumerate(r) if Sx.cl[i] >= 3); mx12 = max(mx12, reld(a12, b12; floor=1.0))
  end
  check("2.1b free-running kernel and reference over whole intervals: per-site N, B, N ≥ 12.7 cm (merge choices at near ties may differ: rel 1e-6)", max(mxN, mxB, mx12) < 1e-6,
        "max rel diff N $(fmt(mxN)) B $(fmt(mxB)) N≥12.7 $(fmt(mx12))")
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
  check("2.3 two simulate! runs are bitwise identical (cell totals and the observation-space prediction)", Sa.N == Sb.N && Sa.B == Sb.B && Sa.H == Sb.H && Sa.A == Sb.A && Sa.PN == Sb.PN && Sa.PB == Sb.PB && Sa.PH == Sb.PH)
  check("2.3b threaded simulate! ($(Threads.nthreads()) threads) = serial site loop with one scratch, bitwise", Sa.N == Sc.N && Sa.B == Sc.B && Sa.H == Sc.H && Sa.A == Sc.A && Sa.PN == Sc.PN && Sa.PB == Sc.PB && Sa.PH == Sc.PH)
end
end # section 2

L = load4()
# ================================================================ 3. numerical diffusion of the atoms
if want(3)
section("3. atoms vs TREE-LEVEL replays and vs many atoms (numerical diffusion of the merges; the cell-level rates)")
let t1 = v1trees(L, Set(DT.iid)), sidx = Dict(iid => s for (s, iid) in enumerate(DT.iid)), kix = Dict(k => i for (i, k) in enumerate(KEYS)), dtv = Dict(r.iid => r.dt for r in eachrow(DT.iv))
  inc = zeros(nk, NCLS); cnt = zeros(nk, NCLS)
  for r in eachrow(t1); r.out == "alive" || continue; j = H4.dclass(r.dia_cm); j == 0 && continue; k = kix[r.key]; inc[k, j] += (r.dia2_cm - r.dia_cm) / dtv[r.iid]; cnt[k, j] += 1; end
  for k in 1:nk, j in 1:NCLS; inc[k, j] = cnt[k, j] >= 20 ? max(inc[k, j] / cnt[k, j], 0.0) : max(sum(inc[:, j]) / max(sum(cnt[:, j]), 1), 0.0); end
  info("3.0 observed mean annual increment (train survivors, all keys) by class [cm/yr]", join([@sprintf("%d:%.3f", j, sum(inc[:, j] .* cnt[:, j]) / max(sum(cnt[:, j]), 1)) for j in 1:NCLS], " "))
  # smooth size-dependent increment: piecewise linear in D between class midpoints (per key)
  mids = [j < NCLS ? (H4.CLS_LO[j] + H4.CLS_HI[j]) / 2 : H4.CLS_LO[NCLS] + 2.54 for j in 1:NCLS]
  incD(k, D) = D <= mids[1] ? inc[k, 1] : D >= mids[end] ? inc[k, end] : (j = searchsortedlast(mids, D); w = (D - mids[j]) / (mids[j+1] - mids[j]); (1 - w) * inc[k, j] + w * inc[k, j+1])
  trees = Dict{Tuple{Int,Int},Vector{Tuple{Float64,Float64}}}()          # (site, key) → [(w, D)] in the builder's order
  for r in eachrow(t1)
    j = H4.dclass(r.dia_cm); j == 0 && continue; s = sidx[r.iid]; push!(get!(trees, (s, kix[r.key]), Tuple{Float64,Float64}[]), (r.w_ha / (ismicro(r.w_ha) ? DT.fm1[s] : DT.f1[s]), r.dia_cm))
  end
  "Atom replay of one (site, key), single lane, no mortality: v1 atoms = the cell's trees merged to natom (as the builder), every year each cell's atoms grow by
  the increment of the cell (rate(k, j, QMD)), land in the class of their diameter, inflow lists (capb) merged to natom."
  function atom_replay(k, tr, yrs, natom, capb, rate; ward=false, sorted=false)
    C = [Ref4.Atoms() for _ in 1:NCLS]
    for (w, d) in tr; Ref4.add!(C[H4.dclass(d)], w, d, 1024); end
    for j in 1:NCLS; Ref4.compact!(C[j], natom; ward=ward, sorted=sorted); end
    for _ in 1:yrs
      nb = [Ref4.Atoms() for _ in 1:NCLS]
      for j in 1:NCLS
        isempty(C[j]) && continue; dD = rate(k, j, Ref4.qm(C[j]))
        for a in C[j]; d = a[2] + dD; t = j == NCLS ? NCLS : max(j, H4.dclass(d)); Ref4.add!(nb[t], a[1], d, capb); end
      end
      for j in 1:NCLS; Ref4.compact!(nb[j], natom; ward=ward, sorted=sorted); end
      C = nb
    end
    C
  end
  rows = String[]; worst = 0.0; worstB = 0.0; worstS = 0.0
  for (lab, trate, crate) in (("per-class increment", (k, d) -> inc[k, H4.dclass(d)], (k, j, q) -> inc[k, j]), ("smooth increment(D)", (k, d) -> incD(k, d), (k, j, q) -> incD(k, q)))
    for yrs in (5, 10, 25), (nm, natom, capb, kw) in (("NATOM $(NA4)", NA4, H4.CAPB, (;)), ("NATOM $(NA4) Ward", NA4, 10^9, (ward=true, sorted=true)), ("NATOM 64", 64, 10^9, (sorted=true,)))
      lab == "smooth increment(D)" && nm == "NATOM $(NA4) Ward" && continue
      Nt = zeros(NCLS); Bt = 0.0; Na = zeros(NCLS); Ba = 0.0; N0 = zeros(NCLS)
      for ((s, k), tr) in trees
        for (w, D) in tr; d = D; for _ in 1:yrs; d += trate(k, d); end; Nt[H4.dclass(d)] += w; Bt += ba_(w, d); N0[H4.dclass(D)] += w; end
        C = atom_replay(k, tr, yrs, natom, capb, crate; kw...)
        for j in 1:NCLS; Na[j] += Ref4.totN(C[j]); Ba += Ref4.totB(C[j]); end
      end
      errs = Float64[]
      for b in (3, 4, 6, 10)
        ct = sum(Nt[b:end]) - sum(N0[b:end]); cm = sum(Na[b:end]) - sum(N0[b:end]); push!(errs, cm / ct - 1)
      end
      if lab == "per-class increment" && nm == "NATOM $(NA4)" && yrs <= 10; worst = max(worst, maximum(abs.(errs))); worstB = max(worstB, abs(Ba / Bt - 1)); end
      if lab == "smooth increment(D)" && nm == "NATOM 64" && yrs <= 10; worstS = max(worstS, maximum(abs.(errs))); end
      push!(rows, @sprintf("%-20s %-13s %2d yr: crossing into ≥ 12.7 / 17.8 / 27.9 / 48.3 cm vs tree level %s; BA %+.3f %%; class TV %.4f", lab, nm, yrs,
                           join([@sprintf("%+.2f %%", 100e) for e in errs], " / "), 100 * (Ba / Bt - 1), 0.5 * sum(abs.(Na .- Nt)) / sum(Nt)))
    end
  end
  for r in rows; println("      ", r); end
  check("3.1 [spec §2.3, round-2 B1] atoms (NATOM $(NA4): v1 merge + inflow merges) vs the tree-level replay of the real train v1 trees, per-class increments: crossing into ≥ 12.7 / 17.8 / 27.9 / 48.3 cm within 2 % over 5 and 10 years, BA within 0.1 % (the closure was +19 %)",
        worst < 0.02 && worstB < 0.001, @sprintf("worst crossing error %.2f %%, worst BA error %.3f %% (lines above)", 100worst, 100worstB))
  info("3.1b [phys] cell-QMD growth rate vs each tree's own (smooth increment(D), merges removed with 64 atoms)", @sprintf("worst crossing error over 5–10 yr %.2f %% (lines above): the within-class size dependence of the increment is not represented", 100worstS))
end
# 3.2 the full model at the warm start (lanes, mixing, mortality, thinning, recruitment): NATOM (the kernel's rule) vs 16 / 64 atoms, Ward merges, per-cell rates.
# Stand metrics and the stand-table NLL (fixed observation parameters; the observation blur of Ref4.obs) of each variant on the same sites. Ingrowth into 12.7 cm =
# N≥12.7 of the full run − N≥12.7 of the same run with classes 1–2 emptied and no recruitment (the trees that crossed 12.7 cm and survived).
subD(idx) = (S=subS(ST, idx), n=DT.n[:, :, idx], D2=DT.D2[:, :, idx], Hs=DT.Hs[:, :, idx], nH=DT.nH[:, :, idx], fs2=DT.fs2[idx], fm2=DT.fm2[idx])
let sub = SUB, res = Dict{String,Any}(), Dx = subD(SUB), LB = "NATOM $(NA4)", KT = G.κ > 0 ? G.κ : 0.1
  Pn = [withP(p; lnF=-Inf) for p in P]; Sx = Dx.S
  nosap(s) = subS(ST, [s]; drop=i -> ST.cl[i] <= 2)                 # site s with classes 1–2 emptied (a copy)
  function metr(kw; GG=G)
    n12 = 0.0; ba = 0.0; nkj = zeros(nk, NCLS); ing = 0.0
    for (q0, s) in enumerate(sub)
      C, Hh, _ = ref_run(ST, s, P, GG; kw...); r = ST.off[s]:ST.off[s+1]-1; rx = Sx.off[q0]:Sx.off[q0+1]-1
      for (q, i) in enumerate(r)
        j = ST.cl[i]; n = Ref4.totN(C[q]); nkj[ST.sp[i], j] += n; j >= 3 && (n12 += n); ba += Ref4.totB(C[q])
        Sx.N[rx[q]] = n; Sx.B[rx[q]] = Ref4.totB(C[q]); Sx.H[rx[q]] = Hh[q]
      end
      PN, PB, PH = Ref4.obs(C, Hh, ST.sp[r], ST.cl[r], nk, GG.sdM, P, GG)
      for k in 1:nk, j in 1:NCLS; x = H4.obsix(Sx, q0, k, j); Sx.PN[x] = PN[k, j]; Sx.PB[x] = PB[k, j]; Sx.PH[x] = PH[k, j]; end
      C0 = ref_run(nosap(s), 1, Pn, GG; kw...)[1]
      ing += sum(Ref4.totN(C[q]) - Ref4.totN(C0[q]) for (q, i) in enumerate(r) if ST.cl[i] >= 3)
    end
    (n12=n12, ba=ba, nkj=nkj, ing=ing, nll=nll4(Dx, OBX))
  end
  V = ((LB, (natom=NA4, capb=H4.CAPB)), ("$(LB) Ward", (natom=NA4, ward=true, capb=10^9)), ("NATOM 32", (natom=32, capb=10^9)), ("NATOM 64", (natom=64, capb=10^9)),
       ("per-cell rates, 64", (natom=64, capb=10^9, peratom=false)))
  for (lab, kw) in V; res[lab] = metr(kw); end
  cmp(a, b) = (n12=a.n12 / b.n12 - 1, ing=a.ing / b.ing - 1, ba=a.ba / b.ba - 1, tv=0.5 * sum(abs.(a.nkj .- b.nkj)) / sum(b.nkj), c2=sum(a.nkj[:, 2]) / sum(b.nkj[:, 2]) - 1,
               c3=sum(a.nkj[:, 3]) / sum(b.nkj[:, 3]) - 1, dnll=(a.nll.total - b.nll.total) / length(sub), dc=(a.nll.count - b.nll.count) / length(sub), ds=(a.nll.size - b.nll.size) / length(sub))
  m4 = cmp(res[LB], res["NATOM 64"]); mw = cmp(res["$(LB) Ward"], res["NATOM 64"]); m16 = cmp(res["NATOM 32"], res["NATOM 64"]); mp = cmp(res["NATOM 64"], res["per-cell rates, 64"])
  fm(m) = @sprintf("N≥12.7 %+.3f %%, ingrowth into 12.7 %+.2f %%, BA %+.3f %%, class 2 %+.2f %%, class 3 %+.2f %%, key×class TV %.4f, NLL %+.3f nats/site (count %+.3f, size %+.3f)",
                   100m.n12, 100m.ing, 100m.ba, 100m.c2, 100m.c3, m.tv, m.dnll, m.dc, m.ds)
  check("3.2 [spec §2.3, r3 B1] full model at the warm start (κ = $(G.κ); $(length(sub)) train sites, whole intervals): $(LB) vs 64 atoms — stand metrics within 0.1 % (N≥12.7, BA) / 2 % (ingrowth, classes 2–3) AND the stand-table NLL within 0.02 nats per site",
        abs(m4.n12) < 1e-3 && abs(m4.ba) < 1e-3 && abs(m4.ing) < 0.02 && abs(m4.c2) < 0.02 && abs(m4.c3) < 0.02 && abs(m4.dnll) < 0.02,
        "$(LB): " * fm(m4) * " | Ward merges: " * fm(mw) * " | NATOM 32: " * fm(m16) * @sprintf(" | (64-atom ingrowth %.1f /ha per site; NLL %.1f)", res["NATOM 64"].ing / length(sub), res["NATOM 64"].nll.total))
  info("3.2b [phys] per-atom (the kernel, r3 I4) vs per-cell (QMD) growth and size-mortality (64 atoms)", "per-atom − per-cell: " * fm(mp))
  GK = withG(G; κ=KT); rk = metr((natom=NA4, capb=H4.CAPB); GG=GK); rk64 = metr((natom=64, capb=10^9); GG=GK)
  e0 = m4.dnll; e1 = (rk.nll.total - rk64.nll.total) / length(sub)
  tk = (rk64.nll.total - res["NATOM 64"].nll.total) / length(sub); ak = (rk.nll.total - res[LB].nll.total) / length(sub)
  check("3.2c [stat, r3 B1] the merge error does not depend on κ (else κ is fitted to the numerics): $(LB) − 64 NLL per site within 0.02 nats at κ = 0 AND κ = $(KT), and the apparent κ effect within ±30 % (or 0.01 nats/site, H13) of the 64-atom one",
        G.κ == 0 && abs(e0) <= 0.02 && abs(e1) <= 0.02 && abs(ak - tk) <= max(0.3 * abs(tk), 0.01),      # ([r4, H13] with an absolute floor of 0.01 nats/site: a ratio is meaningless when the true effect is ≈ 0)
        @sprintf("merge error %+.4f nats/site at κ = 0, %+.4f at κ = %.2f; κ effect (κ = %.2f vs 0): 64 atoms %+.4f nats/site, %s %+.4f (ratio %.3f)", e0, e1, KT, KT, tk, LB, ak, ak / tk))
end
# 3.3 objective roughness (audit r4 V / I1; split by likelihood term, H14; every coordinate, H20). Series: ±20 steps of 1e-4 on the FULL train set and of 1e-3 on
# HBAD4_VERIFY_KSITES (5000) train sites, for EVERY global coordinate and three key coordinates (ε of a common key PITA, of the rarest own key, of the split key
# with the smallest share), NLL with the observation parameters OBX fitted at the warm start; κ and f0 start at their lower bound by design (0 = none) and are
# stepped around 0.1 / 0.05. Per series and term: the step differences around the smooth slope (H12): sd and the largest single deviation (jump).
#   3.3  size + height (the I1 defect): sd ≤ 0.93 at 1e-4 and ≤ 2.7 at 1e-3 for EVERY coordinate (absolute bars); a jump > 2 nats at 1e-4 must be a KNOWN entry
#        of this (role, cache key) and stay ≤ its recorded value + 20 % (H15 / H20), else FAIL;
#   3.3b count (M2, accepted): a regression bar, sd and jump ≤ the recorded values + 10 % + 0.01 nats (ROUGH_REF of this (role, cache key));
#   3.3c the total: INFO (the noise CMA-ES sees per coordinate step).
# References are keyed by (role, cache key) (H20): for an unreferenced combination (another fold, a rebuilt table) the KNOWN / regression parts report INFO with
# their values (and the literal to record), the sd bars still apply. HBAD4_VERIFY_ROUGH_COORDS="b,σv,…" restricts the coordinates (a partial run).
const ROUGH_REF = Dict{Tuple{String,String},Dict{String,Any}}()          # (role, cache key) => Dict(coord => (c4sd, c4j, c3sd, c3j, sh4j))
# k1, recorded 2026-10-27 (round 5, full verifier run): count sd / jump at 1e-4 (full train), at 1e-3 (5000 sites), and the size + height 1e-4 jump. The entries
# with a size + height jump > 2 nats are the KNOWN entries of 3.3 (+20 % slack, H15 / H20): ε:PITA 3.61, b 4.03, pD 3.54, σv 3.01, βT 4.73 — the
# one-height-per-cell dynamics (an atom crossing a class bound takes its new cell's height), on the most sensitive coordinates (8.7 flags).
ROUGH_REF[("k1", "9e31afc324039360")] = Dict("ε:PITA" => (0.1766, 0.6500, 1.0687, 3.7552, 3.6100), "ε:MAVI2" => (0.0046, 0.0227, 0.0145, 0.0650, 0.2991), "ε:QUPR2" => (0.0032, 0.0154, 0.0105, 0.0631, 0.3594), "b" => (0.2433, 0.8644, 2.6134, 9.8467, 4.0267), "a" => (0.1407, 0.5323, 0.7714, 3.7272, 1.4930), "θ" => (0.0025, 0.0083, 0.0208, 0.0917, 0.0803), "ω" => (0.1105, 0.4742, 0.1931, 0.7582, 0.7493), "s" => (0.1607, 0.6644, 0.5054, 1.4920, 1.2798), "c" => (0.1708, 0.7211, 0.5900, 2.1076, 1.2887), "η" => (0.0979, 0.4328, 0.0322, 0.0846, 0.7510), "λ" => (0.0005, 0.0029, 0.0129, 0.0779, 0.0153), "μ" => (0.0005, 0.0029, 0.0129, 0.0776, 0.0152), "kh" => (0.0827, 0.4181, 0.0208, 0.0767, 0.1788), "m0" => (0.0020, 0.0109, 0.0139, 0.0771, 0.0592), "ms" => (0.0021, 0.0116, 0.0147, 0.0755, 0.0635), "σST" => (0.0020, 0.0113, 0.0136, 0.0746, 0.0652), "β" => (0.0717, 0.4343, 0.0786, 0.4618, 0.1334), "ζ" => (0.0021, 0.0114, 0.0134, 0.0768, 0.0604), "mW" => (0.0003, 0.0014, 0.0027, 0.0149, 0.0007), "pD" => (0.1993, 0.9220, 2.3873, 12.5951, 3.5351), "pH" => (0.0651, 0.3360, 0.4389, 1.4431, 1.0718), "pM" => (0.0007, 0.0036, 0.0524, 0.2092, 0.0325), "σv" => (0.6839, 3.1778, 2.9837, 14.4957, 3.0137), "am" => (0.0000, 0.0000, 0.0005, 0.0020, 0.0001), "ρ" => (0.0126, 0.0753, 0.0142, 0.0844, 0.0159), "f0" => (0.1406, 0.3953, 0.7366, 3.0027, 1.7527), "βT" => (0.6817, 2.5938, 3.2966, 11.8390, 4.7285), "κ" => (0.2592, 1.5871, 0.3564, 1.6604, 0.9163), "λ0" => (0.0737, 0.4513, 0.0191, 0.0674, 0.2607), "λ1" => (0.0764, 0.4669, 0.0333, 0.1233, 0.2574), "ε0" => (0.0766, 0.4651, 0.1586, 0.9179, 0.2647), "ωR" => (0.0021, 0.0115, 0.0077, 0.0433, 0.2531), "rM" => (0.0000, 0.0000, 0.0000, 0.0001, 0.0000), "rS" => (0.0000, 0.0000, 0.0002, 0.0004, 0.0000), "bgM" => (0.0000, 0.0000, 0.0002, 0.0004, 0.0565), "bgS" => (0.0000, 0.0000, 0.0012, 0.0024, 0.0712), "σD" => (0.0000, 0.0000, 0.0000, 0.0000, 0.0000), "σH" => (0.0000, 0.0000, 0.0000, 0.0000, 0.0000))
let ksites = parse(Int, get(ENV, "HBAD4_VERIFY_KSITES", "5000")), kidx = sort(randperm(Xoshiro(4), ns)[1:min(ksites, ns)]), ckey = build_key(DATA_DIR)
  sdv(x) = sqrt(sum((x .- sum(x) / length(x)) .^ 2) / (length(x) - 1))
  rough(v) = (dv = diff(v); Xd = hcat(ones(length(dv)), collect(1.0:length(dv))); rd = dv .- Xd * (Xd \ dv); (sd=sdv(rd), jump=maximum(abs.(rd)), mean=sum(dv) / length(dv)))
  Dk = (S=subS(ST, kidx), n=DT.n[:, :, kidx], D2=DT.D2[:, :, kidx], Hs=DT.Hs[:, :, kidx], nH=DT.nH[:, :, kidx], fs2=DT.fs2[kidx], fm2=DT.fm2[kidx])
  own = [k for k in 1:nk if WS.src[k] == KEYS[k] && WS.fac.share[k] == 1 && KEYS[k][1] != '_']; rare = own[argmin(TR.ntrain[own])]
  spl = [k for k in 1:nk if WS.src[k] != KEYS[k]]; splk = spl[argmin(WS.fac.share[spl])]
  coords = vcat(["ε:PITA", "ε:" * KEYS[rare], "ε:" * KEYS[splk]], [nm for nm in SP.names if !startswith(nm, "ε:") && !startswith(nm, "lnF:")])
  only = filter(!isempty, strip.(split(get(ENV, "HBAD4_VERIFY_ROUGH_COORDS", ""), ","))); isempty(only) || (coords = [c for c in coords if c in only])
  ref = get(ROUGH_REF, (ROLE, ckey), nothing); okh = true; okc = true; msh = String[]; msc = String[]; mst = String[]; lit = String[]; nknown = 0
  base0 = Dict("κ" => 0.1, "f0" => 0.05)
  for nm in coords
    zi = findfirst(==(nm), SP.names); zb = copy(SP.z0); haskey(base0, nm) && (zb[zi] = base0[nm]); obsonly = Symbol(nm) in OBS4
    res = Dict{Float64,Any}()
    for (sc, Dx) in ((1e-4, DT), (1e-3, Dk))
      if obsonly; Pb, Gb, _ = SP.decode(zb); H4.simulate!(Dx.S, Pb, Gb); end          # (the dynamics do not see an observation coordinate: one run)
      r = map(-20:20) do i; z = copy(zb); z[zi] += i * sc; P_, G_, O_ = SP.decode(z); obsonly || H4.simulate!(Dx.S, P_, G_); nll4(Dx, O_); end
      res[sc] = (sh=rough([x.size + x.height for x in r]), ct=rough([x.count for x in r]), tt=rough([x.total for x in r]))
    end
    s4 = res[1e-4]; s3 = res[1e-3]
    gsd = s4.sh.sd <= 0.93 && s3.sh.sd <= 2.7; jt = ""
    if s4.sh.jump > 2.0
      rr = ref === nothing ? nothing : get(ref, nm, nothing)
      if ref === nothing
        jt = " (jump > 2: INFO, unreferenced)"; gj = true
      elseif rr !== nothing && rr[5] > 2.0 && s4.sh.jump <= 1.2 * rr[5]
        jt = @sprintf(" KNOWN (≤ %.2f × 1.2)", rr[5]); gj = true; nknown += 1
      else
        jt = " ✗ (jump > 2, not a KNOWN entry within +20 %)"; gj = false
      end
    else
      gj = true
    end
    okh &= gsd && gj
    push!(msh, @sprintf("%s: sd %.3f / %.3f, jump %.2f / %.2f%s%s", nm, s4.sh.sd, s3.sh.sd, s4.sh.jump, s3.sh.jump, gsd ? "" : " ✗ sd", jt))
    if ref !== nothing && haskey(ref, nm)
      rr = ref[nm]; tol(v, r_) = v <= 1.1r_ + 0.01                 # (+10 %, with a 0.01-nat floor: the references are recorded to 4 decimals)
      gc = tol(s4.ct.sd, rr[1]) && tol(s4.ct.jump, rr[2]) && tol(s3.ct.sd, rr[3]) && tol(s3.ct.jump, rr[4]); okc &= gc
      push!(msc, @sprintf("%s: %.3f / %.2f | %.3f / %.2f (ref %.3f / %.2f | %.3f / %.2f)%s", nm, s4.ct.sd, s4.ct.jump, s3.ct.sd, s3.ct.jump, rr[1], rr[2], rr[3], rr[4], gc ? "" : " ✗"))
    else
      push!(msc, @sprintf("%s: %.3f / %.2f | %.3f / %.2f (unreferenced)", nm, s4.ct.sd, s4.ct.jump, s3.ct.sd, s3.ct.jump))
    end
    push!(mst, @sprintf("%s: %.3f / %.2f | %.3f / %.2f", nm, s4.tt.sd, s4.tt.jump, s3.tt.sd, s3.tt.jump))
    push!(lit, @sprintf("\"%s\" => (%.4f, %.4f, %.4f, %.4f, %.4f)", nm, s4.ct.sd, s4.ct.jump, s3.ct.sd, s3.ct.jump, s4.sh.jump))
    println("      3.3 ", msh[end], " | count ", msc[end]); flush(stdout)
  end
  check("3.3 [stat, r4 I1, H14, H15, H20] objective roughness of the SIZE + HEIGHT terms for $(length(coords)) coordinates: step sd ≤ 0.93 (1e-4, full train) / 2.7 (1e-3, $(length(kidx)) sites) for every coordinate; a 1e-4 jump > 2 nats only as a KNOWN entry within +20 % ($(ref === nothing ? "references: none for ($(ROLE), $(ckey)) → INFO" : "$(nknown) KNOWN in use"))",
        okh, "sd 1e-4 / 1e-3, jump 1e-4 / 1e-3: " * join(msh, " | "))
  ref === nothing ? info("3.3b [stat, M2, H20] count-term roughness (unreferenced role / cache key: INFO)", "count sd / jump at 1e-4 | 1e-3: " * join(msc, " | ")) :
                    check("3.3b [stat, M2, H14, H20] count-term roughness: no worse than the recorded values + 10 % (role $(ROLE), cache key $(ckey))", okc, join(msc, " | "))
  info("3.3c [stat, H14, H20] total objective roughness per coordinate (the noise CMA-ES sees): sd / jump at 1e-4 (full train) | 1e-3 ($(length(kidx)) sites)", join(mst, " | "))
  ref === nothing && println("      ROUGH_REF literal for (\"$(ROLE)\", \"$(ckey)\"): Dict(" * join(lit, ", ") * ")")
end
end # section 3

# ================================================================ 4. process semantics at the warm start (ST holds the warm-start run of 1.9 or is rerun here)
H4.simulate!(ST, P, G)
if want(4)
section("4. process semantics (warm start, train)")
let nout = 0, nlive = 0, nh = 0, hmax = 0.0
  for i in eachindex(ST.N)
    ST.N[i] > 1e-6 || continue; nlive += 1; j = ST.cl[i]; q = H4.qmd(ST.N[i], ST.B[i]); p = P[ST.sp[i]]
    (H4.CLS_LO[j] - 1e-9 <= q < H4.CLS_HI[j]) || (nout += 1); ST.H[i] >= 1.37 || (nh += 1); hmax = max(hmax, ST.H[i] / p.Hmax)
  end
  check("4.0 warm-start run (all train sites): every cell's QMD in its class, every height ≥ 1.37 m", nout == 0 && nh == 0,
        "QMD outside $(nout) of $(nlive) live cells; heights < 1.37 m $(nh); max H / Hmax $(round(hmax; digits=3))")
end
let nd = 0, nd0 = 0, nreal = 0
  for s in 1:ns, k in 1:nk
    n3 = sum(ST.N[cellix(ST, s, k, j, v)] for j in 3:NCLS for v in 1:NL)
    n3 > 1e-9 || continue
    n3 < H4.SEED_MIN ? (nd += 1; ST.seed0[k, s] || (nd0 += 1)) : (nreal += 1)
  end
  info("4.1 [spec §2.7] seed sources at v2 (warm start)", "key×plot with ≥ 1 tree/ha ≥ 12.7 cm: $(nreal); with 0 < N < 1 tree/ha (NOT seed sources unless a v1 seed tree): $(nd), of which $(nd0) without a v1 seed tree")
  check("4.1b [spec §2.7] seed0 = a key tree ≥ 12.7 cm on the plot at v1 (= any v1 cell of the key in classes ≥ 3 occupied)",
        all(ST.seed0[k, s] == any(ST.N0[cellix(ST, s, k, j, v)] > 0 for j in 3:NCLS for v in 1:NL) for s in 1:ns, k in 1:nk))
end
let t1 = v1trees(L, Set(DT.iid)), sidx = Dict(iid => s for (s, iid) in enumerate(DT.iid)), kix = Dict(k => i for (i, k) in enumerate(KEYS))
  # Tref = tree-weighted mean of the kernel's T over the key's TRAIN trees (all v1 trees of the key in train intervals, as traits4)
  trn = L.tr[L.tr.set .== "train", :]; Tk = zeros(nk); ck = zeros(nk); cl = CSV.read(CLIM_CSV, DataFrame; select=[:iid, :tmean_iv]); tiv = Dict(r.iid => r.tmean_iv for r in eachrow(cl))
  for r in eachrow(trn); haskey(tiv, r.iid) || continue; k = kix[r.key]; Tk[k] += tiv[r.iid]; ck[k] += 1; end
  mxT = maximum(abs.(Tk ./ ck .- TR.Tref)); okc = all(abs(H4.fT_rel(P[k], withG(G; βT=0.37), P[k].Tref) - 1) < 1e-15 for k in 1:nk)
  dT = [ST.T[sidx[r.iid]] - TR.Tref[kix[r.key]] for r in eachrow(t1)]
  check("4.2 [spec §2.2] temperature: f_T = exp(βT (T − Tref)) = 1 at Tref; Tref = the tree-weighted mean of the kernel's T (period mean annual temperature) over the key's train trees",
        okc && mxT < 1e-9, @sprintf("max |Tref − recomputed| %.2g °C | T − Tref over train v1 trees: mean %+.3f, sd %.2f, 5–95 %% %.2f … %.2f °C (βT = 0.1 → f_T %.2f … %.2f)", mxT, mean(dT), std(dT),
        quantile(dT, 0.05), quantile(dT, 0.95), exp(0.1 * quantile(dT, 0.05)), exp(0.1 * quantile(dT, 0.95))))
end
let co = [CS.sets[nm].S.C for nm in ("train", "val", "test")], allc = reduce(vcat, co), fC = 1 .+ G.βC .* log.(allc ./ H4.CREF)
  yr = [year(r.d1) + r.dt / 2 for r in eachrow(DT.iv)]
  info("4.3 [phys] CO2 term (βC fixed $(G.βC); growth only — 0.3 checks it is absent from survival and the thinning weight)",
       @sprintf("period CO2 %.1f–%.1f ppm → f_CO2 %.3f–%.3f (train mean %.4f); corr(C, interval mid-year) %.3f: a deterministic time trend imposed on growth (±4 %%), not estimated",
       minimum(allc), maximum(allc), minimum(fC), maximum(fC), mean(1 .+ G.βC .* log.(ST.C ./ H4.CREF)), cor(ST.C, yr)))
end
let rd1 = [sum(H4.sdi(ST.N0[i], H4.qmd(ST.N0[i], ST.B0[i])) for i in ST.off[s]:ST.off[s+1]-1 if ST.N0[i] > 0; init=0.0) / G.sdimax for s in 1:ns],
    rd2 = [sum(H4.sdi(ST.N[i], H4.qmd(ST.N[i], ST.B[i])) for i in ST.off[s]:ST.off[s+1]-1 if ST.N[i] > 0; init=0.0) / G.sdimax for s in 1:ns]
  info("4.3b [phys] self-thinning reach (sdimax = $(G.sdimax) /ha, plot-level SDI of the cells)", @sprintf("RD at v1: median %.2f, 99th pct %.2f, sites > 1: %d; simulated v2: median %.2f, sites ≥ 0.999: %d of %d — the thinning binds on few plots; mortality is the hazard",
       median(rd1), quantile(rd1, 0.99), count(>(1), rd1), median(rd2), count(>=(0.999), rd2), ns))
end
let msg = String[]
  for GG in (G, withG(G; κ=0.1))
    na = zeros(Int, NA4)
    serial_run!(ST, P, GG; each=(s, X) -> for i in ST.off[s]:ST.off[s+1]-1; ST.N[i] > 1e-6 && (na[length(H4.cell_atoms(X, ST, s, i))] += 1); end)
    push!(msg, "κ = $(GG.κ): " * join(["$m: $(na[m])" for m in 1:NA4], ", ") * " of $(sum(na))")
  end
  info("4.4 atoms per live cell at v2 (warm start; NATOM $(NA4))", join(msg, " | "))
  H4.simulate!(ST, P, G)
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
pf(v) = (x = [isnan(a) ? 1.0 : a for a in v]; m = mean(x); m <= 0 ? 0.0 : max(m, 0.25))     # [r4 M1] all known non-forest: 0 (no exposure)
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
                        bool_or(coalesce(tree_fate, '') LIKE 'harvest%') harv
                 FROM s.curated_trees WHERE STATECD IN (12, 13) GROUP BY ALL),
         pr AS (SELECT *, lead(md) OVER w md2, lead(trt) OVER w trt2, lead(dst) OVER w dst2, lead(harv) OVER w harv2 FROM vis WINDOW w AS (PARTITION BY st, u, c, p ORDER BY md))
    SELECT st, u, c, p, md d1, md2 d2, trt2, dst2, harv2 FROM pr WHERE md2 IS NOT NULL AND date_diff('day', md, md2) / 365.25 > 2""")
  rs = CSV.read(ROLE_SPLIT, DataFrame); setof = Dict((Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot)) => String(r.set) for r in eachrow(rs))
  nat = ivsql[.!(coalesce.(ivsql.trt2, false) .| coalesce.(ivsql.dst2, false) .| coalesce.(ivsql.harv2, false)), :]
  # [r3 I6] minus the intervals whose v2 visit has no forested subplot (every subplot's forested fraction known and 0 in FIADB SUBP_COND × COND)
  nf = [all(sb -> get(FQ, (Int(r.st), Int(r.u), Int(r.c), Int(r.p), Date(r.d2), sb), (NaN, NaN))[1] == 0, 1:4) for r in eachrow(nat)]
  # [r4 M1, H10] and those whose v1 visit has no forested subplot (no observed v1 stand)
  nf1 = [!nf[i] && all(sb -> get(FQ, (Int(r.st), Int(r.u), Int(r.c), Int(r.p), Date(r.d1), sb), (NaN, NaN))[1] == 0, 1:4) for (i, r) in enumerate(eachrow(nat))]
  nat = nat[.!(nf .| nf1), :]
  a0 = Set((Int(r.st), Int(r.u), Int(r.c), Int(r.p), Date(r.d1), Date(r.d2), get(setof, (r.st, r.u, r.c, r.p), "")) for r in eachrow(nat))
  ivb0 = DataFrame(Arrow.Table(joinpath(DATA_DIR, "intervals.arrow"))); pplots = Set(zip(ivb0.st, ivb0.u, ivb0.c, ivb0.p))
  a1 = Set(x for x in a0 if x[7] != ""); nnoset = length(a0) - length(a1)
  a = Set(x for x in a1 if (x[1], x[2], x[3], x[4]) in pplots); ndrop = length(a1) - length(a)
  b = Set((r.st, r.u, r.c, r.p, Date(r.d1), Date(r.d2), String(r.set)) for r in eachrow(ivall))
  oa = setdiff(a, b); ob = setdiff(b, a)
  check("5.1 natural GA+FL intervals (plot, d1, d2, set), dt > 2: SQL on curated_trees (no TRTCD1 / DSTRBCD1 / harvest fate at v2; [r3 I6] a forested subplot at v2) with the ROLE split, on the product's plots = the builder's three sets", isempty(oa) && isempty(ob),
        "SQL $(length(a)), builder $(length(b)); only SQL $(length(oa)), only builder $(length(ob)) | (+ $(nnoset) on plots outside the role split; $(count(nf)) natural intervals without a forested subplot at v2 excluded by SQL, builder logged $(CS.ndrop); without one at v1: $(count(nf1)) / $(CS.ndrop1))")
  dset = Dict{String,Int}(); for x in setdiff(a1, a); dset[x[7]] = get(dset, x[7], 0) + 1; end
  rsp = Set(keys(setof)); nrp = count(k -> !(k in pplots), rsp)
  info("5.1b [data] role-split plots absent from the interval product (the product's own split dropped them)", "$(nrp) of $(length(rsp)) role-split plots have no product interval; their natural intervals by SQL: $(length(a1) - length(a)) (" *
       join(["$k $v" for (k, v) in sort(collect(dset))], ", ") * ") — HBAD4P's sets are product ∩ role split (the campaign's other models share the product)")
  pl = Dict(nm => Set((r.st, r.u, r.c, r.p) for r in eachrow(CS.sets[nm].iv)) for nm in ("train", "val", "test"))
  dis = isempty(intersect(pl["train"], pl["val"])) && isempty(intersect(pl["train"], pl["test"])) && isempty(intersect(pl["val"], pl["test"]))
  check("5.2 [spec §8 I5] train / val / test disjoint at the PLOT level; every site's set = the role split $(basename(ROLE_SPLIT)) (HBAD4_SPLIT_CSV)", dis && all(get(setof, x, "") == nm for nm in keys(pl) for x in pl[nm]) && CS.split == SPLIT_CSV,
        join(["$k $(length(v)) plots / $(H4.nsites(CS.sets[k].S)) intervals" for (k, v) in pl], ", ") * " | cache built with split '$(CS.split)'")
  if isfile(FOLDS_CSV)
    fo = CSV.read(FOLDS_CSV, DataFrame); fd = Dict((Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot)) => r.fold for r in eachrow(fo))
    fs = Dict(nm => sort(unique(get(fd, x, -1) for x in pl[nm])) for nm in keys(pl))
    check("5.2b sets = whole outer folds: test one fold, val one other fold, train the remaining three (no fold shared)", length(fs["test"]) == 1 && length(fs["val"]) == 1 && length(fs["train"]) == 3 &&
          isempty(intersect(fs["test"], fs["train"], )) && isempty(intersect(fs["val"], fs["train"])) && fs["test"] != fs["val"], "folds: " * join(["$k $(v)" for (k, v) in fs], ", "))
  end
  ivb = DataFrame(Arrow.Table(joinpath(DATA_DIR, "intervals.arrow"))); ivb = ivb[inregion.(ivb.st), :]
  pset = Dict((r.st, r.u, r.c, r.p) => coalesce(r.set, "") for r in eachrow(ivb)); tab = Dict{Tuple{String,String},Int}()
  for nm in keys(pl), x in pl[nm]; kk = (get(pset, x, "NA"), nm); tab[kk] = get(tab, kk, 0) + 1; end
  info("5.2c the product's own `set` (the HBAD3P fit's train / val) → HBAD4P set, plots", join(["$(x)→$(y) $(n)" for ((x, y), n) in sort(collect(tab))], ", ") *
       " — the HBAD3P warm start was SELECTED on the product's val, i.e. on HBAD4P's val fold (+ surplus train): a mild leak into best-val selection only")
end
let c = combine(groupby(v1[(v1.set .== "train") .& (coalesce.(v1.age, 0.0) .> 0) .& (coalesce.(v1.ht, 0.0) .> 0) .& (coalesce.(v1.dia, 0.0) .> 0), :], :sym), nrow => :n)
  bigk = sort(String.(c.sym[c.n .>= MIN_SP_TREES])); sqlkeys = sort(unique(vcat(bigk, ["_" * String(g) for g in unique(v1.grp)])))
  check("5.3 species keys from SQL (train live v1 trees with age and height, ≥ $(MIN_SP_TREES)) = the builder's keys", sqlkeys == KEYS, "SQL $(join(sqlkeys, ",")) | builder $(join(KEYS, ","))")
end
let mx = 0.0, nunk = 0, nall = 0, nmin = 0, nzero = 0
  for nm in ("train", "val", "test"), (s, r) in enumerate(eachrow(CS.sets[nm].iv))
    Dn = CS.sets[nm]; mx = max(mx, maximum(abs.(FF[nm][s, :] .- [Dn.f1[s], Dn.fm1[s], Dn.fs2[s], Dn.fm2[s]])))
    for d in (r.d1, r.d2), sb in 1:4; nall += 1; isnan(get(FQ, (r.st, r.u, r.c, r.p, Date(d), sb), (NaN, NaN))[1]) && (nunk += 1); end
    minimum(FF[nm][s, :]) <= 0.25 + 1e-12 && (nmin += 1)
    f2 = [get(FQ, (r.st, r.u, r.c, r.p, Date(r.d2), sb), (NaN, NaN))[1] for sb in 1:4]; all(x -> x == 0, f2) && (nzero += 1)
  end
  check("5.4 forested fractions f1, fm1, fs2, fm2: FIADB SUBP_COND × COND (independent query) = the builder; [r3 I6] no site left whose v2 visit has no forested subplot", mx < 1e-9 && nzero == 0,
        "max abs diff $(fmt(mx)) | subplot-visits with unknown status (→ forested) $(nunk) of $(nall) | sites with a fraction at the FMIN floor 0.25: $(nmin) | v2 visits with NO forested subplot: $(nzero)")
end
# 5.5 v1 stand table, classes ≥ 3 (classes 1–2 are the empirical-Bayes layer: section 6); the lane split; 5.5b the v1 atoms
let mx = Dict(:N => 0.0, :B => 0.0, :H => 0.0, :A => 0.0), nbad = 0, ncell = 0, worst = "", nmiss = 0, nharv = 0, nnoage = 0, nlane = 0
  trn = v1[(v1.set .== "train") .& (coalesce.(v1.age, 0.0) .> 0), :]; medage = Dict(k => median(trn.age[trn.key .== k]) for k in KEYS)
  for nm in ("train", "val", "test")
    Dn = CS.sets[nm]; S = Dn.S; sid = Dict(iid => s for (s, iid) in enumerate(Dn.iid)); kix = Dict(k => i for (i, k) in enumerate(KEYS))
    acc = Dict{Tuple{Int,Int,Int},Vector{Float64}}()
    for r in eachrow(v1[v1.set .== nm, :])
      ismissing(r.s2) && (nmiss += 1; continue); startswith(coalesce(r.fate2, ""), "harvest") && (nharv += 1; continue)
      DIA = coalesce(r.dia, 0.0); DIA >= 5.0 || continue
      j = min(NCLS, floor(Int, (DIA + 1) / 2)); s = sid[r.iid]; k = kix[r.key]; FF[nm][s, 1] > 0 || continue; w = r.tpa * 2.4710538 / FF[nm][s, 1]
      a = get!(acc, (s, k, j), zeros(6)); a[1] += w; a[2] += w * π / 4e4 * (DIA * 2.54)^2
      h = coalesce(r.ht, 0.0); h > 0 && (a[3] += w * h * 0.3048; a[4] += w); ag = coalesce(r.age, 0.0); ag > 0 ? (a[5] += w * ag; a[6] += w) : (nnoage += 1)
    end
    for s in 1:H4.nsites(S), k in 1:nk, j in 1:NCLS
      i = cellix(S, s, k, j, 1); N = S.N0[i] + S.N0[i+1] + S.N0[i+2]
      N > 0 && !all(isapprox(S.N0[i+v-1], lw[v] * N; rtol=1e-12) && S.H0[i+v-1] == S.H0[i] && S.A0[i+v-1] == S.A0[i] for v in 1:3) && (nlane += 1)
      j >= 3 || continue
      a = get(acc, (s, k, j), nothing)
      (N == 0 && a === nothing) && continue; ncell += 1
      (a === nothing || N == 0) && (nbad += 1; worst = "$(nm) site $s $(KEYS[k]) class $j: builder N $N SQL $(a === nothing ? 0 : a[1])"; continue)
      B = S.B0[i] + S.B0[i+1] + S.B0[i+2]; Aref = a[6] > 0 ? a[5] / a[6] : medage[KEYS[k]]
      d = (N=reld(N, a[1]), B=reld(B, a[2]), H=a[4] > 0 ? reld(S.H0[i], a[3] / a[4]) : 0.0, A=reld(S.A0[i], Aref))
      for (kk, v) in pairs(d); mx[kk] = max(mx[kk], v); end
      (max(d.N, d.B) > 1e-5 || max(d.H, d.A) > 1e-9) && (nbad += 1; worst = "$(nm) site $s $(KEYS[k]) class $j: builder N $(round(N; digits=3)) A $(S.A0[i]) | SQL N $(round(a[1]; digits=3)) A $(Aref)")
    end
  end
  check("5.5 v1 stand table classes ≥ 3 (all sets): SQL tree list (live at v1, not missing / harvested at v2; TPA·2.4710538 / forested fraction) = the builder per (site, key, class): N, B (1e-5: the product's w_ha 14.870893 vs 6.018046·2.4710538), H, A (un-aged → the key's TRAIN median age); every cell split over the lanes by w_v with equal H, A",
        nbad == 0 && nlane == 0, "$(ncell) cells, $(nbad) differ; lane split wrong in $(nlane); max rel diff " * join(["$k $(fmt(v))" for (k, v) in mx], ", ") * (nbad > 0 ? " | e.g. $worst" : "") * " | removed: $(nmiss) missing at v2, $(nharv) harvested; un-aged v1 trees $(nnoage)")
end
let nbad = 0, ncell = 0, worst = "", mxe = 0.0, nmerged = 0, bias = [0.0, 0.0, 0.0, 0.0], nk_ = zeros(Int, NCLS), nsubsap = 0, nmicbig = 0, ntie = 0
  # the builder's v1 atoms = the cell's trees (w = w_ha / f, product row order) merged to NATOM by the documented rule, × the sapling factor (classes 1–2), × w_v per lane
  for nm in ("train", "val", "test")
    Dn = CS.sets[nm]; S = Dn.S; sid = Dict(iid => s for (s, iid) in enumerate(Dn.iid)); kix = Dict(k => i for (i, k) in enumerate(KEYS))
    t1 = v1trees(L, Set(Dn.iid)); cells = Dict{Tuple{Int,Int,Int},Vector{Tuple{Float64,Float64}}}()
    for r in eachrow(t1)
      j = H4.dclass(r.dia_cm); j == 0 && continue; s = sid[r.iid]
      (j <= 2 && !ismicro(r.w_ha)) && (nsubsap += 1); (j >= 3 && ismicro(r.w_ha)) && (nmicbig += 1)
      push!(get!(cells, (s, kix[r.key], j), Tuple{Float64,Float64}[]), (r.w_ha / (ismicro(r.w_ha) ? Dn.fm1[s] : Dn.f1[s]), r.dia_cm))
    end
    for ((s, k, j), tr) in cells
      ncell += 1; ref = Ref4.Atoms(); Ref4.TIE[] = false; for (w, d) in tr; Ref4.add!(ref, w, d, 1024); end; Ref4.compact!(ref, NA4); tie = Ref4.TIE[]
      i = cellix(S, s, k, j, 1); at = atoms_of(S, i; init=true); fN = sum(first, at) / lw[1] / sum(w for (w, _) in tr)     # (sapling classes: the EB factor)
      j >= 3 && abs(fN - 1) > 1e-12 && (nbad += 1)
      ord_(x) = (round(x[2]; digits=9), x[1])                          # (equal diameters may differ by an ulp: order by D to 1e-9, then N)
      ra = sort([(x[1] * fN * lw[1], x[2]) for x in ref]; by=ord_); ka = sort(at; by=ord_)
      same = length(ra) == length(ka) && all(reld(ra[u][1], ka[u][1]) < 1e-11 && reld(ra[u][2], ka[u][2]) < 1e-12 for u in eachindex(ka))
      same || (tie ? (ntie += 1) : (nbad += 1; worst = "$nm site $s $(KEYS[k]) class $j: builder $(ka) ref $(ra)"))
      length(tr) > NA4 && (nmerged += 1)
      # merge bias: N-weighted arithmetic mean D of the atoms vs of the trees (QMD-type merges raise E[D])
      length(tr) > NA4 && j <= 4 && (bias[j] += sum(x[1] * x[2] for x in ref) / sum(first, ref) / (sum(w * d for (w, d) in tr) / sum(w for (w, _) in tr)) - 1; nk_[j] += 1)
    end
  end
  check("5.5b v1 atoms of every (site, key, class) = the reference merge (documented rule: sorted adjacent pairs, 1e-6 cm distance bins, duplicates merged) of its trees w_ha / f (× the empirical-Bayes factor in classes 1–2) × w_v per lane (merges whose distance is near a bin boundary — decided by rounding — are counted, not failed)", nbad == 0,
        "$(ncell) cells, $(nbad) differ without a near tie, $(ntie) at a near tie, $(nmerged) merged (> $(NA4) trees); v1 trees < 12.7 cm on the subplot tally $(nsubsap), ≥ 12.7 cm on the microplot tally $(nmicbig)" * (nbad > 0 ? " | e.g. $worst" : "") *
        " | merged cells' E[D] (atoms / trees − 1) classes 1–4: " * join([@sprintf("%+.3f %%", 100bias[j] / max(nk_[j], 1)) for j in 1:4], ", "))
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
  fd = Dict((Int(r.st), Int(r.u), Int(r.c), Int(r.p), Date(r.md)) => (r.n, r.n5) for r in eachrow(fsum)); nd = 0; nd5 = 0; tot = 0; dif = 0; dif5 = 0; nnot = 0; difs = 0
  for nm in ("train", "val", "test")
    Dn = CS.sets[nm]
    for (s, r) in enumerate(eachrow(Dn.iv))
      key = (r.st, r.u, r.c, r.p, Date(r.d2)); haskey(fd, key) || (nnot += 1; continue)
      b = sum(Dn.n[:, :, s]); b5 = sum(Dn.n[:, 3:end, s]); a, a5 = fd[key]; tot += a
      a != b && (nd += 1; dif += a - b); a5 != b5 && (nd5 += 1; dif5 += a5 - b5); difs += (a - a5) - (b - b5)
    end
  end
  info("5.7 completeness of the v2 table vs FIADB TREE (live, DIA ≥ 1 in, per plot visit; the known RECONCILECD sapling exclusion is being fixed upstream)",
       "plot visits not found in FIADB $(nnot); visits with a different count: all trees $(nd) (FIADB − builder Σ = $(dif) of $(tot); saplings $(difs)), ≥ 5 in $(nd5) (Σ $(dif5)) — RERUN after the curation fix: these must drop to the dead-at-v1 / record-fix residue")
end
let ws = 6.018046 * 2.4710538, wm = 74.965282 * 2.4710538
  check("5.8 tally areas = inverse FIA expansion factors (w_ha × area = 1) and the 4-subplot / 4-microplot circle areas (24 ft, 6.8 ft radius)",
        abs(ws * AREA_SUB - 1) < 1e-12 && abs(wm * AREA_MIC - 1) < 1e-12 && abs(AREA_SUB / (4π * (24 * 0.3048)^2 / 1e4) - 1) < 3e-3 && abs(AREA_MIC / (4π * (6.8 * 0.3048)^2 / 1e4) - 1) < 3e-3,
        @sprintf("subplot %.6f ha (circles %.6f), microplot %.7f ha (circles %.7f)", AREA_SUB, 4π * (24 * 0.3048)^2 / 1e4, AREA_MIC, 4π * (6.8 * 0.3048)^2 / 1e4))
end
let cl = CSV.read(CLIM_CSV, DataFrame; select=[:iid, :st, :u, :c, :p, :tmean_iv]), wa = CSV.read(WATER_CSV, DataFrame)
  cp = Dict(r.iid => (r.st, r.u, r.c, r.p) for r in eachrow(cl)); ct = Dict(r.iid => r.tmean_iv for r in eachrow(cl)); wd = Dict(r.iid => (r.aetpet_iv, r.aetpet_n) for r in eachrow(wa))
  nbad = 0; nmis = 0; mx = 0.0
  for nm in ("train", "val", "test"), (s, r) in enumerate(eachrow(CS.sets[nm].iv))
    get(cp, r.iid, nothing) == (r.st, r.u, r.c, r.p) || (nbad += 1)
    (haskey(wd, r.iid) && haskey(ct, r.iid) && !ismissing(wd[r.iid][1]) && !ismissing(ct[r.iid])) || (nmis += 1; continue)
    S = CS.sets[nm].S; mx = max(mx, abs(S.W[s] - wd[r.iid][1]), abs(S.Wn[s] - wd[r.iid][2]), abs(S.T[s] - ct[r.iid]))
  end
  check("5.9 climate: every site's iid maps to its own plot in the climate table; W, Wn, T = the tables; none missing", nbad == 0 && nmis == 0 && mx < 1e-12,
        "plot mismatches $(nbad), missing $(nmis), max abs diff $(fmt(mx))")
  co = CSV.read(CO2_CSV, DataFrame); cd = Dict(Int(r.year) => Float64(r.co2_ppm) for r in eachrow(co)); mxc = 0.0
  for nm in ("train", "val", "test"), (s, r) in enumerate(eachrow(CS.sets[nm].iv))
    mxc = max(mxc, abs(CS.sets[nm].S.C[s] - mean(cd[y] for y in year(r.d1):max(year(r.d2) - 1, year(r.d1)))))
  end
  check("5.9b CO2 = mean of the annual Mauna Loa series over year(d1) … year(d2) − 1 (the climate period of interval_climate)", mxc < 1e-9, "max abs diff $(fmt(mxc)) ppm")
  dtok = all(abs(CS.sets[nm].S.nyr[s] + CS.sets[nm].S.frac[s] - CS.sets[nm].iv.dt[s]) < 1e-12 && 0 <= CS.sets[nm].S.frac[s] < 1 for nm in ("train", "val", "test") for s in 1:H4.nsites(CS.sets[nm].S))
  check("5.9c simulated span nyr + frac = dt of the interval (dates)", dtok)
end
let t = v1[(v1.set .== "train"), :], st_ = CSV.read(SHADETOL_CSV, DataFrame), lo = CSV.read(LONGEVITY_CSV, DataFrame), cl = CSV.read(CLIM_CSV, DataFrame; select=[:iid, :tmean_iv])
  std_ = Dict(String(r.sym) => Float64(r.shade_class) for r in eachrow(st_)); lod = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lo)); tiv = Dict(r.iid => r.tmean_iv for r in eachrow(cl))
  mx = Dict(:Hmax => 0.0, :ST => 0.0, :longevity => 0.0, :Tref => 0.0)
  for (k, key) in enumerate(KEYS)
    tk = t[(t.key .== key) .& (coalesce.(t.dia, 0.0) .> 0), :]; hs = 0.3048 .* Float64.(collect(skipmissing(tk.ht[coalesce.(tk.ht, 0.0) .> 0])))
    c = combine(groupby(tk, :sym), nrow => :n)
    mx[:Hmax] = max(mx[:Hmax], reld(quantile(hs, 0.995), TR.Hmax[k])); mx[:ST] = max(mx[:ST], reld(sum(get(std_, s, 3.0) * n for (s, n) in zip(c.sym, c.n)) / sum(c.n), TR.ST[k]))
    mx[:longevity] = max(mx[:longevity], reld(sum(get(lod, s, 300.0) * n for (s, n) in zip(c.sym, c.n)) / sum(c.n), TR.longevity[k]))
    tt = tk[haskey.(Ref(tiv), tk.iid), :]; mx[:Tref] = max(mx[:Tref], abs(mean(tiv[i] for i in tt.iid) - TR.Tref[k]))
  end
  check("5.10 traits from the SQL tree list (TRAIN only) = TR: Hmax (99.5 % height), ST and LONGEVITY tree-weighted over member species, T_ref (tree-weighted train period mean)",
        mx[:Hmax] < 1e-9 && mx[:ST] < 1e-9 && mx[:longevity] < 1e-9 && mx[:Tref] < 1e-3, join(["$k $(fmt(v))" for (k, v) in mx], ", ") * " (T_ref: SQL live v1 trees vs the product's tree list)")
end
let f = joinpath(@__DIR__, "data4.jl"), srcd = read(f, String), h0 = build_key(DATA_DIR)
  # [r4] every one-byte edit of the builder at 300 positions must change the hash build_key takes of it (Base.hash of a byte vector samples its elements)
  v = Vector{UInt8}(srcd); hs = hash(srcd); P_ = round.(Int, range(1, length(v); length=300))
  nmiss = count(p -> (w = copy(v); w[p] = w[p] == UInt8('1') ? UInt8('2') : UInt8('1'); hash(String(w)) == hs), P_)
  bk = srcd[findfirst("function build_key", srcd)[1]:findfirst("function cached_sets", srcd)[1]]; strh = occursin("read(joinpath(@__DIR__, \"data4.jl\"), String)", bk) && occursin("HBAD4P.jl\"), String)", bk)
  blk = bk
  deps = all(occursin(x, blk) for x in ("HBAD4P.jl", "REGION", "MIN_SP_TREES", "LANEMODE", "FMIN", "NATOM", "SPLIT_CSV", "WATER_CSV", "CLIM_CSV", "FOREST_CSV", "CO2_CSV", "SHADETOL_CSV", "LONGEVITY_CSV"))
  check("5.11 [r4] cache key covers builder + kernel code (every one-byte edit changes it: the files are hashed as Strings), the switches, the role split and every input file", nmiss == 0 && strh && deps,
        "one-byte edits at 300 positions left the hash unchanged: $(nmiss); code hashed as Strings: $(strh) (Julia $(VERSION); Base.hash is not stable across Julia versions: a new Julia rebuilds the cache); inputs and switches listed: $(deps)")
end
end # section 5

# ================================================================ 6. empirical-Bayes sapling prior
if want(6)
section("6. empirical-Bayes sapling layer (classes 1–2)")
EBR = let trn = ivall[ivall.set .== "train", :], sidx = Dict(iid => s for (s, iid) in enumerate(sort(trn.iid))), kix = Dict(k => i for (i, k) in enumerate(KEYS)), nst = nrow(trn)
  fm = zeros(nst); for r in eachrow(trn); s = sidx[r.iid]; fm[s] = pf([get(FQ, (r.st, r.u, r.c, r.p, Date(r.d1), sb), (NaN, NaN))[2] for sb in 1:4]); end
  n = zeros(Int, nk, 2, nst); adult = falses(nk, nst)
  for r in eachrow(v1[v1.set .== "train", :])
    (ismissing(r.s2) || startswith(coalesce(r.fate2, ""), "harvest")) && continue
    s = sidx[r.iid]; k = kix[r.key]; DIA = coalesce(r.dia, 0.0); DIA >= 5.0 && (adult[k, s] = true)
    (1.0 <= DIA < 5.0) && (n[k, DIA < 3.0 ? 1 : 2, s] += 1)
  end
  e = AREA_MIC .* fm; m = zeros(nk, 2, 2); α = fill(Inf, nk, 2, 2); llk = zeros(nk, 2, 2); llr = zeros(nk, 2, 2); llp = zeros(nk, 2, 2); ng = zeros(Int, nk, 2, 2)
  # (counts y with exposures x as distinct (y, x) pairs with multiplicities; NB log-likelihood by Distributions)
  nbll(U, mg, r) = sum(c * Distributions.logpdf(Distributions.NegativeBinomial(r, r / (r + mg * xi)), Int(yi)) for ((yi, xi), c) in U)
  function gold(f, lo, hi, it)                                      # maximiser of a unimodal f on [lo, hi]
    for _ in 1:it; a1 = lo + 0.382 * (hi - lo); a2 = lo + 0.618 * (hi - lo); f(a1) > f(a2) ? (hi = a2) : (lo = a1); end
    (lo + hi) / 2
  end
  for k in 1:nk, j in 1:2, a in (false, true)
    idx = findall(i -> adult[k, i] == a && e[i] > 0, 1:nst); isempty(idx) && continue; y = n[k, j, idx]; x = e[idx]; ng[k, j, a+1] = length(idx)   # ([r4 M1] unobserved microplots excluded)
    mg = sum(y) / sum(x); mg > 0 || continue
    U = Dict{Tuple{Int,Float64},Int}(); for (yi, xi) in zip(y, x); U[(yi, xi)] = get(U, (yi, xi), 0) + 1; end; U = collect(U)
    # [r3] joint maximum likelihood of (m, α): golden section on ln α over [−8, 8] of the profile max over ln m (golden on ln m ± 3 around the Poisson m)
    pr(lα) = (lm = gold(z -> nbll(U, exp(z), exp(lα)), log(mg) - 3, log(mg) + 3, 60); nbll(U, exp(lm), exp(lα)))
    la = gold(pr, -8.0, 8.0, 70); lm = gold(z -> nbll(U, exp(z), exp(la)), log(mg) - 3, log(mg) + 3, 60)
    m[k, j, a+1] = exp(lm); α[k, j, a+1] = exp(la); llr[k, j, a+1] = nbll(U, exp(lm), exp(la)); llk[k, j, a+1] = nbll(U, CS.eb.m[k, j, a+1], CS.eb.α[k, j, a+1])
    llp[k, j, a+1] = nbll(U, mg, exp(gold(g -> nbll(U, mg, exp(g)), -8.0, 8.0, 70)))       # the round-2 rule: m = Σn/Σe, α profiled
  end
  (m=m, α=α, llk=llk, llr=llr, llp=llp, ng=ng, n=n, e=e, adult=adult)
end
let dm = maximum(reld(EBR.m[i], CS.eb.m[i]) for i in eachindex(EBR.m)), dll = maximum(EBR.llr .- EBR.llk), dα = maximum(abs(log(EBR.α[i]) - log(CS.eb.α[i])) for i in eachindex(EBR.α) if isfinite(EBR.α[i]))
  check("6.1 [r3] sapling prior refitted from SQL (TRAIN plots only; groups key × class × adult on the plot): the joint maximum-likelihood (m, α) of the NB (independent optimiser) = the cached prior",
        dm < 1e-4 && dll < 1e-6, "max rel diff m $(fmt(dm)); cached (m, α) within $(fmt(dll)) nats of the refit's maximum (max |Δ ln α| $(fmt(dα))); α range $(fmt(minimum(CS.eb.α))) – $(fmt(maximum(filter(isfinite, CS.eb.α))))")
  info("6.1b [stat] joint (m, α) vs the round-2 rule m = Σn/Σe (the Poisson MLE) with α profiled", @sprintf("summed log-likelihood gain %.2f nats over all %d groups (refit) | builder's own gain %.2f nats", sum(EBR.llr .- EBR.llp), count(>(0), EBR.ng), CS.eb.gain))
  nb = count(i -> isfinite(EBR.α[i]) && (log(EBR.α[i]) < -7.9 || log(EBR.α[i]) > 7.9), eachindex(EBR.α))
  info("6.1c [stat] NB size α at the search bound e^{±8}", "$(nb) of $(count(isfinite, EBR.α)) groups")
end
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
      Nh = mg > 0 ? (n[k, j, s] + α) / (e + α / mg) : (e > 0 ? n[k, j, s] / e : 0.0)
      i = cellix(S, s, k, j, 1); Nb = S.N0[i] + S.N0[i+1] + S.N0[i+2]; d = reld(Nb, Nh; floor=1e-9)
      d > mx && (mx = d; worst = "$nm site $s $(KEYS[k]) class $j n $(n[k, j, s]): builder $(Nb), refit $(Nh)")
      d > 1e-4 && (nbad += 1)
    end
  end
  check("6.2 v1 sapling density of every (site, key, class 1–2) = posterior mean (n + α)/(e + α/m) from the refit", nbad == 0, "max rel diff $(fmt(mx)); cells > 1e-4: $(nbad)" * (nbad > 0 ? " | $worst" : ""))
end
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
let res = Dict{String,Float64}(), S = DT.S
  surv = 0; newm = 0
  for r in eachrow(L.tr[in.(L.tr.iid, Ref(Set(DT.iid))), :]); (r.out == "alive" && r.dia_cm < 12.7 && r.dia2_cm < 12.7) && (surv += 1); end
  for r in eachrow(L.ig[in.(L.ig.iid, Ref(Set(DT.iid))), :]); r.dia_cm < 12.7 && (newm += 1); end
  raw = zeros(nk, 2, ns)
  t1 = v1trees(L, Set(DT.iid)); sidx = Dict(iid => s for (s, iid) in enumerate(DT.iid)); kix = Dict(k => i for (i, k) in enumerate(KEYS))
  for r in eachrow(t1); j = H4.dclass(r.dia_cm); j in (1, 2) || continue; raw[kix[r.key], j, sidx[r.iid]] += r.w_ha / DT.fm1[sidx[r.iid]]; end
  function cnll(Nfun, rM, bg)
    tot = 0.0
    for s in 1:ns, k in 1:nk, j in 1:2; μ = (Nfun(s, k, j) + bg) * DT.fm2[s] * AREA_MIC; tot -= nb_logpmf(Float64(DT.n[k, j, s]), μ, rM); end
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
"""Background densities of the size / height terms recomputed from their definition (docs §3): per class j the mean / sd of log QMD over the TRAIN v2 cells with
≥ 1 tree; per key × class the mean log mean height (≥ 10 cells, else the class mean of all such cells) with the per-class sd around the key means."""
function bg_ref(D)
  nk = size(D.n, 1); mD = zeros(NCLS); sDb = zeros(NCLS); mH = zeros(nk, NCLS); sHb = zeros(NCLS)
  for j in 1:NCLS
    q = Float64[]; hs = [Float64[] for _ in 1:nk]
    for s in axes(D.n, 3), k in 1:nk
      D.n[k, j, s] > 0 && push!(q, 0.5 * log(D.D2[k, j, s] / D.n[k, j, s])); D.nH[k, j, s] > 0 && push!(hs[k], log(D.Hs[k, j, s] / D.nH[k, j, s]))
    end
    mD[j] = mean(q); sDb[j] = std(q); allh = reduce(vcat, hs); mc = mean(allh)
    for k in 1:nk; mH[k, j] = length(hs[k]) >= 10 ? mean(hs[k]) : mc; end
    sHb[j] = sqrt(sum(sum(abs2, hs[k] .- mH[k, j]; init=0.0) for k in 1:nk) / length(allh))
  end
  (mD=mD, sDb=sDb, mH=mH, sHb=sHb)
end
"""Independent NLL (docs §3): the final atoms of every site (serial kernel run, read from the scratch), the DBH-error blur and the atoms' allometric heights
recomputed over ALL classes (Ref4.obs, Distributions normal cdf); Distributions NB with background for the count; the size / height terms of EVERY observed
cell of a tallied area as the two-component mixture w · model Gaussian + (1 − w) · background Gaussian, w = N / (N + bg), with the background densities
recomputed (bg_ref). Unobserved tallies (f = 0) score nothing."""
function nll_ref(D, Ob, P, G, BGr; exact=true)
  S = D.S; c = 0.0; dsz = 0.0; hh = 0.0; cempty = 0.0; nempty = 0; bgexp = 0.0
  lmix(w, la, lb) = (w == 0 ? lb : (A = log(w) + la; B = log(1 - w) + lb; M = max(A, B); M + log(exp(A - M) + exp(B - M))))
  serial_run!(S, P, G; exact=exact, each=(s, X) -> begin
    r = S.off[s]:S.off[s+1]-1; PN, PB, PH = Ref4.obs([atoms_x(X, S, i; s=s) for i in r], S.H[r], S.sp[r], S.cl[r], S.nk, G.sdM, P, G)
    for k in 1:S.nk, j in 1:NCLS
      f = j >= 3 ? D.fs2[s] : D.fm2[s]; f > 0 || continue
      Np = PN[k, j]; n = D.n[k, j, s]
      area = j >= 3 ? 1 / (6.018046 * 2.4710538) : 1 / (74.965282 * 2.4710538)
      bg = j >= 3 ? Ob.bgS : Ob.bgM; μ = (Np + bg) * f * area; rr = j >= 3 ? Ob.rS : Ob.rM
      l = -Distributions.logpdf(Distributions.NegativeBinomial(rr, rr / (rr + μ)), n); c += l; bgexp += bg * f * area
      (Np < 1e-3 && n > 0) && (cempty += l; nempty += n)
      n > 0 || continue
      w = Np / (Np + bg); lq = log(sqrt(D.D2[k, j, s] / n)); wn(m) = 1 - (1 - w)^m      # (the model component's weight of a mean of m trees, r5 M-1)
      lm = Np > 0 && PB[k, j] > 0 ? Distributions.logpdf(Distributions.Normal(log(sqrt(PB[k, j] / (π / 4e4 * Np))), sqrt(Ob.sD[j]^2 / n + Ob.σD^2)), lq) : -Inf   # (a 1e-300 tail: PB underflows)
      dsz -= lmix(wn(n), lm, Distributions.logpdf(Distributions.Normal(BGr.mD[j], BGr.sDb[j]), lq))
      if D.nH[k, j, s] > 0
        lh = log(D.Hs[k, j, s] / D.nH[k, j, s])
        lm = Np > 0 && PH[k, j] > 0 ? Distributions.logpdf(Distributions.Normal(log(PH[k, j] / Np), sqrt(Ob.sH[j]^2 / D.nH[k, j, s] + Ob.σH^2)), lh) : -Inf
        hh -= lmix(wn(D.nH[k, j, s]), lm, Distributions.logpdf(Distributions.Normal(BGr.mH[k, j], BGr.sHb[j]), lh))
      end
    end
  end)
  (total=c + dsz + hh, count=c, size=dsz, height=hh, cempty=cempty, nempty=nempty, bgexp=bgexp)
end
OBW = OBX                                                             # (fitted at the top, at the warm start)
BGR = bg_ref(DT)
let _ = H4.simulate!(DT.S, P, G; exact=true), a = nll4(DT, OBW), b = nll_ref(DT, OBW, P, G, BGR)
  check("7.2 [r4 I1] nll4 (threaded) = independent recomputation (the final atoms blurred by the DBH error over all classes with the Distributions normal cdf, allometric atom heights; Distributions NB with background; the size / height terms of EVERY observed cell as the mixture w N(model) + (1 − w) N(background), w = N / (N + bg), background recomputed) on all $(ns) train sites after a warm-start run (exact functions)",
        reld(a.total, b.total) < 1e-7 && reld(a.count, b.count) < 1e-7 && reld(a.size, b.size) < 1e-9 && reld(a.height, b.height) < 1e-9,
        @sprintf("nll4 %.3f (count %.1f size %.1f height %.1f) | ref %.3f", a.total, a.count, a.size, a.height, b.total))
  ntr = sum(DT.n)
  info("7.3 [stat] fitted observation parameters at the warm start", @sprintf("rM %.3g rS %.3g bgM %.3g bgS %.3g /ha σD %.3g σH %.3g | background: expected %.1f trees over the whole train table (%.2f %% of the %d tallied); the count NLL of trees in cells the model leaves (near) empty (N < 1e-3 /ha): %.0f nats for %d trees (%.1f %% of the count NLL)",
       OBW.rM, OBW.rS, OBW.bgM, OBW.bgS, OBW.σD, OBW.σH, b.bgexp, 100b.bgexp / ntr, ntr, b.cempty, b.nempty, 100b.cempty / b.count))
  check("7.3b [stat] background densities and spreads are not at their bounds", 1.001e-3 < OBW.bgM < 499 && 1.001e-4 < OBW.bgS < 49 && 1.001e-5 < OBW.σD < 0.499 && 1.001e-5 < OBW.σH < 0.999,
        @sprintf("bgM %.4g (bounds 1e-3–500), bgS %.4g (1e-4–50), σD %.4g, σH %.4g", OBW.bgM, OBW.bgS, OBW.σD, OBW.σH))
end
let wsd = within_sd(DT), sD = wsd[1], sH = wsd[2]
  check("7.4 within-cell spreads sD, sH = recomputation from the TRAIN v2 cells (cells ≥ 2 trees)", maximum(abs.(sD .- CS.sD)) < 1e-12 && maximum(abs.(sH .- CS.sH)) < 1e-12)
  B = CS.bgobs; dmx = maximum(maximum(abs.(getfield(B, f) .- getfield(BGR, f))) for f in (:mD, :sDb, :mH, :sHb))
  check("7.4c [r4 I1] background densities of the size / height mixture (TRAIN, data constants) = recomputation from their definition", dmx < 1e-12,
        @sprintf("max abs diff %.2g | log QMD class 3: mean %.3f sd %.3f; log H sd by class %s", dmx, B.mD[3], B.sDb[3], join(round.(B.sHb; digits=3), " ")))
  df = [sum(max(n - 1, 0) for n in view(DT.n, :, j, :) if n >= 2; init=0) for j in 1:NCLS]
  # exact sd of log D for D uniform on a class [L, U) (the physical fallback of classes with < 30 df): ∫ log² − (∫ log)²
  usd(L, U) = (m1 = (U * log(U) - U - L * log(L) + L) / (U - L); m2 = (U * (log(U)^2 - 2log(U) + 2) - L * (log(L)^2 - 2log(L) + 2)) / (U - L); sqrt(m2 - m1^2))
  sdmax = [j < NCLS ? log(H4.CLS_HI[j] / H4.CLS_LO[j]) / 2 : Inf for j in 1:NCLS]
  bad = [j for j in 1:NCLS-1 if CS.sD[j] > sdmax[j] + 1e-12]
  fb = [j for j in 1:NCLS if df[j] < 30]
  check("7.4b [stat] every class's sD ≤ the largest sd possible inside the class (log U/L / 2); classes with < 30 df use the uniform-spread value", isempty(bad) &&
        all(j -> j == NCLS || isapprox(CS.sD[j], usd(H4.CLS_LO[j], H4.CLS_HI[j]); rtol=0.01), fb),
        "df per class " * join(["$j:$(df[j])" for j in 1:NCLS], " ") * " | fallback classes $(fb): sD " * join([@sprintf("%.4f (uniform exact %.4f)", CS.sD[j], j < NCLS ? usd(H4.CLS_LO[j], H4.CLS_HI[j]) : NaN) for j in fb], ", ") *
        " | the height spread sH of classes with < 30 df pools the neighbouring classes (" * join(["$j: $(round(CS.sH[j]; digits=4))" for j in 1:NCLS if sum(max(n - 1, 0) for n in view(DT.nH, :, j, :) if n >= 2; init=0) < 30], ", ") * ")")
end
let t1 = v1trees(L, Set(DT.iid)), sidx = Dict(iid => s for (s, iid) in enumerate(DT.iid)), kix = Dict(k => i for (i, k) in enumerate(KEYS))
  acc = Dict{Tuple{Int,Int,Int},Vector{Float64}}()
  for r in eachrow(t1)
    r.out == "alive" || continue; j1 = H4.dclass(r.dia_cm); j1 >= 3 || continue; j2 = H4.dclass(r.dia2_cm); j2 == j1 || continue
    a = get!(acc, (sidx[r.iid], kix[r.key], j1), zeros(3)); a[1] += 1; a[2] += r.dia_cm^2; a[3] += r.dia2_cm^2
  end
  res = Float64[]; pred = Float64[]
  for ((s, k, j), a) in acc
    a[1] >= 2 && DT.n[k, j, s] == a[1] || continue
    push!(res, 0.5 * log(a[3] / a[2])); push!(pred, CS.sD[j]^2 / a[1])
  end
  info("7.5 [stat] size-term variance vs the actual v1→v2 change of the same trees (round 2 I4, kept)", @sprintf("%d cells (classes ≥ 3) whose v2 trees are exactly their v1 stayers: var(log QMD2 − log QMD1) = %.3g (incl. the mean growth shift) vs mean sampling variance sD²/n %.3g (ratio %.3f)",
       length(res), var(res), mean(pred), var(res) / mean(pred)))
end
end # section 7

# ================================================================ 8. warm start and parameter space
if want(8)
section("8. warm start / parameter space")
check("8.1 fixed globals as documented (gR 0, mA 10, sdimax 1500, βC $(FIX_DEFAULT[:βC]), DBH error sdM 0.2 cm)", G.gR == 0 && isapprox(G.mA, 10) && isapprox(G.sdimax, 1500) && G.βC == FIX_DEFAULT[:βC] && G.sdM == 0.2)
let t1 = L.tr[(L.tr.set .== "train") .& .!in.(L.tr.out, Ref(("missing", "harvest"))), :], FT = forest4(), ivd = Dict(r.iid => r for r in eachrow(L.iv))
  # HBAD3P's subplot competition (tools/hbad3/lib3p.jl subplot_cohorts, farea = 1): N = 4 Σ w / a, a = max(f_subplot(v1), FMIN) (1 if unknown); RD = Σ N (D/25.4)^1.605 / sdimax
  asub(iid, sb) = (r = ivd[iid]; f = get(FT, (r.st, r.u, r.c, r.p, r.d1, sb), (NaN, NaN))[1]; (isnan(f) || f <= 0) ? 1.0 : max(f, FMIN))
  g = combine(groupby(DataFrame(iid=t1.iid, sb=t1.sb, x=4 .* t1.w_ha .* (t1.dia_cm ./ 25.4) .^ 1.605), [:iid, :sb]), :x => sum => :x)
  rd0 = Dict((r.iid, r.sb) => r.x / 1500 for r in eachrow(g)); rda = Dict((r.iid, r.sb) => r.x / asub(r.iid, r.sb) / 1500 for r in eachrow(g))
  cr = [ismissing(c) || c <= 0 ? 0.4 : c / 100 for c in t1.cr]
  f0 = sum(t1.w_ha .* exp.(gl3[:gR] .* [rd0[(i, b)] for (i, b) in zip(t1.iid, t1.sb)]) .* (cr ./ 0.4) .^ gl3[:φ]) / sum(t1.w_ha)
  fa = sum(t1.w_ha .* exp.(gl3[:gR] .* [rda[(i, b)] for (i, b) in zip(t1.iid, t1.sb)]) .* (cr ./ 0.4) .^ gl3[:φ]) / sum(t1.w_ha)
  global FAC3 = fa
  check("8.2 [spec §8 I2, r3 I2e] crowding / crown factor of the ε rescaling = the tree-weighted mean of HBAD3P's dropped e^{gR RD_subplot} (CR/0.4)^φ at the train v1 state, RD_subplot as HBAD3P computes it (densities per FORESTED subplot ha)",
        abs(fa / WS.fac.F - 1) < 0.01, @sprintf("used %.4f | recomputed with HBAD3P's RD (÷ max(f_subplot, 0.25)) %.4f (%+.2f %%) | without the forested-area division %.4f", WS.fac.F, fa, 100 * (fa / WS.fac.F - 1), f0))
end
# 8.2b HBAD3P computed the light (overtopping) per SUBPLOT (densities × 4 / forested fraction, taller cohorts full, equal ½, own ½); HBAD4P per PLOT (ramp δH = 1 m)
let X = H4.Scratch(ST.off[2] - ST.off[1]), T0 = emptytabs(nk), FT = forest4(), ivd = Dict(r.iid => r for r in eachrow(L.iv)), kix = Dict(k => i for (i, k) in enumerate(KEYS))
  sidx = Dict(DT.iid[s] => s for s in SUB); t1 = v1trees(L, Set(keys(sidx))); t1 = t1[H4.dclass.(t1.dia_cm) .>= 1, :]
  Op = Dict{Tuple{Int,Int,Int},Float64}()
  for s in SUB
    Sx = subS(ST, [s]); H4.load_site!(Sx, X, 1); H4.step_site!(Sx, P, G, X, 1, 1.0, T0)
    for k in 1:nk, j in 1:NCLS; i = cellix(Sx, 1, k, j, 1); Sx.N0[i] > 1e-9 && (Op[(s, k, j)] = X.O[i - Sx.off[1] + 1]); end
  end
  asub(iid, sb) = (r = ivd[iid]; f = get(FT, (r.st, r.u, r.c, r.p, r.d1, sb), (NaN, NaN))[1]; (isnan(f) || f <= 0) ? 1.0 : max(f, FMIN))
  num3 = 0.0; num4 = 0.0; den = 0.0; m3 = 0.0; m4 = 0.0
  for g in groupby(t1, [:iid, :sb])
    a_ = asub(g.iid[1], g.sb[1]); sd = 4 .* g.w_ha .* (g.dia_cm ./ 25.4) .^ 1.605 ./ a_; h = g.ht_m; s = sidx[g.iid[1]]
    for i in 1:nrow(g)
      o = 0.5 * sd[i]; for j in 1:nrow(g); j == i && continue; h[j] > h[i] && (o += sd[j]); h[j] == h[i] && (o += 0.5 * sd[j]); end
      w = g.w_ha[i]; o4 = Op[(s, kix[g.key[i]], H4.dclass(g.dia_cm[i]))]
      num3 += w * exp(-gl3[:a] * o / 1500); num4 += w * exp(-G.a * o4); m3 += w * exp(-gl3[:am] * o / 1500); m4 += w * exp(-G.am * o4); den += w
    end
  end
  global LMR = m3 / m4
  check("8.2b [r3 I2c] warm-start growth level: HBAD3P's subplot light vs HBAD4P's plot light at the v1 state (tree-weighted mean of e^{−a O}, $(length(SUB)) train sites) × the light factor applied to ε within 5 %",
        abs(WS.fac.Lr * num4 / num3 - 1) < 0.05, @sprintf("subplot (HBAD3P) %.4f, plot (HBAD4P) %.4f: ratio %.4f; ε light factor applied %.4f (all train sites) → growth level ratio %.4f | mortality light e^{−am O}: %.4f vs %.4f (ratio %.4f; warm start %.4f)",
        num3 / den, num4 / den, num4 / num3, WS.fac.Lr, WS.fac.Lr * num4 / num3, m3 / den, m4 / den, m3 / m4, WS.fac.Lmr))
end
# 8.3 [r4 I2, r5 I-3] warm-start fecundity, discriminating (H18 / H22): the predicted RECRUITS (the class-1 v2 microplot tally of the warm-start run minus that of the
# same run without recruitment: the model's expected count of trees entering at 1 in during the interval and still in class 1) vs the observed recruits (v2
# ingrowth trees in class 1: trees absent at v1) per key. The warm start moment-matches them on TRAIN (H22): 8.3 checks the train ratio (an independent
# recomputation) and reports the HBAD3P mapping's own ratio; 8.3b reports VAL (out of sample) with z under the NB count noise (INFO)
function recruits_v(Dn)
  Sx = Dn.S; nsx = H4.nsites(Sx); pr = zeros(nk); ob = zeros(nk); pa = zeros(nk); oa = zeros(nk); sidx = Dict(iid => s for (s, iid) in enumerate(Dn.iid))
  ps = zeros(nk, nsx)
  H4.simulate!(Sx, P, G)
  for s in 1:nsx, k in 1:nk, j in 1:2
    Dn.fm2[s] > 0 || continue; μ = Sx.PN[H4.obsix(Sx, s, k, j)] * Dn.fm2[s] * AREA_MIC; pa[k] += μ; oa[k] += Dn.n[k, j, s]; j == 1 && (pr[k] += μ; ps[k, s] += μ)
  end
  H4.simulate!(Sx, [withP(p; lnF=-Inf) for p in P], G)
  for s in 1:nsx, k in 1:nk; Dn.fm2[s] > 0 && (μ0 = Sx.PN[H4.obsix(Sx, s, k, 1)] * Dn.fm2[s] * AREA_MIC; pr[k] -= μ0; ps[k, s] -= μ0); end
  # the variance of a key's total under the observation model's own overdispersion (NB, size rM of the microplot tally): Σ_s μ_s + μ_s² / rM
  vr = [sum(max(ps[k, s], 0.0) + max(ps[k, s], 0.0)^2 / OBX.rM for s in 1:nsx) for k in 1:nk]
  kx = Dict(k => i for (i, k) in enumerate(KEYS))
  for r in eachrow(L.ig); s = get(sidx, r.iid, 0); (s > 0 && Dn.fm2[s] > 0 && H4.dclass(r.dia_cm) == 1 && ismicro(r.w_ha)) && (ob[kx[r.key]] += 1); end
  (pr=pr, ob=ob, pa=pa, oa=oa, vr=vr)
end
let rt = recruits_v(DT), rv = recruits_v(CS.sets["val"])
  okt = [0.8 <= rt.pr[k] / rt.ob[k] <= 1.25 for k in 1:nk]
  check("8.3 [r4 I2, r5 I-3, H18, H22] warm-start fecundity: predicted / observed RECRUITS (v2 class-1 ingrowth, trees absent at v1; recomputed here from the runs and the ingrowth table) on TRAIN within 0.8–1.25 for every key; the HBAD3P mapping alone (shares, light ratio) is reported — it is what H22's moment match corrects",
        all(okt), "train " * join(["$(KEYS[k]) $(round(rt.pr[k] / rt.ob[k]; digits=3)) ($(round(Int, rt.pr[k])) / $(round(Int, rt.ob[k])))$(okt[k] ? "" : " ✗")" for k in 1:nk], ", ") *
        " | the HBAD3P mapping before the match (obs / pred) " * join(["$(KEYS[k]) $(round(WS.fac.recmatch[k]; digits=3))" for k in 1:nk], ", ") *
        " | whole v2 sapling tally pred / obs (train) " * join(["$(KEYS[k]) $(round(rt.pa[k] / rt.oa[k]; digits=3))" for k in 1:nk], ", ") *
        @sprintf(" | subplot / plot recruitment light %.3f–%.3f (%s)", minimum(WS.fac.Rk), maximum(WS.fac.Rk), WS.fac.Rapplied ? "applied" : "not applied"))
  zz = [(rv.ob[k] - rv.pr[k]) / sqrt(rv.vr[k]) for k in 1:nk]
  info("8.3b [stat, H22] warm-start recruits out of sample (VAL; pred / obs, z under the observation model's NB noise Σ μ + μ²/rM)",
       join(["$(KEYS[k]) $(round(rv.pr[k] / rv.ob[k]; digits=3)) ($(round(Int, rv.pr[k])) / $(round(Int, rv.ob[k])), z $(round(zz[k]; digits=1)))" for k in 1:nk], ", ") *
       @sprintf(" — %d of %d keys beyond 2 sd: recruitment varies between folds more than the count noise (spatial clustering); calibration fits lnF", count(>(2), abs.(zz)), nk))
  H4.simulate!(ST, P, G)
end
let num = 0.0, num3 = 0.0, den = 0.0
  # HBAD3P's mortality stress gm = e^{−am O3} cw included the dropped factors (cw = e^{gR RD} … (CR/0.4)^φ) and its subplot light: gm_HBAD3P ≈ gm_HBAD4P × F × LMR (8.2 / 8.2b,
  # independent); the HBAD3P hazard with HBAD3P's own ms vs HBAD4P's with its (rescaled) ms; N-weighted one-year hazard at the v1 state, size term per atom.
  # [r4 V] everything recomputed here from the documented equations: the v1 cells' SDI from their atoms' QMD, the overtopping O(n²) with the 1 m height ramp
  for s in SUB
    Sx = subS(ST, [s]); r_ = Sx.off[1]:Sx.off[2]-1; act = [i for i in r_ if Sx.N0[i] > 1e-9]
    sdq = Dict(i => (at = atoms_of(Sx, i); nn = sum(first, at); nn * (sqrt(sum(x -> x[1] * x[2]^2, at) / nn) / 25.4)^1.605) for i in act)
    Ov = Dict(i => sum(sdq[j] * clamp(0.5 + (Sx.H0[j] - Sx.H0[i]) / 2, 0.0, 1.0) for j in act) / G.sdimax for i in act)
    for (q, i) in enumerate(r_)
      Sx.N0[i] > 1e-9 || continue; p = P[Sx.sp[i]]; O = Ov[i]
      c_ = Sx.W[1]^G.ω * exp(G.σv * lz[Sx.ln[i]]) * H4.fT_rel(p, G, Sx.T[1]); fM = (Sx.planted[1] ? exp(G.pM) : 1.0) * exp(G.mW * (Sx.Wn[1] - Sx.W[1]) / 0.1)
      h(gm, ms, D) = fM * (G.m0 / p.longevity + ms * exp(G.σST * (p.ST - 3)) * exp(-G.β * gm) * (D / 20)^G.ζ) + exp(G.mA * (Sx.A0[i] / p.longevity - 1))
      gm = exp(-G.am * O) * c_
      for (n, D) in atoms_of(Sx, i); num += n * h(gm, G.ms, D); num3 += n * h(gm * FAC3 * LMR, gl3[:ms], D); den += n; end
    end
  end
  r = num / num3
  check("8.4 [spec §8 I2, r3 I2b] warm-start mortality consistent with HBAD3P: the dropped crowding / crown factors and the subplot light also entered HBAD3P's stress gm; the N-weighted hazard at the v1 state (HBAD4P, rescaled ms) / (HBAD3P's ms, gm × F × light ratio) within 5 %",
        abs(r - 1) < 0.05, @sprintf("hazard ratio %.4f over %d sites (gm × %.4f × %.4f; ms %.5f vs HBAD3P %.5f, factor %.4f)", r, length(SUB), FAC3, LMR, G.ms, gl3[:ms], WS.fac.ms))
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
  i = findfirst(==("κ"), SP.names)
  ks = i === nothing ? Float64[] : [(z = copy(SP.z0); z[i] = v; SP.decode(z)[2].κ) for v in (-0.3, 0.0, 0.05, 2.0)]
  check("8.6 [spec §2.6, r3 I2d] κ = 0 (fully persistent lanes: the FVS triple, HBAD3P's persistent vigour nodes) is reachable — linear on [0, 1] — and is the warm start",
        G.κ == 0 && ks == [0.0, 0.0, 0.05, 1.0], "decode κ(z = −0.3, 0, 0.05, 2) = $(ks); warm start κ = $(G.κ)")
end
# 8.7 [r5 I-1, H16] encoding audit of every coordinate at the warm start: z ± 0.05 (the profile σ) and z ± 0.5 must not clamp, and ± 0.05 must change the train
# NLL (OBX) by < 1 % of |NLL|; a coordinate that starts at a bound by design (0 = "none": κ, f0) is flagged, not failed
let rows = String[], bad = String[], flagged = String[], byd = ("κ", "f0"), z0 = copy(SP.z0)
  for f in OBS4; z0[findfirst(==(String(f)), SP.names)] = log(getfield(OBX, f)); end       # (the warm start with the fitted observation parameters)
  H4.simulate!(ST, P, G); ref = nll4(DT, SP.decode(z0)[3]).total
  function val(z, nm)
    P_, G_, O_ = SP.decode(z)
    startswith(nm, "ε:") && return P_[findfirst(==(chopprefix(nm, "ε:")), KEYS)].lnε
    startswith(nm, "lnF:") && return P_[findfirst(==(chopprefix(nm, "lnF:")), KEYS)].lnF
    sy = Symbol(nm); sy in OBS4 ? getfield(O_, sy) : getfield(G_, sy)
  end
  function dnll(i, dz)
    z = copy(z0); z[i] += dz; P_, G_, O_ = SP.decode(z); H4.simulate!(ST, P_, G_); nll4(DT, O_).total - ref
  end
  for (i, nm) in enumerate(SP.names)
    cl = String[]
    for dz in (-0.5, -0.05, 0.05, 0.5)
      z = copy(z0); z[i] += dz; zz = copy(z0); zz[i] += dz * (1 + 1e-6)
      val(z, nm) == val(zz, nm) && push!(cl, @sprintf("%+.2f", dz))                  # (the decoded value does not move: a clamp)
    end
    dn = maximum(abs, [dnll(i, dz) for dz in (-0.05, 0.05)])
    tag = nm in byd ? "FLAG (at a bound by design)" : !isempty(cl) ? "BAD (clamps)" : dn >= 0.01 * abs(ref) ? "FLAG (sensitive: ±0.05 moves the NLL ≥ 1 %)" : "ok"
    startswith(tag, "BAD") && push!(bad, nm); startswith(tag, "FLAG") && push!(flagged, nm)
    push!(rows, @sprintf("%s: start %.4g, clamps %s, |ΔNLL| at ±0.05 %.1f (%.3f %%) %s", nm, val(z0, nm), isempty(cl) ? "none" : join(cl, " "), dn, 100dn / abs(ref), tag))
  end
  H4.simulate!(ST, P, G)
  println("      8.7 encoding table:"); for r in rows; println("        ", r); end
  check("8.7 [r5 I-1, H16] every coordinate's encoding: no clamp at z ± 0.05 / ± 0.5 from the warm start (ρ log on [1e-7, 0.05]); |ΔNLL(±0.05)| ≥ 1 % of |NLL| and a start at a bound by design are FLAGGED",
        isempty(bad), "$(length(SP.names)) coordinates; clamping: $(isempty(bad) ? "none" : join(bad, ", ")); flagged: $(join(flagged, ", ")) (table above)")
end
# 8.8 [r5 I-4, H19] load_z4 refuses a parameter file of another space (here: written with HBAD4P_FREE=gR) and loads it only with allow_mismatch; a file of the
# same space round-trips exactly
let tmp = mktempdir(), SPf = withenv("HBAD4P_FREE" => "gR") do; space4(KEYS, TR, CS.sD, CS.sH; starts4(P, G, OBX)..., ); end
  f1 = save_z4(joinpath(tmp, "same.csv"), SP, SP.z0); okrt = load_z4(f1, SP) == SP.z0
  f2 = save_z4(joinpath(tmp, "other.csv"), SPf, SPf.z0)
  refused = try load_z4(f2, SP); false catch e; occursin("differs", sprint(showerror, e)) end
  forced = try z = load_z4(f2, SP; allow_mismatch=true); length(z) == length(SP.z0) catch; false end
  rm(f1 * ".sig"); nosig = try load_z4(f1, SP); false catch; true end
  check("8.8 [r5 I-4, H19] save_z4 records the space signature (names, encodings, fixed globals, HBAD4P_FIX / FREE); load_z4 refuses a mismatch (or a missing signature) unless allow_mismatch",
        okrt && refused && forced && nosig, "round trip $(okrt); other space refused $(refused); loaded with allow_mismatch $(forced); missing .sig refused $(nosig)")
end
end # section 8

# ================================================================ 9. stand-level quantities and baselines
if want(9)
section("9. stand-level quantities and baselines (train in-sample; val out-of-sample)")
function standv(S, D, which)
  ns_ = H4.nsites(S); N = zeros(ns_); N12 = zeros(ns_); BA = zeros(ns_); LH = zeros(ns_); BA12 = zeros(ns_)
  for s in 1:ns_, k in 1:S.nk, j in 1:NCLS
    r = cellix(S, s, k, j, 1):cellix(S, s, k, j, 3)
    if which == :obs
      a = (j >= 3 ? D.fs2[s] : D.fm2[s]) * area_class(j); a > 0 || continue          # ([r4 M1] an unobserved tally)
      n = D.n[k, j, s] / a; b = D.D2[k, j, s] * π / 4e4 / a; h = D.nH[k, j, s] > 0 ? D.Hs[k, j, s] / D.nH[k, j, s] : 0.0
    else
      Nv = which == :sim ? S.N[r] : S.N0[r]; Bv = which == :sim ? S.B[r] : S.B0[r]; Hv = which == :sim ? S.H[r] : S.H0[r]; n = sum(Nv); b = sum(Bv); h = n > 0 ? sum(Nv .* Hv) / n : 0.0
    end
    N[s] += n; BA[s] += b; j >= 3 && (N12[s] += n; BA12[s] += b; LH[s] += b * h)
  end
  (N=mean(N), N12=mean(N12), BA=mean(BA), QMD12=sqrt(sum(BA12) / sum(N12) * 4e4 / π), LH=sum(LH) / sum(BA12))
end
H4.simulate!(ST, P, G)
let o = standv(ST, DT, :obs), m = standv(ST, DT, :sim), z = standv(ST, DT, :v1)
  info("9.1 plot means at v2 (train, warm start)", @sprintf("N obs %.0f sim %.0f (%+.1f %%) [v1 %.0f] | N≥12.7 obs %.1f sim %.1f (%+.1f %%) | BA obs %.2f sim %.2f (%+.1f %%) [v1 %.2f] | QMD≥12.7 obs %.2f sim %.2f | Lorey H≥12.7 obs %.2f sim %.2f",
       o.N, m.N, 100 * (m.N / o.N - 1), z.N, o.N12, m.N12, 100 * (m.N12 / o.N12 - 1), o.BA, m.BA, 100 * (m.BA / o.BA - 1), z.BA, o.QMD12, m.QMD12, o.LH, m.LH))
  sidx = Dict(iid => s for (s, iid) in enumerate(DT.iid)); obs = 0.0
  for r in eachrow(L.tr[in.(L.tr.iid, Ref(keys(sidx))) .& (L.tr.out .== "alive") .& (L.tr.dia_cm .< 12.7) .& (L.tr.dia2_cm .>= 12.7), :]); obs += r.w2_ha / DT.fs2[sidx[r.iid]]; end
  for r in eachrow(L.ig[in.(L.ig.iid, Ref(keys(sidx))) .& (L.ig.dia_cm .>= 12.7), :]); obs += r.w_ha / DT.fs2[sidx[r.iid]]; end
  Sb = subS(ST, 1:ns; drop=i -> ST.cl[i] <= 2)
  H4.simulate!(Sb, [withP(p; lnF=-Inf) for p in P], G); mb = standv(Sb, DT, :sim)
  info("9.2 ingrowth into 12.7 cm over the interval (train, plot mean)", @sprintf("observed %.1f /ha | warm start %.1f /ha (N≥12.7 full run %.1f − run without classes 1–2 and recruitment %.1f)", obs / ns, m.N12 - mb.N12, m.N12, mb.N12))
  Sb = nothing; GC.gc()
end
# 9.2b [r3 I6] intervals whose v2 visit has NO forested subplot: FIA tallies no tree there; they must be excluded (plotf would return 1.0: scored against an empty tally)
let nzs = [s for (s, r) in enumerate(eachrow(DT.iv)) if all(sb -> get(FQ, (r.st, r.u, r.c, r.p, Date(r.d2), sb), (NaN, NaN))[1] == 0, 1:4)],
    nz1 = [s for (s, r) in enumerate(eachrow(DT.iv)) if all(sb -> get(FQ, (r.st, r.u, r.c, r.p, Date(r.d1), sb), (NaN, NaN))[1] == 0, 1:4)],
    mz2 = [s for (s, r) in enumerate(eachrow(DT.iv)) if all(sb -> get(FQ, (r.st, r.u, r.c, r.p, Date(r.d2), sb), (NaN, NaN))[2] == 0, 1:4)],
    mz1 = [s for (s, r) in enumerate(eachrow(DT.iv)) if all(sb -> get(FQ, (r.st, r.u, r.c, r.p, Date(r.d1), sb), (NaN, NaN))[2] == 0, 1:4)]
  check("9.2b [data, r3 I6, r4 M1] no train interval whose v2 or v1 visit has no forested subplot is scored; a tally with no forested area (all four microplots non-forest) has f = 0 (no exposure, no count term)",
        isempty(nzs) && isempty(nz1) && all(DT.fm2[s] == 0 for s in mz2) && all(DT.fm1[s] == 0 for s in mz1) && sum(DT.n[:, 1:2, mz2]; init=0) == 0,
        "$(length(nzs)) / $(length(nz1)) such train sites left (the builder dropped $(CS.ndrop) / $(CS.ndrop1) over all sets) | all microplots non-forest: v2 $(length(mz2)), v1 $(length(mz1)) train intervals (fm = $(unique(DT.fm2[mz2])) / $(unique(DT.fm1[mz1]))); their v2 sapling tally $(sum(DT.n[:, 1:2, mz2]; init=0))")
end
# [r4 V] the transition-matrix null re-implemented from its definition (docs §3) with table aggregations (not a copy of data4.jl tm_null): train v1 trees
# (density w / f of their tally) by key × v1 class × dt tercile → fraction alive at v2 per v2 class (and its basal area); train ingrowth density per key × class
# × v1 seed presence, per year of the summed interval lengths of the train sites in that key × seed group; heights = the mean v2 height per key × class.
TMV = let sidx = Dict(iid => s for (s, iid) in enumerate(DT.iid)), dtv = DT.iv.dt, qs = quantile(dtv, [1 / 3, 2 / 3])
  terc(dt) = dt <= qs[1] ? 1 : dt <= qs[2] ? 2 : 3
  t = v1trees(L, Set(DT.iid)); t = DataFrame(key=t.key, s=[sidx[i] for i in t.iid], j1=H4.dclass.(t.dia_cm), alive=t.out .== "alive",
                                             j2=[o == "alive" && d > 0 ? H4.dclass(d) : 0 for (o, d) in zip(t.out, t.dia2_cm)],
                                             d2=t.dia2_cm, h2=t.ht2_m, mic=ismicro.(t.w_ha), w0=t.w_ha)
  t.f = [r.mic ? DT.fm1[r.s] : DT.f1[r.s] for r in eachrow(t)]; t = t[(t.j1 .> 0) .& (t.f .> 0), :]; t.w = t.w0 ./ t.f; t.b = terc.(dtv[t.s])
  W0 = Dict((r.key, r.j1, r.b) => r.W for r in eachrow(combine(groupby(t, [:key, :j1, :b]), :w => sum => :W)))
  ta = t[t.alive .& (t.j2 .> 0), :]
  TR_ = Dict((r.key, r.j1, r.b, r.j2) => (r.n, r.ba) for r in eachrow(combine(groupby(ta, [:key, :j1, :b, :j2]), :w => sum => :n, [:w, :d2] => ((w, d) -> sum(w .* π ./ 4e4 .* d .^ 2)) => :ba)))
  g = L.ig[in.(L.ig.iid, Ref(keys(sidx))), :]; g = DataFrame(key=g.key, s=[sidx[i] for i in g.iid], j=H4.dclass.(g.dia_cm), d=g.dia_cm, h=g.ht_m, mic=ismicro.(g.w_ha), w0=g.w_ha)
  g.f = [r.mic ? DT.fm2[r.s] : DT.fs2[r.s] for r in eachrow(g)]; g = g[(g.j .> 0) .& (g.f .> 0), :]; g.w = g.w0 ./ g.f
  kx = Dict(k => i for (i, k) in enumerate(KEYS)); g.seed = [ST.seed0[kx[r.key], r.s] for r in eachrow(g)]
  IG = Dict((r.key, r.j, r.seed) => (r.n, r.ba) for r in eachrow(combine(groupby(g, [:key, :j, :seed]), :w => sum => :n, [:w, :d] => ((w, d) -> sum(w .* π ./ 4e4 .* d .^ 2)) => :ba)))
  DS = Dict((k, sd) => sum(dtv[s] for s in 1:ns if ST.seed0[kx[k], s] == sd; init=0.0) for k in KEYS, sd in (false, true))
  hh = vcat(DataFrame(key=ta.key, j=ta.j2, h=ta.h2), DataFrame(key=g.key, j=g.j, h=g.h)); hh = hh[hh.h .> 0, :]
  HM = Dict((r.key, r.j) => r.h for r in eachrow(combine(groupby(hh, [:key, :j]), :h => mean => :h)))
  (W0=W0, TR=TR_, IG=IG, DS=DS, HM=HM, terc=terc)
end
"The verifier's transition-matrix null of set D into D.S (lane 1) and its observation-space table."
function tm_state_v!(D)
  S = D.S; fill!(S.N, 0.0); fill!(S.B, 0.0); fill!(S.H, 0.0)
  for s in 1:H4.nsites(S), (k, key) in enumerate(KEYS)
    b = TMV.terc(D.iv.dt[s]); n2 = zeros(NCLS); b2 = zeros(NCLS)
    for j1 in 1:NCLS
      i = cellix(S, s, k, j1, 1); N0 = S.N0[i] + S.N0[i+1] + S.N0[i+2]; W = get(TMV.W0, (key, j1, b), 0.0); (N0 > 0 && W > 0) || continue
      for j2 in 1:NCLS; x = get(TMV.TR, (key, j1, b, j2), (0.0, 0.0)); n2[j2] += N0 * x[1] / W; b2[j2] += N0 * x[2] / W; end
    end
    sd = S.seed0[k, s]; ds = TMV.DS[(key, sd)]; rate = ds > 0 ? D.iv.dt[s] / ds : 0.0
    for j in 1:NCLS; x = get(TMV.IG, (key, j, sd), (0.0, 0.0)); n2[j] += x[1] * rate; b2[j] += x[2] * rate; end
    for j in 1:NCLS
      i = cellix(S, s, k, j, 1); S.N[i] = n2[j]; S.B[i] = b2[j]; S.H[i] = get(TMV.HM, (key, j), 1.37 + 0.5 * H4.CLS_LO[j])
      x = H4.obsix(S, s, k, j); S.PN[x] = S.N[i]; S.PB[x] = S.B[i]; S.PH[x] = S.N[i] * S.H[i]
    end
  end
  D
end
let res = Dict{String,Any}(), Dv = CS.sets["val"], iob = [findfirst(==(String(nm)), SP.names) for nm in OBS4]
  # [r3 I3] the tools' transition-matrix null (data4.jl tm_null / tm_state!, built into the cache) = this independent one
  tm_state_v!(DT); a = (copy(DT.S.N), copy(DT.S.B), copy(DT.S.PN), copy(DT.S.PB), copy(DT.S.PH)); tm_state!(DT, CS.TM)
  dtm = maximum(maximum(abs.(x .- y)) / max(maximum(abs.(y)), 1e-12) for (x, y) in zip(a, (DT.S.N, DT.S.B, DT.S.PN, DT.S.PB, DT.S.PH)))
  check("9.3 [r3 I3, r4 V] the evaluation tools' transition-matrix null (data4.jl tm_null / tm_state!, reported by smoke4 as the bar) = the verifier's re-implementation from the definition (table aggregations)", dtm < 1e-9, "max rel diff $(fmt(dtm))")
  fitob(D) = (z = copy(SP.z0); f(x) = (zz = copy(z); zz[iob] = x; nll4(D, SP.decode(zz)[3]).total); r = cmaes(f, z[iob], 0.3; λ=12, maxevals=600, logf=s -> nothing); zz = copy(z); zz[iob] = r.x; SP.decode(zz)[3])
  for (lab, prep!) in (("persistence", D -> null_state!(D)), ("transition matrix", D -> tm_state_v!(D)), ("HBAD4P warm start", D -> (H4.simulate!(D.S, P, G); D)))
    prep!(DT); ob = fitob(DT); a = nll4(DT, ob); prep!(Dv); b = nll4(Dv, ob); res[lab] = (a, b)
  end
  for (k, v) in res; println("      ", @sprintf("%-18s train %10.1f (count %9.1f size %9.1f height %9.1f) | val %10.1f (count %9.1f size %9.1f height %9.1f)", k, v[1].total, v[1].count, v[1].size, v[1].height, v[2].total, v[2].count, v[2].size, v[2].height)); end
  tm = res["transition matrix"][2]; pe = res["persistence"][2]; ws = res["HBAD4P warm start"][2]
  info("9.3b [stat, r3 I3] baselines on val (the transition-matrix null is the bar HBAD4P must beat on val counts; persistence is reported, not the bar)",
       @sprintf("val NLL: persistence %.0f, transition matrix %.0f (%+.0f), warm start %.0f (%+.0f vs persistence, %+.0f vs TM) | val COUNT NLL: persistence %.0f, transition matrix %.0f, warm start %.0f (%+.0f vs TM) — the warm start %s the TM null on val counts",
                pe.total, tm.total, tm.total - pe.total, ws.total, ws.total - pe.total, ws.total - tm.total, pe.count, tm.count, ws.count, ws.count - tm.count, ws.count < tm.count ? "BEATS" : "does NOT yet beat"))
  H4.simulate!(DT.S, P, G)
end
end # section 9

# ================================================================ P. performance
if want("P")
section("P. performance (one simulate! of the $(ns) train sites)")
let Sx = ST, msg = String[]
  for GG in (G, withG(G; κ=0.1))
    H4.simulate!(Sx, P, GG); t2 = Inf; for _ in 1:2; t = time(); H4.simulate!(Sx, P, GG); t2 = min(t2, time() - t); end
    serial_run!(subS(ST, 1:50), P, GG); Sq = subS(ST, 1:ns); t = time(); serial_run!(Sq, P, GG); t1 = time() - t; Sq = nothing
    push!(msg, @sprintf("κ = %.2f: simulate! %.2f s with %d threads, serial %.2f s (scaling %.2f×)", GG.κ, t2, Threads.nthreads(), t1, t1 / t2))
  end
  info("P.1 timing (NATOM $(NA4), compact atom store, occupied cells only)", join(msg, " | ") * " — round-2 kernel (NATOM 4, dense, κ = 0.1): 11.7 s with 2 threads, serial 21.2–23.9 s")
  H4.simulate!(ST, P, G)
end
end

# ================================================================ summary
@isdefined(con) && DBInterface.close!(con)
nf = count(r -> r[1] == "FAIL", RES); np = count(r -> r[1] == "PASS", RES)
println("\n==== SUMMARY: $(np) PASS, $(nf) FAIL, $(count(r -> r[1] == "INFO", RES)) INFO | $(round(time() - t00; digits=0)) s")
for r in RES; r[1] == "FAIL" && println("FAIL  ", r[2]); end
exit(nf > 0 ? 1 : 0)
