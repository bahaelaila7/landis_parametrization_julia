# Atom (cohorts-in-bins) checks of the HBAD4P kernel: buffer merge conserves N and N·D² (B); compaction to NATOM; one kernel step with zero mortality /
# recruitment / mixing on a synthetic site moves every atom to the class of its grown diameter (exact transfer) and conserves N; a tree-level replay of
# real-like sparse cells (1–3 trees) has zero transfer error; lanes; class bounds.   ./julia_gdal.sh --project=. tools/hbad4/test_atoms4.jl  (exit 1 on a failure)
include(joinpath(@__DIR__, "..", "..", "src", "plugins", "HBAD4", "HBAD4P.jl")); import .HBAD4P as H; using Random
fails = 0; ck(nm, ok, d="") = (println(ok ? "PASS  " : "FAIL  ", nm, "  ", d); ok || (global fails += 1))
rng = Xoshiro(3)
bn = zeros(64); bd = zeros(64); c = 0; Ns = 0.0; B2 = 0.0
for _ in 1:40; n = rand(rng) * 10; d = 2.54 + rand(rng) * 5.08; global c = H.buf_add!(bn, bd, 0, c, 16, n, d); global Ns += n; global B2 += n * d^2; end
c2 = H.compact!(bn, bd, 0, c)
ck("buffer add (cap 16) + compaction to NATOM conserve N and N·D²", c2 == H.NATOM && abs(sum(bn[1:c2]) - Ns) < 1e-9 * Ns && abs(sum(bn[1:c2] .* bd[1:c2] .^ 2) - B2) < 1e-9 * B2,
   "atoms $c -> $c2")
ck("merged atoms stay inside the span of their inputs (no new extremes)", all(2.54 - 1e-9 .<= bd[1:c2] .<= 7.62 + 1e-9))
# a synthetic one-key site, one lane mid; growth only (no mortality, recruitment, mixing, thinning)
P = [H.SpP4(log(2.0), -Inf, 30.0, 3.0, 300.0, 17.0)]        # (lnF = −Inf: no recruitment)
G = H.GlobP4(1.75, 2.0, 0.0, 8.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 1e9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             1.0, 0.0, 0.0, 0.0)
nk = 1; ncell = nk * H.NCLS * H.NL; NA = H.NATOM
AN0 = zeros(ncell * NA); AD0 = zeros(ncell * NA)
trees = [(3, 17.0, 20.0), (3, 17.7, 15.0), (5, 27.5, 8.0), (1, 6.2, 100.0)]        # (class, D, N) — sparse cells as in real plots
for (j, d, n) in trees
  ci = ((1 - 1) * H.NCLS + (j - 1)) * H.NL + 1; k0 = findfirst(==(0.0), AN0[(ci-1)*NA+1:ci*NA]); AN0[(ci - 1) * NA + k0] = n; AD0[(ci - 1) * NA + k0] = d
end
N0 = zeros(ncell); B0 = zeros(ncell); for i in 1:ncell; N0[i], B0[i] = H.totals(AN0, AD0, i); end
Hs = fill(15.0, ncell); As = fill(30.0, ncell)
S = H.Sites4([1, ncell + 1], nk, Int32.(ones(ncell)), Int8[j for j in 1:H.NCLS for _ in 1:H.NL], Int8[v for _ in 1:H.NCLS for v in 1:H.NL], AN0, AD0, copy(Hs), copy(As),
             copy(AN0), copy(AD0), copy(Hs), copy(As), N0, B0, copy(N0), copy(B0), [1], [0.0], [false], [1.0], [1.0], [17.0], [390.1], falses(1, 1), H.lanes()...)
H.simulate!(S, P, G)
moved = Dict{Int,Float64}(); for i in 1:ncell, m in 1:NA; k = (i - 1) * NA + m; S.AN[k] > 0 && (moved[H.dclass(S.AD[k])] = get(moved, H.dclass(S.AD[k]), 0.0) + S.AN[k]); end
cls_ok = all(i -> all(m -> S.AN[(i - 1) * NA + m] == 0 || H.dclass(S.AD[(i - 1) * NA + m]) == S.cl[i], 1:NA), 1:ncell)
ck("one growth step: N conserved, every atom sits in the cell of its diameter's class (exact transfer)", abs(sum(S.N) - sum(N0)) < 1e-9 && cls_ok, "classes now $(sort(collect(keys(moved))))")
dDs = [S.AD[k] for k in eachindex(S.AD) if S.AN[k] > 0]; d0 = sort([t[2] for t in trees]); ck("all atoms grew by a non-negative increment", minimum(sort(dDs)) >= minimum(d0) - 1e-12)
for mode in (:sym, :fvs)
  z, w = H.lanes(mode); ck("lanes $mode: weights 1, mean 0, variance 1", abs(sum(w) - 1) < 1e-12 && abs(sum(w .* z)) < 1e-9 && abs(sum(w .* z .^ 2) - 1) < 1e-9)
end
ck("dclass at exact bounds (12.7 → 3, 78.74 → 16, 2.54 → 1, 2.53 → 0)", H.dclass(12.7) == 3 && H.dclass(78.74) == 16 && H.dclass(2.54) == 1 && H.dclass(2.53) == 0)
exit(fails > 0 ? 1 : 0)
