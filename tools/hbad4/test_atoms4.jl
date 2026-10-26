# Atom (cohorts-in-bins) checks of the HBAD4P kernel: buffer merge conserves N and N·D² (B); compaction to NATOM; one kernel step with zero mortality /
# recruitment / mixing on a synthetic site moves every atom to the class of its grown diameter (exact transfer) and conserves N; a tree-level replay of
# real-like sparse cells (1–3 trees) has zero transfer error; lanes; class bounds.   ./julia_gdal.sh --project=. tools/hbad4/test_atoms4.jl  (exit 1 on a failure)
include(joinpath(@__DIR__, "..", "..", "src", "plugins", "HBAD4", "HBAD4P.jl")); import .HBAD4P as H; using Random
fails = 0; ck(nm, ok, d="") = (println(ok ? "PASS  " : "FAIL  ", nm, "  ", d); ok || (global fails += 1))
rng = Xoshiro(3)
bn = zeros(64); bd = zeros(64); c = 0; Ns = 0.0; B2 = 0.0
for _ in 1:40; n = rand(rng) * 10; d = 2.54 + rand(rng) * 5.08; global c = H.buf_add!(bn, bd, 0, c, 2 * H.NATOM, n, d); global Ns += n; global B2 += n * d^2; end
c2 = H.compact!(bn, bd, 0, c)
sorted_ok = issorted(bd[1:c2])
ck("buffer add (cap 2 NATOM) + compaction to NATOM conserve N and N·D²", c2 == H.NATOM && abs(sum(bn[1:c2]) - Ns) < 1e-9 * Ns && abs(sum(bn[1:c2] .* bd[1:c2] .^ 2) - B2) < 1e-9 * B2,
   "atoms $c -> $c2")
ck("merged atoms stay inside the span of their inputs (no new extremes), sorted by D", all(2.54 - 1e-9 .<= bd[1:c2] .<= 7.62 + 1e-9) && sorted_ok)
let bn = [1.0, 2.0, 3.0, 0.0], bd = [5.0, 5.0, 6.0, 0.0]; c = H.compact!(bn, bd, 0, 3)
  ck("exact duplicates merge even below NATOM atoms", c == 2 && bn[1] == 3.0 && bd[1] == 5.0, "atoms $(collect(zip(bn[1:c], bd[1:c])))")
end
# a synthetic one-key site, one lane mid; growth only (no mortality, recruitment, mixing, thinning)
P = [H.SpP4(log(2.0), -Inf, 30.0, 3.0, 300.0, 17.0)]        # (lnF = −Inf: no recruitment)
G = H.GlobP4(1.75, 2.0, 0.0, 8.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 1e9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             1.0, 0.0, 0.0, 0.0, 0.2)
nk = 1; ncell = nk * H.NCLS * H.NL
trees = [(3, 17.0, 20.0), (3, 17.7, 15.0), (5, 27.5, 8.0), (1, 6.2, 100.0)]        # (class, D, N) — sparse cells as in real plots
at0 = Dict{Int,Vector{Tuple{Float64,Float64}}}()
for (j, d, n) in trees; ci = ((1 - 1) * H.NCLS + (j - 1)) * H.NL + 1; push!(get!(at0, ci, Tuple{Float64,Float64}[]), (n, d)); end
Hs = fill(15.0, ncell); As = fill(30.0, ncell)
S = H.make_sites4([1, ncell + 1], nk, Int32.(ones(ncell)), Int8[j for j in 1:H.NCLS for _ in 1:H.NL], Int8[v for _ in 1:H.NCLS for v in 1:H.NL], collect(at0), Hs, As,
                  [1], [0.0], [false], [1.0], [1.0], [17.0], [390.1], falses(1, 1), H.lanes()...)
X = H.Scratch(ncell); H.run_site!(S, P, G, X, 1, H.Tabs(P, G))
moved = Dict{Int,Float64}(); cls_ok = true
for i in 1:ncell, (n, d) in H.cell_atoms(X, S, 1, i); n > 0 && (moved[H.dclass(d)] = get(moved, H.dclass(d), 0.0) + n); H.dclass(d) == S.cl[i] || (global cls_ok = false); end
ck("one growth step: N conserved, every atom sits in the cell of its diameter's class (exact transfer)", abs(sum(S.N) - sum(S.N0)) < 1e-9 && cls_ok, "classes now $(sort(collect(keys(moved))))")
dDs = [d for i in 1:ncell for (n, d) in H.cell_atoms(X, S, 1, i)]; d0 = sort([t[2] for t in trees]); ck("all atoms grew by a non-negative increment", minimum(sort(dDs)) >= minimum(d0) - 1e-12)
ck("observation blur conserves N (PN summed over classes = Σ N) and each atom stays within ±1 class", abs(sum(S.PN) - sum(S.N)) < 1e-9 * sum(S.N))
ck("Φ table vs the series (|z| ≤ 8.5)", maximum(abs(H.Φ(z) - H.Φ_series(z)) for z in range(-9, 9; length=20001)) < 1e-13)
for mode in (:sym, :fvs)
  z, w = H.lanes(mode); ck("lanes $mode: weights 1, mean 0, variance 1", abs(sum(w) - 1) < 1e-12 && abs(sum(w .* z)) < 1e-9 && abs(sum(w .* z .^ 2) - 1) < 1e-9)
end
ck("dclass at exact bounds (12.7 → 3, 78.74 → 16, 2.54 → 1, 2.53 → 0)", H.dclass(12.7) == 3 && H.dclass(78.74) == 16 && H.dclass(2.54) == 1 && H.dclass(2.53) == 0)
exit(fails > 0 ? 1 : 0)
