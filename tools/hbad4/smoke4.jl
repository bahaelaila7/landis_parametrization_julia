# HBAD4P smoke test: build one set, simulate from an HBAD3P warm start, score, time; print stand-level sanity numbers.
#   HBAD2_DATA_DIR=<product> ./julia_gdal.sh --project=. --threads=4 tools/hbad4/smoke4.jl <hbad3p fit dir> [set=train]
include(joinpath(@__DIR__, "data4.jl")); include(joinpath(@__DIR__, "params4.jl"))
t0 = time()
L = load4(); CL = climate4(); FT = forest4(); TR = traits4(L, CL.clim)
println("keys ", join(L.keys, ","), " | intervals ", nrow(L.iv), " (", join(["$s $(count(==(s), L.iv.set))" for s in ("train", "val", "test")], ", "), ") | load ", round(time() - t0; digits=1), " s")
show(TR; allrows=true); println()
set = length(ARGS) > 1 ? ARGS[2] : "train"
t1 = time(); D = build4(L, set, L.keys, CL.W, CL.Wn, CL.T, nothing, FT)
println("built $(set): sites $(HBAD4P.nsites(D.S)) cells $(length(D.S.N0)) v2 tally trees $(D.ntrees) cells without a height $(D.nohgt) | ", round(time() - t1; digits=1), " s")
e3, f3, g3 = hbad3p_start(ARGS[1], L.keys)
SP = space4(L.keys, TR; start_eps=e3, start_lnF=f3, start_glob=g3); P, G, Ob = SP.decode(SP.z0)
println("coordinates $(length(SP.z0)) (2 × $(SP.nk) per key + $(length(SP.gfree)) global); fixed ", SP.fix)
null_state!(D); pn = nll4(D, Ob)
HBAD4P.simulate!(D.S, P, G); t2 = time(); HBAD4P.simulate!(D.S, P, G); ts = time() - t2
t3 = time(); p = nll4(D, Ob); tl = time() - t3
println(@sprintf("NLL start %.1f (count %.1f size %.1f height %.1f) | persistence null %.1f | simulate %.2f s, nll %.2f s", p.total, p.count, p.size, p.height, pn.total, ts, tl))
# stand-level sanity: plot totals at v2 (sim vs obs): N, N ≥ 12.7, BA
S = D.S; ns = HBAD4P.nsites(S); nk = S.nk
Ns = zeros(ns); N12s = zeros(ns); Bs = zeros(ns); No = zeros(ns); N12o = zeros(ns); Bo = zeros(ns); N1 = zeros(ns); B1 = zeros(ns)
for s in 1:ns, k in 1:nk, j in 1:NCLS
  i0 = S.off[s] + ((k - 1) * NCLS + (j - 1)) * NL; n = sum(S.N[i0:i0+2]); b = sum(S.B[i0:i0+2])
  Ns[s] += n; Bs[s] += b; j >= 3 && (N12s[s] += n); N1[s] += sum(S.N0[i0:i0+2]); B1[s] += sum(S.B0[i0:i0+2])
  f = j >= 3 ? D.fs2[s] : D.fm2[s]; a = area_class(j); no = D.n[k, j, s] / (f * a); No[s] += no; j >= 3 && (N12o[s] += no)
  Bo[s] += D.D2[k, j, s] * π / 4 / 1e4 / (f * a)
end
println(@sprintf("plot means: N v1 %.0f | v2 obs %.0f sim %.0f (%.1f %%) | N≥12.7 obs %.0f sim %.0f (%.1f %%) | BA v1 %.2f | v2 obs %.2f sim %.2f (%.1f %%)",
                 mean(N1), mean(No), mean(Ns), 100 * (mean(Ns) / mean(No) - 1), mean(N12o), mean(N12s), 100 * (mean(N12s) / mean(N12o) - 1), mean(B1), mean(Bo), mean(Bs), 100 * (mean(Bs) / mean(Bo) - 1)))
println("all finite: ", all(isfinite, S.N) && all(isfinite, S.B) && all(isfinite, S.H), " | min N ", minimum(S.N), " | QMD outside its class: ",
        count(i -> S.N[i] > 1e-6 && !(HBAD4P.CLS_LO[S.cl[i]] - 1e-6 <= HBAD4P.qmd(S.N[i], S.B[i]) <= HBAD4P.CLS_HI[S.cl[i]] + 1e-6), eachindex(S.N)), " of ", count(>(1e-6), S.N))
