# HBAD4P smoke test: build (cached) the sets of a fold product, simulate from an HBAD3P warm start, score against the persistence null (both with their own fitted
# observation parameters), print stand-level sanity numbers.
#   HBAD2_DATA_DIR=<product> ./julia_gdal.sh --project=. --threads=2 tools/hbad4/smoke4.jl <hbad3p fit dir> [set=train]
include(joinpath(@__DIR__, "data4.jl")); include(joinpath(@__DIR__, "params4.jl")); include(joinpath(@__DIR__, "..", "hbad2", "cmaes.jl"))
t0 = time(); CS = cached_sets(["train", "val", "test"]); set = length(ARGS) > 1 ? ARGS[2] : "train"; D = CS.sets[set]
println("keys ", join(CS.keys, ","), " | sets ", join(["$s $(HBAD4P.nsites(CS.sets[s].S))" for s in keys(CS.sets)], ", "), " | ", round(time() - t0; digits=1), " s")
show(CS.TR; allrows=true); println()
println("sapling layer (classes 1–2, trees / forested ha, plot mean): observed $(round(D.sap.obs; digits=1)) -> empirical Bayes $(round(D.sap.eb; digits=1)); ",
        "within-cell sd log D by class ", round.(CS.sD; digits=3), " | log H ", round.(CS.sH; digits=3))
println("missing climate ", D.nmiss, " | v2 tally trees $(D.ntrees) | cells without a height $(D.nohgt)")
# warm start: HBAD3P's dropped crowding / crown factors, at the train v1 state
S = D.S; ns = HBAD4P.nsites(S)
gl3 = Dict(Symbol(r.name) => Float64(r.value) for r in eachrow(CSV.read(joinpath(ARGS[1], "globals.csv"), DataFrame)))
RD = [sum(HBAD4P.sdi(S.N0[i], HBAD4P.qmd(S.N0[i], S.B0[i])) for i in S.off[s]:S.off[s+1]-1 if S.N0[i] > 0; init=0.0) / 1500 for s in 1:ns]
fac = mean(exp.(get(gl3, :gR, 0.0) .* RD)) * mean((CS.cr_train ./ 0.4) .^ get(gl3, :φ, 0.0))
e3, f3, g3 = hbad3p_start(ARGS[1], CS.keys; eps_factor=fac)
SP = space4(CS.keys, CS.TR, CS.sD, CS.sH; start_eps=e3, start_lnF=f3, start_glob=g3); P, G, Ob = SP.decode(SP.z0)
println("coordinates $(length(SP.z0)) (2 × $(SP.nk) per key + $(length(SP.gfree)) global); fixed ", SP.fix, " | ε rescaled × $(round(fac; digits=3)) for HBAD3P's gR / φ")
# observation parameters of a given state (CMA-ES on the 6 observation globals only)
iob = [findfirst(==(String(nm)), SP.names) for nm in OBS4]
function fit_obs(z)
  f(x) = (zz = copy(z); zz[iob] = x; nll4(D, SP.decode(zz)[3]).total)
  r = cmaes(f, z[iob], 0.3; λ=12, maxevals=600, logf=s -> nothing); zz = copy(z); zz[iob] = r.x; zz
end
null_state!(D); zn = fit_obs(SP.z0); pn = nll4(D, SP.decode(zn)[3])
HBAD4P.simulate!(S, P, G); t2 = time(); HBAD4P.simulate!(S, P, G); ts = time() - t2
zs = fit_obs(SP.z0); p = nll4(D, SP.decode(zs)[3])
println(@sprintf("NLL warm start %.1f (count %.1f size %.1f height %.1f) | persistence null %.1f (count %.1f) | simulate %.2f s", p.total, p.count, p.size, p.height, pn.total, pn.count, ts))
println("observation parameters (warm start): ", Dict(OBS4[i] => round(SP.decode(zs)[3] |> o -> getfield(o, OBS4[i]); sigdigits=3) for i in 1:length(OBS4)))
Ns = zeros(ns); N12s = zeros(ns); Bs = zeros(ns); No = zeros(ns); N12o = zeros(ns); Bo = zeros(ns); N1 = zeros(ns); B1 = zeros(ns)
for s in 1:ns, k in 1:S.nk, j in 1:NCLS
  i0 = S.off[s] + ((k - 1) * NCLS + (j - 1)) * NL; n = sum(S.N[i0:i0+2]); b = sum(S.B[i0:i0+2])
  Ns[s] += n; Bs[s] += b; j >= 3 && (N12s[s] += n); N1[s] += sum(S.N0[i0:i0+2]); B1[s] += sum(S.B0[i0:i0+2])
  f = j >= 3 ? D.fs2[s] : D.fm2[s]; a = area_class(j); no = D.n[k, j, s] / (f * a); No[s] += no; j >= 3 && (N12o[s] += no); Bo[s] += D.D2[k, j, s] * π / 4 / 1e4 / (f * a)
end
println(@sprintf("plot means: N v1 %.0f | v2 obs %.0f sim %.0f (%.1f %%) | N≥12.7 obs %.0f sim %.0f (%.1f %%) | BA v1 %.2f | v2 obs %.2f sim %.2f (%.1f %%)",
                 mean(N1), mean(No), mean(Ns), 100 * (mean(Ns) / mean(No) - 1), mean(N12o), mean(N12s), 100 * (mean(N12s) / mean(N12o) - 1), mean(B1), mean(Bo), mean(Bs), 100 * (mean(Bs) / mean(Bo) - 1)))
out = count(i -> S.N[i] > 1e-6 && !(HBAD4P.CLS_LO[S.cl[i]] - 1e-6 <= HBAD4P.qmd(S.N[i], S.B[i]) <= HBAD4P.CLS_HI[S.cl[i]] + 1e-6), eachindex(S.N))
println("all finite: ", all(isfinite, S.N) && all(isfinite, S.B) && all(isfinite, S.H), " | cells with a QMD outside their class: $out of ", count(>(1e-6), S.N),
        " | seed sources < 1 tree/ha in classes ≥ 3: ", count(s_ -> 0 < s_ < 1, [sum(S.N[HBAD4P.cell(S, s, k, j, v)] for j in 3:NCLS, v in 1:NL) for s in 1:ns, k in 1:S.nk]))
