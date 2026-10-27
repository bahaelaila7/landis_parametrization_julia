# HBAD4P stage 1 of the two-stage calibration (docs/hbad4p_decisions.md H9 / H19): CMA-ES on the TRAIN NLL with κ fixed at 0 (stage1_space4), from the HBAD3P
# warm start (params4.jl warm_start4). Writes the best point as a save_z4 CSV (+ its space signature `.sig`) and a log; kappa_profile4.jl reads it.
#   HBAD4_SPLIT_CSV=… HBAD2_DATA_DIR=<fold product> ./julia_gdal.sh --project=. --threads=4 tools/hbad4/calib4.jl <hbad3p fit dir> [out dir]
# env: HBAD4P_CALIB_EVALS (budget, default 20000), HBAD4P_CALIB_SIGMA (0.1), HBAD4P_CALIB_SEED (1), HBAD4P_CALIB_LAMBDA (CMA-ES default), HBAD4P_CALIB_EVERY
#      (write the running best every n generations, 20), HBAD4P_CALIB_SUB (dry run: the first n train sites only — a test of the tool, NOT a calibration).
include(joinpath(@__DIR__, "data4.jl")); include(joinpath(@__DIR__, "params4.jl")); include(joinpath(@__DIR__, "..", "hbad2", "cmaes.jl"))
length(ARGS) >= 1 || error("usage: calib4.jl <hbad3p fit dir> [out dir]")
const EVALS = parse(Int, get(ENV, "HBAD4P_CALIB_EVALS", "20000")); const SIG = parse(Float64, get(ENV, "HBAD4P_CALIB_SIGMA", "0.1"))
const SEED = parse(Int, get(ENV, "HBAD4P_CALIB_SEED", "1")); const EVERY = parse(Int, get(ENV, "HBAD4P_CALIB_EVERY", "20"))
const SUB = parse(Int, get(ENV, "HBAD4P_CALIB_SUB", "0"))
outdir = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "..", "results", "hbad4", "calib_stage1"); mkpath(outdir)
logio = open(joinpath(outdir, "calib4.log"), "a"); lg(s) = (println(s); println(logio, s); flush(stdout); flush(logio))
CS = cached_sets(["train", "val", "test"]); WS = warm_start4(CS, ARGS[1]; quiet=true)
SP = stage1_space4(CS, WS)
D = let Dn = CS.sets["train"]
  SUB <= 0 ? Dn : begin
    S = Dn.S; idx = 1:min(SUB, HBAD4P.nsites(S)); nc = S.off[2] - S.off[1]; r = 1:last(idx)*nc
    at = [i => HBAD4P.atoms0(S, i) for i in r if S.N0[i] > 0]
    S2 = HBAD4P.make_sites4(S.off[1:last(idx)+1], S.nk, S.sp[r], S.cl[r], S.ln[r], at, S.H0[r], S.A0[r], S.nyr[idx], S.frac[idx], S.planted[idx], S.W[idx],
                            S.Wn[idx], S.T[idx], S.C[idx], S.seed0[:, idx], S.lz, S.lw)
    (S=S2, n=Dn.n[:, :, idx], D2=Dn.D2[:, :, idx], Hs=Dn.Hs[:, :, idx], nH=Dn.nH[:, :, idx], fs2=Dn.fs2[idx], fm2=Dn.fm2[idx])
  end
end
f(z) = begin P, G, Ob = SP.decode(z); HBAD4P.simulate!(D.S, P, G); nll4(D, Ob).total end
lg("calib4 stage 1 $(Dates.now()) | $(length(SP.z0)) coordinates (κ fixed 0; fixed $(SP.fix)) | $(HBAD4P.nsites(D.S)) train sites | budget $(EVALS) σ $(SIG) seed $(SEED)" *
   (SUB > 0 ? " | DRY RUN on $(SUB) sites: not a calibration" : ""))
f0 = f(SP.z0); lg(@sprintf("warm start train NLL %.2f", f0))
best = Ref((f=f0, x=copy(SP.z0)))
cb(gen, x, fx) = (fx < best[].f && (best[] = (f=fx, x=copy(x))); gen % EVERY == 0 && save_z4(joinpath(outdir, "z_stage1_running.csv"), SP, best[].x))
lam = haskey(ENV, "HBAD4P_CALIB_LAMBDA") ? parse(Int, ENV["HBAD4P_CALIB_LAMBDA"]) : 4 + floor(Int, 3log(length(SP.z0)))
r = cmaes(f, copy(SP.z0), SIG; λ=lam, maxevals=EVALS, rng=Xoshiro(SEED), logf=lg, callback=cb)
out = save_z4(joinpath(outdir, "z_stage1.csv"), SP, r.x)
lg(@sprintf("done: %d evaluations, train NLL %.2f (warm start %.2f, %+.2f) → %s (+ .sig)", r.evals, r.f, f0, r.f - f0, out)); close(logio)
