# HBAD4P stage 2 of the two-stage calibration (docs/hbad4p_decisions.md H9): a profile over the lane-mixing rate κ.
# Stage 1 fits everything with κ fixed at 0 (HBAD4P_FIX="κ:0", or the default: κ starts at 0 and may be fixed with HBAD4P_FIX). This tool takes the stage-1
# optimum (a save_z4 CSV), and for κ = 0 and every κ of the profile fixes κ (space4 fix_extra), warm-starts all other free coordinates at the stage-1 values
# and runs the same SHORT CMA-ES on the train NLL (+ a short refit of the observation coordinates); then scores VAL counts with the train-fitted observation
# parameters. κ > 0 is kept only if it improves the val count NLL over κ = 0 by ≥ HBAD4P_PROFILE_GAIN nats (default 20).
#   HBAD4_SPLIT_CSV=… HBAD2_DATA_DIR=<fold product> ./julia_gdal.sh --project=. --threads=4 tools/hbad4/kappa_profile4.jl <hbad3p fit dir> <stage1 z csv> [out dir]
# env: HBAD4P_PROFILE_KAPPAS (default 0.025,0.05,0.1), HBAD4P_PROFILE_EVALS (CMA-ES evaluations per κ, default 600), HBAD4P_PROFILE_SIGMA (0.05),
#      HBAD4P_PROFILE_GAIN (20), HBAD4P_PROFILE_SUB (dry run: the first n train / val sites only — a test of the tool, NOT a profile).
# A stage-1 file `-` uses the warm start (only meaningful for a dry run).
include(joinpath(@__DIR__, "data4.jl")); include(joinpath(@__DIR__, "params4.jl")); include(joinpath(@__DIR__, "..", "hbad2", "cmaes.jl"))
length(ARGS) >= 2 || error("usage: kappa_profile4.jl <hbad3p fit dir> <stage1 z csv | -> [out dir]")
const KAPPAS = parse.(Float64, split(get(ENV, "HBAD4P_PROFILE_KAPPAS", "0.025,0.05,0.1"), ","))
const EVALS = parse(Int, get(ENV, "HBAD4P_PROFILE_EVALS", "600")); const SIG = parse(Float64, get(ENV, "HBAD4P_PROFILE_SIGMA", "0.05"))
const GAIN = parse(Float64, get(ENV, "HBAD4P_PROFILE_GAIN", "20")); const SUB = parse(Int, get(ENV, "HBAD4P_PROFILE_SUB", "0"))
outdir = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "..", "..", "results", "hbad4", "kappa_profile"); mkpath(outdir)
CS = cached_sets(["train", "val", "test"]); WS = warm_start4(CS, ARGS[1]; quiet=true)
"A set restricted to its first n sites (dry runs)."
function first_sites(Dn, n)
  n <= 0 && return Dn
  S = Dn.S; idx = 1:min(n, HBAD4P.nsites(S)); nc = S.off[2] - S.off[1]; r = 1:last(idx)*nc
  at = [i => HBAD4P.atoms0(S, i) for i in r if S.N0[i] > 0]
  S2 = HBAD4P.make_sites4(S.off[1:last(idx)+1], S.nk, S.sp[r], S.cl[r], S.ln[r], at, S.H0[r], S.A0[r], S.nyr[idx], S.frac[idx], S.planted[idx], S.W[idx], S.Wn[idx],
                          S.T[idx], S.C[idx], S.seed0[:, idx], S.lz, S.lw)
  (S=S2, n=Dn.n[:, :, idx], D2=Dn.D2[:, :, idx], Hs=Dn.Hs[:, :, idx], nH=Dn.nH[:, :, idx], fs2=Dn.fs2[idx], fm2=Dn.fm2[idx])
end
Dt = first_sites(CS.sets["train"], SUB); Dv = first_sites(CS.sets["val"], SUB)
SP1 = WS.SP; z1 = ARGS[2] == "-" ? copy(SP1.z0) : load_z4(ARGS[2], SP1)
P1, G1, Ob1 = SP1.decode(z1); G1.κ == 0 || @warn "stage-1 κ = $(G1.κ) (expected 0)"
st = starts4(P1, G1, Ob1)
iob(SP) = [findfirst(==(String(nm)), SP.names) for nm in OBS4]
"Train NLL of z, the val count NLL with z's (train-fitted) observation parameters."
function score(SP, z)
  P, G, Ob = SP.decode(z); HBAD4P.simulate!(Dt.S, P, G); a = nll4(Dt, Ob); HBAD4P.simulate!(Dv.S, P, G); b = nll4(Dv, Ob); (train=a.total, val=b.total, valc=b.count)
end
rows = []
# every κ, κ = 0 included (the stage-1 optimum re-run with the same budget, so all κ get equal effort): the short CMA-ES on all free coordinates, then a short
# refit of the observation coordinates
for κ in vcat(0.0, KAPPAS)
  SP = space4(CS.keys, CS.TR, CS.sD, CS.sH; st..., fix_extra=Dict(:κ => κ)); z0 = copy(SP.z0)
  f(z) = begin P, G, Ob = SP.decode(z); HBAD4P.simulate!(Dt.S, P, G); nll4(Dt, Ob).total end
  r = cmaes(f, z0, SIG; maxevals=EVALS, logf=s -> nothing); z = copy(r.x); io = iob(SP)
  fo(x) = begin zz = copy(z); zz[io] = x; P, G, Ob = SP.decode(zz); HBAD4P.simulate!(Dt.S, P, G); nll4(Dt, Ob).total end
  ro = cmaes(fo, z[io], 0.3; λ=12, maxevals=min(EVALS, 300), logf=s -> nothing); z[io] = ro.x; sc = score(SP, z)   # (the same observation refit as κ = 0)
  push!(rows, (κ=κ, train=sc.train, val=sc.val, valc=sc.valc, evals=r.evals + ro.evals)); save_z4(joinpath(outdir, "z_kappa_$(κ).csv"), SP, z)
  println(@sprintf("κ = %-6.3f train %.1f | val %.1f (count %.1f, %+.1f vs κ = 0) | %d evals", κ, sc.train, sc.val, sc.valc, sc.valc - rows[1].valc, r.evals + ro.evals)); flush(stdout)
end
T = DataFrame(rows); T.dvalc = T.valc .- T.valc[1]; CSV.write(joinpath(outdir, "kappa_profile.csv"), T)
b = argmin(T.dvalc); keep = T.κ[b] > 0 && -T.dvalc[b] >= GAIN
println("decision (H9): ", keep ? "κ = $(T.κ[b]) (val count NLL $(round(T.dvalc[b]; digits=1)) nats vs κ = 0)" : "κ = 0 (no κ improves the val count NLL by ≥ $(GAIN) nats)",
        SUB > 0 ? "  [DRY RUN on $(SUB) sites: not a profile]" : "", " → ", joinpath(outdir, "kappa_profile.csv"))
