# HBAD4P stage 2 of the two-stage calibration (docs/hbad4p_decisions.md H9): a profile over the lane-mixing rate κ.
# Stage 1 fits everything with κ fixed at 0 (HBAD4P_FIX="κ:0", or the default: κ starts at 0 and may be fixed with HBAD4P_FIX). This tool takes the stage-1
# optimum (a save_z4 CSV), and for κ = 0 and every κ of the profile fixes κ (space4 fix_extra), warm-starts all other free coordinates at the stage-1 values
# and runs the same SHORT CMA-ES on the train NLL (+ a short refit of the observation coordinates); then scores VAL counts with the train-fitted observation
# parameters, 3 CMA-ES seeds per κ (HBAD4P_PROFILE_SEEDS). κ > 0 is kept only if its minimum val count NLL over the seeds beats κ = 0's maximum by
# ≥ max(HBAD4P_PROFILE_GAIN = 20, 2 × (pooled replicate sd + 0.005 nats/site × n_val)) (decisions H9 / H21).
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
Dt = first_sites4(CS.sets["train"], SUB); Dv = first_sites4(CS.sets["val"], SUB)
SP1 = stage1_space4(CS, WS); z1 = ARGS[2] == "-" ? copy(SP1.z0) : load_z4(ARGS[2], SP1)     # (refuses a file of another space, H19)
P1, G1, Ob1 = SP1.decode(z1); G1.κ == 0 || @warn "stage-1 κ = $(G1.κ) (expected 0)"
st = starts4(P1, G1, Ob1)
iob(SP) = [findfirst(==(String(nm)), SP.names) for nm in OBS4]
"Train NLL of z, the val count NLL with z's (train-fitted) observation parameters."
function score(SP, z)
  P, G, Ob = SP.decode(z); HBAD4P.simulate!(Dt.S, P, G); a = nll4(Dt, Ob); HBAD4P.simulate!(Dv.S, P, G); b = nll4(Dv, Ob); (train=a.total, val=b.total, valc=b.count)
end
rows = []
const SEEDS = parse.(Int, split(get(ENV, "HBAD4P_PROFILE_SEEDS", "1,2,3"), ","))
# every κ, κ = 0 included (the stage-1 optimum re-run with the same budget, so all κ get equal effort), with 3 CMA-ES seeds each (audit r5 I-5, H21): the short
# CMA-ES on all free coordinates, then a short refit of the observation coordinates on the CACHED simulated state (the dynamics do not depend on them, r5 M-2)
for κ in vcat(0.0, KAPPAS), seed in SEEDS
  SP = space4(CS.keys, CS.TR, CS.sD, CS.sH; st..., fix_extra=Dict(:κ => κ)); z0 = copy(SP.z0)
  f(z) = begin P, G, Ob = SP.decode(z); HBAD4P.simulate!(Dt.S, P, G); nll4(Dt, Ob).total end
  r = cmaes(f, z0, SIG; maxevals=EVALS, rng=Xoshiro(seed), logf=s -> nothing); z = copy(r.x); io = iob(SP)
  let (P, G, _) = SP.decode(z); HBAD4P.simulate!(Dt.S, P, G); end                    # (the state of z, once)
  fo(x) = begin zz = copy(z); zz[io] = x; nll4(Dt, SP.decode(zz)[3]).total end
  ro = cmaes(fo, z[io], 0.3; λ=12, maxevals=min(EVALS, 300), rng=Xoshiro(seed), logf=s -> nothing); z[io] = ro.x; sc = score(SP, z)
  push!(rows, (κ=κ, seed=seed, train=sc.train, val=sc.val, valc=sc.valc, evals=r.evals + ro.evals)); save_z4(joinpath(outdir, "z_kappa_$(κ)_seed$(seed).csv"), SP, z)
  println(@sprintf("κ = %-6.3f seed %d  train %.1f | val %.1f (count %.1f) | %d evals", κ, seed, sc.train, sc.val, sc.valc, r.evals + ro.evals)); flush(stdout)
end
T = DataFrame(rows); CSV.write(joinpath(outdir, "kappa_profile.csv"), T)
# decision rule (H21): κ > 0 is kept only if its MINIMUM val count NLL over the seeds beats κ = 0's MAXIMUM by ≥ max(GAIN, 2 (pooled replicate sd + 0.005 nats/site
# × n_val sites)); among the κ that pass, the lowest minimum
g = combine(groupby(T, :κ), :valc => minimum => :vmin, :valc => maximum => :vmax, :valc => (x -> length(x) > 1 ? var(x) : 0.0) => :v, nrow => :n)
sdp = sqrt(sum((g.n .- 1) .* g.v) / max(sum(g.n .- 1), 1)); nval = HBAD4P.nsites(Dv.S); margin = max(GAIN, 2 * (sdp + 0.005 * nval))
v0max = g.vmax[g.κ .== 0.0][1]; pas = g[(g.κ .> 0) .& (g.vmin .<= v0max - margin), :]
keep = nrow(pas) > 0; kbest = keep ? pas.κ[argmin(pas.vmin)] : 0.0
println(@sprintf("pooled replicate sd %.2f nats, margin %.1f nats (n_val %d); κ = 0 val count max %.1f", sdp, margin, nval, v0max))
for r in eachrow(g); println(@sprintf("  κ = %-6.3f val count min %.1f max %.1f (min − κ0 max %+.1f)", r.κ, r.vmin, r.vmax, r.vmin - v0max)); end
println("decision (H9 / H21): ", keep ? "κ = $(kbest)" : "κ = 0 (no κ > 0 beats κ = 0 by the margin)", SUB > 0 ? "  [DRY RUN on $(SUB) sites: not a profile]" : "",
        " → ", joinpath(outdir, "kappa_profile.csv"))
