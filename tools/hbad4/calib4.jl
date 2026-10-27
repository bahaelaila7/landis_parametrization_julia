# HBAD4P stage 1 of the two-stage calibration (docs/hbad4p_decisions.md H9 / H19 / H25): CMA-ES on the TRAIN NLL with κ fixed at 0 (stage1_space4), from the
# HBAD3P warm start (params4.jl warm_start4), REPLICATED over 3 CMA-ES seeds from the same start (the pre-registered acceptance check of H25):
#   - every seed's best point is scored on VAL (total NLL, with its own train-fitted observation parameters);
#   - ACCEPTED only if the range of the val NLL across the seeds ≤ max(20, 2 · pooled sd) nats (the H21 margin form). The pooled sd is the pooled WITHIN-seed
#     sd of the val NLL under a small jitter of each seed's best point (HBAD4P_CALIB_NOISE_K points at step HBAD4P_CALIB_NOISE_H in every coordinate), i.e.
#     the objective's own noise at the optimum; NOT the between-seed sd, which can never fail the test (for 3 values range ≤ 2 sd always; H25 implementation);
#   - mW, am, pM, ε:MAVI2 are reported as WEAKLY IDENTIFIED (≈ 0–2 nats at ±0.05) with their cross-seed spread; their fitted values are not to be interpreted.
# Writes per seed z_stage1_seed<s>.csv (+ .sig); the seed with the lowest TRAIN NLL as z_stage1.csv if accepted, else z_stage1_REJECTED.csv (kappa_profile4.jl
# reads z_stage1.csv, so a rejected stage 1 cannot feed stage 2 by accident); calib4_stage1.json with the verdict, the per-seed values and the weak
# coordinates; and a log.
#   HBAD4_SPLIT_CSV=… HBAD2_DATA_DIR=<fold product> ./julia_gdal.sh --project=. --threads=4 tools/hbad4/calib4.jl <hbad3p fit dir> [out dir]
# env: HBAD4P_CALIB_EVALS (budget PER SEED, default 20000), HBAD4P_CALIB_SIGMA (0.1), HBAD4P_CALIB_SEEDS (1,2,3; fewer than 3 → verdict NOT ASSESSED, not
#      accepted), HBAD4P_CALIB_LAMBDA (CMA-ES default), HBAD4P_CALIB_EVERY (write the running best every n generations, 20), HBAD4P_CALIB_GAIN (the 20-nat
#      floor), HBAD4P_CALIB_NOISE_K (8), HBAD4P_CALIB_NOISE_H (1e-3), HBAD4P_CALIB_SUB (dry run: the first n train / val sites only — a test of the tool, NOT
#      a calibration; the verdict is then labelled DRY RUN).
include(joinpath(@__DIR__, "data4.jl")); include(joinpath(@__DIR__, "params4.jl")); include(joinpath(@__DIR__, "..", "hbad2", "cmaes.jl"))
using JSON3
length(ARGS) >= 1 || error("usage: calib4.jl <hbad3p fit dir> [out dir]")
haskey(ENV, "HBAD4P_CALIB_SEED") && error("HBAD4P_CALIB_SEED is replaced by HBAD4P_CALIB_SEEDS (H25: stage 1 is replicated over 3 seeds)")
const EVALS = parse(Int, get(ENV, "HBAD4P_CALIB_EVALS", "20000")); const SIG = parse(Float64, get(ENV, "HBAD4P_CALIB_SIGMA", "0.1"))
const SEEDS = parse.(Int, split(get(ENV, "HBAD4P_CALIB_SEEDS", "1,2,3"), ",")); const EVERY = parse(Int, get(ENV, "HBAD4P_CALIB_EVERY", "20"))
const GAIN = parse(Float64, get(ENV, "HBAD4P_CALIB_GAIN", "20")); const NK = parse(Int, get(ENV, "HBAD4P_CALIB_NOISE_K", "8"))
const NH = parse(Float64, get(ENV, "HBAD4P_CALIB_NOISE_H", "1e-3")); const SUB = parse(Int, get(ENV, "HBAD4P_CALIB_SUB", "0"))
allunique(SEEDS) || error("HBAD4P_CALIB_SEEDS must be distinct: $(SEEDS)"); NK >= 2 || error("HBAD4P_CALIB_NOISE_K must be ≥ 2")
const WEAK = ["mW", "am", "pM", "ε:MAVI2"]                                             # (H25: weakly identified, ≈ 0–2 nats at ±0.05)
outdir = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "..", "results", "hbad4", "calib_stage1"); mkpath(outdir)
logio = open(joinpath(outdir, "calib4.log"), "a"); lg(s) = (println(s); println(logio, s); flush(stdout); flush(logio))
CS = cached_sets(["train", "val", "test"]); WS = warm_start4(CS, ARGS[1]; quiet=true)
SP = stage1_space4(CS, WS)
Dt = first_sites4(CS.sets["train"], SUB); Dv = first_sites4(CS.sets["val"], SUB)
ntr = HBAD4P.nsites(Dt.S); nva = HBAD4P.nsites(Dv.S)
f(z) = begin P, G, Ob = SP.decode(z); HBAD4P.simulate!(Dt.S, P, G); nll4(Dt, Ob).total end
"Val NLL (total, count) of z with z's own (train-fitted) observation parameters."
fval(z) = begin P, G, Ob = SP.decode(z); HBAD4P.simulate!(Dv.S, P, G); r = nll4(Dv, Ob); (total=r.total, count=r.count) end
lg("calib4 stage 1 $(Dates.now()) | $(length(SP.z0)) coordinates (κ fixed 0; fixed $(SP.fix)) | $(ntr) train / $(nva) val sites | budget $(EVALS) per seed, " *
   "σ $(SIG), seeds $(SEEDS)" * (SUB > 0 ? " | DRY RUN on $(SUB) sites: not a calibration" : ""))
f0 = f(SP.z0); v0 = fval(SP.z0); lg(@sprintf("warm start train NLL %.2f | val NLL %.2f (count %.2f)", f0, v0.total, v0.count))
lam = haskey(ENV, "HBAD4P_CALIB_LAMBDA") ? parse(Int, ENV["HBAD4P_CALIB_LAMBDA"]) : 4 + floor(Int, 3log(length(SP.z0)))
runs = []
for seed in SEEDS
  best = Ref((f=f0, x=copy(SP.z0)))
  cb(gen, x, fx) = (fx < best[].f && (best[] = (f=fx, x=copy(x))); gen % EVERY == 0 && save_z4(joinpath(outdir, "z_stage1_running_seed$(seed).csv"), SP, best[].x))
  r = cmaes(f, copy(SP.z0), SIG; λ=lam, maxevals=EVALS, rng=Xoshiro(seed), logf=s -> lg("[seed $(seed)] " * s), callback=cb)
  z = copy(r.x); ft = f(z); v = fval(z)
  # the objective's noise at this optimum: the val NLL at NK jittered points z + NH·N(0, I) (its own RNG stream, independent of the CMA-ES one)
  rj = Xoshiro(10_000 + seed); vj = [fval(z .+ NH .* randn(rj, length(z))).total for _ in 1:NK]
  out = save_z4(joinpath(outdir, "z_stage1_seed$(seed).csv"), SP, z)
  push!(runs, (seed=seed, evals=r.evals, train=ft, val=v.total, valc=v.count, jit_sd=std(vj), jit=vj, z=z, value=values4(SP, z), file=basename(out)))
  lg(@sprintf("[seed %d] done: %d evaluations, train NLL %.2f (warm start %+.2f) | val NLL %.2f (count %.2f) | jitter sd %.2f nats (%d points, step %.0e) → %s",
              seed, r.evals, ft, ft - f0, v.total, v.count, std(vj), NK, NH, out))
end
# ---------------------------------------------------------------- the H25 acceptance check
vals = [r.val for r in runs]; rng_v = maximum(vals) - minimum(vals); sd_pool = sqrt(sum(r.jit_sd^2 for r in runs) / length(runs))   # (equal df per seed)
sd_between = length(runs) > 1 ? std(vals) : NaN; margin = max(GAIN, 2sd_pool)
verdict = length(runs) < 3 ? "NOT ASSESSED (fewer than 3 seeds): not accepted" : rng_v <= margin ? "ACCEPTED" : "REJECTED (roughness is the first suspect, H25)"
accepted = startswith(verdict, "ACCEPTED")
SUB > 0 && (verdict *= " [DRY RUN on $(SUB) sites, $(EVALS) evaluations per seed: a test of the tool, NOT a calibration]")
ib = argmin([r.train for r in runs]); rb = runs[ib]                                    # (the seed carried forward: lowest TRAIN NLL, so val stays a check)
fsel = save_z4(joinpath(outdir, accepted ? "z_stage1.csv" : "z_stage1_REJECTED.csv"), SP, rb.z)
accepted || (isfile(joinpath(outdir, "z_stage1.csv")) && @warn "an older z_stage1.csv exists in $(outdir); this run did NOT replace it (not accepted)")
weak = []
for nm in WEAK
  i = findfirst(==(nm), SP.names)
  i === nothing && (push!(weak, (name=nm, present=false)); continue)
  zs = [r.z[i] for r in runs]; vs = [r.value[i] for r in runs]
  push!(weak, (name=nm, present=true, z=zs, value=vs, z_range=maximum(zs) - minimum(zs), value_range=maximum(vs) - minimum(vs), z_start=SP.z0[i]))
end
lg(@sprintf("H25 replication: val NLL per seed %s | range %.2f nats vs margin max(%.0f, 2 × pooled jitter sd %.2f) = %.2f (between-seed sd %.2f, reported) → %s",
            join([@sprintf("%d: %.2f", r.seed, r.val) for r in runs], ", "), rng_v, GAIN, sd_pool, margin, sd_between, verdict))
for w in weak
  w.present ? lg(@sprintf("  weakly identified (H25; not interpreted) %-8s z %s (range %.4f) | value %s (range %.4g)", w.name, join([@sprintf("%.4f", x) for x in w.z], " / "),
                          w.z_range, join([@sprintf("%.4g", x) for x in w.value], " / "), w.value_range)) :
              lg("  weakly identified (H25) $(w.name): not a coordinate of this space")
end
J = Dict(:tool => "calib4.jl stage 1", :time => string(Dates.now()), :dry_run => SUB > 0, :sub_sites => SUB, :n_train => ntr, :n_val => nva,
         :n_coordinates => length(SP.z0), :fixed => string(SP.fix), :budget_per_seed => EVALS, :sigma => SIG, :lambda => lam, :seeds => SEEDS,
         :warm_start => Dict(:train => f0, :val => v0.total, :val_count => v0.count),
         :per_seed => [Dict(:seed => r.seed, :evals => r.evals, :train => r.train, :val => r.val, :val_count => r.valc, :jitter_sd => r.jit_sd, :jitter_val => r.jit,
                            :file => r.file) for r in runs],
         :acceptance => Dict(:rule => "H25: range of val NLL across seeds <= max(gain, 2 * pooled within-seed jitter sd)", :gain => GAIN, :noise_k => NK,
                             :noise_h => NH, :val_range => rng_v, :pooled_sd => sd_pool, :between_seed_sd => isnan(sd_between) ? nothing : sd_between, :margin => margin, :accepted => accepted,
                             :verdict => verdict, :carried_seed => rb.seed, :carried_file => basename(fsel)),
         :weakly_identified => [w.present ? Dict(:name => w.name, :z => w.z, :value => w.value, :z_range => w.z_range, :value_range => w.value_range,
                                                 :z_start => w.z_start) : Dict(:name => w.name, :present => false) for w in weak])
open(joinpath(outdir, "calib4_stage1.json"), "w") do io; JSON3.pretty(io, J); println(io); end
lg("stage 1 carried seed $(rb.seed) (lowest train NLL) → $(fsel) (+ .sig); verdict and per-seed values → $(joinpath(outdir, "calib4_stage1.json"))"); close(logio)
