# Fit HBAD v2 on Sim A tracked cohorts (TRAIN) by CMA-ES on the joint NLL; report TRAIN and VAL (selection) parts and diagnostics.
#   HBAD2_EVALS=40000 ./julia_gdal.sh --project=. --threads=8 tools/hbad2/fit_simA.jl <tag> [init.csv]
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "cmaes.jl"))
tag = length(ARGS) >= 1 ? ARGS[1] : "v2a"; OUTD = joinpath(R2DIR, "fit_" * tag); mkpath(OUTD)
logio = open(joinpath(OUTD, "log.txt"), "w"); lg(s) = (println(s); println(logio, s); flush(logio); flush(stdout))
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
lon = CSV.read("/workspace/species_longevity_ref.csv", DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
TR = build_cohorts(tr, "train", keys_); VA = build_cohorts(tr, "val", keys_)
lg(@sprintf("species keys %d: %s | growth form %d | size split %s", nsp, join(keys_, ","), FORM, string(SIZE_BREAKS)))
lg(@sprintf("train: %d sites, %d cohorts (%d trees) | val: %d sites, %d cohorts | birth class %d yr", length(TR.S.off) - 1, length(TR.S.sp), sum(TR.coh.n1),
            length(VA.S.off) - 1, length(VA.S.sp), BIRTH_W))
# ---- data-based start
pri = CSV.read(joinpath(@__DIR__, "..", "..", "results", "hbad", "priors.csv"), DataFrame); pk = Dict(String(r.key) => r for r in eachrow(pri))
function init_theta()
  θ = zeros(nsp * NF + NGLOB)
  for (s, k) in enumerate(keys_)
    t = tr[(tr.key .== k) .& (tr.set .== "train"), :]
    f = t[(t.out .== "alive") .& in.(t.cclcd, Ref((1.0, 2.0, 3.0))), :]
    Hmax = 1.1 * quantile(t.ht_m, 0.99); Dmax = 1.3 * quantile(t.dia_cm, 0.99); c = 1.3
    Dm = median(f.dia_cm); Hm = median(f.ht_m)
    tD = mean(max.(f.dia2_cm .- f.dia_cm, 0) ./ f.dt); tH = mean(max.(f.ht2_m .- f.ht_m, 0) ./ f.dt)
    rD = tD / (c * Dm * ((Dmax / Dm)^(1 / c) - 1)); rH = tH / (c * Hm * ((Hmax / Hm)^(1 / c) - 1)); cH = c; cD = c
    if FORM == 2                                # power-exponential start: dD peaks at 0.5·40 = 20 cm, dH declines with H
      cD = 0.5; Dmax = 40.0; rD = tD / ((Dm / 20)^cD * exp(-Dm / Dmax))
      cH = 0.3; Hmax = 10.0; rH = tH / ((Hm / 15)^cH * exp(-Hm / Hmax))
    end
    mort = sum(t.out .== "dead") / sum(t.out .!= "missing") / mean(t.dt)
    sdm = haskey(pk, k) && !ismissing(pk[k].sdimax_p95_all) ? pk[k].sdimax_p95_all : pk["_" * (k[1] == '_' ? k[2:2] : first(t.grp))].sdimax_p95_all
    v = Dict(:Hmax => Hmax, :rH => rH, :cH => cH, :Dmax => Dmax, :rD => rD, :cD => cD, :gOD => 1.0, :gRD => 0.0, :gOH => 0.5, :gRH => 0.2,
             :mbg => log(0.7 * mort), :mD => -0.3, :msup => 0.02, :sdimax => sdm)
    for (j, nm) in enumerate(FREE); θ[(s-1)*NF+j] = enc(nm, dec(nm, enc(nm, v[nm]))); end
  end
  θ[nsp*NF+1:end] .= 0.0; θ
end
θ0 = length(ARGS) >= 2 ? read_theta(ARGS[2], nsp) : init_theta()
function obj(θ, D)
  P, G = decode(θ, nsp, longev); simulate!(D.S, P, G); nll_parts(D.S, D.obs)
end
for _ in 1:2; obj(θ0, TR); end
t0 = time(); for _ in 1:10; obj(θ0, TR); end; lg(@sprintf("eval time train %.1f ms", (time() - t0) * 100))
pt = obj(θ0, TR); pv = obj(θ0, VA)
lg(@sprintf("START train total %.1f (mort %.1f D %.1f H %.1f σD %.4f σH %.4f) | val total %.1f (mort %.1f D %.1f H %.1f)", pt.total, pt.mort, pt.D, pt.H, pt.sigD, pt.sigH, pv.total, pv.mort, pv.D, pv.H))
# null: no growth, per-species constant survival (MLE on train)
function null_parts(D, surv)
  nm = 0.0; sD = 0.0; sH = 0.0; c = 0
  for i in eachindex(D.S.N)
    n1 = D.coh.n1[i]; n1 == 0 && continue; s = D.coh.s[i]; p = clamp(surv[D.S.sp[i]]^D.coh.dt[i], 1e-9, 1 - 1e-9)
    nm -= s * log(p) + (n1 - s) * log(1 - p)
    s > 0 && (sD += s * log(D.S.D0[i] / D.coh.D2o[i])^2; sH += s * log(D.S.H0[i] / D.coh.H2o[i])^2; c += 1)
  end
  (total=nm + 0.5c * log(sD / c) + 0.5c * log(sH / c), mort=nm, D=0.5c * log(sD / c), H=0.5c * log(sH / c))
end
surv = [begin m = TR.coh[TR.S.sp .== s, :]; (sum(m.s) / sum(m.n1))^(1 / mean(m.dt)) end for s in 1:nsp]
nt = null_parts(TR, surv); nv = null_parts(VA, surv)
lg(@sprintf("NULL  train total %.1f (mort %.1f D %.1f H %.1f) | val total %.1f (mort %.1f D %.1f H %.1f)", nt.total, nt.mort, nt.D, nt.H, nv.total, nv.mort, nv.D, nv.H))
maxe = parse(Int, get(ENV, "HBAD2_EVALS", "20000"))
res = cmaes(θ -> obj(θ, TR).total, θ0, parse(Float64, get(ENV, "HBAD2_SIGMA", "0.3")); λ=parse(Int, get(ENV, "HBAD2_LAMBDA", "32")), maxevals=maxe, logf=lg, log_every=25,
            callback=(g, b, fb) -> (g % 100 == 0 && CSV.write(joinpath(OUTD, "theta.csv"), DataFrame(theta=b))))
CSV.write(joinpath(OUTD, "theta.csv"), DataFrame(theta=res.x))
pt = obj(res.x, TR); pv = obj(res.x, VA)
lg(@sprintf("FIT   train total %.1f (mort %.1f D %.1f H %.1f σD %.4f σH %.4f) | val total %.1f (mort %.1f D %.1f H %.1f σD %.4f σH %.4f) evals %d",
            pt.total, pt.mort, pt.D, pt.H, pt.sigD, pt.sigH, pv.total, pv.mort, pv.D, pv.H, pv.sigD, pv.sigH, res.evals))
P, G = decode(res.x, nsp, longev)
pdf = DataFrame([nm => [getfield(P[s], nm) for s in 1:nsp] for nm in HBAD2.SP_PARAM_NAMES]); insertcols!(pdf, 1, :key => keys_)
CSV.write(joinpath(OUTD, "params.csv"), pdf); lg(string(pdf)); lg(@sprintf("global mD2 %.3f pH %.3f pD %.3f pM %.3f mSD %.3f", G.mD2, G.pH, G.pD, G.pM, G.mSD))
