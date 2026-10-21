# HBAD3P pipeline stage 7 — recruitment fit, CONSISTENT with the free-run recruitment mechanism (tools/hbad3/lib3p.jl header):
#   Y_sk = saplings (2.54 <= D < 12.7) of key k first tallied at v2 on the forested microplot of subplot s, one row per (TRAIN natural clean interval, subplot
#          with known microplot status, key); E[Y_sk] = e^{lF_k} f_micr2 Σ_y Δ_y e^{−λ0 e^{λ1 (ST_k−3)} RD_y} (seed_ky ? 1 : ε0) s_ky (probe recruits in the
#          observed v1 stand under the fit's kernel and density convention). Poisson likelihood, CMA-ES. ln F (per ha) = lF + ln 4 (one microplot sapling of a
#          subplot site = 4 × RSCALE trees/ha).
# Writes <fit dir>/recruitment.csv (lnF:<key>, λ0, λ1, ε0) and recruit_log.txt: fit, VAL Poisson deviance vs a per-key constant-rate null, observed vs
# expected counts by key / RD class / seed (calibration).
#   (pipeline env) ./julia_gdal.sh --project=. --threads=N tools/hbad3/recruit3p.jl <fit3p tag>
include(joinpath(@__DIR__, "common3.jl")); include(joinpath(@__DIR__, "..", "hbad2", "cmaes.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plugins", "HBAD3", "HBAD3P.jl")); import .HBAD3P
include(joinpath(@__DIR__, "lib3p.jl"))
tag = ARGS[1]; FD = joinpath(R3DIR, "fit3p_" * tag); check_meta(FD); FAREA = meta_farea(FD)
logio = open(joinpath(FD, "recruit_log.txt"), "w"); lg(s) = (println(s); println(logio, s); flush(logio); flush(stdout))
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_); kix = Dict(k => i for (i, k) in enumerate(keys_))
M = load_fit3p(FD, keys_); T = interval_tables(keys_); wp = water_pairs(); IF = interval_forest(T.iv, forest_table())
ivd = Dict(r.iid => r for r in eachrow(T.iv)); regiids = Set(tr.iid)       # load_trees: region + clean (no treatment / disturbance / harvest, dt > 2)
rawby = Dict(k.iid => DataFrame(g) for (k, g) in pairs(groupby(T.raw, :iid)))
igs = T.ig[T.ig.dia_cm .< 12.7, :]; igcnt = Dict((r.iid, r.sb, r.key) => r.n for r in eachrow(combine(groupby(igs, [:iid, :sb, :key]), nrow => :n)))
"Probe sites and observation rows of one set."
function recr_data(set)
  ids = sort([i for i in regiids if ivd[i].set == set && !ivd[i].planted])
  sites = NamedTuple[]; K = Int[]; Y = Float64[]; FM = Float64[]; SI = Int[]; nunk = 0; nnf_y = 0; nnf_v1 = 0
  for i in ids
    r = ivd[i]; t = get(rawby, i, T.raw[1:0, :]); y1 = year(r.d1)
    sp_plot = BitVector([any((t.key .== k) .& (t.dia_cm .>= 12.7)) for k in keys_])
    for sb in 1:4
      f1, fm2 = IF[(i, sb)]
      if isnan(fm2); nunk += 1; continue; end
      if fm2 == 0; nnf_y += sum(get(igcnt, (i, sb, k), 0) for k in keys_); continue; end     # no forested microplot at v2: no tally area
      ts = t[t.sb .== sb, :]; a = FAREA ? area_of(f1) : 1.0
      if f1 == 0 && nrow(ts) == 0; nnf_v1 += 1; continue; end                     # nonforest and empty at v1: no recruitment (as in the free run)
      push!(sites, (coh=subplot_cohorts(ts, kix, y1, a), nyr=floor(Int, r.dt), frac=r.dt - floor(r.dt), W=wp[i][1], Wn=wp[i][2], seedplot=sp_plot, planted=false))
      for k in 1:nsp; push!(K, k); push!(Y, get(igcnt, (i, sb, keys_[k]), 0)); push!(FM, fm2); push!(SI, length(sites)); end
    end
  end
  lg(@sprintf("%s: %d intervals, %d subplot sites, %d saplings | microplot status unknown: %d subplots | saplings on nonforest microplots (dropped): %d | nonforest & empty at v1 (dropped): %d",
              set, length(ids), length(sites), Int(sum(Y)), nunk, nnf_y, nnf_v1))
  nunk <= 0.01 * (length(sites) + nunk) || error("forest status unknown for $(nunk) subplots (> 1 %): check the forest table against the interval dates")
  t0 = time(); ex = probe_exposure(sites, M.P, M.G); lg(@sprintf("  probe exposure %.1f s", time() - t0))
  (K=K, Y=Y, fm=FM, site=SI, ex=ex, μ=zeros(length(Y)))          # μ: work buffer of recr_nll
end
XT = recr_data("train"); XV = recr_data("val")
ny_mean = mean(sum(e.wy) for e in XT.ex)
base = log(sum(XT.Y) / (sum(XT.fm) / nsp * ny_mean) / nsp)
rr = cmaes(z -> recr_nll(z, XT, M.ST), vcat(fill(base, nsp), log(4.0), -0.3, log(0.3)), 0.3; λ=16, maxevals=4000, logf=x -> nothing)
z = rr.x; λ0, λ1, ε0 = exp(z[nsp+1]), z[nsp+2], exp(z[nsp+3])
CSV.write(joinpath(FD, "recruitment.csv"), DataFrame(name=vcat(["lnF:" * k for k in keys_], ["λ0", "λ1", "ε0"]), value=vcat(z[1:nsp] .+ log(4.0), [λ0, λ1, ε0])))
lg(@sprintf("FIT (TRAIN %d rows, evals %d): λ0 %.3f λ1 %.3f ε0 %.4f | F per key (per microplot-sapling units) %s", length(XT.Y), rr.evals, λ0, λ1, ε0,
            join(keys_ .* "=" .* string.(round.(exp.(z[1:nsp]); sigdigits=3)), " ")))
# ---- held-out check: Poisson deviance of the fit vs a per-key constant rate (train rate per unit microplot area-year)
dev(y, μ) = 2sum((y[j] > 0 ? y[j] * log(y[j] / μ[j]) : 0.0) - (y[j] - μ[j]) for j in eachindex(y))
μV = recr_mu!(zeros(length(XV.Y)), z, XV, M.ST)
rk = [sum(XT.Y[XT.K .== k]) / sum(XT.fm[XT.K .== k] .* [sum(XT.ex[s].wy) for s in XT.site[XT.K .== k]]) for k in 1:nsp]
μ0 = [rk[XV.K[j]] * XV.fm[j] * sum(XV.ex[XV.site[j]].wy) for j in eachindex(XV.Y)]
lg(@sprintf("VAL deviance: fit %.1f | constant per-key rate %.1f | obs %d expected %.1f (ratio %.3f)", dev(XV.Y, μV), dev(XV.Y, μ0), Int(sum(XV.Y)), sum(μV), sum(XV.Y) / sum(μV)))
# ---- calibration tables (VAL): observed / expected by key, stand RD class at v1 and seed presence
rd1 = [XV.ex[XV.site[j]].RD[1] for j in eachindex(XV.Y)]; sd1 = [XV.ex[XV.site[j]].seed[XV.K[j], 1] for j in eachindex(XV.Y)]
cal = DataFrame(key=keys_[XV.K], rd=cut_rd.(rd1), seed=sd1, y=XV.Y, mu=μV)
lg("VAL observed / expected by key:  " * join([@sprintf("%s %d/%.1f", r.key, r.y, r.mu) for r in eachrow(combine(groupby(cal, :key), :y => sum => :y, :mu => sum => :mu))], " | "))
lg("VAL observed / expected by RD after year 1:   " * join([@sprintf("%s %d/%.1f", r.rd, r.y, r.mu) for r in eachrow(sort(combine(groupby(cal, :rd), :y => sum => :y, :mu => sum => :mu), :rd))], " | "))
lg("VAL observed / expected by seed: " * join([@sprintf("%s %d/%.1f", r.seed, r.y, r.mu) for r in eachrow(combine(groupby(cal, :seed), :y => sum => :y, :mu => sum => :mu))], " | "))
s_mean = sum(XT.fm[j] * sum(XT.ex[XT.site[j]].wy .* XT.ex[XT.site[j]].surv[XT.K[j], :]) for j in eachindex(XT.Y)) / sum(XT.fm[j] * sum(XT.ex[XT.site[j]].wy) for j in eachindex(XT.Y))
lg(@sprintf("mean probe survival to v2 (TRAIN, entry-year weighted): %.3f  (the factor the old fit ignored)", s_mean))
