# HBAD3P free runs (the harvest-relevant test): every clean natural plot of a region with ≥ 2 remeasurement intervals (R4 split `set`, default val) is
# projected from its FIRST visit to its LAST (annual steps, process recruitment) and compared at the last visit with the observed stand:
#   plot BA (all trees ≥ 2.54 cm, m²/ha), stems ≥ 12.7 cm (TPH), QMD of stems ≥ 12.7 cm, Lorey height (BA-weighted H of stems ≥ 12.7 cm)
# against PERSISTENCE (the first-visit values). Cohort state → plot values through a lognormal within-cohort spread (σ 0.16) at the 12.7 cm threshold.
# Growth / mortality parameters: a fit3p_<tag> fit of the same region; recruitment (ln F per key, λ0, λ1, ε0) fitted here by Poisson on TRAIN sapling
# ingrowth (process form of docs §7.6 R3: F_k × e^{−λ0 e^{λ1 (ST−3)} RD} × (seed present ? 1 : ε0)).
#   (env as fit3p) ./julia_gdal.sh --project=. --threads=4 tools/hbad3/freerun3p.jl <fit3p tag> [set]
include(joinpath(@__DIR__, "common3.jl")); include(joinpath(@__DIR__, "..", "hbad2", "cmaes.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plugins", "HBAD3", "HBAD3P.jl")); import .HBAD3P
tag = ARGS[1]; SET = length(ARGS) >= 2 ? ARGS[2] : "val"; FD = joinpath(R3DIR, "fit3p_" * tag)
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_); kix = Dict(k => i for (i, k) in enumerate(keys_))
spc = CSV.read(joinpath(FD, "species.csv"), DataFrame); String.(spc.key) == keys_ || error("keys of fit $(tag) differ from this region's keys")
gl = CSV.read(joinpath(FD, "globals.csv"), DataFrame); gd = Dict(Symbol(r.name) => Float64(r.value) for r in eachrow(gl))
G = HBAD3P.GlobPP((gd[f] for f in fieldnames(HBAD3P.GlobPP))...)
big = Set(k for k in keys_ if k[1] != '_'); keyof(s, g) = s in big ? s : "_" * g
iv = DataFrame(Arrow.Table(joinpath(DATA_DIR, "intervals.arrow"))); raw = DataFrame(Arrow.Table(joinpath(DATA_DIR, "interval_trees.arrow")))
ig = DataFrame(Arrow.Table(joinpath(DATA_DIR, "interval_ingrowth.arrow")))
regiids = Set(tr.iid)                                  # load_trees applied the region filter (+ clean intervals)
ivr = iv[[i in regiids for i in iv.iid], :]
w = CSV.read(get(ENV, "HBAD3_WATER_CSV", joinpath(R3DIR, "interval_water.csv")), DataFrame); wd = Dict(Int(r.iid) => (r.aetpet_iv, r.aetpet_n) for r in eachrow(w))
# ---------------- recruitment fit (TRAIN): counts of first-tallied saplings (D < 12.7) per subplot-interval and key
raw.key = [keyof(s, g) for (s, g) in zip(raw.sym, raw.grp)]; ig.key = [keyof(s, g) for (s, g) in zip(ig.sym, ig.grp)]
ivd = Dict(r.iid => r for r in eachrow(iv))
natural(i) = !ivd[i].planted
function recr_rows(set)
  ids = Set(i for i in unique(tr.iid) if ivd[i].set == set && natural(i))
  t = raw[[i in ids for i in raw.iid], :]; t.site = t.iid .* 10 .+ t.sb
  t.sdi = 4 .* t.w_ha .* (t.dia_cm ./ 25.4) .^ 1.605
  sd = combine(groupby(t, :site), :sdi => sum => :SDI, :iid => first => :iid)
  seed_sub = Set((r.site, r.key) for r in eachrow(t) if r.dia_cm >= 12.7)
  seed_plot = Set((r.iid, r.key) for r in eachrow(t) if r.dia_cm >= 12.7)
  g = ig[[i in ids for i in ig.iid] .& (ig.dia_cm .< 12.7), :]; g.site = g.iid .* 10 .+ g.sb
  cnt = Dict((r.site, r.key) => r.n for r in eachrow(combine(groupby(g, [:site, :key]), nrow => :n)))
  sites = union(Set(sd.site), Set(g.site)); sdi_of = Dict(r.site => r.SDI for r in eachrow(sd))
  K = Int[]; RD = Float64[]; SEED = Bool[]; DT = Float64[]; Y = Float64[]
  for s in sort(collect(sites)), (k, kk) in kix
    i = s ÷ 10
    push!(K, kk); push!(RD, get(sdi_of, s, 0.0) / 1500); push!(SEED, (s, k) in seed_sub || (i, k) in seed_plot); push!(DT, ivd[i].dt); push!(Y, get(cnt, (s, k), 0))
  end
  (K=K, RD=RD, SEED=SEED, DT=DT, Y=Y)
end
RT = recr_rows("train"); STk = Float64.(spc.ST)
function recr_nll(z, X)
  lF = z[1:nsp]; λ0 = exp(z[nsp+1]); λ1 = z[nsp+2]; e0 = exp(z[nsp+3]); ll = 0.0
  @inbounds for j in eachindex(X.Y)
    k = X.K[j]; μ = X.DT[j] * exp(lF[k] - λ0 * exp(λ1 * (STk[k] - 3)) * X.RD[j]) * (X.SEED[j] ? 1.0 : e0)
    ll += X.Y[j] * log(max(μ, 1e-300)) - μ
  end
  -ll
end
base = log(sum(RT.Y) / sum(RT.DT) / nsp)
rr = cmaes(z -> recr_nll(z, RT), vcat(fill(base, nsp), log(4.0), -0.3, log(0.3)), 0.3; λ=16, maxevals=3000, logf=x -> nothing)
lF = rr.x[1:nsp] .+ log(4.0)           # per subplot-site density (×4: a microplot sapling of a subplot site = 4 × 185.2 trees/ha)
Rg = HBAD3P.RecGP(exp(rr.x[nsp+1]), rr.x[nsp+2], exp(rr.x[nsp+3]))
println(@sprintf("recruitment fit (TRAIN %d rows): λ0 %.2f λ1 %.2f ε0 %.3f | F per key %s", length(RT.Y), Rg.λ0, Rg.λ1, Rg.ε0, join(round.(exp.(rr.x[1:nsp]); sigdigits=3), ",")))
P = [HBAD3P.SpPP(log(spc.eps[k]), spc.Hmax[k], spc.ST[k], spc.longevity[k], lF[k]) for k in 1:nsp]
# ---------------- free-run plots: all intervals of the plot clean, natural, in the region, set == SET, ≥ 2 intervals
ivr.pk = collect(zip(ivr.st, ivr.u, ivr.c, ivr.p))
allp = combine(groupby(iv, [:st, :u, :c, :p]), :iid => (x -> all(i -> i in regiids && natural(i), x)) => :ok, nrow => :n, :set => first => :set)
plots = allp[allp.ok .& (allp.n .>= 2) .& (coalesce.(allp.set, "") .== SET), :]
println("free-run plots ($(SET)): ", nrow(plots))
ivbp = Dict((k.st, k.u, k.c, k.p) => sort(DataFrame(g), :k) for (k, g) in pairs(groupby(iv, [:st, :u, :c, :p])))
σw = 0.16; above(D) = 1 - cdf(Normal(), (log(12.7) - log(D)) / σw)
function standvals(D, H, N)            # per ha; N already per ha of the plot
  ba = 0.0; ba12 = 0.0; n12 = 0.0; s2 = 0.0; lh = 0.0
  for i in eachindex(D)
    D[i] < 2.54 && continue
    b = N[i] * pi / 4 * (D[i] / 100)^2; ba += b
    f = above(D[i]); n12 += f * N[i]; s2 += f * N[i] * D[i]^2; ba12 += f * b; lh += f * b * H[i]
  end
  (BA=ba, N12=n12, QMD=n12 > 0 ? sqrt(s2 / n12) : NaN, LH=ba12 > 0 ? lh / ba12 : NaN)
end
function obsvals(D, H, Wt)             # tree records, Wt = trees/ha of the plot
  ba = 0.0; ba12 = 0.0; n12 = 0.0; s2 = 0.0; lh = 0.0
  for i in eachindex(D)
    (D[i] >= 2.54 && Wt[i] > 0) || continue
    b = Wt[i] * pi / 4 * (D[i] / 100)^2; ba += b
    if D[i] >= 12.7; n12 += Wt[i]; s2 += Wt[i] * D[i]^2; ba12 += b; lh += b * H[i]; end
  end
  (BA=ba, N12=n12, QMD=n12 > 0 ? sqrt(s2 / n12) : NaN, LH=ba12 > 0 ? lh / ba12 : NaN)
end
rawby = Dict(k.iid => DataFrame(g) for (k, g) in pairs(groupby(raw, :iid))); igby = Dict(k.iid => DataFrame(g) for (k, g) in pairs(groupby(ig, :iid)))
off = [1]; spv = Int32[]; N0 = Float64[]; D0 = Float64[]; H0 = Float64[]; A0 = Float64[]; zq = Float64[]; live = Int[]; Ws = Float64[]; Wns = Float64[]
yrs = Int[]; seedm = falses(nsp, 4nrow(plots)); res = NamedTuple[]; pl_site = Int[]
for (pi_, r) in enumerate(eachrow(plots))
  ivs = ivbp[(r.st, r.u, r.c, r.p)]; i1 = ivs.iid[1]; iL = ivs.iid[end]
  T = Dates.value(ivs.d2[end] - ivs.d1[1]) / 365.25
  t1 = rawby[i1]; t1 = t1[(t1.dia_cm .> 0) .& (t1.ht_m .> 0), :]
  ws = mean(first.(wd[i] for i in ivs.iid)); wn = wd[i1][2]
  tL = rawby[iL]; tL = tL[(tL.out .== "alive") .& (tL.dia2_cm .> 0), :]; gL = get(igby, iL, ig[1:0, :])
  o1 = obsvals(t1.dia_cm, t1.ht_m, t1.w_ha); oL = obsvals(vcat(tL.dia2_cm, gL.dia_cm), vcat(tL.ht2_m, gL.ht_m), vcat(tL.w2_ha, gL.w_ha))
  push!(res, (st=r.st, u=r.u, c=r.c, p=r.p, T=T, obs1=o1, obsL=oL))
  for sb in 1:4
    s = length(off); t = t1[t1.sb .== sb, :]
    t.bc = floor.(Int, -t.age ./ 10) .* 100 .+ dclass.(t.dia_cm)
    for k in 1:nsp; seedm[k, s] = any((t1.key .== keys_[k]) .& (t1.dia_cm .>= 12.7)); end
    n = 0
    if nrow(t) > 0
      gg = combine(groupby(t, [:key, :bc]), :w_ha => (x -> 4sum(x)) => :N, [:w_ha, :dia_cm] => ((w, d) -> sqrt(sum(w .* d .^ 2) / sum(w))) => :D,
                   [:w_ha, :ht_m] => ((w, h) -> sum(w .* h) / sum(w)) => :H, [:w_ha, :age] => ((w, a) -> sum(w .* a) / sum(w)) => :A)
      for g in eachrow(gg), q in 1:3
        push!(spv, kix[g.key]); push!(N0, g.N * (1 / 6, 2 / 3, 1 / 6)[q]); push!(D0, g.D); push!(H0, max(g.H, 1.37)); push!(A0, g.A); push!(zq, (-sqrt(3.0), 0.0, sqrt(3.0))[q]); n += 1
      end
    end
    cap = n + 3nsp * (ceil(Int, T / HBAD3P.RCLASS) + 1)
    for _ in 1:(cap-n); push!(spv, 1); push!(N0, 0.0); push!(D0, 2.54); push!(H0, 1.37); push!(A0, 0.0); push!(zq, 0.0); end
    push!(off, off[end] + cap); push!(live, n); push!(Ws, ws); push!(Wns, wn); push!(yrs, round(Int, T)); push!(pl_site, pi_)
  end
end
ns = length(live)
S = HBAD3P.SitesP(off, spv, copy(N0), copy(D0), copy(H0), copy(A0), zeros(Int, ns), zeros(ns), fill(false, ns), copy(zq), Ws, Wns)
F = HBAD3P.FreeSitesP(S, copy(live), zeros(Int, nsp, ns), zeros(Int, nsp, ns), zeros(Int, ns), seedm[:, 1:ns])
HBAD3P.run_free!(F, P, G, Rg, yrs, (s, y) -> nothing)
println("dropped recruit classes: ", sum(F.dropped))
# plot values (subplot sites are per ha already ×4 → plot = mean of the 4 sites)
rows = DataFrame(T=Float64[], var=String[], obs1=Float64[], obsL=Float64[], sim=Float64[])
for (pi_, r) in enumerate(res)
  ss = findall(==(pi_), pl_site); Dv = Float64[]; Hv = Float64[]; Nv = Float64[]
  for s in ss, i in off[s]:off[s]+F.live[s]-1; push!(Dv, S.D[i]); push!(Hv, S.H[i]); push!(Nv, S.N[i] / 4); end
  sv = standvals(Dv, Hv, Nv)
  for v in (:BA, :N12, :QMD, :LH); push!(rows, (r.T, String(v), getfield(r.obs1, v), getfield(r.obsL, v), getfield(sv, v))); end
end
CSV.write(joinpath(FD, "freerun_$(SET).csv"), rows)
println(@sprintf("%-4s %6s | %9s %9s | %22s | %22s", "var", "plots", "obs1 mean", "obsL mean", "SIM: bias  med|rel err|", "PERSIST: bias med|re|"))
for v in ("BA", "N12", "QMD", "LH")
  d = rows[(rows.var .== v) .& isfinite.(rows.obsL) .& isfinite.(rows.sim) .& isfinite.(rows.obs1) .& (rows.obsL .> 0), :]
  re_s = (d.sim .- d.obsL) ./ d.obsL; re_p = (d.obs1 .- d.obsL) ./ d.obsL
  println(@sprintf("%-4s %6d | %9.2f %9.2f | %+8.1f %%  %8.1f %%   | %+8.1f %%  %8.1f %%", v, nrow(d), mean(d.obs1), mean(d.obsL),
                   100 * (mean(d.sim) - mean(d.obsL)) / mean(d.obsL), 100 * median(abs.(re_s)), 100 * (mean(d.obs1) - mean(d.obsL)) / mean(d.obsL), 100 * median(abs.(re_p))))
end
