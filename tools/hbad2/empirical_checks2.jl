# HBAD v2 concept checks, part 2 (uses results/hbad2/trees_indices.arrow from empirical_checks.jl):
#  E1b height measurement error: is the positive dH ~ overtopping effect a regression-to-the-mean artefact? rank by the species
#      H-D curve height (Ĥ) instead of the measured H and compare.
#  E3  species timing of height vs diameter growth over age, for free-growing trees (CCLCD 1-3, low overtopping):
#      mean annual dH (m), dD (cm) and the allocation ratio dH/dD by age class.
using Arrow, DataFrames, CSV, Statistics, LinearAlgebra, Printf
const R = joinpath(@__DIR__, "..", "..", "results", "hbad2")
d = DataFrame(Arrow.Table(joinpath(R, "trees_indices.arrow")))
pri = CSV.read(joinpath(@__DIR__, "..", "..", "results", "hbad", "priors.csv"), DataFrame); pk = Dict(r.key => r for r in eachrow(pri))
prow(s, g) = haskey(pk, s) && !ismissing(pk[s].hd_a) ? pk[s] : pk["_" * g]
CRh(r, D) = 1.3 + r.hd_a * (1 - exp(-r.hd_b * D))^r.hd_c
d.hhat = [CRh(prow(s, g), D) for (s, g, D) in zip(d.sym, d.grp, d.dia_cm)]
d.hres = log.(d.ht_m ./ d.hhat)
a = filter(r -> r.out == "alive" && isfinite(r.ht2_m), d); a.dH = (a.ht2_m .- a.ht_m) ./ a.dt
println("E1b regression to the mean: corr(ln H/Ĥ at v1, dH) = ", round(cor(a.hres, a.dH), digits=3),
        ";  corr(ln H/Ĥ at v1, ln H2/Ĥ(D2)) = ", round(cor(a.hres, log.(a.ht2_m ./ [CRh(prow(s, g), D) for (s, g, D) in zip(a.sym, a.grp, a.dia2_cm)])), digits=3))
# recompute the height-overtopping index with Ĥ ranking
sdi(dd, w) = w * (dd / 25.4)^1.605
d.SDIHh = zeros(nrow(d))
for g in groupby(d, :iid)
  ix = parentindices(g)[1]; h = g.hhat
  for (k, i) in enumerate(ix)
    d.SDIHh[i] = sum(sdi(g.dia_cm[j], g.w_ha[j]) for j in eachindex(h) if h[j] > h[k]; init=0.0)
  end
end
a = filter(r -> r.out == "alive" && isfinite(r.ht2_m), d); a.dH = (a.ht2_m .- a.ht_m) ./ a.dt
keys_ = sort(unique(d.key)); kix = Dict(k => i for (i, k) in enumerate(keys_))
function X(D, cols...)
  F = zeros(nrow(D), length(keys_)); for (i, k) in enumerate(D.key); F[i, kix[k]] = 1; end
  hcat(F, log.(D.dia_cm ./ 20), D.dia_cm ./ 20, log.(D.age ./ 30), (D[!, c] ./ 500 for c in cols)...)
end
ols(X, y) = (X' * X + 1e-8I) \ (X' * y); r2(y, yh) = 1 - sum(abs2, y .- yh) / sum(abs2, y .- mean(y))
TRa = filter(r -> r.set == "train", a); VAa = filter(r -> r.set == "val", a)
yD(D) = log.(max.(D.dia2_cm .- D.dia_cm, 0) ./ D.dt .+ 0.02)
println("E1b held-out R2 (VAL)      ln dD      dH")
for c in ((:SDI,), (:SDIH, :SDI), (:SDIHh, :SDI))
  @printf("  %-14s %10.4f %8.4f\n", join(c, "+"), r2(yD(VAa), X(VAa, c...) * ols(X(TRa, c...), yD(TRa))), r2(VAa.dH, X(VAa, c...) * ols(X(TRa, c...), TRa.dH)))
end
for c in ((:SDIHh, :SDI), (:SDIH, :SDI))
  XX = X(TRa, c...); bH = ols(XX, TRa.dH); bD = ols(XX, yD(TRa)); j = size(XX, 2) - 1; s = std(XX[:, j]); s2 = std(XX[:, j+1])
  @printf("  %-12s overtopping: ln dD %+.3f/SD, dH %+.3f m/yr/SD | stand SDI: ln dD %+.3f/SD, dH %+.3f m/yr/SD\n", c[1], bD[j] * s, bH[j] * s, bD[j+1] * s2, bH[j+1] * s2)
end
# E3 timing: free-growing = dominant/codominant/open (CCLCD 1-3) and in the top 30 % of overtopping-free (SDIHh low)
f = filter(r -> r.cclcd in (1.0, 2.0, 3.0) && r.set != "test", a)
f.dD = max.(f.dia2_cm .- f.dia_cm, 0) ./ f.dt
f.acl = [x < 10 ? 5 : x < 20 ? 15 : x < 30 ? 25 : x < 45 ? 37 : x < 60 ? 52 : x < 80 ? 70 : 90 for x in f.age]
t = combine(groupby(f, [:key, :acl]), nrow => :n, :dH => mean => :dH, :dD => mean => :dD, :ht_m => mean => :H, :dia_cm => mean => :D)
t.ratio = t.dH ./ t.dD; sort!(t, [:key, :acl]); filter!(r -> r.n >= 40, t)
CSV.write(joinpath(R, "timing_free_growing.csv"), t)
println("\nE3 free-growing trees (CCLCD 1-3): mean annual dH m / dD cm / dH:dD by age class")
for g in groupby(t, :key)
  @printf("  %-8s", g.key[1]); for r in eachrow(g); @printf(" a%2d:%.2f/%.2f/%.1f", r.acl, r.dH, r.dD, r.ratio); end; println()
end
