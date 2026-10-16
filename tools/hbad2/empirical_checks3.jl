# HBAD v2 concept check E4: is competition local? Overtopping (SDI of taller trees, ranked by the species H-D curve height Ĥ) and stand SDI
# computed over the tree's own SUBPLOT (expansion ×4 → per ha) vs over the whole PLOT. Same models as E1 (fit TRAIN, score VAL).
#   ./julia_gdal.sh --project=. --threads=4 tools/hbad2/empirical_checks3.jl
using Arrow, DataFrames, CSV, Statistics, LinearAlgebra, Printf
const R = joinpath(@__DIR__, "..", "..", "results", "hbad2")
d = DataFrame(Arrow.Table(joinpath(R, "trees_indices.arrow")))
pri = CSV.read(joinpath(@__DIR__, "..", "..", "results", "hbad", "priors.csv"), DataFrame); pk = Dict(r.key => r for r in eachrow(pri))
prow(s, g) = haskey(pk, s) && !ismissing(pk[s].hd_a) ? pk[s] : pk["_" * g]
d.hhat = [(r = prow(s, g); 1.3 + r.hd_a * (1 - exp(-r.hd_b * D))^r.hd_c) for (s, g, D) in zip(d.sym, d.grp, d.dia_cm)]
sdi(x, w) = w * (x / 25.4)^1.605
for c in (:OHp, :SDIp, :OHs, :SDIs); d[!, c] = zeros(nrow(d)); end
for g in groupby(d, :iid)
  ix = parentindices(g)[1]; h = g.hhat; dd = g.dia_cm; w = g.w_ha; sb = g.sb
  for (k, i) in enumerate(ix)
    op = 0.0; os = 0.0; sp = 0.0; ss = 0.0
    for j in eachindex(h)
      x = sdi(dd[j], w[j]); sp += x
      h[j] > h[k] && (op += x)
      if sb[j] == sb[k]; ss += 4x; h[j] > h[k] && (os += 4x); end
    end
    d.OHp[i] = op; d.SDIp[i] = sp; d.OHs[i] = os; d.SDIs[i] = ss
  end
end
keys_ = sort(unique(d.key)); kix = Dict(k => i for (i, k) in enumerate(keys_))
function X(D, cols...)
  F = zeros(nrow(D), length(keys_)); for (i, k) in enumerate(D.key); F[i, kix[k]] = 1; end
  hcat(F, log.(D.dia_cm ./ 20), D.dia_cm ./ 20, log.(D.age ./ 30), (D[!, c] ./ 500 for c in cols)...)
end
ols(X, y) = (X' * X + 1e-8I) \ (X' * y); r2(y, yh) = 1 - sum(abs2, y .- yh) / sum(abs2, y .- mean(y))
function cll(X, y, off; it=30)
  b = zeros(size(X, 2)); b[1:length(keys_)] .= -4.0
  for _ in 1:it
    η = X * b .+ off; μ = clamp.(1 .- exp.(-exp.(η)), 1e-9, 1 - 1e-9); dμ = exp.(η .- exp.(η)); w = dμ .^ 2 ./ (μ .* (1 .- μ)); z = (η .- off) .+ (y .- μ) ./ dμ
    b = (X' * (w .* X) + 1e-6I) \ (X' * (w .* z))
  end
  b
end
nll(X, b, y, off) = (μ = clamp.(1 .- exp.(-exp.(X * b .+ off)), 1e-9, 1 - 1e-9); -mean(y .* log.(μ) .+ (1 .- y) .* log.(1 .- μ)))
auc(s, y) = (r = invperm(sortperm(s)); n1 = sum(y); n0 = length(y) - n1; (sum(r[y .== 1]) - n1 * (n1 + 1) / 2) / (n1 * n0))
TR = d[d.set .== "train", :]; VA = d[d.set .== "val", :]
A(D) = D[(D.out .== "alive") .& isfinite.(D.dia2_cm), :]; TRa, VAa = A(TR), A(VA)
yD(D) = log.(max.(D.dia2_cm .- D.dia_cm, 0) ./ D.dt .+ 0.02); yH(D) = (D.ht2_m .- D.ht_m) ./ D.dt; yM(D) = Float64.(D.out .== "dead")
@printf("%-22s %9s %9s %10s %9s\n", "indices", "R2 lndD", "R2 dH", "mort NLL", "mort AUC")
for c in ((:OHp, :SDIp), (:OHs, :SDIs), (:OHp, :SDIp, :OHs, :SDIs), (:OHs, :SDIp))
  bD = ols(X(TRa, c...), yD(TRa)); bH = ols(X(TRa, c...), yH(TRa)); bM = cll(X(TR, c...), yM(TR), log.(TR.dt))
  @printf("%-22s %9.4f %9.4f %10.5f %9.4f\n", join(c, "+"), r2(yD(VAa), X(VAa, c...) * bD), r2(yH(VAa), X(VAa, c...) * bH),
          nll(X(VA, c...), bM, yM(VA), log.(VA.dt)), auc(X(VA, c...) * bM, yM(VA)))
end
small = TR.dia_cm .< 12.7; smallv = VA.dia_cm .< 12.7
println("small trees (D < 12.7 cm) only:")
for c in ((:OHp, :SDIp), (:OHs, :SDIs))
  T = TR[small, :]; V = VA[smallv, :]; Ta = A(T); Va = A(V)
  bD = ols(X(Ta, c...), yD(Ta)); bM = cll(X(T, c...), yM(T), log.(T.dt))
  @printf("%-22s %9.4f %9s %10.5f %9.4f\n", join(c, "+"), r2(yD(Va), X(Va, c...) * bD), "", nll(X(V, c...), bM, yM(V), log.(V.dt)), auc(X(V, c...) * bM, yM(V)))
end
