# HBAD v2 concept checks on tracked trees of undisturbed intervals (fit TRAIN, score VAL):
#  E1  does a HEIGHT-based overtopping index (stand density of taller trees) explain diameter growth, height growth and
#      mortality better than the diameter-based BAL of HBAD v1?
#  E2  do height and diameter growth respond differently to crowding (elasticities)?
#   ./julia_gdal.sh --project=. --threads=4 tools/hbad2/empirical_checks.jl
using Arrow, DataFrames, Statistics, LinearAlgebra, Printf, Random
const R = joinpath(@__DIR__, "..", "..", "results", "hbad2")
iv = DataFrame(Arrow.Table(joinpath(R, "intervals.arrow"))); tr = DataFrame(Arrow.Table(joinpath(R, "interval_trees.arrow")))
ivd = Dict(r.iid => r for r in eachrow(iv))
# --- per-tree competition indices at v1 (all live trees of the plot at v1 are the stand)
sdi(d, w) = w * (d / 25.4)^1.605
tr.ba = (pi / 4) .* (tr.dia_cm ./ 100) .^ 2 .* tr.w_ha
n = nrow(tr); BAL = zeros(n); SDIH = zeros(n); RHt = zeros(n); SDIT = zeros(n); BAT = zeros(n)
gi = groupby(tr, :iid)
Threads.@threads for g in collect(gi)
  rows = parentindices(g)[1]; d = g.dia_cm; h = g.ht_m; w = g.w_ha; b = g.ba
  S = sum(sdi(d[j], w[j]) for j in eachindex(d)); B = sum(b)
  o = sortperm(h; rev=true); acc = 0.0; top = 0.0; hs = 0.0   # dominant height: mean H of the tallest 250 trees/ha
  for j in o; take = min(w[j], 250 - acc); take <= 0 && break; hs += take * h[j]; acc += take; end
  hdom = hs / acc
  for (a, i) in enumerate(rows)
    BAL[i] = sum(b[j] for j in eachindex(d) if d[j] > d[a]; init=0.0) + 0.5 * (sum(b[j] for j in eachindex(d) if d[j] == d[a]) - b[a])
    SDIH[i] = sum(sdi(d[j], w[j]) for j in eachindex(d) if h[j] > h[a]; init=0.0) + 0.5 * (sum(sdi(d[j], w[j]) for j in eachindex(d) if h[j] == h[a]) - sdi(d[a], w[a]))
    RHt[i] = h[a] / hdom; SDIT[i] = S; BAT[i] = B
  end
end
tr.BAL = BAL; tr.SDIH = SDIH; tr.RH = RHt; tr.SDI = SDIT; tr.BAS = BAT
tr.dt = [ivd[i].dt for i in tr.iid]; tr.set = [coalesce(ivd[i].set, "") for i in tr.iid]
tr.undist = [!(ivd[i].trt || ivd[i].dst || ivd[i].harv) for i in tr.iid]
d = filter(r -> r.undist && r.out in ("alive", "dead") && r.age > 0 && r.dt > 2, tr)
# species factor: own level for species with >= 3000 train trees, else SFTWD/HRDWD
cnt = combine(groupby(filter(r -> r.set == "train", d), :sym), nrow => :n); big = Set(cnt.sym[cnt.n .>= 3000])
d.key = [s in big ? s : "_" * g for (s, g) in zip(d.sym, d.grp)]; keys_ = sort(unique(d.key)); kix = Dict(k => i for (i, k) in enumerate(keys_))
@printf("trees in undisturbed intervals: %d (train %d, val %d); species levels %d\n", nrow(d), count(==("train"), d.set), count(==("val"), d.set), length(keys_))
x(D) = [log.(D.dia_cm ./ 20) D.dia_cm ./ 20 log.(D.age ./ 30)]
function design(D, idx::Symbol...)
  F = zeros(nrow(D), length(keys_)); for (i, k) in enumerate(D.key); F[i, kix[k]] = 1; end
  cols = Dict(:BAL => D.BAL ./ 10, :SDIH => D.SDIH ./ 500, :RD => D.SDI ./ 500, :RH => D.RH, :lnRH => log.(clamp.(D.RH, 0.05, 2)))
  hcat(F, x(D), (cols[s] for s in idx)...)
end
ols(X, y) = (X' * X + 1e-8I) \ (X' * y)
r2(y, yh) = 1 - sum(abs2, y .- yh) / sum(abs2, y .- mean(y))
function logit_fit(X, y, off; it=30)        # complementary log-log on interval mortality with log(dt) offset: P(dead)=1-exp(-exp(Xb+off))
  b = zeros(size(X, 2)); b[1:length(keys_)] .= -4.0
  for _ in 1:it
    eta = X * b .+ off; mu = clamp.(1 .- exp.(-exp.(eta)), 1e-9, 1 - 1e-9)
    dmu = exp.(eta .- exp.(eta)); wts = dmu .^ 2 ./ (mu .* (1 .- mu)); z = (eta .- off) .+ (y .- mu) ./ dmu
    b = (X' * (wts .* X) + 1e-6I) \ (X' * (wts .* z))
  end
  b
end
nll(X, b, y, off) = (mu = clamp.(1 .- exp.(-exp.(X * b .+ off)), 1e-9, 1 - 1e-9); -mean(y .* log.(mu) .+ (1 .- y) .* log.(1 .- mu)))
function auc(s, y)
  o = sortperm(s); r = invperm(o); n1 = sum(y); n0 = length(y) - n1
  (sum(r[y .== 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
end
TR = filter(r -> r.set == "train", d); VA = filter(r -> r.set == "val", d)
A(D) = filter(r -> r.out == "alive" && isfinite(r.dia2_cm), D)
TRa, VAa = A(TR), A(VA)
yD(D) = log.(max.(D.dia2_cm .- D.dia_cm, 0) ./ D.dt .+ 0.02)      # ln annual diameter increment (cm/yr)
yH(D) = (D.ht2_m .- D.ht_m) ./ D.dt                                   # annual height increment (m/yr), noisy (16 % negative)
yM(D) = Float64.(D.out .== "dead")
specs = [(), (:BAL,), (:SDIH,), (:RD,), (:RH,), (:BAL, :RD), (:SDIH, :RD), (:SDIH, :RH), (:BAL, :SDIH, :RD), (:SDIH, :RD, :RH)]
println("\nE1: held-out (VAL) skill per competition index set (species + size/age terms always in)")
@printf("%-18s %10s %10s %10s %10s\n", "index set", "R2 lndD", "R2 dH", "mort NLL", "mort AUC")
for s in specs
  bD = ols(design(TRa, s...), yD(TRa)); bH = ols(design(TRa, s...), yH(TRa))
  bM = logit_fit(design(TR, s...), yM(TR), log.(TR.dt))
  @printf("%-18s %10.4f %10.4f %10.5f %10.4f\n", isempty(s) ? "none" : join(s, "+"), r2(yD(VAa), design(VAa, s...) * bD), r2(yH(VAa), design(VAa, s...) * bH),
          nll(design(VA, s...), bM, yM(VA), log.(VA.dt)), auc(design(VA, s...) * bM, yM(VA)))
end
println("\nE2: elasticity of diameter vs height growth to crowding (train; standardised coefficient = effect of +1 SD of the index)")
for s in ((:SDIH, :RD), (:BAL, :RD))
  XD = design(TRa, s...); bD = ols(XD, yD(TRa)); bH = ols(XD, yH(TRa))
  mD = mean(max.(TRa.dia2_cm .- TRa.dia_cm, 0) ./ TRa.dt); mH = mean(yH(TRa))
  for (j, nm) in enumerate(s)
    col = size(XD, 2) - length(s) + j; sdx = std(XD[:, col])
    @printf("  %-5s (with %s): ln dD %+.3f per SD (=%+.1f %%)   dH %+.3f m/yr per SD (=%+.1f %% of mean %.3f)\n", nm, join(s, "+"), bD[col] * sdx,
            100 * (exp(bD[col] * sdx) - 1), bH[col] * sdx, 100 * bH[col] * sdx / mH, mH)
  end
end
Arrow.write(joinpath(R, "trees_indices.arrow"), select(d, :iid, :sb, :tr, :key, :sym, :grp, :age, :dia_cm, :ht_m, :w_ha, :cclcd, :cr, :out, :dia2_cm, :ht2_m, :dt, :set, :BAL, :SDIH, :RH, :SDI, :BAS); compress=:zstd)
