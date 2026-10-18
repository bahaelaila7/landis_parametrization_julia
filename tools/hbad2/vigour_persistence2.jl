# E5b: persistence of growth vigour under transforms less sensitive to near-zero increments (E5 used ln(dD + 0.02)).
# For trees with two consecutive undisturbed intervals: residuals of (a) ln(dD + 0.1), (b) ln dD for trees growing > 0.05 cm/yr in both,
# (c) the growth RATIO dD / predicted mean (linear model, same covariates) — corr between intervals and the implied persistent sd.
include(joinpath(@__DIR__, "common.jl"))
iv = DataFrame(Arrow.Table(joinpath(R2DIR, "intervals.arrow"))); raw = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_trees.arrow")))
good = Set(iv.iid[.!(iv.trt .| iv.dst .| iv.harv) .& (iv.dt .> 2)]); ivr = Dict(r.iid => r for r in eachrow(iv))
r = raw[in.(raw.iid, Ref(good)) .& (raw.out .== "alive"), :]
r.dt = [ivr[i].dt for i in r.iid]; r.g = max.(r.dia2_cm .- r.dia_cm, 0) ./ r.dt
allv = raw[in.(raw.iid, Ref(good)), :]; oh = Dict{Tuple{Int,Int,Int},Float64}()
for g in groupby(allv, [:iid, :sb]), a in eachrow(g)
  oh[(a.iid, a.sb, a.tr)] = 4 * sum(g.w_ha[j] * (g.dia_cm[j] / 25.4)^1.605 for j in 1:nrow(g) if g.ht_m[j] > a.ht_m; init=0.0) / 500
end
r.O = [oh[(a.iid, a.sb, a.tr)] for a in eachrow(r)]
keys_ = sort(unique(r.sym)); kix = Dict(k => i for (i, k) in enumerate(keys_))
X(D) = (F = zeros(nrow(D), length(keys_)); for (i, k) in enumerate(D.sym); F[i, kix[k]] = 1; end; hcat(F, log.(D.dia_cm ./ 20), D.dia_cm ./ 20, D.O))
r.pk = [(ivr[i].st, ivr[i].u, ivr[i].c, ivr[i].p, ivr[i].k) for i in r.iid]
function persist(y, mask, nm)
  d = r[mask, :]; yy = y[mask]
  b = (X(d)' * X(d) + 1e-6I) \ (X(d)' * yy); res = yy .- X(d) * b
  key = Dict((p[1:4]..., p[5], sb, t) => x for (p, sb, t, x) in zip(d.pk, d.sb, d.tr, res))
  pr = [(x, key[(p[1:4]..., p[5] + 1, sb, t)]) for (p, sb, t, x) in zip(d.pk, d.sb, d.tr, res) if haskey(key, (p[1:4]..., p[5] + 1, sb, t))]
  c = cor(first.(pr), last.(pr)); sdr = std(res)
  @printf("%-38s pairs %7d  corr %.3f  residual sd %.3f  persistent sd √corr·sd = %.3f\n", nm, length(pr), c, sdr, sqrt(max(c, 0)) * sdr)
end
persist(log.(r.g .+ 0.02), trues(nrow(r)), "ln(dD+0.02)  [E5]")
persist(log.(r.g .+ 0.1), trues(nrow(r)), "ln(dD+0.1)")
persist(log.(max.(r.g, 1e-9)), r.g .> 0.05, "ln dD, dD > 0.05 (this interval)")
sm = r.dia_cm .< 12.7
persist(log.(r.g .+ 0.1), sm, "ln(dD+0.1), saplings+poles (D < 12.7)")
persist(log.(max.(r.g, 1e-9)), sm .& (r.g .> 0.05), "ln dD > 0.05, D < 12.7")
