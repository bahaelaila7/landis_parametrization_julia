# E5: is individual growth vigour persistent? For trees remeasured over two consecutive undisturbed intervals, correlate the
# residuals of ln annual dD (species + ln D + D + subplot overtopping + stand SDI, fitted on interval 1) between interval 1 and 2.
# Persistent vigour (corr ≫ 0) means a cohort is a mix of fast and slow trees → justifies vigour sub-cohorts in the kernel.
include(joinpath(@__DIR__, "common.jl"))
iv = DataFrame(Arrow.Table(joinpath(R2DIR, "intervals.arrow"))); raw = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_trees.arrow")))
good = Set(iv.iid[.!(iv.trt .| iv.dst .| iv.harv) .& (iv.dt .> 2)])
ivk = Dict((r.st, r.u, r.c, r.p, r.k) => r.iid for r in eachrow(iv)); ivr = Dict(r.iid => r for r in eachrow(iv))
r = raw[in.(raw.iid, Ref(good)) .& (raw.out .== "alive"), :]
r.dt = [ivr[i].dt for i in r.iid]; r.y = log.(max.(r.dia2_cm .- r.dia_cm, 0) ./ r.dt .+ 0.02)
# subplot competition indices at v1 (all live trees of the subplot, raw table incl. trees that die later)
allv = raw[in.(raw.iid, Ref(good)), :]
oh = Dict{Tuple{Int,Int,Int},Tuple{Float64,Float64}}()
for g in groupby(allv, [:iid, :sb])
  for a in eachrow(g)
    o = 4 * sum(g.w_ha[j] * (g.dia_cm[j] / 25.4)^1.605 for j in 1:nrow(g) if g.ht_m[j] > a.ht_m; init=0.0)
    s = 4 * sum(g.w_ha .* (g.dia_cm ./ 25.4) .^ 1.605)
    oh[(a.iid, a.sb, a.tr)] = (o, s)
  end
end
r.O = [oh[(a.iid, a.sb, a.tr)][1] for a in eachrow(r)] ./ 500; r.SDI = [oh[(a.iid, a.sb, a.tr)][2] for a in eachrow(r)] ./ 500
keys_ = sort(unique(r.sym)); kix = Dict(k => i for (i, k) in enumerate(keys_))
X(D) = (F = zeros(nrow(D), length(keys_)); for (i, k) in enumerate(D.sym); F[i, kix[k]] = 1; end; hcat(F, log.(D.dia_cm ./ 20), D.dia_cm ./ 20, D.O, D.SDI))
b = (X(r)' * X(r) + 1e-6I) \ (X(r)' * r.y); r.res = r.y .- X(r) * b
# pair interval k and k+1 of the same tree
r.pk = [(ivr[i].st, ivr[i].u, ivr[i].c, ivr[i].p, ivr[i].k) for i in r.iid]
res = Dict((p[1:4]..., p[5], sb, tr) => x for (p, sb, tr, x) in zip(r.pk, r.sb, r.tr, r.res))
pairs = [(x, res[(p[1:4]..., p[5] + 1, sb, tr)]) for (p, sb, tr, x) in zip(r.pk, r.sb, r.tr, r.res) if haskey(res, (p[1:4]..., p[5] + 1, sb, tr))]
a1 = first.(pairs); a2 = last.(pairs)
@printf("E5 trees with two consecutive undisturbed growth intervals: %d; corr(residual ln dD interval k, k+1) = %.3f\n", length(pairs), cor(a1, a2))
small = [(x, y) for ((x, y), p) in zip(pairs, 1:length(pairs))]
@printf("residual sd %.3f; implied persistent share of residual variance ≈ corr = %.2f\n", std(r.res), cor(a1, a2))
