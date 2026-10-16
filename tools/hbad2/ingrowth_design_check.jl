# FIA design check: is ingrowth into the ≥ 5 in (12.7 cm) tally consistent with the tracked microplot saplings that cross 5 in?
# In expectation (per ha, undisturbed intervals): Σ w(macro ingrowth, first tallied ≥ 12.7 cm)  vs  Σ 185.2 · #(tracked saplings, D1 < 12.7, alive with D2 ≥ 12.7).
include(joinpath(@__DIR__, "common.jl"))
iv = DataFrame(Arrow.Table(joinpath(R2DIR, "intervals.arrow"))); raw = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_trees.arrow")))
ig = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_ingrowth.arrow")))
ok = Set(iv.iid[.!(iv.trt .| iv.dst .| iv.harv) .& (iv.dt .> 2)])
r = raw[in.(raw.iid, Ref(ok)), :]; g = ig[in.(ig.iid, Ref(ok)), :]
cross = r[(r.dia_cm .< 12.7) .& (r.out .== "alive") .& (coalesce.(r.dia2_cm, 0.0) .>= 12.7), :]
macro_ing = g[g.dia_cm .>= 12.7, :]
n_iv = length(ok)
@printf("undisturbed intervals %d\n", n_iv)
@printf("tracked microplot saplings crossing 12.7 cm: %d trees → %.1f trees/ha per interval (×185.2 / %d)\n", nrow(cross), nrow(cross) * 185.2 / n_iv, n_iv)
@printf("first-tallied macro trees (≥ 12.7 cm): %d → %.1f trees/ha per interval (w %.2f)\n", nrow(macro_ing), sum(macro_ing.w_ha) / n_iv, mean(macro_ing.w_ha))
@printf("ratio macro ingrowth / sapling crossings = %.2f\n", sum(macro_ing.w_ha) / (nrow(cross) * 185.2))
# how big are the first-tallied macro trees? (crossing trees should be just above 12.7)
@printf("macro ingrowth D quantiles (cm) 10/50/90 %%: %s ; tracked crossers D2: %s\n", round.(quantile(macro_ing.dia_cm, [0.1, 0.5, 0.9]); digits=1), round.(quantile(cross.dia2_cm, [0.1, 0.5, 0.9]); digits=1))
@printf("macro ingrowth ≥ 20 cm (cannot be recent 5-in crossers): %.1f %% of macro ingrowth trees\n", 100 * mean(macro_ing.dia_cm .>= 20))
