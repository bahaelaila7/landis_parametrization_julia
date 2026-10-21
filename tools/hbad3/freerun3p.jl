# HBAD3P pipeline stage 8 — free runs (the harvest-relevant test): every clean natural plot of a region with ≥ 2 remeasurement intervals (R4 split `set`,
# default val) is projected from its FIRST visit to its LAST (annual steps + the final partial year, process recruitment) and compared at the last visit with
# the observed stand: plot BA (all trees ≥ 2.54 cm, m²/ha), stems ≥ 12.7 cm (TPH), QMD of stems ≥ 12.7 cm, Lorey height (BA-weighted H of stems ≥ 12.7 cm),
# against PERSISTENCE (the first-visit values). Cohort state → plot values through a lognormal within-cohort spread (σ 0.16, median = QMD e^{−σ²}) at 12.7 cm.
# Growth / mortality: fit3p_<tag>; recruitment: recruitment.csv of the same directory (stage 7, tools/hbad3/recruit3p.jl). Subplot sites in the fit's density
# convention (lib3p.jl): a subplot NOT forested at the first visit (f_subp = 0) and without trees neither holds nor recruits trees.
#   (pipeline env) ./julia_gdal.sh --project=. --threads=N tools/hbad3/freerun3p.jl <fit3p tag> [set]
include(joinpath(@__DIR__, "common3.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plugins", "HBAD3", "HBAD3P.jl")); import .HBAD3P
include(joinpath(@__DIR__, "lib3p.jl"))
tag = ARGS[1]; SET = length(ARGS) >= 2 ? ARGS[2] : "val"; FD = joinpath(R3DIR, "fit3p_" * tag); check_meta(FD); FAREA = meta_farea(FD)
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_); kix = Dict(k => i for (i, k) in enumerate(keys_))
M = load_fit3p(FD, keys_); P, Rg = load_recruit3p(FD, M.P); G = M.G
T = interval_tables(keys_); wd = water_pairs(); IF = interval_forest(T.iv, forest_table())
println(@sprintf("fit %s (density convention farea=%d) | recruitment λ0 %.3f λ1 %.3f ε0 %.4f", tag, FAREA, Rg.λ0, Rg.λ1, Rg.ε0))
HBAD3P.set_quadrature!(VQ_Z, VQ_W)
ivd = Dict(r.iid => r for r in eachrow(T.iv)); regiids = Set(tr.iid)
natural(i) = !ivd[i].planted
# ---------------- free-run plots: all intervals of the plot clean, natural, in the region, set == SET, ≥ 2 intervals
allp = combine(groupby(T.iv, [:st, :u, :c, :p]), :iid => (x -> all(i -> i in regiids && natural(i), x)) => :ok, nrow => :n, :set => first => :set)
plots = allp[allp.ok .& (allp.n .>= 2) .& (coalesce.(allp.set, "") .== SET), :]
println("free-run plots ($(SET)): ", nrow(plots))
ivbp = Dict((k.st, k.u, k.c, k.p) => sort(DataFrame(g), :k) for (k, g) in pairs(groupby(T.iv, [:st, :u, :c, :p])))
B = build_free(plots, ivbp, T, IF, wd, keys_, kix, FAREA)
println("subplot sites $(length(B.live)): nonforest at the first visit without trees (no recruitment) $(B.nnf)")
F = free_store(B, nsp)
HBAD3P.run_free!(F, P, G, Rg, B.yrs, (s, y) -> nothing; frac=B.fracs)
println("dropped recruit classes: ", sum(F.dropped))
rows = plot_values(F, B)
CSV.write(joinpath(FD, "freerun_$(SET).csv"), rows)
S = F.S; res = B.res; pl_site = B.pl_site; off = B.off; outm = B.outm; grp0 = B.grp0
# decomposition of stems >= 12.7 at the last visit (sums over plots, per ha): simulated (lognormal operator) vs observed (tree ids)
sim3 = zeros(3); obs3 = zeros(3)
for (pi_, r) in enumerate(res)
  obs3 .+= r.ob
  for s in findall(==(pi_), pl_site), i in off[s]:off[s]+F.live[s]-1
    sim3[grp0[i]] += above(S.D[i]) * S.N[i] * outm[s] / 4
  end
end
np_ = length(res)
# FIA cannot separate "crossed" from "new": only microplot saplings are tracked below 12.7 cm, so a subplot tree reaching 12.7 cm outside the microplot is
# first tallied as new. Compare in-growth = crossed + new (observed) with crossed (initial cohorts < 12.7) + recruits (simulated).
println(@sprintf("stems >= 12.7 at the last visit, mean per ha over %d plots — from stems >= 12.7 at the first visit: obs %.1f sim %.1f | IN-GROWTH (obs crossed+new %.1f+%.1f; sim initial small+recruits %.1f+%.1f): obs %.1f sim %.1f (ratio %.3f)",
                 np_, obs3[1] / np_, sim3[1] / np_, obs3[2] / np_, obs3[3] / np_, sim3[2] / np_, sim3[3] / np_, (obs3[2] + obs3[3]) / np_, (sim3[2] + sim3[3]) / np_, (sim3[2] + sim3[3]) / (obs3[2] + obs3[3])))
println(@sprintf("%-4s %6s | %9s %9s | %22s | %22s", "var", "plots", "obs1 mean", "obsL mean", "SIM: bias  med|rel err|", "PERSIST: bias med|re|"))
for v in ("BA", "N12", "QMD", "LH")
  rv = rows[rows.var .== v, :]; nsimnan = count(i -> isfinite(rv.obsL[i]) && rv.obsL[i] > 0 && !isfinite(rv.sim[i]), 1:nrow(rv))
  nsimnan > 0 && println(@sprintf("  %s: %d plots with observed values but no simulated stems ≥ 12.7 cm are EXCLUDED from the row below", v, nsimnan))
  d = rows[(rows.var .== v) .& isfinite.(rows.obsL) .& isfinite.(rows.sim) .& isfinite.(rows.obs1) .& (rows.obsL .> 0), :]
  re_s = (d.sim .- d.obsL) ./ d.obsL; re_p = (d.obs1 .- d.obsL) ./ d.obsL
  println(@sprintf("%-4s %6d | %9.2f %9.2f | %+8.1f %%  %8.1f %%   | %+8.1f %%  %8.1f %%", v, nrow(d), mean(d.obs1), mean(d.obsL),
                   100 * (mean(d.sim) - mean(d.obsL)) / mean(d.obsL), 100 * median(abs.(re_s)), 100 * (mean(d.obs1) - mean(d.obsL)) / mean(d.obsL), 100 * median(abs.(re_p))))
end
