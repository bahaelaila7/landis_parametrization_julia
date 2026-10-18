# Pole (5–12.7 cm) flow budget, model vs data, per ha and year (clean natural plots of SET):
#   stock, inflow (saplings crossing 5 cm), deaths in the pole class, outflow (poles crossing 12.7 cm).
# Model: v2-family free runs (Sim B sites); every cohort-year contributes through the lognormal within-cohort operator (σ_obs):
#   class fractions F(D) = P(D_tree ≥ threshold); inflow = N1·max(0, F5(D1) − F5(D0)); outflow = N1·max(0, F12(D1) − F12(D0));
#   deaths = (N0 − N1)·[F5(D0) − F12(D0)]; stock = N·[F5 − F12]. Per ha of the plot = site sums / SITE_MULT averaged over plots.
# Data: tracked trees of the intervals of the same plots: crossing 5 cm (D1 < 5 ≤ D2, alive), pole deaths (5 ≤ D1 < 12.7, dead), crossing 12.7
#   (5 ≤ D1 < 12.7 ≤ D2, alive), stock at v1 — Σ w_ha / Σ dt (per plot-year); plus first-tallied trees that enter directly ≥ 5 cm (ingrowth).
#   HBAD2_VIGOUR=1 HBAD2_SIGV_MAX=0.7 HBAD2_SITE=subplot HBAD2_FORM=2 HBAD2_SIZE_SPLIT=5,12.7,25,40 ./julia_gdal.sh --project=. --threads=6 tools/hbad2/pole_budget.jl <fit tag> [set]
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "simB_lik.jl"))
using Distributions: Normal, cdf
tag = ARGS[1]; SET = length(ARGS) >= 2 ? ARGS[2] : "val"; FD = joinpath(R2DIR, "fit_" * tag); check_meta(FD)
σo = parse(Float64, get(ENV, "HBAD2_OBS_SIGMA", "0.11"))
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
lon = CSV.read("/workspace/species_longevity_ref.csv", DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
P, G = decode(read_theta(joinpath(FD, "theta.csv"), nsp), nsp, longev)
z = vec(CSV.read(joinpath(FD, "recr.csv"), DataFrame).theta)
R = [RecP(clamp(z[3k-2], -12, 2), clamp(z[3k-1], -1, 4), clamp(z[3k], -10, 2), 2.54, 3.0, 5.0) for k in 1:nsp]
B = build_simB(SET, keys_); _reset!(B); F = B.F; S = F.S
Fge(D, T) = D <= 0 ? 0.0 : 1 - cdf(Normal(), (log(T) - (log(D) - σo^2)) / σo)
acc = zeros(4); years_total = 0.0; rec_add = 0.0; sap_stock = 0.0; sap_dead = 0.0
for s in 1:length(F.live)
  lo = S.off[s]
  for y in 1:B.years[s]
    n = F.live[s]; hi = lo + n - 1
    N0 = S.N[lo:hi]; D0 = S.D[lo:hi]
    n > 0 && HBAD2.step_site!(S, P, G, lo, hi, 1.0, S.planted[s])
    for i in lo:hi
      k = i - lo + 1; d0 = D0[k]; d1 = S.D[i]; n1 = S.N[i]
      f5a, f12a = Fge(d0, 5.0), Fge(d0, 12.7); f5b, f12b = Fge(d1, 5.0), Fge(d1, 12.7)
      acc[1] += n1 * (f5b - f12b)                         # stock (end of year)
      acc[2] += n1 * max(0.0, f5b - f5a)                  # inflow
      acc[3] += (N0[k] - n1) * (f5a - f12a)               # deaths of poles
      acc[4] += n1 * max(0.0, f12b - f12a)                # outflow
    end
    for i in lo:hi; global sap_dead += (N0[i-lo+1] - S.N[i]) * (1 - Fge(D0[i-lo+1], 5.0)); end   # BEFORE recruitment (it merges into open classes)
    nb = sum(S.N0[i] for i in lo:lo+F.live[s]-1; init=0.0)
    HBAD2.recruit_site!(F, R, P, s, y, 1.0, nsp)
    global rec_add += sum(S.N0[i] for i in lo:lo+F.live[s]-1; init=0.0) - nb
    for i in lo:lo+F.live[s]-1; global sap_stock += S.N[i] * (1 - Fge(S.D[i], 5.0)); end
    global years_total += 1 / (SITE_SUB ? 4 : 1)          # plot-years (4 subplot sites = 1 plot)
  end
end
m = acc ./ SITE_MULT ./ years_total
# data
iv = DataFrame(Arrow.Table(joinpath(R2DIR, "intervals.arrow"))); raw = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_trees.arrow")))
ig = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_ingrowth.arrow")))
iv.pk = [(a, b, c, d) for (a, b, c, d) in zip(iv.st, iv.u, iv.c, iv.p)]
pl = combine(groupby(iv, :pk), :set => first => :set, [:trt, :dst, :harv, :planted] => ((a, b, c, d) -> !any(a .| b .| c .| d)) => :clean, nrow => :nint)
okp = Set(pl.pk[(coalesce.(pl.set, "") .== SET) .& pl.clean .& (pl.nint .>= 2)])
ivs = iv[in.(iv.pk, Ref(okp)), :]; ok = Set(ivs.iid); dt = Dict(r.iid => r.dt for r in eachrow(ivs)); pyears = sum(ivs.dt)
r = raw[in.(raw.iid, Ref(ok)) .& (raw.out .!= "missing"), :]; g = ig[in.(ig.iid, Ref(ok)), :]
al = r.out .== "alive"; d2 = coalesce.(r.dia2_cm, 0.0)
dstock = sum(r.w_ha[(r.dia_cm .>= 5) .& (r.dia_cm .< 12.7)] .* [dt[i] for i in r.iid[(r.dia_cm .>= 5) .& (r.dia_cm .< 12.7)]]) / pyears
din = sum(r.w_ha[al .& (r.dia_cm .< 5) .& (d2 .>= 5)]) / pyears
ddead = sum(r.w_ha[(r.out .== "dead") .& (r.dia_cm .>= 5) .& (r.dia_cm .< 12.7)]) / pyears
dout = sum(r.w_ha[al .& (r.dia_cm .>= 5) .& (r.dia_cm .< 12.7) .& (d2 .>= 12.7)]) / pyears
ding = sum(g.w_ha[(g.dia_cm .>= 5) .& (g.dia_cm .< 12.7)]) / pyears
@printf("POLE BUDGET %s %s (trees/ha/yr; stock trees/ha)\n", tag, SET)
@printf("            stock   inflow(5cm)  deaths  outflow(12.7)  | turnover = (deaths+outflow)/stock\n")
@printf("model    %7.1f   %8.2f   %7.2f   %8.2f      | %.3f /yr  (inflow − deaths − outflow = %+.2f)\n", m[1], m[2], m[3], m[4], (m[3] + m[4]) / m[1], m[2] - m[3] - m[4])
@printf("data     %7.1f   %8.2f   %7.2f   %8.2f      | %.3f /yr  (+ first-tallied 5–12.7 cm ingrowth %.2f/yr)\n", dstock, din, ddead, dout, (ddead + dout) / dstock, ding)

# sapling (2.54–5 cm) budget
sm = (sap_stock, rec_add, sap_dead, acc[2]) ./ SITE_MULT ./ years_total
sap = (r.dia_cm .< 5)
dsstock = sum(r.w_ha[sap] .* [dt[i] for i in r.iid[sap]]) / pyears
dsin = sum(g.w_ha[g.dia_cm .< 5]) / pyears
dsdead = sum(r.w_ha[(r.out .== "dead") .& sap]) / pyears
@printf("SAPLING BUDGET (2.54–5 cm)  stock   inflow(recruits / new tallied)  deaths  outflow(5 cm)  | death rate  outflow rate\n")
@printf("model                       %6.1f   %8.2f                       %7.2f  %8.2f       | %.3f /yr   %.3f /yr\n", sm[1], sm[2], sm[3], sm[4], sm[3] / sm[1], sm[4] / sm[1])
@printf("data                        %6.1f   %8.2f                       %7.2f  %8.2f       | %.3f /yr   %.3f /yr\n", dsstock, dsin, dsdead, din, dsdead / dsstock, din / dsstock)
