# Pole (5–12.7 cm) demography: HBAD v2 free runs (Sim B sites of SET) vs the tracked trees of the same clean natural plots.
# Model: every cohort-year spent in the pole class → annual survival and dD, by origin (initial cohort vs recruit) and by
#        overtopping class (SDI of taller cohorts per ha, from the kernel's own snapshot).
# Data:  tracked trees 5–12.7 cm (undisturbed natural intervals of the SET plots) → annualised survival and dD by the same
#        overtopping class (SDI per ha of taller trees on the SUBPLOT, measured heights).
#   HBAD2_SITE=subplot HBAD2_FORM=2 HBAD2_SIZE_SPLIT=5,12.7,25,40 ./julia_gdal.sh --project=. tools/hbad2/pole_demography.jl <θ.csv> [recr.csv|-] [set]
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "simB_lik.jl"))
CLO = parse(Float64, get(ENV, "CLASS_LO", "5")); CHI = parse(Float64, get(ENV, "CLASS_HI", "12.7"))
θpath = ARGS[1]; rpath = length(ARGS) >= 2 && ARGS[2] != "-" ? ARGS[2] : ""; SET = length(ARGS) >= 3 ? ARGS[3] : "val"
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
lon = CSV.read("/workspace/species_longevity_ref.csv", DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
P, G = decode(read_theta(θpath, nsp), nsp, longev)
rc = CSV.read(joinpath(R2DIR, SITE_SUB ? "recruitment_coefs_subplot.csv" : "recruitment_coefs.csv"), DataFrame); rcd = Dict(String(r.key) => r for r in eachrow(rc))
R = isempty(rpath) ? [RecP(rcd[k].rho0, rcd[k].rhoS, rcd[k].rhoL, 2.54, 3.0, 5.0) for k in keys_] :
    (z = vec(CSV.read(rpath, DataFrame).theta); [RecP(z[3k-2], z[3k-1], z[3k], 2.54, 3.0, 5.0) for k in 1:nsp])
B = build_simB(SET, keys_); _reset!(B); F = B.F; S = F.S
ocls(o) = o < 100 ? "<100" : o < 250 ? "100-250" : o < 450 ? "250-450" : ">=450"
acc = Dict{Tuple{String,String},Vector{Float64}}(); rec_in = 0.0; site_years = 0; sap_stock = 0.0       # (origin, O class) → [Σ N·surv weight, Σ N, Σ N dD, n]
for s in 1:length(F.live)
  lo = S.off[s]; live0 = B.live0[s]
  for y in 1:B.years[s]
    n = F.live[s]; hi = lo + n - 1
    preN = S.N[lo:hi]; preD = S.D[lo:hi]
    n > 0 && HBAD2.step_site!(S, P, G, lo, hi, 1.0, S.planted[s])
    smax = (t = sum(B.N0[lo:lo] .* 0) ; 0.0)
    tot = 0.0; wm = 0.0
    for i in lo:hi; x = HBAD2.sdi(preN[i-lo+1], preD[i-lo+1]); tot += x; wm += x * P[S.sp[i]].sdimax; end
    sm = tot > 0 ? wm / tot : 600.0
    for i in lo:hi
      k = i - lo + 1; (CLO <= preD[k] < CHI && preN[k] > 0.5) || continue
      key = (i - lo + 1 > live0 ? "recruit" : "initial", ocls(S.O[i] * sm))
      v = get!(acc, key, zeros(4)); v[1] += S.N[i]; v[2] += preN[k]; v[3] += preN[k] * (S.D[i] - preD[k]); v[4] += 1
    end
    nb = sum(S.N0[i] for i in lo:lo+F.live[s]-1; init=0.0)
    HBAD2.recruit_site!(F, R, P, s, y, 1.0, nsp)
    global rec_in += sum(S.N0[i] for i in lo:lo+F.live[s]-1; init=0.0) - nb; global site_years += 1
    # sapling stock (< 5 cm) and its open/closed split
    for i in lo:lo+F.live[s]-1; S.D[i] < 5 && (global sap_stock += S.N[i]); end
  end
end
@printf("MODEL recruit inflow %.1f trees/ha/yr (per site-year), mean sapling (<5 cm) stock %.0f /ha\n", rec_in / site_years, sap_stock / site_years)
println("MODEL pole cohort-years (SET $SET): annual survival / dD (cm/yr) by origin × overtopping SDI class (per ha)")
for k in sort(collect(keys(acc))); v = acc[k]; @printf("  %-8s %-8s n=%7d  surv %.4f (mort %.2f %%/yr)  dD %.3f\n", k[1], k[2], v[4], v[1] / v[2], 100(1 - v[1] / v[2]), v[3] / v[2]); end
# data side
iv = DataFrame(Arrow.Table(joinpath(R2DIR, "intervals.arrow"))); raw = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_trees.arrow")))
ok = Set(iv.iid[.!(iv.trt .| iv.dst .| iv.harv .| iv.planted) .& (iv.dt .> 2) .& (coalesce.(iv.set, "") .== SET)])
r = raw[in.(raw.iid, Ref(ok)), :]; ivd = Dict(x.iid => x.dt for x in eachrow(iv))
r.O = zeros(nrow(r))
for g in groupby(r, [:iid, :sb])
  ix = parentindices(g)[1]
  for (a, i) in enumerate(ix)
    r.O[i] = 4 * sum(g.w_ha[j] * (g.dia_cm[j] / 25.4)^1.605 for j in eachindex(g.ht_m) if g.ht_m[j] > g.ht_m[a]; init=0.0)
  end
end
p = r[(r.dia_cm .>= CLO) .& (r.dia_cm .< CHI) .& (r.out .!= "missing"), :]; p.dt = [ivd[i] for i in p.iid]; p.c = ocls.(p.O)
println("DATA tracked poles (natural undisturbed, SET $SET): annualised survival / dD by subplot overtopping SDI class")
for g in groupby(sort(p, :c), :c)
  alive = g.out .== "alive"; sv = mean(alive)^(1 / mean(g.dt))
  @printf("  %-8s n=%6d  surv %.4f (mort %.2f %%/yr)  dD %.3f\n", g.c[1], nrow(g), sv, 100(1 - sv), mean((g.dia2_cm[alive] .- g.dia_cm[alive]) ./ g.dt[alive]))
end

ig = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_ingrowth.arrow"))); ig = ig[in.(ig.iid, Ref(ok)) .& (ig.dia_cm .< 12.7), :]
dts = sum(ivd[i] for i in ok)
@printf("DATA sapling ingrowth (new trees ≥ 2.54 cm on microplots) %.1f trees/ha/yr per plot-year; sapling (<5 cm) stock at v1 %.0f /ha\n", sum(ig.w_ha) / dts,
        sum(r.w_ha[r.dia_cm .< 5]) / length(ok))
