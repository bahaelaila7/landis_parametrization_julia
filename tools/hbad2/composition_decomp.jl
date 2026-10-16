# Composition drift diagnosis: per species BA change over the FIRST remeasurement interval of the Sim B plots, split into
#   growth of survivors  Σ_survivors w (D2² − D1²) π/4 ,  mortality loss −Σ_dead w D1² π/4 ,  recruitment (ingrowth BA at v2),
# observed (tracked trees + ingrowth of the interval) vs simulated (free run, cohorts split by origin: initial vs recruit).
#   HBAD2_VIGOUR=1 HBAD2_SITE=subplot HBAD2_FORM=2 HBAD2_SIZE_SPLIT=5,12.7,25,40 ./julia_gdal.sh --project=. tools/hbad2/composition_decomp.jl <θ.csv> <recr.csv|-> [set]
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "simB_lik.jl"))
θpath = ARGS[1]; rpath = ARGS[2] == "-" ? "" : ARGS[2]; SET = length(ARGS) >= 3 ? ARGS[3] : "val"
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_); kix = Dict(k => i for (i, k) in enumerate(keys_))
lon = CSV.read("/workspace/species_longevity_ref.csv", DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
P, G = decode(read_theta(θpath, nsp), nsp, longev)
rc = CSV.read(joinpath(R2DIR, SITE_SUB ? "recruitment_coefs_subplot.csv" : "recruitment_coefs.csv"), DataFrame); rcd = Dict(String(r.key) => r for r in eachrow(rc))
R = isempty(rpath) ? [RecP(rcd[k].rho0, rcd[k].rhoS, rcd[k].rhoL, 2.54, 3.0, 5.0) for k in keys_] :
    (z = vec(CSV.read(rpath, DataFrame).theta); [RecP(clamp(z[3k-2], -12, 2), clamp(z[3k-1], -1, 4), clamp(z[3k], -10, 2), 2.54, 3.0, 5.0) for k in 1:nsp])
B = build_simB(SET, keys_); _reset!(B); F = B.F; S = F.S
ba(N, D) = N * pi / 4 * (D / 100)^2
sim = zeros(nsp, 4)        # growth of initial cohorts, mortality loss of initial cohorts, recruit BA, initial BA   (Σ over sites, per ha / SITE_MULT)
for s in 1:length(F.live)
  lo = S.off[s]; live0 = B.live0[s]; y1 = B.vis_year[s][1]      # first later visit
  N0 = S.N[lo:lo+live0-1]; D0 = S.D[lo:lo+live0-1]
  for i in 1:live0; sim[S.sp[lo+i-1], 4] += ba(N0[i], D0[i]); end
  for y in 1:y1
    n = F.live[s]; n > 0 && HBAD2.step_site!(S, P, G, lo, lo + n - 1, 1.0, S.planted[s]); HBAD2.recruit_site!(F, R, P, s, y, 1.0, nsp)
  end
  for i in 1:live0
    j = lo + i - 1; k = S.sp[j]
    sim[k, 1] += S.N[j] * pi / 4 * ((S.D[j] / 100)^2 - (D0[i] / 100)^2); sim[k, 2] -= (N0[i] - S.N[j]) * pi / 4 * (D0[i] / 100)^2
  end
  for j in lo+live0:lo+F.live[s]-1; sim[S.sp[j], 3] += ba(S.N[j], S.D[j]); end
end
sim ./= SITE_MULT
# observed: first interval of the same plots (clean natural, SET)
iv = DataFrame(Arrow.Table(joinpath(R2DIR, "intervals.arrow"))); raw = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_trees.arrow")))
ig = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_ingrowth.arrow")))
iv.pk = [(a, b, c, d) for (a, b, c, d) in zip(iv.st, iv.u, iv.c, iv.p)]
pl = combine(groupby(iv, :pk), :set => first => :set, [:trt, :dst, :harv, :planted] => ((a, b, c, d) -> !any(a .| b .| c .| d)) => :clean, nrow => :nint,
             [:k, :iid] => ((k, i) -> i[argmin(k)]) => :iid1)
pl = pl[(coalesce.(pl.set, "") .== SET) .& pl.clean .& (pl.nint .>= 2), :]
first_iids = Set(pl.iid1)
big = Set(k for k in keys_ if k[1] != '_'); keyof(s, g) = s in big ? s : "_" * g
r = raw[in.(raw.iid, Ref(first_iids)), :]; g = ig[in.(ig.iid, Ref(first_iids)), :]
obs = zeros(nsp, 4)
for t in eachrow(r)
  k = kix[keyof(t.sym, t.grp)]; obs[k, 4] += ba(t.w_ha, t.dia_cm)
  t.out == "alive" && (obs[k, 1] += t.w_ha * pi / 4 * ((t.dia2_cm / 100)^2 - (t.dia_cm / 100)^2))
  t.out == "dead" && (obs[k, 2] -= ba(t.w_ha, t.dia_cm))
end
for t in eachrow(g); obs[kix[keyof(t.sym, t.grp)], 3] += ba(t.w_ha, t.dia_cm); end
npl = length(first_iids); nsite_pl = length(unique(B.plot_of))
sim ./= nsite_pl; obs ./= npl
@printf("first interval, %s: %d plots (obs) / %d plots (sim); mean per plot, m²/ha\n", SET, npl, nsite_pl)
@printf("%-6s %7s | %8s %8s | %8s %8s | %8s %8s | %8s %8s\n", "key", "BA0", "grow_o", "grow_s", "mort_o", "mort_s", "recr_o", "recr_s", "net_o", "net_s")
for k in sortperm(-obs[:, 4])
  @printf("%-6s %7.3f | %8.3f %8.3f | %8.3f %8.3f | %8.4f %8.4f | %8.3f %8.3f\n", keys_[k], obs[k, 4], obs[k, 1], sim[k, 1], obs[k, 2], sim[k, 2], obs[k, 3], sim[k, 3],
          sum(obs[k, 1:3]), sum(sim[k, 1:3]))
end
