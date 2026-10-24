# HBAD3P pipeline verification (stages vdata / vmodel of tools/hbad3/pipeline3p.sh; docs/hbad3p_pipeline.md §verification). Every check prints PASS / FAIL
# with its numbers; the exit code is the number of FAILs.
#   (pipeline env) ./julia_gdal.sh --project=. --threads=N tools/hbad3/verify3p.jl data            (data products of the region)
#   (pipeline env) ./julia_gdal.sh --project=. --threads=N tools/hbad3/verify3p.jl model <tag>     (kernel, likelihood, fit, recruitment, free run of fit3p_<tag>)
include(joinpath(@__DIR__, "common3.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plugins", "HBAD3", "HBAD3P.jl")); import .HBAD3P
include(joinpath(@__DIR__, "lib3p.jl"))
using Random
MODE = ARGS[1]; NFAIL = Ref(0)
check(id, ok, msg) = (println(rpad(ok ? "PASS" : "FAIL", 5), rpad(id, 5), msg); ok || (NFAIL[] += 1); ok)
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_); kix = Dict(k => i for (i, k) in enumerate(keys_))
T = interval_tables(keys_); ivd = Dict(r.iid => r for r in eachrow(T.iv)); wp = water_pairs()
println("region ", get(ENV, "HBAD2_REGION", "(none)"), " | keys ", join(keys_, ","), " | tracked trees ", nrow(tr))

if MODE == "data"; let
  iv, raw, ig = T.iv, T.raw, T.ig
  # D1 interval product: identifiers, dates, outcomes, FIA plot design
  check("D1a", allunique(iv.iid) && all(iv.d2 .> iv.d1) && all(abs.(iv.dt .- Dates.value.(iv.d2 .- iv.d1) ./ 365.25) .< 1e-9), "interval ids unique, d2 > d1, dt = (d2 − d1)/365.25")
  check("D1b", nrow(unique(iv, [:st, :u, :c, :p, :d1])) == nrow(iv), "one interval per (plot, v1 date)")
  ps = combine(groupby(iv, [:st, :u, :c, :p]), :set => (x -> length(unique(x))) => :n); check("D1c", all(ps.n .== 1), "each plot in exactly one split set ($(nrow(ps)) plots)")
  check("D1d", nrow(unique(raw, [:iid, :sb, :tr])) == nrow(raw), "tracked trees unique per (interval, subplot, tree)")
  check("D1e", issubset(Set(raw.out), Set(["alive", "dead", "harvest", "missing"])), "outcomes ⊂ {alive, dead, harvest, missing}")
  al = raw.out .== "alive"
  check("D1f", all(raw.dia2_cm[al] .> 0) && all(raw.ht2_m[al] .> 0) && all(raw.w2_ha[al] .> 0) && all(isnan.(raw.dia2_cm[.!al])), "alive ⇔ v2 diameter / height / weight present")
  tpa = round.(raw.w_ha ./ 2.47105; digits=3); sap = raw.dia_cm .< 12.7
  check("D1g", all(tpa[sap] .== 74.965) && all(tpa[.!sap] .== 6.018), "FIA annual design: D < 12.7 cm on the microplot (74.965 TPA), else subplot (6.018)")
  igs = ig.dia_cm .< 12.7; itpa = round.(ig.w_ha ./ 2.47105; digits=3)
  check("D1h", all(itpa[igs] .== 74.965) && all(itpa[.!igs] .== 6.018) && all(ig.dia_cm .>= 2.54), "ingrowth: saplings on the microplot (RSCALE = 74.965 × 2.47105 = $(round(74.965 * 2.47105; digits=1)) /ha), D ≥ 2.54")
  check("D1i", isapprox(HBAD3P.RSCALE, 74.965 * 2.47105; rtol=1e-3), "HBAD3P.RSCALE = one microplot sapling per ha of the plot")
  # D2 region / clean-interval filter of load_trees
  ivt = [ivd[i] for i in unique(tr.iid)]
  check("D2a", all(r -> !(r.trt || r.dst || r.harv) && r.dt > 2, ivt), "every fitted interval clean (no treatment / disturbance / harvest) and dt > 2 ($(length(ivt)) intervals)")
  check("D2b", all(r -> r.set in ("train", "val", "test"), ivt), "every fitted interval has a split set")
  # D3 cohorts (independent recomputation on the TRAIN set)
  C = build_cohorts(tr, "train", keys_); g = C.coh; d = tr[tr.set .== "train", :]
  check("D3a", isapprox(sum(g.N1), sum(d.w_ha); rtol=1e-12) && sum(g.n1) == count(d.out .!= "missing") && sum(g.s) == count(d.out .== "alive"), "cohort totals = tree totals (N, counted, alive)")
  check("D3b", all(g.s .<= g.n1) && all(g.D1 .> 0) && all(g.H1 .> 0), "s ≤ n1; D1, H1 > 0")
  rng = Xoshiro(1); bad = 0
  for i in rand(rng, 1:nrow(g), 300)
    m = (d.site .== g.site[i]) .& (d.key .== g.key[i]) .& ((floor.(Int, (d.y1 .- d.age) ./ BIRTH_W) .* 100 .+ dclass.(d.dia_cm)) .== g.bc[i]); w = d.w_ha[m]
    a = d.out[m] .== "alive"
    ok = isapprox(g.D1[i], sqrt(sum(w .* d.dia_cm[m] .^ 2) / sum(w)); rtol=1e-12) && isapprox(g.H1[i], sum(w .* d.ht_m[m]) / sum(w); rtol=1e-12) &&
         (g.s[i] == 0 || isapprox(g.D2o[i], sqrt(sum(w[a] .* d.dia2_cm[m][a] .^ 2) / sum(w[a])); rtol=1e-12))
    bad += !ok
  end
  check("D3c", bad == 0, "300 random cohorts: QMD D1, mean H1, survivors' QMD D2o recomputed from trees ($(bad) mismatches)")
  nq = length(VQ_Z); w = [sum(C.S.N0[C.parent .== p]) for p in rand(rng, 1:nrow(g), 200)]
  check("D3d", isapprox(sum(VQ_W), 1; atol=1e-12) && isapprox(sum(VQ_W .* VQ_Z), 0; atol=1e-12) && isapprox(sum(VQ_W .* VQ_Z .^ 2), 1; atol=1e-9) && length(C.S.N0) == nq * nrow(g),
        "vigour quadrature ($(nq) nodes): weights sum 1, mean 0, variance 1; $(nq) sub-cohorts per cohort")
  okp = all(s -> all(C.coh.site[C.parent[i]] == C.coh.site[C.parent[C.S.off[s]]] for i in C.S.off[s]:C.S.off[s+1]-1), 1:length(C.S.off)-1)
  check("D3e", okp && issorted(C.parent), "site ranges contiguous: every sub-cohort of a site range belongs to that site")
  ck = combine(groupby(DataFrame(site=d.site, key=d.key, bc=floor.(Int, (d.y1 .- d.age) ./ BIRTH_W) .* 100 .+ dclass.(d.dia_cm), sap=d.dia_cm .< 12.7), [:site, :key, :bc]),
               :sap => (x -> any(x) && !all(x)) => :mixed)
  check("D3f", 12.7 in SIZE_BREAKS && !any(ck.mixed), "no cohort mixes microplot saplings (D < 12.7) with subplot trees (D ≥ 12.7): $(count(ck.mixed)) mixed of $(nrow(ck)) (dclass=$(DCLASS))")
  # D4 water, D5 forested area, D6 traits
  ids = unique(tr.iid); W = [get(wp, i, (NaN, NaN)) for i in ids]
  check("D4", all(x -> isfinite(x[1]) && isfinite(x[2]) && 0 < x[1] <= 1.0001 && 0 < x[2] <= 1.0001, W), "water index of every fitted interval: period and normal AET/PET in (0, 1] ($(length(ids)))")
  IF = interval_forest(T.iv, forest_table()); fs = [IF[(i, sb)] for i in ids for sb in 1:4]
  check("D5a", all(x -> !isnan(x[2]), fs), "forest status of every fitted interval-subplot at v2 known ($(length(fs)))")
  stocked = Set((r.iid, r.sb) for r in eachrow(d)); f1s = [IF[k][1] for k in stocked]
  check("D5b", count(isnan, f1s) == 0, "forest status at v1 (or v2) of every stocked TRAIN subplot known")
  println("      stocked TRAIN subplots: f = 1 $(count(==(1), f1s)), partial $(count(x -> 0 < x < 1, f1s)), f = 0 although stocked (area 1 used) $(count(==(0), f1s))")
  sg = combine(groupby(ig[(ig.dia_cm .< 12.7) .& [i in Set(ids) for i in ig.iid], :], [:iid, :sb]), nrow => :n)
  nfy = sum(r.n for r in eachrow(sg) if IF[(r.iid, r.sb)][2] == 0; init=0)
  check("D5c", nfy == 0, "no sapling ingrowth tallied on a nonforest microplot ($(nfy))")
  C2 = build_cohorts(tr, "train", keys_); scale_area!(C2, IF)
  r1 = C.S.N0 ./ C.agg.N0[C.parent]; r2 = C2.S.N0 ./ C2.agg.N0[C2.parent]
  check("D5d", maximum(abs.(r1 .- r2)) < 1e-12 && all(C2.S.N0 .>= C.S.N0 .* (1 - 1e-12)), "area scaling: sub-cohort weights unchanged, densities only increase (÷ a ≤ 1)")
  TD = key_traits(tr, keys_, ones(nsp))
  check("D6", all(1 .<= TD.ST .<= 5), "shade class of every key in [1, 5]")
end; end

if MODE == "model"; let
  tag = ARGS[2]; FD = joinpath(R3DIR, "fit3p_" * tag); check_meta(FD); FAREA = meta_farea(FD)
  M = load_fit3p(FD, keys_); P, G = M.P, M.G
  # K1 independent reference implementation of one step (written from the documented equations, scalar code) vs step_site! on random sites
  function ref_step!(N, D, H, A, CR, z, sp, P, G, Δ, pl, W, Wn)
    n = length(N); SD = [N[i] * (D[i] / 25.4)^1.605 for i in 1:n]; RD = sum(SD) / G.sdimax
    O = [(0.5SD[i] + sum((H[j] > H[i] ? SD[j] : H[j] == H[i] && j != i ? 0.5SD[j] : 0.0) for j in 1:n)) / G.sdimax for i in 1:n]
    Hs(p, x) = 1.37 + (p.Hmax - 1.37) * (1 - exp(-G.s * x / (p.Hmax - 1.37)))^G.c
    V = zeros(n)
    for i in 1:n
      p = P[sp[i]]; cw = exp(G.gR * RD) * W^G.ω * exp(G.σv * z[i]) * (CR[i] / 0.4)^G.φ
      g = (G.f0 + (1 - G.f0) * exp(-G.a * O[i])) * cw; V[i] = min(g, 1.0)
      hyd = max(0.0, 1 - min(H[i] / (1.1p.Hmax), 1.0)^G.θ)
      dV = max(exp(p.lnε) * (D[i] / 20)^G.b * 122.98 * g * hyd * (pl ? exp(G.pD) : 1.0) - G.ρ * D[i]^2 * H[i], 0.0)
      dD = dV * Δ / (2D[i] * max(H[i], 1.37)); Lh = exp(-G.kh * O[i])
      dH = (Hs(p, D[i] + dD) - Hs(p, D[i])) * (1 + G.η * (1 - Lh)) + G.λ * Δ * (Hs(p, D[i]) * (1 + G.μ * (1 - Lh)) - H[i])
      gm = exp(-G.am * O[i]) * cw; fM = (pl ? exp(G.pM) : 1.0) * exp(G.mW * (Wn - W) / 0.1)
      h = fM * (G.m0 / p.longevity + G.ms * exp(G.σST * (p.ST - 3)) * exp(-G.β * gm) * (D[i] / 20)^G.ζ) + exp(G.mA * (A[i] / p.longevity - 1))
      crt = G.crlo + (G.crhi - G.crlo) * exp(-G.kc * O[i])
      N[i] *= exp(-h * Δ); D[i] += dD; H[i] += max((pl ? exp(G.pH) : 1.0) * dH, 0.0)
      CR[i] = clamp(CR[i] + (CR[i] > crt ? G.τr : G.τe) * (crt - CR[i]) * Δ, 0.05, 0.95); A[i] += Δ
    end
    for _ in 1:6
      SD = [N[i] * (D[i] / 25.4)^1.605 for i in 1:n]; ex = sum(SD) - G.sdimax; ex <= 1e-6 * G.sdimax && break
      w = [SD[i] * (1 - V[i]) + 1e-9 * SD[i] for i in 1:n]; ws = sum(w)
      for i in 1:n; SD[i] > 0 && (N[i] *= 1 - min(SD[i], ex * w[i] / ws) / SD[i]); end
    end
  end
  rng = Xoshiro(7); maxd = 0.0; maxdN = 0.0; nst = 0
  for trial in 1:300
    n = rand(rng, 1:9); sp = rand(rng, 1:nsp, n); D = 2.54 .+ 60rand(rng, n); H = [rand(rng) < 0.3 && i > 1 ? 0.0 : 1.4 + 35rand(rng) for i in 1:n]
    for i in 2:n; H[i] == 0.0 && (H[i] = H[i-1]); end                              # ties
    N = (rand(rng) < 0.3 ? 4000.0 : 600.0) .* rand(rng, n) .+ 1; A = 1 .+ 200rand(rng, n); CR = 0.05 .+ 0.9rand(rng, n); z = rand(rng, (-1.36, 0.0, 1.36, 2.86), n)
    Gt = HBAD3P.GlobPP((f in (:sdimax, :mA) ? getfield(G, f) : getfield(G, f) * (1 + 0.3randn(rng)) for f in fieldnames(HBAD3P.GlobPP))...)
    pl = rand(rng, Bool); Wv = 0.3 + 0.7rand(rng); Wn = 0.3 + 0.7rand(rng); Δ = rand(rng) < 0.3 ? rand(rng) : 1.0
    S = HBAD3P.SitesP([1, n + 1], sp, copy(N), copy(D), copy(H), copy(A), [0], [0.0], [pl], z, [Wv], [Wn], copy(CR))
    HBAD3P.step_site!(S, P, Gt, 1, n, Δ, pl, Wv, Wn)
    Nr, Dr, Hr, Ar, CRr = copy(N), copy(D), copy(H), copy(A), copy(CR); ref_step!(Nr, Dr, Hr, Ar, CRr, z, sp, P, Gt, Δ, pl, Wv, Wn)
    nst += sum(N .* (D ./ 25.4) .^ 1.605) > G.sdimax
    dN = maximum(abs.(S.N .- Nr) ./ max.(Nr, 1e-12)); dD_ = maximum(abs.(S.D .- Dr) ./ max.(abs.(Dr), 1)); dH_ = maximum(abs.(S.H .- Hr) ./ max.(abs.(Hr), 1)); dC = maximum(abs.(S.CR .- CRr))   # relative: perturbed globals can drive D to 1e6+
    get(ENV, "K1_DIAG", "0") == "1" && max(dN, dD_, dH_, dC) > 1e-12 && println("      K1 trial $(trial): n $(n) dN $(dN) dD $(dD_) dH $(dH_) dCR $(dC) | N $(round.(S.N; sigdigits=4)) ref $(round.(Nr; sigdigits=4))")
    # N is defined only to the self-thinning stopping rule (both implementations stop at an excess <= 1e-6 sdimax, <= 6 passes): when rounding flips
    # that branch, one implementation thins once more by <= ~1e-6 relative (nested-CV fold k3: 7e-7 and 3e-6 on extinct-level, D ~ 1e6 cohorts)
    maxdN = max(maxdN, dN); maxd = max(maxd, dD_, dH_, dC, maximum(abs.(S.A .- Ar)))
  end
  check("K1", maxd < 1e-9 && maxdN < 1e-5, "step_site! = independent reference step on 300 random sites (ties, planted, partial years, $(nst) over sdimax): max relative diff D/H/CR/A $(maxd), N $(maxdN) (N tolerance 1e-5: the self-thinning stopping rule)")
  # K2 invariants and determinism on the real VAL sites with the fitted parameters
  IFOR = FAREA ? interval_forest(T.iv, forest_table()) : nothing
  VA = sima_sites(tr, "val", keys_, wp, IFOR); S = VA.S
  HBAD3P.simulate!(S, P, G); N1, D1, H1, CR1 = copy(S.N), copy(S.D), copy(S.H), copy(S.CR)
  check("K2a", all(S.N .<= S.N0 .* (1 + 1e-12)) && all(S.N .>= 0) && all(S.D .>= S.D0) && all(S.H .>= S.H0) && all(0.05 .<= S.CR .<= 0.95), "VAL run: N ≤ N0, D ≥ D0, H ≥ H0, CR ∈ [0.05, 0.95]")
  HBAD3P.simulate!(S, P, G); check("K2b", S.N == N1 && S.D == D1 && S.H == H1 && S.CR == CR1, "simulate! deterministic (bitwise) with $(Threads.nthreads()) threads")
  over = 0; for s in 1:length(S.off)-1; lo, hi = S.off[s], S.off[s+1]-1; over += sum(S.N[i] * (S.D[i] / 25.4)^1.605 for i in lo:hi) > G.sdimax * (1 + 1e-6); end
  check("K2c", over <= 0.001 * (length(S.off) - 1), "self-thinning: sites still above sdimax after the run $(over) of $(length(S.off)-1)")
  ns_ = length(S.nyr)                                                        # one whole year through the nyr path = through the partial-year path with frac = 1
  Sa = HBAD3P.SitesP(S.off, S.sp, S.N0, S.D0, S.H0, S.A0, ones(Int, ns_), zeros(ns_), S.planted, S.zq, S.W, S.Wn, S.CR0)
  Sb = HBAD3P.SitesP(S.off, S.sp, S.N0, S.D0, S.H0, S.A0, zeros(Int, ns_), ones(ns_), S.planted, S.zq, S.W, S.Wn, S.CR0)
  HBAD3P.simulate!(Sa, P, G); HBAD3P.simulate!(Sb, P, G)
  check("K2d", Sa.N == Sb.N && Sa.D == Sb.D && Sa.H == Sb.H && Sa.CR == Sb.CR && Sa.N != Sa.N0, "one year via nyr = 1 equals one year via frac = 1 (and changes the state)")
  # L likelihood: profiled σ = numerical optimum; survival term recomputed
  nl = nll_parts(cohort_view(VA), VA.obs); V = cohort_view(VA); o = VA.obs
  e2 = Float64[]; sw = Float64[]; bm = 0.0
  for i in eachindex(V.N)
    o.n1[i] == 0 && continue; p = clamp(V.N[i] / V.N0[i], 1e-9, 1 - 1e-9); bm -= o.s[i] * log(p) + (o.n1[i] - o.s[i]) * log(1 - p)
    o.s[i] > 0 && (push!(e2, log(V.D[i] / o.D2o[i])^2); push!(sw, o.s[i]))
  end
  f(σ) = sum(0.5log(σ^2) + sw[i] * e2[i] / (2σ^2) for i in eachindex(e2)); lo_, hi_ = 1e-3, 2.0
  for _ in 1:200; m1 = lo_ + (hi_ - lo_) / 3; m2 = hi_ - (hi_ - lo_) / 3; f(m1) < f(m2) ? (hi_ = m2) : (lo_ = m1); end
  check("L1", isapprox(f(lo_), nl.D + length(e2) / 2; rtol=1e-9), "D term = min over σ of the precision-weighted Gaussian (− c/2): $(round(nl.D; digits=2)) vs $(round(f(lo_) - length(e2) / 2; digits=2))")
  check("L2", isapprox(bm, nl.mort; rtol=1e-12), "survival term = binomial on tracked tree counts recomputed: $(round(nl.mort; digits=2))")
  # F fit reproducibility: the saved parameters reproduce the logged FIT totals
  lgt = read(joinpath(FD, "log.txt"), String); mt = match(r"FIT   train (-?[0-9.]+) .*VAL (-?[0-9.]+)", lgt)
  if mt === nothing
    check("F1", false, "no FIT line in $(FD)/log.txt")
  else
    TRc = sima_sites(tr, "train", keys_, wp, IFOR); HBAD3P.simulate!(TRc.S, P, G); pt = nll_parts(cohort_view(TRc), TRc.obs).total
    check("F1", abs(pt - parse(Float64, mt[1])) < 0.2 && abs(nl.total - parse(Float64, mt[2])) < 0.2, "saved species.csv / globals.csv reproduce the logged FIT: train $(round(pt; digits=1)) (log $(mt[1])), val $(round(nl.total; digits=1)) (log $(mt[2]))")
  end
  zt = CSV.read(joinpath(FD, "z.csv"), DataFrame)
  check("F2", hasproperty(zt, :name) && allunique(zt.name) && all(startswith.(zt.name[1:nsp], "ε:")), "z.csv carries unique coordinate names (warm starts map by name)")
  # R recruitment: Poisson score per key at the optimum, and the probe exposure vs the free-run mechanism itself
  if isfile(joinpath(FD, "recruitment.csv"))
    Pr, Rg = load_recruit3p(FD, P)
    rl = read(joinpath(FD, "recruit_log.txt"), String)
    check("R0", occursin("VAL deviance", rl), "recruit_log.txt complete")
    # sample of TRAIN natural intervals: probes vs run_free! with the fitted recruitment (one interval each, no initial merging)
    IF = interval_forest(T.iv, forest_table()); rawby = Dict(k.iid => DataFrame(g) for (k, g) in pairs(groupby(T.raw, :iid)))
    ids = sort([i for i in unique(tr.iid) if ivd[i].set == "train" && !ivd[i].planted])[1:7:end]
    sites = NamedTuple[]; fm = Float64[]; am = Float64[]
    for i in ids, sb in 1:4
      f1, fm2 = IF[(i, sb)]; (isnan(fm2) || fm2 == 0) && continue
      t = get(rawby, i, T.raw[1:0, :]); a = area_of(f1)
      push!(sites, (coh=subplot_cohorts(t[t.sb .== sb, :], kix, year(ivd[i].d1), FAREA ? a : 1.0), nyr=floor(Int, ivd[i].dt), frac=ivd[i].dt - floor(ivd[i].dt), W=wp[i][1], Wn=wp[i][2],
                    seedplot=BitVector([any((t.key .== k) .& (t.dia_cm .>= 12.7)) for k in keys_]), planted=false)); push!(fm, fm2); push!(am, FAREA ? 1.0 : a)
    end
    ex = probe_exposure(sites, P, G); zr = vcat([p.lnF - log(4.0) for p in Pr], log(Rg.λ0), Rg.λ1, log(Rg.ε0))
    X = (K=repeat(1:nsp, length(sites)), Y=zeros(nsp * length(sites)), fm=repeat(fm; inner=nsp), site=repeat(1:length(sites); inner=nsp), ex=ex, μ=zeros(nsp * length(sites)))
    μ = recr_mu!(zeros(length(X.K)), zr, X, M.ST)
    # the same sites through run_free! (real recruitment, merging into RCLASS slots, recruits' own competition)
    nq = length(VQ_Z); off = [1]; for st in sites; push!(off, off[end] + length(st.coh.N) + nq * nsp * (ceil(Int, (st.nyr + st.frac) / HBAD3P.RCLASS) + 1)); end   # slots as freerun (lib3p): intervals > 20 yr need > 4 recruit classes
    n = off[end] - 1; spv = zeros(Int32, n); Nv = zeros(n); Dv = fill(2.54, n); Hv = fill(1.37, n); Av = zeros(n); zv = zeros(n); CRv = fill(0.4, n)
    for (s, st) in enumerate(sites); r = off[s]:off[s]+length(st.coh.N)-1; spv[r] .= st.coh.sp; Nv[r] .= st.coh.N; Dv[r] .= st.coh.D; Hv[r] .= st.coh.H; Av[r] .= st.coh.A; zv[r] .= st.coh.zq; CRv[r] .= st.coh.CR; end
    ns = length(sites); HBAD3P.set_quadrature!(VQ_Z, VQ_W)
    Sf = HBAD3P.SitesP(off, spv, Nv, Dv, Hv, Av, zeros(Int, ns), zeros(ns), fill(false, ns), zv, [s.W for s in sites], [s.Wn for s in sites], CRv)
    live0 = [length(s.coh.N) for s in sites]
    F = HBAD3P.FreeSitesP(Sf, copy(live0), zeros(Int, nsp, ns), zeros(Int, nsp, ns), zeros(Int, ns), BitMatrix(hcat([s.seedplot for s in sites]...)), am)
    HBAD3P.run_free!(F, Pr, G, Rg, [s.nyr for s in sites], (s, y) -> nothing; frac=[s.frac for s in sites])
    cnt = zeros(nsp)                        # expected microplot count of the recruits alive with D < 12.7 at the end: density (per forested ha) × fm / (4 RSCALE)
    for s in 1:ns, i in off[s]+live0[s]:off[s]+F.live[s]-1; Sf.D[i] < 12.7 && (cnt[Sf.sp[i]] += Sf.N[i] / am[s] * fm[s] / (4HBAD3P.RSCALE)); end
    mk = [sum(μ[X.K .== k]) for k in 1:nsp]; rt = sum(cnt) / sum(mk)
    check("R1", abs(rt - 1) < 0.05 && sum(F.dropped) == 0, "probe exposure × fitted rate = run_free! recruit survivors (sample of $(ns) TRAIN subplots): total ratio $(round(rt; digits=4)) (merging into 5-yr slots + recruits' own competition)")
    println("      per key run_free!/probe: ", join([@sprintf("%s %.3f", keys_[k], cnt[k] / max(mk[k], 1e-12)) for k in 1:nsp], " "))
    mp = match(r"mean probe survival to v2 \(TRAIN, entry-year weighted\): ([0-9.]+)", rl)
    mp === nothing || check("R2", 0 < parse(Float64, mp[1]) < 1, "mean probe survival to the next visit $(mp[1]) (< 1: recruits die before they are tallied; the old fit assumed 1)")
  else
    check("R0", false, "no recruitment.csv in $(FD) (run stage recruit)")
  end
  # FR free-run construction: at t = 0 the simulated plot BA equals the observed first-visit BA exactly
  ivbp = Dict((k.st, k.u, k.c, k.p) => sort(DataFrame(g), :k) for (k, g) in pairs(groupby(T.iv, [:st, :u, :c, :p])))
  regi = Set(tr.iid)
  allp = combine(groupby(T.iv, [:st, :u, :c, :p]), :iid => (x -> all(i -> i in regi && !ivd[i].planted, x)) => :ok, nrow => :n, :set => first => :set)
  plots = allp[allp.ok .& (allp.n .>= 2) .& (coalesce.(allp.set, "") .== "val"), :][1:3:end, :]
  B = build_free(plots, ivbp, T, interval_forest(T.iv, forest_table()), wp, keys_, kix, FAREA); F0 = free_store(B, nsp)
  rows = plot_values(F0, B); ba = rows[rows.var .== "BA", :]
  nsim = [sum(F0.S.N[i] * B.outm[s] / 4 for s in findall(==(p), B.pl_site) for i in B.off[s]:B.off[s]+F0.live[s]-1; init=0.0) for p in eachindex(B.res)]
  rawby = Dict(k.iid => DataFrame(g) for (k, g) in pairs(groupby(T.raw, :iid)))
  nobs = [sum(get(rawby, ivbp[(r.st, r.u, r.c, r.p)].iid[1], T.raw[1:0, :]).w_ha; init=0.0) for r in B.res]
  check("FR1", maximum(abs.(ba.sim .- ba.obs1)) < 1e-9 * max(1, maximum(ba.obs1)) && maximum(abs.(nsim .- nobs)) < 1e-9 * max(1, maximum(nobs)),
        "free-run sites at t = 0 reproduce the observed first-visit plot BA and stem density ($(nrow(ba)) plots; farea=$(FAREA))")
  IF2 = interval_forest(T.iv, forest_table()); nf = 0                    # independent count: subplots of the plots' first intervals with f = 0 and no tracked tree
  for r in B.res
    i1 = ivbp[(r.st, r.u, r.c, r.p)].iid[1]; t = get(rawby, i1, T.raw[1:0, :])
    nf += count(sb -> IF2[(i1, sb)][1] == 0 && !any(t.sb .== sb), 1:4)
  end
  check("FR2", nf == count(==(0.0), B.rmult) && all(B.rmult .<= 1), "nonforest empty subplots (counted from the forest table and tree lists: $(nf)) are exactly the non-recruiting sites ($(count(==(0.0), B.rmult))); recruitment area ≤ 1")
end; end
println(NFAIL[] == 0 ? "ALL CHECKS PASS" : "$(NFAIL[]) CHECK(S) FAILED")
exit(NFAIL[])
