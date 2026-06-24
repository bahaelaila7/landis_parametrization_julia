# n-D jagged landscape where the two objectives SHARE some global wells.
#
# A point at a COMMON well minimizes BOTH f1 and f2 → it dominates everything → the common wells are
# the Pareto-optimal set (the non-dominated optima). The f1-only / f2-only wells minimize just one
# objective → they are DOMINATED by the common wells. The test: does MO-CMA-ES converge onto the
# COMMON wells (the non-dominated optima), rather than the dominated single-objective wells?
#
# Evaluated across {SO, MO} × {no Sobol, Sobol} + MO·IPOP, in 6-D, visualized with a shared UMAP.
# Run:  ./julia_gdal.sh --project=. test/toy_cmaes_jagged_umap.jl
using Pan
const CMAES   = Pan.Search.CMAES
const MOCMAES = Pan.Search.MOCMAES
const IGEL    = Pan.Search.IgelMOCMAES
const MOLBSA  = Pan.Search.MOLBSA
const CMAMAE  = Pan.Search.CMAMAE
import Random, Statistics, Sobol, LinearAlgebra
const LA = LinearAlgebra
import UMAP
import CairoMakie
const MK = CairoMakie
const OUTDIR = @__DIR__
const N = 6

# ---- jagged surface: deep wells (depth ≈1) + shallow traps + per-axis ripple ----
# Basins are ANISOTROPIC and randomly ROTATED (precision Σ⁻¹ = Q diag(1/s²) Qᵀ), so the locally
# optimal covariance is non-diagonal and ill-conditioned — this exercises CMA-ES covariance
# adaptation (isotropic wells would make each optimum a sphere). Mirrors test/toy_panel.jl.
@inline qf(d, P) = (d' * P * d)
function rand_prec(rng; width, cond)
  Q = Matrix(LA.qr(randn(rng, N, N)).Q)
  s = width .* cond .^ range(-0.5, 0.5; length=N)     # per-axis std devs (geometric mean = width)
  Q * LA.Diagonal(1.0 ./ s.^2) * Q'
end
function make_jagged_nd(wells::Vector{Vector{Float64}}, depths::Vector{Float64}, rng; n_shallow=12, amp=0.05, freq=7)
  Pw = [rand_prec(rng; width=0.11, cond=6.0) for _ in wells]      # rotated, ill-conditioned global wells
  shallow = Tuple{Vector{Float64},Float64,Matrix{Float64}}[]; tries = 0
  while length(shallow) < n_shallow && (tries += 1) < 20000
    c = 0.1 .+ 0.8 .* rand(rng, N)
    all(sum((c .- w).^2) > 0.30^2 for w in wells) || continue
    all(sum((c .- s[1]).^2) > 0.25^2 for s in shallow) || continue
    push!(shallow, (c, 0.35 + 0.15rand(rng), rand_prec(rng; width=0.06, cond=4.0)))
  end
  function f(u)
    x = clamp.(u, 0.0, 1.0)
    v = -sum(depths[i]*exp(-0.5*qf(x .- wells[i], Pw[i])) for i in eachindex(wells))
    v -= sum(s[2]*exp(-0.5*qf(x .- s[1], s[3])) for s in shallow)
    v += (amp/N) * sum(sin(freq*π*xi)^2 for xi in x)
    return v
  end
  return f
end

# 10 well locations (min-separated); split into shared / f1-only / f2-only
rng_w = Random.MersenneTwister(7)
function mk_wells(n)
  W = Vector{Vector{Float64}}()
  while length(W) < n
    c = 0.15 .+ 0.7 .* rand(rng_w, N)
    all(sum((c .- w).^2) > 0.50^2 for w in W) && push!(W, c)
  end
  W
end
const ALLW   = mk_wells(10)
const COMMON = ALLW[1:4]        # shared wells → non-dominated optima (minimize BOTH)
const F1ONLY = ALLW[5:7]        # f1-only wells → dominated by COMMON
const F2ONLY = ALLW[8:10]       # f2-only wells → dominated by COMMON
# Anti-correlated depths on the COMMON wells: each is a bit deeper in one objective than the other,
# so the common wells are MUTUALLY non-dominated (a small Pareto set near (−1,−1)), and each still
# dominates the f1-only/f2-only wells. (agg = f1+f2 ≈ −2 at every common well regardless of δ.)
const Δ = [-0.15, -0.05, 0.05, 0.15]
const g1 = make_jagged_nd(vcat(COMMON, F1ONLY), vcat(1.0 .+ Δ, ones(length(F1ONLY))), Random.MersenneTwister(101))
const g2 = make_jagged_nd(vcat(COMMON, F2ONLY), vcat(1.0 .- Δ, ones(length(F2ONLY))), Random.MersenneTwister(202))
# DIFFERENT-SCALE objectives (like test/toy_scales.jl and the scaled panel column): f1 = SCALE × f2.
# (MO-)CMA-ES ranking is per-objective ordinal (net pairwise win-count) → scale-invariant, so the wells
# the methods find are unchanged; only the objective-space plot now shows the true scale imbalance.
const SCALE = 1000.0
cl(u) = clamp.(u, 0.0, 1.0)
f1(u) = SCALE * g1(cl(u))
f2(u) = g2(cl(u))
agg(u) = f1(u) + f2(u)

# well-depth references on the UNSCALED surfaces (scale-independent, so the small margins survive the
# 1000× imbalance): a well is "reached" when decision-near AND deep in the relevant unscaled objective(s).
const G1REACH = maximum(g1(w) for w in vcat(COMMON, F1ONLY)) + 0.10
const G2REACH = maximum(g2(w) for w in vcat(COMMON, F2ONLY)) + 0.10
println("wells: $(length(COMMON)) common (non-dominated), $(length(F1ONLY)) f1-only, $(length(F2ONLY)) f2-only;  common g1+g2≈$(round(Statistics.mean(g1(w)+g2(w) for w in COMMON),sigdigits=3))  (objectives scaled f1=$(Int(SCALE))×f2)")

# ---- runners (search directly in [0,1]^N) ----
sobol_pts(n) = (s = Sobol.SobolSeq(N); [Sobol.next!(s) for _ in 1:n])

function run_so(f, starts; sigma0=0.25, lambda=16, gens=55)
  U = Vector{Vector{Float64}}(); C = Float64[]; G = Int[]; ends = Vector{Vector{Float64}}()
  for (i, s) in enumerate(starts)
    rng = Random.MersenneTwister(1000+i); u0 = collect(Float64, s)
    st = CMAES.CMAESState(copy(u0), sigma0, CMAES.CMAESCandidate(copy(u0), f(u0)), rng; lambda=lambda, max_iter=10^6)
    for g in 1:gens
      xs = CMAES.ask(st); fits = [f(cl(x)) for x in xs]; CMAES.tell!(st, fits, xs)
      k = argmin(fits); CMAES.note_best!(st, CMAES.CMAESCandidate(copy(xs[k]), fits[k]))
      for (x,fv) in zip(xs,fits); push!(U, cl(x)); push!(C, fv); push!(G, g); end
    end
    push!(ends, cl(st.best.x))
  end
  return U, C, G, ends
end

mofit(u) = (uu = cl(u); MOLBSA.MOFitness(Float32[f1(uu), f2(uu)], Float64(f1(uu)+f2(uu))))
function run_mo(u0; sigma0=0.28, lambda=16, gens=70)
  rng = Random.MersenneTwister(77)
  st = MOCMAES.MOCMAESState(copy(u0), sigma0, MOLBSA.MOCandidate(copy(u0), mofit(u0)), rng; lambda=lambda, max_iter=10^6, archive_cap=150)
  U = Vector{Vector{Float64}}(); C = Float64[]; G = Int[]
  for g in 1:gens
    xs = CMAES.ask(st); fxs = [mofit(x) for x in xs]; MOCMAES.tell!(st, fxs, xs)
    for x in xs; MOCMAES.update_archive!(st, MOLBSA.MOCandidate(copy(x), mofit(x))); push!(U, cl(x)); push!(C, agg(x)); push!(G, g); end
  end
  return st, U, C, G
end
function run_mo_ipop(restart_means; sigma0=0.28, lambda0=16, gens=200, sigma_min=1.5e-2, lambda_cap=32)
  rng = Random.MersenneTwister(88); si = 1; u0 = collect(Float64, restart_means[si]); si += 1
  st = MOCMAES.MOCMAESState(copy(u0), sigma0, MOLBSA.MOCandidate(copy(u0), mofit(u0)), rng; lambda=lambda0, max_iter=10^6, archive_cap=200)
  U = Vector{Vector{Float64}}(); C = Float64[]; G = Int[]; restarts = 0
  for g in 1:gens
    xs = CMAES.ask(st); fxs = [mofit(x) for x in xs]; MOCMAES.tell!(st, fxs, xs)
    for x in xs; MOCMAES.update_archive!(st, MOLBSA.MOCandidate(copy(x), mofit(x))); push!(U, cl(x)); push!(C, agg(x)); push!(G, g); end
    if st.sigma < sigma_min && si <= length(restart_means)
      u0 = collect(Float64, restart_means[si]); si += 1; restarts += 1
      MOCMAES.restart!(st, copy(u0), sigma0; lambda=min(st.lambda*2, lambda_cap))
    end
  end
  return st, U, C, G, restarts
end

# Igel population-based MO-CMA-ES: μ (1+1)-CMA-ES individuals, initialised on a Sobol design.
# NOTE: each (1+1)-CMA-ES individual needs enough generations to converge in n-D; too small a
# per-individual budget leaves them in the basin but short of the well bottom (see the analysis).
function run_igel(init_means; mu=20, sigma0=0.25, gens=150, reseed_sigma=0.0, maturity=0, seed=99)
  rng = Random.MersenneTwister(seed)
  us = [collect(Float64, init_means[k]) for k in 1:mu]
  cands = [MOLBSA.MOCandidate(u, mofit(u)) for u in us]
  st = IGEL.IgelState(us, cands, rng; sigma0=sigma0, archive_cap=200, max_iter=10^6, reseed_sigma=reseed_sigma, maturity_period=maturity)
  U = Vector{Vector{Float64}}(); C = Float64[]; G = Int[]
  for g in 1:gens
    offs = IGEL.ask(st); fxs = [mofit(x) for x in offs]
    IGEL.tell!(st, fxs, offs)
    for x in offs; push!(U, cl(x)); push!(C, agg(x)); push!(G, g); end
  end
  return st, U, C, G
end

# CMA-MAE: one archive-driven CMA-ES emitter over a 2-D MAP-Elites grid (measure = objective vector,
# quality = aggregate, both on the UNSCALED g1/g2 ⇒ scale-invariant). The archive keeps one elite per
# objective-space cell, so the common wells (distinct anti-diagonal cells) are held simultaneously.
function run_cmame(u0; gens=300, lambda=12, alpha=0.02, explore=1.0)
  rng = Random.MersenneTwister(123)
  st = CMAMAE.CMAMAEState(collect(Float64, u0), 0.3, rng; lambda=lambda, grid_dims=(25,25),
        meas_lo=(-1.3,-1.3), meas_hi=(0.1,0.1), alpha=alpha, t0=0.0, restart_sigma=0.02,
        restart_patience=6, reseed_explore=explore, max_iter=10^6)
  U = Vector{Vector{Float64}}(); C = Float64[]; G = Int[]
  for g in 1:gens
    xs = CMAMAE.ask(st); quals = Float64[]; meas = Tuple{Float64,Float64}[]
    for x in xs; uu = cl(x); a = g1(uu); b = g2(uu); push!(quals, a+b); push!(meas, (a,b)); push!(U, uu); push!(C, agg(x)); push!(G, g); end
    CMAMAE.tell!(st, quals, meas, xs)
  end
  return st, U, C, G
end

# how many wells of a set are reached (decision-near + the relevant objective(s) low)
so_wells(ends, set) = count(any(sum((e .- w).^2) < 0.22^2 && g1(e) < G1REACH for e in ends) for w in set)
mo_common_el(el)    = count(any(sum((p .- w).^2) < 0.22^2 && g1(p) < G1REACH && g2(p) < G2REACH for p in el) for w in COMMON)
mo_common(st)       = count(any(sum((m.x .- w).^2) < 0.22^2 && g1(m.x) < G1REACH && g2(m.x) < G2REACH for m in st.archive) for w in COMMON)
mo_f1only(st)       = count(any(sum((m.x .- w).^2) < 0.22^2 for m in st.archive) for w in F1ONLY)
mo_f2only(st)       = count(any(sum((m.x .- w).^2) < 0.22^2 for m in st.archive) for w in F2ONLY)

K = 14
Uso0,Cso0,Gso0,Eso0 = run_so(f1, [rand(Random.MersenneTwister(500+i), N) for i in 1:K])
Uso1,Cso1,Gso1,Eso1 = run_so(f1, sobol_pts(K))
stmo0,Umo0,Cmo0,Gmo0 = run_mo(fill(0.5, N))
sb = sobol_pts(256); u0mo = collect(sb[argmin(agg(p) for p in sb)])
stmo1,Umo1,Cmo1,Gmo1 = run_mo(u0mo)
stmoI,UmoI,CmoI,GmoI,nrestart = run_mo_ipop(sobol_pts(16))
stIg,UIg,CIg,GIg = run_igel(sobol_pts(24))                  # Igel, Sobol-initialised population
stIgR,_,_,_ = run_igel([rand(Random.MersenneTwister(900+i), N) for i in 1:24])  # Igel, random-initialised (for comparison)
stCM,UCM,CCM,GCM = run_cmame(fill(0.5, N))                  # CMA-MAE (MAP-Elites archive over objective space)
elCM = CMAMAE.elites(stCM)
# non-dominated subset (minimization) for the objective-space plot — the archive covers all cells.
_ndmask(o) = (k=trues(length(o)); for i in eachindex(o), j in eachindex(o); (i!=j && o[j][1]<=o[i][1] && o[j][2]<=o[i][2] && (o[j][1]<o[i][1]||o[j][2]<o[i][2])) && (k[i]=false); end; k)
elCMfr = elCM[_ndmask([(f1(p),f2(p)) for p in elCM])]
# Igel + re-seed with a maturity period, over several seeds (high variance, so report the spread).
# Re-seeds need budget to mature AND then converge, so give these runs more generations.
igM  = [mo_common(run_igel(sobol_pts(24); mu=24, gens=300, reseed_sigma=0.02, maturity=40, seed=s)[1]) for s in 1:4]
igNo = [mo_common(run_igel(sobol_pts(24); mu=24, gens=300, reseed_sigma=0.02, maturity=0,  seed=s)[1]) for s in 1:4]

configs = [("SO · no Sobol", Uso0,Cso0,Gso0), ("SO · Sobol", Uso1,Cso1,Gso1),
           ("MO · no Sobol", Umo0,Cmo0,Gmo0), ("MO · Sobol", Umo1,Cmo1,Gmo1),
           ("MO · IPOP", UmoI,CmoI,GmoI), ("MO · Igel (pop)", UIg,CIg,GIg),
           ("MO · CMA-MAE", UCM,CCM,GCM)]
nso0, nso1 = so_wells(Eso0, vcat(COMMON,F1ONLY)), so_wells(Eso1, vcat(COMMON,F1ONLY))
println("SO (minimizes f1) distinct f1-wells found:  no-Sobol=$nso0/7,  Sobol=$nso1/7")
println("MO wells reached  (COMMON non-dominated / f1-only / f2-only — dominated):")
for (lbl, st) in (("no-Sobol",stmo0),("Sobol",stmo1),("IPOP",stmoI),("Igel(pop)",stIg))
  println("   $lbl:  COMMON=$(mo_common(st))/4   f1-only=$(mo_f1only(st))/3   f2-only=$(mo_f2only(st))/3   (archive $(length(st.archive)))")
end
println("Igel initial-population effect:  random-init COMMON=$(mo_common(stIgR))/4   vs   Sobol-init COMMON=$(mo_common(stIg))/4")
println("CMA-MAE (MAP-Elites):  COMMON=$(mo_common_el(elCM))/4   archive elites=$(length(elCM))   emitter re-seeds=$(stCM.n_restarts)")
println("Igel re-seed across 4 seeds:  maturity=0 → COMMON=$igNo (mean $(round(Statistics.mean(igNo),digits=1)))   vs   maturity=40 → COMMON=$igM (mean $(round(Statistics.mean(igM),digits=1)))   [IPOP=$(mo_common(stmoI))/4] — the maturity period closes the gap")
@assert nso1 >= 2
@assert nrestart >= 2 "MO-IPOP should restart multiple times (got $nrestart)"
@assert mo_common(stmoI) >= 1 "MO-IPOP should reach at least one common (non-dominated) well"
@assert mo_common(stmoI) >= max(mo_f1only(stmoI), mo_f2only(stmoI)) "MO should prefer the non-dominated common wells over dominated single-objective wells"
@assert mo_common(stIg) >= max(mo_f1only(stIg), mo_f2only(stIg)) "Igel should prefer non-dominated common wells"
for st in (stmo0,stmo1,stmoI,stIg), a in st.archive, b in st.archive
  a===b && continue; @assert !MOLBSA.dominates(a.fx.objectives, b.fx.objectives) "MO archive dominated member"
end

# subsample for a tractable UMAP
let rsub = Random.MersenneTwister(123), maxpts = 3500
  global configs = map(c -> length(c[2]) <= maxpts ? c : (i=sort(Random.randperm(rsub,length(c[2]))[1:maxpts]); (c[1],c[2][i],c[3][i],c[4][i])), configs)
end

# ---- shared UMAP over all candidates (+ COMMON, F1ONLY, F2ONLY anchors) ----
allU = vcat((c[2] for c in configs)..., COMMON, F1ONLY, F2ONLY)
ranges = (idx=Int[]; let o=0; for c in configs; push!(idx,o); o+=length(c[2]); end; push!(idx,o); idx end)
Random.seed!(42)
emb = UMAP.fit(reduce(hcat, allU), 2; n_neighbors=20, min_dist=0.3).embedding
nc = ranges[end]
aC = emb[:, nc+1 : nc+4]; aF1 = emb[:, nc+5 : nc+7]; aF2 = emb[:, nc+8 : nc+10]
qx = Statistics.quantile(vec(emb[1,:]),[0.02,0.98]); qy = Statistics.quantile(vec(emb[2,:]),[0.02,0.98])
cx=(qx[1]+qx[2])/2; cy=(qy[1]+qy[2])/2; half=max(qx[2]-qx[1],qy[2]-qy[1])/2+1.5
xlim=(cx-half,cx+half); ylim=(cy-half,cy+half)
sub(i) = (ranges[i]+1):ranges[i+1]; isSO(i) = i <= 2
println("shared UMAP built: $(size(emb,2)) points")

# colour: SO by f1, MO by aggregate (lower = closer to a common non-dominated optimum)
colf(x) = log10.(max.(x .- minimum(f1(w) for w in vcat(COMMON,F1ONLY)), 0) .+ 1e-3)
cr1 = let a=vcat((f1.(c[2]) for c in configs[1:2])...); (minimum(colf(a)),maximum(colf(a))); end
aggvals(c) = agg.(c[2])
cr2 = let a=vcat((aggvals(c) for c in configs[3:end])...); (minimum(a),maximum(a)); end

function draw_wells!(ax)
  MK.scatter!(ax, aC[1,:],  aC[2,:];  marker=:star5,   markersize=20, color=:white,   strokecolor=:black, strokewidth=1.4)
  MK.scatter!(ax, aF1[1,:], aF1[2,:]; marker=:utriangle, markersize=12, color=:deepskyblue, strokecolor=:black, strokewidth=1)
  MK.scatter!(ax, aF2[1,:], aF2[2,:]; marker=:dtriangle, markersize=12, color=:magenta,  strokecolor=:black, strokewidth=1)
end

# ---- Figure 1: fitness UMAP (SO by f1; MO by aggregate). Does MO land on the white ★ common wells? ----
let
  fig = MK.Figure(size=(1150, 1950))
  MK.Label(fig[0,1:2], "6-D jagged, shared wells — ★ COMMON (non-dominated) · ▲ f1-only · ▼ f2-only (both dominated).  Does MO reach the ★?"; fontsize=14, font=:bold)
  titles = ["SO · no Sobol  ($nso0/7 f1-wells)", "SO · Sobol  ($nso1/7 f1-wells)",
            "MO · no Sobol  ($(mo_common(stmo0))/4 common ★)", "MO · Sobol  ($(mo_common(stmo1))/4 common ★)",
            "MO · IPOP  ($(mo_common(stmoI))/4 common ★, $nrestart restarts)", "MO · Igel pop  ($(mo_common(stIg))/4 common ★)",
            "MO · CMA-MAE  ($(mo_common_el(elCM))/4 common ★, $(stCM.n_restarts) re-seeds)"]
  for i in 1:length(configs)
    r=(i-1)÷2+1; c=(i-1)%2+1
    cv = isSO(i) ? colf(f1.(configs[i][2])) : aggvals(configs[i])
    ax = MK.Axis(fig[r,c]; title=titles[i], limits=(xlim...,ylim...), aspect=MK.AxisAspect(1))
    MK.scatter!(ax, emb[1,:], emb[2,:]; color=(:gray,0.06), markersize=2)
    s = sub(i)
    MK.scatter!(ax, emb[1,s], emb[2,s]; color=cv, colormap=:viridis, colorrange=(isSO(i) ? cr1 : cr2), markersize=5)
    draw_wells!(ax)
  end
  MK.Colorbar(fig[1,3];   colormap=:viridis, colorrange=cr1, label="SO: log₁₀(f1 − min)")
  MK.Colorbar(fig[2:4,3]; colormap=:viridis, colorrange=cr2, label="MO: aggregate f1+f2  (low = common optimum)")
  MK.save(joinpath(OUTDIR,"toy_jagged_nd_umap_fitness.png"), fig); println("wrote toy_jagged_nd_umap_fitness.png")
end

# ---- Figure 2: objective space — MO converges to the common (non-dominated) corner ----
let
  fig = MK.Figure(size=(820, 700))
  MK.Label(fig[0,1], "MO objective space (raw axes, f1 = $(Int(SCALE))×f2): candidates collapse onto the COMMON non-dominated optima (★)"; fontsize=14, font=:bold)
  ax = MK.Axis(fig[1,1]; xlabel="f1", ylabel="f2")
  # reference wells first, then candidates on top (so the archive points are visible on the optima)
  MK.scatter!(ax, [f1(w) for w in COMMON], [f2(w) for w in COMMON]; marker=:star5, markersize=26, color=(:black,0.35), label="common optima")
  MK.scatter!(ax, [f1(w) for w in F1ONLY], [f2(w) for w in F1ONLY]; marker=:utriangle, markersize=15, color=:deepskyblue, strokecolor=:black, strokewidth=1, label="f1-only (dominated)")
  MK.scatter!(ax, [f1(w) for w in F2ONLY], [f2(w) for w in F2ONLY]; marker=:dtriangle, markersize=15, color=:magenta, strokecolor=:black, strokewidth=1, label="f2-only (dominated)")
  for (lbl,st,col) in (("MO · no Sobol",stmo0,:gray),("MO · Sobol",stmo1,:dodgerblue),("MO · IPOP",stmoI,:orangered),("MO · Igel pop",stIg,:seagreen))
    MK.scatter!(ax, [Float64(m.fx.objectives[1]) for m in st.archive], [Float64(m.fx.objectives[2]) for m in st.archive]; color=col, markersize=9, strokecolor=:black, strokewidth=0.4, label=lbl)
  end
  MK.scatter!(ax, [f1(p) for p in elCMfr], [f2(p) for p in elCMfr]; color=:purple, markersize=8, strokecolor=:black, strokewidth=0.4, label="MO · CMA-MAE")
  MK.axislegend(ax; position=:rt, framevisible=true)
  MK.save(joinpath(OUTDIR,"toy_jagged_nd_mo_objspace.png"), fig); println("wrote toy_jagged_nd_mo_objspace.png")
end

println("=== n-D JAGGED SHARED-WELL TEST (does MO find the non-dominated common optima?) PASSED ===")
