# Comprehensive panel: which optimizer recovers the COMMON (non-dominated) wells of a 6-D jagged
# shared-well problem, across methods × {Sobol init} × {objective scaling}.
#   Methods: SO (CMA-ES on the sum) · MO+IPOP (single-distribution MOCMAES w/ restarts) ·
#            Igel · Igel+niche · Igel+reseed+maturity   (all population/Pareto methods)
#   Toggles: Sobol vs random initial points · objectives same-scale vs f1 = 1000×f2
# Score = mean (over seeds) # of the 4 common wells reached (decision-near + BOTH objectives deep).
# Run:  ./julia_gdal.sh --project=. test/toy_panel.jl
using Pan
const CMAES=Pan.Search.CMAES; const MOCMAES=Pan.Search.MOCMAES; const IGEL=Pan.Search.IgelMOCMAES; const MOLBSA=Pan.Search.MOLBSA; const CMAMAE=Pan.Search.CMAMAE
import Random, Sobol, Statistics, LinearAlgebra
const LA = LinearAlgebra
import CairoMakie
const MK = CairoMakie
const OUTDIR = @__DIR__
const N = 6
const BUDGET = 7200
const SEEDS = 1:6

# ---- fixed surface: 4 common (anti-correlated → non-dominated) + 3 f1-only + 3 f2-only wells ----
# Basins are ANISOTROPIC and randomly ROTATED (precision Σ⁻¹ = Q diag(1/s²) Qᵀ), so the locally
# optimal covariance is non-diagonal and ill-conditioned — this actually exercises CMA-ES's
# covariance adaptation (isotropic wells would make the optimum a sphere).
@inline qf(d, P) = (d' * P * d)
function rand_prec(rng; width, cond)
  Q = Matrix(LA.qr(randn(rng, N, N)).Q)
  s = width .* cond .^ range(-0.5, 0.5; length=N)     # per-axis std devs (geometric mean = width)
  Q * LA.Diagonal(1.0 ./ s.^2) * Q'
end
function mkjag(wells, depths, rng; ns=12, amp=0.05, freq=7)
  Pw = [rand_prec(rng; width=0.11, cond=6.0) for _ in wells]      # rotated, ill-conditioned global wells
  sh = Tuple{Vector{Float64},Float64,Matrix{Float64}}[]; t = 0
  while length(sh) < ns && (t += 1) < 20000
    c = 0.1 .+ 0.8 .* rand(rng, N)
    (all(sum((c.-w).^2) > 0.30^2 for w in wells) && all(sum((c.-s[1]).^2) > 0.25^2 for s in sh)) || continue
    push!(sh, (c, 0.35 + 0.15rand(rng), rand_prec(rng; width=0.06, cond=4.0)))
  end
  function f(u)
    x = clamp.(u, 0, 1)
    v = -sum(depths[i]*exp(-0.5*qf(x .- wells[i], Pw[i])) for i in eachindex(wells))
    v -= sum(s[2]*exp(-0.5*qf(x .- s[1], s[3])) for s in sh)
    v += (amp/N)*sum(sin(freq*π*xi)^2 for xi in x)
    return v
  end
  return f
end
let rng = Random.MersenneTwister(7)
  global W = Vector{Vector{Float64}}()
  while length(W) < 10; c = 0.15 .+ 0.7 .* rand(rng, N); all(sum((c.-w).^2) > 0.50^2 for w in W) && push!(W, c); end
end
const COMMON = W[1:4]; const Δ = [-0.15,-0.05,0.05,0.15]
const g1 = mkjag(vcat(COMMON, W[5:7]), vcat(1 .+ Δ, ones(3)), Random.MersenneTwister(101))   # unscaled surfaces
const g2 = mkjag(vcat(COMMON, W[8:10]), vcat(1 .- Δ, ones(3)), Random.MersenneTwister(202))
const SCALE = 1000        # different-scale objectives: scaled columns use f1 = SCALE×f2 (shared knob with toy_scales.jl / toy_cmaes_jagged_umap.jl)
sob(n) = (s = Sobol.SobolSeq(N); [Sobol.next!(s) for _ in 1:n])
# scale-aware "reached a common well": near it in decision space AND both (unscaled) objectives deep
common(pts) = count(any(sum((p .- w).^2) < 0.20^2 && g1(p) < -0.80 && g2(p) < -0.80 for p in pts) for w in COMMON)

# ---- methods: each returns a set of decision points (the solutions it found) ----
mkmofit(f1, f2) = u -> (uu = clamp.(u,0,1); MOLBSA.MOFitness(Float32[f1(uu), f2(uu)], Float64(f1(uu)+f2(uu))))
function m_so(f1, f2, sobol, seed, budget=BUDGET)
  fso(u) = (uu = clamp.(u,0,1); f1(uu) + f2(uu)); K = 16; gens = max(1, budget ÷ (K*12))
  rng = Random.MersenneTwister(seed); starts = sobol ? sob(K) : [rand(rng, N) for _ in 1:K]
  ends = Vector{Vector{Float64}}()
  for (i, s) in enumerate(starts)
    r = Random.MersenneTwister(1000 + 100seed + i); u0 = collect(Float64, s)
    st = CMAES.CMAESState(copy(u0), 0.25, CMAES.CMAESCandidate(copy(u0), fso(u0)), r; lambda=12, max_iter=10^6)
    for g in 1:gens; xs = CMAES.ask(st); fits = [fso(x) for x in xs]; CMAES.tell!(st, fits, xs); k = argmin(fits); CMAES.note_best!(st, CMAES.CMAESCandidate(copy(xs[k]), fits[k])); end
    push!(ends, clamp.(st.best.x, 0, 1))
  end
  ends, K                                         # K independent restarts
end
function m_moipop(f1, f2, sobol, seed, budget=BUDGET)
  mof = mkmofit(f1, f2); rng = Random.MersenneTwister(seed)
  starts = sobol ? sob(60) : [rand(rng, N) for _ in 1:60]; si = Ref(1)
  nextm() = (m = si[] <= length(starts) ? collect(Float64, starts[si[]]) : rand(rng, N); si[] += 1; m)
  u0 = nextm(); st = MOCMAES.MOCMAESState(u0, 0.3, MOLBSA.MOCandidate(u0, mof(u0)), rng; max_iter=10^6, archive_cap=400)
  evals = 1; stagn = 0; nrestart = 0
  while evals < budget
    xs = CMAES.ask(st); MOCMAES.tell!(st, [mof(x) for x in xs], xs); improved = false
    for x in xs; improved |= MOCMAES.update_archive!(st, MOLBSA.MOCandidate(copy(x), mof(x))); evals += 1; end
    stagn = improved ? 0 : stagn + 1
    if st.sigma < 1e-11 || stagn >= 20
      MOCMAES.restart!(st, nextm(), 0.3; lambda=min(st.lambda*2, 32)); stagn = 0; nrestart += 1
    end
  end
  [m.x for m in st.archive], nrestart            # IPOP restarts
end
function m_igel(f1, f2, sobol, seed, budget=BUDGET; niche=0.0, reseed=0.0, mat=0)
  mof = mkmofit(f1, f2); rng = Random.MersenneTwister(seed); mu = 24; gens = max(1, budget ÷ mu)
  us = sobol ? sob(mu) : [rand(rng, N) for _ in 1:mu]
  st = IGEL.IgelState(us, [MOLBSA.MOCandidate(u, mof(u)) for u in us], rng; sigma0=0.25, archive_cap=400, max_iter=10^6, niche_radius=niche, reseed_sigma=reseed, maturity_period=mat)
  nreseed = 0
  for g in 1:gens; o = IGEL.ask(st); IGEL.tell!(st, [mof(x) for x in o], o); nreseed += count(st._reseed); end
  [m.x for m in st.archive], nreseed             # total individual re-seeds over the run
end
# CMA-MAE: one archive-driven CMA-ES emitter over a 2-D MAP-Elites grid. Measure = objective vector,
# quality = aggregate — both on the UNSCALED base surfaces g1/g2 (so, like the Pareto methods, it is
# scale-invariant; the passed f1/f2 are ignored). The archive holds one elite per objective-space cell,
# so the 4 common wells (distinct cells on the anti-diagonal) are kept simultaneously, not one-per-restart.
function m_cmame(f1, f2, sobol, seed, budget=BUDGET; alpha=0.02, explore=1.0, grid=25, sobreseed=false)
  rng = Random.MersenneTwister(seed); mu = 12; gens = max(1, budget ÷ mu)
  u0 = sobol ? collect(Float64, sob(1)[1]) : rand(rng, N)
  st = CMAMAE.CMAMAEState(u0, 0.3, rng; lambda=mu, grid_dims=(grid,grid),
        meas_lo=(-1.3,-1.3), meas_hi=(0.1,0.1), alpha=alpha, t0=0.0, restart_sigma=0.02,
        restart_patience=6, reseed_explore=explore, sobol_reseed=sobreseed, max_iter=10^6)
  for g in 1:gens
    xs = CMAMAE.ask(st)
    quals = Float64[]; meas = Tuple{Float64,Float64}[]
    for x in xs; uu = clamp.(x,0,1); a = g1(uu); b = g2(uu); push!(quals, a+b); push!(meas, (a,b)); end
    CMAMAE.tell!(st, quals, meas, xs)
  end
  CMAMAE.elites(st), st.n_restarts               # archive elites + emitter re-seeds
end

# ---- LBSA / SA family: drive the ACTUAL Pan optimizers (Pan.Search.{LBSA, SA, MOLBSA}) so the test is
# IMPLEMENTATION-HONEST — all acceptance / temperature-list / Pareto-archive / restart logic is Pan's own
# `search_cmp!` / `mo_delta` / `update_archive!`. The ONLY toy-specific piece is the neighbor proposal: a
# single-coordinate Gaussian step (fixed σ) — the u-space analog of Pan's one-parameter `mutate_params`. ----
const LBSA = Pan.Search.LBSA; const SA = Pan.Search.SA
const _SA_SIGMA = 0.13
_neigh(u, rng) = (v = copy(u); j = rand(rng, 1:length(v)); v[j] = clamp(v[j] + _SA_SIGMA*randn(rng), 0, 1); v)
# SO via Pan LBSA (adaptive list-based temperature): K independent chains (like m_so) driven by
# LBSA.search_cmp! with Pan's internal restart; collect each chain's best.
function m_lbsa_so(f1, f2, sobol, seed, budget=BUDGET)
  fso(u) = (uu = clamp.(u,0,1); f1(uu) + f2(uu)); K = 16; per = max(8, budget ÷ K)
  rng = Random.MersenneTwister(seed); starts = sobol ? sob(K) : [rand(rng, N) for _ in 1:K]
  ends = Vector{Vector{Float64}}()
  for (i, s) in enumerate(starts)
    r = Random.MersenneTwister(2000+100seed+i); u0 = collect(Float64, s); c0 = LBSA.LBSACandidate(u0, fso(u0))
    st = LBSA.LBSAState(c0, c0, r; temp_list_len=15, stretch_len=15, max_iter=10^9)
    for _ in 1:per
      v = _neigh(st.current.x, r); LBSA.search_cmp!(LBSA.LBSACandidate(v, fso(v)), st)
      LBSA.should_restart(st) && (w = rand(r, N); LBSA.restart(st, LBSA.LBSACandidate(w, fso(w))))
    end
    push!(ends, clamp.(st.best.x, 0, 1))
  end
  ends, K
end
# SO via Pan SA (classic fixed geometric cooling): K chains, SA.search_cmp! + SA.search_update_rule!.
function m_sa_so(f1, f2, sobol, seed, budget=BUDGET)
  fso(u) = (uu = clamp.(u,0,1); f1(uu) + f2(uu)); K = 16; per = max(8, budget ÷ K)
  rng = Random.MersenneTwister(seed); starts = sobol ? sob(K) : [rand(rng, N) for _ in 1:K]
  ends = Vector{Vector{Float64}}()
  for (i, s) in enumerate(starts)
    r = Random.MersenneTwister(4000+100seed+i); u0 = collect(Float64, s); cur0 = fso(u0)
    ds = Float64[]; for _ in 1:min(40, per÷4); v=_neigh(u0,r); d=fso(v)-cur0; d>0 && push!(ds,d); end
    t0 = isempty(ds) ? 0.1 : Statistics.mean(ds)/(-log(0.5)); mint = t0*1e-3; al = (mint/t0)^(1/max(1,per))
    c0 = SA.SACandidate(u0, cur0); st = SA.SAState(c0, c0, r; t=t0, initial_t=t0, min_t=mint, alpha=al, max_iter=10^9)
    for it in 1:per
      st.i = it; v = _neigh(st.current.x, r); SA.search_cmp!(SA.SACandidate(v, fso(v)), st); SA.search_update_rule!(st)
    end
    push!(ends, clamp.(st.best.x, 0, 1))
  end
  ends, K
end
# MO via Pan MOLBSA (real Pareto archive + mo_delta acceptance + internal restart): one chain, full
# budget; return the archive points.
function m_lbsa_mo(f1, f2, sobol, seed, budget=BUDGET)
  mof(u) = (uu = clamp.(u,0,1); MOLBSA.MOFitness(Float32[f1(uu), f2(uu)], Float64(f1(uu)+f2(uu))))
  r = Random.MersenneTwister(seed); sbpts = sobol ? sob(256) : Vector{Float64}[]; si = Ref(1)
  fresh() = (si[] <= length(sbpts) ? (w=collect(Float64, sbpts[si[]]); si[]+=1; w) : rand(r, N))
  u0 = fresh(); c0 = MOLBSA.MOCandidate(u0, mof(u0))
  st = MOLBSA.MOLBSAState(c0, c0, r; archive_cap=400, temp_list_len=15, stretch_len=15, max_iter=10^9)
  nrestart = 0
  for _ in 1:budget
    v = _neigh(st.current.x, r); MOLBSA.search_cmp!(MOLBSA.MOCandidate(v, mof(v)), st)
    MOLBSA.should_restart(st) && (w = fresh(); MOLBSA.restart(st, MOLBSA.MOCandidate(w, mof(w))); nrestart += 1)
  end
  [m.x for m in st.archive], nrestart
end
# MO via Pan MOSA (NEW: the MO counterpart of SA — fixed geometric cooling + reheat + Pareto archive,
# mo_delta acceptance): one chain, full budget; return the archive points.
const MOSA = Pan.Search.MOSA
function m_mosa_mo(f1, f2, sobol, seed, budget=BUDGET)
  mof(u) = (uu = clamp.(u,0,1); MOLBSA.MOFitness(Float32[f1(uu), f2(uu)], Float64(f1(uu)+f2(uu))))
  r = Random.MersenneTwister(seed); u0 = sobol ? collect(Float64, sob(1)[1]) : rand(r, N)
  c0 = MOLBSA.MOCandidate(u0, mof(u0))
  ds = Float64[]; for _ in 1:min(60, budget÷4); v=_neigh(u0,r); d=MOLBSA.mo_delta(mof(v), c0.fx); d>0 && push!(ds,d); end
  t0 = isempty(ds) ? 1.0 : Statistics.mean(ds)/(-log(0.5)); mint = t0*1e-3; al = (mint/t0)^(1/max(1,budget))
  st = MOSA.MOSAState(c0, c0, r; archive_cap=400, t=t0, initial_t=t0, min_t=mint, alpha=al, max_iter=10^9)
  for it in 1:budget
    st.i = it; v = _neigh(st.current.x, r); MOSA.search_cmp!(MOLBSA.MOCandidate(v, mof(v)), st); MOSA.search_update_rule!(st)
  end
  [m.x for m in st.archive], 0
end

const METHODS = [
  ("SO (sum)",          (f1,f2,sb,sd,bd) -> m_so(f1,f2,sb,sd,bd)),
  ("MO + IPOP",         (f1,f2,sb,sd,bd) -> m_moipop(f1,f2,sb,sd,bd)),
  ("Igel",              (f1,f2,sb,sd,bd) -> m_igel(f1,f2,sb,sd,bd)),
  ("Igel + niche",      (f1,f2,sb,sd,bd) -> m_igel(f1,f2,sb,sd,bd; niche=0.25)),
  ("Igel + reseed+mat", (f1,f2,sb,sd,bd) -> m_igel(f1,f2,sb,sd,bd; reseed=0.02, mat=40)),
  ("CMA-MAE (rand)",    (f1,f2,sb,sd,bd) -> m_cmame(f1,f2,sb,sd,bd; sobreseed=false)),
  ("CMA-MAE + Sobol",   (f1,f2,sb,sd,bd) -> m_cmame(f1,f2,sb,sd,bd; sobreseed=true)),
  ("SO LBSA (Pan)",     (f1,f2,sb,sd,bd) -> m_lbsa_so(f1,f2,sb,sd,bd)),
  ("SO SA (Pan)",       (f1,f2,sb,sd,bd) -> m_sa_so(f1,f2,sb,sd,bd)),
  ("MO LBSA (Pan)",     (f1,f2,sb,sd,bd) -> m_lbsa_mo(f1,f2,sb,sd,bd)),
  ("MO SA (Pan)",       (f1,f2,sb,sd,bd) -> m_mosa_mo(f1,f2,sb,sd,bd)),
]
const COMBOS = [(false,false),(true,false),(false,true),(true,true)]   # (sobol, scaled)
collabels = ["rand\nsame-scale","Sobol\nsame-scale","rand\nf1=$(SCALE)×f2","Sobol\nf1=$(SCALE)×f2"]

M = fill(NaN, length(METHODS), length(COMBOS)); R = fill(NaN, length(METHODS), length(COMBOS))
println("COMMON WELLS REACHED / 4  (anisotropic rotated basins, $(length(SEEDS)) seeds)")
println(rpad("method",20) * "| " * join(rpad.(replace.(collabels, "\n"=>"/"), 16), "| "))
for (mi,(mname,mfn)) in enumerate(METHODS)
  row = String[]
  for (ci,(sobol,scaled)) in enumerate(COMBOS)
    f1 = scaled ? (u->SCALE*g1(u)) : g1; f2 = g2
    rr = [mfn(f1,f2,sobol,sd,BUDGET) for sd in SEEDS]    # each → (points, restart/reseed count)
    vals = [common(r[1]) for r in rr]; rests = [r[2] for r in rr]
    M[mi,ci] = Statistics.mean(vals); R[mi,ci] = Statistics.mean(rests)
    push!(row, rpad("$(round(M[mi,ci],digits=1)) $(minimum(vals))-$(maximum(vals))", 16))
  end
  println(rpad(mname,20) * "| " * join(row, "| "))
end
println("\nRESTARTS / RE-SEEDS per run (mean):  SO = #restarts · MO+IPOP = #IPOP restarts · Igel = #individual re-seeds · CMA-MAE = #emitter re-seeds")
for (mi,(mname,_)) in enumerate(METHODS)
  println(rpad(mname,20) * "| " * join([rpad(round(R[mi,ci],digits=1), 16) for ci in 1:length(COMBOS)], "| "))
end

# ---- third axis: convergence vs. number of fitness evaluations (Sobol init, both scalings) ----
const BUDGETS = [900, 1800, 3600, 7200, 14400]
SCALINGS = ((false, "same-scale"), (true, "f1 = $(SCALE)×f2"))
B = zeros(length(METHODS), length(SCALINGS), length(BUDGETS))   # mean common wells / 4
println("\nCOMMON WELLS vs. FITNESS EVALUATIONS  (Sobol init, $(length(SEEDS)) seeds)")
for (si,(scaled,stitle)) in enumerate(SCALINGS)
  f1 = scaled ? (u->SCALE*g1(u)) : g1; f2 = g2
  println("  [$stitle]   evals: " * join(rpad.(BUDGETS, 8), ""))
  for (mi,(mname,mfn)) in enumerate(METHODS)
    for (bi,bud) in enumerate(BUDGETS)
      B[mi,si,bi] = Statistics.mean([common(mfn(f1,f2,true,sd,bud)[1]) for sd in SEEDS])
    end
    println("    " * rpad(mname,18) * "| " * join([rpad(round(B[mi,si,bi],digits=1),8) for bi in 1:length(BUDGETS)], ""))
  end
end
let
  fig = MK.Figure(size=(1150, 480))
  MK.Label(fig[0,1:2], "Common (non-dominated) wells recovered vs. # fitness evaluations — Sobol init, mean of $(length(SEEDS)) seeds"; fontsize=15, font=:bold)
  colors = MK.cgrad(:tab20, length(METHODS); categorical=true)
  local ax1
  for (si,(scaled,stitle)) in enumerate(SCALINGS)
    ax = MK.Axis(fig[1,si]; xlabel="fitness evaluations (log)", ylabel="common wells reached / 4",
                 title=stitle, xscale=log10, xticks=(BUDGETS, string.(BUDGETS)))
    for mi in 1:length(METHODS)
      MK.scatterlines!(ax, BUDGETS, B[mi,si,:]; color=colors[mi], label=METHODS[mi][1], markersize=10, linewidth=2)
    end
    MK.ylims!(ax, -0.15, 4.15)
    si == 1 && (ax1 = ax)
  end
  MK.Legend(fig[1,3], ax1, "method")
  MK.save(joinpath(OUTDIR,"toy_budget.png"), fig); println("\nwrote ", joinpath(OUTDIR,"toy_budget.png"))
end

# ---- panel figure (heatmap: methods × toggle-combos, annotated with mean/4) ----
let
  fig = MK.Figure(size=(1020, 1000))
  MK.Label(fig[0,1], "Common (non-dominated) wells recovered / 4 — by method × init × objective-scaling (mean of $(length(SEEDS)) seeds)"; fontsize=15, font=:bold)
  ax = MK.Axis(fig[1,1]; xticks=(1:length(COMBOS), collabels), yticks=(1:length(METHODS), [m[1] for m in METHODS]), yreversed=true)
  hm = MK.heatmap!(ax, 1:length(COMBOS), 1:length(METHODS), permutedims(M); colormap=:viridis, colorrange=(0,4))
  for mi in 1:length(METHODS), ci in 1:length(COMBOS)
    MK.text!(ax, ci, mi; text=string(round(M[mi,ci],digits=1)), align=(:center,:center), color=M[mi,ci] > 2 ? :black : :white, fontsize=18)
  end
  MK.Colorbar(fig[1,2], hm; label="common wells reached / 4")
  MK.save(joinpath(OUTDIR,"toy_panel.png"), fig); println("\nwrote ", joinpath(OUTDIR,"toy_panel.png"))
end
println("=== COMPREHENSIVE PANEL DONE ===")
