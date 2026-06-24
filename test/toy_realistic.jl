# A "realistic" n-D, two-objective fitness landscape — the kind a data-driven loss surface looks like:
# a smooth CONVEX macro-bowl per objective (pulling toward opposite regions → the trade-off) with
# many anisotropic, rotated, VARYING-DEPTH Gaussian wells carved into it (the CONCAVE local minima),
# plus a high-frequency ripple. Some wells are shared by both objectives (Pareto-optimal compromises),
# others are objective-specific (dominated). We test every optimizer on it, at n=2 (so the decision
# space is plottable) and n=10 (the real validation), under the same evaluation budget.
#   Run:  ./julia_gdal.sh --project=. test/toy_realistic.jl
using Pan
const CMAES=Pan.Search.CMAES; const MOCMAES=Pan.Search.MOCMAES; const IGEL=Pan.Search.IgelMOCMAES
const MOLBSA=Pan.Search.MOLBSA; const CMAMAE=Pan.Search.CMAMAE
import Random, Sobol, Statistics, LinearAlgebra
const LA = LinearAlgebra
import CairoMakie
const MK = CairoMakie
const OUTDIR = @__DIR__
const BUDGET_2D = 8000          # n=2 budget
const BUDGET_10D = 24000        # n=10 needs more evals to converge narrow basins in 10-D
const SEEDS = 1:5

# ---- the surface: convex bowls + rotated varying-depth wells + ripple --------------------------------
@inline qf(d, P) = (d' * P * d)
rand_prec(rng, n; width, cond) = (Q = Matrix(LA.qr(randn(rng, n, n)).Q); s = width .* cond .^ range(-0.5, 0.5; length=n); Q * LA.Diagonal(1.0 ./ s .^ 2) * Q')

# Returns (f1, f2, COMMON, F1ONLY, F2ONLY, r1, r2, neardist2) for dimension n.
function make_surface(n::Int; seed::Int=1)
  rng = Random.MersenneTwister(seed)
  cA = fill(0.30, n); cB = fill(0.70, n)                       # opposite convex-bowl centers → trade-off
  sep = 0.20                                                   # well separation (fits 7 wells even at n=2)
  W = Vector{Vector{Float64}}(); tries = 0
  while length(W) < 7 && (tries += 1) < 1000000
    c = 0.12 .+ 0.76 .* rand(rng, n)
    all(sum((c .- w) .^ 2) > sep^2 for w in W) && push!(W, c)
  end
  length(W) == 7 || error("make_surface: only placed $(length(W)) wells at n=$n (separation too tight)")
  COMMON = W[1:3]; F1ONLY = W[4:5]; F2ONLY = W[6:7]
  P = [rand_prec(rng, n; width=0.16, cond=3.0) for _ in W]   # rotated, moderately ill-conditioned basins
  Pc = Dict(W[i] => P[i] for i in eachindex(W))
  d1c = [1.4, 1.1, 0.8]; d2c = [0.8, 1.1, 1.4]                 # anti-correlated, varying-depth (Pareto)
  d1o = [1.2, 1.0]; d2o = [1.2, 1.0]
  bowl(x, c) = (1.5 / n) * sum((x .- c) .^ 2)                  # convex macro structure
  ripple(x) = (0.04 / n) * sum(sin(7π * xi)^2 for xi in x)
  function f1(u)
    x = clamp.(u, 0, 1)
    v = bowl(x, cA) + ripple(x)
    for (i, w) in enumerate(COMMON); v -= d1c[i] * exp(-0.5 * qf(x .- w, Pc[w])); end
    for (i, w) in enumerate(F1ONLY); v -= d1o[i] * exp(-0.5 * qf(x .- w, Pc[w])); end
    v
  end
  function f2(u)
    x = clamp.(u, 0, 1)
    v = bowl(x, cB) + ripple(x)
    for (i, w) in enumerate(COMMON); v -= d2c[i] * exp(-0.5 * qf(x .- w, Pc[w])); end
    for (i, w) in enumerate(F2ONLY); v -= d2o[i] * exp(-0.5 * qf(x .- w, Pc[w])); end
    v
  end
  r1 = maximum(f1(w) for w in COMMON) + 0.12                   # "deep in f1" at a shared well
  r2 = maximum(f2(w) for w in COMMON) + 0.12
  neardist2 = 0.05 * n                                         # decision-near tolerance (per-dim ≈0.22)
  return (f1, f2, COMMON, F1ONLY, F2ONLY, r1, r2, neardist2)
end

# count shared (Pareto-optimal) wells reached: decision-near AND deep in BOTH objectives
common_hit(pts, f1, f2, COMMON, r1, r2, nd2) =
  count(any(sum((p .- w) .^ 2) < nd2 && f1(p) < r1 && f2(p) < r2 for p in pts) for w in COMMON)
# non-dominated subset (for plotting CMA-MAE's archive as a front)
function nd_subset(pts, f1, f2)
  o = [(f1(p), f2(p)) for p in pts]; keep = trues(length(o))
  for i in eachindex(o), j in eachindex(o)
    (i != j && o[j][1] <= o[i][1] && o[j][2] <= o[i][2] && (o[j][1] < o[i][1] || o[j][2] < o[i][2])) && (keep[i] = false)
  end
  pts[keep]
end
sob(n, k) = (s = Sobol.SobolSeq(n); [Sobol.next!(s) for _ in 1:k])

# ---- methods (all N-parametric; each returns (decision-point set, restart/re-seed count)) -----------
mkmofit(f1, f2) = u -> (uu = clamp.(u, 0, 1); MOLBSA.MOFitness(Float32[f1(uu), f2(uu)], Float64(f1(uu) + f2(uu))))

function m_so(f1, f2, n, budget, sobol, seed)
  fso(u) = (uu = clamp.(u, 0, 1); f1(uu) + f2(uu)); K = 16; gens = max(1, budget ÷ (K * 12))
  rng = Random.MersenneTwister(seed); starts = sobol ? sob(n, K) : [rand(rng, n) for _ in 1:K]
  ends = Vector{Vector{Float64}}()
  for (i, s) in enumerate(starts)
    r = Random.MersenneTwister(1000 + 100seed + i); u0 = collect(Float64, s)
    st = CMAES.CMAESState(copy(u0), 0.25, CMAES.CMAESCandidate(copy(u0), fso(u0)), r; lambda=12, max_iter=10^6)
    for g in 1:gens; xs = CMAES.ask(st); fits = [fso(x) for x in xs]; CMAES.tell!(st, fits, xs); k = argmin(fits); CMAES.note_best!(st, CMAES.CMAESCandidate(copy(xs[k]), fits[k])); end
    push!(ends, clamp.(st.best.x, 0, 1))
  end
  ends, K
end
function m_moipop(f1, f2, n, budget, sobol, seed)
  mof = mkmofit(f1, f2); rng = Random.MersenneTwister(seed)
  starts = sobol ? sob(n, 60) : [rand(rng, n) for _ in 1:60]; si = Ref(1)
  nextm() = (m = si[] <= length(starts) ? collect(Float64, starts[si[]]) : rand(rng, n); si[] += 1; m)
  u0 = nextm(); st = MOCMAES.MOCMAESState(u0, 0.3, MOLBSA.MOCandidate(u0, mof(u0)), rng; max_iter=10^6, archive_cap=400)
  evals = 1; stagn = 0; nrestart = 0
  while evals < budget
    xs = CMAES.ask(st); MOCMAES.tell!(st, [mof(x) for x in xs], xs)
    for x in xs; MOCMAES.update_archive!(st, MOLBSA.MOCandidate(copy(x), mof(x))); evals += 1; end
    stagn = (st.sigma < 1e-11 || stagn >= 20) ? 0 : stagn + 1
    if st.sigma < 1e-11 || stagn >= 20
      MOCMAES.restart!(st, nextm(), 0.3; lambda=min(st.lambda * 2, 32)); stagn = 0; nrestart += 1
    end
  end
  [m.x for m in st.archive], nrestart
end
function m_igel(f1, f2, n, budget, sobol, seed; niche=0.0, reseed=0.0, mat=0)
  mof = mkmofit(f1, f2); rng = Random.MersenneTwister(seed); mu = 24; gens = max(1, budget ÷ mu)
  us = sobol ? sob(n, mu) : [rand(rng, n) for _ in 1:mu]
  st = IGEL.IgelState(us, [MOLBSA.MOCandidate(u, mof(u)) for u in us], rng; sigma0=0.25, archive_cap=400, max_iter=10^6, niche_radius=niche, reseed_sigma=reseed, maturity_period=mat)
  nreseed = 0
  for g in 1:gens; o = IGEL.ask(st); IGEL.tell!(st, [mof(x) for x in o], o); nreseed += count(st._reseed); end
  [m.x for m in st.archive], nreseed
end
# CMA-MAE: measure = (f1,f2) with bounds sampled from the surface; quality = per-dim-normalized sum
# (scale-invariant). Returns the archive elites (caller takes the non-dominated subset for plots).
function m_cmame(f1, f2, n, budget, sobol, seed; grid=20, sobreseed=false)
  rng = Random.MersenneTwister(seed)
  smp = [rand(rng, n) for _ in 1:200]; o1 = [f1(p) for p in smp]; o2 = [f2(p) for p in smp]
  lo = (minimum(o1), minimum(o2)); hi = (maximum(o1), maximum(o2))
  qual(a, b) = (a - lo[1]) / (hi[1] - lo[1] + 1e-9) + (b - lo[2]) / (hi[2] - lo[2] + 1e-9)
  mu = 12; gens = max(1, budget ÷ mu)
  u0 = sobol ? collect(Float64, sob(n, 1)[1]) : rand(rng, n)
  st = CMAMAE.CMAMAEState(u0, 0.3, rng; lambda=mu, grid_dims=(grid, grid), meas_lo=lo, meas_hi=hi,
        alpha=0.02, t0=2.0, restart_sigma=0.02, restart_patience=6, reseed_explore=1.0, sobol_reseed=sobreseed, max_iter=10^6)
  for g in 1:gens
    xs = CMAMAE.ask(st); quals = Float64[]; meas = Tuple{Float64,Float64}[]
    for x in xs; uu = clamp.(x, 0, 1); a = f1(uu); b = f2(uu); push!(quals, qual(a, b)); push!(meas, (a, b)); end
    CMAMAE.tell!(st, quals, meas, xs)
  end
  CMAMAE.elites(st), st.n_restarts
end
# LBSA / SA family: drive the ACTUAL Pan optimizers (Pan.Search.{LBSA, SA, MOLBSA}) — implementation-
# honest. Only the neighbor proposal (single-coordinate Gaussian, fixed σ — the analog of Pan's
# one-parameter mutate_params) is toy-specific; all accept/temperature/archive/restart logic is Pan's.
const LBSA = Pan.Search.LBSA; const SA = Pan.Search.SA; const MOSA = Pan.Search.MOSA
const _SA_SIGMA = 0.13
_neigh(u, rng) = (v = copy(u); j = rand(rng, 1:length(v)); v[j] = clamp(v[j] + _SA_SIGMA*randn(rng), 0, 1); v)
function m_lbsa_so(f1, f2, n, budget, sobol, seed)        # Pan LBSA (adaptive list), K chains
  fso(u) = (uu = clamp.(u, 0, 1); f1(uu) + f2(uu)); K = 16; per = max(8, budget ÷ K)
  rng = Random.MersenneTwister(seed); starts = sobol ? sob(n, K) : [rand(rng, n) for _ in 1:K]
  ends = Vector{Vector{Float64}}()
  for (i, s) in enumerate(starts)
    r = Random.MersenneTwister(2000+100seed+i); u0 = collect(Float64, s); c0 = LBSA.LBSACandidate(u0, fso(u0))
    st = LBSA.LBSAState(c0, c0, r; temp_list_len=15, stretch_len=15, max_iter=10^9)
    for _ in 1:per
      v = _neigh(st.current.x, r); LBSA.search_cmp!(LBSA.LBSACandidate(v, fso(v)), st)
      LBSA.should_restart(st) && (w = rand(r, n); LBSA.restart(st, LBSA.LBSACandidate(w, fso(w))))
    end
    push!(ends, clamp.(st.best.x, 0, 1))
  end
  ends, K
end
function m_sa_so(f1, f2, n, budget, sobol, seed)          # Pan SA (fixed geometric cooling), K chains
  fso(u) = (uu = clamp.(u, 0, 1); f1(uu) + f2(uu)); K = 16; per = max(8, budget ÷ K)
  rng = Random.MersenneTwister(seed); starts = sobol ? sob(n, K) : [rand(rng, n) for _ in 1:K]
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
function m_lbsa_mo(f1, f2, n, budget, sobol, seed)        # Pan MOLBSA (real Pareto archive + mo_delta), one chain
  mof(u) = (uu = clamp.(u, 0, 1); MOLBSA.MOFitness(Float32[f1(uu), f2(uu)], Float64(f1(uu)+f2(uu))))
  r = Random.MersenneTwister(seed); sbpts = sobol ? sob(n, 256) : Vector{Float64}[]; si = Ref(1)
  fresh() = (si[] <= length(sbpts) ? (w=collect(Float64, sbpts[si[]]); si[]+=1; w) : rand(r, n))
  u0 = fresh(); c0 = MOLBSA.MOCandidate(u0, mof(u0))
  st = MOLBSA.MOLBSAState(c0, c0, r; archive_cap=400, temp_list_len=15, stretch_len=15, max_iter=10^9)
  nrestart = 0
  for _ in 1:budget
    v = _neigh(st.current.x, r); MOLBSA.search_cmp!(MOLBSA.MOCandidate(v, mof(v)), st)
    MOLBSA.should_restart(st) && (w = fresh(); MOLBSA.restart(st, MOLBSA.MOCandidate(w, mof(w))); nrestart += 1)
  end
  [m.x for m in st.archive], nrestart
end
function m_mosa_mo(f1, f2, n, budget, sobol, seed)        # Pan MOSA (NEW: MO classic SA, fixed cooling), one chain
  mof(u) = (uu = clamp.(u, 0, 1); MOLBSA.MOFitness(Float32[f1(uu), f2(uu)], Float64(f1(uu)+f2(uu))))
  r = Random.MersenneTwister(seed); u0 = sobol ? collect(Float64, sob(n,1)[1]) : rand(r, n)
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
  ("SO (sum)",           (f1,f2,n,bd,sb,sd) -> m_so(f1,f2,n,bd,sb,sd)),
  ("MO + IPOP",          (f1,f2,n,bd,sb,sd) -> m_moipop(f1,f2,n,bd,sb,sd)),
  ("Igel",               (f1,f2,n,bd,sb,sd) -> m_igel(f1,f2,n,bd,sb,sd)),
  ("Igel + reseed+mat",  (f1,f2,n,bd,sb,sd) -> m_igel(f1,f2,n,bd,sb,sd; reseed=0.02, mat=40)),
  ("CMA-MAE (rand)",     (f1,f2,n,bd,sb,sd) -> m_cmame(f1,f2,n,bd,sb,sd; sobreseed=false)),
  ("CMA-MAE + Sobol",    (f1,f2,n,bd,sb,sd) -> m_cmame(f1,f2,n,bd,sb,sd; sobreseed=true)),
  ("SO LBSA (Pan)",      (f1,f2,n,bd,sb,sd) -> m_lbsa_so(f1,f2,n,bd,sb,sd)),
  ("SO SA (Pan)",        (f1,f2,n,bd,sb,sd) -> m_sa_so(f1,f2,n,bd,sb,sd)),
  ("MO LBSA (Pan)",      (f1,f2,n,bd,sb,sd) -> m_lbsa_mo(f1,f2,n,bd,sb,sd)),
  ("MO SA (Pan)",        (f1,f2,n,bd,sb,sd) -> m_mosa_mo(f1,f2,n,bd,sb,sd)),
]

# ===================================================================================================
# n = 2 : visualize the surface + each method's solutions in decision space, and the objective front.
# ===================================================================================================
isempty(ARGS) && println("\n========== n = 2 (visualizable) ==========")
isempty(ARGS) && let n = 2
  f1, f2, COMMON, F1ONLY, F2ONLY, r1, r2, nd2 = make_surface(n; seed=1)
  gx = range(0, 1; length=240); gy = range(0, 1; length=240)
  Zsum = [f1((x, y)) + f2((x, y)) for x in gx, y in gy]
  results = Dict{String,Vector{Vector{Float64}}}()
  println("method                | common/3 (seed-mean over $(length(SEEDS)))")
  for (name, fn) in METHODS
    hits = Int[]; pts1 = Vector{Float64}[]
    for sd in SEEDS
      pts, _ = fn(f1, f2, n, BUDGET_2D, true, sd)
      push!(hits, common_hit(pts, f1, f2, COMMON, r1, r2, nd2))
      sd == first(SEEDS) && (pts1 = pts)
    end
    results[name] = pts1
    println(rpad(name, 22) * "| " * string(round(Statistics.mean(hits), digits=2)) * "  $hits")
  end
  # decision-space panels (one per method) over the f1+f2 contour
  fig = MK.Figure(size=(1500, 1280))
  MK.Label(fig[0, 1:3], "n=2 realistic surface (convex bowls + rotated varying-depth wells): solutions on the f1+f2 landscape — ★ shared(Pareto) ▲ f1-only ▼ f2-only"; fontsize=15, font=:bold)
  for (idx, (name, _)) in enumerate(METHODS)
    r = (idx - 1) ÷ 3 + 1; c = (idx - 1) % 3 + 1
    ax = MK.Axis(fig[r, c]; title=name, aspect=1, limits=(0, 1, 0, 1))
    MK.contourf!(ax, gx, gy, Zsum; levels=22, colormap=:viridis)
    MK.scatter!(ax, [w[1] for w in COMMON], [w[2] for w in COMMON]; marker=:star5, markersize=16, color=:white, strokecolor=:black, strokewidth=1.2)
    MK.scatter!(ax, [w[1] for w in F1ONLY], [w[2] for w in F1ONLY]; marker=:utriangle, markersize=11, color=:deepskyblue, strokecolor=:black, strokewidth=1)
    MK.scatter!(ax, [w[1] for w in F2ONLY], [w[2] for w in F2ONLY]; marker=:dtriangle, markersize=11, color=:magenta, strokecolor=:black, strokewidth=1)
    pts = results[name]
    MK.scatter!(ax, [p[1] for p in pts], [p[2] for p in pts]; color=:orangered, markersize=6, strokecolor=:black, strokewidth=0.3)
  end
  MK.save(joinpath(OUTDIR, "toy_realistic_2d.png"), fig); println("wrote toy_realistic_2d.png")
  # objective space: each method's non-dominated front
  fig2 = MK.Figure(size=(760, 680))
  MK.Label(fig2[0, 1], "n=2 objective space — each method's non-dominated front (lower-left = better)"; fontsize=14, font=:bold)
  ax2 = MK.Axis(fig2[1, 1]; xlabel="f1", ylabel="f2")
  cols = MK.cgrad(:tab20, length(METHODS); categorical=true)
  MK.scatter!(ax2, [f1(w) for w in COMMON], [f2(w) for w in COMMON]; marker=:star5, markersize=22, color=(:black, 0.3), label="shared optima")
  for (idx, (name, _)) in enumerate(METHODS)
    fr = nd_subset(results[name], f1, f2)
    MK.scatter!(ax2, [f1(p) for p in fr], [f2(p) for p in fr]; color=cols[idx], markersize=7, label=name)
  end
  MK.Legend(fig2[1, 2], ax2)
  MK.save(joinpath(OUTDIR, "toy_realistic_2d_objspace.png"), fig2); println("wrote toy_realistic_2d_objspace.png")
end

# ===================================================================================================
# Validation at dimension n (default n=10; n=100 / n=500 via ARGS). Heatmap of best-aggregate-loss
# (the robust, always-differentiating high-D metric) + objective-space fronts; optional UMAP.
# ===================================================================================================
import UMAP
function run_validation(n, budget, seeds, combos, collabels, tag; do_umap=true, methods=METHODS)
  println("\n========== n = $n (validation, budget=$budget, $(length(seeds)) seeds, $(length(methods)) methods) ==========")
  f1b, f2b, COMMON, F1ONLY, F2ONLY, r1, r2, nd2 = make_surface(n; seed=1)
  bestci = min(2, length(combos))                                    # snapshot column for the plots
  H = fill(NaN, length(methods), length(combos)); umap_sets = Dict{String,Vector{Vector{Float64}}}()
  println("method                | best aggregate loss min(f1+f2)  (LOWER = better)")
  println("                      | " * join(rpad.(replace.(collabels, "\n" => "/"), 16), "| "))
  for (mi, (name, fn)) in enumerate(methods)
    row = String[]
    for (ci, (sobol, scaled)) in enumerate(combos)
      f1 = scaled ? (u -> 100 * f1b(u)) : f1b; f2 = f2b
      baggs = Float64[]; local pts1 = Vector{Float64}[]
      for sd in seeds
        pts, _ = fn(f1, f2, n, budget, sobol, sd)
        push!(baggs, isempty(pts) ? Inf : minimum(f1b(p) + f2b(p) for p in pts))    # best compromise (UNSCALED)
        (sd == first(seeds) && ci == bestci) && (pts1 = pts)
      end
      H[mi, ci] = Statistics.mean(baggs); ci == bestci && (umap_sets[name] = pts1)
      push!(row, rpad(string(round(H[mi, ci], digits=2)), 16))
    end
    println(rpad(name, 22) * "| " * join(row, "| ")); flush(stdout)
  end
  lo, hi = extrema(filter(isfinite, vec(H))); mid = (lo + hi) / 2
  fig = MK.Figure(size=(1050, 1080))
  MK.Label(fig[0, 1], "n=$n realistic surface — best aggregate loss min(f1+f2) (LOWER=better; $(length(seeds)) seeds, budget $budget)"; fontsize=13, font=:bold)
  ax = MK.Axis(fig[1, 1]; xticks=(1:length(combos), collabels), yticks=(1:length(methods), [m[1] for m in methods]), yreversed=true)
  hm = MK.heatmap!(ax, 1:length(combos), 1:length(methods), permutedims(H); colormap=MK.Reverse(:viridis), colorrange=(lo, hi))
  for mi in 1:length(methods), ci in 1:length(combos)
    MK.text!(ax, ci, mi; text=string(round(H[mi, ci], digits=2)), align=(:center, :center), color=H[mi, ci] < mid ? :black : :white, fontsize=15)
  end
  MK.Colorbar(fig[1, 2], hm; label="best aggregate loss (lower = better)")
  MK.save(joinpath(OUTDIR, "toy_realistic_$(tag).png"), fig); println("wrote toy_realistic_$(tag).png")
  figo = MK.Figure(size=(820, 680))
  MK.Label(figo[0, 1], "n=$n objective space — each method's non-dominated front (lower-left = better)"; fontsize=13, font=:bold)
  axo = MK.Axis(figo[1, 1]; xlabel="f1", ylabel="f2"); ocols = MK.cgrad(:tab20, length(methods); categorical=true)
  MK.scatter!(axo, [f1b(w) for w in COMMON], [f2b(w) for w in COMMON]; marker=:star5, markersize=20, color=(:black, 0.3), label="shared optima")
  for (idx, nm) in enumerate([m[1] for m in methods])
    fr = nd_subset(umap_sets[nm], f1b, f2b); isempty(fr) || MK.scatter!(axo, [f1b(p) for p in fr], [f2b(p) for p in fr]; color=ocols[idx], markersize=7, label=nm)
  end
  MK.Legend(figo[1, 2], axo); MK.save(joinpath(OUTDIR, "toy_realistic_$(tag)_objspace.png"), figo); println("wrote toy_realistic_$(tag)_objspace.png")
  if do_umap
    names = [m[1] for m in methods]
    allpts = vcat((umap_sets[nm] for nm in names)..., COMMON, F1ONLY, F2ONLY)
    ranges = (idx = Int[]; o = 0; for nm in names; push!(idx, o); o += length(umap_sets[nm]); end; push!(idx, o); idx)
    Random.seed!(7); emb = UMAP.fit(reduce(hcat, allpts), 2; n_neighbors=12, min_dist=0.6).embedding
    nc = ranges[end]; aC = emb[:, nc+1:nc+length(COMMON)]; aF1 = emb[:, nc+length(COMMON)+1:nc+length(COMMON)+length(F1ONLY)]; aF2 = emb[:, nc+length(COMMON)+length(F1ONLY)+1:end]
    qx = Statistics.quantile(vec(emb[1, 1:nc]), [0.01, 0.99]); qy = Statistics.quantile(vec(emb[2, 1:nc]), [0.01, 0.99])
    padx = 0.12*(qx[2]-qx[1])+1; pady = 0.12*(qy[2]-qy[1])+1; xlim = (qx[1]-padx, qx[2]+padx); ylim = (qy[1]-pady, qy[2]+pady)
    figu = MK.Figure(size=(1500, 1280))
    MK.Label(figu[0, 1:3], "n=$n solutions in a shared 2-D UMAP (★ shared · ▲ f1-only · ▼ f2-only) — coloured by aggregate f1+f2"; fontsize=15, font=:bold)
    agg(p) = f1b(p) + f2b(p); cr = let a = [agg(p) for nm in names for p in umap_sets[nm]]; (minimum(a), maximum(a)); end
    for (idx, nm) in enumerate(names)
      rr = (idx-1)÷3+1; cc = (idx-1)%3+1; ax2 = MK.Axis(figu[rr, cc]; title="$nm  (best agg $(round(H[idx,bestci],digits=2)))", limits=(xlim..., ylim...))
      MK.scatter!(ax2, emb[1, :], emb[2, :]; color=(:gray, 0.07), markersize=3)
      s = (ranges[idx]+1):ranges[idx+1]
      isempty(s) || MK.scatter!(ax2, emb[1, s], emb[2, s]; color=[agg(p) for p in umap_sets[nm]], colormap=:viridis, colorrange=cr, markersize=6)
      MK.scatter!(ax2, aC[1, :], aC[2, :]; marker=:star5, markersize=15, color=:white, strokecolor=:black, strokewidth=1.2)
      MK.scatter!(ax2, aF1[1, :], aF1[2, :]; marker=:utriangle, markersize=10, color=:deepskyblue, strokecolor=:black, strokewidth=1)
      MK.scatter!(ax2, aF2[1, :], aF2[2, :]; marker=:dtriangle, markersize=10, color=:magenta, strokecolor=:black, strokewidth=1)
    end
    MK.save(joinpath(OUTDIR, "toy_realistic_$(tag)_umap.png"), figu); println("wrote toy_realistic_$(tag)_umap.png")
  end
end

# Igel uses a full eigen per (1+1) individual per generation → O(μ·n³)/gen, so at n=500 it is given a
# modest budget; all methods share the same eval budget (fair on evaluations).
const FULL_COMBOS = [(false, false), (true, false), (false, true), (true, true)]
const FULL_LABELS = ["rand\nsame", "Sobol\nsame", "rand\nf1=100×f2", "Sobol\nf1=100×f2"]
val_dims = isempty(ARGS) ? [10] : parse.(Int, ARGS)
for n in val_dims
  if n == 10
    run_validation(10, BUDGET_10D, SEEDS, FULL_COMBOS, FULL_LABELS, "10d"; do_umap=true)
  elseif n == 100
    run_validation(100, 24000, 1:3, [(true, false), (true, true)], ["Sobol\nsame", "Sobol\nf1=100×f2"], "100d"; do_umap=false)
  elseif n == 500
    run_validation(500, 9000, 1:2, [(true, false)], ["Sobol\nsame-scale"], "500d"; do_umap=false)
  else
    run_validation(n, 20000, 1:2, [(true, false)], ["Sobol\nsame-scale"], "$(n)d"; do_umap=false)
  end
end
println("=== REALISTIC SURFACE TEST DONE ===")
