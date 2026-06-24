# Harder correctness + visualization tests for the CMA-ES engine on standard global-optimization
# benchmarks (2-D so the population is directly plottable):
#
#   • Ackley     — many local minima + a near-flat outer region, sharp global min at the origin
#   • Rastrigin  — highly multimodal "egg-carton" grid of local minima, global min at the origin
#   • Rosenbrock — non-separable curved "banana" valley, global min at (1,1) (covariance adaptation)
#
# Each is optimized with a best-of-K independent-restart strategy (the cheap analytic surrogate for
# the driver's IPOP restarts). We render the best run's population evolution on the error surface,
# overlay where every restart converged (so you can SEE some getting trapped in local minima while
# the best reaches the global one), and ASSERT the best run reaches the known global optimum.
#
# Run:  ./julia_gdal.sh --project=. test/toy_cmaes_benchmarks.jl
using Pan
const CMAES = Pan.Search.CMAES
import Random
import Sobol
import CairoMakie
const MK = CairoMakie
const OUTDIR = @__DIR__

# --- benchmark functions (minimum value 0) ---
ackley(p)    = -20exp(-0.2sqrt(0.5*(p[1]^2 + p[2]^2))) - exp(0.5*(cos(2π*p[1]) + cos(2π*p[2]))) + 20 + ℯ
rastrigin(p) = 20 + (p[1]^2 - 10cos(2π*p[1])) + (p[2]^2 - 10cos(2π*p[2]))
rosenbrock(p)= 100*(p[2] - p[1]^2)^2 + (1 - p[1])^2

# Run CMA-ES in u-space [0,1]² mapped onto the function's domain [lo,hi]²; capture the population.
function cmaes_on(f, lo, hi, u0t; sigma0=0.3, lambda=14, gens=70, seed=1)
  rng = Random.MersenneTwister(seed)
  u0 = collect(Float64, u0t)                                          # CMAESState wants a Vector
  tox(u) = (lo[1] + u[1]*(hi[1]-lo[1]), lo[2] + u[2]*(hi[2]-lo[2]))   # u-space → domain
  g(u) = f(tox((clamp(u[1],0,1), clamp(u[2],0,1))))
  best = CMAES.CMAESCandidate(copy(u0), g(u0))
  st = CMAES.CMAESState(copy(u0), sigma0, best, rng; lambda=lambda, max_iter=10^6)
  pops = Vector{Vector{NTuple{2,Float64}}}(); means = NTuple{2,Float64}[tox(st.mean)]
  for _ in 1:gens
    xs = CMAES.ask(st); fits = [g(x) for x in xs]
    CMAES.tell!(st, fits, xs)
    k = argmin(fits); CMAES.note_best!(st, CMAES.CMAESCandidate(copy(xs[k]), fits[k]))
    push!(pops, [tox(x) for x in xs]); push!(means, tox(st.mean))
  end
  return st, pops, means, tox(st.best.x)
end

# best-of-K independent restarts (different seeds); returns the best run's capture + all endpoints.
function best_of(f, lo, hi, u0; K=18, kw...)
  runs = [cmaes_on(f, lo, hi, u0; seed=s, kw...) for s in 1:K]
  ends = [r[4] for r in runs]
  fbest = [r[1].best.fx for r in runs]
  bi = argmin(fbest)
  return runs[bi], ends, fbest
end

function plot_benchmark(name, f, lo, hi, xopt, run, ends, fbest, savepath)
  st, pops, means, _ = run
  gx = range(lo[1], hi[1]; length=200); gy = range(lo[2], hi[2]; length=200)
  Z = [f((x, y)) for x in gx, y in gy]
  gens = length(pops); snaps = unique(clamp.([1, gens÷8, gens÷4, gens÷2, 3gens÷4, gens], 1, gens))
  fig = MK.Figure(size=(1500, 1020))
  MK.Label(fig[0, 1:3], "$name — CMA-ES population evolution  (global min ★ at $(xopt); best of $(length(fbest)) restarts = $(round(st.best.fx, sigdigits=3)))";
           fontsize=19, font=:bold)
  for (idx, g) in enumerate(snaps)
    r = (idx-1) ÷ 3 + 1; c = (idx-1) % 3 + 1
    ax = MK.Axis(fig[r, c]; title="generation $g", aspect=1, limits=(lo[1], hi[1], lo[2], hi[2]))
    MK.contourf!(ax, gx, gy, Z; levels=22, colormap=:viridis)
    MK.lines!(ax, [m[1] for m in means[1:g+1]], [m[2] for m in means[1:g+1]]; color=:cyan, linewidth=2.5)
    MK.scatter!(ax, [p[1] for p in pops[g]], [p[2] for p in pops[g]]; color=:orangered, markersize=8, strokecolor=:black, strokewidth=0.4)
    MK.scatter!(ax, [xopt[1]], [xopt[2]]; marker=:star5, markersize=20, color=:white, strokecolor=:black, strokewidth=1.5)
    MK.scatter!(ax, [means[g+1][1]], [means[g+1][2]]; color=:cyan, markersize=14, strokecolor=:black, strokewidth=1)
  end
  n_global = count(<(max(1e-2, minimum(fbest) + 1e-6)), fbest)
  axb = MK.Axis(fig[3, 1:3]; title="where all $(length(fbest)) restarts converged — $n_global reached the global ★ (orange = endpoints; others trapped in local minima)",
                aspect=MK.DataAspect(), limits=(lo[1], hi[1], lo[2], hi[2]), height=300)
  hm = MK.contourf!(axb, gx, gy, Z; levels=22, colormap=:viridis)
  MK.Colorbar(fig[3, 4], hm; label="$name(x,y)   (dark = low / global region)")
  MK.scatter!(axb, [e[1] for e in ends], [e[2] for e in ends]; color=:orangered, markersize=13, strokecolor=:white, strokewidth=1.2)
  MK.scatter!(axb, [xopt[1]], [xopt[2]]; marker=:star5, markersize=26, color=:white, strokecolor=:black, strokewidth=1.5)
  MK.save(savepath, fig)
  println("wrote ", savepath)
end

# ---- Ackley: domain [-5,5], global min 0 at (0,0). Start off-centre. ----
run_a, ends_a, fb_a = best_of(ackley, (-5.0,-5.0), (5.0,5.0), (0.16, 0.82); sigma0=0.32, gens=70)
@assert run_a[1].best.fx < 0.1 "Ackley: best restart did not reach the global min (got $(run_a[1].best.fx))"
println("Ackley:     best=$(round(run_a[1].best.fx, sigdigits=3))   restart spread=[$(round(minimum(fb_a),sigdigits=2)), $(round(maximum(fb_a),sigdigits=2))] — OK")
plot_benchmark("Ackley", ackley, (-5.0,-5.0), (5.0,5.0), (0.0,0.0), run_a, ends_a, fb_a, joinpath(OUTDIR, "toy_ackley.png"))

# ---- Rastrigin: domain [-5.12,5.12], global min 0 at (0,0). Hardest (many local minima). ----
run_r, ends_r, fb_r = best_of(rastrigin, (-5.12,-5.12), (5.12,5.12), (0.2, 0.78); sigma0=0.32, gens=80, K=24)
@assert run_r[1].best.fx < 1.0 "Rastrigin: best restart did not reach the global basin (got $(run_r[1].best.fx))"
println("Rastrigin:  best=$(round(run_r[1].best.fx, sigdigits=3))   ($(count(<(1.0), fb_r))/$(length(fb_r)) restarts found the global basin) — OK")
plot_benchmark("Rastrigin", rastrigin, (-5.12,-5.12), (5.12,5.12), (0.0,0.0), run_r, ends_r, fb_r, joinpath(OUTDIR, "toy_rastrigin.png"))

# ---- Rosenbrock: domain [-2,2], global min 0 at (1,1). Non-separable curved valley. ----
run_b, ends_b, fb_b = best_of(rosenbrock, (-2.0,-2.0), (2.0,2.0), (0.1, 0.9); sigma0=0.25, gens=90, K=8)
@assert run_b[1].best.fx < 1e-2 "Rosenbrock: did not converge into the valley to (1,1) (got $(run_b[1].best.fx))"
println("Rosenbrock: best=$(round(run_b[1].best.fx, sigdigits=3))   converged to (1,1) along the banana valley — OK")
plot_benchmark("Rosenbrock", rosenbrock, (-2.0,-2.0), (2.0,2.0), (1.0,1.0), run_b, ends_b, fb_b, joinpath(OUTDIR, "toy_rosenbrock.png"))

println("=== CMA-ES BENCHMARK TESTS (Ackley / Rastrigin / Rosenbrock) PASSED ===")

# =============================================================================================
# +Sobol cases: a space-filling Sobol design as the INITIAL population, then CMA-ES warm-started
# from the best Sobol point. This is the genuine source of broad space coverage — the per-gen CMA-ES
# population is otherwise a localized Gaussian around the mean. Mirrors the real pipeline option of
# seeding the initial mean from a Sobol candidate (sobol_candidates_db) — here computed in-line.
# =============================================================================================
function cmaes_sobol_on(f, lo, hi; n_sobol=96, sigma0=0.25, lambda=14, gens=70, seed=1)
  rng = Random.MersenneTwister(seed)
  tox(u) = (lo[1] + u[1]*(hi[1]-lo[1]), lo[2] + u[2]*(hi[2]-lo[2]))
  g(u) = f(tox((clamp(u[1],0,1), clamp(u[2],0,1))))
  seq = Sobol.SobolSeq(2)                                  # space-filling design over [0,1]^2
  sob_u = [Sobol.next!(seq) for _ in 1:n_sobol]
  sob_f = [g(u) for u in sob_u]
  u0 = copy(sob_u[argmin(sob_f)])                          # warm-start from the best Sobol point
  best = CMAES.CMAESCandidate(copy(u0), minimum(sob_f))
  st = CMAES.CMAESState(copy(u0), sigma0, best, rng; lambda=lambda, max_iter=10^6)
  pops = Vector{Vector{NTuple{2,Float64}}}(); means = NTuple{2,Float64}[tox(st.mean)]
  for _ in 1:gens
    xs = CMAES.ask(st); fits = [g(x) for x in xs]
    CMAES.tell!(st, fits, xs)
    k = argmin(fits); CMAES.note_best!(st, CMAES.CMAESCandidate(copy(xs[k]), fits[k]))
    push!(pops, [tox(x) for x in xs]); push!(means, tox(st.mean))
  end
  return st, [tox(u) for u in sob_u], sob_f, pops, means
end

function plot_sobol(name, f, lo, hi, xopt, st, sob_x, sob_f, pops, means, savepath)
  gx = range(lo[1], hi[1]; length=200); gy = range(lo[2], hi[2]; length=200)
  Z = [f((x, y)) for x in gx, y in gy]
  gens = length(pops); snaps = unique(clamp.([1, gens÷4, gens÷2, 3gens÷4, gens], 1, gens))
  fig = MK.Figure(size=(1500, 1020))
  MK.Label(fig[0, 1:3], "$name — Sobol warm-start: space-filling initial design → CMA-ES  (best = $(round(st.best.fx, sigdigits=3)))";
           fontsize=19, font=:bold)
  # panel 1: the Sobol design covering the whole space
  ax0 = MK.Axis(fig[1, 1]; title="initial Sobol design — $(length(sob_x)) points (space-filling)", aspect=1, limits=(lo[1], hi[1], lo[2], hi[2]))
  MK.contourf!(ax0, gx, gy, Z; levels=22, colormap=:viridis)
  MK.scatter!(ax0, [p[1] for p in sob_x], [p[2] for p in sob_x]; color=:orange, markersize=7, strokecolor=:black, strokewidth=0.4)
  bi = argmin(sob_f)
  MK.scatter!(ax0, [sob_x[bi][1]], [sob_x[bi][2]]; color=:cyan, markersize=15, strokecolor=:black, strokewidth=1)  # best Sobol → CMA start
  MK.scatter!(ax0, [xopt[1]], [xopt[2]]; marker=:star5, markersize=20, color=:white, strokecolor=:black, strokewidth=1.5)
  for (idx, gg) in enumerate(snaps)            # panels 2..6: CMA-ES generations from the warm start
    pos = idx + 1; r = (pos-1) ÷ 3 + 1; c = (pos-1) % 3 + 1
    ax = MK.Axis(fig[r, c]; title="generation $gg", aspect=1, limits=(lo[1], hi[1], lo[2], hi[2]))
    MK.contourf!(ax, gx, gy, Z; levels=22, colormap=:viridis)
    MK.lines!(ax, [m[1] for m in means[1:gg+1]], [m[2] for m in means[1:gg+1]]; color=:cyan, linewidth=2.5)
    MK.scatter!(ax, [p[1] for p in pops[gg]], [p[2] for p in pops[gg]]; color=:orangered, markersize=8, strokecolor=:black, strokewidth=0.4)
    MK.scatter!(ax, [xopt[1]], [xopt[2]]; marker=:star5, markersize=20, color=:white, strokecolor=:black, strokewidth=1.5)
    MK.scatter!(ax, [means[gg+1][1]], [means[gg+1][2]]; color=:cyan, markersize=14, strokecolor=:black, strokewidth=1)
  end
  MK.save(savepath, fig)
  println("wrote ", savepath)
end

# Shift the optimum OFF the domain centre so Sobol (whose early points include the centre) doesn't
# land on it for free — the best Sobol point only lands NEAR the global basin and CMA-ES then refines.
const SHIFT = (1.7, -2.2)
shifted(f) = p -> f((p[1] - SHIFT[1], p[2] - SHIFT[2]))
for (nm, f, lo, hi, thr) in [
    ("Ackley",    shifted(ackley),    (-5.0,-5.0),   (5.0,5.0),   0.1),
    ("Rastrigin", shifted(rastrigin), (-5.12,-5.12), (5.12,5.12), 1.0),
  ]
  st, sob_x, sob_f, pops, means = cmaes_sobol_on(f, lo, hi; n_sobol=128, sigma0=0.28, gens=70)
  @assert st.best.fx < thr "$nm Sobol warm-start did not reach the global basin (got $(st.best.fx))"
  println("$nm +Sobol: best Sobol point=$(round(minimum(sob_f),sigdigits=3)) → CMA-ES refined to best=$(round(st.best.fx,sigdigits=3)) at ≈$(SHIFT) — OK")
  plot_sobol(nm, f, lo, hi, SHIFT, st, sob_x, sob_f, pops, means, joinpath(OUTDIR, "toy_$(lowercase(nm))_sobol.png"))
end

println("=== +SOBOL WARM-START CASES PASSED ===")
