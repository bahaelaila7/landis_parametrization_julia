# Toy correctness + visualization test for the CMA-ES / MO-CMA-ES engines.
#
# Runs both optimizers on 2-D problems (so the population is directly plottable) and renders the
# evolution of the population over generations on the error surface, then ASSERTS convergence.
#
#   1. Single-objective CMA-ES on a surface with FOUR equal global minima (min-of-wells):
#      shows one run's population cloud contracting onto a minimum, and that independent runs
#      reach the different global minima. → test/toy_cmaes.png
#   2. MO-CMA-ES on a bi-objective problem (two paraboloids) whose Pareto set is the segment
#      between the two minima: shows the archive spreading along the true Pareto set / front. →
#      test/toy_mocmaes.png
#
# Run:  ./julia_gdal.sh --project=. test/toy_cmaes_visual.jl
using Pan
const CMAES   = Pan.Search.CMAES
const MOCMAES = Pan.Search.MOCMAES
const MOLBSA  = Pan.Search.MOLBSA
import Random
import Statistics
import CairoMakie
const MK = CairoMakie

const OUTDIR = joinpath(@__DIR__)
mkpath(OUTDIR)

# ---------------------------------------------------------------------------------------------
# 1. Single-objective CMA-ES — surface with 4 equal global minima
# ---------------------------------------------------------------------------------------------
const WELLS = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
fmulti(p) = minimum(((p[1]-c[1])^2 + (p[2]-c[2])^2) for c in WELLS)   # 4 basins, global min 0 at each

# Run CMA-ES, capturing each generation's offspring cloud and mean.
function run_cmaes_capture(; seed, mean0, lambda=12, gens=25)
  rng = Random.MersenneTwister(seed)
  best = CMAES.CMAESCandidate(copy(mean0), fmulti(mean0))
  st = CMAES.CMAESState(mean0, 0.28, best, rng; lambda=lambda, max_iter=10^6)
  pops = Vector{Vector{Vector{Float64}}}()   # per-gen offspring
  means = Vector{Vector{Float64}}([copy(st.mean)])
  for _ in 1:gens
    xs = CMAES.ask(st)
    fits = [fmulti(x) for x in xs]
    CMAES.tell!(st, fits, xs)
    k = argmin(fits)
    CMAES.note_best!(st, CMAES.CMAESCandidate(copy(xs[k]), fits[k]))
    push!(pops, xs)
    push!(means, copy(st.mean))
  end
  return st, pops, means
end

# One representative run (off-centre start → decisive descent into the nearest basin).
st1, pops1, means1 = run_cmaes_capture(seed=11, mean0=[0.55, 0.62])

# Several independent runs from the centre → symmetry broken by sampling → different minima.
seeds = 1:8
finals = [(s = run_cmaes_capture(seed=sd, mean0=[0.5, 0.5])[1]; s.best.x) for sd in seeds]

# ---- assertions (correctness) ----
@assert st1.best.fx < 1e-6 "CMA-ES did not converge to a global minimum (best=$(st1.best.fx))"
nearest_well(p) = argmin([ (p[1]-c[1])^2 + (p[2]-c[2])^2 for c in WELLS ])
@assert all(fmulti(x) < 1e-4 for x in finals) "some multi-seed run failed to reach a global min"
reached = unique(nearest_well.(finals))
@assert length(reached) >= 2 "multi-seed runs should reach >=2 distinct global minima, got $(length(reached))"
println("CMA-ES: single run best=$(round(st1.best.fx, sigdigits=3)); $(length(reached))/4 distinct global minima reached across $(length(seeds)) seeds — OK")

# ---- figure ----
let
  gx = range(0, 1; length=160); gy = range(0, 1; length=160)
  Z = [fmulti((x, y)) for x in gx, y in gy]
  snaps = [1, 2, 4, 8, 16, length(pops1)]
  fig = MK.Figure(size=(1500, 1020))
  MK.Label(fig[0, 1:3], "Single-objective CMA-ES — surface with 4 equal global minima (min-of-wells)";
           fontsize=20, font=:bold)
  for (idx, g) in enumerate(snaps)
    r = (idx - 1) ÷ 3 + 1; c = (idx - 1) % 3 + 1
    ax = MK.Axis(fig[r, c]; title="generation $g   (best so far = $(round(minimum(fmulti.(reduce(vcat, pops1[1:g]))), sigdigits=3)))",
                 aspect=1, limits=(0, 1, 0, 1))
    MK.contourf!(ax, gx, gy, Z; levels=18, colormap=:viridis)   # low (minima) = dark, high = yellow
    # global minima = white stars (dark basins sit under them)
    MK.scatter!(ax, [c[1] for c in WELLS], [c[2] for c in WELLS]; marker=:star5, markersize=20,
                color=:white, strokecolor=:black, strokewidth=1.5)
    mp = means1[1:g+1]
    MK.lines!(ax, [m[1] for m in mp], [m[2] for m in mp]; color=:cyan, linewidth=2.5)
    pop = pops1[g]
    MK.scatter!(ax, [p[1] for p in pop], [p[2] for p in pop]; color=:orangered, markersize=9,
                strokecolor=:black, strokewidth=0.5)
    MK.scatter!(ax, [means1[g+1][1]], [means1[g+1][2]]; color=:cyan, markersize=15, strokecolor=:black, strokewidth=1)
  end
  # bottom: multi-seed endpoints
  axb = MK.Axis(fig[3, 1:3]; title="8 independent runs from the centre converge to the different global minima",
                aspect=MK.DataAspect(), limits=(0, 1, 0, 1), height=300)
  hm = MK.contourf!(axb, gx, gy, Z; levels=18, colormap=:viridis)
  MK.Colorbar(fig[3, 4], hm; label="error  f(x,y)  (dark = minima)")
  MK.scatter!(axb, [c[1] for c in WELLS], [c[2] for c in WELLS]; marker=:star5, markersize=24,
              color=:white, strokecolor=:black, strokewidth=1.5)
  MK.scatter!(axb, [x[1] for x in finals], [x[2] for x in finals]; color=:orangered, markersize=15,
              strokecolor=:white, strokewidth=1.5)
  MK.save(joinpath(OUTDIR, "toy_cmaes.png"), fig)
  println("wrote ", joinpath(OUTDIR, "toy_cmaes.png"))
end

# ---------------------------------------------------------------------------------------------
# 2. MO-CMA-ES — bi-objective (two paraboloids); Pareto set = segment between the two minima
# ---------------------------------------------------------------------------------------------
const A = (0.20, 0.28); const B = (0.82, 0.78)
const L = sqrt((A[1]-B[1])^2 + (A[2]-B[2])^2)     # length of the Pareto set / front intercept
f1(p) = (p[1]-A[1])^2 + (p[2]-A[2])^2
f2(p) = (p[1]-B[1])^2 + (p[2]-B[2])^2
mofit(p) = MOLBSA.MOFitness(Float32[f1(p), f2(p)], Float64(f1(p) + f2(p)))

function run_mocmaes_capture(; seed, mean0, lambda=14, gens=35, archive_cap=80)
  rng = Random.MersenneTwister(seed)
  rep = MOLBSA.MOCandidate(copy(mean0), mofit(mean0))
  st = MOCMAES.MOCMAESState(mean0, 0.30, rep, rng; lambda=lambda, max_iter=10^6, archive_cap=archive_cap)
  arch_snaps = Dict{Int,Vector{Vector{Float64}}}()   # gen → archive decision points
  for g in 1:gens
    xs = CMAES.ask(st)            # ask is shared (duck-typed); MOCMAES reuses CMAES.ask
    fxs = [mofit(x) for x in xs]
    MOCMAES.tell!(st, fxs, xs)
    for x in xs
      MOCMAES.update_archive!(st, MOLBSA.MOCandidate(copy(x), mofit(x)))
    end
    arch_snaps[g] = [copy(m.x) for m in st.archive]
  end
  return st, arch_snaps
end

stm, arch_snaps = run_mocmaes_capture(seed=7, mean0=[0.5, 0.15])

# ---- assertions (correctness) ----
arch = stm.archive
# (a) valid non-dominated set
for a in arch, b in arch
  a === b && continue
  @assert !MOLBSA.dominates(a.fx.objectives, b.fx.objectives) "MO archive has a dominated member"
end
# (b) members lie ON the true Pareto front: sqrt(f1)+sqrt(f2) ≈ L
front_resid = [abs(sqrt(m.fx.objectives[1]) + sqrt(m.fx.objectives[2]) - L) for m in arch]
@assert Statistics.median(front_resid) < 0.06 * L "archive not on the Pareto front (median resid $(Statistics.median(front_resid)))"
# (c) it SPANS the front (both single-objective extremes approached)
f1v = [m.fx.objectives[1] for m in arch]; f2v = [m.fx.objectives[2] for m in arch]
@assert (maximum(f1v) - minimum(f1v) > 0.4 * L^2) && (maximum(f2v) - minimum(f2v) > 0.4 * L^2) "archive does not span the front"
println("MO-CMA-ES: archive=$(length(arch)) non-dominated, median front residual=$(round(Statistics.median(front_resid), sigdigits=2)) (≈0 ⇒ on front), spans f1∈[$(round(minimum(f1v),sigdigits=2)),$(round(maximum(f1v),sigdigits=2))] — OK")

# ---- figure ----
let
  gx = range(0, 1; length=160); gy = range(0, 1; length=160)
  Zsum = [f1((x, y)) + f2((x, y)) for x in gx, y in gy]
  ts = range(0, 1; length=200)
  front_f1 = [(t*L)^2 for t in ts]; front_f2 = [((1-t)*L)^2 for t in ts]   # true front curve
  snaps = [3, 8, 18, 35]
  fig = MK.Figure(size=(1500, 760))
  MK.Label(fig[0, 1:4], "MO-CMA-ES — two-paraboloid bi-objective; Pareto set = segment between the minima";
           fontsize=20, font=:bold)
  for (c, g) in enumerate(snaps)
    pts = arch_snaps[g]
    # objective space (top): archive vs the true front
    axo = MK.Axis(fig[1, c]; title="gen $g — objective space", xlabel="f1", ylabel="f2", aspect=1)
    MK.lines!(axo, front_f1, front_f2; color=:black, linewidth=2, label="true front")
    MK.scatter!(axo, [f1(p) for p in pts], [f2(p) for p in pts]; color=:dodgerblue, markersize=8,
                strokecolor=:navy, strokewidth=0.4)
    c == 1 && MK.axislegend(axo; position=:rt, framevisible=false)
    # decision space (bottom): archive spreading along the segment A–B
    axd = MK.Axis(fig[2, c]; title="gen $g — decision space", aspect=1, limits=(0, 1, 0, 1))
    MK.contourf!(axd, gx, gy, Zsum; levels=16, colormap=:viridis)
    MK.lines!(axd, [A[1], B[1]], [A[2], B[2]]; color=:white, linestyle=:dash, linewidth=2)
    MK.scatter!(axd, [A[1], B[1]], [A[2], B[2]]; marker=:star5, markersize=22, color=:gold,
                strokecolor=:black, strokewidth=1)
    MK.scatter!(axd, [p[1] for p in pts], [p[2] for p in pts]; color=(:orangered, 0.9), markersize=8,
                strokecolor=:black, strokewidth=0.3)
  end
  MK.save(joinpath(OUTDIR, "toy_mocmaes.png"), fig)
  println("wrote ", joinpath(OUTDIR, "toy_mocmaes.png"))
end

println("=== TOY CMA-ES / MO-CMA-ES VISUAL TESTS PASSED ===")
