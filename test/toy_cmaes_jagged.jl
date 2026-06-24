# Jagged / chaotic landscape with ~10 equal global minima — exercises CMA-ES and MO-CMA-ES in the
# 2×2 matrix {single-objective, multi-objective} × {without Sobol, with Sobol warm-start}.
#
# The surface = 10 deep equal global wells (f ≈ -1) + many shallow wells (local-minimum TRAPS) +
# a high-frequency ripple (jaggedness). A single CMA-ES run finds ONE minimum and can be trapped;
# a space-filling Sobol design spreads the starts so more of the 10 global minima are found. For the
# multi-objective case two such jagged surfaces are traded off and the archive captures the front.
#
# Run:  ./julia_gdal.sh --project=. test/toy_cmaes_jagged.jl
using Pan
const CMAES   = Pan.Search.CMAES
const MOCMAES = Pan.Search.MOCMAES
const MOLBSA  = Pan.Search.MOLBSA
import Random, Sobol, Statistics
import CairoMakie
const MK = CairoMakie
const OUTDIR = @__DIR__

# ---- build a jagged surface: deep global wells + shallow traps + ripple ----
function make_jagged(deep::Vector{NTuple{2,Float64}}, rng; n_shallow=15, amp=0.05, freq=9, dwidth=0.08)
  shallow = NTuple{4,Float64}[]                                  # (cx,cy,depth,width)
  tries = 0
  while length(shallow) < n_shallow && (tries += 1) < 5000
    cx, cy = 0.06 + 0.88rand(rng), 0.06 + 0.88rand(rng)
    all(((cx-d[1])^2 + (cy-d[2])^2) > 0.16^2 for d in deep) || continue        # traps off the global wells
    all(((cx-s[1])^2 + (cy-s[2])^2) > 0.12^2 for s in shallow) || continue     # and not clustered (no deeper-than-global)
    push!(shallow, (cx, cy, 0.40 + 0.18rand(rng), 0.045 + 0.015rand(rng)))     # depth ≤0.58 < global well depth 1
  end
  function f(p)
    x, y = p[1], p[2]
    v = -sum(exp(-((x-c[1])^2 + (y-c[2])^2) / (2*dwidth^2)) for c in deep)          # 10 broad global wells (depth 1)
    v -= sum(s[3]*exp(-((x-s[1])^2 + (y-s[2])^2) / (2*s[4]^2)) for s in shallow)    # narrower/shallower traps
    v += amp*(sin(freq*π*x)^2 * sin(freq*π*y)^2)                                    # jagged ripple
    return v
  end
  return f, shallow
end

# 10 deep wells on a well-separated (≥0.23) perturbed grid so each bottoms at ≈ -1 with negligible
# overlap → genuinely the 10 (roughly-equal) global minima, all clearly deeper than any trap.
const DEEP_A = [(0.15,0.16),(0.40,0.16),(0.65,0.16),(0.88,0.18),(0.15,0.50),
                (0.40,0.50),(0.65,0.50),(0.88,0.52),(0.27,0.84),(0.76,0.84)]
const DEEP_B = [(0.20,0.20),(0.50,0.16),(0.80,0.22),(0.16,0.52),(0.45,0.52),
                (0.72,0.54),(0.30,0.84),(0.58,0.86),(0.86,0.82),(0.63,0.84)]
fA, _ = make_jagged(DEEP_A, Random.MersenneTwister(101))
fB, _ = make_jagged(DEEP_B, Random.MersenneTwister(202))

# The 10 deep wells are the global minima (unambiguously the deepest basins). The ripple makes their
# depths vary slightly; a point is "in a global well" if it is below REACH (well below any shallow
# trap, which bottom out around -0.65). gmin/gmax bracket the global-well depths.
const DEEPVALS = [fA(c) for c in DEEP_A]
const GMIN = minimum(DEEPVALS); const GMAX = maximum(DEEPVALS)
const REACH = GMAX + 0.08
@assert GMAX < -0.85 "deep wells not deep enough vs shallow traps"
println("jagged surface: $(length(DEEP_A)) global minima, depth ∈ [$(round(GMIN,sigdigits=3)), $(round(GMAX,sigdigits=3))]; REACH<$(round(REACH,sigdigits=3))")

# ---- CMA-ES helpers (search directly in [0,1]^2) ----
function run_cmaes(f, u0; sigma0=0.18, lambda=14, gens=60, seed=1)
  rng = Random.MersenneTwister(seed)
  g(u) = f((clamp(u[1],0,1), clamp(u[2],0,1)))
  best = CMAES.CMAESCandidate(copy(u0), g(u0))
  st = CMAES.CMAESState(copy(u0), sigma0, best, rng; lambda=lambda, max_iter=10^6)
  pops = Vector{Vector{NTuple{2,Float64}}}(); means = NTuple{2,Float64}[Tuple(st.mean)]
  for _ in 1:gens
    xs = CMAES.ask(st); fits = [g(x) for x in xs]
    CMAES.tell!(st, fits, xs); k = argmin(fits)
    CMAES.note_best!(st, CMAES.CMAESCandidate(copy(xs[k]), fits[k]))
    push!(pops, [Tuple(x) for x in xs]); push!(means, Tuple(st.mean))
  end
  return st, pops, means, Tuple(st.best.x)
end

sobol_points(n) = (s = Sobol.SobolSeq(2); [Tuple(Sobol.next!(s)) for _ in 1:n])

# how many of the 10 global wells does a set of endpoints reach (within radius, near-global f)?
function distinct_global(f, ends, deep; r=0.09)
  found = falses(length(deep))
  for e in ends, (i, c) in enumerate(deep)
    ((e[1]-c[1])^2 + (e[2]-c[2])^2 < r^2) && f(e) < REACH && (found[i] = true)
  end
  return count(found)
end

# ---------------- SINGLE-OBJECTIVE: without vs with Sobol ----------------
K = 14
rng0 = Random.MersenneTwister(9)
rand_starts  = [(rand(rng0), rand(rng0)) for _ in 1:K]            # random uniform starts (no Sobol)
sobol_starts = sobol_points(K)                                   # space-filling Sobol starts
ends_rand  = [run_cmaes(fA, collect(s); seed=i)[4]  for (i,s) in enumerate(rand_starts)]
ends_sobol = [run_cmaes(fA, collect(s); seed=i)[4]  for (i,s) in enumerate(sobol_starts)]
nrand  = distinct_global(fA, ends_rand,  DEEP_A)
nsobol = distinct_global(fA, ends_sobol, DEEP_A)
# representative single run (no Sobol) from the centre — may get trapped in a shallow local min (the point)
runC = run_cmaes(fA, [0.5,0.5]; seed=3)
# Sobol warm-start single run (best of a space-filling design) — should reach a global well
sob = sobol_points(128); sf = [fA(p) for p in sob]; u0s = collect(sob[argmin(sf)])
runS = run_cmaes(fA, u0s; seed=3)
@assert runS[1].best.fx < REACH "Sobol warm-start did not reach a global well (got $(runS[1].best.fx))"
@assert nsobol >= 3 "Sobol starts should cover several of the 10 global minima (got $nsobol)"
trapped = runC[1].best.fx >= REACH
println("single-obj: $K starts → distinct global minima: random=$nrand/10,  Sobol=$nsobol/10;  centre run $(trapped ? "TRAPPED in a local min (f=$(round(runC[1].best.fx,sigdigits=3)))" : "reached global");  Sobol warm-start f=$(round(runS[1].best.fx,sigdigits=3))")

let
  gx = range(0,1;length=240); gy = range(0,1;length=240); Z=[fA((x,y)) for x in gx, y in gy]
  fig = MK.Figure(size=(1500, 760))
  MK.Label(fig[0,1:4], "Jagged surface, 10 equal global minima (★) — single-objective CMA-ES: without vs with Sobol";
           fontsize=19, font=:bold)
  drawsurf(ax)= (MK.contourf!(ax,gx,gy,Z;levels=26,colormap=:viridis);
                 MK.scatter!(ax,[c[1] for c in DEEP_A],[c[2] for c in DEEP_A];marker=:star5,markersize=15,color=:white,strokecolor=:black,strokewidth=1.2))
  # evolution of one no-Sobol run
  for (c,gg) in enumerate([1, 20, 60])
    ax = MK.Axis(fig[1,c]; title="no Sobol — one run, gen $gg", aspect=1, limits=(0,1,0,1)); drawsurf(ax)
    MK.lines!(ax,[m[1] for m in runC[3][1:gg+1]],[m[2] for m in runC[3][1:gg+1]];color=:cyan,linewidth=2.5)
    MK.scatter!(ax,[p[1] for p in runC[2][gg]],[p[2] for p in runC[2][gg]];color=:orangered,markersize=7,strokecolor=:black,strokewidth=0.3)
  end
  # endpoints: random vs Sobol starts
  axr = MK.Axis(fig[1,4]; title="$K random starts → $nrand/10 minima", aspect=1, limits=(0,1,0,1)); drawsurf(axr)
  MK.scatter!(axr,[s[1] for s in rand_starts],[s[2] for s in rand_starts];color=(:gray,0.7),markersize=7)
  MK.scatter!(axr,[e[1] for e in ends_rand],[e[2] for e in ends_rand];color=:orangered,markersize=12,strokecolor=:white,strokewidth=1)
  axs = MK.Axis(fig[2,4]; title="$K Sobol starts → $nsobol/10 minima", aspect=1, limits=(0,1,0,1)); drawsurf(axs)
  MK.scatter!(axs,[s[1] for s in sobol_starts],[s[2] for s in sobol_starts];color=(:gray,0.7),markersize=7)
  MK.scatter!(axs,[e[1] for e in ends_sobol],[e[2] for e in ends_sobol];color=:orangered,markersize=12,strokecolor=:white,strokewidth=1)
  # Sobol design + warm-started run evolution (sob/u0s/runS computed above)
  ax0 = MK.Axis(fig[2,1]; title="Sobol design (128 pts, space-filling)", aspect=1, limits=(0,1,0,1)); drawsurf(ax0)
  MK.scatter!(ax0,[p[1] for p in sob],[p[2] for p in sob];color=(:orange,0.8),markersize=5)
  MK.scatter!(ax0,[u0s[1]],[u0s[2]];color=:cyan,markersize=13,strokecolor=:black,strokewidth=1)
  for (c,gg) in enumerate([10, 60])
    ax = MK.Axis(fig[2,c+1]; title="Sobol warm-start, gen $gg", aspect=1, limits=(0,1,0,1)); drawsurf(ax)
    MK.lines!(ax,[m[1] for m in runS[3][1:gg+1]],[m[2] for m in runS[3][1:gg+1]];color=:cyan,linewidth=2.5)
    MK.scatter!(ax,[p[1] for p in runS[2][gg]],[p[2] for p in runS[2][gg]];color=:orangered,markersize=7,strokecolor=:black,strokewidth=0.3)
  end
  MK.save(joinpath(OUTDIR,"toy_jagged_singleobj.png"), fig)
  println("wrote ", joinpath(OUTDIR,"toy_jagged_singleobj.png"))
end

# ---------------- MULTI-OBJECTIVE: two jagged surfaces, without vs with Sobol ----------------
function run_mocmaes(f1, f2, u0; sigma0=0.22, lambda=16, gens=60, seed=1, archive_cap=120)
  rng = Random.MersenneTwister(seed)
  mof(u) = (uu=(clamp(u[1],0,1),clamp(u[2],0,1)); MOLBSA.MOFitness(Float32[f1(uu), f2(uu)], Float64(f1(uu)+f2(uu))))
  rep = MOLBSA.MOCandidate(copy(u0), mof(u0))
  st = MOCMAES.MOCMAESState(copy(u0), sigma0, rep, rng; lambda=lambda, max_iter=10^6, archive_cap=archive_cap)
  for _ in 1:gens
    xs = CMAES.ask(st); fxs = [mof(x) for x in xs]
    MOCMAES.tell!(st, fxs, xs)
    for x in xs; MOCMAES.update_archive!(st, MOLBSA.MOCandidate(copy(x), mof(x))); end
  end
  return st
end

stMO   = run_mocmaes(fA, fB, [0.5,0.5]; seed=4)
sobMO  = sobol_points(128); sfMO = [fA(p)+fB(p) for p in sobMO]
stMOs  = run_mocmaes(fA, fB, collect(sobMO[argmin(sfMO)]); seed=4)
for (lbl, st) in (("no-Sobol", stMO), ("Sobol", stMOs))
  for a in st.archive, b in st.archive
    a===b && continue
    @assert !MOLBSA.dominates(a.fx.objectives, b.fx.objectives) "MO archive ($lbl) has a dominated member"
  end
end
println("multi-obj: archive sizes  no-Sobol=$(length(stMO.archive))  Sobol=$(length(stMOs.archive))  (both non-dominated)")

let
  gx = range(0,1;length=200); gy = range(0,1;length=200)
  ZA=[fA((x,y)) for x in gx, y in gy]
  fig = MK.Figure(size=(1300, 760))
  MK.Label(fig[0,1:2], "Two jagged objectives — MO-CMA-ES archive (Pareto front): without vs with Sobol";
           fontsize=19, font=:bold)
  for (row, lbl, st) in ((1,"no Sobol",stMO),(2,"Sobol",stMOs))
    arch = st.archive
    axd = MK.Axis(fig[row,1]; title="$lbl — decision space (f1 surface + archive)", aspect=1, limits=(0,1,0,1))
    MK.contourf!(axd,gx,gy,ZA;levels=22,colormap=:viridis)
    MK.scatter!(axd,[m.x[1] for m in arch],[m.x[2] for m in arch];color=:orangered,markersize=7,strokecolor=:black,strokewidth=0.3)
    axo = MK.Axis(fig[row,2]; title="$lbl — objective space  (archive = Pareto front, n=$(length(arch)))", xlabel="f1", ylabel="f2", aspect=1)
    MK.scatter!(axo,[m.fx.objectives[1] for m in arch],[m.fx.objectives[2] for m in arch];color=:dodgerblue,markersize=7,strokecolor=:navy,strokewidth=0.3)
  end
  MK.save(joinpath(OUTDIR,"toy_jagged_multiobj.png"), fig)
  println("wrote ", joinpath(OUTDIR,"toy_jagged_multiobj.png"))
end

println("=== JAGGED-SURFACE TESTS (SO/MO × with/without Sobol) PASSED ===")
