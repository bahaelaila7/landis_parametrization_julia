# Igel population-based MO-CMA-ES vs the single-distribution MOCMAES on CONTINUOUS Pareto fronts —
# the regime where Igel shines. A population of (1+1)-CMA-ES individuals specialises along the front
# and reaches BOTH extremes (the individual-objective optima), whereas one converging distribution
# clusters near the knee and under-covers the extremes.
#
# Two 2-D problems (decision space is plottable): convex front (two paraboloids) and a concave front.
# Run:  ./julia_gdal.sh --project=. test/toy_igel_mocmaes.jl
using Pan
const CMAES   = Pan.Search.CMAES
const MOCMAES = Pan.Search.MOCMAES
const IGEL    = Pan.Search.IgelMOCMAES
const MOLBSA  = Pan.Search.MOLBSA
const CMAMAE  = Pan.Search.CMAMAE
import Random, Statistics, Sobol
import CairoMakie
const MK = CairoMakie
const OUTDIR = @__DIR__

mofit(f1, f2, u) = (uu = clamp.(u, 0, 1); MOLBSA.MOFitness(Float32[f1(uu), f2(uu)], Float64(f1(uu) + f2(uu))))

# single-distribution MOCMAES (optionally a few IPOP restarts)
function run_mocmaes(f1, f2, u0; sigma0=0.3, gens=90, ipop_every=0)
  rng = Random.MersenneTwister(3); mf(u) = mofit(f1, f2, u)
  st = MOCMAES.MOCMAESState(copy(u0), sigma0, MOLBSA.MOCandidate(copy(u0), mf(u0)), rng; max_iter=10^6, archive_cap=200)
  for g in 1:gens
    xs = CMAES.ask(st); MOCMAES.tell!(st, [mf(x) for x in xs], xs)
    for x in xs; MOCMAES.update_archive!(st, MOLBSA.MOCandidate(copy(x), mf(x))); end
    if ipop_every > 0 && g % ipop_every == 0
      MOCMAES.restart!(st, rand(rng, length(u0)), sigma0)
    end
  end
  st
end

# Igel population-based MO-CMA-ES; initial population from a Sobol design (space-filling) by default.
function run_igel(f1, f2; mu=20, sigma0=0.25, gens=90, seed=5, sobol=true)
  rng = Random.MersenneTwister(seed); mf(u) = mofit(f1, f2, u)
  us = sobol ? (s = Sobol.SobolSeq(2); [Sobol.next!(s) for _ in 1:mu]) : [rand(rng, 2) for _ in 1:mu]
  st = IGEL.IgelState(us, [MOLBSA.MOCandidate(u, mf(u)) for u in us], rng; sigma0=sigma0, archive_cap=200, max_iter=10^6)
  for g in 1:gens
    offs = IGEL.ask(st); IGEL.tell!(st, [mf(x) for x in offs], offs)
  end
  st
end

# CMA-MAE: tiles the (f1,f2) measure space into cells and rewards discovering new cells → spreads the front.
function run_cmame(f1, f2, ref; seed=3)
  rng = Random.MersenneTwister(seed); u0 = [0.5, 0.5]
  st = CMAMAE.CMAMAEState(u0, 0.3, rng; lambda=12, grid_dims=(25, 25),
        meas_lo=(0.0, 0.0), meas_hi=(ref[1], ref[2]), alpha=0.02, t0=ref[1]+ref[2],
        restart_sigma=0.02, restart_patience=6, reseed_explore=0.5, max_iter=10^6)
  for g in 1:150
    xs = CMAMAE.ask(st)
    quals = Float64[]; meas = Tuple{Float64,Float64}[]
    for x in xs; uu = clamp.(x, 0, 1); a = f1(uu); b = f2(uu); push!(quals, a+b); push!(meas, (a, b)); end
    CMAMAE.tell!(st, quals, meas, xs)
  end
  CMAMAE.elites(st)
end
# non-dominated subset (minimization): CMA-MAE's MAP-Elites archive covers ALL objective cells, so we
# show its Pareto front (deep-cell elites) for a fair comparison with the MO archives.
function _nd(o)
  k = trues(length(o))
  for i in eachindex(o), j in eachindex(o)
    (i != j && o[j][1] <= o[i][1] && o[j][2] <= o[i][2] && (o[j][1] < o[i][1] || o[j][2] < o[i][2])) && (k[i] = false)
  end
  k
end

# 2-D dominated hypervolume w.r.t. reference (larger = better/wider front)
function hv2d(st, ref)
  pts = sort([(Float64(m.fx.objectives[1]), Float64(m.fx.objectives[2])) for m in st.archive])
  fr = Tuple{Float64,Float64}[]; bf2 = Inf
  for (a, b) in pts; (b < bf2 && a < ref[1] && b < ref[2]) && (push!(fr, (a, b)); bf2 = b); end
  sort!(fr; by = p -> -p[1]); hv = 0.0; prev = ref[1]
  for (a, b) in fr; hv += (prev - a) * (ref[2] - b); prev = a; end
  hv
end
f1min(st) = minimum(Float64(m.fx.objectives[1]) for m in st.archive)
f2min(st) = minimum(Float64(m.fx.objectives[2]) for m in st.archive)

function panelpair!(fig, col, name, f1, f2, st_mo, st_ig, cm_all, frontf, a, b)
  cm_el = cm_all[_nd([(f1(e), f2(e)) for e in cm_all])]    # CMA-MAE non-dominated front
  # objective space (top)
  axo = MK.Axis(fig[1, col]; title="$name — objective space", xlabel="f1", ylabel="f2", aspect=1)
  ts = range(0, 1; length=200); fr = [frontf(t) for t in ts]
  MK.lines!(axo, [p[1] for p in fr], [p[2] for p in fr]; color=:black, linewidth=2, label="true front")
  MK.scatter!(axo, [Float64(m.fx.objectives[1]) for m in st_mo.archive], [Float64(m.fx.objectives[2]) for m in st_mo.archive]; color=:dodgerblue, markersize=7, label="MOCMAES (single)")
  MK.scatter!(axo, [Float64(m.fx.objectives[1]) for m in st_ig.archive], [Float64(m.fx.objectives[2]) for m in st_ig.archive]; color=:orangered, markersize=6, label="Igel (population)")
  MK.scatter!(axo, [f1(e) for e in cm_el], [f2(e) for e in cm_el]; color=:seagreen, markersize=5, label="CMA-MAE")
  col == 1 && MK.axislegend(axo; position=:rt, framevisible=false)
  # decision space (bottom)
  axd = MK.Axis(fig[2, col]; title="$name — decision space", aspect=1, limits=(0, 1, 0, 1))
  MK.lines!(axd, [a[1], b[1]], [a[2], b[2]]; color=:gray, linestyle=:dash, linewidth=2)
  MK.scatter!(axd, [m.x[1] for m in st_mo.archive], [m.x[2] for m in st_mo.archive]; color=:dodgerblue, markersize=7)
  MK.scatter!(axd, [m.x[1] for m in st_ig.archive], [m.x[2] for m in st_ig.archive]; color=:orangered, markersize=6)
  MK.scatter!(axd, [e[1] for e in cm_el], [e[2] for e in cm_el]; color=:seagreen, markersize=5)
  MK.scatter!(axd, [a[1], b[1]], [a[2], b[2]]; marker=:star5, markersize=18, color=:gold, strokecolor=:black, strokewidth=1)
end

# ---- Problem A: convex front (two paraboloids); Pareto set = segment a..b ----
A = (0.2, 0.3); B = (0.82, 0.78); L = sqrt((A[1]-B[1])^2 + (A[2]-B[2])^2)
fa1(u) = (u[1]-A[1])^2 + (u[2]-A[2])^2
fa2(u) = (u[1]-B[1])^2 + (u[2]-B[2])^2
frontA(t) = ((t*L)^2, ((1-t)*L)^2)                       # sqrt(f1)+sqrt(f2)=L
moA = run_mocmaes(fa1, fa2, [0.5, 0.15])
igA = run_igel(fa1, fa2)
refA = (L^2*1.1, L^2*1.1)
cmA = run_cmame(fa1, fa2, refA)
println("Convex  front:  MOCMAES f1min=$(round(f1min(moA),sigdigits=2)) f2min=$(round(f2min(moA),sigdigits=2)) HV=$(round(hv2d(moA,refA),sigdigits=3)) | Igel f1min=$(round(f1min(igA),sigdigits=2)) f2min=$(round(f2min(igA),sigdigits=2)) HV=$(round(hv2d(igA,refA),sigdigits=3))")

# ---- Problem B: concave front (Schaffer-style on a line); decision = segment too ----
# f1 = d², f2 = (d - L)²  with d = signed distance along the A→B axis → convex in d, but we plot a
# concave-looking front by using sqrt objectives: g1=√f1, g2=√f2 gives a straight front; use squares
# of distance to two points with a wider spread to get a clearly bowed front.
C = (0.15, 0.8); D = (0.85, 0.2); Lc = sqrt((C[1]-D[1])^2 + (C[2]-D[2])^2)
fc1(u) = sqrt((u[1]-C[1])^2 + (u[2]-C[2])^2)             # distance (not squared) → concave front
fc2(u) = sqrt((u[1]-D[1])^2 + (u[2]-D[2])^2)
frontC(t) = (t*Lc, (1-t)*Lc)                             # f1+f2=Lc (straight in (f1,f2); a line front)
moC = run_mocmaes(fc1, fc2, [0.5, 0.5])
igC = run_igel(fc1, fc2)
refC = (Lc*1.1, Lc*1.1)
cmC = run_cmame(fc1, fc2, refC)
println("Line    front:  MOCMAES f1min=$(round(f1min(moC),sigdigits=2)) f2min=$(round(f2min(moC),sigdigits=2)) HV=$(round(hv2d(moC,refC),sigdigits=3)) | Igel f1min=$(round(f1min(igC),sigdigits=2)) f2min=$(round(f2min(igC),sigdigits=2)) HV=$(round(hv2d(igC,refC),sigdigits=3))")

# ---- assertions: Igel reaches BOTH extremes and gives a wider front than single-distribution MO ----
@assert f1min(igA) < 0.04*L^2 && f2min(igA) < 0.04*L^2 "Igel should reach both extremes (convex front)"
@assert hv2d(igA, refA) >= hv2d(moA, refA) "Igel HV should be ≥ single-distribution MO (convex)"
@assert (f1min(igA) <= f1min(moA) + 1e-9) && (f2min(igA) <= f2min(moA) + 1e-9) "Igel should cover extremes at least as well as single MO"
for st in (moA, igA, moC, igC), a in st.archive, b in st.archive
  a === b && continue; @assert !MOLBSA.dominates(a.fx.objectives, b.fx.objectives) "archive has a dominated member"
end
println("Igel reaches both extremes (convex f1min=$(round(f1min(igA),sigdigits=2)), f2min=$(round(f2min(igA),sigdigits=2))) vs single MO ($(round(f1min(moA),sigdigits=2)), $(round(f2min(moA),sigdigits=2))) — OK")
println("CMA-MAE extremes:  convex f1min=$(round(minimum(fa1(e) for e in cmA),sigdigits=2)) f2min=$(round(minimum(fa2(e) for e in cmA),sigdigits=2)) | line f1min=$(round(minimum(fc1(e) for e in cmC),sigdigits=2)) f2min=$(round(minimum(fc2(e) for e in cmC),sigdigits=2))")

# ---- figure ----
let
  fig = MK.Figure(size=(1300, 760))
  MK.Label(fig[0, 1:2], "Igel (population) vs MOCMAES (single distribution): population spreads to BOTH extremes of the front"; fontsize=17, font=:bold)
  panelpair!(fig, 1, "convex front", fa1, fa2, moA, igA, cmA, frontA, A, B)
  panelpair!(fig, 2, "line front",   fc1, fc2, moC, igC, cmC, frontC, C, D)
  MK.save(joinpath(OUTDIR, "toy_igel_vs_mocmaes.png"), fig)
  println("wrote ", joinpath(OUTDIR, "toy_igel_vs_mocmaes.png"))
end

println("=== IGEL MO-CMA-ES TOY TEST PASSED ===")
