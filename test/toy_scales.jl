# Different-SCALE objectives — where MO must beat a scalarized SO. f1 is ~1000× the scale of f2, so
# the SO aggregate (f1+f2) is dominated by f1 and treats f2 as NOISE: SO converges to f1's optimum
# and ignores f2 entirely. MO (Pareto dominance is scale-invariant) trades the two off and covers the
# f2 dimension SO throws away. 2-D so it's plottable.  Run: ./julia_gdal.sh --project=. test/toy_scales.jl
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

const A = (0.3, 0.35); const B = (0.72, 0.68)
f1(u) = 1000.0 * ((u[1]-A[1])^2 + (u[2]-A[2])^2)        # LARGE scale (min 0 at A)
f2(u) = (u[1]-B[1])^2 + (u[2]-B[2])^2                   # small scale  (min 0 at B)
fso(u) = (uu=clamp.(u,0,1); f1(uu) + f2(uu))            # scalarized SO objective (the sum)
mofit(u) = (uu=clamp.(u,0,1); MOLBSA.MOFitness(Float32[f1(uu), f2(uu)], Float64(f1(uu)+f2(uu))))
const F2OPT = (B[1]-A[1])^2 + (B[2]-A[2])^2             # f2 value AT A (what SO is stuck with)

sob(n) = (s=Sobol.SobolSeq(2); [Sobol.next!(s) for _ in 1:n])

# single-objective CMA-ES on the SUM (best of a few restarts)
function run_so()
  best = nothing
  for sd in 1:6
    rng = Random.MersenneTwister(sd); u0 = rand(rng, 2)
    st = CMAES.CMAESState(copy(u0), 0.25, CMAES.CMAESCandidate(copy(u0), fso(u0)), rng; max_iter=10^6)
    for g in 1:80
      xs = CMAES.ask(st); fits = [fso(x) for x in xs]; CMAES.tell!(st, fits, xs)
      k = argmin(fits); CMAES.note_best!(st, CMAES.CMAESCandidate(copy(xs[k]), fits[k]))
    end
    (best === nothing || st.best.fx < best.fx) && (best = st.best)
  end
  best.x
end
function run_mocmaes()
  rng = Random.MersenneTwister(3); u0 = [0.5,0.5]
  st = MOCMAES.MOCMAESState(copy(u0), 0.3, MOLBSA.MOCandidate(copy(u0), mofit(u0)), rng; max_iter=10^6, archive_cap=200)
  for g in 1:90; xs=CMAES.ask(st); MOCMAES.tell!(st,[mofit(x) for x in xs],xs); for x in xs; MOCMAES.update_archive!(st, MOLBSA.MOCandidate(copy(x),mofit(x))); end; end
  st
end
function run_igel()
  rng = Random.MersenneTwister(5); us = sob(20)
  st = IGEL.IgelState(us, [MOLBSA.MOCandidate(u, mofit(u)) for u in us], rng; sigma0=0.22, archive_cap=200, max_iter=10^6)
  for g in 1:120; o=IGEL.ask(st); IGEL.tell!(st,[mofit(x) for x in o],o); end
  st
end
function run_cmame()
  rng = Random.MersenneTwister(7); u0 = [0.5,0.5]
  st = CMAMAE.CMAMAEState(u0, 0.3, rng; lambda=12, grid_dims=(25,25),
        meas_lo=(0.0,0.0), meas_hi=(950.0,1.0), alpha=0.02, t0=2.0, restart_sigma=0.02,
        restart_patience=6, reseed_explore=0.6, max_iter=10^6)
  for g in 1:600
    xs = CMAMAE.ask(st)
    quals = Float64[]; meas = Tuple{Float64,Float64}[]
    for x in xs; uu = clamp.(x,0,1); a = f1(uu); b = f2(uu); push!(quals, f1(uu)/1000 + f2(uu)); push!(meas, (a,b)); end
    CMAMAE.tell!(st, quals, meas, xs)
  end
  return CMAMAE.elites(st)
end
# non-dominated subset (minimization) — CMA-MAE's MAP-Elites archive covers ALL objective cells, so we
# show its Pareto front (the deep-cell elites) for a fair comparison with the MO archives/SO point.
function _nd(o)
  k = trues(length(o))
  for i in eachindex(o), j in eachindex(o)
    (i != j && o[j][1] <= o[i][1] && o[j][2] <= o[i][2] && (o[j][1] < o[i][1] || o[j][2] < o[i][2])) && (k[i] = false)
  end
  k
end

so_x = run_so()
stmo = run_mocmaes()
stig = run_igel()
cmel = run_cmame()
cmfr = cmel[_nd([(f1(e), f2(e)) for e in cmel])]    # CMA-MAE non-dominated front
so_f2 = f2(so_x)
mo_minf2 = minimum(Float64(m.fx.objectives[2]) for m in stmo.archive)
ig_minf2 = minimum(Float64(m.fx.objectives[2]) for m in stig.archive)
println("SO (minimize f1+f2):  converged to f1=$(round(f1(so_x),sigdigits=3)) (≈0, i.e. point A), but f2=$(round(so_f2,sigdigits=3)) — f2 IGNORED (its optimum is 0)")
println("MO single-dist:  archive=$(length(stmo.archive))  reaches f2 as low as $(round(mo_minf2,sigdigits=3))")
println("MO Igel pop:     archive=$(length(stig.archive))  reaches f2 as low as $(round(ig_minf2,sigdigits=3))")
println("MO CMA-MAE:       archive=$(length(cmel))  reaches f2 as low as $(round(minimum(f2(e) for e in cmel),sigdigits=3))")
@assert so_f2 > 0.5*F2OPT "SO should leave f2 near its A-value (ignored), got $so_f2 vs F2OPT=$F2OPT"
@assert min(mo_minf2, ig_minf2) < 0.15*so_f2 "MO should reach much lower f2 than SO (the dimension SO ignores)"
println("→ MO reaches f2≈$(round(min(mo_minf2,ig_minf2),sigdigits=2)) vs SO's f2≈$(round(so_f2,sigdigits=2)): MO recovers the small-scale objective SO treated as noise — OK")

# ---- figure ----
let
  gx=range(0,1;length=200); gy=range(0,1;length=200)
  fig = MK.Figure(size=(1280,620))
  MK.Label(fig[0,1:2], "Different-scale objectives (f1 = 1000×f2): the SO sum ignores f2 (→ point A); MO covers the whole trade-off"; fontsize=15, font=:bold)
  # decision space
  Zso=[log10(fso((x,y))+1) for x in gx, y in gy]
  axd=MK.Axis(fig[1,1]; title="decision space (SO sum surface)", aspect=1, limits=(0,1,0,1))
  MK.contourf!(axd,gx,gy,Zso;levels=20,colormap=:viridis)
  MK.lines!(axd,[A[1],B[1]],[A[2],B[2]];color=:white,linestyle=:dash,linewidth=2)
  MK.scatter!(axd,[m.x[1] for m in stig.archive],[m.x[2] for m in stig.archive];color=:orangered,markersize=7,label="MO archive")
  MK.scatter!(axd,[e[1] for e in cmfr],[e[2] for e in cmfr];color=:seagreen,markersize=5,label="CMA-MAE front")
  MK.scatter!(axd,[so_x[1]],[so_x[2]];color=:cyan,markersize=16,marker=:diamond,strokecolor=:black,strokewidth=1,label="SO solution")
  MK.scatter!(axd,[A[1]],[A[2]];marker=:star5,markersize=20,color=:gold,strokecolor=:black,strokewidth=1,label="f1 opt (A)")
  MK.scatter!(axd,[B[1]],[B[2]];marker=:star4,markersize=20,color=:magenta,strokecolor=:black,strokewidth=1,label="f2 opt (B)")
  MK.axislegend(axd;position=:rb,framevisible=true)
  # objective space (raw values — x and y autoscale independently to each objective's native range)
  axo=MK.Axis(fig[1,2]; title="objective space (raw values, f1 = 1000×f2)", xlabel="f1", ylabel="f2")
  MK.scatter!(axo,[Float64(m.fx.objectives[1]) for m in stmo.archive],[Float64(m.fx.objectives[2]) for m in stmo.archive];color=:dodgerblue,markersize=7,label="MO single")
  MK.scatter!(axo,[Float64(m.fx.objectives[1]) for m in stig.archive],[Float64(m.fx.objectives[2]) for m in stig.archive];color=:orangered,markersize=6,label="MO Igel")
  MK.scatter!(axo,[f1(e) for e in cmfr],[f2(e) for e in cmfr];color=:seagreen,markersize=5,label="MO CMA-MAE")
  MK.scatter!(axo,[f1(so_x)],[so_f2];color=:cyan,markersize=16,marker=:diamond,strokecolor=:black,strokewidth=1,label="SO solution")
  MK.axislegend(axo;position=:rt,framevisible=true)
  MK.save(joinpath(OUTDIR,"toy_scales.png"), fig); println("wrote ", joinpath(OUTDIR,"toy_scales.png"))
end
println("=== DIFFERENT-SCALE OBJECTIVES TOY TEST PASSED ===")
