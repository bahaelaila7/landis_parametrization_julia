# Decision-space NICHING via the Igel population engine (toggled by `niche_radius`), in both flavors:
#   • SO niching  — 1-objective: find MULTIPLE global optima of one function simultaneously.
#   • MO niching  — multi-objective: recover decision-space-scattered non-dominated optima (the
#                   "common wells" that plain Igel collapses onto just one of).
#
# Niching = "niched domination": an individual only competes with (and can only be killed by) others
# within `niche_radius` in decision space, so explorers descending toward a different basin survive.
# 2-D so the population is plottable.  Run:  ./julia_gdal.sh --project=. test/toy_niching.jl
using Pan
const CMAES  = Pan.Search.CMAES
const IGEL   = Pan.Search.IgelMOCMAES
const MOLBSA = Pan.Search.MOLBSA
import Random, Sobol
import CairoMakie
const MK = CairoMakie
const OUTDIR = @__DIR__

sobol_pts(n) = (s = Sobol.SobolSeq(2); [Sobol.next!(s) for _ in 1:n])
dist_wells(pts, wells; r=0.07) = count(any(sum((p .- w).^2) < r^2 for p in pts) for w in wells)

# generic Igel runner (objective builder `mk` → MOFitness); returns the state
function run_igel(mk, mu, niche_radius; sigma0=0.18, gens=80)
  rng = Random.MersenneTwister(4)
  us = sobol_pts(mu)
  st = IGEL.IgelState(us, [MOLBSA.MOCandidate(u, mk(u)) for u in us], rng; sigma0=sigma0, archive_cap=300, max_iter=10^6, niche_radius=niche_radius)
  for g in 1:gens; offs = IGEL.ask(st); IGEL.tell!(st, [mk(x) for x in offs], offs); end
  st
end

# =================== SO niching: one objective, 8 equal global minima ===================
const W8 = [(0.18,0.2),(0.5,0.16),(0.82,0.22),(0.25,0.52),(0.74,0.5),(0.2,0.82),(0.55,0.84),(0.85,0.78)]
fso(u) = minimum((u[1]-w[1])^2 + (u[2]-w[2])^2 for w in W8)            # 8 global minima at 0
sofit(u) = (uu=clamp.(u,0,1); MOLBSA.MOFitness(Float32[fso(uu)], Float64(fso(uu))))
so_plain = run_igel(sofit, 32, 0.0)                                   # standard Igel (no niching)
so_niche = run_igel(sofit, 32, 0.16)                                  # niched
# multiple optima live in the POPULATION (with 1 objective the archive holds only the single best)
nso_plain = dist_wells([ind.x for ind in so_plain.pop], W8)
nso_niche = dist_wells([ind.x for ind in so_niche.pop], W8)
println("SO niching (8 equal global minima):  plain Igel population covers $nso_plain/8,  niched covers $nso_niche/8")
@assert nso_niche >= nso_plain + 2 "niching should let one SO population hold many more optima (plain=$nso_plain niched=$nso_niche)"

# =================== MO niching: shared wells (common = non-dominated) ===================
const COMMON = [(0.22,0.25),(0.4,0.6),(0.68,0.35),(0.8,0.72)]
const F1ONLY = [(0.15,0.78),(0.55,0.15)]
const F2ONLY = [(0.85,0.2),(0.3,0.45)]
Δ = [-0.15,-0.05,0.05,0.15]
mkf(wells, depths) = u -> (x=clamp.(u,0,1); -sum(depths[i]*exp(-((x[1]-wells[i][1])^2+(x[2]-wells[i][2])^2)/(2*0.09^2)) for i in eachindex(wells)))
f1 = mkf(vcat(COMMON,F1ONLY), vcat(1 .+ Δ, ones(2)))
f2 = mkf(vcat(COMMON,F2ONLY), vcat(1 .- Δ, ones(2)))
mofit(u) = (uu=clamp.(u,0,1); MOLBSA.MOFitness(Float32[f1(uu),f2(uu)], Float64(f1(uu)+f2(uu))))
agg(u) = f1(clamp.(u,0,1)) + f2(clamp.(u,0,1))
common_hit(st) = count(any(sum((m.x .- w).^2) < 0.07^2 && agg(m.x) < -1.6 for m in st.archive) for w in COMMON)
mo_plain = run_igel(mofit, 28, 0.0; sigma0=0.22)
mo_niche = run_igel(mofit, 28, 0.2; sigma0=0.22)
println("MO niching (4 common non-dominated wells):  plain Igel archive hits $(common_hit(mo_plain))/4,  niched hits $(common_hit(mo_niche))/4")
@assert common_hit(mo_niche) >= common_hit(mo_plain) "MO niching should reach ≥ as many common wells as plain Igel"

# =================== figures ===================
let
  gx=range(0,1;length=200); gy=range(0,1;length=200)
  fig = MK.Figure(size=(1300,680))
  MK.Label(fig[0,1:2], "Decision-space niching (Igel engine): SO finds many optima at once · MO recovers scattered non-dominated wells"; fontsize=15, font=:bold)
  # SO panels
  Zso=[fso((x,y)) for x in gx, y in gy]
  for (c,(lbl,st,nn)) in enumerate((("SO · plain Igel ($nso_plain/8)",so_plain,nso_plain),("SO · niched ($nso_niche/8)",so_niche,nso_niche)))
    ax=MK.Axis(fig[1,c]; title=lbl, aspect=1, limits=(0,1,0,1))
    MK.contourf!(ax,gx,gy,Zso;levels=18,colormap=:viridis)
    MK.scatter!(ax,[w[1] for w in W8],[w[2] for w in W8];marker=:star5,markersize=18,color=:white,strokecolor=:black,strokewidth=1.2)
    MK.scatter!(ax,[ind.x[1] for ind in st.pop],[ind.x[2] for ind in st.pop];color=:orangered,markersize=8,strokecolor=:black,strokewidth=0.4)
  end
  # MO panels (decision space, f1 surface + archive; common ★, singles ▲▼)
  Zf1=[f1((x,y)) for x in gx, y in gy]
  for (c,(lbl,st)) in enumerate((("MO · plain Igel ($(common_hit(mo_plain))/4 common)",mo_plain),("MO · niched ($(common_hit(mo_niche))/4 common)",mo_niche)))
    ax=MK.Axis(fig[2,c]; title=lbl, aspect=1, limits=(0,1,0,1))
    MK.contourf!(ax,gx,gy,Zf1;levels=18,colormap=:viridis)
    MK.scatter!(ax,[w[1] for w in COMMON],[w[2] for w in COMMON];marker=:star5,markersize=18,color=:white,strokecolor=:black,strokewidth=1.2)
    MK.scatter!(ax,[w[1] for w in F1ONLY],[w[2] for w in F1ONLY];marker=:utriangle,markersize=11,color=:deepskyblue,strokecolor=:black,strokewidth=1)
    MK.scatter!(ax,[w[1] for w in F2ONLY],[w[2] for w in F2ONLY];marker=:dtriangle,markersize=11,color=:magenta,strokecolor=:black,strokewidth=1)
    MK.scatter!(ax,[m.x[1] for m in st.archive],[m.x[2] for m in st.archive];color=:orangered,markersize=8,strokecolor=:black,strokewidth=0.4)
  end
  MK.save(joinpath(OUTDIR,"toy_niching.png"), fig); println("wrote ", joinpath(OUTDIR,"toy_niching.png"))
end
println("=== NICHING TOY TEST (SO + MO, toggled by niche_radius) PASSED ===")
