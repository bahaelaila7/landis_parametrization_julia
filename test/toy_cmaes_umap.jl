# n-DIMENSIONAL toy: CMA-ES on a 10-D multimodal function. The loss surface and the candidate
# points live in R^10 and can't be plotted directly, so we collect EVERY candidate from EVERY
# generation, build a single 2-D UMAP embedding of them all, then show the points generation by
# generation (colour = fitness). This visualizes how the population travels through the search
# space and contracts onto the optimum — in a space we otherwise can't see.
#
# Run:  ./julia_gdal.sh --project=. test/toy_cmaes_umap.jl
include(joinpath(@__DIR__, "..", "src", "search", "CMAES.jl"))   # stdlib-only module; no need to load Pan
using .CMAES
import Random, Statistics, LinearAlgebra
const LA = LinearAlgebra
import UMAP
import CairoMakie
const MK = CairoMakie
const OUTDIR = @__DIR__

# ---- 10-D ROTATED Rastrigin on [-5.12, 5.12]^n, optimized in u-space [0,1]^n ----
# Plain Rastrigin is separable (axis-aligned) → its Hessian is diagonal, so CMA-ES never needs an
# off-diagonal covariance. We apply a fixed random rotation ROT so the surface is NON-separable
# (anisotropic, correlated coordinates) — this genuinely exercises covariance adaptation. The global
# optimum stays at x = 0 (u = 0.5) since ROT·0 = 0.
const N = 10
const ROT = Matrix(LA.qr(randn(Random.MersenneTwister(7), N, N)).Q)
tox(u) = -5.12 .+ 10.24 .* u
rastrigin(x) = 10*length(x) + sum(x[i]^2 - 10cos(2π*x[i]) for i in eachindex(x))
fu(u) = rastrigin(ROT * tox(clamp.(u, 0, 1)))

# Run CMA-ES, capturing every candidate (u-vector + fitness + generation) and the per-gen mean.
function run_capture(u0; sigma0=0.30, lambda=20, gens=80, seed=1)
  rng = Random.MersenneTwister(seed)
  best = CMAES.CMAESCandidate(copy(u0), fu(u0))
  st = CMAES.CMAESState(copy(u0), sigma0, best, rng; lambda=lambda, max_iter=10^6)
  P = Vector{Vector{Float64}}(); F = Float64[]; G = Int[]; M = Vector{Vector{Float64}}()
  for g in 1:gens
    xs = CMAES.ask(st); fits = [fu(x) for x in xs]
    CMAES.tell!(st, fits, xs)
    k = argmin(fits); CMAES.note_best!(st, CMAES.CMAESCandidate(copy(xs[k]), fits[k]))
    for (x, fv) in zip(xs, fits)
      push!(P, clamp.(x, 0, 1)); push!(F, fv); push!(G, g)
    end
    push!(M, copy(st.mean))
  end
  return st, P, F, G, M
end

# best of a few restarts (10-D Rastrigin is highly multimodal); display the best run's trajectory.
u0 = fill(0.2, N)
runs = [run_capture(u0; seed=s) for s in 1:6]
bi = argmin(r[1].best.fx for r in runs)
st, P, F, G, M = runs[bi]
GENS = maximum(G)
println("10-D Rastrigin: start f=$(round(fu(u0),sigdigits=4))  →  best f=$(round(st.best.fx,sigdigits=4)) over $GENS generations, $(length(P)) candidates captured")
@assert st.best.fx < 0.6 * fu(u0) "CMA-ES did not make clear progress in 10-D (best=$(st.best.fx))"

# ---- single UMAP embedding of ALL candidates (+ the per-gen means, co-embedded) ----
allX = reduce(hcat, vcat(P, M))                 # (N features) × (n_points + n_means)
Random.seed!(42)
emb = UMAP.fit(allX, 2; n_neighbors=15, min_dist=0.25).embedding   # (2 × total)
npop = length(P)
PE = emb[:, 1:npop]                              # population points
ME = emb[:, npop+1:end]                          # per-gen means (trajectory in embedding)
xlim = (minimum(emb[1,:])-0.5, maximum(emb[1,:])+0.5)
ylim = (minimum(emb[2,:])-0.5, maximum(emb[2,:])+0.5)
logF = log10.(F .+ 1e-2)                          # fitness colour (log scale: Rastrigin spans ~0..150)
crange = (minimum(logF), maximum(logF))
println("UMAP embedding built: $(size(emb,2)) points in 2-D")

# ---- Figure 1: overview — the whole trajectory coloured by fitness, and by generation ----
let
  fig = MK.Figure(size=(1300, 600))
  MK.Label(fig[0, 1:2], "10-D rotated (non-separable) Rastrigin — UMAP of all $(npop) CMA-ES candidates (can't plot R^10 directly)"; fontsize=19, font=:bold)
  ax1 = MK.Axis(fig[1,1]; title="coloured by fitness (log₁₀)", limits=(xlim..., ylim...))
  sc = MK.scatter!(ax1, PE[1,:], PE[2,:]; color=logF, colormap=:viridis, markersize=6, colorrange=crange)
  MK.lines!(ax1, ME[1,:], ME[2,:]; color=:red, linewidth=2)                    # mean path
  MK.scatter!(ax1, [ME[1,end]], [ME[2,end]]; color=:red, marker=:star5, markersize=18, strokecolor=:white, strokewidth=1)
  MK.Colorbar(fig[1,2][1,2], sc; label="log₁₀ fitness")
  ax2 = MK.Axis(fig[1,2][1,1]; title="coloured by generation", limits=(xlim..., ylim...))
  sc2 = MK.scatter!(ax2, PE[1,:], PE[2,:]; color=G, colormap=:plasma, markersize=6)
  MK.Colorbar(fig[1,2][1,3], sc2; label="generation")
  MK.save(joinpath(OUTDIR, "toy_umap_overview.png"), fig)
  println("wrote ", joinpath(OUTDIR, "toy_umap_overview.png"))
end

# ---- Figure 2: per-generation panels in the fixed UMAP space (colour = fitness) ----
let
  snaps = unique(round.(Int, range(1, GENS; length=12)))
  fig = MK.Figure(size=(1500, 1180))
  MK.Label(fig[0, 1:4], "10-D rotated Rastrigin — population per generation in the fixed UMAP embedding (colour = log₁₀ fitness)"; fontsize=19, font=:bold)
  for (idx, g) in enumerate(snaps)
    r = (idx-1) ÷ 4 + 1; c = (idx-1) % 4 + 1
    sel = findall(==(g), G)
    bestg = minimum(@view F[sel])
    ax = MK.Axis(fig[r, c]; title="gen $g   (best f = $(round(bestg, sigdigits=3)))", limits=(xlim..., ylim...))
    MK.scatter!(ax, PE[1,:], PE[2,:]; color=(:gray, 0.12), markersize=3)        # all points, faint backdrop
    MK.scatter!(ax, PE[1,sel], PE[2,sel]; color=logF[sel], colormap=:viridis, colorrange=crange, markersize=11, strokecolor=:black, strokewidth=0.4)
    MK.lines!(ax, ME[1,1:g], ME[2,1:g]; color=:red, linewidth=1.5)              # mean path so far
    MK.scatter!(ax, [ME[1,g]], [ME[2,g]]; color=:red, marker=:star5, markersize=14, strokecolor=:white, strokewidth=1)
  end
  MK.Colorbar(fig[1:3, 5], colormap=:viridis, colorrange=crange, label="log₁₀ fitness")
  MK.save(joinpath(OUTDIR, "toy_umap_generations.png"), fig)
  println("wrote ", joinpath(OUTDIR, "toy_umap_generations.png"))
end

println("=== n-D UMAP TEST PASSED ===")
