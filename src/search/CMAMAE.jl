module CMAMAE
import Random
import Sobol
import ..CMAES
import ..MOLBSA
import ..MOCMAES: mo_sortperm

export CMAMAEState, CMAMAEMOState, ask, tell!, tell_mo!, elites, elite_count, measure

# CMA-MAE — Covariance Matrix Adaptation MAP-Elites Annealing (Fontaine & Nikolaidis 2022),
# for MINIMIZATION, reusing the CMAES engine as the emitter.
#
# Motivation: a single MO-CMA-ES distribution adapts its covariance to ONE basin per restart, so the
# covariance work is "thrown away" each restart. CMA-MAE keeps a MAP-Elites archive over a behavior
# (measure) space and drives a CMA-ES emitter by *archive improvement* rather than raw fitness:
#   • quality  — a scalar to minimize (here the objective aggregate);
#   • measure  — a behavior descriptor (here the objective vector), tiled into a grid of cells.
# Each cell keeps its best solution (elite) and a soft acceptance THRESHOLD t_e. Offspring are ranked
# by improvement Δ = t_e − quality (minimization: positive ⇒ better than the cell's threshold), and
# the CMA-ES distribution is updated with that ranking. Thresholds relax toward accepted solutions at
# rate α (the archive learning rate): α→0 ≈ plain CMA-ES descent that merely logs an archive; α→1 =
# classic CMA-ME (only strict per-cell improvements drive the emitter). On convergence the emitter
# re-seeds from a random elite, so covariance adaptation now serves COVERAGE of the whole measure
# space — undiscovered cells (t_e = t0) give the largest Δ, pulling the emitter onto new basins.
mutable struct CMAMAEState{TRNG<:Random.AbstractRNG}
  emitter::CMAES.CMAESState                 # the CMA-ES distribution (reused engine)
  rng::TRNG
  # ---- MAP-Elites archive (flat, column-major over `dims`) ----
  dims::NTuple{2,Int}
  lo::NTuple{2,Float64}
  hi::NTuple{2,Float64}
  elite_x::Vector{Union{Nothing,Vector{Float64}}}   # best u-vector per cell
  elite_f::Vector{Float64}                          # its quality (lower = better)
  threshold::Vector{Float64}                        # soft acceptance threshold t_e per cell
  alpha::Float64                                     # archive learning rate
  t0::Float64                                        # initial threshold (ceiling for minimization)
  # ---- emitter management ----
  sigma0::Float64
  restart_sigma::Float64
  restart_patience::Int                              # restart if no archive addition for this many gens
  pending_restart::Bool
  stagnation::Int
  n_restarts::Int
  i::Int
  max_iter::Int
  reseed_explore::Float64                             # fraction of re-seeds from a fresh point (vs elite)
  sobol_reseed::Bool                                  # draw exploratory re-seeds from a Sobol design
  _sobol::Sobol.SobolSeq                              # space-filling source for re-seed locations
end

# mean0/sigma0 are in u-space ([0,1]^d). meas_lo/meas_hi bound the 2-D measure (behavior) space.
function CMAMAEState(mean0::Vector{Float64}, sigma0::Float64, rng::Random.AbstractRNG;
                     lambda::Int=12, grid_dims::NTuple{2,Int}=(25, 25),
                     meas_lo::NTuple{2,Real}, meas_hi::NTuple{2,Real},
                     alpha::Float64=0.02, t0::Float64=0.0, restart_sigma::Float64=0.02,
                     restart_patience::Int=6, reseed_explore::Float64=0.5, sobol_reseed::Bool=false,
                     blocks::Union{Nothing,Vector{Vector{Int}}}=nothing, max_iter::Int=1_000_000)
  cand = CMAES.CMAESCandidate(copy(mean0), Inf)     # emitter's best is unused (we track elites here)
  em = CMAES.CMAESState(copy(mean0), sigma0, cand, rng; lambda=lambda, blocks=blocks, max_iter=max_iter)
  ncell = prod(grid_dims)
  CMAMAEState(em, rng, grid_dims, Tuple(float.(meas_lo)), Tuple(float.(meas_hi)),
    Vector{Union{Nothing,Vector{Float64}}}(nothing, ncell), fill(Inf, ncell), fill(t0, ncell),
    alpha, t0, sigma0, restart_sigma, restart_patience, false, 0, 0, 0, max_iter, reseed_explore,
    sobol_reseed, Sobol.SobolSeq(length(mean0)))
end

# Map a 2-D measure to a flat cell index (clamped to the grid).
@inline function _cell(st::CMAMAEState, m)::Int
  idx = 1; stride = 1
  @inbounds for d in 1:2
    f = (m[d] - st.lo[d]) / (st.hi[d] - st.lo[d])
    j = clamp(floor(Int, f * st.dims[d]), 0, st.dims[d] - 1)
    idx += j * stride; stride *= st.dims[d]
  end
  return idx
end

# Sample λ offspring; if a restart is pending, first re-seed the emitter (fresh point or random elite).
function ask(st::CMAMAEState)::Vector{Vector{Float64}}
  if st.pending_restart
    occ = findall(x -> x !== nothing, st.elite_x)
    # explore: fresh point (Sobol space-filling if enabled, else uniform random) covering scattered
    # basins in decision space; otherwise archive-guided (re-seed from a random elite).
    explore = isempty(occ) || rand(st.rng) < st.reseed_explore
    seed = explore ? (st.sobol_reseed ? clamp.(Sobol.next!(st._sobol), 0.0, 1.0) : rand(st.rng, st.emitter.n)) :
                     copy(st.elite_x[rand(st.rng, occ)])
    CMAES.restart!(st.emitter, seed, st.sigma0)
    st.pending_restart = false
  end
  return CMAES.ask(st.emitter)
end

# One generation: update the archive from (quality, measure) of each offspring, then update the
# CMA-ES distribution with the per-cell improvement ranking. Flags a restart on convergence.
function tell!(st::CMAMAEState, quals::Vector{Float64}, meas::AbstractVector, xs_u::Vector{Vector{Float64}}; emitter_rank::Union{Nothing,Vector{Int}}=nothing)
  λ = length(quals)
  delta = Vector{Float64}(undef, λ)
  improved = false
  @inbounds for k in 1:λ
    c = _cell(st, meas[k]); q = quals[k]
    delta[k] = st.threshold[c] - q                  # improvement over the cell's soft threshold
    if q < st.elite_f[c]                            # keep the best solution per cell (new or better)
      st.elite_f[c] = q
      st.elite_x[c] = copy(xs_u[k])
      improved = true
    end
    if q < st.threshold[c]                          # relax the threshold toward what was accepted
      st.threshold[c] = (1 - st.alpha) * st.threshold[c] + st.alpha * q
    end
  end
  # emitter recombination order: scalar improvement ranking (CMA-MAE default), or a caller-supplied MO
  # net-win ranking (cmame_mo_rank). Both put the best offspring first for _update_distribution!.
  ranking = emitter_rank === nothing ? sortperm(delta; rev=true) : emitter_rank
  CMAES._update_distribution!(st.emitter, ranking, xs_u)
  st.i += 1
  # CMA-ME improvement-emitter restart: re-seed when the emitter has converged (σ collapsed) OR has
  # stopped adding to the archive for `restart_patience` generations — so a wandering emitter that
  # keeps soft-improving (σ never collapses) is still forced to redeploy and cover new basins.
  st.stagnation = improved ? 0 : st.stagnation + 1
  if st.emitter.sigma < st.restart_sigma || st.stagnation >= st.restart_patience
    st.pending_restart = true
    st.stagnation = 0
    st.n_restarts += 1
  end
  return nothing
end

# The decoded elite solutions (one per occupied cell).
elites(st::CMAMAEState)::Vector{Vector{Float64}} = Vector{Float64}[x for x in st.elite_x if x !== nothing]
elite_count(st::CMAMAEState)::Int = count(x -> x !== nothing, st.elite_x)

# ======================================================================================
# Writer-compatible MO state for the real Pan pipeline. Wraps the engine above: the engine
# owns the emitter + per-cell thresholds/quality + improvement ranking + restarts; this layer
# stores the decoded MOCandidate per cell, tracks the best-quality elite as `representative`,
# and exposes the field names the MO writer/driver read (representative/archive/i/best_iteration
# /n_evals + benign diff_avg/t/prob_avg) — mirroring MOCMAESState/IgelState.
mutable struct CMAMAEMOState{Tx,TRNG<:Random.AbstractRNG}
  engine::CMAMAEState{TRNG}
  cell_cand::Vector{Union{Nothing,MOLBSA.MOCandidate{Tx}}}   # decoded elite params per cell
  representative::MOLBSA.MOCandidate{Tx}                     # lowest-aggregate elite (the "best")
  current::MOLBSA.MOCandidate{Tx}
  archive::Vector{MOLBSA.MOCandidate{Tx}}                    # occupied cells (the QD set of param sets)
  best_iterations::Vector{Tuple{Int,Float64,MOLBSA.MOCandidate{Tx}}}
  best_iteration::Int
  i::Int
  n_evals::Int
  max_iter::Int
  t::Float64
  diff_avg::Float64
  prob_avg::Float64
  mo_rank::Bool   # rank the emitter by MO net-win count instead of scalar quality improvement
end

function CMAMAEMOState(mean0::Vector{Float64}, sigma0::Float64, rep::MOLBSA.MOCandidate{Tx}, rng::Random.AbstractRNG;
                       meas_lo::NTuple{2,Real}, meas_hi::NTuple{2,Real}, lambda::Union{Nothing,Int}=nothing,
                       grid::Int=15, alpha::Float64=0.02, t0::Float64=1.0, reseed_explore::Float64=1.0,
                       restart_patience::Int=6, sobol_reseed::Bool=false, mo_rank::Bool=false,
                       blocks::Union{Nothing,Vector{Vector{Int}}}=nothing, max_iter::Int=typemax(Int)) where {Tx}
  λ = isnothing(lambda) ? 4 + floor(Int, 3 * log(length(mean0))) : lambda
  eng = CMAMAEState(copy(mean0), sigma0, rng; lambda=λ, grid_dims=(grid, grid), meas_lo=meas_lo, meas_hi=meas_hi,
        alpha=alpha, t0=t0, restart_sigma=0.02, restart_patience=restart_patience, reseed_explore=reseed_explore,
        sobol_reseed=sobol_reseed, blocks=blocks, max_iter=max_iter)
  ncell = prod(eng.dims)
  CMAMAEMOState{Tx,typeof(rng)}(eng, Vector{Union{Nothing,MOLBSA.MOCandidate{Tx}}}(nothing, ncell),
    rep, rep, MOLBSA.MOCandidate{Tx}[rep], Tuple{Int,Float64,MOLBSA.MOCandidate{Tx}}[],
    0, 0, 0, max_iter, sigma0, 0.0, 0.0, mo_rank)
end

ask(w::CMAMAEMOState) = ask(w.engine)

# 2-D behaviour descriptor from a MOFitness: (Σ Wasserstein coords, Σ AGB coords) — the two
# interleaved halves of _mo_objectives. Quality (what CMA-MAE minimises) is fx.aggregate = their sum.
measure(fx::MOLBSA.MOFitness) = (sum(@view fx.objectives[1:2:end]), sum(@view fx.objectives[2:2:end]))

# One generation with decoded candidates: record per-cell elites + representative, then drive the
# engine with the aggregate quality. Returns true iff the representative (best aggregate) improved.
function tell_mo!(w::CMAMAEMOState, cands::Vector{<:MOLBSA.MOCandidate}, meas::AbstractVector, xs_u::Vector{Vector{Float64}})
  eng = w.engine
  quals = Float64[convert(Float64, c.fx.aggregate) for c in cands]
  localf = copy(eng.elite_f)                       # mirror the engine's per-cell elite as we scan
  is_new_best = false
  for k in eachindex(cands)
    c = _cell(eng, meas[k]); q = quals[k]
    if q < localf[c]
      localf[c] = q
      w.cell_cand[c] = cands[k]
    end
    if q < convert(Float64, w.representative.fx.aggregate)
      w.representative = cands[k]
      w.best_iteration = eng.i + 1
      is_new_best = true
    end
  end
  emitter_rank = w.mo_rank ? mo_sortperm(MOLBSA.MOFitness[c.fx for c in cands]) : nothing
  tell!(eng, quals, meas, xs_u; emitter_rank=emitter_rank)   # ranking + emitter update + thresholds + restart
  w.i = eng.i
  w.t = eng.emitter.sigma
  w.archive = [c for c in w.cell_cand if c !== nothing]
  is_new_best && push!(w.best_iterations, (w.best_iteration, convert(Float64, w.representative.fx.aggregate), w.representative))
  return is_new_best
end

end
