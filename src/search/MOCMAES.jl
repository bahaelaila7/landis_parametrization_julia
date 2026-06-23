module MOCMAES
# Multi-objective CMA-ES. Stands to single-objective CMA-ES (search/CMAES.jl) exactly as MOLBSA
# stands to LBSA: the optimizer ENGINE is reused verbatim, and only the candidate comparison and the
# notion of "best" change. Concretely:
#   1. A candidate is scored by the per-(eco,species,{w,agb}) objective VECTOR (MOLBSA.MOFitness),
#      not a scalar. CMA-ES only needs a best→worst ORDER of the λ offspring to recombine the best μ,
#      so we replace the scalar `sortperm` with a multi-objective `mo_sortperm`.
#   2. "Best" is not a total order, so we keep a Pareto ARCHIVE of non-dominated candidates and a
#      `representative` (min aggregate) for logging / plotting / checkpointing — exactly like MOLBSA.
# The CMA-ES distribution update (CMAES._update_distribution!) and sampling (CMAES.ask) are shared.
import Random
import LinearAlgebra
const LA = LinearAlgebra
import ..CMAES
import ..MOLBSA: MOFitness, MOCandidate, dominates

export MOCMAESState, tell!, update_archive!, restart!, is_search_over, mo_sortperm

Base.@kwdef mutable struct MOCMAESState{Tx,TRNG<:Random.AbstractRNG}
  # ---- MOLBSA-style bookkeeping (field names match start_mo_writer) ----
  representative::MOCandidate{Tx}              # min-aggregate member, for logging/checkpoints
  current::MOCandidate{Tx}                     # best-aggregate offspring of the latest generation
  archive::Vector{MOCandidate{Tx}}             # Pareto-non-dominated set (capped)
  archive_cap::Int = 200
  rng::TRNG
  best_iterations::Vector{Tuple{Int,Float64,MOCandidate{Tx}}}
  best_iteration::Int = 0
  current_iteration::Int = 0
  i::Int = 0                                   # generation counter → checkpoint filenames
  n_evals::Int = 0                             # total fitness evaluations spent (resume-exact budget)
  max_iter::Int = 1_000_000                    # max generations (driver also bounds by TRIALS evals)
  # benign writer log fields (start_mo_writer's non-best branch reads diff_avg/t/prob_avg; t = σ)
  t::Float64 = 0.0
  diff_avg::Float64 = 0.0
  prob_avg::Float64 = 0.0

  # ---- CMA-ES static config (same as CMAESState; recomputed on IPOP restart) ----
  n::Int
  lambda::Int
  mu::Int
  weights::Vector{Float64}
  mu_eff::Float64
  c_sigma::Float64
  d_sigma::Float64
  c_c::Float64
  c_1::Float64
  c_mu::Float64
  chiN::Float64

  # ---- CMA-ES dynamic state (u-space R^d, Float64) ----
  mean::Vector{Float64}
  sigma::Float64
  C::Matrix{Float64}
  p_sigma::Vector{Float64}
  p_c::Vector{Float64}
  B::Matrix{Float64}
  D::Vector{Float64}

  # Hansen-style mixed-integer handling floor (see CMAES.ask); empty ⇒ disabled.
  u_min_std::Vector{Float64} = Float64[]
end

function MOCMAESState(mean0::Vector{Float64}, sigma0::Float64, representative::MOCandidate{Tx}, rng::TRNG;
                      lambda::Union{Nothing,Int}=nothing, max_iter::Int=1_000_000, archive_cap::Int=200,
                      archive::Vector{MOCandidate{Tx}}=MOCandidate{Tx}[representative],
                      best_iterations=Tuple{Int,Float64,MOCandidate{Tx}}[]) where {Tx,TRNG<:Random.AbstractRNG}
  n = length(mean0)
  λ = isnothing(lambda) ? 4 + floor(Int, 3 * log(n)) : lambda
  k = CMAES._strategy_constants(n, λ)
  MOCMAESState{Tx,TRNG}(; representative=representative, current=representative, archive=archive,
    archive_cap=archive_cap, rng=rng, best_iterations=best_iterations, max_iter=max_iter, t=sigma0,
    n=n, lambda=λ, mu=k.mu, weights=k.weights, mu_eff=k.mu_eff,
    c_sigma=k.c_sigma, d_sigma=k.d_sigma, c_c=k.c_c, c_1=k.c_1, c_mu=k.c_mu, chiN=k.chiN,
    mean=copy(mean0), sigma=sigma0, C=Matrix{Float64}(LA.I, n, n),
    p_sigma=zeros(n), p_c=zeros(n), B=Matrix{Float64}(LA.I, n, n), D=ones(n))
end

@inline is_search_over(state::MOCMAESState)::Bool = state.i >= state.max_iter

# Net objective count of `a` vs `b`: (#objectives where a is better/smaller) − (#where worse).
# Higher = `a` better. This is the graded signal behind MOLBSA.mo_delta.
@inline function _net_wins(a::Vector{Float32}, b::Vector{Float32})::Int
  wins = 0
  losses = 0
  @inbounds for k in eachindex(a)
    d = a[k] - b[k]
    if d < 0.0f0
      wins += 1
    elseif d > 0.0f0
      losses += 1
    end
  end
  return wins - losses
end

# Best→worst total order over the λ offspring for the CMA-ES recombination. Each offspring is
# scored by its summed net pairwise objective win-count against the rest of the population (higher =
# better), tie-broken by the scalar aggregate. Generalizes MOLBSA.mo_delta to a population and stays
# informative even when strict Pareto fronts collapse (≈everything non-dominated at many objectives).
function mo_sortperm(fxs::Vector{MOFitness})::Vector{Int}
  m = length(fxs)
  score = zeros(Int, m)
  @inbounds for k in 1:m
    for j in 1:m
      k == j && continue
      score[k] += _net_wins(fxs[k].objectives, fxs[j].objectives)
    end
  end
  return sortperm(1:m; by = k -> (-score[k], fxs[k].aggregate))
end

# One generation update, driven by the multi-objective ranking of the offspring.
function tell!(state::MOCMAESState, fxs::Vector{MOFitness}, xs_u::Vector{Vector{Float64}})
  CMAES._update_distribution!(state, mo_sortperm(fxs), xs_u)
end

# NSGA-II crowding distance per archive member (higher = more isolated). Boundary members — the
# best on any single objective — get Inf so the extremes of the front are never evicted.
function _crowding(archive::Vector{<:MOCandidate})::Vector{Float64}
  N = length(archive)
  N <= 2 && return fill(Inf, N)
  M = length(archive[1].fx.objectives)
  cd = zeros(Float64, N)
  vals = Vector{Float64}(undef, N)
  for o in 1:M
    @inbounds for i in 1:N
      vals[i] = Float64(archive[i].fx.objectives[o])
    end
    order = sortperm(vals)
    span = vals[order[end]] - vals[order[1]]
    cd[order[1]] = Inf
    cd[order[end]] = Inf
    if span > 0
      for r in 2:N-1
        cd[order[r]] += (vals[order[r+1]] - vals[order[r-1]]) / span
      end
    end
  end
  return cd
end

# Insert `cand` into the Pareto archive (dominance logic as in MOLBSA.update_archive!): skip if
# dominated; prune members it dominates; push. When over the cap, evict the MOST CROWDED member
# (smallest crowding distance) rather than the worst aggregate — unlike MOLBSA, the CMA-ES
# distribution converges, so aggregate-based eviction would collapse the archive onto the
# compromise point; crowding eviction retains the spread and extremes discovered during early
# exploration. Returns true only when `cand` improves the representative's aggregate (gates
# checkpoints; the representative is still the min-aggregate member, as in MOLBSA).
function update_archive!(state::MOCMAESState, cand::MOCandidate)::Bool
  objs = cand.fx.objectives
  for m in state.archive
    dominates(m.fx.objectives, objs) && return false
  end
  filter!(m -> !dominates(objs, m.fx.objectives), state.archive)
  push!(state.archive, cand)
  if length(state.archive) > state.archive_cap
    deleteat!(state.archive, argmin(_crowding(state.archive)))
  end
  improved = cand.fx.aggregate < state.representative.fx.aggregate
  if improved
    state.representative = cand
    state.best_iteration = state.i
    push!(state.best_iterations, (state.i, cand.fx.aggregate, cand))
  end
  return improved
end

# IPOP restart: reset the search distribution (optionally with a larger population) while keeping the
# Pareto archive, representative, generation counter, rng and history.
function restart!(state::MOCMAESState, mean0::Vector{Float64}, sigma0::Float64; lambda::Int=state.lambda)
  n = state.n
  k = CMAES._strategy_constants(n, lambda)
  state.lambda = lambda
  state.mu = k.mu
  state.weights = k.weights
  state.mu_eff = k.mu_eff
  state.c_sigma = k.c_sigma
  state.d_sigma = k.d_sigma
  state.c_c = k.c_c
  state.c_1 = k.c_1
  state.c_mu = k.c_mu
  state.chiN = k.chiN
  state.mean = copy(mean0)
  state.sigma = sigma0
  state.C = Matrix{Float64}(LA.I, n, n)
  state.B = Matrix{Float64}(LA.I, n, n)
  state.D = ones(n)
  state.p_sigma = zeros(n)
  state.p_c = zeros(n)
  state.t = sigma0
  return nothing
end
end
