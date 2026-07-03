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

  # ---- CMA-ES config + BLOCK-DIAGONAL distribution (shared engine; CMAES.ask / _update_distribution!
  #      duck-type on .blocks/.n/.lambda/.i/.t/.u_min_std). 1 block ⇒ plain MO-CMA-ES. ----
  n::Int
  lambda::Int
  blocks::Vector{CMAES.CMABlock}

  # Hansen-style mixed-integer handling floor (see CMAES.ask); empty ⇒ disabled.
  u_min_std::Vector{Float64} = Float64[]
end

# `.sigma` = max-over-blocks σ (so a `< ε` collapse test fires only when ALL blocks are tiny); `.mu`
# from the first block (drivers read it for the progress estimate).
function Base.getproperty(s::MOCMAESState, f::Symbol)
  if f === :sigma
    bl = getfield(s, :blocks); return isempty(bl) ? 0.0 : maximum(b.sigma for b in bl)
  elseif f === :mu
    bl = getfield(s, :blocks); return isempty(bl) ? 0 : bl[1].mu
  end
  return getfield(s, f)
end

function MOCMAESState(mean0::Vector{Float64}, sigma0::Float64, representative::MOCandidate{Tx}, rng::TRNG;
                      lambda::Union{Nothing,Int}=nothing, max_iter::Int=1_000_000, archive_cap::Int=200,
                      blocks::Union{Nothing,Vector{Vector{Int}}}=nothing,
                      archive::Vector{MOCandidate{Tx}}=MOCandidate{Tx}[representative],
                      best_iterations=Tuple{Int,Float64,MOCandidate{Tx}}[]) where {Tx,TRNG<:Random.AbstractRNG}
  n = length(mean0)
  λ = isnothing(lambda) ? 4 + floor(Int, 3 * log(n)) : lambda
  blk_idx = isnothing(blocks) ? [collect(1:n)] : blocks
  blks = [CMAES._make_block(idx, mean0, sigma0, λ) for idx in blk_idx]
  MOCMAESState{Tx,TRNG}(; representative=representative, current=representative, archive=archive,
    archive_cap=archive_cap, rng=rng, best_iterations=best_iterations, max_iter=max_iter, t=sigma0,
    n=n, lambda=λ, blocks=blks)
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

# Offspring-ranking selector. false (default) → net pairwise win-count; true → TRUE NSGA-II
# non-dominated sorting (Pareto fronts + crowding). Set from yaml `mocmaes_true_nds`.
const USE_NDS = Ref{Bool}(false)

# Best→worst total order over the λ offspring for the CMA-ES recombination.
function mo_sortperm(fxs::Vector{MOFitness})::Vector{Int}
  USE_NDS[] ? _nds_sortperm(fxs) : _winrank_sortperm(fxs)
end

# Win-count ranking: each offspring scored by its summed net pairwise objective win-count against the
# rest of the population (higher = better), tie-broken by the scalar aggregate. Generalizes
# MOLBSA.mo_delta to a population and stays informative even when strict Pareto fronts collapse
# (≈everything non-dominated at many objectives).
function _winrank_sortperm(fxs::Vector{MOFitness})::Vector{Int}
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

# TRUE NSGA-II ranking: fast non-dominated sort into Pareto fronts (rank 1 = non-dominated), ordered
# best→worst; within a front, more-isolated points (higher crowding distance) rank first, tie-broken by
# scalar aggregate. NOTE: at high objective count most offspring share front 1, so crowding does most of
# the ordering — the classic many-objective degeneracy the win-count avoids (hence the flag defaults off).
function _nds_sortperm(fxs::Vector{MOFitness})::Vector{Int}
  m = length(fxs)
  m <= 1 && return collect(1:m)
  domcount = zeros(Int, m)                     # #offspring dominating p
  dominated = [Int[] for _ in 1:m]             # offspring p dominates
  front = Int[]
  @inbounds for p in 1:m
    op = fxs[p].objectives
    for q in 1:m
      p == q && continue
      oq = fxs[q].objectives
      if dominates(op, oq)
        push!(dominated[p], q)
      elseif dominates(oq, op)
        domcount[p] += 1
      end
    end
    domcount[p] == 0 && push!(front, p)        # front 1
  end
  order = Int[]
  while !isempty(front)
    if length(front) <= 2
      append!(order, sort(front; by = k -> fxs[k].aggregate))
    else
      cd = _crowding_fx(fxs, front)
      append!(order, front[sortperm(1:length(front); by = t -> (-cd[t], fxs[front[t]].aggregate))])
    end
    nxt = Int[]                                # peel next front
    @inbounds for p in front, q in dominated[p]
      domcount[q] -= 1
      domcount[q] == 0 && push!(nxt, q)
    end
    front = nxt
  end
  return order
end

# NSGA-II crowding distance over a subset `idxs` of `fxs` (higher = more isolated; per-objective
# boundary points = Inf). Same formula as `_crowding`, but over offspring MOFitness rather than the archive.
function _crowding_fx(fxs::Vector{MOFitness}, idxs::Vector{Int})::Vector{Float64}
  N = length(idxs)
  N <= 2 && return fill(Inf, N)
  M = length(fxs[idxs[1]].objectives)
  cd = zeros(Float64, N); vals = Vector{Float64}(undef, N)
  for o in 1:M
    @inbounds for i in 1:N; vals[i] = Float64(fxs[idxs[i]].objectives[o]); end
    ord = sortperm(vals)
    span = vals[ord[end]] - vals[ord[1]]
    cd[ord[1]] = Inf; cd[ord[end]] = Inf
    if span > 0
      for r in 2:N-1
        cd[ord[r]] += (vals[ord[r+1]] - vals[ord[r-1]]) / span
      end
    end
  end
  return cd
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
# exploration. Returns (changed, improved): `changed` = `cand` was ACCEPTED (non-dominated → archive
# contents change, whether it grows the archive or swaps a member out); `improved` = it also became the
# new representative (aggregate dropped). improved ⟹ changed. Callers checkpoint the archive on any
# `changed` (so params are pullable at every archive change, not just representative improvements).
function update_archive!(state::MOCMAESState, cand::MOCandidate)
  objs = cand.fx.objectives
  for m in state.archive
    dominates(m.fx.objectives, objs) && return (changed=false, improved=false)
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
  return (changed=true, improved=improved)
end

# IPOP restart: reset the search distribution (optionally with a larger population) while keeping the
# Pareto archive, representative, generation counter, rng and history.
function restart!(state::MOCMAESState, mean0::Vector{Float64}, sigma0::Float64; lambda::Int=state.lambda)
  state.lambda = lambda
  CMAES._reset_blocks!(state.blocks, mean0, sigma0, lambda)
  state.t = sigma0
  return nothing
end
end
