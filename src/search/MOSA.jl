module MOSA
# Multi-objective classic Simulated Annealing — the MO counterpart of SA.jl, the same way MOLBSA is the
# MO counterpart of LBSA. Keeps SA's FIXED geometric cooling schedule (`t *= alpha`) + probability-based
# reheat, but ranks moves by the per-(eco,species,{w,agb}) objective vector via MOLBSA.mo_delta (net
# win/loss count, scale-invariant) and maintains a Pareto archive (mirroring MOLBSA.update_archive!).
import Random
import ..MOLBSA: MOFitness, MOCandidate, mo_delta, dominates

export MOSACandidate, MOSAState, search_cmp!, search_update_rule!, should_restart, update_archive!, restart

const MOSACandidate = MOCandidate                       # reuse the MO candidate type

@inline simulated_annealing_acceptance_rule(rng, diff_fit, t) = exp(-diff_fit / t) > rand(rng, Float64)
@inline threshold_accepting_acceptance_rule(rng, diff_fit, t) = diff_fit < t

@inline should_restart(state)::Bool = state._should_restart
@inline is_search_over(state)::Bool = state.i >= state.max_iter

# Insert `cand` into the Pareto archive if non-dominated (pruning members it dominates, capping by
# worst aggregate); update the representative (min aggregate). Returns true on representative improvement.
@inline function update_archive!(state, cand::MOCandidate)::Bool
  objs = cand.fx.objectives
  for m in state.archive
    dominates(m.fx.objectives, objs) && return false
  end
  filter!(m -> !dominates(objs, m.fx.objectives), state.archive)
  push!(state.archive, cand)
  if length(state.archive) > state.archive_cap
    worst_i = argmax(i -> state.archive[i].fx.aggregate, eachindex(state.archive))
    deleteat!(state.archive, worst_i)
  end
  improved = cand.fx.aggregate < state.representative.fx.aggregate
  if improved
    state.representative = cand
    state.best_iteration = state.i
    push!(state.best_iterations, (state.i, cand.fx.aggregate, cand))
  end
  return improved
end

@inline function restart(state, cand::MOCandidate)::Bool
  state.current = cand
  state._should_restart = false
  state.t = state.initial_t                             # reheat to the start temperature
  return update_archive!(state, cand)
end

# One trial: classic-SA acceptance on the MO net-count delta + geometric-cooling reheat + archive update.
@inline function search_cmp!(next::MOCandidate, state)
  diff_fit = mo_delta(next.fx, state.current.fx)
  prob = (diff_fit <= 0 ? 1.0 : exp(-diff_fit / state.t))
  if isinf(state.diff_avg) || state.i == 1
    state.diff_avg = diff_fit
    state.prob_avg = prob
  else
    state.diff_avg = state.running_average_ratio * (state.diff_avg - diff_fit) + diff_fit
    state.prob_avg = state.running_average_ratio * (state.prob_avg - prob) + prob
  end
  if state.prob_avg < state.reheat_prob_threshold       # SA-style reheat when acceptance stalls
    state.reheat_iter_counter += 1
    if state.reheat_iter_counter >= state.reheat_after_iters
      state.reheat_iter_counter = 0
      state.t = state.initial_t
    end
  else
    state.reheat_iter_counter = 0
  end
  acc = state.acceptance_rule == "TA" ? threshold_accepting_acceptance_rule : simulated_annealing_acceptance_rule
  if diff_fit < 0 || acc(state.rng, diff_fit, state.t)
    state.current = next
    state.current_iteration = state.i
    return update_archive!(state, state.current)
  end
  return false
end

@inline function search_update_rule!(state)::Bool
  state.t *= state.alpha
  state.t < state.min_t || state.i >= state.max_iter
end

Base.@kwdef mutable struct MOSAState{Tx,TRNG<:Random.AbstractRNG}
  representative::MOCandidate{Tx}
  current::MOCandidate{Tx}
  archive::Vector{MOCandidate{Tx}}
  archive_cap::Int = 200
  rng::TRNG
  best_iterations::Vector{Tuple{Int,Float64,MOCandidate{Tx}}}
  best_iteration::Int = 0
  current_iteration::Int = 0
  i::Int = 1
  max_iter::Int = 1_000_000
  t::Float64 = 1.0
  initial_t::Float64 = 1.0
  min_t::Float64 = 1e-4
  alpha::Float64 = 0.999
  acceptance_rule::String = "SA"
  running_average_ratio::Float64 = 0.9
  diff_avg::Float64 = 0.0
  prob_avg::Float64 = 0.0
  reheat_prob_threshold::Float64 = 0.01
  reheat_after_iters::Int = 100
  reheat_iter_counter::Int = 0
  _should_restart::Bool = false
end

function MOSAState(representative::MOCandidate{Tx}, current::MOCandidate{Tx}, rng::TRNG;
                   archive::Vector{MOCandidate{Tx}}=MOCandidate{Tx}[representative],
                   best_iterations=Tuple{Int,Float64,MOCandidate{Tx}}[], kwargs...) where {Tx,TRNG<:Random.AbstractRNG}
  MOSAState{Tx,TRNG}(; representative=representative, current=current, archive=archive,
    rng=rng, best_iterations=best_iterations, kwargs...)
end
end
