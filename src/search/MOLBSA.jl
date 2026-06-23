module MOLBSA
# Multi-objective List-Based Simulated Annealing.
#
# Differs from the scalar LBSA (search/LBSA.jl) in exactly two places:
#   1. The signed energy delta between two candidates is a *net objective count*
#      (how many objectives `next` LOSES on minus how many it WINS on), not a
#      scalar loss difference. A sum-of-losses tie-break adds a |frac| < 0.5
#      term so count ties (and only ties) are decided by the aggregate loss.
#   2. "Best" is not a total order (net-count is intransitive), so instead of a
#      single incumbent we keep a Pareto archive of non-dominated candidates and
#      a `representative` (min aggregate) for logging / plotting / checkpointing.
#
# Everything else — warm-up, the adaptive temperature list `_t_list`, stretches,
# reheat and restart — is the LBSA machinery verbatim, because all of it is
# generic over a single signed `diff_fit`.
import Random
import Statistics

export MOFitness, MOCandidate, MOLBSAState, mo_delta, dominates,
  search_cmp!, should_restart, is_search_over, restart, update_archive!

@enum SearchMethod SimulatedAnnealing ThresholdAccepting

# Vector-valued fitness. `objectives` is the per-(eco,species,{w,agb}) loss vector
# (fixed length & order across all candidates, so it is coordinate-comparable);
# `aggregate` is the old scalar get_total_loss, used only for the tie-break and to
# rank the representative.
struct MOFitness
  objectives::Vector{Float32}
  aggregate::Float64
end

Base.@kwdef struct MOCandidate{Tx}
  x::Tx
  fx::MOFitness
end

# Net energy delta for minimization: positive = `next` is worse ("uphill").
#   diff = (#objectives next loses) - (#objectives next wins) + tie_break
# The tie-break is a relative-aggregate term squashed into (-0.49, 0.49); it only
# decides the sign when the integer count is 0, and otherwise nudges magnitude.
@inline function mo_delta(next::MOFitness, cur::MOFitness)::Float64
  no = next.objectives
  co = cur.objectives
  wins = 0
  losses = 0
  @inbounds for k in eachindex(no)
    d = no[k] - co[k]
    if d < zero(Float32)
      wins += 1
    elseif d > zero(Float32)
      losses += 1
    end
  end
  rel = (next.aggregate - cur.aggregate) / (abs(cur.aggregate) + 1e-10)
  frac = 0.49 * tanh(rel)
  return Float64(losses - wins) + frac
end

# Pareto dominance for minimization: `a` dominates `b` iff a ≤ b on every
# objective and a < b on at least one.
@inline function dominates(a::Vector{Float32}, b::Vector{Float32})::Bool
  strictly = false
  @inbounds for k in eachindex(a)
    a[k] > b[k] && return false
    a[k] < b[k] && (strictly = true)
  end
  return strictly
end

@inline should_restart(state)::Bool = state._should_restart
@inline is_search_over(state)::Bool = state.i >= state.max_iter

# Replaces LBSA's `_check_best`. Inserts `cand` into the Pareto archive if it is
# non-dominated (pruning members it dominates, capping archive size by evicting
# the worst aggregate). Returns the "report-worthy" flag — true only when the
# candidate is non-dominated AND improves the representative's aggregate — which
# gates checkpoints/plots so they don't fire on every weakly-non-dominated point.
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
    state._frozen_no_best_reheats = 0
    state.representative = cand
    state.best_iteration = state.i
    push!(state.best_iterations, (state.i, cand.fx.aggregate, cand))
    state._hill_climbing_best = state.hill_climbing_upon_new_best
  end
  return improved
end

@inline function restart(state, candidate::MOCandidate)::Bool
  state.current = candidate
  state._should_restart = false
  state._frozen_no_best_reheats = 0
  state._warming_up = true
  state.warm_up_record_uphill_only = true
  state._t_list = Float64[]
  return update_archive!(state, state.current)
end

@inline function search_cmp!(next::MOCandidate, state)
  state.i += 1
  diff_fit = mo_delta(next.fx, state.current.fx)
  prob = (diff_fit <= 0 ? 1.0 : exp(-diff_fit / state.t))
  if isinf(state.diff_avg) || state.i == 1
    state.diff_avg = diff_fit
    state.prob_avg = prob
  else
    state.diff_avg = state.running_average_ratio * (state.diff_avg - diff_fit) + diff_fit
    state.prob_avg = state.running_average_ratio * (state.prob_avg - prob) + prob
  end

  accept = false
  if state._warming_up
    accept = diff_fit < 0 || !state.warm_up_greedy_acceptance
    if diff_fit >= 0 || !state.warm_up_record_uphill_only
      if state.search_method == SimulatedAnnealing
        push!(state._t_list, diff_fit * state._neg_inv_lnp0)
      else
        push!(state._t_list, diff_fit)
      end
    end
    if state.temp_list_oversample
      n = 2 * state.temp_list_len
      if length(state._t_list) == n
        sorted_idxs = sortperm(state._t_list, rev=true)
        lo = div(n, 4) + 1
        hi = 3 * div(n, 4)
        sorted_idxs = filter(x -> x >= lo && x <= hi, sorted_idxs)
        state._t_list = state._t_list[sorted_idxs]
        state._t_max_idx = argmax(state._t_list)
        state._t_oldest_idx = 1
        state._warming_up = false
      end
    elseif length(state._t_list) == state.temp_list_len
      state._t_max_idx = argmax(state._t_list)
      state._t_oldest_idx = 1
      state._warming_up = false
    end
    if state.search_method == ThresholdAccepting && length(state._t_list) >= 2
      t_mean = Statistics.mean(state._t_list)
      t_std = Statistics.std(state._t_list)
      state._t_list .= t_mean + 2 * t_std
      state._t_oldest_idx = 1
      state._t_max_idx = 1
    end

    state._initial_t_list = state._t_list[:]
  elseif state._hill_climbing_best > 0
    state._hill_climbing_best -= 1
    accept = diff_fit < 0
  else
    state._m += 1
    if diff_fit < 0
      accept = true
    else
      state._c_up_attempted += 1
      if state.search_method == SimulatedAnnealing
        r = rand(state.rng)
        if exp(-diff_fit / state._t_list[state._t_max_idx]) >= r
          state._c += 1
          accept = true
          state._t_sum += -diff_fit / log(r)
        end
      else
        if diff_fit < state._t_list[state._t_max_idx]
          state._c += 1
          accept = true
          state._t_sum += diff_fit
        end
      end
    end
    #stretch stuff
    if state._m >= state.stretch_len
      state._m = 0
      if state._c > 0
        tsum = state._t_sum / state._c
        if tsum < state._t_list[state._t_max_idx] || !state.cooling_only_schedule
          if state.replace_oldest_instead_of_max
            state._t_list[state._t_oldest_idx] = tsum
            state._t_oldest_idx = (state._t_oldest_idx % length(state._t_list)) + 1
          else
            state._t_list[state._t_max_idx] = tsum
          end
        end
        state._t_max_idx = argmax(state._t_list)
        state._frozen_stretches = 0
      elseif state._c == 0 && state._c_up_attempted / state.stretch_len >= state.up_attempt_stale_ratio
        state._frozen_stretches += 1
        if state._frozen_stretches >= state.reheat_after_frozen_stretches
          state._frozen_no_best_reheats += 1
          # "next worse than the incumbent" → net-count loss against the representative.
          if mo_delta(next.fx, state.representative.fx) > 0 && state._frozen_no_best_reheats >= state.restart_after_no_best_reheats
            state._should_restart = true
            accept = false
          else
            if state.reheat_max_only
              state._t_list[state._t_max_idx] *= state.reheat_factor
            else
              state._t_list .*= state.reheat_factor
            end
          end
          state._frozen_stretches = 0
        end
      end
      state._t_sum = 0.0
      state._c = 0
      state._c_up_attempted = 0
    end

    if length(state._t_list) > 0
      state.t = state._t_list[state._t_max_idx]
    end
  end

  if accept
    state.current = next # immutable, aliasing is fine
    state.current_iteration = state.i
    return update_archive!(state, state.current)
  end
  return false
end

Base.@kwdef mutable struct MOLBSAState{Tx,TRNG<:Random.AbstractRNG}
  representative::MOCandidate{Tx}
  current::MOCandidate{Tx}
  archive::Vector{MOCandidate{Tx}}
  archive_cap::Int = 200
  rng::TRNG
  best_iterations::Vector{Tuple{Int,Float64,MOCandidate{Tx}}}
  best_iteration::Int = 0
  current_iteration::Int = 0
  i::Int = 0
  max_iter::Int = 1000000
  warm_up_greedy_acceptance::Bool = false
  warm_up_record_uphill_only::Bool = false
  temp_list_len::Int = 150
  temp_list_oversample::Bool = false
  stretch_len::Int = 150
  initial_acceptance_prob::Float64 = 0.9
  cooling_only_schedule::Bool = false
  up_attempt_stale_ratio::Float64 = 0.95
  reheat_after_frozen_stretches::Int = 10
  restart_after_no_best_reheats::Int = 3
  reheat_factor::Float64 = 1.5
  reheat_max_only::Bool = false
  replace_oldest_instead_of_max::Bool = true
  search_method::SearchMethod = SimulatedAnnealing
  hill_climbing_upon_new_best::Int = 0

  t::Float64 = 0.0

  _hill_climbing_best::Int = 0
  _warming_up::Bool = true
  _neg_inv_lnp0::Float64 = 0.0
  _m::Int = 0
  _frozen_no_best_reheats::Int = 0
  _frozen_stretches::Int = 0
  _c_up_attempted::Int = 0
  _c::Int = 0
  _t_list::Vector{Float64} = Float64[]
  _initial_t_list::Vector{Float64} = Float64[]
  _t_sum::Float64 = 0.0
  _t_max_idx::Int = 1
  _t_oldest_idx::Int = 1
  _should_restart = false

  sobol_cand_idx::Int = 1

  running_average_ratio = 0.9
  diff_avg::Float64 = 0.0
  prob_avg::Float64 = 0.0
end

function MOLBSAState(representative::MOCandidate{Tx}, current::MOCandidate{Tx}, rng::TRNG;
  archive::Vector{MOCandidate{Tx}}=MOCandidate{Tx}[representative],
  best_iterations=Tuple{Int,Float64,MOCandidate{Tx}}[],
  initial_acceptance_prob=0.9, kwargs...) where {Tx,TRNG<:Random.AbstractRNG}
  MOLBSAState{Tx,TRNG}(; representative=representative, current=current, archive=archive,
    rng=rng, best_iterations=best_iterations,
    initial_acceptance_prob=initial_acceptance_prob,
    _neg_inv_lnp0=-1.0 / log(initial_acceptance_prob),
    kwargs...)
end
end
