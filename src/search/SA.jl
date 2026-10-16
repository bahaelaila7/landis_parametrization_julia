module SA
# Classic simulated annealing with a GEOMETRIC COOLING SCHEDULE (t ← α·t every trial) on a scalar fitness (Step 5b rebuild of the stale driver).
# Metropolis acceptance (`acceptance_rule == "SA"`: accept an uphill move of size Δ with probability exp(-Δ/t)) or threshold accepting ("TA": accept
# when Δ < t). The run ends when the budget is spent or t < min_t. A running average of the acceptance probability triggers a REHEAT to
# `initial_t` when it stays below `reheat_prob_threshold` for `reheat_after_iters` consecutive trials (the search froze).
# Contrast with LBSA (search/LBSA.jl), which has no schedule but an adaptive list of temperatures.
import Random

export SACandidate, SAState, simulated_annealing_acceptance_rule, threshold_accepting_acceptance_rule, search_cmp!, search_update_rule!, is_search_over

Base.@kwdef struct SACandidate{Tx,Tf}
    x::Tx
    fx::Tf
end

@inline simulated_annealing_acceptance_rule(rng, diff_fit, t) = exp(-diff_fit / t) > rand(rng, Float64)
@inline threshold_accepting_acceptance_rule(rng, diff_fit, t) = diff_fit < t

Base.@kwdef mutable struct SAState{Tx,Tf, TRNG <: Random.AbstractRNG}
    best::SACandidate{Tx,Tf}
    current::SACandidate{Tx,Tf}
    rng::TRNG
    best_iterations::Vector{Tuple{Int,Float64,Tf}}
    best_iteration::Int = 0
    current_iteration::Int = 0
    i::Int = 0                       # number of trials compared so far (the start point is trial 0)
    max_iter::Int = 1000
    t::Float64 = 10000.0
    initial_t::Float64 = 10000.0
    min_t::Float64 = 0.01
    alpha::Float64 = 0.9992
    acceptance_rule::String = "SA"   # "SA" (Metropolis) | "TA" (threshold accepting)
    running_average_ratio::Float64 = 0.9
    diff_avg::Float64 = 0.0
    prob_avg::Float64 = 0.0
    reheat_prob_threshold::Float64 = 0.01
    reheat_after_iters::Int = 100
    reheat_iter_counter::Int = 0
end
function SAState(best::SACandidate{Tx,Tf}, current::SACandidate{Tx,Tf}, rng::TRNG; best_iterations=Tuple{Int,Float64,Tf}[], kwargs...) where {Tx,Tf, TRNG <: Random.AbstractRNG}
    SAState{Tx,Tf, TRNG}(; best=best, current=current, rng = rng,
        best_iterations=best_iterations,
        kwargs...)
end

"The search is over: the trial budget is spent or the schedule has cooled below `min_t`."
@inline is_search_over(state)::Bool = state.i >= state.max_iter || state.t < state.min_t

"""
    search_cmp!(next, state) -> is_new_best

Compare the proposal `next` with the current point (trial `state.i += 1`): update the running averages and the reheat counter, apply the
acceptance rule (draws from `state.rng` only for an uphill move under "SA"), move the current point and the best.
"""
@inline function search_cmp!(next, state)
    state.i += 1
    diff_fit = convert(Float64, next.fx) - convert(Float64, state.current.fx)
    prob = (diff_fit <= 0 ? 1.0 : exp(-diff_fit / state.t))
    if state.i == 1
        state.diff_avg = diff_fit
        state.prob_avg = prob
    else
        state.diff_avg = state.running_average_ratio * (state.diff_avg - diff_fit) + diff_fit
        state.prob_avg = state.running_average_ratio * (state.prob_avg - prob) + prob
    end
    if state.prob_avg < state.reheat_prob_threshold
        state.reheat_iter_counter += 1
        if state.reheat_iter_counter >= state.reheat_after_iters
            state.reheat_iter_counter = 0
            state.t = state.initial_t
        end
    else
        state.reheat_iter_counter = 0
    end
    acceptance_rule = state.acceptance_rule == "TA" ? threshold_accepting_acceptance_rule : simulated_annealing_acceptance_rule

    if diff_fit < 0 || acceptance_rule(state.rng, diff_fit, state.t)
        state.current = next
        state.current_iteration = state.i
        cur_fit = convert(Float64, state.current.fx)
        if cur_fit < convert(Float64, state.best.fx)
            state.best = state.current
            state.best_iteration = state.i
            push!(state.best_iterations, (state.i, cur_fit, state.best.fx))
            return true
        end
    end
    return false
end

"Cool the temperature one step (t ← α·t); `true` when the search is over (`is_search_over`)."
@inline function search_update_rule!(state)::Bool
    state.t *= state.alpha
    return is_search_over(state)
end
end
