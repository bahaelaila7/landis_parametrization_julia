module SA
import Random

export  SACandidate, SAState, simulated_annealing_acceptance_rule, threshold_accepting_acceptance_rule, search_cmp!, search_update_rule!

Base.@kwdef struct SACandidate{Tx,Tf}
    x::Tx
    fx::Tf
end

@inline simulated_annealing_acceptance_rule(rng, diff_fit, t) = exp(-diff_fit / t) > rand(Float64)
@inline threshold_accepting_acceptance_rule(rng, diff_fit, t) = diff_fit < t
@inline function search_cmp!(next, state)
        diff_fit = convert(Float64, next.fx) - convert(Float64,state.current.fx)
        if diff_fit < 0 || state.acceptance_rule(state.rng, diff_fit, state.t)
            state.current = next
            state.current_iteration = state.i
            if convert(Float64, state.current.fx) < convert(Float64, state.best.fx)
                state.best = state.current
                state.best_iteration = state.i
                return true
            end
        end
        return false
end
@inline function search_update_rule!(state)::Bool
        state.t *= state.alpha
        state.t < state.min_t || state.i >= state.max_iter
end

Base.@kwdef mutable struct SAState{Tx,Tf}
    best::SACandidate{Tx,Tf}
    current::SACandidate{Tx,Tf}
    rng::Random.Xoshiro
    best_iteration::Int = 0
    current_iteration::Int = 0
    i::Int = 1
    max_iter::Int = 1000
    t::Float64 = 10000.0
    initial_t::Float64 = 10000.0
    min_t::Float64 = 0.01
    alpha::Float64 = 0.9992
    trials_per_iter::Int = 3
    acceptance_rule = simulated_annealing_acceptance_rule
end


function search(state::Union{Nothing,Some{SAState}}, get_neighbor, get_fitness)::SAState
    if isnothing(state)
        first = get_neighbor(nothing; rng=state.rng)
        first_fit = get_fitness(first)
        cur = SACandidate(first, first_fit)
        state = SAState(cur, cur)
    end

    start_i = state.i
    for i in start_i:state.max_iter
        state.i = i
        next_cand = get_neighbor(cur; rng=state.rng)

        next = SACandidate(next_cand, get_fitness(next_cand))

        search_cmp!(next, state)
        if search_update_rule!(state)
            break
        end

    end
end
end
