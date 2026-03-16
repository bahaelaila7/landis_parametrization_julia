module SA
import Random
Base.@kwdef struct SACandidate{Tx,Tf}
    x::Tx
    fx::Tf
end

@inline simulated_annealing_acceptance_rule(rng, diff_fit, t) = exp(-diff_fit / t) > rand(Float64)
@inline threshold_accepting_acceptance_rule(rng, diff_fit, t) = diff_fit < t

Base.@kwdef mutable struct SAState{Tx,Tf}
    best::SACandidate{Tx,Tf}
    current::SACandidate{Tx,Tf}
    rng::Random.Xoshiro
    best_iteration::Int = 0
    current_iteration::Int = 0
    i::Int = 1
    max_iter::Int = 1000
    t::Float64 = 10000
    initial_t::Float64 = 10000
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

        diff_fit = next.fx - state.current.fx
        if diff_fit < 0 || state.acceptance_rule(state.rng, diff_fit, state.t)
            state.current = next
            state.current_iteration = i
            if state.current.fx < state.best.fx
                state.best = state.current
                state.best_iteration = i
            end
        end

        state.t *= state.alpha
        if state.t < state.min_t
            break
        end
    end
end
end
