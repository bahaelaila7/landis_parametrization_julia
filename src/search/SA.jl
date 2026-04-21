module SA
import Random

export SACandidate, SAState, simulated_annealing_acceptance_rule, threshold_accepting_acceptance_rule, search_cmp!, search_update_rule!

Base.@kwdef struct SACandidate{Tx,Tf}
    x::Tx
    fx::Tf
end

@inline simulated_annealing_acceptance_rule(rng, diff_fit, t) = exp(-diff_fit / t) > rand(rng, Float64)
@inline threshold_accepting_acceptance_rule(rng, diff_fit, t) = diff_fit < t
@inline function search_cmp!(next, state)
    diff_fit = convert(Float64, next.fx) - convert(Float64, state.current.fx)
    prob = (diff_fit <= 0 ? 1.0 : exp(-diff_fit / state.t))
    if isinf(state.diff_avg) || state.i == 1
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
        best_fit = convert(Float64, state.best.fx)
        cur_fit = convert(Float64, state.current.fx)
        if  cur_fit < best_fit
            best_fit = cur_fit
            state.best = state.current
            state.best_iteration = state.i
            push!(state.best_iterations, (state.i, best_fit, state.best.fx))
            return true
        end
    end
    return false
end
@inline function search_update_rule!(state)::Bool
    state.t *= state.alpha
    state.t < state.min_t || state.i >= state.max_iter
end

Base.@kwdef mutable struct SAState{Tx,Tf, TRNG <: Random.AbstractRNG}
    best::SACandidate{Tx,Tf}
    current::SACandidate{Tx,Tf}
    rng::TRNG
    best_iterations::Vector{Tuple{Int,Float64,Tf}}
    best_iteration::Int = 0
    current_iteration::Int = 0
    i::Int = 1
    max_iter::Int = 1000
    t::Float64 = 10000.0
    initial_t::Float64 = 10000.0
    min_t::Float64 = 0.01
    alpha::Float64 = 0.9992
    trials_per_iter::Int = 3
    acceptance_rule = "SA"# threshold_accepting_acceptance_rule
    running_average_ratio = 0.9
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
