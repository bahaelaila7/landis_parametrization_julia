module LBSA
import Random
import DataStructures
import Statistics

export LBSACandidate, LBSAState, simulated_annealing_acceptance_rule, threshold_accepting_acceptance_rule, search_cmp!, search_update_rule!

@enum SearchMethod SimulatedAnnealing ThresholdAccepting
Base.@kwdef struct LBSACandidate{Tx,Tf}
    x::Tx
    fx::Tf
end

@inline function is_search_over(state)::Bool
    state.i >= state.max_iter
end
@inline function search_cmp!(next, state)
    state.i += 1
    next_fit = convert(Float64, next.fx)
    cur_fit = convert(Float64, state.current.fx)
    best_fit = convert(Float64, state.best.fx)
    diff_fit = next_fit - cur_fit
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
		state._t_list .= t_mean + 2*t_std
		state._t_oldest_idx = 1
		state._t_max_idx = 1
	end

        state._initial_t_list = state._t_list[:]
    elseif state._hill_climbing_best > 0
	state._hill_climbing_best -= 1
	accept = next_fit < cur_fit
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
			    state._t_oldest_idx = (state._t_oldest_idx  % length(state._t_list)) + 1
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
                    if next_fit > best_fit && state._frozen_no_best_reheats >= state.restart_after_no_best_reheats
                        state._frozen_no_best_reheats = 0
                        state._warming_up = true
                        state.warm_up_record_uphill_only = true
                        state.current = rand(state.best_iterations)[3] #state.best
                        state._t_list = [] #state._initial_t_list[:]
                        #state._t_max_idx = argmax(state._t_list)
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
        cur_fit = next_fit
        if cur_fit < best_fit
            state._frozen_no_best_reheats = 0
            best_fit = cur_fit
            state.best = state.current
            state.best_iteration = state.i
            push!(state.best_iterations, (state.i, best_fit, state.best))
	    state._hill_climbing_best = state.hill_climbing_upon_new_best
            return true
        end
    end
    return false
end

Base.@kwdef mutable struct LBSAState{Tx,Tf,TRNG<:Random.AbstractRNG}
    best::LBSACandidate{Tx,Tf}
    current::LBSACandidate{Tx,Tf}
    rng::TRNG
    best_iterations::Vector{Tuple{Int,Float64,LBSACandidate{Tx,Tf}}}
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
    restart_after_no_best_reheats::Int = 10
    reheat_factor::Float64 = 1.5
    reheat_max_only::Bool = false
    replace_oldest_instead_of_max::Bool = true
    search_method::SearchMethod = SimulatedAnnealing
    hill_climbing_upon_new_best::Int=0

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



    running_average_ratio = 0.9
    diff_avg::Float64 = 0.0
    prob_avg::Float64 = 0.0
end
function LBSAState(best::LBSACandidate{Tx,Tf}, current::LBSACandidate{Tx,Tf}, rng::TRNG; best_iterations=Tuple{Int,Float64,LBSACandidate{Tx,Tf}}[], initial_acceptance_prob=0.9, _neg_inv_lnp0=0.0, kwargs...) where {Tx,Tf, TRNG <: Random.AbstractRNG}
    LBSAState{Tx,Tf,TRNG}(; best=best, current=current, rng=rng,
        best_iterations=best_iterations,
        initial_acceptance_prob=initial_acceptance_prob,
        _neg_inv_lnp0=-1.0 / log(initial_acceptance_prob),
        kwargs...)
end


function search(state::Union{Nothing,Some{LBSAState}}, get_neighbor, get_fitness)::LBSAState
    if isnothing(state)
        first = get_neighbor(nothing; rng=state.rng)
        first_fit = get_fitness(first)
        cur = LBSACandidate(first, first_fit)
        state = LBSAState(cur, cur)
    end

    start_i = state.i
    for i in start_i:state.max_iter
        state.i = i
        next_cand = get_neighbor(cur; rng=state.rng)

        next = LBSACandidate(next_cand, get_fitness(next_cand))

        search_cmp!(next, state)
        if search_update_rule!(state)
            break
        end

    end
end
end
