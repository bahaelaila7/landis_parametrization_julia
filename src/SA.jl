module SA
    struct SACandidate{Tx,Tf}
        x::Tx
        fx::Tf
    end
    mutable struct SAState{Tx,Tf}
        best::SACandidate{Tx,Tf}
        best_iteration::Int
        temperature::Float64
        current_iteration::Int
        max_iter::Int
    end
    function simulated_annealing(state::SAState, get_neighbor, max_iter=1000, initial_temp = 100000, alpha = 0.992, trials_per_iter=3)
        if state === nothing
            start_iter = 1
        else
            start_iter = state.current_iteration
        end
        
        for i in start_iter:max_iter
            println(i)
        end
    end
end
