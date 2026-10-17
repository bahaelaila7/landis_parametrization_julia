struct WriterJob{State}
  save_ckpt::Bool          # persist a checkpoint this generation (MO-CMA-ES: on ANY archive change; others: on new best)
  state::State
  splots::DataFrame
  merged_sites_state::DataFrame
  emp_sample::Union{Nothing,DataFrame}
  sim_sample::Union{Nothing,DataFrame}
  emp_sample_val::Union{Nothing,DataFrame}
  sim_sample_val::Union{Nothing,DataFrame}
end

const STOP = :stop

function start_writer(::Type{State}, output_dir::AbstractString; buffer_size::Int=8) where {State}
  ch = Channel{Union{WriterJob{State},Symbol}}(buffer_size)

  task = Threads.@spawn begin
    #Plots.default(show = false)
    try
      for job in ch
        job === STOP && break
        state = job.state
        if job.save_ckpt
          try
            @info "New best: $(convert(Float64,state.best.fx))" iter = state.i
            mkpath(output_dir)
            JLD2.save_object(joinpath(output_dir, "search_state@$(state.i).jld2"), state)
            PU.save_json(joinpath(output_dir, "best_params@$(state.i).json"), state.best.x)
            JLD2.save_object(joinpath(output_dir, "best_params@$(state.i).jld2"), state.best.x)
          catch e
            @error "writer: save failed" iter = state.i exception = (e, catch_backtrace())
          end

          if !isnothing(job.emp_sample) && !isnothing(job.sim_sample)
            try
              generate_plots(job.emp_sample, job.sim_sample, "training_$(state.i)", convert(Float64, state.best.fx), output_dir)
            catch e
              @error "writer: generate_plots (train) failed" iter = state.i exception = (e, catch_backtrace())
            end
          end
          if !isnothing(job.emp_sample_val) && !isnothing(job.sim_sample_val)
            try
              generate_plots(job.emp_sample_val, job.sim_sample_val, "validation_$(state.i)", convert(Float64, state.best.fx), output_dir)
            catch e
              @error "writer: generate_plots (val) failed" iter = state.i exception = (e, catch_backtrace())
            end
          end
        else
          @info ("Best@$(state.best_iteration): $(convert(Float64, state.best.fx)), Avg diff: $(state.diff_avg), Temp: $(state.t), ratio $(state.diff_avg/state.t), Prob: $(state.prob_avg)")
        end
      end
    catch e
      @error "writer: fatal" exception = (e, catch_backtrace())
      rethrow()
    end
  end

  return ch, task
end

function stop_writer(ch::Channel, task::Task)
  put!(ch, STOP)
  close(ch)
  wait(task)
end
