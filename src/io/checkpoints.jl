function load_best_params(db_path::String; iteration::Union{Nothing,Int}=nothing)
  db = DuckDB.DB(db_path)
  con = DuckDB.connect(db)
  result = isnothing(iteration) ?
           DuckDB.execute(con, "SELECT params_blob FROM total_loss ORDER BY total_loss ASC, iteration ASC LIMIT 1") |> DataFrame :   # (iteration breaks ties of equal losses)
           DuckDB.execute(con, "SELECT params_blob FROM total_loss WHERE iteration = ? ORDER BY iteration LIMIT 1", [iteration]) |> DataFrame
  close(db)
  isempty(result) && return nothing
  return Serialization.deserialize(IOBuffer(result.params_blob[1]))
end

# Non-dominated sort of (W,AGB) candidates → ordering by Pareto front, then aggregate within front.
function _dominance_order(W::Vector{Float64}, A::Vector{Float64}, G::Vector{Float64})
  n = length(W)
  dom(i, j) = (W[i] <= W[j] && A[i] <= A[j]) && (W[i] < W[j] || A[i] < A[j])
  front = fill(0, n); remaining = Set(1:n); fr = 0
  while !isempty(remaining)
    fr += 1
    nd = [i for i in remaining if !any(j -> dom(j, i), remaining)]
    for i in nd; front[i] = fr; delete!(remaining, i); end
  end
  sortperm([(front[i], G[i]) for i in 1:n])   # front rank, then aggregate within front
end

# rank = :dominance (non-dominated sort over ΣW,ΣAGB — exchange-rate-free, keeps the whole tradeoff;
#        the scalar aggregate is ~pure-AGB so don't rank by it), :balanced (min-max ΣW+ΣAGB), or :aggregate.
function load_sobol_candidates(db_path::String; top_frac::Float64=0.5, run_id::Union{Nothing,String}=nothing, rank::Symbol=:dominance)
  isfile(db_path) || return []
  db = DuckDB.DB(db_path)
  con = DuckDB.connect(db)
  try
    cols = Set(string(r.name) for r in (DuckDB.execute(con, "PRAGMA table_info('sobol_results')") |> DataFrame |> eachrow))
    hasWA = ("sumW" in cols) && ("sumAGB" in cols)
    sel = hasWA ? "mean_loss, sumW, sumAGB, params_blob" : "mean_loss, params_blob"
    where = isnothing(run_id) ? "" : " WHERE run_id = ?"
    tie = ("sobol_idx" in cols) ? ", sobol_idx ASC" : ""   # equal losses keep a defined order (the candidates are used by position)
    result = DuckDB.execute(con, "SELECT $sel FROM sobol_results$where ORDER BY mean_loss ASC$tie", isnothing(run_id) ? [] : [String(run_id)]) |> DataFrame
    isempty(result) && return []
    G = Float64.(result.mean_loss)
    order = if rank != :aggregate && hasWA
      W = Float64.(result.sumW); A = Float64.(result.sumAGB)
      if rank == :balanced
        rs(x) = (lo = minimum(x); hi = maximum(x); hi > lo ? (x .- lo) ./ (hi - lo) : zero(x))
        sortperm(rs(W) .+ rs(A))
      else
        _dominance_order(W, A, G)   # :dominance
      end
    else
      sortperm(G)                   # :aggregate
    end
    n_keep = max(1, round(Int, nrow(result) * top_frac))
    keep = order[1:n_keep]
    @info "Loaded $(n_keep) Sobol candidates (top $(round(Int, top_frac*100))% of $(nrow(result)), rank=$rank)"
    return [Serialization.deserialize(IOBuffer(result.params_blob[i])) for i in keep]
  catch e
    @warn "load_sobol_candidates failed" exception = e
    return []
  finally
    close(db)
  end
end
