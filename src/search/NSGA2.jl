module NSGA2
# Vanilla NSGA-II (Deb, Pratap, Agarwal & Meyarivan 2002): the canonical μ-population multi-objective GA.
# Each generation: binary-tournament selection on (front rank, crowding distance) → SBX crossover +
# polynomial mutation in u∈[0,1]^d → λ offspring; then elitist (μ+λ) environmental selection via fast
# non-dominated sorting + crowding keeps the best μ. Uses the SAME MOCandidate/MOFitness, capped Pareto
# archive and start_mo_writer / mo_gen_finalize! interface (representative/current/archive/i/n_evals fields)
# as the CMA-ES-family engines, so it slots straight into the identical comparison pipeline — the closest
# purely-evolutionary sibling to the existing setup.
import Random
import ..MOLBSA: MOFitness, MOCandidate, dominates
import ..Common: crowding_distance, nd_fronts, pareto_insert!, evict_most_crowded

export NSGA2State, ask, tell!, is_search_over

Base.@kwdef mutable struct NSGA2State{Tx,TRNG<:Random.AbstractRNG}
  # ---- MOLBSA-style bookkeeping (field names match start_mo_writer / mo_gen_finalize!) ----
  representative::MOCandidate{Tx}                # min-aggregate archive member (logging/checkpoints)
  current::MOCandidate{Tx}                       # min-aggregate parent of the current population
  archive::Vector{MOCandidate{Tx}}               # capped Pareto-non-dominated set
  archive_cap::Int = 200
  rng::TRNG
  best_iterations::Vector{Tuple{Int,Float64,MOCandidate{Tx}}}   # pass explicitly (Tx isn't bound in @kwdef's default)
  best_iteration::Int = 0
  current_iteration::Int = 0
  i::Int = 0                                     # generation counter → checkpoint filenames
  n_evals::Int = 0
  max_iter::Int = 1_000_000
  # benign writer log fields (non-checkpoint branch reads diff_avg/t/prob_avg)
  t::Float64 = 0.0
  diff_avg::Float64 = 0.0
  prob_avg::Float64 = 0.0
  # ---- NSGA-II population (u∈[0,1]^d parents + their decoded candidates) ----
  pop_u::Vector{Vector{Float64}}
  pop::Vector{MOCandidate{Tx}}
  n::Int                                         # search-space dimensionality d
  mu::Int
  lambda::Int
  eta_c::Float64 = 20.0                          # SBX distribution index
  eta_m::Float64 = 20.0                          # polynomial-mutation distribution index
  p_c::Float64 = 0.9                             # crossover probability (per pair)
  p_m::Float64 = -1.0                            # per-gene mutation prob (<0 ⇒ 1/d)
end

is_search_over(st::NSGA2State)::Bool = st.i >= st.max_iter

const _crowding = crowding_distance   # NSGA-II crowding distance over objective vectors (boundary points → Inf)
const _fronts = nd_fronts           # fast non-dominated sort → front rank (1 = best)

# elitist (μ+λ) environmental selection: indices of the best μ of the combined set by (rank, −crowding).
function _select(objs::Vector{<:AbstractVector}, μ::Int)::Vector{Int}
  rank = _fronts(objs); chosen = Int[]
  for rr in 1:maximum(rank)
    idx = findall(==(rr), rank)
    if length(chosen) + length(idx) <= μ
      append!(chosen, idx)
    else                                         # partial front: keep the most-isolated (highest crowding)
      ord = sortperm(_crowding(objs[idx]); rev=true)
      append!(chosen, idx[ord[1:(μ - length(chosen))]]); break
    end
    length(chosen) >= μ && break
  end
  return chosen
end

# SBX crossover (Deb & Agarwal 1995), bounded to [0,1]; returns two children.
function _sbx(p1::Vector{Float64}, p2::Vector{Float64}, ηc::Float64, rng)
  d = length(p1); c1 = copy(p1); c2 = copy(p2)
  for i in 1:d
    rand(rng) > 0.5 && continue                                 # cross ~half the genes
    x1 = p1[i]; x2 = p2[i]
    abs(x1 - x2) < 1e-14 && continue
    xl, xh = minmax(x1, x2); u = rand(rng)
    β = 1.0 + 2.0 * xl / (xh - xl)                              # toward lower bound 0
    α = 2.0 - β^(-(ηc + 1.0))
    βq = u <= 1.0/α ? (u*α)^(1.0/(ηc+1.0)) : (1.0/(2.0 - u*α))^(1.0/(ηc+1.0))
    ch1 = 0.5 * ((xl + xh) - βq * (xh - xl))
    β = 1.0 + 2.0 * (1.0 - xh) / (xh - xl)                      # toward upper bound 1
    α = 2.0 - β^(-(ηc + 1.0))
    βq = u <= 1.0/α ? (u*α)^(1.0/(ηc+1.0)) : (1.0/(2.0 - u*α))^(1.0/(ηc+1.0))
    ch2 = 0.5 * ((xl + xh) + βq * (xh - xl))
    ch1 = clamp(ch1, 0.0, 1.0); ch2 = clamp(ch2, 0.0, 1.0)
    if rand(rng) < 0.5; c1[i] = ch2; c2[i] = ch1 else c1[i] = ch1; c2[i] = ch2 end
  end
  return c1, c2
end

# polynomial mutation (Deb), bounded to [0,1].
function _polymut!(x::Vector{Float64}, ηm::Float64, pm::Float64, rng)
  mp = 1.0/(ηm + 1.0)
  for i in eachindex(x)
    rand(rng) > pm && continue
    xi = x[i]; u = rand(rng)
    if u < 0.5
      xy = xi                                                    # 1 - delta1, delta1 = xi (bounds 0..1)
      δq = (2.0*u + (1.0 - 2.0*u) * xy^(ηm + 1.0))^mp - 1.0
    else
      xy = 1.0 - xi                                              # 1 - delta2, delta2 = 1 - xi
      δq = 1.0 - (2.0*(1.0 - u) + 2.0*(u - 0.5) * xy^(ηm + 1.0))^mp
    end
    x[i] = clamp(xi + δq, 0.0, 1.0)
  end
  return x
end

@inline _wins(a, b, rank, cd) = rank[a] < rank[b] || (rank[a] == rank[b] && cd[a] > cd[b]) ? a : b

# λ offspring u-vectors via binary-tournament selection + SBX + polynomial mutation over the current pop.
function ask(st::NSGA2State)::Vector{Vector{Float64}}
  objs = [c.fx.objectives for c in st.pop]
  rank = _fronts(objs); cd = zeros(Float64, length(st.pop))
  for rr in 1:maximum(rank); idx = findall(==(rr), rank); cd[idx] .= _crowding(objs[idx]); end
  pm = st.p_m < 0 ? 1.0/st.n : st.p_m; μ = length(st.pop)
  tour() = _wins(rand(st.rng, 1:μ), rand(st.rng, 1:μ), rank, cd)
  offs = Vector{Vector{Float64}}(); sizehint!(offs, st.lambda)
  while length(offs) < st.lambda
    p1 = st.pop_u[tour()]; p2 = st.pop_u[tour()]
    c1, c2 = rand(st.rng) < st.p_c ? _sbx(p1, p2, st.eta_c, st.rng) : (copy(p1), copy(p2))
    _polymut!(c1, st.eta_m, pm, st.rng); _polymut!(c2, st.eta_m, pm, st.rng)
    push!(offs, c1); length(offs) < st.lambda && push!(offs, c2)
  end
  return offs
end

# insert into the capped Pareto archive + track representative / return is-new-best (mirrors MOLBSA.update_archive!).
function _archive!(st::NSGA2State, cand::MOCandidate)::Bool
  pareto_insert!(st.archive, cand, st.archive_cap, m -> m.fx.objectives, a -> evict_most_crowded(a, m -> m.fx.objectives)) || return false
  improved = cand.fx.aggregate < st.representative.fx.aggregate
  if improved
    st.representative = cand; st.best_iteration = st.i
    push!(st.best_iterations, (st.i, cand.fx.aggregate, cand))
  end
  return improved
end

# One generation: fold offspring into the archive, then elitist (μ+λ) selection to the next μ population.
function tell!(st::NSGA2State, off_fxs::Vector{MOFitness}, off_params::Vector, off_us::Vector{Vector{Float64}})::Bool
  offs = [MOCandidate(off_params[k], off_fxs[k]) for k in eachindex(off_fxs)]
  is_new_best = false
  for c in offs; is_new_best |= _archive!(st, c); end
  comb = vcat(st.pop, offs); comb_u = vcat(st.pop_u, off_us)
  sel = _select([c.fx.objectives for c in comb], st.mu)
  st.pop = comb[sel]; st.pop_u = comb_u[sel]
  st.current = st.pop[argmin([Float64(c.fx.aggregate) for c in st.pop])]
  st.i += 1; st.current_iteration = st.i
  return is_new_best
end

end # module
