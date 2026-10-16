module Common
# Shared search utilities (Step 5a). One implementation each of
#   - Pareto dominance, fast non-dominated sorting (optionally niched), NSGA-II crowding distance,
#   - the Pareto archive (dominance filter + capped insert with a pluggable eviction rule),
#   - the (1+1)-CMA success-rule / evolution-path / covariance kernels (Igel, Hansen & Roth 2007),
#   - the search interface `ask / tell! / state / save_state / load_state` that igelmo and gdigel implement.
# Everything here is allocation-light, type-stable plain Julia and has no dependency on the MO candidate types
# (those stay in MOLBSA so existing JLD2 checkpoints keep loading).

export dominates, nd_fronts, crowding_distance, pareto_prune!, pareto_insert!, evict_most_crowded, evict_worst,
       cma11_psucc, cma11_sigma, cma11_path, cma11_cov, AbstractSearch

# ---- search interface ---------------------------------------------------------------------------------------------------
"""
    AbstractSearch

The interface of a resumable search engine (implemented by igelmo and gdigel):

  - `ask(s, ...)`        → the next batch of u-vectors to evaluate;
  - `tell!(s, ...)`      → absorb the evaluated batch (advances the generation);
  - `state(s)`           → the serializable engine state (what a checkpoint contains);
  - `save_state(s, path)`/`load_state(T, path)` → checkpoint round trip: `load_state(T, save_state(s, path))` continues exactly as `s` would.
"""
abstract type AbstractSearch end
function ask end
function tell! end
function state end
function save_state end
function load_state end

# ---- dominance / non-dominated sorting / crowding -------------------------------------------------------------------------
"Pareto dominance for minimization: `a` dominates `b` iff a ≤ b everywhere and a < b somewhere."
@inline function dominates(a::AbstractVector, b::AbstractVector)::Bool
  strictly = false
  @inbounds for k in eachindex(a)
    a[k] > b[k] && return false
    a[k] < b[k] && (strictly = true)
  end
  return strictly
end

"""
    nd_fronts(objs; compete = (p, q) -> true) -> Vector{Int}

Fast non-dominated sort (Deb et al. 2002): the front rank (1 = non-dominated) of every objective vector. `compete(p, q)` restricts which pairs may
dominate one another (niched domination: igelmo with `igel_niche_radius > 0`); the default is all pairs.
"""
function nd_fronts(objs::AbstractVector, compete::F=(p, q) -> true) where {F}
  m = length(objs); S = [Int[] for _ in 1:m]; ndom = zeros(Int, m); rank = zeros(Int, m)
  for p in 1:m, q in 1:m
    p == q && continue
    compete(p, q) || continue
    if dominates(objs[p], objs[q]); push!(S[p], q)
    elseif dominates(objs[q], objs[p]); ndom[p] += 1; end
  end
  cur = [i for i in 1:m if ndom[i] == 0]; rr = 1
  while !isempty(cur)
    for i in cur; rank[i] = rr; end
    nxt = Int[]
    for p in cur, q in S[p]; ndom[q] -= 1; ndom[q] == 0 && push!(nxt, q); end
    cur = nxt; rr += 1
  end
  return rank
end

"NSGA-II crowding distance of a set of points (objective vectors OR decision vectors; higher = more isolated; per-objective boundary points = Inf)."
function crowding_distance(pts::AbstractVector)::Vector{Float64}
  m = length(pts); m <= 2 && return fill(Inf, m)
  M = length(pts[1]); cd = zeros(Float64, m); vals = Vector{Float64}(undef, m)
  for o in 1:M
    @inbounds for i in 1:m; vals[i] = Float64(pts[i][o]); end
    ord = sortperm(vals); span = vals[ord[end]] - vals[ord[1]]
    cd[ord[1]] = Inf; cd[ord[end]] = Inf
    span > 0 && for r in 2:m-1; cd[ord[r]] += (vals[ord[r+1]] - vals[ord[r-1]]) / span; end
  end
  return cd
end

# ---- Pareto archive ---------------------------------------------------------------------------------------------------------
"""
    pareto_prune!(archive, cobjs, objs_of; dom = dominates) -> Bool

First two steps of every archive insert: returns `false` (archive untouched) when some member dominates `cobjs`; otherwise removes the members
`cobjs` dominates and returns `true`. `objs_of(member)` extracts a member's objective vector.
"""
function pareto_prune!(archive::AbstractVector, cobjs, objs_of::F, dom::D=dominates) where {F,D}
  for m in archive; dom(objs_of(m), cobjs) && return false; end
  filter!(m -> !dom(cobjs, objs_of(m)), archive)
  return true
end

"Eviction rule: index of the most crowded member (smallest NSGA-II crowding distance over the members' objectives)."
evict_most_crowded(archive, objs_of::F) where {F} = argmin(crowding_distance([objs_of(m) for m in archive]))
"Eviction rule: index of the member with the worst (largest) aggregate."
evict_worst(archive, aggregate_of::F) where {F} = argmax(i -> aggregate_of(archive[i]), eachindex(archive))

"""
    pareto_insert!(archive, cand, cap, objs_of, evict; dom = dominates) -> Bool

Insert `cand` into the capped Pareto archive: dropped when dominated, prunes what it dominates, and over `cap` evicts `archive[evict(archive)]`.
Returns whether `cand` was accepted (it may itself be the evicted member, as in the historical drivers).
"""
function pareto_insert!(archive::AbstractVector, cand, cap::Integer, objs_of::F, evict::E, dom::D=dominates) where {F,E,D}
  pareto_prune!(archive, objs_of(cand), objs_of, dom) || return false
  push!(archive, cand)
  length(archive) > cap && deleteat!(archive, evict(archive))
  return true
end

# ---- (1+1)-CMA-ES kernels (Igel, Hansen & Roth 2007; constants live with the caller) ---------------------------------------------
"Smoothed success rate: p ← (1−c_p)·p + c_p·succ."
@inline cma11_psucc(p_succ, c_p, succ) = (1 - c_p) * p_succ + c_p * succ
"Step-size success rule: σ ← σ·exp((p_succ − p_target)/(1 − p_target)/d)."
@inline cma11_sigma(sigma, p_succ, d, p_target) = sigma * exp((1 / d) * (p_succ - p_target) / (1 - p_target))
"Evolution path: p_c ← (1−c_c) p_c + √(c_c(2−c_c))·step while p_succ < p_thresh (`active`), else only the decay."
@inline cma11_path(p_c, step, active::Bool, c_c) = active ? (1 - c_c) .* p_c .+ sqrt(c_c * (2 - c_c)) .* step : (1 - c_c) .* p_c
"Rank-one covariance update of one block (`pcb` = the block's slice of the updated path)."
@inline cma11_cov(Cb, pcb, active::Bool, c_c, c_cov) =
  active ? (1 - c_cov) .* Cb .+ c_cov .* (pcb * pcb') :
           (1 - c_cov) .* Cb .+ c_cov .* (pcb * pcb' .+ c_c * (2 - c_c) .* Cb)

end # module Common
