module IgelMOCMAES
# Igel/Hansen/Roth (2007) MO-CMA-ES: the canonical *population-based* multi-objective CMA-ES, in
# contrast to the single-distribution MOCMAES (search/MOCMAES.jl). Here the population is μ
# independent (1+1)-CMA-ES individuals — each with its own step size σ, success rate p_succ,
# covariance C and evolution path p_c. Every generation each parent produces one offspring; the
# combined 2μ set is ranked by non-dominated sorting (ties broken by NSGA-II crowding distance) and
# the best μ survive. Because individuals specialise, the population spreads across the Pareto set —
# covering extremes / multiple optima far better than one converging distribution.
import Random
import LinearAlgebra
import Sobol
const LA = LinearAlgebra
import ..MOLBSA: MOFitness, MOCandidate, dominates

export IgelState, ask, tell!, is_search_over

# Fraction of reseeds (see `ask`) drawn UNIFORM-RANDOM in u-space instead of from the shared Sobol sequence.
# A module-level Ref (NOT an IgelState field) so it can be flipped on a stop→resume WITHOUT changing the
# serialized state layout (old checkpoints still deserialize). 0.0 ⇒ all-Sobol (backward-compatible: the
# guard short-circuits, so no extra RNG draw). Set from `igel_reseed_random_frac` at launch.
const RESEED_RANDOM_FRAC = Ref{Float64}(0.0)

# Live overrides for the two RESEED knobs that are otherwise SERIALIZED IgelState fields (so a resume would
# restore the checkpoint's value and ignore the config). >=0 ⇒ use this instead of the struct field; the sentinel
# <0 ⇒ fall back to the struct field (default: normal resume unchanged). Set from igel_{reseed_sigma,maturity}_override
# at launch, so you can RETUNE reseed rate / maturity shield on a stop→resume without a fresh run.
const RESEED_SIGMA_OVERRIDE = Ref{Float64}(-1.0)
const MATURITY_OVERRIDE     = Ref{Int}(-1)
@inline _reseed_sigma(st) = RESEED_SIGMA_OVERRIDE[] >= 0 ? RESEED_SIGMA_OVERRIDE[] : st.reseed_sigma
@inline _maturity(st)     = MATURITY_OVERRIDE[]     >= 0 ? MATURITY_OVERRIDE[]     : st.maturity_period

# Sampling square-root of the covariance block. false (default) => eigendecomposition V*sqrt(Lambda) (as before).
# true => Cholesky factor L (C = L*L') — SAME C-based algorithm and SAME N(0,C) sampling distribution, but the
# factorization is ~n^3/3 flops vs a symmetric eigendecomposition's much larger constant, so `ask()` is cheaper
# for large blocks (esp. cmaes_single_cov). A module Ref (not a state field) so it's toggleable on resume; C is
# still what's stored, so checkpoints are unchanged. Set from `igel_cholesky` at launch.
const USE_CHOLESKY = Ref{Bool}(false)

# y ~ N(0, Cb): return the transform applied to a fresh standard-normal draw. Cholesky path falls back to a
# jittered retry then eigen if Cb has drifted numerically non-PD (the C update is a PSD combination, but Float
# round-off can nudge it), so it always returns a valid square-root.
@inline function _sample_transform(Cb::AbstractMatrix, rng, m::Int)
  z = randn(rng, m)
  if USE_CHOLESKY[]
    F = LA.cholesky(LA.Symmetric(Cb); check=false)
    LA.issuccess(F) && return F.L * z
    jit = 1e-10 * (sum(LA.diag(Cb)) / m + 1e-30)
    for _ in 1:6
      F = LA.cholesky(LA.Symmetric(Cb + jit * LA.I); check=false)
      LA.issuccess(F) && return F.L * z
      jit *= 10
    end
    # still not PD => fall through to the eigen square-root
  end
  E = LA.eigen(LA.Symmetric(Cb))
  return (E.vectors * LA.Diagonal(sqrt.(max.(E.values, 1e-30)))) * z
end

# one (1+1)-CMA-ES individual (search point + self-adaptive strategy parameters), in u-space.
# `mature_at` is the generation at which the individual becomes subject to normal selection; until
# then (a freshly re-seeded individual) it is protected from removal so it can descend to its basin.
mutable struct Individual
  x::Vector{Float64}
  sigma::Float64
  p_succ::Float64
  C::Matrix{Float64}
  p_c::Vector{Float64}
  fx::MOFitness
  mature_at::Int
end

mutable struct IgelState{Tx,TRNG<:Random.AbstractRNG}
  pop::Vector{Individual}                      # μ parents
  # MOLBSA-style bookkeeping (field names match start_mo_writer)
  representative::MOCandidate{Tx}
  current::MOCandidate{Tx}
  archive::Vector{MOCandidate{Tx}}
  archive_cap::Int
  rng::TRNG
  best_iterations::Vector{Tuple{Int,Float64,MOCandidate{Tx}}}
  best_iteration::Int
  current_iteration::Int
  i::Int
  n_evals::Int
  max_iter::Int
  t::Float64
  diff_avg::Float64
  prob_avg::Float64
  # static (1+1)-CMA constants (Igel/Hansen defaults, one offspring per parent)
  n::Int
  mu::Int
  d::Float64
  p_target::Float64
  c_p::Float64
  p_thresh::Float64
  c_c::Float64
  c_cov::Float64
  niche_radius::Float64                          # >0 ⇒ decision-space niching (niched domination); 0 ⇒ standard Igel
  sigma0::Float64
  reseed_sigma::Float64                          # >0 ⇒ re-seed an individual once its σ drops below this (IPOP-like re-exploration)
  maturity_period::Int                           # generations a re-seed is shielded from removal so it can converge
  _sobol::Sobol.SobolSeq                         # space-filling source for re-seed locations
  _reseed::Vector{Bool}                          # which offspring this generation are fresh re-seeds (set by ask)
  _off::Vector{Individual}                       # offspring of the current generation (set by ask)
  blocks::Vector{Vector{Int}}                    # block-diagonal covariance partition (1 block ⇒ full (1+1)-CMA)
end

# init_us: μ starting u-vectors; init_cands: their evaluated MOCandidates (params + MOFitness)
function IgelState(init_us::Vector{Vector{Float64}}, init_cands::Vector{MOCandidate{Tx}}, rng::TRNG;
                   sigma0::Float64=0.3, archive_cap::Int=200, max_iter::Int=1_000_000, niche_radius::Float64=0.0, reseed_sigma::Float64=0.0, maturity_period::Int=0,
                   init_mature::Bool=false,   # true ⇒ the μ seeds start shielded for `maturity_period` gens (phase 1: mature independently; phase 2: compete)
                   blocks::Union{Nothing,Vector{Vector{Int}}}=nothing) where {Tx,TRNG<:Random.AbstractRNG}
  μ = length(init_us); n = length(init_us[1])
  blk = isnothing(blocks) ? [collect(1:n)] : blocks
  d = 1 + n/2
  p_target = 1 / (5 + sqrt(1.0)/2)              # λ=1 offspring per parent
  c_p = p_target / (2 + p_target)
  p_thresh = 0.44
  c_c = 2 / (n + 2)
  c_cov = 2 / (n^2 + 6)
  m0 = init_mature ? maturity_period : 0   # phase-1 shield window for the seeds (0 ⇒ canonical: seeds compete immediately)
  pop = [Individual(copy(init_us[k]), sigma0, p_target, Matrix{Float64}(LA.I, n, n), zeros(n), init_cands[k].fx, m0) for k in 1:μ]
  rep = init_cands[argmin(c.fx.aggregate for c in init_cands)]
  sob = Sobol.SobolSeq(n); for _ in 1:μ; Sobol.next!(sob); end   # skip past the init points so re-seeds explore new regions
  st = IgelState{Tx,TRNG}(pop, rep, rep, MOCandidate{Tx}[], archive_cap, rng,
    Tuple{Int,Float64,MOCandidate{Tx}}[], 0, 0, 0, μ, max_iter, sigma0, 0.0, 0.0,
    n, μ, d, p_target, c_p, p_thresh, c_c, c_cov, niche_radius, sigma0, reseed_sigma, maturity_period, sob, falses(μ), Individual[], blk)
  for c in init_cands; _archive!(st, c); end
  return st
end

@inline is_search_over(st::IgelState)::Bool = st.i >= st.max_iter

# Sample one offspring per parent from N(x, σ²C); returns the μ offspring u-vectors (clamped to box).
# With reseed_sigma>0, a converged parent (σ < reseed_sigma) instead emits a fresh Sobol point with a
# fresh (1+1) strategy — re-exploration that gives the fixed population IPOP-like cumulative coverage.
function ask(st::IgelState)::Vector{Vector{Float64}}
  n = st.n; offs = Vector{Vector{Float64}}(undef, st.mu); st._off = Vector{Individual}(undef, st.mu); st._reseed = falses(st.mu)
  rs = _reseed_sigma(st)                              # live-overridable reseed threshold (else the struct field)
  for k in 1:st.mu
    ind = st.pop[k]
    if rs > 0 && ind.sigma < rs
      # re-explore from a fresh point: RESEED_RANDOM_FRAC of the time uniform-random in the box, else the
      # shared low-discrepancy Sobol point. Guard short-circuits when frac==0 → no RNG draw, Sobol as before.
      xo = (RESEED_RANDOM_FRAC[] > 0 && rand(st.rng) < RESEED_RANDOM_FRAC[]) ?
             rand(st.rng, n) : clamp.(Sobol.next!(st._sobol), 0.0, 1.0)
      offs[k] = xo
      st._off[k] = Individual(xo, st.sigma0, st.p_target, Matrix{Float64}(LA.I, n, n), zeros(n), ind.fx, 0)  # fresh strategy
      st._reseed[k] = true
    else
      xo = copy(ind.x)                              # per-block: y_b ~ N(0, C[b,b]), assembled into xo
      for idx in st.blocks
        yb = _sample_transform(ind.C[idx, idx], st.rng, length(idx))   # eigen (default) or Cholesky sqrt (USE_CHOLESKY)
        @inbounds for (j, gi) in enumerate(idx); xo[gi] = ind.x[gi] + ind.sigma * yb[j]; end
      end
      xo = clamp.(xo, 0.0, 1.0)
      offs[k] = xo
      st._off[k] = Individual(xo, ind.sigma, ind.p_succ, copy(ind.C), copy(ind.p_c), ind.fx, ind.mature_at)
    end
  end
  return offs
end

# NSGA-II crowding distance over a set of vectors (objectives OR decision points). Boundary → Inf.
function _crowding(pts::Vector{<:AbstractVector{<:Real}})::Vector{Float64}
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

# fast non-dominated sort → front rank (1 = best). With niche_radius>0, two individuals only COMPETE
# (can dominate one another) when within `r` in decision space — so explorers descending toward a
# different optimum aren't killed by an individual already converged at another basin.
function _fronts(Q::Vector{Individual}, r::Float64)::Vector{Int}
  m = length(Q); objs = [ind.fx.objectives for ind in Q]; r2 = r*r
  S = [Int[] for _ in 1:m]; ndom = zeros(Int, m); rank = zeros(Int, m)
  for p in 1:m, q in 1:m
    p == q && continue
    (r > 0 && sum((Q[p].x .- Q[q].x).^2) >= r2) && continue   # only compete within the niche
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

# total preorder key over the combined set: (front rank, −crowding-within-front). Crowding is taken
# in DECISION space when niching (to spread across basins), else in objective space (standard Igel).
function _keys(Q::Vector{Individual}, r::Float64)
  rank = _fronts(Q, r); cd = zeros(Float64, length(Q))
  for rr in 1:maximum(rank)
    idx = findall(==(rr), rank)
    cd[idx] .= r > 0 ? _crowding([Q[i].x for i in idx]) : _crowding([Q[i].fx.objectives for i in idx])
  end
  return [(rank[i], -cd[i]) for i in eachindex(Q)]
end

@inline function _update_stepsize!(ind::Individual, succ::Float64, st::IgelState)
  ind.p_succ = (1 - st.c_p) * ind.p_succ + st.c_p * succ
  ind.sigma *= exp((1 / st.d) * (ind.p_succ - st.p_target) / (1 - st.p_target))
end

# BLOCK-DIAGONAL (1+1)-CMA covariance update: p_c evolves over the full vector, but the rank-1 C update
# is applied only WITHIN each block's submatrix (off-block entries stay zero), so each parameter group
# keeps its own small covariance. With one block this is the standard full-C update.
@inline function _update_cov!(ind::Individual, step::Vector{Float64}, st::IgelState)
  active = ind.p_succ < st.p_thresh
  ind.p_c = active ? (1 - st.c_c) .* ind.p_c .+ sqrt(st.c_c * (2 - st.c_c)) .* step : (1 - st.c_c) .* ind.p_c
  for idx in st.blocks
    pcb = @view ind.p_c[idx]
    Cb = @view ind.C[idx, idx]
    if active
      ind.C[idx, idx] = (1 - st.c_cov) .* Cb .+ st.c_cov .* (pcb * pcb')
    else
      ind.C[idx, idx] = (1 - st.c_cov) .* Cb .+ st.c_cov .* (pcb * pcb' .+ st.c_c * (2 - st.c_c) .* Cb)
    end
  end
end

# Insert into the Pareto archive (dominance + crowding-eviction), mirroring MOCMAES.update_archive!.
function _archive!(st::IgelState, cand::MOCandidate)::Bool
  objs = cand.fx.objectives
  for m in st.archive; dominates(m.fx.objectives, objs) && return false; end
  filter!(m -> !dominates(objs, m.fx.objectives), st.archive)
  push!(st.archive, cand)
  if length(st.archive) > st.archive_cap
    st.archive = st.archive[1:end]               # ensure concrete
    cds = _crowding([m.fx.objectives for m in st.archive])
    deleteat!(st.archive, argmin(cds))
  end
  improved = cand.fx.aggregate < st.representative.fx.aggregate
  if improved
    st.representative = cand; st.best_iteration = st.i
    push!(st.best_iterations, (st.i, cand.fx.aggregate, cand))
  end
  return improved
end

# One generation: rank parents+offspring, do the (1+1) success-based σ/C updates, select best μ,
# fold offspring into the archive. `off_params` are the decoded params for the μ offspring.
function tell!(st::IgelState, off_fxs::Vector{MOFitness}, off_params::Vector)::Bool
  μ = st.mu
  for k in 1:μ; st._off[k].fx = off_fxs[k]; end
  resd = st._reseed
  Q = vcat(st.pop, st._off)                        # 2μ; rank everything together (consistent keys)
  keys = _keys(Q, st.niche_radius)
  # (1+1) success-based σ/C updates for non-reseed lineages (re-seeds start fresh, no update)
  for k in 1:μ
    resd[k] && continue
    succ = keys[μ+k] <= keys[k] ? 1.0 : 0.0
    sig = st.pop[k].sigma
    _update_stepsize!(st.pop[k], succ, st)
    _update_stepsize!(st._off[k], succ, st)
    succ == 1.0 && _update_cov!(st._off[k], (st._off[k].x .- st.pop[k].x) ./ sig, st)
  end
  g = st.i + 1                                      # the generation this offspring becomes
  # PROTECTED slots: a re-seed (retire its stagnated parent, take the fresh offspring, shield it for
  # `maturity_period` gens) or an individual still within its maturity window. Protected lineages keep
  # the (1+1)-better of parent/offspring but are NOT subject to global removal.
  prot = [k for k in 1:μ if resd[k] || st.pop[k].mature_at > st.i]
  new_pop = Individual[]
  for k in prot
    if resd[k]
      ind = st._off[k]; ind.mature_at = g + _maturity(st)   # live-overridable shield length (else the struct field)
    else
      ind = keys[μ+k] <= keys[k] ? st._off[k] : st.pop[k]   # (1+1) winner, maturity preserved
      ind.mature_at = st.pop[k].mature_at
    end
    push!(new_pop, ind)
  end
  # MATURE slots: normal Igel selection over the mature parents + their offspring fills the rest.
  mat = [k for k in 1:μ if !(resd[k] || st.pop[k].mature_at > st.i)]
  Qm = vcat([st.pop[k] for k in mat], [st._off[k] for k in mat])
  Km = vcat([keys[k] for k in mat], [keys[μ+k] for k in mat])
  order = sortperm(Km)
  for j in 1:(μ - length(new_pop)); push!(new_pop, Qm[order[j]]); end
  st.pop = new_pop
  is_new = false
  cur_best = st.representative
  for k in 1:μ
    cand = MOCandidate(off_params[k], off_fxs[k])
    is_new |= _archive!(st, cand)
    off_fxs[k].aggregate < cur_best.fx.aggregate && (cur_best = cand)
  end
  st.current = cur_best
  st.i += 1; st.current_iteration = st.i; st.t = Statistics_mean_sigma(st)
  return is_new
end

# mean σ across the population — a benign progress field for the writer's log line.
@inline Statistics_mean_sigma(st::IgelState)::Float64 = sum(ind.sigma for ind in st.pop) / st.mu
end
