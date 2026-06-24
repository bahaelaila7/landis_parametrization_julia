module CMAES
import Random
import LinearAlgebra
const LA = LinearAlgebra

export CMAESCandidate, CMAESState, CMABlock, ask, tell!, note_best!, restart!, is_search_over

# (μ/μ_w, λ)-CMA-ES with an optional BLOCK-DIAGONAL covariance: instead of one d×d matrix the search
# distribution is a set of independent CMA-ES sub-distributions, one per GROUP of (assumed-)correlated
# coordinates. Each block g owns a subset `idx` of the u-coordinates and carries its own
# mean/σ/C/paths/strategy-constants over its n_g dims. All blocks share the λ offspring and the SAME
# offspring ranking; each updates only its own coordinates. This drops the covariance cost from O(d²)
# storage / O(d³) eigen to Σ_g O(n_g²) / Σ_g O(n_g³), and lets unrelated parameter groups adapt
# independently. With a single block over all coordinates this is exactly standard CMA-ES.
#
# Restart rule (per the grouped design): the search has "collapsed" only when EVERY block's σ has
# collapsed. `state.sigma` is exposed as max-over-blocks σ, so an existing `state.sigma < ε` test fires
# exactly when all blocks are tiny.
Base.@kwdef struct CMAESCandidate{Tx,Tf}
  x::Tx     # decoded params (e.g. BiomassSuccessionParams) — NOT the u-vector, so the writer works
  fx::Tf    # scalar loss (Float64)
end

# Standard Hansen strategy constants for dimension `n` and population `λ`.
function _strategy_constants(n::Int, lambda::Int)
  μ = lambda ÷ 2
  wraw = [log((lambda + 1) / 2) - log(i) for i in 1:μ]
  w = wraw ./ sum(wraw)
  mu_eff = 1.0 / sum(abs2, w)
  c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
  d_sigma = 1 + 2 * max(0.0, sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
  c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
  c_1 = 2 / ((n + 1.3)^2 + mu_eff)
  c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)^2 + mu_eff))
  chiN = sqrt(n) * (1 - 1 / (4n) + 1 / (21 * n^2))
  (; mu=μ, weights=w, mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu, chiN)
end

# One block's distribution over its `idx` u-coordinates (a self-contained CMA-ES of dimension n).
Base.@kwdef mutable struct CMABlock
  idx::Vector{Int}                  # global u-coordinate indices this block owns
  n::Int
  mu::Int
  weights::Vector{Float64}
  mu_eff::Float64
  c_sigma::Float64
  d_sigma::Float64
  c_c::Float64
  c_1::Float64
  c_mu::Float64
  chiN::Float64
  mean::Vector{Float64}             # in u-space, length n
  sigma::Float64
  C::Matrix{Float64}
  p_sigma::Vector{Float64}
  p_c::Vector{Float64}
  B::Matrix{Float64}
  D::Vector{Float64}
  i::Int = 0
end

function _make_block(idx::Vector{Int}, mean0::Vector{Float64}, sigma0::Float64, lambda::Int)
  n = length(idx)
  k = _strategy_constants(n, lambda)
  CMABlock(; idx=idx, n=n, mu=k.mu, weights=k.weights, mu_eff=k.mu_eff, c_sigma=k.c_sigma,
    d_sigma=k.d_sigma, c_c=k.c_c, c_1=k.c_1, c_mu=k.c_mu, chiN=k.chiN,
    mean=mean0[idx], sigma=sigma0, C=Matrix{Float64}(LA.I, n, n),
    p_sigma=zeros(n), p_c=zeros(n), B=Matrix{Float64}(LA.I, n, n), D=ones(n))
end

Base.@kwdef mutable struct CMAESState{Tx,Tf,TRNG<:Random.AbstractRNG}
  # ---- writer/interface-compatible fields (mirror LBSAState) ----
  best::CMAESCandidate{Tx,Tf}
  current::CMAESCandidate{Tx,Tf}
  rng::TRNG
  best_iterations::Vector{Tuple{Int,Float64,CMAESCandidate{Tx,Tf}}}
  best_iteration::Int = 0
  current_iteration::Int = 0
  i::Int = 0
  n_evals::Int = 0
  max_iter::Int = 1_000_000
  t::Float64 = 0.0
  diff_avg::Float64 = 0.0
  prob_avg::Float64 = 0.0

  # ---- config ----
  n::Int                            # total search dimension
  lambda::Int
  blocks::Vector{CMABlock}          # block-diagonal distribution (1 block ⇒ plain CMA-ES)

  # Hansen-style mixed-integer handling floor (per global u-coordinate). Empty ⇒ disabled.
  u_min_std::Vector{Float64} = Float64[]
end

# `state.sigma` = max over blocks (so an existing `< ε` collapse test fires only when ALL blocks are
# tiny). `state.mu` exposes the first block's μ for the drivers' progress estimate; other field reads
# fall through to getfield. (No code writes these computed fields.)
function Base.getproperty(s::CMAESState, f::Symbol)
  if f === :sigma
    bl = getfield(s, :blocks)
    return isempty(bl) ? 0.0 : maximum(b.sigma for b in bl)
  elseif f === :mu
    bl = getfield(s, :blocks)
    return isempty(bl) ? 0 : bl[1].mu
  end
  return getfield(s, f)
end

# mean0/sigma0 in u-space ([0,1]^d). `blocks`: a partition of 1:d into groups (Vector of index
# vectors); `nothing` ⇒ one block over all coordinates (plain CMA-ES).
function CMAESState(mean0::Vector{Float64}, sigma0::Float64, best::CMAESCandidate{Tx,Tf}, rng::TRNG;
                    lambda::Union{Nothing,Int}=nothing, max_iter::Int=1_000_000,
                    blocks::Union{Nothing,Vector{Vector{Int}}}=nothing,
                    best_iterations=Tuple{Int,Float64,CMAESCandidate{Tx,Tf}}[]) where {Tx,Tf,TRNG<:Random.AbstractRNG}
  n = length(mean0)
  λ = isnothing(lambda) ? 4 + floor(Int, 3 * log(n)) : lambda
  blk_idx = isnothing(blocks) ? [collect(1:n)] : blocks
  blks = [_make_block(idx, mean0, sigma0, λ) for idx in blk_idx]
  CMAESState{Tx,Tf,TRNG}(; best=best, current=best, rng=rng, best_iterations=best_iterations,
    max_iter=max_iter, t=sigma0, n=n, lambda=λ, blocks=blks)
end

@inline is_search_over(state::CMAESState)::Bool = state.i >= state.max_iter

# Sample λ candidate u-vectors; each block contributes its own coordinates via x = m + σ·B·D·z.
function ask(state)::Vector{Vector{Float64}}
  n_total, λ = state.n, state.lambda
  integer_handling = length(state.u_min_std) == n_total
  blocks = state.blocks
  xs_u = Vector{Vector{Float64}}(undef, λ)
  for k in 1:λ
    x = Vector{Float64}(undef, n_total)
    for b in blocks
      z = randn(state.rng, b.n)
      y = b.B * (b.D .* z)                       # y ~ N(0, C_g)
      @inbounds for (j, gi) in enumerate(b.idx)
        x[gi] = b.mean[j] + b.sigma * y[j]
      end
    end
    if integer_handling                          # top up the marginal std of coarse discretes
      for b in blocks
        @inbounds for (j, gi) in enumerate(b.idx)
          smin = state.u_min_std[gi]
          smin <= 0.0 && continue
          cur = b.sigma * sqrt(max(b.C[j, j], 0.0))
          cur < smin && (x[gi] += sqrt(smin^2 - cur^2) * randn(state.rng))
        end
      end
    end
    xs_u[k] = clamp.(x, 0.0, 1.0)
  end
  return xs_u
end

# One generation update from the λ candidates' scalar fitnesses (minimization).
function tell!(state::CMAESState, fitnesses::Vector{Float64}, xs_u::Vector{Vector{Float64}})
  _update_distribution!(state, sortperm(fitnesses), xs_u)
end

# One generation update given an explicit best→worst ranking of the λ candidates (`ranking[1]` is
# best). Duck-typed on `.blocks`/`.i`/`.t` so MOCMAESState reuses it with a different ranking.
function _update_distribution!(state, ranking::Vector{Int}, xs_u::Vector{Vector{Float64}})
  for b in state.blocks
    _update_block!(b, ranking, xs_u)
  end
  state.i += 1
  state.current_iteration = state.i
  state.t = isempty(state.blocks) ? 0.0 : maximum(b.sigma for b in state.blocks)
  return nothing
end

# Standard (μ/μ_w,λ)-CMA-ES distribution update for ONE block, over its `idx` coordinates only.
function _update_block!(b::CMABlock, ranking::Vector{Int}, xs_u::Vector{Vector{Float64}})
  n, μ, w = b.n, b.mu, b.weights
  σ, m = b.sigma, b.mean
  cσ, dσ, cc, c1, cμ, μeff, chiN = b.c_sigma, b.d_sigma, b.c_c, b.c_1, b.c_mu, b.mu_eff, b.chiN
  idx = b.idx

  ys = Vector{Vector{Float64}}(undef, μ)             # block-restricted, σ-normalized steps of best μ
  @inbounds for k in 1:μ
    xk = xs_u[ranking[k]]
    yk = Vector{Float64}(undef, n)
    for (j, gi) in enumerate(idx)
      yk[j] = (xk[gi] - m[j]) / σ
    end
    ys[k] = yk
  end

  ymean = zeros(n)
  @inbounds for k in 1:μ
    ymean .+= w[k] .* ys[k]
  end
  b.mean = clamp.(m .+ σ .* ymean, 0.0, 1.0)

  invsqrtC_ymean = b.B * ((b.B' * ymean) ./ b.D)
  b.p_sigma = (1 - cσ) .* b.p_sigma .+ sqrt(cσ * (2 - cσ) * μeff) .* invsqrtC_ymean
  ps_norm = LA.norm(b.p_sigma)

  g = b.i + 1
  hsig = (ps_norm / sqrt(1 - (1 - cσ)^(2g)) / chiN) < (1.4 + 2 / (n + 1)) ? 1.0 : 0.0
  b.p_c = (1 - cc) .* b.p_c .+ hsig * sqrt(cc * (2 - cc) * μeff) .* ymean

  delta_hsig = (1 - hsig) * cc * (2 - cc)
  newC = (1 - c1 - cμ) .* b.C .+ c1 .* (b.p_c * b.p_c' .+ delta_hsig .* b.C)
  @inbounds for k in 1:μ
    newC .+= (cμ * w[k]) .* (ys[k] * ys[k]')
  end

  b.sigma = σ * exp((cσ / dσ) * (ps_norm / chiN - 1))

  Csym = LA.Symmetric(newC)
  F = LA.eigen(Csym)
  b.D = sqrt.(max.(F.values, 1e-30))
  b.B = Matrix(F.vectors)
  b.C = Matrix(Csym)
  b.i = g
  return nothing
end

# Update the incumbent best if `cand` improves on it; returns true on a new global best.
@inline function note_best!(state::CMAESState, cand::CMAESCandidate)::Bool
  state.current = cand
  if convert(Float64, cand.fx) < convert(Float64, state.best.fx)
    state.best = cand
    state.best_iteration = state.i
    push!(state.best_iterations, (state.i, convert(Float64, cand.fx), cand))
    return true
  end
  return false
end

# Re-initialize each block's distribution from a fresh mean/σ/λ (keeping the block partition `idx`).
# Shared by CMAES and MOCMAES IPOP restarts.
function _reset_blocks!(blocks::Vector{CMABlock}, mean0::Vector{Float64}, sigma0::Float64, lambda::Int)
  for b in blocks
    k = _strategy_constants(b.n, lambda)
    b.mu = k.mu; b.weights = k.weights; b.mu_eff = k.mu_eff
    b.c_sigma = k.c_sigma; b.d_sigma = k.d_sigma; b.c_c = k.c_c; b.c_1 = k.c_1; b.c_mu = k.c_mu; b.chiN = k.chiN
    b.mean = mean0[b.idx]; b.sigma = sigma0
    b.C = Matrix{Float64}(LA.I, b.n, b.n); b.B = Matrix{Float64}(LA.I, b.n, b.n); b.D = ones(b.n)
    b.p_sigma = zeros(b.n); b.p_c = zeros(b.n)
  end
end

# IPOP restart: re-initialize every block's distribution (optionally with a larger λ) while keeping the
# block partition, incumbent best, generation counter, rng and history.
function restart!(state::CMAESState, mean0::Vector{Float64}, sigma0::Float64; lambda::Int=state.lambda)
  state.lambda = lambda
  _reset_blocks!(state.blocks, mean0, sigma0, lambda)
  state.t = sigma0
  return nothing
end
end
