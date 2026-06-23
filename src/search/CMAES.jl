module CMAES
import Random
import LinearAlgebra
const LA = LinearAlgebra

export CMAESCandidate, CMAESState, ask, tell!, note_best!, restart!, is_search_over

# Engine-agnostic (μ/μ_w, λ)-CMA-ES, mirroring the LBSA optimizer's interface: a candidate is
# just (x = params struct, fx = scalar loss); all the CMA-ES math lives in u-space R^d (the
# [0,1]^d search box of the vector bridge) and is done in Float64 for numerical stability.
# The driver owns sampling + fitness evaluation; this module is pure CMA-ES math + best tracking.
Base.@kwdef struct CMAESCandidate{Tx,Tf}
  x::Tx     # decoded params (e.g. BiomassSuccessionParams) — NOT the u-vector, so the writer works
  fx::Tf    # scalar loss (Float64); convert(Float64, fx) keeps parity with LBSA's writer/log path
end

# Standard Hansen strategy constants for dimension `n` and population `λ`.
function _strategy_constants(n::Int, lambda::Int)
  μ = lambda ÷ 2
  wraw = [log((lambda + 1) / 2) - log(i) for i in 1:μ]
  w = wraw ./ sum(wraw)                       # normalized so sum(w) == 1
  mu_eff = 1.0 / sum(abs2, w)                 # = (Σw)² / Σw²  (Σw == 1)
  c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
  d_sigma = 1 + 2 * max(0.0, sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
  c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
  c_1 = 2 / ((n + 1.3)^2 + mu_eff)
  c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)^2 + mu_eff))
  chiN = sqrt(n) * (1 - 1 / (4n) + 1 / (21 * n^2))   # E‖N(0,I)‖
  (; mu=μ, weights=w, mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu, chiN)
end

Base.@kwdef mutable struct CMAESState{Tx,Tf,TRNG<:Random.AbstractRNG}
  # ---- writer/interface-compatible fields (mirror LBSAState) ----
  best::CMAESCandidate{Tx,Tf}
  current::CMAESCandidate{Tx,Tf}              # best of the most recent generation
  rng::TRNG
  best_iterations::Vector{Tuple{Int,Float64,CMAESCandidate{Tx,Tf}}}
  best_iteration::Int = 0
  current_iteration::Int = 0
  i::Int = 0                                  # generation counter → checkpoint filenames
  n_evals::Int = 0                            # total fitness evaluations spent (resume-exact budget)
  max_iter::Int = 1_000_000                   # max generations (driver also bounds by TRIALS evals)
  # benign log fields the writer's non-best branch reads (diff_avg/prob_avg unused, t = current σ)
  t::Float64 = 0.0
  diff_avg::Float64 = 0.0
  prob_avg::Float64 = 0.0

  # ---- CMA-ES static config (recomputed on IPOP restart) ----
  n::Int
  lambda::Int
  mu::Int
  weights::Vector{Float64}
  mu_eff::Float64
  c_sigma::Float64
  d_sigma::Float64
  c_c::Float64
  c_1::Float64
  c_mu::Float64
  chiN::Float64

  # ---- CMA-ES dynamic state (u-space R^d, Float64) ----
  mean::Vector{Float64}
  sigma::Float64
  C::Matrix{Float64}
  p_sigma::Vector{Float64}
  p_c::Vector{Float64}
  B::Matrix{Float64}                          # eigenvectors of C
  D::Vector{Float64}                          # sqrt eigenvalues of C (per-axis std devs)

  # Hansen-style mixed-integer handling: per-coordinate floor on the u-space sampling std-dev so
  # discrete coords keep flipping as σ shrinks. Empty ⇒ disabled (the driver sets it from
  # PU.integer_u_min_std when the integer_handling flag is on). See ask.
  u_min_std::Vector{Float64} = Float64[]
end

# mean0/sigma0 are in u-space ([0,1]^d). sigma0 ≈ 0.3 is a broad-but-unsaturating default.
function CMAESState(mean0::Vector{Float64}, sigma0::Float64, best::CMAESCandidate{Tx,Tf}, rng::TRNG;
                    lambda::Union{Nothing,Int}=nothing, max_iter::Int=1_000_000,
                    best_iterations=Tuple{Int,Float64,CMAESCandidate{Tx,Tf}}[]) where {Tx,Tf,TRNG<:Random.AbstractRNG}
  n = length(mean0)
  λ = isnothing(lambda) ? 4 + floor(Int, 3 * log(n)) : lambda
  k = _strategy_constants(n, λ)
  CMAESState{Tx,Tf,TRNG}(; best=best, current=best, rng=rng, best_iterations=best_iterations,
    max_iter=max_iter, t=sigma0,
    n=n, lambda=λ, mu=k.mu, weights=k.weights, mu_eff=k.mu_eff,
    c_sigma=k.c_sigma, d_sigma=k.d_sigma, c_c=k.c_c, c_1=k.c_1, c_mu=k.c_mu, chiN=k.chiN,
    mean=copy(mean0), sigma=sigma0, C=Matrix{Float64}(LA.I, n, n),
    p_sigma=zeros(n), p_c=zeros(n), B=Matrix{Float64}(LA.I, n, n), D=ones(n))
end

@inline is_search_over(state::CMAESState)::Bool = state.i >= state.max_iter

# Sample λ candidate u-vectors x = m + σ·B·D·z, z~N(0,I); clamp into the [0,1]^d box.
# Duck-typed on the distribution fields so MOCMAESState can reuse it (same fields).
function ask(state)::Vector{Vector{Float64}}
  n, λ = state.n, state.lambda
  integer_handling = length(state.u_min_std) == n
  xs_u = Vector{Vector{Float64}}(undef, λ)
  for k in 1:λ
    z = randn(state.rng, n)
    y = state.B * (state.D .* z)              # y ~ N(0, C)
    x = state.mean .+ state.sigma .* y        # x ~ N(m, σ²C)
    if integer_handling
      # Hansen-style: top up the marginal std of each discrete coord to its floor `u_min_std[i]`
      # by adding independent axis noise, so it keeps crossing integer boundaries even at small σ.
      @inbounds for i in 1:n
        smin = state.u_min_std[i]
        smin <= 0.0 && continue
        cur = state.sigma * sqrt(max(state.C[i, i], 0.0))   # current marginal std of x_i
        cur < smin && (x[i] += sqrt(smin^2 - cur^2) * randn(state.rng))
      end
    end
    xs_u[k] = clamp.(x, 0.0, 1.0)             # box constraint: simple clip
  end
  return xs_u
end

# One generation update from the λ candidates' fitnesses (minimization) and their u-vectors.
function tell!(state::CMAESState, fitnesses::Vector{Float64}, xs_u::Vector{Vector{Float64}})
  _update_distribution!(state, sortperm(fitnesses), xs_u)  # ascending: best (lowest loss) first
end

# One generation CMA-ES distribution update, given an explicit best→worst ranking of the λ
# candidates (`ranking[1]` is the best). Duck-typed on the distribution fields so both CMAESState
# and MOCMAESState reuse it — only the ranking differs (scalar sort vs multi-objective sort).
# Uses the *clipped* offsets so the path/covariance updates reflect what was actually evaluated.
function _update_distribution!(state, ranking::Vector{Int}, xs_u::Vector{Vector{Float64}})
  n, λ, μ = state.n, state.lambda, state.mu
  σ, m, w = state.sigma, state.mean, state.weights
  cσ, dσ, cc, c1, cμ, μeff, chiN = state.c_sigma, state.d_sigma, state.c_c, state.c_1, state.c_mu, state.mu_eff, state.chiN

  idx = ranking
  ys = [(xs_u[idx[k]] .- m) ./ σ for k in 1:μ]

  # --- recombination: new mean ---
  ymean = zeros(n)
  for k in 1:μ
    ymean .+= w[k] .* ys[k]
  end
  state.mean = clamp.(m .+ σ .* ymean, 0.0, 1.0)

  # --- step-size evolution path: C^{-1/2}·ymean = B·diag(1/D)·Bᵀ·ymean ---
  invsqrtC_ymean = state.B * ((state.B' * ymean) ./ state.D)
  state.p_sigma = (1 - cσ) .* state.p_sigma .+ sqrt(cσ * (2 - cσ) * μeff) .* invsqrtC_ymean
  ps_norm = LA.norm(state.p_sigma)

  # --- covariance evolution path (with Heaviside h_σ to stall p_c when σ is moving fast) ---
  g = state.i + 1
  hsig = (ps_norm / sqrt(1 - (1 - cσ)^(2g)) / chiN) < (1.4 + 2 / (n + 1)) ? 1.0 : 0.0
  state.p_c = (1 - cc) .* state.p_c .+ hsig * sqrt(cc * (2 - cc) * μeff) .* ymean

  # --- covariance update: rank-one + rank-μ ---
  delta_hsig = (1 - hsig) * cc * (2 - cc)     # tiny correction keeping E[C] unbiased
  newC = (1 - c1 - cμ) .* state.C .+ c1 .* (state.p_c * state.p_c' .+ delta_hsig .* state.C)
  for k in 1:μ
    newC .+= (cμ * w[k]) .* (ys[k] * ys[k]')
  end

  # --- step-size update ---
  state.sigma = σ * exp((cσ / dσ) * (ps_norm / chiN - 1))

  # --- refresh eigendecomposition (enforce symmetry; floor eigenvalues against round-off) ---
  Csym = LA.Symmetric(newC)
  F = LA.eigen(Csym)
  state.D = sqrt.(max.(F.values, 1e-30))
  state.B = Matrix(F.vectors)
  state.C = Matrix(Csym)

  state.i = g
  state.current_iteration = g
  state.t = state.sigma
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

# IPOP restart: reset the search distribution (optionally with a larger population) while
# keeping the incumbent best, generation counter, rng and history.
function restart!(state::CMAESState, mean0::Vector{Float64}, sigma0::Float64; lambda::Int=state.lambda)
  n = state.n
  k = _strategy_constants(n, lambda)
  state.lambda = lambda
  state.mu = k.mu
  state.weights = k.weights
  state.mu_eff = k.mu_eff
  state.c_sigma = k.c_sigma
  state.d_sigma = k.d_sigma
  state.c_c = k.c_c
  state.c_1 = k.c_1
  state.c_mu = k.c_mu
  state.chiN = k.chiN
  state.mean = copy(mean0)
  state.sigma = sigma0
  state.C = Matrix{Float64}(LA.I, n, n)
  state.B = Matrix{Float64}(LA.I, n, n)
  state.D = ones(n)
  state.p_sigma = zeros(n)
  state.p_c = zeros(n)
  state.t = sigma0
  return nothing
end
end
