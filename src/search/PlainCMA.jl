module PlainCMA
# (mu/mu_w, lambda)-CMA-ES with a full covariance in the raw (unscaled) search space: a faithful, resumable port of the prototype's compact CMA-ES
# (tools/hbad2/cmaes.jl; Hansen 2016 tutorial defaults). It is the engine of the HBAD2 calibration, whose parameters live in their own decoded space (log / linear
# coordinates, clamps in the decode), not in the [0,1]^d u-space of the Biomass Succession ParamSpace that Pan's CMAES module works in.
#
# `ask(s)` samples the lambda offspring of the next generation, `tell!(s, fx)` ranks and updates; a state is a plain struct, `save_state` / `load_state` round-trip it through
# JLD2 so that a resumed run continues exactly as the uninterrupted one (same RNG stream, same covariance, same stopping counters).
#
# Stopping (as in the prototype): budget (`evals + lambda > maxevals`), stagnation (the best-so-far improved by less than `ftol * max(1, |f|)` over the last `stall`
# generations) or step-size collapse (`sigma * max(D) < 1e-8`). `finished(s)` tells.
import Random
import LinearAlgebra
import JLD2
const LA = LinearAlgebra
import ..Common: Common, AbstractSearch, ask, tell!, state, save_state, load_state

export PlainCMACandidate, PlainCMAState, PlainCMASearch, ask, tell!, finished, state, save_state, load_state, stop_reason

"Best point so far: search vector `x` and its objective value `fx` (the field names the checkpoint writer reads)."
struct PlainCMACandidate
  x::Vector{Float64}
  fx::Float64
end

mutable struct PlainCMAState{TRNG<:Random.AbstractRNG}
  n::Int; lambda::Int; mu::Int
  w::Vector{Float64}; mueff::Float64; cc::Float64; cs::Float64; c1::Float64; cmu::Float64; ds::Float64; chiN::Float64
  m::Vector{Float64}; sigma::Float64; pc::Vector{Float64}; ps::Vector{Float64}
  C::Matrix{Float64}; B::Matrix{Float64}; Dg::Vector{Float64}
  best::PlainCMACandidate
  evals::Int; eigen_at::Int; gen::Int; i::Int
  hist::Vector{Float64}
  maxevals::Int; ftol::Float64; stall::Int
  rng::TRNG
  X::Matrix{Float64}; fx::Vector{Float64}
  stopped::Bool; reason::String
end

"""
    PlainCMAState(x0, f0, sigma0; lambda=4 + floor(Int, 3log(n)), maxevals=10_000, rng=Random.Xoshiro(1), ftol=1e-9, stall=150)

Initial state at `x0` whose objective value `f0` has been evaluated (it counts as one evaluation, `evals = 1`, as in the prototype).
"""
function PlainCMAState(x0::AbstractVector{<:Real}, f0::Real, sigma0::Real; lambda::Integer=4 + floor(Int, 3log(length(x0))), maxevals::Integer=10_000,
                       rng::Random.AbstractRNG=Random.Xoshiro(1), ftol::Real=1e-9, stall::Integer=150)
  n = length(x0); λ = Int(lambda); μ = λ ÷ 2
  w = log(μ + 0.5) .- log.(1:μ); w ./= sum(w); μeff = 1 / sum(abs2, w)
  cc = (4 + μeff / n) / (n + 4 + 2μeff / n); cs = (μeff + 2) / (n + μeff + 5)
  c1 = 2 / ((n + 1.3)^2 + μeff); cμ = min(1 - c1, 2(μeff - 2 + 1 / μeff) / ((n + 2)^2 + μeff))
  ds = 1 + 2max(0, sqrt((μeff - 1) / (n + 1)) - 1) + cs; chiN = sqrt(n) * (1 - 1 / (4n) + 1 / (21n^2))
  x = Float64.(collect(x0))
  return PlainCMAState{typeof(rng)}(n, λ, μ, w, μeff, cc, cs, c1, cμ, ds, chiN, copy(x), Float64(sigma0), zeros(n), zeros(n),
    Matrix{Float64}(LA.I, n, n), Matrix{Float64}(LA.I, n, n), ones(n), PlainCMACandidate(copy(x), Float64(f0)),
    1, 0, 0, 0, Float64[], Int(maxevals), Float64(ftol), Int(stall), rng, zeros(n, λ), zeros(λ), false, "")
end

"Whether the run is over: stopped by a stopping rule or the next generation would exceed the evaluation budget."
finished(s::PlainCMAState) = s.stopped || s.evals + s.lambda > s.maxevals
"Why the run ended (`budget`, `stagnation`, `step size collapse`), or `` while it runs."
stop_reason(s::PlainCMAState) = s.stopped ? s.reason : (s.evals + s.lambda > s.maxevals ? "budget" : "")

"""
    ask(s) -> Vector{Vector{Float64}}

Sample the next generation: lambda points `m + sigma B (D .* z)`, z standard normal drawn column by column from the state's RNG (advances `gen`).
"""
function ask(s::PlainCMAState)
  s.gen += 1
  n = s.n
  for k in 1:s.lambda
    z = randn(s.rng, n)
    s.X[:, k] = s.m .+ s.sigma .* (s.B * (s.Dg .* z))
  end
  return [s.X[:, k] for k in 1:s.lambda]
end

"""
    tell!(s, fx) -> Bool

Absorb the objective values `fx` of the points returned by the last `ask`: best-so-far, mean / paths / covariance / step-size updates, the periodic eigendecomposition, stopping
tests. Returns whether the best-so-far improved.
"""
function tell!(s::PlainCMAState, fx::AbstractVector{<:Real})
  length(fx) == s.lambda || error("PlainCMA.tell!: expected $(s.lambda) values, got $(length(fx))")
  copyto!(s.fx, fx)
  s.evals += s.lambda
  o = sortperm(s.fx)
  improved = s.fx[o[1]] < s.best.fx
  improved && (s.best = PlainCMACandidate(s.X[:, o[1]], s.fx[o[1]]))
  mold = copy(s.m); s.m = s.X[:, o[1:s.mu]] * s.w
  y = (s.m .- mold) ./ s.sigma
  Cinvsqrt = s.B * LA.Diagonal(1 ./ s.Dg) * s.B'
  s.ps = (1 - s.cs) .* s.ps .+ sqrt(s.cs * (2 - s.cs) * s.mueff) .* (Cinvsqrt * y)
  hs = LA.norm(s.ps) / sqrt(1 - (1 - s.cs)^(2s.gen)) / s.chiN < 1.4 + 2 / (s.n + 1)
  s.pc = (1 - s.cc) .* s.pc .+ hs * sqrt(s.cc * (2 - s.cc) * s.mueff) .* y
  Y = (s.X[:, o[1:s.mu]] .- mold) ./ s.sigma
  s.C = (1 - s.c1 - s.cmu) .* s.C .+ s.c1 .* (s.pc * s.pc' .+ (1 - hs) * s.cc * (2 - s.cc) .* s.C) .+ s.cmu .* (Y * LA.Diagonal(s.w) * Y')
  s.sigma *= exp((s.cs / s.ds) * (LA.norm(s.ps) / s.chiN - 1))
  if s.evals - s.eigen_at > s.lambda / (s.c1 + s.cmu) / s.n / 10
    s.eigen_at = s.evals; Cs = LA.Symmetric((s.C + s.C') / 2); E = LA.eigen(Cs); s.Dg = sqrt.(max.(E.values, 1e-20)); s.B = E.vectors; s.C = Matrix(Cs)
  end
  push!(s.hist, s.best.fx)
  s.i = s.gen
  if length(s.hist) > s.stall && s.hist[end-s.stall] - s.best.fx < s.ftol * max(1, abs(s.best.fx))
    s.stopped = true; s.reason = "stagnation"
  elseif s.sigma * maximum(s.Dg) < 1e-8
    s.stopped = true; s.reason = "step size collapse"
  end
  return improved
end

"""
    PlainCMASearch(state)

The `AbstractSearch` wrapper: `ask(s)`, `tell!(s, fx)`, `state(s)`, `save_state(s, dir)` (writes `search_state@<gen>.jld2` and the `search_state_latest.jld2` link) and
`load_state(PlainCMASearch, path)`.
"""
mutable struct PlainCMASearch{S<:PlainCMAState} <: AbstractSearch
  st::S
end
ask(s::PlainCMASearch) = ask(s.st)
tell!(s::PlainCMASearch, fx) = tell!(s.st, fx)
state(s::PlainCMASearch) = s.st
finished(s::PlainCMASearch) = finished(s.st)

function save_state(s::PlainCMASearch, dir::AbstractString)
  mkpath(dir); fname = "search_state@$(s.st.i).jld2"
  JLD2.save_object(joinpath(dir, fname), s.st)
  link_path = joinpath(dir, "search_state_latest.jld2"); islink(link_path) && rm(link_path); symlink(fname, link_path)
  return joinpath(dir, fname)
end
load_state(::Type{PlainCMASearch}, path::AbstractString) = PlainCMASearch(JLD2.load_object(path))

end
