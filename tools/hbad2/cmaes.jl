# Compact (μ/μ_w, λ)-CMA-ES (Hansen 2016 tutorial defaults), full covariance; minimises f(x). Deterministic for a given rng.
using LinearAlgebra, Random, Printf
function cmaes(f, x0::Vector{Float64}, σ0::Float64; λ=4 + floor(Int, 3log(length(x0))), maxevals=10_000, rng=Xoshiro(1),
               ftol=1e-9, log_every=20, logf=println, callback=nothing)
  n = length(x0); μ = λ ÷ 2
  w = log(μ + 0.5) .- log.(1:μ); w ./= sum(w); μeff = 1 / sum(abs2, w)
  cc = (4 + μeff / n) / (n + 4 + 2μeff / n); cs = (μeff + 2) / (n + μeff + 5)
  c1 = 2 / ((n + 1.3)^2 + μeff); cμ = min(1 - c1, 2(μeff - 2 + 1 / μeff) / ((n + 2)^2 + μeff))
  ds = 1 + 2max(0, sqrt((μeff - 1) / (n + 1)) - 1) + cs; chiN = sqrt(n) * (1 - 1 / (4n) + 1 / (21n^2))
  m = copy(x0); σ = σ0; pc = zeros(n); ps = zeros(n); C = Matrix{Float64}(I, n, n); B = Matrix{Float64}(I, n, n); Dg = ones(n)
  best = copy(x0); fbest = f(x0); evals = 1; eigen_at = 0; gen = 0; hist = Float64[]
  X = zeros(n, λ); Z = zeros(n, λ); fx = zeros(λ)
  while evals + λ <= maxevals
    gen += 1
    for k in 1:λ
      Z[:, k] = randn(rng, n); X[:, k] = m .+ σ .* (B * (Dg .* Z[:, k]))
    end
    for k in 1:λ; fx[k] = f(view(X, :, k)); end
    evals += λ
    o = sortperm(fx); fx[o[1]] < fbest && (fbest = fx[o[1]]; best = X[:, o[1]])
    mold = copy(m); m = X[:, o[1:μ]] * w
    y = (m .- mold) ./ σ
    Cinvsqrt = B * Diagonal(1 ./ Dg) * B'
    ps = (1 - cs) .* ps .+ sqrt(cs * (2 - cs) * μeff) .* (Cinvsqrt * y)
    hs = norm(ps) / sqrt(1 - (1 - cs)^(2gen)) / chiN < 1.4 + 2 / (n + 1)
    pc = (1 - cc) .* pc .+ hs * sqrt(cc * (2 - cc) * μeff) .* y
    Y = (X[:, o[1:μ]] .- mold) ./ σ
    C = (1 - c1 - cμ) .* C .+ c1 .* (pc * pc' .+ (1 - hs) * cc * (2 - cc) .* C) .+ cμ .* (Y * Diagonal(w) * Y')
    σ *= exp((cs / ds) * (norm(ps) / chiN - 1))
    if evals - eigen_at > λ / (c1 + cμ) / n / 10
      eigen_at = evals; C = Symmetric((C + C') / 2); E = eigen(C); Dg = sqrt.(max.(E.values, 1e-20)); B = E.vectors; C = Matrix(C)
    end
    push!(hist, fbest)
    gen % log_every == 0 && logf(@sprintf("CMA gen %d evals %d fbest %.4f fgen %.4f σ %.3g cond %.2g", gen, evals, fbest, fx[o[1]], σ, (maximum(Dg) / minimum(Dg))^2))
    callback === nothing || callback(gen, best, fbest)
    if length(hist) > 60 && hist[end-60] - fbest < ftol * max(1, abs(fbest)); break; end
    σ * maximum(Dg) < 1e-8 && break
  end
  (x=best, f=fbest, evals=evals, m=m, σ=σ)
end
