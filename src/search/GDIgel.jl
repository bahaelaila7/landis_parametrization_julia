# Group-Decomposed Igel (1+1)-CMA-ES ("gdigel") — docs/cmaes_gdigel.md.
#
# Why: the calibration objective is a SUM of ~50 nearly independent per-species-group sub-problems (each group's AGB bias
# depends mostly on its own parameters). Standard / Igel CMA-ES accepts or rejects the WHOLE offspring on aggregate
# objectives, so a random step improves some groups and worsens others; the aggregate change behaves like noise, success
# rates collapse and σ shrinks before anything is learned. Here every lineage keeps an independent (1+1)-CMA strategy per
# BLOCK of coordinates (a block = all u-coordinates that affect the same set of groups), and each block's step is judged
# only by the losses of the groups it affects. One simulation therefore runs ~50 small (1+1)-CMA-ES in parallel.
#
# Generation of one lineage (two evaluations at most):
#   1. y = x + σ_b·N(0, C_b) in every active block b; evaluate y → per-group losses f(y).
#   2. block b succeeds iff s_b(f(y)) < s_b(f(x)), s_b = Σ of the losses of the groups b affects; per-block success-rule
#      σ update and rank-one C update (Igel, Hansen & Roth 2007 constants, computed from the BLOCK size).
#   3. x' = blocks of y that succeeded, blocks of x otherwise. If x' ≠ x and x' ≠ y it is evaluated (interactions between
#      groups — e.g. competition on mixed plots — are thereby measured, not assumed away). If x' is worse in total than
#      the parent, the better (in total) of x and y is kept instead.
# Multiple parameterizations: lineages never compete. A lineage that has converged (all σ_b < sigma_stop, or no total
# improvement for `patience` generations) is recorded as a SOLUTION and restarted from a fresh Sobol point, so the run
# accumulates independent solutions from different basins; every evaluated point is also offered to a behavioural archive
# (points meeting the acceptance criterion), the raw material for the equifinality analysis.
module GDIgel
import LinearAlgebra, Random, Sobol
const LA = LinearAlgebra

mutable struct BlockState
  sigma::Float64
  p_succ::Float64
  C::Matrix{Float64}
  p_c::Vector{Float64}
end

mutable struct Lineage
  id::Int                     # unique lineage id (increments on restart)
  x::Vector{Float64}          # u-space point
  f::Vector{Float64}          # per-group losses at x (NaN = group without data)
  tot::Float64                # Σ f (finite groups)
  bs::Vector{BlockState}
  born::Int                   # generation of the (re)start
  best_tot::Float64           # best total seen by this lineage
  last_improve::Int           # generation of the last ≥ rel_tol improvement of best_tot
end

struct BlockDef
  idx::Vector{Int}            # u-coordinates of the block
  resp::Vector{Int}           # groups whose losses judge the block
  d::Float64; p_target::Float64; c_p::Float64; p_thresh::Float64; c_c::Float64; c_cov::Float64
end
function BlockDef(idx::Vector{Int}, resp::Vector{Int})
  n = length(idx); p_target = 2 / 11
  BlockDef(idx, resp, 1 + n / 2, p_target, p_target / (2 + p_target), 0.44, 2 / (n + 2), 2 / (n^2 + 6))
end

mutable struct GDState
  blocks::Vector{BlockDef}
  lineages::Vector{Lineage}
  n::Int
  sigma0::Float64
  sigma_stop::Float64
  sigma_min::Float64          # floor: below it steps are dominated by deterministic roughness of the simulator
  patience::Int
  rel_tol::Float64
  gen::Int
  n_evals::Int
  next_id::Int
  sobol::Sobol.SobolSeq
  rng::Random.Xoshiro
  solutions::Vector{NamedTuple}     # converged lineages: (lineage, gen, x, f, tot)
  pending::Vector{Any}              # per-lineage transient state between ask/assemble/finish
  mode::Symbol                      # :optimize (minimize group losses) | :explore (spread within the behavioural set)
  tau::Vector{Float64}              # :explore — group g is behavioural iff f_g ≤ tau[g]
  anchors::Vector{Vector{Float64}}  # :explore — points to stay away from (anchor + recorded solutions)
  restart_sigma::Float64            # :explore — u-noise around the anchor for a restarted lineage (0 = exactly the anchor)
end

_tot(f) = (s = 0.0; for v in f; isnan(v) || (s += v); end; s)
_score(f, resp) = (s = 0.0; for g in resp; v = f[g]; isnan(v) || (s += v); end; s)
_fresh_bs(b::BlockDef, σ0) = BlockState(σ0, b.p_target, Matrix{Float64}(LA.I, length(b.idx), length(b.idx)), zeros(length(b.idx)))

function GDState(blocks::Vector{BlockDef}, n::Int; sigma0=0.15, sigma_stop=0.004, sigma_min=0.002, patience=40, rel_tol=1e-3, seed=1,
                 mode::Symbol=:optimize, tau::Vector{Float64}=Float64[], anchors=Vector{Float64}[], restart_sigma::Float64=0.02)
  GDState(blocks, Lineage[], n, sigma0, sigma_stop, sigma_min, patience, rel_tol, 0, 0, 1, Sobol.SobolSeq(n), Random.Xoshiro(seed),
    NamedTuple[], Any[], mode, tau, Vector{Vector{Float64}}(anchors), restart_sigma)
end

# :explore helpers — a block is behavioural iff every group it affects (with data) has f_g ≤ tau; its novelty is the
# distance (block coordinates, u-space) to the nearest OTHER lineage or anchor/solution.
_feasible(f, resp, tau) = all(g -> isnan(f[g]) || f[g] <= tau[g], resp)
function _novelty(st, k::Int, idx::Vector{Int}, x::Vector{Float64})
  d = Inf
  for (j, L) in enumerate(st.lineages); j == k && continue; d = min(d, sqrt(sum(abs2, x[idx] .- L.x[idx]))); end
  for a in st.anchors; d = min(d, sqrt(sum(abs2, x[idx] .- a[idx]))); end
  d
end

"""Add a lineage at u-point x with evaluated group losses f."""
function add_lineage!(st::GDState, x::Vector{Float64}, f::Vector{Float64})
  t = _tot(f)
  push!(st.lineages, Lineage(st.next_id, copy(x), copy(f), t, [_fresh_bs(b, st.sigma0) for b in st.blocks], st.gen, t, st.gen))
  push!(st.pending, nothing); st.next_id += 1
  return st
end

"""Next fresh start point (Sobol, skipping already-used points)."""
next_start!(st::GDState) = clamp.(Sobol.next!(st.sobol), 0.0, 1.0)

function _sample(C::Matrix{Float64}, rng, m::Int)
  z = randn(rng, m); m == 1 && return sqrt(max(C[1, 1], 1e-30)) .* z
  E = LA.eigen(LA.Symmetric(C))
  return (E.vectors * LA.Diagonal(sqrt.(max.(E.values, 1e-30)))) * z
end

"""Offspring of every lineage (u-vectors). Blocks without any group data are left untouched."""
function ask(st::GDState)::Vector{Vector{Float64}}
  out = Vector{Vector{Float64}}(undef, length(st.lineages))
  for (k, L) in enumerate(st.lineages)
    y = copy(L.x)
    for (bi, b) in enumerate(st.blocks)
      all(g -> isnan(L.f[g]), b.resp) && continue
      s = _sample(L.bs[bi].C, st.rng, length(b.idx))
      @inbounds for (j, i) in enumerate(b.idx); y[i] = clamp(L.x[i] + L.bs[bi].sigma * s[j], 0.0, 1.0); end
    end
    out[k] = y
  end
  return out
end

"""Per-block success decision + strategy update. Returns, per lineage, the assembled x' to evaluate, or `nothing` when
x' equals y (all blocks succeeded — f(y) is already known) or x (none succeeded)."""
function assemble!(st::GDState, ys::Vector{Vector{Float64}}, fys::Vector{Vector{Float64}})
  st.n_evals += length(ys)
  out = Vector{Union{Nothing,Vector{Float64}}}(undef, length(st.lineages))
  for (k, L) in enumerate(st.lineages)
    y, fy = ys[k], fys[k]; xp = copy(L.x); nsucc = 0; nact = 0
    for (bi, b) in enumerate(st.blocks)
      all(g -> isnan(L.f[g]), b.resp) && continue
      nact += 1; B = L.bs[bi]
      succ = if st.mode === :explore
        fx_ok = _feasible(L.f, b.resp, st.tau); fy_ok = _feasible(fy, b.resp, st.tau)
        (fy_ok && (!fx_ok || _novelty(st, k, b.idx, y) > _novelty(st, k, b.idx, L.x))) ||
          (!fx_ok && !fy_ok && _score(fy, b.resp) < _score(L.f, b.resp)) ? 1.0 : 0.0
      else
        _score(fy, b.resp) < _score(L.f, b.resp) ? 1.0 : 0.0
      end
      B.p_succ = (1 - b.c_p) * B.p_succ + b.c_p * succ
      if succ == 1.0
        step = (y[b.idx] .- L.x[b.idx]) ./ B.sigma
        active = B.p_succ < b.p_thresh
        B.p_c = active ? (1 - b.c_c) .* B.p_c .+ sqrt(b.c_c * (2 - b.c_c)) .* step : (1 - b.c_c) .* B.p_c
        B.C = active ? (1 - b.c_cov) .* B.C .+ b.c_cov .* (B.p_c * B.p_c') :
                       (1 - b.c_cov) .* B.C .+ b.c_cov .* (B.p_c * B.p_c' .+ b.c_c * (2 - b.c_c) .* B.C)
        xp[b.idx] .= y[b.idx]; nsucc += 1
      end
      B.sigma = clamp(B.sigma * exp((1 / b.d) * (B.p_succ - b.p_target) / (1 - b.p_target)), st.sigma_min, 0.5)   # cap: u-space is [0,1]
    end
    st.pending[k] = (y=y, fy=fy, xp=xp, mode=(nsucc == 0 ? :none : nsucc == nact ? :all : :mixed))
    out[k] = st.pending[k].mode === :mixed ? xp : nothing
  end
  return out
end

"""Install the new parents (fxps[k] = f(x') for lineages that needed it) and handle convergence/restarts.
`restart_eval(x)` evaluates a fresh start point (called for restarted lineages)."""
function finish!(st::GDState, fxps::Vector{Any}, restart_eval)
  st.gen += 1
  for (k, L) in enumerate(st.lineages)
    p = st.pending[k]; p === nothing && continue
    if p.mode === :all
      L.x, L.f = p.y, p.fy
    elseif p.mode === :mixed
      st.n_evals += 1; fxp = fxps[k]
      txp, tx, ty = _tot(fxp), L.tot, _tot(p.fy)
      nbad(f) = count(g -> !isnan(f[g]) && f[g] > st.tau[g], eachindex(f))
      if st.mode === :explore
        if nbad(fxp) <= nbad(L.f)
          L.x, L.f = p.xp, fxp
        elseif nbad(p.fy) <= nbad(L.f)
          L.x, L.f = p.y, p.fy
        end
      elseif txp <= tx || txp <= ty
        L.x, L.f = p.xp, fxp
      elseif ty < tx                       # interactions spoiled the assembly: keep the better whole point
        L.x, L.f = p.y, p.fy
      end
    end
    L.tot = _tot(L.f)
    if L.tot < L.best_tot * (1 - st.rel_tol)
      L.best_tot = L.tot; L.last_improve = st.gen
    end
    st.pending[k] = nothing
    conv = st.mode === :explore ? all(B -> B.sigma <= st.sigma_stop, L.bs) || (st.gen - L.born) >= st.patience :
           all(B -> B.sigma <= st.sigma_stop, L.bs) || (st.gen - L.last_improve) >= st.patience
    if conv && st.gen - L.born >= 5
      push!(st.solutions, (lineage=L.id, gen=st.gen, born=L.born, x=copy(L.x), f=copy(L.f), tot=L.tot))
      st.mode === :explore && push!(st.anchors, copy(L.x))
      x0 = st.mode === :explore && !isempty(st.anchors) ?
           clamp.(st.anchors[1] .+ st.restart_sigma .* randn(st.rng, st.n), 0.0, 1.0) : next_start!(st)
      f0 = restart_eval(x0); st.n_evals += 1
      t0 = _tot(f0)
      st.lineages[k] = Lineage(st.next_id, x0, f0, t0, [_fresh_bs(b, st.sigma0) for b in st.blocks], st.gen, t0, st.gen)
      st.next_id += 1
    end
  end
  return st
end

mean_sigma(L::Lineage) = sum(B.sigma for B in L.bs) / length(L.bs)
end
