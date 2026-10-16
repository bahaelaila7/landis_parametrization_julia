# Step 5a: the shared search utilities (src/search/common.jl) reproduce the pre-refactor per-driver implementations exactly.
# The REF_* functions below are verbatim copies of the code that lived in NSGA2 / IgelMOCMAES / MOCMAES / MOLBSA / GDIgel.
module UnitSearchCommon
using Test, Random, LinearAlgebra, Pan
const C = Pan.Search.Common
const MOLBSA = Pan.Search.MOLBSA; const MOCMAES = Pan.Search.MOCMAES; const NSGA2 = Pan.Search.NSGA2
const Igel = Pan.Search.IgelMOCMAES

REF_dominates(a::Vector{Float32}, b::Vector{Float32}) = (s = false; for k in eachindex(a); a[k] > b[k] && return false; a[k] < b[k] && (s = true); end; s)
function REF_crowding(objs)
  m = length(objs); m <= 2 && return fill(Inf, m)
  M = length(objs[1]); cd = zeros(Float64, m); vals = Vector{Float64}(undef, m)
  for o in 1:M
    @inbounds for i in 1:m; vals[i] = Float64(objs[i][o]); end
    ord = sortperm(vals); span = vals[ord[end]] - vals[ord[1]]
    cd[ord[1]] = Inf; cd[ord[end]] = Inf
    span > 0 && for r in 2:m-1; cd[ord[r]] += (vals[ord[r+1]] - vals[ord[r-1]]) / span; end
  end
  cd
end
function REF_fronts(objs, compete=(p, q) -> true)
  m = length(objs); S = [Int[] for _ in 1:m]; ndom = zeros(Int, m); rank = zeros(Int, m)
  for p in 1:m, q in 1:m
    p == q && continue
    compete(p, q) || continue
    if REF_dominates(objs[p], objs[q]); push!(S[p], q)
    elseif REF_dominates(objs[q], objs[p]); ndom[p] += 1; end
  end
  cur = [i for i in 1:m if ndom[i] == 0]; rr = 1
  while !isempty(cur)
    for i in cur; rank[i] = rr; end
    nxt = Int[]
    for p in cur, q in S[p]; ndom[q] -= 1; ndom[q] == 0 && push!(nxt, q); end
    cur = nxt; rr += 1
  end
  rank
end
# old archive inserts (aggregate-eviction = MOLBSA/MOSA, crowding-eviction = MOCMAES/NSGA2/Igel)
function REF_insert!(arc, cand, cap, how)
  objs = cand.fx.objectives
  for m in arc; REF_dominates(m.fx.objectives, objs) && return false; end
  filter!(m -> !REF_dominates(objs, m.fx.objectives), arc)
  push!(arc, cand)
  if length(arc) > cap
    how === :agg ? deleteat!(arc, argmax(i -> arc[i].fx.aggregate, eachindex(arc))) :
                   deleteat!(arc, argmin(REF_crowding([m.fx.objectives for m in arc])))
  end
  true
end
# old (1+1)-CMA update (IgelMOCMAES / GDIgel inline forms)
function REF_igel_update(p_succ, sigma, p_c, C, step, succ, c_p, d, p_target, p_thresh, c_c, c_cov)
  p_succ = (1 - c_p) * p_succ + c_p * succ
  sigma *= exp((1 / d) * (p_succ - p_target) / (1 - p_target))
  active = p_succ < p_thresh
  p_c = active ? (1 - c_c) .* p_c .+ sqrt(c_c * (2 - c_c)) .* step : (1 - c_c) .* p_c
  C = active ? (1 - c_cov) .* C .+ c_cov .* (p_c * p_c') : (1 - c_cov) .* C .+ c_cov .* (p_c * p_c' .+ c_c * (2 - c_c) .* C)
  p_succ, sigma, p_c, C
end

mkobj(rng, n, M; ties=false) = [Float32.(ties ? rand(rng, 0:3, M) : rand(rng, M)) for _ in 1:n]
mkcand(o, a) = MOLBSA.MOCandidate(x=0, fx=MOLBSA.MOFitness(o, a))

@testset "search/common.jl == pre-refactor implementations" begin
  rng = Random.MersenneTwister(11)
  @testset "dominance, fronts, crowding" for (n, M, ties) in ((1, 2, false), (2, 3, false), (7, 2, true), (30, 4, false), (40, 3, true), (60, 6, false))
    objs = mkobj(rng, n, M; ties)
    @test C.nd_fronts(objs) == REF_fronts(objs)
    @test NSGA2._fronts(objs) == REF_fronts(objs)
    @test isequal(C.crowding_distance(objs), REF_crowding(objs))
    # niched domination: compete only within a radius in a made-up decision space
    nv = [rand(rng, 3) for _ in 1:n]; r = 0.7
    nd = (p, q) -> sqrt(sum(abs2, nv[p] .- nv[q])) < r
    @test C.nd_fronts(objs, nd) == REF_fronts(objs, nd)
    for a in objs, b in objs; @test C.dominates(a, b) == REF_dominates(a, b); @test MOLBSA.dominates(a, b) == REF_dominates(a, b); end
    # MOCMAES NDS ordering still uses the shared crowding
    fxs = [MOLBSA.MOFitness(o, Float64(sum(o))) for o in objs]
    @test sort(MOCMAES.mo_sortperm(fxs; nds=true)) == collect(1:n)
  end
  @testset "archive insert (aggregate and crowding eviction)" for how in (:agg, :crowd), cap in (3, 8, 50), ties in (false, true)
    a1 = typeof(mkcand([0f0], 0.0))[]; a2 = copy(a1); a3 = copy(a1)
    for i in 1:200
      o = mkobj(rng, 1, 3; ties)[1]; c = mkcand(o, Float64(sum(o)) + 0.01i)
      r1 = REF_insert!(a1, c, cap, how)
      r2 = C.pareto_insert!(a2, c, cap, m -> m.fx.objectives, how === :agg ? (a -> C.evict_worst(a, m -> m.fx.aggregate)) : (a -> C.evict_most_crowded(a, m -> m.fx.objectives)))
      @test r1 == r2
      @test [m.fx.objectives for m in a1] == [m.fx.objectives for m in a2]
    end
  end
  @testset "(1+1)-CMA kernels" begin
    for n in (1, 3, 9), succ in (0.0, 1.0), p0 in (0.1, 0.5)
      d = 1 + n / 2; pt = 2 / 11; cp = pt / (2 + pt); pth = 0.44; cc = 2 / (n + 2); ccov = 2 / (n^2 + 6)
      A = rand(rng, n, n); Cm = A * A' + I(n) * 0.1; pc = rand(rng, n); step = randn(rng, n); sg = 0.3
      ps, s2, pc2, C2 = REF_igel_update(p0, sg, pc, Cm, step, succ, cp, d, pt, pth, cc, ccov)
      ps_ = C.cma11_psucc(p0, cp, succ); s_ = C.cma11_sigma(sg, ps_, d, pt); act = ps_ < pth
      pc_ = C.cma11_path(pc, step, act, cc); C_ = C.cma11_cov(Cm, pc_, act, cc, ccov)
      @test ps == ps_ && s2 == s_ && pc2 == pc_ && C2 == C_
    end
  end
end
end # module
