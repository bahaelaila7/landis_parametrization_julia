# Step 5b (B22): the simulated-annealing engine (search/SA.jl) — cooling schedule, acceptance rules, reheat, end conditions.
module UnitSearchSA
using Test, Random, Pan
const SA = Pan.Search.SA

mk(f0; kw...) = (c = SA.SACandidate([0.0], f0); SA.SAState(c, c, Random.MersenneTwister(1); kw...))

@testset "SA engine" begin
  @testset "downhill always accepted, new best reported, i counts trials" begin
    st = mk(5.0; t=10.0, initial_t=10.0, max_iter=100)
    @test SA.search_cmp!(SA.SACandidate([1.0], 4.0), st) === true
    @test st.i == 1 && st.best.fx == 4.0 && st.current.fx == 4.0 && st.best_iteration == 1
    @test SA.search_cmp!(SA.SACandidate([2.0], 4.5), st) === false     # uphill, accepted or not: never a new best
    @test st.i == 2 && st.best.fx == 4.0
  end
  @testset "geometric cooling and the end conditions" begin
    st = mk(1.0; t=1.0, initial_t=1.0, alpha=0.5, min_t=0.2, max_iter=100)
    @test !SA.search_update_rule!(st) && st.t == 0.5
    @test !SA.search_update_rule!(st) && st.t == 0.25
    @test SA.search_update_rule!(st) && st.t == 0.125 && SA.is_search_over(st)   # t < min_t
    st2 = mk(1.0; max_iter=2); st2.i = 2
    @test SA.is_search_over(st2)
  end
  @testset "TA threshold accepting is deterministic, SA draws only uphill" begin
    st = mk(1.0; t=0.5, acceptance_rule="TA")
    SA.search_cmp!(SA.SACandidate([1.0], 1.4), st); @test st.current.fx == 1.4      # Δ=0.4 < t
    SA.search_cmp!(SA.SACandidate([2.0], 2.0), st); @test st.current.fx == 1.4      # Δ=0.6 ≥ t
    st = mk(1.0; t=1.0); r0 = copy(st.rng)
    SA.search_cmp!(SA.SACandidate([1.0], 0.5), st); @test rand(copy(st.rng)) == rand(r0)   # downhill: no rng draw
  end
  @testset "reheat when the acceptance probability stays low" begin
    st = mk(0.0; t=0.001, initial_t=7.0, reheat_after_iters=3, acceptance_rule="TA")
    for k in 1:5; SA.search_cmp!(SA.SACandidate([Float64(k)], 100.0 + k), st); end    # all rejected: prob_avg ≈ 0 < threshold
    @test st.t == 7.0
  end
end
end
