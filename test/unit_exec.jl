# Step 3e: execution / search options (ExecOptions, RunTimer, LBSAOptions, SearchOptions, IgelOptions) - the former module-level Refs.
module UnitExec
using Test, Random
import Pan
using Pan.Search: IgelMOCMAES, MOCMAES, MOLBSA

@testset "ExecOptions" begin
  d = Pan.ExecOptions()
  @test d.parallel_mode === :soa && !d.timing && !d.skip_val && !d.newbest_stats && d.store_val_obj && !d.cv_reselect
  @test !Pan.candidate_parallel(d) && Pan.candidate_parallel(Pan.ExecOptions(parallel_mode=:candidate))
  e = Pan.exec_options_from_env(env=Dict("PAN_TIMING" => "1", "PAN_SKIP_VAL" => "true", "PAN_PARALLEL" => "candidate", "PAN_NEWBEST_STATS" => "yes"))
  @test e.timing && e.skip_val && e.newbest_stats && Pan.candidate_parallel(e)
  @test Pan.exec_options_from_env(env=Dict{String,String}()) |> x -> (x.parallel_mode === :soa && !x.timing && !x.skip_val)
  @test Pan.exec_options_from_env(env=Dict{String,String}(), cv_reselect=true).cv_reselect
  @test_throws ErrorException Pan.exec_options_from_env(env=Dict("PAN_PARALLEL" => "bogus"))
  @test isconcretetype(Pan.ExecOptions) && !ismutabletype(Pan.ExecOptions) && all(isconcretetype, fieldtypes(Pan.ExecOptions))   # immutable, concretely typed
end

@testset "RunTimer: free when off" begin
  @test Pan.run_timer(Pan.ExecOptions()) === nothing && Pan.run_timer(Pan.ExecOptions(timing=true)) isa Pan.RunTimer
  ch = Channel{Int}(4)
  Pan.timed_put!(nothing, ch, 1); take!(ch)                           # compile
  t = Pan.RunTimer(); Pan.timed_put!(t, ch, 2); take!(ch)
  @test t.io > 0 && t.sim == 0
  Pan.reset!(t); @test t.io == 0
  f(c) = (Pan.timed_put!(nothing, c, 3); take!(c))
  f(ch); @test (@allocated f(ch)) <= (@allocated(Pan.timed_put!(t, ch, 4)) + 1024)   # the off path does no more than the bare put!/take!
end

@testset "SearchOptions / LBSAOptions" begin
  s = Pan.SearchOptions()
  @test s.lbsa == Pan.LBSAOptions(150, 0.95, 150, true, false) && !s.mocmaes_nds
  @test s.igel.reseed_sigma_override == -1.0 && s.igel.maturity_override == -1 && !s.igel.use_cholesky && isempty(s.igel.niche_q)
  cfg = Pan.load_config("configs/region4/r4_igelmo_full.yml"; overrides=Dict{String,Any}("lbsa_stretch_len" => 7, "lbsa_cooling_only" => true, "mocmaes_true_nds" => true,
    "igel_cholesky" => true, "igel_reseed_sigma_override" => 0.02, "igel_maturity_override" => 5))
  s2 = Pan.search_options_from_config(cfg)
  @test s2.lbsa.stretch_len == 7 && s2.lbsa.cooling_only && s2.mocmaes_nds && s2.igel.use_cholesky && s2.igel.reseed_sigma_override == 0.02 && s2.igel.maturity_override == 5
  @test typeof(s2) === typeof(s)                                       # one concrete type whatever the values
end

@testset "mo_sortperm nds keyword" begin
  fx(a...) = MOLBSA.MOFitness(collect(Float32.(a)), Float64(sum(a)))
  F = [fx(1, 5), fx(2, 2), fx(5, 1), fx(3, 3), fx(2, 4)]
  @test sort(MOCMAES.mo_sortperm(F)) == 1:5 && sort(MOCMAES.mo_sortperm(F; nds=true)) == 1:5
  @test MOCMAES.mo_sortperm(F; nds=true) == MOCMAES._nds_sortperm(F) && MOCMAES.mo_sortperm(F) == MOCMAES._winrank_sortperm(F)
end

@testset "IgelOptions: explicit, typed, equivalent to the former module Refs" begin
  n = 4; mu = 5
  function mk(; seed=1)
    rng = Random.Xoshiro(seed)
    us = [rand(rng, n) for _ in 1:mu]
    cands = [MOLBSA.MOCandidate(copy(u), MOLBSA.MOFitness([sum(u), sum(1 .- u)], sum(u .^ 2))) for u in us]   # params = the u-vector itself
    IgelMOCMAES.IgelState(us, cands, rng; sigma0=0.2, reseed_sigma=0.05, maturity_period=2, niche_radius=0.0)
  end
  run(opts; gens=6) = begin
    st = mk(); log = Vector{Vector{Float64}}[]
    for _ in 1:gens
      offs = IgelMOCMAES.ask(st, opts); push!(log, copy.(offs))
      fxs = [MOLBSA.MOFitness([sum(o), sum(1 .- o) + 0.1sum(abs2, o .- 0.5)], sum(o .^ 2)) for o in offs]
      IgelMOCMAES.tell!(st, fxs, offs, opts)
    end
    (st, log)
  end
  base = IgelMOCMAES.IgelOptions()
  s1, l1 = run(base); s2, l2 = run(base)
  @test l1 == l2 && length(s1.archive) == length(s2.archive)             # deterministic, no hidden state
  _, lc = run(IgelMOCMAES.IgelOptions(use_cholesky=true))
  @test lc != l1 && all(all(0 .<= o .<= 1) for g in lc for o in g)       # a different square root, same box
  # reseed_sigma_override >= 0 replaces the struct field: a huge threshold reseeds every individual on every ask
  st = mk(); IgelMOCMAES.ask(st, IgelMOCMAES.IgelOptions(reseed_sigma_override=10.0)); @test all(st._reseed)
  st = mk(); IgelMOCMAES.ask(st, base); @test !any(st._reseed)           # sigma0 = 0.2 > reseed_sigma 0.05
  # maturity_override: the shield of a re-seed lasts the override instead of the struct field
  st = mk(); o = IgelMOCMAES.IgelOptions(reseed_sigma_override=10.0, maturity_override=9)
  offs = IgelMOCMAES.ask(st, o); IgelMOCMAES.tell!(st, [MOLBSA.MOFitness([sum(x), 1.0], 1.0) for x in offs], offs, o)
  @test all(ind.mature_at == 1 + 9 for ind in st.pop if ind.mature_at > 0)
  # niching: native-vector maps are type parameters; the archive keeps one member per niche
  q = fill(0.25, n); on = IgelMOCMAES.with_niching(base, q, u -> u, p -> p)
  @test typeof(on) !== typeof(base) && IgelMOCMAES._niching(IgelMOCMAES.IgelState([rand(n) for _ in 1:2], [MOLBSA.MOCandidate(rand(n), MOLBSA.MOFitness([1.0, 1.0], 1.0)) for _ in 1:2], Random.Xoshiro(1); niche_radius=1.0), on)
  st = mk(); st.niche_radius = 1.0; empty!(st.archive)
  c1 = MOLBSA.MOCandidate(fill(0.50, n), MOLBSA.MOFitness([1.0, 2.0], 3.0)); c2 = MOLBSA.MOCandidate(fill(0.55, n), MOLBSA.MOFitness([2.0, 1.0], 2.0)); c3 = MOLBSA.MOCandidate(fill(0.99, n), MOLBSA.MOFitness([1.5, 1.5], 3.0))
  for c in (c1, c2, c3); IgelMOCMAES._archive!(st, c, on); end
  @test length(st.archive) == 2 && c2 in st.archive && c3 in st.archive && !(c1 in st.archive)   # c1/c2 share a niche (within 1 quantum): the better aggregate stays
  st = mk(); st.niche_radius = 1.0; empty!(st.archive)
  for c in (c1, c2, c3); IgelMOCMAES._archive!(st, c, base); end         # niching off (no quanta): plain Pareto archive keeps all three
  @test length(st.archive) == 3
  @test isconcretetype(typeof(on)) && all(isconcretetype, fieldtypes(typeof(on)))   # the maps are type parameters, not Ref{Any}
end
# Step 5a (B26): validation scoring is concurrent in candidate mode ⇒ val evaluations run their sites serially there, for every driver.
@testset "val_sites_parallel (no nested threads in candidate mode)" begin
  @test Pan.val_sites_parallel(Pan.ExecOptions(parallel_mode=:soa))
  @test !Pan.val_sites_parallel(Pan.ExecOptions(parallel_mode=:candidate))
end
end # module
