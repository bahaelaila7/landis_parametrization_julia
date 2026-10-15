# Regression tests for the Step-1 bug fixes (/workspace/pan_refactor/BUGS.md). One @testset per fix, named by its bug id.
module UnitFixes
using Test, Pan

@testset "B21 dual_mode: one parser, Bool accepted" begin
  pm = Pan._parse_dual_mode
  @test pm(false) === :off && pm(nothing) === :off && pm("off") === :off && pm("false") === :off && pm("OFF") === :off
  @test pm(true) === :joint && pm("joint") === :joint && pm("true") === :joint
  @test pm("b") === :b && pm("B") === :b
  @test_throws ErrorException pm("bogus")
  # the disturbance-shock config reader used to call String(::Bool) on `dual_mode: true`
  @test (Pan._configure_disturbance_shock!(Dict{String,Any}("dual_mode" => true, "loss_before_injection" => false)); true)
end

@testset "B1 injection flags do not leak between configs" begin
  # A previous run in the same process left the injection globals on; a config without the keys must get the defaults.
  Pan.OVERRIDE_INJECTION[] = true; Pan.OVERRIDE_INJECTION_SYNC[] = true
  @test Pan._configure_disturbance_shock!(Dict{String,Any}("loss_before_injection" => true)) == ""   # no sync in this config ⇒ no shock
  @test Pan._configure_disturbance_shock!(Dict{String,Any}("override_injection" => true, "override_injection_sync" => true)) != ""
  Pan.OVERRIDE_INJECTION[] = false; Pan.OVERRIDE_INJECTION_SYNC[] = false
  # run_from_yaml: every injection key falls back to a literal default, never to the current Ref value
  src = read(joinpath(pkgdir(Pan), "src", "Pan.jl"), String)
  @test !occursin(r"get_cfg\(\"override_injection[a-z_]*\", *[A-Za-z_]*OVERRIDE_INJECTION", src)
end


@testset "B7 cbal_pct RANKW: a stratum without training cohorts gets weight 0, not NaN" begin
  PU = Pan.PU; FT = Pan.PanCore.FloatType
  saved = (PU.RANKW_MODE[], PU.CELL_NCOH[], PU.RANKW[])
  try
    PU.RANKW_MODE[] = :cbal_pct
    nc = zeros(Int, 4, 3); nc[1, 1] = 40; nc[2, 1] = 5; nc[3, 3] = 12   # eco 2 has species but no cohorts
    PU.CELL_NCOH[] = nc
    esids = [[1, 2], [1, 4], [3]]
    Pan._set_rankw!(zeros(FT, 4, 3), esids; split_size=[10, 0, 4])
    R = PU.RANKW[]
    @test !any(isnan, R)
    @test all(iszero, R[:, 2])
    @test R[1, 1] > 0 && R[2, 1] > 0 && R[3, 3] > 0
    @test sum(R) ≈ 1
  finally
    PU.RANKW_MODE[], PU.CELL_NCOH[], PU.RANKW[] = saved
  end
end


@testset "B5 unknown search_mode is an error, not a silent LBSA run" begin
  @test Pan._check_search_mode("igelmo") === nothing && Pan._check_search_mode("sobol") === nothing
  @test all(m -> Pan._check_search_mode(m) === nothing, Pan.SEARCH_MODES)
  @test_throws ErrorException Pan._check_search_mode("igel_mo")
  @test_throws ErrorException Pan._check_search_mode("sa")
end


@testset "B14 loss_before_injection is defined by the injection mode" begin
  r(d) = Pan._resolve_loss_before_injection(Pan._cfg_getter(Dict{String,Any}(d)))
  sync = ("override_injection" => true, "override_injection_sync" => true)
  @test r([sync...]) === true                                            # the 2026-10-15 rule
  @test r([sync..., "loss_before_injection" => false]) === false         # legacy order on request
  @test r(["override_injection" => true]) === false                      # append mode: legacy order by default
  @test r([sync..., "override_injection_replace" => true]) === false     # replace mode: legacy order by default
  @test_throws ErrorException r(["override_injection" => true, "loss_before_injection" => true])
  @test_throws ErrorException r([sync..., "override_injection_replace" => true, "loss_before_injection" => true])
  @test r(Pair{String,Any}[]) === true                                   # no injection: flag has no effect, keep the default
end


@testset "B11 a scale matrix that does not cover a cell is an error, not 0" begin
  FT = Pan.PanCore.FloatType
  M = FT[1 2; 3 4]
  @test Pan._cell(M, 2, 1) == 3
  @test_throws ErrorException Pan._cell(M, 3, 1)
  @test_throws ErrorException Pan._cell(zeros(FT, 0, 0), 1, 1)
end


@testset "B12 per-run caches are cleared and keyed on every input" begin
  Pan._EXCL_CACHE[(Dict(), 1, 2)] = Pan._ExclPlan(:off, false, Dict(), Dict(), Dict())
  Pan._clear_run_caches!()
  @test isempty(Pan._EXCL_CACHE) && isempty(Pan._SHOCK_REGISTRY)
  src = read(joinpath(pkgdir(Pan), "src", "Pan.jl"), String)
  @test occursin("key = (injection_dict, ref_soa, loss_params)", src)
end

end
