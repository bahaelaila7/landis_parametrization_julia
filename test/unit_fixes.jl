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

end
