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

end
