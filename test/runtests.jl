# Pan test suite.   cd <worktree> && source wt_env.sh && ./julia_gdal.sh --project=. --threads=2 test/runtests.jl
using Test, Pan

@testset "Pan" begin
  # New unit suites: included in-process (each file is wrapped in its own module).
  @testset "unit_pancore" begin include("unit_pancore.jl") end
  @testset "unit_params" begin include("unit_params.jl") end
  @testset "unit_loss" begin include("unit_loss.jl") end
  @testset "unit_hotpath" begin include("unit_hotpath.jl") end
  @testset "unit_inject" begin include("unit_inject.jl") end
  @testset "unit_api" begin include("unit_api.jl") end
  @testset "unit_exec" begin include("unit_exec.jl") end
  @testset "unit_config" begin include("unit_config.jl") end
  @testset "unit_config_reader" begin include("unit_config_reader.jl") end
  @testset "unit_fixes" begin include("unit_fixes.jl") end
  @testset "unit_search_common" begin include("unit_search_common.jl") end
  @testset "unit_scenario_order" begin include("unit_scenario_order.jl") end

  # Pre-existing asserting scripts: run as subprocesses (they call exit(), define globals / @check macros, or print-and-exit-code).
  julia = Base.julia_cmd()
  proj = dirname(@__DIR__)
  for f in ("hbad_unit.jl", "pnet_unit.jl", "disturbance_shock_unit.jl", "mo_gdigel_unit.jl", "split_cap_test.jl", "simab_unit.jl", "resume_search.jl")
    isfile(joinpath(@__DIR__, f)) || continue
    @testset "$f (subprocess)" begin
      cmd = Cmd(`$julia --project=$proj --threads=2 $(joinpath(@__DIR__, f))`; dir=proj)
      @test success(pipeline(cmd; stdout=stdout, stderr=stderr))
    end
  end
end
