# B36: "<x>_from_data: true" with a missing data source is an error (it used to warn and silently leave the parameter in the search).
module UnitLoaderErrors
using Test, Pan
mktempdir() do d
  @test_throws ErrorException Pan._load_shade_tol_table(joinpath(d, "absent.csv"))
  bad = joinpath(d, "bad.csv"); write(bad, "species,cls\nPITA,2\n")
  @test_throws ErrorException Pan._load_shade_tol_table(bad)
  ok = joinpath(d, "ok.csv"); write(ok, "sym,shade_class\npita,2\n")
  @test Pan._load_shade_tol_table(ok) == Dict("PITA" => 2)
  @test_throws ErrorException Pan._load_prob_estab_table(joinpath(d, "nat.csv"), "")
  nat = joinpath(d, "nat.csv"); write(nat, "category,l3,prob_estab\nPITA,8.3.5,0.1\n")
  @test_throws ErrorException Pan._load_prob_estab_table(nat, joinpath(d, "art_absent.csv"))     # the artificial table used to be skipped silently
  @test Pan._load_prob_estab_table(nat, "") == Dict(("PITA", "8.3.5", "natural") => 0.1)
  @test_throws ErrorException Pan._load_prob_estab_table("", "")
  @test_throws ErrorException Pan._load_longevity_table(joinpath(d, "absent.duckdb"))
  @test_throws ErrorException Pan._disturbance_shock_path(Dict{String,Any}("override_injection" => true, "override_injection_sync" => true,
                                                                       "disturbance_full_removals" => joinpath(d, "absent.parquet")))
  @test Pan._disturbance_shock_path(Dict{String,Any}("override_injection" => true, "override_injection_sync" => true, "disturbance_full_removals" => "")) == ""
end
end
