# TreeMap-independent stands (scenario_generative.jl): a request whose plots carry their own cohort table runs on every engine (biomass, pnet, hbad,
# hbad2, hbad3p), starts at the table's structure (trees/ha, basal area) in 2025, is deterministic, and cannot be mixed with FIA plots.
# Needs the national FIA database and the yaml rosters (skipped when the database is missing).
module UnitScenarioGenerative
using Test, Pan
const P = Pan
const SCEN = P.ScenarioConfig()
const ROSTERS = Dict("biomass" => "r4_biomass_v1.yaml", "pnet" => "r4_pnet_v1.yaml", "hbad" => "r4_hbad_v1.yaml", "hbad2" => "r4_hbad2_v2j.yaml", "hbad3p" => "r4_hbad3p_gafl.yaml")
if !isfile(SCEN.fia_db)
  @info "unit_scenario_generative: FIA database not found: skipped"
else
  coh = [Dict("sym" => "PITA", "age" => 25.0, "tph" => 600.0, "qmd_cm" => 20.0, "ht_m" => 18.0, "cr" => 0.4),
         Dict("sym" => "QUNI", "age" => 15.0, "tph" => 300.0, "qmd_cm" => 12.0, "ht_m" => 11.0, "cr" => 0.5),
         Dict("sym" => "_H", "age" => 35.0, "tph" => 100.0, "qmd_cm" => 25.0, "ht_m" => 19.0, "cr" => 0.35),
         Dict("sym" => "ACNI5", "age" => 35.0, "tph" => 40.0, "qmd_cm" => 30.0, "ht_m" => 21.0, "cr" => 0.35),   # unmapped: shares _H's catch-all (one key, two site cohorts)
         Dict("sym" => "_S", "age" => 45.0, "tph" => 30.0, "qmd_cm" => 28.0, "ht_m" => 20.0, "cr" => 0.3)]
  # the worker's own record format (tools/pan_worker.jl _plots): Baker Co. FL, EPA L4 8.5.3.75e
  plots = [(cn="gen:1", acres=12.0, l4="8.5.3.75e", cohorts=coh, lon=-82.31, lat=30.34),
           (cn="gen:2", acres=5.0, l4="8.5.3.75e", cohorts=Any[], lon=-82.30, lat=30.35)]           # a bare stand: regeneration only
  ba_in = sum(c["tph"] * pi / 4 * (c["qmd_cm"] / 100)^2 for c in coh) * P.SC_M2HA_TO_FT2AC          # ft2/ac of the table
  tpa_in = sum(c["tph"] for c in coh) / 2.47105
  plan = Dict("cycles" => 3, "period" => 10, "actions" => Any[])
  for (plug, y) in ROSTERS
    path = joinpath(dirname(@__DIR__), "results", "params_yaml", y)
    isfile(path) || (@info "roster $y missing: $plug skipped"; continue)
    @testset "generative stands on $plug" begin
      ps = P.load_paramset(read(path, String), SCEN)
      r, w = P.run_scenario(ps, plots, plan; seed=2)
      @test r["gen:1"].status == "ok" && r["gen:2"].status == "ok"
      m0 = r["gen:1"].metrics[1]
      @test m0.year == 2025 && all(m -> isfinite(m.ba) && m.ba >= 0, r["gen:1"].metrics) && length(r["gen:1"].metrics) == 4
      @test r["gen:2"].metrics[1].ba < 1.0                                  # biomass-state engines start an empty plot from a 1 g/m2 placeholder cohort
      # the starting structure reproduces the table: exact for the tree-based engines; through the HBAD v1 AGB translation for biomass / pnet / hbad
      if plug in ("hbad2", "hbad3p", "hbad")
        @test isapprox(m0.ba, ba_in; rtol=0.02) && isapprox(m0.tpa, tpa_in; rtol=0.02)
      else
        @test 0.5 * ba_in < m0.ba < 1.5 * ba_in
      end
      @test any(occursin("generative stand cohorts", x) for x in w)
      if plug in ("biomass", "pnet", "hbad")                                 # the softwood group key goes to a softwood catch-all, not the hardwood one
        e = P.scenario_eco(ps, "8.5.3", 0)[1]
        @test P.scenario_species(ps, e, "_S") != P.scenario_species(ps, e, "_H") && P.scenario_species(ps, e, "ACNI5") == P.scenario_species(ps, e, "_H")
      end
      r2, _ = P.run_scenario(ps, plots, plan; seed=2)
      @test [m.ba for m in r2["gen:1"].metrics] == [m.ba for m in r["gen:1"].metrics]
      @test_throws ErrorException P.run_scenario(ps, [plots[1], (cn="20344681010478", acres=1.0, l4="", cohorts=nothing, lon=NaN, lat=NaN)], plan)
      println("$plug: start BA $(round(m0.ba; digits=1)) (table $(round(ba_in; digits=1))) TPA $(round(m0.tpa; digits=0)) (table $(round(tpa_in; digits=0))); year 30 BA $(round(r["gen:1"].metrics[end].ba; digits=1)); bare stand year 30 BA $(round(r["gen:2"].metrics[end].ba; digits=1))")
    end
  end
end
end
