# load_scenario_plots must return plots (and each plot's cohorts) in a DETERMINISTIC order: it feeds the plot ids / site indices (hence the
# per-site RNG seeds) and the order of equal-age cohorts of run_scenario. Unordered DuckDB result sets made run_scenario depend on scheduling.
module UnitScenarioOrder
using Test, Pan, DuckDB, DataFrames, Dates
const P = Pan
mktempdir() do d
  f = joinpath(d, "mini.duckdb")
  db = DuckDB.DB(f); con = DuckDB.DBInterface.connect(db)
  DuckDB.execute(con, "CREATE TABLE PLOT(CN VARCHAR, STATECD INT, UNITCD INT, COUNTYCD INT, PLOT INT, INVYR INT, MEASYEAR INT, MEASMON INT, MEASDAY INT, LAT DOUBLE, LON DOUBLE)")
  DuckDB.execute(con, "CREATE TABLE COND(PLT_CN VARCHAR, SITECLCD INT, CONDPROP_UNADJ DOUBLE, COND_STATUS_CD INT)")
  DuckDB.execute(con, "CREATE TABLE coh(plt_cn VARCHAR, measdate DATE, species_symbol VARCHAR, age_calc INT, agb DOUBLE, epa_l3 VARCHAR, epa_l4 VARCHAR, intro_type VARCHAR)")
  cns = ["c9", "c3", "c7", "c1", "c5"]          # deliberately not sorted
  for (i, c) in enumerate(cns)
    DuckDB.execute(con, "INSERT INTO PLOT VALUES ('$c', 12, 1, 1, $i, 2018, 2018, 6, 15, 30.0, -85.0)")
    DuckDB.execute(con, "INSERT INTO COND VALUES ('$c', 3, 1.0, 1)")
    for (sp, age) in (("QUAL", 40), ("PITA", 40), ("ACRU", 40), ("PITA", 10), ("LITU", 40))   # equal-age ties across species
      DuckDB.execute(con, "INSERT INTO coh VALUES ('12_1_1_$i', DATE '2018-06-15', '$sp', $age, $(100.0 * i + age), '8.3.5', '8.3.5a', NULL)")
    end
  end
  DuckDB.DBInterface.close!(con); DuckDB.close(db)
  scen = P.ScenarioConfig(; fia_db=f, cohort_table="coh")
  out = P.load_scenario_plots(scen, cns)
  @test [r.cn for r in out] == sort(cns)
  for r in out
    @test r.status == "ok"
    @test collect(zip(r.cohorts.sym, r.cohorts.age)) == [("ACRU", 40), ("LITU", 40), ("PITA", 10), ("PITA", 40), ("QUAL", 40)]
  end
end
end

# make_sites (non-spinup) must not depend on the input row order of equal-age cohorts (full tie-break key: age desc, species, biomass).
module UnitMakeSitesTies
using Test, Pan, DataFrames, Dates, Random
const P = Pan
let
  d0 = Date(2018, 6, 15)
  rows = DataFrame(plot_id=fill(1, 6), eco_id=fill(1, 6), eco_species_id=[3, 1, 2, 2, 1, 3], age_calc=[40, 40, 40, 40, 10, 40],
                   agb_sum=[5.0, 7.0, 9.0, 4.0, 3.0, 6.0], measdate=fill(d0, 6), start_measdate=fill(d0, 6))
  esids = [[1, 2, 3]]
  function cohorts(df)
    soa = P.make_sites(df, esids; rng=Random.Xoshiro(1), spinup=false, no_establishment=false)
    s = P.getsite(soa, 1)
    [(s.c_species[j], s.c_bio[j]) for j in 1:Int(s.live)]
  end
  base = try cohorts(rows) catch e; e end
  @test !(base isa Exception)
  for seed in 1:5
    @test cohorts(rows[Random.shuffle(Random.Xoshiro(seed), 1:6), :]) == base
  end
end
end
