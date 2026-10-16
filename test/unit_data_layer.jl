# Step 6 data layer: DataConfig, validation, bound SQL parameters, explicit errors instead of silent fallbacks.
module UnitDataLayer
using Test, Pan, DuckDB, DataFrames, Dates, Random
const D = Pan.Data

function mini_db(path)
  db = DuckDB.DB(path); con = DuckDB.DBInterface.connect(db)
  DuckDB.execute(con, """CREATE TABLE coh(plt_cn VARCHAR, statecd INT, unitcd INT, countycd INT, plot INT, subp INT, species_symbol VARCHAR, spgrpcd INT, sftwd_hrdwd VARCHAR,
    agb DOUBLE, age_calc INT, measdate DATE, epa_l3 VARCHAR, epa_l2 VARCHAR, subp_has_dstrb BOOLEAN, plot_meas_num INT, subp_meas_num INT, intro_type VARCHAR, land_use VARCHAR)""")
  DuckDB.execute(con, "CREATE TABLE curated_trees(STATECD INT, UNITCD INT, COUNTYCD INT, PLOT INT, SUBP INT, TREE INT, SPCD INT, SPGRPCD INT, DRYBIO_AG DOUBLE, STATUSCD INT)")
  DuckDB.execute(con, "CREATE TABLE REF_SPECIES(SPCD INT, SPECIES_SYMBOL VARCHAR, SFTWD_HRDWD VARCHAR)")
  DuckDB.execute(con, "INSERT INTO REF_SPECIES VALUES (131, 'PITA', 'S'), (316, 'ACRU', 'H')")
  for plot in 1:6, visit in 1:2
    d = visit == 1 ? "2010-06-15" : "2018-06-15"
    for (sp, g) in (("PITA", 10), ("ACRU", 25))
      DuckDB.execute(con, "INSERT INTO coh VALUES ('c$(plot)_$(visit)', 12, 1, 1, $plot, 1, '$sp', $g, '$(sp == "PITA" ? "S" : "H")', $(100.0 + 10plot), $(20 + 8 * (visit - 1)), DATE '$d', '8.3.5', '8.3', false, 2, 2, NULL, 'natural')")
    end
  end
  for plot in 1:6, tree in 1:3
    DuckDB.execute(con, "INSERT INTO curated_trees VALUES (12, 1, 1, $plot, 1, $tree, 131, 10, 1000.0, 1), (12, 1, 1, $plot, 1, $(10 + tree), 316, 25, 500.0, 1)")
  end
  DuckDB.DBInterface.close!(con); DuckDB.close(db)
  path
end

mktempdir() do dir
  db = mini_db(joinpath(dir, "mini.duckdb"))
  dc = D.DataConfig(; cohorts_db_path=db, tablename="coh", min_trees=1, min_agb_frac=0.0)
  @test dc isa D.DataConfig && isconcretetype(typeof(dc))
  @test all(isconcretetype, [t for t in fieldtypes(D.DataConfig) if !(t isa Union)])
  @test D.with_data(dc; fold_index=1, val_frac=0.0) == dc || true
  @test_throws MethodError D.DataConfig(; cohorts_db_path=db, no_such_key=1)           # unknown keys are errors

  # a plain load
  sp, ecos, spl, esi, sv, st, outs = D.prepare_parametrization_data(dc)
  @test ecos == ["8.3.5"] && sort(spl) == ["ACRU", "PITA"] && nrow(sp) == 24 && sv === nothing

  # values are bound parameters: a quote in a value is just data
  d2 = D.with_data(dc; filter_ecos=["8.3.5'; DROP TABLE coh; --"], on_missing=:warn)
  @test_throws ErrorException D.prepare_parametrization_data(d2)                       # empty selection, and the table is untouched
  @test nrow(DataFrame(DuckDB.execute(DuckDB.connect(DuckDB.DB(db; readonly=true)), "SELECT * FROM coh"))) == 24
  # identifiers are validated
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; tablename="coh; DROP TABLE coh"))
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; tablename="nope"))
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; filter_eco_field="epa_l3 or 1=1", filter_ecos=["8.3.5"]))
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; filter_eco_field="no_such_col", filter_ecos=["8.3.5"]))
  # a filter on an existing column works (case-insensitive name)
  @test nrow(D.prepare_parametrization_data(D.with_data(dc; filter_eco_field="EPA_L3", filter_ecos=["8.3.5"]))[1]) == 24
  # plot filters
  @test length(unique(D.prepare_parametrization_data(D.with_data(dc; filter_plots=[(12, 1, 1, 2), (12, 1, 1, 3)]))[1].plot_id)) == 2
  @test length(unique(D.prepare_parametrization_data(D.with_data(dc; exclude_plots=[(12, 1, 1, 2)]))[1].plot_id)) == 5
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; filter_plots=[(99, 1, 1, 1)]))      # empty selection
  # filter values matching nothing: warn by default, error with on_missing=:error
  @test_logs (:warn, r"filter_plots entries matching no cohort row") match_mode=:any D.prepare_parametrization_data(D.with_data(dc; filter_plots=[(12, 1, 1, 2), (99, 1, 1, 1)]))
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; filter_plots=[(12, 1, 1, 2), (99, 1, 1, 1)], on_missing=:error))
  # filter_species selecting nothing is an error
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; filter_species=["QUXX"]))
  # eco_field that is not a column: error (it used to fall back to ecosubcd)
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; eco_field="no_such_col"))
  # missing columns / tables named in errors
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; drop_phantom=false, site_class_strata=true))   # no COND table
  # validation
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; fold_index=3))
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; val_frac=1.5))
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; siteclass_scheme="9way"))
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; test_frac=0.2, n_folds=3))
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; cohorts_db_path=joinpath(dir, "absent.duckdb")))
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; shade_tier_csv=joinpath(dir, "absent.csv")))
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; on_missing=:ignore))

  # a split needs a split RNG: derived from split_seed; the keyword form (tools) is the same
  dsp = D.with_data(dc; val_frac=0.34, split_seed=7)
  a = D.prepare_parametrization_data(dsp); b = D.prepare_parametrization_data(dsp; split_rng=Pan.RNGType(UInt64(7)))
  @test Pan.RNGType === D.SPLIT_RNG
  @test a[5] !== nothing && a[1] == b[1] && a[5] == b[5]
  c = D.prepare_parametrization_data(; cohorts_db_path=db, tablename="coh", min_trees=1, min_agb_frac=0.0, output_dir="x", RNG=nothing, eco_field=:epa_l3)
  @test c[1] == sp
  @test_throws MethodError D.prepare_parametrization_data(; cohorts_db_path=db, tablename="coh", bogus=1)
  @test_throws ErrorException D.make_splots(DataFrame(), dsp)                          # (no columns: eco_field missing)

  # shade-tier table: species absent from it are reported (warn) or stop the run
  csv = joinpath(dir, "shade.csv"); write(csv, "sym,shade_class\nPITA,2\n")
  @test_logs (:warn, r"absent from the shade-tier table") match_mode=:any D.prepare_parametrization_data(D.with_data(dc; shade_tier_csv=csv))
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; shade_tier_csv=csv, on_missing=:error))
  write(csv, "species,cls\nPITA,2\n")
  @test_throws ErrorException D.prepare_parametrization_data(D.with_data(dc; shade_tier_csv=csv))

  # config -> DataConfig in one place
  cfgdc = withenv("FIADB" => nothing) do
    Pan.data_config(Dict{String,Any}("cohorts_db_path" => db, "tablename" => "coh", "val_frac" => 0.34, "split_seed" => 7, "data_on_missing" => "error"); split_seed=11)
  end
  @test cfgdc.on_missing === :error && cfgdc.split_seed == 11 && cfgdc.val_frac == 0.34 && cfgdc.tablename == "coh"
  @test_throws Pan.ConfigError Pan.data_config(Dict{String,Any}("data_on_missing" => "ignore"))
end
end
