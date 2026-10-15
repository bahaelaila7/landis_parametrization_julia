# Pan._expandenv and Pan._cfg_data_inputs on small Dict configs (no DB access).
module UnitConfig
using Test, Pan

withenv_(f, kv...) = withenv(f, kv...)

@testset "_expandenv" begin
  withenv("PAN_T_A" => "/data/x", "PAN_T_EMPTY" => "", "PAN_T_B" => "bb") do
    @test Pan._expandenv("") == ""
    @test Pan._expandenv("plain/relative.duckdb") == "plain/relative.duckdb"        # unchanged
    @test Pan._expandenv("\${PAN_T_A}/out") == "/data/x/out"
    @test Pan._expandenv("\$PAN_T_A/out") == "/data/x/out"
    @test Pan._expandenv("\${PAN_T_A}/\${PAN_T_B}") == "/data/x/bb"
    @test Pan._expandenv("\$PAN_T_B-\$PAN_T_B") == "bb-bb"
    @test Pan._expandenv("\${PAN_T_EMPTY}/z") == "/z"
    @test Pan._expandenv("\${PAN_T_DEFINITELY_UNSET}/z") == "/z"                    # missing var -> ""
    @test Pan._expandenv("\$PAN_T_DEFINITELY_UNSET") == ""
    @test Pan._expandenv("~/data.duckdb") == joinpath(homedir(), "data.duckdb")      # leading ~ expanded
    @test Pan._expandenv("a~b") == "a~b"                                             # only a LEADING ~
    @test Pan._expandenv(SubString("\${PAN_T_A}x", 1)) == "/data/x" * "x" && Pan._expandenv("x") isa String
    # \$ followed by a non-word char is left alone
    @test Pan._expandenv("cost\$ 5") == "cost\$ 5"
  end
end

@testset "_cfg_getter" begin
  withenv("PAN_T_A" => "/data/x") do
    g = Pan._cfg_getter(Dict{String,Any}("p" => "\${PAN_T_A}/q", "n" => 3, "v" => ["\${PAN_T_A}"]))
    @test g("p", "dflt") == "/data/x/q"            # strings expanded
    @test g("n", 0) == 3 && g("missing", "d") == "d"
    @test g("v", nothing) == ["\${PAN_T_A}"]        # non-strings pass through un-expanded (lists are NOT expanded)
    @test g("missing", "\${PAN_T_A}") == "/data/x"        # string DEFAULTS are expanded too (expansion is applied to the resolved value)
  end
end

@testset "_cfg_data_inputs defaults" begin
  withenv("FIADB" => nothing) do
    d = Pan._cfg_data_inputs(Dict{String,Any}())
    @test d isa NamedTuple
    @test d.cohorts_db_path == "../data_eco_cohorts.duckdb"
    @test d.tablename == "data_eco_cohorts"
    @test d.eco_field == "epa_l3" && d.filter_eco_field == "epa_l3"
    @test d.filter_ecos == String[] && d.filter_species == String[] && d.filter_states == Int[] && d.siteclass_l3s == String[]
    @test d.filter_plots == NTuple{4,Int}[] && d.exclude_plots == NTuple{4,Int}[]
    @test d.filter_extent === nothing && d.shade_tier_csv === nothing && d.train_cap === nothing
    @test d.skip_disturbances === true && d.spinup === false && d.by_subplot === false
    @test d.val_frac === 0.0 && d.test_frac === 0.0 && d.split_seed === 42 && d.n_folds === 1 && d.fold_index === 1
    @test d.min_trees === 100 && d.min_agb_frac === 0.05
    @test d.stratify_eco_mixed === false && d.single_ecoregion === false && d.stratify_landuse === false && d.site_class_strata === false
    @test d.siteclass_hi_max === 4 && d.siteclass_scheme == "2way"
    @test d.max_exact === 12 && d.max_group === 4 && d.drop_phantom === false
    @test d.train_cap_k === 1.5 && d.train_cap_min === 20
    # field set is exactly the parametrize data-prep kwargs
    @test Set(keys(d)) == Set([:cohorts_db_path, :filter_eco_field, :eco_field, :tablename, :filter_ecos, :filter_plots, :exclude_plots, :filter_species,
      :skip_disturbances, :spinup, :by_subplot, :val_frac, :split_seed, :n_folds, :fold_index, :test_frac, :min_trees, :min_agb_frac, :stratify_eco_mixed,
      :single_ecoregion, :stratify_landuse, :site_class_strata, :siteclass_hi_max, :siteclass_scheme, :shade_tier_csv, :filter_extent, :filter_states,
      :siteclass_l3s, :max_exact, :max_group, :drop_phantom, :train_cap, :train_cap_k, :train_cap_min])
  end
end

@testset "_cfg_data_inputs parsing" begin
  withenv("FIADB" => nothing, "PAN_T_DIR" => "/scratch") do
    cfg = Dict{String,Any}(
      "tablename" => "curated_cohorts_landis", "eco_field" => "land_use", "filter_eco_field" => "epa_l2",
      "cohorts_db_path" => "\${PAN_T_DIR}/db.duckdb",
      "filter_plots" => Any[[12, 1, 3, 4], ["13", "2", "5", "6"]],                 # ints and strings both accepted
      "exclude_plots" => Any[[1, 2, 3, 4]],
      "filter_ecos" => Any["8.5.3", "8.5.4"], "filter_species" => Any["PIEL"],
      "filter_states" => Any[12, 13], "siteclass_l3s" => Any[85, "84"],            # numbers stringified
      "val_frac" => 1, "split_seed" => 7.0, "n_folds" => 5.0,                       # numeric coercions
      "filter_extent" => "null", "train_cap" => "null", "skip_disturbances" => false, "spinup" => true,
      "shade_tier_csv" => "\${PAN_T_DIR}/t.csv")
    d = Pan._cfg_data_inputs(cfg)
    @test d.tablename == "curated_cohorts_landis" && d.eco_field == "land_use" && d.filter_eco_field == "epa_l2"
    @test d.cohorts_db_path == "/scratch/db.duckdb"
    @test d.filter_plots == [(12, 1, 3, 4), (13, 2, 5, 6)] && eltype(d.filter_plots) == NTuple{4,Int}
    @test d.exclude_plots == [(1, 2, 3, 4)]
    @test d.filter_ecos == ["8.5.3", "8.5.4"] && d.filter_species == ["PIEL"] && d.filter_states == [12, 13]
    @test d.siteclass_l3s == ["85", "84"]
    @test d.val_frac === 1.0 && d.split_seed === 7 && d.n_folds === 5
    @test d.filter_extent === nothing                    # the literal string "null" means unset
    @test d.train_cap === nothing                        # likewise
    @test d.skip_disturbances === false && d.spinup === true
    @test d.shade_tier_csv == "/scratch/t.csv"
    # train_cap: integer stays Int, string stays String (e.g. "auto"), float is a type error? -> check the actual behaviour
    @test Pan._cfg_data_inputs(Dict{String,Any}("train_cap" => 30)).train_cap === 30
    @test Pan._cfg_data_inputs(Dict{String,Any}("train_cap" => "auto")).train_cap == "auto"
    @test Pan._cfg_data_inputs(Dict{String,Any}("filter_extent" => "FL")).filter_extent == "FL"
    # shade_tier_csv present-but-null: haskey -> String(get_cfg(...)) (string "null" is NOT mapped to nothing here)
    @test Pan._cfg_data_inputs(Dict{String,Any}("shade_tier_csv" => "null")).shade_tier_csv == "null"
    # numeric keys are NOT parsed from strings (only filter_plots/exclude_plots entries are): Int("7") -> MethodError
    @test_throws MethodError Pan._cfg_data_inputs(Dict{String,Any}("split_seed" => "7"))
    @test_throws InexactError Pan._cfg_data_inputs(Dict{String,Any}("split_seed" => 7.5))
    # a malformed plot spec fails loudly
    @test_throws Exception Pan._cfg_data_inputs(Dict{String,Any}("filter_plots" => Any[[1, 2, 3]]))
    @test_throws Exception Pan._cfg_data_inputs(Dict{String,Any}("filter_plots" => Any[["a", 2, 3, 4]]))
  end
end

@testset "_cfg_data_inputs: FIADB env + exclude_plots_csv" begin
  # env FIADB wins over the yaml value
  withenv("FIADB" => "/env/fiadb.duckdb") do
    @test Pan._cfg_data_inputs(Dict{String,Any}("cohorts_db_path" => "/yaml/x.duckdb")).cohorts_db_path == "/env/fiadb.duckdb"
    @test Pan._cfg_data_inputs(Dict{String,Any}()).cohorts_db_path == "/env/fiadb.duckdb"
  end
  withenv("FIADB" => "") do                               # empty FIADB is treated as unset
    @test Pan._cfg_data_inputs(Dict{String,Any}("cohorts_db_path" => "/yaml/x.duckdb")).cohorts_db_path == "/yaml/x.duckdb"
  end
  mktempdir() do dir
    csv = joinpath(dir, "ex.csv")
    write(csv, "statecd,unitcd,countycd,plot\n12,1,5,9\n13,2,6,10\n")
    withenv("FIADB" => nothing) do
      d = Pan._cfg_data_inputs(Dict{String,Any}("exclude_plots" => Any[[1, 2, 3, 4]], "exclude_plots_csv" => csv))
      @test d.exclude_plots == [(1, 2, 3, 4), (12, 1, 5, 9), (13, 2, 6, 10)]          # inline list first, then CSV rows appended
      d = Pan._cfg_data_inputs(Dict{String,Any}("exclude_plots_csv" => "null"))
      @test d.exclude_plots == NTuple{4,Int}[]
      @test_throws Exception Pan._cfg_data_inputs(Dict{String,Any}("exclude_plots_csv" => joinpath(dir, "nope.csv")))
    end
  end
end
end # module
