# Pan.load_config: coercion, defaults, unknown/retired keys, env expansion, nested gdigel, dual_mode, loss_before_injection,
# and loading of every shipped yaml (configs/**, and the read-only legacy runs/*.yml of the main checkout).
module UnitConfigReader
using Test, Pan, YAML

L(d...) = Pan.load_config(Dict{String,Any}(d...))
cerr(f) = try f(); nothing catch e; e isa Pan.ConfigError ? e.msg : rethrow() end

@testset "defaults come from the schema" begin
  c = L()
  @test c isa Pan.PanConfig && c isa AbstractDict
  @test c["tier"] === 1 && c["seed"] === 404 && c["trials"] === 1000000 && c["n_reps"] === 5
  @test c["spinup"] === false && c["skip_disturbances"] === true
  @test c["cycle_years"] === 8.0 && c["smoothing_variance"] === 1.0 && c["init_perturb_cap"] === 50.0
  @test c["bins_idx"] == vcat(10:10:40, 60:20:120) && eltype(c["bins_idx"]) == Int
  @test c["tablename"] == "data_eco_cohorts" && c["eco_field"] == "epa_l3"
  @test c["train_cap"] === nothing && c["cmaes_lambda"] === nothing && c["resume_from"] === nothing
  @test c["dual_mode"] === :off && c["w_scope"] === :all && c["death_term"] === :none && c["override_injection_disturbance"] === :off
  @test c["agb_hinge_pct_max"] == Inf
  @test c["gdigel"]["mu"] == 8 && c["gdigel"]["weights"] == 8
  @test length(c) == length(Pan.SCHEMA) && haskey(c, "tier") && !haskey(c, "nope")
  @test !Pan.isgiven(c, "tier") && Pan.isgiven(L("tier" => 3), "tier")
  @test Set(keys(c)) == Set(keys(Pan.SCHEMA))
  # defaults are copies: mutating a loaded value does not leak into the next load
  push!(L()["filter_ecos"], "x"); @test isempty(L()["filter_ecos"])
end

@testset "coercion" begin
  # Bool: true/false, 0/1, "true"/"false" in any case
  @test L("spinup" => true)["spinup"] === true && L("spinup" => 1)["spinup"] === true && L("spinup" => 0)["spinup"] === false
  @test L("spinup" => "TRUE")["spinup"] === true && L("spinup" => "False")["spinup"] === false
  @test cerr(() -> L("spinup" => 2)) !== nothing && cerr(() -> L("spinup" => "yes")) !== nothing && cerr(() -> L("spinup" => 1.5)) !== nothing
  # Int: integers, integral floats, numeric strings
  @test L("tier" => 3)["tier"] === 3 && L("tier" => 3.0)["tier"] === 3 && L("tier" => "4")["tier"] === 4 && L("trials" => 1e6)["trials"] === 1000000
  @test cerr(() -> L("tier" => 2.5)) !== nothing && cerr(() -> L("tier" => "x")) !== nothing && cerr(() -> L("tier" => true)) !== nothing
  # Float: reals and numeric strings
  @test L("val_frac" => 1)["val_frac"] === 1.0 && L("val_frac" => "0.25")["val_frac"] === 0.25 && L("cycle_years" => 5)["cycle_years"] === 5.0
  @test L("agb_hinge_pct_max" => "Inf")["agb_hinge_pct_max"] == Inf
  @test cerr(() -> L("val_frac" => "lots")) !== nothing
  # strings stay strings; lists
  @test L("filter_ecos" => Any["8.5.3", 85])["filter_ecos"] == ["8.5.3", "85"] && L("filter_states" => Any[12, 13.0, "14"])["filter_states"] == [12, 13, 14]
  @test cerr(() -> L("filter_ecos" => "8.5.3")) !== nothing
  @test L("anpp_ratio_bounds" => Any[20, 35])["anpp_ratio_bounds"] === [20.0, 35.0] || L("anpp_ratio_bounds" => Any[20, 35])["anpp_ratio_bounds"] == [20.0, 35.0]
  # symbols (with choices)
  @test L("w_scope" => "survivors")["w_scope"] === :survivors && L("death_term" => "likelihood")["death_term"] === :likelihood
  @test occursin("w_scope", cerr(() -> L("w_scope" => "some")))
  # nullable: nothing, "null", "none"; train_cap is int-or-string
  @test L("train_cap" => nothing)["train_cap"] === nothing && L("train_cap" => "null")["train_cap"] === nothing && L("train_cap" => "None")["train_cap"] === nothing
  @test L("train_cap" => 30)["train_cap"] === 30 && L("train_cap" => "auto")["train_cap"] == "auto" && L("train_cap" => 30.0)["train_cap"] === 30
  @test L("resume_from" => "null")["resume_from"] === nothing && L("filter_extent" => "FL")["filter_extent"] == "FL"
  @test L("cmaes_lambda" => 4)["cmaes_lambda"] === 4 && L("nsga2_offspring" => "null")["nsga2_offspring"] === nothing
  @test cerr(() -> L("tier" => nothing)) !== nothing && occursin("must not be null", cerr(() -> L("tier" => nothing)))
  # plots: 4-lists of ints or numeric strings (also tuples); malformed is an error
  @test L("filter_plots" => Any[[12, 1, 3, 4], ["13", "2", "5", "6"], (1, 2, 3, 4)])["filter_plots"] == [(12, 1, 3, 4), (13, 2, 5, 6), (1, 2, 3, 4)]
  @test eltype(L("exclude_plots" => Any[[1, 2, 3, 4]])["exclude_plots"]) == NTuple{4,Int}
  @test cerr(() -> L("filter_plots" => Any[[1, 2, 3]])) !== nothing && cerr(() -> L("filter_plots" => Any[["a", 2, 3, 4]])) !== nothing
  # dicts
  c = L("param_split_species" => Dict("S" => Any["PIEL"]), "longevity_overrides" => Dict("PIEL" => 275), "pnet_generic" => Dict(:k => 1))
  @test c["param_split_species"] == Dict("S" => ["PIEL"]) && c["longevity_overrides"] == Dict("PIEL" => 275.0) && c["pnet_generic"] == Dict("k" => 1)
  @test L("param_tier_merge" => Dict("S" => Dict("PIEL" => Dict("a" => "g"))))["param_tier_merge"]["S"]["PIEL"]["a"] == "g"
  @test cerr(() -> L("param_pooling" => Any[1])) !== nothing
  # disturbance_full_removals: nothing (repo default), false/""/none/off (disabled), or a path; "null" is a path here (as before)
  @test L()["disturbance_full_removals"] === nothing && L("disturbance_full_removals" => false)["disturbance_full_removals"] == ""
  @test L("disturbance_full_removals" => "off")["disturbance_full_removals"] == "" && L("disturbance_full_removals" => "x.parquet")["disturbance_full_removals"] == "x.parquet"
end

@testset "unknown and retired keys are errors" begin
  m = cerr(() -> L("tierr" => 3)); @test m !== nothing && occursin("unknown config key `tierr`", m) && occursin("`tier`", m)
  m = cerr(() -> L("igel_sigma" => 1)); @test occursin("igel_sigma0", m)                  # closest keys are suggested
  m = cerr(() -> L("zzzzqqqq" => 1)); @test occursin("unknown config key", m) && !occursin("Did you mean", m)
  m = cerr(() -> L("eval_tier" => 3)); @test occursin("retired", m) && occursin("`eval_tier`", m) && occursin("use `tier`", m)
  m = cerr(() -> L("search_tier" => 3)); @test occursin("retired", m) && occursin("use `tier`", m)
  m = cerr(() -> L("sobol_m" => 3)); @test occursin("retired", m) && occursin("`n_reps`", m)
  m = cerr(() -> L("eval_tier" => 3, "sobol_m" => 1)); @test occursin("eval_tier", m) && occursin("sobol_m", m)   # all retired keys at once
  # overrides are validated too
  @test occursin("unknown config key", cerr(() -> Pan.load_config(Dict{String,Any}(); overrides=Dict("nope" => 1))))
  # reading an unknown key from a PanConfig is an error, even through get(cfg, k, default)
  c = L(); @test_throws Pan.ConfigError c["tierr"]
  @test_throws Pan.ConfigError get(c, "tierr", 1)
  @test_throws Pan.ConfigError c["tier"] = 2
  # the error names the file
  mktempdir() do dir
    f = joinpath(dir, "bad.yml"); write(f, "eval_tier: 3\n")
    @test occursin("bad.yml", cerr(() -> Pan.load_config(f)))
    @test occursin("not found", cerr(() -> Pan.load_config(joinpath(dir, "none.yml"))))
    g = joinpath(dir, "ok.yml"); write(g, "tier: 2\nspinup: 1\nfilter_plots:\n  - [1, 2, 3, 4]\n")
    c = Pan.load_config(g; overrides=Dict("tier" => 5))
    @test c["tier"] === 5 && c["spinup"] === true && c["filter_plots"] == [(1, 2, 3, 4)] && c.path == g
    @test Pan.load_config(c)["tier"] === 5                                              # a PanConfig reloads to itself
  end
end

@testset "env expansion" begin
  withenv("PAN_T_A" => "/data/x", "PAN_T_B" => "bb", "PAN_T_NOPE" => nothing, "PAN_DATA" => nothing, "PAN_OUT" => nothing) do
    c = L("output_dir" => "\${PAN_T_A}/out", "cohorts_db_path" => "\$PAN_T_A/db.duckdb", "tablename" => "t_\$PAN_T_B", "search_mode" => "igelmo")
    withenv("FIADB" => nothing) do
      c = L("output_dir" => "\${PAN_T_A}/out", "cohorts_db_path" => "\$PAN_T_A/db.duckdb", "tablename" => "t_\$PAN_T_B")
      @test c["output_dir"] == "/data/x/out" && c["cohorts_db_path"] == "/data/x/db.duckdb" && c["tablename"] == "t_bb"
      @test L("output_dir" => "~/o")["output_dir"] == joinpath(homedir(), "o")
      @test L("output_dir" => "plain/rel")["output_dir"] == "plain/rel"
      # lists / dicts are not expanded (as before)
      @test L("filter_ecos" => Any["\${PAN_T_A}"])["filter_ecos"] == ["\${PAN_T_A}"]
      # PAN_DATA / PAN_OUT fall back to the repo-local layout WITHOUT touching the process environment
      c = L("pnet_co2_csv" => "\${PAN_DATA}/co2.csv", "output_dir" => "\${PAN_OUT}/x")
      @test c["pnet_co2_csv"] == "runs/co2.csv" && c["output_dir"] == "runs/outputs/x"
      @test !haskey(ENV, "PAN_DATA") && !haskey(ENV, "PAN_OUT")
      # an undefined variable expands to "" and warns once per name
      empty!(Pan._WARNED_ENV)
      @test_logs (:warn, r"PAN_T_NOPE") @test L("output_dir" => "\${PAN_T_NOPE}/z")["output_dir"] == "/z"
      @test_logs L("output_dir" => "\${PAN_T_NOPE}/z")                                   # second time: silent
      @test Pan._expandenv("\$PAN_T_NOPE") == ""
    end
    # env FIADB wins over the yaml value (empty FIADB is unset)
    withenv("FIADB" => "/env/fiadb.duckdb") do
      @test L("cohorts_db_path" => "/yaml/x.duckdb")["cohorts_db_path"] == "/env/fiadb.duckdb" && L()["cohorts_db_path"] == "/env/fiadb.duckdb"
    end
    withenv("FIADB" => "") do
      @test L("cohorts_db_path" => "/yaml/x.duckdb")["cohorts_db_path"] == "/yaml/x.duckdb"
    end
  end
  # prep_kwargs_from_cfg no longer mutates ENV
  withenv("PAN_DATA" => nothing, "PAN_OUT" => nothing, "FIADB" => nothing) do
    Pan.prep_kwargs_from_cfg(L("val_frac" => 0.0))
    @test !haskey(ENV, "PAN_DATA") && !haskey(ENV, "PAN_OUT")
  end
end

@testset "gdigel sub-dict" begin
  g = L("gdigel" => Dict("mu" => 4, "objective" => "mo", "m" => 3))["gdigel"]
  @test g["mu"] === 4 && g["objective"] == "mo" && g["m"] === 3 && g["weights"] === 4         # m=3 default weights
  @test L("gdigel" => Dict("objective" => "mo"))["gdigel"]["weights"] === 8
  @test L("gdigel" => Dict("weights" => 12))["gdigel"]["weights"] === 12
  @test L("gdigel" => Dict("kappa" => 2))["gdigel"]["kappa"] === 2.0 && L()["gdigel"]["kappa"] === nothing
  @test L("gdigel" => Dict("real_only" => "false", "mu" => "6"))["gdigel"]["real_only"] === false
  m = cerr(() -> L("gdigel" => Dict("sigma" => 0.1))); @test occursin("gdigel.sigma", m) && occursin("sigma0", m)
  @test occursin("gdigel.objective", cerr(() -> L("gdigel" => Dict("objective" => "root"))))
  @test occursin("gdigel.m", cerr(() -> L("gdigel" => Dict("m" => 4))))
  @test occursin("gdigel.mode", cerr(() -> L("gdigel" => Dict("mode" => "wander"))))
  @test occursin("expected a mapping", cerr(() -> L("gdigel" => [1, 2])))
  # resolve_gdigel is idempotent (parametrize_GDIgel calls it on an already-resolved dict)
  r = Pan.resolve_gdigel(Dict("mu" => 3)); @test Pan.resolve_gdigel(r) == r
  # every gdigel key the driver reads exists in the schema
  src = read(joinpath(pkgdir(Pan), "src", "gdigel_driver.jl"), String)
  used = Set(m.captures[1] for m in eachmatch(r"\bo\(\"(\w+)\"\)", src))
  @test !isempty(used) && issubset(used, keys(Pan.GDIGEL_SCHEMA))
end

@testset "dual_mode" begin
  @test L()["dual_mode"] === :off && L("dual_mode" => true)["dual_mode"] === :joint && L("dual_mode" => false)["dual_mode"] === :off
  @test L("dual_mode" => "B")["dual_mode"] === :b && L("dual_mode" => "joint")["dual_mode"] === :joint && L("dual_mode" => "FALSE")["dual_mode"] === :off
  @test occursin("dual_mode", cerr(() -> L("dual_mode" => "bogus")))
  @test Pan.load_config(L("dual_mode" => "b"))["dual_mode"] === :b                          # re-resolving a Symbol is stable
end

@testset "loss_before_injection resolution" begin
  sync = ("override_injection" => true, "override_injection_sync" => true)
  @test L(sync...)["loss_before_injection"] === true                                         # sync injection: new rule
  @test L(sync..., "loss_before_injection" => false)["loss_before_injection"] === false
  @test L("override_injection" => true)["loss_before_injection"] === false                  # append: legacy order
  @test L(sync..., "override_injection_replace" => true)["loss_before_injection"] === false # replace: legacy order
  @test L()["loss_before_injection"] === true                                                # no injection: the flag has no effect
  @test L("loss_before_injection" => "false")["loss_before_injection"] === false && L("loss_before_injection" => 0)["loss_before_injection"] === false
  m = cerr(() -> L("override_injection" => true, "loss_before_injection" => true)); @test m !== nothing && occursin("loss_before_injection", m)
  @test cerr(() -> L(sync..., "override_injection_replace" => true, "loss_before_injection" => true)) !== nothing
  # scoring_from_config builds the immutable options from the config (no module state, nothing to leak between configs)
    sc, sb = Pan.scoring_from_config(L(sync..., "disturbance_full_removals" => false, "exclude_recruits_at_entry" => false, "w_scope" => "survivors",
                                       "death_term" => "likelihood", "override_injection_noise" => 0.1, "override_injection_disturbance" => "scale",
                                       "init_perturb_frac" => 0.05, "init_perturb_cap" => 25, "dual_mode" => "b", "tier_b" => 4, "simB_spinup" => false))
    @test sc.override && sc.sync && !sc.replace && sc.loss_before_injection && Pan.shock_order(sc)
    @test !sc.exclude_recruits_at_entry && sc.w_scope === :survivors && sc.death_term === :likelihood
    @test sc.noise == 0.1 && sc.disturbance === :scale && sc.init_perturb_frac == 0.05 && sc.init_perturb_cap == 25f0
    @test sb.dual_mode === :b && sb.t4_primary && !sb.simb_spinup && sb.tier_b == 4
    @test Pan.param_space_options(L("death_term" => "likelihood")).death_pmort_sima         # death_term likelihood: score-only PROB_MORT in Sim A
    sc0, sb0 = Pan.scoring_from_config(L())                                                  # a plain config gets the defaults (no leak between configs)
    @test !sc0.override && sc0.exclude_recruits_at_entry && sc0.w_scope === :all && sc0.death_term === :none && !Pan.shock_order(sc0)
    @test sb0.dual_mode === :off && !sb0.t4_primary && sb0.simb_spinup && !sb0.simb_disturb_only
    @test !Pan.param_space_options(L()).death_pmort_sima
    @test Pan.scoring_from_config(Dict{String,Any}("w_scope" => "survivors"))[1].w_scope === :survivors    # a raw Dict is loaded through the same reader
    @test Pan.simb_options(L("tier" => 4)).t4_primary && !Pan.simb_options(L("tier" => 4, "dual_mode" => "joint")).t4_primary   # tier 4 without a dual, or dual b
    @test !ismutabletype(Pan.ScoringOptions) && !ismutabletype(Pan.SimBOptions) && all(isconcretetype, fieldtypes(Pan.ScoringOptions)) && all(isconcretetype, fieldtypes(Pan.SimBOptions))   # immutable, concretely typed: free to read in the site loop
    @test Pan.with_options(sc; w_scope=:all).w_scope === :all && Pan.with_options(sc; w_scope=:all).noise == 0.1
end

@testset "param_space_options (the former BSP.* switches)" begin
  o = Pan.param_space_options(L())
  @test o == Pan.PU.ParamSpaceOptions()                                                      # a plain config gives the legacy defaults
  @test !o.fix_growth && !o.fix_maturity && !o.fix_min_rel && !o.fix_pmort && !o.prob_estab_lower_bound
  @test o.bmax_bounds == (12000, 35000) && o.anpp_ratio_bounds == (20f0, 35f0) && o.pmort_upper_a == 0.2f0 && o.pmort_upper_b == 0.02f0
  o2 = Pan.param_space_options(L("fix_growth" => true, "fix_maturity" => true, "fix_min_rel" => true, "fix_pmort" => true, "pmort_upper" => 0.05, "pmort_upper_simA" => 0.1,
                                 "bmax_upper" => 40000, "anpp_ratio_bounds" => [18, 30], "prob_estab_lower_bound" => true, "prob_estab_lower_bound_mult" => 0.8, "death_term" => "likelihood"))
  @test o2.fix_growth && o2.fix_maturity && o2.fix_min_rel && o2.fix_pmort && o2.death_pmort_sima && o2.prob_estab_lower_bound
  @test o2.bmax_bounds == (12000, 40000) && o2.anpp_ratio_bounds == (18f0, 30f0) && o2.pmort_upper_b == 0.05f0 && o2.pmort_upper_a == 0.1f0 && o2.prob_estab_lower_mult == 0.8
  @test !ismutabletype(Pan.PU.ParamSpaceOptions) && all(isconcretetype, fieldtypes(Pan.PU.ParamSpaceOptions))
end

@testset "schema" begin
  @test length(Pan.SCHEMA) == length(Pan.SCHEMA_LIST)                                        # no duplicate keys
  @test length(Pan.GDIGEL_SCHEMA) == length(Pan.GDIGEL_LIST)
  @test all(s -> !isempty(s.doc), Pan.SCHEMA_LIST) && all(s -> !isempty(s.doc), Pan.GDIGEL_LIST)
  @test all(k -> !haskey(Pan.SCHEMA, k), keys(Pan.RETIRED_KEYS))
  # every default coerces back to itself (the schema is self-consistent)
  for s in Pan.SCHEMA_LIST
    s.type === :gdigel && continue
    d = s.default
    d === nothing && (@test s.nullable; continue)
    @test Pan._resolve_value("", s, deepcopy(d)) == d
  end
  @test occursin("`tier`", Pan.config_docs())
  # every top-level key read in src (cfg["..."]) is in the schema
  src = read(joinpath(pkgdir(Pan), "src", "Pan.jl"), String)
  used = Set(m.captures[1] for m in eachmatch(r"\bcfg\[\"(\w+)\"\]", src))
  @test !isempty(used) && issubset(used, keys(Pan.SCHEMA))
end

@testset "every shipped yaml loads" begin
  root = dirname(@__DIR__)
  files = String[]
  for (r, _, fs) in walkdir(joinpath(root, "configs")), f in fs
    endswith(f, ".yml") && push!(files, joinpath(r, f))
  end
  @test length(files) >= 15
  for f in files
    @test (Pan.load_config(f); true)
  end
  # the main checkout's legacy runs/*.yml (read-only): all load except the ones that still use retired keys
  runs = "/workspace/landis_parametrization_julia/runs"
  if isdir(runs)
    expected = Dict("sobol1.yml" => ["eval_tier", "sobol_m"], "sobol2.yml" => ["eval_tier", "sobol_m"],
                    "test1.yml" => ["search_tier"], "test2.yml" => ["search_tier"])
    failed = Dict{String,String}()
    n = 0
    for f in sort(readdir(runs; join=true))
      endswith(f, ".yml") || continue
      n += 1
      m = cerr(() -> Pan.load_config(f)); m === nothing || (failed[basename(f)] = m)
    end
    @test n > 100
    @test Set(keys(failed)) == Set(keys(expected))
    for (f, ks) in expected
      haskey(failed, f) || continue
      @test occursin("retired", failed[f]) && all(k -> occursin("`$k`", failed[f]), ks)
    end
  end
end
end # module
