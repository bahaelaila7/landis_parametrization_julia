using Pan

# Bake the FITTING/search hot paths into the sysimage so a real run has ~no JIT at startup. A 2-generation run
# on the actual config traces everything that dominates cold start: DuckDB data load, species tiering, floor
# loading, make_sites, fit_params, the SoA simulation, IgelMOCMAES (incl. the covariance eigendecomps) and
# checkpoint IO. Guarded so a missing DB at build time still yields a sysimage (bakes whatever compiled).
# Honors $FIADB for the DB path; point PAN_PRECOMPILE_CONFIG at any representative fitting yaml.
try
  cfg = get(ENV, "PAN_PRECOMPILE_CONFIG", "runs/fl853_igelmo_4cell_shade.yml")
  Pan.run_from_yaml(cfg; overrides=Dict("trials" => 2, "n_reps" => 1, "output_dir" => mktempdir()))
catch e
  @warn "precompile fitting run did not finish — sysimage still bakes whatever compiled before the error" exception = (e, catch_backtrace())
end

# Also bake the spatial-treemap path when its inputs are present (harmless skip otherwise).
try
  isdir("../pan_runner/data") && isfile("../pan_runner/data/FL5_22/FL5_22.tif") &&
    Pan.simulate_spatial_treemap(
      data_dir="../pan_runner/data", output_dir=mktempdir(),
      eco_raster="FL5_22/FL5_22_eco_l3.tif", eco_ecocode_mapping="eco_ecocode_l3_mapping.csv",
      biomass_params_path="FL5_22/FL5_22_eco_l3.jld2", treemap_raster="FL5_22/FL5_22.tif",
      treemap_db_path="../pan_runner/data_eco_cohorts.duckdb", treemap_version=2022,
      timehorizon_years=1, output_every_years=1)
catch e
  @warn "precompile spatial run skipped/failed" exception = e
end
