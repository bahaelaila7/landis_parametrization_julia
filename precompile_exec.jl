using Pan

Pan.simulate_spatial_treemap(
  data_dir="../pan_runner/data",
  output_dir=mktempdir(),
  eco_raster="FL5_22/FL5_22_eco_l3.tif",
  eco_ecocode_mapping="eco_ecocode_l3_mapping.csv",
  biomass_params_path="FL5_22/FL5_22_eco_l3.jld2",
  treemap_raster="FL5_22/FL5_22.tif",
  treemap_db_path="../pan_runner/data_eco_cohorts.duckdb",
  treemap_version=2022,
  timehorizon_years=1,
  output_every_years=1,
)
