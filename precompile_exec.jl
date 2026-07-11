using Pan
import YAML

# Bake the FITTING/search hot paths into the sysimage so a real run has ~no JIT at startup: a 2-generation run
# traces DuckDB data load, tiering, floor loading, make_sites, fit_params, the SoA sim, IgelMOCMAES (incl. the
# covariance eigendecomps) and checkpoint IO. This REQUIRES a valid DB at build time — resolve it the same way
# run_from_yaml does ($FIADB wins, else the config's cohorts_db_path). If it's not found, the build still
# succeeds but the fitting path is NOT baked — so set FIADB to your DuckDB before `make sysimage`.
cfg = get(ENV, "PAN_PRECOMPILE_CONFIG", "runs/fl853_igelmo_4cell_shade.yml")
db  = get(ENV, "FIADB", "")
isempty(db) && (db = try String(YAML.load_file(cfg)["cohorts_db_path"]) catch; "" end)

if !isfile(db)
  @warn "precompile: DB not found → the FITTING path will NOT be baked. Set FIADB to your DuckDB before `make sysimage`." db config = cfg
else
  try
    Pan.run_from_yaml(cfg; overrides = Dict("trials" => 2, "n_reps" => 1, "output_dir" => mktempdir()))
  catch e
    @warn "precompile fitting run errored — sysimage bakes whatever compiled before the error" exception = (e, catch_backtrace())
  end
end
