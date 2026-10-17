# Precompile-execution file for `make sysimage`, `make sysimage-deps`, `make sysimage-fit` and `make exec` (PackageCompiler).
#
# It traces the FITTING hot path once so a real run starts without JIT: config parsing, the DuckDB data load, split + reference
# tables, make_sites, fit_params (SoA simulation + tier accumulators), the Igel MO-CMA-ES driver (eigendecompositions), checkpoint IO.
# It needs the region data: point PAN_DATA at a directory that holds slim_r4.duckdb and the CSV tables configs/region4/*.yml use
# (e.g. PAN_DATA=/workspace/landis_region4/data_r4). Without data the build still succeeds, but only the config parser is traced.
#
# Environment (all optional):
#   PAN_PRECOMPILE_CONFIG  config to trace (default configs/region4/gafl_A.yml)
#   PAN_DATA               data directory (see above); PAN_OUT is set to a temp dir here
# The run is the golden-harness size: the ~80 fixed GA+FL plots of test/golden/plots.csv, 2 evaluations, one repetition.
using Pan
import YAML

const ROOT = dirname(@__DIR__)
cfg = get(ENV, "PAN_PRECOMPILE_CONFIG", joinpath(ROOT, "configs", "region4", "gafl_A.yml"))
data = get(ENV, "PAN_DATA", "")
out = mktempdir()
ENV["PAN_OUT"] = out

# config parser over every shipped config (cheap, needs no data)
for f in sort(vcat([joinpath(ROOT, "configs", d, x) for d in ("region4", "pnet") for x in readdir(joinpath(ROOT, "configs", d))]...))
  endswith(f, ".yml") || continue
  try Pan.load_config(YAML.load_file(f)) catch e; @warn "precompile: load_config failed" f exception = e end
end

if !isfile(joinpath(data, "slim_r4.duckdb"))
  @warn "precompile: PAN_DATA does not hold slim_r4.duckdb -> the FITTING path is NOT baked. Set PAN_DATA before `make sysimage`." PAN_DATA = data
else
  plots = [parse.(Int, split(r, ',')[1:4]) for r in readlines(joinpath(ROOT, "test", "golden", "plots.csv"))[2:end]]
  try
    Pan.run_from_yaml(cfg; overrides = Dict{String,Any}("trials" => 2, "n_reps" => 1, "igel_mu" => 4, "igel_sobol_init" => false,
                                                         "filter_plots" => plots, "min_trees" => 20, "n_folds" => 2, "fold_index" => 1,
                                                         "test_frac" => 0.0, "train_cap" => nothing, "output_dir" => out))
  catch e
    @warn "precompile fitting run errored - the image bakes whatever compiled before the error" exception = (e, catch_backtrace())
  end
end
println("precompile_exec: done")
