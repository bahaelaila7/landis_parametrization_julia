# Run search_mode gdigel on a config: ./julia_gdal.sh --project=. --threads=N tools/gd_run.jl <config.yml> <out_dir> <trials> [key=value gdigel options...]
using Pan
opts = Dict{String,Any}("mu" => 8)
for a in ARGS[4:end]; k, v = split(a, '='; limit=2); opts[k] = something(tryparse(Int, v), tryparse(Float64, v), v == "true" ? true : v == "false" ? false : v); end
Pan.run_from_yaml(ARGS[1]; overrides=Dict{String,Any}("output_dir" => ARGS[2], "trials" => parse(Int, ARGS[3]), "search_mode" => "gdigel", "gdigel" => opts, "n_reps" => 1))
