# Sim B igelmo run of the Sim A/B campaign: ./julia_gdal.sh --project=. tools/simab_runB.jl <config.yml> <out_dir> <trials> key=value ...  (top-level yaml overrides)
using Pan
ov = Dict{String,Any}("output_dir" => ARGS[2], "trials" => parse(Int, ARGS[3]))
for a in ARGS[4:end]; k, v = split(a, '='; limit=2); ov[k] = something(tryparse(Int, v), tryparse(Float64, v), v == "true" ? true : v == "false" ? false : String(v)); end
Pan.run_from_yaml(ARGS[1]; overrides=ov)
