# Run search_mode gdigel on a config: ./julia_gdal.sh --project=. --threads=N tools/gd_run.jl <config.yml> <out_dir> <trials> [key=value gdigel options...]
using Pan
opts = Dict{String,Any}("mu" => 8); top = Dict{String,Any}()
for a in ARGS[4:end]
  k, v = split(a, '='; limit=2)
  val = something(tryparse(Int, v), tryparse(Float64, v), v == "true" ? true : v == "false" ? false : String(v))
  k in ("start_from", "resume_from", "seed", "loss_before_injection", "w_scope", "death_term", "exclude_recruits_at_entry", "w_visits") ? (top[k] = val) : (opts[k] = val)   # top-level config keys
end
Pan.run_from_yaml(ARGS[1]; overrides=merge(Dict{String,Any}("output_dir" => ARGS[2], "trials" => parse(Int, ARGS[3]), "search_mode" => "gdigel", "gdigel" => opts, "n_reps" => 1), top))
