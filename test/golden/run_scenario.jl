# ./julia_gdal.sh --project=<repo> --threads=1 <golden>/run_scenario.jl <scenario> <out_root>
# cwd must be the repo root. Writes <out_root>/<scenario>/{run/,status.txt,probe_*}.
import YAML
using Pan
include(joinpath(@__DIR__, "scenarios.jl"))
name, outroot = ARGS[1], ARGS[2]
base, ov, trials = SCENARIOS[name]
sdir = joinpath(outroot, name); rundir = joinpath(sdir, "run"); rm(sdir; recursive=true, force=true); mkpath(rundir)
cfg = YAML.load_file(base)
over = merge(common(), ov, Dict{String,Any}("output_dir" => rundir, "trials" => trials))
if name == "bs_plot_only"
  src = joinpath(outroot, "bs_igelmo", "run"); isdir(joinpath(src, "fold_1")) && (src = joinpath(src, "fold_1"))
  ck = filter(f -> occursin(r"^best_params@\d+\.jld2$", f), readdir(src))
  over["params_path"] = joinpath(src, last(sort(ck; by=f -> parse(Int, match(r"@(\d+)", f)[1]))))
end
include(joinpath(@__DIR__, isdefined(Pan, :POST_SEARCH_HOOK) ? "probe_v0.jl" : "probe_v1.jl"))
install_probe!(merge(cfg, over), sdir)
t0 = time()
status = try
  Pan.run_from_yaml(base; overrides=over); "OK"
catch e
  msg = sprint(showerror, e, catch_backtrace())
  open(io -> println(io, msg), joinpath(sdir, "error_full.txt"), "w")
  m = match(r"(?:nested task error: )?([A-Za-z]+(?:Error|Exception)[^\n]*)", replace(msg, "TaskFailedException" => ""))
  "ERROR " * replace(m === nothing ? first(split(msg, '\n')) : m.captures[1], ',' => ';')[1:min(end, 300)]
end
open(io -> println(io, status), joinpath(sdir, "status.txt"), "w")
open(io -> println(io, round(time() - t0; digits=1)), joinpath(sdir, "walltime.txt"), "w")
println("GOLDEN ", name, " ", status)
