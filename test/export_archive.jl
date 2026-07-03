# Export an MO-CMA-ES archive: one params.jld2 per candidate under candidates/candidate_<i>/, plus
# archive_candidates.csv (candidate, A_W, A_AGB, aggregate). Uses the latest search_state@N.jld2 (or
# a path via ARGS[2]). The per-candidate driver then runs scatter/tost/smape against each params.jld2.
#   Run: ./julia_gdal.sh --project=. test/export_archive.jl <config.yml> [search_state@N.jld2]
using Pan
import JLD2, YAML
cfg = YAML.load_file(ARGS[1]); outdir = cfg["output_dir"]
ckpt = length(ARGS) >= 2 ? ARGS[2] :
  (let fs = filter(f -> occursin(r"search_state@\d+\.jld2$", f), readdir(outdir; join=true));
     isempty(fs) && error("no search_state@N.jld2 in $outdir");
     sort(fs; by=f -> parse(Int, match(r"@(\d+)\.", f).captures[1]))[end] end)
st = JLD2.load_object(ckpt)
archive = collect(st.archive)
println("checkpoint $(basename(ckpt)) (gen $(st.i)): archive = $(length(archive)) candidates")
open(joinpath(outdir, "archive_candidates.csv"), "w") do io
  println(io, "candidate,A_W,A_AGB,aggregate")
  for (i, m) in enumerate(archive)
    o = Float64.(m.fx.objectives)
    aw = length(o) >= 1 ? o[1] : NaN; aagb = length(o) >= 2 ? o[2] : NaN
    println(io, "$i,$aw,$aagb,$(Float64(m.fx.aggregate))")
    d = joinpath(outdir, "candidates", "candidate_$i"); mkpath(d)
    JLD2.save_object(joinpath(d, "params.jld2"), m.x)
  end
end
println("wrote archive_candidates.csv + candidates/candidate_1..$(length(archive))/params.jld2 → $outdir")
