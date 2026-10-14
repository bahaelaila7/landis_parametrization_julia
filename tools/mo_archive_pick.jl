# Pick the knee and the two extremes of a MO-gdigel non-dominated archive (<gd_out>/mo_archive.csv, docs/mo_gdigel.md §5).
#   julia tools/mo_archive_pick.jl <gd_out_dir>      → prints one line per pick:  <role> <params.jld2> <lambda> <A_ref> <W_ref>
# lambda: the generating lineage's weight for the extremes (min_A / min_W); the Chebyshev-balance weight at the point for the knee
# (the λ at which both scalarized terms are equal there). A_ref / W_ref are the run's fixed reference totals (for CALIB_AREF/WREF).
using CSV, DataFrames, JLD2
d = ARGS[1]; df = CSV.read(joinpath(d, "mo_archive.csv"), DataFrame)
u = JLD2.load_object(joinpath(d, "mo_archive_u.jld2"))
isnan0(x) = x isa Number && isnan(x)
function row_for(role)
  i = findfirst(r -> occursin(role, coalesce(r, "")), df.role); i === nothing ? nothing : df[i, :]
end
for role in ("min_A", "knee", "min_W")
  r = row_for(role); r === nothing && continue
  lam = role == "knee" || isnan0(r.lambda_lineage) ? r.lambda_balance : r.lambda_lineage
  println(join([role, joinpath(d, "mo_archive", r.file), round(lam, digits=4), u.aref, u.wref], " "))
end
